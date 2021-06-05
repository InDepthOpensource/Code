import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from models import CBAM, FuseCBAM, BilinearUpSample, BasicBlock, Conv2dNormRelu, conv1x1, ResNet, resnet34, model_zoo, \
    model_urls, LastUnetDecoderBlock


class FuseBlock(nn.Module):
    # input_channels Order: decoder_input_channels, rgb_input_channels, depth_input_channels
    # output channels is an int.
    def __init__(self, input_channels, dilation, internal_channels, use_cbam=True):
        super().__init__()
        # Naming convention: unet_block_x means the input is 1/n times the original resolution.
        self.use_cbam = use_cbam

        self.conv_layers = []
        for i in range(len(dilation)):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(sum(input_channels), internal_channels, 3, padding=dilation[i], dilation=dilation[i],
                          bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))

        if use_cbam:
            self.fuse_cbam = FuseCBAM(input_channels[0], input_channels[1], input_channels[2])

        self.conv_layers = nn.ModuleList(self.conv_layers if internal_channels > 64 else self.conv_layers[:1])

    def forward(self, decoder_x, rgb_x, depth_x):
        if self.use_cbam:
            x = self.fuse_cbam(decoder_x, rgb_x, depth_x)
        else:
            features_to_cat = [i for i in (decoder_x, rgb_x, depth_x) if i.dtype != torch.bool]
            x = torch.cat(features_to_cat, dim=1)
        x = [i(x) for i in self.conv_layers]
        x = torch.sum(torch.stack(x), dim=0)
        return x


class CBAMBasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=True):
        super().__init__(inplanes, planes, stride, downsample)
        if use_cbam:
            self.cbam = CBAM(planes)
        else:
            self.cbam = None
        # We are actually using leaky relu.
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.cbam:
            out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out


class CBAMDecoderBlock(nn.Module):
    def __init__(self, input_channels, dilation, dilation_internal_channels, output_channels,
                 use_cbam_fuse=True, use_cbam_decoder=True, use_resblock=True, num_of_blocks=2, up_sample='bilinear'):
        super().__init__()

        self.fuse_block = FuseBlock(input_channels, dilation, dilation_internal_channels, use_cbam_fuse)

        layers = []
        if use_resblock:
            block = CBAMBasicBlock if use_cbam_decoder else BasicBlock

            downsample = None
            if dilation_internal_channels != output_channels * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(dilation_internal_channels, output_channels * block.expansion, stride=1),
                    nn.BatchNorm2d(output_channels * block.expansion),
                )
            layers.append(block(dilation_internal_channels, output_channels, downsample=downsample))

            for _ in range(1, num_of_blocks):
                layers.append(block(output_channels, output_channels))

        else:
            block = Conv2dNormRelu
            layers.append(block(dilation_internal_channels, output_channels))
            for _ in range(1, num_of_blocks):
                layers.append(block(output_channels, output_channels))

        if up_sample == 'bilinear':
            up_sample = BilinearUpSample()
        else:
            up_sample = None

        layers.append(up_sample)

        self.layers = nn.Sequential(*layers)

        # TODO: did not use the same init for CBAM BN.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, decoder_x, rgb_x, depth_x):
        x = self.fuse_block(decoder_x, rgb_x, depth_x)
        x = self.layers(x)
        return x


class CBAMResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000, in_channel=2):
        super().__init__(block, layers, num_classes, in_channel)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CBAMDilatedUNet(nn.Module):
    def __init__(self, device, num_blocks=2, use_resblock=True, use_cbam_fuse=True, use_cbam_encoder=True,
                 use_cbam_decoder=True):
        super().__init__()
        self.device = device
        self.rgb_encoder = resnet34()

        depth_encoder_basic_block = CBAMBasicBlock if use_cbam_encoder else BasicBlock
        self.depth_encoder = CBAMResNet(depth_encoder_basic_block, [3, 4, 6, 3], in_channel=2)

        # Naming convention: unet_block_x means the input is 1/n times the original resolution.

        # Output is 1/16
        self.unet_block_32 = CBAMDecoderBlock((0, 512, 512), (1, 3), 256, 256,
                                              num_of_blocks=num_blocks, use_resblock=use_resblock,
                                              use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1/8
        self.unet_block_16 = CBAMDecoderBlock((256, 256, 256), (1, 3, 6), 128, 128,
                                              num_of_blocks=num_blocks, use_resblock=use_resblock,
                                              use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1/4
        self.unet_block_8 = CBAMDecoderBlock((128, 128, 128), (1, 3, 6), 64, 64,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1/2
        self.unet_block_4 = CBAMDecoderBlock((64, 64, 64), (1, 3, 6), 64, 64,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1
        self.unet_block_2 = CBAMDecoderBlock((64, 64, 64), (1, 3, 6), 32, 32,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        self.unet_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu)  # Keep the output at 1.

        self.rgb_encoder.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

    def forward(self, color_input, depth_input):
        depth_mask = (depth_input < 10 ** -6) * 0.9 + 0.01
        depth_input = torch.cat((depth_mask, depth_input), 1)

        color_out_2, color_out_4, color_out_8, color_out_16, color_out_32 = self.rgb_encoder(color_input)
        depth_out_2, depth_out_4, depth_out_8, depth_out_16, depth_out_32 = self.depth_encoder(depth_input)

        # Naming convention: unet_out_n is output for unet_block_n, so the size is 1 / (2 * n)
        unet_out_32 = self.unet_block_32(color_out_32.new_tensor(False, dtype=torch.bool), color_out_32, depth_out_32)
        unet_out_16 = self.unet_block_16(unet_out_32, color_out_16, depth_out_16)
        unet_out_8 = self.unet_block_8(unet_out_16, color_out_8, depth_out_8)
        unet_out_4 = self.unet_block_4(unet_out_8, color_out_4, depth_out_4)
        unet_out_2 = self.unet_block_2(unet_out_4, color_out_2, depth_out_2)
        unet_out = self.unet_block(unet_out_2)

        return unet_out


class CBAMUNet(CBAMDilatedUNet):
    def __init__(self, device, num_blocks=2, use_resblock=True, use_cbam_fuse=False, use_cbam_encoder=False,
                 use_cbam_decoder=False):
        super().__init__(device, num_blocks, use_resblock, use_cbam_fuse, use_cbam_encoder, use_cbam_decoder)
        # Output is 1/16
        self.unet_block_32 = CBAMDecoderBlock((0, 512, 512), (1,), 256, 256,
                                              num_of_blocks=num_blocks, use_resblock=use_resblock,
                                              use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1/8
        self.unet_block_16 = CBAMDecoderBlock((256, 256, 256), (1,), 128, 128,
                                              num_of_blocks=num_blocks, use_resblock=use_resblock,
                                              use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1/4
        self.unet_block_8 = CBAMDecoderBlock((128, 128, 128), (1,), 64, 64,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1/2
        self.unet_block_4 = CBAMDecoderBlock((64, 64, 64), (1,), 64, 64,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1
        self.unet_block_2 = CBAMDecoderBlock((64, 64, 64), (1,), 32, 32,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)


class DilatedUNetOneBranch(nn.Module):
    def __init__(self, device, num_blocks=2, use_resblock=True, use_cbam_fuse=False, use_cbam_decoder=False):
        super().__init__()
        self.device = device

        encoder_basic_block = BasicBlock
        self.encoder = CBAMResNet(encoder_basic_block, [3, 4, 6, 3], in_channel=5)

        # Naming convention: unet_block_x means the input is 1/n times the original resolution.

        # Output is 1/16
        self.unet_block_32 = CBAMDecoderBlock((0, 0, 512), (1, 3), 256, 256,
                                              num_of_blocks=num_blocks, use_resblock=use_resblock,
                                              use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1/8
        self.unet_block_16 = CBAMDecoderBlock((256, 0, 256), (1, 3, 6), 128, 128,
                                              num_of_blocks=num_blocks, use_resblock=use_resblock,
                                              use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1/4
        self.unet_block_8 = CBAMDecoderBlock((128, 0, 128), (1, 3, 6), 64, 64,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1/2
        self.unet_block_4 = CBAMDecoderBlock((64, 0, 64), (1, 3, 6), 64, 64,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1
        self.unet_block_2 = CBAMDecoderBlock((64, 0, 64), (1, 3, 6), 32, 32,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        self.unet_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu)  # Keep the output at 1.

    def forward(self, color_input, depth_input):
        depth_mask = (depth_input < 10 ** -6) * 0.9 + 0.01
        depth_input = torch.cat((depth_mask, depth_input), 1)
        inputs = torch.cat((color_input, depth_input), 1)

        out_2, out_4, out_8, out_16, out_32 = self.encoder(inputs)

        # Naming convention: unet_out_n is output for unet_block_n, so the size is 1 / (2 * n)
        unet_out_32 = self.unet_block_32(out_32.new_tensor(False, dtype=torch.bool),
                                         out_32.new_tensor(False, dtype=torch.bool),
                                         out_32)
        unet_out_16 = self.unet_block_16(unet_out_32, out_16.new_tensor(False, dtype=torch.bool), out_16)
        unet_out_8 = self.unet_block_8(unet_out_16, out_8.new_tensor(False, dtype=torch.bool), out_8)
        unet_out_4 = self.unet_block_4(unet_out_8, out_4.new_tensor(False, dtype=torch.bool), out_4)
        unet_out_2 = self.unet_block_2(unet_out_4, out_2.new_tensor(False, dtype=torch.bool), out_2)
        unet_out = self.unet_block(unet_out_2)

        return unet_out


class UNetOneBranch(DilatedUNetOneBranch):
    def __init__(self, device, num_blocks=2, use_resblock=True, use_cbam_fuse=False, use_cbam_decoder=False):
        super().__init__(device, num_blocks, use_resblock, use_cbam_fuse, use_cbam_decoder)
        # Output is 1/16
        self.unet_block_32 = CBAMDecoderBlock((0, 0, 512), (1,), 256, 256,
                                              num_of_blocks=num_blocks, use_resblock=use_resblock,
                                              use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1/8
        self.unet_block_16 = CBAMDecoderBlock((256, 0, 256), (1,), 128, 128,
                                              num_of_blocks=num_blocks, use_resblock=use_resblock,
                                              use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1/4
        self.unet_block_8 = CBAMDecoderBlock((128, 0, 128), (1,), 64, 64,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1/2
        self.unet_block_4 = CBAMDecoderBlock((64, 0, 64), (1,), 64, 64,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)
        # Output is 1
        self.unet_block_2 = CBAMDecoderBlock((64, 0, 64), (1,), 32, 32,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)


# Test and Debug
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # test_cbam = CBAMDilatedUNet(device, 2)
    # test_cbam.train()
    # dummy_rgb = torch.rand(8, 3, 320, 256)
    # dummy_depth = torch.rand(8, 1, 320, 256)
    # dummy_pred = test_cbam(dummy_rgb, dummy_depth)
    #
    # test_cbam = CBAMDilatedUNet(device, 2, use_resblock=False)
    # test_cbam.train()
    # dummy_rgb = torch.rand(8, 3, 320, 256)
    # dummy_depth = torch.rand(8, 1, 320, 256)
    # dummy_pred = test_cbam(dummy_rgb, dummy_depth)

    # test_cbam = CBAMDilatedUNet(device, 2, use_cbam_fuse=False)
    # test_cbam.train()
    # dummy_rgb = torch.rand(8, 3, 320, 256)
    # dummy_depth = torch.rand(8, 1, 320, 256)
    # dummy_pred = test_cbam(dummy_rgb, dummy_depth)
    #
    # test_cbam = CBAMDilatedUNet(device, 2, use_cbam_fuse=False, use_resblock=False)
    # test_cbam.train()
    # dummy_rgb = torch.rand(8, 3, 320, 256)
    # dummy_depth = torch.rand(8, 1, 320, 256)
    # dummy_pred = test_cbam(dummy_rgb, dummy_depth)
    #
    # test_cbam = CBAMDilatedUNet(device, 2, use_cbam_fuse=False, use_cbam_decoder=False)
    # test_cbam.train()
    # dummy_rgb = torch.rand(8, 3, 320, 256)
    # dummy_depth = torch.rand(8, 1, 320, 256)
    # dummy_pred = test_cbam(dummy_rgb, dummy_depth)

    test_cbam = CBAMUNet(device, 2, use_cbam_fuse=False, use_cbam_decoder=False)
    test_cbam.train()
    dummy_rgb = torch.rand(8, 3, 320, 256)
    dummy_depth = torch.rand(8, 1, 320, 256)
    dummy_pred = test_cbam(dummy_rgb, dummy_depth)

    test_cbam = DilatedUNetOneBranch(device, 2, use_cbam_fuse=False, use_cbam_decoder=False)
    test_cbam.train()
    dummy_rgb = torch.rand(8, 3, 320, 256)
    dummy_depth = torch.rand(8, 1, 320, 256)
    dummy_pred = test_cbam(dummy_rgb, dummy_depth)

    test_cbam = UNetOneBranch(device, 2, use_cbam_fuse=False, use_cbam_decoder=False)
    test_cbam.train()
    dummy_rgb = torch.rand(8, 3, 320, 256)
    dummy_depth = torch.rand(8, 1, 320, 256)
    dummy_pred = test_cbam(dummy_rgb, dummy_depth)
