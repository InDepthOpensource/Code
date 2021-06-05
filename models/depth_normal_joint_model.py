import torch
import torch.nn as nn

from models import BasicBlock, LastUnetDecoderBlock, Conv2dNormRelu, conv1x1, resnet34, model_zoo, model_urls
from models.resnet_cbam_dilated import CBAMDilatedUNet, CBAMBasicBlock, CBAMResNet, CBAMDecoderBlock


class MultipleResNetBlock(nn.Module):
    # Pass the results after fusion to 2 resnet blocks.

    def __init__(self, input_channels, output_channels, num_blocks=2):
        super().__init__()
        decoder_layers = []
        block = BasicBlock

        downsample = None
        if input_channels != output_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(input_channels, output_channels * block.expansion, stride=1),
                nn.BatchNorm2d(output_channels * block.expansion),
            )
        decoder_layers.append(block(input_channels, output_channels, downsample=downsample))

        for _ in range(0, num_blocks - 1):
            decoder_layers.append(block(output_channels, output_channels))

        self.decoder_layers = nn.Sequential(*decoder_layers)

        # Init layers according to their activation functions.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.decoder_layers(x)


# TODO: try to use different number of channels in the last few decoder blocks.
#  Also try to separate the depth, normal branches in different stages.
class DilatedUNetDepthNormal(CBAMDilatedUNet):
    def __init__(self, device, depth_model_weights_path=None, full_model_weights_path=None):
        super().__init__(device, 2, use_cbam_encoder=False, use_cbam_fuse=False, use_cbam_decoder=False)
        self.to(device)
        if depth_model_weights_path:
            self.load_state_dict(torch.load(depth_model_weights_path, map_location=device))

        self.depth_final_resblocks = MultipleResNetBlock(32, 32).to(device)
        self.normal_x_final_resblocks = MultipleResNetBlock(32, 32).to(device)
        self.normal_y_final_resblocks = MultipleResNetBlock(32, 32).to(device)
        self.normal_z_final_resblocks = MultipleResNetBlock(32, 32).to(device)

        self.unet_depth_output_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu).to(
            device)  # Keep the output at 1.
        self.unet_normal_x_output_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu).to(device)
        self.unet_normal_y_output_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu).to(device)
        self.unet_normal_z_output_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu).to(device)

        if full_model_weights_path:
            self.load_state_dict(torch.load(full_model_weights_path, map_location=device))

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

        unet_depth_final_resblock_out = self.depth_final_resblocks(unet_out_2)
        unet_normal_x_final_resblock_out = self.normal_x_final_resblocks(unet_out_2)
        unet_normal_y_final_resblock_out = self.normal_y_final_resblocks(unet_out_2)
        unet_normal_z_final_resblock_out = self.normal_z_final_resblocks(unet_out_2)

        unet_depth_out = self.unet_depth_output_block(unet_depth_final_resblock_out)

        unet_normal_x_out = self.unet_normal_x_output_block(unet_normal_x_final_resblock_out)
        unet_normal_y_out = self.unet_normal_y_output_block(unet_normal_y_final_resblock_out)
        unet_normal_z_out = self.unet_normal_z_output_block(unet_normal_z_final_resblock_out)

        unet_normal_out = torch.cat((unet_normal_x_out, unet_normal_y_out, unet_normal_z_out), 1)

        return unet_depth_out, unet_normal_out

    def set_train_normal_decoder(self, is_training):
        normal_decoder = [self.normal_x_final_resblocks,
                          self.normal_y_final_resblocks,
                          self.normal_z_final_resblocks,
                          self.unet_normal_x_output_block,
                          self.unet_normal_y_output_block,
                          self.unet_normal_z_output_block]
        for module in normal_decoder:
            for param in module.parameters():
                param.requires_grad = is_training


class DilatedUNetDepthNormalAlt0(nn.Module):
    def __init__(self, device, num_blocks=2, use_resblock=True, use_cbam_fuse=False, use_cbam_encoder=False,
                 use_cbam_decoder=False):
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
        self.unet_block_2 = CBAMDecoderBlock((64, 64, 64), (1, 3, 6), 64, 64,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)

        self.depth_final_resblocks_0 = MultipleResNetBlock(64, 64)
        self.normal_x_final_resblocks_0 = MultipleResNetBlock(64, 64)
        self.normal_y_final_resblocks_0 = MultipleResNetBlock(64, 64)
        self.normal_z_final_resblocks_0 = MultipleResNetBlock(64, 64)

        self.depth_final_resblocks_1 = MultipleResNetBlock(64, 32)
        self.normal_x_final_resblocks_1 = MultipleResNetBlock(64, 32)
        self.normal_y_final_resblocks_1 = MultipleResNetBlock(64, 32)
        self.normal_z_final_resblocks_1 = MultipleResNetBlock(64, 32)

        self.unet_depth_output_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu)  # Keep the output at 1.
        self.unet_normal_x_output_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu)
        self.unet_normal_y_output_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu)
        self.unet_normal_z_output_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu)

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

        unet_depth_final_resblock_out_0 = self.depth_final_resblocks_0(unet_out_2)
        unet_normal_x_final_resblock_out_0 = self.normal_x_final_resblocks_0(unet_out_2)
        unet_normal_y_final_resblock_out_0 = self.normal_y_final_resblocks_0(unet_out_2)
        unet_normal_z_final_resblock_out_0 = self.normal_z_final_resblocks_0(unet_out_2)

        unet_depth_final_resblock_out_1 = self.depth_final_resblocks_1(unet_depth_final_resblock_out_0)
        unet_normal_x_final_resblock_out_1 = self.normal_x_final_resblocks_1(unet_normal_x_final_resblock_out_0)
        unet_normal_y_final_resblock_out_1 = self.normal_y_final_resblocks_1(unet_normal_y_final_resblock_out_0)
        unet_normal_z_final_resblock_out_1 = self.normal_z_final_resblocks_1(unet_normal_z_final_resblock_out_0)

        unet_depth_out = self.unet_depth_output_block(unet_depth_final_resblock_out_1)

        unet_normal_x_out = self.unet_normal_x_output_block(unet_normal_x_final_resblock_out_1)
        unet_normal_y_out = self.unet_normal_y_output_block(unet_normal_y_final_resblock_out_1)
        unet_normal_z_out = self.unet_normal_z_output_block(unet_normal_z_final_resblock_out_1)

        unet_normal_out = torch.cat((unet_normal_x_out, unet_normal_y_out, unet_normal_z_out), 1)

        return unet_depth_out, unet_normal_out

    def set_train_normal_decoder(self, is_training):
        normal_decoder = [self.normal_x_final_resblocks_0,
                          self.normal_y_final_resblocks_0,
                          self.normal_z_final_resblocks_0,
                          self.normal_x_final_resblocks_1,
                          self.normal_y_final_resblocks_1,
                          self.normal_z_final_resblocks_1,
                          self.unet_normal_x_output_block,
                          self.unet_normal_y_output_block,
                          self.unet_normal_z_output_block]
        for module in normal_decoder:
            for param in module.parameters():
                param.requires_grad = is_training


class DilatedUNetDepthNormalAlt1(nn.Module):
    def __init__(self, device, num_blocks=2, use_resblock=True, use_cbam_fuse=False, use_cbam_encoder=False,
                 use_cbam_decoder=False):
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
        self.unet_block_2 = CBAMDecoderBlock((64, 64, 64), (1, 3, 6), 64, 64,
                                             num_of_blocks=num_blocks, use_resblock=use_resblock,
                                             use_cbam_fuse=use_cbam_fuse, use_cbam_decoder=use_cbam_decoder)

        self.depth_final_resblocks = MultipleResNetBlock(64, 32)
        self.normal_x_final_resblocks = MultipleResNetBlock(64, 32)
        self.normal_y_final_resblocks = MultipleResNetBlock(64, 32)
        self.normal_z_final_resblocks = MultipleResNetBlock(64, 32)

        self.unet_depth_output_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu)  # Keep the output at 1.
        self.unet_normal_x_output_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu)
        self.unet_normal_y_output_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu)
        self.unet_normal_z_output_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu)

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

        unet_depth_final_resblock_out = self.depth_final_resblocks(unet_out_2)
        unet_normal_x_final_resblock_out = self.normal_x_final_resblocks(unet_out_2)
        unet_normal_y_final_resblock_out = self.normal_y_final_resblocks(unet_out_2)
        unet_normal_z_final_resblock_out = self.normal_z_final_resblocks(unet_out_2)

        unet_depth_out = self.unet_depth_output_block(unet_depth_final_resblock_out)

        unet_normal_x_out = self.unet_normal_x_output_block(unet_normal_x_final_resblock_out)
        unet_normal_y_out = self.unet_normal_y_output_block(unet_normal_y_final_resblock_out)
        unet_normal_z_out = self.unet_normal_z_output_block(unet_normal_z_final_resblock_out)

        unet_normal_out = torch.cat((unet_normal_x_out, unet_normal_y_out, unet_normal_z_out), 1)

        return unet_depth_out, unet_normal_out

    def set_train_normal_decoder(self, is_training):
        normal_decoder = [self.normal_x_final_resblocks,
                          self.normal_y_final_resblocks,
                          self.normal_z_final_resblocks,
                          self.unet_normal_x_output_block,
                          self.unet_normal_y_output_block,
                          self.unet_normal_z_output_block]
        for module in normal_decoder:
            for param in module.parameters():
                param.requires_grad = is_training


# Test this model
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DilatedUNetDepthNormalAlt1(device)
    model.to(device)

    model.train()
    dummy_rgb = torch.rand(8, 3, 320, 256)
    dummy_depth = torch.rand(8, 1, 320, 256)
    dummy_pred, dummy_normal = model(dummy_rgb, dummy_depth)
    loss = dummy_pred[-1].mean()
    loss.backward()
    print(dummy_pred.shape)
    print(dummy_normal.shape)
