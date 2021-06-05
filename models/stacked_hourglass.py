import torch
from torch import nn
from models import LastUnetDecoderBlock, resnet34, Conv2dNormRelu, model_zoo, model_urls, BasicBlock, conv1x1, \
    BilinearUpSample, CBAMResNet, BasicDecoderBlock


class RGBOnlyDecoderBlock(nn.Module):
    def __init__(self, input_channels, dilation, internal_channels, output_channels, num_blocks=2,
                 up_sample='bilinear'):
        super().__init__()
        # For fusing with dilation.
        self.diluted_conv_layers = []
        for i in range(len(dilation)):
            self.diluted_conv_layers.append(nn.Sequential(
                nn.Conv2d(sum(input_channels), internal_channels, 3, padding=dilation[i], dilation=dilation[i],
                          bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        self.diluted_conv_layers = nn.ModuleList(self.diluted_conv_layers)

        # Pass the results after fusion to 2 resnet blocks, and then up sample.
        decoder_layers = []
        block = BasicBlock

        downsample = None
        if internal_channels != output_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(internal_channels, output_channels * block.expansion, stride=1),
                nn.BatchNorm2d(output_channels * block.expansion),
            )
        decoder_layers.append(block(internal_channels, output_channels, downsample=downsample))

        for _ in range(0, num_blocks - 1):
            decoder_layers.append(block(output_channels, output_channels))

        if up_sample == 'bilinear':
            up_sample = BilinearUpSample()
        else:
            up_sample = None
        decoder_layers.append(up_sample)

        self.decoder_layers = nn.Sequential(*decoder_layers)

        # Init layers according to their activation functions.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, decoder_x, rgb_x):
        x = torch.cat((decoder_x, rgb_x), dim=1)
        x = [i(x) for i in self.diluted_conv_layers]
        x = torch.sum(torch.stack(x), dim=0)
        x = self.decoder_layers(x)
        return x


class PreliminaryDepth(nn.Module):
    def __init__(self, device, input_channels=3, num_blocks=2):
        super().__init__()
        self.device = device
        self.encoder = resnet34()

        # Naming convention: unet_block_x means the input is 1/n times the original resolution.
        # Output is 1/16
        self.unet_block_32 = RGBOnlyDecoderBlock((0, 512), (1, 3), 256, 256, num_blocks=num_blocks)
        # Output is 1/8
        self.unet_block_16 = RGBOnlyDecoderBlock((256, 256), (1, 3, 6), 128, 128, num_blocks=num_blocks)
        # Output is 1/4
        self.unet_block_8 = RGBOnlyDecoderBlock((128, 128), (1, 3, 6), 64, 64, num_blocks=num_blocks)
        # Output is 1/2
        self.unet_block_4 = RGBOnlyDecoderBlock((64, 64), (1, 3, 6), 64, 64, num_blocks=num_blocks)
        # Output is 1
        self.unet_block_2 = RGBOnlyDecoderBlock((64, 64), (1, 3, 6), 32, 32, num_blocks=num_blocks)
        self.unet_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu)  # Keep the output at 1.

        self.encoder.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

        if input_channels != 3:
            self.encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(self.encoder.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, color_input):
        color_out_2, color_out_4, color_out_8, color_out_16, color_out_32 = self.encoder(color_input)

        # Naming convention: unet_out_n is output for unet_block_n, so the size is 1 / (2 * n)
        unet_out_32 = self.unet_block_32(torch.Tensor().to(self.device), color_out_32)
        unet_out_16 = self.unet_block_16(unet_out_32, color_out_16)
        unet_out_8 = self.unet_block_8(unet_out_16, color_out_8)
        unet_out_4 = self.unet_block_4(unet_out_8, color_out_4)
        unet_out_2 = self.unet_block_2(unet_out_4, color_out_2)
        unet_out = self.unet_block(unet_out_2)

        return unet_out


class DepthRefinement(nn.Module):
    def __init__(self, device, rgb_input_channels=3, depth_input_channels=1, num_blocks=2, mode='prelim_depth_rgb'):
        super().__init__()
        if mode == 'prelim_depth_rgb':
            rgb_input_channels = rgb_input_channels + 1
            depth_input_channels = depth_input_channels + 1
        elif mode == 'prelim_depth_depth':
            rgb_input_channels = rgb_input_channels
            depth_input_channels = depth_input_channels + 2

        self.mode = mode
        self.device = device
        self.encoder = resnet34()
        self.depth_encoder = CBAMResNet(BasicBlock, [3, 4, 6, 3], in_channel=depth_input_channels)

        # Naming convention: unet_block_x means the input is 1/n times the original resolution.
        # Output is 1/16
        self.unet_block_32 = BasicDecoderBlock((0, 512, 512), (1, 3), 256, 256, num_blocks=num_blocks)
        # Output is 1/8
        self.unet_block_16 = BasicDecoderBlock((256, 256, 256), (1, 3, 6), 128, 128, num_blocks=num_blocks)
        # Output is 1/4
        self.unet_block_8 = BasicDecoderBlock((128, 128, 128), (1, 3, 6), 64, 64, num_blocks=num_blocks)
        # Output is 1/2
        self.unet_block_4 = BasicDecoderBlock((64, 64, 64), (1, 3, 6), 64, 64, num_blocks=num_blocks)
        # Output is 1
        self.unet_block_2 = BasicDecoderBlock((64, 64, 64), (1, 3, 6), 32, 32, num_blocks=num_blocks)
        self.unet_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dNormRelu)  # Keep the output at 1.

        self.encoder.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

        if rgb_input_channels != 3:
            self.encoder.conv1 = nn.Conv2d(rgb_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(self.encoder.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, color_input, raw_depth_input, preliminary_depth_input):
        depth_mask = (raw_depth_input < 10 ** -6) * 0.25 + 0.01

        if self.mode == 'prelim_depth_rgb':
            rgb_branch_input = torch.cat((color_input, preliminary_depth_input), 1)
            depth_branch_input = torch.cat((depth_mask, raw_depth_input), 1)
        elif self.mode == 'prelim_depth_depth':
            rgb_branch_input = color_input
            depth_branch_input = torch.cat((depth_mask, raw_depth_input, preliminary_depth_input), 1)

        color_out_2, color_out_4, color_out_8, color_out_16, color_out_32 = self.encoder(rgb_branch_input)
        depth_out_2, depth_out_4, depth_out_8, depth_out_16, depth_out_32 = self.depth_encoder(depth_branch_input)

        # Naming convention: unet_out_n is output for unet_block_n, so the size is 1 / (2 * n)
        unet_out_32 = self.unet_block_32(torch.Tensor().to(self.device), color_out_32, depth_out_32)
        unet_out_16 = self.unet_block_16(unet_out_32, color_out_16, depth_out_16)
        unet_out_8 = self.unet_block_8(unet_out_16, color_out_8, depth_out_8)
        unet_out_4 = self.unet_block_4(unet_out_8, color_out_4, depth_out_4)
        unet_out_2 = self.unet_block_2(unet_out_4, color_out_2, depth_out_2)
        unet_out = self.unet_block(unet_out_2)
        return unet_out


class StackedHourglass(nn.Module):
    def __init__(self, device, rgb_input_channels=3, depth_input_channels=1, num_blocks=2, mode='prelim_depth_rgb'):
        super().__init__()
        self.preliminary_depth = PreliminaryDepth(device, rgb_input_channels, num_blocks)
        self.depth_refinement = DepthRefinement(device, rgb_input_channels, depth_input_channels, num_blocks, mode)

    def forward(self, color_input, raw_depth_input):
        preliminary_depth = self.preliminary_depth(color_input)
        final_depth = self.depth_refinement(color_input, raw_depth_input, preliminary_depth)
        return final_depth, preliminary_depth


class SymmetricalStackedHourglass(nn.Module):
    def __init__(self, device, rgb_input_channels=3, depth_input_channels=1, num_blocks=2, mode='symmetrical'):
        super().__init__()
        self.preliminary_depth = PreliminaryDepth(device, rgb_input_channels, num_blocks)
        self.depth_refinement = PreliminaryDepth(device, rgb_input_channels + depth_input_channels + 2, num_blocks)

    def forward(self, color_input, raw_depth_input):
        preliminary_depth = self.preliminary_depth(color_input)

        depth_mask = (raw_depth_input < 10 ** -6) * 0.15 + 0.01
        second_input = torch.cat((color_input, depth_mask, raw_depth_input, preliminary_depth), 1)
        final_depth = self.depth_refinement(second_input)

        return final_depth, preliminary_depth


if __name__ == '__main__':
    # Sample usage and unit test.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StackedHourglass(device, mode='prelim_depth_depth')
    model.to(device)

    model.train()
    dummy_rgb = torch.rand(8, 3, 320, 256)
    dummy_depth = torch.rand(8, 1, 320, 256)
    dummy_final_pred, dummy_prelim_pred = model(dummy_rgb, dummy_depth)
    loss = dummy_final_pred.mean() + 0.5 * dummy_prelim_pred.mean()
    loss.backward()
    print(len(dummy_final_pred))
    print(dummy_final_pred[-1].shape)

    model = SymmetricalStackedHourglass(device)
    model.train()
    dummy_final_pred, dummy_prelim_pred = model(dummy_rgb, dummy_depth)
    loss = dummy_final_pred.mean() + 0.5 * dummy_prelim_pred.mean()
    loss.backward()
    print(len(dummy_final_pred))
    print(dummy_final_pred[-1].shape)

    model = StackedHourglass(device, mode='prelim_depth_rgb')
    model.train()
    dummy_final_pred, dummy_prelim_pred = model(dummy_rgb, dummy_depth)
    loss = dummy_final_pred.mean() + 0.5 * dummy_prelim_pred.mean()
    loss.backward()
    print(len(dummy_final_pred))
    print(dummy_final_pred[-1].shape)
