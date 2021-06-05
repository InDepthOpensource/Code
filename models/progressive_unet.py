import torch
import torch.nn as nn

from models import resnet34, CBAMBasicBlock, BasicBlock, CBAMResNet, conv1x1, BilinearUpSample, LastUnetDecoderBlock


class BasicDecoderBlock(nn.Module):
    def __init__(self, input_channels, dilation, internal_channels, output_channels, num_blocks=2, up_sample='bilinear'):
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

    def forward(self, decoder_x, rgb_x, depth_x):
        x = torch.cat((decoder_x, rgb_x, depth_x), dim=1)
        x = [i(x) for i in self.diluted_conv_layers]
        x = torch.sum(torch.stack(x), dim=0)
        x = self.decoder_layers(x)
        return x


class ProgressiveUNet(nn.Module):
    def __init__(self, device, use_resblock=True, use_cbam_fuse=False, use_cbam_encoder=False,
                 use_cbam_decoder=False, num_res_block_decoder=2):
        super().__init__()
        if not use_resblock or use_cbam_fuse or use_cbam_decoder:
            raise NotImplementedError('We does not support CBAM as it seems to negatively impact performance')

        self.rgb_encoder = resnet34()

        # Define the input dimensions for decoder block level
        self.decoder_input_lengths = ((0, 512, 512), (256, 256, 256), (128, 128, 128), (64, 64, 64), (64, 64, 64))
        self.decoder_dilation_rates = ((1, 3), (1, 3, 6), (1, 3, 6), (1, 3, 6), (1, 3, 6))
        self.decoder_internal_channels = (256, 128, 64, 64, 32)
        self.decoder_output_channels = (256, 128, 64, 64, 32)

        self.device = device

        depth_encoder_basic_block = CBAMBasicBlock if use_cbam_encoder else BasicBlock
        self.depth_encoder = CBAMResNet(depth_encoder_basic_block, [3, 4, 6, 3], in_channel=2)

        self.decoder = nn.ModuleList([
            BasicDecoderBlock(self.decoder_input_lengths[0],
                              self.decoder_dilation_rates[0],
                              self.decoder_internal_channels[0],
                              self.decoder_output_channels[0],
                              num_blocks=num_res_block_decoder),
            LastUnetDecoderBlock(self.decoder_output_channels[0], 1,
                                 intermediate_ch=self.decoder_output_channels[0] // 2),
            BasicDecoderBlock(self.decoder_input_lengths[1],
                              self.decoder_dilation_rates[1],
                              self.decoder_internal_channels[1],
                              self.decoder_output_channels[1],
                              num_blocks=num_res_block_decoder),
            LastUnetDecoderBlock(self.decoder_output_channels[1], 1,
                                 intermediate_ch=self.decoder_output_channels[1] // 2),
        ])

        self.current_level = 1
        self.num_res_block_decoder = num_res_block_decoder

    # Add another level to the decoder
    def add_decoder_block(self):
        if self.current_level < 4:
            self.current_level += 1
            self.decoder.extend([
                BasicDecoderBlock(self.decoder_input_lengths[self.current_level],
                                  self.decoder_dilation_rates[self.current_level],
                                  self.decoder_internal_channels[self.current_level],
                                  self.decoder_output_channels[self.current_level],
                                  self.num_res_block_decoder).to(self.device),
                LastUnetDecoderBlock(self.decoder_output_channels[self.current_level], 1,
                                     intermediate_ch=self.decoder_output_channels[self.current_level] // 2).to(
                    self.device),
            ])

    def forward(self, color_input, depth_input):
        depth_mask = (depth_input < 10 ** -6) * 0.5 + 0.05
        depth_input = torch.cat((depth_mask, depth_input), 1)

        rgb_representation = self.rgb_encoder(color_input)
        depth_representation = self.depth_encoder(depth_input)

        rgb_representation = rgb_representation[::-1]
        depth_representation = depth_representation[::-1]

        depth_outputs = []
        depth_feature = torch.Tensor().to(self.device)
        for i in range(self.current_level + 1):
            depth_feature = self.decoder[2 * i](depth_feature,
                                                rgb_representation[i],
                                                depth_representation[i])
            depth_output = self.decoder[2 * i + 1](depth_feature)
            depth_outputs.append(depth_output)
        return depth_outputs


# Test current
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProgressiveUNet(device, num_res_block_decoder=3)
    model.to(device)

    model.train()
    dummy_rgb = torch.rand(16, 3, 320, 256)
    dummy_depth = torch.rand(16, 1, 320, 256)
    dummy_pred = model(dummy_rgb, dummy_depth)
    loss = dummy_pred[-1].mean()
    loss.backward()
    print(len(dummy_pred))
    print(dummy_pred[-1].shape)

    for _ in range(3):
        model.add_decoder_block()
        dummy_rgb = torch.rand(16, 3, 320, 256)
        dummy_depth = torch.rand(16, 1, 320, 256)
        dummy_pred = model(dummy_rgb, dummy_depth)
        loss = dummy_pred[-1].mean()
        loss.backward()
        print(len(dummy_pred))
        print(dummy_pred[-1].shape)
