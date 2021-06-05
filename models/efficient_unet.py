import torch
from torch import nn
from models import EfficientNet, BasicDecoderBlock, LastUnetDecoderBlock


class EfficientUNet(nn.Module):
    def __init__(self, device, num_res_block_decoder=2, efficient_net_type='efficientnet-b0',
                 decoder_input_lengths=((0, 320, 320), (112, 112, 112), (40, 40, 40), (24, 24, 24), (16, 16, 16)),
                 decoder_dilation_rates=((1, 3), (1, 3, 6), (1, 3, 6), (1, 3, 6), (1, 3, 6)),
                 decoder_internal_channels=(112, 40, 24, 16, 16),
                 decoder_output_channels=(112, 40, 24, 16, 16)):

        super().__init__()

        self.device = device
        self.rgb_encoder = EfficientNet.from_pretrained(efficient_net_type)
        self.depth_encoder = EfficientNet.from_name(efficient_net_type, in_channels=2)

        self.decoder = nn.ModuleList([
            BasicDecoderBlock(decoder_input_lengths[0], decoder_dilation_rates[0], decoder_internal_channels[0],
                              decoder_output_channels[0], num_blocks=num_res_block_decoder),
            BasicDecoderBlock(decoder_input_lengths[1], decoder_dilation_rates[1], decoder_internal_channels[1],
                              decoder_output_channels[1], num_blocks=num_res_block_decoder),
            BasicDecoderBlock(decoder_input_lengths[2], decoder_dilation_rates[2], decoder_internal_channels[2],
                              decoder_output_channels[2], num_blocks=num_res_block_decoder),
            BasicDecoderBlock(decoder_input_lengths[3], decoder_dilation_rates[3], decoder_internal_channels[3],
                              decoder_output_channels[3], num_blocks=num_res_block_decoder),
            BasicDecoderBlock(decoder_input_lengths[4], decoder_dilation_rates[4], decoder_internal_channels[4],
                              decoder_output_channels[4], num_blocks=num_res_block_decoder),
            LastUnetDecoderBlock(decoder_output_channels[4], 1, intermediate_ch=decoder_output_channels[4] // 2)
        ])

    def forward(self, color_input, depth_input):
        depth_mask = (depth_input < 10 ** -6) * 0.2 + 0.02
        depth_input = torch.cat((depth_mask, depth_input), 1)

        rgb_endpoints = self.rgb_encoder.extract_endpoints(color_input)
        depth_endpoints = self.depth_encoder.extract_endpoints(depth_input)

        rgb_representations = [rgb_endpoints['reduction_5'], rgb_endpoints['reduction_4'], rgb_endpoints['reduction_3'],
                               rgb_endpoints['reduction_2'], rgb_endpoints['reduction_1']]
        depth_representations = [depth_endpoints['reduction_5'], depth_endpoints['reduction_4'],
                                 depth_endpoints['reduction_3'], depth_endpoints['reduction_2'],
                                 depth_endpoints['reduction_1']]

        # for i in rgb_representations:
        #     print(i.shape)

        depth_feature = torch.Tensor().to(self.device)
        for i in range(5):
            depth_feature = self.decoder[i](depth_feature, rgb_representations[i], depth_representations[i])
        return self.decoder[-1](depth_feature)


# Test and Debug
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    efficient_unet = EfficientUNet(device, efficient_net_type='efficientnet-b0')
    efficient_unet.train()
    dummy_rgb = torch.rand(16, 3, 320, 256)
    dummy_depth = torch.rand(16, 1, 320, 256)
    dummy_pred = efficient_unet(dummy_rgb, dummy_depth)
    loss = dummy_pred.mean()
    loss.backward()

    # efficient_unet = EfficientUNet(device, efficient_net_type='efficientnet-b3',
    #                       decoder_input_lengths=(
    #                           (0, 384, 384), (136, 136, 136), (48, 48, 48), (32, 32, 32), (24, 24, 24)),
    #                       decoder_dilation_rates=((1, 3), (1, 3, 6), (1,), (1,), (1,)),
    #                       decoder_internal_channels=(136, 48, 32, 24, 24),
    #                       decoder_output_channels=(136, 48, 32, 24, 24))
    # efficient_unet.train()
    # dummy_rgb = torch.rand(16, 3, 320, 256)
    # dummy_depth = torch.rand(16, 1, 320, 256)
    # dummy_pred = efficient_unet(dummy_rgb, dummy_depth)
    # loss = dummy_pred.mean()
    # loss.backward()
