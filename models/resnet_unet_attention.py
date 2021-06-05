import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet50, resnet34, resnet101
from models import gated_resnet34, gated_resnet50, gated_resnet101, GatedConv2d, model_zoo, model_urls


class BilinearUpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.interpolate = F.interpolate

    def forward(self, x):
        x = self.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class Conv2dNormRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d):
        super().__init__()

        if norm:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=False),
                norm(out_channel),
                nn.ReLU(inplace=True)
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.layers(x)
        return x


class Conv2dRelu(Conv2dNormRelu):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, norm=None):
        super().__init__(in_channel, out_channel, kernel_size, stride, padding, norm)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, conv_type=GatedConv2d, up_sample='bilinear'):
        super().__init__()

        if up_sample == 'bilinear':
            up_sample = BilinearUpSample()
        else:
            up_sample = None

        layers_list = [up_sample] if up_sample is not None else []
        layers_list = [conv_type(in_ch, out_ch, 3, padding=1),
                       conv_type(out_ch, out_ch, 3, padding=1)
                       ] + layers_list

        self.layers = nn.Sequential(*layers_list)

        init_type = 'leaky_relu' if conv_type == GatedConv2d else 'relu'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=init_type)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        return x


# We need to scale up the output to original resolution
class LastUnetDecoderBlock(UnetDecoderBlock):
    def __init__(self, in_ch, out_ch=1, conv_type=Conv2dNormRelu, intermediate_ch=-1):
        super().__init__(in_ch, out_ch, conv_type, 'bilinear')

        intermediate_ch = in_ch if intermediate_ch == -1 else intermediate_ch

        layers_list = [
            conv_type(in_ch, intermediate_ch, 3, padding=1),
            conv_type(intermediate_ch, out_ch, 3, padding=1, norm=None),
        ]

        self.layers = nn.Sequential(*layers_list)

        init_type = 'leaky_relu' if conv_type == GatedConv2d else 'relu'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=init_type)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ResNet34UnetAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.rgb_encoder = resnet34()
        self.depth_encoder = gated_resnet34()

        # Naming convention: unet_block_x means the input is 1/n times the original resolution.
        self.unet_block_32 = UnetDecoderBlock(512, 256)  # Output is 1/16
        self.unet_block_16 = UnetDecoderBlock(512, 128)  # Output is 1/8
        self.unet_block_8 = UnetDecoderBlock(256, 64)  # Output is 1/4
        self.unet_block_4 = UnetDecoderBlock(128, 64)  # Output is 1/2
        self.unet_block_2 = UnetDecoderBlock(128, 32)  # Output is 1
        self.unet_block = LastUnetDecoderBlock(32, 1)  # Keep the output at 1.

        self.rgb_encoder.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

    def forward(self, color_input, depth_input):
        depth_mask = (depth_input < 10 ** -6) * 2.0 + 0.1
        depth_input = torch.cat((depth_mask, depth_input), 1)

        color_out_2, color_out_4, color_out_8, color_out_16, color_out_32 = self.rgb_encoder(color_input)
        depth_out_2, depth_out_4, depth_out_8, depth_out_16, depth_out_32 = self.depth_encoder(depth_input)

        # Naming convention: unet_out_n is output for unet_block_n, so the size is 1 / (2 * n)
        unet_out_32 = self.unet_block_32(color_out_32 + depth_out_32)
        unet_out_16 = self.unet_block_16(torch.cat((color_out_16 + depth_out_16, unet_out_32), 1))
        unet_out_8 = self.unet_block_8(torch.cat((color_out_8 + depth_out_8, unet_out_16), 1))
        unet_out_4 = self.unet_block_4(torch.cat((color_out_4 + depth_out_4, unet_out_8), 1))
        unet_out_2 = self.unet_block_2(torch.cat((color_out_2 + depth_out_2, unet_out_4), 1))
        unet_out = self.unet_block(unet_out_2)

        return unet_out


class ResNet34UnetAttentionNoDecoderNorm(ResNet34UnetAttention):
    def __init__(self):
        super().__init__()
        # Naming convention: unet_block_x means the input is 1/n times the original resolution.
        self.unet_block_32 = UnetDecoderBlock(512, 256, conv_type=Conv2dRelu)  # Output is 1/16
        self.unet_block_16 = UnetDecoderBlock(512, 128, conv_type=Conv2dRelu)  # Output is 1/8
        self.unet_block_8 = UnetDecoderBlock(256, 64, conv_type=Conv2dRelu)  # Output is 1/4
        self.unet_block_4 = UnetDecoderBlock(128, 64, conv_type=Conv2dRelu)  # Output is 1/2
        self.unet_block_2 = UnetDecoderBlock(128, 32, conv_type=Conv2dRelu)  # Output is 1
        self.unet_block = LastUnetDecoderBlock(32, 1, conv_type=Conv2dRelu)  # Keep the output at 1.


class ResNet34Unet(ResNet34UnetAttention):
    def __init__(self):
        super().__init__()
        self.depth_encoder = resnet34(in_channel=2)

        # Naming convention: unet_block_x means the input is 1/n times the original resolution.
        self.unet_block_32 = UnetDecoderBlock(512, 256, Conv2dNormRelu)  # Output is 1/16
        self.unet_block_16 = UnetDecoderBlock(512, 128, Conv2dNormRelu)  # Output is 1/8
        self.unet_block_8 = UnetDecoderBlock(256, 64, Conv2dNormRelu)  # Output is 1/4
        self.unet_block_4 = UnetDecoderBlock(128, 64, Conv2dNormRelu)  # Output is 1/2
        self.unet_block_2 = UnetDecoderBlock(128, 32, Conv2dNormRelu)  # Output is 1
        self.unet_block = LastUnetDecoderBlock(32, 1, Conv2dNormRelu)  # Keep the output at 1.

        self.rgb_encoder.load_state_dict(model_zoo.load_url(model_urls['resnet34']))


class ResNet34UnetNoDecoderNorm(ResNet34UnetAttention):
    def __init__(self):
        super().__init__()
        self.depth_encoder = resnet34(in_channel=2)

        # Naming convention: unet_block_x means the input is 1/n times the original resolution.
        self.unet_block_32 = UnetDecoderBlock(512, 256, Conv2dRelu)  # Output is 1/16
        self.unet_block_16 = UnetDecoderBlock(512, 128, Conv2dRelu)  # Output is 1/8
        self.unet_block_8 = UnetDecoderBlock(256, 64, Conv2dRelu)  # Output is 1/4
        self.unet_block_4 = UnetDecoderBlock(128, 64, Conv2dRelu)  # Output is 1/2
        self.unet_block_2 = UnetDecoderBlock(128, 32, Conv2dRelu)  # Output is 1
        self.unet_block = LastUnetDecoderBlock(32, 1, Conv2dRelu)  # Keep the output at 1.


class ResNet101UnetFullAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.rgb_encoder = resnet101()
        self.depth_encoder = gated_resnet101()

        # Naming convention: unet_block_x means the input is 1/n times the original resolution.
        self.unet_block_32 = UnetDecoderBlock(2048, 1024)  # Output is 1/16
        self.unet_block_16 = UnetDecoderBlock(2048, 512)  # Output is 1/8
        self.unet_block_8 = UnetDecoderBlock(1024, 256)  # Output is 1/4
        self.unet_block_4 = UnetDecoderBlock(512, 64)  # Output is 1/2
        self.unet_block_2 = UnetDecoderBlock(128, 32)  # Output is 1
        self.unet_block = LastUnetDecoderBlock(32, 1)  # Keep the output at 1.

        self.rgb_encoder.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

    def forward(self, color_input, depth_input):
        depth_mask = (depth_input < 10 ** -6) * 2.0 + 0.1
        depth_input = torch.cat((depth_mask, depth_input), 1)

        color_out_2, color_out_4, color_out_8, color_out_16, color_out_32 = self.rgb_encoder(color_input)
        depth_out_2, depth_out_4, depth_out_8, depth_out_16, depth_out_32 = self.depth_encoder(depth_input)

        # Naming convention: unet_out_n is output for unet_block_n, so the size is 1 / (2 * n)
        unet_out_32 = self.unet_block_32(color_out_32 + depth_out_32)
        unet_out_16 = self.unet_block_16(torch.cat((color_out_16 + depth_out_16, unet_out_32), 1))
        unet_out_8 = self.unet_block_8(torch.cat((color_out_8 + depth_out_8, unet_out_16), 1))
        unet_out_4 = self.unet_block_4(torch.cat((color_out_4 + depth_out_4, unet_out_8), 1))
        unet_out_2 = self.unet_block_2(torch.cat((color_out_2 + depth_out_2, unet_out_4), 1))
        unet_out = self.unet_block(unet_out_2)

        return unet_out


class ResNet101UnetDecoderAttention(ResNet101UnetFullAttention):
    def __init__(self):
        super().__init__()
        self.depth_encoder = resnet101(in_channel=2)


class ResNet101UnetNoAttention(ResNet101UnetFullAttention):
    def __init__(self):
        super().__init__()

        self.rgb_encoder = resnet101()
        self.depth_encoder = resnet101(in_channel=2)

        # Naming convention: unet_block_x means the input is 1/n times the original resolution.
        self.unet_block_32 = UnetDecoderBlock(2048, 1024, Conv2dNormRelu)  # Output is 1/16
        self.unet_block_16 = UnetDecoderBlock(2048, 512, Conv2dNormRelu)  # Output is 1/8
        self.unet_block_8 = UnetDecoderBlock(1024, 256, Conv2dNormRelu)  # Output is 1/4
        self.unet_block_4 = UnetDecoderBlock(512, 64, Conv2dNormRelu)  # Output is 1/2
        self.unet_block_2 = UnetDecoderBlock(128, 32, Conv2dNormRelu)  # Output is 1
        self.unet_block = LastUnetDecoderBlock(32, 1, Conv2dNormRelu)  # Keep the output at 1.

        self.rgb_encoder.load_state_dict(model_zoo.load_url(model_urls['resnet101']))


class ResNet10150UnetFullAttention(ResNet101UnetFullAttention):
    def __init__(self):
        super().__init__()
        self.depth_encoder = gated_resnet50()


class ResNet10150UnetDecoderAttention(ResNet101UnetFullAttention):
    def __init__(self):
        super().__init__()
        self.depth_encoder = resnet50(in_channel=2)


class ResNet10150UnetNoAttention(ResNet101UnetNoAttention):
    def __init__(self):
        super().__init__()
        self.depth_encoder = resnet50(in_channel=2)
