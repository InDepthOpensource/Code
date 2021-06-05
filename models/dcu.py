import torch.utils.data
import math
from torch import nn
from models.resnet import conv1x1, conv3x3, BasicBlock


# Implemented according to 3DV paper "Deeper Depth Prediction with Fully Convolutional Residual Networks
class UpProjectionBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpProjectionBlock, self).__init__()

        self.out_channels = out_channels

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1_1 = self.conv1_1(nn.functional.pad(x, [1, 1, 1, 1]))

        out1_2 = self.conv1_2(nn.functional.pad(x, [1, 1, 1, 0]))  # author's interleaving padding in github

        out1_3 = self.conv1_3(nn.functional.pad(x, [1, 0, 1, 1]))  # author's interleaving padding in github

        out1_4 = self.conv1_4(nn.functional.pad(x, [1, 0, 1, 0]))  # author's interleaving padding in github

        out2_1 = self.conv2_1(nn.functional.pad(x, [1, 1, 1, 1]))

        out2_2 = self.conv2_2(nn.functional.pad(x, [1, 1, 1, 0]))  # author's interleaving padding in github

        out2_3 = self.conv2_3(nn.functional.pad(x, [1, 0, 1, 1]))  # author's interleaving padding in github

        out2_4 = self.conv2_4(nn.functional.pad(x, [1, 0, 1, 0]))  # author's interleaving padding in github

        height = out1_1.size()[2]
        width = out1_1.size()[3]

        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).reshape(
            -1, self.out_channels, height, width * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).reshape(
            -1, self.out_channels, height, width * 2)

        out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).reshape(
            -1, self.out_channels, height * 2, width * 2)

        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).reshape(
            -1, self.out_channels, height, width * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).reshape(
            -1, self.out_channels, height, width * 2)

        out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).reshape(
            -1, self.out_channels, height * 2, width * 2)

        out1 = self.bn1_1(out1_1234)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2_1234)

        out = out1 + out2
        out = self.relu(out)

        return out


class DepthCompletionUnit(nn.Module):
    def __init__(self):
        super(DepthCompletionUnit, self).__init__()

        # Resnet blocks for color pipeline.
        self.conv_color_1x_0 = self._make_resnet_block(3, 32, 1)
        self.conv_color_1x_1 = self._make_resnet_block(32, 64, 1)
        self.conv_color_2x_0 = self._make_resnet_block(64, 128, 2)
        self.conv_color_2x_1 = self._make_resnet_block(128, 128, 1)
        self.conv_color_4x_0 = self._make_resnet_block(128, 256, 2)
        self.conv_color_4x_1 = self._make_resnet_block(256, 256, 1)
        self.conv_color_8x_0 = self._make_resnet_block(256, 256, 2)
        self.conv_color_8x_1 = self._make_resnet_block(256, 256, 1)
        self.conv_color_16x_0 = self._make_resnet_block(256, 512, 2)
        self.conv_color_16x_1 = self._make_resnet_block(512, 512, 1)

        # Resnet blocks for depth pipeline.
        self.conv_depth_1x_0 = self._make_resnet_block(1, 32, 1)
        self.conv_depth_1x_1 = self._make_resnet_block(32, 64, 1)
        self.conv_depth_2x_0 = self._make_resnet_block(64, 128, 2)
        self.conv_depth_2x_1 = self._make_resnet_block(128, 128, 1)
        self.conv_depth_4x_0 = self._make_resnet_block(128, 256, 2)
        self.conv_depth_4x_1 = self._make_resnet_block(256, 256, 1)
        self.conv_depth_8x_0 = self._make_resnet_block(256, 256, 2)
        self.conv_depth_8x_1 = self._make_resnet_block(256, 256, 1)
        self.conv_depth_16x_0 = self._make_resnet_block(256, 512, 2)
        self.conv_depth_16x_1 = self._make_resnet_block(512, 512, 1)

        self.deconv_16x = UpProjectionBlock(512, 256)
        self.deconv_8x = UpProjectionBlock(512, 128)
        self.deconv_4x = UpProjectionBlock(384, 64)
        self.deconv_2x = UpProjectionBlock(192, 32)

        self.conv_generate_depth = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_bottleneck(self, in_planes, planes, stride=1):
        return nn.Sequential(
            conv1x1(in_planes, planes, stride)
        )

    def _make_resnet_block(self, in_planes, planes, stride):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        return BasicBlock(in_planes, planes, stride, downsample)

    def forward(self, color_input, depth_input):
        conv_color_1x_out = self.conv_color_1x_1(self.conv_color_1x_0(color_input))
        conv_color_2x_out = self.conv_color_2x_1(self.conv_color_2x_0(conv_color_1x_out))
        conv_color_4x_out = self.conv_color_4x_1(self.conv_color_4x_0(conv_color_2x_out))
        conv_color_8x_out = self.conv_color_8x_1(self.conv_color_8x_0(conv_color_4x_out))
        conv_color_16x_out = self.conv_color_16x_1(self.conv_color_16x_0(conv_color_8x_out))

        conv_depth_1x_out = self.conv_depth_1x_1(self.conv_depth_1x_0(depth_input))
        conv_depth_2x_out = self.conv_depth_2x_1(self.conv_depth_2x_0(conv_depth_1x_out))
        conv_depth_4x_out = self.conv_depth_4x_1(self.conv_depth_4x_0(conv_depth_2x_out))
        conv_depth_8x_out = self.conv_depth_8x_1(self.conv_depth_8x_0(conv_depth_4x_out))
        conv_depth_16x_out = self.conv_depth_16x_1(self.conv_depth_16x_0(conv_depth_8x_out))

        deconv_16x_out = conv_color_16x_out + conv_depth_16x_out
        deconv_16x_out = self.deconv_16x(deconv_16x_out)

        deconv_8x_out = torch.cat((conv_color_8x_out + conv_depth_8x_out, deconv_16x_out), 1)
        deconv_8x_out = self.deconv_8x(deconv_8x_out)

        deconv_4x_out = torch.cat((conv_color_4x_out + conv_depth_4x_out, deconv_8x_out), 1)
        deconv_4x_out = self.deconv_4x(deconv_4x_out)

        deconv_2x_out = torch.cat((conv_color_2x_out + conv_depth_2x_out, deconv_4x_out), 1)
        deconv_2x_out = self.deconv_2x(deconv_2x_out)

        reconstructed_depth = self.relu(self.conv_generate_depth(deconv_2x_out))

        return reconstructed_depth
