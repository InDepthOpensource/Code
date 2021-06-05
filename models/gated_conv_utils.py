import torch
import torch.nn as nn
import torch.nn.functional as F
from models import conv3x3, conv1x1


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm=nn.BatchNorm2d,
                 mask_norm=nn.BatchNorm2d, activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.activation = activation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if norm:
            self.output_norm = norm(out_channels)
        else:
            self.output_norm = None

        if mask_norm:
            self.mask_norm = mask_norm(out_channels)
        else:
            self.mask_norm = None

        self.activation = activation
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        output = self.conv2d(x)
        if self.output_norm:
            output = self.output_norm(output)
        if self.activation:
            output = self.activation(output)

        mask = self.mask_conv2d(x)
        if self.mask_norm:
            mask = self.mask_norm(mask)
        mask = self.sigmoid(mask)

        return output * mask


class GatedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.gated_conv_1 = GatedConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.gated_conv_2 = GatedConv2d(planes, planes, kernel_size=3, padding=1, activation=None)

        self.downsample = downsample
        self.stride = stride

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x
        out = self.gated_conv_1(x)
        out = self.gated_conv_2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leaky_relu(out)

        return out


class GatedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.gated_conv_1 = GatedConv2d(inplanes, planes, kernel_size=1, stride=1, padding=0)
        self.gated_conv_2 = GatedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.gated_conv_3 = GatedConv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0,
                                        activation=None
                                        )
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.gated_conv_1(x)

        out = self.gated_conv_2(out)

        out = self.gated_conv_3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leaky_relu(out)

        return out


class GatedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channel=2):
        super().__init__()
        self.inplanes = 64
        # self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)

        self.gated_conv1 = GatedConv2d(in_channel, 64, kernel_size=7, stride=2, padding=3)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Naming convention here: features_n means the resolution has been reduced to 1/n.
        out_2 = self.gated_conv1(x)
        x = self.maxpool(out_2)

        out_4 = self.layer1(x)
        out_8 = self.layer2(out_4)
        out_16 = self.layer3(out_8)
        out_32 = self.layer4(out_16)

        return out_2, out_4, out_8, out_16, out_32


def gated_resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GatedResNet(GatedBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def gated_resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GatedResNet(GatedBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def gated_resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GatedResNet(GatedBottleneck, [3, 4, 23, 3], **kwargs)
    return model
