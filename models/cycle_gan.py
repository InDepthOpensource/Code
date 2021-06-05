import torch
import torch.nn as nn
import torch.nn.functional as F


# Credit: Adapted from https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/

# Note the ResNetUNet structure used in cycle gan is different from that of ordinary ResNetUNet:
# Downsample is done between blocks, not inside the block.
from models import BilinearUpSample


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


# Note the ResNetUNet structure used in cycle gan is different from that of ordinary ResNetUNet:
# Downsample is done between blocks, not inside the block.
# TODO: currently we only adapt for depth. In the future we may also want to adapt for RGB.
class CycleGanGenerator(nn.Module):
    def __init__(self, input_channels=5, output_channels=1, n_residual_blocks=9, depth_output=True):
        super().__init__()
        self.depth_output = depth_output
        self.register_buffer('output_scale', torch.tensor(0.6))

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channels, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [BilinearUpSample(),
                      nn.Conv2d(in_features, out_features, 3, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_channels, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.depth_output:
            rgb_input = x[:, :3, :, :]
            depth_mask = (x[:, -1, :, :] < 10 ** -6) * 0.9 + 0.01
            depth_mask = torch.unsqueeze(depth_mask, dim=1)
            x = torch.cat((x, depth_mask), dim=1)
        else:
            depth_input = torch.unsqueeze(x[:, -1, :, :], dim=1)
            x = x[:, :3, :, :]

        x = self.model(x)

        if self.depth_output:
            x = x + torch.ones_like(x) * self.output_scale
            x = torch.cat((rgb_input, x), dim=1)
        else:
            x = torch.cat((x, depth_input), dim=1)
        return x


class CycleGanPatchDiscriminator(nn.Module):
    def __init__(self, input_channels=1, discriminate_depth=True):
        super().__init__()
        self.discriminate_depth = discriminate_depth
        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.discriminate_depth:
            # depth_mask = (x[:, -1, :, :] < 10 ** -6) * 0.9 + 0.01
            # depth_mask = torch.unsqueeze(depth_mask, dim=1)
            # x = torch.cat((x, depth_mask), dim=1)
            x = x[:, -1, :, :]
            x = torch.unsqueeze(x, dim=1)
        else:
            x = x[:, :3, :, :]
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
