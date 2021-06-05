import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as autograd


def compute_gradient_penalty(critic, color_input, depth_input, fake_depth, real_depth):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.

       This is currently not used as we choose HingeGAN (same as deepfill) over WGAN-GP
    """
    # Random weight term for interpolation between real and fake samples
    batch_size = color_input.size()[0]
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(color_input)
    # Get random interpolation between real and fake samples
    interpolated = (alpha * real_depth + ((1 - alpha) * fake_depth))
    interpolated = Variable(interpolated, requires_grad=True).cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = critic(color_input, depth_input, interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = autograd(outputs=prob_interpolated, inputs=interpolated,
                         grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                         create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return 10 * ((gradients_norm - 1) ** 2).mean()


class PatchCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Naming convention: conv_x means the input is 1/n times the original resolution.
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.utils.spectral_norm(nn.Conv2d(5, 64, 5, 2, 2))
        self.conv_2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 5, 2, 2))
        self.conv_4 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 5, 2, 2))
        self.conv_8 = nn.utils.spectral_norm(nn.Conv2d(256, 256, 5, 2, 2))
        self.conv_16 = nn.utils.spectral_norm(nn.Conv2d(256, 256, 5, 2, 2))
        self.conv_32 = nn.utils.spectral_norm(nn.Conv2d(256, 1, 5, 2, 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, color_input, depth_input, depth_output):
        depth_output = depth_output / 16.0
        depth_mask = (depth_input < 10 ** -6) * 1.0 + 0.01
        x = torch.cat((color_input, depth_output, depth_mask), 1)
        x = self.leaky_relu(self.conv_1(x))
        x = self.leaky_relu(self.conv_2(x))
        x = self.leaky_relu(self.conv_4(x))
        x = self.leaky_relu(self.conv_8(x))
        x = self.leaky_relu(self.conv_16(x))
        # Output should be (batch_size, 1, 5, 4). Receptive field is 253.
        x = self.leaky_relu(self.conv_32(x))

        return x


class ModifiedVGGCritic(nn.Module):
    """A modified VGG discriminator with input size 256 x 320.
    Borrowed from SRGAN and ESRGAN.
    Args:
        in_channels (int): Channel number of inputs. Default: 5.
        mid_channels (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, in_channels=5, mid_channels=64):
        super().__init__()

        self.conv0_0 = nn.utils.spectral_norm(nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True))
        self.conv0_1 = nn.utils.spectral_norm(nn.Conv2d(
            mid_channels, mid_channels, 4, 2, 1, bias=False))

        self.conv1_0 = nn.utils.spectral_norm(nn.Conv2d(
            mid_channels, mid_channels * 2, 3, 1, 1, bias=False))
        self.conv1_1 = nn.utils.spectral_norm(nn.Conv2d(
            mid_channels * 2, mid_channels * 2, 4, 2, 1, bias=False))

        self.conv2_0 = nn.utils.spectral_norm(nn.Conv2d(
            mid_channels * 2, mid_channels * 4, 3, 1, 1, bias=False))
        self.conv2_1 = nn.utils.spectral_norm(nn.Conv2d(
            mid_channels * 4, mid_channels * 4, 4, 2, 1, bias=False))

        self.conv3_0 = nn.utils.spectral_norm(nn.Conv2d(
            mid_channels * 4, mid_channels * 8, 3, 1, 1, bias=False))
        self.conv3_1 = nn.utils.spectral_norm(nn.Conv2d(
            mid_channels * 8, mid_channels * 8, 4, 2, 1, bias=False))

        self.conv4_0 = nn.utils.spectral_norm(nn.Conv2d(
            mid_channels * 8, mid_channels * 8, 3, 1, 1, bias=False))
        self.conv4_1 = nn.utils.spectral_norm(nn.Conv2d(
            mid_channels * 8, mid_channels * 8, 4, 2, 1, bias=False))

        self.conv5_0 = nn.utils.spectral_norm(nn.Conv2d(
            mid_channels * 8, mid_channels * 8, 3, 1, 1, bias=False))
        self.conv5_1 = nn.utils.spectral_norm(nn.Conv2d(
            mid_channels * 8, mid_channels * 8, 4, 2, 1, bias=False))

        self.linear1 = nn.Linear(mid_channels * 8 * 4 * 5, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # The original authors used the pytorch default init, so we follow the same.

    def forward(self, color_input, depth_input, depth_output):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """

        depth_output = depth_output / 16.0
        depth_mask = (depth_input < 10 ** -6) * 1.0 + 0.01
        x = torch.cat((color_input, depth_output, depth_mask), 1)

        assert x.size(2) == 256 and x.size(3) == 320, (
            f'Input spatial size must be h=256 and w=320, '
            f'but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.conv0_1(feat)) # output spatial size: (64, 64)

        feat = self.lrelu(self.conv1_0(feat))
        feat = self.lrelu(self.conv1_1(feat))  # output spatial size: (32, 32)

        feat = self.lrelu(self.conv2_0(feat))
        feat = self.lrelu(self.conv2_1(feat))  # output spatial size: (16, 16)

        feat = self.lrelu(self.conv3_0(feat))
        feat = self.lrelu(self.conv3_1(feat))  # output spatial size: (8, 8)

        feat = self.lrelu(self.conv4_0(feat))
        feat = self.lrelu(self.conv4_1(feat))  # output spatial size: (4, 4)

        feat = self.lrelu(self.conv5_0(feat))
        feat = self.lrelu(self.conv5_1(feat))  # output spatial size: (4, 4)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out
