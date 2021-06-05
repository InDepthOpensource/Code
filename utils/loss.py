import torch
import torch.nn as nn
from torch import Tensor, mul, dot, ones
from torch.nn import functional as F
from utils import VNLLoss


class BerHuLoss(nn.Module):
    def __init__(self):
        super(BerHuLoss, self).__init__()
        self.register_buffer('cap', torch.tensor((2 ** 16 - 1) / 4000))

    def forward(self, pred, depth_target):
        diff = (depth_target - pred).abs()
        # Cap the max depth to (2 ** 16 - 1) / 4000

        threshold = 0.2 * torch.min(diff.max(), self.cap)

        quadratic_mask = ((diff > threshold) & (depth_target > 10 ** -6)).detach()
        absolute_mask = ((diff <= threshold) & (depth_target > 10 ** -6)).detach()

        less = diff[absolute_mask]
        more = diff[quadratic_mask]

        more = (more ** 2 + threshold ** 2) / (2 * threshold + 10 ** -9)
        loss = torch.cat((less, more)).mean()
        return loss


class MultiScaleLoss(nn.Module):
    def __init__(self, coefficients=(0.025, 0.05, 0.1, 0.2, 1.0), original_size=(256, 320), loss_func=BerHuLoss):
        super().__init__()
        self.coefficients = coefficients
        self.original_size = original_size
        self.loss_func = loss_func()

    def forward(self, pred, depth_target):
        depth_targets = []
        steps = [16, 8, 4, 2]
        with torch.no_grad():
            for i in steps:
                depth_targets.append(nn.functional.interpolate(depth_target,
                                                               size=[self.original_size[0] // i,
                                                                     self.original_size[1] // i],
                                                               mode='bilinear',
                                                               align_corners=True))
            depth_targets.append(depth_target)

        losses = []
        for i in range(len(pred)):
            losses.append(self.loss_func(pred[i], depth_targets[i]))

        losses_item = [i.item() for i in losses]
        losses_item = losses_item + ([0.0] * (len(depth_targets) - len(losses_item)))

        # For Debug.
        # for i in depth_targets:
        #     print('target shape:', i.shape)
        # for i in range(len(pred)):
        #     print('pred shape:', pred[i].shape)
        # print('loss item:', losses_item)
        # print('loss vector', losses)

        # Scale the losses and then sum:
        for i in range(len(pred)):
            losses[i] = self.coefficients[i] * losses[i]
        losses = torch.sum(torch.stack(losses), dim=0)

        return losses_item, losses


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.register_buffer('sobel_x', (1 / 8) * torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', (1 / 8) * torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3))

    def forward(self, pred, depth_target):
        pred = pred.clamp(0.0, (2 ** 16 - 1) / 4000)
        error = pred - depth_target

        gx_diff = F.conv2d(error, self.sobel_x, padding=1).abs()
        gy_diff = F.conv2d(error, self.sobel_y, padding=1).abs()

        mask = (depth_target > 10 ** -6).detach()
        gx_diff = gx_diff[mask]
        gy_diff = gy_diff[mask]
        loss = torch.cat((gx_diff, gy_diff)).clamp_max(0.1).mean()

        return loss


class NormalFromDepthLoss(nn.Module):
    def __init__(self):
        super(NormalFromDepthLoss, self).__init__()
        self.register_buffer('sobel_x', (1 / 8) * torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', (1 / 8) * torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3))

    def forward(self, pred, depth_target):
        mask = (depth_target > 10 ** -6).repeat(1, 2, 1, 1).detach()

        depth_target_gx = F.conv2d(depth_target, self.sobel_x, padding=1).clamp_max(0.1)
        depth_target_gy = F.conv2d(depth_target, self.sobel_y, padding=1).clamp_max(0.1)
        depth_target_normal = torch.cat((depth_target_gx, depth_target_gy), 1)
        depth_target_normal = F.normalize(depth_target_normal, p=2, dim=1)
        depth_target_normal = depth_target_normal[mask]

        pred_gx = F.conv2d(pred, self.sobel_x, padding=1).clamp_max(0.1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1).clamp_max(0.1)
        pred_normal = torch.cat((pred_gx, pred_gy), 1)
        pred_normal = F.normalize(pred_normal, p=2, dim=1)
        pred_normal = pred_normal[mask]

        prod = mul(depth_target_normal, pred_normal)
        loss = 1.0 - prod.mean()

        return loss


class FinalTrainingLoss(nn.Module):
    def __init__(self, berhu_coeff=1.0, gradient_coeff=15.0, normal_from_depth_coeff=0.17):
        super(FinalTrainingLoss, self).__init__()
        self.berhu_loss = BerHuLoss()
        self.gradient_loss = GradientLoss()
        self.vnl_loss = VNLLoss()

        self.use_vnl = (normal_from_depth_coeff > 10 ** -6)

        self.register_buffer('berhu_coeff', torch.tensor(berhu_coeff))
        self.register_buffer('gradient_coeff', torch.tensor(gradient_coeff))
        self.register_buffer('normal_from_depth_coeff', torch.tensor(normal_from_depth_coeff))

    def forward(self, pred, depth_target):
        mask = ((pred < 10 ** -6) | (depth_target < 10 ** -6)).detach()
        pred[mask] = 10 ** -5
        depth_target[mask] = 10 ** -5
        batch_berhu_loss = self.berhu_loss(pred, depth_target)
        batch_gradient_loss = self.gradient_loss(pred, depth_target)
        if self.use_vnl:
            batch_normal_from_depth_loss = self.vnl_loss(pred, depth_target)
        else:
            batch_normal_from_depth_loss = batch_berhu_loss * self.normal_from_depth_coeff
        loss = self.berhu_coeff * batch_berhu_loss + self.gradient_coeff * batch_gradient_loss + \
               self.normal_from_depth_coeff * batch_normal_from_depth_loss
        return loss


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mask = (target > 10 ** -6).detach()
        total_error = ((target - pred) ** 2)[mask].sum()
        num_of_valid_pixels = mask.sum()
        return total_error, num_of_valid_pixels


class L1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mask = (target > 10 ** -6).detach()
        total_error = (target - pred).abs()[mask].sum()
        num_of_valid_pixels = mask.sum()
        return total_error, num_of_valid_pixels


class PercentWithinError(nn.Module):
    def __init__(self, thresh):
        super().__init__()
        self.thresh = thresh

    def forward(self, pred, target):
        mask = (target > 10 ** -6).detach()
        pred_over_gt, gt_over_pred = pred[mask] / target[mask], target[mask] / pred[mask]
        miss_map = torch.max(pred_over_gt, gt_over_pred)
        total_hit = torch.sum(miss_map < self.thresh).float()
        num_of_valid_pixels = mask.sum()
        return total_hit, num_of_valid_pixels
