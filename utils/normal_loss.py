import torch
import math
import torch.nn as nn
from torch import Tensor, mul, dot, ones
from torch.nn import functional as F
from utils import VNLLoss


class CosineDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('normal_length_threshold', torch.tensor(10 ** -6))
        self.register_buffer('cosine_embedding_loss_label', torch.tensor(1))

    def forward(self, pred_normal, target_normal):
        pred_normal = pred_normal.permute(0, 2, 3, 1)
        target_normal = target_normal.permute(0, 2, 3, 1)

        # Only select normal vectors that have valid corresponding ground truth.
        # Note in the matterport dataset, invalid normals are set to 0 in a 0-65535 range
        # So we cast the ranges of values to 0 - 2 and then select by length.
        valid_ground_truth_mask = torch.norm(target_normal + torch.ones_like(target_normal),
                                             dim=-1) > self.normal_length_threshold

        pred_normal = pred_normal[valid_ground_truth_mask]
        target_normal = target_normal[valid_ground_truth_mask]

        # Now, if some of the predicted normal is too small, we deal with this situation.
        invalid_prediction_mask = torch.norm(pred_normal) < self.normal_length_threshold
        pred_normal[invalid_prediction_mask] += self.normal_length_threshold

        loss = F.cosine_embedding_loss(pred_normal, target_normal, self.cosine_embedding_loss_label)
        return loss


class AvgNormalErrorInDegrees(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('normal_length_threshold', torch.tensor(0.0001))
        self.register_buffer('pi_to_degrees_coeff', torch.tensor(180.0) / torch.tensor(math.pi))

    def forward(self, pred_normal, target_normal):
        pred_normal = pred_normal.permute(0, 2, 3, 1)
        target_normal = target_normal.permute(0, 2, 3, 1)

        # Only select normal vectors that have valid corresponding ground truth.
        # Note in the matterport dataset, invalid normals are set to 0 in a 0-65535 range
        # So we cast the ranges of values to 0 - 2 and then select by length.
        valid_ground_truth_mask = torch.norm(target_normal + torch.ones_like(target_normal),
                                             dim=-1) > self.normal_length_threshold

        pred_normal = pred_normal[valid_ground_truth_mask]
        target_normal = target_normal[valid_ground_truth_mask]

        # Now, if some of the predicted normal is too small, we deal with this situation.
        invalid_prediction_mask = torch.norm(pred_normal) < self.normal_length_threshold
        pred_normal[invalid_prediction_mask] += self.normal_length_threshold

        cosine_distance = F.cosine_similarity(pred_normal, target_normal)

        # Dealing with the range of acos to prevent nan.
        error_degrees = torch.acos(torch.clamp(cosine_distance, -1, 1))

        return error_degrees.abs().sum() * self.pi_to_degrees_coeff, error_degrees.numel()


class NormalErrorInDegrees(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('normal_length_threshold', torch.tensor(0.0001))
        self.register_buffer('pi_to_degrees_coeff', torch.tensor(180.0) / torch.tensor(math.pi))

    def forward(self, pred_normal, target_normal):
        pred_normal = pred_normal.permute(0, 2, 3, 1)
        target_normal = target_normal.permute(0, 2, 3, 1)

        # Only select normal vectors that have valid corresponding ground truth.
        # Note in the matterport dataset, invalid normals are set to 0 in a 0-65535 range
        # So we cast the ranges of values to 0 - 2 and then select by length.
        valid_ground_truth_mask = torch.norm(target_normal + torch.ones_like(target_normal),
                                             dim=-1) > self.normal_length_threshold

        pred_normal = pred_normal[valid_ground_truth_mask]
        target_normal = target_normal[valid_ground_truth_mask]

        # Now, if some of the predicted normal is too small, we deal with this situation.
        invalid_prediction_mask = torch.norm(pred_normal) < self.normal_length_threshold
        pred_normal[invalid_prediction_mask] += self.normal_length_threshold

        cosine_distance = F.cosine_similarity(pred_normal, target_normal)

        # Dealing with the range of acos to prevent nan.
        error_degrees = torch.acos(torch.clamp(cosine_distance, -1, 1))

        return error_degrees.abs() * self.pi_to_degrees_coeff, error_degrees.numel()


class L2NormalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('normal_length_threshold', torch.tensor(0.0001))

    def forward(self, pred_normal, target_normal):
        pred_normal = pred_normal.permute(0, 2, 3, 1)
        target_normal = target_normal.permute(0, 2, 3, 1)

        # Only select normal vectors that have valid corresponding ground truth.
        # Note in the matterport dataset, invalid normals are set to 0 in a 0-65535 range
        # So we cast the ranges of values to 0 - 2 and then select by length.
        valid_ground_truth_mask = torch.norm(target_normal + torch.ones_like(target_normal),
                                             dim=-1) > self.normal_length_threshold

        pred_normal = pred_normal[valid_ground_truth_mask]
        target_normal = target_normal[valid_ground_truth_mask]

        error = F.mse_loss(pred_normal, target_normal, reduction='mean')
        return error
