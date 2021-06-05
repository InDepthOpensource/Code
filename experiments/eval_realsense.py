import argparse
import os
import sys
import time
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch.utils.data
import torch.backends.cudnn as cudnn
from math import sqrt
from torch import nn
from torch.utils.data.dataloader import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from models import ResNet34UnetAttention, CBAMDilatedUNet
from experiments.data import RealSenseEvalDataset
from utils import AverageMeter, BerHuLoss, RMSE, GradientLoss, NormalFromDepthLoss, L1, PercentWithinError, VNLLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-file', type=str, default='')
    # parser.add_argument('--eval-file', type=str, default='data/test.blob')
    # parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--load-model', type=str, default='')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-step', type=int, default=5)
    parser.add_argument('--lr-step-gamma', type=float, default=0.5)
    parser.add_argument('--num-epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3242357)
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()

    batch_size = args.batch_size

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = CBAMDilatedUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=False, use_cbam_decoder=False)
    criterion = BerHuLoss()
    rmse_loss = RMSE()
    l1_loss = L1()
    gradient_loss = GradientLoss()
    vnl_loss = VNLLoss()
    normal_from_depth_loss = NormalFromDepthLoss()
    within_105_metric = PercentWithinError(1.05)
    within_110_metric = PercentWithinError(1.10)
    within_125_metric = PercentWithinError(1.25)

    model.to(device)
    criterion.to(device)
    rmse_loss.to(device)
    l1_loss.to(device)
    gradient_loss.to(device)
    normal_from_depth_loss.to(device)
    within_105_metric.to(device)
    within_110_metric.to(device)
    within_125_metric.to(device)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        print('loaded model')

    eval_dataset = RealSenseEvalDataset()
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False)

    print('Eval set length:', len(eval_dataset))

    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    total_within_105 = 0.0
    total_within_110 = 0.0
    total_within_125 = 0.0
    total_valid_pixel = 0.0
    epoch_berhu_loss = AverageMeter()
    epoch_gradient_loss = AverageMeter()
    epoch_normal_from_depth_loss = AverageMeter()
    avg_timer = AverageMeter()
    epoch_vnl_loss = AverageMeter()

    counter = 0

    cmap = cm.get_cmap(name='jet')
    cmap.set_under('w')

    for data in eval_dataloader:
        color_input, depth_input, target, _ = data

        start = time.time()

        color_input = color_input.to(device)
        depth_input = depth_input.to(device)
        target = target.to(device)

        with torch.no_grad():
            pred = (model(color_input, depth_input) * 16.0).clamp(0.0, (2 ** 16 - 1) / 1000)
            batch_berhu_loss = criterion(pred, target).item()
            batch_vnl_loss = vnl_loss(pred, target).item()

            error = (target - pred).abs()
            avg_timer.update(time.time() - start)

            batch_gradient_loss = gradient_loss(pred, target).item()
            batch_normal_from_depth_loss = normal_from_depth_loss(pred, target).item()

            batch_mse, batch_num_of_valid_pixels = [i.item() for i in rmse_loss(pred, target)]
            batch_l1, _ = [i.item() for i in l1_loss(pred, target)]
            batch_within_105, _ = [i.item() for i in within_105_metric(pred, target)]
            batch_within_110, _ = [i.item() for i in within_110_metric(pred, target)]
            batch_within_125, _ = [i.item() for i in within_125_metric(pred, target)]

            if args.save:
                pred, error, depth_input, target = \
                    pred.cpu().numpy(), error.cpu().numpy(), depth_input.cpu().numpy() * 16, target.cpu().numpy()

                for i in range(len(pred)):
                    single_pred = pred[i][0][:][:]
                    single_depth_input = depth_input[i][0][:][:]
                    single_target = target[i][0][:][:]
                    single_error = error[i][0][:][:]

                    imageio.imsave(str(counter) + '_predicted_depth_z16.png', (single_pred * 1000).astype(np.uint16))

                    norm = plt.Normalize(vmin=0.19, vmax=single_pred.max())
                    pred_plot = cmap(norm(single_pred))
                    depth_input_plot = cmap(norm(single_depth_input))
                    target_plot = cmap(norm(single_target))
                    error_plot = cmap(norm(single_error))

                    plt.imsave(str(counter) + '_predicted_depth.png', pred_plot)
                    plt.imsave(str(counter) + '_raw_input_depth.png', depth_input_plot)
                    plt.imsave(str(counter) + '_true_depth.png', target_plot)
                    plt.imsave(str(counter) + '_depth_error.png', error_plot)

                    counter += 1

        total_l1 += batch_l1
        total_mse += batch_mse
        total_within_105 += batch_within_105
        total_within_110 += batch_within_110
        total_within_125 += batch_within_125
        total_valid_pixel += batch_num_of_valid_pixels
        epoch_berhu_loss.update(batch_berhu_loss, len(color_input))
        epoch_gradient_loss.update(batch_gradient_loss, len(color_input))
        epoch_vnl_loss.update(batch_vnl_loss, len(color_input))
        epoch_normal_from_depth_loss.update(batch_normal_from_depth_loss, len(color_input))

    print('eval RMSE: {:.4f}'.format(sqrt(total_mse / total_valid_pixel)))
    print('eval l1 loss: {:.4f}'.format(total_l1 / total_valid_pixel))
    print('eval berhu loss: {:.4f}'.format(epoch_berhu_loss.avg))
    print('eval VNL loss: {:.4f}'.format(epoch_vnl_loss.avg))
    print('eval gradient loss: {:.4f}'.format(epoch_gradient_loss.avg))
    print('eval normal from depth loss: {:.4f}'.format(epoch_normal_from_depth_loss.avg))
    print('eval average fps: {:.4f}'.format(args.batch_size / avg_timer.avg))
    print('eval percent within +/- 1.05: {:.4f}'.format(total_within_105 / total_valid_pixel))
    print('eval percent within +/- 1.10: {:.4f}'.format(total_within_110 / total_valid_pixel))
    print('eval percent within +/- 1.25: {:.4f}'.format(total_within_125 / total_valid_pixel))

    print('eval RMSE in arxiv paper: {:.4f}'.format(
        sqrt(total_mse / (depth_input.shape[-1] * depth_input.shape[-2] * len(eval_dataset)))))
    print('eval l1 loss in arxiv paper: {:.4f}'.format(
        total_l1 / (depth_input.shape[-1] * depth_input.shape[-2] * len(eval_dataset))))
