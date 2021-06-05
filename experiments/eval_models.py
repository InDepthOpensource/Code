# python3 eval_models.py --model-type=cbam6 --load-model=../tmp/sngan_best_l1_cbam6ralsgan.pth --batch-size=16 --seed=3249587
import argparse
import os
import sys
import time
import copy
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

from models import CBAMDilatedUNet, CBAMUNet, DilatedUNetOneBranch, UNetOneBranch, MobileNetUNet, EfficientUNet
from experiments.data import MatterportEvalDataset, SceneNetRGBDEvalSet, SamsungToFEvalDataset
from utils import AverageMeter, BerHuLoss, RMSE, GradientLoss, NormalFromDepthLoss, L1, PercentWithinError, VNLLoss, \
    SSIM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-file', type=str, default='')
    # parser.add_argument('--eval-file', type=str, default='data/test.blob')
    # parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--load-model', type=str, default='../tmp/best_l1.pth')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3242357)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--model-type', type=str, default='cbam6')
    parser.add_argument('--eval-dataset', type=str, default='MatterportEvalDataset')
    parser.add_argument('--bm3d-sigma', type=float, default=0.0)
    parser.add_argument('--gaussian-noise-sigma', type=float, default=0.0)

    args = parser.parse_args()

    batch_size = args.batch_size

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = None
    if args.model_type == 'cbam0':
        model = CBAMDilatedUNet(device, 3, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=True)
    elif args.model_type == 'cbam1':
        model = CBAMDilatedUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=True)
    elif args.model_type == 'cbam2':
        model = CBAMDilatedUNet(device, 3, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam3':
        model = CBAMDilatedUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam4':
        model = CBAMDilatedUNet(device, 2, use_resblock=False, use_cbam_encoder=True, use_cbam_fuse=True,
                                use_cbam_decoder=False)
    elif args.model_type == 'cbam5':
        model = CBAMDilatedUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=False, use_cbam_decoder=False)
    elif args.model_type == 'cbam6':
        model = CBAMDilatedUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=False, use_cbam_decoder=False)
    elif args.model_type == 'cbam7':
        model = CBAMUNet(device, 3, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=True)
    elif args.model_type == 'cbam8':
        model = CBAMUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=True)
    elif args.model_type == 'cbam9':
        model = CBAMUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam9':
        model = CBAMUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam10':
        model = CBAMUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=False, use_cbam_decoder=False)
    elif args.model_type == 'cbam11':
        model = CBAMUNet(device, 3, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam12':
        model = CBAMDilatedUNet(device, 3, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam13':
        model = CBAMUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam14':
        model = CBAMUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=False, use_cbam_decoder=False)
    elif args.model_type == 'cbam15':
        model = CBAMUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=False, use_cbam_decoder=False)
    elif args.model_type == 'efficientunet-b0':
        model = EfficientUNet(device)
    elif args.model_type == 'efficientunet-b1':
        model = EfficientUNet(device, efficient_net_type='efficientnet-b1')
    elif args.model_type == 'efficientunet-b2':
        model = EfficientUNet(device, efficient_net_type='efficientnet-b2',
                              decoder_input_lengths=(
                                  (0, 352, 352), (120, 120, 120), (48, 48, 48), (24, 24, 24), (16, 16, 16)),
                              decoder_internal_channels=(120, 48, 24, 16, 16),
                              decoder_output_channels=(120, 48, 24, 16, 16))
    elif args.model_type == 'efficientunet-b3':
        model = EfficientUNet(device, efficient_net_type='efficientnet-b3',
                              decoder_input_lengths=(
                                  (0, 384, 384), (136, 136, 136), (48, 48, 48), (32, 32, 32), (24, 24, 24)),
                              decoder_internal_channels=(136, 48, 32, 24, 24),
                              decoder_output_channels=(136, 48, 32, 24, 24))
    elif args.model_type == 'efficientunet-b4':
        model = EfficientUNet(device, efficient_net_type='efficientnet-b4',
                              decoder_input_lengths=(
                                  (0, 448, 448), (160, 160, 160), (56, 56, 56), (32, 32, 32), (24, 24, 24)),
                              decoder_internal_channels=(160, 56, 32, 24, 24),
                              decoder_output_channels=(160, 56, 32, 24, 24))
    elif args.model_type == 'DilatedUNetOneBranch':
        model = DilatedUNetOneBranch(device=device)
    elif args.model_type == 'UNetOneBranch':
        model = UNetOneBranch(device=device)
    elif args.model_type == 'MobileNetUNet':
        model = MobileNetUNet(device=device)
    else:
        raise NotImplemented()

    criterion = BerHuLoss()
    rmse_loss = RMSE()
    l1_loss = L1()
    gradient_loss = GradientLoss()
    vnl_loss = VNLLoss()
    normal_from_depth_loss = NormalFromDepthLoss()
    within_105_metric = PercentWithinError(1.05)
    within_110_metric = PercentWithinError(1.10)
    within_125_metric = PercentWithinError(1.25)
    within_125_2_metric = PercentWithinError(1.25 ** 2)
    within_125_3_metric = PercentWithinError(1.25 ** 3)

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

    eval_dataset = None
    if args.eval_dataset == 'MatterportEvalDataset':
        eval_dataset = MatterportEvalDataset(gaussian_noise_sigma=args.gaussian_noise_sigma)
    elif args.eval_dataset == 'SceneNetRGBDEvalSet':
        eval_dataset = SceneNetRGBDEvalSet()
    elif args.eval_dataset == 'SamsungToFEvalSet':
        eval_dataset = SamsungToFEvalDataset(bm3d_sigma=args.bm3d_sigma)
    else:
        raise NotImplementedError

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
    total_within_125_2 = 0.0
    total_within_125_3 = 0.0
    total_valid_pixel = 0.0
    epoch_berhu_loss = AverageMeter()
    epoch_gradient_loss = AverageMeter()
    epoch_normal_from_depth_loss = AverageMeter()
    avg_timer = AverageMeter()
    epoch_vnl_loss = AverageMeter()

    total_deviations = 0

    counter = 0

    cmap = copy.copy(cm.get_cmap("jet"))
    cmap.set_under('w')

    for data in eval_dataloader:
        total_batch_time_start = time.time()
        color_input, depth_input, target, original_color_input = data

        start = time.time()

        color_input = color_input.to(device)
        depth_input = depth_input.to(device)
        target = target.to(device)

        with torch.no_grad():
            pred = (model(color_input, depth_input) * 16.0).clamp(0.0, (2 ** 16 - 1) / 4000)

            avg_timer.update(time.time() - start)

            batch_berhu_loss = criterion(pred, target).item()
            batch_vnl_loss = vnl_loss(pred, target).item()

            error = (target - pred).abs()

            batch_gradient_loss = gradient_loss(pred, target).item()
            batch_normal_from_depth_loss = normal_from_depth_loss(pred, target).item()

            batch_mse, batch_num_of_valid_pixels = [i.item() for i in rmse_loss(pred, target)]
            batch_l1, _ = [i.item() for i in l1_loss(pred, target)]
            batch_within_105, _ = [i.item() for i in within_105_metric(pred, target)]
            batch_within_110, _ = [i.item() for i in within_110_metric(pred, target)]
            batch_within_125, _ = [i.item() for i in within_125_metric(pred, target)]
            batch_within_125_2, _ = [i.item() for i in within_125_2_metric(pred, target)]
            batch_within_125_3, _ = [i.item() for i in within_125_3_metric(pred, target)]

            if args.save:
                pred, error, depth_input, target = \
                    pred.cpu().numpy(), error.cpu().numpy(), depth_input.cpu().numpy() * 16, target.cpu().numpy()
                original_color_input = original_color_input.cpu().numpy() * 255

                error[target < 10 ** -6] = 0.0

                for i in range(len(pred)):
                    single_pred = pred[i][0][:][:]
                    single_depth_input = depth_input[i][0][:][:]
                    single_target = target[i][0][:][:]
                    single_error = error[i][0][:][:]
                    single_original_color_input = original_color_input[i][:][:][:]

                    if (single_depth_input[single_depth_input > 0.01].size > 0):
                        norm = plt.Normalize(vmin=max(single_depth_input[single_depth_input > 0.01].min(), 0.0),
                                             vmax=max(0.5, min(single_pred.max(), single_depth_input[single_depth_input > 0.01].max() + 2.0)))
                    else:
                        norm = plt.Normalize(vmin=1.0, vmax=12.0)

                    pred_plot = cmap(norm(single_pred))
                    depth_input_plot = cmap(norm(single_depth_input))
                    target_plot = cmap(norm(single_target))

                    mask = (single_target > 0.01) & (single_pred > 0.01)
                    pred_over_gt, gt_over_pred = single_pred[mask] / single_target[mask], single_target[mask] / single_pred[mask]
                    miss_map = np.maximum(pred_over_gt, gt_over_pred)
                    total_deviations += (miss_map.sum() - mask.sum())

                    plt.imsave(str(counter) + '_predicted_depth.png', pred_plot)
                    plt.imsave(str(counter) + '_raw_input_depth.png', depth_input_plot)
                    plt.imsave(str(counter) + '_true_depth.png', target_plot)

                    norm = plt.Normalize(vmin=0.1, vmax=2.0)
                    error_plot = cmap(norm(single_error))
                    plt.imsave(str(counter) + '_depth_error.png', error_plot)

                    single_original_color_input = np.moveaxis(single_original_color_input, 0, -1)
                    imageio.imsave(str(counter) + '_rgb.jpg', single_original_color_input.astype(np.uint8))
                    imageio.imsave(str(counter) + '_original_depth_z16.png',
                                   (single_depth_input * 1000).astype(np.uint16))
                    imageio.imsave(str(counter) + '_predicted_depth_z16.png', (single_pred * 1000).astype(np.uint16))
                    imageio.imsave(str(counter) + '_gt_depth_z16.png',
                                   (single_target * 1000).astype(np.uint16))
                    imageio.imsave(str(counter) + '_error_depth_z16.png',
                                   (single_error * 1000).astype(np.uint16))
                    counter += 1

                print('Overall batch FPS (with save):', args.batch_size / (time.time() - total_batch_time_start))

        total_l1 += batch_l1
        total_mse += batch_mse
        total_within_105 += batch_within_105
        total_within_110 += batch_within_110
        total_within_125 += batch_within_125
        total_within_125_2 += batch_within_125_2
        total_within_125_3 += batch_within_125_3

        total_valid_pixel += batch_num_of_valid_pixels
        epoch_berhu_loss.update(batch_berhu_loss, len(color_input))
        epoch_gradient_loss.update(batch_gradient_loss, len(color_input))
        epoch_vnl_loss.update(batch_vnl_loss, len(color_input))
        epoch_normal_from_depth_loss.update(batch_normal_from_depth_loss, len(color_input))

    print('eval RMSE: {:.4f}'.format(sqrt(total_mse / total_valid_pixel)))
    print('eval l1 loss: {:.4f}'.format(total_l1 / total_valid_pixel))
    print('eval mean relative error: {:.4f}'.format(total_deviations / total_valid_pixel))
    print('eval berhu loss: {:.4f}'.format(epoch_berhu_loss.avg))
    print('eval VNL loss: {:.4f}'.format(epoch_vnl_loss.avg))
    print('eval gradient loss: {:.4f}'.format(epoch_gradient_loss.avg))
    print('eval normal from depth loss: {:.4f}'.format(epoch_normal_from_depth_loss.avg))
    print('eval average fps: {:.4f}'.format(args.batch_size / avg_timer.avg))
    print('eval percent within +/- 1.05: {:.4f}'.format(total_within_105 / total_valid_pixel))
    print('eval percent within +/- 1.10: {:.4f}'.format(total_within_110 / total_valid_pixel))
    print('eval percent within +/- 1.25: {:.4f}'.format(total_within_125 / total_valid_pixel))
    print('eval percent within +/- 1.25^2: {:.4f}'.format(total_within_125_2 / total_valid_pixel))
    print('eval percent within +/- 1.25^3: {:.4f}'.format(total_within_125_3 / total_valid_pixel))

    print('eval RMSE in Huang prior paper: {:.4f}'.format(
        sqrt(total_mse / (depth_input.shape[-1] * depth_input.shape[-2] * len(eval_dataset)))))
    print('eval l1 loss in Huang prior paper: {:.4f}'.format(
        total_l1 / (depth_input.shape[-1] * depth_input.shape[-2] * len(eval_dataset))))
