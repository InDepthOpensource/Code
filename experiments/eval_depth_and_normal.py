# python3 eval_models.py --model-type=cbam6 --load-model=../tmp/sngan_best_l1_cbam6ralsgan.pth --batch-size=16 --seed=3249587
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

from models import CBAMDilatedUNet, CBAMUNet, DilatedUNetDepthNormal, DilatedUNetDepthNormalAlt0, \
    DilatedUNetDepthNormalAlt1
from experiments.data import MatterportEvalDataset, SceneNetRGBDEvalSet, SamsungToFEvalDataset
from utils import AverageMeter, BerHuLoss, RMSE, GradientLoss, NormalFromDepthLoss, L1, PercentWithinError, VNLLoss, \
    AvgNormalErrorInDegrees, NormalErrorInDegrees

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
    parser.add_argument('--model-type', type=str, default='DepthNormal')
    parser.add_argument('--eval-dataset', type=str, default='MatterportEvalDataset')
    parser.add_argument('--bm3d-sigma', type=float, default=0.0)

    args = parser.parse_args()

    batch_size = args.batch_size

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    if args.model_type == 'DepthNormal':
        model = DilatedUNetDepthNormal(device)
    elif args.model_type == 'alt0':
        model = DilatedUNetDepthNormalAlt0(device)
    elif args.model_type == 'alt1':
        model = DilatedUNetDepthNormalAlt1(device)

    criterion = BerHuLoss()
    rmse_loss = RMSE()
    l1_loss = L1()
    gradient_loss = GradientLoss()
    vnl_loss = VNLLoss()
    normal_from_depth_loss = NormalFromDepthLoss()
    within_105_metric = PercentWithinError(1.05)
    within_110_metric = PercentWithinError(1.10)
    within_125_metric = PercentWithinError(1.25)
    measure_avg_normal_error_degrees = AvgNormalErrorInDegrees()
    measure_normal_error_degrees = NormalErrorInDegrees()


    model.to(device)
    criterion.to(device)
    rmse_loss.to(device)
    l1_loss.to(device)
    gradient_loss.to(device)
    normal_from_depth_loss.to(device)
    within_105_metric.to(device)
    within_110_metric.to(device)
    within_125_metric.to(device)
    measure_avg_normal_error_degrees.to(device)
    measure_normal_error_degrees.to(device)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        print('loaded model')

    eval_dataset = None
    if args.eval_dataset == 'MatterportEvalDataset':
        eval_dataset = MatterportEvalDataset(use_normal=True)
    elif args.eval_dataset == 'SceneNetRGBDEvalSet':
        eval_dataset = SceneNetRGBDEvalSet()
    elif args.eval_dataset == 'SamsungToFEvalSet':
        eval_dataset = SamsungToFEvalDataset(bm3d_sigma=args.bm3d_sigma, use_normal=True)
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
    total_valid_pixel = 0.0
    epoch_berhu_loss = AverageMeter()
    epoch_gradient_loss = AverageMeter()
    epoch_normal_from_depth_loss = AverageMeter()
    avg_timer = AverageMeter()
    epoch_vnl_loss = AverageMeter()
    total_error_degrees = 0.0
    total_normal_valid_pixels = 0.0

    counter = 0

    cmap = cm.get_cmap(name='jet')
    cmap.set_under('w')

    test_set_total_normal_error = 0
    test_set_normal_within_11_25 = 0
    test_set_normal_within_22_5 = 0
    test_set_normal_within_30 = 0
    test_set_total_normal_valid_count = 0

    for data in eval_dataloader:
        color_input, depth_input, target, original_color_input, normal_target = data

        start = time.time()

        color_input = color_input.to(device)
        depth_input = depth_input.to(device)
        target = target.to(device)
        normal_target = normal_target.to(device)

        with torch.no_grad():
            pred_depth, pred_normal = model(color_input, depth_input)
            # Scale the outputs.
            pred = (pred_depth * 16.0).clamp(0.0, (2 ** 16 - 1) / 4000)
            pred_normal = (2 * (pred_normal - 0.5)).clamp(-1, 1)

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

            batch_avg_normal_error_degrees, batch_normal_num_of_valid_pixels = measure_avg_normal_error_degrees(pred_normal, normal_target)
            batch_normal_error_degrees, _ = measure_normal_error_degrees(pred_normal, normal_target)
            test_set_total_normal_valid_count += batch_normal_num_of_valid_pixels
            test_set_total_normal_error += batch_normal_error_degrees.sum()
            test_set_normal_within_11_25 += batch_normal_error_degrees[batch_normal_error_degrees < 11.25].numel()
            test_set_normal_within_22_5 += batch_normal_error_degrees[batch_normal_error_degrees < 22.5].numel()
            test_set_normal_within_30 += batch_normal_error_degrees[batch_normal_error_degrees < 30].numel()

            if args.save:
                pred, error, depth_input, target = \
                    pred.cpu().numpy(), error.cpu().numpy(), depth_input.cpu().numpy() * 16, target.cpu().numpy()
                original_color_input = original_color_input.cpu().numpy() * 255
                normal_target = (normal_target.cpu().numpy() + 1.0) / 2.0 * 255
                pred_normal = (pred_normal.cpu().numpy() + 1.0) / 2.0 * 255

                error[target < 10 ** -6] = 0.0

                for i in range(len(pred)):
                    single_pred = pred[i][0][:][:]
                    single_depth_input = depth_input[i][0][:][:]
                    single_target = target[i][0][:][:]
                    single_error = error[i][0][:][:]
                    single_original_color_input = original_color_input[i][:][:][:]
                    single_normal_target = normal_target[i][:][:][:]
                    single_pred_normal = pred_normal[i][:][:][:]

                    norm = plt.Normalize(vmin=10 ** -6, vmax=max(single_target.max(), 0.1))

                    pred_plot = cmap(norm(single_pred))
                    depth_input_plot = cmap(norm(single_depth_input))
                    target_plot = cmap(norm(single_target))
                    error_plot = cmap(norm(single_error))

                    plt.imsave(str(counter) + '_predicted_depth.png', pred_plot)
                    plt.imsave(str(counter) + '_raw_input_depth.png', depth_input_plot)
                    plt.imsave(str(counter) + '_true_depth.png', target_plot)
                    plt.imsave(str(counter) + '_depth_error.png', error_plot)

                    single_original_color_input = np.moveaxis(single_original_color_input, 0, -1)
                    imageio.imsave(str(counter) + '_rgb.jpg', single_original_color_input.astype(np.uint8))
                    single_normal_target = np.moveaxis(single_normal_target, 0, -1)
                    imageio.imsave(str(counter) + '_normal_target.png', single_normal_target.astype(np.uint8))
                    single_pred_normal = np.moveaxis(single_pred_normal, 0, -1)
                    imageio.imsave(str(counter) + '_normal_pred.png', single_pred_normal.astype(np.uint8))
                    imageio.imsave(str(counter) + '_original_depth_z16.png', (single_depth_input * 1000).astype(np.uint16))
                    imageio.imsave(str(counter) + '_predicted_depth_z16.png', (single_pred * 1000).astype(np.uint16))

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

        total_error_degrees += batch_avg_normal_error_degrees
        total_normal_valid_pixels += batch_normal_num_of_valid_pixels

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
    print('eval average normal error in degrees: {:.4f}'.format(total_error_degrees / total_normal_valid_pixels))
    print('eval average normal error in degrees: {:.4f}'.format(test_set_total_normal_error / test_set_total_normal_valid_count))
    print('within 11.25', test_set_normal_within_11_25 / test_set_total_normal_valid_count)
    print('within 22.5', test_set_normal_within_22_5 / test_set_total_normal_valid_count)
    print('within 30', test_set_normal_within_22_5 / test_set_total_normal_valid_count)

    print('eval RMSE in arxiv paper: {:.4f}'.format(
        sqrt(total_mse / (depth_input.shape[-1] * depth_input.shape[-2] * len(eval_dataset)))))
    print('eval l1 loss in arxiv paper: {:.4f}'.format(
        total_l1 / (depth_input.shape[-1] * depth_input.shape[-2] * len(eval_dataset))))
