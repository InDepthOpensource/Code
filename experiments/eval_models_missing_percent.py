# python3 eval_models.py --model-type=cbam6 --load-model=../tmp/sngan_best_l1_cbam6ralsgan.pth --batch-size=16 --seed=3249587
import argparse
import random
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
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from models import CBAMDilatedUNet, CBAMUNet
from experiments.data import MatterportEvalDataset, SceneNetRGBDEvalSet, SamsungToFEvalDataset, SAMSUNG_TOF_EVAL_FILE_LIST
from utils import AverageMeter, BerHuLoss, RMSE, GradientLoss, NormalFromDepthLoss, L1, PercentWithinError, VNLLoss, \
    SSIM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-file', type=str, default='')
    # parser.add_argument('--eval-file', type=str, default='data/test.blob')
    # parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--load-model', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3242357)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--model-type', type=str, default='cbam6')
    parser.add_argument('--eval-dataset', type=str, default='MatterportEvalDataset')
    parser.add_argument('--bm3d-sigma', type=float, default=0.0)

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
        model = CBAMDilatedUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam15':
        model = CBAMUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=False, use_cbam_decoder=False)
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
        eval_dataset = MatterportEvalDataset()
    elif args.eval_dataset == 'SceneNetRGBDEvalSet':
        eval_dataset = SceneNetRGBDEvalSet()
    elif args.eval_dataset == 'SamsungToFEvalSet':
        eval_dataset = SamsungToFEvalDataset(bm3d_sigma=args.bm3d_sigma)
    else:
        raise NotImplementedError

    f = open(SAMSUNG_TOF_EVAL_FILE_LIST, 'r')
    samsung_file_list = [i.split()[1] for i in f.readlines()]
    f.close()

    depth_mask_transforms = transforms.Compose((
            transforms.Resize((256, 320), interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ))

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

    counter = 0

    cmap = cm.get_cmap(name='jet')
    cmap.set_under('w')

    error_sum = [0.0 for _ in range(102)]
    pixel_count = [0 for _ in range(102)]


    # for i in random.sample(range(470), 100):
    #     samsung_depth_file = samsung_file_list[i]
    #     samsung_depth_file = Image.open(samsung_depth_file)
    #     samsung_depth_tensor = depth_mask_transforms(samsung_depth_file)

    for data in eval_dataloader:
        total_batch_time_start = time.time()
        color_input, depth_input, target, original_color_input = data

        # samsung_depth_mask = samsung_depth_tensor.repeat(depth_input.shape[0], 1, 1, 1)
        #
        # depth_input[samsung_depth_mask < 0.01] = 0.0

        start = time.time()

        color_input = color_input.to(device)
        depth_input = depth_input.to(device)
        target = target.to(device)

        with torch.no_grad():
            pred = (model(color_input, depth_input) * 16.0).clamp(0.0, (2 ** 16 - 1) / 4000)

            avg_timer.update(time.time() - start)

            error = (target - pred).abs()

            pred, error, depth_input, target = \
                pred.cpu().numpy(), error.cpu().numpy(), depth_input.cpu().numpy() * 16, target.cpu().numpy()
            original_color_input = original_color_input.cpu().numpy() * 255

            error[target < 10 ** -6] = 0.0

            for i in range(len(pred)):
                single_pred = pred[i][0][:][:]
                single_depth_input = depth_input[i][0][:][:]
                single_target = target[i][0][:][:]
                single_error = error[i][0][:][:]

                mask = (single_target > 0.01) & (single_pred > 0.01)
                percent_available = (single_depth_input > 0.01).sum() / (320 * 256)
                percent_available = int(percent_available * 100)
                total_pixels = mask.sum()

                error_sum[percent_available] += np.sum(single_error)
                pixel_count[percent_available] += total_pixels

                print('Overall batch FPS (with save):', args.batch_size / (time.time() - total_batch_time_start))

    print('eval average fps: {:.4f}'.format(args.batch_size / avg_timer.avg))

    error_sum = np.array(error_sum[:100])
    pixel_count = np.array(pixel_count[:100]) + 1

    avg_error_by_percent = error_sum / pixel_count
    print(avg_error_by_percent)
    plt.scatter(np.linspace(0, 1, 100), avg_error_by_percent)
    plt.savefig('error_by_depth_available_percent.png')
    plt.clf()
