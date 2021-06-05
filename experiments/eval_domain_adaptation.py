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
    DilatedUNetDepthNormalAlt1, CycleGanGenerator
from experiments.data import MatterportEvalDataset, SceneNetRGBDEvalSet, DepthAdaptationDataset, \
    SAMSUNG_TOF_EVAL_FILE_LIST, MATTERPORT_TRAIN_FILE_LIST
from utils import AverageMeter, BerHuLoss, RMSE, GradientLoss, NormalFromDepthLoss, L1, PercentWithinError, VNLLoss, AvgNormalErrorInDegrees

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-file', type=str, default='')
    # parser.add_argument('--eval-file', type=str, default='data/test.blob')
    # parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--load-A2B', type=str, default='/home/home4/yz322/CBAMDiluted0708/DepthCompletion/tmp/latest_netG_A2B_weights_common_depth_coeff_2.pth')
    parser.add_argument('--load-B2A', type=str, default='/home/home4/yz322/CBAMDiluted0708/DepthCompletion/tmp/latest_netG_B2A_weights_common_depth_coeff_2.pth')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-step', type=int, default=5)
    parser.add_argument('--lr-step-gamma', type=float, default=0.5)
    parser.add_argument('--num-epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3242357)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--model-type', type=str, default='')
    parser.add_argument('--num-resblocks', type=int, default=9)

    args = parser.parse_args()

    batch_size = args.batch_size

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    tof_eval_dataset = DepthAdaptationDataset(file_list_name=SAMSUNG_TOF_EVAL_FILE_LIST,
                                              use_generated_mask=False, depth_scale_factor=1000.0,
                                              random_mirror=False)
    tof_eval_dataloader = DataLoader(dataset=tof_eval_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     drop_last=False)
    print('TOF Eval set length:', len(tof_eval_dataset))

    netG_A2B = CycleGanGenerator(5, 1, n_residual_blocks=args.num_resblocks)
    netG_B2A = CycleGanGenerator(5, 1, n_residual_blocks=args.num_resblocks)

    netG_A2B.to(device)
    netG_B2A.to(device)

    netG_A2B.load_state_dict(torch.load(args.load_A2B))
    print('loaded A2B model')

    netG_B2A.load_state_dict(torch.load(args.load_B2A))
    print('loaded B2A model')

    netG_A2B.eval()
    netG_B2A.eval()

    cmap = cm.get_cmap(name='jet')
    cmap.set_under('w')

    eps = torch.tensor(10 ** -6)

    counter = 0

    for data in tof_eval_dataloader:
        tof_color, tof_depth = data
        with torch.no_grad():
            real_A = torch.cat((tof_color, tof_depth), dim=1)
            real_A = real_A.to(device)
            fake_B = netG_A2B(real_A)

            fake_B = fake_B.cpu().numpy()
            for i in range(len(fake_B)):
                converted_depth = fake_B[i][-1][:][:]
                norm = plt.Normalize(vmin=10 ** -3, vmax=0.4)
                pred_plot = cmap(norm(converted_depth))
                plt.imsave(str(counter) + '_converted_depth.png', pred_plot)
                print('saved adapted')


                original_depth = tof_depth[i][0][:][:]
                norm = plt.Normalize(vmin=10 ** -3, vmax=0.4)
                pred_plot = cmap(norm(original_depth))
                plt.imsave(str(counter) + '_original_depth.png', pred_plot)
                print('saved original')
                print('min', original_depth.min())
                print('max', original_depth.max())
                counter += 1







