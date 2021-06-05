import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch.utils.data
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import ResNet34UnetAttention, ResNet34Unet
from experiments.data import MatterportEvalDataset, MatterportTrainDataset, NYUV2TrainDataset
from utils import AverageMeter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-file', type=str, default='')
    # parser.add_argument('--eval-file', type=str, default='data/test.blob')
    # parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--train-dataset', type=str, default='matterport')

    args = parser.parse_args()

    batch_size = args.batch_size

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_dataset = MatterportEvalDataset()
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False)

    if args.train_dataset == 'matterport':
        train_dataset = MatterportTrainDataset(use_generated_mask=False)
    elif args.train_dataset == 'nyuv2':
        train_dataset = NYUV2TrainDataset(use_generated_mask=False)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    print('Eval set length:', len(eval_dataset))

    valid_percent = AverageMeter()

    with tqdm(total=(len(eval_dataset) - len(eval_dataset) % args.batch_size)) as t:
        t.set_description('Testing')

        for data in eval_dataloader:
            color_input, depth_input, target, _ = data

            color_input = color_input.to(device)
            depth_input = depth_input.to(device)
            target = target.to(device)

            with torch.no_grad():
                valid = (depth_input < 10 ** -6).float().sum().item()
                total = depth_input.shape[0] * depth_input.shape[1] * depth_input.shape[2] * depth_input.shape[3]
                batch_percent = valid / total

            t.set_postfix(batch_percent='{:.6f}'.format(batch_percent))
            valid_percent.update(valid / total, total)
            t.update(len(color_input))

    print('eval missing percent: {:.4f}'.format(valid_percent.avg))

    valid_percent = AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
        t.set_description('Testing')

        for data in train_dataloader:
            color_input, depth_input, target = data

            with torch.no_grad():
                valid = (depth_input < 10 ** -6).float().sum().item()
                total = depth_input.shape[0] * depth_input.shape[1] * depth_input.shape[2] * depth_input.shape[3]
                batch_percent = valid / total

            t.set_postfix(batch_percent='{:.6f}'.format(batch_percent), batch_loss='{:.6f}'.format(valid_percent.avg))
            valid_percent.update(valid / total, total)
            t.update(len(color_input))

    print('training missing percent: {:.4f}'.format(valid_percent.avg))
