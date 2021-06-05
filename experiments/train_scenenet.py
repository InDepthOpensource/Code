import argparse
import copy
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from math import sqrt

from models import *
from experiments.data import SceneNetRGBDTrainSet, SceneNetRGBDEvalSet
from utils import L1, AverageMeter, BerHuLoss, RMSE, GradientLoss, NormalFromDepthLoss, FinalTrainingLoss, VNLLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-file', type=str, default='')
    # parser.add_argument('--eval-file', type=str, default='data/test.blob')
    # parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--load-model', type=str, default='')
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--lr-step', type=int, default=10)
    parser.add_argument('--lr-step-gamma', type=float, default=0.3)

    # This is the number of "virtual epochs",
    # i.e. how many times we cycle through the a subset of virtual_epoch_len training images.
    parser.add_argument('--num-epochs', type=int, default=50)

    # How often we will run the validation set and step for learning rate scheduler.
    # Other training parameters, such as when to start training RGB encoder, when to start using hybrid loss,
    # also uses virtual epoch.
    # Unit is # of samples.
    parser.add_argument('--virtual-epoch-len', type=int, default=100 * 1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3242357)
    parser.add_argument('--train-rgb-encoder-after', type=int, default=12)
    parser.add_argument('--use-hybrid-loss-after', type=int, default=16)
    parser.add_argument('--model-type', type=str, default='')
    parser.add_argument('--vnl-coeff', type=float, default=0.3)
    parser.add_argument('--gradient-coeff', type=float, default=10.0)
    parser.add_argument('--model-name-suffix', type=str, default='')

    args = parser.parse_args()

    batch_size = args.batch_size

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    berhu_loss = BerHuLoss()
    full_loss = FinalTrainingLoss(gradient_coeff=args.gradient_coeff, normal_from_depth_coeff=args.vnl_coeff)
    rmse_loss = RMSE()
    l1_loss = L1()
    gradient_loss = GradientLoss()
    vnl_loss = VNLLoss(focal_x=295.60, focal_y=309.02)

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
    elif args.model_type == 'cbam10':
        model = CBAMUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=False, use_cbam_decoder=False)
    elif args.model_type == 'cbam11':
        model = CBAMUNet(device, 3, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam12':
        model = CBAMDilatedUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam13':
        model = CBAMUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam14':
        model = CBAMUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=False, use_cbam_decoder=False)
    else:
        raise NotImplemented()

    model.to(device)
    berhu_loss.to(device)
    full_loss.to(device)
    rmse_loss.to(device)
    l1_loss.to(device)
    gradient_loss.to(device)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        print('loaded model')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)

    train_dataset = SceneNetRGBDTrainSet()
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataset = SceneNetRGBDEvalSet()
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=True)

    best_rmse_epoch = 0
    best_rmse = 10 ** 5
    best_l1_epoch = 0
    best_l1 = 10 ** 5

    print('Train set length:', len(train_dataset))
    print('Eval set length:', len(eval_dataset))

    dataloader_iterator = iter(train_dataloader)

    for epoch in range(args.num_epochs):
        # Disable the training for RGB encoder at the very beginning
        for param in model.rgb_encoder.parameters():
            param.requires_grad = (epoch >= args.train_rgb_encoder_after)

        if epoch >= args.use_hybrid_loss_after:
            criterion = full_loss
        else:
            criterion = berhu_loss

        # Training
        model.train()
        epoch_losses = AverageMeter()
        epoch_berhu_loss = AverageMeter()
        epoch_gradient_loss = AverageMeter()
        epoch_normal_from_depth_loss = AverageMeter()
        epoch_start_time = time.time()

        for _ in range(args.virtual_epoch_len // args.batch_size):
            try:
                data = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_dataloader)
                data = next(dataloader_iterator)

            color_input, depth_input, target = data
            color_input = color_input.to(device)
            depth_input = depth_input.to(device)
            target = target.to(device)

            pred = model(color_input, depth_input) * 16.0

            loss = criterion(pred, target)

            batch_loss = loss.item()

            with torch.no_grad():
                batch_berhu_loss = berhu_loss(pred, target).item()
                batch_gradient_loss = gradient_loss(pred, target).item()
                batch_normal_from_depth_loss = vnl_loss(pred, target).item()

                epoch_berhu_loss.update(batch_berhu_loss)
                epoch_gradient_loss.update(batch_gradient_loss)
                epoch_normal_from_depth_loss.update(batch_normal_from_depth_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.update(batch_loss, len(color_input))

        # Advance the Learning rate scheduler and decrease the learning rate accordingly.
        scheduler.step()

        print('current epoch:', epoch)
        print('overall epoch loss: {:.4f}'.format(epoch_losses.avg))
        print('training berhu loss: {:.4f}'.format(epoch_berhu_loss.avg))
        print('training gradient loss: {:.4f}'.format(epoch_gradient_loss.avg))
        print('training normal from depth loss: {:.4f}'.format(epoch_normal_from_depth_loss.avg))
        print('average training fps: {:.4f}', args.virtual_epoch_len / (time.time() - epoch_start_time))

        # Run evaluation.
        model.eval()
        total_l1 = 0.0
        total_mse = 0.0
        total_valid_pixel = 0.0
        epoch_berhu_loss = AverageMeter()
        epoch_gradient_loss = AverageMeter()
        epoch_normal_from_depth_loss = AverageMeter()
        eval_start_time = time.time()

        for data in eval_dataloader:
            color_input, depth_input, target, _ = data

            color_input = color_input.to(device)
            depth_input = depth_input.to(device)
            target = target.to(device)

            with torch.no_grad():
                pred = (model(color_input, depth_input) * 16.0).clamp(0.0, (2 ** 16 - 1) / 1000)
                batch_mse, batch_num_of_valid_pixels = [i.item() for i in rmse_loss(pred, target)]
                batch_l1, _ = [i.item() for i in l1_loss(pred, target)]
                batch_gradient_loss = gradient_loss(pred, target).item()
                batch_normal_from_depth_loss = vnl_loss(pred, target).item()

            total_l1 += batch_l1
            total_mse += batch_mse
            total_valid_pixel += batch_num_of_valid_pixels
            epoch_gradient_loss.update(batch_gradient_loss)
            epoch_normal_from_depth_loss.update(batch_normal_from_depth_loss)

        eval_l1 = total_l1 / total_valid_pixel
        eval_rmse = sqrt(total_mse / total_valid_pixel)

        print('eval RMSE: {:.4f}'.format(eval_rmse))
        print('eval l1 loss: {:.4f}'.format(eval_l1))
        print('eval gradient loss: {:.4f}'.format(epoch_gradient_loss.avg))
        print('eval normal from depth loss: {:.4f}'.format(epoch_normal_from_depth_loss.avg))
        print('average eval fps: : {:.4f}', len(eval_dataset) / (time.time() - eval_start_time))

        weights = copy.deepcopy(model.state_dict())

        # if eval_rmse < best_rmse:
        #     best_rmse_epoch = epoch
        #     best_rmse = eval_rmse
        #     torch.save(weights, r'../tmp/best_rmse_' + args.model_type + '.pth')

        if eval_l1 < best_l1:
            best_l1_epoch = epoch
            best_l1 = eval_l1
            torch.save(weights, r'../tmp/best_l1_' + args.model_type + args.model_name_suffix + '_scenenet.pth')

        print('saving latest epoch')
        torch.save(weights, r'../tmp/latest_' + args.model_type + args.model_name_suffix + '_scenenet.pth')

        print('add best epoch: {}, l1 loss: {:.4f}'.format(best_l1_epoch, best_l1))
        print('-' * 40)
