import argparse
import copy
import os
import sys

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
from experiments.data import MatterportTrainDataset, MatterportEvalDataset, MATTERPORT_TRAIN_FILE_LIST, \
    MATTERPORT_EVAL_FILE_LIST
from utils import L1, AverageMeter, BerHuLoss, RMSE, GradientLoss, FinalTrainingLoss, VNLLoss, AvgNormalErrorInDegrees, \
    L2NormalLoss, CosineDistanceLoss


def run_eval_l1(model, dataloader):
    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    total_valid_pixel = 0.0
    epoch_gradient_loss = AverageMeter()
    epoch_normal_from_depth_loss = AverageMeter()

    for data in dataloader:
        color_input, depth_input, target, _, _ = data

        color_input = color_input.to(device)
        depth_input = depth_input.to(device)
        target = target.to(device)

        with torch.no_grad():
            pred, _ = model(color_input, depth_input)
            pred = (pred * 16.0).clamp(0.0, (2 ** 16 - 1) / 4000)
            batch_mse, batch_num_of_valid_pixels = [i.item() for i in rmse_loss(pred, target)]
            batch_l1, _ = [i.item() for i in l1_loss(pred, target)]
            batch_gradient_loss = gradient_loss(pred, target).item()
            batch_normal_from_depth_loss = vnl_loss(pred, target).item()

        total_l1 += batch_l1
        total_mse += batch_mse
        total_valid_pixel += batch_num_of_valid_pixels
        epoch_gradient_loss.update(batch_gradient_loss)
        epoch_normal_from_depth_loss.update(batch_normal_from_depth_loss)

    model.train()

    eval_l1 = total_l1 / total_valid_pixel
    eval_rmse = sqrt(total_mse / total_valid_pixel)

    print('eval RMSE: {:.4f}'.format(eval_rmse))
    print('eval l1 loss: {:.4f}'.format(eval_l1))
    print('eval gradient loss: {:.4f}'.format(epoch_gradient_loss.avg))
    print('eval normal from depth loss: {:.4f}'.format(epoch_normal_from_depth_loss.avg))

    return eval_l1


def run_eval_normal(model, dataloader):
    model.eval()
    total_error_degrees = 0.0
    total_valid_pixel = 0.0

    for data in dataloader:
        color_input, depth_input, _, _, normal_target = data

        color_input = color_input.to(device)
        depth_input = depth_input.to(device)
        normal_target = normal_target.to(device)

        with torch.no_grad():
            _, pred_normal = model(color_input, depth_input)
            pred_normal = 2 * (pred_normal - 0.5)
            batch_error_degrees, batch_num_of_valid_pixels = measure_normal_error_degrees(pred_normal, normal_target)
            batch_error_degrees = batch_error_degrees.item()
            total_error_degrees += batch_error_degrees
            total_valid_pixel += batch_num_of_valid_pixels

    model.train()

    epoch_normal_error_degrees = total_error_degrees / total_valid_pixel
    print('average error in degrees is', epoch_normal_error_degrees)

    return epoch_normal_error_degrees


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-depth-model', type=str, default='')
    parser.add_argument('--load-depth-normal-model', type=str, default='')
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--lr-step', type=int, default=10)
    parser.add_argument('--lr-step-gamma', type=float, default=0.4)
    parser.add_argument('--lr-reset-epoch', type=int, default=30)
    parser.add_argument('--lr-reset-to', type=float, default=0.0002)
    parser.add_argument('--num-epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=92345792)
    parser.add_argument('--train-rgb-encoder-after', type=int, default=10)
    parser.add_argument('--use-hybrid-loss-after', type=int, default=12)
    parser.add_argument('--vnl-coeff', type=float, default=0.3)
    parser.add_argument('--gradient-coeff', type=float, default=10.0)
    parser.add_argument('--model-name-suffix', type=str, default='')
    parser.add_argument('--min-mask', type=int, default=1)
    parser.add_argument('--max-mask', type=int, default=7)
    parser.add_argument('--random-crop', type=bool, default=True)
    parser.add_argument('--shift-rgb', type=bool, default=False)
    parser.add_argument('--test-step', type=int, default=20000)
    parser.add_argument('--normal-training-loss', type=str, default='L2')
    parser.add_argument('--train-normal-after', type=int, default=13)
    parser.add_argument('--normal-coeff', type=float, default=0.5)
    parser.add_argument('--model-type', type=str, default='')

    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    berhu_loss = BerHuLoss()
    full_loss = None
    vnl_loss = None
    if torch.cuda.is_available():
        full_loss = FinalTrainingLoss(gradient_coeff=args.gradient_coeff, normal_from_depth_coeff=args.vnl_coeff)
        vnl_loss = VNLLoss()
    else:
        print('Warning: you need cuda to perform actual training. '
              'You can only do mock training for debugging without cuda.'
              'The loss function in this mode might be faked')
        full_loss = BerHuLoss()
        vnl_loss = BerHuLoss()
    rmse_loss = RMSE()
    l1_loss = L1()
    gradient_loss = GradientLoss()

    measure_normal_error_degrees = AvgNormalErrorInDegrees()
    normal_loss = CosineDistanceLoss() if args.normal_training_loss == 'cosine' else L2NormalLoss()

    if args.model_type == 'alt0':
        model = DilatedUNetDepthNormalAlt0(device)
        model.to(device)
        if args.load_depth_normal_model:
            model.load_state_dict(torch.load(args.load_depth_normal_model, map_location=device))
            print('loaded model')
    elif args.model_type == 'alt1':
        model = DilatedUNetDepthNormalAlt1(device)
        model.to(device)
        if args.load_depth_normal_model:
            model.load_state_dict(torch.load(args.load_depth_normal_model, map_location=device))
            print('loaded model')
    else:
        model = DilatedUNetDepthNormal(device, depth_model_weights_path=args.load_depth_model,
                                       full_model_weights_path=args.load_depth_normal_model)
        model.to(device)
    print('loaded model')

    berhu_loss.to(device)
    full_loss.to(device)
    rmse_loss.to(device)
    l1_loss.to(device)
    gradient_loss.to(device)
    measure_normal_error_degrees.to(device)
    normal_loss.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)

    use_generated_mask = not (args.min_mask == 0 or args.max_mask == 0)
    train_dataset = MatterportTrainDataset(use_generated_mask=use_generated_mask,
                                           min_mask=args.min_mask, max_mask=args.max_mask, random_crop=args.random_crop,
                                           shift_rgb=args.shift_rgb, use_normal=True)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    print('Train set length:', len(train_dataset))

    eval_dataset = MatterportEvalDataset(use_normal=True)
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False)
    print('Eval set length:', len(eval_dataset))

    best_l1_epoch = 0
    best_l1 = 10 ** 5
    best_normal_epoch = 0
    best_normal_degrees = 10 ** 5

    for epoch in range(args.num_epochs):
        # Disable the training for RGB encoder at the very beginning
        for param in model.rgb_encoder.parameters():
            param.requires_grad = (epoch >= args.train_rgb_encoder_after)

        model.set_train_normal_decoder(epoch >= args.train_normal_after)

        if epoch >= args.use_hybrid_loss_after:
            criterion = full_loss
        else:
            criterion = berhu_loss

        normal_coeff = 0.0
        if epoch >= args.train_normal_after:
            normal_coeff = args.normal_coeff

        # Cycle the learning rate on specified epoch.
        if epoch != 0 and epoch % args.lr_reset_epoch == 0:
            optimizer = optim.Adam(model.parameters(), lr=args.lr_reset_to)
            scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
            print('LR reset')

        model.train()
        epoch_losses = AverageMeter()
        epoch_berhu_loss = AverageMeter()
        epoch_gradient_loss = AverageMeter()
        epoch_normal_from_depth_loss = AverageMeter()
        epoch_normal_loss = AverageMeter()

        batch_counter = 0

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('Training epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                color_input, depth_input, target_depth, target_normal = data

                color_input = color_input.to(device)
                depth_input = depth_input.to(device)
                target_depth = target_depth.to(device)
                target_normal = target_normal.to(device)

                pred_depth, pred_normal = model(color_input, depth_input)
                # Scale the outputs.
                pred_depth = pred_depth * 16.0
                pred_normal = 2 * (pred_normal - 0.5)

                loss = criterion(pred_depth, target_depth) + normal_coeff * normal_loss(pred_normal, target_normal)
                batch_loss = loss.item()

                with torch.no_grad():
                    batch_berhu_loss = berhu_loss(pred_depth, target_depth).item()
                    batch_gradient_loss = gradient_loss(pred_depth, target_depth).item()
                    batch_normal_from_depth_loss = vnl_loss(pred_depth, target_depth).item()
                    batch_normal_loss = normal_loss(pred_normal, target_normal)

                    epoch_berhu_loss.update(batch_berhu_loss)
                    epoch_gradient_loss.update(batch_gradient_loss)
                    epoch_normal_from_depth_loss.update(batch_normal_from_depth_loss)
                    epoch_normal_loss.update(batch_normal_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.update(batch_loss, len(color_input))

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg), batch_loss='{:.6f}'.format(batch_loss))
                t.update(len(color_input))

                batch_counter += 1

                if batch_counter % (args.test_step // args.batch_size) == 0:
                    eval_l1 = run_eval_l1(model, eval_dataloader)
                    if eval_l1 < best_l1:
                        weights = copy.deepcopy(model.state_dict())
                        best_l1_epoch = epoch
                        best_l1 = eval_l1
                        torch.save(weights,
                                   r'../tmp/best_l1_' + 'normal_multitask' + args.model_name_suffix + '.pth')
                        print('add best epoch: {}, l1 loss: {:.4f}'.format(best_l1_epoch, best_l1))

                    eval_normal_error_degrees = run_eval_normal(model, eval_dataloader)
                    if eval_normal_error_degrees < best_normal_degrees:
                        weights = copy.deepcopy(model.state_dict())
                        best_normal_epoch = epoch
                        best_normal_degrees = eval_normal_error_degrees
                        torch.save(weights,
                                   r'../tmp/best_normal_' + 'normal_multitask' + args.model_name_suffix + '.pth')
                        print('add best epoch: {}, normal error degrees: {:.4f}'.format(best_normal_epoch, best_normal_degrees))

            # Advance the Learning rate scheduler and decrease the learning rate accordingly.
            scheduler.step()

        print('training berhu loss: {:.4f}'.format(epoch_berhu_loss.avg))
        print('training gradient loss: {:.4f}'.format(epoch_gradient_loss.avg))
        print('training normal from depth loss: {:.4f}'.format(epoch_normal_from_depth_loss.avg))
        print('training normal loss: {:.4f}'.format(epoch_normal_loss.avg))


        weights = copy.deepcopy(model.state_dict())

        eval_l1 = run_eval_l1(model, eval_dataloader)
        if eval_l1 < best_l1:
            best_l1_epoch = epoch
            best_l1 = eval_l1
            torch.save(weights,
                       r'../tmp/best_l1_' + 'normal_multitask' + args.model_name_suffix + '.pth')
            print('add best epoch: {}, l1 loss: {:.4f}'.format(best_l1_epoch, best_l1))

        eval_normal_error_degrees = run_eval_normal(model, eval_dataloader)
        if eval_normal_error_degrees < best_normal_degrees:
            weights = copy.deepcopy(model.state_dict())
            best_normal_epoch = epoch
            best_normal_degrees = eval_normal_error_degrees
            torch.save(weights,
                       r'../tmp/best_normal_' + 'normal_multitask' + args.model_name_suffix + '.pth')
            print('add best epoch: {}, normal error degrees: {:.4f}'.format(best_normal_epoch, best_normal_degrees))

        print('saving latest epoch')
        torch.save(weights, r'../tmp/latest_' + 'normal_multitask' + args.model_name_suffix + '.pth')
