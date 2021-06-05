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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from tqdm import tqdm
from math import sqrt

from models import *
from experiments.data import MatterportTrainDataset, MatterportEvalDataset, \
    MATTERPORT_TRAIN_FILE_LIST, MATTERPORT_EVAL_FILE_LIST, \
    FILTERED_110_MATTERPORT_EVAL_FILE_LIST, FILTERED_110_MATTERPORT_TRAIN_FILE_LIST, \
    FILTERED_125_MATTERPORT_EVAL_FILE_LIST, FILTERED_125_MATTERPORT_TRAIN_FILE_LIST, \
    SAMSUNG_TOF_TRAIN_FILE_LIST, SAMSUNG_TOF_EVAL_FILE_LIST

from utils import L1, AverageMeter, BerHuLoss, RMSE, GradientLoss, NormalFromDepthLoss, FinalTrainingLoss, VNLLoss


def run_eval(model, dataloader):
    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    total_valid_pixel = 0.0
    epoch_gradient_loss = AverageMeter()
    epoch_normal_from_depth_loss = AverageMeter()

    with tqdm(total=len(dataloader.dataset)) as t:
        t.set_description('Testing')

        for data in dataloader:
            color_input, depth_input, target, _ = data

            color_input = color_input.to(device)
            depth_input = depth_input.to(device)
            target = target.to(device)

            with torch.no_grad():
                pred = (model(color_input, depth_input)[0] * 16.0).clamp(0.0, (2 ** 16 - 1) / 4000)
                batch_mse, batch_num_of_valid_pixels = [i.item() for i in rmse_loss(pred, target)]
                batch_l1, _ = [i.item() for i in l1_loss(pred, target)]
                batch_gradient_loss = gradient_loss(pred, target).item()
                batch_normal_from_depth_loss = vnl_loss(pred, target).item()

            total_l1 += batch_l1
            total_mse += batch_mse
            total_valid_pixel += batch_num_of_valid_pixels
            epoch_gradient_loss.update(batch_gradient_loss)
            epoch_normal_from_depth_loss.update(batch_normal_from_depth_loss)

            batch_l1 = batch_l1 / batch_num_of_valid_pixels
            batch_rmse = sqrt(batch_mse / batch_num_of_valid_pixels)

            t.set_postfix(batch_rmse_loss='{:.6f}'.format(batch_rmse),
                          batch_l1_loss='{:.6f}'.format(batch_l1))
            t.update(len(color_input))

    model.train()

    eval_l1 = total_l1 / total_valid_pixel
    eval_rmse = sqrt(total_mse / total_valid_pixel)

    print('eval RMSE: {:.4f}'.format(eval_rmse))
    print('eval l1 loss: {:.4f}'.format(eval_l1))
    print('eval gradient loss: {:.4f}'.format(epoch_gradient_loss.avg))
    print('eval normal from depth loss: {:.4f}'.format(epoch_normal_from_depth_loss.avg))

    return eval_l1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-file', type=str, default='')
    # parser.add_argument('--eval-file', type=str, default='data/test.blob')
    # parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--load-model', type=str, default='')
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--lr-step', type=int, default=10)
    parser.add_argument('--lr-step-gamma', type=float, default=0.4)
    parser.add_argument('--lr-reset-epoch', type=int, default=30)
    parser.add_argument('--lr-reset-to', type=float, default=0.0002)
    parser.add_argument('--num-epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=92345792)
    parser.add_argument('--train-prelim-rgb-after', type=int, default=11)
    parser.add_argument('--train-refine-rgb-after', type=int, default=12)
    parser.add_argument('--use-hybrid-loss-after', type=int, default=15)
    parser.add_argument('--model-type', type=str, default='prelim_depth_rgb')
    parser.add_argument('--prelim-coeff', type=float, default=0.1)
    parser.add_argument('--vnl-coeff', type=float, default=0.3)
    parser.add_argument('--gradient-coeff', type=float, default=10.0)
    parser.add_argument('--model-name-suffix', type=str, default='')
    parser.add_argument('--use-file-list', type=str, default='original')
    parser.add_argument('--min-mask', type=int, default=1)
    parser.add_argument('--max-mask', type=int, default=7)
    parser.add_argument('--test-step', type=int, default=1000)

    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    berhu_loss = BerHuLoss()
    full_loss = FinalTrainingLoss(gradient_coeff=args.gradient_coeff, normal_from_depth_coeff=args.vnl_coeff)
    rmse_loss = RMSE()
    l1_loss = L1()
    gradient_loss = GradientLoss()
    vnl_loss = VNLLoss()

    model = None
    if args.model_type == 'prelim_depth_rgb':
        model = StackedHourglass(device, mode='prelim_depth_rgb')
    elif args.model_type == 'prelim_depth_depth':
        model = StackedHourglass(device, mode='prelim_depth_depth')
    elif args.model_type == 'symmetrical':
        model = SymmetricalStackedHourglass(device)
    else:
        raise NotImplemented()

    train_file_list = ''
    test_file_list = ''

    if args.use_file_list == '110':
        train_file_list = FILTERED_110_MATTERPORT_TRAIN_FILE_LIST
        test_file_list = FILTERED_110_MATTERPORT_EVAL_FILE_LIST
    elif args.use_file_list == '125':
        train_file_list = FILTERED_125_MATTERPORT_TRAIN_FILE_LIST
        test_file_list = FILTERED_125_MATTERPORT_EVAL_FILE_LIST
    # NEVER use this option to train! This will train on the eval set, so only use it to verify your code can work!!!
    elif args.use_file_list == 'mock':
        print('Warning: NEVER use this option to train! This will train on the eval set, so only use it to verify '
              'your code can work!!!')
        train_file_list = FILTERED_125_MATTERPORT_EVAL_FILE_LIST
        test_file_list = FILTERED_125_MATTERPORT_EVAL_FILE_LIST
    elif args.use_file_list == 'samsungtof':
        train_file_list = SAMSUNG_TOF_TRAIN_FILE_LIST
        test_file_list = SAMSUNG_TOF_EVAL_FILE_LIST
    elif args.use_file_list == 'original':
        train_file_list = MATTERPORT_TRAIN_FILE_LIST
        test_file_list = MATTERPORT_EVAL_FILE_LIST

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

    use_generated_mask = not (args.min_mask == 0 or args.max_mask == 0)
    train_dataset = MatterportTrainDataset(file_list_name=train_file_list, use_generated_mask=use_generated_mask,
                                           min_mask=args.min_mask, max_mask=args.max_mask)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    print('Train set length:', len(train_dataset))

    eval_dataset = MatterportEvalDataset(file_list_name=test_file_list)
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False)
    print('Eval set length:', len(eval_dataset))

    if args.use_file_list != 'original':
        eval_dataset_original = MatterportEvalDataset(file_list_name=MATTERPORT_EVAL_FILE_LIST)
        eval_dataloader_original = DataLoader(dataset=eval_dataset_original,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              drop_last=False)
        print('Original Eval set length:', len(eval_dataset_original))

    best_l1_epoch_filtered = 0
    best_l1_filtered = 10 ** 5

    best_l1_epoch_original = 0
    best_l1_original = 10 ** 5

    prelim_coeff = args.prelim_coeff

    for epoch in range(args.num_epochs):
        # Disable the training for preliminary depth RGB encoder at the very beginning
        for param in model.preliminary_depth.encoder.parameters():
            param.requires_grad = (epoch >= args.train_prelim_rgb_after)

        # Disable the training for depth refinement RGB encoder at the very beginning
        for param in model.depth_refinement.encoder.parameters():
            param.requires_grad = (epoch >= args.train_refine_rgb_after)

        # Must enable training for conv1 and bn1 in the refinement module.
        if args.model_type == 'prelim_depth_rgb' or args.model_type == 'symmetrical':
            for param in model.depth_refinement.encoder.conv1.parameters():
                param.requires_grad = True
            for param in model.depth_refinement.encoder.bn1.parameters():
                param.requires_grad = True

        if epoch >= args.use_hybrid_loss_after:
            criterion = full_loss
        else:
            criterion = berhu_loss

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

        batch_counter = 0

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('Training epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                color_input, depth_input, target = data

                color_input = color_input.to(device)
                depth_input = depth_input.to(device)
                target = target.to(device)

                final_depth, prelim_depth = model(color_input, depth_input)
                final_depth = final_depth * 16.0
                prelim_depth = prelim_depth * 16.0

                loss = criterion(final_depth, target) + berhu_loss(prelim_depth, target) * prelim_coeff

                batch_loss = loss.item()

                with torch.no_grad():
                    batch_berhu_loss = berhu_loss(final_depth, target).item()
                    batch_gradient_loss = gradient_loss(final_depth, target).item()
                    batch_normal_from_depth_loss = vnl_loss(final_depth, target).item()

                    epoch_berhu_loss.update(batch_berhu_loss)
                    epoch_gradient_loss.update(batch_gradient_loss)
                    epoch_normal_from_depth_loss.update(batch_normal_from_depth_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.update(batch_loss, len(color_input))

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg), batch_loss='{:.6f}'.format(batch_loss))
                t.update(len(color_input))

                batch_counter += 1

                if batch_counter % args.test_step == 0:
                    eval_l1 = run_eval(model, eval_dataloader)
                    if eval_l1 < best_l1_filtered:
                        weights = copy.deepcopy(model.state_dict())
                        best_l1_epoch_filtered = epoch
                        best_l1_filtered = eval_l1
                        torch.save(weights,
                                   r'../tmp/best_l1_' + args.use_file_list + args.model_type + args.model_name_suffix + '.pth')
                        print('add best epoch: {}, l1 loss: {:.4f}'.format(best_l1_epoch_filtered, best_l1_filtered))

                    if args.use_file_list != 'original':
                        eval_l1 = run_eval(model, eval_dataloader_original)
                        if eval_l1 < best_l1_original:
                            weights = copy.deepcopy(model.state_dict())
                            best_l1_epoch_original = epoch
                            best_l1_original = eval_l1
                            torch.save(weights,
                                       r'../tmp/best_l1_original_eval_' + args.use_file_list + args.model_type + args.model_name_suffix + '.pth')
                            print('add best epoch on original set: {}, l1 loss: {:.4f}'.format(best_l1_epoch_original,
                                                                                               best_l1_original))

            # Advance the Learning rate scheduler and decrease the learning rate accordingly.
            scheduler.step()

        print('training berhu loss: {:.4f}'.format(epoch_berhu_loss.avg))
        print('training gradient loss: {:.4f}'.format(epoch_gradient_loss.avg))
        print('training normal from depth loss: {:.4f}'.format(epoch_normal_from_depth_loss.avg))

        weights = copy.deepcopy(model.state_dict())

        eval_l1 = run_eval(model, eval_dataloader)
        if eval_l1 < best_l1_filtered:
            best_l1_epoch_filtered = epoch
            best_l1_filtered = eval_l1
            torch.save(weights,
                       r'../tmp/best_l1_' + args.use_file_list + args.model_type + args.model_name_suffix + '.pth')

        if args.use_file_list != 'original':
            eval_l1 = run_eval(model, eval_dataloader_original)
            if eval_l1 < best_l1_original:
                best_l1_epoch_original = epoch
                best_l1_original = eval_l1
                torch.save(weights,
                           r'../tmp/best_l1_original_eval_' + args.use_file_list + args.model_type + args.model_name_suffix + '.pth')

        print('add best epoch: {}, l1 loss: {:.4f}'.format(best_l1_epoch_filtered, best_l1_filtered))
        print('add best epoch on original set: {}, l1 loss: {:.4f}'.format(best_l1_epoch_original,
                                                                           best_l1_original))
        print('saving latest epoch')
        torch.save(weights,
                   r'../tmp/latest_' + args.use_file_list + args.model_type + args.model_name_suffix + '.pth')
