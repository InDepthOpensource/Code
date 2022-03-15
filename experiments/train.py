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
    SAMSUNG_TOF_TRAIN_FILE_LIST, SAMSUNG_TOF_EVAL_FILE_LIST, MatterportTrainAddedNoise, MatterportEvalAddedNoise, \
    MATTERPORT_TRAIN_FILE_LIST_NORMAL, MATTERPORT_TEST_FILE_LIST_NORMAL, CombinedTrainDataset

from utils import L1, AverageMeter, BerHuLoss, RMSE, GradientLoss, NormalFromDepthLoss, FinalTrainingLoss, VNLLoss


def run_eval(model, dataloader):
    model.eval()
    total_l1 = 0.0
    total_mse = 0.0
    total_valid_pixel = 0.0
    epoch_gradient_loss = AverageMeter()

    with tqdm(total=len(dataloader.dataset)) as t:
        t.set_description('Testing')

        for data in dataloader:
            color_input, depth_input, target, _ = data

            color_input = color_input.to(device)
            depth_input = depth_input.to(device)
            target = target.to(device)

            with torch.no_grad():
                pred = (model(color_input, depth_input) * 16.0).clamp(0.0, (2 ** 16 - 1) / 4000)
                batch_mse, batch_num_of_valid_pixels = [i.item() for i in rmse_loss(pred, target)]
                batch_l1, _ = [i.item() for i in l1_loss(pred, target)]
                batch_gradient_loss = gradient_loss(pred, target).item()

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
    parser.add_argument('--train-rgb-encoder-after', type=int, default=12)
    parser.add_argument('--use-hybrid-loss-after', type=int, default=14)
    parser.add_argument('--model-type', type=str, default='cbam6')
    parser.add_argument('--vnl-coeff', type=float, default=0.3)
    parser.add_argument('--gradient-coeff', type=float, default=10.0)
    parser.add_argument('--model-name-suffix', type=str, default='')
    parser.add_argument('--use-file-list', type=str, default='original')
    parser.add_argument('--min-mask', type=int, default=1)
    parser.add_argument('--max-mask', type=int, default=7)
    parser.add_argument('--random-crop', type=bool, default=False)
    parser.add_argument('--shift-rgb', type=bool, default=False)
    parser.add_argument('--test-step', type=int, default=4000)
    parser.add_argument('--train-tof-mask', type=bool, default=False)
    parser.add_argument('--log-compress', type=bool, default=False)

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

    if args.log_compress:
        model.unet_block_2.fuse_block.conv_layers.append(nn.Sequential(
            nn.Conv2d(192, 32, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        model.unet_block_2.fuse_block.conv_layers.append(nn.Sequential(
            nn.Conv2d(192, 32, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        ))

        model.unet_block_4.fuse_block.conv_layers.append(nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        model.unet_block_4.fuse_block.conv_layers.append(nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        ))

        model.unet_block_8.fuse_block.conv_layers.append(nn.Sequential(
            nn.Conv2d(384, 64, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        model.unet_block_8.fuse_block.conv_layers.append(nn.Sequential(
            nn.Conv2d(384, 64, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        ))

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
        train_file_list = MATTERPORT_TRAIN_FILE_LIST_NORMAL
        test_file_list = MATTERPORT_TEST_FILE_LIST_NORMAL

    use_generated_mask = not (args.min_mask == 0 or args.max_mask == 0)

    if args.use_file_list == 'combined':
        train_dataset = CombinedTrainDataset(min_mask=args.min_mask, max_mask=args.max_mask,
                                             random_crop=args.random_crop,
                                             shift_rgb=args.shift_rgb)
        eval_dataset = MatterportEvalDataset(file_list_name=MATTERPORT_TEST_FILE_LIST_NORMAL)
    elif args.use_file_list != 'tofnoise':
        train_dataset = MatterportTrainDataset(file_list_name=train_file_list,
                                               use_generated_mask=use_generated_mask,
                                               min_mask=args.min_mask, max_mask=args.max_mask,
                                               random_crop=args.random_crop,
                                               shift_rgb=args.shift_rgb,
                                               use_samsung_tof_mask=args.train_tof_mask,
                                               simulate_log_compression=args.log_compress)
        eval_dataset = MatterportEvalDataset(file_list_name=test_file_list, simulate_log_compression=args.log_compress)
    else:
        train_dataset = MatterportTrainAddedNoise(file_list_name=MATTERPORT_TRAIN_FILE_LIST_NORMAL,
                                                  use_generated_mask=use_generated_mask,
                                                  min_mask=args.min_mask, max_mask=args.max_mask,
                                                  random_crop=args.random_crop,
                                                  shift_rgb=args.shift_rgb)
        eval_dataset = MatterportEvalAddedNoise(file_list_name=MATTERPORT_TEST_FILE_LIST_NORMAL)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    print('Train set length:', len(train_dataset))
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False)
    print('Eval set length:', len(eval_dataset))

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

    if not (args.use_file_list == 'original' or args.use_file_list == 'combined'):
        eval_dataset_original = MatterportEvalDataset(file_list_name=MATTERPORT_TEST_FILE_LIST_NORMAL)
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

    for epoch in range(args.num_epochs):
        # Disable the training for RGB encoder at the very beginning
        if hasattr(model, 'rgb_encoder'):
            for param in model.rgb_encoder.parameters():
                param.requires_grad = (epoch >= args.train_rgb_encoder_after)

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

                pred = model(color_input, depth_input) * 16.0

                loss = criterion(pred, target)

                batch_loss = loss.item()

                with torch.no_grad():
                    batch_berhu_loss = berhu_loss(pred, target).item()
                    batch_gradient_loss = gradient_loss(pred, target).item()
                    if args.vnl_coeff > 10 ** -6:
                        batch_normal_from_depth_loss = vnl_loss(pred, target).item()
                    else:
                        batch_normal_from_depth_loss = 0.0

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

                    if not (args.use_file_list == 'original' or args.use_file_list == 'combined'):
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

        if not (args.use_file_list == 'original' or args.use_file_list == 'combined'):
            eval_l1 = run_eval(model, eval_dataloader_original)
            if eval_l1 < best_l1_original:
                best_l1_epoch_original = epoch
                best_l1_original = eval_l1
                torch.save(weights,
                           r'../tmp/best_l1_original_eval_' + args.use_file_list + args.model_type + args.model_name_suffix + '.pth')

        print('add best epoch: {}, l1 loss: {:.4f}'.format(best_l1_epoch_filtered, best_l1_filtered))
        print('add best epoch on original set: {}, l1 loss: {:.4f}'.format(best_l1_epoch_original, best_l1_original))

        print('saving latest epoch')
        torch.save(weights, r'../tmp/latest_' + args.use_file_list + args.model_type + args.model_name_suffix + '.pth')
