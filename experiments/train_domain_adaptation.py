import argparse
import copy
import itertools
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
    SAMSUNG_TOF_TRAIN_FILE_LIST, SAMSUNG_TOF_EVAL_FILE_LIST, DepthAdaptationDataset
from utils.misc import ReplayBuffer, AverageMeter


# Credit: Adapted from https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: add code to load four models.
    parser.add_argument('--load-model', type=str, default='')
    parser.add_argument('--g-lr', type=float, default=0.0003)
    parser.add_argument('--d-lr', type=float, default=0.0001)
    parser.add_argument('--lr-step', type=int, default=65)
    parser.add_argument('--lr-step-gamma', type=float, default=0.4)
    parser.add_argument('--lr-reset-epoch', type=int, default=195)
    parser.add_argument('--g-lr-reset-to', type=float, default=0.00015)
    parser.add_argument('--d-lr-reset-to', type=float, default=0.00005)
    parser.add_argument('--num-epochs', type=int, default=520)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=92345792)
    parser.add_argument('--model-type', type=str, default='')
    parser.add_argument('--model-name-suffix', type=str, default='')
    parser.add_argument('--min-mask', type=int, default=1)
    parser.add_argument('--max-mask', type=int, default=7)
    parser.add_argument('--epoch-sample-size', type=int, default=10000)
    parser.add_argument('--random-crop', type=bool, default=True)
    parser.add_argument('--shift-rgb', type=bool, default=False)
    parser.add_argument('--identity_loss_coeff', type=float, default=5.0)
    parser.add_argument('--cycle_loss_coeff', type=float, default=10.0)
    parser.add_argument('--use-common-depth', type=bool, default=False)
    parser.add_argument('--adapt-rgb', type=bool, default=False)
    parser.add_argument('--num-resblocks', type=int, default=9)

    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    # Domain A: ToF captures.
    use_generated_mask = not (args.min_mask == 0 or args.max_mask == 0)
    tof_train_dataset = DepthAdaptationDataset(file_list_name=SAMSUNG_TOF_TRAIN_FILE_LIST,
                                               use_generated_mask=use_generated_mask,
                                               min_mask=args.min_mask, max_mask=args.max_mask,
                                               depth_scale_factor=1000.0, shift_rgb=args.shift_rgb,
                                               random_crop=args.random_crop)
    tof_train_dataloader = DataLoader(dataset=tof_train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True)
    print('TOF train set length:', len(tof_train_dataset))

    # tof_eval_dataset = DepthAdaptationDataset(file_list_name=SAMSUNG_TOF_EVAL_FILE_LIST,
    #                                           use_generated_mask=False, depth_scale_factor=1000.0,
    #                                           random_mirror=False)
    # tof_eval_dataloader = DataLoader(dataset=tof_eval_dataset,
    #                                  batch_size=args.batch_size,
    #                                  shuffle=False,
    #                                  num_workers=args.num_workers,
    #                                  pin_memory=True,
    #                                  drop_last=False)
    # print('TOF Eval set length:', len(tof_eval_dataset))

    # Domain B: Matterport Dataset
    matterport_train_dataset = DepthAdaptationDataset(file_list_name=MATTERPORT_TRAIN_FILE_LIST,
                                                      use_generated_mask=use_generated_mask,
                                                      min_mask=args.min_mask, max_mask=args.max_mask,
                                                      depth_scale_factor=4000.0, shift_rgb=args.shift_rgb,
                                                      random_crop=args.random_crop)
    matterport_train_dataloader = DataLoader(dataset=matterport_train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             drop_last=True)
    print('MATTERPORT train set length:', len(matterport_train_dataset))

    # matterport_eval_dataset = DepthAdaptationDataset(file_list_name=MATTERPORT_EVAL_FILE_LIST,
    #                                                  use_generated_mask=False, depth_scale_factor=4000.0,
    #                                                  random_mirror=False)
    # matterport_eval_dataloader = DataLoader(dataset=matterport_eval_dataset,
    #                                         batch_size=args.batch_size,
    #                                         shuffle=False,
    #                                         num_workers=args.num_workers,
    #                                         pin_memory=True,
    #                                         drop_last=False)
    # print('MATTERPORT Eval set length:', len(matterport_eval_dataset))

    ###### Definition of variables ######
    # Networks

    # Input channels are RGB + Depth + Depth Mask = 5. Output is depth hence 1. Will pad with RGB input to 4 channels.
    # The input channels for CycleGanGenerator.forward(x) will be 5
    # Convention: [batch_size][RGB, Depth, Depth Mask][height][width]
    if not args.adapt_rgb:
        netG_A2B = CycleGanGenerator(5, 1, n_residual_blocks=args.num_resblocks)
        netG_B2A = CycleGanGenerator(5, 1, n_residual_blocks=args.num_resblocks)
        netD_A = CycleGanPatchDiscriminator(1)
        netD_B = CycleGanPatchDiscriminator(1)
    else:
        netG_A2B = CycleGanGenerator(3, 3, depth_output=False, n_residual_blocks=args.num_resblocks)
        netG_B2A = CycleGanGenerator(3, 3, depth_output=False, n_residual_blocks=args.num_resblocks)
        netD_A = CycleGanPatchDiscriminator(3, discriminate_depth=False)
        netD_B = CycleGanPatchDiscriminator(3, discriminate_depth=False)

    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B.to(device)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=args.g_lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    scheduler_G = StepLR(optimizer_G, step_size=args.lr_step, gamma=args.lr_step_gamma)
    scheduler_D_A = StepLR(optimizer_D_A, step_size=args.lr_step, gamma=args.lr_step_gamma)
    scheduler_D_B = StepLR(optimizer_D_B, step_size=args.lr_step, gamma=args.lr_step_gamma)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    target_real = torch.ones((args.batch_size, 1)).to(device)
    target_fake = torch.zeros((args.batch_size, 1)).to(device)
    identity_loss_coeff = torch.tensor(args.identity_loss_coeff).to(device)
    cycle_loss_coeff = torch.tensor(args.cycle_loss_coeff).to(device)
    eps = torch.tensor(10 ** -6)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    for epoch in range(args.num_epochs):
        # Cycle the learning rate on specified epoch.
        if epoch != 0 and epoch % args.lr_reset_epoch == 0:
            optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                           lr=args.g_lr_reset_to, betas=(0.5, 0.999))
            optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.d_lr_reset_to, betas=(0.5, 0.999))
            optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.d_lr_reset_to, betas=(0.5, 0.999))

            scheduler_G = StepLR(optimizer_G, step_size=args.lr_step, gamma=args.lr_step_gamma)
            scheduler_D_A = StepLR(optimizer_D_A, step_size=args.lr_step, gamma=args.lr_step_gamma)
            scheduler_D_B = StepLR(optimizer_D_B, step_size=args.lr_step, gamma=args.lr_step_gamma)
            print('LR reset')

        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()

        tof_train_dataloader_iterator = iter(tof_train_dataloader)
        matterport_train_dataloader_iterator = iter(matterport_train_dataloader)

        epoch_G_loss = AverageMeter()
        epoch_D_A_loss = AverageMeter()
        epoch_D_B_loss = AverageMeter()
        epoch_cycle_loss = AverageMeter()
        epoch_identity_loss = AverageMeter()
        epoch_percent_valid = AverageMeter()

        with tqdm(total=(args.epoch_sample_size - args.epoch_sample_size % args.batch_size)) as t:
            t.set_description('Training epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for _ in range(args.epoch_sample_size // args.batch_size):
                # Acquiring data
                try:
                    tof_color, tof_depth = next(tof_train_dataloader_iterator)
                except StopIteration:
                    tof_train_dataloader_iterator = iter(tof_train_dataloader)
                    tof_color, tof_depth = next(tof_train_dataloader_iterator)

                try:
                    matterport_color, matterport_depth = next(matterport_train_dataloader_iterator)
                except StopIteration:
                    matterport_train_dataloader_iterator = iter(matterport_train_dataloader)
                    matterport_color, matterport_depth = next(matterport_train_dataloader_iterator)

                percent_valid = 0.0
                if args.use_common_depth:
                    with torch.no_grad():
                        common_depth_mask = (tof_depth > eps) & (matterport_depth > eps)
                        percent_valid = (common_depth_mask.sum() * 1.0 / common_depth_mask.numel()).item()
                        tof_depth = tof_depth * common_depth_mask
                        matterport_depth = matterport_depth * common_depth_mask

                real_A = torch.cat((tof_color, tof_depth), dim=1).to(device)
                real_B = torch.cat((matterport_color, matterport_depth), dim=1).to(device)

                ###### Generators A2B and B2A ######
                optimizer_G.zero_grad()
                # Identity loss
                # G_A2B(B) should equal B if real B is fed
                same_B = netG_A2B(real_B)
                loss_identity_B = criterion_identity(same_B, real_B)
                # G_B2A(A) should equal A if real A is fed
                same_A = netG_B2A(real_A)
                loss_identity_A = criterion_identity(same_A, real_A)

                batch_identity_loss = loss_identity_A.item() + loss_identity_B.item()

                # GAN loss
                fake_B = netG_A2B(real_A)
                pred_fake = netD_B(fake_B)
                loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

                fake_A = netG_B2A(real_B)
                pred_fake = netD_A(fake_A)
                loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

                # Cycle loss
                recovered_A = netG_B2A(fake_B)
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A)

                recovered_B = netG_A2B(fake_A)
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

                batch_cycle_loss = loss_cycle_ABA.item() + loss_cycle_BAB.item()

                # Total loss
                loss_G = (loss_identity_A + loss_identity_B) * identity_loss_coeff \
                         + loss_GAN_A2B + loss_GAN_B2A \
                         + (loss_cycle_ABA + loss_cycle_BAB) * cycle_loss_coeff

                batch_G_loss = loss_G.item()

                loss_G.backward()
                optimizer_G.step()
                ###################################

                ###### Discriminator A ######
                optimizer_D_A.zero_grad()

                # Real loss
                pred_real = netD_A(real_A)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = netD_A(fake_A.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake) * 0.5

                batch_D_A_loss = loss_D_A.item()

                loss_D_A.backward()
                optimizer_D_A.step()
                ###################batch_D_A_loss################

                ###### Discriminator B ######
                optimizer_D_B.zero_grad()

                # Real loss
                pred_real = netD_B(real_B)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake) * 0.5

                batch_D_B_loss = loss_D_B.item()

                loss_D_B.backward()
                optimizer_D_B.step()
                ###################################

                t.set_postfix(batch_G_loss='{:.6f}'.format(batch_G_loss), batch_D_A_loss='{:.6f}'.format(batch_D_A_loss),
                              batch_D_B_loss='{:.6f}'.format(batch_D_B_loss), batch_cycle_loss='{:.6f}'.format(batch_cycle_loss),
                              batch_identity_loss='{:.6f}'.format(batch_identity_loss), percent_valid=percent_valid)
                t.update(args.batch_size)

                epoch_G_loss.update(batch_G_loss)
                epoch_D_A_loss.update(batch_D_A_loss)
                epoch_D_B_loss.update(batch_D_B_loss)
                epoch_identity_loss.update(batch_identity_loss)
                epoch_cycle_loss.update(batch_cycle_loss)
                epoch_percent_valid.update(percent_valid)
        # Advance the Learning rate scheduler and decrease the learning rate accordingly.
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        print('training epoch_G_loss loss: {:.4f}'.format(epoch_G_loss.avg))
        print('training epoch_D_A_loss: {:.4f}'.format(epoch_D_A_loss.avg))
        print('training epoch_D_B_loss: {:.4f}'.format(epoch_D_B_loss.avg))
        print('training epoch_identity_loss: {:.4f}'.format(epoch_identity_loss.avg))
        print('training epoch_cycle_loss: {:.4f}'.format(epoch_cycle_loss.avg))

        netG_A2B.eval()
        netG_B2A.eval()
        netD_A.eval()
        netD_B.eval()
        # TODO: find a way to perform evaluation on the eval set.
        #  This is difficult as we cannot use identity loss/cycle loss directly.

        netG_A2B_weights = copy.deepcopy(netG_A2B.state_dict())
        netG_B2A_weights = copy.deepcopy(netG_B2A.state_dict())
        netD_A_weights = copy.deepcopy(netD_A.state_dict())
        netD_B_weights = copy.deepcopy(netD_B.state_dict())

        torch.save(netG_A2B_weights,
                   r'../tmp/latest_netG_A2B_weights_' + args.model_name_suffix + '.pth')
        torch.save(netG_B2A_weights,
                   r'../tmp/latest_netG_B2A_weights_' + args.model_name_suffix + '.pth')

        torch.save(netD_A_weights,
                   r'../tmp/latest_netD_A_weights_' + args.model_name_suffix + '.pth')
        torch.save(netD_B_weights,
                   r'../tmp/latest_netD_B_weights_' + args.model_name_suffix + '.pth')

