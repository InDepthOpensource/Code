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
from experiments.data import MatterportTrainDataset, MatterportEvalDataset, \
    MATTERPORT_TRAIN_FILE_LIST, MATTERPORT_EVAL_FILE_LIST, \
    FILTERED_110_MATTERPORT_EVAL_FILE_LIST, FILTERED_110_MATTERPORT_TRAIN_FILE_LIST, \
    FILTERED_125_MATTERPORT_EVAL_FILE_LIST, FILTERED_125_MATTERPORT_TRAIN_FILE_LIST, \
    SAMSUNG_TOF_TRAIN_FILE_LIST, SAMSUNG_TOF_EVAL_FILE_LIST
from utils import L1, AverageMeter, BerHuLoss, RMSE, GradientLoss, VNLLoss, FinalTrainingLoss


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
                pred = (model(color_input, depth_input) * 16.0).clamp(0.0, (2 ** 16 - 1) / 4000)
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
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lr-step', type=int, default=10)
    parser.add_argument('--lr-step-gamma', type=float, default=0.3)
    parser.add_argument('--num-epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=4563793)
    parser.add_argument('--train-rgb-encoder-after', type=int, default=0)
    parser.add_argument('--use-hybrid-loss-after', type=int, default=0)
    parser.add_argument('--model-type', type=str, default='')

    parser.add_argument('--critic-lr', type=float, default=0.0002)
    # Should we pretrain the critic?
    parser.add_argument('--pretrain-critic-iter', type=int, default=5000)
    # How many times should we train critic every time we train the generator?
    # parser.add_argument('--train-critic-iter', type=int, default=1)
    # control over beta1
    parser.add_argument('--generator-beta1', type=float, default=0.9)
    parser.add_argument('--critic-beta1', type=float, default=0.5)

    parser.add_argument('--lr-reset-epoch', type=int, default=30)
    parser.add_argument('--lr-reset-to', type=float, default=0.0001)
    parser.add_argument('--critic-lr-reset-to', type=float, default=0.0001)

    parser.add_argument('--load-critic-model', type=str, default='')
    parser.add_argument('--critic-loss-coeff', type=float, default=0.025)
    parser.add_argument('--vnl-coeff', type=float, default=0.3)
    parser.add_argument('--gradient-coeff', type=float, default=1.0)
    parser.add_argument('--model-name-suffix', type=str, default='')
    parser.add_argument('--gan-loss-type', type=str, default='hinge')

    parser.add_argument('--use-file-list', type=str, default='original')
    parser.add_argument('--min-mask', type=int, default=1)
    parser.add_argument('--max-mask', type=int, default=7)
    parser.add_argument('--random-crop', type=bool, default=False)
    parser.add_argument('--shift-rgb', type=bool, default=False)
    parser.add_argument('--test-step', type=int, default=800)

    args = parser.parse_args()

    print('batch size:', str(args.batch_size))
    print('learning rate', str(args.lr))
    print('model type:', args.model_type)
    print('gan loss type:', args.gan_loss_type)
    print('critic loss coeff:', str(args.critic_loss_coeff))

    batch_size = args.batch_size

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    berhu_loss = BerHuLoss()
    full_loss = FinalTrainingLoss(gradient_coeff=args.gradient_coeff, normal_from_depth_coeff=args.vnl_coeff)
    rmse_loss = RMSE()
    l1_loss = L1()
    gradient_loss = GradientLoss()
    vnl_loss = VNLLoss()
    bce_logits_loss = nn.BCEWithLogitsLoss()

    generator = None
    if args.model_type == 'cbam0':
        generator = CBAMDilatedUNet(device, 3, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=True)
    elif args.model_type == 'cbam1':
        generator = CBAMDilatedUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=True)
    elif args.model_type == 'cbam2':
        generator = CBAMDilatedUNet(device, 3, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam3':
        generator = CBAMDilatedUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam4':
        generator = CBAMDilatedUNet(device, 2, use_resblock=False, use_cbam_encoder=True, use_cbam_fuse=True,
                                    use_cbam_decoder=False)
    elif args.model_type == 'cbam5':
        generator = CBAMDilatedUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=False, use_cbam_decoder=False)
    elif args.model_type == 'cbam6':
        generator = CBAMDilatedUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=False, use_cbam_decoder=False)
    elif args.model_type == 'cbam7':
        generator = CBAMUNet(device, 3, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=True)
    elif args.model_type == 'cbam8':
        generator = CBAMUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=True)
    elif args.model_type == 'cbam9':
        generator = CBAMUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam10':
        generator = CBAMUNet(device, 2, use_cbam_encoder=True, use_cbam_fuse=False, use_cbam_decoder=False)
    elif args.model_type == 'cbam11':
        generator = CBAMUNet(device, 3, use_cbam_encoder=True, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam12':
        generator = CBAMDilatedUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam13':
        generator = CBAMUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=True, use_cbam_decoder=False)
    elif args.model_type == 'cbam14':
        generator = CBAMUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=False, use_cbam_decoder=False)
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

    use_generated_mask = not (args.min_mask == 0 or args.max_mask == 0)

    critic = ModifiedVGGCritic()

    generator.to(device)
    critic.to(device)
    berhu_loss.to(device)
    full_loss.to(device)
    rmse_loss.to(device)
    l1_loss.to(device)
    gradient_loss.to(device)
    vnl_loss.to(device)
    bce_logits_loss.to(device)

    if args.load_model:
        generator.load_state_dict(torch.load(args.load_model))
        print('loaded generator model')

    if args.load_critic_model:
        critic.load_state_dict(torch.load(args.load_critic_model))
        print('loaded critic model')

    generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.generator_beta1, 0.999))
    generator_scheduler = StepLR(generator_optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)

    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta1, 0.999))
    critic_scheduler = StepLR(critic_optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)

    train_dataset = MatterportTrainDataset(file_list_name=train_file_list, use_generated_mask=use_generated_mask,
                                           min_mask=args.min_mask, max_mask=args.max_mask, random_crop=args.random_crop,
                                           shift_rgb=args.shift_rgb)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataset = MatterportEvalDataset(file_list_name=test_file_list)
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False)

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

    print('Train set length:', len(train_dataset))
    print('Eval set length:', len(eval_dataset))

    batch_counter = 0

    for epoch in range(args.num_epochs):
        # Disable the training for RGB encoder at the very beginning
        for param in generator.rgb_encoder.parameters():
            param.requires_grad = (epoch >= args.train_rgb_encoder_after)

        if epoch >= args.use_hybrid_loss_after:
            criterion = full_loss
        else:
            criterion = berhu_loss

        # Cycle the learning rate on specified epoch.
        if epoch != 0 and epoch % args.lr_reset_epoch == 0:
            generator_optimizer = optim.Adam(generator.parameters(), lr=args.lr_reset_to, betas=(args.generator_beta1, 0.999))
            generator_scheduler = StepLR(generator_optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)

            critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr_reset_to, betas=(args.critic_beta1, 0.999))
            critic_scheduler = StepLR(critic_optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
            print('LR reset')

        generator.train()
        critic.train()
        epoch_losses = AverageMeter()
        epoch_berhu_loss = AverageMeter()
        epoch_gradient_loss = AverageMeter()
        epoch_normal_from_depth_loss = AverageMeter()
        epoch_adv_loss = AverageMeter()
        epoch_critic_loss = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('Training epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                color_input, depth_input, target = data

                color_input = color_input.to(device)
                depth_input = depth_input.to(device)
                target = target.to(device)

                critic_optimizer.zero_grad()
                generator_optimizer.zero_grad()

                # Get critic loss

                # Generator output
                pred = generator(color_input, depth_input) * 16.0
                target_missing_depth_mask = (target < 10 ** -6).detach()
                pred[target_missing_depth_mask] = 0.0

                # True samples
                discriminator_pred_true = critic(color_input, depth_input, target)
                # Fake samples. Detach() stops gradient to propagate back to generator
                discriminator_pred_fake = critic(color_input, depth_input, pred.detach())

                # True sample labels. Should be all ones.
                true_label = torch.ones_like(discriminator_pred_true)
                # Fake sample labels. Should be all zeros.
                fake_label = torch.zeros_like(discriminator_pred_fake)

                # Train critic with selected gan loss.

                # placeholder
                critic_loss = torch.zeros_like(discriminator_pred_fake)
                if args.gan_loss_type == 'rsgan':
                    critic_loss = bce_logits_loss(discriminator_pred_true - discriminator_pred_fake, true_label)
                elif args.gan_loss_type == 'rasgan':
                    critic_loss = (bce_logits_loss(discriminator_pred_true - torch.mean(discriminator_pred_fake),
                                                   true_label) + bce_logits_loss(
                        discriminator_pred_fake - torch.mean(discriminator_pred_true), fake_label)) / 2
                elif args.gan_loss_type == 'ralsgan':  # (y_hat-1)^2 + (y_hat+1)^2 probably 0.0025
                    critic_loss = (torch.mean(
                        (discriminator_pred_true - torch.mean(discriminator_pred_fake) - true_label) ** 2) +
                                   torch.mean((discriminator_pred_fake - torch.mean(
                                       discriminator_pred_true) + true_label) ** 2)) / 2
                elif args.gan_loss_type == 'rahingegan':
                    critic_loss = (torch.mean(torch.nn.ReLU()(
                        1.0 - (discriminator_pred_true - torch.mean(discriminator_pred_fake)))) + torch.mean(
                        torch.nn.ReLU()(1.0 + (discriminator_pred_fake - torch.mean(discriminator_pred_true))))) / 2
                elif args.gan_loss_type == 'hingegan':
                    hinge_pos = torch.mean(F.relu(1 - discriminator_pred_true, inplace=True))
                    hinge_neg = torch.mean(F.relu(1 + discriminator_pred_fake, inplace=True))
                    critic_loss = .5 * hinge_pos + .5 * hinge_neg

                # Train critic
                critic_loss.backward()
                critic_optimizer.step()

                # Get generator loss

                # Adversarial loss based on selected loss type
                # placeholder
                adversarial_loss = torch.zeros_like(discriminator_pred_fake)

                # generate discriminator prediction from real inputs and generator.
                discriminator_pred_true = critic(color_input, depth_input, target)
                discriminator_pred_fake = critic(color_input, depth_input, pred)

                if args.gan_loss_type == 'rsgan':
                    # Non-saturating
                    adversarial_loss = bce_logits_loss(discriminator_pred_fake - discriminator_pred_true, true_label)
                elif args.gan_loss_type == 'rasgan':
                    # Non-saturating
                    adversarial_loss = (bce_logits_loss(discriminator_pred_true - torch.mean(discriminator_pred_fake),
                                                        fake_label) + bce_logits_loss(
                        discriminator_pred_fake - torch.mean(discriminator_pred_true), true_label)) / 2
                elif args.gan_loss_type == 'ralsgan':
                    adversarial_loss = (torch.mean(
                        (discriminator_pred_true - torch.mean(discriminator_pred_fake) + true_label) ** 2) + torch.mean(
                        (discriminator_pred_fake - torch.mean(discriminator_pred_true) - true_label) ** 2)) / 2
                elif args.gan_loss_type == 'rahingegan':
                    # Non-saturating
                    adversarial_loss = (torch.mean(
                        torch.nn.ReLU()(
                            1.0 + (discriminator_pred_true - torch.mean(discriminator_pred_fake)))) + torch.mean(
                        torch.nn.ReLU()(1.0 - (discriminator_pred_fake - torch.mean(discriminator_pred_true))))) / 2
                elif args.gan_loss_type == 'hingegan':
                    adversarial_loss = - discriminator_pred_fake.mean()

                # Other (ordinary) losses
                loss = criterion(pred, target)

                if batch_counter > args.pretrain_critic_iter:
                    loss = args.critic_loss_coeff * adversarial_loss + loss
                batch_loss = loss.item()
                # Train generator
                loss.backward()
                generator_optimizer.step()

                with torch.no_grad():
                    batch_berhu_loss = berhu_loss(pred, target).item()
                    batch_gradient_loss = gradient_loss(pred, target).item()
                    batch_normal_from_depth_loss = vnl_loss(pred, target).item()
                    batch_adv_loss = adversarial_loss.item()

                    epoch_adv_loss.update(batch_adv_loss)
                    epoch_berhu_loss.update(batch_berhu_loss)
                    epoch_gradient_loss.update(batch_gradient_loss)
                    epoch_normal_from_depth_loss.update(batch_normal_from_depth_loss)

                epoch_losses.update(batch_loss, len(color_input))

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg), batch_loss='{:.6f}'.format(batch_loss),
                              batch_adv_loss='{:.6f}'.format(batch_adv_loss))
                t.update(len(color_input))

                batch_counter += 1

                if batch_counter > args.pretrain_critic_iter and batch_counter % args.test_step == 0:
                    eval_l1 = run_eval(generator, eval_dataloader)
                    if eval_l1 < best_l1_filtered:
                        best_l1_epoch_filtered = epoch
                        best_l1_filtered = eval_l1

                        critic_weights = copy.deepcopy(critic.state_dict())
                        generator_weights = copy.deepcopy(generator.state_dict())
                        torch.save(generator_weights,
                                   r'../tmp/gan_best_' + args.model_type + args.gan_loss_type + args.model_name_suffix + '.pth')
                        torch.save(critic_weights,
                                   r'../tmp/gan_best_' + args.model_type + args.gan_loss_type + args.model_name_suffix + '_critic.pth')
                        print('add best epoch: {}, l1 loss: {:.4f}'.format(best_l1_epoch_filtered, best_l1_filtered))

                    if args.use_file_list != 'original':
                        eval_l1 = run_eval(generator, eval_dataloader_original)
                        if eval_l1 < best_l1_original:
                            best_l1_epoch_original = epoch
                            best_l1_original = eval_l1

                            critic_weights = copy.deepcopy(critic.state_dict())
                            generator_weights = copy.deepcopy(generator.state_dict())
                            torch.save(generator_weights,
                                       r'../tmp/gan_best_' + args.model_type + args.gan_loss_type + args.model_name_suffix + '.pth')
                            torch.save(critic_weights,
                                       r'../tmp/gan_best_' + args.model_type + args.gan_loss_type + args.model_name_suffix + '_critic.pth')
                            print('add best epoch on original set: {}, l1 loss: {:.4f}'.format(best_l1_epoch_original,
                                                                                               best_l1_original))
        critic_scheduler.step()
        generator_scheduler.step()

        print('training berhu loss: {:.4f}'.format(epoch_berhu_loss.avg))
        print('training adv loss: {:.4f}'.format(epoch_adv_loss.avg))
        print('training gradient loss: {:.4f}'.format(epoch_gradient_loss.avg))
        print('training vnl from depth loss: {:.4f}'.format(epoch_normal_from_depth_loss.avg))
        print('training adv loss from depth loss: {:.4f}'.format(epoch_adv_loss.avg))

        critic_weights = copy.deepcopy(critic.state_dict())
        generator_weights = copy.deepcopy(generator.state_dict())

        eval_l1 = run_eval(generator, eval_dataloader)
        if eval_l1 < best_l1_filtered:
            best_l1_epoch_filtered = epoch
            best_l1_filtered = eval_l1
            torch.save(generator_weights,
                       r'../tmp/gan_best_' + args.model_type + args.gan_loss_type + args.model_name_suffix + '.pth')
            torch.save(critic_weights,
                       r'../tmp/gan_best_' + args.model_type + args.gan_loss_type + args.model_name_suffix + '_critic.pth')
            print('add best epoch: {}, l1 loss: {:.4f}'.format(best_l1_epoch_filtered, best_l1_filtered))

        if args.use_file_list != 'original':
            eval_l1 = run_eval(generator, eval_dataloader_original)
            if eval_l1 < best_l1_original:
                best_l1_epoch_original = epoch
                best_l1_original = eval_l1
                torch.save(generator_weights,
                           r'../tmp/gan_best_' + args.model_type + args.gan_loss_type + args.model_name_suffix + '.pth')
                torch.save(critic_weights,
                           r'../tmp/gan_best_' + args.model_type + args.gan_loss_type + args.model_name_suffix + '_critic.pth')
                print('add best epoch on original set: {}, l1 loss: {:.4f}'.format(best_l1_epoch_original,
                                                                                   best_l1_original))

        print('saving latest epoch')
        torch.save(generator_weights,
                   r'../tmp/gan_latest_' + args.model_type + args.gan_loss_type + args.model_name_suffix + '.pth')
        torch.save(critic_weights,
                   r'../tmp/gan_latest_' + args.model_type + args.gan_loss_type + args.model_name_suffix + '_critic.pth')

        print('add best epoch: {}, l1 loss: {:.4f}'.format(best_l1_epoch_filtered, best_l1_filtered))
        print('add best epoch on original set: {}, l1 loss: {:.4f}'.format(best_l1_epoch_original, best_l1_original))
