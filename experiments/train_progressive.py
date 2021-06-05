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
from experiments.data import MatterportTrainDataset, MatterportEvalDataset
from utils import L1, AverageMeter, RMSE, GradientLoss, MultiScaleLoss, VNLLoss, BerHuLoss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model', type=str, default='')
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--lr-step', type=int, default=10)
    parser.add_argument('--lr-step-gamma', type=float, default=0.3)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=32598867)
    parser.add_argument('--train-rgb-encoder-after', type=int, default=12)
    parser.add_argument('--use-hybrid-loss-after', type=int, default=16)
    parser.add_argument('--model-type', type=str, default='progressive_unet')
    parser.add_argument('--vnl-coeff', type=float, default=0.3)
    parser.add_argument('--gradient-coeff', type=float, default=10.0)
    parser.add_argument('--model-name-suffix', type=str, default='')
    parser.add_argument('--multiscale-coeff', nargs='+', type=float, default=[0.003125, 0.00625, 0.0125, 0.05, 1.0])
    parser.add_argument('--grow-decoder-step', type=int, default=3)
    # If set to true each epoch has only one batch. For testing this script only.
    parser.add_argument('--test-script', type=bool, default=False)
    parser.add_argument('--init-all-stages', type=bool, default=False)
    parser.add_argument('--num-res-block-decoder', type=int, default=2)
    parser.add_argument('--loss-func', type=str, default='berhu')

    args = parser.parse_args()

    batch_size = args.batch_size
    decoder_stages_to_grow = 3
    epoches_until_next_grow = args.grow_decoder_step

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    loss_func = BerHuLoss if args.loss_func == 'berhu' else nn.L1Loss

    multiscale_loss = MultiScaleLoss(coefficients=args.multiscale_coeff, loss_func=loss_func)
    rmse_loss = RMSE()
    l1_loss = L1()
    gradient_loss = GradientLoss()
    vnl_loss = VNLLoss()

    model = ProgressiveUNet(device, num_res_block_decoder=args.num_res_block_decoder)

    model.to(device)
    multiscale_loss.to(device)
    rmse_loss.to(device)
    l1_loss.to(device)
    gradient_loss.to(device)

    if args.init_all_stages:
        for _ in range(decoder_stages_to_grow):
            model.add_decoder_block()
            print('Grown decoder by one stage. Current stage:', model.current_level)
        decoder_stages_to_grow = 0

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        print('loaded model')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)

    train_dataset = MatterportTrainDataset()
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataset = MatterportEvalDataset()
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False)

    best_rmse_epoch = 0
    best_rmse = 10 ** 5
    best_l1_epoch = 0
    best_l1 = 10 ** 5

    print('Train set length:', len(train_dataset))
    print('Eval set length:', len(eval_dataset))

    for epoch in range(args.num_epochs):
        # Disable the training for RGB encoder at the very beginning
        for param in model.rgb_encoder.parameters():
            param.requires_grad = (epoch >= args.train_rgb_encoder_after)

        model.train()
        epoch_losses = AverageMeter()
        epoch_berhu_loss = [AverageMeter() for _ in range(5)]
        epoch_gradient_loss = AverageMeter()
        epoch_normal_from_depth_loss = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('Training epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                color_input, depth_input, target = data

                color_input = color_input.to(device)
                depth_input = depth_input.to(device)
                target = target.to(device)

                pred = model(color_input, depth_input)
                pred = [i * 16.0 for i in pred]

                batch_berhu_loss, final_loss = multiscale_loss(pred, target)

                if epoch >= args.use_hybrid_loss_after and decoder_stages_to_grow == 0:
                    batch_gradient_loss = gradient_loss(pred[-1], target)
                    batch_vnl_loss = vnl_loss(pred[-1], target)
                    final_loss += (batch_gradient_loss * args.gradient_coeff + batch_vnl_loss * args.vnl_coeff)
                    epoch_gradient_loss.update(batch_gradient_loss.item())
                    epoch_normal_from_depth_loss.update(batch_vnl_loss.item())

                final_loss_item = final_loss.item()
                epoch_losses.update(final_loss_item, len(color_input))

                for i in range(len(epoch_berhu_loss)):
                    epoch_berhu_loss[i].update(batch_berhu_loss[i])

                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg), batch_loss='{:.6f}'.format(final_loss_item))
                t.update(len(color_input))

                if args.test_script:
                    break

            # Advance the Learning rate scheduler and decrease the learning rate accordingly.
            epoches_until_next_grow -= 1
            if decoder_stages_to_grow == 0:
                scheduler.step()
                print('stepped lr scheduler')
            elif epoches_until_next_grow == 0:
                model.add_decoder_block()
                epoches_until_next_grow = args.grow_decoder_step
                decoder_stages_to_grow -= 1
                print('Grown decoder by one stage. Current stage:', model.current_level)

        print('full scale training berhu loss: {:.4f}'.format(epoch_berhu_loss[-1].avg))
        print('training gradient loss: {:.4f}'.format(epoch_gradient_loss.avg))
        print('training normal from depth loss: {:.4f}'.format(epoch_normal_from_depth_loss.avg))
        print('1/16 scale training berhu loss: {:.4f}'.format(epoch_berhu_loss[0].avg))
        print('1/8 scale training berhu loss: {:.4f}'.format(epoch_berhu_loss[1].avg))
        print('1/4 scale training berhu loss: {:.4f}'.format(epoch_berhu_loss[2].avg))
        print('1/2 scale training berhu loss: {:.4f}'.format(epoch_berhu_loss[3].avg))

        if decoder_stages_to_grow == 0:
            model.eval()
            total_l1 = 0.0
            total_mse = 0.0
            total_valid_pixel = 0.0
            epoch_berhu_loss = AverageMeter()
            epoch_gradient_loss = AverageMeter()
            epoch_normal_from_depth_loss = AverageMeter()

            with tqdm(total=(len(eval_dataset) - len(eval_dataset) % args.batch_size)) as t:
                t.set_description('Testing epoch: {}/{}'.format(epoch, args.num_epochs - 1))

                for data in eval_dataloader:
                    color_input, depth_input, target, _ = data

                    color_input = color_input.to(device)
                    depth_input = depth_input.to(device)
                    target = target.to(device)

                    with torch.no_grad():
                        pred = (model(color_input, depth_input)[-1] * 16.0).clamp(0.0, (2 ** 16 - 1) / 4000)
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

            eval_l1 = total_l1 / total_valid_pixel
            eval_rmse = sqrt(total_mse / total_valid_pixel)

            print('eval RMSE: {:.4f}'.format(eval_rmse))
            print('eval l1 loss: {:.4f}'.format(eval_l1))
            print('eval gradient loss: {:.4f}'.format(epoch_gradient_loss.avg))
            print('eval normal from depth loss: {:.4f}'.format(epoch_normal_from_depth_loss.avg))

            weights = copy.deepcopy(model.state_dict())

            # if eval_rmse < best_rmse:
            #     best_rmse_epoch = epoch
            #     best_rmse = eval_rmse
            #     torch.save(weights, r'../tmp/best_rmse_' + args.model_type + '.pth')

            if eval_l1 < best_l1:
                best_l1_epoch = epoch
                best_l1 = eval_l1
                torch.save(weights, r'../tmp/best_l1_' + args.model_type + args.model_name_suffix + '.pth')

            print('saving latest epoch')
            torch.save(weights, r'../tmp/latest_' + args.model_type + args.model_name_suffix + '.pth')

            print('add best epoch: {}, l1 loss: {:.4f}'.format(best_l1_epoch, best_l1))
