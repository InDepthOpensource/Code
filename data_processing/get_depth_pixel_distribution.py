import argparse
import copy
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import *
from experiments.data import MatterportTrainDataset, MatterportEvalDataset, MATTERPORT_EVAL_FILE_LIST, \
    MATTERPORT_TRAIN_FILE_LIST

EPS = 10 ** -4
BASE_ERR_THRESHOLD = 0.3
MIN_PERCENT_BELOW_THRESHOLD = 0.75

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-file', type=str, default='')
    # parser.add_argument('--eval-file', type=str, default='data/test.blob')
    # parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3242357)
    args = parser.parse_args()

    train_dataset = MatterportTrainDataset(use_generated_mask=False, file_list_name=MATTERPORT_TRAIN_FILE_LIST)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataset = MatterportEvalDataset(file_list_name=MATTERPORT_EVAL_FILE_LIST)
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

    # 0.3 + 1.05 * x, 0.3 + 1.10 * x, 0.3 + 1.25 * x, 0.3 + 1.25 ** 2 * x, rest
    total_pixels = 0
    pixel_intervals = [0, 0, 0, 0, 0]

    # histogram for percent of pixels above 0.3 + 1.10 * x v.s. number of frames
    total_frames = 0
    histogram_110 = [0 for _ in range(10)]
    histogram_125 = [1 for _ in range(10)]

    batch_counter = 0
    f = open('matterport_eval_keep_id.txt', 'w+')
    with tqdm(total=(len(eval_dataset) - len(eval_dataset) % args.batch_size)) as t:
        t.set_description('eval_set')

        for data in eval_dataloader:
            color_input, depth_input, target, _ = data

            depth_input *= 16

            depth_input[target < EPS] = 0.0
            target[depth_input < EPS] = EPS
            depth_input[depth_input < EPS] = EPS
            valid_pixels = int(torch.sum(depth_input > EPS))

            error = torch.abs((target - depth_input))
            threshold_105 = BASE_ERR_THRESHOLD + 0.05 * target
            threshold_110 = BASE_ERR_THRESHOLD + 0.10 * target
            threshold_125 = BASE_ERR_THRESHOLD + 0.25 * target
            threshold_125_2 = BASE_ERR_THRESHOLD + 0.5625 * target

            # Get pixel distribution
            total_pixels += valid_pixels
            pixel_intervals[0] += int(torch.sum((2 * EPS < depth_input) & (error <= threshold_105)))
            pixel_intervals[1] += int(torch.sum((threshold_105 < error) & (error <= threshold_110)))
            pixel_intervals[2] += int(torch.sum((threshold_110 < error) & (error <= threshold_125)))
            pixel_intervals[3] += int(torch.sum((threshold_125 < error) & (error <= threshold_125_2)))
            pixel_intervals[4] += int(torch.sum(error > threshold_125_2))

            # Get frame distribution
            if valid_pixels >= target.numel() * 0.15:
                total_frames += 1
                # Count distribution within 1 +/- 10%
                within_110_percent = int(torch.sum((2 * EPS < depth_input) & (error <= threshold_110))) / valid_pixels
                index = min(math.floor(within_110_percent / 0.1), 9)
                histogram_110[index] += 1

                # Count distribution within 1 +/- 25%
                within_125_percent = int(torch.sum((2 * EPS < depth_input) & (error <= threshold_125))) / valid_pixels
                index = min(math.floor(within_125_percent / 0.1), 9)
                histogram_125[index] += 1

                if within_125_percent >= MIN_PERCENT_BELOW_THRESHOLD:
                    f.write(str(batch_counter))
                    f.write('\n')
            else:
                f.write(str(batch_counter))
                f.write('\n')

            batch_counter += 1
            t.update(len(color_input))

        print(batch_counter)
        pixel_intervals_copy = [i / total_pixels for i in pixel_intervals]
        print('pixel distribution is', pixel_intervals_copy)
        histogram_110_copy = [i / total_frames for i in histogram_110]
        print('1.10 distribution is', histogram_110_copy)
        histogram_125_copy = [i / total_frames for i in histogram_125]
        print('1.25 distribution is', histogram_125_copy)
    f.close()

    # 0.3 + 1.05 * x, 0.3 + 1.10 * x, 0.3 + 1.25 * x, 0.3 + 1.25 ** 2 * x, rest
    total_pixels = 0
    pixel_intervals = [0, 0, 0, 0, 0]

    # histogram for percent of pixels above 0.3 + 1.10 * x v.s. number of frames
    total_frames = 0
    histogram_110 = [0 for _ in range(10)]
    histogram_125 = [1 for _ in range(10)]

    batch_counter = 0
    f = open('matterport_train_keep_id.txt', 'w+')
    with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
        t.set_description('training set')
        for data in train_dataloader:
            color_input, depth_input, target = data

            depth_input *= 16

            depth_input[target < EPS] = 0.0
            target[depth_input < EPS] = EPS
            depth_input[depth_input < EPS] = EPS
            valid_pixels = int(torch.sum(depth_input > EPS))

            error = torch.abs((target - depth_input))
            threshold_105 = BASE_ERR_THRESHOLD + 0.05 * target
            threshold_110 = BASE_ERR_THRESHOLD + 0.10 * target
            threshold_125 = BASE_ERR_THRESHOLD + 0.25 * target
            threshold_125_2 = BASE_ERR_THRESHOLD + 0.5625 * target

            # Get pixel distribution
            total_pixels += valid_pixels
            pixel_intervals[0] += int(torch.sum((2 * EPS < depth_input) & (error <= threshold_105)))
            pixel_intervals[1] += int(torch.sum((threshold_105 < error) & (error <= threshold_110)))
            pixel_intervals[2] += int(torch.sum((threshold_110 < error) & (error <= threshold_125)))
            pixel_intervals[3] += int(torch.sum((threshold_125 < error) & (error <= threshold_125_2)))
            pixel_intervals[4] += int(torch.sum(error > threshold_125_2))

            # Get frame distribution
            if valid_pixels >= target.numel() * 0.15:
                total_frames += 1
                # Count distribution within 1 +/- 10%
                within_110_percent = int(torch.sum((2 * EPS < depth_input) & (error <= threshold_110))) / valid_pixels
                index = min(math.floor(within_110_percent / 0.1), 9)
                histogram_110[index] += 1

                # Count distribution within 1 +/- 25%
                within_125_percent = int(torch.sum((2 * EPS < depth_input) & (error <= threshold_125))) / valid_pixels
                index = min(math.floor(within_125_percent / 0.1), 9)
                histogram_125[index] += 1

                if within_125_percent >= MIN_PERCENT_BELOW_THRESHOLD:
                    f.write(str(batch_counter))
                    f.write('\n')
            else:
                f.write(str(batch_counter))
                f.write('\n')

            t.set_postfix(current_batch_counter=batch_counter)
            t.update(len(color_input))

            batch_counter += 1

            if batch_counter % 1000 == 0:
                print(batch_counter)
                pixel_intervals_copy = [i / total_pixels for i in pixel_intervals]
                print('pixel distribution is', pixel_intervals_copy)
                histogram_110_copy = [i / total_frames for i in histogram_110]
                print('1.10 distribution is', histogram_110_copy)
                histogram_125_copy = [i / total_frames for i in histogram_125]
                print('1.25 distribution is', histogram_125_copy)
        print(batch_counter)
        pixel_intervals_copy = [i / total_pixels for i in pixel_intervals]
        print('pixel distribution is', pixel_intervals_copy)
        histogram_110_copy = [i / total_frames for i in histogram_110]
        print('1.10 distribution is', histogram_110_copy)
        histogram_125_copy = [i / total_frames for i in histogram_125]
        print('1.25 distribution is', histogram_125_copy)

        f.close()
