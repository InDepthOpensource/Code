import argparse
import cv2
import os
import numpy as np
import shutil
import random

def write_file_list(filename, file_list):
    f = open(filename, 'w+')
    for i in file_list:
        print(' '.join(i), file=f)
    f.close()

FILE_ROOT = '/usr/xtmp/yz322/samsung_tof/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str,
                        default='/Users/user/Downloads/tof_sequence/tof_captures/')
    args = parser.parse_args()

    training_file_list = []
    test_file_list = []

    total_valid_depth_pixels = 0
    total_pixels = 0

    frame_counter = 0

    for depth_filename in os.listdir(args.folder):
        if (not depth_filename.endswith('z16.png')) or (not depth_filename.startswith('2020-') ):
            continue

        depth_filename = args.folder + depth_filename

        rgb_filename = depth_filename.replace('_z16.png', '_undistorted.jpg')

        try:
            depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(e)
            continue

        if not os.path.exists(rgb_filename) or depth is None:
            continue

        valid_depth = depth > 1

        percent_valid = float(valid_depth.sum() / valid_depth.size)

        total_valid_depth_pixels += int(valid_depth.sum())
        total_pixels += valid_depth.size
        frame_counter += 1

        # if percent_valid > 0.3:
        #     shutil.copy2(rgb_filename, '/Users/user/PycharmProjects/tof_seq_filtered/')
        #     shutil.copy2(depth_filename, '/Users/user/PycharmProjects/tof_seq_filtered/')
        #
        #     rgb_remote_filename = FILE_ROOT + rgb_filename.split(r'/')[-1]
        #     depth_remote_filename = FILE_ROOT + depth_filename.split(r'/')[-1]
        #
        #     if random.random() < 0.1:
        #         test_file_list.append((rgb_remote_filename, depth_remote_filename, depth_remote_filename))
        #         print('added to test, ', rgb_remote_filename, depth_remote_filename)
        #     else:
        #         training_file_list.append((rgb_remote_filename, depth_remote_filename, depth_remote_filename))
        #         print('added to train, ', rgb_remote_filename, depth_remote_filename)

    # write_file_list('samsung_tof_test_file_list.txt', test_file_list)
    # write_file_list('samsung_tof_train_file_list.txt', training_file_list)

    print('Total percentage of valid depth pixels', total_valid_depth_pixels/total_pixels)
    print('Total frames processed', frame_counter)
