import argparse
import cv2
import os
import imageio
from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str,
                        default='/Users/user/Downloads/AllDepthPGM/')
    args = parser.parse_args()

    percent_missing_bins = [0 for _ in range(10)]
    total_pixels = 0
    total_valid_pixels = 0

    for file in os.listdir(args.folder):
        # if not file.endswith('_z16.png'):
        #     continue

        if not file.endswith('.pgm'):
            continue

        depth_filename = args.folder + file

        depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)

        if depth is None:
            continue

        depth = depth[:, 50:270]

        percent_depth_missing = math.floor(float((depth < 1).sum() / depth.size) * 10 - 0.00001)
        percent_missing_bins[percent_depth_missing] += 1
        total_valid_pixels += int((depth > 1).sum())
        total_pixels += depth.size

    print('Percent depth missing', 1.0 - total_valid_pixels / total_pixels)
    percent_missing_bins = np.array(percent_missing_bins) / sum(percent_missing_bins)
    print(list(percent_missing_bins))
    plt.scatter(np.linspace(0, 1, 10), np.array(percent_missing_bins))
    plt.show()
