import argparse
import cv2
import os
import imageio
from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from bm3d import bm3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str,
                        default='/Users/user/Downloads/tof_sequence/tof_captures/')
    args = parser.parse_args()

    brightness_count = np.zeros(100)
    valid_pixel_count = np.zeros(100)


    for file in os.listdir(args.folder):
        if not file.endswith('_z16.png'):
            continue

        depth_filename = args.folder + file
        rgb_filename = depth_filename.replace('_z16.png', '_undistorted.jpg')

        if not (path.exists(rgb_filename) and path.exists(depth_filename)):
            continue

        rgb = cv2.imread(rgb_filename)
        brightness = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) / 255
        depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)

        if rgb is None or depth is None:
            continue

        depth = depth[20:220, 20:300]
        brightness = brightness[20:220, 20:300]

        current_brightness_binned, _ = np.histogram(brightness, 100, (0, 1))
        valid_pixels_binned_by_brightness, _ = np.histogram(brightness[depth > 0], 100, (0, 1))

        brightness_count = brightness_count + current_brightness_binned
        valid_pixel_count = valid_pixel_count + valid_pixels_binned_by_brightness

    percentages = valid_pixel_count / brightness_count
    plt.scatter(np.linspace(0, 1, 100), percentages)

    print(list(np.linspace(0, 1, 100)))
    print(list(percentages))

    plt.show()
