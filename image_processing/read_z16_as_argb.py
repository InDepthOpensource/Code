import argparse
import os

import cv2
import numpy as np
import imageio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str,
                        default='/Users/user/PyCharmProjects/DepthCompletion/image_processing/verify_distance_measurements/')
    args = parser.parse_args()
    argb_depth_file_list = []

    for file in os.listdir(args.folder):
        if file.endswith('.png'):
            filename = os.path.join(args.folder, file)

            argb_depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            lower_8bits = argb_depth[:, :, 1].astype(np.uint32)
            higher_8bits = argb_depth[:, :, 2].astype(np.uint32)
            z16_depth = (higher_8bits << 8) + lower_8bits
            depth = z16_depth
            imageio.imsave(filename.split('.')[0] + '_z16.png', depth.astype(np.uint16))
