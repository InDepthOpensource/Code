import argparse
import cv2
import imageio
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_bilateral

import matplotlib.pyplot as plt
import matplotlib.cm as cm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-file', type=str, default='')
    # parser.add_argument('--eval-file', type=str, default='data/test.blob')
    # parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--rgb', type=str, default='')
    parser.add_argument('--depth', type=str, default='')
    args = parser.parse_args()

    rgb = cv2.imread(args.rgb, cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    depth_float = (depth / 1000.0).astype(np.float32)
    valid_depth_mask = depth > 10 ** -6
    invalid_depth_mask = depth <= 10 ** -6

    depth_denoised = denoise_bilateral(depth_float)

    depth_inpainted = cv2.inpaint(depth_denoised, (invalid_depth_mask * 1).astype(np.uint8), 20, cv2.INPAINT_TELEA)

    cmap = cm.get_cmap(name='jet')
    cmap.set_under('w')
    norm = plt.Normalize(vmin=10 ** -6, vmax=depth_inpainted.max())

    plt.imsave('depth_denoised.png', cmap(norm(depth_float)))
    plt.imsave('depth_inpainted.png', cmap(norm(depth_inpainted)))

    imageio.imsave('depth_denoised_z16.png', (depth_denoised * 1000).astype(np.uint16))
    imageio.imsave('depth_inpainted_denoised_z16.png', (depth_inpainted * 1000).astype(np.uint16))
