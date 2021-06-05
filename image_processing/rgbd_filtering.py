import argparse
import cv2
import numpy as np
from bm3d import bm3d
from skimage.restoration import denoise_nl_means, estimate_sigma

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
    depth_float = (depth / 2 ** 16).astype(np.float32)
    valid_depth_mask = depth > 10 ** -6

    cmap = cm.get_cmap(name='jet')
    cmap.set_under('w')
    norm = plt.Normalize(vmin=10 ** -6, vmax=depth_float.max())

    # Non local denoising
    sigma_est = np.mean(estimate_sigma(depth_float, multichannel=False))
    patch_kw = dict(patch_size=5,  # 5x5 patches
                    patch_distance=6,  # 13x13 search area
                    multichannel=False)
    non_local_depth = denoise_nl_means(depth_float, h=0.4, sigma=0.5,
                                       fast_mode=True, **patch_kw)

    # BM3D
    bm3d_depth = bm3d(depth_float, 0.15)

    # Guided filtering
    guided_filtering_depth = cv2.ximgproc.guidedFilter(rgb, depth_float, 8, 50)

    # Joint Bilateral Filtering
    jbf_depth = cv2.ximgproc.jointBilateralFilter((rgb / 2 ** 8).astype(np.float32),
                                                  depth_float, 30, 20, 20)

    plt.imsave('original_0.png', cmap(norm(depth_float)))
    plt.imsave('result_non_local_0.png', cmap(norm(non_local_depth)))
    plt.imsave('result_bm3d_0.png', cmap(norm(bm3d_depth)))
    plt.imsave('result_guided_0.png', cmap(norm(guided_filtering_depth)))
    plt.imsave('result_jbf_0.png', cmap(norm(jbf_depth)))
