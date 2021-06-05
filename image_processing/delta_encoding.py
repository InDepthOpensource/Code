import argparse
import cv2
import imageio
from os import path
import numpy as np
from bm3d import bm3d
from skimage.restoration import estimate_sigma
import matplotlib.pyplot as plt
import matplotlib.cm as cm

NEAR_CLIP = 0.199
FAR_CLIP = 16.0


def run_bm3d(depth_float):
    bm3d_depth = bm3d(depth_float, 0.03)
    bm3d_depth[bm3d_depth < NEAR_CLIP] = 0.0
    return bm3d_depth


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str,
                        default='/Users/user/PyCharmProjects/DepthCompletion/image_processing/tofsequence/')
    parser.add_argument('--index', nargs='+', type=int, default=[i for i in range(1, 1320)])
    args = parser.parse_args()

    depth_filename = args.folder + str(0) + '.png'
    argb_depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
    lower_8bits = argb_depth[:, :, 1].astype(np.uint16)
    higher_8bits = argb_depth[:, :, 2].astype(np.uint16)
    current_depth = ((higher_8bits << 8) + lower_8bits).astype(np.float) * 0.001
    current_depth = run_bm3d(current_depth)

    for i in args.index:
        depth_filename = args.folder + str(i) + '.png'
        if not path.exists(depth_filename):
            continue

        argb_depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
        lower_8bits = argb_depth[:, :, 1].astype(np.uint16)
        higher_8bits = argb_depth[:, :, 2].astype(np.uint16)
        next_depth = ((higher_8bits << 8) + lower_8bits).astype(np.float) * 0.001

        # cmap = cm.get_cmap(name='jet')
        # cmap.set_under('w')
        # norm = plt.Normalize(vmin=next_depth.min() + 0.01, vmax=next_depth.max())
        # plt.figure()
        # plt.imshow(cmap(norm(next_depth)))
        # plt.legend()
        # plt.show()

        next_depth = run_bm3d(next_depth)

        # cmap = cm.get_cmap(name='jet')
        # cmap.set_under('w')
        # norm = plt.Normalize(vmin=next_depth.min() + 0.01, vmax=next_depth.max())
        # plt.figure()
        # plt.imshow(cmap(norm(next_depth)))
        # plt.legend()
        # plt.show()

        delta = (next_depth - current_depth) * 1000
        max_negative_delta = np.ceil(np.abs(np.min(delta)))
        delta = delta + max_negative_delta

        print(i)
        print(delta.max())

        current_depth = next_depth
        imageio.imsave(str(i) + '_delta_z16.png', delta.astype(np.uint16))
        imageio.imsave(str(i) + '_bm3d_z16.png', current_depth.astype(np.uint16))
