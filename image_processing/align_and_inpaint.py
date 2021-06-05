#!/usr/bin/python3
import os

# Limit number of numpy threads to 1, as more threads do not improve performance
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import argparse
import cv2
import os
import sys
import imageio
import time
from os import path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from math import sqrt
from torch import nn
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from models import CBAMDilatedUNet, CBAMUNet
from utils import AverageMeter, BerHuLoss, RMSE, GradientLoss, NormalFromDepthLoss, L1, PercentWithinError, VNLLoss


NEAR_CLIP = 0.01
FAR_CLIP = 16.0
depth_width = 320
depth_height = 240
rgb_width = 320
rgb_height = 240


def gen_rotation_matrix(q):
    # x, y, z, w = q
    # R = [[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
    #      [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
    #      [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]]
    # R = np.array(R)
    R = np.eye(3)
    return R


def generate_distortion_mappings(intrinsics, distortion_parameters, w, h):
    # Generate new camera matrix from parameters
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(intrinsics, distortion_parameters, (w, h), 0)
    # Generate look-up tables for remapping the camera image
    mapx, mapy = cv2.initUndistortRectifyMap(intrinsics, distortion_parameters, None, newcameramatrix, (w, h), 5)
    return mapx, mapy


def calculate_extrinsics(translation, rotation):
    R = gen_rotation_matrix(rotation)
    t = R @ np.array(translation).reshape((3, 1))

    extrinsic = np.zeros([4, 4])
    for i in range(3):
        for j in range(3):
            extrinsic[i][j] = R[i][j]
    extrinsic[3][3] = 1
    for m in range(3):
        extrinsic[m][3] = t[m]
    return extrinsic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str,
                        default='/Users/user/PycharmProjects/DepthCompletion/image_processing/verify_distance_measurements/')
    parser.add_argument('--model-type', type=str, default='cbam6')
    parser.add_argument('--load-model', type=str, default='/Users/user/PycharmProjects/DepthCompletion/tmp/best_l1_originalcbam6crop_finetune.pth')
    args = parser.parse_args()

    # Generate mappings for alignment, etc
    depth_fx, depth_fy, depth_cx, depth_cy = [280.44355, 280.17345, 158.51317, 119.690275]
    depth_intrinsics = np.array([[depth_fx, 0., depth_cx],
                                    [0., depth_fy, depth_cy],
                                    [0., 0., 1.]])
    depth_distortion = np.array([0.025927328, 0.42491153, 0., 0., -0.8128051])

    # We changed the coordinate system.
    # Holding your Samsung Note 10+ horizontally with selfie camera on your left hand side.
    # x axis point to your right, y axis points downwards, z axis points away from you.
    # depth_translation = [0.0, 0.0, 0.0]
    depth_translation = [-0.0040181815, 0.01131969, 0.0]
    depth_rotation = [0.70518136, -0.7089894, 0.0047044773, -0.0034524237]

    rgb_fx, rgb_fy, rgb_cx, rgb_cy = [259.51452381, 259.3778254, 161.80313492, 117.34850794]
    rgb_intrinsics = np.array([[rgb_fx, 0., rgb_cx],
                                [0., rgb_fy, rgb_cy],
                                [0., 0., 1.]])
    rgb_distortion = np.array([0.0014165342, -0.035910547, 0.0, 0.0, 0.13729763])
    rgb_translation = [0.0, 0.0, 0.0]
    rgb_rotation = [0.7071, -0.7071, 0.0, 0.0]

    depth_extrinsics = calculate_extrinsics(depth_translation, depth_rotation)
    depth_mapx, depth_mapy = generate_distortion_mappings(depth_intrinsics, depth_distortion, depth_width, depth_height)

    rgb_rotation_inverse = gen_rotation_matrix(rgb_rotation).T
    rgb_mapx, rgb_mapy = generate_distortion_mappings(rgb_intrinsics, rgb_distortion, rgb_width, rgb_height)
    
    # Init pytorch
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_type == 'cbam6':
        model = CBAMDilatedUNet(device, 2, use_cbam_encoder=False, use_cbam_fuse=False, use_cbam_decoder=False)
    else:
        raise NotImplemented()

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print('loaded model')

    model.to(device)
    model.eval()

    torch_rgb_transforms = transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ))
    torch_depth_transforms = transforms.Compose((transforms.ToTensor(),))

    depth_resize = transforms.Resize((256, 320), interpolation=Image.BILINEAR)
    color_resize = transforms.Resize((256, 320), interpolation=Image.BICUBIC)

    print('Init success')

    counter = 0

    for file in os.listdir(args.folder):
        if not file.endswith('.png'):
            continue

        depth_filename = args.folder + file
        rgb_filename = depth_filename.replace('.png', '.jpg')

        if not path.exists(rgb_filename):
            continue

        print('new frame!')

        new_frame_time = time.time()
        start = time.time()
        argb_depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
        lower_8bits = argb_depth[:, :, 1].astype(np.uint32)
        higher_8bits = argb_depth[:, :, 2].astype(np.uint32)
        depth = ((higher_8bits << 8) + lower_8bits) * 0.001

        rgb = cv2.imread(rgb_filename, cv2.IMREAD_UNCHANGED)
        rgb = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
        end = time.time()
        print('time to read images', end - start)

        start = time.time()
        # Correct for rgb and depth distortion by remapping the original image to a new image.
        depth_undistorted = cv2.remap(depth, depth_mapx, depth_mapy, cv2.INTER_NEAREST)
        depth_undistorted[depth_undistorted < NEAR_CLIP] = 0.0
        rgb_undistorted = cv2.remap(rgb, rgb_mapx, rgb_mapy, cv2.INTER_LINEAR)
        end = time.time()
        print('time to undistort images', end - start)

        start = time.time()
        # Start of RGBD alignment.
        y, x = np.meshgrid(np.arange(depth_height), np.arange(depth_width), indexing='ij')
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        z = depth_undistorted.reshape(1, -1)

        # Convert points on the depth map to Android sensor coordinate space.
        x = (x - depth_cx) / depth_fx
        y = (y - depth_cy) / depth_fy
        pts = np.vstack((x * z, y * z, z))
        pts = depth_extrinsics[:3, :3] @ pts + depth_extrinsics[:3, 3:]

        # Alternative way to do the coordinate transformation. Possibly faster too.
        # Get rid of points where depth is missing, as we cannot calculate meaningful pixel_x or pixel_y in this case
        pts = pts.T
        pts = pts[pts[:, 2] > NEAR_CLIP]
        pts = pts[(-pts[:, 2]).argsort()]
        pts = pts.T
        
        pixel_x = np.rint(pts[0, :] / pts[2, :] * rgb_fx + rgb_cx).astype(int)
        pixel_y = np.rint(pts[1, :] / pts[2, :] * rgb_fy + rgb_cy).astype(int)
        
        mask = (0 <= pixel_x) & (pixel_x < depth_width) & (0 <= pixel_y) & (pixel_y < depth_height)
        pixel_x = pixel_x[mask]
        pixel_y = pixel_y[mask]
        pixel_z = pts[2, :][mask]
        
        projected_depth = np.zeros_like(depth_undistorted)
        projected_depth[pixel_y, pixel_x] = pixel_z

        # projected_depth = np.ones_like(depth_undistorted) * FAR_CLIP
        # for i in range(pts.shape[-1]):
        #     if pts[2, i] > NEAR_CLIP:
        #         pixel_x = round(pts[0, i] / pts[2, i] * rgb_fx + rgb_cx)
        #         pixel_y = round(pts[1, i] / pts[2, i] * rgb_fy + rgb_cy)

        #         if 0 <= pixel_x < depth_width and 0 <= pixel_y < depth_height:
        #             projected_depth[pixel_y, pixel_x] = max(0.0, min(z[0, i], projected_depth[pixel_y, pixel_x]))

        projected_depth[projected_depth > FAR_CLIP] = 0.0
        end = time.time()
        print('time to align images', end - start)

        # imageio.imsave(depth_filename.split('.')[0] + '_z16.png', (projected_depth * 1000).astype(np.uint16))
        start = time.time()
        rgb_undistorted = cv2.cvtColor(rgb_undistorted, cv2.COLOR_BGR2RGB)
        color_input = torch.unsqueeze(torch_rgb_transforms(color_resize(Image.fromarray(rgb_undistorted))), dim=0)
        projected_depth = projected_depth / 16.0
        depth_input = torch.unsqueeze(torch_depth_transforms(depth_resize(Image.fromarray(projected_depth))), dim=0)
        end = time.time()
        print('time to convert to tensor', end - start)

        start = time.time()

        color_input = color_input.to(device)
        depth_input = depth_input.to(device)

        with torch.no_grad():
            pred = (model(color_input, depth_input) * 16.0).clamp(NEAR_CLIP, FAR_CLIP)
            pred = pred.cpu().numpy()

            end = time.time()
            print('time to inpaint images', end - start)

            start = time.time()
            cv2.imwrite(str(counter) + '_completed_z16.png', (pred[0][0] * 1000).astype(np.uint16))
            end = time.time()
            print('time to save images', end - start)

        print('Total frame time', time.time() - new_frame_time)

        counter += 1

    print('Processed', counter, "images")
