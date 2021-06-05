import argparse
import cv2
import sys
from cv2 import aruco
import imageio
import os
from os import path
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from bm3d import bm3d

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from image_processing import generate_distortion_mappings

NEAR_CLIP = 0.199
FAR_CLIP = 16.0

RGB_WIDTH = 4032
RGB_HEIGHT = 3024
DEPTH_WIDTH = 320
DEPTH_HEIGHT = 240
DEPTH_WINDOW_SIZE = 9

EPS = 10 ** -5


def sample_depth(depth, coordinate):
    depth_window = depth[max(coordinate[0] - DEPTH_WINDOW_SIZE // 2, 0): min(coordinate[0] + DEPTH_WINDOW_SIZE // 2 + 1,
                                                                             DEPTH_HEIGHT),
                   max(coordinate[1] - DEPTH_WINDOW_SIZE // 2, 0): min(coordinate[1] + DEPTH_WINDOW_SIZE // 2 + 1,
                                                                       DEPTH_WIDTH)]

    valid_elements = (depth_window > EPS).sum()
    if valid_elements < 0.6 * DEPTH_WINDOW_SIZE ** 2:
        return 0.0
    else:
        return float(depth_window.sum() / valid_elements)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str,
                        default='/Users/user/PyCharmProjects/DepthCompletion/image_processing/distance_test/')
    args = parser.parse_args()

    fx, fy, cx, cy = [3269.883, 3268.1606, 2038.7195, 1478.5912]
    intrinsics = np.array([[fx, 0., cx],
                           [0., fy, cy],
                           [0., 0., 1.]])

    distortion = np.array([0.0014165342, -0.035910547, 0., 0., 0.13729763])

    distances = []
    percentage = []

    rgb_filenames = []
    for file in os.listdir(args.folder):
        if file.endswith(".jpg"):
            rgb_filenames.append(os.path.join(args.folder, file))

    for rgb_filename in rgb_filenames:
        depth_filename = rgb_filename.split('.')[0] + '_z16.png'

        print('current file', rgb_filename, depth_filename)

        depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED).astype(np.float) * 0.001
        # depth = bm3d(depth, 0.03)

        frame = cv2.imread(rgb_filename)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # It looks like lens correction is already applied for RGB?
        mapx, mapy = generate_distortion_mappings(intrinsics, distortion, RGB_WIDTH, RGB_HEIGHT)
        # Remap the original image to a new image
        gray = cv2.remap(gray, mapx, mapy, cv2.INTER_CUBIC)

        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)

        # plt.figure()
        # plt.imshow(frame_markers)
        # for i in range(len(ids)):
        #     if ids[i] < 10:
        #         c = corners[i][0]
        #         plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))
        # plt.legend()
        # plt.show()

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.195, intrinsics, distortion)

        print('Depth calculated from Aruco')
        marker_count = 0
        total = 0
        for i in range(len(ids)):
            if ids[i] < 10:
                print(ids[i], tvecs[i][0][-1])
                total += tvecs[i][0][-1]
                marker_count += 1
        average_depth = total / marker_count
        print('Average depth calculated from Aruco', average_depth)

        crucial_vertex = []
        # plt.imshow(depth)
        for i in range(len(ids)):
            if ids[i] < 10:
                c = corners[i][0]
                point = [round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH)),
                         round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH))]
                # plt.plot([round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH))],
                #          [round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH))], "o", label="id={0}".format(ids[i]))

                if ids[i] in [8, 0, 4, 1]:
                    crucial_vertex.append((round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH)),
                                           round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH))))
        print('found crucial point', crucial_vertex)
        # plt.legend()
        # plt.show()

        temp = crucial_vertex[-2]
        crucial_vertex[-2] = crucial_vertex[-1]
        crucial_vertex[-1] = temp

        mask_image = Image.new('L', (DEPTH_WIDTH, DEPTH_HEIGHT), 0)
        ImageDraw.Draw(mask_image).polygon(crucial_vertex, outline=1, fill=1)
        mask = np.array(mask_image)

        # plt.imshow(mask)
        # plt.show()

        distances.append(average_depth)
        percentage.append(np.sum((depth[mask > 0] > 0) * 1) / np.sum(mask))

    print(list(distances))
    print(list(percentage))

    plt.scatter(distances, percentage)
    plt.show()
