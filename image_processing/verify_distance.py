import argparse
import cv2
import sys
from cv2 import aruco
# import imageio
import os
from os import path
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import sqrt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from image_processing import generate_distortion_mappings

NEAR_CLIP = 0.199
FAR_CLIP = 16.0

RGB_WIDTH = 4032
RGB_HEIGHT = 3024
DEPTH_WIDTH = 320
DEPTH_HEIGHT = 240
DEPTH_WINDOW_SIZE = 15

ARUCO_MARKER_SIZE = 0.190

EPS = 10 ** -5

CRUCIAL_VERTEX_IDS = [0, 2, 12, 10]


def sample_depth(depth, coordinate):
    coordinate = [int(i) for i in coordinate]
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
    parser.add_argument('--file', type=str,
                        default='/Users/user/PycharmProjects/DepthCompletion/image_processing/verify_distance_measurements/2.jpg')
    args = parser.parse_args()

    fx, fy, cx, cy = [3269.883, 3268.1606, 2038.7195, 1478.5912]
    intrinsics = np.array([[fx, 0., cx],
                           [0., fy, cy],
                           [0., 0., 1.]])

    distortion = np.array([0.0014165342, -0.035910547, 0., 0., 0.13729763])

    rgb_filename = args.file
    depth_filename = rgb_filename.split('.')[0] + '_z16.png'
    depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED).astype(np.float) * 0.001
    # depth = bm3d(depth, 0.03)

    frame = cv2.imread(rgb_filename)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    mapx, mapy = generate_distortion_mappings(intrinsics, distortion, RGB_WIDTH, RGB_HEIGHT)
    gray = cv2.remap(gray, mapx, mapy, cv2.INTER_CUBIC)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    # Note that we already corrected
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, ARUCO_MARKER_SIZE, intrinsics, np.zeros(5))

    plt.figure()
    plt.imshow(frame_markers)

    aruco_coordinates = []

    for i in range(len(ids)):
        c = corners[i][0]
        plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))
        aruco_coordinates.append(list(tvecs[i][0]))
        aruco_coordinates[-1].append(sqrt(aruco_coordinates[-1][0] ** 2
                                          + aruco_coordinates[-1][1] ** 2
                                          + aruco_coordinates[-1][2] ** 2))

    plt.legend()
    plt.show()

    plt.imshow(depth)

    aruco_depth_sum = 0
    rgbd_depth = []
    crucial_vertex = [None for _ in range(4)]

    for i in range(len(ids)):
        c = corners[i][0]
        point = [round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH)),
                 round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH))]
        rgbd_depth.append(sample_depth(depth, point))
        plt.plot([round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH))],
                 [round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH))], "o", label="id={0}".format(ids[i]))

        if ids[i] in CRUCIAL_VERTEX_IDS:
            for j in range(len(CRUCIAL_VERTEX_IDS)):
                if CRUCIAL_VERTEX_IDS[j] == ids[i]:
                    crucial_vertex[j] = (round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH)),
                                         round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH)))

    plt.legend()
    plt.show()

    print('Coordinates from ARUCO markers')
    for i in range(len(ids)):
        print(ids[i], aruco_coordinates[i])

    print('Depth from RGBD with markers')
    for i in range(len(ids)):
        print(ids[i], rgbd_depth[i])
