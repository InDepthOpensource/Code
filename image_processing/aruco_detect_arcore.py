import numpy as np
import cv2, PIL
from cv2 import aruco
import matplotlib.pyplot as plt
import math
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from image_processing import generate_distortion_mappings

RGB_WIDTH = 4032
RGB_HEIGHT = 2268
DEPTH_WIDTH = 160
DEPTH_HEIGHT = 90
DEPTH_WINDOW_SIZE = 3

EPS = 10 ** -5


def sample_depth(depth, coordinate):
    depth_window = depth[max(coordinate[0] - DEPTH_WINDOW_SIZE // 2, 0): min(coordinate[0] + DEPTH_WINDOW_SIZE // 2 + 1, DEPTH_HEIGHT),
                            max(coordinate[1] - DEPTH_WINDOW_SIZE // 2, 0): min(coordinate[1] + DEPTH_WINDOW_SIZE // 2 + 1, DEPTH_WIDTH)]

    valid_elements = (depth_window > EPS).sum()
    if valid_elements < 0.6 * DEPTH_WINDOW_SIZE ** 2:
        return 0.0
    else:
        return float(depth_window.sum() / valid_elements)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str,
                        default='/Users/user/PyCharmProjects/DepthCompletion/image_processing/')
    parser.add_argument('--index', nargs='+', type=int, default=[i for i in range(1, 4)])
    args = parser.parse_args()

    fx, fy, cx, cy = [3269.883, 3268.1606, 2038.7195, 1100.5912]
    intrinsics = np.array([[fx, 0., cx],
                           [0., fy, cy],
                           [0., 0., 1.]])

    distortion = np.array([0.0014165342, -0.035910547, 0., 0., 0.13729763])

    aruco_depth = []
    arcore_depth_samples = []

    print(args.index)

    for index in args.index:
        print('Analyzing pair', index)

        frame = cv2.imread(str(index) + '_arcore.jpg')
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

        plt.figure()
        plt.imshow(frame_markers)
        for i in range(len(ids)):
            c = corners[i][0]
            plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))
        plt.legend()
        plt.show()

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.195, intrinsics, np.zeros(5))

        print('Depth calculated from Aruco')
        for i in range(len(ids)):
            print(ids[i], tvecs[i][0][-1])
            aruco_depth.append(tvecs[i][0][-1])

        arcore_depth = cv2.imread(str(index) + '_arcore_z16.png', cv2.IMREAD_UNCHANGED)
        arcore_depth = arcore_depth * 0.001

        plt.imshow(arcore_depth)
        for i in range(len(ids)):
            c = corners[i][0]
            point = [round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH)), round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH))]
            arcore_depth_samples.append(sample_depth(arcore_depth, point))
            plt.plot([round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH))], [round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH))], "o", label="id={0}".format(ids[i]))
        plt.legend()
        plt.show()

        print('Depth from ARCore')
        for i in range(len(ids)):
            print(ids[i], arcore_depth_samples[i])

    aruco_depth = np.array(aruco_depth)
    arcore_depth_samples = np.array(arcore_depth_samples)
    arcore_depth_error = np.abs(arcore_depth_samples[arcore_depth_samples > 0] - aruco_depth[arcore_depth_samples > 0])

    print('arcore depth error', np.mean(arcore_depth_error))

    print('aruco mean', np.mean(aruco_depth[arcore_depth_samples > 0]))
    print('arcore mean', np.mean(arcore_depth_samples[arcore_depth_samples > 0]))
