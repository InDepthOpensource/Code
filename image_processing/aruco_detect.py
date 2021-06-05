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
RGB_HEIGHT = 3024
DEPTH_WIDTH = 320
DEPTH_HEIGHT = 240
PREDICTION_WIDTH = 320
PREDICTION_HEIGHT = 256
DEPTH_WINDOW_SIZE = 5

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
                        default='/Users/user/PyCharmProjects/DepthCompletion/image_processing/aruco_marker_samples/')
    parser.add_argument('--marker-index', nargs='+', type=int, default=[i + 100 for i in range(1, 28)])
    parser.add_argument('--no-marker-index', nargs='+', type=int, default=[i + 200 for i in range(1, 28)])
    parser.add_argument('--predicted-index', nargs='+', type=int, default=[i for i in range(0, 27)])
    args = parser.parse_args()

    fx, fy, cx, cy = [3269.883, 3268.1606, 2038.7195, 1478.5912]
    intrinsics = np.array([[fx, 0., cx],
                           [0., fy, cy],
                           [0., 0., 1.]])

    distortion = np.array([0.0014165342, -0.035910547, 0., 0., 0.13729763])

    aruco_depth = []
    marker_depth = []
    no_marker_depth = []
    predicted_depth = []

    print(args.predicted_index)

    mapx, mapy = generate_distortion_mappings(intrinsics, distortion, RGB_WIDTH, RGB_HEIGHT)

    for index in range(len(args.marker_index)):
        marker_index = args.marker_index[index]
        no_marker_index = args.no_marker_index[index]
        predicted_index = args.predicted_index[index]

        print('Analyzing pair', marker_index, ',', no_marker_index, 'and', predicted_index)

        frame = cv2.imread(args.folder + str(marker_index) + '.jpg')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Remap the original image to a new image
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

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

        depth_marker = cv2.imread(args.folder + str(marker_index) + '_z16.png', cv2.IMREAD_UNCHANGED)
        depth_marker = depth_marker * 0.001
        depth_no_marker = cv2.imread(args.folder + str(no_marker_index) + '_z16.png', cv2.IMREAD_UNCHANGED)
        # depth_no_marker = cv2.inpaint(depth_no_marker, ((depth_no_marker < EPS) * 1).astype(np.uint8), 50, cv2.INPAINT_TELEA)
        depth_no_marker = depth_no_marker * 0.001
        depth_predicted = cv2.imread(args.folder + str(predicted_index) + '_predicted_depth_z16.png', cv2.IMREAD_UNCHANGED)
        depth_predicted = depth_predicted * 0.001

        plt.imshow(depth_marker)
        for i in range(len(ids)):
            c = corners[i][0]
            point = [int(round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH))),
                     int(round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH)))]
            marker_depth.append(sample_depth(depth_marker, point))
            no_marker_depth.append(sample_depth(depth_no_marker, point))
            plt.plot([round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH))],
                     [round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH))], "o", label="id={0}".format(ids[i]))
            point = [int(round(c[:, 1].mean() * (PREDICTION_HEIGHT / RGB_HEIGHT))),
                     int(round(c[:, 0].mean() * (PREDICTION_WIDTH / RGB_WIDTH)))]
            predicted_depth.append(sample_depth(depth_predicted, point))
        plt.legend()
        plt.show()

        print('Depth from RGBD with markers')
        for i in range(len(ids)):
            print(ids[i], marker_depth[i])

        print('Depth from RGBD with no markers')
        for i in range(len(ids)):
            print(ids[i], no_marker_depth[i])

        print('Depth from RGBD prediction')
        for i in range(len(ids)):
            print(ids[i], predicted_depth[i])

    aruco_depth = np.array(aruco_depth)
    marker_depth = np.array(marker_depth)
    no_marker_depth = np.array(no_marker_depth)
    predicted_depth = np.array(predicted_depth)

    no_marker_error = np.abs(no_marker_depth[no_marker_depth > 0] - aruco_depth[no_marker_depth > 0])
    predicted_error = np.abs(predicted_depth[predicted_depth > 0] - aruco_depth[predicted_depth > 0])

    print('no marker error', np.mean(no_marker_error))
    print('predicted error', np.mean(predicted_error))

    print('no marker mean', np.mean(aruco_depth[no_marker_depth > 0]))
    print('predicted mean', np.mean(aruco_depth[predicted_depth > 0]))
