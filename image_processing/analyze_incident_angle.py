import argparse
import cv2
import sys
from cv2 import aruco
import imageio
import os
from os import path
from math import acos, pi, floor

import scipy.linalg
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from bm3d import bm3d
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from image_processing import generate_distortion_mappings

NEAR_CLIP = 0.199
FAR_CLIP = 16.0

RGB_WIDTH = 4032
RGB_HEIGHT = 3024
DEPTH_WIDTH = 320
DEPTH_HEIGHT = 240
DEPTH_WINDOW_SIZE = 7

ARUCO_MARKER_SIZE = 0.180

EPS = 10 ** -5

CRUCIAL_VERTEX_IDS = [0, 2, 12, 10]


# Input is pixel coordinate (x, y) and the plane fit result C.
# Output is intersection location (x, y, z), incident_angle.
# If the plane and the line is parallel, return None, None.
def ray_cast(x, y, C):
    depth_fx, depth_fy, depth_cx, depth_cy = [280.44355, 280.17345, 158.51317, 119.690275]
    line_directional_vector = np.array([(x - depth_cx) / depth_fx, (y - depth_cy) / depth_fy, 1])
    plane_normal = np.array([-C[0], -C[1], 1])

    # the plane and the line is parallel
    if np.abs(np.dot(plane_normal, line_directional_vector)) < EPS:
        return None, None

    intersect_z = C[2] / (-C[0] * line_directional_vector[0] - C[1] * line_directional_vector[1] + 1)
    intersection_point = intersect_z * line_directional_vector
    cosine_incident_angle = np.abs(np.dot(line_directional_vector, plane_normal)) / (
            np.linalg.norm(plane_normal) * np.linalg.norm(line_directional_vector))
    incident_angle = acos(cosine_incident_angle) / pi * 180

    return intersection_point, incident_angle


def sample_depth(depth, coordinate):
    depth_window = depth[max(coordinate[0] - DEPTH_WINDOW_SIZE // 2, 0): min(coordinate[0] + DEPTH_WINDOW_SIZE // 2 + 1,
                                                                             DEPTH_HEIGHT),
                   max(coordinate[1] - DEPTH_WINDOW_SIZE // 2, 0): min(coordinate[1] + DEPTH_WINDOW_SIZE // 2 + 1,
                                                                       DEPTH_WIDTH)]

    valid_elements = (depth_window > EPS).sum()
    if valid_elements < 0.75 * DEPTH_WINDOW_SIZE ** 2:
        return 0.0
    else:
        return float(depth_window.sum() / valid_elements)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str,
                        default='/Users/user/PyCharmProjects/DepthCompletion/image_processing/analyze_incident_angle/')
    args = parser.parse_args()

    fx, fy, cx, cy = [3269.883, 3268.1606, 2038.7195, 1478.5912]
    intrinsics = np.array([[fx, 0., cx],
                           [0., fy, cy],
                           [0., 0., 1.]])

    distortion = np.array([0.0014165342, -0.035910547, 0., 0., 0.13729763])

    rgb_filenames = []
    for file in os.listdir(args.folder):
        if file.endswith('.jpg') and (not file.endswith('_undistorted.jpg')):
            rgb_filenames.append(os.path.join(args.folder, file))

    total_incident_count = [0 for _ in range(30)]
    valid_incident_count = [0 for _ in range(30)]

    for rgb_filename in rgb_filenames:
        depth_filename = rgb_filename.split('.')[0] + '_z16.png'
        depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED).astype(np.float) * 0.001

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
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, ARUCO_MARKER_SIZE, intrinsics, np.zeros(5))

        # plt.figure()
        # plt.imshow(frame_markers)

        marker_coordinates = []

        for i in range(len(ids)):
            c = corners[i][0]
            # plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "o", label="id={0}".format(ids[i]))
            marker_coordinates.append((tvecs[i][0][0], tvecs[i][0][1], tvecs[i][0][2]))

        # plt.legend()
        # plt.show()

        # plt.imshow(depth)

        crucial_vertex = [None for _ in range(4)]
        for i in range(len(ids)):
            c = corners[i][0]
            point = [round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH)),
                     round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH))]
            # plt.plot([round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH))],
            #          [round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH))], "o", label="id={0}".format(ids[i]))

            if ids[i] in CRUCIAL_VERTEX_IDS:
                for j in range(len(CRUCIAL_VERTEX_IDS)):
                    if CRUCIAL_VERTEX_IDS[j] == ids[i]:
                        crucial_vertex[j] = (round(c[:, 0].mean() * (DEPTH_WIDTH / RGB_WIDTH)),
                                             round(c[:, 1].mean() * (DEPTH_WIDTH / RGB_WIDTH)))

        # plt.legend()
        # plt.show()

        print('The file is', rgb_filename)

        print('Coordinates from ARUCO markers')
        for i in range(len(ids)):
            print(ids[i], marker_coordinates[i])

        mask = np.zeros_like(depth)
        if not (None in crucial_vertex):
            mask_image = Image.new('L', (DEPTH_WIDTH, DEPTH_HEIGHT), 0)
            ImageDraw.Draw(mask_image).polygon(crucial_vertex, outline=1, fill=1)
            mask = np.array(mask_image)
            # plt.imshow(mask)
            # plt.show()
        else:
            print(rgb_filename, ' does not have all four corners detected')

        marker_coordinates = np.array(marker_coordinates)
        # Least square fit linear plane
        A = np.c_[marker_coordinates[:, 0], marker_coordinates[:, 1], np.ones(marker_coordinates.shape[0])]
        # coefficients
        C, _, _, _ = scipy.linalg.lstsq(A, marker_coordinates[:, 2])

        # plot points and fitted surface
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # X, Y = np.meshgrid(np.arange(-5.0, 5.0, 0.5), np.arange(-5.0, 5.0, 0.5))
        # Z = C[0] * X + C[1] * Y + C[2]
        # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        # ax.scatter(marker_coordinates[:, 0], marker_coordinates[:, 1], marker_coordinates[:, 2], c='r', s=50)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.axis('tight')
        # plt.show()

        for y in range(depth.shape[0]):
            for x in range(depth.shape[1]):
                if mask[y, x] > 0:
                    intersection_point, incident_angle = ray_cast(x, y, C)
                    if incident_angle is not None:
                        total_incident_count[floor((incident_angle + EPS) / 3)] += 1
                        if sample_depth(depth, (y, x)) > EPS:
                            valid_incident_count[floor((incident_angle + EPS) / 3)] += 1
        print('-' * 30)

    total_incident_count = np.array(total_incident_count)
    valid_incident_count = np.array(valid_incident_count)
    total_incident_count[total_incident_count == 0] = 1
    percent_valid_by_incident_angle = valid_incident_count / total_incident_count

    print('total_incident_count', total_incident_count)
    print('valid_incident_count', valid_incident_count)

    print(list(np.linspace(0, 90, 30)))
    print(list(percent_valid_by_incident_angle))

    plt.scatter(np.linspace(0, 90, 30), percent_valid_by_incident_angle)
    plt.show()
