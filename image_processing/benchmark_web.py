import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# Limit number of numpy threads to 1, as more threads do not improve performance
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import time
import cv2
import numpy as np
import argparse
# import imageio
from io import BytesIO

import torch
from torchvision import transforms
from quart import Quart, request, send_file
from torch.backends import cudnn
from werkzeug.serving import WSGIRequestHandler

from models import CBAMDilatedUNet
from PIL import Image

NEAR_CLIP = 0.01
FAR_CLIP = 16.0
depth_width = 320
depth_height = 240
rgb_width = 320
rgb_height = 240

app = Quart(__name__)


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


@app.route('/hello', methods=['GET'])
async def hello():
    return 'Hello world!'


@app.route('/inpaint', methods=['POST'])
async def handle_inpaint_request():
    depth_file = None
    rgb_file = None
    request_files = await request.files
    for file in request_files.values():
        suffix = file.filename.split('.')[-1]
        if suffix == 'jpg':
            rgb_file = file
        elif suffix == 'png':
            depth_file = file
        else:
            return 'No filename', 400

    print('new frame!')
    start = time.time()
    rgb = cv2.imdecode(np.frombuffer(rgb_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    argb_depth = cv2.imdecode(np.frombuffer(depth_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    lower_8bits = argb_depth[:, :, 1].astype(np.uint32)
    higher_8bits = argb_depth[:, :, 2].astype(np.uint32)
    depth = ((higher_8bits << 8) + lower_8bits) * 0.001
    depth = cv2.resize(depth, (320, 256), interpolation=cv2.INTER_LINEAR)

    time.sleep(0.0172)

    _, completed_depth_buffer = cv2.imencode('.png', (depth * 1000).astype(np.uint16))
    completed_depth_file = BytesIO(completed_depth_buffer)
    end = time.time()
    print('time to read images', end - start)
    return await send_file(completed_depth_file, attachment_filename=depth_file.filename, as_attachment=True)


@app.cli.command('run')
def run():
    print('Init success')
    app.run(host='0.0.0.0',
            port=8080,
            keyfile='/Users/user/PycharmProjects/DepthCompletion/image_processing/key.pem',
            certfile='/Users/user/PycharmProjects/DepthCompletion/image_processing/cert.pem',
            )
# On machines with hypercorn installed, run the following to enable HTTP/2
# hypercorn --certfile image_processing/cert.pem --keyfile image_processing/key.pem --bind 0.0.0.0:8080 -w 6 image_processing:app
