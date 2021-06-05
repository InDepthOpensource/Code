import argparse
import copy

import skimage
import os
from os import path
from skimage import io, segmentation, color, graph, future, feature, morphology
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import imageio
from bm3d import bm3d
from skimage.future.graph import RAG
from scipy.ndimage import convolve
from skimage.restoration import inpaint
import time

NEAR_CLIP = 0.05
FAR_CLIP = 8.19
FAR_CLIP_SOFT = 4.5
NEAR_CLIP_SOFT = 1.75
depth_width = 320
depth_height = 240
rgb_width = 320
rgb_height = 240

# Limit number of numpy threads to 1, as more threads do not improve performance
os.environ["OMP_NUM_THREADS"] = "2"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "2"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "2"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "2"  # export NUMEXPR_NUM_THREADS=1


def run_bm3d(self, depth_float):
    bm3d_depth = bm3d(depth_float, self.bm3d_sigma)
    bm3d_depth[bm3d_depth < NEAR_CLIP] = 0.0
    return bm3d_depth


def label2avg(label_field, image):
    out = np.zeros_like(image)
    labels = np.unique(label_field)
    for label in labels:
        mask = (label_field == label).nonzero()
        color = image[mask].mean()
        out[mask] = color
    return out


def filter_by_label(label_field, depth, normalized_depth, edge_map):
    out = depth
    labels = np.unique(label_field)
    for label in labels:
        mask = (label_field == label) & (depth > NEAR_CLIP)
        if np.any((mask == 1)):
            avg_depth = depth[mask].mean()
            avg_normalized = normalized_depth[mask].mean()
            remove = False
            if avg_normalized < 0.20 and avg_depth < NEAR_CLIP_SOFT:
                remove = True
            elif avg_normalized > 0.80 and avg_depth > FAR_CLIP_SOFT:
                remove = True
            conv_kernel_0 = np.array([[-1, -1, -1],
                                      [-1, 8, -1],
                                      [-1, -1, -1]])
            conv_kernel_1 = np.ones((7, 7))
            expanded_edge = convolve(mask, conv_kernel_0)
            expanded_edge[expanded_edge < 1] = 0
            expanded_edge[expanded_edge >= 1] = 1
            expanded_edge = convolve(expanded_edge, conv_kernel_1)
            expanded_edge[expanded_edge < 1] = 0
            expanded_edge[expanded_edge >= 1] = 1
            total_expanded_edge_size = np.sum(expanded_edge)
            canny_edge_total_overlap = np.sum(edge_map[expanded_edge == 1])
            if canny_edge_total_overlap > 0.5 * total_expanded_edge_size:
                remove = False
            if remove and mask.sum() < 100 * 100 and np.all((mask[:, :200] == 0)):
                out[mask] = 0.0
                print('removed!')
    return out


def rag_mean_value(image, labels, connectivity=2):
    graph = RAG(labels, connectivity=connectivity)

    for n in graph:
        graph.nodes[n].update({'labels': [n],
                               'pixel count': 0,
                               'total color': 0.0})

    for index in np.ndindex(labels.shape):
        current = labels[index]
        if image[index] > 0.0:
            graph.nodes[current]['pixel count'] += 1
            graph.nodes[current]['total color'] += image[index]

    for n in graph:
        if graph.nodes[n]['pixel count'] > 0.0:
            graph.nodes[n]['mean color'] = (graph.nodes[n]['total color'] /
                                            graph.nodes[n]['pixel count'])
        else:
            graph.nodes[n]['mean color'] = 0

    for x, y, d in graph.edges(data=True):
        diff = graph.nodes[x]['mean color'] - graph.nodes[y]['mean color']
        diff = np.linalg.norm(diff)
        d['weight'] = diff

    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str,
                        default='/Users/user/Downloads/tof_sequence/tof_captures/test/')
    args = parser.parse_args()

    total_removed_frames = 0

    for file in os.listdir(args.folder):
        if not file.endswith('z16.png'):
            continue

        depth_filename = args.folder + file
        rgb_filename = depth_filename.replace('_z16.png', '_undistorted.jpg')

        if not path.exists(rgb_filename):
            continue

        depth = io.imread(depth_filename) / 1000.0
        # depth = bm3d(depth, 0.03)
        original_depth = copy.deepcopy(depth)

        rgb = io.imread(rgb_filename, as_gray=True)

        depth[depth < (NEAR_CLIP - 0.0001)] = 0.0
        depth[depth > (FAR_CLIP - 0.0001)] = 0.0

        # Heuristics
        if depth.max() > NEAR_CLIP:
            edges = feature.canny(rgb)
            norm = plt.Normalize(vmin=depth[depth > 0.01].min() + 0.005, vmax=depth.max() + 0.01)
            normalized_depth = norm(depth)
            slic_labels = segmentation.slic(depth,
                                            compactness=0.01,
                                            n_segments=256,
                                            multichannel=False,
                                            start_label=1)

            g = rag_mean_value(depth, slic_labels)
            merged_labels = future.graph.cut_threshold(slic_labels, g, 0.15)
            depth = filter_by_label(merged_labels, depth, normalized_depth, edges)

        delta = np.abs(original_depth - depth)
        # delta[delta < 0.01] = 0.0
        if np.sum((delta > 0.01)) < 1000:
            delta = np.zeros_like(delta)
        else:
            print('One artifact frame!')
            total_removed_frames += 1
        imageio.imsave(depth_filename.replace('_z16.png', '_artifact_removed_z16.png'),
                       (depth * 1000).astype(np.uint16))
        imageio.imsave(depth_filename.replace('_z16.png', '_artifact_z16.png'),
                       (delta * 1000).astype(np.uint16))
        cmap = cm.get_cmap(name='jet')
        cmap.set_under('w')
        if depth.min() > 0.01:
            norm = plt.Normalize(vmin=depth[depth > 0.01].min() + 0.005, vmax=depth.max() + 0.05)
        else:
            norm = plt.Normalize(vmin=0.005, vmax=0.05)
        depth_delta_color = cmap(norm(delta))
        # depth_color_segmented = label2avg(slic_labels, depth)
        # plt.imshow(segmentation.mark_boundaries(depth_color[:, :, :3], merged_labels, color=(0, 0, 0)))
        plt.imshow(depth_delta_color)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
        plt.savefig(depth_filename.split('.')[0] + '_artifacts.png')
        plt.clf()

    print(total_removed_frames)
