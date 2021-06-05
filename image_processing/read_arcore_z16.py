import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm


if __name__ == '__main__':
    depth_filename = '/Users/user/PycharmProjects/DepthCompletion/image_processing/3_arcore.png'
    argb_depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
    lower_8bits = argb_depth[:, :, 1].astype(np.uint32)
    higher_8bits = argb_depth[:, :, 2].astype(np.uint32)
    depth = ((higher_8bits << 8) + lower_8bits) * 0.001
    imageio.imsave(depth_filename.split('.')[0] + '_z16.png', (depth * 1000).astype(np.uint16))
    cmap = cm.get_cmap(name='jet')
    cmap.set_under('w')
    norm = plt.Normalize(vmin=depth.min() + 0.05, vmax=depth.max())
    plt.imsave(depth_filename.split('.')[0] + '_color.png', cmap(norm(depth)))



