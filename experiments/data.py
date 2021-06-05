import random
import math
import torch
import numpy as np
import imageio
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw
from bm3d import bm3d
from copy import deepcopy

MATTERPORT_TRAIN_FILE_LIST = '../tmp/train_file_list.txt'
MATTERPORT_EVAL_FILE_LIST = '../tmp/test_file_list.txt'

MATTERPORT_TRAIN_FILE_LIST_NORMAL = '../tmp/train_file_list_normal.txt'
MATTERPORT_TEST_FILE_LIST_NORMAL = '../tmp/test_file_list_normal.txt'

FILTERED_110_MATTERPORT_TRAIN_FILE_LIST = '../tmp/filtered_train_file_list.txt'
FILTERED_110_MATTERPORT_EVAL_FILE_LIST = '../tmp/filtered_test_file_list.txt'

FILTERED_125_MATTERPORT_TRAIN_FILE_LIST = '../tmp/filtered_train_file_list_125.txt'
FILTERED_125_MATTERPORT_EVAL_FILE_LIST = '../tmp/filtered_test_file_list_125.txt'

SAMSUNG_TOF_TRAIN_FILE_LIST = '../tmp/train_samsung_tof.txt'
SAMSUNG_TOF_EVAL_FILE_LIST = '../tmp/test_samsung_tof.txt'

SCENENET_RGBD_TRAIN_FILE_LIST = '/usr/project/xtmp/yz322/scenenet_rgbd_train.txt'
SCENENET_RGBD_EVAL_FILE_LIST = '/usr/project/xtmp/yz322/scenenet_rgbd_val_selected.txt'
SCENENET_RGBD_BASE = '/usr/project/xtmp/yz322/'

MATTERPORT_NYUV2_DIODE_COMBINED = '../tmp/combined_train.txt'

NYUV2_TRAIN_FILE_LIST = '../tmp/nyuv2_train_list.txt'

SAMSUNG_TOF_TRAIN_ARTIFACT_FILE_LIST = '../tmp/train_samsung_tof_artifact.txt'
SAMSUNG_TOF_EVAL_ARTIFACT_FILE_LIST = '../tmp/eval_samsung_tof_artifact.txt'


IMAGENET_PCA = {
    'eigval': torch.tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

NEAR_CLIP = 0.1
NORMAL_EPS = 0.0001


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class MatterportTrainDataset(Dataset):
    @staticmethod
    def generate_mask(H, W, min_mask=1, max_mask=7, min_num_vertex=4, max_num_vertex=8):
        mean_angle = 2 * math.pi / 5
        angle_range = 2 * math.pi / 15
        min_width = 30
        max_width = 50
        average_radius = math.sqrt(H * H + W * W) / 13
        mask = Image.new('L', (W, H), 0)
        for _ in range(np.random.randint(min_mask, max_mask)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))
            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))
            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.bool)
        mask = np.reshape(mask, (H, W))
        return mask

    def __init__(self, file_list_name=MATTERPORT_TRAIN_FILE_LIST_NORMAL, use_generated_mask=True, min_mask=1,
                 max_mask=7, depth_scale_factor=4000.0, random_crop=False, shift_rgb=False, use_normal=False,
                 random_mirror=True, use_samsung_tof_mask=False, random_seed=4935743, gaussian_noise_sigma=0.0,
                 noise_file_list_name=SAMSUNG_TOF_TRAIN_FILE_LIST):
        super().__init__()

        self.use_generated_mask = use_generated_mask and (not use_samsung_tof_mask)
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.depth_scale_factor = depth_scale_factor
        self.random_crop = random_crop
        self.shift_rgb = shift_rgb
        self.use_normal = use_normal
        self.random_mirror = random_mirror
        self.use_samsung_tof_mask = use_samsung_tof_mask
        self.gaussian_noise_sigma = gaussian_noise_sigma

        f = open(file_list_name, 'r')
        self.file_list = [i.split() for i in f.readlines()]
        f.close()

        if self.use_samsung_tof_mask:
            f = open(noise_file_list_name, 'r')
            self.noise_file_list = [i.split() for i in f.readlines()]
            f.close()

        np.random.seed(random_seed)
        random.seed(random_seed)

        self.depth_resize = transforms.Resize((256, 320), interpolation=Image.BILINEAR)
        self.color_resize = transforms.Resize((256, 320), interpolation=Image.BICUBIC)
        self.resize_medium = transforms.Resize((512, 640), interpolation=Image.BICUBIC)

        self.rgb_transforms = transforms.Compose((
            # Inception Color Jitter
            transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2, hue=0.06),
            transforms.ToTensor(),
            # Do not use PCA to change lighting - image looks terrible and out of range
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ))

        self.depth_mask_transforms = transforms.Compose((
            transforms.Resize((256, 320), interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop((256, 320), (0.9, 1.0), (0.8, 1.25)),
            transforms.ToTensor()
        ))

    def __getitem__(self, idx):
        try:
            if len(self.file_list[idx]) == 6:
                color_file, depth_file, render_depth_file, normal_x_file, normal_y_file, normal_z_file = self.file_list[
                    idx]
                color_image = Image.open(color_file)
                depth_image = Image.open(depth_file)
                rendered_depth_image = Image.open(render_depth_file)
            elif len(self.file_list[idx]) == 2:
                color_file, depth_file = self.file_list[idx]
                color_image = Image.open(color_file)
                depth_image = Image.open(depth_file)
                rendered_depth_image = deepcopy(depth_image)
        except Exception as e:
            print('Cannot load frame', e)
            idx = random.randrange(len(self.file_list))
            if len(self.file_list[idx]) == 6:
                color_file, depth_file, render_depth_file, normal_x_file, normal_y_file, normal_z_file = self.file_list[
                    idx]
                color_image = Image.open(color_file)
                depth_image = Image.open(depth_file)
                rendered_depth_image = Image.open(render_depth_file)
            elif len(self.file_list[idx]) == 2:
                color_file, depth_file = self.file_list[idx]
                color_image = Image.open(color_file)
                depth_image = Image.open(depth_file)
                rendered_depth_image = deepcopy(depth_image)

        if self.use_samsung_tof_mask:
            _, samsung_depth_file = self.noise_file_list[random.randrange(0, len(self.noise_file_list))]
            samsung_depth_image = Image.open(samsung_depth_file)
            samsung_depth_image = self.depth_mask_transforms(samsung_depth_image)

        depth_related_images = [depth_image, rendered_depth_image]

        if self.use_normal:
            normal_x_image = Image.open(normal_x_file)
            normal_y_image = Image.open(normal_y_file)
            normal_z_image = Image.open(normal_z_file)
            depth_related_images += [normal_x_image, normal_y_image, normal_z_image]

        if self.random_crop:
            width, height = color_image.size
            crop_factor = random.uniform(0.85, 1)
            cropped_width = math.floor(width * crop_factor)
            cropped_height = math.floor(height * crop_factor)

            left = random.randint(0, width - cropped_width)
            top = random.randint(0, height - cropped_height)

            color_image = color_image.crop((left, top, left + cropped_width, top + cropped_height))

            if self.shift_rgb:
                depth_left = random.randint(math.floor(left * 0.5), min(math.floor(left * 1.5), width - cropped_width))
                depth_top = random.randint(math.floor(top * 0.5), min(math.floor(top * 1.5), height - cropped_height))

                depth_related_images = [
                    i.crop((depth_left, depth_top, depth_left + cropped_width, depth_top + cropped_height)) for i in
                    depth_related_images]
            else:
                depth_related_images = [i.crop((left, top, left + cropped_width, top + cropped_height)) for i in
                                        depth_related_images]

        color_image = self.color_resize(color_image)
        depth_related_images = [self.depth_resize(i) for i in depth_related_images]

        # Randomly flip the images horizontally.
        # Note we need to make a invert normal_x if we mirror an image.
        mirrored = False
        if self.random_mirror and random.random() < 0.5:
            color_image = ImageOps.mirror(color_image)
            depth_related_images = [ImageOps.mirror(i) for i in depth_related_images]
            mirrored = True

        depth_image = np.array(depth_related_images[0])

        if self.gaussian_noise_sigma > 10 ** -6:
            gaussian_noise = np.random.normal(1.0, self.gaussian_noise_sigma, np.array(depth_image).shape)
            depth_image = depth_image * gaussian_noise
            depth_image = depth_image.astype(np.float32)

        if self.use_generated_mask:
            generated_mask = self.generate_mask(256, 320, self.min_mask, self.max_mask)
            depth_image[generated_mask] = 0.0

        depth_image = torch.unsqueeze(torch.from_numpy(depth_image), dim=0)
        rendered_depth_image_and_normals = [torch.unsqueeze(torch.from_numpy(np.array(i)), dim=0) for i in
                                            depth_related_images[1:]]

        with torch.no_grad():
            # Intentionally make the input of the network smaller
            if len(self.file_list[idx]) == 6:
                depth_scale_factor = self.depth_scale_factor
            else:
                depth_scale_factor = 1000.0

            depth_image = depth_image / depth_scale_factor / 16
            rendered_depth_image = rendered_depth_image_and_normals[0] / self.depth_scale_factor
            depth_image[depth_image < NEAR_CLIP / 16.0] = 0.0
            rendered_depth_image[rendered_depth_image < NEAR_CLIP] = 0.0

            # Deal with the three normal maps
            if self.use_normal:
                tensor_normal_x = rendered_depth_image_and_normals[1]
                tensor_normal_y = rendered_depth_image_and_normals[2]
                tensor_normal_z = rendered_depth_image_and_normals[3]
                tensor_normal = torch.cat((tensor_normal_x, tensor_normal_y, tensor_normal_z), 0)

                tensor_normal = tensor_normal / 65535.0 * 2.0 - 1.0
                if mirrored:
                    tensor_normal[0, :, :] = - tensor_normal[0, :, :]

            if self.use_samsung_tof_mask:
                depth_image[samsung_depth_image < NEAR_CLIP] = 0.0

        color_image = self.rgb_transforms(color_image)

        if self.use_normal:
            return color_image, depth_image, rendered_depth_image, tensor_normal
        else:
            return color_image, depth_image, rendered_depth_image,

    def __len__(self):
        return len(self.file_list)


class CombinedTrainDataset(MatterportTrainDataset):
    def __init__(self, file_list_name=MATTERPORT_NYUV2_DIODE_COMBINED, use_generated_mask=False, min_mask=1,
                 max_mask=7, depth_scale_factor=4000.0, random_crop=True, shift_rgb=False, use_normal=False,
                 random_mirror=True, use_samsung_tof_mask=True, random_seed=4935743, gaussian_noise_sigma=0.0,
                 noise_file_list_name=SAMSUNG_TOF_TRAIN_FILE_LIST):
        super().__init__(file_list_name, use_generated_mask, min_mask, max_mask,
                         depth_scale_factor, random_crop, shift_rgb, use_normal,
                         random_mirror, use_samsung_tof_mask, random_seed, gaussian_noise_sigma,
                         noise_file_list_name)


class NYUV2TrainDataset(MatterportTrainDataset):
    def __init__(self, file_list_name=NYUV2_TRAIN_FILE_LIST, use_generated_mask=False, min_mask=1,
                 max_mask=7, depth_scale_factor=4000.0, random_crop=True, shift_rgb=False, use_normal=False,
                 random_mirror=True, use_samsung_tof_mask=True, random_seed=4935743, gaussian_noise_sigma=0.0,
                 noise_file_list_name=SAMSUNG_TOF_TRAIN_FILE_LIST):
        super().__init__(file_list_name, use_generated_mask, min_mask, max_mask,
                         depth_scale_factor, random_crop, shift_rgb, use_normal,
                         random_mirror, use_samsung_tof_mask, random_seed, gaussian_noise_sigma,
                         noise_file_list_name)


class DepthAdaptationDataset(Dataset):
    @staticmethod
    def generate_mask(H, W, min_mask=1, max_mask=7):
        min_num_vertex = 4
        max_num_vertex = 8
        mean_angle = 2 * math.pi / 5
        angle_range = 2 * math.pi / 15
        min_width = 30
        max_width = 50
        average_radius = math.sqrt(H * H + W * W) / 13
        mask = Image.new('L', (W, H), 0)
        for _ in range(np.random.randint(min_mask, max_mask)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))
            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))
            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.bool)
        mask = np.reshape(mask, (H, W))
        return mask

    def __init__(self, file_list_name=MATTERPORT_TRAIN_FILE_LIST_NORMAL, use_generated_mask=True, min_mask=1,
                 max_mask=7, depth_scale_factor=4000.0, random_crop=False, shift_rgb=False, random_mirror=True):
        super().__init__()

        self.use_generated_mask = use_generated_mask
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.depth_scale_factor = depth_scale_factor
        self.random_crop = random_crop
        self.shift_rgb = shift_rgb
        self.random_mirror = random_mirror

        f = open(file_list_name, 'r')
        self.file_list = [i.split() for i in f.readlines()]
        f.close()

        self.depth_resize = transforms.Resize((256, 320), interpolation=Image.BILINEAR)
        self.color_resize = transforms.Resize((256, 320), interpolation=Image.BICUBIC)

        self.rgb_transforms = transforms.Compose((
            # Inception Color Jitter
            transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2, hue=0.06),
            transforms.ToTensor(),
            # Do not use PCA to change lighting - image looks terrible and out of range
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ))

    def __getitem__(self, idx):
        color_file, depth_file, = self.file_list[idx][:2]

        color_image = Image.open(color_file)
        depth_image = Image.open(depth_file)

        if self.random_crop:
            width, height = color_image.size
            crop_factor = random.uniform(0.85, 1)
            cropped_width = math.floor(width * crop_factor)
            cropped_height = math.floor(height * crop_factor)

            left = random.randint(0, width - cropped_width)
            top = random.randint(0, height - cropped_height)

            color_image = color_image.crop((left, top, left + cropped_width, top + cropped_height))

            if self.shift_rgb:
                depth_left = random.randint(math.floor(left * 0.5), min(math.floor(left * 1.5), width - cropped_width))
                depth_top = random.randint(math.floor(top * 0.5), min(math.floor(top * 1.5), height - cropped_height))
                depth_image = depth_image.crop(
                    (depth_left, depth_top, depth_left + cropped_width, depth_top + cropped_height))
            else:
                depth_image = depth_image.crop((left, top, left + cropped_width, top + cropped_height))

        color_image = self.color_resize(color_image)
        depth_image = self.depth_resize(depth_image)

        if self.random_mirror and random.random() < 0.5:
            color_image = ImageOps.mirror(color_image)
            depth_image = ImageOps.mirror(depth_image)

        depth_image = np.array(depth_image)

        if self.use_generated_mask:
            generated_mask = self.generate_mask(256, 320, self.min_mask, self.max_mask)
            depth_image[generated_mask] = 0.0

        depth_image = torch.unsqueeze(torch.from_numpy(depth_image), dim=0)

        with torch.no_grad():
            # Intentionally make the input of the network smaller
            depth_image = depth_image / self.depth_scale_factor / 16
            depth_image[depth_image < NEAR_CLIP / 16.0] = 0.0

        color_image = self.rgb_transforms(color_image)

        return color_image, depth_image

    def __len__(self):
        return len(self.file_list)


class MatterportEvalDataset(MatterportTrainDataset):
    def __init__(self, file_list_name=MATTERPORT_TEST_FILE_LIST_NORMAL, use_normal=False,
                 gaussian_noise_sigma=0.0):
        super().__init__(file_list_name=file_list_name, random_crop=False, shift_rgb=False,
                         use_normal=use_normal, random_mirror=False, use_generated_mask=False,
                         gaussian_noise_sigma=gaussian_noise_sigma)

        self.transforms = transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ))

        self.original_rgb_to_tensor = transforms.Compose((
            transforms.ToTensor(),
        ))

    def __getitem__(self, idx):
        color_file = self.file_list[idx][0]
        color_image = self.color_resize(Image.open(color_file))
        original_color_tensor = self.original_rgb_to_tensor(color_image)

        if self.use_normal:
            color_image, depth_image, rendered_depth_image, tensor_normal = super().__getitem__(idx)
            return color_image, depth_image, rendered_depth_image, original_color_tensor, tensor_normal
        else:
            color_image, depth_image, rendered_depth_image = super().__getitem__(idx)
            return color_image, depth_image, rendered_depth_image, original_color_tensor,


class SamsungToFEvalDataset(MatterportTrainDataset):
    def run_bm3d(self, depth_float):
        bm3d_depth = bm3d(depth_float, self.bm3d_sigma)
        bm3d_depth[bm3d_depth < NEAR_CLIP] = 0.0
        return bm3d_depth

    def __init__(self, file_list_name=SAMSUNG_TOF_EVAL_FILE_LIST, bm3d_sigma=0, use_normal=False):
        super().__init__(file_list_name=file_list_name, random_crop=False, shift_rgb=False,
                         random_mirror=False, use_generated_mask=False)
        self.use_normal = False

        self.transforms = transforms.Compose((
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ))

        self.original_rgb_to_tensor = transforms.Compose((
            transforms.ToTensor(),
        ))

        self.depth_scale_factor = 1000.0

        self.bm3d_sigma = bm3d_sigma

        # Of course we can only fake ground truth normal with samsung ToF datasets.
        self.use_normal = use_normal

    def __getitem__(self, idx):
        color_file, depth_file = self.file_list[idx]

        color_image = Image.open(color_file)
        depth_image = Image.open(depth_file)

        color_image = self.color_resize(color_image)
        depth_image = self.depth_resize(depth_image)

        depth_image = np.array(depth_image)
        depth_image = depth_image / self.depth_scale_factor

        if self.bm3d_sigma > 0.000001:
            depth_image = self.run_bm3d(depth_image)

        depth_image = torch.unsqueeze(torch.from_numpy(depth_image.astype(np.float32)), dim=0)

        with torch.no_grad():
            # Intentionally make the input of the network smaller
            depth_image = depth_image / 16
            depth_image[depth_image < NEAR_CLIP / 16.0] = 0.0

        original_color_image = color_image
        color_image = self.rgb_transforms(color_image)
        original_color_tensor = self.original_rgb_to_tensor(original_color_image)

        if not self.use_normal:
            return color_image, depth_image, depth_image, original_color_tensor
        else:
            return color_image, depth_image, depth_image, original_color_tensor, torch.ones_like(color_image)


class RealSenseEvalDataset(MatterportEvalDataset):
    def __init__(self, file_list_name=FILTERED_110_MATTERPORT_EVAL_FILE_LIST):
        super().__init__()
        self.file_list = [201,
                          202,
                          203,
                          204,
                          205,
                          206,
                          207,
                          208,
                          209,
                          210,
                          211,
                          212,
                          213,
                          214,
                          215,
                          216,
                          217,
                          218,
                          219,
                          220,
                          221,
                          222,
                          223,
                          224,
                          225,
                          226,
                          227,
                          400,
                          401,
                          403,
                          404,
                          405,
                          406]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        color_file, depth_file, render_depth_file = str(self.file_list[idx]) + '_undistorted.jpg', str(
            self.file_list[idx]) + '_z16.png', \
                                                    str(self.file_list[idx]) + '_z16.png'
        color_image = self.color_resize(Image.open(color_file))
        depth_image = self.depth_resize(Image.open(depth_file))
        rendered_depth_image = self.depth_resize(Image.open(render_depth_file))

        depth_image = torch.unsqueeze(torch.from_numpy(np.array(depth_image)), dim=0)
        rendered_depth_image = torch.unsqueeze(torch.from_numpy(np.array(rendered_depth_image)), dim=0)

        with torch.no_grad():
            # Intentionally make the input of the network smaller
            depth_image = depth_image * 0.001 / 16
            rendered_depth_image = rendered_depth_image * 0.001
            depth_image[depth_image < NEAR_CLIP / 16] = 0.0
            rendered_depth_image[rendered_depth_image < NEAR_CLIP] = 0.0

        original_color_image = color_image
        color_image = self.rgb_transforms(color_image)
        return color_image, depth_image, rendered_depth_image, self.original_rgb_to_tensor(original_color_image)


class SceneNetRGBDTrainSet(MatterportTrainDataset):
    MAX_MATTERPORT_MASK_INDEX = 17090
    MATTERPORT_MASK_ROOT = '/dev/shm/yz322_mask/'

    def __init__(self, file_list_name=SCENENET_RGBD_TRAIN_FILE_LIST, use_generated_mask=False,
                 use_matterport_mask=True):
        # Not a typo here - first init with shorter MATTERPORT_EVAL_FILE_LIST and then replace with SceneNet list.
        super().__init__(FILTERED_110_MATTERPORT_EVAL_FILE_LIST, use_generated_mask)

        self.use_generated_mask = use_generated_mask
        self.use_matterport_mask = use_matterport_mask

        f = open(file_list_name, 'r')
        self.file_list = [i.strip() for i in f.readlines()]
        f.close()

        self.mask_transforms = transforms.RandomAffine(degrees=90,
                                                       translate=(0.5, 0.5),
                                                       scale=(0.8, 1.2),
                                                       shear=(-0.5, 0.5, -0.5, 0.5),
                                                       resample=Image.BILINEAR,
                                                       fillcolor=255)

    def __getitem__(self, idx):
        color_file = SCENENET_RGBD_BASE + self.file_list[idx]
        depth_file = color_file.replace('photo', 'depth').replace('jpg', 'png')

        color_image = Image.open(color_file)
        depth_image = Image.open(depth_file)

        # Each side must be divisible by 32, so we first crop the image to 300 by 240 and then resize it to 320 by 256.
        left_start = random.randrange(0, 320 - 300)
        color_image = color_image.crop((left_start, 0, left_start + 300, 240))
        depth_image = depth_image.crop((left_start, 0, left_start + 300, 240))
        color_image = self.color_resize(color_image)
        depth_image = self.depth_resize(depth_image)
        rendered_depth_image = depth_image.copy()

        # Randomly flip the images horizontally.
        if random.random() < 0.5:
            color_image = ImageOps.mirror(color_image)
            depth_image = ImageOps.mirror(depth_image)
            rendered_depth_image = ImageOps.mirror(rendered_depth_image)

        depth_image = np.array(depth_image)

        if self.use_generated_mask:
            generated_mask = self.generate_mask(256, 320, 2)
            depth_image[generated_mask] = 0.0

        if self.use_matterport_mask:
            total_mask_count = random.randint(1, 3)
            for _ in range(total_mask_count):
                selected_index = random.randint(0, self.MAX_MATTERPORT_MASK_INDEX)
                mask_filename = self.MATTERPORT_MASK_ROOT + 'mask_' + str(selected_index) + '_z16.png'
                mask_image = Image.open(mask_filename)
                mask_image = self.mask_transforms(mask_image)
                if random.random() < 0.5:
                    mask_image = ImageOps.mirror(mask_image)
                if random.random() < 0.5:
                    mask_image = ImageOps.flip(mask_image)
                tmp_mask = np.array(mask_image)
                # The "white" part are valid regions
                tmp_mask = tmp_mask > 200
                depth_image[~tmp_mask] = 0.0

        depth_image = torch.unsqueeze(torch.from_numpy(depth_image), dim=0)
        rendered_depth_image = torch.unsqueeze(torch.from_numpy(np.array(rendered_depth_image)), dim=0)

        with torch.no_grad():
            # Intentionally make the input of the network smaller
            depth_image = depth_image / 1000.0 / 16
            rendered_depth_image = rendered_depth_image / 1000.0
            depth_image[depth_image < NEAR_CLIP / 16] = 0.0
            rendered_depth_image[rendered_depth_image < NEAR_CLIP] = 0.0

        color_image = self.rgb_transforms(color_image)

        return color_image, depth_image, rendered_depth_image


class SceneNetRGBDEvalSet(SceneNetRGBDTrainSet):
    def __init__(self, file_list_name=SCENENET_RGBD_EVAL_FILE_LIST, use_generated_mask=False, use_matterport_mask=True):
        super().__init__(file_list_name, use_generated_mask, use_matterport_mask)
        self.rgb_transforms = transforms.Compose((
            transforms.ToTensor(),
            # Do not use PCA to change lighting - image looks terrible and out of range
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ))

        self.original_rgb_to_tensor = transforms.Compose((
            transforms.ToTensor(),
        ))

    def __getitem__(self, idx):
        color_file = SCENENET_RGBD_BASE + self.file_list[idx]
        depth_file = color_file.replace('photo', 'depth').replace('jpg', 'png')

        color_image = Image.open(color_file)
        depth_image = Image.open(depth_file)

        # Each side must be divisible by 32, so we first crop the image to 300 by 240 and then resize it to 320 by 256.
        left_start = 10
        color_image = color_image.crop((left_start, 0, left_start + 300, 240))
        depth_image = depth_image.crop((left_start, 0, left_start + 300, 240))
        color_image = self.color_resize(color_image)
        depth_image = self.depth_resize(depth_image)
        rendered_depth_image = depth_image.copy()

        depth_image = np.array(depth_image)

        if self.use_generated_mask:
            generated_mask = self.generate_mask(256, 320, 2)
            depth_image[generated_mask] = 0.0

        # Idea: we want a deterministic mask, so we pick the first one from 0 - 7999,
        # and the second one from 8000 - 15999
        if self.use_matterport_mask:
            # First
            selected_index = idx % 8000
            mask_filename = self.MATTERPORT_MASK_ROOT + 'mask_' + str(selected_index) + '_z16.png'
            mask_image = Image.open(mask_filename)
            tmp_mask = np.array(mask_image)
            # The "white" part are valid regions
            tmp_mask = tmp_mask > 200
            depth_image[~tmp_mask] = 0.0

            # Second
            selected_index += 8000
            mask_filename = self.MATTERPORT_MASK_ROOT + 'mask_' + str(selected_index) + '_z16.png'
            mask_image = Image.open(mask_filename)
            tmp_mask = np.array(mask_image)
            # The "white" part are valid regions
            tmp_mask = tmp_mask > 200
            depth_image[~tmp_mask] = 0.0

        depth_image = torch.unsqueeze(torch.from_numpy(depth_image), dim=0)
        rendered_depth_image = torch.unsqueeze(torch.from_numpy(np.array(rendered_depth_image)), dim=0)

        with torch.no_grad():
            # Intentionally make the input of the network smaller
            depth_image = depth_image / 1000.0 / 16
            rendered_depth_image = rendered_depth_image / 1000.0
            depth_image[depth_image < NEAR_CLIP / 16] = 0.0
            rendered_depth_image[rendered_depth_image < NEAR_CLIP] = 0.0

        original_color_image = color_image
        color_image = self.rgb_transforms(color_image)

        return color_image, depth_image, rendered_depth_image, self.original_rgb_to_tensor(original_color_image)


class MatterportTrainAddedNoise(MatterportTrainDataset):
    # Do not support use normal for now.
    def __init__(self, file_list_name=MATTERPORT_TRAIN_FILE_LIST_NORMAL, use_generated_mask=True, min_mask=1,
                 max_mask=7, depth_scale_factor=4000.0, random_crop=False, shift_rgb=False,
                 use_normal=False, random_mirror=True, random_seed=329423,
                 artifact_file_list_name=SAMSUNG_TOF_TRAIN_ARTIFACT_FILE_LIST):
        super().__init__(file_list_name, use_generated_mask, min_mask, max_mask,
                         depth_scale_factor, random_crop, shift_rgb, False, random_mirror)
        f = open(artifact_file_list_name, 'r')
        self.artifact_file_list = [i.split() for i in f.readlines()]
        f.close()
        np.random.seed(random_seed)
        random.seed(random_seed)

    def __getitem__(self, idx):
        color_image, depth_image, rendered_depth_image = super().__getitem__(idx)
        _, samsung_depth_file = self.artifact_file_list[random.randrange(0, len(self.noise_file_list))]

        samsung_depth_image = Image.open(samsung_depth_file)

        samsung_depth_image = self.depth_resize(samsung_depth_image)
        samsung_depth_image_np = np.array(samsung_depth_image, dtype=np.single) / 1000.0
        samsung_depth_image = torch.unsqueeze(torch.from_numpy(samsung_depth_image_np), dim=0)
        with torch.no_grad():
            # Intentionally make the input of the network smaller
            samsung_depth_image = samsung_depth_image / 16
            samsung_depth_image[samsung_depth_image < NEAR_CLIP / 16.0] = 0.0

        noise_mask = np.zeros_like(samsung_depth_image_np)
        noise_mask[60:180, 220:] = 1
        noise_mask[(samsung_depth_image_np > 1.8) & (samsung_depth_image_np < 5.5)] = 0
        noise_mask[samsung_depth_image_np < 0.01] = 1
        noise_mask = torch.unsqueeze(torch.from_numpy(noise_mask), dim=0)

        depth_image[noise_mask > 0] = samsung_depth_image[noise_mask > 0]

        return color_image, depth_image, rendered_depth_image


class MatterportEvalAddedNoise(MatterportEvalDataset):
    def __init__(self, file_list_name=MATTERPORT_TEST_FILE_LIST_NORMAL, random_seed=329423,
                 artifact_file_list_name=SAMSUNG_TOF_EVAL_ARTIFACT_FILE_LIST):
        super().__init__(file_list_name, False)
        f = open(artifact_file_list_name, 'r')
        self.artifact_file_list = [i.split() for i in f.readlines()]
        f.close()
        np.random.seed(random_seed)
        random.seed(random_seed)

    def __getitem__(self, idx):
        color_image, depth_image, rendered_depth_image, original_color_tensor = super().__getitem__(idx)
        _, samsung_depth_file = self.artifact_file_list[random.randrange(0, len(self.noise_file_list))]

        samsung_depth_image = Image.open(samsung_depth_file)

        samsung_depth_image = self.depth_resize(samsung_depth_image)
        samsung_depth_image_np = np.array(samsung_depth_image, dtype=np.single) / 1000.0
        samsung_depth_image = torch.unsqueeze(torch.from_numpy(samsung_depth_image_np), dim=0)
        with torch.no_grad():
            # Intentionally make the input of the network smaller
            samsung_depth_image = samsung_depth_image / 16
            samsung_depth_image[samsung_depth_image < NEAR_CLIP / 16.0] = 0.0

        noise_mask = np.zeros_like(samsung_depth_image_np)
        noise_mask[60:180, 220:] = 1
        noise_mask[(samsung_depth_image_np > 1.8) & (samsung_depth_image_np < 5.5)] = 0
        noise_mask[samsung_depth_image_np < 0.01] = 1
        noise_mask = torch.unsqueeze(torch.from_numpy(noise_mask), dim=0)

        depth_image[noise_mask > 0] = samsung_depth_image[noise_mask > 0]

        return color_image, depth_image, rendered_depth_image, original_color_tensor
