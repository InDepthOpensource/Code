import os
import sys
import imageio
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from experiments.data import SceneNetRGBDTrainSet, SceneNetRGBDEvalSet

train_dataset = SceneNetRGBDTrainSet(use_generated_mask=False, use_matterport_mask=True)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=16,
                              pin_memory=True,
                              drop_last=True)

total_pixels = 0
valid_pixels = 0

with tqdm(total=(len(train_dataset) - len(train_dataset) % 16)) as t:
    t.set_description('Training epoch: {}/{}'.format(0, 0))

    i = 0
    for data in train_dataloader:
        color_input, depth_input, target = data
        depth_input = depth_input * 16.0

        # for j in range(0, 40):
        #     selected = (j * 0.5 <= target) & (target < j * 0.5 + 0.5)
        #     selected = int(selected.sum())
        #     pixel_counts[j] += selected

        # Valid region will be white, missing region will be black.
        valid_mask = depth_input > 10 ** -5
        valid_mask = valid_mask.cpu().numpy()

        color_numpy = color_input.cpu().numpy()

        total_pixels += depth_input.numel()
        valid_pixels += int(valid_mask.sum())

        for j in range(len(color_input)):
            if i % 200 == 0:
                single_mask = valid_mask[j][0][:][:]
                imageio.imwrite('/usr/project/xtmp/user/scenenet_sample/' + 'mask_' + str(i) + '.png',
                                (single_mask * 255).astype(np.uint8))

                single_image = color_numpy[j][:][:][:]
                single_image = np.moveaxis(single_image, 0, -1)
                imageio.imwrite('/usr/project/xtmp/user/scenenet_sample/' + 'color_' + str(i) + '.jpg', single_image)
            i += 1

        if i > 20000:
            break
        t.update(len(color_input))

print('train percent missing:', 1 - valid_pixels / total_pixels)

train_dataset = SceneNetRGBDEvalSet(use_generated_mask=False, use_matterport_mask=True)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=16,
                              pin_memory=True,
                              drop_last=True)

total_pixels = 0
valid_pixels = 0

with tqdm(total=(len(train_dataset) - len(train_dataset) % 16)) as t:
    t.set_description('Training epoch: {}/{}'.format(0, 0))

    i = 0
    for data in train_dataloader:
        color_input, depth_input, target = data
        depth_input = depth_input * 16.0

        # for j in range(0, 40):
        #     selected = (j * 0.5 <= target) & (target < j * 0.5 + 0.5)
        #     selected = int(selected.sum())
        #     pixel_counts[j] += selected

        # Valid region will be white, missing region will be black.
        valid_mask = depth_input > 10 ** -5
        valid_mask = valid_mask.cpu().numpy()

        color_numpy = color_input.cpu().numpy()

        total_pixels += depth_input.numel()
        valid_pixels += int(valid_mask.sum())

        for j in range(len(color_input)):
            if i % 200 == 0:
                single_mask = valid_mask[j][0][:][:]
                imageio.imwrite('/usr/project/xtmp/user/scenenet_sample/' + 'mask_val_' + str(i) + '.png',
                                (single_mask * 255).astype(np.uint8))

                single_image = color_numpy[j][:][:][:]
                single_image = np.moveaxis(single_image, 0, -1)
                imageio.imsave('/usr/project/xtmp/user/scenenet_sample/' + 'color_val_' + str(i) + '.jpg',
                                single_image)
            i += 1

        if i > 20000:
            break
        t.update(len(color_input))

print('val percent missing:', 1 - valid_pixels / total_pixels)
