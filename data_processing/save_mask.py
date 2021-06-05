import os
import sys
import imageio
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from experiments.data import MatterportTrainDataset, MatterportEvalDataset

train_dataset = MatterportTrainDataset(use_generated_mask=False)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=16,
                              pin_memory=True,
                              drop_last=True)

pixel_counts = [0 for _ in range(42)]

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
        for j in range(len(color_input)):
            single_mask = valid_mask[j][0][:][:]
            if 320 * 256 * 0.6 < single_mask.sum() < 320 * 256 * 0.85:
                imageio.imwrite('/usr/project/xtmp/user/mask/' + 'mask_' + str(i) + '.png',
                                (single_mask * 255).astype(np.uint8))
                i += 1

        t.update(len(color_input))
    print('Total mask found:', i)

# total_pixels = sum(pixel_counts)
# pixel_counts = [i / total_pixels for i in pixel_counts]
# print('matterport depth distribution', pixel_counts)
