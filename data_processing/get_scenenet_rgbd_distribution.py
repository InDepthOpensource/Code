import os
import sys
import imageio
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from experiments.data import SceneNetRGBDTrainSet, SceneNetRGBDEvalSet

train_dataset = SceneNetRGBDTrainSet()
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

        for j in range(0, 40):
            selected = (j * 0.5 <= target) & (target < j * 0.5 + 0.5)
            selected = int(selected.sum())
            pixel_counts[j] += selected

        i += 1
        t.update(len(color_input))
        if i % 10000 == 0:
            print('Total mask found:', i)
            total_pixels = sum(pixel_counts)
            pixel_counts = [i / total_pixels for i in pixel_counts]
            print('scenenet depth distribution', pixel_counts)