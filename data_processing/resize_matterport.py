import argparse
import imageio
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ResizeDatasetHelper(Dataset):
    def __init__(self, file_list_name):
        super().__init__()

        f = open(file_list_name, 'r')
        self.file_list = [i.split() for i in f.readlines()]
        f.close()

        self.depth_resize = transforms.Resize((512, 640), interpolation=Image.BILINEAR)
        self.color_resize = transforms.Resize((512, 640), interpolation=Image.BICUBIC)

    def __getitem__(self, idx):
        color_file, depth_file, render_depth_file, _, _, _ = self.file_list[idx]

        rgb_image = Image.open(color_file.replace('_small.jpg', '.jpg'))
        depth_16_files = [depth_file, render_depth_file]
        depth_16_images = [Image.open(i.replace('_medium.png', '.png')) for i in depth_16_files]

        rgb_image = self.color_resize(rgb_image)
        depth_16_images = [self.depth_resize(i) for i in depth_16_images]

        rgb_image.save(color_file, quality=95)
        for i in range(len(depth_16_images)):
            np_image = np.array(depth_16_images[i], dtype=np.uint16)
            imageio.imsave(depth_16_files[i], np_image)
        return torch.tensor(0)

    def __len__(self):
        return len(self.file_list)


def write_file_list(filename, file_list):
    f = open(filename, 'w+')
    for i in file_list:
        print(' '.join(i), file=f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-list', type=str, default='../tmp/train_file_list_normal_small.txt')
    args = parser.parse_args()

    train_dataset = ResizeDatasetHelper(file_list_name=args.file_list)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=12,
                                  shuffle=False,
                                  num_workers=12,
                                  pin_memory=True,
                                  drop_last=False)
    i = 0
    for data in train_dataloader:
        i += 1
        print('batch', i)

    print('Total processed batches', i)

    f = open(args.file_list, 'r')
    file_list = [i.split() for i in f.readlines()]
    f.close()

    for i in range(len(file_list)):
        line = file_list[i]
        new_color_name = [line[0].replace('.jpg', '_small.jpg')]
        new_depth_16_names = [j.replace('.png', '_medium.png') for j in line[1:]]
        file_list[i] = new_color_name + new_depth_16_names

    write_file_list(args.file_list.replace('.txt', '_small.txt'), file_list)

# python3 resize_matterport.py --file-list=../tmp/train_file_list_normal_small.txt
