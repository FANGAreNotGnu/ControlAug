import json
import numpy as np
import random

from torch.utils.data import Dataset

import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "../../ControlNet"))
from .utils import imread, resize, random_crop_and_resize_images_to_square


class VisPriorDataset(Dataset):
    def __init__(self, annotation_file):
        self.data = []
        with open(annotation_file, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filepath = item['source']
        target_filepath = item['target']
        prompt = item['prompt']

        source = imread(source_filepath)
        target = imread(target_filepath)

        # TODO: currently it's HED only
        H, W, _ = source.shape
        new_H = H // 64 * 64
        new_W = W // 64 * 64

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        source = resize(img=source, new_W=new_W, new_H=new_H)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        target = resize(img=target, new_W=new_W, new_H=new_H)

        return dict(jpg=target, txt=prompt, hint=source)


class CroppedSquareVisPriorDataset(VisPriorDataset):
    def __init__(self, annotation_file, square_size):
        super().__init__(annotation_file=annotation_file)
        self.square_size = square_size
        assert self.square_size // 64 * 64 == self.square_size

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filepath = item['source']
        target_filepath = item['target']
        prompt = item['prompt']

        source = imread(source_filepath)
        target = imread(target_filepath)

        source, target = random_crop_and_resize_images_to_square(imgs=[source,target],square_size=self.square_size)

        # TODO: currently it's HED only

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
