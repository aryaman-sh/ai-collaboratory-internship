import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from data_utils import load_nii
import nibabel as nib

"""
BraTS2021 dataset classes
Paths absolute to cluster

Randomly select an index and get index, mask
"""


class TrainData(Dataset):
    def __init__(self, train_samples, base_path, size=128):
        self.base_path = base_path

        self.all_samples = os.listdir(base_path)
        self.all_samples.sort()
        self.all_samples = self.all_samples[:-150]
        random.shuffle(self.all_samples)
        self.all_samples = self.all_samples[:train_samples]

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, item):
        scan_id = self.all_samples[item]

        # Load T1 image
        volume_t1, affine_t1 = load_nii(path=f'{self.base_path}/{scan_id}/{scan_id}_t1.nii.gz')
        random_index = random.randint(0, volume_t1.shape[2])
        # TODO: Normalise image, check for blanks
        volume_t1 = volume_t1[:][:][random_index]

        # Load Mask
        volume_seg, affine_seg = load_nii(path=f'{self.base_path}/{scan_id}/{scan_id}_seg.nii.gz')
        volume_seg = volume_seg[:][:][random_index]


        volume_t1 = torch.from_numpy(volume_t1).unsqueeze(0)
        volume_seg = torch.from_numpy(volume_seg).unsqueeze(0)

        return volume_t1, volume_seg




