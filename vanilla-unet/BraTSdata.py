"""
Data utils for BraTS 2021 data
"""
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
import random


class BraTStrainingNoBlank(Dataset):
    """
    Load non blank segmentation masks
    """

    def __init__(self, data_path, no_adjacent_slices=0, val_offset=150):
        """
        Init dataloader
        :param data_path: Path to dataset (all cases)
        :param no_adjacent_slices: Number of adjacent slices for image
        :param val_offset: Scans offset, offset used for val scores
        """
        self.base_path = data_path
        self.all_samples = os.listdir(self.base_path)
        self.all_samples.sort()
        self.all_samples = self.all_samples[:-val_offset]
        random.shuffle(self.all_samples)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        patient_id = self.all_samples[idx]

        volume_t1 = self._load_nii(path=f'{self.base_path}/{patient_id}/{patient_id}_t1.nii.gz')
        volume_mask = self._load_nii(path=f'{self.base_path}/{patient_id}/{patient_id}_seg.nii.gz')

        # Normalise volume
        volume_t1 = (volume_t1 - np.min(volume_t1)) / (np.max(volume_t1) - np.min(volume_t1))
        volume_mask = (volume_mask - np.min(volume_mask)) / (np.max(volume_mask) - np.min(volume_mask))
        volume_t1, volume_mask = self._identify_mask(volume_t1, volume_mask, no_adjacent_slices=1)
        volume_t1 = torch.from_numpy(volume_t1)
        volume_mask = torch.from_numpy(volume_mask)
        volume_mask = volume_mask.unsqueeze(0)

        return volume_t1, volume_mask

    @staticmethod
    def _load_nii(path, dtype="float32", primary_axis=2):
        """
        Loads give nii image
        :param path: path to image
        :param primary_axis: Primary axis (the one to slice along)
        :return: numpy image array
        """
        data = nib.load(path, keep_file_open=False)
        volume = data.get_fdata(caching='unchanged')
        if volume.ndim == 4:
            volume = volume.squeeze(-1)

        volume = volume.astype(np.dtype(dtype))

        # move primary axis to first dimension
        volume = np.moveaxis(volume, primary_axis, 0)

        return volume

    @staticmethod
    def _identify_mask(volume_img, volume_mask, no_adjacent_slices=1):
        """
        Identify non empty mask and corresponding scanning region
        If no segmentation mask exists return random slice(s)
        :param volume_img: Image array
        :param volume_mask: Seg mask array
        :return: Ready to segment masks
        """
        # Get all mask indices with a valid seg mask
        non_blank_indices = []
        for i in range(0, volume_mask.shape[0]):
            if len(np.unique(volume_mask[i])) == 1:
                pass
            else:
                non_blank_indices.append(i)

        if non_blank_indices:
            # Non empty
            index_random = random.choice(non_blank_indices)
            volume_img = volume_img[index_random - no_adjacent_slices:index_random + 1 + no_adjacent_slices]
            volume_mask = volume_mask[index_random]
        else:
            # empty
            index_random = math.floor(volume_mask.shape[0]//2)
            volume_img = volume_img[index_random - no_adjacent_slices:index_random + 1 + no_adjacent_slices]
            volume_mask = volume_mask[index_random]

        return volume_img, volume_mask