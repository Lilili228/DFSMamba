import os.path
import random
import torchvision.transforms as transforms
import numpy as np
from data.base_dataset import BaseDataset
import os
from data.paired_augmentation import (
    augment_tensor_pair,
    indexed_sample_id,
    paired_crop_and_flip,
    sample_id_hash,
)

class FlirDataset(BaseDataset):
    def initialize(self, opt, test=False):
        print('ThermalDataset')
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        if test:
            self.A_data = np.load(os.path.join(self.root, "grayscale_test_data.npy"))
            self.B_data = np.load(os.path.join(self.root, "thermal_test_data.npy"))
        else:
            self.A_data = np.load(os.path.join(self.root, "grayscale_training_data.npy"))
            self.B_data = np.load(os.path.join(self.root, "thermal_training_data.npy"))

    def __getitem__(self, index):
        A = self.A_data[index]#.transpose(2, 0, 1)
        B = self.B_data[index]#.transpose(2, 0, 1)

        # A = A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A.copy()).float()
        B = transforms.ToTensor()(B.copy()).float()
        A, B, _ = augment_tensor_pair(A, B, self.opt)
        A, B = paired_crop_and_flip(
            A, B, self.opt, legacy_random_crop=True
        )

        A = transforms.Normalize([0.5], [0.5])(A)
        B = transforms.Normalize([0.5], [0.5])(B)



        split = 'test' if not self.opt.isTrain else 'train'
        sample_id = indexed_sample_id('FLIR:' + split, index)
        return {
            'A': A,
            'B': B,
            'A_paths': sample_id,
            'B_paths': sample_id,
            'sample_id': sample_id,
            'sample_id_hash': sample_id_hash(sample_id),
        }

    def __len__(self):
        return len(self.A_data)

    def name(self):
        return 'FLIR DATASET'
