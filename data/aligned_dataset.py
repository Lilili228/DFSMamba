import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from data.paired_augmentation import (
    augment_pil_pair,
    make_sample_id,
    paired_crop_and_flip,
    sample_id_hash,
)


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        split = AB.size[0] // 2
        A_image = AB.crop((0, 0, split, AB.size[1]))
        B_image = AB.crop((split, 0, AB.size[0], AB.size[1]))
        A_image, B_image = augment_pil_pair(A_image, B_image, self.opt)
        A = transforms.ToTensor()(A_image)
        B = transforms.ToTensor()(B_image)
        A, B = paired_crop_and_flip(
            A,
            B,
            self.opt,
            legacy_flip=True,
            legacy_random_crop=True,
        )

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        sample_id = make_sample_id(AB_path, root=self.root, prefix='aligned')
        return {
            'A': A,
            'B': B,
            'A_paths': AB_path,
            'B_paths': AB_path,
            'sample_id': sample_id,
            'sample_id_hash': sample_id_hash(sample_id),
        }

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
