import random

import numpy as np
import torch
import torch.nn as nn
from PIL.Image import Resampling
from pytorch_wavelets import DWTInverse, DWTForward
from torch.utils.data import DataLoader


class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        transform = random.choice(self.transforms)
        return transform(img)


def apply_preprocess(x, scale=4):
    x = x.split()[0]  # Take only the Y channel

    x_lr = x.resize(size=(x.size[0] // scale, x.size[1] // scale), resample=Resampling.BICUBIC)
    x_bic = x_lr.resize(size=(x.size[0], x.size[1]), resample=Resampling.BICUBIC)

    x = np.array(x) / 255.  # Normalize
    x_lr = np.array(x_lr) / 255.
    x_bic = np.array(x_bic) / 255.

    x = torch.tensor(data=x).float()
    x_lr = torch.tensor(data=x_lr).float()
    x_bic = torch.tensor(data=x_bic).float()

    return x, x_lr, x_bic


class WaveletsTransform(nn.Module):

    def __init__(self):
        super().__init__()
        self.transform = DWTForward(wave='haar', mode='zero')

    def forward(self, x):
        wavelets = self.transform(x=x.unsqueeze(1))
        c_a = wavelets[0].squeeze(1)
        c_h = wavelets[1][0].squeeze(1)[:, 0, :, :]
        c_v = wavelets[1][0].squeeze(1)[:, 1, :, :]
        c_d = wavelets[1][0].squeeze(1)[:, 2, :, :]
        return torch.cat(tensors=[c_a, c_h, c_v, c_d]).view(x.shape[0], 4, c_a.shape[1], c_a.shape[2])


class InverseWaveletsTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = DWTInverse(wave='haar', mode='zero')

    def forward(self, x):
        c_a, c_h, c_v, c_d = torch.split(x, split_size_or_sections=1, dim=1)
        coeffs = (c_a, [torch.cat((c_h, c_v, c_d), dim=1).unsqueeze(1)])
        return self.transform(coeffs=coeffs).squeeze(1)


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            import time
            start = time.time()
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
            print(f'InfiniteDataLoader: {time.time() - start:.2f} seconds to reset the iterator')
        return batch
