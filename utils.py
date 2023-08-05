import random

import numpy as np
import torch
import torch.nn as nn
from PIL.Image import Resampling
from pytorch_wavelets import DWTInverse, DWTForward
from torch.utils.data import DataLoader


def psnr_loss(y_true, y_pred, max_pixel=1.0):
    # assert y_true.shape == y_pred.shape, 'Cannot compute PSNR if two input shapes are not same: %s and %s' % (str(
    #     y_true.shape), str(y_pred.shape))
    mse = torch.mean((y_pred - y_true) ** 2)
    return 10.0 * torch.log10((max_pixel ** 2) / mse)


def ssim_loss(y_true, y_pred):
    # assert y_true.shape == y_pred.shape, 'Cannot compute SSIM if two input shapes are not same: %s and %s' % (str(
    #     y_true.shape), str(y_pred.shape))
    u_true = torch.mean(y_true)
    u_pred = torch.mean(y_pred)
    var_true = torch.var(y_true)
    var_pred = torch.var(y_pred)
    std_true = torch.sqrt(var_true)
    std_pred = torch.sqrt(var_pred)
    c1 = (0.01 * 7) ** 2
    c2 = (0.03 * 7) ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom


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
    x = x.split()[0]  # Y channel

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
        return torch.cat((wavelets[0].unsqueeze(1), wavelets[1][0]), dim=2).squeeze(1)


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
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
