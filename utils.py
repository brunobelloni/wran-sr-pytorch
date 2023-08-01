import random

import numpy as np
import pywt
import torch
from PIL.Image import Resampling


class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        transform = random.choice(self.transforms)
        return transform(img)


def stack_wavelet_coeffs(coeffs):
    c_a, (c_h, c_v, c_d) = coeffs
    return torch.stack(
        tensors=[
            torch.tensor(data=c_a).float(),
            torch.tensor(data=c_h).float(),
            torch.tensor(data=c_v).float(),
            torch.tensor(data=c_d).float(),
        ],
        dim=0,
    )


def apply_wavelet_transform(x, scale=4):
    x = x.split()[0]  # Take only the Y channel

    x_lr = x.resize(size=(x.size[0] // scale, x.size[1] // scale), resample=Resampling.BICUBIC)
    x_bic = x_lr.resize(size=(x.size[0], x.size[1]), resample=Resampling.BICUBIC)

    x = np.array(x) / 255.  # Normalize
    x_lr = np.array(x_lr) / 255.
    x_bic = np.array(x_bic) / 255.

    input_data = pywt.dwt2(data=x_bic, wavelet='haar')
    input_data = stack_wavelet_coeffs(coeffs=input_data)

    target_data = pywt.dwt2(data=x, wavelet='haar')
    # target_data = pywt.dwt2(data=x - x_bic, wavelet='haar')
    target_data = stack_wavelet_coeffs(coeffs=target_data)

    return x, x_lr, x_bic, input_data, target_data
