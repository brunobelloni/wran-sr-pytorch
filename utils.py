import numpy as np
import pywt
import torch
import torch.optim.lr_scheduler as lr_scheduler
from PIL.Image import Resampling


def psnr_loss(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 100 if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse))


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 40:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    return lr


# noinspection PyUnresolvedReferences
class LRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer):
        self.lr_func = lr_schedule
        super(LRScheduler, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch + 1
        return [self.lr_func(epoch) for _ in self.base_lrs]


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

    # target_data = pywt.dwt2(data=x, wavelet='haar')
    target_data = pywt.dwt2(data=x - x_bic, wavelet='haar')
    target_data = stack_wavelet_coeffs(coeffs=target_data)

    return x, x_lr, x_bic, input_data, target_data
