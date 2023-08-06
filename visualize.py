import cv2
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

from train import train_transform, Dataset
from utils import WaveletsTransform, InverseWaveletsTransform

wt = WaveletsTransform()
iwt = InverseWaveletsTransform()


def save_batch_as_image(hr_batch, bic_batch, wt_hr_batch, wt_bic_batch, save_path):
    # def save_batch_as_image(hr_images, bic_images, save_path):
    hr_images = hr_batch * 255.
    bic_images = bic_batch * 255.
    hr_images = hr_images.numpy().astype(np.uint8)  # Convert to numpy
    bic_images = bic_images.numpy().astype(np.uint8)  # Convert to numpy

    # Assume images are grayscale, convert to 3-channel for cv2
    # hr_images = np.stack([hr_images] * 3, axis=-1)
    # bic_images = np.stack([bic_images] * 3, axis=-1)

    concat_images = []

    for idx in range(hr_images.shape[0]):
        # Concatenate 32x32 images vertically to make two 64x32 images
        upper_half_hr = np.concatenate((wt_hr_batch[idx][0], wt_hr_batch[idx][1]), axis=1)
        lower_half_hr = np.concatenate((wt_hr_batch[idx][2], wt_hr_batch[idx][3]), axis=1)
        upper_half_bic = np.concatenate((wt_bic_batch[idx][0], wt_bic_batch[idx][1]), axis=1)
        lower_half_bic = np.concatenate((wt_bic_batch[idx][2], wt_bic_batch[idx][3]), axis=1)

        # Concatenate 64x32 images vertically to make one 64x64 image
        wt_img_hr = np.concatenate((upper_half_hr, lower_half_hr), axis=0)
        wt_img_bic = np.concatenate((upper_half_bic, lower_half_bic), axis=0)

        wt_img_hr = (wt_img_hr - np.min(wt_img_hr)) / (np.max(wt_img_hr) - np.min(wt_img_hr))
        wt_img_hr = (wt_img_hr * 255).astype('uint8')  # Convert data type

        wt_img_bic = (wt_img_bic - np.min(wt_img_bic)) / (np.max(wt_img_bic) - np.min(wt_img_bic))
        wt_img_bic = (wt_img_bic * 255).astype('uint8')  # Convert data type

        # Concatenate 64x64 images horizontally (side by side)
        h_img = np.concatenate((hr_images[idx], wt_img_hr, bic_images[idx], wt_img_bic), axis=1)
        concat_images.append(h_img)

    # Concatenate all the images vertically
    v_img = np.concatenate(concat_images, axis=0)

    # Save the image
    cv2.imwrite(save_path, v_img)


def main():
    dataset = load_dataset("eugenesiow/Div2k")  # Load the dataset

    train_dataset = Dataset(dataset=dataset['train'], transform=train_transform)

    # PyTorch dataloaders
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    for batch_idx, batch in enumerate(dataloader):
        save_batch_as_image(
            hr_batch=batch[0],
            bic_batch=batch[2],
            wt_hr_batch=wt(batch[0]),
            wt_bic_batch=wt(batch[2]),
            save_path=f'batch_{batch_idx}.png',
        )
        print(batch_idx)
        # break  # save only the first batch


if __name__ == '__main__':
    main()
