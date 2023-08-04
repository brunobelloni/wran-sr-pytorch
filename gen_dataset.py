import pickle

import pywt
from PIL import Image
from PIL.Image import Resampling
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

SCALE = 4
PATCH_SIZE = 64


def split_image_into_patches(image, patch_size):
    width, height = image.size
    patches = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            left = x
            upper = y
            right = x + patch_size
            lower = y + patch_size
            patch = image.crop((left, upper, right, lower))
            patches.append(patch)
    return patches


def apply_preprocess(x):
    c_a, (c_h, c_v, c_d) = pywt.dwt2(data=x, wavelet='haar')
    target_data = np.stack((c_a, c_h, c_v, c_d), axis=0)

    x_lr = x.resize(size=(x.size[0] // SCALE, x.size[1] // SCALE), resample=Resampling.BICUBIC)
    x_bic = x_lr.resize(size=(x.size[0], x.size[1]), resample=Resampling.BICUBIC)

    c_a, (c_h, c_v, c_d) = pywt.dwt2(data=x_bic, wavelet='haar')
    input_data = np.stack((c_a, c_h, c_v, c_d), axis=0)

    return input_data, target_data


def main():
    dataset = load_dataset("eugenesiow/Div2k")  # Load the dataset

    train_patches = []
    val_patches = []

    for image_info in tqdm(dataset['train']):
        image = Image.open(fp=image_info['hr']).convert("YCbCr")
        image = image.split()[0]  # Get the Y channel
        patches = split_image_into_patches(image, PATCH_SIZE)
        for x in patches:
            input_data, target_data = apply_preprocess(x=x)
            train_patches.append((input_data, target_data))
        # print(image_info['hr'])

    for index, image_info in tqdm(enumerate(dataset['validation'])):
        if index not in [4, 8, 17, 19, 20, 28, 37, 66, 68, 69]:
            continue
        image = Image.open(fp=image_info['hr']).convert("YCbCr")
        image.convert('RGB').save(f'batches/{index}.png')
        image = image.split()[0]  # Get the Y channel
        patches = split_image_into_patches(image, PATCH_SIZE)
        for x in patches:
            input_data, target_data = apply_preprocess(x=x)
            val_patches.append((input_data, target_data))
        print(image_info['hr'])

    # Save the data as pickle files
    with open('train_data.pickle', 'wb') as f:
        pickle.dump(train_patches, f)

    with open('val_data.pickle', 'wb') as f:
        pickle.dump(val_patches, f)


if __name__ == '__main__':
    main()
