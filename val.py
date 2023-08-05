import random

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from models.wran import WaveletBasedResidualAttentionNet
from train import WIDTH, wt, psnr, ssim, Dataset, val_transform, iwt

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    dataset = load_dataset("eugenesiow/Div2k")  # Load the dataset

    val_dataset = Dataset(
        dataset=[
            # {'hr': 'test_images/comic.bmp', 'crop': (140, 105, 140 + WIDTH, 105 + WIDTH)},
            # {'hr': 'test_images/butterfly.bmp', 'crop': (150, 150, 150 + WIDTH, 150 + WIDTH)},
            {'hr': 'test_images/books.png', 'crop': (0, 0, 0 + WIDTH, 0 + WIDTH)},
            {'hr': 'test_images/cat.png', 'crop': (800, 900, 800 + WIDTH, 900 + WIDTH)},
            {'hr': 'test_images/lion.png', 'crop': (970, 780, 970 + WIDTH, 780 + WIDTH)},
            {'hr': 'test_images/tiger.png', 'crop': (740, 600, 740 + WIDTH, 600 + WIDTH)},
            {'hr': 'test_images/train.png', 'crop': (180, 700, 180 + WIDTH, 700 + WIDTH)},
            {'hr': 'test_images/spiral.png', 'crop': (950, 150, 950 + WIDTH, 150 + WIDTH)},
            {'hr': 'test_images/wolf.png', 'crop': (1200, 300, 1200 + WIDTH, 300 + WIDTH)},
            {'hr': 'test_images/buda.png', 'crop': (1225, 180, 1225 + WIDTH, 180 + WIDTH)},
            {'hr': 'test_images/aligator.png', 'crop': (100, 700, 100 + WIDTH, 700 + WIDTH)},
            {'hr': 'test_images/butterfly.png', 'crop': (900, 1000, 900 + WIDTH, 1000 + WIDTH)},
        ],
        _type='val',
        transform=val_transform,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    model = WaveletBasedResidualAttentionNet(width=WIDTH).to(device)
    model.load_state_dict(torch.load("/home/bruno/Downloads/checkpoints_2/model_100.pth"))

    # Initialize lists to store images for plotting
    hr_images = []
    bic_images = []
    sr_images = []
    sr_bic_images = []

    psnr_sum = 0.0
    ssim_sum = 0.0
    total_images = 0

    for image_hr, image_lr, image_bic in val_dataloader:
        image_bic = image_bic.to(device)
        image_hr = image_hr.to(device)
        input_data = wt(image_bic)
        target_data = wt(image_hr - image_bic)
        outputs = model(input_data)

        psnr_sum += psnr(outputs, target_data).sum().item()
        ssim_sum += ssim(outputs, target_data).sum().item()
        total_images += input_data.size(0)

        outputs = iwt(outputs) * 255.0
        image_hr = image_hr * 255.0
        image_bic = image_bic * 255.0

        for index in range(len(outputs)):
            # Accumulate images for plotting
            hr_images.append(image_hr[index].detach().cpu().numpy())
            bic_images.append(image_bic[index].detach().cpu().numpy())
            sr_images.append(outputs[index].detach().cpu().numpy())
            sr_bic_images.append(outputs[index].detach().cpu().numpy() + image_bic[index].detach().cpu().numpy())
        break

    # Calculate average PSNR and SSIM
    avg_psnr = psnr_sum / total_images
    avg_ssim = ssim_sum / total_images
    print(f'Average PSNR: {avg_psnr}')
    print(f'Average SSIM: {avg_ssim}')

    # Create the plot with 5 random images
    num_images = len(hr_images)
    num_random_images = 3
    random_indices = random.sample(range(num_images), num_random_images)

    plt.figure(figsize=(15, 4 * num_random_images))
    for i, index in enumerate(random_indices):
        plt.subplot(num_random_images, 4, i * 4 + 1)
        plt.imshow(hr_images[index])
        plt.title('HR Image')

        plt.subplot(num_random_images, 4, i * 4 + 2)
        plt.imshow(bic_images[index])
        plt.title('BIC Image')

        plt.subplot(num_random_images, 4, i * 4 + 3)
        plt.imshow(sr_images[index])
        plt.title('SR Image')

        plt.subplot(num_random_images, 4, i * 4 + 4)
        plt.imshow(sr_bic_images[index])
        plt.title('SR Image + BIC Image')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
