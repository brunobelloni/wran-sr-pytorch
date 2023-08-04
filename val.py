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

    val_dataset = Dataset(dataset=dataset['validation'])

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    model = WaveletBasedResidualAttentionNet(width=WIDTH).to(device)
    model.load_state_dict(torch.load("/home/bruno/Downloads/checkpoints/model_1.pth"))

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

        for index in range(len(outputs)):
            current_bic = original[index].detach().cpu().numpy().copy()
            current_bic[:, :, 0] = image_bic[index].detach().cpu().numpy()

            current_sr = original[index].detach().cpu().numpy().copy()
            current_sr[:, :, 0] = outputs[index].detach().cpu().numpy()

            current_sr_bic = original[index].detach().cpu().numpy().copy()
            current_sr_bic[:, :, 0] = (outputs[index].detach().cpu().numpy() + image_bic[index].detach().cpu().numpy())

            # Accumulate images for plotting
            hr_images.append(original[index].detach().cpu().numpy())
            bic_images.append(current_bic)
            sr_images.append(current_sr)
            sr_bic_images.append(current_sr_bic)
        break

    # Calculate average PSNR and SSIM
    avg_psnr = psnr_sum / total_images
    avg_ssim = ssim_sum / total_images
    print(f'Average PSNR: {avg_psnr}')
    print(f'Average SSIM: {avg_ssim}')

    # Create the plot with 5 random images
    num_images = len(hr_images)
    num_random_images = 1
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
