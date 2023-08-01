import random

import torch
import torch.nn as nn
from PIL import Image
from PIL.Image import Resampling
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from models.wran import WaveletBasedResidualAttentionNet
from utils import apply_wavelet_transform, LRScheduler

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

psnr = PeakSignalNoiseRatio().to(device)
ssim = StructuralSimilarityIndexMeasure().to(device)

# Parameters
SCALE = 4
WIDTH = 64

# Define your custom transform
train_transform = transforms.Compose([
    transforms.Resize(size=(WIDTH, WIDTH), interpolation=InterpolationMode.BICUBIC),  # Resize all images
    transforms.RandomHorizontalFlip(p=0.5),  # Add random horizontal flip
    transforms.RandomVerticalFlip(p=0.5),  # Add random vertical flip
    transforms.Lambda(lambda img: img.rotate(  # Add random rotation
        angle=random.choice([0, 0, 0, 90, 180, 270]),
        resample=Resampling.BICUBIC,
    )),
    transforms.Lambda(lambda x: apply_wavelet_transform(x=x, scale=SCALE)),  # Add wavelet transform
])

val_transform = transforms.Compose([
    transforms.Resize(size=(WIDTH, WIDTH), interpolation=InterpolationMode.BICUBIC),  # Resize all images
    transforms.Lambda(lambda x: apply_wavelet_transform(x=x, scale=SCALE)),  # Add wavelet transform
])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, width=64, transform=None):
        self.width = width
        self.transform = transform
        self.hf_dataset = hf_dataset

    def __getitem__(self, idx):
        input_data = Image.open(fp=self.hf_dataset[idx]['hr']).convert("YCbCr")

        if self.transform:
            input_data = self.transform(input_data)

        return input_data

    def __len__(self):
        return len(self.hf_dataset)


def validate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        for x, x_lr, x_bic, input_data, target_data in dataloader:
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            outputs = model(input_data)
            return psnr(outputs, target_data), ssim(outputs, target_data)


def main():
    dataset = load_dataset("eugenesiow/Div2k")  # Load the dataset

    train_dataset = Dataset(hf_dataset=dataset['train'], transform=train_transform)
    val_dataset = Dataset(hf_dataset=dataset['validation'], transform=val_transform)

    # PyTorch dataloaders
    dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=False, num_workers=16, pin_memory=True)

    model = WaveletBasedResidualAttentionNet(width=WIDTH).to(device)

    # wandb.init(project="wransr", entity="brunobelloni")
    # wandb.watch(model)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        lr=0.001,
        eps=1e-08,
        weight_decay=0,
        betas=(0.9, 0.999),
        params=model.parameters(),
    )

    scheduler = LRScheduler(optimizer)

    # Early stopping parameters
    patience, counter, best_psnr = 20, 0, -float('inf')

    # Training loop
    num_epochs = 200
    batches_per_epoch = 500
    dataloader_iterator = iter(dataloader)  # Iterator for the dataloader

    val_psnr, val_ssim = 0, 0
    for epoch in range(num_epochs):
        model.train()
        for _ in (pbar := tqdm(range(batches_per_epoch), unit='batches')):
            try:
                x, x_lr, x_bic, input_data, target_data = next(dataloader_iterator)
            except StopIteration:  # If the dataloader is exhausted, reset the iterator for the next epoch
                dataloader_iterator = iter(dataloader)
                x, x_lr, x_bic, input_data, target_data = next(dataloader_iterator)
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            outputs = model(input_data)  # Forward pass

            loss = criterion(outputs, target_data)  # Compute loss
            optimizer.zero_grad()  # Zero gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            psnr_value = psnr(outputs, target_data)
            ssim_value = ssim(outputs, target_data)
            pbar.set_postfix(
                epoch=f"{epoch + 1}/{num_epochs}",
                loss=f"{loss.item():.6f}",
                psnr=f"{psnr_value:.6f}",
                ssim=f"{ssim_value:.6f}",
                val_ssim=f"{val_ssim:.6f}",
                val_psnr=f"{val_psnr:.6f}",
            )

        scheduler.step()  # Adjust the learning rate

        val_psnr, val_ssim = validate_model(model, val_dataloader)

        if val_psnr > best_psnr: # Check if val_psnr has improved
            best_psnr, counter = val_psnr, 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Val PSNR did not improve for {patience} epochs. Early stopping...")
                break

        # if (epoch + 1) % 5 == 0:
        #     from predict import predict
        #     predict(model, epoch=(epoch + 1), device=device)
        #     # torch.save(model.state_dict(), f'checkpoint/model_{(epoch + 1)}.pth')

    torch.save(model.state_dict(), 'checkpoint/final_model.pth')


if __name__ == '__main__':
    main()
