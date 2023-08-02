import random

import torch
import torch.nn as nn
import torch.nn.parallel as parallel
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm

from models.wran import WaveletBasedResidualAttentionNet
from utils import apply_wavelet_transform, OneOf

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
    OneOf(
        p=1,
        transforms=[
            transforms.RandomCrop(size=(WIDTH, WIDTH)),  # Random crop
            transforms.Resize(size=(WIDTH, WIDTH), interpolation=InterpolationMode.BICUBIC),  # Resize all images
            transforms.RandomResizedCrop(size=(WIDTH, WIDTH), interpolation=InterpolationMode.BICUBIC),  # Random crop
        ],
    ),
    transforms.RandomVerticalFlip(p=0.15),  # Add random vertical flip
    transforms.RandomHorizontalFlip(p=0.15),  # Add random horizontal flip
    # Convert to tensor
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.05, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Add random erasing
    # Convert to PIL image
    transforms.ToPILImage(mode='YCbCr'),
    transforms.RandomPerspective(p=0.01, distortion_scale=0.1, interpolation=InterpolationMode.BICUBIC),
    transforms.RandomApply(
        p=0.01,
        transforms=[transforms.ColorJitter(brightness=0.01, contrast=0.1, saturation=0.1, hue=0.1)],
    ),
    transforms.RandomApply(p=0.01, transforms=[transforms.ElasticTransform(interpolation=InterpolationMode.BICUBIC)]),
    OneOf(
        p=0.15,
        transforms=[
            transforms.RandomAffine(
                shear=10,
                degrees=10,
                scale=(0.9, 1.1),
                translate=(0.01, 0.1),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomRotation(degrees=90, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=180, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=270, interpolation=InterpolationMode.BICUBIC),
        ],
    ),
    transforms.Lambda(lambda x: apply_wavelet_transform(x=x, scale=SCALE)),  # Add wavelet transform
])

val_transform = transforms.Compose([
    transforms.Resize(size=(WIDTH, WIDTH), interpolation=InterpolationMode.BICUBIC),  # Resize all images
    transforms.Lambda(lambda x: apply_wavelet_transform(x=x, scale=SCALE)),  # Add wavelet transform
])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        input_data = Image.open(fp=self.dataset[idx]['hr']).convert("YCbCr")

        if self.transform:
            input_data = self.transform(input_data)

        return input_data

    def __len__(self):
        return len(self.dataset)


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

    train_dataset = Dataset(dataset=dataset['train'], transform=train_transform)
    val_dataset = Dataset(dataset=dataset['validation'], transform=val_transform)

    # PyTorch dataloaders
    dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=False, num_workers=2, pin_memory=True)

    model = WaveletBasedResidualAttentionNet(width=WIDTH).to(device)
    # model.load_state_dict(torch.load("final_model.pth"))

    # Wrap the model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = parallel.DataParallel(model)

    # wandb.init(project="wransr", entity="brunobelloni")
    # wandb.watch(model)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        lr=0.01,
        eps=1e-08,
        weight_decay=0,
        betas=(0.9, 0.999),
        params=model.parameters(),
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=0.90,
        patience=10,
        min_lr=0.0001,
    )

    # Validation metrics
    val_psnr, val_ssim = 0, 0

    # Training loop
    num_epochs = 1_000
    for epoch in (pbar := tqdm(range(num_epochs))):
        model.train()
        for x, x_lr, x_bic, input_data, target_data in dataloader:
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
                loss=f"{loss.item():.6f}",
                psnr=f"{psnr_value:.6f}",
                ssim=f"{ssim_value:.6f}",
                val_ssim=f"{val_ssim:.6f}",
                val_psnr=f"{val_psnr:.6f}",
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            )
        print('')

        if (epoch + 1) % 1 == 0:
            val_psnr, val_ssim = validate_model(model, val_dataloader)
            lr_scheduler.step(val_psnr)  # Adjust the learning rate

            # if (epoch + 1) % 5 == 0:
            from predict import predict
            predict(model, epoch=(epoch + 1), device=device)
        #     # torch.save(model.state_dict(), f'checkpoint/model_{(epoch + 1)}.pth')

    torch.save(model.state_dict(), 'checkpoint/final_model.pth')


if __name__ == '__main__':
    main()
