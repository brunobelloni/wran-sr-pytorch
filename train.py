import random

import torch
import torch.nn as nn
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms import transforms as T, InterpolationMode
from tqdm import tqdm

from models.wran import WaveletBasedResidualAttentionNet
from utils import apply_preprocess, OneOf, WaveletsTransform, InverseWaveletsTransform, InfiniteDataLoader

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

psnr = PeakSignalNoiseRatio().to(device)
ssim = StructuralSimilarityIndexMeasure().to(device)

# Parameters
SCALE = 4
WIDTH = 64

wt = WaveletsTransform().to(device)
iwt = InverseWaveletsTransform().to(device)

# Define your custom transform
train_transform = T.Compose([
    T.RandomCrop(size=(WIDTH, WIDTH)),
    # basic transforms
    T.RandomVerticalFlip(p=0.5),  # Add random horizontal flip
    T.RandomHorizontalFlip(p=0.5),  # Add random horizontal flip
    OneOf(
        p=0.5,
        transforms=[
            T.RandomRotation(degrees=90, interpolation=InterpolationMode.BICUBIC),
            T.RandomRotation(degrees=180, interpolation=InterpolationMode.BICUBIC),
            T.RandomRotation(degrees=270, interpolation=InterpolationMode.BICUBIC),
        ],
    ),
    # convert to tensor for random erasing
    T.ToTensor(),
    T.RandomErasing(p=0.01, scale=(0.02, 0.22), ratio=(0.3, 3.3)),
    T.ToPILImage(mode='YCbCr'),
    # strong transforms
    # OneOf(
    #     p=0.01,
    #     transforms=[
    #         T.RandomPerspective(p=1, distortion_scale=0.1, interpolation=InterpolationMode.BICUBIC),
    #         T.RandomApply(p=1, transforms=[T.ElasticTransform(interpolation=InterpolationMode.BICUBIC)]),
    #         T.RandomApply(p=1, transforms=[T.ColorJitter(brightness=0.01, contrast=0.1, saturation=0.1, hue=0.1)]),
    #     ],
    # ),
    # generate ground truth
    T.Lambda(lambda x: apply_preprocess(x=x, scale=SCALE)),  # Add wavelet transform
])

val_transform = T.Compose([
    T.RandomCrop(size=(WIDTH, WIDTH)),
    T.Lambda(lambda x: apply_preprocess(x=x, scale=SCALE)),  # Add wavelet transform
])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        input_data = Image.open(fp=self.dataset[idx % len(self.dataset)]['hr']).convert("YCbCr")

        if self.transform:
            input_data = self.transform(input_data)

        return input_data

    def __len__(self):
        return len(self.dataset) * 40


def validate_model(model, dataloader):
    model.eval()
    total_psnr, total_ssim, num_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for image_hr, _, image_bic in dataloader:
            image_bic = image_bic.to(device)
            image_hr = image_hr.to(device)
            input_data = wt(image_bic)
            # target_data = wt(image_hr)
            target_data = wt(image_hr - image_bic)
            outputs = model(input_data)

            batch_psnr = psnr(outputs, target_data)
            batch_ssim = ssim(outputs, target_data)
            return batch_psnr, batch_ssim

            num_batches += 1
            total_psnr += batch_psnr.item()
            total_ssim += batch_ssim.item()

    return total_psnr / num_batches, total_ssim / num_batches


def main():
    dataset = load_dataset("eugenesiow/Div2k")  # Load the dataset

    train_dataset = Dataset(dataset=dataset['train'], transform=train_transform)
    val_dataset = Dataset(dataset=dataset['validation'], transform=val_transform)

    # PyTorch dataloaders
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        # drop_last=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    model = WaveletBasedResidualAttentionNet(width=WIDTH).to(device)
    # model.load_state_dict(torch.load("final_model.pth"))

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

    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer,
    #     mode='max',
    #     factor=0.90,
    #     patience=2,
    #     min_lr=0.00000001,
    # )

    # Validation metrics
    val_psnr, val_ssim = 0, 0

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        for image_hr, _, image_bic in (pbar := tqdm(dataloader)):
            image_bic = image_bic.to(device)
            image_hr = image_hr.to(device)
            input_data = wt(image_bic)
            # target_data = wt(image_hr)
            target_data = wt(image_hr - image_bic)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_data)  # Forward pass
            loss = criterion(outputs, target_data)  # Compute loss
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
                lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            )

        if (epoch + 1) % 1 == 0:
            val_psnr, val_ssim = validate_model(model, val_dataloader)
            # lr_scheduler.step(val_ssim)  # Adjust the learning rate

            # if (epoch + 1) % 5 == 0:
            # from predict import predict
            # predict(model, epoch=(epoch + 1), device=device)
        # torch.save(model.state_dict(), f'model_{(epoch + 1)}.pth')

    torch.save(model.state_dict(), 'final_model.pth')


if __name__ == '__main__':
    main()
