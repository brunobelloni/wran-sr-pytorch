import random

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import wandb
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import transforms as T
from tqdm import tqdm

from models.wran import WaveletBasedResidualAttentionNet
from utils import apply_preprocess, WaveletsTransform, InverseWaveletsTransform, psnr_loss, ssim_loss, OneOf

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# psnr = PeakSignalNoiseRatio().to(device)
# ssim = StructuralSimilarityIndexMeasure().to(device)

# Parameters
SCALE = 4
WIDTH = 64
BATCH_SIZE = 64

wt = WaveletsTransform().to(device)
iwt = InverseWaveletsTransform().to(device)


class AlbumentationsTransforms:
    def __init__(self):
        self.transform = A.Compose(
            transforms=[
                A.OneOf(
                    p=0.15,
                    transforms=[
                        A.CLAHE(p=1),
                        A.Sharpen(p=1),
                    ],
                ),
                A.GridDropout(p=0.15, fill_value=0),
                A.ColorJitter(p=0.15, brightness=(.05, .3), contrast=(.05, .3), saturation=(.05, .3), hue=(.05, .3)),
            ],
        )

    def __call__(self, x):
        x = np.array(x)
        augmented = self.transform(image=x)
        return Image.fromarray(augmented['image'])


# Define your custom transform
train_transform = T.Compose([
    # crop/resize
    OneOf(
        p=1,
        transforms=[
            T.RandomCrop(size=(WIDTH, WIDTH), padding_mode='edge'),
            T.RandomResizedCrop(
                size=(WIDTH, WIDTH),
                scale=(0.05, 0.3),
                ratio=(0.8, 1.2),
                interpolation=InterpolationMode.BICUBIC,
            ),
        ],
    ),
    # basic transforms
    T.RandomVerticalFlip(p=0.25),
    T.RandomHorizontalFlip(p=0.25),
    OneOf(
        p=0.25,
        transforms=[
            T.RandomAffine(
                degrees=(1, 15),
                shear=(0.05, 0.3),
                scale=(0.9, 1.1),
                translate=(0.05, 0.3),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.RandomRotation(degrees=(90, 270), interpolation=InterpolationMode.BICUBIC),
        ],
    ),
    # albumentations transforms
    T.Lambda(lambda x: AlbumentationsTransforms()(x)),
    # generate ground truth
    T.Lambda(lambda x: apply_preprocess(x=x, scale=SCALE)),  # Add wavelet transform
])

val_transform = T.Compose([
    T.Lambda(lambda x: apply_preprocess(x=x, scale=SCALE)),  # Add wavelet transform
])

custom_val_dataset = [
    # other set
    # {'hr': 'test_images/comic.bmp', 'crop': (140, 105, 140 + WIDTH, 105 + WIDTH)},
    # {'hr': 'test_images/butterfly.bmp', 'crop': (150, 150, 150 + WIDTH, 150 + WIDTH)},
    # train set
    # {'hr': 'test_images/tiger.png', 'crop': (740, 600, 740 + WIDTH, 600 + WIDTH)},
    # validation dataset
    {'hr': 'test_images/books.png', 'crop': (0, 0, 0 + WIDTH, 0 + WIDTH)},
    {'hr': 'test_images/cat.png', 'crop': (800, 900, 800 + WIDTH, 900 + WIDTH)},
    {'hr': 'test_images/cat_2.png', 'crop': (520, 500, 520 + WIDTH, 500 + WIDTH)},
    {'hr': 'test_images/lion.png', 'crop': (970, 780, 970 + WIDTH, 780 + WIDTH)},
    {'hr': 'test_images/train.png', 'crop': (180, 700, 180 + WIDTH, 700 + WIDTH)},
    {'hr': 'test_images/spiral.png', 'crop': (950, 150, 950 + WIDTH, 150 + WIDTH)},
    {'hr': 'test_images/wolf.png', 'crop': (1200, 300, 1200 + WIDTH, 300 + WIDTH)},
    {'hr': 'test_images/buda.png', 'crop': (1225, 180, 1225 + WIDTH, 180 + WIDTH)},
    {'hr': 'test_images/aligator.png', 'crop': (100, 700, 100 + WIDTH, 700 + WIDTH)},
    {'hr': 'test_images/butterfly.png', 'crop': (900, 1000, 900 + WIDTH, 1000 + WIDTH)},
]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, multiplier=1, _type='train'):
        self._type = _type
        self.dataset = dataset
        self.transform = transform
        self.multiplier = multiplier

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        input_data = Image.open(fp=self.dataset[idx]['hr']).convert("YCbCr")

        if self._type != 'train':
            input_data = input_data.crop(self.dataset[idx]['crop'])

        if self.transform:
            input_data = self.transform(input_data)

        return input_data

    def __len__(self):
        return len(self.dataset) * self.multiplier


def validate_model(model, dataloader, epoch=None, save_image=False):
    model.eval()
    total_psnr, total_ssim, num_batches = 0, 0, 0

    with torch.no_grad():
        for image_hr, _, image_bic in dataloader:
            image_bic = image_bic.to(device)
            image_hr = image_hr.to(device)
            outputs = model(wt(image_bic))

            batch_psnr = psnr_loss(iwt(outputs), image_hr)
            batch_ssim = ssim_loss(iwt(outputs), image_hr)

            num_batches += 1
            total_psnr += batch_psnr.item()
            total_ssim += batch_ssim.item()

    image_pil, caption = None, None
    if save_image and epoch:
        # image_array = np.array(iwt(outputs)[0].detach().cpu() * 255).astype(np.uint8)
        image_array = np.array((iwt(outputs)[0].detach().cpu() + image_bic[0].detach().cpu()) * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_array, mode='L')

        batch_psnr = psnr_loss(iwt(outputs)[0], image_hr[0])
        batch_ssim = ssim_loss(iwt(outputs)[0], image_hr[0])
        caption = f"{batch_psnr.item():.4f}/{batch_ssim.item():.4f}"
        # image_pil.save(f'results/sr_{epoch}.jpg')
        #
        # image_hr_array = np.array(image_hr[0].detach().cpu() * 255).astype(np.uint8)
        # image_hr_pil = Image.fromarray(image_hr_array, mode='L')
        # image_hr_pil.save(f'results/hr.jpg')
        #
        # image_bic_array = np.array(image_bic[0].detach().cpu() * 255).astype(np.uint8)
        # image_bic_pil = Image.fromarray(image_bic_array, mode='L')
        # image_bic_pil.save(f'results/bic.jpg')

    return total_psnr / num_batches, total_ssim / num_batches, image_pil, caption


def main():
    dataset = load_dataset("eugenesiow/Div2k")  # Load the dataset

    train_dataset = Dataset(dataset=dataset['train'], transform=train_transform, multiplier=1, _type='train')
    val_dataset = Dataset(dataset=custom_val_dataset, _type='val', transform=val_transform)
    # val_dataset = Dataset(dataset=dataset['validation'], transform=val_transform)

    # PyTorch dataloaders
    dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=False, num_workers=2, pin_memory=True)

    model = WaveletBasedResidualAttentionNet(width=WIDTH).to(device)
    model.initialize_weights()
    # model.load_state_dict(torch.load("final_model.pth"))

    wandb.init(project="wransr", entity="brunobelloni", save_code=True)
    wandb.watch(model)

    initial_lr, min_lr, lr_decay_factor, lr_decay_epoch = 0.001, 0.0000001, 0.1, 40
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        lr=initial_lr,
        eps=1e-08,
        weight_decay=0,
        betas=(0.9, 0.999),
        params=model.parameters(),
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch: lr_decay_factor ** (epoch // lr_decay_epoch)
        if initial_lr * lr_decay_factor ** (epoch // lr_decay_epoch) > min_lr
        else min_lr
    )

    # Validation metrics
    val_psnr, val_ssim = 0, 0
    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        for index, (image_hr, _, image_bic) in enumerate((pbar := tqdm(dataloader))):
            image_hr = image_hr.to(device)
            image_bic = image_bic.to(device)

            optimizer.zero_grad()  # zero the parameter gradients

            # forward + backward + optimize
            outputs = model(wt(image_bic))  # Forward pass
            loss = criterion(outputs, wt(image_hr - image_bic))  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            psnr_value = psnr_loss(iwt(outputs), image_hr)
            ssim_value = ssim_loss(iwt(outputs), image_hr)

            log = {
                "epoch": epoch + 1,
                "loss": loss.item(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "psnr": psnr_value.item(),
                "ssim": ssim_value.item(),
                "lr": optimizer.param_groups[0]['lr'],
            }
            pbar.set_postfix(**log)
            if index >= (pbar.total - 1):
                val_psnr, val_ssim, image, caption = validate_model(
                    model=model,
                    epoch=epoch + 1,
                    save_image=True,
                    dataloader=val_dataloader,
                )
                if image:
                    log["output_image"] = wandb.Image(data_or_path=image, caption=caption)
            wandb.log(log)

        lr_scheduler.step()  # Adjust the learning rate
        torch.save(model.state_dict(), f'checkpoint/model_{(epoch + 1)}.pth')

    torch.save(model.state_dict(), 'checkpoint/final_model.pth')


if __name__ == '__main__':
    main()
