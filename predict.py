import random

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from train import (WaveletBasedResidualAttentionNet, WIDTH, wt, iwt)
from utils import apply_preprocess

model_path = "/home/bruno/Downloads/checkpoint_55/final_model.pth"

psnr = PeakSignalNoiseRatio(data_range=1.0)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)


def predict(model, epoch=None, device=torch.device('cpu')):
    random.seed(42)
    torch.manual_seed(42)

    image = Image.open("test_images/tiger.png").convert('YCbCr')
    image.save("results/full.jpg")
    image = image.crop((740, 600, 740 + WIDTH, 600 + WIDTH))
    # image = image.crop((0, 0, 0 + WIDTH, 0 + WIDTH))

    image_hr, _, image_bic = apply_preprocess(x=image)

    input_data = wt(image_bic.unsqueeze(0).to(device))
    # target_data = wt(image_hr.unsqueeze(0).to(device))
    # target_data = wt((image_hr.unsqueeze(0) - image_bic.unsqueeze(0)).to(device))

    restored_hr = np.array(image)
    # temp_[:, :, 0] = np.array(image_hr * 255).astype(np.uint8())
    image_hr_pil = Image.fromarray(restored_hr, mode='YCbCr')
    image_hr_pil.save("results/original.jpg")

    restored_bic = np.array(image)
    restored_bic[:, :, 0] = np.array(image_bic * 255).astype(np.uint8())
    image_bic_pil = Image.fromarray(restored_bic, mode='YCbCr')
    image_bic_pil.save("results/bicubic.jpg")

    result = model(input_data)

    image_sr = iwt(result)

    restored_sr = np.array(image)
    restored_sr[:, :, 0] = np.array(((image_sr.squeeze(0) + image_bic.to(device)) * 255).detach().cpu()).astype(
        np.uint8())
    # image_sr = np.array(((image_sr.squeeze(0) + image_bic.to(device)) * 255).detach().cpu()).astype(np.uint8())
    image_sr_pil = Image.fromarray(restored_sr, mode='YCbCr')
    image_sr_pil.save("results/sr.jpg")
    if epoch is None:
        image_sr_pil.save("results/sr.jpg")
    else:
        image_sr_pil.save(f"results/sr_{epoch}.jpg")

    # image_sr_bic = np.array(image_sr + image_bic).astype(np.uint8)
    # image_sr_bic_pil = Image.fromarray(image_sr_bic, 'L')
    # if epoch is None:
    #     image_sr_bic_pil.save("results/sr_bic.jpg")
    # else:
    #     image_sr_bic_pil.save(f"results/sr_bic_{epoch}.jpg")

    restored_hr = torch.tensor(restored_hr / 255).unsqueeze(0).permute(0, 3, 1, 2)
    restored_sr = torch.tensor(restored_sr / 255).unsqueeze(0).permute(0, 3, 1, 2)
    print(f'PSNR: {psnr(restored_sr, restored_hr).item():.6f}')
    print(f'SSIM: {ssim(restored_sr, restored_hr).item():.6f}')

    # # add Cb and Cr channels
    # # final_image = image_sr.squeeze(0).cpu().detach().numpy() * 255.0
    # final_image = (image_sr.squeeze(0).cpu().detach().numpy() + image_bic.detach().numpy()) * 255.0
    # image_array[:, :, 0] = final_image.astype(np.uint8)
    # #
    # # convert to PIL image
    # image_reconstructed_pil = Image.fromarray(image_array, 'YCbCr')
    #
    # if epoch is not None:
    #     image_reconstructed_pil.save("results/output_%d.jpg" % epoch)
    # else:
    #     image_reconstructed_pil.save("results/output.jpg")


def main():
    # Create a model instance and move it to device
    model = WaveletBasedResidualAttentionNet(width=WIDTH)
    model.load_state_dict(torch.load(model_path))  # load model
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    predict(model=model, device=device)


if __name__ == '__main__':
    main()
