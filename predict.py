import random

import numpy as np
import torch
from PIL import Image

from train import (WaveletBasedResidualAttentionNet, WIDTH, wt, iwt)
from utils import apply_preprocess, psnr_loss, ssim_loss

model_path = "/home/bruno/Downloads/checkpoints_2/model_110.pth"


def retore_image():
    pass


def predict(model, epoch=None, device=torch.device('cpu')):
    random.seed(42)
    torch.manual_seed(42)

    image = Image.open("test_images/tiger.png").convert('YCbCr')
    image = image.crop((740, 600, 740 + WIDTH, 600 + WIDTH))

    image_hr, _, image_bic = apply_preprocess(x=image)

    input_data = wt(image_bic.unsqueeze(0).to(device))
    # target_data = wt(image_hr.unsqueeze(0).to(device))
    # target_data = wt((image_hr.unsqueeze(0) - image_bic.unsqueeze(0)).to(device))

    image_hr_pil = Image.fromarray(np.array(image_hr * 255).astype(np.uint8), mode='L')
    image_hr_pil.save("results/original.jpg")

    image_bic_pil = Image.fromarray(np.array(image_bic * 255.0).astype(np.uint8), 'L')
    image_bic_pil.save("results/bicubic.jpg")

    result = model(input_data)

    image_sr = iwt(result)

    image_sr = np.array(((image_sr.squeeze(0) + image_bic.to(device)) * 255).detach().cpu()).astype(np.uint8())
    image_sr_pil = Image.fromarray(image_sr, 'L')
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

    print(f'PSNR: {psnr_loss(iwt(result).squeeze(0), image_hr.to(device)).item():.6f}')
    print(f'SSIM: {ssim_loss(iwt(result).squeeze(0), image_hr.to(device)).item():.6f}')

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
