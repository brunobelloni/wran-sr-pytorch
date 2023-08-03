import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from train import (WaveletBasedResidualAttentionNet, apply_preprocess, WIDTH, wt,
                   iwt)

model_path = "final_model_2k.pth"


def predict(model, epoch=None, device=torch.device('cpu')):
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    image = Image.open("/home/bruno/Downloads/comic.bmp").convert('YCbCr')
    # image = image.resize((WIDTH, WIDTH), resample=Resampling.BICUBIC)
    image = image.crop((140, 105, 140 + WIDTH, 105 + WIDTH))  # comic crop

    image_array = np.array(image)

    image_hr, image_lr, image_bic = apply_preprocess(x=image)

    input_data = wt(image_bic.unsqueeze(0).to(device))
    target_data = wt(image_hr.unsqueeze(0).to(device) - image_bic.unsqueeze(0).to(device))

    image.save("results/original.jpg")  # save original image

    image_array_bicubic = image_array.copy()
    image_array_bicubic[:, :, 0] = np.array(image_bic) * 255.0
    image_array_bicubic = Image.fromarray(image_array_bicubic, 'YCbCr')
    image_array_bicubic.save("results/bicubic.jpg")  # save bicubic image

    # Predict an example image and show it upscaled
    result = model(input_data)

    print('PSNR:', psnr(result, target_data).item())
    print('SSIM:', ssim(result, target_data).item())

    image_sr = iwt(result)

    # add Cb and Cr channels
    image_array[:, :, 0] = (image_sr.squeeze(0).cpu().detach().numpy() + image_bic.detach().numpy()) * 255.0
    # image_array[:, :, 0] = image_sr.squeeze(0).cpu().detach().numpy() * 255.0

    # convert to PIL image
    image_reconstructed_pil = Image.fromarray(image_array, 'YCbCr')

    if epoch is not None:
        image_reconstructed_pil.save("results/output_%d.jpg" % epoch)
    else:
        image_reconstructed_pil.save("results/output.jpg")


def main():
    # Create a model instance and move it to device
    model = WaveletBasedResidualAttentionNet(width=WIDTH)
    model.load_state_dict(torch.load(model_path))  # load model
    model.eval()

    predict(model)


if __name__ == '__main__':
    main()
