import numpy as np
import pywt
import torch
from PIL import Image
from PIL.Image import Resampling

from train import WaveletBasedResidualAttentionNet, apply_wavelet_transform, WIDTH

model_path = "final_model_1k.pth"


def predict(model, epoch=None, device=torch.device('cpu')):
    image = Image.open("/home/bruno/Downloads/tigre.png").convert('YCbCr')
    image = image.resize((WIDTH, WIDTH), resample=Resampling.BICUBIC)
    image_array = np.array(image)

    x, x_lr, x_bic, input_data, target_data = apply_wavelet_transform(x=image)

    image.save("results/original.jpg")  # save original image

    image_array_bicubic = image_array.copy()
    image_array_bicubic[:, :, 0] = np.array(x_bic) * 255.0
    image_array_bicubic = Image.fromarray(image_array_bicubic, 'YCbCr')
    image_array_bicubic.save("results/bicubic.jpg")  # save bicubic image

    # Predict an example image and show it upscaled
    result = model(input_data.unsqueeze(0).to(device))

    # Convert the torch tensors to numpy arrays and rearrange channels
    result = result.squeeze(0).cpu().detach().numpy()
    # print('result', result.shape)

    # reverse wavelet transform
    x_sr = pywt.idwt2((result[0], (result[1], result[2], result[3])), wavelet='haar', mode='zero')

    # add Cb and Cr channels
    # image_array[:, :, 0] = (x_sr + x_bic) * 255.0
    image_array[:, :, 0] = (x_sr * 255.0)

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
