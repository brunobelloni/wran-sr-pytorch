import torch
import torch.nn as nn
import torch.nn.init as init

from models.cbam import CBAM
from models.inception import InceptionModule


class WaveletBasedResidualAttentionNet(nn.Module):
    def __init__(self, input_channels=4, depth=8, ratio=4, width=64, alpha=0.1):
        super().__init__()
        self.depth = depth

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=width, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=alpha)
        )

        self.inception_module = InceptionModule(in_channels=width, width=width, ratio=ratio, alpha=alpha)
        self.attention_module = CBAM(gate_channels=width, reduction_ratio=ratio)

        self.final_layers = nn.Sequential(
            nn.Conv2d(in_channels=width * depth, out_channels=width, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Conv2d(in_channels=width, out_channels=4, kernel_size=3, padding=1),
        )

    def forward(self, x):
        out = self.feature_extraction(x)

        residuals = []
        for _ in range(self.depth):
            residual = out
            out = self.inception_module(out)
            out = self.attention_module(out)
            out += residual
            residuals.append(residual)

        out = self.final_layers(torch.cat(tensors=residuals, dim=1))
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Xavier's initialization for convolutional layers
                init.xavier_uniform_(tensor=m.weight, gain=0.1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
