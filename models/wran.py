import torch.nn as nn

from models.cbam import CBAM
from models.inception import InceptionModule


class WaveletBasedResidualAttentionNet(nn.Module):
    def __init__(self, input_channels=4, depth=8, ratio=4, width=64, alpha=0.1):
        super(WaveletBasedResidualAttentionNet, self).__init__()
        self.depth = depth

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=width, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=alpha)
        )

        self.inception_module = InceptionModule(in_channels=width, width=width, ratio=ratio, alpha=alpha)
        self.attention_module = CBAM(gate_channels=width, reduction_ratio=ratio)

        self.final_layers = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Conv2d(in_channels=width, out_channels=4, kernel_size=3, padding=1),
        )

    def forward(self, x):
        out = self.feature_extraction(x)

        for _ in range(self.depth):
            residual = out
            out = self.inception_module(out)
            out = self.attention_module(out)
            out += residual

        out = self.final_layers(out)
        return out
