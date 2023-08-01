import torch
import torch.nn as nn


# code modified from from:
# https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
# https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

    def forward(self, x):
        max_out = self.fc(self.max_pool(x))
        avg_out = self.fc(self.avg_pool(x))
        out = avg_out + max_out
        return torch.multiply(x, self.sigmoid(out))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return torch.multiply(x, self.sigmoid(out))


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelAttention(in_planes=gate_channels, ratio=reduction_ratio)
        self.spatial_gate = SpatialAttention()

    def forward(self, x):
        x_out = self.channel_gate(x)
        x_out = self.spatial_gate(x_out)
        return x_out
