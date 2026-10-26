"""Handwritten small-image residual network; no torchvision/backbone imports."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # The shortcut must match both spatial dimensions and channel count.
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return torch.relu(self.branch(x) + self.shortcut(x))


class DigitResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.stages = nn.Sequential(
            ResidualBlock(16, 16), ResidualBlock(16, 16),
            ResidualBlock(16, 32, 2), ResidualBlock(32, 32),
            ResidualBlock(32, 64, 2), ResidualBlock(64, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        return self.head(self.pool(self.stages(self.stem(x))).flatten(1))
