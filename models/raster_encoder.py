"""
Encoder for continuous raster inputs: Land Surface Temperature (LST) and
PM2.5. Both are single-channel images, so they share this same encoder
design (used as two separate instances, one per input, in the full model).

Architecture: three Conv-BN-ReLU stages with stride-2 downsampling.
Kaiming initialization is used since ReLU is the activation throughout.
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """One Conv -> BatchNorm -> ReLU block with stride-2 downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RasterEncoder(nn.Module):
    """
    Three-stage Conv-BN-ReLU encoder for a single-channel raster input
    (used separately for LST and for PM2.5).

    Input:  (B, 1, H, W)
    Output: (B, base_channels * 4, H/8, W/8)
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.stage1 = ConvBNReLU(in_channels, base_channels)
        self.stage2 = ConvBNReLU(base_channels, base_channels * 2)
        self.stage3 = ConvBNReLU(base_channels * 2, base_channels * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x
