"""
Encoder for the land cover input, which is categorical (37 class labels
per pixel) rather than continuous like LST/PM2.5, so it needs different
handling upfront.

Pipeline:
  1. Embedding layer: turns each class label into a learned 8-dim vector
     (instead of treating class numbers as if they were on a numeric scale).
  2. CoordConv: appends normalized (x, y) coordinate channels so the
     network can learn spatially-dependent patterns (e.g. "urban core vs
     periphery"), which plain convolutions can't do on their own since
     they're translation-invariant by design.
  3. A residual dilated conv block: widens the receptive field without
     downsampling, so the network sees more spatial context per pixel.
  4. CBAM: re-weights channels/locations by importance (reuses the CBAM
     module built earlier).
"""

import torch
import torch.nn as nn

from models.cbam import CBAM


class CoordConv(nn.Module):
    """
    A standard conv layer, but with two extra input channels containing
    normalized x and y coordinates. This gives the network a sense of
    spatial position, which ordinary convolutions lack.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        # +2 for the appended coordinate channels
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, padding=padding)

    def _make_coord_channels(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        y_coords = torch.linspace(-1, 1, height, device=x.device).view(1, 1, height, 1).expand(batch, 1, height, width)
        x_coords = torch.linspace(-1, 1, width, device=x.device).view(1, 1, 1, width).expand(batch, 1, height, width)
        return torch.cat([x_coords, y_coords], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coords = self._make_coord_channels(x)
        x = torch.cat([x, coords], dim=1)
        return self.conv(x)


class ResidualDilatedBlock(nn.Module):
    """
    Two dilated convolutions with a residual (skip) connection.
    Dilation widens the receptive field (sees more surrounding context)
    without downsampling the image, which matters here since land cover
    detail (e.g. small green patches) shouldn't be shrunk away too early.
    """

    def __init__(self, channels: int, dilation: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # skip connection
        return self.relu(out)


class LandCoverEncoder(nn.Module):
    """
    Full land cover encoding pipeline: embedding -> CoordConv ->
    residual dilated block -> CBAM.

    Input:  (B, H, W) integer class labels in [0, num_classes)
    Output: (B, out_channels, H, W)
    """

    def __init__(
        self,
        num_classes: int = 37,
        embed_dim: int = 8,
        out_channels: int = 32,
        reduction_ratio: int = 8,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.coord_conv = CoordConv(embed_dim, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual_block = ResidualDilatedBlock(out_channels, dilation=2)
        self.cbam = CBAM(out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W) integer labels -> embed -> (B, H, W, embed_dim)
        x = self.embedding(x)
        # rearrange to (B, embed_dim, H, W) for conv layers
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.coord_conv(x)
        x = self.relu(self.bn(x))
        x = self.residual_block(x)
        x = self.cbam(x)
        return x
