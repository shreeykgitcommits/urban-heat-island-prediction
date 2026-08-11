"""
Full UHI prediction model: wires together the LST encoder, PM2.5 encoder,
land cover encoder, and the Attention U-Net fusion module into a single
end-to-end model.

Usage:
    model = UHIModel(input_size=(64, 64))
    uhii_map = model(lst, pm25, landcover)
"""

import torch
import torch.nn as nn

from models.raster_encoder import RasterEncoder
from models.landcover_encoder import LandCoverEncoder
from models.attention_unet import AttentionFusionUNet


class UHIModel(nn.Module):
    def __init__(
        self,
        input_size: tuple,
        base_channels: int = 32,
        landcover_num_classes: int = 37,
        landcover_embed_dim: int = 8,
        reduction_ratio: int = 8,
    ):
        super().__init__()

        # Two raster encoders share the same design but have separate,
        # independently-learned weights (temperature and pollution
        # patterns are different phenomena, so they shouldn't share weights).
        self.lst_encoder = RasterEncoder(in_channels=1, base_channels=base_channels)
        self.pm25_encoder = RasterEncoder(in_channels=1, base_channels=base_channels)

        self.landcover_encoder = LandCoverEncoder(
            num_classes=landcover_num_classes,
            embed_dim=landcover_embed_dim,
            out_channels=base_channels,
            reduction_ratio=reduction_ratio,
        )

        # RasterEncoder ends at base_channels * 4 (three stride-2 stages).
        # LandCoverEncoder ends at base_channels (no downsampling by design,
        # since we didn't want to shrink land cover detail early).
        # We downsample the land cover features to match the raster
        # encoders' spatial size before fusion.
        self.landcover_downsample = nn.Conv2d(
            base_channels, base_channels * 4, kernel_size=3, stride=8, padding=1
        )

        self.fusion = AttentionFusionUNet(
            lst_channels=base_channels * 4,
            pm25_channels=base_channels * 4,
            landcover_channels=base_channels * 4,
            target_size=input_size,
        )

    def forward(self, lst: torch.Tensor, pm25: torch.Tensor, landcover: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lst:       (B, 1, H, W) float tensor
            pm25:      (B, 1, H, W) float tensor
            landcover: (B, H, W) long tensor of class indices
        Returns:
            (B, 1, H, W) predicted UHII map
        """
        lst_feat = self.lst_encoder(lst)
        pm25_feat = self.pm25_encoder(pm25)

        landcover_feat = self.landcover_encoder(landcover)
        landcover_feat = self.landcover_downsample(landcover_feat)

        uhii_map = self.fusion(lst_feat, pm25_feat, landcover_feat)
        return uhii_map


if __name__ == "__main__":
    # Quick sanity check: run a fake batch through the model to confirm
    # shapes line up end-to-end. This does NOT train anything, it just
    # verifies the architecture is wired correctly.
    batch_size = 2
    height, width = 64, 64

    model = UHIModel(input_size=(height, width))

    fake_lst = torch.randn(batch_size, 1, height, width)
    fake_pm25 = torch.randn(batch_size, 1, height, width)
    fake_landcover = torch.randint(0, 37, (batch_size, height, width))

    output = model(fake_lst, fake_pm25, fake_landcover)
    print(f"Output shape: {output.shape}")  # expect (2, 1, 64, 64)
