"""
Attention U-Net fusion module.

Takes the three encoder outputs (LST, PM2.5, land cover), concatenates
them, and uses attention gates to let the network weight each stream's
contribution per spatial location before producing the final full
resolution Urban Heat Island Intensity (UHII) map.

This is deliberately written as a compact single-stage fusion (concat +
attention gate + upsample) rather than a full multi-level U-Net encoder,
since the three input streams already act as the "encoder" side. This
module represents the "decoder + attention" side that turns fused
features back into a full-resolution prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Learns a per-pixel weight (0 to 1) for a feature map, conditioned on
    a "gating signal" from the fused representation. Locations the gate
    considers unimportant get suppressed; important ones pass through.
    """

    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, gating_signal: torch.Tensor) -> torch.Tensor:
        theta_x = self.theta(x)
        phi_g = self.phi(gating_signal)
        # Resize gating signal to match x's spatial size if needed
        if phi_g.shape[-2:] != theta_x.shape[-2:]:
            phi_g = F.interpolate(phi_g, size=theta_x.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.relu(theta_x + phi_g)
        attention = self.sigmoid(self.psi(attention))
        return x * attention


class AttentionFusionUNet(nn.Module):
    """
    Fuses the three encoder streams and decodes to a full-resolution
    UHII prediction map.

    Inputs are the outputs of RasterEncoder (x2, for LST and PM2.5) and
    LandCoverEncoder (x1), all at the same downsampled spatial size.
    """

    def __init__(self, lst_channels: int, pm25_channels: int, landcover_channels: int, target_size: tuple):
        super().__init__()
        self.target_size = target_size  # (H, W) of the original input, for final upsampling

        fused_channels = lst_channels + pm25_channels + landcover_channels
        inter_channels = fused_channels // 2

        # A shared "gating signal" derived from the concatenated features,
        # used to condition each attention gate.
        self.gating_conv = nn.Conv2d(fused_channels, inter_channels, kernel_size=1)

        self.gate_lst = AttentionGate(lst_channels, inter_channels, inter_channels // 2)
        self.gate_pm25 = AttentionGate(pm25_channels, inter_channels, inter_channels // 2)
        self.gate_landcover = AttentionGate(landcover_channels, inter_channels, inter_channels // 2)

        # Decoder: progressively upsample back toward full resolution.
        self.decoder = nn.Sequential(
            nn.Conv2d(fused_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # Final 1x1 conv: collapses to a single-channel UHII prediction map.
        self.output_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, lst_feat: torch.Tensor, pm25_feat: torch.Tensor, landcover_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([lst_feat, pm25_feat, landcover_feat], dim=1)
        gating_signal = self.gating_conv(fused)

        # Re-weight each stream individually using the attention gates
        lst_weighted = self.gate_lst(lst_feat, gating_signal)
        pm25_weighted = self.gate_pm25(pm25_feat, gating_signal)
        landcover_weighted = self.gate_landcover(landcover_feat, gating_signal)

        gated_fused = torch.cat([lst_weighted, pm25_weighted, landcover_weighted], dim=1)

        decoded = self.decoder(gated_fused)

        # Ensure exact match to the original input resolution
        if decoded.shape[-2:] != self.target_size:
            decoded = F.interpolate(decoded, size=self.target_size, mode="bilinear", align_corners=False)

        uhii_map = self.output_conv(decoded)
        return uhii_map
