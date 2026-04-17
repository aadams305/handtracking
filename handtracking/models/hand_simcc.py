"""10-joint SimCC head: horizontal (X) and vertical (Y) 1D classification, 320 bins each."""

from __future__ import annotations

import torch
import torch.nn as nn

from handtracking.models.mobilenet_v4_conv_small import MobileNetV4ConvSmall
from handtracking.topology import NUM_HAND_JOINTS

NUM_JOINTS = NUM_HAND_JOINTS
NUM_BINS = 320  # 2 bins per pixel for 160px span
INPUT_SIZE = 160


class SimCCHead(nn.Module):
    def __init__(self, in_channels: int, num_joints: int = NUM_JOINTS, num_bins: int = NUM_BINS):
        super().__init__()
        self.num_joints = num_joints
        self.num_bins = num_bins
        spatial_size = (INPUT_SIZE // 32) ** 2  # 25 for a 160px input
        
        # Maps backbone features explicitly to spatial heatmaps per-joint without loss of local topography!
        self.feat_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_joints, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_joints),
            nn.ReLU(inplace=True)
        )
        
        # Translates coarse 5x5 spatial responses into high-resolution 1D 320 SimCC bins natively
        self.x_proj = nn.Linear(spatial_size, num_bins)
        self.y_proj = nn.Linear(spatial_size, num_bins)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, H, W] -> e.g., [B, 64, 5, 5]
        b = x.size(0)
        feat = self.feat_conv(x)  # [B, 10, 5, 5]
        feat = feat.view(b, self.num_joints, -1)  # [B, 10, 25]
        
        lx = self.x_proj(feat)  # [B, 10, 320]
        ly = self.y_proj(feat)  # [B, 10, 320]
        return lx, ly


class HandSimCCNet(nn.Module):
    def __init__(self, width_mult: float = 0.5) -> None:
        super().__init__()
        self.backbone = MobileNetV4ConvSmall(width_mult=width_mult)
        self.head = SimCCHead(self.backbone.out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        return self.head(feat)


def decode_simcc_soft_argmax(
    lx: torch.Tensor, ly: torch.Tensor, input_size: int = INPUT_SIZE
) -> torch.Tensor:
    """lx, ly: [B, J, Bins] -> coords [B, J, 2] in pixel space [0, input_size)."""
    b, j, bins = lx.shape
    device = lx.device
    xs = torch.arange(bins, device=device, dtype=lx.dtype) * (input_size / float(bins))
    px = torch.softmax(lx, dim=-1)
    py = torch.softmax(ly, dim=-1)
    x_coord = (px * xs.view(1, 1, -1)).sum(dim=-1)
    y_coord = (py * xs.view(1, 1, -1)).sum(dim=-1)
    return torch.stack([x_coord, y_coord], dim=-1)


def simcc_bin_from_coord(
    coord: torch.Tensor, num_bins: int = NUM_BINS, input_size: int = INPUT_SIZE
) -> torch.Tensor:
    """coord in [0, input_size) -> bin index float for target."""
    return coord.clamp(0, input_size - 1e-4) * (num_bins / float(input_size))
