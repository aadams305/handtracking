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
        
        # 160px input -> stride 32 -> 5x5 spatial grid
        spatial_area = (INPUT_SIZE // 32) ** 2  # 25
        flattened_features = in_channels * spatial_area  # 64 * 25 = 1600
        
        # Double Dense layers decouple output dimension mappings avoiding NCNN slice corruption entirely
        self.x_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, num_joints * num_bins)
        )
        self.y_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, num_joints * num_bins)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        lx = self.x_proj(x).view(b, self.num_joints, self.num_bins)
        ly = self.y_proj(x).view(b, self.num_joints, self.num_bins)
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
