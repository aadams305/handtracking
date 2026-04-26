"""Hand SimCC model: MobileNetV4 backbone + SimCC head (21 joints, 256 bins per axis).

Bins are 1:1 with pixel columns/rows on the 256×256 letterboxed input: joint x in [0,256)
maps to a 256-way distribution over x-bins (same for y). The head applies two 1×1
convs (X and Y) from backbone features to ``num_joints * num_bins`` channels, reshapes
to ``[B, J, num_bins, H, W]``, then **mean-pools over spatial** (H×W). Pooling averages
the same per-bin logits across all spatial locations, which is standard when the
backbone has not yet collapsed to a single cell (stride 32 → 8×8 on 256 input).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from handtracking.models.mobilenet_v4_conv_small import MobileNetV4ConvSmall
from handtracking.topology import NUM_HAND_JOINTS

INPUT_SIZE = 256
NUM_BINS = 256
NUM_JOINTS = NUM_HAND_JOINTS


def simcc_bin_from_coord(coord_px: torch.Tensor, input_size: int, num_bins: int) -> torch.Tensor:
    """Map pixel coordinate in ``[0, input_size)`` to fractional bin index in ``[0, num_bins-1]``."""
    c = coord_px.clamp(0.0, float(input_size) - 1e-6)
    return c * (float(num_bins - 1) / float(input_size))


def decode_simcc_soft_argmax(
    lx: torch.Tensor,
    ly: torch.Tensor,
    input_size: int = INPUT_SIZE,
    num_bins: int = NUM_BINS,
) -> torch.Tensor:
    """Decode SimCC logits to ``[B, J, 2]`` pixel coordinates (expects 1:1 bin↔pixel span)."""
    px = F.softmax(lx, dim=-1)
    py = F.softmax(ly, dim=-1)
    device, dtype = lx.device, lx.dtype
    bins = torch.arange(num_bins, device=device, dtype=dtype).view(1, 1, -1)
    span = float(input_size)
    scale = span / float(num_bins)
    ex = (px * bins).sum(dim=-1) * scale
    ey = (py * bins).sum(dim=-1) * scale
    return torch.stack((ex, ey), dim=-1)


class SimCCHead(nn.Module):
    """Per-axis 1×1 conv to ``J×num_bins`` logits, spatial mean → ``[B, J, num_bins]``."""

    def __init__(self, in_channels: int, num_joints: int = NUM_JOINTS, num_bins: int = NUM_BINS) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.num_bins = num_bins
        out_ch = num_joints * num_bins
        self.conv_x = nn.Conv2d(in_channels, out_ch, kernel_size=1, bias=True)
        self.conv_y = nn.Conv2d(in_channels, out_ch, kernel_size=1, bias=True)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z: [B, C, H, W] — H,W depend on backbone stride (e.g. 8×8 for 256 input, stride 32)
        b, _, h, w = z.shape
        x = self.conv_x(z).view(b, self.num_joints, self.num_bins, h * w).mean(dim=-1)
        y = self.conv_y(z).view(b, self.num_joints, self.num_bins, h * w).mean(dim=-1)
        return x, y


class HandSimCCNet(nn.Module):
    """Backbone (stride 32) + SimCC head; train/eval on ``INPUT_SIZE``×``INPUT_SIZE`` RGB."""

    def __init__(self, width_mult: float = 0.5) -> None:
        super().__init__()
        self.backbone = MobileNetV4ConvSmall(width_mult=width_mult)
        self.head = SimCCHead(self.backbone.out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        return self.head(z)


if __name__ == "__main__":
    from handtracking.losses import SimCCGaussianSoftCELoss

    B, C, H, W = 2, 128, 8, 8
    feat = torch.randn(B, C, H, W)
    head = SimCCHead(C)
    lx, ly = head(feat)
    assert lx.shape == (B, NUM_JOINTS, NUM_BINS)
    assert ly.shape == (B, NUM_JOINTS, NUM_BINS)
    loss_fn = SimCCGaussianSoftCELoss()
    tgt = torch.rand(B, NUM_JOINTS, 2, device=feat.device, dtype=feat.dtype) * (INPUT_SIZE - 1)
    loss = loss_fn(lx, ly, tgt)
    assert loss.shape == ()
    assert torch.isfinite(loss).item()
    print("SimCCHead + SimCCGaussianSoftCELoss smoke:", float(loss))
