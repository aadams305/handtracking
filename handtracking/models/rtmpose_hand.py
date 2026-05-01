"""RTMPose-M hand landmark model: CSPNeXt backbone + RTMCCHead (512-bin SimCC).

Drop-in replacement for HandSimCCNet in the training and inference pipelines.

Key differences from the old MobileNetV4-based model:
    - 512 SimCC bins (split_ratio=2.0) instead of 256 (sub-pixel precision)
    - Pixel-space normalisation: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
      (images in [0, 255] range, NOT pre-divided by 255)
    - GAU self-attention in the head instead of simple 1×1 conv + spatial pool
    - ~13M params (vs ~1.2M for MobileNetV4 w/0.75)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from handtracking.models.cspnext import CSPNeXt
from handtracking.models.rtmcc_head import RTMCCHead
from handtracking.topology import NUM_HAND_JOINTS

INPUT_SIZE = 256
NUM_BINS = 512
NUM_JOINTS = NUM_HAND_JOINTS
SIMCC_SPLIT_RATIO = 2.0

MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)


def decode_simcc(
    pred_x: Tensor,
    pred_y: Tensor,
    input_size: int = INPUT_SIZE,
    simcc_split_ratio: float = SIMCC_SPLIT_RATIO,
) -> Tensor:
    """Decode SimCC logits to [B, J, 2] pixel coordinates."""
    num_bins = pred_x.shape[-1]
    px = F.softmax(pred_x, dim=-1)
    py = F.softmax(pred_y, dim=-1)
    device, dtype = pred_x.device, pred_x.dtype
    bins = torch.arange(num_bins, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    ex = (px * bins).sum(dim=-1) / simcc_split_ratio
    ey = (py * bins).sum(dim=-1) / simcc_split_ratio
    return torch.stack((ex, ey), dim=-1)


def simcc_confidence(pred_x: Tensor, pred_y: Tensor) -> Tensor:
    """Peakedness-based confidence [B] from SimCC logits, no extra head needed."""
    px = F.softmax(pred_x, dim=-1)
    py = F.softmax(pred_y, dim=-1)
    peak_x = px.max(dim=-1).values.mean(dim=-1)
    peak_y = py.max(dim=-1).values.mean(dim=-1)
    num_bins = pred_x.shape[-1]
    raw = (peak_x + peak_y) / 2.0
    floor = 1.0 / num_bins
    return ((raw - floor) / (1.0 - floor)).clamp(0.0, 1.0)


class RTMPoseHand(nn.Module):
    """RTMPose-M for 21-joint hand landmark estimation.

    Architecture: CSPNeXt-M backbone (stride 32) -> RTMCCHead (GAU + SimCC 512 bins).

    This model does NOT include the presence/handedness auxiliary heads from the
    old HandSimCCNet — in the two-stage pipeline, the palm detector handles presence
    and handedness is inferred from the wrist/thumb geometry.
    """

    def __init__(
        self,
        widen_factor: float = 0.75,
        deepen_factor: float = 0.67,
        num_joints: int = NUM_JOINTS,
        input_size: int = INPUT_SIZE,
        simcc_split_ratio: float = SIMCC_SPLIT_RATIO,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio
        self.num_bins = int(input_size * simcc_split_ratio)

        self.backbone = CSPNeXt(
            widen_factor=widen_factor,
            deepen_factor=deepen_factor,
            out_indices=(4,),
            channel_attention=True,
        )

        feat_h = input_size // 32
        feat_w = input_size // 32

        self.head = RTMCCHead(
            in_channels=self.backbone.out_channels,
            out_channels=num_joints,
            input_size=(input_size, input_size),
            in_featuremap_size=(feat_w, feat_h),
            simcc_split_ratio=simcc_split_ratio,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (pred_x [B, J, 512], pred_y [B, J, 512])."""
        feats = self.backbone(x)
        return self.head(feats)

    def forward_decode(self, x: Tensor) -> Tensor:
        """Forward + decode to [B, J, 2] pixel coords."""
        px, py = self.forward(x)
        return decode_simcc(px, py, self.input_size, self.simcc_split_ratio)

    def init_weights(self) -> None:
        self.backbone.init_weights()
        self.head.init_weights()


if __name__ == "__main__":
    model = RTMPoseHand()
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    px, py = model(x)
    print(f"pred_x: {px.shape}, pred_y: {py.shape}")
    coords = decode_simcc(px, py)
    print(f"decoded coords: {coords.shape}")
    conf = simcc_confidence(px, py)
    print(f"confidence: {conf.shape}")
    n = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n:,}")
