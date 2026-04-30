"""Hand SimCC model: MobileNetV4 backbone + SimCC head (21 joints, 256 bins per axis).

Bins are 1:1 with pixel columns/rows on the 256×256 letterboxed input: joint x in [0,256)
maps to a 256-way distribution over x-bins (same for y). The head applies two 1×1
convs (X and Y) from backbone features to ``num_joints * num_bins`` channels, reshapes
to ``[B, J, num_bins, H, W]``, then **mean-pools over spatial** (H×W). Pooling averages
the same per-bin logits across all spatial locations, which is standard when the
backbone has not yet collapsed to a single cell (stride 32 → 8×8 on 256 input).

Additionally includes:
- **Presence head**: sigmoid confidence score [0, 1] — is a hand visible at all?
- **Handedness head**: sigmoid score [0, 1] — 0 = Left, 1 = Right
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


def simcc_confidence(
    lx: torch.Tensor,
    ly: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample confidence from SimCC logit peakedness.

    Returns ``[B]`` float in roughly [0, 1]. When the model is confident about
    joint locations the softmax distributions are sharply peaked (high max prob).
    When there is no hand the distributions are near-uniform (max ≈ 1/num_bins).

    This works as a zero-cost confidence proxy without any additional head or
    retraining — useful for gating output at inference time.
    """
    px = F.softmax(lx, dim=-1)  # [B, J, bins]
    py = F.softmax(ly, dim=-1)
    # Mean of max-probabilities across all joints and both axes
    peak_x = px.max(dim=-1).values.mean(dim=-1)  # [B]
    peak_y = py.max(dim=-1).values.mean(dim=-1)
    # Normalize: uniform peak = 1/num_bins ≈ 0.004, strong peak ≈ 0.3-0.8
    num_bins = lx.shape[-1]
    raw = (peak_x + peak_y) / 2.0
    # Map from [1/num_bins, 1] to [0, 1] roughly
    floor = 1.0 / num_bins
    conf = (raw - floor) / (1.0 - floor)
    return conf.clamp(0.0, 1.0)


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


class PresenceHandednessHead(nn.Module):
    """Lightweight head predicting hand presence (sigmoid) and handedness (sigmoid).

    Architecture: GAP → FC(in, 32) → ReLU → FC(32, 2) → [presence_logit, handedness_logit]
    Presence:   sigmoid(out[0]) → confidence that a hand is in frame
    Handedness: sigmoid(out[1]) → 0 = Left, 1 = Right (only meaningful when presence > threshold)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (presence_logit [B], handedness_logit [B])."""
        x = self.pool(z).flatten(1)  # [B, C]
        out = self.fc(x)  # [B, 2]
        return out[:, 0], out[:, 1]


class HandSimCCNet(nn.Module):
    """Backbone (stride 32) + deconv upsample + SimCC head + presence/handedness head.

    Train/eval on ``INPUT_SIZE``×``INPUT_SIZE`` RGB.

    The deconv upsample lifts the backbone's 8×8 feature map to 16×16,
    giving the SimCC head 4× more spatial cells to average over.  This
    significantly improves landmark localization without meaningful
    parameter or latency cost.
    """

    def __init__(self, width_mult: float = 0.75) -> None:
        super().__init__()
        self.backbone = MobileNetV4ConvSmall(width_mult=width_mult)
        ch = self.backbone.out_channels
        # Upsample 8×8 → 16×16 for finer spatial detail before SimCC
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU6(inplace=True),
        )
        self.head = SimCCHead(ch)
        self.aux_head = PresenceHandednessHead(ch)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (lx, ly, presence_logit, handedness_logit)."""
        z = self.backbone(x)
        # Presence/handedness use raw backbone features (GAP doesn't need spatial detail)
        presence, handedness = self.aux_head(z)
        # SimCC head uses upsampled features for finer localization
        z_up = self.upsample(z)
        lx, ly = self.head(z_up)
        return lx, ly, presence, handedness

    def forward_keypoints_only(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Legacy forward: returns only (lx, ly). For backward compat with old checkpoints."""
        z = self.backbone(x)
        z_up = self.upsample(z)
        return self.head(z_up)


if __name__ == "__main__":
    from handtracking.losses import SimCCGaussianSoftCELoss

    B, C, H, W = 2, 128, 8, 8
    feat = torch.randn(B, C, H, W)
    head = SimCCHead(C)
    lx, ly = head(feat)
    assert lx.shape == (B, NUM_JOINTS, NUM_BINS)
    assert ly.shape == (B, NUM_JOINTS, NUM_BINS)

    # Test presence/handedness head
    aux = PresenceHandednessHead(C)
    pres, hand = aux(feat)
    assert pres.shape == (B,)
    assert hand.shape == (B,)
    print(f"Presence logits: {pres}")
    print(f"Handedness logits: {hand}")

    # Test confidence from SimCC
    conf = simcc_confidence(lx, ly)
    assert conf.shape == (B,)
    print(f"SimCC confidence: {conf}")

    # Test full model
    model = HandSimCCNet(width_mult=0.75)
    inp = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    lx, ly, p, h = model(inp)
    print(f"Full model: lx={lx.shape} ly={ly.shape} presence={p.shape} handedness={h.shape}")

    loss_fn = SimCCGaussianSoftCELoss()
    tgt = torch.rand(B, NUM_JOINTS, 2, device=feat.device, dtype=feat.dtype) * (INPUT_SIZE - 1)
    loss = loss_fn(lx, ly, tgt)
    assert loss.shape == ()
    assert torch.isfinite(loss).item()
    print("SimCCHead + SimCCGaussianSoftCELoss smoke:", float(loss))
