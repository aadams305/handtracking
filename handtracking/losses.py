"""SimCC training loss: soft cross-entropy against Gaussian 1D bin targets (X + Y),
plus optional coordinate regression auxiliary loss for fingertip precision."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from handtracking.models.hand_simcc import INPUT_SIZE, NUM_BINS, decode_simcc_soft_argmax


def gaussian_bin_targets_1d(
    coord_px: torch.Tensor,
    input_size: int = INPUT_SIZE,
    num_bins: int = NUM_BINS,
    sigma_bins: float = 1.0,
) -> torch.Tensor:
    """Per-joint 1D distribution over bins, Gaussian in **bin index** space, normalized to sum 1.

    ``coord_px`` is in pixel space ``[0, input_size)``; mean bin index is linearly mapped
    from pixel to ``[0, num_bins-1]`` for 1:1 bin↔pixel training grids.

    sigma_bins=1.0 gives sharper targets than 1.5 — better for fingertip precision.
    """
    device, dtype = coord_px.device, coord_px.dtype
    c = coord_px.clamp(0.0, float(input_size) - 1e-6)
    mu = c * (float(num_bins - 1) / float(input_size))
    bins = torch.arange(num_bins, device=device, dtype=dtype)
    # [..., num_bins]
    diff = bins - mu.unsqueeze(-1)
    g = torch.exp(-0.5 * (diff / sigma_bins) ** 2)
    return g / (g.sum(dim=-1, keepdim=True) + 1e-8)


def gaussian_bin_targets_xy(
    target_xy: torch.Tensor,
    input_size: int = INPUT_SIZE,
    num_bins: int = NUM_BINS,
    sigma_bins: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``target_xy`` ``[B, J, 2]`` → ``tx, ty`` each ``[B, J, num_bins]``."""
    tx = gaussian_bin_targets_1d(target_xy[..., 0], input_size, num_bins, sigma_bins)
    ty = gaussian_bin_targets_1d(target_xy[..., 1], input_size, num_bins, sigma_bins)
    return tx, ty


# Fingertip joint indices (0-indexed into 21-joint topology) for weighted coord loss
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip


class SimCCGaussianSoftCELoss(nn.Module):
    """``-(target * log_softmax(logits)).sum(-1)`` averaged over batch, joints, axes (X+Y).

    Optionally adds a coordinate regression L1 loss on decoded positions to directly
    supervise spatial accuracy, with extra weight on fingertip joints.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        num_bins: int = NUM_BINS,
        sigma_bins: float = 1.0,
        coord_loss_weight: float = 0.5,
        tip_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_bins = num_bins
        self.sigma_bins = sigma_bins
        self.coord_loss_weight = coord_loss_weight
        self.tip_weight = tip_weight

    def forward(self, lx: torch.Tensor, ly: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
        tx, ty = gaussian_bin_targets_xy(
            target_xy, self.input_size, self.num_bins, self.sigma_bins
        )
        lx_n = lx.reshape(-1, self.num_bins)
        ly_n = ly.reshape(-1, self.num_bins)
        tx_n = tx.reshape(-1, self.num_bins)
        ty_n = ty.reshape(-1, self.num_bins)
        loss_x = -(tx_n * F.log_softmax(lx_n, dim=-1)).sum(dim=-1).mean()
        loss_y = -(ty_n * F.log_softmax(ly_n, dim=-1)).sum(dim=-1).mean()
        ce_loss = loss_x + loss_y

        # Coordinate regression auxiliary loss
        if self.coord_loss_weight > 0:
            pred_xy = decode_simcc_soft_argmax(lx, ly, self.input_size, self.num_bins)  # [B, J, 2]
            # Per-joint L1 error
            l1 = (pred_xy - target_xy).abs()  # [B, J, 2]

            # Build per-joint weight: 1.0 for all, extra for fingertips
            B, J, _ = l1.shape
            weight = torch.ones(J, device=l1.device, dtype=l1.dtype)
            for tip_idx in FINGERTIP_INDICES:
                if tip_idx < J:
                    weight[tip_idx] = self.tip_weight
            weight = weight / weight.mean()  # normalize so mean weight = 1

            weighted_l1 = (l1 * weight.view(1, -1, 1)).mean()
            return ce_loss + self.coord_loss_weight * weighted_l1

        return ce_loss
