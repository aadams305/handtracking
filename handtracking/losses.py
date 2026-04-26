"""SimCC training loss: soft cross-entropy against Gaussian 1D bin targets (X + Y)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from handtracking.models.hand_simcc import INPUT_SIZE, NUM_BINS


def gaussian_bin_targets_1d(
    coord_px: torch.Tensor,
    input_size: int = INPUT_SIZE,
    num_bins: int = NUM_BINS,
    sigma_bins: float = 1.5,
) -> torch.Tensor:
    """Per-joint 1D distribution over bins, Gaussian in **bin index** space, normalized to sum 1.

    ``coord_px`` is in pixel space ``[0, input_size)``; mean bin index is linearly mapped
    from pixel to ``[0, num_bins-1]`` for 1:1 bin↔pixel training grids.
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
    sigma_bins: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``target_xy`` ``[B, J, 2]`` → ``tx, ty`` each ``[B, J, num_bins]``."""
    tx = gaussian_bin_targets_1d(target_xy[..., 0], input_size, num_bins, sigma_bins)
    ty = gaussian_bin_targets_1d(target_xy[..., 1], input_size, num_bins, sigma_bins)
    return tx, ty


class SimCCGaussianSoftCELoss(nn.Module):
    """``-(target * log_softmax(logits)).sum(-1)`` averaged over batch, joints, axes (X+Y)."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        num_bins: int = NUM_BINS,
        sigma_bins: float = 1.5,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_bins = num_bins
        self.sigma_bins = sigma_bins

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
        return loss_x + loss_y
