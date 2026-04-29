"""SimCC training loss: soft cross-entropy against Gaussian 1D bin targets (X + Y),
plus optional coordinate regression auxiliary loss for fingertip precision,
plus presence (BCE) and handedness (BCE) losses."""

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
    """Combined loss: SimCC CE + coordinate L1 + presence BCE + handedness BCE.

    The presence and handedness losses are optional — they activate when the model
    returns 4 outputs (lx, ly, presence_logit, handedness_logit) and the dataset
    provides has_hand / handedness targets.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        num_bins: int = NUM_BINS,
        sigma_bins: float = 1.0,
        coord_loss_weight: float = 0.5,
        tip_weight: float = 2.0,
        presence_weight: float = 1.0,
        handedness_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_bins = num_bins
        self.sigma_bins = sigma_bins
        self.coord_loss_weight = coord_loss_weight
        self.tip_weight = tip_weight
        self.presence_weight = presence_weight
        self.handedness_weight = handedness_weight

    def forward(
        self,
        lx: torch.Tensor,
        ly: torch.Tensor,
        target_xy: torch.Tensor,
        presence_logit: torch.Tensor | None = None,
        handedness_logit: torch.Tensor | None = None,
        has_hand: torch.Tensor | None = None,
        handedness_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tx, ty = gaussian_bin_targets_xy(
            target_xy, self.input_size, self.num_bins, self.sigma_bins
        )
        lx_n = lx.reshape(-1, self.num_bins)
        ly_n = ly.reshape(-1, self.num_bins)
        tx_n = tx.reshape(-1, self.num_bins)
        ty_n = ty.reshape(-1, self.num_bins)
        loss_x = -(tx_n * F.log_softmax(lx_n, dim=-1)).sum(dim=-1).mean()
        loss_y = -(ty_n * F.log_softmax(ly_n, dim=-1)).sum(dim=-1).mean()
        total = loss_x + loss_y

        # Coordinate regression auxiliary loss
        if self.coord_loss_weight > 0:
            pred_xy = decode_simcc_soft_argmax(lx, ly, self.input_size, self.num_bins)  # [B, J, 2]
            l1 = (pred_xy - target_xy).abs()  # [B, J, 2]

            B, J, _ = l1.shape
            weight = torch.ones(J, device=l1.device, dtype=l1.dtype)
            for tip_idx in FINGERTIP_INDICES:
                if tip_idx < J:
                    weight[tip_idx] = self.tip_weight
            weight = weight / weight.mean()

            weighted_l1 = (l1 * weight.view(1, -1, 1)).mean()
            total = total + self.coord_loss_weight * weighted_l1

        # Presence loss (binary cross-entropy)
        if presence_logit is not None and has_hand is not None and self.presence_weight > 0:
            pres_loss = F.binary_cross_entropy_with_logits(presence_logit, has_hand)
            total = total + self.presence_weight * pres_loss

        # Handedness loss (binary cross-entropy, masked for unknown labels)
        if handedness_logit is not None and handedness_label is not None and self.handedness_weight > 0:
            # Only supervise samples where handedness is known (not 0.5)
            known_mask = (handedness_label != 0.5)
            if known_mask.any():
                h_logit = handedness_logit[known_mask]
                h_label = handedness_label[known_mask]
                hand_loss = F.binary_cross_entropy_with_logits(h_logit, h_label)
                total = total + self.handedness_weight * hand_loss

        return total
