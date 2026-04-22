"""Adaptive Wing Loss + label-smoothed SimCC cross-entropy."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def adaptive_wing_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    theta: float = 0.5,
    alpha: float = 2.1,
    omega: float = 14.0,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Adaptive Wing Loss; pred/target share shape."""
    delta = (target - pred).abs()
    t = torch.as_tensor(theta, device=pred.device, dtype=pred.dtype)
    e = torch.as_tensor(epsilon, device=pred.device, dtype=pred.dtype)
    o = torch.as_tensor(omega, device=pred.device, dtype=pred.dtype)
    a = torch.as_tensor(alpha, device=pred.device, dtype=pred.dtype)
    A = o * (1.0 / (1.0 + torch.pow(t / e, a - 1.0)))
    C = t * torch.log1p(torch.pow(t / e, a - 1.0)) - A * t
    y = torch.where(
        delta < t,
        o * torch.log1p(torch.pow(delta / e, a - 1.0)),
        A * delta - C,
    )
    return y.mean()


def simcc_label_smoothing_ce(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    num_bins: int,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    logits: [B, J, Bins], target_bins: [B, J] long indices
    """
    b, j, bins = logits.shape
    log_p = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_p)
        true_dist.scatter_(-1, target_bins.unsqueeze(-1), 1.0)
        true_dist = true_dist * (1.0 - smoothing) + smoothing / float(bins)
    return -(true_dist * log_p).sum(dim=-1).mean()


class SimCCAdaptiveWingLoss(nn.Module):
    """SimCC: smoothed CE on bins + AWing on soft-argmax decoded coords."""

    def __init__(
        self,
        num_bins: int = 512,
        input_size: int = 256,
        label_smoothing: float = 0.1,
        aw_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.input_size = input_size
        self.label_smoothing = label_smoothing
        self.aw_weight = aw_weight

    def forward(
        self,
        lx: torch.Tensor,
        ly: torch.Tensor,
        target_xy: torch.Tensor,
        decode_fn,
    ) -> torch.Tensor:
        """
        target_xy: [B, J, 2] pixel coords in [0, input_size)
        """
        tx = target_xy[..., 0]
        ty = target_xy[..., 1]
        bx = (tx.clamp(0, self.input_size - 1e-4) * (self.num_bins / float(self.input_size))).long()
        by = (ty.clamp(0, self.input_size - 1e-4) * (self.num_bins / float(self.input_size))).long()
        ce_x = simcc_label_smoothing_ce(lx, bx, self.num_bins, self.label_smoothing)
        ce_y = simcc_label_smoothing_ce(ly, by, self.num_bins, self.label_smoothing)
        pred_xy = decode_fn(lx, ly)
        aw = adaptive_wing_loss(pred_xy, target_xy)
        return ce_x + ce_y + self.aw_weight * aw
