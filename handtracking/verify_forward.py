"""Phase 2 verification: single forward pass, print SimCC tensor shapes."""

from __future__ import annotations

import torch

from handtracking.losses import SimCCGaussianSoftCELoss
from handtracking.models.hand_simcc import HandSimCCNet, INPUT_SIZE, NUM_BINS, NUM_JOINTS


def main() -> None:
    m = HandSimCCNet(width_mult=0.5)
    m.eval()
    x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        lx, ly = m(x)
    print("input:", tuple(x.shape))
    exp = (1, NUM_JOINTS, NUM_BINS)
    print("simcc_x (lx):", tuple(lx.shape), f"expected {exp}:", lx.shape == exp)
    print("simcc_y (ly):", tuple(ly.shape), f"expected {exp}:", ly.shape == exp)
    assert lx.shape == exp
    assert ly.shape == exp

    m.train()
    lx2, ly2 = m(x)
    tgt = torch.rand(1, NUM_JOINTS, 2, device=x.device, dtype=x.dtype) * (INPUT_SIZE - 1)
    loss_fn = SimCCGaussianSoftCELoss()
    loss = loss_fn(lx2, ly2, tgt)
    print("loss_smoke:", float(loss.detach()))
    assert torch.isfinite(loss).item()


if __name__ == "__main__":
    main()
