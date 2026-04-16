"""Phase 2 verification: single forward pass, print SimCC tensor shapes."""

from __future__ import annotations

import torch

from handtracking.models.hand_simcc import HandSimCCNet, NUM_BINS, NUM_JOINTS


def main() -> None:
    m = HandSimCCNet(width_mult=0.5)
    m.eval()
    x = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        lx, ly = m(x)
    print("input:", tuple(x.shape))
    print("simcc_x (lx):", tuple(lx.shape), f"expected (1, {NUM_JOINTS}, 320):", lx.shape == (1, NUM_JOINTS, NUM_BINS))
    print("simcc_y (ly):", tuple(ly.shape), f"expected (1, {NUM_JOINTS}, 320):", ly.shape == (1, NUM_JOINTS, NUM_BINS))
    assert lx.shape == (1, NUM_JOINTS, NUM_BINS)
    assert ly.shape == (1, NUM_JOINTS, NUM_BINS)


if __name__ == "__main__":
    main()
