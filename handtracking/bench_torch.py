"""Log mean PyTorch inference time (ms) for HandSimCCNet — useful without NCNN on host."""

from __future__ import annotations

import argparse
import time

import torch

from handtracking.models.hand_simcc import HandSimCCNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-mult", type=float, default=0.5)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cpu")
    m = HandSimCCNet(width_mult=args.width_mult).eval().to(device)
    x = torch.randn(1, 3, 160, 160, device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            m(x)
        t0 = time.perf_counter()
        for _ in range(args.runs):
            m(x)
        t1 = time.perf_counter()
    ms = (t1 - t0) / args.runs * 1000.0
    print(f"inference_time_ms_mean={ms:.3f} (torch cpu, batch=1)")


if __name__ == "__main__":
    main()
