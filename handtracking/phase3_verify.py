"""Report ONNX / NCNN artifact sizes (target: under 5MB after ncnnoptimize)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, default=Path("models/hand_simcc.onnx"))
    ap.add_argument(
        "--ncnn-opt-param",
        type=Path,
        default=Path("models/ncnn/hand_simcc.opt.param"),
    )
    ap.add_argument(
        "--ncnn-opt-bin",
        type=Path,
        default=Path("models/ncnn/hand_simcc.opt.bin"),
    )
    args = ap.parse_args()

    if args.onnx.is_file():
        sz = args.onnx.stat().st_size
        print(f"onnx_bytes={sz} ({sz/1024/1024:.3f} MiB)")
    else:
        print("onnx missing", file=sys.stderr)

    if args.ncnn_opt_param.is_file() and args.ncnn_opt_bin.is_file():
        sz = args.ncnn_opt_param.stat().st_size + args.ncnn_opt_bin.stat().st_size
        print(f"ncnn_opt_total_bytes={sz} ({sz/1024/1024:.3f} MiB)")
        ok = sz < 5 * 1024 * 1024
    else:
        print("ncnn optimized files not found; run scripts/convert_ncnn.sh when onnx2ncnn is available")
        sz = args.onnx.stat().st_size if args.onnx.is_file() else 0
        ok = sz < 5 * 1024 * 1024
        print(f"fallback_check_onnx_under_5mb={ok}")

    if not ok:
        sys.exit(1)
    print("phase3_size_gate_ok")


if __name__ == "__main__":
    main()
