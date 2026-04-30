"""Convert ONNX models to RKNN format for Rockchip RK3588 NPU inference.

Requires rknn-toolkit2 (pip install rknn-toolkit2 on x86 host, or
rknn-toolkit-lite2 on the Orange Pi itself).

Two models to convert:
  1. SimCC landmark model:  models/hand_simcc.onnx → models/hand_simcc.rknn
  2. Palm detector:         models/palm_det.onnx   → models/palm_det.rknn

Usage:
    python export_rknn.py --model landmark --onnx models/hand_simcc.onnx --out models/hand_simcc.rknn
    python export_rknn.py --model palm     --onnx models/palm_det.onnx   --out models/palm_det.rknn
    python export_rknn.py --model all

On the Orange Pi 5 (aarch64), use rknn-toolkit-lite2 instead of rknn-toolkit2.
This script auto-detects which is available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def convert_onnx_to_rknn(
    onnx_path: str,
    rknn_path: str,
    input_size: int,
    quantize: bool = True,
    dataset_txt: str | None = None,
    target_platform: str = "rk3588",
) -> None:
    """Convert a single ONNX model to RKNN."""
    try:
        from rknn.api import RKNN
    except ImportError:
        print(
            "ERROR: rknn-toolkit2 not found.\n"
            "  x86 host:  pip install rknn-toolkit2\n"
            "  aarch64:   pip install rknn-toolkit-lite2\n"
            "See https://github.com/airockchip/rknn-toolkit2",
            file=sys.stderr,
        )
        sys.exit(1)

    rknn = RKNN(verbose=True)

    print(f"Configuring RKNN for {target_platform}...", flush=True)
    rknn.config(
        mean_values=[IMAGENET_MEAN],
        std_values=[IMAGENET_STD],
        target_platform=target_platform,
        quantized_algorithm="normal",
        quantized_method="channel",
    )

    print(f"Loading ONNX: {onnx_path}", flush=True)
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"Failed to load ONNX model (ret={ret})", file=sys.stderr)
        sys.exit(1)

    print("Building RKNN model...", flush=True)
    if quantize and dataset_txt:
        ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    else:
        if quantize:
            print("WARNING: --quantize requested but no --dataset provided. Building without quantization.",
                  file=sys.stderr)
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print(f"Failed to build RKNN model (ret={ret})", file=sys.stderr)
        sys.exit(1)

    print(f"Exporting to {rknn_path}", flush=True)
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"Failed to export RKNN (ret={ret})", file=sys.stderr)
        sys.exit(1)

    rknn.release()
    print(f"Successfully converted: {onnx_path} → {rknn_path}")


def generate_calibration_dataset(
    manifest_path: str,
    output_txt: str,
    max_images: int = 100,
) -> None:
    """Generate a calibration dataset text file (list of image paths) for RKNN quantization."""
    from pathlib import Path
    import random

    manifest = Path(manifest_path)
    if not manifest.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return

    paths: list[str] = []
    with manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            import json
            d = json.loads(line)
            paths.append(d["image_path"])

    random.seed(42)
    if len(paths) > max_images:
        paths = random.sample(paths, max_images)

    with open(output_txt, "w") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"Wrote {len(paths)} calibration paths to {output_txt}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert ONNX models to RKNN for RK3588 NPU")
    ap.add_argument(
        "--model", choices=("landmark", "palm", "all"), default="landmark",
        help="Which model to convert",
    )
    ap.add_argument("--onnx", type=Path, default=None, help="Input ONNX path (auto-set for --model all)")
    ap.add_argument("--out", type=Path, default=None, help="Output RKNN path (auto-set for --model all)")
    ap.add_argument("--target", default="rk3588", help="Target platform (rk3588, rk3566, etc.)")
    ap.add_argument("--quantize", action="store_true", help="Enable INT8 quantization")
    ap.add_argument("--dataset", type=str, default=None, help="Calibration dataset txt for quantization")
    ap.add_argument("--manifest", type=Path, default=Path("data/distilled/manifest.jsonl"),
                     help="Manifest for auto-generating calibration dataset")
    ap.add_argument("--gen-calib", action="store_true", help="Generate calibration dataset from manifest")
    args = ap.parse_args()

    if args.gen_calib:
        generate_calibration_dataset(
            str(args.manifest),
            "data/rknn_calibration.txt",
        )
        if args.dataset is None:
            args.dataset = "data/rknn_calibration.txt"

    if args.model == "all":
        models = [
            ("landmark", Path("models/hand_simcc.onnx"), Path("models/hand_simcc.rknn"), 256),
            ("palm", Path("models/palm_det.onnx"), Path("models/palm_det.rknn"), 192),
        ]
        for name, onnx_p, rknn_p, size in models:
            if not onnx_p.is_file():
                print(f"Skipping {name}: {onnx_p} not found")
                continue
            rknn_p.parent.mkdir(parents=True, exist_ok=True)
            convert_onnx_to_rknn(
                str(onnx_p), str(rknn_p), size,
                quantize=args.quantize,
                dataset_txt=args.dataset,
                target_platform=args.target,
            )
    else:
        if args.model == "landmark":
            onnx_p = args.onnx or Path("models/hand_simcc.onnx")
            rknn_p = args.out or Path("models/hand_simcc.rknn")
            size = 256
        else:
            onnx_p = args.onnx or Path("models/palm_det.onnx")
            rknn_p = args.out or Path("models/palm_det.rknn")
            size = 192

        if not onnx_p.is_file():
            raise SystemExit(f"ONNX file not found: {onnx_p}")
        rknn_p.parent.mkdir(parents=True, exist_ok=True)
        convert_onnx_to_rknn(
            str(onnx_p), str(rknn_p), size,
            quantize=args.quantize,
            dataset_txt=args.dataset,
            target_platform=args.target,
        )


if __name__ == "__main__":
    main()
