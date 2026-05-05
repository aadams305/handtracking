"""Convert ONNX models to RKNN format for Rockchip RK3588 NPU inference.

Supports FP16 and INT8 precision modes, with optional benchmarking.

Usage:
    python export_rknn.py --model landmark --precision fp16
    python export_rknn.py --model landmark --precision int8 --dataset data/rknn_calibration.txt
    python export_rknn.py --model landmark --precision both --dataset data/rknn_calibration.txt
    python export_rknn.py --model landmark --precision both --dataset data/rknn_calibration.txt --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

RTMPOSE_MEAN = [123.675, 116.28, 103.53]
RTMPOSE_STD = [58.395, 57.12, 57.375]

YOLO_MEAN = [0.0, 0.0, 0.0]
YOLO_STD = [255.0, 255.0, 255.0]


def convert_onnx_to_rknn(
    onnx_path: str,
    rknn_path: str,
    input_size: int,
    quantize: bool = False,
    dataset_txt: str | None = None,
    target_platform: str = "rk3588",
    mean_values: list[float] | None = None,
    std_values: list[float] | None = None,
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

    if mean_values is None:
        mean_values = RTMPOSE_MEAN
    if std_values is None:
        std_values = RTMPOSE_STD

    rknn = RKNN(verbose=True)

    precision_str = "INT8" if quantize else "FP16"
    print(f"Configuring RKNN for {target_platform} ({precision_str})...", flush=True)

    config_kwargs = {
        "mean_values": [mean_values],
        "std_values": [std_values],
        "target_platform": target_platform,
    }
    if quantize:
        config_kwargs["quantized_algorithm"] = "normal"
        config_kwargs["quantized_method"] = "channel"

    rknn.config(**config_kwargs)

    print(f"Loading ONNX: {onnx_path}", flush=True)
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"Failed to load ONNX model (ret={ret})", file=sys.stderr)
        sys.exit(1)

    print(f"Building RKNN model ({precision_str})...", flush=True)
    if quantize and dataset_txt:
        ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    else:
        if quantize and not dataset_txt:
            print("WARNING: INT8 requested but no --dataset provided. Building FP16 instead.",
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
    print(f"Successfully converted: {onnx_path} -> {rknn_path}")


def benchmark_rknn(rknn_path: str, input_size: int, iterations: int = 100) -> float:
    """Load an RKNN model and benchmark inference speed. Returns avg ms."""
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        try:
            from rknn.api import RKNN as RKNNLite
        except ImportError:
            print("Cannot benchmark: no RKNN runtime available.", file=sys.stderr)
            return -1.0

    rknn = RKNNLite()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        print(f"Failed to load {rknn_path} for benchmark", file=sys.stderr)
        return -1.0

    try:
        ret = rknn.init_runtime(core_mask=0x07)
    except TypeError:
        ret = rknn.init_runtime()
    if ret != 0:
        print(f"Failed to init runtime for benchmark", file=sys.stderr)
        return -1.0

    dummy = np.random.randint(0, 256, (1, input_size, input_size, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        rknn.inference(inputs=[dummy])

    t0 = time.perf_counter()
    for _ in range(iterations):
        rknn.inference(inputs=[dummy])
    elapsed = (time.perf_counter() - t0) / iterations * 1000

    rknn.release()
    return elapsed


def generate_calibration_dataset(
    manifest_path: str,
    output_txt: str,
    max_images: int = 100,
) -> None:
    """Generate a calibration dataset text file for RKNN INT8 quantization."""
    import json
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
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--target", default="rk3588")
    ap.add_argument(
        "--precision", choices=("fp16", "int8", "both"), default="fp16",
        help="Precision mode: fp16, int8, or both",
    )
    ap.add_argument("--dataset", type=str, default=None, help="Calibration dataset txt (for INT8)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Manifest for auto-generating calibration dataset")
    ap.add_argument("--gen-calib", action="store_true")
    ap.add_argument("--benchmark", action="store_true",
                    help="Run inference benchmark after conversion")
    ap.add_argument("--benchmark-iters", type=int, default=100)
    # Legacy compat
    ap.add_argument("--quantize", action="store_true",
                    help="(deprecated) same as --precision int8")
    args = ap.parse_args()

    # Handle legacy --quantize flag
    if args.quantize and args.precision == "fp16":
        args.precision = "int8"

    if args.gen_calib and args.manifest:
        generate_calibration_dataset(str(args.manifest), "data/rknn_calibration.txt")
        if args.dataset is None:
            args.dataset = "data/rknn_calibration.txt"

    def _palm_onnx_path() -> Path:
        """Prefer YOLO11n palm export, fall back to legacy palm_det.onnx."""
        if args.onnx is not None:
            return args.onnx
        for candidate in (Path("models/yolo_palm.onnx"), Path("models/palm_det.onnx")):
            if candidate.is_file():
                return candidate
        return Path("models/yolo_palm.onnx")

    def get_model_configs() -> list[tuple[str, Path, Path, int, list, list]]:
        """Returns list of (name, onnx_path, rknn_base_path, size, mean, std)."""
        if args.model == "all":
            palm_onnx = _palm_onnx_path()
            palm_base = palm_onnx.with_suffix("")
            return [
                ("landmark", Path("models/rtmpose_hand.onnx"), Path("models/rtmpose_hand"), 256,
                 RTMPOSE_MEAN, RTMPOSE_STD),
                ("palm", palm_onnx, palm_base, 192, YOLO_MEAN, YOLO_STD),
            ]
        elif args.model == "landmark":
            onnx_p = args.onnx or Path("models/rtmpose_hand.onnx")
            base = args.out.with_suffix("") if args.out else Path("models/rtmpose_hand")
            return [("landmark", onnx_p, base, 256, RTMPOSE_MEAN, RTMPOSE_STD)]
        else:
            onnx_p = _palm_onnx_path()
            base = args.out.with_suffix("") if args.out else onnx_p.with_suffix("")
            return [("palm", onnx_p, base, 192, YOLO_MEAN, YOLO_STD)]

    configs = get_model_configs()
    exported: list[tuple[str, str, int]] = []

    for name, onnx_p, base_path, size, mean, std in configs:
        if not onnx_p.is_file():
            print(f"Skipping {name}: {onnx_p} not found")
            continue

        base_path.parent.mkdir(parents=True, exist_ok=True)
        precisions = []
        if args.precision in ("fp16", "both"):
            precisions.append(("fp16", False))
        if args.precision in ("int8", "both"):
            precisions.append(("int8", True))

        for prec_name, do_quant in precisions:
            if args.precision == "both":
                rknn_p = base_path.with_name(f"{base_path.name}_{prec_name}.rknn")
            else:
                rknn_p = base_path.with_suffix(".rknn")

            convert_onnx_to_rknn(
                str(onnx_p), str(rknn_p), size,
                quantize=do_quant, dataset_txt=args.dataset if do_quant else None,
                target_platform=args.target,
                mean_values=mean, std_values=std,
            )
            exported.append((name, str(rknn_p), size))

    if args.benchmark and exported:
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        for name, rknn_p, size in exported:
            avg_ms = benchmark_rknn(rknn_p, size, args.benchmark_iters)
            if avg_ms > 0:
                print(f"  {Path(rknn_p).name:40s}  {avg_ms:.2f} ms/frame  ({1000/avg_ms:.0f} FPS)")
            else:
                print(f"  {Path(rknn_p).name:40s}  FAILED")
        print("=" * 60)


if __name__ == "__main__":
    main()
