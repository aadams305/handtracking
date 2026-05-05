"""Export trained YOLO11n palm detector to ONNX and then RKNN.

Usage:
    # Export to ONNX only
    python3 export_yolo_palm.py --weights runs/detect/yolo_palm/weights/best.pt --out models/yolo_palm.onnx

    # Export to both ONNX and RKNN
    python3 export_yolo_palm.py --weights runs/detect/yolo_palm/weights/best.pt --rknn

    # With custom image size
    python3 export_yolo_palm.py --weights runs/detect/yolo_palm/weights/best.pt --imgsz 192 --rknn
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export_to_onnx(weights: Path, out: Path, imgsz: int, opset: int) -> Path:
    """Export YOLO model to ONNX format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed. Install with: pip install ultralytics")

    model = YOLO(str(weights))
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,
        dynamic=False,
    )

    # Ultralytics exports next to the weights file; move to desired output
    exported = weights.with_suffix(".onnx")
    if exported.exists() and exported != out:
        out.parent.mkdir(parents=True, exist_ok=True)
        exported.rename(out)
    elif not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        # Try common export locations
        for candidate in [weights.parent / "best.onnx", weights.parent.parent / "best.onnx"]:
            if candidate.exists():
                shutil.move(str(candidate), str(out))
                break

    if out.exists():
        print(f"ONNX exported: {out}")
    else:
        print(f"WARNING: Expected ONNX at {out} not found. Check ultralytics output.")
    return out


def export_to_rknn(onnx_path: Path, rknn_path: Path, imgsz: int, precision: str) -> None:
    """Convert YOLO ONNX to RKNN."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from export_rknn import convert_onnx_to_rknn, YOLO_MEAN, YOLO_STD

    if not onnx_path.exists():
        raise SystemExit(f"ONNX file not found: {onnx_path}")

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    quantize = precision == "int8"

    convert_onnx_to_rknn(
        str(onnx_path),
        str(rknn_path),
        input_size=imgsz,
        quantize=quantize,
        dataset_txt=None,
        target_platform="rk3588",
        mean_values=YOLO_MEAN,
        std_values=YOLO_STD,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Export YOLO11n palm detector to ONNX/RKNN")
    ap.add_argument("--weights", type=Path, default=Path("runs/detect/yolo_palm/weights/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/yolo_palm.onnx"))
    ap.add_argument("--imgsz", type=int, default=192)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--rknn", action="store_true", help="Also export to RKNN")
    ap.add_argument("--rknn-out", type=Path, default=None)
    ap.add_argument("--precision", choices=("fp16", "int8"), default="fp16")
    args = ap.parse_args()

    onnx_path = export_to_onnx(args.weights, args.out, args.imgsz, args.opset)

    if args.rknn:
        rknn_out = args.rknn_out or args.out.with_suffix(".rknn")
        export_to_rknn(onnx_path, rknn_out, args.imgsz, args.precision)


if __name__ == "__main__":
    main()
