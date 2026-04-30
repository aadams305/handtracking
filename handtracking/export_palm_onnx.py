"""Export trained BlazePalm detector to ONNX.

Usage:
    python -m handtracking.export_palm_onnx --checkpoint checkpoints/palm_det.pt --out models/palm_det.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from handtracking.models.palm_detector import BlazePalmDetector, PALM_INPUT_SIZE


def main() -> None:
    ap = argparse.ArgumentParser(description="Export palm detector to ONNX")
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/palm_det.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/palm_det.onnx"))
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    wm = float(ckpt.get("width_mult", 0.5))

    model = BlazePalmDetector(width_mult=wm).eval()
    model.load_state_dict(ckpt["model"])

    dummy = torch.randn(1, 3, PALM_INPUT_SIZE, PALM_INPUT_SIZE)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["input"],
        output_names=["box_offsets", "score_logits"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=13,
    )

    try:
        import onnx
        from onnxsim import simplify

        m = onnx.load(str(args.out))
        m_sim, ok = simplify(m)
        if ok:
            onnx.save(m_sim, str(args.out))
            print("ONNX simplified OK")
    except ImportError:
        pass

    print(f"Exported palm detector → {args.out}")


if __name__ == "__main__":
    main()
