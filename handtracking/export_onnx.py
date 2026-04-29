"""Export HandSimCCNet to ONNX with fixed input [1, 3, 256, 256]."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from handtracking.models.hand_simcc import HandSimCCNet, INPUT_SIZE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/hand_simcc.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/hand_simcc.onnx"))
    ap.add_argument("--opset", type=int, default=14)
    args = ap.parse_args()

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    wm = float(ckpt.get("width_mult", 0.5))
    if ckpt.get("qat"):
        raise SystemExit(
            "Checkpoint is QAT/INT8; export from .fp32.pt (saved before convert) instead."
        )
    model = HandSimCCNet(width_mult=wm)
    # strict=False: old checkpoints lack aux_head (presence/handedness) weights
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Pre-trace to force the legacy TorchScript exporter (bypasses broken Dynamo ONNX engine)
    traced_model = torch.jit.trace(model, dummy)

    torch.onnx.export(
        traced_model,
        dummy,
        str(args.out),
        input_names=["input"],
        output_names=["simcc_x", "simcc_y", "presence", "handedness"],
        opset_version=args.opset,
        dynamic_axes=None,
    )
    print(f"Wrote {args.out} with inputs [1,3,{INPUT_SIZE},{INPUT_SIZE}]")


if __name__ == "__main__":
    main()
