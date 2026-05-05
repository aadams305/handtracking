"""Export RTMPoseHand to ONNX with fixed input [1, 3, 256, 256].

Prefers EMA weights from checkpoint by default (better inference quality).

Usage:
    python3 -m handtracking.export_onnx --checkpoint checkpoints/rtmpose_hand.pt --out models/rtmpose_hand.onnx
    python3 -m handtracking.export_onnx --checkpoint checkpoints/rtmpose_hand_latest.pt --use-ema
    python3 -m handtracking.export_onnx --checkpoint checkpoints/rtmpose_hand_latest.pt --no-ema
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from handtracking.models.rtmpose_hand import INPUT_SIZE, RTMPoseHand


def _load_weights(ckpt_path: Path, use_ema: bool) -> dict:
    """Load state dict from checkpoint, preferring EMA weights when available."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        return ckpt

    # Try EMA weights first (from training checkpoints)
    if use_ema and "ema" in ckpt:
        ema_data = ckpt["ema"]
        if isinstance(ema_data, dict) and "module" in ema_data:
            print("Using EMA weights from checkpoint", flush=True)
            return ema_data["module"]

    # Fall back to model weights
    if "model" in ckpt:
        if use_ema:
            print("EMA weights not found, using model weights", flush=True)
        return ckpt["model"]

    return ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/rtmpose_hand.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/rtmpose_hand.onnx"))
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--use-ema", dest="use_ema", action="store_true", default=True,
                    help="Prefer EMA weights from checkpoint (default)")
    ap.add_argument("--no-ema", dest="use_ema", action="store_false",
                    help="Use raw model weights instead of EMA")
    args = ap.parse_args()

    sd = _load_weights(args.checkpoint, args.use_ema)
    model = RTMPoseHand()
    model.load_state_dict(sd, strict=False)
    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        str(args.out),
        input_names=["input"],
        output_names=["simcc_x", "simcc_y"],
        opset_version=args.opset,
        dynamic_axes=None,
    )
    print(f"Wrote {args.out}  input=[1,3,{INPUT_SIZE},{INPUT_SIZE}]  "
          f"outputs: simcc_x=[1,21,512], simcc_y=[1,21,512]")

    try:
        import onnx
        from onnxsim import simplify
        m = onnx.load(str(args.out))
        m_sim, ok = simplify(m)
        if ok:
            onnx.save(m_sim, str(args.out))
            print("ONNX simplified successfully")
        else:
            print("WARNING: onnxsim simplify failed, keeping original")
    except ImportError:
        print("(onnxsim not installed, skipping simplification)")


if __name__ == "__main__":
    main()
