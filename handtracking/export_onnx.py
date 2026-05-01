"""Export RTMPoseHand to ONNX with fixed input [1, 3, 256, 256].

Usage:
    python3 -m handtracking.export_onnx --checkpoint checkpoints/rtmpose_hand.pt --out models/rtmpose_hand.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from handtracking.models.rtmpose_hand import INPUT_SIZE, RTMPoseHand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/rtmpose_hand.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/rtmpose_hand.onnx"))
    ap.add_argument("--opset", type=int, default=14)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = RTMPoseHand()
    model.load_state_dict(sd, strict=False)
    model.eval()

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    traced_model = torch.jit.trace(model, dummy)

    torch.onnx.export(
        traced_model,
        dummy,
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
