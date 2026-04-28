"""
Compare MediaPipe (teacher) vs SimCC student on one image using the same 256×256
letterbox pipeline as training / distill (not a tight hand crop).

Usage (from repo root, with PYTHONPATH=.):

  PYTHONPATH=. python3 -m handtracking.compare_mp_student \\
      --image IMG_7271.jpeg \\
      --checkpoint checkpoints/hand_simcc.pt \\
      --onnx models/hand_simcc.onnx \\
      --out comparison_mp_student.png

Requires: mediapipe, torch, matplotlib, opencv; optional onnxruntime for ONNX panel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from handtracking.dataset import normalize_bgr_tensor
from handtracking.geometry import letterbox_image
from handtracking.models.hand_simcc import HandSimCCNet, INPUT_SIZE, decode_simcc_soft_argmax
from handtracking.simcc_numpy import bgr_letterbox_to_nchw_batch, decode_simcc_soft_argmax_numpy
from handtracking.teacher import MediaPipeTeacher, extract_21_points_pixel
from handtracking.topology import NUM_HAND_JOINTS
from handtracking.viz import EDGES_21


def _resolve_image(path: Path | None) -> Path:
    if path is not None and path.is_file():
        return path
    for cand in (
        Path("IMG_7271.jpeg"),
        Path("img_7271.jpeg"),
        Path("data/demo_images/img_7271.jpeg"),
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "No image found. Pass --image /path/to/photo.jpeg or place it at "
        "IMG_7271.jpeg (repo root), img_7271.jpeg, or data/demo_images/img_7271.jpeg"
    )


def _draw_skeleton(ax, kp_xy: np.ndarray, color: str) -> None:
    for i, j in EDGES_21:
        if i < len(kp_xy) and j < len(kp_xy):
            ax.plot(
                [kp_xy[i, 0], kp_xy[j, 0]],
                [kp_xy[i, 1], kp_xy[j, 1]],
                color=color,
                linewidth=1.5,
                alpha=0.85,
            )
    ax.scatter(kp_xy[:, 0], kp_xy[:, 1], c=color, s=18, zorder=5)


def main() -> None:
    ap = argparse.ArgumentParser(description="MediaPipe vs SimCC (PyTorch / ONNX) on letterboxed input")
    ap.add_argument("--image", type=Path, default=None, help="BGR image path (default: search common names)")
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/hand_simcc.pt"))
    ap.add_argument("--onnx", type=Path, default=Path("models/hand_simcc.onnx"))
    ap.add_argument("--out", type=Path, default=Path("comparison_mp_student.png"))
    ap.add_argument("--width-mult", type=float, default=0.5)
    args = ap.parse_args()

    img_path = _resolve_image(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    lb_bgr, lb = letterbox_image(img, INPUT_SIZE)
    lb_rgb = cv2.cvtColor(lb_bgr, cv2.COLOR_BGR2RGB)

    with MediaPipeTeacher(static_image_mode=True, max_num_hands=1) as teacher:
        tr = teacher.process_bgr(img)
    if not tr.ok or tr.landmarks_norm is None:
        print("MediaPipe did not detect a hand in this image.", file=sys.stderr)
        sys.exit(1)

    kp_src = extract_21_points_pixel(tr.landmarks_norm[:, :2], w, h)
    kp_mp_lb = np.zeros((NUM_HAND_JOINTS, 2), dtype=np.float32)
    for i in range(NUM_HAND_JOINTS):
        kp_mp_lb[i, 0], kp_mp_lb[i, 1] = lb.map_xy_src_to_dst(
            float(kp_src[i, 0]), float(kp_src[i, 1])
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    wm = float(ckpt.get("width_mult", args.width_mult))
    if ckpt.get("qat"):
        raise SystemExit("QAT checkpoint cannot be compared here; use FP32 hand_simcc.pt")
    model = HandSimCCNet(width_mult=wm).eval()
    model.load_state_dict(ckpt["model"], strict=True)

    inp = normalize_bgr_tensor(lb_bgr).unsqueeze(0)
    with torch.inference_mode():
        lx, ly = model(inp)
    pred_pt = decode_simcc_soft_argmax(lx, ly)[0].cpu().numpy()

    onnx_pred: np.ndarray | None = None
    onnx_err: str | None = None
    if args.onnx.is_file():
        try:
            import onnxruntime as ort
        except ImportError:
            onnx_err = "onnxruntime not installed"
        else:
            try:
                sess = ort.InferenceSession(
                    str(args.onnx.resolve()), providers=["CPUExecutionProvider"]
                )
                in_name = sess.get_inputs()[0].name
                ort_in = bgr_letterbox_to_nchw_batch(lb_bgr)
                ox, oy = sess.run(None, {in_name: ort_in})
                onnx_pred = decode_simcc_soft_argmax_numpy(ox, oy)
                d = np.abs(onnx_pred - pred_pt).max()
                onnx_err = f"max|ONNX−PyTorch|={d:.4f}px"
            except Exception as e:
                onnx_pred = None
                onnx_err = f"ONNX failed: {e}"

    ncols = 3 if onnx_pred is not None else 2
    fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 2:
        axs = [axs[0], axs[1]]

    for ax in axs:
        ax.imshow(lb_rgb)
        ax.axis("off")

    axs[0].set_title("MediaPipe (letterbox space, training-style)")
    _draw_skeleton(axs[0], kp_mp_lb, "tomato")

    axs[1].set_title("SimCC student (PyTorch)")
    _draw_skeleton(axs[1], pred_pt, "lime")

    if onnx_pred is not None:
        axs[2].set_title(f"SimCC student (ONNX)\n{onnx_err}")
        _draw_skeleton(axs[2], onnx_pred, "cyan")

    fig.suptitle(f"{img_path.name}  ({INPUT_SIZE}×{INPUT_SIZE} letterbox)", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
