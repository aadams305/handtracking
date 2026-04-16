"""
Live USB camera: V4L2 + MJPG + hand overlay (teacher=MediaPipe, student=SimCC).

Student mode
  - **Green blob**: toy/demo weights collapse to the letterbox center — train on real
    data; on-screen banner when this is detected.
  - **FPS**: use `--backend onnx` + `onnxruntime` and `--infer-every 8` for smoother video.

Usage
  PYTHONPATH=. python3 -m handtracking.live_camera --source teacher
  PYTHONPATH=. python3 -m handtracking.export_onnx --checkpoint checkpoints/hand_simcc.pt
  pip install onnxruntime
  PYTHONPATH=. python3 -m handtracking.live_camera --source student --backend onnx --infer-every 8

Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from handtracking.dataset import normalize_bgr_tensor
from handtracking.geometry import letterbox_image
from handtracking.simcc_numpy import (
    bgr160_to_nchw_batch,
    decode_simcc_soft_argmax_numpy,
    keypoints_collapsed,
)
from handtracking.teacher import MediaPipeTeacher, extract_10_points_pixel
from handtracking.viz import draw_hand_10


def keypoints_160_to_frame(kp_160: np.ndarray, lb) -> np.ndarray:
    """(10,2) in 160 letterbox space -> full-frame pixels."""
    out = np.empty_like(kp_160)
    for i in range(kp_160.shape[0]):
        out[i, 0], out[i, 1] = lb.map_xy_dst_to_src(
            float(kp_160[i, 0]), float(kp_160[i, 1])
        )
    return out


def open_capture(camera: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        cap.open(camera)
    return cap


def draw_student_banner(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]
    pad = 8
    box_h = min(h, 22 * len(lines) + 2 * pad)
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, box_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
    for i, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (pad, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
    return vis


def main() -> None:
    ap = argparse.ArgumentParser(description="Live hand overlay on USB camera")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--source", choices=("teacher", "student"), default="teacher")
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/hand_simcc.pt"))
    ap.add_argument("--onnx", type=Path, default=Path("models/hand_simcc.onnx"))
    ap.add_argument(
        "--backend",
        choices=("pytorch", "onnx"),
        default="onnx",
        help="Student: onnxruntime (faster CPU) or PyTorch",
    )
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--infer-every", type=int, default=6, metavar="N")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    ort_session = None
    ort_in_name = "input"
    net = None
    dev = None
    decode_torch = None

    if args.source == "student":
        want_onnx = args.backend == "onnx"
        if want_onnx and args.onnx.is_file():
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            nthr = args.threads if args.threads > 0 else min(4, max(1, (os.cpu_count() or 4)))
            so.intra_op_num_threads = nthr
            so.inter_op_num_threads = 1
            ort_session = ort.InferenceSession(
                str(args.onnx.resolve()),
                so,
                providers=["CPUExecutionProvider"],
            )
            ort_in_name = ort_session.get_inputs()[0].name
            print(f"Student: ONNX Runtime <- {args.onnx} (input={ort_in_name})")
        elif want_onnx:
            print(
                f"No ONNX at {args.onnx}. Export with:\n"
                f"  PYTHONPATH=. python3 -m handtracking.export_onnx "
                f"--checkpoint {args.checkpoint} --out {args.onnx}\n"
                "Loading PyTorch instead."
            )

        if ort_session is None:
            import torch

            from handtracking.models.hand_simcc import HandSimCCNet, decode_simcc_soft_argmax

            decode_torch = decode_simcc_soft_argmax
            dev = torch.device(
                args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
            )
            if not args.checkpoint.is_file():
                raise SystemExit(f"Missing checkpoint {args.checkpoint}")
            nthr = args.threads if args.threads > 0 else min(4, max(1, (os.cpu_count() or 4)))
            torch.set_num_threads(nthr)
            cv2.setNumThreads(1)
            try:
                ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(args.checkpoint, map_location="cpu")
            wm = float(ckpt.get("width_mult", 0.5))
            if ckpt.get("qat"):
                raise SystemExit("Use FP32 checkpoint for live demo.")
            net = HandSimCCNet(width_mult=wm).eval()
            net.load_state_dict(ckpt["model"], strict=True)
            net = net.to(dev)
            dummy = torch.randn(1, 3, 160, 160, device=dev)
            with torch.inference_mode():
                for _ in range(2):
                    net(dummy)
            print(f"Student: PyTorch on {dev}")

    cap = open_capture(args.camera, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}.")

    win = f"Hand [{args.source}] (q quit)"
    print(win)
    if args.source == "student":
        print(
            "Student overlay needs a trained checkpoint. Demo/tiny training → blob in center; "
            "use --source teacher for reliable landmarks."
        )

    fps_t0 = time.perf_counter()
    fps_n = 0

    def tick_fps() -> None:
        nonlocal fps_t0, fps_n
        fps_n += 1
        dt = time.perf_counter() - fps_t0
        if dt >= 1.0:
            try:
                cv2.setWindowTitle(win, f"{win}  ~{fps_n / dt:.0f} FPS")
            except cv2.error:
                pass
            fps_t0 = time.perf_counter()
            fps_n = 0

    if args.source == "teacher":
        with MediaPipeTeacher(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
        ) as teacher:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                tr = teacher.process_bgr(frame)
                if tr.ok and tr.landmarks_norm is not None:
                    kp = extract_10_points_pixel(tr.landmarks_norm, w, h)
                    vis = draw_hand_10(frame, kp, radius=4)
                else:
                    vis = frame
                tick_fps()
                cv2.imshow(win, vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    else:
        last_kp: np.ndarray | None = None
        fcount = 0
        ie = max(1, args.infer_every)
        collapsed_warn = False
        if ort_session is None:
            import torch

        def infer_frame(lb_img, lb) -> None:
            nonlocal last_kp
            if ort_session is not None:
                inp = bgr160_to_nchw_batch(lb_img)
                out = ort_session.run(None, {ort_in_name: inp})
                lx, ly = out[0], out[1]
                xy = decode_simcc_soft_argmax_numpy(lx, ly)
            else:
                assert net is not None and decode_torch is not None and dev is not None
                inp = normalize_bgr_tensor(lb_img).unsqueeze(0).to(dev)
                with torch.inference_mode():
                    lx, ly = net(inp)
                    xy = decode_torch(lx, ly)[0].cpu().numpy()
            last_kp = keypoints_160_to_frame(xy, lb)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fcount += 1
            run = last_kp is None or (fcount % ie == 0)
            if run:
                lb_img, lb = letterbox_image(frame, 160)
                infer_frame(lb_img, lb)

            if last_kp is not None:
                vis = draw_hand_10(frame, last_kp, radius=4)
                if keypoints_collapsed(last_kp, frame.shape):
                    if not collapsed_warn:
                        collapsed_warn = True
                        print(
                            "[student] Collapsed predictions — train on real hand data, "
                            "or use --source teacher."
                        )
                    vis = draw_student_banner(
                        vis,
                        [
                            "Student weights untrained / toy run — skeleton collapsed.",
                            "Train on FreiHAND+, export ONNX, retry.",
                            "Use: --source teacher  for working landmarks.",
                        ],
                    )
            else:
                vis = frame

            tick_fps()
            cv2.imshow(win, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
