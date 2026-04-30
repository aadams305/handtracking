"""
Live USB camera: SimCC hand overlay only (ONNX *or* PyTorch — no MediaPipe teacher).

Camera I/O matches ``cameraLogi.py`` (V4L2 + MJPG, resolution/FPS set + read-back print).

Speed-oriented defaults: ``--infer-every 8``, ``--preview-max-width 960``, CUDA ORT if
available, vectorized keypoint mapping, ``LINE_8`` skeleton, cached decode constants.

Run from repo root (so ``checkpoints/`` and ``models/`` resolve):

  PYTHONPATH=. python3 camera_student.py --backend onnx
  PYTHONPATH=. python3 camera_student.py --backend pytorch --device cuda

Press ``q`` in the window to quit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Repo-root script: ensure package import when run as ``python3 camera_student.py``
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from handtracking.dataset import normalize_bgr_tensor
from handtracking.geometry import letterbox_image, map_keypoints_lb_to_src
from handtracking.models.hand_simcc import INPUT_SIZE
from handtracking.simcc_numpy import (
    bgr_letterbox_to_nchw_batch,
    decode_simcc_soft_argmax_numpy,
    keypoints_collapsed,
    simcc_confidence_numpy,
)
from handtracking.viz import draw_hand_21


def _fourcc_str(code: float) -> str:
    i = int(code)
    return "".join(chr((i >> (8 * j)) & 0xFF) for j in range(4))


def open_capture_logi_style(camera: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        cap.open(camera)
    return cap


def print_camera_config(cap: cv2.VideoCapture, width: int, height: int, fps: int) -> None:
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    af = cap.get(cv2.CAP_PROP_FPS)
    fourcc_str = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    print("--- Camera configuration ---")
    print(f"Requested: {width}x{height} @ {fps} FPS (MJPG)")
    print(f"Actual:    {aw}x{ah} @ {af} FPS ({fourcc_str})")
    print("-----------------------------")


def draw_banner(frame: np.ndarray, lines: list[str]) -> np.ndarray:
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


def _preview_resize(vis: np.ndarray, max_w: int) -> np.ndarray:
    if max_w <= 0 or vis.shape[1] <= max_w:
        return vis
    pw = max_w
    ph = int(round(vis.shape[0] * (pw / float(vis.shape[1]))))
    return cv2.resize(vis, (pw, ph), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser(description="USB camera + SimCC student (ONNX or PyTorch only)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--fps", type=int, default=30, help="Match cameraLogi.py default (30)")
    ap.add_argument(
        "--backend",
        choices=("onnx", "pytorch"),
        default="onnx",
        help="ONNX Runtime or PyTorch only (no MediaPipe; no fallback if chosen backend is missing)",
    )
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/hand_simcc.pt"))
    ap.add_argument("--onnx", type=Path, default=Path("models/hand_simcc.onnx"))
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Enable GPU acceleration: CUDA EP for ONNX, cuda device for PyTorch. "
             "On Orange Pi 5 (RK3588) there is no CUDA GPU — use RKNN for NPU instead.",
    )
    ap.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="Run model every N frames (1 = every frame for max speed; higher = skip frames)",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="0 = use up to 8 ORT intra-op threads / torch threads",
    )
    ap.add_argument(
        "--preview-max-width",
        type=int,
        default=960,
        help="Resize preview for imshow (0 = full resolution; 960 saves a lot of GUI work)",
    )
    ap.add_argument(
        "--conf-threshold",
        type=float,
        default=0.02,
        help="Minimum SimCC confidence to draw overlay (0.0 = always draw; trained model gives ~0.04 for real hands)",
    )
    args = ap.parse_args()

    os.chdir(_REPO)
    cv2.setUseOptimized(True)

    ort_session = None
    ort_in_name = "input"
    net = None
    dev = None
    decode_torch = None

    if args.backend == "onnx":
        if not args.onnx.is_file():
            raise SystemExit(
                f"ONNX not found: {args.onnx.resolve()}\n"
                "Export with: PYTHONPATH=. python3 -m handtracking.export_onnx "
                f"--checkpoint {args.checkpoint} --out {args.onnx}"
            )
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        nthr = args.threads if args.threads > 0 else min(8, max(1, (os.cpu_count() or 4)))
        so.intra_op_num_threads = nthr
        so.inter_op_num_threads = 1
        avail = ort.get_available_providers()
        prov: list[str] = []
        if args.gpu and "CUDAExecutionProvider" in avail:
            prov.append("CUDAExecutionProvider")
            print("GPU: CUDA ExecutionProvider enabled for ONNX Runtime.")
        elif args.gpu:
            print(
                "GPU: --gpu requested but CUDAExecutionProvider not available.\n"
                "  Available providers: " + str(avail) + "\n"
                "  Falling back to CPU. On RK3588, convert to RKNN for NPU acceleration."
            )
        prov.append("CPUExecutionProvider")
        ort_session = ort.InferenceSession(str(args.onnx.resolve()), so, providers=prov)
        ort_in_name = ort_session.get_inputs()[0].name
        print(
            f"Backend: ONNX  ({args.onnx})  input={ort_in_name}  "
            f"providers={ort_session.get_providers()}  intra_threads={nthr}"
        )
    else:
        import torch
        from handtracking.models.hand_simcc import HandSimCCNet, decode_simcc_soft_argmax

        if not args.checkpoint.is_file():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint.resolve()}")
        decode_torch = decode_simcc_soft_argmax
        use_cuda = (args.gpu or args.device == "cuda") and torch.cuda.is_available()
        dev = torch.device("cuda" if use_cuda else "cpu")
        if args.gpu and not use_cuda:
            print("GPU: --gpu requested but CUDA not available. Using CPU.")
        nthr = args.threads if args.threads > 0 else min(8, max(1, (os.cpu_count() or 4)))
        torch.set_num_threads(nthr)
        cv2.setNumThreads(1)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location="cpu")
        wm = float(ckpt.get("width_mult", 0.5))
        if ckpt.get("qat"):
            raise SystemExit("QAT checkpoint not supported; use FP32 hand_simcc.pt")
        net = HandSimCCNet(width_mult=wm).eval()
        net.load_state_dict(ckpt["model"], strict=False)
        net = net.to(dev)
        dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=dev)
        with torch.inference_mode():
            for _ in range(2):
                net(dummy)
        print(f"Backend: PyTorch  device={dev}  torch_threads={nthr}")

    cap = open_capture_logi_style(args.camera, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera index {args.camera}.")

    print_camera_config(cap, args.width, args.height, args.fps)
    print(
        f"Speed hints: infer_every={args.infer_every}, preview_max_width={args.preview_max_width} "
        "(set preview 0 for full-res window)"
    )
    print("Press 'q' in the video window to quit.")

    win = f"SimCC student ({args.backend})  q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_kp: np.ndarray | None = None
    fcount = 0
    ie = max(1, args.infer_every)
    collapsed_warn = False
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

    last_conf: float = 0.0
    last_hand_str: str = ""
    conf_threshold = args.conf_threshold

    def infer_frame(lb_img, lb) -> None:
        nonlocal last_kp, last_conf, last_hand_str
        if ort_session is not None:
            inp = bgr_letterbox_to_nchw_batch(lb_img)
            out = ort_session.run(None, {ort_in_name: inp})
            lx, ly = out[0], out[1]
            xy = decode_simcc_soft_argmax_numpy(lx, ly)
            # Always use distribution peakedness for confidence —
            # the presence head was not trained with negative samples,
            # so it always outputs ~1.0.  simcc_confidence actually
            # reflects how "peaked" the distributions are.
            last_conf = simcc_confidence_numpy(lx, ly)
            # Use handedness head if available (it IS trained properly)
            if len(out) >= 4:
                hand_sigmoid = 1.0 / (1.0 + np.exp(-float(out[3].flat[0])))
                last_hand_str = "Right" if hand_sigmoid > 0.5 else "Left"
            else:
                last_hand_str = ""
        else:
            import torch
            from handtracking.models.hand_simcc import simcc_confidence
            assert net is not None and decode_torch is not None and dev is not None
            inp = normalize_bgr_tensor(lb_img).unsqueeze(0).to(dev)
            with torch.inference_mode():
                out = net(inp)
                if len(out) == 4:
                    lx, ly, _pres_logit, hand_logit = out
                    hand_p = float(torch.sigmoid(hand_logit).item())
                    last_hand_str = "Right" if hand_p > 0.5 else "Left"
                else:
                    lx, ly = out
                    last_hand_str = ""
                # Always use simcc peakedness for confidence
                last_conf = float(simcc_confidence(lx, ly).mean().item())
                xy = decode_torch(lx, ly)[0].cpu().numpy()
        last_kp = map_keypoints_lb_to_src(lb, xy.astype(np.float32, copy=False))

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break
        fcount += 1
        if last_kp is None or (fcount % ie == 0):
            lb_img, lb = letterbox_image(frame, INPUT_SIZE)
            infer_frame(lb_img, lb)

        vis = frame
        if last_kp is not None and last_conf >= conf_threshold:
            vis = draw_hand_21(frame, last_kp, radius=4, line_type=cv2.LINE_8)
            
            info_lines = [f"conf={last_conf:.2f}"]
            if last_hand_str:
                info_lines.append(last_hand_str)
            info_text = "  ".join(info_lines)
            cv2.putText(
                vis, info_text, (10, vis.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA,
            )

            if keypoints_collapsed(last_kp, frame.shape):
                if not collapsed_warn:
                    collapsed_warn = True
                    print("[student] Predictions collapsed to a blob — check weights / training.")
                vis = draw_banner(
                    vis,
                    [
                        "Collapsed skeleton — train/export a stronger model.",
                        "Try --infer-every 10+ and --preview-max-width 960 for speed.",
                    ],
                )

        vis = _preview_resize(vis, args.preview_max_width)
        tick_fps()
        cv2.imshow(win, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
