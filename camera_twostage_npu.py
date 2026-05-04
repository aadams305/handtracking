"""Two-stage live camera: MediaPipe palm detection (CPU) → RTMPose-M landmarks (NPU).

Uses MediaPipe Hands as the palm detector to provide a bounding box, then crops
the hand region and runs RTMPose-M on the RK3588 NPU for fast landmark regression.

This is a hybrid approach: once a dedicated palm detector RKNN is trained, swap
the detection stage to run fully on NPU.

Usage:
    python3 camera_twostage_npu.py
    python3 camera_twostage_npu.py --landmark-rknn models/rtmpose_hand.rknn --camera 0
    python3 camera_twostage_npu.py --uint8-input  # for INT8 quantized model

Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from handtracking.models.rtmpose_hand import INPUT_SIZE, MEAN as RTMPOSE_MEAN, STD as RTMPOSE_STD, SIMCC_SPLIT_RATIO
from handtracking.simcc_numpy import (
    decode_simcc_soft_argmax_numpy,
    simcc_confidence_numpy,
)
from handtracking.viz import draw_hand_21

PIXEL_MEAN = np.array(RTMPOSE_MEAN, dtype=np.float32)
PIXEL_STD = np.array(RTMPOSE_STD, dtype=np.float32)


class RKNNInference:
    """Wrapper for rknn-toolkit-lite2 inference."""

    def __init__(self, rknn_path: str, core_mask: int = 1) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            try:
                from rknn.api import RKNN as RKNNLite
            except ImportError:
                raise ImportError(
                    "Neither rknn-toolkit-lite2 nor rknn-toolkit2 found."
                )

        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model (ret={ret})")

        core_map = {1: 0x01, 2: 0x02, 3: 0x04, 7: 0x07}
        rknn_core = core_map.get(core_mask, 0x01)

        try:
            ret = self.rknn.init_runtime(core_mask=rknn_core)
        except TypeError:
            ret = self.rknn.init_runtime()

        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime (ret={ret})")

    def run(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return self.rknn.inference(inputs=inputs)

    def release(self) -> None:
        self.rknn.release()


def crop_hand_square(
    image_bgr: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    expand: float = 1.3,
    output_size: int = 256,
) -> tuple[np.ndarray, dict]:
    """Crop a square region around the hand bbox, resize to output_size.

    Returns (cropped_rgb, transform_info).
    """
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = bbox_xywh
    cx, cy = x + bw / 2, y + bh / 2
    side = max(bw, bh) * expand

    x1 = int(cx - side / 2)
    y1 = int(cy - side / 2)
    x2 = int(cx + side / 2)
    y2 = int(cy + side / 2)

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)

    crop = image_bgr[y1c:y2c, x1c:x2c]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    transform = {
        "x1": x1, "y1": y1,
        "crop_size": x2 - x1,
        "output_size": output_size,
    }
    return resized, transform


def map_landmarks_back(kp: np.ndarray, transform: dict) -> np.ndarray:
    """Map landmarks from crop space to source image coordinates."""
    out = kp.copy().astype(np.float32)
    scale = transform["crop_size"] / transform["output_size"]
    out[:, 0] = out[:, 0] * scale + transform["x1"]
    out[:, 1] = out[:, 1] * scale + transform["y1"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-stage: MediaPipe palm (CPU) + RTMPose (NPU)")
    ap.add_argument("--landmark-rknn", type=Path, default=Path("models/rtmpose_hand.rknn"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--core-mask", type=int, default=7,
                    help="NPU core mask: 1=core0, 2=core1, 3=core2, 7=all three")
    ap.add_argument("--conf-threshold", type=float, default=0.02)
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--preview-max-width", type=int, default=960)
    ap.add_argument("--uint8-input", action="store_true",
                    help="Pass uint8 RGB (for INT8 quantized models with built-in normalization)")
    ap.add_argument("--no-display", action="store_true",
                    help="Run headless (no cv2 window, just print FPS)")
    ap.add_argument("--det-scale", type=float, default=0.25,
                    help="Downscale factor for detection (0.25 = 1/4 resolution)")
    ap.add_argument("--det-interval", type=int, default=5,
                    help="Run detection every N frames (reuse bbox between)")
    args = ap.parse_args()

    os.chdir(_REPO)

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3,
    )

    import threading

    print(f"Loading RKNN landmark model: {args.landmark_rknn}", flush=True)
    landmark_npu = RKNNInference(str(args.landmark_rknn), core_mask=args.core_mask)
    print("NPU landmark model loaded.", flush=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    af = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera: {aw}x{ah} @ {af} FPS", flush=True)
    print("Two-stage: MediaPipe palm detection (CPU, async) → RTMPose-M landmarks (NPU)", flush=True)
    print("Press 'q' to quit.", flush=True)

    # --- Async detection thread ---
    det_lock = threading.Lock()
    det_bboxes: list[tuple[int, int, int, int]] = []
    det_busy = False
    det_ms = 0.0

    def _run_detection(frame_bgr: np.ndarray, full_w: int, full_h: int) -> None:
        nonlocal det_bboxes, det_busy, det_ms
        det_h = int(full_h * args.det_scale)
        det_w = int(full_w * args.det_scale)
        small = cv2.resize(frame_bgr, (det_w, det_h), interpolation=cv2.INTER_AREA)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        t = time.perf_counter()
        results = hands.process(rgb_small)
        elapsed = (time.perf_counter() - t) * 1000

        new_bboxes: list[tuple[int, int, int, int]] = []
        if results.multi_hand_landmarks:
            for hlm in results.multi_hand_landmarks:
                xs = [lm.x * full_w for lm in hlm.landmark]
                ys = [lm.y * full_h for lm in hlm.landmark]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                new_bboxes.append((x1, y1, x2 - x1, y2 - y1))

        with det_lock:
            if new_bboxes:
                det_bboxes = new_bboxes
            det_ms = elapsed
            det_busy = False

    # --- Main loop ---
    win = "Two-Stage NPU (q=quit)"
    fps_t0 = time.perf_counter()
    fps_n = 0
    infer_times: list[float] = []
    lm_times: list[float] = []
    last_det_ms = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # Launch async detection if not already running
        with det_lock:
            should_launch = not det_busy
            if should_launch:
                det_busy = True
                last_det_ms = det_ms
        if should_launch:
            threading.Thread(
                target=_run_detection, args=(frame.copy(), w, h), daemon=True
            ).start()

        # Read current bboxes (non-blocking)
        with det_lock:
            current_bboxes = list(det_bboxes)

        vis = frame.copy()
        num_hands = 0
        updated_bboxes: list[tuple[int, int, int, int]] = []

        for bbox_xywh in current_bboxes:
            crop_bgr, transform = crop_hand_square(frame, bbox_xywh, expand=1.3, output_size=INPUT_SIZE)

            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            inp = np.expand_dims(rgb, axis=0)

            t_lm = time.perf_counter()
            lm_out = landmark_npu.run([inp])
            lm_dt = time.perf_counter() - t_lm
            lm_times.append(lm_dt * 1000)

            lx, ly = lm_out[0], lm_out[1]
            conf = simcc_confidence_numpy(lx, ly)

            xy = decode_simcc_soft_argmax_numpy(lx, ly, split_ratio=SIMCC_SPLIT_RATIO)
            kp_crop = xy[0] if xy.ndim == 3 else xy
            kp_src = map_landmarks_back(kp_crop, transform)
            vis = draw_hand_21(vis, kp_src, radius=4, line_type=cv2.LINE_8)
            num_hands += 1

            lm_xs = kp_src[:, 0]
            lm_ys = kp_src[:, 1]
            nx1, ny1 = int(lm_xs.min()), int(lm_ys.min())
            nx2, ny2 = int(lm_xs.max()), int(lm_ys.max())
            updated_bboxes.append((nx1, ny1, nx2 - nx1, ny2 - ny1))

            cv2.putText(vis, f"conf={conf:.3f}", (nx1, ny1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Update cached bboxes from landmarks for tracking
        if updated_bboxes:
            with det_lock:
                det_bboxes = updated_bboxes

        total_dt = time.perf_counter() - t0
        infer_times.append(total_dt * 1000)

        fps_n += 1
        fps_dt = time.perf_counter() - fps_t0
        if fps_dt >= 1.0:
            avg_ms = sum(infer_times[-fps_n:]) / max(1, fps_n)
            avg_lm = sum(lm_times[-fps_n:]) / max(1, fps_n) if lm_times else 0
            fps_str = f"{fps_n / fps_dt:.0f} FPS  total={avg_ms:.1f}ms  det={last_det_ms:.0f}ms(async)  lm={avg_lm:.1f}ms"
            if not args.no_display:
                try:
                    cv2.setWindowTitle(win, f"{win}  ~{fps_str}")
                except cv2.error:
                    pass
            print(f"\r{fps_str}  hands={num_hands}", end="", flush=True)
            fps_t0 = time.perf_counter()
            fps_n = 0

        if not args.no_display:
            if args.preview_max_width > 0 and vis.shape[1] > args.preview_max_width:
                pw = args.preview_max_width
                ph = int(round(vis.shape[0] * (pw / float(vis.shape[1]))))
                vis = cv2.resize(vis, (pw, ph), interpolation=cv2.INTER_AREA)
            cv2.imshow(win, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    print()
    if infer_times:
        avg = sum(infer_times) / len(infer_times)
        p50 = sorted(infer_times)[len(infer_times) // 2]
        p99 = sorted(infer_times)[int(len(infer_times) * 0.99)]
        print(f"Total latency ({len(infer_times)} frames): avg={avg:.1f}ms p50={p50:.1f}ms p99={p99:.1f}ms")
    if lm_times:
        avg_lm = sum(lm_times) / len(lm_times)
        print(f"NPU landmark only ({len(lm_times)} calls): avg={avg_lm:.2f}ms")

    cap.release()
    landmark_npu.release()
    hands.close()
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
