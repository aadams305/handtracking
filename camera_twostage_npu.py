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
from handtracking.one_euro_filter import OneEuroFilter2D
from handtracking.bbox_kalman import BboxKalman

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
    dst_bgr: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Crop a square region around the hand bbox, resize to output_size.

    If ``dst_bgr`` is provided (shape ``(output_size, output_size, 3)``, uint8),
    the final resize writes there and the returned array is the same buffer.
    Otherwise a new array is allocated for the resized crop.

    Returns (cropped_bgr, transform_info).
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

    if dst_bgr is not None:
        if dst_bgr.shape != (output_size, output_size, 3):
            raise ValueError(f"dst_bgr must be ({output_size}, {output_size}, 3), got {dst_bgr.shape}")
        cv2.resize(crop, (output_size, output_size), dst=dst_bgr, interpolation=cv2.INTER_LINEAR)
        out = dst_bgr
    else:
        out = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    transform = {
        "x1": x1, "y1": y1,
        "crop_size": x2 - x1,
        "output_size": output_size,
    }
    return out, transform


def map_landmarks_back(kp: np.ndarray, transform: dict) -> np.ndarray:
    """Map landmarks from crop space to source image coordinates."""
    out = kp.copy().astype(np.float32)
    scale = transform["crop_size"] / transform["output_size"]
    out[:, 0] = out[:, 0] * scale + transform["x1"]
    out[:, 1] = out[:, 1] * scale + transform["y1"]
    return out


def decode_yolo_detections(
    output: np.ndarray,
    img_w: int,
    img_h: int,
    det_size: int,
    conf_thresh: float = 0.4,
    iou_thresh: float = 0.45,
    max_det: int = 4,
) -> list[tuple[int, int, int, int]]:
    """Decode YOLO11n detection output to bounding boxes.

    YOLO11n-detect output shape: (1, 5, num_anchors) where row 0-3 = cx,cy,w,h
    and row 4 = confidence (single class).
    """
    if output.ndim == 3:
        output = output[0]  # Remove batch dim -> (5, N)

    if output.shape[0] == 5:
        # (5, N) -> transpose to (N, 5)
        output = output.T

    # output is (N, 5): [cx, cy, w, h, conf]
    confs = output[:, 4]
    mask = confs >= conf_thresh
    boxes = output[mask]

    if len(boxes) == 0:
        return []

    # Scale from det_size to original image
    scale_x = img_w / det_size
    scale_y = img_h / det_size

    # Convert from cx,cy,w,h to x1,y1,x2,y2 for NMS
    cx = boxes[:, 0] * scale_x
    cy = boxes[:, 1] * scale_y
    w = boxes[:, 2] * scale_x
    h = boxes[:, 3] * scale_y
    scores = boxes[:, 4]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Simple greedy NMS
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area_j = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        iou = inter / (area_i + area_j - inter + 1e-6)

        remaining = np.where(iou < iou_thresh)[0]
        order = order[remaining + 1]

    bboxes = []
    for idx in keep:
        bx = int(x1[idx])
        by = int(y1[idx])
        bw = int(x2[idx] - x1[idx])
        bh = int(y2[idx] - y1[idx])
        bboxes.append((bx, by, bw, bh))

    return bboxes


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
    ap.add_argument("--det-size", type=int, default=192,
                    help="Target detection width in pixels (MediaPipe internal is 192)")
    # Filters
    ap.add_argument("--smooth-min-cutoff", type=float, default=1.0,
                    help="One-Euro filter min_cutoff (lower=less jitter)")
    ap.add_argument("--smooth-beta", type=float, default=0.5,
                    help="One-Euro filter beta (higher=less lag on fast motion)")
    ap.add_argument("--no-smooth", action="store_true", help="Disable One-Euro smoothing")
    ap.add_argument("--smooth-d-cutoff", type=float, default=1.0,
                    help="One-Euro derivative cutoff (Hz)")
    ap.add_argument("--no-kalman", action="store_true", help="Disable Kalman bbox prediction")
    ap.add_argument(
        "--hand-lost-frames",
        type=int,
        default=5,
        help="Reset One-Euro/Kalman for a hand slot after this many frames without "
        "that slot active; use 0 to reset immediately when the slot is unused",
    )
    # Detector backend
    ap.add_argument("--detector", choices=("mediapipe", "yolo"), default="mediapipe",
                    help="Detection backend: mediapipe (CPU) or yolo (NPU)")
    ap.add_argument("--yolo-rknn", type=Path, default=Path("models/yolo_palm.rknn"),
                    help="YOLO palm detector RKNN model path")
    ap.add_argument("--yolo-conf", type=float, default=0.4,
                    help="YOLO detection confidence threshold")
    ap.add_argument("--yolo-iou", type=float, default=0.45,
                    help="YOLO NMS IoU threshold")
    args = ap.parse_args()

    os.chdir(_REPO)

    import threading

    hands = None
    yolo_npu = None
    YOLO_DET_SIZE = args.det_size

    if args.detector == "mediapipe":
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=args.max_hands,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
        )
        print("Detector: MediaPipe (CPU)", flush=True)
    else:
        if not args.yolo_rknn.exists():
            raise SystemExit(f"YOLO RKNN model not found: {args.yolo_rknn}")
        yolo_npu = RKNNInference(str(args.yolo_rknn), core_mask=args.core_mask)
        print(f"Detector: YOLO11n (NPU) - {args.yolo_rknn}", flush=True)

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
    # Compute detection resolution (target ~192px wide)
    det_scale = min(1.0, args.det_size / aw)
    det_w = int(aw * det_scale)
    det_h = int(ah * det_scale)
    print(f"Camera: {aw}x{ah} @ {af} FPS", flush=True)
    print(f"Detection resolution: {det_w}x{det_h} (scale={det_scale:.3f})", flush=True)
    if args.detector == "mediapipe":
        print("Two-stage: MediaPipe palm (CPU, async) → RTMPose-M landmarks (NPU)", flush=True)
    else:
        print("Two-stage: YOLO11n palm (NPU, async) → RTMPose-M landmarks (NPU)", flush=True)
    print("Press 'q' to quit.", flush=True)

    # --- Pre-allocate buffers ---
    crop_bgr_buf = np.empty((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    crop_rgb_buf = np.empty((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    inp_buf = np.empty((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    det_frame_buf = np.empty((det_h, det_w, 3), dtype=np.uint8)
    det_rgb_buf = np.empty((det_h, det_w, 3), dtype=np.uint8)

    # --- Filters (per hand slot) ---
    hand_filters: list[OneEuroFilter2D] = [
        OneEuroFilter2D(
            min_cutoff=args.smooth_min_cutoff,
            beta=args.smooth_beta,
            d_cutoff=args.smooth_d_cutoff,
        )
        for _ in range(args.max_hands)
    ]
    hand_kalmans: list[BboxKalman] = [BboxKalman() for _ in range(args.max_hands)]
    slot_miss_frames = [0] * args.max_hands

    # --- Async detection thread ---
    det_lock = threading.Lock()
    det_bboxes: list[tuple[int, int, int, int]] = []
    det_busy = False
    det_ms = 0.0

    def _run_detection(frame_bgr: np.ndarray, full_w: int, full_h: int) -> None:
        nonlocal det_bboxes, det_busy, det_ms
        cv2.resize(frame_bgr, (det_w, det_h), dst=det_frame_buf, interpolation=cv2.INTER_AREA)
        cv2.cvtColor(det_frame_buf, cv2.COLOR_BGR2RGB, dst=det_rgb_buf)
        t = time.perf_counter()

        new_bboxes: list[tuple[int, int, int, int]] = []

        if args.detector == "mediapipe" and hands is not None:
            results = hands.process(det_rgb_buf)
            if results.multi_hand_landmarks:
                for hlm in results.multi_hand_landmarks:
                    xs = [lm.x * full_w for lm in hlm.landmark]
                    ys = [lm.y * full_h for lm in hlm.landmark]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
                    new_bboxes.append((x1, y1, x2 - x1, y2 - y1))
        elif args.detector == "yolo" and yolo_npu is not None:
            det_inp = np.expand_dims(det_rgb_buf, axis=0)  # (1, H, W, 3) uint8
            det_out = yolo_npu.run([det_inp])
            new_bboxes = decode_yolo_detections(
                det_out[0], full_w, full_h, YOLO_DET_SIZE,
                conf_thresh=args.yolo_conf, iou_thresh=args.yolo_iou,
                max_det=args.max_hands,
            )

        elapsed = (time.perf_counter() - t) * 1000

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
        t_now = time.perf_counter()

        for hand_idx, bbox_xywh in enumerate(current_bboxes):
            if hand_idx >= args.max_hands:
                break

            # Use Kalman-predicted bbox if available
            kf = hand_kalmans[hand_idx]
            if not args.no_kalman:
                if not kf.initialized:
                    kf.init(bbox_xywh)
                else:
                    bbox_xywh = kf.predict()

            _, transform = crop_hand_square(
                frame, bbox_xywh, expand=1.3, output_size=INPUT_SIZE, dst_bgr=crop_bgr_buf,
            )

            cv2.cvtColor(crop_bgr_buf, cv2.COLOR_BGR2RGB, dst=crop_rgb_buf)
            inp_buf[0] = crop_rgb_buf

            t_lm = time.perf_counter()
            lm_out = landmark_npu.run([inp_buf])
            lm_dt = time.perf_counter() - t_lm
            lm_times.append(lm_dt * 1000)

            lx, ly = lm_out[0], lm_out[1]
            conf = simcc_confidence_numpy(lx, ly)

            xy = decode_simcc_soft_argmax_numpy(lx, ly, split_ratio=SIMCC_SPLIT_RATIO)
            kp_crop = xy[0] if xy.ndim == 3 else xy
            kp_src = map_landmarks_back(kp_crop, transform)

            # Apply One-Euro smoothing
            if not args.no_smooth:
                kp_src = hand_filters[hand_idx](kp_src.astype(np.float32), t_now)

            vis = draw_hand_21(vis, kp_src, radius=4, line_type=cv2.LINE_8)
            num_hands += 1

            # Derive bbox from landmarks and update Kalman
            lm_xs = kp_src[:, 0]
            lm_ys = kp_src[:, 1]
            nx1, ny1 = int(lm_xs.min()), int(lm_ys.min())
            nx2, ny2 = int(lm_xs.max()), int(lm_ys.max())
            new_bbox = (nx1, ny1, nx2 - nx1, ny2 - ny1)
            updated_bboxes.append(new_bbox)

            if not args.no_kalman:
                kf.update(new_bbox)

            cv2.putText(vis, f"conf={conf:.3f}", (nx1, ny1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Reset filters for inactive slots (immediate if --hand-lost-frames 0, else after N misses)
        for i in range(args.max_hands):
            if i < len(current_bboxes):
                slot_miss_frames[i] = 0
                continue
            slot_miss_frames[i] += 1
            thresh = args.hand_lost_frames
            if thresh <= 0 or slot_miss_frames[i] >= thresh:
                hand_filters[i].reset()
                hand_kalmans[i].reset()
                slot_miss_frames[i] = 0

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
    if hands is not None:
        hands.close()
    if yolo_npu is not None:
        yolo_npu.release()
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
