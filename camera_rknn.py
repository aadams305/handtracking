"""Live camera inference using RKNN NPU on Orange Pi 5 (RK3588).

Uses rknn-toolkit-lite2 (aarch64) for hardware-accelerated inference on the
RK3588's 6-TOPS NPU. Supports both single-stage (landmark only) and two-stage
(palm detection + landmark regression) pipelines.

Usage (single-stage, landmark only):
    python3 camera_rknn.py --landmark-rknn models/hand_simcc.rknn

Usage (two-stage):
    python3 camera_rknn.py --palm-rknn models/palm_det.rknn --landmark-rknn models/hand_simcc.rknn

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

from handtracking.geometry import letterbox_image, map_keypoints_lb_to_src
from handtracking.models.rtmpose_hand import INPUT_SIZE, MEAN as RTMPOSE_MEAN, STD as RTMPOSE_STD
from handtracking.models.palm_detector import (
    PALM_INPUT_SIZE,
    PalmDetection,
    _generate_anchors,
    crop_palm_region,
    map_landmarks_to_source,
)
from handtracking.simcc_numpy import (
    decode_simcc_soft_argmax_numpy,
    simcc_confidence_numpy,
    keypoints_collapsed,
)
from handtracking.viz import draw_hand_21

PIXEL_MEAN = np.array(RTMPOSE_MEAN, dtype=np.float32)
PIXEL_STD = np.array(RTMPOSE_STD, dtype=np.float32)


def preprocess_for_rknn(image_bgr: np.ndarray, target_size: int) -> np.ndarray:
    """Preprocess BGR image for RKNN: letterbox → RGB → NHWC uint8.

    RKNN handles mean/std normalization internally (configured at export time),
    so we pass uint8 RGB directly.
    """
    lb_img, lb = letterbox_image(image_bgr, target_size)
    rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0), lb


def preprocess_float(image_bgr: np.ndarray, target_size: int) -> tuple[np.ndarray, object]:
    """Preprocess for float-mode RKNN: RTMPose pixel-space normalisation."""
    lb_img, lb = letterbox_image(image_bgr, target_size)
    rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb - PIXEL_MEAN) / PIXEL_STD
    nhwc = np.expand_dims(normalized, axis=0)
    return nhwc, lb


class RKNNInference:
    """Wrapper for rknn-toolkit-lite2 inference on RK3588."""

    def __init__(self, rknn_path: str, core_mask: int = 1) -> None:
        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            try:
                from rknn.api import RKNN as RKNNLite
            except ImportError:
                raise ImportError(
                    "Neither rknn-toolkit-lite2 nor rknn-toolkit2 found.\n"
                    "  Orange Pi 5 (aarch64): pip install rknn-toolkit-lite2\n"
                    "  x86 host (simulation): pip install rknn-toolkit2"
                )

        self.rknn = RKNNLite()
        print(f"Loading RKNN model: {rknn_path}", flush=True)
        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model (ret={ret})")

        core_map = {
            1: 0x01,  # NPU core 0
            2: 0x02,  # NPU core 1
            3: 0x04,  # NPU core 2
            7: 0x07,  # All 3 NPU cores
        }
        rknn_core = core_map.get(core_mask, 0x01)

        try:
            ret = self.rknn.init_runtime(core_mask=rknn_core)
        except TypeError:
            ret = self.rknn.init_runtime()

        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime (ret={ret})")
        print(f"RKNN runtime initialized (core_mask={core_mask})", flush=True)

    def run(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """Run inference. inputs is a list of numpy arrays (one per model input)."""
        outputs = self.rknn.inference(inputs=inputs)
        return outputs

    def release(self) -> None:
        self.rknn.release()


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS. boxes: [N, 4] x1y1x2y2, scores: [N] sorted desc."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep: list[int] = []
    suppressed = np.zeros(len(scores), dtype=bool)
    for i in range(len(scores)):
        if suppressed[i]:
            continue
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[i + 1:])
        yy1 = np.maximum(y1[i], y1[i + 1:])
        xx2 = np.minimum(x2[i], x2[i + 1:])
        yy2 = np.minimum(y2[i], y2[i + 1:])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[i + 1:] - inter + 1e-6)
        suppressed[i + 1:] |= iou > iou_thresh
    return keep


def decode_palm_detections(
    box_offsets: np.ndarray,
    score_logits: np.ndarray,
    score_thresh: float = 0.5,
    iou_thresh: float = 0.3,
    max_hands: int = 2,
) -> list[PalmDetection]:
    """Decode palm detector RKNN outputs into filtered detections."""
    anchors = _generate_anchors(PALM_INPUT_SIZE)
    scores = 1.0 / (1.0 + np.exp(-score_logits[0]))

    cx = anchors[:, 0] + box_offsets[0, :, 0]
    cy = anchors[:, 1] + box_offsets[0, :, 1]
    w = np.exp(np.clip(box_offsets[0, :, 2], -5, 2))
    h = np.exp(np.clip(box_offsets[0, :, 3], -5, 2))

    mask = scores > score_thresh
    if not mask.any():
        return []

    cx_f, cy_f, w_f, h_f, sc_f = cx[mask], cy[mask], w[mask], h[mask], scores[mask]
    x1 = cx_f - w_f / 2
    y1 = cy_f - h_f / 2
    x2 = cx_f + w_f / 2
    y2 = cy_f + h_f / 2

    order = sc_f.argsort()[::-1]
    boxes_sorted = np.stack([x1[order], y1[order], x2[order], y2[order]], axis=1)
    keep = nms_numpy(boxes_sorted, sc_f[order], iou_thresh)

    dets = []
    for k in keep[:max_hands]:
        idx = order[k]
        dets.append(PalmDetection(
            cx=float(cx_f[idx]), cy=float(cy_f[idx]),
            w=float(w_f[idx]), h=float(h_f[idx]),
            score=float(sc_f[idx]),
        ))
    return dets


def main() -> None:
    ap = argparse.ArgumentParser(description="RKNN NPU camera inference on Orange Pi 5 (RK3588)")
    ap.add_argument("--landmark-rknn", type=Path, default=Path("models/rtmpose_hand.rknn"))
    ap.add_argument("--palm-rknn", type=Path, default=None,
                     help="Palm detector RKNN (omit for single-stage)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--core-mask", type=int, default=7,
                     help="NPU core mask: 1=core0, 2=core1, 4=core2, 7=all three")
    ap.add_argument("--conf-threshold", type=float, default=0.02)
    ap.add_argument("--palm-thresh", type=float, default=0.5)
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--preview-max-width", type=int, default=960)
    ap.add_argument("--uint8-input", action="store_true",
                     help="Pass uint8 RGB (for quantized models with built-in normalization)")
    args = ap.parse_args()

    os.chdir(_REPO)

    two_stage = args.palm_rknn is not None and args.palm_rknn.is_file()

    landmark_npu = RKNNInference(str(args.landmark_rknn), core_mask=args.core_mask)
    palm_npu: RKNNInference | None = None
    if two_stage:
        palm_npu = RKNNInference(str(args.palm_rknn), core_mask=args.core_mask)
        print("Two-stage mode: palm detection → landmark regression", flush=True)
    else:
        print("Single-stage mode: landmark regression on full frame", flush=True)

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

    mode_str = "two-stage" if two_stage else "single-stage"
    win = f"RKNN Hand Tracking ({mode_str}) q=quit"
    fps_t0 = time.perf_counter()
    fps_n = 0
    infer_times: list[float] = []

    print("Press 'q' to quit.", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()

        if two_stage and palm_npu is not None:
            if args.uint8_input:
                palm_inp, _ = preprocess_for_rknn(frame, PALM_INPUT_SIZE)
            else:
                palm_inp, _ = preprocess_float(frame, PALM_INPUT_SIZE)

            palm_out = palm_npu.run([palm_inp])
            palms = decode_palm_detections(
                palm_out[0], palm_out[1],
                score_thresh=args.palm_thresh,
                max_hands=args.max_hands,
            )

            vis = frame.copy()
            for det in palms:
                crop, transform = crop_palm_region(frame, det, output_size=INPUT_SIZE)
                if args.uint8_input:
                    lm_inp = np.expand_dims(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), axis=0)
                else:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
                    normalized = (rgb - PIXEL_MEAN) / PIXEL_STD
                    lm_inp = np.expand_dims(normalized, axis=0)

                lm_out = landmark_npu.run([lm_inp])
                lx, ly = lm_out[0], lm_out[1]
                conf = simcc_confidence_numpy(lx, ly)

                if conf >= args.conf_threshold:
                    xy = decode_simcc_soft_argmax_numpy(lx, ly)[0]
                    kp_src = map_landmarks_to_source(xy, transform)
                    vis = draw_hand_21(vis, kp_src, radius=4, line_type=cv2.LINE_8)

                    handedness = ""
                    if len(lm_out) >= 4:
                        hand_sigmoid = 1.0 / (1.0 + np.exp(-float(lm_out[3].flat[0])))
                        handedness = "Right" if hand_sigmoid > 0.5 else "Left"
                    label = f"{conf:.2f} {handedness}"
                    cv2.putText(vis, label, (10, vis.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA)
        else:
            if args.uint8_input:
                lm_inp, lb = preprocess_for_rknn(frame, INPUT_SIZE)
            else:
                lm_inp, lb = preprocess_float(frame, INPUT_SIZE)

            lm_out = landmark_npu.run([lm_inp])
            lx, ly = lm_out[0], lm_out[1]
            conf = simcc_confidence_numpy(lx, ly)

            vis = frame
            if conf >= args.conf_threshold:
                xy = decode_simcc_soft_argmax_numpy(lx, ly)
                kp_src = map_keypoints_lb_to_src(lb, xy[0].astype(np.float32))
                vis = draw_hand_21(frame, kp_src, radius=4, line_type=cv2.LINE_8)

                handedness = ""
                if len(lm_out) >= 4:
                    hand_sigmoid = 1.0 / (1.0 + np.exp(-float(lm_out[3].flat[0])))
                    handedness = "Right" if hand_sigmoid > 0.5 else "Left"
                label = f"conf={conf:.2f} {handedness}"
                cv2.putText(vis, label, (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA)

        dt = time.perf_counter() - t0
        infer_times.append(dt * 1000)

        fps_n += 1
        fps_dt = time.perf_counter() - fps_t0
        if fps_dt >= 1.0:
            avg_ms = sum(infer_times[-fps_n:]) / max(1, fps_n)
            try:
                cv2.setWindowTitle(win, f"{win}  ~{fps_n / fps_dt:.0f} FPS  {avg_ms:.1f}ms")
            except cv2.error:
                pass
            fps_t0 = time.perf_counter()
            fps_n = 0

        if args.preview_max_width > 0 and vis.shape[1] > args.preview_max_width:
            pw = args.preview_max_width
            ph = int(round(vis.shape[0] * (pw / float(vis.shape[1]))))
            vis = cv2.resize(vis, (pw, ph), interpolation=cv2.INTER_AREA)

        cv2.imshow(win, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if infer_times:
        avg = sum(infer_times) / len(infer_times)
        p50 = sorted(infer_times)[len(infer_times) // 2]
        p99 = sorted(infer_times)[int(len(infer_times) * 0.99)]
        print(f"\nLatency stats ({len(infer_times)} frames): avg={avg:.1f}ms p50={p50:.1f}ms p99={p99:.1f}ms")

    cap.release()
    landmark_npu.release()
    if palm_npu is not None:
        palm_npu.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
