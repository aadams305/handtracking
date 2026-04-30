"""Two-stage hand tracking pipeline: BlazePalm detection → SimCC landmark regression.

Usage (live camera):
    PYTHONPATH=. python3 -m handtracking.two_stage_pipeline --palm-onnx models/palm_det.onnx --landmark-onnx models/hand_simcc.onnx

Usage (single image):
    PYTHONPATH=. python3 -m handtracking.two_stage_pipeline --image test.jpg --palm-onnx models/palm_det.onnx --landmark-onnx models/hand_simcc.onnx
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from handtracking.models.palm_detector import (
    PALM_INPUT_SIZE,
    PalmDetection,
    crop_palm_region,
    map_landmarks_to_source,
)
from handtracking.simcc_numpy import (
    bgr_letterbox_to_nchw_batch,
    decode_simcc_soft_argmax_numpy,
    simcc_confidence_numpy,
)
from handtracking.geometry import letterbox_image
from handtracking.models.hand_simcc import INPUT_SIZE
from handtracking.viz import draw_hand_21


@dataclass
class HandResult:
    """Full result for one detected hand."""
    landmarks_src: np.ndarray
    confidence: float
    handedness: str
    palm_box: PalmDetection


class TwoStagePipeline:
    """ONNX-based two-stage pipeline: palm detection → landmark regression."""

    def __init__(
        self,
        palm_onnx: str | Path,
        landmark_onnx: str | Path,
        palm_score_thresh: float = 0.5,
        palm_iou_thresh: float = 0.3,
        landmark_conf_thresh: float = 0.02,
        max_hands: int = 2,
    ) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        provs = ["CPUExecutionProvider"]

        self.palm_sess = ort.InferenceSession(str(palm_onnx), so, providers=provs)
        self.palm_in_name = self.palm_sess.get_inputs()[0].name

        self.lm_sess = ort.InferenceSession(str(landmark_onnx), so, providers=provs)
        self.lm_in_name = self.lm_sess.get_inputs()[0].name

        self.palm_score_thresh = palm_score_thresh
        self.palm_iou_thresh = palm_iou_thresh
        self.landmark_conf_thresh = landmark_conf_thresh
        self.max_hands = max_hands

    def _preprocess_palm(self, image_bgr: np.ndarray) -> np.ndarray:
        """Letterbox to PALM_INPUT_SIZE and normalize to NCHW float32 batch."""
        lb_img, _ = letterbox_image(image_bgr, PALM_INPUT_SIZE)
        return bgr_letterbox_to_nchw_batch(lb_img)

    def _detect_palms(self, image_bgr: np.ndarray) -> list[PalmDetection]:
        """Run palm detector and return filtered detections in normalized coords."""
        inp = self._preprocess_palm(image_bgr)
        box_offsets, score_logits = self.palm_sess.run(None, {self.palm_in_name: inp})

        scores = 1.0 / (1.0 + np.exp(-score_logits[0]))  # sigmoid
        from handtracking.models.palm_detector import _generate_anchors
        anchors = _generate_anchors(PALM_INPUT_SIZE)

        cx = anchors[:, 0] + box_offsets[0, :, 0]
        cy = anchors[:, 1] + box_offsets[0, :, 1]
        w = np.exp(np.clip(box_offsets[0, :, 2], -5, 2))
        h = np.exp(np.clip(box_offsets[0, :, 3], -5, 2))

        mask = scores > self.palm_score_thresh
        if not mask.any():
            return []

        cx_f, cy_f, w_f, h_f, sc_f = cx[mask], cy[mask], w[mask], h[mask], scores[mask]

        x1 = cx_f - w_f / 2
        y1 = cy_f - h_f / 2
        x2 = cx_f + w_f / 2
        y2 = cy_f + h_f / 2

        order = sc_f.argsort()[::-1]
        keep = self._nms_numpy(
            np.stack([x1[order], y1[order], x2[order], y2[order]], axis=1),
            sc_f[order],
            self.palm_iou_thresh,
        )

        dets = []
        for k in keep[: self.max_hands]:
            idx = order[k]
            dets.append(PalmDetection(
                cx=float(cx_f[idx]), cy=float(cy_f[idx]),
                w=float(w_f[idx]), h=float(h_f[idx]),
                score=float(sc_f[idx]),
            ))
        return dets

    @staticmethod
    def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
        """Simple greedy NMS in numpy (boxes in x1y1x2y2 format, already sorted by score desc)."""
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

    def _regress_landmarks(
        self, image_bgr: np.ndarray, det: PalmDetection
    ) -> HandResult | None:
        """Crop palm region, run landmark model, map back to source coords."""
        crop, transform = crop_palm_region(image_bgr, det, output_size=INPUT_SIZE)
        inp = bgr_letterbox_to_nchw_batch(crop)
        out = self.lm_sess.run(None, {self.lm_in_name: inp})
        lx, ly = out[0], out[1]

        conf = simcc_confidence_numpy(lx, ly)
        if conf < self.landmark_conf_thresh:
            return None

        xy = decode_simcc_soft_argmax_numpy(lx, ly)  # [1, 21, 2]
        landmarks_crop = xy[0]

        landmarks_src = map_landmarks_to_source(landmarks_crop, transform)

        handedness = ""
        if len(out) >= 4:
            hand_sigmoid = 1.0 / (1.0 + np.exp(-float(out[3].flat[0])))
            handedness = "Right" if hand_sigmoid > 0.5 else "Left"

        return HandResult(
            landmarks_src=landmarks_src,
            confidence=conf,
            handedness=handedness,
            palm_box=det,
        )

    def process_frame(self, image_bgr: np.ndarray) -> list[HandResult]:
        """Full two-stage pipeline on a single BGR frame."""
        palms = self._detect_palms(image_bgr)
        results: list[HandResult] = []
        for det in palms:
            hand = self._regress_landmarks(image_bgr, det)
            if hand is not None:
                results.append(hand)
        return results


def draw_results(
    frame: np.ndarray,
    results: list[HandResult],
    draw_bbox: bool = True,
) -> np.ndarray:
    """Draw all detected hands with landmarks and optional bounding boxes."""
    vis = frame.copy()
    for hand in results:
        vis = draw_hand_21(vis, hand.landmarks_src, radius=4, line_type=cv2.LINE_8)

        if draw_bbox:
            h, w = vis.shape[:2]
            box = hand.palm_box
            x1 = int(box.x1 * w)
            y1 = int(box.y1 * h)
            x2 = int(box.x2 * w)
            y2 = int(box.y2 * h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)

        label = f"{hand.confidence:.2f}"
        if hand.handedness:
            label += f" {hand.handedness}"
        cv2.putText(
            vis, label,
            (10, vis.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA,
        )
    return vis


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-stage hand tracking: palm detect → SimCC landmarks")
    ap.add_argument("--palm-onnx", type=Path, default=Path("models/palm_det.onnx"))
    ap.add_argument("--landmark-onnx", type=Path, default=Path("models/hand_simcc.onnx"))
    ap.add_argument("--image", type=Path, default=None, help="Single image mode")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--palm-thresh", type=float, default=0.5)
    ap.add_argument("--max-hands", type=int, default=2)
    args = ap.parse_args()

    pipeline = TwoStagePipeline(
        palm_onnx=args.palm_onnx,
        landmark_onnx=args.landmark_onnx,
        palm_score_thresh=args.palm_thresh,
        max_hands=args.max_hands,
    )

    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            raise SystemExit(f"Cannot read image: {args.image}")
        results = pipeline.process_frame(img)
        print(f"Detected {len(results)} hand(s)")
        vis = draw_results(img, results)
        cv2.imwrite("two_stage_output.jpg", vis)
        print("Saved two_stage_output.jpg")
        return

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    win = "Two-Stage Hand Tracking (q=quit)"
    fps_t0 = time.perf_counter()
    fps_n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = pipeline.process_frame(frame)
        vis = draw_results(frame, results)

        fps_n += 1
        dt = time.perf_counter() - fps_t0
        if dt >= 1.0:
            try:
                cv2.setWindowTitle(win, f"{win}  ~{fps_n / dt:.0f} FPS")
            except cv2.error:
                pass
            fps_t0 = time.perf_counter()
            fps_n = 0

        cv2.imshow(win, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
