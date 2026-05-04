"""Compare MediaPipe vs RTMPose-M on a single image.

Outputs side-by-side visualization and prints metrics.

Usage:
    python3 compare_mp_vs_rtmpose.py --image IMG_7271.jpeg --onnx models/rtmpose_hand.onnx
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from handtracking.geometry import letterbox_image, letterbox_params
from handtracking.models.rtmpose_hand import INPUT_SIZE, MEAN, STD, SIMCC_SPLIT_RATIO
from handtracking.simcc_numpy import decode_simcc_soft_argmax_numpy, simcc_confidence_numpy
from handtracking.topology import NUM_HAND_JOINTS, HAND_21_NAMES
from handtracking.viz import draw_hand_21


def run_mediapipe(img_rgb: np.ndarray) -> np.ndarray | None:
    """Run MediaPipe Hands and return 21x2 pixel coords or None."""
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    h, w = img_rgb.shape[:2]
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.3) as hands:
        results = hands.process(img_rgb)
    if results.multi_hand_landmarks is None:
        return None
    lm = results.multi_hand_landmarks[0].landmark
    kp = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)
    return kp


def run_rtmpose_onnx(img_bgr: np.ndarray, onnx_path: str) -> tuple[np.ndarray, float]:
    """Run RTMPose ONNX and return 21x2 pixel coords in source image + confidence."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    lb_img, lb = letterbox_image(img_bgr, INPUT_SIZE)
    rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array(MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(STD, dtype=np.float32).reshape(1, 1, 3)
    normalized = (rgb - mean) / std
    inp = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]  # [1, 3, 256, 256]

    t0 = time.perf_counter()
    out = sess.run(None, {"input": inp.astype(np.float32)})
    dt = time.perf_counter() - t0

    lx, ly = out[0], out[1]
    conf = simcc_confidence_numpy(lx, ly)
    xy_lb = decode_simcc_soft_argmax_numpy(lx, ly, split_ratio=SIMCC_SPLIT_RATIO)  # [21, 2]

    # Map from letterbox back to source image
    kp_src = np.zeros_like(xy_lb)
    kp_src[:, 0] = (xy_lb[:, 0] - lb.pad_x) / lb.scale
    kp_src[:, 1] = (xy_lb[:, 1] - lb.pad_y) / lb.scale

    return kp_src, conf, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="IMG_7271.jpeg")
    ap.add_argument("--onnx", type=str, default="models/rtmpose_hand.onnx")
    ap.add_argument("--out", type=str, default="comparison_rtmpose_vs_mp.png")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {args.image}")
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Image: {args.image} ({w}x{h})")
    print()

    # MediaPipe
    print("Running MediaPipe...")
    t0 = time.perf_counter()
    mp_kp = run_mediapipe(img_rgb)
    mp_dt = time.perf_counter() - t0
    if mp_kp is not None:
        print(f"  MediaPipe: {mp_kp.shape[0]} joints detected ({mp_dt*1000:.1f}ms)")
    else:
        print("  MediaPipe: No hand detected")

    # RTMPose
    print("Running RTMPose-M ONNX...")
    rtm_kp, rtm_conf, rtm_dt = run_rtmpose_onnx(img, args.onnx)
    print(f"  RTMPose: conf={rtm_conf:.3f} ({rtm_dt*1000:.1f}ms)")

    # Metrics (if both detected)
    print()
    if mp_kp is not None:
        errors = np.linalg.norm(rtm_kp - mp_kp, axis=1)
        mpjpe = errors.mean()
        pck_20 = (errors < 20).mean() * 100
        pck_10 = (errors < 10).mean() * 100
        pck_5 = (errors < 5).mean() * 100

        print("=== RTMPose vs MediaPipe (using MP as reference) ===")
        print(f"  MPJPE:    {mpjpe:.2f} px")
        print(f"  PCK@5px:  {pck_5:.1f}%")
        print(f"  PCK@10px: {pck_10:.1f}%")
        print(f"  PCK@20px: {pck_20:.1f}%")
        print(f"  Worst:    {errors.max():.1f}px (joint {errors.argmax()} = {HAND_21_NAMES[errors.argmax()]})")
        print(f"  Best:     {errors.min():.1f}px (joint {errors.argmin()} = {HAND_21_NAMES[errors.argmin()]})")
        print()

        print("Per-joint errors (px):")
        for i, (name, err) in enumerate(zip(HAND_21_NAMES, errors)):
            bar = "█" * int(err / 2)
            print(f"  [{i:2d}] {name:12s}: {err:5.1f}  {bar}")
    else:
        print("Cannot compute metrics — MediaPipe did not detect a hand.")

    # Visualization
    vis_mp = img.copy()
    vis_rtm = img.copy()

    if mp_kp is not None:
        vis_mp = draw_hand_21(vis_mp, mp_kp, radius=5, line_type=cv2.LINE_AA)
        cv2.putText(vis_mp, f"MediaPipe ({mp_dt*1000:.0f}ms)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(vis_mp, "MediaPipe: No detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    vis_rtm = draw_hand_21(vis_rtm, rtm_kp, radius=5, line_type=cv2.LINE_AA)
    cv2.putText(vis_rtm, f"RTMPose-M conf={rtm_conf:.2f} ({rtm_dt*1000:.0f}ms)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Stack side by side
    canvas = np.hstack([vis_mp, vis_rtm])
    cv2.imwrite(args.out, canvas)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
