"""NumPy SimCC decode + preprocessing for ONNX Runtime / RKNN inference.

Supports both:
  - Legacy 256-bin (1:1 pixel-to-bin) with ImageNet normalisation
  - RTMPose 512-bin (split_ratio=2.0) with pixel-space mean/std
"""

from __future__ import annotations

import numpy as np

from handtracking.models.rtmpose_hand import (
    INPUT_SIZE,
    MEAN as RTMPOSE_MEAN,
    NUM_BINS,
    SIMCC_SPLIT_RATIO,
    STD as RTMPOSE_STD,
)
from handtracking.topology import NUM_HAND_JOINTS

NUM_JOINTS = NUM_HAND_JOINTS

_BIN_CACHE: dict[tuple[int, float], np.ndarray] = {}
_RTMPOSE_CHW: tuple[np.ndarray, np.ndarray] | None = None
_IMAGENET_CHW: tuple[np.ndarray, np.ndarray] | None = None


def _bins_f32(num_bins: int, split_ratio: float) -> np.ndarray:
    key = (num_bins, split_ratio)
    if key not in _BIN_CACHE:
        _BIN_CACHE[key] = np.arange(num_bins, dtype=np.float32) / np.float32(split_ratio)
    return _BIN_CACHE[key]


def _rtmpose_mean_std() -> tuple[np.ndarray, np.ndarray]:
    global _RTMPOSE_CHW
    if _RTMPOSE_CHW is None:
        mean = np.array(RTMPOSE_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(RTMPOSE_STD, dtype=np.float32).reshape(3, 1, 1)
        _RTMPOSE_CHW = (mean, std)
    return _RTMPOSE_CHW


def _imagenet_mean_std() -> tuple[np.ndarray, np.ndarray]:
    global _IMAGENET_CHW
    if _IMAGENET_CHW is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        _IMAGENET_CHW = (mean, std)
    return _IMAGENET_CHW


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x, -50.0, 50.0))
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)


def decode_simcc_soft_argmax_numpy(
    lx: np.ndarray,
    ly: np.ndarray,
    split_ratio: float = SIMCC_SPLIT_RATIO,
) -> np.ndarray:
    """Decode SimCC logits to pixel coords.

    Args:
        lx, ly: [J, num_bins] or [1, J, num_bins] logits.
        split_ratio: SimCC split ratio (2.0 for RTMPose 512-bin, 1.0 for legacy 256-bin).

    Returns:
        [J, 2] pixel coordinates in letterbox space.
    """
    if lx.ndim == 3:
        lx, ly = lx[0], ly[0]
    num_bins = lx.shape[-1]
    bins = _bins_f32(num_bins, split_ratio)
    px = softmax(lx.astype(np.float32), axis=-1)
    py = softmax(ly.astype(np.float32), axis=-1)
    x_coord = (px * bins[None, :]).sum(axis=-1)
    y_coord = (py * bins[None, :]).sum(axis=-1)
    return np.stack([x_coord, y_coord], axis=-1)


def simcc_confidence_numpy(lx: np.ndarray, ly: np.ndarray) -> float:
    """Peakedness-based confidence from SimCC logits (no extra head needed)."""
    if lx.ndim == 3:
        lx, ly = lx[0], ly[0]
    px = softmax(lx, axis=-1)
    py = softmax(ly, axis=-1)
    peak_x = np.mean(np.max(px, axis=-1))
    peak_y = np.mean(np.max(py, axis=-1))
    num_bins = lx.shape[-1]
    raw = (peak_x + peak_y) / 2.0
    floor = 1.0 / num_bins
    return float(max(0.0, min(1.0, (raw - floor) / (1.0 - floor))))


def bgr_letterbox_to_nchw_rtmpose(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 [H, W, 3] -> float32 NCHW [1, 3, H, W] with RTMPose pixel-space normalisation."""
    rgb = img_bgr[:, :, ::-1].astype(np.float32)
    ch = np.transpose(rgb, (2, 0, 1))
    mean, std = _rtmpose_mean_std()
    return np.expand_dims((ch - mean) / std, 0)


def bgr_letterbox_to_nchw_batch(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> float32 NCHW [1,3,H,H] with RTMPose normalisation (default for new model)."""
    return bgr_letterbox_to_nchw_rtmpose(img_bgr)


def bgr_letterbox_to_nchw_imagenet(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> float32 NCHW [1,3,H,H] with ImageNet normalisation (legacy model)."""
    rgb = img_bgr[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
    ch = np.transpose(rgb, (2, 0, 1))
    mean, std = _imagenet_mean_std()
    return np.expand_dims((ch - mean) / std, 0)


def keypoints_collapsed(kp_full: np.ndarray, frame_shape: tuple[int, int, ...]) -> bool:
    """True if all joints sit in a tight blob (untrained / degenerate model)."""
    h, w = frame_shape[0], frame_shape[1]
    if kp_full.size == 0:
        return True
    spread = 0.0
    for i in range(len(kp_full)):
        for j in range(i + 1, len(kp_full)):
            spread = max(spread, float(np.linalg.norm(kp_full[i] - kp_full[j])))
    return spread < 0.04 * float(min(h, w))
