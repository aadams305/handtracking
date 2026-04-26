"""NumPy SimCC decode + NCHW input (matches training / PyTorch) for ONNX Runtime."""

from __future__ import annotations

import numpy as np

from handtracking.models.hand_simcc import INPUT_SIZE, NUM_BINS
from handtracking.topology import NUM_HAND_JOINTS

NUM_JOINTS = NUM_HAND_JOINTS


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x, -50.0, 50.0))
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)


def decode_simcc_soft_argmax_numpy(
    lx: np.ndarray,
    ly: np.ndarray,
    input_size: float = float(INPUT_SIZE),
    num_bins: int = NUM_BINS,
) -> np.ndarray:
    """
    lx, ly: (J, num_bins) or (1, J, num_bins) float32 logits -> (J, 2) pixel coords in letterbox.
    """
    if lx.ndim == 3:
        lx = lx[0]
        ly = ly[0]
    bins = np.arange(num_bins, dtype=np.float32) * (input_size / float(num_bins))
    px = softmax(lx.astype(np.float32), axis=-1)
    py = softmax(ly.astype(np.float32), axis=-1)
    x_coord = np.sum(px * bins[None, :], axis=-1)
    y_coord = np.sum(py * bins[None, :], axis=-1)
    return np.stack([x_coord, y_coord], axis=-1)


def bgr_letterbox_to_nchw_batch(img_bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR ``H×H`` letterbox (e.g. 256×256) -> float32 NCHW (1,3,H,H) ImageNet norm."""
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    ch = np.transpose(rgb, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    x = (ch - mean) / std
    return np.expand_dims(x, 0)


def keypoints_collapsed(kp_full: np.ndarray, frame_shape: tuple[int, int, ...]) -> bool:
    """True if all joints sit in a tight blob (untrained / degenerate student)."""
    h, w = frame_shape[0], frame_shape[1]
    if kp_full.size == 0:
        return True
    spread = 0.0
    for i in range(len(kp_full)):
        for j in range(i + 1, len(kp_full)):
            spread = max(spread, float(np.linalg.norm(kp_full[i] - kp_full[j])))
    return spread < 0.04 * float(min(h, w))
