"""Letterbox resize and coordinate transforms for fixed square input (default 256×256)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int
    dst: int = 256

    def map_xy_norm_to_dst(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        """MediaPipe normalized [0,1] x,y in source image -> pixel coords in dst x dst."""
        px = x_norm * self.src_w
        py = y_norm * self.src_h
        return self.map_xy_src_to_dst(px, py)

    def map_xy_src_to_dst(self, px: float, py: float) -> Tuple[float, float]:
        x = px * self.scale + self.pad_x
        y = py * self.scale + self.pad_y
        return x, y

    def map_xy_dst_to_src(self, xd: float, yd: float) -> Tuple[float, float]:
        """Inverse: letterbox dst pixel coords -> source image pixel coords."""
        px = (xd - self.pad_x) / self.scale
        py = (yd - self.pad_y) / self.scale
        return px, py


def letterbox_params(src_w: int, src_h: int, dst: int = 160) -> Letterbox:
    scale = min(dst / src_w, dst / src_h)
    nw = int(round(src_w * scale))
    nh = int(round(src_h * scale))
    pad_x = (dst - nw) * 0.5
    pad_y = (dst - nh) * 0.5
    return Letterbox(
        scale=scale, pad_x=pad_x, pad_y=pad_y, src_w=src_w, src_h=src_h, dst=dst
    )


def letterbox_image(
    image_bgr: np.ndarray, dst: int = 256
) -> Tuple[np.ndarray, Letterbox]:
    import cv2

    h, w = image_bgr.shape[:2]
    lb = letterbox_params(w, h, dst)
    nw = int(round(w * lb.scale))
    nh = int(round(h * lb.scale))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((dst, dst, 3), 114, dtype=np.uint8)
    x0 = int(round(lb.pad_x))
    y0 = int(round(lb.pad_y))
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out, lb
