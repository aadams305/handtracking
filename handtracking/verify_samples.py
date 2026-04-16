"""Verification Phase 1: samples.png with 10 images and 10-point overlays."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from handtracking.dataset_manifest import iter_manifest
from handtracking.geometry import letterbox_image
from handtracking.viz import draw_hand_10


def load_keypoints_on_letterboxed(
    image_path: str, keypoints_xy: list
) -> tuple[np.ndarray, np.ndarray]:
    """Return letterboxed BGR 160x160 and keypoints (10,2) in same space."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    lb_img, _ = letterbox_image(img, 160)
    kp = np.array(keypoints_xy, dtype=np.float32)
    return lb_img, kp


def make_grid(
    manifest_path: Path,
    out_png: Path,
    num: int = 10,
    seed: int = 0,
) -> None:
    rows = list(iter_manifest(manifest_path))
    if len(rows) < num:
        raise SystemExit(
            f"Need at least {num} manifest lines, got {len(rows)}. Run distill_freihand first."
        )
    random.seed(seed)
    pick = random.sample(rows, num)
    cols = 5
    r = 2
    cell = 160
    canvas = np.zeros((r * cell, cols * cell, 3), dtype=np.uint8)
    for i, sample in enumerate(pick):
        lb_img, kp = load_keypoints_on_letterboxed(
            sample.image_path, sample.keypoints_xy
        )
        vis = draw_hand_10(lb_img, kp)
        y, x = divmod(i, cols)
        canvas[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell] = vis
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)
    print(f"Wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/distilled/manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/distilled/samples.png"))
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    make_grid(args.manifest, args.out, num=args.num, seed=args.seed)


if __name__ == "__main__":
    main()
