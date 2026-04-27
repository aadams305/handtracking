"""
Distill FreiHAND (or any folder of images) with MediaPipe teacher -> JSONL manifest.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from handtracking.dataset_manifest import DistilledSample, LetterboxMeta, write_manifest
from handtracking.geometry import letterbox_image
from handtracking.teacher import MediaPipeTeacher, extract_21_points_pixel


def find_freihand_rgb_dir(data_root: Path) -> Path:
    """FreiHAND layout: <root>/training/rgb/*.jpg or similar."""
    for candidate in (
        data_root / "training" / "rgb",
        data_root / "rgb",
        data_root,
    ):
        if candidate.is_dir():
            imgs = list(candidate.glob("*.jpg")) + list(candidate.glob("*.png"))
            if imgs:
                return candidate
    raise FileNotFoundError(
        f"No images under {data_root} (expected training/rgb/*.jpg or rgb/*.jpg)"
    )


def distill_images(
    image_paths: list[Path],
    out_manifest: Path,
    max_samples: int | None = None,
    seed: int = 42,
) -> int:
    random.seed(seed)
    if max_samples:
        image_paths = random.sample(image_paths, min(max_samples, len(image_paths)))
    samples: list[DistilledSample] = []
    with MediaPipeTeacher() as teacher:
        print("MediaPipe ready; distilling with progress bar…", flush=True)
        for p in tqdm(
            image_paths,
            desc="distill",
            file=sys.stderr,
            mininterval=1.0,
            dynamic_ncols=True,
        ):
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            tr = teacher.process_bgr(img)
            if not tr.ok or tr.landmarks_norm is None:
                continue
            kp_src = extract_21_points_pixel(tr.landmarks_norm[:, :2], w, h)
            lb_img, lb = letterbox_image(img, 256)
            kp_dst = np.zeros((21, 2), dtype=np.float32)
            for i in range(21):
                kp_dst[i, 0], kp_dst[i, 1] = lb.map_xy_src_to_dst(
                    float(kp_src[i, 0]), float(kp_src[i, 1])
                )
            meta = LetterboxMeta(
                scale=lb.scale,
                pad_x=lb.pad_x,
                pad_y=lb.pad_y,
                src_w=lb.src_w,
                src_h=lb.src_h,
                dst=256,
            )
            samples.append(
                DistilledSample(
                    image_path=str(p.resolve()),
                    letterbox=meta,
                    keypoints_xy=kp_dst.tolist(),
                )
            )
    write_manifest(samples, out_manifest)
    return len(samples)


def main() -> None:
    ap = argparse.ArgumentParser(description="MediaPipe teacher -> manifest JSONL")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="FreiHAND root (contains training/rgb) or image folder",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="*.jpg",
        help="If --data-root is a directory, glob pattern for images",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/distilled/manifest.jsonl"),
        help="Output JSONL path",
    )
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.data_root is None:
        print(
            "No --data-root: create data/demo_images with a few .jpg files or set FREIHAND path.",
            file=sys.stderr,
        )
        sys.exit(1)

    root = args.data_root
    if root.is_file():
        paths = [root]
    elif (root / "training" / "rgb").is_dir() or (root / "rgb").is_dir():
        rgb = find_freihand_rgb_dir(root)
        print(f"Globbing images under {rgb} (no sort — large FreiHAND sets can take 1–5 min)…", flush=True)
        jp = list(rgb.glob("*.jpg"))
        print(f"  {len(jp)} .jpg", flush=True)
        pn = list(rgb.glob("*.png"))
        print(f"  {len(pn)} .png", flush=True)
        paths = jp + pn
    else:
        paths = sorted(root.glob(args.glob))
        if not paths:
            paths = list(root.rglob("*.jpg")) + list(root.rglob("*.png"))

    if not paths:
        print(f"No images found under {root}", file=sys.stderr)
        sys.exit(1)

    n_img = len(paths)
    print(f"Found {n_img} image files under {root}.", flush=True)
    if args.max_samples:
        print(f"Using --max-samples {args.max_samples} (random subset).", flush=True)
    print(
        "Loading MediaPipe Hand Landmarker (first run may download ~10 MB and take several minutes; no tqdm yet).",
        flush=True,
    )

    n = distill_images(paths, args.out, max_samples=args.max_samples, seed=args.seed)
    print(f"Wrote {n} samples to {args.out}", flush=True)


if __name__ == "__main__":
    main()
