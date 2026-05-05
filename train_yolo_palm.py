"""Train YOLO11n for palm/hand detection from FreiHAND + RHD annotations.

Generates a YOLO-format detection dataset from native ground-truth keypoints
(computes tight bbox around 21 joints, expands by 20%), then trains YOLO11n.

Usage:
    # Generate dataset + train (default)
    python3 train_yolo_palm.py --freihand data/FreiHAND --rhd data/RHD --epochs 100

    # Only generate dataset (skip training)
    python3 train_yolo_palm.py --freihand data/FreiHAND --only-gen-data

    # Resume training from last checkpoint
    python3 train_yolo_palm.py --resume runs/detect/yolo_palm/weights/last.pt
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def load_freihand_bboxes(root: Path) -> list[tuple[Path, tuple[float, float, float, float]]]:
    """Load FreiHAND images and derive bounding boxes from 2D projections of 3D keypoints."""
    entries = []

    for split in ["training", "evaluation"]:
        img_dir = root / split / "rgb"
        if not img_dir.exists():
            img_dir = root / split
            if not img_dir.exists():
                continue

        xyz_file = root / split / "training_xyz.json"
        k_file = root / split / "training_K.json"
        if split == "evaluation":
            xyz_file = root / split / "evaluation_xyz.json"
            k_file = root / split / "evaluation_K.json"

        # Try multiple naming conventions
        for xf in [xyz_file, root / f"{split}_xyz.json", root / "training_xyz.json"]:
            if xf.exists():
                xyz_file = xf
                break
        for kf in [k_file, root / f"{split}_K.json", root / "training_K.json"]:
            if kf.exists():
                k_file = kf
                break

        if not xyz_file.exists() or not k_file.exists():
            continue

        with open(xyz_file) as f:
            all_xyz = json.load(f)
        with open(k_file) as f:
            all_K = json.load(f)

        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        for i, img_path in enumerate(images):
            if i >= len(all_xyz) or i >= len(all_K):
                break
            xyz = np.array(all_xyz[i], dtype=np.float32)  # (21, 3)
            K = np.array(all_K[i], dtype=np.float32)      # (3, 3)

            # Project to 2D
            uv_h = (K @ xyz.T).T  # (21, 3)
            uv = uv_h[:, :2] / uv_h[:, 2:3]  # (21, 2)

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]

            # Compute bbox from keypoints
            x_min, y_min = uv.min(axis=0)
            x_max, y_max = uv.max(axis=0)

            # Expand by 20%
            bw = x_max - x_min
            bh = y_max - y_min
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            bw *= 1.2
            bh *= 1.2

            # Normalize to [0, 1]
            nx = cx / iw
            ny = cy / ih
            nw = bw / iw
            nh = bh / ih

            # Clamp
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            nw = min(nw, 2.0 * min(nx, 1.0 - nx))
            nh = min(nh, 2.0 * min(ny, 1.0 - ny))

            entries.append((img_path, (nx, ny, nw, nh)))

    return entries


def load_rhd_bboxes(root: Path) -> list[tuple[Path, tuple[float, float, float, float]]]:
    """Load RHD images and derive bounding boxes from UV annotations."""
    entries = []

    try:
        import pickle
    except ImportError:
        return entries

    for split in ["training", "evaluation"]:
        img_dir = root / split / "color"
        if not img_dir.exists():
            continue

        anno_file = None
        for name in [f"anno_{split}.pickle", f"{split}.pickle", "anno.pickle"]:
            candidate = root / split / name
            if candidate.exists():
                anno_file = candidate
                break
            candidate = root / name
            if candidate.exists():
                anno_file = candidate
                break

        if anno_file is None:
            continue

        with open(anno_file, "rb") as f:
            annos = pickle.load(f)

        for sample_id, anno in annos.items():
            img_path = img_dir / f"{sample_id:05d}.png"
            if not img_path.exists():
                continue

            uv_vis = np.array(anno["uv_vis"], dtype=np.float32)  # (42, 3) — u, v, vis
            # Use both hands' keypoints that are visible
            visible = uv_vis[:, 2] > 0.5
            if visible.sum() < 5:
                continue

            uv = uv_vis[visible, :2]

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]

            # Split into left (0:21) and right (21:42) hands
            for hand_start in [0, 21]:
                hand_uv = uv_vis[hand_start:hand_start + 21]
                hand_vis = hand_uv[:, 2] > 0.5
                if hand_vis.sum() < 5:
                    continue
                pts = hand_uv[hand_vis, :2]

                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                bw = x_max - x_min
                bh = y_max - y_min
                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                bw *= 1.2
                bh *= 1.2

                nx = cx / iw
                ny = cy / ih
                nw = bw / iw
                nh = bh / ih

                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                nw = min(nw, 2.0 * min(nx, 1.0 - nx))
                nh = min(nh, 2.0 * min(ny, 1.0 - ny))

                if nw > 0.01 and nh > 0.01:
                    entries.append((img_path, (nx, ny, nw, nh)))

    return entries


def generate_yolo_dataset(
    entries: list[tuple[Path, tuple[float, float, float, float]]],
    output_dir: Path,
    val_ratio: float = 0.1,
) -> Path:
    """Create YOLO-format dataset directory from entries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(entries)
    val_count = int(len(entries) * val_ratio)
    val_entries = entries[:val_count]
    train_entries = entries[val_count:]

    def write_entries(items, img_dir, lbl_dir):
        for idx, (img_path, bbox) in enumerate(items):
            ext = img_path.suffix
            dst_img = img_dir / f"{idx:06d}{ext}"
            dst_lbl = lbl_dir / f"{idx:06d}.txt"

            # Symlink or copy image
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img_path.resolve())
                except OSError:
                    shutil.copy2(img_path, dst_img)

            # Write YOLO label (class cx cy w h)
            nx, ny, nw, nh = bbox
            with open(dst_lbl, "w") as f:
                f.write(f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

    write_entries(train_entries, train_img_dir, train_lbl_dir)
    write_entries(val_entries, val_img_dir, val_lbl_dir)

    # Write dataset YAML
    yaml_path = output_dir / "dataset.yaml"
    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

names:
  0: hand
"""
    yaml_path.write_text(yaml_content)

    print(f"YOLO dataset: {len(train_entries)} train, {len(val_entries)} val")
    print(f"Dataset YAML: {yaml_path}")
    return yaml_path


def train_yolo(yaml_path: Path, epochs: int, imgsz: int, resume: str | None = None) -> None:
    """Train YOLO11n-detect on the generated dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics not installed. Install with: pip install ultralytics"
        )

    if resume:
        model = YOLO(resume)
        model.train(resume=True)
    else:
        model = YOLO("yolo11n.pt")
        model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=-1,  # auto batch size
            project="runs/detect",
            name="yolo_palm",
            exist_ok=True,
            device=0,
            workers=8,
            amp=True,
            cos_lr=True,
            close_mosaic=10,
            single_cls=True,
        )

    print(f"Training complete. Best weights: runs/detect/yolo_palm/weights/best.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLO11n palm detector")
    ap.add_argument("--freihand", type=str, default=None, help="FreiHAND dataset root")
    ap.add_argument("--rhd", type=str, default=None, help="RHD dataset root")
    ap.add_argument("--output-dir", type=Path, default=Path("data/yolo_palm"),
                    help="Output directory for YOLO-format dataset")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=192,
                    help="Training image size (matches palm detector input)")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--only-gen-data", action="store_true",
                    help="Only generate dataset, skip training")
    ap.add_argument("--resume", type=str, default=None,
                    help="Resume training from checkpoint")
    args = ap.parse_args()

    if args.resume:
        train_yolo(Path(""), 0, args.imgsz, resume=args.resume)
        return

    if not args.freihand and not args.rhd:
        raise SystemExit("Provide at least one of --freihand or --rhd")

    entries = []
    if args.freihand:
        print(f"Loading FreiHAND bboxes from {args.freihand}...", flush=True)
        fh = load_freihand_bboxes(Path(args.freihand))
        print(f"  FreiHAND: {len(fh)} samples", flush=True)
        entries.extend(fh)

    if args.rhd:
        print(f"Loading RHD bboxes from {args.rhd}...", flush=True)
        rhd = load_rhd_bboxes(Path(args.rhd))
        print(f"  RHD: {len(rhd)} samples", flush=True)
        entries.extend(rhd)

    if not entries:
        raise SystemExit("No valid entries found. Check dataset paths.")

    yaml_path = generate_yolo_dataset(entries, args.output_dir, val_ratio=args.val_ratio)

    if not args.only_gen_data:
        train_yolo(yaml_path, args.epochs, args.imgsz)


if __name__ == "__main__":
    main()
