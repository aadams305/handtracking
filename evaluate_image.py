"""Evaluate model on a single image using letterbox preprocessing (matching training).

Uses MediaPipe as ground truth comparison. Preprocessing matches training:
letterbox to 256×256 with gray padding, ImageNet normalization.
"""
import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from handtracking.models.hand_simcc import HandSimCCNet, decode_simcc_soft_argmax, INPUT_SIZE
from handtracking.topology import NUM_HAND_JOINTS, MEDIAPIPE_INDICES_21
from handtracking.geometry import letterbox_image, letterbox_params
from handtracking.dataset import IMAGENET_MEAN, IMAGENET_STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="IMG_7271.jpeg")
    ap.add_argument("--model", type=str, default="checkpoints/hand_simcc.pt")
    ap.add_argument("--out", type=str, default=None, help="Output image path")
    args = ap.parse_args()

    img_path = args.image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. MediaPipe ground truth (full 21 joints)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    results = hands.process(img_rgb)
    
    has_gt = results.multi_hand_landmarks is not None
    gt_letterbox = None
    
    if has_gt:
        kp21 = results.multi_hand_landmarks[0].landmark
        # Map MediaPipe landmarks to letterbox space (matching training)
        lb = letterbox_params(w, h, INPUT_SIZE)
        gt_letterbox = np.zeros((NUM_HAND_JOINTS, 2), dtype=np.float32)
        for i in range(NUM_HAND_JOINTS):
            src_x = kp21[i].x * w
            src_y = kp21[i].y * h
            gt_letterbox[i, 0], gt_letterbox[i, 1] = lb.map_xy_src_to_dst(src_x, src_y)
    
    # 2. Letterbox the image (same as training pipeline)
    lb_img, lb = letterbox_image(img, INPUT_SIZE)
    lb_rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)
    
    # 3. MobileNetV4 inference
    model = HandSimCCNet(width_mult=0.5)
    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        print(f"Warning: {ckpt_path} not found. Using random weights.")
        model.eval()
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        model.eval()
    
    # Normalize (matching training: ImageNet stats)
    inp = lb_rgb.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    inp = (inp - mean) / std
    inp = torch.from_numpy(inp).unsqueeze(0)
    
    with torch.no_grad():
        lx, ly = model(inp)
        pred_coords = decode_simcc_soft_argmax(lx, ly, input_size=INPUT_SIZE)[0].numpy()
    
    # 4. Compute metrics if GT available
    if has_gt:
        errors = np.linalg.norm(pred_coords - gt_letterbox, axis=1)
        mpjpe = errors.mean()
        pck_20 = (errors < 20).mean() * 100
        pck_10 = (errors < 10).mean() * 100
        print(f"MPJPE: {mpjpe:.2f} px")
        print(f"PCK@20px: {pck_20:.1f}%")
        print(f"PCK@10px: {pck_10:.1f}%")
        print(f"Worst joint error: {errors.max():.2f} px (joint {errors.argmax()})")
        
    # 5. Plot side by side on the letterboxed image
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    axs[0].imshow(lb_rgb)
    if has_gt:
        axs[0].scatter(gt_letterbox[:, 0], gt_letterbox[:, 1], c='r', s=20)
        axs[0].set_title(f"MediaPipe GT (letterbox {INPUT_SIZE})")
    else:
        axs[0].set_title(f"No MediaPipe detection (letterbox {INPUT_SIZE})")
    axs[0].axis('off')
    
    axs[1].imshow(lb_rgb)
    axs[1].scatter(pred_coords[:, 0], pred_coords[:, 1], c='lime', s=20)
    title = f"MobileNetV4 SimCC ({INPUT_SIZE})"
    if has_gt:
        title += f"\nMPJPE={mpjpe:.1f}px PCK@20={pck_20:.0f}%"
    axs[1].set_title(title)
    axs[1].axis('off')
    
    out_path = args.out or "eval_comparison.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Saved comparison to {out_path}")
    

if __name__ == "__main__":
    main()
