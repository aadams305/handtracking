import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from handtracking.models.hand_simcc import HandSimCCNet, decode_simcc_soft_argmax
from handtracking.topology import NUM_HAND_JOINTS, MEDIAPIPE_INDICES_21

def get_crop_with_padding(img, kp21, target_size=256, pad_ratio=0.5):
    h, w = img.shape[:2]
    xs = [p.x * w for p in kp21]
    ys = [p.y * h for p in kp21]
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    bw = xmax - xmin
    bh = ymax - ymin
    
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    size = max(bw, bh) * (1.0 + pad_ratio)
    
    x1, y1 = cx - size / 2, cy - size / 2
    x2, y2 = cx + size / 2, cy + size / 2
    
    src_pts = np.float32([[x1, y1], [x1, y2], [x2, y1]])
    dst_pts = np.float32([[0, 0], [0, target_size], [target_size, 0]])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    
    crop = cv2.warpAffine(img, M, (target_size, target_size), borderValue=(114, 114, 114))
    
    mp_target = []
    for i in MEDIAPIPE_INDICES_21:
        orig_x, orig_y = kp21[i].x * w, kp21[i].y * h
        pts = np.array([[[orig_x, orig_y]]], dtype=np.float32)
        trans_pt = cv2.transform(pts, M)[0][0]
        mp_target.append(trans_pt)
        
    return crop, np.array(mp_target), M

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="IMG_7271.jpeg")
    ap.add_argument("--model", type=str, default="checkpoints/hand_simcc.pt")
    args = ap.parse_args()

    img_path = args.image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Image not found")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. MediaPipe Detector Stage
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.1)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        print("MediaPipe Palm Detector found no hands! Falling back to full image crop (FreiHAND style).")
        crop_256 = cv2.resize(img, (256, 256))
        gt_256 = np.array([])
    else:
        kp21 = results.multi_hand_landmarks[0].landmark
        # 2. Extract perfectly isolated patch
        crop_256, gt_256, _ = get_crop_with_padding(img, kp21, target_size=256, pad_ratio=0.5)
        
    crop_rgb = cv2.cvtColor(crop_256, cv2.COLOR_BGR2RGB)
    
    # 3. Native PyTorch MobileNetV4 Tracking
    model = HandSimCCNet(width_mult=0.5)
    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        print(f"Warning: {ckpt_path} not found. Please train the model using 'python3 -m handtracking.train' first.")
        # Proceed with random weights just to show structure works
        model.eval()
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        model.eval()
    
    inp = crop_rgb.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    inp = (inp - mean) / std
    inp = torch.from_numpy(inp).unsqueeze(0)
    
    with torch.no_grad():
        lx, ly = model(inp)
        pred_coords = decode_simcc_soft_argmax(lx, ly, input_size=256)[0].numpy()
        
    # Plotting Evaluator
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    axs[0].imshow(crop_rgb)
    if gt_256.size > 0:
        axs[0].scatter(gt_256[:, 0], gt_256[:, 1], c='r', s=20)
        axs[0].set_title("MediaPipe Detector Target")
    else:
        axs[0].set_title("MediaPipe Failed (Full Image Fallback)")
    axs[0].axis('off')
    
    axs[1].imshow(crop_rgb)
    axs[1].scatter(pred_coords[:, 0], pred_coords[:, 1], c='lime', s=20)
    axs[1].set_title(f"MobileNet-V4 HighRes (256)")
    axs[1].axis('off')
    
    plt.savefig("/home/aidan/.gemini/antigravity/brain/bca659aa-0337-4138-a674-af4bda8f5ef4/final_eval.png", bbox_inches='tight', dpi=150)
    print("Static PyTorch Evaluation rendering saved!")
    
if __name__ == "__main__":
    main()
