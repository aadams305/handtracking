import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from handtracking.models.hand_simcc import HandSimCCNet, decode_simcc_soft_argmax
from handtracking.topology import NUM_HAND_JOINTS, MEDIAPIPE_INDICES_10

def get_crop_with_padding(img, kp21, pad_ratio=0.5):
    h, w = img.shape[:2]
    xs = [p.x * w for p in kp21]
    ys = [p.y * h for p in kp21]
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    bw = xmax - xmin
    bh = ymax - ymin
    
    # Square crop
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    size = max(bw, bh) * (1.0 + pad_ratio)
    
    # Bounding box
    x1, y1 = int(cx - size / 2), int(cy - size / 2)
    x2, y2 = int(cx + size / 2), int(cy + size / 2)
    
    # Constrain to image bounds by shifting or tightly padding
    x1_safe, y1_safe = max(0, x1), max(0, y1)
    x2_safe, y2_safe = min(w, x2), min(h, y2)
    
    # We will just warp affine the region so out-of-bounds gets padded automatically
    src_pts = np.float32([[x1, y1], [x1, y2], [x2, y1]])
    dst_pts = np.float32([[0, 0], [0, 160], [160, 0]])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    
    crop = cv2.warpAffine(img, M, (160, 160), borderValue=(114, 114, 114))
    
    # We must map MediaPipe's points (which are in 0-1 scale of original) into the 160x160 crop coordinates
    mp_160 = []
    for i in MEDIAPIPE_INDICES_10:
        orig_x, orig_y = kp21[i].x * w, kp21[i].y * h
        # Transform via affine matrix
        pts = np.array([[[orig_x, orig_y]]], dtype=np.float32)
        trans_pt = cv2.transform(pts, M)[0][0]
        mp_160.append(trans_pt)
        
    return crop, np.array(mp_160)

def main():
    img_path = "IMG_7271.jpeg"
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. MediaPipe Tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        print("MediaPipe found no hands!")
        return
        
    kp21 = results.multi_hand_landmarks[0].landmark
    
    # 2. Extract tight square crop centered on MediaPipe's detection
    crop_160, mp_target_160 = get_crop_with_padding(img, kp21, pad_ratio=0.5)
    crop_rgb = cv2.cvtColor(crop_160, cv2.COLOR_BGR2RGB)
    
    # 3. MobileNetV4 Tracking on the *exact same crop*
    model = HandSimCCNet(width_mult=0.5)
    ckpt = torch.load("checkpoints/hand_simcc.pt", map_location="cpu", weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    # Normalize PyTorch style
    inp = crop_rgb.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    inp = (inp - mean) / std
    inp = torch.from_numpy(inp).unsqueeze(0)
    
    with torch.no_grad():
        lx, ly = model(inp)
        pred_coords = decode_simcc_soft_argmax(lx, ly, input_size=160)[0].numpy()
        
    # 4. Plot side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Subplot 1: MediaPipe Output (11 joints)
    axs[0].imshow(crop_rgb)
    axs[0].scatter(mp_target_160[:, 0], mp_target_160[:, 1], c='r', s=20)
    for i, pt in enumerate(mp_target_160):
        axs[0].text(pt[0], pt[1], str(i), color='red', fontsize=8)
    axs[0].set_title("MediaPipe (Ground Truth Target)")
    axs[0].axis('off')
    
    # Subplot 2: MobileNet Output
    axs[1].imshow(crop_rgb)
    axs[1].scatter(pred_coords[:, 0], pred_coords[:, 1], c='lime', s=20)
    for i, pt in enumerate(pred_coords):
        axs[1].text(pt[0], pt[1], str(i), color='lime', fontsize=8)
    axs[1].set_title("MobileNetV4 SimCC (Predicted)")
    axs[1].axis('off')
    
    out_path = "/home/aidan/.gemini/antigravity/brain/75ebae8f-2e05-4ae4-939a-a4a08439d809/artifacts/comparison.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Saved exact comparison to artifacts!")
    
if __name__ == "__main__":
    main()
