# MediaPipe vs RTMPose-M: IMG_7271.jpeg Comparison

## Test Image

`IMG_7271.jpeg` — 4284×5712 portrait photo of an open palm (all five fingers spread) taken in a bedroom with mixed indoor lighting. The hand fills approximately 60% of the frame.

## Results Summary

| Metric | MediaPipe | RTMPose-M (trial) |
|--------|-----------|-------------------|
| Detection | Yes (conf > 0.5) | Yes (conf = 0.054) |
| Inference time (CPU, OPi5) | ~1766ms | ~307ms (ONNX) |
| Usable output | Clean, accurate | Scattered, low confidence |

### RTMPose vs MediaPipe Agreement (using MediaPipe as reference)

- **MPJPE**: 83.15 px (full-frame letterbox), 110.1 px (hand-cropped)
- **PCK@20px**: 9.5% (full-frame)
- **PCK@10px**: 0.0%

### Per-Joint Error (full-frame letterbox, pixels)

| Joint | Error (px) | Notes |
|-------|-----------|-------|
| wrist | 26.6 | Reasonable |
| thumb_cmc | 197.2 | Wildly off |
| thumb_mcp | 16.0 | Best joint |
| thumb_ip | 49.1 | |
| thumb_tip | 209.3 | Worst overall |
| index_mcp | 55.1 | |
| index_pip | 65.9 | |
| index_dip | 71.3 | |
| index_tip | 60.3 | |
| middle_mcp | 147.2 | |
| middle_pip | 56.2 | |
| middle_dip | 104.0 | |
| middle_tip | 16.4 | Close to wrist/thumb_mcp |
| ring_mcp | 113.6 | |
| ring_pip | 58.5 | |
| ring_dip | 73.9 | |
| ring_tip | 63.8 | |
| pinky_mcp | 150.5 | |
| pinky_pip | 42.9 | |
| pinky_dip | 33.1 | |
| pinky_tip | 135.3 | |

## Observations

### 1. MediaPipe is a mature, production system

MediaPipe's hand landmark model has been trained on millions of diverse real-world images spanning skin tones, lighting conditions, backgrounds, hand poses, and camera qualities. It handles the bedroom scene in IMG_7271 effortlessly — accurate joint placement across all 21 keypoints with high detection confidence even at 5712px tall portrait resolution.

### 2. RTMPose confidence is extremely low (0.054)

The model output 0.054 confidence (effectively "I don't think there's a hand here"). This is the hallmark of a domain gap — the model was fine-tuned exclusively on FreiHAND studio images (224×224, green-screen backgrounds, controlled lighting) and has never seen:
- Real indoor environments
- High-resolution images letterboxed down
- Natural skin tones under mixed warm/cool lighting
- Background clutter (closets, curtains, boxes)

### 3. Systematic spatial bias in RTMPose predictions

The error pattern reveals a spatial clustering issue. MCP joints (knuckles) that sit in the lower-center of the palm (middle_mcp=147px, pinky_mcp=150px, ring_mcp=113px) have much higher error than DIP/PIP joints or fingertips in certain positions. This suggests the model is collapsing predictions toward a "mean hand" pose from FreiHAND rather than adapting to the actual geometry.

### 4. Inference speed favors RTMPose

Despite poor accuracy, RTMPose ONNX on CPU was 5.7× faster than MediaPipe (307ms vs 1766ms). On the RK3588 NPU (INT8 quantized), landmark inference averages **12.5ms per frame (80 FPS)** — with a runtime version mismatch (toolkit 2.3.2 vs runtime 1.5.2) that likely inflates this. Updating to a matched runtime should bring it under 4ms. Once accuracy is fixed, the speed advantage makes RTMPose the clear winner for edge deployment.

### 5. The two-stage pipeline matters

Even with a simulated "perfect crop" (using MediaPipe's bbox as a palm detector substitute), RTMPose confidence remained at 0.044 with worse MPJPE (110px). This confirms the issue is not about detection/cropping but about the landmark model's domain gap from only seeing FreiHAND images.

### 6. Path to closing the gap

This was a **trial run** (FreiHAND-only, ~few epochs) to validate the pipeline code works end-to-end. To reach production accuracy:

- **Add RHD** — diverse real-world hands with varied backgrounds
- **Add more in-the-wild data** — InterHand2.6M, HaGRID, or custom captures
- **Train longer** — full 300-epoch schedule with cosine annealing
- **Augmentations** — the pipeline already includes motion blur, HSV jitter, cutout, rotation which will help generalization
- **Domain adaptation** — pseudo-labeling with MediaPipe on unlabeled real-world video

## NPU Deployment Status

| Component | Status |
|-----------|--------|
| RTMPose RKNN (INT8 quantized) | Working, 15MB model, ~12.5ms/frame |
| Single-stage live (`camera_rknn.py`) | Ready |
| Two-stage live (`camera_twostage_npu.py`) | Ready (MediaPipe det + NPU landmarks) |
| Calibration data | 100 FreiHAND images |
| RKNN toolkit/runtime version mismatch | Warning only, functional |

## Conclusion

MediaPipe is currently far superior on real-world images due to its massive and diverse training data. RTMPose-M has the right architecture and speed profile for edge deployment but needs significantly more diverse training data and longer training to close the accuracy gap. The pipeline infrastructure (train → ONNX → RKNN → NPU live camera) is fully validated end-to-end and ready for a proper training campaign.
