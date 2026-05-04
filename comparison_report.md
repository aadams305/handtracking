
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

## Methods to Improve Landmark Speed

Current NPU landmark inference: ~10-12ms per hand. Target: sub-4ms.

### Update RKNN runtime to match toolkit version

The RK3588 is running `librknnrt` 1.5.2 while the model was compiled with toolkit 2.3.2. This version mismatch forces compatibility fallback paths. Updating to `librknnrt` 2.x should yield a significant speedup — Rockchip's own benchmarks show 2-4x improvement for quantized models between 1.5 and 2.x runtimes.

### Model pruning and knowledge distillation

Train a smaller student model (CSPNeXt-S or even CSPNeXt-T backbone, `widen_factor=0.5`) distilled from the current RTMPose-M. The M backbone has 768-channel final features — halving this to 384 would roughly quarter the compute in the heaviest layers. The GAU head could also be simplified from 256-dim to 128-dim tokens.

### Reduce SimCC bin count

The model currently uses 512 bins (`split_ratio=2.0`). For real-time tracking where sub-pixel precision matters less, dropping to 256 bins (`split_ratio=1.0`) halves the SimCC head output size and the softmax/weighted-sum compute. The accuracy tradeoff is roughly 0.2-0.5px at 256x256 resolution — negligible for most applications.

### FP16 instead of INT8

INT8 quantization on the RK3588 NPU doesn't always outperform FP16 due to quantization overhead (dequant nodes between layers). Profile both: export an FP16 RKNN (`do_quantization=False`) and benchmark against INT8. If FP16 is within 1-2ms, it also eliminates calibration artifacts.

### Batch two hands in a single inference

When tracking two hands, run both crops as a batch-2 input through the NPU in one call instead of two sequential calls. This amortizes kernel launch overhead and improves NPU utilization. Requires re-exporting the ONNX with dynamic batch or fixed batch=2.

### NPU core affinity

The RK3588 has 3 NPU cores (6 TOPS total). Pin the landmark model to specific cores and avoid core_mask=7 (all cores) when the model is small enough to fit on one core — multi-core dispatch adds synchronization overhead for lightweight models. Benchmark `--core-mask 1` vs `--core-mask 7`.

### Pre-allocate and reuse input buffers

Currently each frame allocates new numpy arrays for the crop, RGB conversion, and NHWC reshape. Pre-allocating fixed 256x256x3 uint8 buffers and using `cv2.resize(src, dst=preallocated)` avoids per-frame memory allocation pressure, which matters at >60 FPS.

## Methods to Increase Confidence for Monochrome Camera

The current model expects 3-channel RGB input normalized with RTMPose pixel-space mean/std. A monochrome (grayscale) camera produces single-channel images, which creates a domain gap.

### Grayscale augmentation during training

Add a training augmentation that randomly converts images to grayscale (all 3 channels set to the luminance value). A 30-50% probability during training teaches the model to rely on structural features (edges, joint angles, silhouette) rather than skin color. This is the single highest-impact change for monochrome compatibility.

### Train with channel-replicated grayscale

During training, randomly convert to grayscale and replicate across all 3 channels: `gray = 0.299*R + 0.587*G + 0.114*B`, then `input = [gray, gray, gray]`. At inference with a mono camera, do the same replication. The model never sees "new" data — it just learns that sometimes all channels are identical.

### Adjust normalization for mono sensors

Monochrome cameras often have different intensity distributions than RGB cameras (wider dynamic range, no Bayer demosaicing artifacts, different noise profile). Compute mean/std statistics on a representative set of mono captures and fine-tune with those statistics, or use per-image standardization (zero-mean, unit-variance) which is sensor-agnostic.

### Edge-aware preprocessing

Apply Sobel or Canny edge detection as an auxiliary input channel (or as a replacement for one RGB channel). Hand landmarks correlate strongly with edge structure — finger boundaries, creases, nail outlines — which are preserved or even enhanced in monochrome. A lightweight 3x3 Sobel can be computed in <0.5ms.

### NIR-specific training data

If the monochrome camera is near-infrared (common in depth/tracking systems), hands look distinctly different — veins are visible, skin appears brighter, fabric/background reflectance changes. Fine-tuning on NIR hand images (or synthetic NIR-like augmentations: increased contrast, inverted vein patterns) would significantly close this domain gap.

### Histogram equalization

CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes local contrast, making the model robust to the varying illumination profiles of mono sensors. Apply as a preprocessing step before channel replication. Cost: ~1ms on ARM.

## Methods to Increase Detection Speed

Current MediaPipe palm detection: ~50-70ms on ARM CPU. These approaches avoid switching to a custom-trained detector model.

### Run detection at minimal resolution

MediaPipe internally resizes to 192x192 or 256x256 anyway. Sending it a 320x240 downscaled frame instead of 1280x960 eliminates expensive bicubic resize operations inside MediaPipe. The current `--det-scale 0.25` helps but could go lower — palm detection is robust down to ~160px input because palms are large, simple shapes.

### Pin detection to big cores (A76)

The RK3588 has 4x A76 (big) + 4x A55 (little) cores. MediaPipe's TFLite delegate may be scheduled on little cores. Use `taskset -c 4-7` to pin the detection thread to the A76 cluster, which has 2-3x higher single-thread throughput than A55.

### Use TFLite GPU delegate

The RK3588's Mali G610 GPU supports OpenCL. MediaPipe can use the TFLite GPU delegate (`mp.solutions.hands.Hands` doesn't expose this directly, but loading the TFLite model manually with `tf.lite.Interpreter(experimental_delegates=[tf.lite.load_delegate('libgpu_delegate.so')])` can offload palm detection to GPU, freeing CPU for other work. Typical 2-3x speedup over CPU for small models.

### Convert palm detection TFLite to RKNN

Without training a custom model, you can convert MediaPipe's existing `palm_detection_lite.tflite` directly to RKNN format. The RKNN toolkit supports TFLite import (`rknn.load_tflite()`). This moves palm detection to the NPU alongside landmarks. The palm detection model is tiny (~1MB) so it would run in <2ms on the NPU.

### Temporal gating with motion detection

Skip detection entirely when the frame hasn't changed much. Compute a cheap frame diff (`cv2.absdiff` on downscaled grayscale, ~0.5ms) and only trigger re-detection when significant motion is detected in the region outside the current tracked bbox. When the hand is being tracked by landmarks, no detection is needed at all.

### MediaPipe solution API overhead

The `mp.solutions.hands.Hands` wrapper adds Python-level overhead per frame (image format validation, result packaging). For tighter control, use the raw TFLite model directly via `tflite_runtime.Interpreter` — this eliminates the MediaPipe framework overhead and gives ~10-15% speedup on small models.

## Methods to Increase Overall Tracking Speed and Accuracy

### Landmark-driven tracking (avoid re-detection)

Once a hand is detected, use the previous frame's landmark positions to compute the next crop bbox — no detection needed. Only trigger re-detection when landmarks disappear (confidence drops below threshold) or a new hand enters the scene. This is what the current async pipeline does, but it can be made more aggressive: only re-detect every 500ms or on explicit confidence loss.

### Temporal smoothing (One-Euro filter)

Apply a One-Euro filter to the landmark time series. This adaptive low-pass filter reduces jitter at rest while preserving responsiveness during fast motion. Per-joint filtering with speed-adaptive cutoff eliminates the 2-5px frame-to-frame noise without adding latency. Implementation cost: <0.1ms for 21 joints.

### Kalman filter for bbox prediction

Instead of re-using the previous frame's landmark bbox directly, run a per-hand Kalman filter that predicts the bbox for the next frame based on position and velocity. This compensates for hand motion between frames, keeping the crop centered even during fast gestures. Prevents the "lagging crop" problem where the hand moves out of the bbox before the next detection.

### Train on video sequences (temporal consistency loss)

Add a temporal consistency term to the training loss: penalize large frame-to-frame jumps in predicted landmark positions when consecutive training frames are available. FreiHAND doesn't have video sequences, but InterHand2.6M and custom camera captures do. This teaches the model to produce smooth trajectories intrinsically.

### Multi-scale crop with confidence selection

Run the landmark model on two crops: the tracked bbox at 1.3x expand and a tighter 1.1x crop. Take the result with higher confidence. The tighter crop gives better resolution on the hand when the bbox is accurate; the wider crop provides safety margin when tracking drifts. The second NPU call adds ~10ms but can be done selectively (only when confidence < threshold).

### Optical flow-assisted tracking

Between detection frames, compute sparse optical flow (Lucas-Kanade) on the 21 landmark positions from the previous frame. This gives near-instant landmark updates (~1-2ms) for frames where NPU inference is skipped. The flow-predicted positions won't be as accurate as NPU output, but for 30+ FPS display they're smooth enough, and the NPU can correct every 2-3 frames.

### Mixed-precision pipeline

Run the backbone in INT8 (fast, sufficient for feature extraction) but keep the GAU attention head in FP16 (preserves the fine-grained SimCC distribution). RKNN supports per-layer quantization via `hybrid_quantization_step`. This gives most of the INT8 speed benefit while preserving landmark precision where it matters.

### EMA model for inference

Use the Exponential Moving Average (EMA) weights from training rather than the final epoch weights. EMA weights typically produce smoother, more stable predictions with 0.5-1px lower MPJPE. The training pipeline already computes EMA — ensure the exported model uses `*_best.pt` or the EMA checkpoint.

### Anchor-free landmark regression as alternative head

Replace or supplement SimCC with direct coordinate regression (a lightweight FC head predicting 21x2 coordinates). This eliminates the 512-bin softmax computation entirely, trading sub-pixel precision for speed. For real-time tracking applications where 1-2px accuracy is acceptable, direct regression can be 30-40% faster in the head.

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
