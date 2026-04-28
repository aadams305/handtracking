# Notes: MediaPipe ground truth vs SimCC student (PyTorch / ONNX)

This document compares **MediaPipe** (treated as **ground truth** for visualization and discussion) against our **MobileNetV4 + SimCC** student in **PyTorch** and **ONNX**, using **`IMG_7271.jpeg`** and the same **256×256 letterbox** pipeline as training. **Camera distortion is assumed negligible** (single pinhole-style phone image; we do not undistort or pass intrinsics into the student).

## Setup

| Role | Component | Notes |
|------|-----------|--------|
| Ground truth | `MediaPipeTeacher` (Tasks Hand Landmarker) | 21 joints mapped from full image pixels into **letterbox space** with the same `map_xy_src_to_dst` logic used in `distill_freihand`. |
| Student | `HandSimCCNet` + `decode_simcc_soft_argmax` | PyTorch checkpoint `checkpoints/hand_simcc.pt`. |
| Student (deployed) | ONNX `models/hand_simcc.onnx` + ONNX Runtime | Same preprocessing as PyTorch (`bgr_letterbox_to_nchw_batch` vs `normalize_bgr_tensor`). |
| Input | Letterbox `INPUT_SIZE` (= 256) | Matches manifest / training; **not** a tight crop around the hand unless the full frame already fills the square. |

**Important:** “Ground truth” here means **the teacher we distilled from**, not independent motion-capture truth. Any systematic bias in MediaPipe on this pose propagates into labels; the student can only approximate that teacher on held-out poses.

## Quantitative metrics (`IMG_7271.jpeg`, letterbox 256×256 space)

Coordinates are in **pixels on the letterboxed square** (same space as `comparison_mp_student.png`). **Ground truth** = MediaPipe joints mapped with `letterbox_image` + `map_xy_src_to_dst`. **Prediction** = `HandSimCCNet` + `decode_simcc_soft_argmax` (PyTorch; ONNX matches to numerical noise).

| Metric | Value | Meaning |
|--------|------:|---------|
| **MPJPE** (mean per-joint L2) | **18.70 px** | Average Euclidean error per joint vs MediaPipe. |
| **Worst joint** | **thumb_tip** (**45.71 px**) | Largest L2 error; thumb chain errors are large on this frame. |
| **Best joints** | **ring_mcp** (**1.16 px**), **middle_mcp** (**4.69 px**) | MCPs near the palm are easiest here. |
| **PCK @ 20 px** | **57.14%** | Fraction of joints with error **≤ 20 px** (≈ **7.8%** of letterbox width 256). |
| **PCK @ 10 px** | **23.81%** | Stricter “very accurate” joint fraction. |
| **NME (palm scale)** | **25.3%** | MPJPE divided by **wrist→middle_mcp** distance in GT (**73.9 px**): mean error as a fraction of a **palm-sized ruler**. |
| **NME (hand span)** | **12.1%** | MPJPE divided by **max pairwise GT distance** (**154.4 px**): mean error vs **full hand extent**. |

There is **no single canonical “accuracy %”** for 2D landmarks; the table uses **PCK** (thresholded hit rate) and **NME** (error normalized by pose scale). For a one-line summary: **~58% of joints within 20 px** of MediaPipe on this image, with **mean error ~12% of hand span**.

### Per-joint L2 error (px)

| Joint | L2 | Joint | L2 | Joint | L2 |
|-------|---:|-------|---:|-------|---:|
| wrist | 21.0 | thumb_cmc | 20.5 | thumb_mcp | 29.3 |
| thumb_ip | 37.8 | **thumb_tip** | **45.7** | index_mcp | 12.0 |
| index_pip | 8.8 | index_dip | 10.1 | index_tip | 16.5 |
| middle_mcp | 4.7 | middle_pip | 6.7 | middle_dip | 12.7 |
| middle_tip | 21.5 | ring_mcp | 1.2 | ring_pip | 11.4 |
| ring_dip | 20.0 | ring_tip | 30.1 | pinky_mcp | 5.3 |
| pinky_pip | 19.2 | pinky_dip | 26.0 | pinky_tip | 32.2 |

*Recompute anytime with the same preprocessing as `handtracking/compare_mp_student.py` (letterbox `INPUT_SIZE`, then decode).*

## Observed differences (`comparison_mp_student.png`)

On **`IMG_7271.jpeg`** (open palm, portrait framing, gray letterbox bars):

1. **MediaPipe (left)** — Landmarks sit cleanly on anatomical joints and fingertips; the skeleton follows the visible hand closely.

2. **SimCC PyTorch (center)** — The global pose is recognizable, but several joints **lag toward the palm** relative to MediaPipe: **fingertips** (index / middle / ring especially) appear **short**—coordinates sit **distal–proximal** along the finger instead of on the tips. The **thumb** chain is slightly shifted toward the palm center.

3. **SimCC ONNX (right)** — Matches PyTorch numerically (**`max|ONNX − PyTorch|≈ 0`** in the plot title). The cyan overlay is visually the same as the green one; export is behaving as expected.

So: **export fidelity is excellent**; **accuracy vs MediaPipe** still shows a **systematic “finger length” / tip placement bias** on this example—the **quantitative table** aligns with that (**thumb_tip** and **pinky_tip** / **ring_tip** among the largest errors; **ring_mcp** nearly spot-on).

## Live camera: why MobileNetV4 + SimCC ≈ MediaPipe FPS (~24)

On USB capture, **MobileNetV4 + SimCC** has been observed at **~24 FPS**, similar to **MediaPipe** in the same pipeline—not the large gap one might expect from “lighter backbone” intuition alone.

**Why they can look the same speed (not an exhaustive profile):**

1. **Different workloads** — MediaPipe in live mode runs a **full hand stack** (detection, tracking, landmark model, rendering path). Our student run is **only** letterbox → **one** conv stack → **SimCC head** → softmax/decode. Faster *per forward* does not help if the **camera + decode + OpenCV `imshow`** pipeline dominates wall time.

2. **Shared bottlenecks** — **MJPEG decode** (`cv2.VideoCapture` + **1280×960**), **USB bandwidth**, **single-threaded Python loop**, and **display refresh** often cap FPS **before** pure conv FLOPs differ meaningfully.

3. **SimCC decode cost** — For each frame we form **softmax over 256 bins × 2 axes × 21 joints** and weighted sums. That is cheap vs a giant CNN but **not free**, especially on **CPU** ONNX Runtime.

4. **CPU vs GPU** — If both paths use **CPU** (default ORT `CPUExecutionProvider`, MediaPipe XNNPACK on CPU), they **fight for the same cores** with similar order-of-magnitude cost per frame at this resolution.

5. **`infer-every-N`** — Scripts often run the student **every N-th frame** and reuse last keypoints, which **raises apparent FPS** but **does not** compare “full 30 Hz inference” for both stacks unless MediaPipe is throttled the same way.

6. **Expectations from FLOPs alone** — MobileNet is efficient, but **256×256** input and **21×256×2** logits still imply non-trivial memory traffic; MediaPipe’s graph is also heavily optimized. **Wall-clock parity** is plausible without implying the architectures are equivalent.

**To actually measure “model-only” speed:** time **just** `session.run` / `net(inp)` in a tight loop with **cached tensors**, no camera; compare to MediaPipe `detect` on the same fixed crop. That separates **inference** from **capture + UI**.

## Why ground truth and the student can disagree

Assuming **no lens distortion**, remaining gaps are mostly **modeling, supervision, and decoding**—not calibration.

1. **Teacher ≠ physical truth**  
   MediaPipe is a strong prior but still a **learned estimator**. We distill its outputs; the student fits **that** distribution, not true 3D joint positions.

2. **SimCC is per-axis classification + soft-argmax decode**  
   Predictions are **expectations over 256 bins** per axis. That smooths predictions and can **pull extreme coordinates (tips) slightly inward** compared to a detector that directly regresses image coordinates, especially if training logits are soft or peak away from the true bin.

3. **Training coverage vs this image**  
   FreiHAND + letterboxed crops emphasize certain **scales, poses, and backgrounds**. A **domestic indoor palm-forward** frame may sit on the **edge of the manifold** the student saw most often, so tips and thumb are common failure modes.

4. **Augmentations and label noise**  
   Training uses **motion blur, rotate-180, cutout**, etc. That improves robustness but **blurs fine tip supervision**; the network may learn a **conservative** (more proximal) tip to minimize loss under noisy labels.

5. **Wrist / root anchoring**  
   Small errors in early joints **amplify** along each finger in skeleton space. A small palm error can look like “short fingers” when edges are drawn between predicted points.

6. **Letterbox vs “tight crop” evaluation**  
   This comparison intentionally uses **full-frame letterbox** (aligned with training). Other scripts historically used **affine crops around MediaPipe**; metrics and errors **change** with preprocessing. Here we only discuss **letterbox** behavior.

## What to explore to improve accuracy

Roughly ordered by expected leverage:

1. **Loss / targets**  
   - Tighter **Gaussian σ** in bin space, or auxiliary **direct regression** on `(x,y)` after SimCC.  
   - **Wing / L1** on decoded coords *in addition to* distribution loss (careful balance—previous Adaptive Wing was removed for the SimCC rewrite but a **small** coordinate term might help tips).

2. **Decode**  
   - Compare **soft-argmax** vs **hard argmax** vs **integral regression** variants for tip stability.  
   - Slightly **higher input resolution** or more bins (breaks 1:1 pixel policy—tradeoff).

3. **Data**  
   - More **in-the-wild** palms matching user cameras; hard-mining frames where **MP confidence is high** but student loss is high.  
   - **Consistency / temporal** losses if moving to video.

4. **Architecture**  
   - **Higher capacity** (`width_mult`), small **FPN** or extra head convs for spatial precision.  
   - **Spatial attention** in the SimCC head instead of plain mean over `H×W` (current head averages spatial locations—documented in code).

5. **Teacher alignment**  
   - If product goal is “match MediaPipe on phone,” keep MP as teacher but **filter** distill samples by **handedness / confidence** and **balance poses** so tips are not under-represented.

6. **Evaluation protocol**  
   - Report **MPJPE / PCK** on a **held-out** set with identical letterbox, plus **per-joint** error (tips vs MCP).

## ONNX vs PyTorch

For the same letterboxed tensor, **ONNX Runtime** reproduces **PyTorch** decoded coordinates to **floating-point noise** levels. Deployment should focus on **pre/post** (letterbox, decode) consistency rather than export drift.

---

## Repo changes in this session (summary)

Work touched multiple areas; consolidated list:

### SimCC model & training

- **`handtracking/models/hand_simcc.py`**: `INPUT_SIZE=256`, `NUM_BINS=256`, **conv-only SimCC head** (`conv_x` / `conv_y`, spatial mean → `[B,21,256]`), `HandSimCCNet` uses `MobileNetV4ConvSmall` forward; decode / bin helpers updated; `__main__` smoke test.
- **`handtracking/losses.py`**: **Gaussian soft-target cross-entropy** (`SimCCGaussianSoftCELoss`, `gaussian_bin_targets_*`); **removed** Adaptive Wing and old SimCC CE helpers.
- **`handtracking/train.py`**: Trains with **`SimCCGaussianSoftCELoss`**, no `decode_fn` in the loss.

### Data & runtime alignment (256 / 21)

- **`handtracking/geometry.py`**, **`handtracking/dataset_manifest.py`**: Default **`dst=256`**, docstrings for 21 joints.
- **`handtracking/dataset.py`**: Requires manifest **`letterbox.dst == INPUT_SIZE`**; letterbox at `INPUT_SIZE`.
- **`handtracking/simcc_numpy.py`**: Aligns with PyTorch decode; **`bgr_letterbox_to_nchw_batch`**; imports constants from `hand_simcc`.
- **`handtracking/live_camera.py`**: **256** letterbox, **21** landmarks, **`draw_hand_21`**, ONNX path uses numpy helper; **`keypoints_square_to_frame`**.
- **`handtracking/viz.py`**: **`EDGES_21`**, **`draw_hand_21`** (kept 10-point viz for legacy).
- **`handtracking/verify_forward.py`**, **`handtracking/export_onnx.py`**, **`handtracking/bench_torch.py`**: **256×256** I/O and shape asserts.
- **`handtracking/teacher.py`**: Docstring / MediaPipe init **progress prints** (with `flush=True`).
- **`handtracking/distill_freihand.py`**: **Progress logging**, **`list` vs `sorted` glob** for large FreiHAND sets, tqdm on stderr; messages before MediaPipe and distill loop.
- **`handtracking/models/mobilenet_v4_conv_small.py`**: Comment update for 256×256 SimCC.

### C++

- **`cpp/main.cpp`**, **`cpp/main_ros2.cpp`**: **`kSize=256`**, **`kJoints=21`**, **`kBins=256`**, letterbox variable naming; **`#include "net.h"`** for ncnn.

### Tests & Colab

- **`test_simcc_overfit.py`**, **`test_onnx.py`**, **`test_torch_onnx.py`**: Updated for 256 / 256 bins / 21 joints.
- **`colab.txt`**: Fresh-run flow, **`pip install -e . --no-deps`**, GPU device fix in sanity cell, distill/train **`python -u`**, backup guards, **`IMG_7271`** defaults where relevant.

### ONNX & comparison (this comparison workflow)

- **`models/hand_simcc.onnx`**: Produced via **`python -m handtracking.export_onnx`** (gitignored as `*.onnx`).
- **`handtracking/compare_mp_student.py`**: Letterbox-aligned **MediaPipe vs PyTorch vs ONNX** figure; default image search prefers **`IMG_7271.jpeg`** at repo root.
- **`data/demo_images/.gitkeep`**: Placeholder path for optional assets.
- **`comparison_mp_student.png`**: Generated artifact from **`IMG_7271.jpeg`** (typically gitignored or not committed—treat as local output).

### Live USB camera (student only)

- **`camera_student.py`**: V4L2 + MJPG open path inspired by **`cameraLogi.py`**; **ONNX or PyTorch only** (no MediaPipe teacher); letterbox + overlay.
- **`camera.py`**: Docstring points to **`camera_student.py`** for SimCC live testing.

### Documentation

- **`NOTES.md`** (this file): Evaluation assumptions, qualitative + **quantitative** metrics, FPS discussion, and session changelog.

---

*Generated for internal tracking; update as the model and eval pipeline evolve.*
