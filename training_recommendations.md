# Training Recommendations: Next Round

**Current Baseline** (100 epochs, `checkpoints/hand_simcc_best.pt`):
- **MPJPE**: 23.53 px on 256×256
- **PCK@20px**: 47.6%
- **PCK@10px**: 14.3%
- **Worst Joint**: Thumb tip (joint 4) at 60.66 px
- **Dataset**: 94,590 samples (all FreiHAND, 82.5k Right / 12k Left, 0 negatives)
- **Model**: 791K params, width_mult=0.5, 64-channel backbone, 8×8 feature map

---

## Tier 1 — High Impact, Easy to Implement

### 1. Increase `width_mult` from 0.5 → 0.75 or 1.0

This is the single easiest accuracy win. Your backbone only produces **64 channels** at the final stage. The SimCC head does a 1×1 conv from 64 channels to 21×256 = 5,376 channels — that's an **84× expansion** in a single layer, which is too aggressive for the head to learn fine-grained distributions.

| width_mult | Backbone channels | Approx params | Expected impact |
|:---|:---|:---|:---|
| 0.5 (current) | 64 | 791K | Baseline |
| 0.75 | 96 | ~1.6M | Significant accuracy boost |
| 1.0 | 128 | ~2.8M | Best accuracy, still fast on NPU |

```bash
# Train with wider backbone:
--width-mult 0.75
```

> [!TIP]
> Even at width_mult=1.0 (~2.8M params), this model is tiny compared to MediaPipe's ~3M landmark model and will still run fast on the Orange Pi's NPU.

### 2. Reduce `sigma_bins` from 1.0 → 0.75

Your Gaussian bin targets use `sigma_bins=1.0`, meaning the target distribution covers ~3 bins (±1.5σ). For 1:1 pixel-to-bin mapping, this means the model is rewarded even when it's 1-2 pixels off. Sharpening to 0.75 creates tighter supervision:

```bash
--sigma-bins 0.75
```

> [!WARNING]
> Don't go below 0.5 — the targets become too sharp and training becomes unstable.

### 3. Add a Deconv Upsampling Layer Before the SimCC Head

**This is the most impactful architectural change.** Currently, the backbone outputs an **8×8 feature map** (stride 32). The SimCC head mean-pools over these 64 spatial cells, which throws away fine spatial detail. A single transposed convolution upsamples to 16×16 (or 32×32), giving the head 4-16× more spatial resolution to work with.

```python
# In HandSimCCNet.__init__:
self.upsample = nn.Sequential(
    nn.ConvTranspose2d(self.backbone.out_channels, self.backbone.out_channels,
                       kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(self.backbone.out_channels),
    nn.ReLU6(inplace=True),
)

# In forward:
z = self.backbone(x)
z = self.upsample(z)  # 8×8 → 16×16
lx, ly = self.head(z)
```

This adds very few parameters (~16K for a 64-channel deconv) but gives the SimCC head 256 spatial cells instead of 64 to average over, dramatically improving localization.

### 4. Train for 200+ Epochs

100 epochs with cosine annealing likely hasn't fully converged. The loss was still dropping at epoch 100. With ~95K samples and batch_size=32, you get ~2,950 iterations per epoch. Scale up:

```bash
--epochs 200 --warmup-epochs 10
```

### 5. Increase Batch Size

If Colab GPU memory allows, larger batches give more stable gradients and better convergence:

```bash
--batch-size 64    # or even 128 on A100/L4
```

Adjust learning rate proportionally (linear scaling rule):
```bash
--lr 6e-4          # for batch_size 64 (was 3e-4 for batch_size 32)
```

---

## Tier 2 — Moderate Impact, Moderate Effort

### 6. Add Negative Samples (No-Hand Images)

Your dataset has **zero negative samples** (`has_hand_false: 0`). This is why the presence head always outputs 1.0. Fix this by:

1. Downloading a small set of background images (e.g., COCO images without "person" category, random textures, indoor scenes)
2. Adding them to the manifest with `has_hand: false` and dummy keypoints

Target: ~10-15% of dataset as negatives (10K-15K images). This trains the presence head properly AND acts as a regularizer — the model learns that not every input has a hand.

### 7. Add Random Rotation Augmentation (±30°)

Currently you only have 180° rotation (25% chance). Hands appear at many angles. Add continuous random rotation:

```python
def random_rotation(img, kp, max_angle=30, p=0.5):
    if random.random() >= p:
        return img, kp
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img_out = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
    ones = np.ones((kp.shape[0], 1), dtype=np.float32)
    kp_h = np.hstack([kp, ones])
    kp_out = (M @ kp_h.T).T
    kp_out[:, 0] = np.clip(kp_out[:, 0], 0, w - 1)
    kp_out[:, 1] = np.clip(kp_out[:, 1], 0, h - 1)
    return img_out, kp_out.astype(np.float32)
```

### 8. Use Exponential Moving Average (EMA) Weights

EMA of model weights during training produces smoother, more generalizable models. This is standard in pose estimation:

```python
# After each training step:
ema_decay = 0.999
for ema_p, p in zip(ema_model.parameters(), model.parameters()):
    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
```

Save `ema_model` as the best checkpoint instead of the raw training model.

### 9. Increase Coordinate Loss Weight

Currently `coord_loss_weight=0.5`. The L1 coordinate loss directly optimizes the metric you care about (pixel distance). Try increasing it:

```bash
--coord-loss-weight 1.0    # or even 2.0
```

The SimCC CE loss teaches the distribution shape; the L1 loss teaches exact pixel position. Given your MPJPE is 23px, the L1 component needs more influence.

### 10. Handedness Imbalance Fix

Your dataset is **87% Right / 13% Left**. This skew means the handedness head is biased toward "Right." Two fixes:

- **Already partially addressed** by horizontal flip augmentation (which swaps L↔R)
- **Additional fix**: Use weighted BCE loss for handedness — upweight Left examples by ~3-4× in the loss computation

---

## Tier 3 — High Impact, Higher Effort

### 11. Multi-Scale Feature Fusion (FPN-lite)

Instead of only using the last backbone stage (8×8), fuse features from stage3 (16×16) and stage4 (8×8). This gives the SimCC head access to both high-resolution spatial info and high-level semantic info:

```python
# In HandSimCCNet:
self.lateral3 = nn.Conv2d(stage3_channels, 64, 1)
self.lateral4 = nn.Conv2d(stage4_channels, 64, 1)

# In forward:
f3 = self.lateral3(self.backbone.stage3_out)  # 16×16
f4 = self.lateral4(self.backbone.stage4_out)  # 8×8
f4_up = F.interpolate(f4, size=f3.shape[2:], mode='bilinear')
z = f3 + f4_up  # 16×16 fused features
```

### 12. Two-Stage Pipeline (Palm Detection + Landmark)

This is what MediaPipe does and is the single biggest architectural improvement possible. Instead of running SimCC on the full letterboxed image, first detect the hand bounding box, crop tightly, then run SimCC on the aligned crop.

- **Stage 1**: Lightweight palm detector (YOLO-nano or BlazePalm)
- **Stage 2**: Your SimCC model on the tightly cropped + aligned hand

This eliminates the biggest source of error — the model currently has to simultaneously locate the hand in the full image AND regress precise landmarks. Separating these tasks is dramatically easier.

### 13. Add More Training Data

FreiHAND (95K distilled) is a single synthetic dataset. Diversity matters:

| Dataset | Size | Type | Notes |
|:---|:---|:---|:---|
| FreiHAND | 130K | Synthetic | Current source |
| RHD | 44K | Synthetic | Different rendering engine |
| OneHand10K | 10K | Real | Natural images |
| HaGRID | 500K+ | Real | Gesture recognition |
| InterHand2.6M | 2.6M | Real | Multi-hand, studio capture |

Distill through MediaPipe teacher → merge manifests → train.

---

## Recommended Training Configuration

For your next Colab run, I recommend combining Tier 1 changes:

```bash
python3 -m handtracking.train \
  --manifest data/distilled/manifest.jsonl \
  --epochs 200 \
  --batch-size 64 \
  --lr 5e-4 \
  --warmup-epochs 10 \
  --width-mult 0.75 \
  --sigma-bins 0.75 \
  --coord-loss-weight 1.0 \
  --grad-clip 1.0 \
  --num-workers 4 \
  --out checkpoints/hand_simcc.pt
```

**Expected improvements with Tier 1 alone:**
- MPJPE: 23.5px → **~12-15px** (wider backbone + deconv upsample + tighter sigma + more epochs)
- PCK@20px: 47.6% → **~70-80%**
- PCK@10px: 14.3% → **~35-45%**

**With Tier 1 + Tier 2:**
- MPJPE: → **~8-12px**
- PCK@20px: → **~80-90%**

**With Tier 1 + 2 + 3 (two-stage):**
- MPJPE: → **~3-5px** (approaching MediaPipe quality)
