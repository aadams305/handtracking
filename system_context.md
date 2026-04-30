# System Context: MobileNetV4 SimCC Hand Tracking System

## 1. Core Project Objectives
**Primary Goal:** Build a highly accurate, extremely low-latency (sub-4ms) 21-point hand tracking system optimized for Edge AI, specifically the Orange Pi 5 Max (RK3588) and Android NPUs.
**Current Target:** Achieve MediaPipe-level sub-10px Mean Per Joint Position Error (MPJPE) on edge hardware. We have improved from an initial ~50px error to ~23.5px error through single-stage optimization, but we are now transitioning to a two-stage (Detection + Regression) architecture to close the final accuracy gap.

## 2. The ML/Software Stack
*   **Training Framework:** PyTorch (with Python).
*   **Inference Stack:** ONNX Runtime (currently via CPUExecutionProvider). 
*   **Planned NPU Stack:** Rockchip's `rknn-toolkit2` and `rknn.api` for hardware acceleration on the Orange Pi 5.
*   **Teacher/Ground Truth:** MediaPipe Hands (used purely offline to distill ground-truth annotations).
*   **Model Architecture:** 
    *   **Backbone:** Custom `MobileNetV4ConvSmall` (stride 32).
    *   **Heads:** `SimCC` (1D heatmap classification) + `PresenceHandednessHead` (GAP + Fully Connected).

## 3. Data Pipeline & Processing
*   **Distillation:** We use a `distill_freihand.py` script to run images through MediaPipe, capturing the 21 `(X, Y)` landmarks, Left/Right handedness strings, and hand presence logic, saving them to a JSONL manifest.
*   **Preprocessing:** Images are letterboxed (padded with gray value 114) to maintain aspect ratio, resized to `256x256`, and normalized using ImageNet Mean/Std.
*   **Augmentations:** Highly robust HSV color jitter, motion blur, random continuous rotation (±30°), random scale/crop, and horizontal flipping. *Crucial logic:* When horizontally flipping an image, the Left/Right handedness label is explicitly inverted.

## 4. Key Algorithmic Decisions
*   **SimCC Representation:** Instead of standard 2D heatmaps (which are computationally heavy), we use SimCC to predict independent 1D probability distributions for the X and Y axes across 256 bins.
*   **Confidence Gating Heuristic:** Because our dataset currently lacks negative (no-hand) samples, the trained Presence head incorrectly outputs 1.0 continuously. To fix this, we implemented `simcc_confidence()`: a proxy algorithm that calculates the "peakedness" of the SimCC distribution. Real hands produce sharp peaks (~0.04), while background noise produces flat distributions (~0.007). We use a 0.02 threshold to gate visualization.
*   **Auxiliary Fingertip Loss:** We apply a `tip_weight=2.0` multiplier inside the L1 Coordinate auxiliary loss specifically for the 5 fingertip indices. This solved a major regression issue where the thumb and ring finger tips were drastically off-target.

## 5. Current Code State
*   **`models/hand_simcc.py`:** Contains the 4-output architecture: `lx`, `ly`, `presence_logit`, `handedness_logit`. **Recent Change:** Added a Deconv Upsampling layer (16K params) that upscales the backbone's 8x8 feature map to 16x16 before feeding it to the SimCC head, yielding 4x spatial resolution.
*   **`losses.py`:** `SimCCGaussianSoftCELoss` combining 1D Bin Cross Entropy (shape), L1 Coordinate Regression (precision), and two BCE losses for Presence and Handedness.
*   **`dataset.py`:** Yields 4-item tuples: `(image, keypoints, has_hand, handedness)`. Left=0.0, Right=1.0.
*   **`camera_student.py` / `live_camera.py`:** OpenCV inference loops that decode the SimCC soft-argmax, apply the peakedness confidence threshold, and render the skeleton dynamically.

## 6. Iterative Improvements (Do Not Revert)
*   **Cosine Annealing + Warmup:** Replaced standard exponential decay. This significantly reduced terminal MPJPE.
*   **Fingertip Weighting:** Do not remove the auxiliary L1 coordinate loss. It is the only thing keeping the high-DoF thumb joint constrained.
*   **SimCC Confidence Peak:** Do not use the `presence` sigmoid for gating until negative samples are formally introduced to the dataset. Stick to `simcc_confidence()`.

## 7. Pending Hurdles
*   **Architectural Shift (Two-Stage):** The single-stage approach (analyzing a full 256x256 frame) has hit its accuracy ceiling. We need to implement BlazePalm (or YOLO) to detect the palm and crop the image *before* passing it to the SimCC landmark model.
*   **Dataset Expansion:** We need to integrate RHD and OneHand10K datasets into our distillation pipeline to diversify beyond FreiHAND.
*   **Hardware Acceleration:** We need to build the conversion and inference pipeline for Rockchip's RKNN NPU format so we can move off the CPU execution provider.

---

# Prompt / Instructions for Next LLM Session

*Copy and paste the entire System Context above, followed by these specific instructions for the AI:*

**Objective:**
We are continuing the optimization of our Edge Handtracking project based on our latest architectural review. Please execute the following sequence of tasks. Do not proceed to training until the code implementations are confirmed.

**Task 1: Model & Loss Enhancements**
*   Increase the `width_mult` default from 0.5 to 0.75 across the configuration to give the SimCC head a 96-channel feature map.
*   Implement Exponential Moving Average (EMA) for the model weights during the training loop.
*   Increase the `coord_loss_weight` to 1.0.
*   Keep training capped at 100 epochs (do not extend to 200).

**Task 2: Architectural Overhaul (Two-Stage Pipeline)**
*   Implement a BlazePalm-based detection mechanism to transition us from a single-stage model to a two-stage crop-and-regress pipeline. The pipeline should find the hand, extract a tight, oriented bounding box, and feed that crop to the SimCC model. 

**Task 3: Dataset Expansion**
*   Update the dataset/distillation pipeline scripts to support the ingestion of the RHD and OneHand10K datasets alongside FreiHAND.
*   *(Note: Do not implement negative "no-hand" background samples during this sprint. We will handle that in a future iteration).*

**Task 4: NPU Pipeline Setup**
*   Create the necessary Python scripts (`export_rknn.py` and `camera_rknn.py`) utilizing `rknn-toolkit2` to convert our PyTorch/ONNX models and run inference directly on the Orange Pi 5 RK3588 NPU.

**Task 5: Deployment & Execution**
*   If you see opportunities to add specific tracking metrics/logging to the training loop that will help us diagnose MPJPE bottlenecks, implement them. 
*   Update `colab.txt` with all new parameters, architecture requirements, and run commands so I can seamlessly copy-paste the training block into Google Colab.
*   Execute the distillation script locally to prepare the new multi-dataset manifest, and ensure everything is prepped for the Colab transfer.
