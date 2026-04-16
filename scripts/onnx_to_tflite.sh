#!/usr/bin/env bash
# Android path: convert ONNX to TFLite (requires tensorflow or ai-edge-torch in your env).
set -euo pipefail
ONNX="${1:-../models/hand_simcc.onnx}"
OUT="${2:-../models/hand_simcc.tflite}"
echo "Placeholder: use tensorflow lite converter or ai-edge-torch to produce ${OUT} from ${ONNX}"
echo "Example: python -m ai_edge_torch.convert --help (see TensorFlow docs for your version)"
exit 0
