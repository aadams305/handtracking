#!/usr/bin/env bash
# Phase 3: ONNX -> NCNN param/bin, then ncnnoptimize. Requires onnx2ncnn and ncnnoptimize on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="${1:-$ROOT/models/hand_simcc.onnx}"
OUTDIR="${2:-$ROOT/models/ncnn}"
mkdir -p "$OUTDIR"
BASE="$OUTDIR/hand_simcc"
if ! command -v onnx2ncnn >/dev/null 2>&1; then
  echo "onnx2ncnn not found; build NCNN tools or skip this step." >&2
  exit 1
fi
onnx2ncnn "$ONNX" "${BASE}.param" "${BASE}.bin"
if command -v ncnnoptimize >/dev/null 2>&1; then
  ncnnoptimize "${BASE}.param" "${BASE}.bin" "${BASE}.opt.param" "${BASE}.opt.bin" 0
  SZ=$(($(wc -c < "${BASE}.opt.param") + $(wc -c < "${BASE}.opt.bin")))
  echo "ncnnoptimize output: ${BASE}.opt.param + .opt.bin total_bytes=${SZ}"
else
  SZ=$(($(wc -c < "${BASE}.param") + $(wc -c < "${BASE}.bin")))
  echo "ncnn (no optimize): total_bytes=${SZ}"
fi
python3 -c "import sys; sz=int(sys.argv[1]); print('under_5mb', sz < 5*1024*1024)" "$SZ"
