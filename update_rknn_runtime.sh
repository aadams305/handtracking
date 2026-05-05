#!/bin/bash
# Update RKNN runtime (librknnrt.so) to version 2.3.2 on Orange Pi 5 / RK3588.
# This matches the rknn-toolkit2 v2.3.2 used to compile the RKNN models.
#
# Usage:
#   sudo bash update_rknn_runtime.sh
#
# The old runtime is backed up before replacement.

set -euo pipefail

VERSION="2.3.2"
ARCH="aarch64"
LIB_NAME="librknnrt.so"
INSTALL_DIR="/usr/lib"
TMP_DIR="/tmp/rknn_update_$$"

echo "=== RKNN Runtime Update to v${VERSION} ==="

# Check architecture
MACHINE=$(uname -m)
if [ "$MACHINE" != "aarch64" ]; then
    echo "WARNING: This script is intended for aarch64 (Orange Pi 5)."
    echo "  Detected: $MACHINE"
    echo "  Continuing anyway..."
fi

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Please run with sudo."
    exit 1
fi

# Show current version if available
if [ -f "${INSTALL_DIR}/${LIB_NAME}" ]; then
    CURRENT_SIZE=$(stat -c%s "${INSTALL_DIR}/${LIB_NAME}" 2>/dev/null || echo "unknown")
    echo "Current runtime: ${INSTALL_DIR}/${LIB_NAME} (${CURRENT_SIZE} bytes)"
else
    echo "No existing runtime found at ${INSTALL_DIR}/${LIB_NAME}"
fi

mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

echo "Downloading rknn-toolkit2 v${VERSION}..."
wget -q --show-progress "https://github.com/airockchip/rknn-toolkit2/archive/refs/tags/v${VERSION}.zip" -O "rknn-toolkit2.zip"

echo "Extracting runtime library..."
unzip -q "rknn-toolkit2.zip" "rknn-toolkit2-${VERSION}/rknpu2/runtime/Linux/librknn_api/${ARCH}/${LIB_NAME}"

SRC="rknn-toolkit2-${VERSION}/rknpu2/runtime/Linux/librknn_api/${ARCH}/${LIB_NAME}"

if [ ! -f "$SRC" ]; then
    echo "ERROR: Could not find ${LIB_NAME} in archive."
    echo "Trying alternative path..."
    SRC=$(find . -name "${LIB_NAME}" -path "*${ARCH}*" | head -1)
    if [ -z "$SRC" ]; then
        echo "ERROR: ${LIB_NAME} not found in any path. Aborting."
        rm -rf "$TMP_DIR"
        exit 1
    fi
fi

# Backup old runtime
if [ -f "${INSTALL_DIR}/${LIB_NAME}" ]; then
    BACKUP="${INSTALL_DIR}/${LIB_NAME}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "Backing up current runtime to: ${BACKUP}"
    cp "${INSTALL_DIR}/${LIB_NAME}" "$BACKUP"
fi

# Install new runtime
echo "Installing new runtime..."
cp "$SRC" "${INSTALL_DIR}/${LIB_NAME}"
chmod 644 "${INSTALL_DIR}/${LIB_NAME}"
ldconfig

# Verify
NEW_SIZE=$(stat -c%s "${INSTALL_DIR}/${LIB_NAME}")
echo ""
echo "=== Done ==="
echo "Installed: ${INSTALL_DIR}/${LIB_NAME} (${NEW_SIZE} bytes)"
echo "Run 'python3 -c \"from rknnlite.api import RKNNLite; print(RKNNLite())\"' to verify."

# Cleanup
rm -rf "$TMP_DIR"
