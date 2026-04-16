"""
Phase 5: Palm normal, solvePnP (wrist + 4 MCPs), relative splay/curl angles.
Verification: prints rotation vector / euler approx and normal.
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import cv2
import numpy as np

from handtracking.topology import HAND_10_NAMES

# Slot indices: wrist=0, MCP index..pinky = 1..4
IDX_WRIST = 0
IDX_MCP_INDEX = 1
IDX_MCP_PINKY = 4


def palm_normal_from_mcp_plane(
    points_3d: np.ndarray,
) -> np.ndarray:
    """Normal = (index_MCP - wrist) x (pinky_MCP - wrist)."""
    w = points_3d[IDX_WRIST]
    idx_mcp = points_3d[IDX_MCP_INDEX]
    pky_mcp = points_3d[IDX_MCP_PINKY]
    a = idx_mcp - w
    b = pky_mcp - w
    n = np.cross(a, b)
    ln = np.linalg.norm(n) + 1e-8
    return n / ln


def canonical_object_points_mm(scale_mm: float = 100.0) -> np.ndarray:
    """Rough coplanar base in object frame: wrist + 4 MCPs (regular pentagon-ish layout)."""
    # 5 points in mm-ish arbitrary but non-degenerate
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [scale_mm * 0.9, 0.0, 0.0],
            [scale_mm * 0.95, scale_mm * 0.35, 0.0],
            [scale_mm * 0.75, scale_mm * 0.65, 0.0],
            [scale_mm * 0.45, scale_mm * 0.85, 0.0],
        ],
        dtype=np.float64,
    )


def solve_pnp_rigid_base(
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    image_points: (5,2) wrist + 4 MCPs in pixels (order matches topology slots 0-4).
    Returns rvec, tvec, normal (camera frame).
    """
    obj = canonical_object_points_mm()
    ok, rvec, tvec = cv2.solvePnP(
        obj,
        image_points.astype(np.float64),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    pts_cam = (R @ obj.T).T + tvec.reshape(1, 3)
    n = palm_normal_from_mcp_plane(pts_cam)
    return rvec, tvec, n


def angles_splay_curl(
    R: np.ndarray,
) -> Tuple[float, float]:
    """Coarse splay (abduction) and curl from wrist frame (degrees, demo)."""
    # X axis ~ toward index MCP in object frame mapped by R
    ex = R[:, 0]
    ez = R[:, 2]
    splay = math.degrees(math.atan2(ex[1], ex[0] + 1e-8))
    curl = math.degrees(math.atan2(ez[2], ez[0] + 1e-8))
    return splay, curl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="Synthetic image points")
    args = ap.parse_args()
    # User specified default 160 FOV camera
    # At 640x480: fx = (640/2) / tan(80 deg) = ~320 / 5.671 = ~56.4
    fx, fy, cx, cy = 56.4, 56.4, 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    if args.demo:
        # Fake detected 2D base (wrist + MCPs) in pixels
        ip = np.array(
            [
                [cx, cy + 40],
                [cx + 80, cy + 10],
                [cx + 85, cy - 30],
                [cx + 50, cy - 60],
                [cx - 10, cy - 50],
            ],
            dtype=np.float64,
        )
    else:
        raise SystemExit("Use --demo or extend with live keypoints.")

    rvec, tvec, normal = solve_pnp_rigid_base(ip[:5], K, dist)
    R, _ = cv2.Rodrigues(rvec)
    splay, curl = angles_splay_curl(R)

    print("HAND_10_NAMES[0:5] base:", [HAND_10_NAMES[i] for i in range(5)])
    print("rvec_deg:", (rvec * 180.0 / math.pi).ravel())
    print("tvec_mm_approx:", tvec.ravel())
    print("palm_normal_cam:", normal)
    print("splay_deg:", splay, "curl_deg:", curl)


if __name__ == "__main__":
    main()
