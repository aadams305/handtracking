"""Constant-velocity Kalman filter for bounding box tracking.

Predicts bbox position between detection frames, compensating for hand motion.
State: [cx, cy, w, h, vx, vy, vw, vh] (position + velocity).
"""

from __future__ import annotations

import numpy as np


class BboxKalman:
    """Kalman filter for a single bounding box (cx, cy, w, h) with constant velocity model.

    Args:
        process_noise: Process noise scale (higher = trust measurements more).
        measurement_noise: Measurement noise scale (higher = smoother, more lag).
        dt: Expected time step in seconds.
    """

    def __init__(
        self,
        process_noise: float = 50.0,
        measurement_noise: float = 10.0,
        dt: float = 1.0 / 30.0,
    ) -> None:
        self.dt = dt
        self._dim_state = 8  # [cx, cy, w, h, vx, vy, vw, vh]
        self._dim_meas = 4   # [cx, cy, w, h]

        # State vector
        self.x = np.zeros(self._dim_state, dtype=np.float64)

        # State covariance
        self.P = np.eye(self._dim_state, dtype=np.float64) * 1000.0

        # Measurement matrix: we observe [cx, cy, w, h]
        self.H = np.zeros((self._dim_meas, self._dim_state), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Measurement noise
        self.R = np.eye(self._dim_meas, dtype=np.float64) * measurement_noise ** 2

        # Process noise
        self._q = process_noise
        self._initialized = False

    def _transition_matrix(self, dt: float) -> np.ndarray:
        F = np.eye(self._dim_state, dtype=np.float64)
        F[0, 4] = dt  # cx += vx * dt
        F[1, 5] = dt  # cy += vy * dt
        F[2, 6] = dt  # w  += vw * dt
        F[3, 7] = dt  # h  += vh * dt
        return F

    def _process_noise_matrix(self, dt: float) -> np.ndarray:
        q = self._q
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        Q = np.zeros((self._dim_state, self._dim_state), dtype=np.float64)
        for i in range(4):
            Q[i, i] = dt4 * q
            Q[i, i + 4] = dt3 * q
            Q[i + 4, i] = dt3 * q
            Q[i + 4, i + 4] = dt2 * q
        return Q

    def init(self, bbox_xywh: tuple[int, int, int, int]) -> None:
        """Initialize state from first measurement (x, y, w, h)."""
        x, y, w, h = bbox_xywh
        cx = x + w / 2.0
        cy = y + h / 2.0
        self.x = np.array([cx, cy, float(w), float(h), 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(self._dim_state, dtype=np.float64) * 100.0
        self.P[4, 4] = 1000.0
        self.P[5, 5] = 1000.0
        self.P[6, 6] = 100.0
        self.P[7, 7] = 100.0
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def predict(self, dt: float | None = None) -> tuple[int, int, int, int]:
        """Predict next state. Returns predicted bbox as (x, y, w, h)."""
        if dt is None:
            dt = self.dt
        F = self._transition_matrix(dt)
        Q = self._process_noise_matrix(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Clamp w/h to positive
        self.x[2] = max(self.x[2], 10.0)
        self.x[3] = max(self.x[3], 10.0)

        return self._state_to_xywh()

    def update(self, bbox_xywh: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Update state with measurement. Returns corrected bbox as (x, y, w, h)."""
        x, y, w, h = bbox_xywh
        z = np.array([x + w / 2.0, y + h / 2.0, float(w), float(h)], dtype=np.float64)

        y_residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_residual
        I_KH = np.eye(self._dim_state) - K @ self.H
        self.P = I_KH @ self.P

        self.x[2] = max(self.x[2], 10.0)
        self.x[3] = max(self.x[3], 10.0)

        return self._state_to_xywh()

    def _state_to_xywh(self) -> tuple[int, int, int, int]:
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        x = int(cx - w / 2.0)
        y = int(cy - h / 2.0)
        return (x, y, int(w), int(h))

    def reset(self) -> None:
        """Reset filter state."""
        self.x = np.zeros(self._dim_state, dtype=np.float64)
        self.P = np.eye(self._dim_state, dtype=np.float64) * 1000.0
        self._initialized = False
