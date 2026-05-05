"""One-Euro Filter for real-time landmark smoothing.

Reduces jitter at low speeds while preserving responsiveness during fast motion.
Designed for 21x2 hand landmark arrays, with per-joint adaptive filtering.

Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for
Noisy Input in Interactive Systems", CHI 2012.
"""

from __future__ import annotations

import numpy as np


def _smoothing_factor(te: float, cutoff: float) -> float:
    r = 2.0 * np.pi * cutoff * te
    return r / (r + 1.0)


class OneEuroFilter2D:
    """One-Euro filter for (N, 2) landmark arrays.

    Args:
        num_points: Number of points to filter (21 for hand landmarks).
        min_cutoff: Minimum cutoff frequency (Hz). Lower = less jitter at rest.
        beta: Speed coefficient. Higher = less lag during fast motion.
        d_cutoff: Derivative cutoff frequency (Hz).
        fps: Expected frame rate (used for initial dt estimate).
    """

    def __init__(
        self,
        num_points: int = 21,
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
        fps: float = 30.0,
    ) -> None:
        self.num_points = num_points
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None
        self._default_dt = 1.0 / fps

    def reset(self) -> None:
        """Clear filter state (call when hand is lost/re-detected)."""
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    @property
    def initialized(self) -> bool:
        return self._x_prev is not None

    def __call__(self, x: np.ndarray, t: float | None = None) -> np.ndarray:
        """Filter a (N, 2) array of landmark positions.

        Args:
            x: Current landmark positions, shape (num_points, 2).
            t: Current timestamp in seconds. If None, uses default dt.

        Returns:
            Filtered positions, same shape as input.
        """
        assert x.shape == (self.num_points, 2)

        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t if t is not None else 0.0
            return x.copy()

        if t is not None and self._t_prev is not None:
            dt = t - self._t_prev
            if dt <= 0:
                dt = self._default_dt
        else:
            dt = self._default_dt

        # Filter the derivative (speed) with fixed cutoff
        a_d = _smoothing_factor(dt, self.d_cutoff)
        dx = (x - self._x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff based on speed
        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter the signal with adaptive cutoff (per-element)
        a = 2.0 * np.pi * cutoff * dt
        a = a / (a + 1.0)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()
        self._t_prev = t if t is not None else (self._t_prev + dt)

        return x_hat
