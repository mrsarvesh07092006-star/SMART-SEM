"""
SMART-SEM Layer 6 Upgrade: 2D Kinematic Kalman Filter Stage Tracker & Motion Model.

Models SEM mechanical stage trajectory and cumulative drift:
- State vector: [x, y, vx, vy]^T (Position & Velocity in Search pixels)
- Predicts expected stage location prior to imaging: x_pred, y_pred
- Computes Mahalanobis spatial consistency distance for candidate verification
"""

from __future__ import annotations
import math
import numpy as np

class StageKalmanTracker:
    """2D Kinematic Kalman Filter tracking SEM stage motion across inspection sites."""

    def __init__(self, dt: float = 1.0, process_noise_std: float = 2.0, measurement_noise_std: float = 3.0):
        self.dt = dt
        # State: [x, y, vx, vy]
        self.x = np.array([500.0, 500.0, 0.0, 0.0], dtype=np.float64)
        
        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Measurement matrix H (we observe position x, y)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Covariance P
        self.P = np.eye(4, dtype=np.float64) * 50.0

        # Process Noise Q
        q = process_noise_std ** 2
        self.Q = np.array([
            [0.25*dt**4*q, 0.0,          0.5*dt**3*q,  0.0        ],
            [0.0,          0.25*dt**4*q, 0.0,          0.5*dt**3*q],
            [0.5*dt**3*q,  0.0,          dt**2*q,      0.0        ],
            [0.0,          0.5*dt**3*q,  0.0,          dt**2*q    ]
        ], dtype=np.float64)

        # Measurement Noise R
        r = measurement_noise_std ** 2
        self.R = np.eye(2, dtype=np.float64) * r
        self.initialized = False

    def predict(self) -> tuple[float, float, float, float]:
        """Predicts next stage state [x, y, vx, vy] and covariance."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])

    def update(self, measured_x: float, measured_y: float):
        """Updates Kalman state with verified measurement."""
        z = np.array([measured_x, measured_y], dtype=np.float64)
        if not self.initialized:
            self.x = np.array([measured_x, measured_y, 0.0, 0.0], dtype=np.float64)
            self.P = np.eye(4, dtype=np.float64) * 10.0
            self.initialized = True
            return

        y = z - self.H @ self.x # Innovation
        S = self.H @ self.P @ self.H.T + self.R # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def compute_mahalanobis_distance(self, cand_x: float, cand_y: float) -> float:
        """Computes Mahalanobis distance from predicted stage coordinate to candidate location."""
        pos_cov = self.P[:2, :2] + self.R
        diff = np.array([cand_x - self.x[0], cand_y - self.x[1]])
        try:
            inv_cov = np.linalg.inv(pos_cov)
            dist_sq = float(diff.T @ inv_cov @ diff)
            return math.sqrt(max(0.0, dist_sq))
        except Exception:
            return float(math.hypot(diff[0], diff[1]) / 10.0)
