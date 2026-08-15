"""
Navigation Error Simulator for Semiconductor SEM Tools.

Models physical stage positioning imperfections including:
- Stage mechanical drift (random walk step errors)
- Mechanical backlash (directional hysteresis)
- Thermal expansion drift (slow continuous shift)
- Mechanical vibration jitter (high-frequency noise)
- Cumulative multi-step positioning error accumulation
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class NavigationParams:
    # Per-step stage drift (nm)
    drift_sigma_nm: float = 2.0
    # Backlash max offset (nm) on direction reversal
    backlash_max_nm: float = 3.0
    # Thermal expansion drift rate (nm per stage move step)
    thermal_rate_nm: float = 0.5
    # High-frequency mechanical vibration (nm)
    vibration_sigma_nm: float = 1.0
    # Systematic coordinate system offset (nm)
    coord_system_bias_nm: float = 0.0

class NavigationErrorSimulator:
    """Simulates realistic stage movement and positioning error accumulation for SEM inspection."""
    
    def __init__(self, params: NavigationParams | None = None, rng: np.random.Generator | None = None):
        self.params = params or NavigationParams()
        self.rng = rng or np.random.default_rng(42)
        self.current_direction = (1.0, 0.0) # Unit vector
        self.total_steps = 0
        self.cumulative_offset_nm = [0.0, 0.0]

    def simulate_move(self, target_dx_nm: float, target_dy_nm: float) -> tuple[float, float, dict]:
        """
        Simulates moving the SEM stage by target_dx_nm, target_dy_nm.
        Returns (actual_dx_nm, actual_dy_nm, error_details).
        """
        self.total_steps += 1
        p = self.params
        
        # 1. Random stage drift (Gaussian step error)
        drift_x = self.rng.normal(0, p.drift_sigma_nm)
        drift_y = self.rng.normal(0, p.drift_sigma_nm)
        
        # 2. Backlash (triggered on direction change)
        move_len = math.hypot(target_dx_nm, target_dy_nm)
        backlash_x, backlash_y = 0.0, 0.0
        if move_len > 1e-6:
            new_dir = (target_dx_nm / move_len, target_dy_nm / move_len)
            dot_product = new_dir[0] * self.current_direction[0] + new_dir[1] * self.current_direction[1]
            if dot_product < 0.5: # Direction reversed or turned sharply
                reversal_severity = (1.0 - dot_product) / 2.0
                backlash_mag = p.backlash_max_nm * reversal_severity * self.rng.beta(2, 5)
                backlash_x = -new_dir[0] * backlash_mag
                backlash_y = -new_dir[1] * backlash_mag
            self.current_direction = new_dir

        # 3. Thermal drift accumulation
        thermal_x = p.thermal_rate_nm * self.total_steps * 0.1
        thermal_y = p.thermal_rate_nm * self.total_steps * 0.1

        # 4. Vibration jitter
        vib_x = self.rng.normal(0, p.vibration_sigma_nm)
        vib_y = self.rng.normal(0, p.vibration_sigma_nm)

        # Total step offset
        step_error_x = drift_x + backlash_x + thermal_x + vib_x + p.coord_system_bias_nm
        step_error_y = drift_y + backlash_y + thermal_y + vib_y + p.coord_system_bias_nm

        self.cumulative_offset_nm[0] += step_error_x
        self.cumulative_offset_nm[1] += step_error_y

        actual_dx = target_dx_nm + step_error_x
        actual_dy = target_dy_nm + step_error_y

        details = {
            "step": self.total_steps,
            "drift_nm": (drift_x, drift_y),
            "backlash_nm": (backlash_x, backlash_y),
            "thermal_nm": (thermal_x, thermal_y),
            "vibration_nm": (vib_x, vib_y),
            "step_error_total_nm": (step_error_x, step_error_y),
            "cumulative_error_nm": tuple(self.cumulative_offset_nm),
        }
        return actual_dx, actual_dy, details

    def generate_cumulative_trajectory_error(self, n_steps: int = 5) -> tuple[float, float, dict]:
        """Runs a multi-step inspection sequence and returns net accumulated coordinate error in nm."""
        for _ in range(n_steps):
            dx_target = self.rng.uniform(-1000, 1000)
            dy_target = self.rng.uniform(-1000, 1000)
            self.simulate_move(dx_target, dy_target)

        cum_x, cum_y = self.cumulative_offset_nm
        report = {
            "n_steps": n_steps,
            "cumulative_offset_nm": (cum_x, cum_y),
            "cumulative_offset_px_at_10nm": (cum_x / 10.0, cum_y / 10.0), # In search image pixels
            "euclidean_nav_error_nm": float(math.hypot(cum_x, cum_y)),
            "euclidean_nav_error_px": float(math.hypot(cum_x, cum_y) / 10.0),
        }
        return cum_x, cum_y, report

def apply_navigation_error_to_gt(gt_x: float, gt_y: float, nav_error_px: tuple[float, float]) -> tuple[float, float]:
    """
    Applies simulated stage navigation offset to ground truth coordinate
    to model stage positioning uncertainty in search space.
    """
    obs_x = gt_x + nav_error_px[0]
    obs_y = gt_y + nav_error_px[1]
    return obs_x, obs_y
