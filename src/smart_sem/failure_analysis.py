"""
SMART-SEM Failure Analysis & Diagnosis Engine (Agent 4).

Taxonomy of Semiconductor Localization Failures:
- Type A: Periodic Ambiguity (Locked onto wrong periodic DRAM cell / FinFET fin)
- Type B: Scale Mismatch (10x ratio deviation or uncalibrated magnification)
- Type C: Noise Dominance (Poisson shot noise / speckle peak > true structural peak)
- Type D: Drift Distortion (Local raster shear / stage trajectory deformation)
- Type E: Feature Starvation (Homogeneous background lacking unique landmarks)
"""

from __future__ import annotations
import json
import math
import os
import numpy as np

class FailureAnalysisAgent:
    """Diagnoses, categorizes, and logs localization errors into structured taxonomy."""

    def __init__(self, failure_db_path: str = "research/failures_database.json"):
        self.failure_db_path = failure_db_path
        self.records: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.failure_db_path):
            try:
                with open(self.failure_db_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except Exception:
                self.records = []

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.failure_db_path)), exist_ok=True)
        with open(self.failure_db_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def diagnose_failure(
        self,
        sample_id: str,
        architecture: str,
        gt_x: float,
        gt_y: float,
        pred_x: float,
        pred_y: float,
        confidence: float,
        topology_info: dict,
        params: dict | None = None
    ) -> dict:
        """
        Classifies localization error into Type A, B, C, D, or E failure modes.
        """
        dx = pred_x - gt_x
        dy = pred_y - gt_y
        error_px = float(math.hypot(dx, dy))

        if error_px <= 5.0:
            category = "SUCCESS_PASS"
            description = f"Accurate localization within tolerance (error: {error_px:.2f} px)"
            is_failure = False
        else:
            is_failure = True
            pitch_px = topology_info.get("pitch_px_search_est", 7.0)
            
            # Check Type C: Noise Dominance first if confidence is very low
            if confidence < 0.35:
                category = "TYPE_C_NOISE_DOMINANCE"
                description = f"Low signal-to-noise ratio; spurious noise peak selected (confidence={confidence:.3f})"
            elif pitch_px > 2.0:
                pitch_ratio_x = abs(dx) / pitch_px
                pitch_ratio_y = abs(dy) / pitch_px
                rem_x = abs(pitch_ratio_x - round(pitch_ratio_x))
                rem_y = abs(pitch_ratio_y - round(pitch_ratio_y))
                
                if (rem_x < 0.25 and pitch_ratio_x >= 0.8) or (rem_y < 0.25 and pitch_ratio_y >= 0.8):
                    category = "TYPE_A_PERIODIC_AMBIGUITY"
                    k_x = int(round(pitch_ratio_x))
                    k_y = int(round(pitch_ratio_y))
                    description = f"Locked onto adjacent periodic cell (k_x={k_x}, k_y={k_y} at pitch={pitch_px:.1f}px)"
                elif params and (params.get("shear_amplitude_px", 0) > 3.0 or params.get("drift_jitter_px", 0) > 1.5):
                    category = "TYPE_D_DRIFT_DISTORTION"
                    description = f"Severe raster shear/jitter distortion degraded template alignment"
                elif topology_info.get("mean_gradient_magnitude", 0) < 6.0:
                    category = "TYPE_E_FEATURE_STARVATION"
                    description = f"Feature-starved homogeneous region lacking high-frequency edges"
                else:
                    category = "TYPE_B_SCALE_OR_ROTATION_MISMATCH"
                    description = f"Non-rigid deformation or magnification mismatch"
            else:
                category = "TYPE_B_SCALE_MISMATCH"
                description = f"Scale or affine distortion mismatch"

        diagnosis = {
            "sample_id": sample_id,
            "architecture": architecture,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "error_px": error_px,
            "confidence": confidence,
            "category": category,
            "is_failure": is_failure,
            "description": description,
        }

        if is_failure:
            self.records.append(diagnosis)
            self.save()

        return diagnosis
