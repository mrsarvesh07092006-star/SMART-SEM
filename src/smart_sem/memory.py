"""
SMART-SEM Layer 6 & Layer 7: Real Wafer Memory Graph & Historical Defect Prior Engine.

Features:
- Fingerprint Nearest-Neighbor Search (searches prior inspection sessions by pitch & orientation)
- Memory-Guided Localization Search Priors (retrieves expected region bounding box to guide matcher)
- Historical Defect Region Prior Store (logs defect frequencies by wafer zone/lot ID)
- Persistent JSON Storage
"""

from __future__ import annotations
import json
import math
import os
import time
from dataclasses import dataclass, asdict

@dataclass
class WaferFingerprint:
    wafer_id: str
    architecture: str
    pitch_nm: float
    orientation_deg: float
    periodicity_index: float
    defect_hotspots: list[tuple[float, float]] # List of (x, y) coordinates of historical defects
    total_sites_inspected: int = 0
    avg_confidence: float = 0.0

class WaferMemoryGraph:
    """Session-persistent wafer memory graph with nearest-neighbor retrieval and defect priors."""

    def __init__(self, memory_file: str = "wafer_memory_store.json"):
        self.memory_file = memory_file
        self.fingerprints: dict[str, dict] = {}
        self.inspection_history: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.fingerprints = data.get("fingerprints", {})
                    self.inspection_history = data.get("inspection_history", [])
            except Exception:
                self.fingerprints = {}
                self.inspection_history = []

    def save(self):
        data = {
            "fingerprints": self.fingerprints,
            "inspection_history": self.inspection_history[-500:],
        }
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def find_nearest_wafer_prior(self, architecture: str, pitch_nm: float, orientation_deg: float) -> dict | None:
        """Finds nearest historical wafer fingerprint matching architecture, pitch, and orientation."""
        best_fp = None
        min_dist = float("inf")

        for w_id, fp in self.fingerprints.items():
            if fp.get("architecture") == architecture:
                dp = abs(fp.get("pitch_nm", 0.0) - pitch_nm)
                da = abs(fp.get("orientation_deg", 0.0) - orientation_deg)
                dist = dp + 0.5 * da
                if dist < min_dist:
                    min_dist = dist
                    best_fp = fp

        return best_fp if min_dist < 15.0 else None

    def get_search_region_prior(self, architecture: str, pitch_nm: float, orientation_deg: float) -> tuple[float, float, float, float] | None:
        """
        Retrieves search bounding box prior (x0, y0, x1, y1) based on historical successful matches.
        """
        prior_fp = self.find_nearest_wafer_prior(architecture, pitch_nm, orientation_deg)
        if not prior_fp or not self.inspection_history:
            return None

        # Filter history for this architecture
        matched_history = [
            h for h in self.inspection_history 
            if h.get("architecture") == architecture and h.get("confidence", 0.0) > 0.75
        ]
        if not matched_history:
            return None

        xs = [h["pred_x"] for h in matched_history]
        ys = [h["pred_y"] for h in matched_history]

        mean_x, std_x = float(np.mean(xs)), float(np.std(xs)) if len(xs) > 1 else 150.0
        mean_y, std_y = float(np.mean(ys)), float(np.std(ys)) if len(ys) > 1 else 150.0

        r_x = max(150.0, 2.0 * std_x)
        r_y = max(150.0, 2.0 * std_y)

        x0, y0 = max(0.0, mean_x - r_x), max(0.0, mean_y - r_y)
        x1, y1 = min(1000.0, mean_x + r_x), min(1000.0, mean_y + r_y)
        return (x0, y0, x1, y1)

    def update_fingerprint(self, wafer_id: str, architecture: str, topology_info: dict):
        fp = self.fingerprints.get(wafer_id, {
            "wafer_id": wafer_id,
            "architecture": architecture,
            "pitch_nm": topology_info.get("pitch_nm", 0.0),
            "orientation_deg": topology_info.get("orientation_deg", 0.0),
            "periodicity_index": topology_info.get("periodicity_index", 0.0),
            "defect_hotspots": [],
            "total_sites_inspected": 0,
            "avg_confidence": 0.0,
        })
        fp["total_sites_inspected"] += 1
        self.fingerprints[wafer_id] = fp
        self.save()

    def log_inspection(self, sample_id: str, gt_xy: tuple[float, float], pred_xy: tuple[float, float], loc_res: dict, amb_info: dict, architecture: str = "dram_1x"):
        dx = pred_xy[0] - gt_xy[0]
        dy = pred_xy[1] - gt_xy[1]
        dist = (dx**2 + dy**2) ** 0.5

        entry = {
            "timestamp": time.time(),
            "sample_id": sample_id,
            "architecture": architecture,
            "gt_x": gt_xy[0], "gt_y": gt_xy[1],
            "pred_x": pred_xy[0], "pred_y": pred_xy[1],
            "error_px": dist,
            "confidence": loc_res.get("confidence", 0.0),
            "ambiguity_score": amb_info.get("peak_ratio", 0.0),
            "failure_reason": amb_info.get("ambiguity_class", "UNKNOWN"),
        }
        self.inspection_history.append(entry)
        self.save()
