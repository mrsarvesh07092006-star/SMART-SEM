"""
SMART-SEM Failure Gallery Generator.

For every failed sample (error > 5.0 px):
Saves:
- reference.png
- search.png
- overlay_visualization.png (GT = Green box/cross, Predicted = Red box/cross, Top-10 candidates = Yellow circles)
- error_diagnostic.json (Top peaks, peak ratio, pitch offset, spatial entropy, failure category)
"""

from __future__ import annotations
import csv
import json
import math
import os
import sys
import shutil
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.smart_sem.localization_engine import smart_sem_localize
from src.smart_sem.topology import discover_topology
from src.smart_sem.failure_analysis import FailureAnalysisAgent

def generate_failure_gallery(manifest_path: str = "results/dataset/manifest.csv", out_dir: str = "results/failure_gallery"):
    os.makedirs(out_dir, exist_ok=True)
    dataset_root = os.path.dirname(os.path.abspath(manifest_path))
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    failure_agent = FailureAnalysisAgent()
    failed_count = 0

    print(f"==================================================")
    print(f" Generating Failure Gallery in: {out_dir}")
    print(f"==================================================")

    for r in rows:
        sid = r["sample_id"]
        arch = r["architecture"]
        gt_x, gt_y = float(r["gt_x"]), float(r["gt_y"])

        ref_path = os.path.join(dataset_root, r["reference_path"])
        search_path = os.path.join(dataset_root, r["search_path"])

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None or search_img is None:
            continue

        topo = discover_topology(ref_img)
        loc = smart_sem_localize(ref_img, search_img, top_k=10)

        pred_x, pred_y = loc["pred_x"], loc["pred_y"]
        err_px = float(math.hypot(pred_x - gt_x, pred_y - gt_y))

        if err_px > 5.0:
            failed_count += 1
            sample_dir = os.path.join(out_dir, sid)
            os.makedirs(sample_dir, exist_ok=True)

            # Copy reference and search images
            cv2.imwrite(os.path.join(sample_dir, "reference.png"), ref_img)
            cv2.imwrite(os.path.join(sample_dir, "search.png"), search_img)

            # Build visualization
            search_bgr = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

            # Draw Ground Truth (Green)
            cv2.circle(search_bgr, (int(round(gt_x)), int(round(gt_y))), 12, (0, 255, 0), 2)
            cv2.drawMarker(search_bgr, (int(round(gt_x)), int(round(gt_y))), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            cv2.rectangle(search_bgr, (int(round(gt_x - 50)), int(round(gt_y - 50))), (int(round(gt_x + 50)), int(round(gt_y + 50))), (0, 255, 0), 2)

            # Draw Predicted Peak #1 (Red)
            cv2.circle(search_bgr, (int(round(pred_x)), int(round(pred_y))), 10, (0, 0, 255), 2)
            cv2.drawMarker(search_bgr, (int(round(pred_x)), int(round(pred_y))), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
            cv2.rectangle(search_bgr, (int(round(pred_x - 50)), int(round(pred_y - 50))), (int(round(pred_x + 50)), int(round(pred_y + 50))), (0, 0, 255), 2)

            # Draw Other Top-10 Candidates (Yellow)
            candidates = loc.get("top_k_candidates", [])
            for c in candidates[1:]:
                cx, cy = int(round(c["center_x"])), int(round(c["center_y"]))
                cv2.circle(search_bgr, (cx, cy), 6, (0, 255, 255), 1)
                cv2.putText(search_bgr, f"#{c['rank']}", (cx + 8, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Header text
            cv2.putText(search_bgr, f"FAILED: {sid} ({arch}) | Error: {err_px:.1f}px", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(search_bgr, f"GT: ({gt_x:.1f}, {gt_y:.1f}) [Green] | Pred: ({pred_x:.1f}, {pred_y:.1f}) [Red]", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imwrite(os.path.join(sample_dir, "overlay_visualization.png"), search_bgr)

            # Diagnose failure
            diag = failure_agent.diagnose_failure(
                sample_id=sid,
                architecture=arch,
                gt_x=gt_x, gt_y=gt_y,
                pred_x=pred_x, pred_y=pred_y,
                confidence=loc["confidence"],
                topology_info=topo
            )

            # Save error diagnostic JSON
            diagnostic_data = {
                "sample_id": sid,
                "architecture": arch,
                "error_px": err_px,
                "gt_x": gt_x, "gt_y": gt_y,
                "pred_x": pred_x, "pred_y": pred_y,
                "confidence_score": loc["confidence"],
                "peak_ratio": loc["peak_ratio"],
                "confidence_margin": float(loc["confidence"] / (loc["top_k_candidates"][1]["score"] + 1e-6)) if len(loc["top_k_candidates"]) > 1 else 999.0,
                "entropy": loc["entropy"],
                "failure_category": diag["category"],
                "diagnosis_description": diag["description"],
                "pitch_nm": topo.get("pitch_nm", 0.0),
                "pitch_px_search": topo.get("pitch_px_search_est", 0.0),
                "top_10_candidates": loc["top_k_candidates"],
            }

            with open(os.path.join(sample_dir, "error_diagnostic.json"), "w", encoding="utf-8") as f:
                json.dump(diagnostic_data, f, indent=2)

            print(f" [!] Logged failure for {sid} ({arch}) -> Error: {err_px:.1f}px [{diag['category']}]")

    print(f"\n[OK] Failure Gallery generated: {failed_count} failure cases saved to {out_dir}\n")

if __name__ == "__main__":
    generate_failure_gallery()
