#!/usr/bin/env python3
"""
SMART-SEM Standalone Single-Pair Inference Script.
Official Evaluation Script for Applied Materials Drift-Sense Track.

Usage:
    python infer.py --reference <path_to_ref.png> --search <path_to_search.png>
    OR
    python infer.py <path_to_ref.png> <path_to_search.png>

Outputs:
    Prints the predicted center coordinates (x, y) of the reference pattern within the search image.
"""

from __future__ import annotations
import argparse
import os
import sys
import math
import cv2
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.smart_sem.localization_engine import smart_sem_localize

def locate_reference(ref_path: str, search_path: str, stage_prior: tuple[float, float] | None = None) -> tuple[float, float, dict]:
    """
    Locates the 100x high-mag reference image inside the 10x low-mag search image.
    Returns (pred_x, pred_y, telemetry_dict).
    """
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    if not os.path.exists(search_path):
        raise FileNotFoundError(f"Search image not found: {search_path}")

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    if ref_img is None:
        raise ValueError(f"Could not decode reference image: {ref_path}")
    if search_img is None:
        raise ValueError(f"Could not decode search image: {search_path}")

    # Run SMART-SEM Localization Engine
    loc_res = smart_sem_localize(
        ref_img,
        search_img,
        scales=(9.5, 9.8, 10.0, 10.2, 10.5),
        stage_prior_xy=stage_prior
    )

    pred_x = float(loc_res["pred_x"])
    pred_y = float(loc_res["pred_y"])

    # Center-tie-breaker rule (as specified in problem statement: if multiple matches, prefer closest to image center)
    if loc_res.get("is_ambiguous", False) and len(loc_res.get("top_k_candidates", [])) > 1:
        cands = loc_res["top_k_candidates"]
        top_score = cands[0].get("ranking_score", cands[0]["score"])
        # Candidates within 3% of top score
        tie_cands = [c for c in cands if (top_score - c.get("ranking_score", c["score"])) < 0.03]
        if len(tie_cands) > 1 and stage_prior is None:
            sh, sw = search_img.shape
            center_xy = (sw / 2.0, sh / 2.0)
            tie_cands.sort(key=lambda c: math.hypot(c["center_x"] - center_xy[0], c["center_y"] - center_xy[1]))
            pred_x = float(tie_cands[0]["center_x"])
            pred_y = float(tie_cands[0]["center_y"])

    return pred_x, pred_y, loc_res

def main():
    parser = argparse.ArgumentParser(description="SMART-SEM Single-Pair Inference Entrypoint")
    parser.add_argument("pos_ref", nargs="?", default=None, help="Positional argument: path to reference image")
    parser.add_argument("pos_search", nargs="?", default=None, help="Positional argument: path to search image")
    parser.add_argument("--reference", "-r", type=str, default=None, help="Flag argument: path to reference image")
    parser.add_argument("--search", "-s", type=str, default=None, help="Flag argument: path to search image")
    parser.add_argument("--stage-prior-x", type=float, default=None, help="Optional simulated stage prior X coordinate")
    parser.add_argument("--stage-prior-y", type=float, default=None, help="Optional simulated stage prior Y coordinate")
    parser.add_argument("--json", action="store_true", help="Output full JSON telemetry payload")
    args = parser.parse_args()

    ref_path = args.reference or args.pos_ref
    search_path = args.search or args.pos_search

    if not ref_path or not search_path:
        print("Usage: python infer.py --reference <ref.png> --search <search.png>")
        print("   OR: python infer.py <ref.png> <search.png>")
        sys.exit(1)

    stage_prior = None
    if args.stage_prior_x is not None and args.stage_prior_y is not None:
        stage_prior = (args.stage_prior_x, args.stage_prior_y)

    pred_x, pred_y, telemetry = locate_reference(ref_path, search_path, stage_prior=stage_prior)

    if args.json:
        out = {
            "pred_x": pred_x,
            "pred_y": pred_y,
            "confidence": telemetry.get("confidence", 0.0),
            "ambiguity_class": telemetry.get("ambiguity_class", "UNKNOWN"),
            "entropy": telemetry.get("entropy", 0.0),
        }
        print(json.dumps(out, indent=2))
    else:
        # Standard Clean Output for Evaluators
        print(f"Predicted Center: ({pred_x:.4f}, {pred_y:.4f})")
        print(f"{pred_x:.4f}, {pred_y:.4f}")

if __name__ == "__main__":
    main()
