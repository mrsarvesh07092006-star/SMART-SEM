"""
SMART-SEM Failure Replay & Hard Negative Dataset Builder (Priority 4).

Collects candidate feature vectors across all benchmark samples and failure cases:
- Labels each candidate as: correct: true (error <= 5.0 px) vs correct: false (error > 5.0 px)
- Saves structured dataset to `research/ranking_training_dataset.json` for failure replay & learned ranker training.
"""

from __future__ import annotations
import csv
import json
import math
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.smart_sem.localization_engine import smart_sem_localize
from src.smart_sem.candidate_reranker import CandidateReRanker

def build_failure_replay_dataset(
    manifest_path: str = "results/dataset/manifest.csv",
    out_json: str = "research/ranking_training_dataset.json"
):
    dataset_root = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    reranker = CandidateReRanker()
    dataset_records = []

    print(f"==================================================")
    print(f" Building Failure Replay Dataset from {len(rows)} samples...")
    print(f"==================================================")

    for r in rows:
        sid = r["sample_id"]
        arch = r["architecture"]
        gt_x, gt_y = float(r["gt_x"]), float(r["gt_y"])
        obs_x = float(r["obs_gt_x"]) if "obs_gt_x" in r else None
        obs_y = float(r["obs_gt_y"]) if "obs_gt_y" in r else None
        stage_prior = (obs_x, obs_y) if (obs_x and obs_y) else None

        ref_path = os.path.join(dataset_root, r["reference_path"])
        search_path = os.path.join(dataset_root, r["search_path"])

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        if ref_img is None or search_img is None: continue

        loc_res = smart_sem_localize(ref_img, search_img, top_k=10, stage_prior_xy=stage_prior)
        cands = loc_res.get("top_k_candidates", [])
        
        tw, th = 100, 100
        ref_tmpl = cv2.resize(ref_img, (tw, th), interpolation=cv2.INTER_AREA)

        # Gradient maps
        gx_ref = cv2.Sobel(ref_tmpl.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy_ref = cv2.Sobel(ref_tmpl.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        mag_ref = np.sqrt(gx_ref**2 + gy_ref**2)
        mag_ref_norm = (mag_ref - np.mean(mag_ref)) / (np.std(mag_ref) + 1e-6)

        gx_s = cv2.Sobel(search_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy_s = cv2.Sobel(search_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        mag_s = np.sqrt(gx_s**2 + gy_s**2)

        sample_candidates = []
        for c in cands:
            cand_dict = {"center_x": c["center_x"], "center_y": c["center_y"], "score": c["score"], "tw": tw, "th": th}
            feats = reranker.extract_candidate_features(
                cand_dict, ref_tmpl, search_img,
                mag_ref_norm, mag_s,
                stage_prior_xy=stage_prior
            )
            dist_to_gt = float(math.hypot(c["center_x"] - gt_x, c["center_y"] - gt_y))
            is_correct = bool(dist_to_gt <= 5.0)

            sample_candidates.append({
                "rank": c["rank"],
                "center_x": c["center_x"],
                "center_y": c["center_y"],
                "dist_to_gt": dist_to_gt,
                "is_correct": is_correct,
                "features": feats
            })

        dataset_records.append({
            "sample_id": sid,
            "architecture": arch,
            "gt_x": gt_x, "gt_y": gt_y,
            "stage_prior": stage_prior,
            "candidates": sample_candidates
        })

    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, indent=2)

    print(f"[OK] Failure Replay dataset built and saved to: {out_json}")
    return dataset_records

if __name__ == "__main__":
    build_failure_replay_dataset()
