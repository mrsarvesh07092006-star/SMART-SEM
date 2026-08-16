import csv
import cv2
import os
import math
import sys
import numpy as np

sys.path.insert(0, '.')
from src.smart_sem.localization_engine import subpixel_refine_2d
from src.smart_sem.candidate_reranker import CandidateReRanker

with open('results/dataset/manifest.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

def test_local_nms(pair_id, local_nms_radius=3, n_local=12, dist_weight=0.08):
    row = next(r for r in rows if r['sample_id'] == pair_id)
    ref = cv2.imread(os.path.join('results/dataset', row['reference_path']), cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(os.path.join('results/dataset', row['search_path']), cv2.IMREAD_GRAYSCALE)
    gt = (float(row['gt_x']), float(row['gt_y']))
    obs = (float(row['obs_gt_x']), float(row['obs_gt_y']))
    
    tw, th = 100, 100
    tmpl = cv2.resize(ref, (tw, th), interpolation=cv2.INTER_AREA)
    score_map = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
    
    # Extract candidates
    candidates = []
    # Local ROI
    sx_p, sy_p = obs
    roi_half = 60
    rx0 = max(0, int(sx_p - roi_half - tw / 2.0))
    ry0 = max(0, int(sy_p - roi_half - th / 2.0))
    rx1 = min(score_map.shape[1], int(sx_p + roi_half - tw / 2.0))
    ry1 = min(score_map.shape[0], int(sy_p + roi_half - th / 2.0))
    
    roi_res = score_map[ry0:ry1, rx0:rx1].copy()
    for _ in range(n_local):
        _, r_val, _, r_loc = cv2.minMaxLoc(roi_res)
        if r_val < 0.1: break
        gmx = rx0 + r_loc[0]
        gmy = ry0 + r_loc[1]
        sub_x, sub_y = subpixel_refine_2d(score_map, gmx, gmy)
        rcx = sub_x + tw / 2.0
        rcy = sub_y + th / 2.0
        
        candidates.append({
            "rank": len(candidates) + 1,
            "center_x": float(rcx),
            "center_y": float(rcy),
            "score": float(r_val),
            "top_left": (int(gmx), int(gmy)),
            "tw": tw,
            "th": th
        })
        roi_res[max(0, r_loc[1]-local_nms_radius):min(roi_res.shape[0], r_loc[1]+local_nms_radius+1),
                max(0, r_loc[0]-local_nms_radius):min(roi_res.shape[1], r_loc[0]+local_nms_radius+1)] = -1.0
                
    reranker = CandidateReRanker(weights={
        "w_score": 1.0,
        "w_tcs": 0.30,
        "w_junc": 0.25,
        "w_grad_corr": 0.25,
        "w_stage_dist": dist_weight,
        "w_var_ratio": 0.20,
    })
    
    best, ranked = reranker.rerank(candidates, tmpl, search, stage_prior_xy=obs, confidence_margin=1.01)
    err = math.hypot(best["center_x"] - gt[0], best["center_y"] - gt[1])
    print(f"{pair_id} -> Best: ({best['center_x']:.2f}, {best['center_y']:.2f}) | GT: {gt} | Error: {err:.2f}px")
    return err

for p in ['pair_0013', 'pair_0015', 'pair_0029']:
    test_local_nms(p, local_nms_radius=3, n_local=15, dist_weight=0.08)
