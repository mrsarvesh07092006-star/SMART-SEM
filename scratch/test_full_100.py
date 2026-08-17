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

def evaluate_all(dist_weight=0.06, shear_bracket=True):
    errors = []
    p5, p2, p1 = 0, 0, 0
    
    for row in rows:
        sample_id = row['sample_id']
        ref = cv2.imread(os.path.join('results/dataset', row['reference_path']), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join('results/dataset', row['search_path']), cv2.IMREAD_GRAYSCALE)
        gt = (float(row['gt_x']), float(row['gt_y']))
        obs = (float(row['obs_gt_x']), float(row['obs_gt_y'])) if 'obs_gt_x' in row else None
        
        candidates = []
        best_score_map = None
        best_tw, best_th = 100, 100
        best_val_global = -1.0
        
        scales = [9.5, 9.8, 10.0, 10.2, 10.5]
        shears = [-0.03, 0.0, 0.03] if shear_bracket else [0.0]
        
        for scale in scales:
            tw = max(10, int(round(ref.shape[1] / scale)))
            th = max(10, int(round(ref.shape[0] / scale)))
            base_tmpl = cv2.resize(ref, (tw, th), interpolation=cv2.INTER_AREA)
            
            for s_y in shears:
                if s_y != 0.0:
                    M = np.float32([[1, 0, 0], [s_y, 1, 0]])
                    tmpl = cv2.warpAffine(base_tmpl, M, (tw, th))
                else:
                    tmpl = base_tmpl
                    
                res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                if max_val > best_val_global:
                    best_val_global = max_val
                    best_score_map = res
                    best_tw, best_th = tw, th
                    
        # Extract Global Candidates
        map_copy = best_score_map.copy()
        for rank in range(1, 20):
            _, val, _, loc = cv2.minMaxLoc(map_copy)
            if val < 0.1: break
            mx, my = loc
            sx, sy = subpixel_refine_2d(best_score_map, mx, my)
            candidates.append({
                "rank": rank,
                "center_x": float(sx + best_tw / 2.0),
                "center_y": float(sy + best_th / 2.0),
                "score": float(val),
                "top_left": (int(mx), int(my)),
                "tw": best_tw,
                "th": best_th,
            })
            map_copy[max(0, my-12):min(map_copy.shape[0], my+13),
                     max(0, mx-12):min(map_copy.shape[1], mx+13)] = -1.0
                     
        # Extract Local Stage-Gated Candidates
        if obs is not None:
            sx_p, sy_p = obs
            roi_half = 65
            rx0 = max(0, int(sx_p - roi_half - best_tw / 2.0))
            ry0 = max(0, int(sy_p - roi_half - best_th / 2.0))
            rx1 = min(best_score_map.shape[1], int(sx_p + roi_half - best_tw / 2.0))
            ry1 = min(best_score_map.shape[0], int(sy_p + roi_half - best_th / 2.0))
            
            if rx1 > rx0 and ry1 > ry0:
                roi_res = best_score_map[ry0:ry1, rx0:rx1].copy()
                for _ in range(15):
                    _, r_val, _, r_loc = cv2.minMaxLoc(roi_res)
                    if r_val < 0.1: break
                    gmx = rx0 + r_loc[0]
                    gmy = ry0 + r_loc[1]
                    sub_x, sub_y = subpixel_refine_2d(best_score_map, gmx, gmy)
                    rcx = sub_x + best_tw / 2.0
                    rcy = sub_y + best_th / 2.0
                    
                    if not any(math.hypot(rcx - c["center_x"], rcy - c["center_y"]) < 2.0 for c in candidates):
                        candidates.append({
                            "rank": len(candidates) + 1,
                            "center_x": float(rcx),
                            "center_y": float(rcy),
                            "score": float(r_val),
                            "top_left": (int(gmx), int(gmy)),
                            "tw": best_tw,
                            "th": best_th,
                        })
                    roi_res[max(0, r_loc[1]-3):min(roi_res.shape[0], r_loc[1]+4),
                            max(0, r_loc[0]-3):min(roi_res.shape[1], r_loc[0]+4)] = -1.0

        tmpl_best = cv2.resize(ref, (best_tw, best_th), interpolation=cv2.INTER_AREA)
        reranker = CandidateReRanker(weights={
            "w_score": 1.0,
            "w_tcs": 0.30,
            "w_junc": 0.25,
            "w_grad_corr": 0.25,
            "w_stage_dist": dist_weight,
            "w_var_ratio": 0.20,
        })
        
        best, ranked = reranker.rerank(candidates, tmpl_best, search, stage_prior_xy=obs, confidence_margin=1.01)
        err = math.hypot(best["center_x"] - gt[0], best["center_y"] - gt[1])
        errors.append(err)
        if err <= 5.0: p5 += 1
        if err <= 2.0: p2 += 1
        if err <= 1.0: p1 += 1
        
    print(f"DistWeight={dist_weight:.2f}, Shears={shear_bracket} -> Pass@5px: {p5}/{len(rows)} ({p5/len(rows)*100.1:.1f}%) | Median: {np.median(errors):.2f}px | Mean: {np.mean(errors):.2f}px | Worst: {np.max(errors):.2f}px")
    return errors

evaluate_all(dist_weight=0.08, shear_bracket=True)
