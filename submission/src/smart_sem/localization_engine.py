"""
SMART-SEM Unified Industrial Localization Engine (v3).

Features:
1. Multi-Scale ZNCC Primary Correlation (9.5x to 10.5x)
2. Non-Maximum Suppression (NMS) Top-K Candidate Extraction
3. Ambiguity Confidence Margin Calculation: ACR = Score(Peak 1) / Score(Peak 2)
4. Multi-Feature Candidate Re-Ranker:
   - ZNCC Correlation Score
   - Topology Consistency Score (TCS) (Line & Corner Density Verification)
   - Sobel Gradient Cross-Correlation
   - 2D Kinematic Kalman Filter Stage Tracker & Mahalanobis Distance Prior
5. 2D Parabolic Sub-Pixel Peak Regression
6. Spatial Shannon Entropy & Softmax Probability Distribution
"""

from __future__ import annotations
import math
import numpy as np
import cv2

from src.smart_sem.candidate_reranker import CandidateReRanker
from src.smart_sem.kalman_memory import StageKalmanTracker

def subpixel_refine_2d(val_map: np.ndarray, max_x_idx: int, max_y_idx: int) -> tuple[float, float]:
    """Refines discrete peak index (x, y) to sub-pixel coordinates using 2D parabolic quadratic fitting."""
    h, w = val_map.shape
    if max_x_idx <= 0 or max_x_idx >= w - 1 or max_y_idx <= 0 or max_y_idx >= h - 1:
        return float(max_x_idx), float(max_y_idx)

    patch = val_map[max_y_idx - 1:max_y_idx + 2, max_x_idx - 1:max_x_idx + 2]
    
    denom_x = 2.0 * (2.0 * patch[1, 1] - patch[1, 0] - patch[1, 2]) + 1e-7
    denom_y = 2.0 * (2.0 * patch[1, 1] - patch[0, 1] - patch[2, 1]) + 1e-7
    
    dx = (patch[1, 2] - patch[1, 0]) / denom_x
    dy = (patch[2, 1] - patch[0, 1]) / denom_y

    sub_x = float(max_x_idx + np.clip(dx, -0.5, 0.5))
    sub_y = float(max_y_idx + np.clip(dy, -0.5, 0.5))
    return sub_x, sub_y

def compute_spatial_entropy(similarity_map: np.ndarray) -> float:
    """Computes normalized spatial Shannon entropy of the correlation heatmap."""
    if similarity_map is None or similarity_map.size == 0:
        return 0.0
    pos_map = np.clip(similarity_map, 0.0, None)
    total = float(np.sum(pos_map))
    if total < 1e-6:
        return 0.0
    p = pos_map / total
    nz_p = p[p > 1e-7]
    h = -float(np.sum(nz_p * np.log2(nz_p)))
    max_h = np.log2(float(similarity_map.size))
    return float(h / max_h) if max_h > 0 else 0.0

def smart_sem_localize(
    reference_img: np.ndarray,
    search_img: np.ndarray,
    scales: tuple[float, ...] = (9.5, 9.8, 10.0, 10.2, 10.5),
    top_k: int = 25,
    nms_radius: int = 5,
    stage_prior_xy: tuple[float, float] | None = None,
    kalman_tracker: StageKalmanTracker | None = None,
) -> dict:
    """
    Unified multi-scale scale-aware localization engine with learned re-ranking and sub-pixel fitting.
    """
    ref_h, ref_w = reference_img.shape
    search_h, search_w = search_img.shape

    best_match = None
    best_score_map = None

    for scale in scales:
        tw = max(int(round(ref_w / scale)), 1)
        th = max(int(round(ref_h / scale)), 1)

        if tw >= search_w or th >= search_h:
            continue

        template = cv2.resize(reference_img, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if best_match is None or max_val > best_match["score"]:
            sub_x, sub_y = subpixel_refine_2d(res, max_loc[0], max_loc[1])
            best_match = {
                "x": sub_x + tw / 2.0,
                "y": sub_y + th / 2.0,
                "score": float(max_val),
                "scale": scale,
                "template_w": tw,
                "template_h": th,
            }
            best_score_map = res.copy()

    # Extract Top-K candidates via Non-Maximum Suppression (NMS)
    candidates = []
    if best_score_map is not None and best_match is not None:
        map_copy = best_score_map.copy()
        tw, th = best_match["template_w"], best_match["template_h"]
        
        # 1. Global extraction
        for rank in range(1, top_k + 1):
            _, val, _, loc = cv2.minMaxLoc(map_copy)
            if val < 0.1 or math.isnan(val):
                break
            mx, my = loc
            sx, sy = subpixel_refine_2d(best_score_map, mx, my)
            cx = sx + tw / 2.0
            cy = sy + th / 2.0

            candidates.append({
                "rank": rank,
                "center_x": float(cx),
                "center_y": float(cy),
                "score": float(val),
                "top_left": (int(mx), int(my)),
                "tw": tw,
                "th": th,
            })

            y_min, y_max = max(0, my - nms_radius), min(map_copy.shape[0], my + nms_radius + 1)
            x_min, x_max = max(0, mx - nms_radius), min(map_copy.shape[1], mx + nms_radius + 1)
            map_copy[y_min:y_max, x_min:x_max] = -1.0

        # 2. Local Stage-Gated ROI extraction (if stage prior is available)
        if stage_prior_xy is not None:
            sx_p, sy_p = stage_prior_xy
            roi_half = 70
            rx0 = max(0, int(sx_p - roi_half - tw / 2.0))
            ry0 = max(0, int(sy_p - roi_half - th / 2.0))
            rx1 = min(best_score_map.shape[1], int(sx_p + roi_half - tw / 2.0))
            ry1 = min(best_score_map.shape[0], int(sy_p + roi_half - th / 2.0))

            if rx1 > rx0 and ry1 > ry0:
                roi_res = best_score_map[ry0:ry1, rx0:rx1].copy()
                local_nms_radius = 3  # Matches FinFET fin pitch (3.0 px)
                for _ in range(15):
                    _, r_val, _, r_loc = cv2.minMaxLoc(roi_res)
                    if r_val < 0.1 or math.isnan(r_val):
                        break
                    gmx = rx0 + r_loc[0]
                    gmy = ry0 + r_loc[1]
                    sub_x, sub_y = subpixel_refine_2d(best_score_map, gmx, gmy)
                    rcx = sub_x + tw / 2.0
                    rcy = sub_y + th / 2.0

                    if not any(math.hypot(rcx - c["center_x"], rcy - c["center_y"]) < 2.0 for c in candidates):
                        candidates.append({
                            "rank": len(candidates) + 1,
                            "center_x": float(rcx),
                            "center_y": float(rcy),
                            "score": float(r_val),
                            "top_left": (int(gmx), int(gmy)),
                            "tw": tw,
                            "th": th,
                        })

                    roi_res[max(0, r_loc[1]-local_nms_radius):min(roi_res.shape[0], r_loc[1]+local_nms_radius+1),
                            max(0, r_loc[0]-local_nms_radius):min(roi_res.shape[1], r_loc[0]+local_nms_radius+1)] = -1.0

    entropy = compute_spatial_entropy(best_score_map)

    # Compute Ambiguity Confidence Ratio (ACR)
    if len(candidates) > 1:
        top1_score = candidates[0]["score"]
        top2_score = candidates[1]["score"]
        peak_ratio = float(top2_score / (top1_score + 1e-6))
        confidence_margin = float(top1_score / (top2_score + 1e-6))
        peak_sep = float(math.hypot(
            candidates[0]["center_x"] - candidates[1]["center_x"],
            candidates[0]["center_y"] - candidates[1]["center_y"]
        ))
    else:
        top1_score = candidates[0]["score"] if candidates else 0.0
        top2_score = 0.0
        peak_ratio = 0.0
        confidence_margin = 999.0
        peak_sep = 999.0

    # Apply Multi-Feature Candidate Re-Ranker
    reranker = CandidateReRanker()
    ref_template = cv2.resize(reference_img, (best_match["template_w"], best_match["template_h"]), interpolation=cv2.INTER_AREA) if best_match else reference_img

    best_cand, reranked_candidates = reranker.rerank(
        candidates,
        ref_template=ref_template,
        search_img=search_img,
        stage_prior_xy=stage_prior_xy,
        kalman_tracker=kalman_tracker,
        confidence_margin=confidence_margin
    )

    final_pred_x = best_cand.get("center_x", best_match["x"] if best_match else 500.0)
    final_pred_y = best_cand.get("center_y", best_match["y"] if best_match else 500.0)
    final_score = best_cand.get("score", best_match["score"] if best_match else 0.0)

    # Probability distribution across candidates (Softmax with T=0.08)
    scores = np.array([c.get("ranking_score", c["score"]) for c in reranked_candidates]) if reranked_candidates else np.array([0.0])
    exp_s = np.exp((scores - np.max(scores)) / 0.08)
    probs = (exp_s / np.sum(exp_s)).tolist()

    for c, p in zip(reranked_candidates, probs):
        c["probability"] = float(p)

    # Classify ambiguity
    if confidence_margin < 1.05:
        ambiguity_class = "REPEATED_PATTERN_AMBIGUITY"
    elif top1_score < 0.35:
        ambiguity_class = "LOW_SIGNAL_NOISE"
    elif entropy > 0.85:
        ambiguity_class = "HIGH_ENTROPY_DIFFUSE_MATCH"
    else:
        ambiguity_class = "HIGH_CONFIDENCE_UNIQUE_MATCH"

    return {
        "pred_x": final_pred_x,
        "pred_y": final_pred_y,
        "confidence": final_score,
        "scale": best_match["scale"] if best_match else 10.0,
        "top_k_candidates": reranked_candidates,
        "similarity_map": best_score_map,
        "entropy": entropy,
        "peak_ratio": peak_ratio,
        "confidence_margin": confidence_margin,
        "peak_separation_px": peak_sep,
        "ambiguity_class": ambiguity_class,
        "is_ambiguous": bool(confidence_margin < 1.05),
    }
