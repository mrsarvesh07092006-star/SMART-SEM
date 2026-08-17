"""
SMART-SEM Advanced Edge-Aware & Scale-Aware Localization Engine.

Features:
- Dual Intensity + Sobel Gradient Fusion (resolves periodic pattern ambiguity)
- Explicit 10:1 physical scale handling with multi-scale bracket search
- Sub-pixel quadratic interpolation around similarity peak
- Rotation compensation (-1.5 deg to +1.5 deg)
- Multi-hypothesis Top-K candidate generation using Non-Maximum Suppression (NMS)
"""

from __future__ import annotations
import math
import numpy as np
import cv2

def subpixel_refine_2d(val_map: np.ndarray, max_x_idx: int, max_y_idx: int) -> tuple[float, float]:
    """Refines discrete peak index (x, y) to sub-pixel coordinates using 2D parabolic fitting."""
    h, w = val_map.shape
    if max_x_idx <= 0 or max_x_idx >= w - 1 or max_y_idx <= 0 or max_y_idx >= h - 1:
        return float(max_x_idx), float(max_y_idx)

    patch = val_map[max_y_idx - 1:max_y_idx + 2, max_x_idx - 1:max_x_idx + 2]
    
    dx = (patch[1, 2] - patch[1, 0]) / (2.0 * (2.0 * patch[1, 1] - patch[1, 0] - patch[1, 2]) + 1e-7)
    dy = (patch[2, 1] - patch[0, 1]) / (2.0 * (2.0 * patch[1, 1] - patch[0, 1] - patch[2, 1]) + 1e-7)

    sub_x = float(max_x_idx + np.clip(dx, -0.5, 0.5))
    sub_y = float(max_y_idx + np.clip(dy, -0.5, 0.5))
    return sub_x, sub_y

def extract_top_k_candidates(score_map: np.ndarray, template_w: int, template_h: int, top_k: int = 5, min_dist_px: float = 15.0) -> list[dict]:
    """Extracts Top-K local peaks from similarity map with non-maximum suppression (NMS)."""
    candidates = []
    map_copy = score_map.copy()
    h, w = map_copy.shape

    for rank in range(1, top_k + 1):
        _, max_val, _, max_loc = cv2.minMaxLoc(map_copy)
        if max_val < -1.0 or math.isnan(max_val):
            break

        mx, my = max_loc
        sub_x, sub_y = subpixel_refine_2d(score_map, mx, my)

        cx = sub_x + template_w / 2.0
        cy = sub_y + template_h / 2.0

        candidates.append({
            "rank": rank,
            "center_x": float(cx),
            "center_y": float(cy),
            "score": float(max_val),
            "top_left": (int(mx), int(my)),
        })

        r = int(min_dist_px)
        y_min, y_max = max(0, my - r), min(h, my + r + 1)
        x_min, x_max = max(0, mx - r), min(w, mx + r + 1)
        map_copy[y_min:y_max, x_min:x_max] = -1.0

    return candidates

def smart_sem_localize(
    reference_img: np.ndarray,
    search_img: np.ndarray,
    scales: tuple[float, ...] = (9.5, 9.8, 10.0, 10.2, 10.5),
    rotations_deg: tuple[float, ...] = (0.0,),
    top_k: int = 5,
) -> dict:
    """
    Runs multi-scale, edge-enhanced ZNCC localization on reference and search images.
    Combines intensity matching with Sobel gradient magnitude matching.
    """
    ref_h, ref_w = reference_img.shape
    search_h, search_w = search_img.shape

    # Compute Sobel edge maps for structural clarity
    ref_grad_x = cv2.Sobel(reference_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    ref_grad_y = cv2.Sobel(reference_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    ref_grad = np.sqrt(ref_grad_x**2 + ref_grad_y**2)

    search_grad_x = cv2.Sobel(search_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    search_grad_y = cv2.Sobel(search_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    search_grad = np.sqrt(search_grad_x**2 + search_grad_y**2)

    best_match = None
    best_score_map = None

    for angle in rotations_deg:
        if abs(angle) > 1e-3:
            M = cv2.getRotationMatrix2D((ref_w / 2.0, ref_h / 2.0), angle, 1.0)
            rot_ref = cv2.warpAffine(reference_img, M, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            rot_grad = cv2.warpAffine(ref_grad, M, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            rot_ref = reference_img
            rot_grad = ref_grad

        for scale in scales:
            tw = max(int(round(ref_w / scale)), 1)
            th = max(int(round(ref_h / scale)), 1)

            if tw >= search_w or th >= search_h:
                continue

            tmpl_intensity = cv2.resize(rot_ref, (tw, th), interpolation=cv2.INTER_AREA)
            tmpl_grad = cv2.resize(rot_grad, (tw, th), interpolation=cv2.INTER_AREA)

            res_intensity = cv2.matchTemplate(search_img, tmpl_intensity, cv2.TM_CCOEFF_NORMED)
            res_grad = cv2.matchTemplate(search_grad.astype(np.uint8), tmpl_grad.astype(np.uint8), cv2.TM_CCOEFF_NORMED)

            # Dual Intensity + Sobel Gradient Fusion (60% intensity + 40% gradient)
            fused_res = 0.60 * res_intensity + 0.40 * res_grad

            _, max_val, _, max_loc = cv2.minMaxLoc(fused_res)

            if best_match is None or max_val > best_match["score"]:
                sub_x, sub_y = subpixel_refine_2d(fused_res, max_loc[0], max_loc[1])
                best_match = {
                    "x": sub_x + tw / 2.0,
                    "y": sub_y + th / 2.0,
                    "score": float(max_val),
                    "scale": scale,
                    "rotation_deg": angle,
                    "template_w": tw,
                    "template_h": th,
                }
                best_score_map = fused_res.copy()

    # Extract Top-K candidates
    if best_score_map is not None and best_match is not None:
        candidates = extract_top_k_candidates(
            best_score_map,
            best_match["template_w"],
            best_match["template_h"],
            top_k=top_k
        )
    else:
        candidates = []

    # Center preference tie-breaker for close candidates
    search_center_x, search_center_y = search_w / 2.0, search_h / 2.0
    if len(candidates) > 1 and (candidates[0]["score"] - candidates[1]["score"]) < 0.03:
        c1_dist = math.hypot(candidates[0]["center_x"] - search_center_x, candidates[0]["center_y"] - search_center_y)
        c2_dist = math.hypot(candidates[1]["center_x"] - search_center_x, candidates[1]["center_y"] - search_center_y)
        if c2_dist < c1_dist:
            candidates[0], candidates[1] = candidates[1], candidates[0]
            best_match["x"] = candidates[0]["center_x"]
            best_match["y"] = candidates[0]["center_y"]
            best_match["score"] = candidates[0]["score"]

    return {
        "pred_x": best_match["x"] if best_match else 500.0,
        "pred_y": best_match["y"] if best_match else 500.0,
        "confidence": best_match["score"] if best_match else 0.0,
        "scale": best_match["scale"] if best_match else 10.0,
        "rotation_deg": best_match["rotation_deg"] if best_match else 0.0,
        "top_k_candidates": candidates,
        "similarity_map": best_score_map,
    }
