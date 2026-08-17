"""
Confusion Map & Ambiguity Scoring Engine for Applied Materials SEM Challenge.

Provides explainable failure analysis:
- Generates similarity heatmaps (Confusion Maps) showing all potential pattern match locations
- Computes Ambiguity Score ratio (Top2 / Top1 confidence)
- Classifies root-cause failure modes (Repeated-pattern ambiguity vs Low-SNR vs Boundary edge)
- Visualizes confusion maps overlaid with Ground-Truth and Predictions
"""

from __future__ import annotations
import math
import numpy as np
import cv2

def compute_ambiguity_metrics(loc_result: dict, topology_info: dict | None = None) -> dict:
    """Calculates ambiguity score, peak ratio, and failure root cause from localization results."""
    candidates = loc_result.get("top_k_candidates", [])
    if len(candidates) == 0:
        return {
            "ambiguity_score": 1.0,
            "peak_ratio": 1.0,
            "failure_reason": "NO_MATCH_FOUND",
            "is_ambiguous": True
        }

    top1 = candidates[0]
    top1_score = top1["score"]

    if len(candidates) > 1:
        top2_score = candidates[1]["score"]
        peak_ratio = float(top2_score / (top1_score + 1e-6))
    else:
        top2_score = 0.0
        peak_ratio = 0.0

    ambiguity_score = float(np.clip(peak_ratio, 0.0, 1.0))

    # Root Cause Classification
    if top1_score < 0.35:
        reason = "LOW_SIGNAL_NOISE"
    elif peak_ratio > 0.85:
        reason = "REPEATED_PATTERN_AMBIGUITY"
    elif top1["center_x"] < 30 or top1["center_x"] > 970 or top1["center_y"] < 30 or top1["center_y"] > 970:
        reason = "BORDER_EDGE_DISTORTION"
    else:
        reason = "SUCCESS_HIGH_CONFIDENCE"

    return {
        "ambiguity_score": ambiguity_score,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "peak_ratio": peak_ratio,
        "failure_reason": reason,
        "is_ambiguous": bool(ambiguity_score > 0.80),
    }

def render_confusion_map(
    search_img: np.ndarray,
    similarity_map: np.ndarray,
    gt_x: float | None = None,
    gt_y: float | None = None,
    pred_x: float | None = None,
    pred_y: float | None = None,
    candidates: list[dict] | None = None
) -> np.ndarray:
    """Renders a colorized Confusion Map overlaid on the Search image with GT and candidate markers."""
    h_s, w_s = search_img.shape
    h_m, w_m = similarity_map.shape

    # Pad similarity map back to search image dimensions
    pad_y = (h_s - h_m) // 2
    pad_x = (w_s - w_m) // 2
    
    padded_map = np.zeros((h_s, w_s), dtype=np.float32)
    padded_map[pad_y:pad_y + h_m, pad_x:pad_x + w_m] = similarity_map

    # Normalize similarity to 0-255
    map_min, map_max = padded_map.min(), padded_map.max()
    if map_max - map_min > 1e-6:
        norm_map = ((padded_map - map_min) / (map_max - map_min) * 255.0).astype(np.uint8)
    else:
        norm_map = np.zeros((h_s, w_s), dtype=np.uint8)

    # Colorize heatmap
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

    # Blend with grayscale search image
    search_bgr = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(search_bgr, 0.5, heatmap, 0.5, 0)

    # Draw Ground Truth (Green circle)
    if gt_x is not None and gt_y is not None:
        cv2.circle(overlay, (int(round(gt_x)), int(round(gt_y))), 12, (0, 255, 0), 2)
        cv2.drawMarker(overlay, (int(round(gt_x)), int(round(gt_y))), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)

    # Draw Predicted Top-1 (Red circle)
    if pred_x is not None and pred_y is not None:
        cv2.circle(overlay, (int(round(pred_x)), int(round(pred_y))), 10, (0, 0, 255), 2)
        cv2.drawMarker(overlay, (int(round(pred_x)), int(round(pred_y))), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)

    # Draw Top-K Candidates (Yellow circles)
    if candidates:
        for cand in candidates[1:]: # Skip Top-1
            cx, cy = int(round(cand["center_x"])), int(round(cand["center_y"]))
            cv2.circle(overlay, (cx, cy), 6, (0, 255, 255), 1)

    return overlay
