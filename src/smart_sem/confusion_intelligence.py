"""
SMART-SEM Layer 4: Confusion Intelligence Engine.

Classifies search image into structural risk zones:
- Repeated Pattern Zones (High ambiguity risk due to periodic cells)
- Unique Feature Zones (High confidence, distinct edges or separator strips)
- Low-Contrast Risk Zones (High noise sensitivity, featureless background)

Renders multi-color visual confusion maps and outputs structural risk analysis.
"""

from __future__ import annotations
import numpy as np
import cv2

def segment_confusion_risk_zones(search_img: np.ndarray, similarity_map: np.ndarray | None = None) -> dict:
    """
    Segments search image into Unique, Repeated, and Low-Contrast Risk zones.
    Returns segmented mask and percentage metrics.
    """
    h, w = search_img.shape
    img_float = search_img.astype(np.float32)

    # Gradient magnitude
    grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Local variance / texture density
    blur = cv2.GaussianBlur(img_float, (9, 9), 0)
    local_var = cv2.GaussianBlur((img_float - blur)**2, (9, 9), 0)

    # Define thresholds
    low_contrast_mask = (grad_mag < 8.0) & (local_var < 15.0)
    high_edge_mask = (grad_mag > 25.0)
    repeated_pattern_mask = (~low_contrast_mask) & (~high_edge_mask)

    total_px = float(h * w)
    unique_pct = float(np.sum(high_edge_mask) / total_px * 100.0)
    repeated_pct = float(np.sum(repeated_pattern_mask) / total_px * 100.0)
    low_contrast_pct = float(np.sum(low_contrast_mask) / total_px * 100.0)

    # Create 3-channel zone map (RGB)
    # Red = Repeated Pattern Zone, Green = Unique Feature Zone, Blue = Low-Contrast Risk Zone
    zone_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    zone_rgb[repeated_pattern_mask] = [220, 38, 38]   # Red
    zone_rgb[high_edge_mask] = [22, 163, 74]        # Green
    zone_rgb[low_contrast_mask] = [37, 99, 235]       # Blue

    return {
        "unique_zone_pct": unique_pct,
        "repeated_zone_pct": repeated_pct,
        "low_contrast_zone_pct": low_contrast_pct,
        "primary_scene_type": "PERIODIC_ARRAY" if repeated_pct > 50.0 else ("UNIQUE_COMPLEX" if unique_pct > 30.0 else "FEATURELESS_NOISY"),
        "zone_segmentation_rgb": zone_rgb,
    }

def render_advanced_confusion_intelligence_map(
    search_img: np.ndarray,
    similarity_map: np.ndarray,
    ambiguity_info: dict,
    gt_xy: tuple[float, float] | None = None,
    pred_xy: tuple[float, float] | None = None
) -> np.ndarray:
    """
    Renders comprehensive Confusion Intelligence visual dashboard image.
    Side-by-side overlay showing similarity heatmap and risk zone segmentation.
    """
    h_s, w_s = search_img.shape
    search_bgr = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

    # 1. Similarity Heatmap Overlay
    if similarity_map is not None:
        h_m, w_m = similarity_map.shape
        pad_y = (h_s - h_m) // 2
        pad_x = (w_s - w_m) // 2
        padded_map = np.zeros((h_s, w_s), dtype=np.float32)
        padded_map[pad_y:pad_y + h_m, pad_x:pad_x + w_m] = similarity_map
        m_min, m_max = padded_map.min(), padded_map.max()
        norm_map = (((padded_map - m_min) / (m_max - m_min + 1e-6)) * 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        heatmap_overlay = cv2.addWeighted(search_bgr, 0.45, heatmap, 0.55, 0)
    else:
        heatmap_overlay = search_bgr.copy()

    # 2. Risk Zone Segmentation Overlay
    zone_res = segment_confusion_risk_zones(search_img, similarity_map)
    zone_rgb = zone_res["zone_segmentation_rgb"]
    zone_bgr = cv2.cvtColor(zone_rgb, cv2.COLOR_RGB2BGR)
    zone_overlay = cv2.addWeighted(search_bgr, 0.50, zone_bgr, 0.50, 0)

    # Draw GT (Green) and Pred (Red) on heatmap overlay
    if gt_xy:
        cv2.circle(heatmap_overlay, (int(round(gt_xy[0])), int(round(gt_xy[1]))), 12, (0, 255, 0), 2)
        cv2.drawMarker(heatmap_overlay, (int(round(gt_xy[0])), int(round(gt_xy[1]))), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
    if pred_xy:
        cv2.circle(heatmap_overlay, (int(round(pred_xy[0])), int(round(pred_xy[1]))), 10, (0, 0, 255), 2)

    # Draw candidates
    candidates = ambiguity_info.get("top_candidates_distribution", [])
    for cand in candidates[1:]:
        cx, cy = int(round(cand["center_x"])), int(round(cand["center_y"]))
        cv2.circle(heatmap_overlay, (cx, cy), 6, (0, 255, 255), 1)

    # Side-by-side dashboard canvas (1000x2000 px)
    combined = np.hstack([heatmap_overlay, zone_overlay])

    # Text annotations
    cv2.putText(combined, "SIMILARITY CONFUSION HEATMAP", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(combined, f"Ambiguity: {ambiguity_info.get('ambiguity_class', 'N/A')}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(combined, "STRUCTURAL RISK ZONES (Red=Repeated, Green=Unique)", (w_s + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, f"Repeated: {zone_res['repeated_zone_pct']:.1f}% | Unique: {zone_res['unique_zone_pct']:.1f}%", (w_s + 30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return combined
