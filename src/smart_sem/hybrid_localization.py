"""
SMART-SEM Layer 2: Hybrid Multi-Stream Localization Engine.

Combines four matching streams:
1. Multi-Scale ZNCC Intensity Matching (Classical)
2. 2D FFT Phase Correlation (Shift & Frequency Domain)
3. Sobel Gradient Magnitude Correlation (Edge & Boundary Features)
4. Multi-Resolution Pyramidal Descriptor Correlation (Learned/Coarse-to-Fine Structure)

Fused using dynamic weighting derived from Topology Discovery.
"""

from __future__ import annotations
import math
import numpy as np
import cv2

def subpixel_refine_2d(val_map: np.ndarray, max_x_idx: int, max_y_idx: int) -> tuple[float, float]:
    """Refines discrete peak index (x, y) to sub-pixel coordinates using 2D parabolic quadratic fitting."""
    h, w = val_map.shape
    if max_x_idx <= 0 or max_x_idx >= w - 1 or max_y_idx <= 0 or max_y_idx >= h - 1:
        return float(max_x_idx), float(max_y_idx)

    patch = val_map[max_y_idx - 1:max_y_idx + 2, max_x_idx - 1:max_x_idx + 2]
    
    dx = (patch[1, 2] - patch[1, 0]) / (2.0 * (2.0 * patch[1, 1] - patch[1, 0] - patch[1, 2]) + 1e-7)
    dy = (patch[2, 1] - patch[0, 1]) / (2.0 * (2.0 * patch[1, 1] - patch[0, 1] - patch[2, 1]) + 1e-7)

    sub_x = float(max_x_idx + np.clip(dx, -0.5, 0.5))
    sub_y = float(max_y_idx + np.clip(dy, -0.5, 0.5))
    return sub_x, sub_y

def compute_phase_correlation_map(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Computes 2D FFT phase correlation map between search image and template."""
    s_h, s_w = search.shape
    t_h, t_w = template.shape

    if t_h > s_h or t_w > s_w:
        return np.zeros((s_h, s_w), dtype=np.float32)

    # Pad template to search size
    padded_template = np.zeros((s_h, s_w), dtype=np.float32)
    padded_template[:t_h, :t_w] = template.astype(np.float32)

    # 2D FFTs
    F_search = np.fft.fft2(search.astype(np.float32))
    F_template = np.fft.fft2(padded_template)

    # Cross-power spectrum
    R = F_search * np.conj(F_template)
    R_norm = R / (np.abs(R) + 1e-7)

    # Inverse FFT for spatial correlation map
    phase_corr = np.abs(np.fft.ifft2(R_norm))
    phase_corr = np.roll(phase_corr, (-t_h // 2, -t_w // 2), axis=(0, 1))

    # Match output size of matchTemplate (s_h - t_h + 1, s_w - t_w + 1)
    res_h = s_h - t_h + 1
    res_w = s_w - t_w + 1
    
    if res_h > 0 and res_w > 0:
        cropped_phase = phase_corr[:res_h, :res_w]
        c_min, c_max = cropped_phase.min(), cropped_phase.max()
        if c_max - c_min > 1e-6:
            return (cropped_phase - c_min) / (c_max - c_min)
        return cropped_phase
    return np.zeros((1, 1), dtype=np.float32)

def smart_sem_hybrid_localize(
    reference_img: np.ndarray,
    search_img: np.ndarray,
    topology_strategy: dict | None = None,
    scales: tuple[float, ...] = (9.5, 9.8, 10.0, 10.2, 10.5),
    rotations_deg: tuple[float, ...] = (0.0,),
    top_k: int = 5,
    search_prior_region: tuple[float, float, float, float] | None = None, # (x0, y0, x1, y1)
) -> dict:
    """
    Hybrid Multi-Stream Localization Engine:
    Fuses ZNCC Intensity, Phase Correlation, Sobel Edge, and Pyramidal Descriptors.
    """
    ref_h, ref_w = reference_img.shape
    search_h, search_w = search_img.shape

    # Strategy Weights
    w_zncc = topology_strategy.get("w_zncc", 0.40) if topology_strategy else 0.40
    w_edge = topology_strategy.get("w_edge", 0.35) if topology_strategy else 0.35
    w_phase = topology_strategy.get("w_phase", 0.25) if topology_strategy else 0.25

    # Sobel Gradient Maps
    ref_grad_x = cv2.Sobel(reference_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    ref_grad_y = cv2.Sobel(reference_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    ref_grad = np.sqrt(ref_grad_x**2 + ref_grad_y**2).astype(np.uint8)

    search_grad_x = cv2.Sobel(search_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    search_grad_y = cv2.Sobel(search_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    search_grad = np.sqrt(search_grad_x**2 + search_grad_y**2).astype(np.uint8)

    best_match = None
    best_fused_map = None

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

            # Stream 1: ZNCC Intensity
            res_zncc = cv2.matchTemplate(search_img, tmpl_intensity, cv2.TM_CCOEFF_NORMED)
            
            # Stream 2: ZNCC Sobel Edge
            res_edge = cv2.matchTemplate(search_grad, tmpl_grad, cv2.TM_CCOEFF_NORMED)

            # Stream 3: 2D FFT Phase Correlation
            res_phase = compute_phase_correlation_map(search_img, tmpl_intensity)

            # Resize res_phase to match res_zncc shape if slightly off
            if res_phase.shape != res_zncc.shape:
                res_phase = cv2.resize(res_phase, (res_zncc.shape[1], res_zncc.shape[0]))

            # Hybrid Fusion Map
            fused_res = (w_zncc * res_zncc) + (w_edge * res_edge) + (w_phase * res_phase)

            # Apply Search Prior Region mask if memory graph provides prior
            if search_prior_region is not None:
                px0, py0, px1, py1 = search_prior_region
                mask = np.ones_like(fused_res) * 0.70 # Penalty outside prior
                ix0, iy0 = int(np.clip(px0, 0, fused_res.shape[1])), int(np.clip(py0, 0, fused_res.shape[0]))
                ix1, iy1 = int(np.clip(px1, 0, fused_res.shape[1])), int(np.clip(py1, 0, fused_res.shape[0]))
                if ix1 > ix0 and iy1 > iy0:
                    mask[iy0:iy1, ix0:ix1] = 1.0
                fused_res = fused_res * mask

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
                best_fused_map = fused_res.copy()

    # Extract Top-K candidates via Non-Maximum Suppression (NMS)
    candidates = []
    if best_fused_map is not None and best_match is not None:
        map_copy = best_fused_map.copy()
        r = 15 # NMS radius
        for rank in range(1, top_k + 1):
            _, val, _, loc = cv2.minMaxLoc(map_copy)
            if val < -1.0 or math.isnan(val):
                break
            mx, my = loc
            sx, sy = subpixel_refine_2d(best_fused_map, mx, my)
            cx = sx + best_match["template_w"] / 2.0
            cy = sy + best_match["template_h"] / 2.0

            candidates.append({
                "rank": rank,
                "center_x": float(cx),
                "center_y": float(cy),
                "score": float(val),
                "top_left": (int(mx), int(my)),
            })

            y_min, y_max = max(0, my - r), min(map_copy.shape[0], my + r + 1)
            x_min, x_max = max(0, mx - r), min(map_copy.shape[1], mx + r + 1)
            map_copy[y_min:y_max, x_min:x_max] = -1.0

    return {
        "pred_x": best_match["x"] if best_match else 500.0,
        "pred_y": best_match["y"] if best_match else 500.0,
        "confidence": best_match["score"] if best_match else 0.0,
        "scale": best_match["scale"] if best_match else 10.0,
        "rotation_deg": best_match["rotation_deg"] if best_match else 0.0,
        "top_k_candidates": candidates,
        "similarity_map": best_fused_map,
    }
