"""
SMART-SEM FinFET Structure Tensor & Phase Disambiguation Engine (Pillar A).

Computes:
1. 2D Structure Tensor J = [ [Ix^2, Ix*Iy], [Ix*Iy, Iy^2] ]
2. Coherence Energy C = ((lambda1 - lambda2) / (lambda1 + lambda2 + eps))^2
3. Cross-Grating Junction Nodes (Fin-Gate line crossings)
4. Sub-Pitch Phase Correlation for grating disambiguation
"""

from __future__ import annotations
import math
import numpy as np
import cv2

def compute_structure_tensor_signature(img_patch: np.ndarray) -> dict:
    """Computes structure tensor coherence, dominant orientation, and junction node count."""
    if img_patch.size == 0:
        return {"coherence": 0.0, "orientation_rad": 0.0, "junction_count": 0}

    f = img_patch.astype(np.float32)
    # Gaussian smoothing before gradient
    blurred = cv2.GaussianBlur(f, (3, 3), 1.0)

    # Gradients
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    # Tensor components
    jxx = cv2.GaussianBlur(gx * gx, (5, 5), 1.5)
    jyy = cv2.GaussianBlur(gy * gy, (5, 5), 1.5)
    jxy = cv2.GaussianBlur(gx * gy, (5, 5), 1.5)

    # Eigenvalues of structure tensor
    trace = jxx + jyy
    det = jxx * jyy - jxy * jxy
    discriminant = np.sqrt(np.maximum(0.0, trace * trace - 4.0 * det))
    lambda1 = 0.5 * (trace + discriminant)
    lambda2 = 0.5 * (trace - discriminant)

    # Local Coherence
    coherence = np.mean(((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)) ** 2)

    # Dominant orientation angle
    mean_jxy = float(np.mean(jxy))
    mean_diff = float(np.mean(jxx - jyy))
    dominant_angle = 0.5 * math.atan2(2.0 * mean_jxy, mean_diff + 1e-6)

    # Fin-Gate Crossings: Response in both gradient directions
    th_x = max(15.0, float(np.percentile(np.abs(gx), 70)))
    th_y = max(15.0, float(np.percentile(np.abs(gy), 70)))
    junction_map = (np.abs(gx) >= th_x) & (np.abs(gy) >= th_y)
    junction_count = int(np.sum(junction_map))

    return {
        "coherence": float(coherence),
        "orientation_rad": float(dominant_angle),
        "junction_count": junction_count,
        "lambda1_mean": float(np.mean(lambda1)),
        "lambda2_mean": float(np.mean(lambda2)),
    }

def compute_finfet_junction_similarity(ref_patch: np.ndarray, cand_patch: np.ndarray) -> float:
    """Computes structural junction & orientation similarity between FinFET patches."""
    if ref_patch.shape != cand_patch.shape or ref_patch.size == 0 or cand_patch.size == 0:
        return 0.5

    sig_ref = compute_structure_tensor_signature(ref_patch)
    sig_cand = compute_structure_tensor_signature(cand_patch)

    # Orientation similarity
    d_theta = abs(sig_ref["orientation_rad"] - sig_cand["orientation_rad"])
    angle_sim = math.cos(d_theta)

    # Coherence similarity
    coh_diff = abs(sig_ref["coherence"] - sig_cand["coherence"])
    coh_sim = math.exp(-coh_diff * 4.0)

    # Junction count similarity
    j_ref, j_cand = sig_ref["junction_count"], sig_cand["junction_count"]
    junc_sim = math.exp(-abs(j_ref - j_cand) / (max(j_ref, 1) + 5.0))

    return float(np.clip(0.40 * angle_sim + 0.30 * coh_sim + 0.30 * junc_sim, 0.0, 1.0))
