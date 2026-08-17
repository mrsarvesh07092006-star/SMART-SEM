"""
SMART-SEM Layer 5: Memory-Guided Topology Discovery & Strategy Adaptation Engine.

Analyzes structural patterns from high-magnification Reference patches:
- 2D FFT Power Spectrum Pitch & Orientation Analysis
- Periodicity & ambiguity risk index
- Dynamic Matching Strategy Generation (adapts ZNCC, Edge, and Phase correlation weights based on wafer topology)
"""

from __future__ import annotations
import math
import numpy as np
import cv2

def discover_topology(reference_img: np.ndarray, pixel_size_nm: float = 1.0) -> dict:
    """
    Analyzes 2D FFT spectrum and spatial gradients of reference_img to discover wafer topology
    and adapt localization parameters.
    """
    img_float = reference_img.astype(np.float32)
    h, w = img_float.shape

    # 1. 2D FFT Power Spectrum Analysis
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    windowed = (img_float - np.mean(img_float)) * window
    
    dft = np.fft.fft2(windowed)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift) ** 2

    # Mask out DC component center
    cy, cx = h // 2, w // 2
    cv2.circle(magnitude_spectrum, (cx, cy), 5, 0, -1)

    max_val = np.max(magnitude_spectrum)
    norm_spectrum = magnitude_spectrum / (max_val + 1e-7)

    # Spectral peaks above 20% threshold
    peaks_mask = norm_spectrum > 0.20
    peak_y, peak_x = np.where(peaks_mask)

    freq_distances = []
    orientations = []

    for py, px in zip(peak_y, peak_x):
        dy = py - cy
        dx = px - cx
        dist = math.hypot(dx, dy)
        if dist > 3:
            freq_distances.append(dist)
            angle = (math.degrees(math.atan2(dy, dx)) + 180.0) % 180.0
            orientations.append(angle)

    if len(freq_distances) > 0:
        dom_freq = float(np.median(freq_distances))
        pitch_px = float(w / dom_freq) if dom_freq > 0 else float(w)
        pitch_nm = float(pitch_px * pixel_size_nm)
        periodicity_score = float(np.clip(len(freq_distances) / 30.0, 0.1, 0.95))
        dom_orientation_deg = float(np.median(orientations))
    else:
        pitch_px = float(w)
        pitch_nm = float(w * pixel_size_nm)
        periodicity_score = 0.05
        dom_orientation_deg = 0.0

    # Gradient Analysis
    grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)
    avg_grad = float(np.mean(mag))

    # Structural Classification
    if periodicity_score > 0.55:
        pattern_class = "DRAM_PERIODIC_MATRIX" if periodicity_score > 0.75 else "FINFET_GRATING_ARRAY"
    elif avg_grad > 15.0:
        pattern_class = "LOGIC_COMPLEX_EDGE"
    else:
        pattern_class = "ISOLATED_STRUCTURE"

    # DYNAMIC STRATEGY ADAPTATION BASED ON TOPOLOGY
    # Periodic patterns require heavier Sobel edge weighting + Phase correlation to prevent false intensity matches
    if periodicity_score > 0.60:
        w_zncc = 0.30
        w_edge = 0.45
        w_phase = 0.25
        top_k_candidates = 7
        ambiguity_cutoff = 0.82
    elif pattern_class == "LOGIC_COMPLEX_EDGE":
        w_zncc = 0.50
        w_edge = 0.35
        w_phase = 0.15
        top_k_candidates = 5
        ambiguity_cutoff = 0.88
    else: # Default
        w_zncc = 0.40
        w_edge = 0.35
        w_phase = 0.25
        top_k_candidates = 5
        ambiguity_cutoff = 0.85

    adaptive_strategy = {
        "w_zncc": w_zncc,
        "w_edge": w_edge,
        "w_phase": w_phase,
        "top_k_candidates": top_k_candidates,
        "ambiguity_cutoff": ambiguity_cutoff,
        "scales_bracket": (9.5, 9.8, 10.0, 10.2, 10.5),
    }

    return {
        "pitch_px_ref": pitch_px,
        "pitch_nm": pitch_nm,
        "pitch_px_search_est": pitch_px / 10.0,
        "orientation_deg": dom_orientation_deg,
        "periodicity_index": periodicity_score,
        "mean_gradient_magnitude": avg_grad,
        "pattern_class": pattern_class,
        "is_periodic_ambiguous": bool(periodicity_score > 0.55),
        "adaptive_strategy": adaptive_strategy,
    }
