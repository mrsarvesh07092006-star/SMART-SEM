"""
SMART-SEM Layer 8: Cross-Modal SEM <-> RGB Extension Engine (Bonus).

Demonstrates modality-invariant wafer pattern matching:
- Extracts modality-independent structural edge & topology representation from grayscale SEM and RGB Optical images
- Computes cross-modal structural similarity matrix
- Enables transfer of wafer localization algorithms across imaging modalities
"""

from __future__ import annotations
import numpy as np
import cv2

def convert_rgb_to_structural_representation(rgb_img: np.ndarray) -> np.ndarray:
    """Converts 3-channel RGB optical microscopy image into modality-invariant structural representation."""
    if len(rgb_img.shape) == 3 and rgb_img.shape[2] == 3:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb_img.copy()

    # Structural Sobolev gradient magnitude
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    struct_mag = np.sqrt(gx**2 + gy**2)

    # Normalize to 0-255
    m_min, m_max = struct_mag.min(), struct_mag.max()
    if m_max - m_min > 1e-6:
        norm_struct = ((struct_mag - m_min) / (m_max - m_min) * 255.0).astype(np.uint8)
    else:
        norm_struct = np.zeros_like(gray, dtype=np.uint8)

    return norm_struct

def cross_modal_sem_rgb_match(sem_ref: np.ndarray, rgb_search: np.ndarray, scale: float = 10.0) -> dict:
    """
    Performs cross-modal matching between grayscale SEM reference patch and RGB optical search image.
    """
    sem_struct = convert_rgb_to_structural_representation(sem_ref)
    rgb_struct = convert_rgb_to_structural_representation(rgb_search)

    ref_h, ref_w = sem_struct.shape
    tw = max(int(round(ref_w / scale)), 1)
    th = max(int(round(ref_h / scale)), 1)

    template = cv2.resize(sem_struct, (tw, th), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(rgb_struct, template, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    return {
        "cross_modal_score": float(max_val),
        "pred_x": max_loc[0] + tw / 2.0,
        "pred_y": max_loc[1] + th / 2.0,
        "modality": "SEM_RGB_CROSS_MODAL",
    }
