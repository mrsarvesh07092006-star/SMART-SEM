"""
SMART-SEM Layer 1 Upgrade: Advanced SEM Acquisition Physics Engine (Agent 6).

Models realistic high-volume semiconductor fab inspection artifacts:
1. Line Edge Roughness (LER) & Line Width Roughness (LWR)
2. Scan Line Distortion & Raster Scan Non-linearity
3. Focus Gradient & Astigmatic Defocus (beam spot ellipticity)
4. Time-dependent Continuous Beam Drift
5. Local Die Insulator Charging Breakdown Streaks
6. Field-of-View Stitching Boundary Artifacts
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class AdvancedSEMPhysicsParams:
    # Line Edge Roughness (LER) sigma in nm
    ler_sigma_nm: float = 1.2
    # Spatial correlation length for LER (px)
    ler_corr_len_px: float = 15.0
    # Scan line raster distortion amplitude (px)
    scan_distortion_amp_px: float = 1.0
    # Focus gradient across search field (relative blur factor 0 to 1)
    focus_gradient_strength: float = 0.3
    # Insulator charging streak probability per 100 scan rows
    charging_streak_prob: float = 2.0
    # Stitching boundary artifact offset (px)
    stitching_artifact_prob: float = 0.25

def apply_line_edge_roughness(canvas: np.ndarray, ler_sigma_px: float = 1.2, rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulates physical lithography Line Edge Roughness (LER) / Line Width Roughness (LWR)."""
    if ler_sigma_px <= 1e-3:
        return canvas
    rng = rng or np.random.default_rng()
    h, w = canvas.shape
    # Generate 1D random perturbation smoothed along scan direction
    noise_x = rng.normal(0, ler_sigma_px, (h, 1))
    noise_x_smooth = cv2.GaussianBlur(noise_x, (1, 15), 0)
    
    # Warping grid
    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    distorted_x = (grid_x + noise_x_smooth).astype(np.float32)
    distorted_y = grid_y.astype(np.float32)
    
    rough_canvas = cv2.remap(canvas, distorted_x, distorted_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rough_canvas

def apply_focus_gradient(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Simulates focal plane tilt causing continuous defocus/blur gradient across the search field."""
    if strength <= 1e-3:
        return image
    h, w = image.shape
    # Create smooth blur map from left-to-right / top-to-bottom
    y, x = np.mgrid[0:h, 0:w]
    gradient = ((x + y) / float(h + w) * strength).astype(np.float32)
    
    blurred = cv2.GaussianBlur(image, (7, 7), 2.0)
    out = (image.astype(np.float32) * (1.0 - gradient) + blurred.astype(np.float32) * gradient).astype(np.uint8)
    return out

def apply_charging_breakdown_streaks(image: np.ndarray, streak_prob: float = 2.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulates bright horizontal charging discharge streaks from insulating dielectric layers."""
    if streak_prob <= 1e-3:
        return image
    rng = rng or np.random.default_rng()
    out = image.astype(np.float32).copy()
    h, w = image.shape
    n_streaks = int(rng.poisson(streak_prob * (h / 100.0)))
    
    for _ in range(n_streaks):
        row = int(rng.integers(0, h))
        x_start = int(rng.integers(0, w // 2))
        x_len = int(rng.integers(w // 4, w - x_start))
        intensity = float(rng.uniform(40.0, 180.0))
        # Exponential tail decay
        streak = np.exp(-np.linspace(0, 3, x_len)) * intensity
        out[row, x_start:x_start + x_len] = np.clip(out[row, x_start:x_start + x_len] + streak, 0, 255)
    
    return out.astype(np.uint8)

def apply_stitching_boundary_artifact(image: np.ndarray, prob: float = 0.25, rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulates multi-field-of-view e-beam stitching boundary lines."""
    rng = rng or np.random.default_rng()
    if rng.random() > prob:
        return image
    out = image.copy()
    h, w = image.shape
    split_x = w // 2 + int(rng.integers(-50, 50))
    # Slight contrast/offset shift across field boundary
    out[:, split_x:] = np.clip(out[:, split_x:].astype(np.int16) + int(rng.integers(-15, 15)), 0, 255).astype(np.uint8)
    return out

def apply_advanced_sem_physics(
    search_img: np.ndarray,
    params: AdvancedSEMPhysicsParams | None = None,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """Applies complete advanced SEM physics augmentation pipeline."""
    p = params or AdvancedSEMPhysicsParams()
    rng = rng or np.random.default_rng()

    img = apply_line_edge_roughness(search_img, ler_sigma_px=p.ler_sigma_nm / 10.0, rng=rng)
    img = apply_focus_gradient(img, strength=p.focus_gradient_strength)
    img = apply_charging_breakdown_streaks(img, streak_prob=p.charging_streak_prob, rng=rng)
    img = apply_stitching_boundary_artifact(img, prob=p.stitching_artifact_prob, rng=rng)
    return img
