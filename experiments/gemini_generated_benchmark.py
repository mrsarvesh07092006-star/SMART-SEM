"""
Gemini / AI-Generated SEM Benchmark Pipeline.
Evaluates Classical Baseline vs SMART-SEM Engine on generative nanoscale SEM wafer images.
"""

from __future__ import annotations
import os
import sys
import math
import json
import csv
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline_solution.zncc import zncc_match
from src.smart_sem.localization_engine import smart_sem_localize
from src.smart_sem.navigation import NavigationErrorSimulator
from src.sem_imaging import (
    gaussian_psf_blur,
    add_shot_noise,
    add_charging_streaks,
    apply_raster_drift,
    apply_vignette,
    apply_gamma
)

def crop_sem_canvas(img_path: str) -> np.ndarray:
    """Loads image and crops away bottom annotation/scale bar if present."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    h, w = img.shape
    # Crop out bottom 8% which usually contains microscope metadata bar
    clean_canvas = img[0:int(h * 0.92), 0:w]
    # Resize to standard 1000x1000 search canvas
    return cv2.resize(clean_canvas, (1000, 1000), interpolation=cv2.INTER_LANCZOS4)

def synthesize_search_image(
    canvas: np.ndarray,
    dose: float = 120.0,
    blur_spot_nm: float = 18.0,
    charging_prob: float = 0.25,
    drift_shear_px: float = 1.5,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """Applies realistic SEM physical degradation pipeline to create Search Image."""
    rng = rng or np.random.default_rng(42)
    
    # 1. Optical/Beam Blur (at 10nm/px search scale)
    blurred = gaussian_psf_blur(canvas, spot_size_nm=blur_spot_nm, pixel_size_nm=10.0, astigmatism_ratio=1.2)
    
    # 2. Scan Drift & Shear
    drifted = apply_raster_drift(blurred, shear_amplitude_px=drift_shear_px, jitter_std_px=0.4, rng=rng)
    
    # 3. Low-Dose Poisson Shot Noise
    noisy = add_shot_noise(drifted, dose=dose, rng=rng)
    
    # 4. Charging Breakdown Streaks
    charged = add_charging_streaks(noisy, streak_prob=charging_prob, intensity=40.0, rng=rng)
    
    return np.clip(charged, 0, 255).astype(np.uint8)

def run_gemini_benchmark(image_paths: list[str], n_trials_per_img: int = 10, out_dir: str = "results/gemini_benchmark"):
    os.makedirs(out_dir, exist_ok=True)
    out_viz_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(out_viz_dir, exist_ok=True)
    
    all_pairs = []
    pair_idx = 0
    
    for img_idx, p in enumerate(image_paths):
        canvas = crop_sem_canvas(p)
        c_h, c_w = canvas.shape
        crop_size = 100 # 100x100 template inside 1000x1000 search (10:1 ratio)
        
        for t in range(n_trials_per_img):
            rng = np.random.default_rng(1000 + pair_idx * 37)
            
            # Pick a random valid crop location
            x0 = int(rng.integers(50, c_w - crop_size - 50))
            y0 = int(rng.integers(50, c_h - crop_size - 50))
            gt_cx = x0 + crop_size / 2.0
            gt_cy = y0 + crop_size / 2.0
            
            # Reference crop at high quality
            ref_crop = canvas[y0:y0+crop_size, x0:x0+crop_size]
            # High dose imaging for reference (10x higher resolution / 1000x1000 equivalent)
            ref_upscaled = cv2.resize(ref_crop, (1000, 1000), interpolation=cv2.INTER_CUBIC)
            ref_img = add_shot_noise(ref_upscaled, dose=2000.0, rng=rng)
            
            # Synthesize degraded search image
            search_img = synthesize_search_image(
                canvas,
                dose=float(rng.uniform(60.0, 180.0)),
                blur_spot_nm=float(rng.uniform(12.0, 24.0)),
                charging_prob=float(rng.uniform(0.15, 0.35)),
                drift_shear_px=float(rng.uniform(0.8, 2.5)),
                rng=rng
            )
            
            # Simulate stage navigation prior
            nav_sim = NavigationErrorSimulator(rng=rng)
            _, _, nav_rep = nav_sim.generate_cumulative_trajectory_error(n_steps=2)
            obs_x = gt_cx + nav_rep["cumulative_offset_px_at_10nm"][0]
            obs_y = gt_cy + nav_rep["cumulative_offset_px_at_10nm"][1]
            
            all_pairs.append({
                "pair_id": f"ai_sem_{pair_idx:04d}",
                "source_img": os.path.basename(p),
                "ref_img": ref_img,
                "search_img": search_img,
                "gt": (gt_cx, gt_cy),
                "stage_prior": (obs_x, obs_y),
                "gt_box": (x0, y0, crop_size, crop_size)
            })
            pair_idx += 1
            
    print(f"\n==================================================")
    print(f" RUNNING AI-GENERATED (GEMINI) SEM BENCHMARK ({len(all_pairs)} PAIRS)")
    print(f"==================================================")
    
    # 1. Evaluate Classical ZNCC Baseline
    zncc_errors, zncc_times = [], []
    for s in all_pairs:
        t0 = cv2.getTickCount()
        m = zncc_match(s["ref_img"], s["search_img"], scales=(10.0,))
        dt = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
        err = math.hypot(m["x"] - s["gt"][0], m["y"] - s["gt"][1])
        zncc_errors.append(err)
        zncc_times.append(dt)
        
    # 2. Evaluate SMART-SEM Industrial Engine
    smart_errors, smart_times = [], []
    for idx, s in enumerate(all_pairs):
        t0 = cv2.getTickCount()
        loc = smart_sem_localize(s["ref_img"], s["search_img"], stage_prior_xy=s["stage_prior"])
        dt = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
        err = math.hypot(loc["pred_x"] - s["gt"][0], loc["pred_y"] - s["gt"][1])
        smart_errors.append(err)
        smart_times.append(dt)
        
        # Save visual comparison for select samples
        if idx < 6:
            viz = cv2.cvtColor(s["search_img"], cv2.COLOR_GRAY2BGR)
            # Ground truth (GREEN)
            gx, gy, gw, gh = s["gt_box"]
            cv2.rectangle(viz, (int(gx), int(gy)), (int(gx+gw), int(gy+gh)), (0, 255, 0), 2)
            cv2.putText(viz, "GT Location", (int(gx), int(gy) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            
            # Predicted (RED)
            px, py = int(round(loc["pred_x"])), int(round(loc["pred_y"]))
            cv2.rectangle(viz, (px - gw//2, py - gh//2), (px + gw//2, py + gh//2), (0, 0, 255), 2)
            cv2.circle(viz, (px, py), 4, (0, 0, 255), -1)
            cv2.putText(viz, f"Pred (Err: {err:.2f}px)", (px - gw//2, py + gh//2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            
            out_img_path = os.path.join(out_viz_dir, f"{s['pair_id']}_overlay.png")
            cv2.imwrite(out_img_path, viz)
            
    def compute_metrics(name: str, errors: list[float], times: list[float]):
        n = len(errors)
        p5 = sum(1 for e in errors if e <= 5.0) / n * 100.0
        p2 = sum(1 for e in errors if e <= 2.0) / n * 100.0
        p1 = sum(1 for e in errors if e <= 1.0) / n * 100.0
        p05 = sum(1 for e in errors if e <= 0.5) / n * 100.0
        return {
            "method": name,
            "pass_rate_5px_pct": p5,
            "pass_rate_2px_pct": p2,
            "pass_rate_1px_pct": p1,
            "pass_rate_05px_pct": p05,
            "median_error_px": float(np.median(errors)),
            "mean_error_px": float(np.mean(errors)),
            "worst_error_px": float(np.max(errors)),
            "mean_runtime_ms": float(np.mean(times))
        }
        
    res_zncc = compute_metrics("Classical ZNCC Baseline", zncc_errors, zncc_times)
    res_smart = compute_metrics("SMART-SEM Industrial Engine", smart_errors, smart_times)
    
    summary = [res_zncc, res_smart]
    
    # Save CSV
    csv_file = os.path.join(out_dir, "gemini_leaderboard.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(res_zncc.keys()))
        writer.writeheader()
        writer.writerow(res_zncc)
        writer.writerow(res_smart)
        
    # Save JSON
    json_file = os.path.join(out_dir, "gemini_metrics.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 60)
    print(" AI-GENERATED (GEMINI) SEM BENCHMARK LEADERBOARD")
    print("==================================================")
    for r in summary:
        print(f" -> {r['method']:<30} | Pass@5: {r['pass_rate_5px_pct']:.1f}% | Pass@1: {r['pass_rate_1px_pct']:.1f}% | Median: {r['median_error_px']:.2f}px | Mean: {r['mean_error_px']:.2f}px | Worst: {r['worst_error_px']:.2f}px")
        
    return summary

if __name__ == "__main__":
    img1 = r"C:\Users\mrsar\.gemini\antigravity\brain\31196bce-327b-4cbe-ba44-ced01e7f8d38\sem_nano_reference_1786898626900.jpg"
    img2 = r"C:\Users\mrsar\.gemini\antigravity\brain\31196bce-327b-4cbe-ba44-ced01e7f8d38\sem_dram_nano_1786899036714.jpg"
    run_gemini_benchmark([img1, img2], n_trials_per_img=10)
