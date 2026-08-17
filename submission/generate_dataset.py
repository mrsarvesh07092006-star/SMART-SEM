#!/usr/bin/env python3
"""
SMART-SEM Synthetic Dataset Generator CLI.

Generates paired Reference and Search images with:
- SEM imaging physics (blur, shot noise, charging, speckle, astigmatism)
- Navigation error simulation (stage drift, backlash, thermal expansion, vibration)
- Ground-truth target center coordinates & per-pair metadata manifest CSV
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import GenerationParams, generate_sample, SEARCH_VARIANTS
from src.smart_sem.navigation import NavigationErrorSimulator, NavigationParams, apply_navigation_error_to_gt

def main():
    parser = argparse.ArgumentParser(description="SMART-SEM Synthetic Dataset Generator")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of sample pairs to generate (default: 30)")
    parser.add_argument("--out-dir", type=str, default="results/dataset", help="Output directory for generated dataset")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--apply-nav-errors", action="store_true", default=True, help="Simulate realistic stage navigation errors")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    architectures = ["dram_1x", "finfet_10nm"]

    manifest_rows = []

    print(f"==================================================")
    print(f" SMART-SEM Dataset Generator")
    print(f" Generating {args.num_samples} synthetic pair(s)...")
    print(f" Output directory: {args.out_dir}")
    print(f" Base Seed: {args.seed}")
    print(f"==================================================")

    for i in range(args.num_samples):
        sample_seed = args.seed + i * 101
        arch = architectures[i % len(architectures)]

        # Setup custom params per sample for high diversity
        params = GenerationParams()
        
        # Vary noise levels and degradations across samples
        if i % 5 == 1:
            params.dose_search = 55.0 # Low dose noise
        elif i % 5 == 2:
            params.shear_amplitude_px = 3.5 # Heavy drift
        elif i % 5 == 3:
            params.speckle_sigma = 0.25 # Speckle & impulse noise
            params.salt_pepper_prob = 0.01
        elif i % 5 == 4:
            params.charging_streak_prob = 3.0 # Charging streak artifacts
            params.charging_streak_intensity = 2.0

        rng = np.random.default_rng(sample_seed)
        sample = generate_sample(arch, rng, params)

        sample_id = f"pair_{i:04d}"
        ref_filename = f"{sample_id}_ref.png"
        search_filename = f"{sample_id}_search.png"

        ref_path = os.path.join(images_dir, ref_filename)
        search_path = os.path.join(images_dir, search_filename)

        # Save images
        cv2.imwrite(ref_path, sample["reference_img"])
        cv2.imwrite(search_path, sample["search_img"])

        gt_x = sample["gt_x"]
        gt_y = sample["gt_y"]

        # Apply Navigation Error Simulator if enabled
        if args.apply_nav_errors:
            nav_sim = NavigationErrorSimulator(NavigationParams(drift_sigma_nm=2.5, backlash_max_nm=3.0), rng=rng)
            cum_x_nm, cum_y_nm, nav_report = nav_sim.generate_cumulative_trajectory_error(n_steps=i % 4 + 1)
            nav_dx_px = cum_x_nm / 10.0 # 10nm per px in search image
            nav_dy_px = cum_y_nm / 10.0
            obs_gt_x, obs_gt_y = apply_navigation_error_to_gt(gt_x, gt_y, (nav_dx_px, nav_dy_px))
        else:
            nav_dx_px, nav_dy_px = 0.0, 0.0
            obs_gt_x, obs_gt_y = gt_x, gt_y
            nav_report = {}

        # Save metadata JSON
        meta = {
            "sample_id": sample_id,
            "architecture": arch,
            "seed": sample_seed,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "nav_dx_px": nav_dx_px,
            "nav_dy_px": nav_dy_px,
            "obs_gt_x": obs_gt_x,
            "obs_gt_y": obs_gt_y,
            "navigation_report": nav_report,
            "params": sample["params"],
        }
        with open(os.path.join(images_dir, f"{sample_id}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        manifest_rows.append({
            "sample_id": sample_id,
            "architecture": arch,
            "reference_path": os.path.relpath(ref_path, args.out_dir),
            "search_path": os.path.relpath(search_path, args.out_dir),
            "gt_x": f"{gt_x:.2f}",
            "gt_y": f"{gt_y:.2f}",
            "nav_dx_px": f"{nav_dx_px:.2f}",
            "nav_dy_px": f"{nav_dy_px:.2f}",
            "obs_gt_x": f"{obs_gt_x:.2f}",
            "obs_gt_y": f"{obs_gt_y:.2f}",
            "seed": sample_seed,
        })

        if (i + 1) % 5 == 0 or i == args.num_samples - 1:
            print(f" Generated [{i+1}/{args.num_samples}] pairs...")

    # Write manifest CSV
    fieldnames = ["sample_id", "architecture", "reference_path", "search_path", "gt_x", "gt_y", "nav_dx_px", "nav_dy_px", "obs_gt_x", "obs_gt_y", "seed"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n[OK] Dataset generation complete!")
    print(f" Manifest saved to: {manifest_path}")
    print(f" Images saved in: {images_dir}\n")

if __name__ == "__main__":
    main()
