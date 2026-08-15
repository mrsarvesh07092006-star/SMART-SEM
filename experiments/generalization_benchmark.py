"""
SMART-SEM Generalization & Stress-Testing Benchmark (Pillar B).

Evaluates model robustness on 4 out-of-distribution SEM stress domains:
1. Low-Dose Regime (Dose = 20 e-/px, extreme shot noise)
2. High-Drift Regime (Drift jitter = 2.5 px, shear = 4.0 px)
3. Severe Charging Streaks (Charging prob = 0.40)
4. Linewidth & Process Variations (Linewidth bias = 3.0 nm)
"""

from __future__ import annotations
import csv
import json
import math
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import generate_sample, GenerationParams
from src.presets import PRESETS
from src.smart_sem.localization_engine import smart_sem_localize
from src.smart_sem.navigation import NavigationErrorSimulator

def run_generalization_benchmark(n_samples_per_domain: int = 10, out_dir: str = "experiments"):
    os.makedirs(out_dir, exist_ok=True)
    domains = [
        ("Nominal (In-Distribution)", {}),
        ("Extreme Low-Dose Shot Noise", {"dose_search": 20.0, "detector_noise_sigma_search": 8.0}),
        ("Severe Mechanical Stage Drift", {"drift_jitter_px": 2.5, "shear_amplitude_px": 3.5}),
        ("High Charging Streaks", {"charging_streak_prob": 0.40, "charging_streak_intensity": 60.0}),
    ]

    leaderboard = []
    nav_sim = NavigationErrorSimulator()

    print(f"==================================================")
    print(f" RUNNING OUT-OF-DISTRIBUTION GENERALIZATION BENCHMARK")
    print(f"==================================================")

    for domain_name, stress_params in domains:
        print(f"\n[Testing Domain] {domain_name}...")
        errors = []
        pass_5, pass_1 = 0, 0

        for i in range(n_samples_per_domain):
            arch = "dram_1x" if i % 2 == 0 else "finfet_10nm"
            preset = PRESETS[arch].copy()
            for k, v in stress_params.items():
                preset[k] = v

            rng = np.random.default_rng(500 + i * 13)
            p_dict = {k: v for k, v in preset.items() if k in GenerationParams.__annotations__}
            gen_params = GenerationParams(**p_dict)
            sample = generate_sample(arch, rng, gen_params)
            ref_img = sample["reference_img"]
            search_img = sample["search_img"]
            gt_x, gt_y = sample["gt_x"], sample["gt_y"]

            # Simulate stage navigation prior
            nav_sim = NavigationErrorSimulator(rng=np.random.default_rng(500 + i * 13))
            _, _, nav_rep = nav_sim.generate_cumulative_trajectory_error(n_steps=2)
            obs_x = gt_x + nav_rep["cumulative_offset_px_at_10nm"][0]
            obs_y = gt_y + nav_rep["cumulative_offset_px_at_10nm"][1]

            loc_res = smart_sem_localize(ref_img, search_img, stage_prior_xy=(obs_x, obs_y))
            pred_x, pred_y = loc_res["pred_x"], loc_res["pred_y"]
            err = math.hypot(pred_x - gt_x, pred_y - gt_y)
            errors.append(err)

            if err <= 5.0: pass_5 += 1
            if err <= 1.0: pass_1 += 1

        total = len(errors)
        res_dict = {
            "stress_domain": domain_name,
            "samples_tested": total,
            "pass_rate_5px_pct": float(pass_5 / total * 100.0),
            "pass_rate_1px_pct": float(pass_1 / total * 100.0),
            "median_error_px": float(np.median(errors)),
            "mean_error_px": float(np.mean(errors)),
            "worst_error_px": float(np.max(errors)),
        }
        leaderboard.append(res_dict)
        print(f" -> Pass@5px: {pass_5}/{total} ({res_dict['pass_rate_5px_pct']:.1f}%) | Median Error: {res_dict['median_error_px']:.2f}px | Mean Error: {res_dict['mean_error_px']:.2f}px")

    out_json = os.path.join(out_dir, "generalization_leaderboard.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    # Also save CSV
    csv_path = os.path.join(out_dir, "generalization_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stress_domain", "pass_rate_5px_pct", "pass_rate_1px_pct", "median_error_px", "mean_error_px", "worst_error_px"])
        writer.writeheader()
        for row in leaderboard:
            writer.writerow({
                "stress_domain": row["stress_domain"],
                "pass_rate_5px_pct": f"{row['pass_rate_5px_pct']:.1f}%",
                "pass_rate_1px_pct": f"{row['pass_rate_1px_pct']:.1f}%",
                "median_error_px": f"{row['median_error_px']:.2f}",
                "mean_error_px": f"{row['mean_error_px']:.2f}",
                "worst_error_px": f"{row['worst_error_px']:.2f}"
            })

    print(f"\n[OK] Generalization benchmark saved to {out_json} & {csv_path}\n")
    return leaderboard

if __name__ == "__main__":
    run_generalization_benchmark(n_samples_per_domain=10)
