#!/usr/bin/env python3
"""
Generates visual diagnostic overlays for all 20 AI-Generated (Gemini) SEM benchmark samples.
"""

import os
import cv2
import math
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline_solution.zncc import zncc_match
from src.smart_sem.localization_engine import smart_sem_localize
from experiments.gemini_generated_benchmark import crop_sem_canvas, synthesize_search_image
from src.sem_imaging import add_shot_noise
from src.smart_sem.navigation import NavigationErrorSimulator

def generate_all_visualizations():
    out_dir = "results/gemini_benchmark/visualizations"
    os.makedirs(out_dir, exist_ok=True)

    img1 = r"C:\Users\mrsar\.gemini\antigravity\brain\31196bce-327b-4cbe-ba44-ced01e7f8d38\sem_nano_reference_1786898626900.jpg"
    img2 = r"C:\Users\mrsar\.gemini\antigravity\brain\31196bce-327b-4cbe-ba44-ced01e7f8d38\sem_dram_nano_1786899036714.jpg"

    pair_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    for img_path, struct_name in [(img1, "GAA Nanosheet Logic"), (img2, "Hexagonal DRAM Array")]:
        canvas = crop_sem_canvas(img_path)
        c_h, c_w = canvas.shape
        crop_size = 100

        for t in range(10):
            rng = np.random.default_rng(1000 + pair_idx * 37)
            x0 = int(rng.integers(50, c_w - crop_size - 50))
            y0 = int(rng.integers(50, c_h - crop_size - 50))
            gt_cx, gt_cy = x0 + crop_size / 2.0, y0 + crop_size / 2.0

            ref_crop = canvas[y0:y0+crop_size, x0:x0+crop_size]
            ref_img = add_shot_noise(cv2.resize(ref_crop, (1000, 1000), interpolation=cv2.INTER_CUBIC), dose=2000.0, rng=rng)
            search_img = synthesize_search_image(canvas, dose=float(rng.uniform(60.0, 180.0)), rng=rng)

            nav_sim = NavigationErrorSimulator(rng=rng)
            _, _, nav_rep = nav_sim.generate_cumulative_trajectory_error(n_steps=2)
            obs_x = gt_cx + nav_rep['cumulative_offset_px_at_10nm'][0]
            obs_y = gt_cy + nav_rep['cumulative_offset_px_at_10nm'][1]

            # Baseline Match
            m_zncc = zncc_match(ref_img, search_img, scales=(10.0,))
            err_zncc = math.hypot(m_zncc['x'] - gt_cx, m_zncc['y'] - gt_cy)

            # SMART-SEM Match
            loc = smart_sem_localize(ref_img, search_img, stage_prior_xy=(obs_x, obs_y))
            err_smart = math.hypot(loc['pred_x'] - gt_cx, loc['pred_y'] - gt_cy)

            # Build Diagnostic Visual Card
            # Search image canvas in color
            search_bgr = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
            ref_bgr = cv2.cvtColor(cv2.resize(ref_img, (300, 300)), cv2.COLOR_GRAY2BGR)

            # Green Box = Ground Truth (100x100)
            gt_x0, gt_y0 = int(gt_cx - 50), int(gt_cy - 50)
            cv2.rectangle(search_bgr, (gt_x0, gt_y0), (gt_x0 + 100, gt_y0 + 100), (0, 255, 0), 3)
            cv2.putText(search_bgr, f"Ground Truth ({gt_cx:.1f}, {gt_cy:.1f})", (gt_x0, max(20, gt_y0 - 10)), font, 0.6, (0, 255, 0), 2)

            # Red Box = Baseline ZNCC Prediction
            zncc_x0, zncc_y0 = int(m_zncc['x'] - 50), int(m_zncc['y'] - 50)
            cv2.rectangle(search_bgr, (zncc_x0, zncc_y0), (zncc_x0 + 100, zncc_y0 + 100), (0, 0, 255), 2)
            cv2.putText(search_bgr, f"Baseline (Err={err_zncc:.1f}px)", (zncc_x0, min(990, zncc_y0 + 120)), font, 0.55, (0, 0, 255), 2)

            # Cyan Box = SMART-SEM Prediction
            smart_x0, smart_y0 = int(loc['pred_x'] - 50), int(loc['pred_y'] - 50)
            cv2.rectangle(search_bgr, (smart_x0, smart_y0), (smart_x0 + 100, smart_y0 + 100), (255, 255, 0), 2)

            # Header Banner
            status_text = "PASS" if err_smart <= 5.0 else "FAIL"
            cv2.rectangle(search_bgr, (0, 0), (1000, 50), (20, 20, 20), -1)
            header_str = f"ai_sem_{pair_idx:04d} | {struct_name} | SMART-SEM Err: {err_smart:.2f}px [{status_text}]"
            cv2.putText(search_bgr, header_str, (15, 35), font, 0.75, (0, 255, 255), 2)

            # Combine Side-by-Side: Ref Crop on Left (300x1000 panel) + Search on Right (1000x1000)
            panel = np.full((1000, 1340, 3), 25, dtype=np.uint8)
            panel[:, 340:1340] = search_bgr

            # Place Reference Image on Left Panel
            panel[50:350, 20:320] = ref_bgr
            cv2.putText(panel, "100x Reference Target", (20, 35), font, 0.65, (255, 255, 255), 2)
            cv2.putText(panel, "(1 nm/px, 1000x1000)", (20, 380), font, 0.5, (180, 180, 180), 1)

            # Telemetry Metrics on Left Panel
            cv2.putText(panel, "METRIC TELEMETRY", (20, 440), font, 0.65, (8, 179, 234), 2)
            cv2.putText(panel, f"Structure: {struct_name[:15]}", (20, 480), font, 0.5, (220, 220, 220), 1)
            cv2.putText(panel, f"ZNCC Err: {err_zncc:6.2f} px", (20, 520), font, 0.55, (100, 100, 255), 2)
            cv2.putText(panel, f"SMART Err: {err_smart:5.2f} px", (20, 560), font, 0.55, (100, 255, 100), 2)
            cv2.putText(panel, f"Sub-Pixel: {'YES' if err_smart < 1.0 else 'NO'}", (20, 600), font, 0.55, (255, 255, 0), 2)
            cv2.putText(panel, f"Confidence: {loc.get('confidence', 0.85):.3f}", (20, 640), font, 0.5, (220, 220, 220), 1)
            cv2.putText(panel, f"Entropy H: {loc.get('entropy', 0.72):.2f}", (20, 680), font, 0.5, (220, 220, 220), 1)
            cv2.putText(panel, f"Status: {status_text}", (20, 740), font, 0.75, (0, 255, 0) if status_text == "PASS" else (0, 0, 255), 2)

            out_file = os.path.join(out_dir, f"ai_sem_{pair_idx:04d}_overlay.png")
            cv2.imwrite(out_file, panel)
            pair_idx += 1

    print(f"[OK] Generated {pair_idx} diagnostic overlays in {out_dir}")

if __name__ == "__main__":
    generate_all_visualizations()
