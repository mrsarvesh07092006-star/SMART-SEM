import cv2
import math
import os
import sys
import numpy as np

sys.path.insert(0, '.')
from baseline_solution.zncc import zncc_match
from src.smart_sem.localization_engine import smart_sem_localize
from experiments.gemini_generated_benchmark import crop_sem_canvas, synthesize_search_image
from src.sem_imaging import add_shot_noise
from src.smart_sem.navigation import NavigationErrorSimulator

img1 = r'C:\Users\mrsar\.gemini\antigravity\brain\31196bce-327b-4cbe-ba44-ced01e7f8d38\sem_nano_reference_1786898626900.jpg'
img2 = r'C:\Users\mrsar\.gemini\antigravity\brain\31196bce-327b-4cbe-ba44-ced01e7f8d38\sem_dram_nano_1786899036714.jpg'

pair_idx = 0
header = f"{'Pair ID':<12} | {'Structure':<16} | {'GT Center':<16} | {'ZNCC Err':<10} | {'SMART-SEM Err':<14} | {'Status':<8}"
print(header)
print("-" * len(header))

for img_path, img_name in [(img1, "GAA Nanosheet"), (img2, "Hexagonal DRAM")]:
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
        
        m = zncc_match(ref_img, search_img, scales=(10.0,))
        err_zncc = math.hypot(m['x'] - gt_cx, m['y'] - gt_cy)
        
        loc = smart_sem_localize(ref_img, search_img, stage_prior_xy=(obs_x, obs_y))
        err_smart = math.hypot(loc['pred_x'] - gt_cx, loc['pred_y'] - gt_cy)
        
        status = "PASS" if err_smart <= 5.0 else "FAIL"
        print(f"ai_sem_{pair_idx:04d}   | {img_name:<16} | ({gt_cx:5.1f}, {gt_cy:5.1f}) | {err_zncc:6.2f} px   | {err_smart:6.2f} px       | {status}")
        pair_idx += 1
