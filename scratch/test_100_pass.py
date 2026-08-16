import csv
import cv2
import os
import math
import numpy as np
import sys
sys.path.insert(0, '.')
from src.smart_sem.localization_engine import smart_sem_localize

with open('results/dataset/manifest.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

for pair_id in ['pair_0013', 'pair_0015', 'pair_0029']:
    row = next(r for r in rows if r['sample_id'] == pair_id)
    ref = cv2.imread(os.path.join('results/dataset', row['reference_path']), cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(os.path.join('results/dataset', row['search_path']), cv2.IMREAD_GRAYSCALE)
    gt = (float(row['gt_x']), float(row['gt_y']))
    obs = (float(row['obs_gt_x']), float(row['obs_gt_y']))
    
    loc = smart_sem_localize(ref, search, stage_prior_xy=obs)
    err = math.hypot(loc['pred_x'] - gt[0], loc['pred_y'] - gt[1])
    print(f"\n{pair_id} -> Current Pred: ({loc['pred_x']:.2f}, {loc['pred_y']:.2f}) | GT: {gt} | Stage Obs: {obs} | Error: {err:.2f}px")
    for i, c in enumerate(loc['top_k_candidates'][:8]):
        c_err = math.hypot(c['center_x'] - gt[0], c['center_y'] - gt[1])
        c_sdist = math.hypot(c['center_x'] - obs[0], c['center_y'] - obs[1])
        print(f"   [Rank {i+1}] ({c['center_x']:.2f}, {c['center_y']:.2f}) | ZNCC={c['score']:.4f} | StageDist={c_sdist:.2f}px | Err={c_err:.2f}px")
