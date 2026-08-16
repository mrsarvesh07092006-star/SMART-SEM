import csv
import cv2
import os
import math
import sys
import numpy as np

with open('results/dataset/manifest.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

row13 = next(r for r in rows if r['sample_id'] == 'pair_0013')
ref = cv2.imread(os.path.join('results/dataset', row13['reference_path']), cv2.IMREAD_GRAYSCALE)
search = cv2.imread(os.path.join('results/dataset', row13['search_path']), cv2.IMREAD_GRAYSCALE)
gt = (float(row13['gt_x']), float(row13['gt_y']))

tw, th = 100, 100
tmpl = cv2.resize(ref, (tw, th), interpolation=cv2.INTER_AREA)
score_map = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)

gt_top_left_x = int(round(gt[0] - tw / 2.0))
gt_top_left_y = int(round(gt[1] - th / 2.0))

print(f"GT Top Left: ({gt_top_left_x}, {gt_top_left_y})")
print(f"Score at GT: {score_map[gt_top_left_y, gt_top_left_x]:.4f}")

# Look at 15x15 window around GT
gt_win = score_map[gt_top_left_y-15:gt_top_left_y+16, gt_top_left_x-15:gt_top_left_x+16]
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gt_win)
print(f"Max score in GT window: {max_val:.4f} at offset (dx={max_loc[0]-15}, dy={max_loc[1]-15})")
