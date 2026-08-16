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

# Test vertical & horizontal shear compensation angles [-3 deg, +3 deg]
best_err = 999.0
best_pred = None

for shear_y in [-0.08, -0.05, -0.03, 0.0, 0.03, 0.05, 0.08]:
    M = np.float32([[1, 0, 0], [shear_y, 1, 0]])
    sheared_tmpl = cv2.warpAffine(tmpl, M, (tw, th))
    res = cv2.matchTemplate(search, sheared_tmpl, cv2.TM_CCOEFF_NORMED)
    
    # Check around GT
    gt_tl_x = int(round(gt[0] - tw / 2.0))
    gt_tl_y = int(round(gt[1] - th / 2.0))
    
    # Check in +/- 40 px window around GT
    win = res[max(0, gt_tl_y-40):min(res.shape[0], gt_tl_y+41), max(0, gt_tl_x-40):min(res.shape[1], gt_tl_x+41)]
    _, max_val, _, max_loc = cv2.minMaxLoc(win)
    px = gt_tl_x - 40 + max_loc[0] + tw / 2.0
    py = gt_tl_y - 40 + max_loc[1] + th / 2.0
    err = math.hypot(px - gt[0], py - gt[1])
    print(f"Shear {shear_y:+.2f} -> Score: {max_val:.4f}, Pred: ({px:.1f}, {py:.1f}), Err: {err:.2f}px")
