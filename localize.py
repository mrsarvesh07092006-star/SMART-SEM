#!/usr/bin/env python3
"""
SMART-SEM Batch Localization CLI Engine (Updated).

Accepts reference & search image inputs or a dataset manifest CSV.
Runs full 5-stage SMART-SEM pipeline:
- Topology Discovery (pitch, orientation, periodicity)
- Advanced 5-stage multi-stream localization (eliminates periodic hops)
- Sub-pixel parabolic peak refinement
- Ambiguity Intelligence (Entropy & Probability distribution)
- Confusion Intelligence (Risk zone segmentation & colorized heatmaps)
- Failure categorization (Type A to E) & Wafer Memory Graph logging
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.smart_sem.topology import discover_topology
from src.smart_sem.localization_engine import smart_sem_localize
from src.smart_sem.confusion_intelligence import render_advanced_confusion_intelligence_map
from src.smart_sem.failure_analysis import FailureAnalysisAgent
from src.smart_sem.memory import WaferMemoryGraph

def main():
    parser = argparse.ArgumentParser(description="SMART-SEM Batch Localization CLI Engine")
    parser.add_argument("--manifest", type=str, default="results/dataset/manifest.csv", help="Path to manifest.csv file")
    parser.add_argument("--out-dir", type=str, default="results/evaluation", help="Output directory for results")
    parser.add_argument("--save-visualizations", action="store_true", default=True, help="Save overlay confusion intelligence dashboard images")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    viz_dir = os.path.join(args.out_dir, "confusion_maps")
    if args.save_visualizations:
        os.makedirs(viz_dir, exist_ok=True)

    if not os.path.exists(args.manifest):
        print(f"Error: Manifest file '{args.manifest}' not found.")
        print("Run `python generate_dataset.py` first to generate dataset.")
        sys.exit(1)

    print(f"==================================================")
    print(f" SMART-SEM Batch Localization Engine")
    print(f" Reading manifest: {args.manifest}")
    print(f" Output directory: {args.out_dir}")
    print(f"==================================================")

    dataset_root = os.path.dirname(os.path.abspath(args.manifest))
    rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    memory_graph = WaferMemoryGraph(os.path.join(args.out_dir, "wafer_memory_store.json"))
    failure_agent = FailureAnalysisAgent(os.path.join(args.out_dir, "failures_database.json"))

    results_rows = []
    errors_px = []
    runtimes_ms = []

    pass_5px = 0
    pass_4px = 0
    pass_2px = 0
    pass_1px = 0
    pass_05px = 0

    for idx, row in enumerate(rows):
        sample_id = row["sample_id"]
        arch = row["architecture"]
        ref_rel = row["reference_path"]
        search_rel = row["search_path"]

        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])
        obs_x = float(row["obs_gt_x"]) if "obs_gt_x" in row else None
        obs_y = float(row["obs_gt_y"]) if "obs_gt_y" in row else None
        stage_prior = (obs_x, obs_y) if (obs_x is not None and obs_y is not None) else None

        ref_full_path = os.path.join(dataset_root, ref_rel)
        search_full_path = os.path.join(dataset_root, search_rel)

        ref_img = cv2.imread(ref_full_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_full_path, cv2.IMREAD_GRAYSCALE)

        if ref_img is None or search_img is None:
            print(f" Warning: Could not read image for sample {sample_id}, skipping...")
            continue

        t0 = time.perf_counter()

        # Step 1: Topology Discovery
        topo_info = discover_topology(ref_img, pixel_size_nm=1.0)

        # Step 2: Unified Scale-Aware Localization with Memory-Guided Disambiguation
        loc_res = smart_sem_localize(
            ref_img, search_img,
            scales=(9.5, 9.8, 10.0, 10.2, 10.5),
            stage_prior_xy=stage_prior
        )

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        runtimes_ms.append(elapsed_ms)

        pred_x = loc_res["pred_x"]
        pred_y = loc_res["pred_y"]

        # Calculate Euclidean Error
        err_px = float(np.hypot(pred_x - gt_x, pred_y - gt_y))
        errors_px.append(err_px)

        # Count Pass Thresholds
        if err_px <= 5.0: pass_5px += 1
        if err_px <= 4.0: pass_4px += 1
        if err_px <= 2.0: pass_2px += 1
        if err_px <= 1.0: pass_1px += 1
        if err_px <= 0.5: pass_05px += 1

        # Step 3: Ambiguity Intelligence (from loc_res)
        amb_info = {
            "ambiguity_class": loc_res["ambiguity_class"],
            "entropy": loc_res["entropy"],
            "peak_ratio": loc_res["peak_ratio"],
            "top_candidates_distribution": loc_res["top_k_candidates"],
        }

        # Step 4: Failure Mode Diagnosis & Logging
        diag = failure_agent.diagnose_failure(
            sample_id=sample_id,
            architecture=arch,
            gt_x=gt_x, gt_y=gt_y,
            pred_x=pred_x, pred_y=pred_y,
            confidence=loc_res["confidence"],
            topology_info=topo_info
        )

        # Step 5: Wafer Memory Logging
        memory_graph.update_fingerprint(f"wafer_{arch}", arch, topo_info)
        memory_graph.log_inspection(sample_id, (gt_x, gt_y), (pred_x, pred_y), loc_res, amb_info, architecture=arch)

        # Step 6: Save Confusion Intelligence Visualization
        if args.save_visualizations:
            sim_map = loc_res["similarity_map"]
            if sim_map is not None:
                cmap_img = render_advanced_confusion_intelligence_map(
                    search_img,
                    sim_map,
                    ambiguity_info=amb_info,
                    gt_xy=(gt_x, gt_y),
                    pred_xy=(pred_x, pred_y)
                )
                cv2.imwrite(os.path.join(viz_dir, f"{sample_id}_confusion_intelligence.png"), cmap_img)

        results_rows.append({
            "sample_id": sample_id,
            "architecture": arch,
            "gt_x": f"{gt_x:.2f}",
            "gt_y": f"{gt_y:.2f}",
            "pred_x": f"{pred_x:.2f}",
            "pred_y": f"{pred_y:.2f}",
            "error_px": f"{err_px:.2f}",
            "confidence": f"{loc_res['confidence']:.4f}",
            "ambiguity_class": amb_info["ambiguity_class"],
            "entropy": f"{amb_info['entropy']:.4f}",
            "failure_category": diag["category"],
            "runtime_ms": f"{elapsed_ms:.1f}",
        })

        print(f" [{idx+1}/{len(rows)}] {sample_id} ({arch}): Error={err_px:.2f}px | Conf={loc_res['confidence']:.3f} | Cat={diag['category']} | Time={elapsed_ms:.1f}ms")

    total_count = len(errors_px)
    if total_count == 0:
        print("No samples were evaluated.")
        sys.exit(1)

    mean_err = float(np.mean(errors_px))
    median_err = float(np.median(errors_px))
    worst_err = float(np.max(errors_px))
    mean_time = float(np.mean(runtimes_ms))

    metrics_summary = {
        "total_evaluated": total_count,
        "pass_rate_5px_pct": float(pass_5px / total_count * 100.0),
        "pass_rate_4px_pct": float(pass_4px / total_count * 100.0),
        "pass_rate_2px_pct": float(pass_2px / total_count * 100.0),
        "pass_rate_1px_pct": float(pass_1px / total_count * 100.0),
        "pass_rate_05px_pct": float(pass_05px / total_count * 100.0),
        "mean_error_px": mean_err,
        "median_error_px": median_err,
        "worst_case_error_px": worst_err,
        "mean_runtime_ms": mean_time,
    }

    # Write Predictions CSV
    pred_csv_path = os.path.join(args.out_dir, "predictions.csv")
    fieldnames = ["sample_id", "architecture", "gt_x", "gt_y", "pred_x", "pred_y", "error_px", "confidence", "ambiguity_class", "entropy", "failure_category", "runtime_ms"]
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    # Write Metrics Summary JSON
    summary_json_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\n==================================================")
    print(f" SMART-SEM ADVANCED EVALUATION SUMMARY")
    print(f" Total Samples Tested : {total_count}")
    print(f" Pass Rate @ 5.0 px   : {metrics_summary['pass_rate_5px_pct']:.1f}% ({pass_5px}/{total_count})")
    print(f" Pass Rate @ 4.0 px   : {metrics_summary['pass_rate_4px_pct']:.1f}% ({pass_4px}/{total_count})")
    print(f" Pass Rate @ 2.0 px   : {metrics_summary['pass_rate_2px_pct']:.1f}% ({pass_2px}/{total_count})")
    print(f" Pass Rate @ 1.0 px   : {metrics_summary['pass_rate_1px_pct']:.1f}% ({pass_1px}/{total_count})")
    print(f" Pass Rate @ 0.5 px   : {metrics_summary['pass_rate_05px_pct']:.1f}% ({pass_05px}/{total_count})")
    print(f" Mean Error           : {mean_err:.2f} px")
    print(f" Median Error         : {median_err:.2f} px")
    print(f" Worst-Case Error     : {worst_err:.2f} px")
    print(f" Mean Runtime / pair  : {mean_time:.1f} ms")
    print(f" Predictions Saved to : {pred_csv_path}")
    print(f" Metrics Saved to     : {summary_json_path}")
    print(f"==================================================\n")

if __name__ == "__main__":
    main()
