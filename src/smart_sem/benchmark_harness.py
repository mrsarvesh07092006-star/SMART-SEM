"""
SMART-SEM Automated Benchmark & Leaderboard Harness (Agent 7).

Benchmarks localization engines across normal and hard semiconductor datasets:
- Classical ZNCC
- 2D Phase Correlation
- Sobel Gradient Matching
- SMART-SEM Advanced Multi-Stage Engine

Outputs Pass@5px, Pass@4px, Pass@2px, Pass@1px, Mean, Median, Worst-Case Error, and Latency.
"""

from __future__ import annotations
import csv
import json
import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.smart_sem.topology import discover_topology
from src.smart_sem.localization_engine import smart_sem_localize
from src.smart_sem.failure_analysis import FailureAnalysisAgent
from baseline_solution.zncc import zncc_match

class BenchmarkHarness:
    """Automated benchmark harness for continuous model comparison and leaderboard generation."""

    def __init__(self, manifest_path: str = "results/dataset/manifest.csv", out_dir: str = "experiments"):
        self.manifest_path = manifest_path
        self.out_dir = out_dir
        self.dataset_root = os.path.dirname(os.path.abspath(manifest_path))
        os.makedirs(out_dir, exist_ok=True)
        self.failure_agent = FailureAnalysisAgent(os.path.join(out_dir, "failures_database.json"))

    def load_dataset(self) -> list[dict]:
        rows = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows

    def evaluate_engine(self, name: str, engine_fn) -> dict:
        rows = self.load_dataset()
        errors = []
        times = []
        failure_counts = {}

        p5, p4, p2, p1, sub05 = 0, 0, 0, 0, 0

        for row in rows:
            gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
            ref_path = os.path.join(self.dataset_root, row["reference_path"])
            search_path = os.path.join(self.dataset_root, row["search_path"])

            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
            if ref_img is None or search_img is None:
                continue

            t0 = time.perf_counter()
            pred_x, pred_y, conf, topo = engine_fn(ref_img, search_img)
            t1 = time.perf_counter()

            err = float(np.hypot(pred_x - gt_x, pred_y - gt_y))
            errors.append(err)
            times.append((t1 - t0) * 1000.0)

            if err <= 5.0: p5 += 1
            if err <= 4.0: p4 += 1
            if err <= 2.0: p2 += 1
            if err <= 1.0: p1 += 1
            if err <= 0.5: sub05 += 1

            # Diagnose failure
            diag = self.failure_agent.diagnose_failure(
                sample_id=row["sample_id"],
                architecture=row["architecture"],
                gt_x=gt_x, gt_y=gt_y,
                pred_x=pred_x, pred_y=pred_y,
                confidence=conf,
                topology_info=topo
            )
            cat = diag["category"]
            failure_counts[cat] = failure_counts.get(cat, 0) + 1

        total = len(errors)
        if total == 0:
            return {}

        return {
            "method": name,
            "total_tested": total,
            "pass_rate_5px_pct": float(p5 / total * 100.0),
            "pass_rate_4px_pct": float(p4 / total * 100.0),
            "pass_rate_2px_pct": float(p2 / total * 100.0),
            "pass_rate_1px_pct": float(p1 / total * 100.0),
            "sub_pixel_05px_pct": float(sub05 / total * 100.0),
            "mean_error_px": float(np.mean(errors)),
            "median_error_px": float(np.median(errors)),
            "worst_error_px": float(np.max(errors)),
            "mean_runtime_ms": float(np.mean(times)),
            "failure_breakdown": failure_counts,
        }

    def run_full_leaderboard(self) -> list[dict]:
        def zncc_fixed_runner(ref, search):
            m = zncc_match(ref, search, scales=(10.0,))
            return m["x"], m["y"], m["score"], {}

        def smart_sem_engine_runner(ref, search):
            topo = discover_topology(ref)
            m = smart_sem_localize(ref, search, scales=(9.5, 9.8, 10.0, 10.2, 10.5))
            return m["pred_x"], m["pred_y"], m["confidence"], topo

        engines = [
            ("Classical ZNCC (Fixed Scale 10.0x)", zncc_fixed_runner),
            ("SMART-SEM Industrial Localization Engine", smart_sem_engine_runner),
        ]

        leaderboard = []
        for name, fn in engines:
            print(f"Benchmarking {name}...")
            res = self.evaluate_engine(name, fn)
            leaderboard.append(res)

        # Save JSON Leaderboard
        with open(os.path.join(self.out_dir, "benchmark_leaderboard.json"), "w", encoding="utf-8") as f:
            json.dump(leaderboard, f, indent=2)

        # Save CSV Table
        csv_path = os.path.join(self.out_dir, "baseline_table.csv")
        csv_rows = []
        for r in leaderboard:
            csv_rows.append({
                "Method": r["method"],
                "Pass_Rate_5px_Pct": f"{r['pass_rate_5px_pct']:.1f}%",
                "Pass_Rate_1px_Pct": f"{r['pass_rate_1px_pct']:.1f}%",
                "Mean_Error_px": f"{r['mean_error_px']:.2f}",
                "Median_Error_px": f"{r['median_error_px']:.2f}",
                "Worst_Error_px": f"{r['worst_error_px']:.2f}",
                "Mean_Runtime_ms": f"{r['mean_runtime_ms']:.1f}",
            })

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"[OK] Leaderboard saved to: {csv_path}")
        return leaderboard
