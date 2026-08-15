#!/usr/bin/env python3
"""
SMART-SEM Primary Benchmark Entrypoint.
Runs end-to-end evaluation on the dataset and prints the official leaderboard.

Usage:
    python run_benchmark.py
"""

from __future__ import annotations
import os
import sys
import subprocess

def main():
    print("=" * 60)
    print(" SMART-SEM: Industrial Semiconductor Alignment Benchmark")
    print(" Applied Materials Drift-Sense Track | Semicon India 2026")
    print("=" * 60)
    
    # Check if dataset exists, if not generate it
    manifest_path = os.path.join("results", "dataset", "manifest.csv")
    if not os.path.exists(manifest_path):
        print("\n[!] Dataset manifest not found. Generating 30 benchmark pairs...")
        cmd_gen = [sys.executable, "generate_dataset.py", "--num-samples", "30", "--out-dir", "results/dataset"]
        subprocess.run(cmd_gen, check=True)
    
    print("\n[*] Executing SMART-SEM Batch Localization Engine...")
    cmd_eval = [sys.executable, "localize.py", "--manifest", "results/dataset/manifest.csv", "--out-dir", "results/evaluation"]
    subprocess.run(cmd_eval, check=True)
    
    # Display Leaderboard
    table_path = os.path.join("experiments", "baseline_table.csv")
    if os.path.exists(table_path):
        print("\n" + "=" * 60)
        print(" OFFICIAL BENCHMARK LEADERBOARD (experiments/baseline_table.csv)")
        print("=" * 60)
        with open(table_path, "r", encoding="utf-8") as f:
            print(f.read())
            
    print("[OK] Benchmark completed successfully.")

if __name__ == "__main__":
    main()
