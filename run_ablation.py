#!/usr/bin/env python3
"""
SMART-SEM Component Ablation Study Entrypoint.
Runs step-by-step layer ablation and outputs experiments/ablation_table.csv.

Usage:
    python run_ablation.py
"""

import os
import sys
import subprocess

def main():
    print("=" * 60)
    print(" SMART-SEM: Component Ablation Study Runner")
    print("=" * 60)
    cmd = [sys.executable, "experiments/ablation_study.py"]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
