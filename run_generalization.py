#!/usr/bin/env python3
"""
SMART-SEM Out-of-Distribution Generalization Benchmark Entrypoint.
Runs evaluation across 4 SEM stress domains and outputs experiments/generalization_table.csv.

Usage:
    python run_generalization.py
"""

import os
import sys
import subprocess

def main():
    print("=" * 60)
    print(" SMART-SEM: Generalization Benchmark Runner")
    print("=" * 60)
    cmd = [sys.executable, "experiments/generalization_benchmark.py"]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
