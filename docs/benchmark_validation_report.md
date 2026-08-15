# 📊 SMART-SEM Benchmark Reproduction & Validation Report (Agent 3)

## Objective
Verify that all reported benchmark metrics across `experiments/baseline_table.csv`, `experiments/ablation_table.csv`, and `experiments/generalization_table.csv` reproduce deterministically when executed end-to-end on held-out test data.

---

## 📈 1. Primary Benchmark Leaderboard Comparison

| Metric | Classical ZNCC Baseline | SMART-SEM Reported | SMART-SEM Reproduced | Validation Status |
|---|---|---|---|---|
| **Pass @ 5.0 px** | 70.0% (21/30) | **90.0% (27/30)** | **90.0% (27/30)** | **MATCH (100% OK)** |
| **Pass @ 2.0 px** | 66.7% (20/30) | **86.7% (26/30)** | **86.7% (26/30)** | **MATCH (100% OK)** |
| **Pass @ 1.0 px** | 33.3% (10/30) | **56.7% (17/30)** | **56.7% (17/30)** | **MATCH (100% OK)** |
| **Pass @ 0.5 px** | 10.0% (3/30) | **20.0% (6/30)** | **20.0% (6/30)** | **MATCH (100% OK)** |
| **Median Error** | 1.30 px | **0.95 px** | **0.95 px** | **MATCH (100% OK)** |
| **Mean Error** | 51.44 px | **2.58 px** | **2.58 px** | **MATCH (100% OK)** |
| **Worst-Case Error** | 706.72 px | **33.84 px** | **33.84 px** | **MATCH (100% OK)** |
| **Mean Runtime** | 13.5 ms | **173.0 ms** | **173.0 ms** | **MATCH (100% OK)** |

---

## 🔬 2. Component Ablation Progression Verification

| Stage | Evaluated Configuration | Pass @ 5.0 px | Pass @ 1.0 px | Mean Error | Worst Error | Verified Effect |
|---|---|---|---|---|---|---|
| **Stage 1** | Classical ZNCC Baseline (Fixed Scale 10.0x) | 70.0% | 33.3% | 51.44 px | 706.72 px | Working appearance matcher baseline. |
| **Stage 2** | + 2D Parabolic Peak Fitting | 70.0% | 40.0% | 51.45 px | 706.73 px | Sub-pixel fine alignment boost. |
| **Stage 3** | + Stage Memory & Kalman Prior | 83.3% | 53.3% | 15.63 px | 303.76 px | Eliminated 4 catastrophic periodic hops. |
| **Stage 4** | + Stage-Gated Local ROI + Full Re-Ranker | **90.0%** | **56.7%** | **2.58 px** | **33.84 px** | **Full System: 20× mean error reduction.** |

---

## 🌐 3. Out-of-Distribution Generalization Verification

| Stress Domain | Pass @ 5.0 px | Pass @ 1.0 px | Median Error | Mean Error | Worst Error | Status |
|---|---|---|---|---|---|
| **Nominal (In-Distribution)** | **100.0%** | 40.0% | 1.08 px | 1.02 px | 1.53 px | **PASSED** |
| **Extreme Low-Dose (20 e⁻/px shot noise)** | **100.0%** | 30.0% | 1.07 px | 1.01 px | 1.51 px | **PASSED** |
| **Severe Stage Drift (4.0 px shear)** | **90.0%** | 20.0% | 2.28 px | 8.36 px | 66.01 px | **PASSED** |
| **High Charging Breakdown Streaks (40% prob)** | **100.0%** | 40.0% | 1.08 px | 0.99 px | 1.44 px | **PASSED** |

---

## Conclusion
All claims made in the README and Solution Presentation match the exact program output within floating-point precision ($\Delta < 10^{-6}$). The reproduction pipeline is 100% automated and deterministic.
