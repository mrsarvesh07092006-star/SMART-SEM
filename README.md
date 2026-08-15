# SMART-SEM: Industrial Semiconductor Wafer Localization Platform
## Applied Materials — Drift-Sense Hackathon Submission | Semicon India 2026

**SMART-SEM** is an industrial-grade, explainable cross-magnification localization platform for semiconductor inspection tools (e-beam / SEM scanners). It combines physical acquisition simulation with **2D Kinematic Kalman Stage Tracking**, **Topology Consistency Scoring**, **FinFET Structure Tensors**, and **Stage-Gated Multi-Domain Candidate Re-Ranking** to eliminate periodic grating ambiguity and catastrophic alignment hops.

---

## 📊 Benchmark Leaderboard (30 Independently Generated Test Pairs)

| Method | Pass @ 5.0 px | Pass @ 2.0 px | Pass @ 1.0 px | Pass @ 0.5 px | Median Error | Mean Error | Worst Error | Latency |
|---|---|---|---|---|---|---|---|---|
| **Classical ZNCC Baseline** | 70.0% (21/30) | 66.7% (20/30) | 33.3% (10/30) | 10.0% (3/30) | 1.30 px | 51.44 px | 706.72 px | **13.5 ms** |
| **SMART-SEM Industrial Engine** | **90.0% (27/30)** | **86.7% (26/30)** | **56.7% (17/30)** | **20.0% (6/30)** | **0.95 px** | **2.58 px** | **33.84 px** | **173.0 ms** |

*Files: `experiments/baseline_table.csv` and `experiments/benchmark_leaderboard.json`*

---

## 🔬 Component Ablation Study

Quantifies the step-by-step contribution of each algorithmic layer:

| Algorithmic Component | Pass @ 5.0 px | Pass @ 1.0 px | Median Error | Mean Error | Worst Error | Impact / Rationale |
|---|---|---|---|---|---|---|
| **1. Classical ZNCC Baseline** | 70.0% | 33.3% | 1.30 px | 51.44 px | 706.72 px | Coarse template matching baseline. |
| **2. + 2D Parabolic Peak Fitting** | 70.0% | 40.0% | 1.31 px | 51.45 px | 706.73 px | Boosts fine-accuracy sub-pixel precision (< 0.5 px). |
| **3. + Stage Memory & Kalman Prior** | 83.3% | 53.3% | 0.98 px | 15.63 px | 303.76 px | Eliminates 4 catastrophic periodic hops in DRAM/FinFET. |
| **4. + Stage-Gated ROI + Re-Ranker** | **90.0%** | **56.7%** | **0.95 px** | **2.58 px** | **33.84 px** | **Full System: 20× mean error reduction & 21× worst-case cut.** |

*File: `experiments/ablation_table.csv`*

---

## 🌐 Out-of-Distribution Generalization Benchmark

Stress-tested on 4 severe out-of-distribution SEM acquisition regimes:

| Stress Domain | Pass @ 5.0 px | Pass @ 1.0 px | Median Error | Mean Error | Worst Error |
|---|---|---|---|---|---|
| **Nominal (In-Distribution)** | **100.0%** | 40.0% | 1.08 px | 1.02 px | 1.53 px |
| **Extreme Low-Dose (20 e⁻/px shot noise)** | **100.0%** | 30.0% | 1.07 px | 1.01 px | 1.51 px |
| **Severe Stage Drift (4.0 px shear)** | **90.0%** | 20.0% | 2.28 px | 8.36 px | 66.01 px |
| **High Charging Breakdown Streaks (40% prob)** | **100.0%** | 40.0% | 1.08 px | 0.99 px | 1.44 px |

*File: `experiments/generalization_table.csv`*

---

## 🏛️ System Architecture

```text
smart-sem/
├── research/                    # Research database & training artifacts
│   ├── literature_review.md     # Survey of 20+ computer vision & SEM papers
│   ├── baseline_methods.md      # ZNCC, Phase Correlation, SIFT, LightGlue, LoFTR
│   ├── novelty_gaps.md          # Identified industry gaps & SMART-SEM novelty
│   ├── proposed_method.md       # Multi-layer system specification
│   ├── techniques.json          # 10 CV/SEM alignment techniques analyzed
│   ├── github_repos.json        # 8 SOTA mined repositories
│   ├── patent_database.json     # Industrial patents (AMAT, KLA, ASML)
│   └── ranking_training_dataset.json # Feature vectors with ground-truth binary labels
├── experiments/                 # Automated benchmarks & ablation artifacts
│   ├── baseline_table.csv       # Leaderboard summary
│   ├── benchmark_leaderboard.json
│   ├── ablation_table.csv       # Step-by-step component ablation
│   ├── generalization_table.csv # Out-of-distribution stress benchmark
│   └── generalization_benchmark.py
├── results/
│   ├── failure_gallery/         # Visual failure diagnostics with overlays & JSONs
│   └── evaluation/              # Batch prediction outputs & metrics summary
├── notebooks/                   # 7 Independent Google Colab Notebooks
├── tests/                       # Automated unit tests (17/17 passing OK)
├── src/smart_sem/               # Core Industrial Engine:
│   ├── localization_engine.py   # Unified scale-aware matcher & candidate pool
│   ├── candidate_reranker.py    # Multi-domain feature re-ranker
│   ├── kalman_memory.py         # 2D Kinematic Kalman Stage Tracker
│   ├── topology_verification.py # Line/Corner graph consistency (TCS)
│   ├── finfet_structure_tensor.py # 2D Structure Tensor & FinFET junction analysis
│   ├── navigation.py            # SEM Stage Drift, Backlash, and Vibration Simulator
│   ├── failure_gallery.py       # Visual diagnostic generator
│   └── failure_analysis.py      # Automated Types A–E taxonomy engine
├── generate_dataset.py          # Synthetic dataset generator CLI
├── localize.py                  # Standalone batch localization engine CLI
├── solution_presentation.pptx   # Official 12-slide dark-themed presentation
└── requirements.txt             # Environment dependencies
```

---

## 🛠️ Quickstart Commands

### 1. Installation & Environment
```bash
pip install -r requirements.txt
```

### 2. Dataset Generation (30 Varied Pairs)
```bash
python generate_dataset.py --num-samples 30 --out-dir results/dataset
```

### 3. Batch Localization & Metric Evaluation
```bash
python localize.py --manifest results/dataset/manifest.csv --out-dir results/evaluation
```

### 4. Run Ablation & Generalization Benchmarks
```bash
python experiments/ablation_study.py
python experiments/generalization_benchmark.py
```

### 5. Run Automated Unit Tests
```bash
python -m unittest discover tests
```

### 6. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
