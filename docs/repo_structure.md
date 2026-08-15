# 📁 SMART-SEM Repository Structure & Dependency Graph (Agent 1 Audit)

## Overview
- **Repository Name**: SMART-SEM (Semiconductor-Aware Cross-Magnification Alignment Engine)
- **Track**: Applied Materials Drift-Sense Track — Semicon India Hackathon 2026
- **Architecture**: 6-Layer Modular Alignment Stack (Physics $\rightarrow$ Kalman Memory $\rightarrow$ Scale Search $\rightarrow$ Structure Tensor $\rightarrow$ Topology Verification $\rightarrow$ Multi-Domain Re-Ranking)

---

## 🗺️ Complete Directory Map

```text
smart-sem/
├── README.md                      # Comprehensive project documentation & leaderboard
├── requirements.txt               # Production dependencies (NumPy, OpenCV, SciPy, PyYAML)
├── setup.py                       # Setuptools packaging installer
├── LICENSE                        # Open-source MIT License
├── colab_requirements_check.py    # Environment & GPU validator for Colab
├── colab_setup.ipynb              # Top-level executable master Colab notebook
│
├── run_benchmark.py               # [ENTRYPOINT_BENCHMARK] Runs 30-pair official benchmark
├── run_ablation.py                # [ENTRYPOINT_ABLATION] Runs 4-layer component ablation
├── run_generalization.py          # [ENTRYPOINT_GENERALIZATION] Runs 4-domain stress tests
├── generate_dataset.py            # [ENTRYPOINT_DATASET] Generates 30 synthetic SEM pairs
├── localize.py                    # [ENTRYPOINT_RUN] Standalone batch localization engine
├── app.py                         # [ENTRYPOINT_UI] Interactive Streamlit visual inspection app
├── create_presentation.py         # Compiles 12-slide submission PPTX
├── solution_presentation.pptx     # 12-slide dark-themed presentation deck
│
├── src/smart_sem/                 # Core Algorithmic Framework:
│   ├── localization_engine.py     # Unified scale-aware matcher & candidate pool
│   ├── candidate_reranker.py      # Multi-domain feature re-ranker
│   ├── kalman_memory.py           # 2D Kinematic Kalman Stage Tracker
│   ├── topology_verification.py   # Line/Corner graph consistency scoring (TCS)
│   ├── finfet_structure_tensor.py # 2D Structure Tensor & FinFET junction analysis
│   ├── navigation.py              # SEM Stage Drift, Backlash, and Vibration Simulator
│   ├── advanced_physics.py        # Lithography LER, charging streaks, focus gradient
│   ├── failure_gallery.py         # Visual diagnostic generator
│   ├── failure_analysis.py        # Automated Types A–E taxonomy engine
│   ├── memory.py                  # Wafer Memory Graph
│   ├── confusion_intelligence.py  # Risk zone segmentation & overlay heatmaps
│   └── cross_modal.py             # SEM <-> RGB Optical Extension (Bonus)
│
├── experiments/                   # Benchmark harness & verified leaderboard outputs
│   ├── baseline_table.csv         # Leaderboard summary CSV
│   ├── benchmark_leaderboard.json # Leaderboard JSON
│   ├── ablation_table.csv         # 4-layer ablation results CSV
│   ├── ablation_study.py          # Ablation runner
│   ├── generalization_table.csv   # Stress benchmark CSV
│   └── generalization_benchmark.py# Generalization runner
│
├── research/                      # Literature, patents, and training artifacts
│   ├── literature_review.md       # Survey of 20+ computer vision & SEM papers
│   ├── baseline_methods.md        # Analysis of ZNCC, SIFT, LightGlue, LoFTR
│   ├── novelty_gaps.md            # Identified industry gaps & SMART-SEM novelty
│   ├── proposed_method.md         # Multi-layer system specification
│   ├── patent_database.json       # AMAT, KLA, ASML patents
│   ├── github_repos.json          # 8 SOTA mined repositories
│   ├── techniques.json            # 10 CV/SEM alignment techniques analyzed
│   └── ranking_training_dataset.json # Feature vectors with ground-truth binary labels
│
├── results/
│   ├── failure_gallery/           # Visual failure diagnostics (9 cases with overlays & JSONs)
│   ├── dataset/                   # 30 generated DRAM & FinFET pairs + manifest.csv
│   └── evaluation/                # Batch prediction outputs & metrics summary JSON
│
├── notebooks/                     # 8 Independent Google Colab Notebooks
│   ├── 00_SMART_SEM_Master_Colab.ipynb
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_physics_engine.ipynb
│   ├── 03_localization.ipynb
│   ├── 04_memory_graph.ipynb
│   ├── 05_ambiguity.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_final_demo.ipynb
│
└── tests/                         # Automated unit test suite (17/17 passing OK)
    ├── test_localization.py
    ├── test_navigation.py
    ├── test_ambiguity.py
    ├── test_failure_analysis.py
    ├── test_advanced_physics.py
    └── test_reranker.py
```

---

## 🔗 Entrypoint Identification

1. **`ENTRYPOINT_RUN`**: `python localize.py --manifest results/dataset/manifest.csv --out-dir results/evaluation`
2. **`ENTRYPOINT_BENCHMARK`**: `python run_benchmark.py`
3. **`ENTRYPOINT_ABLATION`**: `python run_ablation.py`
4. **`ENTRYPOINT_GENERALIZATION`**: `python run_generalization.py`
5. **`ENTRYPOINT_UI`**: `streamlit run app.py`
