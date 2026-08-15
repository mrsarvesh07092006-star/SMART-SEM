# SMART-SEM: Proposed Method Architecture

## Architectural Overview

SMART-SEM implements an 8-layer semiconductor-aware localization architecture:

```
[Layer 1: SEM Physics & Navigation Simulator]
   ├── Image Physics (Blur, Shot Noise, Charging, Speckle, Astigmatism)
   └── Navigation Mechanics (Stage Drift, Backlash, Thermal Shift, Vibration)
            ↓
[Layer 5: Topology Discovery & Strategy Adaptation]
   ├── 2D FFT Power Spectrum Analysis (Pitch & Orientation Estimation)
   └── Dynamic Parameter Adjustment (Search Bracket, Candidate Count, Ambiguity Cutoff)
            ↓
[Layer 6 & 7: Wafer Memory Graph & Historical Defect Priors]
   ├── Fingerprint Nearest-Neighbor Search
   └── Search Region Prior Guidance
            ↓
[Layer 2: Hybrid Multi-Stream Localization Engine]
   ├── Stream A: Multi-Scale ZNCC Intensity Correlation
   ├── Stream B: 2D FFT Phase Correlation
   └── Stream C: Sobel Gradient Magnitude Correlation
   └── Fusion: Weighted Hybrid Candidate Ranking + Sub-Pixel Fitting
            ↓
[Layer 3: Ambiguity Intelligence Module]
   ├── Multi-Hypothesis Top-K Extraction (NMS)
   ├── Entropy Calculation & Confidence Calibration
   └── Ambiguity Risk Categorization
            ↓
[Layer 4: Confusion Intelligence]
   ├── Colorized Similarity Heatmap (Confusion Map)
   └── Zone Segmentation (Repeated Pattern vs Unique vs Risk Regions)
            ↓
[Layer 8: Cross-Modal SEM ↔ RGB Extension]
   └── Modality-Invariant Topology Representation & Transfer
```

---

## Technical Specifications
- **Scale ratio**: 10:1 physical scale difference ($1\text{ nm/px}$ Ref vs $10\text{ nm/px}$ Search).
- **Sub-pixel refinement**: 2D parabolic quadratic interpolation around discrete similarity peaks.
- **Explainability outputs**: `predictions.csv`, `metrics_summary.json`, colorized overlay confusion maps.
