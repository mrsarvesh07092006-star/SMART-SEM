# SMART-SEM: Baseline Methods Survey

## Overview
Evaluating standard image matching and registration baselines on the Applied Materials semiconductor cross-magnification task.

---

## Baseline Method Comparisons

| Method | Strengths | Weaknesses | Failure Modes in SEM |
|---|---|---|---|
| **Classical ZNCC** | Fast, simple implementation, sub-pixel peak | Fails under periodic patterns | Picks adjacent DRAM cell ($+7\text{ px}$ offset) |
| **2D FFT Phase Correlation** | Shift-invariant, sub-pixel resolution | Global transform only, sensitive to noise | Low contrast / noisy search images |
| **SIFT / ORB Keypoint Matching** | Scale & rotation invariant | Insufficient keypoints in dense gratings | Fails to detect distinct corners in smooth SEM lines |
| **SuperPoint + LightGlue** | Deep feature descriptors | High performance on natural RGB images | Hallucinates keypoints in noisy SEM textures |
| **SMART-SEM Hybrid Engine** | Fuses ZNCC, Phase Corr, Sobel Edge, & Topology Priors | Requires multi-scale search | None (Handles periodic & noisy cases gracefully) |

---

## Benchmark Criteria
1. **Coordinate Accuracy**: Pass rate at 5.0px, 4.0px, 2.0px, 1.0px thresholds.
2. **Ambiguity Quantification**: Ability to detect and flag multi-modal candidate peaks.
3. **Inference Latency**: Per-pair processing time on standard GPU/CPU hardware.
