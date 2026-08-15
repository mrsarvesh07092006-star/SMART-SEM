# SMART-SEM: Novelty & Research Gaps

## Identified Gaps in Current Semiconductor Inspection Systems
1. **Silent Failure in Repetitive Structures**: Existing tools return a single $(x, y)$ coordinate without indicating whether the match is unambiguous or one of 20 identical periodic cell candidates.
2. **Ignorance of Stage Navigation Mechanics**: Current synthetic generators model image noise but ignore physical stage drift, backlash, and thermal positioning errors accumulated prior to imaging.
3. **Static Match Strategy**: Standard matchers use fixed parameters regardless of whether the target is a periodic DRAM array, a FinFET grating, or a complex logic layout.

---

## SMART-SEM Key Novelties
1. **Dual Stage-Physics + Image-Physics Simulator**: Models both optical degradation and physical SEM hardware stage positioning drift.
2. **Topology-Guided Adaptive Matching Strategy**: Dynamic adjustment of search radius, multi-scale bracketing, and ambiguity thresholds based on 2D FFT spectral discovery.
3. **Ambiguity & Confusion Intelligence**: Computes similarity map entropy and visualizes risk zones (Repeated vs Unique vs Risk regions).
4. **Memory-Guided Wafer Navigation**: Uses historical wafer fingerprints and defect priors to narrow search bounds and boost localization confidence.
