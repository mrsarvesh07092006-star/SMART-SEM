"""
SMART-SEM Layer 3: Ambiguity Intelligence Module.

Calculates:
- Full similarity map Shannon Entropy
- Softmax Candidate Probability Distribution across Top-K peaks
- Calibrated Confidence Score
- Peak Separation Distance & Ambiguity Risk Categorization
"""

from __future__ import annotations
import math
import numpy as np

def compute_similarity_entropy(similarity_map: np.ndarray) -> float:
    """Computes spatial Shannon entropy of normalized similarity map."""
    if similarity_map is None or similarity_map.size == 0:
        return 0.0

    # Clip to non-negative values and normalize to probability distribution
    clipped = np.clip(similarity_map, 0.0, None)
    total = np.sum(clipped)
    if total < 1e-6:
        return 0.0

    prob = clipped / total
    # Non-zero probabilities
    nz_prob = prob[prob > 1e-8]
    entropy = -np.sum(nz_prob * np.log2(nz_prob))
    # Normalize by max possible entropy log2(N)
    max_entropy = np.log2(float(similarity_map.size))
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    return norm_entropy

def analyze_ambiguity_intelligence(loc_result: dict) -> dict:
    """
    Computes candidate probability distribution, entropy, peak separation, and calibrated confidence.
    """
    candidates = loc_result.get("top_k_candidates", [])
    sim_map = loc_result.get("similarity_map")

    if not candidates:
        return {
            "confidence_calibrated": 0.0,
            "entropy": 1.0,
            "top_candidates_distribution": [],
            "peak_separation_px": 0.0,
            "ambiguity_class": "EXTREME_AMBIGUITY_NO_MATCH",
            "is_ambiguous": True,
        }

    scores = np.array([c["score"] for c in candidates])
    # Softmax candidate probability distribution (temperature tau=0.1)
    exp_scores = np.exp((scores - np.max(scores)) / 0.1)
    probs = (exp_scores / np.sum(exp_scores)).tolist()

    top1 = candidates[0]
    top1_score = top1["score"]

    if len(candidates) > 1:
        top2 = candidates[1]
        top2_score = top2["score"]
        peak_sep = float(math.hypot(top1["center_x"] - top2["center_x"], top1["center_y"] - top2["center_y"]))
        peak_ratio = top2_score / (top1_score + 1e-6)
    else:
        top2_score = 0.0
        peak_sep = 999.0
        peak_ratio = 0.0

    entropy = compute_similarity_entropy(sim_map)

    # Confidence calibration: penalize high entropy and high peak ratio
    calibrated_conf = float(top1_score * (1.0 - 0.5 * peak_ratio) * (1.0 - 0.3 * entropy))
    calibrated_conf = float(np.clip(calibrated_conf, 0.0, 1.0))

    # Categorize Ambiguity Risk
    if peak_ratio > 0.85:
        amb_class = "REPEATED_PATTERN_AMBIGUITY"
    elif entropy > 0.80:
        amb_class = "HIGH_ENTROPY_DIFFUSE_MATCH"
    elif top1_score < 0.35:
        amb_class = "LOW_SIGNAL_NOISE"
    else:
        amb_class = "HIGH_CONFIDENCE_UNIQUE_MATCH"

    candidate_distribution = []
    for c, p in zip(candidates, probs):
        candidate_distribution.append({
            "rank": c["rank"],
            "center_x": c["center_x"],
            "center_y": c["center_y"],
            "raw_score": c["score"],
            "probability": float(p),
        })

    return {
        "confidence_calibrated": calibrated_conf,
        "raw_top1_score": top1_score,
        "raw_top2_score": top2_score,
        "peak_ratio": float(peak_ratio),
        "entropy": entropy,
        "peak_separation_px": peak_sep,
        "top_candidates_distribution": candidate_distribution,
        "ambiguity_class": amb_class,
        "is_ambiguous": bool(peak_ratio > 0.80 or entropy > 0.85),
    }
