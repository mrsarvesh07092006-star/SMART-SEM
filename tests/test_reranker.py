"""
Unit tests for Candidate Re-Ranker, Kalman Tracker, and Topology Verification.
"""
import unittest
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smart_sem.kalman_memory import StageKalmanTracker
from src.smart_sem.topology_verification import compute_topology_consistency_score, extract_patch_topology_signature
from src.smart_sem.candidate_reranker import CandidateReRanker

class TestRerankerAndKalman(unittest.TestCase):
    def test_kalman_tracker(self):
        tracker = StageKalmanTracker()
        tracker.update(500.0, 500.0)
        px, py, vx, vy = tracker.predict()
        self.assertAlmostEqual(px, 500.0, delta=1.0)
        self.assertAlmostEqual(py, 500.0, delta=1.0)
        dist = tracker.compute_mahalanobis_distance(505.0, 500.0)
        self.assertGreater(dist, 0.0)

    def test_topology_verification(self):
        patch1 = np.zeros((100, 100), dtype=np.uint8)
        for i in range(0, 100, 10):
            patch1[:, i:i+2] = 255
        
        patch2 = patch1.copy()
        tcs = compute_topology_consistency_score(patch1, patch2)
        self.assertGreaterEqual(tcs, 0.90)

    def test_candidate_reranker(self):
        reranker = CandidateReRanker()
        ref = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        search = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        cands = [
            {"rank": 1, "center_x": 500.0, "center_y": 500.0, "score": 0.85, "tw": 100, "th": 100},
            {"rank": 2, "center_x": 548.0, "center_y": 500.0, "score": 0.84, "tw": 100, "th": 100},
        ]
        best, ranked = reranker.rerank(cands, ref, search, stage_prior_xy=(548.0, 500.0), confidence_margin=1.01)
        self.assertEqual(len(ranked), 2)
        self.assertIn("ranking_score", ranked[0])

    def test_finfet_structure_tensor(self):
        from src.smart_sem.finfet_structure_tensor import compute_structure_tensor_signature, compute_finfet_junction_similarity
        p1 = np.zeros((100, 100), dtype=np.uint8)
        for i in range(0, 100, 10): p1[:, i:i+2] = 255
        for j in range(0, 100, 10): p1[j:j+2, :] = 255
        sig = compute_structure_tensor_signature(p1)
        self.assertIn("coherence", sig)
        self.assertGreater(sig["junction_count"], 0)
        sim = compute_finfet_junction_similarity(p1, p1.copy())
        self.assertGreaterEqual(sim, 0.90)

if __name__ == "__main__":
    unittest.main()
