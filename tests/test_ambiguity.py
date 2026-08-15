"""
Unit tests for Ambiguity Intelligence and Wafer Memory Graph.
"""
import unittest
import numpy as np
import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smart_sem.ambiguity_intelligence import analyze_ambiguity_intelligence
from src.smart_sem.memory import WaferMemoryGraph

class TestAmbiguityAndMemory(unittest.TestCase):
    def test_ambiguity_intelligence(self):
        loc_res = {
            "top_k_candidates": [
                {"rank": 1, "center_x": 500.0, "center_y": 500.0, "score": 0.85},
                {"rank": 2, "center_x": 570.0, "center_y": 500.0, "score": 0.80},
            ],
            "similarity_map": np.random.rand(100, 100).astype(np.float32)
        }
        amb_res = analyze_ambiguity_intelligence(loc_res)
        self.assertIn("confidence_calibrated", amb_res)
        self.assertIn("ambiguity_class", amb_res)
        self.assertEqual(amb_res["ambiguity_class"], "REPEATED_PATTERN_AMBIGUITY")

    def test_wafer_memory_graph(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mem_file = f.name
        try:
            mem = WaferMemoryGraph(mem_file)
            mem.update_fingerprint("wafer_01", "dram_1x", {"pitch_nm": 70.0, "orientation_deg": 0.0})
            
            prior = mem.find_nearest_wafer_prior("dram_1x", 70.5, 0.5)
            self.assertIsNotNone(prior)
            self.assertEqual(prior["wafer_id"], "wafer_01")
        finally:
            if os.path.exists(mem_file):
                os.remove(mem_file)

if __name__ == "__main__":
    unittest.main()
