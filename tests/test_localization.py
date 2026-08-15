"""
Unit tests for Hybrid Localization Engine & Topology Discovery.
"""
import unittest
import numpy as np
import cv2
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smart_sem.topology import discover_topology
from src.smart_sem.hybrid_localization import smart_sem_hybrid_localize

class TestLocalization(unittest.TestCase):
    def test_topology_discovery(self):
        ref = np.zeros((1000, 1000), dtype=np.uint8)
        for x in range(0, 1000, 70):
            ref[:, x:x+10] = 255
        res = discover_topology(ref)
        self.assertIn("pitch_nm", res)
        self.assertIn("adaptive_strategy", res)

    def test_hybrid_localization(self):
        ref = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        search = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        search[450:550, 450:550] = cv2.resize(ref, (100, 100))

        loc_res = smart_sem_hybrid_localize(ref, search, scales=(10.0,))
        self.assertIn("pred_x", loc_res)
        self.assertIn("confidence", loc_res)
        self.assertIn("top_k_candidates", loc_res)

if __name__ == "__main__":
    unittest.main()
