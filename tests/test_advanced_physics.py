"""
Unit tests for Advanced SEM Acquisition Physics Engine (Agent 6).
"""
import unittest
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smart_sem.advanced_physics import (
    apply_line_edge_roughness,
    apply_focus_gradient,
    apply_charging_breakdown_streaks,
    apply_stitching_boundary_artifact,
    apply_advanced_sem_physics
)

class TestAdvancedPhysics(unittest.TestCase):
    def setUp(self):
        self.img = np.random.randint(50, 200, (1000, 1000), dtype=np.uint8)

    def test_line_edge_roughness(self):
        res = apply_line_edge_roughness(self.img, ler_sigma_px=1.5)
        self.assertEqual(res.shape, (1000, 1000))
        self.assertEqual(res.dtype, np.uint8)

    def test_focus_gradient(self):
        res = apply_focus_gradient(self.img, strength=0.4)
        self.assertEqual(res.shape, (1000, 1000))

    def test_charging_streaks(self):
        res = apply_charging_breakdown_streaks(self.img, streak_prob=3.0)
        self.assertEqual(res.shape, (1000, 1000))

    def test_full_advanced_pipeline(self):
        res = apply_advanced_sem_physics(self.img)
        self.assertEqual(res.shape, (1000, 1000))

if __name__ == "__main__":
    unittest.main()
