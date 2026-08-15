"""
Unit tests for Navigation Error Simulator.
"""
import unittest
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smart_sem.navigation import NavigationErrorSimulator, NavigationParams, apply_navigation_error_to_gt

class TestNavigation(unittest.TestCase):
    def test_navigation_simulator_move(self):
        nav_sim = NavigationErrorSimulator(NavigationParams(drift_sigma_nm=2.0), rng=np.random.default_rng(42))
        ax, ay, details = nav_sim.simulate_move(100.0, 50.0)
        self.assertEqual(details["step"], 1)
        self.assertIn("cumulative_error_nm", details)

    def test_navigation_trajectory(self):
        nav_sim = NavigationErrorSimulator(rng=np.random.default_rng(42))
        cum_x, cum_y, report = nav_sim.generate_cumulative_trajectory_error(n_steps=5)
        self.assertEqual(report["n_steps"], 5)
        self.assertIn("euclidean_nav_error_px", report)

    def test_apply_navigation_offset(self):
        obs_x, obs_y = apply_navigation_error_to_gt(500.0, 500.0, (2.5, -1.0))
        self.assertEqual(obs_x, 502.5)
        self.assertEqual(obs_y, 499.0)

if __name__ == "__main__":
    unittest.main()
