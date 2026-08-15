"""
Unit tests for Failure Analysis and Diagnosis Agent (Agent 4).
"""
import unittest
import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smart_sem.failure_analysis import FailureAnalysisAgent

class TestFailureAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.db_path = self.tmp.name
        self.tmp.close()
        self.agent = FailureAnalysisAgent(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_periodic_ambiguity_diagnosis(self):
        diag = self.agent.diagnose_failure(
            sample_id="test_01",
            architecture="dram_1x",
            gt_x=500.0, gt_y=500.0,
            pred_x=549.0, pred_y=500.0, # Exactly 7 * 7px pitch offset
            confidence=0.82,
            topology_info={"pitch_px_search_est": 7.0}
        )
        self.assertTrue(diag["is_failure"])
        self.assertEqual(diag["category"], "TYPE_A_PERIODIC_AMBIGUITY")

    def test_noise_dominance_diagnosis(self):
        diag = self.agent.diagnose_failure(
            sample_id="test_02",
            architecture="finfet_10nm",
            gt_x=500.0, gt_y=500.0,
            pred_x=300.0, pred_y=200.0,
            confidence=0.25, # Very low confidence
            topology_info={"pitch_px_search_est": 5.0}
        )
        self.assertTrue(diag["is_failure"])
        self.assertEqual(diag["category"], "TYPE_C_NOISE_DOMINANCE")

if __name__ == "__main__":
    unittest.main()
