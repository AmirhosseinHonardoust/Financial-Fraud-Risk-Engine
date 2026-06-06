from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.threshold_policy import (
    build_threshold_policy_candidates,
    build_threshold_policy_summary,
    save_threshold_policy_artifacts,
)


class ThresholdPolicyTests(unittest.TestCase):
    def _threshold_results(self) -> list[dict]:
        return [
            {
                "threshold": 0.20,
                "tp": 95,
                "fp": 55,
                "tn": 845,
                "fn": 5,
                "precision": 0.633,
                "recall": 0.950,
                "f1": 0.760,
                "false_positive_rate": 0.061,
                "specificity": 0.939,
                "flagged_rate": 0.150,
                "cost": 105.0,
                "normalized_cost": 0.105,
            },
            {
                "threshold": 0.40,
                "tp": 88,
                "fp": 20,
                "tn": 880,
                "fn": 12,
                "precision": 0.815,
                "recall": 0.880,
                "f1": 0.846,
                "false_positive_rate": 0.022,
                "specificity": 0.978,
                "flagged_rate": 0.108,
                "cost": 140.0,
                "normalized_cost": 0.140,
            },
            {
                "threshold": 0.65,
                "tp": 72,
                "fp": 5,
                "tn": 895,
                "fn": 28,
                "precision": 0.935,
                "recall": 0.720,
                "f1": 0.813,
                "false_positive_rate": 0.006,
                "specificity": 0.994,
                "flagged_rate": 0.077,
                "cost": 285.0,
                "normalized_cost": 0.285,
            },
        ]

    def test_build_threshold_policy_candidates_returns_expected_policies(self) -> None:
        candidates = build_threshold_policy_candidates(self._threshold_results())
        policies = {row["policy"] for row in candidates}

        self.assertIn("cost_optimized", policies)
        self.assertIn("balanced_f1", policies)
        self.assertIn("high_recall", policies)
        self.assertIn("high_precision", policies)
        self.assertIn("review_capacity", policies)

        for row in candidates:
            self.assertIn("threshold", row)
            self.assertIn("rationale", row)
            self.assertGreaterEqual(float(row["threshold"]), 0.0)
            self.assertLessEqual(float(row["threshold"]), 1.0)

    def test_build_threshold_policy_summary_contains_cost_assumptions(self) -> None:
        rows = self._threshold_results()
        summary = build_threshold_policy_summary(
            rows,
            rows[0],
            cost_false_positive=1.0,
            cost_false_negative=10.0,
        )

        self.assertEqual(summary["recommended_policy"], "cost_optimized")
        self.assertIn("cost_assumptions", summary)
        self.assertIn("policy_candidates", summary)
        self.assertGreater(len(summary["policy_candidates"]), 0)

    def test_save_threshold_policy_artifacts_writes_json_csv_and_markdown(self) -> None:
        rows = self._threshold_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_threshold_policy_artifacts(
                rows,
                rows[0],
                metrics_dir=Path(tmpdir),
                cost_false_positive=1.0,
                cost_false_negative=10.0,
            )

            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["markdown"].exists())

            data = json.loads(paths["json"].read_text())
            self.assertIn("policy_candidates", data)
            self.assertIn("cost_assumptions", data)
            self.assertIn("Policy candidates", paths["markdown"].read_text())


if __name__ == "__main__":
    unittest.main()
