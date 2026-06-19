from __future__ import annotations

import unittest

import pandas as pd

from src.config import TARGET_COL
from src.dashboard_utils import add_risk_band, build_model_metadata, summarize_scored_transactions


class DashboardHelperTests(unittest.TestCase):
    def test_add_risk_band_assigns_expected_labels(self) -> None:
        df = pd.DataFrame(
            {
                "fraud_probability": [0.10, 0.30, 0.60, 0.90],
                "fraud_flag": [0, 0, 1, 1],
            }
        )

        result = add_risk_band(df)

        self.assertEqual(
            result["risk_band"].tolist(),
            ["Low", "Medium", "High", "Critical"],
        )

    def test_summarize_scored_transactions_returns_dashboard_metrics(self) -> None:
        df = pd.DataFrame(
            {
                "fraud_probability": [0.90, 0.70, 0.20, 0.10],
                "fraud_flag": [1, 1, 0, 0],
                TARGET_COL: [1, 0, 0, 0],
            }
        )

        summary = summarize_scored_transactions(df, threshold=0.5)

        self.assertEqual(summary["total_transactions"], 4)
        self.assertEqual(summary["flagged_count"], 2)
        self.assertAlmostEqual(summary["flagged_rate"], 0.5)
        self.assertAlmostEqual(summary["true_fraud_rate"], 0.25)
        self.assertEqual(summary["selected_threshold"], 0.5)
        self.assertGreaterEqual(summary["p95_probability"], 0.0)
        self.assertLessEqual(summary["p95_probability"], 1.0)

    def test_summarize_scored_transactions_handles_missing_target(self) -> None:
        df = pd.DataFrame(
            {
                "fraud_probability": [0.80, 0.20],
                "fraud_flag": [1, 0],
            }
        )

        summary = summarize_scored_transactions(df, threshold=0.5)

        self.assertIsNone(summary["true_fraud_rate"])
        self.assertEqual(summary["flagged_count"], 1)

    def test_build_model_metadata_prefers_saved_values(self) -> None:
        class DummyPipeline:
            named_steps = {"clf": object()}

        metadata = build_model_metadata(
            DummyPipeline(),
            {"threshold": 0.35},
            {
                "roc_auc": 0.91,
                "average_precision": 0.77,
                "brier_score": 0.08,
                "n_train_samples": 100,
            },
            {"best_threshold": {"threshold": 0.5, "cost": 10}},
        )

        self.assertEqual(metadata["saved_threshold"], 0.35)
        self.assertEqual(metadata["roc_auc"], 0.91)
        self.assertEqual(metadata["average_precision"], 0.77)
        self.assertEqual(metadata["brier_score"], 0.08)
        self.assertIn("synthetic", metadata["data_note"].lower())


if __name__ == "__main__":
    unittest.main()
