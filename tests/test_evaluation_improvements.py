from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.config import TARGET_COL
from src.evaluate import (
    compute_baseline_metrics,
    compute_probability_metrics,
    compute_threshold_metrics,
    pick_best_threshold,
)


class EvaluationImprovementTests(unittest.TestCase):
    def test_probability_metrics_are_valid(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.05, 0.25, 0.75, 0.95])

        metrics = compute_probability_metrics(y_true, y_proba)

        self.assertIn("roc_auc", metrics)
        self.assertIn("average_precision", metrics)
        self.assertIn("brier_score", metrics)
        self.assertGreaterEqual(metrics["roc_auc"], 0.0)
        self.assertLessEqual(metrics["roc_auc"], 1.0)
        self.assertGreaterEqual(metrics["average_precision"], 0.0)
        self.assertLessEqual(metrics["average_precision"], 1.0)
        self.assertGreaterEqual(metrics["brier_score"], 0.0)

    def test_threshold_metrics_include_tradeoff_fields(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.05, 0.30, 0.70, 0.95])
        thresholds = [0.25, 0.50, 0.75]

        results = compute_threshold_metrics(y_true, y_proba, thresholds)
        best = pick_best_threshold(results)

        self.assertEqual(len(results), len(thresholds))
        self.assertIn(best, results)

        for row in results:
            for key in [
                "threshold",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "specificity",
                "flagged_rate",
                "cost",
                "normalized_cost",
            ]:
                self.assertIn(key, row)

            self.assertGreaterEqual(row["false_positive_rate"], 0.0)
            self.assertLessEqual(row["false_positive_rate"], 1.0)
            self.assertGreaterEqual(row["specificity"], 0.0)
            self.assertLessEqual(row["specificity"], 1.0)
            self.assertGreaterEqual(row["flagged_rate"], 0.0)
            self.assertLessEqual(row["flagged_rate"], 1.0)

    def test_baseline_metrics_return_expected_strategies(self) -> None:
        df = pd.DataFrame(
            {
                "transaction_id": range(1, 9),
                "user_id": range(101, 109),
                "amount": [10, 20, 30, 40, 50, 60, 70, 80],
                "hour": [1, 2, 3, 4, 5, 6, 7, 8],
                "device_risk_score": [0.1, 0.2, 0.1, 0.8, 0.9, 0.7, 0.2, 0.85],
                "ip_risk_score": [0.1, 0.1, 0.2, 0.8, 0.7, 0.9, 0.2, 0.8],
                "transaction_type": [
                    "purchase",
                    "purchase",
                    "purchase",
                    "transfer",
                    "transfer",
                    "transfer",
                    "purchase",
                    "transfer",
                ],
                "merchant_category": [
                    "retail",
                    "retail",
                    "grocery",
                    "crypto",
                    "crypto",
                    "electronics",
                    "grocery",
                    "crypto",
                ],
                "country": ["DE", "DE", "FR", "RU", "RU", "CN", "FR", "CN"],
                TARGET_COL: [0, 0, 0, 1, 1, 1, 0, 1],
            }
        )

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        baselines = compute_baseline_metrics(X, y, X, y)

        self.assertIn("majority_class", baselines)
        self.assertIn("empirical_prior", baselines)
        self.assertIn("stratified_random", baselines)

        for metrics in baselines.values():
            self.assertIn("roc_auc", metrics)
            self.assertIn("average_precision", metrics)
            self.assertIn("brier_score", metrics)
