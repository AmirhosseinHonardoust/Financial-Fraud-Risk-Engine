from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.config import NUMERIC_FEATURES, TARGET_COL
from src.features import build_pipeline
from src.score_new_transactions import score_dataframe
from src.validation import (
    DataValidationError,
    validate_binary_target,
    validate_scoring_dataframe,
    validate_threshold,
    validate_training_dataframe,
)


class ValidationAndScoringTests(unittest.TestCase):
    def _demo_transactions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "transaction_id": range(1, 9),
                "user_id": [101, 102, 103, 104, 105, 106, 107, 108],
                "amount": [20.0, 450.0, 35.0, 800.0, 15.0, 900.0, 42.0, 1000.0],
                "hour": [9, 23, 10, 2, 14, 1, 11, 3],
                "device_risk_score": [0.05, 0.9, 0.1, 0.85, 0.02, 0.95, 0.07, 0.88],
                "ip_risk_score": [0.03, 0.8, 0.12, 0.9, 0.04, 0.93, 0.1, 0.86],
                "transaction_type": [
                    "purchase",
                    "transfer",
                    "purchase",
                    "transfer",
                    "purchase",
                    "transfer",
                    "purchase",
                    "transfer",
                ],
                "merchant_category": [
                    "retail",
                    "crypto",
                    "grocery",
                    "electronics",
                    "retail",
                    "crypto",
                    "grocery",
                    "electronics",
                ],
                "country": ["DE", "RU", "DE", "CN", "DE", "RU", "FR", "CN"],
                TARGET_COL: [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

    def test_training_dataframe_validation_accepts_valid_schema(self) -> None:
        df = self._demo_transactions()
        validate_training_dataframe(df)

    def test_scoring_dataframe_does_not_require_target(self) -> None:
        df = self._demo_transactions().drop(columns=[TARGET_COL])
        validate_scoring_dataframe(df)

    def test_missing_required_feature_raises_clear_error(self) -> None:
        df = self._demo_transactions().drop(columns=[NUMERIC_FEATURES[0]])
        with self.assertRaisesRegex(DataValidationError, "missing required columns"):
            validate_scoring_dataframe(df)

    def test_invalid_numeric_feature_raises(self) -> None:
        df = self._demo_transactions()
        df["amount"] = df["amount"].astype(object)
        df.loc[0, "amount"] = "not-a-number"
        with self.assertRaisesRegex(DataValidationError, "invalid numeric"):
            validate_scoring_dataframe(df)

    def test_binary_target_validation_rejects_non_binary_labels(self) -> None:
        df = self._demo_transactions()
        df.loc[0, TARGET_COL] = 2
        with self.assertRaisesRegex(DataValidationError, "binary"):
            validate_binary_target(df)

    def test_threshold_validation_rejects_invalid_values(self) -> None:
        for value in [-0.1, 1.1]:
            with self.subTest(value=value), self.assertRaises(DataValidationError):
                validate_threshold(value)

    def test_score_dataframe_adds_probability_and_flag(self) -> None:
        df = self._demo_transactions()
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        model = build_pipeline()
        model.fit(X, y)

        scored = score_dataframe(df, threshold=0.5, model=model)

        self.assertIn("fraud_probability", scored.columns)
        self.assertIn("fraud_flag", scored.columns)
        self.assertIn("reason_codes", scored.columns)
        self.assertEqual(len(scored), len(df))
        self.assertTrue(scored["fraud_probability"].between(0, 1).all())
        self.assertTrue(set(scored["fraud_flag"].unique()).issubset({0, 1}))
        self.assertTrue(
            np.all(
                scored["fraud_probability"].to_numpy()[:-1]
                >= scored["fraud_probability"].to_numpy()[1:]
            )
        )


if __name__ == "__main__":
    unittest.main()
