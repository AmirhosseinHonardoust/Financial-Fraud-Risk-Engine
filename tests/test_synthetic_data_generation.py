from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL
from src.generate_synthetic_data import generate_synthetic_fraud_dataset, main
from src.validation import validate_training_dataframe


class SyntheticDataGenerationTests(unittest.TestCase):
    def test_generator_returns_valid_training_schema(self) -> None:
        df = generate_synthetic_fraud_dataset(n_samples=600, fraud_rate=0.08, seed=7)

        validate_training_dataframe(df)
        required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL])
        self.assertTrue(required.issubset(df.columns))
        self.assertEqual(len(df), 600)
        self.assertTrue(set(df[TARGET_COL].unique()).issubset({0, 1}))

    def test_generated_fraud_rate_is_reasonable(self) -> None:
        df = generate_synthetic_fraud_dataset(
            n_samples=2_000,
            fraud_rate=0.08,
            label_noise=0.02,
            seed=42,
        )

        fraud_rate = float(df[TARGET_COL].mean())
        self.assertGreater(fraud_rate, 0.04)
        self.assertLess(fraud_rate, 0.14)

    def test_generated_classes_have_feature_overlap(self) -> None:
        df = generate_synthetic_fraud_dataset(n_samples=2_000, seed=42)
        legit = df[df[TARGET_COL] == 0]
        fraud = df[df[TARGET_COL] == 1]

        # Fraud tends to be riskier, but the distributions should overlap.
        self.assertGreater(fraud["device_risk_score"].median(), legit["device_risk_score"].median())
        self.assertGreater(fraud["ip_risk_score"].median(), legit["ip_risk_score"].median())

        legit_high_risk_share = (legit["device_risk_score"] > 0.5).mean()
        fraud_low_risk_share = (fraud["device_risk_score"] < 0.5).mean()

        self.assertGreater(legit_high_risk_share, 0.01)
        self.assertGreater(fraud_low_risk_share, 0.05)

    def test_generator_cli_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "fraud.csv"
            main(
                [
                    "--output",
                    str(output),
                    "--rows",
                    "250",
                    "--fraud-rate",
                    "0.08",
                    "--seed",
                    "123",
                ]
            )

            self.assertTrue(output.exists())
            df = pd.read_csv(output)
            self.assertEqual(len(df), 250)
            validate_training_dataframe(df)


if __name__ == "__main__":
    unittest.main()
