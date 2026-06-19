from __future__ import annotations

import unittest

import pandas as pd

from src.reason_codes import (
    add_reason_codes,
    high_amount_cutoff,
    humanize_feature_name,
    reason_codes_for_row,
    shap_reason_codes,
    split_reason_codes,
)


class ReasonCodeTests(unittest.TestCase):
    def _demo_scored(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "transaction_id": [1, 2, 3],
                "amount": [25.0, 900.0, 80.0],
                "hour": [12, 2, 23],
                "device_risk_score": [0.10, 0.92, 0.65],
                "ip_risk_score": [0.08, 0.88, 0.70],
                "transaction_type": ["purchase", "transfer", "purchase"],
                "merchant_category": ["grocery", "crypto", "retail"],
                "country": ["DE", "RU", "FR"],
                "fraud_probability": [0.05, 0.97, 0.55],
                "fraud_flag": [0, 1, 1],
            }
        )

    def test_high_amount_cutoff_uses_batch_quantile(self) -> None:
        df = self._demo_scored()
        cutoff = high_amount_cutoff(df, quantile=0.50)
        self.assertEqual(cutoff, 80.0)

    def test_reason_codes_for_high_risk_row_are_analyst_friendly(self) -> None:
        row = self._demo_scored().iloc[1]
        reasons = reason_codes_for_row(
            row,
            threshold=0.5,
            amount_cutoff=800.0,
            max_reasons=10,
        )

        joined = " | ".join(reasons).lower()
        self.assertIn("critical model risk", joined)
        self.assertIn("device risk", joined)
        self.assertIn("ip risk", joined)
        self.assertIn("amount", joined)

    def test_add_reason_codes_adds_string_column(self) -> None:
        result = add_reason_codes(self._demo_scored(), threshold=0.5)

        self.assertIn("reason_codes", result.columns)
        self.assertEqual(len(result), 3)
        self.assertTrue(result["reason_codes"].str.len().gt(0).all())

    def test_humanize_feature_name_handles_transformed_names(self) -> None:
        self.assertEqual(
            humanize_feature_name("numeric__device_risk_score"),
            "device risk score",
        )
        self.assertEqual(
            humanize_feature_name("categorical__merchant_category_crypto"),
            "merchant category = crypto",
        )

    def test_shap_reason_codes_use_direction(self) -> None:
        reasons = shap_reason_codes(
            [0.8, -0.4, 0.1],
            [
                "numeric__device_risk_score",
                "categorical__transaction_type_purchase",
                "numeric__amount",
            ],
            max_reasons=2,
        )

        self.assertEqual(len(reasons), 2)
        self.assertIn("device risk score increased fraud risk", reasons[0])
        self.assertIn("transaction type = purchase reduced fraud risk", reasons[1])

    def test_split_reason_codes_handles_empty_values(self) -> None:
        self.assertEqual(split_reason_codes(None), [])
        self.assertEqual(
            split_reason_codes("High risk; Large amount"), ["High risk", "Large amount"]
        )


if __name__ == "__main__":
    unittest.main()
