from __future__ import annotations

import unittest

import numpy as np

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
)
from src.data_prep import load_raw_data, train_test_split_stratified
from src.evaluate import compute_threshold_metrics, pick_best_threshold
from src.features import build_pipeline


class WorkflowContractTests(unittest.TestCase):
    def test_raw_data_has_expected_columns_and_binary_target(self) -> None:
        df = load_raw_data()

        expected_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL])
        self.assertTrue(expected_columns.issubset(df.columns))
        self.assertGreater(len(df), 0)
        self.assertEqual(set(df[TARGET_COL].dropna().unique()), {0, 1})

    def test_stratified_split_preserves_fraud_rate(self) -> None:
        df = load_raw_data()
        train_df, test_df = train_test_split_stratified(df)

        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)
        self.assertEqual(set(train_df[TARGET_COL].unique()), {0, 1})
        self.assertEqual(set(test_df[TARGET_COL].unique()), {0, 1})

        raw_rate = df[TARGET_COL].mean()
        self.assertAlmostEqual(train_df[TARGET_COL].mean(), raw_rate, delta=0.01)
        self.assertAlmostEqual(test_df[TARGET_COL].mean(), raw_rate, delta=0.01)

    def test_pipeline_can_fit_and_predict_probabilities(self) -> None:
        df = load_raw_data()
        # Build a small balanced sample so the classifier sees both classes.
        sample = (
            df.groupby(TARGET_COL, group_keys=False)
            .head(40)
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )

        X = sample.drop(columns=[TARGET_COL])
        y = sample[TARGET_COL]

        pipeline = build_pipeline()
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)[:, 1]

        self.assertEqual(len(proba), len(sample))
        self.assertTrue(np.isfinite(proba).all())
        self.assertTrue(((proba >= 0.0) & (proba <= 1.0)).all())

    def test_threshold_metrics_have_valid_values(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.05, 0.30, 0.70, 0.95])
        thresholds = [0.25, 0.50, 0.75]

        results = compute_threshold_metrics(y_true, y_proba, thresholds)
        best = pick_best_threshold(results)

        self.assertEqual(len(results), len(thresholds))
        self.assertIn(best, results)

        for row in results:
            self.assertIn("threshold", row)
            self.assertIn("precision", row)
            self.assertIn("recall", row)
            self.assertIn("f1", row)
            self.assertIn("cost", row)
            self.assertGreaterEqual(row["precision"], 0.0)
            self.assertLessEqual(row["precision"], 1.0)
            self.assertGreaterEqual(row["recall"], 0.0)
            self.assertLessEqual(row["recall"], 1.0)
            self.assertGreaterEqual(row["cost"], 0.0)
