from __future__ import annotations

import unittest

import numpy as np

try:
    import shap  # noqa: F401
    import streamlit  # noqa: F401

    HAS_DASHBOARD_DEPS = True
except Exception:  # pragma: no cover - streamlit/shap are optional heavy deps
    HAS_DASHBOARD_DEPS = False


@unittest.skipUnless(HAS_DASHBOARD_DEPS, "streamlit and shap are required")
class AppDashboardTests(unittest.TestCase):
    """Smoke tests for app.py helpers; skipped cleanly without streamlit/shap."""

    def test_plot_single_shap_bar_returns_figure(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.figure

        import app

        shap_values = np.array([0.5, -0.3, 0.1, 0.0, 0.2])
        feature_names = np.array(["f0", "f1", "f2", "f3", "f4"])

        fig = app.plot_single_shap_bar(shap_values, feature_names, max_features=3)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_explain_single_transaction_end_to_end(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.figure

        import app
        from src.config import TARGET_COL
        from src.features import build_pipeline
        from src.generate_synthetic_data import generate_synthetic_fraud_dataset
        from src.score_new_transactions import score_dataframe

        df = generate_synthetic_fraud_dataset(n_samples=300, fraud_rate=0.2, seed=5)
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        model = build_pipeline()
        model.fit(X, y)

        scored = score_dataframe(df, threshold=0.5, model=model)
        fig, reasons = app.explain_single_transaction(model, scored, 0)

        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertIsInstance(reasons, list)
        self.assertTrue(len(reasons) > 0)


if __name__ == "__main__":
    unittest.main()
