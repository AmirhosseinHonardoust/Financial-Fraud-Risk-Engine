from __future__ import annotations

from typing import Any

import pandas as pd

from .config import TARGET_COL


def add_risk_band(df_scored: pd.DataFrame) -> pd.DataFrame:
    """Add analyst-friendly risk bands based on fraud probability."""
    df = df_scored.copy()
    probabilities = df["fraud_probability"].clip(lower=0.0, upper=1.0)

    df["risk_band"] = pd.cut(
        probabilities,
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True,
    ).astype(str)

    return df


def summarize_scored_transactions(
    df_scored: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """Return dashboard summary metrics for a scored transaction dataframe."""
    n_rows = int(len(df_scored))
    fraud_probability = df_scored["fraud_probability"] if n_rows else pd.Series(dtype=float)
    fraud_flag = df_scored["fraud_flag"] if n_rows else pd.Series(dtype=int)

    summary: dict[str, Any] = {
        "total_transactions": n_rows,
        "selected_threshold": float(threshold),
        "flagged_count": int(fraud_flag.sum()) if n_rows else 0,
        "flagged_rate": float(fraud_flag.mean()) if n_rows else 0.0,
        "average_probability": float(fraud_probability.mean()) if n_rows else 0.0,
        "p95_probability": float(fraud_probability.quantile(0.95)) if n_rows else 0.0,
        "max_probability": float(fraud_probability.max()) if n_rows else 0.0,
    }

    if TARGET_COL in df_scored.columns and n_rows:
        summary["true_fraud_rate"] = float(df_scored[TARGET_COL].mean())
    else:
        summary["true_fraud_rate"] = None

    return summary


def build_model_metadata(
    model,
    threshold_info: dict[str, Any],
    train_metrics: dict[str, Any],
    evaluation_summary: dict[str, Any],
) -> dict[str, Any]:
    """Collect lightweight model metadata for the dashboard."""
    model_step = getattr(model, "named_steps", {}).get("clf") if model is not None else None
    best_threshold = evaluation_summary.get("best_threshold", {})
    threshold_value = threshold_info.get("threshold", best_threshold.get("threshold", 0.5))

    return {
        "model_type": type(model_step).__name__ if model_step is not None else type(model).__name__,
        "saved_threshold": threshold_value,
        "roc_auc": train_metrics.get("roc_auc", evaluation_summary.get("roc_auc")),
        "average_precision": train_metrics.get(
            "average_precision",
            evaluation_summary.get("average_precision"),
        ),
        "brier_score": train_metrics.get("brier_score", evaluation_summary.get("brier_score")),
        "positive_rate_train": train_metrics.get("positive_rate_train"),
        "positive_rate_test": train_metrics.get(
            "positive_rate_test",
            evaluation_summary.get("positive_rate_test"),
        ),
        "n_train_samples": train_metrics.get("n_train_samples"),
        "n_test_samples": train_metrics.get(
            "n_test_samples",
            evaluation_summary.get("n_test_samples"),
        ),
        "threshold_cost": best_threshold.get("cost"),
        "threshold_recall": best_threshold.get("recall"),
        "threshold_precision": best_threshold.get("precision"),
        "data_note": (
            "Synthetic demo data with overlapping classes and label noise. Metrics validate "
            "the workflow and should not be treated as real-world fraud benchmark performance."
        ),
    }


def format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def format_float(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"
