from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL

PREDICTION_COLUMNS = ["fraud_probability", "fraud_flag"]
REQUIRED_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_TRAINING_COLUMNS = REQUIRED_FEATURE_COLUMNS + [TARGET_COL]


class DataValidationError(ValueError):
    """Raised when transaction data does not satisfy the expected schema."""


def _as_list(values: Iterable[str]) -> list[str]:
    return [str(value) for value in values]


def validate_non_empty_dataframe(df: pd.DataFrame, *, context: str = "dataframe") -> None:
    """Validate universal dataframe contracts."""
    if df is None:
        raise DataValidationError(f"{context} is None.")

    if df.empty:
        raise DataValidationError(f"{context} is empty.")

    if len(df.columns) == 0:
        raise DataValidationError(f"{context} has no columns.")

    if df.columns.duplicated().any():
        duplicates = sorted({str(col) for col in df.columns[df.columns.duplicated(keep=False)]})
        raise DataValidationError(f"{context} contains duplicate column names: {duplicates}")


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    context: str = "dataframe",
) -> None:
    """Validate that a dataframe has all required columns."""
    validate_non_empty_dataframe(df, context=context)

    required = _as_list(required_columns)
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise DataValidationError(
            f"{context} is missing required columns: {missing}. "
            f"Required columns are: {required}"
        )


def validate_no_missing_required_values(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    context: str = "dataframe",
) -> None:
    """Validate that required columns do not contain missing values."""
    cols = [col for col in required_columns if col in df.columns]
    missing_counts = df[cols].isna().sum()
    bad = {str(col): int(count) for col, count in missing_counts.items() if int(count) > 0}

    if bad:
        raise DataValidationError(f"{context} has missing values in required columns: {bad}")


def validate_numeric_features(df: pd.DataFrame, *, context: str = "dataframe") -> None:
    """Validate that numeric features can be interpreted as finite numeric values."""
    errors: dict[str, str] = {}

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")

        if numeric.isna().any():
            errors[col] = "contains non-numeric or missing values"
            continue

        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            errors[col] = "contains non-finite values"

    if errors:
        raise DataValidationError(f"{context} has invalid numeric features: {errors}")


def validate_binary_target(df: pd.DataFrame, *, context: str = "dataframe") -> None:
    """Validate that the target column exists and contains binary 0/1 labels."""
    validate_required_columns(df, [TARGET_COL], context=context)

    target = df[TARGET_COL]

    if target.isna().any():
        raise DataValidationError(
            f"{context} target column {TARGET_COL!r} contains missing values."
        )

    values = set(pd.to_numeric(target, errors="coerce").dropna().astype(int).unique())
    original_non_null = target.dropna()

    if len(original_non_null) != len(target):
        raise DataValidationError(
            f"{context} target column {TARGET_COL!r} contains invalid values."
        )

    if not values.issubset({0, 1}) or not values:
        raise DataValidationError(
            f"{context} target column {TARGET_COL!r} must contain binary 0/1 labels. "
            f"Observed values: {sorted(values)}"
        )


def validate_training_dataframe(df: pd.DataFrame, *, context: str = "training data") -> None:
    """Validate a dataframe used for training/evaluation."""
    validate_required_columns(df, REQUIRED_TRAINING_COLUMNS, context=context)
    validate_no_missing_required_values(df, REQUIRED_TRAINING_COLUMNS, context=context)
    validate_numeric_features(df, context=context)
    validate_binary_target(df, context=context)


def validate_scoring_dataframe(df: pd.DataFrame, *, context: str = "scoring data") -> None:
    """Validate a dataframe used for scoring new transactions.

    Scoring data does not need the target column. If the target is present, it is
    preserved in the output but excluded from model features.
    """
    validate_required_columns(df, REQUIRED_FEATURE_COLUMNS, context=context)
    validate_no_missing_required_values(df, REQUIRED_FEATURE_COLUMNS, context=context)
    validate_numeric_features(df, context=context)


def prepare_features_for_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature columns for model prediction while allowing label/prediction columns."""
    validate_scoring_dataframe(df, context="scoring data")
    return df.drop(columns=[TARGET_COL, *PREDICTION_COLUMNS], errors="ignore")


def validate_threshold(threshold: float) -> float:
    """Validate and normalize a decision threshold."""
    threshold = float(threshold)

    if not 0.0 <= threshold <= 1.0:
        raise DataValidationError(f"Threshold must be between 0 and 1. Received: {threshold}")

    return threshold
