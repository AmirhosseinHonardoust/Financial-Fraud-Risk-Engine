from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import MODELS_DIR
from .reason_codes import add_reason_codes
from .validation import (
    DataValidationError,
    prepare_features_for_scoring,
    validate_threshold,
)


def load_model_and_threshold():
    """Load the persisted fraud model and selected decision threshold."""
    model_path = MODELS_DIR / "fraud_pipeline.joblib"
    threshold_path = MODELS_DIR / "threshold.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python -m src.train_model` first."
        )

    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Threshold file not found at {threshold_path}. Run `python -m src.evaluate` first."
        )

    model = joblib.load(model_path)

    with threshold_path.open() as f:
        threshold_info = json.load(f)

    threshold = validate_threshold(float(threshold_info["threshold"]))
    return model, threshold


def score_dataframe(
    df: pd.DataFrame,
    *,
    threshold: float,
    model,
) -> pd.DataFrame:
    """Score a dataframe and return a copy with fraud probabilities and flags."""
    threshold = validate_threshold(threshold)
    features = prepare_features_for_scoring(df)

    probs = model.predict_proba(features)[:, 1]
    preds = (probs >= threshold).astype(int)

    df_scored = df.copy()
    df_scored["fraud_probability"] = probs
    df_scored["fraud_flag"] = preds
    df_scored = df_scored.sort_values("fraud_probability", ascending=False).reset_index(drop=True)
    df_scored = add_reason_codes(df_scored, threshold=threshold)
    return df_scored


def score_file(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
) -> Path:
    """Score a CSV of transactions and save a download-ready scored CSV."""
    model, threshold = load_model_and_threshold()

    input_csv = Path(input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    df_scored = score_dataframe(df, threshold=threshold, model=model)

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_scored.csv")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(output_csv, index=False)

    print(f"Saved scored transactions to: {output_csv}")
    print(f"Rows scored: {len(df_scored):,}")
    print(f"Flagged transactions: {int(df_scored['fraud_flag'].sum()):,}")
    return output_csv


def main():
    # Example usage:
    # python -m src.score_new_transactions data/raw/synthetic_fraud_dataset.csv
    import argparse

    parser = argparse.ArgumentParser(description="Score new transactions for fraud risk.")
    parser.add_argument("input_csv", type=str, help="Path to CSV with transaction data.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Where to save scored CSV. Defaults to '<input>_scored.csv'.",
    )
    args = parser.parse_args()

    try:
        score_file(args.input_csv, args.output_csv)
    except DataValidationError as exc:
        raise SystemExit(f"Input validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
