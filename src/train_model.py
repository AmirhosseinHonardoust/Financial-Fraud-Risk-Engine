import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report

from .config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    TARGET_COL,
)
from .features import build_pipeline
from .validation import validate_training_dataframe


def load_processed_data():
    train_path = PROCESSED_DATA_DIR / "transactions_train.csv"
    test_path = PROCESSED_DATA_DIR / "transactions_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed train/test files not found in {PROCESSED_DATA_DIR}. "
            f"Run data_prep.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    validate_training_dataframe(train_df, context="processed training data")
    validate_training_dataframe(test_df, context="processed test data")
    return train_df, test_df


def train_and_evaluate() -> dict:
    train_df, test_df = load_processed_data()

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Probabilities for ROC-AUC and later threshold tuning
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)

    # Default 0.5 threshold just for a baseline classification report
    y_pred_default = (y_proba >= 0.5).astype(int)

    cls_report = classification_report(
        y_test,
        y_pred_default,
        output_dict=True,
        digits=3,
    )

    metrics = {
        "roc_auc": float(roc_auc),
        "classification_report_default_threshold": cls_report,
        "n_test_samples": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
    }

    # Persist model
    model_path = MODELS_DIR / "fraud_pipeline.joblib"
    joblib.dump(pipeline, model_path)

    # Persist metrics
    metrics_path = METRICS_DIR / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to:   {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"ROC-AUC (test): {roc_auc:.4f}")

    return metrics


def main() -> None:
    metrics = train_and_evaluate()
    print("\n=== Metrics summary ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
