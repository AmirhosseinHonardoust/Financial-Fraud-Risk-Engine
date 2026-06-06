import json
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from .config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    TARGET_COL,
    THRESHOLD_GRID,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
)
from .validation import validate_training_dataframe


def load_model_and_data():
    model_path = MODELS_DIR / "fraud_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")

    model = joblib.load(model_path)

    test_path = PROCESSED_DATA_DIR / "transactions_test.csv"
    test_df = pd.read_csv(test_path)
    validate_training_dataframe(test_df, context="processed test data")
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    return model, X_test, y_test


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: List[float],
) -> List[Dict]:
    """Evaluate different thresholds and compute metrics + simple cost."""
    results = []

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        cost = COST_FALSE_NEGATIVE * fn + COST_FALSE_POSITIVE * fp

        results.append(
            {
                "threshold": thr,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cost": cost,
            }
        )

    return results


def pick_best_threshold(results: List[Dict]) -> Dict:
    """Choose the threshold that minimizes cost."""
    best = min(results, key=lambda r: r["cost"])
    return best


def plot_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title("ROC Curve - Fraud Detection")
    roc_path = FIGURES_DIR / "roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title("Precision-Recall Curve - Fraud Detection")
    pr_path = FIGURES_DIR / "pr_curve.png"
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve to: {roc_path}")
    print(f"Saved PR curve to:  {pr_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Not fraud", "Fraud"],
        yticklabels=["Not fraud", "Fraud"],
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix (threshold = {threshold:.2f})",
    )

    # Write counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")


def main() -> None:
    model, X_test, y_test = load_model_and_data()
    y_proba = model.predict_proba(X_test)[:, 1]

    # Threshold search
    results = compute_threshold_metrics(y_test.values, y_proba, THRESHOLD_GRID)
    best = pick_best_threshold(results)

    # Save threshold search results
    threshold_search_path = METRICS_DIR / "threshold_search.json"
    with threshold_search_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Save best threshold
    threshold_path = MODELS_DIR / "threshold.json"
    with threshold_path.open("w") as f:
        json.dump(best, f, indent=2)

    print("\n=== Best threshold based on cost ===")
    print(json.dumps(best, indent=2))

    # Plots
    plot_roc_pr_curves(y_test.values, y_proba)
    y_pred_best = (y_proba >= best["threshold"]).astype(int)
    plot_confusion_matrix(y_test.values, y_pred_best, best["threshold"])


if __name__ == "__main__":
    main()
