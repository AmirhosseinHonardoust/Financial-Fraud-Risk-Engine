from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    TARGET_COL,
    THRESHOLD_GRID,
)
from .threshold_policy import save_threshold_policy_artifacts
from .validation import validate_training_dataframe


def as_repo_relative_path(path: Path) -> str:
    """Return a portable repo-relative path for saved artifacts.

    This prevents JSON artifacts from containing local absolute paths such as:
    C:\\Users\\Amir\\Desktop\\...
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_model_and_data():
    model_path = MODELS_DIR / "fraud_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")

    model = joblib.load(model_path)

    test_path = PROCESSED_DATA_DIR / "transactions_test.csv"
    train_path = PROCESSED_DATA_DIR / "transactions_train.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}. Run data_prep.py first.")

    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}. Run data_prep.py first.")

    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    validate_training_dataframe(train_df, context="processed training data")
    validate_training_dataframe(test_df, context="processed test data")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    return model, X_train, y_train, X_test, y_test


def compute_probability_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Compute probability-based fraud metrics.

    Average precision is the area under the precision-recall curve and is usually
    more informative than ROC-AUC for imbalanced fraud problems.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }


def compute_baseline_metrics(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict[str, float | str]]:
    """Evaluate simple baselines so model performance has context."""
    baselines: dict[str, dict[str, float | str]] = {}

    for name, strategy in {
        "majority_class": "most_frequent",
        "empirical_prior": "prior",
        "stratified_random": "stratified",
    }.items():
        clf = DummyClassifier(strategy=strategy, random_state=42)
        clf.fit(X_train, y_train)

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)[:, 1]
        else:
            proba = clf.predict(X_test)

        metrics: dict[str, float | str] = dict(
            compute_probability_metrics(y_test.to_numpy(), proba)
        )
        metrics["strategy"] = strategy
        baselines[name] = metrics

    return baselines


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    """Evaluate thresholds and compute classification, rate, and cost metrics."""
    results = []
    n = len(y_true)

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        flagged_rate = (tp + fp) / n if n > 0 else 0.0
        cost = COST_FALSE_NEGATIVE * fn + COST_FALSE_POSITIVE * fp
        normalized_cost = cost / n if n > 0 else 0.0

        results.append(
            {
                "threshold": float(thr),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "false_positive_rate": float(fpr),
                "specificity": float(specificity),
                "flagged_rate": float(flagged_rate),
                "cost": float(cost),
                "normalized_cost": float(normalized_cost),
            }
        )

    return results


def pick_best_threshold(results: list[dict]) -> dict:
    """Choose the threshold that minimizes cost, then prefers higher recall."""
    return min(results, key=lambda r: (r["cost"], -r["recall"], -r["precision"]))


def save_threshold_search_csv(results: list[dict]) -> None:
    path = METRICS_DIR / "threshold_search.csv"
    if not results:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved threshold search CSV to: {as_repo_relative_path(path)}")


def plot_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title("ROC Curve - Fraud Detection")
    roc_path = FIGURES_DIR / "roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    PrecisionRecallDisplay(
        precision=precision,
        recall=recall,
        average_precision=avg_precision,
    ).plot()
    plt.title("Precision-Recall Curve - Fraud Detection")
    pr_path = FIGURES_DIR / "pr_curve.png"
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curve to: {as_repo_relative_path(roc_path)}")
    print(f"Saved PR curve to:  {as_repo_relative_path(pr_path)}")


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Calibration Curve - Fraud Probability")
    ax.legend()

    path = FIGURES_DIR / "calibration_curve.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print(f"Saved calibration curve to: {as_repo_relative_path(path)}")


def plot_threshold_cost_curve(results: list[dict]) -> None:
    df = pd.DataFrame(results)

    fig, ax = plt.subplots()
    ax.plot(df["threshold"], df["cost"], marker="o")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Cost")
    ax.set_title("Threshold Cost Curve")

    path = FIGURES_DIR / "threshold_cost_curve.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print(f"Saved threshold cost curve to: {as_repo_relative_path(path)}")


def plot_threshold_tradeoffs(results: list[dict]) -> None:
    df = pd.DataFrame(results)

    fig, ax = plt.subplots()
    ax.plot(df["threshold"], df["precision"], marker="o", label="Precision")
    ax.plot(df["threshold"], df["recall"], marker="o", label="Recall")
    ax.plot(df["threshold"], df["false_positive_rate"], marker="o", label="False positive rate")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Threshold Tradeoffs")
    ax.legend()

    path = FIGURES_DIR / "threshold_tradeoffs.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print(f"Saved threshold tradeoff curve to: {as_repo_relative_path(path)}")


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

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    print(f"Saved confusion matrix to: {as_repo_relative_path(cm_path)}")


def main() -> None:
    model, X_train, y_train, X_test, y_test = load_model_and_data()
    y_proba = model.predict_proba(X_test)[:, 1]

    probability_metrics = compute_probability_metrics(y_test.values, y_proba)
    baseline_metrics = compute_baseline_metrics(X_train, y_train, X_test, y_test)

    results = compute_threshold_metrics(y_test.values, y_proba, THRESHOLD_GRID)
    best = pick_best_threshold(results)

    threshold_search_path = METRICS_DIR / "threshold_search.json"
    with threshold_search_path.open("w") as f:
        json.dump(results, f, indent=2)

    save_threshold_search_csv(results)

    policy_paths = save_threshold_policy_artifacts(
        results,
        best,
        metrics_dir=METRICS_DIR,
        cost_false_positive=COST_FALSE_POSITIVE,
        cost_false_negative=COST_FALSE_NEGATIVE,
    )

    threshold_path = MODELS_DIR / "threshold.json"
    with threshold_path.open("w") as f:
        json.dump(best, f, indent=2)

    evaluation_summary = {
        **probability_metrics,
        "positive_rate_test": float(np.mean(y_test.values)),
        "n_test_samples": int(len(y_test)),
        "best_threshold": best,
        "baseline_metrics": baseline_metrics,
        "cost_false_negative": COST_FALSE_NEGATIVE,
        "cost_false_positive": COST_FALSE_POSITIVE,
        "artifacts": {
            "threshold_search_json": as_repo_relative_path(threshold_search_path),
            "threshold_search_csv": as_repo_relative_path(METRICS_DIR / "threshold_search.csv"),
            "threshold_json": as_repo_relative_path(threshold_path),
            "roc_curve": as_repo_relative_path(FIGURES_DIR / "roc_curve.png"),
            "pr_curve": as_repo_relative_path(FIGURES_DIR / "pr_curve.png"),
            "calibration_curve": as_repo_relative_path(FIGURES_DIR / "calibration_curve.png"),
            "threshold_cost_curve": as_repo_relative_path(FIGURES_DIR / "threshold_cost_curve.png"),
            "threshold_tradeoffs": as_repo_relative_path(FIGURES_DIR / "threshold_tradeoffs.png"),
            "confusion_matrix": as_repo_relative_path(FIGURES_DIR / "confusion_matrix.png"),
        },
        "threshold_policy_artifacts": {
            key: as_repo_relative_path(path) for key, path in policy_paths.items()
        },
    }

    summary_path = METRICS_DIR / "evaluation_summary.json"
    with summary_path.open("w") as f:
        json.dump(evaluation_summary, f, indent=2)

    print("\n=== Probability metrics ===")
    print(json.dumps(probability_metrics, indent=2))
    print("\n=== Best threshold based on cost ===")
    print(json.dumps(best, indent=2))
    print(f"\nSaved evaluation summary to: {as_repo_relative_path(summary_path)}")
    print(f"Saved selected threshold to:  {as_repo_relative_path(threshold_path)}")
    print(f"Saved threshold search to:    {as_repo_relative_path(threshold_search_path)}")

    plot_roc_pr_curves(y_test.values, y_proba)
    plot_calibration(y_test.values, y_proba)
    plot_threshold_cost_curve(results)
    plot_threshold_tradeoffs(results)

    y_pred_best = (y_proba >= best["threshold"]).astype(int)
    plot_confusion_matrix(y_test.values, y_pred_best, best["threshold"])


if __name__ == "__main__":
    main()
