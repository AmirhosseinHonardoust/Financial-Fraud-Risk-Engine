from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import METRICS_DIR, PROCESSED_DATA_DIR, TARGET_COL  # noqa: E402
from src.dashboard_utils import (  # noqa: E402
    add_risk_band,
    build_model_metadata,
    format_float,
    format_percent,
    summarize_scored_transactions,
)
from src.reason_codes import (  # noqa: E402
    positive_class_shap_values,
    shap_reason_codes,
    split_reason_codes,
)
from src.score_new_transactions import score_dataframe  # noqa: E402
from src.validation import (  # noqa: E402
    REQUIRED_FEATURE_COLUMNS,
    DataValidationError,
    validate_scoring_dataframe,
    validate_threshold,
)

# ------------- Helpers for loading model, metadata, and data ------------------


@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "models" / "fraud_pipeline.joblib"
    if not model_path.exists():
        st.error(f"Model not found at {model_path}. " "Run `python -m src.train_model` first.")
        st.stop()

    return joblib.load(model_path)


@st.cache_data
def load_threshold() -> dict[str, Any]:
    threshold_path = PROJECT_ROOT / "models" / "threshold.json"
    if not threshold_path.exists():
        return {"threshold": 0.5, "note": "threshold.json not found; using 0.5"}

    with threshold_path.open() as f:
        return json.load(f)


@st.cache_data
def load_json_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}

    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    """Use processed test data as a demo if user does not upload data."""
    test_path = PROCESSED_DATA_DIR / "transactions_test.csv"
    if not test_path.exists():
        st.error(
            f"Processed test data not found at {test_path}. " "Run `python -m src.data_prep` first."
        )
        st.stop()

    return pd.read_csv(test_path)


def score_transactions(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Validate and score transactions with fraud probabilities and flags."""
    validate_scoring_dataframe(df, context="uploaded/sample transaction data")
    threshold = validate_threshold(threshold)
    return score_dataframe(df, threshold=threshold, model=model)


# ------------- SHAP explanation helpers ------------------


@st.cache_resource
def get_shap_explainer(_model):
    """Create a TreeExplainer for the underlying RandomForest."""
    preprocessor = _model.named_steps["preprocess"]
    clf = _model.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    feature_names = preprocessor.get_feature_names_out()
    return explainer, preprocessor, feature_names


def plot_single_shap_bar(
    shap_values: np.ndarray,
    feature_names: np.ndarray,
    max_features: int = 10,
):
    """Plot top-|SHAP| features for a single prediction."""
    shap_values = np.asarray(shap_values).reshape(-1)
    feature_names = np.asarray(feature_names)

    n_features = min(len(shap_values), len(feature_names))
    shap_values = shap_values[:n_features]
    feature_names = feature_names[:n_features]

    abs_vals = np.abs(shap_values)
    idx_sorted = np.argsort(abs_vals)[::-1][:max_features]

    selected_shap = shap_values[idx_sorted]
    selected_names = feature_names[idx_sorted]

    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(selected_names))

    ax.barh(y_pos, selected_shap)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(selected_names)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value")
    ax.set_title("Top feature contributions for this transaction")

    plt.tight_layout()
    return fig


def explain_single_transaction(model, df_scored: pd.DataFrame, row_idx: int):
    """Compute SHAP values for a single row and return a matplotlib figure."""
    explainer, preprocessor, feature_names = get_shap_explainer(model)

    cols_to_drop = [TARGET_COL, "fraud_probability", "fraud_flag", "risk_band", "reason_codes"]
    features_df = df_scored.drop(columns=[c for c in cols_to_drop if c in df_scored.columns])

    x_row = features_df.iloc[[row_idx]]
    x_transformed = preprocessor.transform(x_row)

    try:
        import scipy.sparse as sp

        x_for_shap = x_transformed.toarray() if sp.issparse(x_transformed) else x_transformed
    except ImportError:
        x_for_shap = x_transformed

    shap_vals = explainer.shap_values(x_for_shap)

    shap_for_fraud_class = positive_class_shap_values(shap_vals)[0]

    fig = plot_single_shap_bar(shap_for_fraud_class, feature_names)
    reasons = shap_reason_codes(shap_for_fraud_class, feature_names, max_reasons=5)
    return fig, reasons


# ------------- Streamlit UI ------------------


def main():
    st.set_page_config(
        page_title="Financial Fraud Risk Engine",
        layout="wide",
    )

    st.title("Financial Fraud Risk Engine")
    st.markdown("""
This app wraps a trained fraud detection model into an **interactive risk dashboard**.

- Upload transaction data or use the built-in test set  
- Tune the **decision threshold**  
- Explore **risk distribution**, **risk bands**, and **flagged transactions**  
- Download scored transactions for analyst review  
- Inspect **feature-level explanations** for individual transactions
""")

    st.warning(
        "This dashboard uses a synthetic demo dataset with overlapping classes and label noise. "
        "It demonstrates fraud-risk workflow design, not real-world fraud benchmark performance."
    )

    model = load_model()
    threshold_info = load_threshold()
    train_metrics = load_json_file(METRICS_DIR / "metrics.json")
    evaluation_summary = load_json_file(METRICS_DIR / "evaluation_summary.json")
    model_metadata = build_model_metadata(
        model,
        threshold_info,
        train_metrics,
        evaluation_summary,
    )
    default_threshold = float(model_metadata.get("saved_threshold") or 0.5)

    st.sidebar.header("Controls")

    st.sidebar.markdown("### Decision threshold")
    thr = st.sidebar.slider(
        "Fraud if probability ≥ threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.01,
    )
    st.sidebar.caption(f"Saved threshold: {default_threshold:.2f}")

    st.sidebar.markdown("### Data source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV with transactions",
        type=["csv"],
    )

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.sidebar.success("Using uploaded data.")
    else:
        df_raw = load_sample_data()
        st.sidebar.info("No file uploaded. Using sample test data from the project.")

    with st.sidebar.expander("Required upload columns"):
        st.write(REQUIRED_FEATURE_COLUMNS)
        st.caption("The target column is optional for scoring.")

    with st.sidebar.expander("Model metadata"):
        st.write(
            {
                "model_type": model_metadata.get("model_type"),
                "saved_threshold": model_metadata.get("saved_threshold"),
                "roc_auc": model_metadata.get("roc_auc"),
                "average_precision": model_metadata.get("average_precision"),
                "brier_score": model_metadata.get("brier_score"),
                "n_train_samples": model_metadata.get("n_train_samples"),
                "n_test_samples": model_metadata.get("n_test_samples"),
            }
        )
        st.caption(model_metadata["data_note"])

    try:
        df_scored = score_transactions(model, df_raw, thr)
        df_scored = add_risk_band(df_scored)
    except DataValidationError as exc:
        st.error(f"Input validation failed: {exc}")
        st.info(
            "Please upload a CSV containing the required transaction feature columns. "
            "The target column is optional for scoring."
        )
        st.code("\n".join(REQUIRED_FEATURE_COLUMNS), language="text")
        st.stop()

    summary = summarize_scored_transactions(df_scored, thr)

    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total transactions", f"{summary['total_transactions']:,}")
    col2.metric("Flagged transactions", f"{summary['flagged_count']:,}")
    col3.metric("Flagged rate", format_percent(summary["flagged_rate"]))
    col4.metric("P95 fraud probability", format_float(summary["p95_probability"]))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Average probability", format_float(summary["average_probability"]))
    col6.metric("Max probability", format_float(summary["max_probability"]))
    col7.metric("True fraud rate", format_percent(summary["true_fraud_rate"]))
    col8.metric("Selected threshold", f"{thr:.2f}")

    st.markdown("### Model quality snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", format_float(model_metadata.get("roc_auc")))
    m2.metric("PR-AUC", format_float(model_metadata.get("average_precision")))
    m3.metric("Brier score", format_float(model_metadata.get("brier_score")))
    m4.metric("Saved threshold", format_float(model_metadata.get("saved_threshold"), digits=2))

    st.subheader("Risk Distribution & Analyst Review Queue")

    c1, c2 = st.columns([2, 3])

    with c1:
        st.markdown("**Fraud probability distribution**")

        fig, ax = plt.subplots()
        ax.hist(df_scored["fraud_probability"], bins=30)
        ax.axvline(thr, color="red", linestyle="--", label=f"threshold = {thr:.2f}")
        ax.set_xlabel("Fraud probability")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

        st.markdown("**Risk band counts**")
        risk_order = ["Low", "Medium", "High", "Critical"]
        risk_counts = df_scored["risk_band"].value_counts().reindex(risk_order, fill_value=0)
        st.bar_chart(risk_counts)

    with c2:
        st.markdown("**Top high-risk transactions**")
        top_n = st.slider("Show top N by fraud probability", 5, 100, 20)
        display_cols = [
            col
            for col in [
                "transaction_id",
                "user_id",
                "fraud_probability",
                "fraud_flag",
                "risk_band",
                "reason_codes",
                "amount",
                "hour",
                "device_risk_score",
                "ip_risk_score",
                "transaction_type",
                "merchant_category",
                "country",
                TARGET_COL,
            ]
            if col in df_scored.columns
        ]
        top_risky = df_scored.head(top_n)
        st.dataframe(top_risky[display_cols], width="stretch")

    scored_csv = df_scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download scored transactions CSV",
        data=scored_csv,
        file_name=f"fraud_scored_transactions_threshold_{thr:.2f}.csv",
        mime="text/csv",
    )

    st.subheader("Explain a Single Transaction")

    if len(df_scored) == 0:
        st.warning("No data available for explanation.")
        return

    id_col = None
    for candidate in ["transaction_id", "id", "txn_id"]:
        if candidate in df_scored.columns:
            id_col = candidate
            break

    if id_col is not None:
        options = df_scored[id_col].tolist()
        selected_id = st.selectbox(f"Select {id_col} to explain", options, index=0)
        row_idx = int(df_scored.index[df_scored[id_col] == selected_id][0])
    else:
        row_idx = int(
            st.number_input(
                "Row index to explain (0-based)",
                min_value=0,
                max_value=len(df_scored) - 1,
                value=0,
                step=1,
            )
        )

    row = df_scored.iloc[row_idx]

    st.markdown("**Selected transaction**")
    st.dataframe(row.to_frame().T, width="stretch")

    st.markdown("**Model prediction**")
    st.write(
        f"Fraud probability: **{row['fraud_probability']:.4f}**, "
        f"risk band: **{row['risk_band']}**, "
        f"fraud flag at threshold {thr:.2f}: **{int(row['fraud_flag'])}**"
    )

    if "reason_codes" in row.index:
        st.markdown("**Analyst reason codes**")
        for reason in split_reason_codes(row.get("reason_codes")):
            st.write(f"- {reason}")

    st.markdown("**Feature contribution (SHAP)**")

    with st.spinner("Computing feature contributions..."):
        fig_shap, shap_reasons = explain_single_transaction(model, df_scored, row_idx)
        st.pyplot(fig_shap)

    st.markdown("**SHAP-based reason codes**")
    for reason in shap_reasons:
        st.write(f"- {reason}")


if __name__ == "__main__":
    main()
