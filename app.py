import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, TARGET_COL  # type: ignore
from src.score_new_transactions import score_dataframe  # type: ignore
from src.validation import DataValidationError, validate_scoring_dataframe, validate_threshold  # type: ignore


# ------------- Helpers for loading model & data ------------------


@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "models" / "fraud_pipeline.joblib"
    if not model_path.exists():
        st.error(
            f"Model not found at {model_path}. "
            "Run `python -m src.train_model` first."
        )
        st.stop()

    model = joblib.load(model_path)
    return model


@st.cache_data
def load_threshold():
    threshold_path = PROJECT_ROOT / "models" / "threshold.json"
    if not threshold_path.exists():
        # Fallback: default threshold 0.5
        return {"threshold": 0.5, "note": "threshold.json not found; using 0.5"}

    with threshold_path.open() as f:
        thr_info = json.load(f)
    return thr_info


@st.cache_data
def load_sample_data():
    """Use processed test data as a demo if user doesn't upload."""
    test_path = PROCESSED_DATA_DIR / "transactions_test.csv"
    if not test_path.exists():
        st.error(
            f"Processed test data not found at {test_path}. "
            "Run `python -m src.data_prep` first."
        )
        st.stop()

    df = pd.read_csv(test_path)
    return df


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
    """
    Plot top-|SHAP| features for a single prediction.

    We defensively:
    - flatten shap_values to 1D
    - align feature_names length to shap_values length
    """
    # Ensure numpy arrays
    shap_values = np.asarray(shap_values).reshape(-1)  # flatten to 1D
    feature_names = np.asarray(feature_names)

    # Align lengths in case SHAP/feature_names mismatch a bit
    n_features = min(len(shap_values), len(feature_names))
    shap_values = shap_values[:n_features]
    feature_names = feature_names[:n_features]

    # Sort by absolute impact
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
    ax.set_xlabel("SHAP value (impact on fraud probability)")
    ax.set_title("Top feature contributions for this transaction")

    plt.tight_layout()
    return fig



def explain_single_transaction(model, df_scored: pd.DataFrame, row_idx: int):
    """Compute SHAP values for a single row and return a matplotlib figure."""
    explainer, preprocessor, feature_names = get_shap_explainer(model)

    # Drop target and prediction columns when computing SHAP input
    cols_to_drop = [TARGET_COL, "fraud_probability", "fraud_flag"]
    features_df = df_scored.drop(columns=[c for c in cols_to_drop if c in df_scored.columns])

    x_row = features_df.iloc[[row_idx]]  # keep as DataFrame
    x_transformed = preprocessor.transform(x_row)

    # For tree models, SHAP expects a dense array
    try:
        import scipy.sparse as sp

        if sp.issparse(x_transformed):
            x_for_shap = x_transformed.toarray()
        else:
            x_for_shap = x_transformed
    except ImportError:
        x_for_shap = x_transformed

    shap_vals = explainer.shap_values(x_for_shap)

    # Binary classifier: can be list [class0, class1] *or* a single array
    if isinstance(shap_vals, list):
        # class 1 (fraud), first sample
        shap_for_fraud_class = shap_vals[1][0]
    else:
        # single array -> first sample
        shap_for_fraud_class = shap_vals[0]

    fig = plot_single_shap_bar(shap_for_fraud_class, feature_names)
    return fig




# ------------- Streamlit UI ------------------


def main():
    st.set_page_config(
        page_title="Financial Fraud Risk Engine",
        layout="wide",
    )

    st.title("Financial Fraud Risk Engine")
    st.markdown(
        """
This app wraps a trained fraud detection model into an **interactive risk dashboard**.

- Upload transaction data or use the built-in test set  
- Tune the **decision threshold**  
- Explore **risk distribution** and **flagged transactions**  
- Inspect **feature-level explanations** for individual transactions
"""
    )

    model = load_model()
    threshold_info = load_threshold()
    default_threshold = float(threshold_info.get("threshold", 0.5))

    # Sidebar controls
    st.sidebar.header("Controls")

    st.sidebar.markdown("### Threshold")
    thr = st.sidebar.slider(
        "Decision threshold (fraud if probability ≥ threshold)",
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.01,
    )

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

    # Score data
    try:
        df_scored = score_transactions(model, df_raw, thr)
    except DataValidationError as exc:
        st.error(f"Input validation failed: {exc}")
        st.info(
            "Please upload a CSV containing the required transaction feature columns. "
            "The target column is optional for scoring."
        )
        st.stop()

    # --- Layout: metrics row ---
    st.subheader("Overview")

    n_rows = len(df_scored)
    fraud_rate = df_scored.get(TARGET_COL, pd.Series([np.nan] * n_rows)).mean()
    predicted_rate = df_scored["fraud_flag"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total transactions", f"{n_rows:,}")
    if TARGET_COL in df_scored.columns:
        col2.metric("True fraud rate (label)", f"{fraud_rate * 100:.2f}%")
    else:
        col2.metric("True fraud rate (label)", "Unknown (no labels)")
    col3.metric("Flagged by model", f"{predicted_rate * 100:.2f}%")

    # --- Risk distribution & flags ---
    st.subheader("Risk Distribution & Flagged Transactions")

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

    with c2:
        st.markdown("**Top high-risk transactions**")
        top_n = st.slider("Show top N by fraud probability", 5, 100, 20)
        top_risky = df_scored.sort_values("fraud_probability", ascending=False).head(top_n)
        st.dataframe(top_risky)

    scored_csv = df_scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download scored transactions CSV",
        data=scored_csv,
        file_name="fraud_scored_transactions.csv",
        mime="text/csv",
    )

    # --- Per-transaction explanation ---
    st.subheader("Explain a single transaction")

    if len(df_scored) == 0:
        st.warning("No data available for explanation.")
        return

    # Let user pick by index or transaction_id (if available)
    id_col = None
    for candidate in ["transaction_id", "id", "txn_id"]:
        if candidate in df_scored.columns:
            id_col = candidate
            break

    if id_col is not None:
        options = df_scored[id_col].tolist()
        selected_id = st.selectbox(f"Select {id_col} to explain", options, index=0)
        row_idx = df_scored.index[df_scored[id_col] == selected_id][0]
    else:
        row_idx = st.number_input(
            "Row index to explain (0-based)",
            min_value=0,
            max_value=len(df_scored) - 1,
            value=0,
            step=1,
        )
        selected_id = row_idx

    row = df_scored.iloc[row_idx]

    st.markdown("**Selected transaction (raw)**")
    st.write(row.to_frame().T)

    st.markdown("**Model prediction**")
    st.write(
        f"Fraud probability: **{row['fraud_probability']:.4f}**, "
        f"fraud_flag (with threshold {thr:.2f}): **{int(row['fraud_flag'])}**"
    )

    st.markdown("**Feature contribution (SHAP)**")

    with st.spinner("Computing feature contributions..."):
        fig_shap = explain_single_transaction(model, df_scored, row_idx)
        st.pyplot(fig_shap)


if __name__ == "__main__":
    main()
