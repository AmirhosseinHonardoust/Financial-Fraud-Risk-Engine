import joblib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import shap

from .config import FIGURES_DIR, MODELS_DIR, PROCESSED_DATA_DIR, TARGET_COL
from .reason_codes import positive_class_shap_values


def load_model_and_test():
    model_path = MODELS_DIR / "fraud_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")

    model = joblib.load(model_path)

    test_path = PROCESSED_DATA_DIR / "transactions_test.csv"
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    return model, X_test, y_test


def compute_and_plot_global_shap(
    model,
    X: pd.DataFrame,
    max_samples: int = 2000,
) -> None:
    """
    Compute SHAP values on a subset of the data and save a summary plot.

    For tree models (RandomForest), TreeExplainer is appropriate.
    """
    # Subsample for speed if needed
    X_sample = X.sample(max_samples, random_state=0) if len(X) > max_samples else X

    # Extract underlying estimator after preprocessing
    preprocessor = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    X_transformed = preprocessor.transform(X_sample)

    # Convert sparse matrices to dense for SHAP summary plot
    X_for_shap = X_transformed.toarray() if sp.issparse(X_transformed) else X_transformed

    explainer = shap.TreeExplainer(clf)
    shap_values = positive_class_shap_values(explainer.shap_values(X_for_shap))

    # SHAP expects names for features; use transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    shap.summary_plot(
        shap_values,  # class 1 (fraud)
        X_for_shap,
        feature_names=feature_names,
        show=False,
    )

    shap_path = FIGURES_DIR / "shap_summary.png"
    plt.title("SHAP Summary Plot - Fraud Model (Class 1: Fraud)")
    plt.savefig(shap_path, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP summary plot to: {shap_path}")


def main():
    model, X_test, y_test = load_model_and_test()
    print(f"Loaded model and test data with shape {X_test.shape}")
    compute_and_plot_global_shap(model, X_test)


if __name__ == "__main__":
    main()
