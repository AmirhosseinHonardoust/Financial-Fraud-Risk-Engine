from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

SCORE_COLUMNS = {"fraud_probability", "fraud_flag", "risk_band", "reason_codes"}


FRIENDLY_FEATURE_NAMES = {
    "amount": "transaction amount",
    "hour": "transaction hour",
    "device_risk_score": "device risk score",
    "ip_risk_score": "IP risk score",
    "transaction_type": "transaction type",
    "merchant_category": "merchant category",
    "country": "country signal",
}


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    return str(value)


def high_amount_cutoff(df: pd.DataFrame, quantile: float = 0.95) -> float | None:
    """Return a robust high-amount cutoff from the current scoring batch."""
    if "amount" not in df.columns or len(df) == 0:
        return None

    values = pd.to_numeric(df["amount"], errors="coerce").dropna()
    if len(values) == 0:
        return None

    return float(values.quantile(quantile))


def reason_codes_for_row(
    row: pd.Series | dict,
    *,
    threshold: float = 0.5,
    amount_cutoff: float | None = None,
    max_reasons: int = 5,
) -> list[str]:
    """Create analyst-friendly rule-based reason codes for one transaction.

    These reason codes are intentionally simple and deterministic. They summarize
    obvious risk drivers in the input features and score output. They are not a
    causal explanation and should be reviewed together with SHAP/model evidence.

    Accepts either a pandas Series or a plain mapping (e.g. a row from
    ``DataFrame.to_dict("records")``); both support ``.get`` so no per-row Series
    construction is needed.
    """
    reasons: list[str] = []

    fraud_probability = _safe_float(row.get("fraud_probability"))
    if fraud_probability is not None:
        if fraud_probability >= 0.75:
            reasons.append("Critical model risk score")
        elif fraud_probability >= threshold:
            reasons.append("Model score is above the review threshold")

    device_risk = _safe_float(row.get("device_risk_score"))
    if device_risk is not None:
        if device_risk >= 0.80:
            reasons.append("High device risk score")
        elif device_risk >= 0.60:
            reasons.append("Elevated device risk score")

    ip_risk = _safe_float(row.get("ip_risk_score"))
    if ip_risk is not None:
        if ip_risk >= 0.80:
            reasons.append("High IP risk score")
        elif ip_risk >= 0.60:
            reasons.append("Elevated IP risk score")

    amount = _safe_float(row.get("amount"))
    if amount is not None and amount_cutoff is not None and amount >= amount_cutoff:
        reasons.append("Transaction amount is high for this batch")

    hour = _safe_float(row.get("hour"))
    if hour is not None and (hour <= 5 or hour >= 23):
        reasons.append("Transaction occurred during unusual hours")

    transaction_type = _safe_str(row.get("transaction_type")).lower()
    if transaction_type in {"transfer", "withdrawal", "wire"}:
        reasons.append(f"Transaction type '{transaction_type}' is higher risk in the demo data")

    merchant_category = _safe_str(row.get("merchant_category")).lower()
    if merchant_category in {"crypto", "electronics", "luxury"}:
        reasons.append(f"Merchant category '{merchant_category}' is higher risk in the demo data")

    country = _safe_str(row.get("country"))
    if country.upper() in {"RU", "CN"}:
        reasons.append("Country signal is associated with higher synthetic-demo risk")

    if not reasons:
        reasons.append("No strong rule-based risk drivers identified")

    return reasons[:max_reasons]


def add_reason_codes(
    df_scored: pd.DataFrame,
    *,
    threshold: float = 0.5,
    max_reasons: int = 5,
) -> pd.DataFrame:
    """Add a semicolon-separated reason-code column to scored transactions."""
    df = df_scored.copy()
    cutoff = high_amount_cutoff(df)

    df["reason_codes"] = [
        "; ".join(
            reason_codes_for_row(
                record,
                threshold=threshold,
                amount_cutoff=cutoff,
                max_reasons=max_reasons,
            )
        )
        for record in df.to_dict("records")
    ]

    return df


def humanize_feature_name(feature_name: str) -> str:
    """Convert transformed sklearn feature names into analyst-friendly text."""
    raw = str(feature_name)

    for prefix in ("numeric__", "num__", "categorical__", "cat__"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break

    # OneHotEncoder feature names often look like: merchant_category_crypto.
    for base in ["transaction_type", "merchant_category", "country"]:
        if raw.startswith(base + "_"):
            value = raw[len(base) + 1 :].replace("_", " ")
            friendly_base = FRIENDLY_FEATURE_NAMES.get(base, base.replace("_", " "))
            return f"{friendly_base} = {value}"

    return FRIENDLY_FEATURE_NAMES.get(raw, raw.replace("_", " "))


def positive_class_shap_values(shap_values: object) -> np.ndarray:
    """Return SHAP values for the positive (fraud) class, across shap versions.

    ``TreeExplainer.shap_values`` has returned different shapes over time for
    binary classifiers:

    - legacy shap: a list ``[class_0_array, class_1_array]``;
    - modern shap (>= 0.43): a single array shaped
      ``(n_samples, n_features, n_classes)``.

    This normalizes both (and an already-2-D array) to the class-1 values so the
    rest of the code can stay version-agnostic.
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[..., 1]
    return arr


def shap_reason_codes(
    shap_values: Iterable[float],
    feature_names: Iterable[str],
    *,
    max_reasons: int = 5,
) -> list[str]:
    """Convert SHAP values into concise analyst-friendly reason codes."""
    values = np.asarray(list(shap_values), dtype=float).reshape(-1)
    names = np.asarray(list(feature_names), dtype=object).reshape(-1)

    n = min(len(values), len(names))
    if n == 0:
        return ["No SHAP reason codes available"]

    values = values[:n]
    names = names[:n]

    order = np.argsort(np.abs(values))[::-1]
    reasons: list[str] = []

    for idx in order:
        value = float(values[idx])
        if np.isclose(value, 0.0):
            continue

        feature = humanize_feature_name(str(names[idx]))
        direction = "increased" if value > 0 else "reduced"
        reasons.append(f"{feature} {direction} fraud risk")

        if len(reasons) >= max_reasons:
            break

    return reasons or ["No strong SHAP drivers identified"]


def split_reason_codes(value: str | float | None) -> list[str]:
    """Split a saved reason-code string back into displayable list items."""
    if value is None or pd.isna(value):
        return []

    return [item.strip() for item in str(value).split(";") if item.strip()]
