from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DATA_PATH, TARGET_COL
from .validation import validate_training_dataframe

TRANSACTION_TYPES = ["purchase", "transfer", "withdrawal", "refund"]
MERCHANT_CATEGORIES = [
    "retail",
    "grocery",
    "travel",
    "electronics",
    "gaming",
    "crypto",
]
COUNTRIES = ["DE", "FR", "NL", "GB", "US", "CN", "RU"]


def _normalize_probs(probs: Sequence[float]) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")
    return arr / total


def _sample_by_class(
    rng: np.random.Generator,
    is_fraud: np.ndarray,
    legit_probs: Sequence[float],
    fraud_probs: Sequence[float],
    values: Sequence[str],
) -> np.ndarray:
    """Sample categorical values from class-conditional distributions."""
    out = np.empty(len(is_fraud), dtype=object)
    legit_mask = ~is_fraud
    fraud_mask = is_fraud

    out[legit_mask] = rng.choice(
        values, size=int(legit_mask.sum()), p=_normalize_probs(legit_probs)
    )
    out[fraud_mask] = rng.choice(
        values, size=int(fraud_mask.sum()), p=_normalize_probs(fraud_probs)
    )
    return out


def generate_synthetic_fraud_dataset(
    *,
    n_samples: int = 5_000,
    fraud_rate: float = 0.08,
    label_noise: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a harder synthetic fraud dataset with overlapping class patterns.

    The original demo data was nearly deterministic and perfectly separable. This
    generator keeps the same public schema, but introduces overlap, stochastic
    labels, and false-positive-looking legitimate transactions so evaluation is
    more meaningful.
    """
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100 for a useful fraud demo dataset.")

    if not 0.0 < fraud_rate < 0.5:
        raise ValueError("fraud_rate must be between 0 and 0.5.")

    if not 0.0 <= label_noise < 0.3:
        raise ValueError("label_noise must be between 0 and 0.3.")

    rng = np.random.default_rng(seed)

    is_fraud = rng.random(n_samples) < fraud_rate

    # Inject controlled label noise so the problem is not perfectly separable.
    flip_mask = rng.random(n_samples) < label_noise
    is_fraud = np.where(flip_mask, ~is_fraud, is_fraud)

    legit_mask = ~is_fraud
    fraud_mask = is_fraud

    amount = np.empty(n_samples)
    amount[legit_mask] = rng.lognormal(mean=3.75, sigma=0.85, size=int(legit_mask.sum()))
    amount[fraud_mask] = rng.lognormal(mean=4.45, sigma=0.95, size=int(fraud_mask.sum()))

    # Add false-positive-looking legitimate high-value transactions.
    high_legit = legit_mask & (rng.random(n_samples) < 0.08)
    amount[high_legit] *= rng.uniform(3.0, 8.0, size=int(high_legit.sum()))
    amount = np.clip(amount, 2.0, 8_000.0).round(2)

    hour = np.empty(n_samples, dtype=int)
    legit_hour_probs = _normalize_probs(
        [
            0.015,
            0.012,
            0.010,
            0.010,
            0.012,
            0.018,
            0.030,
            0.050,
            0.065,
            0.075,
            0.075,
            0.070,
            0.065,
            0.060,
            0.055,
            0.052,
            0.050,
            0.048,
            0.045,
            0.040,
            0.035,
            0.030,
            0.023,
            0.015,
        ]
    )
    fraud_hour_probs = _normalize_probs(
        [
            0.070,
            0.080,
            0.085,
            0.075,
            0.055,
            0.040,
            0.030,
            0.025,
            0.025,
            0.030,
            0.035,
            0.035,
            0.035,
            0.035,
            0.035,
            0.035,
            0.035,
            0.040,
            0.045,
            0.050,
            0.055,
            0.060,
            0.060,
            0.060,
        ]
    )

    hour[legit_mask] = rng.choice(
        np.arange(24),
        size=int(legit_mask.sum()),
        p=legit_hour_probs,
    )
    hour[fraud_mask] = rng.choice(
        np.arange(24),
        size=int(fraud_mask.sum()),
        p=fraud_hour_probs,
    )

    device_risk_score = np.empty(n_samples)
    ip_risk_score = np.empty(n_samples)

    # Overlapping beta distributions: fraud tends to be higher risk, but there is
    # meaningful overlap and noisy legitimate behavior.
    device_risk_score[legit_mask] = rng.beta(1.4, 5.5, size=int(legit_mask.sum()))
    device_risk_score[fraud_mask] = rng.beta(3.0, 2.2, size=int(fraud_mask.sum()))

    ip_risk_score[legit_mask] = rng.beta(1.3, 5.2, size=int(legit_mask.sum()))
    ip_risk_score[fraud_mask] = rng.beta(2.8, 2.4, size=int(fraud_mask.sum()))

    noisy_legit = legit_mask & (rng.random(n_samples) < 0.06)
    device_risk_score[noisy_legit] = rng.beta(3.0, 2.0, size=int(noisy_legit.sum()))
    ip_risk_score[noisy_legit] = rng.beta(2.8, 2.2, size=int(noisy_legit.sum()))

    low_risk_fraud = fraud_mask & (rng.random(n_samples) < 0.12)
    device_risk_score[low_risk_fraud] = rng.beta(1.5, 4.5, size=int(low_risk_fraud.sum()))
    ip_risk_score[low_risk_fraud] = rng.beta(1.5, 4.2, size=int(low_risk_fraud.sum()))

    device_risk_score = np.round(np.clip(device_risk_score, 0, 1), 4).astype(float)
    ip_risk_score = np.round(np.clip(ip_risk_score, 0, 1), 4).astype(float)

    transaction_type = _sample_by_class(
        rng,
        is_fraud,
        legit_probs=[0.74, 0.12, 0.09, 0.05],
        fraud_probs=[0.44, 0.30, 0.20, 0.06],
        values=TRANSACTION_TYPES,
    )

    merchant_category = _sample_by_class(
        rng,
        is_fraud,
        legit_probs=[0.33, 0.25, 0.13, 0.16, 0.08, 0.05],
        fraud_probs=[0.15, 0.08, 0.18, 0.20, 0.16, 0.23],
        values=MERCHANT_CATEGORIES,
    )

    country = _sample_by_class(
        rng,
        is_fraud,
        legit_probs=[0.38, 0.18, 0.13, 0.12, 0.10, 0.05, 0.04],
        fraud_probs=[0.18, 0.10, 0.08, 0.11, 0.12, 0.20, 0.21],
        values=COUNTRIES,
    )

    df = pd.DataFrame(
        {
            "transaction_id": np.arange(1, n_samples + 1),
            "user_id": rng.integers(10_000, 99_999, size=n_samples),
            "amount": amount,
            "hour": hour,
            "device_risk_score": device_risk_score,
            "ip_risk_score": ip_risk_score,
            "transaction_type": transaction_type,
            "merchant_category": merchant_category,
            "country": country,
            TARGET_COL: is_fraud.astype(int),
        }
    )

    validate_training_dataframe(df, context="generated synthetic fraud dataset")
    return df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a harder synthetic fraud dataset for the demo pipeline."
    )
    parser.add_argument("--output", default=str(RAW_DATA_PATH), help="Output CSV path.")
    parser.add_argument("--rows", type=int, default=5_000, help="Number of rows to generate.")
    parser.add_argument(
        "--fraud-rate", type=float, default=0.08, help="Approximate base fraud rate."
    )
    parser.add_argument(
        "--label-noise", type=float, default=0.03, help="Fraction of labels to flip."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    df = generate_synthetic_fraud_dataset(
        n_samples=args.rows,
        fraud_rate=args.fraud_rate,
        label_noise=args.label_noise,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved synthetic fraud dataset to: {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Fraud rate: {df[TARGET_COL].mean():.3%}")


if __name__ == "__main__":
    main()
