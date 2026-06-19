from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "synthetic_fraud_dataset.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

# Create directories if they don't exist
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, METRICS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Column configuration
TARGET_COL = "is_fraud"

ID_COLS = ["transaction_id", "user_id"]

CATEGORICAL_FEATURES = [
    "transaction_type",
    "merchant_category",
    "country",
]

NUMERIC_FEATURES = [
    "amount",
    "hour",
    "device_risk_score",
    "ip_risk_score",
]

# We can optionally include IDs as numeric features; here we keep them separate
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ID_COLS

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Threshold search configuration
THRESHOLD_GRID = [i / 100 for i in range(5, 100, 5)]  # 0.05, 0.10, ..., 0.95

# Very simple "business" cost assumptions
COST_FALSE_NEGATIVE = 10.0  # missing a fraud is very expensive
COST_FALSE_POSITIVE = 1.0  # incorrectly flagging a normal transaction
