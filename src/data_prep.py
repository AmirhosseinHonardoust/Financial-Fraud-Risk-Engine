import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
)
from .validation import validate_training_dataframe


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the raw synthetic fraud dataset."""
    csv_path = RAW_DATA_PATH if path is None else path
    df = pd.read_csv(csv_path)
    validate_training_dataframe(df, context="raw fraud dataset")
    return df


def train_test_split_stratified(df: pd.DataFrame):
    """Create a stratified train/test split on the fraud label."""
    validate_training_dataframe(df, context="raw fraud dataset")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )

    return train_df, test_df


def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save train and test data to the processed folder."""
    train_path = PROCESSED_DATA_DIR / "transactions_train.csv"
    test_path = PROCESSED_DATA_DIR / "transactions_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train data to: {train_path}")
    print(f"Saved test data to:  {test_path}")


def main() -> None:
    df = load_raw_data()
    print(f"Loaded raw data with shape: {df.shape}")

    train_df, test_df = train_test_split_stratified(df)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    print(train_df[TARGET_COL].value_counts(normalize=True).rename("train_class_ratio"))
    print(test_df[TARGET_COL].value_counts(normalize=True).rename("test_class_ratio"))

    save_processed(train_df, test_df)


if __name__ == "__main__":
    main()
