"""
Data ingestion: load raw transactions from CSV and perform basic validation.
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"transaction_id", "user_id", "timestamp", "amount", "merchant"}


def load_transactions(path: str) -> pd.DataFrame:
    """Load transactions from a CSV file and validate required columns."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount"] = df["amount"].astype(float)
    logger.info("Loaded %d transactions from %s", len(df), path)
    return df


def load_labeled_transactions(path: str) -> pd.DataFrame:
    """Load labeled transactions (includes 'is_fraud' column)."""
    df = load_transactions(path)
    if "is_fraud" not in df.columns:
        raise ValueError("Labeled dataset must contain 'is_fraud' column")
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df
