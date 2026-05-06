"""
Feature store: persist and retrieve computed feature sets.
"""
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    """Simple file-backed feature store using Parquet."""

    def __init__(self, path: str = "data/feature_store.parquet"):
        self.path = path

    def save(self, df: pd.DataFrame) -> None:
        """Persist features to disk."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        df.to_parquet(self.path, index=False)
        logger.info("Saved %d rows to feature store at %s", len(df), self.path)

    def load(self) -> pd.DataFrame:
        """Load features from disk."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Feature store not found at {self.path}")
        df = pd.read_parquet(self.path)
        logger.info("Loaded %d rows from feature store at %s", len(df), self.path)
        return df

    def exists(self) -> bool:
        return os.path.exists(self.path)
