"""
Configuration settings for the fraud detection system.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Statistical thresholds
ZSCORE_THRESHOLD: float = float(os.getenv("ZSCORE_THRESHOLD", "3.0"))
NIGHT_AMOUNT_MULTIPLIER: float = float(os.getenv("NIGHT_AMOUNT_MULTIPLIER", "3.0"))
ROLLING_WINDOW: int = int(os.getenv("ROLLING_WINDOW", "7"))

# Isolation Forest
IF_N_ESTIMATORS: int = int(os.getenv("IF_N_ESTIMATORS", "200"))
IF_CONTAMINATION: float = float(os.getenv("IF_CONTAMINATION", "0.02"))

# Decision engine weights
WEIGHT_STATISTICAL: float = 0.2
WEIGHT_UNSUPERVISED: float = 0.3
WEIGHT_SUPERVISED: float = 0.2
WEIGHT_LLM_HIGH: float = 0.5

ALERT_THRESHOLD: float = 0.7
MONITOR_THRESHOLD: float = 0.4

# Feature store
FEATURE_STORE_PATH: str = os.getenv("FEATURE_STORE_PATH", "data/feature_store.parquet")

# Data paths
RAW_DATA_PATH: str = os.getenv("RAW_DATA_PATH", "data/raw_transactions.csv")
LABELED_DATA_PATH: str = os.getenv("LABELED_DATA_PATH", "data/labeled_transactions.csv")
