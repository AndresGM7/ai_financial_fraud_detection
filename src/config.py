"""
config.py
─────────
Central configuration via Pydantic BaseSettings.
All values can be overridden by environment variables or a .env file.

Interview talking point:
  "I centralise configuration in a typed settings object so every module
   reads the same source-of-truth and environment-specific overrides
   require zero code changes — just swap the .env file."
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Providers ─────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o")
    anthropic_api_key: str = Field(default="")
    llm_provider: Literal["openai", "bedrock", "mock"] = Field(default="mock")

    # ── AWS ───────────────────────────────────────────────────────────────────
    aws_access_key_id: str = Field(default="")
    aws_secret_access_key: str = Field(default="")
    aws_default_region: str = Field(default="us-east-1")
    aws_bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    dynamodb_table_name: str = Field(default="fraud_alerts")
    sns_alert_topic_arn: str = Field(default="")

    # ── Feature Store ─────────────────────────────────────────────────────────
    feature_store_path: Path = Field(default=Path("data/processed/feature_store.parquet"))
    knowledge_base_path: Path = Field(default=Path("data/knowledge_base"))

    # ── Detection Thresholds ──────────────────────────────────────────────────
    zscore_threshold: float = Field(default=3.0, ge=1.0, le=10.0)
    ewma_span: int = Field(default=20, description="EWMA span (# transactions)")
    cusum_drift_threshold: float = Field(default=5.0)
    isolation_forest_contamination: float = Field(default=0.02, ge=0.001, le=0.5)
    lof_n_neighbors: int = Field(default=20)

    # ── Decision Engine ───────────────────────────────────────────────────────
    weight_statistical: float = Field(default=0.20)
    weight_unsupervised: float = Field(default=0.25)
    weight_supervised: float = Field(default=0.15)
    weight_llm: float = Field(default=0.40)
    alert_score_threshold: float = Field(default=0.70)
    monitor_score_threshold: float = Field(default=0.40)
    alert_cooldown_hours: int = Field(default=24)

    # ── LLM Cost Controls ─────────────────────────────────────────────────────
    llm_max_tokens_input: int = Field(default=2000)
    llm_max_tokens_output: int = Field(default=500)
    llm_cost_per_1k_input: float = Field(default=0.005)
    llm_cost_per_1k_output: float = Field(default=0.015)
    llm_daily_budget_usd: float = Field(default=50.0)
    llm_temperature: float = Field(default=0.0)

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="fraud_detection")

    # ── Data Generation ───────────────────────────────────────────────────────
    synthetic_n_users: int = Field(default=500)
    synthetic_n_transactions: int = Field(default=50_000)
    synthetic_fraud_rate: float = Field(default=0.02)
    random_seed: int = Field(default=42)

    @field_validator("weight_statistical", "weight_unsupervised", "weight_supervised", "weight_llm")
    @classmethod
    def weights_positive(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("Weights must be between 0 and 1")
        return v


# Singleton — import this everywhere
settings = Settings()

