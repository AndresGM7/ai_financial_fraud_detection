"""
FastAPI application: exposes the fraud detection pipeline as an HTTP endpoint.
"""
import logging
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.enrichment import enrich_transactions, normalize_merchant
from src.features import build_features, get_feature_columns
from src.models.statistical import combine_statistical
from src.decision_engine import combine_signals
from src.llm.rag_retriever import retrieve_context
from src.llm.prompt_builder import build_prompt, build_system_prompt
from src.llm.output_parser import parse_llm_output
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Fraud Detection API",
    description=(
        "Hybrid fraud detection system combining statistical rules, "
        "unsupervised ML, supervised ML, and LLM reasoning via RAG."
    ),
    version="1.0.0",
)

# In-memory transaction history for RAG context (replace with DB in production)
_transaction_history: pd.DataFrame = pd.DataFrame()


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User identifier")
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant: str = Field(..., description="Merchant name")
    llm_enabled: bool = Field(False, description="Enable LLM reasoning (requires API key)")


class TransactionResponse(BaseModel):
    transaction_id: str
    decision: str
    score: float
    signals: dict
    llm_assessment: Optional[dict] = None


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/analyze", response_model=TransactionResponse)
def analyze(request: TransactionRequest) -> TransactionResponse:
    """
    Analyze a single transaction for fraud risk.

    Runs the full detection pipeline:
    1. Enrich transaction with temporal/merchant features
    2. Compute statistical signals
    3. Optionally run LLM reasoning with RAG context
    4. Fuse all signals via the decision engine
    """
    global _transaction_history

    # Build a single-row DataFrame for this transaction
    tx_dict = request.model_dump()
    tx_df = pd.DataFrame([tx_dict])

    # Combine with history for rolling stats
    if not _transaction_history.empty:
        combined = pd.concat([_transaction_history, tx_df], ignore_index=True)
    else:
        combined = tx_df.copy()

    try:
        enriched = enrich_transactions(combined)
        featured = build_features(enriched)
    except Exception as exc:
        logger.exception("Feature engineering failed")
        raise HTTPException(status_code=500, detail=f"Feature engineering error: {exc}")

    # Get the row for the current transaction
    current_row = featured[featured["transaction_id"] == request.transaction_id]
    if current_row.empty:
        raise HTTPException(status_code=500, detail="Transaction not found after enrichment")

    tx_row = current_row.iloc[0]

    # Statistical signal
    stat_score = float(combine_statistical(current_row).iloc[0])

    # Unsupervised: use z_amount as a simple proxy (no trained model here)
    unsup_score = float(abs(tx_row.get("z_amount", 0)) > 3.0)

    # Supervised: not available without a trained model
    sup_score = None

    # LLM reasoning (optional)
    llm_result = {"risk": "low", "reason": "LLM disabled", "action": "ignore", "confidence": 0.0}
    if request.llm_enabled:
        try:
            from src.llm.llm_client import LLMClient
            context = retrieve_context(request.user_id, enriched)
            prompt = build_prompt(tx_row.to_dict(), context)
            system_prompt = build_system_prompt()
            client = LLMClient()
            raw = client.complete(system_prompt, prompt)
            llm_result = parse_llm_output(raw)
        except Exception as exc:
            logger.warning("LLM call failed, using default: %s", exc)

    # Decision
    decision_result = combine_signals(stat_score, unsup_score, sup_score, llm_result)

    # Update history
    _transaction_history = pd.concat(
        [_transaction_history, tx_df], ignore_index=True
    ).tail(10000)

    return TransactionResponse(
        transaction_id=request.transaction_id,
        decision=decision_result["decision"],
        score=decision_result["score"],
        signals=decision_result["signals"],
        llm_assessment=llm_result if request.llm_enabled else None,
    )


@app.delete("/history")
def clear_history() -> dict:
    """Clear the in-memory transaction history."""
    global _transaction_history
    _transaction_history = pd.DataFrame()
    return {"status": "cleared"}
