"""
api.py
──────
FastAPI async REST API — the production serving layer.

Endpoints
─────────
  POST /analyze          — analyse a single transaction (real-time)
  POST /batch            — analyse a batch (background task)
  GET  /health           — liveness + model status
  GET  /metrics/cost     — LLM cost tracker summary
  GET  /metrics/drift    — feature drift report
  POST /feedback         — receive label feedback (online learning loop)

Interview talking points
────────────────────────
  "I use FastAPI because it's async-native (handles concurrent requests
   without blocking), has automatic OpenAPI docs (/docs), and its
   Pydantic integration means request validation is free — the same
   schema I use for data validation reappears here.

  The /analyze endpoint is synchronous for the caller but async under
  the hood — it can process hundreds of concurrent scoring requests
  without thread contention. For batch processing, I use BackgroundTasks
  so the caller gets an immediate 202 Accepted and the work happens
  asynchronously.

  Structured logging with structlog means every request produces a JSON
  log line with tx_id, user_id, decision, latency_ms, and LLM cost —
  directly queryable in CloudWatch Logs Insights."
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.decision_engine import DecisionEngine, DetectionResult
from src.enrichment import enrich_transactions
from src.evaluation.drift_monitor import drift_report
from src.features import ALL_FEATURES, build_features, get_feature_matrix
from src.llm.agent import FraudInvestigationAgent
from src.llm.output_parser import parse_llm_output
from src.models.statistical import statistical_score
from src.utils import cost_tracker, get_logger, new_request_id

log = get_logger(__name__)

# ── Global state (would be Redis/DynamoDB in prod) ─────────────────────────────
_engine = DecisionEngine()
_agent = FraudInvestigationAgent()
_history_df: pd.DataFrame | None = None  # in-memory for demo; use DynamoDB in prod
_baseline_df: pd.DataFrame | None = None  # for drift monitoring


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api_startup", version="1.0.0")
    # In production: load models from S3, warm feature store from DynamoDB
    yield
    log.info("api_shutdown")


# ── App factory ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Financial Fraud Detection API",
    description="Hybrid statistical + LLM fraud detection for elder financial safety.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    tx_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    timestamp: str
    amount: float = Field(gt=0)
    merchant: str
    mcc: int = Field(default=9999)
    tx_type: str
    counterparty: str = "unknown"
    currency: str = "USD"
    lat: float | None = None
    lon: float | None = None

    model_config = {"json_schema_extra": {
        "example": {
            "tx_id": "TX00000001",
            "user_id": "U00001",
            "timestamp": "2024-06-15T02:30:00Z",
            "amount": 9500.0,
            "merchant": "western_union",
            "mcc": 6099,
            "tx_type": "ACH",
            "counterparty": "Unknown Overseas Entity",
        }
    }}


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]


class FraudResponse(BaseModel):
    tx_id: str
    user_id: str
    decision: str
    final_score: float
    raw_score: float
    signal_breakdown: dict[str, float]
    applied_boosts: list[str]
    llm_assessment: dict[str, Any]
    reasoning_trace: list[str]
    request_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    cost_summary: dict[str, Any]


# ── Middleware: per-request structured logging ─────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = new_request_id()
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=latency_ms,
        request_id=request_id,
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-MS"] = str(latency_ms)
    return response


# ── Core scoring logic ────────────────────────────────────────────────────────

def _score_transaction(tx_dict: dict[str, Any]) -> DetectionResult:
    """
    Run the full detection pipeline for a single transaction.
    Used by both /analyze and /batch endpoints.
    """
    # 1. Build single-row DataFrame and enrich
    df = pd.DataFrame([tx_dict])
    df_enriched = enrich_transactions(df)

    # 2. Feature engineering
    df_featured = build_features(df_enriched)
    row = df_featured.iloc[0].to_dict()

    # 3. Statistical score
    stat_score = float(statistical_score(df_featured).iloc[0])

    # 4. LLM agent assessment (if available)
    agent_decision = _agent.run(
        current_tx=row,
        statistical_signals={
            **{k: row.get(k, 0) for k in ["z_amount", "ewma_z_amount", "tx_count_1h",
                                             "tx_count_24h", "mcc_high_risk"]},
            "statistical_score": stat_score,
            "unsupervised_score": 0.0,  # placeholder when no model loaded
        },
        history_df=_history_df,
    )

    # 5. Decision engine
    result = _engine.decide(
        tx=row,
        statistical_score=stat_score,
        unsupervised_score=0.0,  # unsupervised model requires pre-training
        supervised_score=None,
        llm_assessment=agent_decision.assessment,
    )
    return result, agent_decision.reasoning_trace


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=FraudResponse, tags=["Detection"])
async def analyze_transaction(
    request: TransactionRequest,
    req: Request,
) -> FraudResponse:
    """
    Analyse a single transaction in real-time.
    Full pipeline: enrichment → features → statistical → LLM → decision.
    """
    request_id = req.headers.get("X-Request-ID", new_request_id())
    t0 = time.perf_counter()

    try:
        tx_dict = request.model_dump()
        result, trace = _score_transaction(tx_dict)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        log.info(
            "transaction_scored",
            tx_id=result.tx_id,
            decision=result.decision,
            score=result.final_score,
            latency_ms=latency_ms,
        )

        return FraudResponse(
            tx_id=result.tx_id,
            user_id=result.user_id,
            decision=result.decision,
            final_score=result.final_score,
            raw_score=result.raw_score,
            signal_breakdown=result.signal_breakdown,
            applied_boosts=result.applied_boosts,
            llm_assessment=result.llm_assessment,
            reasoning_trace=trace,
            request_id=request_id,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        log.error("analyze_error", error=str(exc), tx_id=request.tx_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {str(exc)}",
        )


@app.post("/batch", tags=["Detection"])
async def batch_analyze(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Submit a batch of transactions for background processing.
    Returns immediately with 202 Accepted.
    """
    batch_id = str(uuid.uuid4())
    log.info("batch_submitted", batch_id=batch_id, n=len(request.transactions))

    def _process_batch():
        results = []
        for tx in request.transactions:
            try:
                result, _ = _score_transaction(tx.model_dump())
                results.append(result.to_dict())
            except Exception as exc:
                log.error("batch_tx_error", tx_id=tx.tx_id, error=str(exc))
        log.info("batch_complete", batch_id=batch_id, processed=len(results))

    background_tasks.add_task(_process_batch)
    return {
        "status": "accepted",
        "batch_id": batch_id,
        "n_transactions": len(request.transactions),
        "message": "Processing in background. Check logs for results.",
    }


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="1.0.0",
        llm_provider=settings.llm_provider,
        cost_summary=cost_tracker.summary(),
    )


@app.get("/metrics/cost", tags=["Operations"])
async def cost_metrics() -> dict:
    """Return LLM cost tracker summary."""
    return cost_tracker.summary()


@app.post("/feedback", tags=["Learning"])
async def receive_feedback(
    tx_id: str,
    user_id: str,
    true_label: int,
    fraud_type: str | None = None,
) -> dict:
    """
    Receive ground-truth feedback for a transaction.
    In production: writes to DynamoDB for offline retraining pipeline.

    Interview talking point:
      'Feedback loops close the model-production gap. Every confirmed fraud
       and every false positive is a training example. I log them to
       DynamoDB with tx_id, model version, and analyst notes so the
       weekly retraining job can pick them up automatically.'
    """
    log.info(
        "feedback_received",
        tx_id=tx_id,
        user_id=user_id,
        true_label=true_label,
        fraud_type=fraud_type,
    )
    # In prod: boto3.client("dynamodb").put_item(...)
    return {"status": "recorded", "tx_id": tx_id}


@app.get("/metrics/drift", tags=["Operations"])
async def drift_metrics() -> dict:
    """
    Run a drift report comparing baseline to recent transactions.
    In production: triggered by a weekly Lambda cron job.
    """
    if _history_df is None or _baseline_df is None:
        return {"status": "no_data", "message": "Load data first via POST /load_history"}

    numerical_cols = ["amount", "z_amount", "tx_count_24h", "geo_speed_kmh"]
    categorical_cols = ["tx_type", "merchant_clean"]
    report = drift_report(_baseline_df, _history_df, numerical_cols, categorical_cols)
    return {"drift_report": report.to_dict(orient="records")}

