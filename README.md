# AI Financial Fraud Detection

A **hybrid financial risk detection system** that combines statistical anomaly detection, supervised/unsupervised ML, and LLM-based contextual reasoning via RAG — all orchestrated in a modular pipeline with rigorous evaluation and cost/latency control.

## Architecture

```
Raw Data
  ↓
Ingestion          (src/ingestion.py)
  ↓
Data Enrichment    (src/enrichment.py)  ← merchant normalization, temporal features
  ↓
Feature Store      (src/feature_store.py)
  ↓
Detection Layer
   ├─ Statistical  (src/models/statistical.py)   Z-score, EWMA, rules
   ├─ Unsupervised (src/models/unsupervised.py)  IsolationForest, LOF
   ├─ Supervised   (src/models/supervised.py)    LogReg, XGBoost [when labels exist]
   └─ LLM          (src/llm/)                    RAG reasoning via OpenAI
  ↓
Decision Engine    (src/decision_engine.py)  signal fusion + thresholds + policies
  ↓
Evaluation         (src/evaluation/)         offline + time-aware cross-validation
  ↓
API                (src/api.py)              FastAPI + structured logging
```

## Project Structure

```
fraud-detection-ai/
├── data/
│   ├── raw_transactions.csv          # synthetic unlabeled transactions
│   └── labeled_transactions.csv      # synthetic labeled transactions (~5% fraud)
├── src/
│   ├── config.py                     # configuration (env-aware)
│   ├── ingestion.py                  # data loading & validation
│   ├── enrichment.py                 # merchant normalization, temporal features
│   ├── features.py                   # z-score, new-merchant flag, tx frequency
│   ├── feature_store.py              # Parquet-backed feature persistence
│   ├── decision_engine.py            # weighted signal fusion
│   ├── api.py                        # FastAPI endpoint
│   ├── utils.py                      # logging helpers
│   ├── models/
│   │   ├── statistical.py            # z-score rule, night rule, EWMA
│   │   ├── unsupervised.py           # IsolationForest, LOF
│   │   └── supervised.py             # LogisticRegression, XGBoost
│   ├── llm/
│   │   ├── rag_retriever.py          # user behavioral context for RAG
│   │   ├── prompt_builder.py         # structured prompt construction
│   │   ├── llm_client.py             # OpenAI API wrapper + MockLLMClient
│   │   └── output_parser.py          # JSON response parsing & validation
│   └── evaluation/
│       ├── metrics.py                # precision, recall, F1, ROC-AUC
│       ├── cross_validation.py       # time-aware walk-forward CV
│       └── error_analysis.py         # FP/FN analysis
└── tests/                            # pytest test suite (65 tests)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API

```bash
uvicorn src.api:app --reload
```

### 3. Analyze a transaction

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_001",
    "user_id": "user_42",
    "timestamp": "2024-06-15T02:30:00",
    "amount": 4999.99,
    "merchant": "Unknown Vendor",
    "llm_enabled": false
  }'
```

**Response:**
```json
{
  "transaction_id": "tx_001",
  "decision": "alert",
  "score": 0.7,
  "signals": {
    "statistical": 1,
    "unsupervised": 1,
    "supervised": null,
    "llm_risk": "low",
    "llm_reason": "LLM disabled",
    "llm_action": "ignore"
  }
}
```

### 4. Enable LLM reasoning (requires OpenAI API key)

```bash
export OPENAI_API_KEY=sk-...
```

Then set `"llm_enabled": true` in the request.

### 5. Run tests

```bash
pytest tests/ -v
```

## Detection Layers

### Statistical (rule-based, zero latency)
- **Z-score rule**: flags transactions where `|z_amount| > 3σ` vs user's 7-tx rolling mean
- **Night rule**: flags nighttime transactions (`00:00–05:59`) with amount > 3× rolling mean
- **EWMA anomaly**: flags deviations from exponentially weighted moving average

### Unsupervised (no labels required)
- **Isolation Forest**: ensemble of random partitioning trees
- **Local Outlier Factor**: density-based anomaly detection

### Supervised (when labels exist)
- **Logistic Regression**: interpretable, fast baseline
- **XGBoost**: gradient-boosted trees with class imbalance handling

### LLM + RAG
- Retrieves user's behavioral profile (avg amount, typical merchants, night activity ratio)
- Builds a structured prompt with the current transaction context
- Parses structured JSON response (`risk`, `reason`, `action`, `confidence`)

## Decision Engine

Signals are fused with configurable weights:

| Signal | Weight |
|--------|--------|
| Statistical | 0.2 |
| Unsupervised | 0.3 |
| Supervised | 0.2 |
| LLM high-risk boost | +0.5 |

Thresholds: `score ≥ 0.7 → alert`, `score ≥ 0.4 → monitor`, else `ignore`.

## Evaluation

Uses **time-based walk-forward splits** (not standard k-fold) to prevent temporal leakage — a critical consideration for sequential financial data.

```python
from src.evaluation.cross_validation import cross_validate_model
from src.models.supervised import train_logreg

results = cross_validate_model(train_logreg, X, y, df, n_splits=5)
```

## Configuration

All settings configurable via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model |
| `ZSCORE_THRESHOLD` | `3.0` | Z-score alert threshold |
| `IF_CONTAMINATION` | `0.02` | Isolation Forest contamination rate |
| `ALERT_THRESHOLD` | `0.7` | Decision engine alert threshold |
| `MONITOR_THRESHOLD` | `0.4` | Decision engine monitor threshold |

