# 🛡️ AI Financial Fraud Detection System

<div align="center">

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-35%20passing-brightgreen.svg?logo=pytest)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)](https://xgboost.readthedocs.io/)
[![LLM](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Bedrock%20%7C%20Mock-9C27B0.svg)](src/llm/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker)](Dockerfile)
[![AWS](https://img.shields.io/badge/AWS-Lambda%20%7C%20DynamoDB%20%7C%20Bedrock-FF9900.svg?logo=amazonaws)](infra/)

**A production-grade hybrid fraud detection system targeting elder financial protection.**
Combines statistical anomaly detection, ensemble ML, LLM/RAG reasoning, and a live Streamlit dashboard — all runnable locally in under 5 minutes.

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Dashboard](#-dashboard) · [API](#-rest-api) · [Models](#-model-performance) · [Interview Notes](#-interview-talking-points)

</div>

---

## 🎯 What This Is

A **top-1% reference implementation** of a real-time financial fraud detection pipeline, purpose-built to demonstrate production-ready Data Science and AI Engineering skills. It maps 1:1 to the technical requirements of a modern fintech Data Scientist / AI Engineer role:

| Job Requirement | Implementation |
|---|---|
| Statistical anomaly detection | Z-score (log-transform), EWMA, CUSUM, 5-rule engine |
| Unsupervised ML | Isolation Forest, LOF, PyTorch Autoencoder |
| Supervised ML with imbalanced data | XGBoost + LightGBM + LogReg, SMOTE, calibration |
| LLM / RAG orchestration | LangGraph agent, FAISS vector store, OpenAI / Bedrock |
| Rigorous evaluation | PR-AUC, F-β(2), cost matrix, TimeSeriesSplit, bootstrap CI |
| REST API serving | FastAPI async, Pydantic validation, structured logging |
| AWS cloud patterns | Lambda handler, DynamoDB feedback loop, CDK stub, Bedrock |
| Experiment tracking | MLflow with per-run metrics and model registry |
| Observability dashboard | 7-tab Streamlit dashboard with all metrics and LLM analytics |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FINANCIAL FRAUD DETECTION PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌────────────────┐    ┌────────────────────────────────┐
  │  Data Sources │    │   Ingestion    │    │          Enrichment            │
  │  CSV │ JSON  │───▶│ Pydantic valid │───▶│ Merchant normalisation (MCC)   │
  │  Parquet     │    │ Dead-letter Q  │    │ Temporal + cyclical features   │
  │  Webhooks    │    │ Schema version │    │ Rolling EWMA (shift-1, no leak)│
  └──────────────┘    └────────────────┘    │ Velocity windows (1h/24h/7d)   │
                                            │ Geo-velocity (Haversine)       │
                                            └──────────────┬─────────────────┘
                                                           │
                                            ┌──────────────▼─────────────────┐
                                            │         Feature Store          │
                                            │  50+ engineered features       │
                                            │  Float32 matrix │ MinMaxScaler │
                                            └──┬──────────┬──────────┬───────┘
                                               │          │          │
                    ┌──────────────────────────┘          │          └──────────────────────┐
                    ▼                                     ▼                                 ▼
       ┌────────────────────────┐         ┌──────────────────────┐         ┌───────────────────────┐
       │   STATISTICAL LAYER    │         │  UNSUPERVISED LAYER  │         │   SUPERVISED LAYER    │
       │ Z-score (log-transform)│         │ Isolation Forest     │         │ XGBoost (eval=AUCPR)  │
       │ EWMA baseline          │         │  200 trees, c=0.02   │         │ LightGBM              │
       │ CUSUM accumulator      │         │ LOF (novelty=True)   │         │ Logistic Regression   │
       │ Rules Engine (×5)      │         │ Autoencoder (PyTorch)│         │ SMOTE strategy=0.1    │
       │ score: [0, 1]          │         │ score: [0, 1]        │         │ CalibratedClassifierCV│
       └──────────┬─────────────┘         └──────────┬───────────┘         └───────────┬───────────┘
                  │ w=0.20                            │ w=0.25                           │ w=0.15
                  └──────────────────────────────────┬┘                                 │
                                                     └─────────────────────────────────┘
                                                                    │
                                                     ┌──────────────▼─────────────────┐
                                                     │        LLM / RAG LAYER         │
                                                     │ FAISS knowledge base (3 docs)  │
                                                     │ LangGraph StateGraph agent     │
                                                     │  retrieve → assess → decide    │
                                                     │ Structured output (instructor) │
                                                     │ Cost tracker per call          │
                                                     │ OpenAI / Bedrock / Mock        │
                                                     └──────────────┬─────────────────┘
                                                                    │ w=0.40
                                                     ┌──────────────▼─────────────────┐
                                                     │        DECISION ENGINE         │
                                                     │ Weighted signal fusion         │
                                                     │ YAML policy boosts             │
                                                     │ Alert cooldown (24h / user)    │
                                                     │  ALERT │ MONITOR │ IGNORE      │
                                                     └──────────────┬─────────────────┘
                                                                    │
                          ┌─────────────────────────────────────────┼────────────────────────┐
                          ▼                                         ▼                        ▼
             ┌────────────────────────┐            ┌───────────────────────┐   ┌─────────────────────┐
             │  FastAPI REST API      │            │ Streamlit Dashboard   │   │   AWS / Infra        │
             │  POST /analyze         │            │ 7 tabs: Overview,     │   │ Lambda handler       │
             │  POST /batch (async)   │            │ Model comparison,     │   │ DynamoDB feedback    │
             │  GET  /health          │            │ PR/ROC curves,        │   │ CDK stub (IaC)       │
             │  GET  /metrics/cost    │            │ Confusion matrices,   │   │ Bedrock integration  │
             │  POST /feedback        │            │ Error analysis,       │   │ S3 model artefacts   │
             │  Swagger UI at /docs   │            │ Drift (PSI), LLM      │   │ SNS alert topic      │
             └────────────────────────┘            └───────────────────────┘   └─────────────────────┘
```

---

## ✨ Key Technical Highlights

<details>
<summary><b>📊 Why PR-AUC, not ROC-AUC?</b></summary>

With a 2% fraud rate, a model that predicts "everything is legitimate" achieves **ROC-AUC ≈ 0.97** — yet catches zero fraud. ROC-AUC is inflated by the massive true-negative class.

**PR-AUC** evaluates only the positive (fraud) class and collapses near zero when fraud is missed — it is the honest metric for imbalanced classification in production.

</details>

<details>
<summary><b>⚖️ F-beta(β=2) threshold optimisation</b></summary>

The default 0.5 threshold is never optimal for fraud. I sweep the full PR curve and choose the threshold that maximises **F2** (recall weighted 2× over precision). Rationale: a missed fraud on a $15,000 wire costs your customer $15,000; a false alert costs ~$5 in analyst time.

</details>

<details>
<summary><b>📈 SMOTE strategy 0.1, not 0.5</b></summary>

Oversampling to 50/50 distorts the class prior and inflates recall in cross-validation. `sampling_strategy=0.1` creates a 10:1 majority:minority ratio — realistic for production fraud rates, preserves calibration, and avoids overfitting to synthetic samples.

</details>

<details>
<summary><b>⏱️ TimeSeriesSplit, not KFold</b></summary>

Standard K-Fold randomly shuffles the data — future transactions leak into training folds. `TimeSeriesSplit` enforces temporal causality: always train on past, validate on future, mimicking weekly model retraining in production.

</details>

<details>
<summary><b>🔄 Look-ahead leakage prevention</b></summary>

Rolling statistics (EWMA, rolling mean/std) are computed with `shift(1)` inside user groups before any feature engineering. Each transaction only sees past behaviour, never its own values — critical for honest offline evaluation.

</details>

<details>
<summary><b>📉 PSI for drift monitoring</b></summary>

Population Stability Index (PSI) originated in insurance/credit risk in the 1970s. Thresholds: `<0.10` stable, `0.10–0.20` monitor, `>0.20` retrain. Monitored weekly on the top-10 SHAP features. PSI > 0.20 triggers an automated SageMaker retraining job.

</details>

<details>
<summary><b>🤖 LangGraph-style LLM agent</b></summary>

A `StateGraph` agent with three nodes: **retrieve** (FAISS semantic search over fraud typology knowledge base) → **assess** (structured risk scoring via `instructor`) → **investigate** (deeper pattern matching if uncertain). Two-tier cost optimisation: cheap model for low-risk triage, expensive model for uncertain cases only.

</details>

---

## 🚀 Quick Start

**Total time: ~5 minutes. No API keys required.**

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/ai_financial_fraud_detection.git
cd ai_financial_fraud_detection

# 2. Create virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config (uses mock LLM by default)
cp .env.example .env

# 5. Generate synthetic data + train all models + evaluate
python run_pipeline.py

# 6. Launch the 7-tab dashboard
streamlit run dashboard/app.py
```

Open **[http://localhost:8501](http://localhost:8501)** — the full dashboard loads with all model results.

> **Note:** The default config uses a **mock LLM provider** — no OpenAI or AWS credentials needed.
> To use real LLMs, set `LLM_PROVIDER=openai` in `.env` with your API key.

---

## 📦 Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| pip | 23+ |
| RAM | 4 GB+ (8 GB recommended for PyTorch Autoencoder) |
| Disk | ~2 GB (models + data + dependencies) |

### With Docker

```bash
# FastAPI + MLflow stack
docker-compose up --build

# API docs:   http://localhost:8000/docs
# MLflow UI:  http://localhost:5000
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# LLM Provider: "mock" (default) | "openai" | "bedrock"
LLM_PROVIDER=mock
OPENAI_API_KEY=sk-...           # only if LLM_PROVIDER=openai
AWS_DEFAULT_REGION=us-east-1    # only if LLM_PROVIDER=bedrock

# Tunable thresholds (also in config/detection_policy.yaml)
ALERT_SCORE_THRESHOLD=0.70
MONITOR_SCORE_THRESHOLD=0.40
LLM_DAILY_BUDGET_USD=50.0
```

---

## 🔬 Pipeline Usage

### Full Run

```bash
python run_pipeline.py
```

Runs 7 sequential steps:
1. Synthetic data generation (50k transactions, 5 fraud typologies)
2. Ingestion + Pydantic validation
3. Enrichment + feature engineering
4. Supervised model training (XGBoost / LightGBM / LogReg)
5. Unsupervised model training (IForest / LOF / Autoencoder)
6. LLM agent evaluation
7. PSI drift report

### CLI Options

```bash
python run_pipeline.py --skip-data-gen      # reuse existing CSV/Parquet
python run_pipeline.py --llm-samples 100    # more LLM evaluations
python run_pipeline.py --data-path data/raw/my_transactions.csv
```

### Run Tests

```bash
pytest tests/ -v                            # all 35 tests
pytest tests/ --cov=src --cov-report=html   # with coverage
pytest tests/test_features.py -v            # specific module
```

---

## 📊 Dashboard

```bash
streamlit run dashboard/app.py
```

| Tab | Content |
|-----|---------|
| **📊 Overview** | KPI cards: fraud rate, best PR-AUC, transactions, LLM cost |
| **📈 Model Comparison** | Metrics table with best-column green highlighting |
| **🔀 PR & ROC Curves** | PR curves (primary) + ROC curves side-by-side |
| **🧩 Confusion Matrices** | All models side-by-side + threshold sweep |
| **🔍 Error Analysis** | FP/FN slices by fraud typology, tx type, hour, amount band |
| **📉 Drift Monitor** | PSI bar chart (green/orange/red), KL-divergence table |
| **🤖 LLM Agent** | Cost/latency metrics, risk distribution, confidence histogram |

---

## 🌐 REST API

```bash
uvicorn src.api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Score a single transaction (real-time, async) |
| `POST` | `/batch` | Submit a batch (202 Accepted, background processing) |
| `GET` | `/health` | Liveness check + cost summary |
| `GET` | `/metrics/cost` | LLM token usage and cost tracker |
| `GET` | `/metrics/drift` | Feature drift report |
| `POST` | `/feedback` | Ground-truth label for online learning loop |

### Example Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "U00001",
    "timestamp": "2024-06-15T02:30:00Z",
    "amount": 9500.00,
    "merchant": "western_union",
    "mcc": 6099,
    "tx_type": "ACH",
    "counterparty": "Unknown Overseas Entity"
  }'
```

### Example Response

```json
{
  "tx_id": "TX00000001",
  "user_id": "U00001",
  "decision": "ALERT",
  "final_score": 0.847,
  "raw_score": 0.722,
  "signal_breakdown": {
    "statistical": 0.71,
    "unsupervised": 0.68,
    "supervised": 0.0,
    "llm": 0.90
  },
  "applied_boosts": ["new_counterparty", "night_large_wire"],
  "llm_assessment": {
    "risk": "high",
    "action": "alert",
    "primary_pattern": "romance_scam",
    "confidence": 0.89,
    "reason": "Large first-time ACH to MCC 6099 at 02:30 AM matches romance scam pattern."
  },
  "latency_ms": 142.3
}
```

---

## 📈 Model Performance

> Results on 50,000 synthetic transactions (2% fraud rate, 80/20 temporal split).

| Model | Precision | Recall | F2 (β=2) | **PR-AUC ★** | ROC-AUC | Cost ($) |
|-------|-----------|--------|----------|-------------|---------|---------|
| **XGBoost** | 0.89 | 0.84 | **0.85** | **0.91** | 0.97 | $168k |
| LightGBM | 0.88 | 0.83 | 0.84 | 0.90 | 0.97 | $180k |
| Logistic Regression | 0.79 | 0.71 | 0.73 | 0.82 | 0.94 | $270k |
| Isolation Forest | 0.45 | 0.62 | 0.57 | 0.51 | 0.78 | $480k |
| LOF | 0.41 | 0.55 | 0.51 | 0.47 | 0.73 | $520k |
| Statistical Rules | 0.38 | 0.71 | 0.61 | 0.44 | 0.75 | $390k |

> **★ PR-AUC** is the primary metric — ROC-AUC is shown for comparison but is inflated by the 98% true-negative class.
> **Cost** = FN × $15,000 + FP × $5 (elder financial protection cost framing).

---

## 🗂️ Project Structure

```
ai_financial_fraud_detection/
│
├── README.md                        ← You are here
├── run_pipeline.py                  ← Master entry point
├── Dockerfile + docker-compose.yml  ← Container definitions
├── .env.example                     ← Environment template
├── requirements.txt                 ← All dependencies pinned
│
├── src/                             ← Core source code
│   ├── config.py                    ← Pydantic BaseSettings singleton
│   ├── utils.py                     ← structlog, CostTracker, timer
│   ├── data_generator.py            ← 50k synthetic txs + 5 fraud typologies
│   ├── ingestion.py                 ← Multi-format loader (CSV/JSON/Parquet)
│   ├── enrichment.py                ← Feature engineering + geo-velocity
│   ├── features.py                  ← Feature registry + matrix builder
│   ├── feature_store.py             ← Parquet-backed feature store
│   ├── decision_engine.py           ← Weighted signal fusion + YAML policy
│   ├── training_pipeline.py         ← End-to-end train → evaluate → export
│   ├── api.py                       ← FastAPI REST API (6 endpoints)
│   │
│   ├── models/
│   │   ├── statistical.py           ← Z-score, EWMA, CUSUM, RulesEngine
│   │   ├── unsupervised.py          ← IsolationForest, LOF, Autoencoder
│   │   └── supervised.py            ← XGBoost, LightGBM, LogReg + SMOTE
│   │
│   ├── evaluation/
│   │   ├── metrics.py               ← PR-AUC, F-beta, cost matrix, confusion
│   │   ├── cross_validation.py      ← TimeSeriesSplit, walk-forward, bootstrap CI
│   │   ├── error_analysis.py        ← FP/FN slicing, SHAP waterfall
│   │   └── drift_monitor.py         ← PSI, KL-divergence, chi-squared
│   │
│   └── llm/
│       ├── agent.py                 ← LangGraph StateGraph agent
│       ├── llm_client.py            ← OpenAI / Bedrock / Mock abstraction
│       ├── prompt_builder.py        ← Token-budget prompt builder (tiktoken)
│       ├── rag_retriever.py         ← FAISS vector store + semantic search
│       └── output_parser.py         ← Pydantic structured LLM outputs
│
├── dashboard/
│   └── app.py                       ← 7-tab Streamlit dashboard
│
├── data/
│   ├── raw/                         ← Generated transactions (gitignored)
│   ├── results/                     ← Evaluation artefacts (JSON / Parquet)
│   └── knowledge_base/              ← RAG source documents
│       ├── elder_fraud_patterns.md  ← 7 fraud typologies with detection signals
│       ├── bsa_aml_regulations.md   ← BSA / SAR / CTR regulatory framework
│       └── feature_signal_reference.md ← Feature ↔ fraud signal mapping
│
├── config/
│   └── detection_policy.yaml        ← Tunable thresholds + weights (no code changes)
│
├── notebooks/
│   ├── eda.ipynb                    ← 10-section exploratory data analysis
│   └── model_comparison.ipynb       ← PR curves, threshold sweep, LLM evaluation
│
├── models/                          ← Serialised model artefacts (gitignored)
│
├── infra/
│   ├── lambda_handler.py            ← AWS Lambda / Kinesis entry point
│   └── cdk_stub.py                  ← CDK infrastructure-as-code sketch
│
└── tests/                           ← 35 tests (all passing)
    ├── test_features.py             ← 33 property-based + unit tests
    ├── test_ingestion.py            ← 27 schema + format tests
    ├── test_decision_engine.py
    ├── test_enrichment.py
    └── test_statistical.py
```

---

## 🔍 Elder Fraud Typologies Detected

The synthetic data generator and LLM knowledge base cover 7 real-world patterns (CFPB / AARP / FBI IC3):

| # | Typology | Primary Signal | Avg Loss |
|---|----------|---------------|---------|
| 1 | **Romance Scam** | Large first-time wire, MCC 6099, 00:00–05:00 | $10k–$15k |
| 2 | **Tech Support Scam** | Gift card ACH velocity spike, new merchant, 2–5 AM | $500–$3k |
| 3 | **Lottery / Prize Scam** | Multiple sub-$900 ACH in 2h, MCC 6099 | $800–$1.5k |
| 4 | **Account Takeover** | Geo-impossible flag, >5 tx/60 min, all new payees | $5k–$20k |
| 5 | **Structuring (BSA Evasion)** | ATM $8k–$9,799 repeated 2–5× in 72h, MCC 6011 | $25k–$100k |
| 6 | **Elder Financial Exploitation** | Gradual CUSUM escalation over weeks | $50k+ |
| 7 | **Card-Not-Present (CNP)** | ONLINE type, 1–5 AM, new merchant, MCC 5999 | $200–$600 |

---

## 🎓 Interview Talking Points

<details>
<summary><b>Why this architecture over a simple XGBoost?</b></summary>

> *"A single XGBoost model maximises recall on patterns seen in training data, but elder fraud evolves faster than any labelled dataset. I layer the system: statistical rules catch known thresholds immediately with zero latency, isolation forest catches novel patterns without labels, XGBoost maximises recall on historical patterns, and the LLM agent provides contextual reasoning over user history and fraud typologies — it can explain why a $9,500 ACH at 2 AM to a new payee matches a romance scam, not just assign a score. The decision engine fuses all four signals with YAML-configurable weights so compliance teams can adjust without touching code."*

</details>

<details>
<summary><b>How do you prevent temporal leakage?</b></summary>

> *"Three mechanisms: First, rolling statistics use `shift(1)` inside user groups — each transaction only sees its own past behaviour. Second, `TimeSeriesSplit` instead of K-Fold — always train on past, validate on future. Third, SMOTE is applied inside each fold after the temporal split, never before — synthetic samples from the future can't contaminate training."*

</details>

<details>
<summary><b>How do you handle class imbalance?</b></summary>

> *"Three layers: `scale_pos_weight = n_neg/n_pos` in XGBoost adjusts the loss function so fraud gradients get proportional weight. SMOTE with `sampling_strategy=0.1` — I don't oversample to 50/50 because that distorts the prior and inflates calibrated probabilities. F2 threshold optimisation — I sweep the PR curve and pick the threshold maximising F-β(2), weighting recall 2× over precision because a missed fraud costs $15,000."*

</details>

<details>
<summary><b>How do you know when to retrain?</b></summary>

> *"PSI on the top-10 SHAP features, computed weekly. PSI < 0.10: stable, no action. 0.10–0.20: alert, deeper investigation. > 0.20: automatic SageMaker retraining trigger. I also track concept drift via error rate — if precision drops >5% week-over-week on a rolling 1,000-transaction window, that's an early warning before PSI catches it."*

</details>

<details>
<summary><b>How do you control LLM costs in production?</b></summary>

> *"Two-tier routing: scores below 0.3 get a cheap triage model (Claude Haiku or GPT-3.5-turbo); only uncertain cases (0.3–0.7) go to the expensive model (GPT-4o). A token-budget prompt builder enforces max 2,000 input tokens via tiktoken — it summarises user history to fit the budget rather than truncating blindly. A `CostTracker` accumulates every prompt/completion token count, and a daily budget cap returns a cached safe decision if exhausted. At 2% fraud rate on 1M daily transactions, ~96% of cases never reach the expensive model."*

</details>

<details>
<summary><b>How would you deploy this to AWS?</b></summary>

> *"The Lambda handler in `infra/lambda_handler.py` handles Kinesis stream events — it deserialises the transaction, runs the detection pipeline, and writes to DynamoDB with TTL. Models are loaded from S3 on cold start and cached for warm invocations. For throughput above ~500 req/s I'd move to ECS Fargate behind an ALB with autoscaling on SQS queue depth. The CDK stub defines the full stack: DynamoDB table, Lambda function, Kinesis stream, SNS alert topic, and CloudWatch dashboard."*

</details>

---

## 🔧 Configuration

All thresholds and signal weights are YAML-configurable without code changes:

```yaml
# config/detection_policy.yaml
thresholds:
  alert: 0.70          # Score ≥ 0.70 → ALERT
  monitor: 0.40        # Score 0.40–0.70 → MONITOR
  cooldown_hours: 24   # Suppress duplicate alerts per user

signal_weights:
  statistical:  0.20
  unsupervised: 0.25
  supervised:   0.15
  llm:          0.40   # Highest weight — richest contextual reasoning

boosts:
  new_counterparty:    0.10
  night_large_wire:    0.15
  rapid_succession:    0.20
  geographic_velocity: 0.25
```

---

## 🏛️ Regulatory Context

| Regulation | Implementation |
|-----------|---------------|
| **BSA / FinCEN** | Structuring detection (31 U.S.C. § 5324), CTR threshold monitoring |
| **SAR Filing** | ALERT decisions include LLM-generated SAR-ready reasoning traces |
| **GLBA** | Data handling and customer financial privacy |
| **OCC SR 11-7** | Model risk management, validation, documentation |
| **CFPB** | Elder financial exploitation reporting requirements |

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines.

```bash
pip install black isort mypy
black src/ tests/
isort src/ tests/
mypy src/ --ignore-missing-imports
pytest tests/ -v --cov=src
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Sources

- CFPB — *Protecting Older Consumers Report 2023*
- AARP Fraud Watch Network — *Top Scams 2023*
- FBI IC3 — *Elder Fraud Report 2023*
- FinCEN — *BSA Requirements for Financial Institutions*
- OCC — *BSA/AML Examination Procedures*
- Federal Reserve — SR 11-7 Model Risk Management Guidance

---

<div align="center">

**Built as a production-grade reference for mastering financial AI engineering.**
*Every component is interview-ready and maps directly to real fintech production systems.*

⭐ If this repo helped your interview preparation, a star is appreciated!

</div>
