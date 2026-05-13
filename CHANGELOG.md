# Changelog

All notable changes to this project are documented in this file.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) + [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-05-12

### 🎉 Initial Release

**Complete production-grade financial fraud detection system for elder financial protection.**

#### Added — Data Layer
- Synthetic data generator: 50,000 transactions across 5 elder-fraud typologies (romance scam, tech support, lottery, ATO, structuring, EFE, CNP)
- Multi-format ingestion pipeline: CSV, JSON, Parquet, webhook events with Pydantic schema validation and dead-letter queue
- Feature store backed by Parquet with shift(1) look-ahead leakage prevention
- 50+ engineered features: Z-score (log-transform), EWMA, CUSUM, velocity windows (1h/24h/7d), geo-velocity (Haversine), cyclical time encoding, MCC risk classification

#### Added — Models
- **Supervised:** XGBoost (eval_metric=aucpr), LightGBM, Logistic Regression with SMOTE (strategy=0.1) and CalibratedClassifierCV
- **Unsupervised:** Isolation Forest (200 trees, contamination=0.02), LOF (novelty=True, minkowski), PyTorch Autoencoder
- **Statistical:** Z-score rule engine (5 rules), EWMA baseline, CUSUM accumulator

#### Added — Evaluation
- PR-AUC as primary metric with full justification for imbalanced classification
- F-β(2) threshold optimisation sweeping the full PR curve
- Cost-matrix evaluation: FN × $15,000 + FP × $5
- TimeSeriesSplit cross-validation (5 folds) with bootstrap confidence intervals
- PSI drift monitoring with KL-divergence and chi-squared for categorical features
- Error analysis: FP/FN slicing by fraud typology, transaction type, hour band, amount decile

#### Added — LLM / RAG
- LangGraph-style StateGraph agent: retrieve → assess → investigate
- FAISS vector store over 3 knowledge-base documents (elder fraud patterns, BSA/AML regulations, feature signal reference)
- OpenAI GPT-4o, Anthropic Bedrock, and Mock provider abstraction
- Structured outputs via `instructor` with Pydantic response models
- Token-budget prompt builder with tiktoken (max 2,000 input tokens)
- Per-call CostTracker with daily budget enforcement ($50 default)

#### Added — API & Serving
- FastAPI async REST API with 6 endpoints: `/analyze`, `/batch`, `/health`, `/metrics/cost`, `/metrics/drift`, `/feedback`
- Pydantic request/response validation with OpenAPI docs at `/docs`
- Structured JSON logging via structlog (CloudWatch-compatible)
- Per-request latency headers (X-Latency-MS, X-Request-ID)
- BackgroundTasks for async batch processing (202 Accepted pattern)

#### Added — Dashboard
- 7-tab Streamlit dashboard: Overview, Model Comparison, PR/ROC Curves, Confusion Matrices, Error Analysis, Drift Monitor, LLM Agent
- Interactive Plotly charts with best-metric highlighting
- Threshold sweep slider for best model
- LLM cost/latency analytics with per-call log

#### Added — Infrastructure
- Docker + docker-compose (FastAPI + MLflow)
- AWS Lambda handler for Kinesis stream events
- CDK stub: DynamoDB, Lambda, Kinesis, SNS, CloudWatch
- MLflow experiment tracking with per-run params and metrics
- YAML-configurable detection policy (no code changes needed for threshold tuning)

#### Added — Testing
- 35 tests across 5 test files
- Property-based tests with Hypothesis
- pytest-cov, pytest-asyncio support

---

## [Unreleased]

### Planned
- GitHub Actions CI/CD pipeline (test + lint + docker build)
- Graph neural network layer for counterparty network analysis
- Calibration curves in dashboard
- Streaming real-time updates with Streamlit `st.rerun`
- Terraform alternative to CDK stub

