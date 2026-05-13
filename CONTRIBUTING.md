# Contributing to AI Financial Fraud Detection System

Thank you for your interest in contributing! This document outlines the process for reporting issues, proposing enhancements, and submitting pull requests.

---

## 🗺️ Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)

---

## Code of Conduct

This project adheres to the standard open-source code of conduct. Be respectful, constructive, and collaborative. Harassment or discrimination of any kind will not be tolerated.

---

## Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/ai_financial_fraud_detection.git
cd ai_financial_fraud_detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

### 3. Install Dependencies (including dev tools)

```bash
pip install -r requirements.txt
pip install black isort mypy pylint
```

### 4. Set Up Environment

```bash
cp .env.example .env
# Edit .env as needed — mock LLM provider works out of the box
```

---

## Development Workflow

```bash
# 1. Create a feature branch
git checkout -b feature/your-feature-name

# 2. Make your changes

# 3. Format code
black src/ tests/
isort src/ tests/

# 4. Type check
mypy src/ --ignore-missing-imports

# 5. Run tests
pytest tests/ -v

# 6. Commit with a clear message
git commit -m "feat: add XYZ anomaly detector to unsupervised layer"

# 7. Push and open a PR
git push origin feature/your-feature-name
```

---

## Code Standards

### Style

- **Formatter:** [Black](https://black.readthedocs.io/) (line length 100)
- **Import order:** [isort](https://pycqa.github.io/isort/) with Black compatibility
- **Docstrings:** NumPy or Google style for public classes and functions
- **Type hints:** Required on all public function signatures

### Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Module | `snake_case` | `drift_monitor.py` |
| Class | `PascalCase` | `XGBoostDetector` |
| Function/Method | `snake_case` | `predict_proba` |
| Constant | `UPPER_SNAKE_CASE` | `ALL_FEATURES` |
| Private | `_leading_underscore` | `_score_transaction` |

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add SHAP waterfall plot to error analysis tab
fix: correct EWMA shift direction to prevent look-ahead leakage
docs: add interview talking point for PSI drift monitoring
test: add property-based tests for feature matrix builder
refactor: extract cost tracker into separate utils module
perf: vectorise geo-velocity calculation with NumPy
```

---

## Testing Requirements

All new code **must** include tests. Coverage thresholds:

| Module | Min Coverage |
|--------|-------------|
| `src/models/` | 80% |
| `src/evaluation/` | 80% |
| `src/features.py` | 90% |
| `src/decision_engine.py` | 85% |

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=75
```

### Test Types

- **Unit tests** — Pure function logic, edge cases, boundary values
- **Property-based tests** — Use `hypothesis` for statistical functions; never hardcode random seeds in tests that should be independent
- **Integration tests** — Full pipeline runs on small synthetic datasets

### Writing Tests

```python
# Good — specific, fast, no side effects
def test_zscore_flags_3_sigma_transaction():
    row = pd.Series({"amount": 10_000, "user_mean": 100, "user_std": 50})
    score = compute_zscore(row)
    assert score > 3.0

# Avoid — slow, depends on disk, environment-sensitive
def test_full_pipeline_runs():
    run("data/raw/labeled_transactions.csv")  # ❌ too broad
```

---

## Pull Request Process

1. **Open a Draft PR** early — to discuss approach before investing significant code
2. **Fill in the PR template** — describe what changed, why, and how to test it
3. **All CI checks must pass** — tests, lint, type checks
4. **At least one approval** required before merging
5. **Squash merge** to keep history clean

### PR Title Format

```
[feat] Add Autoencoder anomaly score to LLM prompt context
[fix] Prevent division-by-zero in PSI calculation for zero-variance features
[docs] Document F-beta threshold optimisation rationale
```

---

## Areas for Contribution

| Area | Ideas |
|------|-------|
| **New Models** | Graph neural network for counterparty network analysis |
| **Features** | Device fingerprint features, NLP on merchant names |
| **Evaluation** | Calibration curves, economic value of information |
| **LLM** | Multi-agent debate for uncertain transactions |
| **Dashboard** | Real-time streaming updates via Streamlit `st.rerun` |
| **Infrastructure** | Terraform alternative to CDK stub, GitHub Actions CI/CD |
| **Data** | Realistic transaction graph generation (NetworkX) |
| **Documentation** | Sequence diagrams for API request flow |

---

## Questions?

Open an issue with the `question` label, or start a GitHub Discussion.

