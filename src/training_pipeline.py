"""
src/training_pipeline.py
────────────────────────
End-to-end training, evaluation, and results exporter.

Runs every model layer, computes all metrics, and writes
dashboard-ready artefacts to data/results/.

Output files
────────────
  data/results/dataset_stats.json      ← EDA overview
  data/results/model_results.json      ← per-model metrics + curves
  data/results/pr_curve_data.parquet   ← precision/recall/threshold tuples
  data/results/error_analysis.json     ← FP/FN sliced by dimension
  data/results/feature_importance.json ← XGBoost top-20 features
  data/results/drift_report.json       ← PSI + KL-divergence per feature
  data/results/llm_results.json        ← LLM cost, latency, decisions
  models/                              ← serialised model artefacts (.joblib / .pt)

Run:
    python -m src.training_pipeline
    # or via:  python run_pipeline.py

Interview talking point:
  "I separate training from serving. The pipeline writes evaluation
   artefacts to disk; the dashboard reads them. Training can run
   nightly in a SageMaker Processing Job without touching the live API."
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.config import settings
from src.enrichment import enrich_transactions
from src.evaluation.cross_validation import bootstrap_ci
from src.evaluation.drift_monitor import drift_report
from src.evaluation.error_analysis import full_error_analysis
from src.evaluation.metrics import evaluate, optimal_threshold, pr_curve_data
from src.features import ALL_FEATURES, build_features, get_feature_matrix, get_labels
from src.ingestion import DataIngestionPipeline
from src.llm.agent import FraudInvestigationAgent
from src.llm.rag_retriever import KnowledgeBaseRetriever, UserHistoryRetriever
from src.models.statistical import statistical_score
from src.models.supervised import LGBMDetector, LogisticDetector, XGBoostDetector
from src.models.unsupervised import IForestDetector, LOFDetector
from src.utils import cost_tracker, get_logger, timer

log = get_logger(__name__)

RESULTS_DIR = Path("data/results")
MODELS_DIR  = Path("models")


def _ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(name: str, data: object) -> None:
    (RESULTS_DIR / name).write_text(json.dumps(data, indent=2, default=str))


def _strip_non_serialisable(res: dict) -> dict:
    """Remove keys holding live Python objects before JSON serialisation."""
    return {k: v for k, v in res.items() if k not in ("detector", "scaler")}


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Load & prepare
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(labeled_path: str = "data/raw/labeled_transactions.csv") -> pd.DataFrame:
    log.info("pipeline_step", step="load_and_prepare")
    pipe = DataIngestionPipeline()
    df, result = pipe.load(labeled_path)
    log.info("loaded", rows=len(df), rejection_rate=result.rejection_rate)

    with timer("enrichment"):
        df_enriched = enrich_transactions(df)

    with timer("feature_engineering"):
        df_featured = build_features(df_enriched)

    return df_featured


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Dataset statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_dataset_stats(df: pd.DataFrame) -> dict:
    n_fraud = int(df["is_fraud"].sum()) if "is_fraud" in df.columns else 0
    n_total = len(df)

    fraud_by_type: dict = {}
    if "fraud_type" in df.columns:
        raw = (
            df["fraud_type"]
            .fillna("legitimate")
            .value_counts()
            .to_dict()
        )
        fraud_by_type = {str(k): int(v) for k, v in raw.items()}

    return {
        "n_rows":          n_total,
        "n_users":         int(df["user_id"].nunique()),
        "n_fraud":         n_fraud,
        "fraud_rate":      round(n_fraud / max(n_total, 1), 6),
        "date_range": {
            "start": str(df["timestamp"].min()),
            "end":   str(df["timestamp"].max()),
        },
        "fraud_by_type":   fraud_by_type,
        "tx_by_type":      {str(k): int(v) for k, v in df["tx_type"].value_counts().items()},
        "amount_stats": {
            "mean":   round(float(df["amount"].mean()), 2),
            "median": round(float(df["amount"].median()), 2),
            "std":    round(float(df["amount"].std()), 2),
            "p95":    round(float(df["amount"].quantile(0.95)), 2),
            "max":    round(float(df["amount"].max()), 2),
        },
        "hour_distribution": (
            {str(k): int(v) for k, v in df["hour"].value_counts().sort_index().items()}
            if "hour" in df.columns else {}
        ),
        "features_used": ALL_FEATURES,
        "n_features":    len(ALL_FEATURES),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Supervised models
# ─────────────────────────────────────────────────────────────────────────────

def train_supervised(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, np.ndarray, list[str]]:
    """
    Train Logistic Regression, XGBoost, LightGBM.
    Uses time-ordered 80/20 split — no random shuffle (temporal integrity).
    """
    X_raw, _         = get_feature_matrix(df, scale=False)
    X_scaled, scaler = get_feature_matrix(df, scale=True)
    y                = get_labels(df)

    if y is None:
        log.warning("no_labels_skipping_supervised")
        return {}, df, np.array([]), []

    feature_names = [c for c in ALL_FEATURES if c in df.columns]
    split_idx     = int(len(X_raw) * 0.8)

    X_tr,  X_te  = X_raw[:split_idx],    X_raw[split_idx:]
    Xs_tr, Xs_te = X_scaled[:split_idx], X_scaled[split_idx:]
    y_tr,  y_te  = y[:split_idx],        y[split_idx:]
    df_test       = df.iloc[split_idx:].copy().reset_index(drop=True)

    results: dict[str, dict] = {}

    # ── MLflow (best-effort — skip if server not running) ────────────────────
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        use_mlflow = True
    except Exception:
        use_mlflow = False

    def _log_mlflow(name: str, params: dict, metrics: dict):
        if not use_mlflow:
            return
        try:
            import mlflow
            with mlflow.start_run(run_name=name):
                mlflow.log_params(params)
                mlflow.log_metrics(
                    {k: float(v) for k, v in metrics.items()
                     if isinstance(v, (int, float)) and v is not None}
                )
        except Exception as exc:
            log.warning("mlflow_log_failed", error=str(exc))

    # ── Logistic Regression ──────────────────────────────────────────────────
    log.info("training_model", name="LogisticRegression")
    lr = LogisticDetector()
    lr.fit(Xs_tr, y_tr)
    y_sc_lr  = lr.predict_proba(Xs_te)
    opt_t_lr, opt_lr = optimal_threshold(y_te, y_sc_lr)
    y_pd_lr  = (y_sc_lr >= opt_t_lr).astype(int)
    m_lr     = evaluate(y_te, y_pd_lr, y_sc_lr)
    ci_lr    = bootstrap_ci(y_te, y_sc_lr,
                            lambda a, b: float(np.mean(b[a == 1])) - float(np.mean(b[a == 0])))
    fpr_lr, tpr_lr, _ = roc_curve(y_te, y_sc_lr)

    results["Logistic Regression"] = {
        "metrics":           m_lr,
        "optimal_threshold": opt_lr,
        "pr_curve":          pr_curve_data(y_te, y_sc_lr, "Logistic Regression").to_dict(orient="records"),
        "roc_curve":         {"fpr": fpr_lr.tolist(), "tpr": tpr_lr.tolist()},
        "bootstrap_ci":      ci_lr,
        "feature_importance": None,
        "y_test":            y_te.tolist(),
        "y_score":           y_sc_lr.tolist(),
        "y_pred":            y_pd_lr.tolist(),
    }
    lr.save(MODELS_DIR / "logreg.joblib")
    _log_mlflow("logistic_regression", {"C": 1.0}, m_lr)
    log.info("model_done", name="LogisticRegression", pr_auc=m_lr.get("pr_auc"))

    # ── XGBoost ──────────────────────────────────────────────────────────────
    log.info("training_model", name="XGBoost")
    xgb = XGBoostDetector()
    xgb.fit(X_tr, y_tr, feature_names=feature_names, eval_set=(X_te, y_te))
    y_sc_xgb = xgb.predict_proba(X_te)
    opt_t_xgb, opt_xgb = optimal_threshold(y_te, y_sc_xgb)
    y_pd_xgb = (y_sc_xgb >= opt_t_xgb).astype(int)
    m_xgb    = evaluate(y_te, y_pd_xgb, y_sc_xgb)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_te, y_sc_xgb)

    results["XGBoost"] = {
        "metrics":           m_xgb,
        "optimal_threshold": opt_xgb,
        "pr_curve":          pr_curve_data(y_te, y_sc_xgb, "XGBoost").to_dict(orient="records"),
        "roc_curve":         {"fpr": fpr_xgb.tolist(), "tpr": tpr_xgb.tolist()},
        "bootstrap_ci":      {},
        "feature_importance": xgb.feature_importance_df().head(20).to_dict(orient="records"),
        "y_test":            y_te.tolist(),
        "y_score":           y_sc_xgb.tolist(),
        "y_pred":            y_pd_xgb.tolist(),
    }
    xgb.save(MODELS_DIR / "xgboost.joblib")
    _log_mlflow("xgboost", {"n_estimators": 500, "max_depth": 6}, m_xgb)
    log.info("model_done", name="XGBoost", pr_auc=m_xgb.get("pr_auc"))

    # ── LightGBM ─────────────────────────────────────────────────────────────
    try:
        log.info("training_model", name="LightGBM")
        lgbm = LGBMDetector()
        lgbm.fit(X_tr, y_tr)
        y_sc_lgbm = lgbm.predict_proba(X_te)
        opt_t_lgbm, opt_lgbm = optimal_threshold(y_te, y_sc_lgbm)
        y_pd_lgbm = (y_sc_lgbm >= opt_t_lgbm).astype(int)
        m_lgbm    = evaluate(y_te, y_pd_lgbm, y_sc_lgbm)
        fpr_lgbm, tpr_lgbm, _ = roc_curve(y_te, y_sc_lgbm)

        results["LightGBM"] = {
            "metrics":           m_lgbm,
            "optimal_threshold": opt_lgbm,
            "pr_curve":          pr_curve_data(y_te, y_sc_lgbm, "LightGBM").to_dict(orient="records"),
            "roc_curve":         {"fpr": fpr_lgbm.tolist(), "tpr": tpr_lgbm.tolist()},
            "bootstrap_ci":      {},
            "feature_importance": None,
            "y_test":            y_te.tolist(),
            "y_score":           y_sc_lgbm.tolist(),
            "y_pred":            y_pd_lgbm.tolist(),
        }
        lgbm.save(MODELS_DIR / "lgbm.joblib")
        _log_mlflow("lightgbm", {"n_estimators": 500, "num_leaves": 63}, m_lgbm)
        log.info("model_done", name="LightGBM", pr_auc=m_lgbm.get("pr_auc"))
    except ImportError:
        log.warning("lightgbm_not_installed_skipping")

    return results, df_test, y_te, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# 4 · Unsupervised models
# ─────────────────────────────────────────────────────────────────────────────

def train_unsupervised(df: pd.DataFrame) -> dict:
    X_raw, _    = get_feature_matrix(df, scale=False)
    X_scaled, _ = get_feature_matrix(df, scale=True)
    y           = get_labels(df)

    split_idx  = int(len(X_raw) * 0.8)
    X_tr_r     = X_raw[:split_idx]
    X_te_r     = X_raw[split_idx:]
    X_tr_s     = X_scaled[:split_idx]
    X_te_s     = X_scaled[split_idx:]
    y_te       = y[split_idx:] if y is not None else np.zeros(len(X_te_r))
    df_test    = df.iloc[split_idx:].copy().reset_index(drop=True)

    results: dict[str, dict] = {}

    # ── Isolation Forest ─────────────────────────────────────────────────────
    log.info("training_model", name="IsolationForest")
    iforest = IForestDetector()
    iforest.fit(X_tr_r)
    sc_if  = iforest.predict_scores(X_te_r)
    fl_if  = iforest.predict_flags(X_te_r)
    m_if   = evaluate(y_te, fl_if, sc_if)
    fpr_if, tpr_if, _ = roc_curve(y_te, sc_if)

    results["Isolation Forest"] = {
        "metrics":           m_if,
        "optimal_threshold": {},
        "pr_curve":          pr_curve_data(y_te, sc_if, "Isolation Forest").to_dict(orient="records"),
        "roc_curve":         {"fpr": fpr_if.tolist(), "tpr": tpr_if.tolist()},
        "bootstrap_ci":      {},
        "feature_importance": None,
        "y_test":            y_te.tolist(),
        "y_score":           sc_if.tolist(),
        "y_pred":            fl_if.tolist(),
    }
    iforest.save(MODELS_DIR / "iforest.joblib")
    log.info("model_done", name="IsolationForest", pr_auc=m_if.get("pr_auc"))

    # ── LOF ──────────────────────────────────────────────────────────────────
    log.info("training_model", name="LOF")
    lof = LOFDetector()
    lof.fit(X_tr_s)
    sc_lof = lof.predict_scores(X_te_s)
    fl_lof = (sc_lof > 0.5).astype(int)
    m_lof  = evaluate(y_te, fl_lof, sc_lof)
    fpr_lof, tpr_lof, _ = roc_curve(y_te, sc_lof)

    results["LOF"] = {
        "metrics":           m_lof,
        "optimal_threshold": {},
        "pr_curve":          pr_curve_data(y_te, sc_lof, "LOF").to_dict(orient="records"),
        "roc_curve":         {"fpr": fpr_lof.tolist(), "tpr": tpr_lof.tolist()},
        "bootstrap_ci":      {},
        "feature_importance": None,
        "y_test":            y_te.tolist(),
        "y_score":           sc_lof.tolist(),
        "y_pred":            fl_lof.tolist(),
    }
    lof.save(MODELS_DIR / "lof.joblib")
    log.info("model_done", name="LOF", pr_auc=m_lof.get("pr_auc"))

    # ── Statistical baseline ──────────────────────────────────────────────────
    log.info("computing_baseline", name="StatisticalRules")
    sc_stat = statistical_score(df_test).values
    fl_stat = (sc_stat > 0.5).astype(int)
    m_stat  = evaluate(y_te, fl_stat, sc_stat)
    fpr_st, tpr_st, _ = roc_curve(y_te, sc_stat)

    results["Statistical Rules"] = {
        "metrics":           m_stat,
        "optimal_threshold": {},
        "pr_curve":          pr_curve_data(y_te, sc_stat, "Statistical Rules").to_dict(orient="records"),
        "roc_curve":         {"fpr": fpr_st.tolist(), "tpr": tpr_st.tolist()},
        "bootstrap_ci":      {},
        "feature_importance": None,
        "y_test":            y_te.tolist(),
        "y_score":           sc_stat.tolist(),
        "y_pred":            fl_stat.tolist(),
    }
    log.info("model_done", name="StatisticalRules", pr_auc=m_stat.get("pr_auc"))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5 · LLM evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_llm(df: pd.DataFrame, n_samples: int = 30) -> dict:
    """
    Run the LLM agent on a stratified sample of test transactions.
    Collects accuracy, cost, latency, and per-decision details.
    Uses mock LLM provider by default (no API key required).
    """
    log.info("llm_evaluation_start", n=n_samples)
    agent     = FraudInvestigationAgent()
    retriever = UserHistoryRetriever(KnowledgeBaseRetriever())

    split_idx = int(len(df) * 0.8)
    df_test   = df.iloc[split_idx:].copy().reset_index(drop=True)
    has_labels = "is_fraud" in df_test.columns

    # Stratified sample
    if has_labels and df_test["is_fraud"].sum() > 0:
        n_fraud = min(n_samples // 2, int(df_test["is_fraud"].sum()))
        n_legit = min(n_samples - n_fraud, int((df_test["is_fraud"] == 0).sum()))
        idx_f   = df_test[df_test["is_fraud"] == 1].sample(n_fraud, random_state=42).index
        idx_l   = df_test[df_test["is_fraud"] == 0].sample(n_legit, random_state=42).index
        sample  = df_test.loc[list(idx_f) + list(idx_l)].reset_index(drop=True)
    else:
        sample = df_test.sample(min(n_samples, len(df_test)), random_state=42)

    decisions, latencies, llm_preds, true_labels = [], [], [], []
    risk_map = {"low": 0, "medium": 0, "high": 1}

    for _, row in sample.iterrows():
        row_dict = row.to_dict()
        t0 = time.perf_counter()

        decision = agent.run(
            current_tx=row_dict,
            statistical_signals={
                "z_amount":            row_dict.get("z_amount", 0),
                "ewma_z_amount":       row_dict.get("ewma_z_amount", 0),
                "tx_count_1h":         row_dict.get("tx_count_1h", 0),
                "tx_count_24h":        row_dict.get("tx_count_24h", 0),
                "mcc_high_risk":       row_dict.get("mcc_high_risk", 0),
                "is_new_counterparty": row_dict.get("is_new_counterparty", 0),
                "geo_impossible":      row_dict.get("geo_impossible", 0),
                "statistical_score":   0.5,
                "unsupervised_score":  0.5,
            },
            history_df=df,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        pred       = risk_map.get(decision.assessment.risk.value, 0)
        true_label = int(row_dict.get("is_fraud", 0))

        llm_preds.append(pred)
        true_labels.append(true_label)
        latencies.append(round(latency_ms, 2))
        decisions.append({
            "tx_id":       str(row_dict.get("tx_id", "")),
            "risk":        decision.assessment.risk.value,
            "action":      decision.assessment.action.value,
            "pattern":     decision.assessment.primary_pattern,
            "reason":      decision.assessment.reason,
            "confidence":  decision.assessment.confidence,
            "llm_calls":   decision.llm_calls,
            "latency_ms":  round(latency_ms, 2),
            "true_label":  true_label,
        })

    y_true_arr = np.array(true_labels)
    y_pred_arr = np.array(llm_preds)
    llm_metrics = evaluate(y_true_arr, y_pred_arr, None)

    return {
        "n_evaluated":    len(decisions),
        "metrics":        llm_metrics,
        "cost_summary":   cost_tracker.summary(),
        "call_log":       cost_tracker.call_log[-len(decisions):],
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "decisions":      decisions,
        "latencies":      latencies,
        "provider":       settings.llm_provider,
        "model":          settings.openai_model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6 · Drift report
# ─────────────────────────────────────────────────────────────────────────────

def compute_drift(df: pd.DataFrame) -> list[dict]:
    """
    Compare first 40 % (baseline) vs last 40 % (current) of data.
    Produces PSI and KL-divergence per feature.
    """
    n        = len(df)
    baseline = df.iloc[: int(n * 0.4)]
    current  = df.iloc[int(n * 0.6):]

    num_cols = [c for c in
                ["amount", "z_amount", "ewma_z_amount",
                 "tx_count_24h", "amount_sum_24h", "geo_speed_kmh"]
                if c in df.columns]
    cat_cols = [c for c in ["tx_type", "merchant_clean"] if c in df.columns]

    report = drift_report(baseline, current, num_cols, cat_cols)
    return report.to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────────────
# 7 · Error analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_error_analysis(df: pd.DataFrame, y_pred: np.ndarray) -> dict[str, list]:
    split_idx = int(len(df) * 0.8)
    df_test   = df.iloc[split_idx:].copy().reset_index(drop=True)
    slices    = full_error_analysis(df_test, y_pred)
    return {k: v.to_dict(orient="records") for k, v in slices.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Master run
# ─────────────────────────────────────────────────────────────────────────────

def run(
    data_path:   str = "data/raw/labeled_transactions.csv",
    llm_samples: int = 30,
) -> None:
    _ensure_dirs()
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  FRAUD DETECTION TRAINING PIPELINE")
    print(SEP)

    # ── Step 1: Prepare data ──────────────────────────────────────────────────
    print("\n[1/7] Loading & preparing data…")
    df = load_and_prepare(data_path)

    stats = compute_dataset_stats(df)
    _save_json("dataset_stats.json", stats)
    print(f"      ✓ {stats['n_rows']:,} transactions | fraud rate {stats['fraud_rate']:.2%}")

    # ── Step 2: Supervised models ─────────────────────────────────────────────
    print("\n[2/7] Training supervised models (LogReg / XGBoost / LightGBM)…")
    sup_results, df_test, y_te, feature_names = train_supervised(df)
    print(f"      ✓ Trained {len(sup_results)} supervised models")

    # ── Step 3: Unsupervised / Statistical ───────────────────────────────────
    print("\n[3/7] Training unsupervised models (IForest / LOF / Statistical)…")
    unsup_results = train_unsupervised(df)
    print(f"      ✓ Trained {len(unsup_results)} unsupervised/rule models")

    # ── Step 4: Merge & save model results ───────────────────────────────────
    print("\n[4/7] Saving model evaluation results…")
    all_results = {
        name: _strip_non_serialisable(res)
        for name, res in {**sup_results, **unsup_results}.items()
    }
    _save_json("model_results.json", all_results)

    # PR curve Parquet (all models combined)
    pr_rows = []
    for model_name, res in all_results.items():
        for row in res.get("pr_curve", []):
            row["model"] = model_name
            pr_rows.append(row)
    if pr_rows:
        pd.DataFrame(pr_rows).to_parquet(
            str(RESULTS_DIR / "pr_curve_data.parquet"), index=False
        )

    # Feature importance (XGBoost)
    fi = sup_results.get("XGBoost", {}).get("feature_importance") or []
    _save_json("feature_importance.json", fi)
    print(f"      ✓ Results saved for {len(all_results)} models")

    # ── Step 5: Error analysis ────────────────────────────────────────────────
    print("\n[5/7] Computing error analysis (FP/FN slices)…")
    best_preds = np.array(
        sup_results.get("XGBoost",
          sup_results.get("LightGBM",
          sup_results.get("Logistic Regression",
          unsup_results.get("Isolation Forest", {})
        ))).get("y_pred", [0] * max(len(y_te), 1))
    )
    error_data = compute_error_analysis(df, best_preds)
    _save_json("error_analysis.json", error_data)
    dims = list(error_data.keys())
    print(f"      ✓ Sliced across {len(dims)} dimensions: {', '.join(dims)}")

    # ── Step 6: LLM evaluation ────────────────────────────────────────────────
    print(f"\n[6/7] Evaluating LLM agent on {llm_samples} sample transactions…")
    llm_data = evaluate_llm(df, n_samples=llm_samples)
    _save_json("llm_results.json", llm_data)
    _save_json("llm_cost_log.json", cost_tracker.summary())
    print(f"      ✓ {llm_data['n_evaluated']} evaluated | "
          f"avg latency {llm_data['avg_latency_ms']:.0f} ms | "
          f"cost ${cost_tracker.summary()['total_cost_usd']:.4f}")

    # ── Step 7: Drift ──────────────────────────────────────────────────────────
    print("\n[7/7] Computing feature drift report (PSI)…")
    drift_data = compute_drift(df)
    _save_json("drift_report.json", drift_data)
    n_drifted = sum(1 for r in drift_data if r.get("significant_drift"))
    print(f"      ✓ {len(drift_data)} features | {n_drifted} with significant drift")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PIPELINE COMPLETE")
    print(SEP)
    print(f"\n  Results directory : {RESULTS_DIR.resolve()}/")
    print(f"  Models directory  : {MODELS_DIR.resolve()}/")
    print("\n  Model PR-AUC summary:")
    for name, res in sorted(
        all_results.items(),
        key=lambda x: x[1]["metrics"].get("pr_auc") or 0,
        reverse=True,
    ):
        pr = res["metrics"].get("pr_auc")
        rc = res["metrics"].get("recall")
        pr_s = f"{pr:.4f}" if pr is not None else " N/A  "
        rc_s = f"{rc:.4f}" if rc is not None else " N/A  "
        print(f"    {name:<25}  PR-AUC={pr_s}  Recall={rc_s}")

    print(f"\n  LLM cost  : ${cost_tracker.summary()['total_cost_usd']:.6f}")
    print("\n  Launch dashboard:")
    print("    streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    run()

