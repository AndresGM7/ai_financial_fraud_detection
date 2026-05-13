"""
models/supervised.py
────────────────────
Supervised classifiers — used when labeled data is available.

Models
──────
  1. Logistic Regression (calibrated) — interpretable baseline, fast
  2. XGBoost                          — gradient boosting, handles imbalance
  3. LightGBM                         — faster XGBoost alternative, better
                                        on large sparse feature sets
  4. SMOTE oversampling               — address severe class imbalance
  5. Calibrated probabilities         — isotonic regression to make
                                        predict_proba meaningful

Interview talking points
────────────────────────
  "Financial fraud datasets are severely imbalanced — typically 0.1–2%
   positive rate. Two common pitfalls: (1) naively maximising accuracy
   (a model that always predicts 0 gets 98% accuracy but is useless);
   (2) using ROC-AUC as the primary metric for imbalanced data. I prefer
   PR-AUC (area under precision-recall curve) because it focuses on the
   minority class performance which is what we actually care about.

  SMOTE generates synthetic minority samples in feature space (not
  oversampling raw rows) to balance training data without duplicating
  information. I apply it ONLY to training folds — never to validation,
  to avoid optimistic estimates.

  Calibration with isotonic regression ensures predict_proba() outputs
  are true probabilities, not just relative scores — critical for the
  decision engine's weighted threshold logic."

  XGBoost hyperparameters:
    scale_pos_weight = n_negative / n_positive  — built-in imbalance handling
    max_depth = 6, learning_rate = 0.05        — regularised to prevent overfit
    subsample / colsample_bytree = 0.8         — feature bagging
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from src.config import settings
from src.utils import get_logger, timed

log = get_logger(__name__)


# ── Helper: class weight ratio ────────────────────────────────────────────────

def _pos_weight(y: np.ndarray) -> float:
    """scale_pos_weight for XGBoost = n_neg / n_pos."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float(n_neg / max(n_pos, 1))


# ── 1. Logistic Regression ────────────────────────────────────────────────────

class LogisticDetector:
    """
    L2-regularised logistic regression with StandardScaler and calibration.
    Acts as an interpretable baseline — coefficients are directly explainable.
    """

    def __init__(self, C: float = 1.0) -> None:
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C,
                class_weight="balanced",
                max_iter=1000,
                random_state=settings.random_seed,
            )),
        ])
        self.calibrated: CalibratedClassifierCV | None = None
        self.is_fitted = False

    @timed
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticDetector":
        X_res, y_res = _apply_smote(X, y)
        self.calibrated = CalibratedClassifierCV(
            self.pipeline, cv=3, method="isotonic"
        )
        self.calibrated.fit(X_res, y_res)
        self.is_fitted = True
        train_pr_auc = average_precision_score(y, self.predict_proba(X))
        log.info("logreg_fitted", train_pr_auc=round(train_pr_auc, 4))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrated.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def coef_summary(self, feature_names: list[str]) -> pd.DataFrame:
        """Return sorted coefficient table for interpretability."""
        lr = self.calibrated.estimator.named_steps["clf"]
        scaler = self.calibrated.estimator.named_steps["scaler"]
        coefs = lr.coef_[0]
        return (
            pd.DataFrame({"feature": feature_names, "coefficient": coefs})
            .sort_values("coefficient", key=abs, ascending=False)
        )

    def save(self, path: Path) -> None:
        joblib.dump(self.calibrated, path)

    @classmethod
    def load(cls, path: Path) -> "LogisticDetector":
        det = cls()
        det.calibrated = joblib.load(path)
        det.is_fitted = True
        return det


# ── 2. XGBoost ────────────────────────────────────────────────────────────────

class XGBoostDetector:
    """
    XGBoost classifier with SMOTE, SHAP explanations, and MLflow logging.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.params: dict[str, Any] = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "use_label_encoder": False,
            "eval_metric": "aucpr",    # AP on train set
            "random_state": settings.random_seed,
            "n_jobs": -1,
            **kwargs,
        }
        self.model: XGBClassifier | None = None
        self.explainer: shap.TreeExplainer | None = None
        self.feature_names: list[str] = []
        self.is_fitted = False

    @timed
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        eval_set: tuple | None = None,
    ) -> "XGBoostDetector":
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        X_res, y_res = _apply_smote(X, y)

        self.params["scale_pos_weight"] = _pos_weight(y_res)
        self.model = XGBClassifier(**self.params)

        fit_kwargs: dict = {}
        if eval_set:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["verbose"] = False

        self.model.fit(X_res, y_res, **fit_kwargs)
        self.explainer = shap.TreeExplainer(self.model)
        self.is_fitted = True

        pr_auc = average_precision_score(y, self.predict_proba(X))
        log.info("xgboost_fitted", pr_auc=round(pr_auc, 4), n_estimators=self.params["n_estimators"])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for interpretability / error analysis."""
        return self.explainer.shap_values(X)

    def feature_importance_df(self) -> pd.DataFrame:
        return (
            pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
        )

    def save(self, path: Path) -> None:
        joblib.dump({
            "model": self.model,
            "explainer": self.explainer,
            "feature_names": self.feature_names,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "XGBoostDetector":
        det = cls()
        data = joblib.load(path)
        det.model = data["model"]
        det.explainer = data["explainer"]
        det.feature_names = data["feature_names"]
        det.is_fitted = True
        return det


# ── 3. LightGBM ───────────────────────────────────────────────────────────────

class LGBMDetector:
    """
    LightGBM — faster than XGBoost on sparse/high-cardinality features,
    better support for categorical columns natively.
    """

    def __init__(self) -> None:
        if not HAS_LGBM:
            raise ImportError("lightgbm not installed")
        self.model: LGBMClassifier | None = None
        self.is_fitted = False

    @timed
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMDetector":
        X_res, y_res = _apply_smote(X, y)
        self.model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            random_state=settings.random_seed,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(X_res, y_res)
        self.is_fitted = True
        pr_auc = average_precision_score(y, self.predict_proba(X))
        log.info("lgbm_fitted", pr_auc=round(pr_auc, 4))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)


# ── SMOTE utility ─────────────────────────────────────────────────────────────

def _apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE only when minority class is under-represented.
    sampling_strategy=0.1 means minority:majority = 1:10 after resampling.

    Interview talking point:
      "I don't oversample to 50/50 — that's unrealistic and inflates
       recall at the expense of precision. I target a moderate ratio that
       gives the model enough signal without distorting the prior."
    """
    n_minority = y.sum()
    if n_minority < 10:
        log.warning("smote_skipped", reason="too_few_minority_samples", n=int(n_minority))
        return X, y

    min_neighbors = min(5, int(n_minority) - 1)
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=min_neighbors,
        random_state=settings.random_seed,
    )
    X_res, y_res = smote.fit_resample(X, y)
    log.info("smote_applied", before=len(y), after=len(y_res), minority_after=int(y_res.sum()))
    return X_res, y_res

