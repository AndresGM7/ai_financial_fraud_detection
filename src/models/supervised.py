"""
Supervised detection models: Logistic Regression and XGBoost.
Used when labeled data is available.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGBOOST_AVAILABLE = False


def train_logreg(
    X: np.ndarray, y: np.ndarray, max_iter: int = 1000
) -> LogisticRegression:
    """Train a logistic regression classifier."""
    model = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=42)
    model.fit(X, y)
    return model


def predict_logreg(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Return binary predictions from logistic regression."""
    return model.predict(X).astype(int)


def predict_proba_logreg(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Return fraud probability from logistic regression."""
    return model.predict_proba(X)[:, 1]


def train_xgboost(X: np.ndarray, y: np.ndarray) -> "XGBClassifier":
    """Train an XGBoost classifier (requires xgboost package)."""
    if not _XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    scale_pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_xgboost(model: "XGBClassifier", X: np.ndarray) -> np.ndarray:
    """Return binary predictions from XGBoost."""
    return model.predict(X).astype(int)


def predict_proba_xgboost(model: "XGBClassifier", X: np.ndarray) -> np.ndarray:
    """Return fraud probability from XGBoost."""
    return model.predict_proba(X)[:, 1]
