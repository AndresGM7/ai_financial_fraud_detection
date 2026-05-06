"""
Unsupervised anomaly detection: Isolation Forest and Local Outlier Factor.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.config import IF_N_ESTIMATORS, IF_CONTAMINATION


def train_isolation_forest(
    X: np.ndarray,
    n_estimators: int = IF_N_ESTIMATORS,
    contamination: float = IF_CONTAMINATION,
) -> IsolationForest:
    """Train an Isolation Forest model."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def predict_iforest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Return binary anomaly flags (1 = anomaly) from Isolation Forest."""
    return (model.predict(X) == -1).astype(int)


def fit_predict_lof(
    X: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = IF_CONTAMINATION,
) -> np.ndarray:
    """Fit and predict Local Outlier Factor (transductive only)."""
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1,
    )
    preds = lof.fit_predict(X)
    return (preds == -1).astype(int)


def build_scaler(X: np.ndarray) -> StandardScaler:
    """Fit a standard scaler on X."""
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler
