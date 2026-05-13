"""
models/unsupervised.py
─────────────────────
Unsupervised anomaly detectors — no labels required.

Models
──────
  1. IsolationForest  — ensemble of random trees; anomalies are isolated
                        with fewer splits (fast, handles high-dimensional data)
  2. LocalOutlierFactor — density-based; compares local density of a point
                          to its k-nearest neighbours
  3. Autoencoder       — neural network trained to reconstruct normal
                         transactions; high reconstruction error → anomaly

Interview talking points
────────────────────────
  "I start with unsupervised methods because fraud labels are scarce and
   noisy — analysts label 'suspicious' not 'definitely fraudulent', and
   labels arrive days after the transaction (feedback delay problem).

  IsolationForest scales to millions of transactions with O(n log n)
  complexity. LOF is better for local density anomalies — e.g. a user
  who always buys at small coffee shops suddenly making a large wire.
  The Autoencoder catches complex non-linear patterns neither IF nor LOF
  would find.

  I calibrate all three to output a probability-like score in [0,1]
  using quantile scaling on the out-of-bag scores, then average them for
  the final 'unsupervised_score' signal."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

from src.config import settings
from src.utils import get_logger, timed

log = get_logger(__name__)

# ── 1. Isolation Forest ───────────────────────────────────────────────────────

class IForestDetector:
    """
    Isolation Forest wrapper with score normalisation.

    Key parameters
    ──────────────
    contamination : fraction of dataset expected to be anomalous.
                    Set to ~0.02 for financial fraud (2% is typical).
                    This controls the decision threshold but NOT the scores.
    n_estimators  : 200 trees → stable scores (diminishing returns after ~100)
    """

    def __init__(
        self,
        contamination: float | None = None,
        n_estimators: int = 200,
    ) -> None:
        self.contamination = contamination or settings.isolation_forest_contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=self.contamination,
            random_state=settings.random_seed,
            n_jobs=-1,
        )
        self._score_scaler = MinMaxScaler()
        self.is_fitted = False

    @timed
    def fit(self, X: np.ndarray) -> "IForestDetector":
        self.model.fit(X)
        raw_scores = -self.model.score_samples(X)  # negate: higher = more anomalous
        self._score_scaler.fit(raw_scores.reshape(-1, 1))
        self.is_fitted = True
        log.info("iforest_fitted", n_samples=len(X))
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1]. Higher = more anomalous."""
        raw = -self.model.score_samples(X)
        return self._score_scaler.transform(raw.reshape(-1, 1)).flatten()

    def predict_flags(self, X: np.ndarray) -> np.ndarray:
        """Binary flag: 1 = anomaly, 0 = normal."""
        return (self.model.predict(X) == -1).astype(int)

    def save(self, path: Path) -> None:
        joblib.dump({"model": self.model, "scaler": self._score_scaler}, path)
        log.info("iforest_saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "IForestDetector":
        detector = cls()
        data = joblib.load(path)
        detector.model = data["model"]
        detector._score_scaler = data["scaler"]
        detector.is_fitted = True
        return detector


# ── 2. Local Outlier Factor ───────────────────────────────────────────────────

class LOFDetector:
    """
    LOF wrapper. Note: LOF is transductive — predict() only works on training
    data unless novelty=True. We use novelty=True for production serving.

    Interview talking point:
      "LOF computes the ratio of a point's local reachability density to
       its neighbours'. A score > 1 means the point is in a lower-density
       region than its neighbours — i.e. it's more spread out, more unusual.
       Scores >> 1 indicate strong outliers."
    """

    def __init__(self, n_neighbors: int | None = None) -> None:
        self.n_neighbors = n_neighbors or settings.lof_n_neighbors
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            novelty=True,       # allows predict on new data
            contamination=settings.isolation_forest_contamination,
            n_jobs=-1,
        )
        self._score_scaler = MinMaxScaler()
        self.is_fitted = False

    @timed
    def fit(self, X: np.ndarray) -> "LOFDetector":
        self.model.fit(X)
        raw = -self.model.score_samples(X)
        self._score_scaler.fit(raw.reshape(-1, 1))
        self.is_fitted = True
        log.info("lof_fitted", n_samples=len(X))
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        raw = -self.model.score_samples(X)
        return self._score_scaler.transform(raw.reshape(-1, 1)).flatten()

    def save(self, path: Path) -> None:
        joblib.dump({"model": self.model, "scaler": self._score_scaler}, path)

    @classmethod
    def load(cls, path: Path) -> "LOFDetector":
        detector = cls()
        data = joblib.load(path)
        detector.model = data["model"]
        detector._score_scaler = data["scaler"]
        detector.is_fitted = True
        return detector


# ── 3. Autoencoder ────────────────────────────────────────────────────────────

class TransactionAutoencoder(nn.Module):
    """
    Shallow autoencoder for tabular transaction data.

    Architecture
    ────────────
    Encoder: input_dim → 64 → 32 → bottleneck (16)
    Decoder: 16 → 32 → 64 → input_dim

    Loss: MSE reconstruction error on NORMAL transactions.
    At inference: high reconstruction error → anomaly.

    Interview talking point:
      "The bottleneck forces the model to learn a compressed representation
       of normal behaviour. Anomalous transactions can't be well-reconstructed
       from that compressed space — hence the high error. This is the same
       principle behind anomaly detection in NLP with autoencoders, but
       applied to tabular financial features."
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=1)


class AutoencoderDetector:
    """Training + inference wrapper for TransactionAutoencoder."""

    def __init__(self, input_dim: int = 28, epochs: int = 30, lr: float = 1e-3) -> None:
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.model: TransactionAutoencoder | None = None
        self._threshold: float = 0.0
        self._score_scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @timed
    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        """
        Train on NORMAL transactions only (no labels needed).
        Threshold set at the 95th percentile of training reconstruction errors.
        """
        self.input_dim = X.shape[1]
        self.model = TransactionAutoencoder(self.input_dim).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        tensor = torch.FloatTensor(X).to(self.device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tensor),
            batch_size=256,
            shuffle=True,
        )

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimiser.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                log.info("autoencoder_epoch", epoch=epoch+1, loss=round(epoch_loss, 4))

        # Calibrate threshold and scaler on training data
        errors = self._get_errors(X)
        self._threshold = float(np.percentile(errors, 95))
        self._score_scaler.fit(errors.reshape(-1, 1))
        log.info("autoencoder_fitted", threshold=self._threshold)
        return self

    def _get_errors(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self.device)
            errors = self.model.reconstruction_error(t).cpu().numpy()
        return errors

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        errors = self._get_errors(X)
        return self._score_scaler.transform(errors.reshape(-1, 1)).flatten()

    def predict_flags(self, X: np.ndarray) -> np.ndarray:
        errors = self._get_errors(X)
        return (errors > self._threshold).astype(int)

    def save(self, path: Path) -> None:
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "threshold": self._threshold,
            "scaler": self._score_scaler,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "AutoencoderDetector":
        det = cls()
        data = torch.load(path, map_location="cpu")
        det.input_dim = data["input_dim"]
        det.model = TransactionAutoencoder(det.input_dim)
        det.model.load_state_dict(data["state_dict"])
        det._threshold = data["threshold"]
        det._score_scaler = data["scaler"]
        return det


# ── Ensemble ──────────────────────────────────────────────────────────────────

class UnsupervisedEnsemble:
    """
    Combines IForest + LOF + Autoencoder into a single unsupervised_score.
    Soft-voting: average of normalised scores.
    """

    def __init__(self) -> None:
        self.iforest = IForestDetector()
        self.lof = LOFDetector()
        self.autoencoder: AutoencoderDetector | None = None
        self.is_fitted = False

    @timed
    def fit(self, X: np.ndarray) -> "UnsupervisedEnsemble":
        self.iforest.fit(X)
        self.lof.fit(X)
        self.autoencoder = AutoencoderDetector(input_dim=X.shape[1])
        self.autoencoder.fit(X)
        self.is_fitted = True
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        s_if = self.iforest.predict_scores(X)
        s_lof = self.lof.predict_scores(X)
        s_ae = self.autoencoder.predict_scores(X) if self.autoencoder else np.zeros(len(X))
        return ((s_if + s_lof + s_ae) / 3).clip(0, 1)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self.iforest.save(model_dir / "iforest.joblib")
        self.lof.save(model_dir / "lof.joblib")
        if self.autoencoder:
            self.autoencoder.save(model_dir / "autoencoder.pt")

