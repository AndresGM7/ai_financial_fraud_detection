"""
llm/rag_retriever.py
────────────────────
RAG (Retrieval-Augmented Generation) context builder.

Two retrieval channels
──────────────────────
  1. User history context  — summary statistics from user's last N transactions
                             (structured, no NLP embedding needed)
  2. Knowledge base        — FAISS vector index over fraud typology documents
                             (CFPB elder-fraud patterns, ACH/WIRE rules)

Interview talking points
────────────────────────
  "RAG solves a core limitation of pure LLMs: they have no access to
   real-time user behaviour or institution-specific fraud rules. By
   retrieving the user's recent transaction profile and relevant fraud
   typologies before calling the LLM, I give the model the context it
   needs to make a grounded decision — not a hallucinated one.

  I keep the knowledge base as plain text documents rather than a heavy
  vector database (Pinecone, Weaviate) because at our scale (hundreds of
  fraud pattern documents), FAISS in-process is faster and cheaper. We
  add a database only when we hit ~millions of documents.

  The retriever returns a structured context dict, not a raw string — this
  makes the prompt builder deterministic and the context size predictable."
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from src.config import settings
from src.utils import get_logger

log = get_logger(__name__)


# ── Fraud knowledge base documents ───────────────────────────────────────────
# In prod: load from S3 / DynamoDB. Here: embedded for portability.

FRAUD_TYPOLOGIES = [
    {
        "id": "romance_scam",
        "title": "Romance / Relationship Scam",
        "description": (
            "Fraudster builds emotional relationship with older adult online, "
            "then requests wire transfers or gift cards. Pattern: large first-time "
            "wire to an unknown foreign entity, typically $5,000–$50,000. "
            "Often preceded by WhatsApp/email contact. Frequent in ACH and WIRE channels."
        ),
        "red_flags": ["first_time_counterparty", "wire_transfer", "large_amount", "overseas"],
        "avg_loss_usd": 10000,
    },
    {
        "id": "tech_support_scam",
        "title": "Tech Support Scam",
        "description": (
            "Caller impersonates Microsoft/Apple, claims computer is infected, "
            "asks victim to purchase gift cards or wire funds. Pattern: gift card "
            "merchant purchases followed by rapid ACH. Average loss $500–$3,000."
        ),
        "red_flags": ["gift_card_merchant", "rapid_succession", "unusual_hour"],
        "avg_loss_usd": 1500,
    },
    {
        "id": "lottery_prize_scam",
        "title": "Lottery / Prize Scam",
        "description": (
            "Victim told they won a prize but must pay fees first via Western Union "
            "or MoneyGram. Pattern: multiple small ACH transfers to money-transfer MCCs "
            "(MCC 6099) within hours. Strong correlation with is_night and new_counterparty."
        ),
        "red_flags": ["money_transfer_mcc", "rapid_succession", "new_counterparty"],
        "avg_loss_usd": 800,
    },
    {
        "id": "account_takeover",
        "title": "Account Takeover (ATO)",
        "description": (
            "Credential theft via phishing, then fraudster logs in and initiates "
            "rapid transfers to money mules. Pattern: unusual login location, "
            ">5 Zelle/ACH transactions to new payees within 60 minutes, "
            "geo_impossible flag likely."
        ),
        "red_flags": ["geo_impossible", "rapid_succession", "zelle", "new_counterparty"],
        "avg_loss_usd": 5000,
    },
    {
        "id": "structuring",
        "title": "Structuring / Smurfing",
        "description": (
            "Multiple ATM withdrawals just below $10,000 BSA reporting threshold "
            "to avoid Suspicious Activity Report (SAR) filing. Pattern: 2–5 ATM "
            "withdrawals of $8,000–$9,999 within days. Can indicate elder financial "
            "exploitation by family member or caregiver."
        ),
        "red_flags": ["atm_withdrawal", "near_10k_threshold", "repeated_pattern"],
        "avg_loss_usd": 25000,
    },
    {
        "id": "card_not_present",
        "title": "Card-Not-Present (CNP) Fraud",
        "description": (
            "Stolen card credentials used for online purchases, often between 1–5 AM "
            "at merchants the cardholder has never used. Pattern: multiple ONLINE "
            "transactions at new merchants during night hours."
        ),
        "red_flags": ["online_transaction", "night_hours", "new_merchant", "small_amounts"],
        "avg_loss_usd": 300,
    },
    {
        "id": "elder_financial_exploitation",
        "title": "Elder Financial Exploitation (EFE)",
        "description": (
            "Family member, caregiver, or trusted person abuses access to older adult's "
            "accounts. Pattern: gradual escalation of transfer amounts to internal "
            "accounts, changes in beneficiary, unusual POA activity. CUSUM detects "
            "the slow drift better than Z-score."
        ),
        "red_flags": ["gradual_drift", "cusum_alert", "beneficiary_change"],
        "avg_loss_usd": 50000,
    },
]


# ── Simple TF-IDF encoder (no API needed for KB) ──────────────────────────────

class KnowledgeBaseRetriever:
    """
    FAISS-backed retriever for fraud typology documents.
    Uses simple bag-of-words TF-IDF embeddings (no external API required).
    Swap with OpenAI text-embedding-3-small for production quality.
    """

    def __init__(self, top_k: int = 2) -> None:
        self.top_k = top_k
        self.documents = FRAUD_TYPOLOGIES
        self._index: faiss.IndexFlatIP | None = None
        self._vectors: np.ndarray | None = None
        self._vocab: dict[str, int] = {}
        self._build_index()

    def _tokenise(self, text: str) -> list[str]:
        return text.lower().split()

    def _tfidf_vector(self, text: str) -> np.ndarray:
        """Simple TF-IDF representation over the knowledge base vocabulary."""
        tokens = self._tokenise(text)
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for tok in tokens:
            if tok in self._vocab:
                vec[self._vocab[tok]] += 1
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def _build_index(self) -> None:
        """Build vocabulary and FAISS index from knowledge base documents."""
        all_text = " ".join(
            d["title"] + " " + d["description"] for d in self.documents
        )
        tokens = set(self._tokenise(all_text))
        self._vocab = {tok: i for i, tok in enumerate(sorted(tokens))}

        vectors = np.stack([
            self._tfidf_vector(d["title"] + " " + d["description"])
            for d in self.documents
        ])
        self._vectors = vectors

        self._index = faiss.IndexFlatIP(vectors.shape[1])
        faiss.normalize_L2(vectors)
        self._index.add(vectors)
        log.info("kb_index_built", n_docs=len(self.documents), vocab_size=len(self._vocab))

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Return top-k most relevant fraud typology documents for a query."""
        q_vec = self._tfidf_vector(query).reshape(1, -1)
        faiss.normalize_L2(q_vec)
        scores, indices = self._index.search(q_vec, min(self.top_k, len(self.documents)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                doc = self.documents[idx].copy()
                doc["relevance_score"] = float(score)
                results.append(doc)
        return results


# ── User history context builder ──────────────────────────────────────────────

@dataclass
class UserContext:
    user_id: str
    avg_amount: float
    std_amount: float
    median_amount: float
    p95_amount: float
    common_merchants: list[str]
    common_tx_types: list[str]
    night_activity_ratio: float
    new_counterparty_rate: float
    avg_tx_count_per_day: float
    top_fraud_typologies: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "avg_amount": round(self.avg_amount, 2),
            "std_amount": round(self.std_amount, 2),
            "median_amount": round(self.median_amount, 2),
            "p95_amount": round(self.p95_amount, 2),
            "common_merchants": self.common_merchants,
            "common_tx_types": self.common_tx_types,
            "night_activity_ratio": round(self.night_activity_ratio, 3),
            "new_counterparty_rate": round(self.new_counterparty_rate, 3),
            "avg_tx_per_day": round(self.avg_tx_count_per_day, 2),
            "relevant_fraud_patterns": [
                {"id": t["id"], "title": t["title"]}
                for t in self.top_fraud_typologies
            ],
        }


class UserHistoryRetriever:
    """
    Builds a user behavioural context from recent transaction history.
    This is the 'R' in RAG — structured retrieval, not semantic search.
    """

    def __init__(self, kb_retriever: KnowledgeBaseRetriever | None = None) -> None:
        self.kb = kb_retriever or KnowledgeBaseRetriever()

    def retrieve(
        self,
        user_id: str,
        history_df: pd.DataFrame,
        current_tx: dict[str, Any],
        lookback_n: int = 50,
    ) -> UserContext:
        """
        Build UserContext from transaction history + current transaction signals.

        Parameters
        ----------
        user_id     : Account identifier
        history_df  : Full enriched DataFrame (all users)
        current_tx  : The transaction being evaluated (as dict)
        lookback_n  : How many recent transactions to use for the profile
        """
        user_df = history_df[history_df["user_id"] == user_id].tail(lookback_n)

        if len(user_df) == 0:
            # Cold start: no history — return defaults
            return UserContext(
                user_id=user_id,
                avg_amount=float(current_tx.get("amount", 0)),
                std_amount=0.0,
                median_amount=float(current_tx.get("amount", 0)),
                p95_amount=float(current_tx.get("amount", 0)),
                common_merchants=[],
                common_tx_types=[],
                night_activity_ratio=0.0,
                new_counterparty_rate=0.0,
                avg_tx_count_per_day=0.0,
            )

        amounts = user_df["amount"]

        # Build query for KB from current transaction signals
        query_parts = [str(current_tx.get("merchant", "")), str(current_tx.get("tx_type", ""))]
        if current_tx.get("is_new_counterparty"):
            query_parts.append("new counterparty")
        if current_tx.get("is_night"):
            query_parts.append("night transaction")
        if current_tx.get("geo_impossible"):
            query_parts.append("impossible travel account takeover")
        query = " ".join(query_parts)

        relevant_typologies = self.kb.retrieve(query)

        return UserContext(
            user_id=user_id,
            avg_amount=float(amounts.mean()),
            std_amount=float(amounts.std()),
            median_amount=float(amounts.median()),
            p95_amount=float(amounts.quantile(0.95)),
            common_merchants=list(user_df["merchant_clean"].value_counts().head(5).index),
            common_tx_types=list(user_df["tx_type"].value_counts().head(3).index),
            night_activity_ratio=float(user_df["is_night"].mean()) if "is_night" in user_df else 0.0,
            new_counterparty_rate=float(user_df["is_new_counterparty"].mean()) if "is_new_counterparty" in user_df else 0.0,
            avg_tx_count_per_day=float(len(user_df) / max((user_df["timestamp"].max() - user_df["timestamp"].min()).days, 1)),
            top_fraud_typologies=relevant_typologies,
        )

