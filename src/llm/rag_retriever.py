"""
RAG Retriever: build user behavioral context from transaction history.
"""
import pandas as pd


def retrieve_context(user_id: str, df: pd.DataFrame) -> dict:
    """
    Retrieve a summary of the user's recent transaction history to use as
    retrieval-augmented context for the LLM prompt.

    Args:
        user_id: The user identifier.
        df: Enriched transaction DataFrame.

    Returns:
        A dictionary with behavioral statistics for the user.
    """
    user_df = df[df["user_id"] == user_id].tail(50)

    if user_df.empty:
        return {
            "avg_amount": 0.0,
            "std_amount": 0.0,
            "common_merchants": [],
            "night_activity": 0.0,
            "weekend_activity": 0.0,
            "tx_count": 0,
        }

    return {
        "avg_amount": round(float(user_df["amount"].mean()), 2),
        "std_amount": round(float(user_df["amount"].std(skipna=True) or 0.0), 2),
        "common_merchants": list(
            user_df["merchant_clean"].value_counts().head(3).index
        ),
        "night_activity": round(float(user_df["is_night"].mean()), 3),
        "weekend_activity": round(float(user_df.get("is_weekend", pd.Series([0])).mean()), 3),
        "tx_count": int(len(user_df)),
    }
