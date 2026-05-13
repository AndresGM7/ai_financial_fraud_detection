"""
infra/lambda_handler.py
───────────────────────
AWS Lambda entry point — mirrors the API scoring logic but follows
the Lambda event/context signature expected by API Gateway.

Architecture
────────────
  API Gateway → Lambda (this handler) → [DynamoDB log, SNS alert]

Why Lambda?
───────────
  Serverless: auto-scales to thousands of concurrent fraud checks,
  costs ~$0 when idle, no server management. At Carefull's scale
  (bank partner integrations), each bank webhook triggers one Lambda
  invocation — perfectly event-driven.

Interview talking points
────────────────────────
  "I wrap the same scoring logic that runs in FastAPI inside a Lambda
   handler. The key difference: Lambda has a cold-start penalty (~500ms)
   on first invocation. I mitigate this by keeping models in /tmp (Lambda
   ephemeral storage) so subsequent calls in the same container reuse the
   loaded model — the 'container reuse' pattern.

  For latency-sensitive paths, I'd use Lambda with Provisioned Concurrency
  to eliminate cold starts — at ~$0.015/hour per provisioned instance.
  The cost vs. latency tradeoff is explicit and configurable."
"""

from __future__ import annotations

import json
import os
from typing import Any

# Lambda reuses containers — models cached at module level (/tmp or module scope)
_scoring_cache: dict = {}


def _get_pipeline():
    """Lazy-load pipeline — cached in Lambda container for warm invocations."""
    if "pipeline" not in _scoring_cache:
        from src.decision_engine import DecisionEngine
        from src.llm.agent import FraudInvestigationAgent
        _scoring_cache["engine"] = DecisionEngine()
        _scoring_cache["agent"] = FraudInvestigationAgent()
    return _scoring_cache["engine"], _scoring_cache["agent"]


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    AWS Lambda handler for API Gateway proxy integration.

    Expected event format (API Gateway REST):
    {
        "httpMethod": "POST",
        "body": "{...transaction JSON...}",
        "headers": {...}
    }

    Returns API Gateway-compatible response.
    """
    try:
        # Parse request body
        if isinstance(event.get("body"), str):
            tx_dict = json.loads(event["body"])
        elif isinstance(event.get("body"), dict):
            tx_dict = event["body"]
        else:
            # Direct Lambda invocation (not via API Gateway)
            tx_dict = event

        # Run scoring pipeline
        import pandas as pd
        from src.enrichment import enrich_transactions
        from src.features import build_features
        from src.models.statistical import statistical_score

        engine, agent = _get_pipeline()

        df = pd.DataFrame([tx_dict])
        df_enriched = enrich_transactions(df)
        df_featured = build_features(df_enriched)
        row = df_featured.iloc[0].to_dict()

        stat_score = float(statistical_score(df_featured).iloc[0])
        agent_decision = agent.run(
            current_tx=row,
            statistical_signals={"statistical_score": stat_score},
        )
        result = engine.decide(
            tx=row,
            statistical_score=stat_score,
            unsupervised_score=0.0,
            supervised_score=None,
            llm_assessment=agent_decision.assessment,
        )

        # Write alert to DynamoDB (production pattern)
        if result.decision == "alert":
            _write_dynamodb_alert(result.to_dict())

        # Publish SNS notification for "alert" decisions
        if result.decision == "alert":
            _publish_sns_alert(result.to_dict())

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result.to_dict()),
        }

    except Exception as exc:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc)}),
        }


def _write_dynamodb_alert(result: dict) -> None:
    """
    Persist fraud alert to DynamoDB.
    In production: enables querying by user_id, date, or decision type.
    """
    try:
        import boto3
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(os.environ.get("DYNAMODB_TABLE_NAME", "fraud_alerts"))
        table.put_item(Item={
            "tx_id": result["tx_id"],
            "user_id": result["user_id"],
            "timestamp": result["timestamp"],
            "decision": result["decision"],
            "final_score": str(result["final_score"]),
            "applied_boosts": json.dumps(result.get("applied_boosts", [])),
            "llm_reason": result.get("llm_assessment", {}).get("reason", ""),
        })
    except Exception as exc:
        # Non-fatal: log but don't fail the scoring response
        print(f"[WARN] DynamoDB write failed: {exc}")


def _publish_sns_alert(result: dict) -> None:
    """
    Publish high-priority fraud alert to SNS topic.
    Subscribers: bank partner webhook, internal Slack, email to family contact.
    """
    try:
        import boto3
        sns = boto3.client("sns")
        topic_arn = os.environ.get("SNS_ALERT_TOPIC_ARN", "")
        if not topic_arn:
            return
        message = (
            f"FRAUD ALERT\n"
            f"User: {result['user_id']}\n"
            f"Transaction: {result['tx_id']}\n"
            f"Score: {result['final_score']:.3f}\n"
            f"Reason: {result.get('llm_assessment', {}).get('reason', 'N/A')}"
        )
        sns.publish(
            TopicArn=topic_arn,
            Message=message,
            Subject=f"Fraud Alert — User {result['user_id']}",
        )
    except Exception as exc:
        print(f"[WARN] SNS publish failed: {exc}")

