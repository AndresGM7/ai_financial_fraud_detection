"""
infra/cdk_stub.py
─────────────────
AWS CDK infrastructure-as-code stub.
Documents the target AWS architecture for the fraud detection system.

Architecture
────────────
  ┌─────────────────────────────────────────────────────────────────┐
  │  Bank Partner / Mobile App                                       │
  └───────────────────┬─────────────────────────────────────────────┘
                      │  HTTPS webhook
  ┌───────────────────▼─────────────────────────────────────────────┐
  │  API Gateway (REST)                                              │
  └───────────────────┬─────────────────────────────────────────────┘
                      │
  ┌───────────────────▼─────────────────────────────────────────────┐
  │  Lambda (fraud_scorer)                                           │
  │  - python3.11 runtime                                            │
  │  - 1024 MB memory (FAISS + PyTorch)                              │
  │  - 30s timeout                                                   │
  │  - Provisioned concurrency: 5 (eliminate cold starts)            │
  └───────┬──────────────┬──────────────────────┬───────────────────┘
          │              │                       │
  ┌───────▼──────┐  ┌────▼─────────┐  ┌─────────▼──────────────────┐
  │  DynamoDB    │  │  Bedrock     │  │  Redshift Serverless         │
  │  fraud_alerts│  │  Claude 3    │  │  (offline training data)     │
  │  (per-alert  │  │  (LLM calls) │  │                              │
  │   log)       │  │              │  │                              │
  └──────────────┘  └──────────────┘  └──────────────────────────────┘
          │
  ┌───────▼──────────────────────────────────────────────────────────┐
  │  SNS Topic: fraud-alerts                                          │
  │  Subscribers:                                                     │
  │    - SQS → Lambda (send email/push to family contact)            │
  │    - SQS → Lambda (update bank partner webhook)                  │
  └──────────────────────────────────────────────────────────────────┘

Interview talking point
───────────────────────
  "CDK lets me define cloud infrastructure in Python — the same language
   as the application. The entire stack (Lambda, DynamoDB, SNS, IAM roles)
   is version-controlled, reproducible, and deployable in one command:
   'cdk deploy'. This is critical for compliance: every infrastructure
   change has a code review and a Git history."
"""

from __future__ import annotations

# ── CDK stack (requires: pip install aws-cdk-lib constructs) ──────────────────
# Uncomment to activate when deploying to AWS.

# import aws_cdk as cdk
# from aws_cdk import (
#     Stack,
#     Duration,
#     aws_lambda as _lambda,
#     aws_apigateway as apigw,
#     aws_dynamodb as dynamodb,
#     aws_sns as sns,
#     aws_sns_subscriptions as subs,
#     aws_iam as iam,
# )
# from constructs import Construct
#
#
# class FraudDetectionStack(Stack):
#     def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
#         super().__init__(scope, construct_id, **kwargs)
#
#         # ── DynamoDB: fraud alert log ──────────────────────────────────────
#         alerts_table = dynamodb.Table(
#             self, "FraudAlertsTable",
#             table_name="fraud_alerts",
#             partition_key=dynamodb.Attribute(
#                 name="tx_id", type=dynamodb.AttributeType.STRING
#             ),
#             sort_key=dynamodb.Attribute(
#                 name="user_id", type=dynamodb.AttributeType.STRING
#             ),
#             billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
#             removal_policy=cdk.RemovalPolicy.RETAIN,
#         )
#
#         # ── SNS: alert fanout ─────────────────────────────────────────────
#         alert_topic = sns.Topic(self, "FraudAlertTopic", topic_name="fraud-alerts")
#
#         # ── Lambda: fraud scorer ──────────────────────────────────────────
#         scorer_fn = _lambda.Function(
#             self, "FraudScorerFn",
#             function_name="fraud-scorer",
#             runtime=_lambda.Runtime.PYTHON_3_11,
#             code=_lambda.Code.from_asset(".", bundling=cdk.BundlingOptions(
#                 image=_lambda.Runtime.PYTHON_3_11.bundling_image,
#                 command=["bash", "-c",
#                     "pip install -r requirements.txt -t /asset-output && cp -r src /asset-output"]
#             )),
#             handler="infra.lambda_handler.handler",
#             memory_size=1024,
#             timeout=Duration.seconds(30),
#             environment={
#                 "DYNAMODB_TABLE_NAME": alerts_table.table_name,
#                 "SNS_ALERT_TOPIC_ARN": alert_topic.topic_arn,
#                 "LLM_PROVIDER": "bedrock",
#             },
#         )
#
#         # Grant permissions
#         alerts_table.grant_read_write_data(scorer_fn)
#         alert_topic.grant_publish(scorer_fn)
#         scorer_fn.add_to_role_policy(iam.PolicyStatement(
#             actions=["bedrock:InvokeModel"],
#             resources=["*"],
#         ))
#
#         # ── API Gateway ───────────────────────────────────────────────────
#         api = apigw.RestApi(self, "FraudDetectionApi",
#             rest_api_name="fraud-detection",
#             description="Elder fraud detection API",
#         )
#         analyze_resource = api.root.add_resource("analyze")
#         analyze_resource.add_method(
#             "POST",
#             apigw.LambdaIntegration(scorer_fn),
#         )
#
#
# # ── Deploy ────────────────────────────────────────────────────────────────────
# # app = cdk.App()
# # FraudDetectionStack(app, "FraudDetectionStack",
# #     env=cdk.Environment(account="123456789012", region="us-east-1")
# # )
# # app.synth()


# ── Architecture summary (always active) ─────────────────────────────────────
ARCHITECTURE = {
    "compute": {
        "api": "AWS Lambda (python3.11, 1024 MB, 30s timeout)",
        "batch": "AWS Lambda (triggered by S3 PutObject event)",
        "training": "AWS SageMaker Processing Job (weekly retraining)",
    },
    "storage": {
        "alerts": "DynamoDB (fraud_alerts table, PAY_PER_REQUEST)",
        "features": "S3 (Parquet, partitioned by date/user)",
        "models": "S3 (versioned model artifacts)",
        "training_data": "Amazon Redshift Serverless",
    },
    "ml": {
        "llm": "AWS Bedrock (Claude 3 Sonnet / Haiku for cost tiering)",
        "embeddings": "AWS Bedrock Titan Embeddings (for RAG in prod)",
        "vector_store": "FAISS in-Lambda for small KB; OpenSearch for scale",
    },
    "messaging": {
        "alerts": "SNS → family contact email/push",
        "stream": "Kinesis Data Streams (real-time tx ingestion at scale)",
    },
    "observability": {
        "logs": "CloudWatch Logs (structlog JSON format)",
        "metrics": "CloudWatch Metrics (latency, cost, fraud rate)",
        "tracing": "AWS X-Ray (end-to-end request tracing)",
        "ml_tracking": "MLflow (experiment tracking, model registry)",
    },
}


if __name__ == "__main__":
    import json
    print(json.dumps(ARCHITECTURE, indent=2))

