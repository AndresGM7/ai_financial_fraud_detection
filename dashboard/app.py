"""
dashboard/app.py
────────────────
Streamlit dashboard — 7 tabs covering every metric a professional reviews.

Run:
    streamlit run dashboard/app.py

Prerequisites:
    python run_pipeline.py   # generates data/results/ files
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🛡️ Fraud Detection AI — Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RESULTS_DIR = Path("data/results")

# Colour palette: consistent across every chart
MODEL_PALETTE = {
    "Logistic Regression": "#2196F3",
    "XGBoost":             "#4CAF50",
    "LightGBM":            "#FF9800",
    "Isolation Forest":    "#9C27B0",
    "LOF":                 "#F44336",
    "Statistical Rules":   "#607D8B",
    "LLM Agent":           "#E91E63",
}

# ── CSS overrides ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main > div { padding-top: 0.5rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 6px 6px 0 0; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_json(name: str):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


@st.cache_data(ttl=60)
def load_raw_csv() -> pd.DataFrame:
    p = Path("data/raw/labeled_transactions.csv")
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["timestamp"])


def results_exist() -> bool:
    return (RESULTS_DIR / "model_results.json").exists()


# ── Reusable chart helpers ─────────────────────────────────────────────────────

def confusion_matrix_heatmap(
    cm: list[list[int]], title: str, colour: str = "#2196F3"
) -> go.Figure:
    labels = ["Legitimate", "Fraud"]
    z      = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
    annot  = [[f"TN={z[0][0]}", f"FP={z[0][1]}"],
               [f"FN={z[1][0]}", f"TP={z[1][1]}"]]
    fig = ff.create_annotated_heatmap(
        z=z,
        x=labels,
        y=labels,
        annotation_text=annot,
        colorscale=[[0, "#FFFFFF"], [1, colour]],
        showscale=True,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(side="bottom")
    return fig


def pr_chart(model_results: dict, fraud_rate: float) -> go.Figure:
    fig = go.Figure()
    for name, res in model_results.items():
        pr_data = res.get("pr_curve", [])
        if not pr_data:
            continue
        pr_df  = pd.DataFrame(pr_data)
        pr_auc = res["metrics"].get("pr_auc") or 0
        fig.add_trace(go.Scatter(
            x=pr_df["recall"], y=pr_df["precision"],
            name=f"{name} ({pr_auc:.3f})",
            mode="lines",
            line=dict(color=MODEL_PALETTE.get(name, "#aaa"), width=2.5),
        ))
    fig.add_hline(
        y=fraud_rate, line_dash="dash", line_color="gray",
        annotation_text=f"Random ({fraud_rate:.2%})",
    )
    fig.update_layout(
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision (PPV)",
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.8)"),
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


def roc_chart(model_results: dict) -> go.Figure:
    fig = go.Figure()
    for name, res in model_results.items():
        roc  = res.get("roc_curve", {})
        auc  = res["metrics"].get("roc_auc")
        if not roc or auc is None:
            continue
        label = f"{name} ({auc:.3f})"
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"],
            name=label,
            mode="lines",
            line=dict(color=MODEL_PALETTE.get(name, "#aaa"), width=2.5),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Random",
        mode="lines", line=dict(color="#ccc", dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall)",
        legend=dict(x=0.5, y=0.05, bgcolor="rgba(255,255,255,0.8)"),
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.title("🛡️ Financial Fraud Detection — AI System Dashboard")
st.caption(
    "Hybrid Statistical + Unsupervised + Supervised + LLM/RAG pipeline | "
    "Elder financial safety | Carefull-style architecture"
)

if not results_exist():
    st.error(
        "⚠️  No results found in `data/results/`.  \n"
        "Run the training pipeline first:  \n"
        "```\npython run_pipeline.py\n```"
    )
    st.stop()

# ── Load all result artefacts (cached) ────────────────────────────────────────
stats        = load_json("dataset_stats.json")    or {}
model_res    = load_json("model_results.json")    or {}
error_data   = load_json("error_analysis.json")   or {}
fi_data      = load_json("feature_importance.json") or []
drift_data   = load_json("drift_report.json")     or []
llm_data     = load_json("llm_results.json")      or {}
cost_log     = load_json("llm_cost_log.json")     or {}
raw_df       = load_raw_csv()

fraud_rate   = stats.get("fraud_rate", 0.02)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dataset",
    "🤖 Model Comparison",
    "📈 Confusion Matrices",
    "🔍 Error Analysis",
    "💰 LLM Metrics",
    "📉 Drift Monitor",
    "🎮 Live Demo",
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 · Dataset Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("📊 Dataset Overview")

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Transactions",  f"{stats.get('n_rows', 0):,}")
    c2.metric("Unique Users",  f"{stats.get('n_users', 0):,}")
    c3.metric("Fraud Cases",   f"{stats.get('n_fraud', 0):,}")
    c4.metric("Fraud Rate",    f"{stats.get('fraud_rate', 0):.2%}")
    c5.metric("Features",      f"{stats.get('n_features', 0)}")
    c6.metric("Model Formats", "CSV · Parquet · JSON")

    st.divider()

    # Row 1: fraud donut + tx-type bar
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fraud_types = stats.get("fraud_by_type", {})
        if fraud_types:
            labels = list(fraud_types.keys())
            vals   = list(fraud_types.values())
            # Colour the 'legitimate' slice grey
            colour_map = {
                "legitimate":           "#90CAF9",
                "account_takeover":     "#F44336",
                "romance_scam":         "#E91E63",
                "lottery_scam":         "#FF9800",
                "structuring":          "#9C27B0",
                "card_not_present":     "#607D8B",
                "tech_support_scam":    "#FF5722",
            }
            colours = [colour_map.get(l, "#BBDEFB") for l in labels]
            fig_ft = go.Figure(go.Pie(
                labels=labels, values=vals,
                hole=0.45,
                marker=dict(colors=colours),
                textinfo="label+percent",
            ))
            fig_ft.update_layout(
                title="Transaction Distribution by Fraud Typology",
                height=380, legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_ft, use_container_width=True)
            st.caption(
                "**Interview:** Elder fraud has 7 distinct typologies — each requires "
                "different detection signals. Romance scams need new-counterparty + large-wire "
                "detection; structuring needs CUSUM for gradual drift."
            )

    with r1c2:
        tx_types = stats.get("tx_by_type", {})
        if tx_types:
            fig_tx = px.bar(
                x=list(tx_types.keys()),
                y=list(tx_types.values()),
                title="Transaction Volume by Payment Rail",
                labels={"x": "Type", "y": "Count"},
                color=list(tx_types.keys()),
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_tx.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig_tx, use_container_width=True)
            st.caption(
                "**Interview:** ACH, Wire, and Zelle are highest-risk rails for elder fraud. "
                "POS/CNP fraud is lower-value but higher-frequency."
            )

    # Row 2: amount distribution + hour heatmap
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        if not raw_df.empty and "amount" in raw_df.columns:
            fig_amt = px.histogram(
                raw_df, x="amount",
                log_x=False, log_y=True,
                nbins=80,
                color="is_fraud" if "is_fraud" in raw_df.columns else None,
                color_discrete_map={0: "#2196F3", 1: "#F44336"},
                labels={"amount": "Amount (USD)", "is_fraud": "Fraud"},
                title="Amount Distribution (log Y scale)",
                barmode="overlay",
            )
            fig_amt.update_layout(height=350)
            st.plotly_chart(fig_amt, use_container_width=True)
            st.caption(
                "**Interview:** Transaction amounts are log-normally distributed (right-skewed). "
                "I log-transform before computing Z-scores for better calibration. "
                "Red bars show fraud transactions clustering at extreme amounts."
            )

    with r2c2:
        hour_dist = stats.get("hour_distribution", {})
        if hour_dist:
            hours  = list(range(24))
            counts = [hour_dist.get(str(h), hour_dist.get(h, 0)) for h in hours]
            fig_h  = go.Figure(go.Bar(
                x=hours, y=counts,
                marker=dict(
                    color=counts,
                    colorscale="Blues",
                    colorbar=dict(title="Count"),
                ),
            ))
            fig_h.add_vrect(x0=-0.5, x1=5.5, fillcolor="red",
                            opacity=0.08, line_width=0,
                            annotation_text="Night hours (risk zone)")
            fig_h.update_layout(
                title="Transaction Volume by Hour of Day",
                xaxis_title="Hour UTC (0 = Midnight)",
                yaxis_title="Transactions",
                height=350,
            )
            st.plotly_chart(fig_h, use_container_width=True)
            st.caption(
                "**Interview:** 00:00–05:59 is the 'night zone' — transactions here get "
                "is_night=1 flag and attract score boosts. CNP and ATO fraud spikes at 2–4 AM."
            )

    # Amount stats table
    amt_s = stats.get("amount_stats", {})
    if amt_s:
        st.subheader("Amount Statistics")
        st.dataframe(
            pd.DataFrame([amt_s]).rename(columns={
                "mean": "Mean ($)", "median": "Median ($)",
                "std": "Std Dev ($)", "p95": "P95 ($)", "max": "Max ($)",
            }),
            use_container_width=True, hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 · Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("🤖 Model Comparison")

    # ── Metrics table ──────────────────────────────────────────────────────────
    rows = []
    for name, res in model_res.items():
        m = res.get("metrics", {})
        opt = res.get("optimal_threshold", {})
        rows.append({
            "Model":         name,
            "Precision":     m.get("precision"),
            "Recall":        m.get("recall"),
            "F1":            m.get("f1"),
            "F2 (β=2)":      m.get("f2.0"),
            "PR-AUC ★":       m.get("pr_auc"),
            "ROC-AUC":       m.get("roc_auc"),
            "TP":            m.get("true_positives"),
            "FP":            m.get("false_positives"),
            "FN":            m.get("false_negatives"),
            "Cost ($)":      m.get("total_cost_usd"),
            "Opt Threshold": opt.get("optimal_threshold"),
        })

    metrics_df = pd.DataFrame(rows).set_index("Model")

    def highlight_best(col: pd.Series) -> list[str]:
        lower_better = col.name in ("FP", "FN", "Cost ($)")
        na_mask      = col.isna()
        if na_mask.all():
            return [""] * len(col)
        best = col.min() if lower_better else col.max()
        return [
            "background-color:#c8e6c9; font-weight:bold" if (not na_mask[i] and col.iloc[i] == best) else ""
            for i in range(len(col))
        ]

    fmt = {
        "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}",
        "F2 (β=2)": "{:.4f}", "PR-AUC ★": "{:.4f}", "ROC-AUC": "{:.4f}",
        "Cost ($)": "${:,.2f}", "Opt Threshold": "{:.3f}",
    }
    styled = (
        metrics_df.style
        .apply(highlight_best)
        .format(fmt, na_rep="N/A")
    )
    st.subheader("📋 All Models — Metrics Table (🟢 = best per column)")
    st.dataframe(styled, use_container_width=True, height=320)
    st.caption(
        "**Interview:** PR-AUC ★ is my primary metric for imbalanced fraud data. "
        "ROC-AUC is inflated by the large true-negative class at 2% fraud rate. "
        "F2 (β=2) weights recall 2× over precision — catching fraud matters more than avoiding false alerts."
    )

    st.divider()

    # ── PR vs ROC ─────────────────────────────────────────────────────────────
    col_pr, col_roc = st.columns(2)

    with col_pr:
        st.subheader("Precision-Recall Curves")
        st.caption("Primary evaluation — AUC score in legend")
        st.plotly_chart(pr_chart(model_res, fraud_rate), use_container_width=True)

    with col_roc:
        st.subheader("ROC Curves")
        st.caption("Reported for completeness — inflated for imbalanced data")
        st.plotly_chart(roc_chart(model_res), use_container_width=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.subheader("📡 Multi-Metric Radar (normalised to [0, 1])")
    radar_keys   = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    radar_labels = ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"]
    fig_radar    = go.Figure()

    for name, res in model_res.items():
        m    = res.get("metrics", {})
        vals = [float(m.get(k) or 0) for k in radar_keys]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=name,
            line_color=MODEL_PALETTE.get(name, "#aaa"),
            opacity=0.7,
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=450,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption(
        "**Interview:** Radar charts expose tradeoffs visually — a model strong on "
        "Recall but weak on Precision (high FP rate) might create alert fatigue. "
        "We want balanced models OR we tune the threshold post-training."
    )

    # ── Cost-weighted bar chart ────────────────────────────────────────────────
    st.subheader("💸 Business Cost per Model (FN cost=$15k, FP cost=$5)")
    cost_data = [
        {"Model": name, "Total Cost ($)": res["metrics"].get("total_cost_usd", 0) or 0}
        for name, res in model_res.items()
    ]
    cost_df = pd.DataFrame(cost_data).sort_values("Total Cost ($)")
    fig_cost = px.bar(
        cost_df, x="Total Cost ($)", y="Model", orientation="h",
        color="Total Cost ($)", color_continuous_scale="RdYlGn_r",
        title="Cost-Weighted Score (lower = better business outcome)",
    )
    fig_cost.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig_cost, use_container_width=True)
    st.caption(
        "**Interview:** A false negative on a $15k wire costs the elderly customer $15k. "
        "A false positive costs ~$5 in CS time. Cost-weighted score aligns ML optimisation "
        "with the actual financial impact — this is how you pitch model value to leadership."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 · Confusion Matrices
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("📈 Confusion Matrices & Threshold Analysis")

    model_names  = list(model_res.keys())
    cols_per_row = 3

    for chunk in [model_names[i:i+cols_per_row] for i in range(0, len(model_names), cols_per_row)]:
        cols = st.columns(len(chunk))
        for col, name in zip(cols, chunk):
            res = model_res[name]
            cm  = res["metrics"].get("confusion_matrix")
            if not cm:
                continue
            with col:
                st.plotly_chart(
                    confusion_matrix_heatmap(cm, name, MODEL_PALETTE.get(name, "#2196F3")),
                    use_container_width=True,
                )
                m = res["metrics"]
                st.dataframe(
                    pd.DataFrame([{
                        "FPR":  f"{m.get('false_positive_rate', 0):.3f}",
                        "FNR":  f"{m.get('false_negative_rate', 0):.3f}",
                        "F2":   f"{m.get('f2.0', 0) or 0:.3f}",
                        "Threshold": res.get("optimal_threshold", {}).get("optimal_threshold", "—"),
                    }]),
                    hide_index=True, use_container_width=True,
                )

    # ── Threshold sweep ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🎚️ Threshold Sweep — Precision / Recall / F2 Trade-off")
    sel = st.selectbox("Select model for threshold analysis", model_names, key="thresh_sel")

    res_sel = model_res[sel]
    y_t     = np.array(res_sel.get("y_test",  []))
    y_s     = np.array(res_sel.get("y_score", []))

    if len(y_t) > 1 and len(np.unique(y_t)) > 1:
        from sklearn.metrics import precision_recall_curve
        p_arr, r_arr, t_arr = precision_recall_curve(y_t, y_s)
        f2_arr = (5 * p_arr * r_arr) / np.maximum(4 * p_arr + r_arr, 1e-8)
        t_plot = np.append(t_arr, 1.0)

        fig_thresh = go.Figure()
        fig_thresh.add_trace(go.Scatter(x=t_plot, y=p_arr,   name="Precision",
                                        line=dict(color="#2196F3", width=2)))
        fig_thresh.add_trace(go.Scatter(x=t_plot, y=r_arr,   name="Recall",
                                        line=dict(color="#F44336", width=2)))
        fig_thresh.add_trace(go.Scatter(x=t_plot, y=f2_arr,  name="F2 (β=2)",
                                        line=dict(color="#4CAF50", width=2.5,
                                                  dash="dot")))

        opt_t = res_sel.get("optimal_threshold", {}).get("optimal_threshold")
        if opt_t:
            fig_thresh.add_vline(
                x=float(opt_t), line_dash="dash", line_color="#FF9800",
                annotation_text=f"Optimal = {opt_t:.3f}",
                annotation_position="top right",
            )

        fig_thresh.update_layout(
            xaxis_title="Decision Threshold",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1.05]),
            height=380,
            legend=dict(x=0.4, y=0.05),
        )
        st.plotly_chart(fig_thresh, use_container_width=True)
        st.caption(
            "**Interview:** The default 0.5 threshold is almost never optimal for "
            "imbalanced fraud. I sweep over all thresholds and pick the one that "
            "maximises F2 — which weights recall 2× over precision. "
            "In production this threshold is stored in `detection_policy.yaml` "
            "so compliance can adjust it without code changes."
        )

        # Distribution of scores overlaid
        st.subheader("Score Distribution — Fraud vs Legitimate")
        df_dist              = pd.DataFrame({"score": y_s, "label": y_t})
        df_dist["class"]     = df_dist["label"].map({0: "Legitimate", 1: "Fraud"})
        fig_dist = px.histogram(
            df_dist, x="score", color="class", nbins=60, barmode="overlay",
            opacity=0.7,
            color_discrete_map={"Legitimate": "#2196F3", "Fraud": "#F44336"},
            title=f"Score Distribution — {sel}",
            labels={"score": "Predicted Score", "class": "Class"},
        )
        if opt_t:
            fig_dist.add_vline(x=float(opt_t), line_dash="dot",
                               annotation_text="Optimal threshold",
                               line_color="#FF9800")
        fig_dist.update_layout(height=350)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption(
            "**Interview:** Good separation between the two distributions = high "
            "discriminative power. Overlapping distributions indicate the model "
            "struggles to distinguish fraud from legitimate — investigate feature "
            "importance to understand why."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 · Error Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("🔍 Error Analysis — FP / FN Slicing")

    if not error_data:
        st.warning("No error analysis data. Re-run `python run_pipeline.py`.")
    else:
        dim_labels = {
            "fraud_type":          "Fraud Typology",
            "tx_type":             "Transaction Type",
            "is_night":            "Night Transaction (0/1)",
            "is_weekend":          "Weekend (0/1)",
            "mcc_high_risk":       "High-Risk MCC (0/1)",
            "is_new_counterparty": "New Counterparty (0/1)",
            "amount_band":         "Amount Band",
        }
        dimension = st.selectbox(
            "Slice analysis by",
            options=list(error_data.keys()),
            format_func=lambda x: dim_labels.get(x, x.replace("_", " ").title()),
        )

        slice_df = pd.DataFrame(error_data[dimension])

        if not slice_df.empty:
            slice_col = slice_df.columns[0]

            ea1, ea2 = st.columns(2)

            with ea1:
                x_vals = slice_df[slice_col].astype(str)
                fig_pr_s = go.Figure()
                fig_pr_s.add_bar(x=x_vals, y=slice_df["precision"],
                                 name="Precision", marker_color="#2196F3")
                fig_pr_s.add_bar(x=x_vals, y=slice_df["recall"],
                                 name="Recall",    marker_color="#F44336")
                fig_pr_s.update_layout(
                    barmode="group",
                    title=f"Precision & Recall by {dim_labels.get(dimension, dimension)}",
                    yaxis=dict(range=[0, 1.05]),
                    height=380,
                )
                st.plotly_chart(fig_pr_s, use_container_width=True)

            with ea2:
                fig_err = go.Figure()
                fig_err.add_bar(x=x_vals, y=slice_df["fp"],
                                name="False Positives (alert fatigue)", marker_color="#FF9800")
                fig_err.add_bar(x=x_vals, y=slice_df["fn"],
                                name="False Negatives (missed fraud)",  marker_color="#9C27B0")
                fig_err.update_layout(
                    barmode="stack",
                    title="FP and FN Counts (stacked)",
                    height=380,
                )
                st.plotly_chart(fig_err, use_container_width=True)

            st.caption(
                "**Interview:** Slice analysis exposes failure modes that aggregate "
                "metrics hide. A model with PR-AUC=0.88 might miss 90% of romance scams "
                "while catching all CNP fraud. Without slicing you'd deploy a model that "
                "completely fails the highest-value fraud type."
            )
            st.dataframe(slice_df, use_container_width=True)

    # ── Feature importance ─────────────────────────────────────────────────────
    if fi_data:
        st.divider()
        st.subheader("🏆 XGBoost Feature Importance (Top 20 by Gain)")
        fi_df  = pd.DataFrame(fi_data).head(20)
        fi_fig = px.bar(
            fi_df, x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Greens",
            title="Feature Importance — XGBoost (Gain)",
        )
        fi_fig.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            height=500,
        )
        st.plotly_chart(fi_fig, use_container_width=True)
        st.caption(
            "**Interview:** Feature importance (Gain) shows which features most reduce "
            "training loss. But 'importance' doesn't equal 'direction' — a feature "
            "with high importance could push scores up OR down depending on its value. "
            "SHAP values go further: they show the contribution of each feature for "
            "each individual prediction, enabling per-transaction explainability "
            "required for compliance (GLBA / ECOA)."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 · LLM Metrics
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.header("💰 LLM Usage, Cost & Performance")

    if not llm_data:
        st.warning("No LLM results. Run `python run_pipeline.py` first.")
    else:
        daily_budget = 50.0
        total_cost   = cost_log.get("total_cost_usd", 0) or 0
        n_calls      = max(cost_log.get("total_calls", 1), 1)
        cost_per_tx  = total_cost / n_calls
        budget_pct   = min(total_cost / daily_budget * 100, 100)

        # ── KPI row ────────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("LLM Calls",    f"{cost_log.get('total_calls', 0):,}")
        k2.metric("Input Tokens", f"{cost_log.get('total_input_tokens', 0):,}")
        k3.metric("Output Tokens",f"{cost_log.get('total_output_tokens', 0):,}")
        k4.metric("Total Cost",   f"${total_cost:.4f}")
        k5.metric("Avg Latency",  f"{llm_data.get('avg_latency_ms', 0):.0f} ms")
        k6.metric("P95 Latency",  f"{llm_data.get('p95_latency_ms', 0):.0f} ms")

        st.divider()
        row_l1, row_l2 = st.columns(2)

        # ── Budget gauge ───────────────────────────────────────────────────────
        with row_l1:
            bar_colour = (
                "#4CAF50" if budget_pct < 50 else
                "#FF9800" if budget_pct < 80 else "#F44336"
            )
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(total_cost, 4),
                number={"prefix": "$", "valueformat": ".4f"},
                title={"text": "Daily LLM Spend (USD)", "font": {"size": 16}},
                delta={"reference": daily_budget * 0.5,
                       "prefix": "$", "valueformat": ".4f"},
                gauge={
                    "axis": {"range": [0, daily_budget]},
                    "bar":  {"color": bar_colour},
                    "steps": [
                        {"range": [0,                daily_budget * 0.5], "color": "#e8f5e9"},
                        {"range": [daily_budget*0.5, daily_budget*0.8],   "color": "#fff3e0"},
                        {"range": [daily_budget*0.8, daily_budget],       "color": "#ffebee"},
                    ],
                    "threshold": {
                        "line":      {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value":     daily_budget,
                    },
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            m1, m2 = st.columns(2)
            m1.metric("Cost / call",      f"${cost_per_tx:.6f}")
            m2.metric("Budget remaining", f"${max(daily_budget - total_cost, 0):.4f}")

            st.info(
                "**Cost optimisation (interview):**  \n"
                "Two-tier LLM: GPT-4o-mini ($0.00015/1k) for triage, "
                "GPT-4o ($0.005/1k) only for uncertain high-risk cases. "
                "Saves ~60% LLM spend with <2% recall loss.  \n"
                "AWS Bedrock Claude Haiku: $0.00025/1k — stays in VPC (GLBA)."
            )

        # ── Latency distribution ───────────────────────────────────────────────
        with row_l2:
            latencies = llm_data.get("latencies", [])
            if latencies:
                fig_lat = px.histogram(
                    x=latencies, nbins=20,
                    title="LLM Response Latency Distribution",
                    labels={"x": "Latency (ms)", "y": "Calls"},
                    color_discrete_sequence=["#2196F3"],
                )
                fig_lat.add_vline(
                    x=float(np.mean(latencies)), line_dash="dash",
                    annotation_text=f"Mean={np.mean(latencies):.0f}ms",
                    line_color="#F44336",
                )
                fig_lat.add_vline(
                    x=float(np.percentile(latencies, 95)), line_dash="dot",
                    annotation_text=f"P95={np.percentile(latencies, 95):.0f}ms",
                    line_color="#FF9800",
                )
                fig_lat.update_layout(height=300)
                st.plotly_chart(fig_lat, use_container_width=True)
                st.caption(
                    "**Interview:** P95 latency matters most — a slow tail affects "
                    "real-time alerting SLAs. Mock provider shows near-zero ms; "
                    "real GPT-4o averages 800–1200 ms P50."
                )

        # ── LLM risk + action distribution ────────────────────────────────────
        decisions = llm_data.get("decisions", [])
        if decisions:
            st.divider()
            st.subheader("LLM Assessment Distribution")
            dec_df = pd.DataFrame(decisions)

            d1, d2 = st.columns(2)
            with d1:
                risk_counts = dec_df["risk"].value_counts().reset_index()
                risk_counts.columns = ["risk", "count"]
                fig_risk = px.pie(
                    risk_counts, names="risk", values="count",
                    title="Risk Level Assignments",
                    color="risk",
                    color_discrete_map={
                        "low":    "#4CAF50",
                        "medium": "#FF9800",
                        "high":   "#F44336",
                    },
                    hole=0.35,
                )
                fig_risk.update_traces(textinfo="label+percent+value")
                st.plotly_chart(fig_risk, use_container_width=True)

            with d2:
                act_counts = dec_df["action"].value_counts().reset_index()
                act_counts.columns = ["action", "count"]
                fig_act = px.bar(
                    act_counts, x="action", y="count",
                    title="Recommended Actions",
                    color="action",
                    color_discrete_map={
                        "ignore":  "#4CAF50",
                        "monitor": "#FF9800",
                        "alert":   "#F44336",
                    },
                    labels={"action": "Action", "count": "Count"},
                )
                fig_act.update_layout(showlegend=False, height=360)
                st.plotly_chart(fig_act, use_container_width=True)

            # LLM accuracy metrics
            if "true_label" in dec_df.columns:
                st.subheader("LLM Agent — Accuracy vs Ground Truth")
                m = llm_data.get("metrics", {})

                cm = m.get("confusion_matrix")
                lm1, lm2 = st.columns(2)
                with lm1:
                    perf_rows = [
                        {"Metric": "Precision",       "Value": f"{m.get('precision', 0):.4f}"},
                        {"Metric": "Recall",          "Value": f"{m.get('recall', 0):.4f}"},
                        {"Metric": "F1",              "Value": f"{m.get('f1', 0):.4f}"},
                        {"Metric": "True Positives",  "Value": m.get("true_positives", 0)},
                        {"Metric": "False Positives", "Value": m.get("false_positives", 0)},
                        {"Metric": "False Negatives", "Value": m.get("false_negatives", 0)},
                        {"Metric": "True Negatives",  "Value": m.get("true_negatives", 0)},
                    ]
                    st.dataframe(pd.DataFrame(perf_rows), hide_index=True, use_container_width=True)
                    st.caption(
                        "**Interview:** The LLM agent's precision/recall uses 'high' risk → "
                        "predicted fraud. Medium risk maps to legitimate in binary evaluation — "
                        "in prod the decision engine uses continuous scores, not just the binary."
                    )
                with lm2:
                    if cm:
                        st.plotly_chart(
                            confusion_matrix_heatmap(cm, "LLM Agent", "#E91E63"),
                            use_container_width=True,
                        )

            # Decisions table
            st.subheader("Sample LLM Decisions (first 20)")
            show_cols = [c for c in
                         ["tx_id", "risk", "action", "confidence", "pattern",
                          "reason", "true_label", "latency_ms", "llm_calls"]
                         if c in dec_df.columns]
            st.dataframe(dec_df[show_cols].head(20), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 6 · Drift Monitor
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.header("📉 Feature Drift Monitor (PSI & KL-Divergence)")

    if not drift_data:
        st.warning("No drift data. Run `python run_pipeline.py`.")
    else:
        drift_df   = pd.DataFrame(drift_data)
        num_df     = drift_df[drift_df["type"] == "numerical"].dropna(subset=["psi"])

        n_stable   = int((num_df["psi"] < 0.10).sum())
        n_moderate = int(((num_df["psi"] >= 0.10) & (num_df["psi"] < 0.20)).sum())
        n_drifted  = int((num_df["psi"] >= 0.20).sum())

        # PSI KPI row
        s1, s2, s3 = st.columns(3)
        s1.metric("🟢 Stable",     f"{n_stable} features",   help="PSI < 0.10")
        s2.metric("🟡 Moderate",   f"{n_moderate} features", help="0.10 ≤ PSI < 0.20")
        s3.metric("🔴 Needs Retrain", f"{n_drifted} features", help="PSI ≥ 0.20")

        if not num_df.empty:
            sorted_num = num_df.sort_values("psi", ascending=False).reset_index(drop=True)

            def psi_colour(v: float) -> str:
                if   v < 0.10: return "#4CAF50"
                elif v < 0.20: return "#FF9800"
                else:          return "#F44336"

            colours = [psi_colour(float(v)) for v in sorted_num["psi"]]
            fig_psi = go.Figure(go.Bar(
                x=sorted_num["feature"],
                y=sorted_num["psi"].astype(float),
                marker_color=colours,
                text=[f"{v:.4f}" for v in sorted_num["psi"].astype(float)],
                textposition="outside",
            ))
            fig_psi.add_hline(y=0.10, line_dash="dash", line_color="#FF9800",
                              annotation_text="Monitor threshold (0.10)",
                              annotation_position="top right")
            fig_psi.add_hline(y=0.20, line_dash="dot",  line_color="#F44336",
                              annotation_text="Retrain threshold (0.20)",
                              annotation_position="top right")
            fig_psi.update_layout(
                title="Population Stability Index (PSI) per Feature",
                xaxis_title="Feature",
                yaxis_title="PSI",
                height=420,
            )
            st.plotly_chart(fig_psi, use_container_width=True)
            st.caption(
                "**Interview:** PSI is the industry-standard drift metric at banks. "
                "PSI < 0.10 = stable; 0.10–0.20 = monitor; > 0.20 = retrain. "
                "I compute it weekly on the top-10 SHAP features — monitoring all "
                "features is too noisy and expensive."
            )

        # KL divergence chart
        kl_df = drift_df.dropna(subset=["kl_divergence"]) if "kl_divergence" in drift_df.columns else pd.DataFrame()
        if not kl_df.empty:
            fig_kl = px.bar(
                kl_df.sort_values("kl_divergence", ascending=False),
                x="feature", y="kl_divergence",
                title="KL-Divergence per Feature (baseline → current)",
                color="significant_drift",
                color_discrete_map={True: "#F44336", False: "#4CAF50"},
                labels={"kl_divergence": "KL Divergence", "significant_drift": "Drifted"},
            )
            fig_kl.update_layout(height=350)
            st.plotly_chart(fig_kl, use_container_width=True)
            st.caption(
                "**Interview:** KL-divergence is asymmetric — KL(current||baseline). "
                "I prefer Jensen-Shannon divergence for symmetric comparison, but KL "
                "is more established in the MLOps tooling ecosystem."
            )

        # Full table
        st.subheader("Full Drift Report")
        show_cols = [c for c in
                     ["feature", "type", "psi", "psi_flag", "kl_divergence",
                      "significant_drift", "chi2_statistic", "p_value"]
                     if c in drift_df.columns]
        st.dataframe(
            drift_df[show_cols].style.apply(
                lambda col: [
                    "background-color:#ffebee" if (col.name == "significant_drift" and v) else ""
                    for v in col
                ]
            ),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 7 · Live Demo
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
    st.header("🎮 Live Demo — Full Pipeline on a Single Transaction")
    st.caption(
        "Runs enrichment → feature engineering → statistical rules → "
        "LLM agent → decision engine in real time. "
        "No API key required (mock LLM mode)."
    )

    # ── Preset buttons ─────────────────────────────────────────────────────────
    st.subheader("⚡ Quick presets")
    p1, p2, p3, p4 = st.columns(4)
    use_romance    = p1.button("💔 Romance Scam")
    use_normal     = p2.button("✅ Normal Purchase")
    use_ato        = p3.button("🔓 Account Takeover")
    use_structuring = p4.button("🏧 Structuring")

    presets = {
        "romance":     dict(user_id="U00001", amount=12000.0, merchant="wire_transfer",
                            tx_type="WIRE", mcc=6012, hour=2,
                            counterparty="Unknown Overseas Entity",
                            ts="2024-06-15T02:30:00+00:00"),
        "normal":      dict(user_id="U00002", amount=45.0, merchant="starbucks",
                            tx_type="POS",  mcc=5812, hour=10,
                            counterparty="Starbucks",
                            ts="2024-06-15T10:15:00+00:00"),
        "ato":         dict(user_id="U00003", amount=350.0, merchant="zelle_payee",
                            tx_type="ZELLE", mcc=9999, hour=3,
                            counterparty="MoneyMule123",
                            ts="2024-06-15T03:05:00+00:00"),
        "structuring": dict(user_id="U00004", amount=9200.0, merchant="atm_withdrawal",
                            tx_type="ATM",  mcc=6011, hour=14,
                            counterparty="CASH",
                            ts="2024-06-15T14:00:00+00:00"),
    }

    # Pick preset or keep defaults
    if use_romance:
        default = presets["romance"]
    elif use_ato:
        default = presets["ato"]
    elif use_structuring:
        default = presets["structuring"]
    elif use_normal:
        default = presets["normal"]
    else:
        default = presets["romance"]  # initial default

    # ── Input form ──────────────────────────────────────────────────────────────
    with st.form("tx_form"):
        f1, f2, f3 = st.columns(3)
        user_id      = f1.text_input("User ID",          value=default["user_id"])
        amount       = f2.number_input("Amount ($)",      value=float(default["amount"]), min_value=0.01, step=10.0)
        merchant     = f3.text_input("Merchant",          value=default["merchant"])

        f4, f5, f6 = st.columns(3)
        tx_type     = f4.selectbox("Transaction Type",
                                   ["WIRE", "ACH", "ZELLE", "POS", "ATM", "ONLINE"],
                                   index=["WIRE","ACH","ZELLE","POS","ATM","ONLINE"].index(default["tx_type"]))
        mcc         = f5.number_input("MCC Code",          value=int(default["mcc"]),  step=1)
        counterparty = f6.text_input("Counterparty",       value=default["counterparty"])

        f7, f8 = st.columns(2)
        hour   = f7.slider("Hour (UTC 0–23)",              min_value=0, max_value=23, value=default["hour"])
        ts_str = f8.text_input("Timestamp (ISO 8601)",     value=default["ts"])

        submitted = st.form_submit_button("🔍 Analyse Transaction", type="primary", use_container_width=True)

    # ── Run pipeline ───────────────────────────────────────────────────────────
    if submitted:
        with st.spinner("Running full detection pipeline…"):
            try:
                import sys
                sys.path.insert(0, ".")

                from src.decision_engine import DecisionEngine
                from src.enrichment import enrich_transactions
                from src.features import build_features
                from src.llm.agent import FraudInvestigationAgent
                from src.models.statistical import RulesEngine, statistical_score

                tx_dict = {
                    "tx_id": "DEMO_001", "user_id": user_id,
                    "timestamp": ts_str,  "amount": float(amount),
                    "merchant":  merchant, "mcc": int(mcc),
                    "tx_type":   tx_type,  "counterparty": counterparty,
                    "currency":  "USD",    "lat": 40.71, "lon": -74.01,
                }

                df_in      = pd.DataFrame([tx_dict])
                df_enr     = enrich_transactions(df_in)
                df_feat    = build_features(df_enr)
                row        = df_feat.iloc[0].to_dict()
                stat_s     = float(statistical_score(df_feat).iloc[0])

                rules_eng  = RulesEngine()
                df_rules   = rules_eng.apply(df_feat)

                agent_out  = FraudInvestigationAgent().run(
                    current_tx=row,
                    statistical_signals={
                        "z_amount":            row.get("z_amount", 0),
                        "ewma_z_amount":       row.get("ewma_z_amount", 0),
                        "tx_count_1h":         row.get("tx_count_1h", 0),
                        "tx_count_24h":        row.get("tx_count_24h", 0),
                        "statistical_score":   stat_s,
                        "unsupervised_score":  0.0,
                        "mcc_high_risk":       row.get("mcc_high_risk", 0),
                        "is_new_counterparty": row.get("is_new_counterparty", 0),
                        "geo_impossible":      row.get("geo_impossible", 0),
                    },
                    history_df=None,
                )

                result = DecisionEngine().decide(
                    tx=row,
                    statistical_score=stat_s,
                    unsupervised_score=0.0,
                    supervised_score=None,
                    llm_assessment=agent_out.assessment,
                )

                # ── Display results ────────────────────────────────────────────
                decision_icon = {"alert": "🔴", "monitor": "🟡", "ignore": "🟢"}.get(
                    result.decision, "⚪"
                )
                st.divider()
                st.subheader(f"{decision_icon} Decision: **{result.decision.upper()}**")

                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("Final Score",   f"{result.final_score:.4f}")
                r2.metric("Raw Score",     f"{result.raw_score:.4f}")
                r3.metric("Stat Score",    f"{stat_s:.4f}")
                r4.metric("LLM Risk",      agent_out.assessment.risk.value.upper())
                r5.metric("Confidence",    f"{agent_out.assessment.confidence:.0%}")

                # Score breakdown bar
                bd  = result.signal_breakdown
                fig_bd = go.Figure(go.Bar(
                    x     = list(bd.keys()),
                    y     = list(bd.values()),
                    text  = [f"{v:.4f}" for v in bd.values()],
                    textposition="outside",
                    marker_color=[MODEL_PALETTE.get(k.capitalize(), "#607D8B")
                                  for k in bd.keys()],
                ))
                fig_bd.add_hline(
                    y=result.final_score, line_dash="dot",
                    annotation_text=f"Final score = {result.final_score:.4f}",
                    line_color="#F44336",
                )
                fig_bd.update_layout(
                    title="Signal Contribution to Risk Score",
                    yaxis_title="Score Contribution",
                    yaxis=dict(range=[0, 1.1]),
                    height=300,
                )
                st.plotly_chart(fig_bd, use_container_width=True)

                # Enriched features + LLM assessment
                d1, d2 = st.columns(2)
                with d1:
                    st.subheader("🔬 Enriched Features")
                    st.json({
                        "merchant_clean":       row.get("merchant_clean"),
                        "hour":                 row.get("hour"),
                        "is_night":             bool(row.get("is_night", 0)),
                        "is_new_counterparty":  bool(row.get("is_new_counterparty", 0)),
                        "mcc_high_risk":        bool(row.get("mcc_high_risk", 0)),
                        "tx_count_1h":          row.get("tx_count_1h", 0),
                        "tx_count_24h":         row.get("tx_count_24h", 0),
                        "z_amount":             round(row.get("z_amount", 0), 3),
                        "ewma_z_amount":        round(row.get("ewma_z_amount", 0), 3),
                        "geo_impossible":       bool(row.get("geo_impossible", 0)),
                        "geo_speed_kmh":        round(row.get("geo_speed_kmh", 0), 1),
                    })

                with d2:
                    st.subheader("🤖 LLM Assessment")
                    st.json(agent_out.assessment.to_dict())

                # Triggered rules
                rule_cols = [c for c in df_rules.columns if c.startswith("rule_")]
                triggered = [c.replace("rule_", "") for c in rule_cols
                             if df_rules.iloc[0][c] == 1]
                if triggered:
                    st.warning(f"⚠️ Rules triggered: **{', '.join(triggered)}**")

                # Applied score boosts
                if result.applied_boosts:
                    boost_str = " | ".join(result.applied_boosts)
                    st.info(f"⚡ Score boosts applied: **{boost_str}**")

                # Reasoning trace
                st.subheader("🔍 Agent Reasoning Trace")
                for i, step in enumerate(agent_out.reasoning_trace, 1):
                    icon = "✅" if "complete" in step else "➡️"
                    st.markdown(f"{icon} `{i}.` {step}")

            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                import traceback
                st.code(traceback.format_exc())

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🛡️ **AI Financial Fraud Detection System** · "
    "Statistical · Unsupervised · Supervised · LLM/RAG · Elder Financial Safety  |  "
    "Carefull-style architecture · Built for interview mastery"
)

