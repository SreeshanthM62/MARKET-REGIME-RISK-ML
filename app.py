from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.target import LABELS

st.set_page_config(page_title="Market Regime & Risk", layout="wide")

ROOT = Path(__file__).parent
PRED_PATH = ROOT / "results/predictions/oos_predictions.csv"
STRAT_PATH = ROOT / "results/predictions/strategy_returns.csv"
SUMMARY_PATH = ROOT / "results/metrics/research_summary.json"
IMP_PATH = ROOT / "results/metrics/feature_importance.csv"
REGIME_PATH = ROOT / "results/metrics/regime_summary.csv"


@st.cache_data
def load_outputs():
    required = [PRED_PATH, STRAT_PATH, SUMMARY_PATH, IMP_PATH, REGIME_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Research outputs are missing. Run `python scripts/download_data.py` "
            "and then `python scripts/run_research.py` first."
        )
    predictions = pd.read_csv(PRED_PATH, index_col=0, parse_dates=True)
    strategy = pd.read_csv(STRAT_PATH, index_col=0, parse_dates=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    importance = pd.read_csv(IMP_PATH)
    regime = pd.read_csv(REGIME_PATH)
    return predictions, strategy, summary, importance, regime


def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x * 100:.1f}%"


def main():
    st.title("Market Regime & Risk Prediction")
    st.caption(
        "Research dashboard for SPY regime classification and risk-aware exposure. "
        "Model outputs are historical research results, not investment advice."
    )

    try:
        predictions, strategy, summary, importance, regime = load_outputs()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    best_model = summary["best_model"]
    latest = predictions.iloc[-1]
    current_regime = int(latest["predicted_regime"])
    confidence = float(latest["prediction_probability"])
    risk_map = {0: "High", 1: "Moderate", 2: "Lower"}
    exposure_map = {0: 0, 1: 50, 2: 100}

    st.subheader("Latest Available Model View")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Regime", LABELS[current_regime])
    c2.metric("Model", best_model)
    c3.metric("Model Probability", f"{confidence * 100:.1f}%")
    c4.metric("Suggested Exposure", f"{exposure_map[current_regime]}%")

    st.info(
        f"Risk context: **{risk_map[current_regime]}**. "
        "Model probability is the classifier's probability estimate; it is not a probability of profit."
    )

    st.subheader("Regime Timeline")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions["SPY_close"],
            mode="lines",
            name="SPY",
        )
    )
    colors = {0: "rgba(220, 80, 80, 0.12)", 1: "rgba(150, 150, 150, 0.10)", 2: "rgba(70, 160, 90, 0.12)"}
    for regime_id in [0, 1, 2]:
        mask = predictions["predicted_regime"] == regime_id
        if not mask.any():
            continue
        dates = predictions.index[mask]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predictions.loc[mask, "SPY_close"],
                mode="markers",
                marker=dict(size=4),
                name=LABELS[regime_id],
            )
        )
    fig.update_layout(height=430, xaxis_title="", yaxis_title="SPY Price")
    st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Model Performance", "Feature Analysis", "Risk Analysis", "Strategy Comparison"]
    )

    with tab1:
        rows = []
        for name, values in summary["model_metrics"].items():
            rows.append(
                {
                    "Model": name,
                    "Validation Accuracy": values["validation"]["accuracy"],
                    "Validation Macro F1": values["validation"]["macro_f1"],
                    "Test Accuracy": values["test"]["accuracy"],
                    "Test Macro F1": values["test"]["macro_f1"],
                    "Test Balanced Accuracy": values["test"]["balanced_accuracy"],
                }
            )
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        cm = np.zeros((3, 3), dtype=int)
        y_true = predictions["regime"].astype(int)
        y_pred = predictions["predicted_regime"].astype(int)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Bearish", "Neutral", "Bullish"],
            y=["Bearish", "Neutral", "Bullish"],
            labels={"x": "Predicted", "y": "Actual"},
            title="Out-of-Sample Confusion Matrix",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab2:
        imp = importance.head(12).sort_values("importance_mean")
        fig_imp = px.bar(
            imp,
            x="importance_mean",
            y="feature",
            orientation="h",
            error_x="importance_std",
            title="Permutation Importance on Test Set",
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(
            "Importance indicates contribution to classification performance under feature permutation; "
            "it does not establish causality."
        )

    with tab3:
        st.dataframe(regime.set_index("regime"), use_container_width=True)
        fig_vol = px.bar(
            regime,
            x="regime",
            y="annualized_volatility",
            title="Realized Volatility by Predicted Regime",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        fig_dd = px.bar(
            regime,
            x="regime",
            y="max_drawdown",
            title="Maximum Drawdown by Predicted Regime",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab4:
        strategy_curve = (1 + strategy[["buy_hold_return", "strategy_return"]].fillna(0)).cumprod()
        strategy_curve.columns = ["Buy & Hold", "ML Regime Overlay"]
        fig_perf = px.line(
            strategy_curve,
            title="Out-of-Sample Growth of $1",
            labels={"value": "Growth", "index": "Date"},
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        metrics = pd.DataFrame(
            {
                "Buy & Hold": summary["buy_hold_metrics"],
                "ML Regime Overlay": summary["strategy_metrics"],
            }
        )
        st.dataframe(metrics, use_container_width=True)

        st.caption(
            "The overlay uses 100% / 50% / 0% exposure for Bullish / Neutral / Bearish "
            "and deducts a simple 5 bps cost when exposure changes."
        )

    with st.sidebar:
        st.header("Research Setup")
        st.write(f"**Test period:** {summary['test_period'][0]} → {summary['test_period'][1]}")
        st.write(f"**Target horizon:** {summary['target_horizon_days']} trading days")
        st.write(
            f"**Thresholds:** Bearish < {summary['bearish_threshold']:.2%}; "
            f"Bullish > {summary['bullish_threshold']:.2%}"
        )
        st.divider()
        st.write("Data: Yahoo Finance via yfinance.")
        st.write("Primary asset: SPY.")
        st.write("Supporting variables: QQQ, TLT, GLD, VIX, 10Y yield index, DXY.")


if __name__ == "__main__":
    main()
