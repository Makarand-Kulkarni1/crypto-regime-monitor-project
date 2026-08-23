"""
dashboard.py
Live monitoring dashboard for the crypto market regime prediction system.

Reads from market_data.db (updated every 15 minutes by predict.py via GitHub
Actions) and shows:
- Current predicted regime + confidence breakdown
- Price chart color-coded by historical predicted regime
- Live accuracy: for predictions whose target time has already passed, we
  recompute what the regime ACTUALLY turned out to be (using the same
  labeling rule the model was trained on) and check if the model was right -
  genuine closed-loop model monitoring, not just a static display.
- Feature importance from the trained model
- Recent prediction log with shift markers

Run locally:
    streamlit run dashboard.py
"""

import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Resolve paths relative to THIS FILE's location, not the current working
# directory. This matters because Streamlit Community Cloud always runs the
# app with the working directory set to the repo root, regardless of where
# the entrypoint file lives - a hardcoded "../data/..." path (which works
# fine when running `cd src && streamlit run dashboard.py` locally) would
# silently resolve to the wrong location on Cloud. Using __file__ makes this
# work identically in both environments.
BASE_DIR = Path(__file__).resolve().parent  # .../src
sys.path.insert(0, str(BASE_DIR))  # ensure local imports below work regardless of cwd

from features import compute_features
from regime_utils import label_regime

DB_PATH = BASE_DIR.parent / "data" / "market_data.db"
MODEL_PATH = BASE_DIR.parent / "models" / "regime_classifier.joblib"
SYMBOL = "btcusdt"

REGIME_COLORS = {
    "Low-Vol Trending": "#2ecc71",
    "Ranging / Choppy": "#95a5a6",
    "High-Vol Breakout": "#e74c3c",
}

st.set_page_config(page_title="Crypto Regime Monitor", page_icon="📊", layout="wide")


@st.cache_data(ttl=60)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    prices = pd.read_sql(f"SELECT * FROM {SYMBOL} ORDER BY open_time", conn)
    predictions = pd.read_sql("SELECT * FROM predictions ORDER BY run_time", conn)
    conn.close()

    prices["datetime"] = pd.to_datetime(prices["open_time"], unit="ms").astype("datetime64[ns]")
    predictions["run_time"] = pd.to_datetime(predictions["run_time"]).dt.tz_localize(None).astype("datetime64[ns]")
    predictions["latest_candle_time"] = pd.to_datetime(predictions["latest_candle_time"]).dt.tz_localize(None).astype("datetime64[ns]")

    return prices, predictions


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=60)
def compute_actual_outcomes(prices: pd.DataFrame, predictions: pd.DataFrame,
                             vol_ratio_threshold: float, trend_strength_threshold: float):
    """
    For each past prediction whose target time (latest_candle_time + horizon)
    has already occurred, compute what regime ACTUALLY happened at that time
    and compare it to what was predicted. This is what makes the dashboard a
    genuine monitoring tool rather than just a static display of predictions.
    """
    prices_indexed = prices.set_index("datetime").sort_index()
    features_df = compute_features(prices_indexed).dropna()

    results = []
    now = datetime.now()

    for _, pred in predictions.iterrows():
        horizon_minutes = pred["horizon_candles"] * 15
        target_time = pred["latest_candle_time"] + timedelta(minutes=horizon_minutes)

        if target_time > now:
            continue  # target time hasn't happened yet - can't evaluate this one

        # Find the closest available candle at or after the target time
        future_candles = features_df[features_df.index >= target_time]
        if future_candles.empty:
            continue

        actual_row = future_candles.iloc[0]
        actual_regime = label_regime(
            actual_row["vol_ratio"], actual_row["trend_strength"],
            vol_ratio_threshold, trend_strength_threshold
        )

        results.append({
            "run_time": pred["run_time"],
            "predicted_regime": pred["predicted_regime"],
            "actual_regime": actual_regime,
            "correct": pred["predicted_regime"] == actual_regime,
        })

    return pd.DataFrame(results)


def main():
    st.title("📊 Crypto Market Regime Monitor")
    st.caption("Live BTC/USD regime forecasting — updated automatically every 15 minutes via GitHub Actions")

    try:
        prices, predictions = load_data()
    except Exception as e:
        st.error(f"Couldn't load data from {DB_PATH}. Make sure you've run ingest.py and predict.py at least once. ({e})")
        return

    if predictions.empty:
        st.warning("No predictions logged yet. Run `python predict.py` at least once to populate the dashboard.")
        return

    try:
        bundle = load_model_bundle()
    except Exception as e:
        st.error(f"Couldn't load model from {MODEL_PATH}. ({e})")
        return

    latest = predictions.iloc[-1]

    # ============================================================
    # Sidebar controls
    # ============================================================
    st.sidebar.header("Settings")
    lookback_days = st.sidebar.slider("Price chart lookback (days)", min_value=1, max_value=60, value=30)
    st.sidebar.divider()
    st.sidebar.caption(
        f"**Model:** Random Forest\n\n"
        f"**Horizon:** {int(latest['horizon_candles'])} candles "
        f"({int(latest['horizon_candles'])*15} min ahead)\n\n"
        f"**Data source:** Coinbase Exchange\n\n"
        f"**Automation:** GitHub Actions, every 15 min"
    )

    # ============================================================
    # Top row: current status
    # ============================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Predicted Regime (1hr ahead)", latest["predicted_regime"])

    with col2:
        st.metric("Latest Price", f"${latest['latest_close']:,.2f}")

    with col3:
        last_run = latest["run_time"]
        now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        minutes_ago = (now_naive - last_run).total_seconds() / 60
        st.metric("Last Updated", f"{minutes_ago:.0f} min ago")

    with col4:
        total_shifts = int(predictions["regime_shift_detected"].sum())
        st.metric("Regime Shifts Detected (all-time)", total_shifts)

    if latest["regime_shift_detected"]:
        st.warning(f"⚠️ Regime shift detected on the most recent run — forecast changed to **{latest['predicted_regime']}**")

    # --- Confidence breakdown for the current prediction ---
    if pd.notna(latest.get("prob_high_vol_breakout")):
        st.subheader("Current Prediction Confidence")
        prob_data = {
            "High-Vol Breakout": latest["prob_high_vol_breakout"],
            "Low-Vol Trending": latest["prob_low_vol_trending"],
            "Ranging / Choppy": latest["prob_ranging_choppy"],
        }
        fig_conf = go.Figure(go.Bar(
            x=list(prob_data.values()), y=list(prob_data.keys()), orientation="h",
            marker=dict(color=[REGIME_COLORS[k] for k in prob_data.keys()]),
            text=[f"{v:.0%}" for v in prob_data.values()], textposition="outside"
        ))
        fig_conf.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis=dict(range=[0, 1], tickformat=".0%"))
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("Confidence breakdown unavailable for this prediction (logged before probability tracking was added).")

    st.divider()

    # ============================================================
    # Live accuracy tracking - the closed-loop monitoring section
    # ============================================================
    st.subheader("🎯 Live Model Accuracy")
    st.caption("Comparing past predictions against what actually happened, once the forecast target time has passed.")

    outcomes = compute_actual_outcomes(
        prices, predictions,
        bundle["vol_ratio_threshold"], bundle["trend_strength_threshold"]
    )

    if outcomes.empty:
        st.info(
            "No predictions have reached their target time yet (each prediction needs "
            f"{int(latest['horizon_candles'])*15} minutes to pass before it can be checked). "
            "This will populate automatically as the system keeps running."
        )
    else:
        acc_col1, acc_col2, acc_col3 = st.columns(3)
        overall_accuracy = outcomes["correct"].mean()
        with acc_col1:
            st.metric("Live Accuracy", f"{overall_accuracy:.0%}", help="Based on resolved predictions only")
        with acc_col2:
            st.metric("Predictions Evaluated", len(outcomes))
        with acc_col3:
            majority_baseline = outcomes["actual_regime"].value_counts(normalize=True).max()
            st.metric("Naive Baseline", f"{majority_baseline:.0%}", help="Accuracy of always guessing the most common regime")

        # Rolling accuracy over time
        outcomes_sorted = outcomes.sort_values("run_time").reset_index(drop=True)
        outcomes_sorted["rolling_accuracy"] = outcomes_sorted["correct"].expanding(min_periods=1).mean()

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=outcomes_sorted["run_time"], y=outcomes_sorted["rolling_accuracy"],
            mode="lines+markers", name="Cumulative Accuracy", line=dict(color="#3498db")
        ))
        fig_acc.add_hline(y=majority_baseline, line_dash="dash", line_color="gray",
                           annotation_text="Naive baseline")
        fig_acc.update_layout(height=250, yaxis=dict(range=[0, 1], tickformat=".0%"),
                               margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_acc, use_container_width=True)

    st.divider()

    # ============================================================
    # Price chart colored by predicted regime
    # ============================================================
    st.subheader("Price History")

    cutoff = prices["datetime"].max() - timedelta(days=lookback_days)
    prices_windowed = prices[prices["datetime"] >= cutoff]

    merged = pd.merge_asof(
        prices_windowed.sort_values("datetime"),
        predictions[["run_time", "predicted_regime"]].sort_values("run_time"),
        left_on="datetime", right_on="run_time",
        direction="backward"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged["datetime"], y=merged["close"],
        mode="lines", line=dict(color="rgba(150,150,150,0.4)", width=1),
        showlegend=False, hoverinfo="skip"
    ))

    for regime, color in REGIME_COLORS.items():
        subset = merged[merged["predicted_regime"] == regime]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset["datetime"], y=subset["close"],
                mode="markers", name=regime,
                marker=dict(color=color, size=5)
            ))

    fig.update_layout(
        height=450, xaxis_title="Time", yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ============================================================
    # Regime distribution + feature importance
    # ============================================================
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Predicted Regime Distribution")
        regime_counts = predictions["predicted_regime"].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=regime_counts.index, values=regime_counts.values,
            marker=dict(colors=[REGIME_COLORS.get(r, "#333") for r in regime_counts.index]),
            hole=0.4
        )])
        fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("What Drives the Model's Predictions")
        model = bundle["model"]
        feature_cols = bundle["feature_cols"]
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig_imp = go.Figure(go.Bar(
            x=importances.values, y=importances.index, orientation="h",
            marker=dict(color="#9b59b6")
        ))
        fig_imp.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # ============================================================
    # Recent predictions log
    # ============================================================
    st.subheader("Recent Predictions")
    display_df = predictions[["run_time", "latest_close", "predicted_regime", "regime_shift_detected"]].tail(20).iloc[::-1].copy()
    display_df["run_time"] = display_df["run_time"].dt.strftime("%Y-%m-%d %H:%M UTC")
    display_df["regime_shift_detected"] = display_df["regime_shift_detected"].map({1: "🔴 Shift", 0: ""})
    display_df.columns = ["Run Time", "Price", "Predicted Regime", "Shift?"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption(
        "Model: Random Forest classifier forecasting market regime 60 minutes ahead, "
        "using volume/volatility/momentum features. Data source: Coinbase Exchange public API. "
        "Automated via GitHub Actions cron every 15 minutes."
    )


if __name__ == "__main__":
    main()
