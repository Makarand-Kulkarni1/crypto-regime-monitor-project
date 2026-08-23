"""
predict.py
The core automation script. Designed to run on a schedule (e.g. every 15 min
via GitHub Actions cron).

What it does each run:
1. Fetches the latest candles from Binance (appends to market_data.db)
2. Computes features on the full history
3. Loads the trained model
4. Predicts the regime HORIZON candles ahead (the current forecast)
5. Logs the prediction to a predictions table
6. Compares against the previous run's prediction - if the forecasted regime
   changed, prints an alert (visible in GitHub Actions logs; can be wired to
   email/Slack/Telegram later)

Usage:
    python predict.py --symbol BTCUSDT --interval 15m
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd

sys.path.append(".")
from features import compute_features
from ingest import create_table, fetch_klines, store_klines

DB_PATH = "../data/market_data.db"
MODEL_PATH = "../models/regime_classifier.joblib"


def create_predictions_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            run_time TEXT PRIMARY KEY,
            latest_candle_time TEXT,
            latest_close REAL,
            predicted_regime TEXT,
            horizon_candles INTEGER,
            regime_shift_detected INTEGER
        )
    """)
    conn.commit()


def get_last_prediction(conn):
    cur = conn.execute("""
        SELECT predicted_regime FROM predictions
        ORDER BY run_time DESC LIMIT 1
    """)
    row = cur.fetchone()
    return row[0] if row else None


def log_prediction(conn, run_time, latest_candle_time, latest_close,
                    predicted_regime, horizon, shift_detected):
    conn.execute("""
        INSERT OR REPLACE INTO predictions
        (run_time, latest_candle_time, latest_close, predicted_regime, horizon_candles, regime_shift_detected)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (run_time, str(latest_candle_time), latest_close, predicted_regime, horizon, int(shift_detected)))
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Live regime prediction pipeline")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    args = parser.parse_args()

    run_time = datetime.now(timezone.utc).isoformat()
    print(f"=== Prediction run started: {run_time} ===")

    conn = sqlite3.connect(DB_PATH)
    create_table(conn, args.symbol)
    create_predictions_table(conn)

    # --- 1. Fetch latest candles ---
    print(f"Fetching latest {args.interval} candles for {args.symbol}...")
    klines = fetch_klines(args.symbol, args.interval, limit=100)
    stored = store_klines(conn, args.symbol, klines)
    print(f"Stored {stored} new candle rows (duplicates ignored)")

    # --- 2. Load full history and compute features ---
    table = args.symbol.lower()
    df = pd.read_sql(f"SELECT * FROM {table} ORDER BY open_time", conn)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("datetime").sort_index()

    features_df = compute_features(df)
    features_df = features_df.dropna()

    if features_df.empty:
        print("ERROR: not enough data to compute features. Exiting.")
        conn.close()
        sys.exit(1)

    # --- 3. Load trained model ---
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    horizon = bundle["horizon"]

    # --- 4. Predict on the most recent row (this is "now") ---
    latest_row = features_df.iloc[[-1]][feature_cols]
    predicted_regime = model.predict(latest_row)[0]
    probs = model.predict_proba(latest_row)[0]
    prob_dict = dict(zip(model.classes_, probs.round(3)))

    latest_candle_time = features_df.index[-1]
    latest_close = features_df["close"].iloc[-1]

    print(f"Latest candle: {latest_candle_time} | close={latest_close:.2f}")
    print(f"Predicted regime ({horizon} candles / {horizon*15}min ahead): {predicted_regime}")
    print(f"Class probabilities: {prob_dict}")

    # --- 5. Compare to last run's prediction, detect shift ---
    last_prediction = get_last_prediction(conn)
    shift_detected = last_prediction is not None and last_prediction != predicted_regime

    if shift_detected:
        print(f"*** REGIME SHIFT ALERT: {last_prediction} -> {predicted_regime} ***")
    else:
        print(f"No regime shift (previous prediction: {last_prediction})")

    # --- 6. Log this run ---
    log_prediction(conn, run_time, latest_candle_time, latest_close,
                    predicted_regime, horizon, shift_detected)

    conn.close()
    print("=== Prediction run complete ===")


if __name__ == "__main__":
    main()
