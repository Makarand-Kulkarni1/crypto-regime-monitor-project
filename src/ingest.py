"""
ingest.py
Pulls live OHLCV (candlestick) data from Binance's public REST API
and stores it in a local SQLite database.

No API key required — Binance's /klines endpoint is public.

Usage:
    python ingest.py --symbol BTCUSDT --interval 15m --limit 500
"""

import argparse
import sqlite3
import time
from datetime import datetime, timezone

import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
DB_PATH = "../data/market_data.db"

# Binance kline response columns, in order
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
]


def create_table(conn, symbol: str):
    """Create the table for a given symbol if it doesn't exist."""
    table = symbol.lower()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            open_time INTEGER PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            close_time INTEGER,
            num_trades INTEGER,
            fetched_at TEXT
        )
    """)
    conn.commit()


def fetch_klines(symbol: str, interval: str, limit: int = 500, end_time: int = None):
    """Fetch candlestick data from Binance public API.

    end_time: optional millisecond timestamp - fetches candles ending at or before this time.
              Used for pagination to walk backwards through history.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time is not None:
        params["endTime"] = end_time
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_klines_bulk(symbol: str, interval: str, total: int):
    """Fetch `total` candles by paginating backwards in batches of up to 1000.

    Binance limits each call to 1000 candles, so for larger requests we walk
    backwards in time: fetch the most recent batch, note the earliest open_time
    in that batch, then request the next batch ending just before it, and so on.
    """
    all_klines = []
    end_time = None
    remaining = total

    while remaining > 0:
        batch_limit = min(1000, remaining)
        batch = fetch_klines(symbol, interval, limit=batch_limit, end_time=end_time)
        if not batch:
            break

        all_klines = batch + all_klines  # prepend since we're walking backwards
        remaining -= len(batch)

        earliest_open_time = batch[0][0]
        end_time = earliest_open_time - 1  # next batch ends right before this one starts

        print(f"  fetched {len(batch)} candles, {remaining} remaining, "
              f"earliest so far: {datetime.fromtimestamp(earliest_open_time/1000, tz=timezone.utc)}")

        if len(batch) < batch_limit:
            # Binance returned fewer than requested - likely hit the start of available history
            break

        time.sleep(0.3)  # be polite to the API, avoid rate limits

    return all_klines


def store_klines(conn, symbol: str, klines: list):
    """Insert klines into SQLite, ignoring duplicates (same open_time)."""
    table = symbol.lower()
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for k in klines:
        row = dict(zip(KLINE_COLUMNS, k))
        rows.append((
            int(row["open_time"]),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row["volume"]),
            int(row["close_time"]),
            int(row["num_trades"]),
            fetched_at
        ))

    conn.executemany(f"""
        INSERT OR IGNORE INTO {table}
        (open_time, open, high, low, close, volume, close_time, num_trades, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Fetch live crypto OHLCV data from Binance")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair, e.g. BTCUSDT")
    parser.add_argument("--interval", default="15m", help="Candle interval: 1m, 5m, 15m, 1h, 4h, 1d")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles to fetch (max 1000 per call)")
    parser.add_argument("--total", type=int, default=None,
                         help="Total candles to fetch via pagination (use instead of --limit for >1000 candles)")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    create_table(conn, args.symbol)

    if args.total:
        print(f"Fetching {args.total} x {args.interval} candles for {args.symbol} (paginated)...")
        klines = fetch_klines_bulk(args.symbol, args.interval, args.total)
    else:
        print(f"Fetching {args.limit} x {args.interval} candles for {args.symbol}...")
        klines = fetch_klines(args.symbol, args.interval, args.limit)

    count = store_klines(conn, args.symbol, klines)
    print(f"Stored {count} candles (duplicates ignored) into {DB_PATH}")

    conn.close()


if __name__ == "__main__":
    main()
