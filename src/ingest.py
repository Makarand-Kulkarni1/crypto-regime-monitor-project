"""
ingest.py
Pulls live OHLCV (candlestick) data from Coinbase Exchange's public REST API
and stores it in a local SQLite database.

No API key required — Coinbase's /candles endpoint is public.

Why Coinbase instead of Binance: Binance blocks all API requests from US IP
addresses (HTTP 451), which includes GitHub Actions runners (hosted in US
datacenters). Coinbase Exchange has no such geo-restriction, so this same
script works identically whether run locally or in GitHub Actions.

Usage:
    python ingest.py --symbol BTCUSDT --interval 15m --limit 500
    python ingest.py --symbol BTCUSDT --interval 15m --total 5000
"""

import argparse
import sqlite3
import time
from datetime import datetime, timezone, timedelta

import requests

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product_id}/candles"
DB_PATH = "../data/market_data.db"

# Coinbase candle granularity only accepts these exact values (seconds)
INTERVAL_TO_GRANULARITY = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400
}

# Downstream schema stays identical to the original Binance-based version,
# so store_klines/create_table/predict.py/features.py need zero changes.
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
]

REQUEST_HEADERS = {"User-Agent": "crypto-regime-monitor/1.0"}


def symbol_to_product_id(symbol: str) -> str:
    """Convert a Binance-style symbol (e.g. BTCUSDT) to a Coinbase product_id (e.g. BTC-USD)."""
    symbol = symbol.upper()
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}-USD"
    if symbol.endswith("USD"):
        base = symbol[:-3]
        return f"{base}-USD"
    raise ValueError(f"Don't know how to convert symbol '{symbol}' to a Coinbase product_id")


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


def _coinbase_candles_to_klines(candles: list, granularity: int) -> list:
    """Convert Coinbase's [time, low, high, open, close, volume] rows into the
    same 12-column shape store_klines() already expects (Binance-style)."""
    klines = []
    for c in candles:
        t, low, high, open_, close, volume = c
        open_time_ms = int(t) * 1000
        close_time_ms = open_time_ms + granularity * 1000 - 1
        klines.append([
            open_time_ms, open_, high, low, close, volume,
            close_time_ms, 0, 0, 0, 0, 0  # unused fields, kept for schema compatibility
        ])
    # Coinbase returns newest-first; sort ascending to match Binance's ordering
    klines.sort(key=lambda row: row[0])
    return klines


def fetch_klines(symbol: str, interval: str, limit: int = 300, end_time: int = None):
    """Fetch candlestick data from Coinbase's public API.

    end_time: optional millisecond timestamp - fetches the `limit` candles ending
              at or before this time. Used for pagination to walk backwards.
    limit: capped at 300, Coinbase's max per request.
    """
    if interval not in INTERVAL_TO_GRANULARITY:
        raise ValueError(f"Unsupported interval '{interval}'. Use one of {list(INTERVAL_TO_GRANULARITY)}")

    granularity = INTERVAL_TO_GRANULARITY[interval]
    limit = min(limit, 300)
    product_id = symbol_to_product_id(symbol)
    url = COINBASE_CANDLES_URL.format(product_id=product_id)

    if end_time is not None:
        end_dt = datetime.fromtimestamp(end_time / 1000, tz=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(seconds=granularity * limit)

    params = {
        "granularity": granularity,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
    }
    resp = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=10)
    resp.raise_for_status()
    candles = resp.json()
    return _coinbase_candles_to_klines(candles, granularity)


def fetch_klines_bulk(symbol: str, interval: str, total: int):
    """Fetch `total` candles by paginating backwards in batches of up to 300
    (Coinbase's per-request limit)."""
    all_klines = []
    end_time = None
    remaining = total

    while remaining > 0:
        batch_limit = min(300, remaining)
        batch = fetch_klines(symbol, interval, limit=batch_limit, end_time=end_time)
        if not batch:
            break

        all_klines = batch + all_klines  # prepend since we're walking backwards
        remaining -= len(batch)

        earliest_open_time = batch[0][0]
        # The `end` request parameter is exclusive (candles are returned for
        # t < end), so setting the next batch's end boundary to exactly
        # `earliest_open_time` fetches candles up to, but not including, the
        # one we already have - contiguous, no gap, no overlap. Verified with
        # a boundary-precision unit test.
        end_time = earliest_open_time

        print(f"  fetched {len(batch)} candles, {remaining} remaining, "
              f"earliest so far: {datetime.fromtimestamp(earliest_open_time/1000, tz=timezone.utc)}")

        if len(batch) < batch_limit:
            # Coinbase returned fewer than requested - likely hit the start of available history
            break

        time.sleep(0.34)  # stay comfortably under Coinbase's public rate limit (~3 req/sec)

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
    parser = argparse.ArgumentParser(description="Fetch live crypto OHLCV data from Coinbase")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair, e.g. BTCUSDT (mapped to Coinbase's BTC-USD)")
    parser.add_argument("--interval", default="15m", help="Candle interval: 1m, 5m, 15m, 1h, 6h, 1d")
    parser.add_argument("--limit", type=int, default=300, help="Number of candles to fetch (max 300 per call)")
    parser.add_argument("--total", type=int, default=None,
                         help="Total candles to fetch via pagination (use instead of --limit for >300 candles)")
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
