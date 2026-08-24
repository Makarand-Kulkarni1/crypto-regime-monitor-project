# 📊 Crypto Market Regime Monitor

**Live BTC/USD market regime forecasting — a fully automated, cloud-deployed ML pipeline that fetches live data, predicts market conditions 1 hour ahead, and continuously monitors its own accuracy against reality.**

![Python](https://img.shields.io/badge/python-3.11-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red)
![Automation](https://img.shields.io/badge/automation-GitHub%20Actions-black)
![Status](https://img.shields.io/badge/status-live-brightgreen)

---

## What this project does

Most crypto ML projects try to predict *price* directly — a weak target, since short-term crypto price movement is close to a random walk. Instead, this project predicts something more tractable and genuinely useful: **which market regime BTC/USD is likely to be in one hour from now** — calm and trending, calm and directionless, or breaking into high volatility.

The system runs **continuously and unattended in the cloud**: every 15 minutes, GitHub Actions fetches fresh market data, computes technical indicators, generates a forecast, and logs the result — including automatically checking past predictions against what actually happened, so model accuracy is measured in production, not just at training time.

**[🔗 Live Dashboard →](https://crypto-regime-monitor-project.streamlit.app/)**

![Dashboard Overview](assets/screenshots/dashboard-overview.png)
*Current predicted regime, latest price, and live model accuracy — updated automatically every 15 minutes.*

---

## Why this project is different from a typical fresher ML project

- **Forecasts the future, not the present.** The model predicts the regime *ahead of time* using only leading indicators — not features that trivially reconstruct an already-known label (see [Data Leakage Catch](#the-data-leakage-catch-and-fix) below).
- **Fully automated, not a one-off notebook.** A scheduled cloud pipeline (GitHub Actions) runs the system every 15 minutes with no manual intervention, and survives machine restarts, laptop being closed, etc.
- **Closed-loop accuracy monitoring.** The dashboard doesn't just show predictions — it checks them against reality once enough time has passed and reports live accuracy vs. a naive baseline, the same way real MLOps systems monitor model drift in production.
- **Debugged like a real system, not a toy.** This README documents real engineering problems encountered and fixed along the way (data leakage, a hard geo-blocked API dependency, timezone bugs, a pagination boundary bug) — because handling things going wrong is most of real ML engineering.

---

## Architecture

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  Coinbase Exchange │────▶│  ingest.py         │────▶│  market_data.db    │
│  API               │     │  (fetch OHLCV)      │     │  (SQLite)          │
└───────────────────┘     └───────────────────┘     └──────────┬────────┘
                                                                  │
                                                                  ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  regime_classifier │◀────│  features.py        │◀────│  Feature Engineering│
│  .joblib (RF model)│     │  (shared module)     │     │  (vol, RSI, ATR,    │
└──────────┬────────┘     └───────────────────┘     │  volume, trend)     │
           │                                          └───────────────────┘
           ▼
┌───────────────────┐     ┌───────────────────┐
│  predict.py         │────▶│  predictions table  │
│  (forecast + shift   │     │  (probabilities,     │
│  detection)          │     │  shift alerts)        │
└───────────────────┘     └──────────┬────────┘
           ▲                                          │
           │  runs every 15 min                        ▼
┌───────────────────┐                        ┌───────────────────┐
│  GitHub Actions     │                        │  dashboard.py       │
│  (cron schedule)     │                        │  (Streamlit, live    │
└───────────────────┘                        │  accuracy tracking)  │
                                               └───────────────────┘
```

---

## Repository structure

```
crypto-regime-monitor/
├── .github/workflows/
│   └── predict.yml              # GitHub Actions: runs predict.py every 15 min
├── data/
│   └── market_data.db           # SQLite: raw candles + prediction log
├── models/
│   └── regime_classifier.joblib # Trained Random Forest + saved thresholds
├── notebooks/
│   ├── 02_feature_engineering.ipynb
│   └── 03_regime_labeling_and_model.ipynb
├── src/
│   ├── ingest.py                # Live OHLCV data fetching (Coinbase Exchange API)
│   ├── features.py              # Shared feature engineering (used by notebooks + pipeline)
│   ├── regime_utils.py          # Shared regime-labeling rule (training + monitoring)
│   ├── predict.py               # Automation entrypoint: fetch → predict → log → alert
│   └── dashboard.py             # Streamlit live monitoring dashboard
└── requirements.txt
```

---

## How the model works

**Problem framing:** classify the market into one of three regimes, **1 hour ahead of time**:

| Regime | Meaning |
|---|---|
| 🟢 Low-Vol Trending | Calm market, clear directional move |
| ⚪ Ranging / Choppy | Calm market, no clear direction |
| 🔴 High-Vol Breakout | Volatility is expanding — the risk regime |

**Labeling:** since there's no ground-truth "regime" label anywhere, labels are derived from the data itself using quantile thresholds on realized volatility ratio and trend strength — an approach that adapts to whatever asset/timeframe is used, rather than relying on arbitrary fixed numbers.

**Model:** Random Forest classifier, chosen deliberately over a deep learning approach — interpretable, robust on a modest dataset size, and defensible in review (feature importances are directly inspectable, not a black box).

**Evaluation:** chronological train/test split (never randomly shuffled — that would leak future information into training, a classic time-series mistake). Evaluated with per-class precision/recall, not just accuracy, since the "Ranging/Choppy" class is naturally more common than "High-Vol Breakout."

**Forecast horizon:** tested horizons from 15 minutes to 3 hours ahead and measured lift over a naive baseline at each. Performance peaked at 30–60 minutes and declined on both sides — too short, the signal hasn't developed yet; too long, too much can happen in between. **60 minutes was chosen** over the marginally-better 30-minute horizon because the extra lead time is more operationally useful for an alerting system, even at a negligible accuracy cost — a product decision, not just a metrics decision.

---

## Engineering challenges (and how they were caught and fixed)

Real ML engineering is mostly about catching things going wrong before they reach production. A few worth calling out:

### The data leakage catch and fix
The first version of the model scored **~99–100% accuracy** — a huge red flag, not a win. The regime label had been built from `vol_ratio` and `trend_strength`, and those same columns were also included as model inputs — so the model was just reverse-engineering the threshold rule, not learning anything real. Fixed by excluding label-defining features from the input set and reframing the task from *nowcasting* (predicting the already-known current regime) to genuine *forecasting* (predicting the regime ahead of time, using only independent leading indicators). Final honest accuracy: **~63%** against a **~44% naive baseline** — a real, defensible lift.

### The Binance geo-block
Local testing worked perfectly against Binance's API. The moment the pipeline moved to GitHub Actions' cloud runners, it started failing with an HTTP 451 error — Binance blocks all US-based IP addresses, and GitHub's runners are hosted in US datacenters. Diagnosed via the error code (not guesswork), then migrated the data source to Coinbase Exchange's public API, which has no such restriction, without changing anything downstream (feature engineering, the model, or the automation logic) since the new fetch function was built to output an identical schema.

### Boundary and timezone bugs
Building the pagination logic for bulk historical data fetches introduced a subtle off-by-one-candle gap at batch boundaries, caught with a targeted unit test before it ever reached production. Separately, mixing timezone-aware and timezone-naive timestamps between two tables caused a dashboard crash — fixed and re-verified against data shaped exactly like the real database, not generic test data.

---

## Live automation

The entire prediction pipeline runs **without any manual intervention**, via a GitHub Actions cron job:

- Every 15 minutes: fetch new candles → recompute features → forecast the regime 1 hour ahead → log the prediction and its confidence → detect and flag regime shifts → commit results back to the repo
- No servers to maintain, no always-on machine required
- Fully visible: every run's logs are in the repo's Actions tab

## Dashboard

The Streamlit dashboard (`src/dashboard.py`) provides live, closed-loop monitoring — not just a static display of predictions.

**Live accuracy tracking** — the standout feature. The dashboard automatically checks past predictions against what actually happened once enough time has passed, and plots cumulative accuracy against a naive baseline. This is genuine closed-loop model monitoring, the same concept real MLOps systems use in production:
![Live accuracy tracking](assets/screenshots/live-accuracy-chart.png)

**Price history, color-coded by predicted regime:**
![Price history chart](assets/screenshots/price-history-chart.png)

**Regime distribution and feature importance** — showing what's actually driving the model's predictions, not a black box:
![Regime distribution and feature importance](assets/screenshots/regime-distribution-feature-importance.png)

**Recent predictions log:**
![Recent predictions log](assets/screenshots/recent-predictions-table.png)

Run locally:
```bash
pip install -r requirements.txt
cd src
streamlit run dashboard.py
```

---

## Setup

```bash
git clone https://github.com/<your-username>/crypto-regime-monitor-project.git
cd crypto-regime-monitor-project
pip install -r requirements.txt

# Fetch historical data
cd src
python ingest.py --symbol BTCUSDT --interval 15m --total 5000

# Run a prediction manually
python predict.py --symbol BTCUSDT --interval 15m

# Launch the dashboard
streamlit run dashboard.py
```

To enable the cloud automation on your own fork: push to GitHub, enable **Settings → Actions → General → Read and write permissions**, then trigger `.github/workflows/predict.yml` manually or wait for the cron schedule.

---

## Honest limitations

- Trained on a single asset (BTC/USD) and a single timeframe (15-minute candles) — not yet validated across other pairs or intervals
- Backtested accuracy was ~63% vs. a ~44% naive baseline. **Live production accuracy is being tracked in real time by the dashboard itself** (see screenshot above) — as of this writing it's around 61% on a still-small sample of resolved predictions, and will become more statistically meaningful as more predictions accumulate over time. This live number, not the backtest number, is the one to trust as the sample grows — which is exactly why the tracker exists rather than just reporting a single static backtest score.
- Labeling is rule-based (quantile thresholds), not ground truth — a reasonable and explainable approach given no true regime labels exist, but a design choice worth defending, not treating as objective fact

---

## Possible next steps

- Multi-asset support (ETH, SOL, etc.) to test generalization
- Slack/Telegram alerting on regime shifts, not just dashboard display
- Model retraining pipeline triggered automatically when live accuracy drifts below a threshold
- Backtesting a simple trading strategy conditioned on predicted regime, to translate ML accuracy into a business/PnL metric

---

## Tech stack

**Data & ML:** Python, pandas, NumPy, scikit-learn, SQLite
**Automation:** GitHub Actions (cron scheduling)
**Dashboard:** Streamlit, Plotly
**Data source:** Coinbase Exchange public API
