# ProTrader AI — AI-Powered Stock Analytics Platform (NSE / Indian Markets)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.2.x-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-028cf0.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A research-grade stock analytics platform for **Indian markets (NSE)** that combines a
calibrated multi-day directional signal, multi-source sentiment, institutional-flow
tracking, Bayesian multi-expert fusion, and chart-pattern detection — served through a
**Next.js + FastAPI** web app and a secondary **Streamlit** app, with a paper-trading
pipeline and a self-resolving accuracy ledger that keeps the reported numbers honest.

> **For educational and research purposes only. Not financial advice.**

---

## 📋 Table of Contents

- [Honest Results (Real Numbers)](#-honest-results-real-numbers)
- [Two App Surfaces](#-two-app-surfaces)
- [What Makes This Novel](#-what-makes-this-novel)
- [Architecture](#-architecture)
- [Data Sources](#-data-sources)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Surface](#-api-surface)
- [Daily Operations](#-daily-operations-paper-trading--accuracy)
- [Deployment](#-deployment)
- [Known Limitations](#-known-limitations)
- [Disclaimer](#-disclaimer)

---

## 📊 Honest Results (Real Numbers)

This project's defining principle is that **every reported number is the measured,
walk-forward out-of-sample result — never an in-sample fit, and never inflated.**
That stance produced two findings worth stating up front:

### 1. There is no honest *next-day* single-name directional edge

On liquid NSE large-caps (2018–2026) we tested five independent approaches —
per-ticker ensemble, cross-sectional logistic regression (AUC ≈ 0.49), LightGBM +
reversion (AUC ≈ 0.45), trend/momentum gating, and market-regime gating. **None beat
equity drift.** Claims of 60–70% next-day direction are not honestly achievable on
this universe. After fixing a calibration bug (below), the model's reported next-day
"up" probability correctly sits **near 0.50**, where it belongs.

### 2. Calibration fix (high value)

`models/hybrid_model.py` previously reported a next-day directional probability of
**~80%** while the real hit-rate was **~51%** — Expected Calibration Error (ECE) of
**0.345** and a Brier score worse than a coin flip. Root cause: the isotonic
calibrator was fit on one prediction surface and applied to a different one, and the
final tree/RNN blend was never itself calibrated. After the fix (matching-surface
isotonic + a final Platt scaler fit on the validation fold):

| Metric | Before | After |
|--------|--------|-------|
| ECE    | 0.345  | **0.049** |
| Reported P(up) | ~0.80 (false confidence) | ~0.50 (honest) |

### 3. Where a *real* edge exists: the 30-day swing signal

`models/directional_signal.py` answers a different, tractable question:
**"Over the next ~30 trading days, is this name more likely to be higher — and is the
setup convincing enough to act on?"** It is deliberately **long / neutral only**
(confident short calls on single names are destroyed by upward drift — ~10% precision
OOS). Measured walk-forward on 54 NSE names over 2018–2026 (91,471 train rows,
**50,272 out-of-sample rows**):

| Metric | Value |
|--------|-------|
| Horizon | 30 trading days (~1.5 months) |
| Fires UP only above conviction threshold | τ\* = 0.63 |
| OOS precision of the fired bucket | **60.6%** |
| "Always-up" base rate (the bar to beat) | 58.0% |
| Net edge | **+2.5 pp**, positive in **4 of 5 years** |
| Coverage (how often it fires) | 5.6% — it stays quiet by design |
| Ranking AUC | 0.47 — near-random *by design*; the edge lives in the high-conviction tail, not the ordering |

Per-year OOS precision vs. base rate: 2022 `61.6 / 54.0` · 2023 `68.3 / 65.3` ·
2024 `53.3 / 55.7` (below base) · 2025 `69.0 / 58.2` · 2026 `54.5 / 47.1`.

The shipped model lives at `models/saved/directional_signal.pkl` with its
walk-forward stats baked in; retrain monthly with
`DirectionalSignal(horizon=30).train(years=8)`.

> **The honesty contract:** the "expected hit-rate" shown in the UI is the measured
> OOS precision of the fired bucket. If the model isn't trained, the signal returns a
> clearly-flagged `NEUTRAL`.

---

## 🖥️ Two App Surfaces

Both surfaces share the same `services/` + `schemas/` contract, so every feature stays
consistent across them.

| Surface | Stack | Status | Run |
|---------|-------|--------|-----|
| **Web app (primary)** | Next.js 15 (App Router, React 19) → FastAPI backend | The main app used for testing and daily use | `run.bat` |
| **Streamlit app (secondary)** | `app.py` (single-file Streamlit dashboard) | Kept working; not the primary path | `streamlit run app.py` |

---

## 🎯 What Makes This Novel

### 1. Calibrated, conviction-gated swing signal
A long/neutral 30-day directional model whose displayed confidence is its *measured*
out-of-sample precision, not an in-sample probability. It abstains 94% of the time and
only speaks when the setup clears a tuned conviction threshold. (See
[Honest Results](#-honest-results-real-numbers).)

### 2. Bayesian Dynamic Fusion Framework
Three specialized networks, each tracking its own prediction uncertainty (σ²). Weights
shift every cycle toward the most-accurate expert:

```
w_i = exp(-σ²_i) / Σ exp(-σ²_j)
```

| Expert | Architecture | Input |
|--------|-------------|-------|
| Technical Expert | GRU (128→64→32) | OHLCV-derived features |
| Sentiment Expert | Dense NN (64→32→16) | multi-source sentiment features |
| Volatility Expert | MLP | India VIX + stock-volatility features |

### 3. 27-Feature Hybrid Model (XGBoost + GRU)
A 50/50 ensemble of XGBoost (100 trees, `max_depth=3`) and a GRU (128→64→32) over 27
engineered features across Core Technical (5), Enhanced Technical (9), Advanced
Technical (4), Sentiment (3), Institutional FII/DII (4), and Volatility (2). All
features are stationary (log returns, oscillators, ratios) to avoid non-stationarity.

### 4. 4-Source Sentiment with Temporal & Event Weighting
RSS feeds (30%), NewsAPI (25%), Reddit (25%), Google Trends (20%), with temporal decay
`w = exp(-0.5 × days_old)` and event multipliers (earnings 2.0×, regulatory 1.8×,
dividend 1.5×, management 1.3×). NLP via **DistilRoBERTa-Financial** (zero-shot). An
optional **LLM sentiment-alpha** path (`services/llm_sentiment_alpha.py`) and a **v2
HF-hosted ensemble** (FinBERT + base learners + LR stacker) extend this.

### 5. Hurst-Exponent Regime Detection
R/S rescaled-range analysis classifies trending (H > 0.55), mean-reverting (H < 0.45),
or random-walk (H ≈ 0.50) regimes to modulate trend-following confidence.

### 6. Multi-Timeframe Chart Pattern Detection
ZigZag-based detection (3% reversal, O(n)) with a SciPy `argrelextrema` fallback;
detects double tops/bottoms, head-and-shoulders, channels, flags, triangles, wedges,
with volume confirmation. Optional Roboflow Vision API for image-based classification.

### 7. FII/DII Flows with 6-Source Fallback Chain
`NSE API → nselib → MoneyControl → Trendlyne → Gemini AI parsing → manual input`.

### 8. Operational backbone
A **paper-trading pipeline**, a self-resolving **accuracy ledger** (predictions are
later marked Hit/Miss against real prices), a versioned **model registry** with atomic
promotion, **Celery** background jobs, and **observability** (structlog, Sentry, OTLP
tracing) — all gated so the app runs free-tier and DB-less locally.

---

## 🔧 Architecture

```
                         ┌──────────────────────────────┐
        Browser  ───────►│  Next.js 15 (frontend/)       │
                         │  • Supabase SSR auth/watchlists│
                         │  • typed fetch client          │
                         └───────────────┬────────────────┘
                                         │  /api/v1/*
                                         ▼
        ┌────────────────────────────────────────────────────────────┐
        │  FastAPI  (api/main.py)                                      │
        │  routers: stocks · sentiment · analysis · predict · backtest │
        │           jobs · accuracy · models · ws · auth · watchlists  │
        │           · alerts · health                                  │
        └───────────────┬───────────────────────────┬─────────────────┘
                        │                            │
                        ▼                            ▼
        ┌──────────────────────────┐    ┌──────────────────────────────┐
        │  services/  (the contract)│    │  workers/  (Celery + beat)    │
        │  prediction · backtest    │    │  paper-trade & ledger jobs,   │
        │  sentiment · technicals   │    │  tick publisher/broker        │
        │  paper_trade · ledger ... │    └──────────────────────────────┘
        └───────────┬───────────────┘
                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ models/  hybrid · directional_signal · fusion · experts ·        │
   │          cross_sectional · portfolio · backtester · registry     │
   │ data/    yfinance · FII/DII · VIX · multi-sentiment · options ·   │
   │          intraday · upstox · macro                                │
   │ db/      Supabase (RLS) + local SQLite ledger fallback            │
   └────────────────────────────────────────────────────────────────┘

   Same services/ + schemas/ layer also powers the Streamlit app (app.py).
```

Hard invariants the backend inherits:
- **No `streamlit` imports** inside `services/` / `schemas/` — those are the shared contract.
- **Secrets only via `config.settings._get_secret()`** (Streamlit secrets → `.env` → env vars).
- **Free-tier first** — every endpoint works without paid keys; the v2 ensemble (`HF_TOKEN`) `503`s cleanly when absent, and auth-only endpoints `401` in dev when `SUPABASE_JWT_SECRET` is unset.
- **Local no-DB mode** — the frontend auto-bypasses Supabase when `NEXT_PUBLIC_SUPABASE_*` are absent (`frontend/utils/supabase/stub.ts`); the backend runs on a local SQLite ledger.

---

## 📊 Data Sources

| Source | Type | Data |
|--------|------|------|
| Yahoo Finance (yfinance) | Market data | OHLCV, fundamentals (P/E, ROE, debt, market cap) |
| NSE India | Institutional | FII/DII net flows (daily) |
| India VIX (^INDIAVIX) | Macro | Market fear/volatility index (synthetic fallback) |
| Upstox | Market data | Instruments + intraday/option-chain (optional) |
| NewsAPI | News | Global financial news |
| RSS Feeds (6 sources) | News | Moneycontrol, ET, LiveMint, Business Standard, Google News |
| Reddit (PRAW) | Social | 4 Indian-market subreddits |
| Google Trends (pytrends) | Retail sentiment | Search-volume proxy |
| Roboflow Vision API | Pattern | AI chart-pattern classification (optional) |

---

## 📁 Project Structure

```
finance/
├── api/                        # FastAPI backend (primary surface)
│   ├── main.py                 # app factory, CORS, middleware, router wiring
│   ├── routers/                # stocks, sentiment, analysis, predict, backtest,
│   │                           #   jobs, accuracy, models, ws, auth, watchlists, alerts, health
│   ├── observability/          # structlog, Sentry, OTLP tracing, request-id middleware
│   ├── auth.py · deps.py · jobs.py · rate_limit.py
│
├── frontend/                   # Next.js 15 web app (primary surface) — see frontend/README.md
│   ├── app/                    # App Router pages: dashboard, stock/[ticker] tabs, accuracy, login
│   ├── components/ · hooks/ · lib/ · utils/supabase/
│
├── services/                   # Shared service layer (the contract for BOTH surfaces)
│   ├── prediction_service.py · backtest_service.py · sentiment_service.py
│   ├── paper_trade_service.py · ledger_service.py · ledger_backfill.py
│   ├── technicals_service.py · pattern_service.py · stock_service.py
│   ├── v2_ensemble_service.py · llm_sentiment_alpha.py · live_monitor.py
│
├── schemas/                    # Pydantic DTOs (prediction, backtest, sentiment, ledger, …)
│
├── models/                     # ML models
│   ├── hybrid_model.py         # 27-feature XGBoost + GRU ensemble (calibrated)
│   ├── directional_signal.py   # 30-day calibrated swing signal  ⭐
│   ├── fusion_framework.py     # Bayesian multi-expert fusion
│   ├── technical_expert.py · sentiment_expert.py · volatility_expert.py
│   ├── cross_sectional_trainer.py · portfolio_constructor.py · signal_combiner.py
│   ├── visual_analyst.py · backtester.py · optimizer.py · registry.py
│   ├── nse_costs.py · transaction_costs.py · alpha_signals.py
│   └── saved/                  # persisted artifacts (directional_signal.pkl, cross_sectional_*)
│
├── data/                       # data sources (yfinance, FII/DII, VIX, sentiment, options, intraday, upstox, macro)
│   └── ledger/                 # local SQLite ledger + LLM sentiment cache
│
├── db/                         # Supabase client + RLS schema + SQLite ledger + alerts
├── workers/                    # Celery app, beat schedule, tasks, tick publisher/broker
├── ui/                         # Streamlit-only: charts.py, ai_analysis.py
├── utils/                      # technical_indicators, risk_manager, roboflow_client
├── config/                     # settings.py (secret resolution), instruments
├── scripts/                    # run_paper_trade, promote_model, populate_instruments
├── models-registry/            # versioned model registry (active.json + v1/) — see its README
├── tests/                      # pytest suite: api/ services/ models/ schemas/ data/ workers/ scripts/
├── docs/figures/               # research-paper figures + app screenshots
│
├── app.py                      # Streamlit app (secondary surface)
├── run.bat                     # dev launcher (backend :8000 + frontend :3000)
├── run_backtest.py             # standalone backtest entry point
├── requirements.txt            # Python deps (target: Python 3.11)
├── Procfile · railway.toml     # deployment (web/worker/beat/ticks)
└── indian_stocks.csv           # NSE symbol list
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11** (deps pin `numpy<2` / `tensorflow-cpu<2.17`; 3.13 has no compatible wheels)
- **Node.js 18+** (for the Next.js frontend)

### 1. Backend env (one-time)
```bat
"%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

### 2. Frontend env (one-time)
```bash
cd frontend
npm install
cp .env.local.example .env.local   # optional — see frontend/README.md
```

### 3. Run the primary app (FastAPI + Next.js)
```bat
run.bat
```
Backend → http://localhost:8000 (Swagger at `/docs`) · Frontend → http://localhost:3000.
Without `NEXT_PUBLIC_SUPABASE_*` set, the frontend runs in local no-DB mode (fake
signed-in user, auth gate skipped) so you can use it with zero external services.

### 4. (Optional) Run the secondary Streamlit app
```bash
streamlit run app.py
```

### API keys (all optional — the platform degrades gracefully)
Create `.env` in the project root:
```env
GEMINI_API_KEY=...          # AI commentary
NEWS_API_KEY=...            # enhanced news sentiment
REDDIT_CLIENT_ID=...        # Reddit sentiment
REDDIT_CLIENT_SECRET=...
ANTHROPIC_API_KEY=...       # in-app Claude analysis + LLM sentiment alpha
HF_TOKEN=...                # v2 hosted ensemble (else /sentiment/v2 returns 503)
ROBOFLOW_API_KEY=...        # vision pattern detection
SUPABASE_JWT_SECRET=...     # enable auth-gated endpoints
REDIS_URL=...               # enable Celery background jobs

# Trade Desk (P5) — live order placement via Zerodha Kite. ALL THREE required
# for live mode; any missing → orders are simulated (dry-run). Also requires
# `pip install kiteconnect`. The access token expires daily (SEBI rule).
KITE_API_KEY=...
KITE_ACCESS_TOKEN=...
LIVE_TRADING=1
TRADE_CAPITAL_PER_SLOT=25000  # default ₹ per approved position
```

---

## 🌐 API Surface

Base prefix: `/api/v1` (full interactive docs at `/docs`).

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` · `/readyz` | liveness / readiness |
| `GET /me` | current Supabase user |
| `GET /stocks` · `/stocks/{ticker}` | search + OHLCV + fundamentals |
| `GET /stocks/{ticker}/sentiment[/v2]` | sentiment + v2 ensemble |
| `POST /stocks/{ticker}/predict` | enqueue prediction → `job_id` (includes swing signal) |
| `GET /swing/scan` | run the 30-day swing signal across the universe (cached ≤30 min; `?refresh=1` to force) |
| `GET /trade/broker/status` | active execution backend (dry-run vs live Kite) |
| `POST /trade/proposals` · `GET /trade/proposals` | analyze tickers → stage sized order proposals |
| `POST /trade/proposals/{id}/approve` · `/reject` | the ONLY execution path — per-order human approval |
| `POST /stocks/{ticker}/backtest` | enqueue backtest → `job_id` |
| `GET /jobs/{id}` | poll job status / result |
| `GET /accuracy[/recent]` · `POST /accuracy/resolve` | ledger rollups + resolution |
| `GET /models/active` | active model version |
| `GET/POST /watchlists` · `/alerts` | per-user CRUD (RLS) |
| `WS /ws/prices?tickers=` | live price stream |

Regenerate the typed frontend client after backend changes:
`cd frontend && npm run codegen` (reads `http://localhost:8000/openapi.json`).

---

## 🔁 Daily Operations (Paper Trading & Accuracy)

If you are **not** running the Celery job queue, run these from the project root in the
`.venv` (see `daily_commands.txt` for the full guide):

```bash
# Near market close (~15:20 IST): predict + log paper trades, mark open positions
# Trades the 30-day swing signal when trained (the validated edge): bracketless
# entry, 10% disaster stop, time exit at the horizon (~42 calendar days).
# Use --engine nextday for the legacy next-day comparison book.
python -m scripts.run_paper_trade            # add --dry-run to preview, --tickers to scope

# After close (~16:00 IST): resolve past predictions into Hit/Miss against real prices
python -m services.ledger_backfill           # or: curl -X POST localhost:8000/api/v1/accuracy/resolve
```

With `REDIS_URL` set, Celery beat automates this (`run_paper_trade` ~15:45 IST,
`ledger_backfill` ~16:30 IST) — see `Procfile` for `web` / `worker` / `beat` / `ticks`.

### Trade Desk (P5) — approve-before-execute broker layer

The dashboard's **Trade Desk** panel turns analysis into sized order proposals
(qty, entry, 10% disaster stop, exit-by date) and waits for an explicit
per-order approval. The approve endpoint is the only code path that talks to a
broker, and it is **simulated (dry-run) by default** — live placement requires
`KITE_API_KEY` + `KITE_ACCESS_TOKEN` + `LIVE_TRADING=1`. In live mode the
protective stop is a server-side Zerodha **GTT** so it survives the app being
offline. Every approved order is also logged to the accuracy ledger
(`model_version=swing30d-live[...]`), so live vs paper vs backtest results are
directly comparable from the same table.

---

## ☁️ Deployment

- **FastAPI backend** → Railway (`railway.toml`, NIXPACKS, healthcheck `/api/v1/healthz`).
- **Streamlit app** → Streamlit Cloud (Python 3.11, `packages.txt` for `libgomp1` etc.).
- **Frontend** → any Next.js host (Vercel-style); point `NEXT_PUBLIC_API_URL` at the backend.
- **Model registry** → local dir by default; set `MODEL_REGISTRY_URI=s3://…` (+ `S3_ENDPOINT_URL` for R2) to switch backends. See `models-registry/README.md`.

CI: `.github/workflows/ci.yml` runs lint + schema/service smoke tests on Python 3.11
(heavy ML deps are exercised at deploy time, not in CI).

---

## ⚠️ Known Limitations

| Limitation | Detail |
|-----------|--------|
| **No next-day single-name edge** | By design we report ~0.50; the tradable signal is the 30-day swing model, not next-day direction. |
| **Swing signal is selective** | Fires only ~5.6% of the time and is long/neutral only — it abstains far more than it acts. |
| **EOD-first** | Built around end-of-day prices; intraday/options paths are optional and secondary. |
| **NSE source reliability** | FII/DII and VIX rely on fallback chains when NSE is down. |
| **NLP zero-shot** | DistilRoBERTa-Financial is not fine-tuned on Indian-market language. |
| **CPU training** | GRU/ensemble training is slow on CPU; retraining is a periodic offline job. |

---

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.** Not financial advice. Past
performance does not guarantee future results. Always consult a SEBI-registered advisor
before investing.

---

## 📄 License

MIT License — free for personal and research use.

---

## 🙏 Credits

Market data: Yahoo Finance, NSE India, Upstox · Sentiment NLP:
[DistilRoBERTa-Financial](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis),
FinBERT · AI commentary: Google Gemini, Anthropic Claude · Patterns: SciPy, Roboflow ·
Backtesting: custom vectorized engine with NSE transaction-cost model.

---

**Version:** 0.2.x (API) · **Architecture:** Next.js + FastAPI (primary) + Streamlit (secondary) · **Last updated:** June 2026
