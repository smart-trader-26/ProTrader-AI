# ProTrader AI — Master Task List

**Two parallel tracks:**
1. **IMPROVEMENTS** — make predictions more accurate, add real confidence
   intervals, build an accuracy-testing framework, integrate the v2
   trained Colab model, add intraday + option-chain + macro features.
2. **MIGRATION** — move from single-file Streamlit to a production
   Next.js + FastAPI + Postgres + Redis stack.

Tracks **interleave**. Phase 0 (refactor) unblocks both — do it first.
After that, every IMP task can land into either the Streamlit UI or the
new FastAPI backend with one line change.

Each task lists: **scope · effort · dependency · acceptance criterion.**

---

## PHASE 0 — Foundation refactor (DO FIRST)

> Goal: stop breaking when we add things. Extract pure logic from
> `app.py` so both Streamlit and FastAPI call the same code.

| # | Task | Effort | Dep |
|---|------|--------|-----|
| 0.1 | Create `services/` directory with `stock_service.py`, `sentiment_service.py`, `prediction_service.py`, `backtest_service.py`, `pattern_service.py` | 1 day | — |
| 0.2 | Move all analytical functions out of `app.py` into `services/`. `app.py` only does Streamlit rendering + service calls. | 1 day | 0.1 |
| 0.3 | Create `schemas/` with Pydantic DTOs (`StockData`, `Prediction`, `SentimentResult`, etc.) — typed contracts every service speaks | 0.5 day | 0.1 |
| 0.4 | Add `pytest` + `tests/services/` with smoke tests per service (mock yfinance) | 0.5 day | 0.2 |
| 0.5 | Set up GitHub Actions: lint (ruff) + type-check (mypy) + tests on PR | 0.5 day | 0.4 |

**Acceptance:** `streamlit run app.py` works identically; CI is green;
`from services.prediction_service import predict` works in a notebook.

---

## TRACK A — IMPROVEMENTS

### A1 · Calibrate the existing model (quick wins, 2-3 days)

| # | Task | Effort | Dep |
|---|------|--------|-----|
| A1.1 | **Verify isotonic calibration is fit on OOF data, not in-sample.** Currently `hybrid_model.py` does isotonic regression — make sure it's fit on the walk-forward holdout, not the train fold. | 0.5 day | 0 |
| A1.2 | Add **calibration plot** (predicted prob vs. actual hit rate) and **ECE (Expected Calibration Error)** to the dashboard. Should be ≤ 5% for a trustworthy model. | 0.5 day | A1.1 |
| A1.3 | **Cache the FinBERT model resource** (`@st.cache_resource` instead of module global) — saves the 30 s cold-start on every Streamlit Cloud reboot. | 0.5 hr | 0 |
| A1.4 | **Cache yfinance per (ticker, date_range)** in Redis (or `st.cache_data` for now). Currently every rerun re-downloads. | 0.5 day | 0 |
| A1.5 | **Calibrate confidence threshold per-ticker.** Some tickers need 65% prob to be "buy", others 55%. Compute from holdout AUC per ticker. | 1 day | A1.1 |

### A2 · Integrate the v2 trained Colab model (2-3 days)

> **NOT** as a replacement — as **one more input** to the existing
> Ridge meta-stacker. v2's bullish-probability becomes a feature column
> alongside XGB / LGBM / CatBoost / GRU. The stacker auto-learns its
> weight. No conflict with the 8-feature SentimentExpertModel.

| # | Task | Effort | Dep |
|---|------|--------|-----|
| A2.1 | Push v2 sentiment model dir to **HuggingFace Hub** (free, 5 GB free storage, fast CDN). Repo: `your-user/protrader-finbert`. | 1 hr | — |
| A2.2 | Push v2 `master_*.pkl` files to git LFS or HuggingFace as a separate "models" repo. | 1 hr | — |
| A2.3 | Create `services/v2_ensemble_service.py` with `predict_v2(ticker, news_df) -> {prob_up, model_breakdown}`. Loads `.pkl`s from HF on first call, caches in `/tmp`. | 0.5 day | 0 |
| A2.4 | Add `v2_prob_up` as a column to the meta-stacker training matrix in `hybrid_model.py`. | 0.5 day | A2.3 |
| A2.5 | Retrain meta-stacker with the new column on existing OOF predictions; verify Sharpe / accuracy doesn't drop. | 0.5 day | A2.4 |
| A2.6 | Add a "Sentiment Ensemble (v2)" panel under the Sentiment tab showing v2's standalone probability, model breakdown bar chart, and category distribution. | 0.5 day | A2.3 |

### A3 · Intraday data, free (3-4 days)

> Real-time without paid keys. Three-source ladder: yfinance 1-min →
> Yahoo unofficial WebSocket → Dhan/Upstox API (free, just KYC).

| # | Task | Effort | Dep |
|---|------|--------|-----|
| A3.1 | Add `data/intraday.py`: `get_intraday_bars(ticker, interval='1m', period='5d')` via yfinance. 1-min available for last 7 days, 5-min for 60 days. | 0.5 day | 0 |
| A3.2 | Add **auto-refresh widget** to dashboard during NSE hours (09:15–15:30 IST). Use `streamlit-autorefresh` (free). Default 60 s interval. | 0.5 day | A3.1 |
| A3.3 | Build `data/yahoo_ws.py`: connect to `wss://streamer.finance.yahoo.com/?version=2`, subscribe to tickers, ~30 s latency, free. | 1 day | 0 |
| A3.4 | Sign up for **Dhan API** (free, KYC required — store key in `.env`). Add `data/dhan_client.py` with WebSocket tick stream. | 1 day | A3.3 |
| A3.5 | Add intraday-aware features: 5-min RSI, VWAP distance, opening-range breakout flag. Feed into hybrid model only when intraday data is available. | 1 day | A3.1 |

### A4 · Option chain features (1-2 days)

| # | Task | Effort | Dep |
|---|------|--------|-----|
| A4.1 | Add `data/option_chain.py` using NSE public endpoint `https://www.nseindia.com/api/option-chain-equities?symbol=X`. Handle the cookie/User-Agent dance (NSE returns 401 without it). | 1 day | 0 |
| A4.2 | Extract features per ticker per day: **PCR (put-call ratio), max-pain, IV-skew, OI-change top-3 strikes, weighted IV**. | 0.5 day | A4.1 |
| A4.3 | Add to hybrid model feature matrix; retrain; A/B compare accuracy with/without options features. | 0.5 day | A4.2 |

### A5 · Macro features (1 day)

| # | Task | Effort | Dep |
|---|------|--------|-----|
| A5.1 | Add `data/macro.py` pulling free yfinance series: `INR=X` (USD/INR), `CL=F` (crude), `^TNX` (US 10Y), `GC=F` (gold), `^GSPC` (S&P 500), `^VIX` (US VIX). | 0.5 day | 0 |
| A5.2 | Add macro features (1-day, 5-day pct changes) to hybrid feature matrix. | 0.5 day | A5.1 |
| A5.3 | Retrain + A/B compare. | inline | — |

### A6 · Probabilistic forecasting (3-5 days)

> Replace point estimates with prediction intervals — gives users a real
> "discrete confidence" output. Two complementary approaches.

| # | Task | Effort | Dep |
|---|------|--------|-----|
| A6.1 | **Quantile regression** version of XGB: train 3 models (`quantile_alpha=0.1, 0.5, 0.9`). Outputs `(P10, P50, P90)` for tomorrow's return. | 1 day | 0 |
| A6.2 | **Conformal prediction wrapper** around the existing point estimator: distribution-free 90% intervals with coverage guarantee. Library: `mapie` (free). | 1 day | A6.1 |
| A6.3 | Update prediction response schema: replace single `pred_price` with `{pred_price, ci_low, ci_high, confidence_level}`. | 0.5 day | A6.2 |
| A6.4 | Render in UI as a shaded band on the price chart (instead of just a dotted line). | 0.5 day | A6.3 |

### A7 · Accuracy ledger (CRITICAL — 2 days)

> The single most important task in this whole list. Without it, you
> cannot prove the model works.

| # | Task | Effort | Dep |
|---|------|--------|-----|
| A7.1 | Create `predictions` table (SQLite for now, Postgres in Phase 1): `ticker, made_at, target_date, pred_dir, pred_price, ci_low, ci_high, confidence, actual_price (NULL), hit (NULL), model_version` | 0.5 day | 0 |
| A7.2 | After every `predict()` call, append a row to `predictions`. | 0.5 day | A7.1 |
| A7.3 | Daily Celery / cron job at 16:00 IST: for every row where `target_date == yesterday`, fill `actual_price` and `hit` from yfinance. | 0.5 day | A7.1 |
| A7.4 | Add **Accuracy** tab to the UI showing rolling 7d/30d/90d directional accuracy per ticker, calibration plot, ECE, Brier score. | 0.5 day | A7.3 |
| A7.5 | Add `accuracy.historical_30d` to every `predict()` API response so users see "model was right 62% of the time on this ticker over 30d." | inline | A7.3 |

### A8 · Backtest framework (2-3 days)

| # | Task | Effort | Dep |
|---|------|--------|-----|
| A8.1 | Audit `models/backtester.py` — verify costs are realistic: brokerage 0.03%, STT 0.025% on sell, slippage 5 bps, ₹20 minimum charge, GST on brokerage. | 0.5 day | 0 |
| A8.2 | Add metrics: **Sharpe, Sortino, Calmar, Max DD, Profit Factor, Win Rate, Avg Win / Avg Loss, Expectancy.** | 0.5 day | A8.1 |
| A8.3 | Add **Diebold-Mariano test** vs. buy-and-hold (statistical significance of outperformance, p-value). | 0.5 day | A8.2 |
| A8.4 | **Hard 3-way split**: train 2018-2022, validate 2023, **holdout 2024+ never touched until the very end**. Add a "holdout report" button that reveals holdout metrics on demand. | 1 day | A8.2 |

### A9 · Live paper-trading loop (1 week, do AFTER A7)

| # | Task | Effort | Dep |
|---|------|--------|-----|
| A9.1 | Sign up Dhan / Upstox developer account (free, KYC). Get sandbox API key. | 1 hr | A3.4 |
| A9.2 | Build `services/paper_trade_service.py` that places virtual orders via the sandbox per the model's signals. | 1 day | A7.5 |
| A9.3 | Run for 30 trading days. Compare paper P&L to backtest prediction. Backtest typically overstates by 20-50%; this is your reality check. | 30 days elapsed | A9.2 |

---

## TRACK B — MIGRATION (see [MIGRATION.md](MIGRATION.md) for full architecture)

### B1 · FastAPI backend (1-2 weeks)

| # | Task | Effort | Dep |
|---|------|--------|-----|
| B1.1 | `pip install fastapi uvicorn pydantic`. Create `api/main.py` with app factory. | 0.5 day | 0 |
| B1.2 | `api/routers/stocks.py` — endpoints: `GET /stocks?q=`, `GET /stocks/{ticker}/ohlcv`, `GET /stocks/{ticker}/fundamentals` | 1 day | B1.1, 0.2 |
| B1.3 | `api/routers/sentiment.py` — `GET /stocks/{ticker}/sentiment` (calls `services/sentiment_service`) | 0.5 day | B1.1 |
| B1.4 | `api/routers/predict.py` — `POST /stocks/{ticker}/predict` returns `job_id` immediately; `GET /jobs/{id}` for polling | 1 day | B1.1, A7.1 |
| B1.5 | `api/routers/backtest.py` — async backtest with same job pattern | 0.5 day | B1.4 |
| B1.6 | OpenAPI Swagger UI at `/docs`; generate TypeScript client with `openapi-typescript` for the future Next.js frontend | 0.5 day | B1.5 |
| B1.7 | Deploy FastAPI to Railway / Render alongside the Streamlit app — both running | 0.5 day | B1.6 |

### B2 · Background workers (1 week)

| # | Task | Effort | Dep |
|---|------|--------|-----|
| B2.1 | `pip install celery redis`. Configure Redis (Upstash free tier). | 0.5 day | B1.1 |
| B2.2 | Move slow tasks to Celery: news fetch every 5 min, model retrain weekly, alert eval every 1 min, backtest jobs. | 2 days | B2.1 |
| B2.3 | `celery beat` schedule. | 0.5 day | B2.2 |

### B3 · PostgreSQL + Auth (1 week)

| # | Task | Effort | Dep |
|---|------|--------|-----|
| B3.1 | Sign up Supabase (free Postgres + Auth). | 0.5 hr | — |
| B3.2 | SQLAlchemy + Alembic. Migrate `predictions` table from SQLite → Postgres. | 1 day | A7.1, B3.1 |
| B3.3 | Add tables: `users, tickers, watchlists, alerts, predictions, backtests` per [MIGRATION.md § Phase 4](MIGRATION.md). | 1 day | B3.2 |
| B3.4 | Integrate Clerk (or Supabase Auth) — JWT verified on every FastAPI request via dependency. | 1 day | B3.3 |
| B3.5 | Per-user rate limiting via `slowapi` middleware. | 0.5 day | B3.4 |

### B4 · Next.js frontend (2-3 weeks)

| # | Task | Effort | Dep |
|---|------|--------|-----|
| B4.1 | `npx create-next-app@latest protrader-web --typescript --tailwind --app`. Add `shadcn/ui`. | 0.5 day | — |
| B4.2 | Set up TanStack Query + typed API client from B1.6 OpenAPI schema. | 0.5 day | B4.1 |
| B4.3 | Pages: `/`, `/dashboard/[ticker]`, `/watchlist`, `/sign-in`. | 3 days | B4.2 |
| B4.4 | `components/PriceChart.tsx` using **TradingView Lightweight Charts** (free, MIT, 45 KB). | 1 day | B4.3 |
| B4.5 | `components/SentimentPanel.tsx`, `PredictionCard.tsx`, `BacktestResults.tsx` (mostly Plotly.js for the heavy charts). | 2 days | B4.4 |
| B4.6 | Server-Sent Events hook for analysis progress (no more progress bar polling). | 1 day | B1.4 |
| B4.7 | WebSocket hook for live prices (subscribes to FastAPI `/ws/prices`). | 1 day | A3.4 |
| B4.8 | Auth pages wired to Clerk; protected routes redirect when not signed in. | 1 day | B3.4 |
| B4.9 | Deploy frontend to **Vercel**. Backend URL via env var. | 0.5 day | B4.8, B1.7 |

### B5 · Real-time tick layer (1 week)

| # | Task | Effort | Dep |
|---|------|--------|-----|
| B5.1 | Single Celery worker maintains Dhan / Upstox WebSocket connection. | 1 day | A3.4, B2.1 |
| B5.2 | Worker fans ticks out via Redis Pub/Sub channel `ticks:{symbol}`. | 0.5 day | B5.1 |
| B5.3 | FastAPI `/ws/prices?tickers=...` subscribes to Redis Pub/Sub, forwards to browser. | 0.5 day | B5.2 |
| B5.4 | Frontend WebSocket hook (B4.7) consumes the stream; live-update price + bid/ask. | done in B4.7 | B5.3 |

### B6 · Model serving (3-5 days)

| # | Task | Effort | Dep |
|---|------|--------|-----|
| B6.1 | **Decision:** in-process load (1 GB RAM) vs. dedicated BentoML/Triton server. Start in-process; switch only if latency demands it. | 0.5 day | A2.3 |
| B6.2 | Model registry on R2 / S3: versioned paths `models/v{n}/master_xgb.pkl`. Workers load on startup. | 1 day | A2.2 |
| B6.3 | Add `/api/v1/models/active` endpoint returning current model version + train date so the UI can display "Model v3, trained 2026-04-12". | 0.5 day | B6.2 |
| B6.4 | (Optional) MLflow for experiment tracking once you train more than 1×/month. | 1 day | B6.2 |

### B7 · Observability + payments (1 week)

| # | Task | Effort | Dep |
|---|------|--------|-----|
| B7.1 | **Sentry** for FE + BE error tracking (free tier 5K events/mo). | 0.5 day | B1.7, B4.9 |
| B7.2 | **PostHog** for product analytics (free tier 1M events/mo). | 0.5 day | B4.9 |
| B7.3 | **Stripe** Checkout + customer portal. Free tier = 3 stocks, Pro ₹499/mo = unlimited + alerts + API access. | 2 days | B3.4 |
| B7.4 | GitHub Actions: build + deploy on merge to main (Vercel auto-deploys; Railway needs a webhook). | 0.5 day | — |

---

## Recommended order (calendar week)

| Week | Focus |
|------|-------|
| 1 | Phase 0 (refactor) + A1 (calibration + caching) + tf-keras deploy fix verified live |
| 2 | A2 (v2 model integration) + A7 (accuracy ledger SQLite version) |
| 3 | A3 (intraday yfinance + auto-refresh) + A5 (macro features) |
| 4 | A4 (option chain) + A6 (quantile / conformal intervals) + A8 (backtest hardening) |
| 5 | B1 (FastAPI backend stand-up) + B2 (Celery workers) |
| 6 | B3 (Postgres + auth) + B6 (model registry) |
| 7-9 | B4 (Next.js frontend) |
| 10 | B5 (real-time tick layer) + A9 (paper-trading loop start) |
| 11 | B7 (observability + payments) |
| 12 | A9 ongoing — 30-day paper-trade verdict |

**Hard rule:** do not skip A7 (accuracy ledger). Without it the entire
product is unfalsifiable. Build it first, then *every* later improvement
has a real, measurable, per-ticker accuracy delta to point to.

---

## Single-line acceptance criterion for the whole project

> *On a 90-day rolling window of paper-trading on 50 Nifty stocks, the
> model's average directional accuracy is ≥ 58% with ECE ≤ 5%, and the
> backtest Sharpe is ≥ 1.2 with Diebold-Mariano p < 0.05 vs.
> buy-and-hold.*

If that's true, the product is real. If not, every claim is
hand-waving — keep iterating before adding payments.
