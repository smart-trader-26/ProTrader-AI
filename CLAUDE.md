# CLAUDE.md — Session-Start Briefing

> **Read this first every new session.** It's the running ledger of what's done,
> what's next, and the invariants you can't break. Update it as you ship work —
> treat it like a build log, not a spec.

---

## 1. What this project is

**ProTrader AI** — a Streamlit app that predicts Indian stock prices using:
- yfinance for OHLCV
- RSS / NewsAPI / Reddit for headlines
- FinBERT (HuggingFace) for sentiment
- An ensemble (LR + RF + XGB + LGBM + meta) for direction & price
- Plotly for charts

The end goal (per the user): turn this into a real product with a Next.js
frontend + FastAPI backend, while progressively raising prediction accuracy
to ≥58% directional with ECE ≤5% and Sharpe ≥1.2 on paper trading.

The two parallel tracks for getting there are documented in
[docs/TASKS.md](docs/TASKS.md):
- **Track A — Improvements** (A1 calibration → A9 paper trading)
- **Track B — Migration** (B1 FastAPI → B7 observability + payments)

---

## 2. Hard invariants — do not break

1. **HuggingFace pipelines must use `framework="pt"`.** TF/Keras 3 breaks the
   Streamlit Cloud build. See [data/news_sentiment.py](data/news_sentiment.py)
   and [data/multi_sentiment.py](data/multi_sentiment.py). If you add another
   `pipeline()` call anywhere, force PyTorch.
2. **All secrets go through `config.settings._get_secret()`.** Never read
   `os.environ` or `st.secrets` directly from feature code.
3. **Default to free APIs.** Anything new must work on a zero-key install
   (yfinance + RSS), with paid/keyed sources as opt-in upgrades. See
   [docs/API_KEYS.md](docs/API_KEYS.md).
4. **Don't mock data sources in tests.** Real network → real bugs caught.
   Skip cleanly when offline; never assert on mocked yfinance output.
5. **Never bypass git hooks** (`--no-verify`) or `pip install --break-system-packages`.
6. **The v2 Colab model is integrated as ONE FEATURE into the meta-stacker**,
   not as a wholesale replacement of `SentimentExpertModel`. See A2 in TASKS.

---

## 3. Repo layout (only the files that matter)

```
finance/
├── app.py                       ← Streamlit entry point
├── requirements.txt             ← pinned for Python 3.11 + Streamlit Cloud
├── runtime.txt                  ← python-3.11
├── packages.txt                 ← apt deps for cloud (build-essential, libgomp1)
├── .streamlit/
│   ├── config.toml              ← (do NOT add enableCORS=false — conflicts w/ XSRF)
│   └── secrets.toml.example     ← template; real secrets.toml is gitignored
├── config/
│   └── settings.py              ← _get_secret() chain: st.secrets → .env → env → ""
├── data/
│   ├── news_sentiment.py        ← FinBERT + 6-cat classifier + keyword overrides
│   ├── multi_sentiment.py       ← multi-source aggregator (delegates to news_sentiment)
│   └── ...                      ← stock_data.py, indicators.py, etc.
├── models/                      ← fusion_framework + experts (technical, sentiment, vol)
├── ui/                          ← Streamlit page modules
├── sentiment_analysis_v2.py     ← Colab-only training script (NOT imported in app)
├── api/                         ← FastAPI backend (B1) — main.py + routers/
│   ├── main.py                  ← app factory, lifespan, CORS, router includes
│   ├── jobs.py                  ← in-process JobStore (Celery placeholder, B2)
│   ├── deps.py                  ← FastAPI dependencies (override-friendly)
│   └── routers/                 ← stocks, sentiment, predict, backtest, jobs,
│                                  accuracy, models, ws, health
├── Procfile, railway.toml       ← FastAPI deploy stubs (B1.7)
└── docs/
    ├── TASKS.md                 ← the unified roadmap (A1–A9 + B1–B7)
    ├── DEPLOYMENT.md            ← Streamlit Cloud how-to
    ├── API.md                   ← FastAPI public contract (B1)
    ├── IMPROVEMENTS.md          ← rationale doc (read-only context)
    ├── MIGRATION.md             ← Next.js + FastAPI architecture
    └── API_KEYS.md              ← every key + HuggingFace model upload guide
```

---

## 4. Status ledger — keep this current

### Done
- [x] Streamlit Cloud deploy unblocked (Python 3.11, `tensorflow-cpu`, `tf-keras`,
      `framework="pt"` on all pipelines).
- [x] Secret loader (`_get_secret`) — keys flow from st.secrets → .env → env.
- [x] FinBERT pipeline + bullish/bearish keyword overrides + 6-cat classifier
      ([data/news_sentiment.py](data/news_sentiment.py)).
- [x] `multi_sentiment.analyze_text` delegates to `news_sentiment.analyze_sentiment`
      so all paths benefit from keyword overrides.
- [x] Docs written: TASKS, DEPLOYMENT, IMPROVEMENTS, MIGRATION, API_KEYS.
- [x] **PHASE 0.1 / 0.3 / 0.4** — [services/](services/) (5 thin wrappers),
      [schemas/](schemas/) (Pydantic DTOs), [tests/](tests/) (pytest skeleton,
      network-gated smoke tests). Notebook contract verified:
      `from services.prediction_service import predict` imports cleanly.
- [x] **PHASE 0.2** — satisfied implicitly: `app.py` has no analytical code
      beyond one `_hint` UI helper; services now provide the backend seam for
      B1. Full UI cutover (replace `create_hybrid_model` in app.py with
      `services.prediction_service.predict`) deferred to B1 because the UI
      still consumes raw metrics (SHAP dict, weights, etc.) not on the bundle.
- [x] **PHASE 0.5** — [.github/workflows/ci.yml](.github/workflows/ci.yml)
      (ruff + pytest on PR/push), [pyproject.toml](pyproject.toml) with ruff
      config. mypy skipped until types land (post-B1).
- [x] **A1.1** — audited: isotonic is fit on OOF (`_generate_oof_predictions`
      expanding folds), applied to test preds. Assertion added in
      [models/hybrid_model.py](models/hybrid_model.py).
- [x] **A1.2** — `_compute_calibration_report()` + `create_calibration_chart()`;
      ECE + reliability curve rendered in an expander under the SHAP section.
      Trustworthy threshold ≤ 5%.
- [x] **A1.3** — FinBERT via `@st.cache_resource` — deduped the duplicate load
      that `multi_sentiment` used to do in parallel (1 GB RAM win).
- [x] **A1.4** — `@st.cache_data(ttl=900)` on `get_stock_data`; 1800s on
      `get_stock_info` and `get_fundamental_data`.
- [x] **A1.5** — `_compute_threshold_tuning()` learns τ\* via Youden's J on
      holdout, exposes AUC + accuracy\_at\_τ\*; "Bullish Probability" metric
      now reads the per-ticker τ instead of the fixed 55/45 band.
- [x] **A2.1 / A2.2** — v2 ensemble pushed to HuggingFace
      (`EnteiTiger3/protrader-sentiment-v2`): fine-tuned FinBERT +
      `master_{lr,rf,xgb,lgb,stacker}.pkl`. ~340 MB total.
- [x] **A2.3** — [services/v2_ensemble_service.py](services/v2_ensemble_service.py)
      with `predict_v2(ticker, headlines)` → `V2EnsemblePrediction` DTO.
      Lazy-downloads from HF on first call (cached in `~/.cache/huggingface`),
      thread-safe, falls back to weighted-average if stacker pkl missing.
      End-to-end test passes in ~52s (downloads + predicts).
- [x] **A2.6** — "🤖 Sentiment Ensemble (v2)" expander under Tab 6 with:
      stacked consensus metric, base-learner breakdown bar chart (LR/RF/XGB/LGBM),
      category distribution, `_hint` glossary. Gated behind `HF_TOKEN`.
- [x] **A4.1 / A4.2 / A4.3** — [data/option_chain.py](data/option_chain.py) NSE
      cookie-dance fetcher + scalar feature extractor (PCR, max-pain distance,
      ATM IV, IV skew, OI concentration, call/put walls, weighted IV). Wired
      into `create_hybrid_model` as broadcast `opt_*` columns; zeros on failure.
- [x] **A5.1 / A5.2 / A5.3** — [data/macro.py](data/macro.py) yfinance fetcher
      for INR=X / CL=F / ^TNX / GC=F / ^GSPC / ^VIX; 1d + 5d log returns
      (clipped ±0.2 / ±0.4). Left-joined onto feature matrix with ffill.
- [x] **A6.1 / A6.2 / A6.3 / A6.4** — quantile XGB at α=0.1/0.9 + split-conformal
      half-width (finite-sample corrected) inside `create_hybrid_model`; surfaced
      as `conformal_halfwidth` in metrics and rendered as a day-1 interval
      expander + P5/P95 forecast band chart in the UI.
- [x] **A7.1 – A7.5** — [services/ledger_service.py](services/ledger_service.py)
      SQLite ledger with UNIQUE(ticker, made_at, target_date) dedupe, idempotent
      `log_prediction` + `log_from_future_df`, `backfill_actuals` with injectable
      `price_fetcher`, `accuracy_window` (hit rate + Brier + ECE), CLI at
      [services/ledger_backfill.py](services/ledger_backfill.py). Wired into
      `services/prediction_service.predict()` + `app.py` tab1 logger + inline
      30d accuracy badge + Tab 9 "🎯 Accuracy" dashboard.
- [x] **A3.1 / A3.2 / A3.3 / A3.5** — [data/intraday.py](data/intraday.py)
      yfinance bar fetcher with 1m→2m→5m→15m→60m fallback ladder.
      [data/yahoo_ws.py](data/yahoo_ws.py) unofficial Yahoo WS connector
      (varint protobuf decode, auto-reconnect with exponential backoff).
      [data/intraday_features.py](data/intraday_features.py) 5-min RSI (Wilder,
      epsilon-guarded against pure-trend divide-by-zero), session VWAP distance
      (±10% clipped), opening-range-breakout flag. Autorefresh widget gated
      behind `is_nse_market_open()` in app.py. **A3.4 still blocked** on Upstox KYC.
- [x] **A8.1 / A8.2 / A8.3 / A8.4** — [models/nse_costs.py](models/nse_costs.py)
      line-item NSE round-trip cost model (brokerage + STT + exchange + SEBI
      + stamp duty + GST + slippage; ₹20 brokerage cap, DELIVERY vs INTRADAY
      schedules). Wired into `VectorizedBacktester.run_backtest(cost_model=…)`,
      replacing the flat 0.1%. Added Sortino / Calmar / Expectancy to
      `_compute_metrics_from_returns`. Added `diebold_mariano_test(model_err,
      bench_err)` with Harvey small-sample correction. [services/backtest_split.py](services/backtest_split.py)
      3-way train/val/holdout split (2022-12-31 / 2023-12-31 / present) with
      `reveal_holdout(confirm="I have not touched the holdout")` gate.
- [x] **A9.2** — [services/paper_trade_service.py](services/paper_trade_service.py)
      broker-agnostic paper-trading engine. Uses same A7 ledger DB (new
      `paper_fills` + `paper_positions` tables). Fills against injectable
      `fill_source` (default: yfinance live close). Closes on stop/target/
      signal-flip. P&L deducts NSE line-item costs. Ready to swap fill source
      for Upstox sandbox when KYC completes.
- [x] **Tests** — 85 offline tests across ledger / option_chain / macro /
      intraday / nse_costs / backtester metrics / backtest_split / paper_trade /
      api routers. Injected fill sources + tmp_path SQLite keep the suite
      offline; `network` marker gates NSE / yfinance smoke tests.
- [x] **B1.1 – B1.7** — FastAPI app at [api/main.py](api/main.py) (lifespan,
      CORS, OpenAPI). Routers: stocks/sentiment/predict/backtest/jobs/accuracy/
      models/ws under `/api/v1`. Job pattern via [api/jobs.py](api/jobs.py)
      `JobStore` (thread-pool, TTL eviction, swappable for Celery in B2).
      Predict + backtest return 202 + `job_id`; clients poll `/jobs/{id}`.
      OpenAPI at `/docs` + `/openapi.json` for `openapi-typescript` codegen.
      [Procfile](Procfile) + [railway.toml](railway.toml) for deploy.
      [docs/API.md](docs/API.md) is the public contract doc.
- [x] **B5.3 (stub)** — [api/routers/ws.py](api/routers/ws.py) `WS /api/v1/ws/prices`
      polls yfinance every 15 s. Wire format settled; swap polling for Redis
      Pub/Sub once B5.1/B5.2 land.
- [x] **B6.3** — [api/routers/models.py](api/routers/models.py) `GET /models/active`
      returns version + mtime of hybrid_model.py. Swaps to R2/S3 metadata in B6.2.
- [x] **B2.1 / B2.2 / B2.3** — Celery app at [workers/celery_app.py](workers/celery_app.py)
      (Redis broker + result backend, JSON serializer, late ACK).
      [workers/tasks.py](workers/tasks.py) `TASK_REGISTRY` pairs sync + Celery
      callables for predict / backtest / ledger_backfill / news_refresh /
      alert_eval — routers never name a callable, just `store.enqueue("kind",
      …)`. [workers/beat_schedule.py](workers/beat_schedule.py) crontabs:
      ledger backfill 16:30 IST, news refresh + alert eval every 5 min / 1 min
      during NSE session.
- [x] **B2 JobStore refactor** — [api/jobs.py](api/jobs.py) `JobStore`
      Protocol with `InProcessJobStore` (default) + `CeleryJobStore` (when
      `REDIS_URL` set). Picks at runtime via `workers.celery_app.IS_AVAILABLE`.
      `CeleryJobStore` mirrors the in-process API by storing per-job meta in
      a `protrader:jobs:{id}` Redis hash + reading state via `AsyncResult`.
- [x] **B3.1** — Supabase env vars (`DATABASE_URL`, `SUPABASE_URL`,
      `SUPABASE_ANON_KEY`, `SUPABASE_JWT_SECRET`, `REDIS_URL`) in
      [config/settings.py](config/settings.py). All optional; API falls back
      to in-process jobs + dev-mode auth bypass when unset.
- [x] **B3.2** — [db/base.py](db/base.py) lazy SQLAlchemy engine + session
      factory (psycopg v3 sync, `postgres://` → `postgresql+psycopg://`
      normaliser). [db/models.py](db/models.py) 8 ORM models matching the
      Supabase schema. [db/pg_ledger.py](db/pg_ledger.py) drop-in for
      `services.ledger_service` (log_prediction / backfill_actuals /
      accuracy_window / recent_rows) — uses `INSERT … ON CONFLICT DO NOTHING`
      for idempotency. [db/alerts_service.py](db/alerts_service.py) is the
      `evaluate_active_alerts()` invoked by the Celery beat tick.
- [x] **B3.3** — [db/sql/001_supabase_schema.sql](db/sql/001_supabase_schema.sql)
      paste-into-Supabase script: tables (user_profiles, predictions,
      watchlists, watchlist_tickers, alerts, backtests, paper_fills,
      paper_positions) + indexes + RLS policies (`auth.uid() = user_id`) +
      `handle_new_user` trigger that auto-creates a profile row on signup.
- [x] **B3.4** — [api/auth.py](api/auth.py) `current_user` / `optional_user`
      FastAPI deps. Verifies HS256 Supabase JWT via `SUPABASE_JWT_SECRET`,
      enforces `aud=authenticated` + `exp`. Dev mode 401s `current_user` and
      `None`s `optional_user` when secret unset. [api/routers/auth.py](api/routers/auth.py)
      adds `GET/PATCH /api/v1/me`.
- [x] **B3.5** — [api/rate_limit.py](api/rate_limit.py) `slowapi.Limiter`
      keyed on JWT `sub` (per-user) with IP fallback. Backed by Redis when
      `REDIS_URL` set, in-memory otherwise. `@limiter.limit("10/minute")` on
      `/predict`, `"20/minute"` on `/backtest`. Wired into
      [api/main.py](api/main.py) via `SlowAPIMiddleware` + custom 429 JSON
      handler.
- [x] **B3 watchlists / alerts routers** — [api/routers/watchlists.py](api/routers/watchlists.py)
      + [api/routers/alerts.py](api/routers/alerts.py) full CRUD, all auth-required,
      ownership enforced server-side (never trust client `user_id`). Alert
      `re-arm`: PATCH `active=true` clears `triggered_at`.
- [x] **docs/SUPABASE.md** — one-page setup guide (create project → run SQL
      → copy keys to `.env` → smoke test `/me`).
- [x] **Tests (B2/B3)** — 30 offline API tests now (was 23): `test_auth.py`
      (JWT roundtrip + dev-mode bypass + bad/expired token), `test_rate_limit.py`
      (429 after quota exhausted), `test_watchlists.py` + `test_alerts.py`
      (CRUD + cross-user isolation, SQLite in-memory via `StaticPool` so the
      models render without Postgres). 107 total tests across the repo.

### In progress
- [ ] _(none — B2 + B3 landed. Track A remainder still gated on Upstox KYC;
      Track B continues with B4 — Next.js frontend.)_

### Up next (per TASKS.md, in order)
- [ ] **B4** — Next.js frontend. OpenAPI schema at `/openapi.json` is ready;
      run `npx openapi-typescript` to generate the typed client.
- [ ] **B5.1 / B5.2** — Single Celery worker maintains the Upstox WS, fans
      ticks via Redis Pub/Sub `ticks:{symbol}`. Replaces the polling stub in
      [api/routers/ws.py](api/routers/ws.py).
- [ ] **B6.1 / B6.2** — Decide in-process vs BentoML/Triton; versioned model
      registry on R2 / S3. `/models/active` then reads blob metadata.
- [ ] **A2.4 / A2.5** — Track A remainder. Wire v2 `prob_up` into the Ridge
      meta-stacker. **Blocked by data:** historical per-day news backfill.
      Options: (1) GNews scrape, (2) inference-time late-blend (skips retrain).
- [ ] **A3.4 / A9.1 / A9.3** — Upstox tick stream + 30-day paper-trade run.
      Blocked on KYC (verification in progress 2026-04-18). Implementation
      ready: `PaperTradeService(fill_source=_upstox_sandbox_fill)`.

When you finish a task, **move it from "In progress" / "Up next" to "Done"**
with a one-line note (commit SHA if applicable). Don't let this drift.

---

## 5. How to run / test locally

```bash
# install
pip install -r requirements.txt

# run streamlit app
streamlit run app.py

# run FastAPI backend (Track B1)
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs

# smoke test sentiment without booting Streamlit
python -c "from data.news_sentiment import analyze_sentiment; \
           print(analyze_sentiment('Reliance beats earnings estimates'))"
```

If the user reports a deploy error, first place to look is the Streamlit Cloud
**Manage app → Logs** panel. Most failures so far have been
TF/Keras 3 incompatibility — fixed by `framework="pt"` + `tf-keras` dep.

---

## 6. Working agreement with the user

- The user codes solo and makes the architectural calls — propose, don't impose.
- Push back when I'm wrong (the v2-as-replacement vs v2-as-feature debate is the
  reference example: user was right, I had to correct course).
- Keep responses tight. Show code, not paragraphs.
- Don't start implementation until the user explicitly says "go" / "start".
- Free-tier first. If a task needs a paid service, flag it and offer the free
  alternative.

---

## 7. Open questions / things to ask before coding

When the user says "let's start":
1. Which task to start with — PHASE 0 refactor, or jump to A1 calibration?
2. Has the v2 model been pushed to HuggingFace yet? (gates A2)
3. Has Dhan / Upstox KYC been done? (gates A3.3 real-time ticks)

---

## 8. Memory note

The user's auto-memory at
`C:\Users\divya\.claude\projects\c--Users-divya-Desktop-finance\memory\MEMORY.md`
already has unrelated context (a paper-humanization project). Keep ProTrader
session state in **this file** (`CLAUDE.md`), not in auto-memory — auto-memory
is for cross-project facts about the user, not project build logs.
