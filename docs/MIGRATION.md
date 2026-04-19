# ProTrader AI — Migration to Production Web App

**Goal:** move from a single-user Streamlit prototype to a production
multi-user web product with **Next.js (frontend) + FastAPI (backend) +
PostgreSQL + Redis + a model-serving layer**.

This is a **what-to-do** doc — phased plan, target architecture, file
layout, and the contract between layers. It does not try to write the
code; each phase has its own scope and ticket.

---

## 1. Why migrate (and why now)

| Limitation today (Streamlit) | Production need |
|------------------------------|-----------------|
| Single `app.py` = 1825 lines mixing data, model, UI | Separation of concerns |
| Each user's "Launch Analysis" reruns full pipeline | Cached predictions, async jobs |
| No auth, no per-user state, no DB | Watchlists, alerts, history |
| 1 GB RAM ceiling on Streamlit Cloud free tier | Horizontal scale (containers) |
| UI is server-rendered Python — no rich charts, no mobile | TradingView-class UX |
| Models retrained inline every request | Batch training, model registry |
| No real-time price feed (yfinance is 20-60 s polled) | WebSocket tick stream |
| No payment / metering | Subscriptions, free vs. pro tier |

---

## 2. Target architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     CLIENT  (Next.js 14 + Tailwind)              │
│  ─ App Router pages: /, /dashboard/[ticker], /watchlist, /auth  │
│  ─ TanStack Query for REST fetches                               │
│  ─ TradingView Lightweight Charts for price + overlays           │
│  ─ Plotly.js for SHAP / feature importance                       │
│  ─ Server-Sent Events stream for analysis progress               │
│  ─ WebSocket for live prices                                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTPS + WSS
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│           EDGE  (Cloudflare or Vercel Edge)                      │
│  ─ Auth check (Clerk / Auth.js JWT)                              │
│  ─ Rate limit per user                                           │
│  ─ Static asset CDN                                              │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                BACKEND  (FastAPI on Render / Railway)            │
│   /api/v1/                                                       │
│     auth/...                                                     │
│     stocks/{ticker}                  ← OHLCV, fundamentals       │
│     stocks/{ticker}/sentiment        ← multi-source              │
│     stocks/{ticker}/predict          ← model inference           │
│     stocks/{ticker}/backtest         ← async, returns job_id    │
│     watchlists/...                                               │
│     alerts/...                                                   │
│     jobs/{id}                        ← polled by client          │
│     ws/prices                        ← live tick fan-out         │
│     sse/analysis/{job_id}            ← progress stream           │
└──────┬───────────────────┬──────────────────┬────────────────────┘
       │                   │                  │
       ▼                   ▼                  ▼
┌─────────────┐    ┌──────────────┐   ┌───────────────────┐
│  PostgreSQL │    │    Redis     │   │  Model Serving    │
│  (Supabase) │    │  (Upstash)   │   │  (BentoML or      │
│  - users    │    │  - cache     │   │   FastAPI w/ heavy│
│  - tickers  │    │  - rate lim  │   │   models loaded   │
│  - history  │    │  - sessions  │   │   at startup, on  │
│  - alerts   │    │  - pub/sub   │   │   GPU box if FB)  │
└─────────────┘    └──────────────┘   └───────────────────┘
                          ▲
                          │
              ┌───────────┴───────────┐
              │   Celery / Arq        │
              │   workers             │
              │  - news fetch (5min)  │
              │  - model retrain (W)  │
              │  - alert eval (1min)  │
              │  - backtest jobs      │
              └───────────────────────┘
                          ▲
                          │
              ┌───────────┴───────────┐
              │  Object Storage       │
              │  (R2 / S3)            │
              │  - model .pkl files   │
              │  - cached news bodies │
              │  - exported reports   │
              └───────────────────────┘
```

---

## 3. Phased migration plan

### Phase 0 — Refactor in place (1 week)

Goal: turn `app.py` from a script into importable services without
breaking the Streamlit UI. After this phase Streamlit and the future
FastAPI both call the same code paths.

```
finance/
├── services/                           ← NEW
│   ├── __init__.py
│   ├── stock_service.py               # get_stock_data → also returns DTO
│   ├── sentiment_service.py           # wraps multi_sentiment
│   ├── prediction_service.py          # wraps create_hybrid_model
│   ├── backtest_service.py
│   └── pattern_service.py
├── schemas/                            ← NEW (Pydantic)
│   ├── stock.py
│   ├── sentiment.py
│   └── prediction.py
├── data/                               ← unchanged for now
├── models/                             ← unchanged for now
├── app.py                              ← thin: import services + render
└── api/                                ← Phase 1
```

**Acceptance:** `streamlit run app.py` still works, but every analytical
call goes through `services/*`. Pydantic schemas in `schemas/*` validate
inputs/outputs. Zero `streamlit` import inside `services/` or `models/`.

### Phase 1 — Stand up FastAPI alongside (1-2 weeks)

```
api/
├── main.py                             # FastAPI app factory
├── deps.py                             # DB, auth, settings
├── routers/
│   ├── stocks.py
│   ├── sentiment.py
│   ├── predict.py
│   ├── backtest.py
│   └── auth.py
└── middleware/
    ├── rate_limit.py
    └── logging.py
```

Run with `uvicorn api.main:app --reload`. Endpoints call the same
`services/` Phase 0 created.

Key endpoints (REST, JSON):

```
GET  /api/v1/stocks?q=relianc                    → search/autocomplete
GET  /api/v1/stocks/{ticker}/ohlcv?from&to       → OHLCV
GET  /api/v1/stocks/{ticker}/fundamentals
GET  /api/v1/stocks/{ticker}/sentiment           → multi-source
POST /api/v1/stocks/{ticker}/predict             → returns job_id
GET  /api/v1/jobs/{job_id}                       → status + result
GET  /api/v1/sse/analysis/{job_id}               → progress stream
WS   /api/v1/ws/prices?tickers=RELIANCE,TCS      → live ticks
```

**Acceptance:** Postman/Thunder Client can hit every endpoint, Swagger
UI at `/docs` is generated, response schemas typed end-to-end.

### Phase 2 — Move heavy work to background workers (1 week)

- Add **Celery** (or **Arq** if you want pure asyncio) backed by Redis.
- News fetch every 5 min for tickers in any user's watchlist.
- Model retraining weekly via a cron-style job (`celery beat`).
- Backtest jobs run on the worker, FastAPI returns `202 + job_id`.
- Alert evaluation every 1 min (compares latest price to user-set rules).

**Why:** keeps the FastAPI process under 200 ms response time. Long
work (60 s model train, 30 s sentiment scrape) doesn't block.

### Phase 3 — Build Next.js frontend (2-3 weeks)

```bash
npx create-next-app@latest protrader-web \
    --typescript --tailwind --app --eslint
cd protrader-web
npm i @tanstack/react-query lightweight-charts plotly.js \
       react-plotly.js @clerk/nextjs zustand zod
```

```
protrader-web/
├── app/
│   ├── layout.tsx
│   ├── page.tsx                        # landing
│   ├── (auth)/sign-in/page.tsx
│   ├── (app)/
│   │   ├── dashboard/page.tsx         # ticker list
│   │   ├── stocks/[ticker]/page.tsx   # main analysis
│   │   ├── watchlist/page.tsx
│   │   └── alerts/page.tsx
│   └── api/                           # only for Next-side stuff
├── components/
│   ├── PriceChart.tsx                 # TradingView lightweight
│   ├── SentimentPanel.tsx
│   ├── PredictionCard.tsx
│   ├── BacktestResults.tsx
│   └── ui/                            # shadcn/ui primitives
├── lib/
│   ├── api.ts                         # typed client w/ TanStack Query
│   ├── ws.ts                          # WebSocket hook
│   └── sse.ts                         # EventSource hook
├── store/                             # Zustand global state
└── styles/
```

**Recommended UI library:** **shadcn/ui** (copy-paste components,
Tailwind, fully owned) over Chakra / MUI for performance + customisation.

**Charts:** **TradingView Lightweight Charts** (45 KB, MIT, used by
TradingView themselves) for price + indicators. Plotly only where you
need 3D / SHAP / heatmap.

### Phase 4 — Auth + single-user persistence (1 week)

> **Scope change (2026-04-19):** personal-use only. Single Supabase
> account, no multi-tenant, no payments. Auth exists so the FastAPI
> backend stays lockable when exposed to the internet — not to segment
> customers.

- **Auth:** Clerk (fastest) or Auth.js + Supabase Auth (more control).
  JWT sent as `Authorization: Bearer …` to FastAPI; FastAPI verifies
  with Clerk's JWKS endpoint.
- **DB schema** (Supabase / Neon):

  ```sql
  users        (id, email, clerk_id, plan, created_at)
  tickers      (id, symbol, name, sector, exchange)
  watchlists   (id, user_id, ticker_id, created_at)
  alerts       (id, user_id, ticker_id, rule_jsonb, channel, active)
  predictions  (id, ticker_id, model_version, made_at, target_date,
                pred_price, pred_dir, confidence, actual_price, hit)
  backtests    (id, user_id, ticker_id, params_jsonb, results_jsonb,
                created_at)
  ```

- `predictions` table is **the accuracy ledger** — every prediction the
  model makes is logged with `actual_price` filled in next-day by the
  Celery worker. Queries against this table give you live, real,
  per-ticker, per-regime accuracy.

- **Payments:** dropped. Single-user product; no checkout surface.

### Phase 5 — Real-time data layer (1 week)

- yfinance is REST-polled. For real-time, integrate **Zerodha Kite
  Connect** (or **Alice Blue** free) WebSocket feed.
- Backend: a single Celery worker maintains the upstream WebSocket,
  fans ticks out via Redis pub/sub.
- FastAPI WS endpoint `/api/v1/ws/prices` subscribes to Redis pub/sub
  channels named `ticks:{symbol}` and forwards to the browser.

### Phase 6 — Model serving (1 week)

Two options:

**(a) In-process** — load the saved sklearn / FinBERT models inside the
FastAPI worker at startup. Simplest, fine for ≤ 100 req/min. 1 GB RAM
per worker.

**(b) Dedicated server** — run **BentoML** or **NVIDIA Triton** on a GPU
box (RunPod, Modal). FastAPI just makes HTTP calls. Right when:

- FinBERT inference is the bottleneck and you want batching
- You're going to fine-tune your own model (the v2 Colab one)
- You need GPU — Streamlit Cloud / Railway have none

**Model registry:** S3 / R2 bucket with versioned paths
(`models/v1/master_xgb.pkl`, `models/v2/...`). Workers load on startup.
Add **MLflow** if you train more than 1×/month.

### Phase 7 — Observability & deploy (3-5 days)

- **Sentry** for backend + frontend error tracking
- **PostHog** for product analytics (which features get used)
- **OpenTelemetry** traces from frontend → FastAPI → DB
- **GitHub Actions:** lint + test on PR; build + deploy on merge to main

**Hosting recommendation:**

| Layer | Where | Cost (est) |
|------|-------|-----------|
| Next.js | Vercel | Free → $20/mo |
| FastAPI | Railway / Render | $5-20/mo |
| Workers | Railway worker | $5-10/mo |
| Postgres | Supabase / Neon | Free → $25/mo |
| Redis | Upstash | Free → $10/mo |
| Object storage | Cloudflare R2 | ~$0 (10 GB free) |
| Auth | Clerk | Free up to 10 K MAU |
| GPU inference (optional) | Modal / RunPod | $0.50-2/hr (on-demand) |

**Total: $0-100/month** depending on traffic.

---

## 4. API contract (Pydantic / TypeScript shared)

This is the **single source of truth** for what flows between layers.
Generate the TypeScript client from FastAPI's OpenAPI schema with
`openapi-typescript` so frontend and backend never drift.

```ts
// generated from FastAPI

interface PredictionRequest {
  ticker: string;
  start_date: string;        // ISO yyyy-mm-dd
  end_date: string;
  forecast_days: number;     // 1..30
  enable_dynamic_fusion?: boolean;
}

interface PredictionResponse {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  result?: {
    ticker: string;
    forecast: Array<{
      date: string;
      pred_price: number;
      ci_low: number;        // 90% CI lower
      ci_high: number;       // 90% CI upper
    }>;
    confidence: number;      // 0..1, calibrated
    direction: "bullish" | "bearish" | "neutral";
    model_breakdown: Record<string, number>;  // weight per model
    sentiment: {
      score: number;         // -1..1
      label: string;
      sources: Record<string, { available: boolean; score: number }>;
    };
    accuracy: {
      historical_30d: number;  // % directional accuracy this ticker
      calibration_ece: number; // expected calibration error
      sample_size: number;
    };
  };
}
```

---

## 5. Confidence + accuracy as first-class API fields

Every prediction response **must** carry:

- `confidence: float` — calibrated bullish probability (Platt or
  isotonic on OOF data)
- `ci_low / ci_high` — quantile or conformal prediction interval
- `accuracy.historical_30d` — rolling directional accuracy from
  `predictions` table for this ticker
- `accuracy.calibration_ece` — Expected Calibration Error (lower = the
  model's % is trustworthy)
- `accuracy.sample_size` — how many predictions back this number

This is what makes the product *credible* vs. another "AI says BUY" toy.

---

## 6. Migration order — what to do this week vs. next month

**This week (1-2 days):**
- Phase 0 refactor — extract services from `app.py`. Streamlit keeps
  working.

**Next 2 weeks:**
- Phase 1: stand up FastAPI with 4 read endpoints (stocks, sentiment,
  fundamentals, predict). Streamlit and FastAPI share `services/`.
- Phase 2: move predictions to a Celery worker.

**Month 2:**
- Phase 3: Next.js frontend, ship MVP with auth.
- Phase 4: auth + single-user persistence.

**Month 3:**
- Phase 5: real-time prices.
- Phase 6: dedicated model server.
- Phase 7: observability + auto-deploy.

**During all of this**, keep the Streamlit app live as the "internal
admin" UI — it's the fastest way to test backend changes without waiting
for the React build.

---

## 7. What NOT to do during migration

- **Don't** rewrite the models — port them as-is.
- **Don't** delete `app.py` until the Next.js MVP is in users' hands.
- **Don't** roll your own auth — use Clerk or Auth.js.
- **Don't** use SQLAlchemy 2.0 + async + Pydantic v2 + FastAPI for the
  first time all at once. Pick known-working versions; complexity adds
  later.
- **Don't** put model files in git. R2 / S3 / HuggingFace Hub.
- **Don't** ship without the `predictions` accuracy-ledger table — it's
  what proves the product works to users (and to you).

---

## 8. First PR after this doc

Phase 0 only. One PR:

```
feat(refactor): extract analytical services from app.py
- Add services/{stock,sentiment,prediction,backtest,pattern}_service.py
- Add schemas/ Pydantic DTOs
- Update app.py to import services (no behaviour change)
- Add tests under tests/services/
```

Once that's green, the rest of the migration becomes additive, not
destructive. That is the whole point of doing Phase 0 first.
