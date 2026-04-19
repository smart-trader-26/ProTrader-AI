# ProTrader AI — FastAPI Backend (Track B1)

Production backend that wraps the same `services/` layer the Streamlit UI
uses, so every feature stays consistent across both surfaces.

## Run locally

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the auto-generated Swagger UI, or
`http://localhost:8000/openapi.json` for the raw schema.

## Endpoints (v1)

All endpoints sit under `/api/v1`.

| Method | Path | Notes |
|--------|------|-------|
| `GET`  | `/healthz` | Liveness probe (Railway / Render). |
| `GET`  | `/readyz`  | Readiness — extends to Redis / PG once B2 / B3 land. |
| `GET`  | `/stocks?q=` | Substring search across the local NSE universe. |
| `GET`  | `/stocks/{ticker}/ohlcv?start=&end=` | Typed `StockHistory` (default last 6mo). |
| `GET`  | `/stocks/{ticker}/fundamentals` | Typed `Fundamentals`. |
| `GET`  | `/stocks/{ticker}/info` | Raw info dict (Streamlit parity). |
| `GET`  | `/stocks/{ticker}/sentiment` | FinBERT + 6-cat aggregate. |
| `GET`  | `/stocks/{ticker}/sentiment/v2` | V2 ensemble (gated behind `HF_TOKEN`). |
| `POST` | `/sentiment/score` | Score arbitrary text. |
| `POST` | `/stocks/{ticker}/predict` | **Returns 202 + `job_id`.** Poll `/jobs/{id}`. |
| `POST` | `/stocks/{ticker}/backtest` | **Returns 202 + `job_id`.** Poll `/jobs/{id}`. |
| `GET`  | `/jobs/{id}` | Status + result for a queued job. |
| `GET`  | `/accuracy?ticker=&days=` | A7 rolling accuracy window. |
| `GET`  | `/accuracy/recent?ticker=` | Recent ledger rows. |
| `GET`  | `/models/active` | Active model version + train date. |
| `WS`   | `/ws/prices?tickers=` | Stub price stream (B5.3 stub). |

## Job pattern (B1.4 / B1.5)

Long-running endpoints return immediately with a job ID:

```bash
curl -X POST http://localhost:8000/api/v1/stocks/RELIANCE.NS/predict \
     -H 'Content-Type: application/json' -d '{"horizon_days": 10}'
# → { "job_id": "abc...", "status": "queued", "poll_url": "/api/v1/jobs/abc..." }

curl http://localhost:8000/api/v1/jobs/abc...
# → { "status": "running" }                         # poll again
# → { "status": "succeeded", "result": { ... } }    # ready
```

The job store is in-process today (`api/jobs.py`). When **B2** lands (Celery
+ Redis), swap `JobStore` for a Redis-backed implementation — router code
doesn't change.

## Generating a TypeScript client (B1.6)

For the Next.js frontend (B4), generate a typed client from the OpenAPI
schema:

```bash
# In the Next.js repo
npx openapi-typescript http://localhost:8000/openapi.json -o lib/api.d.ts
```

Then use TanStack Query with the generated types:

```ts
import type { paths } from "@/lib/api"
type Bundle = paths["/api/v1/jobs/{job_id}"]["get"]["responses"]["200"]["content"]["application/json"]
```

## Deployment (B1.7)

- **Railway**: `railway.toml` already declares the start command and health check.
- **Render**: use the `Procfile` (`web: uvicorn ...`).
- Set `PROTRADER_CORS_ORIGINS=https://your-frontend.vercel.app` so the
  browser can call the API.
- The Streamlit app keeps deploying to Streamlit Cloud (see
  [docs/DEPLOYMENT.md](DEPLOYMENT.md)) — both surfaces run side-by-side
  until B4 replaces the Streamlit UI.

## What's NOT here yet

| Area | Track | Status |
|------|-------|--------|
| Celery + Redis worker | B2 | Stubbed by `api/jobs.JobStore`. |
| Postgres + Auth (Clerk / Supabase) | B3 | Ledger still SQLite. |
| Next.js frontend | B4 | OpenAPI schema is ready for codegen. |
| Real-time tick layer | B5 | `/ws/prices` polls yfinance every 15s. |
| Model registry on R2 / S3 | B6 | `/models/active` reports in-process version. |
| Sentry / PostHog / Stripe | B7 | Not wired. |
