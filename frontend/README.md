# ProTrader AI — Frontend (B4)

Next.js 15 App Router + `@supabase/ssr` for auth/watchlists/alerts (direct to
Supabase via RLS), with a typed fetch client that calls the FastAPI backend
for predict/backtest/sentiment/accuracy.

## Stack

- Next.js 15.0.3 (App Router, React 19 RC)
- TypeScript (strict), Tailwind CSS
- `@supabase/ssr` for server + middleware auth cookies
- `openapi-typescript` for regenerating the FastAPI client

## Setup

```bash
cd frontend
npm install
cp .env.local.example .env.local
# fill in values — see below
npm run dev          # http://localhost:3000
```

### `.env.local`

| var                          | source                                                          |
| ---------------------------- | --------------------------------------------------------------- |
| `NEXT_PUBLIC_SUPABASE_URL`   | Supabase dashboard → Project Settings → API → **Project URL**   |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase dashboard → Project Settings → API → **anon public** |
| `NEXT_PUBLIC_API_URL`        | FastAPI base URL (no trailing slash). Dev: `http://localhost:8000` |

## Architecture

```
Browser ── Supabase JS SDK ──► Supabase (watchlists, alerts — RLS-gated)
   │
   └─ cookies (access_token) ──► Next.js middleware → updateSession()
                                         │
                                         ▼
                    Server Components / Route Handlers
                                         │
                                         ▼
                         FastAPI (Bearer <token>)
                                         │
                                         ▼
                Supabase admin client (service_role, bypasses RLS)
```

- **RLS-backed CRUD** (`/dashboard`): watchlists + alerts are read/written
  directly via `@supabase/supabase-js` — the DB enforces ownership, no
  FastAPI hop.
- **Compute-heavy calls** (`/stock/:ticker`, `/accuracy`): the server
  component forwards the Supabase session's `access_token` as a Bearer to
  FastAPI. The backend verifies HS256 with `SUPABASE_JWT_SECRET`.

## Regenerating the API client

With the FastAPI backend running on `http://localhost:8000`:

```bash
npm run codegen
```

That dumps a typed schema to `lib/api-types.ts`. Until then, a hand-curated
subset lives in `lib/types.ts` covering the endpoints the starter UI hits.

## Backend expectations

The frontend assumes these env vars are set on the FastAPI side:

| var                         | purpose                                          |
| --------------------------- | ------------------------------------------------ |
| `SUPABASE_URL`              | REST + realtime base                             |
| `SUPABASE_ANON_KEY`         | per-user client (RLS in effect)                  |
| `SUPABASE_SERVICE_ROLE_KEY` | worker/admin client (bypasses RLS)               |
| `SUPABASE_JWT_SECRET`       | verifies the Bearer token from `@supabase/ssr`   |
| `REDIS_URL` (optional)      | Celery broker + rate-limit store                 |

CORS on the backend (`api/main.py`) must allow `http://localhost:3000`.

## Pages

- `/` — landing page (redirects to `/dashboard` when signed in)
- `/login` — email + password (Supabase auth). Swap to magic-link or OAuth
  providers via the Supabase dashboard; the callback route is wired at
  `/auth/callback`.
- `/dashboard` — watchlists + alerts. Direct Supabase reads/writes.
- `/stock/[ticker]` — OHLCV sparkline, on-demand prediction, sentiment panel.
- `/accuracy` — rolling hit-rate / Brier / ECE + recent ledger rows.

## Production

1. Build: `npm run build`
2. Deploy to Vercel (or any Node host). Set the three env vars above.
3. Point `NEXT_PUBLIC_API_URL` at your Railway/Fly FastAPI deploy.
4. In Supabase dashboard → Authentication → URL Configuration:
   - **Site URL**: `https://your-frontend-domain`
   - **Redirect URLs**: add `/auth/callback` for email confirmations.
