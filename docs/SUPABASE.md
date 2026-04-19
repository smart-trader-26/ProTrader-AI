# Supabase setup (B3)

The FastAPI backend uses Supabase for two things:

1. **Postgres database** — SQLAlchemy talks to it via `psycopg` v3.
2. **Auth** — Supabase issues HS256 JWTs that we verify server-side.

Everything below is a one-time setup. Once `.env` is filled, `uvicorn
api.main:app --reload` brings the auth-protected routes online.

---

## 1. Create the project

1. Sign in at <https://supabase.com> → **New project**.
2. Name it (e.g. `protrader-prod`), pick the region closest to your
   users (Mumbai = `ap-south-1` for Indian retail).
3. Set a database password — save it somewhere; you'll need it for the
   connection string.

The free tier gives 500 MB Postgres + 50 k MAU, more than enough for
dev + a small private beta.

---

## 2. Run the schema

1. **SQL Editor** → **New query**.
2. Paste the entire contents of
   [db/sql/001_supabase_schema.sql](../db/sql/001_supabase_schema.sql).
3. Click **Run**. You should see "Success. No rows returned."

What it creates:
- 8 tables: `user_profiles`, `predictions`, `watchlists`,
  `watchlist_tickers`, `alerts`, `backtests`, `paper_fills`,
  `paper_positions`.
- A trigger that auto-creates a `user_profiles` row on signup.
- Row Level Security policies so the Supabase client (anon / auth keys)
  can only read each user's own rows. The FastAPI service key bypasses
  RLS — that's intentional, the API enforces ownership in code.

Verify in **Table Editor** that every table is listed and the **RLS**
toggle reads "Enabled".

The script is idempotent — re-running it is safe.

---

## 3. Copy keys into `.env`

Project Settings → **Database**:

```
DATABASE_URL=postgresql://postgres.<project-ref>:<password>@aws-0-ap-south-1.pooler.supabase.com:6543/postgres
```

Use the **Connection pooling → Transaction pooler** URL (port 6543) for
serverless-friendly connection counts. For local dev a direct connection
(port 5432) works fine.

Project Settings → **API**:

```
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOi…   # public; safe in the FE bundle
SUPABASE_JWT_SECRET=<32+ char random>  # JWT Settings → JWT Secret
```

`SUPABASE_JWT_SECRET` is what the backend uses to verify tokens. **Never
expose it in the FE.**

---

## 4. Smoke test

```bash
# 1. Backend boots and connects to Postgres
uvicorn api.main:app --reload --port 8000
# → look for "SQLAlchemy engine ready" in the logs

# 2. /me returns 401 without a token
curl -s http://localhost:8000/api/v1/me
# → {"detail": "missing bearer token"}

# 3. With a Supabase JWT (sign in via the FE, copy from devtools):
curl -s http://localhost:8000/api/v1/me \
     -H "Authorization: Bearer <jwt>"
# → {"id": "<uuid>", "email": "you@…", "plan": "free", ...}
```

---

## 5. Optional: Celery + Redis (B2)

The job queue uses Redis as broker + result backend. Free options:

- **Local**: `docker run -p 6379:6379 redis:7`
- **Hosted**: <https://upstash.com> free tier (10 k commands / day).

Then add to `.env`:

```
REDIS_URL=redis://default:<password>@<host>:<port>
```

Start the worker(s):

```bash
celery -A workers.celery_app worker --loglevel=info --pool=solo
celery -A workers.celery_app beat   --loglevel=info
```

When `REDIS_URL` is unset, the API silently falls back to an
in-process `ThreadPoolExecutor` job store — fine for development, but
beat schedules don't run.

---

## 6. Production deploy notes

- **Connection pooling**: use the Supabase pooler (port 6543) on Railway /
  Fly / any serverless host. Direct connections cap at 60 on the free
  tier — easy to exhaust under load.
- **CORS**: set `PROTRADER_CORS_ORIGINS` to your prod FE origin (comma
  separated for multiple). Default is localhost only.
- **Secrets**: never commit `.env`. Use the host's secret manager
  (Railway Variables, Fly Secrets, etc.).
- **RLS**: the FastAPI service uses the connection string directly (so
  it bypasses RLS). If you ever expose Supabase's auto-REST or use the
  `supabase-py` SDK from the backend, RLS will kick in — you'll need
  the service-role key, NOT the anon key.
