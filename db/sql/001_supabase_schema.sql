-- =====================================================================
-- ProTrader AI — Supabase Postgres schema (B3.3)
-- =====================================================================
-- Run this once in the Supabase SQL editor (Project → SQL → New query →
-- paste → Run). It creates every table the API needs, plus Row Level
-- Security policies so a logged-in user only ever sees their own rows.
--
-- Re-running is safe — every CREATE uses IF NOT EXISTS and every policy
-- is dropped + recreated.
--
-- After running, copy these from Supabase → Project Settings → API into
-- your .env:
--   DATABASE_URL          → Settings → Database → Connection string → URI
--                           (use the "Direct connection" string, NOT the
--                           pooler — SQLAlchemy + psycopg manages its
--                           own pool. Pooler URL works too but mind the
--                           statement-cache caveats.)
--   SUPABASE_URL          → Settings → API → Project URL
--   SUPABASE_ANON_KEY     → Settings → API → anon / public key
--   SUPABASE_JWT_SECRET   → Settings → API → JWT Settings → JWT Secret
-- =====================================================================

-- =====================================================================
-- 1. user_profiles  (one row per auth.users row)
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.user_profiles (
    id           UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email        TEXT,
    display_name TEXT,
    plan         TEXT NOT NULL DEFAULT 'free',  -- 'free' | 'pro' | 'enterprise'
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Auto-create a profile row whenever a new auth user signs up.
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO public.user_profiles (id, email)
    VALUES (NEW.id, NEW.email)
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- =====================================================================
-- 2. predictions  (A7 ledger, multi-tenant)
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.predictions (
    id                BIGSERIAL PRIMARY KEY,
    ticker            TEXT NOT NULL,
    made_at           TIMESTAMPTZ NOT NULL,
    target_date       DATE NOT NULL,
    pred_dir          TEXT NOT NULL,            -- up | down | flat
    pred_price        DOUBLE PRECISION NOT NULL,
    anchor_price      DOUBLE PRECISION,
    ci_low            DOUBLE PRECISION,
    ci_high           DOUBLE PRECISION,
    confidence_level  DOUBLE PRECISION NOT NULL DEFAULT 0.90,
    prob_up           DOUBLE PRECISION,
    horizon_days      INTEGER NOT NULL DEFAULT 1,
    model_version     TEXT NOT NULL,
    actual_price      DOUBLE PRECISION,
    hit               SMALLINT,
    user_id           UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    CONSTRAINT uq_pred UNIQUE (ticker, made_at, target_date)
);

CREATE INDEX IF NOT EXISTS idx_pred_ticker      ON public.predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_pred_target_date ON public.predictions(target_date);
CREATE INDEX IF NOT EXISTS idx_pred_user        ON public.predictions(user_id);

-- =====================================================================
-- 3. watchlists  (named groups of tickers per user)
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.watchlists (
    id          BIGSERIAL PRIMARY KEY,
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL DEFAULT 'Default',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_wl_name UNIQUE (user_id, name)
);
CREATE INDEX IF NOT EXISTS idx_wl_user ON public.watchlists(user_id);

CREATE TABLE IF NOT EXISTS public.watchlist_tickers (
    watchlist_id  BIGINT NOT NULL REFERENCES public.watchlists(id) ON DELETE CASCADE,
    ticker        TEXT NOT NULL,
    added_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (watchlist_id, ticker)
);

-- =====================================================================
-- 4. alerts  (price / probability thresholds the worker evaluates)
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.alerts (
    id            BIGSERIAL PRIMARY KEY,
    user_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    ticker        TEXT NOT NULL,
    kind          TEXT NOT NULL,  -- price_above|price_below|prob_up_above|prob_up_below
    threshold     DOUBLE PRECISION NOT NULL,
    active        BOOLEAN NOT NULL DEFAULT TRUE,
    triggered_at  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_alerts_user        ON public.alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_ticker      ON public.alerts(ticker);
CREATE INDEX IF NOT EXISTS idx_alerts_active      ON public.alerts(active) WHERE active = TRUE;

-- =====================================================================
-- 5. backtests  (one row per completed backtest job)
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.backtests (
    id              BIGSERIAL PRIMARY KEY,
    user_id         UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    ticker          TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    initial_capital DOUBLE PRECISION NOT NULL,
    final_equity    DOUBLE PRECISION NOT NULL,
    metrics         JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_bt_user   ON public.backtests(user_id);
CREATE INDEX IF NOT EXISTS idx_bt_ticker ON public.backtests(ticker);

-- =====================================================================
-- 6. paper_fills + paper_positions  (A9 paper-trading mirror)
-- =====================================================================
CREATE TABLE IF NOT EXISTS public.paper_fills (
    id            BIGSERIAL PRIMARY KEY,
    user_id       UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    ticker        TEXT NOT NULL,
    opened_at     TIMESTAMPTZ NOT NULL,
    closed_at     TIMESTAMPTZ,
    side          VARCHAR(8) NOT NULL,
    qty           INTEGER NOT NULL,
    entry_price   DOUBLE PRECISION NOT NULL,
    exit_price    DOUBLE PRECISION,
    gross_pnl     DOUBLE PRECISION NOT NULL DEFAULT 0,
    costs         DOUBLE PRECISION NOT NULL DEFAULT 0,
    net_pnl       DOUBLE PRECISION NOT NULL DEFAULT 0,
    reason_entry  TEXT,
    reason_exit   TEXT
);
CREATE INDEX IF NOT EXISTS idx_pf_user   ON public.paper_fills(user_id);
CREATE INDEX IF NOT EXISTS idx_pf_ticker ON public.paper_fills(ticker);

CREATE TABLE IF NOT EXISTS public.paper_positions (
    id            BIGSERIAL PRIMARY KEY,
    user_id       UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    ticker        TEXT NOT NULL,
    side          VARCHAR(8) NOT NULL,
    qty           INTEGER NOT NULL,
    entry_price   DOUBLE PRECISION NOT NULL,
    opened_at     TIMESTAMPTZ NOT NULL,
    stop_price    DOUBLE PRECISION,
    target_price  DOUBLE PRECISION,
    CONSTRAINT uq_pos_user_ticker UNIQUE (user_id, ticker)
);

-- =====================================================================
-- Row Level Security (RLS)
-- =====================================================================
-- Enable on every per-user table.
ALTER TABLE public.user_profiles      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.watchlists         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.watchlist_tickers  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alerts             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.backtests          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.paper_fills        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.paper_positions    ENABLE ROW LEVEL SECURITY;

-- Service role (used by FastAPI) bypasses RLS automatically. The policies
-- below cover direct access from the Supabase client SDK (anon + auth).
--
-- Pattern: each policy checks `auth.uid() = user_id`. For watchlist_tickers
-- we have to join through watchlists since the table itself doesn't carry
-- user_id.

-- user_profiles: read your own row, update your own row.
DROP POLICY IF EXISTS "profiles_self_read"   ON public.user_profiles;
DROP POLICY IF EXISTS "profiles_self_update" ON public.user_profiles;
CREATE POLICY "profiles_self_read"
    ON public.user_profiles FOR SELECT
    USING (auth.uid() = id);
CREATE POLICY "profiles_self_update"
    ON public.user_profiles FOR UPDATE
    USING (auth.uid() = id);

-- predictions: per-user CRUD; anonymous predictions (user_id IS NULL)
-- visible only to the service role.
DROP POLICY IF EXISTS "predictions_self_read"   ON public.predictions;
DROP POLICY IF EXISTS "predictions_self_write"  ON public.predictions;
DROP POLICY IF EXISTS "predictions_self_update" ON public.predictions;
DROP POLICY IF EXISTS "predictions_self_delete" ON public.predictions;
CREATE POLICY "predictions_self_read"
    ON public.predictions FOR SELECT
    USING (auth.uid() = user_id);
CREATE POLICY "predictions_self_write"
    ON public.predictions FOR INSERT
    WITH CHECK (auth.uid() = user_id);
CREATE POLICY "predictions_self_update"
    ON public.predictions FOR UPDATE
    USING (auth.uid() = user_id);
CREATE POLICY "predictions_self_delete"
    ON public.predictions FOR DELETE
    USING (auth.uid() = user_id);

-- watchlists
DROP POLICY IF EXISTS "watchlists_self_all" ON public.watchlists;
CREATE POLICY "watchlists_self_all"
    ON public.watchlists FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- watchlist_tickers (join-based)
DROP POLICY IF EXISTS "watchlist_tickers_self_all" ON public.watchlist_tickers;
CREATE POLICY "watchlist_tickers_self_all"
    ON public.watchlist_tickers FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM public.watchlists w
             WHERE w.id = watchlist_tickers.watchlist_id
               AND w.user_id = auth.uid()
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.watchlists w
             WHERE w.id = watchlist_tickers.watchlist_id
               AND w.user_id = auth.uid()
        )
    );

-- alerts
DROP POLICY IF EXISTS "alerts_self_all" ON public.alerts;
CREATE POLICY "alerts_self_all"
    ON public.alerts FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- backtests
DROP POLICY IF EXISTS "backtests_self_read"  ON public.backtests;
DROP POLICY IF EXISTS "backtests_self_write" ON public.backtests;
CREATE POLICY "backtests_self_read"
    ON public.backtests FOR SELECT
    USING (auth.uid() = user_id);
CREATE POLICY "backtests_self_write"
    ON public.backtests FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- paper_fills + paper_positions
DROP POLICY IF EXISTS "paper_fills_self_all"     ON public.paper_fills;
DROP POLICY IF EXISTS "paper_positions_self_all" ON public.paper_positions;
CREATE POLICY "paper_fills_self_all"
    ON public.paper_fills FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);
CREATE POLICY "paper_positions_self_all"
    ON public.paper_positions FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- =====================================================================
-- Done. Verify in the Table Editor:
--   • user_profiles, predictions, watchlists, watchlist_tickers,
--     alerts, backtests, paper_fills, paper_positions all listed
--   • RLS toggle showing "Enabled" on each
-- =====================================================================
