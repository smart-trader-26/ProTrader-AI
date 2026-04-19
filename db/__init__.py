"""
Persistence layer (B3, framework mode) — Supabase over HTTPS via supabase-py.

The backend no longer holds a direct Postgres connection. All DB traffic
goes through Supabase's REST / PostgREST gateway:

  • `db.supabase_client` — client factory (admin + per-user).
  • `db.pg_ledger`       — predictions ledger (admin client, bypasses RLS).
  • `db.alerts_service`  — alert evaluator (admin client).

The Streamlit app's local SQLite ledger
([services/ledger_service.py](../services/ledger_service.py)) still works
offline and is used as the fallback when Supabase is not configured.

Schema lives in [db/sql/001_supabase_schema.sql](sql/001_supabase_schema.sql).
Run that file in the Supabase SQL editor once per project.
"""
