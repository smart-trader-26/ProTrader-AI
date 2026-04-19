"""
Celery + scheduled-task layer (Track B2).

Two entry points the deploy environment needs:

    celery -A workers.celery_app worker --loglevel=info
    celery -A workers.celery_app beat   --loglevel=info

The worker process executes tasks; the beat process emits scheduled jobs
(news refresh, ledger backfill, alert evaluation). Both speak to the same
Redis broker via `REDIS_URL`. When `REDIS_URL` is unset the FastAPI
backend silently falls back to an in-process thread-pool — local dev does
not require Redis.
"""
