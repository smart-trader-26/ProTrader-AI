web: uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
worker: celery -A workers.celery_app worker --loglevel=info --concurrency=2
beat: celery -A workers.celery_app beat --loglevel=info
ticks: python -m workers.tick_publisher --interval ${TICK_INTERVAL:-15}
