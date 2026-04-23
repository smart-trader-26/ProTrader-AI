"""
Beat schedule (B2.3).

Builder pattern keeps `celery_app.py` clean and lets tests assert against
the schedule without booting Celery. Every task referenced here must be
registered in `workers/tasks.py`.

Schedules are in **UTC** (Celery convention). NSE close = 10:00 UTC; we
backfill actuals at 11:00 UTC = 16:30 IST so any 16:00-IST close prices
are settled in yfinance.
"""

from __future__ import annotations

from typing import Any


def build_beat_schedule(crontab) -> dict[str, dict[str, Any]]:
    return {
        # Generate predictions and enter paper trades every morning.
        # 03:30 UTC = 09:00 IST (15 minutes before NSE opens). Mon-Fri only.
        "paper-trade-daily": {
            "task": "protrader.paper_trade",
            "schedule": crontab(hour=3, minute=30, day_of_week="mon-fri"),
            "kwargs": {"dry_run": False},
        },
        # Resolve actuals for any prediction whose target_date has passed.
        # 11:00 UTC = 16:30 IST = ~30 min after NSE close. Mon-Fri only.
        "ledger-backfill-daily": {
            "task": "protrader.ledger_backfill",
            "schedule": crontab(hour=11, minute=0, day_of_week="mon-fri"),
            "kwargs": {},
        },
        # Refresh top-of-watchlist sentiment every 5 min during NSE hours
        # (03:45 – 10:00 UTC = 09:15 – 15:30 IST). Mon–Fri only.
        "news-refresh-market-hours": {
            "task": "protrader.news_refresh",
            "schedule": crontab(
                minute="*/5",
                hour="3-10",
                day_of_week="mon-fri",
            ),
            "kwargs": {},
        },
        # Evaluate active alerts every minute during market hours.
        "alert-eval-market-hours": {
            "task": "protrader.alert_eval",
            "schedule": crontab(
                minute="*",
                hour="3-10",
                day_of_week="mon-fri",
            ),
            "kwargs": {},
        },
    }
