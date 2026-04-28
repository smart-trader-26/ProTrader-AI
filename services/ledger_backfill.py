"""
Ledger backfill runner (A7.3).

Usage:
    python -m services.ledger_backfill           # resolves every row with
                                                 #  target_date <= today
    python -m services.ledger_backfill --up-to 2026-04-18

Idempotent — safe to run hourly in cron or once daily at 16:00 IST.
Exits non-zero only if the underlying sqlite / yfinance raise.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from services import ledger_service


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ledger_backfill", description=__doc__)
    parser.add_argument(
        "--up-to",
        type=lambda s: date.fromisoformat(s),
        default=None,
        help="Resolve rows with target_date <= YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Override the default SQLite path.",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help=(
            "Run the Phase-1C migration first: nullify prob_up for non-day-1 rows "
            "so calibration metrics are computed on genuine directional calls only."
        ),
    )
    args = parser.parse_args(argv)

    if args.migrate:
        updated = ledger_service.migrate_nullify_non_day1_prob_up(db_path=args.db)
        print(f"Migration: nullified prob_up on {updated} non-day-1 row(s).")

    resolved = ledger_service.backfill_actuals(
        up_to=args.up_to, db_path=args.db
    )
    print(f"Resolved {resolved} prediction row(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
