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
    args = parser.parse_args(argv)

    resolved = ledger_service.backfill_actuals(
        up_to=args.up_to, db_path=args.db
    )
    print(f"Resolved {resolved} prediction row(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
