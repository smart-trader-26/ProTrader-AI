"""
Live performance monitor (Phase 3 — P3-10, P3-11).

Reads the prediction ledger (SQLite) and computes three tiers of performance
diagnostics:

  Daily:    ECE, hit-rate, mean confidence, n_resolved
  Weekly:   Rolling 7-day Sharpe (PnL-weighted, not just hit-rate)
  Monthly:  Signal attribution by ticker and horizon; ECE trend chart data

Auto-suspend (P3-11):
  If the rolling 60-day Sharpe falls below 0 for 3 consecutive daily runs,
  the monitor writes a JSON flag to `data/ledger/trading_suspended.json`.
  The prediction service checks this flag before emitting new predictions;
  if the flag is active, predictions are emitted with a SUSPENDED warning
  rather than being blocked entirely (we still need to see what the model
  says, even when suspended, to diagnose issues and decide when to resume).

Sharpe calculation from the ledger
-----------------------------------
For each resolved prediction:
  signal  = (prob_up − 0.5) × 2     (ranges −1..+1)
  ret     = (actual_price − anchor_price) / anchor_price
  pnl     = signal × ret             (dollar-normalised PnL)

Rolling Sharpe = mean(pnl[-N:]) / std(pnl[-N:]) × sqrt(252)
where N = window_days.

This is the "virtual" or "paper" Sharpe because we don't track actual position
sizes. The directional hit-rate is implicitly the same as Sharpe > 0.

Usage
-----
    from services.live_monitor import LiveMonitor

    monitor = LiveMonitor()

    # Run all three tiers (typically called by a daily cron job):
    report = monitor.run_daily_report(ticker=None)  # None = all tickers
    print(report.daily.sharpe_60d)
    print(report.suspension_active)

    # Check suspension status only (fast, called from prediction_service):
    suspended = LiveMonitor.is_trading_suspended()

    # Manually resume after investigating:
    LiveMonitor.resume_trading("Model retrained with Phase 3 data")
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_LEDGER_PATH        = Path("data/ledger/predictions.sqlite")
_SUSPENSION_PATH    = Path("data/ledger/trading_suspended.json")
_METRICS_PATH       = Path("data/ledger/live_metrics.json")
_LOCK               = threading.Lock()

# ── Thresholds ────────────────────────────────────────────────────────────────
_SHARPE_SUSPEND_THRESHOLD   = 0.0     # suspend if 60d Sharpe falls below this
_SHARPE_CONSECUTIVE_DAYS    = 3       # must be below threshold for N days to trigger
_ECE_WARN_THRESHOLD         = 0.15    # log warning if ECE > 15%
_DAILY_ECE_MIN_SAMPLES      = 10
_WEEKLY_SHARPE_WINDOW       = 7       # days
_MONTHLY_ATTRIBUTION_WINDOW = 30      # days
_ROLLING_SHARPE_WINDOW      = 60      # days (for auto-suspend decision)
_TRADING_DAYS_ANN           = 252


# ── DTOs ─────────────────────────────────────────────────────────────────────

@dataclass
class DailyMetrics:
    """Single-day snapshot of model accuracy."""

    as_of_date:          str   = ""
    ticker:              Optional[str] = None  # None = all tickers

    # Accuracy
    n_resolved:          int   = 0
    n_total:             int   = 0
    directional_accuracy: float = 0.0   # 0..1

    # Calibration
    ece:                 float = 0.0
    brier_score:         float = 0.0
    mean_confidence:     float = 0.5   # mean |prob_up − 0.5|

    # Sharpe (using virtual PnL)
    sharpe_60d:          float = 0.0
    sharpe_30d:          float = 0.0

    # Warning flags
    ece_warning:         bool  = False
    sharpe_negative:     bool  = False


@dataclass
class WeeklyMetrics:
    """7-day rolling performance."""

    week_ending:         str   = ""
    n_resolved:          int   = 0
    sharpe_7d:           float = 0.0
    hit_rate_7d:         float = 0.0
    mean_pnl_7d:         float = 0.0
    vol_pnl_7d:          float = 0.0


@dataclass
class MonthlyAttribution:
    """30-day signal attribution."""

    month_ending:        str   = ""
    n_resolved:          int   = 0

    # Per-ticker performance
    top_tickers:         list[dict] = field(default_factory=list)   # [{ticker, n, acc, sharpe}]
    bottom_tickers:      list[dict] = field(default_factory=list)

    # ECE trend (list of {date, ece} over the month)
    ece_trend:           list[dict] = field(default_factory=list)

    # Sharpe by horizon
    sharpe_by_horizon:   dict[str, float] = field(default_factory=dict)


@dataclass
class LiveReport:
    """Full live monitoring report from one run."""

    run_at:              str  = ""
    ticker:              Optional[str] = None
    daily:               DailyMetrics = field(default_factory=DailyMetrics)
    weekly:              WeeklyMetrics = field(default_factory=WeeklyMetrics)
    monthly:             MonthlyAttribution = field(default_factory=MonthlyAttribution)

    suspension_active:   bool = False
    suspension_reason:   str  = ""

    def to_dict(self) -> dict:
        return {
            "run_at":    self.run_at,
            "ticker":    self.ticker,
            "daily":     self.daily.__dict__,
            "weekly":    self.weekly.__dict__,
            "monthly":   {
                k: v for k, v in self.monthly.__dict__.items()
                if k != "ece_trend"   # exclude large list from top-level summary
            },
            "suspension_active": self.suspension_active,
            "suspension_reason": self.suspension_reason,
        }


# ── Suspension flag I/O ───────────────────────────────────────────────────────

class _SuspensionFlag:
    """Read / write the trading suspension flag file."""

    @staticmethod
    def is_active() -> bool:
        """Return True if trading is currently suspended."""
        with _LOCK:
            if not _SUSPENSION_PATH.exists():
                return False
            try:
                data = json.loads(_SUSPENSION_PATH.read_text())
                return bool(data.get("suspended", False))
            except Exception:
                return False

    @staticmethod
    def activate(reason: str, sharpe_60d: float) -> None:
        """Write the suspension flag."""
        _SUSPENSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "suspended":    True,
            "reason":       reason,
            "activated_at": datetime.utcnow().isoformat(),
            "sharpe_60d":   round(sharpe_60d, 4),
            "resume_after": (date.today() + timedelta(days=7)).isoformat(),
        }
        with _LOCK:
            _SUSPENSION_PATH.write_text(json.dumps(payload, indent=2))
        logger.warning("LiveMonitor: TRADING SUSPENDED — %s (60d Sharpe = %.3f)", reason, sharpe_60d)

    @staticmethod
    def deactivate(reason: str) -> None:
        """Clear the suspension flag."""
        payload = {
            "suspended":    False,
            "reason":       reason,
            "deactivated_at": datetime.utcnow().isoformat(),
        }
        with _LOCK:
            _SUSPENSION_PATH.write_text(json.dumps(payload, indent=2))
        logger.info("LiveMonitor: trading resumed — %s", reason)

    @staticmethod
    def get_state() -> dict:
        with _LOCK:
            if not _SUSPENSION_PATH.exists():
                return {"suspended": False}
            try:
                return json.loads(_SUSPENSION_PATH.read_text())
            except Exception:
                return {"suspended": False}


# ── Database helpers ──────────────────────────────────────────────────────────

def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else _LEDGER_PATH
    if not path.exists():
        raise FileNotFoundError(f"Ledger not found: {path}")
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    return conn


def _load_resolved_rows(
    ticker: Optional[str],
    days: int,
    db_path: Optional[Path] = None,
    now: Optional[date]    = None,
) -> pd.DataFrame:
    """
    Load resolved prediction rows from the last `days` (by target_date).

    Returns a DataFrame with columns:
        ticker, made_at, target_date, pred_dir, pred_price,
        anchor_price, prob_up, horizon_days, actual_price, hit
    """
    now_d   = now or date.today()
    cutoff  = (now_d - timedelta(days=days)).isoformat()
    where   = "target_date >= ? AND hit IS NOT NULL AND prob_up IS NOT NULL"
    params: list = [cutoff]

    if ticker:
        where  += " AND ticker = ?"
        params.append(ticker)

    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                f"SELECT ticker, made_at, target_date, pred_dir, pred_price, "
                f"       anchor_price, prob_up, horizon_days, actual_price, hit "
                f"  FROM predictions "
                f" WHERE {where} "
                f" ORDER BY target_date ASC",
                params,
            ).fetchall()
    except FileNotFoundError:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["target_date"] = pd.to_datetime(df["target_date"])
    df["made_at"]     = pd.to_datetime(df["made_at"])
    return df


# ── Analytics helpers ─────────────────────────────────────────────────────────

def _compute_ece(prob_up: np.ndarray, hit: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error: mean |confidence − accuracy| over bins.
    Lower is better. Perfectly calibrated model = 0.0.
    """
    if len(prob_up) < 5:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    n     = len(prob_up)

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob_up >= lo) & (prob_up < hi)
        if mask.sum() == 0:
            continue
        conf_mean = float(prob_up[mask].mean())
        acc_mean  = float(hit[mask].mean())
        ece      += (mask.sum() / n) * abs(conf_mean - acc_mean)

    return float(ece)


def _compute_virtual_pnl(df: pd.DataFrame) -> np.ndarray:
    """
    Compute virtual per-prediction PnL:
        signal = (prob_up − 0.5) × 2     (−1..+1)
        ret    = (actual − anchor) / anchor
        pnl    = signal × ret

    Rows with missing prices are dropped.
    """
    valid = (
        df["prob_up"].notna() &
        df["actual_price"].notna() &
        df["anchor_price"].notna() &
        (df["anchor_price"] > 0)
    )
    d = df[valid].copy()
    if d.empty:
        return np.array([], dtype=float)

    signal = (d["prob_up"].values - 0.5) * 2.0
    ret    = (d["actual_price"].values - d["anchor_price"].values) / d["anchor_price"].values
    return (signal * ret).astype(float)


def _sharpe_from_pnl(pnl: np.ndarray, ann_factor: float = 252.0) -> float:
    """Annualised Sharpe from a daily PnL array."""
    if len(pnl) < 3:
        return 0.0
    mu  = float(np.mean(pnl))
    sig = float(np.std(pnl, ddof=1))
    if sig < 1e-12:
        return 0.0
    return float(mu / sig * np.sqrt(ann_factor))


def _brier_score(prob_up: np.ndarray, hit: np.ndarray) -> float:
    """Mean squared error of probability forecasts. Lower is better (0 = perfect)."""
    if len(prob_up) == 0:
        return 0.5
    return float(np.mean((prob_up - hit) ** 2))


# ── Main monitor class ────────────────────────────────────────────────────────

class LiveMonitor:
    """
    Live performance monitor and auto-suspension controller.

    Typical usage pattern (daily cron at market close):
        monitor = LiveMonitor()
        report  = monitor.run_daily_report()
        monitor.save_metrics(report)
        if report.suspension_active:
            alert_team(report.suspension_reason)
    """

    def __init__(
        self,
        db_path:    Optional[Path] = None,
        now:        Optional[date] = None,
    ) -> None:
        self._db_path = db_path
        self._now     = now or date.today()

    # ── Public: suspension check (fast) ──────────────────────────────────────

    @staticmethod
    def is_trading_suspended() -> bool:
        """
        Fast check: return True if the trading suspension flag is active.
        Called by the prediction service before emitting predictions.
        """
        return _SuspensionFlag.is_active()

    @staticmethod
    def resume_trading(reason: str = "Manual resume") -> None:
        """
        Clear the suspension flag. Call this after diagnosing and fixing
        the issue that caused the suspension.
        """
        _SuspensionFlag.deactivate(reason)

    @staticmethod
    def get_suspension_state() -> dict:
        """Return the full suspension flag payload as a dict."""
        return _SuspensionFlag.get_state()

    # ── Public: full report ───────────────────────────────────────────────────

    def run_daily_report(
        self,
        ticker: Optional[str] = None,
    ) -> LiveReport:
        """
        Compute all three tiers of performance diagnostics and evaluate
        the auto-suspension criteria.

        Parameters
        ----------
        ticker : str | None
            If provided, restrict analysis to this ticker. None = all tickers.

        Returns
        -------
        LiveReport
        """
        daily   = self._compute_daily(ticker)
        weekly  = self._compute_weekly(ticker)
        monthly = self._compute_monthly(ticker)

        # ── Auto-suspend logic (P3-11) ────────────────────────────────────────
        suspension_active = _SuspensionFlag.is_active()
        suspension_reason = ""

        if daily.sharpe_60d < _SHARPE_SUSPEND_THRESHOLD and daily.n_resolved >= _DAILY_ECE_MIN_SAMPLES:
            # Load the saved metrics to check how many consecutive days Sharpe < 0
            consecutive_bad = self._count_consecutive_bad_sharpe_days()
            if consecutive_bad >= _SHARPE_CONSECUTIVE_DAYS and not suspension_active:
                reason = (
                    f"Rolling 60-day Sharpe={daily.sharpe_60d:.3f} < 0 "
                    f"for {consecutive_bad} consecutive days"
                )
                _SuspensionFlag.activate(reason, daily.sharpe_60d)
                suspension_active = True
                suspension_reason = reason
            elif suspension_active:
                state  = _SuspensionFlag.get_state()
                suspension_reason = state.get("reason", "")
        else:
            # Sharpe recovered — auto-resume if we had 5+ consecutive good days
            if suspension_active:
                consecutive_good = self._count_consecutive_good_sharpe_days()
                if consecutive_good >= 5:
                    resume_reason = (
                        f"Rolling 60-day Sharpe recovered to {daily.sharpe_60d:.3f} "
                        f"for {consecutive_good} consecutive days"
                    )
                    _SuspensionFlag.deactivate(resume_reason)
                    suspension_active = False
                    suspension_reason = resume_reason

        report = LiveReport(
            run_at             = datetime.utcnow().isoformat(),
            ticker             = ticker,
            daily              = daily,
            weekly             = weekly,
            monthly            = monthly,
            suspension_active  = suspension_active,
            suspension_reason  = suspension_reason,
        )
        return report

    # ── Public: persist metrics ───────────────────────────────────────────────

    def save_metrics(self, report: LiveReport) -> None:
        """
        Append today's metrics to the live metrics JSON log.
        Used by the consecutive-day counters.
        """
        _METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        history: list[dict] = []
        if _METRICS_PATH.exists():
            try:
                history = json.loads(_METRICS_PATH.read_text())
            except Exception:
                history = []

        entry = {
            "date":             self._now.isoformat(),
            "sharpe_60d":       round(report.daily.sharpe_60d, 4),
            "sharpe_30d":       round(report.daily.sharpe_30d, 4),
            "ece":              round(report.daily.ece, 4),
            "n_resolved":       report.daily.n_resolved,
            "directional_acc":  round(report.daily.directional_accuracy, 4),
            "suspension":       report.suspension_active,
        }
        history.append(entry)
        # Keep only last 365 entries
        history = history[-365:]
        _METRICS_PATH.write_text(json.dumps(history, indent=2))

    # ── Private: daily metrics ────────────────────────────────────────────────

    def _compute_daily(self, ticker: Optional[str]) -> DailyMetrics:
        """Compute daily ECE, hit rate, and rolling Sharpe."""
        df_60 = _load_resolved_rows(ticker, days=60,  db_path=self._db_path, now=self._now)
        df_30 = _load_resolved_rows(ticker, days=30,  db_path=self._db_path, now=self._now)
        df_7  = _load_resolved_rows(ticker, days=7,   db_path=self._db_path, now=self._now)

        # Use 30-day window for daily metrics (ECE, hit rate)
        if df_30.empty:
            return DailyMetrics(
                as_of_date = self._now.isoformat(),
                ticker     = ticker,
            )

        hit_arr  = df_30["hit"].astype(float).values
        prob_arr = df_30["prob_up"].astype(float).values

        hit_rate = float(np.mean(hit_arr)) if len(hit_arr) > 0 else 0.0
        ece      = _compute_ece(prob_arr, hit_arr)
        brier    = _brier_score(prob_arr, hit_arr)
        mean_conf = float(np.mean(np.abs(prob_arr - 0.5))) if len(prob_arr) > 0 else 0.0

        # Rolling Sharpe from virtual PnL
        pnl_60 = _compute_virtual_pnl(df_60)
        pnl_30 = _compute_virtual_pnl(df_30)
        sharpe_60 = _sharpe_from_pnl(pnl_60)
        sharpe_30 = _sharpe_from_pnl(pnl_30)

        n_total_query = _load_resolved_rows(ticker, days=30, db_path=self._db_path, now=self._now)

        return DailyMetrics(
            as_of_date           = self._now.isoformat(),
            ticker               = ticker,
            n_resolved           = len(df_30),
            n_total              = len(df_30),
            directional_accuracy = hit_rate,
            ece                  = ece,
            brier_score          = brier,
            mean_confidence      = mean_conf,
            sharpe_60d           = sharpe_60,
            sharpe_30d           = sharpe_30,
            ece_warning          = ece > _ECE_WARN_THRESHOLD and len(df_30) >= _DAILY_ECE_MIN_SAMPLES,
            sharpe_negative      = sharpe_60 < _SHARPE_SUSPEND_THRESHOLD,
        )

    # ── Private: weekly metrics ───────────────────────────────────────────────

    def _compute_weekly(self, ticker: Optional[str]) -> WeeklyMetrics:
        """7-day rolling Sharpe and hit rate."""
        df = _load_resolved_rows(ticker, days=_WEEKLY_SHARPE_WINDOW,
                                 db_path=self._db_path, now=self._now)

        if df.empty:
            return WeeklyMetrics(week_ending=self._now.isoformat())

        pnl     = _compute_virtual_pnl(df)
        hit_arr = df["hit"].astype(float).values

        return WeeklyMetrics(
            week_ending = self._now.isoformat(),
            n_resolved  = len(df),
            sharpe_7d   = _sharpe_from_pnl(pnl),
            hit_rate_7d = float(np.mean(hit_arr)) if len(hit_arr) > 0 else 0.0,
            mean_pnl_7d = float(np.mean(pnl)) if len(pnl) > 0 else 0.0,
            vol_pnl_7d  = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0,
        )

    # ── Private: monthly attribution ─────────────────────────────────────────

    def _compute_monthly(self, ticker: Optional[str]) -> MonthlyAttribution:
        """30-day attribution by ticker and ECE trend."""
        df = _load_resolved_rows(ticker, days=_MONTHLY_ATTRIBUTION_WINDOW,
                                 db_path=self._db_path, now=self._now)

        if df.empty:
            return MonthlyAttribution(month_ending=self._now.isoformat())

        # Per-ticker attribution
        ticker_stats: list[dict] = []
        for t, grp in df.groupby("ticker"):
            hit_arr = grp["hit"].astype(float).values
            pnl     = _compute_virtual_pnl(grp)
            ticker_stats.append({
                "ticker": str(t),
                "n":      len(grp),
                "acc":    round(float(np.mean(hit_arr)), 3),
                "sharpe": round(_sharpe_from_pnl(pnl), 3),
                "ece":    round(_compute_ece(grp["prob_up"].values, hit_arr), 3),
            })

        ticker_stats_sorted = sorted(ticker_stats, key=lambda x: -x["sharpe"])
        top_5    = ticker_stats_sorted[:5]
        bottom_5 = ticker_stats_sorted[-5:]

        # ECE trend: daily ECE over the month
        ece_trend: list[dict] = []
        df["date_only"] = df["target_date"].dt.date
        for dt, grp in df.groupby("date_only"):
            if len(grp) < 5:
                continue
            ece_d = _compute_ece(grp["prob_up"].astype(float).values,
                                  grp["hit"].astype(float).values)
            ece_trend.append({"date": str(dt), "ece": round(ece_d, 4)})

        # Sharpe by horizon
        sharpe_by_horizon: dict[str, float] = {}
        for h, grp in df.groupby("horizon_days"):
            pnl_h = _compute_virtual_pnl(grp)
            sharpe_by_horizon[str(h)] = round(_sharpe_from_pnl(pnl_h), 3)

        return MonthlyAttribution(
            month_ending         = self._now.isoformat(),
            n_resolved           = len(df),
            top_tickers          = top_5,
            bottom_tickers       = bottom_5,
            ece_trend            = ece_trend,
            sharpe_by_horizon    = sharpe_by_horizon,
        )

    # ── Private: consecutive-day counters ─────────────────────────────────────

    def _count_consecutive_bad_sharpe_days(self) -> int:
        """Return number of consecutive days ending today where 60d Sharpe < 0."""
        return self._count_consecutive_flag(below_threshold=True)

    def _count_consecutive_good_sharpe_days(self) -> int:
        """Return number of consecutive days ending today where 60d Sharpe >= 0."""
        return self._count_consecutive_flag(below_threshold=False)

    def _count_consecutive_flag(self, below_threshold: bool) -> int:
        if not _METRICS_PATH.exists():
            return 0
        try:
            history = json.loads(_METRICS_PATH.read_text())
        except Exception:
            return 0

        # Walk backwards from today
        count = 0
        for entry in reversed(history):
            sharpe = float(entry.get("sharpe_60d", 0.0))
            n_res  = int(entry.get("n_resolved", 0))
            if n_res < _DAILY_ECE_MIN_SAMPLES:
                break  # not enough data — stop counting
            if below_threshold:
                if sharpe < _SHARPE_SUSPEND_THRESHOLD:
                    count += 1
                else:
                    break
            else:
                if sharpe >= _SHARPE_SUSPEND_THRESHOLD:
                    count += 1
                else:
                    break

        return count


# ── Convenience: standalone health check ─────────────────────────────────────

def quick_health_check(
    db_path: Optional[Path] = None,
    ticker: Optional[str] = None,
) -> dict:
    """
    Return a minimal health-check dict suitable for a FastAPI /health endpoint.

    Includes: suspension status, 30d Sharpe, 30d ECE, 30d hit rate.
    """
    monitor = LiveMonitor(db_path=db_path)
    daily   = monitor._compute_daily(ticker)
    susp    = _SuspensionFlag.get_state()

    return {
        "status":       "SUSPENDED" if susp.get("suspended") else "OK",
        "suspension":   susp,
        "sharpe_30d":   round(daily.sharpe_30d, 3),
        "sharpe_60d":   round(daily.sharpe_60d, 3),
        "ece_30d":      round(daily.ece, 4),
        "hit_rate_30d": round(daily.directional_accuracy, 3),
        "n_resolved":   daily.n_resolved,
        "ece_warning":  daily.ece_warning,
        "as_of":        daily.as_of_date,
    }
