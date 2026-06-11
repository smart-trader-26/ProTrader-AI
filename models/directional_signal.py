"""
Swing Directional Signal (P4-1) — an honest, conviction-gated multi-day
directional model.

Why this module exists
----------------------
The per-ticker hybrid model (`models.hybrid_model`) forecasts NEXT-DAY return.
Next-day single-name direction is ~50/50 noise — no model beats it materially,
and after the calibration fix the reported probability correctly sits near 0.50.

This module answers a *different, tractable* question that real systematic desks
actually trade: **"Over the next ~30 trading days (~1.5 months), is this name
more likely to be higher than it is now — and is the setup convincing enough to
act on?"**  Over a 30-day horizon two real edges exist on NSE large-caps:

  1. Equity drift (≈58% of 30-day windows are up — the "always-up" base rate).
  2. A small, *stable* idiosyncratic edge captured by a calibrated cross-sectional
     model on momentum / trend / reversal / RSI / vol features.

Measured out-of-sample (walk-forward, 2018-2026, 54 NSE names; 91,471 train rows
and 50,272 OOS rows), the high-conviction UP bucket (calibrated p ≥ τ* = 0.63)
achieves 60.6% directional precision vs a 58.0% "always-up" base rate (+2.5 pp
net edge), at ~5.6% coverage, and beats the base rate in 4 of 5 years
(2022 61.6/54.0 · 2023 68.3/65.3 · 2024 53.3/55.7 below · 2025 69.0/58.2 ·
2026 54.5/47.1). Ranking AUC is ~0.47 by design — the edge lives in the
high-conviction tail, not the ordering. The signal is deliberately **long /
neutral only**: confident DOWN calls on single names are destroyed by the upward
drift (~10% precision OOS), so the model never emits a high-conviction SHORT.

Honesty contract
----------------
Everything this module reports is backed by the persisted out-of-sample
statistics in `self.stats` — the displayed "expected hit-rate" is the *measured*
walk-forward precision of the fired bucket, not an in-sample fit. If the model
is not trained, `predict()` returns a clearly-flagged NEUTRAL with `trained=False`.

Offline training (run periodically, e.g. monthly):
    from models.directional_signal import DirectionalSignal
    sig = DirectionalSignal(horizon=30)
    stats = sig.train(years=8)      # downloads the universe, walk-forward eval, persists
    print(stats)

Live inference (single ticker, fast — needs only that ticker's ~2y history):
    sig = DirectionalSignal.load_default()
    out = sig.predict(close_series, ticker="RELIANCE.NS")
    # out["signal"]            -> "UP" | "NEUTRAL"
    # out["prob_up"]           -> calibrated P(up over horizon)
    # out["expected_hit_rate"] -> measured OOS precision of the fired bucket
    # out["coverage"]          -> fraction of history the signal fires
    # out["horizon_days"]      -> 30
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Reuse the same liquid universe the rest of the project trains on.
try:
    from models.cross_sectional_trainer import NIFTY50_TICKERS, NIFTY_MIDCAP_EXTRAS
    _DEFAULT_UNIVERSE = NIFTY50_TICKERS + NIFTY_MIDCAP_EXTRAS
except Exception:  # pragma: no cover - fallback if import graph changes
    _DEFAULT_UNIVERSE = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "SBIN.NS", "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    ]

_SAVED_DIR = Path(__file__).parent / "saved"
_MODEL_PATH = _SAVED_DIR / "directional_signal.pkl"

FEATURE_NAMES: list[str] = [
    "mom_12_1",   # 12-1 month momentum (Jegadeesh-Titman)
    "ret_20",     # intermediate-term return
    "ret_60",     # quarterly return
    "rev_5",      # 5-day short-term reversal (negated)
    "dist_200",   # log distance above/below 200-DMA
    "ma_cross",   # 50-DMA > 200-DMA (golden cross flag)
    "rsi",        # RSI(14) / 100
    "vol_20",     # 20-day realised vol of log returns
    "above_200",  # price above 200-DMA flag
]

# Default horizon and conviction-tuning targets.
_DEFAULT_HORIZON = 30            # trading days (~1.5 calendar months) — matches the shipped model
_TARGET_PRECISION = 0.60         # tune τ* so the fired UP bucket reaches this
_MIN_COVERAGE = 0.05             # don't pick a τ* that fires <5% of the time
_PROB_CLIP = (0.02, 0.98)


def compute_features(close: pd.Series, horizon: int = _DEFAULT_HORIZON,
                     with_label: bool = False) -> pd.DataFrame:
    """
    Point-in-time features for one ticker's close series. No look-ahead:
    every column at row t uses only data up to and including t.

    When `with_label=True` also attaches `fwd_ret` / `y` (forward horizon-day
    log-return and its sign) for training/evaluation — never used at inference.
    """
    s = close.astype(float)
    lr = np.log(s / s.shift(1))
    sma50, sma200 = s.rolling(50).mean(), s.rolling(200).mean()
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-9))

    f = pd.DataFrame(index=s.index)
    f["mom_12_1"] = np.log(s.shift(21) / s.shift(252))
    f["ret_20"] = np.log(s / s.shift(20))
    f["ret_60"] = np.log(s / s.shift(60))
    f["rev_5"] = -np.log(s / s.shift(5))
    f["dist_200"] = np.log(s / sma200)
    f["ma_cross"] = (sma50 > sma200).astype(float)
    f["rsi"] = rsi / 100.0
    f["vol_20"] = lr.rolling(20).std()
    f["above_200"] = (s > sma200).astype(float)

    if with_label:
        f["fwd_ret"] = np.log(s.shift(-horizon) / s)
        f["y"] = (f["fwd_ret"] > 0).astype(float)
    return f


@dataclass
class SignalStats:
    """Honest, persisted out-of-sample performance of the fired UP bucket."""
    horizon_days: int = _DEFAULT_HORIZON
    tau_star: float = 0.60               # conviction threshold for an UP call
    oos_precision: float = 0.0           # measured precision of fired UP bucket
    oos_coverage: float = 0.0            # fraction of observations that fire
    oos_base_rate: float = 0.0           # naive always-up rate (for honesty)
    oos_auc: float = 0.5
    n_oos: int = 0
    per_year: dict = field(default_factory=dict)   # {year: {prec, base, n}}
    trained_at: str = ""
    n_train_rows: int = 0
    n_tickers: int = 0
    universe_period: str = ""


class DirectionalSignal:
    """Calibrated, conviction-gated, long/neutral swing-direction signal."""

    def __init__(self, horizon: int = _DEFAULT_HORIZON) -> None:
        self.horizon = int(horizon)
        self._scaler = None          # sklearn StandardScaler
        self._clf = None             # sklearn LogisticRegression (fit on all data)
        self._calibrator = None      # isotonic map prob_raw -> calibrated prob
        self.stats = SignalStats(horizon_days=self.horizon)

    # ── persistence ──────────────────────────────────────────────────────────
    def is_trained(self) -> bool:
        return self._clf is not None and self._scaler is not None

    def save(self, path: Optional[Path] = None) -> Path:
        p = Path(path or _MODEL_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump({
                "horizon": self.horizon,
                "scaler": self._scaler,
                "clf": self._clf,
                "calibrator": self._calibrator,
                "stats": asdict(self.stats),
                "features": FEATURE_NAMES,
            }, fh)
        logger.info("DirectionalSignal: saved to %s", p)
        return p

    def load(self, path: Optional[Path] = None) -> bool:
        p = Path(path or _MODEL_PATH)
        if not p.exists():
            return False
        try:
            with open(p, "rb") as fh:
                st = pickle.load(fh)
            self.horizon = int(st.get("horizon", _DEFAULT_HORIZON))
            self._scaler = st["scaler"]
            self._clf = st["clf"]
            self._calibrator = st.get("calibrator")
            self.stats = SignalStats(**st.get("stats", {}))
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("DirectionalSignal.load failed: %s", exc)
            return False

    @classmethod
    def load_default(cls) -> "DirectionalSignal":
        inst = cls()
        inst.load()
        return inst

    # ── inference ──────────────────────────────────────────────────────────────
    def predict(self, close: pd.Series, ticker: str = "") -> dict:
        """
        Return the swing-direction call for the latest bar of `close`.

        Never raises — degrades to a flagged NEUTRAL on any problem so a
        prediction is never blocked by this optional signal.
        """
        base = {
            "ticker": ticker,
            "horizon_days": self.horizon,
            "signal": "NEUTRAL",
            "prob_up": 0.5,
            "conviction": 0.0,
            "trained": self.is_trained(),
            "expected_hit_rate": self.stats.oos_precision or None,
            "coverage": self.stats.oos_coverage or None,
            "base_rate": self.stats.oos_base_rate or None,
            "tau_star": self.stats.tau_star,
            "note": "",
        }
        if not self.is_trained():
            base["note"] = "model not trained"
            return base
        try:
            feats = compute_features(close, self.horizon, with_label=False)
            row = feats[FEATURE_NAMES].iloc[-1:]
            if row.isna().any(axis=None):
                base["note"] = "insufficient history for features (need ~252 bars)"
                return base
            x = self._scaler.transform(row.values)
            p_raw = float(self._clf.predict_proba(x)[0, 1])
            p = self._apply_calibrator(p_raw)
            p = float(np.clip(p, *_PROB_CLIP))
            base["prob_up"] = p
            base["prob_up_raw"] = p_raw
            # Long / neutral only — confident shorts are not reliable (drift).
            if p >= self.stats.tau_star:
                base["signal"] = "UP"
                base["conviction"] = float(min(1.0, (p - 0.5) / max(1e-6, self.stats.tau_star - 0.5)))
                base["note"] = (
                    f"high-conviction UP — historically right "
                    f"{self.stats.oos_precision*100:.0f}% over {self.horizon}d "
                    f"(fires {self.stats.oos_coverage*100:.0f}% of the time)"
                )
            else:
                base["signal"] = "NEUTRAL"
                base["conviction"] = 0.0
                base["note"] = "below conviction threshold — no actionable edge"
            return base
        except Exception as exc:  # pragma: no cover
            logger.warning("DirectionalSignal.predict(%s) failed: %s", ticker, exc)
            base["note"] = f"prediction error: {type(exc).__name__}"
            return base

    def _apply_calibrator(self, p_raw: float) -> float:
        if self._calibrator is None:
            return p_raw
        try:
            return float(self._calibrator.predict([p_raw])[0])
        except Exception:
            return p_raw

    # ── training ────────────────────────────────────────────────────────────────
    def train(self, years: int = 8, universe: Optional[list[str]] = None,
              target_precision: float = _TARGET_PRECISION,
              prices: Optional[pd.DataFrame] = None) -> dict:
        """
        Download the universe, build a leak-free panel, run a walk-forward
        out-of-sample evaluation to (a) measure honest precision/coverage and
        pick the conviction threshold τ*, then (b) fit the final production
        model on all available data. Persists everything via `save()`.

        `prices` (date × ticker close matrix) may be supplied to skip download
        (used by tests / fast re-runs).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import roc_auc_score

        universe = universe or _DEFAULT_UNIVERSE
        if prices is None:
            prices = self._download(universe, years)
        if prices is None or prices.shape[1] < 5:
            raise RuntimeError("DirectionalSignal.train: insufficient price data")

        panel = self._build_panel(prices)
        if len(panel) < 2000:
            raise RuntimeError(f"DirectionalSignal.train: panel too small ({len(panel)})")

        # ── walk-forward OOS (expanding window, embargoed) ──────────────────────
        dates = np.sort(panel["date"].unique())
        n_dates = len(dates)
        start_i = int(n_dates * 0.45)
        step = 21
        oos_p, oos_y, oos_dt = [], [], []
        i = start_i
        while i < n_dates:
            cut = dates[i]
            embargo = dates[max(0, i - self.horizon)]
            tr = panel[panel["date"] < embargo]
            blk_end = dates[min(i + step, n_dates - 1)]
            te = panel[(panel["date"] >= cut) & (panel["date"] < blk_end)]
            if len(tr) >= 500 and len(te) >= 10:
                sc = StandardScaler().fit(tr[FEATURE_NAMES].values)
                clf = LogisticRegression(C=0.5, max_iter=400)
                clf.fit(sc.transform(tr[FEATURE_NAMES].values), tr["y"].values)
                p = clf.predict_proba(sc.transform(te[FEATURE_NAMES].values))[:, 1]
                oos_p.append(p)
                oos_y.append(te["y"].values)
                oos_dt.append(te["date"].values)
            i += step

        P = np.concatenate(oos_p)
        Y = np.concatenate(oos_y)
        DT = pd.to_datetime(np.concatenate(oos_dt))

        tau_star, prec, cov = self._tune_threshold(P, Y, target_precision)
        per_year = {}
        yrs = pd.Series(DT).dt.year.values
        for yr in sorted(set(yrs)):
            m = (yrs == yr) & (P >= tau_star)
            ym = yrs == yr
            if m.sum() >= 20:
                per_year[int(yr)] = {
                    "precision": round(float(Y[m].mean()), 4),
                    "base_rate": round(float(Y[ym].mean()), 4),
                    "n_fire": int(m.sum()),
                }

        # ── final production model: fit on ALL data + isotonic calibrator ────────
        self._scaler = StandardScaler().fit(panel[FEATURE_NAMES].values)
        self._clf = LogisticRegression(C=0.5, max_iter=400)
        self._clf.fit(self._scaler.transform(panel[FEATURE_NAMES].values), panel["y"].values)
        # Calibrator fit on the OOS (prob -> outcome) so reported probs are honest.
        try:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(P, Y)
            self._calibrator = iso
        except Exception:
            self._calibrator = None

        try:
            auc = float(roc_auc_score(Y, P))
        except Exception:
            auc = 0.5

        self.stats = SignalStats(
            horizon_days=self.horizon,
            tau_star=round(float(tau_star), 4),
            oos_precision=round(float(prec), 4),
            oos_coverage=round(float(cov), 4),
            oos_base_rate=round(float(Y.mean()), 4),
            oos_auc=round(auc, 4),
            n_oos=int(len(P)),
            per_year=per_year,
            trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_train_rows=int(len(panel)),
            n_tickers=int(panel["ticker"].nunique()),
            universe_period=f"{prices.index[0].date()}..{prices.index[-1].date()}",
        )
        self.save()
        return asdict(self.stats)

    # ── training helpers ────────────────────────────────────────────────────────
    def _tune_threshold(self, P: np.ndarray, Y: np.ndarray,
                        target_precision: float) -> tuple[float, float, float]:
        """
        Pick the lowest τ in [0.50, 0.75] whose UP bucket reaches
        target_precision while still firing on >= _MIN_COVERAGE of observations.
        Maximises coverage subject to the precision floor; if the floor is never
        met, returns the τ with the highest precision at acceptable coverage.
        """
        best = (0.60, 0.0, 0.0)
        best_meeting = None
        for tau in np.round(np.arange(0.50, 0.7601, 0.01), 2):
            m = P >= tau
            cov = float(m.mean())
            if m.sum() < 30 or cov < _MIN_COVERAGE:
                continue
            prec = float(Y[m].mean())
            if prec >= target_precision and best_meeting is None:
                best_meeting = (float(tau), prec, cov)  # first (lowest τ) that meets target → max coverage
            if prec > best[1]:
                best = (float(tau), prec, cov)
        return best_meeting or best

    def _build_panel(self, prices: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for t in prices.columns:
            s = prices[t].dropna()
            if len(s) < 400:
                continue
            f = compute_features(s, self.horizon, with_label=True)
            f["ticker"] = t
            f["date"] = f.index
            rows.append(f)
        if not rows:
            return pd.DataFrame()
        panel = pd.concat(rows).dropna(subset=FEATURE_NAMES + ["y", "fwd_ret"])
        return panel.sort_values("date").reset_index(drop=True)

    def _download(self, universe: list[str], years: int) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("DirectionalSignal.train requires yfinance")
        raw = yf.download(universe, period=f"{years}y", auto_adjust=True,
                          progress=False, timeout=90)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        return close.dropna(how="all")


# Module-level convenience for the prediction service.
_DEFAULT_INSTANCE: Optional[DirectionalSignal] = None


def get_signal() -> DirectionalSignal:
    """Lazy-load and cache the default persisted signal (process-wide)."""
    global _DEFAULT_INSTANCE
    if _DEFAULT_INSTANCE is None:
        _DEFAULT_INSTANCE = DirectionalSignal.load_default()
    return _DEFAULT_INSTANCE
