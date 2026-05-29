"""
Mean-variance signal combiner (Phase 3 — P3-5).

Combines four alpha signals into a single composite signal that maximises the
ex-ante Sharpe ratio, as derived in Grinold & Kahn (2000) "Active Portfolio
Management", Chapter 6.

Mathematical derivation
-----------------------
Let s_i be the i-th alpha signal (zero-mean, unit-variance normalised).
Let IC_i be the Information Coefficient (rank-correlation with future returns).
Let Σ be the n×n signal covariance matrix (correlations between signals).

The maximum ex-ante Sharpe is achieved by the weight vector:

    w* = Σ^{-1} × IC   (up to a positive scalar)

In practice:
  1. We estimate IC_i from historical ledger data (rolling 60-day IC).
     For signals with insufficient history, we use principled priors.
  2. Σ is estimated from pairwise correlations of signal values over a
     rolling 120-day window.
  3. We regularise Σ with ledger shrinkage (λ = 0.3) toward the identity
     to prevent overfitting to the estimation window.

The four signals (with prior ICs):
    momentum (P3-1):   IC ≈ 0.030  (Jegadeesh-Titman, widely replicated)
    reversal (P3-2):   IC ≈ 0.020  (short-term, noisier)
    options  (P3-3):   IC ≈ 0.025  (PCR/IV skew; regime-dependent)
    llm_sent (P3-4):   IC ≈ 0.035  (novelty × materiality alpha; higher if API live)

The combiner is designed to degrade gracefully:
  • If fewer than 2 signals are available, it returns the available single signal.
  • If Σ^{-1} is ill-conditioned, it falls back to the IC-weighted average.
  • All edge cases return a value in [−1, +1].

Usage
-----
    from models.signal_combiner import SignalCombiner, SignalBundle

    combiner = SignalCombiner()

    # At inference time (per-prediction call):
    bundle = SignalBundle(
        ticker           = "RELIANCE.NS",
        momentum_signal  = 0.40,     # from alpha_signals.py
        reversal_signal  = -0.10,    # from alpha_signals.py
        options_signal   = 0.25,     # from options_alpha.py
        llm_signal       = 0.60,     # from llm_sentiment_alpha.py
    )
    combined = combiner.combine(bundle)
    # combined.composite_signal  — single float in [−1, +1]
    # combined.weights           — dict of signal weights used
    # combined.ex_ante_sharpe    — estimated Sharpe contribution

    # Fit from historical data (run periodically, e.g. weekly):
    combiner.fit(history_df)    # DataFrame with columns matching SignalBundle fields
    combiner.save()             # persist weights to disk
    combiner.load()             # restore weights
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Persistence ───────────────────────────────────────────────────────────────
_SAVED_DIR    = Path(__file__).parent / "saved"
_COMBINER_PATH = _SAVED_DIR / "signal_combiner.pkl"

# ── Signal names and their prior ICs ─────────────────────────────────────────
_SIGNAL_NAMES = ["momentum", "reversal", "options", "llm"]

_PRIOR_IC: dict[str, float] = {
    "momentum": 0.030,
    "reversal": 0.020,
    "options":  0.025,
    "llm":      0.035,
}

# Shrinkage parameter toward identity: 0 = no shrinkage, 1 = identity
_SHRINKAGE_LAMBDA = 0.30

# Minimum samples needed before we trust an estimated IC over the prior
_MIN_IC_SAMPLES = 30

# Output clip
_COMPOSITE_CLIP = 1.0


# ── DTOs ─────────────────────────────────────────────────────────────────────

@dataclass
class SignalBundle:
    """All alpha signals for one ticker at one point in time."""

    ticker: str

    momentum_signal:  Optional[float] = None   # P3-1: 12-1 month momentum
    reversal_signal:  Optional[float] = None   # P3-2: 5-day reversal
    options_signal:   Optional[float] = None   # P3-3: PCR Δ + IV skew Δ
    llm_signal:       Optional[float] = None   # P3-4: novelty × materiality

    # Optional: weights of each signal in the composite (filled in after combine())
    weights_used: dict[str, float] = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        """Return signals as a length-4 float array; NaN for missing signals."""
        return np.array([
            self.momentum_signal if self.momentum_signal is not None else np.nan,
            self.reversal_signal if self.reversal_signal is not None else np.nan,
            self.options_signal  if self.options_signal  is not None else np.nan,
            self.llm_signal      if self.llm_signal      is not None else np.nan,
        ], dtype=float)

    def available_mask(self) -> np.ndarray:
        """Boolean mask: True for signals with a non-NaN value."""
        return ~np.isnan(self.as_array())

    def n_available(self) -> int:
        return int(self.available_mask().sum())


@dataclass
class CombinedSignal:
    """Output of `SignalCombiner.combine()`."""

    ticker: str
    composite_signal:  float         # primary output: −1..+1
    weights:           dict[str, float] = field(default_factory=dict)
    ex_ante_ic:        float = 0.0   # estimated IC of the composite
    ex_ante_sharpe:    float = 0.0   # IC × sqrt(252) — rough annualised Sharpe
    n_signals:         int   = 0
    method:            str   = "mv"  # "mv" | "ic_weighted" | "equal" | "single"


# ── Main class ────────────────────────────────────────────────────────────────

class SignalCombiner:
    """
    Combines momentum, reversal, options, and LLM alpha signals using
    mean-variance (MV) weights that maximise the ex-ante Sharpe ratio.

    The combiner can be fit from historical signal-return data to update the
    IC estimates and signal covariance matrix. If not fit (or not enough data),
    it falls back to IC-weighted averaging using the prior ICs.

    Thread-safety: `combine()` is read-only and thread-safe after `fit()`.
    `fit()` and `save()` should be called from a single writer thread.
    """

    def __init__(self) -> None:
        # Estimated ICs (start from priors)
        self._ic:    np.ndarray = np.array([_PRIOR_IC[n] for n in _SIGNAL_NAMES], dtype=float)
        # Signal covariance matrix (start from identity — assume uncorrelated signals)
        self._sigma: np.ndarray = np.eye(len(_SIGNAL_NAMES), dtype=float)
        # MV weights (Σ^{-1} × IC, normalised to sum to 1)
        self._mv_weights: np.ndarray = self._compute_mv_weights(self._ic, self._sigma)
        # Fit metadata
        self._fit_samples:  int  = 0
        self._last_fit_date: str = ""

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        _SAVED_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "ic":            self._ic,
            "sigma":         self._sigma,
            "mv_weights":    self._mv_weights,
            "fit_samples":   self._fit_samples,
            "last_fit_date": self._last_fit_date,
        }
        with open(_COMBINER_PATH, "wb") as fh:
            pickle.dump(state, fh)
        logger.info("SignalCombiner: saved to %s", _COMBINER_PATH)

    def load(self) -> bool:
        if not _COMBINER_PATH.exists():
            return False
        try:
            with open(_COMBINER_PATH, "rb") as fh:
                state = pickle.load(fh)
            self._ic            = np.array(state["ic"],         dtype=float)
            self._sigma         = np.array(state["sigma"],      dtype=float)
            self._mv_weights    = np.array(state["mv_weights"], dtype=float)
            self._fit_samples   = int(state.get("fit_samples",  0))
            self._last_fit_date = str(state.get("last_fit_date", ""))
            logger.info(
                "SignalCombiner: loaded (n=%d, date=%s)",
                self._fit_samples, self._last_fit_date,
            )
            return True
        except Exception as exc:
            logger.warning("SignalCombiner.load failed: %s", exc)
            return False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        history_df: pd.DataFrame,
        return_col: str  = "future_return_5d",
        window: int      = 120,
    ) -> dict:
        """
        Estimate IC and signal covariance from historical data.

        Parameters
        ----------
        history_df : pd.DataFrame
            Long-format DataFrame with one row per (ticker, date). Required
            columns: momentum_signal, reversal_signal, options_signal,
            llm_signal, future_return_5d (5-day forward excess return over NIFTY).
        return_col : str
            Column name of the forward return target.
        window : int
            Rolling estimation window (bars). Only the last `window` rows
            are used to estimate IC and Σ.

        Returns
        -------
        dict with updated IC estimates and method metadata.
        """
        signal_cols = [f"{n}_signal" for n in _SIGNAL_NAMES]
        required    = signal_cols + [return_col]
        missing     = [c for c in required if c not in history_df.columns]
        if missing:
            logger.warning("SignalCombiner.fit: missing columns %s — using priors", missing)
            return {"method": "prior", "n_samples": 0}

        df = history_df[required].dropna(subset=[return_col]).tail(window).copy()
        n  = len(df)

        if n < _MIN_IC_SAMPLES:
            logger.info(
                "SignalCombiner.fit: only %d samples (need %d) — keeping priors",
                n, _MIN_IC_SAMPLES,
            )
            return {"method": "prior", "n_samples": n}

        # ── IC estimation (rank correlation per signal) ───────────────────────
        ret_rank = df[return_col].rank(pct=True)
        for i, name in enumerate(_SIGNAL_NAMES):
            col   = f"{name}_signal"
            valid = df[[col, return_col]].dropna()
            if len(valid) < _MIN_IC_SAMPLES:
                continue  # keep prior for this signal
            ic_est = float(valid[col].rank(pct=True).corr(valid[return_col].rank(pct=True)))
            if not np.isnan(ic_est):
                # Blend toward prior for stability (Bayesian shrinkage)
                shrink_w      = min(1.0, n / 252)   # full weight after 252 samples
                self._ic[i]   = shrink_w * ic_est + (1 - shrink_w) * _PRIOR_IC[name]

        # ── Signal covariance (rank correlations between signals) ─────────────
        sig_matrix = df[signal_cols].fillna(0.0).values
        if sig_matrix.shape[0] >= 10:
            corr = np.corrcoef(sig_matrix.T)
            # Replace NaNs with 0 (uncorrelated)
            corr = np.where(np.isnan(corr), 0.0, corr)
            np.fill_diagonal(corr, 1.0)

            # Ledger shrinkage toward identity
            n_sig         = len(_SIGNAL_NAMES)
            corr_shrunk   = (1 - _SHRINKAGE_LAMBDA) * corr + _SHRINKAGE_LAMBDA * np.eye(n_sig)
            self._sigma   = corr_shrunk

        # Recompute MV weights
        self._mv_weights    = self._compute_mv_weights(self._ic, self._sigma)
        self._fit_samples   = n
        self._last_fit_date = str(pd.Timestamp.now().date())

        logger.info(
            "SignalCombiner.fit: n=%d, IC=%s, MV_weights=%s",
            n,
            [f"{v:.3f}" for v in self._ic],
            [f"{v:.3f}" for v in self._mv_weights],
        )
        return {
            "method":     "mv_fitted",
            "n_samples":  n,
            "ic":         {n: float(v) for n, v in zip(_SIGNAL_NAMES, self._ic)},
            "mv_weights": {n: float(v) for n, v in zip(_SIGNAL_NAMES, self._mv_weights)},
        }

    # ── Combine ───────────────────────────────────────────────────────────────

    def combine(self, bundle: SignalBundle) -> CombinedSignal:
        """
        Combine the signals in `bundle` into a single composite alpha signal.

        The combination uses MV weights restricted to the available signals.
        When only one signal is available, it is returned directly.
        """
        s_arr = bundle.as_array()
        avail = ~np.isnan(s_arr)
        n_avail = int(avail.sum())

        if n_avail == 0:
            return CombinedSignal(
                ticker           = bundle.ticker,
                composite_signal = 0.0,
                n_signals        = 0,
                method           = "zero",
            )

        if n_avail == 1:
            idx = int(np.where(avail)[0][0])
            sig = float(np.clip(s_arr[idx], -_COMPOSITE_CLIP, _COMPOSITE_CLIP))
            return CombinedSignal(
                ticker           = bundle.ticker,
                composite_signal = sig,
                weights          = {_SIGNAL_NAMES[idx]: 1.0},
                ex_ante_ic       = float(self._ic[idx]),
                ex_ante_sharpe   = float(self._ic[idx] * np.sqrt(252)),
                n_signals        = 1,
                method           = "single",
            )

        # ── MV weights restricted to available signals ────────────────────────
        ic_avail    = self._ic[avail]
        sigma_avail = self._sigma[np.ix_(avail, avail)]

        try:
            composite, w_avail, method = self._mv_combine(
                s_arr[avail], ic_avail, sigma_avail
            )
        except Exception as exc:
            logger.debug("SignalCombiner.combine MV failed: %s — falling back", exc)
            composite, w_avail, method = self._ic_weighted_combine(
                s_arr[avail], ic_avail
            )

        # Reconstruct full weight dict
        weights_full: dict[str, float] = {}
        avail_names  = [_SIGNAL_NAMES[i] for i in range(len(_SIGNAL_NAMES)) if avail[i]]
        for name, w in zip(avail_names, w_avail):
            weights_full[name] = float(w)

        composite = float(np.clip(composite, -_COMPOSITE_CLIP, _COMPOSITE_CLIP))

        # Ex-ante IC of the composite = IC^T × w (approximate)
        ex_ante_ic = float(np.dot(ic_avail, w_avail))
        ex_ante_sharpe = float(ex_ante_ic * np.sqrt(252))

        return CombinedSignal(
            ticker           = bundle.ticker,
            composite_signal = composite,
            weights          = weights_full,
            ex_ante_ic       = ex_ante_ic,
            ex_ante_sharpe   = ex_ante_sharpe,
            n_signals        = n_avail,
            method           = method,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_mv_weights(
        ic: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        """
        Compute MV weights w* = Σ^{-1} × IC, normalised to L1 sum = 1.

        Falls back to IC-proportional weights if Σ is singular.
        """
        n = len(ic)
        try:
            sigma_inv = np.linalg.inv(sigma + 1e-6 * np.eye(n))
            raw       = sigma_inv @ ic
        except np.linalg.LinAlgError:
            raw = ic.copy()

        # Clip negative weights to 0 (no short-selling signals — long-only model)
        raw = np.maximum(raw, 0.0)
        total = float(raw.sum())
        if total < 1e-10:
            return np.ones(n) / n   # equal weight fallback
        return raw / total

    @staticmethod
    def _mv_combine(
        signals: np.ndarray,
        ic: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[float, np.ndarray, str]:
        """
        MV combination: composite = w^T × s, where w = Σ^{-1} × IC normalised.
        Returns (composite, weights, method_str).
        """
        n = len(signals)
        sigma_inv = np.linalg.inv(sigma + 1e-6 * np.eye(n))
        raw       = sigma_inv @ ic
        raw       = np.maximum(raw, 0.0)   # long-only
        total     = float(raw.sum())
        if total < 1e-10:
            w = np.ones(n) / n
            return float(np.dot(w, signals)), w, "equal"
        w = raw / total
        return float(np.dot(w, signals)), w, "mv"

    @staticmethod
    def _ic_weighted_combine(
        signals: np.ndarray,
        ic: np.ndarray,
    ) -> tuple[float, np.ndarray, str]:
        """
        IC-weighted average as a fallback when MV optimisation fails.
        Returns (composite, weights, method_str).
        """
        w_raw = np.maximum(ic, 0.0)
        total = float(w_raw.sum())
        if total < 1e-10:
            w = np.ones(len(signals)) / len(signals)
        else:
            w = w_raw / total
        return float(np.dot(w, signals)), w, "ic_weighted"

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_weights_summary(self) -> dict:
        """Return current IC estimates and MV weights for logging / display."""
        return {
            "signal_names":   _SIGNAL_NAMES,
            "ic_estimates":   {n: float(v) for n, v in zip(_SIGNAL_NAMES, self._ic)},
            "mv_weights":     {n: float(v) for n, v in zip(_SIGNAL_NAMES, self._mv_weights)},
            "fit_samples":    self._fit_samples,
            "last_fit_date":  self._last_fit_date,
        }
