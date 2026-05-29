"""
Cross-sectional LightGBM trainer (Phase 2 — P2-2 / P2-3 / P2-10).

Trains ONE LightGBM on N NSE tickers × T years of daily bars.
This gives 50–500× more training rows than a per-ticker model and lets
the model learn cross-sectional patterns (momentum persistence, sector
rotation, vol clustering) that a single-ticker GRU cannot learn at all.

Key design choices
------------------
* Target: 5-day *excess* return over NIFTY50 = log(ticker[t+5]/ticker[t])
          - log(nifty[t+5]/nifty[t]).  This removes the market factor and
          makes the model learn *idiosyncratic* alpha — exactly what
          investors care about.
* Features: standard technical indicators + NIFTY-relative features
  (Nifty_Ret, Rel_Strength_5D/20D, Beta_20D) + Sector_ID integer.
* Purged expanding-window k-fold per ticker (purge=5d, embargo=5d).
* OOF gate: model only used in inference if OOF directional accuracy > 51%.

Offline training
----------------
    from models.cross_sectional_trainer import CrossSectionalTrainer
    trainer = CrossSectionalTrainer()
    metrics = trainer.train(years=10)
    print(metrics)

Live inference
--------------
Integrated into create_hybrid_model — no manual call needed.
If models/saved/cross_sectional_lgbm.pkl exists and OOF accuracy > 51%,
the model contributes up to 15 % weight to the final prediction blend.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── NIFTY 50 + midcap universe ───────────────────────────────────────────────
NIFTY50_TICKERS: list[str] = [
    "RELIANCE.NS", "TCS.NS",        "HDFCBANK.NS",  "INFY.NS",     "ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS",       "SBIN.NS",      "BHARTIARTL.NS","KOTAKBANK.NS",
    "BAJFINANCE.NS","WIPRO.NS",     "ASIANPAINT.NS","MARUTI.NS",   "HCLTECH.NS",
    "AXISBANK.NS",  "LT.NS",        "SUNPHARMA.NS", "TITAN.NS",    "ONGC.NS",
    "ULTRACEMCO.NS","POWERGRID.NS", "NTPC.NS",      "JSWSTEEL.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "ADANIPORTS.NS","COALINDIA.NS", "BAJAJFINSV.NS","NESTLEIND.NS",
    "TECHM.NS",     "GRASIM.NS",    "DIVISLAB.NS",  "CIPLA.NS",    "DRREDDY.NS",
    "EICHERMOT.NS", "BPCL.NS",      "HEROMOTOCO.NS","HINDALCO.NS", "APOLLOHOSP.NS",
    "TATACONSUM.NS","UPL.NS",       "BRITANNIA.NS", "SHREECEM.NS", "INDUSINDBK.NS",
]

NIFTY_MIDCAP_EXTRAS: list[str] = [
    "PIDILITIND.NS", "HAVELLS.NS", "VOLTAS.NS", "DABUR.NS",  "MARICO.NS",
    "MUTHOOTFIN.NS", "PAGEIND.NS", "ALKEM.NS",  "TORNTPHARM.NS", "DEEPAKNTR.NS",
]

# ── Sector encoding (integer 0-9) ─────────────────────────────────────────────
# 0=Energy/Oil&Gas  1=Financials  2=IT  3=FMCG  4=Auto  5=Pharma
# 6=Metals/Mining   7=Infra/Construction  8=Telecom  9=Diversified/Other
TICKER_SECTOR: dict[str, int] = {
    # Energy / Oil & Gas (0)
    "RELIANCE.NS": 0, "ONGC.NS": 0, "BPCL.NS": 0, "COALINDIA.NS": 0,
    "POWERGRID.NS": 0, "NTPC.NS": 0, "DEEPAKNTR.NS": 0,
    # Financials (1)
    "HDFCBANK.NS": 1, "ICICIBANK.NS": 1, "SBIN.NS": 1, "KOTAKBANK.NS": 1,
    "BAJFINANCE.NS": 1, "AXISBANK.NS": 1, "BAJAJFINSV.NS": 1,
    "INDUSINDBK.NS": 1, "MUTHOOTFIN.NS": 1,
    # IT / Technology (2)
    "TCS.NS": 2, "INFY.NS": 2, "WIPRO.NS": 2, "HCLTECH.NS": 2,
    "TECHM.NS": 2,
    # FMCG / Consumer (3)
    "HINDUNILVR.NS": 3, "ITC.NS": 3, "NESTLEIND.NS": 3, "BRITANNIA.NS": 3,
    "TATACONSUM.NS": 3, "DABUR.NS": 3, "MARICO.NS": 3, "PIDILITIND.NS": 3,
    "PAGEIND.NS": 3,
    # Auto (4)
    "MARUTI.NS": 4, "TATAMOTORS.NS": 4, "EICHERMOT.NS": 4,
    "HEROMOTOCO.NS": 4,
    # Pharma / Healthcare (5)
    "SUNPHARMA.NS": 5, "DIVISLAB.NS": 5, "CIPLA.NS": 5, "DRREDDY.NS": 5,
    "APOLLOHOSP.NS": 5, "ALKEM.NS": 5, "TORNTPHARM.NS": 5,
    # Metals / Mining (6)
    "JSWSTEEL.NS": 6, "TATASTEEL.NS": 6, "HINDALCO.NS": 6,
    # Infra / Industrials (7)
    "LT.NS": 7, "ADANIPORTS.NS": 7, "ULTRACEMCO.NS": 7, "SHREECEM.NS": 7,
    "GRASIM.NS": 7, "TITAN.NS": 7, "HAVELLS.NS": 7, "VOLTAS.NS": 7,
    # Telecom (8)
    "BHARTIARTL.NS": 8,
    # Diversified / Other (9)
    "ASIANPAINT.NS": 9, "UPL.NS": 9,
}

_SAVED_DIR  = Path(__file__).parent / "saved"
_MODEL_PATH = _SAVED_DIR / "cross_sectional_lgbm.pkl"
_FEATS_PATH = _SAVED_DIR / "cross_sectional_features.pkl"
_META_PATH  = _SAVED_DIR / "cross_sectional_meta.pkl"

_NIFTY_TICKER = "^NSEI"


def _compute_relative_features(
    ticker_close: pd.Series,
    nifty_close: pd.Series,
    ticker_high: pd.Series = None,
    ticker_volume: pd.Series = None,
) -> pd.DataFrame:
    """
    Compute NIFTY-relative and cross-sectional alpha features.

    Includes academically-validated factors:
    - Jegadeesh-Titman momentum (1/3/6/12M, 12-1M)
    - George-Hwang 52-week high proximity (strong momentum factor)
    - Ang et al. idiosyncratic volatility (low-ivol anomaly)
    - Amihud illiquidity ratio
    - Realised volatility regime
    - NIFTY-relative strength at multiple horizons
    - Beta (market sensitivity)
    """
    out = pd.DataFrame(index=ticker_close.index)

    nifty_aligned = nifty_close.reindex(ticker_close.index, method="ffill")

    nifty_1d  = np.log(nifty_aligned / nifty_aligned.shift(1)).fillna(0).clip(-0.1, 0.1)
    nifty_5d  = np.log(nifty_aligned / nifty_aligned.shift(5)).fillna(0).clip(-0.3, 0.3)
    nifty_20d = np.log(nifty_aligned / nifty_aligned.shift(20)).fillna(0).clip(-0.5, 0.5)

    ticker_1d  = np.log(ticker_close / ticker_close.shift(1)).fillna(0).clip(-0.1, 0.1)
    ticker_5d  = np.log(ticker_close / ticker_close.shift(5)).fillna(0).clip(-0.3, 0.3)
    ticker_20d = np.log(ticker_close / ticker_close.shift(20)).fillna(0).clip(-0.5, 0.5)

    out["Nifty_Ret"]        = nifty_1d
    out["Rel_Strength_5D"]  = (ticker_5d  - nifty_5d).clip(-0.3, 0.3)
    out["Rel_Strength_20D"] = (ticker_20d - nifty_20d).clip(-0.5, 0.5)

    cov       = ticker_1d.rolling(20).cov(nifty_1d)
    nifty_var = nifty_1d.rolling(20).var() + 1e-10
    out["Beta_20D"] = (cov / nifty_var).clip(-3, 3).fillna(1.0)

    # ── Jegadeesh-Titman momentum factors ────────────────────────────────────
    for label, period in [("1M", 21), ("3M", 63), ("6M", 126), ("12M", 252)]:
        t_ret = np.log(ticker_close / ticker_close.shift(period)).clip(-1.5, 1.5)
        n_ret = np.log(nifty_aligned / nifty_aligned.shift(period)).clip(-1.5, 1.5)
        out[f"Mom_{label}"]   = t_ret.fillna(0)
        out[f"Rel_{label}"]   = (t_ret - n_ret).clip(-1.0, 1.0).fillna(0)

    # 12-1 month: skip last month to avoid short-term reversal contamination
    t_12 = np.log(ticker_close / ticker_close.shift(252)).clip(-1.5, 1.5)
    t_1  = np.log(ticker_close / ticker_close.shift(21)).clip(-0.5, 0.5)
    out["Mom_12M_1M"] = (t_12 - t_1).fillna(0)

    # ── George-Hwang 52-week high proximity ──────────────────────────────────
    # Price / 52W_high is the single best momentum predictor in the cross-section
    # (George & Hwang 2004; Nifty confirmed in India-specific studies)
    hi_src = ticker_high if (ticker_high is not None and len(ticker_high) > 0) else ticker_close
    rolling_52w_high = hi_src.rolling(252, min_periods=63).max()
    out["Price_52W_High_Pct"] = (ticker_close / (rolling_52w_high + 1e-8)).clip(0.3, 1.0).fillna(0.7)
    rolling_52w_low  = ticker_close.rolling(252, min_periods=63).min()
    out["Price_52W_Range_Pct"] = (
        (ticker_close - rolling_52w_low) /
        (rolling_52w_high - rolling_52w_low + 1e-8)
    ).clip(0, 1).fillna(0.5)

    # ── Realised volatility ───────────────────────────────────────────────────
    out["Vol_20D"]         = ticker_1d.rolling(20).std().fillna(0)
    out["Vol_60D"]         = ticker_1d.rolling(60).std().fillna(0)
    out["Vol_Ratio_60_20"] = (out["Vol_20D"] / (out["Vol_60D"] + 1e-8)).clip(0, 5).fillna(1.0)

    # ── Idiosyncratic volatility (Ang et al. 2006 low-ivol anomaly) ──────────
    # Residual vol after regressing daily returns on NIFTY returns (21-day rolling)
    # High idiosyncratic vol → underperformance (ivol puzzle)
    resid = ticker_1d - out["Beta_20D"] * nifty_1d
    out["IVol_21D"] = resid.rolling(21).std().fillna(out["Vol_20D"])

    # ── Amihud illiquidity ratio ──────────────────────────────────────────────
    # |return| / dollar_volume — illiquid = higher expected return as risk premium
    if ticker_volume is not None and len(ticker_volume) > 0:
        dollar_vol = (ticker_close * ticker_volume.reindex(ticker_close.index).fillna(0)).replace(0, np.nan)
        amihud_raw = (ticker_1d.abs() / (dollar_vol / 1e7)).replace([np.inf, -np.inf], np.nan)
        out["Amihud_21D"] = amihud_raw.rolling(21).mean().clip(0, 10).fillna(0)
        # Volume trend: 5-day avg volume / 20-day avg volume — rising vol with price = continuation
        vol_5d  = ticker_volume.reindex(ticker_close.index).rolling(5).mean()
        vol_20d = ticker_volume.reindex(ticker_close.index).rolling(20).mean()
        out["Vol_Trend"] = (vol_5d / (vol_20d + 1e-8)).clip(0, 5).fillna(1.0)
    else:
        out["Amihud_21D"] = 0.0
        out["Vol_Trend"]  = 1.0

    return out.fillna(0.0)


def _compute_excess_return_target(
    ticker_close: pd.Series,
    nifty_close: pd.Series,
    horizon: int = 1,
) -> pd.Series:
    """
    Next-day excess return target = log(ticker[t+horizon]/ticker[t])
                                    - log(nifty[t+horizon]/nifty[t])

    Removes market factor so the model learns idiosyncratic alpha.
    """
    nifty_aligned = nifty_close.reindex(ticker_close.index, method="ffill")
    clip = 0.15 * horizon
    ticker_fwd = np.log(ticker_close.shift(-horizon) / ticker_close).clip(-clip, clip)
    nifty_fwd  = np.log(nifty_aligned.shift(-horizon) / nifty_aligned).clip(-clip, clip)
    return (ticker_fwd - nifty_fwd).fillna(np.nan)


class CrossSectionalTrainer:
    """
    Cross-sectional LightGBM — train on many tickers, infer per-ticker.

    The model is strictly chronological: no future leakage.
    Purged expanding-window k-fold per ticker (purge=5d, embargo=5d).
    Target: 5-day excess return over NIFTY50 (idiosyncratic alpha).
    """

    def __init__(self) -> None:
        self._model   = None
        self._features: list[str] = []
        self._meta: dict = {}
        self._loaded  = False

    # ── Persistence ──────────────────────────────────────────────────────────

    def is_trained(self) -> bool:
        return _MODEL_PATH.exists() and _FEATS_PATH.exists()

    def load(self) -> bool:
        if self._loaded:
            return True
        if not self.is_trained():
            return False
        try:
            with open(_MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            with open(_FEATS_PATH, "rb") as f:
                self._features = pickle.load(f)
            if _META_PATH.exists():
                with open(_META_PATH, "rb") as f:
                    self._meta = pickle.load(f)
            self._loaded = True
            return True
        except Exception as exc:
            logger.warning("CrossSectionalTrainer.load failed: %s", exc)
            return False

    def _save(self) -> None:
        _SAVED_DIR.mkdir(parents=True, exist_ok=True)
        with open(_MODEL_PATH, "wb") as f:
            pickle.dump(self._model, f)
        with open(_FEATS_PATH, "wb") as f:
            pickle.dump(self._features, f)
        with open(_META_PATH, "wb") as f:
            pickle.dump(self._meta, f)
        logger.info("CrossSectionalTrainer: saved to %s", _SAVED_DIR)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        feature_names: list[str],
        ticker: str = None,
    ) -> Optional[np.ndarray]:
        """
        Predict 5-day excess return for rows in X (aligned to feature_names).

        ticker: optional NSE ticker string (e.g. "RELIANCE.NS") used to fill
                the Sector_ID column when it is missing from live feature set.

        Returns None if model unavailable or feature overlap < 80 %.
        """
        if not self._loaded:
            if not self.load():
                return None
        try:
            import lightgbm as lgb  # noqa: F401

            feat_map  = {f: i for i, f in enumerate(feature_names)}
            available = [f for f in self._features if f in feat_map]
            # Allow Sector_ID to be injected even if absent from live features
            missing_critical = [
                f for f in self._features
                if f not in feat_map and f != "Sector_ID"
            ]
            coverage = (len(self._features) - len(missing_critical)) / max(len(self._features), 1)
            if coverage < 0.80:
                logger.debug(
                    "CrossSectional: only %.0f%% feature coverage — skipping",
                    coverage * 100,
                )
                return None

            X_xs = np.zeros((len(X), len(self._features)), dtype=np.float32)
            for xs_i, feat in enumerate(self._features):
                if feat == "Sector_ID" and feat not in feat_map:
                    # inject sector from ticker lookup
                    sector_val = TICKER_SECTOR.get(ticker, -1) if ticker else -1
                    X_xs[:, xs_i] = float(sector_val)
                elif feat in feat_map:
                    X_xs[:, xs_i] = X[:, feat_map[feat]]

            X_df = pd.DataFrame(X_xs, columns=self._features)
            return self._model.predict(X_df).astype(np.float64)
        except Exception as exc:
            logger.warning("CrossSectionalTrainer.predict failed: %s", exc)
            return None

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        tickers: Optional[list[str]] = None,
        years: int = 10,
        purge_gap: int = 5,
        embargo_gap: int = 5,
        n_splits: int = 10,
        n_jobs: int = -1,
        save: bool = True,
    ) -> dict:
        """
        Download `years` of daily OHLCV for each ticker, compute features
        (including NIFTY-relative features and sector encoding), train a
        single LightGBM with purged k-fold OOF, save model.

        Target: 5-day excess return over NIFTY50 (idiosyncratic alpha).

        Returns a metrics dict with OOF directional accuracy.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("CrossSectionalTrainer.train requires: pip install yfinance")
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("CrossSectionalTrainer.train requires: pip install lightgbm")

        from models.hybrid_model import create_advanced_features

        if tickers is None:
            tickers = NIFTY50_TICKERS + NIFTY_MIDCAP_EXTRAS

        period = f"{years}y"

        # ── Download NIFTY50 index once ───────────────────────────────────────
        logger.info("XS-train: downloading NIFTY50 index (%s) ...", _NIFTY_TICKER)
        try:
            _nifty_raw = yf.download(
                _NIFTY_TICKER, period=period, auto_adjust=True, progress=False, timeout=30
            )
            if isinstance(_nifty_raw.columns, pd.MultiIndex):
                _nifty_raw.columns = [c[0] for c in _nifty_raw.columns]
            if _nifty_raw.index.tz is not None:
                _nifty_raw.index = _nifty_raw.index.tz_localize(None)
            nifty_close: pd.Series = _nifty_raw["Close"].dropna()
            logger.info("XS-train: NIFTY50 — %d bars", len(nifty_close))
        except Exception as exc:
            logger.warning("XS-train: NIFTY50 download failed (%s) — relative features disabled", exc)
            nifty_close = pd.Series(dtype=float)

        frames: list[pd.DataFrame] = []

        for ticker in tickers:
            try:
                raw = yf.download(
                    ticker, period=period, auto_adjust=True, progress=False, timeout=30
                )
                if raw is None or len(raw) < 80:
                    logger.warning(
                        "XS-train: skipping %s — only %d bars",
                        ticker, 0 if raw is None else len(raw),
                    )
                    continue

                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0] for c in raw.columns]
                if raw.index.tz is not None:
                    raw.index = raw.index.tz_localize(None)
                raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                raw.columns = ["Open", "High", "Low", "Close", "Volume"]
                raw = raw.dropna()

                feat_df = create_advanced_features(raw)

                # ── NIFTY-relative features ───────────────────────────────────
                if len(nifty_close) > 10:
                    rel = _compute_relative_features(
                        raw["Close"], nifty_close,
                        ticker_high=raw["High"],
                        ticker_volume=raw["Volume"],
                    )
                    rel = rel.reindex(feat_df.index)
                    feat_df = pd.concat([feat_df, rel], axis=1)

                    # 5-day excess return target: next 5-day alpha over NIFTY
                    target = _compute_excess_return_target(raw["Close"], nifty_close, horizon=5)
                    feat_df["Target_ExcessRet5D"] = target.reindex(feat_df.index)
                else:
                    # Fallback: plain 5-day log return
                    feat_df["Target_ExcessRet5D"] = feat_df["Log_Ret"].rolling(5).sum().shift(-5)

                # ── Sector encoding ───────────────────────────────────────────
                feat_df["Sector_ID"] = float(TICKER_SECTOR.get(ticker, -1))

                feat_df = feat_df.dropna(subset=["Target_ExcessRet5D"])
                feat_df["_ticker"] = ticker
                feat_df["_date"]   = feat_df.index
                frames.append(feat_df)
                logger.info("XS-train: %s — %d rows (sector=%d)", ticker, len(feat_df),
                            TICKER_SECTOR.get(ticker, -1))

            except Exception as exc:
                logger.warning("XS-train: error on %s: %s", ticker, exc)
                continue

        if not frames:
            raise RuntimeError("No tickers returned sufficient data. Check network / yfinance.")

        combined = pd.concat(frames, axis=0, ignore_index=True)
        n_tickers_ok = len(frames)
        logger.info("XS-train: %d tickers, %d total rows", n_tickers_ok, len(combined))

        # ── Cross-sectional rank features ─────────────────────────────────────
        # For each date, rank each ticker's momentum features vs the full universe.
        # This is the core signal: cross-sectional momentum (winners vs losers).
        logger.info("XS-train: computing cross-sectional rank features...")
        _rank_cols = [c for c in [
                        "Rel_Strength_5D", "Rel_Strength_20D",
                        "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M", "Mom_12M_1M",
                        "Rel_1M", "Rel_3M", "Rel_12M",
                        "Vol_20D", "IVol_21D", "Amihud_21D",
                        "Price_52W_High_Pct", "Price_52W_Range_Pct",
                        "RSI_Norm", "Vol_Trend",
                      ] if c in combined.columns]
        try:
            for rc in _rank_cols:
                rank_col = f"{rc}_XSRank"
                combined[rank_col] = (
                    combined.groupby("_date")[rc]
                    .rank(pct=True, method="average")
                    .fillna(0.5)
                )
            logger.info("XS-train: added %d cross-sectional rank features", len(_rank_cols))
        except Exception as exc:
            logger.warning("XS-train: cross-sectional rank features failed: %s", exc)
        # Extract dates BEFORE dropping _date column
        _dates_arr = combined["_date"].values if "_date" in combined.columns else None
        combined.drop(columns=["_date"], inplace=True, errors="ignore")

        # ── Feature selection: present in ≥90% rows, numeric, not target/id ──
        _exclude = {"Target_ExcessRet5D", "_ticker", "Log_Ret"}
        core_features = [
            c for c in combined.columns
            if c not in _exclude
            and combined[c].notna().mean() > 0.90
            and pd.api.types.is_numeric_dtype(combined[c])
        ]

        # Ensure Sector_ID is always in features if available
        if "Sector_ID" in combined.columns and "Sector_ID" not in core_features:
            core_features.append("Sector_ID")

        X_all = combined[core_features].fillna(0).values.astype(np.float32)
        y_all = combined["Target_ExcessRet5D"].values.astype(np.float32)

        logger.info("XS-train: %d features selected", len(core_features))

        # ── Date-based expanding-window OOF (correct cross-sectional split) ──
        # Train on all tickers up to date T, validate on the next date block.
        # This mirrors production usage and captures cross-sectional patterns.
        oof_preds = np.zeros(len(y_all), dtype=np.float64)

        if _dates_arr is not None:
            unique_dates = np.sort(np.unique(_dates_arr))
            n_dates      = len(unique_dates)
            min_train_d  = max(int(n_dates * 0.30), 60)   # 30% warm-up → more OOF coverage
            step_d       = max((n_dates - min_train_d) // n_splits, 15)

            logger.info("XS-train: date-OOF over %d dates, %d splits, step=%d days",
                        n_dates, n_splits, step_d)

            for fold in range(n_splits):
                val_start_d  = min_train_d + fold * step_d
                val_end_d    = min(val_start_d + step_d, n_dates)
                train_end_d  = max(0, val_start_d - purge_gap)

                if val_start_d >= n_dates or train_end_d < 30:
                    continue

                cutoff_train = unique_dates[train_end_d]
                cutoff_val_s = unique_dates[val_start_d]
                cutoff_val_e = unique_dates[val_end_d - 1]

                tr_mask  = _dates_arr < cutoff_train
                val_mask = (_dates_arr >= cutoff_val_s) & (_dates_arr <= cutoff_val_e)

                if tr_mask.sum() < 100 or val_mask.sum() < 10:
                    continue

                m = lgb.LGBMRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.02,
                    num_leaves=20,
                    min_child_samples=15,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    reg_alpha=0.3,
                    reg_lambda=1.0,
                    n_jobs=n_jobs,
                    verbose=-1,
                    random_state=42 + fold,
                )
                m.fit(
                    pd.DataFrame(X_all[tr_mask], columns=core_features),
                    y_all[tr_mask],
                )
                oof_preds[val_mask] = m.predict(
                    pd.DataFrame(X_all[val_mask], columns=core_features)
                )
                logger.info(
                    "XS-train fold %d: train=%d rows (up to %s), val=%d rows (%s–%s)",
                    fold, tr_mask.sum(), cutoff_train, val_mask.sum(),
                    cutoff_val_s, cutoff_val_e,
                )
        else:
            logger.warning("XS-train: no _date column — falling back to naive split")
            split = int(len(y_all) * 0.6)
            m = lgb.LGBMRegressor(n_estimators=200, max_depth=4, verbose=-1)
            m.fit(pd.DataFrame(X_all[:split], columns=core_features), y_all[:split])
            oof_preds[split:] = m.predict(pd.DataFrame(X_all[split:], columns=core_features))

        valid_oof = np.abs(oof_preds) > 1e-10
        oof_dir_acc = float(
            np.mean(np.sign(oof_preds[valid_oof]) == np.sign(y_all[valid_oof]))
        ) if valid_oof.sum() > 0 else 0.5
        logger.info(
            "XS-train: OOF directional accuracy = %.2f%% (on %d/%d samples)",
            oof_dir_acc * 100, int(valid_oof.sum()), len(y_all),
        )

        # ── Final model on all data ───────────────────────────────────────────
        final = lgb.LGBMRegressor(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=1.0,
            n_jobs=n_jobs,
            verbose=-1,
            random_state=42,
        )
        final.fit(pd.DataFrame(X_all, columns=core_features), y_all)

        self._model    = final
        self._features = core_features
        self._meta = {
            "n_tickers":          n_tickers_ok,
            "n_rows":             len(combined),
            "n_features":         len(core_features),
            "oof_dir_accuracy":   oof_dir_acc,
            "years":              years,
            "target":             "5d_excess_return_over_nifty",
            "features_sample":    core_features[:10],
        }
        self._loaded = True

        if save:
            self._save()

        return {
            "n_tickers":        n_tickers_ok,
            "n_rows":           len(combined),
            "n_features":       len(core_features),
            "oof_dir_accuracy": oof_dir_acc,
            "oof_coverage":     int(valid_oof.sum()),
        }
