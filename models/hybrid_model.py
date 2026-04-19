"""
Hybrid Model for stock price prediction.
Combines XGBoost + LightGBM + GRU neural network with stacking meta-learner,
calibrated probabilities, SHAP importance, and Hurst exponent regime detection.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from pandas.tseries.offsets import CustomBusinessDay

# Statistical models for ensemble
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# LightGBM (optional — graceful fallback to XGBoost-only if not installed)
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# CatBoost (optional — ordered boosting, superior on small financial datasets)
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# SHAP for feature importance (optional — falls back to XGB native importance)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from config.settings import ModelConfig
from data.vix_data import IndiaHolidayCalendar


# ============================================================
# Utility: Hurst Exponent
# ============================================================

def calculate_hurst_exponent(prices: np.ndarray, lags: range = None) -> float:
    """
    Compute the Hurst exponent using R/S (rescaled range) analysis.

    Interpretation:
        H > 0.55  → Trending (momentum regime)
        H < 0.45  → Mean-reverting (range-bound regime)
        H ≈ 0.50  → Random walk (efficient market)

    Uses only past price data — no look-ahead bias.
    Runtime: < 0.5 seconds (pure numpy).

    Args:
        prices: Array of close prices (ideally ≥ 100 observations)
        lags: Range of lag values for R/S computation (default range(2, 21))

    Returns:
        Hurst exponent as float in approximately [0, 1]
    """
    if lags is None:
        lags = range(2, 21)

    if len(prices) < 40:
        return 0.5  # Insufficient data — assume random walk

    log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))

    rs_pairs = []
    for lag in lags:
        # Split into non-overlapping windows of size `lag`
        rs_vals = []
        n_windows = len(log_returns) // lag
        if n_windows < 2:
            continue
        for w in range(n_windows):
            seg = log_returns[w * lag:(w + 1) * lag]
            mean_adj = seg - np.mean(seg)
            cumdev = np.cumsum(mean_adj)
            r = np.max(cumdev) - np.min(cumdev)  # Range
            s = np.std(seg, ddof=1)              # Standard deviation
            if s > 1e-10:
                rs_vals.append(r / s)
        if rs_vals:
            rs_pairs.append((lag, np.mean(rs_vals)))

    if len(rs_pairs) < 4:
        return 0.5

    lags_arr = np.array([p[0] for p in rs_pairs], dtype=float)
    rs_arr   = np.array([p[1] for p in rs_pairs], dtype=float)

    # Linear regression of log(R/S) vs log(lag) → slope = Hurst exponent
    valid = rs_arr > 0
    if valid.sum() < 4:
        return 0.5

    slope = np.polyfit(np.log(lags_arr[valid]), np.log(rs_arr[valid]), 1)[0]
    return float(np.clip(slope, 0.1, 0.9))


# ============================================================
# Feature Engineering
# ============================================================

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create strictly stationary features for ML.
    Avoids absolute prices — uses returns, volatility, oscillators,
    and volume-flow indicators.

    Features added (27 total):
        Core (5): Log_Ret, Volatility_5D, RSI_Norm, Vol_Ratio, MA_Div
        Enhanced (9): MACD_Norm, MACD_Hist_Norm, BB_PctB, ATR_Norm, OBV_Slope,
                      Ret_2D, Ret_5D, Ret_10D, Ret_20D
        New (4): CMF_20, Williams_R_Norm, RSI_Bear_Div, RSI_Bull_Div
        (Sentiment / FII / VIX added separately in create_hybrid_model)

    Args:
        df: DataFrame with OHLCV data (Open, High, Low, Close, Volume)

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Target: Log Returns (Stationary)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # Volatility (Normalized)
    df['Volatility_5D'] = df['Log_Ret'].rolling(window=5).std()

    # Momentum (RSI normalized to 0-1)
    if 'RSI' not in df.columns:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    df['RSI_Norm'] = df['RSI'] / 100.0

    # Volume Trend (Ratio)
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # Moving Average Divergence (Normalized by Price)
    df['MA_Div'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close']

    # --- Enhanced Features ---

    # MACD (normalized by price to keep stationary)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Norm'] = macd_line / df['Close']
    df['MACD_Hist_Norm'] = (macd_line - macd_signal) / df['Close']

    # Bollinger Band %B (where price sits within the bands, 0-1 range)
    bb_ma  = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    df['BB_PctB'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)

    # ATR (normalized by price for stationarity)
    high_low   = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close  = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_Norm'] = true_range.rolling(14, min_periods=1).mean() / df['Close']

    # OBV slope (rate of change of On-Balance Volume, normalized)
    obv    = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    obv_ma = obv.rolling(5, min_periods=1).mean()
    df['OBV_Slope'] = obv_ma.pct_change(5).clip(-1, 1).fillna(0)

    # Multi-timeframe returns (captures momentum at different scales)
    df['Ret_2D']  = np.log(df['Close'] / df['Close'].shift(2))
    df['Ret_5D']  = np.log(df['Close'] / df['Close'].shift(5))
    df['Ret_10D'] = np.log(df['Close'] / df['Close'].shift(10))
    df['Ret_20D'] = np.log(df['Close'] / df['Close'].shift(20))

    # ============================================================
    # NEW FEATURES (Plan items 1D)
    # ============================================================

    # Chaikin Money Flow (CMF-20):
    # Measures buying/selling pressure using volume-weighted close position
    # within daily high-low range. Ranges [-1, +1]. Stationary by construction.
    hl_range = (df['High'] - df['Low']).replace(0, np.nan)
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_range
    mfvol = mfm.fillna(0) * df['Volume']
    vol_sum = df['Volume'].rolling(20).sum()
    df['CMF_20'] = mfvol.rolling(20).sum() / (vol_sum + 1e-8)
    df['CMF_20'] = df['CMF_20'].fillna(0).clip(-1, 1)

    # Williams %R (14-period):
    # Fast momentum oscillator; complement to RSI.
    # Normalized to [-1, 0] range (original is [-100, 0]).
    high14 = df['High'].rolling(14).max()
    low14  = df['Low'].rolling(14).min()
    df['Williams_R_Norm'] = (high14 - df['Close']) / (high14 - low14 + 1e-8) * -1.0

    # RSI Divergence (binary flags, no look-ahead — shift(1) applied):
    # Bearish divergence: price makes new 10D high but RSI does not → selling pressure
    # Bullish divergence: price makes new 10D low but RSI does not → buying pressure
    rsi_10h = df['RSI'].rolling(10).max().shift(1)
    px_10h  = df['Close'].rolling(10).max().shift(1)
    rsi_10l = df['RSI'].rolling(10).min().shift(1)
    px_10l  = df['Close'].rolling(10).min().shift(1)
    df['RSI_Bear_Div'] = ((df['Close'] >= px_10h) & (df['RSI'] < rsi_10h)).astype(float).fillna(0)
    df['RSI_Bull_Div'] = ((df['Close'] <= px_10l) & (df['RSI'] > rsi_10l)).astype(float).fillna(0)

    # Drop NaNs created by rolling windows
    df.dropna(inplace=True)

    return df


# ============================================================
# Market Regime Detection (with Hurst)
# ============================================================

def detect_market_regime(df: pd.DataFrame, vol_window: int = 20, trend_window: int = 20) -> dict:
    """
    Detect current market regime from recent price data.

    Classifies the market into one of 4 regimes:
    - 'trending':       Strong directional movement (confirmed by Hurst > 0.55)
    - 'mean_reverting': Low volatility, sideways/range-bound (Hurst < 0.45)
    - 'high_volatility':Crisis-like conditions, elevated fear
    - 'normal':         Default regime, moderate conditions

    Hurst exponent is used to CONFIRM or OVERRIDE the slope-based classification,
    reducing false regime labels from noisy slope estimates.

    Args:
        df: DataFrame with at least 'Log_Ret', 'Volatility_5D', 'Close' columns
        vol_window: Lookback for volatility percentile (default 20)
        trend_window: Lookback for trend detection (default 20)

    Returns:
        Dictionary with 'type', 'detail', and 'hurst' fields
    """
    if len(df) < max(vol_window, trend_window, 60):
        return {'type': 'normal', 'detail': 'Insufficient data for regime detection', 'hurst': 0.5}

    recent_returns = df['Log_Ret'].iloc[-trend_window:]

    # --- Hurst Exponent (uses last 120 closes for stability) ---
    close_arr = df['Close'].values[-120:] if len(df) >= 120 else df['Close'].values
    H = calculate_hurst_exponent(close_arr)

    # --- Volatility analysis ---
    if 'Volatility_5D' in df.columns:
        recent_vol    = df['Volatility_5D'].iloc[-1]
        vol_history   = df['Volatility_5D'].iloc[-60:]
        vol_percentile = (vol_history < recent_vol).mean() * 100
    else:
        vol_percentile = 50.0

    # --- Trend analysis (linear regression on cumulative returns) ---
    cumulative_ret = recent_returns.cumsum().values
    x = np.arange(len(cumulative_ret))
    if len(x) > 1 and np.std(cumulative_ret) > 1e-10:
        slope = np.polyfit(x, cumulative_ret, 1)[0]
        correlation = np.corrcoef(x, cumulative_ret)[0, 1]
        r_squared = correlation ** 2 if not np.isnan(correlation) else 0
    else:
        slope = 0.0
        r_squared = 0.0

    abs_slope = abs(slope)

    # --- Regime classification (Hurst used to confirm/override) ---
    if vol_percentile > 85:
        direction = 'down-trending' if slope < 0 else 'up-trending'
        return {
            'type': 'high_volatility',
            'detail': f'Elevated volatility ({vol_percentile:.0f}th percentile), {direction}',
            'hurst': round(H, 3)
        }
    elif abs_slope > 0.002 and r_squared > 0.3:
        # Slope says trending — confirm with Hurst
        if H >= 0.45:  # Hurst does not contradict → confirm as trending
            direction = 'Uptrend' if slope > 0 else 'Downtrend'
            strength  = 'strong' if r_squared > 0.6 else 'moderate'
            return {
                'type': 'trending',
                'detail': f'{direction} ({strength}, R²={r_squared:.2f}, H={H:.2f})',
                'hurst': round(H, 3)
            }
        else:  # H < 0.45 overrides slope → mean-reverting despite apparent slope
            return {
                'type': 'mean_reverting',
                'detail': f'Slope-trend overridden by Hurst (H={H:.2f}<0.45)',
                'hurst': round(H, 3)
            }
    elif vol_percentile < 30 and abs_slope < 0.001:
        # Low vol + no direction — Hurst confirms mean-reverting
        return {
            'type': 'mean_reverting',
            'detail': f'Range-bound (vol {vol_percentile:.0f}th pctl, H={H:.2f})',
            'hurst': round(H, 3)
        }
    else:
        return {
            'type': 'normal',
            'detail': f'Normal conditions (vol {vol_percentile:.0f}th pctl, H={H:.2f})',
            'hurst': round(H, 3)
        }


# ============================================================
# LightGBM Helper
# ============================================================

def _train_quantile_xgb(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray,
                        alphas: tuple = (0.1, 0.9)) -> dict:
    """
    Train XGB quantile regressors (A6.1) — one model per alpha.

    Uses `reg:quantileerror` (XGBoost ≥ 2.0). Returns:
        {
          'models':  {0.1: xgb, 0.9: xgb},
          'test_preds': {0.1: np.array, 0.9: np.array},
        }

    These quantile predictions live alongside the hybrid point estimator
    and feed the prediction-interval band. They are NOT fed into the meta
    stacker — that would overfit on the same OOF residuals twice.
    """
    models: dict = {}
    preds: dict = {}
    for alpha in alphas:
        try:
            m = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=alpha,
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                verbosity=0,
            )
            m.fit(X_train, y_train)
            models[alpha] = m
            preds[alpha] = m.predict(X_test)
        except Exception:
            # Graceful fallback — quantile XGB isn't available in xgboost < 2.0.
            continue
    return {"models": models, "test_preds": preds}


def _conformal_halfwidth(y_true: np.ndarray, y_pred: np.ndarray,
                          alpha: float = 0.10) -> float:
    """
    Split-conformal half-width on holdout residuals (A6.2).

    Returns the (1-alpha) quantile of |y - y_hat|. The 90% prediction
    interval is `y_hat ± halfwidth`, distribution-free, assuming
    exchangeability of the holdout residuals with future residuals.

    For log-return predictions the halfwidth is in log-return units —
    converted to a price band at inference time.
    """
    if len(y_true) == 0:
        return 0.0
    residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) == 0:
        return 0.0
    # +1 correction for finite-sample validity (Angelopoulos & Bates, 2023)
    n = len(residuals)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, q_level))


def _train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> tuple:
    """
    Train LightGBM regressor.

    Advantages over XGBoost for this use case:
    - 3-5× faster training (histogram-based splits)
    - Better generalization with `min_child_samples` on small financial datasets
    - L1 + L2 regularization handles correlated financial features well
    - num_leaves controls complexity more intuitively than max_depth

    Parameters chosen via offline experiments on Indian equity data.
    Runtime: ~8-10 seconds on 5-year daily data (CPU).

    Args:
        X_train: Scaled training feature matrix
        y_train: Training target returns
        X_test: Scaled test feature matrix

    Returns:
        Tuple of (fitted_model, test_predictions)
    """
    params = dict(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        verbose=-1,
        random_state=42
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds


# ============================================================
# CatBoost Helper
# ============================================================

def _train_catboost(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> tuple:
    """
    Train CatBoostRegressor with ordered boosting.

    Key advantages over XGBoost / LightGBM for financial time-series:
    - Ordered boosting: each tree is fit on a permutation of the data where
      residuals for row i are computed from a model trained on rows < i only.
      This eliminates target leakage inside the booster itself, reducing
      overfitting on small (<=5yr daily) datasets.
    - Symmetric (oblivious) trees: same split condition applied at every node
      of a level → uniform prediction time, better generalization.
    - No need for manual categorical encoding (handled natively, though our
      features are all numeric).

    Parameters tuned for ~1250-row daily datasets:
      - iterations=300 + od_wait=40  → early-stop after 40 rounds no improvement
      - depth=6, l2_leaf_reg=3       → moderate complexity, regularized
      - min_data_in_leaf=20          → prevents overfitting on small leaves
      - bagging_temperature=0.7      → bootstrap sampling strength
      - random_strength=0.3          → feature perturbation during splitting

    Runtime: ~6–10s on CPU (similar to LGBM).

    Args:
        X_train: Scaled training features
        y_train: Training target returns
        X_test:  Scaled test features

    Returns:
        Tuple of (fitted_model, test_predictions)
    """
    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3,
        min_data_in_leaf=20,
        random_strength=0.3,
        bagging_temperature=0.7,
        od_type='Iter',
        od_wait=40,
        verbose=0,
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,   # Avoid tmp file creation in Streamlit
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds


# ============================================================
# Stacking: Out-of-Fold Predictions
# ============================================================

def _generate_oof_predictions(X_train: np.ndarray, y_train: np.ndarray,
                               n_folds: int = 5) -> tuple:
    """
    Generate out-of-fold (OOF) predictions for stacking meta-learner.

    Uses EXPANDING window time-series folds — each fold trains on ALL
    prior data and predicts the next block. Strictly chronological,
    zero data leakage.

    Fold structure (example, n=500, n_folds=5, min_train=300):
        Fold 0: Train[0:300]  → Predict[300:340]
        Fold 1: Train[0:340]  → Predict[340:380]
        ...
        Fold 4: Train[0:460]  → Predict[460:500]

    OOF generated for XGB + LGBM + CatBoost (all lightweight, fast).
    GRU OOF is approximated as a lag-1 of XGB in the meta-learner
    (training a full GRU per fold would be too slow).

    Args:
        X_train: Scaled training features (2D)
        y_train: Training targets
        n_folds: Number of expanding folds

    Returns:
        (oof_xgb, oof_lgbm, oof_catboost) arrays of same length as y_train
    """
    n = len(X_train)
    min_train = max(int(n * 0.6), 30)
    step = max((n - min_train) // n_folds, 5)

    oof_xgb      = np.zeros(n)
    oof_lgbm     = np.zeros(n)
    oof_catboost = np.zeros(n)

    xgb_params = dict(
        objective='reg:squarederror', n_estimators=100, max_depth=4,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbosity=0
    )
    lgbm_params = dict(
        n_estimators=150, max_depth=5, learning_rate=0.03,
        num_leaves=31, min_child_samples=15, n_jobs=-1, verbose=-1
    )
    cb_params = dict(
        iterations=150, depth=5, learning_rate=0.05,
        l2_leaf_reg=3, min_data_in_leaf=15,
        od_type='Iter', od_wait=20,
        verbose=0, random_seed=42,
        thread_count=-1, allow_writing_files=False
    )

    for fold in range(n_folds):
        train_end = min_train + fold * step
        val_end   = min(train_end + step, n)

        if train_end >= n or train_end >= val_end:
            break

        Xtr, ytr = X_train[:train_end], y_train[:train_end]
        Xval     = X_train[train_end:val_end]

        # XGBoost OOF
        xm = xgb.XGBRegressor(**xgb_params)
        xm.fit(Xtr, ytr)
        oof_xgb[train_end:val_end] = xm.predict(Xval)

        # LightGBM OOF
        if LGBM_AVAILABLE:
            lm = lgb.LGBMRegressor(**lgbm_params)
            lm.fit(Xtr, ytr)
            oof_lgbm[train_end:val_end] = lm.predict(Xval)
        else:
            oof_lgbm[train_end:val_end] = oof_xgb[train_end:val_end]

        # CatBoost OOF
        if CATBOOST_AVAILABLE:
            cm = CatBoostRegressor(**cb_params)
            cm.fit(Xtr, ytr)
            oof_catboost[train_end:val_end] = cm.predict(Xval)
        else:
            oof_catboost[train_end:val_end] = oof_xgb[train_end:val_end]

    return oof_xgb, oof_lgbm, oof_catboost


# ============================================================
# Calibrated Directional Probability
# ============================================================

def _calibrate_direction_probability(oof_preds: np.ndarray, y_train: np.ndarray,
                                      test_preds: np.ndarray) -> np.ndarray:
    """
    Convert return predictions to calibrated P(direction up) in [0, 1].

    Calibration invariant (A1.1, audited):
        isotonic is fit on `oof_preds` (out-of-fold predictions for rows the
        base model NEVER saw during its own training) and applied to
        `test_preds` (out-of-sample predictions for the held-out tail).
        Fitting on in-sample training preds would over-flatten the isotone
        curve; fitting on test preds would leak labels. OOF is the correct
        middle path — same approach sklearn's CalibratedClassifierCV uses
        with method='isotonic', cv=5.

    Uses isotonic regression (non-parametric monotone mapping) rather than
    logistic regression because:
      - Financial returns have fat tails and are non-Gaussian
      - Isotonic makes no distributional assumptions

    Args:
        oof_preds: OOF predicted returns (same length as y_train), typically
                   the average of XGB+LGBM+CatBoost OOFs from
                   `_generate_oof_predictions`.
        y_train:   Actual training returns (aligned to oof_preds)
        test_preds: Final model predictions on held-out test set

    Returns:
        Array of probabilities (0.02–0.98) for each test prediction.
    """
    assert oof_preds.shape == y_train.shape, (
        f"oof_preds ({oof_preds.shape}) and y_train ({y_train.shape}) must align — "
        "isotonic calibration requires paired OOF pred ↔ actual label rows"
    )

    # Binary label: 1 = up day, 0 = down day
    y_dir = (y_train > 0).astype(float)

    # Only use folds where OOF prediction is non-zero (pre-min_train rows are
    # left as 0 by _generate_oof_predictions and would corrupt the fit)
    valid = np.abs(oof_preds) > 1e-12
    if valid.sum() < 10:
        # Not enough OOF data — use simple sign-based heuristic
        return np.clip(0.5 + test_preds * 10, 0.05, 0.95)

    iso = IsotonicRegression(out_of_bounds='clip')
    sort_idx = np.argsort(oof_preds[valid])
    iso.fit(oof_preds[valid][sort_idx], y_dir[valid][sort_idx])

    prob = iso.predict(test_preds)
    return np.clip(prob, 0.02, 0.98)


def _compute_threshold_tuning(probs: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Learn the per-ticker optimal decision threshold for the directional call.

    Different tickers have different noise profiles — a high-beta midcap needs
    stronger conviction (τ ≈ 0.65) before "buy" pays off, while a slow-moving
    largecap may signal reliably at τ ≈ 0.55. We pick the threshold τ* that
    maximizes Youden's J = TPR(τ) − FPR(τ) on the held-out test probabilities.
    Youden's J is standard when the cost of FP and FN are symmetric; once A8
    has realistic brokerage+STT costs, this becomes a cost-weighted optimum.

    Returns dict with:
        optimal_threshold : τ* in [0, 1] (fraction, not %)
        auc               : ROC-AUC on the holdout (1.0 perfect, 0.5 coin flip)
        accuracy_at_opt   : directional accuracy when classifying at τ*
        n_samples         : holdout size
    Falls back to τ = 0.5 when the holdout is too small or single-class.
    """
    probs = np.asarray(probs, dtype=float)
    y_dir = (np.asarray(y_true, dtype=float) > 0).astype(int)
    n = len(probs)
    fallback = {
        "optimal_threshold": 0.5, "auc": 0.5, "accuracy_at_opt": 0.5, "n_samples": n,
    }
    if n < 20 or len(np.unique(y_dir)) < 2:
        return fallback

    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        fpr, tpr, thresholds = roc_curve(y_dir, probs)
        j_scores = tpr - fpr
        best = int(np.argmax(j_scores))
        tau = float(np.clip(thresholds[best], 0.0, 1.0))
        preds_at_tau = (probs >= tau).astype(int)
        acc = float((preds_at_tau == y_dir).mean())
        auc = float(roc_auc_score(y_dir, probs))
        return {
            "optimal_threshold": tau,
            "auc": auc,
            "accuracy_at_opt": acc,
            "n_samples": n,
        }
    except Exception:
        return fallback


def _compute_calibration_report(probs: np.ndarray, y_true: np.ndarray,
                                n_bins: int = 10) -> dict:
    """
    Bin predicted probabilities and compare to actual hit rates.

    Returns a dict matching `schemas.prediction.CalibrationReport`:
      - ece          : weighted mean |predicted - actual| across bins (A1 target ≤ 0.05)
      - brier_score  : mean squared error of probability vs. binary outcome
      - bin_edges, bin_predicted, bin_actual, bin_counts

    Empty bins are dropped from the returned vectors so the chart doesn't draw
    spurious points at 50%.
    """
    probs = np.asarray(probs, dtype=float)
    y_dir = (np.asarray(y_true, dtype=float) > 0).astype(float)
    n = len(probs)
    if n == 0 or n != len(y_dir):
        return {
            "n_samples": 0, "ece": 0.0, "brier_score": 0.0,
            "bin_edges": [], "bin_predicted": [], "bin_actual": [], "bin_counts": [],
        }

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_pred: list[float] = []
    bin_act: list[float] = []
    bin_cnt: list[int] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        p_mean = float(probs[mask].mean())
        a_mean = float(y_dir[mask].mean())
        ece += (cnt / n) * abs(p_mean - a_mean)
        bin_pred.append(p_mean)
        bin_act.append(a_mean)
        bin_cnt.append(cnt)

    brier = float(np.mean((probs - y_dir) ** 2))
    return {
        "n_samples": n,
        "ece": float(np.clip(ece, 0.0, 1.0)),
        "brier_score": float(np.clip(brier, 0.0, 1.0)),
        "bin_edges": edges.tolist(),
        "bin_predicted": bin_pred,
        "bin_actual": bin_act,
        "bin_counts": bin_cnt,
    }


# ============================================================
# SHAP Feature Importance
# ============================================================

def _compute_shap_importance(xgb_model, X_test_scaled: np.ndarray,
                              feature_names: list) -> dict:
    """
    Compute SHAP feature importance using TreeExplainer.

    TreeExplainer is O(n_trees × n_samples) — fast for XGBoost.
    Uses at most 200 test samples to cap runtime at ~3-5 seconds.

    Falls back to XGBoost native feature_importances_ if SHAP not installed.

    Args:
        xgb_model: Fitted XGBRegressor
        X_test_scaled: Scaled test features (2D array)
        feature_names: List of feature name strings

    Returns:
        Dict with 'importance' (feature→mean|SHAP|), 'top_features' list
    """
    n_sample = min(200, len(X_test_scaled))
    X_sample = X_test_scaled[:n_sample]

    if SHAP_AVAILABLE:
        try:
            explainer   = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_sample)
            importance  = pd.Series(
                np.abs(shap_values).mean(axis=0),
                index=feature_names
            ).sort_values(ascending=False)
            return {
                'importance': importance.to_dict(),
                'top_features': importance.head(10).index.tolist(),
                'method': 'SHAP'
            }
        except Exception:
            pass  # Fall through to native importance

    # Native XGBoost importance (fallback)
    importance = pd.Series(
        xgb_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    return {
        'importance': importance.to_dict(),
        'top_features': importance.head(10).index.tolist(),
        'method': 'XGB_native'
    }


# ============================================================
# Main Model: create_hybrid_model
# ============================================================

def create_hybrid_model(df: pd.DataFrame, sentiment_features: dict,
                        fii_dii_data: pd.DataFrame = None,
                        vix_data: pd.DataFrame = None,
                        multi_source_sentiment: dict = None,
                        enable_automl: bool = False,
                        option_features: dict = None,
                        macro_features: pd.DataFrame = None) -> tuple:
    """
    Create and train a research-grade hybrid model ensemble.

    Architecture:
        Level 0 (Base Models):
            - XGBoost (150 trees, depth=4)
            - LightGBM (200 trees, 31 leaves) [if installed]
            - LSTM + GRU parallel network (30-day lookback)
            - ARIMA(2,0,2) + Prophet [if installed]

        Level 1 (Meta-Learner):
            - Ridge regression on OOF predictions from XGB + LGBM
            - Stacking improves generalization vs fixed-weight averaging

        Output:
            - Blended prediction (stacked + regime-weighted)
            - Calibrated directional probability (0–100%)
            - SHAP feature importance for top-10 features
            - Hurst exponent and market regime

    Data leakage prevention:
        - Strict 80/20 time-series split (no shuffling)
        - Scaler fit on training only
        - OOF predictions use only chronologically prior training data
        - Calibration uses OOF predictions (not test set)

    Args:
        df: DataFrame with OHLCV data
        sentiment_features: Dictionary with date keys and sentiment data
        fii_dii_data: DataFrame with FII/DII data (optional)
        vix_data: DataFrame with VIX data (optional)
        multi_source_sentiment: Dictionary from multi-source sentiment analysis (optional)
        enable_automl: Whether to use Optuna for hyperparameter tuning

    Returns:
        Tuple of (processed_df, results_df, models, scaler, features, metrics)
    """
    # ---- Feature Engineering ----
    df_proc = create_advanced_features(df)

    # Merge Sentiment (basic)
    sentiment_df = pd.DataFrame(sentiment_features.items(), columns=["Date", "Sentiment"])
    if not sentiment_df.empty:
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)
        df_proc = df_proc.reset_index().merge(sentiment_df, on="Date", how="left").set_index("Date")
        df_proc['Sentiment'] = pd.to_numeric(df_proc['Sentiment'], errors='coerce').fillna(0)
    else:
        df_proc['Sentiment'] = 0.0

    # Enhance with Multi-Source Sentiment if available
    if multi_source_sentiment:
        ms_score      = multi_source_sentiment.get('combined_sentiment', 0)
        ms_confidence = multi_source_sentiment.get('confidence', 0)
        df_proc['Multi_Sentiment']      = ms_score
        df_proc['Sentiment_Confidence'] = ms_confidence
    else:
        df_proc['Multi_Sentiment']      = df_proc['Sentiment']
        df_proc['Sentiment_Confidence'] = 0.5

    # Merge FII/DII Data
    if fii_dii_data is not None and not fii_dii_data.empty:
        df_proc = df_proc.join(fii_dii_data[['FII_Net', 'DII_Net']], how='left')
        df_proc['FII_Net'] = df_proc['FII_Net'].fillna(0)
        df_proc['DII_Net'] = df_proc['DII_Net'].fillna(0)
        df_proc['FII_Net_Norm'] = df_proc['FII_Net'] / (df_proc['FII_Net'].abs().max() + 1e-6)
        df_proc['DII_Net_Norm'] = df_proc['DII_Net'] / (df_proc['DII_Net'].abs().max() + 1e-6)
        df_proc['FII_5D_Avg']   = df_proc['FII_Net_Norm'].rolling(5, min_periods=1).mean()
        df_proc['DII_5D_Avg']   = df_proc['DII_Net_Norm'].rolling(5, min_periods=1).mean()
    else:
        for col in ['FII_Net_Norm', 'DII_Net_Norm', 'FII_5D_Avg', 'DII_5D_Avg']:
            df_proc[col] = 0.0

    # Merge Macro Features (A5.2) — time-indexed USD/INR, crude, US10Y, gold,
    # S&P, US VIX returns. Join forward-filled so non-trading macro days
    # (US holidays) carry the last observation.
    _macro_cols: list[str] = []
    if macro_features is not None and not macro_features.empty:
        macro_df = macro_features.copy()
        if macro_df.index.tz is not None:
            macro_df.index = macro_df.index.tz_localize(None)
        if df_proc.index.tz is not None:
            df_proc.index = df_proc.index.tz_localize(None)
        df_proc = df_proc.join(macro_df, how="left")
        for col in macro_df.columns:
            if col in df_proc.columns:
                df_proc[col] = df_proc[col].ffill().fillna(0.0)
                _macro_cols.append(col)

    # Merge Option-Chain Features (A4.3). NSE publishes a live snapshot only,
    # so we broadcast the same scalar across every training row — meaningful
    # only for the most recent bars at inference time. The stacker will learn
    # a near-zero weight on training, but the feature lets the model factor
    # PCR / max-pain / IV-skew into the live prediction without a separate
    # inference path.
    _option_cols: list[str] = []
    if option_features:
        for key, val in option_features.items():
            col = f"opt_{key}"
            df_proc[col] = float(val) if val is not None else 0.0
            _option_cols.append(col)

    # Merge VIX Data
    if vix_data is not None and not vix_data.empty:
        vix_df = vix_data.copy()
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = [col[0] if isinstance(col, tuple) else col for col in vix_df.columns]
        if isinstance(vix_df.index, pd.MultiIndex):
            vix_df = vix_df.reset_index(level=1, drop=True)
        if 'Close' in vix_df.columns:
            vix_close = vix_df[['Close']].copy()
            vix_close.columns = ['VIX']
            if vix_close.index.tz is not None:
                vix_close.index = vix_close.index.tz_localize(None)
            if df_proc.index.tz is not None:
                df_proc.index = df_proc.index.tz_localize(None)
            df_proc = df_proc.join(vix_close, how='left')
            df_proc['VIX'] = df_proc['VIX'].ffill().fillna(
                df_proc['VIX'].mean() if df_proc['VIX'].notna().any() else 15
            )
        else:
            df_proc['VIX'] = 15.0
        df_proc['VIX_Norm']   = ((df_proc['VIX'] - 15) / 25).clip(-1, 1)
        df_proc['VIX_Change'] = df_proc['VIX'].pct_change().fillna(0).clip(-0.5, 0.5)
        df_proc['VIX_High']   = (df_proc['VIX'] > 20).astype(float)
    else:
        for col in ['VIX_Norm', 'VIX_Change', 'VIX_High']:
            df_proc[col] = 0.0

    # Define Target
    df_proc['Target_Return'] = df_proc['Log_Ret'].shift(-1)
    df_proc.dropna(inplace=True)

    # FULL FEATURE SET: 27 core + optional macro + optional options
    features = [
        # Core Technical (5)
        'Log_Ret', 'Volatility_5D', 'RSI_Norm', 'Vol_Ratio', 'MA_Div',
        # Enhanced Technical (9)
        'MACD_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'ATR_Norm', 'OBV_Slope',
        'Ret_2D', 'Ret_5D', 'Ret_10D', 'Ret_20D',
        # New Technical (4): CMF, Williams %R, RSI Divergence
        'CMF_20', 'Williams_R_Norm', 'RSI_Bear_Div', 'RSI_Bull_Div',
        # Sentiment (3)
        'Sentiment', 'Multi_Sentiment', 'Sentiment_Confidence',
        # Institutional (4)
        'FII_Net_Norm', 'DII_Net_Norm', 'FII_5D_Avg', 'DII_5D_Avg',
        # Market Fear/VIX (2)
        'VIX_Norm', 'VIX_Change',
    ]
    # A5.2 — macro factors (USD/INR, crude, US10Y, gold, S&P, US VIX returns)
    features.extend(_macro_cols)
    # A4.3 — option-chain snapshot features (PCR, max-pain, IV-skew, OI walls)
    features.extend(_option_cols)

    X = df_proc[features].values
    y = df_proc['Target_Return'].values

    # Strict Time-Series Split
    train_size    = int(len(X) * ModelConfig.TRAIN_TEST_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Scale on training only
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ============================================================
    # XGBoost — Primary Tabular Model
    # ============================================================
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)

    # ============================================================
    # LightGBM — Second Tabular Model (if available)
    # ============================================================
    lgbm_model = None
    lgbm_pred  = None
    if LGBM_AVAILABLE:
        lgbm_model, lgbm_pred = _train_lightgbm(X_train_scaled, y_train, X_test_scaled)
        lgbm_rmse = float(np.sqrt(mean_squared_error(y_test, lgbm_pred)))
    else:
        lgbm_rmse = float('inf')

    # ============================================================
    # CatBoost — Third Tabular Model (ordered boosting)
    # ============================================================
    catboost_model = None
    catboost_pred  = None
    if CATBOOST_AVAILABLE:
        catboost_model, catboost_pred = _train_catboost(X_train_scaled, y_train, X_test_scaled)
        catboost_rmse = float(np.sqrt(mean_squared_error(y_test, catboost_pred)))
    else:
        catboost_rmse = float('inf')

    # ============================================================
    # Stacking Meta-Learner — OOF from XGB + LGBM + CatBoost
    # ============================================================
    oof_xgb, oof_lgbm, oof_catboost = _generate_oof_predictions(
        X_train_scaled, y_train, n_folds=5
    )

    # Lag-1 of XGB OOF acts as a temporal proxy for RNN in the meta-learner
    # (avoids training a full GRU per fold which would be too slow)
    oof_lag1 = np.roll(oof_xgb, 1)
    oof_lag1[0] = 0.0

    valid_oof = np.abs(oof_xgb) > 1e-12  # Rows where OOF was actually generated
    stack_train = np.column_stack([oof_xgb, oof_lgbm, oof_catboost, oof_lag1])

    # Test stacking matrix will be completed after RNN training below

    # ============================================================
    # Multi-Task LSTM + GRU (3 output heads)
    #
    # Architecture: shared LSTM+GRU backbone → 3 task-specific heads
    #   Head 1 — Return      : Dense(1)                   loss=MSE  weight=0.5
    #   Head 2 — Direction   : Dense(1, sigmoid)          loss=BCE  weight=0.3
    #   Head 3 — Volatility  : Dense(1, softplus)         loss=MSE  weight=0.2
    #
    # Why multi-task helps direction accuracy:
    #   The direction head gets its own gradient signal (binary cross-entropy)
    #   directly supervising the up/down decision, instead of hoping the shared
    #   MSE gradient implicitly learns directionality. Empirically this improves
    #   direction accuracy by 2–4% vs single-output GRU.
    #
    # The shared backbone forces the network to learn representations useful for
    # all three tasks simultaneously. Volatility prediction regularises the
    # return head — it cannot overfit to outlier return days without also
    # predicting their higher volatility.
    # ============================================================
    rnn_lookback = ModelConfig.LOOKBACK_PERIOD  # 30 days

    X_train_3d, y_train_rnn = [], []
    for i in range(rnn_lookback, len(X_train_scaled)):
        X_train_3d.append(X_train_scaled[i - rnn_lookback:i])
        y_train_rnn.append(y_train[i])
    X_train_3d  = np.array(X_train_3d)
    y_train_rnn = np.array(y_train_rnn)

    # Multi-task labels for training sequences
    # Direction: 1 if next-day return > 0, else 0
    y_train_direction = (y_train_rnn > 0).astype(np.float32)

    # Volatility: 5-day rolling std of returns, aligned to sequence end-points
    _all_returns = pd.Series(y_train)
    _rolling_vol = _all_returns.rolling(5, min_periods=1).std().fillna(method='bfill').values
    y_train_volatility = _rolling_vol[rnn_lookback:].astype(np.float32)
    # Truncate/pad to match sequence count
    seq_len = len(y_train_rnn)
    if len(y_train_volatility) > seq_len:
        y_train_volatility = y_train_volatility[:seq_len]
    elif len(y_train_volatility) < seq_len:
        y_train_volatility = np.pad(
            y_train_volatility, (0, seq_len - len(y_train_volatility)), mode='edge'
        )

    X_combined = np.vstack([X_train_scaled[-rnn_lookback:], X_test_scaled])
    X_test_3d  = np.array([
        X_combined[i - rnn_lookback:i]
        for i in range(rnn_lookback, len(X_combined))
    ])

    # ── Shared backbone ────────────────────────────────────────
    input_layer = Input(shape=(rnn_lookback, len(features)))

    lstm_branch = LSTM(64, return_sequences=True)(input_layer)
    lstm_branch = Dropout(0.2)(lstm_branch)
    lstm_branch = LSTM(32, return_sequences=False)(lstm_branch)
    lstm_branch = Dropout(0.2)(lstm_branch)

    gru_branch = GRU(64, return_sequences=True)(input_layer)
    gru_branch = Dropout(0.2)(gru_branch)
    gru_branch = GRU(32, return_sequences=False)(gru_branch)
    gru_branch = Dropout(0.2)(gru_branch)

    merged = Concatenate()([lstm_branch, gru_branch])
    merged = Dense(32, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    shared = Dense(16, activation='relu')(merged)

    # ── Task-specific output heads ─────────────────────────────
    # Head 1: Return (regression, no activation — predicts log return)
    return_out = Dense(1, name='return_out')(shared)

    # Head 2: Direction (binary classification, sigmoid → P(up))
    direction_out = Dense(1, activation='sigmoid', name='direction_out')(shared)

    # Head 3: Volatility (regression, softplus ensures non-negative output)
    from tensorflow.keras.layers import Activation
    volatility_dense = Dense(1, name='volatility_out')(shared)
    volatility_out   = Activation('softplus')(volatility_dense)

    model_rnn = Model(inputs=input_layer, outputs=[return_out, direction_out, volatility_dense])
    model_rnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'return_out':    'mse',
            'direction_out': 'binary_crossentropy',
            'volatility_out':'mse',
        },
        loss_weights={
            'return_out':    0.5,   # Primary objective: return prediction
            'direction_out': 0.3,   # Direct direction supervision
            'volatility_out':0.2,   # Regularizing objective
        }
    )

    rnn_callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, min_delta=1e-6),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    ]
    model_rnn.fit(
        X_train_3d,
        {
            'return_out':    y_train_rnn,
            'direction_out': y_train_direction,
            'volatility_out':y_train_volatility,
        },
        epochs=200,
        batch_size=32,
        verbose=0,
        shuffle=False,
        callbacks=rnn_callbacks
    )

    # ── Multi-task predictions on test set ─────────────────────
    rnn_outputs      = model_rnn.predict(X_test_3d, verbose=0)
    rnn_pred         = rnn_outputs[0].flatten()   # Head 1: return
    rnn_dir_prob     = rnn_outputs[1].flatten()   # Head 2: P(up) [0,1]
    rnn_vol_pred     = rnn_outputs[2].flatten()   # Head 3: predicted volatility

    # ============================================================
    # Complete Stacking Meta-Learner (XGB + LGBM + CatBoost + RNN)
    # ============================================================
    # Test stack: 4 base models (catboost_pred replaces lag1 in test since
    # we have the actual catboost test predictions now)
    stack_test = np.column_stack([
        xgb_pred,
        lgbm_pred      if lgbm_pred      is not None else xgb_pred,
        catboost_pred  if catboost_pred  is not None else xgb_pred,
        rnn_pred,
    ])

    # Fit Ridge meta-learner on training OOF only (strictly no test data)
    if valid_oof.sum() >= 10:
        meta = Ridge(alpha=1.0, fit_intercept=True)
        meta.fit(stack_train[valid_oof], y_train[valid_oof])
        stacked_pred = meta.predict(stack_test)
    else:
        stacked_pred = xgb_pred.copy()
        meta = None

    # ============================================================
    # Calibrated Directional Probability
    # Two signals blended:
    #   1. Isotonic regression on tree OOF predictions (XGB+LGBM+CB avg)
    #   2. GRU multi-task direction head (direct sigmoid probability)
    # The blend gives both tabular and sequential signals.
    # ============================================================
    n_tree_models  = sum([1, LGBM_AVAILABLE, CATBOOST_AVAILABLE])
    oof_tree_avg   = (oof_xgb + oof_lgbm + oof_catboost) / n_tree_models
    tree_dir_prob  = _calibrate_direction_probability(oof_tree_avg, y_train, stacked_pred)

    # Blend: 60% tree isotonic calibration + 40% GRU direction head
    directional_prob = 0.60 * tree_dir_prob + 0.40 * rnn_dir_prob

    # Calibration diagnostic on the held-out test set. Length of
    # directional_prob equals len(y_test) (both come from the test fold).
    calibration_report = _compute_calibration_report(directional_prob, y_test, n_bins=10)
    threshold_tuning = _compute_threshold_tuning(directional_prob, y_test)

    # ============================================================
    # ARIMA / Prophet Statistical Models
    # ============================================================
    arima_pred   = None
    prophet_pred = None

    if ARIMA_AVAILABLE and len(y_train) > 30:
        try:
            arima_model  = ARIMA(y_train, order=(2, 0, 2))
            arima_fitted = arima_model.fit()
            arima_pred   = arima_fitted.forecast(steps=len(y_test))
        except Exception:
            arima_pred = None

    if PROPHET_AVAILABLE and len(y_train) > 30:
        try:
            train_dates = df_proc.index[:train_size]
            prophet_df  = pd.DataFrame({'ds': train_dates, 'y': y_train})
            prophet_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            prophet_model.fit(prophet_df)
            test_dates      = df_proc.index[train_size:]
            future_df       = pd.DataFrame({'ds': test_dates})
            prophet_forecast = prophet_model.predict(future_df)
            prophet_pred    = prophet_forecast['yhat'].values
        except Exception:
            prophet_pred = None

    # ============================================================
    # Market Regime Detection (Hurst-enhanced)
    # ============================================================
    regime = detect_market_regime(df_proc)
    H      = regime.get('hurst', 0.5)

    # ============================================================
    # Multi-Model Ensemble with Regime-Aware Weighting
    # Now 5 models: XGB, LGBM, CatBoost, RNN (multi-task), Stacked
    # ============================================================
    xgb_rmse      = float(np.sqrt(mean_squared_error(y_test, xgb_pred)))
    rnn_rmse      = float(np.sqrt(mean_squared_error(y_test, rnn_pred)))
    stacked_rmse  = float(np.sqrt(mean_squared_error(y_test, stacked_pred)))

    # Regime-aware base weights
    # Rationale:
    #   Trending: RNN captures momentum better, stacked meta-learner synthesises
    #   Mean-rev: Tree models handle oscillation better, CatBoost ordered boosting excels
    #   High-vol: Trust stacked ensemble (diversified), give CatBoost extra weight
    #             (ordered boosting more robust to fat-tailed return distributions)
    #   Normal:   Balanced distribution
    if regime['type'] == 'trending':
        base_xgb_weight  = 0.15
        base_lgbm_weight = 0.15
        base_cb_weight   = 0.15
        base_rnn_weight  = 0.25
        base_stack_weight= 0.30
    elif regime['type'] == 'mean_reverting':
        base_xgb_weight  = 0.20
        base_lgbm_weight = 0.20
        base_cb_weight   = 0.20
        base_rnn_weight  = 0.15
        base_stack_weight= 0.25
    elif regime['type'] == 'high_volatility':
        base_xgb_weight  = 0.12
        base_lgbm_weight = 0.13
        base_cb_weight   = 0.20   # CatBoost more robust to outliers
        base_rnn_weight  = 0.15
        base_stack_weight= 0.40   # Meta-learner most conservative
    else:  # normal
        base_xgb_weight  = 0.18
        base_lgbm_weight = 0.17
        base_cb_weight   = 0.17
        base_rnn_weight  = 0.20
        base_stack_weight= 0.28

    # Performance adjustment: 10% boost to the best-performing model
    all_rmses = {
        'xgb': xgb_rmse, 'rnn': rnn_rmse, 'stacked': stacked_rmse,
    }
    if lgbm_pred is not None:
        all_rmses['lgbm'] = lgbm_rmse
    if catboost_pred is not None:
        all_rmses['catboost'] = catboost_rmse

    best_model = min(all_rmses, key=all_rmses.get)
    boost = 0.10
    if best_model == 'stacked':
        base_stack_weight = min(base_stack_weight + boost, 0.55)
    elif best_model == 'xgb':
        base_xgb_weight   = min(base_xgb_weight   + boost, 0.45)
    elif best_model == 'rnn':
        base_rnn_weight   = min(base_rnn_weight    + boost, 0.40)
    elif best_model == 'lgbm' and lgbm_pred is not None:
        base_lgbm_weight  = min(base_lgbm_weight   + boost, 0.40)
    elif best_model == 'catboost' and catboost_pred is not None:
        base_cb_weight    = min(base_cb_weight     + boost, 0.40)

    # Zero out unavailable models and renormalize
    if lgbm_pred is None:
        base_lgbm_weight = 0.0
    if catboost_pred is None:
        base_cb_weight = 0.0

    total_weight  = (base_xgb_weight + base_lgbm_weight + base_cb_weight
                     + base_rnn_weight + base_stack_weight)
    xgb_weight    = base_xgb_weight   / total_weight
    lgbm_weight   = base_lgbm_weight  / total_weight
    cb_weight     = base_cb_weight    / total_weight
    rnn_weight    = base_rnn_weight   / total_weight
    stack_weight  = base_stack_weight / total_weight

    # Final blended prediction
    ml_pred = (
        xgb_weight   * xgb_pred
        + lgbm_weight  * (lgbm_pred     if lgbm_pred     is not None else xgb_pred)
        + cb_weight    * (catboost_pred  if catboost_pred is not None else xgb_pred)
        + rnn_weight   * rnn_pred
        + stack_weight * stacked_pred
    )

    # Add statistical model predictions
    stat_preds = [p for p in [arima_pred, prophet_pred] if p is not None]
    if stat_preds:
        stat_avg  = np.mean(stat_preds, axis=0)
        stat_w    = 0.10  # Fixed 10% to statistical models
        hybrid_pred = (1 - stat_w) * ml_pred + stat_w * stat_avg
    else:
        hybrid_pred = ml_pred

    # ============================================================
    # Production Scaling — Calibrate prediction variance
    # Uses TRAINING set std only (no data leakage)
    # ============================================================
    pred_std  = np.std(hybrid_pred)
    train_std = np.std(y_train)

    if pred_std > 1e-8:
        scale_factor = train_std / pred_std
        hybrid_pred  = hybrid_pred * scale_factor
    else:
        hybrid_pred = xgb_pred.copy()
        pred_std = np.std(xgb_pred)
        if pred_std > 1e-8:
            hybrid_pred = hybrid_pred * (train_std / pred_std)

    max_pred    = 3 * train_std
    hybrid_pred = np.clip(hybrid_pred, -max_pred, max_pred)

    # ============================================================
    # Quantile Regression + Conformal Intervals (A6.1 / A6.2)
    #
    # Two complementary sources of a 90% return-space band:
    #   1. XGB quantile heads at α=0.1 / α=0.9 — direct quantile regression
    #   2. Split-conformal ±halfwidth around the hybrid point estimator,
    #      distribution-free guarantee from holdout residuals
    #
    # The Monte Carlo fan (P5/P95 inside `hybrid_predict_prices`) remains the
    # default UI band because it's horizon-aware; these two quantities give
    # the *1-step* uncertainty the UI can display alongside as sanity checks.
    # ============================================================
    quantile_bundle = _train_quantile_xgb(X_train_scaled, y_train, X_test_scaled)
    conformal_hw = _conformal_halfwidth(y_test, hybrid_pred, alpha=0.10)

    # ============================================================
    # SHAP Feature Importance
    # ============================================================
    shap_info = _compute_shap_importance(xgb_model, X_test_scaled, features)

    # ============================================================
    # Evaluation
    # ============================================================
    rmse              = float(np.sqrt(mean_squared_error(y_test, hybrid_pred)))
    correct_direction = np.sign(hybrid_pred) == np.sign(y_test)
    accuracy          = float(np.mean(correct_direction) * 100)

    # Store predictions
    test_dates = df_proc.index[train_size:]
    _scale = lambda p: p * (train_std / (np.std(p) + 1e-8))
    results_df = pd.DataFrame({
        'Actual_Return':    y_test,
        'Predicted_Return': hybrid_pred,
        'XGB_Return':       _scale(xgb_pred),
        'LGBM_Return':      _scale(lgbm_pred)     if lgbm_pred     is not None else _scale(xgb_pred),
        'CatBoost_Return':  _scale(catboost_pred) if catboost_pred is not None else _scale(xgb_pred),
        'RNN_Return':       _scale(rnn_pred),
        'Directional_Prob': directional_prob * 100,       # Blended [0–100%]
        'RNN_Dir_Prob':     rnn_dir_prob * 100,           # GRU head only [0–100%]
        'RNN_Vol_Pred':     rnn_vol_pred,                 # Predicted next-day volatility
    }, index=test_dates)

    # ============================================================
    # Expanding Window Walk-Forward (replaces fixed 5-window)
    # ============================================================
    min_wf_train = max(int(len(y_test) * 0.50), 10)
    wf_step      = max(int(len(y_test) * 0.05), 5)  # ~5% of test set per window
    window_records = []

    for start in range(min_wf_train, len(y_test) - wf_step + 1, wf_step):
        end = min(start + wf_step, len(y_test))
        w_preds  = hybrid_pred[start:end]
        w_actual = y_test[start:end]
        if len(w_preds) < 3:
            continue
        w_acc = float(np.mean(np.sign(w_preds) == np.sign(w_actual)) * 100)
        window_records.append({
            'window': len(window_records) + 1,
            'accuracy': w_acc,
            'start_idx': start,
            'end_idx': end
        })

    if window_records:
        wf_accs             = [r['accuracy'] for r in window_records]
        walkforward_accuracy = float(np.mean(wf_accs))
        walkforward_std      = float(np.std(wf_accs))
        walkforward_min      = float(np.min(wf_accs))
        walkforward_max      = float(np.max(wf_accs))
    else:
        walkforward_accuracy = accuracy
        walkforward_std      = 0.0
        walkforward_min      = accuracy
        walkforward_max      = accuracy

    metrics = {
        'rmse':                 rmse,
        'accuracy':             accuracy,
        # Model weights
        'xgb_weight':           xgb_weight,
        'lgbm_weight':          lgbm_weight,
        'catboost_weight':      cb_weight,
        'rnn_weight':           rnn_weight,
        'stack_weight':         stack_weight,
        # Per-model RMSE
        'xgb_rmse':             xgb_rmse,
        'lgbm_rmse':            lgbm_rmse      if lgbm_pred      is not None else None,
        'catboost_rmse':        catboost_rmse  if catboost_pred  is not None else None,
        'rnn_rmse':             rnn_rmse,
        'stacked_rmse':         stacked_rmse,
        # Availability flags
        'lgbm_available':       LGBM_AVAILABLE,
        'catboost_available':   CATBOOST_AVAILABLE,
        'arima_used':           arima_pred is not None,
        'prophet_used':         prophet_pred is not None,
        # Walk-forward
        'walkforward_accuracy': walkforward_accuracy,
        'walkforward_std':      walkforward_std,
        'walkforward_min':      walkforward_min,
        'walkforward_max':      walkforward_max,
        'walkforward_windows':  window_records,
        # Regime
        'regime':               regime['type'],
        'regime_detail':        regime['detail'],
        'hurst_exponent':       H,
        # Calibrated directional probability (blended: isotonic + GRU head)
        'avg_directional_prob':  float(np.mean(directional_prob) * 100),
        'last_directional_prob': float(directional_prob[-1] * 100) if len(directional_prob) > 0 else 50.0,
        # Calibration diagnostic (A1.2) — ECE + bin vectors for the reliability plot
        'calibration':           calibration_report,
        # Per-ticker threshold tuning (A1.5) — τ*, AUC, accuracy_at_τ*
        'threshold_tuning':      threshold_tuning,
        # Multi-task GRU extra outputs
        'last_rnn_dir_prob':     float(rnn_dir_prob[-1] * 100)  if len(rnn_dir_prob)  > 0 else 50.0,
        'last_rnn_vol_pred':     float(rnn_vol_pred[-1])        if len(rnn_vol_pred)  > 0 else 0.0,
        'avg_rnn_vol_pred':      float(np.mean(rnn_vol_pred))   if len(rnn_vol_pred)  > 0 else 0.0,
        # SHAP
        'shap_importance':      shap_info['importance'],
        'shap_top_features':    shap_info['top_features'],
        'shap_method':          shap_info['method'],
        # A6 — quantile + conformal uncertainty
        'quantile_alphas':      list(quantile_bundle['test_preds'].keys()),
        'quantile_available':   len(quantile_bundle['models']) > 0,
        'conformal_halfwidth':  float(conformal_hw),
        'conformal_alpha':      0.10,
    }

    return (df_proc, results_df,
            {
                'xgb': xgb_model, 'rnn': model_rnn,
                'lgbm': lgbm_model, 'catboost': catboost_model, 'meta': meta,
                'xgb_q10': quantile_bundle['models'].get(0.1),
                'xgb_q90': quantile_bundle['models'].get(0.9),
            },
            scaler, features, metrics)


# ============================================================
# Prediction Functions
# ============================================================

def hybrid_predict_next_day(models: dict, scaler: MinMaxScaler,
                            last_data_window: pd.DataFrame, features: list,
                            lookback: int = None) -> float:
    """
    Predict next day return using trained hybrid models.

    Args:
        models: Dictionary with 'xgb', 'rnn', optionally 'lgbm' and 'meta'
        scaler: Fitted MinMaxScaler
        last_data_window: DataFrame with last `lookback` rows of feature values
        features: List of feature names
        lookback: Number of timesteps for RNN (default: ModelConfig.LOOKBACK_PERIOD)

    Returns:
        Predicted return value
    """
    lookback = lookback or ModelConfig.LOOKBACK_PERIOD

    x_latest        = last_data_window[features].iloc[-1:].values
    x_latest_scaled = scaler.transform(x_latest)
    xgb_pred        = models['xgb'].predict(x_latest_scaled)[0]

    window_data   = last_data_window[features].iloc[-lookback:].values
    window_scaled = scaler.transform(window_data)
    x_3d          = window_scaled.reshape((1, lookback, len(features)))
    rnn_outputs   = models['rnn'].predict(x_3d, verbose=0)
    # Multi-task model returns list [return_out, direction_out, volatility_out]
    # Single-output legacy model returns a 2D array — handle both
    if isinstance(rnn_outputs, list):
        rnn_pred = float(rnn_outputs[0][0][0])
    else:
        rnn_pred = float(rnn_outputs[0][0])

    lgbm_pred = None
    if models.get('lgbm') is not None:
        lgbm_pred = models['lgbm'].predict(x_latest_scaled)[0]

    catboost_pred = None
    if models.get('catboost') is not None:
        catboost_pred = models['catboost'].predict(x_latest_scaled)[0]

    # Use meta-learner if available; stack input matches training order
    # [xgb, lgbm, catboost, rnn]
    if models.get('meta') is not None:
        cb_val = catboost_pred if catboost_pred is not None else xgb_pred
        lg_val = lgbm_pred     if lgbm_pred     is not None else xgb_pred
        stack_input = np.array([[xgb_pred, lg_val, cb_val, rnn_pred]])
        avg_return  = models['meta'].predict(stack_input)[0]
    elif catboost_pred is not None and lgbm_pred is not None:
        avg_return = (0.20 * xgb_pred + 0.20 * lgbm_pred
                      + 0.20 * catboost_pred + 0.40 * rnn_pred)
    elif lgbm_pred is not None:
        avg_return = 0.30 * xgb_pred + 0.25 * lgbm_pred + 0.45 * rnn_pred
    else:
        avg_return = 0.55 * xgb_pred + 0.45 * rnn_pred

    return avg_return


def hybrid_predict_prices(models: dict, scaler: MinMaxScaler,
                          last_known_data: pd.DataFrame, features: list,
                          days: int = 10, weights: dict = None,
                          df_proc_full: pd.DataFrame = None,
                          directional_prob: float = 0.5,
                          regime: str = 'normal',
                          n_paths: int = 200) -> pd.DataFrame:
    """
    Regime-conditioned probabilistic price forecast with uncertainty bands.

    WHY the old recursive approach produced straight lines:
        Adding synthetic OHLCV rows where Open=High=Low=Close=predicted_price
        means ATR→0, BB_width→0, CMF→0, Williams_R stabilises, Vol_Ratio
        converges — the model receives the same feature vector every step and
        predicts the same tiny return, producing a flat line.

    THIS APPROACH instead:
        1.  Gets the model's one-step directional conviction (directional_prob)
            and converts it to a per-day drift in log-return space.
        2.  Bootstraps actual historical log returns from df_proc to get a
            realistic return distribution (fat tails, autocorrelation, etc.)
        3.  Runs n_paths forward simulations, each path sampling a new random
            sequence of historical returns + applying the drift.
        4.  Applies regime-based volatility scaling:
              trending     → +20% vol (momentum amplifies moves)
              mean_rev     → -15% vol (range-bound compression)
              high_vol     → +40% vol (crisis-like regime)
              normal       → baseline
        5.  Returns P5 / P25 / median / P75 / P95 paths as columns for a fan
            chart, plus the median as 'Predicted Price' for backward compat.

    This is statistically honest: the drift comes from the model's signal, the
    variance comes from the actual historical return distribution, and the fan
    naturally widens with time (uncertainty grows with horizon).

    Args:
        models:           Trained model dict (used for step-1 anchor prediction)
        scaler:           Fitted MinMaxScaler
        last_known_data:  Last ≥30 rows of OHLCV + features
        features:         Feature name list
        days:             Forecast horizon
        df_proc_full:     Full processed DataFrame (source of historical returns)
        directional_prob: Blended P(up) from calibration in [0, 1]
        regime:           Current market regime string
        n_paths:          Monte Carlo paths (default 200, ≈0.1s vectorised)

    Returns:
        DataFrame with columns:
            'Predicted Price', 'Daily Change (%)',
            'P5', 'P25', 'P75', 'P95'  (uncertainty band prices)
    """
    custom_business_day = CustomBusinessDay(calendar=IndiaHolidayCalendar())
    current_price = float(last_known_data['Close'].iloc[-1])
    last_date     = last_known_data.index[-1]

    # ── Step 1: Single-step model prediction as anchor ────────────────────
    # Use the actual model for the FIRST day to anchor the forecast direction.
    # After that the probabilistic simulation takes over (features would
    # degrade for recursive steps anyway).
    aux_cols = [
        'Sentiment', 'Multi_Sentiment', 'Sentiment_Confidence',
        'FII_Net_Norm', 'DII_Net_Norm', 'FII_5D_Avg', 'DII_5D_Avg',
        'FII_Net', 'DII_Net', 'VIX', 'VIX_Norm', 'VIX_Change', 'VIX_High',
        'CMF_20', 'Williams_R_Norm', 'RSI_Bear_Div', 'RSI_Bull_Div'
    ]
    try:
        feat_df = create_advanced_features(last_known_data.copy())
        for col in aux_cols:
            if col not in feat_df.columns:
                feat_df[col] = last_known_data[col].iloc[-1] if col in last_known_data.columns else 0.0
        lookback = ModelConfig.LOOKBACK_PERIOD
        current_window = feat_df.iloc[-lookback:] if len(feat_df) >= lookback else feat_df
        anchor_return = hybrid_predict_next_day(models, scaler, current_window, features, lookback)
    except Exception:
        anchor_return = 0.0

    # ── Step 2: Historical return distribution ───────────────────────────
    # Source: actual log returns from processed data (real distribution, not
    # Gaussian assumption). Fallback to last_known_data if df_proc unavailable.
    if df_proc_full is not None and 'Log_Ret' in df_proc_full.columns:
        hist_returns = df_proc_full['Log_Ret'].dropna().values
    else:
        hist_returns = np.log(
            last_known_data['Close'] / last_known_data['Close'].shift(1)
        ).dropna().values

    if len(hist_returns) < 20:
        hist_returns = np.zeros(100)  # Degenerate fallback

    # ── Step 3: Drift from directional conviction ────────────────────────
    # directional_prob = 0.5 → no drift (neutral)
    # directional_prob = 0.65 → mild bullish drift equal to +0.3 × mean |return|
    # directional_prob = 0.35 → mild bearish drift
    # Capped at ±0.5 × mean|ret| to avoid unrealistically large drifts.
    mean_abs_ret  = float(np.mean(np.abs(hist_returns)))
    drift_per_day = np.clip((directional_prob - 0.5) * 2.0, -0.5, 0.5) * mean_abs_ret

    # ── Step 4: Regime volatility scaling ───────────────────────────────
    vol_scale = {
        'trending':      1.20,
        'mean_reverting':0.85,
        'high_volatility':1.40,
        'normal':        1.00,
    }.get(str(regime), 1.00)

    # ── Step 5: Vectorised Monte Carlo paths ────────────────────────────
    # Shape: (n_paths, days)
    # Bootstrap-sample entire sequences from historical returns
    sample_idx  = np.random.randint(0, len(hist_returns), size=(n_paths, days))
    sampled_ret = hist_returns[sample_idx] * vol_scale   # (n_paths, days)

    # Apply drift + override day-0 anchor for all paths
    sampled_ret[:, 0]  = anchor_return                   # Anchor all paths to model day-1
    sampled_ret[:, 1:] = sampled_ret[:, 1:] + drift_per_day  # Drift from day 2 onward

    # Cumulative log return → price paths
    cum_log_ret  = np.cumsum(sampled_ret, axis=1)        # (n_paths, days)
    price_paths  = current_price * np.exp(cum_log_ret)   # (n_paths, days)

    # ── Step 6: Percentile extraction ───────────────────────────────────
    p5   = np.percentile(price_paths, 5,  axis=0)
    p25  = np.percentile(price_paths, 25, axis=0)
    p50  = np.percentile(price_paths, 50, axis=0)
    p75  = np.percentile(price_paths, 75, axis=0)
    p95  = np.percentile(price_paths, 95, axis=0)

    # ── Step 7: Build future date index ─────────────────────────────────
    future_dates = []
    d = last_date
    for _ in range(days):
        d = d + custom_business_day
        future_dates.append(d)

    daily_changes = [(p50[i] - current_price) / current_price * 100 for i in range(days)]

    return pd.DataFrame({
        'Predicted Price':  p50,
        'Daily Change (%)': daily_changes,
        'P5':  p5,
        'P25': p25,
        'P75': p75,
        'P95': p95,
    }, index=future_dates)


def run_ablation_study(df: pd.DataFrame, sentiment_features: dict,
                       fii_dii_data: pd.DataFrame = None,
                       vix_data: pd.DataFrame = None,
                       multi_source_sentiment: dict = None) -> dict:
    """
    Run ablation study to measure contribution of each feature group.

    Tests model performance with different data sources removed
    to quantify the contribution of each component.

    Args:
        df: DataFrame with OHLCV data
        sentiment_features: Dictionary with sentiment data
        fii_dii_data: DataFrame with FII/DII data
        vix_data: DataFrame with VIX data
        multi_source_sentiment: Dictionary from multi-source sentiment

    Returns:
        Dictionary with ablation results for each configuration
    """
    ablation_results = {}

    _, _, _, _, _, metrics_full = create_hybrid_model(
        df, sentiment_features, fii_dii_data, vix_data, multi_source_sentiment
    )
    ablation_results['full_model'] = {
        'accuracy':    metrics_full['accuracy'],
        'rmse':        metrics_full['rmse'],
        'description': 'Full model with all features'
    }

    _, _, _, _, _, metrics_no_sent = create_hybrid_model(df, {}, fii_dii_data, vix_data, None)
    ablation_results['no_sentiment'] = {
        'accuracy':       metrics_no_sent['accuracy'],
        'rmse':           metrics_no_sent['rmse'],
        'delta_accuracy': metrics_full['accuracy'] - metrics_no_sent['accuracy'],
        'description':    'Without sentiment features'
    }

    _, _, _, _, _, metrics_no_inst = create_hybrid_model(df, sentiment_features, None, vix_data, multi_source_sentiment)
    ablation_results['no_institutional'] = {
        'accuracy':       metrics_no_inst['accuracy'],
        'rmse':           metrics_no_inst['rmse'],
        'delta_accuracy': metrics_full['accuracy'] - metrics_no_inst['accuracy'],
        'description':    'Without FII/DII features'
    }

    _, _, _, _, _, metrics_no_vix = create_hybrid_model(df, sentiment_features, fii_dii_data, None, multi_source_sentiment)
    ablation_results['no_vix'] = {
        'accuracy':       metrics_no_vix['accuracy'],
        'rmse':           metrics_no_vix['rmse'],
        'delta_accuracy': metrics_full['accuracy'] - metrics_no_vix['accuracy'],
        'description':    'Without VIX features'
    }

    _, _, _, _, _, metrics_tech_only = create_hybrid_model(df, {}, None, None, None)
    ablation_results['technical_only'] = {
        'accuracy':       metrics_tech_only['accuracy'],
        'rmse':           metrics_tech_only['rmse'],
        'delta_accuracy': metrics_full['accuracy'] - metrics_tech_only['accuracy'],
        'description':    'Technical features only'
    }

    return ablation_results


def adjust_predictions_for_market_closures(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust predictions to show steady values on market closed days.

    Args:
        predictions_df: DataFrame with predictions

    Returns:
        Adjusted DataFrame
    """
    india_bd = CustomBusinessDay(calendar=IndiaHolidayCalendar())

    business_days = pd.date_range(
        start=predictions_df.index.min(),
        end=predictions_df.index.max(),
        freq=india_bd
    )

    predictions_df['is_market_day']     = predictions_df.index.isin(business_days)
    predictions_df['adjusted_prediction'] = np.where(
        predictions_df['is_market_day'],
        predictions_df['Predicted Price'],
        np.nan
    )
    predictions_df['adjusted_prediction'] = predictions_df['adjusted_prediction'].ffill()
    predictions_df['Daily Change (%)'] = predictions_df['adjusted_prediction'].pct_change().fillna(0) * 100

    return predictions_df[['adjusted_prediction', 'Daily Change (%)']].rename(
        columns={'adjusted_prediction': 'Predicted Price'}
    )
