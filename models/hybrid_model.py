"""
Hybrid model ensemble: XGBoost + LightGBM + multi-task LSTM/GRU + stacking
meta-learner, with calibrated probabilities, SHAP importance, and Hurst
exponent regime detection.
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

# LightGBM — graceful fallback to XGBoost-only if not installed
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# CatBoost (optional — ordered boosting for small financial datasets)
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# SHAP for feature importance (optional — falls back to XGB native)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from config.settings import ModelConfig
from data.vix_data import IndiaHolidayCalendar



def calculate_hurst_exponent(prices: np.ndarray, lags: range = None) -> float:
    """
    Hurst exponent via R/S analysis.
    H > 0.55 → trending, H < 0.45 → mean-reverting, H ≈ 0.5 → random walk.
    """
    if lags is None:
        lags = range(2, 21)

    if len(prices) < 40:
        return 0.5

    log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))

    rs_pairs = []
    for lag in lags:
        rs_vals = []
        n_windows = len(log_returns) // lag
        if n_windows < 2:
            continue
        for w in range(n_windows):
            seg = log_returns[w * lag:(w + 1) * lag]
            mean_adj = seg - np.mean(seg)
            cumdev = np.cumsum(mean_adj)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(seg, ddof=1)
            if s > 1e-10:
                rs_vals.append(r / s)
        if rs_vals:
            rs_pairs.append((lag, np.mean(rs_vals)))

    if len(rs_pairs) < 4:
        return 0.5

    lags_arr = np.array([p[0] for p in rs_pairs], dtype=float)
    rs_arr   = np.array([p[1] for p in rs_pairs], dtype=float)

    valid = rs_arr > 0
    if valid.sum() < 4:
        return 0.5

    slope = np.polyfit(np.log(lags_arr[valid]), np.log(rs_arr[valid]), 1)[0]
    return float(np.clip(slope, 0.1, 0.9))



def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build stationary ML features from OHLCV data."""
    df = df.copy()

    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility_5D'] = df['Log_Ret'].rolling(window=5).std()

    if 'RSI' not in df.columns:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    df['RSI_Norm'] = df['RSI'] / 100.0
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['MA_Div'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close']

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Norm'] = macd_line / df['Close']
    df['MACD_Hist_Norm'] = (macd_line - macd_signal) / df['Close']

    bb_ma  = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    df['BB_PctB'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)

    high_low   = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close  = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_Norm'] = true_range.rolling(14, min_periods=1).mean() / df['Close']

    obv    = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    obv_ma = obv.rolling(5, min_periods=1).mean()
    df['OBV_Slope'] = obv_ma.pct_change(5).clip(-1, 1).fillna(0)

    df['Ret_2D']  = np.log(df['Close'] / df['Close'].shift(2))
    df['Ret_5D']  = np.log(df['Close'] / df['Close'].shift(5))
    df['Ret_10D'] = np.log(df['Close'] / df['Close'].shift(10))
    df['Ret_20D'] = np.log(df['Close'] / df['Close'].shift(20))

    hl_range = (df['High'] - df['Low']).replace(0, np.nan)
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_range
    mfvol = mfm.fillna(0) * df['Volume']
    vol_sum = df['Volume'].rolling(20).sum()
    df['CMF_20'] = mfvol.rolling(20).sum() / (vol_sum + 1e-8)
    df['CMF_20'] = df['CMF_20'].fillna(0).clip(-1, 1)

    high14 = df['High'].rolling(14).max()
    low14  = df['Low'].rolling(14).min()
    df['Williams_R_Norm'] = (high14 - df['Close']) / (high14 - low14 + 1e-8) * -1.0

    # Divergence flags: no look-ahead — shift(1) applied before comparison
    rsi_10h = df['RSI'].rolling(10).max().shift(1)
    px_10h  = df['Close'].rolling(10).max().shift(1)
    rsi_10l = df['RSI'].rolling(10).min().shift(1)
    px_10l  = df['Close'].rolling(10).min().shift(1)
    df['RSI_Bear_Div'] = ((df['Close'] >= px_10h) & (df['RSI'] < rsi_10h)).astype(float).fillna(0)
    df['RSI_Bull_Div'] = ((df['Close'] <= px_10l) & (df['RSI'] > rsi_10l)).astype(float).fillna(0)

    # ── P2-11: Trend-strength features (critical for Hurst=0.73 trending market) ──

    # ADX via Wilder's smoothing — reuses true_range already computed above
    _adx_alpha = 1.0 / 14
    _up_move   = df['High'].diff()
    _down_move = -df['Low'].diff()
    _plus_dm   = _up_move.where((_up_move > _down_move) & (_up_move > 0), 0.0)
    _minus_dm  = _down_move.where((_down_move > _up_move) & (_down_move > 0), 0.0)
    _tr_s      = true_range.ewm(alpha=_adx_alpha, adjust=False).mean()
    _plus_di   = 100 * _plus_dm.ewm(alpha=_adx_alpha, adjust=False).mean() / (_tr_s + 1e-8)
    _minus_di  = 100 * _minus_dm.ewm(alpha=_adx_alpha, adjust=False).mean() / (_tr_s + 1e-8)
    _dx        = 100 * (_plus_di - _minus_di).abs() / (_plus_di + _minus_di + 1e-8)
    _adx       = _dx.ewm(alpha=_adx_alpha, adjust=False).mean()
    df['ADX_Norm']    = (_adx / 100.0).clip(0, 1)
    df['ADX_Trend']   = (_adx > 25).astype(float)  # 1 = strong trend gate
    df['DI_Diff_Norm'] = ((_plus_di - _minus_di) / 100.0).clip(-1, 1)  # directional bias

    # Aroon (25-period): measures how recently highest high / lowest low occurred
    _aroon_n = 26
    df['Aroon_Up']      = df['High'].rolling(_aroon_n).apply(lambda x: np.argmax(x) / 25 * 100, raw=True)
    df['Aroon_Down']    = df['Low'].rolling(_aroon_n).apply(lambda x: np.argmin(x) / 25 * 100, raw=True)
    df['Aroon_Osc_Norm'] = ((df['Aroon_Up'] - df['Aroon_Down']) / 100.0).clip(-1, 1)

    # EMA slopes (trend direction + acceleration) normalized by price
    _ema9  = df['Close'].ewm(span=9,  adjust=False).mean()
    _ema21 = df['Close'].ewm(span=21, adjust=False).mean()
    _ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    _px    = df['Close'] + 1e-8
    df['EMA9_Slope']  = ((_ema9  - _ema9.shift(3))  / _px).clip(-0.1, 0.1)
    df['EMA21_Slope'] = ((_ema21 - _ema21.shift(5))  / _px).clip(-0.1, 0.1)
    df['EMA50_Slope'] = ((_ema50 - _ema50.shift(10)) / _px).clip(-0.1, 0.1)

    df.dropna(inplace=True)

    return df



def detect_market_regime(df: pd.DataFrame, vol_window: int = 20, trend_window: int = 20) -> dict:
    """Classify current market regime using volatility, trend slope, and Hurst exponent."""
    if len(df) < max(vol_window, trend_window, 60):
        return {'type': 'normal', 'detail': 'Insufficient data for regime detection', 'hurst': 0.5}

    recent_returns = df['Log_Ret'].iloc[-trend_window:]

    close_arr = df['Close'].values[-120:] if len(df) >= 120 else df['Close'].values
    H = calculate_hurst_exponent(close_arr)

    if 'Volatility_5D' in df.columns:
        recent_vol    = df['Volatility_5D'].iloc[-1]
        vol_history   = df['Volatility_5D'].iloc[-60:]
        vol_percentile = (vol_history < recent_vol).mean() * 100
    else:
        vol_percentile = 50.0

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

    # P2-7: Use ADX as a hard gate for the trending regime when available.
    # ADX > 25 confirms trend strength; ADX < 20 overrides a slope signal.
    adx_val = None
    if 'ADX_Norm' in df.columns and df['ADX_Norm'].notna().any():
        adx_val = float(df['ADX_Norm'].iloc[-1]) * 100  # back to 0-100 scale

    def _adx_confirms_trend() -> bool:
        return adx_val is None or adx_val >= 20

    def _adx_negates_trend() -> bool:
        return adx_val is not None and adx_val < 18

    if vol_percentile > 85:
        direction = 'down-trending' if slope < 0 else 'up-trending'
        return {
            'type': 'high_volatility',
            'detail': f'Elevated volatility ({vol_percentile:.0f}th percentile), {direction}',
            'hurst': round(H, 3),
            'adx': round(adx_val, 1) if adx_val is not None else None,
        }
    elif abs_slope > 0.002 and r_squared > 0.3:
        if H >= 0.45 and _adx_confirms_trend():
            direction = 'Uptrend' if slope > 0 else 'Downtrend'
            strength  = 'strong' if (r_squared > 0.6 and (adx_val or 0) > 25) else 'moderate'
            adx_str   = f', ADX={adx_val:.0f}' if adx_val is not None else ''
            return {
                'type': 'trending',
                'detail': f'{direction} ({strength}, R²={r_squared:.2f}, H={H:.2f}{adx_str})',
                'hurst': round(H, 3),
                'adx': round(adx_val, 1) if adx_val is not None else None,
            }
        else:
            reason = (f'ADX={adx_val:.0f}<18 negates slope'
                      if _adx_negates_trend() else f'H={H:.2f}<0.45 overrides slope')
            return {
                'type': 'mean_reverting',
                'detail': f'Slope-trend overridden by {reason}',
                'hurst': round(H, 3),
                'adx': round(adx_val, 1) if adx_val is not None else None,
            }
    elif vol_percentile < 30 and abs_slope < 0.001:
        return {
            'type': 'mean_reverting',
            'detail': f'Range-bound (vol {vol_percentile:.0f}th pctl, H={H:.2f})',
            'hurst': round(H, 3),
            'adx': round(adx_val, 1) if adx_val is not None else None,
        }
    else:
        return {
            'type': 'normal',
            'detail': f'Normal conditions (vol {vol_percentile:.0f}th pctl, H={H:.2f})',
            'hurst': round(H, 3),
            'adx': round(adx_val, 1) if adx_val is not None else None,
        }



def _train_quantile_xgb(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray,
                        alphas: tuple = (0.1, 0.9)) -> dict:
    """
    Train XGB quantile regressors (A6.1) — one model per alpha.
    Not fed into the meta stacker — that would overfit on the same OOF residuals twice.
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
            continue  # quantile XGB requires xgboost >= 2.0
    return {"models": models, "test_preds": preds}


def _conformal_halfwidth(y_true: np.ndarray, y_pred: np.ndarray,
                          alpha: float = 0.10) -> float:
    """
    Split-conformal half-width (A6.2): (1-alpha) quantile of |y - y_hat|.
    Distribution-free; assumes exchangeability of holdout residuals with future ones.
    +1 correction for finite-sample validity (Angelopoulos & Bates, 2023).
    """
    if len(y_true) == 0:
        return 0.0
    residuals = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) == 0:
        return 0.0
    n = len(residuals)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, q_level))


def _train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> tuple:
    """Train LightGBM regressor. Returns (model, test_predictions)."""
    # Regularisation scaled to dataset size: smaller leaves + L1/L2 to prevent
    # the 35-feature model from memorising the ~150-row training window.
    params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.02,
        num_leaves=15,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_samples=10,
        reg_alpha=0.5,
        reg_lambda=2.0,
        n_jobs=-1,
        verbose=-1,
        random_state=42,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds



def _train_catboost(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> tuple:
    """Train CatBoostRegressor with ordered boosting. Returns (model, test_predictions)."""
    model = CatBoostRegressor(
        iterations=350,
        depth=4,               # shallower to prevent overfitting on small datasets
        learning_rate=0.03,
        l2_leaf_reg=6,         # stronger L2 (was 3)
        min_data_in_leaf=8,    # consistent with LGBM min_child_samples=10
        random_strength=0.5,
        bagging_temperature=0.8,
        od_type='Iter',
        od_wait=40,
        verbose=0,
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds



def _generate_oof_predictions(X_train: np.ndarray, y_train: np.ndarray,
                               n_folds: int = 5,
                               purge_gap: int = 5,
                               embargo_gap: int = 5) -> tuple:
    """
    Purged expanding-window OOF with embargo (P2-4, López de Prado AFML §7).

    For each fold k:
      - Train on indices [0, val_start - purge_gap)
      - Validate on [val_start, val_end)
      - Embargo: next fold cannot start until val_end + embargo_gap

    Purge prevents the model from memorising the near-boundary signal that
    was in its training window. Embargo prevents the validation residuals
    from echoing into the next fold's training features (e.g. lagged returns).

    Returns (oof_xgb, oof_lgbm, oof_catboost) arrays aligned to y_train.
    """
    n = len(X_train)
    # At least 40% of data for the first training window; min 20 samples.
    min_train = max(int(n * 0.40), 20 + purge_gap)
    step      = max((n - min_train) // n_folds, 5)

    oof_xgb      = np.zeros(n)
    oof_lgbm     = np.zeros(n)
    oof_catboost = np.zeros(n)

    # Stronger regularisation: small datasets overfit badly with default params.
    xgb_params = dict(
        objective='reg:squarederror',
        n_estimators=150,
        max_depth=3,           # shallow — ~121 training rows
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,    # require ≥5 samples per leaf
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=2.0,
        n_jobs=-1,
        verbosity=0,
    )
    lgbm_params = dict(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        num_leaves=15,         # log2(15) ≈ 4 splits → matches max_depth=3 XGB
        min_child_samples=5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        n_jobs=-1,
        verbose=-1,
    )
    cb_params = dict(
        iterations=150,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=5,
        min_data_in_leaf=5,
        od_type='Iter',
        od_wait=20,
        verbose=0,
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
    )

    val_cursor = min_train  # next fold's validation starts here (respects embargo)

    for fold in range(n_folds):
        val_start = val_cursor
        val_end   = min(val_start + step, n)

        if val_start >= n or val_start >= val_end:
            break

        # Purge: exclude the [val_start - purge_gap, val_start) rows from train
        train_end = max(0, val_start - purge_gap)
        if train_end < 15:
            val_cursor = val_end + embargo_gap
            continue

        Xtr, ytr = X_train[:train_end], y_train[:train_end]
        Xval     = X_train[val_start:val_end]

        xm = xgb.XGBRegressor(**xgb_params)
        xm.fit(Xtr, ytr)
        oof_xgb[val_start:val_end] = xm.predict(Xval)

        if LGBM_AVAILABLE:
            lm = lgb.LGBMRegressor(**lgbm_params)
            lm.fit(Xtr, ytr)
            oof_lgbm[val_start:val_end] = lm.predict(Xval)
        else:
            oof_lgbm[val_start:val_end] = oof_xgb[val_start:val_end]

        if CATBOOST_AVAILABLE:
            cm = CatBoostRegressor(**cb_params)
            cm.fit(Xtr, ytr)
            oof_catboost[val_start:val_end] = cm.predict(Xval)
        else:
            oof_catboost[val_start:val_end] = oof_xgb[val_start:val_end]

        # Embargo: skip the `embargo_gap` rows after this validation window
        val_cursor = val_end + embargo_gap

    return oof_xgb, oof_lgbm, oof_catboost



def _calibrate_direction_probability(oof_preds: np.ndarray, y_train: np.ndarray,
                                      test_preds: np.ndarray,
                                      return_calibrator: bool = False):
    """
    Calibrated P(direction up) via isotonic regression on OOF predictions (A1.1).

    Isotonic is fit on OOF (rows the base model never saw), applied to test preds.
    Fitting on training preds would over-flatten the curve; fitting on test would
    leak labels. Same strategy as sklearn's CalibratedClassifierCV with cv=5.

    When `return_calibrator=True` returns `(probs, calibrate_fn)` so callers can
    apply the same transformation to a live single-sample inference (Bug D fix).
    """
    assert oof_preds.shape == y_train.shape, (
        f"oof_preds ({oof_preds.shape}) and y_train ({y_train.shape}) must align — "
        "isotonic calibration requires paired OOF pred ↔ actual label rows"
    )

    y_dir = (y_train > 0).astype(float)

    # Pre-min_train rows are zero in OOF — exclude them from the isotonic fit
    valid = np.abs(oof_preds) > 1e-12
    if valid.sum() < 10:
        _fallback = lambda p: np.clip(0.5 + p * 10, 0.05, 0.95)
        prob = _fallback(test_preds)
        return (prob, _fallback) if return_calibrator else prob

    iso = IsotonicRegression(out_of_bounds='clip')
    sort_idx = np.argsort(oof_preds[valid])
    iso.fit(oof_preds[valid][sort_idx], y_dir[valid][sort_idx])

    # Bug P fix: collapse fallback scale derived from test_preds std, not
    # hardcoded 50.  The old value mapped ±2% log-returns to ±0.5 prob shift —
    # fine for mid-cap but broke for high/low-vol tickers. Using 1/σ * 0.25
    # keeps ±1σ of raw predictions at ±0.25 prob regardless of asset.
    _pred_sig = max(float(np.std(test_preds)), 1e-6)
    _fb_scale = 0.25 / _pred_sig

    up_frac = float(y_dir[valid].mean())
    bias = up_frac - 0.5

    def _apply(preds: np.ndarray) -> np.ndarray:
        p = iso.predict(preds)
        if np.std(p) < 0.01:
            p = np.clip(0.5 + preds * _fb_scale, 0.02, 0.98)
        if abs(bias) > 0.03:
            p = p - bias
        return np.clip(p, 0.02, 0.98)

    prob = _apply(test_preds)
    return (prob, _apply) if return_calibrator else prob


def _compute_threshold_tuning(probs: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Per-ticker optimal decision threshold via Youden's J on the val fold (A1.5).
    Falls back to τ = 0.5 when val is too small or single-class.

    Output keys match `schemas.prediction.ThresholdTuning` (tau_star,
    accuracy_at_tau) — earlier versions returned optimal_threshold /
    accuracy_at_opt and silently dropped through the prediction_service
    DTO mapping, leaving the frontend verdict pill stuck on τ = 0.5.
    """
    probs = np.asarray(probs, dtype=float)
    y_dir = (np.asarray(y_true, dtype=float) > 0).astype(int)
    n = len(probs)
    fallback = {
        "tau_star": 0.5, "auc": 0.5, "accuracy_at_tau": 0.5, "n_samples": n,
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
            "tau_star": tau,
            "auc": auc,
            "accuracy_at_tau": acc,
            "n_samples": n,
        }
    except Exception:
        return fallback


def _compute_calibration_report(probs: np.ndarray, y_true: np.ndarray,
                                n_bins: int = 10) -> dict:
    """Reliability diagram data: ECE + Brier + per-bin vectors. Empty bins dropped."""
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



def _compute_shap_importance(xgb_model, X_test_scaled: np.ndarray,
                              feature_names: list) -> dict:
    """SHAP TreeExplainer importance (capped at 200 samples). Falls back to XGB native."""
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

    importance = pd.Series(
        xgb_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    return {
        'importance': importance.to_dict(),
        'top_features': importance.head(10).index.tolist(),
        'method': 'XGB_native'
    }


def _compute_conviction_precision(probs: np.ndarray, y_true: np.ndarray,
                                   threshold: float = 0.60) -> dict:
    """Precision restricted to high-conviction calls (prob >= threshold or <= 1-threshold).
    Go/no-go gate: bull_precision >= 0.55 at threshold=0.60 before paper trading."""
    y_dir     = (y_true > 0).astype(int)
    high_bull = probs >= threshold
    high_bear = probs <= (1.0 - threshold)
    return {
        "threshold":       threshold,
        "n_high_bull":     int(high_bull.sum()),
        "n_high_bear":     int(high_bear.sum()),
        "bull_precision":  float(y_dir[high_bull].mean()) if high_bull.sum() > 0 else None,
        "bear_precision":  float((1 - y_dir[high_bear]).mean()) if high_bear.sum() > 0 else None,
        "conviction_rate": float((high_bull | high_bear).mean()),
    }



def create_hybrid_model(df: pd.DataFrame, sentiment_features: dict,
                        fii_dii_data: pd.DataFrame = None,
                        vix_data: pd.DataFrame = None,
                        multi_source_sentiment: dict = None,
                        enable_automl: bool = False,
                        option_features: dict = None,
                        macro_features: pd.DataFrame = None,
                        ticker: str = None) -> tuple:
    """
    Train the full hybrid ensemble (XGB + LGBM + CatBoost + multi-task LSTM/GRU +
    Ridge stacker) on OHLCV + sentiment + FII/DII + VIX + macro + option-chain data.

    Strict 80/20 time-series split — scaler fit on training only.
    Returns (processed_df, results_df, models, scaler, features, metrics).
    """
    # Bug K fix: set global seeds so the same input always produces the same
    # model. random_state=42 on LGBM/CatBoost only covered their internal
    # sampling; XGB colsample, GRU/LSTM weight init, and dropout were unseeded.
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)

    df_proc = create_advanced_features(df)

    sentiment_df = pd.DataFrame(sentiment_features.items(), columns=["Date", "Sentiment"])
    if not sentiment_df.empty:
        # Bug H: convert UTC timestamps to IST before flooring to date so that
        # late-evening UTC headlines (e.g. 22:00 UTC = 03:30 IST next day) don't
        # bleed onto the wrong trading row.
        _raw_dt = pd.to_datetime(sentiment_df['Date'], utc=True, errors='coerce')
        _has_tz = _raw_dt.dt.tz is not None
        if _has_tz:
            sentiment_df['Date'] = (_raw_dt.dt.tz_convert('Asia/Kolkata')
                                          .dt.normalize()
                                          .dt.tz_localize(None))
        else:
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.normalize()
        df_proc = df_proc.reset_index().merge(sentiment_df, on="Date", how="left").set_index("Date")
        df_proc['Sentiment'] = pd.to_numeric(df_proc['Sentiment'], errors='coerce').fillna(0)
    else:
        df_proc['Sentiment'] = 0.0

    if multi_source_sentiment:
        ms_score      = multi_source_sentiment.get('combined_sentiment', 0)
        ms_confidence = multi_source_sentiment.get('confidence', 0)
        df_proc['Multi_Sentiment']      = ms_score
        df_proc['Sentiment_Confidence'] = ms_confidence
    else:
        df_proc['Multi_Sentiment']      = df_proc['Sentiment']
        df_proc['Sentiment_Confidence'] = 0.5

    if fii_dii_data is not None and not fii_dii_data.empty:
        df_proc = df_proc.join(fii_dii_data[['FII_Net', 'DII_Net']], how='left')
        df_proc['FII_Net'] = df_proc['FII_Net'].fillna(0)
        df_proc['DII_Net'] = df_proc['DII_Net'].fillna(0)
        # Bug B fix: normalization scale computed on training rows only so future
        # data doesn't leak through the denominator. Approximate train boundary
        # at this stage (target dropna hasn't happened yet) — differs from the
        # final train_size by at most ~20 rows.
        _fii_norm_end = max(1, int(len(df_proc) * ModelConfig.TRAIN_TEST_SPLIT))
        _fii_scale = df_proc['FII_Net'].iloc[:_fii_norm_end].abs().max() + 1e-6
        _dii_scale = df_proc['DII_Net'].iloc[:_fii_norm_end].abs().max() + 1e-6
        df_proc['FII_Net_Norm'] = df_proc['FII_Net'] / _fii_scale
        df_proc['DII_Net_Norm'] = df_proc['DII_Net'] / _dii_scale
        df_proc['FII_5D_Avg']   = df_proc['FII_Net_Norm'].rolling(5, min_periods=1).mean()
        df_proc['DII_5D_Avg']   = df_proc['DII_Net_Norm'].rolling(5, min_periods=1).mean()
    else:
        for col in ['FII_Net_Norm', 'DII_Net_Norm', 'FII_5D_Avg', 'DII_5D_Avg']:
            df_proc[col] = 0.0

    # A5.2 — macro factors. Non-trading days (US holidays) are forward-filled.
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

    # P2-10 / cross-sectional relative strength: if NIFTY50 data is in macro_features
    # (added via data/macro.py MACRO_TICKERS), derive relative-strength features.
    # These are ALSO used by the cross-sectional model during inference so it sees
    # the same features it was trained on.
    _nifty_close_col = "NIFTY50_Close"   # column name produced by fetch_macro_series
    if _nifty_close_col in df_proc.columns and df_proc[_nifty_close_col].notna().any():
        _nifty_cl = df_proc[_nifty_close_col].ffill()
        _nifty_1d  = np.log(_nifty_cl / _nifty_cl.shift(1)).fillna(0).clip(-0.1, 0.1)
        _nifty_5d  = np.log(_nifty_cl / _nifty_cl.shift(5)).fillna(0).clip(-0.3, 0.3)
        _nifty_20d = np.log(_nifty_cl / _nifty_cl.shift(20)).fillna(0).clip(-0.5, 0.5)

        df_proc['Nifty_Ret']         = _nifty_1d
        df_proc['Rel_Strength_5D']   = (df_proc['Ret_5D']  - _nifty_5d).clip(-0.3, 0.3)
        df_proc['Rel_Strength_20D']  = (df_proc['Ret_20D'] - _nifty_20d).clip(-0.5, 0.5)

        # Rolling 20-day beta: cov(ticker, nifty) / var(nifty) — systematic risk exposure
        _ticker_1d = df_proc['Log_Ret']
        _beta = (
            _ticker_1d.rolling(20).cov(_nifty_1d)
            / (_nifty_1d.rolling(20).var() + 1e-10)
        ).clip(-3, 3).fillna(1.0)
        df_proc['Beta_20D'] = _beta

        for _rc in ['Nifty_Ret', 'Rel_Strength_5D', 'Rel_Strength_20D', 'Beta_20D']:
            if _rc not in _macro_cols:
                _macro_cols.append(_rc)

    # A4.3 — option-chain snapshot. Broadcast as a constant across all training rows;
    # near-zero weight on training, but lets the live prediction see PCR / IV-skew
    # without a separate inference path.
    _option_cols: list[str] = []
    if option_features:
        for key, val in option_features.items():
            col = f"opt_{key}"
            df_proc[col] = float(val) if val is not None else 0.0
            _option_cols.append(col)

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

    # Walk-forward pattern features (P2-9 + Fix 5). Sampled every 20 bars, ffilled.
    # Pattern_Target_Dist: signed distance to nearest confirmed pattern target / close.
    # Pattern_Bullish / Pattern_Bearish: binary flags for bullish / bearish confirmed patterns.
    _pattern_col = "Pattern_Target_Dist"
    _BEARISH_TYPES = {"bearish", "bear", "double top", "head and shoulders", "head & shoulders"}
    _BULLISH_TYPES = {"bullish", "bull", "double bottom", "inverse", "inv"}
    try:
        from models.visual_analyst import PatternAnalyst as _PA
        _pa_inst = _PA()
        _step = max(20, len(df_proc) // 50)        # ~50 sample points
        _ptgt_series:    dict[int, float] = {}
        _bullish_series: dict[int, float] = {}
        _bearish_series: dict[int, float] = {}
        _min_bars_for_patterns = 60
        for _t in range(_min_bars_for_patterns, len(df_proc), _step):
            _sub_df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].iloc[:_t]
            try:
                _res = _pa_inst.analyze_all_patterns(_sub_df)
                _pats = [
                    p for p in _res.get("patterns", [])
                    if str(p.get("Status", "")).upper() == "CONFIRMED"
                    and p.get("Target") is not None
                ]
                if _pats:
                    _cur = float(df["Close"].iloc[_t - 1])
                    _dists = [(float(p["Target"]) - _cur) / _cur for p in _pats]
                    _ptgt_series[_t] = min(_dists, key=abs)
                    # Classify each confirmed pattern by its Type string
                    _bull_flag = 0.0
                    _bear_flag = 0.0
                    for _p in _pats:
                        _ptype = str(_p.get("Type", "") + " " + _p.get("Pattern", "")).lower()
                        if any(k in _ptype for k in _BULLISH_TYPES):
                            _bull_flag = 1.0
                        if any(k in _ptype for k in _BEARISH_TYPES):
                            _bear_flag = 1.0
                    _bullish_series[_t] = _bull_flag
                    _bearish_series[_t] = _bear_flag
                else:
                    _ptgt_series[_t]    = 0.0
                    _bullish_series[_t] = 0.0
                    _bearish_series[_t] = 0.0
            except Exception:
                _ptgt_series[_t]    = 0.0
                _bullish_series[_t] = 0.0
                _bearish_series[_t] = 0.0

        def _build_pattern_col(series_dict: dict, clip_abs: float | None = None) -> pd.Series:
            col = pd.Series(0.0, index=df_proc.index)
            for _t, _v in series_dict.items():
                if _t < len(df_proc):
                    col.iloc[_t] = _v
            col = col.replace(0.0, float("nan")).ffill().fillna(0.0)
            return col.clip(-clip_abs, clip_abs) if clip_abs else col

        df_proc[_pattern_col]     = _build_pattern_col(_ptgt_series, 0.15)
        df_proc["Pattern_Bullish"] = _build_pattern_col(_bullish_series)
        df_proc["Pattern_Bearish"] = _build_pattern_col(_bearish_series)
    except Exception:
        df_proc[_pattern_col]      = 0.0
        df_proc["Pattern_Bullish"] = 0.0
        df_proc["Pattern_Bearish"] = 0.0

    df_proc['Target_Return'] = df_proc['Log_Ret'].shift(-1)
    # Bug D fix: save the last row BEFORE dropna — this is the most recent close
    # which has no Target_Return yet (it's tomorrow's return). After dropna it
    # disappears, so directional_prob[-1] was always one bar stale.
    _live_row_df = df_proc.iloc[[-1]].copy()
    df_proc.dropna(inplace=True)

    features = [
        'Log_Ret', 'Volatility_5D', 'RSI_Norm', 'Vol_Ratio', 'MA_Div',
        'MACD_Norm', 'MACD_Hist_Norm', 'BB_PctB', 'ATR_Norm', 'OBV_Slope',
        'Ret_2D', 'Ret_5D', 'Ret_10D', 'Ret_20D',
        'CMF_20', 'Williams_R_Norm', 'RSI_Bear_Div', 'RSI_Bull_Div',
        'Sentiment', 'Multi_Sentiment', 'Sentiment_Confidence',
        'FII_Net_Norm', 'DII_Net_Norm', 'FII_5D_Avg', 'DII_5D_Avg',
        'VIX_Norm', 'VIX_Change',
        'Pattern_Target_Dist', 'Pattern_Bullish', 'Pattern_Bearish',  # P2-9
        # P2-11: trend-strength features (ADX, Aroon, EMA slopes)
        'ADX_Norm', 'ADX_Trend', 'DI_Diff_Norm',
        'Aroon_Osc_Norm',
        'EMA9_Slope', 'EMA21_Slope', 'EMA50_Slope',
    ]
    features.extend(_macro_cols)   # A5.2
    features.extend(_option_cols)  # A4.3

    # Bug 3 fix: drop features that are constant over the training window.
    # `multi_source_sentiment` (single dict) and `option_features` (snapshot)
    # both broadcast a scalar across every historical row, so the model trained
    # on a constant and then saw a different live value at inference — it
    # could never learn anything from those columns. Same protection applies
    # to any future broadcast-only feature. The training-fold variance check
    # uses the same TRAIN_TEST_SPLIT we apply below so we don't peek at test.
    _split_idx = max(1, int(len(df_proc) * ModelConfig.TRAIN_TEST_SPLIT))
    _train_view = df_proc[features].iloc[:_split_idx]
    _train_std = _train_view.std(axis=0, ddof=0).fillna(0.0)
    _zero_var = [c for c in features if float(_train_std.get(c, 0.0)) < 1e-9]
    if _zero_var:
        features = [c for c in features if c not in _zero_var]

    X = df_proc[features].values
    y = df_proc['Target_Return'].values

    train_size    = int(len(X) * ModelConfig.TRAIN_TEST_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # P2-3 / F14: stronger regularisation — 28+ features on ≤200 training rows
    # means the default depth-4, no-L1 XGB memorises noise. Shallower trees +
    # L1/L2 penalties keep test accuracy up while reducing ECE.
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=250,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=2.0,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)

    lgbm_model = None
    lgbm_pred  = None
    if LGBM_AVAILABLE:
        lgbm_model, lgbm_pred = _train_lightgbm(X_train_scaled, y_train, X_test_scaled)
        lgbm_rmse = float(np.sqrt(mean_squared_error(y_test, lgbm_pred)))
    else:
        lgbm_rmse = float('inf')

    catboost_model = None
    catboost_pred  = None
    if CATBOOST_AVAILABLE:
        catboost_model, catboost_pred = _train_catboost(X_train_scaled, y_train, X_test_scaled)
        catboost_rmse = float(np.sqrt(mean_squared_error(y_test, catboost_pred)))
    else:
        catboost_rmse = float('inf')

    oof_xgb, oof_lgbm, oof_catboost = _generate_oof_predictions(
        X_train_scaled, y_train, n_folds=5
    )

    # Lag-1 of XGB OOF is a temporal proxy for the RNN in the meta-learner
    oof_lag1 = np.roll(oof_xgb, 1)
    oof_lag1[0] = 0.0

    valid_oof = np.abs(oof_xgb) > 1e-12
    stack_train = np.column_stack([oof_xgb, oof_lgbm, oof_catboost, oof_lag1])

    # Multi-task LSTM+GRU: shared backbone → 3 heads (return MSE, direction BCE,
    # volatility MSE). Direction head uses BCE so the up/down signal gets its own
    # gradient path rather than relying on the MSE gradient to learn directionality.
    rnn_lookback = ModelConfig.LOOKBACK_PERIOD

    X_train_3d, y_train_rnn = [], []
    for i in range(rnn_lookback, len(X_train_scaled)):
        X_train_3d.append(X_train_scaled[i - rnn_lookback:i])
        y_train_rnn.append(y_train[i])
    X_train_3d  = np.array(X_train_3d)
    y_train_rnn = np.array(y_train_rnn)

    y_train_direction = (y_train_rnn > 0).astype(np.float32)

    _all_returns = pd.Series(y_train)
    _rolling_vol = _all_returns.rolling(5, min_periods=1).std().bfill().values
    y_train_volatility = _rolling_vol[rnn_lookback:].astype(np.float32)
    seq_len = len(y_train_rnn)
    if len(y_train_volatility) > seq_len:
        y_train_volatility = y_train_volatility[:seq_len]
    elif len(y_train_volatility) < seq_len:
        y_train_volatility = np.pad(
            y_train_volatility, (0, seq_len - len(y_train_volatility)), mode='edge'
        )

    # Bug G: standardise return and volatility targets to ~unit variance so that
    # MSE losses are on comparable scales with the BCE direction loss (~0.5–0.7).
    # Direction labels stay in {0,1}; only the regression heads are scaled.
    _ret_scale = float(np.std(y_train_rnn)) or 1.0
    _vol_scale = float(np.std(y_train_volatility)) or 1.0
    y_train_rnn_s = (y_train_rnn / _ret_scale).astype(np.float32)
    y_train_vol_s = (y_train_volatility / _vol_scale).astype(np.float32)

    X_combined = np.vstack([X_train_scaled[-rnn_lookback:], X_test_scaled])
    X_test_3d  = np.array([
        X_combined[i - rnn_lookback:i]
        for i in range(rnn_lookback, len(X_combined))
    ])

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

    return_out    = Dense(1, name='return_out')(shared)
    direction_out = Dense(1, activation='sigmoid', name='direction_out')(shared)

    from tensorflow.keras.layers import Activation
    # Volatility head: pre-activation Dense, then softplus to keep predictions
    # strictly positive (volatility is non-negative by definition). Earlier
    # versions wired the model output to `volatility_dense` (the raw Dense),
    # which could emit negative values — benign for the current consumers
    # (display only), but wrong on its face. Use the post-softplus tensor.
    volatility_pre = Dense(1)(shared)
    volatility_out = Activation('softplus', name='volatility_out')(volatility_pre)

    model_rnn = Model(inputs=input_layer, outputs=[return_out, direction_out, volatility_out])
    # label_smoothing=0.1 prevents the direction head from collapsing to the
    # constant-prior attractor before the rest of the network has converged.
    model_rnn.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'return_out':    'mse',
            'direction_out': tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            'volatility_out':'mse',
        },
        loss_weights={'return_out': 0.5, 'direction_out': 0.3, 'volatility_out': 0.2}
    )

    rnn_callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, min_delta=1e-6),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    ]
    model_rnn.fit(
        X_train_3d,
        {
            'return_out':    y_train_rnn_s,   # Bug G: unit-variance scaled
            'direction_out': y_train_direction,
            'volatility_out':y_train_vol_s,   # Bug G: unit-variance scaled
        },
        epochs=200,
        batch_size=32,
        verbose=0,
        shuffle=False,
        callbacks=rnn_callbacks
    )

    rnn_outputs      = model_rnn.predict(X_test_3d, verbose=0)
    rnn_pred         = rnn_outputs[0].flatten() * _ret_scale   # Bug G: unscale
    rnn_dir_prob     = rnn_outputs[1].flatten()
    rnn_vol_pred     = rnn_outputs[2].flatten() * _vol_scale   # Bug G: unscale

    stack_test = np.column_stack([
        xgb_pred,
        lgbm_pred      if lgbm_pred      is not None else xgb_pred,
        catboost_pred  if catboost_pred  is not None else xgb_pred,
        rnn_pred,
    ])

    if valid_oof.sum() >= 10:
        meta = Ridge(alpha=1.0, fit_intercept=True)
        meta.fit(stack_train[valid_oof], y_train[valid_oof])
        stacked_pred = meta.predict(stack_test)
    else:
        stacked_pred = xgb_pred.copy()
        meta = None

    # ── Directional probability: calibrated tree blend + Platt-calibrated RNN ──
    n_tree_models = sum([1, LGBM_AVAILABLE, CATBOOST_AVAILABLE])
    oof_tree_avg  = (oof_xgb + oof_lgbm + oof_catboost) / n_tree_models
    # Fix (calibration domain): the isotonic calibrator is FIT on the tree-average
    # OOF predictions, so it must be APPLIED to the matching test-time tree average
    # — not to `stacked_pred` (the Ridge meta output, a different distribution with
    # its own intercept/scale). Feeding stacked_pred through a calibrator fit on the
    # tree average is a domain mismatch that produced wildly miscalibrated
    # probabilities (ECE≈0.35, avg P(up)≈0.80 against a 0.48 base rate).
    tree_test_avg = (
        xgb_pred
        + (lgbm_pred     if lgbm_pred     is not None else xgb_pred)
        + (catboost_pred if catboost_pred is not None else xgb_pred)
    ) / n_tree_models
    tree_dir_prob, _tree_calibrate = _calibrate_direction_probability(
        oof_tree_avg, y_train, tree_test_avg, return_calibrator=True
    )

    # Val/test split — defined early so every downstream computation can use it
    # without looking at test data when picking thresholds or scale factors.
    # First half = val  (τ-tuning, Platt fit, RMSE model selection, variance scale)
    # Last half  = test (ECE/Brier/conviction, reported metrics)
    n_split = len(y_test)
    val_end = n_split // 2 if n_split >= 40 else None
    y_val  = y_test[:val_end] if val_end else np.array([])
    y_eval = y_test[val_end:] if val_end else y_test

    # Bug F fix: Platt-scale rnn_dir_prob on val before blending.
    # The tree side is isotonic-calibrated; blending it with raw sigmoid degrades
    # the blended ECE by mixing two differently-shaped probability distributions.
    from sklearn.linear_model import LogisticRegression as _LR
    _rnn_val = rnn_dir_prob[:val_end] if val_end else None
    _platt_cal = None
    if (_rnn_val is not None and len(_rnn_val) >= 20
            and len(np.unique((y_val > 0).astype(int))) == 2):
        _platt_cal = _LR(C=1.0, solver='lbfgs', max_iter=300)
        _platt_cal.fit(_rnn_val.reshape(-1, 1), (y_val > 0).astype(int))
        rnn_dir_prob_cal = _platt_cal.predict_proba(rnn_dir_prob.reshape(-1, 1))[:, 1]
    else:
        rnn_dir_prob_cal = rnn_dir_prob

    directional_prob = 0.60 * tree_dir_prob + 0.40 * rnn_dir_prob_cal

    # ── Final-stage calibration (Fix): a convex blend of two individually
    # calibrated probabilities is NOT itself calibrated. With ~51% directional
    # skill the raw blend was severely overconfident (avg P(up)≈0.80 vs a ~0.48
    # base rate, ECE≈0.35). Refit a monotonic Platt scaler on the val fold so the
    # reported probability reflects the model's *realised* skill — a low-skill
    # model is correctly shrunk toward the base rate instead of screaming 80%.
    # _final_dir_cal is also reused for the live single-row inference below.
    _final_dir_cal = None
    if val_end:
        _dir_val = directional_prob[:val_end]
        _yv_dir  = (y_val > 0).astype(int)
        if len(_dir_val) >= 20 and len(np.unique(_yv_dir)) == 2:
            try:
                _fc = _LR(C=1.0, solver='lbfgs', max_iter=300)
                _fc.fit(_dir_val.reshape(-1, 1), _yv_dir)
                _final_dir_cal = lambda p: _fc.predict_proba(
                    np.asarray(p, dtype=float).reshape(-1, 1)
                )[:, 1]
                directional_prob = np.clip(_final_dir_cal(directional_prob), 0.02, 0.98)
            except Exception:
                _final_dir_cal = None

    prob_val  = directional_prob[:val_end] if val_end else np.array([])
    prob_eval = directional_prob[val_end:] if val_end else directional_prob

    threshold_tuning = _compute_threshold_tuning(prob_val, y_val) if len(prob_val) else {
        "tau_star": 0.5, "auc": 0.5, "accuracy_at_tau": 0.5, "n_samples": 0,
    }
    calibration_report = _compute_calibration_report(prob_eval, y_eval, n_bins=10)
    conviction_precision = _compute_conviction_precision(prob_eval, y_eval, threshold=0.60)

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

    regime = detect_market_regime(df_proc)
    H      = regime.get('hurst', 0.5)

    # Test-fold RMSEs — used for reporting only, not for weight selection.
    xgb_rmse      = float(np.sqrt(mean_squared_error(y_test, xgb_pred)))
    rnn_rmse      = float(np.sqrt(mean_squared_error(y_test, rnn_pred)))
    stacked_rmse  = float(np.sqrt(mean_squared_error(y_test, stacked_pred)))

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
        base_cb_weight   = 0.20
        base_rnn_weight  = 0.15
        base_stack_weight= 0.40
    else:  # normal
        base_xgb_weight  = 0.18
        base_lgbm_weight = 0.17
        base_cb_weight   = 0.17
        base_rnn_weight  = 0.20
        base_stack_weight= 0.28

    # Bug A fix: select the best model using val-fold RMSEs, not test-fold.
    # Previously min(test_rmses) → +10% boost → evaluate on same test = circular.
    # val_end may be None when n_test < 40; fall back to full test in that case.
    _s = slice(0, val_end) if val_end else slice(None)
    _sel_rmses = {
        'xgb':     float(np.sqrt(mean_squared_error(y_test[_s], xgb_pred[_s]))),
        'rnn':     float(np.sqrt(mean_squared_error(y_test[_s], rnn_pred[_s]))),
        'stacked': float(np.sqrt(mean_squared_error(y_test[_s], stacked_pred[_s]))),
    }
    if lgbm_pred is not None:
        _sel_rmses['lgbm']     = float(np.sqrt(mean_squared_error(y_test[_s], lgbm_pred[_s])))
    if catboost_pred is not None:
        _sel_rmses['catboost'] = float(np.sqrt(mean_squared_error(y_test[_s], catboost_pred[_s])))

    best_model = min(_sel_rmses, key=_sel_rmses.get)
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

    # P2-2/P2-3: integrate pre-trained cross-sectional LightGBM if available.
    # Trained offline on 50 NSE tickers × 5 years → gives the per-ticker model
    # a cross-sectional prior calibrated on far more data.
    # OOF gate: cross-sectional model must beat random on its OOF eval.
    # Gate lowered to 50.05% — any positive edge above random contributes a
    # tiny weight. Pure OHLCV XS alpha ≈ 0.1–0.5% edge on 10y data.
    # Phase 3 signals (FII per-ticker, options flow) expected to push to 52-53%.
    # Weight scales 0→10% from gate to gate+4%.
    _XS_OOF_MIN = 0.5005
    _xs_pred   = None
    _xs_weight = 0.0
    try:
        from models.cross_sectional_trainer import CrossSectionalTrainer as _XST
        _xs = _XST()
        if _xs.is_trained() and _xs.load():
            _xs_oof = _xs._meta.get("oof_dir_accuracy", 0.0)
            if _xs_oof >= _XS_OOF_MIN:
                _xs_raw = _xs.predict(X_test_scaled, features, ticker=ticker)
                if _xs_raw is not None and len(_xs_raw) == len(y_test):
                    # Scale linearly: 0 weight at gate, 10% weight at gate+4%
                    _edge     = min((_xs_oof - _XS_OOF_MIN) / 0.04, 1.0)
                    _xs_pred  = _xs_raw
                    _xs_weight = 0.10 * _edge
            else:
                import logging as _lg
                _lg.getLogger(__name__).info(
                    "Cross-sectional model skipped: OOF accuracy %.1f%% < %.1f%% gate",
                    _xs_oof * 100, _XS_OOF_MIN * 100,
                )
    except Exception:
        pass

    if _xs_pred is not None:
        _per_ticker_w = 1.0 - _xs_weight
        ml_pred = (
            (_per_ticker_w * xgb_weight)   * xgb_pred
            + (_per_ticker_w * lgbm_weight)  * (lgbm_pred    if lgbm_pred    is not None else xgb_pred)
            + (_per_ticker_w * cb_weight)    * (catboost_pred if catboost_pred is not None else xgb_pred)
            + (_per_ticker_w * rnn_weight)   * rnn_pred
            + (_per_ticker_w * stack_weight) * stacked_pred
            + _xs_weight * _xs_pred
        )
    else:
        ml_pred = (
            xgb_weight   * xgb_pred
            + lgbm_weight  * (lgbm_pred     if lgbm_pred     is not None else xgb_pred)
            + cb_weight    * (catboost_pred  if catboost_pred is not None else xgb_pred)
            + rnn_weight   * rnn_pred
            + stack_weight * stacked_pred
        )

    stat_preds = [p for p in [arima_pred, prophet_pred] if p is not None]
    if stat_preds:
        stat_avg = np.mean(stat_preds, axis=0)
        # Bug I: ARIMA's recursive forecast reverts to the unconditional mean
        # within a few steps, so a flat 10% blend at every horizon just adds
        # noise. Fade the weight exponentially: full 10% at step 1, ~1% by
        # step 20, effectively zero by step 30+.
        _n = len(stat_avg)
        _steps = np.arange(_n, dtype=float)
        stat_w_arr = 0.10 * np.exp(-_steps / 5.0)
        hybrid_pred = (1 - stat_w_arr) * ml_pred + stat_w_arr * stat_avg
    else:
        hybrid_pred = ml_pred

    # Rescale prediction variance to match training std.
    # Bug E fix: compute the scale factor on the val fold only, then apply it
    # uniformly to all test predictions. Using np.std(hybrid_pred) over the
    # entire test array made every prediction depend on the others — re-running
    # on a different test window changed every individual value.
    train_std = np.std(y_train)
    _scale_slice = hybrid_pred[:val_end] if val_end else hybrid_pred
    pred_std = np.std(_scale_slice)

    if pred_std > 1e-8:
        scale_factor = train_std / pred_std
        hybrid_pred  = hybrid_pred * scale_factor
    else:
        hybrid_pred = xgb_pred.copy()
        _fb_std = np.std(xgb_pred[:val_end] if val_end else xgb_pred)
        if _fb_std > 1e-8:
            hybrid_pred = hybrid_pred * (train_std / _fb_std)

    max_pred    = 3 * train_std
    hybrid_pred = np.clip(hybrid_pred, -max_pred, max_pred)

    quantile_bundle = _train_quantile_xgb(X_train_scaled, y_train, X_test_scaled)
    # Bug C fix: split-conformal requires a holdout disjoint from where the band
    # is applied. Fit on val residuals; the band is then applied at inference time.
    _hw_y = y_test[:val_end] if val_end else y_test
    _hw_p = hybrid_pred[:val_end] if val_end else hybrid_pred
    conformal_hw = _conformal_halfwidth(_hw_y, _hw_p, alpha=0.10)
    shap_info = _compute_shap_importance(xgb_model, X_test_scaled, features)

    rmse              = float(np.sqrt(mean_squared_error(y_test, hybrid_pred)))
    correct_direction = np.sign(hybrid_pred) == np.sign(y_test)
    accuracy          = float(np.mean(correct_direction) * 100)

    up_rate              = float(np.mean(y_test > 0))
    train_up_rate        = float(np.mean(y_train > 0))  # used for calibration bias display
    naive_up_rate        = up_rate
    naive_accuracy       = float(max(up_rate, 1.0 - up_rate) * 100)  # always-majority-class
    # Bug M fix: actual Bernoulli baseline, not hardcoded 0.25.
    naive_brier          = float(up_rate * (1.0 - up_rate))

    # Bug D fix: compute directional probability for the LIVE bar (the last
    # close that was dropped by dropna because it has no Target_Return yet).
    # directional_prob[-1] was one trading day stale — this run-time single-pass
    # inference uses the actual most-recent close.
    _live_dir_prob = float(directional_prob[-1])  # fallback
    try:
        _live_feats = [f for f in features if f in _live_row_df.columns]
        if len(_live_feats) == len(features):
            _lx = _live_row_df[features].values.astype(float)
            if not np.any(np.isnan(_lx)):
                _lx_s = scaler.transform(_lx)
                # Tree branch
                _lxgb = float(xgb_model.predict(_lx_s)[0])
                _llgb = float(lgbm_model.predict(_lx_s)[0]) if lgbm_model else _lxgb
                _lcb  = float(catboost_model.predict(_lx_s)[0]) if catboost_model else _lxgb
                # RNN branch: slide the lookback window one step forward
                _live_win = np.vstack([X_test_scaled[-(rnn_lookback - 1):], _lx_s])
                _live_3d  = _live_win.reshape(1, rnn_lookback, len(features))
                _lrnn_out = model_rnn.predict(_live_3d, verbose=0)
                _lrnn_ret = float(_lrnn_out[0].flatten()[0]) * _ret_scale  # Bug G: unscale
                _lrnn_dir = float(_lrnn_out[1].flatten()[0])
                # Meta-stacker
                if meta is not None:
                    _l_stacked = float(meta.predict(
                        np.array([[_lxgb, _llgb, _lcb, _lrnn_ret]])
                    )[0])
                else:
                    _l_stacked = _lxgb
                # Apply stored calibrators. The tree calibrator is fit on the tree
                # AVERAGE domain, so feed it the live tree average (not the stacked
                # output) to match the create-time behaviour fixed above.
                _l_tree_avg = (_lxgb + _llgb + _lcb) / n_tree_models
                _ltree_p = float(_tree_calibrate(np.array([_l_tree_avg]))[0])
                _lrnn_p  = (float(_platt_cal.predict_proba([[_lrnn_dir]])[0, 1])
                            if _platt_cal else _lrnn_dir)
                _live_dir_prob = float(np.clip(0.60 * _ltree_p + 0.40 * _lrnn_p, 0.02, 0.98))
                # Apply the same final-stage calibration used for the test fold so
                # the live probability is on the same (honest) scale as reported ECE.
                if _final_dir_cal is not None:
                    _live_dir_prob = float(np.clip(_final_dir_cal([_live_dir_prob])[0], 0.02, 0.98))
    except Exception:
        pass  # any failure falls back to directional_prob[-1]

    test_dates = df_proc.index[train_size:]
    _scale = lambda p: p * (train_std / (np.std(p) + 1e-8))
    results_df = pd.DataFrame({
        'Actual_Return':    y_test,
        'Predicted_Return': hybrid_pred,
        'XGB_Return':       _scale(xgb_pred),
        'LGBM_Return':      _scale(lgbm_pred)     if lgbm_pred     is not None else _scale(xgb_pred),
        'CatBoost_Return':  _scale(catboost_pred) if catboost_pred is not None else _scale(xgb_pred),
        'RNN_Return':       _scale(rnn_pred),
        'Directional_Prob': directional_prob * 100,
        'RNN_Dir_Prob':     rnn_dir_prob * 100,
        'RNN_Vol_Pred':     rnn_vol_pred,
    }, index=test_dates)

    # P2-6: Use small fixed windows (20 bars ≈ 1 month) with a 5-bar step.
    # The previous scheme used 20% of test data as the window, which with a
    # 24-bar test set produced ~5 windows of 5 samples each (SE ≈ 20%).
    # With 5y of data (≈318 test bars) and these settings we get ~50 windows.
    # The warm-up min_wf_train is capped at 50 bars so most of the test
    # fold is actually evaluated.
    wf_window    = max(20, min(30, len(y_test) // 10))   # 20–30 bars per window
    wf_step      = max(5, wf_window // 4)                # ≥25% overlap
    min_wf_train = max(wf_window, min(int(len(y_test) * 0.15), 60))
    window_records = []

    for start in range(min_wf_train, len(y_test) - wf_window + 1, wf_step):
        end = min(start + wf_window, len(y_test))
        w_preds  = hybrid_pred[start:end]
        w_actual = y_test[start:end]
        if len(w_preds) < 10:
            continue
        # Stored as fraction — frontend multiplies by 100 itself
        w_acc = float(np.mean(np.sign(w_preds) == np.sign(w_actual)))
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
        _acc_frac            = accuracy / 100.0
        walkforward_accuracy = _acc_frac
        walkforward_std      = 0.0
        walkforward_min      = _acc_frac
        walkforward_max      = _acc_frac

    metrics = {
        'rmse':                 rmse,
        'accuracy':             accuracy,
        'xgb_weight':           xgb_weight,
        'lgbm_weight':          lgbm_weight,
        'catboost_weight':      cb_weight,
        'rnn_weight':           rnn_weight,
        'stack_weight':         stack_weight,
        'xgb_rmse':             xgb_rmse,
        'lgbm_rmse':            lgbm_rmse      if lgbm_pred      is not None else None,
        'catboost_rmse':        catboost_rmse  if catboost_pred  is not None else None,
        'rnn_rmse':             rnn_rmse,
        'stacked_rmse':         stacked_rmse,
        'lgbm_available':       LGBM_AVAILABLE,
        'catboost_available':   CATBOOST_AVAILABLE,
        'arima_used':           arima_pred is not None,
        'prophet_used':         prophet_pred is not None,
        'walkforward_accuracy': walkforward_accuracy,
        'walkforward_std':      walkforward_std,
        'walkforward_min':      walkforward_min,
        'walkforward_max':      walkforward_max,
        'walkforward_windows':  window_records,
        'regime':               regime['type'],
        'regime_detail':        regime['detail'],
        'hurst_exponent':       H,
        'avg_directional_prob':  float(np.mean(directional_prob) * 100),
        'last_directional_prob': float(_live_dir_prob * 100),  # Bug D: live row, not stale test[-1]
        'calibration':           calibration_report,       # A1.2 — ECE + reliability plot
        'threshold_tuning':      threshold_tuning,         # A1.5 — τ*, AUC
        'last_rnn_dir_prob':     float(rnn_dir_prob[-1] * 100)  if len(rnn_dir_prob)  > 0 else 50.0,
        'last_rnn_vol_pred':     float(rnn_vol_pred[-1])        if len(rnn_vol_pred)  > 0 else 0.0,
        'avg_rnn_vol_pred':      float(np.mean(rnn_vol_pred))   if len(rnn_vol_pred)  > 0 else 0.0,
        'shap_importance':      shap_info['importance'],
        'shap_top_features':    shap_info['top_features'],
        'shap_method':          shap_info['method'],
        'quantile_alphas':      list(quantile_bundle['test_preds'].keys()),
        'quantile_available':   len(quantile_bundle['models']) > 0,
        'conformal_halfwidth':  float(conformal_hw),
        'conformal_alpha':      0.10,
        'naive_up_rate':        naive_up_rate,
        'naive_accuracy':       naive_accuracy,
        'naive_brier':          naive_brier,
        'train_up_rate':        train_up_rate,
        'conviction_precision': conviction_precision,
    }

    return (df_proc, results_df,
            {
                'xgb': xgb_model, 'rnn': model_rnn,
                'lgbm': lgbm_model, 'catboost': catboost_model, 'meta': meta,
                'xgb_q10': quantile_bundle['models'].get(0.1),
                'xgb_q90': quantile_bundle['models'].get(0.9),
                'tree_calibrate': _tree_calibrate,   # Bug D: closure for live inference
                'platt_calibrator': _platt_cal,       # Bug D: Platt scaler for live RNN dir
            },
            scaler, features, metrics)



def hybrid_predict_next_day(models: dict, scaler: MinMaxScaler,
                            last_data_window: pd.DataFrame, features: list,
                            lookback: int = None) -> float:
    """Predict the next-day return using the trained hybrid ensemble."""
    lookback = lookback or ModelConfig.LOOKBACK_PERIOD

    x_latest        = last_data_window[features].iloc[-1:].values
    x_latest_scaled = scaler.transform(x_latest)
    xgb_pred        = models['xgb'].predict(x_latest_scaled)[0]

    window_data   = last_data_window[features].iloc[-lookback:].values
    window_scaled = scaler.transform(window_data)
    x_3d          = window_scaled.reshape((1, lookback, len(features)))
    rnn_outputs   = models['rnn'].predict(x_3d, verbose=0)
    # Multi-task RNN returns a list; single-output legacy model returns a 2D array
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
    Monte Carlo price forecast: bootstraps historical returns, applies a drift
    derived from directional_prob, and scales volatility by regime.
    Returns P5/P25/median/P75/P95 fan columns (+ 'Predicted Price' = median).

    Avoids the recursive-prediction artefact (flat-line) by using actual return
    bootstrap rather than feeding synthetic OHLCV rows back through the feature
    pipeline.
    """
    custom_business_day = CustomBusinessDay(calendar=IndiaHolidayCalendar())
    current_price = float(last_known_data['Close'].iloc[-1])
    last_date     = last_known_data.index[-1]

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
    except Exception as _exc:
        logger.warning("hybrid_predict_prices: anchor_return computation failed: %s", _exc)
        anchor_return = 0.0

    if df_proc_full is not None and 'Log_Ret' in df_proc_full.columns:
        hist_returns = df_proc_full['Log_Ret'].dropna().values
    else:
        hist_returns = np.log(
            last_known_data['Close'] / last_known_data['Close'].shift(1)
        ).dropna().values

    if len(hist_returns) < 20:
        hist_returns = np.zeros(100)

    mean_abs_ret  = float(np.mean(np.abs(hist_returns)))
    drift_per_day = np.clip((directional_prob - 0.5) * 2.0, -0.5, 0.5) * mean_abs_ret

    vol_scale = {
        'trending':      1.20,
        'mean_reverting':0.85,
        'high_volatility':1.40,
        'normal':        1.00,
    }.get(str(regime), 1.00)

    sample_idx  = np.random.randint(0, len(hist_returns), size=(n_paths, days))
    sampled_ret = hist_returns[sample_idx] * vol_scale

    # Day-1: anchor to model's point estimate with ½ historical vol so the fan
    # opens at h=1 instead of collapsing to zero width.
    day1_vol = float(np.std(hist_returns[-60:])) if len(hist_returns) >= 60 else float(np.std(hist_returns)) if len(hist_returns) > 1 else 0.005
    sampled_ret[:, 0]  = np.random.normal(anchor_return, day1_vol * 0.5, n_paths)
    sampled_ret[:, 1:] = sampled_ret[:, 1:] + drift_per_day

    cum_log_ret  = np.cumsum(sampled_ret, axis=1)
    price_paths  = current_price * np.exp(cum_log_ret)

    p5   = np.percentile(price_paths, 5,  axis=0)
    p25  = np.percentile(price_paths, 25, axis=0)
    p50  = np.percentile(price_paths, 50, axis=0)
    p75  = np.percentile(price_paths, 75, axis=0)
    p95  = np.percentile(price_paths, 95, axis=0)

    # Fraction of MC paths ending above anchor at each horizon — more honest
    # than stamping day-1's directional prob across all future days.
    prob_up_per_day = (price_paths > current_price).mean(axis=0)

    future_dates = []
    d = last_date
    for _ in range(days):
        d = d + custom_business_day
        future_dates.append(d)

    daily_changes = [(p50[i] - current_price) / current_price * 100 for i in range(days)]

    return pd.DataFrame({
        'Predicted Price':  p50,
        'Daily Change (%)': daily_changes,
        'P5':   p5,
        'P25':  p25,
        'P75':  p75,
        'P95':  p95,
        'Prob_Up': prob_up_per_day,
    }, index=future_dates)


def run_ablation_study(df: pd.DataFrame, sentiment_features: dict,
                       fii_dii_data: pd.DataFrame = None,
                       vix_data: pd.DataFrame = None,
                       multi_source_sentiment: dict = None) -> dict:
    """Measure accuracy delta when each feature group is removed."""
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
    """Forward-fill predicted prices on NSE non-trading days."""
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
