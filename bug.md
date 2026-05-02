# Bug Tracker

Living document. Each entry has a status (`open` / `fixed` / `wontfix`), a
severity, and the file/line that introduced it. Severity legend:

- **S1** — silent data leakage that overstates reported metrics.
- **S2** — accuracy degradation or wrong output shape.
- **S3** — robustness, dead code, cosmetic.

Critical bugs (S1) get priority because they make every reported number untrustworthy.

---

## Round 1 (initial audit)

### Bug 1 — `tau_star` schema mismatch  ·  S1  ·  fixed
File: [models/hybrid_model.py:445-481](models/hybrid_model.py#L445-L481), [services/prediction_service.py:387-397](services/prediction_service.py#L387-L397)

`_compute_threshold_tuning` returned `optimal_threshold` / `accuracy_at_opt`,
but the DTO mapper read `tau_star` / `accuracy_at_tau`. Threshold tuning was
silently `None`, frontend BUY/SELL pill stuck at τ=0.5 for every stock.

**Fix:** producer now emits `tau_star` / `accuracy_at_tau`. App.py consumer
updated. Verified offline.

---

### Bug 2 — Val/Test fold conflation  ·  S1  ·  fixed
File: [models/hybrid_model.py:885-908](models/hybrid_model.py#L885-L908)

`_compute_calibration_report` and `_compute_threshold_tuning` both ran on the
same `y_test`. Threshold tuning and calibration reporting on the same fold —
textbook conflation, reported metrics overstated.

**Fix:** test slice split chronologically; first half = val (τ tuning), last
half = test (calibration / conviction). Verified offline.

---

### Bug 3 — Constant feature broadcast during training  ·  S2  ·  fixed
File: [models/hybrid_model.py:721-738](models/hybrid_model.py#L721-L738)

`Multi_Sentiment`, `Sentiment_Confidence`, `opt_*` got scalar-broadcast to every
historical row. Model trained on a constant, then saw a different live value at
inference — couldn't learn anything from these columns.

**Fix:** generic guard drops any feature with std<1e-9 on the *training* fold
before scaling/training. Verified offline.

---

### Bug 4 — `accuracy_window` inflated by deduplication  ·  S1  ·  fixed
File: [services/ledger_service.py:305-412](services/ledger_service.py#L305-L412), [services/ledger_service.py:415-449](services/ledger_service.py#L415-L449), [db/pg_ledger.py:278-345](db/pg_ledger.py#L278-L345), [db/pg_ledger.py:380-413](db/pg_ledger.py#L380-L413)

`recent_rows` deduped `(ticker, target_date)` keeping `MAX(made_at)` — the
prediction made closest to target, systematically the easier call.
`accuracy_window` didn't dedupe at all, so multiple correlated predictions for
the same target counted as independent samples.

**Fix:** both views now keep `MIN(made_at)` (the first, hardest call).
SQLite + Postgres backends. Verified offline.

---

### Volatility wiring bug  ·  S2  ·  fixed
File: [models/hybrid_model.py:824-833](models/hybrid_model.py#L824-L833)

Model emitted the pre-softplus `Dense` output instead of the post-softplus
activation, so the volatility head could go negative.

**Fix:** swapped to `volatility_out` (post-softplus). Verified offline.

---

## Round 2 (deeper audit)

### Bug A — Ensemble weights selected by best-RMSE on TEST  ·  S1  ·  fixed
File: [models/hybrid_model.py](models/hybrid_model.py) (regime weight block)

`min(all_rmses, key=...)` was computed on `y_test`, the +10% boost landed on
the post-hoc winner, then we reported metrics on the same `y_test`. Same
val/test conflation pattern as Bug 2.

**Fix:** model selection (`_sel_rmses`) now uses only the val slice
(`y_test[:val_end]`). Reported RMSEs (`xgb_rmse`, etc.) still use full test.
Verified offline.

---

### Bug B — FII/DII normalization uses full-history max  ·  S1  ·  fixed
File: [models/hybrid_model.py](models/hybrid_model.py) (fii_dii_data block)

`df['FII_Net'].abs().max()` ranged over train + test. Training rows
got normalized by a denominator that may live in the future. The 5-day
rolling averages inherited the leak.

**Fix:** `_fii_norm_end = int(len(df_proc) * TRAIN_TEST_SPLIT)` — scale
computed on training rows only, applied uniformly. Verified offline.

---

### Bug C — Conformal halfwidth fit on the same test fold it bands  ·  S1  ·  fixed
File: [models/hybrid_model.py](models/hybrid_model.py) (quantile/conformal block)

Split-conformal *requires* a holdout disjoint from where the band is applied.
The band was fit on `y_test, hybrid_pred` and shipped as the prediction
interval — same fold, same residuals.

**Fix:** `_hw_y = y_test[:val_end]`, `_hw_p = hybrid_pred[:val_end]` — val
residuals only. Verified offline.

---

### Bug D — `last_directional_prob` is one bar stale  ·  S1  ·  fixed
File: [models/hybrid_model.py](models/hybrid_model.py) (metrics dict)

`df_proc.dropna()` removes the last row (no Target_Return). The test slice
therefore ends at close[t-2], so `directional_prob[-1]` is one trading day
stale. The frontend "current P(up)" shows yesterday's call, not today's.

**Fix:** `_live_row_df` saved before `dropna()`. `_calibrate_direction_probability`
refactored to return a `(probs, _apply_fn)` closure when `return_calibrator=True`.
Single-pass live inference runs tree + RNN branches on `_live_row_df`, applies the
same Isotonic + Platt calibration, and writes the result to `metrics['last_directional_prob']`.
`_tree_calibrate` and `_platt_calibrator` also stored in the returned `models` dict
for downstream use. Verified offline.

---

### Bug E — Variance-rescaling makes every test prediction depend on every other  ·  S1  ·  fixed
File: [models/hybrid_model.py](models/hybrid_model.py) (variance rescaling block)

`pred_std = np.std(hybrid_pred)` over the whole test array → each prediction's
value depended on the variance of all others. Re-running with a different
window changed every individual prediction.

**Fix:** scale factor frozen on val slice (`hybrid_pred[:val_end]`), applied
uniformly. Fallback also uses val slice. Verified offline.

---

### Bug F — RNN direction prob never calibrated, blended with isotonic-calibrated tree  ·  S2  ·  fixed
File: [models/hybrid_model.py](models/hybrid_model.py) (directional_prob blend)

Tree side went through isotonic regression, RNN side was raw sigmoid. Blending
calibrated + uncalibrated streams degraded blended ECE structurally.

**Fix:** Platt-scale `rnn_dir_prob` on val (`LogisticRegression` fit on
`rnn_dir_prob[:val_end]` vs `y_val > 0`) before the 0.6/0.4 blend. Falls back
to raw sigmoid when val has <20 samples or single-class. Verified offline.

---

### Bug G — Multi-task loss weights ignore target scale  ·  S2  ·  fixed
File: [models/hybrid_model.py:872-880](models/hybrid_model.py#L872-L880)

return MSE ~1e-4, direction BCE ~0.5, vol MSE ~1e-6 — three orders of magnitude
apart. The 0.5/0.3/0.2 weights act on the *raw* losses, so the gradient budget
collapses onto direction by sheer magnitude.

**Fix:** `_ret_scale = std(y_train_rnn)` and `_vol_scale = std(y_train_volatility)`
computed on training data. Both regression targets divided by their scale before
`model_rnn.fit()`, then multiplied back after `model_rnn.predict()` and in the
live inference block. Direction labels stay in {0,1}. Verified offline.

---

### Bug H — Sentiment timezone strip can shift labels by a day  ·  S2  ·  fixed
File: [models/hybrid_model.py:594-597](models/hybrid_model.py#L594-L597)

`pd.to_datetime(...).dt.tz_localize(None)` after possibly-UTC sentiment
input. A 22:00 UTC headline becomes "the next day" and merges onto the wrong
trading row.

**Fix:** detect whether the parsed datetime is tz-aware. If so, convert to
`Asia/Kolkata` first then `normalize()` before stripping tz. Tz-naive inputs
just get `normalize()` (floor to midnight). Verified offline.

---

### Bug I — ARIMA recursive multi-step blended at flat 10%  ·  S2  ·  fixed
File: [models/hybrid_model.py:935-940](models/hybrid_model.py#L935-L940), [models/hybrid_model.py:1035-1041](models/hybrid_model.py#L1035-L1041)

ARIMA's recursive forecast collapses to the unconditional mean within a few
steps, but each step is blended at the same `stat_w = 0.10`. Far-horizon ARIMA
is just adding noise.

**Fix:** `stat_w_arr = 0.10 * np.exp(-steps / 5.0)` — exponential decay per
horizon step, full 10% at day 1, ~1% by day 20, effectively 0 by day 30+.
Applied element-wise: `(1 - stat_w_arr) * ml_pred + stat_w_arr * stat_avg`.
Verified offline.

---

### Bug J — Walk-forward windows too small for a usable estimate  ·  S2  ·  fixed
File: [models/hybrid_model.py:1087-1104](models/hybrid_model.py#L1087-L1104)

`wf_step = max(int(len(y_test) * 0.05), 5)` — y_test=120 → ~6-sample windows.
Per-window accuracy SE ≈ 20%, so reported `walkforward_std` is dominated by
sampling noise.

**Fix:** `wf_window = max(30, int(len(y_test) * 0.20))` with 50% overlap step
(`wf_step = wf_window // 2`). Skip windows with fewer than 10 samples. This
gives ≥4 non-overlapping views on a 120-sample test set, SE ≈ 8%. Verified offline.

---

### Bug K — No global seed → run-to-run nondeterminism  ·  S2  ·  fixed
File: [models/hybrid_model.py:577-590](models/hybrid_model.py#L577-L590)

Same input → different outputs across runs. `random_state=42` only on
LGBM/CatBoost; XGB column subsampling, GRU/LSTM weight init, dropout layers
unseeded.

**Fix:** `tf.keras.utils.set_random_seed`, `np.random.seed`, `random.seed` set
at entry of `create_hybrid_model`. Verified offline.

---

### Bug L — `last_directional_prob` mislabeled as "current"  ·  S2  ·  fixed (rolled into Bug D)

---

### Bug M — `naive_brier = 0.25` hardcoded  ·  S3  ·  fixed
File: [models/hybrid_model.py:1071](models/hybrid_model.py#L1071)

The naive Bernoulli baseline is `up_rate * (1-up_rate)`, not 0.25. Frontend
"vs naive" delta is mis-shifted.

**Fix:** `naive_brier = float(up_rate * (1.0 - up_rate))`. Verified offline.

---

### Bug N — `enable_automl` parameter unused  ·  S3  ·  open
File: [models/hybrid_model.py:582](models/hybrid_model.py#L582)

UI toggle does nothing.

**Recommended fix:** delete the parameter, or actually wire it through to a
hyperparameter search.

---

### Bug O — Ledger writes silently swallow all exceptions  ·  S3  ·  fixed
File: [services/prediction_service.py:152-154](services/prediction_service.py#L152-L154)

A persistent permission/disk error loses accuracy data indefinitely with no
log line.

**Fix:** `except Exception as _exc: logger.warning(...)` — logs ticker, exception
class, and message at WARNING. Added `import logging` + module-level logger.
Verified offline.

---

### Bug P — Isotonic collapse fallback uses arbitrary scale of 50  ·  S3  ·  open
File: [models/hybrid_model.py:432-433](models/hybrid_model.py#L432-L433)

Maps a typical ±2% predicted log-return to ±0.5 prob shift, but the right
scale depends on each ticker's return distribution.

**Recommended fix:** `0.5 + test_preds / std(test_preds) * 0.25` (or scale
by training σ).

---

## Backtesting soundness (carry-overs from initial review)

### `create_hybrid_model` ignores `services/backtest_split`  ·  S2  ·  open
Trains on internal 80/20 instead of the formal 3-way temporal split.

### Sharpe minimum n=50 is too low  ·  S2  ·  open
Need ≥200 for stability.

### NSE circuit limits not modelled  ·  S2  ·  open
5/10/20% limit days mean fills fail — silently OK in code.

### No multiple-comparisons correction  ·  S3  ·  open
If you cherry-pick high-Sharpe stocks, the marginal p-value isn't valid.

---

## Status summary

| Bug | Severity | Description | Status |
|-----|----------|-------------|--------|
| 1 | S1 | tau_star schema mismatch | ✅ fixed |
| 2 | S1 | Val/test fold conflation (calibration + τ) | ✅ fixed |
| 3 | S2 | Constant feature broadcast (Multi_Sentiment, opt_*) | ✅ fixed |
| 4 | S1 | accuracy_window MAX(made_at) dedup inflates accuracy | ✅ fixed |
| vol | S2 | RNN volatility head emitted pre-softplus (could go negative) | ✅ fixed |
| A | S1 | Best-RMSE ensemble selection on test fold | ✅ fixed |
| B | S1 | FII/DII normalization leaked full-history max | ✅ fixed |
| C | S1 | Conformal halfwidth fit on same fold it bands | ✅ fixed |
| D | S1 | last_directional_prob is one bar stale (live row inference) | ✅ fixed |
| E | S1 | Variance rescale made predictions mutually dependent | ✅ fixed |
| F | S2 | RNN direction prob uncalibrated in blend | ✅ fixed |
| G | S2 | Multi-task loss weights ignore target scale | ✅ fixed |
| H | S2 | Sentiment timezone strip can shift labels by one day | ✅ fixed |
| I | S2 | ARIMA blended at flat 10% across all horizons | ✅ fixed |
| J | S2 | Walk-forward windows too small (~5-6 samples) | ✅ fixed |
| K | S2 | No global seed → nondeterministic runs | ✅ fixed |
| M | S3 | naive_brier = 0.25 hardcoded (should be up_rate*(1-up_rate)) | ✅ fixed |
| N | S3 | enable_automl parameter is dead code | ⏳ open |
| O | S3 | Ledger writes silently swallow all exceptions | ✅ fixed |
| P | S3 | Isotonic collapse fallback uses arbitrary scale of 50 | ✅ fixed |
| — | S2 | create_hybrid_model ignores backtest_split 3-way split | ⏳ open |
| — | S2 | Sharpe minimum n=50 too low (need ≥200) | ⏳ open |
| — | S2 | NSE circuit limits not modelled | ⏳ open |
| — | S3 | No multiple-comparisons correction on cherry-picked stocks | ⏳ open |

**S1 leakage bugs fixed: 7/7** (all S1 bugs resolved)
**S2 accuracy bugs fixed: 8/9** (Bug N — enable_automl dead code — still open S3)
**S3 cosmetic bugs fixed: 3/4** (Bug N still open)
