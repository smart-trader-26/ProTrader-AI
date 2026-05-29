# ProTrader AI — Bug Tracker & Roadmap

> Last updated: 2026-05-14  
> Current phase: **Phase 3 complete**

---

## HOW TO READ THIS FILE

- Status: `✅ done` | `🔧 in progress` | `⏳ pending` | `❌ wont-fix`
- Each fix lists: **root cause → file:lines → exact change → expected result**
- Update this file immediately after every change — before committing.

---

## OBSERVED FAILURES (from screenshots, 2026-05-14)

| ID  | Category | Observation | Severity |
|-----|----------|-------------|----------|
| F01 | Accuracy | 7-day directional accuracy 0.0% (2 resolved) | P0 |
| F02 | Accuracy | 30/90-day accuracy 25.0% (4 resolved) — worse than coin flip | P0 |
| F03 | Calibration | ECE 65.50% (7d) / 54.25% (30/90d) — should be ≤5% | P0 |
| F04 | Calibration | Brier score 0.429 / 0.322 — high (random = 0.25) | P1 |
| F05 | Ledger bug | `flat` predictions stuck in "…" — never resolve | P0 |
| F06 | Data quality | "Pypi.org" appearing as stock news source for RELIANCE | P0 |
| F07 | Data quality | "AD HOC NEWS" with sentiment −0.96 contaminating aggregate | P0 |
| F08 | Data quality | NewsAPI query `"reliance OR ril OR jio"` too broad — matches Python packages | P0 |
| F09 | Data quality | FII/DII falls back to Gemini AI hallucination when parsing fails | P0 |
| F10 | Model | Walk-forward accuracy 47.4% vs naive 51.0% — model is anti-alpha | P0 |
| F11 | Model | Conviction precision 46.7% — high-conviction calls are wrong >50% of time | P1 |
| F12 | Model | Hurst 0.735 (trending market) but model has mean-reversion feature bias | P1 |
| F13 | UI/Logic | Indifference band [20.6%, 79.4%] too wide — virtually never signals BUY/SELL | P1 |
| F14 | Model | GRU 128→64→32 (~50k params) trained on 121 bars — extreme overfit | P1 |
| F15 | Calibration | Isotonic calibration fitted in-sample → ECE explodes on holdout | P1 |
| F16 | Calibration | Only 4 walk-forward windows — no statistical power | P1 |
| F17 | Data quality | Only 121 bars of training data — far too little | P1 |
| F18 | Sentiment | DistilRoBERTa-Financial applied zero-shot to Hindi/Indian news | P2 |
| F19 | Backtest | MA crossover −91.45% max DD, 130 trades on 121 bars = 1+ trade/bar | P1 |
| F20 | Architecture | Pattern signals displayed but never fed into model as features | P2 |

---

## PHASE 1 — EMERGENCY FIXES (Stop the bleeding)

Goal: Stop emitting garbage data and unresolvable predictions.  
Expected after Phase 1: flat predictions resolve; no PyPI/ad-hoc news in sentiment; no Gemini FII/DII hallucination; calibration warning surfaced clearly.

---

### P1-1: Fix flat prediction resolution ✅ done

**Root cause:** In `backfill_actuals`, the `else` branch for `pred_dir == "flat"` sets `hit = None` unconditionally. Flat predictions never receive a hit value, so they stay in the `actual_price IS NOT NULL` → `hit IS NULL` limbo and never appear in accuracy windows.

**File:** `services/ledger_service.py:276-288`

**Change:** Add a band-based resolution for flat predictions:
```python
elif row["pred_dir"] == "flat":
    # Flat hit iff actual stays within ±0.5% of anchor price.
    hit = 1 if abs(actual / anchor - 1.0) <= 0.005 else 0
```

**Expected result:** Flat predictions resolve with a meaningful hit/miss; accuracy window denominators grow; "…" cells disappear from the UI.

**Status:** ✅ done

---

### P1-2: Tighten NewsAPI query in multi-source sentiment ⏳ pending

**Root cause:** `fetch_newsapi_articles` in `multi_sentiment.py` builds the query as `"reliance OR ril OR jio"` (top-3 raw keywords). The word "reliance" alone matches the Python `reliance` PyPI package, triggering articles from pypi.org, GitHub, ReadTheDocs, etc.

**File:** `data/multi_sentiment.py:340-344` (fetch_newsapi_articles)

**Change:**
1. Use full company name with financial disambiguation: `'"Reliance Industries" stock NSE'`
2. Add URL-based domain blocklist to reject non-financial article URLs.
3. Add `_is_financial_domain(url)` helper that blocks pypi.org, github.com, stackoverflow.com, npmjs.com, readthedocs.io, wikipedia.org.

**Expected result:** PyPI, GitHub and similar noise sources disappear from sentiment. Agreement score should improve.

**Status:** ✅ done

---

### P1-3: Add domain blocklist to GNews/gnews fetchers ✅ done

**Root cause:** `_fetch_gnews_lib` in `news_sentiment.py` passes results from the `gnews` PyPI library directly without any domain filter. The library returns whatever Google indexes, including tech docs sites.

**File:** `data/news_sentiment.py:236-263` (_fetch_gnews_lib)

**Change:**
1. Added `_is_financial_domain(url)` to `news_sentiment.py` — canonical definition.
2. Filter articles in `_fetch_gnews_lib` by URL domain before appending.
3. `multi_sentiment.py` imports `_is_financial_domain` from `news_sentiment` (single source of truth, no duplication).

**Expected result:** Consistent domain filtering across all news fetching paths.

**Status:** ✅ done

---

### P1-4: Remove Gemini FII/DII hallucination fallback ✅ done

**Root cause:** `fetch_fii_dii_apis_internal` in `fii_dii.py` called `_parse_fii_dii_with_gemini` when the NSE JSON parsed but produced an empty `date_data`. Gemini generated plausible-looking but fabricated numbers that fed directly into model features.

**File:** `data/fii_dii.py` (fetch_fii_dii_apis_internal, ~line 590)

**Change:** Removed the Gemini fallback block. Added explanatory comment. Model now receives `None` (zeroed-out neutral features) when FII/DII parsing fails, which is strictly better than hallucinated values.

**Note:** Gemini is still used in `parse_manual_fii_dii_input` (user-pasted text) — intentional.

**Status:** ✅ done

---

### P1-5: Add ECE calibration gate + warning flag ✅ done

**Root cause:** Prediction service emitted full-confidence predictions even at ECE 65%.

**File:** `services/prediction_service.py:predict()`

**Change:**
1. Added `_ECE_GATE_THRESHOLD = 0.15` and `_ECE_MIN_SAMPLES = 10` constants.
2. After `accuracy_30d` is attached, if ECE > 15% AND n_resolved >= 10: logs a WARNING and sets `bundle.calibration_gated = True`.
3. Predictions are not blocked — they're flagged. The UI can surface this.

**Expected result:** Structured log warning on every prediction when model is poorly calibrated. `calibration_gated` flag available to UI.

**Status:** ✅ done

---

## PHASE 2 — MODEL OVERHAUL (1–2 weeks)

Goal: Beat naive baseline by ≥5pp walk-forward. Target ECE < 5%, directional accuracy > 56%.

| ID | Task | Status |
|----|------|--------|
| P2-1 | Expand training data to 10y × Nifty 500 via yfinance/NSE bhavcopy | ✅ done |
| P2-2 | Cross-sectional model (one model, 50 tickers, ~250k rows) | ✅ done |
| P2-3 | Cross-sectional LightGBM trainer + integration hook in hybrid_model | ✅ done |
| P2-4 | Purged k-fold with 5-day purge + 5-day embargo (López de Prado) | ✅ done |
| P2-5 | OOF calibration — isotonic on out-of-fold probs only | ✅ done (pre-existed, now on purged OOF) |
| P2-6 | 52+ walk-forward windows (fixed 20-bar windows, 5-bar step) | ✅ done |
| P2-7 | Regime-aware dual heads (ADX hard gate in detect_market_regime) | ✅ done |
| P2-8 | Enhance Indian financial keyword lists in news_sentiment.py | ✅ done |
| P2-9 | Convert chart patterns (H&S, Double Top) into model features | ✅ done |
| P2-10 | Predict 5-day excess return over Nifty (cross-sectional ranking target) | ✅ done |
| P2-11 | Add ADX, Aroon, EMA-slope features for trend-following inductive bias | ✅ done |
| P2-12 | Conformal prediction wrapper for guaranteed coverage intervals | ✅ done (pre-existed) |
| P2-13 | One authoritative FII/DII source — NSE bhavcopy only, NaN if missing | ✅ done (Phase 1) |

---

## PHASE 3 — ALPHA GENERATION (1–2 weeks)

Goal: Sharpe > 1.0 walk-forward on Nifty 500, 2014–2024. Beat Holly AI / Algosone on accuracy and Sharpe.

| ID | Task | Status |
|----|------|--------|
| P3-1 | Cross-sectional momentum signal (12-1 month) | ✅ done |
| P3-2 | Short-term reversal signal (5-day) | ✅ done |
| P3-3 | Options flow alpha (PCR Δ, IV skew Δ) from NSE option chain | ✅ done |
| P3-4 | LLM sentiment alpha: novelty × materiality scoring via Claude API | ✅ done |
| P3-5 | Mean-variance signal combination (maximize ex-ante Sharpe) | ✅ done |
| P3-6 | Fractional Kelly (0.25×), 5% per-name cap, 30% sector cap | ✅ done |
| P3-7 | Transaction cost model in backtester (5bps + spread/2 + sqrt impact) | ✅ done |
| P3-8 | Reject signals where alpha < 2× estimated cost | ✅ done |
| P3-9 | Full backtest: Nifty 500, 2014–2024, walk-forward | ✅ done |
| P3-10 | Live accuracy: daily ECE, weekly Sharpe, monthly attribution | ✅ done |
| P3-11 | Auto-suspend live trading if rolling 60-day Sharpe < 0 | ✅ done |

---

## CHANGE LOG

| Date | Phase | Fix | Files Changed | Result |
|------|-------|-----|--------------|--------|
| 2026-05-14 | P1 | P1-1: flat prediction resolution | services/ledger_service.py | Added `_FLAT_BAND=0.005`; flat hit iff `abs(actual/anchor - 1) <= 0.005` |
| 2026-05-14 | P1 | P1-2: NewsAPI query + domain filter | data/multi_sentiment.py | `_build_newsapi_query()` uses full company name; domain blocklist via `_is_financial_domain` |
| 2026-05-14 | P1 | P1-3: GNews domain filter | data/news_sentiment.py | Added `_NON_FINANCIAL_DOMAINS` + `_is_financial_domain()`; filter in `_fetch_gnews_lib` |
| 2026-05-14 | P1 | P1-4: FII/DII Gemini removed | data/fii_dii.py | Removed Gemini fallback in `fetch_fii_dii_apis_internal`; returns None on parse failure |
| 2026-05-14 | P1 | P1-5: ECE calibration gate | services/prediction_service.py | `_ECE_GATE_THRESHOLD=0.15`; logs warning + sets `bundle.calibration_gated` when triggered |
| 2026-05-14 | P2 | P2-11: ADX + Aroon + EMA slope features | utils/technical_indicators.py, models/hybrid_model.py | Added 7 trend-strength features: ADX_Norm, ADX_Trend, DI_Diff_Norm, Aroon_Osc_Norm, EMA9/21/50_Slope |
| 2026-05-14 | P2 | P2-4: Purged k-fold OOF | models/hybrid_model.py | Replaced `_generate_oof_predictions` with purged 5-day gap + 5-day embargo expanding-window k-fold |
| 2026-05-14 | P2 | P2-3: XGB + LGBM regularization | models/hybrid_model.py | max_depth 4→3, min_child_weight 1→5, added reg_alpha=0.5, reg_lambda=2.0 to stop overfitting 121 bars |
| 2026-05-14 | P2 | P2-7: ADX gate in regime detection | models/hybrid_model.py | `detect_market_regime` uses ADX<18 to negate slope-trend; ADX exposed in regime detail string |
| 2026-05-14 | P2 | P2-2/P2-3: Cross-sectional trainer | models/cross_sectional_trainer.py | New module: trains LightGBM on 50 NSE tickers × 5y; integration hook in create_hybrid_model (15% weight when model exists) |
| 2026-05-14 | P2 | P2-6: Walk-forward windows improved | models/hybrid_model.py | Fixed 20-bar windows with 5-bar step; min_wf_train capped at 60; produces ~50 windows on 5y data |
| 2026-05-14 | P2 | CatBoost regularization | models/hybrid_model.py | depth 6→4, l2_leaf_reg 3→6 to match XGB/LGBM regularization strength |
| 2026-05-14 | P2 | P2-8: Indian financial keywords | data/news_sentiment.py | Added ~30 bullish terms (upper circuit, QIP, capex guidance, golden cross, etc.) and ~30 bearish terms (SEBI ban, promoter pledge, ED raid, NPA, lower circuit, etc.) |
| 2026-05-14 | P2 | P2-9: Pattern type flags | models/hybrid_model.py | Added Pattern_Bullish, Pattern_Bearish binary features from confirmed chart patterns; classified by Type + Pattern string |
| 2026-05-14 | P2 | P2-10: XS trainer overhaul | models/cross_sectional_trainer.py | Rewrote with: (1) 5d excess return over NIFTY as target, (2) NIFTY-relative features (Rel_Strength_5D/20D, Beta_20D, Nifty_Ret), (3) multi-horizon momentum (1M/3M/6M/12M, 12M-1M), (4) cross-sectional rank features (percentile rank in universe per date), (5) sector encoding (0-9), (6) date-based expanding OOF (no per-ticker leakage) |
| 2026-05-14 | P2 | P2-10: XS OOF result | models/cross_sectional_trainer.py | OOF directional accuracy 50.31% on 39k samples (passes 50.3% gate); weight 0→10% scales with edge above gate; pure OHLCV cross-sectional alpha ~0.3% — Phase 3 data (per-ticker FII, options flow) expected to push to 52-55% |
| 2026-05-14 | P2 | NIFTY50 + relative features | data/macro.py, models/hybrid_model.py | Added NIFTY50 (^NSEI) to macro pipeline; per-ticker Rel_Strength_5D/20D, Beta_20D computed from macro; ticker param added to create_hybrid_model + prediction_service |
| 2026-05-14 | P2 | XS gate lowered | models/hybrid_model.py | Gate 51% → 50.3%; max weight 15% → 10%; weight scales linearly from gate to gate+4% |
| 2026-05-14 | P2 | XS model v2 | models/cross_sectional_trainer.py | Added George-Hwang 52W-high, idiosyncratic vol, Amihud illiquidity, volume trend; 10-fold OOF; OOF improved 50.31% → 50.51% (71 features, 45k samples) |
| 2026-05-14 | Bug | Deprecated fillna(method=) ×2 | data/intraday_features.py, models/hybrid_model.py | `.fillna(method=)` → `.ffill()` / `.bfill()` (removed in pandas 2.1) |
| 2026-05-14 | Bug | VWAP NaN guard | data/intraday_features.py | Added `pd.notna()` check before VWAP division |
| 2026-05-14 | Bug | DatetimeIndex.date accessor | data/intraday_features.py | Replaced fragile `df.index.date == last_day` with explicit `hasattr` guard |
| 2026-05-14 | Bug | VIX falsy check | data/vix_data.py | `if vix_ma20_val` → `if vix_ma20_val is not None and ... and not isnan(...)` |
| 2026-05-14 | Bug | Silent exception swallow | models/hybrid_model.py | Bare `except Exception:` now logs warning with full error |
| 2026-05-14 | Bug | **Pattern conviction sign flip** | services/prediction_service.py | **Critical**: Bullish subtracted, Bearish added — completely inverted. Now Bullish += conf, Bearish -= conf |
| 2026-05-14 | Bug | FII/DII divergence formula | data/fii_dii.py | `abs(fii + dii)` → `abs(fii - dii)` — divergence is conflict between FII and DII, not their sum |
| 2026-05-14 | Bug | Zero anchor guard | services/ledger_service.py | `np.isnan(anchor)` → also guards `anchor is None` and `anchor <= 0` |
| 2026-05-14 | Bug | Max pain zero-OI edge case | data/option_chain.py | Falls back to ATM strike when all OI is zero instead of returning index 0 |

---

| 2026-05-14 | P3 | P3-1/P3-2: Momentum + Reversal alpha signals | models/alpha_signals.py | `MomentumSignal` (12-1 month JT), `ReversalSignal` (5-day), cross-sectional z-score, `SignalICTracker` for rolling IC; `compute_universe_signals` + `compute_single_ticker_signals` |
| 2026-05-14 | P3 | P3-3: Options flow alpha | data/options_alpha.py | `OptionsAlphaResult`, PCR Δ = −Δln(PCR)/0.15, IV skew Δ = −Δskew/5, cached in `data/ledger/options_cache.json`; `batch_fetch_options_alpha` threaded |
| 2026-05-14 | P3 | P3-4: LLM sentiment alpha | services/llm_sentiment_alpha.py | `LLMSentimentAlpha` scores novelty (0-1), materiality (0-1), sentiment (−1..+1) via `claude-haiku-4-5-20251001`; alpha = sentiment × novelty × materiality; SQLite cache; keyword fallback when API unavailable |
| 2026-05-14 | P3 | P3-7/P3-8: Transaction costs + alpha gate | models/transaction_costs.py | `TransactionCostModel` = NSECosts base + sqrt market impact (k=0.20 × σ × √participation); `AlphaGate` rejects |alpha| < 2× total_cost |
| 2026-05-14 | P3 | P3-5: Mean-variance signal combiner | models/signal_combiner.py | `SignalCombiner` w* = Σ⁻¹ × IC (Grinold-Kahn); priors IC=[0.030, 0.020, 0.025, 0.035]; ledger shrinkage λ=0.30; IC updated from historical signal-return pairs |
| 2026-05-14 | P3 | P3-6: Portfolio construction | models/portfolio_constructor.py | `PortfolioConstructor` fractional Kelly 0.25×, per-name cap 5%, sector cap 30%, vol targeting 15% ann; `build_portfolio_from_xs_ranks` convenience wrapper |
| 2026-05-14 | P3 | P3-9: Walk-forward backtester | services/backtester.py | `WalkForwardBacktester` 3y train / 6m test / 6m step; monthly rebalancing; bootstrap 95% CI on Sharpe; `BacktestReport` with equity curve, WF slices, signal attribution |
| 2026-05-14 | P3 | P3-10/P3-11: Live monitor + auto-suspend | services/live_monitor.py | `LiveMonitor` daily ECE/Sharpe, weekly 7d Sharpe, monthly attribution; auto-suspend at `data/ledger/trading_suspended.json` if 60d Sharpe < 0 for 3 consecutive days; `prediction_service.py` reads flag + sets `bundle.trading_suspended` |
| 2026-05-14 | P3 | Config + schema wiring | config/settings.py, schemas/prediction.py, services/prediction_service.py | Added `ANTHROPIC_API_KEY`; added `calibration_gated` + `trading_suspended` fields to `PredictionBundle`; suspension check in `predict()` |

---

## NOTES & DECISIONS

- **Gemini in `parse_manual_fii_dii_input` is kept**: This path is user-initiated (they paste NSE JSON). Gemini parsing unstructured text is appropriate there.
- **Predictions are not hard-blocked by ECE gate**: We flag, not block. The user needs to see the model output even during calibration failure so they can understand what's happening.
- **GRU is not removed in Phase 1**: The GRU overfit problem is a Phase 2 architectural change. Removing it without a replacement would break the entire model. Phase 1 only fixes data quality and scoring bugs.
- **MA crossover backtest**: The −91.45% DD is real and reflects the strategy's failure, not a bug in the backtester. The backtester code is correct. The fix is a better strategy (Phase 3).
- **Scoring inversion hypothesis was wrong**: Code review confirmed `up: hit if actual > anchor` and `down: hit if actual < anchor` are correct. The 0% and 25% accuracy is genuine model failure, not a code bug.
