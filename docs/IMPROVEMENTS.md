# ProTrader AI — Improvements & Roadmap

_Last updated: 2026-04-17_

This document captures (1) the comparison between the current sentiment pipeline
and the standalone `sentiment_analysis_v2.py` Colab notebook, (2) what was
integrated, (3) what was deliberately **not** integrated and why, and (4) a
prioritised roadmap of further improvements that would meaningfully move the
needle in real-time live trading.

---

## 1. Sentiment Engine: Current vs. v2 Colab

### Current implementation (production, in this repo)

| File                          | Role                                              |
|-------------------------------|---------------------------------------------------|
| `data/news_sentiment.py`      | FinBERT pipeline + Google News / ET RSS / NewsAPI |
| `data/multi_sentiment.py`     | 4-source aggregator (RSS / NewsAPI / Reddit / GT) |
| `models/sentiment_expert.py`  | Dense NN that turns sentiment features → return   |

**Strengths**

- 4 independent sources with auto-reweighting when one is unavailable.
- Temporal decay (λ = 0.5, exponential).
- 5-bucket event classifier (`earnings`, `regulatory`, `dividend`,
  `management`, `general`) with per-event multiplier weights.
- Source-disagreement signal (std of 4 source scores) drives a confidence
  penalty up to 40% — protects against echo-chamber false signals.
- Streamlit-native caching (`@st.cache_data`).

**Gaps that v2 highlighted**

- Raw FinBERT output without keyword overrides leaves easy precision on
  the table for unambiguous signals like "Wipro Q3 misses estimates".
- Only 5 event categories; v2's 6-category schema separates
  `Analyst Ratings`, `Market Action`, and `Deals & Acquisitions` which
  the original lumped into `general`.

### v2 implementation (`sentiment_analysis_v2.py`, Colab)

**Strengths**

- BART zero-shot 6-category classifier (vs keyword matching).
- Bullish/bearish keyword overrides → confidence boost to 0.88 on
  clear domain signals.
- Per-category sentiment encoding (no mode-collapse on multi-event days).
- Time-decay λ = 0.1 over a 5-day lookback (smoother than current λ = 0.5
  same-day-only).
- Walk-forward validation + LR/RF/XGB/LGBM stacking benchmark.

**Why we cannot lift it wholesale into Streamlit Cloud**

- Colab-specific: `google.colab`, IPython widgets, `!pip install`,
  `display`, `clear_output`.
- BART-large-mnli is 1.6 GB and ~5–8 s per query on CPU — Streamlit Cloud
  has no GPU and ~1 GB RAM headroom.
- Mistral-7B-4bit benchmark requires `bitsandbytes` + CUDA.
- GNews / Hindi translation rate-limit faster than RSS feeds.
- No multi-source aggregation — single source is more fragile than the
  current 4-source mix.

### Verdict

**Hybrid wins.** Current architecture stays as the backbone; we lift the two
highest-ROI ideas from v2:

1. **Bullish/Bearish keyword overrides** ([data/news_sentiment.py:54-80](../data/news_sentiment.py#L54-L80))
   — applied inside `analyze_sentiment()` and inherited by
   `MultiSourceSentiment.analyze_text()` ([data/multi_sentiment.py:228-247](../data/multi_sentiment.py#L228-L247)).
2. **6-category schema + lightweight keyword classifier** —
   `categorize_headline()` and `VALID_CATEGORIES` exposed for downstream use
   without a 1.6 GB BART download.

---

## 2. What was integrated (this commit)

| Change | File | Risk |
|--------|------|------|
| Bullish/Bearish keyword overrides → boost confidence to ≥0.88 | [data/news_sentiment.py](../data/news_sentiment.py) | Low — additive |
| 6-category labelling helper `categorize_headline()` | [data/news_sentiment.py](../data/news_sentiment.py) | Low — new function |
| `MultiSourceSentiment.analyze_text` delegates to enhanced engine | [data/multi_sentiment.py](../data/multi_sentiment.py) | Low — same return type |
| `st.secrets` → `.env` → OS-env resolver for keys | [config/settings.py](../config/settings.py) | Low — backwards compatible |
| Pinned `requirements.txt` for Streamlit Cloud (Python 3.11, tf-cpu) | [requirements.txt](../requirements.txt) | Medium — version pins |
| `.streamlit/config.toml`, `secrets.toml.example`, `runtime.txt`, `packages.txt` | new files | Low |

---

## 3. What was deliberately NOT integrated

| v2 feature | Why skipped |
|-----------|-------------|
| BART-large zero-shot categorisation | 1.6 GB model, ~6 s/query on CPU. Replaced with keyword fallback for free Streamlit tier. |
| Mistral-7B benchmark | 4 GB even at 4-bit, needs CUDA + bitsandbytes. |
| GNews + Hindi translation | Rate-limited; current RSS aggregator is more robust. |
| Walk-forward + LR/RF/XGB/LGBM stacking | The hybrid model in `models/hybrid_model.py` already stacks XGB + LGBM + CatBoost + GRU with a Ridge meta-learner; duplicating in the sentiment expert adds complexity without a clear win. |
| Per-category 12-feature vector | Would force a refactor of `SentimentExpertModel` (currently 8-feature). Tracked in roadmap §4.B. |

---

## 4. Roadmap — what would actually help live trading

Ordered by **expected accuracy gain ÷ implementation effort**.

### A. Quick wins (≤ 1 day each)

1. **Cache FinBERT model artefacts on Streamlit Cloud.**
   First-run cold-start downloads ~440 MB → ~30 s delay. Wrap
   `_get_sentiment_pipeline()` in `@st.cache_resource`. _Files:_
   `data/news_sentiment.py`.

2. **Increase RSS feed coverage** for sentiment recall. Add Bloomberg
   Quint, Mint markets, NDTV Profit RSS to `RSS_FEEDS`. Each new feed
   raises `article_count` by 10–25% on average tickers.

3. **Show last sentiment update timestamp** in the Multi-Source Sentiment
   tab. Users currently can't tell if data is fresh or 30 min stale.

4. **Persist FII/DII history.** A small SQLite table avoids re-hitting
   NSE on every rerun and survives between user sessions.

### B. Mid-term (1–3 days)

5. **Per-category sentiment vector for the Sentiment Expert.** Rebuild
   the dense NN with 12-feature input (6 sentiment + 6 count) so it can
   learn that Earnings sentiment ≠ Macro sentiment ≠ Analyst-rating
   sentiment. v2 showed a measurable F1 lift from this change.

6. **Intraday refresh loop.** Add an auto-refresh widget
   (`st_autorefresh`) on the dashboard tab that re-fetches news every
   5 min during NSE hours (09:15–15:30 IST). Currently the user must
   click "Launch Analysis" manually.

7. **Volume-confirmation overlay on signals.** Bullish forecast +
   above-average volume = high-conviction signal; below-average volume =
   ignore. One filter line in `app.py` after the model output.

8. **Sector beta hedge calculator.** Given a long Reliance signal, suggest
   a Nifty50 short to neutralise sector exposure. Pure math against
   `df_stock['Close'].pct_change().corr(nifty_returns)`.

### C. Long-term (1+ week)

9. **Replace keyword categoriser with DistilBERT fine-tuned on Indian
   financial headlines.** Train once on the 30-headline benchmark from
   v2 + 200 hand-labelled additions. Distill to 67M params (~250 MB)
   so it still runs on Streamlit Cloud CPU.

10. **Live order book / depth integration.** Either via Zerodha Kite
    Connect (paid) or Alice Blue (free). Current system is purely
    end-of-day; bid-ask imbalance is the highest-signal short-horizon
    feature in equities.

11. **Walk-forward sentiment-only model from v2 Colab as a separate tab.**
    Train it offline on the master CSV, ship the saved `.pkl` files in
    `models/saved/`, and load with `joblib.load()` for a sub-second
    sentiment-only "second opinion" panel.

12. **WebSocket price feed** (yfinance is REST-polled; latency is
    20–60 s). Switch to a streaming source for true real-time.

---

## 5. Known issues to fix before "live"

- `app.py:24` imports `analyze_sentiment` from `news_sentiment` — still
  works, but the function now applies keyword overrides which slightly
  shifts the historical baseline. Document in the changelog.
- `tensorflow` was unpinned and pulled the GPU build on cloud — now
  pinned to `tensorflow-cpu==2.15–2.16` in `requirements.txt`.
- `roboflow` import in `utils/roboflow_client.py` is optional but its
  default API key is hard-coded in `settings.py`. Move to `st.secrets`
  with a clear "feature disabled if missing" UI message.
- The `=1.0.0` and `=4.0.0` files in the repo root (29 KB and 0 B) look
  like accidental `pip install` redirect artefacts. Safe to delete.

---

## 6. How to run live

1. **Local:** see `docs/DEPLOYMENT.md` § "Run locally".
2. **Streamlit Cloud:** see `docs/DEPLOYMENT.md` § "Deploy to Streamlit
   Community Cloud".

After deploy, the highest-leverage thing to verify is:

- `Launch Analysis` on RELIANCE returns within 90 s.
- Multi-Source Sentiment tab shows ≥1 source available.
- Backtest tab loads without `tensorflow` errors.

If any of those fail, the cloud Python version is wrong — set it to
**3.11** in the Streamlit Cloud advanced settings.
