# Colab implementations — 4 research objectives, 4 notebooks

Simplified, fast, real-data implementations of the 4 research objectives in
`research-papers/extras/Title with ROs.txt`, built from the 6 papers in `research-papers/extras/`.
Each notebook is **self-contained** — open it in Google Colab, set the runtime to a T4 GPU
(Runtime → Change runtime type → T4 GPU), and run all cells top to bottom. No local setup, no repo
checkout needed inside Colab.

## Why these exist

The main project in this repo is a full production system (live ledger, calibration pipeline,
multiple services). That is not something you can casually demo in a classroom or a committee
meeting. These 4 notebooks strip each research objective down to something that:

- runs in a few minutes on the free Colab T4 tier,
- uses **only real, freely-fetched data** (no simulated rows, ever — see "Data sources" below),
- reports honest, walk-forward-evaluated numbers rather than cherry-picked ones,
- is small enough to read top-to-bottom before a presentation.

## Objective → paper → notebook mapping

| RO | Objective (short) | Primary paper(s) in `extras/` | Notebook |
|----|---|---|---|
| RO1 | Multi-source data fusion (financial + news + social + macro) | *A Multi-Source Data Fusion Framework for Enhanced Stock Market Prediction* (MSFN) | `01_RO1_multisource_fusion.ipynb` |
| RO2 | Hybrid LLM/sentiment + indicators, optimized DL architecture, real-time | *Multiplicative Information Gating* (LLM polarity/novelty/materiality gating) + *Enhancing Financial Forecasting via Multimodal Learning and Sliding-Window PSO* | `02_RO2_llm_gating_calibrated_forecasting.ipynb` |
| RO3 | Explainable AI, sentiment-noise impact, sentiment↔price temporal dynamics | *A Controlled Experiment on SHAP-Based Explainable AI for Portfolio Rebalancing* + *A Dynamic-Causal Hybrid Framework* (transparency angle) | `03_RO3_explainability_sentiment_dynamics.ipynb` |
| RO4 | Cross-market validation, practical application, benchmark datasets | *From Signal Fusion to Asset Allocation* + *A Dynamic-Causal Hybrid Framework* (regime/sector robustness) + the IJCDS paper's NIFTY-vs-Sensex comparison | `04_RO4_regime_portfolio_crossmarket.ipynb` |

All 6 papers in `extras/` are used somewhere above — none were dropped.

## Paper-specific companion notebook (not one of the 4 ROs)

`tubitak_mig_colab.ipynb` is a different kind of notebook from the 4 above: it is a single-paper code-proof
companion for *Multiplicative information gating* (`research-papers/research-paper-8-tubitak/tubitak_mig.tex`),
built to be shown to judges alongside that paper specifically. Where notebooks 1-4 are compressed,
structurally-comparable demos on freely-fetched data (see "How to read the results" below), this one goes
further where it can: Table 1 (live-ledger interval coverage) and Table 2 (the 30-day conviction gate,
by year) are reproduced from a direct export of this project's own real `data/ledger/predictions.sqlite`
and its own persisted walk-forward training-run statistics (`models/saved/directional_signal.pkl`,
`models/saved/cross_sectional_meta.pkl`) — an **exact, decimal-for-decimal match** to the paper, not an
approximation. The calibration-repair and accuracy-ceiling sections then demonstrate the same mechanisms
live on freshly-pulled real data, honestly caveated where a small live sample can't be expected to match
the full 8-year production figure. See the notebook's own final markdown cell for the full number-by-number
comparison against the paper.

## Tuned to match this project's own production findings

All 4 notebooks predict a **21-trading-day (~30 calendar day) "swing" forward return**, not next-day. This
is deliberate: this project's own production live ledger established that single-name next-period direction
carries no usable edge (≈50.1% out-of-sample, ranking AUC ≈0.47), and that real, gate-able edge shows up
specifically at a ~30-day swing horizon instead. Predicting 1-day or 5-day moves in a demo notebook would
mostly showcase noise — these notebooks target the horizon where there is something real to find, so their
findings are structurally comparable to the production results rather than answering an easier, different
question. Notebook 4's rebalance cadence follows the same horizon for the same reason.

Notebook 2 goes a step further and reproduces the production calibration **structure**, not just the horizon:
a two-stage recalibration (isotonic bias correction, then conformalized interval calibration — the compressed
analogue of the production pipeline's "repair a stacking domain mismatch, then fix coverage" approach), and
a high-conviction operating point reported with a Wilson confidence interval, explicitly benchmarked in-notebook
against the production ledger's own published 58.0%-base / 60.6%-gated / ~94%-abstention result (95% CI
58.8–62.4, n=619–621). In one validation run while building this notebook, the demo's own base rate (58.3%)
and high-conviction rate (62.5%, n=8) landed remarkably close to those production numbers — on a sample this
small that is closer to fortunate than guaranteed, so the notebook explicitly tells you to compare the *shape*
of the hit-rate-vs-abstention curve on your own run, not chase an exact percentage match.

Ticker universes were also widened for larger real cross-sectional samples: Notebook 1 uses 15 NIFTY50 names
(up from 10), Notebook 2 uses ~35 (up from ~20) since it needs enough recent-news-covered rows to make the
high-conviction point meaningful at all.

## Data sources (100% real, nothing synthesized)

| Source | What it provides | How it's fetched |
|---|---|---|
| Yahoo Finance (`yfinance`) | NSE/BSE daily OHLCV, quarterly financials, EPS surprises, recent news headlines | live pull, no key |
| FRED (`fredgraph.csv` endpoint) | USD/INR daily rate, India GDP growth | live pull, no key |
| RBI Monetary Policy Committee decisions | repo rate history | small hand-curated table of public record (rates are accurate; pre-2019 exact announcement *dates* may be off by a few days — negligible at daily resolution, re-verify at rbi.org.in if exact-day precision matters) |
| FinBERT (`ProsusAI/finbert`) | financial sentiment polarity on real headlines | HuggingFace, downloaded at runtime |
| TF-IDF cosine distance (scikit-learn) | "novelty" of a headline vs. the ticker's recent headlines | computed locally, no model download |
| keyword heuristic | "materiality" of a headline | computed locally |

**GDELT was deliberately not used** — it rate-limits to roughly 1 request/5s and returned HTTP 429 even
below that rate during testing, which is incompatible with "runs quickly and reliably." Yahoo Finance's
own `.news` endpoint is the sole news source, which means **real sentiment coverage is limited to a
recent window (typically the trailing few weeks) per ticker** — this is a genuine constraint of free
data, and every notebook says so explicitly wherever it matters, rather than silently padding history
with fabricated "neutral" sentiment.

## How to read the results (please don't skip this)

Every notebook prints honest metrics against naive baselines (zero-return, mean-return, equal-weight
buy-and-hold, index buy-and-hold) and explicitly flags small-sample sections. In local validation runs
while building these notebooks:

- Notebook 1's 3-source fusion model did **not** beat naive baselines on a small ticker subset — a real,
  disclosed finding, consistent with this project's broader experience that single-name short-horizon
  direction rarely beats a baseline by much.
- Notebook 2's two-stage-recalibrated, high-conviction hit rate did land above its own no-gate baseline in
  validation (58.3% base → 62.5% at ~22% coverage, n=8, wide CI) — a small-sample echo of the production
  gating result, reported with its confidence interval attached rather than as a bare, overconfident number.
- Notebook 4's regime-conditional strategy underperformed simple buy-and-hold on NIFTY, and roughly matched
  it on Sensex, in the same validation run — a genuine cross-market disagreement, reported as-is rather than
  smoothed over, because that disagreement (or agreement, on your own rerun) *is* the honest answer RO4 asks
  for.

Do not "fix" a notebook to make its number look better before a demo — if a rerun produces a weak or
negative result, that is the correct thing to present, with the same honesty this project's own research
papers insist on. Present R² / MAE / calibration / Sharpe alongside directional accuracy; a directional
accuracy near the base up-move rate is not edge.

## Practical notes for presenting

- Each notebook ends with a "Summary for a presentation slide" markdown cell — pull numbers from there,
  not from this README or from the original papers' own claimed results.
- Universe sizes and dates are variables near the top of each notebook (`TICKERS`, `START`, `TRAIN_END`,
  `VAL_END`) if you want to widen/narrow scope.
- Re-running months later will pick up new price/news data automatically; the RBI repo-rate table and
  FRED series may need a manual check if the notebook is run much later than mid-2026.
