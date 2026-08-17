# Paper 14 — AECE

**An Attenuation-Corrected Matched Filter with Closed-Loop Gain Design for
News-Sentiment Aggregation in Equity Forecasting**

Target: *Advances in Electrical and Computer Engineering* (aece.ro).
SCIE + Scopus, JCR IF 0.900, CiteScore 2.2, acceptance rate 19 %,
APC 300 EUR (+25 EUR per page beyond 8), double-blind, ~95 day review.

## What the paper claims

The step that turns a stream of headlines into one number per instrument per
session is normally a uniform average over a window of 1/3/7 days, chosen by
grid search. This paper treats it as a filter design problem.

1. **Identification.** Sentiment scorers are noisy indicators of one latent
   quantity. The cross-covariance of two *independent* scorers estimates the
   latent autocovariance at every pair of lags, because the errors of two
   different scorers are uncorrelated whereas a scorer's own autocovariance is
   contaminated at the diagonal. This identifies the noise-free response
   kernel; three scorers additionally identify each scorer's reliability
   ratio, and a fourth over-identifies the model so the assumption is testable.
2. **Design.** The minimum-MSE aggregation filter is the noise-whitened
   matched filter for that kernel. Its *width* is set by scorer reliability,
   not by how long news actually matters — which is why multi-day averaging
   appears to help. Reliability rises with headline count within a session,
   giving a natural experiment with no analyst intervention.
3. **Loop.** Deployed as an LMS feedback loop, measurement noise tightens the
   mean-square stability bound to `2λ/tr(R_SS)` and makes the error-minimising
   gain scale as `sqrt(λ)`. Both are computed from training data and then
   tested out of sample against the empirically optimal gain.
4. **Evidence.** The same machinery is applied to two responses —
   next-session market-adjusted return and next-session Parkinson range
   volatility — and recovers a short kernel for the first and a persistent one
   for the second.

## Data (all real, nothing synthetic or augmented)

- **News**: FNSPID `Stock_news/All_external.csv`
  (<https://huggingface.co/datasets/Zihan1004/FNSPID>), 13.06 M rows, of which
  3.25 M carry a ticker. The 300 most densely covered symbols are kept →
  **807,784 real headlines**, 2010–2020.
- **Prices**: Yahoo Finance daily bars, split/dividend adjusted, incl. high/low
  for the Parkinson volatility proxy. Market factor SPY, betas fitted on
  training data only.
- **Scorers**: FinBERT (`ProsusAI/finbert`), Loughran–McDonald word lists,
  VADER, Harvard General Inquirer IV-4. Four families, no shared training
  corpus or vocabulary construction.

### Timing convention

Every headline carries a UTC stamp, converted to `America/New_York` and
assigned to the first session whose 16:00 close follows it. A headline
published after the close belongs to the next session. Session sentiment
therefore contains only pre-close information and predicts the response
realised afterwards.

The LMS loop applies **session-delayed updates**: the label for session `t`
is only observable at the close of `t+1`, so weights used to predict a
session absorb observations up to the previous session only. Updating within
a session would feed a not-yet-observable return into predictions for the
other instruments of the same session. (An earlier version of the loop did
exactly that and reported roughly triple the true IC.)

## Pipeline

```
bash run_all.sh <work_dir> <path/to/All_external.csv>
```

| step | script | output |
|---|---|---|
| 1 | `01_build_corpus.py` | `corpus.parquet`, symbol counts |
| 2 | `02_score_sentiment.py` | `headline_scores.parquet` (incremental cache) |
| 3 | `03_build_panel.py` | `panel.parquet`, `dataset.json` |
| 4 | `04_kernel.py` | `kernel*.json`, `R_SS*.npy` |
| 5 | `05_matched_filter.py` | `filter_comparison*.csv` |
| 6 | `06_closed_loop.py` | `closed_loop*.json`, gain sweep |
| 7 | `07_figures.py` | `figures/*.png` |
| 8 | `make_paper.py` | `aece_matched_filter.docx` → `.doc` |

Steps 4–6 run twice, `--target ret_adj` and `--target lrv_innov`.

## Evaluation protocol

- Train 2010 → 2016-12-31, test 2017 → 2020. Nothing after the split
  influences the kernel, the smoothness penalty, the scaling regression, the
  standardisation statistics or the loop gain.
- Baseline is the fixed window with the best **training** performance, since a
  practitioner cannot select a window on test data. The window that happens to
  win on test data is also reported, as an upper bound nobody could pick in
  advance.
- Count-weighted windows are included as the stronger baseline (a session with
  20 headlines should not count the same as one with 1).
- Filters are scored through their training-fitted forecast so that a filter
  with negative taps (as volatility gives) is not penalised for its sign.
- Newey–West t statistics; Diebold–Mariano for equal squared error; cluster
  bootstrap over symbols for kernel intervals.

## References

`analysis/refs.py` holds all 40 entries. `analysis/verify_refs.py` re-checks
every one field-by-field (title, first author, year, volume, issue, pages)
against the CrossRef API. Round one caught a DOI that resolved to a
*different* FinBERT paper, one that resolved to a book review rather than the
book, three AECE self-citations with wrong authors/pages, and a dead DOI.
Re-run it after any edit:

```
python analysis/verify_refs.py
```

## Journal compliance checklist

- [x] Microsoft Word `.doc` (converted from `.docx` via Word COM)
- [x] Even page count: 8, 10 or 12
- [x] Abstract ≤ 200 words, no citations, plain English characters
- [x] Exactly 5 index terms, alphabetical, **all from AECE's published
      keyword list**: adaptive filters, least mean squares methods, matched
      filters, sentiment analysis, stock markets
- [x] ≥ 25 references, permanent DOI links on all
- [x] Equations numbered right-aligned, one formula per line, real Word
      (OMML) equations rather than images
- [x] Table captions above, figure captions below, ≥ 8 pt, grayscale-safe
- [x] Conflict of Interest and Publisher's Note sections present
- [x] Template instruction box removed
