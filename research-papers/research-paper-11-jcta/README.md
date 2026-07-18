# Research Paper 11 — JCTA

**Target journal:** Journal of Computing Theories and Applications (JCTA), Univ. Dian
Nuswantoro, Indonesia. Scopus Q3 / SINTA 2. Free APC (for international authors).
Single Word/A4 document, IEEE references, ORCID for corresponding author, AI-use
statement in acknowledgements, max 25% similarity. Research articles 12–20 pp.

**Title:** *Confidence Is Not Coverage: Magnitude-Dependent Miscoverage and Conformal
Recalibration of Prediction Intervals in Deployed Equity Forecasting.*

**Scope fit:** Machine Learning / Intelligent Systems / Data Mining (an empirical
uncertainty-quantification + conformal-prediction study on a deployed ML system).

**Core contribution (distinct from all prior papers):**
- Studies the *coverage of the 90% price intervals* of the deployed system on 619
  resolved, point-in-time ledger forecasts (object = prediction intervals, not class
  probabilities).
- New finding: the **confidence–coverage inversion** — coverage is flat across interval
  width (0.68–0.73) but collapses with forecast magnitude (95.7% → 47.1%); the band
  fails to widen for confident forecasts (corr(width,|f|)=0.27).
- A minimal split-conformal multiplicative width recalibration restores coverage
  70.6% → 94.4% walk-forward (Winkler 0.218 → 0.185); an honest negative result on
  magnitude-conditioning (right-signed multipliers 0.89/1.40/2.02 but not yet warranted
  by the sample).

**Data:** `data/ledger/predictions.sqlite` (619 resolved forecasts, 2026-04-19 to
2026-06-12, 6 NSE names). All figures in `figures/` are generated directly from it;
`figures/summary*.json` hold the computed numbers.

**Build:** `pdflatex jcta_interval_reliability.tex` (×2). 13 pages.

**Files:** `jcta_interval_reliability.tex` (source), `.pdf` (compiled),
`figures/` (vector PDFs + numeric summaries), `templates/` (JCTA Word template).

> Distinct from Paper 12 (ECTI-CIT), which studies the *point forecast's efficiency*
> (Mincer–Zarnowitz + shrinkage). Same deployed system/ledger; different statistical
> object, method, and literature.
