# Research Paper 12 — ECTI-CIT

**Target journal:** ECTI Transactions on Computer and Information Technology (ECTI-CIT),
Thailand. Scopus, CiteScore ~1.6. **Free APC.** Two-column `eec_style` class, 10pt,
8–12 pages, double-blind review (suggest 3 reviewers on submission). The `eec_style.cls`
and template are bundled here.

**Title:** *When the Forecast Loses to No-Change: Point-Forecast Inefficiency and a
Shrink-to-Drift Correction in a Deployed Equity-Return Model.*

**Scope fit:** AI / Machine Learning + Data Science (forecast evaluation, model
auditing of a deployed system).

**Core contribution (distinct from all prior papers and from Paper 11):**
- Studies the *efficiency of the point forecast* (object = the median return forecast,
  not its interval) via a **Mincer–Zarnowitz** regression on 619 ledger forecasts.
- New finding: slope **b = −0.23** (bootstrap 95% CI [−0.35, −0.12]) — excludes both 1
  (inefficient) and 0 (perversely signed); over-extrapolation so severe the bold
  forecasts reverse (Theil U₂ = 1.30 > 1: worse than no-change; directional accuracy
  46.4%).
- Repair: the error-minimizing **shrinkage k\*** is statistically indistinguishable
  from 0 (CI [−0.33, 0.09]) → discard the forecast's direction, predict the drift.
  Out-of-sample RMSE 0.053 → 0.031 (66% MSE reduction); Diebold–Mariano: drift beats
  raw (6.4), drift ≈ optimal shrinkage (1.8, n.s.). Honest ticker-jackknife shows the
  inefficiency is concentrated in some names (RELIANCE/TCS).

**Data:** `data/ledger/predictions.sqlite`. Figures in `figures/` generated from it;
`figures/summary_fc.json` holds the computed numbers.

**Build:** `pdflatex ecti_forecast_efficiency.tex` (×2). Needs `eec_style.cls` and
`a1.eps` (both present). 8 pages.

**Note:** Author block is included per request; for the double-blind review copy, remove
the author block and affiliation footnotes (marked with a comment in the source).

> Distinct from Paper 11 (JCTA), which studies the *interval coverage* (conformal
> recalibration). Same deployed system/ledger; different statistical object, method,
> and literature.
