# Research Paper 8 — The Recalibration Cadence Law

**"How Often Should You Recalibrate? A Closed-Form Cadence Law for Deployed
Probabilistic Forecasters"** (IEEEtran, 6 pp)

The law: for post-recalibration calibration gap `g(t) = γ·t^β`, forecast rate ρ,
recalibration cost c (Brier-equivalent units):

    T* = [ (2β+1)·c / (2β·ρ·γ²) ]^(1/(2β+1))

β=1 → cube-root law `(3c/2γ²)^(1/3)`; β=½ → EOQ square-root law `(2c/γ²)^(1/2)`.

## Reproduce (all numbers in the paper come from these, in order)

```
.venv\Scripts\python.exe analysis\ledger_gamma.py        # γ, β, E0 from live ledger
.venv\Scripts\python.exe analysis\sim_recalibration.py   # E1–E4 Monte Carlo (seed=7)
.venv\Scripts\python.exe analysis\make_figures.py        # fig_*.pdf
pdflatex recalibration_cadence_law.tex                   # run twice
```

- Ledger source: `data/ledger/predictions.sqlite` snapshot 2026-06-12
  (564 resolved interval forecasts, 6 NSE tickers, 2026-04-19 → 2026-06-11).
- Key measured numbers: E0 ≈ 14.8 pp baseline under-coverage at t=1;
  γ̂ = 0.80 ± 0.67 pp/day (coverage channel) vs 0.76 pp/day (ECE telemetry);
  β̂ = 0.23 ± 0.22 (weakly identified — paper reports the family over β).
- Worked cadences at c = 0.01: 6.2 d (cube-root), 2.8 d (affine, E0=14.8pp),
  1.9 d (diffusive).

## Before submission (TODOs left in the .tex)

1. Author list / affiliations (placeholder block; see paper-5/7 for house style).
2. Fill author names for three arXiv citations (2604.06438, 2603.13156,
   2505.00356) — flagged with `%% TODO` comments in the bibliography.
3. Re-run `ledger_gamma.py` against a fresher ledger snapshot before camera-ready;
   the γ confidence interval narrows as the ledger accumulates.
4. Consider an arXiv preprint first to timestamp priority for the law.
