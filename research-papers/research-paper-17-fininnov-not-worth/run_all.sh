#!/usr/bin/env bash
# Full reproduction, in order.  Every step is resumable and every step writes its
# output to results/ or cache/, so a re-run costs only the steps whose inputs
# changed.  Step 1 is the only one that touches the network.
set -euo pipefail

PY=../../.venv/Scripts/python.exe
cd "$(dirname "$0")"

echo "== 1. score the three axes (resumable; skips anything already cached) =="
$PY analysis/01_score_axes.py --symbols 60 --batch 250 --workers 4 --gap 2 \
    --model gemini-3.1-flash-lite

echo "== 1b. second scoring pass on a subsample, for test-retest reliability =="
$PY analysis/01_score_axes.py --retest 2000 --batch 250 --workers 4 \
    --model gemini-3.1-flash-lite

echo "== 2. build the symbol-session panel =="
$PY analysis/02_build_mig_panel.py

echo "== 3. predictive regressions, horse race, free exponents =="
$PY analysis/03_horse_race.py --exponents

echo "== 4. selective forecasting, with and without A =="
$PY analysis/04_selective_gate.py --horizons 1,5,21

echo "== 5. reliability of the measurement instrument =="
$PY analysis/07_reliability.py

echo "== 6. robustness sweeps =="
$PY analysis/08_robustness.py

echo "== 7. figures and tables =="
$PY analysis/05_figures.py
$PY analysis/06_tables.py

echo "== 8. references and manuscript =="
$PY analysis/verify_refs.py || true
pdflatex -interaction=nonstopmode fininnov_mig.tex
bibtex fininnov_mig
pdflatex -interaction=nonstopmode fininnov_mig.tex
pdflatex -interaction=nonstopmode fininnov_mig.tex

echo "done: fininnov_mig.pdf"
