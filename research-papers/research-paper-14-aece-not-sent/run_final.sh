#!/usr/bin/env bash
# Final pass: merge shard caches -> panel -> analysis x2 -> figures -> paper.
set -euo pipefail
W="C:/Users/divya/AppData/Local/Temp/claude/c--Users-divya-Desktop-finance/46e0220a-4c07-4d85-a952-1dd92b27aa0d/scratchpad/work"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="c:/Users/divya/Desktop/finance/.venv/Scripts/python.exe"
RES="$HERE/results"
FIG="$HERE/figures"
export PYTHONIOENCODING=utf-8

echo "== merge sentiment caches =="
# --skip-finbert: m1 always comes from the GPU pass (m1_gpu.parquet); the
# merge step's NaN check aborts if any headline still lacks a score, so
# this can never silently fall back to hours of CPU inference.
"$PY" "$HERE/analysis/02_score_sentiment.py" --indir "$W" --skip-finbert
echo "== panel =="
rm -f "$W/panel.parquet"
"$PY" "$HERE/analysis/03_build_panel.py" --indir "$W"
cp "$W/dataset.json" "$RES/dataset.json"

for T in ret_adj lrv_innov; do
  echo "== kernel ($T) =="
  "$PY" "$HERE/analysis/04_kernel.py" --indir "$W" --outdir "$RES" \
        --K 12 --boot 200 --target "$T"
  echo "== filters ($T) =="
  "$PY" "$HERE/analysis/05_matched_filter.py" --indir "$W" --outdir "$RES" \
        --K 12 --horizons 1,5 --target "$T"
  echo "== closed loop ($T) =="
  "$PY" "$HERE/analysis/06_closed_loop.py" --indir "$W" --outdir "$RES" \
        --K 12 --target "$T"
done

echo "== figures =="
"$PY" "$HERE/analysis/07_figures.py" --outdir "$RES" --figdir "$FIG"
echo "== manuscript =="
"$PY" "$HERE/analysis/make_paper.py" \
      --template "$HERE/templates/AECE_template.docx" \
      --results "$RES" --figdir "$FIG" \
      --out "$HERE/aece_matched_filter.docx"
echo "ALL DONE"
