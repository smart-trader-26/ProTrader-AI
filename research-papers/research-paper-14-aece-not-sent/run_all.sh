#!/usr/bin/env bash
# Full pipeline, from the raw corpus to the submitted manuscript.
#   bash run_all.sh <work_dir> <fnspid_csv>
set -euo pipefail

WORK="${1:?work dir}"
SRC="${2:?path to FNSPID All_external.csv}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="c:/Users/divya/Desktop/finance/.venv/Scripts/python.exe"
RES="$HERE/results"
FIG="$HERE/figures"
export PYTHONIOENCODING=utf-8

mkdir -p "$RES" "$FIG"

echo "== 1. corpus =="
"$PY" "$HERE/analysis/01_build_corpus.py" --src "$SRC" --outdir "$WORK" --top 300
echo "== 2. sentiment =="
"$PY" "$HERE/analysis/02_score_sentiment.py" --indir "$WORK" --batch 128
echo "== 3. panel =="
"$PY" "$HERE/analysis/03_build_panel.py" --indir "$WORK"
cp "$WORK/dataset.json" "$RES/dataset.json"

for T in ret_adj lrv_innov; do
  echo "== 4. kernel ($T) =="
  "$PY" "$HERE/analysis/04_kernel.py" --indir "$WORK" --outdir "$RES" \
        --K 12 --boot 200 --target "$T"
  echo "== 5. filters ($T) =="
  "$PY" "$HERE/analysis/05_matched_filter.py" --indir "$WORK" --outdir "$RES" \
        --K 12 --horizons 1,5 --target "$T"
  echo "== 6. closed loop ($T) =="
  "$PY" "$HERE/analysis/06_closed_loop.py" --indir "$WORK" --outdir "$RES" \
        --K 12 --target "$T"
done

echo "== 7. figures =="
"$PY" "$HERE/analysis/07_figures.py" --outdir "$RES" --figdir "$FIG"

echo "== 8. manuscript =="
"$PY" "$HERE/analysis/make_paper.py" \
      --template "$HERE/templates/AECE_template.docx" \
      --results "$RES" --figdir "$FIG" \
      --out "$HERE/aece_matched_filter.docx"
echo "done"
