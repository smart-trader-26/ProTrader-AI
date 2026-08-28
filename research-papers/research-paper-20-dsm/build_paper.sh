#!/usr/bin/env bash
# Build both Data Science and Management manuscript files.
#
#   ./build_paper.sh
#
# Produces, next to this script:
#   dsm_validity_screen_BLIND.pdf     <- the file to upload (DSM is double blind)
#   dsm_validity_screen_authors.pdf   <- same paper, names shown, for the record
#
# The analysis pipeline is unchanged from the study in
# ../research-paper-17-fininnov-not-worth (steps 01-09 and the scoring cache);
# only step 10, the cost accounting, is new here.  Re-running it is optional and
# touches no network:
#
#   python analysis/10_cost_accounting.py
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

SRC=dsm_validity_screen

build () {                       # $1 = output stem, $2 = \blindtrue|\blindfalse
  local stem="$1" flag="$2"
  printf '%% Written by build_paper.sh -- do not edit by hand.\n%s\n' "$flag" > blindflag.tex
  cp "$SRC.tex" "$stem.tex"
  pdflatex -interaction=nonstopmode "$stem" > /dev/null || true
  bibtex "$stem" > /dev/null || true
  pdflatex -interaction=nonstopmode "$stem" > /dev/null || true
  pdflatex -interaction=nonstopmode "$stem" > /dev/null || true
  echo "== $stem.pdf =="
  grep -c "Warning" "$stem.log" || true
  grep -nE "^! |Undefined control sequence|Citation .* undefined|Reference .* undefined" "$stem.log" | head -20 || true
}

echo "== regenerating the cost table from the stored results =="
python analysis/10_cost_accounting.py > /dev/null

build dsm_validity_screen_BLIND   '\blindtrue'
build dsm_validity_screen_authors '\blindfalse'

# leave the tree in the blinded state, which is what gets submitted
printf '%% Written by build_paper.sh -- do not edit by hand.\n\\blindtrue\n' > blindflag.tex
python analysis/verify_manuscript.py
echo "ALL DONE"
