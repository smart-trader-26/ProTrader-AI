#!/usr/bin/env bash
# Build the IJCISIM manuscript.
#
#   ./build_paper.sh
#
# Produces, next to this script:
#   ijcisim_validity_screen.docx   <- the file to upload
#   ijcisim_validity_screen.pdf    <- the same file as Word renders it
#
# IJCISIM's submission checklist asks for a Microsoft Word file, so the .docx is
# the deliverable and the .pdf is only a rendering of it for review.  Word does
# the conversion, so the PDF is what an editor opening the .docx would see; that
# step needs Word installed and is skipped automatically if it is not.
#
# The analysis pipeline is unchanged from the study in
# ../research-paper-17-fininnov-not-worth (steps 01-09 and the scoring cache).
# Only the two manuscript-side steps are re-run here:
#
#   10_cost_accounting.py   counts the fitted objects for Table 4
#   06b_tables_json.py      re-derives every table and every in-prose number
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PY="c:/Users/divya/Desktop/finance/.venv/Scripts/python.exe"
CACHE="${CACHE:-../research-paper-17-fininnov-not-worth/cache}"
STEM=ijcisim_validity_screen
export PYTHONIOENCODING=utf-8

echo "== re-deriving the numbers from results/ =="
"$PY" analysis/10_cost_accounting.py --no-latex > /dev/null
"$PY" analysis/06b_tables_json.py --cache "$CACHE"

echo
echo "== assembling the manuscript =="
"$PY" analysis/make_paper_ijcisim.py --out "$STEM.docx"

echo
echo "== rendering to PDF through Word =="
"$PY" -c "
import sys
sys.path.insert(0, 'analysis')
try:
    from ijcisim_docbuild import to_pdf
    to_pdf('$STEM.docx', '$STEM.pdf')
except Exception as exc:                       # Word not installed, or busy
    print('PDF step skipped:', exc)
"
"$PY" analysis/verify_manuscript.py
echo "ALL DONE"
