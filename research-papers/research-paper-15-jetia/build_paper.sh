#!/usr/bin/env bash
# Build both JETIA manuscript files from the result artefacts in results/.
#
#   ./build_paper.sh
#
# Produces, next to this script:
#   jetia_matched_filter_BLIND.docx  + .pdf   <- upload the .docx
#   jetia_matched_filter_authors.docx + .pdf  <- same paper, names shown
#
# The manuscript is delivered as .docx.  JETIA asks for "Microsoft Word
# format" and ships a .docx template, and .docx keeps the equations as real
# OMML objects; the legacy .doc round trip degrades them.
#
# The analysis pipeline itself is unchanged from the earlier version of this
# study and lives in ../research-paper-14-aece-not-sent (steps 1-7 and the
# data cache).  Only the manuscript step is duplicated here.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
PY="c:/Users/divya/Desktop/finance/.venv/Scripts/python.exe"
export PYTHONIOENCODING=utf-8

build () {                       # $1 = output stem, $2... = extra flags
  local stem="$1"; shift
  "$PY" "$HERE/analysis/make_paper_jetia.py" \
        --template "$HERE/templates/JETIA_Template.docx" \
        --results  "$HERE/results" \
        --figdir   "$HERE/figures" \
        --out      "$HERE/$stem.docx" \
        --figwidth 4.0 "$@"
  "$PY" -c "
import sys, os
here = r'$HERE'
sys.path.insert(0, os.path.join(here, 'analysis'))
from jetia_docbuild import to_pdf
to_pdf(os.path.join(here, '$stem.docx'), os.path.join(here, '$stem.pdf'))
"
}

echo "== blinded manuscript (the file to submit) =="
build jetia_matched_filter_BLIND --blind
echo
echo "== author-identified manuscript (for the record) =="
build jetia_matched_filter_authors
echo
echo "== reference distribution =="
"$PY" "$HERE/analysis/ref_stats.py" \
      --cited "$HERE/jetia_matched_filter_BLIND_cited.json"
echo "ALL DONE"
