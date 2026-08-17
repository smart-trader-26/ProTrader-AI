"""Check the reference list against JETIA's four distribution rules.

  * at least 50 per cent published within the last five years;
  * at least 70 per cent international relative to the corresponding
    author's country (India for this manuscript);
  * at least 40 per cent available online;
  * fewer than 10 per cent citations to articles published in JETIA.

Run against the tag order the manuscript actually cited, which
make_paper_jetia.py writes next to the .docx as <name>_cited.json:

    python analysis/ref_stats.py --cited jetia_matched_filter_cited.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

from refs import REFS, YEARS                              # noqa: E402

THIS_YEAR = 2026
WINDOW = 5                        # "the last five years" -> 2021 and later

# Entries whose venue or whose corresponding author is based in India.  The
# rule is about references that are international relative to the
# corresponding author's country, so these are the ones that do not count.
DOMESTIC = set()                  # none: every venue cited is international


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cited", default=None,
                    help="JSON list of cited tags; defaults to all entries")
    args = ap.parse_args()

    if args.cited and os.path.exists(args.cited):
        tags = json.load(open(args.cited))
    else:
        tags = sorted(REFS)
        print("(no --cited file given; checking every entry in refs.py)")

    n = len(tags)
    missing_year = [t for t in tags if t not in YEARS]
    recent = [t for t in tags if YEARS.get(t, 0) >= THIS_YEAR - WINDOW]
    online = [t for t in tags if re.search(r"doi:\S+", REFS.get(t, ""))]
    jetia = [t for t in tags if "ITEGAM-JETIA" in REFS.get(t, "")]
    intl = [t for t in tags if t not in DOMESTIC]

    def line(label, got, need, ok_if_below=False):
        pct = 100.0 * got / n if n else 0.0
        ok = (pct < need) if ok_if_below else (pct >= need)
        rel = "<" if ok_if_below else ">="
        print(f"{'OK ' if ok else '!! '}{label}: {got}/{n} = {pct:.1f}% "
              f"(rule: {rel} {need}%)")
        return ok

    print(f"references cited: {n}")
    if missing_year:
        print(f"!! no year recorded for: {missing_year}")
    good = [
        line(f"published {THIS_YEAR - WINDOW} or later", len(recent), 50),
        line("international", len(intl), 70),
        line("available online (has a DOI)", len(online), 40),
        line("published in JETIA", len(jetia), 10, ok_if_below=True),
    ]
    print(f"\noldest entry: {min(YEARS.get(t, 0) for t in tags)}; "
          f"newest entry: {max(YEARS.get(t, 0) for t in tags)}")
    print("ALL RULES SATISFIED" if all(good) and not missing_year
          else "SOME RULES NOT SATISFIED")
    sys.exit(0 if all(good) and not missing_year else 1)


if __name__ == "__main__":
    main()
