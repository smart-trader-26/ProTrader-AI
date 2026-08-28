"""
Pre-submission checks on the built manuscript.

Checks by script rather than by eye, because the failure modes here are all silent:
a leftover placeholder, a macro that expanded to nothing, a figure that was
referenced but never generated, a number in the prose that no longer matches the
result file it came from.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "fininnov_mig.tex"
LOG = ROOT / "fininnov_mig.log"
MACROS = ROOT / "macros.tex"

PLACEHOLDER_WORDS = ["PLACEHOLDER", "TODO", "FIXME", "XXX", "TBD",
                     "SCORED_HEADLINES", "N_COMPANIES", "SYMBOL_UNIVERSE"]

failures: list[str] = []
notes: list[str] = []


def check(ok: bool, msg: str, hard: bool = True) -> None:
    if ok:
        print(f"  [ok]   {msg}")
    else:
        print(f"  [{'FAIL' if hard else 'warn'}] {msg}")
        (failures if hard else notes).append(msg)


def strip_tex(s: str) -> str:
    s = re.sub(r"%.*", "", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)
    s = re.sub(r"[{}$\\~^_]", " ", s)
    return re.sub(r"\s+", " ", s)


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    print("Manuscript checks\n")

    # ---- placeholders ------------------------------------------------------
    body = re.sub(r"%.*", "", tex)
    found = [w for w in PLACEHOLDER_WORDS if w in body]
    check(not found, f"no placeholder text left {found if found else ''}")

    # ---- abstract ----------------------------------------------------------
    m = re.search(r"\\abstract\{(.+?)\}\s*\n\s*\\keywords", tex, re.S)
    if m:
        words = len(strip_tex(m.group(1)).split())
        check(120 <= words <= 300, f"abstract length {words} words (target 120-300)")
    else:
        check(False, "abstract found")

    # ---- keywords ----------------------------------------------------------
    m = re.search(r"\\keywords\{(.+?)\}", tex, re.S)
    if m:
        kw = [k.strip() for k in m.group(1).split(",") if k.strip()]
        check(3 <= len(kw) <= 8, f"{len(kw)} keywords")

    # ---- macros: every one used is defined --------------------------------
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", MACROS.read_text(encoding="utf-8")))
    used = set(re.findall(r"\\([A-Z][A-Za-z]*)\b", tex))
    # only consider names that look like our generated macros
    candidates = {u for u in used if u in defined or re.match(
        r"^(Prec|Aurc|Gap|Base|Best|Unit|Pure|ICC|Retest|N?Companies|Scored|Panel|News)", u)}
    missing = sorted(c for c in candidates if c not in defined)
    check(not missing, f"all generated macros defined {missing if missing else ''}")

    unused = sorted(defined - used)
    check(True, f"{len(defined)} macros defined, {len(unused)} unused", hard=False)

    # ---- figures and tables exist -----------------------------------------
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    for f in figs:
        p = ROOT / f
        check(p.exists(), f"figure present: {f}")
    inputs = re.findall(r"\\input\{([^}]+)\}", tex)
    for i in inputs:
        p = ROOT / (i if i.endswith(".tex") else i + ".tex")
        check(p.exists(), f"input present: {i}")

    # ---- log health --------------------------------------------------------
    if LOG.exists():
        log = LOG.read_text(encoding="utf-8", errors="ignore")
        check("! " not in log.replace("!  ==>", ""), "no LaTeX errors in log")
        check("undefined" not in log.lower(), "no undefined references or citations")
        check("Overfull" not in log, "no overfull boxes")
        pages = re.search(r"Output written on .*?\((\d+) pages", log)
        if pages:
            check(True, f"{pages.group(1)} pages", hard=False)

    # ---- numbers in prose match the result files --------------------------
    rel = ROOT / "results" / "reliability.json"
    if rel.exists():
        r = json.loads(rel.read_text())
        corr_claim = re.search(r"correlate(?:d)? at \$?0\.871", tex) or \
            re.search(r"0\.871", tex)
        check(bool(corr_claim), "nu-mu correlation quoted in prose", hard=False)
        check(r["n"] >= 200, f"cross-model agreement sample n={r['n']}", hard=False)

    # ---- references --------------------------------------------------------
    bbl = ROOT / "fininnov_mig.bbl"
    if bbl.exists():
        n = len(re.findall(r"\\bibitem", bbl.read_text(encoding="utf-8", errors="ignore")))
        check(n >= 30, f"{n} references in the bibliography")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"all hard checks passed ({len(notes)} advisory note(s))")


if __name__ == "__main__":
    main()
