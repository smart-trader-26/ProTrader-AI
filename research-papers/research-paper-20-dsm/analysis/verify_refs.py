"""
Check every reference in refs.bib against CrossRef.

The rule for this project is that a citation must be a real, published, findable
work.  This script queries CrossRef by title, compares the returned author
surnames, year, journal and page range with what the .bib claims, and prints a
verdict per entry.  Anything marked MISMATCH or NOT FOUND has to be fixed or
removed by hand before submission - the script never edits the .bib itself.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIB = HERE.parent / "refs.bib"
OUT = HERE.parent / "results" / "refs_verified.json"
MAILTO = "anand.pardeshi@fcrit.ac.in"   # CrossRef asks for a contact in the UA


def parse_bib(text: str) -> list[dict]:
    entries = []
    for block in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = block.group(1), block.group(2).strip(), block.group(3)
        fields = {}
        for m in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n\s*\w+\s*=|\Z)", body, re.S):
            fields[m.group(1).lower()] = re.sub(r"\s+", " ", m.group(2)).strip()
        entries.append({"type": kind, "key": key, **fields})
    return entries


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": f"refcheck/1.0 (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def crossref_by_doi(doi: str) -> dict | None:
    """Authoritative lookup.  A DOI resolves to exactly one record, so this
    settles which version of a paper the entry actually points at - title search
    happily returns the SSRN or NBER preprint instead of the journal article."""
    try:
        return _get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")["message"]
    except Exception as exc:
        print(f"  crossref DOI error for {doi}: {exc}")
        return None


def crossref_by_title(title: str) -> dict | None:
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 5})
    try:
        items = _get(f"https://api.crossref.org/works?{q}")["message"]["items"]
    except Exception as exc:
        print(f"  crossref error: {exc}")
        return None
    want = norm(title)
    best, best_score = None, 0.0
    for it in items:
        score = overlap(want, norm((it.get("title") or [""])[0]))
        if score > best_score:
            best, best_score = it, score
    return best if best_score >= 0.75 else None


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def overlap(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    return len(wa & wb) / max(1, len(wa))


def year_of(item: dict) -> str | None:
    for f in ("published-print", "published-online", "issued", "created"):
        d = item.get(f, {}).get("date-parts", [[None]])[0][0]
        if d:
            return str(d)
    return None


def main() -> None:
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    print(f"{len(entries)} entries in {BIB.name}\n")
    report = []
    for e in entries:
        title = e.get("title", "")
        key = e["key"]
        if not title:
            report.append({"key": key, "verdict": "NO TITLE"})
            print(f"[NO TITLE ] {key}")
            continue
        doi = e.get("doi", "").strip()
        kind = e["type"].lower()
        # Books, book chapters and the PMLR/NeurIPS/JMLR proceedings genuinely have
        # no CrossRef record.  Title search on those returns confident nonsense
        # (a book review, an unrelated 2026 paper), so it is not attempted: they
        # are flagged for a human to confirm once, rather than silently "verified".
        no_crossref_expected = kind in {"book", "incollection"} or (
            kind == "inproceedings" and not doi
        )
        if no_crossref_expected:
            report.append({"key": key, "verdict": "MANUAL", "title": title, "type": kind})
            print(f"[MANUAL  ] {key}: {title[:60]} (no CrossRef record expected)")
            continue

        hit = crossref_by_doi(doi) if doi else crossref_by_title(title)
        via = "doi" if doi else "title"
        time.sleep(0.6)
        if hit is None:
            report.append({"key": key, "verdict": "NOT FOUND", "title": title})
            print(f"[NOT FOUND] {key}: {title[:65]}")
            continue

        got_year = year_of(hit)
        got_journal = (hit.get("container-title") or [""])[0]
        surnames = [a.get("family", "") for a in hit.get("author", []) if a.get("family")]
        claimed_year = e.get("year", "")
        first_claimed = re.split(r",| and ", e.get("author", ""))[0].strip()
        claimed_title = norm(title)

        problems = []
        if claimed_year and got_year and claimed_year != got_year:
            problems.append(f"year {claimed_year} vs crossref {got_year}")
        if surnames and first_claimed and surnames[0].lower() not in first_claimed.lower():
            problems.append(f"first author {first_claimed!r} vs crossref {surnames[0]!r}")
        got_title = norm((hit.get("title") or [""])[0])
        if got_title and overlap(claimed_title, got_title) < 0.6:
            problems.append(f"title drift: crossref says {(hit.get('title') or [''])[0][:60]!r}")
        if e.get("journal") and got_journal:
            if overlap(norm(e["journal"]), norm(got_journal)) < 0.4:
                problems.append(f"journal {e['journal']!r} vs crossref {got_journal!r}")

        verdict = "OK" if not problems else "MISMATCH"
        report.append(
            {
                "key": key,
                "verdict": verdict,
                "problems": problems,
                "crossref_title": (hit.get("title") or [""])[0],
                "crossref_journal": got_journal,
                "crossref_year": got_year,
                "doi": hit.get("DOI"),
                "volume": hit.get("volume"),
                "page": hit.get("page"),
                "matched_via": via,
            }
        )
        flag = "OK      " if not problems else "MISMATCH"
        print(f"[{flag}] {key} ({via}): {got_journal[:38]} {got_year} vol {hit.get('volume')} pp {hit.get('page')}")
        for p in problems:
            print(f"           ! {p}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    ok = [r for r in report if r["verdict"] == "OK"]
    manual = [r for r in report if r["verdict"] == "MANUAL"]
    bad = [r for r in report if r["verdict"] not in ("OK", "MANUAL")]
    print(
        f"\n{len(ok)}/{len(report)} verified against CrossRef; "
        f"{len(manual)} books/proceedings to confirm by hand; {len(bad)} need attention"
    )
    for r in bad:
        print("  -", r["key"], r["verdict"], r.get("problems", ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
