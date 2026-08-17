"""Check every entry of refs.REFS field-by-field against CrossRef.

Flags: unresolvable DOI, title mismatch, wrong first author, wrong year,
wrong volume/issue/pages.  Anything flagged must be corrected before the
manuscript is built.
"""
from __future__ import annotations

import io
import json
import re
import sys
import time

sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402

from refs import REFS  # noqa: E402

API = "https://api.crossref.org/works/"
MAILTO = "aece-refcheck@example.org"


def fetch(doi: str):
    url = API + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={
        "User-Agent": f"ref-verify/1.0 (mailto:{MAILTO})"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.load(r)["message"]
    except Exception as e:
        return {"__error__": str(e)}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def year_of(m: dict):
    for k in ("published-print", "published-online", "issued"):
        dp = m.get(k, {}).get("date-parts", [[None]])
        if dp and dp[0] and dp[0][0]:
            return dp[0][0]
    return None


def main() -> None:
    bad = []
    for key, txt in REFS.items():
        mdoi = re.search(r"doi:(\S+)$", txt.strip())
        if not mdoi:
            print(f"[{key}] NO DOI IN ENTRY")
            bad.append(key)
            continue
        doi = mdoi.group(1)
        m = fetch(doi)
        if "__error__" in m:
            print(f"[{key}] DOI UNRESOLVED {doi}: {m['__error__']}")
            bad.append(key)
            time.sleep(0.3)
            continue

        problems = []
        # title
        ct = ((m.get("title") or [""]) or [""])[0]
        mt = re.search(r'"([^"]+),"', txt)
        mine = mt.group(1) if mt else txt.split(".")[0]
        if ct and norm(ct)[:35] not in norm(mine) and \
           norm(mine)[:35] not in norm(ct):
            problems.append(f"TITLE mine='{mine[:60]}' crossref='{ct[:60]}'")
        # first author
        au = m.get("author") or []
        if au:
            fam = au[0].get("family", "")
            if fam and norm(fam) not in norm(txt):
                problems.append(f"AUTHOR1 crossref='{fam}' not in entry")
            n_au = len(au)
            listed = txt.count(",") and None
            if n_au >= 6 and "et al" not in txt:
                problems.append(f"needs 'et al.' ({n_au} authors)")
        # year
        yr = year_of(m)
        if yr and str(yr) not in txt:
            problems.append(f"YEAR crossref={yr}")
        # volume / pages
        vol = m.get("volume")
        if vol and f"vol. {vol}" not in txt:
            problems.append(f"VOLUME crossref={vol}")
        pg = m.get("page")
        if pg and pg.replace("--", "-") not in txt.replace("--", "-"):
            problems.append(f"PAGES crossref={pg}")
        iss = m.get("issue")
        if iss and f"no. {iss}" not in txt and vol:
            problems.append(f"ISSUE crossref={iss}")

        if problems:
            bad.append(key)
            print(f"[{key}] {doi}")
            for p in problems:
                print(f"    - {p}")
            authors = ", ".join(f"{a.get('given','')} {a.get('family','')}"
                                .strip() for a in au[:8])
            print(f"    crossref: {ct[:80]}")
            print(f"    authors : {authors[:110]}")
            print(f"    where   : {((m.get('container-title') or ['']) or [''])[0][:60]}"
                  f" | vol {vol} no {iss} pp {pg} ({yr})")
        else:
            print(f"[{key}] OK")
        time.sleep(0.25)

    print(f"\n{len(REFS) - len(bad)}/{len(REFS)} clean; "
          f"needs attention: {bad}")


if __name__ == "__main__":
    main()
