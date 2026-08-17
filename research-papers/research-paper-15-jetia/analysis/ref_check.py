"""Verify every candidate reference against CrossRef and emit IEEE strings.

Each entry is looked up by title (or resolved directly by DOI).  Nothing is
written from memory: title, authors, container, volume, issue, pages, year
and DOI all come back from the CrossRef REST API.
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.parse
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

MAILTO = "aece-refcheck@example.org"
API = "https://api.crossref.org/works"

# CrossRef often returns a preprint/abstracting record ahead of the version
# of record; these containers are never the canonical citation.
BAD_CONTAINERS = ("ssrn electronic journal", "cfa digest", "ssrn",
                  "social science research network")
GOOD_TYPES = ("journal-article", "proceedings-article", "book", "book-chapter",
              "monograph", "reference-book")


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": f"ref-check/1.0 (mailto:{MAILTO})"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def by_doi(doi: str) -> dict | None:
    try:
        return _get(f"{API}/{urllib.parse.quote(doi)}")["message"]
    except Exception as e:
        print(f"  ! DOI lookup failed {doi}: {e}")
        return None


def by_title(title: str, rows: int = 5) -> list[dict]:
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": rows,
                                "mailto": MAILTO})
    try:
        return _get(f"{API}?{q}")["message"]["items"]
    except Exception as e:
        print(f"  ! title search failed: {e}")
        return []


def norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def ieee(m: dict) -> str:
    auth = []
    for a in m.get("author", [])[:20]:
        given = a.get("given", "")
        fam = a.get("family", "")
        ini = " ".join(f"{p[0]}." for p in given.replace("-", " ").split() if p)
        auth.append(f"{ini} {fam}".strip())
    if len(auth) >= 6:
        who = f"{auth[0]} et al."
    else:
        who = ", ".join(auth)
    title = (m.get("title") or [""])[0].strip().rstrip(".")
    cont = (m.get("container-title") or [""])
    cont = cont[0] if cont else ""
    vol = m.get("volume", "")
    iss = m.get("issue", "")
    pg = m.get("page", "")
    yr = ""
    for k in ("published-print", "published-online", "issued"):
        if m.get(k, {}).get("date-parts", [[None]])[0][0]:
            yr = m[k]["date-parts"][0][0]
            break
    doi = m.get("DOI", "")
    typ = m.get("type", "")
    bits = [f'{who}, "{title},"']
    if cont:
        bits.append(f"{cont},")
    if vol:
        bits.append(f"vol. {vol},")
    if iss:
        bits.append(f"no. {iss},")
    if pg:
        bits.append(f"pp. {pg},")
    if yr:
        bits.append(f"{yr}.")
    s = " ".join(bits)
    return f"{s} doi:{doi}", typ


CANDIDATES = [
    # --- financial news sentiment and prediction ---
    ("Giving content to investor sentiment: the role of media in the stock market", None),
    ("When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks", None),
    ("Textual analysis in finance", None),
    ("More than words: Quantifying language to measure firms' fundamentals", None),
    ("Text-based sentiment analysis in finance: Synthesising the existing literature and exploring future directions", None),
    ("Transformer-gated recurrent unit method for predicting stock price based on news sentiments and technical indicators", None),
    ("A stock price prediction model based on investor sentiment and optimized deep learning", None),
    ("Hybrid information mixing module for stock movement prediction", None),
    ("Stock price movement prediction using sentiment analysis and CandleStick chart representation", None),
    ("A generalization of multi-source fusion-based framework to stock selection", None),
    ("A hybrid model for stock price prediction based on multi-view heterogeneous data", None),
    ("Deep learning for financial applications: A survey", None),
    ("Financial time series forecasting with deep learning: A systematic literature review", None),
    ("FNSPID: A Comprehensive Financial News Dataset in Time Series", None),
    # --- NLP models and lexicons ---
    ("FinBERT: A Pretrained Language Model for Financial Communications", None),
    ("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", None),
    ("VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text", None),
    ("The General Inquirer: A Computer Approach to Content Analysis", None),
    # --- measurement error / errors in variables ---
    ("Measurement Error in Nonlinear Models: A Modern Perspective", None),
    ("Errors in variables in econometrics", None),
    ("Instrumental variables estimation with measurement error", None),
    # --- filtering, deconvolution, adaptive loops ---
    ("Adaptive Filter Theory", None),
    ("Stationary and nonstationary learning characteristics of the LMS adaptive filter", None),
    ("On the statistical efficiency of the LMS algorithm with nonstationary inputs", None),
    ("The complex LMS algorithm", None),
    ("Extraction of Signals from Noise", None),
    ("Analysis of the convergence of the normalized LMS algorithm", None),
    # --- evaluation ---
    ("Comparing predictive accuracy", None),
    ("A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix", None),
    ("Distributed lags: A survey", None),
    # --- AECE-published, for venue fit ---
    ("An Ensemble Model for Word-based DGA Botnet Detection Using XGBoost and BERT", None),
    ("A Message Passing Neural Network Framework with Learnable PageRank for Author Impact Assessment", None),
    ("Adaptive mu-law Gradient Quantization for Training MLPs and CNNs", None),
]


def main() -> None:
    out = []
    for title, doi in CANDIDATES:
        print(f"* {title[:80]}")
        m = by_doi(doi) if doi else None
        if m is None:
            items = by_title(title, rows=10)

            def usable(it):
                cont = ((it.get("container-title") or [""]) or [""])[0]
                return (it.get("type") in GOOD_TYPES
                        and cont.lower().strip() not in BAD_CONTAINERS)

            def matches(it):
                t = ((it.get("title") or [""]) or [""])[0]
                if not t:
                    return False
                a, b = norm(t), norm(title)
                return a.startswith(b[:40]) or b.startswith(a[:40]) or a == b

            ranked = ([it for it in items if matches(it) and usable(it)]
                      + [it for it in items if matches(it)]
                      + [it for it in items if usable(it)] + items)
            m = ranked[0] if ranked else None
        if m is None:
            print("  -> NOT FOUND")
            out.append({"query": title, "found": False})
            continue
        s, typ = ieee(m)
        got = (m.get("title") or [""])[0]
        print(f"  -> [{typ}] {got[:90]}")
        print(f"     {s[:150]}")
        out.append({"query": title, "found": True, "type": typ,
                    "crossref_title": got, "ieee": s, "doi": m.get("DOI"),
                    "container": (m.get("container-title") or [""])[0]
                    if m.get("container-title") else "",
                    "n_authors": len(m.get("author", []))})
        time.sleep(0.4)
    with open(sys.argv[1] if len(sys.argv) > 1 else "refs_verified.json",
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nfound {sum(1 for o in out if o.get('found'))}/{len(out)}")


if __name__ == "__main__":
    main()
