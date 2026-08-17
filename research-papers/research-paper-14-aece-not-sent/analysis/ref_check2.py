"""Round 2: pin canonical records by DOI and verify the returned title.

Any DOI that resolves to a different work is immediately visible because
the CrossRef title is printed next to the expectation.
"""
from __future__ import annotations

import io
import json
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])

from ref_check import by_doi, by_title, ieee, norm  # noqa: E402

# (expected short title, doi or None)
PINNED = [
    ("Textual analysis in finance", "10.1146/annurev-financial-012820-032249"),
    ("BERT: Pre-training of deep bidirectional transformers",
     "10.18653/v1/N19-1423"),
    ("Comparing predictive accuracy", "10.1080/07350015.1995.10524599"),
    ("An introduction to matched filters", "10.1109/TIT.1960.1057571"),
    ("An analysis of the factors which determine signal/noise discrimination",
     "10.1109/PROC.1963.2383"),
    ("Stationary and nonstationary learning characteristics of the LMS",
     "10.1109/PROC.1976.10286"),
    ("On the convergence behavior of the LMS and the normalized LMS",
     "10.1109/78.257263"),
    ("Measurement error in nonlinear models", "10.1201/9781420010138"),
    ("Measurement error models", "10.1002/9780470316665"),
    ("Structural equation methods in the social sciences",
     "10.2307/1913807"),
    ("Stock price reaction to news and no-news",
     "10.1016/S0304-405X(02)00223-3"),
    ("Stock movement prediction from tweets and historical prices",
     "10.18653/v1/P18-1183"),
    ("Listening to chaotic whispers", "10.1145/3159652.3159690"),
    ("Deep attentive learning for stock movement prediction",
     "10.18653/v1/2020.emnlp-main.676"),
    ("Which news moves stock prices?", "10.1093/rfs/hhy036"),
    ("Adaptive filters (Sayed)", "10.1002/9780470374122"),
]

UNPINNED = [
    "Measurement error in survey data",
    "Attention allocation and return co-movement of news",
    "A deep learning framework for financial time series using stacked autoencoders and long short term memory",
    "Predicting stock market index using fusion of machine learning techniques",
    "Sentiment analysis of financial news: mechanics and statistics",
    "The use of news articles for stock market prediction: a survey",
]


def main() -> None:
    out = []
    for want, doi in PINNED:
        m = by_doi(doi)
        if m is None:
            print(f"!! UNRESOLVED {doi}  (wanted: {want})")
            out.append({"query": want, "doi": doi, "found": False})
            continue
        got = ((m.get("title") or [""]) or [""])[0]
        s, typ = ieee(m)
        agree = norm(want)[:25] in norm(got) or norm(got)[:25] in norm(want)
        flag = "OK " if agree else "?? "
        print(f"{flag}{doi}\n    want: {want}\n    got : {got[:88]}  [{typ}]")
        print(f"    {s[:160]}")
        out.append({"query": want, "found": True, "match_ok": bool(agree),
                    "type": typ, "crossref_title": got, "ieee": s,
                    "doi": m.get("DOI"),
                    "container": ((m.get("container-title") or [""]) or [""])[0]})
    for t in UNPINNED:
        items = by_title(t, rows=6)
        if not items:
            print(f"!! NOT FOUND {t}")
            continue
        m = items[0]
        got = ((m.get("title") or [""]) or [""])[0]
        s, typ = ieee(m)
        print(f"?  search: {t}\n    got : {got[:88]}  [{typ}]\n    {s[:160]}")
        out.append({"query": t, "found": True, "match_ok": None, "type": typ,
                    "crossref_title": got, "ieee": s, "doi": m.get("DOI"),
                    "container": ((m.get("container-title") or [""]) or [""])[0]})
    with open("analysis/refs_verified2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
