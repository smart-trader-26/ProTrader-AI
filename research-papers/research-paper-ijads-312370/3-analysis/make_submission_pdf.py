"""Bind the response to reviewers at the front of the revised article.

The Editor's letter asks for "a detailed response to the reviewers' recommendations"
inserted "at the beginning of your revised article". This produces that single file.

The response pages are numbered R1..Rn and the manuscript pages 1..m, so every "p. n"
reference inside the response points unambiguously at a manuscript page, not at a page
of the combined file. Bookmarks are added for the two parts.

Usage:  python 3-analysis/make_submission_pdf.py
Output: 2-revised-manuscript/IJADS-312370-revised-with-response.pdf
"""
import os
import sys

import pypdf

HERE = os.path.dirname(os.path.abspath(__file__))
MS_DIR = os.path.abspath(os.path.join(HERE, "..", "2-revised-manuscript"))
RESPONSE = os.path.join(MS_DIR, "response-to-reviewers.pdf")
ARTICLE = os.path.join(MS_DIR, "ai67.pdf")
OUT = os.path.join(MS_DIR, "IJADS-312370-revised-with-response.pdf")


def main():
    for f in (RESPONSE, ARTICLE):
        if not os.path.exists(f):
            sys.exit("missing: %s  (run assemble.py and pdflatex first)" % f)

    writer = pypdf.PdfWriter()

    resp = pypdf.PdfReader(RESPONSE)
    for page in resp.pages:
        writer.add_page(page)
    n_resp = len(resp.pages)

    art = pypdf.PdfReader(ARTICLE)
    for page in art.pages:
        writer.add_page(page)
    n_art = len(art.pages)

    writer.add_outline_item("Response to the Editor and Reviewers", 0)
    writer.add_outline_item("Revised article", n_resp)

    writer.add_metadata(
        {
            "/Title": "IJADS-312370 revised article, with response to reviewers",
            "/Subject": "From Signal Fusion to Asset Allocation: A Decision-Theoretic "
            "Model for Portfolio Construction Under Regime-Based Sentiment and Volatility",
            "/Author": "Anandkumar Pardeshi, Sujata Deshmukh",
        }
    )

    with open(OUT, "wb") as fh:
        writer.write(fh)

    print("wrote %s" % OUT)
    print("  response : %2d pages (R1-R%d)" % (n_resp, n_resp))
    print("  article  : %2d pages (1-%d)" % (n_art, n_art))
    print("  total    : %2d pages" % (n_resp + n_art))


if __name__ == "__main__":
    main()
