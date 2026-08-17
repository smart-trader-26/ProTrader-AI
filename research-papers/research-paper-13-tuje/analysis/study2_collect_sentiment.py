"""
Study 2, step 1 -- collect REAL headlines and score them with the REAL
production sentiment model.

Headlines: yfinance's `.news` endpoint (the same and only news source used
in production -- colab-notebooks/README.md:81-86 confirms GDELT was
rejected and "Yahoo Finance's own .news endpoint is the sole news source").
This endpoint only returns a ticker's most recent ~10 stories, so the
resulting sample is real but small and covers a sparse, recent window
(observed here: ~Sept 2025 to Jul 2026) -- this is a hard data-availability
constraint, not a design choice, and is reported honestly in the paper as
this case study's scope.

Sentiment model: ProsusAI/finbert via data.news_sentiment.analyze_sentiment,
i.e. the exact function used by the deployed app (data/news_sentiment.py:431),
including its keyword-override rule. No headline text or score is invented.
"""

import json
import sys
from pathlib import Path

import yfinance as yf

HERE = Path(__file__).parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(HERE.parent.parent.parent))  # repo root, for `data.news_sentiment`
from data.news_sentiment import analyze_sentiment  # noqa: E402

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "MARUTI.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "SUNPHARMA.NS", "LT.NS", "BAJFINANCE.NS",
]


def main():
    all_rows = []
    for tkr in TICKERS:
        try:
            news = yf.Ticker(tkr).news
        except Exception as e:
            print(f"{tkr}: fetch failed ({e})")
            continue
        print(f"{tkr}: {len(news)} real headlines")
        for item in news:
            c = item.get("content", {})
            title = c.get("title", "") or ""
            summary = c.get("summary", "") or ""
            pub = c.get("pubDate") or c.get("displayTime")
            if not title or not pub:
                continue
            text = f"{title}. {summary}".strip()
            label, conf = analyze_sentiment(text, source_language="en")
            signed = conf if label == "positive" else (-conf if label == "negative" else 0.0)
            all_rows.append({
                "ticker": tkr, "pub_date_utc": pub, "title": title,
                "label": label, "confidence": float(conf), "signed_sentiment": float(signed),
            })

    with open(OUT / "study2_headline_sentiment.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    print(f"\nTotal real, FinBERT-scored headlines: {len(all_rows)}")
    print(f"Saved -> {OUT / 'study2_headline_sentiment.json'}")


if __name__ == "__main__":
    main()
