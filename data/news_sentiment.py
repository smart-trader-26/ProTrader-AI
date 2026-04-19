"""
News fetching and sentiment analysis with multiple source fallbacks.
Uses NewsAPI, Google News RSS, and Economic Times as sources.

Sentiment engine: ProsusAI/finbert (FinBERT) augmented with high-precision
bullish/bearish keyword overrides — adapted from the v2 Colab benchmark
where this hybrid scored higher accuracy than raw FinBERT alone on
Indian financial headlines.
"""

import re
import requests
import feedparser
import streamlit as st
from transformers import pipeline
from datetime import datetime, timedelta

from config.settings import NEWS_API_KEY, DataConfig


# ==============================================
# SENTIMENT MODEL CONFIGURATION
# ==============================================
SENTIMENT_MODEL = "ProsusAI/finbert"


# ==============================================
# v2-DERIVED FINANCIAL CATEGORIES + KEYWORDS
# ==============================================
# Six categories used by the Colab benchmark (sentiment_analysis_v2.py).
# Each headline is bucketed by the dominant signal terms it contains.
VALID_CATEGORIES = [
    "Earnings & Output", "Analyst Ratings", "Market Action",
    "Deals & Acquisitions", "Macro & Policy", "Other"
]

CATEGORY_KEYWORDS = {
    "Earnings & Output": [
        "earnings", "profit", "revenue", "net profit", "eps", "ebitda",
        "results", "guidance", "q1", "q2", "q3", "q4", "quarterly",
        "annual results", "topline", "bottomline", "margin",
    ],
    "Analyst Ratings": [
        "upgrade", "downgrade", "buy rating", "sell rating", "outperform",
        "underperform", "neutral rating", "target price", "price target",
        "initiates coverage", "goldman", "morgan stanley", "citi", "clsa",
        "macquarie", "jefferies", "nomura",
    ],
    "Market Action": [
        "sensex", "nifty", "fii", "dii", "rally", "surge", "crash",
        "tumble", "soar", "plunge", "52-week high", "52-week low",
        "circuit", "lower circuit", "upper circuit", "panic selling",
        "block deal", "bulk deal",
    ],
    "Deals & Acquisitions": [
        "acquisition", "acquires", "merger", "demerger", "joint venture",
        "stake sale", "open offer", "takeover", "buyout", "spin-off",
    ],
    "Macro & Policy": [
        "rbi", "sebi", "repo rate", "inflation", "gdp", "budget",
        "policy", "regulation", "tariff", "tax", "gst", "fed",
        "monetary policy", "fiscal", "ministry",
    ],
}

# High-precision keyword overrides (taken from sentiment_analysis_v2.py).
# Triggered before model output: if a clear signal word is present AND the
# model agrees (or is neutral), we boost confidence to 0.88+.
BULLISH_KEYWORDS = [
    "buy", "strong buy", "upgrade", "bullish", "surge", "rally", "soar",
    "record", "beat", "beats", "outperform", "positive", "growth", "win",
    "wins", "high", "all-time high", "52-week high", "approve", "approved",
    "expansion", "launch", "partnership",
]
BEARISH_KEYWORDS = [
    "sell", "strong sell", "downgrade", "bearish", "drop", "crash",
    "plunge", "slump", "loss", "losses", "penalty", "fraud", "default",
    "miss", "misses", "decline", "warning", "cut", "disappoints",
    "probe", "investigation", "lawsuit", "fine", "raid",
]


@st.cache_resource(show_spinner="Loading FinBERT (first call only)…")
def _get_sentiment_pipeline():
    """
    Load the FinBERT sentiment pipeline once per Streamlit session.

    `@st.cache_resource` keeps the ~500 MB model resident across reruns and
    shares it with every module that imports this function (previously both
    news_sentiment and multi_sentiment loaded their own copy → 2× RAM).
    Outside Streamlit (pytest, FastAPI) the decorator degrades to a no-op
    and the function still loads on first call.

    `framework="pt"` forces PyTorch (hard invariant CLAUDE.md §2.1). Without
    it, transformers loads the TF head and crashes on Streamlit Cloud where
    tensorflow-cpu>=2.16 ships with Keras 3 (incompatible with HF).
    """
    return pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        framework="pt",
    )


# ==============================================
# NEWS FETCHING
# ==============================================

def _fetch_google_news_rss(stock_symbol: str, company_name: str = None) -> list:
    """Fetch news from Google News RSS feed (no API key required)."""
    articles = []
    search_term = company_name or stock_symbol
    search_term = f"{search_term} India stock"
    rss_url = f"https://news.google.com/rss/search?q={search_term}&hl=en-IN&gl=IN&ceid=IN:en"

    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:20]:
            try:
                pub_date = entry.get('published', '')
                if pub_date:
                    try:
                        parsed_date = datetime(*entry.published_parsed[:6])
                        pub_date = parsed_date.isoformat()
                    except Exception:
                        pub_date = datetime.now().isoformat()

                articles.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', ''),
                    'publishedAt': pub_date,
                    'source': {'name': 'Google News'},
                    'url': entry.get('link', '')
                })
            except Exception:
                continue
    except Exception:
        pass

    return articles


def _fetch_economic_times_rss(stock_symbol: str) -> list:
    """Fetch news from Economic Times RSS feeds."""
    articles = []
    rss_urls = [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    ]

    for rss_url in rss_urls:
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:15]:
                try:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    pub_date = entry.get('published', '')
                    if pub_date:
                        try:
                            parsed_date = datetime(*entry.published_parsed[:6])
                            pub_date = parsed_date.isoformat()
                        except Exception:
                            pub_date = datetime.now().isoformat()

                    articles.append({
                        'title': title,
                        'description': summary,
                        'publishedAt': pub_date,
                        'source': {'name': 'Economic Times'},
                        'url': entry.get('link', '')
                    })
                except Exception:
                    continue
        except Exception:
            continue

    return articles


def get_news(stock_symbol: str) -> list:
    """
    Fetch news articles from multiple sources with NewsAPI → Google News → ET fallback.
    Deduplicates by title prefix.
    """
    all_articles = []
    company_name = DataConfig.STOCK_NAME_MAPPING.get(stock_symbol, stock_symbol)

    # Source 1: NewsAPI (if configured)
    if NEWS_API_KEY:
        params = {
            "q": company_name,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt"
        }
        try:
            response = requests.get(DataConfig.NEWS_API_URL, params=params, timeout=10)
            if response.status_code == 200:
                newsapi_articles = response.json().get("articles", [])
                all_articles.extend(newsapi_articles)
        except Exception:
            pass

    # Source 2: Google News RSS
    google_articles = _fetch_google_news_rss(stock_symbol, company_name)
    all_articles.extend(google_articles)

    # Source 3: Economic Times RSS
    et_articles = _fetch_economic_times_rss(stock_symbol)
    all_articles.extend(et_articles)

    # Deduplicate by title prefix
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title = article.get('title', '').lower()[:50]
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(article)

    return unique_articles


# ==============================================
# SENTIMENT ANALYSIS (FinBERT + keyword overrides)
# ==============================================

def _normalize_label(raw_label: str) -> str:
    """Normalize various FinBERT/RoBERTa label conventions to {positive,negative,neutral}."""
    l = (raw_label or "").lower()
    if l in ("positive", "pos", "label_2"):
        return "positive"
    if l in ("negative", "neg", "label_0"):
        return "negative"
    return "neutral"


def analyze_sentiment(text: str) -> tuple:
    """
    Analyze sentiment with FinBERT + high-precision keyword overrides.

    Override rule (from sentiment_analysis_v2.py Cell 5):
      - If text contains a strong bullish/bearish keyword AND the model
        agrees (or is neutral), confidence is boosted to >= 0.88.
      - Otherwise the raw FinBERT output is returned unchanged.
      - This avoids both keyword false-positives (model disagrees) and
        sub-confident model output on clear signals (e.g. "Wipro misses
        revenue estimates" → model returns 0.65 confidence; with override → 0.88).

    Returns:
        (sentiment_label: 'positive'|'negative'|'neutral', confidence: float)
    """
    if not text:
        return "neutral", 0.0

    text_lower = text.lower()
    pipe = _get_sentiment_pipeline()

    try:
        result = pipe(text[:512])[0]
    except Exception:
        return "neutral", 0.5

    raw_label = _normalize_label(result.get('label', 'neutral'))
    raw_score = float(result.get('score', 0.5))

    # Bearish keyword override
    if any(w in text_lower for w in BEARISH_KEYWORDS):
        if raw_label in ("negative", "neutral"):
            return "negative", max(raw_score, 0.88)

    # Bullish keyword override
    if any(w in text_lower for w in BULLISH_KEYWORDS):
        if raw_label in ("positive", "neutral"):
            return "positive", max(raw_score, 0.88)

    return raw_label, raw_score


def categorize_headline(text: str) -> str:
    """
    Lightweight 6-category classifier (no GPU required).
    Uses keyword matching to assign one of the v2 financial categories.
    Falls back to 'Other' when no category keywords match.
    """
    if not text:
        return "Other"
    text_lower = text.lower()
    best_cat = "Other"
    best_hits = 0
    for cat, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > best_hits:
            best_hits = hits
            best_cat = cat
    return best_cat


def filter_relevant_news(news_articles: list, stock_name: str) -> list:
    """Filter news articles to keep only those relevant to the stock."""
    filtered_articles = []
    company_name = DataConfig.STOCK_NAME_MAPPING.get(stock_name, stock_name)
    patterns = [
        stock_name.lower(),
        company_name.lower(),
        stock_name.lower().replace(' ', ''),
    ]
    sector_keywords = ['nifty', 'sensex', 'market', 'stock', 'share',
                       'equity', 'bse', 'nse']

    for article in news_articles:
        title = (article.get('title') or '').lower()
        description = (article.get('description') or '').lower()
        combined = f"{title} {description}"

        stock_match = any(pattern in combined for pattern in patterns)
        market_match = any(kw in combined for kw in sector_keywords)

        if stock_match:
            article['relevance'] = 'high'
            filtered_articles.append(article)
        elif market_match and len(filtered_articles) < 10:
            article['relevance'] = 'medium'
            filtered_articles.append(article)

    filtered_articles.sort(
        key=lambda x: x.get('relevance', 'low') == 'high', reverse=True
    )
    return filtered_articles


def analyze_news_sentiment(news_articles: list, stock_symbol: str) -> dict:
    """Analyze sentiment of all relevant news articles, grouped by date."""
    filtered_news = filter_relevant_news(news_articles, stock_symbol)
    daily_sentiment = {}

    for article in filtered_news:
        text = f"{article.get('title', '')} {article.get('description', '')}".strip()
        sentiment, score = analyze_sentiment(text)
        date = article.get("publishedAt", "")[:10]

        if date in daily_sentiment:
            daily_sentiment[date].append((sentiment, score))
        else:
            daily_sentiment[date] = [(sentiment, score)]

    return daily_sentiment


def get_sentiment_summary(daily_sentiment: dict) -> dict:
    """Generate a summary of daily sentiment data."""
    if not daily_sentiment:
        return {
            "text": "Neutral (No recent news)",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "sentiment_ratio": 0.5
        }

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for date_sentiments in daily_sentiment.values():
        for label, _ in date_sentiments:
            if label == 'positive':
                positive_count += 1
            elif label == 'negative':
                negative_count += 1
            else:
                neutral_count += 1

    total = positive_count + negative_count

    if total > 0:
        sentiment_ratio = positive_count / total
        if sentiment_ratio > 0.6:
            text = f"Bullish ({positive_count}/{total} positive articles)"
        elif sentiment_ratio < 0.4:
            text = f"Bearish ({negative_count}/{total} negative articles)"
        else:
            text = f"Mixed ({positive_count} positive, {negative_count} negative)"
    else:
        sentiment_ratio = 0.5
        text = "Neutral (No sentiment data)"

    return {
        "text": text,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "sentiment_ratio": sentiment_ratio
    }
