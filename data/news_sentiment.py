"""
News fetching and sentiment analysis with multiple source fallbacks.
Uses NewsAPI, Google News RSS, and Economic Times as sources.
"""

import re
import requests
import feedparser
import streamlit as st
from transformers import pipeline
from datetime import datetime, timedelta

from config.settings import NEWS_API_KEY, DataConfig


# Sentiment model configuration
SENTIMENT_MODEL = "ProsusAI/finbert"

# Lazy-loaded sentiment pipeline
_sentiment_pipeline = None


def _get_sentiment_pipeline():
    """Get or initialize the DistilRoBERTa-Financial sentiment pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    return _sentiment_pipeline


def _fetch_google_news_rss(stock_symbol: str, company_name: str = None) -> list:
    """
    Fetch news from Google News RSS feed.
    This works without API key.
    
    Args:
        stock_symbol: Stock symbol (e.g., "RELIANCE")
        company_name: Optional company name for better search
        
    Returns:
        List of news articles
    """
    articles = []
    
    # Build search query
    search_term = company_name or stock_symbol
    search_term = f"{search_term} India stock"
    
    # Google News RSS URL
    rss_url = f"https://news.google.com/rss/search?q={search_term}&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:20]:  # Get last 20 articles
            try:
                # Parse date
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
    """
    Fetch news from Economic Times RSS.
    
    Args:
        stock_symbol: Stock symbol
        
    Returns:
        List of news articles
    """
    articles = []
    
    # ET Markets RSS feeds
    rss_urls = [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",  # Markets
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",  # Stocks
    ]
    
    for rss_url in rss_urls:
        try:
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:15]:
                try:
                    # Only include if stock symbol or related terms appear
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
    Fetch news articles for a stock from multiple sources.
    Tries NewsAPI first, then Google News RSS, then Economic Times.
    
    Args:
        stock_symbol: Stock symbol (e.g., "RELIANCE")
    
    Returns:
        List of news articles
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
    
    # Source 2: Google News RSS (always available, no API key needed)
    google_articles = _fetch_google_news_rss(stock_symbol, company_name)
    all_articles.extend(google_articles)
    
    # Source 3: Economic Times RSS for general market news
    et_articles = _fetch_economic_times_rss(stock_symbol)
    all_articles.extend(et_articles)
    
    # Remove duplicates based on title similarity
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title = article.get('title', '').lower()[:50]  # First 50 chars
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(article)
    
    return unique_articles


def analyze_sentiment(text: str) -> tuple:
    """
    Analyze sentiment of text using FinBERT.
    
    Args:
        text: Text to analyze
    
    Returns:
        Tuple of (sentiment_label, confidence_score)
    """
    if not text:
        return "neutral", 0.0
    
    pipeline = _get_sentiment_pipeline()
    result = pipeline(text[:512])[0]  # 512 token limit
    return result['label'], result['score']


def filter_relevant_news(news_articles: list, stock_name: str) -> list:
    """
    Filter news articles to keep only those relevant to the stock.
    Now with broader matching including sector keywords.
    
    Args:
        news_articles: List of news articles from API
        stock_name: Stock name to filter by
    
    Returns:
        Filtered list of relevant articles
    """
    filtered_articles = []
    
    # Build search patterns
    company_name = DataConfig.STOCK_NAME_MAPPING.get(stock_name, stock_name)
    patterns = [
        stock_name.lower(),
        company_name.lower(),
        stock_name.lower().replace(' ', ''),
    ]
    
    # Add sector keywords for broader matching
    sector_keywords = ['nifty', 'sensex', 'market', 'stock', 'share', 'equity', 'bse', 'nse']
    
    for article in news_articles:
        title = (article.get('title') or '').lower()
        description = (article.get('description') or '').lower()
        combined = f"{title} {description}"
        
        # Check for stock-specific match
        stock_match = any(pattern in combined for pattern in patterns)
        
        # Check for general market news (fallback)
        market_match = any(kw in combined for kw in sector_keywords)
        
        if stock_match:
            article['relevance'] = 'high'
            filtered_articles.append(article)
        elif market_match and len(filtered_articles) < 10:  # Include some market news
            article['relevance'] = 'medium'
            filtered_articles.append(article)
    
    # Sort by relevance (high first)
    filtered_articles.sort(key=lambda x: x.get('relevance', 'low') == 'high', reverse=True)
    
    return filtered_articles


def analyze_news_sentiment(news_articles: list, stock_symbol: str) -> dict:
    """
    Analyze sentiment of all relevant news articles grouped by date.
    
    Args:
        news_articles: List of news articles
        stock_symbol: Stock symbol for filtering
    
    Returns:
        Dictionary with date keys and list of (sentiment, score) tuples
    """
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
    """
    Generate a summary of sentiment data.
    
    Args:
        daily_sentiment: Dictionary of daily sentiments
    
    Returns:
        Dictionary with sentiment statistics
    """
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
