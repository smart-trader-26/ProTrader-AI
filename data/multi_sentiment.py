"""
Multi-Source Sentiment Analysis System.
Combines RSS feeds, NewsAPI, Reddit, and Google Trends for maximum accuracy.
"""

import math
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import streamlit as st

# RSS Feed parsing
import feedparser

# Google Trends
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    TrendReq = None

# Reddit API
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    praw = None

from config.settings import (
    NEWS_API_KEY,
    REDDIT_CLIENT_ID, 
    REDDIT_CLIENT_SECRET, 
    REDDIT_USER_AGENT
)

# NewsAPI configuration
NEWS_API_URL = "https://newsapi.org/v2/everything"


# ==============================================
# RSS FEED SOURCES (Most Reliable)
# ==============================================

RSS_FEEDS = {
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_news": "https://www.moneycontrol.com/rss/latestnews.xml",
    "economic_times_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "economic_times_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
}

# Stock name mapping for relevance filtering
STOCK_KEYWORDS = {
    "RELIANCE": ["reliance", "ril", "jio", "mukesh ambani"],
    "TCS": ["tcs", "tata consultancy", "tata tech"],
    "INFY": ["infosys", "infy", "salil parekh"],
    "HDFCBANK": ["hdfc bank", "hdfc", "housing development"],
    "ICICIBANK": ["icici bank", "icici"],
    "SBIN": ["sbi", "state bank of india"],
    "BHARTIARTL": ["bharti airtel", "airtel"],
    "ITC": ["itc limited", "itc hotels"],
    "KOTAKBANK": ["kotak mahindra", "kotak bank"],
    "LT": ["larsen", "l&t", "larsen toubro"],
    "NIFTY": ["nifty", "nifty50", "nifty 50", "index"],
    "BANKNIFTY": ["bank nifty", "banknifty", "banking index"],
    "FII": ["fii", "foreign institutional", "fiis"],
    "DII": ["dii", "domestic institutional", "diis"],
}

# Subreddits for Indian market sentiment
INDIAN_MARKET_SUBREDDITS = [
    "IndianStockMarket",
    "DalalStreetTalks",
    "IndiaInvestments",
    "indianstreetbets",
]

# ==============================================
# EVENT CLASSIFICATION CONSTANTS
# ==============================================

EVENT_WEIGHTS = {
    'earnings': 2.0,
    'regulatory': 1.8,
    'dividend': 1.5,
    'management': 1.3,
    'general': 1.0,
}

EVENT_KEYWORDS = {
    'earnings': [
        'quarterly results', 'q1', 'q2', 'q3', 'q4', 'net profit', 'revenue',
        'eps', 'pat', 'ebitda', 'guidance', 'earnings', 'profit', 'loss', 'turnover',
        'annual results', 'financial results', 'consolidated results'
    ],
    'dividend': [
        'dividend', 'bonus share', 'buyback', 'stock split', 'interim dividend',
        'final dividend', 'record date', 'ex-date', 'rights issue'
    ],
    'management': [
        'ceo', 'md ', 'board', 'appointment', 'resignation', 'cfo', 'director',
        'chairman', 'managing director', 'chief executive', 'promoter', 'stake'
    ],
    'regulatory': [
        'sebi', 'rbi', 'government', 'policy', 'ban', 'compliance', 'penalty',
        'notice', 'regulation', 'nse', 'bse', 'ministry', 'court', 'litigation',
        'investigation', 'probe', 'fii', 'fpi', 'tax', 'gst'
    ],
}


def _temporal_weight(pub_date: datetime, lambda_decay: float = 0.5) -> float:
    """
    Exponential decay weight for article age.
    w = exp(-λ × days_old): same-day=1.0, 1-day=0.61, 2-day=0.37, 3-day=0.22
    """
    try:
        # Handle timezone-aware datetimes
        now = datetime.now()
        if hasattr(pub_date, 'tzinfo') and pub_date.tzinfo is not None:
            from datetime import timezone
            now = datetime.now(timezone.utc)
        days_old = max(0, (now - pub_date).total_seconds() / 86400)
    except Exception:
        days_old = 1.0  # Default to 1-day-old if date parsing fails
    return math.exp(-lambda_decay * days_old)


def _classify_article_event(title: str, body: str = '') -> Tuple[str, float]:
    """
    Classify article into event type using keyword matching.
    Returns (event_type, event_weight).
    Priority: earnings > regulatory > dividend > management > general
    """
    text = (title + ' ' + body).lower()
    for event_type, keywords in EVENT_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return event_type, EVENT_WEIGHTS[event_type]
    return 'general', EVENT_WEIGHTS['general']


class MultiSourceSentiment:
    """
    Multi-source sentiment aggregator for high-accuracy market analysis.
    
    Sources:
    1. RSS News Feeds (Moneycontrol, ET, LiveMint, Business Standard) - 30%
    2. NewsAPI (global financial news) - 25%
    3. Reddit (Indian market subreddits) - 25%
    4. Google Trends (retail interest proxy) - 20%
    """
    
    def __init__(self, sentiment_model: str = "ProsusAI/finbert"):
        """
        Initialize multi-source sentiment analyzer.
        
        Args:
            sentiment_model: HuggingFace model for sentiment classification
        """
        self.sentiment_model = sentiment_model
        self._sentiment_pipeline = None
        self._reddit_client = None
        self._pytrends = None
        self._dynamic_keywords: Dict[str, List[str]] = {}  # Session cache for dynamic keywords
        
    def _get_sentiment_pipeline(self):
        """Lazy load sentiment pipeline."""
        if self._sentiment_pipeline is None:
            from transformers import pipeline
            self._sentiment_pipeline = pipeline("sentiment-analysis", model=self.sentiment_model)
        return self._sentiment_pipeline
    
    def _get_reddit_client(self):
        """Initialize Reddit client if credentials available."""
        if self._reddit_client is None and PRAW_AVAILABLE:
            if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
                try:
                    self._reddit_client = praw.Reddit(
                        client_id=REDDIT_CLIENT_ID,
                        client_secret=REDDIT_CLIENT_SECRET,
                        user_agent=REDDIT_USER_AGENT
                    )
                except Exception as e:
                    st.warning(f"Reddit client init failed: {str(e)[:50]}")
        return self._reddit_client
    
    def _get_pytrends(self):
        """Initialize PyTrends client."""
        if self._pytrends is None and PYTRENDS_AVAILABLE:
            try:
                self._pytrends = TrendReq(hl='en-IN', tz=330)  # India timezone
            except Exception:
                pass
        return self._pytrends
    
    def _build_dynamic_keyword_map(self, symbol: str) -> List[str]:
        """
        Build keyword list for any stock using yfinance ticker info.
        No extra API call if ticker info is already fetched; falls back to symbol only.
        Result cached for the session lifetime.
        """
        if symbol in self._dynamic_keywords:
            return self._dynamic_keywords[symbol]

        keywords = [symbol.lower()]
        try:
            import yfinance as yf
            ticker_str = symbol if '.' in symbol else f"{symbol}.NS"
            info = yf.Ticker(ticker_str).info
            long_name = info.get('longName', '') or info.get('shortName', '')
            if long_name:
                keywords.append(long_name.lower())
                first_word = long_name.split()[0].lower()
                if len(first_word) > 3 and first_word not in keywords:
                    keywords.append(first_word)
        except Exception:
            pass  # Keep symbol-only keywords on failure

        self._dynamic_keywords[symbol] = keywords
        return keywords

    def analyze_text(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of a single text.
        
        Returns:
            Tuple of (label: 'positive'|'negative'|'neutral', confidence: 0-1)
        """
        if not text or len(text.strip()) < 10:
            return "neutral", 0.5
        
        try:
            pipeline = self._get_sentiment_pipeline()
            result = pipeline(text[:512])[0]
            return result['label'].lower(), result['score']
        except Exception:
            return "neutral", 0.5
    
    # ==============================================
    # RSS FEED COLLECTION (Most Reliable Source)
    # ==============================================
    
    @st.cache_data(ttl=900)  # Cache for 15 minutes
    def fetch_rss_news(_self, stock_symbol: str = None, max_articles: int = 50) -> List[Dict]:
        """
        Fetch news from multiple RSS feeds.
        
        Args:
            stock_symbol: Optional stock to filter for
            max_articles: Maximum articles to fetch per feed
        
        Returns:
            List of article dictionaries
        """
        all_articles = []
        keywords = []
        
        if stock_symbol:
            keywords = STOCK_KEYWORDS.get(stock_symbol.upper(), [stock_symbol.lower()])
        
        for feed_name, feed_url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:max_articles]:
                    title = entry.get('title', '')
                    summary = entry.get('summary', entry.get('description', ''))
                    published = entry.get('published', entry.get('updated', ''))
                    link = entry.get('link', '')
                    
                    # Parse date
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        else:
                            pub_date = datetime.now()
                    except Exception:
                        pub_date = datetime.now()
                    
                    # Filter by keyword if specified
                    if keywords:
                        text_lower = f"{title} {summary}".lower()
                        if not any(kw in text_lower for kw in keywords):
                            continue
                    
                    all_articles.append({
                        'source': feed_name,
                        'title': title,
                        'summary': summary[:500] if summary else '',
                        'date': pub_date,
                        'link': link,
                        'type': 'rss'
                    })
                    
            except Exception as e:
                continue  # Skip failed feeds silently
        
        # Sort by date (newest first)
        all_articles.sort(key=lambda x: x['date'], reverse=True)
        
        return all_articles[:max_articles * 2]  # Return top articles
    
    # ==============================================
    # NEWSAPI COLLECTION (Global Financial News)
    # ==============================================
    
    @st.cache_data(ttl=900)  # Cache for 15 minutes
    def fetch_newsapi_articles(_self, stock_symbol: str = None, max_articles: int = 30) -> List[Dict]:
        """
        Fetch news from NewsAPI.
        
        Args:
            stock_symbol: Stock symbol to search for
            max_articles: Maximum articles to fetch
        
        Returns:
            List of article dictionaries
        """
        if not NEWS_API_KEY:
            return []
        
        all_articles = []
        
        # Build search query
        if stock_symbol:
            keywords = STOCK_KEYWORDS.get(stock_symbol.upper(), [stock_symbol])
            query = " OR ".join(keywords[:3])  # Use top 3 keywords
        else:
            query = "stock market India NSE"
        
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles
        }
        
        try:
            response = requests.get(NEWS_API_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                
                for article in articles:
                    title = article.get("title", "")
                    description = article.get("description", "")
                    published = article.get("publishedAt", "")
                    source_name = article.get("source", {}).get("name", "NewsAPI")
                    
                    # Parse date
                    try:
                        pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    except Exception:
                        pub_date = datetime.now()
                    
                    all_articles.append({
                        'source': f"NewsAPI: {source_name}",
                        'title': title,
                        'summary': description[:500] if description else '',
                        'date': pub_date,
                        'type': 'newsapi'
                    })
                    
        except Exception as e:
            pass  # Return empty list on error
        
        return all_articles
    
    # ==============================================
    # REDDIT COLLECTION
    # ==============================================
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def fetch_reddit_posts(_self, stock_symbol: str = None, max_posts: int = 30) -> List[Dict]:
        """
        Fetch posts from Indian market subreddits.
        
        Args:
            stock_symbol: Optional stock to filter for
            max_posts: Maximum posts per subreddit
        
        Returns:
            List of post dictionaries
        """
        reddit = _self._get_reddit_client()
        if reddit is None:
            return []
        
        keywords = []
        if stock_symbol:
            keywords = STOCK_KEYWORDS.get(stock_symbol.upper(), [stock_symbol.lower()])
        
        all_posts = []
        
        for subreddit_name in INDIAN_MARKET_SUBREDDITS:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                
                for post in subreddit.new(limit=max_posts):
                    title = post.title
                    text = post.selftext
                    
                    # Filter by keyword if specified
                    if keywords:
                        content_lower = f"{title} {text}".lower()
                        if not any(kw in content_lower for kw in keywords):
                            continue
                    
                    all_posts.append({
                        'source': f"r/{subreddit_name}",
                        'title': title,
                        'summary': text[:500] if text else '',
                        'date': datetime.fromtimestamp(post.created_utc),
                        'score': post.score,
                        'comments': post.num_comments,
                        'type': 'reddit'
                    })
                    
            except Exception as e:
                continue  # Skip failed subreddits
        
        # Sort by score (engagement) and date
        all_posts.sort(key=lambda x: (x['score'], x['date']), reverse=True)
        
        return all_posts[:max_posts * 2]
    
    # ==============================================
    # GOOGLE TRENDS (Retail Interest Proxy)
    # ==============================================
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_google_trends(_self, stock_symbol: str, days: int = 7) -> Dict:
        """
        Fetch Google Trends data for a stock/keyword.
        
        Args:
            stock_symbol: Stock symbol to search
            days: Lookback period
        
        Returns:
            Dictionary with trend data and signals
        """
        pytrends = _self._get_pytrends()
        if pytrends is None:
            return {'available': False, 'signal': 0, 'trend': 'unknown'}
        
        try:
            # Build search keywords
            keywords = [stock_symbol]
            if stock_symbol in STOCK_KEYWORDS:
                keywords.extend(STOCK_KEYWORDS[stock_symbol][:2])  # Add up to 2 aliases
            
            # Limit to 5 keywords max
            keywords = keywords[:5]
            
            timeframe = f'now {days}-d'
            pytrends.build_payload(keywords, timeframe=timeframe, geo='IN')
            
            interest = pytrends.interest_over_time()
            
            if interest.empty:
                return {'available': False, 'signal': 0, 'trend': 'unknown'}
            
            # Calculate trend signal
            recent_avg = interest[keywords[0]].tail(2).mean()
            earlier_avg = interest[keywords[0]].head(2).mean()
            
            if earlier_avg > 0:
                trend_change = (recent_avg - earlier_avg) / earlier_avg * 100
            else:
                trend_change = 0
            
            # Determine trend direction
            if trend_change > 20:
                trend = 'rising_fast'
                signal = 0.3  # High interest can mean euphoria (be cautious)
            elif trend_change > 5:
                trend = 'rising'
                signal = 0.1
            elif trend_change < -20:
                trend = 'falling_fast'
                signal = -0.2  # Falling interest might mean capitulation
            elif trend_change < -5:
                trend = 'falling'
                signal = -0.1
            else:
                trend = 'stable'
                signal = 0
            
            return {
                'available': True,
                'signal': signal,
                'trend': trend,
                'change_pct': trend_change,
                'current_interest': recent_avg,
                'keywords': keywords
            }
            
        except Exception as e:
            return {'available': False, 'signal': 0, 'trend': 'error', 'error': str(e)[:50]}
    
    # ==============================================
    # COMBINED SENTIMENT ANALYSIS
    # ==============================================
    
    def analyze_all_sources(self, stock_symbol: str) -> Dict:
        """
        Fetch and analyze sentiment from all sources with temporal decay,
        event classification, and source disagreement detection.

        Returns:
            Comprehensive sentiment analysis dictionary including:
            - source_disagreement: std of source scores (>0.2 = high disagreement)
            - confidence_penalty: reduction applied due to disagreement
            - event_breakdown: count per event type across all articles
        """
        results = {
            'stock': stock_symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'combined_sentiment': 0,
            'combined_label': 'neutral',
            'confidence': 0,
            'article_count': 0,
            'all_items': [],
            'source_disagreement': 0.0,
            'confidence_penalty': 0.0,
            'event_breakdown': {},
        }

        all_sentiment_items = []
        event_counts: Dict[str, int] = {}  # For event breakdown pie chart

        # Build effective keyword list (dynamic for unknown stocks)
        if stock_symbol.upper() not in STOCK_KEYWORDS:
            effective_keywords = self._build_dynamic_keyword_map(stock_symbol.upper())
        else:
            effective_keywords = STOCK_KEYWORDS[stock_symbol.upper()]

        # ------------------------------------------------------------------ #
        # 1. RSS News  (Base weight: 30%)
        # ------------------------------------------------------------------ #
        rss_articles = self.fetch_rss_news(stock_symbol, max_articles=30)
        rss_sentiments = []
        rss_weighted_sum = 0.0
        rss_weight_total = 0.0

        for article in rss_articles:
            text = f"{article['title']} {article['summary']}"
            label, score = self.analyze_text(text)

            if label == 'positive':
                sentiment_value = score
            elif label == 'negative':
                sentiment_value = -score
            else:
                sentiment_value = 0.0

            # Temporal decay weight
            pub_date = article.get('date', datetime.now())
            if isinstance(pub_date, str):
                try:
                    pub_date = datetime.fromisoformat(pub_date)
                except Exception:
                    pub_date = datetime.now()
            t_weight = _temporal_weight(pub_date)

            # Event classification weight
            event_type, e_weight = _classify_article_event(article['title'], article.get('summary', ''))
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            article_weight = t_weight * e_weight
            rss_weighted_sum += sentiment_value * article_weight
            rss_weight_total += article_weight

            rss_sentiments.append({
                'text': article['title'][:100],
                'source': article['source'],
                'sentiment': label,
                'score': score,
                'value': sentiment_value,
                'event_type': event_type,
                'temporal_weight': round(t_weight, 3),
                'date': pub_date.isoformat() if hasattr(pub_date, 'isoformat') else str(pub_date),
            })

            all_sentiment_items.append({
                'Date': str(pub_date)[:16],
                'Source': f"RSS ({article['source']})",
                'Text': article['title'][:100] + "...",
                'Label': label,
                'Score': f"{score:.2f}",
                'Event': event_type,
            })

        rss_avg = rss_weighted_sum / (rss_weight_total + 1e-8) if rss_sentiments else 0.0

        results['sources']['rss'] = {
            'available': len(rss_sentiments) > 0,
            'count': len(rss_sentiments),
            'average_sentiment': float(rss_avg),
            'weight': 0.30,
            'articles': rss_sentiments[:10],
        }

        # ------------------------------------------------------------------ #
        # 2. NewsAPI  (Base weight: 25%)
        # ------------------------------------------------------------------ #
        newsapi_articles = self.fetch_newsapi_articles(stock_symbol, max_articles=20)
        newsapi_sentiments = []
        newsapi_weighted_sum = 0.0
        newsapi_weight_total = 0.0

        for article in newsapi_articles:
            text = f"{article['title']} {article['summary']}"
            label, score = self.analyze_text(text)

            if label == 'positive':
                sentiment_value = score
            elif label == 'negative':
                sentiment_value = -score
            else:
                sentiment_value = 0.0

            pub_date = article.get('date', datetime.now())
            if isinstance(pub_date, str):
                try:
                    pub_date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                except Exception:
                    pub_date = datetime.now()
            t_weight = _temporal_weight(pub_date)

            event_type, e_weight = _classify_article_event(article['title'], article.get('summary', ''))
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            article_weight = t_weight * e_weight
            newsapi_weighted_sum += sentiment_value * article_weight
            newsapi_weight_total += article_weight

            newsapi_sentiments.append({
                'text': article['title'][:100],
                'source': article['source'],
                'sentiment': label,
                'score': score,
                'value': sentiment_value,
                'event_type': event_type,
                'temporal_weight': round(t_weight, 3),
            })

            all_sentiment_items.append({
                'Date': str(pub_date)[:16],
                'Source': article.get('source', 'NewsAPI'),
                'Text': text[:100] + "...",
                'Label': label,
                'Score': f"{score:.2f}",
                'Event': event_type,
            })

        newsapi_avg = newsapi_weighted_sum / (newsapi_weight_total + 1e-8) if newsapi_sentiments else 0.0

        results['sources']['newsapi'] = {
            'available': len(newsapi_sentiments) > 0,
            'count': len(newsapi_sentiments),
            'average_sentiment': float(newsapi_avg),
            'weight': 0.25,
            'articles': newsapi_sentiments[:10],
        }

        # ------------------------------------------------------------------ #
        # 3. Reddit  (Base weight: 25%)
        # ------------------------------------------------------------------ #
        reddit_posts = self.fetch_reddit_posts(stock_symbol, max_posts=20)
        reddit_sentiments = []
        reddit_weighted_sum = 0.0
        reddit_weight_total = 0.0

        for post in reddit_posts:
            text = f"{post['title']} {post['summary']}"
            label, score = self.analyze_text(text)

            if label == 'positive':
                sentiment_value = score
            elif label == 'negative':
                sentiment_value = -score
            else:
                sentiment_value = 0.0

            pub_date = post.get('date', datetime.now())
            t_weight = _temporal_weight(pub_date)

            event_type, e_weight = _classify_article_event(post['title'], post.get('summary', ''))
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            # Engagement boost: Reddit upvotes add credibility (max 50% boost)
            engagement_boost = min(post.get('score', 0) / 100, 0.5)
            article_weight = t_weight * e_weight * (1 + engagement_boost)
            reddit_weighted_sum += sentiment_value * article_weight
            reddit_weight_total += article_weight

            reddit_sentiments.append({
                'text': post['title'][:100],
                'source': post['source'],
                'sentiment': label,
                'score': score,
                'value': sentiment_value,
                'event_type': event_type,
                'temporal_weight': round(t_weight, 3),
                'engagement': post.get('score', 0),
            })

            all_sentiment_items.append({
                'Date': str(pub_date)[:16],
                'Source': f"Reddit ({post['source']})",
                'Text': post['title'][:100] + "...",
                'Label': label,
                'Score': f"{score:.2f}",
                'Event': event_type,
            })

        reddit_avg = reddit_weighted_sum / (reddit_weight_total + 1e-8) if reddit_sentiments else 0.0

        results['sources']['reddit'] = {
            'available': len(reddit_sentiments) > 0,
            'count': len(reddit_sentiments),
            'average_sentiment': float(reddit_avg),
            'weight': 0.25,
            'posts': reddit_sentiments[:10],
        }

        # ------------------------------------------------------------------ #
        # 4. Google Trends  (Base weight: 20%)
        # ------------------------------------------------------------------ #
        trends_data = self.fetch_google_trends(stock_symbol)
        trends_signal = trends_data.get('signal', 0)

        results['sources']['google_trends'] = {
            'available': trends_data.get('available', False),
            'trend': trends_data.get('trend', 'unknown'),
            'signal': trends_signal,
            'change_pct': trends_data.get('change_pct', 0),
            'weight': 0.20,
        }

        # ------------------------------------------------------------------ #
        # WEIGHTED ENSEMBLE CALCULATION
        # ------------------------------------------------------------------ #

        base_weights = {'rss': 0.30, 'newsapi': 0.25, 'reddit': 0.25, 'trends': 0.20}
        source_avail_map = {
            'rss': results['sources']['rss']['available'],
            'newsapi': results['sources']['newsapi']['available'],
            'reddit': results['sources']['reddit']['available'],
            'trends': results['sources']['google_trends']['available'],
        }

        total_available_weight = sum(w for k, w in base_weights.items() if source_avail_map.get(k, False))

        if total_available_weight == 0:
            results['combined_sentiment'] = 0
            results['combined_label'] = 'neutral'
            results['confidence'] = 0
            results['event_breakdown'] = event_counts
            return results

        adjusted_weights = {
            k: (w / total_available_weight if source_avail_map.get(k, False) else 0.0)
            for k, w in base_weights.items()
        }

        combined = (
            rss_avg * adjusted_weights['rss']
            + newsapi_avg * adjusted_weights['newsapi']
            + reddit_avg * adjusted_weights['reddit']
            + trends_signal * adjusted_weights['trends']
        )

        results['combined_sentiment'] = float(combined)

        # ------------------------------------------------------------------ #
        # SOURCE DISAGREEMENT SIGNAL (4B)
        # ------------------------------------------------------------------ #
        available_scores = []
        if source_avail_map['rss']:
            available_scores.append(rss_avg)
        if source_avail_map['newsapi']:
            available_scores.append(newsapi_avg)
        if source_avail_map['reddit']:
            available_scores.append(reddit_avg)
        if source_avail_map['trends']:
            available_scores.append(trends_signal)

        if len(available_scores) >= 2:
            disagreement_std = float(np.std(available_scores))
        else:
            disagreement_std = 0.0

        if disagreement_std > 0.2:
            confidence_penalty = min(disagreement_std * 0.5, 0.4)
        else:
            confidence_penalty = 0.0

        results['source_disagreement'] = round(disagreement_std, 3)
        results['confidence_penalty'] = round(confidence_penalty, 3)

        # ------------------------------------------------------------------ #
        # LABEL, CONFIDENCE, EVENT BREAKDOWN
        # ------------------------------------------------------------------ #
        if combined > 0.15:
            results['combined_label'] = 'bullish'
        elif combined > 0.05:
            results['combined_label'] = 'slightly_bullish'
        elif combined < -0.15:
            results['combined_label'] = 'bearish'
        elif combined < -0.05:
            results['combined_label'] = 'slightly_bearish'
        else:
            results['combined_label'] = 'neutral'

        total_articles = len(rss_sentiments) + len(newsapi_sentiments) + len(reddit_sentiments)
        results['article_count'] = total_articles

        # Confidence: based on article volume + signal strength − disagreement penalty
        base_confidence = min(total_articles / 25, 1.0) * min(abs(combined) * 5, 1.0)
        results['confidence'] = max(0.0, base_confidence - confidence_penalty)

        results['all_items'] = all_sentiment_items
        results['event_breakdown'] = event_counts

        return results
    
    def get_sentiment_for_model(self, stock_symbol: str) -> Dict:
        """
        Get sentiment features formatted for ML model input.
        
        Args:
            stock_symbol: Stock symbol
        
        Returns:
            Dictionary with sentiment features for model
        """
        analysis = self.analyze_all_sources(stock_symbol)
        
        # Convert to model features
        features = {
            'sentiment_score': analysis['combined_sentiment'],
            'sentiment_label': analysis['combined_label'],
            'sentiment_confidence': analysis['confidence'],
            'rss_sentiment': analysis['sources'].get('rss', {}).get('average_sentiment', 0),
            'reddit_sentiment': analysis['sources'].get('reddit', {}).get('average_sentiment', 0),
            'trends_signal': analysis['sources'].get('google_trends', {}).get('signal', 0),
            'article_count': analysis['article_count'],
            'data_quality': 'high' if analysis['article_count'] > 10 else 'medium' if analysis['article_count'] > 5 else 'low',
            'source_disagreement': analysis.get('source_disagreement', 0.0),
            'confidence_penalty': analysis.get('confidence_penalty', 0.0),
            'event_breakdown': analysis.get('event_breakdown', {}),
        }

        return features


# ==============================================
# CONVENIENCE FUNCTIONS
# ==============================================

_multi_sentiment_instance = None

def get_multi_sentiment_analyzer() -> MultiSourceSentiment:
    """Get singleton instance of multi-source sentiment analyzer."""
    global _multi_sentiment_instance
    if _multi_sentiment_instance is None:
        _multi_sentiment_instance = MultiSourceSentiment()
    return _multi_sentiment_instance


def analyze_stock_sentiment(stock_symbol: str) -> Dict:
    """
    Convenience function to analyze sentiment for a stock.
    
    Args:
        stock_symbol: Stock symbol (e.g., "RELIANCE", "TCS")
    
    Returns:
        Comprehensive sentiment analysis
    """
    analyzer = get_multi_sentiment_analyzer()
    return analyzer.analyze_all_sources(stock_symbol)


def get_sentiment_features(stock_symbol: str) -> Dict:
    """
    Get sentiment features for ML model integration.
    
    Args:
        stock_symbol: Stock symbol
    
    Returns:
        Dictionary of sentiment features
    """
    analyzer = get_multi_sentiment_analyzer()
    return analyzer.get_sentiment_for_model(stock_symbol)
