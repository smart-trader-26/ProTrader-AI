# ProTrader AI — AI-Powered Stock Analytics Platform

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00.svg)](https://tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-028cf0.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An advanced AI-powered stock prediction and analysis platform for **Indian markets (NSE)**, combining multi-source sentiment analysis, institutional flow tracking, Bayesian multi-expert fusion, and mathematical chart pattern detection into a single interactive dashboard.

> **For educational and research purposes only. Not financial advice.**

---

## 📋 Table of Contents

- [What Makes This Novel](#-what-makes-this-novel)
- [Data Sources](#-data-sources)
- [Model Architecture](#-model-architecture)
- [Explainability Framework](#-explainability-framework)
- [Backtesting & Validation](#-backtesting--validation)
- [Platform Features](#-platform-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Keys Setup](#-api-keys-setup)
- [Known Limitations](#-known-limitations)
- [Disclaimer](#-disclaimer)

---

## 🎯 What Makes This Novel

### 1. Bayesian Dynamic Fusion Framework

Three specialized neural networks, each tracking its own prediction uncertainty. Weights shift automatically to the most accurate expert:

```
w_i = exp(-σ²_i) / Σ exp(-σ²_j)
```

| Expert | Architecture | Input |
|--------|-------------|-------|
| Technical Expert | GRU (128→64→32) | 20 OHLCV-derived features |
| Sentiment Expert | Dense NN (64→32→16) | 8 multi-source sentiment features |
| Volatility Expert | MLP | India VIX + stock volatility features |

Unlike fixed-weight ensembles, this framework re-weights experts every prediction cycle based on their recent error variance (σ²).

---

### 2. 27-Feature Hybrid Model (XGBoost + GRU)

A 50/50 ensemble of XGBoost (100 trees, max_depth=3) and GRU (128→64→32) trained on 27 engineered features:

| Category | Count | Features |
|----------|-------|---------|
| **Core Technical** | 5 | Log Returns, Volatility_5D, RSI_Norm, Volume Ratio, MA Divergence |
| **Enhanced Technical** | 9 | MACD_Norm, MACD_Hist_Norm, Bollinger %B, ATR_Norm, OBV_Slope, 2D/5D/10D/20D Returns |
| **Advanced Technical** | 4 | Chaikin Money Flow (CMF_20), Williams %R, RSI Bearish Divergence, RSI Bullish Divergence |
| **Sentiment** | 3 | Base Sentiment Score, Multi-Source Score, Confidence |
| **Institutional** | 4 | FII Net (normalized), DII Net (normalized), FII 5D Rolling Avg, DII 5D Rolling Avg |
| **Volatility** | 2 | VIX Normalized, VIX Change Rate |

All features are stationary (log returns, oscillators, ratios) to avoid non-stationarity issues.

---

### 3. 4-Source Sentiment Aggregation with Temporal & Event Weighting

| Source | Weight | Details |
|--------|--------|---------|
| RSS Feeds | 30% | Moneycontrol, Economic Times, LiveMint, Business Standard, Google News |
| NewsAPI | 25% | Global financial news with dynamic stock keyword mapping |
| Reddit | 25% | r/IndianStockMarket, r/DalalStreetTalks, r/IndiaInvestments, r/indianstreetbets |
| Google Trends | 20% | Retail investor interest via search volume |

**Temporal decay**: `w = exp(-0.5 × days_old)` — same-day articles weighted 1.0, 3-day-old articles weighted 0.22.

**Event classification multipliers**:
- Earnings announcements: 2.0×
- Regulatory news: 1.8×
- Dividend announcements: 1.5×
- Management changes: 1.3×
- General news: 1.0×

NLP model: **DistilRoBERTa-Financial** (`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`), a transformer pre-trained on financial corpora, applied without fine-tuning.

---

### 4. Hurst Exponent Market Regime Detection

Uses R/S (Rescaled Range) analysis to classify market regimes before prediction:

```
H = slope of log(R/S) vs log(lag)

H > 0.55 → Trending market
H < 0.45 → Mean-reverting market
H ≈ 0.50 → Random walk
H with high volatility → Volatile regime
```

The regime classification modulates confidence in trend-following signals.

---

### 5. Multi-Timeframe Chart Pattern Detection

**ZigZag-based detection** (primary):
- Threshold: 3% price reversal for daily data
- O(n) time complexity, adaptive to volatility

**Scipy peak detection** (fallback):
- `scipy.signal.argrelextrema` with orders 3, 5, 7 for multi-scale scanning

**Patterns detected**: Double Top, Double Bottom, Head & Shoulders, Inverse Head & Shoulders, Ascending/Descending Channels, Bull/Bear Flags, Triangles, Wedges, Rounding Bottoms, Support/Resistance levels.

Volume confirmation: Breakout volume > 1.5× MA20, Reversal volume > 1.2× MA20.

**Roboflow Vision API** integration available for AI-based chart image classification.

---

### 6. FII/DII Data with 6-Source Fallback Chain

Official Foreign/Domestic Institutional Investor flow data with automatic fallbacks:
```
NSE API → nselib library → MoneyControl → Trendlyne → Gemini AI parsing → Manual input UI
```

---

## 📊 Data Sources

| Source | Type | Data |
|--------|------|------|
| Yahoo Finance (yfinance) | Market data | OHLCV, fundamentals (P/E, ROE, debt ratios, market cap) |
| NSE India | Institutional | FII/DII net flows (daily) |
| India VIX (^INDIAVIX) | Macroeconomic | Market fear/volatility index |
| NewsAPI | News | Global financial news aggregation |
| RSS Feeds (6 sources) | News | Indian financial news (Moneycontrol, ET, LiveMint, Business Standard, Google News) |
| Reddit (PRAW) | Social media | 4 Indian market subreddits |
| Google Trends (pytrends) | Retail sentiment | Search volume as retail interest proxy |
| Roboflow Vision API | Pattern data | AI chart pattern classification |

---

## 🔧 Model Architecture

### Hybrid Model Pipeline

```
Raw Market Data (OHLCV)
        │
        ▼
Feature Engineering (27 features)
        ├── Core Technical  (5): Log Returns, Volatility, RSI, Volume Ratio, MA Divergence
        ├── Enhanced Tech   (9): MACD, Bollinger %B, ATR, OBV Slope, Multi-timeframe Returns
        ├── Advanced Tech   (4): CMF_20, Williams %R, RSI Bull/Bear Divergence
        ├── Sentiment       (3): Base Score, Multi-Source Score, Confidence
        ├── Institutional   (4): FII/DII Net Normalized, 5D Rolling Averages
        └── Volatility      (2): VIX Normalized, VIX Change Rate
              │
        ┌─────┴──────┐
        ▼            ▼
   XGBoost       GRU Network
  (100 trees)  (128→64→32 units)
   max_depth=3   BatchNorm + 0.3 dropout
        │            │
        └─────┬───────┘
              ▼
       50/50 Ensemble
              │
              ▼
    Predicted Return + Regime Classification
```

### Dynamic Fusion Framework

```
                    ┌─────────────────┐
                    │   Stock Data    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Technical   │   │   Sentiment   │   │   Volatility  │
│    Expert     │   │    Expert     │   │    Expert     │
│  GRU (128→32) │   │ Dense (64→16) │   │    MLP        │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        │    σ²_tech        │    σ²_sent        │    σ²_vol
        └────────────────────┴────────────────────┘
                             │
                             ▼
              Bayesian Weight Calculator
           w_i = exp(-σ²_i) / Σ exp(-σ²_j)
                             │
                             ▼
                    Combined Prediction
```

---

## 🔍 Explainability Framework

| Method | What It Shows |
|--------|--------------|
| **SHAP Values** | Per-feature attribution — which of the 27 features drove the prediction up or down |
| **Uncertainty Quantification** | Per-expert σ² (mean squared error over last 10 predictions) — visible in Dynamic Fusion tab |
| **Expert Weight Visualization** | Real-time bar chart showing current Technical / Sentiment / Volatility weights |
| **Sentiment Source Breakdown** | Individual scores from each source (RSS, NewsAPI, Reddit, Trends) with article counts |
| **Trade Setup Transparency** | Entry, stop loss, and target prices with explicit ATR-based derivation and risk/reward ratio |
| **Hurst Exponent Display** | Current regime value and classification shown alongside prediction |
| **Pattern Explanation** | Each detected pattern shows location, confidence score, volume confirmation, and price target |

---

## 📈 Backtesting & Validation

### Benchmarks
- **NIFTY 50 buy-and-hold** (market baseline)
- **MA Crossover** (20-day / 50-day moving average signals)
- **52-Week Momentum** (breakout-based signals)

### Metrics
| Metric | Description |
|--------|-------------|
| Direction Accuracy | % of correct up/down predictions |
| RMSE | Root Mean Square Error of return predictions |
| Sharpe Ratio | Annualized risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| Win Rate | % of profitable trades |
| Profit Factor | Total gross wins / total gross losses |

### Statistical Significance Tests
- **Binomial test**: Direction accuracy vs. 50% random baseline
- **Paired t-test**: Model RMSE vs. random walk RMSE
- **Bootstrap 95% CI**: Confidence interval on Sharpe ratio difference
- **Cohen's d**: Effect size calculation
- **Monte Carlo simulation**: 1,000 path resampling for strategy robustness

### Transaction Costs
NSE round-trip cost of 0.1% applied (0.05% brokerage + 0.025% STT + 0.003% exchange charges).

Walk-forward validation used throughout — no look-ahead bias.

---

## 🖥️ Platform Features

| Tab | Description |
|-----|-------------|
| 📊 **Dashboard** | Main prediction output, direction accuracy chart, Gemini AI commentary |
| 🔬 **Dynamic Fusion** | Real-time expert weight visualization, per-expert uncertainty tracking |
| 📈 **Technicals & Risk** | Fibonacci levels, ATR, Kelly Criterion position sizing, trade setup calculator |
| 🏛️ **Fundamentals** | P/E, ROE, debt-to-equity, market cap, analyst target prices |
| 💼 **FII/DII Analysis** | Daily and cumulative institutional flow charts with 5D rolling averages |
| 📰 **Multi-Source Sentiment** | Per-source sentiment scores, article counts, event classification, temporal trends |
| 🛠️ **Backtest** | Strategy vs. benchmarks, equity curve, Monte Carlo CI, statistical p-values |
| 📐 **Pattern Analysis** | Detected chart patterns with confidence scores, targets, and volume confirmation |

---

## 📁 Project Structure

```
finance/
├── app.py                      # Main Streamlit application (~29,000 lines)
├── config/
│   └── settings.py             # API keys, model config, Indian holiday calendar
├── data/
│   ├── stock_data.py           # Yahoo Finance OHLCV + fundamentals
│   ├── fii_dii.py              # NSE FII/DII data with 6-source fallback chain
│   ├── vix_data.py             # India VIX + synthetic NIFTY-based fallback
│   ├── news_sentiment.py       # NewsAPI + FinBERT sentiment pipeline
│   └── multi_sentiment.py      # 4-source sentiment aggregator (~695 lines)
├── models/
│   ├── hybrid_model.py         # 27-feature XGBoost + GRU ensemble
│   ├── fusion_framework.py     # Bayesian multi-expert fusion framework
│   ├── technical_expert.py     # GRU-based technical price model
│   ├── sentiment_expert.py     # Dense NN for sentiment features
│   ├── volatility_expert.py    # MLP for VIX + volatility analysis
│   ├── visual_analyst.py       # ZigZag + scipy chart pattern detection
│   ├── backtester.py           # Vectorized backtesting with Monte Carlo
│   └── optimizer.py            # Optuna hyperparameter tuning
├── ui/
│   ├── charts.py               # Plotly interactive chart generation
│   └── ai_analysis.py          # Google Gemini AI commentary integration
├── utils/
│   ├── technical_indicators.py # RSI, MACD, ATR, OBV, Bollinger Bands, CMF
│   └── risk_manager.py         # Fibonacci levels, Kelly Criterion, ATR stop-loss
├── indian_stocks.csv           # NSE stock symbols list
├── .env                        # API keys (gitignored)
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/protrader-ai.git
cd protrader-ai
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional but Recommended)

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_key          # AI analysis commentary
NEWS_API_KEY=your_newsapi_key           # Enhanced news sentiment
REDDIT_CLIENT_ID=your_reddit_id         # Reddit social sentiment
REDDIT_CLIENT_SECRET=your_reddit_secret
ROBOFLOW_API_KEY=your_roboflow_key      # Vision-based pattern detection
```

The platform works without any API keys — it falls back gracefully using RSS feeds, Yahoo Finance news, and synthetic VIX.

### 3. Run

```bash
streamlit run app.py
```

---

## 🔑 API Keys Setup

### Gemini API (Free)
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Add to `.env`: `GEMINI_API_KEY=your_key`

### NewsAPI (Free tier — 100 requests/day)
1. Visit [newsapi.org/register](https://newsapi.org/register)
2. Sign up and get key
3. Add to `.env`: `NEWS_API_KEY=your_key`

### Reddit API (Free)
1. Visit [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Create a "script" type application
3. Add both `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` to `.env`

---

## ⚠️ Known Limitations

| Limitation | Detail |
|-----------|--------|
| **EOD data only** | Uses end-of-day prices; not suitable for intraday trading |
| **NSE API reliability** | FII/DII data may be unavailable when NSE website is down (6-source fallback chain mitigates this) |
| **India VIX availability** | Falls back to synthetic NIFTY volatility proxy when ^INDIAVIX is unavailable |
| **Training speed** | GRU training is slow on CPU; GPU recommended for faster turnaround |
| **Market scope** | Optimized for NSE India; other markets work but lack FII/DII and VIX integration |
| **NLP model** | DistilRoBERTa-Financial is applied zero-shot — not fine-tuned on Indian market-specific language |

---

## 📦 Core Dependencies

```
streamlit>=1.28.0
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
xgboost>=2.0.0
tensorflow>=2.13.0
transformers>=4.30.0
plotly>=5.15.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
requests>=2.31.0
feedparser>=6.0.0
praw>=7.7.0
pytrends>=4.9.0
google-generativeai>=0.3.0
shap>=0.42.0
```

Optional (conditional imports, platform works without):
`prophet`, `statsmodels`, `lightgbm`, `catboost`, `optuna`, `nselib`

---

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Do not use for real trading without extensive independent backtesting
- Always consult a SEBI-registered financial advisor before investing

---

## 📄 License

MIT License — free for personal and research use.

---

## 🙏 Credits

- **Market Data**: Yahoo Finance, NSE India
- **Sentiment NLP**: [DistilRoBERTa-Financial](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
- **AI Commentary**: Google Gemini
- **Pattern Detection**: SciPy signal processing, Roboflow Vision API
- **Backtesting**: Custom vectorized engine with NSE transaction cost model

---

**Version**: 4.0 | **Last Updated**: March 2026
