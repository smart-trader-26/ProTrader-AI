# Research Paper — Objective Mapping

**Topic**: Integrating Multi-Source Data with Sentiment Analysis and Language Models to Enhance Stock Market Decision Making

This document maps each research objective 1:1 to what the ProTrader AI project implements, identifying specific evidence, code locations, and framing suggestions for the paper.

---

## Objective 1

> *To investigate and develop a comprehensive model that integrates diverse data sources—such as financial metrics, news articles, social media sentiment, and macroeconomic factors—aiming to enhance the accuracy of stock market predictions by providing a holistic analysis.*

### What the Project Implements

| Data Category | Sources | Code Location |
|--------------|---------|---------------|
| **Financial metrics** | OHLCV (daily), P/E ratio, ROE, debt-to-equity, market cap, EPS, analyst target prices | `data/stock_data.py` |
| **News articles** | NewsAPI (global), Moneycontrol RSS, Economic Times RSS, LiveMint RSS, Business Standard RSS, Google News RSS | `data/multi_sentiment.py`, `data/news_sentiment.py` |
| **Social media sentiment** | Reddit — r/IndianStockMarket, r/DalalStreetTalks, r/IndiaInvestments, r/indianstreetbets | `data/multi_sentiment.py` |
| **Macroeconomic factors** | India VIX (fear/volatility index), FII/DII institutional flows, Google Trends (retail investor interest proxy) | `data/vix_data.py`, `data/fii_dii.py`, `data/multi_sentiment.py` |

### Integration Mechanism

All data streams converge into a **27-feature vector** fed to the hybrid prediction model (`models/hybrid_model.py`):
- 5 core technical + 9 enhanced technical + 4 advanced technical
- 3 sentiment features (multi-source aggregated score, confidence, base score)
- 4 institutional features (FII/DII net flows, 5-day rolling averages)
- 2 volatility features (VIX normalized, VIX change rate)

The **multi-source sentiment aggregator** (`data/multi_sentiment.py`) fuses 4 independent data streams with weighted averaging (RSS 30%, NewsAPI 25%, Reddit 25%, Google Trends 20%), producing a single confidence-weighted sentiment signal.

### Paper Framing

The paper can show an **ablation study** comparing model performance with different subsets of data sources (e.g., technical-only baseline vs. technical + sentiment vs. full 4-source model), demonstrating that each additional data stream measurably improves direction prediction accuracy. The India VIX and FII/DII inputs are particularly novel as macroeconomic proxies specific to the Indian market context.

---

## Objective 2

> *To design a hybrid framework that combines large language models (LLMs), sentiment analysis techniques, and conventional financial indicators, focusing on optimizing deep learning architectures for real-time stock market predictions.*

### What the Project Implements

#### Large Language Models

| Model | Role | Code Location |
|-------|------|---------------|
| **DistilRoBERTa-Financial** (`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`) | Classifies news text as positive/negative/neutral with confidence scores | `data/multi_sentiment.py`, `data/news_sentiment.py` |
| **Google Gemini** | Generates natural-language AI commentary on analysis results | `ui/ai_analysis.py` |

DistilRoBERTa-Financial is a transformer-based LLM fine-tuned on financial corpora, applied here via the HuggingFace Transformers library (`transformers>=4.30.0`).

#### Sentiment Analysis Techniques

- **Event-type weighting**: Keyword classification assigns amplification multipliers (earnings 2.0×, regulatory 1.8×, dividend 1.5×, management 1.3×)
- **Temporal decay weighting**: `w = exp(-λ × days_old)` where λ=0.5, reducing the influence of stale articles
- **Confidence-weighted aggregation**: Article count and source availability factor into final confidence score

#### Conventional Financial Indicators

All computed in `utils/technical_indicators.py`:
- **Momentum**: RSI (14), MACD (12/26/9), Williams %R
- **Volatility**: ATR (14), Bollinger Bands (%B), rolling standard deviation
- **Volume**: OBV slope, Chaikin Money Flow (CMF_20), Volume/MA ratio
- **Trend**: Multi-timeframe log returns (2D, 5D, 10D, 20D), MA divergence

#### Deep Learning Architectures

| Architecture | Specs | Purpose |
|-------------|-------|---------|
| **GRU (Hybrid Model)** | 128→64→32 units, BatchNorm, 0.3 dropout, 30-day lookback | Primary sequential price predictor |
| **GRU (Technical Expert)** | 128→64→32 units, MinMaxScaler, lr=0.001 | Technical pattern expert in fusion framework |
| **Dense NN (Sentiment Expert)** | 64→32→16 units, ReLU, BatchNorm, 0.3 dropout | Sentiment signal expert in fusion framework |
| **MLP (Volatility Expert)** | Variable depth, VIX + stock volatility inputs | Volatility regime expert in fusion framework |
| **XGBoost** | 100 trees, max_depth=3, lr=0.05 | Ensemble component in hybrid model |

#### Optimization

- Walk-forward validation prevents look-ahead bias
- BatchNormalization and dropout (0.3) for regularization
- Hurst exponent regime detection modulates prediction confidence
- Bayesian weight formula automatically optimizes expert contribution: `w_i = exp(-σ²_i) / Σ exp(-σ²_j)`
- Optuna integration available in `models/optimizer.py` for hyperparameter search

### Paper Framing

The Bayesian multi-expert fusion (`models/fusion_framework.py`) is the core architectural novelty — it differs from standard static ensembles by adapting in real-time to each expert's recent error variance. The paper can position this as **architecture optimization through uncertainty-aware dynamic weighting**, and compare it against fixed-weight baselines (equal weighting, accuracy-weighted voting) to demonstrate improvement.

---

## Objective 3

> *To establish and evaluate an explainable AI framework capable of delivering transparent and interpretable predictions in stock selection, analyzing the impact of sentiment data noise and exploring the temporal dynamics between sentiment shifts and stock price movements.*

### What the Project Implements

#### Explainable AI Framework

| Method | What It Reveals | Code Location |
|--------|----------------|---------------|
| **SHAP (SHapley Additive exPlanations)** | Per-feature attribution for XGBoost — shows which of the 27 features pushed prediction up/down | `models/hybrid_model.py` |
| **Uncertainty quantification** | Each expert tracks σ² over its last 10 predictions — model exposes its own confidence | `models/fusion_framework.py` |
| **Expert weight visualization** | Real-time display of Technical / Sentiment / Volatility expert weights | `app.py` (Dynamic Fusion tab) |
| **Per-source sentiment breakdown** | Individual sentiment scores and article counts from RSS, NewsAPI, Reddit, and Google Trends | `app.py` (Multi-Source Sentiment tab) |
| **Trade setup transparency** | Entry, stop-loss, and target prices explicitly derived from ATR and Fibonacci levels | `utils/risk_manager.py` |
| **Hurst exponent display** | Market regime value and classification shown alongside each prediction | `models/hybrid_model.py` |
| **Pattern explanation** | Each detected chart pattern shows confidence score, volume confirmation, and price target | `models/visual_analyst.py` |

#### Sentiment Data Noise — How It's Addressed

The system has three specific noise-reduction mechanisms:

1. **Temporal decay** (`data/multi_sentiment.py`):
   - Formula: `w = exp(-0.5 × days_old)`
   - Same-day article weight: 1.0; 3-day-old article weight: 0.22; 7-day-old weight: 0.03
   - Prevents stale sentiment from distorting real-time signals

2. **Confidence weighting** (`data/multi_sentiment.py`):
   - Confidence = f(article count, source availability)
   - Low article count → low confidence → reduced influence on final score
   - If multi-source finds 0 articles, system enriches with Yahoo Finance news before scoring

3. **Source weight tuning** (`data/multi_sentiment.py`):
   - RSS feeds (authoritative financial sources) weighted highest at 30%
   - Reddit (high noise, high recency value) weighted at 25%
   - Google Trends (indirect proxy) weighted lowest at 20%

#### Temporal Dynamics Between Sentiment and Price

- **Multi-timeframe returns** (2D, 5D, 10D, 20D) in the 27-feature set capture lagged price responses
- **Event classification** identifies when sentiment spikes are earnings/regulatory-driven (typically precede large price moves) vs. general chatter
- **Hurst exponent tracking** over time shows how sentiment regime persistence changes — trending regimes with positive sentiment have different dynamics than mean-reverting regimes with the same sentiment score

### Paper Framing

The paper can present the XAI framework as addressing the "black box" criticism of ML-based finance models. Three complementary interpretability dimensions:
1. **Feature-level attribution** (SHAP): What drove this specific prediction?
2. **Model-level confidence** (uncertainty quantification): How much does the model trust its own output?
3. **Data-level transparency** (per-source sentiment breakdown): Where did the sentiment signal come from?

For temporal dynamics, the paper can show lag analysis — plotting sentiment shifts (from the aggregated 4-source score) against subsequent 2D/5D price returns to empirically establish lead/lag relationships.

---

## Objective 4

> *To evaluate and validate the result and practical application of the proposed model across different stock markets, exploring transfer learning techniques for cross-market applications and validating the model against benchmark datasets and real-world market data.*

### What the Project Implements

#### Validation Against Benchmarks

| Benchmark | Description | Code Location |
|-----------|-------------|---------------|
| NIFTY 50 buy-and-hold | Market-level passive baseline | `models/backtester.py` |
| MA Crossover (20/50) | Standard technical strategy baseline | `models/backtester.py` |
| 52-Week Momentum | Breakout-based momentum baseline | `models/backtester.py` |

All benchmarks use the same NSE transaction cost model (0.1% round-trip) for fair comparison.

#### Statistical Validation

| Test | Purpose | Implementation |
|------|---------|----------------|
| **Binomial test** | Direction accuracy vs. 50% random baseline (null hypothesis: model is no better than a coin flip) | `models/backtester.py` |
| **Paired t-test** | Model RMSE vs. random walk RMSE | `models/backtester.py` |
| **Bootstrap CI (95%)** | Confidence interval on Sharpe ratio difference (model vs. benchmark) | `models/backtester.py` |
| **Cohen's d** | Effect size — distinguishes statistical from practical significance | `models/backtester.py` |
| **Monte Carlo simulation** | 1,000 path resampling for strategy robustness under different market sequences | `models/backtester.py` |

Walk-forward validation is used throughout — training always precedes test data, no look-ahead bias.

#### Transfer Learning

The project uses **pre-trained LLM transfer** as its cross-domain learning mechanism:

- **DistilRoBERTa-Financial** was pre-trained on English financial news corpora (SEC filings, Reuters, Bloomberg) and is applied **zero-shot** to Indian financial news from Moneycontrol, Economic Times, and LiveMint — no fine-tuning on Indian data required
- This is cross-domain transfer: from US/global financial language to Indian market-specific language and terminology
- **Roboflow Vision API** (pre-trained on candlestick pattern images) is similarly applied zero-shot to NSE stock charts

#### Cross-Market Generalization

The system is architecturally market-agnostic in several components:

| Component | Generalization Mechanism |
|-----------|------------------------|
| Technical indicators | RSI, MACD, ATR, OBV, Williams %R — compute identically on any OHLCV series |
| Hurst exponent | R/S analysis applies to any price time series |
| DistilRoBERTa-Financial | Applied zero-shot to any financial news text |
| Dynamic keyword mapping | Extracts company-specific keywords from `yfinance.Ticker.info` for any ticker worldwide |
| Bayesian fusion | Expert weighting formula is market-independent |

Current limitation: FII/DII integration and India VIX are NSE-specific. Cross-market validation would replace these with market-specific institutional flow equivalents (e.g., S&P 500 flows for US markets).

#### Real-World Market Data

All data is fetched from live production APIs:
- Yahoo Finance (real OHLCV, real fundamentals)
- NSE India official API (real FII/DII)
- NewsAPI (real current financial news)
- Reddit live API (real community sentiment)
- Google Trends (real search volume)

No synthetic datasets — all backtesting uses historical real-market data.

### Paper Framing

The statistical validation section is one of the strongest aspects of the paper relative to typical ML finance papers that report only accuracy. The combination of binomial significance testing + bootstrapped Sharpe CI + Monte Carlo robustness testing provides a rigorous statistical case for the model's practical utility. The transfer learning angle can be framed as **zero-shot cross-domain NLP transfer** — demonstrating that a US financial corpus-trained LLM generalizes to Indian market news without any fine-tuning, with sentiment classification accuracy as the metric.

---

## Summary Table

| Objective | Coverage | Key Evidence | Strongest Claim |
|-----------|----------|-------------|----------------|
| **1 — Multi-source integration** | Full | 27-feature model, 4-source sentiment, FII/DII, VIX | Holistic fusion of 7+ independent data streams |
| **2 — Hybrid LLM + DL framework** | Full | DistilRoBERTa-Financial + GRU + XGBoost + Bayesian fusion | Novel uncertainty-adaptive expert weighting |
| **3 — Explainable AI + temporal dynamics** | Full | SHAP + uncertainty quant + temporal decay + per-source breakdown | 3-axis XAI framework for financial predictions |
| **4 — Evaluation + transfer learning** | Full (with caveat) | Monte Carlo + 5 statistical tests + zero-shot LLM transfer | Rigorous statistical validation beyond simple accuracy |

**Caveat for Objective 4**: Explicit multi-market comparative study (NSE vs. NYSE etc.) is not implemented. The transfer learning claim should be scoped as zero-shot NLP transfer from global financial corpora to Indian market text, not as a full multi-market neural architecture study.
