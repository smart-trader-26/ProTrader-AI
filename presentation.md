---
marp: true
theme: default
paginate: true
---

# Slide 1: Problem Statement & Research Gap
**(LO 6.1: Define a problem statement along with the research gap)**

* **Problem Statement**: Financial markets like the NSE are highly complex, volatile, and influenced by a multitude of factors (technicals, sentiment, institutional flows). Single-domain ML models often fail to capture this complexity dynamically.
* **Traditional Approach Limitations**: Standard time-series models (e.g., ARIMA) or single static models (e.g., simplistic Random Forests) fail to adapt to abrupt market regime changes and often ignore unstructured data like multilingual news sentiment.
* **Research Gap**: There is a critical lack of dynamic ensemble frameworks capable of automatically adjusting their reliance on distinct specialized models based on real-time market uncertainty. There is a need for a solution that simultaneously leverages heterogeneous data sources (technicals, sentiment, and systemic volatility) under a unified probabilistic framework.

---

# Slide 2: Recent ML Techniques vs. Traditional Techniques
**(LO 6.2: Identify the recent ML technique alternative to traditional techniques)**

* **Traditional Techniques**: Fixed-weight model ensembles, basic linear regression, or simplistic technical rules (like Moving Average crossovers).
* **Recent ML Alternatives (Our 5-Model Ensemble + Bayesian Fusion)**:
  * **Gradient-Boosting & Deep-Learning Hybrid**: Integrates top-tier tree models (**XGBoost, LightGBM, and CatBoost**) to handle structured tabular data seamlessly, operating in parallel with a complex **LSTM-GRU neural architecture** for deep sequential time-series memory.
  * **Ridge Meta-Stacker**: Out-of-fold predictions from these models (alongside ARIMA/Prophet baselines) feed into a Ridge regression meta-stacker utilizing **isotonic probability calibration** for unified accuracy.
  * **Bayesian Dynamic Fusion**: Actively re-weights 3 domain-specific experts (GRU Technical, Dense NN Sentiment, MLP Volatility) based on exponential negative squared uncertainty (σ²).

---

# Slide 3: Designing the Solution & Tools Used
**(LO 6.1: Choose ML technique and tool, use it to design a solution)**

* **Key ML Tools Chosen**: Python, TensorFlow (for LSTM-GRU/MLP neural networks), standalone libraries for XGBoost/LightGBM/CatBoost, Hugging Face Transformers (`DistilRoBERTa-Financial` for NLP), and Streamlit.
* **Solution Architecture Flow**:
  1. **Regime Classifier**: Uses **Hurst Exponent (R/S analysis)** before prediction to detect current market conditions (Trending vs. Mean-reverting vs. Random Walk).
  2. **Feature Engineering Pipeline**: Computation of **27 completely stationary features** spanning:
     * Core & Advanced Technicals (Log Returns, RSI, MACD)
     * Real-time Institutional Flows (FII/DII logic)
     * Systemic Volatility Indices (India VIX)
  3. **Multi-Model Ingestion**: The 27 features load into the 5-model ensemble stack, are fused via the Ridge meta-learner, and supervised by the Bayesian uncertainty layer to yield a final adaptive prediction.

---

# Slide 4: Advanced Implementations in the ML Framework
**(LO 6.2: Use it to design a solution with the help of ML tools)**

* **4-Source Sentiment Aggregation**: Evaluates text via NLP utilizing real-time data from RSS Feeds (30%), NewsAPI (25%), Reddit (25%), and Google Trends (20%). Uses event-classification multipliers (e.g., 2.0x weight for earnings calls) and temporal decay formulas.
* **Algorithmic Chart Pattern Detection**: Integrates multi-timeframe pattern recognition using **ZigZag algorithms and SciPy peak detection (orders 3, 5, 7)** augmented with Roboflow Vision AI integration to find specific archetypes (e.g. Double Tops).
* **Robust Explainability (XAI)**:
  * **SHAP Values** reveal explicit per-feature attribution (what individually drives the model's output).
  * Clear visual tracking of real-time unquantified variance (σ²) across the ML experts.

---

# Slide 5: Validation Methodology & Outcomes
**(LO 6.1: Effectively present the solution)**

* **Validation Methodology Engine**: Rigorous walk-forward validation and backtesting eliminating look-ahead bias, benchmarked directly against NIFTY-50 Buy-and-Hold and generalized Moving Average crossover signals, considering actual transaction costs.
* **Quantifiable Evaluation Metrics**:
  * Profitability & Risk metrics: Sharpe Ratio, Max Drawdown, Fractional Kelly sizing, and absolute Win Rate.
  * Model specific KPIs: Direction Accuracy (% correct price direction) and Root Mean Square Error (RMSE).
* **Ensuring Statistical Significance**: Uses Binomial tests for measuring directional advantage over 50% random chance, Paired t-tests (Model RMSE vs Random Walk baseline), and Monte Carlo Resampling for portfolio robustness.

---

# Slide 6: Conclusion
**(LO 6.1 & 6.2: Presentation Outcomes)**

* **Mathematical Adaptability**: By implementing the Bayesian Dynamic Fusion framework atop the Ridge Stack, the model mathematically adapts to distinct market regimes, shifting priority toward the highest-performing expert in real-time rather than sticking to static generalized assumptions.
* **Actionable Accuracy**: The dense combination of a 5-model hybrid (incorporating XGBoost, LightGBM, CatBoost, LSTM-GRU) alongside multi-source NLP sentiment and institutional tracking completely resolves the highlighted research gap.
* **Final Verdict**: Utilizing computationally advanced, multimodal machine learning ensembling offers a statistically significant, profitable edge over traditional singular market strategies. It successfully merges unstructured sentiment analysis, rigorous technical feature engineering, and uncertainty-based volatility.
