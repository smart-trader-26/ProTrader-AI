"""
AI Analysis utilities.
Gemini integration and fallback analysis generation.
"""

import streamlit as st

from config.settings import GEMINI_API_KEY, DEEPSEEK_API_KEY, ModelConfig, TradingConfig

# Optional Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# Optional OpenAI import for DeepSeek
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


def initialize_gemini():
    """Initialize Gemini API."""
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception:
        return None

def initialize_deepseek():
    """Initialize DeepSeek API client."""
    if not OPENAI_AVAILABLE or not DEEPSEEK_API_KEY:
        return None
    try:
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    except Exception:
        return None


def generate_ai_analysis(stock_symbol: str, current_price: float, 
                         predicted_prices, metrics: dict, fundamentals: dict,
                         sentiment_summary: dict, technical_indicators: dict,
                         volatility_data, fusion_weights: dict = None,
                         fii_dii_data=None, vix_data=None, patterns: list = None) -> str:
    """
    Generate comprehensive AI analysis using DeepSeek (primary) and Gemini (secondary).
    Combines insights for best results.
    """
    deepseek_client = initialize_deepseek()
    gemini_model = initialize_gemini()
    
    # If no AI available, use fallback
    if not deepseek_client and not gemini_model:
        return generate_fallback_analysis(stock_symbol, current_price, predicted_prices, 
                                          metrics, sentiment_summary, technical_indicators)

    # Prepare Data Context
    # -------------------
    if not predicted_prices.empty:
        price_forecast_end = predicted_prices['Predicted Price'].iloc[-1]
        forecast_days = len(predicted_prices)
    else:
        price_forecast_end = current_price
        forecast_days = 0
    
    forecast_return = ((price_forecast_end - current_price) / current_price) * 100
    
    # Sentiment Text
    sentiment_text = "Neutral"
    if sentiment_summary:
        pos = sum(1 for s in sentiment_summary.values() for l, _ in s if l == 'positive')
        neg = sum(1 for s in sentiment_summary.values() for l, _ in s if l == 'negative')
        total = pos + neg
        if total > 0:
            ratio = pos / total
            if ratio > 0.6: sentiment_text = f"Bullish ({pos}/{total} positive)"
            elif ratio < 0.4: sentiment_text = f"Bearish ({neg}/{total} negative)"
            else: sentiment_text = f"Mixed ({pos} pos, {neg} neg)"

    # FII/DII Text
    fii_dii_text = "Data Unavailable"
    if fii_dii_data is not None and not fii_dii_data.empty:
        last = fii_dii_data.iloc[-1]
        fii = last.get('FII_Net', 0) / 1e7
        dii = last.get('DII_Net', 0) / 1e7
        fii_dii_text = f"FII: ₹{fii:+.2f}Cr | DII: ₹{dii:+.2f}Cr"
    
    # Patterns Text
    patterns_text = "No strong patterns."
    if patterns:
        p_list = [f"{p.get('Pattern')} ({p.get('Type')}, {p.get('Confidence')}% conf)" for p in patterns[:3]]
        patterns_text = ", ".join(p_list)

    # Prompt Construction
    # -------------------
    context = f"""
    STOCK ANALYSIS REQUEST: {stock_symbol}
    Current Price: ₹{current_price:,.2f}
    
    PREDICTIVE MODEL ({forecast_days} day horizon):
    - Target: ₹{price_forecast_end:,.2f} ({forecast_return:+.2f}%)
    - Direction: {'UP' if forecast_return > 0 else 'DOWN'}
    - Model Accuracy: {metrics.get('accuracy', 0):.1f}% (RMSE: {metrics.get('rmse', 0):.4f})
    
    TECHNICALS:
    - RSI: {technical_indicators.get('RSI', 'N/A')}
    - MACD: {technical_indicators.get('MACD_Histogram', 'N/A')}
    - Patterns: {patterns_text}
    - Volatility (20D): {technical_indicators.get('Volatility_20D', 0)*100:.2f}%
    
    FUNDAMENTALS:
    - P/E: {fundamentals.get('Forward P/E', 'N/A')}
    - P/B: {fundamentals.get('Price/Book', 'N/A')}
    - PEG: {fundamentals.get('PEG Ratio', 'N/A')}
    
    MARKET CONTEXT:
    - Institutional Flows: {fii_dii_text}
    - News Sentiment: {sentiment_text}
    - India VIX: {vix_data.iloc[-1]['Close'] if hasattr(vix_data, 'iloc') else 'N/A'}
    """
    
    system_prompt = """You are a senior hedge fund portfolio manager. 
    Analyze the provided stock data and give a high-precision trading verdict.
    
    KEY GUIDELINES:
    1. **Be Layman-Friendly:** Use simple language. Avoid jargon where possible. If using jargon, explain it briefly.
    2. **Point-Wise Only:** Do not write paragraphs. Use concise bullet points.
    3. **Resolve Contradictions:** If the AI Model predicts DOWN but Fundamentals/Technicals are BULLISH, weigh the evidence. If Model Accuracy is < 50%, TRUST THE TECHNICALS/FUNDAMENTALS more.
    4. **Verdict Consistency:** value in 'Quick Verdict' card might differ if Model Accuracy is low. You are the EXPERT. If model is unreliable, override it with your expert logic based on RSI/Flows/Patterns. 
    5. **Be Decisive:** Do not hedge. Give a clear direction.
    
    Structure the response in Markdown."""
    
    user_prompt = f"""{context}
    
    Provide your analysis in this EXACT format (keep it compact):
    
    ### 🎯 DeepSeek & Gemini Verdict: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
    **Confidence:** [High/Medium/Low]
    
    ### 🧠 Expert Rationale
    - **Alpha Signal:** [Why the model predicts this direction]
    - **pattern Recognition:** [Comment on the patterns/technicals]
    - **Macro Flow:** [Comment on FII/DII and VIX context]
    
    ### ⚠️ Critical Risks
    - [Key risk 1]
    - [Key risk 2]
    
    ### 💰 Execution Strategy
    - **Entry Zone:** ₹[Specific Price Range]
    - **Target 1:** ₹[Conservative Target]
    - **Target 2:** ₹[Aggressive Target]
    - **Stop Loss:** ₹[Specific Level]
    
    Do not use disclaimers. Assume I am a professional trader. Be concise."""
    
    # Execution Logic: Combined Best Result
    # -------------------------------------
    analysis_text = ""
    
    # 1. Try DeepSeek first (Primary Expert)
    if deepseek_client:
        try:
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False
            )
            analysis_text = response.choices[0].message.content
        except Exception as e:
            st.warning(f"DeepSeek error: {e}")
            
    # 2. If DeepSeek passed, use it. If failed (or not available), use Gemini.
    if analysis_text:
        return analysis_text
    
    # 3. Gemini Fallback
    if gemini_model:
        try:
            response = gemini_model.generate_content(user_prompt)
            return response.text
        except Exception as e:
             st.warning(f"Gemini error: {e}")
             
    return generate_fallback_analysis(stock_symbol, current_price, predicted_prices, 
                                      metrics, sentiment_summary, technical_indicators)


def generate_fallback_analysis(stock_symbol: str, current_price: float,
                               predicted_prices, metrics: dict,
                               sentiment_summary: dict, technical_indicators: dict) -> str:
    """
    Generate structured analysis without Gemini API (template-based fallback).
    
    Args:
        stock_symbol: Stock ticker symbol
        current_price: Current stock price
        predicted_prices: DataFrame with predicted prices
        metrics: Dictionary with model metrics
        sentiment_summary: Dictionary with sentiment data
        technical_indicators: Dictionary with technical indicators
    
    Returns:
        Markdown-formatted analysis string
    """
    if not predicted_prices.empty:
        price_forecast_end = predicted_prices['Predicted Price'].iloc[-1]
    else:
        price_forecast_end = current_price
    
    forecast_return = ((price_forecast_end - current_price) / current_price) * 100
    accuracy = metrics.get('accuracy', 50)
    
    # Determine verdict
    if accuracy < ModelConfig.LOW_CONFIDENCE_THRESHOLD:
        confidence = "Low Confidence"
    elif accuracy < ModelConfig.MEDIUM_CONFIDENCE_THRESHOLD:
        confidence = "Moderate Confidence"
    else:
        confidence = "Good Confidence"
    
    if forecast_return > 5 and accuracy > 60:
        verdict = "BUY 🟢"
        outlook = "Bullish"
    elif forecast_return > 2 and accuracy > 55:
        verdict = "HOLD (Positive Bias) 🟡"
        outlook = "Slightly Bullish"
    elif forecast_return < -5 and accuracy > 60:
        verdict = "SELL 🔴"
        outlook = "Bearish"
    elif forecast_return < -2 and accuracy > 55:
        verdict = "HOLD (Caution) 🟡"
        outlook = "Slightly Bearish"
    else:
        verdict = "HOLD 🟡"
        outlook = "Neutral"
    
    rsi = technical_indicators.get('RSI', 50)
    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    
    volatility = technical_indicators.get('Volatility_5D', 0)
    vol_text = "High volatility detected - position size accordingly" if volatility > 0.02 else "Normal volatility levels"
    
    return f"""
### 🎯 VERDICT: {verdict}
**{confidence}** | Model Accuracy: {accuracy:.1f}%

### 📊 OUTLOOK
- **Short-term:** {outlook} | Predicted {forecast_return:+.1f}% move
- **RSI Signal:** {rsi_signal} ({rsi:.1f})

### 💡 KEY INSIGHT
The hybrid AI model (XGBoost + GRU) predicts a {'positive' if forecast_return > 0 else 'negative'} return over the forecast period. {'However, model accuracy is below 55%, suggesting low predictive confidence.' if accuracy < 55 else 'Model shows reasonable directional accuracy on test data.'}

### ⚠️ RISK FACTORS
- Model predictions are probabilistic, not guarantees
- {vol_text}
- External market factors may override technical signals

*Analysis generated using template mode.*
"""


def generate_recommendation(predicted_prices, current_price: float, 
                            accuracy: float, avg_sentiment: float) -> tuple:
    """
    Generate investment recommendation based on predictions.
    
    Args:
        predicted_prices: DataFrame with predicted prices
        current_price: Current stock price
        accuracy: Model directional accuracy
        avg_sentiment: Average sentiment score
    
    Returns:
        Tuple of (recommendation_label, reason_text)
    """
    avg_prediction = predicted_prices['Predicted Price'].mean()
    price_change = ((avg_prediction - current_price) / current_price) * 100
    
    # Enhanced sentiment factor with confidence scaling
    sentiment_factor = 1 + (avg_sentiment * (accuracy/100))
    adjusted_change = price_change * sentiment_factor
    
    # Modified thresholds with confidence weighting
    confidence_weight = accuracy / 100
    
    if adjusted_change > TradingConfig.STRONG_BUY_THRESHOLD * confidence_weight and accuracy > 72:
        return "STRONG BUY", "High confidence in significant price increase"
    elif adjusted_change > TradingConfig.BUY_THRESHOLD * confidence_weight and accuracy > 65:
        return "BUY", "Good confidence in moderate price increase"
    elif adjusted_change > 0 and accuracy > 60:
        return "HOLD (Positive)", "Potential for slight growth"
    elif adjusted_change < TradingConfig.STRONG_SELL_THRESHOLD * confidence_weight and accuracy > 72:
        return "STRONG SELL", "High confidence in significant price drop"
    elif adjusted_change < TradingConfig.SELL_THRESHOLD * confidence_weight and accuracy > 65:
        return "SELL", "Good confidence in moderate price drop"
    elif adjusted_change < 0 and accuracy > 60:
        return "HOLD (Caution)", "Potential for slight decline"
    else:
        return "HOLD", "Unclear direction - consider other factors"
