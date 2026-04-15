"""
ProTrader AI: Professional Stock Analytics Platform

Main Streamlit application entry point.
Combines hybrid AI models with technical, sentiment, and volatility analysis
for Indian equity markets.
"""

import datetime
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Import configuration
from config.settings import UIConfig, ModelConfig, TradingConfig

# Import data utilities
from data.stock_data import get_stock_data, get_fundamental_data, get_indian_stocks
from data.fii_dii import get_fii_dii_data, extract_fii_dii_features, fetch_fii_dii_apis_internal, render_manual_fii_dii_input
from data.news_sentiment import get_news, analyze_sentiment, filter_relevant_news
from data.vix_data import get_india_vix_data

# Import models
from models.hybrid_model import create_hybrid_model, hybrid_predict_prices
from models.fusion_framework import DynamicFusionFramework
from models.backtester import VectorizedBacktester

# Import utilities
from utils.technical_indicators import calculate_technical_indicators
from utils.risk_manager import RiskManager

# Import UI components
from ui.charts import (
    create_candlestick_chart,
    create_accuracy_comparison_chart,
    create_fii_dii_chart
)
from ui.ai_analysis import generate_ai_analysis


# ==============================================
# HINT / GLOSSARY HELPER
# ==============================================

def _hint(title: str, items: dict, icon: str = "💡"):
    """
    Render a collapsible glossary box explaining technical terms in plain English.

    Args:
        title: Expander label shown to the user
        items: {term: plain_english_explanation} — ordered dict
        icon:  Leading emoji for the expander label
    """
    with st.expander(f"{icon} {title}", expanded=False):
        for term, explanation in items.items():
            st.markdown(
                f"<div style='margin-bottom:10px;'>"
                f"<span style='color:#00d4ff;font-weight:600;font-size:13px;'>{term}</span>"
                f"<span style='color:#aaa;font-size:12px;'> — {explanation}</span>"
                f"</div>",
                unsafe_allow_html=True
            )


# ==============================================
# STREAMLIT APP CONFIGURATION
# ==============================================
st.set_page_config(page_title=UIConfig.PAGE_TITLE, layout=UIConfig.PAGE_LAYOUT)
st.title("🏆 ProTrader AI: Professional Stock Analytics")

# ==============================================
# SIDEBAR CONFIGURATION
# ==============================================
st.sidebar.header("Configuration")
indian_stocks = get_indian_stocks()
selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start", datetime.date(2020, 1, 1))  # 5 years for better accuracy
with col2:
    end_date = st.date_input("End", datetime.date.today())

st.sidebar.subheader("Advanced Settings")
enable_dynamic_fusion = st.sidebar.checkbox("Dynamic Fusion Framework", value=True)
enable_automl = st.sidebar.checkbox("AutoML Optimization (Slow)", value=False)
forecast_days = st.sidebar.slider("Forecast Horizon", 1, 30, 10)

chart_type = st.sidebar.radio("Primary Chart", ["Candlestick", "Line"])

# ==============================================
# MAIN ANALYSIS LOGIC
# ==============================================
show_analysis = False
df_stock = None
fundamentals = {}
news_articles = []

# State management for analysis persistence
if 'is_running_analysis' not in st.session_state:
    st.session_state['is_running_analysis'] = False

if st.sidebar.button("Launch Analysis", type="primary") or st.session_state['is_running_analysis']:
    st.session_state['is_running_analysis'] = True
    ticker = f"{selected_stock}.NS"
    
    # Create a main container for loading state
    loading_container = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        import time as _time
        _t0 = _time.time()
        _timings = {}

        # Step 1: Fetch Stock Data (20%)
        status_text.text("📊 Fetching stock data from Yahoo Finance...")
        _ts = _time.time()
        df_stock = get_stock_data(ticker, start_date, end_date)
        _timings['Stock Data (Yahoo Finance)'] = round(_time.time() - _ts, 2)
        progress_bar.progress(20)

        # Step 2: Fetch Fundamentals (35%)
        status_text.text("🏛️ Loading fundamental data...")
        _ts = _time.time()
        fundamentals = get_fundamental_data(ticker)
        _timings['Fundamentals'] = round(_time.time() - _ts, 2)
        progress_bar.progress(35)

        # Step 3: Fetch News (50%)
        status_text.text("📰 Fetching news articles...")
        _ts = _time.time()
        news_articles = get_news(selected_stock)
        _timings['News Articles'] = round(_time.time() - _ts, 2)
        progress_bar.progress(50)

        # Step 4: Fetch FII/DII (65%)
        status_text.text("💼 Loading FII/DII institutional data...")
        _ts = _time.time()
        # Check auto-fetch status first
        auto_data = fetch_fii_dii_apis_internal(None, start_date, end_date)

        if auto_data is None or auto_data.empty:
            # Try getting manual data
            fii_dii_data = get_fii_dii_data(None, start_date, end_date)

            # BLOCKING CHECK: If no manual data yet, Ask for it in MAIN APP
            if fii_dii_data is None or fii_dii_data.empty:
                progress_bar.empty()
                status_text.empty()

                with loading_container.container():
                     render_manual_fii_dii_input()

                # Stop further execution, but 'is_running_analysis' remains True
                # So when user submits and reruns, we come back here
                st.stop()
        else:
            fii_dii_data = auto_data

        _timings['FII/DII Data'] = round(_time.time() - _ts, 2)
        st.session_state['fii_dii_data'] = fii_dii_data
        progress_bar.progress(65)

        # Step 5: Fetch VIX (80%)
        status_text.text("📈 Loading India VIX volatility data...")
        _ts = _time.time()
        vix_data = get_india_vix_data(start_date, end_date)
        _timings['India VIX Data'] = round(_time.time() - _ts, 2)
        st.session_state['vix_data'] = vix_data
        progress_bar.progress(80)

        # Step 6: Multi-Source Sentiment (90%)
        status_text.text("🧠 Analyzing multi-source sentiment...")
        _ts = _time.time()
        try:
            from data.multi_sentiment import analyze_stock_sentiment
            multi_sentiment = analyze_stock_sentiment(selected_stock)

            # ── Fallback: enrich with Yahoo Finance news (Step 3) if RSS/NewsAPI
            # found 0 articles for this stock (common for mid/small-caps not in
            # the keyword dict).
            if multi_sentiment and multi_sentiment.get('article_count', 0) == 0 and news_articles:
                from data.news_sentiment import analyze_sentiment, filter_relevant_news
                relevant = filter_relevant_news(news_articles, selected_stock) or news_articles
                _yf_items = []
                _yf_sum = 0.0
                for _art in relevant[:20]:
                    _text = f"{_art.get('title', '')} {_art.get('description', '')}".strip()
                    if not _text:
                        continue
                    _label, _score = analyze_sentiment(_text)
                    _val = _score if _label == 'positive' else (-_score if _label == 'negative' else 0.0)
                    _yf_sum += _val
                    _date = str(_art.get('publishedAt', ''))[:16]
                    _yf_items.append({
                        'Date': _date,
                        'Source': f"Yahoo Finance ({_art.get('source', {}).get('name', 'News') if isinstance(_art.get('source'), dict) else str(_art.get('source', 'News'))})",
                        'Text': str(_art.get('title', ''))[:100] + '...',
                        'Label': _label,
                        'Score': f"{_val:+.2f}",
                        'Event': 'general',
                    })
                if _yf_items:
                    _yf_avg = _yf_sum / len(_yf_items)
                    multi_sentiment['all_items'] = _yf_items
                    multi_sentiment['article_count'] = len(_yf_items)
                    multi_sentiment['combined_sentiment'] = round(_yf_avg, 4)
                    multi_sentiment['combined_label'] = (
                        'bullish' if _yf_avg > 0.05 else 'bearish' if _yf_avg < -0.05 else 'neutral'
                    )
                    multi_sentiment['confidence'] = min(len(_yf_items) / 15, 1.0) * min(abs(_yf_avg) * 5, 1.0)
                    multi_sentiment['sources']['yahoo_finance'] = {
                        'available': True,
                        'count': len(_yf_items),
                        'average_sentiment': round(_yf_avg, 4),
                        'weight': 1.0,
                        'articles': [{'text': i['Text'], 'sentiment': i['Label'],
                                      'score': float(i['Score']), 'source': 'Yahoo Finance',
                                      'date': i['Date']} for i in _yf_items],
                    }
                    multi_sentiment['_fallback_source'] = 'Yahoo Finance News'

            st.session_state['multi_sentiment'] = multi_sentiment
        except Exception:
            st.session_state['multi_sentiment'] = None
        _timings['Multi-Source Sentiment'] = round(_time.time() - _ts, 2)
        progress_bar.progress(90)

        # Step 7: Pattern Detection (95%)
        status_text.text("📐 Detecting chart patterns...")
        _ts = _time.time()
        try:
            from models.visual_analyst import PatternAnalyst
            analyst = PatternAnalyst(order=5)
            pattern_analysis = analyst.analyze_all_patterns(df_stock)
            st.session_state['pattern_analysis'] = pattern_analysis
        except Exception:
            st.session_state['pattern_analysis'] = None
        _timings['Pattern Detection'] = round(_time.time() - _ts, 2)
        progress_bar.progress(95)

        _timings['_total_prefetch'] = round(_time.time() - _t0, 2)
        st.session_state['pipeline_timings'] = _timings

        # Final: Store everything (100%)
        status_text.text("✅ Finalizing analysis...")
        st.session_state['df_stock'] = df_stock
        st.session_state['fundamentals'] = fundamentals
        st.session_state['news_articles'] = news_articles
        st.session_state['selected_stock'] = selected_stock
        st.session_state['analysis_done'] = True
        st.session_state['forecast_days'] = forecast_days
        st.session_state['enable_dynamic_fusion'] = enable_dynamic_fusion
        st.session_state['chart_type'] = chart_type
        st.session_state['start_date'] = start_date
        st.session_state['end_date'] = end_date
        st.session_state['is_running_analysis'] = False # Done!
        progress_bar.progress(100)
        
        # Clear loading indicators
        progress_bar.empty()
        status_text.empty()
        loading_container.empty()
        
        if df_stock is not None and not df_stock.empty:
            df_stock = df_stock.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
            st.session_state['df_stock'] = df_stock
            show_analysis = True
            st.rerun()  # Rerun to show analysis
    
    except Exception as e:
        st.session_state['is_running_analysis'] = False # Reset on error
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error during analysis: {str(e)}")

# Restore from session state if available
elif st.session_state.get('analysis_done') and st.session_state.get('df_stock') is not None:
    df_stock = st.session_state['df_stock']
    fundamentals = st.session_state.get('fundamentals', {})
    news_articles = st.session_state.get('news_articles', [])
    selected_stock = st.session_state.get('selected_stock', selected_stock)
    forecast_days = st.session_state.get('forecast_days', forecast_days)
    enable_dynamic_fusion = st.session_state.get('enable_dynamic_fusion', enable_dynamic_fusion)
    chart_type = st.session_state.get('chart_type', chart_type)
    start_date = st.session_state.get('start_date', start_date)
    end_date = st.session_state.get('end_date', end_date)
    show_analysis = True

# ==============================================
# MAIN CONTENT TABS
# ==============================================
if show_analysis and df_stock is not None and not df_stock.empty:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Dashboard", 
        "🔬 Dynamic Fusion", 
        "📈 Technicals & Risk", 
        "🏛️ Fundamentals",
        "💼 FII/DII Analysis",
        "📰 Multi-Source Sentiment",
        "🛠️ Backtest",
        "📐 Pattern Analysis"
    ])

    # ==========================================
    # TAB 1: Main Dashboard
    # ==========================================
    with tab1:
        # Top Stats Row
        current_price = df_stock['Close'].iloc[-1]
        price_change = df_stock['Close'].iloc[-1] - df_stock['Close'].iloc[-2]
        pct_change = (price_change / df_stock['Close'].iloc[-2]) * 100
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"₹{current_price:,.2f}", f"{pct_change:.2f}%")
        m2.metric("Market Cap", f"{fundamentals.get('MarketCap', 'N/A')}")
        m3.metric("P/E Ratio", f"{fundamentals.get('Forward P/E', 'N/A')}")
        m4.metric("Volume", f"{df_stock['Volume'].iloc[-1]:,}")

        # Main Chart
        st.subheader(f"Price Action: {selected_stock}")
        if chart_type == "Candlestick":
            st.plotly_chart(create_candlestick_chart(df_stock), use_container_width=True)
        else:
            st.line_chart(df_stock["Close"])
            
        # Sentiment Summary
        filtered_news = filter_relevant_news(news_articles, selected_stock)
        daily_sentiment = {}
        if filtered_news:
            st.info(f"Analyzed {len(filtered_news)} recent news articles for sentiment integration.")
            for article in filtered_news:
                text = f"{article.get('title','')} {article.get('description','')}".strip()
                sentiment, score = analyze_sentiment(text)
                date = article.get("publishedAt", "")[0:10]
                
                if date in daily_sentiment:
                    daily_sentiment[date].append((sentiment, score))
                else:
                    daily_sentiment[date] = [(sentiment, score)]
        else:
            st.warning("No specific news found. Using technicals only.")

        # Run Hybrid Model
        st.subheader("🤖 AI Price Forecast")
        with st.spinner("Running Hybrid AI Models (Sequential Train/Test)..."):
            st.header("📊 Hybrid AI Model Analysis (Professional Validation)")
            
            # Use pre-fetched FII/DII Data from session state
            fii_dii_data = st.session_state.get('fii_dii_data', pd.DataFrame())
            
            if fii_dii_data is not None and not fii_dii_data.empty:
                fii_dii_features = extract_fii_dii_features(fii_dii_data)
                st.info(f"✅ Official NSE FII/DII data integrated | FII 5D Net: ₹{fii_dii_features['fii_net_5d']/1e7:.2f}Cr | DII 5D Net: ₹{fii_dii_features['dii_net_5d']/1e7:.2f}Cr")
            else:
                fii_dii_data = pd.DataFrame()
                st.warning("⚠️ FII/DII data unavailable. Using technical + sentiment only.")
            
            # Use pre-fetched VIX Data from session state
            vix_data = st.session_state.get('vix_data')
            if vix_data is not None and not vix_data.empty:
                vix_value = vix_data['Close'].iloc[-1] if 'Close' in vix_data.columns else 15
                # Ensure we have a scalar value, not a Series
                latest_vix = float(vix_value) if not isinstance(vix_value, (int, float)) else vix_value
                st.info(f"📊 India VIX integrated | Current: {latest_vix:.2f}")
            else:
                vix_data = None
                st.warning("⚠️ VIX data unavailable. Using default volatility.")
            
            # Use pre-fetched multi-source sentiment from session state
            multi_source_sentiment = st.session_state.get('multi_sentiment')
            if multi_source_sentiment and multi_source_sentiment.get('combined_sentiment') is not None:
                st.info(f"📰 Multi-source sentiment integrated | Score: {multi_source_sentiment.get('combined_sentiment', 0):+.3f} | Label: {multi_source_sentiment.get('combined_label', 'neutral')}")
            
            # Train hybrid model with ALL data sources
            import time as _time
            _model_t0 = _time.time()
            df_proc, results_df, models, scaler, features, metrics = create_hybrid_model(
                df_stock,
                daily_sentiment if daily_sentiment else {},
                fii_dii_data=fii_dii_data,
                vix_data=vix_data,
                multi_source_sentiment=multi_source_sentiment
            )
            _model_elapsed = round(_time.time() - _model_t0, 2)
            _pt = st.session_state.get('pipeline_timings', {})
            _pt['Model Training (XGB+LGBM+CatBoost+GRU+Stack)'] = _model_elapsed
            st.session_state['pipeline_timings'] = _pt
            
            # Display Metrics
            _hint("What do these model metrics mean?", {
                "Test RMSE (Root Mean Square Error)":
                    "On average, how far off (in daily return units) the model's predictions are. "
                    "Lower is better. E.g. 0.008 means the model is typically off by ~0.8% per day.",
                "Directional Accuracy":
                    "The % of days where the model correctly predicted whether the stock would go UP or DOWN. "
                    "50% = random guessing. Above 55% is useful. Above 65% is research-grade.",
                "XGBoost Weight / GRU Weight":
                    "How much influence each model has in the final blended prediction. "
                    "The system automatically gives more weight to whichever model performed better on recent data.",
                "Bullish Probability":
                    "The model's confidence (0–100%) that the stock will go UP tomorrow. "
                    "Above 55% = mild buy signal. Below 45% = mild sell signal. Around 50% = no clear view.",
                "LGBM / CatBoost Weight":
                    "LightGBM and CatBoost are two other 'tree-based' AI models alongside XGBoost. "
                    "Think of them as three different analysts looking at the same data — their combined view is more reliable than any one alone.",
                "Meta-Stacker Weight":
                    "A 'judge' model (Ridge regression) that has learned the best way to combine XGBoost + LightGBM + CatBoost + GRU. "
                    "It gets higher weight when it outperforms the individual models.",
                "Next-Day Volatility":
                    "The neural network's prediction of how 'jumpy' the stock will be tomorrow. "
                    "Higher = expect larger price swings. Useful for position sizing.",
                "Hurst Exponent":
                    "A number from 0 to 1 that tells you the market's 'memory'. "
                    ">0.55 = Trending (momentum stocks — ride the wave). "
                    "<0.45 = Mean-Reverting (range-bound — buy dips, sell rallies). "
                    "~0.50 = Random Walk (hard to predict — be cautious).",
            })
            st.subheader("Strict Walk-Forward Validation Results")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Test RMSE", f"{metrics['rmse']:.4f}")
            with c2:
                st.metric("Directional Accuracy", f"{metrics['accuracy']:.2f}%")
            with c3:
                xgb_w = metrics.get('xgb_weight', 0.5) * 100
                st.metric("XGBoost Weight", f"{xgb_w:.1f}%")
            with c4:
                gru_w = metrics.get('gru_weight', 0.5) * 100
                st.metric("GRU Weight", f"{gru_w:.1f}%")

            # Second metrics row: new research-grade outputs
            cr1, cr2, cr3, cr4 = st.columns(4)
            with cr1:
                dir_prob = metrics.get('last_directional_prob', None)
                if dir_prob is not None:
                    prob_delta = "Bullish" if dir_prob > 55 else ("Bearish" if dir_prob < 45 else "Neutral")
                    st.metric("Bullish Probability", f"{dir_prob:.1f}%", delta=prob_delta)
                else:
                    st.metric("Bullish Probability", "N/A")
            with cr2:
                lgbm_w  = metrics.get('lgbm_weight', 0) * 100
                cb_w    = metrics.get('catboost_weight', 0) * 100
                cb_avail = metrics.get('catboost_available', False)
                st.metric("LGBM / CatBoost Weight",
                          f"{lgbm_w:.1f}% / {cb_w:.1f}%" if cb_avail else f"{lgbm_w:.1f}% / N/A")
            with cr3:
                stack_w  = metrics.get('stack_weight', 0) * 100
                rnn_vol  = metrics.get('last_rnn_vol_pred', None)
                vol_str  = f" | Next-Day Vol: {rnn_vol:.4f}" if rnn_vol is not None else ""
                st.metric("Meta-Stacker Weight", f"{stack_w:.1f}%", delta=vol_str if vol_str else None)
            with cr4:
                hurst_val = metrics.get('hurst_exponent', None)
                if hurst_val is not None:
                    market_char = "Trending" if hurst_val > 0.55 else ("Mean-Rev." if hurst_val < 0.45 else "Random")
                    st.metric("Hurst Exponent", f"{hurst_val:.3f}", delta=market_char)
                else:
                    st.metric("Hurst Exponent", "N/A")

            # SHAP Feature Importance
            shap_importance = metrics.get('shap_importance')
            if shap_importance:
                with st.expander("🔍 SHAP Feature Importance (Top-10) — Which signals matter most?", expanded=False):
                    shap_df = pd.DataFrame(
                        list(shap_importance.items()), columns=['Feature', 'Mean |SHAP|']
                    ).sort_values('Mean |SHAP|', ascending=True).tail(10)
                    fig_shap = go.Figure(go.Bar(
                        x=shap_df['Mean |SHAP|'], y=shap_df['Feature'],
                        orientation='h', marker_color='#00d4ff'
                    ))
                    fig_shap.update_layout(
                        template="plotly_dark", height=320,
                        title="XGBoost SHAP — Mean Absolute Feature Impact",
                        xaxis_title="Mean |SHAP value|", yaxis_title=""
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                    st.caption(
                        "Each bar shows how much a feature pushes the prediction up or down on average. "
                        "Longer bar = that signal has a bigger impact on the model's output. "
                        "Features like RSI, CMF, and MACD appearing at the top means the model is "
                        "relying on momentum and volume flow — which is economically sensible."
                    )

            # Predicted vs Actual Returns
            _hint("How to read this chart", {
                "Blue line (Actual Returns)":
                    "What the stock actually did each day — the ground truth.",
                "Orange line (Predicted Returns)":
                    "What the model predicted for each day. It will never be perfectly aligned, "
                    "but good models track the direction (up/down) most of the time.",
                "What to look for":
                    "When both lines move in the same direction on the same day, that's a correct prediction. "
                    "The model doesn't need to predict the exact size — just getting UP vs DOWN right is what matters "
                    "(that's what Directional Accuracy measures).",
                "Why the orange line looks smoother":
                    "ML models tend to predict small returns and rarely predict extreme moves. "
                    "This is normal — it reduces false alarms even though it misses big spikes.",
            })
            st.subheader("Predicted vs Actual Returns")
            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual_Return'], 
                                         name='Actual Returns', mode='lines', line=dict(color='blue', width=1)))
            fig_val.add_trace(go.Scatter(x=results_df.index, y=results_df['Predicted_Return'], 
                                         name='Predicted Returns', mode='lines', line=dict(color='orange', width=1)))
            fig_val.update_layout(template="plotly_dark")
            st.plotly_chart(fig_val, use_container_width=True)

            # Store results
            st.session_state['results_df'] = results_df
            st.session_state['metrics'] = metrics
            st.session_state['df_proc'] = df_proc
            st.session_state['models'] = models
            st.session_state['scaler'] = scaler
            st.session_state['features'] = features
            
            # Forecast
            future_prices = hybrid_predict_prices(
                models, scaler, df_proc.iloc[-60:], features,
                days=forecast_days,
                df_proc_full=df_proc,
                directional_prob=metrics.get('last_directional_prob', 50.0) / 100.0,
                regime=metrics.get('regime', 'normal'),
                n_paths=200,
            )
            st.session_state['future_prices'] = future_prices
            
            # AI Expert Analysis
            st.markdown("---")
            st.header("🤖 AI Expert Analysis")
            
            technical_indicators = {
                'RSI': df_proc['RSI_Norm'].iloc[-1] * 100 if 'RSI_Norm' in df_proc.columns else 50,
                'Volatility_5D': df_proc['Volatility_5D'].iloc[-1] if 'Volatility_5D' in df_proc.columns else 0,
                'Volatility_20D': df_proc.get('Volatility_20D', pd.Series([0])).iloc[-1] if 'Volatility_20D' in df_proc.columns else 0,
                'Price_vs_MA20': df_proc['MA_Div'].iloc[-1] if 'MA_Div' in df_proc.columns else 0,
                'MACD_Histogram': df_proc.get('MACD_Histogram', pd.Series([0])).iloc[-1] if 'MACD_Histogram' in df_proc.columns else 0
            }
            
            with st.spinner("🧠 Generating Advanced AI Expert Analysis (DeepSeek V3 + Gemini 2.5)..."):
                gemini_analysis = generate_ai_analysis(
                    stock_symbol=selected_stock,
                    current_price=current_price,
                    predicted_prices=future_prices,
                    metrics=metrics,
                    fundamentals=fundamentals,
                    sentiment_summary=daily_sentiment,
                    technical_indicators=technical_indicators,
                    volatility_data=df_proc['Volatility_5D'].iloc[-1] if 'Volatility_5D' in df_proc.columns else 0,
                    fii_dii_data=st.session_state.get('fii_dii_data'),
                    vix_data=st.session_state.get('vix_data'),
                    patterns=st.session_state.get('pattern_analysis', {}).get('patterns') if st.session_state.get('pattern_analysis') else None
                )
                gemini_used = "template mode" not in gemini_analysis.lower()
            
            # Verdict Card Logic
            forecast_return = ((future_prices['Predicted Price'].iloc[-1] - current_price) / current_price) * 100 if not future_prices.empty else 0
            model_accuracy = metrics.get('accuracy', 0)
            
            if model_accuracy < 50:
                # If model is unreliable, default to neutral/uncertain unless return is extreme
                gradient = UIConfig.GRADIENT_NEUTRAL
                border_color = UIConfig.COLOR_NEUTRAL
                verdict_text = "UNCERTAIN"
            elif forecast_return > 3:
                gradient = UIConfig.GRADIENT_BULLISH
                border_color = UIConfig.COLOR_BULLISH
                verdict_text = "BULLISH"
            elif forecast_return < -3:
                gradient = UIConfig.GRADIENT_BEARISH
                border_color = UIConfig.COLOR_BEARISH
                verdict_text = "BEARISH"
            else:
                gradient = UIConfig.GRADIENT_NEUTRAL
                border_color = UIConfig.COLOR_NEUTRAL
                verdict_text = "NEUTRAL"
            
            st.markdown(f"""
            <div style="background: {gradient}; padding: 25px; border-radius: 15px; border: 2px solid {border_color}; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 28px; font-weight: bold; color: white;">{selected_stock}</span>
                        <span style="font-size: 18px; color: #aaa; margin-left: 10px;">Quick Verdict</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 32px; font-weight: bold; color: {border_color};">{verdict_text}</div>
                        <div style="font-size: 16px; color: #ccc;">Forecast: {forecast_return:+.2f}% | Accuracy: {metrics['accuracy']:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed Analysis - Use proper Streamlit markdown rendering
            analysis_mode = "✨ Powered by DeepSeek & Gemini" if gemini_used else "📝 Template Analysis"
            st.markdown(f"### 🧠 AI Expert Analysis <span style='font-size: 14px; color: {'#00ff88' if gemini_used else '#ffaa00'};'>({analysis_mode})</span>", unsafe_allow_html=True)
            st.markdown(gemini_analysis)
            
            st.markdown("---")
            
            # Forecast Plot — probabilistic fan chart
            _hint("How to read the Forecast Fan Chart", {
                "Median Path (bold line)": "The model's best single guess for where the stock price is headed. Think of it as the 'expected' scenario — not a guarantee, just the most likely path.",
                "P25–P75 Inner Band (darker shading)": "The 'comfortable range' — there is a 50% chance the actual price will land inside this band. If the band is narrow, the model is fairly confident.",
                "P5–P95 Outer Band (lighter shading)": "The 'tail scenarios' band. There is a 90% chance the price stays within this range. It gets wider every day because the future becomes more uncertain the further out you look.",
                "Pessimistic Line (red dotted, P5)": "The worst-case scenario that the model thinks has only a 5% chance of happening — one-in-twenty bad outcomes.",
                "Optimistic Line (green dotted, P95)": "The best-case scenario with only a 5% chance of being exceeded — one-in-twenty great outcomes.",
                "Today Vertical Line": "The boundary between what actually happened (historical prices, left side) and what the model is forecasting (right side).",
                "Why the bands widen": "The model bootstraps real historical daily moves. Over 10 days, small daily errors compound — just like rolling a dice: each extra day multiplies uncertainty, so bands naturally fan outward.",
                "How drift works": "If the model says 60% bullish probability, it adds a small upward nudge each day. If 40% (bearish), a small downward nudge. The actual daily moves are sampled from real past returns — not a straight line.",
            })
            st.subheader("📈 Price Forecast with Uncertainty Bands")
            dir_prob_pct = metrics.get('last_directional_prob', 50.0)
            regime_label = metrics.get('regime', 'normal').replace('_', ' ').title()

            fig_forecast = go.Figure()

            # ── Historical prices (last 60 days) ──
            fig_forecast.add_trace(go.Scatter(
                x=df_stock.index[-60:], y=df_stock['Close'][-60:],
                name="Historical", line=dict(color=UIConfig.COLOR_PRIMARY, width=2)
            ))

            # ── Uncertainty bands (P5–P95 shaded, P25–P75 inner band) ──
            if 'P95' in future_prices.columns and 'P5' in future_prices.columns:
                # Outer band: P5–P95
                fig_forecast.add_trace(go.Scatter(
                    x=list(future_prices.index) + list(future_prices.index[::-1]),
                    y=list(future_prices['P95']) + list(future_prices['P5'][::-1]),
                    fill='toself', fillcolor='rgba(0,212,255,0.08)',
                    line=dict(color='rgba(0,0,0,0)'), name='P5–P95 Range',
                    hoverinfo='skip'
                ))
            if 'P75' in future_prices.columns and 'P25' in future_prices.columns:
                # Inner band: P25–P75
                fig_forecast.add_trace(go.Scatter(
                    x=list(future_prices.index) + list(future_prices.index[::-1]),
                    y=list(future_prices['P75']) + list(future_prices['P25'][::-1]),
                    fill='toself', fillcolor='rgba(0,212,255,0.18)',
                    line=dict(color='rgba(0,0,0,0)'), name='P25–P75 Range',
                    hoverinfo='skip'
                ))
            if 'P5' in future_prices.columns:
                fig_forecast.add_trace(go.Scatter(
                    x=future_prices.index, y=future_prices['P5'],
                    name='Pessimistic (P5)', line=dict(color='#ff6b6b', width=1, dash='dot'),
                    opacity=0.7
                ))
            if 'P95' in future_prices.columns:
                fig_forecast.add_trace(go.Scatter(
                    x=future_prices.index, y=future_prices['P95'],
                    name='Optimistic (P95)', line=dict(color='#7bed9f', width=1, dash='dot'),
                    opacity=0.7
                ))

            # ── Median forecast (bold centre line) ──
            fig_forecast.add_trace(go.Scatter(
                x=future_prices.index, y=future_prices['Predicted Price'],
                name='Median Forecast', line=dict(color=UIConfig.COLOR_SECONDARY, width=2.5),
            ))

            # Vertical separator: today  (add_vline is buggy with date strings in
            # this plotly version, so draw shape + annotation manually)
            today_str = df_stock.index[-1].strftime('%Y-%m-%d')
            fig_forecast.add_shape(
                type='line',
                x0=today_str, x1=today_str, y0=0, y1=1,
                xref='x', yref='paper',
                line=dict(color='rgba(255,255,255,0.3)', dash='dash', width=1)
            )
            fig_forecast.add_annotation(
                x=today_str, y=1, xref='x', yref='paper',
                text='Today', showarrow=False,
                font=dict(color='rgba(255,255,255,0.6)', size=11),
                xanchor='left', yanchor='top'
            )

            fig_forecast.update_layout(
                template="plotly_dark",
                title=(f"{selected_stock} — {forecast_days}-Day Probabilistic Forecast  |  "
                       f"Bullish Prob: {dir_prob_pct:.1f}%  |  Regime: {regime_label}"),
                yaxis_title="Price (₹)",
                xaxis_title="Date",
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            st.caption(
                "Shaded bands show the range of 200 simulated price paths. "
                "Median path uses the model's directional signal as drift. "
                "Band width grows with time, reflecting genuine forecast uncertainty."
            )

            # Accuracy Comparison Chart
            st.subheader("🎯 Model Accuracy: Actual vs Predicted Prices")
            accuracy_chart = create_accuracy_comparison_chart(df_stock, results_df, future_prices)
            st.plotly_chart(accuracy_chart, use_container_width=True)
            
            with st.expander("📊 View Detailed Forecast Table"):
                st.dataframe(future_prices.style.format("{:.2f}"))

    # ==========================================
    # TAB 2: Dynamic Fusion
    # ==========================================
    with tab2:
        _hint("What is Dynamic Fusion and how to read the Weight Evolution chart?", {
            "Dynamic Fusion Framework": "Instead of always giving each AI model a fixed influence, this system checks how accurate each model has been recently and gives more weight to whoever is performing best right now.",
            "Technical Expert Weight %": "Reflects how much the price-pattern GRU neural network is being trusted today. High = the market is behaving in a technically predictable way (clear trends, patterns working).",
            "Sentiment Expert Weight %": "Reflects how much the news/Reddit sentiment model is being trusted. High = fundamental/news-driven market (earnings season, major announcements).",
            "Volatility Expert Weight %": "Reflects how much the India VIX / fear model is being trusted. High = the market is currently driven by fear or greed rather than fundamentals or technicals.",
            "Weight Evolution chart": "Shows how the three experts' influence has shifted over the last 15 days. A large swing in one expert's weight signals a market regime change — e.g., a sudden jump in Volatility weight means fear spiked.",
            "Why weights change": "Each expert's weight is proportional to exp(−σ²) where σ is its recent prediction error. When an expert makes accurate predictions, its error is small, its weight grows. Poor predictions → weight shrinks automatically.",
        })
        st.header("🔬 Dynamic Fusion Framework")

        st.markdown(f"""
        <div style="background: {UIConfig.GRADIENT_DARK}; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
            <h4 style="color: #e94560;">Bayesian Multi-Expert System</h4>
            <p style="color: #eee;">Dynamically combines three specialized AI models:</p>
            <ul style="color: #aaa;">
                <li><strong style="color: {UIConfig.COLOR_PRIMARY};">Technical Expert</strong> - GRU neural network analyzing price patterns</li>
                <li><strong style="color: {UIConfig.COLOR_BULLISH};">Sentiment Expert</strong> - FinBERT Financial Transformer analyzing news</li>
                <li><strong style="color: {UIConfig.COLOR_SECONDARY};">Volatility Expert</strong> - MLP analyzing India VIX & market fear</li>
            </ul>
            <p style="color: #888; font-size: 12px;">Weights adjusted using: w = exp(-σ²) / Σ exp(-σ²)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if enable_dynamic_fusion:
            vix_data = get_india_vix_data(start_date, end_date)
            fusion_framework = DynamicFusionFramework()
            
            fmt_sentiment = {k: [(s, c) for s, c in v] for k, v in daily_sentiment.items()}
            
            # Get multi-source sentiment if available
            multi_source_sentiment = st.session_state.get('multi_sentiment', None)
            
            with st.spinner("🧠 Training Expert Models..."):
                try:
                    fusion_framework.train_models(
                        df_stock, fmt_sentiment, vix_data,
                        multi_source_sentiment=multi_source_sentiment
                    )
                    
                    recent_preds = []
                    sim_range = df_stock.index[-15:]
                    
                    for date in sim_range:
                        try:
                            curr_idx = df_stock.index.get_loc(date)
                            if curr_idx < 30:
                                continue
                            stock_slice = df_stock.iloc[:curr_idx]
                            
                            res = fusion_framework.predict(stock_slice, fmt_sentiment, vix_data)
                            recent_preds.append({
                                'Date': date,
                                'Actual': df_stock.loc[date, 'Close'],
                                'Fusion': res['combined_prediction'],
                                'Tech_W': res['weights']['technical'],
                                'Sent_W': res['weights']['sentiment'],
                                'Vol_W': res['weights']['volatility']
                            })
                        except Exception:
                            continue
                    
                    if recent_preds:
                        res_df = pd.DataFrame(recent_preds).set_index('Date')
                        
                        st.subheader("📊 Current Expert Weights")
                        latest_weights = recent_preds[-1]
                        
                        w1, w2, w3 = st.columns(3)
                        w1.metric("Technical 📈", f"{latest_weights['Tech_W']*100:.1f}%")
                        w2.metric("Sentiment 📰", f"{latest_weights['Sent_W']*100:.1f}%")
                        w3.metric("Volatility ⚡", f"{latest_weights['Vol_W']*100:.1f}%")
                        
                        st.markdown("---")
                        
                        # Weights Chart
                        st.subheader("📈 Weight Evolution (Last 15 Days)")
                        fig_w = go.Figure()
                        fig_w.add_trace(go.Scatter(x=res_df.index, y=res_df['Tech_W'], name='Technical', stackgroup='one', line=dict(color=UIConfig.COLOR_PRIMARY)))
                        fig_w.add_trace(go.Scatter(x=res_df.index, y=res_df['Sent_W'], name='Sentiment', stackgroup='one', line=dict(color=UIConfig.COLOR_BULLISH)))
                        fig_w.add_trace(go.Scatter(x=res_df.index, y=res_df['Vol_W'], name='Volatility', stackgroup='one', line=dict(color=UIConfig.COLOR_SECONDARY)))
                        fig_w.update_layout(template="plotly_dark", title="Dynamic Expert Influence", yaxis=dict(tickformat='.0%'))
                        st.plotly_chart(fig_w, use_container_width=True)
                    else:
                        st.warning("⚠️ Insufficient data for dynamic fusion analysis.")
                except Exception as e:
                    st.error(f"❌ Dynamic Fusion training failed: {str(e)}")
        else:
            st.info("👆 Enable 'Dynamic Fusion Framework' in the sidebar to use this feature.")

    # ==========================================
    # TAB 3: Technicals & Risk
    # ==========================================
    with tab3:
        st.header("Risk Management & Technicals")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Key Levels (Fibonacci)")
            fibs = RiskManager.calculate_fibonacci_levels(df_stock)
            fib_df = pd.DataFrame.from_dict(fibs, orient='index', columns=['Price'])
            st.table(fib_df.style.format("₹{:.2f}"))
            
        with c2:
            st.subheader("Risk Metrics")
            atr = RiskManager.calculate_atr(df_stock)
            st.metric("ATR (Volatility)", f"₹{atr:.2f}")
            
            st.markdown("### 🛡️ Trade Setup Calculator")
            
            pred_price = future_prices['Predicted Price'].iloc[-1] if not future_prices.empty else current_price
            setup = RiskManager.get_trade_setup(current_price, pred_price, atr, metrics['accuracy']/100)
            
            st.write(f"**Action:** {setup['Direction']}")
            st.write(f"**Entry:** ₹{setup['Entry']:.2f}")
            st.write(f"**Stop Loss:** ₹{setup['Stop Loss']:.2f} ({(setup['Stop Loss']-current_price)/current_price*100:.2f}%)")
            st.write(f"**Target:** ₹{setup['Target']:.2f} ({(setup['Target']-current_price)/current_price*100:.2f}%)")
            st.metric("Risk/Reward Ratio", f"1:{setup['Risk/Reward']:.2f}")

    # ==========================================
    # TAB 4: Fundamentals
    # ==========================================
    with tab4:
        st.header("Fundamental Health")
        
        f_cols = ["Forward P/E", "PEG Ratio", "Price/Book", "Debt/Equity", "ROE", "Profit Margins"]
        
        fc1, fc2, fc3 = st.columns(3)
        for i, key in enumerate(f_cols):
            val = fundamentals.get(key)
            if pd.isna(val): val = "N/A"
            elif isinstance(val, (int, float)): val = f"{val:.2f}"
            
            if i % 3 == 0: fc1.metric(key, val)
            elif i % 3 == 1: fc2.metric(key, val)
            else: fc3.metric(key, val)
            
        st.caption("Data source: Yahoo Finance")

    # ==========================================
    # TAB 5: FII/DII Analysis
    # ==========================================
    with tab5:
        st.header("💼 Institutional Investor Analysis (FII/DII)")
        
        st.markdown(f"""
        <div style="background: {UIConfig.GRADIENT_DARK}; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
            <h4 style="color: #e94560;">📊 Official NSE India Data</h4>
            <p style="color: #eee;"><strong>FII:</strong> Foreign entities investing in Indian markets.</p>
            <p style="color: #eee;"><strong>DII:</strong> Indian mutual funds, insurance companies.</p>
        </div>
        """, unsafe_allow_html=True)
        
        fii_dii_data = get_fii_dii_data(None, start_date, end_date)
        
        if not fii_dii_data.empty:
            fii_features = extract_fii_dii_features(fii_dii_data, lookback=20)
            
            col_fii1, col_fii2, col_fii3, col_fii4 = st.columns(4)
            
            fii_net_20d = fii_features['fii_net_5d']
            dii_net_20d = fii_features['dii_net_5d']
            
            col_fii1.metric("FII Net (20D)", f"₹{fii_net_20d/1e7:.2f}Cr", 
                           delta="Buying" if fii_net_20d > 0 else "Selling")
            col_fii2.metric("DII Net (20D)", f"₹{dii_net_20d/1e7:.2f}Cr",
                           delta="Buying" if dii_net_20d > 0 else "Selling")
            col_fii3.metric("FII Trend", "Bullish 🟢" if fii_features['fii_trend'] > 0 else "Bearish 🔴")
            col_fii4.metric("DII Trend", "Bullish 🟢" if fii_features['dii_trend'] > 0 else "Bearish 🔴")
            
            st.markdown("---")
            
            fig_activity, fig_cumulative = create_fii_dii_chart(fii_dii_data)
            
            st.subheader("📊 Institutional Activity Over Time")
            st.plotly_chart(fig_activity, use_container_width=True)
            
            st.subheader("📈 Cumulative Institutional Positions")
            st.plotly_chart(fig_cumulative, use_container_width=True)
        else:
            st.error("❌ Could not fetch FII/DII data.")

    # ==========================================
    # TAB 6: Multi-Source Sentiment Analysis
    # ==========================================
    with tab6:
        _hint("What is Sentiment Analysis and why does it matter?", {
            "Sentiment Score": "A number from -1.0 (very negative/bearish) to +1.0 (very positive/bullish). 0.0 is neutral. It measures the overall 'mood' of the news and discussion about this stock right now.",
            "Confidence %": "How much the system trusts its own reading. Low confidence = sources disagreed a lot, or very few articles found. High confidence = sources were in clear agreement.",
            "Sources Agree ✓": "All 4 data sources (RSS news, NewsAPI, Reddit, Google Trends) gave similar scores. The signal is more reliable when all sources point the same way.",
            "Sources Disagree ⚠": "Different sources gave conflicting signals — e.g., news is bullish but Reddit is bearish. This often happens around uncertain events. Take the overall score with extra caution.",
            "Temporal Decay": "Older news is given less weight than today's news. A 3-day-old article has only 22% the weight of a same-day article. Keeps the score relevant to what is happening now.",
            "Event Type Weight": "Earnings reports (2× weight) and regulatory/SEBI news (1.8×) influence the score more than random general chatter (1×). The system auto-identifies article type from keywords.",
            "Earnings / Regulatory / Dividend / Management / General": "The pie chart shows what category most articles fall into. If most articles are 'Earnings', the sentiment is likely driven by results season.",
        })
        st.header("📰 Multi-Source Sentiment Analysis")

        st.markdown(f"""
        <div style="background: {UIConfig.GRADIENT_DARK}; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
            <h4 style="color: #e94560;">🔬 High-Accuracy Sentiment Engine</h4>
            <p style="color: #eee;">Combines 4 reliable sources for maximum accuracy:</p>
            <ul style="color: #aaa;">
                <li><strong style="color: {UIConfig.COLOR_PRIMARY};">RSS News (30%)</strong> - Moneycontrol, Economic Times, LiveMint, Business Standard</li>
                <li><strong style="color: {UIConfig.COLOR_SECONDARY};">NewsAPI (25%)</strong> - Global financial news aggregation</li>
                <li><strong style="color: {UIConfig.COLOR_BULLISH};">Reddit (25%)</strong> - r/IndianStockMarket, r/DalalStreetTalks, r/IndiaInvestments</li>
                <li><strong style="color: #ffaa00;">Google Trends (20%)</strong> - Retail interest proxy</li>
            </ul>
            <p style="color: #888; font-size: 12px;">Weights auto-adjust if a source is unavailable.</p>
        </div>
        """, unsafe_allow_html=True)

        from data.multi_sentiment import analyze_stock_sentiment

        # Sentiment is already analyzed in the main loading sequence (Step 6)

        
        if 'multi_sentiment' in st.session_state and st.session_state['multi_sentiment']:
            result = st.session_state['multi_sentiment']
            
            # Overall Sentiment Card
            sentiment_label = result['combined_label']
            sentiment_score = result['combined_sentiment']
            
            if 'bullish' in sentiment_label:
                sent_color = UIConfig.COLOR_BULLISH
                sent_gradient = UIConfig.GRADIENT_BULLISH
            elif 'bearish' in sentiment_label:
                sent_color = UIConfig.COLOR_BEARISH
                sent_gradient = UIConfig.GRADIENT_BEARISH
            else:
                sent_color = UIConfig.COLOR_NEUTRAL
                sent_gradient = UIConfig.GRADIENT_NEUTRAL
            
            st.markdown(f"""
            <div style="background: {sent_gradient}; padding: 25px; border-radius: 15px; border: 2px solid {sent_color}; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 24px; font-weight: bold; color: white;">{selected_stock} Sentiment</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 28px; font-weight: bold; color: {sent_color};">{sentiment_label.upper().replace('_', ' ')}</div>
                        <div style="font-size: 16px; color: #ccc;">Score: {sentiment_score:+.3f} | Confidence: {result['confidence']*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Fallback notice
            if result.get('_fallback_source'):
                st.info(
                    f"ℹ️ RSS/NewsAPI/Reddit found no articles for **{selected_stock}** "
                    f"(stock may not appear in financial news feeds by its ticker symbol). "
                    f"Sentiment computed from **Yahoo Finance News** ({result['article_count']} articles) as fallback."
                )

            # Source disagreement alert
            disagreement = result.get('source_disagreement', 0.0)
            conf_penalty = result.get('confidence_penalty', 0.0)
            if disagreement > 0.2:
                st.warning(f"Sources Disagree ⚠ — Std of source scores: {disagreement:.3f}. Confidence reduced by {conf_penalty*100:.0f}%.")
            elif result.get('article_count', 0) > 0 and not result.get('_fallback_source'):
                st.success("Sources Agree ✓ — Low inter-source disagreement.")

            # ── Event breakdown pie chart ──────────────────────────────────────
            event_breakdown = result.get('event_breakdown', {})
            if event_breakdown:
                import plotly.express as px
                ev_df = pd.DataFrame(list(event_breakdown.items()), columns=['Event Type', 'Count'])
                fig_ev = px.pie(
                    ev_df, names='Event Type', values='Count',
                    title='News Event Breakdown (All Sources)',
                    color_discrete_sequence=['#00d4ff', '#ff6b35', '#7bed9f', '#ffd700', '#a29bfe']
                )
                fig_ev.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig_ev, use_container_width=True)

            # ── Full News Table (all articles analysed) ───────────────────────
            if 'all_items' in result and result['all_items']:
                st.subheader("📋 All Articles Analysed")
                df_sent_raw = pd.DataFrame(result['all_items'])
                # Colour-code Label column
                def _colour_label(val):
                    if val == 'positive':
                        return 'background-color: rgba(0,255,136,0.18); color:#00ff88; font-weight:600'
                    elif val == 'negative':
                        return 'background-color: rgba(255,68,68,0.18); color:#ff4444; font-weight:600'
                    return 'color:#aaa'
                # Reorder columns nicely
                _cols_order = [c for c in ['Date', 'Source', 'Text', 'Label', 'Score', 'Event'] if c in df_sent_raw.columns]
                df_sent_display = df_sent_raw[_cols_order].copy()
                df_sent_display.columns = [c.title() for c in df_sent_display.columns]
                styled = df_sent_display.style.applymap(
                    _colour_label, subset=['Label'] if 'Label' in df_sent_display.columns else []
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)
                st.markdown("---")

            # Source breakdown
            _hint("How to read the Source Breakdown section", {
                "RSS News (30%)": "Financial news sites like Moneycontrol, Economic Times, LiveMint. Reliable, professionally written. Score close to +1 means many positive headlines today.",
                "NewsAPI (25%)": "Broader global news aggregation. Useful for catching international news that affects the stock.",
                "Reddit (25%)": "Community discussion from Indian investing subreddits. Reflects retail investor mood. High engagement (upvotes) articles get more weight.",
                "Google Trends (20%)": "Measures how much people are searching for the stock. Rising search interest often precedes price moves as retail investors start paying attention.",
                "Average Sentiment per source": "Each source's weighted average score. If RSS is +0.3 but Reddit is -0.2, the sources are in disagreement — the combined score will be muted and confidence will be lower.",
            })
            st.subheader("📊 Source Breakdown")
            
            s1, s2, s3, s4 = st.columns(4)
            
            rss_data = result['sources'].get('rss', {})
            newsapi_data = result['sources'].get('newsapi', {})
            reddit_data = result['sources'].get('reddit', {})
            trends_data = result['sources'].get('google_trends', {})
            yf_data = result['sources'].get('yahoo_finance', {})

            # If Yahoo Finance fallback is active, show it prominently in a single column
            if result.get('_fallback_source'):
                yf_score = yf_data.get('average_sentiment', 0)
                st.markdown("**📡 Yahoo Finance News (Fallback — 100% weight)**")
                yf_c1, yf_c2 = st.columns(2)
                yf_c1.metric("Articles", yf_data.get('count', 0))
                yf_c2.metric("Sentiment", f"{yf_score:+.3f}",
                             delta="Positive" if yf_score > 0 else "Negative" if yf_score < 0 else "Neutral")
                st.caption("RSS / NewsAPI / Reddit returned 0 articles for this ticker. Yahoo Finance News used instead.")
            else:
                with s1:
                    st.markdown("**📰 RSS News (30%)**")
                    if rss_data.get('available'):
                        rss_score = rss_data.get('average_sentiment', 0)
                        st.metric("Articles", rss_data.get('count', 0))
                        st.metric("Sentiment", f"{rss_score:+.3f}", delta="Positive" if rss_score > 0 else "Negative" if rss_score < 0 else "Neutral")
                    else:
                        st.warning("No RSS data")

                with s2:
                    st.markdown("**🌐 NewsAPI (25%)**")
                    if newsapi_data.get('available'):
                        newsapi_score = newsapi_data.get('average_sentiment', 0)
                        st.metric("Articles", newsapi_data.get('count', 0))
                        st.metric("Sentiment", f"{newsapi_score:+.3f}", delta="Positive" if newsapi_score > 0 else "Negative" if newsapi_score < 0 else "Neutral")
                    else:
                        from config.settings import NEWS_API_KEY
                        if NEWS_API_KEY:
                            st.info("No NewsAPI articles found")
                        else:
                            st.info("Add NEWS_API_KEY to .env")

                with s3:
                    st.markdown("**💬 Reddit (25%)**")
                    if reddit_data.get('available'):
                        reddit_score = reddit_data.get('average_sentiment', 0)
                        st.metric("Posts", reddit_data.get('count', 0))
                        st.metric("Sentiment", f"{reddit_score:+.3f}", delta="Positive" if reddit_score > 0 else "Negative" if reddit_score < 0 else "Neutral")
                    else:
                        st.info("Add Reddit API to .env")

                with s4:
                    st.markdown("**📈 Trends (20%)**")
                    if trends_data.get('available'):
                        trend_text = trends_data.get('trend', 'unknown').replace('_', ' ').title()
                        st.metric("Trend", trend_text)
                        st.metric("Change", f"{trends_data.get('change_pct', 0):+.1f}%")
                    else:
                        st.info("Unavailable")
            
            st.markdown("---")

            # ── Per-source detailed expandable tables ─────────────────────────
            def _source_table(articles_list, key_map):
                """Convert a list of article dicts to a styled dataframe."""
                if not articles_list:
                    st.info("No articles from this source.")
                    return
                rows = []
                for a in articles_list:
                    sent = a.get('sentiment', a.get('label', 'neutral'))
                    emoji = "🟢" if sent == 'positive' else "🔴" if sent == 'negative' else "⚪"
                    rows.append({
                        'Date': str(a.get('date', a.get('Date', '')))[:16],
                        'Headline': a.get(key_map.get('text', 'text'), '')[:120],
                        'Sentiment': f"{emoji} {sent.title()}",
                        'Score': f"{a.get('score', a.get('Score', 0)):+.2f}",
                        'Event': a.get('event_type', a.get('Event', 'general')),
                        'Temp.Weight': a.get('temporal_weight', 1.0),
                    })
                df_tbl = pd.DataFrame(rows)
                st.dataframe(df_tbl, use_container_width=True, hide_index=True)

            rss_arts = rss_data.get('articles', [])
            newsapi_arts = newsapi_data.get('articles', [])
            reddit_posts = reddit_data.get('posts', [])
            yf_arts = yf_data.get('articles', [])

            # If Yahoo Finance fallback, show its articles prominently (expanded)
            if result.get('_fallback_source') and yf_arts:
                with st.expander(f"📡 Yahoo Finance News ({len(yf_arts)} articles — fallback source)", expanded=True):
                    _source_table(yf_arts, {'text': 'text'})
            else:
                with st.expander(f"📰 RSS News Articles ({len(rss_arts)} fetched)", expanded=False):
                    _source_table(rss_arts, {'text': 'text'})

                with st.expander(f"🌐 NewsAPI Articles ({len(newsapi_arts)} fetched)", expanded=False):
                    _source_table(newsapi_arts, {'text': 'text'})

            with st.expander(f"💬 Reddit Posts ({len(reddit_posts)} fetched)", expanded=False):
                if reddit_posts:
                    rows = []
                    for p in reddit_posts:
                        sent = p.get('sentiment', 'neutral')
                        emoji = "🟢" if sent == 'positive' else "🔴" if sent == 'negative' else "⚪"
                        rows.append({
                            'Date': str(p.get('date', ''))[:16],
                            'Subreddit': p.get('source', ''),
                            'Title': p.get('title', p.get('text', ''))[:120],
                            'Sentiment': f"{emoji} {sent.title()}",
                            'Score': f"{p.get('score', 0):+.2f}",
                            'Upvotes': p.get('engagement', p.get('score', 0)),
                            'Event': p.get('event_type', 'general'),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No Reddit posts fetched.")
        else:
            st.warning("⚠️ Multi-source sentiment data unavailable.")
            if st.button("🔄 Retry Sentiment Analysis"):
                with st.spinner("Retrying sentiment analysis..."):
                    try:
                        sentiment_result = analyze_stock_sentiment(selected_stock)
                        st.session_state['multi_sentiment'] = sentiment_result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Retry failed: {e}")

    # ==========================================
    # TAB 7: Backtesting
    # ==========================================
    with tab7:
        _hint("What is Backtesting and how to interpret these results?", {
            "Backtesting": "Simulating how the model's buy/sell signals would have performed on past data. It is NOT a guarantee of future returns — think of it as a 'what if' test on historical prices.",
            "Total Return (After Cost)": "The total profit or loss if you had followed every signal, after deducting transaction costs. E.g., +45% means ₹1 lakh would have grown to ₹1.45 lakh.",
            "Pre-cost vs After-cost": "The difference shows how much trading friction (brokerage, taxes, charges) eroded the raw strategy profit. NSE round-trip cost is ~0.1% per trade.",
            "Sharpe Ratio": "Risk-adjusted return. Above 1.0 is considered good; above 2.0 is excellent. It answers: 'How much profit did you earn per unit of risk taken?' A high Sharpe means you didn't take wild risks to earn the return.",
            "Max Drawdown": "The worst peak-to-trough loss during the period. E.g., -25% means at some point your portfolio fell 25% from its peak before recovering. Smaller is better — this measures pain tolerance.",
            "Win Rate": "Percentage of trades that were profitable. 60% means 6 out of every 10 trades made money. High win rate + good Sharpe is the gold standard.",
            "N Trades": "Total number of buy/sell signals generated. Fewer trades = lower transaction costs but potentially missed opportunities. Very high N Trades can indicate over-trading.",
        })
        st.header("🛠️ Strategy Backtest")

        if 'results_df' in st.session_state and st.session_state['results_df'] is not None:
            backtest_data = st.session_state['results_df'].copy()

            if 'Predicted_Return' in backtest_data.columns and len(backtest_data) > 0:
                with st.spinner("Running backtest simulation..."):
                    backtest_data['Signal'] = np.where(backtest_data['Predicted_Return'] > TradingConfig.SIGNAL_THRESHOLD, 1, 0)
                    backtest_data['Signal'] = np.where(backtest_data['Predicted_Return'] < -TradingConfig.SIGNAL_THRESHOLD, -1, backtest_data['Signal'])

                    bt = VectorizedBacktester(backtest_data, backtest_data['Signal'])
                    res = bt.run_backtest()

                    _hint("How to read the Backtest Metrics", {
                        "Equity Curve": "A line chart showing how ₹1,00,000 would have grown (or shrunk) over time if you followed every model signal. Going up = profitable period; going down = drawdown period.",
                        "Sharpe Ratio (detailed)": "Calculated as (Average Daily Return − Risk-Free Rate) ÷ Std Dev of Returns, then annualised. A Sharpe of 1.5 means you earned 1.5 units of return per unit of risk — good for a stock strategy.",
                        "Flat periods on equity curve": "Periods where the model gave no signal (0 = stay out of market). The strategy waits, which is deliberate — it only trades when it has conviction.",
                        "After-Cost Delta (shown in metric card)": "The number in the small text below the main metric. Positive delta means costs were small relative to gross returns. Large negative delta means the strategy over-traded.",
                    })
                    st.markdown(f"""
                    <div style="background: {UIConfig.GRADIENT_DARK}; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                        <h3 style="color: #e94560;">📊 Backtest Performance Summary (After Transaction Costs)</h3>
                        <p style="color: #aaa; font-size: 13px;">NSE round-trip cost 0.1% applied on each trade. Pre-cost vs after-cost shown.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Total Return (After Cost)", f"{res['Total Return']*100:.2f}%",
                              delta=f"Pre-cost: {res.get('Total Return (Gross)', res['Total Return'])*100:.2f}%")
                    b2.metric("Sharpe Ratio", f"{res['Sharpe Ratio']:.2f}")
                    b3.metric("Max Drawdown", f"{res['Max Drawdown']*100:.2f}%")
                    b4.metric("Win Rate", f"{res['Win Rate']*100:.1f}%  ({res.get('N Trades', '?')} trades)")

                    st.markdown("---")

                    # ── Benchmark Comparison Table ──────────────────────────────────
                    _hint("How to read the Benchmark Comparison Table", {
                        "Our Model (5-Model Ensemble)": "The full AI strategy: XGBoost + LightGBM + CatBoost + GRU + Meta-Stacker, all combined. This is the main system.",
                        "XGBoost Only / LGBM Only / CatBoost Only": "Each individual tree model used in isolation as a standalone signal. Shows how much the ensemble adds vs any single model.",
                        "RNN (GRU) Only": "Just the neural network's return prediction used as the signal. Demonstrates the neural net's standalone performance.",
                        "MA Crossover": "Classic simple rule: Buy when 20-day MA crosses above 50-day MA. Used by millions of retail traders.",
                        "52W Momentum": "Buy when price hits a new 52-week high; sell at 52-week low. A well-known factor in Indian markets.",
                        "NIFTY Buy-and-Hold": "Simply buying the NIFTY 50 index and holding it the whole period. If our model can't beat NIFTY B&H after costs, it is not adding value.",
                        "What to look for": "Ideally: 5-Model Ensemble has the highest Total Return and Sharpe, with the smallest Max Drawdown.",
                    })
                    st.subheader("📊 Strategy vs Benchmark Comparison")
                    with st.spinner("Computing individual model + benchmark comparison..."):
                        try:
                            from data.vix_data import fetch_nifty_benchmark
                            from models.backtester import _compute_metrics_from_returns
                            nifty_df = fetch_nifty_benchmark(
                                df_stock.index[0].strftime('%Y-%m-%d'),
                                df_stock.index[-1].strftime('%Y-%m-%d')
                            )
                            nifty_returns = nifty_df['Return'] if (not nifty_df.empty and 'Return' in nifty_df.columns) else None
                            cmp_results = bt.run_benchmark_comparison(
                                close_prices=df_stock['Close'],
                                nifty_returns=nifty_returns
                            )

                            # ── Add individual model rows ─────────────────────
                            _model_cols = {
                                'XGBoost Only': 'XGB_Return',
                                'LightGBM Only': 'LGBM_Return',
                                'CatBoost Only': 'CatBoost_Return',
                                'RNN (GRU) Only': 'RNN_Return',
                            }
                            for model_label, col in _model_cols.items():
                                if col in backtest_data.columns:
                                    _m_sig = np.where(backtest_data[col] > TradingConfig.SIGNAL_THRESHOLD, 1,
                                                      np.where(backtest_data[col] < -TradingConfig.SIGNAL_THRESHOLD, -1, 0))
                                    _m_ret = (_m_sig * backtest_data['Actual_Return'].values)
                                    _m_core = _compute_metrics_from_returns(_m_ret)
                                    cmp_results[model_label] = {
                                        'Total Return (%)': round(_m_core['total_return'] * 100, 2),
                                        'Sharpe Ratio': round(_m_core['sharpe_ratio'], 2),
                                        'Max Drawdown (%)': round(_m_core['max_drawdown'] * 100, 2),
                                        'Win Rate (%)': round(_m_core['win_rate'] * 100, 1),
                                        'N Trades': int(np.sum(np.abs(np.diff(np.concatenate([[0], _m_sig]))) > 0)),
                                    }

                            if cmp_results:
                                # Build display table with ordering: Ensemble first, then individual, then simple baselines
                                _order = ['Our Model', 'XGBoost Only', 'LightGBM Only', 'CatBoost Only',
                                          'RNN (GRU) Only', 'MA Crossover (20/50)', '52W Momentum', 'NIFTY 50 B&H']
                                cmp_rows = []
                                _seen = set()
                                for name in _order + list(cmp_results.keys()):
                                    if name in cmp_results and name not in _seen:
                                        _seen.add(name)
                                        sr = cmp_results[name]
                                        cmp_rows.append({
                                            'Strategy': name,
                                            'Total Return (%)': sr.get('Total Return (%)', '—'),
                                            'Sharpe Ratio': sr.get('Sharpe Ratio', '—'),
                                            'Max Drawdown (%)': sr.get('Max Drawdown (%)', '—'),
                                            'Win Rate (%)': sr.get('Win Rate (%)', '—'),
                                            'N Trades': sr.get('N Trades', '—'),
                                        })
                                cmp_df = pd.DataFrame(cmp_rows)

                                def _hl_best(s):
                                    """Highlight the best value in each numeric column."""
                                    styles = [''] * len(s)
                                    try:
                                        vals = pd.to_numeric(s, errors='coerce')
                                        if s.name in ('Total Return (%)', 'Sharpe Ratio', 'Win Rate (%)'):
                                            best = vals.idxmax()
                                        elif s.name == 'Max Drawdown (%)':
                                            best = vals.idxmax()  # max drawdown is negative; largest = least bad
                                        else:
                                            return styles
                                        styles[best] = 'background-color: rgba(0,212,255,0.25); font-weight:700'
                                    except Exception:
                                        pass
                                    return styles

                                st.dataframe(
                                    cmp_df.style.apply(_hl_best, axis=0, subset=['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']),
                                    use_container_width=True, hide_index=True
                                )
                        except Exception as e_cmp:
                            st.caption(f"Benchmark comparison unavailable: {str(e_cmp)[:120]}")

                    st.markdown("---")

                    # ── Equity Curve ─────────────────────────────────────────────────
                    st.subheader("📈 Equity Curve")
                    fig_equity = go.Figure()
                    fig_equity.add_trace(go.Scatter(
                        x=res['Equity Curve'].index, y=res['Equity Curve'].values,
                        name="Our Strategy", fill='tozeroy',
                        fillcolor='rgba(0, 212, 255, 0.2)',
                        line=dict(color=UIConfig.COLOR_PRIMARY, width=2)
                    ))
                    fig_equity.update_layout(template="plotly_dark", title="Strategy Equity Curve (₹100,000 Initial)")
                    st.plotly_chart(fig_equity, use_container_width=True)

                    # ── Monte Carlo Fan Chart ─────────────────────────────────────────
                    _hint("How to read the Monte Carlo Simulation", {
                        "Monte Carlo Simulation": "Runs 1,000 imaginary 'alternate histories' by randomly reshuffling the actual daily returns this strategy produced. Each path represents a different order in which those same returns could have occurred.",
                        "Why shuffle returns?": "The real performance you see in the equity curve depended partly on the order trades happened. If good days came first (lucky) or bad days came first (unlucky), the final result differs. Monte Carlo shows the full range of realistic outcomes.",
                        "Median Path (blue bold line)": "The middle path — 500 of the 1,000 simulations ended above this, 500 below. This is the 'expected' outcome if luck averages out.",
                        "P5–P95 Band (shaded area)": "90% of the 1,000 simulations ended inside this range. The P5 (bottom red dotted) is the unlucky scenario; P95 (top green dotted) is the lucky scenario.",
                        "Probability of Profit": "What percentage of the 1,000 simulations ended with a positive return. E.g., 72% means 720 out of 1,000 alternate histories were profitable.",
                        "Median Max Drawdown": "In the typical (median) simulation, this is the worst peak-to-trough loss experienced. Smaller means the strategy is more resilient to bad luck.",
                        "P5 / P95 Return": "The return in the worst 5% of simulations (P5) and the best 5% of simulations (P95). This is your realistic downside vs upside range.",
                    })
                    st.subheader("🎲 Monte Carlo Simulation (1,000 Paths)")
                    with st.spinner("Running Monte Carlo..."):
                        try:
                            mc = bt.run_monte_carlo(n_simulations=1000)
                            dates_mc = res['Equity Curve'].index
                            n_mc = len(mc['equity_p50'])
                            mc_index = dates_mc[-n_mc:] if len(dates_mc) >= n_mc else dates_mc

                            fig_mc = go.Figure()
                            fig_mc.add_trace(go.Scatter(
                                x=list(mc_index) + list(mc_index[::-1]),
                                y=list(mc['equity_p95']) + list(mc['equity_p5'][::-1]),
                                fill='toself', fillcolor='rgba(0,212,255,0.1)',
                                line=dict(color='rgba(0,0,0,0)'), name='P5–P95 Band'
                            ))
                            fig_mc.add_trace(go.Scatter(
                                x=mc_index, y=mc['equity_p50'],
                                name='Median Path', line=dict(color='#00d4ff', width=2)
                            ))
                            fig_mc.add_trace(go.Scatter(
                                x=mc_index, y=mc['equity_p5'],
                                name='P5 (Worst)', line=dict(color='#ff6b6b', width=1, dash='dot')
                            ))
                            fig_mc.add_trace(go.Scatter(
                                x=mc_index, y=mc['equity_p95'],
                                name='P95 (Best)', line=dict(color='#7bed9f', width=1, dash='dot')
                            ))
                            fig_mc.update_layout(
                                template="plotly_dark",
                                title=f"Monte Carlo: Median Return {mc['median_return']*100:.1f}% | P5: {mc['p5_return']*100:.1f}% | P95: {mc['p95_return']*100:.1f}% | P(Positive): {mc['prob_positive']*100:.0f}%",
                                yaxis_title="Equity (₹ normalised to 1.0)"
                            )
                            st.plotly_chart(fig_mc, use_container_width=True)

                            mc_c1, mc_c2, mc_c3 = st.columns(3)
                            mc_c1.metric("Probability of Profit", f"{mc['prob_positive']*100:.0f}%")
                            mc_c2.metric("Median Max Drawdown", f"{mc['median_max_drawdown']*100:.1f}%")
                            mc_c3.metric("P5 / P95 Return", f"{mc['p5_return']*100:.1f}% / {mc['p95_return']*100:.1f}%")

                        except Exception as e_mc:
                            st.caption(f"Monte Carlo unavailable: {str(e_mc)[:80]}")

                    # ── Pipeline Timing Table ─────────────────────────────────────────
                    st.markdown("---")
                    st.subheader("⏱️ Pipeline Step Timings")
                    _pt_data = st.session_state.get('pipeline_timings', {})
                    if _pt_data:
                        _timing_rows = [
                            {'Step': k, 'Time (seconds)': v}
                            for k, v in _pt_data.items()
                            if not k.startswith('_')
                        ]
                        _timing_total = _pt_data.get('_total_prefetch', 0)
                        _timing_df = pd.DataFrame(_timing_rows)
                        # Highlight the slowest step
                        def _hl_slow(s):
                            if s.name != 'Time (seconds)':
                                return [''] * len(s)
                            mx = s.max()
                            return [
                                'background-color: rgba(255,107,107,0.25); font-weight:700' if v == mx else ''
                                for v in s
                            ]
                        st.dataframe(
                            _timing_df.style.apply(_hl_slow, axis=0),
                            use_container_width=True, hide_index=True
                        )
                        st.caption(f"Total pre-fetch time: {_timing_total}s  |  Slowest step highlighted in red.")
                    else:
                        st.caption("Timing data not available — re-run analysis to capture timings.")
            else:
                st.warning("⚠️ Insufficient prediction data for backtesting.")
        else:
            st.info("👈 Please run stock analysis from the Dashboard tab first.")

    # ==========================================
    # TAB 8: Pattern Analysis (Mathematical)
    # ==========================================
    with tab8:
        _hint("What are Chart Patterns and how does this work?", {
            "Chart Patterns": "Repeating price formations that have historically been followed by predictable moves. Examples: Head & Shoulders (bearish reversal), Double Bottom (bullish reversal), Ascending Triangle (bullish continuation).",
            "Bullish Pattern": "Price formation suggesting the stock is likely to go UP. Shown in green. Examples: Double Bottom, Inverse Head & Shoulders, Ascending Channel.",
            "Bearish Pattern": "Price formation suggesting the stock is likely to go DOWN. Shown in red. Examples: Head & Shoulders, Double Top, Descending Wedge.",
            "Neutral / Continuation Pattern": "The trend is likely to CONTINUE in the same direction. Shown in orange. Examples: Symmetric Triangle, Flag, Rectangle.",
            "Confidence %": "How closely the price data matches the textbook pattern definition. 90% means a very clean, clear pattern. 60% means a rough match — treat with caution.",
            "Hurst Exponent (H)": "A mathematical measure of whether the market is currently trending, mean-reverting, or random. H > 0.55 = Trending (momentum strategies work better). H < 0.45 = Mean-Reverting (contrarian strategies work better). H ≈ 0.5 = Random Walk (no clear edge from pattern trading alone).",
            "Volume Confirmed 🔊 Vol✓": "The pattern's breakout bar had volume more than 1.5× the 20-day average. High volume on a breakout is a strong confirmation that the move is real, not a fake-out.",
            "Multi-Timeframe 🔗 Multi-TF": "This same pattern was ALSO detected on the weekly chart (not just daily). When a pattern appears on multiple timeframes simultaneously, it is far more reliable.",
            "Target Price": "The theoretical price target if the pattern plays out. Calculated mathematically from the pattern's height (e.g., for Head & Shoulders: neckline − head height).",
            "Support level (green line S)": "A price floor where the stock has historically bounced up. Buyers tend to step in here. The more times price has bounced from this level, the stronger the support.",
            "Resistance level (red line R)": "A price ceiling where the stock has historically stalled or fallen. Sellers tend to emerge here. A break above resistance with high volume is a strong bullish signal.",
        })
        st.header("📐 Mathematical Pattern Analysis")

        st.markdown(f"""
        <div style="background: {UIConfig.GRADIENT_DARK}; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
            <h4 style="color: #e94560;">🔬 Proven Pattern Detection</h4>
            <p style="color: #eee;">Uses scipy peak/trough detection and classic technical analysis algorithms.</p>
            <p style="color: #888; font-size: 12px;">No experimental ML - mathematically validated patterns only.</p>
        </div>
        """, unsafe_allow_html=True)

        from models.visual_analyst import PatternAnalyst
        
        # Use pre-calculated analysis from loading step
        analysis = st.session_state.get('pattern_analysis')
        
        if not analysis:
            # Fallback if not in session state
            analyst = PatternAnalyst(order=5)
            analysis = analyst.analyze_all_patterns(df_stock)
        
        # Overall Bias
        bias = analysis['overall_bias']
        bias_color = UIConfig.COLOR_BULLISH if bias == "Bullish" else UIConfig.COLOR_BEARISH if bias == "Bearish" else UIConfig.COLOR_NEUTRAL

        hurst_val_pa = analysis.get('hurst_exponent')
        market_char_pa = analysis.get('market_character', '')
        hurst_display = f" | Hurst: {hurst_val_pa:.3f} ({market_char_pa})" if hurst_val_pa is not None else ""

        st.markdown(f"""
        <div style="background: {UIConfig.GRADIENT_DARK}; padding: 15px; border-radius: 10px; border-left: 4px solid {bias_color}; margin-bottom: 20px;">
            <h3 style="color: {bias_color}; margin: 0;">Overall Bias: {bias}</h3>
            <p style="color: #aaa; margin: 5px 0 0 0;">{analysis['pattern_count']} patterns detected{hurst_display}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.subheader("Detected Patterns")
            
            # ---------------------------------------------
            # Interactive Re-Analysis Tool
            # ---------------------------------------------
            with st.expander("🔄 Re-Analyze / Zoom", expanded=True):
                st.caption("Adjust the analysis window to detect patterns in specific ranges.")
                analysis_window = st.slider("Analysis Window (Days)", min_value=30, max_value=365, value=60)
                
                if st.button("🔍 Run Advanced AI Vision"):
                    with st.spinner("Processing image with Roboflow Vision AI..."):
                        # Re-run analysis on sliced data
                        df_slice = df_stock.tail(analysis_window)
                        analyst = PatternAnalyst()
                        new_patterns = analyst.analyze_patterns_with_vision(df_slice)
                        
                        # Merge with math patterns for full view
                        math_patterns = analyst.analyze_all_patterns(df_slice) # This includes both now
                        st.session_state['pattern_analysis'] = math_patterns
                        analysis = math_patterns
                        st.success(f"Analysed {analysis_window} days. Found {len(new_patterns)} vision patterns.")
                        st.rerun()

            # Display Patterns
            if analysis['patterns']:
                for p in analysis['patterns']:
                    p_type = p.get('Type', '')
                    color = "green" if "Bullish" in p_type else "red" if "Bearish" in p_type else "orange"
                    vol_badge = " 🔊 Vol✓" if p.get('volume_confirmed') else ""
                    tf_badge = " 🔗 Multi-TF" if p.get('Timeframe_Confluence') else ""

                    with st.container():
                        st.markdown(f"""
                        <div style="border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 10px;">
                            <strong style="font-size: 18px;">{p['Pattern']}</strong>{vol_badge}{tf_badge}<br>
                            <span style="color: {color};">{p_type}</span> • <span style="color: #aaa;">Confidence: {p['Confidence']}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if 'Meta' in p:  # Debug info for vision patterns
                            with st.expander("🛠️ AI Debug Data"):
                                st.json(p['Meta'])
            else:
                st.info("No high-confidence patterns detected in current window.")
                
        with col_p2:
            st.subheader("Pattern Visualization")
            
            # Create Chart with Annotations
            fig_pat = go.Figure()
            
            # Use the relevant window (default 60 or session state)
            window_size = 60 # Default
            # If we tracked the window used for analysis, use it. For now, match the standard view.
            df_viz = df_stock.tail(window_size) 
            
            fig_pat.add_trace(go.Candlestick(
                x=df_viz.index,
                open=df_viz['Open'], high=df_viz['High'],
                low=df_viz['Low'], close=df_viz['Close'],
                name='Price'
            ))
            
            # Overlay Pattern Labels on Chart
            y_positions = []
            chart_high = df_viz['High'].max()
            chart_low = df_viz['Low'].min()
            y_step = (chart_high - chart_low) * 0.06  # 6% spacing between labels
            
            for idx, p in enumerate(analysis['patterns']):
                p_type = p.get('Type', '')
                conf = p.get('Confidence', 0)
                target = p.get('Target', 'N/A')
                
                # Determine color
                if 'Bullish' in p_type:
                    ann_color = '#00ff88'
                elif 'Bearish' in p_type:
                    ann_color = '#ff4444'
                else:
                    ann_color = '#ffa500'
                
                # Stagger label Y positions to avoid overlap
                y_pos = chart_high + y_step * (idx + 1)
                
                # Draw neckline if available
                viz_x0 = df_viz.index[0].strftime('%Y-%m-%d')
                viz_x1 = df_viz.index[-1].strftime('%Y-%m-%d')
                if 'Neckline' in p:
                    fig_pat.add_shape(type="line",
                        x0=viz_x0, x1=viz_x1,
                        y0=p['Neckline'], y1=p['Neckline'],
                        line=dict(color=ann_color, width=1.5, dash="dash"),
                    )
                    fig_pat.add_annotation(
                        x=viz_x0, y=p['Neckline'],
                        text=f"{p['Pattern']} Neckline",
                        showarrow=False, font=dict(color=ann_color, size=9),
                        xanchor="left", yshift=10
                    )

                # Add compact label for every pattern
                label_text = f"{p['Pattern']} ({conf:.0f}%)"
                if target != 'N/A':
                    try:
                        label_text += f" → ₹{float(target):,.0f}"
                    except (ValueError, TypeError):
                        pass

                fig_pat.add_annotation(
                    x=viz_x1, y=y_pos,
                    text=label_text,
                    showarrow=False,
                    font=dict(color="white", size=10),
                    bgcolor=ann_color,
                    bordercolor=ann_color,
                    borderwidth=1,
                    borderpad=3,
                    xanchor="right",
                    opacity=0.9
                )
            
            fig_pat.update_layout(
                template="plotly_dark",
                title=f"Pattern Analysis ({window_size} Day View)",
                height=550,
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_pat, use_container_width=True)
            # Trend Analysis
            _hint("How to read Trend Analysis", {
                "Trend (Bullish/Bearish/Neutral)": "Determined by comparing today's price to the 20-day and 50-day moving averages. Bullish = price above both MAs. Bearish = price below both MAs.",
                "Trend Strength %": "How far the current price is from its moving averages, expressed as a percentage. Higher = stronger trend. Below 30% often means the trend is weak or near exhaustion.",
                "MA Signal": "Moving Average crossover status. 'Golden Cross' (20-day crosses above 50-day) = classically bullish. 'Death Cross' (20-day crosses below 50-day) = classically bearish.",
                "Price Structure": "Higher Highs & Higher Lows = confirmed uptrend. Lower Highs & Lower Lows = confirmed downtrend. Mixed = sideways/consolidation.",
                "Slope": "The angle of the best-fit line through recent prices. Positive slope = going up. Larger absolute value = steeper trend. Near 0 = flat/sideways.",
            })
            st.subheader("📈 Trend Analysis")
            trend = analysis['trend']

            t1, t2, t3 = st.columns(3)
            trend_color = UIConfig.COLOR_BULLISH if trend['Trend'] == "Bullish" else UIConfig.COLOR_BEARISH if trend['Trend'] == "Bearish" else UIConfig.COLOR_NEUTRAL
            t1.metric("Trend", trend['Trend'])
            t2.metric("Strength", f"{trend['Strength']:.1f}%")
            t3.metric("MA Signal", trend['MA_Signal'])

            st.info(f"Price Structure: **{trend['Structure']}** | Slope: {trend['Slope']:.4f}")
            
        with col_p2:
            # Support/Resistance
            st.subheader("🎯 Support & Resistance")
            sr = analysis['support_resistance']
            
            s1, s2 = st.columns(2)
            s1.metric("Nearest Resistance", f"₹{sr['Nearest_Resistance']}" if sr['Nearest_Resistance'] != 'N/A' else "N/A")
            s2.metric("Nearest Support", f"₹{sr['Nearest_Support']}" if sr['Nearest_Support'] != 'N/A' else "N/A")
            
            st.caption(f"Current Price: ₹{sr['Current_Price']}")
        
        st.markdown("---")
        
        # Detected Patterns
        st.subheader("🔍 Detected Chart Patterns")
        
        if analysis['patterns']:
            # Check for Vision AI Pattern first
            vision_pattern = next((p for p in analysis['patterns'] if p['Pattern'] == 'Vision AI Analysis'), None)
            if vision_pattern:
                st.subheader("👁️ Vision AI Analysis")
                st.info("Direct Object Detection Results from Roboflow")
                st.json(vision_pattern.get('Meta', {}))
                st.markdown("---")

            patterns_df = pd.DataFrame(analysis['patterns'])
            
            # Clean up: select meaningful columns, remove None/NaN, format numbers
            core_cols = ['Pattern', 'Type', 'Confidence', 'Target', 'Status']
            extra_cols = ['Neckline', 'Peak_Price', 'Trough_Price', 'Head_Price', 'Channel_Width']
            
            # Only include extra columns that have at least one non-None value
            display_cols = core_cols.copy()
            for col in extra_cols:
                if col in patterns_df.columns and patterns_df[col].notna().any() and not (patterns_df[col] == 'N/A').all():
                    display_cols.append(col)
            
            # Filter to existing columns
            display_cols = [c for c in display_cols if c in patterns_df.columns]
            patterns_df = patterns_df[display_cols]
            
            # Format numeric columns
            for col in patterns_df.columns:
                if col in ['Neckline', 'Target', 'Peak_Price', 'Trough_Price', 'Head_Price']:
                    patterns_df[col] = patterns_df[col].apply(
                        lambda x: f'₹{float(x):,.2f}' if x is not None and x != 'N/A' and str(x) != 'nan' else '—'
                    )
                elif col == 'Confidence':
                    patterns_df[col] = patterns_df[col].apply(lambda x: f'{x:.1f}%')
            
            # Fill remaining NaN/None with em-dash
            patterns_df = patterns_df.fillna('—').replace('None', '—').replace('nan', '—')
            
            # Sort by confidence (need to parse back for sorting)
            patterns_df = patterns_df.reset_index(drop=True)
            
            # Color code by type
            def highlight_pattern(row):
                if 'Bullish' in row.get('Type', ''):
                    return ['background-color: rgba(0, 255, 136, 0.15)'] * len(row)
                elif 'Bearish' in row.get('Type', ''):
                    return ['background-color: rgba(255, 68, 68, 0.15)'] * len(row)
                return ['background-color: rgba(255, 165, 0, 0.1)'] * len(row)
            
            st.dataframe(patterns_df.style.apply(highlight_pattern, axis=1), use_container_width=True, hide_index=True)
        else:
            st.info("📊 No classic chart patterns detected in recent price action. This could indicate consolidation or a range-bound market.")
        
        # CLEAN Chart - Just candlestick with support/resistance
        st.subheader("📈 Price Chart with Key Levels")
        
        fig_pattern = go.Figure()
        
        # Candlestick - last 60 days for cleaner view
        df_chart = df_stock.tail(60)
        fig_pattern.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart['Open'],
            high=df_chart['High'],
            low=df_chart['Low'],
            close=df_chart['Close'],
            name='Price',
            increasing_line_color=UIConfig.COLOR_BULLISH,
            decreasing_line_color=UIConfig.COLOR_BEARISH
        ))
        
        # Add ONLY support/resistance lines (clean)
        sr = analysis['support_resistance']
        _resist = sr['Nearest_Resistance']
        _support = sr['Nearest_Support']
        if _resist != 'N/A' and isinstance(_resist, (int, float)):
            fig_pattern.add_hline(
                y=float(_resist),
                line_dash="dash",
                line_color="rgba(255, 68, 68, 0.7)",
                line_width=2,
                annotation_text=f"R: ₹{float(_resist):.0f}",
                annotation_position="right"
            )
        if _support != 'N/A' and isinstance(_support, (int, float)):
            fig_pattern.add_hline(
                y=float(_support),
                line_dash="dash",
                line_color="rgba(0, 255, 136, 0.7)",
                line_width=2,
                annotation_text=f"S: ₹{float(_support):.0f}",
                annotation_position="right"
            )
        
        # Annotate top 3 patterns compactly in top-right corner
        if analysis['patterns']:
            sorted_patterns = sorted(analysis['patterns'], key=lambda x: x.get('Confidence', 0), reverse=True)
            chart_high = df_chart['High'].max()
            chart_low = df_chart['Low'].min()
            y_step = (chart_high - chart_low) * 0.05
            chart_x1 = df_chart.index[-1].strftime('%Y-%m-%d')

            for idx, p in enumerate(sorted_patterns[:3]):
                pattern_name = p.get('Pattern', 'Unknown')
                pattern_type = p.get('Type', '')
                confidence = p.get('Confidence', 0)
                target = p.get('Target', 'N/A')

                marker_color = UIConfig.COLOR_BULLISH if 'Bullish' in pattern_type else UIConfig.COLOR_BEARISH if 'Bearish' in pattern_type else '#ffa500'

                # Build compact label
                label = f"<b>{pattern_name}</b> ({confidence:.0f}%)"
                if target != 'N/A':
                    try:
                        label += f" → ₹{float(target):,.0f}"
                    except (ValueError, TypeError):
                        pass

                # Stack annotations vertically from top
                fig_pattern.add_annotation(
                    x=chart_x1,
                    y=chart_high + y_step * (idx + 1),
                    text=label,
                    showarrow=False,
                    font=dict(color="white", size=10),
                    bgcolor=marker_color,
                    bordercolor=marker_color,
                    borderwidth=1,
                    borderpad=4,
                    xanchor="right",
                    opacity=0.9
                )

            # Show remaining count if more than 3
            remaining = len(sorted_patterns) - 3
            if remaining > 0:
                fig_pattern.add_annotation(
                    x=chart_x1,
                    y=chart_high + y_step * 4,
                    text=f"<i>+{remaining} more patterns</i>",
                    showarrow=False,
                    font=dict(color="#888", size=9),
                    xanchor="right"
                )
        
        fig_pattern.update_layout(
            template="plotly_dark", 
            height=500, 
            xaxis_rangeslider_visible=False,
            title=f"Price Chart | Trend: {analysis['trend']['Trend']} ({analysis['trend']['Strength']:.0f}%)",
            showlegend=False,
            margin=dict(r=100)
        )
        st.plotly_chart(fig_pattern, use_container_width=True)
        
        # Pattern Guide
        with st.expander("📚 Pattern Guide"):
            st.markdown("""
            **Reversal Patterns:**
            - 🟢 **Double Bottom** — Two troughs at similar levels → bullish reversal
            - 🟢 **Inverse Head & Shoulders** — Three troughs, middle deepest → strong bullish
            - 🔴 **Double Top** — Two peaks at similar levels → bearish reversal
            - 🔴 **Head & Shoulders** — Three peaks, middle highest → strong bearish
            
            **Continuation & Geometric:**
            - 🟢 **Ascending Triangle** — Flat resistance + rising support → bullish breakout
            - 🔴 **Descending Triangle** — Falling resistance + flat support → bearish breakout
            - ⚪ **Symmetrical Triangle** — Converging slopes → breakout either direction
            - 🟢 **Falling Wedge** — Both slopes falling, converging → bullish reversal
            - 🔴 **Rising Wedge** — Both slopes rising, converging → bearish reversal
            
            **Channel & Structural:**
            - 📊 **Ascending/Descending/Horizontal Channel** — Parallel trend corridors
            - 📈 **Higher Highs & Higher Lows** — Active uptrend structure (very reliable)
            - 📉 **Lower Highs & Lower Lows** — Active downtrend structure
            - ⏳ **Consolidation / Squeeze** — Tight range, imminent breakout expected
            
            **Confidence Score:** Higher = more aligned peaks/troughs. Target = measured move.
            """)

else:
    # Welcome screen when no analysis has been run yet
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px;">
        <h2 style="color: #e94560;">👋 Welcome to ProTrader AI</h2>
        <p style="color: #aaa; font-size: 18px;">Professional Stock Analytics Platform for Indian Markets</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; text-align: center;">
            <h3 style="color: #00ff88;">🤖 AI-Powered</h3>
            <p style="color: #aaa;">Hybrid XGBoost + GRU models with dynamic ensemble weighting</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; text-align: center;">
            <h3 style="color: #e94560;">📊 Multi-Source Data</h3>
            <p style="color: #aaa;">FII/DII flows, VIX, sentiment from news, Reddit, Google Trends</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; text-align: center;">
            <h3 style="color: #ffd700;">📐 Pattern Detection</h3>
            <p style="color: #aaa;">Mathematical chart pattern recognition with support/resistance</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h3 style="color: #fff;">🚀 Get Started</h3>
        <p style="color: #aaa; font-size: 16px;">
            1. Select a stock from the sidebar<br>
            2. Choose your date range<br>
            3. Click <strong style="color: #e94560;">"Launch Analysis"</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

