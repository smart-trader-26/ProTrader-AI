"""
Chart creation utilities using Plotly.
Functions for candlestick, accuracy, weights, and forecast visualizations.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config.settings import UIConfig


def create_candlestick_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a candlestick chart from OHLCV data.
    
    Args:
        df: DataFrame with Open, High, Low, Close columns
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    )])
    fig.update_layout(
        title="Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    return fig


def create_accuracy_comparison_chart(df_stock: pd.DataFrame, 
                                      results_df: pd.DataFrame, 
                                      future_prices: pd.DataFrame) -> go.Figure:
    """
    Create a comprehensive chart showing:
    1. Historical actual prices
    2. Model predictions on test data
    3. Future forecast
    
    Args:
        df_stock: Historical stock data
        results_df: Test period results with Actual_Return and Predicted_Return
        future_prices: Future price predictions
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # 1. Historical prices (full dataset)
    fig.add_trace(go.Scatter(
        x=df_stock.index,
        y=df_stock['Close'],
        name='Actual Price (Historical)',
        line=dict(color=UIConfig.COLOR_PRIMARY, width=2),
        mode='lines'
    ))
    
    # 2. Convert returns to prices for test period
    if not results_df.empty and 'Actual_Return' in results_df.columns:
        try:
            # Get the starting price for test period
            test_start_idx = df_stock.index.get_loc(results_df.index[0])
            start_price = df_stock['Close'].iloc[test_start_idx - 1]
            
            # FIX: Use actual prices as anchor points to prevent divergence
            # Instead of cumulative predicted returns, apply predicted return to previous ACTUAL price
            actual_prices_test = []
            predicted_prices_test = []
            
            for i in range(len(results_df)):
                # Get actual price at this point
                actual_price = df_stock['Close'].iloc[test_start_idx + i]
                actual_prices_test.append(actual_price)
                
                # Apply predicted return to previous actual price (anchored approach)
                if i == 0:
                    prev_actual = start_price
                else:
                    prev_actual = df_stock['Close'].iloc[test_start_idx + i - 1]
                
                predicted_price = prev_actual * np.exp(results_df['Predicted_Return'].iloc[i])
                predicted_prices_test.append(predicted_price)
            
            # Validation: Check if predictions are within reasonable bounds (50% of actual)
            if predicted_prices_test and actual_prices_test:
                avg_actual = np.mean(actual_prices_test)
                avg_predicted = np.mean(predicted_prices_test)
                divergence = abs(avg_predicted - avg_actual) / avg_actual
                
                if divergence > 0.5:
                    # If predictions diverge too much, log warning and use simpler approach
                    import streamlit as st
                    st.warning(f"⚠️ Prediction divergence detected ({divergence*100:.1f}%). Chart may be inaccurate.")
            
            # Plot predicted prices on test data
            fig.add_trace(go.Scatter(
                x=results_df.index,
                y=predicted_prices_test,
                name='Model Prediction (Test Period)',
                line=dict(color=UIConfig.COLOR_SECONDARY, width=2, dash='dot'),
                mode='lines'
            ))
        except Exception:
            pass  # Skip if index lookup fails
    
    # 3. Future forecast
    if not future_prices.empty:
        fig.add_trace(go.Scatter(
            x=future_prices.index,
            y=future_prices['Predicted Price'],
            name='Future Forecast',
            line=dict(color=UIConfig.COLOR_BULLISH, width=2, dash='dash'),
            mode='lines+markers',
            marker=dict(size=6)
        ))
    
    # Add vertical line separating historical and forecast
    if not future_prices.empty:
        # Convert timestamp to string to avoid Plotly annotation bug
        last_date = df_stock.index[-1]
        if hasattr(last_date, 'isoformat'):
            last_date_str = last_date.isoformat()
        else:
            last_date_str = str(last_date)
        
        # Add vertical line without annotation (annotation causes Timestamp bug)
        fig.add_shape(
            type="line",
            x0=last_date_str,
            x1=last_date_str,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dash")
        )
        # Add annotation separately
        fig.add_annotation(
            x=last_date_str,
            y=1.02,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            font=dict(color="gray", size=12)
        )
    
    fig.update_layout(
        template="plotly_dark",
        title="📊 Model Accuracy Visualization: Actual vs Predicted Prices",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode='x unified',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_dynamic_weights_visualization(weights_history: dict) -> go.Figure:
    """
    Create visualization for dynamic model weights over time.
    
    Args:
        weights_history: Dictionary with dates as keys and weight dicts as values
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    dates = list(weights_history.keys())
    tech_weights = [w['technical'] for w in weights_history.values()]
    sent_weights = [w['sentiment'] for w in weights_history.values()]
    vol_weights = [w['volatility'] for w in weights_history.values()]
    
    fig.add_trace(go.Scatter(
        x=dates, y=tech_weights,
        mode='lines+markers',
        name='Technical Model Weight',
        line=dict(color='blue', width=2),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=sent_weights,
        mode='lines+markers',
        name='Sentiment Model Weight',
        line=dict(color='green', width=2),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=vol_weights,
        mode='lines+markers',
        name='Volatility Model Weight',
        line=dict(color='red', width=2),
        stackgroup='one'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title='Dynamic Model Weights Over Time',
        xaxis_title='Date',
        yaxis_title='Weight',
        hovermode='x unified',
        yaxis=dict(tickformat='.0%'),
        showlegend=True
    )
    
    return fig


def create_uncertainty_visualization(uncertainties_history: dict) -> go.Figure:
    """
    Create visualization for model uncertainties over time.
    
    Args:
        uncertainties_history: Dictionary with dates as keys and uncertainty dicts as values
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    dates = list(uncertainties_history.keys())
    tech_unc = [u['technical'] for u in uncertainties_history.values()]
    sent_unc = [u['sentiment'] for u in uncertainties_history.values()]
    vol_unc = [u['volatility'] for u in uncertainties_history.values()]
    
    fig.add_trace(go.Scatter(
        x=dates, y=tech_unc,
        mode='lines',
        name='Technical Model Uncertainty',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=sent_unc,
        mode='lines',
        name='Sentiment Model Uncertainty',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=vol_unc,
        mode='lines',
        name='Volatility Model Uncertainty',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title='Model Uncertainties Over Time',
        xaxis_title='Date',
        yaxis_title='Uncertainty (σ²)',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def create_model_performance_radar(weights: dict, uncertainties: dict) -> go.Figure:
    """
    Create radar chart for model performance comparison.
    
    Args:
        weights: Dictionary of model weights
        uncertainties: Dictionary of model uncertainties
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    models = ['Technical', 'Sentiment', 'Volatility']
    
    # Inverse of uncertainty = confidence
    confidences = [1/(u+1e-6) for u in uncertainties.values()]
    weight_vals = list(weights.values())

    fig.add_trace(go.Scatterpolar(
        r=weight_vals,
        theta=models,
        fill='toself',
        name='Model Weights',
        line_color='blue'
    ))

    fig.add_trace(go.Scatterpolar(
        r=confidences,
        theta=models,
        fill='toself',
        name='Model Confidence',
        line_color='red'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(weight_vals), max(confidences))]
            )),
        showlegend=True,
        title='Model Performance Radar Chart',
        template="plotly_dark"
    )
    
    return fig


def create_forecast_chart(historical_df: pd.DataFrame, 
                          future_df: pd.DataFrame,
                          stock_name: str,
                          forecast_days: int) -> go.Figure:
    """
    Create a forecast visualization chart.
    
    Args:
        historical_df: Historical stock data
        future_df: Future price predictions
        stock_name: Stock symbol name
        forecast_days: Number of forecast days
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_df.index[-60:], 
        y=historical_df['Close'][-60:], 
        name="Historical",
        line=dict(color=UIConfig.COLOR_PRIMARY, width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=future_df.index, 
        y=future_df['Predicted Price'], 
        name="AI Forecast", 
        line=dict(dash='dot', color=UIConfig.COLOR_SECONDARY, width=2)
    ))
    
    fig.update_layout(
        template="plotly_dark",
        title=f"{stock_name} - {forecast_days} Day Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode='x unified'
    )
    
    return fig


def create_calibration_chart(report: dict) -> go.Figure:
    """
    Reliability diagram for the calibrated directional classifier.

    A well-calibrated model's markers fall on the y = x diagonal. Bubble
    size encodes the number of test samples in each probability bin, so the
    viewer can tell a tight 5-sample bin from a noisy 100-sample one.

    Args:
        report: Dict from hybrid_model._compute_calibration_report
                (keys: bin_predicted, bin_actual, bin_counts, ece, brier_score, n_samples)

    Returns:
        Plotly Figure. Empty (with an annotation) when no test samples.
    """
    preds = report.get("bin_predicted") or []
    acts = report.get("bin_actual") or []
    counts = report.get("bin_counts") or []
    n = int(report.get("n_samples", 0))
    ece = float(report.get("ece", 0.0))
    brier = float(report.get("brier_score", 0.0))

    fig = go.Figure()
    # Perfect calibration reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color="#888", dash="dash"),
        name="Perfect calibration",
        hoverinfo="skip",
    ))

    if preds and acts:
        sizes = np.sqrt(np.array(counts, dtype=float))
        sizes = 6 + 30 * (sizes / sizes.max()) if sizes.max() > 0 else sizes + 6
        fig.add_trace(go.Scatter(
            x=preds, y=acts,
            mode="markers+lines",
            marker=dict(size=sizes, color=UIConfig.COLOR_PRIMARY,
                        line=dict(color="white", width=1)),
            line=dict(color=UIConfig.COLOR_PRIMARY, width=2),
            name="Observed",
            text=[f"n={c}" for c in counts],
            hovertemplate="predicted=%{x:.2f}<br>actual=%{y:.2f}<br>%{text}<extra></extra>",
        ))
    else:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="No test-set probabilities available",
            showarrow=False, font=dict(color="#888"),
        )

    # Color ECE badge by quality (A1 target: ECE ≤ 0.05)
    if ece <= 0.05:
        ece_color = "#3ddc97"    # green — well-calibrated
    elif ece <= 0.10:
        ece_color = "#f4a261"    # amber — workable
    else:
        ece_color = "#e76f51"    # red — untrustworthy

    fig.update_layout(
        template="plotly_dark",
        title=(
            f"Calibration Reliability "
            f"(<span style='color:{ece_color}'>ECE = {ece*100:.2f}%</span>, "
            f"Brier = {brier:.3f}, n = {n})"
        ),
        xaxis=dict(title="Predicted P(up)", range=[0, 1], tickformat=".0%"),
        yaxis=dict(title="Observed hit rate", range=[0, 1], tickformat=".0%"),
        showlegend=True,
        height=420,
    )
    return fig


def create_accuracy_rollup_chart(rollups: dict) -> go.Figure:
    """
    Bar chart of rolling directional accuracy at 7d / 30d / 90d windows (A7.4).

    Args:
        rollups: {"7d": AccuracyWindow, "30d": ..., "90d": ...} — None/zero-row
                 windows are rendered as empty bars with a tooltip.

    Returns:
        Plotly Figure. Empty annotation when no resolved predictions exist.
    """
    labels = list(rollups.keys())
    values = []
    counts = []
    for label in labels:
        w = rollups[label]
        acc = (w.directional_accuracy or 0.0) * 100.0 if w else 0.0
        values.append(acc)
        counts.append(w.n_resolved if w else 0)

    colors = []
    for v in values:
        if v >= 58:
            colors.append(UIConfig.COLOR_BULLISH)
        elif v >= 50:
            colors.append(UIConfig.COLOR_NEUTRAL)
        else:
            colors.append(UIConfig.COLOR_BEARISH)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%<br>n={c}" for v, c in zip(values, counts)],
        textposition="outside",
        hovertemplate="window=%{x}<br>acc=%{y:.2f}%<br>n=%{text}<extra></extra>",
    ))
    fig.add_hline(
        y=50, line_dash="dash", line_color="#888",
        annotation_text="coin flip (50%)", annotation_position="right",
    )
    fig.add_hline(
        y=58, line_dash="dot", line_color=UIConfig.COLOR_BULLISH,
        annotation_text="target (58%)", annotation_position="right",
    )
    fig.update_layout(
        template="plotly_dark",
        title="Rolling Directional Accuracy",
        xaxis_title="Window",
        yaxis=dict(title="Directional accuracy (%)", range=[0, 100]),
        height=360,
    )
    if sum(counts) == 0:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="No resolved predictions yet — run the ledger backfill after target dates pass",
            showarrow=False, font=dict(color="#888"),
        )
    return fig


def create_forecast_band_chart(
    historical_df: pd.DataFrame,
    future_df: pd.DataFrame,
    stock_name: str,
    forecast_days: int,
) -> go.Figure:
    """
    Forecast chart with the P5–P95 (or conformal) band shaded (A6.4).

    Args:
        historical_df: Historical OHLCV (last ~60 days shown)
        future_df:     DataFrame with index=future dates and columns
                       'Predicted Price' + band columns ('P5'/'P95' or
                       'ci_low'/'ci_high' — whichever the model produced).
        stock_name:    Stock symbol (chart title)
        forecast_days: Horizon (chart title)

    Returns:
        Plotly Figure. Falls back to a plain forecast line when no band
        columns are present.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical_df.index[-60:],
        y=historical_df['Close'][-60:],
        name="Historical",
        line=dict(color=UIConfig.COLOR_PRIMARY, width=2),
    ))

    lo_col = "P5" if "P5" in future_df.columns else ("ci_low" if "ci_low" in future_df.columns else None)
    hi_col = "P95" if "P95" in future_df.columns else ("ci_high" if "ci_high" in future_df.columns else None)

    if lo_col and hi_col:
        # Upper bound (no fill — reference line for the band)
        fig.add_trace(go.Scatter(
            x=future_df.index,
            y=future_df[hi_col],
            name="Upper band",
            line=dict(color="rgba(0,255,136,0.35)", width=1),
            hoverinfo="skip",
            showlegend=False,
        ))
        # Lower bound — fills the region above it, up to the upper trace
        fig.add_trace(go.Scatter(
            x=future_df.index,
            y=future_df[lo_col],
            name="90% prediction band",
            fill="tonexty",
            line=dict(color="rgba(0,255,136,0.35)", width=1),
            fillcolor="rgba(0, 255, 136, 0.15)",
            hovertemplate="lo=%{y:.2f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=future_df.index,
        y=future_df['Predicted Price'],
        name="AI Forecast (median)",
        line=dict(dash='dot', color=UIConfig.COLOR_SECONDARY, width=2),
    ))

    fig.update_layout(
        template="plotly_dark",
        title=f"{stock_name} — {forecast_days} Day Forecast with 90% Band",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode='x unified',
    )
    return fig


def create_fii_dii_chart(fii_dii_data: pd.DataFrame) -> tuple:
    """
    Create FII/DII activity and cumulative charts.
    
    Args:
        fii_dii_data: DataFrame with FII/DII data
    
    Returns:
        Tuple of (activity_chart, cumulative_chart)
    """
    # Activity Chart
    fig_activity = go.Figure()
    
    fig_activity.add_trace(go.Bar(
        x=fii_dii_data.index,
        y=fii_dii_data['FII_Net'] / 1e7,
        name='FII Net',
        marker_color=UIConfig.COLOR_PRIMARY
    ))
    
    fig_activity.add_trace(go.Bar(
        x=fii_dii_data.index,
        y=fii_dii_data['DII_Net'] / 1e7,
        name='DII Net',
        marker_color=UIConfig.COLOR_SECONDARY
    ))
    
    fig_activity.update_layout(
        template="plotly_dark",
        title="Daily FII & DII Net Activity (₹ Crores)",
        xaxis_title="Date",
        yaxis_title="Net Activity (₹ Cr)",
        hovermode='x unified',
        barmode='group'
    )
    
    # Cumulative Chart
    fig_cumulative = go.Figure()
    
    fig_cumulative.add_trace(go.Scatter(
        x=fii_dii_data.index,
        y=fii_dii_data['FII_Cumulative'] / 1e7,
        name='FII Cumulative',
        line=dict(color=UIConfig.COLOR_PRIMARY, width=2),
        fill='tozeroy'
    ))
    
    fig_cumulative.add_trace(go.Scatter(
        x=fii_dii_data.index,
        y=fii_dii_data['DII_Cumulative'] / 1e7,
        name='DII Cumulative',
        line=dict(color=UIConfig.COLOR_SECONDARY, width=2),
        fill='tozeroy'
    ))
    
    fig_cumulative.update_layout(
        template="plotly_dark",
        title="Cumulative Institutional Positions (₹ Crores)",
        xaxis_title="Date",
        yaxis_title="Cumulative Position (₹ Cr)",
        hovermode='x unified'
    )
    
    return fig_activity, fig_cumulative
