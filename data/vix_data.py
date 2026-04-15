"""
India VIX data fetching and Indian market calendar.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay

from config.settings import DataConfig, INDIAN_HOLIDAYS


class IndiaHolidayCalendar(AbstractHolidayCalendar):
    """
    Custom Indian stock market holiday calendar.
    Includes major Indian holidays when markets are closed.
    """
    rules = [
        Holiday(h["name"], month=h["month"], day=h["day"])
        for h in INDIAN_HOLIDAYS
    ]


def get_india_business_day():
    """
    Get a CustomBusinessDay object for Indian markets.
    
    Returns:
        CustomBusinessDay with Indian holidays
    """
    return CustomBusinessDay(calendar=IndiaHolidayCalendar())


def get_india_vix_data(start_date, end_date) -> pd.DataFrame:
    """
    Fetch India VIX data from Yahoo Finance.
    
    If India VIX data is unavailable, generates synthetic VIX based on 
    NIFTY volatility as a proxy.
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with VIX data (Open, High, Low, Close, Volume)
    """
    try:
        vix_ticker = DataConfig.INDIA_VIX_TICKER
        vix_data = yf.download(vix_ticker, start=start_date, end=end_date, progress=False)
        
        if vix_data.empty:
            # Fallback: Create synthetic VIX data based on NIFTY volatility
            nifty_data = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
            if not nifty_data.empty:
                # Calculate rolling volatility as proxy for VIX
                returns = nifty_data['Close'].pct_change()
                annualized_vol = returns.rolling(20).std() * 100 * 16  # Annualized volatility
                
                vix_data = pd.DataFrame({
                    'Open': annualized_vol,
                    'High': annualized_vol * 1.1,
                    'Low': annualized_vol * 0.9,
                    'Close': annualized_vol,
                    'Volume': nifty_data['Volume']
                })
                vix_data = vix_data.dropna()
        
        return vix_data
    except Exception as e:
        st.warning(f"Could not fetch India VIX data: {str(e)}")
        return pd.DataFrame()


def fetch_nifty_benchmark(start_date, end_date) -> pd.DataFrame:
    """
    Fetch NIFTY 50 daily returns for use as buy-and-hold benchmark in backtesting.

    Args:
        start_date: Start date (str or datetime)
        end_date: End date (str or datetime)

    Returns:
        DataFrame with columns ['Close', 'Return'] aligned to trading days.
        Returns empty DataFrame if download fails.
    """
    try:
        nifty_data = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
        if nifty_data.empty:
            return pd.DataFrame()

        # Flatten MultiIndex columns if present (yfinance ≥ 0.2)
        if isinstance(nifty_data.columns, pd.MultiIndex):
            nifty_data.columns = [c[0] for c in nifty_data.columns]

        df = pd.DataFrame()
        df['Close'] = nifty_data['Close']
        df['Return'] = df['Close'].pct_change().fillna(0)
        df = df.dropna(subset=['Close'])
        return df
    except Exception:
        return pd.DataFrame()


def extract_volatility_features(vix_data: pd.DataFrame, stock_data: pd.DataFrame) -> dict:
    """
    Extract volatility-related features for model input.
    
    Args:
        vix_data: DataFrame with VIX data
        stock_data: DataFrame with stock data (should include Volatility_20D)
    
    Returns:
        Dictionary of volatility features
    """
    features = {
        'vix_current': 0.0,
        'vix_vs_ma20': 1.0,
        'stock_volatility': 0.0,
        'vix_trend': 0
    }
    
    if vix_data is None or vix_data.empty:
        return features
    
    try:
        # Latest VIX close
        if 'Close' in vix_data.columns and len(vix_data) > 0:
            last_vix = vix_data['Close'].iloc[-1]
            if isinstance(last_vix, (pd.Series, np.ndarray)):
                arr = np.asarray(last_vix).ravel()
                if arr.size > 0 and not np.isnan(arr[-1]):
                    features['vix_current'] = float(arr[-1])
            else:
                if pd.notna(last_vix):
                    features['vix_current'] = float(last_vix)
        
        # VIX vs 20-day MA
        if len(vix_data) >= 20:
            vix_ma20 = vix_data['Close'].rolling(20).mean().iloc[-1]
            if isinstance(vix_ma20, (pd.Series, np.ndarray)):
                arr = np.asarray(vix_ma20).ravel()
                if arr.size > 0 and not np.isnan(arr[-1]):
                    vix_ma20_val = float(arr[-1])
                else:
                    vix_ma20_val = None
            else:
                vix_ma20_val = float(vix_ma20) if pd.notna(vix_ma20) else None
            
            if vix_ma20_val and vix_ma20_val != 0:
                features['vix_vs_ma20'] = features['vix_current'] / vix_ma20_val
        
        # VIX trend (comparing last 5 days avg to previous 5 days)
        if len(vix_data) >= 10:
            recent_avg = vix_data['Close'].iloc[-5:].mean()
            previous_avg = vix_data['Close'].iloc[-10:-5].mean()
            features['vix_trend'] = 1 if recent_avg > previous_avg else -1
        
    except Exception:
        pass
    
    # Stock volatility
    if stock_data is not None and not stock_data.empty:
        if 'Volatility_20D' in stock_data.columns:
            last_vol = stock_data['Volatility_20D'].iloc[-1]
            if pd.notna(last_vol):
                features['stock_volatility'] = float(last_vol)
    
    return features
