"""
Technical indicators calculation functions.
Provides RSI, MACD, ATR, OBV, and comprehensive indicator generation.
"""

import pandas as pd
import numpy as np


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: Series of closing prices
        period: RSI period (default 14)
    
    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Series of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    
    Returns:
        Tuple of (MACD line, Signal line)
    """
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        df: DataFrame with High, Low, Close columns
        period: ATR period (default 14)
    
    Returns:
        Series of ATR values
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period, min_periods=1).mean()


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume.
    
    Args:
        df: DataFrame with Close and Volume columns
    
    Returns:
        Series of OBV values
    """
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive technical indicators for a stock.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with all technical indicators added
    """
    df = df.copy()
    
    # Basic indicators
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(5, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(200, min_periods=1).mean()
    
    # Moving Average Ratios
    df['MA_Ratio_5_20'] = df['MA5'] / df['MA20']
    df['MA_Ratio_5_50'] = df['MA5'] / df['MA50']
    df['MA_Ratio_20_200'] = df['MA20'] / df['MA200']
    
    # Volatility Measures
    df['Volatility_5D'] = df['Returns'].rolling(5, min_periods=1).std()
    df['Volatility_20D'] = df['Returns'].rolling(20, min_periods=1).std()
    df['ATR'] = calculate_atr(df)
    
    # Volume Indicators
    df['Volume_MA5'] = df['Volume'].rolling(5, min_periods=1).mean()
    df['Volume_MA20'] = df['Volume'].rolling(20, min_periods=1).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    df['OBV'] = calculate_obv(df)
    
    # Momentum Indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Support/Resistance Levels
    df['Pivot_Point'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['Pivot_Point'] - df['Low']
    df['S1'] = 2 * df['Pivot_Point'] - df['High']
    
    # Price Position
    df['Price_vs_MA20'] = df['Close'] / df['MA20'] - 1
    df['Price_vs_MA50'] = df['Close'] / df['MA50'] - 1
    
    # Gap Analysis
    df['Gap'] = df['Open'] / df['Close'].shift(1) - 1
    
    return df.dropna()
