"""
Stock data fetching utilities.
Uses yfinance to fetch stock data, info, and fundamental data.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

from config.settings import DataConfig


@st.cache_data
def get_indian_stocks() -> list:
    """
    Load list of Indian stock symbols from CSV file.
    
    Returns:
        List of stock symbols
    """
    file_path = DataConfig.INDIAN_STOCKS_FILE
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="utf-8")
        df.columns = df.columns.str.strip()
        if "SYMBOL" in df.columns:
            return df["SYMBOL"].dropna().tolist()
        else:
            st.error("Error: 'SYMBOL' column not found.")
            return []
    else:
        st.error(f"File '{file_path}' not found.")
        return DataConfig.DEFAULT_STOCKS


def get_stock_data(ticker: str, start, end) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., "RELIANCE.NS")
        start: Start date
        end: End date
    
    Returns:
        DataFrame with OHLCV data
    """
    stock = yf.Ticker(ticker)
    
    # Ensure start/end are passed as ISO date strings
    try:
        start_str = pd.to_datetime(start).strftime('%Y-%m-%d')
    except Exception:
        start_str = start
    try:
        end_str = pd.to_datetime(end).strftime('%Y-%m-%d')
    except Exception:
        end_str = end

    data = stock.history(start=start_str, end=end_str)
    
    if not data.empty:
        data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        data.index = data.index.tz_localize(None)
        data = data.sort_index()
    
    return data


def get_stock_info(ticker: str) -> dict:
    """
    Fetch basic stock information from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with stock info
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    def format_value(value, format_str):
        if value == "N/A" or value is None:
            return "N/A"
        return format_str.format(value)
    
    return {
        "Market Cap": format_value(info.get("marketCap"), "{:,} INR"),
        "P/E Ratio": format_value(info.get("trailingPE"), "{}"),
        "ROCE": format_value(info.get("returnOnCapitalEmployed"), "{:.2f}%"),
        "Current Price": format_value(info.get("currentPrice"), "{:.2f} INR"),
        "Book Value": format_value(info.get("bookValue"), "{:.2f} INR"),
        "ROE": format_value(info.get("returnOnEquity"), "{:.2f}%"),
        "Dividend Yield": format_value(info.get("dividendYield"), "{:.2f}%"),
        "Face Value": format_value(info.get("faceValue"), "{:.2f} INR"),
        "High": format_value(info.get("dayHigh"), "{:.2f} INR"),
        "Low": format_value(info.get("dayLow"), "{:.2f} INR"),
    }


def get_fundamental_data(ticker: str) -> dict:
    """
    Fetch fundamental data using yfinance.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with fundamental metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Safe extraction with default values
        fundamentals = {
            "Forward P/E": info.get("forwardPE", np.nan),
            "PEG Ratio": info.get("pegRatio", np.nan),
            "Price/Book": info.get("priceToBook", np.nan),
            "Debt/Equity": info.get("debtToEquity", np.nan),
            "ROE": info.get("returnOnEquity", np.nan),
            "Profit Margins": info.get("profitMargins", np.nan),
            "Revenue Growth": info.get("revenueGrowth", np.nan),
            "Free Cashflow": info.get("freeCashflow", np.nan),
            "Target Price (Analyst)": info.get("targetMeanPrice", np.nan)
        }
        return fundamentals
    except Exception as e:
        st.warning(f"Could not fetch fundamentals: {e}")
        return {}
