"""
Configuration settings for the ProTrader AI application.

API keys are resolved with this priority chain:
  1. Streamlit Cloud secrets (st.secrets)  - production
  2. .env file via python-dotenv           - local development
  3. OS environment variables              - CI / containers
  4. Hard-coded defaults                   - last resort
"""

import os
from dotenv import load_dotenv

# Load .env for local development (no-op on Streamlit Cloud)
load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """
    Resolve a config value from Streamlit secrets first, then env vars.
    Safe to call before Streamlit is initialized — it never raises.
    """
    # Try Streamlit secrets (only available when running inside Streamlit)
    try:
        import streamlit as st
        # st.secrets behaves like a dict; access can raise StreamlitSecretNotFoundError
        if hasattr(st, "secrets") and key in st.secrets:
            value = st.secrets[key]
            if value:
                return str(value)
    except Exception:
        pass

    # Fallback: environment variable (loaded from .env or OS)
    return os.environ.get(key, default)


# ==============================================
# API Keys
# ==============================================
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY", "")
NEWS_API_KEY = _get_secret("NEWS_API_KEY", "")
DEEPSEEK_API_KEY = _get_secret("DEEPSEEK_API_KEY", "")

# Reddit API (for multi-source sentiment)
REDDIT_CLIENT_ID = _get_secret("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = _get_secret("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = _get_secret("REDDIT_USER_AGENT", "ProTraderAI/1.0")

# Roboflow API (for pattern detection)
ROBOFLOW_API_KEY = _get_secret("ROBOFLOW_API_KEY", "9KHbf18pSI2tt8dqZeOL")
ROBOFLOW_WORKSPACE = _get_secret("ROBOFLOW_WORKSPACE", "financeas")
ROBOFLOW_WORKFLOW_ID = _get_secret("ROBOFLOW_WORKFLOW_ID", "custom-workflow")


# ==============================================
# Model Configuration
# ==============================================
class ModelConfig:
    """Configuration constants for ML models"""

    # Confidence thresholds
    LOW_CONFIDENCE_THRESHOLD = 52
    MEDIUM_CONFIDENCE_THRESHOLD = 60
    HIGH_CONFIDENCE_THRESHOLD = 72

    # Model parameters
    LOOKBACK_PERIOD = 30
    TECHNICAL_FEATURES = 25
    MAX_ERROR_WINDOW = 10

    # XGBoost defaults
    XGB_N_ESTIMATORS = 100
    XGB_MAX_DEPTH = 3
    XGB_LEARNING_RATE = 0.05

    # GRU defaults
    GRU_UNITS_1 = 128
    GRU_UNITS_2 = 64
    GRU_UNITS_3 = 32
    GRU_DROPOUT = 0.3
    GRU_LEARNING_RATE = 0.001

    # Training
    DEFAULT_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 32
    TRAIN_TEST_SPLIT = 0.8


# ==============================================
# Trading Configuration
# ==============================================
class TradingConfig:
    """Configuration for trading and risk management"""

    # Risk thresholds
    STRONG_BUY_THRESHOLD = 7
    BUY_THRESHOLD = 3
    SELL_THRESHOLD = -3
    STRONG_SELL_THRESHOLD = -7

    # ATR multipliers for stop loss
    SL_MULTIPLIER_HIGH_CONFIDENCE = 2.0
    SL_MULTIPLIER_LOW_CONFIDENCE = 1.5

    # Target risk/reward ratio
    MIN_RISK_REWARD = 1.5

    # Backtesting
    DEFAULT_INITIAL_CAPITAL = 100000
    SIGNAL_THRESHOLD = 0.001


# ==============================================
# Data Configuration
# ==============================================
class DataConfig:
    """Configuration for data fetching"""

    # Stock data file
    INDIAN_STOCKS_FILE = "indian_stocks.csv"

    # Default stocks if file not found
    DEFAULT_STOCKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

    # Stock name mapping for news API
    STOCK_NAME_MAPPING = {
        "RELIANCE": "Reliance Industries",
        "TCS": "Tata Consultancy Services",
        "INFY": "Infosys",
        "HDFCBANK": "HDFC Bank",
        "ICICIBANK": "ICICI Bank"
    }

    # API endpoints
    NEWS_API_URL = "https://newsapi.org/v2/everything"
    NSE_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
    NSE_ACTIVITY_URL = "https://www.nseindia.com/api/fiidiiActivity"

    # India VIX ticker
    INDIA_VIX_TICKER = "^INDIAVIX"

    # Cache TTL (seconds)
    FII_DII_CACHE_TTL = 3600  # 1 hour


# ==============================================
# UI Configuration
# ==============================================
class UIConfig:
    """Configuration for Streamlit UI"""

    PAGE_TITLE = "Pro Stock AI"
    PAGE_LAYOUT = "wide"

    # Chart colors
    COLOR_BULLISH = "#00ff88"
    COLOR_BEARISH = "#ff4444"
    COLOR_NEUTRAL = "#ffaa00"
    COLOR_PRIMARY = "#00d4ff"
    COLOR_SECONDARY = "#ff6b6b"

    # Gradient backgrounds
    GRADIENT_BULLISH = "linear-gradient(135deg, #1a472a 0%, #2d5a3d 50%, #3d6b4f 100%)"
    GRADIENT_BEARISH = "linear-gradient(135deg, #4a1a1a 0%, #5a2d2d 50%, #6b3d3d 100%)"
    GRADIENT_NEUTRAL = "linear-gradient(135deg, #3a3a1a 0%, #4a4a2d 50%, #5a5a3d 100%)"
    GRADIENT_DARK = "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"


# ==============================================
# Indian Holiday Calendar (for market days)
# ==============================================
INDIAN_HOLIDAYS = [
    {"name": "Republic Day", "month": 1, "day": 26},
    {"name": "Independence Day", "month": 8, "day": 15},
    {"name": "Gandhi Jayanti", "month": 10, "day": 2},
    {"name": "Diwali", "month": 10, "day": 24},  # Approximate - varies yearly
    {"name": "Holi", "month": 3, "day": 25},     # Approximate - varies yearly
]
