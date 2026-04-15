"""
FII/DII (Foreign/Domestic Institutional Investor) data fetching.
Fetches official data from NSE India website with robust retry logic.
"""

import pandas as pd
import numpy as np
import requests
import time
import streamlit as st

from config.settings import DataConfig

# Optional nselib import
try:
    from nselib.capital_market import fiidii as nselib_fiidii
    NSELIB_AVAILABLE = True
except ImportError:
    NSELIB_AVAILABLE = False
    nselib_fiidii = None

# Optional Gemini import for parsing fallback
try:
    import google.generativeai as genai
    from config.settings import GEMINI_API_KEY
    GEMINI_AVAILABLE = bool(GEMINI_API_KEY)
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    GEMINI_API_KEY = None


def _create_nse_session():
    """
    Create a robust session with NSE cookies.
    NSE requires proper browser-like headers and cookies from main page.
    
    Returns:
        Tuple of (session, headers)
    """
    # More realistic browser headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Origin': 'https://www.nseindia.com',
        'Referer': 'https://www.nseindia.com/reports/fii-dii',
        'Sec-Ch-Ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
    }
    
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        # Visit the main page first to get cookies - this is CRITICAL
        main_response = session.get(
            'https://www.nseindia.com', 
            headers=headers, 
            timeout=15
        )
        
        # Small delay to seem more human-like
        time.sleep(0.5)
        
        # Also visit the FII/DII reports page to get additional cookies
        session.get(
            'https://www.nseindia.com/reports/fii-dii',
            headers=headers,
            timeout=10
        )
        
        time.sleep(0.3)
        
    except Exception as e:
        pass  # Continue even if cookie fetch fails, API might still work
    
    return session, headers


def _try_fetch_nse_fii_dii(session, headers, max_retries=3):
    """
    Try to fetch FII/DII data with retry logic.
    
    Args:
        session: Requests session with cookies
        headers: Request headers
        max_retries: Number of retry attempts
        
    Returns:
        JSON data or None if failed
    """
    # Multiple endpoints to try
    endpoints = [
        "https://www.nseindia.com/api/fiidiiTradeReact",
        "https://www.nseindia.com/api/fii-dii",
    ]
    
    for endpoint in endpoints:
        for attempt in range(max_retries):
            try:
                response = session.get(
                    endpoint, 
                    headers=headers, 
                    timeout=20
                )
                
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '')
                    
                    # Check if response is actually JSON
                    if 'application/json' in content_type or response.text.strip().startswith(('{', '[')):
                        try:
                            json_data = response.json()
                            if json_data:  # Not empty
                                return json_data
                        except Exception:
                            pass
                
                elif response.status_code == 401:
                    # Session expired, refresh
                    session, headers = _create_nse_session()
                    
                # Wait before retry
                time.sleep(1 + attempt)
                
            except requests.exceptions.Timeout:
                time.sleep(2)
                continue
            except Exception:
                time.sleep(1)
                continue
    
    return None


def _fetch_from_nselib():
    """
    Fetch FII/DII data using the nselib Python library.
    Second fallback after NSE API direct fetch.
    
    Install: pip install nselib
    
    Returns:
        DataFrame or None
    """
    if not NSELIB_AVAILABLE:
        return None
    
    try:
        # Use nselib to get FII/DII data
        df = nselib_fiidii()
        
        if df is not None and not df.empty:
            # Standardize column names to match our format
            df_standardized = pd.DataFrame()
            
            # Try to map columns (nselib may have different column names)
            date_cols = [c for c in df.columns if 'date' in c.lower()]
            if date_cols:
                df_standardized['Date'] = pd.to_datetime(df[date_cols[0]])
            else:
                df_standardized['Date'] = df.index
            
            # Map FII columns
            for col in df.columns:
                col_lower = col.lower()
                if 'fii' in col_lower and 'buy' in col_lower:
                    df_standardized['FII_Buy_Value'] = pd.to_numeric(df[col], errors='coerce') * 1e7
                elif 'fii' in col_lower and 'sell' in col_lower:
                    df_standardized['FII_Sell_Value'] = pd.to_numeric(df[col], errors='coerce') * 1e7
                elif 'fii' in col_lower and 'net' in col_lower:
                    df_standardized['FII_Net'] = pd.to_numeric(df[col], errors='coerce') * 1e7
                elif 'dii' in col_lower and 'buy' in col_lower:
                    df_standardized['DII_Buy_Value'] = pd.to_numeric(df[col], errors='coerce') * 1e7
                elif 'dii' in col_lower and 'sell' in col_lower:
                    df_standardized['DII_Sell_Value'] = pd.to_numeric(df[col], errors='coerce') * 1e7
                elif 'dii' in col_lower and 'net' in col_lower:
                    df_standardized['DII_Net'] = pd.to_numeric(df[col], errors='coerce') * 1e7
            
            # Calculate net if not present
            if 'FII_Net' not in df_standardized.columns and 'FII_Buy_Value' in df_standardized.columns:
                df_standardized['FII_Net'] = df_standardized.get('FII_Buy_Value', 0) - df_standardized.get('FII_Sell_Value', 0)
            if 'DII_Net' not in df_standardized.columns and 'DII_Buy_Value' in df_standardized.columns:
                df_standardized['DII_Net'] = df_standardized.get('DII_Buy_Value', 0) - df_standardized.get('DII_Sell_Value', 0)
            
            # Set index and calculate cumulative
            if 'Date' in df_standardized.columns:
                df_standardized = df_standardized.set_index('Date').sort_index()
            
            if 'FII_Net' in df_standardized.columns:
                df_standardized['FII_Cumulative'] = df_standardized['FII_Net'].cumsum()
            if 'DII_Net' in df_standardized.columns:
                df_standardized['DII_Cumulative'] = df_standardized['DII_Net'].cumsum()
            
            return df_standardized
            
    except Exception as e:
        pass
    
    return None


def parse_manual_fii_dii_input(raw_text: str) -> pd.DataFrame:
    """
    Parse manually pasted FII/DII data.
    First tries direct JSON parsing, then Gemini fallback.
    """
    if not raw_text:
        return None
        
    try:
        import json
        
        # 1. Try strict JSON Parsing first (Fastest/Cheapest)
        try:
            # Clean up potentially messy input (remove markdown code blocks if any)
            clean_text = raw_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            
            data = json.loads(clean_text)
            
            if isinstance(data, dict) and 'data' in data:
                records = data['data']
            elif isinstance(data, list):
                records = data
            else:
                records = []
            
            if records:
                # Use the existing processing logic
                date_data = {}
                for record in records:
                    try:
                        date_str = record.get('date', '')
                        category = record.get('category', '').upper()
                        
                        # Handle potential string format from NSE
                        def safe_float(val):
                            if isinstance(val, str):
                                return float(val.replace(',', ''))
                            return float(val or 0)
                            
                        buy_val = safe_float(record.get('buyValue', record.get('Buy Value', 0)))
                        sell_val = safe_float(record.get('sellValue', record.get('Sell Value', 0)))
                        net_val = safe_float(record.get('netValue', record.get('Net Value', 0)))
                        
                        parsed_date = pd.to_datetime(date_str, errors='coerce')
                        if pd.isna(parsed_date):
                            continue
                            
                        if parsed_date not in date_data:
                            date_data[parsed_date] = {
                                'FII_Buy_Value': 0, 'DII_Buy_Value': 0,
                                'FII_Sell_Value': 0, 'DII_Sell_Value': 0,
                                'FII_Net': 0, 'DII_Net': 0
                            }
                        
                        if 'FII' in category or 'FPI' in category:
                            date_data[parsed_date]['FII_Buy_Value'] = buy_val * 1e7
                            date_data[parsed_date]['FII_Sell_Value'] = sell_val * 1e7
                            date_data[parsed_date]['FII_Net'] = net_val * 1e7
                        elif 'DII' in category:
                            date_data[parsed_date]['DII_Buy_Value'] = buy_val * 1e7
                            date_data[parsed_date]['DII_Sell_Value'] = sell_val * 1e7
                            date_data[parsed_date]['DII_Net'] = net_val * 1e7
                            
                    except Exception:
                        continue
                
                if date_data:
                    fii_dii_records = [{'Date': k, **v} for k, v in date_data.items()]
                    df = pd.DataFrame(fii_dii_records)
                    df = df.set_index('Date').sort_index()
                    df['FII_Cumulative'] = df['FII_Net'].cumsum()
                    df['DII_Cumulative'] = df['DII_Net'].cumsum()
                    return df
        except json.JSONDecodeError:
            pass  # Fall through to Gemini

        # 2. Gemini Fallback (Only if enabled and available)
        if GEMINI_AVAILABLE:
            try:
                return _parse_fii_dii_with_gemini(raw_text)
            except Exception as e:
                if "429" in str(e):
                    st.error("⚠️ AI Quota Exceeded (429). Please paste strictly valid JSON.")
                else:
                    st.warning(f"AI parsing failed: {e}")
                return None
                
    except Exception as e:
        print(f"Manual parsing error: {e}")
        
    return None


# NSE API URL for manual fallback
NSE_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"


def _parse_fii_dii_with_gemini(raw_json_text: str) -> pd.DataFrame:
    """
    Use Gemini AI to parse FII/DII data when manual parsing fails.
    
    Args:
        raw_json_text: Raw JSON response from NSE API
        
    Returns:
        DataFrame or None if parsing fails
    """
    if not GEMINI_AVAILABLE or not raw_json_text:
        return None
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""Extract FII/DII data from this NSE API response and return as a structured JSON array.

INPUT DATA:
{raw_json_text}

REQUIRED OUTPUT FORMAT (JSON array only, no markdown):
[
  {{"date": "YYYY-MM-DD", "fii_buy": 12345.67, "fii_sell": 12345.67, "fii_net": 12345.67, "dii_buy": 12345.67, "dii_sell": 12345.67, "dii_net": 12345.67}},
  ...
]

Rules:
1. Convert all dates to YYYY-MM-DD format
2. All values should be numbers (no quotes), in Crores
3. Return ONLY the JSON array, no explanation
"""

        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Clean up response (remove markdown code blocks if present)
        if result_text.startswith('```'):
            result_text = result_text.split('\n', 1)[1]
        if result_text.endswith('```'):
            result_text = result_text.rsplit('\n', 1)[0]
        result_text = result_text.strip()
        
        # Parse the JSON
        import json
        parsed_data = json.loads(result_text)
        
        if parsed_data and isinstance(parsed_data, list):
            records = []
            for item in parsed_data:
                try:
                    records.append({
                        'Date': pd.to_datetime(item.get('date')),
                        'FII_Buy_Value': float(item.get('fii_buy', 0)) * 1e7,
                        'FII_Sell_Value': float(item.get('fii_sell', 0)) * 1e7,
                        'FII_Net': float(item.get('fii_net', 0)) * 1e7,
                        'DII_Buy_Value': float(item.get('dii_buy', 0)) * 1e7,
                        'DII_Sell_Value': float(item.get('dii_sell', 0)) * 1e7,
                        'DII_Net': float(item.get('dii_net', 0)) * 1e7,
                    })
                except Exception:
                    continue
            
            if records:
                df = pd.DataFrame(records)
                df = df.set_index('Date').sort_index()
                df['FII_Cumulative'] = df['FII_Net'].cumsum()
                df['DII_Cumulative'] = df['DII_Net'].cumsum()
                return df
                
    except Exception as e:
        st.warning(f"Gemini parsing failed: {str(e)[:50]}")
    
    return None


def _fetch_from_moneycontrol():
    """
    Alternative: Fetch FII/DII data from MoneyControl API.
    This is a backup when NSE API is unavailable.
    
    Returns:
        DataFrame or None
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        
        # MoneyControl FII/DII data endpoint
        url = "https://api.moneycontrol.com/mcapi/v1/fii-dii/activity"
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data and 'data' in data:
                records = []
                for item in data['data']:
                    try:
                        records.append({
                            'Date': pd.to_datetime(item.get('date')),
                            'FII_Buy_Value': float(item.get('fii_buy', 0)) * 1e7,
                            'FII_Sell_Value': float(item.get('fii_sell', 0)) * 1e7,
                            'FII_Net': float(item.get('fii_net', 0)) * 1e7,
                            'DII_Buy_Value': float(item.get('dii_buy', 0)) * 1e7,
                            'DII_Sell_Value': float(item.get('dii_sell', 0)) * 1e7,
                            'DII_Net': float(item.get('dii_net', 0)) * 1e7,
                        })
                    except Exception:
                        continue
                
                if records:
                    df = pd.DataFrame(records)
                    df = df.set_index('Date').sort_index()
                    df['FII_Cumulative'] = df['FII_Net'].cumsum()
                    df['DII_Cumulative'] = df['DII_Net'].cumsum()
                    return df
                    
    except Exception:
        pass
    
    return None


def _fetch_from_trendlyne():
    """
    Alternative: Fetch FII/DII data from Trendlyne.
    Third fallback source when NSE and MoneyControl are unavailable.
    
    Returns:
        DataFrame or None
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/json',
            'Referer': 'https://trendlyne.com/equity/market-activity/',
        }
        
        # Trendlyne FII/DII page
        url = "https://trendlyne.com/equity/market-activity/fii-dii-activity/"
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            # Try to parse JSON if available
            try:
                data = response.json()
                if data and isinstance(data, list):
                    records = []
                    for item in data:
                        try:
                            records.append({
                                'Date': pd.to_datetime(item.get('date')),
                                'FII_Buy_Value': float(item.get('fii_buy', 0)) * 1e7,
                                'FII_Sell_Value': float(item.get('fii_sell', 0)) * 1e7,
                                'FII_Net': float(item.get('fii_net', 0)) * 1e7,
                                'DII_Buy_Value': float(item.get('dii_buy', 0)) * 1e7,
                                'DII_Sell_Value': float(item.get('dii_sell', 0)) * 1e7,
                                'DII_Net': float(item.get('dii_net', 0)) * 1e7,
                            })
                        except Exception:
                            continue
                    
                    if records:
                        df = pd.DataFrame(records)
                        df = df.set_index('Date').sort_index()
                        df['FII_Cumulative'] = df['FII_Net'].cumsum()
                        df['DII_Cumulative'] = df['DII_Net'].cumsum()
                        return df
            except Exception:
                pass
                    
    except Exception:
        pass
    
    return None


@st.cache_data(ttl=DataConfig.FII_DII_CACHE_TTL)
def fetch_fii_dii_apis_internal(ticker=None, start_date=None, end_date=None) -> pd.DataFrame:
    """Internal cached function to fetch from APIs only."""
    # 1. Try NSE API first (primary source)
    session, headers = _create_nse_session()
    json_data = _try_fetch_nse_fii_dii(session, headers)
    
    if json_data is None:
        # 2. Try nselib library as second fallback
        df = _fetch_from_nselib()
        if df is not None and not df.empty:
            return df
        
        # 3. Try MoneyControl as third fallback
        df = _fetch_from_moneycontrol()
        if df is not None and not df.empty:
            return df
        
        # 4. Try Trendlyne as fourth fallback
        df = _fetch_from_trendlyne()
        if df is not None and not df.empty:
            return df
            
        return None
    
    try:
        # Parse NSE response
        if isinstance(json_data, dict):
            records = json_data.get('data', [])
        elif isinstance(json_data, list):
            records = json_data
        else:
            return None
        
        if not records:
            return None
        
        # Parse records - NSE API returns separate records for FII and DII
        date_data = {}
        
        for record in records:
            try:
                date_str = record.get('date', '')
                category = record.get('category', '').upper()
                
                # Parse values
                buy_val = float(record.get('buyValue', record.get('Buy Value', 0)) or 0)
                sell_val = float(record.get('sellValue', record.get('Sell Value', 0)) or 0)
                net_val = float(record.get('netValue', record.get('Net Value', 0)) or 0)
                
                # Try multiple date formats
                parsed_date = None
                for date_format in ['%d-%b-%Y', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']:
                    try:
                        parsed_date = pd.to_datetime(date_str, format=date_format)
                        break
                    except Exception:
                        continue
                
                if parsed_date is None:
                    parsed_date = pd.to_datetime(date_str, errors='coerce')
                
                if pd.isna(parsed_date):
                    continue
                
                if parsed_date not in date_data:
                    date_data[parsed_date] = {
                        'FII_Buy_Value': 0, 'FII_Sell_Value': 0, 'FII_Net': 0,
                        'DII_Buy_Value': 0, 'DII_Sell_Value': 0, 'DII_Net': 0
                    }
                
                if 'FII' in category or 'FPI' in category:
                    date_data[parsed_date]['FII_Buy_Value'] = buy_val * 1e7
                    date_data[parsed_date]['FII_Sell_Value'] = sell_val * 1e7
                    date_data[parsed_date]['FII_Net'] = net_val * 1e7
                elif 'DII' in category:
                    date_data[parsed_date]['DII_Buy_Value'] = buy_val * 1e7
                    date_data[parsed_date]['DII_Sell_Value'] = sell_val * 1e7
                    date_data[parsed_date]['DII_Net'] = net_val * 1e7
                    
            except (ValueError, TypeError):
                continue
        
        if date_data:
            fii_dii_records = [{'Date': k, **v} for k, v in date_data.items()]
            df = pd.DataFrame(fii_dii_records)
            df = df.set_index('Date').sort_index()
            df = df[~df.index.duplicated(keep='last')]
            df['FII_Cumulative'] = df['FII_Net'].cumsum()
            df['DII_Cumulative'] = df['DII_Net'].cumsum()
            return df
            
        # Try Gemini fallback for raw JSON
        if GEMINI_AVAILABLE and isinstance(json_data, (list, dict)):
            import json
            raw_json_text = json.dumps(json_data)
            df = _parse_fii_dii_with_gemini(raw_json_text)
            if df is not None and not df.empty:
                return df

    except Exception:
        pass
        
    return None


def get_fii_dii_data(ticker=None, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Fetch FII/DII data with manual fallback support.
    Not cached directly to allow manual input updates.
    """
    # 1. Try automated APIs (Cached)
    df = fetch_fii_dii_apis_internal(ticker, start_date, end_date)
    
    if df is not None and not df.empty:
        # Filter by date if needed (caching might return wider range)
        if start_date and end_date:
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            return df.loc[mask]
        return df

    # 2. Check for manual input in session state
    if 'manual_fii_dii_input' in st.session_state and st.session_state.manual_fii_dii_input:
        df_manual = parse_manual_fii_dii_input(st.session_state.manual_fii_dii_input)
        if df_manual is not None and not df_manual.empty:
            return df_manual

    return pd.DataFrame()


def render_manual_fii_dii_input():
    """Render the manual input widget for FII/DII data (Main App Version)."""
    st.markdown("""
    <div style="background-color: #2d1b1b; padding: 15px; border-radius: 10px; border: 1px solid #ff4b4b; margin: 10px 0;">
        <h4 style="color: #ff4b4b; margin: 0;">⚠️ Action Required: FII/DII Data Missing</h4>
        <p style="color: #ccc; font-size: 14px; margin-top: 5px;">
            Automated fetching failed. The analysis requires this data to proceed.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(f"👉 **Step 1:** Open [NSE FII/DII Reports]({NSE_FII_DII_URL}) in a new tab.")
    
    manual_input = st.text_area(
        "👉 **Step 2:** Copy the raw JSON content and paste it here:",
        height=200,
        key="fii_dii_manual_area",
        placeholder='Example: [{"category":"DII","date":"16-Jan-2026","buyValue":"19135.42",...}]',
        help="Paste the full JSON array from the NSE website."
    )
    
    if manual_input:
        st.session_state.manual_fii_dii_input = manual_input
        st.success("✅ Data received! Resuming analysis...")
        time.sleep(1) # Brief pause for user to see success
        st.rerun()


def extract_fii_dii_features(fii_dii_data: pd.DataFrame, lookback: int = 5) -> dict:
    """
    Extract features from FII/DII data for model integration.
    
    Args:
        fii_dii_data: DataFrame with FII/DII data
        lookback: Number of days to look back
    
    Returns:
        Dictionary of extracted features
    """
    if fii_dii_data is None or fii_dii_data.empty:
        return {
            'fii_net_5d': 0,
            'dii_net_5d': 0,
            'fii_trend': 0,
            'dii_trend': 0,
            'institutional_divergence': 0
        }
    
    recent_data = fii_dii_data.tail(lookback)
    
    if recent_data.empty:
        return {
            'fii_net_5d': 0,
            'dii_net_5d': 0,
            'fii_trend': 0,
            'dii_trend': 0,
            'institutional_divergence': 0
        }
    
    fii_net_sum = recent_data['FII_Net'].sum()
    dii_net_sum = recent_data['DII_Net'].sum()
    
    features = {
        'fii_net_5d': fii_net_sum,
        'dii_net_5d': dii_net_sum,
        'fii_trend': 1 if recent_data['FII_Net'].mean() > 0 else -1,
        'dii_trend': 1 if recent_data['DII_Net'].mean() > 0 else -1,
        'institutional_divergence': abs(fii_net_sum + dii_net_sum)
    }
    
    return features
