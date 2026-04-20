"""
Advanced Pattern Detection System with Multi-Timeframe Scanning.
Uses ZigZag indicator + scipy peak detection with adaptive thresholds.
Includes volume-weighted S/R, multi-timeframe confluence, and Hurst-based
pattern reliability scoring. Fallback mechanisms ensure output always exists.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks
from scipy.stats import linregress
from typing import List, Dict, Optional, Tuple
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config.settings import ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW_ID
from utils.roboflow_client import RoboflowClient


def _hurst_exponent_pa(prices: np.ndarray) -> float:
    """Hurst exponent via R/S analysis. Standalone for pattern analyst use."""
    if len(prices) < 40:
        return 0.5
    log_ret = np.diff(np.log(np.maximum(prices, 1e-10)))
    rs_pairs = []
    for lag in range(2, 21):
        n_w = len(log_ret) // lag
        if n_w < 2:
            continue
        rs_vals = []
        for w in range(n_w):
            seg = log_ret[w * lag:(w + 1) * lag]
            s = np.std(seg, ddof=1)
            if s > 1e-10:
                mc = seg - np.mean(seg)
                rs_vals.append((np.max(np.cumsum(mc)) - np.min(np.cumsum(mc))) / s)
        if rs_vals:
            rs_pairs.append((lag, np.mean(rs_vals)))
    if len(rs_pairs) < 4:
        return 0.5
    lags_a = np.array([p[0] for p in rs_pairs], dtype=float)
    rs_a   = np.array([p[1] for p in rs_pairs], dtype=float)
    valid  = rs_a > 0
    if valid.sum() < 4:
        return 0.5
    slope = np.polyfit(np.log(lags_a[valid]), np.log(rs_a[valid]), 1)[0]
    return float(np.clip(slope, 0.1, 0.9))


class PatternAnalyst:
    """
    Advanced pattern detection with multi-timeframe scanning and adaptive thresholds.
    
    Key Features:
    - Multi-timeframe scanning (order 3, 5, 7) to catch patterns at all scales
    - Adaptive thresholds that relax if no patterns found
    - Channel, flag, and rounding bottom detection
    - Micro-pattern detection for consolidating markets
    - Fallback mechanism: always tries to find something useful
    """
    
    def __init__(self, order: int = 3):
        self.order = order
        self.min_pattern_height = 0.01  # 1% minimum pattern height
        self.max_patterns_per_type = 3
        
        # Multi-timeframe orders: small catches minor swings, large catches major ones
        self.scan_orders = [3, 5, 7]
        
        # Initialize Roboflow Client
        self.vision_client = None
        if ROBOFLOW_API_KEY:
            self.vision_client = RoboflowClient(api_key=ROBOFLOW_API_KEY)
            
    def _generate_chart_image(self, df: pd.DataFrame, window: int = 60) -> Optional[bytes]:
        """Generate a chart image for vision analysis."""
        try:
            df_slice = df.tail(window).copy()
            if df_slice.empty:
                return None
                
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            ax.plot(df_slice.index, df_slice['Close'], color='black', linewidth=2)
            ax.grid(True, alpha=0.3)
            plt.title(f"Price Action ({window}D)", fontsize=10)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()
            
        except Exception as e:
            print(f"Error generating chart image: {e}")
            return None

    def analyze_patterns_with_vision(self, df: pd.DataFrame) -> List[Dict]:
        """Use Roboflow Vision API to detect patterns."""
        if not self.vision_client:
            return []
            
        try:
            image_bytes = self._generate_chart_image(df)
            if not image_bytes:
                return []
                
            result = self.vision_client.run_workflow(
                workspace=ROBOFLOW_WORKSPACE,
                workflow_id=ROBOFLOW_WORKFLOW_ID,
                images={"image": image_bytes},
                use_cache=True
            )
            
            vision_patterns = []
            
            predictions = []
            if isinstance(result, list) and len(result) > 0:
                if 'outputs' in result[0]:
                    outputs = result[0]['outputs']
                    if len(outputs) > 0 and 'predictions' in outputs[0]:
                        preds_node = outputs[0]['predictions']
                        if 'predictions' in preds_node:
                            predictions = preds_node['predictions']
                        elif isinstance(preds_node, list):
                            predictions = preds_node
            elif isinstance(result, dict) and 'predictions' in result:
                 predictions = result['predictions']
            
            class_map = {
                "W_Bottom": "Double Bottom",
                "M_Head": "Double Top", 
                "H_S": "Head & Shoulders",
                "Inv_H_S": "Inverse H&S"
            }
            
            for pred in predictions:
                label_raw = pred.get('class', 'Unknown')
                label_human = class_map.get(label_raw, label_raw.replace("_", " "))
                
                conf = pred.get('confidence', 0)
                if conf < 0.3: continue  # Lower threshold for vision
                
                x, y = pred.get('x', 0), pred.get('y', 0)
                w, h = pred.get('width', 0), pred.get('height', 0)
                
                p_type = 'Bullish' if 'Bottom' in label_human or 'Inv' in label_human or 'Bull' in label_human else 'Bearish'
                
                vision_patterns.append({
                    'Pattern': label_human,
                    'Type': f"{p_type} (AI Vision)",
                    'Confidence': round(conf * 100, 1),
                    'Target': 'N/A',
                    'Meta': {
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                        'raw_label': label_raw
                    },
                    'Status': 'Detected'
                })
                
            return vision_patterns
            
        except Exception as e:
            print(f"Vision analysis failed: {e}")
            return []
    
    def _zigzag_indicator(self, close: pd.Series, high: pd.Series = None,
                          low: pd.Series = None, threshold: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
        """
        ZigZag indicator: identifies swing highs/lows where price reverses >= threshold%.

        This is threshold-based (3% default for daily data), making it trader-aligned.
        More robust than window-based argrelextrema:
        - Threshold-based: catches large swings regardless of local noise
        - O(n) time complexity
        - Naturally adaptive to volatility

        Args:
            close: Close price series
            high:  High price series (optional; uses close if None)
            low:   Low price series (optional; uses close if None)
            threshold: Minimum reversal % to mark a swing (0.03 = 3%)

        Returns:
            (peaks_idx, troughs_idx) as numpy arrays of integer positions
        """
        if high is None:
            high = close
        if low is None:
            low = close

        n = len(close)
        if n < 5:
            return np.array([]), np.array([])

        close_arr = close.values
        high_arr  = high.values
        low_arr   = low.values

        pivots    = []  # (idx, 'peak'|'trough')
        direction = 0   # 0=unset, 1=looking for top, -1=looking for bottom
        last_idx  = 0

        for i in range(1, n):
            if direction >= 0:  # Looking for top (or unset → start looking for top)
                if high_arr[i] >= high_arr[last_idx]:
                    last_idx = i
                elif (high_arr[last_idx] - close_arr[i]) / (high_arr[last_idx] + 1e-10) >= threshold:
                    pivots.append((last_idx, 'peak'))
                    direction = -1
                    last_idx  = i
            else:  # direction < 0: looking for bottom
                if low_arr[i] <= low_arr[last_idx]:
                    last_idx = i
                elif (close_arr[i] - low_arr[last_idx]) / (low_arr[last_idx] + 1e-10) >= threshold:
                    pivots.append((last_idx, 'trough'))
                    direction = 1
                    last_idx  = i

        peaks   = np.array([p[0] for p in pivots if p[1] == 'peak'],   dtype=int)
        troughs = np.array([p[0] for p in pivots if p[1] == 'trough'], dtype=int)
        return peaks, troughs

    def find_peaks_and_troughs(self, prices: pd.Series, order: int = None,
                                use_zigzag: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find peaks and troughs using ZigZag indicator (primary) with scipy fallback.

        ZigZag is used first because it is threshold-based and more trader-aligned.
        Falls back to scipy find_peaks if ZigZag returns too few points.
        """
        prices_arr  = prices.values
        use_order   = order if order is not None else self.order
        price_range = prices_arr.max() - prices_arr.min()
        if price_range == 0:
            return np.array([]), np.array([])

        # Primary: ZigZag indicator
        if use_zigzag and len(prices) >= 10:
            # Threshold scales with order: order=3→2%, order=5→2.5%, order=7→3%
            threshold = 0.015 + (use_order - 3) * 0.005
            threshold = max(threshold, 0.015)
            peaks, troughs = self._zigzag_indicator(prices, threshold=threshold)
            if len(peaks) >= 2 or len(troughs) >= 2:
                return peaks, troughs

        # Fallback: scipy find_peaks (for very low-volatility or short series)
        min_prominence = price_range * 0.005
        peaks,   _ = find_peaks( prices_arr, prominence=min_prominence, distance=use_order)
        troughs, _ = find_peaks(-prices_arr, prominence=min_prominence, distance=use_order)
        return peaks, troughs
    
    def _validate_pattern_quality(self, df: pd.DataFrame, start_idx: int, end_idx: int, 
                                   pattern_height: float, current_price: float,
                                   relaxed: bool = False) -> bool:
        """
        Validate pattern quality. If relaxed=True, uses much looser criteria.
        """
        min_height = self.min_pattern_height if not relaxed else 0.005  # 0.5% when relaxed
        
        if pattern_height < current_price * min_height:
            return False
        
        pattern_duration = end_idx - start_idx
        max_dur = 80 if relaxed else 60
        min_dur = 2 if relaxed else 3
        if pattern_duration < min_dur or pattern_duration > max_dur:
            return False
        
        recency = 120 if relaxed else 90
        if len(df) - end_idx > recency:
            return False
        
        return True
    
    def _check_volume_confirmation(self, df: pd.DataFrame, breakout_idx: int,
                                    pattern_type: str = '') -> bool:
        """
        Volume confirmation at breakout/reversal.

        For breakout patterns (Triangle, Channel, Wedge):
            Requires breakout bar volume > 1.5× 20-day average.

        For reversal patterns (Double Top/Bottom, H&S):
            Checks if reversal bar volume > 1.2× 20-day average.

        Returns True if volume confirms, False if not (or if Volume data missing).
        """
        if 'Volume' not in df.columns:
            return True  # No volume data → assume confirmed
        try:
            window_start = max(0, breakout_idx - 20)
            vol_ma20     = df['Volume'].iloc[window_start:breakout_idx].mean()
            if vol_ma20 == 0 or np.isnan(vol_ma20):
                return True

            breakout_vol = df['Volume'].iloc[min(breakout_idx, len(df) - 1)]

            # Breakout patterns need stronger volume confirmation
            if any(t in pattern_type for t in ['Triangle', 'Channel', 'Wedge', 'Continuation']):
                return float(breakout_vol) > vol_ma20 * 1.5
            else:
                # Reversal patterns: moderate confirmation
                return float(breakout_vol) > vol_ma20 * 1.2
        except Exception:
            return True
    
    def detect_double_top(self, df: pd.DataFrame, order: int = None, relaxed: bool = False) -> List[Dict]:
        """
        Detect Double Top pattern with adaptive tolerance.
        Scans with given order for multi-timeframe capability.
        """
        patterns = []
        window = 120 if relaxed else 90
        df_analysis = df.tail(window)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices, order=order)
        
        if len(peak_idx) < 2 or len(trough_idx) < 1:
            return patterns
        
        tolerance = 0.05 if relaxed else 0.03  # 5% when relaxed, 3% normally
        
        for i in range(len(peak_idx) - 1):
            p1_idx, p2_idx = peak_idx[i], peak_idx[i + 1]
            p1_price, p2_price = prices.iloc[p1_idx], prices.iloc[p2_idx]
            
            price_diff_pct = abs(p1_price - p2_price) / p1_price
            if price_diff_pct > tolerance:
                continue
            
            troughs_between = trough_idx[(trough_idx > p1_idx) & (trough_idx < p2_idx)]
            if len(troughs_between) == 0:
                continue
            
            trough_price = prices.iloc[troughs_between[0]]
            avg_peak = (p1_price + p2_price) / 2
            pattern_height = avg_peak - trough_price
            
            if not self._validate_pattern_quality(df_analysis, p1_idx, p2_idx, pattern_height, current_price, relaxed):
                continue
            
            confidence = (1 - price_diff_pct) * 100
            height_score = min(pattern_height / (current_price * 0.05), 1) * 15
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price < trough_price
            target = trough_price - pattern_height

            # Keypoint dates for chart overlay
            try:
                kp = [
                    {'date': str(df_analysis.index[p1_idx].date()), 'price': round(float(p1_price), 2), 'label': 'Peak 1'},
                    {'date': str(df_analysis.index[troughs_between[0]].date()), 'price': round(float(trough_price), 2), 'label': 'Neckline'},
                    {'date': str(df_analysis.index[p2_idx].date()), 'price': round(float(p2_price), 2), 'label': 'Peak 2'},
                ]
            except Exception:
                kp = []

            patterns.append({
                'Pattern': 'Double Top',
                'Type': 'Bearish Reversal',
                'Neckline': round(trough_price, 2),
                'Target': round(target, 2),
                'Confidence': round(confidence, 1),
                'Status': 'CONFIRMED' if confirmed else 'Forming',
                'Peak_Price': round(avg_peak, 2),
                'keypoints': kp,
            })
        
        # Return best patterns
        patterns.sort(key=lambda x: x['Confidence'], reverse=True)
        return patterns[:self.max_patterns_per_type]
    
    def detect_double_bottom(self, df: pd.DataFrame, order: int = None, relaxed: bool = False) -> List[Dict]:
        """Detect Double Bottom pattern with adaptive tolerance."""
        patterns = []
        window = 120 if relaxed else 90
        df_analysis = df.tail(window)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices, order=order)
        
        if len(trough_idx) < 2 or len(peak_idx) < 1:
            return patterns
        
        tolerance = 0.05 if relaxed else 0.03
        
        for i in range(len(trough_idx) - 1):
            t1_idx, t2_idx = trough_idx[i], trough_idx[i + 1]
            t1_price, t2_price = prices.iloc[t1_idx], prices.iloc[t2_idx]
            
            price_diff_pct = abs(t1_price - t2_price) / t1_price
            if price_diff_pct > tolerance:
                continue
            
            peaks_between = peak_idx[(peak_idx > t1_idx) & (peak_idx < t2_idx)]
            if len(peaks_between) == 0:
                continue
            
            peak_price = prices.iloc[peaks_between[0]]
            avg_trough = (t1_price + t2_price) / 2
            pattern_height = peak_price - avg_trough
            
            if not self._validate_pattern_quality(df_analysis, t1_idx, t2_idx, pattern_height, current_price, relaxed):
                continue
            
            confidence = (1 - price_diff_pct) * 100
            height_score = min(pattern_height / (current_price * 0.05), 1) * 15
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price > peak_price
            target = peak_price + pattern_height

            try:
                kp = [
                    {'date': str(df_analysis.index[t1_idx].date()), 'price': round(float(t1_price), 2), 'label': 'Bottom 1'},
                    {'date': str(df_analysis.index[peaks_between[0]].date()), 'price': round(float(peak_price), 2), 'label': 'Neckline'},
                    {'date': str(df_analysis.index[t2_idx].date()), 'price': round(float(t2_price), 2), 'label': 'Bottom 2'},
                ]
            except Exception:
                kp = []

            patterns.append({
                'Pattern': 'Double Bottom',
                'Type': 'Bullish Reversal',
                'Neckline': round(peak_price, 2),
                'Target': round(target, 2),
                'Confidence': round(confidence, 1),
                'Status': 'CONFIRMED' if confirmed else 'Forming',
                'Trough_Price': round(avg_trough, 2),
                'keypoints': kp,
            })
        
        patterns.sort(key=lambda x: x['Confidence'], reverse=True)
        return patterns[:self.max_patterns_per_type]
    
    def detect_head_and_shoulders(self, df: pd.DataFrame, order: int = None, relaxed: bool = False) -> List[Dict]:
        """Detect Head & Shoulders with adaptive validation."""
        patterns = []
        window = 120 if relaxed else 90
        df_analysis = df.tail(window)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices, order=order)
        
        if len(peak_idx) < 3 or len(trough_idx) < 2:
            return patterns
        
        min_head_height = 0.01 if relaxed else 0.015
        max_shoulder_diff = 0.06 if relaxed else 0.04
        
        for i in range(len(peak_idx) - 2):
            ls_idx, h_idx, rs_idx = peak_idx[i], peak_idx[i+1], peak_idx[i+2]
            ls_price = prices.iloc[ls_idx]
            h_price = prices.iloc[h_idx]
            rs_price = prices.iloc[rs_idx]
            
            avg_shoulder = (ls_price + rs_price) / 2
            head_height_pct = (h_price - avg_shoulder) / avg_shoulder
            
            if head_height_pct < min_head_height:
                continue
            
            shoulder_diff = abs(ls_price - rs_price) / ls_price
            if shoulder_diff > max_shoulder_diff:
                continue
            
            troughs_1 = trough_idx[(trough_idx > ls_idx) & (trough_idx < h_idx)]
            troughs_2 = trough_idx[(trough_idx > h_idx) & (trough_idx < rs_idx)]
            
            if len(troughs_1) == 0 or len(troughs_2) == 0:
                continue
            
            neckline = (prices.iloc[troughs_1[0]] + prices.iloc[troughs_2[0]]) / 2
            pattern_height = h_price - neckline
            
            if not self._validate_pattern_quality(df_analysis, ls_idx, rs_idx, pattern_height, current_price, relaxed):
                continue
            
            confidence = (1 - shoulder_diff) * 100
            height_score = min(head_height_pct * 100, 15)
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price < neckline
            target = neckline - pattern_height

            try:
                kp = [
                    {'date': str(df_analysis.index[ls_idx].date()), 'price': round(float(ls_price), 2), 'label': 'Left Shoulder'},
                    {'date': str(df_analysis.index[h_idx].date()), 'price': round(float(h_price), 2), 'label': 'Head'},
                    {'date': str(df_analysis.index[rs_idx].date()), 'price': round(float(rs_price), 2), 'label': 'Right Shoulder'},
                    {'date': str(df_analysis.index[troughs_1[0]].date()), 'price': round(float(neckline), 2), 'label': 'Neckline L'},
                    {'date': str(df_analysis.index[troughs_2[0]].date()), 'price': round(float(neckline), 2), 'label': 'Neckline R'},
                ]
            except Exception:
                kp = []

            patterns.append({
                'Pattern': 'Head & Shoulders',
                'Type': 'Bearish Reversal',
                'Neckline': round(neckline, 2),
                'Target': round(target, 2),
                'Confidence': round(confidence, 1),
                'Status': 'CONFIRMED' if confirmed else 'Forming',
                'Head_Price': round(h_price, 2),
                'keypoints': kp,
            })
        
        patterns.sort(key=lambda x: x['Confidence'], reverse=True)
        return patterns[:self.max_patterns_per_type]
    
    def detect_inverse_head_and_shoulders(self, df: pd.DataFrame, order: int = None, relaxed: bool = False) -> List[Dict]:
        """Detect Inverse H&S with adaptive validation."""
        patterns = []
        window = 120 if relaxed else 90
        df_analysis = df.tail(window)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices, order=order)
        
        if len(trough_idx) < 3 or len(peak_idx) < 2:
            return patterns
        
        min_head_depth = 0.01 if relaxed else 0.015
        max_shoulder_diff = 0.06 if relaxed else 0.04
        
        for i in range(len(trough_idx) - 2):
            ls_idx, h_idx, rs_idx = trough_idx[i], trough_idx[i+1], trough_idx[i+2]
            ls_price = prices.iloc[ls_idx]
            h_price = prices.iloc[h_idx]
            rs_price = prices.iloc[rs_idx]
            
            avg_shoulder = (ls_price + rs_price) / 2
            head_depth_pct = (avg_shoulder - h_price) / avg_shoulder
            
            if head_depth_pct < min_head_depth:
                continue
            
            shoulder_diff = abs(ls_price - rs_price) / ls_price
            if shoulder_diff > max_shoulder_diff:
                continue
            
            peaks_1 = peak_idx[(peak_idx > ls_idx) & (peak_idx < h_idx)]
            peaks_2 = peak_idx[(peak_idx > h_idx) & (peak_idx < rs_idx)]
            
            if len(peaks_1) == 0 or len(peaks_2) == 0:
                continue
            
            neckline = (prices.iloc[peaks_1[0]] + prices.iloc[peaks_2[0]]) / 2
            pattern_height = neckline - h_price
            
            if not self._validate_pattern_quality(df_analysis, ls_idx, rs_idx, pattern_height, current_price, relaxed):
                continue
            
            confidence = (1 - shoulder_diff) * 100
            height_score = min(head_depth_pct * 100, 15)
            confidence = min(confidence + height_score, 99)
            
            confirmed = current_price > neckline
            target = neckline + pattern_height

            try:
                kp = [
                    {'date': str(df_analysis.index[ls_idx].date()), 'price': round(float(ls_price), 2), 'label': 'Left Shoulder'},
                    {'date': str(df_analysis.index[h_idx].date()), 'price': round(float(h_price), 2), 'label': 'Head'},
                    {'date': str(df_analysis.index[rs_idx].date()), 'price': round(float(rs_price), 2), 'label': 'Right Shoulder'},
                    {'date': str(df_analysis.index[peaks_1[0]].date()), 'price': round(float(neckline), 2), 'label': 'Neckline L'},
                    {'date': str(df_analysis.index[peaks_2[0]].date()), 'price': round(float(neckline), 2), 'label': 'Neckline R'},
                ]
            except Exception:
                kp = []

            patterns.append({
                'Pattern': 'Inverse H&S',
                'Type': 'Bullish Reversal',
                'Neckline': round(neckline, 2),
                'Target': round(target, 2),
                'Confidence': round(confidence, 1),
                'Status': 'CONFIRMED' if confirmed else 'Forming',
                'Head_Price': round(h_price, 2),
                'keypoints': kp,
            })
        
        patterns.sort(key=lambda x: x['Confidence'], reverse=True)
        return patterns[:self.max_patterns_per_type]
    
    def detect_trend(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Detect current trend using multiple confirmation methods."""
        prices = df['Close'].tail(window)
        
        x = np.arange(len(prices))
        slope, intercept, r_value, _, _ = linregress(x, prices.values)
        r_squared = r_value ** 2
        
        ma_short = df['Close'].rolling(5).mean().iloc[-1]
        ma_long = df['Close'].rolling(20).mean().iloc[-1]
        
        recent_highs = df['High'].tail(10)
        recent_lows = df['Low'].tail(10)
        
        higher_highs = (recent_highs.diff().dropna() > 0).sum() / len(recent_highs.diff().dropna())
        higher_lows = (recent_lows.diff().dropna() > 0).sum() / len(recent_lows.diff().dropna())
        
        bullish_score = 0
        bearish_score = 0
        
        if slope > 0:
            bullish_score += 1 + (r_squared * 0.5)
        else:
            bearish_score += 1 + (r_squared * 0.5)
            
        if ma_short > ma_long:
            bullish_score += 1
        else:
            bearish_score += 1
            
        if higher_highs > 0.6 and higher_lows > 0.6:
            bullish_score += 1
        elif higher_highs < 0.4 and higher_lows < 0.4:
            bearish_score += 1
        
        total_score = bullish_score + bearish_score
        if bullish_score > bearish_score:
            trend = "Bullish"
            strength = (bullish_score / total_score) * 100 if total_score > 0 else 50
        elif bearish_score > bullish_score:
            trend = "Bearish"
            strength = (bearish_score / total_score) * 100 if total_score > 0 else 50
        else:
            trend = "Neutral"
            strength = 50
        
        return {
            'Trend': trend,
            'Strength': round(strength, 1),
            'Slope': round(slope, 4),
            'R_Squared': round(r_squared, 3),
            'MA_Signal': 'Bullish' if ma_short > ma_long else 'Bearish',
            'Structure': 'Higher Highs/Lows' if higher_highs > 0.6 else 'Lower Highs/Lows' if higher_highs < 0.4 else 'Mixed'
        }
    
    def detect_support_resistance(self, df: pd.DataFrame, lookback: int = 90) -> Dict:
        """
        Detect significant S/R levels using volume-weighted clustering.

        Improvements over price-only clustering:
        - S/R strength = touch_count × avg_volume_at_touches / 20D_avg_volume
        - High-volume touches are far stronger than low-volume ones
        - Round number detection (₹50, ₹100, ₹500, ₹1000 increments)
          flagged as psychological S/R — common in Indian retail markets
        """
        df_slice      = df.tail(lookback).copy()
        prices        = df_slice['Close']
        current_price = float(prices.iloc[-1])
        has_volume    = 'Volume' in df_slice.columns

        avg_vol = float(df_slice['Volume'].rolling(20).mean().iloc[-1]) if has_volume else 1.0
        if avg_vol == 0 or np.isnan(avg_vol):
            avg_vol = 1.0

        # Collect candidate S/R prices from ZigZag swing points
        all_peak_prices   = []
        all_trough_prices = []

        # Use ZigZag swings (more meaningful than multi-order scipy)
        zz_peaks, zz_troughs = self._zigzag_indicator(prices, threshold=0.02)
        all_peak_prices.extend([float(prices.iloc[i]) for i in zz_peaks])
        all_trough_prices.extend([float(prices.iloc[i]) for i in zz_troughs])

        # Supplement with scipy for low-volatility stocks
        for ord_val in [3, 5]:
            pk, tr = find_peaks(prices.values, prominence=np.ptp(prices.values) * 0.005, distance=ord_val)
            tk, _ = find_peaks(-prices.values, prominence=np.ptp(prices.values) * 0.005, distance=ord_val)
            all_peak_prices.extend([float(prices.iloc[i]) for i in pk])
            all_trough_prices.extend([float(prices.iloc[i]) for i in tk])

        def volume_weighted_cluster(candidate_prices: list, tol: float = 0.005) -> list:
            """
            Cluster price levels within tol% of each other.
            Strength = touch_count × avg_vol_at_touches / 20D_avg_vol
            """
            if not candidate_prices:
                return []

            results = []
            candidates = sorted(set(candidate_prices))

            for price_level in candidates:
                # Find all bars where High/Low touched this level (within tolerance)
                if has_volume:
                    touches = df_slice[
                        (df_slice['High'] >= price_level * (1 - tol)) &
                        (df_slice['Low']  <= price_level * (1 + tol))
                    ]
                    n_touches = len(touches)
                    vol_at_touches = float(touches['Volume'].mean()) if n_touches > 0 else avg_vol
                else:
                    n_touches     = 1
                    vol_at_touches = avg_vol

                vol_ratio    = vol_at_touches / (avg_vol + 1e-10)
                strength     = n_touches * vol_ratio

                if n_touches >= 1:
                    results.append({'price': price_level, 'strength': strength, 'touches': n_touches})

            # Merge overlapping clusters (within tol)
            merged = []
            results.sort(key=lambda x: x['price'])
            for item in results:
                if merged and abs(item['price'] - merged[-1]['price']) / merged[-1]['price'] < tol:
                    # Keep stronger one
                    if item['strength'] > merged[-1]['strength']:
                        merged[-1] = item
                else:
                    merged.append(item)

            return sorted(merged, key=lambda x: x['strength'], reverse=True)

        resistance_clusters = volume_weighted_cluster(all_peak_prices)
        support_clusters    = volume_weighted_cluster(all_trough_prices)

        resistance_levels = [c['price'] for c in resistance_clusters]
        support_levels    = [c['price'] for c in support_clusters]

        # --- Round Number Detection ---
        # ₹50 increments near current price (within ±15%)
        round_number_levels = []
        for increment in [50, 100, 500, 1000]:
            if current_price < 50 and increment > 50:
                continue
            base = round(current_price / increment) * increment
            for offset in [-2, -1, 0, 1, 2]:
                rn = base + offset * increment
                if rn > 0 and abs(rn - current_price) / current_price < 0.15:
                    round_number_levels.append({
                        'price': float(rn),
                        'strength': 0.5,  # Moderate strength
                        'touches': 0,
                        'is_round_number': True
                    })

        # Merge round numbers into S/R
        for rn in round_number_levels:
            if rn['price'] > current_price * 1.002:
                resistance_clusters.append(rn)
            elif rn['price'] < current_price * 0.998:
                support_clusters.append(rn)

        resistance_above = sorted([r for r in resistance_levels if r > current_price * 1.002])
        support_below    = sorted([s for s in support_levels if s < current_price * 0.998], reverse=True)

        nearest_resistance = resistance_above[0] if resistance_above else None
        nearest_support    = support_below[0]    if support_below    else None

        return {
            'Current_Price':      round(current_price, 2),
            'Nearest_Resistance': round(nearest_resistance, 2) if nearest_resistance else 'N/A',
            'Nearest_Support':    round(nearest_support, 2)    if nearest_support    else 'N/A',
            'All_Resistance':     [round(r, 2) for r in sorted(set(resistance_levels), reverse=True)[:5]],
            'All_Support':        [round(s, 2) for s in sorted(set(support_levels))[:5]],
            'Resistance_Details': resistance_clusters[:5],
            'Support_Details':    support_clusters[:5]
        }
    
    def detect_triangle_pattern(self, df: pd.DataFrame, order: int = None) -> List[Dict]:
        """
        Detect Triangle Patterns with relaxed R² requirements.
        Tries multiple window sizes for better detection.
        """
        patterns = []
        
        for window in [30, 40, 50]:
            df_analysis = df.tail(window)
            prices = df_analysis['Close']
            highs = df_analysis['High']
            lows = df_analysis['Low']
            current_price = prices.iloc[-1]
            
            peak_idx, _ = self.find_peaks_and_troughs(highs, order=order or 3)
            _, trough_idx = self.find_peaks_and_troughs(lows, order=order or 3)
            
            if len(peak_idx) < 2 or len(trough_idx) < 2:
                continue
            
            # Use last 2-4 peaks/troughs
            n_points = min(4, len(peak_idx), len(trough_idx))
            recent_peaks = highs.iloc[peak_idx[-n_points:]]
            recent_troughs = lows.iloc[trough_idx[-n_points:]]
            
            x_peaks = np.arange(len(recent_peaks))
            slope_res, _, r_res, _, _ = linregress(x_peaks, recent_peaks.values)
            
            x_troughs = np.arange(len(recent_troughs))
            slope_sup, _, r_sup, _, _ = linregress(x_troughs, recent_troughs.values)
            
            # Relaxed R² requirement (0.3 instead of 0.6)
            min_r2 = 0.3
            if r_res**2 < min_r2 or r_sup**2 < min_r2:
                continue
            
            # Normalize slopes by price for comparability
            price_scale = current_price / 100
            norm_slope_res = slope_res / price_scale
            norm_slope_sup = slope_sup / price_scale
            
            # Ascending Triangle: Flat resistance, rising support
            if abs(norm_slope_res) < 0.3 and norm_slope_sup > 0.1:
                confidence = (r_res**2 + r_sup**2) / 2 * 100
                # Bonus for convergence tightness
                range_pct = (recent_peaks.max() - recent_troughs.min()) / current_price * 100
                confidence = min(confidence + range_pct, 95)
                patterns.append({
                    'Pattern': f'Ascending Triangle ({window}D)',
                    'Type': 'Bullish Continuation',
                    'Confidence': round(confidence, 1),
                    'Target': round(current_price * 1.05, 2),
                    'Status': 'Forming'
                })
                
            # Descending Triangle: Falling resistance, flat support
            elif norm_slope_res < -0.1 and abs(norm_slope_sup) < 0.3:
                confidence = (r_res**2 + r_sup**2) / 2 * 100
                range_pct = (recent_peaks.max() - recent_troughs.min()) / current_price * 100
                confidence = min(confidence + range_pct, 95)
                patterns.append({
                    'Pattern': f'Descending Triangle ({window}D)',
                    'Type': 'Bearish Continuation',
                    'Confidence': round(confidence, 1),
                    'Target': round(current_price * 0.95, 2),
                    'Status': 'Forming'
                })
                
            # Symmetrical Triangle: Converging slopes
            elif norm_slope_res < -0.05 and norm_slope_sup > 0.05:
                confidence = (r_res**2 + r_sup**2) / 2 * 100
                range_pct = (recent_peaks.max() - recent_troughs.min()) / current_price * 100
                confidence = min(confidence + range_pct, 95)
                bias = 'Bullish' if norm_slope_sup > abs(norm_slope_res) else 'Bearish'
                patterns.append({
                    'Pattern': f'Symmetrical Triangle ({window}D)',
                    'Type': f'{bias} Breakout Pending',
                    'Confidence': round(confidence, 1),
                    'Target': round(current_price * (1.05 if bias == 'Bullish' else 0.95), 2),
                    'Status': 'Forming'
                })
        
        # Deduplicate (keep best per type)
        seen_types = {}
        for p in sorted(patterns, key=lambda x: x['Confidence'], reverse=True):
            base_type = p['Pattern'].split(' (')[0]
            if base_type not in seen_types:
                seen_types[base_type] = p
        
        return list(seen_types.values())

    def detect_wedge_pattern(self, df: pd.DataFrame, order: int = None) -> List[Dict]:
        """Detect Rising/Falling Wedges with multiple windows."""
        patterns = []
        
        for window in [30, 40, 50]:
            df_analysis = df.tail(window)
            highs = df_analysis['High']
            lows = df_analysis['Low']
            current_price = df_analysis['Close'].iloc[-1]
            
            peak_idx, _ = self.find_peaks_and_troughs(highs, order=order or 3)
            _, trough_idx = self.find_peaks_and_troughs(lows, order=order or 3)
            
            if len(peak_idx) < 2 or len(trough_idx) < 2:
                continue
            
            n_points = min(4, len(peak_idx), len(trough_idx))
            x_peaks = np.arange(n_points)
            slope_res, _, r_res, _, _ = linregress(x_peaks, highs.iloc[peak_idx[-n_points:]].values)
            slope_sup, _, r_sup, _, _ = linregress(x_peaks, lows.iloc[trough_idx[-n_points:]].values)
            
            if r_res**2 < 0.3 or r_sup**2 < 0.3:
                continue
            
            # Rising Wedge: Both slopes positive, support steeper (converging)
            if slope_res > 0 and slope_sup > 0 and slope_sup > slope_res * 0.5:
                confidence = (r_res**2 + r_sup**2) / 2 * 100
                patterns.append({
                    'Pattern': f'Rising Wedge ({window}D)',
                    'Type': 'Bearish Reversal',
                    'Confidence': round(confidence, 1),
                    'Target': round(lows.iloc[trough_idx[-n_points]], 2),
                    'Status': 'Forming'
                })
                    
            # Falling Wedge: Both slopes negative, resistance steeper (converging)
            elif slope_res < 0 and slope_sup < 0 and slope_res < slope_sup * 0.5:
                confidence = (r_res**2 + r_sup**2) / 2 * 100
                patterns.append({
                    'Pattern': f'Falling Wedge ({window}D)',
                    'Type': 'Bullish Reversal',
                    'Confidence': round(confidence, 1),
                    'Target': round(highs.iloc[peak_idx[-n_points]], 2),
                    'Status': 'Forming'
                })
        
        # Deduplicate
        seen_types = {}
        for p in sorted(patterns, key=lambda x: x['Confidence'], reverse=True):
            base_type = p['Pattern'].split(' (')[0]
            if base_type not in seen_types:
                seen_types[base_type] = p
        return list(seen_types.values())
    
    def detect_channel_pattern(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Price Channels (Ascending, Descending, Horizontal).
        NEW pattern type for consolidating/trending markets.
        """
        patterns = []
        
        for window in [30, 50]:
            df_analysis = df.tail(window)
            prices = df_analysis['Close']
            highs = df_analysis['High']
            lows = df_analysis['Low']
            current_price = prices.iloc[-1]
            
            # Fit regression lines to highs and lows
            x = np.arange(len(df_analysis))
            
            slope_h, intercept_h, r_h, _, _ = linregress(x, highs.values)
            slope_l, intercept_l, r_l, _, _ = linregress(x, lows.values)
            
            # Both lines must be reasonably parallel (slopes within 50% of each other)
            if abs(slope_h) > 0 and abs(slope_l) > 0:
                slope_ratio = min(abs(slope_h), abs(slope_l)) / max(abs(slope_h), abs(slope_l))
            else:
                slope_ratio = 1.0 if abs(slope_h - slope_l) < 0.01 else 0.0
            
            if slope_ratio < 0.3:
                continue
            
            # Both lines need decent fit
            if r_h**2 < 0.3 or r_l**2 < 0.3:
                continue
            
            # Calculate channel width
            channel_width = (highs.mean() - lows.mean()) / current_price * 100
            
            avg_slope = (slope_h + slope_l) / 2
            norm_slope = avg_slope / (current_price / 100)
            
            if norm_slope > 0.05:
                pattern_name = 'Ascending Channel'
                pattern_type = 'Bullish Trend'
                target = round(current_price * 1.03, 2)
            elif norm_slope < -0.05:
                pattern_name = 'Descending Channel'
                pattern_type = 'Bearish Trend'
                target = round(current_price * 0.97, 2)
            else:
                pattern_name = 'Horizontal Channel'
                pattern_type = 'Range-Bound'
                # Target is the channel boundaries
                upper = intercept_h + slope_h * len(x)
                lower = intercept_l + slope_l * len(x)
                if current_price < (upper + lower) / 2:
                    target = round(upper, 2)
                else:
                    target = round(lower, 2)
            
            confidence = (r_h**2 + r_l**2) / 2 * 80 + slope_ratio * 20
            confidence = min(confidence, 95)
            
            # Position within channel
            upper_now = intercept_h + slope_h * (len(x) - 1)
            lower_now = intercept_l + slope_l * (len(x) - 1)
            position_pct = (current_price - lower_now) / (upper_now - lower_now) * 100 if upper_now != lower_now else 50
            
            patterns.append({
                'Pattern': f'{pattern_name} ({window}D)',
                'Type': pattern_type,
                'Confidence': round(confidence, 1),
                'Target': target,
                'Status': f'Price at {round(position_pct)}% of channel',
                'Channel_Width': f'{round(channel_width, 1)}%'
            })
        
        # Deduplicate
        seen_types = {}
        for p in sorted(patterns, key=lambda x: x['Confidence'], reverse=True):
            base_type = p['Pattern'].split(' (')[0]
            if base_type not in seen_types:
                seen_types[base_type] = p
        return list(seen_types.values())
    
    def detect_consolidation(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Consolidation / Range-Bound patterns.
        Useful when no classical patterns exist — tells user the market is coiling.
        """
        patterns = []
        df_analysis = df.tail(20)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        # Check if price is in a tight range
        price_range = (prices.max() - prices.min()) / current_price * 100
        volatility = prices.pct_change().std() * 100
        
        if price_range < 5:  # Less than 5% range in 20 days
            # Bollinger Band squeeze indicator
            ma20 = prices.mean()
            std20 = prices.std()
            bb_width = (2 * std20 / ma20) * 100
            
            confidence = max(40, 90 - price_range * 10)  # Tighter range = higher confidence
            
            # Determine likely breakout direction from trend
            x = np.arange(len(prices))
            slope, _, _, _, _ = linregress(x, prices.values)
            
            if slope > 0:
                bias = 'Bullish'
                target = round(current_price * (1 + price_range / 100), 2)
            elif slope < 0:
                bias = 'Bearish'
                target = round(current_price * (1 - price_range / 100), 2)
            else:
                bias = 'Neutral'
                target = round(current_price, 2)
            
            patterns.append({
                'Pattern': 'Consolidation / Squeeze',
                'Type': f'{bias} Breakout Expected',
                'Confidence': round(confidence, 1),
                'Target': target,
                'Status': f'Range: {round(price_range, 1)}%, BB Width: {round(bb_width, 1)}%',
                'Range_Pct': round(price_range, 1)
            })
        
        return patterns
    
    def detect_higher_high_lower_low(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Higher Highs / Higher Lows (uptrend) or Lower Highs / Lower Lows (downtrend).
        Simple but very reliable structural pattern.
        """
        patterns = []
        df_analysis = df.tail(40)
        prices = df_analysis['Close']
        current_price = prices.iloc[-1]
        
        peak_idx, trough_idx = self.find_peaks_and_troughs(prices, order=3)
        
        if len(peak_idx) < 3 or len(trough_idx) < 3:
            return patterns
        
        # Check last 3 peaks and troughs
        recent_peaks = [prices.iloc[idx] for idx in peak_idx[-3:]]
        recent_troughs = [prices.iloc[idx] for idx in trough_idx[-3:]]
        
        # Higher Highs and Higher Lows = Uptrend
        hh = all(recent_peaks[i] > recent_peaks[i-1] for i in range(1, len(recent_peaks)))
        hl = all(recent_troughs[i] > recent_troughs[i-1] for i in range(1, len(recent_troughs)))
        
        # Lower Highs and Lower Lows = Downtrend
        lh = all(recent_peaks[i] < recent_peaks[i-1] for i in range(1, len(recent_peaks)))
        ll = all(recent_troughs[i] < recent_troughs[i-1] for i in range(1, len(recent_troughs)))
        
        if hh and hl:
            avg_rise = np.mean([recent_peaks[i] - recent_peaks[i-1] for i in range(1, len(recent_peaks))])
            target = round(current_price + avg_rise, 2)
            patterns.append({
                'Pattern': 'Higher Highs & Higher Lows',
                'Type': 'Bullish Trend Structure',
                'Confidence': round(85.0, 1),
                'Target': target,
                'Status': 'Active Uptrend',
                'Last_HH': round(recent_peaks[-1], 2),
                'Last_HL': round(recent_troughs[-1], 2)
            })
        elif lh and ll:
            avg_drop = np.mean([recent_peaks[i-1] - recent_peaks[i] for i in range(1, len(recent_peaks))])
            target = round(current_price - avg_drop, 2)
            patterns.append({
                'Pattern': 'Lower Highs & Lower Lows',
                'Type': 'Bearish Trend Structure',
                'Confidence': round(85.0, 1),
                'Target': target,
                'Status': 'Active Downtrend',
                'Last_LH': round(recent_peaks[-1], 2),
                'Last_LL': round(recent_troughs[-1], 2)
            })
        elif hh and not hl:
            patterns.append({
                'Pattern': 'Higher Highs (Diverging)',
                'Type': 'Weakening Bullish',
                'Confidence': 65.0,
                'Target': round(recent_peaks[-1], 2),
                'Status': 'Watch for reversal'
            })
        elif lh and not ll:
            patterns.append({
                'Pattern': 'Lower Highs (Compressing)',
                'Type': 'Building Bearish',
                'Confidence': 65.0,
                'Target': round(recent_troughs[-1], 2),
                'Status': 'Watch for breakdown'
            })
        
        return patterns

    def _run_detection_pass(self, df: pd.DataFrame, order: int, relaxed: bool = False) -> List[Dict]:
        """Run all pattern detectors with given parameters."""
        all_patterns = []
        all_patterns.extend(self.detect_double_top(df, order=order, relaxed=relaxed))
        all_patterns.extend(self.detect_double_bottom(df, order=order, relaxed=relaxed))
        all_patterns.extend(self.detect_head_and_shoulders(df, order=order, relaxed=relaxed))
        all_patterns.extend(self.detect_inverse_head_and_shoulders(df, order=order, relaxed=relaxed))
        return all_patterns

    def _get_weekly_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample daily OHLCV to weekly. No API call — computed from existing data.
        Used for multi-timeframe confluence detection.
        """
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                return pd.DataFrame()
            weekly = df.resample('W').agg({
                'Open':   'first',
                'High':   'max',
                'Low':    'min',
                'Close':  'last',
                'Volume': 'sum'
            }).dropna()
            return weekly if len(weekly) >= 10 else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def analyze_all_patterns(self, df: pd.DataFrame, _weekly_call: bool = False) -> Dict:
        """
        Run comprehensive pattern analysis with:
        1. ZigZag-based swing detection (primary)
        2. Multi-timeframe confluence (daily + weekly)
        3. Hurst exponent confidence adjustment
        4. Volume confirmation flagging
        5. Auto-fallback to relaxed parameters if needed

        Args:
            df: Daily OHLCV DataFrame
            _weekly_call: Internal flag to prevent infinite recursion on weekly resampling

        Returns:
            Dict with patterns, trend, S/R, bias, Hurst, market_character
        """
        all_patterns = []

        # === Hurst Exponent (uses last 120 closes, < 0.5s) ===
        H = _hurst_exponent_pa(df['Close'].values[-120:] if len(df) >= 120 else df['Close'].values)
        if H > 0.55:
            market_character = 'Trending'
        elif H < 0.45:
            market_character = 'Mean-Reverting'
        else:
            market_character = 'Random Walk'

        # === PASS 1: Multi-timeframe classical pattern scan ===
        for scan_order in self.scan_orders:
            all_patterns.extend(self._run_detection_pass(df, order=scan_order, relaxed=False))

        # === PASS 2: Geometric patterns (triangles, wedges, channels) ===
        all_patterns.extend(self.detect_triangle_pattern(df))
        all_patterns.extend(self.detect_wedge_pattern(df))
        all_patterns.extend(self.detect_channel_pattern(df))

        # === PASS 3: Structural patterns (always run) ===
        all_patterns.extend(self.detect_higher_high_lower_low(df))
        all_patterns.extend(self.detect_consolidation(df))

        # === PASS 4: Fallback — if fewer than 2 high-confidence patterns ===
        high_conf = [p for p in all_patterns if p and p.get('Confidence', 0) >= 50]
        if len(high_conf) < 2:
            for scan_order in [2, 3, 5]:
                all_patterns.extend(self._run_detection_pass(df, order=scan_order, relaxed=True))

        # === PASS 5: Vision patterns (Roboflow) ===
        vision_patterns = self.analyze_patterns_with_vision(df)
        all_patterns.extend(vision_patterns)

        # === Filter and deduplicate ===
        valid_patterns = [p for p in all_patterns if p and p.get('Confidence', 0) >= 40]

        seen = {}
        for p in sorted(valid_patterns, key=lambda x: x.get('Confidence', 0), reverse=True):
            base_name = p['Pattern'].split(' (')[0]
            if base_name not in seen:
                seen[base_name] = p

        daily_patterns = list(seen.values())

        # === Multi-Timeframe Confluence (weekly resampling, ~2-3s) ===
        weekly_pattern_names = set()
        if not _weekly_call:
            weekly_df = self._get_weekly_ohlcv(df)
            if not weekly_df.empty:
                try:
                    weekly_analyst  = PatternAnalyst(order=3)
                    weekly_result   = weekly_analyst.analyze_all_patterns(weekly_df, _weekly_call=True)
                    weekly_pattern_names = {
                        p['Pattern'].split(' (')[0]
                        for p in weekly_result.get('patterns', [])
                    }
                except Exception:
                    weekly_pattern_names = set()

        # === Apply Hurst Adjustment + Volume Confirmation + Confluence ===
        for pattern in daily_patterns:
            ptype = pattern.get('Type', '')
            conf  = pattern.get('Confidence', 50)

            # Volume confirmation (uses last bar of df as proxy breakout idx)
            vol_confirmed = self._check_volume_confirmation(df, len(df) - 1, ptype)
            pattern['volume_confirmed'] = vol_confirmed
            if vol_confirmed:
                conf = min(conf + 8, 99)

            # Multi-timeframe confluence boost
            base_name = pattern['Pattern'].split(' (')[0]
            if base_name in weekly_pattern_names:
                conf = min(conf * 1.4, 99)
                pattern['timeframe_confluence'] = True
            else:
                pattern['timeframe_confluence'] = False

            # Hurst confidence adjustment:
            # Reversal patterns (Double Top/Bottom, H&S) more reliable when H < 0.5 (mean-reverting)
            # Continuation/Trend patterns more reliable when H > 0.5 (trending)
            if 'Reversal' in ptype:
                if H < 0.45:
                    conf = min(conf * 1.20, 99)  # Mean-reverting → reversals more likely
                elif H > 0.60:
                    conf = conf * 0.80            # Strongly trending → reversals less likely
            elif any(t in ptype for t in ['Continuation', 'Trend', 'Structure']):
                if H > 0.55:
                    conf = min(conf * 1.20, 99)   # Trending → continuation more likely
                elif H < 0.40:
                    conf = conf * 0.80            # Mean-reverting → continuation less likely

            pattern['Confidence'] = round(float(conf), 1)

        # Re-sort after adjustments
        daily_patterns.sort(key=lambda x: x.get('Confidence', 0), reverse=True)

        # === Trend and S/R ===
        trend     = self.detect_trend(df)
        sr_levels = self.detect_support_resistance(df)

        # === Bias ===
        bullish_count = sum(1 for p in daily_patterns if 'Bullish' in p.get('Type', ''))
        bearish_count = sum(1 for p in daily_patterns if 'Bearish' in p.get('Type', ''))

        if bullish_count > bearish_count:
            pattern_bias = "Bullish"
        elif bearish_count > bullish_count:
            pattern_bias = "Bearish"
        else:
            pattern_bias = trend['Trend']

        return {
            'patterns':         daily_patterns[:8],
            'trend':            trend,
            'support_resistance': sr_levels,
            'overall_bias':     pattern_bias,
            'pattern_count':    len(daily_patterns),
            'hurst_exponent':   round(H, 3),
            'market_character': market_character
        }


# Backward compatibility
VISUAL_AI_AVAILABLE = True
