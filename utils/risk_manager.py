"""
Risk management utilities.
Includes Fibonacci levels, Kelly Criterion, and trade setup calculations.
"""

import pandas as pd
import numpy as np


class RiskManager:
    """Professional Risk Management & Position Sizing"""
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate the latest Average True Range value.
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period (default 14)
        
        Returns:
            Latest ATR value
        """
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1]

    @staticmethod
    def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 90) -> dict:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of days to look back for high/low
        
        Returns:
            Dictionary of Fibonacci levels
        """
        recent_data = df.iloc[-lookback:]
        max_price = recent_data['High'].max()
        min_price = recent_data['Low'].min()
        diff = max_price - min_price
        
        levels = {
            "0.0% (Low)": min_price,
            "23.6%": min_price + 0.236 * diff,
            "38.2%": min_price + 0.382 * diff,
            "50.0%": min_price + 0.5 * diff,
            "61.8%": min_price + 0.618 * diff,
            "100.0% (High)": max_price
        }
        return levels

    @staticmethod
    def kelly_criterion(win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Formula: f* = (bp - q) / b
        Where:
            b = odds received (win/loss ratio)
            p = probability of winning
            q = probability of losing (1-p)
        
        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Average win / Average loss
        
        Returns:
            Optimal position size as fraction (0-1)
        """
        if win_loss_ratio == 0:
            return 0
        return max(0, (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio)

    @staticmethod
    def get_trade_setup(current_price: float, prediction: float, atr: float, 
                       confidence: float) -> dict:
        """
        Generate a complete trade setup with entry, stop loss, and target.
        
        Args:
            current_price: Current stock price
            prediction: Predicted price
            atr: Average True Range
            confidence: Model confidence (0-1)
        
        Returns:
            Dictionary with Direction, Entry, Stop Loss, Target, Risk/Reward
        """
        direction = "LONG" if prediction > current_price else "SHORT"
        
        # Dynamic Stop Loss based on ATR
        # Tighter stop for lower confidence
        sl_multiplier = 2.0 if confidence > 0.7 else 1.5
        stop_dist = atr * sl_multiplier
        
        if direction == "LONG":
            stop_loss = current_price - stop_dist
            # Target: 1.5x risk at minimum
            target = current_price + (stop_dist * 1.5)
        else:
            stop_loss = current_price + stop_dist
            target = current_price - (stop_dist * 1.5)
            
        risk_reward = abs(target - current_price) / abs(current_price - stop_loss)
        
        return {
            "Direction": direction,
            "Entry": current_price,
            "Stop Loss": stop_loss,
            "Target": target,
            "Risk/Reward": risk_reward
        }
    
    @staticmethod
    def calculate_position_size(capital: float, risk_percent: float, 
                                entry: float, stop_loss: float) -> dict:
        """
        Calculate position size based on risk management rules.
        
        Args:
            capital: Total account capital
            risk_percent: Maximum risk per trade as percentage (e.g., 2 for 2%)
            entry: Entry price
            stop_loss: Stop loss price
        
        Returns:
            Dictionary with shares, position value, and actual risk
        """
        risk_amount = capital * (risk_percent / 100)
        risk_per_share = abs(entry - stop_loss)
        
        if risk_per_share == 0:
            return {"shares": 0, "position_value": 0, "risk_amount": 0}
        
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry
        
        return {
            "shares": shares,
            "position_value": position_value,
            "risk_amount": risk_amount
        }
