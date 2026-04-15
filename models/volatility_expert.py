"""
Volatility Expert Model.
MLP model for volatility (VIX) based return prediction.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from config.settings import ModelConfig


class VolatilityExpertModel:
    """MLP model for volatility (VIX) data analysis."""
    
    def __init__(self):
        """Initialize the Volatility Expert Model."""
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        self.recent_errors = []
        self.max_error_window = ModelConfig.MAX_ERROR_WINDOW
        
    def _build_model(self) -> Sequential:
        """Build the MLP neural network architecture."""
        model = Sequential([
            Dense(32, activation='relu', input_shape=(3,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(learning_rate=ModelConfig.GRU_LEARNING_RATE), 
            loss='mse', 
            metrics=['mae']
        )
        return model
    
    def extract_volatility_features(self, vix_data: pd.DataFrame, 
                                     stock_data: pd.DataFrame) -> np.ndarray:
        """
        Extract volatility features from VIX and stock data.
        
        Args:
            vix_data: DataFrame with VIX OHLCV data
            stock_data: DataFrame with stock data (should include Volatility_20D)
        
        Returns:
            Feature array of shape (1, 3)
        """
        if vix_data is None or stock_data is None or vix_data.empty or stock_data.empty:
            return np.array([[0.0, 0.0, 0.0]])

        # Ensure we operate on the 'Close' series for VIX
        if 'Close' in vix_data.columns:
            vix_close = vix_data['Close']
        else:
            vix_close = pd.Series(dtype=float)

        # Latest VIX close (scalar)
        latest_vix_close = 0.0
        if len(vix_close) > 0:
            last = vix_close.iloc[-1]
            try:
                if isinstance(last, (pd.Series, np.ndarray)):
                    arr = np.asarray(last).ravel()
                    if arr.size > 0 and not np.isnan(arr[-1]):
                        latest_vix_close = float(arr[-1])
                else:
                    if pd.notna(last):
                        latest_vix_close = float(last)
            except Exception:
                latest_vix_close = 0.0

        # VIX vs MA20 (safe computation)
        vix_vs_ma = 1.0
        if len(vix_close) >= 20:
            try:
                vix_ma20_raw = vix_close.rolling(20).mean().iloc[-1]
                if isinstance(vix_ma20_raw, (pd.Series, np.ndarray)):
                    vix_ma20_arr = np.asarray(vix_ma20_raw).ravel()
                    vix_ma20 = float(vix_ma20_arr[-1]) if vix_ma20_arr.size > 0 and not np.isnan(vix_ma20_arr[-1]) else None
                else:
                    vix_ma20 = float(vix_ma20_raw) if pd.notna(vix_ma20_raw) else None

                if vix_ma20 and vix_ma20 != 0:
                    vix_vs_ma = latest_vix_close / vix_ma20
                else:
                    vix_vs_ma = 1.0
            except Exception:
                vix_vs_ma = 1.0

        # Latest stock volatility (scalar)
        if 'Volatility_20D' in stock_data.columns and len(stock_data) > 0:
            latest_stock_vol = stock_data['Volatility_20D'].iloc[-1]
            latest_stock_vol = float(latest_stock_vol) if not pd.isna(latest_stock_vol) else 0.0
        else:
            latest_stock_vol = 0.0

        features = [latest_vix_close, vix_vs_ma, latest_stock_vol]
        return np.array(features, dtype=float).reshape(1, -1)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 30, batch_size: int = 16):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size
        
        Returns:
            Training history
        """
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        return history
    
    def predict(self, vix_data: pd.DataFrame, stock_data: pd.DataFrame) -> float:
        """
        Make prediction based on volatility data.
        
        Args:
            vix_data: DataFrame with VIX data
            stock_data: DataFrame with stock data
        
        Returns:
            Predicted return value
        """
        features = self.extract_volatility_features(vix_data, stock_data)
        prediction = self.model.predict(features, verbose=0)[0][0]
        return prediction
    
    def update_errors(self, true_value: float, predicted_value: float):
        """Update error tracking for uncertainty calculation."""
        error = (true_value - predicted_value) ** 2
        self.recent_errors.append(error)
        
        if len(self.recent_errors) > self.max_error_window:
            self.recent_errors.pop(0)
    
    def get_uncertainty(self) -> float:
        """Calculate uncertainty (variance of recent errors)."""
        if len(self.recent_errors) == 0:
            return 1.0
        
        return np.mean(self.recent_errors)
