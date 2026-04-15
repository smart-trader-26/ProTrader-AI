"""
Technical Expert Model.
GRU-based neural network for technical price pattern analysis.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from config.settings import ModelConfig


class TechnicalExpertModel:
    """GRU-based model for technical data analysis."""
    
    def __init__(self, lookback: int = None, n_features: int = None):
        """
        Initialize the Technical Expert Model.
        
        Args:
            lookback: Number of time steps to look back
            n_features: Number of input features
        """
        self.lookback = lookback or ModelConfig.LOOKBACK_PERIOD
        self.n_features = n_features or ModelConfig.TECHNICAL_FEATURES
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        self.recent_errors = []
        self.max_error_window = ModelConfig.MAX_ERROR_WINDOW
        
    def _build_model(self) -> Sequential:
        """Build the GRU neural network architecture."""
        model = Sequential([
            GRU(ModelConfig.GRU_UNITS_1, return_sequences=True, 
                input_shape=(self.lookback, self.n_features)),
            BatchNormalization(),
            Dropout(ModelConfig.GRU_DROPOUT),
            GRU(ModelConfig.GRU_UNITS_2, return_sequences=True),
            BatchNormalization(),
            Dropout(ModelConfig.GRU_DROPOUT),
            GRU(ModelConfig.GRU_UNITS_3),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(
            optimizer=Adam(learning_rate=ModelConfig.GRU_LEARNING_RATE), 
            loss='mse', 
            metrics=['mae']
        )
        return model
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, lookback: int = None):
        """
        Prepare sequential data for GRU training.
        
        Args:
            X: Feature matrix
            y: Target values
            lookback: Number of time steps (uses self.lookback if None)
        
        Returns:
            Tuple of (X_3d, y_3d) ready for training
        """
        lookback = lookback or self.lookback
        X_scaled = self.scaler.fit_transform(X)
        X_3d = []
        y_3d = []
        
        for i in range(lookback, len(X_scaled)):
            X_3d.append(X_scaled[i-lookback:i])
            y_3d.append(y[i])
        
        return np.array(X_3d), np.array(y_3d)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = None, batch_size: int = None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
        
        Returns:
            Training history
        """
        epochs = epochs or ModelConfig.DEFAULT_EPOCHS
        batch_size = batch_size or ModelConfig.DEFAULT_BATCH_SIZE
        
        X_seq, y_seq = self.prepare_data(X_train, y_train, self.lookback)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_data(X_val, y_val, self.lookback)
            validation_data = (X_val_seq, y_val_seq)
        
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=0
        )
        return history
    
    def predict(self, X: np.ndarray) -> float:
        """
        Make prediction for the latest data point.
        
        Args:
            X: Feature matrix (should include at least lookback rows)
        
        Returns:
            Predicted return value
        """
        X_scaled = self.scaler.transform(X)
        
        if len(X_scaled) >= self.lookback:
            X_seq = X_scaled[-self.lookback:].reshape(1, self.lookback, -1)
            prediction = self.model.predict(X_seq, verbose=0)[0][0]
        else:
            # Pad with zeros if not enough data
            padding = np.zeros((self.lookback - len(X_scaled), X_scaled.shape[1]))
            X_padded = np.vstack([padding, X_scaled])
            X_seq = X_padded.reshape(1, self.lookback, -1)
            prediction = self.model.predict(X_seq, verbose=0)[0][0]
        
        return prediction
    
    def update_errors(self, true_value: float, predicted_value: float):
        """
        Update error tracking for uncertainty calculation.
        
        Args:
            true_value: Actual return value
            predicted_value: Predicted return value
        """
        error = (true_value - predicted_value) ** 2
        self.recent_errors.append(error)
        
        # Keep only last N errors
        if len(self.recent_errors) > self.max_error_window:
            self.recent_errors.pop(0)
    
    def get_uncertainty(self) -> float:
        """
        Calculate uncertainty (variance of recent errors).
        
        Returns:
            Uncertainty value (higher = less confident)
        """
        if len(self.recent_errors) == 0:
            return 1.0  # Maximum uncertainty if no data
        
        return np.mean(self.recent_errors)
