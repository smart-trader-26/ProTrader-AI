"""
Sentiment Expert Model.
Dense neural network for sentiment-based return prediction.
Supports multi-source sentiment (RSS, NewsAPI, Reddit, Google Trends).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from config.settings import ModelConfig


class SentimentExpertModel:
    """Dense network model for sentiment data analysis."""
    
    def __init__(self):
        """Initialize the Sentiment Expert Model."""
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        self.recent_errors = []
        self.max_error_window = ModelConfig.MAX_ERROR_WINDOW
        
    def _build_model(self) -> Model:
        """Build the dense neural network architecture."""
        # Updated to 8 features for multi-source sentiment
        input_layer = Input(shape=(8,))
        x = Dense(64, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        output_layer = Dense(1)(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=Adam(learning_rate=ModelConfig.GRU_LEARNING_RATE), 
            loss='mse', 
            metrics=['mae']
        )
        return model
    
    def extract_sentiment_features(self, sentiment_data) -> np.ndarray:
        """
        Extract numerical features from sentiment data.
        Supports both legacy format and multi-source format.
        
        Args:
            sentiment_data: Either:
                - List of (sentiment_label, confidence) tuples (legacy)
                - Dict from multi-source sentiment analysis (new)
        
        Returns:
            Feature array of shape (1, 8)
        """
        # Handle new multi-source sentiment format
        if isinstance(sentiment_data, dict) and 'combined_sentiment' in sentiment_data:
            sources = sentiment_data.get('sources', {})
            
            features = [
                # Combined metrics
                sentiment_data.get('combined_sentiment', 0),
                sentiment_data.get('confidence', 0.5),
                
                # Individual source sentiments
                sources.get('rss', {}).get('average_sentiment', 0),
                sources.get('newsapi', {}).get('average_sentiment', 0),
                sources.get('reddit', {}).get('average_sentiment', 0),
                sources.get('google_trends', {}).get('signal', 0),
                
                # Source availability (more sources = higher confidence)
                sum([1 for s in ['rss', 'newsapi', 'reddit', 'google_trends'] 
                     if sources.get(s, {}).get('available', False)]) / 4.0,
                
                # Article count normalized
                min(sentiment_data.get('article_count', 0) / 50.0, 1.0)
            ]
            return np.array(features).reshape(1, -1)
        
        # Handle legacy format: List of (sentiment_label, confidence) tuples
        if isinstance(sentiment_data, list) and len(sentiment_data) > 0:
            sentiments = [s[0] for s in sentiment_data]
            confidences = [s[1] for s in sentiment_data]
            
            sentiment_values = []
            for sentiment, confidence in zip(sentiments, confidences):
                if sentiment == 'positive':
                    sentiment_values.append(confidence)
                elif sentiment == 'negative':
                    sentiment_values.append(-confidence)
                else:
                    sentiment_values.append(0)
            
            if sentiment_values:
                features = [
                    np.mean(sentiment_values),  # Average sentiment
                    np.std(sentiment_values) if len(sentiment_values) > 1 else 0,
                    len([v for v in sentiment_values if v > 0]) / len(sentiment_values),
                    len([v for v in sentiment_values if v < 0]) / len(sentiment_values),
                    np.max(sentiment_values) if sentiment_values else 0,
                    0, 0, 0  # Placeholder for multi-source features
                ]
            else:
                features = [0, 0, 0.5, 0.5, 0, 0, 0, 0]
        else:
            features = [0, 0, 0.5, 0.5, 0, 0, 0, 0]  # Neutral if no sentiment data
        
        return np.array(features).reshape(1, -1)
    
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
    
    def predict(self, sentiment_data: list) -> float:
        """
        Make prediction based on sentiment data.
        
        Args:
            sentiment_data: List of (sentiment_label, confidence) tuples
        
        Returns:
            Predicted return value
        """
        features = self.extract_sentiment_features(sentiment_data)
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
