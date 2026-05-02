"""Dynamic fusion of three expert models using uncertainty-based weighting."""

import numpy as np
import pandas as pd
import streamlit as st

from .technical_expert import TechnicalExpertModel
from .sentiment_expert import SentimentExpertModel
from .volatility_expert import VolatilityExpertModel
from utils.technical_indicators import calculate_technical_indicators


class DynamicFusionFramework:
    """
    Combines technical, sentiment, and volatility experts.
    Weights: w_i = exp(-σ_i²) / Σ exp(-σ_j²)
    """

    def __init__(self):
        self.technical_model = TechnicalExpertModel()
        self.sentiment_model = SentimentExpertModel()
        self.volatility_model = VolatilityExpertModel()
        
        self.model_predictions = {
            'technical': [],
            'sentiment': [],
            'volatility': []
        }
        self.true_values = []
        
    def calculate_dynamic_weights(self) -> tuple:
        uncertainties = {
            'technical': self.technical_model.get_uncertainty(),
            'sentiment': self.sentiment_model.get_uncertainty(),
            'volatility': self.volatility_model.get_uncertainty()
        }
        
        weights = {}
        total_weight = 0

        for model_name, uncertainty in uncertainties.items():
            uncertainty = max(uncertainty, 1e-6)
            weight = np.exp(-uncertainty)
            weights[model_name] = weight
            total_weight += weight
        
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] /= total_weight
        else:
            for model_name in weights:
                weights[model_name] = 1/3
        
        return weights, uncertainties
    
    def train_models(self, stock_data: pd.DataFrame, sentiment_data: dict,
                     vix_data: pd.DataFrame, multi_source_sentiment: dict = None):
        self.multi_source_sentiment = multi_source_sentiment
        
        stock_data_with_indicators = calculate_technical_indicators(stock_data)

        tech_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA20', 'MA50', 'MA_Ratio_5_20',
            'Volatility_5D', 'Volatility_20D', 'ATR',
            'Volume_Ratio', 'RSI', 'MACD', 'MACD_Histogram',
            'Price_vs_MA20', 'Price_vs_MA50', 'Gap'
        ]
        
        available_features = [f for f in tech_features if f in stock_data_with_indicators.columns]
        X_tech = stock_data_with_indicators[available_features]
        y_tech = stock_data_with_indicators['Returns'].shift(-1).dropna()
        X_tech = X_tech.iloc[:-1]

        split_idx = int(len(X_tech) * 0.8)
        X_tech_train, X_tech_test = X_tech.iloc[:split_idx], X_tech.iloc[split_idx:]
        y_tech_train, y_tech_test = y_tech.iloc[:split_idx], y_tech.iloc[split_idx:]
        
        n_features_actual = X_tech.shape[1]
        if getattr(self.technical_model, 'n_features', None) != n_features_actual:
            self.technical_model.n_features = n_features_actual
            self.technical_model.model = self.technical_model._build_model()

        self.technical_model.train(X_tech_train, y_tech_train, X_tech_test, y_tech_test)
        
        sentiment_features = []
        sentiment_targets = []
        
        for date in stock_data_with_indicators.index:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in sentiment_data:
                daily_sentiments = sentiment_data[date_str]
                features = self.sentiment_model.extract_sentiment_features(daily_sentiments)
                sentiment_features.append(features[0])
                
                idx = stock_data_with_indicators.index.get_loc(date)
                if idx + 1 < len(stock_data_with_indicators):
                    target = stock_data_with_indicators['Returns'].iloc[idx + 1]
                    sentiment_targets.append(target)
        
        if len(sentiment_features) > 10:
            X_sent = np.array(sentiment_features)[:len(sentiment_targets)]
            y_sent = np.array(sentiment_targets)[:len(sentiment_features)]
            self.sentiment_model.train(X_sent, y_sent)
        
        volatility_features = []
        volatility_targets = []

        for i in range(len(stock_data_with_indicators)):
            if i >= 20:
                vix_slice = vix_data.iloc[:i+1] if len(vix_data) > i else vix_data
                stock_slice = stock_data_with_indicators.iloc[:i+1]
                
                features = self.volatility_model.extract_volatility_features(vix_slice, stock_slice)
                volatility_features.append(features[0])
                
                if i + 1 < len(stock_data_with_indicators):
                    target = stock_data_with_indicators['Returns'].iloc[i + 1]
                    volatility_targets.append(target)
        
        if len(volatility_features) > 10:
            X_vol = np.array(volatility_features)[:len(volatility_targets)]
            y_vol = np.array(volatility_targets)[:len(volatility_features)]
            self.volatility_model.train(X_vol, y_vol)
    
    def predict(self, stock_data: pd.DataFrame, sentiment_data: dict,
                vix_data: pd.DataFrame) -> dict:
        stock_features = calculate_technical_indicators(stock_data)
        
        if hasattr(self.technical_model.scaler, 'feature_names_in_'):
            tech_features = [f for f in self.technical_model.scaler.feature_names_in_ 
                           if f in stock_features.columns]
        else:
            tech_features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA5', 'MA20', 'MA50', 'MA_Ratio_5_20',
                'Volatility_5D', 'Volatility_20D', 'ATR',
                'Volume_Ratio', 'RSI', 'MACD', 'MACD_Histogram',
                'Price_vs_MA20', 'Price_vs_MA50', 'Gap'
            ]
            tech_features = [f for f in tech_features if f in stock_features.columns]
        
        if len(tech_features) < 5:
            tech_features = [col for col in stock_features.columns 
                           if col not in ['Returns', 'Target', 'Predicted']]
        
        tech_features = tech_features[:self.technical_model.n_features]
        
        X_tech = stock_features[tech_features].iloc[-self.technical_model.lookback:]
        
        if len(X_tech) < self.technical_model.lookback:
            st.warning(f"Insufficient data for technical model. Need {self.technical_model.lookback} days, have {len(X_tech)}.")

        tech_pred = self.technical_model.predict(X_tech)

        if hasattr(self, 'multi_source_sentiment') and self.multi_source_sentiment:
            sentiment_pred = self.sentiment_model.predict(self.multi_source_sentiment)
        else:
            latest_date = stock_data.index[-1].strftime('%Y-%m-%d')
            latest_sentiment = sentiment_data.get(latest_date, [])
            sentiment_pred = self.sentiment_model.predict(latest_sentiment)
        
        volatility_pred = self.volatility_model.predict(vix_data, stock_features)

        weights, uncertainties = self.calculate_dynamic_weights()

        self.model_predictions['technical'].append(tech_pred)
        self.model_predictions['sentiment'].append(sentiment_pred)
        self.model_predictions['volatility'].append(volatility_pred)

        combined_pred = (
            weights['technical'] * tech_pred +
            weights['sentiment'] * sentiment_pred +
            weights['volatility'] * volatility_pred
        )
        
        return {
            'combined_prediction': combined_pred,
            'individual_predictions': {
                'technical': tech_pred,
                'sentiment': sentiment_pred,
                'volatility': volatility_pred
            },
            'weights': weights,
            'uncertainties': uncertainties
        }
    
    def update_model_performance(self, true_return: float):
        self.true_values.append(true_return)

        if len(self.true_values) > 1 and len(self.model_predictions['technical']) > 0:
            last_true = self.true_values[-2]
            for model_name in ['technical', 'sentiment', 'volatility']:
                if len(self.model_predictions[model_name]) > 0:
                    last_pred = self.model_predictions[model_name][-1]
                    
                    if model_name == 'technical':
                        self.technical_model.update_errors(last_true, last_pred)
                    elif model_name == 'sentiment':
                        self.sentiment_model.update_errors(last_true, last_pred)
                    elif model_name == 'volatility':
                        self.volatility_model.update_errors(last_true, last_pred)
