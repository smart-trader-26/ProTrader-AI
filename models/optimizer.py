"""
Model Optimizer using Optuna for hyperparameter tuning.
Bayesian optimization for XGBoost and GRU models.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Optional Optuna import
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False


class ModelOptimizer:
    """
    Bayesian Hyperparameter Optimization using Optuna.
    
    Optimizes hyperparameters for XGBoost and GRU models
    to minimize prediction RMSE.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initialize the optimizer with train/test data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def optimize_xgb(self, n_trials: int = 10) -> dict:
        """
        Optimize XGBoost hyperparameters.
        
        Args:
            n_trials: Number of optimization trials
        
        Returns:
            Best hyperparameters dictionary, or None if Optuna unavailable
        """
        if not OPTUNA_AVAILABLE:
            return None
        
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'reg:squarederror',
                'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**param)
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, preds))
            return rmse
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params

    def optimize_gru(self, input_shape: tuple, X_train_3d=None, 
                     X_test_3d=None, n_trials: int = 5) -> dict:
        """
        Optimize GRU hyperparameters.
        
        Args:
            input_shape: Input shape for GRU (timesteps, features)
            X_train_3d: 3D training data (optional, uses reshaped self.X_train if None)
            X_test_3d: 3D test data (optional)
            n_trials: Number of optimization trials
        
        Returns:
            Best hyperparameters dictionary, or None if Optuna unavailable
        """
        if not OPTUNA_AVAILABLE:
            return None
            
        # Use provided data if available
        train_x = X_train_3d if X_train_3d is not None else self.X_train.reshape(-1, 1, self.X_train.shape[1])
        test_x = X_test_3d if X_test_3d is not None else self.X_test.reshape(-1, 1, self.X_test.shape[1])
            
        def objective(trial):
            units1 = trial.suggest_int('units1', 32, 128)
            units2 = trial.suggest_int('units2', 16, 64)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            
            model = Sequential([
                GRU(units1, input_shape=input_shape, return_sequences=True),
                Dropout(dropout),
                GRU(units2),
                Dropout(dropout),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
            
            # Short training for speed
            model.fit(train_x, self.y_train, epochs=10, batch_size=32, verbose=0)
            
            preds = model.predict(test_x, verbose=0)
            rmse = np.sqrt(mean_squared_error(self.y_test, preds))
            return rmse
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params
    
    def get_optimized_xgb_model(self, params: dict = None):
        """
        Get an XGBoost model with optimized (or default) parameters.
        
        Args:
            params: Hyperparameters (uses optimization if None)
        
        Returns:
            Trained XGBRegressor
        """
        if params is None:
            params = self.optimize_xgb() or {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.05,
                'objective': 'reg:squarederror',
                'n_jobs': -1
            }
        
        model = xgb.XGBRegressor(**params)
        model.fit(self.X_train, self.y_train)
        return model
    
    def is_available(self) -> bool:
        """Check if Optuna is available for optimization."""
        return OPTUNA_AVAILABLE
