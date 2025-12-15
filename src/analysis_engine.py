"""
Analysis Engine - Continuous ML Prediction
==========================================
Calculates features and runs ML predictions continuously.
Runs in background - dashboard closing doesn't stop it.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """
    LSTM Neural Network for price direction prediction.

    Architecture:
    - Input: sequence of normalized features
    - LSTM layers with dropout
    - Fully connected output layer
    - Sigmoid activation for probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, _ = self.lstm(x)

        # Take last timestep output
        last_out = lstm_out[:, -1, :]

        # Predict probability
        return self.fc(last_out)


class FeatureCalculator:
    """Calculate technical indicators from OHLCV data."""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical features.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            DataFrame with added feature columns
        """
        df = df.copy()

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # =================================================================
        # PRICE-BASED FEATURES
        # =================================================================

        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving Averages
        for period in [7, 14, 21, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Price relative to MAs
        df['price_sma_7_ratio'] = df['close'] / df['sma_7']
        df['price_sma_21_ratio'] = df['close'] / df['sma_21']
        df['price_sma_50_ratio'] = df['close'] / df['sma_50']

        # =================================================================
        # VOLATILITY FEATURES
        # =================================================================

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Historical Volatility
        df['volatility_14'] = df['returns'].rolling(14).std() * np.sqrt(252)
        df['volatility_30'] = df['returns'].rolling(30).std() * np.sqrt(252)

        # =================================================================
        # MOMENTUM FEATURES
        # =================================================================

        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ROC (Rate of Change)
        df['roc_10'] = df['close'].pct_change(10) * 100
        df['roc_20'] = df['close'].pct_change(20) * 100

        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

        # =================================================================
        # VOLUME FEATURES
        # =================================================================

        # Volume MA
        df['volume_sma_14'] = df['volume'].rolling(14).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_14']

        # OBV (On Balance Volume) - simplified
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        df['obv_sma'] = df['obv'].rolling(14).mean()

        # =================================================================
        # TREND FEATURES
        # =================================================================

        # ADX (Average Directional Index) - simplified
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # Trend strength
        df['trend_7'] = (df['close'] - df['close'].shift(7)) / df['close'].shift(7)
        df['trend_14'] = (df['close'] - df['close'].shift(14)) / df['close'].shift(14)

        # =================================================================
        # PATTERN FEATURES
        # =================================================================

        # Candle patterns
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_wick_upper'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['candle_wick_lower'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_body_ratio'] = df['candle_body'] / (df['high'] - df['low'] + 0.0001)

        # Higher highs, lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

        return df

    @staticmethod
    def get_feature_columns() -> list:
        """Get list of feature column names used for prediction."""
        return [
            'returns', 'log_returns',
            'price_sma_7_ratio', 'price_sma_21_ratio', 'price_sma_50_ratio',
            'atr_14', 'bb_width', 'bb_position',
            'volatility_14', 'volatility_30',
            'rsi_14', 'stoch_k', 'stoch_d',
            'macd', 'macd_signal', 'macd_hist',
            'roc_10', 'roc_20', 'williams_r',
            'volume_ratio',
            'adx', 'plus_di', 'minus_di',
            'trend_7', 'trend_14',
            'candle_body_ratio', 'higher_high', 'lower_low'
        ]


class AnalysisEngine:
    """
    Continuous analysis engine.

    - Calculates features from price data
    - Runs ML model predictions
    - Generates trading signals
    - Runs forever until stopped
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize analysis engine."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Model settings
        self.model_path = Path(self.config['model']['path'])
        self.sequence_length = self.config['model']['sequence_length']
        self.hidden_size = self.config['model']['hidden_size']
        self.num_layers = self.config['model']['num_layers']

        # Analysis settings
        self.min_confidence = self.config['analysis']['min_confidence']
        self.update_interval = self.config['analysis']['update_interval']

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._model: Optional[LSTMModel] = None
        self._last_prediction: Optional[dict] = None
        self._callbacks: list[Callable] = []

        # Feature scaler parameters (loaded with model or computed)
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

        self.feature_columns = FeatureCalculator.get_feature_columns()

        logger.info("AnalysisEngine initialized")

    def load_model(self) -> bool:
        """Load trained model from disk."""
        if not self.model_path.exists():
            logger.warning(f"Model not found: {self.model_path}")
            logger.info("Running in FEATURE-ONLY mode (no ML predictions)")
            return False

        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')

            input_size = len(self.feature_columns)
            self._model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            )
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.eval()

            # Load scaler parameters if saved
            if 'feature_means' in checkpoint:
                self._feature_means = checkpoint['feature_means']
                self._feature_stds = checkpoint['feature_stds']

            logger.info(f"Model loaded from {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features for given price data."""
        return FeatureCalculator.calculate_all(df)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization."""
        if self._feature_means is not None:
            return (features - self._feature_means) / (self._feature_stds + 1e-8)

        # If no saved scaler, compute from current data
        means = np.nanmean(features, axis=0)
        stds = np.nanstd(features, axis=0)
        return (features - means) / (stds + 1e-8)

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Make prediction for next candle.

        Args:
            df: DataFrame with OHLCV data (will calculate features)

        Returns:
            Dict with prediction details
        """
        # Calculate features
        df_features = self.calculate_features(df)

        # Get current price info
        current_price = df['close'].iloc[-1]
        current_time = df['datetime'].iloc[-1] if 'datetime' in df else datetime.utcnow()

        # Get feature values
        feature_df = df_features[self.feature_columns].iloc[-self.sequence_length:]

        # Handle missing values (using ffill() instead of deprecated fillna(method='ffill'))
        feature_df = feature_df.ffill().fillna(0)

        # Check if we have enough data
        if len(feature_df) < self.sequence_length:
            return {
                'timestamp': current_time,
                'price': current_price,
                'signal': 'WAIT',
                'confidence': 0,
                'reason': f'Need {self.sequence_length} candles, have {len(feature_df)}'
            }

        # Prepare input
        features = feature_df.values

        # Normalize
        features = self._normalize_features(features)

        # Replace any remaining NaN/Inf
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

        # Make prediction
        if self._model is not None:
            # ML prediction
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
                prob = self._model(x).item()

            # Determine signal
            if prob > 0.5:
                signal = 'BUY' if prob >= self.min_confidence else 'WEAK_BUY'
                confidence = prob
            else:
                signal = 'SELL' if (1 - prob) >= self.min_confidence else 'WEAK_SELL'
                confidence = 1 - prob

        else:
            # No model - use technical analysis only
            rsi = df_features['rsi_14'].iloc[-1]
            macd_hist = df_features['macd_hist'].iloc[-1]
            bb_position = df_features['bb_position'].iloc[-1]

            # Simple rules-based signal
            bullish_signals = 0
            bearish_signals = 0

            if rsi < 30:
                bullish_signals += 1
            elif rsi > 70:
                bearish_signals += 1

            if macd_hist > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1

            if bb_position < 0.2:
                bullish_signals += 1
            elif bb_position > 0.8:
                bearish_signals += 1

            total = bullish_signals + bearish_signals
            if bullish_signals > bearish_signals:
                signal = 'BUY'
                confidence = bullish_signals / total if total > 0 else 0.5
            elif bearish_signals > bullish_signals:
                signal = 'SELL'
                confidence = bearish_signals / total if total > 0 else 0.5
            else:
                signal = 'NEUTRAL'
                confidence = 0.5

        # Calculate stop loss and take profit levels
        atr = df_features['atr_14'].iloc[-1]

        if signal in ['BUY', 'WEAK_BUY']:
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (4 * atr)  # 2:1 ratio
        elif signal in ['SELL', 'WEAK_SELL']:
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (4 * atr)
        else:
            stop_loss = None
            take_profit = None

        result = {
            'timestamp': current_time,
            'price': current_price,
            'signal': signal,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr,
            'rsi': df_features['rsi_14'].iloc[-1],
            'macd_hist': df_features['macd_hist'].iloc[-1],
            'bb_position': df_features['bb_position'].iloc[-1],
            'using_ml': self._model is not None
        }

        self._last_prediction = result
        return result

    def on_new_data(self, df: pd.DataFrame):
        """
        Callback when new data arrives.
        Called by DataService when new candles are fetched.
        """
        try:
            prediction = self.predict(df)

            # Notify registered callbacks (signal service)
            for callback in self._callbacks:
                try:
                    callback(prediction)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Prediction error: {e}")

    def register_callback(self, callback: Callable):
        """Register callback for new predictions."""
        self._callbacks.append(callback)

    def get_last_prediction(self) -> Optional[dict]:
        """Get most recent prediction."""
        return self._last_prediction

    @property
    def is_ready(self) -> bool:
        """Check if engine is ready for predictions."""
        return True  # Can always run (with or without ML model)

    def get_status(self) -> dict:
        """Get engine status."""
        return {
            'model_loaded': self._model is not None,
            'model_path': str(self.model_path),
            'feature_count': len(self.feature_columns),
            'sequence_length': self.sequence_length,
            'min_confidence': self.min_confidence,
            'last_prediction': self._last_prediction
        }


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = AnalysisEngine()
    engine.load_model()

    print("Engine Status:", engine.get_status())
