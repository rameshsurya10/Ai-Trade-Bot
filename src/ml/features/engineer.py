"""
Feature Engineering Pipeline
=============================

Comprehensive feature engineering for the trading system.

Feature Categories:
1. Price-based: Returns, log returns, price ratios
2. Technical: RSI, MACD, BB, ATR, ADX, Stochastic
3. Volume: Volume ratios, OBV
4. Volatility: Historical vol, ATR, BB width
5. Decomposition: SVMD IMFs
6. Regime: HMM state probabilities
7. Lagged: Past values of key features

Research shows:
- Top 10-15 features often outperform using all features
- SHAP can reduce features while improving precision/recall
- Feature selection is critical for avoiding overfitting
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for engineered features."""
    features: pd.DataFrame
    feature_names: List[str]
    sequence_data: Optional[np.ndarray] = None  # For LSTM (samples, seq_len, features)
    tabular_data: Optional[np.ndarray] = None  # For boosting (samples, features)
    targets: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class FeatureEngineer:
    """
    Complete Feature Engineering Pipeline.

    Handles:
    1. Technical indicator calculation
    2. SVMD decomposition integration
    3. Feature normalization
    4. Sequence creation for LSTM
    5. Feature selection
    """

    def __init__(
        self,
        sequence_length: int = 60,
        include_svmd: bool = True,
        include_regime: bool = True,
        n_lags: int = 5
    ):
        """
        Initialize feature engineer.

        Args:
            sequence_length: Sequence length for LSTM
            include_svmd: Whether to include SVMD decomposition
            include_regime: Whether to include regime features
            n_lags: Number of lagged features to create
        """
        self.sequence_length = sequence_length
        self.include_svmd = include_svmd
        self.include_regime = include_regime
        self.n_lags = n_lags

        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._feature_names: List[str] = []
        self._is_fitted = False

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical features
        """
        df = df.copy()

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # =====================================================================
        # PRICE-BASED FEATURES
        # =====================================================================
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        for period in [7, 14, 21, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Price ratios to MAs
        df['price_sma_7_ratio'] = df['close'] / df['sma_7']
        df['price_sma_21_ratio'] = df['close'] / df['sma_21']
        df['price_sma_50_ratio'] = df['close'] / df['sma_50']
        df['price_ema_21_ratio'] = df['close'] / df['ema_21']

        # =====================================================================
        # VOLATILITY FEATURES
        # =====================================================================

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_7'] = tr.rolling(7).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # Historical volatility
        df['volatility_7'] = df['returns'].rolling(7).std() * np.sqrt(252)
        df['volatility_14'] = df['returns'].rolling(14).std() * np.sqrt(252)
        df['volatility_30'] = df['returns'].rolling(30).std() * np.sqrt(252)

        # =====================================================================
        # MOMENTUM FEATURES
        # =====================================================================

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_7'] = self._calculate_rsi(df['close'], 7)

        # Stochastic Oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ROC (Rate of Change)
        df['roc_5'] = df['close'].pct_change(5) * 100
        df['roc_10'] = df['close'].pct_change(10) * 100
        df['roc_20'] = df['close'].pct_change(20) * 100

        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)

        # CCI (Commodity Channel Index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (typical_price - typical_price.rolling(20).mean()) / \
                    (0.015 * typical_price.rolling(20).std() + 1e-10)

        # =====================================================================
        # VOLUME FEATURES
        # =====================================================================
        if 'volume' in df.columns:
            df['volume_sma_14'] = df['volume'].rolling(14).mean()
            df['volume_sma_7'] = df['volume'].rolling(7).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma_14'] + 1e-10)
            df['volume_change'] = df['volume'].pct_change()

            # OBV (On Balance Volume)
            volume_direction = np.sign(df['close'].diff())
            df['obv'] = (df['volume'] * volume_direction).fillna(0).cumsum()
            df['obv_sma'] = df['obv'].rolling(14).mean()

            # Money Flow
            mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
                           (df['high'] - df['low'] + 1e-10)
            df['money_flow'] = mf_multiplier * df['volume']

        # =====================================================================
        # TREND FEATURES
        # =====================================================================

        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['di_diff'] = plus_di - minus_di

        # Trend strength
        df['trend_7'] = (df['close'] - df['close'].shift(7)) / (df['close'].shift(7) + 1e-10)
        df['trend_14'] = (df['close'] - df['close'].shift(14)) / (df['close'].shift(14) + 1e-10)
        df['trend_21'] = (df['close'] - df['close'].shift(21)) / (df['close'].shift(21) + 1e-10)

        # =====================================================================
        # PATTERN FEATURES
        # =====================================================================
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_wick_upper'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['candle_wick_lower'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_body_ratio'] = df['candle_body'] / (df['high'] - df['low'] + 1e-10)

        # Higher highs / lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_close'] = (df['close'] > df['close'].shift(1)).astype(int)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI for given period."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def add_lagged_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n_lags: int = None
    ) -> pd.DataFrame:
        """Add lagged versions of specified columns."""
        if n_lags is None:
            n_lags = self.n_lags

        df = df.copy()

        for col in columns:
            if col in df.columns:
                for lag in range(1, n_lags + 1):
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return df

    def add_svmd_features(
        self,
        df: pd.DataFrame,
        column: str = 'close'
    ) -> pd.DataFrame:
        """Add SVMD decomposition features."""
        if not self.include_svmd:
            return df

        try:
            from ..decomposition import SVMDDecomposer

            decomposer = SVMDDecomposer(max_modes=5)
            df_with_imfs, result = decomposer.decompose_dataframe(df, column)

            logger.info(f"Added {result.n_modes} SVMD modes")
            return df_with_imfs

        except Exception as e:
            logger.warning(f"SVMD decomposition failed: {e}")
            return df

    def add_regime_features(
        self,
        df: pd.DataFrame,
        regime_detector=None
    ) -> pd.DataFrame:
        """Add regime detection features."""
        if not self.include_regime or regime_detector is None:
            return df

        try:
            result = regime_detector.detect(df)

            df = df.copy()
            for regime, prob in result.regime_probabilities.items():
                df[f'regime_prob_{regime.name.lower()}'] = prob

            df['regime_current'] = result.current_regime.value

            return df

        except Exception as e:
            logger.warning(f"Regime feature calculation failed: {e}")
            return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        base_features = [
            # Price
            'returns', 'log_returns',
            'price_sma_7_ratio', 'price_sma_21_ratio', 'price_sma_50_ratio',

            # Volatility
            'atr_14', 'atr_7', 'bb_width', 'bb_position',
            'volatility_7', 'volatility_14',

            # Momentum
            'rsi_14', 'rsi_7', 'stoch_k', 'stoch_d',
            'macd', 'macd_signal', 'macd_hist',
            'roc_5', 'roc_10', 'williams_r', 'cci',

            # Volume
            'volume_ratio', 'volume_change',

            # Trend
            'adx', 'plus_di', 'minus_di', 'di_diff',
            'trend_7', 'trend_14',

            # Pattern
            'candle_body_ratio', 'higher_high', 'lower_low', 'higher_close'
        ]

        return base_features

    def fit_transform(
        self,
        df: pd.DataFrame,
        regime_detector=None
    ) -> FeatureSet:
        """
        Fit feature engineer and transform data.

        Args:
            df: DataFrame with OHLCV data
            regime_detector: Optional regime detector for regime features

        Returns:
            FeatureSet with all features
        """
        # Calculate technical features
        df_features = self.calculate_technical_features(df)

        # Add SVMD features
        if self.include_svmd:
            df_features = self.add_svmd_features(df_features)

        # Add regime features
        if self.include_regime and regime_detector is not None:
            df_features = self.add_regime_features(df_features, regime_detector)

        # Add lagged features for key indicators
        lag_columns = ['rsi_14', 'macd_hist', 'bb_position', 'returns']
        df_features = self.add_lagged_features(df_features, lag_columns)

        # Get feature columns
        feature_cols = [col for col in self.get_feature_columns() if col in df_features.columns]

        # Add SVMD columns if present
        imf_cols = [col for col in df_features.columns if col.startswith('imf_')]
        feature_cols.extend(imf_cols)

        # Add regime columns if present
        regime_cols = [col for col in df_features.columns if col.startswith('regime_')]
        feature_cols.extend(regime_cols)

        # Add lag columns
        lag_cols = [col for col in df_features.columns if '_lag_' in col]
        feature_cols.extend(lag_cols)

        self._feature_names = feature_cols

        # Get feature matrix
        features = df_features[feature_cols].values

        # Create target (next candle direction)
        targets = (df_features['close'].shift(-1) > df_features['close']).astype(int).values

        # Handle NaN values
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

        # Fit normalization
        self._feature_means = np.nanmean(features, axis=0)
        self._feature_stds = np.nanstd(features, axis=0) + 1e-8

        # Normalize
        features_normalized = (features - self._feature_means) / self._feature_stds

        # Create sequences for LSTM
        # Exclude the last sample (no valid target for it due to shift(-1))
        sequence_data = self._create_sequences(features_normalized[:-1])

        # Tabular data for boosting (use last point features)
        # Note: sequence_data has len(features) - sequence_length - 1 samples
        # Tabular and targets need to match this length
        tabular_data = features_normalized[self.sequence_length:-1]  # Drop first seq_len and last
        targets = targets[self.sequence_length:-1]  # Align with sequences (drop last as shift(-1) creates NaN)

        self._is_fitted = True

        return FeatureSet(
            features=df_features,
            feature_names=feature_cols,
            sequence_data=sequence_data,
            tabular_data=tabular_data,
            targets=targets,
            metadata={
                'n_features': len(feature_cols),
                'n_samples': len(tabular_data),
                'sequence_length': self.sequence_length
            }
        )

    def transform(
        self,
        df: pd.DataFrame,
        regime_detector=None
    ) -> FeatureSet:
        """
        Transform new data using fitted parameters.

        Args:
            df: DataFrame with OHLCV data
            regime_detector: Optional regime detector

        Returns:
            FeatureSet with transformed features
        """
        if not self._is_fitted:
            return self.fit_transform(df, regime_detector)

        # Calculate features
        df_features = self.calculate_technical_features(df)

        if self.include_svmd:
            df_features = self.add_svmd_features(df_features)

        if self.include_regime and regime_detector is not None:
            df_features = self.add_regime_features(df_features, regime_detector)

        lag_columns = ['rsi_14', 'macd_hist', 'bb_position', 'returns']
        df_features = self.add_lagged_features(df_features, lag_columns)

        # Use fitted feature names
        feature_cols = [col for col in self._feature_names if col in df_features.columns]
        features = df_features[feature_cols].values

        # Handle NaN
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

        # Normalize using fitted parameters
        features_normalized = (features - self._feature_means[:len(feature_cols)]) / \
                             self._feature_stds[:len(feature_cols)]

        # Create sequences
        sequence_data = self._create_sequences(features_normalized)
        tabular_data = features_normalized[self.sequence_length:]

        return FeatureSet(
            features=df_features,
            feature_names=feature_cols,
            sequence_data=sequence_data,
            tabular_data=tabular_data,
            targets=None,
            metadata={'n_features': len(feature_cols)}
        )

    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM from feature matrix."""
        sequences = []
        for i in range(self.sequence_length, len(features)):
            sequences.append(features[i-self.sequence_length:i])

        return np.array(sequences) if sequences else np.array([])

    def get_feature_importance(
        self,
        model,
        feature_names: List[str] = None
    ) -> Dict[str, float]:
        """Get feature importance from fitted model."""
        if feature_names is None:
            feature_names = self._feature_names

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(feature_names, importances))

        return {}
