"""
Multi-Symbol Analysis Engine
============================
Runs analysis on multiple symbols simultaneously.

Features:
- Per-symbol analysis with shared model
- Symbol-specific configurations
- Aggregated signal management
- Portfolio-level insights
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import yaml

from .analysis_engine import AnalysisEngine, FeatureCalculator, LSTMModel
from .data.hybrid_data_service import HybridDataService, CandleUpdate

logger = logging.getLogger(__name__)


@dataclass
class SymbolConfig:
    """Configuration for a single symbol."""
    symbol: str
    enabled: bool = True
    min_confidence: float = 0.55
    risk_multiplier: float = 1.0  # Adjust position size relative to default

    @classmethod
    def from_dict(cls, data: dict) -> 'SymbolConfig':
        return cls(
            symbol=data['symbol'],
            enabled=data.get('enabled', True),
            min_confidence=data.get('min_confidence', 0.55),
            risk_multiplier=data.get('risk_multiplier', 1.0),
        )


@dataclass
class SymbolPrediction:
    """Prediction result for a single symbol."""
    symbol: str
    timestamp: datetime
    price: float
    signal: str
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    atr: Optional[float] = None
    rsi: Optional[float] = None
    macd_hist: Optional[float] = None
    using_ml: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'price': self.price,
            'signal': self.signal,
            'confidence': self.confidence,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'atr': self.atr,
            'rsi': self.rsi,
            'macd_hist': self.macd_hist,
            'using_ml': self.using_ml,
        }


class MultiSymbolEngine:
    """
    Multi-symbol analysis engine.

    Manages analysis for multiple trading pairs simultaneously
    using a shared ML model and per-symbol configurations.

    Usage:
        engine = MultiSymbolEngine.from_config("config.yaml")
        engine.on_signal(lambda pred: print(f"{pred.symbol}: {pred.signal}"))
        engine.start(data_service)
    """

    def __init__(
        self,
        symbols: List[SymbolConfig],
        model_path: str,
        sequence_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        default_min_confidence: float = 0.55,
    ):
        """Initialize multi-symbol engine."""
        self.symbols = {s.symbol: s for s in symbols}
        self.model_path = Path(model_path)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.default_min_confidence = default_min_confidence

        # Shared model
        self._model: Optional[LSTMModel] = None
        self._feature_means = None
        self._feature_stds = None

        # Per-symbol data buffers
        self._data_buffers: Dict[str, pd.DataFrame] = {}

        # Per-symbol last predictions
        self._predictions: Dict[str, SymbolPrediction] = {}

        # Callbacks
        self._signal_callbacks: List[Callable[[SymbolPrediction], None]] = []

        # State
        self._running = False
        self._lock = threading.Lock()

        self.feature_columns = FeatureCalculator.get_feature_columns()

        logger.info(f"MultiSymbolEngine initialized with {len(symbols)} symbols")

    @classmethod
    def from_config(cls, config_path: str) -> 'MultiSymbolEngine':
        """Create engine from config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        data_config = config.get('data', {})
        model_config = config.get('model', {})
        analysis_config = config.get('analysis', {})

        # Build symbol list
        symbols_config = data_config.get('symbols', [])
        if not symbols_config:
            # Fallback to single symbol
            symbol = data_config.get('symbol', 'BTC/USDT')
            symbols_config = [{'symbol': symbol}]

        # Parse symbol configs
        symbols = []
        for s in symbols_config:
            if isinstance(s, str):
                symbols.append(SymbolConfig(symbol=s))
            else:
                symbols.append(SymbolConfig.from_dict(s))

        return cls(
            symbols=symbols,
            model_path=model_config.get('path', 'models/best_model.pt'),
            sequence_length=model_config.get('sequence_length', 60),
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2),
            default_min_confidence=analysis_config.get('min_confidence', 0.55),
        )

    def load_model(self) -> bool:
        """Load shared ML model."""
        import torch

        if not self.model_path.exists():
            logger.warning(f"Model not found: {self.model_path}")
            logger.info("Running in FEATURE-ONLY mode")
            return False

        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')

            input_size = len(self.feature_columns)
            self._model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.eval()

            if 'feature_means' in checkpoint:
                self._feature_means = checkpoint['feature_means']
                self._feature_stds = checkpoint['feature_stds']

            logger.info(f"Model loaded: {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Model load error: {e}")
            return False

    def on_signal(self, callback: Callable[[SymbolPrediction], None]):
        """Register callback for signal predictions."""
        self._signal_callbacks.append(callback)

    def _notify_signal(self, prediction: SymbolPrediction):
        """Notify callbacks of new signal."""
        for callback in self._signal_callbacks:
            try:
                callback(prediction)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    def _normalize_features(self, features) -> Any:
        """Normalize features using z-score."""
        import numpy as np

        if self._feature_means is not None:
            return (features - self._feature_means) / (self._feature_stds + 1e-8)

        means = np.nanmean(features, axis=0)
        stds = np.nanstd(features, axis=0)
        return (features - means) / (stds + 1e-8)

    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[SymbolPrediction]:
        """Run analysis on a single symbol."""
        import torch
        import numpy as np

        if df.empty or len(df) < self.sequence_length:
            logger.debug(f"{symbol}: Insufficient data ({len(df)} candles)")
            return None

        symbol_config = self.symbols.get(symbol)
        if symbol_config and not symbol_config.enabled:
            return None

        min_confidence = (
            symbol_config.min_confidence if symbol_config
            else self.default_min_confidence
        )

        try:
            # Calculate features
            df_features = FeatureCalculator.calculate_all(df)

            current_price = df['close'].iloc[-1]
            current_time = df['datetime'].iloc[-1] if 'datetime' in df else datetime.utcnow()

            # Get feature values
            feature_df = df_features[self.feature_columns].iloc[-self.sequence_length:]
            feature_df = feature_df.ffill().fillna(0)

            features = feature_df.values
            features = self._normalize_features(features)
            features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

            # Make prediction
            if self._model is not None:
                with torch.no_grad():
                    x = torch.FloatTensor(features).unsqueeze(0)
                    prob = self._model(x).item()

                if prob > 0.5:
                    signal = 'BUY' if prob >= min_confidence else 'WEAK_BUY'
                    confidence = prob
                else:
                    signal = 'SELL' if (1 - prob) >= min_confidence else 'WEAK_SELL'
                    confidence = 1 - prob
            else:
                # Technical analysis fallback
                rsi = df_features['rsi_14'].iloc[-1]
                macd_hist = df_features['macd_hist'].iloc[-1]
                bb_position = df_features['bb_position'].iloc[-1]

                bullish = sum([rsi < 30, macd_hist > 0, bb_position < 0.2])
                bearish = sum([rsi > 70, macd_hist < 0, bb_position > 0.8])

                if bullish > bearish:
                    signal = 'BUY'
                    confidence = bullish / 3
                elif bearish > bullish:
                    signal = 'SELL'
                    confidence = bearish / 3
                else:
                    signal = 'NEUTRAL'
                    confidence = 0.5

            # Calculate levels
            atr = df_features['atr_14'].iloc[-1]

            if signal in ['BUY', 'WEAK_BUY']:
                stop_loss = current_price - (2 * atr)
                take_profit = current_price + (4 * atr)
            elif signal in ['SELL', 'WEAK_SELL']:
                stop_loss = current_price + (2 * atr)
                take_profit = current_price - (4 * atr)
            else:
                stop_loss = None
                take_profit = None

            prediction = SymbolPrediction(
                symbol=symbol,
                timestamp=current_time,
                price=current_price,
                signal=signal,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr=atr,
                rsi=df_features['rsi_14'].iloc[-1],
                macd_hist=df_features['macd_hist'].iloc[-1],
                using_ml=self._model is not None,
            )

            with self._lock:
                self._predictions[symbol] = prediction

            return prediction

        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            return None

    def handle_candle(self, symbol: str, candle: CandleUpdate):
        """Handle new candle from data service."""
        # Normalize symbol format
        clean_symbol = symbol.replace("-", "/").upper()
        if "/" not in clean_symbol:
            # Add /USDT if not present
            if clean_symbol.endswith("USDT"):
                clean_symbol = clean_symbol[:-4] + "/USDT"
            elif clean_symbol.endswith("USD"):
                clean_symbol = clean_symbol[:-3] + "/USD"

        with self._lock:
            if clean_symbol not in self._data_buffers:
                self._data_buffers[clean_symbol] = pd.DataFrame()

            # Convert candle to dataframe row
            new_row = pd.DataFrame([{
                'timestamp': candle.timestamp,
                'datetime': datetime.fromtimestamp(candle.timestamp / 1000),
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
            }])

            # Append or update
            df = self._data_buffers[clean_symbol]
            if not df.empty and df['timestamp'].iloc[-1] == candle.timestamp:
                # Update last row
                df.iloc[-1] = new_row.iloc[0]
            else:
                df = pd.concat([df, new_row], ignore_index=True)

            # Keep only last N candles for memory efficiency
            max_candles = self.sequence_length * 3
            if len(df) > max_candles:
                df = df.iloc[-max_candles:]

            self._data_buffers[clean_symbol] = df

        # Analyze on closed candles only
        if candle.is_closed:
            prediction = self.analyze_symbol(clean_symbol, df)
            if prediction and prediction.signal not in ['NEUTRAL', 'WAIT']:
                self._notify_signal(prediction)

    def start(self, data_service: HybridDataService):
        """Start analysis with data service."""
        self._running = True
        self.load_model()

        # Register for candle updates
        data_service.on_candle(self.handle_candle)

        logger.info("MultiSymbolEngine started")

    def stop(self):
        """Stop analysis."""
        self._running = False
        logger.info("MultiSymbolEngine stopped")

    def get_prediction(self, symbol: str) -> Optional[SymbolPrediction]:
        """Get latest prediction for symbol."""
        with self._lock:
            return self._predictions.get(symbol)

    def get_all_predictions(self) -> Dict[str, SymbolPrediction]:
        """Get all latest predictions."""
        with self._lock:
            return dict(self._predictions)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio-level analysis summary."""
        with self._lock:
            predictions = list(self._predictions.values())

        if not predictions:
            return {
                'symbols_analyzed': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'neutral_signals': 0,
                'avg_confidence': 0,
                'strongest_buy': None,
                'strongest_sell': None,
            }

        buy_signals = [p for p in predictions if 'BUY' in p.signal]
        sell_signals = [p for p in predictions if 'SELL' in p.signal]
        neutral_signals = [p for p in predictions if p.signal in ['NEUTRAL', 'WAIT']]

        strongest_buy = max(buy_signals, key=lambda p: p.confidence) if buy_signals else None
        strongest_sell = max(sell_signals, key=lambda p: p.confidence) if sell_signals else None

        return {
            'symbols_analyzed': len(predictions),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'neutral_signals': len(neutral_signals),
            'avg_confidence': sum(p.confidence for p in predictions) / len(predictions),
            'strongest_buy': strongest_buy.to_dict() if strongest_buy else None,
            'strongest_sell': strongest_sell.to_dict() if strongest_sell else None,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            'running': self._running,
            'model_loaded': self._model is not None,
            'symbols': list(self.symbols.keys()),
            'predictions_count': len(self._predictions),
            'data_buffers': {s: len(df) for s, df in self._data_buffers.items()},
        }


# CLI usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    from .data import HybridDataService

    engine = MultiSymbolEngine.from_config("config.yaml")
    service = HybridDataService.from_config("config.yaml")

    def on_signal(pred: SymbolPrediction):
        print(f"[{pred.signal}] {pred.symbol}: ${pred.price:.2f} "
              f"(conf: {pred.confidence:.1%})")

    engine.on_signal(on_signal)
    engine.start(service)
    service.start()

    try:
        while service.is_running:
            time.sleep(10)
            summary = engine.get_portfolio_summary()
            print(f"\nPortfolio: {summary['buy_signals']} buys, "
                  f"{summary['sell_signals']} sells")
    except KeyboardInterrupt:
        print("\nStopping...")
        engine.stop()
        service.stop()
