"""
Integration Tests
=================
End-to-end tests for the trading system components.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAnalysisEngine:
    """Test analysis engine components."""

    def test_feature_calculator_import(self):
        """Test FeatureCalculator can be imported."""
        from src.analysis_engine import FeatureCalculator
        assert FeatureCalculator is not None

    def test_feature_calculator_columns(self):
        """Test feature columns are defined."""
        from src.analysis_engine import FeatureCalculator
        columns = FeatureCalculator.get_feature_columns()
        assert len(columns) > 40  # Should have 40+ features
        assert 'rsi_14' in columns
        assert 'macd' in columns
        assert 'atr_14' in columns

    def test_feature_calculation(self):
        """Test feature calculation on sample data."""
        from src.analysis_engine import FeatureCalculator

        # Create sample OHLCV data
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='1h')
        df = pd.DataFrame({
            'open': 100 + np.random.randn(n).cumsum(),
            'high': 101 + np.random.randn(n).cumsum(),
            'low': 99 + np.random.randn(n).cumsum(),
            'close': 100 + np.random.randn(n).cumsum(),
            'volume': np.random.uniform(1000, 10000, n)
        }, index=dates)

        # Fix high/low consistency
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        # Calculate features
        features_df = FeatureCalculator.calculate_all(df)

        # Verify features were added
        assert len(features_df.columns) > len(df.columns)
        assert 'rsi_14' in features_df.columns
        assert not features_df['rsi_14'].isna().all()


class TestAdvancedPredictor:
    """Test advanced predictor algorithms."""

    def test_advanced_predictor_import(self):
        """Test AdvancedPredictor can be imported."""
        from src.advanced_predictor import (
            AdvancedPredictor,
            FourierAnalyzer,
            KalmanFilter,
            EntropyAnalyzer,
            MarkovChain,
            MonteCarlo
        )
        assert AdvancedPredictor is not None
        assert FourierAnalyzer is not None
        assert KalmanFilter is not None
        assert EntropyAnalyzer is not None
        assert MarkovChain is not None
        assert MonteCarlo is not None

    def test_fourier_analyzer(self):
        """Test Fourier analysis."""
        from src.advanced_predictor import FourierAnalyzer

        # Create oscillating price data
        x = np.linspace(0, 4 * np.pi, 100)
        prices = 100 + 10 * np.sin(x) + np.random.randn(100)

        analyzer = FourierAnalyzer()
        result = analyzer.analyze(prices)

        assert 'signal' in result
        assert 'dominant_period' in result
        assert 'cycle_phase' in result
        assert result['signal'] in ['BULLISH', 'BEARISH', 'NEUTRAL']

    def test_kalman_filter(self):
        """Test Kalman filter."""
        from src.advanced_predictor import KalmanFilter

        # Create noisy price data with trend
        prices = 100 + np.arange(50) * 0.5 + np.random.randn(50) * 2

        kf = KalmanFilter()
        result = kf.filter(prices)

        assert 'smoothed_price' in result
        assert 'trend' in result
        assert result['trend'] in ['UP', 'DOWN', 'SIDEWAYS']
        # Smoothed price should be close to raw price
        assert abs(result['smoothed_price'] - prices[-1]) < 10

    def test_entropy_analyzer(self):
        """Test entropy analysis."""
        from src.advanced_predictor import EntropyAnalyzer

        # Create returns data
        returns = np.random.randn(100) * 0.02  # 2% daily volatility

        analyzer = EntropyAnalyzer()
        result = analyzer.analyze(returns)

        assert 'entropy' in result
        assert 'regime' in result
        assert result['regime'] in ['TRENDING', 'NORMAL', 'CHOPPY', 'VOLATILE']
        assert 0 <= result['normalized_entropy'] <= 1

    def test_markov_chain(self):
        """Test Markov chain analysis."""
        from src.advanced_predictor import MarkovChain

        # Create returns
        returns = np.random.randn(100) * 0.02

        mc = MarkovChain()
        result = mc.analyze(returns)

        assert 'current_state' in result
        assert 'prob_up' in result
        assert 'prob_down' in result
        assert 0 <= result['prob_up'] <= 1
        assert 0 <= result['prob_down'] <= 1

    def test_monte_carlo(self):
        """Test Monte Carlo simulation."""
        from src.advanced_predictor import MonteCarlo

        returns = np.random.randn(100) * 0.02
        current_price = 100

        mc = MonteCarlo(n_simulations=100)  # Reduced for speed
        result = mc.simulate(current_price, returns)

        assert 'expected_return' in result
        assert 'prob_profit' in result
        assert 'risk_score' in result
        assert 0 <= result['prob_profit'] <= 1
        assert 0 <= result['risk_score'] <= 1

    def test_advanced_predictor_ensemble(self):
        """Test full ensemble prediction."""
        from src.advanced_predictor import AdvancedPredictor

        # Create sample DataFrame
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'open': 100 + np.random.randn(n).cumsum(),
            'high': 101 + np.random.randn(n).cumsum(),
            'low': 99 + np.random.randn(n).cumsum(),
            'close': 100 + np.random.randn(n).cumsum(),
            'volume': np.random.uniform(1000, 10000, n)
        })
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        predictor = AdvancedPredictor()
        result = predictor.predict(df, lstm_probability=0.6, atr=2.0)

        assert result.direction in ['BUY', 'SELL', 'NEUTRAL']
        assert 0 <= result.confidence <= 1
        assert result.stop_loss > 0
        assert result.take_profit > 0


class TestPortfolioComponents:
    """Test portfolio management components."""

    def test_position_sizers_import(self):
        """Test position sizers can be imported."""
        from src.portfolio.sizing import (
            EqualWeightSizer,
            RiskParitySizer,
            KellyCriterionSizer,
            VolatilityTargetSizer
        )
        assert EqualWeightSizer is not None
        assert RiskParitySizer is not None
        assert KellyCriterionSizer is not None
        assert VolatilityTargetSizer is not None

    def test_equal_weight_sizer(self):
        """Test equal weight position sizing."""
        from src.portfolio.sizing import EqualWeightSizer
        from unittest.mock import MagicMock

        sizer = EqualWeightSizer(max_positions=10, reserve_cash=0.05)

        # Mock portfolio
        portfolio = MagicMock()
        portfolio.total_value = 100000

        result = sizer.calculate_size(portfolio, 'BTC/USD', price=50000)

        assert result.symbol == 'BTC/USD'
        assert result.quantity > 0
        assert result.weight == 0.1  # 1/10 positions
        assert result.value > 0

    def test_kelly_criterion(self):
        """Test Kelly criterion calculation."""
        from src.portfolio.sizing import KellyCriterionSizer

        sizer = KellyCriterionSizer(fraction=0.5)

        # Win rate 60%, win/loss ratio 2:1
        kelly = sizer.calculate_kelly(
            win_rate=0.6,
            avg_win=0.04,  # 4% avg win
            avg_loss=0.02  # 2% avg loss
        )

        # Kelly should be positive with edge
        assert kelly > 0


class TestRiskManagement:
    """Test risk management components."""

    def test_risk_models_import(self):
        """Test risk models can be imported."""
        from src.portfolio.risk import (
            MaximumDrawdownRisk,
            MaximumPositionSizeRisk,
            RiskManager
        )
        assert MaximumDrawdownRisk is not None
        assert MaximumPositionSizeRisk is not None
        assert RiskManager is not None

    def test_max_drawdown_risk(self):
        """Test max drawdown risk check."""
        from src.portfolio.risk import MaximumDrawdownRisk
        from unittest.mock import MagicMock

        risk = MaximumDrawdownRisk(max_drawdown=0.20)

        # Mock portfolio under limit
        portfolio = MagicMock()
        portfolio.current_drawdown = 0.10  # 10% drawdown

        result = risk.check(portfolio)
        assert result['passed'] is True

        # Mock portfolio over limit
        portfolio.current_drawdown = 0.25  # 25% drawdown
        result = risk.check(portfolio)
        assert result['passed'] is False


class TestUniverseFilters:
    """Test universe filter components."""

    def test_filters_import(self):
        """Test filters can be imported."""
        from src.universe.filters import (
            VolumeFilter,
            PriceFilter,
            VolatilityFilter,
            MomentumFilter,
            CompositeFilter
        )
        assert VolumeFilter is not None
        assert PriceFilter is not None
        assert VolatilityFilter is not None
        assert MomentumFilter is not None
        assert CompositeFilter is not None

    def test_volume_filter(self):
        """Test volume filtering."""
        from src.universe.filters import VolumeFilter
        from src.universe.manager import SecurityInfo
        from datetime import datetime

        filter_ = VolumeFilter(min_volume=1000000)

        # Create test securities
        candidates = [
            SecurityInfo('BTC/USD', 50000, datetime.now(), volume_24h=5000000),
            SecurityInfo('SHIB/USD', 0.00001, datetime.now(), volume_24h=100000),
            SecurityInfo('ETH/USD', 3000, datetime.now(), volume_24h=2000000),
        ]

        result = filter_.apply(candidates)

        # Should filter out low volume
        assert len(result) == 2
        symbols = [s.symbol for s in result]
        assert 'SHIB/USD' not in symbols

    def test_composite_filter(self):
        """Test composite filter pipeline."""
        from src.universe.filters import (
            CompositeFilter,
            VolumeFilter,
            PriceFilter
        )
        from src.universe.manager import SecurityInfo
        from datetime import datetime

        # Build filter pipeline
        pipeline = (CompositeFilter("TestPipeline")
                   .add(VolumeFilter(min_volume=100000))
                   .add(PriceFilter(min_price=1.0)))

        candidates = [
            SecurityInfo('BTC/USD', 50000, datetime.now(), volume_24h=5000000),
            SecurityInfo('SHIB/USD', 0.00001, datetime.now(), volume_24h=500000),
            SecurityInfo('LOW/USD', 2.0, datetime.now(), volume_24h=50000),
        ]

        result = pipeline.apply(candidates)

        # Only BTC passes both filters
        assert len(result) == 1
        assert result[0].symbol == 'BTC/USD'


class TestBrokerageAbstraction:
    """Test brokerage abstraction layer."""

    def test_brokerage_base_import(self):
        """Test brokerage base classes can be imported."""
        from src.brokerages.base import BaseBrokerage, CashBalance, Position
        from src.brokerages.orders import Order, OrderType, OrderSide, OrderStatus

        assert BaseBrokerage is not None
        assert Order is not None
        assert OrderType is not None
        assert OrderSide is not None

    def test_order_creation(self):
        """Test order object creation."""
        from src.brokerages.orders import Order, OrderType, OrderSide

        order = Order(
            symbol='BTC/USD',
            quantity=0.1,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY
        )

        assert order.symbol == 'BTC/USD'
        assert order.quantity == 0.1
        assert order.order_type == OrderType.MARKET
        assert order.side == OrderSide.BUY


class TestDatabase:
    """Test database operations."""

    def test_database_import(self):
        """Test database can be imported."""
        from src.core.database import Database
        assert Database is not None

    def test_database_creation(self):
        """Test database can be created."""
        from src.core.database import Database

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            db = Database(db_path)
            assert db is not None

            # Test basic operations
            db.save_candle(
                symbol='BTC/USD',
                timestamp=datetime.now(),
                open_=100, high=101, low=99, close=100.5,
                volume=1000
            )

            candles = db.get_candles('BTC/USD', limit=10)
            assert len(candles) > 0

            db.close()
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestLiveTradingRunner:
    """Test live trading runner."""

    def test_runner_import(self):
        """Test runner can be imported."""
        from src.live_trading import LiveTradingRunner, TradingMode, RunnerStatus
        assert LiveTradingRunner is not None
        assert TradingMode is not None
        assert RunnerStatus is not None

    def test_trading_modes(self):
        """Test trading mode enum."""
        from src.live_trading import TradingMode

        assert TradingMode.PAPER.value == 'paper'
        assert TradingMode.LIVE.value == 'live'
        assert TradingMode.BACKTEST.value == 'backtest'


class TestConfigLoading:
    """Test configuration loading."""

    def test_config_exists(self):
        """Test config.yaml exists."""
        config_path = Path(__file__).parent.parent / 'config.yaml'
        assert config_path.exists()

    def test_config_parsing(self):
        """Test config can be parsed."""
        import yaml

        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check required sections
        assert 'data' in config
        assert 'model' in config
        assert 'signals' in config
        assert 'portfolio' in config
        assert 'risk' in config

        # Check values
        assert config['data']['symbol'] == 'BTC/USDT'
        assert config['portfolio']['initial_capital'] == 10000
        assert config['risk']['max_drawdown_percent'] == 20.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
