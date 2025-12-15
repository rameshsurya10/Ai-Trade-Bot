"""
Tests for Core Module
=====================

Tests configuration, types, and database operations.
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.types import Signal, SignalType, SignalStrength, Candle, Prediction
from src.core.config import Config
from src.core.database import Database


class TestSignalType:
    """Test SignalType enum."""

    def test_values(self):
        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.NEUTRAL.value == "NEUTRAL"


class TestSignalStrength:
    """Test SignalStrength enum."""

    def test_values(self):
        assert SignalStrength.STRONG.value == "STRONG"
        assert SignalStrength.MEDIUM.value == "MEDIUM"
        assert SignalStrength.WEAK.value == "WEAK"


class TestCandle:
    """Test Candle dataclass."""

    def test_creation(self):
        candle = Candle(
            timestamp=1700000000000,
            datetime=datetime(2023, 11, 14, 12, 0, 0),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
        )

        assert candle.timestamp == 1700000000000
        assert candle.open == 50000.0
        assert candle.close == 50500.0

    def test_is_bullish(self):
        bullish = Candle(
            timestamp=1, datetime=datetime.now(),
            open=100, high=110, low=95, close=105, volume=1
        )
        assert bullish.is_bullish is True

        bearish = Candle(
            timestamp=1, datetime=datetime.now(),
            open=105, high=110, low=95, close=100, volume=1
        )
        assert bearish.is_bullish is False

    def test_body_size(self):
        candle = Candle(
            timestamp=1, datetime=datetime.now(),
            open=100, high=110, low=90, close=105, volume=1
        )
        assert candle.body_size == 5

    def test_range_size(self):
        candle = Candle(
            timestamp=1, datetime=datetime.now(),
            open=100, high=110, low=90, close=105, volume=1
        )
        assert candle.range_size == 20

    def test_to_dict(self):
        candle = Candle(
            timestamp=1, datetime=datetime(2023, 1, 1),
            open=100, high=110, low=90, close=105, volume=1
        )
        d = candle.to_dict()
        assert 'timestamp' in d
        assert 'open' in d
        assert 'close' in d


class TestSignal:
    """Test Signal dataclass."""

    def test_creation(self):
        signal = Signal(
            timestamp=datetime.utcnow(),
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.68,
            price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
        )

        assert signal.signal_type == SignalType.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.confidence == 0.68

    def test_confidence_validation(self):
        with pytest.raises(ValueError):
            Signal(confidence=1.5)  # > 1

        with pytest.raises(ValueError):
            Signal(confidence=-0.1)  # < 0

    def test_is_actionable(self):
        # Strong BUY is actionable
        signal = Signal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.7,
        )
        assert signal.is_actionable is True

        # Neutral is not actionable
        neutral = Signal(
            signal_type=SignalType.NEUTRAL,
            strength=SignalStrength.WEAK,
            confidence=0.5,
        )
        assert neutral.is_actionable is False

        # Weak signal is not actionable
        weak = Signal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            confidence=0.5,
        )
        assert weak.is_actionable is False

    def test_risk_reward_ratio(self):
        signal = Signal(
            price=50000.0,
            stop_loss=49000.0,  # Risk: 1000
            take_profit=52000.0,  # Reward: 2000
        )
        assert signal.risk_reward_ratio == 2.0

        # No levels set
        no_levels = Signal(price=50000.0)
        assert no_levels.risk_reward_ratio is None

    def test_format_message(self):
        signal = Signal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.68,
            price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
        )
        msg = signal.format_message()
        assert "BUY" in msg
        assert "STRONG" in msg
        assert "50,000" in msg


class TestPrediction:
    """Test Prediction dataclass."""

    def test_creation(self):
        pred = Prediction(
            timestamp=datetime.utcnow(),
            price=50000.0,
            probability=0.65,
            signal_type=SignalType.BUY,
            confidence=0.65,
        )

        assert pred.probability == 0.65
        assert pred.using_ml is True

    def test_to_dict(self):
        pred = Prediction(
            timestamp=datetime.utcnow(),
            price=50000.0,
            probability=0.65,
            signal_type=SignalType.BUY,
            confidence=0.65,
            rsi=45.0,
        )
        d = pred.to_dict()
        assert d['price'] == 50000.0
        assert d['rsi'] == 45.0


class TestConfig:
    """Test Config loading."""

    def test_default_config(self):
        config = Config()
        assert config.data.symbol == "BTC-USD"
        assert config.analysis.min_confidence == 0.55
        assert config.signals.risk_per_trade == 0.02

    def test_validation(self):
        config = Config()

        # Valid config should pass
        config.validate()  # No exception

        # Invalid confidence
        config.analysis.min_confidence = 1.5
        with pytest.raises(ValueError):
            config.validate()


class TestDatabase:
    """Test Database operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        db = Database(db_path)
        yield db

        # Cleanup
        db.close()
        Path(db_path).unlink(missing_ok=True)

    def test_init_creates_tables(self, temp_db):
        """Test that initialization creates required tables."""
        with temp_db.connection() as conn:
            cursor = conn.cursor()

            # Check candles table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='candles'"
            )
            assert cursor.fetchone() is not None

            # Check signals table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
            )
            assert cursor.fetchone() is not None

    def test_save_and_get_signal(self, temp_db):
        """Test saving and retrieving signals."""
        signal = Signal(
            timestamp=datetime.utcnow(),
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.68,
            price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
        )

        # Save
        signal_id = temp_db.save_signal(signal)
        assert signal_id > 0

        # Retrieve
        signals = temp_db.get_signals(limit=1)
        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY
        assert signals[0].confidence == 0.68

    def test_update_signal_outcome(self, temp_db):
        """Test updating signal outcomes."""
        signal = Signal(
            timestamp=datetime.utcnow(),
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.68,
            price=50000.0,
        )
        signal_id = temp_db.save_signal(signal)

        # Update outcome
        temp_db.update_signal_outcome(
            signal_id=signal_id,
            outcome='WIN',
            outcome_price=52000.0,
            pnl_percent=4.0,
        )

        # Verify
        signals = temp_db.get_signals(limit=1)
        assert signals[0].actual_outcome == 'WIN'
        assert signals[0].pnl_percent == 4.0

    def test_get_performance_stats(self, temp_db):
        """Test performance statistics calculation."""
        # Add some signals with outcomes
        for outcome, pnl in [('WIN', 2.0), ('WIN', 1.5), ('LOSS', -1.0)]:
            signal = Signal(
                timestamp=datetime.utcnow(),
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                confidence=0.6,
                price=50000.0,
                actual_outcome=outcome,
                pnl_percent=pnl,
            )
            temp_db.save_signal(signal)

        stats = temp_db.get_performance_stats()
        assert stats['total_signals'] == 3
        assert stats['resolved_trades'] == 3
        assert stats['winners'] == 2
        assert stats['losers'] == 1
        assert stats['win_rate'] == pytest.approx(2/3, rel=0.01)

    def test_signal_count(self, temp_db):
        """Test signal counting."""
        assert temp_db.get_signal_count() == 0

        temp_db.save_signal(Signal(confidence=0.5))
        assert temp_db.get_signal_count() == 1

        temp_db.save_signal(Signal(confidence=0.5))
        assert temp_db.get_signal_count() == 2


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
