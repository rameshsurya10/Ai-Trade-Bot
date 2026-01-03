"""
Test multi-timeframe data provider functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import logging
from src.data.provider import UnifiedDataProvider, Candle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_multi_timeframe_subscription():
    """Test subscribing to multiple intervals for same symbol."""

    provider = UnifiedDataProvider.get_instance()

    # Test 1: Subscribe to BTC/USDT with multiple intervals
    logger.info("Test 1: Multi-interval subscription")
    provider.subscribe('BTC/USDT', exchange='binance', interval='1h')
    provider.subscribe('BTC/USDT', exchange='binance', interval='4h')
    provider.subscribe('BTC/USDT', exchange='binance', interval='1d')

    # Verify subscriptions
    subs = provider.get_subscriptions()
    logger.info(f"Subscriptions: {subs}")

    assert ('BTC/USDT', '1h') in subs, "1h subscription missing"
    assert ('BTC/USDT', '4h') in subs, "4h subscription missing"
    assert ('BTC/USDT', '1d') in subs, "1d subscription missing"
    logger.info("✓ All subscriptions registered correctly")

    # Test 2: Candle callback with interval
    logger.info("\nTest 2: Candle callback receives interval")

    received_candles = []

    def on_candle(candle: Candle, interval: str):
        """Callback that receives interval."""
        logger.info(f"Received closed candle: {candle.symbol} @ {interval}")
        received_candles.append((candle, interval))

    provider.on_candle(on_candle)

    # Test 3: Get candles for specific interval
    logger.info("\nTest 3: Get candles by interval")

    # Note: Since we just started, buffers might be empty
    # In real usage, WebSocket would populate these

    df_1h = provider.get_candles('BTC/USDT', interval='1h', limit=10)
    df_4h = provider.get_candles('BTC/USDT', interval='4h', limit=10)
    df_1d = provider.get_candles('BTC/USDT', interval='1d', limit=10)

    logger.info(f"1h candles: {len(df_1h)} rows")
    logger.info(f"4h candles: {len(df_4h)} rows")
    logger.info(f"1d candles: {len(df_1d)} rows")
    logger.info("✓ get_candles works with interval parameter")

    # Test 4: Unsubscribe specific interval
    logger.info("\nTest 4: Unsubscribe specific interval")

    provider.unsubscribe('BTC/USDT', interval='1d')
    subs = provider.get_subscriptions()

    assert ('BTC/USDT', '1d') not in subs, "1d should be unsubscribed"
    assert ('BTC/USDT', '1h') in subs, "1h should still be subscribed"
    assert ('BTC/USDT', '4h') in subs, "4h should still be subscribed"
    logger.info("✓ Selective unsubscribe works")

    # Test 5: Unsubscribe all intervals for symbol
    logger.info("\nTest 5: Unsubscribe all intervals")

    provider.unsubscribe('BTC/USDT')  # No interval = all intervals
    subs = provider.get_subscriptions()

    assert ('BTC/USDT', '1h') not in subs, "All intervals should be unsubscribed"
    assert ('BTC/USDT', '4h') not in subs, "All intervals should be unsubscribed"
    logger.info("✓ Unsubscribe all intervals works")

    # Test 6: Multiple symbols, multiple intervals
    logger.info("\nTest 6: Multiple symbols with multiple intervals")

    provider.subscribe('BTC/USDT', 'ETH/USDT', exchange='binance', interval='1h')
    provider.subscribe('BTC/USDT', 'ETH/USDT', exchange='binance', interval='4h')

    subs = provider.get_subscriptions()
    logger.info(f"Subscriptions: {subs}")

    assert ('BTC/USDT', '1h') in subs
    assert ('BTC/USDT', '4h') in subs
    assert ('ETH/USDT', '1h') in subs
    assert ('ETH/USDT', '4h') in subs
    logger.info("✓ Multiple symbols + intervals work correctly")

    logger.info("\n" + "="*60)
    logger.info("All multi-timeframe tests passed!")
    logger.info("="*60)


def test_candle_dataclass():
    """Test Candle dataclass has interval field."""

    logger.info("\nTest: Candle dataclass interval field")

    candle = Candle(
        symbol='BTC/USDT',
        timestamp=1234567890,
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=100.0,
        is_closed=True,
        interval='1h'
    )

    assert candle.interval == '1h', "Candle should have interval field"
    logger.info(f"✓ Candle has interval: {candle.interval}")

    # Test default value
    candle2 = Candle(
        symbol='ETH/USDT',
        timestamp=1234567890,
        open=3000.0,
        high=3100.0,
        low=2900.0,
        close=3050.0,
        volume=50.0
    )

    assert candle2.interval == '1h', "Candle should have default interval"
    logger.info(f"✓ Candle has default interval: {candle2.interval}")


if __name__ == '__main__':
    try:
        test_candle_dataclass()
        test_multi_timeframe_subscription()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
