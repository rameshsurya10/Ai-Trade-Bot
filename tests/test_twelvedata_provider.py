"""
Tests for Twelve Data Forex Data Provider
==========================================
"""

import time
import threading
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from src.data.twelvedata_provider import TwelveDataProvider, INTERVAL_MAP


# --- Fixtures ---

@pytest.fixture
def default_config():
    return {
        'enabled': True,
        'backfill_count': 10,
        'buffer_size': 100,
        'polling': {'divisor': 5, 'min_seconds': 1, 'max_seconds': 60},
        'rate_limit': {'per_minute': 8, 'per_day': 800, 'warn_threshold': 700},
    }


@pytest.fixture
def mock_response_ok():
    """Successful Twelve Data time_series response."""
    return {
        'meta': {'symbol': 'EUR/USD', 'interval': '1h'},
        'values': [
            {
                'datetime': '2024-06-15 14:00:00',
                'open': '1.08500', 'high': '1.08700',
                'low': '1.08400', 'close': '1.08650', 'volume': '0',
            },
            {
                'datetime': '2024-06-15 13:00:00',
                'open': '1.08300', 'high': '1.08550',
                'low': '1.08200', 'close': '1.08500', 'volume': '0',
            },
        ],
    }


@pytest.fixture
def mock_backfill_response():
    """Backfill response with 3 candles (newest first)."""
    return {
        'meta': {'symbol': 'EUR/USD', 'interval': '1h'},
        'values': [
            {'datetime': '2024-06-15 14:00:00', 'open': '1.0850', 'high': '1.0870', 'low': '1.0840', 'close': '1.0865', 'volume': '0'},
            {'datetime': '2024-06-15 13:00:00', 'open': '1.0830', 'high': '1.0855', 'low': '1.0820', 'close': '1.0850', 'volume': '0'},
            {'datetime': '2024-06-15 12:00:00', 'open': '1.0810', 'high': '1.0835', 'low': '1.0800', 'close': '1.0830', 'volume': '0'},
        ],
    }


def make_mock_get(response_data, status_code=200):
    """Create a mock for requests.Session.get."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = response_data
    mock_resp.text = str(response_data)
    return mock_resp


# --- Init Tests ---

class TestInit:
    def test_default_config(self):
        provider = TwelveDataProvider()
        assert provider._backfill_count == 500
        assert provider._max_per_minute == 8
        assert provider._max_per_day == 800
        assert not provider.is_connected

    def test_custom_config(self, default_config):
        provider = TwelveDataProvider(default_config)
        assert provider._backfill_count == 10
        assert provider._buffer_size == 100
        assert provider._poll_divisor == 5

    def test_no_api_key(self):
        with patch.dict('os.environ', {}, clear=True):
            provider = TwelveDataProvider()
            assert provider._api_key == ''


# --- Connection Tests ---

class TestConnect:
    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_connect_success(self, default_config, mock_response_ok):
        provider = TwelveDataProvider(default_config)
        provider._session.get = MagicMock(return_value=make_mock_get(mock_response_ok))
        assert provider.connect() is True
        assert provider.is_connected

    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': ''})
    def test_connect_no_key(self, default_config):
        provider = TwelveDataProvider(default_config)
        assert provider.connect() is False
        assert not provider.is_connected

    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_connect_api_error(self, default_config):
        provider = TwelveDataProvider(default_config)
        error_resp = {'status': 'error', 'message': 'Invalid API key'}
        provider._session.get = MagicMock(return_value=make_mock_get(error_resp))
        assert provider.connect() is False

    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_connect_idempotent(self, default_config, mock_response_ok):
        provider = TwelveDataProvider(default_config)
        provider._session.get = MagicMock(return_value=make_mock_get(mock_response_ok))
        assert provider.connect() is True
        # Second call should return True immediately without API call
        provider._session.get.reset_mock()
        assert provider.connect() is True
        provider._session.get.assert_not_called()

    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_disconnect(self, default_config, mock_response_ok):
        provider = TwelveDataProvider(default_config)
        provider._session.get = MagicMock(return_value=make_mock_get(mock_response_ok))
        provider.connect()
        provider.disconnect()
        assert not provider.is_connected


# --- Subscription Tests ---

class TestSubscription:
    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_subscribe(self, default_config):
        provider = TwelveDataProvider(default_config)
        provider.subscribe("EUR/USD", "1h")
        assert ("EUR/USD", "1h") in provider.get_subscriptions()

    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_subscribe_invalid_interval(self, default_config):
        provider = TwelveDataProvider(default_config)
        provider.subscribe("EUR/USD", "3h")  # Invalid
        assert len(provider.get_subscriptions()) == 0

    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_subscribe_duplicate(self, default_config):
        provider = TwelveDataProvider(default_config)
        provider.subscribe("EUR/USD", "1h")
        provider.subscribe("EUR/USD", "1h")  # Duplicate
        assert len(provider.get_subscriptions()) == 1

    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_unsubscribe_specific(self, default_config):
        provider = TwelveDataProvider(default_config)
        provider.subscribe("EUR/USD", "1h")
        provider.subscribe("EUR/USD", "15m")
        provider.unsubscribe("EUR/USD", "1h")
        subs = provider.get_subscriptions()
        assert ("EUR/USD", "1h") not in subs
        assert ("EUR/USD", "15m") in subs

    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_unsubscribe_all_intervals(self, default_config):
        provider = TwelveDataProvider(default_config)
        provider.subscribe("EUR/USD", "1h")
        provider.subscribe("EUR/USD", "15m")
        provider.unsubscribe("EUR/USD")
        assert len(provider.get_subscriptions()) == 0


# --- Backfill Tests ---

class TestBackfill:
    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_backfill_on_subscribe(self, default_config, mock_backfill_response):
        provider = TwelveDataProvider(default_config)
        provider._session.get = MagicMock(return_value=make_mock_get(mock_backfill_response))
        provider._connected = True
        provider.subscribe("EUR/USD", "1h")

        sub = provider._subscriptions[("EUR/USD", "1h")]
        assert len(sub.candle_buffer) == 3
        # Oldest first (reversed from API response)
        assert sub.candle_buffer[0].close == 1.083
        assert sub.candle_buffer[-1].close == 1.0865

    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_backfill_saves_to_db(self, default_config, mock_backfill_response):
        provider = TwelveDataProvider(default_config)
        provider._session.get = MagicMock(return_value=make_mock_get(mock_backfill_response))
        provider._connected = True

        mock_db = MagicMock()
        provider.set_database(mock_db)
        provider.subscribe("EUR/USD", "1h")

        assert mock_db.save_candles.call_count == 3


# --- Candle Detection Tests ---

class TestCandleDetection:
    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_new_candle_detected(self, default_config, mock_response_ok):
        provider = TwelveDataProvider(default_config)
        provider._session.get = MagicMock(return_value=make_mock_get(mock_response_ok))

        sub = MagicMock()
        sub.symbol = "EUR/USD"
        sub.interval = "1h"
        sub.last_candle_ts = None
        sub.candle_buffer = []

        callbacks_received = []
        provider.on_candle_closed(lambda c: callbacks_received.append(c))

        provider._check_for_new_candle(sub)

        assert len(callbacks_received) == 1
        assert callbacks_received[0].symbol == "EUR/USD"
        assert callbacks_received[0].close == 1.085  # Second candle (closed)

    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_duplicate_candle_ignored(self, default_config, mock_response_ok):
        provider = TwelveDataProvider(default_config)
        provider._session.get = MagicMock(return_value=make_mock_get(mock_response_ok))

        # Parse the timestamp of the closed candle
        closed_ts = TwelveDataProvider._parse_timestamp('2024-06-15 13:00:00')

        sub = MagicMock()
        sub.symbol = "EUR/USD"
        sub.interval = "1h"
        sub.last_candle_ts = closed_ts  # Already seen
        sub.candle_buffer = []

        callbacks_received = []
        provider.on_candle_closed(lambda c: callbacks_received.append(c))

        provider._check_for_new_candle(sub)

        assert len(callbacks_received) == 0  # No new candle


# --- Rate Limit Tests ---

class TestRateLimit:
    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_daily_limit_raises(self, default_config):
        provider = TwelveDataProvider(default_config)
        provider._calls_today = 800  # At limit

        with pytest.raises(RuntimeError, match="daily API limit"):
            provider._wait_for_rate_limit()

    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_counter_increment(self, default_config):
        provider = TwelveDataProvider(default_config)
        provider._increment_counters()
        assert provider._calls_this_minute == 1
        assert provider._calls_today == 1


# --- Status Tests ---

class TestStatus:
    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_get_status(self, default_config):
        provider = TwelveDataProvider(default_config)
        provider._connected = True
        provider._calls_today = 50

        status = provider.get_status()
        assert status['connected'] is True
        assert status['calls_today'] == 50
        assert status['calls_remaining'] == 750
        assert status['subscriptions'] == 0


# --- Candle Building Tests ---

class TestCandleBuilding:
    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_build_candle(self, default_config):
        provider = TwelveDataProvider(default_config)
        data = {
            'datetime': '2024-06-15 13:00:00',
            'open': '1.08300', 'high': '1.08550',
            'low': '1.08200', 'close': '1.08500', 'volume': '0',
        }
        candle = provider._build_candle(data, "EUR/USD", "1h")

        assert candle.symbol == "EUR/USD"
        assert candle.interval == "1h"
        assert candle.open == 1.083
        assert candle.high == 1.0855
        assert candle.low == 1.082
        assert candle.close == 1.085
        assert candle.volume == 0.0
        assert candle.is_closed is True

    def test_parse_timestamp(self):
        ts = TwelveDataProvider._parse_timestamp("2024-06-15 13:00:00")
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.day == 15
        assert dt.hour == 13


# --- Lifecycle Tests ---

class TestLifecycle:
    @patch.dict('os.environ', {'TWELVE_DATA_API_KEY': 'test_key'})
    def test_start_stop(self, default_config):
        provider = TwelveDataProvider(default_config)
        provider.start()
        assert provider._running is True
        assert provider._poll_thread is not None
        assert provider._poll_thread.is_alive()

        provider.stop()
        assert provider._running is False


# --- Interval Map Tests ---

class TestIntervalMap:
    def test_all_intervals_mapped(self):
        expected = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        for interval in expected:
            assert interval in INTERVAL_MAP

    def test_1h_maps_correctly(self):
        assert INTERVAL_MAP['1h'] == '1h'

    def test_15m_maps_correctly(self):
        assert INTERVAL_MAP['15m'] == '15min'

    def test_1d_maps_correctly(self):
        assert INTERVAL_MAP['1d'] == '1day'
