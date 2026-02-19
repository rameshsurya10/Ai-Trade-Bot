"""
Tests for CapitalDataProvider
=============================
Unit tests covering initialization, subscription, candle conversion,
polling logic, reconnection, and callback dispatch.
"""

import threading
import time
from collections import deque
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from src.core.types import Candle, Tick
from src.data.capital_provider import (
    CapitalDataProvider,
    RESOLUTION_MAP,
    INTERVAL_SECONDS,
    _CapitalSubscription,
)


# ========== Fixtures ==========


@pytest.fixture
def base_config():
    """Minimal config dict for CapitalDataProvider."""
    return {
        "demo": True,
        "leverage": 30.0,
        "buffer_size": 100,
        "backfill_count": 10,
        "max_reconnect_attempts": 3,
        "base_reconnect_delay": 0.01,  # Fast for tests
        "polling": {
            "divisor": 10,
            "min_seconds": 1,
            "max_seconds": 60,
        },
    }


@pytest.fixture
def mock_brokerage():
    """A mocked CapitalBrokerage instance."""
    brokerage = MagicMock()
    brokerage.connect.return_value = True
    brokerage.disconnect.return_value = None
    brokerage.get_candles.return_value = []
    brokerage.get_current_price.return_value = None
    return brokerage


@pytest.fixture
def provider(base_config, mock_brokerage):
    """CapitalDataProvider with a mocked brokerage."""
    with patch(
        "src.data.capital_provider.CapitalBrokerage", return_value=mock_brokerage
    ):
        p = CapitalDataProvider(base_config)
    p._brokerage = mock_brokerage
    return p


def _make_raw_candle(time_str: str, o: float, h: float, l: float, c: float, v: int = 100):
    """Helper to build a raw candle dict matching brokerage.get_candles() output."""
    return {
        "time": time_str,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
    }


# ========== Initialization ==========


class TestInit:
    def test_default_config(self, provider, base_config):
        assert provider._buffer_size == 100
        assert provider._backfill_count == 10
        assert provider._max_reconnect_attempts == 3
        assert provider._connected is False
        assert provider._running is False

    def test_resolution_map_completeness(self):
        expected = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}
        assert set(RESOLUTION_MAP.keys()) == expected

    def test_interval_seconds_completeness(self):
        assert set(INTERVAL_SECONDS.keys()) == set(RESOLUTION_MAP.keys())


# ========== Connection ==========


class TestConnection:
    def test_connect_success(self, provider, mock_brokerage):
        mock_brokerage.connect.return_value = True
        assert provider.connect() is True
        assert provider.is_connected is True

    def test_connect_failure(self, provider, mock_brokerage):
        mock_brokerage.connect.return_value = False
        assert provider.connect() is False
        assert provider.is_connected is False

    def test_connect_exception(self, provider, mock_brokerage):
        mock_brokerage.connect.side_effect = Exception("network error")
        assert provider.connect() is False
        assert provider.is_connected is False

    def test_disconnect(self, provider, mock_brokerage):
        provider._connected = True
        provider.disconnect()
        mock_brokerage.disconnect.assert_called_once()
        assert provider.is_connected is False


# ========== Subscription ==========


class TestSubscription:
    def test_subscribe_single(self, provider, mock_brokerage):
        provider._connected = True
        mock_brokerage.get_candles.return_value = [
            _make_raw_candle("2024/01/15 13:00:00", 1.09, 1.10, 1.08, 1.095),
            _make_raw_candle("2024/01/15 14:00:00", 1.095, 1.11, 1.09, 1.10),
        ]

        provider.subscribe("EUR/USD", interval="1h")

        assert ("EUR/USD", "1h") in provider.get_subscriptions()
        mock_brokerage.get_candles.assert_called_once_with(
            symbol="EUR/USD", resolution="HOUR", max_candles=10
        )

    def test_subscribe_multiple(self, provider, mock_brokerage):
        provider._connected = True
        mock_brokerage.get_candles.return_value = [
            _make_raw_candle("2024/01/15 14:00:00", 1.09, 1.10, 1.08, 1.095),
            _make_raw_candle("2024/01/15 15:00:00", 1.095, 1.11, 1.09, 1.10),
        ]

        provider.subscribe("EUR/USD", "GBP/USD", interval="1h")

        subs = provider.get_subscriptions()
        assert ("EUR/USD", "1h") in subs
        assert ("GBP/USD", "1h") in subs

    def test_subscribe_duplicate_ignored(self, provider, mock_brokerage):
        provider._connected = True
        mock_brokerage.get_candles.return_value = [
            _make_raw_candle("2024/01/15 14:00:00", 1.09, 1.10, 1.08, 1.095),
            _make_raw_candle("2024/01/15 15:00:00", 1.095, 1.11, 1.09, 1.10),
        ]

        provider.subscribe("EUR/USD", interval="1h")
        provider.subscribe("EUR/USD", interval="1h")  # duplicate

        assert mock_brokerage.get_candles.call_count == 1

    def test_subscribe_invalid_interval(self, provider):
        provider.subscribe("EUR/USD", interval="2h")  # unsupported
        assert len(provider.get_subscriptions()) == 0

    def test_unsubscribe_specific_interval(self, provider, mock_brokerage):
        provider._connected = True
        mock_brokerage.get_candles.return_value = [
            _make_raw_candle("2024/01/15 14:00:00", 1.09, 1.10, 1.08, 1.095),
            _make_raw_candle("2024/01/15 15:00:00", 1.095, 1.11, 1.09, 1.10),
        ]

        provider.subscribe("EUR/USD", interval="1h")
        provider.subscribe("EUR/USD", interval="4h")
        provider.unsubscribe("EUR/USD", interval="1h")

        subs = provider.get_subscriptions()
        assert ("EUR/USD", "1h") not in subs
        assert ("EUR/USD", "4h") in subs

    def test_unsubscribe_all_intervals(self, provider, mock_brokerage):
        provider._connected = True
        mock_brokerage.get_candles.return_value = [
            _make_raw_candle("2024/01/15 14:00:00", 1.09, 1.10, 1.08, 1.095),
            _make_raw_candle("2024/01/15 15:00:00", 1.095, 1.11, 1.09, 1.10),
        ]

        provider.subscribe("EUR/USD", interval="1h")
        provider.subscribe("EUR/USD", interval="4h")
        provider.unsubscribe("EUR/USD")

        assert len(provider.get_subscriptions()) == 0


# ========== Data Retrieval ==========


class TestDataRetrieval:
    def test_get_candles_empty(self, provider):
        df = provider.get_candles("EUR/USD")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "timestamp" in df.columns

    def test_get_candles_with_data(self, provider, mock_brokerage):
        provider._connected = True
        mock_brokerage.get_candles.return_value = [
            _make_raw_candle("2024/01/15 13:00:00", 1.09, 1.10, 1.08, 1.095),
            _make_raw_candle("2024/01/15 14:00:00", 1.095, 1.11, 1.09, 1.10),
            _make_raw_candle("2024/01/15 15:00:00", 1.10, 1.12, 1.09, 1.11),
        ]

        provider.subscribe("EUR/USD", interval="1h")
        df = provider.get_candles("EUR/USD", interval="1h")

        assert len(df) == 2  # 3 raw - 1 forming = 2 closed
        assert df.iloc[0]["open"] == 1.09

    def test_get_candles_limit(self, provider, mock_brokerage):
        provider._connected = True
        # Create many candles
        candles = [
            _make_raw_candle(f"2024/01/15 {h:02d}:00:00", 1.09, 1.10, 1.08, 1.095)
            for h in range(20)
        ]
        mock_brokerage.get_candles.return_value = candles

        provider.subscribe("EUR/USD", interval="1h")
        df = provider.get_candles("EUR/USD", interval="1h", limit=5)

        assert len(df) == 5

    def test_get_candles_auto_interval(self, provider, mock_brokerage):
        provider._connected = True
        mock_brokerage.get_candles.return_value = [
            _make_raw_candle("2024/01/15 13:00:00", 1.09, 1.10, 1.08, 1.095),
            _make_raw_candle("2024/01/15 14:00:00", 1.095, 1.11, 1.09, 1.10),
        ]

        provider.subscribe("EUR/USD", interval="1h")
        df = provider.get_candles("EUR/USD")  # no interval specified

        assert len(df) == 1

    def test_get_tick_none(self, provider):
        assert provider.get_tick("EUR/USD") is None

    def test_get_tick_with_data(self, provider):
        tick = Tick(symbol="EUR/USD", price=1.095, timestamp=1000, quantity=0.0)
        sub = _CapitalSubscription(
            symbol="EUR/USD",
            interval="1h",
            resolution="HOUR",
            latest_tick=tick,
        )
        provider._subscriptions[("EUR/USD", "1h")] = sub

        result = provider.get_tick("EUR/USD")
        assert result is not None
        assert result.price == 1.095


# ========== Callbacks ==========


class TestCallbacks:
    def test_on_candle_callback(self, provider):
        received = []
        provider.on_candle(lambda c: received.append(c))

        candle = Candle(
            timestamp=1000, datetime=datetime.now(timezone.utc),
            open=1.09, high=1.10, low=1.08, close=1.095, volume=100,
            symbol="EUR/USD", interval="1h",
        )

        for cb in provider._candle_callbacks:
            cb(candle)

        assert len(received) == 1
        assert received[0].symbol == "EUR/USD"

    def test_on_candle_closed_callback(self, provider):
        received = []
        provider.on_candle_closed(lambda c: received.append(c))

        candle = Candle(
            timestamp=1000, datetime=datetime.now(timezone.utc),
            open=1.09, high=1.10, low=1.08, close=1.095, volume=100,
            symbol="EUR/USD", interval="1h",
        )

        for cb in provider._candle_closed_callbacks:
            cb(candle)

        assert len(received) == 1

    def test_on_tick_callback(self, provider):
        received = []
        provider.on_tick(lambda t: received.append(t))

        tick = Tick(symbol="EUR/USD", price=1.095, timestamp=1000, quantity=0.0)

        for cb in provider._tick_callbacks:
            cb(tick)

        assert len(received) == 1
        assert received[0].price == 1.095


# ========== Candle Conversion ==========


class TestCandleConversion:
    def test_raw_to_candle_slash_format(self, provider):
        raw = _make_raw_candle("2024/01/15 14:00:00", 1.09, 1.10, 1.08, 1.095, 500)
        candle = provider._raw_to_candle(raw, "EUR/USD", "1h")

        assert candle is not None
        assert candle.symbol == "EUR/USD"
        assert candle.interval == "1h"
        assert candle.open == 1.09
        assert candle.high == 1.10
        assert candle.low == 1.08
        assert candle.close == 1.095
        assert candle.volume == 500
        assert candle.is_closed is True
        assert candle.datetime.year == 2024

    def test_raw_to_candle_iso_format(self, provider):
        raw = _make_raw_candle("2024-01-15T14:00:00", 1.09, 1.10, 1.08, 1.095)
        candle = provider._raw_to_candle(raw, "EUR/USD", "1h")

        assert candle is not None
        assert candle.datetime.month == 1
        assert candle.datetime.hour == 14

    def test_raw_to_candle_empty_time(self, provider):
        raw = _make_raw_candle("", 1.09, 1.10, 1.08, 1.095)
        candle = provider._raw_to_candle(raw, "EUR/USD", "1h")

        # Should still produce a candle (falls back to current time)
        assert candle is not None

    def test_parse_time_to_ms_formats(self):
        parse = CapitalDataProvider._parse_time_to_ms

        # Slash format
        ms = parse("2024/01/15 14:00:00")
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.hour == 14

        # ISO format
        ms2 = parse("2024-01-15T14:00:00")
        assert ms == ms2

        # Dash-space format
        ms3 = parse("2024-01-15 14:00:00")
        assert ms == ms3


# ========== Polling Logic ==========


class TestPolling:
    def test_poll_subscription_detects_new_candle(self, provider, mock_brokerage):
        provider._connected = True

        sub = _CapitalSubscription(
            symbol="EUR/USD",
            interval="1h",
            resolution="HOUR",
            last_candle_time="2024/01/15 13:00:00",
        )

        mock_brokerage.get_candles.return_value = [
            _make_raw_candle("2024/01/15 12:00:00", 1.08, 1.09, 1.07, 1.085),
            _make_raw_candle("2024/01/15 13:00:00", 1.085, 1.095, 1.08, 1.09),
            _make_raw_candle("2024/01/15 14:00:00", 1.09, 1.10, 1.08, 1.095),
            _make_raw_candle("2024/01/15 15:00:00", 1.095, 1.11, 1.09, 1.10),
            _make_raw_candle("2024/01/15 16:00:00", 1.10, 1.12, 1.09, 1.11),
        ]

        received = []
        provider.on_candle_closed(lambda c: received.append(c))

        provider._poll_subscription(sub)

        # The second-to-last (15:00) should be detected as new closed candle
        assert len(received) == 1
        assert received[0].close == 1.10
        assert sub.last_candle_time == "2024/01/15 15:00:00"

    def test_poll_subscription_no_new_candle(self, provider, mock_brokerage):
        provider._connected = True

        sub = _CapitalSubscription(
            symbol="EUR/USD",
            interval="1h",
            resolution="HOUR",
            last_candle_time="2024/01/15 15:00:00",
        )

        mock_brokerage.get_candles.return_value = [
            _make_raw_candle("2024/01/15 14:00:00", 1.09, 1.10, 1.08, 1.095),
            _make_raw_candle("2024/01/15 15:00:00", 1.095, 1.11, 1.09, 1.10),
            _make_raw_candle("2024/01/15 16:00:00", 1.10, 1.12, 1.09, 1.11),
            _make_raw_candle("2024/01/15 17:00:00", 1.10, 1.12, 1.09, 1.11),
            _make_raw_candle("2024/01/15 18:00:00", 1.10, 1.12, 1.09, 1.11),  # forming
        ]

        received = []
        provider.on_candle_closed(lambda c: received.append(c))

        provider._poll_subscription(sub)

        # 17:00 is new, should fire
        assert len(received) == 1
        assert sub.last_candle_time == "2024/01/15 17:00:00"

    def test_poll_updates_tick(self, provider, mock_brokerage):
        provider._connected = True

        sub = _CapitalSubscription(
            symbol="EUR/USD",
            interval="1h",
            resolution="HOUR",
            last_candle_time="2024/01/15 15:00:00",
        )
        provider._subscriptions[("EUR/USD", "1h")] = sub

        mock_brokerage.get_candles.return_value = [
            _make_raw_candle("2024/01/15 14:00:00", 1.09, 1.10, 1.08, 1.095),
            _make_raw_candle("2024/01/15 15:00:00", 1.095, 1.11, 1.09, 1.10),
            _make_raw_candle("2024/01/15 16:00:00", 1.10, 1.12, 1.09, 1.11),
            _make_raw_candle("2024/01/15 17:00:00", 1.10, 1.12, 1.09, 1.10),
            _make_raw_candle("2024/01/15 18:00:00", 1.10, 1.12, 1.09, 1.115),  # forming
        ]

        provider._poll_subscription(sub)

        tick = provider.get_tick("EUR/USD")
        assert tick is not None
        assert tick.price == 1.115

    def test_poll_not_connected_skips(self, provider, mock_brokerage):
        provider._connected = False
        sub = _CapitalSubscription(
            symbol="EUR/USD", interval="1h", resolution="HOUR"
        )

        provider._poll_subscription(sub)
        mock_brokerage.get_candles.assert_not_called()


# ========== Adaptive Sleep ==========


class TestAdaptiveSleep:
    def test_sleep_1h_interval(self, provider):
        sub = _CapitalSubscription(symbol="EUR/USD", interval="1h", resolution="HOUR")
        sleep = provider._compute_poll_sleep([sub])
        # 3600 / 10 = 360, clamped to max 60
        assert sleep == 60.0

    def test_sleep_1m_interval(self, provider):
        sub = _CapitalSubscription(symbol="EUR/USD", interval="1m", resolution="MINUTE")
        sleep = provider._compute_poll_sleep([sub])
        # 60 / 10 = 6, clamped between 1 and 60
        assert sleep == 6.0

    def test_sleep_mixed_intervals(self, provider):
        sub1 = _CapitalSubscription(symbol="EUR/USD", interval="1h", resolution="HOUR")
        sub5 = _CapitalSubscription(symbol="GBP/USD", interval="5m", resolution="MINUTE_5")
        sleep = provider._compute_poll_sleep([sub1, sub5])
        # min(3600, 300) = 300, 300/10 = 30
        assert sleep == 30.0

    def test_sleep_empty_subs(self, provider):
        sleep = provider._compute_poll_sleep([])
        assert sleep == 60.0  # max_poll_seconds from config


# ========== Reconnection ==========


class TestReconnection:
    def test_reconnect_success_on_first_attempt(self, provider, mock_brokerage):
        provider._connected = False
        mock_brokerage.connect.return_value = True

        provider._attempt_reconnect()

        assert provider.is_connected is True
        assert provider._stats["reconnects"] == 1

    def test_reconnect_success_on_second_attempt(self, provider, mock_brokerage):
        provider._connected = False
        mock_brokerage.connect.side_effect = [False, True]

        provider._attempt_reconnect()

        assert provider.is_connected is True
        assert mock_brokerage.connect.call_count == 2

    def test_reconnect_all_attempts_fail(self, provider, mock_brokerage):
        provider._connected = False
        mock_brokerage.connect.return_value = False

        provider._attempt_reconnect()

        assert provider.is_connected is False
        assert mock_brokerage.connect.call_count == 3  # max_reconnect_attempts


# ========== Status ==========


class TestStatus:
    def test_get_status(self, provider):
        provider._connected = True
        provider._running = True
        provider._stats["candles_received"] = 42

        status = provider.get_status()

        assert status["provider"] == "capital.com"
        assert status["running"] is True
        assert status["connected"] is True
        assert status["candles_received"] == 42

    def test_set_database(self, provider):
        db = MagicMock()
        provider.set_database(db)
        assert provider._database is db


# ========== Lifecycle ==========


class TestLifecycle:
    def test_start_stop(self, provider):
        provider.start()
        assert provider.is_running is True
        assert provider._poll_thread is not None
        assert provider._poll_thread.is_alive()

        provider.stop()
        assert provider.is_running is False

    def test_start_idempotent(self, provider):
        provider.start()
        thread1 = provider._poll_thread

        provider.start()  # should not create new thread
        thread2 = provider._poll_thread

        assert thread1 is thread2
        provider.stop()


# ========== Database Persistence ==========


class TestPersistence:
    def test_persist_candle(self, provider):
        db = MagicMock()
        provider._database = db

        candle = Candle(
            timestamp=1000, datetime=datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc),
            open=1.09, high=1.10, low=1.08, close=1.095, volume=100,
            symbol="EUR/USD", interval="1h",
        )
        sub = _CapitalSubscription(symbol="EUR/USD", interval="1h", resolution="HOUR")

        provider._persist_candle(candle, sub)

        db.save_candles.assert_called_once()
        call_args = db.save_candles.call_args
        df = call_args[0][0]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert call_args[1]["symbol"] == "EUR/USD"
        assert call_args[1]["interval"] == "1h"

    def test_persist_buffer(self, provider):
        db = MagicMock()
        provider._database = db

        sub = _CapitalSubscription(symbol="EUR/USD", interval="1h", resolution="HOUR")
        for i in range(3):
            sub.buffer.append(
                Candle(
                    timestamp=i * 1000,
                    datetime=datetime(2024, 1, 15, i, 0, tzinfo=timezone.utc),
                    open=1.09, high=1.10, low=1.08, close=1.095, volume=100,
                    symbol="EUR/USD", interval="1h",
                )
            )

        provider._persist_buffer(sub)

        db.save_candles.assert_called_once()
        df = db.save_candles.call_args[0][0]
        assert len(df) == 3

    def test_persist_failure_does_not_crash(self, provider):
        db = MagicMock()
        db.save_candles.side_effect = Exception("DB error")
        provider._database = db

        candle = Candle(
            timestamp=1000, datetime=datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc),
            open=1.09, high=1.10, low=1.08, close=1.095, volume=100,
            symbol="EUR/USD", interval="1h",
        )
        sub = _CapitalSubscription(symbol="EUR/USD", interval="1h", resolution="HOUR")

        # Should not raise
        provider._persist_candle(candle, sub)
