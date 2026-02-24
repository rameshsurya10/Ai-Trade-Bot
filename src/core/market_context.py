"""
Market Context
==============
Detect whether a symbol belongs to crypto, forex, or equity markets.
Used by LiveTradingRunner to route symbols to the correct data provider and brokerage.
"""

from enum import Enum

from src.brokerages.utils.symbol_normalizer import get_symbol_normalizer


class MarketType(Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    EQUITY = "equity"


# Exchanges that are explicitly forex
_FOREX_EXCHANGES = {"mt5", "metatrader5", "twelvedata", "oanda", "fxcm"}

# Exchanges that are explicitly crypto
_CRYPTO_EXCHANGES = {"binance", "coinbase", "kraken", "bybit", "okx", "kucoin"}


def detect_market_type(symbol: str, exchange: str) -> MarketType:
    """
    Detect market type from symbol and exchange name.

    Args:
        symbol: Trading pair (e.g., "EUR/USD", "BTC/USDT")
        exchange: Exchange/broker name (e.g., "mt5", "binance")

    Returns:
        MarketType enum value
    """
    exchange_lower = exchange.lower()

    if exchange_lower in _FOREX_EXCHANGES:
        return MarketType.FOREX

    if exchange_lower in _CRYPTO_EXCHANGES:
        return MarketType.CRYPTO

    # Fallback: check if it looks like a forex pair
    normalizer = get_symbol_normalizer()
    if normalizer.is_valid_forex_pair(symbol):
        return MarketType.FOREX

    return MarketType.CRYPTO
