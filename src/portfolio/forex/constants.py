"""
Forex Constants and Currency Pair Configurations
=================================================

Defines core constants and configurations for Forex trading:
- Currency pair specifications (pip sizes, precisions)
- Lot type definitions (Standard, Mini, Micro)
- Regulatory leverage limits
- Default settings

Usage:
    from src.portfolio.forex.constants import FOREX_PAIRS, LotType, US_MAX_LEVERAGE

    pair_config = FOREX_PAIRS["EUR/USD"]
    print(f"Pip size: {pair_config.pip_size}")  # 0.0001
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class LotType(Enum):
    """Forex lot sizes (units per lot)."""
    STANDARD = 100000    # 100,000 units - $10 per pip on EUR/USD
    MINI = 10000         # 10,000 units - $1 per pip on EUR/USD
    MICRO = 1000         # 1,000 units - $0.10 per pip on EUR/USD
    NANO = 100           # 100 units - $0.01 per pip on EUR/USD


class AssetType(Enum):
    """Trading asset types."""
    FOREX = "forex"
    CRYPTO = "crypto"
    EQUITY = "equity"
    COMMODITY = "commodity"
    INDEX = "index"


@dataclass(frozen=True)
class CurrencyPairConfig:
    """
    Configuration for a Forex currency pair.

    Attributes:
        symbol: Standard symbol format (e.g., "EUR/USD")
        base_currency: Base currency code (e.g., "EUR")
        quote_currency: Quote currency code (e.g., "USD")
        pip_size: Size of one pip (0.0001 for most, 0.01 for JPY pairs)
        pip_decimal_places: Number of decimal places for pips (4 or 2)
        min_lot_size: Minimum tradeable lot size
        max_leverage: Maximum allowed leverage for this pair
        typical_spread: Typical spread in pips (for estimation)
        category: Pair category (major, minor, cross, exotic)
    """
    symbol: str
    base_currency: str
    quote_currency: str
    pip_size: float
    pip_decimal_places: int
    min_lot_size: float
    max_leverage: float
    typical_spread: float
    category: str = "major"

    @property
    def oanda_symbol(self) -> str:
        """Convert to OANDA format (EUR_USD)."""
        return self.symbol.replace("/", "_")

    @property
    def binance_symbol(self) -> str:
        """Convert to Binance format (EURUSD)."""
        return self.symbol.replace("/", "")

    def is_jpy_pair(self) -> bool:
        """Check if this is a JPY pair (2 decimal pip)."""
        return "JPY" in self.symbol


# Major Pairs (USD as base or quote)
MAJOR_PAIRS: Dict[str, CurrencyPairConfig] = {
    "EUR/USD": CurrencyPairConfig(
        symbol="EUR/USD",
        base_currency="EUR",
        quote_currency="USD",
        pip_size=0.0001,
        pip_decimal_places=4,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=0.8,
        category="major"
    ),
    "GBP/USD": CurrencyPairConfig(
        symbol="GBP/USD",
        base_currency="GBP",
        quote_currency="USD",
        pip_size=0.0001,
        pip_decimal_places=4,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=1.2,
        category="major"
    ),
    "USD/JPY": CurrencyPairConfig(
        symbol="USD/JPY",
        base_currency="USD",
        quote_currency="JPY",
        pip_size=0.01,
        pip_decimal_places=2,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=0.9,
        category="major"
    ),
    "USD/CHF": CurrencyPairConfig(
        symbol="USD/CHF",
        base_currency="USD",
        quote_currency="CHF",
        pip_size=0.0001,
        pip_decimal_places=4,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=1.5,
        category="major"
    ),
    "AUD/USD": CurrencyPairConfig(
        symbol="AUD/USD",
        base_currency="AUD",
        quote_currency="USD",
        pip_size=0.0001,
        pip_decimal_places=4,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=1.2,
        category="major"
    ),
    "USD/CAD": CurrencyPairConfig(
        symbol="USD/CAD",
        base_currency="USD",
        quote_currency="CAD",
        pip_size=0.0001,
        pip_decimal_places=4,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=1.5,
        category="major"
    ),
    "NZD/USD": CurrencyPairConfig(
        symbol="NZD/USD",
        base_currency="NZD",
        quote_currency="USD",
        pip_size=0.0001,
        pip_decimal_places=4,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=1.8,
        category="major"
    ),
}

# Cross Pairs (non-USD pairs)
CROSS_PAIRS: Dict[str, CurrencyPairConfig] = {
    "EUR/GBP": CurrencyPairConfig(
        symbol="EUR/GBP",
        base_currency="EUR",
        quote_currency="GBP",
        pip_size=0.0001,
        pip_decimal_places=4,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=1.5,
        category="cross"
    ),
    "EUR/JPY": CurrencyPairConfig(
        symbol="EUR/JPY",
        base_currency="EUR",
        quote_currency="JPY",
        pip_size=0.01,
        pip_decimal_places=2,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=1.5,
        category="cross"
    ),
    "GBP/JPY": CurrencyPairConfig(
        symbol="GBP/JPY",
        base_currency="GBP",
        quote_currency="JPY",
        pip_size=0.01,
        pip_decimal_places=2,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=2.5,
        category="cross"
    ),
    "EUR/CHF": CurrencyPairConfig(
        symbol="EUR/CHF",
        base_currency="EUR",
        quote_currency="CHF",
        pip_size=0.0001,
        pip_decimal_places=4,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=1.8,
        category="cross"
    ),
    "EUR/AUD": CurrencyPairConfig(
        symbol="EUR/AUD",
        base_currency="EUR",
        quote_currency="AUD",
        pip_size=0.0001,
        pip_decimal_places=4,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=2.0,
        category="cross"
    ),
    "AUD/JPY": CurrencyPairConfig(
        symbol="AUD/JPY",
        base_currency="AUD",
        quote_currency="JPY",
        pip_size=0.01,
        pip_decimal_places=2,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=2.0,
        category="cross"
    ),
    "CHF/JPY": CurrencyPairConfig(
        symbol="CHF/JPY",
        base_currency="CHF",
        quote_currency="JPY",
        pip_size=0.01,
        pip_decimal_places=2,
        min_lot_size=0.01,
        max_leverage=50.0,
        typical_spread=2.5,
        category="cross"
    ),
}

# Combined forex pairs dictionary
FOREX_PAIRS: Dict[str, CurrencyPairConfig] = {**MAJOR_PAIRS, **CROSS_PAIRS}

# List of all supported forex symbols
FOREX_SYMBOLS: List[str] = list(FOREX_PAIRS.keys())

# Regulatory Leverage Limits
US_MAX_LEVERAGE = 50.0      # NFA/CFTC regulation for US traders
EU_MAX_LEVERAGE = 30.0      # ESMA regulation for EU retail traders
UK_MAX_LEVERAGE = 30.0      # FCA regulation for UK retail traders
AU_MAX_LEVERAGE = 30.0      # ASIC regulation for AU retail traders
OFFSHORE_MAX_LEVERAGE = 500.0  # Some offshore brokers

# Default Account Settings
DEFAULT_ACCOUNT_CURRENCY = "USD"
DEFAULT_LOT_TYPE = LotType.STANDARD
DEFAULT_MAX_LEVERAGE = US_MAX_LEVERAGE

# Margin Levels
MARGIN_CALL_LEVEL = 0.50    # 50% margin level triggers margin call
STOP_OUT_LEVEL = 0.30       # 30% margin level triggers forced liquidation

# Trading Session Times (UTC)
TRADING_SESSIONS = {
    "sydney": {"open": 22, "close": 7},      # 22:00 - 07:00 UTC
    "tokyo": {"open": 0, "close": 9},        # 00:00 - 09:00 UTC
    "london": {"open": 8, "close": 17},      # 08:00 - 17:00 UTC
    "new_york": {"open": 13, "close": 22},   # 13:00 - 22:00 UTC
}

# Rollover Time (5 PM New York = 22:00 UTC in winter, 21:00 in summer)
ROLLOVER_HOUR_UTC = 22

# Correlation Groups (for risk management)
CORRELATION_GROUPS = {
    "usd_longs": ["EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD"],
    "usd_shorts": ["USD/JPY", "USD/CHF", "USD/CAD"],
    "eur_crosses": ["EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/AUD"],
    "jpy_crosses": ["EUR/JPY", "GBP/JPY", "AUD/JPY", "CHF/JPY"],
}


def get_pair_config(symbol: str) -> CurrencyPairConfig:
    """
    Get configuration for a currency pair.

    Args:
        symbol: Currency pair in any format (EUR/USD, EUR_USD, EURUSD)

    Returns:
        CurrencyPairConfig for the pair

    Raises:
        ValueError: If pair is not found
    """
    # Normalize symbol to standard format
    normalized = symbol.upper().replace("_", "/").replace("-", "/")
    if "/" not in normalized and len(normalized) == 6:
        normalized = f"{normalized[:3]}/{normalized[3:]}"

    if normalized in FOREX_PAIRS:
        return FOREX_PAIRS[normalized]

    raise ValueError(f"Unknown currency pair: {symbol}")


def is_forex_pair(symbol: str) -> bool:
    """
    Check if a symbol is a configured Forex pair.

    Args:
        symbol: Symbol to check (any format)

    Returns:
        True if it's a Forex pair
    """
    try:
        get_pair_config(symbol)
        return True
    except ValueError:
        return False


def is_jpy_pair(symbol: str) -> bool:
    """
    Check if a symbol involves JPY (uses 2 decimal pip).

    Args:
        symbol: Currency pair symbol

    Returns:
        True if JPY pair
    """
    return "JPY" in symbol.upper()


def get_current_session() -> str:
    """
    Get the current trading session based on UTC time.

    Returns:
        Session name (sydney, tokyo, london, new_york, or overlap)
    """
    from datetime import datetime
    hour = datetime.utcnow().hour

    active_sessions = []
    for session, times in TRADING_SESSIONS.items():
        if times["open"] <= times["close"]:
            if times["open"] <= hour < times["close"]:
                active_sessions.append(session)
        else:
            # Crosses midnight
            if hour >= times["open"] or hour < times["close"]:
                active_sessions.append(session)

    if len(active_sessions) > 1:
        return f"{active_sessions[0]}_{active_sessions[-1]}_overlap"
    elif active_sessions:
        return active_sessions[0]
    else:
        return "closed"
