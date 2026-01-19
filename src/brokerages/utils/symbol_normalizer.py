"""
Symbol Normalizer
=================

Normalize currency pair symbols between different formats.

Formats:
- Standard: "EUR/USD" (internal format)
- Capital.com: "EURUSD" (no separator)
- MT4/MT5: "EURUSD" (no separator)
- Underscore: "EUR_USD" (underscore separator)

Usage:
    from src.brokerages.utils import SymbolNormalizer, to_capital_format

    normalizer = SymbolNormalizer()

    # Convert to standard format
    standard = normalizer.to_standard("EURUSD")
    # Returns: "EUR/USD"

    # Convert to Capital.com format
    capital = normalizer.to_capital("EUR/USD")
    # Returns: "EURUSD"

    # Helper functions
    capital_symbol = to_capital_format("EUR/USD")
    standard_symbol = from_capital_format("EURUSD")
"""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


# Standard Forex pairs (internal format)
STANDARD_PAIRS = {
    # Majors
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD",
    # Crosses
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "EUR/CHF",
    "EUR/AUD", "AUD/JPY", "CHF/JPY",
    # Additional common pairs
    "EUR/CAD", "GBP/CAD", "AUD/CAD", "NZD/CAD",
    "GBP/CHF", "AUD/CHF", "NZD/CHF",
    "GBP/AUD", "EUR/NZD", "GBP/NZD", "AUD/NZD",
    "CAD/JPY", "NZD/JPY",
}

# Currency codes
CURRENCY_CODES = {
    "EUR", "USD", "GBP", "JPY", "CHF",
    "AUD", "CAD", "NZD", "SGD", "HKD",
    "SEK", "NOK", "DKK", "PLN", "MXN",
    "ZAR", "TRY", "CNH", "INR",
}


class SymbolNormalizer:
    """
    Normalize currency pair symbols between different broker formats.

    Supports conversion between:
    - Standard format: EUR/USD (slash separator)
    - Capital.com format: EURUSD (no separator)
    - Underscore format: EUR_USD (underscore separator)
    - MT4/MT5 format: EURUSD (no separator)

    Example:
        normalizer = SymbolNormalizer()

        # All these return "EUR/USD"
        normalizer.to_standard("EUR_USD")
        normalizer.to_standard("EURUSD")
        normalizer.to_standard("EUR/USD")
    """

    def __init__(self):
        """Initialize symbol normalizer."""
        self._cache: Dict[str, str] = {}
        self._reverse_cache: Dict[str, Dict[str, str]] = {
            "capital": {},
            "mt4": {},
            "underscore": {},
        }

    def to_standard(self, symbol: str) -> str:
        """
        Convert any format to standard (EUR/USD).

        Args:
            symbol: Symbol in any format

        Returns:
            Symbol in standard format

        Example:
            to_standard("EUR_USD") -> "EUR/USD"
            to_standard("EURUSD") -> "EUR/USD"
        """
        if not symbol:
            return symbol

        cache_key = symbol.upper()
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._normalize_to_standard(symbol)
        self._cache[cache_key] = result
        return result

    def to_capital(self, symbol: str) -> str:
        """
        Convert to Capital.com format (EURUSD).

        Args:
            symbol: Symbol in any format

        Returns:
            Symbol in Capital.com format

        Example:
            to_capital("EUR/USD") -> "EURUSD"
        """
        standard = self.to_standard(symbol)
        return standard.replace("/", "")

    def to_underscore(self, symbol: str) -> str:
        """
        Convert to underscore format (EUR_USD).

        Args:
            symbol: Symbol in any format

        Returns:
            Symbol in underscore format

        Example:
            to_underscore("EUR/USD") -> "EUR_USD"
        """
        standard = self.to_standard(symbol)
        return standard.replace("/", "_")

    def to_mt4(self, symbol: str) -> str:
        """
        Convert to MT4/MT5 format (EURUSD).

        Args:
            symbol: Symbol in any format

        Returns:
            Symbol in MT4 format

        Example:
            to_mt4("EUR/USD") -> "EURUSD"
        """
        return self.to_capital(symbol)

    def to_compact(self, symbol: str) -> str:
        """
        Convert to compact format (EURUSD).

        Same as Capital.com/MT4 format.
        """
        return self.to_capital(symbol)

    def from_capital(self, symbol: str) -> str:
        """
        Convert from Capital.com format to standard.

        Args:
            symbol: Symbol in Capital.com format

        Returns:
            Symbol in standard format

        Example:
            from_capital("EURUSD") -> "EUR/USD"
        """
        return self.to_standard(symbol)

    def from_underscore(self, symbol: str) -> str:
        """
        Convert from underscore format to standard.

        Args:
            symbol: Symbol in underscore format

        Returns:
            Symbol in standard format

        Example:
            from_underscore("EUR_USD") -> "EUR/USD"
        """
        return self.to_standard(symbol)

    def is_valid_forex_pair(self, symbol: str) -> bool:
        """
        Check if symbol is a valid Forex pair.

        Args:
            symbol: Symbol to check

        Returns:
            True if valid Forex pair
        """
        try:
            standard = self.to_standard(symbol)
            return standard in STANDARD_PAIRS or self._looks_like_forex(standard)
        except ValueError:
            return False

    def get_currencies(self, symbol: str) -> Tuple[str, str]:
        """
        Extract base and quote currencies from symbol.

        Args:
            symbol: Currency pair

        Returns:
            Tuple of (base_currency, quote_currency)

        Example:
            get_currencies("EUR/USD") -> ("EUR", "USD")
        """
        standard = self.to_standard(symbol)
        parts = standard.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid symbol format: {symbol}")
        return (parts[0], parts[1])

    def is_jpy_pair(self, symbol: str) -> bool:
        """Check if pair involves JPY."""
        return "JPY" in symbol.upper()

    def get_quote_currency(self, symbol: str) -> str:
        """Get the quote currency (second currency)."""
        _, quote = self.get_currencies(symbol)
        return quote

    def get_base_currency(self, symbol: str) -> str:
        """Get the base currency (first currency)."""
        base, _ = self.get_currencies(symbol)
        return base

    def _normalize_to_standard(self, symbol: str) -> str:
        """Internal normalization logic."""
        symbol = symbol.strip().upper()

        # Already standard format
        if "/" in symbol:
            parts = symbol.split("/")
            if len(parts) == 2 and all(p in CURRENCY_CODES for p in parts):
                return symbol
            raise ValueError(f"Invalid symbol format: {symbol}")

        # Underscore format (EUR_USD)
        if "_" in symbol:
            parts = symbol.split("_")
            if len(parts) == 2 and all(p in CURRENCY_CODES for p in parts):
                return f"{parts[0]}/{parts[1]}"
            raise ValueError(f"Invalid symbol format: {symbol}")

        # Compact format (EURUSD) - 6 characters
        if len(symbol) == 6:
            base = symbol[:3]
            quote = symbol[3:]
            if base in CURRENCY_CODES and quote in CURRENCY_CODES:
                return f"{base}/{quote}"

        # Unknown format
        raise ValueError(f"Cannot parse symbol: {symbol}")

    def _looks_like_forex(self, standard_symbol: str) -> bool:
        """Check if symbol looks like a forex pair."""
        if "/" not in standard_symbol:
            return False
        parts = standard_symbol.split("/")
        return len(parts) == 2 and all(p in CURRENCY_CODES for p in parts)


# Singleton instance
_normalizer = SymbolNormalizer()


def normalize_forex_symbol(symbol: str) -> str:
    """
    Normalize forex symbol to standard format.

    Convenience function using singleton normalizer.

    Args:
        symbol: Symbol in any format

    Returns:
        Symbol in standard format (EUR/USD)
    """
    return _normalizer.to_standard(symbol)


def to_capital_format(symbol: str) -> str:
    """
    Convert symbol to Capital.com format.

    Convenience function using singleton normalizer.

    Args:
        symbol: Symbol in any format

    Returns:
        Symbol in Capital.com format (EURUSD)
    """
    return _normalizer.to_capital(symbol)


def from_capital_format(symbol: str) -> str:
    """
    Convert from Capital.com format to standard.

    Convenience function using singleton normalizer.

    Args:
        symbol: Symbol in Capital.com format

    Returns:
        Symbol in standard format (EUR/USD)
    """
    return _normalizer.from_capital(symbol)


def to_underscore_format(symbol: str) -> str:
    """
    Convert symbol to underscore format.

    Convenience function using singleton normalizer.

    Args:
        symbol: Symbol in any format

    Returns:
        Symbol in underscore format (EUR_USD)
    """
    return _normalizer.to_underscore(symbol)


def from_underscore_format(symbol: str) -> str:
    """
    Convert from underscore format to standard.

    Convenience function using singleton normalizer.

    Args:
        symbol: Symbol in underscore format

    Returns:
        Symbol in standard format (EUR/USD)
    """
    return _normalizer.from_underscore(symbol)


def get_symbol_normalizer() -> SymbolNormalizer:
    """Get singleton SymbolNormalizer instance."""
    return _normalizer
