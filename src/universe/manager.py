"""
Universe Manager (Lean-Inspired)
================================
Manages tradable asset universe with dynamic selection.

Features:
- Asset candidate management
- Filter pipeline
- Scheduled rebalancing
- Universe change tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Set

logger = logging.getLogger(__name__)


class SecurityType(Enum):
    """Asset security types."""
    EQUITY = "equity"           # Stocks
    CRYPTO = "crypto"           # Cryptocurrency
    FOREX = "forex"             # Currency pairs
    FUTURES = "futures"         # Futures contracts
    OPTIONS = "options"         # Options contracts
    INDEX = "index"             # Indices
    COMMODITY = "commodity"     # Commodities


@dataclass
class SecurityInfo:
    """Information about a tradable security."""
    symbol: str
    security_type: SecurityType
    exchange: str = ""
    base_currency: str = "USD"
    quote_currency: str = ""

    # Market data
    price: float = 0.0
    volume_24h: float = 0.0
    market_cap: float = 0.0

    # Calculated metrics
    volatility: float = 0.0
    momentum: float = 0.0
    liquidity_score: float = 0.0

    # Metadata
    last_updated: datetime = field(default_factory=datetime.utcnow)
    is_tradable: bool = True
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'type': self.security_type.value,
            'exchange': self.exchange,
            'price': self.price,
            'volume_24h': self.volume_24h,
            'market_cap': self.market_cap,
            'volatility': self.volatility,
            'momentum': self.momentum,
            'liquidity_score': self.liquidity_score,
            'is_tradable': self.is_tradable,
        }


@dataclass
class UniverseSettings:
    """Universe configuration settings."""
    # Selection
    max_securities: int = 20           # Max assets in universe
    min_securities: int = 1            # Min assets required

    # Rebalancing
    rebalance_interval_hours: int = 24  # Hours between rebalances
    rebalance_on_change: bool = True    # Rebalance when universe changes

    # Data requirements
    min_history_days: int = 30          # Minimum history required
    min_volume: float = 0.0             # Minimum 24h volume

    # Risk
    max_correlation: float = 0.8        # Max correlation between assets


class UniverseManager:
    """
    Universe Manager (Lean-Inspired).

    Manages the selection and filtering of tradable assets.

    Similar to Lean's Universe Selection framework but simplified:
    - AddUniverse() -> add_candidates()
    - CoarseFilter -> volume/price filters
    - FineFilter -> fundamental/momentum filters

    Example:
        # Initialize
        universe = UniverseManager(max_securities=10)

        # Add filter pipeline
        universe.add_filter(VolumeFilter(min_volume=100000))
        universe.add_filter(MomentumFilter(lookback=20, top_n=10))

        # Add candidates
        universe.add_candidates(crypto_symbols, SecurityType.CRYPTO)

        # Get selected universe
        selected = universe.get_universe()

        # Update with new data
        universe.update_metrics(symbol_data)

        # Reselect
        universe.reselect()
    """

    def __init__(
        self,
        settings: UniverseSettings = None,
        on_change: Callable[[List[str], List[str]], None] = None
    ):
        """
        Initialize universe manager.

        Args:
            settings: Universe configuration
            on_change: Callback for universe changes (added, removed)
        """
        self.settings = settings or UniverseSettings()

        # Candidates and selection
        self._candidates: Dict[str, SecurityInfo] = {}
        self._selected: Set[str] = set()
        self._filters: List['UniverseFilter'] = []

        # Change tracking
        self._last_rebalance: Optional[datetime] = None
        self._change_callbacks: List[Callable] = []
        if on_change:
            self._change_callbacks.append(on_change)

        # History
        self._selection_history: List[tuple] = []  # (timestamp, selected)

        logger.info("UniverseManager initialized")

    # =========================================================================
    # FILTER MANAGEMENT
    # =========================================================================

    def add_filter(self, filter_: 'UniverseFilter'):
        """Add a filter to the pipeline."""
        from .filters import UniverseFilter
        if not isinstance(filter_, UniverseFilter):
            raise TypeError("Filter must be UniverseFilter instance")
        self._filters.append(filter_)
        logger.info(f"Added filter: {filter_.name}")

    def remove_filter(self, name: str):
        """Remove a filter by name."""
        self._filters = [f for f in self._filters if f.name != name]

    def clear_filters(self):
        """Remove all filters."""
        self._filters.clear()

    # =========================================================================
    # CANDIDATE MANAGEMENT
    # =========================================================================

    def add_candidates(
        self,
        symbols: List[str],
        security_type: SecurityType,
        exchange: str = ""
    ):
        """
        Add candidate securities.

        Args:
            symbols: List of ticker symbols
            security_type: Type of security
            exchange: Exchange name
        """
        for symbol in symbols:
            if symbol not in self._candidates:
                self._candidates[symbol] = SecurityInfo(
                    symbol=symbol,
                    security_type=security_type,
                    exchange=exchange
                )
        logger.info(f"Added {len(symbols)} {security_type.value} candidates")

    def add_candidate(self, info: SecurityInfo):
        """Add a single candidate with full info."""
        self._candidates[info.symbol] = info

    def remove_candidate(self, symbol: str):
        """Remove a candidate."""
        if symbol in self._candidates:
            del self._candidates[symbol]
            self._selected.discard(symbol)

    def update_candidate(self, symbol: str, **kwargs):
        """Update candidate metrics."""
        if symbol in self._candidates:
            info = self._candidates[symbol]
            for key, value in kwargs.items():
                if hasattr(info, key):
                    setattr(info, key, value)
            info.last_updated = datetime.utcnow()

    def update_metrics(self, data: Dict[str, dict]):
        """
        Batch update metrics for multiple securities.

        Args:
            data: Dict of symbol -> {price, volume, volatility, ...}
        """
        for symbol, metrics in data.items():
            if symbol in self._candidates:
                self.update_candidate(symbol, **metrics)

    @property
    def candidates(self) -> List[SecurityInfo]:
        """Get all candidates."""
        return list(self._candidates.values())

    @property
    def candidate_count(self) -> int:
        """Get candidate count."""
        return len(self._candidates)

    # =========================================================================
    # UNIVERSE SELECTION
    # =========================================================================

    def select(self, force: bool = False) -> List[str]:
        """
        Run selection to determine active universe.

        Args:
            force: Force reselection even if interval not elapsed

        Returns:
            List of selected symbols
        """
        # Check if rebalance needed
        if not force and self._last_rebalance:
            elapsed = datetime.utcnow() - self._last_rebalance
            if elapsed < timedelta(hours=self.settings.rebalance_interval_hours):
                return list(self._selected)

        # Start with all tradable candidates
        candidates = [c for c in self._candidates.values() if c.is_tradable]

        logger.info(f"Selecting from {len(candidates)} candidates")

        # Apply filters in order
        for filter_ in self._filters:
            candidates = filter_.apply(candidates)
            logger.debug(f"After {filter_.name}: {len(candidates)} candidates")

        # Apply max limit
        if len(candidates) > self.settings.max_securities:
            candidates = candidates[:self.settings.max_securities]

        # Check minimum
        if len(candidates) < self.settings.min_securities:
            logger.warning(
                f"Only {len(candidates)} candidates, min is {self.settings.min_securities}"
            )

        # Get old and new for change detection
        old_selected = self._selected.copy()
        new_selected = {c.symbol for c in candidates}

        # Update selection
        self._selected = new_selected
        self._last_rebalance = datetime.utcnow()

        # Record history
        self._selection_history.append((datetime.utcnow(), list(new_selected)))

        # Notify changes
        added = new_selected - old_selected
        removed = old_selected - new_selected

        if added or removed:
            self._notify_change(list(added), list(removed))

        logger.info(f"Selected {len(self._selected)} securities")
        return list(self._selected)

    def reselect(self) -> List[str]:
        """Force reselection."""
        return self.select(force=True)

    def get_universe(self) -> List[str]:
        """Get current universe (selected symbols)."""
        if not self._selected:
            self.select()
        return list(self._selected)

    def get_universe_info(self) -> List[SecurityInfo]:
        """Get SecurityInfo for selected symbols."""
        return [self._candidates[s] for s in self._selected if s in self._candidates]

    def is_selected(self, symbol: str) -> bool:
        """Check if symbol is in current universe."""
        return symbol in self._selected

    # =========================================================================
    # CHANGE TRACKING
    # =========================================================================

    def on_change(self, callback: Callable[[List[str], List[str]], None]):
        """Register callback for universe changes."""
        self._change_callbacks.append(callback)

    def _notify_change(self, added: List[str], removed: List[str]):
        """Notify callbacks of universe change."""
        logger.info(f"Universe changed: +{len(added)} -{len(removed)}")

        for callback in self._change_callbacks:
            try:
                callback(added, removed)
            except Exception as e:
                logger.error(f"Change callback error: {e}")

    # =========================================================================
    # PREDEFINED UNIVERSES
    # =========================================================================

    @classmethod
    def crypto_top_20(cls) -> 'UniverseManager':
        """Create universe with top 20 cryptocurrencies."""
        universe = cls(UniverseSettings(max_securities=20))

        # Top crypto pairs
        symbols = [
            "BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "SOL/USD",
            "ADA/USD", "DOGE/USD", "DOT/USD", "MATIC/USD", "SHIB/USD",
            "LTC/USD", "AVAX/USD", "LINK/USD", "ATOM/USD", "UNI/USD",
            "XLM/USD", "ETC/USD", "FIL/USD", "NEAR/USD", "ALGO/USD"
        ]
        universe.add_candidates(symbols, SecurityType.CRYPTO, "binance")

        return universe

    @classmethod
    def forex_majors(cls) -> 'UniverseManager':
        """Create universe with major forex pairs."""
        universe = cls(UniverseSettings(max_securities=10))

        symbols = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
            "EUR/JPY", "GBP/JPY"
        ]
        universe.add_candidates(symbols, SecurityType.FOREX)

        return universe

    @classmethod
    def us_tech(cls) -> 'UniverseManager':
        """Create universe with US tech stocks."""
        universe = cls(UniverseSettings(max_securities=15))

        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "AMD", "INTC", "CRM",
            "NFLX", "ADBE", "ORCL", "CSCO", "IBM"
        ]
        universe.add_candidates(symbols, SecurityType.EQUITY, "nasdaq")

        return universe

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> dict:
        """Get universe manager status."""
        return {
            'candidates': self.candidate_count,
            'selected': len(self._selected),
            'max_securities': self.settings.max_securities,
            'filters': [f.name for f in self._filters],
            'last_rebalance': self._last_rebalance.isoformat() if self._last_rebalance else None,
            'symbols': list(self._selected),
        }

    def __repr__(self) -> str:
        return f"UniverseManager(candidates={self.candidate_count}, selected={len(self._selected)})"
