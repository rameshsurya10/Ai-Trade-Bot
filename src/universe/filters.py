"""
Universe Filters (Lean-Inspired)
================================
Filter algorithms for universe selection.

Similar to Lean's CoarseFilter and FineFilter functions.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Callable

from .manager import SecurityInfo

logger = logging.getLogger(__name__)


class UniverseFilter(ABC):
    """
    Abstract base class for universe filters.

    Filters reduce the candidate universe based on specific criteria.
    """

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    @abstractmethod
    def apply(self, candidates: List[SecurityInfo]) -> List[SecurityInfo]:
        """
        Apply filter to candidates.

        Args:
            candidates: List of candidate securities

        Returns:
            Filtered list of candidates
        """
        pass


class VolumeFilter(UniverseFilter):
    """
    Filter by trading volume.

    Ensures adequate liquidity for trading.
    """

    def __init__(
        self,
        min_volume: float = 100000,
        max_volume: float = None,
        volume_field: str = 'volume_24h'
    ):
        """
        Args:
            min_volume: Minimum 24h volume
            max_volume: Maximum 24h volume (optional)
            volume_field: Field name for volume data
        """
        super().__init__("VolumeFilter")
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.volume_field = volume_field

    def apply(self, candidates: List[SecurityInfo]) -> List[SecurityInfo]:
        """Filter by volume range."""
        result = []
        for c in candidates:
            volume = getattr(c, self.volume_field, 0)

            if volume < self.min_volume:
                continue

            if self.max_volume and volume > self.max_volume:
                continue

            result.append(c)

        return result


class PriceFilter(UniverseFilter):
    """
    Filter by price range.

    Excludes penny stocks or overly expensive assets.
    """

    def __init__(
        self,
        min_price: float = 0.0,
        max_price: float = float('inf')
    ):
        """
        Args:
            min_price: Minimum price
            max_price: Maximum price
        """
        super().__init__("PriceFilter")
        self.min_price = min_price
        self.max_price = max_price

    def apply(self, candidates: List[SecurityInfo]) -> List[SecurityInfo]:
        """Filter by price range."""
        return [
            c for c in candidates
            if self.min_price <= c.price <= self.max_price
        ]


class VolatilityFilter(UniverseFilter):
    """
    Filter by volatility.

    Select assets with volatility in target range.
    """

    def __init__(
        self,
        min_volatility: float = 0.0,
        max_volatility: float = 1.0,
        sort_by_volatility: bool = False,
        descending: bool = True
    ):
        """
        Args:
            min_volatility: Minimum annualized volatility
            max_volatility: Maximum annualized volatility
            sort_by_volatility: Sort results by volatility
            descending: Sort order (True = highest first)
        """
        super().__init__("VolatilityFilter")
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.sort_by_volatility = sort_by_volatility
        self.descending = descending

    def apply(self, candidates: List[SecurityInfo]) -> List[SecurityInfo]:
        """Filter by volatility range."""
        result = [
            c for c in candidates
            if self.min_volatility <= c.volatility <= self.max_volatility
        ]

        if self.sort_by_volatility:
            result.sort(key=lambda x: x.volatility, reverse=self.descending)

        return result


class MomentumFilter(UniverseFilter):
    """
    Filter by momentum.

    Selects top momentum assets.
    Used for momentum-based strategies.
    """

    def __init__(
        self,
        top_n: int = 10,
        min_momentum: float = None,
        lookback_days: int = 20
    ):
        """
        Args:
            top_n: Number of top momentum assets to select
            min_momentum: Minimum momentum threshold
            lookback_days: Lookback period for momentum calc
        """
        super().__init__("MomentumFilter")
        self.top_n = top_n
        self.min_momentum = min_momentum
        self.lookback_days = lookback_days

    def apply(self, candidates: List[SecurityInfo]) -> List[SecurityInfo]:
        """Filter and sort by momentum."""
        # Apply minimum threshold if set
        if self.min_momentum is not None:
            candidates = [c for c in candidates if c.momentum >= self.min_momentum]

        # Sort by momentum (descending)
        candidates.sort(key=lambda x: x.momentum, reverse=True)

        # Take top N
        return candidates[:self.top_n]


class LiquidityFilter(UniverseFilter):
    """
    Filter by liquidity score.

    Combines volume and spread for overall liquidity assessment.
    """

    def __init__(
        self,
        min_score: float = 0.5,
        top_n: int = None
    ):
        """
        Args:
            min_score: Minimum liquidity score (0-1)
            top_n: Optional limit to top N by liquidity
        """
        super().__init__("LiquidityFilter")
        self.min_score = min_score
        self.top_n = top_n

    def apply(self, candidates: List[SecurityInfo]) -> List[SecurityInfo]:
        """Filter by liquidity score."""
        # Filter by minimum score
        result = [c for c in candidates if c.liquidity_score >= self.min_score]

        # Sort by liquidity (descending)
        result.sort(key=lambda x: x.liquidity_score, reverse=True)

        # Apply top N limit if set
        if self.top_n:
            result = result[:self.top_n]

        return result


class MarketCapFilter(UniverseFilter):
    """
    Filter by market capitalization.

    Used for large-cap / mid-cap / small-cap selection.
    """

    def __init__(
        self,
        min_market_cap: float = 0,
        max_market_cap: float = float('inf'),
        top_n: int = None
    ):
        """
        Args:
            min_market_cap: Minimum market cap
            max_market_cap: Maximum market cap
            top_n: Optional limit to top N by market cap
        """
        super().__init__("MarketCapFilter")
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.top_n = top_n

    def apply(self, candidates: List[SecurityInfo]) -> List[SecurityInfo]:
        """Filter by market cap range."""
        result = [
            c for c in candidates
            if self.min_market_cap <= c.market_cap <= self.max_market_cap
        ]

        # Sort by market cap (descending)
        result.sort(key=lambda x: x.market_cap, reverse=True)

        if self.top_n:
            result = result[:self.top_n]

        return result


class TagFilter(UniverseFilter):
    """
    Filter by tags.

    Select assets with specific tags (sector, category, etc.).
    """

    def __init__(
        self,
        include_tags: List[str] = None,
        exclude_tags: List[str] = None,
        require_all: bool = False
    ):
        """
        Args:
            include_tags: Tags that must be present
            exclude_tags: Tags that must not be present
            require_all: Require all include_tags (vs any)
        """
        super().__init__("TagFilter")
        self.include_tags = include_tags or []
        self.exclude_tags = exclude_tags or []
        self.require_all = require_all

    def apply(self, candidates: List[SecurityInfo]) -> List[SecurityInfo]:
        """Filter by tags."""
        result = []

        for c in candidates:
            tags_set = set(c.tags)

            # Check exclusions
            if any(tag in tags_set for tag in self.exclude_tags):
                continue

            # Check inclusions
            if self.include_tags:
                if self.require_all:
                    if not all(tag in tags_set for tag in self.include_tags):
                        continue
                else:
                    if not any(tag in tags_set for tag in self.include_tags):
                        continue

            result.append(c)

        return result


class CustomFilter(UniverseFilter):
    """
    Custom filter using user-provided function.

    Allows arbitrary filtering logic.
    """

    def __init__(
        self,
        name: str,
        filter_func: Callable[[SecurityInfo], bool],
        sort_key: Callable[[SecurityInfo], float] = None,
        descending: bool = True
    ):
        """
        Args:
            name: Filter name
            filter_func: Function returning True for accepted securities
            sort_key: Optional sort key function
            descending: Sort order
        """
        super().__init__(name)
        self.filter_func = filter_func
        self.sort_key = sort_key
        self.descending = descending

    def apply(self, candidates: List[SecurityInfo]) -> List[SecurityInfo]:
        """Apply custom filter function."""
        result = [c for c in candidates if self.filter_func(c)]

        if self.sort_key:
            result.sort(key=self.sort_key, reverse=self.descending)

        return result


class CompositeFilter(UniverseFilter):
    """
    Composite filter combining multiple filters.

    Applies filters in sequence (pipeline).
    """

    def __init__(self, name: str = "CompositeFilter"):
        super().__init__(name)
        self._filters: List[UniverseFilter] = []

    def add(self, filter_: UniverseFilter) -> 'CompositeFilter':
        """Add filter to pipeline (chainable)."""
        self._filters.append(filter_)
        return self

    def apply(self, candidates: List[SecurityInfo]) -> List[SecurityInfo]:
        """Apply all filters in sequence."""
        result = candidates

        for filter_ in self._filters:
            if filter_.enabled:
                result = filter_.apply(result)

        return result


# =========================================================================
# PRESET FILTER COMBINATIONS
# =========================================================================

def create_crypto_filter(
    min_volume: float = 1000000,
    min_momentum: float = 0.0,
    top_n: int = 20
) -> CompositeFilter:
    """
    Create standard crypto filter.

    Filters for volume, then selects top momentum.
    """
    return (CompositeFilter("CryptoFilter")
            .add(VolumeFilter(min_volume=min_volume))
            .add(MomentumFilter(top_n=top_n, min_momentum=min_momentum)))


def create_equity_filter(
    min_price: float = 5.0,
    min_volume: float = 500000,
    min_market_cap: float = 1e9,  # $1B
    top_n: int = 50
) -> CompositeFilter:
    """
    Create standard equity filter.

    Filters penny stocks, illiquid stocks, and small caps.
    """
    return (CompositeFilter("EquityFilter")
            .add(PriceFilter(min_price=min_price))
            .add(VolumeFilter(min_volume=min_volume))
            .add(MarketCapFilter(min_market_cap=min_market_cap, top_n=top_n)))


def create_volatility_trading_filter(
    min_volatility: float = 0.20,
    max_volatility: float = 0.80,
    min_volume: float = 100000,
    top_n: int = 10
) -> CompositeFilter:
    """
    Create filter for volatility trading strategies.

    Selects moderately volatile, liquid assets.
    """
    return (CompositeFilter("VolatilityTradingFilter")
            .add(VolumeFilter(min_volume=min_volume))
            .add(VolatilityFilter(
                min_volatility=min_volatility,
                max_volatility=max_volatility,
                sort_by_volatility=True
            ))
            .add(LiquidityFilter(top_n=top_n)))
