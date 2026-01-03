"""
Universe Selection Module (Lean-Inspired)
=========================================
Dynamic asset selection and filtering for trading.

Inspired by QuantConnect's Lean engine universe selection.

Features:
- Coarse universe filtering (volume, price)
- Fine universe selection (fundamentals)
- Custom filters (momentum, volatility)
- Scheduled rebalancing

Usage:
    from src.universe import UniverseManager, VolumeFilter, MomentumFilter

    universe = UniverseManager()

    # Add filters
    universe.add_filter(VolumeFilter(min_volume=1000000))
    universe.add_filter(MomentumFilter(lookback=20, top_n=10))

    # Get selected assets
    selected = universe.select(candidates)
"""

from .manager import (
    UniverseManager,
    SecurityType,
    UniverseSettings,
)
from .filters import (
    UniverseFilter,
    VolumeFilter,
    PriceFilter,
    VolatilityFilter,
    MomentumFilter,
    LiquidityFilter,
    CompositeFilter,
)

__all__ = [
    # Manager
    'UniverseManager',
    'SecurityType',
    'UniverseSettings',
    # Filters
    'UniverseFilter',
    'VolumeFilter',
    'PriceFilter',
    'VolatilityFilter',
    'MomentumFilter',
    'LiquidityFilter',
    'CompositeFilter',
]
