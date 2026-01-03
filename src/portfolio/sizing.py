"""
Position Sizing Algorithms (Lean-Inspired)
==========================================
Various position sizing strategies for portfolio construction.

Features:
- Equal weight allocation
- Risk parity
- Kelly Criterion
- Volatility targeting
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import PortfolioManager

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    symbol: str
    quantity: float
    weight: float           # Portfolio weight (0-1)
    value: float            # Dollar value
    reason: str = ""


class PositionSizer(ABC):
    """
    Abstract base class for position sizing algorithms.

    Implement custom sizing by subclassing.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate_size(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        price: float,
        **kwargs
    ) -> SizingResult:
        """
        Calculate position size for a symbol.

        Args:
            portfolio: Current portfolio state
            symbol: Target symbol
            price: Current market price
            **kwargs: Algorithm-specific parameters

        Returns:
            SizingResult with recommended size
        """
        pass

    @abstractmethod
    def calculate_weights(
        self,
        portfolio: 'PortfolioManager',
        symbols: List[str],
        prices: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate target weights for multiple symbols.

        Args:
            portfolio: Current portfolio state
            symbols: List of symbols to size
            prices: Current prices per symbol
            **kwargs: Algorithm-specific parameters

        Returns:
            Dictionary of symbol -> target weight
        """
        pass


class EqualWeightSizer(PositionSizer):
    """
    Equal Weight Position Sizing.

    Allocates equal weight to each position.
    Simple but effective diversification.
    """

    def __init__(self, max_positions: int = 10, reserve_cash: float = 0.05):
        """
        Args:
            max_positions: Maximum number of positions
            reserve_cash: Cash reserve (0.05 = 5%)
        """
        super().__init__("EqualWeight")
        self.max_positions = max_positions
        self.reserve_cash = reserve_cash

    def calculate_size(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        price: float,
        **kwargs
    ) -> SizingResult:
        """Calculate equal weight position size."""
        # Available capital (minus reserve)
        available = portfolio.total_value * (1 - self.reserve_cash)

        # Weight per position
        weight = 1.0 / self.max_positions

        # Position value and quantity
        position_value = available * weight
        quantity = position_value / price if price > 0 else 0

        # Round appropriately
        if price > 1:
            quantity = int(quantity)
        else:
            quantity = round(quantity, 4)

        return SizingResult(
            symbol=symbol,
            quantity=quantity,
            weight=weight,
            value=quantity * price,
            reason=f"Equal weight: {weight:.1%} of portfolio"
        )

    def calculate_weights(
        self,
        portfolio: 'PortfolioManager',
        symbols: List[str],
        prices: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """Calculate equal weights for all symbols."""
        n = min(len(symbols), self.max_positions)
        if n == 0:
            return {}

        weight = (1 - self.reserve_cash) / n
        return {symbol: weight for symbol in symbols[:n]}


class RiskParitySizer(PositionSizer):
    """
    Risk Parity Position Sizing.

    Allocates based on inverse volatility so each position
    contributes equal risk to the portfolio.

    Requires historical volatility data.
    """

    def __init__(
        self,
        target_risk: float = 0.15,  # 15% annual volatility target
        min_weight: float = 0.02,    # 2% minimum weight
        max_weight: float = 0.30,    # 30% maximum weight
    ):
        """
        Args:
            target_risk: Target portfolio volatility
            min_weight: Minimum position weight
            max_weight: Maximum position weight
        """
        super().__init__("RiskParity")
        self.target_risk = target_risk
        self.min_weight = min_weight
        self.max_weight = max_weight

    def calculate_size(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        price: float,
        volatility: float = 0.20,  # Default 20% volatility
        **kwargs
    ) -> SizingResult:
        """Calculate risk parity position size."""
        if volatility <= 0:
            volatility = 0.20

        # Inverse volatility weight
        weight = self.target_risk / volatility

        # Apply constraints
        weight = max(self.min_weight, min(weight, self.max_weight))

        # Calculate quantity
        position_value = portfolio.total_value * weight
        quantity = position_value / price if price > 0 else 0

        if price > 1:
            quantity = int(quantity)
        else:
            quantity = round(quantity, 4)

        return SizingResult(
            symbol=symbol,
            quantity=quantity,
            weight=weight,
            value=quantity * price,
            reason=f"Risk parity: vol={volatility:.1%}, weight={weight:.1%}"
        )

    def calculate_weights(
        self,
        portfolio: 'PortfolioManager',
        symbols: List[str],
        prices: Dict[str, float],
        volatilities: Dict[str, float] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate risk parity weights.

        Args:
            volatilities: Dictionary of symbol -> annualized volatility
        """
        volatilities = volatilities or {}

        # Get inverse volatilities
        inv_vols = {}
        for symbol in symbols:
            vol = volatilities.get(symbol, 0.20)  # Default 20%
            if vol > 0:
                inv_vols[symbol] = 1.0 / vol
            else:
                inv_vols[symbol] = 5.0  # 1/0.20

        # Normalize
        total_inv_vol = sum(inv_vols.values())
        if total_inv_vol == 0:
            return {s: 1.0 / len(symbols) for s in symbols}

        weights = {
            symbol: inv_vol / total_inv_vol
            for symbol, inv_vol in inv_vols.items()
        }

        # Apply constraints
        weights = {
            s: max(self.min_weight, min(w, self.max_weight))
            for s, w in weights.items()
        }

        # Re-normalize after constraints
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        return weights


class KellyCriterionSizer(PositionSizer):
    """
    Kelly Criterion Position Sizing.

    Maximizes long-term growth based on win rate and payoff ratio.
    Often used with fractional Kelly (e.g., half Kelly) for safety.

    Formula: f* = (bp - q) / b
    Where:
        f* = fraction to bet
        b = payoff ratio (avg win / avg loss)
        p = win probability
        q = lose probability (1 - p)
    """

    def __init__(
        self,
        fraction: float = 0.5,      # Half Kelly (safer)
        max_weight: float = 0.25,   # Cap at 25%
        min_trades: int = 30,       # Minimum trades for statistics
    ):
        """
        Args:
            fraction: Kelly fraction (0.5 = half Kelly)
            max_weight: Maximum position weight
            min_trades: Minimum trades before using Kelly
        """
        super().__init__("KellyCriterion")
        self.fraction = fraction
        self.max_weight = max_weight
        self.min_trades = min_trades

    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly fraction.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade (dollars or percent)
            avg_loss: Average losing trade (positive value)

        Returns:
            Kelly fraction (can be negative if edge is negative)
        """
        if avg_loss == 0:
            return 0.0

        b = avg_win / avg_loss  # Payoff ratio
        p = win_rate
        q = 1 - p

        kelly = (b * p - q) / b

        # Apply fractional Kelly
        return kelly * self.fraction

    def calculate_size(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        price: float,
        win_rate: float = 0.55,
        avg_win: float = 0.02,      # 2% average win
        avg_loss: float = 0.01,     # 1% average loss
        **kwargs
    ) -> SizingResult:
        """Calculate Kelly criterion position size."""
        kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)

        # Check for negative edge
        if kelly <= 0:
            return SizingResult(
                symbol=symbol,
                quantity=0,
                weight=0,
                value=0,
                reason=f"Negative edge: Kelly={kelly:.2%}"
            )

        # Apply max weight constraint
        weight = min(kelly, self.max_weight)

        # Calculate quantity
        position_value = portfolio.total_value * weight
        quantity = position_value / price if price > 0 else 0

        if price > 1:
            quantity = int(quantity)
        else:
            quantity = round(quantity, 4)

        return SizingResult(
            symbol=symbol,
            quantity=quantity,
            weight=weight,
            value=quantity * price,
            reason=f"Kelly={kelly:.2%} (applied {weight:.2%})"
        )

    def calculate_weights(
        self,
        portfolio: 'PortfolioManager',
        symbols: List[str],
        prices: Dict[str, float],
        statistics: Dict[str, dict] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate Kelly weights for multiple symbols.

        Args:
            statistics: Dict of symbol -> {win_rate, avg_win, avg_loss}
        """
        statistics = statistics or {}
        weights = {}

        for symbol in symbols:
            stats = statistics.get(symbol, {})
            win_rate = stats.get('win_rate', 0.55)
            avg_win = stats.get('avg_win', 0.02)
            avg_loss = stats.get('avg_loss', 0.01)

            kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)
            weights[symbol] = max(0, min(kelly, self.max_weight))

        # Normalize if total exceeds 1
        total = sum(weights.values())
        if total > 1:
            weights = {s: w / total for s, w in weights.items()}

        return weights


class VolatilityTargetSizer(PositionSizer):
    """
    Volatility Target Position Sizing.

    Sizes positions to achieve a target volatility contribution
    to the portfolio.

    Common in managed futures and CTAs.
    """

    def __init__(
        self,
        target_volatility: float = 0.10,  # 10% target vol
        vol_lookback: int = 20,           # Days for vol calculation
        vol_scalar: float = 16.0,         # Annualization factor (sqrt(256))
    ):
        """
        Args:
            target_volatility: Target annual volatility
            vol_lookback: Days for volatility calculation
            vol_scalar: Annualization factor
        """
        super().__init__("VolatilityTarget")
        self.target_volatility = target_volatility
        self.vol_lookback = vol_lookback
        self.vol_scalar = vol_scalar

    def calculate_size(
        self,
        portfolio: 'PortfolioManager',
        symbol: str,
        price: float,
        current_volatility: float = 0.02,  # Daily volatility
        **kwargs
    ) -> SizingResult:
        """Calculate volatility-targeted position size."""
        # Annualize volatility
        annual_vol = current_volatility * self.vol_scalar

        if annual_vol <= 0:
            return SizingResult(
                symbol=symbol,
                quantity=0,
                weight=0,
                value=0,
                reason="Zero volatility"
            )

        # Target weight to achieve target volatility
        weight = self.target_volatility / annual_vol

        # Cap at 100% (no leverage)
        weight = min(weight, 1.0)

        # Calculate quantity
        position_value = portfolio.total_value * weight
        quantity = position_value / price if price > 0 else 0

        if price > 1:
            quantity = int(quantity)
        else:
            quantity = round(quantity, 4)

        return SizingResult(
            symbol=symbol,
            quantity=quantity,
            weight=weight,
            value=quantity * price,
            reason=f"Vol target: current={annual_vol:.1%}, weight={weight:.1%}"
        )

    def calculate_weights(
        self,
        portfolio: 'PortfolioManager',
        symbols: List[str],
        prices: Dict[str, float],
        volatilities: Dict[str, float] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate volatility-targeted weights.

        Args:
            volatilities: Dict of symbol -> daily volatility
        """
        volatilities = volatilities or {}
        weights = {}

        for symbol in symbols:
            daily_vol = volatilities.get(symbol, 0.02)
            annual_vol = daily_vol * self.vol_scalar

            if annual_vol > 0:
                weight = min(self.target_volatility / annual_vol, 1.0)
            else:
                weight = 0.10  # Default 10%

            weights[symbol] = weight

        # Normalize if total exceeds 1
        total = sum(weights.values())
        if total > 1:
            weights = {s: w / total for s, w in weights.items()}

        return weights


class CompositeSizer:
    """
    Composite Position Sizer.

    Combines multiple sizing algorithms with configurable weights.
    """

    def __init__(self):
        self._sizers: List[tuple] = []  # (sizer, weight)

    def add_sizer(self, sizer: PositionSizer, weight: float = 1.0):
        """Add a position sizer with weight."""
        self._sizers.append((sizer, weight))

    def calculate_weights(
        self,
        portfolio: 'PortfolioManager',
        symbols: List[str],
        prices: Dict[str, float],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate composite weights from all sizers.

        Returns weighted average of all sizers.
        """
        if not self._sizers:
            return {s: 1.0 / len(symbols) for s in symbols}

        # Collect weights from each sizer
        all_weights = []
        total_weight = 0

        for sizer, weight in self._sizers:
            try:
                w = sizer.calculate_weights(portfolio, symbols, prices, **kwargs)
                all_weights.append((w, weight))
                total_weight += weight
            except Exception as e:
                logger.error(f"Sizer {sizer.name} error: {e}")

        if not all_weights:
            return {s: 1.0 / len(symbols) for s in symbols}

        # Weighted average
        final_weights = {s: 0.0 for s in symbols}
        for weights, sizer_weight in all_weights:
            for symbol in symbols:
                final_weights[symbol] += weights.get(symbol, 0) * (sizer_weight / total_weight)

        return final_weights
