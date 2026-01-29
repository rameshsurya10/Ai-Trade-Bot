"""
Strategy Analyzer & Comparator
================================

Analyzes trading patterns from 1-year data and identifies distinct strategies.

Features:
1. Discovers strategies from historical trade outcomes
2. Names strategies based on behavior patterns
3. Compares strategy performance
4. Ranks strategies by win rate, profit, risk-adjusted returns
5. Identifies best strategy per market regime

Strategy Types Discovered:
- Momentum Breakout (rides strong trends)
- Mean Reversion (buys dips, sells peaks)
- Scalping (quick in-and-out)
- Swing Trading (holds 4-24 hours)
- Trend Following (follows long-term direction)
- Counter-Trend (fades extremes)
- Volatility Expansion (trades high volatility)
- Range Trading (trades sideways markets)
"""

import logging
import sqlite3
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Represents a discovered trading strategy."""
    name: str
    description: str
    total_trades: int
    win_rate: float
    avg_profit_pct: float
    avg_loss_pct: float
    profit_factor: float  # Gross profit / Gross loss
    sharpe_ratio: float
    max_drawdown_pct: float
    avg_holding_hours: float
    best_timeframe: str
    best_regime: str
    confidence_threshold: float
    pattern_signature: str  # What pattern it looks for


class StrategyAnalyzer:
    """
    Analyzes historical trades to discover and rank strategies.

    Process:
    1. Load all trade outcomes from database
    2. Cluster trades by behavior (holding time, entry conditions, regime)
    3. Name each cluster based on characteristics
    4. Calculate performance metrics per strategy
    5. Rank strategies by multiple criteria
    """

    def __init__(self, database_path: str, config_path: str = "config.yaml"):
        self.db_path = database_path
        self.strategies: Dict[str, Strategy] = {}

        # Load configuration
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from yaml file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('strategy_analysis', {})
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}. Using defaults.")
            return {}

    def discover_strategies(self, lookback_days: int = 365) -> Dict[str, Strategy]:
        """
        Discover strategies from historical trade data.

        Args:
            lookback_days: How far back to analyze (default: 1 year)

        Returns:
            Dict of strategy_name -> Strategy
        """
        logger.info(f"Discovering strategies from last {lookback_days} days...")

        # Load trade outcomes
        trades_df = self._load_trade_outcomes(lookback_days)

        # Get minimum thresholds from config
        min_total_trades = self.config.get('min_total_trades', 10)
        min_trades_per_strategy = self.config.get('min_trades_per_strategy', 3)

        if len(trades_df) < min_total_trades:
            logger.warning(f"Only {len(trades_df)} trades found. Need at least {min_total_trades} for analysis.")
            return {}

        logger.info(f"Analyzing {len(trades_df)} trades...")

        # Classify trades into strategy types (vectorized for performance)
        trades_df['strategy_type'] = self._classify_trades_vectorized(trades_df)

        # Calculate metrics per strategy type
        strategies = {}

        for strategy_name in trades_df['strategy_type'].unique():
            strategy_trades = trades_df[trades_df['strategy_type'] == strategy_name]

            if len(strategy_trades) < min_trades_per_strategy:
                logger.debug(f"Skipping {strategy_name}: only {len(strategy_trades)} trades (need {min_trades_per_strategy})")
                continue

            strategy = self._calculate_strategy_metrics(strategy_name, strategy_trades)
            strategies[strategy_name] = strategy

        self.strategies = strategies
        logger.info(f"Discovered {len(strategies)} distinct strategies")

        return strategies

    def _load_trade_outcomes(self, lookback_days: int) -> pd.DataFrame:
        """Load trade outcomes from database."""
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()

        query = """
        SELECT
            symbol,
            interval,
            entry_price,
            exit_price,
            entry_time,
            exit_time,
            predicted_direction,
            predicted_confidence,
            was_correct,
            pnl_percent,
            regime,
            is_paper_trade
        FROM trade_outcomes
        WHERE entry_time >= ?
        ORDER BY entry_time ASC
        """

        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=(cutoff_date,))

                if len(df) > 0:
                    # Calculate holding time
                    df['entry_time'] = pd.to_datetime(df['entry_time'])
                    df['exit_time'] = pd.to_datetime(df['exit_time'])
                    df['holding_hours'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600

                return df
        except Exception as e:
            logger.error(f"Error loading trade outcomes: {e}")
            return pd.DataFrame()

    def _classify_trades_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized version of trade classification for performance.
        10-50x faster than .apply(axis=1) approach.
        """
        result = pd.Series('General Strategy', index=df.index)

        # Get config thresholds once
        classification = self.config.get('classification', {})
        scalping_hours = classification.get('scalping_hours', 1)
        momentum_conf = classification.get('momentum_confidence', 0.85)
        momentum_min = classification.get('momentum_min_hours', 1)
        momentum_max = classification.get('momentum_max_hours', 4)
        swing_min = classification.get('swing_min_hours', 4)
        swing_max = classification.get('swing_max_hours', 24)
        position_min = classification.get('position_min_hours', 24)

        # Scalping: < 1 hour
        mask_scalping = df['holding_hours'] < scalping_hours
        result[mask_scalping] = 'Scalping'

        # Momentum Breakout: High confidence + short hold
        mask_momentum = (
            (df['predicted_confidence'] >= momentum_conf) &
            (df['holding_hours'] >= momentum_min) &
            (df['holding_hours'] < momentum_max)
        )
        result[mask_momentum] = 'Momentum Breakout'

        # Swing Trading: Medium hold
        mask_swing = (df['holding_hours'] >= swing_min) & (df['holding_hours'] <= swing_max)
        has_regime = 'regime' in df.columns
        if has_regime:
            result[mask_swing & (df['regime'] == 'TRENDING')] = 'Swing Trend Following'
            result[mask_swing & (df['regime'] != 'TRENDING')] = 'Swing Mean Reversion'
        else:
            result[mask_swing] = 'Swing Trading'

        # Position Trading: Long hold
        mask_position = df['holding_hours'] > position_min
        result[mask_position] = 'Position Trading'

        # Regime-based strategies (only if not already classified)
        if has_regime:
            unclassified = result == 'General Strategy'
            result[unclassified & (df['regime'] == 'VOLATILE')] = 'Volatility Expansion'
            result[unclassified & (df['regime'] == 'CHOPPY')] = 'Range Trading'
            result[unclassified & (df['regime'] == 'TRENDING')] = 'Trend Following'

        return result

    def _calculate_strategy_metrics(self, name: str, trades: pd.DataFrame) -> Strategy:
        """Calculate comprehensive metrics for a strategy."""

        # Basic metrics
        total_trades = len(trades)
        wins = trades['was_correct'].sum()
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Profit metrics
        winning_trades = trades[trades['was_correct']]
        losing_trades = trades[~trades['was_correct']]

        avg_profit = winning_trades['pnl_percent'].mean() if len(winning_trades) > 0 else 0.0
        avg_loss = abs(losing_trades['pnl_percent'].mean()) if len(losing_trades) > 0 else 0.0

        gross_profit = winning_trades['pnl_percent'].sum() if len(winning_trades) > 0 else 0.0
        gross_loss = abs(losing_trades['pnl_percent'].sum()) if len(losing_trades) > 0 else 0.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Risk-adjusted returns (Sharpe Ratio)
        returns = trades['pnl_percent'].values
        trading_days = self.config.get('metrics', {}).get('sharpe_trading_days', 252)
        sharpe = (returns.mean() / returns.std() * np.sqrt(trading_days)) if returns.std() > 0 else 0.0

        # Max drawdown
        cumulative_returns = (1 + returns / 100).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        # Holding time
        avg_holding = trades['holding_hours'].mean()

        # Best performing timeframe
        if 'interval' in trades.columns:
            timeframe_performance = trades.groupby('interval')['was_correct'].mean()
            best_timeframe = timeframe_performance.idxmax() if len(timeframe_performance) > 0 else 'unknown'
        else:
            best_timeframe = 'unknown'

        # Best regime
        if 'regime' in trades.columns:
            regime_performance = trades.groupby('regime')['was_correct'].mean()
            best_regime = regime_performance.idxmax() if len(regime_performance) > 0 else 'NORMAL'
        else:
            best_regime = 'NORMAL'

        # Average confidence
        avg_confidence = trades['predicted_confidence'].mean() if 'predicted_confidence' in trades.columns else 0.0

        # Pattern signature
        pattern = self._generate_pattern_signature(name, trades)

        # Description
        description = self._generate_description(name, win_rate, avg_profit, avg_holding)

        return Strategy(
            name=name,
            description=description,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_profit_pct=avg_profit,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_drawdown,
            avg_holding_hours=avg_holding,
            best_timeframe=best_timeframe,
            best_regime=best_regime,
            confidence_threshold=avg_confidence,
            pattern_signature=pattern
        )

    def _generate_pattern_signature(self, name: str, trades: pd.DataFrame) -> str:
        """Generate human-readable pattern signature."""
        avg_confidence = trades['predicted_confidence'].mean()

        # Safely get dominant direction
        if 'predicted_direction' in trades.columns and len(trades) > 0:
            mode_series = trades['predicted_direction'].mode()
            dominant_direction = mode_series[0] if len(mode_series) > 0 else 'BOTH'
        else:
            dominant_direction = 'BOTH'

        signature = f"{name}: "

        if avg_confidence >= 0.85:
            signature += "High confidence (>85%), "
        elif avg_confidence >= 0.70:
            signature += "Medium confidence (70-85%), "
        else:
            signature += "Low confidence (<70%), "

        if dominant_direction == 'BUY':
            signature += "Long-biased, "
        elif dominant_direction == 'SELL':
            signature += "Short-biased, "
        else:
            signature += "Balanced, "

        avg_holding = trades['holding_hours'].mean()
        if avg_holding < 1:
            signature += "Ultra-short hold (<1h)"
        elif avg_holding < 4:
            signature += "Short hold (1-4h)"
        elif avg_holding < 24:
            signature += "Medium hold (4-24h)"
        else:
            signature += "Long hold (>24h)"

        return signature

    def _generate_description(self, name: str, win_rate: float, avg_profit: float, avg_holding: float) -> str:
        """Generate human-readable description."""
        descriptions = {
            "Scalping": "Rapid entries and exits, capitalizes on small price movements",
            "Momentum Breakout": "Enters on strong momentum signals, rides trend acceleration",
            "Swing Trend Following": "Follows multi-day trends, holds through minor pullbacks",
            "Swing Mean Reversion": "Buys dips and sells rallies in ranging markets",
            "Position Trading": "Long-term holds, focuses on major trend changes",
            "Volatility Expansion": "Trades breakouts during high volatility periods",
            "Range Trading": "Profits from price oscillation in sideways markets",
            "Trend Following": "Rides established trends until reversal signals",
            "General Strategy": "Mixed approach across various market conditions"
        }

        base_desc = descriptions.get(name, "Strategy derived from historical patterns")

        # Add performance context
        if win_rate >= 0.60:
            perf = "High win rate"
        elif win_rate >= 0.50:
            perf = "Moderate win rate"
        else:
            perf = "Needs improvement"

        return f"{base_desc}. {perf} ({win_rate*100:.1f}%), avg profit {avg_profit:.2f}%"

    def rank_strategies(self, by: str = 'sharpe') -> List[Tuple[str, Strategy]]:
        """
        Rank strategies by performance metric.

        Args:
            by: Metric to rank by ('sharpe', 'win_rate', 'profit_factor', 'total_profit')

        Returns:
            List of (strategy_name, Strategy) sorted by metric
        """
        if not self.strategies:
            logger.warning("No strategies to rank. Run discover_strategies() first.")
            return []

        ranking_key = {
            'sharpe': lambda s: s.sharpe_ratio,
            'win_rate': lambda s: s.win_rate,
            'profit_factor': lambda s: s.profit_factor,
            'total_profit': lambda s: s.avg_profit_pct * s.win_rate  # Expected value
        }.get(by, lambda s: s.sharpe_ratio)

        ranked = sorted(
            self.strategies.items(),
            key=lambda x: ranking_key(x[1]),
            reverse=True
        )

        return ranked

    def get_best_strategy(self, by: str = 'sharpe') -> Tuple[str, Strategy]:
        """Get single best strategy by metric."""
        ranked = self.rank_strategies(by=by)
        return ranked[0] if ranked else (None, None)

    def get_strategy_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table of all strategies."""
        if not self.strategies:
            return pd.DataFrame()

        data = []
        for name, strategy in self.strategies.items():
            data.append({
                'Strategy': name,
                'Trades': strategy.total_trades,
                'Win Rate': f"{strategy.win_rate*100:.1f}%",
                'Avg Profit': f"{strategy.avg_profit_pct:.2f}%",
                'Avg Loss': f"-{strategy.avg_loss_pct:.2f}%",
                'Profit Factor': f"{strategy.profit_factor:.2f}",
                'Sharpe Ratio': f"{strategy.sharpe_ratio:.2f}",
                'Max DD': f"-{strategy.max_drawdown_pct:.1f}%",
                'Avg Hold': f"{strategy.avg_holding_hours:.1f}h",
                'Best Timeframe': strategy.best_timeframe,
                'Best Regime': strategy.best_regime
            })

        df = pd.DataFrame(data)

        # Sort by Sharpe Ratio (descending)
        df = df.sort_values('Sharpe Ratio', ascending=False)

        return df

    def get_strategy_report(self, strategy_name: str) -> str:
        """Generate detailed report for a specific strategy."""
        if strategy_name not in self.strategies:
            return f"Strategy '{strategy_name}' not found."

        strategy = self.strategies[strategy_name]

        report = f"""
╔══════════════════════════════════════════════════════════════╗
  STRATEGY ANALYSIS: {strategy_name}
╚══════════════════════════════════════════════════════════════╝

Description:
  {strategy.description}

Pattern Signature:
  {strategy.pattern_signature}

Performance Metrics:
  Total Trades:       {strategy.total_trades}
  Win Rate:           {strategy.win_rate*100:.1f}%
  Average Profit:     +{strategy.avg_profit_pct:.2f}%
  Average Loss:       -{strategy.avg_loss_pct:.2f}%
  Profit Factor:      {strategy.profit_factor:.2f}x
  Sharpe Ratio:       {strategy.sharpe_ratio:.2f}
  Max Drawdown:       -{strategy.max_drawdown_pct:.1f}%

Behavior:
  Average Hold Time:  {strategy.avg_holding_hours:.1f} hours
  Best Timeframe:     {strategy.best_timeframe}
  Best Market Regime: {strategy.best_regime}
  Confidence Level:   {strategy.confidence_threshold*100:.1f}%

Risk Assessment:
  {"✅ LOW RISK" if strategy.max_drawdown_pct < 10 else "⚠️ MEDIUM RISK" if strategy.max_drawdown_pct < 20 else "🔴 HIGH RISK"}
  {"✅ PROFITABLE" if strategy.profit_factor > 1.5 else "⚠️ MARGINAL" if strategy.profit_factor > 1.0 else "❌ UNPROFITABLE"}
  {"✅ GOOD RISK/REWARD" if strategy.sharpe_ratio > 1.0 else "⚠️ MODERATE RISK/REWARD" if strategy.sharpe_ratio > 0 else "❌ POOR RISK/REWARD"}

Recommendation:
  {self._get_recommendation(strategy)}
"""
        return report

    def _get_recommendation(self, strategy: Strategy) -> str:
        """Generate recommendation for strategy usage."""
        if strategy.sharpe_ratio > 1.5 and strategy.win_rate > 0.55:
            return "🌟 EXCELLENT - Deploy with confidence in live trading"
        elif strategy.sharpe_ratio > 1.0 and strategy.win_rate > 0.50:
            return "✅ GOOD - Suitable for live trading with proper risk management"
        elif strategy.sharpe_ratio > 0.5 and strategy.win_rate > 0.45:
            return "⚠️ ACCEPTABLE - Use cautiously, monitor closely"
        elif strategy.win_rate > 0.50:
            return "⚠️ INCONSISTENT - High win rate but volatile returns, needs optimization"
        else:
            return "❌ NOT RECOMMENDED - Continue paper trading and model improvement"

    def save_analysis(self, output_file: str = "strategy_analysis.txt"):
        """Save complete analysis to file."""
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("STRATEGY ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")

            # Summary table
            f.write("STRATEGY COMPARISON TABLE\n")
            f.write("-"*70 + "\n")
            comparison = self.get_strategy_comparison_table()
            f.write(comparison.to_string(index=False))
            f.write("\n\n")

            # Detailed reports
            f.write("DETAILED STRATEGY REPORTS\n")
            f.write("="*70 + "\n")

            for strategy_name in self.strategies.keys():
                f.write(self.get_strategy_report(strategy_name))
                f.write("\n" + "="*70 + "\n")

        logger.info(f"Analysis saved to {output_file}")
