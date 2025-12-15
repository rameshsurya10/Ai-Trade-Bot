"""
Tests for Backtesting Module
============================

Tests backtest metrics calculation.
"""

import pytest
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.metrics import BacktestMetrics, calculate_metrics


class TestBacktestMetrics:
    """Test BacktestMetrics dataclass."""

    def test_default_values(self):
        metrics = BacktestMetrics()
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl_percent == 0.0

    def test_to_dict(self):
        metrics = BacktestMetrics(
            total_trades=10,
            winners=6,
            losers=4,
            win_rate=0.6,
            total_pnl_percent=5.0,
        )
        d = metrics.to_dict()
        assert d['total_trades'] == 10
        assert d['win_rate'] == 0.6

    def test_summary_generation(self):
        metrics = BacktestMetrics(
            total_trades=50,
            winners=28,
            losers=22,
            win_rate=0.56,
            total_pnl_percent=10.5,
            avg_pnl_percent=0.21,
            profit_factor=1.5,
            sharpe_ratio=1.2,
            max_drawdown=5.0,
        )
        summary = metrics.summary()
        assert "BACKTEST RESULTS" in summary
        assert "56.0%" in summary  # Win rate
        assert "10.50%" in summary  # Total PnL


class TestCalculateMetrics:
    """Test metrics calculation function."""

    def test_empty_trades(self):
        metrics = calculate_metrics([])
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0

    def test_all_winners(self):
        trades = [
            {'pnl_percent': 2.0, 'is_winner': True, 'duration_minutes': 60, 'strength': 'STRONG'},
            {'pnl_percent': 1.5, 'is_winner': True, 'duration_minutes': 90, 'strength': 'STRONG'},
            {'pnl_percent': 3.0, 'is_winner': True, 'duration_minutes': 45, 'strength': 'MEDIUM'},
        ]
        metrics = calculate_metrics(trades)

        assert metrics.total_trades == 3
        assert metrics.winners == 3
        assert metrics.losers == 0
        assert metrics.win_rate == 1.0
        assert metrics.total_pnl_percent == 6.5

    def test_all_losers(self):
        trades = [
            {'pnl_percent': -1.0, 'is_winner': False, 'duration_minutes': 60, 'strength': 'STRONG'},
            {'pnl_percent': -2.0, 'is_winner': False, 'duration_minutes': 90, 'strength': 'MEDIUM'},
        ]
        metrics = calculate_metrics(trades)

        assert metrics.total_trades == 2
        assert metrics.winners == 0
        assert metrics.losers == 2
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl_percent == -3.0

    def test_mixed_trades(self):
        trades = [
            {'pnl_percent': 2.0, 'is_winner': True, 'duration_minutes': 60, 'strength': 'STRONG'},
            {'pnl_percent': -1.0, 'is_winner': False, 'duration_minutes': 30, 'strength': 'STRONG'},
            {'pnl_percent': 1.5, 'is_winner': True, 'duration_minutes': 90, 'strength': 'MEDIUM'},
            {'pnl_percent': -0.5, 'is_winner': False, 'duration_minutes': 120, 'strength': 'MEDIUM'},
            {'pnl_percent': 3.0, 'is_winner': True, 'duration_minutes': 45, 'strength': 'STRONG'},
        ]
        metrics = calculate_metrics(trades)

        assert metrics.total_trades == 5
        assert metrics.winners == 3
        assert metrics.losers == 2
        assert metrics.win_rate == 0.6
        assert metrics.total_pnl_percent == 5.0  # 2 + 1.5 + 3 - 1 - 0.5

    def test_profit_factor(self):
        trades = [
            {'pnl_percent': 4.0, 'is_winner': True, 'duration_minutes': 60},  # Gross profit: 4
            {'pnl_percent': -2.0, 'is_winner': False, 'duration_minutes': 60},  # Gross loss: 2
        ]
        metrics = calculate_metrics(trades)

        # Profit factor = gross profit / gross loss = 4 / 2 = 2.0
        assert metrics.profit_factor == pytest.approx(2.0, rel=0.01)

    def test_max_drawdown(self):
        # Equity curve: 0 -> 2 -> 1 -> 3 -> 2 -> 4
        # Peak: 2, then 3, then 4
        # Drawdowns: 1 (from 2 to 1), 1 (from 3 to 2)
        trades = [
            {'pnl_percent': 2.0, 'is_winner': True, 'duration_minutes': 60},  # Equity: 2
            {'pnl_percent': -1.0, 'is_winner': False, 'duration_minutes': 60},  # Equity: 1, DD: 1
            {'pnl_percent': 2.0, 'is_winner': True, 'duration_minutes': 60},  # Equity: 3
            {'pnl_percent': -1.0, 'is_winner': False, 'duration_minutes': 60},  # Equity: 2, DD: 1
            {'pnl_percent': 2.0, 'is_winner': True, 'duration_minutes': 60},  # Equity: 4
        ]
        metrics = calculate_metrics(trades)
        assert metrics.max_drawdown == pytest.approx(1.0, rel=0.01)

    def test_streak_calculation(self):
        trades = [
            {'pnl_percent': 1.0, 'is_winner': True, 'duration_minutes': 60},
            {'pnl_percent': 1.0, 'is_winner': True, 'duration_minutes': 60},
            {'pnl_percent': 1.0, 'is_winner': True, 'duration_minutes': 60},
            {'pnl_percent': -1.0, 'is_winner': False, 'duration_minutes': 60},
            {'pnl_percent': -1.0, 'is_winner': False, 'duration_minutes': 60},
        ]
        metrics = calculate_metrics(trades)

        assert metrics.max_consecutive_winners == 3
        assert metrics.max_consecutive_losers == 2

    def test_signal_strength_win_rates(self):
        trades = [
            {'pnl_percent': 2.0, 'is_winner': True, 'strength': 'STRONG', 'duration_minutes': 60},
            {'pnl_percent': 1.0, 'is_winner': True, 'strength': 'STRONG', 'duration_minutes': 60},
            {'pnl_percent': -1.0, 'is_winner': False, 'strength': 'STRONG', 'duration_minutes': 60},
            {'pnl_percent': 1.0, 'is_winner': True, 'strength': 'MEDIUM', 'duration_minutes': 60},
            {'pnl_percent': -1.0, 'is_winner': False, 'strength': 'MEDIUM', 'duration_minutes': 60},
            {'pnl_percent': -1.0, 'is_winner': False, 'strength': 'MEDIUM', 'duration_minutes': 60},
        ]
        metrics = calculate_metrics(trades)

        # Strong: 2 wins, 1 loss = 66.7%
        assert metrics.strong_signal_win_rate == pytest.approx(2/3, rel=0.01)

        # Medium: 1 win, 2 losses = 33.3%
        assert metrics.medium_signal_win_rate == pytest.approx(1/3, rel=0.01)

    def test_expectancy(self):
        # Win rate 60%, avg win 2%, avg loss -1%
        trades = [
            {'pnl_percent': 2.0, 'is_winner': True, 'duration_minutes': 60},
            {'pnl_percent': 2.0, 'is_winner': True, 'duration_minutes': 60},
            {'pnl_percent': 2.0, 'is_winner': True, 'duration_minutes': 60},
            {'pnl_percent': -1.0, 'is_winner': False, 'duration_minutes': 60},
            {'pnl_percent': -1.0, 'is_winner': False, 'duration_minutes': 60},
        ]
        metrics = calculate_metrics(trades)

        # Expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
        # = (0.6 * 2.0) + (0.4 * -1.0)
        # = 1.2 - 0.4 = 0.8
        assert metrics.expectancy == pytest.approx(0.8, rel=0.1)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
