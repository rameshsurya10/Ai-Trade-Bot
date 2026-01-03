"""
Paper Trading Validation Monitor
=================================

Real-time monitoring dashboard for paper trading validation.

Displays:
- Current mode (LEARNING/TRADING)
- Latest confidence levels
- Portfolio value and P&L
- Win rate and trade statistics
- Safety check status
- Recent trades

Usage:
    python tests/monitor_paper_trading.py
"""

import sys
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import Database
from src.core.config import load_config


class PaperTradingMonitor:
    """Real-time monitoring for paper trading validation."""

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize monitor."""
        self.config = load_config(config_path)
        self.db = Database(self.config['database']['path'])

        # Monitoring state
        self.last_update = datetime.utcnow()
        self.refresh_interval = 5  # seconds

    def run(self):
        """Run monitoring dashboard."""
        print("Paper Trading Validation Monitor")
        print("Press Ctrl+C to stop")
        print("="*80 + "\n")

        try:
            while True:
                self._clear_screen()
                self._display_dashboard()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

    def _clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _display_dashboard(self):
        """Display monitoring dashboard."""
        now = datetime.utcnow()

        print("="*80)
        print(f"PAPER TRADING VALIDATION MONITOR - {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*80)

        # Get latest learning state
        try:
            learning_state = self._get_latest_learning_state()
            if learning_state:
                print(f"\n📊 CURRENT STATUS")
                print(f"  Mode: {learning_state['mode']}")
                print(f"  Confidence: {learning_state.get('confidence_score', 0):.2%}")
                print(f"  Since: {learning_state.get('entered_at', 'Unknown')}")
            else:
                print("\n📊 CURRENT STATUS: No data yet")
        except Exception as e:
            print(f"\n📊 CURRENT STATUS: Error - {e}")

        # Get portfolio summary
        try:
            portfolio = self._get_portfolio_summary()
            print(f"\n💰 PORTFOLIO")
            print(f"  Value: ${portfolio['total_value']:.2f}")
            print(f"  P&L: ${portfolio['total_pnl']:.2f} ({portfolio['total_pnl_pct']:.2f}%)")
            print(f"  Cash: ${portfolio['cash']:.2f}")
        except Exception as e:
            print(f"\n💰 PORTFOLIO: Error - {e}")

        # Get trade statistics
        try:
            trade_stats = self._get_trade_statistics()
            print(f"\n📈 TRADE STATISTICS")
            print(f"  Total Trades: {trade_stats['total_trades']}")
            print(f"  Wins: {trade_stats['wins']} | Losses: {trade_stats['losses']}")
            print(f"  Win Rate: {trade_stats['win_rate']:.2%}")
            print(f"  Avg P&L per Trade: ${trade_stats['avg_pnl']:.2f}")
        except Exception as e:
            print(f"\n📈 TRADE STATISTICS: Error - {e}")

        # Get recent signals
        try:
            recent_signals = self._get_recent_signals(limit=5)
            print(f"\n🎯 RECENT SIGNALS ({len(recent_signals)})")
            for signal in recent_signals[:5]:
                print(
                    f"  [{signal['datetime']}] {signal['symbol']}: "
                    f"{signal['direction']} @ {signal['confidence']:.2%} "
                    f"({'Paper' if signal.get('is_paper') else 'Live'})"
                )
        except Exception as e:
            print(f"\n🎯 RECENT SIGNALS: Error - {e}")

        # Get retraining history
        try:
            retrainings = self._get_recent_retrainings(limit=3)
            print(f"\n🔄 RECENT RETRAININGS ({len(retrainings)})")
            for retrain in retrainings[:3]:
                status_icon = "✓" if retrain['status'] == 'success' else "✗"
                print(
                    f"  {status_icon} [{retrain['triggered_at']}] "
                    f"{retrain['symbol']} @ {retrain['interval']}: "
                    f"{retrain['trigger_reason']} → "
                    f"{retrain.get('validation_confidence', 0):.2%}"
                )
        except Exception as e:
            print(f"\n🔄 RECENT RETRAININGS: Error - {e}")

        # Safety checks
        try:
            safety_status = self._check_safety_limits(portfolio, trade_stats)
            print(f"\n🛡️  SAFETY CHECKS")
            for check_name, check_result in safety_status.items():
                icon = "✓" if check_result['passed'] else "⚠"
                print(f"  {icon} {check_name}: {check_result['message']}")
        except Exception as e:
            print(f"\n🛡️  SAFETY CHECKS: Error - {e}")

        print("\n" + "="*80)
        print(f"Last Update: {now.strftime('%H:%M:%S')} | Refresh: {self.refresh_interval}s")

    def _get_latest_learning_state(self) -> dict:
        """Get latest learning state."""
        # Query database for latest learning state
        query = """
            SELECT mode, confidence_score, entered_at, symbol, interval
            FROM learning_states
            ORDER BY id DESC
            LIMIT 1
        """

        result = self.db.execute_query(query)
        if result:
            return {
                'mode': result[0][0],
                'confidence_score': result[0][1],
                'entered_at': result[0][2],
                'symbol': result[0][3],
                'interval': result[0][4]
            }
        return None

    def _get_portfolio_summary(self) -> dict:
        """Get portfolio summary."""
        # Get paper brokerage positions
        query = """
            SELECT
                SUM(CASE WHEN status = 'open' THEN quantity * current_price ELSE 0 END) as position_value,
                SUM(CASE WHEN status = 'closed' THEN pnl ELSE 0 END) as realized_pnl
            FROM positions
        """

        result = self.db.execute_query(query)

        if result:
            position_value = result[0][0] or 0.0
            realized_pnl = result[0][1] or 0.0
        else:
            position_value = 0.0
            realized_pnl = 0.0

        initial_capital = self.config['portfolio']['initial_capital']
        cash = initial_capital + realized_pnl - position_value
        total_value = cash + position_value

        return {
            'total_value': total_value,
            'cash': cash,
            'position_value': position_value,
            'total_pnl': total_value - initial_capital,
            'total_pnl_pct': ((total_value - initial_capital) / initial_capital) * 100
        }

    def _get_trade_statistics(self) -> dict:
        """Get trade statistics."""
        query = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as losses,
                AVG(pnl_percent) as avg_pnl_pct,
                AVG(pnl_absolute) as avg_pnl
            FROM trade_outcomes
            WHERE exit_time IS NOT NULL
        """

        result = self.db.execute_query(query)

        if result and result[0][0]:
            total_trades = result[0][0]
            wins = result[0][1] or 0
            losses = result[0][2] or 0
            avg_pnl = result[0][4] or 0.0

            win_rate = wins / total_trades if total_trades > 0 else 0.0

            return {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl
            }

        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0
        }

    def _get_recent_signals(self, limit: int = 10) -> list:
        """Get recent trading signals."""
        query = """
            SELECT datetime, symbol, direction, confidence, is_paper
            FROM signals
            ORDER BY timestamp DESC
            LIMIT ?
        """

        result = self.db.execute_query(query, (limit,))

        if result:
            return [
                {
                    'datetime': row[0],
                    'symbol': row[1],
                    'direction': row[2],
                    'confidence': row[3],
                    'is_paper': row[4]
                }
                for row in result
            ]

        return []

    def _get_recent_retrainings(self, limit: int = 5) -> list:
        """Get recent retraining events."""
        query = """
            SELECT triggered_at, symbol, interval, trigger_reason,
                   status, validation_confidence
            FROM retraining_history
            ORDER BY id DESC
            LIMIT ?
        """

        result = self.db.execute_query(query, (limit,))

        if result:
            return [
                {
                    'triggered_at': row[0],
                    'symbol': row[1],
                    'interval': row[2],
                    'trigger_reason': row[3],
                    'status': row[4],
                    'validation_confidence': row[5] or 0.0
                }
                for row in result
            ]

        return []

    def _check_safety_limits(self, portfolio: dict, trade_stats: dict) -> dict:
        """Check safety limits."""
        safety_checks = {}

        # Check 1: Drawdown
        max_dd_limit = 15.0  # 15%
        current_dd = abs(min(0, portfolio['total_pnl_pct']))

        safety_checks['Max Drawdown'] = {
            'passed': current_dd <= max_dd_limit,
            'message': f"{current_dd:.2f}% (limit: {max_dd_limit}%)"
        }

        # Check 2: Win Rate
        min_win_rate = 0.45  # 45%
        current_win_rate = trade_stats['win_rate']

        if trade_stats['total_trades'] >= 10:
            safety_checks['Win Rate'] = {
                'passed': current_win_rate >= min_win_rate,
                'message': f"{current_win_rate:.2%} (min: {min_win_rate:.2%})"
            }
        else:
            safety_checks['Win Rate'] = {
                'passed': True,
                'message': f"{current_win_rate:.2%} (insufficient trades)"
            }

        # Check 3: Portfolio Value
        min_value_pct = 0.85  # Don't lose more than 15% of capital
        initial_capital = self.config['portfolio']['initial_capital']
        min_value = initial_capital * min_value_pct

        safety_checks['Portfolio Value'] = {
            'passed': portfolio['total_value'] >= min_value,
            'message': f"${portfolio['total_value']:.2f} (min: ${min_value:.2f})"
        }

        return safety_checks


def main():
    """Run monitor."""
    monitor = PaperTradingMonitor()
    monitor.run()


if __name__ == '__main__':
    main()
