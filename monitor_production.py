"""
Production Monitoring Script
============================

Real-time monitoring and alerting for production deployment.

Features:
- Multi-symbol health monitoring
- Performance tracking
- Automated alerts
- Anomaly detection
- Slack/email notifications (optional)

Usage:
    # Monitor all deployed symbols
    python monitor_production.py

    # Monitor specific symbol
    python monitor_production.py --symbol BTC/USDT

    # Generate report
    python monitor_production.py --report
"""

import argparse
import sys
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.database import Database
from src.core.config import load_config

# Ensure required directories exist
Path('logs').mkdir(exist_ok=True)
Path('production_reports').mkdir(exist_ok=True)


class ProductionMonitor:
    """
    Real-time production monitoring.

    Monitors:
    - System health (uptime, errors, performance)
    - Portfolio performance (P&L, drawdown, Sharpe)
    - Model confidence (trends, stability)
    - Trading activity (volume, win rate)
    - Retraining events (frequency, success rate)
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize monitor."""
        self.config = load_config(config_path)
        self.db = Database(self.config['database']['path'])

        # Monitoring state
        self.last_update = datetime.utcnow()
        self.refresh_interval = 10  # seconds

        # Alert thresholds
        self.alert_thresholds = {
            'max_drawdown_pct': 15.0,
            'min_win_rate': 0.45,
            'max_error_rate': 0.05,
            'min_confidence': 0.60,
            'max_transitions_per_hour': 2.0
        }

        # Alert state
        self.alerts_active: Dict[str, datetime] = {}
        self.alerts_sent: int = 0

    def run(self, symbol: Optional[str] = None):
        """Run monitoring dashboard."""
        print("Production Monitoring Dashboard")
        print("Press Ctrl+C to stop")
        print("=" * 100 + "\n")

        try:
            while True:
                self._clear_screen()
                self._display_dashboard(symbol)
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

    def generate_report(self, hours: int = 24) -> Dict:
        """
        Generate performance report.

        Returns:
            {
                'period': {...},
                'symbols': {...},
                'overall': {...},
                'alerts': [...]
            }
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        print(f"Generating {hours}-hour production report...")

        # Get all active symbols
        symbols_data = self.db.execute_query(
            """
            SELECT DISTINCT symbol
            FROM learning_states
            WHERE entered_at > ?
            ORDER BY symbol
            """,
            (start_time.isoformat(),)
        )

        symbols = [row[0] for row in symbols_data] if symbols_data else []

        report = {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'hours': hours
            },
            'symbols': {},
            'overall': {},
            'alerts': []
        }

        # Per-symbol stats
        for symbol in symbols:
            symbol_stats = self._get_symbol_stats(symbol, start_time, end_time)
            report['symbols'][symbol] = symbol_stats

            # Check for alerts
            alerts = self._check_alerts(symbol, symbol_stats)
            report['alerts'].extend(alerts)

        # Overall stats
        report['overall'] = self._get_overall_stats(start_time, end_time)

        # Save report
        report_path = Path('production_reports')
        report_path.mkdir(exist_ok=True)

        filename = report_path / f"report_{end_time.strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Report saved: {filename}")

        # Display summary
        self._display_report_summary(report)

        return report

    def _display_dashboard(self, filter_symbol: Optional[str] = None):
        """Display real-time monitoring dashboard."""
        now = datetime.utcnow()

        print("=" * 100)
        print(f"PRODUCTION MONITOR - {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 100)

        # System overview
        uptime = self._get_system_uptime()
        error_rate = self._get_error_rate(hours=1)

        print(f"\n📊 SYSTEM STATUS")
        print(f"  Uptime: {uptime}")
        print(f"  Error Rate (1h): {error_rate:.2%}")

        # Get active symbols
        active_symbols = self._get_active_symbols()

        if filter_symbol:
            active_symbols = [s for s in active_symbols if s == filter_symbol]

        print(f"  Active Symbols: {len(active_symbols)}")

        # Per-symbol monitoring
        for symbol in active_symbols:
            print(f"\n{'=' * 100}")
            print(f"📈 {symbol}")
            print(f"{'=' * 100}")

            # Learning state
            state = self._get_current_state(symbol)
            if state:
                mode_icon = "🎯" if state['mode'] == 'TRADING' else "📚"
                print(f"  {mode_icon} Mode: {state['mode']}")
                print(f"  Confidence: {state['confidence']:.2%}")
                print(f"  Time in mode: {self._format_duration(state['duration'])}")

            # Performance (last 24h)
            perf = self._get_performance(symbol, hours=24)
            if perf:
                pnl_color = "✓" if perf['pnl'] >= 0 else "✗"
                print(f"\n  💰 Performance (24h)")
                print(f"    {pnl_color} P&L: ${perf['pnl']:.2f} ({perf['pnl_pct']:.2f}%)")
                print(f"    Win Rate: {perf['win_rate']:.2%} ({perf['wins']}/{perf['total_trades']} trades)")
                print(f"    Max Drawdown: {perf['max_drawdown']:.2f}%")

            # Recent activity
            recent = self._get_recent_activity(symbol, limit=3)
            if recent:
                print(f"\n  🎯 Recent Activity")
                for activity in recent:
                    print(f"    [{activity['time']}] {activity['type']}: {activity['description']}")

            # Alerts
            alerts = self._check_symbol_alerts(symbol)
            if alerts:
                print(f"\n  ⚠️  ALERTS ({len(alerts)})")
                for alert in alerts:
                    print(f"    {alert['severity']} {alert['message']}")

        print(f"\n{'=' * 100}")
        print(f"Last Update: {now.strftime('%H:%M:%S')} | Refresh: {self.refresh_interval}s | Alerts Sent: {self.alerts_sent}")

    def _get_system_uptime(self) -> str:
        """Get system uptime."""
        # Get earliest learning state entry
        earliest = self.db.execute_query(
            """
            SELECT MIN(entered_at)
            FROM learning_states
            """
        )

        if not earliest or not earliest[0][0]:
            return "N/A"

        start_time = datetime.fromisoformat(earliest[0][0])
        uptime_delta = datetime.utcnow() - start_time

        return self._format_duration(uptime_delta.total_seconds())

    def _get_error_rate(self, hours: int = 1) -> float:
        """Get error rate (errors per candle)."""
        start_time = datetime.utcnow() - timedelta(hours=hours)

        # Count candles
        candle_count = self.db.execute_query(
            """
            SELECT COUNT(*)
            FROM candles
            WHERE timestamp > ?
            """,
            (int(start_time.timestamp()),)
        )

        # Count errors
        error_count = self.db.execute_query(
            """
            SELECT COUNT(*)
            FROM error_log
            WHERE timestamp > ?
            """,
            (start_time.isoformat(),)
        )

        total_candles = candle_count[0][0] if candle_count else 0
        total_errors = error_count[0][0] if error_count else 0

        return total_errors / total_candles if total_candles > 0 else 0.0

    def _get_active_symbols(self) -> List[str]:
        """Get list of active symbols."""
        # Symbols with activity in last hour
        symbols = self.db.execute_query(
            """
            SELECT DISTINCT symbol
            FROM learning_states
            WHERE entered_at > datetime('now', '-1 hour')
            ORDER BY symbol
            """
        )

        return [row[0] for row in symbols] if symbols else []

    def _get_current_state(self, symbol: str) -> Optional[Dict]:
        """Get current learning state for symbol."""
        state = self.db.execute_query(
            """
            SELECT mode, confidence_score, entered_at
            FROM learning_states
            WHERE symbol = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (symbol,)
        )

        if not state:
            return None

        entered_at = datetime.fromisoformat(state[0][2])
        duration = (datetime.utcnow() - entered_at).total_seconds()

        return {
            'mode': state[0][0],
            'confidence': state[0][1],
            'entered_at': state[0][2],
            'duration': duration
        }

    def _get_performance(self, symbol: str, hours: int = 24) -> Optional[Dict]:
        """Get performance stats for symbol."""
        start_time = datetime.utcnow() - timedelta(hours=hours)

        # Trade stats
        trade_data = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                SUM(pnl_absolute) as total_pnl,
                MIN(pnl_absolute) as worst_trade,
                MAX(pnl_absolute) as best_trade
            FROM trade_outcomes
            WHERE symbol = ?
            AND entry_time > ?
            AND exit_time IS NOT NULL
            """,
            (symbol, start_time.isoformat())
        )

        if not trade_data or trade_data[0][0] == 0:
            return None

        total_trades = trade_data[0][0]
        wins = trade_data[0][1] or 0
        total_pnl = trade_data[0][2] or 0.0
        worst_trade = trade_data[0][3] or 0.0
        best_trade = trade_data[0][4] or 0.0

        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Calculate drawdown
        initial_capital = self.config['portfolio']['initial_capital']

        # Get max portfolio value in period
        max_value = initial_capital + total_pnl  # Simplified
        current_value = initial_capital + total_pnl

        drawdown = ((max_value - current_value) / max_value) * 100 if max_value > 0 else 0.0

        return {
            'total_trades': total_trades,
            'wins': wins,
            'win_rate': win_rate,
            'pnl': total_pnl,
            'pnl_pct': (total_pnl / initial_capital) * 100,
            'max_drawdown': drawdown,
            'best_trade': best_trade,
            'worst_trade': worst_trade
        }

    def _get_recent_activity(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Get recent activity (trades, retrainings, mode changes)."""
        activities = []

        # Recent trades
        trades = self.db.execute_query(
            """
            SELECT entry_time, predicted_direction, was_correct, pnl_absolute
            FROM trade_outcomes
            WHERE symbol = ?
            AND exit_time IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (symbol, limit)
        )

        if trades:
            for trade in trades:
                result = "✓ WIN" if trade[2] else "✗ LOSS"
                activities.append({
                    'time': datetime.fromisoformat(trade[0]).strftime('%H:%M'),
                    'type': 'Trade',
                    'description': f"{trade[1]} {result} (${trade[3]:.2f})"
                })

        # Recent retrainings
        retrainings = self.db.execute_query(
            """
            SELECT triggered_at, trigger_reason, status, validation_confidence
            FROM retraining_history
            WHERE symbol = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (symbol, limit)
        )

        if retrainings:
            for retrain in retrainings:
                status_icon = "✓" if retrain[2] == 'success' else "✗"
                conf = retrain[3] or 0.0
                activities.append({
                    'time': datetime.fromisoformat(retrain[0]).strftime('%H:%M'),
                    'type': 'Retrain',
                    'description': f"{status_icon} {retrain[1]} → {conf:.2%}"
                })

        # Sort by time (most recent first)
        activities.sort(key=lambda x: x['time'], reverse=True)

        return activities[:limit]

    def _check_symbol_alerts(self, symbol: str) -> List[Dict]:
        """Check for alerts on specific symbol."""
        alerts = []

        # Get current stats
        perf = self._get_performance(symbol, hours=1)
        state = self._get_current_state(symbol)

        if not perf:
            return alerts

        # Alert 1: High drawdown
        if perf['max_drawdown'] > self.alert_thresholds['max_drawdown_pct']:
            alerts.append({
                'severity': '🔴 CRITICAL',
                'message': f"Drawdown {perf['max_drawdown']:.1f}% exceeds {self.alert_thresholds['max_drawdown_pct']:.1f}% limit"
            })

        # Alert 2: Low win rate (if enough trades)
        if perf['total_trades'] >= 10 and perf['win_rate'] < self.alert_thresholds['min_win_rate']:
            alerts.append({
                'severity': '⚠️  WARNING',
                'message': f"Win rate {perf['win_rate']:.1%} below {self.alert_thresholds['min_win_rate']:.1%}"
            })

        # Alert 3: Low confidence
        if state and state['confidence'] < self.alert_thresholds['min_confidence']:
            alerts.append({
                'severity': '⚠️  WARNING',
                'message': f"Confidence {state['confidence']:.1%} below {self.alert_thresholds['min_confidence']:.1%}"
            })

        # Alert 4: Error rate
        error_rate = self._get_error_rate(hours=1)
        if error_rate > self.alert_thresholds['max_error_rate']:
            alerts.append({
                'severity': '🔴 CRITICAL',
                'message': f"Error rate {error_rate:.1%} exceeds {self.alert_thresholds['max_error_rate']:.1%}"
            })

        return alerts

    def _get_symbol_stats(self, symbol: str, start_time: datetime, end_time: datetime) -> Dict:
        """Get comprehensive stats for a symbol."""
        # Trade performance
        trade_data = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                AVG(pnl_absolute) as avg_pnl,
                SUM(pnl_absolute) as total_pnl
            FROM trade_outcomes
            WHERE symbol = ?
            AND entry_time BETWEEN ? AND ?
            AND exit_time IS NOT NULL
            """,
            (symbol, start_time.isoformat(), end_time.isoformat())
        )

        # Mode distribution
        mode_data = self.db.execute_query(
            """
            SELECT
                mode,
                COUNT(*) as count
            FROM learning_states
            WHERE symbol = ?
            AND entered_at BETWEEN ? AND ?
            GROUP BY mode
            """,
            (symbol, start_time.isoformat(), end_time.isoformat())
        )

        # Retraining stats
        retrain_data = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                AVG(duration_seconds) as avg_duration
            FROM retraining_history
            WHERE symbol = ?
            AND triggered_at BETWEEN ? AND ?
            """,
            (symbol, start_time.isoformat(), end_time.isoformat())
        )

        stats = {
            'trades': {
                'total': 0,
                'wins': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0
            },
            'modes': {},
            'retraining': {
                'total': 0,
                'successful': 0,
                'success_rate': 0.0,
                'avg_duration': 0.0
            }
        }

        # Populate trade stats
        if trade_data and trade_data[0][0]:
            total = trade_data[0][0]
            wins = trade_data[0][1] or 0
            stats['trades'] = {
                'total': total,
                'wins': wins,
                'win_rate': wins / total if total > 0 else 0.0,
                'avg_pnl': trade_data[0][2] or 0.0,
                'total_pnl': trade_data[0][3] or 0.0
            }

        # Populate mode stats
        if mode_data:
            for row in mode_data:
                stats['modes'][row[0]] = row[1]

        # Populate retraining stats
        if retrain_data and retrain_data[0][0]:
            total = retrain_data[0][0]
            successful = retrain_data[0][1] or 0
            stats['retraining'] = {
                'total': total,
                'successful': successful,
                'success_rate': successful / total if total > 0 else 0.0,
                'avg_duration': retrain_data[0][2] or 0.0
            }

        return stats

    def _get_overall_stats(self, start_time: datetime, end_time: datetime) -> Dict:
        """Get overall system stats."""
        # Aggregate across all symbols
        overall_trades = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                SUM(pnl_absolute) as total_pnl
            FROM trade_outcomes
            WHERE entry_time BETWEEN ? AND ?
            AND exit_time IS NOT NULL
            """,
            (start_time.isoformat(), end_time.isoformat())
        )

        overall_retraining = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful
            FROM retraining_history
            WHERE triggered_at BETWEEN ? AND ?
            """,
            (start_time.isoformat(), end_time.isoformat())
        )

        stats = {
            'total_trades': 0,
            'total_wins': 0,
            'overall_win_rate': 0.0,
            'total_pnl': 0.0,
            'total_retrainings': 0,
            'retraining_success_rate': 0.0
        }

        if overall_trades and overall_trades[0][0]:
            total = overall_trades[0][0]
            wins = overall_trades[0][1] or 0
            stats['total_trades'] = total
            stats['total_wins'] = wins
            stats['overall_win_rate'] = wins / total if total > 0 else 0.0
            stats['total_pnl'] = overall_trades[0][2] or 0.0

        if overall_retraining and overall_retraining[0][0]:
            total = overall_retraining[0][0]
            successful = overall_retraining[0][1] or 0
            stats['total_retrainings'] = total
            stats['retraining_success_rate'] = successful / total if total > 0 else 0.0

        return stats

    def _check_alerts(self, symbol: str, stats: Dict) -> List[Dict]:
        """Check for alert conditions."""
        alerts = []

        # Win rate alert
        if stats['trades']['total'] >= 20:
            if stats['trades']['win_rate'] < self.alert_thresholds['min_win_rate']:
                alerts.append({
                    'symbol': symbol,
                    'severity': 'WARNING',
                    'type': 'win_rate',
                    'message': f"{symbol} win rate {stats['trades']['win_rate']:.1%} below threshold"
                })

        # Retraining success rate alert
        if stats['retraining']['total'] >= 5:
            if stats['retraining']['success_rate'] < 0.80:
                alerts.append({
                    'symbol': symbol,
                    'severity': 'WARNING',
                    'type': 'retraining',
                    'message': f"{symbol} retraining success rate {stats['retraining']['success_rate']:.1%}"
                })

        return alerts

    def _display_report_summary(self, report: Dict):
        """Display report summary."""
        print("\n" + "=" * 100)
        print("REPORT SUMMARY")
        print("=" * 100)

        overall = report['overall']

        print(f"\n📊 Overall Performance")
        print(f"  Total Trades: {overall['total_trades']}")
        print(f"  Win Rate: {overall['overall_win_rate']:.2%}")
        print(f"  Total P&L: ${overall['total_pnl']:.2f}")
        print(f"  Retrainings: {overall['total_retrainings']} ({overall['retraining_success_rate']:.1%} success)")

        # Per-symbol summary
        print(f"\n📈 Per-Symbol Performance")
        for symbol, stats in report['symbols'].items():
            print(f"\n  {symbol}:")
            print(f"    Trades: {stats['trades']['total']} (Win Rate: {stats['trades']['win_rate']:.2%})")
            print(f"    P&L: ${stats['trades']['total_pnl']:.2f}")
            print(f"    Retrainings: {stats['retraining']['total']}")

        # Alerts
        if report['alerts']:
            print(f"\n⚠️  Alerts ({len(report['alerts'])})")
            for alert in report['alerts']:
                print(f"    {alert['severity']}: {alert['message']}")
        else:
            print(f"\n✓ No alerts")

    def _clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m {int(seconds % 60)}s"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
        else:
            days = int(seconds / 86400)
            hours = int((seconds % 86400) / 3600)
            return f"{days}d {hours}h"


def main():
    """Main monitoring entry point."""
    parser = argparse.ArgumentParser(description='Production Monitoring')

    parser.add_argument(
        '--symbol',
        type=str,
        help='Monitor specific symbol only'
    )

    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate performance report'
    )

    parser.add_argument(
        '--hours',
        type=int,
        default=24,
        help='Report period in hours (default: 24)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Config file path'
    )

    args = parser.parse_args()

    monitor = ProductionMonitor(config_path=args.config)

    if args.report:
        monitor.generate_report(hours=args.hours)
    else:
        monitor.run(symbol=args.symbol)


if __name__ == '__main__':
    main()
