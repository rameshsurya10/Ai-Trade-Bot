#!/usr/bin/env python3
"""
Run Backtest
============

Validates the trading strategy against historical data.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --start 2024-01-01 --end 2024-06-01

Output:
    - Win rate
    - Total PnL
    - Sharpe ratio
    - Max drawdown
    - And more...

CRITICAL: Run this BEFORE using the bot with real money!
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from datetime import datetime

from src.backtesting import BacktestEngine, BacktestMetrics
from src.backtesting.engine import BacktestConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Backtest trading strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --start 2024-01-01
    python scripts/run_backtest.py --start 2024-01-01 --end 2024-06-01
    python scripts/run_backtest.py --max-positions 3
        """
    )

    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--start',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--max-positions',
        type=int,
        default=1,
        help='Maximum concurrent positions'
    )
    parser.add_argument(
        '--slippage',
        type=float,
        default=0.05,
        help='Slippage percentage per trade'
    )
    parser.add_argument(
        '--commission',
        type=float,
        default=0.1,
        help='Commission percentage per trade'
    )
    parser.add_argument(
        '--max-hold',
        type=int,
        default=24,
        help='Maximum candles to hold a position'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--save',
        help='Save results to file (CSV)'
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("BACKTEST - Strategy Validation")
    print("=" * 60)
    print()

    # Build config
    bt_config = BacktestConfig(
        max_open_positions=args.max_positions,
        slippage_percent=args.slippage,
        commission_percent=args.commission,
        max_hold_candles=args.max_hold,
    )

    if args.start:
        bt_config.start_date = datetime.fromisoformat(args.start)
        print(f"Start Date: {args.start}")

    if args.end:
        bt_config.end_date = datetime.fromisoformat(args.end)
        print(f"End Date:   {args.end}")

    print(f"Max Positions: {args.max_positions}")
    print(f"Slippage:      {args.slippage}%")
    print(f"Commission:    {args.commission}%")
    print(f"Max Hold:      {args.max_hold} candles")
    print()

    # Create engine
    engine = BacktestEngine(args.config, bt_config)

    # Load data
    print("Loading historical data...")
    if not engine.load_data():
        print("ERROR: Failed to load data. Run 'python scripts/download_data.py' first.")
        sys.exit(1)

    # Progress callback
    def progress(current, total):
        if not args.quiet:
            pct = current / total * 100 if total > 0 else 0
            bar_len = 40
            filled = int(bar_len * current / total) if total > 0 else 0
            bar = '=' * filled + '-' * (bar_len - filled)
            print(f"\rBacktesting: [{bar}] {pct:.1f}%", end="", flush=True)

    # Run backtest
    print("\nRunning backtest...")
    results = engine.run(progress_callback=None if args.quiet else progress)

    if not args.quiet:
        print("\n")

    # Print results
    print(results.summary())

    # Save trades if requested
    if args.save:
        trades_df = engine.get_trade_details()
        if not trades_df.empty:
            trades_df.to_csv(args.save, index=False)
            print(f"\nTrades saved to: {args.save}")

    # Final assessment
    print()
    if results.total_trades >= 30:
        if results.win_rate >= 0.52 and results.profit_factor >= 1.2:
            print("RECOMMENDATION: Strategy shows promise. Consider paper trading.")
        elif results.win_rate >= 0.48:
            print("RECOMMENDATION: Strategy is marginal. More data/tuning needed.")
        else:
            print("RECOMMENDATION: Strategy is losing. Do NOT use with real money.")
    else:
        print("RECOMMENDATION: Need more data for reliable assessment (30+ trades).")

    print()


if __name__ == "__main__":
    main()
