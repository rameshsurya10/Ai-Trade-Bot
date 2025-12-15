#!/usr/bin/env python3
"""
Performance Report
==================

Generates a performance report for real trading signals.

Usage:
    python scripts/performance_report.py
    python scripts/performance_report.py --days 30

Shows:
    - Win rate
    - Total PnL
    - Best/worst trades
    - Recent performance
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

from src.core.config import Config
from src.tracking import PerformanceTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)


def main():
    parser = argparse.ArgumentParser(description='Generate performance report')
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Days to analyze for recent performance'
    )

    args = parser.parse_args()

    # Load config
    config = Config.load(args.config)

    # Create tracker
    tracker = PerformanceTracker(config)

    # Generate and print report
    print(tracker.generate_report())

    # Also print recent period if requested
    if args.days != 7:
        print(f"\n--- Custom Period: Last {args.days} Days ---")
        recent = tracker.get_recent_performance(args.days)
        print(f"Signals:  {recent['total_signals']}")
        print(f"Resolved: {recent['resolved']}")
        print(f"Win Rate: {recent['win_rate']:.1%}")
        print(f"Total PnL: {recent['total_pnl']:+.2f}%")


if __name__ == "__main__":
    main()
