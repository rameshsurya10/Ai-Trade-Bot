#!/usr/bin/env python3
"""
Strategy Analysis Script
========================

Analyzes your trading history to discover and rank strategies.

Usage:
    python scripts/analyze_strategies.py

Output:
    - Prints strategy comparison table
    - Shows best strategy
    - Saves detailed report to strategy_analysis.txt
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.learning.strategy_analyzer import StrategyAnalyzer

def main():
    print("="*70)
    print("STRATEGY ANALYZER")
    print("="*70)
    print()

    # Initialize analyzer
    analyzer = StrategyAnalyzer(database_path="data/trading.db")

    # Discover strategies from last 365 days
    print("🔍 Discovering strategies from historical data...")
    strategies = analyzer.discover_strategies(lookback_days=365)

    if not strategies:
        print("❌ No strategies found. Need at least 10 trades in database.")
        print("   Run the system for a while to collect trade data.")
        return

    print(f"✅ Discovered {len(strategies)} distinct strategies\n")

    # Show comparison table
    print("📊 STRATEGY COMPARISON TABLE")
    print("-"*70)
    comparison = analyzer.get_strategy_comparison_table()
    print(comparison.to_string(index=False))
    print()

    # Show best strategy
    best_name, best_strategy = analyzer.get_best_strategy(by='sharpe')
    print("="*70)
    print(f"🏆 BEST STRATEGY (by Sharpe Ratio): {best_name}")
    print("="*70)
    print(analyzer.get_strategy_report(best_name))

    # Show other rankings
    print("\n📈 RANKINGS BY DIFFERENT METRICS")
    print("-"*70)

    print("\n🥇 By Win Rate:")
    for i, (name, strategy) in enumerate(analyzer.rank_strategies('win_rate')[:3], 1):
        print(f"  {i}. {name}: {strategy.win_rate*100:.1f}%")

    print("\n💰 By Profit Factor:")
    for i, (name, strategy) in enumerate(analyzer.rank_strategies('profit_factor')[:3], 1):
        print(f"  {i}. {name}: {strategy.profit_factor:.2f}x")

    print("\n📊 By Total Expected Profit:")
    for i, (name, strategy) in enumerate(analyzer.rank_strategies('total_profit')[:3], 1):
        expected = strategy.avg_profit_pct * strategy.win_rate
        print(f"  {i}. {name}: {expected:.2f}% per trade")

    # Save detailed report
    print("\n💾 Saving detailed analysis...")
    analyzer.save_analysis("strategy_analysis.txt")
    print("✅ Report saved to: strategy_analysis.txt")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
