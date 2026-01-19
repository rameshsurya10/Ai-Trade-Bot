#!/usr/bin/env python3
"""
System Verification Script
===========================

Verifies that all components are properly set up and ready to use.
"""

import sys
from pathlib import Path

def check_file(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size
        print(f"✅ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT FOUND")
        return False

def check_database():
    """Check database and show stats."""
    import sqlite3

    db_path = "data/trading.db"

    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check candles
    cursor.execute("SELECT COUNT(*) FROM candles")
    candle_count = cursor.fetchone()[0]
    print(f"✅ Database has {candle_count:,} candles")

    # Check trade outcomes
    cursor.execute("SELECT COUNT(*) FROM trade_outcomes")
    trade_count = cursor.fetchone()[0]
    print(f"✅ Database has {trade_count:,} trade outcomes")

    # Check recent trades
    if trade_count > 0:
        cursor.execute("""
            SELECT symbol, COUNT(*) as count,
                   AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) * 100 as win_rate
            FROM trade_outcomes
            GROUP BY symbol
        """)

        print("\n📊 Trade Summary by Symbol:")
        for row in cursor.fetchall():
            symbol, count, win_rate = row
            print(f"   {symbol}: {count} trades, {win_rate:.1f}% win rate")

    conn.close()
    return True

def main():
    print("="*70)
    print("AI TRADE BOT - SYSTEM VERIFICATION")
    print("="*70)
    print()

    print("📁 Checking Core Files...")
    print("-"*70)

    files_ok = True

    # Core system files
    files_ok &= check_file("run_trading.py", "Main trading script")
    files_ok &= check_file("dashboard_simple.py", "Clean dashboard")
    files_ok &= check_file("scripts/analyze_strategies.py", "Strategy analyzer")

    print()
    print("📁 Checking Learning System Components...")
    print("-"*70)

    files_ok &= check_file("src/learning/strategic_learning_bridge.py", "Strategic Learning Bridge")
    files_ok &= check_file("src/learning/continuous_learner.py", "Continuous Learning System")
    files_ok &= check_file("src/learning/strategy_analyzer.py", "Strategy Analyzer")
    files_ok &= check_file("src/learning/retraining_engine.py", "Retraining Engine")
    files_ok &= check_file("src/learning/outcome_tracker.py", "Outcome Tracker")

    print()
    print("📁 Checking Documentation...")
    print("-"*70)

    files_ok &= check_file("LIVE_CANDLE_TRAINING_FLOW.md", "Training flow guide")
    files_ok &= check_file("STRATEGY_DISCOVERY_GUIDE.md", "Strategy guide")
    files_ok &= check_file("COMPLETE_SYSTEM_GUIDE.md", "Complete system guide")
    files_ok &= check_file("SIMPLE_SOLUTION.md", "Simple solution guide")

    print()
    print("💾 Checking Database...")
    print("-"*70)

    db_ok = check_database()

    print()
    print("="*70)

    if files_ok and db_ok:
        print("✅ SYSTEM READY!")
        print("="*70)
        print()
        print("Next steps:")
        print()
        print("1. Start trading bot:")
        print("   python run_trading.py")
        print()
        print("2. View dashboard (optional):")
        print("   streamlit run dashboard.py")
        print()
        print("3. Analyze strategies (after 50+ trades):")
        print("   python scripts/analyze_strategies.py")
        print()
        return 0
    else:
        print("❌ SYSTEM NOT READY")
        print("="*70)
        print()
        print("Some components are missing. Please check errors above.")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
