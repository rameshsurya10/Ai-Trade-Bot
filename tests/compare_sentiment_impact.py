"""
Sentiment Impact Comparison
============================

Compare performance with and without sentiment features.

Tests:
1. Technical indicators only (32 features)
2. Technical + Sentiment (39 features)

Measures:
- Win rate improvement
- P&L improvement
- Confidence levels
- Mode transition differences

Usage:
    python tests/compare_sentiment_impact.py --symbol BTC/USDT --days 90
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest_continuous_learning import ContinuousLearningBacktest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentImpactComparison:
    """Compare trading performance with/without sentiment."""

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize comparison engine."""
        self.config_path = config_path
        self.results_dir = Path('backtest_results/comparisons')
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_comparison(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Run two backtests and compare results.

        Args:
            symbol: Trading pair
            start_date: Backtest start
            end_date: Backtest end

        Returns:
            Comparison results dict
        """
        logger.info("="*80)
        logger.info("SENTIMENT IMPACT COMPARISON")
        logger.info("="*80)

        # Run backtest WITHOUT sentiment
        logger.info("\n[1/2] Running backtest WITHOUT sentiment features...")
        backtest_no_sentiment = ContinuousLearningBacktest(
            config_path=self.config_path,
            results_dir=str(self.results_dir / 'no_sentiment')
        )

        results_no_sentiment = backtest_no_sentiment.run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            use_sentiment=False
        )

        # Run backtest WITH sentiment
        logger.info("\n[2/2] Running backtest WITH sentiment features...")
        backtest_with_sentiment = ContinuousLearningBacktest(
            config_path=self.config_path,
            results_dir=str(self.results_dir / 'with_sentiment')
        )

        results_with_sentiment = backtest_with_sentiment.run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            use_sentiment=True
        )

        # Compare results
        comparison = self._compare_results(
            results_no_sentiment,
            results_with_sentiment
        )

        # Save comparison
        self._save_comparison(comparison)

        # Print comparison
        self._print_comparison(comparison)

        return comparison

    def _compare_results(
        self,
        no_sentiment: Dict,
        with_sentiment: Dict
    ) -> Dict:
        """Compare two backtest results."""
        # Performance comparison
        perf_no = no_sentiment['performance']
        perf_with = with_sentiment['performance']

        win_rate_improvement = (
            perf_with['win_rate'] - perf_no['win_rate']
        ) * 100  # Convert to percentage points

        pnl_improvement = (
            (perf_with['total_pnl'] - perf_no['total_pnl']) /
            abs(perf_no['total_pnl']) * 100
            if perf_no['total_pnl'] != 0 else 0
        )

        trades_diff = perf_with['trades_executed'] - perf_no['trades_executed']

        # Mode comparison
        modes_no = no_sentiment['modes']
        modes_with = with_sentiment['modes']

        trading_mode_diff = (
            modes_with['trading_percent'] - modes_no['trading_percent']
        )

        confidence_diff = (
            modes_with['avg_trading_confidence'] -
            modes_no['avg_trading_confidence']
        ) * 100

        comparison = {
            'metadata': {
                'symbol': no_sentiment['metadata']['symbol'],
                'period': f"{no_sentiment['metadata']['start_date']} to {no_sentiment['metadata']['end_date']}",
                'comparison_date': datetime.utcnow().isoformat()
            },
            'technical_only': {
                'win_rate': perf_no['win_rate'],
                'total_pnl': perf_no['total_pnl'],
                'trades': perf_no['trades_executed'],
                'trading_mode_pct': modes_no['trading_percent'],
                'avg_confidence': modes_no['avg_trading_confidence']
            },
            'technical_plus_sentiment': {
                'win_rate': perf_with['win_rate'],
                'total_pnl': perf_with['total_pnl'],
                'trades': perf_with['trades_executed'],
                'trading_mode_pct': modes_with['trading_percent'],
                'avg_confidence': modes_with['avg_trading_confidence']
            },
            'improvements': {
                'win_rate_improvement_pct': win_rate_improvement,
                'pnl_improvement_pct': pnl_improvement,
                'pnl_difference_usd': perf_with['total_pnl'] - perf_no['total_pnl'],
                'trades_difference': trades_diff,
                'trading_mode_diff_pct': trading_mode_diff,
                'confidence_diff_pct': confidence_diff
            },
            'verdict': self._generate_verdict(
                win_rate_improvement,
                pnl_improvement,
                trading_mode_diff
            )
        }

        return comparison

    def _generate_verdict(
        self,
        win_rate_improvement: float,
        pnl_improvement: float,
        trading_mode_diff: float
    ) -> Dict:
        """Generate verdict on sentiment impact."""
        # Criteria for positive impact
        criteria_met = 0
        total_criteria = 3

        if win_rate_improvement > 2.0:  # >2% improvement
            criteria_met += 1

        if pnl_improvement > 10.0:  # >10% P&L improvement
            criteria_met += 1

        if trading_mode_diff > 5.0:  # >5% more time in trading mode
            criteria_met += 1

        if criteria_met >= 2:
            impact = "POSITIVE"
            recommendation = "Sentiment features significantly improve trading performance"
        elif criteria_met == 1:
            impact = "MIXED"
            recommendation = "Sentiment features show some improvement but not conclusive"
        else:
            impact = "NEGATIVE"
            recommendation = "Sentiment features do not improve performance significantly"

        return {
            'impact': impact,
            'criteria_met': criteria_met,
            'total_criteria': total_criteria,
            'recommendation': recommendation
        }

    def _save_comparison(self, comparison: Dict):
        """Save comparison results."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = self.results_dir / f"comparison_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Comparison saved to: {filename}")

    def _print_comparison(self, comparison: Dict):
        """Print comparison summary."""
        print("\n" + "="*80)
        print("SENTIMENT IMPACT COMPARISON RESULTS")
        print("="*80)

        # Metadata
        meta = comparison['metadata']
        print(f"\nSymbol: {meta['symbol']}")
        print(f"Period: {meta['period']}")

        # Technical Only
        tech = comparison['technical_only']
        print(f"\n--- TECHNICAL ONLY (32 features) ---")
        print(f"Win Rate: {tech['win_rate']:.2%}")
        print(f"Total P&L: ${tech['total_pnl']:.2f}")
        print(f"Trades: {tech['trades']}")
        print(f"Trading Mode: {tech['trading_mode_pct']:.1f}%")
        print(f"Avg Confidence: {tech['avg_confidence']:.2%}")

        # Technical + Sentiment
        sent = comparison['technical_plus_sentiment']
        print(f"\n--- TECHNICAL + SENTIMENT (39 features) ---")
        print(f"Win Rate: {sent['win_rate']:.2%}")
        print(f"Total P&L: ${sent['total_pnl']:.2f}")
        print(f"Trades: {sent['trades']}")
        print(f"Trading Mode: {sent['trading_mode_pct']:.1f}%")
        print(f"Avg Confidence: {sent['avg_confidence']:.2%}")

        # Improvements
        imp = comparison['improvements']
        print(f"\n--- IMPROVEMENTS WITH SENTIMENT ---")
        print(f"Win Rate: {imp['win_rate_improvement_pct']:+.2f} percentage points")
        print(f"P&L: {imp['pnl_improvement_pct']:+.2f}% (${imp['pnl_difference_usd']:+.2f})")
        print(f"Trades: {imp['trades_difference']:+d}")
        print(f"Trading Mode: {imp['trading_mode_diff_pct']:+.1f}%")
        print(f"Confidence: {imp['confidence_diff_pct']:+.2f}%")

        # Verdict
        verdict = comparison['verdict']
        print(f"\n--- VERDICT ---")
        print(f"Impact: {verdict['impact']}")
        print(f"Criteria Met: {verdict['criteria_met']}/{verdict['total_criteria']}")
        print(f"Recommendation: {verdict['recommendation']}")

        print("\n" + "="*80 + "\n")


def main():
    """Run comparison from command line."""
    parser = argparse.ArgumentParser(
        description='Compare sentiment impact on trading performance'
    )
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair')
    parser.add_argument('--days', type=int, default=90, help='Days to backtest')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')

    args = parser.parse_args()

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    # Run comparison
    comparison = SentimentImpactComparison(config_path=args.config)

    try:
        results = comparison.run_comparison(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date
        )

        # Print final verdict
        verdict = results['verdict']
        if verdict['impact'] == 'POSITIVE':
            logger.info("✓ Sentiment features IMPROVE trading performance!")
            return 0
        elif verdict['impact'] == 'MIXED':
            logger.warning("⚠ Sentiment features show MIXED results")
            return 0
        else:
            logger.warning("✗ Sentiment features do NOT improve performance")
            return 1

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
