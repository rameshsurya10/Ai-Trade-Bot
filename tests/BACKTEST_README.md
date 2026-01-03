# Continuous Learning System - Backtesting Guide

## Overview

Comprehensive backtesting suite for the continuous learning trading system.

## Scripts

### 1. `backtest_continuous_learning.py`
Main backtesting engine that simulates the complete continuous learning system.

**Features:**
- Multi-timeframe signal aggregation
- Confidence-based mode transitions (LEARNING ↔ TRADING)
- Retraining simulation
- Sentiment feature integration
- Performance metrics tracking

**Usage:**
```bash
# Basic backtest (90 days, BTC/USDT)
python tests/backtest_continuous_learning.py

# Custom symbol and period
python tests/backtest_continuous_learning.py --symbol ETH/USDT --days 180

# Without sentiment features
python tests/backtest_continuous_learning.py --no-sentiment

# Custom config file
python tests/backtest_continuous_learning.py --config custom_config.yaml
```

**Arguments:**
- `--symbol`: Trading pair (default: BTC/USDT)
- `--days`: Days to backtest (default: 90)
- `--no-sentiment`: Disable sentiment features
- `--config`: Config file path (default: config.yaml)

### 2. `compare_sentiment_impact.py`
Compares performance with and without sentiment features.

**Usage:**
```bash
# Run comparison
python tests/compare_sentiment_impact.py

# Custom parameters
python tests/compare_sentiment_impact.py --symbol BTC/USDT --days 90
```

**Output:**
- Win rate comparison
- P&L improvement
- Trading mode differences
- Confidence level changes
- **Verdict**: POSITIVE/MIXED/NEGATIVE impact

## Backtest Results

Results are saved to `backtest_results/` directory:

```
backtest_results/
├── backtest_20260102_143022.json    # Individual backtest results
├── comparisons/
│   ├── no_sentiment/                # Results without sentiment
│   ├── with_sentiment/              # Results with sentiment
│   └── comparison_20260102_143500.json  # Comparison results
```

## Metrics Tracked

### Performance Metrics
- **Total Candles**: Number of candles processed
- **Predictions Made**: Total predictions generated
- **Trades Executed**: Number of trades placed
- **Win Rate**: Percentage of winning trades
- **Total P&L**: Total profit/loss in USD
- **P&L %**: Return on initial capital

### Mode Metrics
- **Learning Mode %**: Time spent in learning mode
- **Trading Mode %**: Time spent in trading mode
- **Mode Transitions**: Number of mode switches
- **Avg Confidence (Learning)**: Average confidence during learning
- **Avg Confidence (Trading)**: Average confidence during trading

### Retraining Metrics
- **Retrainings Triggered**: Number of retraining events
- **Retraining Reasons**: Loss, drift, consecutive losses, etc.

## Expected Results

### Baseline Performance (Technical Only - 32 features)
- **Win Rate**: 50-55%
- **Mode**: 60-70% Learning, 30-40% Trading
- **Confidence**: 65-75% average

### With Sentiment (39 features)
- **Win Rate**: 52-58% (expected +2-5% improvement)
- **Mode**: 50-60% Learning, 40-50% Trading (more confident trading)
- **Confidence**: 70-80% average (higher confidence)

**Success Criteria:**
1. Win rate improvement ≥ 2%
2. P&L improvement ≥ 10%
3. More time in trading mode ≥ 5%

Meeting 2/3 criteria = **POSITIVE** sentiment impact

## Troubleshooting

### "No historical data available"
**Solution:** Ensure database has historical data:
```bash
# Run data collection first
python run_analysis.py --days 180
```

### "Model not found"
**Solution:** Train models for all timeframes:
```bash
# Train models
python src/multi_currency_system.py --train
```

### "Import errors"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

## Interpreting Results

### Win Rate
- **<50%**: System is worse than random - investigate features
- **50-55%**: Acceptable baseline
- **55-60%**: Good performance
- **>60%**: Excellent performance (verify not overfitting)

### Mode Distribution
- **>80% Learning**: Confidence threshold too high or model quality issues
- **>80% Trading**: Risk of overconfidence - review confidence gate
- **50-50 balanced**: Healthy adaptive system

### Mode Transitions
- **<5 transitions**: System stuck in one mode
- **5-20 transitions**: Healthy adaptation
- **>50 transitions**: Oscillating - increase hysteresis

## Next Steps

After backtesting:
1. **Review results** - Check if win rate ≥ 55%
2. **Analyze mode transitions** - Ensure healthy adaptation
3. **Verify sentiment impact** - Run comparison script
4. **Adjust thresholds** if needed in `config.yaml`
5. **Proceed to paper trading** validation (Week 4 Day 27-28)

## Advanced Usage

### Custom Date Range
```python
from backtest_continuous_learning import ContinuousLearningBacktest
from datetime import datetime

backtest = ContinuousLearningBacktest()

results = backtest.run_backtest(
    symbol='BTC/USDT',
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31),
    intervals=['1h', '4h', '1d'],
    use_sentiment=True
)
```

### Batch Testing Multiple Symbols
```bash
for symbol in BTC/USDT ETH/USDT; do
    python tests/backtest_continuous_learning.py --symbol $symbol --days 90
done
```

## Performance Optimization

For faster backtests:
- Reduce `--days` to 30-60
- Use fewer intervals (e.g., only 1h, 4h)
- Disable sentiment with `--no-sentiment`

## Support

For issues or questions:
- Check logs in `backtest_results/`
- Review configuration in `config.yaml`
- Verify database integrity: `sqlite3 data/trading.db "PRAGMA integrity_check;"`
