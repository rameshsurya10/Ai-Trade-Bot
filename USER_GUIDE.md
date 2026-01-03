# Continuous Learning Trading System - User Guide

## Welcome!

This guide will help you operate the continuous learning trading system, from initial setup to production deployment.

**What is this system?**
- **Adaptive AI Trading**: Automatically learns from every trade and market condition
- **Multi-Timeframe Analysis**: Combines signals from 1m, 5m, 15m, 1h, 4h, and 1d intervals
- **Confidence-Based Trading**: Only trades when model confidence ≥ 80%
- **Immediate Retraining**: Adapts instantly after any losing trade
- **News Integration**: Incorporates market sentiment from news and social media

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Components](#system-components)
3. [Configuration](#configuration)
4. [Running the System](#running-the-system)
5. [Monitoring](#monitoring)
6. [Understanding Modes](#understanding-modes)
7. [Safety Features](#safety-features)
8. [Common Workflows](#common-workflows)
9. [FAQ](#faq)

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- Stable internet connection
- API keys (optional for news features)

### Installation

```bash
# 1. Clone/download the repository
cd Ai-Trade-Bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys (optional)
cp .env.example .env
# Edit .env with your API keys

# 4. Verify installation
python -c "from deploy_production import ProductionDeployment; print('✓ Installation OK')"
```

### First Run

```bash
# 1. Train initial models
python src/multi_currency_system.py --train --symbol BTC/USDT

# 2. Run backtest to verify
python tests/backtest_continuous_learning.py --days 90

# 3. Start paper trading validation
python run_paper_trading_validation.py --duration 48

# 4. Monitor in separate terminal
python tests/monitor_paper_trading.py
```

---

## System Components

### Core Components

1. **Continuous Learning Engine** (`src/learning/continuous_learner.py`)
   - Orchestrates the entire learning pipeline
   - Manages mode transitions (LEARNING ↔ TRADING)
   - Triggers retraining on losses

2. **Multi-Timeframe Predictor** (`src/advanced_predictor.py`)
   - Generates predictions across all timeframes
   - Combines technical indicators (32 features)
   - Integrates sentiment features (7 features)

3. **Signal Aggregator** (`src/multi_timeframe/aggregator.py`)
   - Weights predictions from different timeframes
   - Produces final BUY/SELL/NEUTRAL signal
   - Calculates aggregate confidence

4. **Confidence Gate** (`src/learning/confidence_gate.py`)
   - Decides LEARNING vs TRADING mode
   - 80% threshold with 5% hysteresis
   - Prevents mode oscillation

5. **Retraining Engine** (`src/learning/retraining_engine.py`)
   - Triggered on any loss (immediate)
   - Trains until confidence ≥ 80%
   - Uses EWC to prevent forgetting

6. **News Collector** (`src/news/collector.py`)
   - Fetches from NewsAPI and Alpha Vantage
   - Analyzes sentiment with VADER
   - Generates 7 sentiment features

### Tools & Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `deploy_production.py` | Production deployment | After successful validation |
| `monitor_production.py` | Real-time monitoring | During production |
| `run_paper_trading_validation.py` | 48-hour validation | Before production |
| `tests/backtest_continuous_learning.py` | Historical testing | After training models |
| `dashboard_continuous_learning.py` | Web dashboard | Visual monitoring |

---

## Configuration

### Main Configuration File: `config.yaml`

#### Key Settings

**1. Symbols to Trade**
```yaml
symbols:
  - BTC/USDT
  - ETH/USDT
  # Add more symbols here
```

**2. Timeframes**
```yaml
timeframes:
  enabled: true
  aggregation_method: weighted_vote  # How to combine timeframe signals

  intervals:
    - interval: 1h
      enabled: true
      weight: 0.25  # Higher weight = more influence
      sequence_length: 60  # Lookback candles
```

**3. Confidence Threshold**
```yaml
continuous_learning:
  confidence:
    trading_threshold: 0.80  # 80% confidence required for live trading
    hysteresis: 0.05         # Prevents rapid mode switching
```

**4. Retraining Triggers**
```yaml
continuous_learning:
  retraining:
    on_loss: true              # Retrain immediately on any loss
    consecutive_loss_threshold: 3
    drift_threshold: 0.7
    max_epochs: 50
    target_confidence: 0.80    # Train until this confidence
```

**5. Risk Management**
```yaml
risk:
  max_drawdown_percent: 15.0
  daily_loss_limit: 0.05
  max_position_size_percent: 0.10
```

**6. News & Sentiment**
```yaml
news:
  enabled: true  # Set to false to disable
  sources:
    newsapi:
      enabled: true
      api_key_env: NEWSAPI_KEY  # Set in .env
```

For complete configuration details, see [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md).

---

## Running the System

### 1. Training Models

**Train all timeframes for a symbol:**
```bash
python src/multi_currency_system.py --train --symbol BTC/USDT
```

**Train specific timeframe:**
```bash
python src/multi_currency_system.py --train --symbol BTC/USDT --interval 1h
```

**Expected output:**
```
Training BTC/USDT @ 1h...
Training data: 4821 train, 1205 val
Epoch 1/30: Loss=0.0523, Val Acc=72.34%, Val Conf=68.12%
Epoch 10/30: Loss=0.0234, Val Acc=84.21%, Val Conf=79.45%
Epoch 20/30: Loss=0.0198, Val Acc=88.56%, Val Conf=84.23%
✓ Training complete: Val Conf=84.23%
Model saved: models/model_BTC_USDT_1h.pt
```

### 2. Backtesting

**Run backtest with sentiment:**
```bash
python tests/backtest_continuous_learning.py --symbol BTC/USDT --days 90
```

**Without sentiment:**
```bash
python tests/backtest_continuous_learning.py --symbol BTC/USDT --days 90 --no-sentiment
```

**Compare sentiment impact:**
```bash
python tests/compare_sentiment_impact.py --symbol BTC/USDT --days 90
```

**Expected results:**
- Win rate: 52-58% (with sentiment)
- Learning mode: 50-60% of time
- Trading mode: 40-50% of time
- Mode transitions: 5-20 over 90 days

### 3. Paper Trading Validation

**Start 48-hour validation:**
```bash
# Terminal 1
python run_paper_trading_validation.py --duration 48

# Terminal 2 - Monitor
python tests/monitor_paper_trading.py

# Terminal 3 - Logs
tail -f logs/paper_trading_validation.log
```

**What to look for:**
- ✅ Win rate ≥ 50%
- ✅ Max drawdown ≤ 15%
- ✅ Error rate < 5%
- ✅ P&L > $0
- ✅ < 2 mode transitions per hour

**Validation passes if 4/5 criteria met.**

### 4. Production Deployment

**Phase 1: Single Symbol (24 hours)**
```bash
python deploy_production.py --phase 1 --symbol BTC/USDT
```

**Phase 3: Full Rollout (after Phase 1 approval)**
```bash
python deploy_production.py --phase 3
```

**Emergency Rollback:**
```bash
python deploy_production.py --rollback
```

Or press **Ctrl+C** in deployment terminal.

---

## Monitoring

### Dashboard (Web UI)

```bash
streamlit run dashboard_continuous_learning.py
```

Open browser: http://localhost:8501

**Features:**
- 🎯 Current mode (TRADING/LEARNING)
- 📈 Confidence trends over time
- 📰 News sentiment panel
- 🔄 Retraining history
- 📊 Multi-timeframe signals
- 💰 Performance by mode
- 🛡️ Safety status

### Terminal Monitor

```bash
python monitor_production.py
```

Updates every 10 seconds with:
- System status
- Per-symbol performance
- Recent activity
- Active alerts

### Generate Reports

```bash
# 24-hour report
python monitor_production.py --report --hours 24

# Saved to: production_reports/report_YYYYMMDD_HHMMSS.json
```

### Logs

```bash
# Real-time logs
tail -f logs/production_deployment.log

# Last 100 lines
tail -100 logs/production_deployment.log

# Search for errors
grep ERROR logs/production_deployment.log
```

---

## Understanding Modes

### LEARNING Mode

**When:**
- Model confidence < 80%
- After retraining completes
- During model adaptation

**What happens:**
- Executes **paper trades** only
- Learns from every candle
- Builds confidence
- No real money at risk

**Indicator:** 📚 Blue badge

### TRADING Mode

**When:**
- Model confidence ≥ 80%
- Successful retraining
- High-quality signals

**What happens:**
- Executes **live trades** (or paper if configured)
- Continues learning
- Maintains high confidence
- Real trading enabled

**Indicator:** 🎯 Green badge

### Mode Transitions

**LEARNING → TRADING:**
- Confidence reaches 80%
- Model is validated
- Signal: ✓ "Transitioned to TRADING mode"

**TRADING → LEARNING:**
- Any losing trade (retraining triggered)
- Confidence drops below 75% (hysteresis)
- Signal: ← "Transitioned to LEARNING mode"

**Healthy System:**
- 40-60% in each mode
- 5-20 transitions per week
- Smooth confidence trends

**Problematic:**
- > 80% stuck in one mode
- > 2 transitions per hour (oscillating)
- Confidence never reaching 80%

---

## Safety Features

### Automated Safety Checks

The system automatically monitors:

1. **Max Drawdown (15% limit)**
   - System stops if portfolio drops 15%
   - Automatic rollback initiated

2. **Win Rate (45% minimum)**
   - Alert if win rate drops below 45%
   - Manual intervention required

3. **Error Rate (5% maximum)**
   - Alert if errors exceed 5% of candles
   - Investigation needed

4. **System Stability**
   - Alert if > 2 mode transitions per hour
   - May indicate unstable model

5. **Daily Loss Limit**
   - Stops trading if daily loss exceeds 5%
   - Resets at midnight UTC

### Manual Controls

**Emergency Stop:**
```bash
# In deployment terminal
Ctrl+C

# Or force rollback
python deploy_production.py --rollback
```

**Pause Learning (Paper Trading Only):**
```yaml
# config.yaml
continuous_learning:
  enabled: false  # Disables learning, keeps prediction only
```

**Disable Specific Symbol:**
```yaml
# config.yaml
symbols:
  - BTC/USDT
  # - ETH/USDT  # Comment out to disable
```

---

## Common Workflows

### Daily Operations

**Morning Routine (5 minutes):**
```bash
# 1. Check overnight performance
python monitor_production.py --report --hours 24

# 2. Check for alerts
grep "ALERT" logs/production_deployment.log

# 3. Verify system healthy
# Open dashboard: streamlit run dashboard_continuous_learning.py
```

**If all green:** No action needed. System self-manages.

**If warnings:** Check specific issues in logs.

### Weekly Review

```bash
# Generate 7-day report
python monitor_production.py --report --hours 168

# Review:
# - Overall win rate
# - Per-symbol performance
# - Retraining frequency
# - Any recurring alerts
```

### Adding New Symbol

```bash
# 1. Add to config
# Edit config.yaml, add symbol to symbols list

# 2. Train models
python src/multi_currency_system.py --train --symbol NEW/USDT

# 3. Backtest
python tests/backtest_continuous_learning.py --symbol NEW/USDT --days 90

# 4. If backtest good (win rate ≥ 55%), deploy
# System automatically includes new symbols on next restart
```

### Adjusting Confidence Threshold

```yaml
# config.yaml
continuous_learning:
  confidence:
    trading_threshold: 0.82  # Increase for more conservative
    # or
    trading_threshold: 0.75  # Decrease for more aggressive
```

**After changing:**
```bash
# Restart system
Ctrl+C  # Stop current deployment
python deploy_production.py --phase 3  # Restart
```

### Tuning Timeframe Weights

If one timeframe performs better:

```yaml
# config.yaml
timeframes:
  intervals:
    - interval: 1h
      weight: 0.30  # Increase from 0.25
    - interval: 4h
      weight: 0.20  # Decrease from 0.25
```

Monitor impact over 24-48 hours.

---

## FAQ

### Q: How long does initial training take?

**A:** 5-15 minutes per symbol per timeframe.
- BTC/USDT with 6 timeframes: ~1 hour
- ETH/USDT with 6 timeframes: ~1 hour

### Q: Can I run multiple symbols simultaneously?

**A:** Yes! Configure in `config.yaml`:
```yaml
symbols:
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT
```

System handles all in parallel.

### Q: What if I don't have news API keys?

**A:** System works without news (32 technical features only).

Disable in config:
```yaml
news:
  enabled: false
```

Expected performance: ~2-5% lower win rate.

### Q: How much historical data is needed?

**A:** Minimum 1000 candles per timeframe.
- 1h timeframe: 1000 hours = ~42 days
- 1d timeframe: 1000 days = ~3 years

More data = better models.

### Q: Can I use this with real money?

**A:** Yes, but:
1. Complete all validation steps first
2. Start with small position sizes
3. Monitor closely for first week
4. Gradually increase allocation

**Recommended:** Start with 1-5% of portfolio.

### Q: What if win rate drops below 50%?

**A:** System automatically:
1. Stays in LEARNING mode
2. Retrains models
3. Builds confidence back up

**Manual action:** Review logs for root cause.

### Q: How do I know if retraining is working?

**A:** Check dashboard "Retraining History" panel:
- ✓ Success rate should be ≥ 80%
- Avg confidence after retraining ≥ 80%
- Duration: 2-10 minutes per retrain

### Q: Can I trade multiple exchanges?

**A:** Currently supports Binance.
Future: Alpaca, other exchanges planned.

### Q: What's the recommended computer specs?

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Disk: 10GB free
- Internet: Stable connection

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB
- Disk: 50GB SSD
- Internet: Low latency (< 100ms to exchange)

### Q: How do I backup my data?

**A:** Automatic backups recommended:
```bash
# Add to crontab (daily 3 AM)
0 3 * * * cp /path/to/trading.db /path/to/backups/trading_$(date +\%Y\%m\%d).db
```

Backup:
- Database: `data/trading.db`
- Models: `models/*.pt`
- Config: `config.yaml`

### Q: What if I see "Model not found" error?

**A:** Run training:
```bash
python src/multi_currency_system.py --train
```

Verify models exist:
```bash
ls models/*.pt
```

Should see files like:
```
model_BTC_USDT_1h.pt
model_BTC_USDT_4h.pt
...
```

### Q: How do I update the system?

**A:**
```bash
# 1. Stop system
Ctrl+C

# 2. Backup
cp -r models models_backup
cp data/trading.db data/trading_backup.db

# 3. Pull updates
git pull  # or download new version

# 4. Install dependencies
pip install -r requirements.txt

# 5. Restart
python deploy_production.py --phase 3
```

### Q: Can I test changes without affecting production?

**A:** Yes! Use separate config:
```bash
# Copy config
cp config.yaml config_test.yaml

# Edit config_test.yaml

# Run with test config
python deploy_production.py --phase 1 --config config_test.yaml
```

---

## Next Steps

1. ✅ **Complete Setup** - Install and configure
2. ✅ **Train Models** - Get initial models trained
3. ✅ **Run Backtest** - Verify historical performance
4. ✅ **Paper Trading** - 48-hour validation
5. ✅ **Phase 1 Deploy** - Single symbol, 24 hours
6. ✅ **Phase 3 Deploy** - Full production
7. 📊 **Monitor & Tune** - Ongoing optimization

---

## Getting Help

- **Troubleshooting:** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Configuration:** See [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)
- **Monitoring:** See [MONITORING_PLAYBOOK.md](MONITORING_PLAYBOOK.md)
- **Deployment:** See [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)

---

**Happy Trading! 🚀**

Remember: The system learns and adapts. Give it time to build confidence, monitor regularly, and trust the process.
