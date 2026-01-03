# Troubleshooting Guide

Common issues, solutions, and debugging strategies for the continuous learning trading system.

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Import and Dependency Errors](#import-and-dependency-errors)
3. [Training Issues](#training-issues)
4. [Prediction Errors](#prediction-errors)
5. [Learning Mode Issues](#learning-mode-issues)
6. [Retraining Problems](#retraining-problems)
7. [Database Issues](#database-issues)
8. [WebSocket / Data Issues](#websocket--data-issues)
9. [News and Sentiment Errors](#news-and-sentiment-errors)
10. [Performance Problems](#performance-problems)
11. [Production Deployment Issues](#production-deployment-issues)
12. [Emergency Procedures](#emergency-procedures)

---

## Quick Diagnostics

### System Health Check

Run this first when encountering issues:

```bash
# 1. Check Python version
python --version  # Should be 3.9+

# 2. Test imports
python -c "
from deploy_production import ProductionDeployment
from monitor_production import ProductionMonitor
print('✓ Imports OK')
"

# 3. Verify database
sqlite3 data/trading.db "PRAGMA integrity_check;"  # Should return 'ok'

# 4. Check models exist
ls -lh models/*.pt

# 5. Test configuration
python -c "from src.core.config import load_config; c = load_config(); print('✓ Config OK')"

# 6. Check dependencies
python deploy_production.py --help  # Will fail if dependencies missing
```

### Log Analysis

```bash
# Recent errors
grep ERROR logs/*.log | tail -20

# Recent warnings
grep WARNING logs/*.log | tail -20

# Count error types
grep ERROR logs/*.log | awk '{print $4}' | sort | uniq -c | sort -rn

# Failed retrainings
grep "Retraining failed" logs/*.log
```

---

## Import and Dependency Errors

### Error: `ModuleNotFoundError: No module named 'torch'`

**Cause:** PyTorch not installed

**Solution:**
```bash
pip install torch torchvision torchaudio
```

### Error: `ModuleNotFoundError: No module named 'sklearn'`

**Cause:** scikit-learn not installed

**Solution:**
```bash
pip install scikit-learn
```

### Error: `No module named 'src.core.config'`

**Cause:** Running script from wrong directory

**Solution:**
```bash
# Must run from project root
cd /path/to/Ai-Trade-Bot
python deploy_production.py --help
```

### Error: `ImportError: cannot import name 'load_config'`

**Cause:** Old version of config.py

**Solution:**
```bash
# Verify load_config function exists
grep -n "def load_config" src/core/config.py

# Should see: def load_config(config_path: str = 'config.yaml') -> dict:
```

If not found, update `src/core/config.py` (see CRITICAL_FIXES_APPLIED.md).

### Error: `ImportError: cannot import name 'PaperBrokerage'`

**Cause:** Incorrect import path

**Solution:**
```bash
# Verify correct import
grep "from src.paper_trading import PaperBrokerage" deploy_production.py

# Should NOT be: from src.brokerages.paper_brokerage
```

---

## Training Issues

### Error: `Insufficient data: 234 candles`

**Symptoms:**
```
ValueError: Insufficient data: 234 candles (need >= 1000)
```

**Cause:** Not enough historical data

**Solution:**
```bash
# 1. Check current data
sqlite3 data/trading.db "
SELECT symbol, interval, COUNT(*) as candles
FROM candles
GROUP BY symbol, interval;
"

# 2. Collect more data
python run_analysis.py --days 180 --symbol BTC/USDT

# 3. Wait for data accumulation (if running live)
# 1h interval needs: 1000 hours = ~42 days
# 1d interval needs: 1000 days = ~3 years
```

### Error: `CUDA out of memory`

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Cause:** GPU memory exhausted

**Solution:**
```bash
# Option 1: Force CPU
export CUDA_VISIBLE_DEVICES=""
python src/multi_currency_system.py --train

# Option 2: Reduce batch size
# Edit src/multi_currency_system.py
# Change: batch_size = 32  →  batch_size = 16

# Option 3: Reduce sequence length in config.yaml
timeframes:
  intervals:
    - interval: 1h
      sequence_length: 30  # Reduce from 60
```

### Error: `Model diverged (loss = nan)`

**Symptoms:**
```
Epoch 5/30: Loss=nan, Val Acc=0.00%
```

**Cause:** Learning rate too high or bad data

**Solution:**
```bash
# 1. Check for NaN in data
python -c "
import pandas as pd
df = pd.read_sql('SELECT * FROM candles LIMIT 1000',
                 'sqlite:///data/trading.db')
print('NaN count:', df.isnull().sum().sum())
"

# 2. Reduce learning rate
# Edit config.yaml
continuous_learning:
  online_learning:
    learning_rate: 0.00001  # Reduce from 0.0001

# 3. Re-train from scratch
rm models/*.pt
python src/multi_currency_system.py --train
```

### Training Takes Too Long (> 30 minutes per timeframe)

**Symptoms:**
```
Training BTC/USDT @ 1h... (stuck for 30+ minutes)
```

**Causes:**
1. Too much data
2. Large model
3. Slow hardware

**Solutions:**
```yaml
# config.yaml - Reduce complexity

model:
  architecture:
    hidden_size: 64      # Reduce from 128
    num_layers: 1        # Reduce from 2

timeframes:
  intervals:
    - interval: 1h
      sequence_length: 30  # Reduce from 60

continuous_learning:
  retraining:
    max_epochs: 30       # Reduce from 50
```

---

## Prediction Errors

### Error: `Model not found: models/model_BTC_USDT_1h.pt`

**Cause:** Model hasn't been trained

**Solution:**
```bash
# Train the missing model
python src/multi_currency_system.py --train --symbol BTC/USDT --interval 1h

# Or train all
python src/multi_currency_system.py --train
```

### Error: `Feature count mismatch: expected 39, got 32`

**Cause:** Sentiment disabled but model trained with sentiment

**Solutions:**
```bash
# Option 1: Enable sentiment
# config.yaml
news:
  enabled: true

# Option 2: Retrain without sentiment
# config.yaml
model:
  features:
    include_sentiment: false
    total_features: 32

# Then retrain
rm models/*.pt
python src/multi_currency_system.py --train
```

### Error: `Prediction confidence always < 50%`

**Symptoms:**
- Never enters TRADING mode
- Confidence stuck at 40-60%

**Diagnosis:**
```bash
# Check recent predictions
sqlite3 data/trading.db "
SELECT symbol, interval, AVG(confidence) as avg_conf
FROM predictions
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY symbol, interval;
"
```

**Solutions:**
1. **Retrain with more data:**
   ```bash
   # Collect more historical data
   python run_analysis.py --days 365

   # Retrain
   python src/multi_currency_system.py --train
   ```

2. **Lower threshold temporarily:**
   ```yaml
   # config.yaml
   continuous_learning:
     confidence:
       trading_threshold: 0.70  # Lower from 0.80
   ```

3. **Check feature quality:**
   ```bash
   # Verify sentiment features exist
   sqlite3 data/trading.db "
   SELECT COUNT(*) FROM sentiment_features;
   "
   # Should have data if news enabled
   ```

---

## Learning Mode Issues

### Stuck in LEARNING Mode (Never Trades Live)

**Symptoms:**
```
Mode: LEARNING (95% of time)
Confidence: 65-75% (never reaches 80%)
```

**Diagnosis:**
```bash
# Check confidence history
sqlite3 data/trading.db "
SELECT
  symbol,
  interval,
  AVG(confidence_score) as avg_conf,
  MAX(confidence_score) as max_conf,
  COUNT(*) as state_changes
FROM learning_states
WHERE entered_at > datetime('now', '-7 days')
GROUP BY symbol, interval;
"
```

**Solutions:**

1. **Temporary: Lower threshold**
   ```yaml
   continuous_learning:
     confidence:
       trading_threshold: 0.75  # Lower from 0.80
   ```

2. **Permanent: Improve model**
   ```bash
   # More training data
   python run_analysis.py --days 365

   # Retrain all
   rm models/*.pt
   python src/multi_currency_system.py --train
   ```

3. **Check data quality**
   ```bash
   # Missing data?
   sqlite3 data/trading.db "
   SELECT interval, COUNT(*) as gaps
   FROM (
     SELECT interval,
            timestamp - LAG(timestamp) OVER (PARTITION BY interval ORDER BY timestamp) as gap
     FROM candles
   )
   WHERE gap > 3600  -- More than 1 hour gap
   GROUP BY interval;
   "
   ```

### Excessive Mode Oscillation

**Symptoms:**
```
Mode transitions: 45 in last 12 hours
Keeps switching: LEARNING ↔ TRADING
```

**Cause:** Confidence hovering near threshold

**Solutions:**

1. **Increase hysteresis:**
   ```yaml
   continuous_learning:
     confidence:
       hysteresis: 0.10  # Increase from 0.05
   ```

2. **Add confidence smoothing:**
   ```yaml
   continuous_learning:
     confidence:
       smoothing_alpha: 0.2  # Decrease from 0.3 (slower changes)
   ```

3. **Check for model instability:**
   ```bash
   # Analyze confidence volatility
   sqlite3 data/trading.db "
   SELECT
     symbol,
     interval,
     STDEV(confidence_score) as conf_volatility
   FROM confidence_history
   WHERE timestamp > datetime('now', '-24 hours')
   GROUP BY symbol, interval;
   "
   # High volatility (> 0.15) indicates unstable model
   ```

---

## Retraining Problems

### Retraining Triggers Too Frequently

**Symptoms:**
```
Retraining events: 25 in last 6 hours
```

**Diagnosis:**
```bash
# Check trigger reasons
sqlite3 data/trading.db "
SELECT trigger_reason, COUNT(*) as count
FROM retraining_history
WHERE triggered_at > datetime('now', '-6 hours')
GROUP BY trigger_reason
ORDER BY count DESC;
"
```

**Solutions:**

1. **If `loss_immediate` too frequent:**
   ```yaml
   continuous_learning:
     retraining:
       min_interval_hours: 2  # Add cooldown (from 1)
   ```

2. **If `drift` too sensitive:**
   ```yaml
   continuous_learning:
     retraining:
       drift_threshold: 0.80  # Increase from 0.70
   ```

3. **Check if losing too many trades:**
   ```bash
   # Analyze recent performance
   sqlite3 data/trading.db "
   SELECT
     COUNT(*) as total,
     SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
     ROUND(100.0 * SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate
   FROM trade_outcomes
   WHERE entry_time > datetime('now', '-24 hours');
   "
   # Win rate < 45% needs investigation
   ```

### Retraining Never Reaches Target Confidence

**Symptoms:**
```
Retraining completed but confidence still low
Best: 72.15%, Target: 80.00%
```

**Diagnosis:**
```bash
# Check retraining history
sqlite3 data/trading.db "
SELECT
  symbol,
  interval,
  trigger_reason,
  validation_confidence,
  n_epochs
FROM retraining_history
WHERE status = 'below_target'
ORDER BY id DESC
LIMIT 10;
"
```

**Solutions:**

1. **Increase max epochs:**
   ```yaml
   continuous_learning:
     retraining:
       max_epochs: 100  # Increase from 50
   ```

2. **Check training data quality:**
   ```bash
   # Verify enough recent data
   sqlite3 data/trading.db "
   SELECT interval, COUNT(*) as recent_candles
   FROM candles
   WHERE timestamp > unixepoch('now') - 86400*30  -- Last 30 days
   GROUP BY interval;
   "
   # Should have 1000+ candles
   ```

3. **Lower target temporarily:**
   ```yaml
   continuous_learning:
     retraining:
       target_confidence: 0.75  # Lower from 0.80
   ```

### Retraining Crashes with OOM (Out of Memory)

**Symptoms:**
```
RuntimeError: Out of memory. Tried to allocate X MiB
```

**Solutions:**

1. **Reduce replay buffer:**
   ```yaml
   continuous_learning:
     experience_replay:
       buffer_size: 5000  # Reduce from 10000
   ```

2. **Reduce batch size:**
   - Edit `src/learning/retraining_engine.py`
   - Find: `batch_size = 32`
   - Change to: `batch_size = 16`

3. **Use CPU for retraining:**
   ```yaml
   model:
     force_cpu: true  # Add this line
   ```

---

## Database Issues

### Error: `database is locked`

**Symptoms:**
```
sqlite3.OperationalError: database is locked
```

**Cause:** Multiple processes accessing database

**Solutions:**

1. **Enable WAL mode:**
   ```yaml
   # config.yaml
   database:
     wal_mode: true
   ```

2. **Increase timeout:**
   ```yaml
   database:
     timeout: 60.0  # Increase from 30.0
   ```

3. **Check for stuck processes:**
   ```bash
   # Find processes using database
   lsof data/trading.db

   # Kill if necessary
   pkill -f "python.*trading"
   ```

### Database Corruption

**Symptoms:**
```
PRAGMA integrity_check: *** in database main ***
```

**Solution:**
```bash
# 1. Backup current database
cp data/trading.db data/trading_corrupted_$(date +%Y%m%d).db

# 2. Try repair
sqlite3 data/trading.db "
PRAGMA integrity_check;
REINDEX;
VACUUM;
"

# 3. If repair fails, restore from backup
cp backups/trading_YYYYMMDD.db data/trading.db

# 4. If no backup, export/reimport
sqlite3 data/trading.db .dump > trading_dump.sql
rm data/trading.db
sqlite3 data/trading.db < trading_dump.sql
```

### Database Growing Too Large (> 10GB)

**Solutions:**

1. **Vacuum database:**
   ```bash
   sqlite3 data/trading.db "VACUUM;"
   ```

2. **Archive old data:**
   ```bash
   # Export old data
   sqlite3 data/trading.db "
   .mode csv
   .output archive_$(date +%Y%m%d).csv
   SELECT * FROM candles WHERE timestamp < unixepoch('now') - 86400*180;
   "

   # Delete old data
   sqlite3 data/trading.db "
   DELETE FROM candles WHERE timestamp < unixepoch('now') - 86400*180;
   VACUUM;
   "
   ```

3. **Configure auto-cleanup:**
   ```yaml
   database:
     auto_cleanup:
       enabled: true
       keep_days: 180  # Keep last 6 months
   ```

---

## WebSocket / Data Issues

### Error: `WebSocket connection failed`

**Symptoms:**
```
ConnectionError: WebSocket connection to wss://stream.binance.com failed
```

**Solutions:**

1. **Check internet connection:**
   ```bash
   ping 8.8.8.8
   curl https://api.binance.com/api/v3/ping
   ```

2. **Check firewall:**
   ```bash
   # Allow outbound WebSocket
   sudo ufw allow out 443/tcp
   ```

3. **Use proxy (if behind firewall):**
   ```yaml
   # config.yaml
   network:
     proxy: "http://proxy.example.com:8080"
   ```

### Missing Candle Data

**Symptoms:**
```
WARNING: Gap detected in candles (1h interval)
Last candle: 2026-01-02 10:00:00
Next candle: 2026-01-02 14:00:00  # Missing 3 hours
```

**Solutions:**

1. **Backfill missing data:**
   ```bash
   python run_analysis.py --backfill --days 7
   ```

2. **Check WebSocket status:**
   ```bash
   # Monitor connection
   tail -f logs/production_deployment.log | grep "WebSocket"
   ```

3. **Restart data provider:**
   ```bash
   # Rollback and redeploy
   python deploy_production.py --rollback
   python deploy_production.py --phase 3
   ```

---

## News and Sentiment Errors

### Error: `NewsAPI rate limit exceeded`

**Symptoms:**
```
WARNING: NewsAPI rate limit exceeded (100/day)
```

**Solutions:**

1. **Reduce fetch frequency:**
   ```yaml
   news:
     sources:
       newsapi:
         fetch_interval: 3600  # Increase to 1 hour (from 30 min)
   ```

2. **Disable NewsAPI temporarily:**
   ```yaml
   news:
     sources:
       newsapi:
         enabled: false
       alphavantage:
         enabled: true  # Use this only
   ```

3. **Upgrade API plan** (if needed)

### Sentiment Features All Zero

**Symptoms:**
```
sentiment_1h: 0.0
sentiment_6h: 0.0
sentiment_24h: 0.0
```

**Diagnosis:**
```bash
# Check news articles
sqlite3 data/trading.db "
SELECT COUNT(*) as article_count
FROM news_articles
WHERE timestamp > unixepoch('now') - 86400;
"
# Should have 20+ articles per day
```

**Solutions:**

1. **Verify API keys:**
   ```bash
   cat .env | grep NEWSAPI_KEY
   # Should show your key (not empty)
   ```

2. **Test news collection manually:**
   ```bash
   python -c "
   from src.news.collector import NewsCollector
   from src.core.config import load_config

   collector = NewsCollector(load_config())
   articles = collector.fetch_recent(['BTC'])
   print(f'Fetched {len(articles)} articles')
   "
   ```

3. **Check sentiment calculation:**
   ```bash
   # Verify VADER working
   python -c "
   from src.news.sentiment import SentimentAnalyzer

   analyzer = SentimentAnalyzer()
   score = analyzer.analyze('Bitcoin is bullish and going to the moon')
   print(f'Sentiment score: {score}')  # Should be positive
   "
   ```

---

## Performance Problems

### High CPU Usage (> 80%)

**Causes:**
1. Too many symbols
2. Too many timeframes
3. Continuous retraining

**Solutions:**

1. **Reduce active symbols:**
   ```yaml
   symbols:
     - BTC/USDT  # Start with 1
     # - ETH/USDT  # Disable others
   ```

2. **Disable short timeframes:**
   ```yaml
   timeframes:
     intervals:
       - interval: 1m
         enabled: false  # Very compute-intensive
       - interval: 5m
         enabled: false
   ```

3. **Increase retraining cooldown:**
   ```yaml
   continuous_learning:
     retraining:
       min_interval_hours: 4  # Reduce frequency
   ```

### High Memory Usage (> 8GB)

**Solutions:**

1. **Reduce buffer sizes:**
   ```yaml
   continuous_learning:
     experience_replay:
       buffer_size: 3000  # Reduce from 10000
   ```

2. **Lower sequence length:**
   ```yaml
   timeframes:
     intervals:
       - interval: 1h
         sequence_length: 30  # Reduce from 60
   ```

3. **Force garbage collection:**
   - Edit `src/learning/continuous_learner.py`
   - Add: `import gc; gc.collect()` after retraining

### Slow Predictions (> 5 seconds)

**Diagnosis:**
```bash
# Profile prediction speed
python -c "
import time
from src.advanced_predictor import UnbreakablePredictor

predictor = UnbreakablePredictor()

start = time.time()
# Make prediction
elapsed = time.time() - start
print(f'Prediction took {elapsed:.2f}s')
"
```

**Solutions:**

1. **Use CPU if GPU slow:**
   ```yaml
   model:
     force_cpu: true
   ```

2. **Reduce model size:**
   ```yaml
   model:
     architecture:
       hidden_size: 64  # Reduce from 128
   ```

3. **Cache features:**
   - Already implemented in `analysis_engine.py`
   - Verify cache working

---

## Production Deployment Issues

### Pre-Deployment Checks Fail

**Symptoms:**
```
❌ Pre-deployment checks failed
Checks passed: 3/5
```

**Solutions:**

1. **Check which failed:**
   ```bash
   # Run deployment with verbose logging
   python deploy_production.py --phase 1 2>&1 | grep "❌"
   ```

2. **Common failures:**
   - Database integrity: `sqlite3 data/trading.db "PRAGMA integrity_check;"`
   - Missing models: `python src/multi_currency_system.py --train`
   - No historical data: `python run_analysis.py --days 90`
   - Bad config: `python -c "from src.core.config import load_config; load_config()"`

### Phase 1 Validation Fails (< 4 criteria)

**Example:**
```
Criteria Met: 3/5
✓ error_rate: 0.0042
✗ win_rate: 0.4521 (threshold: 0.5000)
✗ max_drawdown: 16.34 (threshold: 15.0000)
```

**Solutions:**

1. **If win rate low:**
   - Retrain with more data
   - Check feature quality
   - Review losing trades for patterns

2. **If drawdown high:**
   - Reduce position sizes
   - Tighten stop losses
   - Lower risk limits

3. **Extend validation:**
   ```bash
   # Run for 72 hours instead of 48
   python run_paper_trading_validation.py --duration 72
   ```

---

## Emergency Procedures

### System Unresponsive

```bash
# 1. Check if running
ps aux | grep python

# 2. Check logs for errors
tail -50 logs/production_deployment.log

# 3. Force stop
pkill -f "deploy_production"
pkill -f "python.*continuous"

# 4. Restart
python deploy_production.py --phase 3
```

### Rapid Portfolio Decline

```bash
# 1. IMMEDIATE STOP
python deploy_production.py --rollback

# 2. Check recent trades
sqlite3 data/trading.db "
SELECT * FROM trade_outcomes
ORDER BY id DESC
LIMIT 20;
"

# 3. Analyze issue
# - All losses? → Model problem
# - High slippage? → Market volatility
# - Bad signals? → Review confidence

# 4. Fix and re-validate
# Retrain, adjust config, test in backtest
```

### Data Corruption Detected

```bash
# 1. Stop system
Ctrl+C

# 2. Backup corrupted DB
cp data/trading.db data/corrupted_$(date +%Y%m%d%H%M%S).db

# 3. Restore from latest backup
cp backups/trading_LATEST.db data/trading.db

# 4. Verify integrity
sqlite3 data/trading.db "PRAGMA integrity_check;"

# 5. Restart
python deploy_production.py --phase 3
```

---

## Getting More Help

### Enable Debug Logging

```yaml
# config.yaml
logging:
  level: DEBUG  # Change from INFO
```

```bash
# Restart with debug logging
python deploy_production.py --phase 3

# Very verbose output
tail -f logs/production_deployment.log
```

### Collect Diagnostic Info

```bash
# Create diagnostic bundle
./scripts/collect_diagnostics.sh

# Or manually:
mkdir diagnostics_$(date +%Y%m%d)
cp logs/*.log diagnostics_*/
cp config.yaml diagnostics_*/
sqlite3 data/trading.db .dump > diagnostics_*/database_dump.sql
```

### Report an Issue

Include:
1. Error message (full traceback)
2. Configuration (sanitize API keys!)
3. Recent logs (last 100 lines)
4. System info (`python --version`, `pip list`)
5. Steps to reproduce

---

**See Also:**
- [USER_GUIDE.md](USER_GUIDE.md) - General usage
- [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) - Configuration reference
- [MONITORING_PLAYBOOK.md](MONITORING_PLAYBOOK.md) - Operational monitoring
