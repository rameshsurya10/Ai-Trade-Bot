# Paper Trading Validation Guide

## Overview

48-hour live validation of the continuous learning trading system using paper trading.

**Purpose:** Verify system stability, safety, and performance before production deployment.

## Quick Start

### 1. Start Paper Trading Validation

```bash
# Run validation for 48 hours
python run_paper_trading_validation.py --duration 48

# Custom symbols and duration
python run_paper_trading_validation.py --symbols BTC/USDT ETH/USDT --duration 72
```

### 2. Monitor Progress (in separate terminal)

```bash
python tests/monitor_paper_trading.py
```

### 3. View Logs

```bash
tail -f logs/paper_trading_validation.log
```

---

## Validation Criteria

System must meet **4 out of 5 criteria** to be APPROVED for production:

| # | Criterion | Threshold | Weight |
|---|-----------|-----------|--------|
| 1 | Win Rate | ≥ 50% | Critical |
| 2 | Max Drawdown | ≤ 15% | Critical |
| 3 | Error Rate | < 5% | Critical |
| 4 | Profitability | P&L > $0 | Important |
| 5 | Stability | < 2 mode transitions/hour | Important |

### Criterion Details

#### 1. Win Rate ≥ 50%
- **Minimum:** 50% (better than random)
- **Target:** 55-60%
- **Excellent:** > 60%
- **Note:** Calculated after minimum 20 trades

#### 2. Max Drawdown ≤ 15%
- **Acceptable:** 0-10%
- **Warning:** 10-15%
- **STOP:** > 15%
- **Measured:** Peak-to-trough percentage decline

#### 3. Error Rate < 5%
- **Target:** < 1%
- **Acceptable:** 1-5%
- **Critical:** > 5%
- **Includes:** Exception handling, data errors, connection issues

#### 4. Profitability
- **Minimum:** P&L > $0
- **Target:** P&L > 5% of initial capital
- **Note:** Must be profitable after fees/slippage simulation

#### 5. System Stability
- **Normal:** 1-5 mode transitions per day
- **Acceptable:** < 2 transitions per hour
- **Problematic:** > 2 transitions per hour (oscillating)

---

## Pre-Validation Checklist

### Environment Setup

- [ ] **Python Environment**
  ```bash
  python --version  # Should be 3.9+
  pip install -r requirements.txt
  ```

- [ ] **Database Ready**
  ```bash
  sqlite3 data/trading.db "SELECT COUNT(*) FROM candles;"
  # Should have historical data
  ```

- [ ] **Models Trained**
  ```bash
  ls models/*.pt
  # Should see model files for all timeframes
  ```

- [ ] **API Keys Configured** (`.env`)
  ```bash
  cat .env | grep -E "BINANCE|NEWSAPI|ALPHAVANTAGE"
  # Should show API keys (masked)
  ```

- [ ] **Logs Directory**
  ```bash
  mkdir -p logs
  ```

- [ ] **Validation Results Directory**
  ```bash
  mkdir -p validation_results
  ```

### Configuration Verification

- [ ] **config.yaml checked**
  - `live_trading.mode: paper` ✓
  - `continuous_learning.enabled: true` ✓
  - `news.enabled: true` ✓
  - `timeframes.enabled: true` ✓

- [ ] **Intervals Enabled**
  ```yaml
  timeframes:
    intervals:
      - interval: 5m
        enabled: true
        weight: 0.15
      - interval: 1h
        enabled: true
        weight: 0.25
      # etc...
  ```

- [ ] **Safety Limits Set**
  ```yaml
  risk:
    max_drawdown_percent: 15.0
    daily_loss_limit: 0.05
  ```

---

## Running Validation

### Step 1: Start Validation

```bash
# Terminal 1 - Run validation
python run_paper_trading_validation.py --duration 48 --symbols BTC/USDT

# You should see:
# ================================================================================
# PAPER TRADING VALIDATION STARTED
# ================================================================================
# Symbols: BTC/USDT
# Duration: 2 days, 0:00:00
# End Time: 2026-01-04 12:00:00
# ================================================================================
```

### Step 2: Start Monitor

```bash
# Terminal 2 - Real-time monitoring
python tests/monitor_paper_trading.py

# Displays live dashboard with:
# - Current mode (LEARNING/TRADING)
# - Portfolio value and P&L
# - Win rate
# - Recent trades
# - Safety check status
```

### Step 3: Check Logs Periodically

```bash
# Terminal 3 - Watch logs
tail -f logs/paper_trading_validation.log

# Look for:
# - Candle close events
# - Predictions and signals
# - Mode transitions
# - Retraining events
# - Any errors or warnings
```

---

## What to Monitor

### Every Hour

- [ ] **Portfolio Value Trend**
  - Should be stable or growing
  - Red flag if declining rapidly

- [ ] **Error Count**
  - Should be minimal (< 1% of candles)
  - Investigate if > 5%

- [ ] **Mode Distribution**
  - Healthy: 40-60% in each mode
  - Warning: > 80% in LEARNING (confidence issues)
  - Warning: > 80% in TRADING (overconfidence)

### Every 6 Hours

- [ ] **Win Rate Check**
  - After 20+ trades, should be ≥ 50%
  - If < 45%, investigate model quality

- [ ] **Retraining Frequency**
  - Normal: 1-3 retrainings per day
  - Too many: > 10 per day (model instability)
  - Too few: 0 in 24 hours (no adaptation)

- [ ] **Safety Limits**
  - Drawdown should stay < 15%
  - No critical safety warnings

### Daily

- [ ] **Generate Progress Report**
  - Review validation statistics
  - Check performance trends
  - Verify system stability

- [ ] **Database Backup**
  ```bash
  cp data/trading.db backups/trading_$(date +%Y%m%d).db
  ```

---

## Expected Behavior

### Normal Operation

```
[2026-01-02 14:23:15] [BTC/USDT @ 1h] Candle closed: O=43250.00 H=43350.00 L=43200.00 C=43300.00
[2026-01-02 14:23:15] [BTC/USDT @ 1h] Mode: LEARNING | Signal: BUY | Confidence: 72.45% | Executed: True
[2026-01-02 14:23:15] Paper trade executed: BUY BTC/USDT @ $43300.00

[2026-01-02 15:23:15] [BTC/USDT @ 1h] Candle closed: O=43300.00 H=43400.00 L=43280.00 C=43380.00
[2026-01-02 15:23:15] [BTC/USDT @ 1h] Mode: LEARNING | Signal: NEUTRAL | Confidence: 68.12% | Executed: False

[2026-01-02 16:23:15] [BTC/USDT @ 1h] Candle closed: O=43380.00 H=43420.00 L=43350.00 C=43400.00
[2026-01-02 16:23:15] [BTC/USDT @ 1h] Mode: TRADING | Signal: BUY | Confidence: 81.23% | Executed: True
[2026-01-02 16:23:15] ✓ Transitioned to TRADING mode (confidence: 81.23%)
```

### Mode Transition

```
[2026-01-02 18:45:30] ✓ [BTC/USDT @ 1h] Transitioned to TRADING mode (confidence: 82.15%)
[2026-01-02 18:45:30] Mode transition: LEARNING → TRADING

[2026-01-03 08:15:45] ← [BTC/USDT @ 1h] Transitioned to LEARNING mode (reason: Confidence dropped below 75%)
[2026-01-03 08:15:45] Mode transition: TRADING → LEARNING
```

### Retraining Event

```
[2026-01-02 22:30:00] [BTC/USDT @ 1h] Trade closed: ✗ LOSS (-2.34%)
[2026-01-02 22:30:01] [BTC/USDT @ 1h] Retraining triggered: loss_immediate
[2026-01-02 22:30:01] ⚙ [BTC/USDT_1h] Starting retraining (reason: loss_immediate)
[2026-01-02 22:32:15] [BTC/USDT_1h] Training data: 4821 train, 1205 val
[2026-01-02 22:35:42] ✓ [BTC/USDT_1h] Retraining successful! Confidence: 83.45%, Duration: 341.2s
[2026-01-02 22:35:42] ✓ [BTC/USDT @ 1h] Transitioned to TRADING mode (confidence: 83.45%)
```

---

## Troubleshooting

### Issue: No predictions being made

**Symptoms:**
```
[ERROR] Failed to get prediction for BTC/USDT @ 1h: Model not found
```

**Solution:**
```bash
# Train models for all timeframes
python src/multi_currency_system.py --train
```

### Issue: High error rate (> 5%)

**Symptoms:**
```
⚠ SAFETY WARNING: Error rate 7.23% is high
```

**Possible Causes:**
- Network connectivity issues
- Database corruption
- Invalid data

**Solution:**
1. Check network connection
2. Verify database integrity: `sqlite3 data/trading.db "PRAGMA integrity_check;"`
3. Review error logs for specific issues

### Issue: Excessive mode oscillation

**Symptoms:**
```
Mode transitions: 45 (in 12 hours)
```

**Solution:**
1. Increase hysteresis in `config.yaml`:
   ```yaml
   continuous_learning:
     confidence:
       hysteresis: 0.10  # Increase from 0.05
   ```

### Issue: Stuck in LEARNING mode

**Symptoms:**
```
Learning Mode: 95.2% (11,430 candles)
Trading Mode: 4.8% (570 candles)
```

**Possible Causes:**
- Confidence never reaches 80%
- Model quality issues
- Threshold too high

**Solution:**
1. Review model performance
2. Check if retraining is working
3. Consider adjusting threshold (temporarily):
   ```yaml
   continuous_learning:
     confidence:
       trading_threshold: 0.75  # Lower from 0.80
   ```

### Issue: News integration failing

**Symptoms:**
```
[WARNING] Sentiment feature integration failed
```

**Solution:**
1. Check API keys in `.env`
2. Verify news collector is running
3. Check API rate limits

---

## Validation Results

### Interpreting the Final Report

After 48 hours, you'll see:

```
================================================================================
FINAL VALIDATION REPORT
================================================================================

Duration: 48.0 hours

Candles Processed: 2,880
Predictions Made: 2,850
Trades Executed: 145
Mode Transitions: 8
Errors: 12

Portfolio Value: $10,485.67
Total P&L: $485.67
Win Rate: 56.21%
Max Drawdown: 8.34%

--- PRODUCTION READINESS ---
Criteria Met: 5/5
  ✓ win_rate: 0.5621 (threshold: 0.5000)
  ✓ max_drawdown: 8.34 (threshold: 15.0000)
  ✓ error_rate: 0.0042 (threshold: 0.0500)
  ✓ profitability: 485.67 (threshold: 0.0000)
  ✓ stability: 0.1667 (threshold: 2.0000)

✓ System is READY for production deployment
================================================================================
```

### Decision Matrix

| Criteria Met | Recommendation | Next Steps |
|--------------|----------------|------------|
| 5/5 | **APPROVED** | Proceed to production |
| 4/5 | **APPROVED** | Proceed with monitoring |
| 3/5 | **CONDITIONAL** | Extended validation or parameter tuning |
| < 3/5 | **NOT READY** | Fix critical issues, re-validate |

---

## Post-Validation Steps

### If APPROVED (4-5 criteria met)

1. **Review Final Report**
   ```bash
   cat validation_results/validation_report_*.json | jq .
   ```

2. **Backup Trained Models**
   ```bash
   cp -r models models_validated_$(date +%Y%m%d)
   ```

3. **Document Performance Baseline**
   - Record win rate, P&L, max drawdown
   - Use as baseline for production monitoring

4. **Proceed to Week 5: Production Deployment**

### If CONDITIONAL (3 criteria met)

1. **Extend Validation**
   ```bash
   python run_paper_trading_validation.py --duration 96  # 4 days
   ```

2. **Tune Parameters**
   - Adjust confidence threshold
   - Modify retraining triggers
   - Re-run validation

### If NOT READY (< 3 criteria met)

1. **Identify Root Cause**
   - Review error logs
   - Check model quality
   - Verify data integrity

2. **Fix Critical Issues**
   - Retrain models with more data
   - Fix bugs/errors
   - Improve feature engineering

3. **Re-run Backtesting**
   ```bash
   python tests/backtest_continuous_learning.py --days 180
   ```

4. **Re-validate with Paper Trading**

---

## Support & Resources

### Log Files
- **Validation Log:** `logs/paper_trading_validation.log`
- **Error Log:** `logs/error.log`
- **Database Log:** `data/trading.db` (query with sqlite3)

### Reports
- **Validation Reports:** `validation_results/validation_report_*.json`
- **Backtest Results:** `backtest_results/*.json`

### Monitoring
- **Real-time Monitor:** `python tests/monitor_paper_trading.py`
- **Database Queries:**
  ```sql
  -- Recent trades
  SELECT * FROM trade_outcomes ORDER BY id DESC LIMIT 10;

  -- Retraining history
  SELECT * FROM retraining_history ORDER BY id DESC LIMIT 10;

  -- Learning state
  SELECT * FROM learning_states ORDER BY id DESC LIMIT 10;
  ```

### Emergency Stop
```bash
# Graceful shutdown
Ctrl+C

# Force kill (if needed)
pkill -f run_paper_trading_validation
```

---

## Success Metrics Summary

**Minimum Requirements:**
- Win rate ≥ 50%
- Max drawdown ≤ 15%
- Error rate < 5%
- System runs stable for 48 hours
- At least 50 trades executed

**Good Performance:**
- Win rate ≥ 55%
- Max drawdown ≤ 10%
- Error rate < 1%
- P&L > 5% of capital

**Excellent Performance:**
- Win rate ≥ 60%
- Max drawdown ≤ 5%
- Error rate < 0.5%
- P&L > 10% of capital

---

**Ready to validate? Run:**
```bash
python run_paper_trading_validation.py --duration 48
```

**Good luck! 🚀**
