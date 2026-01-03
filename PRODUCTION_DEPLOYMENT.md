# Production Deployment Guide

## Overview

Gradual rollout strategy for continuous learning trading system to production environment.

**Strategy:**
- **Phase 1:** Single symbol (BTC/USDT) for 24 hours
- **Phase 2:** Automated validation and approval
- **Phase 3:** Full rollout to all configured symbols
- **Monitoring:** Continuous health monitoring and alerting

---

## Prerequisites

### System Requirements

- [ ] **Paper Trading Validation Complete**
  - 48-hour validation passed
  - 4/5 criteria met (win rate, drawdown, errors, profitability, stability)
  - Results documented in `validation_results/`

- [ ] **Backtesting Results**
  - Win rate ≥ 55%
  - Sentiment integration tested
  - Multi-timeframe aggregation validated

- [ ] **Infrastructure Ready**
  - Database optimized and backed up
  - All models trained and validated
  - API keys configured (`.env`)
  - Logging infrastructure operational
  - Monitoring dashboard accessible

- [ ] **Code Quality**
  - All tests passing
  - Code reviewed (code-reviewer agent)
  - Performance optimized
  - Dead code eliminated
  - Security scan passed

### Configuration Verification

Check `config.yaml` settings:

```yaml
# Production settings
live_trading:
  mode: paper  # Start with paper, switch to live later
  enabled: true

continuous_learning:
  enabled: true
  confidence:
    trading_threshold: 0.80
    hysteresis: 0.05
  retraining:
    on_loss: true
    max_epochs: 50

# Risk limits
risk:
  max_drawdown_percent: 15.0
  daily_loss_limit: 0.05
  max_position_size_percent: 0.10

# Symbols for production
symbols:
  - BTC/USDT
  - ETH/USDT
  # Add more after Phase 1 validation
```

### Pre-Deployment Checklist

```bash
# 1. Backup database
cp data/trading.db data/trading_backup_$(date +%Y%m%d_%H%M%S).db

# 2. Backup models
cp -r models models_backup_$(date +%Y%m%d)

# 3. Verify models exist
ls models/*.pt

# 4. Check database integrity
sqlite3 data/trading.db "PRAGMA integrity_check;"

# 5. Test API connectivity
python -c "from src.data.provider import UnifiedDataProvider; p = UnifiedDataProvider.get_instance(); print('✓ Provider OK')"

# 6. Verify historical data
sqlite3 data/trading.db "SELECT symbol, COUNT(*) FROM candles GROUP BY symbol;"

# 7. Check logs directory
mkdir -p logs production_reports

# 8. Test dashboard
streamlit run dashboard_continuous_learning.py --server.headless true &
sleep 5
pkill -f streamlit
```

---

## Phase 1: Single Symbol Deployment (24 Hours)

### Goal
Validate system stability and safety with one symbol in production environment.

### Steps

#### 1. Start Phase 1 Deployment

```bash
# Terminal 1: Deploy single symbol
python deploy_production.py --phase 1 --symbol BTC/USDT

# You should see:
# ================================================================================
# PHASE 1: SINGLE SYMBOL DEPLOYMENT
# ================================================================================
# Symbol: BTC/USDT
# Duration: 24 hours
# Start Time: 2026-01-02 12:00:00
# ================================================================================
#
# Running pre-deployment checks...
# ✓ Database integrity: OK
# ✓ All models exist (6 timeframes)
# ✓ Historical data: 12,345 candles
# ✓ Configuration valid
# ✓ Data provider initialized
# ================================================================================
# Pre-deployment checks: 5/5 passed
# ================================================================================
#
# Initializing system components...
# ✓ Components initialized
# Deploying BTC/USDT...
#   ✓ Subscribed to BTC/USDT @ 1m
#   ✓ Subscribed to BTC/USDT @ 5m
#   ✓ Subscribed to BTC/USDT @ 15m
#   ✓ Subscribed to BTC/USDT @ 1h
#   ✓ Subscribed to BTC/USDT @ 4h
#   ✓ Subscribed to BTC/USDT @ 1d
# ✅ BTC/USDT deployed successfully
#
# ✓ Deployment started. Monitoring BTC/USDT for 24 hours...
```

#### 2. Start Real-time Monitor (Separate Terminal)

```bash
# Terminal 2: Production monitoring
python monitor_production.py --symbol BTC/USDT

# Displays:
# ====================================================================================================
# PRODUCTION MONITOR - 2026-01-02 12:05:00 UTC
# ====================================================================================================
#
# 📊 SYSTEM STATUS
#   Uptime: 5m 0s
#   Error Rate (1h): 0.00%
#   Active Symbols: 1
#
# ====================================================================================================
# 📈 BTC/USDT
# ====================================================================================================
#   🎯 Mode: TRADING
#   Confidence: 82.45%
#   Time in mode: 15m 30s
#
#   💰 Performance (24h)
#     ✓ P&L: $125.50 (1.26%)
#     Win Rate: 58.33% (7/12 trades)
#     Max Drawdown: 3.21%
#
#   🎯 Recent Activity
#     [11:58] Trade: BUY ✓ WIN ($45.20)
#     [11:45] Retrain: ✓ loss_immediate → 82.45%
#     [11:30] Trade: SELL ✗ LOSS (-$12.30)
#
# ====================================================================================================
# Last Update: 12:05:00 | Refresh: 10s | Alerts Sent: 0
```

#### 3. Monitor Dashboard (Web Interface)

```bash
# Terminal 3: Start Streamlit dashboard
streamlit run dashboard_continuous_learning.py

# Open browser: http://localhost:8501
# Monitor:
# - Mode indicator (TRADING/LEARNING)
# - Confidence trends
# - News sentiment
# - Retraining history
# - Multi-timeframe signals
```

#### 4. Watch Logs

```bash
# Terminal 4: Tail production logs
tail -f logs/production_deployment.log

# Look for:
# - Candle close events
# - Predictions and signals
# - Mode transitions
# - Retraining events
# - Health check results
# - Any errors or warnings
```

### Monitoring During Phase 1

#### Every Hour

- [ ] **Check Monitor Dashboard**
  - Portfolio value stable or growing
  - No critical alerts
  - Win rate ≥ 50%
  - Drawdown < 10%

- [ ] **Review Logs**
  - No repeated errors
  - Retraining working as expected
  - Mode transitions reasonable (< 2 per hour)

#### Every 6 Hours

- [ ] **Generate Progress Report**
  ```bash
  python monitor_production.py --report --hours 6
  ```

- [ ] **Check Database Size**
  ```bash
  ls -lh data/trading.db
  # Should grow steadily but not excessively
  ```

- [ ] **Verify Retraining**
  ```sql
  sqlite3 data/trading.db "
    SELECT COUNT(*), AVG(validation_confidence)
    FROM retraining_history
    WHERE triggered_at > datetime('now', '-6 hours');
  "
  ```

#### Daily

- [ ] **Full System Health Check**
  - Review 24-hour report
  - Check all safety criteria
  - Verify no data anomalies
  - Backup database

### Health Check Criteria

System automatically monitors:

| Check | Threshold | Action if Failed |
|-------|-----------|------------------|
| Max Drawdown | ≤ 15% | Automatic rollback |
| Win Rate | ≥ 40% (after 20 trades) | Alert + investigation |
| Error Rate | < 5% | Alert + investigation |
| Mode Stability | < 2 transitions/hour | Alert |
| System Uptime | > 95% | Alert |

### Expected Behavior

**Normal Operation:**

```
[12:23:15] [BTC/USDT @ 1h] Candle closed: O=43250.00 C=43300.00
[12:23:15] [BTC/USDT] 6 timeframe predictions aggregated
[12:23:15] [BTC/USDT] TRADING MODE: BUY @ 82.45% (live)
[12:23:15] ✓ Paper trade executed: BUY BTC/USDT @ $43300.00
[12:23:15] ✓ Health check passed (3 consecutive)
```

**Mode Transition:**

```
[14:45:30] [BTC/USDT @ 1h] Trade closed: ✗ LOSS (-2.34%)
[14:45:31] [BTC/USDT @ 1h] Retraining triggered: loss_immediate
[14:45:31] ⚙ [BTC/USDT_1h] Starting retraining (reason: loss_immediate)
[14:48:15] ✓ [BTC/USDT_1h] Retraining successful! Confidence: 83.45%, Duration: 164.2s
[14:48:15] ✓ [BTC/USDT @ 1h] Transitioned to TRADING mode (confidence: 83.45%)
```

**Retraining Event:**

```
[16:30:00] [BTC/USDT @ 4h] Trade closed: ✗ LOSS (-1.89%)
[16:30:01] [BTC/USDT @ 4h] Retraining triggered: loss_immediate
[16:30:01] ⚙ [BTC/USDT_4h] Starting retraining (reason: loss_immediate)
[16:30:01] [BTC/USDT_4h] Training data: 4821 train, 1205 val
[16:32:45] Epoch 15/50: Loss=0.0234, Val Acc=87.45%, Val Conf=81.23%
[16:33:12] ✓ Target confidence reached: 81.23%
[16:33:12] ✓ [BTC/USDT_4h] Retraining successful! Confidence: 81.23%, Duration: 191.5s
[16:33:12] ✓ [BTC/USDT @ 4h] Transitioned to TRADING mode (confidence: 81.23%)
```

### After 24 Hours

System will automatically validate results:

```
================================================================================
PHASE 1 COMPLETE - VALIDATING RESULTS
================================================================================
Validating Phase 1 results...
================================================================================
PHASE 1 VALIDATION RESULTS
================================================================================
Criteria Met: 5/5

✓ win_rate: 0.5621 (threshold: 0.5000)
✓ max_drawdown: 8.34 (threshold: 15.0000)
✓ error_rate: 0.0042 (threshold: 0.0500)
✓ profitability: 485.67 (threshold: 0.0000)
✓ stability: 0.1667 (threshold: 2.0000)

Verdict: APPROVED
================================================================================
✅ Phase 1 APPROVED - Ready for Phase 3 (Full Rollout)
```

### If Phase 1 Fails

If system does NOT meet 4/5 criteria:

1. **Stop Deployment**
   ```bash
   # Press Ctrl+C in Terminal 1
   # System will gracefully shutdown
   ```

2. **Analyze Results**
   ```bash
   # Generate detailed report
   python monitor_production.py --report --hours 24

   # Check which criterion failed
   cat production_reports/report_*.json | jq '.alerts'
   ```

3. **Identify Root Cause**
   - Review logs: `tail -1000 logs/production_deployment.log`
   - Check database: `sqlite3 data/trading.db`
   - Analyze trades: Check win rate, drawdown, error patterns

4. **Fix Issues**
   - Adjust parameters in `config.yaml`
   - Retrain models if needed
   - Fix any bugs identified

5. **Re-run Paper Trading Validation**
   ```bash
   python run_paper_trading_validation.py --duration 48
   ```

6. **Restart Phase 1**

---

## Phase 3: Full Production Rollout

### Prerequisites

- ✅ Phase 1 approved (4/5 criteria met)
- ✅ No critical alerts during Phase 1
- ✅ Retraining working correctly
- ✅ User approval to proceed

### Steps

#### 1. Review Phase 1 Results

```bash
# Generate final Phase 1 report
python monitor_production.py --report --hours 24

# Review report
cat production_reports/report_*.json | jq '.'
```

#### 2. Update Configuration (if needed)

Based on Phase 1 learnings, adjust `config.yaml`:

```yaml
# Example adjustments
continuous_learning:
  confidence:
    trading_threshold: 0.82  # Increase if too aggressive
    hysteresis: 0.07         # Increase if oscillating

  retraining:
    target_confidence: 0.82  # Match trading threshold
    patience: 12             # Increase if retraining too often

# Add more symbols
symbols:
  - BTC/USDT   # Validated in Phase 1
  - ETH/USDT   # Add for Phase 3
  - SOL/USDT   # Add for Phase 3
  - BNB/USDT   # Add for Phase 3
```

#### 3. Train Models for New Symbols

```bash
# Train models for all new symbols and timeframes
for symbol in ETH/USDT SOL/USDT BNB/USDT; do
    echo "Training $symbol..."
    python src/multi_currency_system.py --train --symbol $symbol
done
```

#### 4. Start Phase 3 Deployment

```bash
# Terminal 1: Deploy all symbols
python deploy_production.py --phase 3

# You should see:
# ================================================================================
# PHASE 3: FULL PRODUCTION ROLLOUT
# ================================================================================
# Deploying to 4 symbols:
#   - BTC/USDT
#   - ETH/USDT
#   - SOL/USDT
#   - BNB/USDT
# ================================================================================
# ✓ BTC/USDT already deployed (Phase 1)
# Deploying ETH/USDT...
#   ✓ Subscribed to ETH/USDT @ 1m
#   ✓ Subscribed to ETH/USDT @ 5m
#   ...
# ✅ ETH/USDT deployed successfully
# [Wait 30 seconds]
# Deploying SOL/USDT...
# ...
# ================================================================================
# ✅ FULL ROLLOUT COMPLETE - 4 symbols active
# ================================================================================
# Continuous monitoring active. Press Ctrl+C to stop.
```

#### 5. Monitor All Symbols

```bash
# Terminal 2: Production monitoring (all symbols)
python monitor_production.py

# Shows dashboard for all active symbols
```

### Monitoring During Phase 3

#### Continuous

- Real-time monitoring dashboard (Terminal 2)
- Web dashboard (`streamlit run dashboard_continuous_learning.py`)
- Log tailing (`tail -f logs/production_deployment.log`)

#### Hourly

- Check all symbols for health
- Verify no critical alerts
- Review recent trades

#### Every 6 Hours

```bash
# Generate comprehensive report
python monitor_production.py --report --hours 6

# Check per-symbol performance
cat production_reports/report_*.json | jq '.symbols'
```

#### Daily

```bash
# Full 24-hour report
python monitor_production.py --report --hours 24

# Backup database
cp data/trading.db backups/trading_$(date +%Y%m%d).db

# Backup models
cp -r models models_backup_$(date +%Y%m%d)
```

### Performance Tuning

After 48 hours of Phase 3:

1. **Analyze Per-Symbol Performance**
   ```bash
   python monitor_production.py --report --hours 48
   ```

2. **Identify Underperforming Symbols**
   - Win rate < 50%
   - Excessive retraining
   - High error rate

3. **Tune Parameters** (in `config.yaml`)
   ```yaml
   # Example: Adjust per-timeframe weights
   timeframes:
     intervals:
       - interval: 1h
         weight: 0.30  # Increase if 1h performs well
       - interval: 4h
         weight: 0.25  # Decrease if underperforming
   ```

4. **Retrain Models with More Data**
   ```bash
   # If a symbol/timeframe underperforms, retrain
   python src/multi_currency_system.py --train --symbol ETH/USDT --interval 4h
   ```

5. **Monitor Improvements**
   - Compare before/after performance
   - Allow 24 hours for validation
   - Iterate as needed

---

## Emergency Procedures

### Rollback

If critical issues occur:

```bash
# Immediate rollback
python deploy_production.py --rollback

# You should see:
# ================================================================================
# INITIATING ROLLBACK
# ================================================================================
# Stopping continuous learning system...
# Closing all open positions...
# Disconnecting data provider...
# ✓ Rollback complete. System stopped.
# Data preserved in database for analysis.
```

### Troubleshooting

#### Issue: High Error Rate

**Symptoms:**
```
⚠️  WARNING: Error rate 7.23% exceeds 5% threshold
```

**Diagnosis:**
```bash
# Check recent errors
sqlite3 data/trading.db "
  SELECT timestamp, error_type, message
  FROM error_log
  ORDER BY timestamp DESC
  LIMIT 20;
"
```

**Solutions:**
1. Network issues → Check API connectivity
2. Data quality → Verify WebSocket connection
3. Code bugs → Review stack traces, fix, redeploy

#### Issue: Stuck in LEARNING Mode

**Symptoms:**
```
Mode: LEARNING (95% of time)
Confidence: 68-75% (never reaches 80%)
```

**Diagnosis:**
```bash
# Check confidence history
sqlite3 data/trading.db "
  SELECT AVG(confidence_score), MAX(confidence_score)
  FROM confidence_history
  WHERE symbol = 'BTC/USDT'
  AND timestamp > datetime('now', '-24 hours');
"
```

**Solutions:**
1. Lower threshold temporarily:
   ```yaml
   continuous_learning:
     confidence:
       trading_threshold: 0.75  # Lower from 0.80
   ```

2. Retrain models with more data
3. Review feature quality (especially sentiment)

#### Issue: Excessive Retraining

**Symptoms:**
```
Retraining events: 25 in last 6 hours
```

**Diagnosis:**
```bash
# Check retraining triggers
sqlite3 data/trading.db "
  SELECT trigger_reason, COUNT(*)
  FROM retraining_history
  WHERE triggered_at > datetime('now', '-6 hours')
  GROUP BY trigger_reason;
"
```

**Solutions:**
1. If `loss_immediate` is too frequent:
   ```yaml
   continuous_learning:
     retraining:
       min_interval_hours: 2  # Add cooldown
   ```

2. If `drift` triggers too often:
   ```yaml
   continuous_learning:
     retraining:
       drift_threshold: 0.80  # Increase from 0.70
   ```

#### Issue: High Drawdown

**Symptoms:**
```
🔴 CRITICAL: Drawdown 12.5% approaching 15% limit
```

**Immediate Actions:**
1. Stop live trading (if enabled)
2. Switch to paper trading only
3. Analyze losing trades

**Investigation:**
```bash
# Find worst trades
sqlite3 data/trading.db "
  SELECT symbol, predicted_direction, pnl_absolute, confidence
  FROM trade_outcomes
  WHERE pnl_absolute < 0
  ORDER BY pnl_absolute ASC
  LIMIT 10;
"
```

**Prevention:**
```yaml
# Tighten risk controls
risk:
  max_position_size_percent: 0.05  # Reduce from 0.10
  daily_loss_limit: 0.03           # Reduce from 0.05
  max_drawdown_percent: 12.0       # Lower threshold
```

---

## Success Criteria

### Phase 1 Success (24 hours, single symbol)

**Minimum (4/5 required):**
- ✅ Win rate ≥ 50%
- ✅ Max drawdown ≤ 15%
- ✅ Error rate < 5%
- ✅ Profitability > $0
- ✅ Stability < 2 mode transitions/hour

**Good Performance:**
- Win rate ≥ 55%
- Max drawdown ≤ 10%
- Error rate < 1%
- P&L > 3% of capital
- < 1 transition/hour

### Phase 3 Success (48+ hours, all symbols)

**Minimum:**
- All symbols meeting Phase 1 minimum criteria
- System uptime ≥ 99%
- No critical failures
- Retraining working correctly

**Good Performance:**
- Overall win rate ≥ 55%
- Overall P&L positive across all symbols
- Sharpe ratio ≥ 1.2
- Max drawdown ≤ 10%

---

## Post-Deployment

### Week 1: Intensive Monitoring

- [ ] Check dashboard every 2 hours
- [ ] Generate daily reports
- [ ] Tune parameters as needed
- [ ] Document any issues

### Week 2-4: Stabilization

- [ ] Monitor daily
- [ ] Weekly performance review
- [ ] Optimize underperforming symbols
- [ ] Consider adding more symbols

### Ongoing

- [ ] Weekly backup (database + models)
- [ ] Monthly performance review
- [ ] Quarterly parameter optimization
- [ ] Continuous monitoring for drift

---

## Rollout to Live Trading

**ONLY after successful paper trading in production for ≥ 2 weeks:**

1. **Verify Consistent Performance**
   - Win rate ≥ 55% sustained
   - No critical failures
   - Retraining working reliably

2. **Update Configuration**
   ```yaml
   live_trading:
     mode: live  # Change from paper
     enabled: true

   # Start with small position sizes
   risk:
     max_position_size_percent: 0.02  # Very conservative
   ```

3. **Deploy Gradual**
   - Start with 1 symbol
   - Small position sizes
   - Monitor for 1 week
   - Gradually increase

4. **Risk Management**
   - Daily loss limits enforced
   - Stop-loss on all positions
   - Real-time monitoring
   - Quick rollback capability

---

## Support

### Log Files
- `logs/production_deployment.log` - Deployment logs
- `logs/error.log` - Error tracking
- `production_reports/*.json` - Performance reports

### Database Queries
```sql
-- Recent performance
SELECT * FROM trade_outcomes ORDER BY id DESC LIMIT 20;

-- Retraining history
SELECT * FROM retraining_history ORDER BY id DESC LIMIT 10;

-- Current mode
SELECT * FROM learning_states ORDER BY id DESC LIMIT 5;

-- Portfolio value
SELECT SUM(pnl) FROM positions WHERE status = 'closed';
```

### Emergency Contacts
- System Admin: [contact info]
- DevOps: [contact info]
- On-call: [contact info]

---

## Checklist Summary

**Before Phase 1:**
- [ ] Paper trading validation complete (4/5 criteria)
- [ ] Database backed up
- [ ] Models trained and validated
- [ ] Configuration verified
- [ ] Pre-deployment checks passed (5/5)

**During Phase 1:**
- [ ] Deployment script running (Terminal 1)
- [ ] Monitor running (Terminal 2)
- [ ] Dashboard accessible (Terminal 3)
- [ ] Logs being tailed (Terminal 4)
- [ ] Hourly health checks
- [ ] 6-hour progress reports

**Phase 1 Complete:**
- [ ] 24 hours elapsed
- [ ] Validation passed (4/5 criteria)
- [ ] No critical alerts
- [ ] Ready for Phase 3

**Before Phase 3:**
- [ ] Phase 1 approved
- [ ] New symbols configured
- [ ] New models trained
- [ ] User approval obtained

**During Phase 3:**
- [ ] All symbols deployed
- [ ] Continuous monitoring active
- [ ] Hourly checks on all symbols
- [ ] Daily backups

**Phase 3 Success:**
- [ ] 48+ hours stable operation
- [ ] All symbols performing well
- [ ] System ready for long-term production

---

**Ready to deploy?**

```bash
# Start Phase 1
python deploy_production.py --phase 1 --symbol BTC/USDT
```

**Good luck! 🚀**
