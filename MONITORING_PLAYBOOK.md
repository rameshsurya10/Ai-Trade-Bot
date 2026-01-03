# Monitoring Playbook

Operational guide for monitoring the continuous learning trading system in production.

---

## Table of Contents

1. [Monitoring Overview](#monitoring-overview)
2. [Daily Operations](#daily-operations)
3. [Key Metrics](#key-metrics)
4. [Alert Levels](#alert-levels)
5. [Dashboard Monitoring](#dashboard-monitoring)
6. [Log Monitoring](#log-monitoring)
7. [Database Queries](#database-queries)
8. [Performance Baselines](#performance-baselines)
9. [Incident Response](#incident-response)
10. [Weekly Review](#weekly-review)
11. [Monthly Optimization](#monthly-optimization)

---

## Monitoring Overview

### Monitoring Philosophy

**Proactive vs Reactive:**
- 80% automated monitoring (system self-checks)
- 15% scheduled reviews (daily/weekly)
- 5% reactive (incident response)

**What to Monitor:**
1. **System Health** - Uptime, errors, resource usage
2. **Model Performance** - Win rate, confidence, accuracy
3. **Financial Performance** - P&L, drawdown, risk metrics
4. **Learning Activity** - Retraining frequency, mode transitions
5. **Data Quality** - WebSocket health, missing data, latency

### Monitoring Tools

| Tool | Purpose | Frequency |
|------|---------|-----------|
| `monitor_production.py` | Real-time terminal dashboard | Continuous (10s refresh) |
| `dashboard_continuous_learning.py` | Web dashboard with charts | On-demand |
| Log files | Detailed event tracking | Continuous |
| Reports | Periodic summaries | Hourly, daily, weekly |
| Database queries | Ad-hoc investigation | As needed |

---

## Daily Operations

### Morning Routine (5-10 minutes)

**Time:** Start of trading day (9:00 AM local)

**1. Generate Overnight Report**
```bash
python monitor_production.py --report --hours 24
```

**Check:**
- ✅ Overall win rate ≥ 50%
- ✅ No critical alerts
- ✅ System uptime ≥ 99%
- ✅ All symbols active

**2. Review Dashboard**
```bash
streamlit run dashboard_continuous_learning.py
```

**Quick scan:**
- 🎯 Current modes (LEARNING vs TRADING)
- 📈 Confidence trends (smooth, not oscillating)
- 💰 Portfolio value (stable or growing)
- 🛡️ Safety status (all green)

**3. Check for Alerts**
```bash
grep -i "alert\|critical\|warning" logs/production_deployment.log | tail -20
```

**4. Verify Active Processes**
```bash
ps aux | grep python | grep -E "deploy|monitor"
```

**Expected output:**
```
user  12345  deploy_production.py --phase 3
user  12346  monitor_production.py
```

### Mid-Day Check (2 minutes)

**Time:** Midday (12:00 PM local)

**Quick health check:**
```bash
# 1. Check recent performance
python monitor_production.py --report --hours 6

# 2. Scan logs for errors
grep ERROR logs/production_deployment.log | tail -10

# 3. Verify no stuck processes
top -b -n 1 | grep python
```

**If all green:** No action needed.
**If warnings:** Investigate and take action.

### End-of-Day Review (10 minutes)

**Time:** End of trading day (5:00 PM local)

**1. Daily Report**
```bash
python monitor_production.py --report --hours 24 > daily_report_$(date +%Y%m%d).txt
```

**2. Database Backup**
```bash
cp data/trading.db backups/trading_$(date +%Y%m%d).db
```

**3. Performance Review**
```bash
sqlite3 data/trading.db "
SELECT
  DATE(entry_time) as date,
  COUNT(*) as trades,
  SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
  ROUND(100.0 * SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate_pct,
  ROUND(SUM(pnl_absolute), 2) as daily_pnl
FROM trade_outcomes
WHERE DATE(entry_time) = DATE('now')
GROUP BY DATE(entry_time);
"
```

**4. Update Trading Journal**

Record in journal:
```
Date: YYYY-MM-DD
Trades: X
Win Rate: X%
P&L: $X.XX
Alerts: [any alerts encountered]
Notes: [observations, changes made]
```

---

## Key Metrics

### System Health Metrics

#### Uptime
**Target:** ≥ 99%

**Check:**
```bash
# System uptime
python monitor_production.py --report --hours 24 | grep "Uptime"
```

**Alert if:** < 95%

#### Error Rate
**Target:** < 1%
**Warning:** 1-5%
**Critical:** > 5%

**Check:**
```bash
sqlite3 data/trading.db "
SELECT
  ROUND(100.0 * (SELECT COUNT(*) FROM error_log WHERE timestamp > datetime('now', '-24 hours'))
  / (SELECT COUNT(*) FROM candles WHERE timestamp > unixepoch('now') - 86400), 2) as error_rate_pct;
"
```

#### WebSocket Health
**Target:** Connected, no gaps

**Check:**
```bash
# Last candle for each symbol
sqlite3 data/trading.db "
SELECT symbol, interval, MAX(datetime(timestamp, 'unixepoch')) as last_candle
FROM candles
GROUP BY symbol, interval;
"
# Should be within last hour for active intervals
```

### Model Performance Metrics

#### Win Rate
**Minimum:** 50% (better than random)
**Target:** 55-60%
**Excellent:** > 60%

**Check:**
```bash
sqlite3 data/trading.db "
SELECT
  symbol,
  COUNT(*) as trades,
  ROUND(100.0 * SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate_pct
FROM trade_outcomes
WHERE exit_time > datetime('now', '-7 days')
GROUP BY symbol
HAVING COUNT(*) >= 20;
"
```

**Alert if:** < 45% (after 20+ trades)

#### Confidence Level
**Target:** 70-85% average

**Check:**
```bash
sqlite3 data/trading.db "
SELECT
  symbol,
  interval,
  ROUND(AVG(confidence_score) * 100, 2) as avg_confidence_pct,
  ROUND(MIN(confidence_score) * 100, 2) as min_confidence_pct,
  ROUND(MAX(confidence_score) * 100, 2) as max_confidence_pct
FROM confidence_history
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY symbol, interval;
"
```

**Alert if:**
- Average < 60%
- Never reaches 80% (stuck in LEARNING)
- High volatility (std dev > 15%)

#### Mode Distribution
**Healthy:** 40-60% in each mode

**Check:**
```bash
sqlite3 data/trading.db "
SELECT
  mode,
  COUNT(*) as state_count,
  ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM learning_states WHERE entered_at > datetime('now', '-7 days')), 2) as pct
FROM learning_states
WHERE entered_at > datetime('now', '-7 days')
GROUP BY mode;
"
```

**Alert if:**
- > 80% in LEARNING (confidence issues)
- > 80% in TRADING (overconfidence risk)

### Financial Metrics

#### Portfolio Value
**Target:** Stable or growing

**Check:**
```bash
sqlite3 data/trading.db "
SELECT
  datetime(timestamp, 'unixepoch') as time,
  total_value,
  pnl,
  ROUND(100.0 * pnl / (total_value - pnl), 2) as roi_pct
FROM portfolio_snapshots
WHERE timestamp > unixepoch('now') - 604800  -- Last week
ORDER BY timestamp DESC
LIMIT 10;
"
```

#### Drawdown
**Warning:** 10-15%
**Critical:** > 15%

**Check:**
```bash
python monitor_production.py --report --hours 168 | grep "Drawdown"
```

**Calculation:**
```
Max Drawdown = (Peak Value - Current Value) / Peak Value × 100%
```

#### Daily P&L
**Monitor:** Consistency, not just total

**Check:**
```bash
sqlite3 data/trading.db "
SELECT
  DATE(entry_time) as date,
  ROUND(SUM(pnl_absolute), 2) as daily_pnl,
  COUNT(*) as trades
FROM trade_outcomes
WHERE exit_time > datetime('now', '-30 days')
GROUP BY DATE(entry_time)
ORDER BY date DESC
LIMIT 30;
"
```

**Alert if:**
- 3+ consecutive red days
- Single day loss > 5% of portfolio

### Learning Activity Metrics

#### Retraining Frequency
**Healthy:** 1-5 per day per symbol
**Too many:** > 10 per day
**Too few:** 0 in 48 hours

**Check:**
```bash
sqlite3 data/trading.db "
SELECT
  symbol,
  interval,
  COUNT(*) as retraining_count,
  ROUND(AVG(duration_seconds), 1) as avg_duration_sec,
  ROUND(AVG(validation_confidence) * 100, 2) as avg_post_conf_pct
FROM retraining_history
WHERE triggered_at > datetime('now', '-24 hours')
GROUP BY symbol, interval;
"
```

#### Retraining Success Rate
**Target:** ≥ 80%

**Check:**
```bash
sqlite3 data/trading.db "
SELECT
  COUNT(*) as total_retrains,
  SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
  ROUND(100.0 * SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate_pct
FROM retraining_history
WHERE triggered_at > datetime('now', '-7 days');
"
```

**Alert if:** < 70%

#### Mode Transitions
**Healthy:** 5-20 per week
**Too many:** > 2 per hour (oscillating)
**Too few:** < 1 per week (stuck)

**Check:**
```bash
sqlite3 data/trading.db "
SELECT
  COUNT(*) as transitions_last_24h
FROM learning_states
WHERE entered_at > datetime('now', '-24 hours');
"
```

---

## Alert Levels

### Green (Normal)

**Indicators:**
- ✅ Win rate ≥ 55%
- ✅ Drawdown ≤ 10%
- ✅ Error rate < 1%
- ✅ System uptime ≥ 99%
- ✅ Confidence 70-85%

**Action:** None. Continue monitoring.

### Yellow (Warning)

**Indicators:**
- ⚠️ Win rate 45-55%
- ⚠️ Drawdown 10-15%
- ⚠️ Error rate 1-5%
- ⚠️ Confidence 60-70%
- ⚠️ Frequent retraining (> 5/day)

**Action:**
1. Increase monitoring frequency (hourly checks)
2. Review recent trades for patterns
3. Check logs for recurring errors
4. Consider parameter tuning

### Red (Critical)

**Indicators:**
- 🔴 Win rate < 45% (after 20+ trades)
- 🔴 Drawdown > 15%
- 🔴 Error rate > 5%
- 🔴 System crash/unresponsive
- 🔴 Rapid mode oscillation (> 2/hour)

**Action:**
1. **Immediate:** Consider rollback
2. Investigate root cause
3. Implement fixes
4. Re-validate before redeployment

---

## Dashboard Monitoring

### Web Dashboard (Streamlit)

**Start:**
```bash
streamlit run dashboard_continuous_learning.py
```

**URL:** http://localhost:8501

### Dashboard Panels

#### 1. Mode Indicator
**What to look for:**
- Current mode stable (not rapidly changing)
- Confidence level appropriate for mode
- Time in mode reasonable

**Red flags:**
- Mode changes every 5-10 minutes
- Confidence near threshold (79-81%)

#### 2. Confidence Trend Chart
**Healthy pattern:**
- Smooth trend lines
- Gradual transitions
- 70-85% range

**Unhealthy pattern:**
- Erratic spikes
- Stuck below 70%
- Sawtooth pattern (oscillation)

#### 3. News Sentiment Panel
**Check:**
- Recent articles present (not all zeros)
- Sentiment scores reasonable (-1.0 to 1.0)
- Source diversity > 0.5

**Issues:**
- All sentiment = 0.0 (news collection failing)
- All positive or all negative (API issue)

#### 4. Retraining History
**Healthy pattern:**
- Success rate ≥ 80%
- Post-training confidence ≥ 80%
- Duration 2-10 minutes

**Issues:**
- Many failures (red dots)
- Confidence not improving
- Very long duration (> 15 min)

#### 5. Safety Status
**All checks should show:**
- Drawdown: ✓ PASS (green)
- Win Rate: ✓ PASS (green)
- System Stability: ✓ PASS (green)

**If yellow/red:**
- Review metric details
- Check recent trades
- Consider intervention

---

## Log Monitoring

### Real-time Log Tail

```bash
tail -f logs/production_deployment.log
```

### What to Watch For

#### Normal Operation
```
[INFO] [BTC/USDT @ 1h] Candle closed
[INFO] [BTC/USDT] TRADING MODE: BUY @ 82.45%
[INFO] ✓ Health check passed (5 consecutive)
[INFO] [BTC/USDT_1h] Retraining successful! Confidence: 84.23%
```

#### Warning Signs
```
[WARNING] Confidence dropped to 76.12%
[WARNING] 3 consecutive losses detected
[WARNING] Retraining triggered: loss_immediate
```

#### Critical Issues
```
[ERROR] WebSocket connection lost
[ERROR] Database locked (timeout)
[ERROR] Prediction failed: Model not found
[CRITICAL] Max drawdown exceeded: 16.2%
```

### Log Analysis Commands

**Error frequency:**
```bash
grep ERROR logs/production_deployment.log | wc -l
```

**Error types:**
```bash
grep ERROR logs/production_deployment.log | awk -F: '{print $3}' | sort | uniq -c | sort -rn
```

**Recent warnings:**
```bash
grep WARNING logs/production_deployment.log | tail -20
```

**Retraining events:**
```bash
grep "Retraining" logs/production_deployment.log | tail -10
```

**Mode transitions:**
```bash
grep "Transitioned to" logs/production_deployment.log | tail -10
```

---

## Database Queries

### Portfolio Status
```sql
SELECT
  datetime('now') as current_time,
  (SELECT initial_capital FROM config LIMIT 1) as initial_capital,
  SUM(CASE WHEN status = 'closed' THEN pnl ELSE 0 END) as realized_pnl,
  SUM(CASE WHEN status = 'open' THEN (current_price - entry_price) * quantity ELSE 0 END) as unrealized_pnl,
  COUNT(CASE WHEN status = 'open' THEN 1 END) as open_positions
FROM positions;
```

### Recent Trades
```sql
SELECT
  entry_time,
  symbol,
  predicted_direction,
  CASE WHEN was_correct = 1 THEN '✓ WIN' ELSE '✗ LOSS' END as result,
  ROUND(pnl_absolute, 2) as pnl_usd,
  ROUND(pnl_percent, 2) as pnl_pct,
  ROUND(confidence, 4) as confidence
FROM trade_outcomes
ORDER BY id DESC
LIMIT 20;
```

### Model Performance by Timeframe
```sql
SELECT
  symbol,
  interval,
  COUNT(*) as trades,
  ROUND(100.0 * SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate_pct,
  ROUND(AVG(confidence), 4) as avg_confidence,
  ROUND(SUM(pnl_absolute), 2) as total_pnl
FROM trade_outcomes
WHERE exit_time > datetime('now', '-7 days')
GROUP BY symbol, interval
HAVING COUNT(*) >= 10
ORDER BY win_rate_pct DESC;
```

### Retraining Effectiveness
```sql
SELECT
  triggered_at,
  trigger_reason,
  ROUND(validation_confidence * 100, 2) as post_conf_pct,
  duration_seconds,
  n_epochs,
  status
FROM retraining_history
ORDER BY id DESC
LIMIT 10;
```

### Mode Transition History
```sql
SELECT
  datetime(entered_at) as time,
  symbol,
  interval,
  mode,
  ROUND(confidence_score * 100, 2) as confidence_pct,
  reason
FROM learning_states
ORDER BY id DESC
LIMIT 20;
```

---

## Performance Baselines

### Establish Baselines

**After first 7 days of production:**

```bash
# Generate baseline report
python monitor_production.py --report --hours 168 > baseline_report.txt
```

**Document baseline metrics:**
```
System Baseline (YYYY-MM-DD to YYYY-MM-DD)
===========================================

Win Rate:
  BTC/USDT: 56.2%
  ETH/USDT: 54.8%
  Overall: 55.5%

Confidence:
  Average: 76.3%
  TRADING mode avg: 83.1%
  LEARNING mode avg: 68.5%

Retraining:
  Frequency: 3.2 per day per symbol
  Success rate: 87.3%
  Avg duration: 4.2 minutes

Mode Distribution:
  TRADING: 52%
  LEARNING: 48%
  Transitions: 12 per week

P&L:
  Daily average: $127.45
  Best day: $342.10
  Worst day: -$89.20
  Sharpe ratio: 1.47
```

### Deviation Alerts

**Set alerts for significant deviation from baseline:**

| Metric | Baseline | Alert Threshold |
|--------|----------|-----------------|
| Win Rate | 55.5% | < 50% or > 65% |
| Avg Confidence | 76.3% | < 65% or > 90% |
| Retraining/day | 3.2 | < 1 or > 8 |
| Mode transitions/week | 12 | < 5 or > 25 |

---

## Incident Response

### Incident Levels

#### P0 (Critical) - Immediate Response

**Examples:**
- System crash
- Drawdown > 15%
- Database corruption
- Complete loss of connectivity

**Response Time:** < 15 minutes

**Actions:**
1. Execute emergency rollback
2. Assess damage
3. Identify root cause
4. Implement fix
5. Test thoroughly before redeployment

#### P1 (High) - Urgent Response

**Examples:**
- Win rate < 40% (sustained)
- Error rate > 10%
- Multiple retraining failures
- Rapid mode oscillation

**Response Time:** < 1 hour

**Actions:**
1. Stop new trades (optional)
2. Investigate logs
3. Identify pattern
4. Implement mitigation
5. Monitor closely

#### P2 (Medium) - Scheduled Response

**Examples:**
- Win rate 45-50%
- Slow performance
- Minor errors

**Response Time:** < 4 hours

**Actions:**
1. Document issue
2. Schedule investigation
3. Implement fix during maintenance window
4. Verify resolution

### Incident Documentation Template

```markdown
## Incident: [Brief Title]

**Date:** YYYY-MM-DD
**Time:** HH:MM UTC
**Severity:** P0 / P1 / P2
**Status:** Open / Resolved

### Impact
- [What was affected]
- [Financial impact if any]
- [Duration of impact]

### Timeline
- HH:MM - Issue detected
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Fix implemented
- HH:MM - System restored

### Root Cause
[Detailed explanation of what went wrong]

### Resolution
[What was done to fix it]

### Prevention
[What was changed to prevent recurrence]

### Action Items
- [ ] Update monitoring
- [ ] Adjust thresholds
- [ ] Document learnings
```

---

## Weekly Review

### Weekly Review Checklist

**Time:** Friday end-of-day (30 minutes)

**1. Performance Summary**
```bash
python monitor_production.py --report --hours 168
```

**Review:**
- [ ] Overall win rate vs baseline
- [ ] P&L vs target
- [ ] Drawdown stayed within limits
- [ ] All symbols performing

**2. Model Health**
```bash
sqlite3 data/trading.db < weekly_model_review.sql
```

**Check:**
- [ ] Retraining success rate ≥ 80%
- [ ] Confidence levels stable
- [ ] No degradation trends

**3. System Stability**
- [ ] Uptime ≥ 99.5%
- [ ] Error rate < 1%
- [ ] No critical incidents

**4. Data Quality**
- [ ] No significant data gaps
- [ ] News collection working
- [ ] All timeframes updating

**5. Tune if Needed**
```yaml
# Example: Adjust underperforming timeframe
timeframes:
  intervals:
    - interval: 5m
      weight: 0.10  # Reduce from 0.15 (underperforming)
```

**6. Update Journal**

Record weekly summary:
```
Week: YYYY-MM-DD to YYYY-MM-DD
Trades: X
Win Rate: X%
Weekly P&L: $X.XX
Notable events: [list]
Changes made: [list]
Next week focus: [priorities]
```

---

## Monthly Optimization

### Monthly Optimization Process

**Time:** First Monday of month (2-3 hours)

**1. Comprehensive Analysis**
```bash
# Generate month report
python monitor_production.py --report --hours 720 > monthly_report_$(date +%Y%m).txt
```

**2. Per-Symbol Performance**

Identify:
- Best performers (maintain current settings)
- Underperformers (tune or disable)
- Opportunities (new symbols to add)

**3. Parameter Tuning**

**If win rate declining:**
- Increase confidence threshold
- Adjust timeframe weights
- Review feature quality

**If too conservative (rare trades):**
- Lower confidence threshold
- Reduce hysteresis
- Enable more timeframes

**4. Model Retraining**

**Full retraining if:**
- Market regime changed significantly
- Win rate < 52% for 2+ weeks
- Major news events affected patterns

```bash
# Backup current models
cp -r models models_backup_$(date +%Y%m)

# Full retraining
python src/multi_currency_system.py --train --all
```

**5. Infrastructure Maintenance**

- [ ] Database vacuum and optimization
- [ ] Archive old logs (> 90 days)
- [ ] Review disk space
- [ ] Update dependencies if needed

**6. Update Documentation**

- Update baselines
- Document parameter changes
- Review and update alerts
- Update runbooks if needed

---

## Monitoring Best Practices

### Do's

✅ **Automate routine checks** - Use cron for scheduled reports
✅ **Document everything** - Trading journal, incident log
✅ **Trust the system** - Don't override without good reason
✅ **Review patterns** - Look for trends, not just point values
✅ **Maintain baselines** - Know what "normal" looks like
✅ **Test changes** - Backtest before deploying to production

### Don'ts

❌ **Don't panic trade** - Let system adapt, give it time
❌ **Don't ignore alerts** - Investigate all warnings
❌ **Don't overtune** - Change one thing at a time
❌ **Don't disable learning** - System needs to adapt
❌ **Don't neglect backups** - Always have recent backup
❌ **Don't skip reviews** - Maintain regular schedule

---

## Quick Reference Card

### Daily Checklist
```
☐ Morning report (10 min)
☐ Dashboard scan (2 min)
☐ Mid-day health check (2 min)
☐ End-of-day review (10 min)
☐ Backup database
```

### Emergency Commands
```bash
# Stop system
python deploy_production.py --rollback

# Check status
python monitor_production.py --report --hours 1

# Restart
python deploy_production.py --phase 3
```

### Key Metrics (Quick Check)
```sql
-- Win rate (last 7 days)
SELECT ROUND(100.0 * SUM(was_correct) / COUNT(*), 2) FROM trade_outcomes WHERE entry_time > datetime('now', '-7 days');

-- Current drawdown
SELECT ROUND(((MAX(total_value) - (SELECT total_value FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1)) / MAX(total_value)) * 100, 2) FROM portfolio_snapshots;

-- Errors (last 24h)
SELECT COUNT(*) FROM error_log WHERE timestamp > datetime('now', '-24 hours');
```

---

**See Also:**
- [USER_GUIDE.md](USER_GUIDE.md) - General usage
- [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) - Configuration reference
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving
- [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) - Deployment procedures
