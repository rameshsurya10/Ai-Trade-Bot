# Comprehensive Testing Checklist

## Pre-Deployment Testing Protocol

This checklist ensures all components work properly before production deployment.

---

## Phase 1: Import & Dependency Tests ✓

### 1.1 Core Imports
```bash
# Test all core imports
python -c "
from src.core.database import Database
from src.core.config import load_config
from src.core.metrics import MetricsTracker
from src.core.resilience import with_retry
from src.core.validation import validate_symbol
print('✓ Core imports OK')
"
```

### 1.2 Learning Components
```bash
python -c "
from src.learning.confidence_gate import ConfidenceGate
from src.learning.outcome_tracker import OutcomeTracker
from src.learning.retraining_engine import RetrainingEngine
from src.learning.continuous_learner import ContinuousLearningSystem
from src.learning.state_manager import LearningStateManager
print('✓ Learning components OK')
"
```

### 1.3 Multi-Timeframe Components
```bash
python -c "
from src.multi_timeframe.model_manager import MultiTimeframeModelManager
from src.multi_timeframe.aggregator import SignalAggregator
print('✓ Multi-timeframe components OK')
"
```

### 1.4 News Components
```bash
python -c "
from src.news.collector import NewsCollector
from src.news.sentiment import SentimentAnalyzer
from src.news.aggregator import SentimentAggregator
print('✓ News components OK')
"
```

### 1.5 Deployment Scripts
```bash
python -c "
from deploy_production import ProductionDeployment
from monitor_production import ProductionMonitor
print('✓ Deployment scripts OK')
"
```

### 1.6 Dependencies Check
```bash
python deploy_production.py --help 2>&1 | grep -i "usage"
# Should show usage without errors
```

---

## Phase 2: Configuration Tests ✓

### 2.1 Load Configuration
```bash
python -c "
from src.core.config import load_config
config = load_config('config.yaml')
print(f'✓ Config loaded: {len(config)} sections')
assert 'symbols' in config
assert 'timeframes' in config
assert 'continuous_learning' in config
print('✓ All required sections present')
"
```

### 2.2 Validate Configuration Structure
```bash
python -c "
from src.core.config import load_config
config = load_config()

# Check required keys
required = ['symbols', 'timeframes', 'continuous_learning', 'risk', 'portfolio', 'database']
for key in required:
    assert key in config, f'Missing: {key}'
    print(f'✓ {key} present')

print('✓ Configuration structure valid')
"
```

### 2.3 Environment Variables
```bash
# Check .env exists
if [ -f .env ]; then
    echo "✓ .env file exists"
    # Check for API keys (don't print values)
    grep -q "NEWSAPI_KEY" .env && echo "✓ NEWSAPI_KEY configured" || echo "⚠ NEWSAPI_KEY missing (optional)"
    grep -q "ALPHAVANTAGE_KEY" .env && echo "✓ ALPHAVANTAGE_KEY configured" || echo "⚠ ALPHAVANTAGE_KEY missing (optional)"
else
    echo "⚠ .env file not found (optional for news features)"
fi
```

---

## Phase 3: Database Tests ✓

### 3.1 Database Creation & Schema
```bash
python -c "
from src.core.database import Database
from src.core.config import load_config

config = load_config()
db = Database(config['database']['path'])

# Test schema exists
tables = [
    'candles',
    'signals',
    'learning_states',
    'trade_outcomes',
    'news_articles',
    'sentiment_features',
    'retraining_history',
    'confidence_history'
]

for table in tables:
    result = db.execute_query(f\"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{table}'\")
    if result and result[0][0] > 0:
        print(f'✓ Table {table} exists')
    else:
        print(f'✗ Table {table} MISSING')

print('✓ Database schema validated')
"
```

### 3.2 Database Integrity
```bash
sqlite3 data/trading.db "PRAGMA integrity_check;" | grep "ok" && echo "✓ Database integrity OK" || echo "✗ Database corrupted"
```

### 3.3 Database Write Test
```bash
python -c "
from src.core.database import Database
from src.core.config import load_config
from datetime import datetime

config = load_config()
db = Database(config['database']['path'])

# Test write
test_time = datetime.utcnow().isoformat()
db.execute_query(
    \"INSERT INTO learning_states (symbol, interval, mode, confidence_score, entered_at) VALUES (?, ?, ?, ?, ?)\",
    ('TEST/USDT', '1h', 'LEARNING', 0.75, test_time)
)

# Test read
result = db.execute_query(\"SELECT * FROM learning_states WHERE symbol='TEST/USDT' ORDER BY id DESC LIMIT 1\")
assert result is not None
print('✓ Database read/write OK')

# Cleanup
db.execute_query(\"DELETE FROM learning_states WHERE symbol='TEST/USDT'\")
print('✓ Database cleanup OK')
"
```

---

## Phase 4: Model Tests ✓

### 4.1 Check Models Exist
```bash
echo "Checking for trained models..."
ls -lh models/*.pt 2>/dev/null || echo "⚠ No models found - run training first"
echo ""
echo "Expected models for BTC/USDT:"
for interval in 1m 5m 15m 1h 4h 1d; do
    if [ -f "models/model_BTC_USDT_${interval}.pt" ]; then
        echo "✓ BTC/USDT @ ${interval}"
    else
        echo "✗ BTC/USDT @ ${interval} MISSING"
    fi
done
```

### 4.2 Model Loading Test
```bash
python -c "
from src.multi_timeframe.model_manager import MultiTimeframeModelManager
from src.core.config import load_config
import os

config = load_config()
manager = MultiTimeframeModelManager(config=config)

symbol = 'BTC/USDT'
interval = '1h'

model_path = manager.get_model_path(symbol, interval)
print(f'Model path: {model_path}')

if os.path.exists(model_path):
    try:
        model = manager.load_model(symbol, interval)
        print(f'✓ Model loaded successfully for {symbol} @ {interval}')
    except Exception as e:
        print(f'✗ Model load failed: {e}')
else:
    print(f'⚠ Model not found: {model_path}')
    print('Run: python src/multi_currency_system.py --train')
"
```

---

## Phase 5: Component Integration Tests ✓

### 5.1 Confidence Gate Test
```bash
python -c "
from src.learning.confidence_gate import ConfidenceGate

gate = ConfidenceGate(
    trading_threshold=0.80,
    hysteresis=0.05
)

# Test LEARNING → TRADING transition
can_trade, reason = gate.should_trade(
    confidence=0.82,
    current_mode='LEARNING',
    regime='NORMAL'
)
assert can_trade == True, 'Should allow trading at 82%'
print('✓ LEARNING → TRADING transition OK')

# Test TRADING → LEARNING transition
can_trade, reason = gate.should_trade(
    confidence=0.74,
    current_mode='TRADING',
    regime='NORMAL'
)
assert can_trade == False, 'Should exit trading at 74%'
print('✓ TRADING → LEARNING transition OK')

# Test hysteresis (stay in TRADING)
can_trade, reason = gate.should_trade(
    confidence=0.76,
    current_mode='TRADING',
    regime='NORMAL'
)
assert can_trade == True, 'Should stay in TRADING at 76% (hysteresis)'
print('✓ Hysteresis working correctly')

print('✓ Confidence gate tests passed')
"
```

### 5.2 Signal Aggregator Test
```bash
python -c "
from src.multi_timeframe.aggregator import SignalAggregator, TimeframeSignal
from datetime import datetime

# Create aggregator with weights
weights = {
    '1h': 0.30,
    '4h': 0.40,
    '1d': 0.30
}

aggregator = SignalAggregator(
    interval_weights=weights,
    aggregation_method='weighted_vote'
)

# Create test signals
signals = {
    '1h': TimeframeSignal(
        interval='1h',
        direction='BUY',
        confidence=0.85,
        lstm_prob=0.65,
        timestamp=datetime.utcnow()
    ),
    '4h': TimeframeSignal(
        interval='4h',
        direction='BUY',
        confidence=0.80,
        lstm_prob=0.60,
        timestamp=datetime.utcnow()
    ),
    '1d': TimeframeSignal(
        interval='1d',
        direction='SELL',
        confidence=0.75,
        lstm_prob=0.55,
        timestamp=datetime.utcnow()
    )
}

# Aggregate
result = aggregator.aggregate(signals)

print(f'Aggregated direction: {result.direction}')
print(f'Aggregated confidence: {result.confidence:.2%}')
assert result.direction in ['BUY', 'SELL', 'NEUTRAL']
assert 0 <= result.confidence <= 1.0
print('✓ Signal aggregation OK')
"
```

### 5.3 Sentiment Analyzer Test
```bash
python -c "
from src.news.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Test positive sentiment
pos_score = analyzer.analyze('Bitcoin is bullish and going to the moon! 🚀')
print(f'Positive text score: {pos_score:.3f}')
assert pos_score > 0, 'Should be positive'

# Test negative sentiment
neg_score = analyzer.analyze('Bitcoin crash and dump, bearish market')
print(f'Negative text score: {neg_score:.3f}')
assert neg_score < 0, 'Should be negative'

# Test neutral sentiment
neu_score = analyzer.analyze('Bitcoin price unchanged today')
print(f'Neutral text score: {neu_score:.3f}')

print('✓ Sentiment analysis working')
"
```

---

## Phase 6: Pre-Deployment Checks ✓

### 6.1 Run Automated Pre-Deployment Checks
```bash
python -c "
from deploy_production import ProductionDeployment
from src.core.config import load_config

config = load_config()
deployment = ProductionDeployment(config_path='config.yaml')

# Run pre-deployment checks
symbol = config['symbols'][0] if 'symbols' in config and len(config['symbols']) > 0 else 'BTC/USDT'

print(f'Running pre-deployment checks for {symbol}...')
print('=' * 80)

passed = deployment._pre_deployment_checks(symbol)

if passed:
    print('=' * 80)
    print('✅ ALL PRE-DEPLOYMENT CHECKS PASSED')
    print('=' * 80)
else:
    print('=' * 80)
    print('⚠ SOME CHECKS FAILED - Review output above')
    print('=' * 80)
"
```

### 6.2 Directory Structure Check
```bash
echo "Checking required directories..."
for dir in logs models data backups production_reports; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/ exists"
    else
        echo "⚠ $dir/ missing - creating..."
        mkdir -p "$dir"
        echo "  ✓ Created $dir/"
    fi
done
```

### 6.3 Permissions Check
```bash
echo "Checking file permissions..."
if [ -w data/trading.db ]; then
    echo "✓ Database writable"
else
    echo "✗ Database not writable"
fi

if [ -w logs/ ]; then
    echo "✓ Logs directory writable"
else
    echo "✗ Logs directory not writable"
fi
```

---

## Phase 7: Functional Tests ✓

### 7.1 Test Complete Prediction Pipeline (Dry Run)
```bash
python -c "
from src.advanced_predictor import UnbreakablePredictor
from src.core.config import load_config
import pandas as pd
import numpy as np

config = load_config()

print('Testing prediction pipeline...')

# Create predictor
try:
    predictor = UnbreakablePredictor(config=config)
    print('✓ Predictor initialized')
except Exception as e:
    print(f'✗ Predictor initialization failed: {e}')
    exit(1)

# Note: Full prediction test requires trained models and data
# This is a smoke test to verify imports and initialization work

print('✓ Prediction pipeline smoke test passed')
print('⚠ Full prediction test requires trained models')
"
```

### 7.2 Test Monitoring System
```bash
python -c "
from monitor_production import ProductionMonitor

monitor = ProductionMonitor()
print('✓ Monitor initialized')

# Test report generation (quick)
import sys
from io import StringIO

# Capture output
old_stdout = sys.stdout
sys.stdout = StringIO()

try:
    # This will generate report based on database
    # Even if empty, should not crash
    monitor.generate_report(hours=1)
    sys.stdout = old_stdout
    print('✓ Monitor report generation OK')
except Exception as e:
    sys.stdout = old_stdout
    print(f'✗ Monitor report failed: {e}')
"
```

---

## Phase 8: Performance & Quality Checks ✓

### 8.1 Code Quality Scan
```bash
echo "Running code quality checks..."

# Check for print statements (should use logging)
echo "Checking for print statements in production code..."
grep -r "print(" src/ --include="*.py" | grep -v "# print" | wc -l | \
    awk '{if ($1 > 0) print "⚠ Found", $1, "print statements - should use logging"; else print "✓ No print statements found"}'

# Check for hardcoded paths
echo "Checking for hardcoded paths..."
grep -r "'/home/" src/ --include="*.py" | wc -l | \
    awk '{if ($1 > 0) print "✗ Found", $1, "hardcoded paths"; else print "✓ No hardcoded paths"}'

# Check for TODO comments
echo "Checking for TODO comments..."
grep -r "TODO" src/ --include="*.py" | wc -l | \
    awk '{if ($1 > 0) print "⚠ Found", $1, "TODO comments"; else print "✓ No TODOs"}'
```

### 8.2 Security Scan
```bash
echo "Running security checks..."

# Check for hardcoded API keys (basic check)
echo "Checking for potential API keys in code..."
grep -r "api_key\s*=\s*['\"]" src/ --include="*.py" | grep -v "api_key_env" | wc -l | \
    awk '{if ($1 > 0) print "✗ CRITICAL: Found", $1, "hardcoded API keys"; else print "✓ No hardcoded API keys"}'

# Check .env is in .gitignore
if grep -q "\.env" .gitignore 2>/dev/null; then
    echo "✓ .env in .gitignore"
else
    echo "⚠ .env not in .gitignore - add it!"
fi
```

---

## Phase 9: Integration Testing ✓

### 9.1 End-to-End Smoke Test
```bash
python -c "
print('Running end-to-end smoke test...')
print('=' * 80)

from src.core.database import Database
from src.core.config import load_config
from src.learning.confidence_gate import ConfidenceGate
from src.learning.state_manager import LearningStateManager

# Initialize components
config = load_config()
db = Database(config['database']['path'])
gate = ConfidenceGate()
state_mgr = LearningStateManager(db)

print('✓ All core components initialized')

# Test workflow
symbol = 'TEST/USDT'
interval = '1h'

# Simulate mode transition
current_mode = state_mgr.get_current_mode(symbol, interval) or 'LEARNING'
print(f'Current mode: {current_mode}')

# Test confidence check
can_trade, reason = gate.should_trade(0.85, current_mode, 'NORMAL')
print(f'Can trade (85% confidence): {can_trade} - {reason}')

# Cleanup
db.execute_query(\"DELETE FROM learning_states WHERE symbol='TEST/USDT'\")

print('=' * 80)
print('✓ End-to-end smoke test PASSED')
"
```

---

## Phase 10: Documentation Verification ✓

### 10.1 Check All Documentation Files Exist
```bash
echo "Verifying documentation..."
docs=(
    "USER_GUIDE.md"
    "CONFIGURATION_GUIDE.md"
    "TROUBLESHOOTING.md"
    "MONITORING_PLAYBOOK.md"
    "PRODUCTION_DEPLOYMENT.md"
    "CRITICAL_FIXES_APPLIED.md"
    "PROJECT_SUMMARY.md"
    "tests/BACKTEST_README.md"
    "tests/PAPER_TRADING_VALIDATION.md"
)

for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        lines=$(wc -l < "$doc")
        echo "✓ $doc ($lines lines)"
    else
        echo "✗ $doc MISSING"
    fi
done
```

---

## Test Results Summary

### Quick Test Command
```bash
# Run all quick tests in sequence
bash << 'EOF'
set -e
echo "=== RUNNING COMPREHENSIVE TEST SUITE ==="
echo ""

echo "1. Testing imports..."
python -c "from deploy_production import ProductionDeployment; print('✓')"

echo "2. Testing configuration..."
python -c "from src.core.config import load_config; load_config(); print('✓')"

echo "3. Testing database..."
python -c "from src.core.database import Database; db = Database('data/trading.db'); print('✓')"

echo "4. Testing confidence gate..."
python -c "from src.learning.confidence_gate import ConfidenceGate; ConfidenceGate(); print('✓')"

echo "5. Testing signal aggregator..."
python -c "from src.multi_timeframe.aggregator import SignalAggregator; print('✓')"

echo ""
echo "=== ALL QUICK TESTS PASSED ✓ ==="
EOF
```

---

## Manual Testing Checklist

- [ ] **Import Tests** - All imports work
- [ ] **Configuration** - Config loads without errors
- [ ] **Database** - Schema exists, read/write works
- [ ] **Models** - Can load trained models
- [ ] **Confidence Gate** - Threshold logic works
- [ ] **Signal Aggregation** - Weighted voting works
- [ ] **Sentiment** - VADER analysis functional
- [ ] **Pre-Deployment** - All checks pass
- [ ] **Monitoring** - Dashboard and reports work
- [ ] **Documentation** - All guides present

---

## Next Steps After Testing

### If All Tests Pass ✅
1. Proceed to training: `python src/multi_currency_system.py --train`
2. Run backtest: `python tests/backtest_continuous_learning.py --days 90`
3. Paper trading validation: `python run_paper_trading_validation.py --duration 48`
4. Deploy to production: `python deploy_production.py --phase 1`

### If Tests Fail ✗
1. Review error messages
2. Check TROUBLESHOOTING.md for solutions
3. Fix issues
4. Re-run tests
5. Contact support if needed

---

**Testing Status:** Ready for execution
**Estimated Time:** 15-30 minutes for full suite
**Prerequisites:** Python 3.9+, dependencies installed

**Run this checklist before any deployment!**
