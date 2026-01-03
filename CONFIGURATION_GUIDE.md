# Configuration Guide

Complete reference for configuring the continuous learning trading system.

---

## Table of Contents

1. [Configuration File Structure](#configuration-file-structure)
2. [Core Settings](#core-settings)
3. [Timeframes Configuration](#timeframes-configuration)
4. [Continuous Learning Settings](#continuous-learning-settings)
5. [Risk Management](#risk-management)
6. [News & Sentiment](#news--sentiment)
7. [Portfolio Settings](#portfolio-settings)
8. [Model Configuration](#model-configuration)
9. [Database Settings](#database-settings)
10. [Advanced Settings](#advanced-settings)
11. [Environment Variables](#environment-variables)
12. [Configuration Examples](#configuration-examples)

---

## Configuration File Structure

The main configuration file is `config.yaml` in the project root.

**Basic structure:**
```yaml
# Trading symbols
symbols: [...]

# Exchange settings
exchange: "binance"

# Multi-timeframe configuration
timeframes: {...}

# Continuous learning settings
continuous_learning: {...}

# Risk management
risk: {...}

# News & sentiment
news: {...}

# Portfolio
portfolio: {...}

# Model settings
model: {...}

# Database
database: {...}
```

---

## Core Settings

### Symbols

**Syntax:**
```yaml
symbols:
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT
```

**Parameters:**
- Format: `BASE/QUOTE` (e.g., BTC/USDT)
- Supported: Any symbols available on configured exchange
- Minimum: 1 symbol
- Recommended: Start with 1-3 symbols

**Example:**
```yaml
# Conservative (1 symbol)
symbols:
  - BTC/USDT

# Moderate (2-3 symbols)
symbols:
  - BTC/USDT
  - ETH/USDT
  - BNB/USDT

# Aggressive (5+ symbols)
symbols:
  - BTC/USDT
  - ETH/USDT
  - BNB/USDT
  - SOL/USDT
  - MATIC/USDT
```

### Exchange

**Syntax:**
```yaml
exchange: "binance"
```

**Supported Exchanges:**
- `binance` - Binance (default)
- Future: `alpaca`, `coinbase`, `kraken`

---

## Timeframes Configuration

### Overview

Multi-timeframe configuration controls which intervals to analyze and how to combine their signals.

### Basic Structure

```yaml
timeframes:
  enabled: true  # Master switch
  aggregation_method: weighted_vote  # How to combine signals

  intervals:
    - interval: 1m     # Timeframe identifier
      enabled: true    # Enable/disable this timeframe
      weight: 0.10     # Influence on final signal
      sequence_length: 120  # Lookback candles
      min_training_interval_minutes: 60  # Retraining cooldown

    # ... more intervals
```

### Aggregation Methods

**`weighted_vote` (Recommended)**
- Combines signals using weighted average
- Higher weight = more influence
- Formula: `Σ(weight × confidence × direction)`

**`majority`**
- Most common direction wins
- Ignores confidence levels
- Good for high-agreement scenarios

**`alignment_required`**
- ALL timeframes must agree
- Very conservative
- Low trade frequency

**Example:**
```yaml
timeframes:
  aggregation_method: weighted_vote  # Change to: majority, alignment_required
```

### Interval Configuration

Each timeframe has these settings:

#### `interval` (Required)

**Supported intervals:**
- `1m` - 1 minute
- `5m` - 5 minutes
- `15m` - 15 minutes
- `1h` - 1 hour
- `4h` - 4 hours
- `1d` - 1 day

#### `enabled` (Required)

```yaml
enabled: true   # Active
enabled: false  # Disabled
```

Use to temporarily disable without deleting configuration.

#### `weight` (Required)

**Range:** 0.0 to 1.0
**Total should sum to 1.0** (not enforced, but recommended)

**Guidelines:**
- Higher timeframe = higher weight (generally)
- 1h, 4h: 0.20-0.30 (most reliable)
- 1d: 0.15-0.20 (long-term trend)
- 15m: 0.15-0.20 (short-term confirmation)
- 5m, 1m: 0.10-0.15 (noise reduction)

**Example:**
```yaml
# Balanced configuration
intervals:
  - interval: 1m
    weight: 0.10
  - interval: 5m
    weight: 0.15
  - interval: 15m
    weight: 0.15
  - interval: 1h
    weight: 0.25  # Highest weight
  - interval: 4h
    weight: 0.20
  - interval: 1d
    weight: 0.15
# Total: 1.00
```

#### `sequence_length`

**Purpose:** How many candles to look back for prediction

**Guidelines by interval:**
- 1m: 120-240 (2-4 hours of data)
- 5m: 72-144 (6-12 hours)
- 15m: 96-192 (1-2 days)
- 1h: 60-120 (2.5-5 days)
- 4h: 42-84 (7-14 days)
- 1d: 30-90 (1-3 months)

**Impact:**
- Higher = more context, slower training
- Lower = faster training, less context

**Example:**
```yaml
- interval: 1h
  sequence_length: 60  # 60 hours = 2.5 days
```

#### `min_training_interval_minutes`

**Purpose:** Minimum time between retrainings (cooldown)

**Guidelines:**
- 1m: 30-60 minutes
- 5m: 60-120 minutes
- 15m: 120-180 minutes
- 1h: 240-480 minutes (4-8 hours)
- 4h: 480-960 minutes (8-16 hours)
- 1d: 1440+ minutes (24+ hours)

**Example:**
```yaml
- interval: 1h
  min_training_interval_minutes: 240  # 4 hours minimum
```

### Complete Timeframes Example

```yaml
timeframes:
  enabled: true
  aggregation_method: weighted_vote

  intervals:
    # Short-term (fast signals, low weight)
    - interval: 1m
      enabled: false  # Often too noisy
      weight: 0.05
      sequence_length: 120
      min_training_interval_minutes: 60

    - interval: 5m
      enabled: true
      weight: 0.15
      sequence_length: 72
      min_training_interval_minutes: 60

    - interval: 15m
      enabled: true
      weight: 0.15
      sequence_length: 96
      min_training_interval_minutes: 120

    # Medium-term (reliable, high weight)
    - interval: 1h
      enabled: true
      weight: 0.25  # Highest
      sequence_length: 60
      min_training_interval_minutes: 240

    - interval: 4h
      enabled: true
      weight: 0.20
      sequence_length: 42
      min_training_interval_minutes: 480

    # Long-term (trend confirmation)
    - interval: 1d
      enabled: true
      weight: 0.20
      sequence_length: 30
      min_training_interval_minutes: 1440
```

---

## Continuous Learning Settings

### Confidence Gate

**Purpose:** Controls when to trade live vs paper

```yaml
continuous_learning:
  confidence:
    trading_threshold: 0.80  # Enter TRADING mode
    hysteresis: 0.05         # Exit at 0.75 (prevents oscillation)
    smoothing_alpha: 0.3     # EMA smoothing (0-1)
    regime_adjustment: true  # Adjust by market regime
```

#### `trading_threshold`

**Range:** 0.5 to 0.95
**Default:** 0.80 (80%)

**Guidelines:**
- 0.75: Aggressive (more trades, higher risk)
- 0.80: **Recommended** (balanced)
- 0.85: Conservative (fewer trades, safer)
- 0.90: Very conservative (rare trades)

#### `hysteresis`

**Range:** 0.01 to 0.10
**Default:** 0.05

**Purpose:** Prevents rapid mode switching

**How it works:**
- Enter TRADING: confidence ≥ 0.80
- Exit TRADING: confidence < 0.75 (0.80 - 0.05)

**Guidelines:**
- 0.03: Less stable (more transitions)
- 0.05: **Recommended**
- 0.10: Very stable (rare transitions)

#### `smoothing_alpha`

**Range:** 0.0 to 1.0
**Default:** 0.3

**Purpose:** Exponential moving average smoothing

**Formula:** `smoothed = alpha * new + (1 - alpha) * old`

**Guidelines:**
- 0.1: Slow adaptation
- 0.3: **Recommended**
- 0.5: Fast adaptation
- 1.0: No smoothing

#### `regime_adjustment`

**Default:** true

**Purpose:** Adjust threshold by market conditions

**Adjustments:**
- TRENDING: -5% (easier to trade)
- CHOPPY: +5% (harder to trade)
- VOLATILE: +10% (much harder)

**Example:**
```yaml
# Normal markets
continuous_learning:
  confidence:
    trading_threshold: 0.80
    regime_adjustment: false  # Fixed 80%

# Adaptive (recommended)
continuous_learning:
  confidence:
    trading_threshold: 0.80
    regime_adjustment: true  # 75-90% depending on regime
```

### Retraining Configuration

```yaml
continuous_learning:
  retraining:
    on_loss: true                 # Retrain on any loss
    consecutive_loss_threshold: 3  # Or 3 consecutive
    drift_threshold: 0.7          # Or concept drift detected
    min_interval_hours: 1         # Cooldown between retrains
    max_epochs: 50               # Max training epochs
    target_confidence: 0.80      # Train until this confidence
    patience: 10                 # Early stopping patience
```

#### Triggers

**`on_loss`** (Recommended: true)
- Retrain immediately after ANY losing trade
- User requirement for immediate adaptation

**`consecutive_loss_threshold`**
- Backup trigger if `on_loss` disabled
- Default: 3 consecutive losses

**`drift_threshold`**
- Range: 0.0 to 1.0
- Default: 0.7
- Triggers on concept drift detection

#### Training Parameters

**`max_epochs`**
- Maximum training iterations
- Default: 50
- Range: 20-100

**`target_confidence`**
- Train until this validation confidence
- Default: 0.80 (matches trading threshold)
- Must be ≥ trading_threshold

**`patience`**
- Early stopping if no improvement
- Default: 10
- Range: 5-20

**`min_interval_hours`**
- Cooldown between retrains
- Prevents excessive retraining
- Default: 1 hour

### Experience Replay

```yaml
continuous_learning:
  experience_replay:
    buffer_size: 10000       # Max samples to store
    replay_mix_ratio: 0.3    # 30% replay, 70% new data
    prioritize_losses: true  # Weight losses 2.0x
```

#### `buffer_size`

**Default:** 10000 samples

**Guidelines:**
- 5000: Small memory footprint
- 10000: **Recommended**
- 20000: Large history

#### `replay_mix_ratio`

**Range:** 0.0 to 0.5
**Default:** 0.3 (30% replay)

**Purpose:** Mix old samples with new data

**Guidelines:**
- 0.1: Mostly new data
- 0.3: **Recommended**
- 0.5: Equal mix

#### `prioritize_losses`

**Default:** true

**Impact:**
- true: Losses weighted 2.0x (learn from mistakes)
- false: All samples equal weight

### EWC (Elastic Weight Consolidation)

```yaml
continuous_learning:
  ewc:
    lambda: 1000.0          # Regularization strength
    fisher_sample_size: 200  # Samples for Fisher matrix
```

#### `lambda`

**Default:** 1000.0

**Purpose:** Prevents catastrophic forgetting

**Guidelines:**
- 100: Weak regularization (fast forgetting)
- 1000: **Recommended**
- 10000: Strong regularization (slow adaptation)

#### `fisher_sample_size`

**Default:** 200 samples

**Purpose:** Compute parameter importance

**Guidelines:**
- 100: Fast computation
- 200: **Recommended**
- 500: High accuracy

### Online Learning

```yaml
continuous_learning:
  online_learning:
    enabled: true           # Incremental updates every candle
    learning_rate: 0.0001  # Small for stability
    update_frequency: 1    # Every N candles
```

#### `enabled`

**Default:** true
**Recommended:** true

**Purpose:** Learn from every candle completion

#### `learning_rate`

**Default:** 0.0001
**Range:** 0.00001 to 0.001

**Guidelines:**
- 0.00001: Very slow learning
- 0.0001: **Recommended**
- 0.001: Fast learning (risky)

---

## Risk Management

```yaml
risk:
  max_drawdown_percent: 15.0     # Stop if portfolio drops 15%
  daily_loss_limit: 0.05         # Stop if -5% in a day
  max_position_size_percent: 0.10  # Max 10% per position
  stop_loss_percent: 0.02        # 2% stop loss
  take_profit_percent: 0.04      # 4% take profit (2:1 R:R)
```

### `max_drawdown_percent`

**Default:** 15.0 (15%)
**Range:** 5.0 to 30.0

**Purpose:** Maximum allowed portfolio decline

**Action:** Automatic rollback if exceeded

**Guidelines:**
- 5-10%: Conservative
- 15%: **Recommended**
- 20-30%: Aggressive (risky)

### `daily_loss_limit`

**Default:** 0.05 (5%)
**Range:** 0.01 to 0.10

**Purpose:** Daily loss circuit breaker

**Action:** Stop trading for the day

### `max_position_size_percent`

**Default:** 0.10 (10% of portfolio)
**Range:** 0.02 to 0.25

**Purpose:** Diversification / risk per trade

**Guidelines:**
- 0.02-0.05: Very conservative
- 0.10: **Recommended**
- 0.20+: Aggressive (risky)

### `stop_loss_percent` & `take_profit_percent`

**Defaults:**
- Stop Loss: 2%
- Take Profit: 4% (2:1 risk/reward)

**Risk/Reward Ratios:**
```yaml
# 1:1 (breakeven at 50% win rate)
stop_loss_percent: 0.02
take_profit_percent: 0.02

# 2:1 (recommended)
stop_loss_percent: 0.02
take_profit_percent: 0.04

# 3:1 (aggressive)
stop_loss_percent: 0.02
take_profit_percent: 0.06
```

---

## News & Sentiment

```yaml
news:
  enabled: true

  sources:
    newsapi:
      enabled: true
      api_key_env: NEWSAPI_KEY  # Environment variable
      fetch_interval: 1800      # 30 minutes
      max_requests_per_day: 100

    alphavantage:
      enabled: true
      api_key_env: ALPHAVANTAGE_KEY
      fetch_interval: 7200  # 2 hours
      max_requests_per_day: 25

  sentiment:
    analyzer: vader
    custom_lexicon:
      bullish: 0.8
      bearish: -0.8
      moon: 0.6
      dump: -0.7
      # Add more crypto-specific terms

  features:
    lookback_hours: [1, 6, 24]  # Time windows
    time_weighted: true         # Recent news weighted more
    decay_rate: 0.1            # Exponential decay
```

### Enabling/Disabling News

**Disable all news:**
```yaml
news:
  enabled: false
```

**Disable specific source:**
```yaml
news:
  enabled: true
  sources:
    newsapi:
      enabled: false  # Disable NewsAPI
    alphavantage:
      enabled: true   # Keep Alpha Vantage
```

### Fetch Intervals

**Purpose:** How often to collect news

**NewsAPI (100 requests/day):**
- 1800 seconds (30 min) = 48 requests/day ✓

**Alpha Vantage (25 requests/day):**
- 7200 seconds (2 hours) = 12 requests/day ✓

**Adjust based on your API limits.**

### Custom Lexicon

**Add crypto-specific sentiment terms:**
```yaml
sentiment:
  custom_lexicon:
    # Positive
    bullish: 0.8
    moon: 0.6
    pump: 0.7
    breakout: 0.6
    rally: 0.7

    # Negative
    bearish: -0.8
    dump: -0.7
    crash: -0.9
    selloff: -0.7
    correction: -0.5

    # Neutral/Technical
    consolidation: 0.0
    support: 0.3
    resistance: -0.3
```

---

## Portfolio Settings

```yaml
portfolio:
  initial_capital: 10000.0    # Starting capital (USD)
  fee_percent: 0.1           # 0.1% trading fee
  slippage_percent: 0.05     # 0.05% slippage
  max_open_positions: 5      # Max concurrent positions
```

### `initial_capital`

**Purpose:** Starting portfolio value

**Guidelines:**
- Paper trading: Use realistic amount
- Live trading: Start small (1-5% of real capital)

### `fee_percent`

**Default:** 0.1% (Binance taker fee)

**Exchange fees:**
- Binance: 0.1% (maker/taker)
- Coinbase Pro: 0.5%
- Kraken: 0.16-0.26%

### `slippage_percent`

**Default:** 0.05% (5 basis points)

**Guidelines:**
- High liquidity (BTC/USDT): 0.01-0.05%
- Medium liquidity: 0.05-0.10%
- Low liquidity: 0.10-0.50%

---

## Model Configuration

```yaml
model:
  features:
    technical_features: 32     # OHLCV indicators
    sentiment_features: 7      # News sentiment
    total_features: 39         # 32 + 7
    include_sentiment: true    # Use sentiment

  architecture:
    hidden_size: 128          # LSTM hidden dimension
    num_layers: 2             # LSTM depth
    dropout: 0.2              # Dropout rate
    bidirectional: false      # Unidirectional LSTM

  ab_testing:
    enabled: true             # Track both predictions
    track_both_predictions: true
```

### Feature Configuration

**`include_sentiment`**
- true: 39 features (technical + sentiment)
- false: 32 features (technical only)

**Impact:**
- With sentiment: ~2-5% better win rate
- Without sentiment: Still functional

### Architecture

**`hidden_size`**
- Default: 128
- Range: 64 to 256
- Impact: Model capacity

**`num_layers`**
- Default: 2
- Range: 1 to 4
- Impact: Model depth

**`dropout`**
- Default: 0.2 (20%)
- Range: 0.1 to 0.5
- Purpose: Prevent overfitting

---

## Database Settings

```yaml
database:
  path: data/trading.db      # SQLite database path
  wal_mode: true            # Write-Ahead Logging
  timeout: 30.0             # Query timeout (seconds)
  backup_interval_hours: 24  # Auto-backup frequency
```

### `wal_mode`

**Default:** true
**Recommended:** true

**Purpose:** Concurrent reads/writes

### `timeout`

**Default:** 30.0 seconds

**Purpose:** Prevent hung queries

---

## Advanced Settings

### Logging

```yaml
logging:
  level: INFO              # DEBUG, INFO, WARNING, ERROR
  file: logs/trading.log
  max_size_mb: 10
  backup_count: 3
```

### Backtesting

```yaml
backtest:
  initial_cash: 100000.0
  max_hold_candles: 24     # Max position duration
  use_trailing_stop: false
  trailing_stop_percent: 1.0
```

---

## Environment Variables

Store sensitive data in `.env` file:

```bash
# News API Keys
NEWSAPI_KEY=your_newsapi_key_here
ALPHAVANTAGE_KEY=your_alphavantage_key_here

# Exchange API Keys (future)
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret

# Notifications (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

**Never commit `.env` to git!**

---

## Configuration Examples

### Conservative (Low Risk)

```yaml
symbols:
  - BTC/USDT  # Single symbol

continuous_learning:
  confidence:
    trading_threshold: 0.85  # High threshold
    hysteresis: 0.08         # Wide band

risk:
  max_drawdown_percent: 10.0
  max_position_size_percent: 0.05

timeframes:
  intervals:
    - interval: 1h
      weight: 0.30
    - interval: 4h
      weight: 0.35  # Favor higher timeframes
    - interval: 1d
      weight: 0.35
```

### Moderate (Balanced)

```yaml
symbols:
  - BTC/USDT
  - ETH/USDT
  - BNB/USDT

continuous_learning:
  confidence:
    trading_threshold: 0.80  # Standard
    hysteresis: 0.05

risk:
  max_drawdown_percent: 15.0
  max_position_size_percent: 0.10

timeframes:
  intervals:
    - interval: 5m
      weight: 0.15
    - interval: 15m
      weight: 0.15
    - interval: 1h
      weight: 0.25
    - interval: 4h
      weight: 0.20
    - interval: 1d
      weight: 0.25
```

### Aggressive (High Frequency)

```yaml
symbols:
  - BTC/USDT
  - ETH/USDT
  - BNB/USDT
  - SOL/USDT
  - MATIC/USDT

continuous_learning:
  confidence:
    trading_threshold: 0.75  # Lower threshold
    hysteresis: 0.03         # Narrow band

risk:
  max_drawdown_percent: 20.0
  max_position_size_percent: 0.15

timeframes:
  intervals:
    - interval: 1m
      enabled: true
      weight: 0.10
    - interval: 5m
      weight: 0.20  # Favor shorter timeframes
    - interval: 15m
      weight: 0.25
    - interval: 1h
      weight: 0.25
    - interval: 4h
      weight: 0.15
    - interval: 1d
      weight: 0.05
```

---

## Validation

After editing `config.yaml`, validate:

```bash
python -c "from src.core.config import load_config; c = load_config(); print('✓ Config valid')"
```

---

## Best Practices

1. **Start Conservative** - Use moderate config, tune based on results
2. **One Change at a Time** - Adjust one parameter, monitor for 24-48 hours
3. **Document Changes** - Comment your config with reasons
4. **Backup Before Changing** - `cp config.yaml config_backup.yaml`
5. **Test in Backtest First** - Verify parameter changes in backtest before production

---

**See Also:**
- [USER_GUIDE.md](USER_GUIDE.md) - General usage
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) - Deployment guide
