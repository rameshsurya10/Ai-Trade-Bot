# Model Learning: Static vs Continuous Updates

**Your Question:** Does the model fetch live signals using static downloaded data, or does it auto-update with new candles?

---

## Current System: STATIC MODEL + LIVE DATA

### How It Works:

```
┌──────────────────────────────────────────────────────────────┐
│ 1. TRAINING PHASE (One-time, offline)                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Download Historical Data (6 months)                         │
│         ↓                                                    │
│  Train LSTM Model                                            │
│         ↓                                                    │
│  Save model weights to: data/lstm_model.pt                   │
│         ↓                                                    │
│  MODEL IS NOW FROZEN ❄️                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 2. PREDICTION PHASE (Continuous, real-time)                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Every new candle (1 hour):                                  │
│    1. Fetch LATEST 500 candles from Binance ✅ LIVE DATA    │
│    2. Calculate 28 features from new data                    │
│    3. Use FROZEN model to predict                            │
│    4. Generate signal                                        │
│                                                              │
│  Model weights: NEVER CHANGE ❌                              │
│  Input data: ALWAYS FRESH ✅                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## What "Static" Means:

### ❌ STATIC MODEL (Current System):
```python
# Training (once):
model.train(historical_data_from_january_to_june)
model.save("lstm_model.pt")  # Weights frozen here

# Prediction (continuous):
while True:
    new_candles = binance.fetch_latest_500_candles()  # ✅ FRESH DATA
    prediction = model.predict(new_candles)  # ❌ FROZEN WEIGHTS
    # Model learned from Jan-June data
    # But predicts on July-December NEW candles
```

**Problem:**
- Model trained on January-June 2024
- In December 2024, market conditions changed
- Model still uses January-June "knowledge"
- **Model does NOT learn from new data automatically**

---

## What "Continuous Learning" Would Mean:

### ✅ ONLINE LEARNING (NOT implemented in current system):
```python
# Continuous training:
while True:
    new_candles = binance.fetch_latest_500_candles()

    # Predict
    prediction = model.predict(new_candles)

    # Wait for actual outcome
    time.sleep(3600)  # 1 hour
    actual_result = check_if_prediction_was_correct()

    # UPDATE MODEL WITH NEW DATA
    model.retrain_on_new_candle(new_candles, actual_result)  # ✅ LEARNS
    model.save("lstm_model.pt")  # Weights updated
```

**Benefit:**
- Model adapts to new market conditions
- Learns from its mistakes
- Gets better over time

**Risk:**
- Can overfit to recent data
- Might forget old patterns
- Can degrade if recent data is bad

---

## Your Current System Detailed:

### File: [run_analysis.py](run_analysis.py#L150-L156)
```python
def _on_new_candles(self, df):
    """Handle new candles from data service."""
    # Get LATEST 500 candles from exchange
    full_df = self.data_service.get_candles(limit=500)  # ✅ FRESH DATA

    if len(full_df) >= 100:
        self.analysis_engine.on_new_data(full_df)
```

### File: [src/analysis_engine.py](src/analysis_engine.py#L420-L428)
```python
# LSTM NEURAL NETWORK PREDICTION
with torch.no_grad():  # ❌ NO GRADIENTS = NO LEARNING
    x = torch.FloatTensor(features).unsqueeze(0)
    lstm_prob = self._model(x).item()  # Using FROZEN weights
```

**Key Line:** `with torch.no_grad():`
- This means: "Don't calculate gradients"
- No gradients = No backpropagation = No learning
- Model only does **INFERENCE** (prediction), not **TRAINING**

---

## Data Flow Diagram:

```
TRAINING (One-time):
┌─────────────────┐
│ Historical Data │ (Jan-Jun 2024)
│  180 days       │
└────────┬────────┘
         ↓
┌────────────────┐
│  Train LSTM    │
│  100 epochs    │
└────────┬───────┘
         ↓
┌────────────────┐
│ lstm_model.pt  │ ← WEIGHTS FROZEN HERE ❄️
└────────────────┘


PREDICTION (Continuous):
Every 1 hour:
┌─────────────────┐
│ Binance API     │ ← LIVE CONNECTION
│ Get latest      │
│ 500 candles     │
└────────┬────────┘
         ↓
┌────────────────┐
│ Calculate      │
│ 28 features    │
└────────┬───────┘
         ↓
┌────────────────┐
│ Load frozen    │
│ lstm_model.pt  │ ← SAME WEIGHTS FROM TRAINING
└────────┬───────┘
         ↓
┌────────────────┐
│ Predict        │
│ (no learning)  │
└────────┬───────┘
         ↓
┌────────────────┐
│ Generate       │
│ BUY/SELL       │
└────────────────┘
```

---

## Answer to Your Question:

### Q: Is the model using static downloaded data or live data?

**A: BOTH!**

**Static (Frozen):**
- Model weights (what it learned)
- Training happened once on historical data
- Weights never change during predictions

**Live (Fresh):**
- Input data (what it predicts on)
- Fetches latest 500 candles every hour
- Always uses most recent prices

### Analogy:
```
Model = Teacher who graduated in 2020
Data = Current news the teacher reads

The teacher (model):
- Learned from 2020 textbooks ❄️ FROZEN
- Reads 2024 newspapers ✅ FRESH
- Uses 2020 knowledge to analyze 2024 news
- Does NOT learn new things from 2024 news ❌
```

---

## Auto-Update System (Optional, in multi_currency_system.py):

**File:** [src/multi_currency_system.py](src/multi_currency_system.py)

### This file DOES have auto-retraining:
```python
def needs_retrain(self) -> bool:
    """Check if model needs retraining based on performance."""
    if self.total_signals < 20:
        return False  # Not enough data
    if self.win_rate < 0.45:
        return True  # Performing below 45% - RETRAIN!

    days_since_retrain = (datetime.utcnow() - self.last_retrain).days
    return days_since_retrain >= 30  # RETRAIN every 30 days
```

### When it retrains:
1. **Win rate drops below 45%** → Retrain immediately
2. **Every 30 days** → Retrain to adapt to new market
3. **After 100 trades** → Initial retrain with real data

### How it retrains:
```python
def _schedule_retrain(self, symbol: str):
    """Schedule model retraining in background."""
    # Fetch fresh data from last 6 months
    df = self.data_service.fetch_historical(days=180)

    # Retrain model with NEW data
    success = self.trainer.train_model(symbol, df, epochs=50)

    # Save updated model
    if success:
        self.performance[symbol].last_retrain = datetime.utcnow()
```

---

## Two Systems Available:

### System 1: analysis_engine.py (CURRENT - NO AUTO-UPDATE)
```yaml
Learning: ❌ No
Data: ✅ Live (fetches new candles every hour)
Model: ❄️ Frozen (same weights forever)
When to retrain: Manual (run scripts/train_model.py)
```

### System 2: multi_currency_system.py (ADVANCED - AUTO-UPDATE)
```yaml
Learning: ✅ Yes (auto-retrain every 30 days or if win rate < 45%)
Data: ✅ Live (fetches new candles every hour)
Model: 🔄 Updated (retrains automatically)
When to retrain: Automatic
```

---

## Which System Are You Using?

### Current Setup: **analysis_engine.py** (Static Model)

**Check your run_analysis.py:**
```python
from src.analysis_engine import AnalysisEngine  # ← Static model

# NOT using:
# from src.multi_currency_system import MultiCurrencySystem  # ← Auto-retrain
```

---

## Recommendation:

### Option A: Keep Static Model (Simpler)
**Pros:**
- Predictable behavior
- No risk of overfitting to bad recent data
- Faster (no training overhead)

**Cons:**
- Model gets "stale" over time
- Doesn't adapt to new market conditions
- Must manually retrain periodically

**How to update:**
```bash
# Every 1-2 months, manually retrain:
venv/bin/python scripts/download_data.py  # Fresh data
venv/bin/python scripts/train_model.py --epochs 100  # Retrain
venv/bin/python scripts/run_backtest.py  # Verify still works
# Then restart engine
```

---

### Option B: Use Auto-Retrain System (Advanced)
**Pros:**
- Adapts to market changes automatically
- Learns from performance
- No manual intervention needed

**Cons:**
- More complex
- Risk of overfitting
- Higher resource usage (CPU for training)

**How to enable:**
```python
# Edit run_analysis.py:
from src.multi_currency_system import MultiCurrencySystem

# Change:
self.analysis_engine = AnalysisEngine(config_path)
# To:
self.multi_currency = MultiCurrencySystem(config_path)
```

**Config in config.yaml:**
```yaml
auto_training:
  enabled: true
  min_trades_before_retrain: 50
  max_days_between_retrain: 30
  min_win_rate_threshold: 0.45  # Retrain if below 45%
```

---

## Summary Table:

| Feature | Static Model (Current) | Auto-Retrain (Optional) |
|---------|----------------------|------------------------|
| **Input data** | ✅ Live (new candles hourly) | ✅ Live (new candles hourly) |
| **Model weights** | ❄️ Frozen (from training) | 🔄 Updated (every 30d or if bad) |
| **Learning** | ❌ No | ✅ Yes |
| **Adapts to market** | ❌ No (manual retrain) | ✅ Yes (automatic) |
| **Complexity** | Low | High |
| **Resource usage** | Low | High (retrains in background) |
| **Risk of overfitting** | Low | Medium |
| **Maintenance** | Manual (retrain every 1-2 months) | Automatic |

---

## Final Answer:

**Your system right now:**
- ✅ Uses **LIVE data** (fetches new candles every hour from Binance)
- ❌ Uses **STATIC model** (weights frozen from when you trained it)
- ❌ Does **NOT auto-update** the model with new candles
- ✅ You must **manually retrain** every 1-2 months to keep it fresh

**Think of it like:**
- You download a weather prediction app in January
- The app fetches LIVE weather data every day ✅
- But the prediction algorithm was coded in January ❄️
- In July, it still uses January's algorithm to analyze July's weather
- You need to download an app update to get the improved algorithm

---

## To Switch to Auto-Learning:

1. Use [src/multi_currency_system.py](src/multi_currency_system.py) instead of analysis_engine.py
2. Enable auto_training in config.yaml
3. System will retrain automatically when:
   - Win rate drops below 45%
   - 30 days since last retrain
   - After first 100 trades

---

**Which do you prefer?**
- Option A: Keep it simple (manual retrain every 1-2 months)
- Option B: Enable auto-retrain (more advanced, adapts automatically)
