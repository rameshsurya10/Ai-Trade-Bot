# 🚀 How to Run the Dashboard

## Quick Start (1 Command)

```bash
venv/bin/streamlit run dashboard.py
```

Then open: **http://localhost:8501**

---

## Detailed Steps

### 1. Open Terminal

```bash
cd /home/development1/Desktop/Ai-Trade-Bot
```

### 2. Start Dashboard

```bash
venv/bin/streamlit run dashboard.py
```

You'll see:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.9.107:8501
```

### 3. Open Browser

The dashboard will auto-open, or go to: **http://localhost:8501**

### 4. Enable Auto-Refresh

In the dashboard controls:
- Find the **"Auto Refresh"** checkbox
- Make sure it's **✅ CHECKED**
- This enables live updates every 2 seconds

---

## What You'll See

```
🚀 AI Trade Bot - Modern Dashboard
📊 BTC/USDT

Current Price: $88,XXX.XX (+1.37%)
🟢 LIVE | Updates: 45 | Last: 18:45:32

[Control Panel]
Exchange: Binance | Symbol: BTC/USDT | Timeframe: 1m | ☑ Auto Refresh

[Metrics]
Updates: 45 | Candles: 200 | RSI: 64.2 | 24h High: $89,477 | 24h Low: $84,450

[Interactive Chart]
📊 Live Price Chart
[200 candlesticks with volume bars]

[Scroll Down for AI Predictions...]

🤖 AI Predictions & Mathematical Analysis
AI Signal: BUY 54.2% | Stop Loss: $88,000 | Take Profit: $88,500 | Regime: TRENDING

🧠 LSTM Deep Learning Predictions
[Feature engineering with 12 indicators]

📡 Recent Trading Signals
[Signal history table]

📋 Recent Candles
[20 latest candles with OHLCV data]
```

---

## Stop Dashboard

Press **Ctrl + C** in terminal

Or kill it:
```bash
pkill -f streamlit
```

---

## Optional: Run Analysis Engine

For continuous ML predictions:

```bash
# New terminal
venv/bin/python run_analysis.py
```

---

## Troubleshooting

### "Command not found"
```bash
# Make sure you're in the right directory
cd /home/development1/Desktop/Ai-Trade-Bot

# Check venv exists
ls venv/bin/streamlit
```

### No data showing
- Check internet connection
- Enable "Auto Refresh" checkbox
- Refresh browser (Ctrl + Shift + R)

### AI predictions missing
- Scroll down past the chart
- Wait 5 seconds for data to load
- Check terminal for errors

---

**That's it! Dashboard should be running.** 🎯
