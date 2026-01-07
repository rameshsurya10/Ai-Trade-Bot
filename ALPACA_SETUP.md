# Alpaca Trading Setup Guide

## ✅ What's Working

Your Alpaca integration is **fully functional**! Here's what you can do:

### Crypto Trading (Real-time WebSocket) ✅
- **BTC/USD** - Bitcoin
- **ETH/USD** - Ethereum
- **Real-time data** via WebSocket
- **$100,000** paper trading account
- **$200,000** buying power

### Stock Trading (REST API) ✅
- **All US Stocks**: AAPL, TSLA, MSFT, GOOGL, AMZN, NVDA, META
- **ETFs**: SPY, QQQ
- **Data updated** via REST API polling (15-second delay)
- **Paper trading** enabled

## 📊 How to Use Alpaca in Dashboard

### Step 1: Open Dashboard
```bash
streamlit run dashboard.py
```

### Step 2: Select Alpaca
Look at bottom-left sidebar "Quick Settings":
- **Exchange:** Change from "binance" → "alpaca"
- **Symbol:** Select stock or crypto
- **Timeframe:** Choose interval

### Step 3: Choose Asset Type

#### For Crypto (Real-time WebSocket):
- Select: **BTC/USD** or **ETH/USD**
- Data updates: Real-time (< 1 second)

#### For Stocks (REST API):
- Select: **AAPL**, **TSLA**, **MSFT**, etc.
- Data updates: Every 15 seconds
- Market hours: Mon-Fri 9:30 AM - 4:00 PM ET

## 🔑 Your Account Details

```
Account ID: PA3NCTPH34IU
Cash: $100,000
Buying Power: $200,000
Status: ACTIVE
Type: Paper Trading (Safe!)
```

## 📈 Current Prices

- **AAPL (Apple):** $263.36
- **BTC/USD:** $92,172.24

## ⚠️ Stock WebSocket Limitation

- **Free paper accounts** don't include stock WebSocket access
- **Workaround:** Dashboard uses REST API polling (works great!)
- **Upgrade option:** Alpaca paid plans include real-time stock WebSocket

## 🎯 Recommended Usage

### For Day Trading:
- Use **crypto** (BTC/USD, ETH/USD) - real-time WebSocket

### For Swing Trading:
- Use **stocks** (AAPL, TSLA, etc.) - REST API is sufficient

### For Learning:
- Try both! Everything works in paper trading mode (no real money)

## 🚀 Next Steps

1. **Test It:** Switch to Alpaca in dashboard and watch charts update
2. **Paper Trade:** Make fake trades to test strategies
3. **Backtest:** Run historical simulations
4. **Go Live (Optional):** Switch to real account when ready

## 💡 Pro Tips

- **Market hours matter** for stocks (closed weekends/holidays)
- **Crypto trades 24/7** (including weekends)
- **Paper trading is unlimited** - practice as much as you want
- **No risk** - all trades are simulated

Your setup is **production-ready**! 🎉
