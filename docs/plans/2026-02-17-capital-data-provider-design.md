# Capital.com Data Provider + Dual-Market Learning Flow

**Date:** 2026-02-17
**Status:** Approved

## Problem

The AI Trade Bot supports crypto (Binance) but forex learning is blocked on Linux because:
- MT5 requires a Windows bridge (not available)
- Capital.com brokerage exists (1,068 LOC) but has NO data provider
- The runner only routes forex symbols to MT5DataProvider

## Solution

### Track 1: Start Crypto Learning (Immediate)
- Paper mode, no API keys needed
- Binance WebSocket for live candle data
- Full learning pipeline: predictions + online learning + confidence tracking

### Track 2: Build CapitalDataProvider
- New polling-based data provider (~400 LOC)
- Same interface as UnifiedDataProvider (drop-in)
- Uses existing `CapitalBrokerage.get_candles()` for data
- Wire into runner's `_start_streams()` routing

## Architecture

```
LiveTradingRunner
  ├── UnifiedDataProvider (Binance WebSocket) → Crypto symbols
  ├── CapitalDataProvider (Capital.com REST)  → Forex symbols  [NEW]
  └── MT5DataProvider (MT5 polling)           → Forex symbols  [existing, Windows-only]
```

All providers fire `on_candle_closed()` → same learning pipeline.

## CapitalDataProvider Design

- **Pattern:** Polling (Capital.com has no candle WebSocket)
- **Poll interval:** Once per candle interval (1h = 3600s, 15m = 900s)
- **Backfill:** 100 candles on subscribe via REST
- **Reconnection:** Exponential backoff, 5 retries
- **Thread safety:** Single polling thread for all subscriptions
- **DB persistence:** Auto-save via `set_database()`

## Files

| File | Action | Scope |
|------|--------|-------|
| `src/data/capital_provider.py` | CREATE | ~400 LOC - polling data provider |
| `src/live_trading/runner.py` | MODIFY | ~30 lines - add Capital.com routing |
| `run_trading.py` | MODIFY | ~10 lines - Capital.com symbol addition |
| `config.yaml` | MODIFY | Capital.com credentials section |

## Config

```yaml
capital:
  enabled: true
  demo: true
  api_key_env: CAPITAL_API_KEY        # Read from env var
  identifier_env: CAPITAL_IDENTIFIER  # Read from env var
  password_env: CAPITAL_PASSWORD      # Read from env var
  poll_interval_multiplier: 0.9       # Poll at 90% of interval (catch close early)
```
