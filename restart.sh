#!/bin/bash

echo "=================================================="
echo "🔄 RESTARTING AI Trade Bot Dashboard"
echo "=================================================="
echo

# Step 1: Stop everything
echo "1️⃣  Stopping all running processes..."
pkill -f "streamlit run dashboard" 2>/dev/null
pkill -f "run_analysis.py" 2>/dev/null
venv/bin/python stop_analysis.py 2>/dev/null
sleep 2
echo "   ✅ All processes stopped"
echo

# Step 2: Clean up PID files
echo "2️⃣  Cleaning up PID files..."
rm -f data/.analysis.pid run_analysis.pid 2>/dev/null
echo "   ✅ PID files cleaned"
echo

# Step 3: Check dependencies
echo "3️⃣  Checking dependencies..."
venv/bin/python3 -c "
import streamlit
from src.data.provider import UnifiedDataProvider
print('   ✅ Dependencies OK')
" || { echo "   ❌ Missing dependencies!"; exit 1; }
echo

# Step 4: Test WebSocket
echo "4️⃣  Testing WebSocket connection..."
timeout 8 venv/bin/python3 << 'EOF'
from src.data.provider import UnifiedDataProvider
import time

provider = UnifiedDataProvider.get_instance()
provider.subscribe('BTC/USDT', exchange='binance', interval='1m')

ticks = []

def handle_tick(tick):
    ticks.append(tick.price)

provider.on_tick(handle_tick)
provider.start()
time.sleep(5)
provider.stop()

if len(ticks) > 0:
    print(f'   ✅ WebSocket working: {len(ticks)} ticks, last price ${ticks[-1]:,.2f}')
else:
    print('   ❌ WebSocket not receiving data')
    exit(1)
EOF

echo

# Step 5: Start dashboard
echo "5️⃣  Starting dashboard..."
echo "   URL: http://localhost:8501"
echo "   To stop: Press Ctrl+C"
echo
echo "=================================================="
echo "🎯 INSTRUCTIONS:"
echo "=================================================="
echo "1. Dashboard will open in browser"
echo "2. Select: Exchange=Binance, Symbol=BTC/USDT, Timeframe=1m"
echo "3. Click '▶️ Start Stream' button"
echo "4. Watch ticks counter increase!"
echo
echo "   If nothing happens:"
echo "   - Check browser console (F12)"
echo "   - Look for '✅ Stream started' message"
echo "   - Watch terminal for 'Ticks received' logs"
echo "=================================================="
echo

sleep 2
venv/bin/streamlit run dashboard.py

