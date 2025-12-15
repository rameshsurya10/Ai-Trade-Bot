#!/bin/bash
# AI Trade Bot - Start Control Panel
# ===================================
#
# Double-click this file or run:
#     ./start.sh
#
# This opens the Control Panel UI where you can:
# - Download data
# - Train model
# - Start/Stop analysis
# - View signals
# - All with buttons!

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           AI TRADE BOT - STARTING CONTROL PANEL              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run setup first: bash setup.sh"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Read port from config (default 8501)
PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('dashboard',{}).get('port',8501))" 2>/dev/null || echo "8501")

echo "✅ Virtual environment activated"
echo ""
echo "🚀 Starting Control Panel..."
echo "   This will open in your browser at: http://localhost:${PORT}"
echo ""
echo "   Press Ctrl+C to stop the Control Panel"
echo ""

# Start streamlit with port from config (suppress non-critical websocket warnings)
streamlit run dashboard.py --server.port ${PORT} --server.headless true --logger.level error 2>/dev/null
