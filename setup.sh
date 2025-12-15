#!/bin/bash
# AI Trade Bot - Setup Script
# ===========================
#
# Run this script to set up everything:
#     chmod +x setup.sh
#     ./setup.sh
#
# Or:
#     bash setup.sh

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                AI TRADE BOT - SETUP                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  This script will set up everything you need.                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Project directory: $SCRIPT_DIR"

# Step 1: Create virtual environment
echo ""
echo "============================================================"
echo "STEP 1: Creating virtual environment"
echo "============================================================"

if [ -d "venv" ]; then
    echo "   ✅ Virtual environment already exists"
else
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo "   ✅ Created virtual environment"
    else
        echo "   ❌ Failed to create virtual environment"
        echo "   Try: sudo apt install python3-venv"
        exit 1
    fi
fi

# Step 2: Activate virtual environment
echo ""
echo "============================================================"
echo "STEP 2: Activating virtual environment"
echo "============================================================"

source venv/bin/activate
if [ $? -eq 0 ]; then
    echo "   ✅ Virtual environment activated"
else
    echo "   ❌ Failed to activate virtual environment"
    exit 1
fi

# Step 3: Upgrade pip
echo ""
echo "============================================================"
echo "STEP 3: Upgrading pip"
echo "============================================================"

pip install --upgrade pip --quiet
echo "   ✅ pip upgraded"

# Step 4: Install dependencies
echo ""
echo "============================================================"
echo "STEP 4: Installing dependencies"
echo "============================================================"

pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "   ✅ All dependencies installed"
else
    echo "   ⚠️  Some dependencies may have failed"
fi

# Step 5: Create directories
echo ""
echo "============================================================"
echo "STEP 5: Creating directories"
echo "============================================================"

mkdir -p data models logs
echo "   ✅ Created data/"
echo "   ✅ Created models/"
echo "   ✅ Created logs/"

# Step 6: Setup config
echo ""
echo "============================================================"
echo "STEP 6: Setting up configuration"
echo "============================================================"

if [ -f "config.yaml" ]; then
    echo "   ✅ config.yaml already exists"
else
    if [ -f "config.example.yaml" ]; then
        cp config.example.yaml config.yaml
        echo "   ✅ Created config.yaml from template"
    else
        echo "   ⚠️  No config template found"
    fi
fi

# Step 7: Verify installation
echo ""
echo "============================================================"
echo "STEP 7: Verifying installation"
echo "============================================================"

python3 -c "
import sys
errors = []

modules = [
    ('yaml', 'pyyaml'),
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('torch', 'torch'),
    ('ccxt', 'ccxt'),
]

optional = [
    ('plyer', 'plyer'),
    ('streamlit', 'streamlit'),
    ('plotly', 'plotly'),
]

for module, package in modules:
    try:
        __import__(module)
        print(f'   ✅ {module}')
    except ImportError:
        errors.append(package)
        print(f'   ❌ {module}')

print('')
print('   Optional packages:')
for module, package in optional:
    try:
        __import__(module)
        print(f'   ✅ {module}')
    except ImportError:
        print(f'   ⚠️  {module} (optional)')

if errors:
    print(f'\n   Missing required packages: {errors}')
    sys.exit(1)
"

# Step 8: Summary
echo ""
echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1. ACTIVATE VIRTUAL ENVIRONMENT (do this every time):"
echo "   source venv/bin/activate"
echo ""
echo "2. DOWNLOAD DATA (required for training):"
echo "   python scripts/download_data.py --days 365"
echo ""
echo "3. TRAIN MODEL (optional - works without it):"
echo "   python scripts/train_model.py"
echo ""
echo "4. TEST NOTIFICATIONS:"
echo "   python scripts/test_notifications.py"
echo ""
echo "5. START ANALYSIS:"
echo "   python run_analysis.py"
echo ""
echo "6. VIEW DASHBOARD (optional, in another terminal):"
echo "   source venv/bin/activate"
echo "   streamlit run dashboard.py"
echo ""
echo "7. STOP ANALYSIS:"
echo "   python stop_analysis.py"
echo ""
echo "⚠️  IMPORTANT:"
echo "   - Always activate venv first: source venv/bin/activate"
echo "   - This is a SIGNAL system - NO auto-trading"
echo "   - YOU decide when to trade manually"
echo ""
