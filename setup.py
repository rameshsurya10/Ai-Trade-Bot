#!/usr/bin/env python3
"""
AI Trade Bot - Setup Script
===========================

Run this script to set up everything:
    python setup.py

This will:
1. Install all required dependencies
2. Create necessary directories
3. Copy config template
4. Test notifications
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print setup banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                AI TRADE BOT - SETUP                          ║
╠══════════════════════════════════════════════════════════════╣
║  This script will set up everything you need.                ║
╚══════════════════════════════════════════════════════════════╝
    """)


def run_command(cmd, description):
    """Run a command safely without shell injection risk."""
    print(f"\n📦 {description}...")
    try:
        # Split command into list for safer execution (no shell=True)
        cmd_list = cmd.split() if isinstance(cmd, str) else cmd
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"   ✅ {description} - Done!")
            return True
        else:
            print(f"   ⚠️  {description} - Warning: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"   ❌ {description} - Error: {e}")
        return False


def main():
    print_banner()

    project_dir = Path(__file__).parent
    os.chdir(project_dir)

    print(f"📁 Project directory: {project_dir}")

    # Step 1: Create directories
    print("\n" + "="*60)
    print("STEP 1: Creating directories")
    print("="*60)

    dirs = ['data', 'models', 'logs']
    for d in dirs:
        dir_path = project_dir / d
        dir_path.mkdir(exist_ok=True)
        print(f"   ✅ Created {d}/")

    # Step 2: Install dependencies
    print("\n" + "="*60)
    print("STEP 2: Installing dependencies")
    print("="*60)

    # Upgrade pip first
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    )

    # Install requirements
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies from requirements.txt"
    )

    if not success:
        print("\n⚠️  Some dependencies may have failed to install.")
        print("   Try installing manually:")
        print("   pip install pandas numpy torch ccxt pyyaml plyer streamlit plotly")

    # Step 3: Copy config if needed
    print("\n" + "="*60)
    print("STEP 3: Setting up configuration")
    print("="*60)

    config_path = project_dir / "config.yaml"
    example_path = project_dir / "config.example.yaml"

    if config_path.exists():
        print("   ✅ config.yaml already exists")
    elif example_path.exists():
        import shutil
        shutil.copy(example_path, config_path)
        print("   ✅ Created config.yaml from template")
    else:
        print("   ⚠️  No config template found")

    # Step 4: Verify installation
    print("\n" + "="*60)
    print("STEP 4: Verifying installation")
    print("="*60)

    verification_script = '''
import sys
errors = []

try:
    import yaml
    print("   ✅ yaml")
except ImportError:
    errors.append("yaml")
    print("   ❌ yaml")

try:
    import pandas
    print("   ✅ pandas")
except ImportError:
    errors.append("pandas")
    print("   ❌ pandas")

try:
    import numpy
    print("   ✅ numpy")
except ImportError:
    errors.append("numpy")
    print("   ❌ numpy")

try:
    import torch
    print("   ✅ torch")
except ImportError:
    errors.append("torch")
    print("   ❌ torch")

try:
    import ccxt
    print("   ✅ ccxt")
except ImportError:
    errors.append("ccxt")
    print("   ❌ ccxt")

try:
    from plyer import notification
    print("   ✅ plyer")
except ImportError:
    errors.append("plyer")
    print("   ❌ plyer (optional - fallback available)")

try:
    import streamlit
    print("   ✅ streamlit")
except ImportError:
    errors.append("streamlit")
    print("   ❌ streamlit (optional - for dashboard)")

sys.exit(len([e for e in errors if e not in ["plyer", "streamlit"]]))
'''

    result = subprocess.run(
        [sys.executable, "-c", verification_script],
        capture_output=True,
        text=True
    )
    print(result.stdout)

    # Step 5: Summary
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)

    print("""
📋 NEXT STEPS:

1. DOWNLOAD DATA (required for training):
   python scripts/download_data.py --days 365

2. TRAIN MODEL (optional - works without it):
   python scripts/train_model.py

3. TEST NOTIFICATIONS:
   python scripts/test_notifications.py

4. START ANALYSIS:
   python run_analysis.py

5. VIEW DASHBOARD (optional):
   streamlit run dashboard.py

6. STOP ANALYSIS:
   python stop_analysis.py

📝 CONFIGURATION:
   Edit config.yaml to customize:
   - Trading pair (BTC-USD, ETH-USD, etc.)
   - Update interval
   - Notification settings
   - Telegram alerts (optional)

⚠️  IMPORTANT:
   - This is a SIGNAL system - NO auto-trading
   - YOU decide when to trade manually
   - Analysis runs 24/7 until YOU stop it
   - Desktop notifications work even with browser closed
    """)


if __name__ == "__main__":
    main()
