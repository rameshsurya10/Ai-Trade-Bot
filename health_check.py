#!/usr/bin/env python3
"""
System Health Check
===================
Verifies that all core components can load and function.
"""

import sys
import traceback
from pathlib import Path

def print_status(test_name: str, passed: bool, message: str = ""):
    """Print test result with color."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_name}")
    if message:
        print(f"       {message}")
    print()

def check_dependencies():
    """Check if required packages are installed."""
    print("="*70)
    print("1. CHECKING DEPENDENCIES")
    print("="*70)

    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'torch': 'torch',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'ccxt': 'ccxt',
        'yaml': 'pyyaml',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib'
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print_status(f"{package}", True)
        except ImportError:
            print_status(f"{package}", False, f"Run: pip install {package}")
            missing.append(package)

    return len(missing) == 0, missing

def check_config():
    """Check if config file exists and is valid."""
    print("="*70)
    print("2. CHECKING CONFIGURATION")
    print("="*70)

    config_path = Path("config.yaml")
    if not config_path.exists():
        print_status("config.yaml exists", False, "File not found")
        return False

    print_status("config.yaml exists", True)

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check critical keys
        required_keys = ['model', 'data', 'continuous_learning', 'portfolio']
        for key in required_keys:
            if key in config:
                print_status(f"Config key: {key}", True)
            else:
                print_status(f"Config key: {key}", False, "Missing from config")
                return False

        return True
    except Exception as e:
        print_status("config.yaml valid", False, str(e))
        return False

def check_models():
    """Check if trained models exist and can be loaded."""
    print("="*70)
    print("3. CHECKING TRAINED MODELS")
    print("="*70)

    models_dir = Path("models/unbreakable/ensemble")
    if not models_dir.exists():
        print_status("Models directory exists", False, f"{models_dir} not found")
        return False

    print_status("Models directory exists", True, str(models_dir))

    # Check each model file
    expected_files = {
        'tcn_lstm_attention.pt': 'PyTorch TCN-LSTM-Attention',
        'xgboost.joblib': 'XGBoost',
        'lightgbm.joblib': 'LightGBM',
        'catboost.joblib': 'CatBoost',
        'meta_learner.joblib': 'Meta-Learner (Stacking)',
        'scaler.joblib': 'Feature Scaler',
        'weights.joblib': 'Ensemble Weights'
    }

    all_exist = True
    for filename, description in expected_files.items():
        file_path = models_dir / filename
        exists = file_path.exists()
        if exists:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print_status(f"{description}", True, f"{size_mb:.2f} MB")
        else:
            print_status(f"{description}", False, f"{filename} not found")
            all_exist = False

    if not all_exist:
        return False

    # Try to load models
    print()
    print("Loading models into memory...")
    print()

    try:
        import torch
        import joblib

        # Load PyTorch model
        tcn_path = models_dir / 'tcn_lstm_attention.pt'
        checkpoint = torch.load(tcn_path, map_location='cpu', weights_only=False)
        print_status("Load PyTorch model", True, f"State dict keys: {len(checkpoint.get('model_state_dict', checkpoint).keys())}")

        # Load XGBoost
        xgb_model = joblib.load(models_dir / 'xgboost.joblib')
        print_status("Load XGBoost model", True, f"Type: {type(xgb_model).__name__}")

        # Load LightGBM
        lgb_model = joblib.load(models_dir / 'lightgbm.joblib')
        print_status("Load LightGBM model", True, f"Type: {type(lgb_model).__name__}")

        # Load CatBoost
        cat_model = joblib.load(models_dir / 'catboost.joblib')
        print_status("Load CatBoost model", True, f"Type: {type(cat_model).__name__}")

        # Load Meta-Learner
        meta_model = joblib.load(models_dir / 'meta_learner.joblib')
        print_status("Load Meta-Learner", True, f"Type: {type(meta_model).__name__}")

        # Load Scaler
        scaler = joblib.load(models_dir / 'scaler.joblib')
        print_status("Load Feature Scaler", True, f"Type: {type(scaler).__name__}")

        return True

    except Exception as e:
        print_status("Load models", False, f"Error: {str(e)}")
        traceback.print_exc()
        return False

def check_data_connection():
    """Test connection to Binance (read-only, no API keys needed)."""
    print("="*70)
    print("4. CHECKING DATA CONNECTION")
    print("="*70)

    try:
        import ccxt

        # Create Binance exchange instance (public endpoints)
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })

        print_status("CCXT library loaded", True, f"Version: {ccxt.__version__}")

        # Test public API (no authentication needed)
        print("Testing connection to Binance public API...")
        markets = exchange.load_markets()
        print_status("Load markets", True, f"{len(markets)} trading pairs available")

        # Check if BTC/USDT exists
        if 'BTC/USDT' in markets:
            print_status("BTC/USDT market exists", True)
        else:
            print_status("BTC/USDT market exists", False)
            return False

        # Try to fetch recent candles (public endpoint)
        print("Fetching last 10 BTC/USDT candles (1h timeframe)...")
        candles = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=10)

        if candles and len(candles) > 0:
            latest = candles[-1]
            timestamp, open_price, high, low, close, volume = latest
            import datetime
            dt = datetime.datetime.fromtimestamp(timestamp / 1000)

            print_status(
                "Fetch live candles",
                True,
                f"Latest: {dt} | Close: ${close:,.2f} | Volume: {volume:,.0f}"
            )
            return True
        else:
            print_status("Fetch live candles", False, "No data returned")
            return False

    except Exception as e:
        print_status("Data connection", False, str(e))
        traceback.print_exc()
        return False

def check_core_modules():
    """Check if core trading modules can be imported."""
    print("="*70)
    print("5. CHECKING CORE MODULES")
    print("="*70)

    modules_to_test = [
        ('src.data.provider', 'Unified Data Provider'),
        ('src.ml.ensemble.stacking', 'Ensemble Model'),
        ('src.learning.continuous_learner', 'Continuous Learning'),
        ('src.brokerages.base', 'Base Brokerage'),
        ('src.live_trading.runner', 'Live Trading Runner'),
        ('src.data_service', 'Data Service'),
    ]

    all_imported = True
    for module_path, description in modules_to_test:
        try:
            __import__(module_path)
            print_status(description, True, module_path)
        except Exception as e:
            print_status(description, False, f"{str(e)[:80]}")
            all_imported = False

    return all_imported

def check_database():
    """Check database setup."""
    print("="*70)
    print("6. CHECKING DATABASE")
    print("="*70)

    db_path = Path("data/trading.db")

    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print_status("Database exists", True, f"{db_path} ({size_mb:.2f} MB)")

        # Try to connect
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            print_status("Database accessible", True, f"{len(tables)} tables found")

            # Check for candles
            if ('candles',) in tables:
                cursor.execute("SELECT COUNT(*) FROM candles;")
                count = cursor.fetchone()[0]
                print_status("Historical candles", True, f"{count:,} candles stored")
            else:
                print_status("Historical candles", False, "Table 'candles' not found")

            conn.close()
            return True

        except Exception as e:
            print_status("Database accessible", False, str(e))
            return False
    else:
        print_status("Database exists", False, f"{db_path} not found (will be created on first run)")
        return True  # Not a critical error

def main():
    """Run all health checks."""
    print()
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "SYSTEM HEALTH CHECK" + " "*29 + "║")
    print("╚" + "="*68 + "╝")
    print()

    results = {}

    # Run all checks
    results['dependencies'], missing = check_dependencies()

    if not results['dependencies']:
        print()
        print("⚠️  CRITICAL: Missing dependencies detected!")
        print("    Run: pip install " + " ".join(missing))
        print()
        return False

    results['config'] = check_config()
    results['models'] = check_models()
    results['connection'] = check_data_connection()
    results['modules'] = check_core_modules()
    results['database'] = check_database()

    # Summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for check, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {check.upper()}")

    print()
    print(f"Result: {passed}/{total} checks passed")
    print()

    if passed == total:
        print("╔" + "="*68 + "╗")
        print("║" + " "*15 + "🎉 ALL SYSTEMS OPERATIONAL 🎉" + " "*22 + "║")
        print("╚" + "="*68 + "╝")
        print()
        print("Next steps:")
        print("  1. Run: python run_trading.py")
        print("  2. Let it run for 24-48 hours in PAPER mode")
        print("  3. Analyze results: python scripts/analyze_strategies.py")
        print()
        return True
    else:
        print("╔" + "="*68 + "╗")
        print("║" + " "*16 + "⚠️  ISSUES DETECTED ⚠️" + " "*27 + "║")
        print("╚" + "="*68 + "╝")
        print()
        print("Fix the failed checks above before running the trading system.")
        print()
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nHealth check cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
