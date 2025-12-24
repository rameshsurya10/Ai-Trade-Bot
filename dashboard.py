"""
AI Trade Bot - Modern Real-Time Dashboard (REBUILT - ACTUALLY WORKS)
====================================================================
No WebSocket complexity. Just simple REST API that WORKS.
Real-time updates with modern broker-style UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import time
import logging
import subprocess
import os
import signal

# Import CCXT for SIMPLE REST API (no WebSocket complexity)
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    st.error("❌ ccxt not installed. Run: pip install ccxt")

# Import AI/Math modules for predictions and analysis
try:
    from src.advanced_predictor import AdvancedPredictor
    from src.math_engine import MathEngine
    from src.analysis_engine import FeatureCalculator
    from src.multi_currency_system import MultiCurrencySystem
    from src.data_service import DataService
    from src.core.database import Database
    from src.notifier import Notifier

    # Import advanced dashboard features
    from src.backtesting.visual_backtester import VisualBacktester
    from src.paper_trading import PaperTradingSimulator, OrderSide, OrderType
    from src.dashboard_features import render_backtesting_interface, render_paper_trading
    from src.dashboard_features_part2 import render_portfolio_tracking, render_risk_management, render_realtime_alerts

    import yaml
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    MultiCurrencySystem = None
    DataService = None
    Database = None
    Notifier = None
    logger.warning(f"AI modules not available: {e}")

# Setup
ROOT = Path(__file__).parent
PID_FILE = ROOT / "run_analysis.pid"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Trade Bot Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force Light Theme CSS - Clean Professional UI
st.markdown("""
<style>
    /* Force Light Theme Globally */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stSidebar"], [data-testid="stSidebarContent"],
    .main, .block-container {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Sidebar Light Theme */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }

    [data-testid="stSidebar"] * {
        color: #1a1a2e !important;
    }

    [data-testid="stSidebar"] label {
        color: #1a1a2e !important;
    }

    /* All text elements */
    p, span, div, h1, h2, h3, h4, h5, h6, label,
    .stMarkdown, .stText, [data-testid="stMarkdownContainer"] {
        color: #1a1a2e !important;
    }

    /* Selectbox and inputs */
    .stSelectbox label, .stSlider label, .stCheckbox label,
    .stRadio label, .stTextInput label, .stNumberInput label {
        color: #1a1a2e !important;
    }

    .stSelectbox > div > div {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
    }

    /* Tables */
    .stDataFrame, table, th, td {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #1a1a2e !important;
        background-color: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        color: #1a1a2e !important;
    }

    .streamlit-expanderContent {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
    }

    /* Header Container */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1rem;
        color: white !important;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .header-container * {
        color: white !important;
    }

    .price-display {
        font-size: 3rem;
        font-weight: 800;
        color: white !important;
    }

    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .status-live {
        background: #38ef7d;
        color: #000 !important;
    }

    .status-offline {
        background: #f5576c;
        color: white !important;
    }

    /* Metric Cards - Light Theme */
    .metric-card {
        background: #ffffff !important;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
        border: 1px solid #e0e0e0;
    }

    .metric-card * {
        color: #1a1a2e !important;
    }

    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0;
        color: #1a1a2e !important;
    }

    .metric-card .metric-label {
        font-size: 0.75rem;
        color: #555555 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .metric-card.bullish { border-left-color: #38ef7d; }
    .metric-card.bearish { border-left-color: #f5576c; }

    /* Info boxes */
    .stAlert, [data-testid="stAlert"] {
        background-color: #f0f7ff !important;
        color: #1a1a2e !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #667eea !important;
        color: white !important;
        border: none !important;
    }

    .stButton > button:hover {
        background-color: #5a6fd6 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1a1a2e !important;
    }

    [data-testid="stMetricLabel"] {
        color: #555555 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #38ef7d !important;
    }

    /* JSON display */
    .stJson {
        background-color: #f8f9fa !important;
    }

    /* Code blocks */
    code, pre {
        background-color: #f8f9fa !important;
        color: #1a1a2e !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'update_count' not in st.session_state:
    st.session_state.update_count = 0
if 'selected_exchange' not in st.session_state:
    st.session_state.selected_exchange = 'binance'
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'BTC/USDT'
if 'selected_timeframe' not in st.session_state:
    st.session_state.selected_timeframe = '1m'
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None
# Multi-currency tracking
if 'tracked_currencies' not in st.session_state:
    st.session_state.tracked_currencies = ['BTC/USDT']
# Algorithm weights (adjustable)
if 'algorithm_weights' not in st.session_state:
    st.session_state.algorithm_weights = {
        'fourier': 0.15, 'kalman': 0.25, 'entropy': 0.10,
        'markov': 0.20, 'lstm': 0.30
    }
# Dashboard view mode
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'trading'  # trading, analysis, config, advanced
# Show advanced panels
if 'show_algorithm_details' not in st.session_state:
    st.session_state.show_algorithm_details = False
if 'show_all_indicators' not in st.session_state:
    st.session_state.show_all_indicators = False

# Advanced Features - Paper Trading Simulator
if 'paper_trader' not in st.session_state:
    if AI_AVAILABLE:
        st.session_state.paper_trader = PaperTradingSimulator(initial_capital=10000, commission=0.001)
    else:
        st.session_state.paper_trader = None

# Advanced Features - Database
if 'db' not in st.session_state:
    if AI_AVAILABLE:
        db_path = ROOT / "data" / "trading.db"
        st.session_state.db = Database(str(db_path))
    else:
        st.session_state.db = None

# Helper functions (defined before use)
def is_analysis_running():
    """Check if analysis engine is running."""
    if not PID_FILE.exists():
        return False, None
    try:
        pid_text = PID_FILE.read_text().strip()
        if not pid_text:
            PID_FILE.unlink(missing_ok=True)
            return False, None

        pid = int(pid_text)
        if pid <= 0:
            logger.warning(f"Invalid PID in file: {pid}")
            PID_FILE.unlink(missing_ok=True)
            return False, None

        os.kill(pid, 0)  # Check if process exists (doesn't actually kill it)
        return True, pid
    except (ProcessLookupError, ValueError) as e:
        logger.debug(f"PID file invalid or process dead: {e}")
        PID_FILE.unlink(missing_ok=True)
        return False, None
    except Exception as e:
        logger.error(f"Unexpected error checking process: {e}")
        return False, None

def fetch_market_data(exchange_obj, symbol, timeframe='1m', limit=100):
    """Fetch REAL market data using REST API (NO WebSocket complexity)."""
    # Validate inputs
    if not isinstance(symbol, str) or '/' not in symbol:
        return {'ticker': None, 'df': None, 'success': False, 'error': 'Invalid symbol format. Use BTC/USDT format.'}

    if limit < 1 or limit > 1000:
        return {'ticker': None, 'df': None, 'success': False, 'error': 'Limit must be between 1-1000'}

    try:
        # Fetch ticker (current price, 24h stats)
        ticker = exchange_obj.fetch_ticker(symbol)

        # Fetch OHLCV (candlestick data)
        ohlcv = exchange_obj.fetch_ohlcv(symbol, timeframe, limit=limit)

        # Convert to DataFrame and drop unused timestamp column
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop(columns=['timestamp'])  # Remove unused column to save memory

        return {
            'ticker': ticker,
            'df': df,
            'success': True,
            'error': None
        }
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.error(f"Exchange error fetching data: {e}")
        return {
            'ticker': None,
            'df': None,
            'success': False,
            'error': f'Exchange error: {str(e)}'
        }
    except Exception as e:
        logger.error(f"Unexpected error fetching data: {e}")
        return {
            'ticker': None,
            'df': None,
            'success': False,
            'error': f'Error: {str(e)}'
        }

def calculate_rsi(prices, period=14):
    """Calculate RSI from price data."""
    if len(prices) < period:
        return 50.0

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ==============================================================================
# INITIALIZATION (after helper functions are defined)
# ==============================================================================

# Initialize exchange
if 'exchange_obj' not in st.session_state and CCXT_AVAILABLE:
    try:
        st.session_state.exchange_obj = ccxt.binance({'enableRateLimit': True})
        logger.info("✅ Exchange initialized: Binance")
    except Exception as e:
        logger.error(f"Failed to initialize exchange: {e}")
        st.session_state.exchange_obj = None

# Initialize AI/Math engines
if 'predictor' not in st.session_state and AI_AVAILABLE:
    try:
        st.session_state.predictor = AdvancedPredictor()
        st.session_state.math_engine = MathEngine()
        logger.info("✅ AI/Math engines initialized")
    except Exception as e:
        logger.error(f"Failed to initialize AI engines: {e}")
        st.session_state.predictor = None
        st.session_state.math_engine = None

# Auto-start Analysis Engine on first load
if 'engine_auto_started' not in st.session_state:
    st.session_state.engine_auto_started = True
    running, pid = is_analysis_running()
    if not running:
        logger.info("🚀 Auto-starting Analysis Engine...")
        try:
            venv_py = ROOT / "venv" / "bin" / "python"
            if venv_py.exists():
                subprocess.Popen(
                    [str(venv_py), "run_analysis.py"],
                    cwd=str(ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.info("✅ Analysis Engine started automatically")
                time.sleep(2)  # Give it time to create PID file
        except Exception as e:
            logger.error(f"Failed to auto-start Analysis Engine: {e}")


def main():
    # ==========================================================================
    # SIDEBAR - All Controls & Configuration
    # ==========================================================================
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")

        # View Mode Selection
        st.markdown("### 📊 Dashboard View")
        view_options = ["Trading", "Analysis", "Advanced", "Configuration"]
        default_view = st.session_state.view_mode.title()
        if default_view not in view_options:
            default_view = "Trading"

        view_mode = st.radio(
            "Select View",
            view_options,
            index=view_options.index(default_view),
            horizontal=False
        )
        st.session_state.view_mode = view_mode.lower()

        st.markdown("---")

        # Multi-Currency Management
        st.markdown("### 💱 Multi-Currency")
        available_currencies = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT',
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD'
        ]
        selected_currency = st.selectbox(
            "Active Currency",
            available_currencies,
            index=available_currencies.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in available_currencies else 0
        )
        if selected_currency != st.session_state.selected_symbol:
            st.session_state.selected_symbol = selected_currency

        # Add/Remove currencies
        with st.expander("Manage Tracked Currencies"):
            new_currency = st.selectbox("Add Currency", [c for c in available_currencies if c not in st.session_state.tracked_currencies])
            if st.button("➕ Add", key="add_currency"):
                if new_currency not in st.session_state.tracked_currencies:
                    st.session_state.tracked_currencies.append(new_currency)
                    st.rerun()

            if len(st.session_state.tracked_currencies) > 1:
                remove_currency = st.selectbox("Remove Currency", st.session_state.tracked_currencies)
                if st.button("➖ Remove", key="remove_currency"):
                    st.session_state.tracked_currencies.remove(remove_currency)
                    st.rerun()

            st.write(f"**Tracking:** {', '.join(st.session_state.tracked_currencies)}")

        st.markdown("---")

        # Algorithm Weights (Adjustable)
        st.markdown("### 🎛️ Algorithm Weights")
        with st.expander("Adjust Weights", expanded=False):
            weights = st.session_state.algorithm_weights
            weights['fourier'] = st.slider("Fourier (Cycles)", 0.0, 1.0, weights['fourier'], 0.05)
            weights['kalman'] = st.slider("Kalman (Trend)", 0.0, 1.0, weights['kalman'], 0.05)
            weights['entropy'] = st.slider("Entropy (Regime)", 0.0, 1.0, weights['entropy'], 0.05)
            weights['markov'] = st.slider("Markov (States)", 0.0, 1.0, weights['markov'], 0.05)
            weights['lstm'] = st.slider("LSTM (Deep Learning)", 0.0, 1.0, weights['lstm'], 0.05)

            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                for k in weights:
                    weights[k] = weights[k] / total
            st.session_state.algorithm_weights = weights

            st.write("**Current Weights:**")
            for alg, w in weights.items():
                st.write(f"- {alg.title()}: {w:.1%}")

        st.markdown("---")

        # Display Options
        st.markdown("### 👁️ Display Options")
        st.session_state.show_algorithm_details = st.checkbox(
            "Show Algorithm Details",
            value=st.session_state.show_algorithm_details
        )
        st.session_state.show_all_indicators = st.checkbox(
            "Show All 27 Indicators",
            value=st.session_state.show_all_indicators
        )

        st.markdown("---")

        # Configuration Panel
        if st.session_state.view_mode == 'configuration':
            st.markdown("### ⚙️ System Configuration")

            # Load current config
            config_path = ROOT / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                with st.expander("Signal Settings", expanded=True):
                    risk_per_trade = st.slider("Risk Per Trade %", 0.5, 5.0, float(config.get('signals', {}).get('risk_per_trade', 2.0)), 0.5)
                    risk_reward = st.slider("Risk/Reward Ratio", 1.0, 5.0, float(config.get('signals', {}).get('risk_reward_ratio', 2.0)), 0.5)
                    signal_cooldown = st.number_input("Signal Cooldown (min)", 1, 120, int(config.get('signals', {}).get('cooldown_minutes', 60)))

                with st.expander("Auto-Retrain Settings"):
                    auto_retrain = st.checkbox("Enable Auto-Retrain", value=config.get('auto_training', {}).get('enabled', True))
                    min_win_rate = st.slider("Min Win Rate %", 30, 60, int(config.get('auto_training', {}).get('min_win_rate_threshold', 0.45) * 100))
                    max_days = st.number_input("Max Days Between Retrain", 7, 90, int(config.get('auto_training', {}).get('max_days_between_retrain', 30)))

                with st.expander("Notification Settings"):
                    desktop_notify = st.checkbox("Desktop Notifications", value=config.get('notifications', {}).get('desktop', {}).get('enabled', True))
                    sound_notify = st.checkbox("Sound Alerts", value=config.get('notifications', {}).get('desktop', {}).get('sound', True))

                if st.button("💾 Save Configuration"):
                    # Update config
                    config['signals'] = config.get('signals', {})
                    config['signals']['risk_per_trade'] = risk_per_trade
                    config['signals']['risk_reward_ratio'] = risk_reward
                    config['signals']['cooldown_minutes'] = signal_cooldown

                    config['auto_training'] = config.get('auto_training', {})
                    config['auto_training']['enabled'] = auto_retrain
                    config['auto_training']['min_win_rate_threshold'] = min_win_rate / 100
                    config['auto_training']['max_days_between_retrain'] = max_days

                    config['notifications'] = config.get('notifications', {})
                    config['notifications']['desktop'] = config['notifications'].get('desktop', {})
                    config['notifications']['desktop']['enabled'] = desktop_notify
                    config['notifications']['desktop']['sound'] = sound_notify

                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    st.success("✅ Configuration saved!")

    # ==========================================================================
    # MAIN CONTENT
    # ==========================================================================

    # Process control
    running, pid = is_analysis_running()

    # Fetch REAL data (fetch on first load OR when auto-refresh is ON)
    data = None
    error_message = None
    if st.session_state.exchange_obj:
        # Always fetch data (even if auto-refresh is OFF, need initial data)
        data = fetch_market_data(
            st.session_state.exchange_obj,
            st.session_state.selected_symbol,
            st.session_state.selected_timeframe,
            limit=200
        )
        if data['success']:
            st.session_state.update_count += 1
            st.session_state.last_fetch_time = datetime.now()
        else:
            error_message = data['error']
            logger.error(f"Data fetch failed: {error_message}")

    # Calculate values
    if data and data['success']:
        ticker = data['ticker']
        df = data['df']

        current_price = ticker['last']
        price_change_pct = ticker['percentage'] or 0
        high_24h = ticker['high']
        low_24h = ticker['low']
        volume_24h = ticker['quoteVolume']

        # Calculate RSI
        if len(df) >= 14:
            rsi = calculate_rsi(df['close'].values)
        else:
            rsi = 50.0

        rsi_sentiment = "bullish" if rsi > 60 else "bearish" if rsi < 40 else "neutral"
        rsi_label = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"

        status = "LIVE"
    else:
        # Fallback values if fetch failed or auto-refresh is OFF
        df = None  # Critical: Set df to None to prevent stale data
        current_price = 0
        price_change_pct = 0
        high_24h = 0
        low_24h = 0
        volume_24h = 0
        rsi = 50
        rsi_sentiment = "neutral"
        rsi_label = "No Data"
        status = "OFFLINE"
        df = None

    # Header - Build variables first to avoid f-string issues
    price_color = '#38ef7d' if price_change_pct >= 0 else '#f5576c'
    price_arrow = '▲' if price_change_pct >= 0 else '▼'
    status_class = 'status-live' if status == 'LIVE' else 'status-offline'
    status_text = '🟢 LIVE' if status == 'LIVE' else '🔴 OFFLINE'
    engine_text = f'✅ Running (PID: {pid})' if running else '⏸️ Stopped'
    engine_color = '#38ef7d' if running else '#f5576c'

    st.markdown(f"""
    <div class="header-container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 2rem; font-weight: 800;">📈 AI Trade Bot Pro</h1>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">{st.session_state.selected_symbol} • {st.session_state.selected_exchange.title()}</p>
            </div>
            <div style="text-align: right;">
                <div class="price-display">${current_price:,.2f}</div>
                <div style="color: {price_color}; font-size: 1.2rem; margin-top: 0.3rem;">
                    {price_arrow} {abs(price_change_pct):.2f}%
                </div>
            </div>
        </div>
        <div style="margin-top: 1rem; display: flex; gap: 1rem; align-items: center; justify-content: space-between;">
            <div style="display: flex; gap: 1rem; align-items: center;">
                <span class="status-badge {status_class}">{status_text}</span>
                <span style="opacity: 0.8;">{datetime.now().strftime('%H:%M:%S')}</span>
            </div>
            <span style="background: {engine_color}33; color: {engine_color}; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem;">
                {engine_text}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display error message if API call failed
    if error_message:
        st.error(f"⚠️ Failed to fetch market data: {error_message}")

    # Control Panel
    st.markdown("---")
    ctrl_cols = st.columns([2, 2, 2, 2, 2, 2])

    with ctrl_cols[0]:
        exchanges = ['binance', 'coinbase', 'bybit', 'kraken']
        selected_exchange = st.selectbox(
            "Exchange",
            exchanges,
            index=exchanges.index(st.session_state.selected_exchange),
            key='exchange_selector'
        )
        if selected_exchange != st.session_state.selected_exchange:
            st.session_state.selected_exchange = selected_exchange
            # Reinitialize exchange
            if CCXT_AVAILABLE:
                try:
                    if selected_exchange == 'binance':
                        st.session_state.exchange_obj = ccxt.binance({'enableRateLimit': True})
                    elif selected_exchange == 'coinbase':
                        st.session_state.exchange_obj = ccxt.coinbase({'enableRateLimit': True})
                    elif selected_exchange == 'bybit':
                        st.session_state.exchange_obj = ccxt.bybit({'enableRateLimit': True})
                    elif selected_exchange == 'kraken':
                        st.session_state.exchange_obj = ccxt.kraken({'enableRateLimit': True})
                except Exception:
                    pass
            st.rerun()

    with ctrl_cols[1]:
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
        symbol = st.selectbox("Symbol", symbols, index=symbols.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in symbols else 0)
        if symbol != st.session_state.selected_symbol:
            st.session_state.selected_symbol = symbol
            st.rerun()

    with ctrl_cols[2]:
        timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        timeframe = st.selectbox("Timeframe", timeframes, index=timeframes.index(st.session_state.selected_timeframe) if st.session_state.selected_timeframe in timeframes else 0)
        if timeframe != st.session_state.selected_timeframe:
            st.session_state.selected_timeframe = timeframe
            st.rerun()

    with ctrl_cols[3]:
        st.write("")
        st.write("")
        auto = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
        if auto != st.session_state.auto_refresh:
            st.session_state.auto_refresh = auto

    with ctrl_cols[4]:
        st.write("")
        st.write("")
        if running:
            if st.button("⏹️ Stop Engine", type="secondary", use_container_width=True):
                try:
                    os.kill(pid, signal.SIGTERM)
                    PID_FILE.unlink(missing_ok=True)
                except Exception:
                    pass
                time.sleep(1)
                st.rerun()
        else:
            if st.button("▶️ Start Engine", type="primary", use_container_width=True):
                venv_py = ROOT / "venv" / "bin" / "python"
                py = str(venv_py) if venv_py.exists() else "python3"
                subprocess.Popen(
                    [py, str(ROOT / "run_analysis.py")],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=ROOT
                )
                time.sleep(2)
                st.rerun()

    with ctrl_cols[5]:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Now", type="secondary", use_container_width=True):
            st.rerun()

    # Metrics Row
    st.markdown("---")
    met_cols = st.columns(6)

    # Format volume
    if volume_24h >= 1e9:
        vol_str = f"${volume_24h/1e9:.1f}B"
    elif volume_24h >= 1e6:
        vol_str = f"${volume_24h/1e6:.1f}M"
    else:
        vol_str = f"${volume_24h:,.0f}"

    metrics_data = [
        ("Updates", f"{st.session_state.update_count}", "Auto-refresh" if st.session_state.auto_refresh else "Manual", "bullish" if st.session_state.auto_refresh else "neutral"),
        ("Candles", f"{len(df) if df is not None else 0}", st.session_state.selected_timeframe, "bullish" if df is not None else "neutral"),
        ("RSI (14)", f"{rsi:.1f}", rsi_label, rsi_sentiment),
        ("24H High", f"${high_24h:,.2f}", "Resistance", "bullish"),
        ("24H Low", f"${low_24h:,.2f}", "Support", "bearish"),
        ("24H Volume", vol_str, "Trading", "neutral"),
    ]

    for col, (label, value, delta, sentiment) in zip(met_cols, metrics_data):
        with col:
            st.markdown(f"""
            <div class="metric-card {sentiment}">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size: {'1.8rem' if len(str(value)) > 10 else '2rem'};">{value}</div>
                <div style="color: {'#38ef7d' if sentiment == 'bullish' else '#f5576c' if sentiment == 'bearish' else '#6c757d'}; font-size: 0.9rem; font-weight: 600;">
                    {delta}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # AI Predictions Section
    if AI_AVAILABLE and st.session_state.predictor and df is not None and len(df) >= 100:
        st.markdown("---")
        st.markdown("### 🤖 AI Predictions & Mathematical Analysis")

        try:
            # Get AI prediction
            prediction = st.session_state.predictor.predict(df)

            # Get mathematical indicators from MathEngine (pass full DataFrame)
            math_analysis = st.session_state.math_engine.analyze(df)

            # Display predictions in columns
            pred_cols = st.columns([1, 1, 1, 1])

            with pred_cols[0]:
                direction_color = "#38ef7d" if prediction.direction == "BUY" else "#f5576c" if prediction.direction == "SELL" else "#ffc107"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">AI Signal</div>
                    <div class="metric-value" style="color: {direction_color}; font-size: 2.5rem;">
                        {prediction.direction}
                    </div>
                    <div style="color: #6c757d; font-size: 0.9rem;">
                        Confidence: {prediction.confidence * 100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with pred_cols[1]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Stop Loss</div>
                    <div class="metric-value" style="color: #f5576c; font-size: 1.8rem;">
                        ${prediction.stop_loss:,.2f}
                    </div>
                    <div style="color: #6c757d; font-size: 0.9rem;">
                        Risk: {prediction.monte_carlo_risk * 100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with pred_cols[2]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Take Profit</div>
                    <div class="metric-value" style="color: #38ef7d; font-size: 1.8rem;">
                        ${prediction.take_profit:,.2f}
                    </div>
                    <div style="color: #6c757d; font-size: 0.9rem;">
                        R:R {prediction.risk_reward_ratio:.2f}:1
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with pred_cols[3]:
                hurst_color = "#38ef7d" if math_analysis.hurst_exponent > 0.5 else "#f5576c"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Market Regime</div>
                    <div class="metric-value" style="color: {hurst_color}; font-size: 1.8rem;">
                        {math_analysis.hurst_regime}
                    </div>
                    <div style="color: #6c757d; font-size: 0.9rem;">
                        Hurst: {math_analysis.hurst_exponent:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Mathematical details (basic view)
            with st.expander("🔬 Mathematical Breakdown", expanded=False):
                math_cols = st.columns(3)

                with math_cols[0]:
                    st.markdown("**Fourier Analysis**")
                    st.write(f"Cycle Signal: {prediction.fourier_signal:.3f}")

                    st.markdown("**Kalman Filter**")
                    st.write(f"Trend Direction: {prediction.kalman_trend:.3f}")

                    st.markdown("**Wavelet Analysis**")
                    st.write(f"Signal: {math_analysis.wavelet_signal:.3f}")

                with math_cols[1]:
                    st.markdown("**Hurst Exponent**")
                    st.write(f"Value: {math_analysis.hurst_exponent:.3f}")
                    st.write(f"Regime: {math_analysis.hurst_regime}")

                    st.markdown("**Markov Chain**")
                    st.write(f"Transition Prob: {prediction.markov_probability * 100:.1f}%")

                with math_cols[2]:
                    st.markdown("**Risk Metrics**")
                    st.write(f"Monte Carlo Risk: {prediction.monte_carlo_risk * 100:.1f}%")
                    st.write(f"Jump Probability: {math_analysis.jump_probability * 100:.1f}%")
                    st.write(f"Crash Indicator: {math_analysis.crash_indicator * 100:.1f}%")

            # ==========================================================================
            # FULL ALGORITHM DETAILS (when checkbox enabled)
            # ==========================================================================
            if st.session_state.show_algorithm_details:
                st.markdown("---")
                st.markdown("### 🔬 Complete Algorithm Analysis")

                # Algorithm weights visualization
                st.markdown("#### ⚖️ Current Algorithm Weights")
                weights = st.session_state.algorithm_weights
                weights_df = pd.DataFrame([
                    {"Algorithm": k.title(), "Weight": v, "Percentage": f"{v*100:.1f}%"}
                    for k, v in weights.items()
                ])
                st.dataframe(weights_df, use_container_width=True, hide_index=True)

                # Detailed algorithm panels
                alg_tabs = st.tabs([
                    "🌊 Fourier", "📡 Kalman", "🎲 Entropy", "🔗 Markov", "🧠 LSTM",
                    "📈 Wavelet", "📊 Hurst", "🔄 Ornstein-Uhlenbeck", "⚡ Jump Detection", "🔢 Eigenvalue"
                ])

                with alg_tabs[0]:  # Fourier
                    st.markdown("#### Fourier Analysis (Cycle Detection)")
                    st.markdown("""
                    **Purpose:** Detects hidden cycles and periodicities in price data.

                    **How it works:**
                    - Decomposes price into frequency components
                    - Identifies dominant cycles (daily, weekly patterns)
                    - Predicts next cycle phase
                    """)

                    # Main signal metric
                    st.metric("Fourier Signal", f"{prediction.fourier_signal:.4f}")
                    signal_strength = "Strong" if abs(prediction.fourier_signal) > 0.5 else "Moderate" if abs(prediction.fourier_signal) > 0.2 else "Weak"
                    st.info(f"Signal Strength: {signal_strength}")

                    # Detailed Fourier analysis (if available)
                    fourier_details = getattr(prediction, 'fourier_details', None)
                    if fourier_details and not fourier_details.get('error'):
                        st.markdown("---")
                        st.markdown("##### 📊 Detailed Cycle Analysis")

                        # Cycle metrics in columns
                        fc1, fc2, fc3 = st.columns(3)
                        with fc1:
                            period = fourier_details.get('dominant_period', 0)
                            if period > 0:
                                st.metric("Dominant Cycle", f"{period:.1f} candles")
                                # Interpret the period
                                if period < 12:
                                    cycle_type = "Short-term (intraday)"
                                elif period < 48:
                                    cycle_type = "Medium-term (daily)"
                                elif period < 168:
                                    cycle_type = "Weekly pattern"
                                else:
                                    cycle_type = "Long-term cycle"
                                st.caption(cycle_type)
                            else:
                                st.metric("Dominant Cycle", "N/A")

                        with fc2:
                            phase = fourier_details.get('phase', 0)
                            phase_deg = np.degrees(phase) if phase else 0
                            st.metric("Current Phase", f"{phase_deg:.1f}°")
                            # Phase interpretation
                            if -45 <= phase_deg <= 45:
                                phase_meaning = "Peak zone (reversal likely)"
                            elif 45 < phase_deg <= 135:
                                phase_meaning = "Uptrend phase"
                            elif -135 <= phase_deg < -45:
                                phase_meaning = "Downtrend phase"
                            else:
                                phase_meaning = "Trough zone (reversal likely)"
                            st.caption(phase_meaning)

                        with fc3:
                            interpretation = fourier_details.get('interpretation', 'NEUTRAL')
                            if interpretation == 'BULLISH':
                                st.success(f"📈 {interpretation}")
                            elif interpretation == 'BEARISH':
                                st.error(f"📉 {interpretation}")
                            else:
                                st.info(f"➡️ {interpretation}")
                            st.caption("Cycle direction")

                        # Frequency spectrum visualization
                        freqs = fourier_details.get('dominant_frequencies', [])
                        mags = fourier_details.get('magnitudes', [])
                        if freqs and mags:
                            st.markdown("##### 📈 Frequency Spectrum (Top 3)")
                            # Create bar chart for frequency magnitudes
                            freq_df = pd.DataFrame({
                                'Frequency': [f'{f:.4f}' for f in freqs],
                                'Magnitude': mags,
                                'Period': [f'{1/f:.1f} candles' if f > 0.0001 else 'N/A' for f in freqs]
                            })
                            st.dataframe(freq_df, use_container_width=True)

                            # Visual bar chart
                            fig_freq = go.Figure(data=[
                                go.Bar(
                                    x=[f'Cycle {i+1}\n({1/f:.0f} candles)' if f > 0.0001 else f'Cycle {i+1}' for i, f in enumerate(freqs)],
                                    y=mags,
                                    marker_color=['#667eea', '#764ba2', '#f093fb']
                                )
                            ])
                            fig_freq.update_layout(
                                title="Dominant Cycle Strengths",
                                xaxis_title="Cycle",
                                yaxis_title="Magnitude",
                                height=250,
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            st.plotly_chart(fig_freq, use_container_width=True)

                        # Cycle prediction
                        st.markdown("##### 🔮 Cycle Prediction")
                        period = fourier_details.get('dominant_period', 0)
                        if period > 0:
                            # Calculate where we are in the cycle
                            phase_norm = (phase_deg + 180) / 360  # Normalize to 0-1
                            candles_into_cycle = int(period * phase_norm)
                            candles_remaining = int(period - candles_into_cycle)

                            pred_c1, pred_c2 = st.columns(2)
                            with pred_c1:
                                st.metric("Position in Cycle", f"{candles_into_cycle}/{int(period)} candles")
                            with pred_c2:
                                st.metric("Until Next Reversal", f"~{candles_remaining} candles")

                            # Progress bar for cycle
                            st.progress(min(1.0, phase_norm), text=f"Cycle Progress: {phase_norm*100:.0f}%")
                    else:
                        st.warning("⚠️ Insufficient data for detailed Fourier analysis (need 64+ candles)")

                with alg_tabs[1]:  # Kalman
                    st.markdown("#### Kalman Filter (Trend Estimation)")
                    st.markdown("""
                    **Purpose:** Optimal estimation of true price trend with noise filtering.

                    **How it works:**
                    - Predicts price using state-space model
                    - Updates prediction based on new observations
                    - Adapts to volatility changes
                    """)
                    st.metric("Kalman Trend", f"{prediction.kalman_trend:.4f}")
                    trend_dir = "Bullish" if prediction.kalman_trend > 0 else "Bearish"
                    st.info(f"Trend Direction: {trend_dir}")

                with alg_tabs[2]:  # Entropy
                    st.markdown("#### Shannon Entropy (Regime Detection)")
                    st.markdown("""
                    **Purpose:** Measures market disorder and regime changes.

                    **How it works:**
                    - Calculates information entropy of returns
                    - High entropy = uncertain market
                    - Low entropy = predictable patterns
                    """)
                    entropy_val = getattr(prediction, 'entropy_signal', 0.5)
                    st.metric("Market Entropy", f"{entropy_val:.4f}")
                    regime = "High Uncertainty" if entropy_val > 0.7 else "Moderate" if entropy_val > 0.4 else "Low Uncertainty"
                    st.info(f"Market Regime: {regime}")

                with alg_tabs[3]:  # Markov
                    st.markdown("#### Markov Chain (State Transitions)")
                    st.markdown("""
                    **Purpose:** Models market as state machine with transition probabilities.

                    **States:** Strong Bear, Bear, Neutral, Bull, Strong Bull

                    **How it works:**
                    - Learns historical state transitions
                    - Predicts next likely state
                    - Provides probability of direction change
                    """)

                    # Main probability metric
                    st.metric("Bullish Probability", f"{prediction.markov_probability * 100:.1f}%")

                    # Detailed Markov analysis (if available)
                    markov_details = getattr(prediction, 'markov_details', None)
                    if markov_details:
                        st.markdown("---")
                        st.markdown("##### 📊 Transition Matrix Visualization")

                        # Current state indicator
                        state_names = markov_details.get('state_names', ['Strong Bear', 'Bear', 'Neutral', 'Bull', 'Strong Bull'])
                        current_state = markov_details.get('current_state', 2)
                        current_state_name = state_names[current_state] if 0 <= current_state < len(state_names) else 'Unknown'

                        mk_c1, mk_c2, mk_c3 = st.columns(3)
                        with mk_c1:
                            st.metric("Current State", current_state_name)
                        with mk_c2:
                            bullish_prob = markov_details.get('bullish_probability', 0.5)
                            st.metric("Bullish Probability", f"{bullish_prob*100:.1f}%")
                        with mk_c3:
                            bearish_prob = markov_details.get('bearish_probability', 0.5)
                            st.metric("Bearish Probability", f"{bearish_prob*100:.1f}%")

                        # State color mapping
                        state_colors = {
                            'Strong Bear': '#ff4136',
                            'Bear': '#ff851b',
                            'Neutral': '#aaaaaa',
                            'Bull': '#2ecc40',
                            'Strong Bull': '#01ff70'
                        }

                        # Visual state indicator
                        st.markdown("##### 🎯 Market State")
                        state_html = f"""
                        <div style="display: flex; justify-content: center; gap: 10px; margin: 20px 0;">
                        """
                        for i, state in enumerate(state_names):
                            is_current = i == current_state
                            color = state_colors.get(state, '#aaa')
                            border = "3px solid #000" if is_current else "1px solid #ccc"
                            opacity = "1" if is_current else "0.4"
                            state_html += f"""
                            <div style="padding: 15px 20px; background: {color}; border-radius: 10px;
                                        border: {border}; opacity: {opacity}; text-align: center;
                                        font-weight: {'bold' if is_current else 'normal'}; color: white;
                                        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                                {state}
                                {"← CURRENT" if is_current else ""}
                            </div>
                            """
                        state_html += "</div>"
                        st.markdown(state_html, unsafe_allow_html=True)

                        # Transition matrix heatmap
                        transition_matrix = markov_details.get('transition_matrix', None)
                        if transition_matrix and len(transition_matrix) > 0:
                            st.markdown("##### 📈 Transition Probability Matrix")
                            st.caption("Shows probability of moving FROM row state TO column state")

                            # Convert to numpy array
                            tm = np.array(transition_matrix)

                            # Create heatmap
                            fig_tm = go.Figure(data=go.Heatmap(
                                z=tm,
                                x=state_names,
                                y=state_names,
                                colorscale='RdYlGn',
                                text=[[f'{v*100:.1f}%' for v in row] for row in tm],
                                texttemplate='%{text}',
                                textfont={"size": 12},
                                hovertemplate='From %{y} → To %{x}: %{z:.1%}<extra></extra>'
                            ))

                            fig_tm.update_layout(
                                title="State Transition Probabilities",
                                xaxis_title="To State",
                                yaxis_title="From State",
                                height=350,
                                margin=dict(l=0, r=0, t=40, b=0)
                            )

                            st.plotly_chart(fig_tm, use_container_width=True)

                            # Next state prediction
                            st.markdown("##### 🔮 Next State Prediction")
                            if 0 <= current_state < len(tm):
                                current_row = tm[current_state]
                                # Get probabilities for each next state
                                pred_data = []
                                for i, prob in enumerate(current_row):
                                    pred_data.append({
                                        'Next State': state_names[i],
                                        'Probability': f'{prob*100:.1f}%',
                                        'Likelihood': '🟢 High' if prob > 0.3 else '🟡 Medium' if prob > 0.15 else '🔴 Low'
                                    })
                                st.dataframe(pd.DataFrame(pred_data), use_container_width=True)

                                # Most likely next state
                                most_likely_idx = np.argmax(current_row)
                                st.success(f"📌 Most likely next state: **{state_names[most_likely_idx]}** ({current_row[most_likely_idx]*100:.1f}%)")
                    else:
                        st.info("Detailed Markov analysis not available")

                with alg_tabs[4]:  # LSTM
                    st.markdown("#### LSTM Neural Network (Deep Learning)")
                    st.markdown("""
                    **Purpose:** Learns complex non-linear patterns in price history.

                    **Architecture:**
                    - Long Short-Term Memory cells
                    - Remembers important patterns
                    - Forgets irrelevant noise
                    """)
                    model_path = ROOT / "data" / "lstm_model.pt"
                    if model_path.exists():
                        st.success("✅ Model trained and active")
                    else:
                        st.warning("⚠️ Model not trained")

                with alg_tabs[5]:  # Wavelet
                    st.markdown("#### Wavelet Analysis (Multi-Scale Patterns)")
                    st.markdown("""
                    **Purpose:** Analyzes patterns at multiple time scales simultaneously.

                    **Better than Fourier because:**
                    - Handles non-stationary signals
                    - Provides time-localized frequency info
                    - Detects transient events
                    """)
                    st.metric("Wavelet Signal", f"{math_analysis.wavelet_signal:.4f}")

                with alg_tabs[6]:  # Hurst
                    st.markdown("#### Hurst Exponent (Memory Analysis)")
                    st.markdown("""
                    **Purpose:** Determines if market is trending or mean-reverting.

                    **Interpretation:**
                    - H > 0.5: **Trending** (follow the trend)
                    - H < 0.5: **Mean-Reverting** (fade the move)
                    - H = 0.5: **Random Walk** (unpredictable)
                    """)
                    h_col1, h_col2 = st.columns(2)
                    with h_col1:
                        st.metric("Hurst Exponent", f"{math_analysis.hurst_exponent:.4f}")
                    with h_col2:
                        regime_color = "#38ef7d" if math_analysis.hurst_exponent > 0.5 else "#f5576c"
                        st.markdown(f"<div style='padding:1rem;background:{regime_color}20;border-radius:10px;text-align:center'><b>{math_analysis.hurst_regime}</b></div>", unsafe_allow_html=True)

                with alg_tabs[7]:  # OU
                    st.markdown("#### Ornstein-Uhlenbeck Process (Mean Reversion)")
                    st.markdown("""
                    **Purpose:** Mathematical model for mean-reverting assets.

                    **Provides:**
                    - Equilibrium price target
                    - Half-life (time to mean reversion)
                    - Entry/exit signals for range trading
                    """)
                    ou_col1, ou_col2 = st.columns(2)
                    with ou_col1:
                        st.metric("Equilibrium Price", f"${math_analysis.ou_equilibrium:,.2f}")
                    with ou_col2:
                        st.metric("Half-Life", f"{math_analysis.ou_half_life:.1f} candles")

                with alg_tabs[8]:  # Jump
                    st.markdown("#### Jump Detection (Crash/Rally Prediction)")
                    st.markdown("""
                    **Purpose:** Detects abnormal price movements and crash probability.

                    **Uses:**
                    - Poisson jump process modeling
                    - Tail risk estimation
                    - Early warning signals
                    """)
                    j_col1, j_col2 = st.columns(2)
                    with j_col1:
                        jump_color = "#f5576c" if math_analysis.jump_probability > 0.3 else "#38ef7d"
                        st.markdown(f"<div style='text-align:center'><span style='font-size:0.8rem;color:#6c757d'>Jump Probability</span><br><span style='font-size:1.5rem;font-weight:700;color:{jump_color}'>{math_analysis.jump_probability * 100:.1f}%</span></div>", unsafe_allow_html=True)
                    with j_col2:
                        crash_color = "#f5576c" if math_analysis.crash_indicator > 0.5 else "#38ef7d"
                        st.markdown(f"<div style='text-align:center'><span style='font-size:0.8rem;color:#6c757d'>Crash Risk</span><br><span style='font-size:1.5rem;font-weight:700;color:{crash_color}'>{math_analysis.crash_indicator * 100:.1f}%</span></div>", unsafe_allow_html=True)

                with alg_tabs[9]:  # Eigenvalue
                    st.markdown("#### Eigenvalue Analysis (Signal vs Noise)")
                    st.markdown("""
                    **Purpose:** Separates true market signal from random noise.

                    **Based on Random Matrix Theory:**
                    - Analyzes correlation matrix eigenvalues
                    - Detects regime changes
                    - Identifies market inefficiencies
                    """)
                    st.metric("Signal/Noise Ratio", f"{math_analysis.eigenvalue_ratio:.4f}")
                    quality = "High Quality Signal" if math_analysis.eigenvalue_ratio > 1.5 else "Moderate" if math_analysis.eigenvalue_ratio > 1.0 else "Low Quality (Noisy)"
                    st.info(f"Signal Quality: {quality}")

        except Exception as e:
            st.warning(f"⚠️ AI prediction unavailable: {str(e)}")
            logger.error(f"Prediction error: {e}")

    elif df is not None and len(df) < 100:
        st.info("ℹ️ Need at least 100 candles for AI predictions. Current: " + str(len(df)))

    # ==============================================================================
    # ALL 27 TECHNICAL INDICATORS DISPLAY (when checkbox enabled)
    # ==============================================================================
    if st.session_state.show_all_indicators and df is not None and len(df) >= 50:
        st.markdown("---")
        st.markdown("### 📊 All 27 Technical Indicators")

        try:
            # Calculate all features
            features_df = FeatureCalculator.calculate_all(df)
            latest = features_df.iloc[-1]

            # Group indicators by category
            indicator_groups = {
                "📈 Price-Based": {
                    "Returns": (latest.get('returns', 0) * 100, "%"),
                    "Log Returns": (latest.get('log_returns', 0) * 100, "%"),
                    "Price/SMA7": (latest.get('price_sma_7_ratio', 1), "x"),
                    "Price/SMA21": (latest.get('price_sma_21_ratio', 1), "x"),
                    "Price/SMA50": (latest.get('price_sma_50_ratio', 1), "x"),
                },
                "📊 Volatility": {
                    "ATR (14)": (latest.get('atr_14', 0), "$"),
                    "BB Width": (latest.get('bb_width', 0) * 100, "%"),
                    "BB Position": (latest.get('bb_position', 0.5) * 100, "%"),
                    "Volatility 14d": (latest.get('volatility_14', 0) * 100, "%"),
                    "Volatility 30d": (latest.get('volatility_30', 0) * 100, "%"),
                },
                "🎯 Momentum": {
                    "RSI (14)": (latest.get('rsi_14', 50), ""),
                    "Stochastic K": (latest.get('stoch_k', 50), ""),
                    "Stochastic D": (latest.get('stoch_d', 50), ""),
                    "MACD": (latest.get('macd', 0), ""),
                    "MACD Signal": (latest.get('macd_signal', 0), ""),
                    "MACD Histogram": (latest.get('macd_hist', 0), ""),
                    "ROC (10)": (latest.get('roc_10', 0), "%"),
                    "ROC (20)": (latest.get('roc_20', 0), "%"),
                    "Williams %R": (latest.get('williams_r', -50), ""),
                },
                "📦 Volume": {
                    "Volume Ratio": (latest.get('volume_ratio', 1), "x"),
                },
                "📐 Trend": {
                    "ADX": (latest.get('adx', 25), ""),
                    "+DI": (latest.get('plus_di', 25), ""),
                    "-DI": (latest.get('minus_di', 25), ""),
                    "Trend 7d": (latest.get('trend_7', 0) * 100, "%"),
                    "Trend 14d": (latest.get('trend_14', 0) * 100, "%"),
                },
                "🕯️ Candle Patterns": {
                    "Body Ratio": (latest.get('candle_body_ratio', 0.5) * 100, "%"),
                    "Higher High": ("Yes" if latest.get('higher_high', 0) == 1 else "No", ""),
                    "Lower Low": ("Yes" if latest.get('lower_low', 0) == 1 else "No", ""),
                },
            }

            # Display in expanders by group
            for group_name, indicators in indicator_groups.items():
                with st.expander(group_name, expanded=True):
                    cols = st.columns(5)
                    for idx, (name, (value, suffix)) in enumerate(indicators.items()):
                        with cols[idx % 5]:
                            if isinstance(value, str):
                                st.metric(name, value)
                            else:
                                formatted = f"{value:.2f}{suffix}" if not pd.isna(value) else "N/A"
                                # Color coding for certain indicators
                                if name == "RSI (14)":
                                    color = "#38ef7d" if value > 60 else "#f5576c" if value < 40 else "#6c757d"
                                elif "Trend" in name:
                                    color = "#38ef7d" if value > 0 else "#f5576c"
                                else:
                                    color = "#2d3748"
                                st.markdown(f"<div style='text-align:center'><span style='font-size:0.8rem;color:#6c757d'>{name}</span><br><span style='font-size:1.2rem;font-weight:700;color:{color}'>{formatted}</span></div>", unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"⚠️ Could not calculate indicators: {str(e)}")
            logger.error(f"Indicators error: {e}")

    # ==============================================================================
    # LSTM Deep Learning Predictions
    # ==============================================================================
    if AI_AVAILABLE and df is not None and len(df) >= 100:
        st.markdown("---")
        st.markdown("### 🧠 LSTM Deep Learning Predictions")

        try:
            # Calculate features
            features_df = FeatureCalculator.calculate_all(df)

            # Check if we have trained model
            model_path = ROOT / "data" / "lstm_model.pt"
            if model_path.exists():
                st.success("✅ Using trained LSTM model")

                # Show feature importance (top 10)
                with st.expander("📊 Feature Analysis", expanded=False):
                    st.write("**Top Technical Indicators:**")
                    feature_cols = [col for col in features_df.columns if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]
                    if len(feature_cols) > 0:
                        recent_features = features_df[feature_cols].tail(1)
                        feat_cols = st.columns(3)

                        for idx, feat in enumerate(feature_cols[:12]):  # Show first 12 features
                            col_idx = idx % 3
                            with feat_cols[col_idx]:
                                value = recent_features[feat].values[0]
                                st.metric(feat, f"{value:.4f}")
            else:
                st.info("ℹ️ LSTM model not trained yet. Run `python train_model.py` to train the deep learning model.")

        except Exception as e:
            st.warning(f"⚠️ LSTM prediction unavailable: {str(e)}")
            logger.error(f"LSTM error: {e}")

    # ==============================================================================
    # MULTI-CURRENCY PERFORMANCE TRACKING
    # ==============================================================================
    if len(st.session_state.tracked_currencies) > 1 and AI_AVAILABLE:
        st.markdown("---")
        st.markdown("### 💱 Multi-Currency Performance")

        try:
            # Initialize MultiCurrencySystem if available
            if MultiCurrencySystem is not None:
                mcs_cols = st.columns(len(st.session_state.tracked_currencies))

                for idx, currency in enumerate(st.session_state.tracked_currencies):
                    with mcs_cols[idx]:
                        # Fetch quick ticker for each currency
                        if st.session_state.exchange_obj:
                            try:
                                ticker = st.session_state.exchange_obj.fetch_ticker(currency)
                                price = ticker['last']
                                change = ticker['percentage'] or 0
                                change_color = "#38ef7d" if change >= 0 else "#f5576c"

                                st.markdown(f"""
                                <div class="metric-card" style="border-left-color: {change_color}">
                                    <div class="metric-label">{currency}</div>
                                    <div class="metric-value" style="font-size: 1.5rem;">${price:,.2f}</div>
                                    <div style="color: {change_color}; font-size: 0.9rem; font-weight: 600;">
                                        {'▲' if change >= 0 else '▼'} {abs(change):.2f}%
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">{currency}</div>
                                    <div class="metric-value" style="font-size: 1.2rem;">Loading...</div>
                                </div>
                                """, unsafe_allow_html=True)

                # Performance comparison chart
                with st.expander("📊 Price Performance Comparison", expanded=False):
                    perf_data = []
                    for currency in st.session_state.tracked_currencies:
                        try:
                            ohlcv = st.session_state.exchange_obj.fetch_ohlcv(currency, '1h', limit=24)
                            if ohlcv:
                                start_price = ohlcv[0][4]  # First close
                                end_price = ohlcv[-1][4]   # Last close
                                pct_change = ((end_price - start_price) / start_price) * 100
                                perf_data.append({
                                    "Currency": currency,
                                    "24h Change": f"{pct_change:+.2f}%",
                                    "Start": f"${start_price:,.2f}",
                                    "Current": f"${end_price:,.2f}"
                                })
                        except Exception:
                            pass

                    if perf_data:
                        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

                # AI Trading Performance per Currency
                with st.expander("🤖 AI Trading Performance per Currency", expanded=True):
                    st.markdown("##### Per-Currency Trading Statistics")
                    st.caption("Track win rate, PnL, and model status for each currency")

                    # Try to get performance from MultiCurrencySystem
                    try:
                        config_path = ROOT / "config.yaml"
                        if config_path.exists():
                            with open(config_path) as f:
                                config = yaml.safe_load(f)
                            mcs = MultiCurrencySystem(config)
                            perf_report = mcs.get_performance_report()

                            if perf_report:
                                # Create detailed performance table
                                perf_table = []
                                for symbol, stats in perf_report.items():
                                    win_rate_str = stats.get('win_rate', '0.0%')
                                    win_rate_val = float(win_rate_str.replace('%', '')) / 100 if isinstance(win_rate_str, str) else stats.get('win_rate', 0)

                                    # Determine performance status
                                    if win_rate_val >= 0.55:
                                        status = "🟢 Excellent"
                                    elif win_rate_val >= 0.50:
                                        status = "🟡 Good"
                                    elif win_rate_val >= 0.45:
                                        status = "🟠 Fair"
                                    else:
                                        status = "🔴 Needs Retrain"

                                    perf_table.append({
                                        'Currency': symbol,
                                        'Total Signals': stats.get('total_signals', 0),
                                        'Win Rate': win_rate_str,
                                        'Total PnL': stats.get('total_pnl', '0.00%'),
                                        'Status': status,
                                        'Needs Retrain': '⚠️ Yes' if stats.get('needs_retrain', False) else '✅ No',
                                        'Last Retrain': stats.get('last_retrain', 'Never'),
                                        'Retraining': '🔄' if stats.get('retrain_in_progress', False) else ''
                                    })

                                if perf_table:
                                    st.dataframe(pd.DataFrame(perf_table), use_container_width=True, hide_index=True)

                                    # Summary metrics
                                    st.markdown("---")
                                    sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)

                                    total_signals = sum(s.get('total_signals', 0) for s in perf_report.values())
                                    currencies_needing_retrain = sum(1 for s in perf_report.values() if s.get('needs_retrain', False))

                                    with sum_c1:
                                        st.metric("Total Currencies", len(perf_report))
                                    with sum_c2:
                                        st.metric("Total Signals", total_signals)
                                    with sum_c3:
                                        st.metric("Need Retrain", currencies_needing_retrain)
                                    with sum_c4:
                                        active_retrains = sum(1 for s in perf_report.values() if s.get('retrain_in_progress', False))
                                        st.metric("Active Retrains", active_retrains)

                                    # Performance chart
                                    if len(perf_table) > 1:
                                        st.markdown("##### 📈 Win Rate Comparison")
                                        win_rates = []
                                        for item in perf_table:
                                            wr_str = item['Win Rate']
                                            if isinstance(wr_str, str):
                                                wr = float(wr_str.replace('%', ''))
                                            else:
                                                wr = wr_str * 100
                                            win_rates.append(wr)

                                        fig_wr = go.Figure(data=[
                                            go.Bar(
                                                x=[p['Currency'] for p in perf_table],
                                                y=win_rates,
                                                marker_color=[
                                                    '#2ecc40' if wr >= 55 else '#ffdc00' if wr >= 50 else '#ff851b' if wr >= 45 else '#ff4136'
                                                    for wr in win_rates
                                                ],
                                                text=[f'{wr:.1f}%' for wr in win_rates],
                                                textposition='outside'
                                            )
                                        ])
                                        fig_wr.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="50% baseline")
                                        fig_wr.update_layout(
                                            title="Win Rate by Currency",
                                            xaxis_title="Currency",
                                            yaxis_title="Win Rate (%)",
                                            yaxis_range=[0, 100],
                                            height=300,
                                            margin=dict(l=0, r=0, t=40, b=0)
                                        )
                                        st.plotly_chart(fig_wr, use_container_width=True)
                            else:
                                st.info("No performance data available yet. Start trading to collect statistics.")
                            mcs.cleanup()
                    except Exception as e:
                        # Fallback: Show placeholder with tracked currencies
                        st.info("AI trading system not initialized. Performance tracking will begin when trading starts.")
                        fallback_data = [{
                            'Currency': currency,
                            'Total Signals': 0,
                            'Win Rate': '0.0%',
                            'Total PnL': '0.00%',
                            'Status': '⚪ No Data',
                            'Needs Retrain': 'N/A',
                            'Last Retrain': 'Never',
                            'Retraining': ''
                        } for currency in st.session_state.tracked_currencies]
                        st.dataframe(pd.DataFrame(fallback_data), use_container_width=True, hide_index=True)

        except Exception as e:
            st.warning(f"⚠️ Multi-currency tracking unavailable: {str(e)}")
            logger.error(f"Multi-currency error: {e}")

    # ==============================================================================
    # ANALYSIS VIEW MODE - Additional Panels
    # ==============================================================================
    if st.session_state.view_mode == 'analysis' and df is not None and len(df) >= 20:
        st.markdown("---")
        st.markdown("### 📈 Advanced Analysis View")

        analysis_tabs = st.tabs(["📊 Statistics", "📉 Correlations", "🎯 Support/Resistance", "📋 Data Quality"])

        with analysis_tabs[0]:  # Statistics
            st.markdown("#### Price Statistics")
            stats_col1, stats_col2, stats_col3 = st.columns(3)

            with stats_col1:
                st.metric("Mean Price", f"${df['close'].mean():,.2f}")
                st.metric("Median Price", f"${df['close'].median():,.2f}")
                st.metric("Std Dev", f"${df['close'].std():,.2f}")

            with stats_col2:
                st.metric("Min Price", f"${df['close'].min():,.2f}")
                st.metric("Max Price", f"${df['close'].max():,.2f}")
                st.metric("Range", f"${df['close'].max() - df['close'].min():,.2f}")

            with stats_col3:
                returns = df['close'].pct_change().dropna()
                st.metric("Avg Return", f"{returns.mean() * 100:.4f}%")
                st.metric("Return Std", f"{returns.std() * 100:.4f}%")
                skewness = returns.skew() if len(returns) > 2 else 0
                st.metric("Skewness", f"{skewness:.4f}")

        with analysis_tabs[1]:  # Correlations
            st.markdown("#### Price-Volume Correlation")
            correlation = df['close'].corr(df['volume'])
            st.metric("Close-Volume Correlation", f"{correlation:.4f}")

            # Rolling correlation
            if len(df) >= 20:
                rolling_corr = df['close'].rolling(20).corr(df['volume'])
                st.line_chart(rolling_corr.dropna(), use_container_width=True)

        with analysis_tabs[2]:  # Support/Resistance
            st.markdown("#### Key Levels")
            # Simple support/resistance based on recent highs/lows
            recent_20 = df.tail(20)
            recent_50 = df.tail(50) if len(df) >= 50 else df

            sr_col1, sr_col2 = st.columns(2)
            with sr_col1:
                st.markdown("**Resistance Levels:**")
                st.write(f"R1 (20-bar high): ${recent_20['high'].max():,.2f}")
                st.write(f"R2 (50-bar high): ${recent_50['high'].max():,.2f}")

            with sr_col2:
                st.markdown("**Support Levels:**")
                st.write(f"S1 (20-bar low): ${recent_20['low'].min():,.2f}")
                st.write(f"S2 (50-bar low): ${recent_50['low'].min():,.2f}")

            # Current position relative to levels
            current = df['close'].iloc[-1]
            r1 = recent_20['high'].max()
            s1 = recent_20['low'].min()
            position_pct = ((current - s1) / (r1 - s1)) * 100 if r1 != s1 else 50
            st.progress(min(max(position_pct / 100, 0), 1))
            st.caption(f"Price Position: {position_pct:.1f}% between S1 and R1")

        with analysis_tabs[3]:  # Data Quality
            st.markdown("#### Data Quality Metrics")
            dq_col1, dq_col2 = st.columns(2)

            with dq_col1:
                st.metric("Total Candles", len(df))
                null_count = df.isnull().sum().sum()
                st.metric("Missing Values", null_count)
                if 'datetime' in df.columns:
                    time_range = df['datetime'].max() - df['datetime'].min()
                    st.metric("Time Range", str(time_range))

            with dq_col2:
                # Check for gaps
                if len(df) >= 2 and 'datetime' in df.columns:
                    time_diffs = df['datetime'].diff().dropna()
                    expected_diff = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                    gaps = (time_diffs > expected_diff * 2).sum()
                    st.metric("Data Gaps", gaps)

                # Volume anomalies
                vol_mean = df['volume'].mean()
                vol_std = df['volume'].std()
                anomalies = ((df['volume'] > vol_mean + 3*vol_std) | (df['volume'] < vol_mean - 3*vol_std)).sum()
                st.metric("Volume Anomalies", anomalies)

    # ==============================================================================
    # SIGNAL HISTORY & PERFORMANCE TRACKING
    # ==============================================================================
    if AI_AVAILABLE and df is not None:
        st.markdown("---")
        st.markdown("### 📡 Signal History & Performance")

        try:
            # Initialize Database for comprehensive tracking
            signals_db_path = ROOT / "data" / "trading.db"

            if signals_db_path.exists() and Database is not None:
                db = Database(str(signals_db_path))

                # Create tabs for different views
                signal_tabs = st.tabs(["📊 Performance Stats", "📋 Signal History", "📈 Trade Results"])

                with signal_tabs[0]:  # Performance Stats
                    st.markdown("#### Trading Performance")

                    # Get performance stats from database
                    try:
                        stats = db.get_performance_stats()

                        # Display key metrics in cards
                        perf_cols = st.columns(4)

                        with perf_cols[0]:
                            win_rate = stats.get('win_rate', 0) * 100
                            win_color = "#38ef7d" if win_rate >= 50 else "#f5576c"
                            st.markdown(f"""
                            <div class="metric-card" style="border-left-color: {win_color}">
                                <div class="metric-label">Win Rate</div>
                                <div class="metric-value" style="color: {win_color}; font-size: 2rem;">{win_rate:.1f}%</div>
                                <div style="color: #6c757d; font-size: 0.85rem;">
                                    {stats.get('winners', 0)}W / {stats.get('losers', 0)}L
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with perf_cols[1]:
                            total_pnl = stats.get('total_pnl', 0)
                            pnl_color = "#38ef7d" if total_pnl >= 0 else "#f5576c"
                            st.markdown(f"""
                            <div class="metric-card" style="border-left-color: {pnl_color}">
                                <div class="metric-label">Total PnL</div>
                                <div class="metric-value" style="color: {pnl_color}; font-size: 2rem;">{total_pnl:+.2f}%</div>
                                <div style="color: #6c757d; font-size: 0.85rem;">
                                    Avg: {stats.get('avg_pnl', 0):+.2f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with perf_cols[2]:
                            total_signals = stats.get('total_signals', 0)
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Total Signals</div>
                                <div class="metric-value" style="font-size: 2rem;">{total_signals}</div>
                                <div style="color: #6c757d; font-size: 0.85rem;">
                                    Resolved: {stats.get('resolved_trades', 0)}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with perf_cols[3]:
                            # Calculate profit factor
                            winners = stats.get('winners', 0)
                            losers = stats.get('losers', 0)
                            profit_factor = winners / losers if losers > 0 else winners if winners > 0 else 0
                            pf_color = "#38ef7d" if profit_factor >= 1.5 else "#ffc107" if profit_factor >= 1 else "#f5576c"
                            st.markdown(f"""
                            <div class="metric-card" style="border-left-color: {pf_color}">
                                <div class="metric-label">Profit Factor</div>
                                <div class="metric-value" style="color: {pf_color}; font-size: 2rem;">{profit_factor:.2f}</div>
                                <div style="color: #6c757d; font-size: 0.85rem;">
                                    {'Good' if profit_factor >= 1.5 else 'Neutral' if profit_factor >= 1 else 'Poor'}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    except Exception as e:
                        st.info("No performance data yet. Trade outcomes will appear here once signals are resolved.")
                        logger.debug(f"Performance stats error: {e}")

                with signal_tabs[1]:  # Signal History
                    st.markdown("#### Recent Signals")

                    try:
                        # Get signals from database
                        signals_list = db.get_signals(limit=20)

                        if signals_list:
                            # Convert to display format
                            signals_data = []
                            for sig in signals_list:
                                outcome_emoji = "✅" if sig.actual_outcome == "WIN" else "❌" if sig.actual_outcome == "LOSS" else "⏳"
                                direction_emoji = "🟢" if sig.signal_type.value == "BUY" else "🔴" if sig.signal_type.value == "SELL" else "⚪"

                                signals_data.append({
                                    "Time": sig.timestamp.strftime('%m-%d %H:%M') if sig.timestamp else "N/A",
                                    "Direction": f"{direction_emoji} {sig.signal_type.value}",
                                    "Strength": sig.strength.value if sig.strength else "N/A",
                                    "Confidence": f"{sig.confidence*100:.1f}%" if sig.confidence else "N/A",
                                    "Entry": f"${sig.price:,.2f}" if sig.price else "N/A",
                                    "SL": f"${sig.stop_loss:,.2f}" if sig.stop_loss else "N/A",
                                    "TP": f"${sig.take_profit:,.2f}" if sig.take_profit else "N/A",
                                    "Outcome": outcome_emoji,
                                    "PnL": f"{sig.pnl_percent:+.2f}%" if sig.pnl_percent else "-"
                                })

                            st.dataframe(
                                pd.DataFrame(signals_data),
                                use_container_width=True,
                                hide_index=True,
                                height=400
                            )

                            # Summary below table
                            pending = sum(1 for s in signals_list if s.actual_outcome in [None, 'PENDING'])
                            st.caption(f"Showing last {len(signals_list)} signals | ⏳ {pending} pending resolution")

                        else:
                            st.info("No trading signals generated yet. Signals will appear here when confidence thresholds are met.")

                    except Exception as e:
                        st.info("Signal history not available. Start the analysis engine to generate signals.")
                        logger.debug(f"Signals error: {e}")

                with signal_tabs[2]:  # Trade Results
                    st.markdown("#### Trade Results History")

                    try:
                        trade_results = db.get_trade_results(limit=20)

                        if trade_results:
                            trades_data = []
                            for trade in trade_results:
                                hit_emoji = "🎯" if trade.get('hit_target') else "🛑" if trade.get('hit_stop') else "⏹️"
                                pnl = trade.get('pnl_percent', 0)
                                pnl_color = "green" if pnl > 0 else "red" if pnl < 0 else "gray"

                                trades_data.append({
                                    "Entry Time": trade.get('entry_time', 'N/A')[:16] if trade.get('entry_time') else "N/A",
                                    "Direction": trade.get('direction', 'N/A'),
                                    "Entry $": f"${trade.get('entry_price', 0):,.2f}",
                                    "Exit $": f"${trade.get('exit_price', 0):,.2f}",
                                    "Result": hit_emoji,
                                    "PnL %": f"{pnl:+.2f}%",
                                    "PnL $": f"${trade.get('pnl_absolute', 0):,.2f}"
                                })

                            st.dataframe(
                                pd.DataFrame(trades_data),
                                use_container_width=True,
                                hide_index=True
                            )

                        else:
                            st.info("No trade results yet. Results will appear here after trades are executed and resolved.")

                    except Exception as e:
                        st.info("Trade results not available.")
                        logger.debug(f"Trade results error: {e}")

                # Close database connection
                db.close()

            else:
                st.info("💡 Signal database not found. Start the analysis engine (`python run_analysis.py`) to begin tracking signals.")

        except Exception as e:
            st.warning(f"⚠️ Could not load signal data: {str(e)}")
            logger.error(f"Signals error: {e}")

    # ==============================================================================
    # UPDATE PAPER TRADING POSITIONS WITH CURRENT PRICES
    # ==============================================================================
    if st.session_state.paper_trader and data and data['success']:
        # Update all positions with current market price
        try:
            prices = {st.session_state.selected_symbol: current_price}
            st.session_state.paper_trader.update_positions(prices)
        except Exception as e:
            logger.debug(f"Could not update paper trading positions: {e}")

    # ==============================================================================
    # ADVANCED FEATURES VIEW - ALL 5 CRITICAL FEATURES
    # ==============================================================================
    if st.session_state.view_mode == 'advanced' and AI_AVAILABLE:
        st.markdown("---")
        st.markdown("## 🚀 Advanced Trading Features")
        st.markdown("*Professional-grade tools for backtesting, paper trading, portfolio tracking, and risk management*")

        # Create tabs for all 5 advanced features
        feature_tabs = st.tabs([
            "📊 Backtesting",
            "💼 Paper Trading",
            "💰 Portfolio",
            "🛡️ Risk Management",
            "🔔 Alerts"
        ])

        with feature_tabs[0]:  # BACKTESTING
            if st.session_state.db:
                render_backtesting_interface(st.session_state.db)
            else:
                st.error("Database not available. Please ensure AI modules are installed.")

        with feature_tabs[1]:  # PAPER TRADING
            if st.session_state.paper_trader and data and data['success']:
                render_paper_trading(
                    st.session_state.paper_trader,
                    current_price,
                    st.session_state.selected_symbol
                )
            else:
                st.error("Paper trading not available. Please ensure AI modules are installed and market data is loaded.")

        with feature_tabs[2]:  # PORTFOLIO TRACKING
            if st.session_state.db:
                render_portfolio_tracking(st.session_state.db, st.session_state.paper_trader)
            else:
                st.error("Portfolio tracking not available. Please ensure database is initialized.")

        with feature_tabs[3]:  # RISK MANAGEMENT
            if st.session_state.db:
                render_risk_management(st.session_state.db, st.session_state.paper_trader)
            else:
                st.error("Risk management not available. Please ensure database is initialized.")

        with feature_tabs[4]:  # REAL-TIME ALERTS
            render_realtime_alerts()

        st.markdown("---")

    elif st.session_state.view_mode == 'advanced' and not AI_AVAILABLE:
        st.markdown("---")
        st.error("🚨 Advanced Features Unavailable")
        st.warning("""
        Advanced trading features require AI modules to be installed.

        **To enable these features:**
        ```bash
        pip install -r requirements.txt
        ```

        Then restart the dashboard.
        """)
        st.markdown("---")

    # ==============================================================================
    # SYSTEM STATUS PANEL
    # ==============================================================================
    st.markdown("---")
    st.markdown("### 🖥️ System Status")

    status_tabs = st.tabs(["📊 Database", "📡 Data Collection", "🔔 Notifications"])

    with status_tabs[0]:  # Database Stats
        st.markdown("#### Database Statistics")

        try:
            signals_db_path = ROOT / "data" / "trading.db"
            if signals_db_path.exists() and Database is not None:
                db = Database(str(signals_db_path))

                db_cols = st.columns(4)

                # Get candle count
                try:
                    candle_count = db.get_candle_count()
                except Exception:
                    candle_count = 0

                # Get signal count
                try:
                    signal_count = db.get_signal_count()
                except Exception:
                    signal_count = 0

                with db_cols[0]:
                    st.metric("Total Candles", f"{candle_count:,}")

                with db_cols[1]:
                    st.metric("Total Signals", signal_count)

                with db_cols[2]:
                    # Database file size
                    db_size = signals_db_path.stat().st_size / (1024 * 1024)  # MB
                    st.metric("Database Size", f"{db_size:.2f} MB")

                with db_cols[3]:
                    # Last modified
                    import os
                    last_mod = datetime.fromtimestamp(os.path.getmtime(signals_db_path))
                    st.metric("Last Update", last_mod.strftime('%H:%M:%S'))

                db.close()

            else:
                st.info("Database not initialized. Start the analysis engine to create it.")

        except Exception as e:
            st.warning(f"Could not get database stats: {e}")
            logger.debug(f"Database stats error: {e}")

    with status_tabs[1]:  # Data Collection Status
        st.markdown("#### Data Collection Status")

        try:
            # Check if DataService is available and get status
            data_cols = st.columns(4)

            with data_cols[0]:
                # Analysis engine status
                engine_running, engine_pid = is_analysis_running()
                engine_color = "#38ef7d" if engine_running else "#f5576c"
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {engine_color}">
                    <div class="metric-label">Analysis Engine</div>
                    <div class="metric-value" style="color: {engine_color}; font-size: 1.5rem;">
                        {'Running' if engine_running else 'Stopped'}
                    </div>
                    <div style="color: #6c757d; font-size: 0.85rem;">
                        {f'PID: {engine_pid}' if engine_running else 'Not started'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with data_cols[1]:
                # Exchange connection status
                exchange_ok = st.session_state.get('exchange_obj') is not None
                exc_color = "#38ef7d" if exchange_ok else "#f5576c"
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {exc_color}">
                    <div class="metric-label">Exchange</div>
                    <div class="metric-value" style="color: {exc_color}; font-size: 1.5rem;">
                        {'Connected' if exchange_ok else 'Disconnected'}
                    </div>
                    <div style="color: #6c757d; font-size: 0.85rem;">
                        {st.session_state.get('selected_exchange', 'N/A').title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with data_cols[2]:
                # API mode indicator
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Data Mode</div>
                    <div class="metric-value" style="font-size: 1.5rem;">REST API</div>
                    <div style="color: #6c757d; font-size: 0.85rem;">
                        Auto-refresh: {'ON' if st.session_state.get('auto_refresh') else 'OFF'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with data_cols[3]:
                # Last fetch time
                last_fetch = st.session_state.get('last_fetch_time')
                fetch_str = last_fetch.strftime('%H:%M:%S') if last_fetch else 'Never'
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Last Fetch</div>
                    <div class="metric-value" style="font-size: 1.5rem;">{fetch_str}</div>
                    <div style="color: #6c757d; font-size: 0.85rem;">
                        Updates: {st.session_state.get('update_count', 0)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Could not get data collection status: {e}")
            logger.debug(f"Data collection status error: {e}")

    with status_tabs[2]:  # Notification Status
        st.markdown("#### Notification System")

        try:
            notif_cols = st.columns(3)

            with notif_cols[0]:
                # Desktop notifications
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Desktop Notifications</div>
                    <div class="metric-value" style="font-size: 1.2rem;">Enabled</div>
                    <div style="color: #6c757d; font-size: 0.85rem;">
                        System alerts
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("🔔 Test Desktop", key="test_desktop"):
                    try:
                        if Notifier is not None:
                            notifier = Notifier()
                            notifier._send_desktop("Test Notification", "Dashboard test - notifications working!")
                            st.success("Desktop notification sent!")
                        else:
                            st.warning("Notifier not available")
                    except Exception as e:
                        st.error(f"Failed: {e}")

            with notif_cols[1]:
                # Sound alerts
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Sound Alerts</div>
                    <div class="metric-value" style="font-size: 1.2rem;">Available</div>
                    <div style="color: #6c757d; font-size: 0.85rem;">
                        Audio feedback
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("🔊 Test Sound", key="test_sound"):
                    try:
                        if Notifier is not None:
                            notifier = Notifier()
                            notifier._play_sound()
                            st.success("Sound played!")
                        else:
                            st.warning("Notifier not available")
                    except Exception as e:
                        st.error(f"Failed: {e}")

            with notif_cols[2]:
                # Telegram status
                config_path = ROOT / "config.yaml"
                telegram_configured = False
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                        telegram_configured = bool(config.get('notifications', {}).get('telegram', {}).get('bot_token'))
                    except Exception:
                        pass

                tg_color = "#38ef7d" if telegram_configured else "#ffc107"
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {tg_color}">
                    <div class="metric-label">Telegram</div>
                    <div class="metric-value" style="color: {tg_color}; font-size: 1.2rem;">
                        {'Configured' if telegram_configured else 'Not Configured'}
                    </div>
                    <div style="color: #6c757d; font-size: 0.85rem;">
                        {'Bot connected' if telegram_configured else 'Add bot token in config'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Could not get notification status: {e}")
            logger.debug(f"Notification status error: {e}")

    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(2)  # 2 second refresh
        st.rerun()

if __name__ == "__main__":
    main()
