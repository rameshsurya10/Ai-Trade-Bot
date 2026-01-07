"""
AI Trade Bot - Unified Dashboard
=================================
Single-page dashboard with all trading features.

Features:
- Real-time market data
- AI predictions with ensemble analysis
- Portfolio tracking
- Paper trading
- Backtesting
- Risk management
- Performance metrics
- System monitoring

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
import yaml
import os
import signal
import subprocess
import html
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
PID_FILE = ROOT / "run_analysis.pid"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Trade Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MODULE IMPORTS (with fallbacks)
# =============================================================================

# WebSocket Data Provider (REQUIRED - no fallback)
# Import directly to avoid torch dependency from src/__init__.py
from src.data.provider import UnifiedDataProvider

try:
    from src.core.database import Database
    from src.core.metrics import MetricsCalculator, SignalQualityScorer
    from src.core.validation import OrderValidator
    from src.paper_trading import PaperTradingSimulator, OrderSide, OrderType
    from src.advanced_predictor import AdvancedPredictor
    from src.news.collector import NewsCollector
    from src.news.aggregator import SentimentAggregator
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    logger.warning(f"AI modules not available: {e}")

# =============================================================================
# STYLES
# =============================================================================

st.markdown("""
<style>
    /* Clean Light Theme */
    .stApp { background-color: #f8f9fa; }

    /* Hide Streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Navigation */
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: white;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0.3rem 0;
    }
    .metric-card .delta { font-size: 0.9rem; }
    .metric-card.positive { border-left-color: #28a745; }
    .metric-card.negative { border-left-color: #dc3545; }
    .metric-card.warning { border-left-color: #ffc107; }

    /* Signal Card */
    .signal-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .signal-buy { border: 3px solid #28a745; }
    .signal-sell { border: 3px solid #dc3545; }
    .signal-neutral { border: 3px solid #6c757d; }

    /* Tables */
    .dataframe { font-size: 0.85rem; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Navigation
        'page': 'Dashboard',

        # Market settings
        'exchange': 'binance',
        'symbol': 'BTC/USDT',
        'timeframe': '1h',

        # Auto-refresh
        'auto_refresh': True,
        'refresh_interval': 5,

        # WebSocket Data Provider (ONLY data source)
        'data_provider': None,

        # AI/ML
        'predictor': None,
        'advanced_predictor': None,

        # Paper trading
        'paper_trader': None,
        'paper_capital': 10000,

        # Database
        'db': None,

        # Metrics
        'metrics_calculator': None,
        'signal_scorer': None,

        # Order validator
        'order_validator': None,

        # News & Sentiment
        'news_collector': None,
        'sentiment_aggregator': None,

        # Tracked currencies
        'tracked_symbols': ['BTC/USDT', 'ETH/USDT'],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# INITIALIZATION
# =============================================================================

@st.cache_resource
def get_data_provider():
    """Get singleton UnifiedDataProvider instance."""
    return UnifiedDataProvider.get_instance(str(CONFIG_PATH))


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_config() -> dict:
    """
    Load configuration file with caching.

    Returns:
        dict: Configuration dictionary with '_error' key if failed, empty dict if file doesn't exist.
    """
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
            return {'_error': f"Configuration file is malformed: {e}"}
        except PermissionError as e:
            logger.error(f"Permission denied reading config: {e}")
            return {'_error': f"Permission denied: Cannot read {CONFIG_PATH}. Check file permissions."}
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            return {'_error': f"Error reading configuration: {e}"}
    logger.warning(f"Config file not found: {CONFIG_PATH}")
    return {}


def initialize_components():
    """Initialize all components."""
    # WebSocket Data Provider (ONLY data source)
    if st.session_state.data_provider is None:
        st.session_state.data_provider = get_data_provider()
        logger.info("Got data provider singleton")

    # Always check if provider needs to be started (may be cached but not running)
    provider = st.session_state.data_provider
    if provider and not provider.is_running:
        logger.info("Starting WebSocket provider...")

        # Subscribe to tracked symbols
        for symbol in st.session_state.tracked_symbols:
            provider.subscribe(
                symbol,
                exchange=st.session_state.exchange,
                interval=st.session_state.timeframe
            )

        # Start the provider
        provider.start()
        logger.info(f"WebSocket provider started for {st.session_state.tracked_symbols}")

    # Paper trader
    if st.session_state.paper_trader is None and AI_AVAILABLE:
        st.session_state.paper_trader = PaperTradingSimulator(
            initial_cash=st.session_state.paper_capital
        )

    # Advanced predictor
    if st.session_state.advanced_predictor is None and AI_AVAILABLE:
        st.session_state.advanced_predictor = AdvancedPredictor()

    # Database
    if st.session_state.db is None and AI_AVAILABLE:
        db_path = ROOT / "data" / "trading.db"
        db_path.parent.mkdir(exist_ok=True)
        st.session_state.db = Database(str(db_path))

    # Metrics calculator
    if st.session_state.metrics_calculator is None and AI_AVAILABLE:
        st.session_state.metrics_calculator = MetricsCalculator()

    # Signal scorer
    if st.session_state.signal_scorer is None and AI_AVAILABLE:
        st.session_state.signal_scorer = SignalQualityScorer()

    # Order validator
    if st.session_state.order_validator is None and AI_AVAILABLE:
        config = load_config()
        st.session_state.order_validator = OrderValidator(config.get('risk', {}))

    # News collector and sentiment aggregator
    if st.session_state.news_collector is None and AI_AVAILABLE:
        config = load_config()
        news_config = config.get('news', {})
        if news_config.get('enabled', False):
            # Validate required API keys
            required_keys = ['NEWSAPI_KEY']
            missing_keys = [k for k in required_keys if not os.getenv(k)]

            if missing_keys:
                warning_msg = f"News enabled but missing API keys: {', '.join(missing_keys)}"
                logger.warning(warning_msg)
                st.session_state.news_collector = None
                st.session_state.sentiment_aggregator = None
                st.warning(f"⚠️ {warning_msg}. Add them to .env file.")
            else:
                try:
                    st.session_state.news_collector = NewsCollector(
                        database=st.session_state.db,
                        config=news_config
                    )

                    st.session_state.sentiment_aggregator = SentimentAggregator(
                        database=st.session_state.db,
                        config=news_config.get('features', {})
                    )
                    logger.info("Sentiment aggregator initialized")

                    # Start news collector after both components are initialized
                    st.session_state.news_collector.start()
                    logger.info("News collector started")
                except ValueError as e:
                    error_msg = f"Invalid news configuration: {e}"
                    logger.error(error_msg)
                    st.session_state.news_collector = None
                    st.session_state.sentiment_aggregator = None
                    st.error(f"❌ {error_msg}")
                except Exception as e:
                    error_msg = f"Failed to initialize news components: {e}"
                    logger.error(error_msg)
                    st.session_state.news_collector = None
                    st.session_state.sentiment_aggregator = None
                    st.error(f"❌ {error_msg}")

initialize_components()

# =============================================================================
# CLEANUP HANDLERS
# =============================================================================

def cleanup_resources() -> None:
    """Cleanup background threads and connections on shutdown."""
    try:
        # Check if Streamlit session state is still available
        if not hasattr(st, 'session_state'):
            return

        # Stop news collector if running
        news_collector = st.session_state.get('news_collector')
        if news_collector:
            try:
                if hasattr(news_collector, 'stop'):
                    news_collector.stop()
                    logger.info("News collector stopped")
            except Exception as e:
                logger.warning(f"Error stopping news collector: {e}")

        # Stop data provider if running
        data_provider = st.session_state.get('data_provider')
        if data_provider:
            try:
                if hasattr(data_provider, 'stop'):
                    data_provider.stop()
                    logger.info("Data provider stopped")
            except Exception as e:
                logger.warning(f"Error stopping data provider: {e}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Register cleanup handler
import atexit
atexit.register(cleanup_resources)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_engine_running() -> tuple:
    """Check if analysis engine is running."""
    if not PID_FILE.exists():
        return False, None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True, pid
    except (ProcessLookupError, ValueError):
        PID_FILE.unlink(missing_ok=True)
        return False, None


def fetch_market_data(symbol: str, timeframe: str = '1h', limit: int = 200) -> dict:
    """Fetch market data from WebSocket provider (real-time only)."""
    provider = st.session_state.data_provider

    if not provider:
        return {'success': False, 'error': 'WebSocket provider not initialized'}

    if not provider.is_running:
        return {'success': False, 'error': 'WebSocket provider not running - click Refresh'}

    try:
        # Get real-time price from tick
        tick = provider.get_tick(symbol)
        price = tick.price if tick else 0

        # Get buffered candles (includes current in-progress candle)
        df = provider.get_candles(symbol, limit=limit)

        # Get provider status for debugging
        status = provider.get_status()
        ticks_received = status.get('ticks_received', 0)

        if df.empty:
            if ticks_received == 0:
                return {'success': False, 'error': f'Connecting to WebSocket... (no data yet)'}
            elif price > 0:
                # Have tick data but no candles yet - create minimal candle from tick
                df = pd.DataFrame([{
                    'timestamp': tick.timestamp,
                    'datetime': datetime.fromtimestamp(tick.timestamp / 1000) if tick.timestamp is not None else datetime.utcnow(),
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': tick.quantity if tick else 0
                }])
            else:
                return {'success': False, 'error': f'WebSocket connected, waiting for data... ({ticks_received} ticks)'}

        # Calculate 24h stats from candles
        if len(df) >= 24:
            last_24h = df.tail(24)
            high_24h = last_24h['high'].max()
            low_24h = last_24h['low'].min()
            volume_24h = last_24h['volume'].sum()
            open_24h = last_24h.iloc[0]['open']
            change_pct = ((price - open_24h) / open_24h * 100) if open_24h > 0 else 0
        else:
            high_24h = df['high'].max()
            low_24h = df['low'].min()
            volume_24h = df['volume'].sum()
            change_pct = 0

        return {
            'success': True,
            'source': 'websocket',
            'df': df,
            'price': price,
            'change_pct': change_pct,
            'high_24h': high_24h,
            'low_24h': low_24h,
            'volume_24h': volume_24h,
        }

    except Exception as e:
        logger.error(f"WebSocket data error: {e}")
        return {'success': False, 'error': str(e)}


def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Calculate RSI."""
    if len(prices) < period:
        return 50.0

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss < 1e-10:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def format_number(value: float, prefix: str = '$') -> str:
    """Format large numbers."""
    if abs(value) >= 1e9:
        return f"{prefix}{value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{prefix}{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{prefix}{value/1e3:.1f}K"
    else:
        return f"{prefix}{value:,.2f}"


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_metric_card(label: str, value: str, delta: str = "", card_class: str = ""):
    """Render a metric card with XSS protection."""
    # Escape user-controlled content
    label_escaped = html.escape(label)
    value_escaped = html.escape(value)
    delta_html = f'<div class="delta">{html.escape(delta)}</div>' if delta else ''
    st.markdown(f"""
        <div class="metric-card {card_class}">
            <div class="label">{label_escaped}</div>
            <div class="value">{value_escaped}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def render_signal_card(direction: str, confidence: float, price: float, stop_loss: float, take_profit: float):
    """Render basic signal card (legacy function for compatibility)."""
    signal_class = f"signal-{direction.lower()}"
    color = "#28a745" if direction == "BUY" else "#dc3545" if direction == "SELL" else "#6c757d"

    st.markdown(f"""
        <div class="signal-card {signal_class}">
            <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 0.5rem;">AI SIGNAL</div>
            <div style="font-size: 3rem; font-weight: 800; color: {color};">{direction}</div>
            <div style="font-size: 1.2rem; margin-top: 0.5rem;">Confidence: {confidence*100:.1f}%</div>
            <div style="margin-top: 1rem; display: flex; justify-content: space-around;">
                <div>
                    <div style="font-size: 0.75rem; color: #6c757d;">STOP LOSS</div>
                    <div style="color: #dc3545; font-weight: 600;">${stop_loss:,.2f}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #6c757d;">ENTRY</div>
                    <div style="font-weight: 600;">${price:,.2f}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #6c757d;">TAKE PROFIT</div>
                    <div style="color: #28a745; font-weight: 600;">${take_profit:,.2f}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_enhanced_signal_card(prediction, price: float, account_balance: float = 10000):
    """Render comprehensive signal card with all metrics."""
    signal_class = f"signal-{prediction.direction.lower()}"
    color = "#28a745" if prediction.direction == "BUY" else "#dc3545" if prediction.direction == "SELL" else "#6c757d"

    # Calculate dollar amounts for position
    position_size_dollars = account_balance * prediction.kelly_fraction
    shares = position_size_dollars / price if price > 0 else 0
    expected_profit_dollars = shares * abs(prediction.take_profit - price)
    expected_loss_dollars = shares * abs(price - prediction.stop_loss)

    st.markdown(f"""
        <div class="signal-card {signal_class}">
            <!-- Main Signal -->
            <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 0.5rem;">🤖 AI SIGNAL</div>
            <div style="font-size: 3rem; font-weight: 800; color: {color};">{prediction.direction}</div>
            <div style="font-size: 1.2rem; margin-top: 0.5rem;">
                Confidence: {prediction.confidence*100:.1f}%
            </div>

            <!-- Price Levels -->
            <div style="margin-top: 1.5rem; display: flex; justify-content: space-around; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                <div style="text-align: center;">
                    <div style="font-size: 0.75rem; color: #6c757d; font-weight: 600;">STOP LOSS</div>
                    <div style="color: #dc3545; font-weight: 700; font-size: 1.1rem;">${prediction.stop_loss:,.2f}</div>
                    <div style="font-size: 0.7rem; color: #999;">-{prediction.expected_loss_pct*100:.1f}%</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.75rem; color: #6c757d; font-weight: 600;">ENTRY</div>
                    <div style="font-weight: 700; font-size: 1.1rem;">${price:,.2f}</div>
                    <div style="font-size: 0.7rem; color: #999;">Current</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.75rem; color: #6c757d; font-weight: 600;">TAKE PROFIT</div>
                    <div style="color: #28a745; font-weight: 700; font-size: 1.1rem;">${prediction.take_profit:,.2f}</div>
                    <div style="font-size: 0.7rem; color: #999;">+{prediction.expected_profit_pct*100:.1f}%</div>
                </div>
            </div>

            <!-- Risk Metrics -->
            <div style="margin-top: 1rem; padding: 1rem; background: white; border-radius: 8px; border: 1px solid #e0e0e0;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div>
                        <span style="color: #6c757d; font-size: 0.8rem;">Risk/Reward:</span>
                        <span style="font-weight: 700; margin-left: 0.3rem;">1:{prediction.risk_reward_ratio:.2f}</span>
                    </div>
                    <div>
                        <span style="color: #6c757d; font-size: 0.8rem;">Position Size:</span>
                        <span style="font-weight: 700; margin-left: 0.3rem;">{prediction.kelly_fraction*100:.1f}%</span>
                    </div>
                    <div>
                        <span style="color: #6c757d; font-size: 0.8rem;">Expected Profit:</span>
                        <span style="color: #28a745; font-weight: 700; margin-left: 0.3rem;">${expected_profit_dollars:,.0f}</span>
                    </div>
                    <div>
                        <span style="color: #6c757d; font-size: 0.8rem;">Expected Loss:</span>
                        <span style="color: #dc3545; font-weight: 700; margin-left: 0.3rem;">${expected_loss_dollars:,.0f}</span>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_monte_carlo_section(prediction):
    """Render Monte Carlo simulation results."""
    st.markdown("#### 🎲 Monte Carlo Simulation")

    col1, col2, col3 = st.columns(3)

    with col1:
        prob_profit = prediction.monte_carlo_prob_profit * 100
        color = "positive" if prob_profit > 50 else "negative"
        render_metric_card(
            "Probability of Profit",
            f"{prob_profit:.1f}%",
            card_class=color
        )

    with col2:
        prob_tp = prediction.monte_carlo_prob_take_profit * 100
        render_metric_card(
            "Hit Take Profit",
            f"{prob_tp:.1f}%",
            card_class="positive"
        )

    with col3:
        prob_sl = prediction.monte_carlo_prob_stop_loss * 100
        render_metric_card(
            "Hit Stop Loss",
            f"{prob_sl:.1f}%",
            card_class="negative"
        )

    # Additional metrics
    col4, col5, col6 = st.columns(3)

    with col4:
        render_metric_card(
            "Value at Risk (5%)",
            f"{prediction.monte_carlo_var_5pct*100:.2f}%"
        )

    with col5:
        render_metric_card(
            "Daily Volatility",
            f"{prediction.monte_carlo_volatility_daily*100:.2f}%"
        )

    with col6:
        render_metric_card(
            "Annual Volatility",
            f"{prediction.monte_carlo_volatility_annual*100:.1f}%"
        )


def render_sentiment_section(prediction):
    """Render news sentiment analysis."""
    if prediction.sentiment_score is None:
        st.info("📰 Sentiment analysis not available (news collector not running)")
        return

    st.markdown("#### 📰 News Sentiment")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sentiment = prediction.sentiment_score
        if sentiment > 0.2:
            color = "positive"
            emoji = "📈"
            label = "Bullish"
        elif sentiment < -0.2:
            color = "negative"
            emoji = "📉"
            label = "Bearish"
        else:
            color = ""
            emoji = "➖"
            label = "Neutral"

        render_metric_card(
            "Overall Sentiment",
            f"{emoji} {label}",
            f"{sentiment:+.2f}",
            card_class=color
        )

    with col2:
        if prediction.sentiment_1h is not None:
            render_metric_card(
                "1H Sentiment",
                f"{prediction.sentiment_1h:+.2f}"
            )

    with col3:
        if prediction.sentiment_6h is not None:
            render_metric_card(
                "6H Sentiment",
                f"{prediction.sentiment_6h:+.2f}"
            )

    with col4:
        if prediction.news_volume_1h is not None:
            render_metric_card(
                "News Volume (1H)",
                str(prediction.news_volume_1h)
            )

    # Sentiment momentum indicator
    if prediction.sentiment_momentum is not None:
        momentum = prediction.sentiment_momentum
        if abs(momentum) > 0.1:
            trend = "📈 Improving" if momentum > 0 else "📉 Declining"
            st.info(f"**Sentiment Trend:** {trend} ({momentum:+.2f}/hour)")


def render_algorithm_breakdown(prediction):
    """Render detailed algorithm analysis."""
    st.markdown("#### 🧠 Algorithm Analysis")

    # Create tabs for different algorithm categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Ensemble",
        "🌊 Fourier",
        "📈 Kalman",
        "🎰 Markov",
        "💹 Entropy"
    ])

    with tab1:
        st.markdown("**Algorithm Contribution Weights**")
        weights = prediction.ensemble_weights

        # Display as a table
        weight_df = pd.DataFrame([
            {"Algorithm": k.upper(), "Weight": f"{v*100:.1f}%", "Raw": v}
            for k, v in weights.items()
        ]).sort_values("Raw", ascending=False)

        st.dataframe(weight_df[["Algorithm", "Weight"]], use_container_width=True, hide_index=True)

        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=[k.upper() for k in weights.keys()],
            values=list(weights.values()),
            hole=0.3
        )])
        fig.update_layout(title="Ensemble Composition", height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Signal", prediction.fourier_signal)
        with col2:
            st.metric("Cycle Phase", f"{prediction.fourier_cycle_phase:.2f}")
        with col3:
            st.metric("Dominant Period", f"{prediction.fourier_dominant_period:.1f}")

        st.info(f"""
        **Interpretation:**
        - Phase **{prediction.fourier_cycle_phase:.2f}** means we are {int(prediction.fourier_cycle_phase*100)}% through the current cycle
        - The dominant cycle period is **{prediction.fourier_dominant_period:.0f}** candles
        - Signal: **{prediction.fourier_signal}** (cycle-based trend prediction)
        """)

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trend", prediction.kalman_trend)
        with col2:
            st.metric("Smoothed Price", f"${prediction.kalman_smoothed_price:,.2f}")
        with col3:
            st.metric("Velocity", f"{prediction.kalman_velocity:.4f}")

        st.info(f"""
        **Interpretation:**
        - Kalman filter detected **{prediction.kalman_trend}** trend
        - Noise-filtered price estimate: **${prediction.kalman_smoothed_price:,.2f}**
        - Price velocity (momentum): **{prediction.kalman_velocity:.4f}**
        - Estimation uncertainty: **{prediction.kalman_error_covariance:.4f}**
        """)

    with tab4:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current State", prediction.markov_state)
        with col2:
            st.metric("P(Up)", f"{prediction.markov_probability*100:.1f}%")
        with col3:
            st.metric("P(Down)", f"{prediction.markov_prob_down*100:.1f}%")

        st.write("**State Transition Probabilities:**")
        st.write(f"- Probability of UP move: **{prediction.markov_probability*100:.1f}%**")
        st.write(f"- Probability of DOWN move: **{prediction.markov_prob_down*100:.1f}%**")
        st.write(f"- Probability of NEUTRAL move: **{prediction.markov_prob_neutral*100:.1f}%**")

    with tab5:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Regime", prediction.entropy_regime)
        with col2:
            st.metric("Entropy (Normalized)", f"{prediction.entropy_value:.2f}")
        with col3:
            st.metric("Sample Size", prediction.entropy_n_samples)

        regime_descriptions = {
            "TRENDING": "📈 Low entropy - Clear directional movement",
            "NORMAL": "➖ Medium entropy - Balanced market conditions",
            "CHOPPY": "📊 High entropy - Sideways price action",
            "VOLATILE": "⚡ Very high entropy - Unstable conditions"
        }

        st.info(f"""
        **Interpretation:** {regime_descriptions.get(prediction.entropy_regime, 'Unknown regime')}

        - Entropy score: **{prediction.entropy_value:.2f}** (0 = trending, 1 = random)
        - Raw entropy: **{prediction.entropy_raw_value:.2f}**
        - Based on **{prediction.entropy_n_samples}** recent observations
        """)


def render_price_chart(df: pd.DataFrame, symbol: str):
    """Render candlestick chart with volume."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Volume bars
    colors = ['#dc3545' if c < o else '#28a745' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df['datetime'], y=df['volume'], marker_color=colors, name='Volume', opacity=0.5),
        row=2, col=1
    )

    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

    return fig


# =============================================================================
# PAGE: DASHBOARD
# =============================================================================

def page_dashboard():
    """Main dashboard page."""
    # Fetch market data
    data = fetch_market_data(
        st.session_state.symbol,
        st.session_state.timeframe,
        limit=200
    )

    if not data['success']:
        error_msg = data.get('error', 'Unknown error')
        # Show info message instead of error (WebSocket is loading data)
        if 'waiting for data' in error_msg.lower() or 'connecting' in error_msg.lower():
            st.info(f"🔄 Loading market data... {error_msg}")
            st.info("💡 The WebSocket is connected and receiving data. The dashboard will update automatically once candles are available.")
        else:
            st.warning(f"⚠️ {error_msg}")
        return

    df = data['df']
    price = data['price']
    change_pct = data['change_pct']

    # Header metrics
    st.markdown("### Market Overview")

    cols = st.columns(6)

    with cols[0]:
        color_class = "positive" if change_pct >= 0 else "negative"
        arrow = "▲" if change_pct >= 0 else "▼"
        render_metric_card(
            st.session_state.symbol,
            f"${price:,.2f}",
            f"{arrow} {abs(change_pct):.2f}%",
            color_class
        )

    with cols[1]:
        render_metric_card("24H High", f"${data['high_24h']:,.2f}")

    with cols[2]:
        render_metric_card("24H Low", f"${data['low_24h']:,.2f}")

    with cols[3]:
        render_metric_card("24H Volume", format_number(data['volume_24h']))

    with cols[4]:
        rsi = calculate_rsi(df['close'].values)
        rsi_class = "negative" if rsi > 70 else "positive" if rsi < 30 else ""
        render_metric_card("RSI (14)", f"{rsi:.1f}", "", rsi_class)

    with cols[5]:
        # Show data source and connection status
        provider = st.session_state.data_provider
        if provider and provider.is_connected:
            source = "WebSocket"
            source_class = "positive"
        elif data.get('source') == 'websocket':
            source = "WS Buffered"
            source_class = "warning"
        elif data.get('source') == 'rest':
            source = "REST"
            source_class = "warning"
        else:
            source = "Offline"
            source_class = "negative"
        render_metric_card("Data Source", source, "Real-time" if source == "WebSocket" else "", source_class)

    st.markdown("---")

    # Main content - two columns
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # Price chart
        fig = render_price_chart(df, st.session_state.symbol)
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        # AI Prediction Section
        st.markdown("### 🤖 AI Prediction")

        if AI_AVAILABLE and len(df) >= 50:
            try:
                predictor = st.session_state.advanced_predictor
                if predictor:
                    # Calculate ATR
                    high_low = df['high'] - df['low']
                    atr = float(high_low.rolling(14).mean().iloc[-1]) if len(high_low) >= 14 else price * 0.02

                    # Fetch sentiment features
                    sentiment_features = None
                    if st.session_state.db and st.session_state.sentiment_aggregator:
                        try:
                            latest_timestamp = int(df.iloc[-1]['timestamp'])
                            sentiment_features = st.session_state.db.get_sentiment_features(latest_timestamp)
                            if sentiment_features:
                                logger.info(f"Sentiment features loaded for AI prediction")
                        except Exception as e:
                            logger.warning(f"Could not fetch sentiment: {e}")
                            sentiment_features = None

                    # Make prediction with all available data
                    prediction = predictor.predict(
                        df=df,
                        lstm_probability=0.55,  # TODO: Replace with trained LSTM model
                        atr=atr,
                        sentiment_features=sentiment_features
                    )

                    # Get account balance for position sizing
                    paper_trader = st.session_state.paper_trader
                    account_balance = paper_trader.total_value if paper_trader else 10000

                    # Main enhanced signal card
                    render_enhanced_signal_card(prediction, price, account_balance)

                    st.markdown("---")

                    # Monte Carlo probabilities section
                    render_monte_carlo_section(prediction)

                    st.markdown("---")

                    # Sentiment section (if available)
                    if prediction.sentiment_score is not None:
                        render_sentiment_section(prediction)
                        st.markdown("---")

                    # Detailed algorithm breakdown (expandable)
                    with st.expander("🔍 Detailed Algorithm Analysis", expanded=False):
                        render_algorithm_breakdown(prediction)

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                st.error(f"❌ Prediction unavailable: {e}")

                # Show debug info
                if st.checkbox("Show debug info"):
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info(f"⏳ AI predictions require 50+ candles of data. Currently: {len(df)}")


# =============================================================================
# PAGE: PORTFOLIO
# =============================================================================

def page_portfolio():
    """Portfolio tracking page."""
    st.markdown("### Portfolio Overview")

    paper_trader = st.session_state.paper_trader

    if paper_trader is None:
        st.warning("Paper trading not initialized")
        return

    # Portfolio metrics
    cols = st.columns(4)

    with cols[0]:
        render_metric_card("Total Value", f"${paper_trader.total_value:,.2f}")

    with cols[1]:
        render_metric_card("Cash", f"${paper_trader.cash:,.2f}")

    with cols[2]:
        pnl = paper_trader.total_value - paper_trader.initial_capital
        pnl_pct = (pnl / paper_trader.initial_capital) * 100
        card_class = "positive" if pnl >= 0 else "negative"
        render_metric_card("P&L", f"${pnl:,.2f}", f"{pnl_pct:+.2f}%", card_class)

    with cols[3]:
        positions = paper_trader.get_positions()
        render_metric_card("Positions", str(len(positions)))

    st.markdown("---")

    # Positions table
    st.markdown("### Open Positions")

    if positions:
        pos_data = []
        for pos in positions:
            pos_data.append({
                'Symbol': pos.symbol,
                'Side': pos.side.value,
                'Quantity': f"{pos.quantity:.6f}",
                'Entry Price': f"${pos.entry_price:,.2f}",
                'Current Price': f"${pos.current_price:,.2f}",
                'P&L': f"${pos.unrealized_pnl:,.2f}",
                'P&L %': f"{pos.unrealized_pnl_percent:.2f}%"
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")

    # Trade history
    st.markdown("### Trade History")

    trades = paper_trader.get_trades()
    if trades:
        trade_data = []
        for t in trades[-20:]:  # Last 20 trades
            trade_data.append({
                'Time': t.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Symbol': t.symbol,
                'Side': t.side.value,
                'Quantity': f"{t.quantity:.6f}",
                'Price': f"${t.price:,.2f}",
                'P&L': f"${t.realized_pnl:,.2f}" if t.realized_pnl else "-"
            })
        st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)
    else:
        st.info("No trades yet")


# =============================================================================
# PAGE: FOREX MARKETS
# =============================================================================

def page_forex_markets():
    """Forex markets monitoring page."""
    st.markdown("### Forex Markets")

    # Try to import forex module
    try:
        from src.portfolio.forex import (
            FOREX_PAIRS, MAJOR_PAIRS, CROSS_PAIRS,
            get_current_session
        )
        forex_available = True
    except ImportError as e:
        st.error(f"Forex module not available: {e}")
        forex_available = False
        return

    # Load config
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        forex_config = config.get('forex', {})
        forex_enabled = forex_config.get('enabled', False)
    except Exception:
        forex_config = {}
        forex_enabled = False

    # Status row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = "Active" if forex_enabled else "Disabled"
        status_color = "positive" if forex_enabled else "warning"
        render_metric_card("Forex Status", status, card_class=status_color)

    with col2:
        session = get_current_session() if forex_available else "unknown"
        session_display = session.replace("_", " ").title()
        render_metric_card("Market Session", session_display)

    with col3:
        total_pairs = len(FOREX_PAIRS) if forex_available else 0
        render_metric_card("Available Pairs", str(total_pairs))

    with col4:
        leverage = forex_config.get('leverage', {}).get('default', 50)
        render_metric_card("Max Leverage", f"{leverage}:1")

    st.markdown("---")

    # Forex pairs display
    st.markdown("### Currency Pairs")

    # Tabs for Majors and Crosses
    tab_majors, tab_crosses, tab_all = st.tabs(["Majors (7)", "Crosses (7)", "All Pairs (14)"])

    with tab_majors:
        st.markdown("#### Major Currency Pairs")
        _render_forex_pairs_grid(MAJOR_PAIRS if forex_available else {})

    with tab_crosses:
        st.markdown("#### Cross Currency Pairs")
        _render_forex_pairs_grid(CROSS_PAIRS if forex_available else {})

    with tab_all:
        st.markdown("#### All Currency Pairs")
        _render_forex_pairs_grid(FOREX_PAIRS if forex_available else {})

    st.markdown("---")

    # Position sizing calculator
    st.markdown("### Position Size Calculator")

    col1, col2 = st.columns(2)

    with col1:
        calc_symbol = st.selectbox(
            "Currency Pair",
            list(FOREX_PAIRS.keys()) if forex_available else ["EUR/USD"],
            key="forex_calc_symbol"
        )
        account_balance = st.number_input("Account Balance ($)", value=10000.0, min_value=100.0, step=100.0)
        risk_percent = st.number_input("Risk Per Trade (%)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)

    with col2:
        stop_pips = st.number_input("Stop Loss (pips)", value=50.0, min_value=1.0, step=1.0)
        current_price = st.number_input("Current Price", value=1.1000, min_value=0.0001, step=0.0001, format="%.4f")

    if st.button("Calculate Position Size"):
        try:
            from src.portfolio.forex import ForexPositionSizer
            sizer = ForexPositionSizer(max_risk_percent=risk_percent)
            result = sizer.calculate_position_size(
                symbol=calc_symbol,
                account_equity=account_balance,
                stop_pips=stop_pips,
                current_price=current_price
            )

            st.success(f"""
            **Position Size:** {result.lots:.2f} lots ({result.units:,.0f} units)

            **Risk Amount:** ${result.risk_amount:.2f}

            **Margin Required:** ${result.margin_required:,.2f}

            **Pip Value:** ${result.pip_value:.2f}/pip
            """)

            if result.was_reduced:
                st.warning(f"Position reduced: {result.reduction_reason}")

        except Exception as e:
            st.error(f"Calculation error: {e}")

    st.markdown("---")

    # OANDA connection status
    st.markdown("### OANDA Connection")

    oanda_key = os.getenv("OANDA_API_KEY", "")
    oanda_account = os.getenv("OANDA_ACCOUNT_ID", "")

    if oanda_key and oanda_account:
        st.success("OANDA credentials configured")

        if st.button("Test OANDA Connection"):
            try:
                from src.brokerages.oanda import OandaBrokerage
                broker = OandaBrokerage(practice=True)
                if broker.connect():
                    summary = broker.get_account_summary()
                    st.success(f"""
                    **Connected to OANDA**

                    Account ID: {summary.get('id', 'N/A')}

                    Balance: ${summary.get('balance', 0):,.2f}

                    NAV: ${summary.get('nav', 0):,.2f}

                    Margin Available: ${summary.get('margin_available', 0):,.2f}
                    """)
                    broker.disconnect()
                else:
                    st.error("Failed to connect to OANDA")
            except Exception as e:
                st.error(f"Connection error: {e}")
    else:
        st.warning("OANDA credentials not configured. Set OANDA_API_KEY and OANDA_ACCOUNT_ID in .env")

    # Trading session info
    st.markdown("---")
    st.markdown("### Trading Sessions (UTC)")

    session_data = {
        'Session': ['Sydney', 'Tokyo', 'London', 'New York'],
        'Open (UTC)': ['21:00', '00:00', '08:00', '13:00'],
        'Close (UTC)': ['06:00', '09:00', '17:00', '22:00'],
        'Best Pairs': ['AUD/USD, NZD/USD', 'USD/JPY, EUR/JPY', 'EUR/USD, GBP/USD', 'EUR/USD, USD/CAD'],
        'Liquidity': ['Low', 'Medium', 'High', 'High']
    }
    st.dataframe(pd.DataFrame(session_data), use_container_width=True, hide_index=True)


def _render_forex_pairs_grid(pairs: dict):
    """Render forex pairs in a grid layout."""
    if not pairs:
        st.info("No pairs available")
        return

    # Convert to list for grid
    pairs_list = list(pairs.items())

    # 4 columns grid
    cols = st.columns(4)

    for i, (symbol, config) in enumerate(pairs_list):
        with cols[i % 4]:
            pip_size = config.pip_size if hasattr(config, 'pip_size') else 0.0001
            typical_spread = config.typical_spread if hasattr(config, 'typical_spread') else 1.0
            category = config.category if hasattr(config, 'category') else "major"

            # Card styling based on category
            border_color = "#667eea" if category == "major" else "#764ba2"

            st.markdown(f"""
            <div style="
                background: white;
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid {border_color};
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            ">
                <div style="font-size: 1.1rem; font-weight: 700; color: #1a1a2e;">{symbol}</div>
                <div style="font-size: 0.75rem; color: #6c757d; margin-top: 0.3rem;">
                    Pip: {pip_size} | Spread: ~{typical_spread:.1f}
                </div>
                <div style="font-size: 0.7rem; color: #999; margin-top: 0.2rem;">
                    {category.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE: PAPER TRADING
# =============================================================================

def page_paper_trading():
    """Paper trading page."""
    st.markdown("### Paper Trading")

    paper_trader = st.session_state.paper_trader

    if paper_trader is None:
        st.warning("Paper trading not initialized")
        return

    # Fetch current price
    data = fetch_market_data(st.session_state.symbol, limit=1)
    current_price = data.get('price', 0) if data['success'] else 0

    # Order form
    st.markdown("#### Place Order")

    col1, col2 = st.columns(2)

    with col1:
        order_side = st.radio("Side", ["BUY", "SELL"], horizontal=True)
        quantity = st.number_input("Quantity", min_value=0.0001, value=0.01, step=0.001, format="%.4f")

    with col2:
        order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])
        if order_type == "LIMIT":
            limit_price = st.number_input("Limit Price", value=float(current_price), step=0.01)
        else:
            limit_price = current_price

    # Order preview
    order_value = quantity * limit_price
    st.markdown(f"**Order Value:** ${order_value:,.2f}")

    # Validate order
    if st.session_state.order_validator:
        order = {
            'symbol': st.session_state.symbol,
            'quantity': quantity,
            'side': order_side,
            'order_type': order_type,
            'price': limit_price,
            'current_price': current_price
        }

        portfolio = {
            'total_value': paper_trader.total_value,
            'cash': paper_trader.cash,
            'daily_pnl': 0
        }

        validation = st.session_state.order_validator.validate(order, portfolio)

        if validation.warnings:
            for w in validation.warnings:
                st.warning(w)

        if validation.errors:
            for e in validation.errors:
                st.error(e)

    # Submit button
    if st.button("Submit Order", type="primary", disabled=not validation.is_valid if validation else False):
        try:
            side = OrderSide.BUY if order_side == "BUY" else OrderSide.SELL
            otype = OrderType.MARKET if order_type == "MARKET" else OrderType.LIMIT

            trade = paper_trader.place_order(
                symbol=st.session_state.symbol,
                side=side,
                quantity=quantity,
                order_type=otype,
                price=limit_price
            )

            if trade:
                st.success(f"Order executed: {order_side} {quantity} {st.session_state.symbol} @ ${limit_price:,.2f}")
                st.rerun()
            else:
                st.error("Order failed")
        except Exception as e:
            st.error(f"Order error: {e}")


# =============================================================================
# PAGE: BACKTESTING
# =============================================================================

def page_backtesting():
    """Backtesting page."""
    st.markdown("### Backtesting")

    # Parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        lookback_days = st.number_input("Lookback (days)", min_value=7, max_value=365, value=30)

    with col2:
        initial_capital = st.number_input("Initial Capital", min_value=1000, value=10000, step=1000)

    with col3:
        strategy = st.selectbox("Strategy", ["LSTM Ensemble", "Momentum", "Mean Reversion"])

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            # Fetch historical data
            data = fetch_market_data(st.session_state.symbol, '1h', limit=lookback_days * 24)

            if not data['success']:
                error_msg = data.get('error', 'Unknown error')
                if 'waiting for data' in error_msg.lower() or 'connecting' in error_msg.lower():
                    st.info(f"🔄 Loading historical data... {error_msg}")
                else:
                    st.warning(f"⚠️ {error_msg}")
                return

            df = data['df']

            # Simple backtest simulation
            equity = [initial_capital]
            trades = []
            position = 0
            entry_price = 0

            for i in range(50, len(df)):
                # Simple RSI strategy
                rsi = calculate_rsi(df['close'].iloc[:i].values)
                price = df['close'].iloc[i]

                if position == 0 and rsi < 30:  # Buy signal
                    position = equity[-1] * 0.95 / price
                    entry_price = price
                elif position > 0 and rsi > 70:  # Sell signal
                    pnl = position * (price - entry_price)
                    equity.append(equity[-1] + pnl)
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': pnl,
                        'pnl_percent': (price - entry_price) / entry_price * 100
                    })
                    position = 0
                else:
                    equity.append(equity[-1])

            # Calculate metrics
            equity_series = pd.Series(equity, index=df['datetime'].iloc[50:50+len(equity)])

            if st.session_state.metrics_calculator:
                metrics = st.session_state.metrics_calculator.calculate(equity_series, trades)

            # Display results
            st.markdown("#### Backtest Results")

            cols = st.columns(4)
            with cols[0]:
                render_metric_card("Total Return", f"{metrics.total_return:.2%}")
            with cols[1]:
                render_metric_card("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
            with cols[2]:
                render_metric_card("Max Drawdown", f"{metrics.max_drawdown:.2%}")
            with cols[3]:
                render_metric_card("Win Rate", f"{metrics.win_rate:.1%}")

            # Equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_series.index,
                y=equity_series.values,
                mode='lines',
                name='Equity',
                line=dict(color='#667eea', width=2)
            ))
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: RISK MANAGEMENT
# =============================================================================

def page_risk_management():
    """Risk management page."""
    st.markdown("### Risk Management")

    paper_trader = st.session_state.paper_trader
    config = load_config()
    risk_config = config.get('risk', {})

    # Risk limits
    st.markdown("#### Risk Limits")

    cols = st.columns(4)

    with cols[0]:
        max_dd = risk_config.get('max_drawdown_percent', 20)
        render_metric_card("Max Drawdown Limit", f"{max_dd}%")

    with cols[1]:
        daily_limit = risk_config.get('daily_loss_limit', 0.05) * 100
        render_metric_card("Daily Loss Limit", f"{daily_limit}%")

    with cols[2]:
        max_pos = risk_config.get('max_position_percent', 0.25) * 100
        render_metric_card("Max Position Size", f"{max_pos}%")

    with cols[3]:
        sector_exp = risk_config.get('max_sector_exposure', 0.40) * 100
        render_metric_card("Max Sector Exposure", f"{sector_exp}%")

    st.markdown("---")

    # Current risk status
    st.markdown("#### Current Risk Status")

    if paper_trader:
        pnl = paper_trader.total_value - paper_trader.initial_capital
        pnl_pct = (pnl / paper_trader.initial_capital) * 100

        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=abs(min(0, pnl_pct)),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Current Drawdown"},
            gauge={
                'axis': {'range': [0, max_dd]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, max_dd * 0.5], 'color': "lightgreen"},
                    {'range': [max_dd * 0.5, max_dd * 0.8], 'color': "yellow"},
                    {'range': [max_dd * 0.8, max_dd], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_dd
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Circuit breaker status
    st.markdown("#### Circuit Breaker")

    validator = st.session_state.order_validator
    if validator:
        if validator.circuit_breaker_active:
            st.error(f"Circuit Breaker ACTIVE: {validator.circuit_breaker_reason}")
            if st.button("Deactivate Circuit Breaker"):
                validator.deactivate_circuit_breaker()
                st.rerun()
        else:
            st.success("Circuit Breaker: Inactive")
            if st.button("Activate Circuit Breaker"):
                validator.activate_circuit_breaker("Manual activation")
                st.rerun()


# =============================================================================
# PAGE: PERFORMANCE
# =============================================================================

def page_performance():
    """Performance metrics page."""
    st.markdown("### Performance Analytics")

    paper_trader = st.session_state.paper_trader

    if paper_trader is None:
        st.warning("Paper trading not initialized")
        return

    trades = paper_trader.get_trades()

    if len(trades) < 2:
        st.info("Need at least 2 trades for performance analysis")
        return

    # Build equity curve
    equity = [paper_trader.initial_capital]
    for t in trades:
        if t.realized_pnl:
            equity.append(equity[-1] + t.realized_pnl)

    equity_series = pd.Series(equity)
    equity_series.index = pd.date_range(end=datetime.now(), periods=len(equity), freq='h')

    # Calculate metrics
    trade_dicts = [{'pnl': t.realized_pnl or 0} for t in trades]

    if st.session_state.metrics_calculator:
        metrics = st.session_state.metrics_calculator.calculate(equity_series, trade_dicts)

        # Display metrics
        st.markdown("#### Key Metrics")

        cols = st.columns(5)

        with cols[0]:
            render_metric_card("Total Return", f"{metrics.total_return:.2%}")
        with cols[1]:
            render_metric_card("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
        with cols[2]:
            render_metric_card("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")
        with cols[3]:
            render_metric_card("Max Drawdown", f"{metrics.max_drawdown:.2%}")
        with cols[4]:
            render_metric_card("Win Rate", f"{metrics.win_rate:.1%}")

        cols2 = st.columns(5)

        with cols2[0]:
            render_metric_card("Total Trades", str(metrics.total_trades))
        with cols2[1]:
            render_metric_card("Winners", str(metrics.winning_trades))
        with cols2[2]:
            render_metric_card("Losers", str(metrics.losing_trades))
        with cols2[3]:
            render_metric_card("Profit Factor", f"{metrics.profit_factor:.2f}")
        with cols2[4]:
            render_metric_card("Avg Trade", f"${metrics.avg_win - abs(metrics.avg_loss):,.2f}")

        st.markdown("---")

        # Equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_series.index,
            y=equity_series.values,
            mode='lines',
            fill='tozeroy',
            name='Equity',
            line=dict(color='#667eea', width=2)
        ))
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE: NEWS & SENTIMENT
# =============================================================================

def page_news_sentiment():
    """News & Sentiment analysis page."""
    st.markdown("### News & Sentiment Analysis")

    db = st.session_state.db
    news_collector = st.session_state.news_collector

    if not AI_AVAILABLE or db is None:
        st.warning("News features not available")
        return

    # News collector status
    col1, col2, col3 = st.columns(3)

    with col1:
        if news_collector and hasattr(news_collector, '_running') and news_collector._running:
            status = "🟢 Active"
            status_class = "positive"
        else:
            status = "🔴 Inactive"
            status_class = "negative"
        render_metric_card("News Collector", status, "", status_class)

    with col2:
        # Count articles in database
        try:
            articles = db.get_news_articles(limit=1000)
            article_count = len(articles)
            render_metric_card("Total Articles", str(article_count))
        except Exception as e:
            logger.error(f"Failed to fetch articles: {e}")
            render_metric_card("Total Articles", "0")
            article_count = 0

    with col3:
        # Get recent articles (last 24h)
        try:
            since = int((datetime.utcnow() - timedelta(hours=24)).timestamp())
            recent = db.get_news_articles(since_timestamp=since, limit=1000)
            render_metric_card("24h Articles", str(len(recent)))
        except Exception as e:
            logger.error(f"Failed to fetch recent articles: {e}")
            render_metric_card("24h Articles", "0")

    st.markdown("---")

    if article_count == 0:
        st.info("""
        **No news articles found in database.**

        News collection requires:
        1. API keys set in `.env` file (`NEWSAPI_KEY`, `ALPHAVANTAGE_KEY`)
        2. News collection enabled in `config.yaml`
        3. Analysis engine running (`python run_analysis.py`)

        Once configured, news will be collected automatically every 30 minutes.
        """)
        return

    # Recent articles
    st.markdown("### Recent Articles")

    try:
        # Get recent articles
        articles = db.get_news_articles(
            symbol=st.session_state.symbol.split('/')[0],
            limit=20
        )

        if not articles:
            st.info(f"No articles found for {st.session_state.symbol}")
        else:
            for article in articles[:10]:  # Show top 10
                # Determine sentiment color
                sentiment = article.get('sentiment_score', 0)
                if sentiment > 0.2:
                    sentiment_color = "#28a745"  # Green (bullish)
                    sentiment_label = "📈 Bullish"
                elif sentiment < -0.2:
                    sentiment_color = "#dc3545"  # Red (bearish)
                    sentiment_label = "📉 Bearish"
                else:
                    sentiment_color = "#6c757d"  # Gray (neutral)
                    sentiment_label = "➖ Neutral"

                # Format timestamp
                try:
                    dt = datetime.fromisoformat(article['datetime'])
                    time_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    time_str = "Unknown"

                # Render article card
                st.markdown(f"""
                <div style="background: white; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; border-left: 4px solid {sentiment_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="color: {sentiment_color}; font-weight: 600;">{sentiment_label}</span>
                        <span style="color: #6c757d; font-size: 0.85rem;">{article['source']} • {time_str}</span>
                    </div>
                    <div style="font-weight: 600; margin-bottom: 0.3rem;">{article['title']}</div>
                    <div style="color: #6c757d; font-size: 0.9rem;">{article.get('description', '')[:200]}...</div>
                    {f'<div style="margin-top: 0.5rem;"><a href="{article["url"]}" target="_blank" style="color: #667eea; text-decoration: none;">Read more →</a></div>' if article.get('url') else ''}
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading articles: {e}")

    st.markdown("---")

    # Sentiment trend chart
    st.markdown("### Sentiment Trend")

    try:
        # Get sentiment features for recent candles
        data = fetch_market_data(st.session_state.symbol, limit=48)  # Last 48 hours
        if data['success'] and not data['df'].empty:
            df = data['df']

            sentiment_data = []
            for _, row in df.iterrows():
                timestamp = int(row.get('timestamp', 0))
                if timestamp:
                    features = db.get_sentiment_features(timestamp)
                    if features:
                        sentiment_data.append({
                            'datetime': row['datetime'],
                            'sentiment_1h': features.get('sentiment_1h', 0),
                            'sentiment_6h': features.get('sentiment_6h', 0),
                            'news_volume': features.get('news_volume_1h', 0)
                        })

            if sentiment_data:
                df_sentiment = pd.DataFrame(sentiment_data)

                # Create chart
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )

                # Sentiment line
                fig.add_trace(
                    go.Scatter(
                        x=df_sentiment['datetime'],
                        y=df_sentiment['sentiment_1h'],
                        mode='lines',
                        name='1H Sentiment',
                        line=dict(color='#667eea', width=2)
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df_sentiment['datetime'],
                        y=df_sentiment['sentiment_6h'],
                        mode='lines',
                        name='6H Sentiment',
                        line=dict(color='#764ba2', width=2, dash='dash')
                    ),
                    row=1, col=1
                )

                # News volume
                fig.add_trace(
                    go.Bar(
                        x=df_sentiment['datetime'],
                        y=df_sentiment['news_volume'],
                        name='News Volume',
                        marker_color='#667eea',
                        opacity=0.5
                    ),
                    row=2, col=1
                )

                fig.update_layout(
                    title="Sentiment & News Volume",
                    height=500,
                    showlegend=True
                )

                fig.update_yaxes(title_text="Sentiment", row=1, col=1)
                fig.update_yaxes(title_text="Articles", row=2, col=1)

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available yet. Wait for news collection to populate features.")
    except Exception as e:
        st.error(f"Error loading sentiment trend: {e}")


# =============================================================================
# PAGE: SETTINGS
# =============================================================================

def page_settings():
    """Settings page."""
    st.markdown("### Settings")

    config = load_config()

    tabs = st.tabs(["Trading", "Risk", "Notifications", "System"])

    with tabs[0]:  # Trading
        st.markdown("#### Trading Settings")

        col1, col2 = st.columns(2)

        with col1:
            new_symbol = st.text_input("Default Symbol", value=config.get('data', {}).get('symbol', 'BTC/USDT'))
            exchange_options = ['binance', 'alpaca', 'coinbase', 'bybit', 'kraken']
            current_exchange = config.get('data', {}).get('exchange', 'binance')
            new_exchange = st.selectbox("Exchange", exchange_options,
                                        index=exchange_options.index(current_exchange) if current_exchange in exchange_options else 0)
            all_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
            current_interval = config.get('data', {}).get('interval', '1h')
            new_interval = st.selectbox("Timeframe", all_timeframes,
                                        index=all_timeframes.index(current_interval) if current_interval in all_timeframes else 5)

        with col2:
            new_capital = st.number_input("Initial Capital", value=config.get('portfolio', {}).get('initial_capital', 10000))
            new_position_method = st.selectbox("Position Sizing", ['equal_weight', 'risk_parity', 'kelly', 'volatility_target'])

    with tabs[1]:  # Risk
        st.markdown("#### Risk Settings")

        col1, col2 = st.columns(2)

        with col1:
            max_dd = st.slider("Max Drawdown %", 5, 50, int(config.get('risk', {}).get('max_drawdown_percent', 20)))
            daily_limit = st.slider("Daily Loss Limit %", 1, 20, int(config.get('risk', {}).get('daily_loss_limit', 0.05) * 100))

        with col2:
            max_pos = st.slider("Max Position Size %", 5, 50, int(config.get('risk', {}).get('max_position_percent', 0.25) * 100))
            sector_exp = st.slider("Max Sector Exposure %", 10, 80, int(config.get('risk', {}).get('max_sector_exposure', 0.40) * 100))

    with tabs[2]:  # Notifications
        st.markdown("#### Notification Settings")

        desktop = st.checkbox("Desktop Notifications", value=config.get('notifications', {}).get('desktop', {}).get('enabled', True))
        sound = st.checkbox("Sound Alerts", value=config.get('notifications', {}).get('desktop', {}).get('sound', True))
        telegram = st.checkbox("Telegram", value=config.get('notifications', {}).get('telegram', {}).get('enabled', False))

        if telegram:
            bot_token = st.text_input("Bot Token", type="password")
            chat_id = st.text_input("Chat ID")

    with tabs[3]:  # System
        st.markdown("#### System Settings")

        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 1, 60, 5)

        st.markdown("#### Engine Control")

        running, pid = is_engine_running()

        if running:
            st.success(f"Engine running (PID: {pid})")
            if st.button("Stop Engine"):
                try:
                    os.kill(pid, signal.SIGTERM)
                    PID_FILE.unlink(missing_ok=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to stop: {e}")
        else:
            st.warning("Engine stopped")
            if st.button("Start Engine"):
                try:
                    venv_py = ROOT / "venv" / "bin" / "python"
                    py = str(venv_py) if venv_py.exists() else "python3"
                    subprocess.Popen([py, str(ROOT / "run_analysis.py")],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start: {e}")

    if st.button("Save Settings", type="primary"):
        st.success("Settings saved!")


# =============================================================================
# MAIN NAVIGATION
# =============================================================================

def main():
    """Main application."""
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## AI Trade Bot")
        st.markdown("---")

        # Page selection
        pages = {
            "📊 Dashboard": "Dashboard",
            "💱 Forex Markets": "Forex Markets",
            "💼 Portfolio": "Portfolio",
            "📝 Paper Trade": "Paper Trading",
            "📈 Backtest": "Backtesting",
            "🛡️ Risk": "Risk Management",
            "📉 Performance": "Performance",
            "📰 News & Sentiment": "News & Sentiment",
            "⚙️ Settings": "Settings"
        }

        for label, page_name in pages.items():
            if st.button(label, use_container_width=True,
                        type="primary" if st.session_state.page == page_name else "secondary"):
                st.session_state.page = page_name
                st.rerun()

        st.markdown("---")

        # Quick settings
        st.markdown("### Quick Settings")

        exchanges = ['binance', 'alpaca', 'coinbase', 'bybit', 'kraken']
        new_exchange = st.selectbox("Exchange", exchanges,
                                    index=exchanges.index(st.session_state.exchange) if st.session_state.exchange in exchanges else 0)
        if new_exchange != st.session_state.exchange:
            st.session_state.exchange = new_exchange
            # Provider will use new exchange on next subscription
            st.rerun()

        # Symbol list changes based on exchange
        if st.session_state.exchange == 'alpaca':
            # US Stocks and popular assets
            symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'SPY', 'QQQ', 'BTC/USD', 'ETH/USD']
        else:
            # Crypto pairs for other exchanges
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT']

        new_symbol = st.selectbox("Symbol", symbols,
                                  index=symbols.index(st.session_state.symbol) if st.session_state.symbol in symbols else 0)
        if new_symbol != st.session_state.symbol:
            old_symbol = st.session_state.symbol
            st.session_state.symbol = new_symbol

            # Subscribe to new symbol
            provider = st.session_state.data_provider
            if provider and provider.is_running:
                logger.info(f"Symbol changed: {old_symbol} -> {new_symbol}")
                provider.subscribe(new_symbol, exchange=st.session_state.exchange, interval=st.session_state.timeframe)
            st.rerun()

        # All supported timeframes from 1 minute to 1 year
        timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        new_tf = st.selectbox("Timeframe", timeframes,
                             index=timeframes.index(st.session_state.timeframe) if st.session_state.timeframe in timeframes else 5)
        if new_tf != st.session_state.timeframe:
            # Timeframe changed - resubscribe with new interval
            old_tf = st.session_state.timeframe
            st.session_state.timeframe = new_tf

            provider = st.session_state.data_provider
            if provider and provider.is_running:
                logger.info(f"Timeframe changed: {old_tf} -> {new_tf}")
                # subscribe() handles interval change, clears buffer, and reconnects
                for symbol in st.session_state.tracked_symbols:
                    provider.subscribe(symbol, exchange=st.session_state.exchange, interval=new_tf)
            st.rerun()

        st.markdown("---")

        # Auto-refresh
        auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh

        if st.button("🔄 Refresh Now"):
            st.rerun()

    # Render selected page
    page_functions = {
        "Dashboard": page_dashboard,
        "Forex Markets": page_forex_markets,
        "Portfolio": page_portfolio,
        "Paper Trading": page_paper_trading,
        "Backtesting": page_backtesting,
        "Risk Management": page_risk_management,
        "Performance": page_performance,
        "News & Sentiment": page_news_sentiment,
        "Settings": page_settings,
    }

    # Auto-refresh (non-flickering) - runs BEFORE page rendering to prevent flicker
    if st.session_state.auto_refresh:
        # Convert seconds to milliseconds, runs in background without blocking
        st_autorefresh(interval=st.session_state.refresh_interval * 1000, key="data_refresh")

    page_func = page_functions.get(st.session_state.page, page_dashboard)
    page_func()


if __name__ == "__main__":
    main()
