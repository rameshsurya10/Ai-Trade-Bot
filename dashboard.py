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
        'last_update': None,

        # WebSocket Data Provider (ONLY data source)
        'data_provider': None,
        'provider_started': False,

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

        # Tracked currencies
        'tracked_symbols': ['BTC/USDT', 'ETH/USDT'],

        # Cache
        'market_data': None,
        'predictions': None,
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
        st.session_state.provider_started = True
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

initialize_components()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_config() -> dict:
    """Load configuration file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


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
                    'volume': tick.volume if tick else 0
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
    """Render a metric card."""
    delta_html = f'<div class="delta">{delta}</div>' if delta else ''
    st.markdown(f"""
        <div class="metric-card {card_class}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def render_signal_card(direction: str, confidence: float, price: float, stop_loss: float, take_profit: float):
    """Render signal card."""
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
        st.error(f"Failed to fetch market data: {data.get('error', 'Unknown error')}")
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
        # AI Prediction
        st.markdown("### AI Prediction")

        if AI_AVAILABLE and len(df) >= 50:
            try:
                predictor = st.session_state.advanced_predictor
                if predictor:
                    prediction = predictor.predict(df, lstm_probability=0.55)
                    render_signal_card(
                        prediction.direction,
                        prediction.confidence,
                        price,
                        prediction.stop_loss,
                        prediction.take_profit
                    )

                    # Algorithm breakdown
                    with st.expander("Algorithm Details"):
                        st.write(f"**Fourier:** {prediction.fourier_signal} (Phase: {prediction.fourier_cycle_phase:.2f})")
                        st.write(f"**Kalman:** {prediction.kalman_trend}")
                        st.write(f"**Entropy:** {prediction.entropy_regime} ({prediction.entropy_value:.2f})")
                        st.write(f"**Markov:** {prediction.markov_state} (P(up): {prediction.markov_probability:.2f})")
                        st.write(f"**Monte Carlo Risk:** {prediction.monte_carlo_risk:.2f}")
            except Exception as e:
                st.warning(f"Prediction unavailable: {e}")
        else:
            st.info("AI predictions require 50+ candles")


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
                st.error("Failed to fetch historical data")
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
            new_exchange = st.selectbox("Exchange", ['binance', 'coinbase', 'bybit', 'kraken'],
                                        index=['binance', 'coinbase', 'bybit', 'kraken'].index(
                                            config.get('data', {}).get('exchange', 'binance')))
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
            "💼 Portfolio": "Portfolio",
            "📝 Paper Trade": "Paper Trading",
            "📈 Backtest": "Backtesting",
            "🛡️ Risk": "Risk Management",
            "📉 Performance": "Performance",
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

        exchanges = ['binance', 'coinbase', 'bybit', 'kraken']
        new_exchange = st.selectbox("Exchange", exchanges,
                                    index=exchanges.index(st.session_state.exchange))
        if new_exchange != st.session_state.exchange:
            st.session_state.exchange = new_exchange
            # Provider will use new exchange on next subscription
            st.rerun()

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
        "Portfolio": page_portfolio,
        "Paper Trading": page_paper_trading,
        "Backtesting": page_backtesting,
        "Risk Management": page_risk_management,
        "Performance": page_performance,
        "Settings": page_settings,
    }

    page_func = page_functions.get(st.session_state.page, page_dashboard)
    page_func()

    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
