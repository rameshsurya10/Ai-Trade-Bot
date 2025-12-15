#!/usr/bin/env python3
"""
AI Trade Bot - Professional Trading Dashboard
==============================================

Complete trading dashboard with controls, charts, and signals.

Usage:
    streamlit run dashboard.py
    ./start.sh
"""

import html
import os
import sys
import subprocess
import time
import logging
import warnings
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict

# Suppress warnings
logging.getLogger('tornado.access').setLevel(logging.ERROR)
logging.getLogger('tornado.application').setLevel(logging.ERROR)
logging.getLogger('tornado.general').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import yaml

try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Required: pip install streamlit plotly")
    sys.exit(1)


# =============================================================================
# CONFIGURATION CONSTANTS (Previously hardcoded values)
# =============================================================================

# Indicator periods - customize these for different trading strategies
INDICATOR_CONFIG = {
    'sma_fast': 20,          # Fast SMA period
    'sma_slow': 50,          # Slow SMA period
    'ema_fast': 12,          # Fast EMA period (for MACD)
    'ema_slow': 26,          # Slow EMA period (for MACD)
    'macd_signal': 9,        # MACD signal line period
    'rsi_period': 14,        # RSI lookback period
    'bb_period': 20,         # Bollinger Bands period
    'bb_std': 2,             # Bollinger Bands standard deviations
    'volume_sma': 20,        # Volume SMA period
}

# RSI thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Chart configuration
CHART_CONFIG = {
    'height': 620,
    'row_heights': [0.5, 0.15, 0.15, 0.2],
    'vertical_spacing': 0.025,
}

# Colors - centralized color scheme
COLORS = {
    'bullish': '#10b981',
    'bearish': '#ef4444',
    'sma_fast': '#f59e0b',
    'sma_slow': '#3b82f6',
    'rsi': '#8b5cf6',
    'macd': '#3b82f6',
    'macd_signal': '#f59e0b',
    'bb_fill': 'rgba(139,92,246,0.05)',
    'bb_line': 'rgba(139,92,246,0.3)',
    'bg_dark': '#06080d',
    'grid': 'rgba(30,42,58,0.5)',
    'text': '#94a3b8',
}

# Data limits
DATA_LIMITS = {
    'candles_default': 200,
    'signals_default': 20,
    'log_tail_chars': 2000,
}

# Signal confidence thresholds
CONFIDENCE_HIGH = 0.65

# Pattern detection thresholds
PATTERN_CONFIG = {
    'doji_body_ratio': 0.1,
    'wick_body_ratio': 2,
    'wick_upper_limit': 0.5,
    'engulfing_ratio': 1.5,
}

# Training defaults
TRAINING_DEFAULTS = {
    'days_options': [30, 90, 180, 365, 730],
    'days_default_index': 3,
    'epochs_min': 50,
    'epochs_max': 200,
    'epochs_default': 100,
}

# Analysis minimum data requirement
MIN_CANDLES_FOR_ANALYSIS = 100
MIN_CANDLES_FOR_TREND = 50

# RSI scale
RSI_MIN = 0
RSI_MAX = 100
RSI_MIDPOINT = 50

# Process timing (seconds)
PROCESS_DELAYS = {
    'after_stop': 1,
    'after_start': 2,
}

# Chart line widths
LINE_WIDTHS = {
    'thin': 1,
    'medium': 1.5,
    'thick': 2,
}


# =============================================================================
# PROFESSIONAL CSS THEME
# =============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-dark: #06080d;
    --bg-card: #0c1017;
    --bg-card-hover: #111820;
    --bg-elevated: #141c28;
    --bg-input: #0a0f16;
    --border: #1e2a3a;
    --border-light: #2a3a4d;
    --text-primary: #ffffff;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-green: #10b981;
    --accent-green-glow: rgba(16, 185, 129, 0.2);
    --accent-red: #ef4444;
    --accent-red-glow: rgba(239, 68, 68, 0.2);
    --accent-blue: #3b82f6;
    --accent-blue-glow: rgba(59, 130, 246, 0.15);
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    --gradient-blue: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    --gradient-green: linear-gradient(135deg, #10b981 0%, #059669 100%);
    --gradient-red: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: var(--bg-dark) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }

.main .block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 1920px !important;
}

/* ========== HEADER ========== */
.main-header {
    background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-dark) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.header-logo {
    width: 48px;
    height: 48px;
    background: var(--gradient-blue);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 16px var(--accent-blue-glow);
}

.header-text h1 {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.02em;
}

.header-text p {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin: 0.15rem 0 0;
}

.status-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.25rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.status-live {
    background: var(--accent-green-glow);
    color: var(--accent-green);
    border: 1px solid var(--accent-green);
    box-shadow: 0 0 20px var(--accent-green-glow);
}

.status-offline {
    background: rgba(239, 68, 68, 0.1);
    color: var(--accent-red);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background: currentColor;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.9); }
}

/* ========== METRIC CARDS ========== */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1rem;
    margin-bottom: 1.25rem;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient-blue);
    opacity: 0;
    transition: opacity 0.25s ease;
}

.metric-card:hover {
    border-color: var(--border-light);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.metric-card:hover::before {
    opacity: 1;
}

.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

.metric-delta {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.35rem;
}

.delta-up { color: var(--accent-green); }
.delta-down { color: var(--accent-red); }

/* ========== TABS ========== */
.stTabs {
    background: transparent;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 6px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: var(--text-muted);
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0.7rem 1.25rem;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg-elevated);
    color: var(--text-secondary);
}

.stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: white !important;
    font-weight: 600;
    box-shadow: 0 4px 12px var(--accent-blue-glow);
}

/* ========== SECTION PANELS ========== */
.panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.panel-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}

.panel-icon {
    width: 36px;
    height: 36px;
    background: var(--bg-elevated);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}

.panel-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}

/* ========== BUTTONS ========== */
.stButton > button {
    background: var(--gradient-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.85rem 1.5rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 12px var(--accent-blue-glow) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px var(--accent-blue-glow) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ========== INPUTS ========== */
.stSelectbox > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

.stSelectbox > div > div:focus-within {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px var(--accent-blue-glow) !important;
}

.stSlider > div > div > div {
    background: var(--accent-blue) !important;
}

.stSlider [data-baseweb="slider"] {
    margin-top: 0.5rem;
}

/* ========== STATUS BOX ========== */
.status-box {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.status-running {
    background: var(--accent-green-glow);
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.status-stopped {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.status-box-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}

.status-box-text {
    flex: 1;
}

.status-box-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-primary);
}

.status-box-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.1rem;
}

/* ========== SIGNAL CARDS ========== */
.signal-item {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: all 0.2s ease;
}

.signal-item:hover {
    border-color: var(--border-light);
    transform: translateX(4px);
}

.signal-item.buy {
    border-left: 4px solid var(--accent-green);
}

.signal-item.sell {
    border-left: 4px solid var(--accent-red);
}

.signal-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
}

.signal-icon.buy {
    background: var(--accent-green-glow);
}

.signal-icon.sell {
    background: var(--accent-red-glow);
}

.signal-content {
    flex: 1;
}

.signal-type {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text-primary);
}

.signal-time {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
}

.signal-price {
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1rem;
    color: var(--text-primary);
}

.signal-conf {
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
}

.conf-high {
    background: var(--accent-green-glow);
    color: var(--accent-green);
}

.conf-med {
    background: rgba(245, 158, 11, 0.15);
    color: var(--accent-amber);
}

/* ========== PATTERN CARDS ========== */
.pattern-item {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.2s ease;
}

.pattern-item:hover {
    border-color: var(--border-light);
}

.pattern-info h4 {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}

.pattern-info p {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin: 0.25rem 0 0;
}

.pattern-tag {
    padding: 0.4rem 0.85rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.tag-bullish {
    background: var(--accent-green-glow);
    color: var(--accent-green);
}

.tag-bearish {
    background: var(--accent-red-glow);
    color: var(--accent-red);
}

.tag-neutral {
    background: var(--bg-card);
    color: var(--text-muted);
}

/* ========== INDICATOR BADGES ========== */
.indicator-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}

.indicator-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 0.9rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
}

.badge-bullish {
    background: var(--accent-green-glow);
    color: var(--accent-green);
}

.badge-bearish {
    background: var(--accent-red-glow);
    color: var(--accent-red);
}

.badge-neutral {
    background: var(--bg-elevated);
    color: var(--text-secondary);
    border: 1px solid var(--border);
}

/* ========== LOGS ========== */
.log-viewer {
    background: var(--bg-dark);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    max-height: 250px;
    overflow-y: auto;
    line-height: 1.6;
}

/* ========== FOOTER ========== */
.footer {
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    text-align: center;
}

.footer-warning {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: var(--accent-amber);
    font-size: 0.85rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.footer-note {
    color: var(--text-muted);
    font-size: 0.8rem;
    margin-top: 0.75rem;
}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-light); }

/* ========== STREAMLIT OVERRIDES ========== */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}

div[data-testid="stExpander"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
}

div[data-testid="stExpander"] summary {
    font-weight: 600;
    color: var(--text-primary);
}

.stMarkdown a {
    color: var(--accent-blue);
}
</style>
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_config() -> dict:
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


@contextmanager
def get_db_connection():
    config = load_config()
    db_path = PROJECT_ROOT / config.get('database', {}).get('path', 'data/trading.db')
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def is_analysis_running():
    pid_file = PROJECT_ROOT / "data" / ".analysis.pid"
    if not pid_file.exists():
        return False, None
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True, pid
    except (ProcessLookupError, ValueError, PermissionError):
        return False, None


def get_stats() -> dict:
    config = load_config()
    db_path = PROJECT_ROOT / config.get('database', {}).get('path', 'data/trading.db')
    if not db_path.exists():
        return {'candles': 0, 'signals': 0, 'price': None}
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM candles")
            candles = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM signals")
            signals = cursor.fetchone()[0]
            cursor.execute("SELECT close FROM candles ORDER BY timestamp DESC LIMIT 1")
            latest = cursor.fetchone()
            return {'candles': candles, 'signals': signals, 'price': latest[0] if latest else None}
    except Exception:
        return {'candles': 0, 'signals': 0, 'price': None}


def run_script(script: str, args: list = None) -> dict:
    try:
        venv_py = PROJECT_ROOT / "venv" / "bin" / "python"
        py = str(venv_py) if venv_py.exists() else "python3"
        cmd = [py, str(PROJECT_ROOT / script)]
        if args:
            cmd.extend([str(a) for a in args])
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300)
        return {'success': result.returncode == 0, 'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as e:
        return {'success': False, 'stdout': '', 'stderr': str(e)}


def load_candles(limit: int = None) -> pd.DataFrame:
    if limit is None:
        limit = DATA_LIMITS['candles_default']
    config = load_config()
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query('''
                SELECT timestamp, datetime, open, high, low, close, volume
                FROM candles WHERE symbol = ? AND interval = ?
                ORDER BY timestamp DESC LIMIT ?
            ''', conn, params=(config['data']['symbol'], config['data']['interval'], limit))
        if not df.empty:
            df = df.sort_values('timestamp')
            df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception:
        return pd.DataFrame()


def load_signals(limit: int = None) -> pd.DataFrame:
    if limit is None:
        limit = DATA_LIMITS['signals_default']
    try:
        with get_db_connection() as conn:
            return pd.read_sql_query('''
                SELECT datetime, signal, confidence, price FROM signals
                ORDER BY timestamp DESC LIMIT ?
            ''', conn, params=(limit,))
    except Exception:
        return pd.DataFrame()


# =============================================================================
# TECHNICAL ANALYSIS
# =============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Moving Averages
    df['sma_fast'] = df['close'].rolling(INDICATOR_CONFIG['sma_fast']).mean()
    df['sma_slow'] = df['close'].rolling(INDICATOR_CONFIG['sma_slow']).mean()

    # MACD
    ema_fast = df['close'].ewm(span=INDICATOR_CONFIG['ema_fast'], adjust=False).mean()
    ema_slow = df['close'].ewm(span=INDICATOR_CONFIG['ema_slow'], adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=INDICATOR_CONFIG['macd_signal'], adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(INDICATOR_CONFIG['rsi_period']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(INDICATOR_CONFIG['rsi_period']).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(INDICATOR_CONFIG['bb_period']).mean()
    bb_std = df['close'].rolling(INDICATOR_CONFIG['bb_period']).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std * INDICATOR_CONFIG['bb_std'])
    df['bb_lower'] = df['bb_mid'] - (bb_std * INDICATOR_CONFIG['bb_std'])

    # Volume
    df['vol_sma'] = df['volume'].rolling(INDICATOR_CONFIG['volume_sma']).mean()

    return df


def detect_patterns(df: pd.DataFrame) -> List[Dict]:
    patterns = []
    if len(df) < 5:
        return patterns

    last, prev = df.iloc[-1], df.iloc[-2]
    body = abs(last['close'] - last['open'])
    upper_wick = last['high'] - max(last['close'], last['open'])
    lower_wick = min(last['close'], last['open']) - last['low']
    total_range = last['high'] - last['low']

    if total_range == 0:
        return patterns

    body_ratio = body / total_range

    # Doji pattern
    if body_ratio < PATTERN_CONFIG['doji_body_ratio']:
        patterns.append({'name': 'Doji', 'type': 'neutral', 'desc': 'Market indecision - potential reversal'})

    # Hammer / Hanging Man
    if lower_wick > body * PATTERN_CONFIG['wick_body_ratio'] and upper_wick < body * PATTERN_CONFIG['wick_upper_limit']:
        if last['close'] > prev['close']:
            patterns.append({'name': 'Hammer', 'type': 'bullish', 'desc': 'Strong bullish reversal signal'})
        else:
            patterns.append({'name': 'Hanging Man', 'type': 'bearish', 'desc': 'Bearish reversal warning'})

    # Shooting Star / Inverted Hammer
    if upper_wick > body * PATTERN_CONFIG['wick_body_ratio'] and lower_wick < body * PATTERN_CONFIG['wick_upper_limit']:
        if last['close'] < prev['close']:
            patterns.append({'name': 'Shooting Star', 'type': 'bearish', 'desc': 'Bearish reversal at resistance'})
        else:
            patterns.append({'name': 'Inverted Hammer', 'type': 'bullish', 'desc': 'Potential bullish reversal'})

    # Engulfing patterns
    prev_body = abs(prev['close'] - prev['open'])
    if body > prev_body * PATTERN_CONFIG['engulfing_ratio']:
        if last['close'] > last['open'] and prev['close'] < prev['open']:
            patterns.append({'name': 'Bullish Engulfing', 'type': 'bullish', 'desc': 'Strong bullish reversal pattern'})
        elif last['close'] < last['open'] and prev['close'] > prev['open']:
            patterns.append({'name': 'Bearish Engulfing', 'type': 'bearish', 'desc': 'Strong bearish reversal pattern'})

    return patterns


def get_trend(df: pd.DataFrame) -> Dict:
    if len(df) < MIN_CANDLES_FOR_TREND:
        return {'trend': 'N/A', 'score': 0, 'rsi': RSI_MIDPOINT}

    last = df.iloc[-1]
    score = 0

    # Price vs Fast SMA
    if pd.notna(last.get('sma_fast')) and last['close'] > last['sma_fast']:
        score += 1
    elif pd.notna(last.get('sma_fast')):
        score -= 1

    # Price vs Slow SMA
    if pd.notna(last.get('sma_slow')) and last['close'] > last['sma_slow']:
        score += 1
    elif pd.notna(last.get('sma_slow')):
        score -= 1

    # SMA crossover
    if pd.notna(last.get('sma_fast')) and pd.notna(last.get('sma_slow')) and last['sma_fast'] > last['sma_slow']:
        score += 1
    elif pd.notna(last.get('sma_fast')) and pd.notna(last.get('sma_slow')):
        score -= 1

    # MACD
    if pd.notna(last.get('macd')) and pd.notna(last.get('macd_signal')) and last['macd'] > last['macd_signal']:
        score += 1
    elif pd.notna(last.get('macd')):
        score -= 1

    # RSI
    rsi = last.get('rsi', RSI_MIDPOINT)
    if pd.notna(rsi):
        if rsi > RSI_MIDPOINT:
            score += 0.5
        else:
            score -= 0.5

    # Determine trend label
    if score >= 3:
        trend = 'Strong Buy'
    elif score >= 1:
        trend = 'Bullish'
    elif score <= -3:
        trend = 'Strong Sell'
    elif score <= -1:
        trend = 'Bearish'
    else:
        trend = 'Neutral'

    return {'trend': trend, 'score': score, 'rsi': rsi if pd.notna(rsi) else RSI_MIDPOINT}


# =============================================================================
# CHART
# =============================================================================

def create_chart(df: pd.DataFrame) -> go.Figure:
    df = calculate_indicators(df)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=CHART_CONFIG['vertical_spacing'],
        row_heights=CHART_CONFIG['row_heights']
    )

    green = COLORS['bullish']
    red = COLORS['bearish']

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing=dict(line=dict(color=green, width=1), fillcolor=green),
        decreasing=dict(line=dict(color=red, width=1), fillcolor=red),
        showlegend=False, name='Price'
    ), row=1, col=1)

    # SMAs
    sma_fast_label = f"SMA {INDICATOR_CONFIG['sma_fast']}"
    sma_slow_label = f"SMA {INDICATOR_CONFIG['sma_slow']}"
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['sma_fast'], name=sma_fast_label,
        line=dict(color=COLORS['sma_fast'], width=LINE_WIDTHS['medium'])
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['sma_slow'], name=sma_slow_label,
        line=dict(color=COLORS['sma_slow'], width=LINE_WIDTHS['medium'])
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['bb_upper'],
        line=dict(color=COLORS['bb_line'], width=LINE_WIDTHS['thin']),
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['bb_lower'],
        line=dict(color=COLORS['bb_line'], width=LINE_WIDTHS['thin']),
        fill='tonexty', fillcolor=COLORS['bb_fill'],
        showlegend=False
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['rsi'], name='RSI',
        line=dict(color=COLORS['rsi'], width=LINE_WIDTHS['thick'])
    ), row=2, col=1)
    fig.add_hrect(y0=RSI_OVERBOUGHT, y1=RSI_MAX, fillcolor='rgba(239,68,68,0.1)', line_width=0, row=2, col=1)
    fig.add_hrect(y0=RSI_MIN, y1=RSI_OVERSOLD, fillcolor='rgba(16,185,129,0.1)', line_width=0, row=2, col=1)
    fig.add_hline(y=RSI_OVERBOUGHT, line_dash="dot", line_color="rgba(239,68,68,0.5)", line_width=LINE_WIDTHS['thin'], row=2, col=1)
    fig.add_hline(y=RSI_OVERSOLD, line_dash="dot", line_color="rgba(16,185,129,0.5)", line_width=LINE_WIDTHS['thin'], row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['macd'], name='MACD',
        line=dict(color=COLORS['macd'], width=LINE_WIDTHS['thick'])
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['macd_signal'], name='Signal',
        line=dict(color=COLORS['macd_signal'], width=LINE_WIDTHS['thick'])
    ), row=3, col=1)
    hist_colors = [green if v >= 0 else red for v in df['macd_hist'].fillna(0)]
    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['macd_hist'],
        marker_color=hist_colors, opacity=0.7, showlegend=False
    ), row=3, col=1)

    # Volume
    vol_colors = [green if r['close'] >= r['open'] else red for _, r in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df['datetime'], y=df['volume'],
        marker_color=vol_colors, opacity=0.6, showlegend=False
    ), row=4, col=1)

    # Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['bg_dark'],
        plot_bgcolor=COLORS['bg_dark'],
        height=CHART_CONFIG['height'],
        margin=dict(l=60, r=20, t=10, b=20),
        font=dict(family='Inter', color=COLORS['text'], size=11),
        legend=dict(
            orientation='h', y=1.02, x=0.5, xanchor='center',
            bgcolor='rgba(0,0,0,0)', font=dict(size=11)
        ),
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )

    # Grid styling
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLORS['grid'], showgrid=True, zeroline=False, row=i, col=1)
        fig.update_yaxes(gridcolor=COLORS['grid'], showgrid=True, zeroline=False, row=i, col=1)

    fig.update_yaxes(title_text='Price', title_font_size=10, row=1, col=1)
    fig.update_yaxes(title_text='RSI', title_font_size=10, range=[RSI_MIN, RSI_MAX], row=2, col=1)
    fig.update_yaxes(title_text='MACD', title_font_size=10, row=3, col=1)
    fig.update_yaxes(title_text='Volume', title_font_size=10, row=4, col=1)

    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(page_title="AI Trade Bot", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    config = load_config()
    running, pid = is_analysis_running()
    stats = get_stats()
    symbol = config.get('data', {}).get('symbol', 'BTC-USD')

    # ===== HEADER =====
    status_html = f'<span class="pulse-dot"></span>LIVE' if running else 'OFFLINE'
    status_class = 'status-live' if running else 'status-offline'

    st.markdown(f'''
        <div class="main-header">
            <div class="header-left">
                <div class="header-logo">📊</div>
                <div class="header-text">
                    <h1>AI Trade Bot</h1>
                    <p>{symbol} • Real-time Analysis</p>
                </div>
            </div>
            <div class="status-pill {status_class}">{status_html}</div>
        </div>
    ''', unsafe_allow_html=True)

    # ===== METRICS =====
    df = load_candles(DATA_LIMITS['candles_default'])
    if not df.empty:
        df = calculate_indicators(df)
        last, prev = df.iloc[-1], df.iloc[-2]
        price = last['close']
        change = price - prev['close']
        change_pct = (change / prev['close']) * 100
        trend_data = get_trend(df)
        rsi = trend_data.get('rsi', RSI_MIDPOINT)
    else:
        price, change, change_pct, rsi = 0, 0, 0, RSI_MIDPOINT
        trend_data = {'trend': 'N/A'}

    delta_class = 'delta-up' if change >= 0 else 'delta-down'
    delta_sign = '+' if change >= 0 else ''

    st.markdown(f'''
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Price</div>
                <div class="metric-value">${price:,.2f}</div>
                <div class="metric-delta {delta_class}">{delta_sign}{change:,.2f} ({delta_sign}{change_pct:.2f}%)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Trend</div>
                <div class="metric-value">{trend_data['trend']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">RSI ({INDICATOR_CONFIG['rsi_period']})</div>
                <div class="metric-value">{rsi:.1f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Candles</div>
                <div class="metric-value">{stats['candles']:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Signals</div>
                <div class="metric-value">{stats['signals']:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Status</div>
                <div class="metric-value">{'Active' if running else 'Idle'}</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # ===== TABS =====
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['📈 Chart', '⚡ Controls', '📡 Signals', '🔍 Patterns', '⚙️ Settings'])

    # ----- TAB 1: CHART -----
    with tab1:
        if df.empty:
            st.warning("No data available. Go to Controls tab to download data.")
        else:
            fig = create_chart(df)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

            # Indicators
            last = df.iloc[-1]
            ma_bull = last['sma_fast'] > last['sma_slow'] if pd.notna(last.get('sma_fast')) and pd.notna(last.get('sma_slow')) else False
            macd_bull = last['macd'] > last['macd_signal'] if pd.notna(last.get('macd')) else False
            rsi_val = last.get('rsi', RSI_MIDPOINT)

            ma_cls = 'badge-bullish' if ma_bull else 'badge-bearish'
            macd_cls = 'badge-bullish' if macd_bull else 'badge-bearish'
            rsi_cls = 'badge-bearish' if rsi_val > RSI_OVERBOUGHT else 'badge-bullish' if rsi_val < RSI_OVERSOLD else 'badge-neutral'
            rsi_txt = 'Overbought' if rsi_val > RSI_OVERBOUGHT else 'Oversold' if rsi_val < RSI_OVERSOLD else 'Neutral'

            st.markdown(f'''
                <div class="indicator-row">
                    <div class="indicator-badge {ma_cls}">MA: {'Bullish' if ma_bull else 'Bearish'}</div>
                    <div class="indicator-badge {macd_cls}">MACD: {'Bullish' if macd_bull else 'Bearish'}</div>
                    <div class="indicator-badge {rsi_cls}">RSI: {rsi_txt} ({rsi_val:.0f})</div>
                    <div class="indicator-badge badge-neutral">Vol: {last['volume']/1e6:.2f}M</div>
                </div>
            ''', unsafe_allow_html=True)

    # ----- TAB 2: CONTROLS -----
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('''
                <div class="panel">
                    <div class="panel-header">
                        <div class="panel-icon">📥</div>
                        <h3 class="panel-title">Data & Training</h3>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            days = st.selectbox(
                "Historical data period:",
                TRAINING_DEFAULTS['days_options'],
                index=TRAINING_DEFAULTS['days_default_index'],
                format_func=lambda x: f"{x} days"
            )
            if st.button("📥 Download Data", use_container_width=True):
                with st.spinner(f"Downloading {days} days of data..."):
                    result = run_script("scripts/download_data.py", ["--days", str(days)])
                    if result['success']:
                        st.success("✅ Data downloaded successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Download failed")
                        st.code(result['stderr'][:500])

            st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
            epochs = st.slider(
                "Training epochs:",
                TRAINING_DEFAULTS['epochs_min'],
                TRAINING_DEFAULTS['epochs_max'],
                TRAINING_DEFAULTS['epochs_default']
            )
            if st.button("🧠 Train Model", use_container_width=True):
                with st.spinner(f"Training model ({epochs} epochs)..."):
                    result = run_script("scripts/train_model.py", ["--epochs", str(epochs)])
                    if result['success']:
                        st.success("✅ Model trained!")
                    else:
                        st.error("❌ Training failed")

        with col2:
            st.markdown('''
                <div class="panel">
                    <div class="panel-header">
                        <div class="panel-icon">🚀</div>
                        <h3 class="panel-title">Analysis Control</h3>
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            running, pid = is_analysis_running()
            if running:
                st.markdown(f'''
                    <div class="status-box status-running">
                        <div class="status-box-icon" style="background: rgba(16,185,129,0.2);">🟢</div>
                        <div class="status-box-text">
                            <div class="status-box-title" style="color: #10b981;">Analysis Running</div>
                            <div class="status-box-sub">Process ID: {pid}</div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
                if st.button("🛑 Stop Analysis", use_container_width=True):
                    run_script("stop_analysis.py")
                    time.sleep(PROCESS_DELAYS['after_stop'])
                    st.rerun()
            else:
                st.markdown('''
                    <div class="status-box status-stopped">
                        <div class="status-box-icon" style="background: rgba(245,158,11,0.2);">⏸️</div>
                        <div class="status-box-text">
                            <div class="status-box-title" style="color: #f59e0b;">Analysis Stopped</div>
                            <div class="status-box-sub">Ready to start</div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
                if st.button("▶️ Start Analysis", use_container_width=True):
                    if stats['candles'] < MIN_CANDLES_FOR_ANALYSIS:
                        st.error("⚠️ Not enough data! Download data first.")
                    else:
                        venv_py = PROJECT_ROOT / "venv" / "bin" / "python"
                        py = str(venv_py) if venv_py.exists() else "python3"
                        log_file = PROJECT_ROOT / "data" / "analysis_output.log"
                        # Use devnull for subprocess output, process writes its own logs
                        with open(log_file, "a") as log:
                            subprocess.Popen(
                                [py, str(PROJECT_ROOT / "run_analysis.py")],
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                cwd=PROJECT_ROOT,
                                start_new_session=True
                            )
                        time.sleep(PROCESS_DELAYS['after_start'])
                        st.rerun()

            st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔔 Test Alert", use_container_width=True):
                    run_script("scripts/test_notifications.py")
                    st.success("Notification sent!")
            with c2:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()

    # ----- TAB 3: SIGNALS -----
    with tab3:
        signals = load_signals()
        if signals.empty:
            st.info("No signals yet. Start analysis to generate trading signals.")
        else:
            for _, r in signals.iterrows():
                sig = html.escape(str(r['signal']))
                is_buy = 'BUY' in sig.upper()
                cls = 'buy' if is_buy else 'sell'
                icon = '📈' if is_buy else '📉'
                conf = r['confidence']
                conf_cls = 'conf-high' if conf >= CONFIDENCE_HIGH else 'conf-med'
                dt = html.escape(str(r['datetime']))
                st.markdown(f'''
                    <div class="signal-item {cls}">
                        <div class="signal-icon {cls}">{icon}</div>
                        <div class="signal-content">
                            <div class="signal-type">{sig}</div>
                            <div class="signal-time">{dt}</div>
                        </div>
                        <div class="signal-price">${r['price']:,.2f}</div>
                        <div class="signal-conf {conf_cls}">{conf:.0%}</div>
                    </div>
                ''', unsafe_allow_html=True)

    # ----- TAB 4: PATTERNS -----
    with tab4:
        if df.empty:
            st.info("No data to analyze.")
        else:
            patterns = detect_patterns(df)
            if patterns:
                for p in patterns:
                    tag_cls = f"tag-{p['type']}"
                    icon = '📈' if p['type'] == 'bullish' else '📉' if p['type'] == 'bearish' else '➡️'
                    st.markdown(f'''
                        <div class="pattern-item">
                            <div class="pattern-info">
                                <h4>{icon} {p['name']}</h4>
                                <p>{p['desc']}</p>
                            </div>
                            <div class="pattern-tag {tag_cls}">{p['type']}</div>
                        </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No significant patterns detected in recent price action.")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Bullish Patterns**\n\nHammer, Bullish Engulfing, Morning Star, Inverted Hammer")
            with c2:
                st.markdown("**Bearish Patterns**\n\nShooting Star, Bearish Engulfing, Evening Star, Hanging Man")

    # ----- TAB 5: SETTINGS -----
    with tab5:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Data Configuration**")
            st.markdown(f"- Symbol: `{config.get('data',{}).get('symbol','N/A')}`")
            st.markdown(f"- Interval: `{config.get('data',{}).get('interval','N/A')}`")
            st.markdown(f"- Exchange: `{config.get('data',{}).get('exchange','N/A')}`")
        with c2:
            st.markdown("**Analysis Settings**")
            st.markdown(f"- Update: `{config.get('analysis',{}).get('update_interval','N/A')}s`")
            st.markdown(f"- Min Confidence: `{config.get('analysis',{}).get('min_confidence',0):.0%}`")
        with c3:
            st.markdown("**Model Status**")
            model_path = PROJECT_ROOT / config.get('model', {}).get('path', 'models/best_model.pt')
            if model_path.exists():
                st.markdown("✅ Model ready")
            else:
                st.markdown("⚠️ Not trained")

        with st.expander("📋 System Logs"):
            c1, c2 = st.columns(2)
            log_tail = DATA_LIMITS['log_tail_chars']
            with c1:
                st.markdown("**Trading Log**")
                log_path = PROJECT_ROOT / "data" / "trading.log"
                if log_path.exists():
                    content = html.escape(log_path.read_text()[-log_tail:])
                    st.markdown(f'<div class="log-viewer"><pre>{content}</pre></div>', unsafe_allow_html=True)
                else:
                    st.info("No log file")
            with c2:
                st.markdown("**Analysis Output**")
                out_path = PROJECT_ROOT / "data" / "analysis_output.log"
                if out_path.exists():
                    content = html.escape(out_path.read_text()[-log_tail:])
                    st.markdown(f'<div class="log-viewer"><pre>{content}</pre></div>', unsafe_allow_html=True)
                else:
                    st.info("No output log")

    # ===== FOOTER =====
    st.markdown('''
        <div class="footer">
            <div class="footer-warning">⚠️ Signal System Only — No automatic trading. You make all decisions.</div>
            <p class="footer-note">Analysis runs in background. Closing this page won't stop it.</p>
        </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
