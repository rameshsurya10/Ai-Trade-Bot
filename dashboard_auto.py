"""
AI Trade Bot - Auto-Learning Dashboard
======================================
Shows: Performance metrics, auto-retrain status, live signals
Focus: Monitoring auto-learning system
"""

import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import logging
import os
import yaml

# Setup
ROOT = Path(__file__).parent
PID_FILE = ROOT / "data" / ".analysis.pid"
DATA_DIR = ROOT / "data"
DB_FILE = DATA_DIR / "trading.db"
CONFIG_FILE = ROOT / "config.yaml"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Trade Bot - Auto-Learning",
    page_icon="🧠",
    layout="wide"
)

# Load config
def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

# Check process status
def is_process_running(pid_file):
    if not pid_file.exists():
        return False, None
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True, pid
    except:
        pid_file.unlink(missing_ok=True)
        return False, None

# Get signals from database
def get_signals(limit=50):
    if not DB_FILE.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_FILE)
        query = """
            SELECT timestamp, direction, confidence, entry_price,
                   stop_loss, take_profit, status
            FROM signals
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error reading signals: {e}")
        return pd.DataFrame()

# Calculate performance metrics
def calculate_performance(signals_df):
    if signals_df.empty or len(signals_df) < 5:
        return {
            'total_signals': 0,
            'win_rate': 0,
            'avg_confidence': 0,
            'long_signals': 0,
            'short_signals': 0
        }

    total = len(signals_df)

    # Count by direction
    long_signals = len(signals_df[signals_df['direction'] == 'LONG'])
    short_signals = len(signals_df[signals_df['direction'] == 'SHORT'])

    # Average confidence
    avg_conf = signals_df['confidence'].mean() if 'confidence' in signals_df else 0

    # Win rate (if status column exists)
    if 'status' in signals_df.columns:
        completed = signals_df[signals_df['status'].isin(['WIN', 'LOSS'])]
        wins = len(completed[completed['status'] == 'WIN'])
        win_rate = (wins / len(completed) * 100) if len(completed) > 0 else 0
    else:
        win_rate = 0

    return {
        'total_signals': total,
        'win_rate': win_rate,
        'avg_confidence': avg_conf * 100,
        'long_signals': long_signals,
        'short_signals': short_signals
    }

# Check if retrain needed
def check_retrain_status(perf):
    auto_config = config.get('auto_training', {})
    if not auto_config.get('enabled', False):
        return False, "Auto-retrain disabled"

    total = perf['total_signals']
    win_rate = perf['win_rate']
    min_threshold = auto_config.get('min_win_rate_threshold', 0.45) * 100
    min_trades = auto_config.get('min_trades_before_retrain', 50)

    if total < 20:
        return False, f"Need {20 - total} more signals for evaluation"

    if win_rate < min_threshold and total >= 20:
        return True, f"Win rate {win_rate:.1f}% < {min_threshold:.1f}% threshold"

    if total >= 100:
        return True, "100+ signals collected - initial retrain recommended"

    return False, "Performance OK, no retrain needed"

# Main UI
st.title("🧠 AI Trading Bot - Auto-Learning Dashboard")
st.markdown("**Real-time monitoring • Performance tracking • Auto-retrain status**")
st.markdown("---")

# Status Section
col1, col2, col3, col4 = st.columns(4)

running, pid = is_process_running(PID_FILE)

with col1:
    if running:
        st.success("🟢 **RUNNING**")
        st.caption(f"PID: {pid}")
    else:
        st.error("🔴 **STOPPED**")
        st.caption("Not running")

with col2:
    auto_enabled = config.get('auto_training', {}).get('enabled', False)
    if auto_enabled:
        st.info("🔄 **AUTO-RETRAIN**")
        st.caption("Enabled")
    else:
        st.warning("⏸️ **STATIC MODEL**")
        st.caption("Disabled")

with col3:
    model_path = Path(config.get('model', {}).get('models_dir', 'models')) / "model_BTC_USDT.pt"
    if model_path.exists():
        mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
        days_old = (datetime.now() - mod_time).days
        st.success("✅ **MODEL LOADED**")
        st.caption(f"Last updated: {days_old}d ago")
    else:
        st.warning("⚠️ **NO MODEL**")
        st.caption("Will auto-train")

with col4:
    signals_df = get_signals(limit=200)
    perf = calculate_performance(signals_df)
    if perf['total_signals'] > 0:
        st.metric("📊 **Total Signals**", perf['total_signals'])
    else:
        st.info("📊 **Waiting**")
        st.caption("No signals yet")

st.markdown("---")

# Performance Metrics
st.subheader("📈 Performance Metrics")

if perf['total_signals'] > 0:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Win Rate",
            f"{perf['win_rate']:.1f}%",
            delta=None,
            help="Percentage of successful predictions"
        )

    with col2:
        st.metric(
            "Avg Confidence",
            f"{perf['avg_confidence']:.1f}%",
            delta=None,
            help="Average prediction confidence"
        )

    with col3:
        st.metric(
            "Long Signals",
            perf['long_signals'],
            delta=None,
            help="Number of LONG (buy) signals"
        )

    with col4:
        st.metric(
            "Short Signals",
            perf['short_signals'],
            delta=None,
            help="Number of SHORT (sell) signals"
        )
else:
    st.info("⏳ Waiting for signals to calculate performance metrics...")

st.markdown("---")

# Auto-Retrain Status
st.subheader("🔄 Auto-Retrain Status")

if auto_enabled:
    auto_config = config.get('auto_training', {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Retrain Triggers:**")
        st.write(f"• Win rate < {auto_config.get('min_win_rate_threshold', 0.45) * 100:.0f}%")
        st.write(f"• Every {auto_config.get('max_days_between_retrain', 30)} days")
        st.write(f"• After {auto_config.get('min_trades_before_retrain', 50)} signals")

    with col2:
        needs_retrain, reason = check_retrain_status(perf)
        st.markdown("**Current Status:**")
        if needs_retrain:
            st.warning(f"⚠️ Retrain Recommended")
            st.caption(reason)
        else:
            st.success("✅ No retrain needed")
            st.caption(reason)

    with col3:
        st.markdown("**Last Retrain:**")
        if model_path.exists():
            mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
            st.write(mod_time.strftime("%Y-%m-%d %H:%M"))

            days_ago = (datetime.now() - mod_time).days
            if days_ago >= auto_config.get('max_days_between_retrain', 30):
                st.warning(f"⏰ {days_ago} days ago (due for retrain)")
            else:
                st.info(f"🕐 {days_ago} days ago")
        else:
            st.warning("Never (no model found)")
else:
    st.info("🔒 Auto-retrain is disabled. Using static model.")
    st.caption("To enable: Set `auto_training.enabled: true` in config.yaml and use run_analysis_auto.py")

st.markdown("---")

# Recent Signals
st.subheader("📡 Recent Signals")

if not signals_df.empty:
    # Format timestamp
    signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
    signals_df['time'] = signals_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

    # Format confidence
    if 'confidence' in signals_df.columns:
        signals_df['conf %'] = (signals_df['confidence'] * 100).round(1)

    # Format prices
    if 'entry_price' in signals_df.columns:
        signals_df['entry'] = signals_df['entry_price'].apply(lambda x: f"${x:,.2f}")
    if 'stop_loss' in signals_df.columns:
        signals_df['SL'] = signals_df['stop_loss'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
    if 'take_profit' in signals_df.columns:
        signals_df['TP'] = signals_df['take_profit'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")

    # Color code direction
    def color_direction(val):
        if val == 'LONG':
            return 'background-color: #1e4620'
        elif val == 'SHORT':
            return 'background-color: #4a1c1c'
        return ''

    # Display table
    display_cols = ['time', 'direction', 'conf %', 'entry', 'SL', 'TP']
    if 'status' in signals_df.columns:
        display_cols.append('status')

    styled_df = signals_df[display_cols].head(20).style.applymap(
        color_direction,
        subset=['direction']
    )

    st.dataframe(styled_df, use_container_width=True, height=400)

    # Stats
    col1, col2 = st.columns(2)
    with col1:
        recent_24h = signals_df[signals_df['timestamp'] > datetime.now() - timedelta(hours=24)]
        st.caption(f"📊 Signals in last 24h: {len(recent_24h)}")

    with col2:
        if len(signals_df) > 0:
            latest = signals_df.iloc[0]
            time_ago = (datetime.now() - latest['timestamp']).total_seconds() / 60
            st.caption(f"🕐 Last signal: {time_ago:.0f} minutes ago")
else:
    st.info("⏳ No signals yet. Waiting for trading opportunities...")

st.markdown("---")

# Control Section
st.subheader("🎮 System Control")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Start Auto-Learning System:**")
    st.code("venv/bin/python run_analysis_auto.py", language="bash")

    if st.button("📋 Copy Command"):
        st.info("Command copied! (Use Ctrl+C in your terminal)")

with col2:
    st.markdown("**Stop System:**")
    st.code("python stop_analysis.py", language="bash")

    if st.button("🛑 Stop Now") and running:
        try:
            os.kill(pid, 15)  # SIGTERM
            time.sleep(1)
            st.success("✅ System stopped")
            st.rerun()
        except:
            st.error("❌ Failed to stop system")

st.markdown("---")

# Auto-refresh
st.caption("🔄 Auto-refreshes every 30 seconds")
time.sleep(0.1)  # Small delay to prevent too-fast refresh
st.rerun()
