"""
AI Trade Bot - Core Workflow Dashboard
========================================
Focus: Download → Train → Backtest → Deploy
No fancy UI, just CORE FUNCTIONALITY.
"""

import streamlit as st
import pandas as pd
import subprocess
import sys
from pathlib import Path
import logging
import os
import signal
import time

# Setup
ROOT = Path(__file__).parent
PID_FILE = ROOT / "run_analysis.pid"
DATA_DIR = ROOT / "data"
MODEL_FILE = DATA_DIR / "lstm_model.pt"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Trade Bot - Core Workflow",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 AI Trading Bot - Core Workflow")
st.markdown("**Download → Train → Backtest → Deploy**")
st.markdown("---")

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

# ============================================================================
# WORKFLOW PROGRESS
# ============================================================================
st.header("📊 Workflow Status")

col1, col2, col3, col4 = st.columns(4)

# Check what's completed
data_exists = (DATA_DIR / "historical").exists() and len(list((DATA_DIR / "historical").glob("*.csv"))) > 0
model_exists = MODEL_FILE.exists()
backtest_results = (DATA_DIR / "backtest_results.json").exists()
engine_running, engine_pid = is_process_running(PID_FILE)

with col1:
    st.metric("1. Data", "✅" if data_exists else "❌")
with col2:
    st.metric("2. Model", "✅" if model_exists else "❌")
with col3:
    st.metric("3. Backtest", "✅" if backtest_results else "❌")
with col4:
    st.metric("4. Engine", "✅" if engine_running else "❌")

st.markdown("---")

# ============================================================================
# STEP 1: DOWNLOAD DATA
# ============================================================================
st.header("📥 1. Download Historical Data")

if not data_exists:
    st.warning("Download 6+ months of data for training")
    if st.button("📥 Download Data Now", type="primary"):
        with st.spinner("Downloading..."):
            result = subprocess.run(
                [sys.executable, "scripts/download_data.py"],
                cwd=str(ROOT),
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("✅ Downloaded!")
                st.code(result.stdout)
                st.rerun()
            else:
                st.error("❌ Failed")
                st.code(result.stderr)
else:
    st.success("✅ Data ready")
    data_files = list((DATA_DIR / "historical").glob("*.csv"))
    st.write(f"Files: {len(data_files)}")

st.markdown("---")

# ============================================================================
# STEP 2: TRAIN MODEL
# ============================================================================
st.header("🧠 2. Train LSTM Model")

if not data_exists:
    st.info("Download data first")
elif not model_exists:
    st.warning("Model not trained")
    epochs = st.slider("Epochs", 10, 200, 100)
    if st.button("🧠 Train Model", type="primary"):
        with st.spinner(f"Training {epochs} epochs (30-60 min)..."):
            result = subprocess.run(
                [sys.executable, "scripts/train_model.py", "--epochs", str(epochs)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=3600
            )
            if result.returncode == 0:
                st.success("✅ Trained!")
                st.code(result.stdout)
                st.rerun()
            else:
                st.error("❌ Failed")
                st.code(result.stderr)
else:
    st.success("✅ Model trained")
    st.write(f"File: {MODEL_FILE.stat().st_size / 1024 / 1024:.2f} MB")

st.markdown("---")

# ============================================================================
# STEP 3: BACKTEST
# ============================================================================
st.header("📊 3. Backtest Strategy")

if not model_exists:
    st.info("Train model first")
elif not backtest_results:
    st.warning("Strategy not tested")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", pd.to_datetime("2024-01-01"))
    with col2:
        end_date = st.date_input("End", pd.to_datetime("2024-12-01"))

    if st.button("📊 Run Backtest", type="primary"):
        with st.spinner("Backtesting (5-10 min)..."):
            result = subprocess.run(
                [sys.executable, "scripts/run_backtest.py", "--start", str(start_date), "--end", str(end_date)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=600
            )
            if result.returncode == 0:
                st.success("✅ Completed!")
                st.code(result.stdout)
                st.rerun()
            else:
                st.error("❌ Failed")
                st.code(result.stderr)
else:
    st.success("✅ Backtest completed")
    import json
    with open(DATA_DIR / "backtest_results.json") as f:
        results = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win Rate", f"{results.get('win_rate', 0)*100:.1f}%")
    col2.metric("Profit Factor", f"{results.get('profit_factor', 0):.2f}")
    col3.metric("Total PnL", f"${results.get('total_pnl', 0):,.0f}")
    col4.metric("Drawdown", f"{results.get('max_drawdown', 0)*100:.1f}%")

st.markdown("---")

# ============================================================================
# STEP 4: DEPLOY
# ============================================================================
st.header("🚀 4. Deploy Engine")

if not backtest_results:
    st.info("Complete backtest first")
else:
    if engine_running:
        st.success(f"✅ Running (PID: {engine_pid})")
        if st.button("⏹️ Stop", type="secondary"):
            os.kill(engine_pid, signal.SIGTERM)
            PID_FILE.unlink(missing_ok=True)
            st.rerun()
    else:
        st.warning("⏸️ Engine stopped")
        if st.button("▶️ Start Engine", type="primary"):
            venv_py = ROOT / "venv" / "bin" / "python"
            subprocess.Popen(
                [str(venv_py), "run_analysis.py"],
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(2)
            st.rerun()

    # Show signals if running
    if engine_running:
        st.markdown("---")
        st.header("📡 Recent Signals")
        signals_db = DATA_DIR / "trading.db"
        if signals_db.exists():
            import sqlite3
            conn = sqlite3.connect(signals_db)
            signals = pd.read_sql_query("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 20", conn)
            conn.close()
            if len(signals) > 0:
                st.dataframe(signals, use_container_width=True)
            else:
                st.info("No signals yet")

st.markdown("---")
st.caption("AI Trade Bot | Core Workflow - No Fluff, Just Function")
