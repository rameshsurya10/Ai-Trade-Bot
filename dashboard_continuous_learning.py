"""
Continuous Learning Dashboard
==============================

Real-time monitoring dashboard for the continuous learning trading system.

Features:
- Learning mode indicators (LEARNING/TRADING)
- Confidence trend charts
- News sentiment panel
- Retraining history visualization
- Multi-timeframe signal strength
- Performance metrics by mode
- Safety status monitoring

Usage:
    streamlit run dashboard_continuous_learning.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import json

# Imports
from src.core.database import Database
from src.core.config import load_config

# Page config
st.set_page_config(
    page_title="Continuous Learning Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }

    /* Mode Indicators */
    .mode-learning {
        background: linear-gradient(135deg, #FFA726 0%, #FB8C00 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
    }

    .mode-trading {
        background: linear-gradient(135deg, #66BB6A 0%, #43A047 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
    }

    /* Confidence Badge */
    .confidence-high {
        background: #43A047;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
    }

    .confidence-medium {
        background: #FFA726;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
    }

    .confidence-low {
        background: #EF5350;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
    }

    /* Metric Cards */
    .metric-card {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 4px solid #667eea;
    }

    /* Safety Status */
    .safety-pass {
        color: #66BB6A;
        font-weight: bold;
    }

    .safety-warn {
        color: #FFA726;
        font-weight: bold;
    }

    .safety-fail {
        color: #EF5350;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize
@st.cache_resource
def init_database():
    """Initialize database connection."""
    config = load_config()
    return Database(config['database']['path']), config

db, config = init_database()

# =============================================================================
# HEADER
# =============================================================================

st.title("🧠 Continuous Learning Dashboard")
st.markdown("Real-time monitoring of adaptive trading system")

# Auto-refresh
if st.sidebar.checkbox("Auto-refresh", value=True):
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10)
    st_autorefresh = st.empty()
    import time
    time.sleep(refresh_interval)
    st.rerun()

# =============================================================================
# CURRENT STATUS
# =============================================================================

st.header("📊 Current Status")

# Get latest learning state
try:
    query = """
        SELECT symbol, interval, mode, confidence_score, entered_at, reason
        FROM learning_states
        ORDER BY id DESC
        LIMIT 1
    """
    result = db.execute_query(query)

    if result:
        symbol, interval, mode, confidence, entered_at, reason = result[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Mode indicator
            if mode == 'TRADING':
                st.markdown(
                    f'<div class="mode-trading">🎯 TRADING MODE</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="mode-learning">📚 LEARNING MODE</div>',
                    unsafe_allow_html=True
                )

        with col2:
            # Confidence
            if confidence >= 0.80:
                conf_class = "confidence-high"
            elif confidence >= 0.60:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"

            st.markdown(
                f'<div class="{conf_class}">Confidence: {confidence:.1%}</div>',
                unsafe_allow_html=True
            )

        with col3:
            st.metric("Symbol", symbol)
            st.metric("Timeframe", interval)

        with col4:
            st.metric("In Mode Since", entered_at)
            if reason:
                st.caption(f"Reason: {reason}")

    else:
        st.info("No learning state data available yet")

except Exception as e:
    st.error(f"Error loading current status: {e}")

# =============================================================================
# CONFIDENCE TREND
# =============================================================================

st.header("📈 Confidence Trend")

try:
    query = """
        SELECT timestamp, confidence_score, mode, symbol, interval
        FROM confidence_history
        ORDER BY timestamp DESC
        LIMIT 200
    """
    result = db.execute_query(query)

    if result:
        df_confidence = pd.DataFrame(result, columns=[
            'timestamp', 'confidence', 'mode', 'symbol', 'interval'
        ])
        df_confidence['timestamp'] = pd.to_datetime(df_confidence['timestamp'])
        df_confidence = df_confidence.sort_values('timestamp')

        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Confidence Level', 'Mode'),
            vertical_spacing=0.1
        )

        # Confidence line
        fig.add_trace(
            go.Scatter(
                x=df_confidence['timestamp'],
                y=df_confidence['confidence'],
                mode='lines+markers',
                name='Confidence',
                line=dict(color='#667eea', width=2),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ),
            row=1, col=1
        )

        # Threshold lines
        fig.add_hline(
            y=0.80, line_dash="dash", line_color="green",
            annotation_text="Trading Threshold (80%)",
            row=1, col=1
        )

        fig.add_hline(
            y=0.75, line_dash="dash", line_color="orange",
            annotation_text="Exit Threshold (75%)",
            row=1, col=1
        )

        # Mode as scatter
        df_confidence['mode_numeric'] = df_confidence['mode'].map({
            'TRADING': 1,
            'LEARNING': 0
        })

        fig.add_trace(
            go.Scatter(
                x=df_confidence['timestamp'],
                y=df_confidence['mode_numeric'],
                mode='markers',
                name='Mode',
                marker=dict(
                    color=df_confidence['mode_numeric'],
                    colorscale=[[0, '#FFA726'], [1, '#66BB6A']],
                    size=10,
                    line=dict(width=1, color='white')
                ),
                text=df_confidence['mode'],
                hovertemplate='%{text}<extra></extra>'
            ),
            row=2, col=1
        )

        fig.update_yaxes(title_text="Confidence", row=1, col=1, range=[0, 1])
        fig.update_yaxes(
            title_text="Mode",
            row=2, col=1,
            ticktext=['LEARNING', 'TRADING'],
            tickvals=[0, 1]
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            template='plotly_dark',
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Confidence", f"{df_confidence.iloc[-1]['confidence']:.1%}")

        with col2:
            avg_learning = df_confidence[df_confidence['mode'] == 'LEARNING']['confidence'].mean()
            st.metric("Avg (Learning)", f"{avg_learning:.1%}")

        with col3:
            avg_trading = df_confidence[df_confidence['mode'] == 'TRADING']['confidence'].mean()
            st.metric("Avg (Trading)", f"{avg_trading:.1%}")

        with col4:
            transitions = (df_confidence['mode'] != df_confidence['mode'].shift(1)).sum()
            st.metric("Mode Transitions", transitions)

    else:
        st.info("No confidence history available yet")

except Exception as e:
    st.error(f"Error loading confidence trend: {e}")

# =============================================================================
# NEWS SENTIMENT
# =============================================================================

st.header("📰 News Sentiment Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    try:
        query = """
            SELECT datetime, title, sentiment_score, sentiment_label, source
            FROM news_articles
            WHERE processed = 1
            ORDER BY timestamp DESC
            LIMIT 20
        """
        result = db.execute_query(query)

        if result:
            df_news = pd.DataFrame(result, columns=[
                'datetime', 'title', 'sentiment', 'label', 'source'
            ])

            # Sentiment chart
            fig = go.Figure()

            colors = df_news['sentiment'].apply(
                lambda x: '#66BB6A' if x > 0.05 else '#EF5350' if x < -0.05 else '#FFA726'
            )

            fig.add_trace(go.Bar(
                x=df_news.index,
                y=df_news['sentiment'],
                marker_color=colors,
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                              'Sentiment: %{y:.2f}<br>' +
                              'Source: %{customdata[1]}<extra></extra>',
                customdata=df_news[['title', 'source']].values
            ))

            fig.update_layout(
                title="Recent News Sentiment",
                xaxis_title="Article #",
                yaxis_title="Sentiment Score",
                height=400,
                template='plotly_dark',
                showlegend=False
            )

            fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)

            st.plotly_chart(fig, use_container_width=True)

            # News list
            st.subheader("Recent Headlines")
            for _, row in df_news.head(10).iterrows():
                sentiment_emoji = "🟢" if row['sentiment'] > 0.05 else "🔴" if row['sentiment'] < -0.05 else "🟡"
                st.markdown(
                    f"{sentiment_emoji} **{row['label']}** ({row['sentiment']:.2f}) - {row['title'][:80]}... "
                    f"*({row['source']})*"
                )

        else:
            st.info("No news data available yet")

    except Exception as e:
        st.error(f"Error loading news sentiment: {e}")

with col2:
    try:
        # Sentiment aggregation
        query = """
            SELECT sentiment_1h, sentiment_6h, sentiment_24h,
                   news_volume_1h, source_diversity
            FROM sentiment_features
            ORDER BY id DESC
            LIMIT 1
        """
        result = db.execute_query(query)

        if result:
            sent_1h, sent_6h, sent_24h, volume, diversity = result[0]

            st.subheader("Aggregated Sentiment")

            # Sentiment metrics
            st.metric("1 Hour", f"{sent_1h:.2f}")
            st.metric("6 Hours", f"{sent_6h:.2f}")
            st.metric("24 Hours", f"{sent_24h:.2f}")

            st.metric("News Volume (1h)", volume)
            st.metric("Source Diversity", f"{diversity:.2f}")

            # Momentum
            momentum = sent_1h - sent_6h
            momentum_emoji = "📈" if momentum > 0 else "📉"
            st.metric(
                "Sentiment Momentum",
                f"{momentum_emoji} {momentum:.2f}"
            )

        else:
            st.info("No sentiment aggregation yet")

    except Exception as e:
        st.error(f"Error loading sentiment aggregation: {e}")

# =============================================================================
# RETRAINING HISTORY
# =============================================================================

st.header("🔄 Retraining History")

try:
    query = """
        SELECT triggered_at, symbol, interval, trigger_reason,
               status, validation_confidence, duration_seconds, epochs_trained
        FROM retraining_history
        ORDER BY id DESC
        LIMIT 20
    """
    result = db.execute_query(query)

    if result:
        df_retrain = pd.DataFrame(result, columns=[
            'triggered_at', 'symbol', 'interval', 'reason',
            'status', 'confidence', 'duration', 'epochs'
        ])

        col1, col2 = st.columns([3, 1])

        with col1:
            # Retraining timeline
            fig = go.Figure()

            # Success/failure markers
            success_mask = df_retrain['status'] == 'success'

            fig.add_trace(go.Scatter(
                x=df_retrain[success_mask]['triggered_at'],
                y=df_retrain[success_mask]['confidence'],
                mode='markers',
                name='Success',
                marker=dict(
                    size=12,
                    color='#66BB6A',
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>Success</b><br>' +
                              'Time: %{x}<br>' +
                              'Confidence: %{y:.1%}<br>' +
                              'Symbol: %{customdata[0]}<br>' +
                              'Reason: %{customdata[1]}<extra></extra>',
                customdata=df_retrain[success_mask][['symbol', 'reason']].values
            ))

            fig.add_trace(go.Scatter(
                x=df_retrain[~success_mask]['triggered_at'],
                y=df_retrain[~success_mask]['confidence'],
                mode='markers',
                name='Failed/Ongoing',
                marker=dict(
                    size=12,
                    color='#EF5350',
                    symbol='x',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>Failed</b><br>' +
                              'Time: %{x}<br>' +
                              'Symbol: %{customdata[0]}<br>' +
                              'Reason: %{customdata[1]}<extra></extra>',
                customdata=df_retrain[~success_mask][['symbol', 'reason']].values
            ))

            fig.add_hline(
                y=0.80, line_dash="dash", line_color="green",
                annotation_text="Target (80%)"
            )

            fig.update_layout(
                title="Retraining Events",
                xaxis_title="Time",
                yaxis_title="Validation Confidence",
                height=400,
                template='plotly_dark',
                hovermode='closest'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Summary")

            total = len(df_retrain)
            success = (df_retrain['status'] == 'success').sum()
            success_rate = (success / total * 100) if total > 0 else 0

            st.metric("Total Retrainings", total)
            st.metric("Success Rate", f"{success_rate:.0f}%")

            avg_duration = df_retrain['duration'].mean()
            st.metric("Avg Duration", f"{avg_duration:.0f}s")

            avg_epochs = df_retrain['epochs'].mean()
            st.metric("Avg Epochs", f"{avg_epochs:.0f}")

            # Trigger reasons
            st.subheader("Trigger Reasons")
            reason_counts = df_retrain['reason'].value_counts()
            for reason, count in reason_counts.items():
                st.text(f"{reason}: {count}")

        # Recent retraining table
        st.subheader("Recent Retraining Events")
        display_df = df_retrain.head(10)[['triggered_at', 'symbol', 'interval', 'reason', 'status', 'confidence']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
        st.dataframe(display_df, use_container_width=True)

    else:
        st.info("No retraining history available yet")

except Exception as e:
    st.error(f"Error loading retraining history: {e}")

# =============================================================================
# MULTI-TIMEFRAME SIGNALS
# =============================================================================

st.header("🎯 Multi-Timeframe Signals")

try:
    # Get latest signals per interval
    query = """
        SELECT symbol, datetime, direction, confidence, metadata
        FROM signals
        WHERE datetime > datetime('now', '-1 hour')
        ORDER BY timestamp DESC
        LIMIT 50
    """
    result = db.execute_query(query)

    if result:
        signals = []
        for row in result:
            symbol, dt, direction, confidence, metadata = row
            try:
                if metadata:
                    meta = json.loads(metadata)
                    if 'timeframe_signals' in meta:
                        for interval, signal_data in meta['timeframe_signals'].items():
                            signals.append({
                                'symbol': symbol,
                                'interval': interval,
                                'direction': signal_data.get('direction', 'NEUTRAL'),
                                'confidence': signal_data.get('confidence', 0.0)
                            })
            except:
                pass

        if signals:
            df_signals = pd.DataFrame(signals)

            # Group by interval
            intervals = sorted(df_signals['interval'].unique())

            fig = go.Figure()

            for interval in intervals:
                interval_data = df_signals[df_signals['interval'] == interval]
                avg_confidence = interval_data['confidence'].mean()

                fig.add_trace(go.Bar(
                    name=interval,
                    x=[interval],
                    y=[avg_confidence],
                    marker_color='#667eea' if avg_confidence >= 0.80 else '#FFA726'
                ))

            fig.add_hline(
                y=0.80, line_dash="dash", line_color="green",
                annotation_text="Trading Threshold"
            )

            fig.update_layout(
                title="Average Confidence by Timeframe (Last Hour)",
                xaxis_title="Timeframe",
                yaxis_title="Average Confidence",
                height=400,
                template='plotly_dark',
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Signal breakdown
            col1, col2, col3 = st.columns(3)

            with col1:
                buy_signals = len(df_signals[df_signals['direction'] == 'BUY'])
                st.metric("BUY Signals", buy_signals)

            with col2:
                sell_signals = len(df_signals[df_signals['direction'] == 'SELL'])
                st.metric("SELL Signals", sell_signals)

            with col3:
                neutral_signals = len(df_signals[df_signals['direction'] == 'NEUTRAL'])
                st.metric("NEUTRAL Signals", neutral_signals)

        else:
            st.info("No multi-timeframe signal data in metadata")

    else:
        st.info("No recent signals available")

except Exception as e:
    st.error(f"Error loading multi-timeframe signals: {e}")

# =============================================================================
# PERFORMANCE BY MODE
# =============================================================================

st.header("📊 Performance by Mode")

col1, col2 = st.columns(2)

with col1:
    st.subheader("LEARNING Mode")
    try:
        query = """
            SELECT COUNT(*) as trades,
                   SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                   AVG(pnl_percent) as avg_pnl
            FROM trade_outcomes
            WHERE is_paper_trade = 1
        """
        result = db.execute_query(query)

        if result and result[0][0]:
            trades, wins, avg_pnl = result[0]
            win_rate = (wins / trades * 100) if trades > 0 else 0

            st.metric("Total Trades", trades)
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Avg P&L", f"{avg_pnl:.2f}%")
        else:
            st.info("No LEARNING mode trades yet")

    except Exception as e:
        st.error(f"Error: {e}")

with col2:
    st.subheader("TRADING Mode")
    try:
        query = """
            SELECT COUNT(*) as trades,
                   SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                   AVG(pnl_percent) as avg_pnl
            FROM trade_outcomes
            WHERE is_paper_trade = 0
        """
        result = db.execute_query(query)

        if result and result[0][0]:
            trades, wins, avg_pnl = result[0]
            win_rate = (wins / trades * 100) if trades > 0 else 0

            st.metric("Total Trades", trades)
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Avg P&L", f"{avg_pnl:.2f}%")
        else:
            st.info("No TRADING mode trades yet")

    except Exception as e:
        st.error(f"Error: {e}")

# =============================================================================
# SAFETY STATUS
# =============================================================================

st.header("🛡️ Safety Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Drawdown Check")
    try:
        # Calculate current drawdown
        query = """
            SELECT
                MAX(CASE WHEN pnl_absolute IS NOT NULL THEN pnl_absolute ELSE 0 END) as max_pnl,
                SUM(CASE WHEN pnl_absolute IS NOT NULL THEN pnl_absolute ELSE 0 END) as current_pnl
            FROM trade_outcomes
        """
        result = db.execute_query(query)

        if result:
            max_pnl, current_pnl = result[0]
            max_pnl = max_pnl or 0
            current_pnl = current_pnl or 0

            drawdown = ((max_pnl - current_pnl) / (max_pnl or 1)) * 100

            if drawdown <= 10:
                status_class = "safety-pass"
                status = "✓ PASS"
            elif drawdown <= 15:
                status_class = "safety-warn"
                status = "⚠ WARNING"
            else:
                status_class = "safety-fail"
                status = "✗ FAIL"

            st.markdown(f'<p class="{status_class}">{status}</p>', unsafe_allow_html=True)
            st.metric("Current Drawdown", f"{drawdown:.1f}%")
            st.caption("Limit: 15%")

    except Exception as e:
        st.error(f"Error: {e}")

with col2:
    st.subheader("Win Rate Check")
    try:
        query = """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins
            FROM trade_outcomes
            WHERE exit_time IS NOT NULL
        """
        result = db.execute_query(query)

        if result and result[0][0]:
            total, wins = result[0]
            win_rate = (wins / total * 100) if total > 0 else 0

            if win_rate >= 50:
                status_class = "safety-pass"
                status = "✓ PASS"
            elif win_rate >= 45:
                status_class = "safety-warn"
                status = "⚠ WARNING"
            else:
                status_class = "safety-fail"
                status = "✗ FAIL"

            st.markdown(f'<p class="{status_class}">{status}</p>', unsafe_allow_html=True)
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.caption("Minimum: 45%")
        else:
            st.info("Insufficient trades")

    except Exception as e:
        st.error(f"Error: {e}")

with col3:
    st.subheader("System Stability")
    try:
        query = """
            SELECT COUNT(*) as transitions,
                   MIN(entered_at) as first_time
            FROM learning_states
        """
        result = db.execute_query(query)

        if result and result[0][0]:
            transitions, first_time = result[0]

            # Calculate transitions per hour
            if first_time:
                first_dt = datetime.fromisoformat(first_time)
                hours_elapsed = (datetime.utcnow() - first_dt).total_seconds() / 3600
                trans_per_hour = transitions / max(hours_elapsed, 1)
            else:
                trans_per_hour = 0

            if trans_per_hour < 1:
                status_class = "safety-pass"
                status = "✓ PASS"
            elif trans_per_hour < 2:
                status_class = "safety-warn"
                status = "⚠ WARNING"
            else:
                status_class = "safety-fail"
                status = "✗ FAIL"

            st.markdown(f'<p class="{status_class}">{status}</p>', unsafe_allow_html=True)
            st.metric("Transitions/Hour", f"{trans_per_hour:.2f}")
            st.caption("Limit: < 2.0")
        else:
            st.info("No data yet")

    except Exception as e:
        st.error(f"Error: {e}")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
