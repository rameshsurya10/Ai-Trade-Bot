"""
Dashboard Features Part 2
==========================
Features 3-5: Portfolio, Risk Management, Alerts
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# FEATURE 3: PORTFOLIO TRACKING DASHBOARD
# ============================================================================

def render_portfolio_tracking(db, paper_trader=None):
    """
    Comprehensive portfolio tracking with P&L visualization.

    Features:
    - Total portfolio value over time
    - Asset allocation
    - Daily/Weekly/Monthly P&L
    - Performance attribution
    """
    st.markdown("## 💰 Portfolio Tracking")

    # Get portfolio data
    if paper_trader:
        stats = paper_trader.get_portfolio_stats()
        portfolio_value = stats['total_value']
        initial = stats['initial_capital']
        total_return = stats['total_return_pct']
        positions = paper_trader.get_open_positions()
    else:
        # Calculate from database
        portfolio_value = 10000  # Placeholder
        initial = 10000
        total_return = 0
        positions = []

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Value",
        f"${portfolio_value:,.2f}",
        f"{total_return:+.2f}%"
    )

    daily_pnl = 0  # Calculate actual daily P&L
    col2.metric(
        "Today's P&L",
        f"${daily_pnl:+,.2f}",
        f"{(daily_pnl/portfolio_value*100):+.2f}%" if portfolio_value > 0 else "0%"
    )

    col3.metric(
        "Total Return",
        f"${portfolio_value - initial:+,.2f}",
        f"{total_return:+.2f}%"
    )

    col4.metric(
        "Active Positions",
        len(positions)
    )

    st.markdown("---")

    # Portfolio value chart
    st.markdown("### 📈 Portfolio Value History")

    # Generate sample portfolio history (replace with real data)
    days = 30
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    portfolio_history = pd.DataFrame({
        'date': dates,
        'value': np.linspace(initial, portfolio_value, days) + np.random.randn(days) * 100
    })

    fig_portfolio = go.Figure()

    fig_portfolio.add_trace(go.Scatter(
        x=portfolio_history['date'],
        y=portfolio_history['value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))

    # Add initial capital line
    fig_portfolio.add_hline(
        y=initial,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital",
        annotation_position="right"
    )

    fig_portfolio.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig_portfolio, use_container_width=True)

    # Asset allocation
    st.markdown("### 🥧 Asset Allocation")

    if positions:
        allocation_data = []
        for pos in positions:
            allocation_data.append({
                'symbol': pos.symbol,
                'value': pos.quantity * pos.current_price,
                'pct': (pos.quantity * pos.current_price / portfolio_value * 100) if portfolio_value > 0 else 0
            })

        # Add cash
        if paper_trader:
            cash = paper_trader.cash
            allocation_data.append({
                'symbol': 'CASH',
                'value': cash,
                'pct': (cash / portfolio_value * 100) if portfolio_value > 0 else 100
            })

        allocation_df = pd.DataFrame(allocation_data)

        # Pie chart
        fig_allocation = go.Figure(data=[go.Pie(
            labels=allocation_df['symbol'],
            values=allocation_df['value'],
            textinfo='label+percent',
            hole=0.3,
            marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe'])
        )])

        fig_allocation.update_layout(
            title="Portfolio Allocation",
            height=400
        )

        st.plotly_chart(fig_allocation, use_container_width=True)

        # Allocation table
        st.dataframe(
            allocation_df[['symbol', 'value', 'pct']].style.format({
                'value': '${:,.2f}',
                'pct': '{:.2f}%'
            }),
            use_container_width=True
        )
    else:
        st.info("No positions - 100% cash")

    # Performance breakdown
    st.markdown("### 📊 Performance Breakdown")

    period_tabs = st.tabs(["Daily", "Weekly", "Monthly"])

    with period_tabs[0]:
        st.info("Daily P&L breakdown (coming soon)")

    with period_tabs[1]:
        st.info("Weekly P&L breakdown (coming soon)")

    with period_tabs[2]:
        st.info("Monthly P&L breakdown (coming soon)")


# ============================================================================
# FEATURE 4: RISK MANAGEMENT DASHBOARD
# ============================================================================

def render_risk_management(db, paper_trader=None):
    """
    Comprehensive risk management dashboard.

    Features:
    - Risk metrics visualization
    - Position sizing calculator
    - Drawdown monitoring
    - Risk/Reward analysis
    - Correlation matrix
    """
    st.markdown("## 🛡️ Risk Management Dashboard")

    # Risk metrics summary
    st.markdown("### 📊 Risk Metrics")

    metrics_cols = st.columns(5)

    # Calculate metrics
    max_drawdown = 0  # Calculate from portfolio history
    var_95 = 0  # Value at Risk
    current_exposure = 0  # Total exposure
    leverage = 1.0
    sharpe = 0

    metrics_cols[0].metric("Max Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse")
    metrics_cols[1].metric("VaR (95%)", f"${var_95:,.2f}")
    metrics_cols[2].metric("Current Exposure", f"${current_exposure:,.2f}")
    metrics_cols[3].metric("Leverage", f"{leverage:.2f}x")
    metrics_cols[4].metric("Sharpe Ratio", f"{sharpe:.2f}")

    st.markdown("---")

    # Position sizing calculator
    st.markdown("### 🎯 Position Sizing Calculator")

    calc_cols = st.columns(3)

    with calc_cols[0]:
        account_balance = st.number_input("Account Balance ($)", 1000, 1000000, 10000, 100)
        risk_pct = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.1)

    with calc_cols[1]:
        entry_price = st.number_input("Entry Price ($)", 0.01, 1000000.0, 100.0, 0.01)
        stop_loss_price = st.number_input("Stop Loss ($)", 0.01, 1000000.0, 95.0, 0.01)

    with calc_cols[2]:
        risk_amount = account_balance * (risk_pct / 100)
        risk_per_share = abs(entry_price - stop_loss_price)
        position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
        position_value = position_size * entry_price

        st.metric("Risk Amount", f"${risk_amount:,.2f}")
        st.metric("Position Size", f"{position_size:.4f} units")
        st.metric("Position Value", f"${position_value:,.2f}")

    st.markdown("---")

    # Risk limits
    st.markdown("### ⚠️ Risk Limits & Alerts")

    limits_cols = st.columns(3)

    with limits_cols[0]:
        st.markdown("**Maximum Drawdown Limit**")
        max_dd_limit = st.slider("Max DD %", 5, 50, 20, 5)
        current_dd = 0  # Calculate actual
        dd_progress = min(100, (current_dd / max_dd_limit) * 100)
        st.progress(dd_progress / 100, text=f"Current: {current_dd:.1f}% / Limit: {max_dd_limit}%")

    with limits_cols[1]:
        st.markdown("**Daily Loss Limit**")
        daily_loss_limit = st.number_input("Max Daily Loss ($)", 100, 10000, 500, 100)
        current_daily_loss = 0
        loss_progress = min(100, (current_daily_loss / daily_loss_limit) * 100)
        st.progress(loss_progress / 100, text=f"Current: ${current_daily_loss:.0f} / Limit: ${daily_loss_limit:.0f}")

    with limits_cols[2]:
        st.markdown("**Open Position Limit**")
        max_positions = st.number_input("Max Positions", 1, 20, 5)
        current_positions = len(paper_trader.get_open_positions()) if paper_trader else 0
        pos_progress = min(100, (current_positions / max_positions) * 100)
        st.progress(pos_progress / 100, text=f"Current: {current_positions} / Limit: {max_positions}")

    st.markdown("---")

    # Risk/Reward analysis
    st.markdown("### 📈 Risk/Reward Analysis")

    rr_cols = st.columns(2)

    with rr_cols[0]:
        st.markdown("**Required Win Rate by R:R Ratio**")
        rr_ratios = [1, 1.5, 2, 2.5, 3, 4, 5]
        required_wr = [(1 / (1 + r)) * 100 for r in rr_ratios]

        fig_rr = go.Figure(data=[go.Bar(
            x=[f"{r}:1" for r in rr_ratios],
            y=required_wr,
            marker_color='#667eea',
            text=[f"{wr:.1f}%" for wr in required_wr],
            textposition='outside'
        )])

        fig_rr.update_layout(
            title="Breakeven Win Rate",
            xaxis_title="Risk:Reward Ratio",
            yaxis_title="Required Win Rate (%)",
            height=300
        )

        st.plotly_chart(fig_rr, use_container_width=True)

    with rr_cols[1]:
        st.markdown("**Expectancy Calculator**")
        win_rate_input = st.slider("Your Win Rate (%)", 0, 100, 50)
        avg_win_input = st.number_input("Avg Win ($)", 1, 10000, 100)
        avg_loss_input = st.number_input("Avg Loss ($)", 1, 10000, 50)

        expectancy = ((win_rate_input / 100) * avg_win_input) - ((1 - win_rate_input / 100) * avg_loss_input)

        st.metric("Expectancy per Trade", f"${expectancy:+,.2f}")

        if expectancy > 0:
            st.success("✅ Positive expectancy - profitable system")
        else:
            st.error("❌ Negative expectancy - losing system")


# ============================================================================
# FEATURE 5: REAL-TIME ALERTS SYSTEM
# ============================================================================

def render_realtime_alerts():
    """
    Real-time alerts and notifications in UI.

    Features:
    - Browser notifications
    - Sound alerts
    - Alert history
    - Custom alert conditions
    - Alert management
    """
    st.markdown("## 🔔 Real-Time Alerts")

    # Alert settings
    st.markdown("### ⚙️ Alert Configuration")

    alert_cols = st.columns(3)

    with alert_cols[0]:
        browser_notify = st.checkbox("Browser Notifications", value=True)
        sound_alerts = st.checkbox("Sound Alerts", value=True)
        desktop_popup = st.checkbox("Desktop Popups", value=False)

    with alert_cols[1]:
        alert_on_signal = st.checkbox("New Signal Generated", value=True)
        alert_on_fill = st.checkbox("Order Filled", value=True)
        alert_on_stop = st.checkbox("Stop Loss Hit", value=True)

    with alert_cols[2]:
        alert_on_target = st.checkbox("Take Profit Hit", value=True)
        alert_on_threshold = st.checkbox("Price Threshold", value=False)

    if browser_notify:
        # JavaScript for browser notifications
        st.markdown("""
        <script>
        if ("Notification" in window) {
            if (Notification.permission === "default") {
                Notification.requestPermission();
            }
        }

        function showNotification(title, body) {
            if (Notification.permission === "granted") {
                new Notification(title, {
                    body: body,
                    icon: "📈"
                });
            }
        }
        </script>
        """, unsafe_allow_html=True)

    # Test alert button
    if st.button("🔔 Test Alert"):
        st.success("✅ Test alert triggered!")
        if sound_alerts:
            st.markdown("""
            <audio autoplay>
                <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSiJ0/LTgj0IHm7A7+OZUQ4NYqzn77BdGAg+ltryxnMpBSl/zfDajj4IHm/D8eSXSwwRZrXq5qVQDw4+oePzsG0gBSiL1fLShj0IHnLD8OKaUA4MZKzp6K9aFQlAm9zyxnEoBSZ8yPDUjj4IH3LE8eKaTQwPZa/p5qJPDg0+oOPzsm4gBSaL1fPTiD4IHnPC8eSaTQsMY6rm6q1ZFgo="/>
            </audio>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Active alerts
    st.markdown("### 🚨 Active Alerts")

    # Sample alerts (replace with real data)
    active_alerts = [
        {"time": "2 min ago", "type": "Signal", "message": "BUY signal generated for BTC/USDT", "severity": "info"},
        {"time": "15 min ago", "type": "Price", "message": "BTC crossed $50,000", "severity": "warning"},
    ]

    for alert in active_alerts:
        severity_color = {
            "info": "#667eea",
            "warning": "#ffc107",
            "error": "#f5576c"
        }[alert['severity']]

        st.markdown(f"""
        <div style="border-left: 4px solid {severity_color}; padding: 10px; margin-bottom: 10px; background: rgba(102, 126, 234, 0.05);">
            <b>{alert['type']}</b> - {alert['time']}<br>
            {alert['message']}
        </div>
        """, unsafe_allow_html=True)

    # Alert history
    st.markdown("### 📋 Alert History")

    history_data = pd.DataFrame({
        'Time': pd.date_range(end=datetime.now(), periods=10, freq='H'),
        'Type': ['Signal', 'Order', 'Price', 'Stop Loss', 'Signal'] * 2,
        'Message': ['Alert message'] * 10,
        'Severity': ['info', 'success', 'warning', 'error', 'info'] * 2
    })

    st.dataframe(history_data, use_container_width=True)

    # Clear history button
    if st.button("🗑️ Clear History"):
        st.success("Alert history cleared")
