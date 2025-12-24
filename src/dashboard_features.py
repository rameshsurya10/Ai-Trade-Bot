"""
Dashboard Features - Complete Feature Set
==========================================
All 5 critical features for production-ready dashboard:
1. Backtesting Interface
2. Paper Trading Simulator
3. Portfolio Tracking
4. Risk Management
5. Real-time Alerts
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from .backtesting.visual_backtester import VisualBacktester, BacktestResult
from .paper_trading import PaperTradingSimulator, OrderSide, OrderType
from .core.database import Database

logger = logging.getLogger(__name__)


# ============================================================================
# FEATURE 1: BACKTESTING INTERFACE
# ============================================================================

def render_backtesting_interface(db: Database):
    """
    Complete backtesting interface with visual results.

    Features:
    - Run backtest on historical data
    - Equity curve visualization
    - Comprehensive metrics
    - Monthly breakdown
    - Trade analysis
    """
    st.markdown("## 📊 Backtesting Interface")
    st.markdown("Test your strategy on historical data before going live")

    # Backtesting configuration
    col1, col2, col3 = st.columns(3)

    with col1:
        backtest_days = st.number_input("Backtest Period (days)", 7, 365, 30)
        initial_capital = st.number_input("Initial Capital ($)", 1000, 1000000, 10000, 1000)

    with col2:
        risk_per_trade = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
        commission = st.slider("Commission (%)", 0.0, 1.0, 0.1, 0.05) / 100

    with col3:
        slippage = st.slider("Slippage (%)", 0.0, 1.0, 0.05, 0.01) / 100

    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                # Get historical data
                from .data_service import DataService
                ds = DataService()
                df = ds.get_candles(limit=backtest_days * 24)  # Assuming hourly

                if len(df) < 100:
                    st.error(f"Insufficient data: {len(df)} candles. Need at least 100.")
                    return

                # Get signals from database
                signals_df = pd.read_sql_query(
                    "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 1000",
                    db._get_connection()
                )

                if signals_df.empty:
                    st.warning("No historical signals found. Generate some signals first.")
                    return

                # Run backtest
                backtester = VisualBacktester(initial_capital=initial_capital)
                result = backtester.run_backtest(
                    df=df,
                    signals=signals_df,
                    risk_per_trade=risk_per_trade,
                    commission=commission,
                    slippage=slippage
                )

                # Store result in session state
                st.session_state.backtest_result = result
                st.success("✅ Backtest complete!")

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                logger.error(f"Backtest error: {e}", exc_info=True)

    # Display results if available
    if hasattr(st.session_state, 'backtest_result'):
        result = st.session_state.backtest_result

        st.markdown("---")
        st.markdown("### 📈 Backtest Results")

        # Key metrics
        metrics_cols = st.columns(6)
        metrics = [
            ("Total Return", f"{result.total_return:+.2f}%", result.total_return > 0),
            ("Win Rate", f"{result.win_rate:.1f}%", result.win_rate > 50),
            ("Profit Factor", f"{result.profit_factor:.2f}", result.profit_factor > 1),
            ("Sharpe Ratio", f"{result.sharpe_ratio:.2f}", result.sharpe_ratio > 1),
            ("Max Drawdown", f"{result.max_drawdown:.2f}%", result.max_drawdown < 20),
            ("Total Trades", str(result.total_trades), True)
        ]

        for col, (label, value, is_good) in zip(metrics_cols, metrics):
            delta_color = "normal" if is_good else "inverse"
            col.metric(label, value, delta_color=delta_color)

        # Equity curve chart
        st.markdown("#### 💰 Equity Curve")
        fig_equity = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Portfolio Value', 'Drawdown'),
            row_heights=[0.7, 0.3]
        )

        # Equity line
        fig_equity.add_trace(
            go.Scatter(
                x=result.equity_curve['timestamp'],
                y=result.equity_curve['equity'],
                name='Equity',
                line=dict(color='#667eea', width=2),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ),
            row=1, col=1
        )

        # Drawdown
        fig_equity.add_trace(
            go.Scatter(
                x=result.equity_curve['timestamp'],
                y=result.equity_curve['drawdown'],
                name='Drawdown',
                line=dict(color='#f5576c', width=2),
                fill='tozeroy',
                fillcolor='rgba(245, 87, 108, 0.1)'
            ),
            row=2, col=1
        )

        fig_equity.update_layout(
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # Trade statistics
        st.markdown("#### 📊 Trade Statistics")
        stats_cols = st.columns(4)

        with stats_cols[0]:
            st.metric("Winning Trades", result.winning_trades)
            st.metric("Average Win", f"${result.avg_win:,.2f}")
            st.metric("Largest Win", f"${result.largest_win:,.2f}")

        with stats_cols[1]:
            st.metric("Losing Trades", result.losing_trades)
            st.metric("Average Loss", f"${result.avg_loss:,.2f}")
            st.metric("Largest Loss", f"${result.largest_loss:,.2f}")

        with stats_cols[2]:
            st.metric("Expectancy", f"${result.expectancy:,.2f}")
            st.metric("Avg Trade Duration", f"{result.avg_trade_duration:.1f}h")
            st.metric("Recovery Factor", f"{result.recovery_factor:.2f}")

        with stats_cols[3]:
            st.metric("Consecutive Wins", result.consecutive_wins)
            st.metric("Consecutive Losses", result.consecutive_losses)
            st.metric("Calmar Ratio", f"{result.calmar_ratio:.2f}")

        # Monthly returns
        st.markdown("#### 📅 Monthly Performance")
        if not result.monthly_returns.empty:
            fig_monthly = go.Figure(data=[
                go.Bar(
                    x=result.monthly_returns['month'],
                    y=result.monthly_returns['return'],
                    marker_color=[
                        '#38ef7d' if r > 0 else '#f5576c'
                        for r in result.monthly_returns['return']
                    ],
                    text=result.monthly_returns['return'].apply(lambda x: f"{x:+.1f}%"),
                    textposition='outside'
                )
            ])
            fig_monthly.update_layout(
                title="Monthly Returns",
                xaxis_title="Month",
                yaxis_title="Return (%)",
                height=300
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

        # Trade list
        with st.expander("📋 All Trades"):
            trades_df = pd.DataFrame(result.trades)
            if not trades_df.empty:
                st.dataframe(
                    trades_df[['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price', 'pnl_percent', 'pnl_absolute']],
                    use_container_width=True
                )


# ============================================================================
# FEATURE 2: PAPER TRADING SIMULATOR
# ============================================================================

def render_paper_trading(simulator: PaperTradingSimulator, current_price: float, symbol: str):
    """
    Full paper trading interface.

    Features:
    - Place orders
    - View open positions
    - Track P&L
    - Order history
    """
    st.markdown("## 💼 Paper Trading Simulator")
    st.markdown("Practice trading with virtual money")

    # Portfolio summary
    stats = simulator.get_portfolio_stats()

    summary_cols = st.columns(5)
    summary_cols[0].metric(
        "Portfolio Value",
        f"${stats['total_value']:,.2f}",
        f"{stats['total_return_pct']:+.2f}%"
    )
    summary_cols[1].metric("Cash", f"${stats['cash']:,.2f}")
    summary_cols[2].metric(
        "Total P&L",
        f"${stats['total_pnl']:+,.2f}",
        f"{stats['total_return_pct']:+.2f}%"
    )
    summary_cols[3].metric("Open Positions", stats['num_positions'])
    summary_cols[4].metric("Win Rate", f"{stats['win_rate']:.1f}%")

    st.markdown("---")

    # Trading interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📝 Place Order")

        order_side = st.radio("Side", ["BUY", "SELL"], horizontal=True)
        order_quantity = st.number_input("Quantity", 0.001, 1000.0, 0.1, 0.001)
        order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])

        limit_price = None
        if order_type == "LIMIT":
            limit_price = st.number_input("Limit Price", 0.01, 1000000.0, current_price, 0.01)

        if st.button("🚀 Place Order", type="primary"):
            try:
                order = simulator.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY if order_side == "BUY" else OrderSide.SELL,
                    quantity=order_quantity,
                    order_type=OrderType.MARKET if order_type == "MARKET" else OrderType.LIMIT,
                    price=limit_price
                )

                if order_type == "MARKET":
                    success = simulator.execute_market_order(order, current_price)
                    if success:
                        st.success(f"✅ Order executed: {order_side} {order_quantity} @ ${current_price:,.2f}")
                    else:
                        st.error("❌ Order rejected (insufficient funds or position)")
                else:
                    st.info(f"Limit order placed: {order_side} {order_quantity} @ ${limit_price:,.2f}")

            except Exception as e:
                st.error(f"Order failed: {e}")

    with col2:
        st.markdown("### 📊 Open Positions")

        positions = simulator.get_open_positions()
        if positions:
            for pos in positions:
                with st.container():
                    pos_color = "#38ef7d" if pos.unrealized_pnl > 0 else "#f5576c"
                    st.markdown(f"""
                    <div style="border: 2px solid {pos_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <b>{pos.symbol}</b> - {pos.side.value.upper()}<br>
                        Qty: {pos.quantity} | Entry: ${pos.entry_price:,.2f}<br>
                        Current: ${pos.current_price:,.2f}<br>
                        <b style="color: {pos_color};">P&L: ${pos.unrealized_pnl:+,.2f} ({pos.unrealized_pnl_pct:+.2f}%)</b>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"Close {pos.symbol}", key=f"close_{pos.symbol}"):
                        simulator._close_position(pos.symbol, pos.current_price, "MANUAL")
                        st.rerun()
        else:
            st.info("No open positions")

    # Trade history
    st.markdown("### 📋 Trade History")
    trades = simulator.get_trade_history(50)
    if trades:
        trades_df = pd.DataFrame(trades)
        st.dataframe(
            trades_df[['symbol', 'side', 'quantity', 'entry_price', 'exit_price', 'pnl', 'pnl_pct']],
            use_container_width=True
        )
    else:
        st.info("No trade history yet")

    # Reset button
    if st.button("🔄 Reset Simulator"):
        simulator.reset()
        st.success("Simulator reset to initial capital")
        st.rerun()


# Continue in next message due to length...
