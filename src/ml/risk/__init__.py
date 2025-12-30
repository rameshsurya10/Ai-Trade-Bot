"""
Risk Management Module
=======================

Implements ML-based risk management:
1. NGBoost for uncertainty quantification
2. Fractional Kelly criterion for position sizing
3. Dynamic stop-loss/take-profit using LSTM
4. Conformal prediction for confidence intervals

Research shows proper risk management is MORE important than prediction accuracy.
"""

from .manager import RiskManager, PositionSizer, DynamicStopLoss

__all__ = ['RiskManager', 'PositionSizer', 'DynamicStopLoss']
