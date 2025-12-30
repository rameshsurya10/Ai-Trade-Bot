"""
Feature Engineering Module
===========================

Comprehensive feature engineering for trading:
1. Technical indicators (RSI, MACD, BB, ATR, etc.)
2. SVMD decomposed features (IMFs)
3. Regime features (from HMM)
4. Lagged features
5. Volume features
6. Volatility features

Research shows feature engineering is often more important than model choice.
SHAP-based feature selection helps identify the most predictive features.
"""

from .engineer import FeatureEngineer

__all__ = ['FeatureEngineer']
