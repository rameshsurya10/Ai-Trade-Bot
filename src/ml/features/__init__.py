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

NEW: Adaptive Feature Selection
- Research shows top 10-15 features outperform using all features
- Different indicators work better in different market regimes
- SHAP-based feature importance for interpretability
"""

from .engineer import FeatureEngineer
from .selector import (
    AdaptiveFeatureSelector,
    MarketRegime,
    IndicatorRanking,
    FeatureSelectionResult,
    get_features_for_regime,
    get_standard_features,
    STANDARD_FEATURE_SETS,
    REGIME_INDICATOR_EFFECTIVENESS,
    INDICATOR_CATEGORIES
)

__all__ = [
    'FeatureEngineer',
    'AdaptiveFeatureSelector',
    'MarketRegime',
    'IndicatorRanking',
    'FeatureSelectionResult',
    'get_features_for_regime',
    'get_standard_features',
    'STANDARD_FEATURE_SETS',
    'REGIME_INDICATOR_EFFECTIVENESS',
    'INDICATOR_CATEGORIES'
]
