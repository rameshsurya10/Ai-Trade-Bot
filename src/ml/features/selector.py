"""
Adaptive Feature Selector
==========================

Selects optimal indicators based on:
1. Current market regime (Bull/Bear/Sideways/Volatile)
2. Feature importance from trained models
3. Research-backed indicator effectiveness

Key Principles:
- Top 10-15 features outperform using all features (less overfitting)
- Different indicators work better in different market conditions
- SHAP values provide interpretable feature importance

Research Sources:
- Neptune.ai 2024: Feature selection for crypto trading
- Springer 2024: Regime-dependent technical indicators
- MDPI 2024: SHAP-based feature selection
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


@dataclass
class IndicatorRanking:
    """Ranking result for indicators."""
    name: str
    importance: float
    rank: int
    category: str  # momentum, trend, volatility, volume, pattern
    regime_effectiveness: Dict[str, float]  # effectiveness per regime


@dataclass
class FeatureSelectionResult:
    """Result of feature selection."""
    selected_features: List[str]
    feature_rankings: List[IndicatorRanking]
    regime: str
    selection_method: str
    n_features_original: int
    n_features_selected: int


# =============================================================================
# RESEARCH-BACKED INDICATOR EFFECTIVENESS BY REGIME
# =============================================================================
# Based on academic research and backtesting studies:
# - Bull markets: Momentum indicators perform best
# - Bear markets: Volatility and reversal indicators
# - Sideways: Mean-reversion indicators (RSI, Bollinger)
# - Volatile: ATR-based, volatility indicators

REGIME_INDICATOR_EFFECTIVENESS = {
    MarketRegime.BULL: {
        # Momentum indicators excel in trends
        'macd_hist': 0.95,
        'roc_10': 0.88,
        'trend_14': 0.92,
        'adx': 0.85,
        'di_diff': 0.80,
        # Price ratios good for trend following
        'price_sma_21_ratio': 0.78,
        # Moderate effectiveness
        'rsi_14': 0.65,  # RSI less useful in strong trends
        'bb_position': 0.60,
        'volume_ratio': 0.70,
        # Lower effectiveness in trends
        'stoch_k': 0.55,
        'williams_r': 0.50,
        'cci': 0.55,
        # Candlestick patterns - continuation patterns better
        'three_white_soldiers': 0.85,  # Strong continuation
        'marubozu_bull': 0.80,
        'engulfing_bull': 0.65,  # Less useful in trend
        'pattern_signal': 0.70,
    },
    MarketRegime.BEAR: {
        # Volatility indicators for risk management
        'atr_14': 0.95,
        'volatility_14': 0.92,
        'bb_width': 0.85,
        # Reversal detection - candlestick patterns critical
        'hammer': 0.90,  # Key reversal signal
        'morning_star': 0.88,
        'engulfing_bull': 0.85,  # Reversal signal
        'doji': 0.75,  # Indecision at bottom
        'rsi_14': 0.80,  # Good for oversold detection
        'stoch_k': 0.75,
        'williams_r': 0.72,
        # Trend confirmation (bearish)
        'adx': 0.82,
        'di_diff': 0.78,
        'macd_hist': 0.75,
        # Volume for capitulation
        'volume_ratio': 0.85,
        'volume_change': 0.80,
        # Bearish continuation
        'three_black_crows': 0.82,
        'pattern_signal': 0.85,
        'lower_low': 0.70,
    },
    MarketRegime.SIDEWAYS: {
        # Mean reversion indicators excel
        'rsi_14': 0.95,
        'bb_position': 0.95,
        'stoch_k': 0.90,
        'williams_r': 0.85,
        'cci': 0.82,
        # Range-bound indicators
        'bb_width': 0.75,  # Narrow bands = range
        'atr_14': 0.70,
        # Reversal patterns work well in ranges
        'engulfing_bull': 0.88,
        'engulfing_bear': 0.88,
        'hammer': 0.85,
        'shooting_star': 0.85,
        'doji': 0.80,  # Indecision common
        'pattern_signal': 0.85,
        # Trend indicators less useful
        'adx': 0.60,  # Low ADX confirms range
        'macd_hist': 0.55,
        'trend_14': 0.50,
        # Volume for breakout detection
        'volume_ratio': 0.72,
    },
    MarketRegime.VOLATILE: {
        # Volatility management critical
        'atr_14': 0.98,
        'volatility_14': 0.95,
        'bb_width': 0.90,
        # Quick reversal patterns important
        'engulfing_bull': 0.90,
        'engulfing_bear': 0.90,
        'morning_star': 0.85,
        'evening_star': 0.85,
        'shooting_star': 0.82,
        'hammer': 0.82,
        'pattern_signal': 0.88,
        # Quick momentum for fast moves
        'roc_10': 0.85,
        'macd_hist': 0.82,
        'rsi_14': 0.78,
        # Volume for conviction
        'volume_ratio': 0.88,
        'volume_change': 0.85,
        # Pattern basics
        'candle_body_ratio': 0.75,
        'higher_high': 0.70,
        'lower_low': 0.70,
        # Trend less reliable
        'adx': 0.65,
        'trend_14': 0.55,
    }
}

# =============================================================================
# INDICATOR CATEGORIES
# =============================================================================
INDICATOR_CATEGORIES = {
    # Momentum - measures speed of price change
    'momentum': [
        'rsi_14', 'rsi_7', 'stoch_k', 'stoch_d',
        'macd', 'macd_signal', 'macd_hist',
        'roc_5', 'roc_10', 'williams_r', 'cci'
    ],
    # Trend - measures direction and strength
    'trend': [
        'adx', 'plus_di', 'minus_di', 'di_diff',
        'trend_7', 'trend_14', 'trend_21',
        'price_sma_7_ratio', 'price_sma_21_ratio', 'price_sma_50_ratio',
        'price_ema_21_ratio'
    ],
    # Volatility - measures price variability
    'volatility': [
        'atr_14', 'atr_7', 'bb_width', 'bb_position',
        'volatility_7', 'volatility_14', 'volatility_30'
    ],
    # Volume - measures trading activity
    'volume': [
        'volume_ratio', 'volume_change', 'obv', 'obv_sma', 'money_flow'
    ],
    # Price - raw price-based features
    'price': [
        'returns', 'log_returns'
    ],
    # Pattern - candlestick patterns (basic)
    'pattern': [
        'candle_body_ratio', 'candle_wick_upper', 'candle_wick_lower',
        'higher_high', 'lower_low', 'higher_close'
    ],
    # Candlestick - Japanese candlestick patterns (reversal/continuation)
    'candlestick': [
        # Single candle
        'doji', 'hammer', 'inverted_hammer', 'shooting_star', 'hanging_man',
        'marubozu_bull', 'marubozu_bear', 'spinning_top',
        # Two candle
        'engulfing_bull', 'engulfing_bear', 'harami_bull', 'harami_bear',
        'piercing_line', 'dark_cloud', 'tweezer_top', 'tweezer_bottom',
        # Three candle
        'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows',
        # Aggregates
        'bullish_patterns', 'bearish_patterns', 'pattern_signal', 'reversal_pattern'
    ]
}


def get_indicator_category(indicator: str) -> str:
    """Get category for an indicator."""
    for category, indicators in INDICATOR_CATEGORIES.items():
        if indicator in indicators:
            return category
    # Check for lagged features
    if '_lag_' in indicator:
        base = indicator.split('_lag_')[0]
        return get_indicator_category(base)
    # Check for regime/IMF features
    if indicator.startswith('regime_'):
        return 'regime'
    if indicator.startswith('imf_'):
        return 'decomposition'
    return 'other'


class AdaptiveFeatureSelector:
    """
    Selects optimal features based on market regime and importance.

    Strategy:
    1. Use research-backed defaults per regime
    2. Refine with actual feature importance from models
    3. Combine regime effectiveness + model importance
    4. Select top N features
    """

    # Default number of features per regime
    DEFAULT_N_FEATURES = {
        MarketRegime.BULL: 12,      # Fewer, trend-focused
        MarketRegime.BEAR: 15,      # More, risk-focused
        MarketRegime.SIDEWAYS: 10,  # Few, mean-reversion focused
        MarketRegime.VOLATILE: 14,  # More, volatility-focused
    }

    # Minimum features per category to ensure diversity
    MIN_CATEGORY_FEATURES = {
        'momentum': 2,
        'trend': 2,
        'volatility': 2,
        'volume': 1,
        'price': 1,
        'pattern': 1,
    }

    def __init__(
        self,
        n_features: int = 15,
        use_shap: bool = True,
        regime_weight: float = 0.4,
        importance_weight: float = 0.6
    ):
        """
        Initialize feature selector.

        Args:
            n_features: Maximum number of features to select
            use_shap: Whether to use SHAP for importance (if available)
            regime_weight: Weight for regime-based effectiveness (0-1)
            importance_weight: Weight for model importance (0-1)
        """
        self.n_features = n_features
        self.use_shap = use_shap
        self.regime_weight = regime_weight
        self.importance_weight = importance_weight

        self._feature_importance: Dict[str, float] = {}
        self._shap_values: Optional[np.ndarray] = None
        self._rankings_cache: Dict[str, List[IndicatorRanking]] = {}

    def calculate_importance(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Calculate feature importance from trained models.

        Uses:
        1. Tree-based importance (XGBoost, LightGBM, CatBoost)
        2. SHAP values (if available and use_shap=True)
        3. Average across models

        Args:
            models: Dict of model name -> trained model
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names

        Returns:
            Dict of feature name -> importance score
        """
        importance_scores = {name: [] for name in feature_names}

        for model_name, model in models.items():
            # Skip PyTorch models (no direct feature importance)
            if hasattr(model, 'parameters'):  # PyTorch model
                continue

            # Get feature importance from boosting models
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if len(importances) == len(feature_names):
                    for i, name in enumerate(feature_names):
                        importance_scores[name].append(importances[i])
                    logger.info(f"Got importance from {model_name}")

            # Try to get model's internal feature importance
            elif hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                importances = model.model.feature_importances_
                if len(importances) == len(feature_names):
                    for i, name in enumerate(feature_names):
                        importance_scores[name].append(importances[i])
                    logger.info(f"Got importance from {model_name}.model")

        # Try SHAP if available and requested
        if self.use_shap:
            shap_importance = self._calculate_shap_importance(models, X, feature_names)
            if shap_importance:
                for name, score in shap_importance.items():
                    importance_scores[name].append(score)
                logger.info("Added SHAP importance scores")

        # Average importance across all models
        final_importance = {}
        for name, scores in importance_scores.items():
            if scores:
                final_importance[name] = np.mean(scores)
            else:
                final_importance[name] = 0.0

        # Normalize to 0-1
        max_imp = max(final_importance.values()) if final_importance else 1
        if max_imp > 0:
            final_importance = {k: v / max_imp for k, v in final_importance.items()}

        self._feature_importance = final_importance
        return final_importance

    def _calculate_shap_importance(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Calculate SHAP-based feature importance."""
        try:
            import shap
        except ImportError:
            logger.debug("SHAP not installed, skipping SHAP importance")
            return None

        shap_importance = {name: [] for name in feature_names}

        for model_name, model in models.items():
            # Skip non-tree models
            if hasattr(model, 'parameters'):  # PyTorch
                continue

            try:
                # Get the underlying model
                actual_model = model.model if hasattr(model, 'model') else model

                # Use TreeExplainer for boosting models
                if hasattr(actual_model, 'feature_importances_'):
                    # Sample data for efficiency
                    sample_size = min(500, len(X))
                    X_sample = X[:sample_size]

                    explainer = shap.TreeExplainer(actual_model)
                    shap_values = explainer.shap_values(X_sample)

                    # Handle multi-output
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Class 1 (up)

                    # Mean absolute SHAP value per feature
                    mean_shap = np.abs(shap_values).mean(axis=0)

                    if len(mean_shap) == len(feature_names):
                        for i, name in enumerate(feature_names):
                            shap_importance[name].append(mean_shap[i])

                        logger.debug(f"SHAP calculated for {model_name}")

            except Exception as e:
                logger.debug(f"SHAP failed for {model_name}: {e}")
                continue

        # Average across models
        final_shap = {}
        for name, scores in shap_importance.items():
            if scores:
                final_shap[name] = np.mean(scores)

        if final_shap:
            # Normalize
            max_val = max(final_shap.values())
            if max_val > 0:
                final_shap = {k: v / max_val for k, v in final_shap.items()}

        self._shap_values = final_shap
        return final_shap if final_shap else None

    def rank_features(
        self,
        feature_names: List[str],
        regime: MarketRegime
    ) -> List[IndicatorRanking]:
        """
        Rank features for a specific regime.

        Combines:
        1. Research-backed regime effectiveness
        2. Model-derived importance

        Args:
            feature_names: List of available features
            regime: Current market regime

        Returns:
            Sorted list of IndicatorRanking (highest to lowest)
        """
        cache_key = f"{regime.value}_{len(feature_names)}"
        if cache_key in self._rankings_cache and self._feature_importance:
            return self._rankings_cache[cache_key]

        regime_effectiveness = REGIME_INDICATOR_EFFECTIVENESS.get(regime, {})
        rankings = []

        for name in feature_names:
            # Get base name for lagged features
            base_name = name.split('_lag_')[0] if '_lag_' in name else name

            # Regime effectiveness (research-backed)
            regime_score = regime_effectiveness.get(base_name, 0.5)

            # Model importance (data-driven)
            model_score = self._feature_importance.get(name, 0.0)

            # Combined score
            combined_score = (
                self.regime_weight * regime_score +
                self.importance_weight * model_score
            )

            # Get category
            category = get_indicator_category(name)

            # Calculate effectiveness across all regimes
            regime_eff = {}
            for r in MarketRegime:
                r_eff = REGIME_INDICATOR_EFFECTIVENESS.get(r, {})
                regime_eff[r.value] = r_eff.get(base_name, 0.5)

            rankings.append(IndicatorRanking(
                name=name,
                importance=combined_score,
                rank=0,  # Will be set after sorting
                category=category,
                regime_effectiveness=regime_eff
            ))

        # Sort by importance (descending)
        rankings.sort(key=lambda x: x.importance, reverse=True)

        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1

        self._rankings_cache[cache_key] = rankings
        return rankings

    def select_features(
        self,
        feature_names: List[str],
        regime: MarketRegime,
        n_features: Optional[int] = None,
        ensure_diversity: bool = True
    ) -> FeatureSelectionResult:
        """
        Select optimal features for the current regime.

        Args:
            feature_names: List of available features
            regime: Current market regime
            n_features: Number of features to select (None = use regime default)
            ensure_diversity: Ensure minimum features from each category

        Returns:
            FeatureSelectionResult with selected features
        """
        if n_features is None:
            n_features = self.DEFAULT_N_FEATURES.get(regime, self.n_features)

        # Rank all features
        rankings = self.rank_features(feature_names, regime)

        if ensure_diversity:
            selected = self._select_with_diversity(rankings, n_features, regime)
        else:
            selected = [r.name for r in rankings[:n_features]]

        return FeatureSelectionResult(
            selected_features=selected,
            feature_rankings=rankings,
            regime=regime.value,
            selection_method='adaptive' if self._feature_importance else 'regime_default',
            n_features_original=len(feature_names),
            n_features_selected=len(selected)
        )

    def _select_with_diversity(
        self,
        rankings: List[IndicatorRanking],
        n_features: int,
        regime: MarketRegime
    ) -> List[str]:
        """Select features ensuring category diversity."""
        selected = []
        selected_by_category = {cat: 0 for cat in INDICATOR_CATEGORIES.keys()}
        selected_by_category['regime'] = 0
        selected_by_category['decomposition'] = 0
        selected_by_category['other'] = 0

        # First pass: ensure minimum per category
        for ranking in rankings:
            if len(selected) >= n_features:
                break

            cat = ranking.category
            min_required = self.MIN_CATEGORY_FEATURES.get(cat, 0)

            if selected_by_category.get(cat, 0) < min_required:
                selected.append(ranking.name)
                selected_by_category[cat] = selected_by_category.get(cat, 0) + 1

        # Second pass: fill remaining with top-ranked
        for ranking in rankings:
            if len(selected) >= n_features:
                break
            if ranking.name not in selected:
                selected.append(ranking.name)

        return selected

    def print_rankings(
        self,
        feature_names: List[str],
        regime: MarketRegime,
        top_n: int = 20
    ) -> str:
        """
        Print feature rankings in a formatted table.

        Returns formatted string for display.
        """
        rankings = self.rank_features(feature_names, regime)

        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"FEATURE IMPORTANCE RANKING - {regime.value.upper()} REGIME")
        lines.append(f"{'='*60}")
        lines.append(f"{'Rank':<6} {'Feature':<25} {'Score':<10} {'Category':<12}")
        lines.append(f"{'-'*60}")

        for ranking in rankings[:top_n]:
            bar_len = int(ranking.importance * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            lines.append(
                f"{ranking.rank:<6} {ranking.name:<25} {ranking.importance:.4f}    {ranking.category:<12}"
            )
            lines.append(f"       {bar}")

        lines.append(f"{'='*60}")
        lines.append(f"Total features: {len(rankings)}")
        lines.append(f"Recommended for {regime.value}: {self.DEFAULT_N_FEATURES.get(regime, 12)}")

        return '\n'.join(lines)


# =============================================================================
# STANDARD FEATURE SETS
# =============================================================================
# Pre-defined feature sets for quick deployment

STANDARD_FEATURE_SETS = {
    'minimal': [
        # 10 essential features for any market
        'rsi_14', 'macd_hist', 'bb_position', 'atr_14',
        'adx', 'volume_ratio', 'trend_14', 'returns',
        'pattern_signal', 'engulfing_bull'  # Key candlestick
    ],
    'balanced': [
        # 15 balanced features - RECOMMENDED
        'rsi_14', 'macd_hist', 'bb_position', 'bb_width', 'atr_14',
        'adx', 'di_diff', 'stoch_k', 'volume_ratio',
        'trend_14', 'returns', 'roc_10',
        # Candlestick patterns
        'pattern_signal', 'engulfing_bull', 'engulfing_bear'
    ],
    'with_patterns': [
        # 20 features with full candlestick patterns
        'rsi_14', 'macd_hist', 'bb_position', 'atr_14',
        'adx', 'di_diff', 'stoch_k', 'volume_ratio',
        'trend_14', 'returns', 'volatility_14',
        # Candlestick patterns
        'doji', 'hammer', 'shooting_star',
        'engulfing_bull', 'engulfing_bear',
        'morning_star', 'evening_star',
        'pattern_signal', 'candle_body_ratio'
    ],
    'trend_following': [
        # Best for Bull/Bear trending markets
        'macd_hist', 'adx', 'di_diff',
        'trend_14', 'roc_10', 'price_sma_21_ratio',
        'atr_14', 'volume_ratio', 'returns',
        # Continuation patterns
        'three_white_soldiers', 'three_black_crows',
        'marubozu_bull', 'marubozu_bear', 'pattern_signal'
    ],
    'mean_reversion': [
        # Best for Sideways/Ranging markets
        'rsi_14', 'bb_position', 'stoch_k',
        'williams_r', 'cci', 'bb_width', 'atr_14',
        'volume_ratio', 'returns',
        # Reversal patterns
        'engulfing_bull', 'engulfing_bear',
        'hammer', 'shooting_star', 'doji', 'pattern_signal'
    ],
    'volatility_focused': [
        # Best for Volatile markets
        'atr_14', 'bb_width', 'volatility_14',
        'roc_10', 'macd_hist', 'rsi_14',
        'volume_ratio', 'volume_change',
        # Quick reversal patterns
        'engulfing_bull', 'engulfing_bear',
        'morning_star', 'evening_star',
        'hammer', 'shooting_star', 'pattern_signal'
    ]
}


def get_standard_features(set_name: str = 'balanced') -> List[str]:
    """Get a standard feature set by name."""
    return STANDARD_FEATURE_SETS.get(set_name, STANDARD_FEATURE_SETS['balanced'])


def get_features_for_regime(regime: str) -> List[str]:
    """Get recommended features for a market regime."""
    regime_mapping = {
        'bull': 'trend_following',
        'bear': 'volatility_focused',
        'sideways': 'mean_reversion',
        'volatile': 'volatility_focused',
        'choppy': 'mean_reversion',
    }
    set_name = regime_mapping.get(regime.lower(), 'balanced')
    return STANDARD_FEATURE_SETS.get(set_name, STANDARD_FEATURE_SETS['balanced'])
