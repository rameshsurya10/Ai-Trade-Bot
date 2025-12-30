"""
Regime Detection Module
========================

Implements GMM-HMM (Gaussian Mixture Hidden Markov Model) for
market regime detection. Best performer 2006-2023 in research.

Regimes:
- BULL: Trending up, follow momentum
- BEAR: Trending down, defensive
- SIDEWAYS: Mean-reverting, range trading

Sources:
- QuantStart: HMM for Regime Detection
- LSEG Developer Portal: Market Regime Detection
"""

from .detector import RegimeDetector, MarketRegime

__all__ = ['RegimeDetector', 'MarketRegime']
