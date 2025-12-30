"""
Unbreakable Trading System - ML Module
=======================================

Research-backed architecture for robust trading predictions.

ARCHITECTURE LAYERS:
1. SVMD Signal Decomposition (73% RMSE reduction)
2. GMM-HMM Regime Detection (Best 2006-2023)
3. Base Models: TCN-LSTM-Attention + XGBoost + LightGBM
4. Stacking Meta-Learner (XGBoost/Ridge)
5. Risk Management (NGBoost + Fractional Kelly)
6. Continuous Learning (EWC + Concept Drift Detection)

Based on 50+ academic papers from 2024-2025.
"""

from .decomposition import SVMDDecomposer
from .regime import RegimeDetector
from .models import TCNLSTMAttention, create_base_models
from .ensemble import StackingEnsemble
from .risk import RiskManager
from .learning import ContinualLearner
from .features import FeatureEngineer
from .predictor import UnbreakablePredictor

__all__ = [
    'SVMDDecomposer',
    'RegimeDetector',
    'TCNLSTMAttention',
    'create_base_models',
    'StackingEnsemble',
    'RiskManager',
    'ContinualLearner',
    'FeatureEngineer',
    'UnbreakablePredictor'
]

__version__ = '2.0.0'
