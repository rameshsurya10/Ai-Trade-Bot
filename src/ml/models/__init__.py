"""
Base Models Module
==================

Implements diverse base learners for the ensemble:
1. TCN-LSTM-Attention: Deep learning with temporal convolutions + recurrence + attention
2. XGBoost: Gradient boosting for tabular features
3. LightGBM: Fast gradient boosting for large data
4. CatBoost: Handles categorical features well

Research shows diverse base learners improve ensemble performance.
"""

from .tcn_lstm import TCNLSTMAttention
from .boosting import XGBoostModel, LightGBMModel, CatBoostModel
from .factory import create_base_models

__all__ = [
    'TCNLSTMAttention',
    'XGBoostModel',
    'LightGBMModel',
    'CatBoostModel',
    'create_base_models'
]
