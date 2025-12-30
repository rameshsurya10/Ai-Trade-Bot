"""
Ensemble Module
================

Implements stacking ensemble with meta-learner.

Stacking combines base model predictions using a meta-learner,
achieving better performance than any single model.

Research shows XGBoost or Ridge as meta-learner prevents overfitting.
"""

from .stacking import StackingEnsemble, RegimeAwareEnsemble

__all__ = ['StackingEnsemble', 'RegimeAwareEnsemble']
