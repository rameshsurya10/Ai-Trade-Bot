"""
Continuous Learning Components
===============================

Components for continuous learning, confidence gating, and outcome tracking.

Main Components:
- PerformanceBasedLearner: NEW - Per-candle learning (profit=reinforce, loss=retrain)
- ContinuousLearningSystem: Main orchestrator for multi-timeframe trading
- ConfidenceGate: 80% confidence threshold gating
- OutcomeTracker: Trade outcome monitoring and retraining triggers
- RetrainingEngine: Adaptive retraining until confidence ≥ 80%
- LearningStateManager: LEARNING ↔ TRADING mode management
"""

from .confidence_gate import ConfidenceGate, ConfidenceGateConfig, create_confidence_gate
from .state_manager import LearningStateManager
from .outcome_tracker import OutcomeTracker
from .retraining_engine import RetrainingEngine
from .continuous_learner import ContinuousLearningSystem
from .performance_learner import (
    PerformanceBasedLearner,
    PerformanceLearnerConfig,
    RetrainLevel,
    CandleOutcome,
    LearningState,
    create_performance_learner
)

__all__ = [
    # Performance-based learning (NEW)
    'PerformanceBasedLearner',
    'PerformanceLearnerConfig',
    'RetrainLevel',
    'CandleOutcome',
    'LearningState',
    'create_performance_learner',

    # Main orchestrator
    'ContinuousLearningSystem',

    # Core components
    'ConfidenceGate',
    'ConfidenceGateConfig',
    'create_confidence_gate',
    'LearningStateManager',
    'OutcomeTracker',
    'RetrainingEngine'
]
