"""
Continuous Learning Components
===============================

Components for continuous learning, confidence gating, and outcome tracking.

Main Components:
- ContinuousLearningSystem: Main orchestrator
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

__all__ = [
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
