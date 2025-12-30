"""
Continuous Learning Module
===========================

Implements techniques to keep models adaptive without catastrophic forgetting:

1. Elastic Weight Consolidation (EWC) - Preserve important weights
2. Experience Replay - Remember past patterns
3. Concept Drift Detection - Detect when market changes
4. Online Learning - Incremental updates

Sources:
- Springer 2025: Dynamic Neuroplastic Networks for Finance
- arXiv 2024: MetaDA for Stock Trends
- ScienceDirect 2024: Incremental RL + SSL for Trading
"""

from .continual import ContinualLearner, EWC, ConceptDriftDetector

__all__ = ['ContinualLearner', 'EWC', 'ConceptDriftDetector']
