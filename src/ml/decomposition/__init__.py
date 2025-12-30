"""
Signal Decomposition Module
============================

Implements SVMD (Successive Variational Mode Decomposition) for
signal preprocessing. Research shows 73% RMSE reduction vs raw LSTM.

Sources:
- Journal of Big Data 2025
- Wiley Journal of Probability and Statistics 2025
"""

from .svmd import SVMDDecomposer

__all__ = ['SVMDDecomposer']
