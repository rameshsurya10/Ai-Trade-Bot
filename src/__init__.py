"""
AI Trade Bot - Professional Trading Signal System
=================================================

A production-ready trading signal system with ML predictions.

Modules:
--------
- core: Configuration, database, types, logging
- data: WebSocket and polling data feeds
- analysis_engine: Feature engineering and ML predictions
- multi_currency_system: Multi-currency trading with auto-learning
- signal_service: Signal filtering and management
- notifier: Multi-channel notifications
- math_engine: Advanced mathematical algorithms
- advanced_predictor: Fourier, Kalman, Monte Carlo predictions

Usage:
------
    from src import DataService, AnalysisEngine, SignalService
    from src.core import Config, Database
    from src.multi_currency_system import MultiCurrencySystem

Quick Start:
------------
    # Run analysis
    python run_analysis.py

    # Start dashboard
    streamlit run dashboard.py
"""

__version__ = "2.1.0"
__author__ = "AI Trade Bot"

# Core services (backward compatible)
from .data_service import DataService
from .analysis_engine import AnalysisEngine, FeatureCalculator, LSTMModel
from .signal_service import SignalService
from .notifier import Notifier

# Multi-currency system with auto-learning
from .multi_currency_system import MultiCurrencySystem, CurrencyConfig, PerformanceStats

# Utilities (kept for backward compatibility)
from .utils import (
    load_config,
    get_config_value,
    get_db_connection,
    get_db_path,
    get_project_root,
    get_data_config,
    get_signal_config,
    get_model_config,
    get_dashboard_config,
    get_notification_config,
    get_analysis_config
)

__all__ = [
    # Services
    "DataService",
    "AnalysisEngine",
    "FeatureCalculator",
    "LSTMModel",
    "SignalService",
    "Notifier",
    # Multi-currency
    "MultiCurrencySystem",
    "CurrencyConfig",
    "PerformanceStats",
    # Utils
    "load_config",
    "get_config_value",
    "get_db_connection",
    "get_db_path",
    "get_project_root",
    "get_data_config",
    "get_signal_config",
    "get_model_config",
    "get_dashboard_config",
    "get_notification_config",
    "get_analysis_config",
]
