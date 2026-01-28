"""
Shared Utilities
================
Common functions used across all modules.
Prevents code duplication and ensures consistency.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager

import yaml

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Cache for config to avoid repeated file reads
_config_cache: Optional[dict] = None
_config_path: Optional[Path] = None


def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


def load_config(config_path: Optional[str] = None, force_reload: bool = False) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file (default: config.yaml in project root)
        force_reload: Force reload even if cached

    Returns:
        Configuration dictionary
    """
    global _config_cache, _config_path

    # Determine config path
    if config_path:
        path = Path(config_path)
    else:
        path = PROJECT_ROOT / "config.yaml"

    # Return cached if available and same path
    if not force_reload and _config_cache and _config_path == path:
        return _config_cache

    # Load config
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}

    with open(path, 'r') as f:
        _config_cache = yaml.safe_load(f)
        _config_path = path

    return _config_cache


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a nested config value using dot notation.

    Args:
        key_path: Dot-separated path like 'data.symbol' or 'analysis.min_confidence'
        default: Default value if key not found

    Returns:
        Config value or default

    Example:
        >>> get_config_value('data.symbol', 'BTC-USD')
        'BTC-USD'
        >>> get_config_value('signals.risk_per_trade', 0.02)
        0.02
    """
    config = load_config()
    keys = key_path.split('.')

    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def get_db_path() -> Path:
    """Get database path from config."""
    db_path_str = get_config_value('database.path', 'data/trading.db')
    return PROJECT_ROOT / db_path_str


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Ensures connections are properly closed.

    Usage:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM candles")
    """
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_model_path() -> Path:
    """Get model path from config."""
    model_path_str = get_config_value('model.path', 'models/best_model.pt')
    return PROJECT_ROOT / model_path_str


def get_dashboard_config() -> dict:
    """Get dashboard configuration."""
    return {
        'host': get_config_value('dashboard.host', 'localhost'),
        'port': get_config_value('dashboard.port', 8501),
        'theme': get_config_value('dashboard.theme', 'dark')
    }


def get_data_config() -> dict:
    """Get data collection configuration."""
    return {
        'symbol': get_config_value('data.symbol', 'BTC-USD'),
        'exchange': get_config_value('data.exchange', 'coinbase'),
        'interval': get_config_value('data.interval', '1h'),
        'history_days': get_config_value('data.history_days', 365)
    }


def get_analysis_config() -> dict:
    """Get analysis configuration."""
    return {
        'update_interval': get_config_value('analysis.update_interval', 60),
        'min_confidence': get_config_value('analysis.min_confidence', 0.55),
        'lookback_period': get_config_value('analysis.lookback_period', 100)
    }


def get_signal_config() -> dict:
    """Get signal configuration."""
    return {
        'risk_per_trade': get_config_value('signals.risk_per_trade', 0.02),
        'risk_reward_ratio': get_config_value('signals.risk_reward_ratio', 2.0),
        'strong_signal': get_config_value('signals.strong_signal', 0.65),
        'medium_signal': get_config_value('signals.medium_signal', 0.55),
        'cooldown_minutes': get_config_value('signals.cooldown_minutes', 60)
    }


def get_notification_config() -> dict:
    """Get notification configuration."""
    return {
        'desktop': get_config_value('notifications.desktop', True),
        'sound': get_config_value('notifications.sound', True),
        'sound_file': get_config_value('notifications.sound_file', ''),
        'telegram': {
            'enabled': get_config_value('notifications.telegram.enabled', False),
            'bot_token': get_config_value('notifications.telegram.bot_token', ''),
            'chat_id': get_config_value('notifications.telegram.chat_id', '')
        }
    }


def get_model_config() -> dict:
    """Get model configuration."""
    return {
        'path': get_config_value('model.path', 'models/best_model.pt'),
        'sequence_length': get_config_value('model.sequence_length', 60),
        'hidden_size': get_config_value('model.hidden_size', 128),
        'num_layers': get_config_value('model.num_layers', 2),
        'dropout': get_config_value('model.dropout', 0.2)
    }


# For testing
if __name__ == "__main__":
    print("Project Root:", get_project_root())
    print("DB Path:", get_db_path())
    print("Model Path:", get_model_path())
    print("\nData Config:", get_data_config())
    print("Signal Config:", get_signal_config())
    print("Dashboard Config:", get_dashboard_config())
