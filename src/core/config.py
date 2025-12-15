"""
Configuration Management
========================
Centralized configuration with validation and defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class DataConfig:
    """Data collection settings."""
    symbol: str = "BTC-USD"
    exchange: str = "coinbase"
    interval: str = "1h"
    history_days: int = 365


@dataclass
class AnalysisConfig:
    """Analysis engine settings."""
    update_interval: int = 60  # seconds
    min_confidence: float = 0.55
    lookback_period: int = 100


@dataclass
class ModelConfig:
    """ML model settings."""
    path: str = "models/best_model.pt"
    sequence_length: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2


@dataclass
class SignalConfig:
    """Signal generation settings."""
    risk_per_trade: float = 0.02
    risk_reward_ratio: float = 2.0
    strong_signal: float = 0.65
    medium_signal: float = 0.55
    cooldown_minutes: int = 60


@dataclass
class NotificationConfig:
    """
    Notification settings.

    SECURITY: Telegram credentials should be set via environment variables:
        TELEGRAM_BOT_TOKEN
        TELEGRAM_CHAT_ID
    """
    desktop: bool = True
    sound: bool = True
    sound_file: Optional[str] = None
    telegram_enabled: bool = False
    # These are loaded from environment variables for security
    _telegram_bot_token: str = field(default="", repr=False)
    _telegram_chat_id: str = field(default="", repr=False)

    @property
    def telegram_bot_token(self) -> str:
        """Get Telegram bot token from environment variable (secure)."""
        import os
        return os.getenv('TELEGRAM_BOT_TOKEN', self._telegram_bot_token)

    @property
    def telegram_chat_id(self) -> str:
        """Get Telegram chat ID from environment variable (secure)."""
        import os
        return os.getenv('TELEGRAM_CHAT_ID', self._telegram_chat_id)


@dataclass
class DatabaseConfig:
    """Database settings."""
    path: str = "data/trading.db"


@dataclass
class LoggingConfig:
    """Logging settings."""
    level: str = "INFO"
    file: str = "data/trading.log"
    max_size_mb: int = 10
    backup_count: int = 3


@dataclass
class Config:
    """
    Main configuration class.

    Loads from YAML file with sensible defaults.
    All settings are validated on load.
    """
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    _config_path: Optional[Path] = field(default=None, repr=False)

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "Config":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            Config instance with loaded values
        """
        path = Path(config_path)

        if not path.exists():
            # Return defaults if no config file
            return cls()

        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        config = cls()
        config._config_path = path

        # Load each section
        if 'data' in data:
            config.data = DataConfig(**data['data'])

        if 'analysis' in data:
            config.analysis = AnalysisConfig(**data['analysis'])

        if 'model' in data:
            config.model = ModelConfig(**data['model'])

        if 'signals' in data:
            config.signals = SignalConfig(**data['signals'])

        if 'notifications' in data:
            notif = data['notifications']
            telegram = notif.get('telegram', {})
            # Note: telegram credentials are loaded from environment variables
            # for security - see NotificationConfig properties
            config.notifications = NotificationConfig(
                desktop=notif.get('desktop', True),
                sound=notif.get('sound', True),
                sound_file=notif.get('sound_file'),
                telegram_enabled=telegram.get('enabled', False),
                _telegram_bot_token=telegram.get('bot_token', ''),
                _telegram_chat_id=telegram.get('chat_id', ''),
            )

        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])

        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])

        # Validate
        config.validate()

        return config

    def validate(self):
        """Validate configuration values."""
        # Confidence thresholds
        if not 0 < self.analysis.min_confidence < 1:
            raise ValueError(f"min_confidence must be 0-1, got {self.analysis.min_confidence}")

        if not 0 < self.signals.strong_signal < 1:
            raise ValueError(f"strong_signal must be 0-1, got {self.signals.strong_signal}")

        if self.signals.medium_signal >= self.signals.strong_signal:
            raise ValueError("medium_signal must be less than strong_signal")

        # Risk settings
        if not 0 < self.signals.risk_per_trade < 0.5:
            raise ValueError(f"risk_per_trade must be 0-0.5, got {self.signals.risk_per_trade}")

        if self.signals.risk_reward_ratio < 1:
            raise ValueError(f"risk_reward_ratio must be >= 1, got {self.signals.risk_reward_ratio}")

        # Model settings
        if self.model.sequence_length < 10:
            raise ValueError(f"sequence_length must be >= 10, got {self.model.sequence_length}")

    def get_model_path(self) -> Path:
        """Get absolute path to model file."""
        return Path(self.model.path)

    def get_db_path(self) -> Path:
        """Get absolute path to database file."""
        path = Path(self.database.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_log_path(self) -> Path:
        """Get absolute path to log file."""
        path = Path(self.logging.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data': {
                'symbol': self.data.symbol,
                'exchange': self.data.exchange,
                'interval': self.data.interval,
                'history_days': self.data.history_days,
            },
            'analysis': {
                'update_interval': self.analysis.update_interval,
                'min_confidence': self.analysis.min_confidence,
                'lookback_period': self.analysis.lookback_period,
            },
            'model': {
                'path': self.model.path,
                'sequence_length': self.model.sequence_length,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
            },
            'signals': {
                'risk_per_trade': self.signals.risk_per_trade,
                'risk_reward_ratio': self.signals.risk_reward_ratio,
                'strong_signal': self.signals.strong_signal,
                'medium_signal': self.signals.medium_signal,
                'cooldown_minutes': self.signals.cooldown_minutes,
            },
            'notifications': {
                'desktop': self.notifications.desktop,
                'sound': self.notifications.sound,
                'telegram_enabled': self.notifications.telegram_enabled,
            },
            'database': {
                'path': self.database.path,
            },
            'logging': {
                'level': self.logging.level,
                'file': self.logging.file,
            },
        }
