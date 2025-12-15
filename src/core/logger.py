"""
Logging Configuration
=====================
Centralized logging setup with file rotation.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Global logger cache
_loggers = {}


def setup_logger(
    name: str = "trading_bot",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size_mb: int = 10,
    backup_count: int = 3,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        max_size_mb: Max log file size before rotation
        backup_count: Number of backup files to keep
        console: Whether to log to console

    Returns:
        Configured logger
    """
    # Check cache
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers

    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    # Cache
    _loggers[name] = logger

    return logger


def get_logger(name: str = "trading_bot") -> logging.Logger:
    """
    Get a logger by name.

    If the logger doesn't exist, returns a basic logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]

    # Return basic logger if not set up
    return logging.getLogger(name)


def silence_external_loggers():
    """Silence noisy external libraries."""
    noisy_loggers = [
        'urllib3',
        'ccxt',
        'asyncio',
        'websocket',
        'tornado',
        'streamlit',
    ]

    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
