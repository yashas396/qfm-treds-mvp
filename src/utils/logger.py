"""
QGAI Quantum Financial Modeling - TReDS MVP
Logging Configuration

This module provides centralized logging configuration using loguru.

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False


def setup_logger(
    name: str = "treds_fraud_detection",
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "1 week"
) -> "logger":
    """
    Set up and configure the logger.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to save logs to file
        log_dir: Directory for log files
        rotation: Log rotation size/time
        retention: Log retention period

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("my_app", level="DEBUG")
        >>> logger.info("Application started")
    """
    if LOGURU_AVAILABLE:
        return _setup_loguru(name, level, log_to_file, log_dir, rotation, retention)
    else:
        return _setup_standard_logging(name, level, log_to_file, log_dir)


def _setup_loguru(
    name: str,
    level: str,
    log_to_file: bool,
    log_dir: Optional[Path],
    rotation: str,
    retention: str
) -> "logger":
    """Configure loguru logger."""
    # Remove default handler
    logger.remove()

    # Console handler with formatting
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
    )

    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    return logger


def _setup_standard_logging(
    name: str,
    level: str,
    log_to_file: bool,
    log_dir: Optional[Path]
) -> logging.Logger:
    """Fallback to standard logging if loguru not available."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level))
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "treds_fraud_detection") -> "logger":
    """
    Get a logger instance.

    Args:
        name: Logger name (module name)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    if LOGURU_AVAILABLE:
        return logger.bind(name=name)
    else:
        return logging.getLogger(name)


# Create default logger on import
default_logger = setup_logger(log_to_file=False)


if __name__ == "__main__":
    # Test logging
    test_logger = setup_logger("test", level="DEBUG")
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
