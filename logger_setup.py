"""
logger_setup.py

Centralized logging configuration for the AI Object Detection System.

This module sets up a single, reusable logger that writes messages to
both the console (with simple color coding for readability) and to a
persistent log file on disk. Every other module in this project should
import `get_logger()` from here instead of creating its own logger,
so that all log output stays consistent and ends up in one place.
"""

import logging
import os
from datetime import datetime

import config


class ColoredFormatter(logging.Formatter):
    """
    A custom logging formatter that adds ANSI color codes to console
    output based on the severity level of the log message.

    Colors are only cosmetic (they make terminal output easier to scan)
    and have no effect on what gets written to the log file, since the
    file handler uses a separate, plain-text formatter.
    """

    # ANSI escape codes for terminal colors.
    COLORS: dict[str, str] = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET: str = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Apply a color to the log level name before formatting."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(name: str = "ObjectDetectionSystem") -> logging.Logger:
    """
    Create and configure a logger instance.

    The logger writes:
        - Colored, human-friendly messages to the console.
        - Plain, timestamped messages to a persistent log file
          (config.LOG_FILE_PATH).

    Args:
        name: The name assigned to the logger. Using a fixed default
            name means repeated calls to get_logger() across different
            modules return loggers that behave consistently.

    Returns:
        A configured logging.Logger instance ready to use.
    """
    # Ensure the logs directory exists before attaching a file handler.
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger() is called multiple
    # times (e.g. imported from several modules in the same run).
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # --- Console handler (colored) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # --- File handler (plain text, persistent) ---
    file_handler = logging.FileHandler(config.LOG_FILE_PATH, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def log_detection_event(logger: logging.Logger, image_name: str, object_count: int) -> None:
    """
    Convenience helper to log a standardized message whenever a
    detection run completes.

    Args:
        logger: The logger instance to write the message to.
        image_name: The filename of the image that was processed.
        object_count: The number of objects detected in that image.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(
        f"Detection completed on '{image_name}' at {timestamp} "
        f"-> {object_count} object(s) found."
    )
