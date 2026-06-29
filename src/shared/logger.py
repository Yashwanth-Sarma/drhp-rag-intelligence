"""
src/shared/logger.py

Structured logging for every FinSight module.
Every log entry includes: timestamp, module name, operation, status, duration.

Usage:
    from src.shared.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Chunks embedded", extra={"count": 42, "duration_ms": 312})
"""

import logging
import sys
import time
from functools import wraps
from typing import Callable

import json


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON.
    Makes logs parseable by tools like Datadog, CloudWatch, LangSmith.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        # Include any extra fields passed via extra={}
        for key, value in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName"
            ):
                log_entry[key] = value

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.
    
    Args:
        name: Module name — use __name__ in every module.
    
    Returns:
        Configured Logger instance.
    
    Example:
        logger = get_logger(__name__)
        logger.info("Pipeline started")
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.INFO)

    # Console handler — readable format for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler — JSON format for production parsing
    try:
        file_handler = logging.FileHandler("logs/finsight.log", mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    except FileNotFoundError:
        # logs/ directory doesn't exist yet — create it
        import os
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/finsight.log", mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def log_duration(operation_name: str) -> Callable:
    """
    Decorator that logs execution time of any function.
    
    Usage:
        @log_duration("embed_chunks")
        def embed_chunks(chunks):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = round((time.time() - start) * 1000)
                logger.info(
                    f"{operation_name} completed",
                    extra={"operation": operation_name, "duration_ms": duration_ms, "status": "success"}
                )
                return result
            except Exception as e:
                duration_ms = round((time.time() - start) * 1000)
                logger.error(
                    f"{operation_name} failed: {e}",
                    extra={"operation": operation_name, "duration_ms": duration_ms, "status": "error"}
                )
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Logger test — INFO level")
    logger.warning("Logger test — WARNING level")
    logger.error("Logger test — ERROR level")
    print("Logger working. Check logs/finsight.log for JSON output.")