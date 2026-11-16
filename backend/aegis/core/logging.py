# backend/aegis/core/logging.py
"""Structured-logging initialisation.

Wraps *structlog* with a consistent configuration used by every
module in the application.  Call ``setup_logging()`` once at
application startup; use ``get_logger(__name__)`` everywhere else.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(*, level: int = logging.INFO) -> None:
    """Configure structlog (and the stdlib ``logging`` bridge).

    Call this once at application startup, typically from
    ``main.py``.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route structlog through the stdlib logging framework so that
    # third-party libraries (e.g. uvicorn) are captured too.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for the caller's module."""
    return structlog.get_logger(name or __name__)
