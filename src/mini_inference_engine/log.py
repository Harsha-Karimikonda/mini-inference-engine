"""Centralized logging for Mini-Together.

Logging is intentionally configured by the application entry point so importing
the package does not unexpectedly change the host application's logging setup.
Set ``MINI_LOG_LEVEL`` (for example, ``DEBUG``) and ``MINI_LOG_FORMAT=json``
to customize the output.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


LOGGER_NAME = "mini_inference_engine"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key in ("endpoint", "request_id", "worker", "status", "duration_ms", "queue_size"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging() -> logging.Logger:
    """Configure and return the application logger.

    The handler is installed once, making this safe when tests or ASGI servers
    create more than one application instance in the same process.
    """

    logger = logging.getLogger(LOGGER_NAME)
    level_name = os.getenv("MINI_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    if not any(getattr(handler, "_mini_together_handler", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler._mini_together_handler = True  # type: ignore[attr-defined]
        if os.getenv("MINI_LOG_FORMAT", "text").lower() == "json":
            handler.setFormatter(_JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger


def get_logger(component: str | None = None) -> logging.Logger:
    """Return a logger under the application's namespace."""

    return logging.getLogger(f"{LOGGER_NAME}.{component}" if component else LOGGER_NAME)
