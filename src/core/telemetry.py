from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

LOGGER_NAME = "regulated_ai"


def configure_json_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def pseudonymous_customer_key(customer_id: str) -> str:
    """Return a short one-way key so raw customer identifiers are not written to logs."""
    return hashlib.sha256(customer_id.encode("utf-8")).hexdigest()[:16]


def log_event(event: str, **fields: Any) -> None:
    if os.getenv("DISABLE_STRUCTURED_LOGGING") == "1":
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    configure_json_logging().info(json.dumps(payload, sort_keys=True, default=str))
