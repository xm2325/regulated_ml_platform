from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.core.telemetry import log_event, pseudonymous_customer_key


def emit_decision_audit(record: dict[str, Any]) -> None:
    """Emit a redacted audit event to stdout and optionally append it to a local JSONL sink.

    A real deployment should send this event to an immutable external audit store. The local
    sink exists only for development and demonstration.
    """
    safe = dict(record)
    customer_id = str(safe.pop("customer_id", ""))
    safe["customer_key"] = pseudonymous_customer_key(customer_id)
    log_event("decision_audit", **safe)

    if settings.audit_log_path:
        path = Path(settings.audit_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe, sort_keys=True, default=str) + "\n")
