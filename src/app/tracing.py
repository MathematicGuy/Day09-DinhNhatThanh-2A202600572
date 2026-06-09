from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def make_trace_event(
    *,
    run_id: str,
    case_id: str | None,
    node: str,
    event: str,
    status: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
    warnings: list[str] | None = None,
    error: str | None = None,
    latency_ms: int = 0,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "case_id": case_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": node,
        "event": event,
        "status": status,
        "input": json_safe(input_payload),
        "output": json_safe(output_payload),
        "warnings": warnings or [],
        "error": error,
        "latency_ms": latency_ms,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
