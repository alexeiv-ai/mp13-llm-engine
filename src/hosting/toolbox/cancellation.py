"""Toolbox cancellation detection helpers."""
from __future__ import annotations

from typing import Any


def is_canceled_tool_error(tool_call: Any) -> bool:
    if isinstance(tool_call, dict):
        error_text = str(tool_call.get("error") or "").strip().lower()
    else:
        error_text = str(getattr(tool_call, "error", "") or "").strip().lower()
    return error_text == "canceled" or error_text.startswith("execution canceled:")


def should_resubmit_canceled_tool_call(
    tool_call: Any,
    *,
    non_restartable: bool = False,
) -> bool:
    return is_canceled_tool_error(tool_call) and not bool(non_restartable)


def _is_coarse_cancel_execution_error(exc: BaseException) -> bool:
    message = str(exc or "").strip().lower()
    if not message:
        return False
    cancel_markers = (
        "toolbox_executor_missing",
        "engine_not_found",
        "no output",
        "connection reset",
        "broken pipe",
        "end of file",
        "eoferror",
        "worker_exception",
    )
    return any(marker in message for marker in cancel_markers)
