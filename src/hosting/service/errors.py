"""Service-specific exception types."""
from __future__ import annotations

from typing import Any, Dict, Optional


class ToolboxRolloutError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "toolbox_rollout_failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(str(message or code))
        self.code = str(code or "toolbox_rollout_failed")
        self.details = dict(details or {})

    def to_error_payload(self) -> Dict[str, Any]:
        return {
            "error": "toolbox_rollout_failed",
            "error_code": self.code,
            "error_details": dict(self.details or {}),
        }

