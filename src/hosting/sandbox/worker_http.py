from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


RpcInvoker = Callable[[str, Dict[str, Any]], Dict[str, Any]]


@dataclass
class BrokeredHttpClient:
    engine_id: str
    rpc_invoke: RpcInvoker

    def fetch(
        self,
        *,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body_b64: str = "",
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 1024 * 1024,
    ) -> Dict[str, Any]:
        return self.rpc_invoke(
            "sandbox-http-fetch",
            {
                "engine_id": self.engine_id,
                "url": str(url or ""),
                "method": str(method or "GET"),
                "headers": dict(headers or {}),
                "body_b64": str(body_b64 or ""),
                "timeout_seconds": float(timeout_seconds or 30.0),
                "max_response_bytes": int(max_response_bytes or 1024 * 1024),
            },
        )
