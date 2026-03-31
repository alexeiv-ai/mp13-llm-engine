from __future__ import annotations

import base64
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .policy import WorkerSandboxPolicy


class BrokeredHttpError(RuntimeError):
    pass


@dataclass
class BrokeredHttpClient:
    policy: WorkerSandboxPolicy

    def _require_enabled(self) -> None:
        if not self.policy.enabled:
            raise PermissionError("sandbox_disabled")
        if not self.policy.brokered_io.http:
            raise PermissionError("brokered_http_disabled")
        if str(self.policy.network.mode or "").strip().lower() != "brokered_only":
            raise PermissionError("brokered_http_requires_network_mode_brokered_only")

    def _validate_url(self, url: str) -> urllib.parse.SplitResult:
        parsed = urllib.parse.urlsplit(str(url or "").strip())
        if parsed.scheme not in {"http", "https"}:
            raise PermissionError("brokered_http_scheme_not_allowed")
        host = str(parsed.hostname or "").strip().lower()
        if not host:
            raise PermissionError("brokered_http_host_required")
        allowed_hosts = {str(x or "").strip().lower() for x in list(self.policy.network.allow_hosts or []) if str(x or "").strip()}
        if allowed_hosts and host not in allowed_hosts:
            raise PermissionError(f"brokered_http_host_not_allowed:{host}")
        allow_prefixes = [str(x or "").strip() for x in list(self.policy.network.allow_url_prefixes or []) if str(x or "").strip()]
        if allow_prefixes and not any(str(url).startswith(prefix) for prefix in allow_prefixes):
            raise PermissionError(f"brokered_http_url_not_allowed:{url}")
        return parsed

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
        self._require_enabled()
        parsed = self._validate_url(url)
        body = b""
        if str(body_b64 or "").strip():
            try:
                body = base64.b64decode(str(body_b64), validate=True)
            except Exception as exc:
                raise ValueError(f"invalid body_b64: {exc}") from exc
        req_headers: Dict[str, str] = {}
        for key, value in dict(headers or {}).items():
            raw_key = str(key or "").strip()
            if not raw_key:
                continue
            low = raw_key.lower()
            if low in {"host", "content-length", "connection"}:
                continue
            req_headers[raw_key] = str(value)
        req = urllib.request.Request(
            urllib.parse.urlunsplit(parsed),
            data=body if body else None,
            headers=req_headers,
            method=str(method or "GET").strip().upper() or "GET",
        )
        with urllib.request.urlopen(req, timeout=max(0.1, float(timeout_seconds or 30.0))) as resp:  # noqa: S310
            raw = resp.read(max(1024, int(max_response_bytes or 1024 * 1024)) + 1)
            truncated = len(raw) > int(max_response_bytes or 1024 * 1024)
            if truncated:
                raw = raw[: int(max_response_bytes or 1024 * 1024)]
            return {
                "url": urllib.parse.urlunsplit(parsed),
                "status_code": int(getattr(resp, "status", 200) or 200),
                "headers": dict(getattr(resp, "headers", {}) or {}),
                "body_b64": base64.b64encode(raw).decode("ascii") if raw else "",
                "body_size": len(raw),
                "truncated": truncated,
            }
