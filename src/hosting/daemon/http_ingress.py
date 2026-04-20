"""HTTP ingress daemon for engine host worker APIs."""
from __future__ import annotations

import base64
import http.server
import json
import logging
import os
import secrets
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..service.host_service import EngineHostService
from .constants import DEFAULT_HTTP_INGRESS_PORT
from .paths import _default_http_pid_file
from .pidfile import DaemonPidFile

logger = logging.getLogger(__name__)


class EngineHostHttpIngressDaemon:
    """
    HTTP ingress daemon that proxies worker API calls via EngineHostService.

    Endpoints:
      GET  /health
      POST /__shutdown__               {"shutdown_token":"..."}
      *    /proxy/<engine_id>/<path>   (also supports /api/engine-host/proxy/<engine_id>/<path>)

    Auth/session model matches `proxy-request` command:
      - session token from Authorization: Bearer <token>, X-Session-Token, or query/session JSON field
      - traffic scope with engine allowlist enforced by EngineHostService.authorize_command
    """

    def __init__(
        self,
        *,
        port: int = DEFAULT_HTTP_INGRESS_PORT,
        pid_file: Optional[Path] = None,
        engines_state_file: Optional[Path] = None,
        control_state_file: Optional[Path] = None,
    ):
        self.port = int(port or DEFAULT_HTTP_INGRESS_PORT)
        self.pid_file = DaemonPidFile(pid_file or _default_http_pid_file())
        self.shutdown_token = secrets.token_urlsafe(24)
        self.svc = EngineHostService(
            engines_state_file=engines_state_file,
            control_state_file=control_state_file,
        )
        self.svc.assert_runtime_policy_safe()
        self._server: Optional[http.server.ThreadingHTTPServer] = None
        self._stop_event = threading.Event()

    @staticmethod
    def _status_from_auth_error(error_text: str) -> int:
        err = str(error_text or "").strip().lower()
        if err in {
            "session_token_required",
            "missing_or_invalid_session_token",
            "session_revoked",
        }:
            return 401
        return 403

    @staticmethod
    def _extract_token(headers: http.client.HTTPMessage, query: Dict[str, List[str]], payload: Dict[str, Any]) -> str:  # type: ignore[name-defined]
        authz = str(headers.get("Authorization") or "").strip()
        if authz.lower().startswith("bearer "):
            token = authz[7:].strip()
            if token:
                return token
        x_token = str(headers.get("X-Session-Token") or "").strip()
        if x_token:
            return x_token
        for key in ("session_token", "auth_token", "token"):
            val = str(payload.get(key) or "").strip()
            if val:
                return val
            qv = query.get(key)
            if qv:
                val = str(qv[0] or "").strip()
                if val:
                    return val
        return ""

    @staticmethod
    def _proxy_route(path: str) -> Optional[tuple[str, str]]:
        raw = str(path or "")
        for prefix in ("/proxy/", "/api/engine-host/proxy/"):
            if raw.startswith(prefix):
                tail = raw[len(prefix):]
                if not tail:
                    return None
                parts = tail.split("/", 1)
                engine_id = str(parts[0] or "").strip()
                proxied = "/" + str(parts[1] or "").lstrip("/") if len(parts) > 1 else "/"
                if engine_id:
                    return engine_id, proxied
        return None

    def run(self) -> None:
        daemon = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def _send_http_error(
                self,
                status: int,
                message: str,
                *,
                error_code: str = "http_error",
                error_details: Optional[Dict[str, Any]] = None,
            ) -> None:
                raw = json.dumps(
                    {
                        "ok": False,
                        "error": str(message or "error"),
                        "error_code": str(error_code or "http_error"),
                        "error_details": dict(error_details or {}),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _read_payload(self) -> tuple[Dict[str, Any], bytes]:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    return {}, b""
                try:
                    raw = self.rfile.read(length)
                except Exception:
                    return {}, b""
                if not raw:
                    return {}, b""
                try:
                    parsed = json.loads(raw.decode("utf-8", errors="replace"))
                    return (dict(parsed or {}) if isinstance(parsed, dict) else {}), raw
                except Exception:
                    return {}, raw

            def _handle_proxy(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                route = daemon._proxy_route(parsed.path)
                if not route:
                    self._send_json(
                        404,
                        {"ok": False, "error": "not_found", "error_code": "route_not_found", "error_details": {}},
                    )
                    return
                engine_id, proxied_path = route
                query_map = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                payload, raw_body = self._read_payload()
                token = daemon._extract_token(self.headers, query_map, payload)
                body_b64 = ""
                if self.command in {"POST", "PUT", "PATCH", "DELETE"}:
                    # If caller sent JSON payload containing body_b64 use it, otherwise pass raw request body.
                    explicit = str(payload.get("body_b64") or "").strip()
                    if explicit:
                        body_b64 = explicit
                    else:
                        body_b64 = base64.b64encode(raw_body).decode("ascii") if raw_body else ""

                req_payload: Dict[str, Any] = {
                    "engine_id": engine_id,
                    "method": str(self.command or "GET"),
                    "path": proxied_path,
                    "query": parsed.query,
                    "headers": {str(k): str(v) for k, v in self.headers.items()},
                    "body_b64": body_b64,
                    "timeout_seconds": float(payload.get("timeout_seconds") or 30.0),
                    "max_response_bytes": int(payload.get("max_response_bytes") or 1024 * 1024),
                }
                if token:
                    req_payload["session_token"] = token
                try:
                    daemon.svc.authorize_command("proxy-request", req_payload)
                    result = daemon.svc.proxy_request(
                        engine_id=str(req_payload["engine_id"]),
                        method=str(req_payload["method"]),
                        path=str(req_payload["path"]),
                        query=str(req_payload["query"]),
                        headers=dict(req_payload["headers"]),
                        body_b64=str(req_payload["body_b64"]),
                        timeout_seconds=float(req_payload["timeout_seconds"]),
                        max_response_bytes=int(req_payload["max_response_bytes"]),
                    )
                    self._send_json(200, {"ok": True, "result": result})
                except PermissionError as exc:
                    status = daemon._status_from_auth_error(str(exc))
                    self._send_json(
                        status,
                        {
                            "ok": False,
                            "error": "auth_failed",
                            "error_code": str(exc or "auth_failed"),
                            "error_details": {"reason": str(exc or "auth_failed")},
                        },
                    )
                except Exception as exc:
                    self._send_json(
                        500,
                        {
                            "ok": False,
                            "error": "internal_error",
                            "error_code": "internal_error",
                            "error_details": {"message": str(exc)},
                        },
                    )

            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path == "/health":
                    auth = daemon.svc.auth_status()
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "mode": "http-ingress",
                            "pid": os.getpid(),
                            "port": daemon.port,
                            "started_at": daemon.pid_file.read().get("started_at") if daemon.pid_file.read() else None,
                            "daemon_version": str(auth.get("daemon_version") or ""),
                            "capabilities": dict(auth.get("capabilities") or {}),
                        },
                    )
                    return
                self._handle_proxy()

            def do_POST(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path == "/__shutdown__":
                    payload, _ = self._read_payload()
                    token = str(payload.get("shutdown_token") or "").strip()
                    if token and token == daemon.shutdown_token:
                        self._send_json(200, {"ok": True, "result": "shutting_down"})
                        daemon._stop_event.set()
                        if daemon._server is not None:
                            threading.Thread(target=daemon._server.shutdown, daemon=True).start()
                        return
                    self._send_json(
                        403,
                        {
                            "ok": False,
                            "error": "invalid_shutdown_token",
                            "error_code": "invalid_shutdown_token",
                            "error_details": {},
                        },
                    )
                    return
                self._handle_proxy()

            def do_PUT(self) -> None:  # noqa: N802
                self._handle_proxy()

            def do_PATCH(self) -> None:  # noqa: N802
                self._handle_proxy()

            def do_DELETE(self) -> None:  # noqa: N802
                self._handle_proxy()

            def do_HEAD(self) -> None:  # noqa: N802
                self._handle_proxy()

            def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
                logger.debug("HTTP ingress: " + str(fmt), *args)

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), _Handler)
        # Capture actual port for port=0 flow.
        self.port = int(self._server.server_address[1])
        self.pid_file.write(pid=os.getpid(), port=self.port, shutdown_token=self.shutdown_token)
        logger.info("EngineHostHttpIngressDaemon starting on 127.0.0.1:%d", self.port)
        try:
            self._server.serve_forever(poll_interval=0.2)
        finally:
            try:
                self._server.server_close()
            except Exception:
                pass
            self.pid_file.remove()
            logger.info("EngineHostHttpIngressDaemon stopped")
