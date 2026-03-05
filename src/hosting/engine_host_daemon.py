"""
Long-lived daemon server for engine host control.

Start in foreground:
  python -m hosting.engine_host_cli --daemon

Start detached in background:
  python -m hosting.engine_host_cli --daemon --background

The daemon binds to 127.0.0.1:<port> (default 19876) and accepts persistent
client connections using line-delimited JSON:

  Request:  {"seq": N, "cmd": "discover-running", "payload": {}}\n
  Response: {"seq": N, "ok": true, "result": [...]}\n
  Error:    {"seq": N, "ok": false, "error": "message"}\n

Built-in commands:
  __ping__     -> {"seq": N, "ok": true, "result": "pong"}
  __shutdown__ -> requires {"shutdown_token": "..."} in payload; stops daemon
"""
from __future__ import annotations

import asyncio
import base64
import http.client
import http.server
import json
import logging
import os
import secrets
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DAEMON_PORT = 19876
DEFAULT_HTTP_INGRESS_PORT = 19877


def _default_state_dir() -> Path:
    try:
        from mp13_engine.mp13_config_paths import get_default_config_dir  # type: ignore
        return (Path(get_default_config_dir()) / "backend").expanduser().resolve()
    except Exception:
        return (Path.home() / ".mp13-llm" / "backend").expanduser().resolve()


def _default_pid_file() -> Path:
    return _default_state_dir() / "host_daemon.pid"


def _default_http_pid_file() -> Path:
    return _default_state_dir() / "host_daemon_http.pid"


class DaemonPidFile:
    """Read/write the daemon PID file used for discovery by CLI and channel."""

    def __init__(self, path: Optional[Path] = None):
        self.path = (Path(path) if path else _default_pid_file()).expanduser().resolve()

    def write(self, *, pid: int, port: int, shutdown_token: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "pid": int(pid),
            "port": int(port),
            "started_at": time.time(),
            "shutdown_token": str(shutdown_token),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def read(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return dict(raw) if isinstance(raw, dict) else None
        except Exception:
            return None

    def remove(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            pass

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            p = int(pid or 0)
            if p <= 0:
                return False
            os.kill(p, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False

    def is_alive(self) -> bool:
        info = self.read()
        if not info:
            return False
        return self._pid_alive(int(info.get("pid") or 0))

    def get_port(self) -> Optional[int]:
        info = self.read()
        if not info:
            return None
        port = int(info.get("port") or 0)
        return port if port > 0 else None

    def get_shutdown_token(self) -> Optional[str]:
        info = self.read()
        if not info:
            return None
        return str(info.get("shutdown_token") or "").strip() or None


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
        from .engine_host_service import EngineHostService

        self.port = int(port or DEFAULT_HTTP_INGRESS_PORT)
        self.pid_file = DaemonPidFile(pid_file or _default_http_pid_file())
        self.shutdown_token = secrets.token_urlsafe(24)
        self.svc = EngineHostService(
            engines_state_file=engines_state_file,
            control_state_file=control_state_file,
        )
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
            def _send_http_error(self, status: int, message: str) -> None:
                raw = json.dumps({"ok": False, "error": str(message or "error")}, ensure_ascii=False).encode("utf-8")
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

            def _is_websocket_upgrade(self) -> bool:
                upgrade = str(self.headers.get("Upgrade") or "").strip().lower()
                conn = str(self.headers.get("Connection") or "").strip().lower()
                key = str(self.headers.get("Sec-WebSocket-Key") or "").strip()
                return upgrade == "websocket" and "upgrade" in conn and bool(key)

            def _read_backend_http_response(self, backend: socket.socket, *, max_bytes: int = 65536) -> bytes:
                buf = bytearray()
                while len(buf) < max_bytes:
                    chunk = backend.recv(4096)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if b"\r\n\r\n" in buf:
                        break
                return bytes(buf)

            def _copy_stream(self, src: socket.socket, dst: socket.socket) -> None:
                try:
                    while True:
                        data = src.recv(32768)
                        if not data:
                            break
                        dst.sendall(data)
                except Exception:
                    pass
                finally:
                    try:
                        dst.shutdown(socket.SHUT_WR)
                    except Exception:
                        pass

            def _run_tunnel(self, client_sock: socket.socket, backend_sock: socket.socket) -> None:
                t1 = threading.Thread(target=self._copy_stream, args=(client_sock, backend_sock), daemon=True)
                t2 = threading.Thread(target=self._copy_stream, args=(backend_sock, client_sock), daemon=True)
                t1.start()
                t2.start()
                t1.join(timeout=3600)
                t2.join(timeout=3600)

            def _handle_websocket_proxy(self) -> bool:
                parsed = urllib.parse.urlsplit(self.path)
                route = daemon._proxy_route(parsed.path)
                if not route:
                    self._send_http_error(404, "not_found")
                    return True
                engine_id, proxied_path = route
                query_map = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                token = daemon._extract_token(self.headers, query_map, {})

                req_payload: Dict[str, Any] = {
                    "engine_id": engine_id,
                    "method": "GET",
                    "path": proxied_path,
                    "query": parsed.query,
                    "headers": {str(k): str(v) for k, v in self.headers.items()},
                }
                if token:
                    req_payload["session_token"] = token
                try:
                    daemon.svc.authorize_command("proxy-request", req_payload)
                except PermissionError as exc:
                    self._send_http_error(daemon._status_from_auth_error(str(exc)), f"auth_failed: {exc}")
                    return True

                # Reuse traffic policy path/method constraints for websocket upgrade path.
                policy = daemon.svc._traffic_policy_for_engine(engine_id)  # noqa: SLF001
                allowed_methods = set(str(x).upper() for x in list(policy.get("allowed_methods") or []))
                if allowed_methods and "GET" not in allowed_methods:
                    self._send_http_error(403, "auth_failed: proxy_method_not_allowed:GET")
                    return True
                prefixes = [str(x) for x in list(policy.get("allowed_path_prefixes") or ["/"])]
                if prefixes and not any(proxied_path.startswith(px if px else "/") for px in prefixes):
                    self._send_http_error(403, f"auth_failed: proxy_path_not_allowed:{proxied_path}")
                    return True

                reg = daemon.svc.get_registration(engine_id) or {}
                endpoint = str(reg.get("endpoint") or "").strip()
                if not endpoint:
                    self._send_http_error(400, "engine endpoint is not registered")
                    return True

                target_url = daemon.svc._join_endpoint_path(endpoint, proxied_path, query=parsed.query)  # noqa: SLF001
                target = urllib.parse.urlsplit(target_url)
                scheme = str(target.scheme or "http").lower()
                ws_scheme = "ws"
                if scheme in {"https", "wss"}:
                    ws_scheme = "wss"
                host = str(target.hostname or "").strip()
                if not host:
                    self._send_http_error(400, "invalid_target_endpoint")
                    return True
                port = int(target.port or (443 if ws_scheme == "wss" else 80))
                request_uri = urllib.parse.urlunsplit(("", "", target.path or "/", target.query or "", ""))

                # Build backend upgrade request.
                raw_headers: List[str] = [
                    f"GET {request_uri} HTTP/1.1",
                    f"Host: {host}:{port}" if target.port else f"Host: {host}",
                    "Upgrade: websocket",
                    "Connection: Upgrade",
                    f"Sec-WebSocket-Key: {str(self.headers.get('Sec-WebSocket-Key') or '').strip()}",
                    f"Sec-WebSocket-Version: {str(self.headers.get('Sec-WebSocket-Version') or '13').strip() or '13'}",
                ]
                for hk in ("Sec-WebSocket-Protocol", "Sec-WebSocket-Extensions", "Origin", "User-Agent", "Cookie"):
                    hv = str(self.headers.get(hk) or "").strip()
                    if hv:
                        raw_headers.append(f"{hk}: {hv}")
                if bool(policy.get("allow_authorization_header", False)):
                    authz = str(self.headers.get("Authorization") or "").strip()
                    if authz:
                        raw_headers.append(f"Authorization: {authz}")
                backend_req = ("\r\n".join(raw_headers) + "\r\n\r\n").encode("utf-8")

                backend_sock: Optional[socket.socket] = None
                try:
                    backend_sock = socket.create_connection((host, port), timeout=10.0)
                    if ws_scheme == "wss":
                        ctx = ssl.create_default_context()
                        backend_sock = ctx.wrap_socket(backend_sock, server_hostname=host)
                    backend_sock.sendall(backend_req)
                    backend_resp = self._read_backend_http_response(backend_sock)
                    if not backend_resp:
                        self._send_http_error(502, "upstream_ws_handshake_failed")
                        try:
                            backend_sock.close()
                        except Exception:
                            pass
                        return True
                    # Forward upstream handshake response verbatim.
                    self.connection.sendall(backend_resp)
                    status_line = backend_resp.split(b"\r\n", 1)[0].decode("latin-1", errors="replace")
                    if " 101 " not in f" {status_line} ":
                        try:
                            backend_sock.close()
                        except Exception:
                            pass
                        return True
                    self.close_connection = True
                    self._run_tunnel(self.connection, backend_sock)
                    try:
                        backend_sock.close()
                    except Exception:
                        pass
                    return True
                except Exception:
                    if backend_sock is not None:
                        try:
                            backend_sock.close()
                        except Exception:
                            pass
                    self._send_http_error(502, "upstream_ws_connect_failed")
                    return True

            def _handle_proxy(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                route = daemon._proxy_route(parsed.path)
                if not route:
                    self._send_json(404, {"ok": False, "error": "not_found"})
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
                    self._send_json(status, {"ok": False, "error": f"auth_failed: {exc}"})
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})

            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path == "/health":
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "mode": "http-ingress",
                            "pid": os.getpid(),
                            "port": daemon.port,
                            "started_at": daemon.pid_file.read().get("started_at") if daemon.pid_file.read() else None,
                        },
                    )
                    return
                if self._is_websocket_upgrade() and self._handle_websocket_proxy():
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
                    self._send_json(403, {"ok": False, "error": "invalid_shutdown_token"})
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


class EngineHostDaemon:
    """
    Asyncio TCP server that routes line-delimited JSON requests to EngineHostService.

    Usage::

        daemon = EngineHostDaemon(port=19876)
        asyncio.run(daemon.run())  # blocks until __shutdown__ or SIGINT
    """

    def __init__(
        self,
        *,
        port: int = DEFAULT_DAEMON_PORT,
        pid_file: Optional[Path] = None,
        engines_state_file: Optional[Path] = None,
        control_state_file: Optional[Path] = None,
    ):
        from .engine_host_service import EngineHostService
        self.port = int(port or DEFAULT_DAEMON_PORT)
        self.pid_file = DaemonPidFile(pid_file)
        self.shutdown_token = secrets.token_urlsafe(24)
        self.svc = EngineHostService(
            engines_state_file=engines_state_file,
            control_state_file=control_state_file,
        )
        self._server: Optional[asyncio.AbstractServer] = None
        self._stop_event: Optional[asyncio.Event] = None

    async def run(self) -> None:
        """Start server, write PID file, run until stop event, clean up."""
        self._stop_event = asyncio.Event()
        self.pid_file.write(pid=os.getpid(), port=self.port, shutdown_token=self.shutdown_token)
        logger.info("EngineHostDaemon starting on 127.0.0.1:%d", self.port)
        try:
            self._server = await asyncio.start_server(
                self._handle_client,
                "127.0.0.1",
                self.port,
                limit=2 ** 20,
            )
            async with self._server:
                await self._stop_event.wait()
        finally:
            self.pid_file.remove()
            logger.info("EngineHostDaemon stopped")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        logger.debug("Client connected: %s", peer)
        try:
            while True:
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=300.0)
                except asyncio.TimeoutError:
                    break
                if not line:
                    break
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                response = await self._dispatch(raw)
                writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
                await writer.drain()
                # Stop serving this client after __shutdown__ is accepted
                if response.get("result") == "shutting_down" and response.get("ok"):
                    break
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:
            logger.warning("Client error %s: %s", peer, exc)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug("Client disconnected: %s", peer)

    async def _dispatch(self, raw_line: str) -> Dict[str, Any]:
        try:
            req = json.loads(raw_line)
        except Exception:
            return {"seq": -1, "ok": False, "error": "parse_error"}
        seq = int(req.get("seq") or 0)
        cmd = str(req.get("cmd") or "").strip()
        payload = dict(req.get("payload") or {})

        if cmd == "__ping__":
            return {"seq": seq, "ok": True, "result": "pong"}

        if cmd == "__shutdown__":
            token = str(payload.get("shutdown_token") or "")
            if token and token == self.shutdown_token:
                assert self._stop_event is not None
                self._stop_event.set()
                return {"seq": seq, "ok": True, "result": "shutting_down"}
            return {"seq": seq, "ok": False, "error": "invalid_shutdown_token"}

        try:
            self.svc.authorize_command(cmd, payload)
            result = await asyncio.to_thread(self._call_service, cmd, payload)
            return {"seq": seq, "ok": True, "result": result}
        except PermissionError as exc:
            return {"seq": seq, "ok": False, "error": f"auth_failed: {exc}"}
        except Exception as exc:
            return {"seq": seq, "ok": False, "error": str(exc)}

    def _call_service(self, cmd: str, payload: Dict[str, Any]) -> Any:
        """Synchronous dispatch to EngineHostService (runs in thread pool)."""
        svc = self.svc
        if cmd == "discover-running":
            return svc.discover_running()
        if cmd == "spawn":
            return svc.spawn(
                engine_id=str(payload.get("engine_id") or ""),
                command=list(payload.get("command") or []),
                cwd=payload.get("cwd"),
                env=dict(payload.get("env") or {}),
                endpoint=payload.get("endpoint"),
            )
        if cmd == "get-registration":
            return svc.get_registration(str(payload.get("engine_id") or ""))
        if cmd == "shutdown":
            return svc.shutdown(
                str(payload.get("engine_id") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 8.0),
            )
        if cmd == "ensure-running":
            return svc.ensure_running(str(payload.get("engine_id") or ""))
        if cmd == "remove-registration":
            return svc.remove_registration(str(payload.get("engine_id") or ""))
        if cmd == "claim-engine":
            return svc.claim_engine(
                str(payload.get("engine_id") or ""),
                backend_id=payload.get("backend_id"),
                exclusive=bool(payload.get("exclusive", False)),
            )
        if cmd == "claim-endpoint":
            return svc.claim_endpoint(
                backend_id=payload.get("backend_id"),
                exclusive=bool(payload.get("exclusive", False)),
            )
        if cmd == "claim-status":
            return svc.get_claim_status(str(payload.get("engine_id") or ""))
        if cmd == "issue-token":
            return svc.issue_token(
                str(payload.get("engine_id") or ""),
                backend_id=payload.get("backend_id"),
            )
        if cmd == "validate-token":
            return svc.validate_token(
                str(payload.get("engine_id") or ""),
                str(payload.get("token") or ""),
            )
        if cmd == "claim-resource":
            return svc.claim_resource(
                str(payload.get("resource_kind") or ""),
                str(payload.get("resource_id") or ""),
                backend_id=payload.get("backend_id"),
                exclusive=bool(payload.get("exclusive", False)),
            )
        if cmd == "resource-claim-status":
            return svc.get_resource_claim_status(
                str(payload.get("resource_kind") or ""),
                str(payload.get("resource_id") or ""),
            )
        if cmd == "issue-resource-token":
            return svc.issue_resource_token(
                str(payload.get("resource_kind") or ""),
                str(payload.get("resource_id") or ""),
                backend_id=payload.get("backend_id"),
            )
        if cmd == "validate-resource-token":
            return svc.validate_resource_token(
                str(payload.get("resource_kind") or ""),
                str(payload.get("resource_id") or ""),
                str(payload.get("token") or ""),
            )
        if cmd == "list-configs":
            return svc.list_engine_configs()
        if cmd == "create-config":
            return svc.create_engine_config(
                name=str(payload.get("name") or "engine_config"),
                config=dict(payload.get("config") or {}),
                overwrite=bool(payload.get("overwrite", False)),
            )
        if cmd == "models-from-config":
            return svc.models_from_config(str(payload.get("config_path") or "default"))
        if cmd == "connect-from-config":
            return svc.connect_from_config(
                config_path=str(payload.get("config_path") or "default"),
                engine_id=payload.get("engine_id"),
                model_path=payload.get("model_path"),
            )
        if cmd == "inspect-capabilities":
            return svc.inspect_engine_capabilities(
                str(payload.get("engine_id") or ""),
                str(payload.get("endpoint") or ""),
            )
        if cmd == "logs-tail":
            return svc.logs_tail(
                str(payload.get("engine_id") or ""),
                lines=int(payload.get("lines") or 200),
                max_bytes=int(payload.get("max_bytes") or 65536),
            )
        if cmd == "logs-follow":
            return svc.logs_follow(
                str(payload.get("engine_id") or ""),
                cursor=int(payload.get("cursor") or 0),
                max_bytes=int(payload.get("max_bytes") or 65536),
                max_lines=int(payload.get("max_lines") or 500),
            )
        if cmd == "proxy-request":
            return svc.proxy_request(
                engine_id=str(payload.get("engine_id") or ""),
                method=str(payload.get("method") or "GET"),
                path=str(payload.get("path") or "/"),
                query=str(payload.get("query") or ""),
                headers=dict(payload.get("headers") or {}),
                body_b64=str(payload.get("body_b64") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                max_response_bytes=int(payload.get("max_response_bytes") or 1024 * 1024),
            )
        if cmd == "proxy-ws-open":
            return svc.proxy_ws_open(
                engine_id=str(payload.get("engine_id") or ""),
                path=str(payload.get("path") or "/"),
                query=str(payload.get("query") or ""),
                headers=dict(payload.get("headers") or {}),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
            )
        if cmd == "proxy-ws-send":
            return svc.proxy_ws_send(
                ws_id=str(payload.get("ws_id") or ""),
                text=payload.get("text"),
                data_b64=str(payload.get("data_b64") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
            )
        if cmd == "proxy-ws-recv":
            return svc.proxy_ws_recv(
                ws_id=str(payload.get("ws_id") or ""),
                timeout_seconds=float(payload.get("timeout_seconds") or 30.0),
                max_bytes=int(payload.get("max_bytes") or (1024 * 1024)),
            )
        if cmd == "proxy-ws-close":
            return svc.proxy_ws_close(
                ws_id=str(payload.get("ws_id") or ""),
                code=int(payload.get("code") or 1000),
                reason=str(payload.get("reason") or ""),
            )
        if cmd == "get-control-config":
            return svc.get_control_config()
        if cmd == "set-control-config":
            return svc.set_control_config(
                ssh_key=payload.get("ssh_key"),
                require_auth=payload.get("require_auth"),
                traffic_policy=dict(payload.get("traffic_policy") or {}),
                engine_traffic_policies=dict(payload.get("engine_traffic_policies") or {}),
                websocket_session_policy=dict(payload.get("websocket_session_policy") or {}),
            )
        if cmd == "auth-status":
            return svc.auth_status()
        if cmd == "auth-list-keys":
            return svc.auth_list_keys()
        if cmd == "auth-list-sessions":
            return svc.auth_list_sessions(
                key_id=payload.get("key_id"),
                scope=payload.get("scope"),
                role=payload.get("role"),
                token_preview_contains=payload.get("token_preview_contains"),
                limit=int(payload.get("limit") or 100),
                offset=int(payload.get("offset") or 0),
            )
        if cmd == "auth-list-issued-tokens":
            return svc.auth_list_issued_tokens(
                engine_id=payload.get("engine_id"),
                resource_kind=payload.get("resource_kind"),
                resource_id=payload.get("resource_id"),
                backend_id=payload.get("backend_id"),
                token_preview_contains=payload.get("token_preview_contains"),
                limit=int(payload.get("limit") or 100),
                offset=int(payload.get("offset") or 0),
            )
        if cmd == "auth-upsert-key":
            return svc.auth_upsert_key(
                key_id=str(payload.get("key_id") or ""),
                key_secret=str(payload.get("key_secret") or ""),
                role=str(payload.get("role") or ""),
                auth_method=str(payload.get("auth_method") or "shared_secret"),
                public_key=str(payload.get("public_key") or ""),
                allowed_configs=list(payload.get("allowed_configs") or []),
                allowed_engines=list(payload.get("allowed_engines") or []),
                disabled=bool(payload.get("disabled", False)),
            )
        if cmd == "auth-revoke-key":
            return svc.auth_revoke_key(str(payload.get("key_id") or ""))
        if cmd == "auth-issue-session":
            return svc.auth_issue_session(
                key_id=str(payload.get("key_id") or ""),
                key_secret=str(payload.get("key_secret") or ""),
                scope=str(payload.get("scope") or "control"),
                ttl_seconds=int(payload.get("ttl_seconds") or 900),
                config_paths=list(payload.get("config_paths") or []),
                engine_ids=list(payload.get("engine_ids") or []),
                ssh_binding=dict(payload.get("ssh_binding") or {}),
            )
        if cmd == "auth-begin-challenge":
            return svc.auth_begin_challenge(
                key_id=str(payload.get("key_id") or ""),
                scope=str(payload.get("scope") or "control"),
                ttl_seconds=int(payload.get("ttl_seconds") or 120),
                config_paths=list(payload.get("config_paths") or []),
                engine_ids=list(payload.get("engine_ids") or []),
                ssh_binding=dict(payload.get("ssh_binding") or {}),
            )
        if cmd == "auth-complete-challenge":
            return svc.auth_complete_challenge(
                challenge_id=str(payload.get("challenge_id") or ""),
                signature_ssh=str(payload.get("signature_ssh") or ""),
                presented_ssh_binding=dict(payload.get("_ssh_session_binding") or {}),
            )
        if cmd == "auth-revoke-session":
            return svc.auth_revoke_session(str(payload.get("token") or ""))
        if cmd == "host-metrics":
            return svc.get_host_metrics()
        raise ValueError(f"Unknown command '{cmd}'")


def run_daemon_foreground(
    *,
    port: int = DEFAULT_DAEMON_PORT,
    pid_file: Optional[Path] = None,
    engines_state_file: Optional[Path] = None,
    control_state_file: Optional[Path] = None,
) -> None:
    """Start daemon in the foreground (blocks until stopped)."""
    daemon = EngineHostDaemon(
        port=port,
        pid_file=pid_file,
        engines_state_file=engines_state_file,
        control_state_file=control_state_file,
    )
    asyncio.run(daemon.run())


def run_http_ingress_foreground(
    *,
    port: int = DEFAULT_HTTP_INGRESS_PORT,
    pid_file: Optional[Path] = None,
    engines_state_file: Optional[Path] = None,
    control_state_file: Optional[Path] = None,
) -> None:
    """Start HTTP ingress daemon in the foreground (blocks until stopped)."""
    daemon = EngineHostHttpIngressDaemon(
        port=port,
        pid_file=pid_file,
        engines_state_file=engines_state_file,
        control_state_file=control_state_file,
    )
    daemon.run()


def start_daemon_background(
    *,
    port: int = DEFAULT_DAEMON_PORT,
    pid_file: Optional[Path] = None,
    engines_state_file: Optional[Path] = None,
    control_state_file: Optional[Path] = None,
    wait_ready_seconds: float = 8.0,
) -> Dict[str, Any]:
    """
    Spawn daemon as a detached background process and wait until it is connectable.

    Returns {"pid": N, "port": P} on success.
    Raises RuntimeError if daemon does not become reachable within wait_ready_seconds.
    """
    argv: List[str] = [sys.executable, "-m", "hosting.engine_host_cli", "--daemon", "--port", str(port)]
    if pid_file:
        argv += ["--pid-file", str(pid_file)]
    if engines_state_file:
        argv += ["--engines-state-file", str(engines_state_file)]
    if control_state_file:
        argv += ["--control-state-file", str(control_state_file)]

    # Build environment with src dir on PYTHONPATH so connectors package is found
    import os as _os
    env = dict(_os.environ)
    src_root = str(Path(__file__).resolve().parents[1])
    py_path = str(env.get("PYTHONPATH") or "")
    if src_root not in py_path.split(_os.pathsep):
        env["PYTHONPATH"] = src_root if not py_path else f"{src_root}{_os.pathsep}{py_path}"

    kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": env,
    }
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(argv, **kwargs)  # noqa: S603
    spawned_pid = int(proc.pid)

    # Poll until PID file appears and socket is connectable
    pid_info = DaemonPidFile(pid_file)
    deadline = time.time() + max(1.0, float(wait_ready_seconds))
    while time.time() < deadline:
        time.sleep(0.15)
        if not pid_info.is_alive():
            continue
        actual_port = pid_info.get_port()
        if not actual_port:
            continue
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect(("127.0.0.1", actual_port))
            s.close()
            info = pid_info.read() or {}
            return {"pid": int(info.get("pid") or spawned_pid), "port": actual_port}
        except OSError:
            continue

    raise RuntimeError(
        f"Engine host daemon did not become ready within {wait_ready_seconds}s "
        f"(spawned pid={spawned_pid}, port={port})"
    )


def start_http_ingress_background(
    *,
    port: int = DEFAULT_HTTP_INGRESS_PORT,
    pid_file: Optional[Path] = None,
    engines_state_file: Optional[Path] = None,
    control_state_file: Optional[Path] = None,
    wait_ready_seconds: float = 8.0,
) -> Dict[str, Any]:
    """
    Spawn HTTP ingress daemon as a detached background process and wait until healthy.

    Returns {"pid": N, "port": P} on success.
    """
    argv: List[str] = [sys.executable, "-m", "hosting.engine_host_cli", "--daemon-http", "--http-port", str(port)]
    if pid_file:
        argv += ["--pid-file", str(pid_file)]
    if engines_state_file:
        argv += ["--engines-state-file", str(engines_state_file)]
    if control_state_file:
        argv += ["--control-state-file", str(control_state_file)]

    import os as _os

    env = dict(_os.environ)
    src_root = str(Path(__file__).resolve().parents[1])
    py_path = str(env.get("PYTHONPATH") or "")
    if src_root not in py_path.split(_os.pathsep):
        env["PYTHONPATH"] = src_root if not py_path else f"{src_root}{_os.pathsep}{py_path}"

    kwargs: Dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": env,
    }
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        kwargs["close_fds"] = True
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(argv, **kwargs)  # noqa: S603
    spawned_pid = int(proc.pid)

    pid_info = DaemonPidFile(pid_file or _default_http_pid_file())
    deadline = time.time() + max(1.0, float(wait_ready_seconds))
    while time.time() < deadline:
        time.sleep(0.15)
        if not pid_info.is_alive():
            continue
        actual_port = pid_info.get_port()
        if not actual_port:
            continue
        try:
            conn = http.client.HTTPConnection("127.0.0.1", actual_port, timeout=1.0)  # type: ignore[name-defined]
            conn.request("GET", "/health")
            resp = conn.getresponse()
            _ = resp.read()
            conn.close()
            if int(resp.status) == 200:
                info = pid_info.read() or {}
                return {"pid": int(info.get("pid") or spawned_pid), "port": actual_port}
        except Exception:
            continue

    raise RuntimeError(
        f"Engine host HTTP ingress daemon did not become ready within {wait_ready_seconds}s "
        f"(spawned pid={spawned_pid}, port={port})"
    )
