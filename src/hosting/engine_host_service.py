"""
Standalone engine host service logic.

This module is intentionally backend-agnostic: it only manages engine-host
process lifecycle and generic control-plane state (claims/tokens/resources).
"""
from __future__ import annotations

import json
import os
import re
import secrets
import signal
import hashlib
import shlex
import subprocess
import sys
import time
import hmac
import base64
import urllib.error
import urllib.parse
import urllib.request
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _default_state_dir() -> Path:
    try:
        from mp13_engine.mp13_config_paths import get_default_config_dir  # type: ignore

        return (Path(get_default_config_dir()) / "backend").expanduser().resolve()
    except Exception:
        return (Path.home() / ".mp13-llm" / "backend").expanduser().resolve()


DEFAULT_STATE_DIR = _default_state_dir()
DEFAULT_ENGINES_STATE_FILE = DEFAULT_STATE_DIR / "managed_engines.json"
DEFAULT_CONTROL_STATE_FILE = DEFAULT_STATE_DIR / "engine_host_control.json"


class EngineHostService:
    """File-backed engine host service for terminal-command control."""
    _metrics_lock = threading.Lock()
    _runtime_metrics: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        *,
        engines_state_file: Optional[Path] = None,
        control_state_file: Optional[Path] = None,
    ):
        self.engines_state_file = (engines_state_file or DEFAULT_ENGINES_STATE_FILE).expanduser().resolve()
        self.control_state_file = (control_state_file or DEFAULT_CONTROL_STATE_FILE).expanduser().resolve()
        self._ensure_metrics_initialized()

    @classmethod
    def _ensure_metrics_initialized(cls) -> None:
        with cls._metrics_lock:
            if isinstance(cls._runtime_metrics, dict):
                return
            cls._runtime_metrics = {
                "started_at": time.time(),
                "proxy": {
                    "inflight_total": 0,
                    "inflight_by_engine": {},
                    "inflight_peak": 0,
                    "total": 0,
                    "ok": 0,
                    "http_error": 0,
                    "failed": 0,
                    "request_bytes": 0,
                    "response_bytes": 0,
                    "last_status_code": None,
                    "last_error": None,
                    "last_request_at": 0.0,
                    "last_response_at": 0.0,
                    "recent_limit": 100,
                    "recent_requests": [],
                },
                "auth": {
                    "denied": 0,
                    "last_denied_reason": None,
                    "last_denied_at": 0.0,
                },
            }

    @classmethod
    def _metrics_proxy_start(cls, engine_id: str, request_bytes: int) -> None:
        cls._ensure_metrics_initialized()
        with cls._metrics_lock:
            assert isinstance(cls._runtime_metrics, dict)
            proxy = dict(cls._runtime_metrics.get("proxy") or {})
            inflight_by_engine = dict(proxy.get("inflight_by_engine") or {})
            eid = str(engine_id or "").strip() or "unknown"
            inflight_by_engine[eid] = int(inflight_by_engine.get(eid) or 0) + 1
            proxy["inflight_by_engine"] = inflight_by_engine
            proxy["inflight_total"] = int(proxy.get("inflight_total") or 0) + 1
            proxy["inflight_peak"] = max(
                int(proxy.get("inflight_peak") or 0),
                int(proxy.get("inflight_total") or 0),
            )
            proxy["total"] = int(proxy.get("total") or 0) + 1
            proxy["request_bytes"] = int(proxy.get("request_bytes") or 0) + max(0, int(request_bytes or 0))
            proxy["last_request_at"] = time.time()
            cls._runtime_metrics["proxy"] = proxy

    @classmethod
    def _metrics_proxy_finish(
        cls,
        engine_id: str,
        *,
        status_code: Optional[int] = None,
        response_bytes: int = 0,
        http_error: bool = False,
        failed: bool = False,
        error_message: Optional[str] = None,
        method: Optional[str] = None,
        path: Optional[str] = None,
        started_at: Optional[float] = None,
        truncated: Optional[bool] = None,
        request_bytes: int = 0,
    ) -> None:
        cls._ensure_metrics_initialized()
        with cls._metrics_lock:
            assert isinstance(cls._runtime_metrics, dict)
            proxy = dict(cls._runtime_metrics.get("proxy") or {})
            inflight_by_engine = dict(proxy.get("inflight_by_engine") or {})
            eid = str(engine_id or "").strip() or "unknown"
            current = int(inflight_by_engine.get(eid) or 0)
            if current <= 1:
                inflight_by_engine.pop(eid, None)
            else:
                inflight_by_engine[eid] = current - 1
            proxy["inflight_by_engine"] = inflight_by_engine
            proxy["inflight_total"] = max(0, int(proxy.get("inflight_total") or 0) - 1)
            proxy["response_bytes"] = int(proxy.get("response_bytes") or 0) + max(0, int(response_bytes or 0))
            proxy["last_response_at"] = time.time()
            if status_code is not None:
                proxy["last_status_code"] = int(status_code)
            if http_error:
                proxy["http_error"] = int(proxy.get("http_error") or 0) + 1
                outcome = "http_error"
            elif failed:
                proxy["failed"] = int(proxy.get("failed") or 0) + 1
                if error_message:
                    proxy["last_error"] = str(error_message)
                outcome = "failed"
            else:
                proxy["ok"] = int(proxy.get("ok") or 0) + 1
                outcome = "ok"
            now = time.time()
            entry = {
                "timestamp": now,
                "engine_id": eid,
                "method": str(method or ""),
                "path": str(path or ""),
                "status_code": int(status_code) if status_code is not None else None,
                "outcome": outcome,
                "request_bytes": max(0, int(request_bytes or 0)),
                "response_bytes": max(0, int(response_bytes or 0)),
                "duration_ms": int(max(0.0, (now - float(started_at or now)) * 1000.0)),
                "truncated": bool(truncated) if truncated is not None else None,
                "error": str(error_message or "") or None,
            }
            recent = list(proxy.get("recent_requests") or [])
            recent.append(entry)
            limit = max(10, int(proxy.get("recent_limit") or 100))
            if len(recent) > limit:
                recent = recent[-limit:]
            proxy["recent_requests"] = recent
            cls._runtime_metrics["proxy"] = proxy

    @classmethod
    def _metrics_auth_denied(cls, reason: str) -> None:
        cls._ensure_metrics_initialized()
        with cls._metrics_lock:
            assert isinstance(cls._runtime_metrics, dict)
            auth = dict(cls._runtime_metrics.get("auth") or {})
            auth["denied"] = int(auth.get("denied") or 0) + 1
            auth["last_denied_reason"] = str(reason or "denied")
            auth["last_denied_at"] = time.time()
            cls._runtime_metrics["auth"] = auth

    def get_host_metrics(self) -> Dict[str, Any]:
        self._ensure_metrics_initialized()
        with self._metrics_lock:
            assert isinstance(self._runtime_metrics, dict)
            snapshot = json.loads(json.dumps(self._runtime_metrics))
        snapshot["pid"] = os.getpid()
        snapshot["runtime_scope"] = "process"
        snapshot["recommended_mode"] = "daemon"
        snapshot["engines_state_file"] = str(self.engines_state_file)
        snapshot["control_state_file"] = str(self.control_state_file)
        snapshot["timestamp"] = time.time()
        return snapshot

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if int(pid or 0) <= 0:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False

    @staticmethod
    def _normalize_backend_id(backend_id: Optional[str]) -> str:
        raw = str(backend_id or "").strip()
        return raw or "backend:unknown"

    @staticmethod
    def _resource_key(resource_kind: str, resource_id: str) -> str:
        return f"{str(resource_kind or '').strip().lower()}:{str(resource_id or '').strip()}"

    def _read_json(self, path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        if not path.exists():
            return dict(default)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            return dict(default)
        except Exception:
            return dict(default)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_engines(self) -> List[Dict[str, Any]]:
        data = self._read_json(self.engines_state_file, {"version": 1, "engines": []})
        rows = data.get("engines")
        return rows if isinstance(rows, list) else []

    def _write_engines(self, rows: List[Dict[str, Any]]) -> None:
        self._write_json(
            self.engines_state_file,
            {"version": 1, "updated_at": time.time(), "engines": list(rows or [])},
        )

    def _read_control(self) -> Dict[str, Any]:
        payload = self._read_json(
            self.control_state_file,
            {
                "version": 1,
                "control_config": {
                    "ssh_key": None,
                    "require_auth": False,
                    "auth": {"keys": {}, "sessions": {}},
                    "config_store_mode": "store_only",
                    "traffic_policy": {
                        "allowed_methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
                        "allowed_path_prefixes": ["/"],
                        "request_header_allowlist": [
                            "accept",
                            "content-type",
                            "authorization",
                            "x-request-id",
                            "x-trace-id",
                            "x-correlation-id",
                            "user-agent",
                        ],
                        "response_header_allowlist": [
                            "content-type",
                            "content-length",
                            "cache-control",
                            "etag",
                            "last-modified",
                            "x-request-id",
                            "x-trace-id",
                            "x-correlation-id",
                            "date",
                            "server",
                        ],
                        "allow_authorization_header": False,
                        "max_request_bytes": 1024 * 1024,
                        "max_response_bytes": 1024 * 1024,
                    },
                },
                "claims_by_engine": {},
                "endpoint_claim": {"owners": [], "exclusive_owner": None, "claimed_at": 0.0},
                "tokens": {},
                "resource_claims": {},
                "resource_tokens": {},
            },
        )
        payload.setdefault(
            "control_config",
            {
                "ssh_key": None,
                "require_auth": False,
                "auth": {"keys": {}, "sessions": {}},
                "config_store_mode": "store_only",
                "traffic_policy": {},
            },
        )
        payload.setdefault("claims_by_engine", {})
        payload.setdefault("endpoint_claim", {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        payload.setdefault("tokens", {})
        payload.setdefault("resource_claims", {})
        payload.setdefault("resource_tokens", {})
        cfg = dict(payload.get("control_config") or {})
        cfg.setdefault("ssh_key", None)
        cfg.setdefault("require_auth", False)
        cfg.setdefault("config_store_mode", "store_only")
        raw_policy = dict(cfg.get("traffic_policy") or {})
        raw_policy.setdefault("allowed_methods", ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
        raw_policy.setdefault("allowed_path_prefixes", ["/"])
        raw_policy.setdefault(
            "request_header_allowlist",
            ["accept", "content-type", "authorization", "x-request-id", "x-trace-id", "x-correlation-id", "user-agent"],
        )
        raw_policy.setdefault(
            "response_header_allowlist",
            ["content-type", "content-length", "cache-control", "etag", "last-modified", "x-request-id", "x-trace-id", "x-correlation-id", "date", "server"],
        )
        raw_policy.setdefault("allow_authorization_header", False)
        raw_policy.setdefault("max_request_bytes", 1024 * 1024)
        raw_policy.setdefault("max_response_bytes", 1024 * 1024)
        cfg["traffic_policy"] = raw_policy
        auth = dict(cfg.get("auth") or {})
        auth.setdefault("keys", {})
        auth.setdefault("sessions", {})
        cfg["auth"] = auth
        payload["control_config"] = cfg
        return payload

    def _write_control(self, payload: Dict[str, Any]) -> None:
        out = dict(payload or {})
        out["version"] = 1
        out["updated_at"] = time.time()
        self._write_json(self.control_state_file, out)

    @staticmethod
    def _safe_config_name(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "").strip()).strip("_")
        return cleaned or "engine_config"

    def _logs_dir(self) -> Path:
        return self.engines_state_file.parent / "logs"

    def _engine_log_path(self, engine_id: str) -> Path:
        stem = self._safe_config_name(str(engine_id or "engine"))
        return (self._logs_dir() / f"{stem}.log").expanduser().resolve()

    def _default_config_path(self) -> Path:
        try:
            from mp13_engine.mp13_config_paths import get_default_config_dir  # type: ignore

            p = Path(get_default_config_dir()) / "mp13_config.json"
            return p.expanduser().resolve()
        except Exception:
            return (Path.home() / ".mp13-llm" / "mp13_config.json").expanduser().resolve()

    def _config_store_dir(self) -> Path:
        base = self._default_config_path().parent
        return (base / "backend" / "configs").expanduser().resolve()

    def _config_store_mode(self) -> str:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        mode = str(cfg.get("config_store_mode") or "store_only").strip().lower()
        return mode if mode in {"store_only"} else "store_only"

    @staticmethod
    def _normalize_traffic_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        p = dict(policy or {})
        allowed_methods = [str(x or "").strip().upper() for x in list(p.get("allowed_methods") or []) if str(x or "").strip()]
        if not allowed_methods:
            allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
        allowed_path_prefixes = [str(x or "").strip() for x in list(p.get("allowed_path_prefixes") or []) if str(x or "").strip()]
        if not allowed_path_prefixes:
            allowed_path_prefixes = ["/"]
        req_headers = [str(x or "").strip().lower() for x in list(p.get("request_header_allowlist") or []) if str(x or "").strip()]
        resp_headers = [str(x or "").strip().lower() for x in list(p.get("response_header_allowlist") or []) if str(x or "").strip()]
        if not req_headers:
            req_headers = ["accept", "content-type", "authorization", "x-request-id", "x-trace-id", "x-correlation-id", "user-agent"]
        if not resp_headers:
            resp_headers = ["content-type", "content-length", "cache-control", "etag", "last-modified", "x-request-id", "x-trace-id", "x-correlation-id", "date", "server"]
        max_req = max(1024, int(p.get("max_request_bytes") or (1024 * 1024)))
        max_resp = max(1024, int(p.get("max_response_bytes") or (1024 * 1024)))
        return {
            "allowed_methods": sorted(list(set(allowed_methods))),
            "allowed_path_prefixes": sorted(list(set(allowed_path_prefixes))),
            "request_header_allowlist": sorted(list(set(req_headers))),
            "response_header_allowlist": sorted(list(set(resp_headers))),
            "allow_authorization_header": bool(p.get("allow_authorization_header", False)),
            "max_request_bytes": max_req,
            "max_response_bytes": max_resp,
        }

    def _traffic_policy(self) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        return self._normalize_traffic_policy(dict(cfg.get("traffic_policy") or {}))

    def _normalize_config_selector(self, config_path: str) -> str:
        raw = str(config_path or "").strip()
        if not raw or raw.lower() == "default":
            return "default"
        if any(x in raw for x in ["/", "\\", ":"]) or raw.startswith("."):
            raise ValueError("config_path must be 'default' or a config name in hosted config store")
        stem = Path(raw if Path(raw).suffix else f"{raw}.json").stem
        safe = self._safe_config_name(stem)
        if safe != stem:
            raise ValueError("config_path contains unsupported characters")
        return safe

    def _resolve_json_config_path(self, config_path: str) -> Path:
        default_path = self._default_config_path()
        selector = self._normalize_config_selector(config_path)
        if selector == "default":
            return default_path
        if self._config_store_mode() != "store_only":
            raise ValueError("Unsupported config store mode")
        return (self._config_store_dir() / f"{selector}.json").expanduser().resolve()

    @staticmethod
    def _hash_secret(secret: str) -> str:
        raw = str(secret or "")
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _prune_expired_sessions(self, auth: Dict[str, Any]) -> int:
        sessions = dict(auth.get("sessions") or {})
        now = time.time()
        removed = 0
        for token, meta in list(sessions.items()):
            expires = float((meta or {}).get("expires_at") or 0.0)
            if expires > 0 and now >= expires:
                sessions.pop(token, None)
                removed += 1
        auth["sessions"] = sessions
        return removed

    def _extract_session_token(self, payload: Optional[Dict[str, Any]]) -> str:
        p = dict(payload or {})
        token = str(p.get("session_token") or p.get("auth_token") or "").strip()
        return token

    def _validate_session(
        self,
        control: Dict[str, Any],
        token: str,
        *,
        required_scope: str,
        requested_config: Optional[str] = None,
        requested_engine: Optional[str] = None,
    ) -> Dict[str, Any]:
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        self._prune_expired_sessions(auth)
        sessions = dict(auth.get("sessions") or {})
        session = dict(sessions.get(str(token or "").strip()) or {})
        if not session:
            raise PermissionError("missing_or_invalid_session_token")
        if bool(session.get("revoked", False)):
            raise PermissionError("session_revoked")
        key_role = str(session.get("role") or "").strip().lower()
        scope = str(session.get("scope") or "").strip().lower()
        if key_role == "management":
            return session
        if scope != required_scope:
            raise PermissionError("insufficient_scope")
        if required_scope == "config":
            allowed = set(str(x) for x in list(session.get("allowed_configs") or []))
            if requested_config and "*" not in allowed and str(requested_config) not in allowed:
                raise PermissionError("config_access_denied")
        if required_scope == "traffic":
            allowed_engines = set(str(x) for x in list(session.get("allowed_engines") or []))
            if requested_engine and "*" not in allowed_engines and str(requested_engine) not in allowed_engines:
                raise PermissionError("engine_access_denied")
        return session

    def auth_status(self) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        self._prune_expired_sessions(auth)
        keys = dict(auth.get("keys") or {})
        sessions = dict(auth.get("sessions") or {})
        return {
            "require_auth": bool(cfg.get("require_auth", False)),
            "config_store_mode": str(cfg.get("config_store_mode") or "store_only"),
            "keys_count": len(keys),
            "sessions_count": len(sessions),
            "roles": sorted(list({str((v or {}).get("role") or "") for v in keys.values() if isinstance(v, dict)})),
        }

    def auth_list_keys(self) -> List[Dict[str, Any]]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        out: List[Dict[str, Any]] = []
        for key_id, meta in dict(auth.get("keys") or {}).items():
            m = dict(meta or {})
            out.append(
                {
                    "key_id": str(key_id),
                    "role": str(m.get("role") or ""),
                    "disabled": bool(m.get("disabled", False)),
                    "created_at": float(m.get("created_at") or 0.0),
                    "updated_at": float(m.get("updated_at") or 0.0),
                    "allowed_configs": list(m.get("allowed_configs") or []),
                    "allowed_engines": list(m.get("allowed_engines") or []),
                }
            )
        out.sort(key=lambda x: str(x.get("key_id") or ""))
        return out

    def auth_upsert_key(
        self,
        *,
        key_id: str,
        key_secret: str,
        role: str,
        allowed_configs: Optional[List[str]] = None,
        allowed_engines: Optional[List[str]] = None,
        disabled: bool = False,
    ) -> Dict[str, Any]:
        kid = str(key_id or "").strip()
        secret = str(key_secret or "").strip()
        role_norm = str(role or "").strip().lower()
        if not kid:
            raise ValueError("key_id is required")
        if not secret:
            raise ValueError("key_secret is required")
        if role_norm not in {"management", "config", "traffic"}:
            raise ValueError("role must be 'management', 'config', or 'traffic'")
        normalized_allowed: List[str] = []
        normalized_engines: List[str] = []
        if role_norm == "config":
            raw_rows = list(allowed_configs or [])
            if not raw_rows:
                normalized_allowed = ["*"]
            else:
                for row in raw_rows:
                    rs = str(row or "").strip()
                    if not rs:
                        continue
                    if rs == "*":
                        normalized_allowed.append("*")
                        continue
                    normalized_allowed.append(self._normalize_config_selector(rs))
                if not normalized_allowed:
                    normalized_allowed = ["*"]
        if role_norm == "traffic":
            raw_engines = list(allowed_engines or [])
            if not raw_engines:
                normalized_engines = ["*"]
            else:
                for row in raw_engines:
                    rs = str(row or "").strip()
                    if not rs:
                        continue
                    if rs == "*":
                        normalized_engines.append("*")
                        continue
                    normalized_engines.append(self._safe_config_name(rs))
                if not normalized_engines:
                    normalized_engines = ["*"]
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        keys = dict(auth.get("keys") or {})
        now = time.time()
        existing = dict(keys.get(kid) or {})
        keys[kid] = {
            "role": role_norm,
            "secret_hash": self._hash_secret(secret),
            "created_at": float(existing.get("created_at") or now),
            "updated_at": now,
            "disabled": bool(disabled),
            "allowed_configs": normalized_allowed,
            "allowed_engines": normalized_engines,
        }
        auth["keys"] = keys
        cfg["auth"] = auth
        control["control_config"] = cfg
        self._write_control(control)
        return {
            "key_id": kid,
            "role": role_norm,
            "disabled": bool(disabled),
            "allowed_configs": normalized_allowed,
            "allowed_engines": normalized_engines,
        }

    def auth_revoke_key(self, key_id: str) -> Dict[str, Any]:
        kid = str(key_id or "").strip()
        if not kid:
            raise ValueError("key_id is required")
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        keys = dict(auth.get("keys") or {})
        existed = kid in keys
        if existed:
            keys.pop(kid, None)
        sessions = dict(auth.get("sessions") or {})
        revoked_sessions = 0
        for tok, meta in list(sessions.items()):
            if str((meta or {}).get("key_id") or "") == kid:
                sessions.pop(tok, None)
                revoked_sessions += 1
        auth["keys"] = keys
        auth["sessions"] = sessions
        cfg["auth"] = auth
        control["control_config"] = cfg
        self._write_control(control)
        return {"key_id": kid, "revoked": bool(existed), "revoked_sessions": revoked_sessions}

    def auth_issue_session(
        self,
        *,
        key_id: str,
        key_secret: str,
        scope: str = "control",
        ttl_seconds: int = 900,
        config_paths: Optional[List[str]] = None,
        engine_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        kid = str(key_id or "").strip()
        secret = str(key_secret or "").strip()
        if not kid:
            raise ValueError("key_id is required")
        if not secret:
            raise ValueError("key_secret is required")
        scope_norm = str(scope or "control").strip().lower()
        if scope_norm not in {"control", "config", "traffic"}:
            raise ValueError("scope must be 'control', 'config', or 'traffic'")
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        self._prune_expired_sessions(auth)
        keys = dict(auth.get("keys") or {})
        key_meta = dict(keys.get(kid) or {})
        if not key_meta:
            raise PermissionError("unknown_key_id")
        if bool(key_meta.get("disabled", False)):
            raise PermissionError("key_disabled")
        expected_hash = str(key_meta.get("secret_hash") or "")
        provided_hash = self._hash_secret(secret)
        if not expected_hash or not hmac.compare_digest(expected_hash, provided_hash):
            raise PermissionError("invalid_key_secret")
        role = str(key_meta.get("role") or "").strip().lower()
        if role == "config" and scope_norm != "config":
            raise PermissionError("config_role_cannot_issue_non_config_scope")
        if role == "traffic" and scope_norm != "traffic":
            raise PermissionError("traffic_role_cannot_issue_non_traffic_scope")
        allowed_configs: List[str] = []
        allowed_engines: List[str] = []
        key_allowed = set(str(x) for x in list(key_meta.get("allowed_configs") or []))
        key_allowed_engines = set(str(x) for x in list(key_meta.get("allowed_engines") or []))
        if scope_norm == "config":
            requested = list(config_paths or [])
            if not requested:
                if key_allowed:
                    allowed_configs = sorted(list(key_allowed))
                else:
                    allowed_configs = ["*"]
            else:
                rows: List[str] = []
                for r in requested:
                    rs = str(r or "").strip()
                    if not rs:
                        continue
                    if rs == "*":
                        rows.append("*")
                    else:
                        rows.append(self._normalize_config_selector(rs))
                if "*" in key_allowed:
                    allowed_configs = sorted(list(set(rows or ["*"])))
                else:
                    clipped = [x for x in rows if x in key_allowed]
                    if not clipped:
                        raise PermissionError("no_allowed_config_overlap")
                    allowed_configs = sorted(list(set(clipped)))
        if scope_norm == "traffic":
            requested_engines = list(engine_ids or [])
            if not requested_engines:
                if key_allowed_engines:
                    allowed_engines = sorted(list(key_allowed_engines))
                else:
                    allowed_engines = ["*"]
            else:
                rows: List[str] = []
                for row in requested_engines:
                    rs = str(row or "").strip()
                    if not rs:
                        continue
                    if rs == "*":
                        rows.append("*")
                    else:
                        rows.append(self._safe_config_name(rs))
                if "*" in key_allowed_engines:
                    allowed_engines = sorted(list(set(rows or ["*"])))
                else:
                    clipped = [x for x in rows if x in key_allowed_engines]
                    if not clipped:
                        raise PermissionError("no_allowed_engine_overlap")
                    allowed_engines = sorted(list(set(clipped)))
        token = secrets.token_urlsafe(36)
        ttl = max(60, min(int(ttl_seconds or 900), 24 * 3600))
        now = time.time()
        sessions = dict(auth.get("sessions") or {})
        sessions[token] = {
            "key_id": kid,
            "role": role,
            "scope": scope_norm,
            "issued_at": now,
            "expires_at": now + ttl,
            "allowed_configs": allowed_configs,
            "allowed_engines": allowed_engines,
        }
        auth["sessions"] = sessions
        cfg["auth"] = auth
        control["control_config"] = cfg
        self._write_control(control)
        return {
            "status": "ok",
            "token": token,
            "scope": scope_norm,
            "role": role,
            "expires_at": now + ttl,
            "ttl_seconds": ttl,
            "allowed_configs": allowed_configs,
            "allowed_engines": allowed_engines,
        }

    def auth_revoke_session(self, token: str) -> Dict[str, Any]:
        tok = str(token or "").strip()
        if not tok:
            raise ValueError("token is required")
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        sessions = dict(auth.get("sessions") or {})
        existed = tok in sessions
        sessions.pop(tok, None)
        auth["sessions"] = sessions
        cfg["auth"] = auth
        control["control_config"] = cfg
        self._write_control(control)
        return {"token": tok, "revoked": bool(existed)}

    def authorize_command(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> None:
        c = str(cmd or "").strip()
        if not c:
            self._metrics_auth_denied("empty_command")
            raise PermissionError("empty_command")
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        require_auth = bool(cfg.get("require_auth", False))
        auth = dict(cfg.get("auth") or {})
        keys_count = len(dict(auth.get("keys") or {}))
        if not require_auth:
            return
        # Bootstrap: allow key provisioning if no keys are present.
        if c in {"auth-upsert-key", "auth-status"} and keys_count == 0:
            return
        if c in {"auth-issue-session"}:
            # Session issuance authenticates with key_id/key_secret in payload.
            return
        token = self._extract_session_token(payload)
        if not token:
            self._metrics_auth_denied("session_token_required")
            raise PermissionError("session_token_required")
        if c in {
            "discover-running",
            "spawn",
            "get-registration",
            "shutdown",
            "ensure-running",
            "remove-registration",
            "claim-engine",
            "claim-endpoint",
            "claim-status",
            "issue-token",
            "validate-token",
            "claim-resource",
            "resource-claim-status",
            "issue-resource-token",
            "validate-resource-token",
            "inspect-capabilities",
            "logs-tail",
            "logs-follow",
            "get-control-config",
            "set-control-config",
            "auth-upsert-key",
            "auth-revoke-key",
            "auth-list-keys",
            "auth-revoke-session",
            "host-metrics",
        }:
            try:
                _ = self._validate_session(control, token, required_scope="control")
            except PermissionError as exc:
                self._metrics_auth_denied(str(exc))
                raise
            return
        if c in {"proxy-request"}:
            p = dict(payload or {})
            requested_engine = str(p.get("engine_id") or "").strip()
            try:
                _ = self._validate_session(control, token, required_scope="traffic", requested_engine=requested_engine)
            except PermissionError as exc:
                self._metrics_auth_denied(str(exc))
                raise
            return
        if c in {"list-configs", "create-config", "models-from-config", "connect-from-config"}:
            requested_config = None
            p = dict(payload or {})
            if c in {"models-from-config", "connect-from-config"}:
                requested_config = self._normalize_config_selector(str(p.get("config_path") or "default"))
            try:
                _ = self._validate_session(control, token, required_scope="config", requested_config=requested_config)
            except PermissionError as exc:
                self._metrics_auth_denied(str(exc))
                raise
            return
        raise PermissionError(f"auth_policy_missing_for_command:{c}")

    def _merge_default_and_selected_config(self, config_path: str) -> Dict[str, Any]:
        default_path = self._default_config_path()
        selected_path = self._resolve_json_config_path(config_path)
        default_data: Dict[str, Any] = {}
        selected_data: Dict[str, Any] = {}
        if default_path.exists():
            try:
                default_data = json.loads(default_path.read_text(encoding="utf-8")) or {}
            except Exception:
                default_data = {}
        if selected_path.exists():
            selected_data = json.loads(selected_path.read_text(encoding="utf-8")) or {}
        if selected_path.resolve() == default_path.resolve():
            return selected_data if isinstance(selected_data, dict) else {}
        merged = dict(default_data) if isinstance(default_data, dict) else {}
        if isinstance(selected_data, dict):
            for k, v in selected_data.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    nested = dict(merged[k])
                    nested.update(v)
                    merged[k] = nested
                else:
                    merged[k] = v
        return merged

    @staticmethod
    def _resolve_path_token(value: str, *, config_dir: Path) -> Path:
        raw = (value or "").strip()
        if not raw:
            return config_dir
        if raw.startswith("@home"):
            rest = raw[5:].lstrip("/\\")
            return (Path.home() / rest).resolve()
        if raw.startswith("@project"):
            rest = raw[8:].lstrip("/\\")
            return (Path.cwd() / rest).resolve()
        if raw.startswith("@config"):
            rest = raw[7:].lstrip("/\\")
            return (config_dir / rest).resolve()
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (config_dir / p).resolve()

    def list_engine_configs(self) -> List[Dict[str, Any]]:
        default_path = self._default_config_path()
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        def _config_meta(path_str: str) -> Dict[str, Any]:
            try:
                merged = self._merge_default_and_selected_config(path_str)
            except Exception as e:
                return {"has_spawn_command": False, "connect_reason": f"invalid_config: {e}"}
            host_cfg = merged.get("engine_host") if isinstance(merged.get("engine_host"), dict) else {}
            command_spec = host_cfg.get("spawn_command") if isinstance(host_cfg, dict) else None
            if not command_spec:
                command_spec = merged.get("spawn_command")
            has_spawn = bool(command_spec)
            if not has_spawn:
                worker_profile = host_cfg.get("worker_profile") if isinstance(host_cfg, dict) and isinstance(host_cfg.get("worker_profile"), dict) else None
                if not worker_profile and isinstance(merged.get("worker_profile"), dict):
                    worker_profile = dict(merged.get("worker_profile") or {})
                if worker_profile:
                    has_spawn = bool((self._worker_profile_to_command(worker_profile) or {}).get("command"))
            return {"has_spawn_command": has_spawn, "connect_reason": None if has_spawn else "missing_spawn_command"}
        if default_path.exists():
            row = {"name": "default", "path": str(default_path), "is_default": True}
            row.update(_config_meta(str(default_path)))
            out.append(row)
            seen.add(str(default_path.resolve()))
        cfg_dir = self._config_store_dir()
        if cfg_dir.exists():
            for fp in sorted(cfg_dir.glob("*.json"), key=lambda p: p.name.lower()):
                try:
                    rp = str(fp.resolve())
                except Exception:
                    rp = str(fp)
                if rp in seen:
                    continue
                row = {"name": fp.stem, "path": str(fp), "is_default": False}
                row.update(_config_meta(str(fp)))
                out.append(row)
                seen.add(rp)
        return out

    def create_engine_config(self, *, name: str, config: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
        cfg_dir = self._config_store_dir()
        cfg_dir.mkdir(parents=True, exist_ok=True)
        stem = self._safe_config_name(name)
        path = (cfg_dir / f"{stem}.json").resolve()
        if path.exists() and not bool(overwrite):
            raise ValueError(f"Config '{stem}' already exists")
        existed = path.exists()
        payload = dict(config or {})
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"name": stem, "path": str(path), "created": True, "overwrote": bool(existed and overwrite)}

    def models_from_config(self, config_path: str) -> List[Dict[str, Any]]:
        cfg = self._merge_default_and_selected_config(config_path)
        selected_path = self._resolve_json_config_path(config_path)
        config_dir = selected_path.parent
        category_dirs = cfg.get("category_dirs") if isinstance(cfg.get("category_dirs"), dict) else {}
        models_root_raw = category_dirs.get("models_root_dir") or cfg.get("models_root_dir") or "@project/.."
        models_root = self._resolve_path_token(str(models_root_raw), config_dir=config_dir)
        results: List[Dict[str, Any]] = []
        if not models_root.exists():
            return results
        for child in models_root.iterdir():
            if not child.is_dir():
                continue
            safes = list(child.glob("*.safetensors"))
            if safes:
                results.append({"name": child.name, "path": str(child), "safetensors_count": len(safes)})
        results.sort(key=lambda x: str(x.get("name") or "").lower())
        return results

    @staticmethod
    def _replace_template(value: Optional[str], *, engine_id: str, config_path: str, model_path: Optional[str], endpoint: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        out = str(value)
        out = out.replace("{engine_id}", str(engine_id))
        out = out.replace("{config_path}", str(config_path))
        out = out.replace("{model_path}", str(model_path or ""))
        out = out.replace("{endpoint}", str(endpoint or ""))
        return out

    def _next_engine_id(self, base_name: str) -> str:
        existing = {str(x.get("engine_id") or "") for x in self._read_engines()}
        if base_name not in existing:
            return base_name
        idx = 2
        while f"{base_name}_{idx}" in existing:
            idx += 1
        return f"{base_name}_{idx}"

    @staticmethod
    def _check_module_available(python: str, module_name: str) -> Tuple[bool, str]:
        """
        Check whether *module_name* is importable by *python*.

        Runs a tiny subprocess so it works even when the calling process lives
        in a different venv (e.g. the docs venv checking the engine venv).
        Returns (True, "") on success, (False, reason) on failure.
        """
        try:
            result = subprocess.run(  # noqa: S603
                [python, "-c", f"import {module_name}"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                return True, ""
            stderr = (result.stderr or "").strip()
            last_line = stderr.splitlines()[-1] if stderr else "import failed"
            return False, last_line
        except FileNotFoundError:
            return False, f"Python executable not found: {python}"
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def _worker_profile_to_command(worker_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Translate simple worker profile knobs into concrete spawn settings.
        This keeps spawn command out of user-facing UX while preserving host-only execution details.

        Supported kinds:
          http_server / placeholder  — plain ``python -m http.server <port>`` (dev/test stand-in)
          mp13_engine                — real mp13_engine worker; verifies engine availability first
        """
        wp = dict(worker_profile or {})
        kind = str(wp.get("kind") or "http_server").strip().lower()
        try:
            port = int(wp.get("port") or 9001)
        except Exception:
            port = 9001
        if port <= 0:
            port = 9001
        endpoint = str(wp.get("endpoint") or "").strip() or f"http://127.0.0.1:{port}"
        cwd = str(wp.get("cwd") or "").strip() or None

        if kind in {"http_server", "http.server", "placeholder"}:
            # Placeholder launcher — useful for dev/test; replace with engine-native when ready.
            cmd = ["python", "-m", "http.server", str(port)]
            return {"command": cmd, "endpoint": endpoint, "cwd": cwd}

        if kind == "mp13_engine":
            # Resolve the Python executable that has mp13_engine installed.
            # Priority: worker_profile.engine_python → MP13_ENGINE_PYTHON env var → sys.executable
            python = str(wp.get("engine_python") or "").strip()
            if not python:
                python = os.environ.get("MP13_ENGINE_PYTHON", "").strip()
            if not python:
                python = sys.executable
            # Verify availability before committing to a spawn.
            ok, err_detail = EngineHostService._check_module_available(python, "mp13_engine")
            if not ok:
                return {
                    "command": [],
                    "endpoint": endpoint,
                    "cwd": cwd,
                    "error": (
                        f"mp13_engine is not available in Python '{python}': {err_detail}. "
                        f"Set engine_python in the worker_profile or MP13_ENGINE_PYTHON env var "
                        f"to point at a Python that has the full engine installed."
                    ),
                    "error_kind": "engine_not_available",
                }
            module = str(wp.get("module") or "mp13_engine").strip()
            cmd = [python, "-m", module, "--port", str(port)]
            extra_args = [str(a) for a in list(wp.get("args") or [])]
            if extra_args:
                cmd.extend(extra_args)
            return {"command": cmd, "endpoint": endpoint, "cwd": cwd}

        return {"command": [], "endpoint": endpoint, "cwd": cwd}

    def connect_from_config(self, *, config_path: str, engine_id: Optional[str] = None, model_path: Optional[str] = None) -> Dict[str, Any]:
        selected = self._resolve_json_config_path(config_path)
        cfg = self._merge_default_and_selected_config(config_path)
        if not isinstance(cfg, dict):
            cfg = {}
        host_cfg = cfg.get("engine_host") if isinstance(cfg.get("engine_host"), dict) else {}
        base_name = self._safe_config_name(Path(selected).stem or "engine")
        requested = self._safe_config_name(engine_id) if str(engine_id or "").strip() else ""
        eid = self._next_engine_id(requested or base_name)

        command_spec = host_cfg.get("spawn_command") if isinstance(host_cfg, dict) else None
        if not command_spec:
            command_spec = cfg.get("spawn_command")
        if isinstance(command_spec, str):
            command_list = shlex.split(command_spec)
        elif isinstance(command_spec, list):
            command_list = [str(x) for x in command_spec]
        else:
            command_list = []
        worker_profile = host_cfg.get("worker_profile") if isinstance(host_cfg, dict) and isinstance(host_cfg.get("worker_profile"), dict) else None
        if not worker_profile and isinstance(cfg.get("worker_profile"), dict):
            worker_profile = dict(cfg.get("worker_profile") or {})
        profile_spawn = self._worker_profile_to_command(worker_profile) if worker_profile else {"command": [], "endpoint": None, "cwd": None}
        if profile_spawn.get("error"):
            return {
                "status": "failed",
                "engine_id": eid,
                "config_path": str(selected),
                "reason": str(profile_spawn.get("error_kind") or "worker_profile_error"),
                "message": str(profile_spawn["error"]),
            }
        if not command_list and list(profile_spawn.get("command") or []):
            command_list = [str(x) for x in list(profile_spawn.get("command") or [])]
        if not command_list:
            return {
                "status": "failed",
                "engine_id": eid,
                "config_path": str(selected),
                "reason": "missing_spawn_command",
                "message": "Config is missing engine_host.spawn_command (or spawn_command).",
            }

        configured_model = (
            ((cfg.get("engine_params") or {}).get("base_model_path") if isinstance(cfg.get("engine_params"), dict) else None)
            or cfg.get("base_model_path")
            or cfg.get("model")
            or cfg.get("base_model_name_or_path")
        )
        effective_model_path = str(model_path or configured_model or "").strip() or None
        if not effective_model_path:
            return {
                "status": "needs_model",
                "engine_id": eid,
                "config_path": str(selected),
                "models": self.models_from_config(config_path),
                "message": "Config loaded but no model is configured. Select a model folder and connect again.",
            }

        endpoint_raw = host_cfg.get("endpoint") if isinstance(host_cfg, dict) else None
        if not endpoint_raw:
            endpoint_raw = cfg.get("endpoint")
        if not endpoint_raw:
            endpoint_raw = profile_spawn.get("endpoint")
        cwd_raw = host_cfg.get("cwd") if isinstance(host_cfg, dict) else None
        if not cwd_raw:
            cwd_raw = profile_spawn.get("cwd")
        env_raw = host_cfg.get("env") if isinstance(host_cfg, dict) and isinstance(host_cfg.get("env"), dict) else {}
        endpoint = self._replace_template(endpoint_raw, engine_id=eid, config_path=str(selected), model_path=effective_model_path, endpoint=endpoint_raw)
        cwd = self._replace_template(cwd_raw, engine_id=eid, config_path=str(selected), model_path=effective_model_path, endpoint=endpoint)
        env = {
            str(k): str(
                self._replace_template(str(v), engine_id=eid, config_path=str(selected), model_path=effective_model_path, endpoint=endpoint) or ""
            )
            for k, v in dict(env_raw or {}).items()
        }
        env.setdefault("MP13_ENGINE_CONFIG_PATH", str(selected))
        env.setdefault("MP13_ENGINE_ID", str(eid))
        env.setdefault("MP13_MODEL_PATH", str(effective_model_path))
        command = [
            self._replace_template(x, engine_id=eid, config_path=str(selected), model_path=effective_model_path, endpoint=endpoint) or ""
            for x in command_list
        ]
        command = [c for c in command if c]
        try:
            rec = self.spawn(engine_id=eid, command=command, cwd=cwd, endpoint=endpoint, env=env)
            return {
                "status": "ok",
                "engine_id": eid,
                "config_path": str(selected),
                "model_path": effective_model_path,
                "managed_engine": rec,
            }
        except Exception as e:
            return {
                "status": "failed",
                "engine_id": eid,
                "config_path": str(selected),
                "model_path": effective_model_path,
                "reason": "spawn_failed",
                "message": str(e),
            }

    def get_control_config(self) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        return {
            "ssh_key": cfg.get("ssh_key"),
            "require_auth": bool(cfg.get("require_auth", False)),
            "config_store_mode": str(cfg.get("config_store_mode") or "store_only"),
            "traffic_policy": self._normalize_traffic_policy(dict(cfg.get("traffic_policy") or {})),
            "keys_count": len(dict(auth.get("keys") or {})),
            "sessions_count": len(dict(auth.get("sessions") or {})),
        }

    def set_control_config(
        self,
        *,
        ssh_key: Optional[str] = None,
        require_auth: Optional[bool] = None,
        traffic_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        if ssh_key is not None:
            cfg["ssh_key"] = str(ssh_key).strip() if ssh_key else None
        if require_auth is not None:
            cfg["require_auth"] = bool(require_auth)
        cfg.setdefault("config_store_mode", "store_only")
        cfg.setdefault("auth", {"keys": {}, "sessions": {}})
        cfg["traffic_policy"] = self._normalize_traffic_policy(
            dict(cfg.get("traffic_policy") or {}) | dict(traffic_policy or {})
        )
        control["control_config"] = cfg
        self._write_control(control)
        auth = dict(cfg.get("auth") or {})
        return {
            "ssh_key": cfg.get("ssh_key"),
            "require_auth": bool(cfg.get("require_auth", False)),
            "config_store_mode": str(cfg.get("config_store_mode") or "store_only"),
            "traffic_policy": self._normalize_traffic_policy(dict(cfg.get("traffic_policy") or {})),
            "keys_count": len(dict(auth.get("keys") or {})),
            "sessions_count": len(dict(auth.get("sessions") or {})),
        }

    @staticmethod
    def _probe_url(endpoint: str, path: str) -> str:
        raw = str(endpoint or "").strip()
        if not raw:
            return ""
        parsed = urllib.parse.urlsplit(raw if "://" in raw else f"http://{raw}")
        p = str(parsed.path or "").strip()
        if not p or p == "/":
            p = path
        elif p.endswith("/"):
            p = f"{p}{path.lstrip('/')}"
        return urllib.parse.urlunsplit((parsed.scheme or "http", parsed.netloc, p, "", ""))

    def inspect_engine_capabilities(self, engine_id: str, endpoint: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        ep = str(endpoint or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        if not ep:
            raise ValueError("endpoint is required")
        checks = {
            "health": self._probe_url(ep, "/health"),
            "capabilities": self._probe_url(ep, "/capabilities"),
            "inference": self._probe_url(ep, "/inference"),
            "ws": self._probe_url(ep, "/ws"),
        }
        supported: Dict[str, bool] = {}
        details: Dict[str, Any] = {"engine_id": eid, "endpoint": ep, "checked_at": time.time(), "checks": {}}
        for key, url in checks.items():
            ok = False
            code = None
            err = None
            if key == "ws":
                # ws endpoint probe is heuristic: if URL exists in endpoint contract we expose it as available.
                ok = bool(url)
            else:
                req = urllib.request.Request(url, method="GET")
                try:
                    with urllib.request.urlopen(req, timeout=2.5) as resp:  # noqa: S310
                        code = int(getattr(resp, "status", 200) or 200)
                        ok = 200 <= code < 500
                except urllib.error.HTTPError as e:
                    code = int(e.code)
                    ok = code in {401, 403, 405}
                    err = f"http_{e.code}"
                except Exception as e:
                    err = str(e)
            supported[key] = bool(ok)
            details["checks"][key] = {"ok": bool(ok), "url": url, "status_code": code, "error": err}
        details["supported"] = supported
        details["engine_api"] = {
            "health": bool(supported.get("health")),
            "capabilities": bool(supported.get("capabilities")),
            "inference": bool(supported.get("inference")),
            "ws": bool(supported.get("ws")),
        }
        return details

    def _find_registration(self, engine_id: str) -> Optional[Dict[str, Any]]:
        eid = str(engine_id or "").strip()
        for row in self._read_engines():
            if str(row.get("engine_id") or "") == eid:
                return dict(row)
        return None

    def discover_running(self, *, prune_stale: bool = True) -> List[Dict[str, Any]]:
        rows = self._read_engines()
        out: List[Dict[str, Any]] = []
        stale_ids: List[str] = []
        now = time.time()
        for row in rows:
            item = dict(row)
            pid = int(item.get("pid") or 0)
            alive = self._pid_alive(pid)
            item["alive"] = alive
            item["uptime_seconds"] = max(0.0, now - float(item.get("spawned_at") or now))
            out.append(item)
            if not alive:
                stale_ids.append(str(item.get("engine_id") or ""))
        if prune_stale and stale_ids:
            keep = [r for r in rows if str(r.get("engine_id") or "") not in set(stale_ids)]
            self._write_engines(keep)
            out = [x for x in out if x.get("alive")]
        out.sort(key=lambda x: str(x.get("engine_id") or ""))
        return out

    def get_registration(self, engine_id: str) -> Optional[Dict[str, Any]]:
        return self._find_registration(engine_id)

    def register_spawned(
        self,
        *,
        engine_id: str,
        pid: int,
        command: List[str],
        cwd: Optional[str] = None,
        endpoint: Optional[str] = None,
        source: str = "engine_host_spawned",
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        record = {
            "engine_id": eid,
            "pid": int(pid or 0),
            "command": [str(x) for x in (command or [])],
            "cwd": str(cwd) if cwd else None,
            "spawned_at": time.time(),
            "owner_host_pid": os.getpid(),
            "source": str(source or "engine_host_spawned"),
            "endpoint": str(endpoint).strip() if endpoint else None,
            "log_path": str(self._engine_log_path(eid)),
        }
        rows = [r for r in self._read_engines() if str(r.get("engine_id") or "") != eid]
        rows.append(record)
        self._write_engines(rows)
        return record

    def spawn(
        self,
        *,
        engine_id: str,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not list(command or []):
            raise ValueError("command is required")
        log_path = self._engine_log_path(str(engine_id or ""))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(log_path, "ab")
        proc = subprocess.Popen(  # noqa: S603,S607
            [str(x) for x in command],
            cwd=str(cwd) if cwd else None,
            env=(dict(os.environ) | {str(k): str(v) for k, v in dict(env or {}).items()}),
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
        log_fp.close()
        return self.register_spawned(
            engine_id=engine_id,
            pid=int(proc.pid),
            command=[str(x) for x in command],
            cwd=cwd,
            endpoint=endpoint,
        )

    def remove_registration(self, engine_id: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        rows = self._read_engines()
        kept = [r for r in rows if str(r.get("engine_id") or "") != eid]
        changed = len(kept) != len(rows)
        if changed:
            self._write_engines(kept)
        return {"engine_id": eid, "removed": changed}

    def shutdown(self, engine_id: str, *, timeout_seconds: float = 8.0) -> Dict[str, Any]:
        entry = self._find_registration(engine_id)
        if not entry:
            return {"status": "not_found", "engine_id": str(engine_id), "alive": False}
        pid = int(entry.get("pid") or 0)
        eid = str(entry.get("engine_id") or engine_id)
        if pid <= 0:
            self.remove_registration(eid)
            return {"status": "invalid_pid", "engine_id": eid, "alive": False}
        if not self._pid_alive(pid):
            self.remove_registration(eid)
            return {"status": "already_stopped", "engine_id": eid, "pid": pid, "alive": False}
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.time() + max(0.1, float(timeout_seconds))
            while time.time() < deadline:
                if not self._pid_alive(pid):
                    break
                time.sleep(0.1)
        except Exception:
            pass
        if self._pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        alive = self._pid_alive(pid)
        if not alive:
            self.remove_registration(eid)
        return {"status": "stopped" if not alive else "stop_failed", "engine_id": eid, "pid": pid, "alive": alive}

    def ensure_running(self, engine_id: str) -> Dict[str, Any]:
        entry = self._find_registration(engine_id)
        if not entry:
            return {"status": "not_found", "engine_id": str(engine_id), "alive": False}
        eid = str(entry.get("engine_id") or engine_id)
        pid = int(entry.get("pid") or 0)
        command = [str(x) for x in list(entry.get("command") or []) if str(x).strip()]
        cwd = entry.get("cwd")
        endpoint = entry.get("endpoint")
        if pid > 0 and self._pid_alive(pid):
            return {"status": "running", "engine_id": eid, "pid": pid, "alive": True, "endpoint": endpoint}
        if not command:
            return {
                "status": "cannot_respawn",
                "engine_id": eid,
                "pid": pid,
                "alive": False,
                "reason": "missing_command_metadata",
                "endpoint": endpoint,
            }
        log_path = self._engine_log_path(eid)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(log_path, "ab")
        proc = subprocess.Popen(  # noqa: S603,S607
            command,
            cwd=str(cwd) if cwd else None,
            env=dict(os.environ),
            stdin=subprocess.DEVNULL,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
        log_fp.close()
        reg = self.register_spawned(engine_id=eid, pid=int(proc.pid), command=command, cwd=str(cwd) if cwd else None, endpoint=str(endpoint) if endpoint else None)
        return {
            "status": "respawned",
            "engine_id": eid,
            "previous_pid": pid,
            "pid": int(reg.get("pid") or 0),
            "alive": True,
            "endpoint": reg.get("endpoint"),
        }

    @staticmethod
    def _tail_lines_from_file(path: Path, *, lines: int, max_bytes: int) -> List[str]:
        if not path.exists():
            return []
        max_bytes = max(1024, int(max_bytes or 65536))
        lines = max(1, int(lines or 200))
        size = int(path.stat().st_size)
        start = max(0, size - max_bytes)
        with open(path, "rb") as f:
            f.seek(start)
            raw = f.read(max_bytes)
        text = raw.decode("utf-8", errors="replace")
        rows = text.splitlines()
        if len(rows) > lines:
            rows = rows[-lines:]
        return rows

    def logs_tail(self, engine_id: str, *, lines: int = 200, max_bytes: int = 65536) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        reg = self._find_registration(eid) or {}
        path = Path(str(reg.get("log_path") or self._engine_log_path(eid))).expanduser().resolve()
        out_lines = self._tail_lines_from_file(path, lines=lines, max_bytes=max_bytes)
        size = int(path.stat().st_size) if path.exists() else 0
        return {
            "engine_id": eid,
            "log_path": str(path),
            "exists": bool(path.exists()),
            "lines": out_lines,
            "cursor": size,
            "alive": bool(reg.get("pid")) and self._pid_alive(int(reg.get("pid") or 0)),
        }

    def logs_follow(self, engine_id: str, *, cursor: int = 0, max_bytes: int = 65536, max_lines: int = 500) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        reg = self._find_registration(eid) or {}
        path = Path(str(reg.get("log_path") or self._engine_log_path(eid))).expanduser().resolve()
        if not path.exists():
            return {"engine_id": eid, "log_path": str(path), "exists": False, "lines": [], "cursor": 0, "has_more": False}
        size = int(path.stat().st_size)
        pos = max(0, int(cursor or 0))
        if pos > size:
            pos = size
        read_limit = max(1024, int(max_bytes or 65536))
        with open(path, "rb") as f:
            f.seek(pos)
            raw = f.read(read_limit)
            new_cursor = int(f.tell())
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) > int(max_lines or 500):
            lines = lines[-int(max_lines or 500):]
        return {
            "engine_id": eid,
            "log_path": str(path),
            "exists": True,
            "lines": lines,
            "cursor": new_cursor,
            "has_more": bool(new_cursor < size),
            "alive": bool(reg.get("pid")) and self._pid_alive(int(reg.get("pid") or 0)),
        }

    @staticmethod
    def _join_endpoint_path(endpoint: str, req_path: str, query: str = "") -> str:
        raw_endpoint = str(endpoint or "").strip()
        if not raw_endpoint:
            return ""
        parsed = urllib.parse.urlsplit(raw_endpoint if "://" in raw_endpoint else f"http://{raw_endpoint}")
        incoming = str(req_path or "").strip() or "/"
        if not incoming.startswith("/"):
            incoming = f"/{incoming}"
        path = incoming
        if str(query or "").strip():
            q = str(query or "").lstrip("?")
            return urllib.parse.urlunsplit((parsed.scheme or "http", parsed.netloc, path, q, ""))
        return urllib.parse.urlunsplit((parsed.scheme or "http", parsed.netloc, path, "", ""))

    def proxy_request(
        self,
        *,
        engine_id: str,
        method: str = "GET",
        path: str = "/",
        query: str = "",
        headers: Optional[Dict[str, str]] = None,
        body_b64: str = "",
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 1024 * 1024,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        req_started_at = time.time()
        m = str(method or "GET").strip().upper()
        req_path = str(path or "/").strip() or "/"
        if not req_path.startswith("/"):
            req_path = f"/{req_path}"
        if not eid:
            raise ValueError("engine_id is required")
        reg = self._find_registration(eid) or {}
        endpoint = str(reg.get("endpoint") or "").strip()
        if not endpoint:
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message="engine endpoint is not registered",
                method=m,
                path=req_path,
                started_at=req_started_at,
            )
            raise ValueError("engine endpoint is not registered")
        url = self._join_endpoint_path(endpoint, path, query=query)
        if not url:
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message="failed to build proxy url",
                method=m,
                path=req_path,
                started_at=req_started_at,
            )
            raise ValueError("failed to build proxy url")
        traffic_policy = self._traffic_policy()
        if not re.fullmatch(r"[A-Z]+", m):
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message="invalid method",
                method=m,
                path=req_path,
                started_at=req_started_at,
            )
            raise ValueError("invalid method")
        allowed_methods = set(str(x).upper() for x in list(traffic_policy.get("allowed_methods") or []))
        if allowed_methods and m not in allowed_methods:
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message=f"proxy_method_not_allowed:{m}",
                method=m,
                path=req_path,
                started_at=req_started_at,
            )
            raise PermissionError(f"proxy_method_not_allowed:{m}")
        prefixes = [str(x) for x in list(traffic_policy.get("allowed_path_prefixes") or ["/"])]
        if prefixes and not any(req_path.startswith(px if px else "/") for px in prefixes):
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message=f"proxy_path_not_allowed:{req_path}",
                method=m,
                path=req_path,
                started_at=req_started_at,
            )
            raise PermissionError(f"proxy_path_not_allowed:{req_path}")
        body_raw = b""
        if str(body_b64 or "").strip():
            try:
                body_raw = base64.b64decode(str(body_b64), validate=True)
            except Exception as exc:
                self._metrics_proxy_finish(
                    eid,
                    failed=True,
                    error_message=f"invalid body_b64: {exc}",
                    method=m,
                    path=req_path,
                    started_at=req_started_at,
                )
                raise ValueError(f"invalid body_b64: {exc}") from exc
        max_req = int(traffic_policy.get("max_request_bytes") or (1024 * 1024))
        if len(body_raw) > max_req:
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message=f"request body too large ({len(body_raw)} > {max_req})",
                method=m,
                path=req_path,
                started_at=req_started_at,
                request_bytes=len(body_raw),
            )
            raise ValueError(f"request body too large ({len(body_raw)} > {max_req})")
        self._metrics_proxy_start(eid, request_bytes=len(body_raw))
        header_allow = set(str(x).lower() for x in list(traffic_policy.get("request_header_allowlist") or []))
        allow_authz = bool(traffic_policy.get("allow_authorization_header", False))
        req_headers: Dict[str, str] = {}
        for k, v in dict(headers or {}).items():
            key = str(k or "").strip()
            if not key:
                continue
            low = key.lower()
            if low == "authorization" and not allow_authz:
                continue
            if header_allow and low not in header_allow:
                continue
            req_headers[key] = str(v)
        req = urllib.request.Request(url, data=body_raw if body_raw else None, method=m, headers=req_headers)
        policy_lim = max(1024, int(traffic_policy.get("max_response_bytes") or (1024 * 1024)))
        lim = min(max(1024, int(max_response_bytes or policy_lim)), policy_lim)
        timeout = max(1.0, float(timeout_seconds or 30.0))
        resp_allow = set(str(x).lower() for x in list(traffic_policy.get("response_header_allowlist") or []))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                status = int(getattr(resp, "status", 200) or 200)
                response_headers: Dict[str, str] = {}
                for k, v in dict(getattr(resp, "headers", {}) or {}).items():
                    key = str(k or "").strip()
                    if not key:
                        continue
                    if resp_allow and key.lower() not in resp_allow:
                        continue
                    response_headers[key] = str(v)
                raw = resp.read(lim + 1)
                truncated = len(raw) > lim
                if truncated:
                    raw = raw[:lim]
                out_b64 = base64.b64encode(raw).decode("ascii")
                self._metrics_proxy_finish(
                    eid,
                    status_code=status,
                    response_bytes=len(raw),
                    http_error=False,
                    failed=False,
                    method=m,
                    path=req_path,
                    started_at=req_started_at,
                    truncated=bool(truncated),
                    request_bytes=len(body_raw),
                )
                return {
                    "engine_id": eid,
                    "endpoint": endpoint,
                    "url": url,
                    "status_code": status,
                    "headers": response_headers,
                    "body_b64": out_b64,
                    "body_size": len(raw),
                    "truncated": bool(truncated),
                }
        except urllib.error.HTTPError as exc:
            raw = exc.read(lim + 1)
            truncated = len(raw) > lim
            if truncated:
                raw = raw[:lim]
            out_b64 = base64.b64encode(raw).decode("ascii")
            response_headers: Dict[str, str] = {}
            for k, v in dict(exc.headers or {}).items():
                key = str(k or "").strip()
                if not key:
                    continue
                if resp_allow and key.lower() not in resp_allow:
                    continue
                response_headers[key] = str(v)
            self._metrics_proxy_finish(
                eid,
                status_code=int(exc.code),
                response_bytes=len(raw),
                http_error=True,
                method=m,
                path=req_path,
                started_at=req_started_at,
                truncated=bool(truncated),
                error_message=f"http_{int(exc.code)}",
                request_bytes=len(body_raw),
            )
            return {
                "engine_id": eid,
                "endpoint": endpoint,
                "url": url,
                "status_code": int(exc.code),
                "headers": response_headers,
                "body_b64": out_b64,
                "body_size": len(raw),
                "truncated": bool(truncated),
                "http_error": True,
            }
        except Exception as exc:
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message=str(exc),
                method=m,
                path=req_path,
                started_at=req_started_at,
                request_bytes=len(body_raw),
            )
            raise
        finally:
            # Ensure we decrement inflight in paths where finish wasn't called yet.
            with self._metrics_lock:
                assert isinstance(self._runtime_metrics, dict)
                proxy = dict(self._runtime_metrics.get("proxy") or {})
                inflight_by_engine = dict(proxy.get("inflight_by_engine") or {})
                current = int(inflight_by_engine.get(eid) or 0)
                if current > 0:
                    if current == 1:
                        inflight_by_engine.pop(eid, None)
                    else:
                        inflight_by_engine[eid] = current - 1
                    proxy["inflight_by_engine"] = inflight_by_engine
                    proxy["inflight_total"] = max(0, int(proxy.get("inflight_total") or 0) - 1)
                    self._runtime_metrics["proxy"] = proxy

    def _revoke_engine_tokens(self, control: Dict[str, Any], engine_id: str) -> int:
        tokens = dict(control.get("tokens") or {})
        revoked = 0
        for token, meta in list(tokens.items()):
            if str((meta or {}).get("engine_id") or "") == str(engine_id):
                tokens.pop(token, None)
                revoked += 1
        control["tokens"] = tokens
        return revoked

    def _revoke_all_tokens(self, control: Dict[str, Any]) -> int:
        t = dict(control.get("tokens") or {})
        r = dict(control.get("resource_tokens") or {})
        revoked = len(t) + len(r)
        control["tokens"] = {}
        control["resource_tokens"] = {}
        return revoked

    def claim_engine(self, engine_id: str, *, backend_id: Optional[str], exclusive: bool = False) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        bid = self._normalize_backend_id(backend_id)
        if not eid:
            raise ValueError("engine_id is required")
        control = self._read_control()
        claims = dict(control.get("claims_by_engine") or {})
        claim = dict(claims.get(eid) or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        owners = set(claim.get("owners") or [])
        displaced: List[str] = []
        revoked = 0
        if exclusive:
            displaced = sorted([o for o in owners if o != bid])
            claim["owners"] = [bid]
            claim["exclusive_owner"] = bid
            claim["claimed_at"] = time.time()
            revoked = self._revoke_engine_tokens(control, eid)
        else:
            previous_exclusive = str(claim.get("exclusive_owner") or "")
            if previous_exclusive and previous_exclusive != bid:
                displaced = [previous_exclusive]
                revoked = self._revoke_engine_tokens(control, eid)
            owners.add(bid)
            claim["owners"] = sorted(list(owners))
            claim["exclusive_owner"] = None
            claim["claimed_at"] = time.time()
        claims[eid] = claim
        control["claims_by_engine"] = claims
        self._write_control(control)
        return {
            "scope": "engine",
            "engine_id": eid,
            "backend_id": bid,
            "mode": "exclusive" if exclusive else "shared",
            "owners": list(claim.get("owners") or []),
            "exclusive_owner": claim.get("exclusive_owner"),
            "displaced_backends": displaced,
            "revoked_tokens": revoked,
        }

    def claim_endpoint(self, *, backend_id: Optional[str], exclusive: bool = False) -> Dict[str, Any]:
        bid = self._normalize_backend_id(backend_id)
        control = self._read_control()
        endpoint = dict(control.get("endpoint_claim") or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        owners = set(endpoint.get("owners") or [])
        displaced: List[str] = []
        revoked = 0
        if exclusive:
            displaced = sorted([o for o in owners if o != bid])
            endpoint = {"owners": [bid], "exclusive_owner": bid, "claimed_at": time.time()}
            control["claims_by_engine"] = {}
            control["resource_claims"] = {}
            revoked = self._revoke_all_tokens(control)
        else:
            previous_exclusive = str(endpoint.get("exclusive_owner") or "")
            if previous_exclusive and previous_exclusive != bid:
                displaced = [previous_exclusive]
                revoked = self._revoke_all_tokens(control)
            owners.add(bid)
            endpoint = {"owners": sorted(list(owners)), "exclusive_owner": None, "claimed_at": time.time()}
        control["endpoint_claim"] = endpoint
        self._write_control(control)
        return {
            "scope": "endpoint",
            "backend_id": bid,
            "mode": "exclusive" if exclusive else "shared",
            "owners": list(endpoint.get("owners") or []),
            "exclusive_owner": endpoint.get("exclusive_owner"),
            "displaced_backends": displaced,
            "revoked_tokens": revoked,
        }

    def get_claim_status(self, engine_id: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        control = self._read_control()
        claim = dict((control.get("claims_by_engine") or {}).get(eid) or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        endpoint = dict(control.get("endpoint_claim") or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        token_count = 0
        for meta in dict(control.get("tokens") or {}).values():
            if str((meta or {}).get("engine_id") or "") == eid:
                token_count += 1
        return {
            "engine_id": eid,
            "engine_claim": claim,
            "endpoint_claim": endpoint,
            "issued_tokens": token_count,
        }

    def issue_token(self, engine_id: str, *, backend_id: Optional[str]) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        bid = self._normalize_backend_id(backend_id)
        control = self._read_control()
        endpoint_exclusive = str((control.get("endpoint_claim") or {}).get("exclusive_owner") or "")
        if endpoint_exclusive and endpoint_exclusive != bid:
            return {"status": "denied", "engine_id": eid, "backend_id": bid, "token": None, "denied_reason": "endpoint_exclusive_conflict", "endpoint_exclusive_owner": endpoint_exclusive}
        claim = dict((control.get("claims_by_engine") or {}).get(eid) or {})
        exclusive_owner = str(claim.get("exclusive_owner") or "")
        if exclusive_owner and exclusive_owner != bid:
            return {"status": "denied", "engine_id": eid, "backend_id": bid, "token": None, "denied_reason": "engine_exclusive_conflict", "engine_exclusive_owner": exclusive_owner}
        owners = set(claim.get("owners") or [])
        if owners and bid not in owners:
            return {"status": "denied", "engine_id": eid, "backend_id": bid, "token": None, "denied_reason": "engine_shared_claim_not_member", "engine_owners": sorted(list(owners))}
        token = secrets.token_urlsafe(24)
        tokens = dict(control.get("tokens") or {})
        tokens[token] = {"engine_id": eid, "backend_id": bid, "issued_at": time.time()}
        control["tokens"] = tokens
        self._write_control(control)
        return {"status": "ok", "engine_id": eid, "backend_id": bid, "token": token, "issued_at": tokens[token]["issued_at"]}

    def validate_token(self, engine_id: str, token: str) -> bool:
        control = self._read_control()
        meta = dict(control.get("tokens") or {}).get(str(token or "").strip())
        return bool(meta and str(meta.get("engine_id") or "") == str(engine_id or ""))

    def claim_resource(self, resource_kind: str, resource_id: str, *, backend_id: Optional[str], exclusive: bool = False) -> Dict[str, Any]:
        rkind = str(resource_kind or "").strip().lower()
        rid = str(resource_id or "").strip()
        if rkind == "engine":
            return self.claim_engine(rid, backend_id=backend_id, exclusive=exclusive)
        bid = self._normalize_backend_id(backend_id)
        rkey = self._resource_key(rkind, rid)
        control = self._read_control()
        claims = dict(control.get("resource_claims") or {})
        claim = dict(claims.get(rkey) or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        owners = set(claim.get("owners") or [])
        displaced: List[str] = []
        revoked = 0
        if exclusive:
            displaced = sorted([o for o in owners if o != bid])
            claim["owners"] = [bid]
            claim["exclusive_owner"] = bid
            claim["claimed_at"] = time.time()
            res_tokens = dict(control.get("resource_tokens") or {})
            for t, meta in list(res_tokens.items()):
                if str((meta or {}).get("resource_key") or "") == rkey:
                    res_tokens.pop(t, None)
                    revoked += 1
            control["resource_tokens"] = res_tokens
        else:
            previous_exclusive = str(claim.get("exclusive_owner") or "")
            if previous_exclusive and previous_exclusive != bid:
                displaced = [previous_exclusive]
            owners.add(bid)
            claim["owners"] = sorted(list(owners))
            claim["exclusive_owner"] = None
            claim["claimed_at"] = time.time()
        claims[rkey] = claim
        control["resource_claims"] = claims
        self._write_control(control)
        return {
            "scope": "resource",
            "resource_kind": rkind,
            "resource_id": rid,
            "backend_id": bid,
            "mode": "exclusive" if exclusive else "shared",
            "owners": list(claim.get("owners") or []),
            "exclusive_owner": claim.get("exclusive_owner"),
            "displaced_backends": displaced,
            "revoked_tokens": revoked,
        }

    def get_resource_claim_status(self, resource_kind: str, resource_id: str) -> Dict[str, Any]:
        rkind = str(resource_kind or "").strip().lower()
        rid = str(resource_id or "").strip()
        if rkind == "engine":
            return self.get_claim_status(rid)
        rkey = self._resource_key(rkind, rid)
        control = self._read_control()
        claim = dict((control.get("resource_claims") or {}).get(rkey) or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        endpoint = dict(control.get("endpoint_claim") or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        issued_tokens = 0
        for meta in dict(control.get("resource_tokens") or {}).values():
            if str((meta or {}).get("resource_key") or "") == rkey:
                issued_tokens += 1
        return {
            "resource_kind": rkind,
            "resource_id": rid,
            "resource_claim": claim,
            "endpoint_claim": endpoint,
            "issued_tokens": issued_tokens,
        }

    def issue_resource_token(self, resource_kind: str, resource_id: str, *, backend_id: Optional[str]) -> Dict[str, Any]:
        rkind = str(resource_kind or "").strip().lower()
        rid = str(resource_id or "").strip()
        if rkind == "engine":
            out = self.issue_token(rid, backend_id=backend_id)
            out["resource_kind"] = "engine"
            out["resource_id"] = rid
            return out
        bid = self._normalize_backend_id(backend_id)
        rkey = self._resource_key(rkind, rid)
        control = self._read_control()
        endpoint_exclusive = str((control.get("endpoint_claim") or {}).get("exclusive_owner") or "")
        if endpoint_exclusive and endpoint_exclusive != bid:
            return {"status": "denied", "resource_kind": rkind, "resource_id": rid, "backend_id": bid, "token": None, "denied_reason": "endpoint_exclusive_conflict", "endpoint_exclusive_owner": endpoint_exclusive}
        claim = dict((control.get("resource_claims") or {}).get(rkey) or {})
        exclusive_owner = str(claim.get("exclusive_owner") or "")
        if exclusive_owner and exclusive_owner != bid:
            return {"status": "denied", "resource_kind": rkind, "resource_id": rid, "backend_id": bid, "token": None, "denied_reason": "resource_exclusive_conflict", "resource_exclusive_owner": exclusive_owner}
        owners = set(claim.get("owners") or [])
        if owners and bid not in owners:
            return {"status": "denied", "resource_kind": rkind, "resource_id": rid, "backend_id": bid, "token": None, "denied_reason": "resource_shared_claim_not_member", "resource_owners": sorted(list(owners))}
        token = secrets.token_urlsafe(24)
        res_tokens = dict(control.get("resource_tokens") or {})
        res_tokens[token] = {"resource_kind": rkind, "resource_id": rid, "resource_key": rkey, "backend_id": bid, "issued_at": time.time()}
        control["resource_tokens"] = res_tokens
        self._write_control(control)
        return {"status": "ok", "resource_kind": rkind, "resource_id": rid, "backend_id": bid, "token": token, "issued_at": res_tokens[token]["issued_at"]}

    def validate_resource_token(self, resource_kind: str, resource_id: str, token: str) -> bool:
        rkind = str(resource_kind or "").strip().lower()
        rid = str(resource_id or "").strip()
        if rkind == "engine":
            return self.validate_token(rid, token)
        control = self._read_control()
        meta = dict(control.get("resource_tokens") or {}).get(str(token or "").strip())
        return bool(meta and str(meta.get("resource_kind") or "") == rkind and str(meta.get("resource_id") or "") == rid)
