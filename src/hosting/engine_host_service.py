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
import socket
import ssl
import struct
import tempfile
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
    _ws_lock = threading.Lock()
    _ws_sessions: Optional[Dict[str, Dict[str, Any]]] = None

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
                    "challenge_begin_total": 0,
                    "challenge_complete_ok": 0,
                    "challenge_complete_failed": 0,
                    "challenge_replay_suspected": 0,
                    "challenge_recent_limit": 100,
                    "challenge_recent_events": [],
                },
            }

    @classmethod
    def _ensure_ws_initialized(cls) -> None:
        with cls._ws_lock:
            if isinstance(cls._ws_sessions, dict):
                return
            cls._ws_sessions = {}

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

    @classmethod
    def _metrics_challenge_event(
        cls,
        *,
        event: str,
        key_id: Optional[str] = None,
        challenge_id: Optional[str] = None,
        reason: Optional[str] = None,
        replay_suspected: bool = False,
    ) -> None:
        cls._ensure_metrics_initialized()
        with cls._metrics_lock:
            assert isinstance(cls._runtime_metrics, dict)
            auth = dict(cls._runtime_metrics.get("auth") or {})
            ev = str(event or "").strip().lower()
            if ev == "begin":
                auth["challenge_begin_total"] = int(auth.get("challenge_begin_total") or 0) + 1
            elif ev == "complete_ok":
                auth["challenge_complete_ok"] = int(auth.get("challenge_complete_ok") or 0) + 1
            else:
                auth["challenge_complete_failed"] = int(auth.get("challenge_complete_failed") or 0) + 1
            if replay_suspected:
                auth["challenge_replay_suspected"] = int(auth.get("challenge_replay_suspected") or 0) + 1
            entry = {
                "timestamp": time.time(),
                "event": ev,
                "key_id": str(key_id or "") or None,
                "challenge_id_preview": cls._token_preview(str(challenge_id or ""), prefix=6, suffix=4) if challenge_id else None,
                "reason": str(reason or "") or None,
                "replay_suspected": bool(replay_suspected),
            }
            recent = list(auth.get("challenge_recent_events") or [])
            recent.append(entry)
            limit = max(10, int(auth.get("challenge_recent_limit") or 100))
            if len(recent) > limit:
                recent = recent[-limit:]
            auth["challenge_recent_events"] = recent
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
    def _actor_id_from_session_key(key_id: Optional[str]) -> str:
        kid = str(key_id or "").strip()
        if not kid:
            return "key:unknown"
        return f"key:{kid}"

    @staticmethod
    def _deny_payload(code: str, message: str, **details: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "status": "denied",
            "denied_code": str(code or "denied"),
            "denied_reason": str(message or code or "denied"),
        }
        if details:
            out["details"] = dict(details)
        return out

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
                    "auth": {"keys": {}, "sessions": {}, "challenges": {}},
                    "config_store_mode": "store_only",
                    "claim_acl_policy": {
                        "owner_ttl_seconds": 120,
                        "audit_event_limit": 200,
                    },
                    "engine_traffic_policies": {},
                    "websocket_session_policy": {
                        "max_sessions": 128,
                        "idle_timeout_seconds": 300,
                        "max_lifetime_seconds": 3600,
                    },
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
                "claim_owner_keepalive": {},
                "claim_audit_events": [],
            },
        )
        payload.setdefault(
            "control_config",
            {
                "ssh_key": None,
                "require_auth": False,
                "auth": {"keys": {}, "sessions": {}, "challenges": {}},
                "config_store_mode": "store_only",
                "claim_acl_policy": {},
                "engine_traffic_policies": {},
                "websocket_session_policy": {},
                "traffic_policy": {},
            },
        )
        payload.setdefault("claims_by_engine", {})
        payload.setdefault("endpoint_claim", {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        payload.setdefault("tokens", {})
        payload.setdefault("resource_claims", {})
        payload.setdefault("resource_tokens", {})
        payload.setdefault("claim_owner_keepalive", {})
        payload.setdefault("claim_audit_events", [])
        cfg = dict(payload.get("control_config") or {})
        cfg.setdefault("ssh_key", None)
        cfg.setdefault("require_auth", False)
        cfg.setdefault("config_store_mode", "store_only")
        raw_claim_acl = dict(cfg.get("claim_acl_policy") or {})
        cfg["claim_acl_policy"] = {
            "owner_ttl_seconds": max(10, min(int(raw_claim_acl.get("owner_ttl_seconds") or 120), 24 * 3600)),
            "audit_event_limit": max(20, min(int(raw_claim_acl.get("audit_event_limit") or 200), 2000)),
        }
        cfg.setdefault("engine_traffic_policies", {})
        cfg.setdefault("websocket_session_policy", {})
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
        engine_policies = dict(cfg.get("engine_traffic_policies") or {})
        normalized_engine_policies: Dict[str, Dict[str, Any]] = {}
        for raw_engine_id, policy in engine_policies.items():
            eid = self._safe_config_name(str(raw_engine_id or "").strip())
            if not eid:
                continue
            normalized_engine_policies[eid] = self._normalize_traffic_policy(dict(policy or {}))
        cfg["engine_traffic_policies"] = normalized_engine_policies
        raw_ws = dict(cfg.get("websocket_session_policy") or {})
        cfg["websocket_session_policy"] = self._normalize_websocket_session_policy(raw_ws)
        auth = dict(cfg.get("auth") or {})
        auth.setdefault("keys", {})
        auth.setdefault("sessions", {})
        auth.setdefault("challenges", {})
        cfg["auth"] = auth
        payload["control_config"] = cfg
        return payload

    def _write_control(self, payload: Dict[str, Any]) -> None:
        out = dict(payload or {})
        out["version"] = 1
        out["updated_at"] = time.time()
        self._write_json(self.control_state_file, out)

    @staticmethod
    def _claim_scope_key(scope: str, resource_kind: Optional[str], resource_id: Optional[str]) -> str:
        s = str(scope or "").strip().lower()
        if s == "engine":
            return f"engine:{str(resource_id or '').strip()}"
        if s == "endpoint":
            return "endpoint:*"
        kind = str(resource_kind or "").strip().lower()
        rid = str(resource_id or "").strip()
        return f"resource:{kind}:{rid}"

    def _claim_acl_policy(self, control: Dict[str, Any]) -> Dict[str, int]:
        cfg = dict(control.get("control_config") or {})
        policy = dict(cfg.get("claim_acl_policy") or {})
        return {
            "owner_ttl_seconds": max(10, min(int(policy.get("owner_ttl_seconds") or 120), 24 * 3600)),
            "audit_event_limit": max(20, min(int(policy.get("audit_event_limit") or 200), 2000)),
        }

    def _owner_keepalive_map(self, control: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, val in dict(control.get("claim_owner_keepalive") or {}).items():
            k = str(key or "").strip()
            if not k:
                continue
            try:
                out[k] = float(val)
            except Exception:
                continue
        return out

    def _touch_claim_owner_keepalive(self, control: Dict[str, Any], owner_id: str) -> None:
        oid = str(owner_id or "").strip()
        if not oid:
            return
        keepalive = self._owner_keepalive_map(control)
        keepalive[oid] = time.time()
        control["claim_owner_keepalive"] = keepalive

    def _is_owner_active(self, control: Dict[str, Any], owner_id: str, *, now: Optional[float] = None) -> bool:
        oid = str(owner_id or "").strip()
        if not oid:
            return False
        policy = self._claim_acl_policy(control)
        ttl = float(policy["owner_ttl_seconds"])
        seen = float(self._owner_keepalive_map(control).get(oid) or 0.0)
        current = float(now if now is not None else time.time())
        return seen > 0.0 and (current - seen) <= ttl

    def _active_and_orphan_owners(
        self,
        control: Dict[str, Any],
        owners: List[str],
        *,
        now: Optional[float] = None,
    ) -> Tuple[List[str], List[str]]:
        current = float(now if now is not None else time.time())
        active: List[str] = []
        orphan: List[str] = []
        for owner in [str(x or "").strip() for x in list(owners or []) if str(x or "").strip()]:
            if self._is_owner_active(control, owner, now=current):
                active.append(owner)
            else:
                orphan.append(owner)
        return sorted(list(set(active))), sorted(list(set(orphan)))

    def _append_claim_audit_event(
        self,
        control: Dict[str, Any],
        *,
        event_type: str,
        command: str,
        scope: str,
        resource_kind: Optional[str],
        resource_id: Optional[str],
        actor_id: str,
        decision: str,
        code: str,
        transition: Optional[str],
        mode: Optional[str],
        peer_host: Optional[str],
        owners_before: Optional[List[str]] = None,
        owners_after: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        policy = self._claim_acl_policy(control)
        limit = int(policy["audit_event_limit"])
        rows = list(control.get("claim_audit_events") or [])
        rows.append(
            {
                "schema_version": 1,
                "event_id": secrets.token_urlsafe(10),
                "timestamp": time.time(),
                "event_type": str(event_type or "claim_event"),
                "command": str(command or ""),
                "scope": str(scope or ""),
                "resource_kind": str(resource_kind or "") or None,
                "resource_id": str(resource_id or "") or None,
                "resource_key": self._claim_scope_key(scope, resource_kind, resource_id),
                "actor_id": str(actor_id or ""),
                "peer_host": str(peer_host or "") or None,
                "decision": str(decision or "deny"),
                "code": str(code or "unknown"),
                "transition": str(transition or "") or None,
                "mode": str(mode or "") or None,
                "owners_before": sorted(list(set(str(x or "").strip() for x in list(owners_before or []) if str(x or "").strip()))),
                "owners_after": sorted(list(set(str(x or "").strip() for x in list(owners_after or []) if str(x or "").strip()))),
                "details": dict(details or {}),
            }
        )
        if len(rows) > limit:
            rows = rows[-limit:]
        control["claim_audit_events"] = rows

    def _actor_id_from_payload(self, control: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> str:
        p = dict(payload or {})
        token = self._extract_session_token(p)
        if token:
            auth = dict(dict(control.get("control_config") or {}).get("auth") or {})
            self._prune_expired_sessions(auth)
            session = dict(dict(auth.get("sessions") or {}).get(token) or {})
            key_id = str(session.get("key_id") or "").strip()
            if key_id:
                return self._actor_id_from_session_key(key_id)
        return self._normalize_backend_id(p.get("backend_id"))

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

    def _traffic_policy_for_engine(self, engine_id: str) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        base = self._normalize_traffic_policy(dict(cfg.get("traffic_policy") or {}))
        engine_policies = dict(cfg.get("engine_traffic_policies") or {})
        eid = self._safe_config_name(str(engine_id or "").strip())
        override = dict(engine_policies.get(eid) or {})
        if not override:
            return base
        merged = dict(base)
        merged.update(override)
        return self._normalize_traffic_policy(merged)

    @staticmethod
    def _normalize_websocket_session_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        p = dict(policy or {})
        max_sessions = max(1, min(int(p.get("max_sessions") or 128), 4096))
        idle_timeout = max(5, min(int(p.get("idle_timeout_seconds") or 300), 24 * 3600))
        max_lifetime = max(30, min(int(p.get("max_lifetime_seconds") or 3600), 7 * 24 * 3600))
        return {
            "max_sessions": max_sessions,
            "idle_timeout_seconds": idle_timeout,
            "max_lifetime_seconds": max_lifetime,
        }

    def _websocket_session_policy(self) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        return self._normalize_websocket_session_policy(dict(cfg.get("websocket_session_policy") or {}))

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

    @staticmethod
    def _token_preview(token: str, *, prefix: int = 8, suffix: int = 4) -> str:
        tok = str(token or "").strip()
        if not tok:
            return ""
        if len(tok) <= (prefix + suffix + 3):
            return tok[: max(1, len(tok) // 2)] + "..."
        return f"{tok[:prefix]}...{tok[-suffix:]}"

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

    def _prune_expired_challenges(self, auth: Dict[str, Any]) -> int:
        challenges = dict(auth.get("challenges") or {})
        now = time.time()
        removed = 0
        for challenge_id, meta in list(challenges.items()):
            expires = float((meta or {}).get("expires_at") or 0.0)
            if expires > 0 and now >= expires:
                challenges.pop(challenge_id, None)
                removed += 1
        auth["challenges"] = challenges
        return removed

    @staticmethod
    def _verify_ssh_signature(*, key_id: str, public_key: str, challenge: str, signature_ssh: str) -> bool:
        """
        Verify OpenSSH armored signature over challenge text using ssh-keygen -Y verify.

        Returns False on verification failure or tool error.
        """
        kid = str(key_id or "").strip()
        pub = str(public_key or "").strip()
        ch = str(challenge or "")
        sig = str(signature_ssh or "").strip()
        if not (kid and pub and ch and sig):
            return False
        try:
            with tempfile.TemporaryDirectory(prefix="host_auth_") as td:
                tdp = Path(td)
                data_file = tdp / "challenge.txt"
                sig_file = tdp / "challenge.sig"
                allowed_file = tdp / "allowed_signers"
                data_file.write_text(ch, encoding="utf-8")
                sig_file.write_text(sig, encoding="utf-8")
                allowed_file.write_text(f"{kid} {pub}\n", encoding="utf-8")
                proc = subprocess.run(  # noqa: S603
                    [
                        "ssh-keygen",
                        "-Y",
                        "verify",
                        "-f",
                        str(allowed_file),
                        "-I",
                        kid,
                        "-n",
                        "engine-host-auth",
                        "-s",
                        str(sig_file),
                    ],
                    input=ch,
                    text=True,
                    capture_output=True,
                    timeout=15.0,
                    check=False,
                )
                return int(proc.returncode) == 0
        except Exception:
            return False

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
        presented_ssh_binding: Optional[Dict[str, Any]] = None,
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
        expected_binding = dict(session.get("ssh_binding") or {})
        if expected_binding:
            presented = dict(presented_ssh_binding or {})
            if not presented:
                raise PermissionError("ssh_binding_required")
            expected_target = str(expected_binding.get("target") or "").strip()
            expected_fp = str(expected_binding.get("key_fingerprint") or "").strip()
            got_target = str(presented.get("target") or "").strip()
            got_fp = str(presented.get("key_fingerprint") or "").strip()
            if expected_target and expected_target != got_target:
                raise PermissionError("ssh_binding_mismatch")
            if expected_fp and expected_fp != got_fp:
                raise PermissionError("ssh_binding_mismatch")
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
        self._prune_expired_challenges(auth)
        keys = dict(auth.get("keys") or {})
        sessions = dict(auth.get("sessions") or {})
        challenges = dict(auth.get("challenges") or {})
        return {
            "require_auth": bool(cfg.get("require_auth", False)),
            "config_store_mode": str(cfg.get("config_store_mode") or "store_only"),
            "keys_count": len(keys),
            "sessions_count": len(sessions),
            "challenges_count": len(challenges),
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
                    "auth_method": str(m.get("auth_method") or "shared_secret"),
                    "created_at": float(m.get("created_at") or 0.0),
                    "updated_at": float(m.get("updated_at") or 0.0),
                    "allowed_configs": list(m.get("allowed_configs") or []),
                    "allowed_engines": list(m.get("allowed_engines") or []),
                }
            )
        out.sort(key=lambda x: str(x.get("key_id") or ""))
        return out

    def auth_list_sessions(
        self,
        *,
        key_id: Optional[str] = None,
        scope: Optional[str] = None,
        role: Optional[str] = None,
        token_preview_contains: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        self._prune_expired_sessions(auth)
        sessions = dict(auth.get("sessions") or {})
        now = time.time()
        rows: List[Dict[str, Any]] = []
        key_id_filter = str(key_id or "").strip()
        scope_filter = str(scope or "").strip().lower()
        role_filter = str(role or "").strip().lower()
        preview_filter = str(token_preview_contains or "").strip().lower()
        for token, meta in sessions.items():
            m = dict(meta or {})
            expires_at = float(m.get("expires_at") or 0.0)
            remaining = max(0, int(expires_at - now)) if expires_at > 0 else None
            row = {
                "token_preview": self._token_preview(str(token)),
                "key_id": str(m.get("key_id") or ""),
                "role": str(m.get("role") or ""),
                "scope": str(m.get("scope") or ""),
                "issued_at": float(m.get("issued_at") or 0.0),
                "expires_at": expires_at,
                "ttl_remaining_seconds": remaining,
                "allowed_configs": list(m.get("allowed_configs") or []),
                "allowed_engines": list(m.get("allowed_engines") or []),
                "ssh_binding": dict(m.get("ssh_binding") or {}),
            }
            if key_id_filter and str(row["key_id"]) != key_id_filter:
                continue
            if scope_filter and str(row["scope"]).lower() != scope_filter:
                continue
            if role_filter and str(row["role"]).lower() != role_filter:
                continue
            if preview_filter and preview_filter not in str(row["token_preview"]).lower():
                continue
            rows.append(row)
        rows.sort(key=lambda x: (str(x.get("key_id") or ""), float(x.get("issued_at") or 0.0)))
        total = len(rows)
        page_offset = max(0, int(offset or 0))
        page_limit = max(1, min(int(limit or 100), 1000))
        page = rows[page_offset: page_offset + page_limit]
        next_offset = page_offset + len(page)
        return {
            "sessions_count": total,
            "timestamp": now,
            "offset": page_offset,
            "limit": page_limit,
            "count": len(page),
            "has_more": bool(next_offset < total),
            "next_offset": next_offset if next_offset < total else None,
            "sessions": page,
        }

    def auth_list_issued_tokens(
        self,
        *,
        engine_id: Optional[str] = None,
        resource_kind: Optional[str] = None,
        resource_id: Optional[str] = None,
        backend_id: Optional[str] = None,
        token_preview_contains: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        control = self._read_control()
        now = time.time()
        engine_tokens: List[Dict[str, Any]] = []
        resource_tokens: List[Dict[str, Any]] = []
        engine_filter = str(engine_id or "").strip()
        rk_filter = str(resource_kind or "").strip().lower()
        rid_filter = str(resource_id or "").strip()
        backend_filter = str(backend_id or "").strip()
        preview_filter = str(token_preview_contains or "").strip().lower()
        for token, meta in dict(control.get("tokens") or {}).items():
            m = dict(meta or {})
            row = {
                "token_preview": self._token_preview(str(token)),
                "engine_id": str(m.get("engine_id") or ""),
                "backend_id": str(m.get("backend_id") or ""),
                "issued_at": float(m.get("issued_at") or 0.0),
            }
            if engine_filter and str(row["engine_id"]) != engine_filter:
                continue
            if backend_filter and str(row["backend_id"]) != backend_filter:
                continue
            if preview_filter and preview_filter not in str(row["token_preview"]).lower():
                continue
            engine_tokens.append(row)
        for token, meta in dict(control.get("resource_tokens") or {}).items():
            m = dict(meta or {})
            row = {
                "token_preview": self._token_preview(str(token)),
                "resource_kind": str(m.get("resource_kind") or ""),
                "resource_id": str(m.get("resource_id") or ""),
                "resource_key": str(m.get("resource_key") or ""),
                "backend_id": str(m.get("backend_id") or ""),
                "issued_at": float(m.get("issued_at") or 0.0),
            }
            if rk_filter and str(row["resource_kind"]).lower() != rk_filter:
                continue
            if rid_filter and str(row["resource_id"]) != rid_filter:
                continue
            if backend_filter and str(row["backend_id"]) != backend_filter:
                continue
            if preview_filter and preview_filter not in str(row["token_preview"]).lower():
                continue
            resource_tokens.append(row)
        engine_tokens.sort(key=lambda x: (str(x.get("engine_id") or ""), float(x.get("issued_at") or 0.0)))
        resource_tokens.sort(key=lambda x: (str(x.get("resource_key") or ""), float(x.get("issued_at") or 0.0)))
        merged: List[Dict[str, Any]] = []
        for row in engine_tokens:
            merged.append({"kind": "engine", **row})
        for row in resource_tokens:
            merged.append({"kind": "resource", **row})
        merged.sort(
            key=lambda x: (
                str(x.get("kind") or ""),
                str(x.get("engine_id") or x.get("resource_key") or ""),
                float(x.get("issued_at") or 0.0),
            )
        )
        total = len(merged)
        page_offset = max(0, int(offset or 0))
        page_limit = max(1, min(int(limit or 100), 1000))
        page = merged[page_offset: page_offset + page_limit]
        next_offset = page_offset + len(page)
        return {
            "timestamp": now,
            "engine_tokens_count": len(engine_tokens),
            "resource_tokens_count": len(resource_tokens),
            "engine_tokens": engine_tokens,
            "resource_tokens": resource_tokens,
            "total_count": total,
            "offset": page_offset,
            "limit": page_limit,
            "count": len(page),
            "has_more": bool(next_offset < total),
            "next_offset": next_offset if next_offset < total else None,
            "tokens": page,
        }

    def auth_upsert_key(
        self,
        *,
        key_id: str,
        key_secret: str = "",
        role: str,
        auth_method: str = "shared_secret",
        public_key: str = "",
        allowed_configs: Optional[List[str]] = None,
        allowed_engines: Optional[List[str]] = None,
        disabled: bool = False,
    ) -> Dict[str, Any]:
        kid = str(key_id or "").strip()
        secret = str(key_secret or "").strip()
        role_norm = str(role or "").strip().lower()
        method = str(auth_method or "shared_secret").strip().lower()
        pubkey = str(public_key or "").strip()
        if not kid:
            raise ValueError("key_id is required")
        if role_norm not in {"management", "config", "traffic"}:
            raise ValueError("role must be 'management', 'config', or 'traffic'")
        if method not in {"shared_secret", "public_key"}:
            raise ValueError("auth_method must be 'shared_secret' or 'public_key'")
        if method == "shared_secret" and not secret:
            raise ValueError("key_secret is required for shared_secret auth_method")
        if method == "public_key" and not pubkey:
            raise ValueError("public_key is required for public_key auth_method")
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
            "auth_method": method,
            "secret_hash": self._hash_secret(secret) if method == "shared_secret" else "",
            "public_key": pubkey if method == "public_key" else "",
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
            "auth_method": method,
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
        ssh_binding: Optional[Dict[str, Any]] = None,
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
        auth_method = str(key_meta.get("auth_method") or "shared_secret").strip().lower()
        if auth_method != "shared_secret":
            raise PermissionError("auth_method_requires_challenge_flow")
        expected_hash = str(key_meta.get("secret_hash") or "")
        provided_hash = self._hash_secret(secret)
        if not expected_hash or not hmac.compare_digest(expected_hash, provided_hash):
            raise PermissionError("invalid_key_secret")
        return self._issue_session_for_key(
            key_id=kid,
            key_meta=key_meta,
            scope=scope_norm,
            ttl_seconds=ttl_seconds,
            config_paths=config_paths,
            engine_ids=engine_ids,
            ssh_binding=ssh_binding,
            control=control,
        )

    def _issue_session_for_key(
        self,
        *,
        key_id: str,
        key_meta: Dict[str, Any],
        scope: str,
        ttl_seconds: int,
        config_paths: Optional[List[str]],
        engine_ids: Optional[List[str]],
        ssh_binding: Optional[Dict[str, Any]],
        control: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        scope_norm = str(scope or "control").strip().lower()
        if scope_norm not in {"control", "config", "traffic"}:
            raise ValueError("scope must be 'control', 'config', or 'traffic'")
        control_payload = dict(control or self._read_control())
        cfg = dict(control_payload.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        self._prune_expired_sessions(auth)
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
        binding_target = str((ssh_binding or {}).get("target") or "").strip()
        binding_fp = str((ssh_binding or {}).get("key_fingerprint") or "").strip()
        normalized_binding = {
            "target": binding_target or None,
            "key_fingerprint": binding_fp or None,
        } if (binding_target or binding_fp) else {}
        sessions = dict(auth.get("sessions") or {})
        sessions[token] = {
            "key_id": str(key_id or ""),
            "role": role,
            "scope": scope_norm,
            "issued_at": now,
            "expires_at": now + ttl,
            "allowed_configs": allowed_configs,
            "allowed_engines": allowed_engines,
            "ssh_binding": normalized_binding,
        }
        auth["sessions"] = sessions
        cfg["auth"] = auth
        control_payload["control_config"] = cfg
        self._write_control(control_payload)
        return {
            "status": "ok",
            "token": token,
            "scope": scope_norm,
            "role": role,
            "expires_at": now + ttl,
            "ttl_seconds": ttl,
            "allowed_configs": allowed_configs,
            "allowed_engines": allowed_engines,
            "ssh_binding": normalized_binding,
        }

    def auth_begin_challenge(
        self,
        *,
        key_id: str,
        scope: str = "control",
        ttl_seconds: int = 120,
        config_paths: Optional[List[str]] = None,
        engine_ids: Optional[List[str]] = None,
        ssh_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kid = str(key_id or "").strip()
        if not kid:
            self._metrics_challenge_event(event="begin_failed", key_id=kid or None, reason="key_id_required")
            raise ValueError("key_id is required")
        scope_norm = str(scope or "control").strip().lower()
        if scope_norm not in {"control", "config", "traffic"}:
            self._metrics_challenge_event(event="begin_failed", key_id=kid, reason="invalid_scope")
            raise ValueError("scope must be 'control', 'config', or 'traffic'")
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        self._prune_expired_challenges(auth)
        keys = dict(auth.get("keys") or {})
        key_meta = dict(keys.get(kid) or {})
        if not key_meta:
            self._metrics_challenge_event(event="begin_failed", key_id=kid, reason="unknown_key_id")
            raise PermissionError("unknown_key_id")
        if bool(key_meta.get("disabled", False)):
            self._metrics_challenge_event(event="begin_failed", key_id=kid, reason="key_disabled")
            raise PermissionError("key_disabled")
        auth_method = str(key_meta.get("auth_method") or "shared_secret").strip().lower()
        if auth_method != "public_key":
            self._metrics_challenge_event(event="begin_failed", key_id=kid, reason="auth_method_is_not_public_key")
            raise PermissionError("auth_method_is_not_public_key")
        role = str(key_meta.get("role") or "").strip().lower()
        if role == "config" and scope_norm != "config":
            self._metrics_challenge_event(event="begin_failed", key_id=kid, reason="config_role_cannot_issue_non_config_scope")
            raise PermissionError("config_role_cannot_issue_non_config_scope")
        if role == "traffic" and scope_norm != "traffic":
            self._metrics_challenge_event(event="begin_failed", key_id=kid, reason="traffic_role_cannot_issue_non_traffic_scope")
            raise PermissionError("traffic_role_cannot_issue_non_traffic_scope")
        challenge_id = secrets.token_urlsafe(18)
        nonce = secrets.token_urlsafe(24)
        issued_at = time.time()
        ttl = max(30, min(int(ttl_seconds or 120), 600))
        expires_at = issued_at + ttl
        binding_target = str((ssh_binding or {}).get("target") or "").strip()
        binding_fp = str((ssh_binding or {}).get("key_fingerprint") or "").strip()
        normalized_binding = {
            "target": binding_target or None,
            "key_fingerprint": binding_fp or None,
        } if (binding_target or binding_fp) else {}
        challenge_text = json.dumps(
            {
                "kind": "engine-host-auth-challenge",
                "challenge_id": challenge_id,
                "key_id": kid,
                "nonce": nonce,
                "issued_at": issued_at,
                "expires_at": expires_at,
                "scope": scope_norm,
                "ssh_binding_target": normalized_binding.get("target"),
                "ssh_binding_key_fingerprint": normalized_binding.get("key_fingerprint"),
            },
            separators=(",", ":"),
        )
        challenges = dict(auth.get("challenges") or {})
        challenges[challenge_id] = {
            "key_id": kid,
            "scope": scope_norm,
            "issued_at": issued_at,
            "expires_at": expires_at,
            "config_paths": list(config_paths or []),
            "engine_ids": list(engine_ids or []),
            "ssh_binding": normalized_binding,
            "challenge": challenge_text,
        }
        auth["challenges"] = challenges
        cfg["auth"] = auth
        control["control_config"] = cfg
        self._write_control(control)
        self._metrics_challenge_event(event="begin", key_id=kid, challenge_id=challenge_id)
        return {
            "status": "ok",
            "challenge_id": challenge_id,
            "key_id": kid,
            "scope": scope_norm,
            "challenge": challenge_text,
            "expires_at": expires_at,
            "ttl_seconds": ttl,
        }

    def auth_complete_challenge(
        self,
        *,
        challenge_id: str,
        signature_ssh: str,
        presented_ssh_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cid = str(challenge_id or "").strip()
        sig = str(signature_ssh or "").strip()
        if not cid:
            self._metrics_challenge_event(event="complete_failed", reason="challenge_id_required")
            raise ValueError("challenge_id is required")
        if not sig:
            self._metrics_challenge_event(event="complete_failed", challenge_id=cid, reason="signature_required")
            raise ValueError("signature_ssh is required")
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        self._prune_expired_challenges(auth)
        challenges = dict(auth.get("challenges") or {})
        item = dict(challenges.get(cid) or {})
        if not item:
            self._metrics_challenge_event(
                event="complete_failed",
                challenge_id=cid,
                reason="missing_or_expired_challenge",
                replay_suspected=True,
            )
            raise PermissionError("missing_or_expired_challenge")
        key_id = str(item.get("key_id") or "").strip()
        scope = str(item.get("scope") or "control").strip().lower()
        expected_binding = dict(item.get("ssh_binding") or {})
        if expected_binding:
            presented = dict(presented_ssh_binding or {})
            expected_target = str(expected_binding.get("target") or "").strip()
            expected_fp = str(expected_binding.get("key_fingerprint") or "").strip()
            got_target = str(presented.get("target") or "").strip()
            got_fp = str(presented.get("key_fingerprint") or "").strip()
            if (expected_target and expected_target != got_target) or (expected_fp and expected_fp != got_fp):
                self._metrics_challenge_event(
                    event="complete_failed",
                    key_id=key_id,
                    challenge_id=cid,
                    reason="ssh_binding_mismatch",
                    replay_suspected=True,
                )
                raise PermissionError("ssh_binding_mismatch")
        keys = dict(auth.get("keys") or {})
        key_meta = dict(keys.get(key_id) or {})
        if not key_meta:
            self._metrics_challenge_event(event="complete_failed", key_id=key_id, challenge_id=cid, reason="unknown_key_id")
            raise PermissionError("unknown_key_id")
        if bool(key_meta.get("disabled", False)):
            self._metrics_challenge_event(event="complete_failed", key_id=key_id, challenge_id=cid, reason="key_disabled")
            raise PermissionError("key_disabled")
        auth_method = str(key_meta.get("auth_method") or "shared_secret").strip().lower()
        if auth_method != "public_key":
            self._metrics_challenge_event(event="complete_failed", key_id=key_id, challenge_id=cid, reason="auth_method_is_not_public_key")
            raise PermissionError("auth_method_is_not_public_key")
        public_key = str(key_meta.get("public_key") or "").strip()
        challenge_text = str(item.get("challenge") or "")
        if not self._verify_ssh_signature(
            key_id=key_id,
            public_key=public_key,
            challenge=challenge_text,
            signature_ssh=sig,
        ):
            self._metrics_challenge_event(
                event="complete_failed",
                key_id=key_id,
                challenge_id=cid,
                reason="invalid_challenge_signature",
                replay_suspected=True,
            )
            raise PermissionError("invalid_challenge_signature")
        # one-time challenge
        challenges.pop(cid, None)
        auth["challenges"] = challenges
        cfg["auth"] = auth
        control["control_config"] = cfg
        self._write_control(control)
        role = str(key_meta.get("role") or "").strip().lower()
        if role not in {"management", "config", "traffic"}:
            self._metrics_challenge_event(event="complete_failed", key_id=key_id, challenge_id=cid, reason="invalid_role")
            return {"status": "denied"}
        out = self._issue_session_for_key(
            key_id=key_id,
            key_meta=key_meta,
            scope=scope,
            ttl_seconds=900,
            config_paths=list(item.get("config_paths") or []),
            engine_ids=list(item.get("engine_ids") or []),
            ssh_binding=dict(item.get("ssh_binding") or {}),
            control=control,
        )
        self._metrics_challenge_event(event="complete_ok", key_id=key_id, challenge_id=cid)
        return out

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
        if c in {"auth-begin-challenge", "auth-complete-challenge"}:
            # Challenge issuance/completion perform their own key-based verification.
            return
        token = self._extract_session_token(payload)
        if not token:
            self._metrics_auth_denied("session_token_required")
            raise PermissionError("session_token_required")
        presented_ssh_binding = dict((payload or {}).get("_ssh_session_binding") or {})
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
            "auth-list-sessions",
            "auth-list-issued-tokens",
            "auth-revoke-session",
            "host-metrics",
        }:
            try:
                _ = self._validate_session(
                    control,
                    token,
                    required_scope="control",
                    presented_ssh_binding=presented_ssh_binding,
                )
            except PermissionError as exc:
                self._metrics_auth_denied(str(exc))
                raise
            return
        if c in {"proxy-request"}:
            p = dict(payload or {})
            requested_engine = str(p.get("engine_id") or "").strip()
            try:
                _ = self._validate_session(
                    control,
                    token,
                    required_scope="traffic",
                    requested_engine=requested_engine,
                    presented_ssh_binding=presented_ssh_binding,
                )
            except PermissionError as exc:
                self._metrics_auth_denied(str(exc))
                raise
            return
        if c in {"proxy-ws-open", "proxy-ws-send", "proxy-ws-recv", "proxy-ws-close"}:
            p = dict(payload or {})
            requested_engine = str(p.get("engine_id") or "").strip()
            if c in {"proxy-ws-send", "proxy-ws-recv", "proxy-ws-close"} and not requested_engine:
                ws_id = str(p.get("ws_id") or "").strip()
                sess = self._ws_session_get(ws_id) if ws_id else None
                requested_engine = str((sess or {}).get("engine_id") or "").strip()
            try:
                _ = self._validate_session(
                    control,
                    token,
                    required_scope="traffic",
                    requested_engine=requested_engine,
                    presented_ssh_binding=presented_ssh_binding,
                )
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
                _ = self._validate_session(
                    control,
                    token,
                    required_scope="config",
                    requested_config=requested_config,
                    presented_ssh_binding=presented_ssh_binding,
                )
            except PermissionError as exc:
                self._metrics_auth_denied(str(exc))
                raise
            return
        raise PermissionError(f"auth_policy_missing_for_command:{c}")

    def enforce_daemon_claim_policy(
        self,
        cmd: str,
        payload: Optional[Dict[str, Any]],
        *,
        peer_host: Optional[str],
        is_localhost: bool,
    ) -> Dict[str, Any]:
        c = str(cmd or "").strip()
        p = dict(payload or {})
        control = self._read_control()
        actor_id = self._actor_id_from_payload(control, p)
        p["backend_id"] = actor_id
        p["_claim_actor_id"] = actor_id
        p["_daemon_peer_host"] = str(peer_host or "") or None

        claim_cmds = {"claim-engine", "claim-endpoint", "claim-resource"}
        sensitive_engine_cmds = {
            "spawn",
            "get-registration",
            "shutdown",
            "ensure-running",
            "remove-registration",
            "logs-tail",
            "logs-follow",
            "inspect-capabilities",
            "issue-token",
            "issue-resource-token",
        }

        if c in claim_cmds and (not is_localhost) and (not bool(p.get("exclusive", False))):
            return {
                "ok": False,
                "error": "access_denied",
                "error_code": "non_localhost_shared_claim_denied",
                "error_details": {
                    "command": c,
                    "actor_id": actor_id,
                    "peer_host": str(peer_host or ""),
                },
                "payload": p,
            }

        if c in claim_cmds and bool(p.get("force_override", False)) and is_localhost:
            confirmation = str(p.get("force_override_confirmation") or "").strip()
            if confirmation != "CONFIRM_LOCALHOST_FORCE_OVERRIDE":
                return {
                    "ok": False,
                    "error": "access_denied",
                    "error_code": "localhost_force_override_confirmation_required",
                    "error_details": {
                        "command": c,
                        "actor_id": actor_id,
                    },
                    "payload": p,
                }

        if c in sensitive_engine_cmds:
            if c in {"issue-resource-token"} and str(p.get("resource_kind") or "").strip().lower() != "engine":
                # Non-engine resource token checks are enforced in issue_resource_token().
                return {"ok": True, "payload": p}
            engine_id = str(p.get("engine_id") or p.get("resource_id") or "").strip()
            if engine_id:
                claim = dict((control.get("claims_by_engine") or {}).get(engine_id) or {})
                owners = [str(x or "").strip() for x in list(claim.get("owners") or []) if str(x or "").strip()]
                active_owners, _ = self._active_and_orphan_owners(control, owners)
                exclusive_owner = str(claim.get("exclusive_owner") or "").strip()
                if exclusive_owner and (exclusive_owner not in active_owners):
                    exclusive_owner = ""
                if exclusive_owner and exclusive_owner != actor_id:
                    return {
                        "ok": False,
                        "error": "access_denied",
                        "error_code": "engine_exclusive_conflict",
                        "error_details": {
                            "engine_id": engine_id,
                            "actor_id": actor_id,
                            "engine_exclusive_owner": exclusive_owner,
                        },
                        "payload": p,
                    }
                if active_owners and actor_id not in active_owners:
                    return {
                        "ok": False,
                        "error": "access_denied",
                        "error_code": "engine_shared_claim_not_member",
                        "error_details": {
                            "engine_id": engine_id,
                            "actor_id": actor_id,
                            "engine_owners": active_owners,
                        },
                        "payload": p,
                    }
                self._touch_claim_owner_keepalive(control, actor_id)
                self._write_control(control)
        return {"ok": True, "payload": p}

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
        engine_policies = dict(cfg.get("engine_traffic_policies") or {})
        return {
            "ssh_key": cfg.get("ssh_key"),
            "require_auth": bool(cfg.get("require_auth", False)),
            "config_store_mode": str(cfg.get("config_store_mode") or "store_only"),
            "claim_acl_policy": self._claim_acl_policy(control),
            "traffic_policy": self._normalize_traffic_policy(dict(cfg.get("traffic_policy") or {})),
            "engine_traffic_policies": {
                str(k): self._normalize_traffic_policy(dict(v or {}))
                for k, v in engine_policies.items()
            },
            "engine_traffic_policies_count": len(engine_policies),
            "websocket_session_policy": self._normalize_websocket_session_policy(
                dict(cfg.get("websocket_session_policy") or {})
            ),
            "keys_count": len(dict(auth.get("keys") or {})),
            "sessions_count": len(dict(auth.get("sessions") or {})),
        }

    def set_control_config(
        self,
        *,
        ssh_key: Optional[str] = None,
        require_auth: Optional[bool] = None,
        traffic_policy: Optional[Dict[str, Any]] = None,
        engine_traffic_policies: Optional[Dict[str, Dict[str, Any]]] = None,
        websocket_session_policy: Optional[Dict[str, Any]] = None,
        claim_acl_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        if ssh_key is not None:
            cfg["ssh_key"] = str(ssh_key).strip() if ssh_key else None
        if require_auth is not None:
            cfg["require_auth"] = bool(require_auth)
        cfg.setdefault("config_store_mode", "store_only")
        cfg.setdefault("auth", {"keys": {}, "sessions": {}})
        cfg.setdefault("engine_traffic_policies", {})
        cfg.setdefault("websocket_session_policy", {})
        cfg.setdefault("claim_acl_policy", {"owner_ttl_seconds": 120, "audit_event_limit": 200})
        cfg["traffic_policy"] = self._normalize_traffic_policy(
            dict(cfg.get("traffic_policy") or {}) | dict(traffic_policy or {})
        )
        if engine_traffic_policies is not None:
            incoming = dict(engine_traffic_policies or {})
            normalized: Dict[str, Dict[str, Any]] = {}
            for raw_engine_id, policy in incoming.items():
                eid = self._safe_config_name(str(raw_engine_id or "").strip())
                if not eid:
                    continue
                normalized[eid] = self._normalize_traffic_policy(dict(policy or {}))
            cfg["engine_traffic_policies"] = normalized
        if websocket_session_policy is not None:
            cfg["websocket_session_policy"] = self._normalize_websocket_session_policy(
                dict(websocket_session_policy or {})
            )
        if claim_acl_policy is not None:
            raw_claim_acl = dict(cfg.get("claim_acl_policy") or {}) | dict(claim_acl_policy or {})
            cfg["claim_acl_policy"] = {
                "owner_ttl_seconds": max(10, min(int(raw_claim_acl.get("owner_ttl_seconds") or 120), 24 * 3600)),
                "audit_event_limit": max(20, min(int(raw_claim_acl.get("audit_event_limit") or 200), 2000)),
            }
        control["control_config"] = cfg
        self._write_control(control)
        auth = dict(cfg.get("auth") or {})
        engine_policies = dict(cfg.get("engine_traffic_policies") or {})
        return {
            "ssh_key": cfg.get("ssh_key"),
            "require_auth": bool(cfg.get("require_auth", False)),
            "config_store_mode": str(cfg.get("config_store_mode") or "store_only"),
            "claim_acl_policy": self._claim_acl_policy(control),
            "traffic_policy": self._normalize_traffic_policy(dict(cfg.get("traffic_policy") or {})),
            "engine_traffic_policies": {
                str(k): self._normalize_traffic_policy(dict(v or {}))
                for k, v in engine_policies.items()
            },
            "engine_traffic_policies_count": len(engine_policies),
            "websocket_session_policy": self._normalize_websocket_session_policy(
                dict(cfg.get("websocket_session_policy") or {})
            ),
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

    @staticmethod
    def _to_ws_url(endpoint: str, req_path: str, query: str = "") -> str:
        base = EngineHostService._join_endpoint_path(endpoint, req_path, query=query)
        if not base:
            return ""
        parsed = urllib.parse.urlsplit(base)
        scheme = str(parsed.scheme or "http").lower()
        if scheme in {"ws", "wss"}:
            ws_scheme = scheme
        else:
            ws_scheme = "wss" if scheme == "https" else "ws"
        return urllib.parse.urlunsplit((ws_scheme, parsed.netloc, parsed.path, parsed.query, ""))

    @staticmethod
    def _read_until(sock: socket.socket, marker: bytes, *, max_bytes: int = 65536, timeout_seconds: float = 10.0) -> bytes:
        buf = bytearray()
        sock.settimeout(max(0.2, float(timeout_seconds or 10.0)))
        while marker not in buf and len(buf) < max_bytes:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    @staticmethod
    def _read_exact(sock: socket.socket, n: int, *, timeout_seconds: float = 10.0) -> bytes:
        out = bytearray()
        sock.settimeout(max(0.2, float(timeout_seconds or 10.0)))
        while len(out) < n:
            chunk = sock.recv(n - len(out))
            if not chunk:
                raise ConnectionError("socket_closed")
            out.extend(chunk)
        return bytes(out)

    @staticmethod
    def _ws_frame_encode(opcode: int, payload: bytes, *, masked: bool = True, fin: bool = True) -> bytes:
        first = (0x80 if fin else 0x00) | (int(opcode) & 0x0F)
        plen = len(payload)
        mask_bit = 0x80 if masked else 0x00
        if plen < 126:
            head = bytes([first, mask_bit | plen])
        elif plen <= 0xFFFF:
            head = bytes([first, mask_bit | 126]) + struct.pack("!H", plen)
        else:
            head = bytes([first, mask_bit | 127]) + struct.pack("!Q", plen)
        if not masked:
            return head + payload
        mask_key = os.urandom(4)
        masked_payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return head + mask_key + masked_payload

    @staticmethod
    def _ws_frame_read(sock: socket.socket, *, max_bytes: int = 1024 * 1024, timeout_seconds: float = 30.0) -> Dict[str, Any]:
        head = EngineHostService._read_exact(sock, 2, timeout_seconds=timeout_seconds)
        b1, b2 = head[0], head[1]
        fin = bool(b1 & 0x80)
        opcode = int(b1 & 0x0F)
        masked = bool(b2 & 0x80)
        plen = int(b2 & 0x7F)
        if plen == 126:
            plen = int(struct.unpack("!H", EngineHostService._read_exact(sock, 2, timeout_seconds=timeout_seconds))[0])
        elif plen == 127:
            plen = int(struct.unpack("!Q", EngineHostService._read_exact(sock, 8, timeout_seconds=timeout_seconds))[0])
        if plen > int(max_bytes or (1024 * 1024)):
            raise ValueError("websocket_frame_too_large")
        mask_key = EngineHostService._read_exact(sock, 4, timeout_seconds=timeout_seconds) if masked else b""
        payload = EngineHostService._read_exact(sock, plen, timeout_seconds=timeout_seconds) if plen > 0 else b""
        if masked and mask_key:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return {"fin": fin, "opcode": opcode, "payload": payload}

    @classmethod
    def _ws_session_get(cls, ws_id: str) -> Optional[Dict[str, Any]]:
        cls._ensure_ws_initialized()
        with cls._ws_lock:
            assert isinstance(cls._ws_sessions, dict)
            sess = cls._ws_sessions.get(str(ws_id or "").strip())
            return dict(sess or {}) if isinstance(sess, dict) else None

    @classmethod
    def _ws_session_set(cls, ws_id: str, sess: Dict[str, Any]) -> None:
        cls._ensure_ws_initialized()
        with cls._ws_lock:
            assert isinstance(cls._ws_sessions, dict)
            cls._ws_sessions[str(ws_id)] = dict(sess or {})

    @classmethod
    def _ws_session_pop(cls, ws_id: str) -> Optional[Dict[str, Any]]:
        cls._ensure_ws_initialized()
        with cls._ws_lock:
            assert isinstance(cls._ws_sessions, dict)
            sess = cls._ws_sessions.pop(str(ws_id or "").strip(), None)
            return dict(sess or {}) if isinstance(sess, dict) else None

    @classmethod
    def _ws_cleanup(cls, *, policy: Dict[str, Any], now: Optional[float] = None) -> Dict[str, int]:
        cls._ensure_ws_initialized()
        ts_now = float(now if now is not None else time.time())
        idle_limit = int(policy.get("idle_timeout_seconds") or 300)
        life_limit = int(policy.get("max_lifetime_seconds") or 3600)
        max_sessions = int(policy.get("max_sessions") or 128)
        to_close: List[socket.socket] = []
        removed_idle = 0
        removed_lifetime = 0
        removed_cap = 0
        with cls._ws_lock:
            assert isinstance(cls._ws_sessions, dict)
            sessions = cls._ws_sessions
            for sid, sess in list(sessions.items()):
                s = dict(sess or {})
                created_at = float(s.get("created_at") or ts_now)
                last_io = float(s.get("last_io_at") or created_at)
                remove_reason = ""
                if ts_now - created_at > life_limit:
                    remove_reason = "lifetime"
                elif ts_now - last_io > idle_limit:
                    remove_reason = "idle"
                if remove_reason:
                    popped = sessions.pop(sid, None)
                    sock = (popped or {}).get("socket") if isinstance(popped, dict) else None
                    if isinstance(sock, socket.socket):
                        to_close.append(sock)
                    if remove_reason == "idle":
                        removed_idle += 1
                    else:
                        removed_lifetime += 1
            if len(sessions) > max_sessions:
                ordered = sorted(
                    sessions.items(),
                    key=lambda item: float(((item[1] or {}).get("last_io_at") or 0.0)),
                )
                overflow = len(sessions) - max_sessions
                for sid, _ in ordered[:overflow]:
                    popped = sessions.pop(sid, None)
                    sock = (popped or {}).get("socket") if isinstance(popped, dict) else None
                    if isinstance(sock, socket.socket):
                        to_close.append(sock)
                    removed_cap += 1
        for sock in to_close:
            try:
                sock.close()
            except Exception:
                pass
        return {
            "removed_idle": removed_idle,
            "removed_lifetime": removed_lifetime,
            "removed_cap": removed_cap,
        }

    @classmethod
    def _ws_count(cls) -> int:
        cls._ensure_ws_initialized()
        with cls._ws_lock:
            assert isinstance(cls._ws_sessions, dict)
            return len(cls._ws_sessions)

    @classmethod
    def _ws_evict_oldest(cls, *, count: int = 1) -> int:
        cls._ensure_ws_initialized()
        to_close: List[socket.socket] = []
        evicted = 0
        with cls._ws_lock:
            assert isinstance(cls._ws_sessions, dict)
            sessions = cls._ws_sessions
            ordered = sorted(
                sessions.items(),
                key=lambda item: float(((item[1] or {}).get("last_io_at") or 0.0)),
            )
            for sid, _ in ordered[: max(0, int(count or 0))]:
                popped = sessions.pop(sid, None)
                sock = (popped or {}).get("socket") if isinstance(popped, dict) else None
                if isinstance(sock, socket.socket):
                    to_close.append(sock)
                evicted += 1
        for sock in to_close:
            try:
                sock.close()
            except Exception:
                pass
        return evicted

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
        traffic_policy = self._traffic_policy_for_engine(eid)
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

    def proxy_ws_open(
        self,
        *,
        engine_id: str,
        path: str = "/",
        query: str = "",
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        ws_policy = self._websocket_session_policy()
        _ = self._ws_cleanup(policy=ws_policy)
        max_sessions = int(ws_policy.get("max_sessions") or 128)
        active = self._ws_count()
        if active >= max_sessions:
            _ = self._ws_evict_oldest(count=(active - max_sessions + 1))
        eid = str(engine_id or "").strip()
        req_path = str(path or "/").strip() or "/"
        if not req_path.startswith("/"):
            req_path = f"/{req_path}"
        if not eid:
            raise ValueError("engine_id is required")
        reg = self._find_registration(eid) or {}
        endpoint = str(reg.get("endpoint") or "").strip()
        if not endpoint:
            raise ValueError("engine endpoint is not registered")
        ws_url = self._to_ws_url(endpoint, req_path, query=query)
        if not ws_url:
            raise ValueError("failed to build websocket url")
        traffic_policy = self._traffic_policy_for_engine(eid)
        prefixes = [str(x) for x in list(traffic_policy.get("allowed_path_prefixes") or ["/"])]
        if prefixes and not any(req_path.startswith(px if px else "/") for px in prefixes):
            raise PermissionError(f"proxy_path_not_allowed:{req_path}")

        parsed = urllib.parse.urlsplit(ws_url)
        host = str(parsed.hostname or "").strip()
        if not host:
            raise ValueError("invalid websocket host")
        port = int(parsed.port or (443 if parsed.scheme == "wss" else 80))
        request_uri = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query or "", ""))

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

        ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
        lines = [
            f"GET {request_uri} HTTP/1.1",
            f"Host: {host}:{port}" if parsed.port else f"Host: {host}",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Key: {ws_key}",
            "Sec-WebSocket-Version: 13",
        ]
        if "Sec-WebSocket-Protocol" in req_headers:
            lines.append(f"Sec-WebSocket-Protocol: {req_headers['Sec-WebSocket-Protocol']}")
        if "Origin" in req_headers:
            lines.append(f"Origin: {req_headers['Origin']}")
        if "User-Agent" in req_headers:
            lines.append(f"User-Agent: {req_headers['User-Agent']}")
        if "Authorization" in req_headers:
            lines.append(f"Authorization: {req_headers['Authorization']}")
        raw_req = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")

        sock: Optional[socket.socket] = None
        try:
            sock = socket.create_connection((host, port), timeout=max(1.0, float(timeout_seconds or 30.0)))
            if parsed.scheme == "wss":
                ctx = ssl.create_default_context()
                sock = ctx.wrap_socket(sock, server_hostname=host)
            sock.sendall(raw_req)
            raw_resp = self._read_until(
                sock,
                b"\r\n\r\n",
                max_bytes=65536,
                timeout_seconds=max(1.0, float(timeout_seconds or 30.0)),
            )
            if b"\r\n\r\n" not in raw_resp:
                raise RuntimeError("upstream_ws_handshake_failed")
            head = raw_resp.split(b"\r\n\r\n", 1)[0].decode("latin-1", errors="replace")
            lines_resp = head.split("\r\n")
            status_line = lines_resp[0] if lines_resp else ""
            if " 101 " not in f" {status_line} ":
                raise RuntimeError(f"upstream_ws_handshake_failed:{status_line}")
            headers_resp: Dict[str, str] = {}
            for row in lines_resp[1:]:
                if ":" not in row:
                    continue
                k, v = row.split(":", 1)
                headers_resp[str(k).strip()] = str(v).strip()
            accept = str(headers_resp.get("Sec-WebSocket-Accept") or "").strip()
            expected = base64.b64encode(
                hashlib.sha1((ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("utf-8")).digest()
            ).decode("ascii")
            if not accept or accept != expected:
                raise RuntimeError("invalid_ws_handshake_accept")
            ws_id = secrets.token_urlsafe(24)
            self._ws_session_set(
                ws_id,
                {
                    "engine_id": eid,
                    "url": ws_url,
                    "created_at": time.time(),
                    "last_io_at": time.time(),
                    "socket": sock,
                    "closed": False,
                },
            )
            return {
                "status": "ok",
                "ws_id": ws_id,
                "engine_id": eid,
                "url": ws_url,
                "created_at": time.time(),
                "subprotocol": str(headers_resp.get("Sec-WebSocket-Protocol") or "") or None,
            }
        except Exception:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
            raise

    def proxy_ws_send(
        self,
        *,
        ws_id: str,
        text: Optional[str] = None,
        data_b64: str = "",
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        ws_policy = self._websocket_session_policy()
        _ = self._ws_cleanup(policy=ws_policy)
        sid = str(ws_id or "").strip()
        if not sid:
            raise ValueError("ws_id is required")
        sess = self._ws_session_get(sid)
        if not sess:
            raise ValueError("ws_session_not_found")
        sock = sess.get("socket")
        if not isinstance(sock, socket.socket):
            raise ValueError("ws_session_invalid_socket")
        if text is not None and str(data_b64 or "").strip():
            raise ValueError("provide either text or data_b64, not both")
        if text is not None:
            payload = str(text).encode("utf-8")
            opcode = 0x1
            kind = "text"
        else:
            payload = base64.b64decode(str(data_b64 or "") or "", validate=True) if str(data_b64 or "").strip() else b""
            opcode = 0x2
            kind = "binary"
        frame = self._ws_frame_encode(opcode, payload, masked=True, fin=True)
        sock.settimeout(max(0.2, float(timeout_seconds or 30.0)))
        sock.sendall(frame)
        sess["last_io_at"] = time.time()
        self._ws_session_set(sid, sess)
        return {"status": "ok", "ws_id": sid, "sent_kind": kind, "sent_bytes": len(payload)}

    def proxy_ws_recv(
        self,
        *,
        ws_id: str,
        timeout_seconds: float = 30.0,
        max_bytes: int = 1024 * 1024,
    ) -> Dict[str, Any]:
        ws_policy = self._websocket_session_policy()
        _ = self._ws_cleanup(policy=ws_policy)
        sid = str(ws_id or "").strip()
        if not sid:
            raise ValueError("ws_id is required")
        sess = self._ws_session_get(sid)
        if not sess:
            raise ValueError("ws_session_not_found")
        sock = sess.get("socket")
        if not isinstance(sock, socket.socket):
            raise ValueError("ws_session_invalid_socket")
        try:
            frame = self._ws_frame_read(
                sock,
                max_bytes=max(1, int(max_bytes or (1024 * 1024))),
                timeout_seconds=max(0.2, float(timeout_seconds or 30.0)),
            )
        except socket.timeout:
            return {"status": "timeout", "ws_id": sid}
        opcode = int(frame.get("opcode") or 0)
        payload = bytes(frame.get("payload") or b"")
        sess["last_io_at"] = time.time()
        self._ws_session_set(sid, sess)
        if opcode == 0x1:
            return {"status": "ok", "ws_id": sid, "event": "text", "text": payload.decode("utf-8", errors="replace")}
        if opcode == 0x2:
            return {"status": "ok", "ws_id": sid, "event": "binary", "data_b64": base64.b64encode(payload).decode("ascii"), "bytes": len(payload)}
        if opcode == 0x8:
            self.proxy_ws_close(ws_id=sid)
            return {"status": "ok", "ws_id": sid, "event": "close"}
        if opcode == 0x9:
            pong = self._ws_frame_encode(0xA, payload, masked=True, fin=True)
            sock.sendall(pong)
            return {"status": "ok", "ws_id": sid, "event": "ping"}
        if opcode == 0xA:
            return {"status": "ok", "ws_id": sid, "event": "pong"}
        return {"status": "ok", "ws_id": sid, "event": f"opcode_{opcode}"}

    def proxy_ws_close(
        self,
        *,
        ws_id: str,
        code: int = 1000,
        reason: str = "",
    ) -> Dict[str, Any]:
        ws_policy = self._websocket_session_policy()
        _ = self._ws_cleanup(policy=ws_policy)
        sid = str(ws_id or "").strip()
        if not sid:
            raise ValueError("ws_id is required")
        sess = self._ws_session_pop(sid)
        if not sess:
            return {"status": "not_found", "ws_id": sid}
        sock = sess.get("socket")
        if isinstance(sock, socket.socket):
            try:
                payload = struct.pack("!H", int(code or 1000))
                if str(reason or "").strip():
                    payload += str(reason).encode("utf-8", errors="replace")[:123]
                frame = self._ws_frame_encode(0x8, payload, masked=True, fin=True)
                sock.sendall(frame)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
        return {"status": "closed", "ws_id": sid}

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

    def claim_engine(
        self,
        engine_id: str,
        *,
        backend_id: Optional[str],
        exclusive: bool = False,
        force_override: bool = False,
        actor_id: Optional[str] = None,
        peer_host: Optional[str] = None,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        bid = str(actor_id or "").strip() or self._normalize_backend_id(backend_id)
        if not eid:
            raise ValueError("engine_id is required")
        control = self._read_control()
        claims = dict(control.get("claims_by_engine") or {})
        claim = dict(claims.get(eid) or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        owners_before = [str(x or "").strip() for x in list(claim.get("owners") or []) if str(x or "").strip()]
        active_owners, orphan_owners = self._active_and_orphan_owners(control, owners_before)
        owners = set(active_owners)
        previous_exclusive = str(claim.get("exclusive_owner") or "").strip()
        if previous_exclusive and previous_exclusive not in owners:
            previous_exclusive = ""
        if previous_exclusive:
            claim["exclusive_owner"] = previous_exclusive
        else:
            claim["exclusive_owner"] = None
        displaced: List[str] = []
        revoked = 0
        transition = "claimed"
        if exclusive:
            blocked_by = sorted([o for o in owners if o != bid])
            if blocked_by and not bool(force_override):
                self._append_claim_audit_event(
                    control,
                    event_type="claim_deny",
                    command="claim-engine",
                    scope="engine",
                    resource_kind="engine",
                    resource_id=eid,
                    actor_id=bid,
                    decision="deny",
                    code="exclusive_owner_conflict",
                    transition=None,
                    mode="exclusive",
                    peer_host=peer_host,
                    owners_before=owners_before,
                    owners_after=owners_before,
                    details={"blocking_owners": blocked_by},
                )
                self._write_control(control)
                out = self._deny_payload(
                    "exclusive_owner_conflict",
                    "exclusive owner conflict",
                    engine_id=eid,
                    backend_id=bid,
                    blocking_owners=blocked_by,
                )
                out.update({"engine_id": eid, "backend_id": bid, "mode": "exclusive"})
                return out
            if blocked_by and bool(force_override):
                transition = "force_override"
            elif orphan_owners:
                transition = "orphan_takeover"
            claim["owners"] = [bid]
            claim["exclusive_owner"] = bid
            claim["claimed_at"] = time.time()
            revoked = self._revoke_engine_tokens(control, eid)
            displaced = sorted([o for o in owners_before if o != bid])
        else:
            if previous_exclusive and previous_exclusive != bid:
                if not bool(force_override):
                    self._append_claim_audit_event(
                        control,
                        event_type="claim_deny",
                        command="claim-engine",
                        scope="engine",
                        resource_kind="engine",
                        resource_id=eid,
                        actor_id=bid,
                        decision="deny",
                        code="engine_exclusive_conflict",
                        transition=None,
                        mode="shared",
                        peer_host=peer_host,
                        owners_before=owners_before,
                        owners_after=owners_before,
                        details={"engine_exclusive_owner": previous_exclusive},
                    )
                    self._write_control(control)
                    out = self._deny_payload(
                        "engine_exclusive_conflict",
                        "engine exclusive conflict",
                        engine_id=eid,
                        backend_id=bid,
                        engine_exclusive_owner=previous_exclusive,
                    )
                    out.update({"engine_id": eid, "backend_id": bid, "mode": "shared"})
                    return out
                transition = "force_override"
                displaced = [previous_exclusive]
                revoked = self._revoke_engine_tokens(control, eid)
            owners.add(bid)
            claim["owners"] = sorted(list(owners))
            claim["exclusive_owner"] = None
            claim["claimed_at"] = time.time()
            if bid in owners_before:
                transition = "refreshed"
            elif orphan_owners:
                transition = "orphan_takeover"
            else:
                transition = "joined_shared"
        claims[eid] = claim
        control["claims_by_engine"] = claims
        self._touch_claim_owner_keepalive(control, bid)
        self._append_claim_audit_event(
            control,
            event_type="claim_grant",
            command="claim-engine",
            scope="engine",
            resource_kind="engine",
            resource_id=eid,
            actor_id=bid,
            decision="grant",
            code="ok",
            transition=transition,
            mode="exclusive" if exclusive else "shared",
            peer_host=peer_host,
            owners_before=owners_before,
            owners_after=list(claim.get("owners") or []),
            details={"orphan_owners": orphan_owners, "force_override": bool(force_override)},
        )
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
            "transition": transition,
        }

    def claim_endpoint(
        self,
        *,
        backend_id: Optional[str],
        exclusive: bool = False,
        force_override: bool = False,
        actor_id: Optional[str] = None,
        peer_host: Optional[str] = None,
    ) -> Dict[str, Any]:
        bid = str(actor_id or "").strip() or self._normalize_backend_id(backend_id)
        control = self._read_control()
        endpoint = dict(control.get("endpoint_claim") or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        owners_before = [str(x or "").strip() for x in list(endpoint.get("owners") or []) if str(x or "").strip()]
        active_owners, orphan_owners = self._active_and_orphan_owners(control, owners_before)
        owners = set(active_owners)
        displaced: List[str] = []
        revoked = 0
        transition = "claimed"
        if exclusive:
            blocked_by = sorted([o for o in owners if o != bid])
            if blocked_by and not bool(force_override):
                self._append_claim_audit_event(
                    control,
                    event_type="claim_deny",
                    command="claim-endpoint",
                    scope="endpoint",
                    resource_kind="endpoint",
                    resource_id="*",
                    actor_id=bid,
                    decision="deny",
                    code="exclusive_owner_conflict",
                    transition=None,
                    mode="exclusive",
                    peer_host=peer_host,
                    owners_before=owners_before,
                    owners_after=owners_before,
                    details={"blocking_owners": blocked_by},
                )
                self._write_control(control)
                out = self._deny_payload(
                    "exclusive_owner_conflict",
                    "exclusive owner conflict",
                    backend_id=bid,
                    blocking_owners=blocked_by,
                )
                out.update({"scope": "endpoint", "backend_id": bid, "mode": "exclusive"})
                return out
            if blocked_by and bool(force_override):
                transition = "force_override"
            elif orphan_owners:
                transition = "orphan_takeover"
            displaced = sorted([o for o in owners_before if o != bid])
            endpoint = {"owners": [bid], "exclusive_owner": bid, "claimed_at": time.time()}
            control["claims_by_engine"] = {}
            control["resource_claims"] = {}
            revoked = self._revoke_all_tokens(control)
        else:
            previous_exclusive = str(endpoint.get("exclusive_owner") or "")
            if previous_exclusive and previous_exclusive != bid:
                if self._is_owner_active(control, previous_exclusive) and not bool(force_override):
                    self._append_claim_audit_event(
                        control,
                        event_type="claim_deny",
                        command="claim-endpoint",
                        scope="endpoint",
                        resource_kind="endpoint",
                        resource_id="*",
                        actor_id=bid,
                        decision="deny",
                        code="endpoint_exclusive_conflict",
                        transition=None,
                        mode="shared",
                        peer_host=peer_host,
                        owners_before=owners_before,
                        owners_after=owners_before,
                        details={"endpoint_exclusive_owner": previous_exclusive},
                    )
                    self._write_control(control)
                    out = self._deny_payload(
                        "endpoint_exclusive_conflict",
                        "endpoint exclusive conflict",
                        backend_id=bid,
                        endpoint_exclusive_owner=previous_exclusive,
                    )
                    out.update({"scope": "endpoint", "backend_id": bid, "mode": "shared"})
                    return out
                transition = "force_override"
                displaced = [previous_exclusive]
                revoked = self._revoke_all_tokens(control)
            owners.add(bid)
            endpoint = {"owners": sorted(list(owners)), "exclusive_owner": None, "claimed_at": time.time()}
            if bid in owners_before:
                transition = "refreshed"
            elif orphan_owners:
                transition = "orphan_takeover"
            else:
                transition = "joined_shared"
        control["endpoint_claim"] = endpoint
        self._touch_claim_owner_keepalive(control, bid)
        self._append_claim_audit_event(
            control,
            event_type="claim_grant",
            command="claim-endpoint",
            scope="endpoint",
            resource_kind="endpoint",
            resource_id="*",
            actor_id=bid,
            decision="grant",
            code="ok",
            transition=transition,
            mode="exclusive" if exclusive else "shared",
            peer_host=peer_host,
            owners_before=owners_before,
            owners_after=list(endpoint.get("owners") or []),
            details={"orphan_owners": orphan_owners, "force_override": bool(force_override)},
        )
        self._write_control(control)
        return {
            "scope": "endpoint",
            "backend_id": bid,
            "mode": "exclusive" if exclusive else "shared",
            "owners": list(endpoint.get("owners") or []),
            "exclusive_owner": endpoint.get("exclusive_owner"),
            "displaced_backends": displaced,
            "revoked_tokens": revoked,
            "transition": transition,
        }

    def get_claim_status(self, engine_id: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        control = self._read_control()
        claim = dict((control.get("claims_by_engine") or {}).get(eid) or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        endpoint = dict(control.get("endpoint_claim") or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        active_owners, orphan_owners = self._active_and_orphan_owners(control, list(claim.get("owners") or []))
        token_count = 0
        for meta in dict(control.get("tokens") or {}).values():
            if str((meta or {}).get("engine_id") or "") == eid:
                token_count += 1
        return {
            "engine_id": eid,
            "engine_claim": claim,
            "active_owners": active_owners,
            "orphan_owners": orphan_owners,
            "endpoint_claim": endpoint,
            "issued_tokens": token_count,
        }

    def issue_token(self, engine_id: str, *, backend_id: Optional[str]) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        bid = self._normalize_backend_id(backend_id)
        control = self._read_control()
        endpoint_exclusive = str((control.get("endpoint_claim") or {}).get("exclusive_owner") or "")
        if endpoint_exclusive and (not self._is_owner_active(control, endpoint_exclusive)):
            endpoint_exclusive = ""
        if endpoint_exclusive and endpoint_exclusive != bid:
            out = self._deny_payload(
                "endpoint_exclusive_conflict",
                "endpoint exclusive conflict",
                endpoint_exclusive_owner=endpoint_exclusive,
            )
            out.update({"engine_id": eid, "backend_id": bid, "token": None, "endpoint_exclusive_owner": endpoint_exclusive})
            return out
        claim = dict((control.get("claims_by_engine") or {}).get(eid) or {})
        exclusive_owner = str(claim.get("exclusive_owner") or "")
        if exclusive_owner and (not self._is_owner_active(control, exclusive_owner)):
            exclusive_owner = ""
        if exclusive_owner and exclusive_owner != bid:
            out = self._deny_payload(
                "engine_exclusive_conflict",
                "engine exclusive conflict",
                engine_exclusive_owner=exclusive_owner,
            )
            out.update({"engine_id": eid, "backend_id": bid, "token": None, "engine_exclusive_owner": exclusive_owner})
            return out
        active_owners, _ = self._active_and_orphan_owners(control, list(claim.get("owners") or []))
        owners = set(active_owners)
        if owners and bid not in owners:
            out = self._deny_payload(
                "engine_shared_claim_not_member",
                "engine shared claim not member",
                engine_owners=sorted(list(owners)),
            )
            out.update({"engine_id": eid, "backend_id": bid, "token": None, "engine_owners": sorted(list(owners))})
            return out
        token = secrets.token_urlsafe(24)
        tokens = dict(control.get("tokens") or {})
        tokens[token] = {"engine_id": eid, "backend_id": bid, "issued_at": time.time()}
        control["tokens"] = tokens
        self._touch_claim_owner_keepalive(control, bid)
        self._write_control(control)
        return {"status": "ok", "engine_id": eid, "backend_id": bid, "token": token, "issued_at": tokens[token]["issued_at"]}

    def validate_token(self, engine_id: str, token: str) -> bool:
        control = self._read_control()
        meta = dict(control.get("tokens") or {}).get(str(token or "").strip())
        return bool(meta and str(meta.get("engine_id") or "") == str(engine_id or ""))

    def claim_resource(
        self,
        resource_kind: str,
        resource_id: str,
        *,
        backend_id: Optional[str],
        exclusive: bool = False,
        force_override: bool = False,
        actor_id: Optional[str] = None,
        peer_host: Optional[str] = None,
    ) -> Dict[str, Any]:
        rkind = str(resource_kind or "").strip().lower()
        rid = str(resource_id or "").strip()
        if rkind == "engine":
            return self.claim_engine(
                rid,
                backend_id=backend_id,
                exclusive=exclusive,
                force_override=force_override,
                actor_id=actor_id,
                peer_host=peer_host,
            )
        bid = str(actor_id or "").strip() or self._normalize_backend_id(backend_id)
        rkey = self._resource_key(rkind, rid)
        control = self._read_control()
        claims = dict(control.get("resource_claims") or {})
        claim = dict(claims.get(rkey) or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        owners_before = [str(x or "").strip() for x in list(claim.get("owners") or []) if str(x or "").strip()]
        active_owners, orphan_owners = self._active_and_orphan_owners(control, owners_before)
        owners = set(active_owners)
        displaced: List[str] = []
        revoked = 0
        transition = "claimed"
        if exclusive:
            blocked_by = sorted([o for o in owners if o != bid])
            if blocked_by and not bool(force_override):
                self._append_claim_audit_event(
                    control,
                    event_type="claim_deny",
                    command="claim-resource",
                    scope="resource",
                    resource_kind=rkind,
                    resource_id=rid,
                    actor_id=bid,
                    decision="deny",
                    code="exclusive_owner_conflict",
                    transition=None,
                    mode="exclusive",
                    peer_host=peer_host,
                    owners_before=owners_before,
                    owners_after=owners_before,
                    details={"blocking_owners": blocked_by},
                )
                self._write_control(control)
                out = self._deny_payload(
                    "exclusive_owner_conflict",
                    "exclusive owner conflict",
                    resource_kind=rkind,
                    resource_id=rid,
                    backend_id=bid,
                    blocking_owners=blocked_by,
                )
                out.update({"scope": "resource", "resource_kind": rkind, "resource_id": rid, "backend_id": bid, "mode": "exclusive"})
                return out
            if blocked_by and bool(force_override):
                transition = "force_override"
            elif orphan_owners:
                transition = "orphan_takeover"
            displaced = sorted([o for o in owners_before if o != bid])
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
                if self._is_owner_active(control, previous_exclusive) and not bool(force_override):
                    self._append_claim_audit_event(
                        control,
                        event_type="claim_deny",
                        command="claim-resource",
                        scope="resource",
                        resource_kind=rkind,
                        resource_id=rid,
                        actor_id=bid,
                        decision="deny",
                        code="resource_exclusive_conflict",
                        transition=None,
                        mode="shared",
                        peer_host=peer_host,
                        owners_before=owners_before,
                        owners_after=owners_before,
                        details={"resource_exclusive_owner": previous_exclusive},
                    )
                    self._write_control(control)
                    out = self._deny_payload(
                        "resource_exclusive_conflict",
                        "resource exclusive conflict",
                        resource_kind=rkind,
                        resource_id=rid,
                        backend_id=bid,
                        resource_exclusive_owner=previous_exclusive,
                    )
                    out.update({"scope": "resource", "resource_kind": rkind, "resource_id": rid, "backend_id": bid, "mode": "shared"})
                    return out
                transition = "force_override"
                displaced = [previous_exclusive]
            owners.add(bid)
            claim["owners"] = sorted(list(owners))
            claim["exclusive_owner"] = None
            claim["claimed_at"] = time.time()
            if bid in owners_before:
                transition = "refreshed"
            elif orphan_owners:
                transition = "orphan_takeover"
            else:
                transition = "joined_shared"
        claims[rkey] = claim
        control["resource_claims"] = claims
        self._touch_claim_owner_keepalive(control, bid)
        self._append_claim_audit_event(
            control,
            event_type="claim_grant",
            command="claim-resource",
            scope="resource",
            resource_kind=rkind,
            resource_id=rid,
            actor_id=bid,
            decision="grant",
            code="ok",
            transition=transition,
            mode="exclusive" if exclusive else "shared",
            peer_host=peer_host,
            owners_before=owners_before,
            owners_after=list(claim.get("owners") or []),
            details={"orphan_owners": orphan_owners, "force_override": bool(force_override)},
        )
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
            "transition": transition,
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
        active_owners, orphan_owners = self._active_and_orphan_owners(control, list(claim.get("owners") or []))
        issued_tokens = 0
        for meta in dict(control.get("resource_tokens") or {}).values():
            if str((meta or {}).get("resource_key") or "") == rkey:
                issued_tokens += 1
        return {
            "resource_kind": rkind,
            "resource_id": rid,
            "resource_claim": claim,
            "active_owners": active_owners,
            "orphan_owners": orphan_owners,
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
        if endpoint_exclusive and (not self._is_owner_active(control, endpoint_exclusive)):
            endpoint_exclusive = ""
        if endpoint_exclusive and endpoint_exclusive != bid:
            out = self._deny_payload(
                "endpoint_exclusive_conflict",
                "endpoint exclusive conflict",
                endpoint_exclusive_owner=endpoint_exclusive,
            )
            out.update({"resource_kind": rkind, "resource_id": rid, "backend_id": bid, "token": None, "endpoint_exclusive_owner": endpoint_exclusive})
            return out
        claim = dict((control.get("resource_claims") or {}).get(rkey) or {})
        exclusive_owner = str(claim.get("exclusive_owner") or "")
        if exclusive_owner and (not self._is_owner_active(control, exclusive_owner)):
            exclusive_owner = ""
        if exclusive_owner and exclusive_owner != bid:
            out = self._deny_payload(
                "resource_exclusive_conflict",
                "resource exclusive conflict",
                resource_exclusive_owner=exclusive_owner,
            )
            out.update({"resource_kind": rkind, "resource_id": rid, "backend_id": bid, "token": None, "resource_exclusive_owner": exclusive_owner})
            return out
        active_owners, _ = self._active_and_orphan_owners(control, list(claim.get("owners") or []))
        owners = set(active_owners)
        if owners and bid not in owners:
            out = self._deny_payload(
                "resource_shared_claim_not_member",
                "resource shared claim not member",
                resource_owners=sorted(list(owners)),
            )
            out.update({"resource_kind": rkind, "resource_id": rid, "backend_id": bid, "token": None, "resource_owners": sorted(list(owners))})
            return out
        token = secrets.token_urlsafe(24)
        res_tokens = dict(control.get("resource_tokens") or {})
        res_tokens[token] = {"resource_kind": rkind, "resource_id": rid, "resource_key": rkey, "backend_id": bid, "issued_at": time.time()}
        control["resource_tokens"] = res_tokens
        self._touch_claim_owner_keepalive(control, bid)
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
