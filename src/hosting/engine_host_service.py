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
import posixpath
import subprocess
import sys
import time
import hmac
import base64
import tempfile
import shutil
from contextlib import contextmanager
from multiprocessing.connection import Client as MPClient
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mp13_engine.mp13_toolbox import ToolsView

from .sandbox import (
    BrokeredFilesystem,
    HostBrokeredHttpClient,
    WorkerLaunchRequest,
    WorkerSandboxPolicy,
    launch_worker_process,
)


def _default_toolboxes_state_file() -> Path:
    return (DEFAULT_STATE_DIR / "toolbox_sandboxes.json").expanduser().resolve()

def _default_state_dir() -> Path:
    # Keep hosting bootstrap lightweight: avoid importing mp13_engine package
    # during module import to prevent unrelated heavy dependency side-effects.
    return (Path.home() / ".mp13-llm" / "hosting" / "state").expanduser().resolve()


def _default_hosting_root() -> Path:
    return (Path.home() / ".mp13-llm" / "hosting").expanduser().resolve()


DEFAULT_STATE_DIR = _default_state_dir()
DEFAULT_HOSTING_ROOT = _default_hosting_root()
DEFAULT_ENGINES_STATE_FILE = DEFAULT_STATE_DIR / "managed_engines.json"
DEFAULT_CONTROL_STATE_FILE = DEFAULT_HOSTING_ROOT / "access_control.json"
DEFAULT_TOOLBOXES_STATE_FILE = _default_toolboxes_state_file()
DAEMON_VERSION = "2.1.0"

ROLE_ADMIN = "admin"
ROLE_CONFIG_EDITOR = "config_editor"
ROLE_WORKER_USER = "worker_user"
ROLE_MODEL_USER_WITH_MODEL_CONTROL = "model_user_with_model_control"
ROLE_MODEL_USER = "model_user"
ROLE_DIAGNOSTIC_USER = "diagnostic_user"
ROLE_TRANSPORT = "transport"

LIFECYCLE_PROFILE_FOREGROUND = "foreground_terminal_bound"
LIFECYCLE_PROFILE_DETACHED = "detached_user_process"
LIFECYCLE_PROFILE_SERVICE = "service_managed"
VALID_LIFECYCLE_PROFILES = {
    LIFECYCLE_PROFILE_FOREGROUND,
    LIFECYCLE_PROFILE_DETACHED,
    LIFECYCLE_PROFILE_SERVICE,
}

VALID_AUTH_ROLES = {
    ROLE_ADMIN,
    ROLE_CONFIG_EDITOR,
    ROLE_WORKER_USER,
    ROLE_MODEL_USER_WITH_MODEL_CONTROL,
    ROLE_MODEL_USER,
    ROLE_DIAGNOSTIC_USER,
    ROLE_TRANSPORT,
}

VALID_FORCE_OVERRIDE_REASONS = {
    "stale_owner_unreachable",
    "owner_malicious",
    "security_incident",
    "policy_recovery",
}
EMERGENCY_FORCE_OVERRIDE_REASONS = {
    "stale_owner_unreachable",
    "owner_malicious",
    "security_incident",
}


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


class EngineHostService:
    """File-backed engine host service for terminal-command control."""
    _metrics_lock = threading.Lock()
    _runtime_metrics: Optional[Dict[str, Any]] = None
    _toolbox_lock_guard = threading.Lock()
    _toolbox_locks: Dict[str, threading.RLock] = {}

    def __init__(
        self,
        *,
        engines_state_file: Optional[Path] = None,
        control_state_file: Optional[Path] = None,
    ):
        self.engines_state_file = (engines_state_file or DEFAULT_ENGINES_STATE_FILE).expanduser().resolve()
        raw_control = (control_state_file or DEFAULT_CONTROL_STATE_FILE).expanduser().resolve()
        if raw_control.suffix:
            self.hosting_root = raw_control.parent.resolve()
            self.control_state_file = self.hosting_root / "access_control.json"
        else:
            self.hosting_root = raw_control.resolve()
            self.control_state_file = self.hosting_root / "access_control.json"
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
        snapshot["hosting_root"] = str(self.hosting_root)
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
    def daemon_capabilities() -> Dict[str, bool]:
        return {
            "claim_acl_v2": True,
            "structured_denials_v1": True,
            "force_override_confirmation_v1": True,
            "ipc_rpc_v1": True,
            "structured_progress_events_v1": True,
            "reachability_status_v1": True,
            "non_blocking_ops_v1": True,
            "lifecycle_profiles_v1": True,
        }

    @staticmethod
    def _progress_event(stage: str, status: str, message: str, **extra: Any) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "stage": str(stage or "unknown"),
            "status": str(status or "info"),
            "message": str(message or ""),
            "timestamp": time.time(),
        }
        if extra:
            event.update({str(k): v for k, v in extra.items()})
        return event

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

    def _toolboxes_state_file(self) -> Path:
        return (self.hosting_root / "state" / "toolbox_sandboxes.json").resolve()

    def _read_toolboxes(self) -> Dict[str, Any]:
        return self._read_json(
            self._toolboxes_state_file(),
            {
                "version": 1,
                "toolboxes": {},
                "environment_descriptions": {
                    "base": {
                        "name": "base",
                        "base_env_name": None,
                        "extra_packages": [],
                        "allow_online_install": False,
                    }
                },
            },
        )

    def _write_toolboxes(self, payload: Dict[str, Any]) -> None:
        row = dict(payload or {})
        row["version"] = 1
        row.setdefault(
            "environment_descriptions",
            {
                "base": {
                    "name": "base",
                    "base_env_name": None,
                    "extra_packages": [],
                    "allow_online_install": False,
                }
            },
        )
        row["updated_at"] = time.time()
        self._write_json(self._toolboxes_state_file(), row)

    @classmethod
    def _toolbox_lock_for(cls, toolbox_id: str) -> threading.RLock:
        tid = str(toolbox_id or "").strip()
        if not tid:
            return threading.RLock()
        with cls._toolbox_lock_guard:
            lock = cls._toolbox_locks.get(tid)
            if lock is None:
                lock = threading.RLock()
                cls._toolbox_locks[tid] = lock
            return lock

    @contextmanager
    def _locked_toolbox(self, toolbox_id: str):
        lock = self._toolbox_lock_for(toolbox_id)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def _run_locked_toolbox_call(self, toolbox_id: str, callback: Any, /, *args: Any, **kwargs: Any) -> Any:
        with self._locked_toolbox(toolbox_id):
            return callback(*args, **kwargs)

    @contextmanager
    def _locked_toolboxes(self, toolbox_ids: List[str]):
        normalized_ids = sorted(
            {
                str(item or "").strip()
                for item in list(toolbox_ids or [])
                if str(item or "").strip()
            }
        )
        locks = [self._toolbox_lock_for(toolbox_id) for toolbox_id in normalized_ids]
        for lock in locks:
            lock.acquire()
        try:
            yield normalized_ids
        finally:
            for lock in reversed(locks):
                lock.release()

    def toolbox_environment_description_list(self) -> Dict[str, Any]:
        from .toolbox_harness import ToolboxEnvironmentManager

        state = self._read_toolboxes()
        envs = dict(state.get("environment_descriptions") or {})
        out: Dict[str, Any] = {}
        for name, row in envs.items():
            normalized = ToolboxEnvironmentManager.normalize_environment_description(dict(row or {}), name=str(name or ""))
            out[str(normalized["name"])] = normalized
        if "base" not in out:
            out["base"] = ToolboxEnvironmentManager.normalize_environment_description({}, name="base")
        return {"status": "ok", "environment_descriptions": out}

    def toolbox_environment_description_get(self, name: str) -> Dict[str, Any]:
        from .toolbox_harness import ToolboxEnvironmentManager

        env_name = str(name or "").strip() or "base"
        state = self._read_toolboxes()
        envs = dict(state.get("environment_descriptions") or {})
        payload = dict(envs.get(env_name) or ({}) if env_name == "base" else envs.get(env_name) or {})
        if not payload and env_name != "base":
            raise ValueError(f"environment description '{env_name}' not found")
        return ToolboxEnvironmentManager.normalize_environment_description(payload, name=env_name)

    def toolbox_environment_description_effective_get(self, name: str) -> Dict[str, Any]:
        from .toolbox_harness import ToolboxEnvironmentManager

        env_name = str(name or "").strip() or "base"
        state = self._read_toolboxes()
        envs = dict(state.get("environment_descriptions") or {})
        if env_name != "base" and env_name not in envs:
            raise ValueError(f"environment description '{env_name}' not found")
        return ToolboxEnvironmentManager.resolve_environment_description(envs, name=env_name)

    def toolbox_environment_description_upsert(
        self,
        *,
        name: str,
        base_env_name: Optional[str] = None,
        extra_packages: Optional[List[str]] = None,
        allow_online_install: bool = False,
    ) -> Dict[str, Any]:
        from .toolbox_harness import ToolboxEnvironmentManager

        env_name = str(name or "").strip()
        if not env_name:
            raise ValueError("name is required")
        if env_name == "base" and base_env_name:
            raise ValueError("base environment cannot inherit from another environment")
        state = self._read_toolboxes()
        envs = dict(state.get("environment_descriptions") or {})
        if base_env_name and str(base_env_name).strip() not in envs and str(base_env_name).strip() != "base":
            raise ValueError(f"base environment '{base_env_name}' not found")
        normalized = ToolboxEnvironmentManager.normalize_environment_description(
            {
                "base_env_name": str(base_env_name or "").strip() or None,
                "extra_packages": list(extra_packages or []),
                "allow_online_install": bool(allow_online_install),
            },
            name=env_name,
        )
        envs[env_name] = normalized
        state["environment_descriptions"] = envs
        self._write_toolboxes(state)
        return {"status": "ok", "environment_description": normalized}

    def toolbox_environment_description_clone(
        self,
        *,
        source_name: str,
        target_name: str,
        extra_packages: Optional[List[str]] = None,
        allow_online_install: Optional[bool] = None,
    ) -> Dict[str, Any]:
        from .toolbox_harness import ToolboxEnvironmentManager

        src = str(source_name or "").strip()
        dst = str(target_name or "").strip()
        if not src:
            raise ValueError("source_name is required")
        if not dst:
            raise ValueError("target_name is required")
        if src == dst:
            raise ValueError("target_name must differ from source_name")
        source = self.toolbox_environment_description_get(src)
        state = self._read_toolboxes()
        envs = dict(state.get("environment_descriptions") or {})
        if dst in envs:
            raise ValueError(f"environment description '{dst}' already exists")
        normalized = ToolboxEnvironmentManager.normalize_environment_description(
            {
                "base_env_name": src,
                "extra_packages": list(extra_packages if extra_packages is not None else list(source.get("extra_packages") or [])),
                "allow_online_install": bool(source.get("allow_online_install", False))
                if allow_online_install is None
                else bool(allow_online_install),
            },
            name=dst,
        )
        envs[dst] = normalized
        state["environment_descriptions"] = envs
        self._write_toolboxes(state)
        return {"status": "ok", "environment_description": normalized}

    def toolbox_environment_resolve_requirements(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env = self.toolbox_environment_description_get(environment_name)
        effective_env = self.toolbox_environment_description_effective_get(environment_name)
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        selected = {str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()}
        required_packages: List[str] = []
        seen: set[str] = set()
        for req in list(toolbox_row.get("requests") or []):
            row = dict(req or {})
            key = f"{str(row.get('module_name') or '').strip()}:{str(row.get('callable_name') or '').strip()}"
            profile = dict(row.get("sandbox_profile") or {})
            if str(profile.get("environment_name") or "base").strip() != str(environment_name or "base").strip():
                continue
            if selected and key not in selected:
                continue
            for pkg in list(profile.get("required_imports") or []):
                name = str(pkg or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    required_packages.append(name)
        for req in list(toolbox_row.get("manual_requests") or []):
            row = dict(req or {})
            key = f"manual:{str(row.get('module_name') or '').strip()}:{str(row.get('callable_name') or '').strip()}"
            profile = dict(row.get("sandbox_profile") or {})
            if str(profile.get("environment_name") or "base").strip() != str(environment_name or "base").strip():
                continue
            if selected and key not in selected:
                continue
            for pkg in list(profile.get("required_imports") or []):
                name = str(pkg or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    required_packages.append(name)
        extra_packages = [
            str(item or "").strip()
            for item in list(effective_env.get("effective_extra_packages") or [])
            if str(item or "").strip()
        ]
        missing_packages = [pkg for pkg in required_packages if pkg not in set(extra_packages)]
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": str(environment_name or "base").strip() or "base",
            "environment_lineage": list(effective_env.get("lineage") or []),
            "required_packages": required_packages,
            "configured_extra_packages": [str(item or "").strip() for item in list(env.get("extra_packages") or []) if str(item or "").strip()],
            "effective_extra_packages": extra_packages,
            "missing_packages": missing_packages,
        }

    @staticmethod
    def _toolbox_profile_required_packages(
        profile_row: Optional[Dict[str, Any]],
        *,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        selected = {str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()}
        required_packages: List[str] = []
        seen: set[str] = set()
        matched_tool_keys: List[str] = []
        for req in list(dict(profile_row or {}).get("requests") or []):
            row = dict(req or {})
            key = f"{str(row.get('module_name') or '').strip()}:{str(row.get('callable_name') or '').strip()}"
            if selected and key not in selected:
                continue
            if key:
                matched_tool_keys.append(key)
            profile = dict(row.get("sandbox_profile") or {})
            for pkg in list(profile.get("required_imports") or []):
                name = str(pkg or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    required_packages.append(name)
        for req in list(dict(profile_row or {}).get("manual_requests") or []):
            row = dict(req or {})
            key = f"manual:{str(row.get('module_name') or '').strip()}:{str(row.get('callable_name') or '').strip()}"
            if selected and key not in selected:
                continue
            if key:
                matched_tool_keys.append(key)
            profile = dict(row.get("sandbox_profile") or {})
            for pkg in list(profile.get("required_imports") or []):
                name = str(pkg or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    required_packages.append(name)
        return {
            "required_packages": required_packages,
            "tool_keys": matched_tool_keys,
        }

    @staticmethod
    def _toolbox_runtime_defaults(
        toolbox_row: Optional[Dict[str, Any]],
        *,
        python_executable: Optional[str] = None,
        worker_profile_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        existing = dict(dict(toolbox_row or {}).get("runtime") or {})
        python_value = str(
            python_executable
            if python_executable is not None
            else existing.get("python_executable") or ""
        ).strip() or None
        worker_value = str(
            worker_profile_class
            if worker_profile_class is not None
            else existing.get("worker_profile_class") or "generic"
        ).strip() or "generic"
        return {
            **existing,
            "python_executable": python_value,
            "worker_profile_class": worker_value,
        }

    @staticmethod
    def _append_toolbox_cancel_event(
        toolbox_row: Optional[Dict[str, Any]],
        *,
        engine_ids: Sequence[str],
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        respawn: bool = True,
        non_restartable: bool = False,
    ) -> Dict[str, Any]:
        row = dict(toolbox_row or {})
        runtime = dict(row.get("runtime") or {})
        events = list(runtime.get("cancel_events") or [])
        events.append(
            {
                "timestamp": time.time(),
                "engine_ids": [
                    str(item or "").strip()
                    for item in list(engine_ids or [])
                    if str(item or "").strip()
                ],
                "tool_name": str(tool_name or "").strip() or None,
                "tool_call_id": str(tool_call_id or "").strip() or None,
                "respawn": bool(respawn),
                "non_restartable": bool(non_restartable),
            }
        )
        if len(events) > 25:
            events = events[-25:]
        runtime["cancel_events"] = events
        row["runtime"] = runtime
        return row

    def _toolbox_uses_environment_name(self, toolbox_row: Optional[Dict[str, Any]], environment_name: str) -> bool:
        env_name = str(environment_name or "base").strip() or "base"
        row = dict(toolbox_row or {})
        for req in list(row.get("requests") or []):
            profile = dict(dict(req or {}).get("sandbox_profile") or {})
            if str(profile.get("environment_name") or "base").strip() == env_name:
                return True
        for req in list(row.get("manual_requests") or []):
            profile = dict(dict(req or {}).get("sandbox_profile") or {})
            if str(profile.get("environment_name") or "base").strip() == env_name:
                return True
        intrinsics = dict(row.get("intrinsics") or {})
        profile = dict(intrinsics.get("sandbox_profile") or {})
        names = [
            str(item or "").strip()
            for item in list(intrinsics.get("names") or [])
            if str(item or "").strip()
        ]
        if names and str(profile.get("environment_name") or "base").strip() == env_name:
            return True
        return False

    def _toolbox_uses_environment_dependency(self, toolbox_row: Optional[Dict[str, Any]], environment_name: str) -> bool:
        env_name = str(environment_name or "base").strip() or "base"
        row = dict(toolbox_row or {})
        names_to_check: List[str] = []
        for req in list(row.get("requests") or []):
            profile = dict(dict(req or {}).get("sandbox_profile") or {})
            name = str(profile.get("environment_name") or "base").strip() or "base"
            if name:
                names_to_check.append(name)
        for req in list(row.get("manual_requests") or []):
            profile = dict(dict(req or {}).get("sandbox_profile") or {})
            name = str(profile.get("environment_name") or "base").strip() or "base"
            if name:
                names_to_check.append(name)
        intrinsics = dict(row.get("intrinsics") or {})
        profile = dict(intrinsics.get("sandbox_profile") or {})
        intrinsic_names = [str(item or "").strip() for item in list(intrinsics.get("names") or []) if str(item or "").strip()]
        if intrinsic_names:
            names_to_check.append(str(profile.get("environment_name") or "base").strip() or "base")
        for candidate in names_to_check:
            try:
                effective = self.toolbox_environment_description_effective_get(candidate)
            except Exception:
                continue
            if env_name in {str(item or "").strip() for item in list(effective.get("lineage") or []) if str(item or "").strip()}:
                return True
        return False

    @staticmethod
    def _toolbox_tool_non_restartable(toolbox_row: Optional[Dict[str, Any]], tool_name: str) -> bool:
        target = str(tool_name or "").strip()
        if not target:
            return False
        row = dict(toolbox_row or {})
        for req in list(row.get("requests") or []):
            request = dict(req or {})
            if str(request.get("callable_name") or "").strip() == target:
                return bool(request.get("non_restartable", False))
        for req in list(row.get("manual_requests") or []):
            request = dict(req or {})
            fn = dict(dict(request.get("tool_definition") or {}).get("function") or {})
            if str(fn.get("name") or "").strip() == target:
                return bool(request.get("non_restartable", False))
        return False

    def _rebuild_toolbox_from_persisted_state(
        self,
        *,
        toolbox_id: str,
        toolbox_row: Dict[str, Any],
        state: Dict[str, Any],
        action: str,
    ) -> Dict[str, Any]:
        from .toolbox_harness import (
            SandboxProfileSpec,
            ToolboxAutoAssignmentRequest,
            ToolboxBundleStager,
            ToolboxManualAssignmentRequest,
            ToolboxSandboxOrchestrator,
        )

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        row = dict(toolbox_row or {})
        auto_requests = [
            ToolboxAutoAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(row.get("requests") or [])
        ]
        manual_requests = [
            ToolboxManualAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(row.get("manual_requests") or [])
        ]
        intrinsics_row = dict(row.get("intrinsics") or {})
        intrinsic_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsics_row.get("names") or []) if str(item or "").strip()],
            include_guides=bool(intrinsics_row.get("with_intrinsic_guides", False)),
        )
        intrinsic_profile = (
            SandboxProfileSpec.from_dict(dict(intrinsics_row.get("sandbox_profile") or {}))
            if intrinsic_names
            else None
        )
        with_intrinsic_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
        existing_profiles = dict(row.get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(row)
        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        spawned_engine_ids: List[str] = []
        replaced_engine_ids: List[str] = []
        new_profiles: Dict[str, Dict[str, Any]] = {}
        ready_rollout: Dict[str, Dict[str, Any]] = {}

        if auto_requests or manual_requests or intrinsic_names:
            orchestrator = ToolboxSandboxOrchestrator(
                service=self,
                stager=ToolboxBundleStager(self.hosting_root),
                python_executable=runtime.get("python_executable"),
            )
            assignments = orchestrator.spawn_assignments(
                toolbox_id=tid,
                requests=auto_requests,
                manual_requests=manual_requests,
                intrinsic_tool_names=intrinsic_names,
                intrinsic_profile=intrinsic_profile,
                with_intrinsic_guides=with_intrinsic_guides,
                worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
            )
            try:
                ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
            except Exception:
                for item in assignments:
                    reg = dict(item.registration or {})
                    engine_id = str(reg.get("engine_id") or "").strip()
                    if engine_id:
                        self._retire_toolbox_registration(engine_id)
                self._cleanup_unused_toolbox_environments(state)
                raise
            for item in assignments:
                profile_id = item.sandbox_profile.normalized_profile_id()
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    spawned_engine_ids.append(engine_id)
                old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
                bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
                if old_engine_id and old_engine_id != engine_id:
                    replaced_engine_ids.append(old_engine_id)
                profile_auto_requests = [
                    req.to_runtime_dict()
                    for req in auto_requests
                    if req.sandbox_profile.normalized_profile_id() == profile_id
                ]
                profile_manual_requests = [
                    req.to_runtime_dict()
                    for req in manual_requests
                    if req.sandbox_profile.normalized_profile_id() == profile_id
                ]
                profile_row = {
                    "sandbox_profile": item.sandbox_profile.to_dict(),
                    "requests": profile_auto_requests,
                    "manual_requests": profile_manual_requests,
                    "engine_id": engine_id,
                    "bundle_revision": bundle_revision,
                    "environment": dict(reg.get("environment") or {}),
                    "rollout": dict(ready_rollout.get(engine_id) or {}),
                    "rollout_history": self._append_toolbox_rollout_history(
                        dict(existing_profiles.get(profile_id) or {}),
                        rollout=dict(ready_rollout.get(engine_id) or {}),
                        action=action,
                        engine_id=engine_id,
                        bundle_revision=bundle_revision,
                        replaced_engine_id=old_engine_id,
                    ),
                }
                new_profiles[profile_id] = profile_row
            for profile_id, old_engine_id in old_regs_by_profile.items():
                if profile_id not in new_profiles and old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
        else:
            for old_engine_id in old_regs_by_profile.values():
                if old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)

        updated_row: Dict[str, Any] = {
            "toolbox_id": tid,
            "requests": [req.to_runtime_dict() for req in auto_requests],
            "manual_requests": [req.to_runtime_dict() for req in manual_requests],
            "profiles": new_profiles,
            "runtime": runtime,
        }
        if intrinsic_names:
            updated_row["intrinsics"] = {
                "names": intrinsic_names,
                "sandbox_profile": (intrinsic_profile or SandboxProfileSpec(profile_id="default")).to_dict(),
                "with_intrinsic_guides": with_intrinsic_guides,
            }
        return {
            "toolbox_row": updated_row,
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "toolbox_removed": not auto_requests and not manual_requests and not intrinsic_names,
        }

    def toolbox_environment_apply(
        self,
        *,
        environment_name: str,
        toolbox_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        env_name = str(environment_name or "base").strip() or "base"
        _ = self.toolbox_environment_description_get(env_name)
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        selected = {
            str(item or "").strip()
            for item in list(toolbox_ids or [])
            if str(item or "").strip()
        }
        affected_ids: List[str] = []
        for tid, row in toolboxes.items():
            toolbox_id = str(tid or "").strip()
            if selected and toolbox_id not in selected:
                continue
            if self._toolbox_uses_environment_dependency(dict(row or {}), env_name):
                affected_ids.append(toolbox_id)
        affected_ids.sort()
        rebuilt: Dict[str, Any] = {}
        for tid in affected_ids:
            result = self._rebuild_toolbox_from_persisted_state(
                toolbox_id=tid,
                toolbox_row=dict(toolboxes.get(tid) or {}),
                state=state,
                action="apply_environment",
            )
            if bool(result.get("toolbox_removed")):
                toolboxes.pop(tid, None)
            else:
                toolboxes[tid] = dict(result.get("toolbox_row") or {})
            rebuilt[tid] = {
                "profiles": dict(result.get("profiles") or {}),
                "spawned_engine_ids": list(result.get("spawned_engine_ids") or []),
                "ready_engine_ids": list(result.get("ready_engine_ids") or []),
                "rollout": dict(result.get("rollout") or {}),
                "replaced_engine_ids": list(result.get("replaced_engine_ids") or []),
            }
            state["toolboxes"] = toolboxes
            self._write_toolboxes(state)
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "environment_name": env_name,
            "affected_toolbox_ids": affected_ids,
            "toolboxes": rebuilt,
            "removed_environment_keys": removed_environment_keys,
        }

    def toolbox_environment_realize(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from .toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        effective_env = self.toolbox_environment_description_effective_get(env_name)
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        realized_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            required_packages = list(packages.get("required_packages") or [])
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            effective_extra = [
                str(item or "").strip()
                for item in list(effective_env.get("effective_extra_packages") or [])
                if str(item or "").strip()
            ]
            missing_packages = [pkg for pkg in required_packages if pkg not in set(effective_extra)]
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.realize_environment(
                spec,
                environment_description=effective_env,
                required_packages=required_packages,
                missing_packages=missing_packages,
                toolbox_id=tid,
                sandbox_profile_id=str(profile_id or "").strip(),
                tool_keys=matched_tool_keys,
            )
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["realization"] = dict(metadata.get("realization") or {})
            profiles[str(profile_id)] = profile_row
            realized_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": realized_profiles,
        }

    def toolbox_environment_sync_description(
        self,
        *,
        toolbox_id: str,
        source_environment_name: str,
        target_environment_name: Optional[str] = None,
        tool_keys: Optional[List[str]] = None,
        apply: bool = False,
        realize: bool = False,
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        source_name = str(source_environment_name or "base").strip() or "base"
        target_name = str(target_environment_name or "").strip() or source_name
        resolved = self.toolbox_environment_resolve_requirements(
            toolbox_id=tid,
            environment_name=source_name,
            tool_keys=tool_keys,
        )
        missing_packages = [
            str(item or "").strip()
            for item in list(resolved.get("missing_packages") or [])
            if str(item or "").strip()
        ]
        source_desc = self.toolbox_environment_description_get(source_name)
        direct_packages = [
            str(item or "").strip()
            for item in list(source_desc.get("extra_packages") or [])
            if str(item or "").strip()
        ]
        merged_packages: List[str] = []
        seen: set[str] = set()
        for item in direct_packages + missing_packages:
            if item and item not in seen:
                seen.add(item)
                merged_packages.append(item)

        if target_name == source_name:
            env_result = self.toolbox_environment_description_upsert(
                name=source_name,
                base_env_name=str(source_desc.get("base_env_name") or "").strip() or None,
                extra_packages=merged_packages,
                allow_online_install=bool(source_desc.get("allow_online_install", False)),
            )
        else:
            env_result = self.toolbox_environment_description_clone(
                source_name=source_name,
                target_name=target_name,
                extra_packages=merged_packages,
                allow_online_install=bool(source_desc.get("allow_online_install", False)),
            )

        apply_result: Optional[Dict[str, Any]] = None
        realize_result: Optional[Dict[str, Any]] = None
        if apply:
            apply_result = self.toolbox_environment_apply(
                environment_name=target_name,
                toolbox_ids=[tid],
            )
        if realize:
            realize_result = self.toolbox_environment_realize(
                toolbox_id=tid,
                environment_name=target_name,
                tool_keys=tool_keys,
            )
        return {
            "status": "ok",
            "toolbox_id": tid,
            "source_environment_name": source_name,
            "target_environment_name": target_name,
            "resolved": resolved,
            "environment_description": dict(env_result.get("environment_description") or {}),
            "updated_direct_extra_packages": merged_packages,
            "apply_result": apply_result,
            "realize_result": realize_result,
        }

    def toolbox_environment_prepare_install(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from .toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        effective_env = self.toolbox_environment_description_effective_get(env_name)
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        planned_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            required_packages = list(packages.get("required_packages") or [])
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            effective_extra = [
                str(item or "").strip()
                for item in list(effective_env.get("effective_extra_packages") or [])
                if str(item or "").strip()
            ]
            missing_packages = [pkg for pkg in required_packages if pkg not in set(effective_extra)]
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.prepare_install_plan(
                spec,
                environment_description=effective_env,
                required_packages=required_packages,
                missing_packages=missing_packages,
                toolbox_id=tid,
                sandbox_profile_id=str(profile_id or "").strip(),
                tool_keys=matched_tool_keys,
            )
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["realization"] = dict(metadata.get("realization") or {})
            profile_row["environment"]["install_plan"] = dict(metadata.get("install_plan") or {})
            profiles[str(profile_id)] = profile_row
            planned_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": planned_profiles,
        }

    def toolbox_environment_lock_install(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from .toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        locked_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.lock_install_plan(spec)
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["install_plan"] = dict(metadata.get("install_plan") or {})
            profile_row["environment"]["install_lock"] = dict(metadata.get("install_lock") or {})
            profiles[str(profile_id)] = profile_row
            locked_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": locked_profiles,
        }

    def toolbox_environment_resolve_install_lock(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
        allow_resolution: bool = False,
    ) -> Dict[str, Any]:
        from .toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        resolved_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.resolve_install_lock(
                spec,
                allow_resolution=bool(allow_resolution),
            )
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["install_plan"] = dict(metadata.get("install_plan") or {})
            profile_row["environment"]["install_resolution"] = dict(metadata.get("install_resolution") or {})
            profile_row["environment"]["resolved_install_lock"] = dict(metadata.get("resolved_install_lock") or {})
            profiles[str(profile_id)] = profile_row
            resolved_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": resolved_profiles,
        }

    def toolbox_environment_verify_install_lock(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from .toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        verified_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.verify_install_lock(spec)
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["install_plan"] = dict(metadata.get("install_plan") or {})
            profile_row["environment"]["install_lock"] = dict(metadata.get("install_lock") or {})
            profile_row["environment"]["resolved_install_lock"] = dict(metadata.get("resolved_install_lock") or {})
            profile_row["environment"]["install_lock_verification"] = dict(metadata.get("install_lock_verification") or {})
            profiles[str(profile_id)] = profile_row
            verified_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": verified_profiles,
        }

    def toolbox_environment_verify_install_receipt(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from .toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        verified_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.verify_install_receipt(spec)
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["install_lock"] = dict(metadata.get("install_lock") or {})
            profile_row["environment"]["resolved_install_lock"] = dict(metadata.get("resolved_install_lock") or {})
            profile_row["environment"]["install_lock_verification"] = dict(metadata.get("install_lock_verification") or {})
            profile_row["environment"]["install_receipt"] = dict(metadata.get("install_receipt") or {})
            profile_row["environment"]["install_receipt_verification"] = dict(metadata.get("install_receipt_verification") or {})
            profiles[str(profile_id)] = profile_row
            verified_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": verified_profiles,
        }

    def toolbox_environment_execute_install(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
        allow_execution: bool = False,
    ) -> Dict[str, Any]:
        from .toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        executed_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.execute_install_plan(
                spec,
                allow_execution=bool(allow_execution),
            )
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["install_plan"] = dict(metadata.get("install_plan") or {})
            profile_row["environment"]["install_lock"] = dict(metadata.get("install_lock") or {})
            profile_row["environment"]["install_resolution"] = dict(metadata.get("install_resolution") or {})
            profile_row["environment"]["resolved_install_lock"] = dict(metadata.get("resolved_install_lock") or {})
            profile_row["environment"]["install_lock_verification"] = dict(metadata.get("install_lock_verification") or {})
            profile_row["environment"]["install_execution"] = dict(metadata.get("install_execution") or {})
            profile_row["environment"]["install_receipt"] = dict(metadata.get("install_receipt") or {})
            profile_row["environment"]["install_receipt_verification"] = dict(metadata.get("install_receipt_verification") or {})
            profiles[str(profile_id)] = profile_row
            executed_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": executed_profiles,
        }

    def _cleanup_unused_toolbox_environments(self, state: Optional[Dict[str, Any]] = None) -> List[str]:
        payload = dict(state or self._read_toolboxes() or {})
        toolboxes = dict(payload.get("toolboxes") or {})
        referenced_keys: set[str] = set()
        referenced_paths: set[str] = set()
        env_root = (self.hosting_root / "toolbox_venvs").resolve()
        for toolbox_row in toolboxes.values():
            profiles = dict(dict(toolbox_row or {}).get("profiles") or {})
            for profile_row in profiles.values():
                environment = dict(dict(profile_row or {}).get("environment") or {})
                venv_key = str(environment.get("venv_key") or "").strip()
                if venv_key:
                    referenced_keys.add(venv_key)
                raw_path = str(environment.get("venv_path") or "").strip()
                if not raw_path:
                    continue
                try:
                    resolved = Path(raw_path).expanduser().resolve()
                except Exception:
                    continue
                try:
                    if resolved == env_root or env_root in resolved.parents:
                        referenced_paths.add(str(resolved))
                except Exception:
                    continue
        removed: List[str] = []
        if not env_root.exists():
            return removed
        for child in env_root.iterdir():
            if not child.is_dir():
                continue
            try:
                resolved_child = str(child.expanduser().resolve())
            except Exception:
                resolved_child = ""
            if child.name in referenced_keys or (resolved_child and resolved_child in referenced_paths):
                continue
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child.name)
        return removed

    def _toolbox_reference_report(
        self,
        state: Optional[Dict[str, Any]] = None,
        *,
        include_reachability: bool = False,
        reachability_timeout_seconds: float = 0.35,
    ) -> Dict[str, Any]:
        payload = dict(state or self._read_toolboxes() or {})
        toolboxes = dict(payload.get("toolboxes") or {})
        referenced_engine_ids = self._referenced_toolbox_engine_ids(payload)
        referenced_env_keys: set[str] = set()
        referenced_env_roots: set[str] = set()
        referenced_env_key_reasons: Dict[str, List[Dict[str, Any]]] = {}
        referenced_env_root_reasons: Dict[str, List[Dict[str, Any]]] = {}
        env_root = (self.hosting_root / "toolbox_venvs").resolve()
        profiles_by_toolbox: Dict[str, Any] = {}
        for toolbox_id, raw_toolbox in toolboxes.items():
            toolbox_row = dict(raw_toolbox or {})
            profiles = dict(toolbox_row.get("profiles") or {})
            out_profiles: Dict[str, Any] = {}
            for profile_id, raw_profile in profiles.items():
                profile_row = dict(raw_profile or {})
                environment = dict(profile_row.get("environment") or {})
                venv_key = str(environment.get("venv_key") or "").strip()
                if venv_key:
                    referenced_env_keys.add(venv_key)
                    referenced_env_key_reasons.setdefault(venv_key, []).append(
                        {
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "kind": "profile_environment",
                        }
                    )
                raw_env_path = str(environment.get("venv_path") or "").strip()
                if raw_env_path:
                    try:
                        resolved_env_path = Path(raw_env_path).expanduser().resolve()
                    except Exception:
                        resolved_env_path = None
                    if resolved_env_path is not None:
                        try:
                            if resolved_env_path == env_root or env_root in resolved_env_path.parents:
                                env_root_value = str(resolved_env_path)
                                referenced_env_roots.add(env_root_value)
                                referenced_env_root_reasons.setdefault(env_root_value, []).append(
                                    {
                                        "toolbox_id": str(toolbox_id),
                                        "sandbox_profile_id": str(profile_id),
                                        "kind": "profile_environment",
                                        "venv_key": venv_key or None,
                                    }
                                )
                        except Exception:
                            pass
                out_profiles[str(profile_id)] = {
                    "engine_id": str(profile_row.get("engine_id") or "").strip() or None,
                    "bundle_revision": str(profile_row.get("bundle_revision") or "").strip() or None,
                    "environment": environment,
                    "sandbox_profile": dict(profile_row.get("sandbox_profile") or {}),
                    "request_count": len(list(profile_row.get("requests") or [])),
                    "rollout": dict(profile_row.get("rollout") or {}),
                }
            profiles_by_toolbox[str(toolbox_id)] = {
                "profiles": out_profiles,
                "runtime": dict(toolbox_row.get("runtime") or {}),
            }

        live_toolbox_regs: Dict[str, Dict[str, Any]] = {}
        stale_engine_ids: List[str] = []
        referenced_bundle_roots: set[str] = set()
        referenced_bundle_root_reasons: Dict[str, List[Dict[str, Any]]] = {}
        for reg in self._read_engines():
            row = dict(reg or {})
            if str(row.get("executor_kind") or "").strip() != "toolbox_executor_v1":
                continue
            engine_id = str(row.get("engine_id") or "").strip()
            bundle = dict(row.get("bundle") or {})
            bundle_root = str(bundle.get("bundle_root") or "").strip()
            is_referenced = engine_id in referenced_engine_ids
            if bundle_root and is_referenced:
                try:
                    resolved_bundle_root = str(Path(bundle_root).expanduser().resolve())
                    referenced_bundle_roots.add(resolved_bundle_root)
                    referenced_bundle_root_reasons.setdefault(resolved_bundle_root, []).append(
                        {
                            "engine_id": engine_id,
                            "toolbox_id": self._registration_toolbox_id(row) or None,
                            "sandbox_profile_id": self._registration_sandbox_profile_id(row) or None,
                            "kind": "live_registration",
                        }
                    )
                except Exception:
                    pass
            live_toolbox_regs[engine_id] = {
                "toolbox_id": self._registration_toolbox_id(row) or None,
                "sandbox_profile_id": self._registration_sandbox_profile_id(row),
                "bundle_root": bundle_root or None,
                "environment": dict(row.get("environment") or {}),
                "allowed_tool_names": sorted(list(self._registration_allowed_tool_names(row) or set())),
                "referenced": is_referenced,
            }
            if include_reachability and engine_id:
                reachability = self._probe_registration_reachability(
                    row,
                    timeout_seconds=reachability_timeout_seconds,
                )
                live_toolbox_regs[engine_id]["reachable"] = bool(reachability.get("reachable", False))
                live_toolbox_regs[engine_id]["reachability"] = dict(reachability or {})
            if engine_id and not is_referenced:
                stale_engine_ids.append(engine_id)

        bundles_root = (self.hosting_root / "toolbox_bundles").resolve()
        stale_bundle_roots: List[str] = []
        if bundles_root.exists():
            for child in bundles_root.iterdir():
                if not child.is_dir():
                    continue
                if not self._bundle_directory_is_referenced(
                    child,
                    referenced_bundle_roots=referenced_bundle_roots,
                ):
                    stale_bundle_roots.append(child.name)
        stale_environment_keys: List[str] = []
        if env_root.exists():
            for child in env_root.iterdir():
                if not child.is_dir():
                    continue
                try:
                    resolved_child = str(child.expanduser().resolve())
                except Exception:
                    resolved_child = ""
                if child.name not in referenced_env_keys and resolved_child not in referenced_env_roots:
                    stale_environment_keys.append(child.name)

        return {
            "status": "ok",
            "toolboxes": profiles_by_toolbox,
            "live_registrations": live_toolbox_regs,
            "referenced_engine_ids": sorted(referenced_engine_ids),
            "referenced_environment_keys": sorted(referenced_env_keys),
            "referenced_environment_roots": sorted(referenced_env_roots),
            "referenced_environment_key_reasons": {
                str(key): list(rows)
                for key, rows in sorted(referenced_env_key_reasons.items())
            },
            "referenced_environment_root_reasons": {
                str(key): list(rows)
                for key, rows in sorted(referenced_env_root_reasons.items())
            },
            "referenced_bundle_root_reasons": {
                str(key): list(rows)
                for key, rows in sorted(referenced_bundle_root_reasons.items())
            },
            "stale_engine_ids": sorted(stale_engine_ids),
            "stale_bundle_roots": sorted(stale_bundle_roots),
            "stale_environment_keys": sorted(stale_environment_keys),
            "summary": {
                "toolbox_count": len(profiles_by_toolbox),
                "live_registration_count": len(live_toolbox_regs),
                "referenced_engine_count": len(referenced_engine_ids),
                "stale_engine_count": len(stale_engine_ids),
                "stale_bundle_count": len(stale_bundle_roots),
                "stale_environment_count": len(stale_environment_keys),
            },
        }

    @staticmethod
    def _bundle_directory_is_referenced(
        directory: Path,
        *,
        referenced_bundle_roots: set[str],
    ) -> bool:
        try:
            base = directory.expanduser().resolve()
        except Exception:
            return False
        for raw in set(referenced_bundle_roots or set()):
            try:
                ref = Path(raw).expanduser().resolve()
            except Exception:
                continue
            if ref == base:
                return True
            try:
                ref.relative_to(base)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _toolbox_profile_expected_tool_names(
        toolbox_row: Optional[Dict[str, Any]],
        profile_id: str,
    ) -> List[str]:
        from .toolbox_harness import (
            ToolboxAutoAssignmentRequest,
            ToolboxManualAssignmentRequest,
        )

        toolbox_payload = dict(toolbox_row or {})
        profile_key = str(profile_id or "").strip()
        names: set[str] = set()

        for raw_request in list(toolbox_payload.get("requests") or []):
            try:
                req = ToolboxAutoAssignmentRequest.from_runtime_dict(dict(raw_request or {}))
            except Exception:
                continue
            if req.sandbox_profile.normalized_profile_id() != profile_key:
                continue
            tool_name = str(req.callable_name or "").strip()
            if tool_name:
                names.add(tool_name)

        for raw_request in list(toolbox_payload.get("manual_requests") or []):
            try:
                req = ToolboxManualAssignmentRequest.from_runtime_dict(dict(raw_request or {}))
            except Exception:
                continue
            if req.sandbox_profile.normalized_profile_id() != profile_key:
                continue
            fn = dict(dict(req.tool_definition or {}).get("function") or {})
            tool_name = str(fn.get("name") or "").strip()
            if tool_name:
                names.add(tool_name)

        intrinsics_row = dict(toolbox_payload.get("intrinsics") or {})
        if intrinsics_row:
            intrinsic_profile = dict(intrinsics_row.get("sandbox_profile") or {})
            intrinsic_profile_id = str(intrinsic_profile.get("profile_id") or "").strip() or "default"
            if intrinsic_profile_id == profile_key:
                include_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
                intrinsic_names = [
                    str(item or "").strip()
                    for item in list(intrinsics_row.get("names") or [])
                    if str(item or "").strip()
                ]
                names.update(
                    EngineHostService._normalize_intrinsic_tool_names(
                        intrinsic_names,
                        include_guides=include_guides,
                    )
                )
        return sorted(names)

    def toolbox_consistency(self) -> Dict[str, Any]:
        state = self._read_toolboxes()
        refs = self._toolbox_reference_report(
            state,
            include_reachability=True,
        )
        toolboxes = dict(state.get("toolboxes") or {})
        live_regs = {
            str(engine_id or "").strip(): dict(row or {})
            for engine_id, row in dict(refs.get("live_registrations") or {}).items()
            if str(engine_id or "").strip()
        }
        referenced_engine_ids = {
            str(item or "").strip()
            for item in list(refs.get("referenced_engine_ids") or [])
            if str(item or "").strip()
        }
        issues: List[Dict[str, Any]] = []

        for toolbox_id, raw_toolbox in toolboxes.items():
            toolbox_row = dict(raw_toolbox or {})
            profiles = dict(toolbox_row.get("profiles") or {})
            for profile_id, raw_profile in profiles.items():
                profile_row = dict(raw_profile or {})
                engine_id = str(profile_row.get("engine_id") or "").strip()
                sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
                expected_profile_id = str(sandbox_profile.get("profile_id") or "").strip() or str(profile_id or "").strip()
                environment = dict(profile_row.get("environment") or {})
                venv_key = str(environment.get("venv_key") or "").strip()
                venv_path = str(environment.get("venv_path") or "").strip()
                expected_tool_names = self._toolbox_profile_expected_tool_names(toolbox_row, str(profile_id or ""))

                if not engine_id:
                    issues.append(
                        {
                            "issue": "missing_profile_engine_id",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                        }
                    )
                    continue
                if engine_id not in referenced_engine_ids:
                    issues.append(
                        {
                            "issue": "unreferenced_profile_engine_id",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                        }
                    )
                live = dict(live_regs.get(engine_id) or {})
                if not live:
                    issues.append(
                        {
                            "issue": "missing_live_registration",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                        }
                    )
                    continue
                if live.get("reachable") is False:
                    reachability = dict(live.get("reachability") or {})
                    issues.append(
                        {
                            "issue": "live_registration_unreachable",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                            "probe": str(reachability.get("probe") or "").strip() or None,
                            "reachability_error": str(reachability.get("error") or "").strip() or None,
                        }
                    )
                live_toolbox_id = str(live.get("toolbox_id") or "").strip()
                if live_toolbox_id and live_toolbox_id != str(toolbox_id):
                    issues.append(
                        {
                            "issue": "registration_toolbox_id_mismatch",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                            "expected_toolbox_id": str(toolbox_id),
                            "actual_toolbox_id": live_toolbox_id,
                        }
                    )
                live_profile_id = str(live.get("sandbox_profile_id") or "").strip()
                if live_profile_id and live_profile_id != expected_profile_id:
                    issues.append(
                        {
                            "issue": "registration_profile_id_mismatch",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                            "expected_sandbox_profile_id": expected_profile_id,
                            "actual_sandbox_profile_id": live_profile_id,
                        }
                    )
                live_allowed = sorted(
                    [
                        str(item or "").strip()
                        for item in list(live.get("allowed_tool_names") or [])
                        if str(item or "").strip()
                    ]
                )
                if live_allowed != expected_tool_names:
                    issues.append(
                        {
                            "issue": "registration_allowed_tool_names_mismatch",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                            "expected_tool_names": expected_tool_names,
                            "actual_tool_names": live_allowed,
                        }
                    )

                if venv_key and not venv_path:
                    issues.append(
                        {
                            "issue": "environment_path_missing",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                            "venv_key": venv_key,
                        }
                    )
                elif venv_key and venv_path:
                    try:
                        env_path = Path(venv_path).expanduser().resolve()
                    except Exception:
                        env_path = None
                    if env_path is None or not env_path.exists():
                        issues.append(
                            {
                                "issue": "environment_path_missing_on_disk",
                                "toolbox_id": str(toolbox_id),
                                "sandbox_profile_id": str(profile_id),
                                "engine_id": engine_id,
                                "venv_key": venv_key,
                                "venv_path": venv_path,
                            }
                        )
                    else:
                        metadata_path = env_path / "environment.json"
                        if not metadata_path.exists():
                            issues.append(
                                {
                                    "issue": "environment_metadata_missing",
                                    "toolbox_id": str(toolbox_id),
                                    "sandbox_profile_id": str(profile_id),
                                    "engine_id": engine_id,
                                    "venv_key": venv_key,
                                    "venv_path": str(env_path),
                                }
                            )

        return {
            "status": "ok",
            "issue_count": len(issues),
            "issues": issues,
            "references": refs,
            "summary": {
                "issue_count": len(issues),
                "toolbox_count": len(toolboxes),
                "referenced_engine_count": len(referenced_engine_ids),
            },
        }

    def toolbox_review_snapshot(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        scoped_ids = {
            str(item or "").strip()
            for item in list(toolbox_ids or [])
            if str(item or "").strip()
        }
        references = self.toolbox_references()
        consistency = self.toolbox_consistency()
        consistency_references = dict(consistency.get("references") or {})
        if consistency_references:
            references = {
                **references,
                "live_registrations": dict(consistency_references.get("live_registrations") or dict(references.get("live_registrations") or {})),
                "summary": {
                    **dict(references.get("summary") or {}),
                    **dict(consistency_references.get("summary") or {}),
                },
            }
        if scoped_ids:
            filtered_toolboxes = {
                str(k): dict(v or {})
                for k, v in dict(references.get("toolboxes") or {}).items()
                if str(k or "").strip() in scoped_ids
            }
            filtered_issues = [
                dict(item or {})
                for item in list(consistency.get("issues") or [])
                if str(dict(item or {}).get("toolbox_id") or "").strip() in scoped_ids
            ]
            references = {
                **references,
                "toolboxes": filtered_toolboxes,
                "summary": {
                    **dict(references.get("summary") or {}),
                    "toolbox_count": len(filtered_toolboxes),
                },
            }
            consistency = {
                **consistency,
                "issue_count": len(filtered_issues),
                "issues": filtered_issues,
                "summary": {
                    **dict(consistency.get("summary") or {}),
                    "issue_count": len(filtered_issues),
                    "toolbox_count": len(filtered_toolboxes),
                },
            }
        issues = [dict(item or {}) for item in list(consistency.get("issues") or [])]
        issues_by_toolbox: Dict[str, List[Dict[str, Any]]] = {}
        for item in issues:
            toolbox_id = str(item.get("toolbox_id") or "").strip()
            issues_by_toolbox.setdefault(toolbox_id, []).append(item)

        toolbox_rows: Dict[str, Dict[str, Any]] = {}
        live_regs = dict(references.get("live_registrations") or {})
        for toolbox_id, raw_toolbox in dict(references.get("toolboxes") or {}).items():
            toolbox_row = dict(raw_toolbox or {})
            profile_rows: List[Dict[str, Any]] = []
            for profile_id, raw_profile in dict(toolbox_row.get("profiles") or {}).items():
                profile_row = dict(raw_profile or {})
                rollout = dict(profile_row.get("rollout") or {})
                environment = dict(profile_row.get("environment") or {})
                engine_id = str(profile_row.get("engine_id") or "").strip()
                live_reg = dict(live_regs.get(engine_id) or {})
                tool_names = [
                    str(item or "").strip()
                    for item in list(
                        rollout.get("all_registered_tool_names")
                        or live_reg.get("all_registered_tool_names")
                        or live_reg.get("allowed_tool_names")
                        or []
                    )
                    if str(item or "").strip()
                ]
                advertised_tool_names = [
                    str(item or "").strip()
                    for item in list(
                        rollout.get("advertised_tool_names")
                        or live_reg.get("advertised_tool_names")
                        or []
                    )
                    if str(item or "").strip()
                ]
                hidden_allowed_tool_names = [
                    str(item or "").strip()
                    for item in list(
                        rollout.get("hidden_allowed_tool_names")
                        or live_reg.get("hidden_allowed_tool_names")
                        or []
                    )
                    if str(item or "").strip()
                ]
                profile_rows.append(
                    {
                        "sandbox_profile_id": str(dict(profile_row.get("sandbox_profile") or {}).get("profile_id") or profile_id or "").strip(),
                        "environment_name": str(environment.get("environment_name") or "").strip() or None,
                        "all_registered_tool_names": sorted(tool_names),
                        "advertised_tool_names": sorted(advertised_tool_names),
                        "hidden_allowed_tool_names": sorted(hidden_allowed_tool_names),
                        "engine_id": engine_id or None,
                        "reachable": live_reg.get("reachable"),
                        "ready": bool(rollout.get("ready", False)),
                        "warmup_ms": int(rollout.get("warmup_ms") or 0),
                    }
                )
            profile_rows.sort(key=lambda item: (str(item.get("environment_name") or ""), str(item.get("sandbox_profile_id") or "")))
            toolbox_issues = list(issues_by_toolbox.get(str(toolbox_id or "").strip()) or [])
            toolbox_rows[str(toolbox_id)] = {
                "profile_count": len(profile_rows),
                "profiles": profile_rows,
                "issue_count": len(toolbox_issues),
                "issue_names": sorted(
                    {
                        str(item.get("issue") or "").strip()
                        for item in toolbox_issues
                        if str(item.get("issue") or "").strip()
                    }
                ),
                "live_registration_count": sum(
                    1 for item in profile_rows if str(item.get("engine_id") or "").strip()
                ),
            }

        issue_count = len(issues)
        return {
            "status": "ok",
            "toolbox_ids": sorted(scoped_ids),
            "toolboxes": toolbox_rows,
            "issues": issues,
            "recommended_action": "reconcile" if issue_count > 0 else "observe",
            "references_summary": dict(references.get("summary") or {}),
            "consistency_summary": {
                **dict(consistency.get("summary") or {}),
                "issue_count": issue_count,
            },
            "summary": {
                "toolbox_count": int(dict(dict(references or {}).get("summary") or {}).get("toolbox_count") or 0),
                "issue_count": issue_count,
                "stale_engine_count": int(dict(dict(references or {}).get("summary") or {}).get("stale_engine_count") or 0),
                "stale_bundle_count": int(dict(dict(references or {}).get("summary") or {}).get("stale_bundle_count") or 0),
                "stale_environment_count": int(dict(dict(references or {}).get("summary") or {}).get("stale_environment_count") or 0),
            },
        }

    def toolbox_repair(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
    ) -> Dict[str, Any]:
        from .toolbox_harness import (
            SandboxProfileSpec,
            ToolboxAutoAssignmentRequest,
            ToolboxBundleStager,
            ToolboxManualAssignmentRequest,
            ToolboxSandboxOrchestrator,
        )

        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        consistency = self.toolbox_consistency()
        issues = list(consistency.get("issues") or [])
        inconsistent_toolbox_ids = sorted(
            {
                str(item.get("toolbox_id") or "").strip()
                for item in issues
                if str(item.get("toolbox_id") or "").strip()
            }
        )
        requested_toolbox_ids = [
            str(item or "").strip()
            for item in list(toolbox_ids or [])
            if str(item or "").strip()
        ]
        if requested_toolbox_ids:
            if only_inconsistent:
                inconsistent_set = set(inconsistent_toolbox_ids)
                target_toolbox_ids = [
                    item for item in requested_toolbox_ids
                    if item in inconsistent_set
                ]
            else:
                target_toolbox_ids = requested_toolbox_ids
        elif only_inconsistent:
            target_toolbox_ids = inconsistent_toolbox_ids
        else:
            target_toolbox_ids = sorted(str(item or "").strip() for item in toolboxes.keys() if str(item or "").strip())

        repaired: Dict[str, Any] = {}
        skipped: Dict[str, str] = {}

        with self._locked_toolboxes(target_toolbox_ids):
            state = self._read_toolboxes()
            toolboxes = dict(state.get("toolboxes") or {})

            for tid in target_toolbox_ids:
                toolbox_row_existing = dict(toolboxes.get(tid) or {})
                if not toolbox_row_existing:
                    skipped[tid] = "toolbox_not_found"
                    continue
                runtime = dict(toolbox_row_existing.get("runtime") or {})
                auto_requests = [
                    ToolboxAutoAssignmentRequest.from_runtime_dict(dict(item or {}))
                    for item in list(toolbox_row_existing.get("requests") or [])
                ]
                manual_requests = [
                    ToolboxManualAssignmentRequest.from_runtime_dict(dict(item or {}))
                    for item in list(toolbox_row_existing.get("manual_requests") or [])
                ]
                intrinsics_row = dict(toolbox_row_existing.get("intrinsics") or {})
                intrinsic_names = self._normalize_intrinsic_tool_names(
                    [str(item or "").strip() for item in list(intrinsics_row.get("names") or []) if str(item or "").strip()],
                    include_guides=bool(intrinsics_row.get("with_intrinsic_guides", False)),
                )
                intrinsic_profile = (
                    SandboxProfileSpec.from_dict(dict(intrinsics_row.get("sandbox_profile") or {}))
                    if intrinsic_names
                    else None
                )
                with_intrinsic_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
                existing_profiles = dict(toolbox_row_existing.get("profiles") or {})
                old_regs_by_profile: Dict[str, str] = {}
                for reg in self._toolbox_executor_registrations(tid):
                    old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

                if not auto_requests and not manual_requests and not intrinsic_names:
                    skipped[tid] = "toolbox_has_no_persisted_requests"
                    continue

                orchestrator = ToolboxSandboxOrchestrator(
                    service=self,
                    stager=ToolboxBundleStager(self.hosting_root),
                    python_executable=str(runtime.get("python_executable") or "").strip() or None,
                )
                assignments = orchestrator.spawn_assignments(
                    toolbox_id=tid,
                    requests=auto_requests,
                    manual_requests=manual_requests,
                    intrinsic_tool_names=intrinsic_names,
                    intrinsic_profile=intrinsic_profile,
                    with_intrinsic_guides=with_intrinsic_guides,
                    worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
                )
                try:
                    ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
                except Exception:
                    for item in assignments:
                        reg = dict(item.registration or {})
                        engine_id = str(reg.get("engine_id") or "").strip()
                        if engine_id:
                            self._retire_toolbox_registration(engine_id)
                    self._cleanup_unused_toolbox_environments(state)
                    raise

                new_profiles: Dict[str, Dict[str, Any]] = {}
                spawned_engine_ids: List[str] = []
                replaced_engine_ids: List[str] = []
                for item in assignments:
                    profile_id = item.sandbox_profile.normalized_profile_id()
                    reg = dict(item.registration or {})
                    engine_id = str(reg.get("engine_id") or "").strip()
                    if engine_id:
                        spawned_engine_ids.append(engine_id)
                    old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
                    bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
                    if old_engine_id and old_engine_id != engine_id:
                        replaced_engine_ids.append(old_engine_id)
                    profile_auto_requests = [
                        req.to_runtime_dict()
                        for req in auto_requests
                        if req.sandbox_profile.normalized_profile_id() == profile_id
                    ]
                    profile_manual_requests = [
                        req.to_runtime_dict()
                        for req in manual_requests
                        if req.sandbox_profile.normalized_profile_id() == profile_id
                    ]
                    new_profiles[profile_id] = {
                        "sandbox_profile": item.sandbox_profile.to_dict(),
                        "requests": profile_auto_requests,
                        "manual_requests": profile_manual_requests,
                        "engine_id": engine_id,
                        "bundle_revision": bundle_revision,
                        "environment": dict(reg.get("environment") or {}),
                        "rollout": dict(ready_rollout.get(engine_id) or {}),
                        "rollout_history": self._append_toolbox_rollout_history(
                            dict(existing_profiles.get(profile_id) or {}),
                            rollout=dict(ready_rollout.get(engine_id) or {}),
                            action="repair",
                            engine_id=engine_id,
                            bundle_revision=bundle_revision,
                            replaced_engine_id=old_engine_id,
                        ),
                    }
                for profile_id, old_engine_id in old_regs_by_profile.items():
                    if profile_id not in new_profiles and old_engine_id:
                        replaced_engine_ids.append(old_engine_id)
                for old_engine_id in replaced_engine_ids:
                    self._retire_toolbox_registration(old_engine_id)

                toolboxes[tid] = {
                    "toolbox_id": tid,
                    "requests": [req.to_runtime_dict() for req in auto_requests],
                    "manual_requests": [req.to_runtime_dict() for req in manual_requests],
                    "profiles": new_profiles,
                    "runtime": runtime,
                    **(
                        {
                            "intrinsics": {
                                "names": intrinsic_names,
                                "sandbox_profile": (intrinsic_profile or SandboxProfileSpec(profile_id="default")).to_dict(),
                                "with_intrinsic_guides": with_intrinsic_guides,
                            }
                        }
                        if intrinsic_names
                        else {}
                    ),
                }
                repaired[tid] = {
                    "profiles": new_profiles,
                    "spawned_engine_ids": spawned_engine_ids,
                    "ready_engine_ids": list(ready_rollout.keys()),
                    "rollout": ready_rollout,
                    "replaced_engine_ids": replaced_engine_ids,
                }

            state["toolboxes"] = toolboxes
            self._write_toolboxes(state)
            removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        result = {
            "status": "ok",
            "requested_toolbox_ids": requested_toolbox_ids,
            "target_toolbox_ids": target_toolbox_ids,
            "inconsistent_toolbox_ids": inconsistent_toolbox_ids,
            "repaired_toolbox_ids": sorted(
                str(item or "").strip()
                for item in repaired.keys()
                if str(item or "").strip()
            ),
            "skipped_toolbox_ids": sorted(
                str(item or "").strip()
                for item in skipped.keys()
                if str(item or "").strip()
            ),
            "removed_environment_keys": removed_environment_keys,
            "outcome": "repaired" if repaired else "noop",
            "summary": {
                "requested_toolbox_count": len(requested_toolbox_ids),
                "target_toolbox_count": len(target_toolbox_ids),
                "repaired_toolbox_count": len(repaired),
                "skipped_toolbox_count": len(skipped),
                "removed_environment_count": len(removed_environment_keys),
            },
        }
        if details:
            result["repaired"] = repaired
            result["skipped"] = skipped
        return result

    def toolbox_reconcile(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
    ) -> Dict[str, Any]:
        requested_toolbox_ids = [
            str(item or "").strip()
            for item in list(toolbox_ids or [])
            if str(item or "").strip()
        ]
        before = self.toolbox_consistency()
        repair = self.toolbox_repair(
            toolbox_ids=toolbox_ids,
            only_inconsistent=only_inconsistent,
        )
        gc_out = self.toolbox_gc()
        after = self.toolbox_consistency()
        repair_requested_toolbox_ids = list(dict(repair or {}).get("requested_toolbox_ids") or requested_toolbox_ids)
        repair_inconsistent_toolbox_ids = list(dict(repair or {}).get("inconsistent_toolbox_ids") or [])
        repaired_toolbox_ids = sorted(
            str(item or "").strip()
            for item in dict(dict(repair or {}).get("repaired") or {}).keys()
            if str(item or "").strip()
        )
        repair_target_toolbox_ids = list(dict(repair or {}).get("target_toolbox_ids") or repaired_toolbox_ids)
        skipped_toolbox_ids = sorted(
            str(item or "").strip()
            for item in dict(dict(repair or {}).get("skipped") or {}).keys()
            if str(item or "").strip()
        )
        result = {
            "status": "ok",
            "toolbox_ids": requested_toolbox_ids,
            "requested_toolbox_ids": repair_requested_toolbox_ids,
            "target_toolbox_ids": repair_target_toolbox_ids,
            "inconsistent_toolbox_ids": repair_inconsistent_toolbox_ids,
            "repaired_toolbox_ids": repaired_toolbox_ids,
            "skipped_toolbox_ids": skipped_toolbox_ids,
            "removed_engine_ids": list(dict(gc_out or {}).get("removed_engine_ids") or []),
            "removed_bundle_roots": list(dict(gc_out or {}).get("removed_bundle_roots") or []),
            "removed_environment_keys": list(dict(gc_out or {}).get("removed_environment_keys") or []),
            "outcome": (
                "repaired"
                if repaired_toolbox_ids
                else "noop"
            ),
            "summary": {
                "before_issue_count": int(dict(before or {}).get("issue_count") or 0),
                "after_issue_count": int(dict(after or {}).get("issue_count") or 0),
                "removed_engine_count": len(list(dict(gc_out or {}).get("removed_engine_ids") or [])),
                "removed_bundle_count": len(list(dict(gc_out or {}).get("removed_bundle_roots") or [])),
                "removed_environment_count": len(list(dict(gc_out or {}).get("removed_environment_keys") or [])),
                "repaired_toolbox_count": len(repaired_toolbox_ids),
                "requested_toolbox_count": len(repair_requested_toolbox_ids),
                "target_toolbox_count": len(repair_target_toolbox_ids),
            },
        }
        if details:
            result["before"] = before
            result["repair"] = repair
            result["gc"] = gc_out
            result["after"] = after
        return result

    @staticmethod
    def _referenced_toolbox_engine_ids(state: Optional[Dict[str, Any]] = None) -> set[str]:
        payload = dict(state or {})
        toolboxes = dict(payload.get("toolboxes") or {})
        referenced: set[str] = set()
        for toolbox_row in toolboxes.values():
            profiles = dict(dict(toolbox_row or {}).get("profiles") or {})
            for profile_row in profiles.values():
                engine_id = str(dict(profile_row or {}).get("engine_id") or "").strip()
                if engine_id:
                    referenced.add(engine_id)
        return referenced

    def _cleanup_stale_toolbox_registrations(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = dict(state or self._read_toolboxes() or {})
        referenced_engine_ids = self._referenced_toolbox_engine_ids(payload)
        removed: List[str] = []
        removed_bundle_roots: List[str] = []
        removed_details: List[Dict[str, Any]] = []
        for reg in self._read_engines():
            row = dict(reg or {})
            if str(row.get("executor_kind") or "").strip() != "toolbox_executor_v1":
                continue
            engine_id = str(row.get("engine_id") or "").strip()
            if not engine_id or engine_id in referenced_engine_ids:
                continue
            bundle = dict(row.get("bundle") or {})
            raw_root = str(bundle.get("bundle_root") or "").strip()
            bundle_name = ""
            if raw_root:
                try:
                    bundle_name = Path(raw_root).expanduser().resolve().name
                except Exception:
                    bundle_name = ""
            self._retire_toolbox_registration(engine_id)
            removed.append(engine_id)
            if bundle_name:
                removed_bundle_roots.append(bundle_name)
            removed_details.append(
                {
                    "engine_id": engine_id,
                    "toolbox_id": self._registration_toolbox_id(row) or None,
                    "sandbox_profile_id": self._registration_sandbox_profile_id(row) or None,
                    "bundle_root": raw_root or None,
                    "bundle_name": bundle_name or None,
                    "reason": "unreferenced_live_registration",
                }
            )
        return {
            "removed_engine_ids": removed,
            "removed_bundle_roots": removed_bundle_roots,
            "removed_registration_details": removed_details,
        }

    def _cleanup_unused_toolbox_bundles(self) -> Dict[str, Any]:
        bundles_root = (self.hosting_root / "toolbox_bundles").resolve()
        removed: List[str] = []
        removed_details: List[Dict[str, Any]] = []
        if not bundles_root.exists():
            return {"removed_bundle_roots": removed, "removed_bundle_details": removed_details}
        referenced_roots: set[str] = set()
        for reg in self._read_engines():
            row = dict(reg or {})
            if str(row.get("executor_kind") or "").strip() != "toolbox_executor_v1":
                continue
            bundle = dict(row.get("bundle") or {})
            raw = str(bundle.get("bundle_root") or "").strip()
            if not raw:
                continue
            try:
                referenced_roots.add(str(Path(raw).expanduser().resolve()))
            except Exception:
                continue
        for child in bundles_root.iterdir():
            if not child.is_dir():
                continue
            if self._bundle_directory_is_referenced(
                child,
                referenced_bundle_roots=referenced_roots,
            ):
                continue
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child.name)
            removed_details.append(
                {
                    "bundle_name": child.name,
                    "bundle_root": str(child),
                    "reason": "unreferenced_bundle_directory",
                }
            )
        return {
            "removed_bundle_roots": removed,
            "removed_bundle_details": removed_details,
        }

    def toolbox_gc(self) -> Dict[str, Any]:
        state = self._read_toolboxes()
        stale = self._cleanup_stale_toolbox_registrations(state)
        bundle_gc = self._cleanup_unused_toolbox_bundles()
        removed_bundle_roots = list(stale.get("removed_bundle_roots") or [])
        removed_bundle_roots.extend(list(bundle_gc.get("removed_bundle_roots") or []))
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        removed_environment_details = [
            {
                "venv_key": str(item or "").strip(),
                "reason": "unreferenced_environment_directory",
            }
            for item in list(removed_environment_keys or [])
            if str(item or "").strip()
        ]
        return {
            "status": "ok",
            "removed_engine_ids": list(stale.get("removed_engine_ids") or []),
            "removed_bundle_roots": list(dict.fromkeys([item for item in removed_bundle_roots if str(item or "").strip()])),
            "removed_environment_keys": removed_environment_keys,
            "removed_registration_details": list(stale.get("removed_registration_details") or []),
            "removed_bundle_details": list(bundle_gc.get("removed_bundle_details") or []),
            "removed_environment_details": removed_environment_details,
            "summary": {
                "removed_engine_count": len(list(stale.get("removed_engine_ids") or [])),
                "removed_bundle_count": len(list(dict.fromkeys([item for item in removed_bundle_roots if str(item or "").strip()]))),
                "removed_environment_count": len(removed_environment_keys),
            },
        }

    def toolbox_references(self) -> Dict[str, Any]:
        return self._toolbox_reference_report()

    @staticmethod
    def _append_toolbox_rollout_history(
        existing_profile_row: Optional[Dict[str, Any]],
        *,
        rollout: Dict[str, Any],
        action: str,
        engine_id: str,
        bundle_revision: str,
        replaced_engine_id: str = "",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        history = list(dict(existing_profile_row or {}).get("rollout_history") or [])
        entry = {
            "action": str(action or "").strip() or "rollout",
            "engine_id": str(engine_id or "").strip(),
            "bundle_revision": str(bundle_revision or "").strip(),
            "replaced_engine_id": str(replaced_engine_id or "").strip() or None,
            "ready": bool(dict(rollout or {}).get("ready", False)),
            "ready_at": dict(rollout or {}).get("ready_at"),
            "warmup_ms": int(dict(rollout or {}).get("warmup_ms") or 0),
        }
        history.append(entry)
        max_items = max(1, int(limit or 10))
        if len(history) > max_items:
            history = history[-max_items:]
        return history

    @staticmethod
    def _default_control_payload() -> Dict[str, Any]:
        return {
            "version": 1,
            "control_config": {
                "ssh_key": None,
                "require_auth": False,
                "auth": {"keys": {}, "sessions": {}, "challenges": {}},
                "access_profile": {"connectivity_mode": "local_only"},
                "endpoint_mode_default": "exclusive",
                "lifecycle_profile": LIFECYCLE_PROFILE_DETACHED,
                "lifecycle_policy": {
                    "on_terminal_disconnect": "keep_daemon_running",
                    "terminal_control_enabled": True,
                    "owner_disconnect_shutdown": False,
                },
                "config_store_mode": "store_only",
                "claim_acl_policy": {
                    "owner_ttl_seconds": 120,
                    "audit_event_limit": 200,
                },
                "engine_traffic_policies": {},
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
            "ownership_change_notices": {},
            "auth_audit_events": [],
        }

    def _control_layout(self) -> Dict[str, Path]:
        return {
            "root": self.hosting_root,
            "access_control": self.hosting_root / "access_control.json",
            "keys": self.hosting_root / "keyring" / "keys.json",
            "sessions": self.hosting_root / "state" / "sessions.json",
            "challenges": self.hosting_root / "state" / "challenges.json",
            "runtime_state": self.hosting_root / "state" / "runtime_state.json",
            "auth_audit": self.hosting_root / "audit" / "auth_audit.json",
            "claim_audit": self.hosting_root / "audit" / "claim_audit.json",
        }

    def _read_control(self) -> Dict[str, Any]:
        default_payload = self._default_control_payload()
        layout = self._control_layout()
        access_default = dict(default_payload.get("control_config") or {})
        access_payload = self._read_json(
            layout["access_control"],
            {"version": 1, "updated_at": 0.0, "control_config": access_default},
        )
        runtime_payload = self._read_json(
            layout["runtime_state"],
            {
                "version": 1,
                "updated_at": 0.0,
                "claims_by_engine": {},
                "endpoint_claim": {"owners": [], "exclusive_owner": None, "claimed_at": 0.0},
                "tokens": {},
                "resource_claims": {},
                "resource_tokens": {},
                "claim_owner_keepalive": {},
                "ownership_change_notices": {},
            },
        )
        keys_payload = self._read_json(layout["keys"], {"version": 1, "updated_at": 0.0, "keys": {}})
        sessions_payload = self._read_json(layout["sessions"], {"version": 1, "updated_at": 0.0, "sessions": {}})
        challenges_payload = self._read_json(
            layout["challenges"],
            {"version": 1, "updated_at": 0.0, "challenges": {}},
        )
        auth_audit_payload = self._read_json(layout["auth_audit"], {"version": 1, "updated_at": 0.0, "events": []})
        claim_audit_payload = self._read_json(
            layout["claim_audit"],
            {"version": 1, "updated_at": 0.0, "events": []},
        )
        payload = {
            "version": 1,
            "updated_at": max(
                float(access_payload.get("updated_at") or 0.0),
                float(runtime_payload.get("updated_at") or 0.0),
                float(keys_payload.get("updated_at") or 0.0),
                float(sessions_payload.get("updated_at") or 0.0),
                float(challenges_payload.get("updated_at") or 0.0),
                float(auth_audit_payload.get("updated_at") or 0.0),
                float(claim_audit_payload.get("updated_at") or 0.0),
            ),
            "control_config": dict(access_payload.get("control_config") or access_default),
            "claims_by_engine": dict(runtime_payload.get("claims_by_engine") or {}),
            "endpoint_claim": dict(
                runtime_payload.get("endpoint_claim") or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0}
            ),
            "tokens": dict(runtime_payload.get("tokens") or {}),
            "resource_claims": dict(runtime_payload.get("resource_claims") or {}),
            "resource_tokens": dict(runtime_payload.get("resource_tokens") or {}),
            "claim_owner_keepalive": dict(runtime_payload.get("claim_owner_keepalive") or {}),
            "ownership_change_notices": dict(runtime_payload.get("ownership_change_notices") or {}),
            "claim_audit_events": list(claim_audit_payload.get("events") or []),
            "auth_audit_events": list(auth_audit_payload.get("events") or []),
        }
        cfg = dict(payload.get("control_config") or {})
        cfg["auth"] = {
            "keys": dict(keys_payload.get("keys") or {}),
            "sessions": dict(sessions_payload.get("sessions") or {}),
            "challenges": dict(challenges_payload.get("challenges") or {}),
        }
        payload["control_config"] = cfg
        payload.setdefault(
            "control_config",
            {
                "ssh_key": None,
                "require_auth": False,
                "auth": {"keys": {}, "sessions": {}, "challenges": {}},
                "access_profile": {"connectivity_mode": "local_only"},
                "endpoint_mode_default": "exclusive",
                "lifecycle_profile": LIFECYCLE_PROFILE_DETACHED,
                "lifecycle_policy": {
                    "on_terminal_disconnect": "keep_daemon_running",
                    "terminal_control_enabled": True,
                    "owner_disconnect_shutdown": False,
                },
                "config_store_mode": "store_only",
                "claim_acl_policy": {},
                "engine_traffic_policies": {},
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
        payload.setdefault("ownership_change_notices", {})
        payload.setdefault("auth_audit_events", [])
        cfg = dict(payload.get("control_config") or {})
        cfg.setdefault("ssh_key", None)
        cfg.setdefault("require_auth", False)
        cfg.setdefault("access_profile", {"connectivity_mode": "local_only"})
        cfg.setdefault("endpoint_mode_default", "exclusive")
        cfg["lifecycle_profile"] = self._normalize_lifecycle_profile(cfg.get("lifecycle_profile"))
        cfg["lifecycle_policy"] = self._normalize_lifecycle_policy(
            cfg["lifecycle_profile"],
            dict(cfg.get("lifecycle_policy") or {}),
        )
        cfg["endpoint_mode_default"] = (
            "exclusive"
            if str(cfg.get("endpoint_mode_default") or "").strip().lower() == "exclusive"
            else "shared"
        )
        cfg.setdefault("config_store_mode", "store_only")
        raw_claim_acl = dict(cfg.get("claim_acl_policy") or {})
        cfg["claim_acl_policy"] = {
            "owner_ttl_seconds": max(10, min(int(raw_claim_acl.get("owner_ttl_seconds") or 120), 24 * 3600)),
            "audit_event_limit": max(20, min(int(raw_claim_acl.get("audit_event_limit") or 200), 2000)),
        }
        cfg.setdefault("engine_traffic_policies", {})
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
        layout = self._control_layout()
        cfg = dict(out.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        cfg_without_auth = dict(cfg)
        cfg_without_auth.pop("auth", None)
        self._write_json(
            layout["access_control"],
            {
                "version": 1,
                "updated_at": out["updated_at"],
                "control_config": cfg_without_auth,
            },
        )
        self._write_json(
            layout["keys"],
            {"version": 1, "updated_at": out["updated_at"], "keys": dict(auth.get("keys") or {})},
        )
        self._write_json(
            layout["sessions"],
            {"version": 1, "updated_at": out["updated_at"], "sessions": dict(auth.get("sessions") or {})},
        )
        self._write_json(
            layout["challenges"],
            {"version": 1, "updated_at": out["updated_at"], "challenges": dict(auth.get("challenges") or {})},
        )
        self._write_json(
            layout["runtime_state"],
            {
                "version": 1,
                "updated_at": out["updated_at"],
                "claims_by_engine": dict(out.get("claims_by_engine") or {}),
                "endpoint_claim": dict(out.get("endpoint_claim") or {}),
                "tokens": dict(out.get("tokens") or {}),
                "resource_claims": dict(out.get("resource_claims") or {}),
                "resource_tokens": dict(out.get("resource_tokens") or {}),
                "claim_owner_keepalive": dict(out.get("claim_owner_keepalive") or {}),
                "ownership_change_notices": dict(out.get("ownership_change_notices") or {}),
            },
        )
        self._write_json(
            layout["auth_audit"],
            {"version": 1, "updated_at": out["updated_at"], "events": list(out.get("auth_audit_events") or [])},
        )
        self._write_json(
            layout["claim_audit"],
            {"version": 1, "updated_at": out["updated_at"], "events": list(out.get("claim_audit_events") or [])},
        )

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

    def _ownership_change_notice_map(self, control: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        rows: Dict[str, Dict[str, Any]] = {}
        for key, val in dict(control.get("ownership_change_notices") or {}).items():
            actor = str(key or "").strip()
            if not actor:
                continue
            meta = dict(val or {})
            rows[actor] = meta
        return rows

    def _get_ownership_change_notice(self, control: Dict[str, Any], actor_id: str) -> Optional[Dict[str, Any]]:
        actor = str(actor_id or "").strip()
        if not actor:
            return None
        rows = self._ownership_change_notice_map(control)
        notice = dict(rows.get(actor) or {})
        return notice if notice else None

    def _clear_ownership_change_notice(self, control: Dict[str, Any], actor_id: str) -> None:
        actor = str(actor_id or "").strip()
        if not actor:
            return
        rows = self._ownership_change_notice_map(control)
        if actor in rows:
            rows.pop(actor, None)
            control["ownership_change_notices"] = rows

    def _record_ownership_change_notices(
        self,
        control: Dict[str, Any],
        *,
        displaced_owners: List[str],
        replaced_by: str,
        scope: str,
        resource_kind: Optional[str],
        resource_id: Optional[str],
        reason: Optional[str],
        emergency: bool,
        peer_host: Optional[str],
        command: str,
    ) -> None:
        rows = self._ownership_change_notice_map(control)
        rep = str(replaced_by or "").strip()
        now = time.time()
        for owner in [str(x or "").strip() for x in list(displaced_owners or []) if str(x or "").strip()]:
            if not owner or owner == rep:
                continue
            notice = {
                "schema_version": 1,
                "owner_id": owner,
                "replaced_by": rep,
                "scope": str(scope or ""),
                "resource_kind": str(resource_kind or "") or None,
                "resource_id": str(resource_id or "") or None,
                "reason": str(reason or "") or None,
                "emergency": bool(emergency),
                "changed_at": now,
                "active": True,
            }
            rows[owner] = notice
            self._append_claim_audit_event(
                control,
                event_type="ownership_changed_notice",
                command=str(command or ""),
                scope=str(scope or ""),
                resource_kind=resource_kind,
                resource_id=resource_id,
                actor_id=rep,
                decision="grant",
                code="ownership_changed_notice",
                transition="force_override",
                mode=None,
                peer_host=peer_host,
                owners_before=[owner],
                owners_after=[rep],
                details={"displaced_owner": owner, "force_override_reason": reason, "force_override_emergency": bool(emergency)},
                severity="high",
            )
        control["ownership_change_notices"] = rows

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
        severity: Optional[str] = None,
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
                "severity": str(severity or "normal").strip().lower() or "normal",
                "owners_before": sorted(list(set(str(x or "").strip() for x in list(owners_before or []) if str(x or "").strip()))),
                "owners_after": sorted(list(set(str(x or "").strip() for x in list(owners_after or []) if str(x or "").strip()))),
                "details": dict(details or {}),
            }
        )
        if len(rows) > limit:
            rows = rows[-limit:]
        control["claim_audit_events"] = rows

    def _append_auth_audit_event(
        self,
        control: Dict[str, Any],
        *,
        event_type: str,
        actor_key_id: Optional[str],
        target_key_id: Optional[str] = None,
        target_token_preview: Optional[str] = None,
        result: str = "ok",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        rows = list(control.get("auth_audit_events") or [])
        rows.append(
            {
                "schema_version": 1,
                "event_id": secrets.token_urlsafe(10),
                "timestamp": time.time(),
                "event_type": str(event_type or "auth_event"),
                "actor_key_id": str(actor_key_id or "") or None,
                "target_key_id": str(target_key_id or "") or None,
                "target_token_preview": str(target_token_preview or "") or None,
                "result": str(result or "ok"),
                "details": dict(details or {}),
            }
        )
        if len(rows) > 500:
            rows = rows[-500:]
        control["auth_audit_events"] = rows

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
    def _normalize_force_override_reason(reason: Optional[str]) -> str:
        return str(reason or "").strip().lower()

    def _emergency_override_predicate(
        self,
        *,
        reason: str,
        active_conflicting_owners: List[str],
        orphan_conflicting_owners: List[str],
    ) -> Optional[str]:
        if reason == "stale_owner_unreachable":
            if active_conflicting_owners:
                return "stale_owner_unreachable_requires_orphan_owner"
            if not orphan_conflicting_owners:
                return "stale_owner_unreachable_requires_orphan_owner"
            return None
        if reason in {"owner_malicious", "security_incident"}:
            if not active_conflicting_owners:
                return "emergency_reason_requires_active_conflicting_owner"
            return None
        return "force_override_emergency_reason_invalid"

    @staticmethod
    def _connectivity_mode(cfg: Dict[str, Any]) -> str:
        access_profile = dict(cfg.get("access_profile") or {})
        raw = str(access_profile.get("connectivity_mode") or "local_only").strip().lower()
        if raw in {"local_only", "ssh_tunnel_only", "truly_remote"}:
            return raw
        return "local_only"

    def _requires_ssh_binding(self, cfg: Dict[str, Any]) -> bool:
        return self._connectivity_mode(cfg) != "local_only"

    def _classify_connect_worker_class(
        self,
        *,
        config_path: str,
        payload: Optional[Dict[str, Any]],
    ) -> str:
        p = dict(payload or {})
        cfg: Dict[str, Any] = {}
        try:
            cfg = self._merge_default_and_selected_config(config_path)
        except Exception:
            cfg = {}

        def _norm(v: Any) -> str:
            return str(v or "").strip().lower()

        hosting_cfg = dict(cfg.get("hosting") or {}) if isinstance(cfg.get("hosting"), dict) else {}
        marker = _norm(
            cfg.get("worker_kind")
            or cfg.get("worker_type")
            or hosting_cfg.get("worker_kind")
            or hosting_cfg.get("worker_type")
        )
        if marker in {"generic", "non_model", "worker", "generic_worker"}:
            return "generic"
        if marker in {"model", "model_engine", "engine"}:
            return "model"

        worker_command = cfg.get("worker_command")
        spawn_cfg = dict(cfg.get("spawn") or {}) if isinstance(cfg.get("spawn"), dict) else {}
        spawn_command = spawn_cfg.get("command")
        if isinstance(worker_command, list) and worker_command:
            return "generic"
        if isinstance(spawn_command, list) and spawn_command:
            return "generic"

        configured_model = (
            ((cfg.get("engine_params") or {}).get("base_model_path") if isinstance(cfg.get("engine_params"), dict) else None)
            or cfg.get("base_model_path")
            or cfg.get("model")
            or cfg.get("base_model_name_or_path")
        )
        if str(p.get("model_path") or "").strip() or str(configured_model or "").strip():
            return "model"
        return "unknown"

    @staticmethod
    def _normalize_worker_profile_class(value: Optional[str]) -> str:
        v = str(value or "").strip().lower()
        if v in {"model", "generic"}:
            return v
        return "unknown"

    def _authorize_role_for_engine_profile(self, *, role: str, engine_id: str) -> None:
        eid = str(engine_id or "").strip()
        if not eid:
            return
        reg = self._find_registration(eid)
        if not isinstance(reg, dict):
            return
        profile = self._normalize_worker_profile_class(str(reg.get("worker_profile_class") or ""))
        r = str(role or "").strip().lower()
        if profile == "generic" and r in {ROLE_MODEL_USER, ROLE_MODEL_USER_WITH_MODEL_CONTROL}:
            raise PermissionError("insufficient_role")

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

    @staticmethod
    def _role_allowed_scopes(role: str) -> set[str]:
        r = str(role or "").strip().lower()
        if r == ROLE_ADMIN:
            return {"control", "config", "traffic"}
        if r == ROLE_CONFIG_EDITOR:
            return {"control", "config", "traffic"}
        if r == ROLE_WORKER_USER:
            return {"control", "traffic"}
        if r == ROLE_MODEL_USER_WITH_MODEL_CONTROL:
            return {"traffic"}
        if r == ROLE_MODEL_USER:
            return {"traffic"}
        if r == ROLE_DIAGNOSTIC_USER:
            return {"control"}
        return set()

    @staticmethod
    def _commands_allowed_for_role(role: str) -> set[str]:
        r = str(role or "").strip().lower()
        all_non_bootstrap = {
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
            "sandbox-fs-list",
            "sandbox-fs-read-text",
            "sandbox-fs-write-text",
            "sandbox-fs-mkdir",
            "sandbox-fs-stat",
            "sandbox-http-fetch",
            "toolbox-describe",
            "toolbox-gate",
            "toolbox-execute",
            "toolbox-cancel",
            "toolbox-gc",
            "toolbox-references",
            "toolbox-consistency",
            "toolbox-review-snapshot",
            "toolbox-repair",
            "toolbox-reconcile",
            "get-control-config",
            "set-control-config",
            "auth-status",
            "auth-list-keys",
            "auth-list-sessions",
            "auth-list-issued-tokens",
            "auth-audit-list",
            "auth-upsert-key",
            "auth-revoke-key",
            "auth-issue-session",
            "auth-begin-challenge",
            "auth-complete-challenge",
            "auth-revoke-session",
            "proxy-request",
            "proxy-rpc-call",
            "proxy-rpc-open",
            "proxy-rpc-send",
            "proxy-rpc-recv",
            "proxy-rpc-close",
            "proxy-stream-open",
            "proxy-stream-send",
            "proxy-stream-recv",
            "proxy-stream-close",
            "host-metrics",
            "list-configs",
            "create-config",
            "models-from-config",
            "connect-from-config",
            "op-start",
            "op-status",
            "set-endpoint-mode-override",
            "get-endpoint-mode-effective",
            "get-lifecycle-policy-effective",
        }
        if r == ROLE_ADMIN:
            return all_non_bootstrap
        if r == ROLE_CONFIG_EDITOR:
            return {
                "discover-running",
                "spawn",
                "get-registration",
                "shutdown",
                "ensure-running",
                "remove-registration",
                "claim-engine",
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
                "sandbox-fs-list",
                "sandbox-fs-read-text",
                "sandbox-fs-write-text",
                "sandbox-fs-mkdir",
                "sandbox-fs-stat",
                "sandbox-http-fetch",
                "toolbox-describe",
                "toolbox-gate",
                "toolbox-execute",
                "toolbox-cancel",
                "toolbox-gc",
                "toolbox-references",
                "toolbox-consistency",
                "toolbox-review-snapshot",
                "toolbox-repair",
                "toolbox-reconcile",
                "toolbox-register-auto",
                "toolbox-unregister-auto",
                "toolbox-register-intrinsics",
                "toolbox-unregister-intrinsics",
                "toolbox-register-manual",
                "toolbox-unregister-manual",
                "toolbox-environment-list",
                "toolbox-environment-upsert",
                "toolbox-environment-clone",
                "toolbox-environment-resolve",
                "toolbox-environment-apply",
                "toolbox-environment-realize",
                "toolbox-environment-sync",
                "toolbox-environment-prepare-install",
                "toolbox-environment-lock-install",
                "toolbox-environment-resolve-install-lock",
                "toolbox-environment-verify-install-lock",
                "toolbox-environment-verify-install-receipt",
                "toolbox-environment-execute-install",
                "host-metrics",
                "list-configs",
                "create-config",
                "models-from-config",
                "connect-from-config",
                "get-control-config",
                "get-lifecycle-policy-effective",
                "auth-status",
            }
        if r == ROLE_WORKER_USER:
            return {
                "discover-running",
                "spawn",
                "get-registration",
                "shutdown",
                "ensure-running",
                "remove-registration",
                "claim-engine",
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
                "sandbox-fs-list",
                "sandbox-fs-read-text",
                "sandbox-fs-write-text",
                "sandbox-fs-mkdir",
                "sandbox-fs-stat",
                "sandbox-http-fetch",
                "toolbox-describe",
                "toolbox-gate",
                "toolbox-execute",
                "toolbox-cancel",
                "toolbox-gc",
                "toolbox-references",
                "toolbox-consistency",
                "toolbox-review-snapshot",
                "toolbox-repair",
                "toolbox-reconcile",
                "toolbox-register-auto",
                "toolbox-unregister-auto",
                "toolbox-register-intrinsics",
                "toolbox-unregister-intrinsics",
                "toolbox-register-manual",
                "toolbox-unregister-manual",
                "toolbox-environment-list",
                "toolbox-environment-upsert",
                "toolbox-environment-clone",
                "toolbox-environment-resolve",
                "toolbox-environment-apply",
                "toolbox-environment-realize",
                "toolbox-environment-sync",
                "toolbox-environment-prepare-install",
                "toolbox-environment-lock-install",
                "toolbox-environment-resolve-install-lock",
                "toolbox-environment-verify-install-lock",
                "toolbox-environment-verify-install-receipt",
                "toolbox-environment-execute-install",
                "host-metrics",
                "models-from-config",
                "connect-from-config",
                "proxy-request",
                "proxy-rpc-call",
                "proxy-rpc-open",
                "proxy-rpc-send",
                "proxy-rpc-recv",
                "proxy-rpc-close",
                "proxy-stream-open",
                "proxy-stream-send",
                "proxy-stream-recv",
                "proxy-stream-close",
                "auth-status",
            }
        if r == ROLE_MODEL_USER_WITH_MODEL_CONTROL:
            return {
                "models-from-config",
                "connect-from-config",
                "proxy-request",
                "proxy-rpc-call",
                "proxy-rpc-open",
                "proxy-rpc-send",
                "proxy-rpc-recv",
                "proxy-rpc-close",
                "proxy-stream-open",
                "proxy-stream-send",
                "proxy-stream-recv",
                "proxy-stream-close",
                "auth-status",
            }
        if r == ROLE_MODEL_USER:
            return {
                "proxy-request",
                "proxy-rpc-call",
                "proxy-rpc-open",
                "proxy-rpc-send",
                "proxy-rpc-recv",
                "proxy-rpc-close",
                "proxy-stream-open",
                "proxy-stream-send",
                "proxy-stream-recv",
                "proxy-stream-close",
                "auth-status",
            }
        if r == ROLE_DIAGNOSTIC_USER:
            return {
                "discover-running",
                "get-registration",
                "claim-status",
                "resource-claim-status",
                "inspect-capabilities",
                "logs-tail",
                "logs-follow",
                "sandbox-fs-list",
                "sandbox-fs-read-text",
                "sandbox-fs-write-text",
                "sandbox-fs-mkdir",
                "sandbox-fs-stat",
                "sandbox-http-fetch",
                "toolbox-describe",
                "toolbox-gate",
                "toolbox-cancel",
                "toolbox-gc",
                "toolbox-references",
                "toolbox-consistency",
                "toolbox-review-snapshot",
                "toolbox-repair",
                "toolbox-reconcile",
                "toolbox-register-auto",
                "toolbox-unregister-auto",
                "toolbox-register-intrinsics",
                "toolbox-unregister-intrinsics",
                "toolbox-register-manual",
                "toolbox-unregister-manual",
                "toolbox-environment-list",
                "toolbox-environment-upsert",
                "toolbox-environment-clone",
                "toolbox-environment-resolve",
                "toolbox-environment-apply",
                "toolbox-environment-realize",
                "toolbox-environment-sync",
                "toolbox-environment-prepare-install",
                "toolbox-environment-lock-install",
                "toolbox-environment-resolve-install-lock",
                "toolbox-environment-verify-install-lock",
                "toolbox-environment-verify-install-receipt",
                "toolbox-environment-execute-install",
                "host-metrics",
                "get-control-config",
                "get-lifecycle-policy-effective",
                "auth-status",
            }
        return set()

    def _authorize_role_for_command(self, *, role: str, cmd: str) -> None:
        allowed = self._commands_allowed_for_role(role)
        if str(cmd or "").strip() not in allowed:
            raise PermissionError("insufficient_role")

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
        if self._requires_ssh_binding(cfg):
            expected_binding = dict(session.get("ssh_binding") or {})
            if not expected_binding:
                raise PermissionError("ssh_binding_required_for_remote_connectivity")
            presented = dict(presented_ssh_binding or {})
            if not presented:
                raise PermissionError("ssh_binding_required_for_remote_connectivity")
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
        if key_role == ROLE_ADMIN:
            return session
        if key_role not in VALID_AUTH_ROLES:
            raise PermissionError("invalid_role")
        allowed_scopes = self._role_allowed_scopes(key_role)
        if required_scope not in allowed_scopes:
            raise PermissionError("insufficient_scope")
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
            "daemon_version": DAEMON_VERSION,
            "capabilities": self.daemon_capabilities(),
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
            row = {
                "key_id": str(key_id),
                "role": str(m.get("role") or ""),
                "disabled": bool(m.get("disabled", False)),
                "auth_method": str(m.get("auth_method") or "shared_secret"),
                "allowed_configs": list(m.get("allowed_configs") or []),
                "allowed_engines": list(m.get("allowed_engines") or []),
            }
            if "created_at" in m:
                row["created_at"] = float(m.get("created_at") or 0.0)
            if "updated_at" in m:
                row["updated_at"] = float(m.get("updated_at") or 0.0)
            out.append(row)
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

    def auth_list_audit_events(
        self,
        *,
        event_type: Optional[str] = None,
        actor_key_id: Optional[str] = None,
        target_key_id: Optional[str] = None,
        result: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        control = self._read_control()
        now = time.time()
        rows: List[Dict[str, Any]] = []
        event_filter = str(event_type or "").strip().lower()
        actor_filter = str(actor_key_id or "").strip()
        target_filter = str(target_key_id or "").strip()
        result_filter = str(result or "").strip().lower()
        for item in list(control.get("auth_audit_events") or []):
            row = dict(item or {})
            ev = str(row.get("event_type") or "").strip().lower()
            actor = str(row.get("actor_key_id") or "").strip()
            target = str(row.get("target_key_id") or "").strip()
            res = str(row.get("result") or "").strip().lower()
            if event_filter and ev != event_filter:
                continue
            if actor_filter and actor != actor_filter:
                continue
            if target_filter and target != target_filter:
                continue
            if result_filter and res != result_filter:
                continue
            rows.append(row)
        rows.sort(
            key=lambda x: (
                float(x.get("timestamp") or 0.0),
                str(x.get("event_type") or ""),
                str(x.get("target_key_id") or ""),
            ),
            reverse=True,
        )
        total = len(rows)
        page_offset = max(0, int(offset or 0))
        page_limit = max(1, min(int(limit or 100), 1000))
        page = rows[page_offset: page_offset + page_limit]
        next_offset = page_offset + len(page)
        return {
            "timestamp": now,
            "total_count": total,
            "offset": page_offset,
            "limit": page_limit,
            "count": len(page),
            "has_more": bool(next_offset < total),
            "next_offset": next_offset if next_offset < total else None,
            "events": page,
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
        if role_norm not in VALID_AUTH_ROLES:
            raise ValueError(
                "role must be one of: "
                + ", ".join(sorted(VALID_AUTH_ROLES))
            )
        if method not in {"shared_secret", "public_key"}:
            raise ValueError("auth_method must be 'shared_secret' or 'public_key'")
        if role_norm == ROLE_TRANSPORT and method != "public_key":
            raise ValueError("transport role requires public_key auth_method")
        if method == "shared_secret" and not secret:
            raise ValueError("key_secret is required for shared_secret auth_method")
        if method == "public_key" and not pubkey:
            raise ValueError("public_key is required for public_key auth_method")
        normalized_allowed: List[str] = []
        normalized_engines: List[str] = []
        if role_norm == ROLE_CONFIG_EDITOR:
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
        if role_norm in {
            ROLE_WORKER_USER,
            ROLE_MODEL_USER_WITH_MODEL_CONTROL,
            ROLE_MODEL_USER,
        }:
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
        existing = dict(keys.get(kid) or {})
        preserved = {
            str(k): v
            for k, v in existing.items()
            if str(k)
            not in {
                "role",
                "auth_method",
                "secret_hash",
                "public_key",
                "disabled",
                "allowed_configs",
                "allowed_engines",
                "created_at",
                "updated_at",
            }
        }
        keys[kid] = preserved | {
            "role": role_norm,
            "auth_method": method,
            "secret_hash": self._hash_secret(secret) if method == "shared_secret" else "",
            "public_key": pubkey if method == "public_key" else "",
            "disabled": bool(disabled),
            "allowed_configs": normalized_allowed,
            "allowed_engines": normalized_engines,
        }
        auth["keys"] = keys
        cfg["auth"] = auth
        control["control_config"] = cfg
        self._append_auth_audit_event(
            control,
            event_type="auth_upsert_key",
            actor_key_id=None,
            target_key_id=kid,
            result="ok",
            details={"role": role_norm, "auth_method": method, "disabled": bool(disabled)},
        )
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
        self._append_auth_audit_event(
            control,
            event_type="auth_revoke_key",
            actor_key_id=None,
            target_key_id=kid,
            result="ok" if bool(existed) else "not_found",
            details={"revoked_sessions": revoked_sessions},
        )
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
        progress_events: List[Dict[str, Any]] = [
            self._progress_event("bootstrap_handshake.validate_key", "running", "Validating credentials"),
        ]
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
        if not bool(cfg.get("require_auth", False)):
            raise PermissionError("require_auth_disabled_disallows_session_commands")
        # Shared-secret bootstrap is local-only. Remote-capable profiles must use
        # public-key challenge flow for session issuance.
        if self._requires_ssh_binding(cfg):
            raise PermissionError("shared_secret_bootstrap_not_supported_for_remote_connectivity")
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
        out = self._issue_session_for_key(
            key_id=kid,
            key_meta=key_meta,
            scope=scope_norm,
            ttl_seconds=ttl_seconds,
            config_paths=config_paths,
            engine_ids=engine_ids,
            ssh_binding=ssh_binding,
            control=control,
        )
        progress_events.append(
            self._progress_event("bootstrap_handshake.issue_session", "completed", "Session issued")
        )
        if isinstance(out, dict):
            out.setdefault("stage", "completed")
            out.setdefault("progress_events", progress_events)
        return out

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
        if not bool(cfg.get("require_auth", False)):
            raise PermissionError("require_auth_disabled_disallows_session_commands")
        auth = dict(cfg.get("auth") or {})
        self._prune_expired_sessions(auth)
        role = str(key_meta.get("role") or "").strip().lower()
        if role not in VALID_AUTH_ROLES:
            raise PermissionError("invalid_role")
        if role == ROLE_TRANSPORT:
            raise PermissionError("transport_role_cannot_issue_session")
        if scope_norm not in self._role_allowed_scopes(role):
            raise PermissionError("role_cannot_issue_requested_scope")
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
        progress_events: List[Dict[str, Any]] = [
            self._progress_event("bootstrap_handshake.challenge_prepare", "running", "Preparing challenge"),
        ]
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
        if not bool(cfg.get("require_auth", False)):
            self._metrics_challenge_event(
                event="begin_failed",
                key_id=kid,
                reason="require_auth_disabled_disallows_session_commands",
            )
            raise PermissionError("require_auth_disabled_disallows_session_commands")
        if self._requires_ssh_binding(cfg) and not dict(ssh_binding or {}):
            self._metrics_challenge_event(event="begin_failed", key_id=kid, reason="ssh_binding_required_for_remote_connectivity")
            raise PermissionError("ssh_binding_required_for_remote_connectivity")
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
        if role not in VALID_AUTH_ROLES:
            self._metrics_challenge_event(event="begin_failed", key_id=kid, reason="invalid_role")
            raise PermissionError("invalid_role")
        if role == ROLE_TRANSPORT:
            self._metrics_challenge_event(event="begin_failed", key_id=kid, reason="transport_role_cannot_issue_session")
            raise PermissionError("transport_role_cannot_issue_session")
        if scope_norm not in self._role_allowed_scopes(role):
            self._metrics_challenge_event(event="begin_failed", key_id=kid, reason="role_cannot_issue_requested_scope")
            raise PermissionError("role_cannot_issue_requested_scope")
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
        out = {
            "status": "ok",
            "challenge_id": challenge_id,
            "key_id": kid,
            "scope": scope_norm,
            "challenge": challenge_text,
            "expires_at": expires_at,
            "ttl_seconds": ttl,
            "stage": "challenge_issued",
        }
        progress_events.append(
            self._progress_event("bootstrap_handshake.challenge_issued", "completed", "Challenge issued")
        )
        out["progress_events"] = progress_events
        return out

    def auth_complete_challenge(
        self,
        *,
        challenge_id: str,
        signature_ssh: str,
        presented_ssh_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        progress_events: List[Dict[str, Any]] = [
            self._progress_event("bootstrap_handshake.verify_signature", "running", "Verifying challenge signature"),
        ]
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
        if not bool(cfg.get("require_auth", False)):
            self._metrics_challenge_event(
                event="complete_failed",
                challenge_id=cid,
                reason="require_auth_disabled_disallows_session_commands",
            )
            raise PermissionError("require_auth_disabled_disallows_session_commands")
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
        if role not in VALID_AUTH_ROLES or role == ROLE_TRANSPORT:
            self._metrics_challenge_event(event="complete_failed", key_id=key_id, challenge_id=cid, reason="invalid_role")
            progress_events.append(
                self._progress_event("bootstrap_handshake.issue_session", "failed", "Role is not allowed")
            )
            return {"status": "denied", "stage": "failed", "progress_events": progress_events}
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
        progress_events.append(
            self._progress_event("bootstrap_handshake.issue_session", "completed", "Session issued")
        )
        if isinstance(out, dict):
            out.setdefault("stage", "completed")
            out.setdefault("progress_events", progress_events)
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
        self._append_auth_audit_event(
            control,
            event_type="auth_revoke_session",
            actor_key_id=None,
            target_token_preview=self._token_preview(tok),
            result="ok" if bool(existed) else "not_found",
            details={},
        )
        self._write_control(control)
        return {"token": tok, "revoked": bool(existed)}

    def reset_hosting_access(self) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        keys = dict(auth.get("keys") or {})
        sessions = dict(auth.get("sessions") or {})
        challenges = dict(auth.get("challenges") or {})
        cfg["auth"] = {"keys": {}, "sessions": {}, "challenges": {}}
        control["control_config"] = cfg
        self._append_auth_audit_event(
            control,
            event_type="auth_reset_local_helper",
            actor_key_id=None,
            target_key_id=None,
            result="ok",
            details={
                "cleared_keys": len(keys),
                "cleared_sessions": len(sessions),
                "cleared_challenges": len(challenges),
            },
        )
        self._write_control(control)
        return {
            "status": "ok",
            "cleared_keys": len(keys),
            "cleared_sessions": len(sessions),
            "cleared_challenges": len(challenges),
            "require_auth": bool(cfg.get("require_auth", False)),
            "control_state_file": str(self.control_state_file),
        }

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
            try:
                self._validate_require_auth_disabled_safe_profile(cfg)
            except PermissionError as exc:
                self._metrics_auth_denied(str(exc))
                raise
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
            "sandbox-fs-list",
            "sandbox-fs-read-text",
            "sandbox-fs-write-text",
            "sandbox-fs-mkdir",
            "sandbox-fs-stat",
            "sandbox-http-fetch",
            "get-control-config",
            "set-control-config",
            "auth-upsert-key",
            "auth-revoke-key",
            "auth-list-keys",
            "auth-list-sessions",
            "auth-list-issued-tokens",
            "auth-audit-list",
            "auth-revoke-session",
            "host-metrics",
            "op-start",
            "op-status",
            "set-endpoint-mode-override",
            "get-endpoint-mode-effective",
            "get-lifecycle-policy-effective",
        }:
            try:
                session = self._validate_session(
                    control,
                    token,
                    required_scope="control",
                    presented_ssh_binding=presented_ssh_binding,
                )
                self._authorize_role_for_command(
                    role=str((session or {}).get("role") or ""),
                    cmd=c,
                )
            except PermissionError as exc:
                self._metrics_auth_denied(str(exc))
                raise
            return
        if c in {"proxy-request", "sandbox-http-fetch"}:
            p = dict(payload or {})
            requested_engine = str(p.get("engine_id") or "").strip()
            try:
                session = self._validate_session(
                    control,
                    token,
                    required_scope="traffic",
                    requested_engine=requested_engine,
                    presented_ssh_binding=presented_ssh_binding,
                )
                self._authorize_role_for_command(
                    role=str((session or {}).get("role") or ""),
                    cmd=c,
                )
                self._authorize_role_for_engine_profile(
                    role=str((session or {}).get("role") or ""),
                    engine_id=requested_engine,
                )
            except PermissionError as exc:
                self._metrics_auth_denied(str(exc))
                raise
            return
        if c in {
            "proxy-rpc-call",
            "proxy-rpc-open",
            "proxy-rpc-send",
            "proxy-rpc-recv",
            "proxy-rpc-close",
            "proxy-stream-open",
            "proxy-stream-send",
            "proxy-stream-recv",
            "proxy-stream-close",
        }:
            p = dict(payload or {})
            requested_engine = str(p.get("engine_id") or "").strip()
            if c in {
                "proxy-rpc-call",
                "proxy-rpc-open",
                "proxy-rpc-send",
                "proxy-rpc-recv",
                "proxy-rpc-close",
                "proxy-stream-open",
                "proxy-stream-send",
                "proxy-stream-recv",
                "proxy-stream-close",
            } and not requested_engine:
                self._metrics_auth_denied("engine_id_required")
                raise PermissionError("engine_id_required")
            try:
                session = self._validate_session(
                    control,
                    token,
                    required_scope="traffic",
                    requested_engine=requested_engine,
                    presented_ssh_binding=presented_ssh_binding,
                )
                self._authorize_role_for_command(
                    role=str((session or {}).get("role") or ""),
                    cmd=c,
                )
            except PermissionError as exc:
                self._metrics_auth_denied(str(exc))
                raise
            return
        if c in {"list-configs", "create-config"}:
            requested_config = None
            p = dict(payload or {})
            try:
                session = self._validate_session(
                    control,
                    token,
                    required_scope="config",
                    requested_config=requested_config,
                    presented_ssh_binding=presented_ssh_binding,
                )
                self._authorize_role_for_command(
                    role=str((session or {}).get("role") or ""),
                    cmd=c,
                )
            except PermissionError as exc:
                self._metrics_auth_denied(str(exc))
                raise
            return
        if c in {"models-from-config", "connect-from-config"}:
            p = dict(payload or {})
            requested_config = self._normalize_config_selector(str(p.get("config_path") or "default"))
            session = None
            # Prefer config-scope session; allow traffic-scope for model-oriented roles.
            for required_scope in ("config", "traffic"):
                try:
                    session = self._validate_session(
                        control,
                        token,
                        required_scope=required_scope,
                        requested_config=requested_config if required_scope == "config" else None,
                        requested_engine=str(p.get("engine_id") or "").strip() if required_scope == "traffic" else None,
                        presented_ssh_binding=presented_ssh_binding,
                    )
                    break
                except PermissionError:
                    session = None
            if not isinstance(session, dict):
                self._metrics_auth_denied("insufficient_scope")
                raise PermissionError("insufficient_scope")
            try:
                self._authorize_role_for_command(
                    role=str((session or {}).get("role") or ""),
                    cmd=c,
                )
            except PermissionError as exc:
                self._metrics_auth_denied(str(exc))
                raise
            if c == "connect-from-config":
                requested_model_override = str(p.get("model_path") or "").strip()
                role = str((session or {}).get("role") or "").strip().lower()
                can_override_model = role in {
                    ROLE_ADMIN,
                    ROLE_CONFIG_EDITOR,
                    ROLE_WORKER_USER,
                    ROLE_MODEL_USER_WITH_MODEL_CONTROL,
                }
                if requested_model_override and not can_override_model:
                    self._metrics_auth_denied("insufficient_role")
                    raise PermissionError("insufficient_role")
                worker_class = self._classify_connect_worker_class(
                    config_path=requested_config,
                    payload=p,
                )
                can_use_generic_worker = role in {
                    ROLE_ADMIN,
                    ROLE_CONFIG_EDITOR,
                    ROLE_WORKER_USER,
                }
                if worker_class == "generic" and not can_use_generic_worker:
                    self._metrics_auth_denied("insufficient_role")
                    raise PermissionError("insufficient_role")
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
        owner_notice = self._get_ownership_change_notice(control, actor_id)
        if owner_notice and c not in claim_cmds:
            return {
                "ok": False,
                "error": "access_denied",
                "error_code": "ownership_changed_reclaim_required",
                "error_details": {
                    "command": c,
                    "actor_id": actor_id,
                    "notice": owner_notice,
                },
                "payload": p,
            }
        control_cfg = dict(control.get("control_config") or {})
        require_auth_enabled = bool(control_cfg.get("require_auth", False))
        endpoint_mode_default = self._endpoint_mode_default(control_cfg)
        if c in claim_cmds:
            if not require_auth_enabled:
                # No-auth mode is intentionally single-client safe.
                p["exclusive"] = True
            elif "exclusive" not in p:
                p["exclusive"] = bool(endpoint_mode_default == "exclusive")
        sensitive_engine_cmds = {
            "spawn",
            "get-registration",
            "shutdown",
            "ensure-running",
            "remove-registration",
            "logs-tail",
            "logs-follow",
            "sandbox-fs-list",
            "sandbox-fs-read-text",
            "sandbox-fs-write-text",
            "sandbox-fs-mkdir",
            "sandbox-fs-stat",
            "sandbox-http-fetch",
            "inspect-capabilities",
            "issue-token",
            "issue-resource-token",
            "proxy-rpc-call",
            "proxy-rpc-open",
            "proxy-rpc-send",
            "proxy-rpc-recv",
            "proxy-rpc-close",
            "proxy-stream-open",
            "proxy-stream-send",
            "proxy-stream-recv",
            "proxy-stream-close",
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
            reason = self._normalize_force_override_reason(p.get("force_override_reason"))
            if reason not in VALID_FORCE_OVERRIDE_REASONS:
                return {
                    "ok": False,
                    "error": "access_denied",
                    "error_code": "force_override_reason_required",
                    "error_details": {
                        "command": c,
                        "actor_id": actor_id,
                        "allowed_reasons": sorted(list(VALID_FORCE_OVERRIDE_REASONS)),
                    },
                    "payload": p,
                }
            emergency = bool(p.get("force_override_emergency", False))
            if emergency and reason not in EMERGENCY_FORCE_OVERRIDE_REASONS:
                return {
                    "ok": False,
                    "error": "access_denied",
                    "error_code": "force_override_emergency_reason_invalid",
                    "error_details": {
                        "command": c,
                        "actor_id": actor_id,
                        "allowed_emergency_reasons": sorted(list(EMERGENCY_FORCE_OVERRIDE_REASONS)),
                    },
                    "payload": p,
                }
            if emergency:
                active_conflicting_owners: List[str] = []
                orphan_conflicting_owners: List[str] = []
                if c == "claim-endpoint":
                    endpoint = dict(control.get("endpoint_claim") or {})
                    owners = [str(x or "").strip() for x in list(endpoint.get("owners") or []) if str(x or "").strip()]
                    active_owners, orphan_owners = self._active_and_orphan_owners(control, owners)
                    active_conflicting_owners = sorted([o for o in active_owners if o != actor_id])
                    orphan_conflicting_owners = sorted([o for o in orphan_owners if o != actor_id])
                elif c == "claim-engine":
                    engine_id = str(p.get("engine_id") or "").strip()
                    claim = dict((control.get("claims_by_engine") or {}).get(engine_id) or {})
                    owners = [str(x or "").strip() for x in list(claim.get("owners") or []) if str(x or "").strip()]
                    active_owners, orphan_owners = self._active_and_orphan_owners(control, owners)
                    active_conflicting_owners = sorted([o for o in active_owners if o != actor_id])
                    orphan_conflicting_owners = sorted([o for o in orphan_owners if o != actor_id])
                elif c == "claim-resource":
                    rkind = str(p.get("resource_kind") or "").strip().lower()
                    rid = str(p.get("resource_id") or "").strip()
                    if rkind == "engine":
                        claim = dict((control.get("claims_by_engine") or {}).get(rid) or {})
                    else:
                        claim = dict((control.get("resource_claims") or {}).get(self._resource_key(rkind, rid)) or {})
                    owners = [str(x or "").strip() for x in list(claim.get("owners") or []) if str(x or "").strip()]
                    active_owners, orphan_owners = self._active_and_orphan_owners(control, owners)
                    active_conflicting_owners = sorted([o for o in active_owners if o != actor_id])
                    orphan_conflicting_owners = sorted([o for o in orphan_owners if o != actor_id])
                predicate = self._emergency_override_predicate(
                    reason=reason,
                    active_conflicting_owners=active_conflicting_owners,
                    orphan_conflicting_owners=orphan_conflicting_owners,
                )
                if predicate:
                    return {
                        "ok": False,
                        "error": "access_denied",
                        "error_code": "force_override_emergency_predicate_not_met",
                        "error_details": {
                            "command": c,
                            "actor_id": actor_id,
                            "reason": reason,
                            "predicate": predicate,
                            "active_conflicting_owners": active_conflicting_owners,
                            "orphan_conflicting_owners": orphan_conflicting_owners,
                        },
                        "payload": p,
                    }
            if emergency:
                p["force_override_reason"] = reason
                p["force_override_emergency"] = True
                return {"ok": True, "payload": p}
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
            p["force_override_reason"] = reason
            p["force_override_emergency"] = False
        if c in claim_cmds and bool(p.get("force_override", False)) and (not is_localhost):
            reason = self._normalize_force_override_reason(p.get("force_override_reason"))
            if reason not in VALID_FORCE_OVERRIDE_REASONS:
                return {
                    "ok": False,
                    "error": "access_denied",
                    "error_code": "force_override_reason_required",
                    "error_details": {
                        "command": c,
                        "actor_id": actor_id,
                        "allowed_reasons": sorted(list(VALID_FORCE_OVERRIDE_REASONS)),
                    },
                    "payload": p,
                }
            emergency = bool(p.get("force_override_emergency", False))
            if emergency and reason not in EMERGENCY_FORCE_OVERRIDE_REASONS:
                return {
                    "ok": False,
                    "error": "access_denied",
                    "error_code": "force_override_emergency_reason_invalid",
                    "error_details": {
                        "command": c,
                        "actor_id": actor_id,
                        "allowed_emergency_reasons": sorted(list(EMERGENCY_FORCE_OVERRIDE_REASONS)),
                    },
                    "payload": p,
                }
            if emergency:
                active_conflicting_owners = []
                orphan_conflicting_owners = []
                if c == "claim-endpoint":
                    endpoint = dict(control.get("endpoint_claim") or {})
                    owners = [str(x or "").strip() for x in list(endpoint.get("owners") or []) if str(x or "").strip()]
                    active_owners, orphan_owners = self._active_and_orphan_owners(control, owners)
                    active_conflicting_owners = sorted([o for o in active_owners if o != actor_id])
                    orphan_conflicting_owners = sorted([o for o in orphan_owners if o != actor_id])
                elif c == "claim-engine":
                    engine_id = str(p.get("engine_id") or "").strip()
                    claim = dict((control.get("claims_by_engine") or {}).get(engine_id) or {})
                    owners = [str(x or "").strip() for x in list(claim.get("owners") or []) if str(x or "").strip()]
                    active_owners, orphan_owners = self._active_and_orphan_owners(control, owners)
                    active_conflicting_owners = sorted([o for o in active_owners if o != actor_id])
                    orphan_conflicting_owners = sorted([o for o in orphan_owners if o != actor_id])
                elif c == "claim-resource":
                    rkind = str(p.get("resource_kind") or "").strip().lower()
                    rid = str(p.get("resource_id") or "").strip()
                    if rkind == "engine":
                        claim = dict((control.get("claims_by_engine") or {}).get(rid) or {})
                    else:
                        claim = dict((control.get("resource_claims") or {}).get(self._resource_key(rkind, rid)) or {})
                    owners = [str(x or "").strip() for x in list(claim.get("owners") or []) if str(x or "").strip()]
                    active_owners, orphan_owners = self._active_and_orphan_owners(control, owners)
                    active_conflicting_owners = sorted([o for o in active_owners if o != actor_id])
                    orphan_conflicting_owners = sorted([o for o in orphan_owners if o != actor_id])
                predicate = self._emergency_override_predicate(
                    reason=reason,
                    active_conflicting_owners=active_conflicting_owners,
                    orphan_conflicting_owners=orphan_conflicting_owners,
                )
                if predicate:
                    return {
                        "ok": False,
                        "error": "access_denied",
                        "error_code": "force_override_emergency_predicate_not_met",
                        "error_details": {
                            "command": c,
                            "actor_id": actor_id,
                            "reason": reason,
                            "predicate": predicate,
                            "active_conflicting_owners": active_conflicting_owners,
                            "orphan_conflicting_owners": orphan_conflicting_owners,
                        },
                        "payload": p,
                    }
            p["force_override_reason"] = reason
            p["force_override_emergency"] = emergency

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
        engine_python = self._engine_python_executable()
        engine_runtime_ok, _engine_runtime_err = self._check_module_discoverable(engine_python, "mp13_engine")
        def _config_meta(path_str: str) -> Dict[str, Any]:
            try:
                _ = self._merge_default_and_selected_config(path_str)
            except Exception as e:
                return {"has_spawn_command": False, "connect_reason": f"invalid_config: {e}"}
            return {
                "has_spawn_command": bool(engine_runtime_ok),
                "connect_reason": None if engine_runtime_ok else "engine_not_available",
            }
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
        Check whether engine runtime symbols are importable by *python*.

        Runs a tiny subprocess so it works even when the calling process lives
        in a different venv (e.g. the docs venv checking the engine venv).
        Returns (True, "") on success, (False, reason) on failure.
        """
        try:
            result = subprocess.run(  # noqa: S603
                [python, "-c", f"from {module_name} import MP13Engine"],
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
    def _check_module_discoverable(python: str, module_name: str) -> Tuple[bool, str]:
        """
        Lightweight module check for UX surfaces (e.g., list-configs).

        Uses importlib.find_spec instead of importing heavy module trees.
        """
        probe = (
            "import importlib.util, sys; "
            f"sys.exit(0 if importlib.util.find_spec({module_name!r}) else 1)"
        )
        try:
            result = subprocess.run(  # noqa: S603
                [python, "-c", probe],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                return True, ""
            stderr = (result.stderr or "").strip()
            last_line = stderr.splitlines()[-1] if stderr else "module not discoverable"
            return False, last_line
        except FileNotFoundError:
            return False, f"Python executable not found: {python}"
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def _engine_python_executable() -> str:
        python = os.environ.get("MP13_ENGINE_PYTHON", "").strip()
        return python or sys.executable

    @staticmethod
    def _allocate_ipc_address(engine_id: str) -> Tuple[str, str]:
        raw_engine = str(engine_id or "engine")
        safe_engine = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_engine).strip("_") or "engine"
        nonce = secrets.token_hex(6)
        if os.name == "nt":
            return "AF_PIPE", f"\\\\.\\pipe\\mp13-host-{safe_engine}-{nonce}"
        base = posixpath.abspath(posixpath.expanduser(str(tempfile.gettempdir() or "/tmp")))
        engine_hash = hashlib.sha256(raw_engine.encode("utf-8", errors="ignore")).hexdigest()[:12]
        short_engine = safe_engine[:24].rstrip("_-") or "engine"
        filename = f"mp13-host-{short_engine}-{engine_hash}-{nonce}.sock"
        return "AF_UNIX", posixpath.join(base, filename)

    @staticmethod
    def _parse_worker_authkey_token(token: Optional[str]) -> bytes:
        raw = str(token or "").strip()
        if not raw:
            return b""
        return raw.encode("utf-8", errors="ignore")

    def _proxy_request_via_ipc(
        self,
        *,
        reg: Dict[str, Any],
        engine_id: str,
        method: str,
        path: str,
        query: str,
        headers: Dict[str, str],
        body_b64: str,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        family = str(reg.get("worker_ipc_family") or "").strip()
        address = str(reg.get("worker_ipc_address") or "").strip()
        auth_token = str(reg.get("worker_auth_token") or "").strip()
        endpoint = str(reg.get("endpoint") or "").strip() or "ipc://local"
        if not family or not address:
            raise ValueError("engine ipc endpoint is not registered")
        authkey = self._parse_worker_authkey_token(auth_token)
        payload = {
            "kind": "http_request",
            "engine_id": str(engine_id or "").strip(),
            "method": str(method or "GET").strip().upper(),
            "path": str(path or "/").strip() or "/",
            "query": str(query or ""),
            "headers": dict(headers or {}),
            "body_b64": str(body_b64 or ""),
        }
        conn = None
        try:
            conn = MPClient(address=address, family=family, authkey=authkey)
            conn.send(payload)
            if not conn.poll(max(0.1, float(timeout_seconds or 30.0))):
                raise TimeoutError("ipc worker timeout")
            resp = conn.recv()
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        if not isinstance(resp, dict):
            raise RuntimeError("invalid ipc worker response")
        if str(resp.get("status") or "").strip().lower() == "error":
            msg = str(resp.get("message") or "ipc worker error")
            raise RuntimeError(msg)
        status_code = int(resp.get("status_code") or 500)
        out_headers = dict(resp.get("headers") or {})
        out_body_b64 = str(resp.get("body_b64") or "")
        return {
            "engine_id": str(engine_id),
            "endpoint": endpoint,
            "url": f"ipc://{engine_id}{path}",
            "status_code": status_code,
            "headers": out_headers,
            "body_b64": out_body_b64,
            "body_size": len(base64.b64decode(out_body_b64)) if out_body_b64 else 0,
            "truncated": False,
        }

    def _ipc_call(self, *, reg: Dict[str, Any], payload: Dict[str, Any], timeout_seconds: float = 30.0) -> Dict[str, Any]:
        family = str(reg.get("worker_ipc_family") or "").strip()
        address = str(reg.get("worker_ipc_address") or "").strip()
        auth_token = str(reg.get("worker_auth_token") or "").strip()
        if not family or not address:
            raise ValueError("engine ipc endpoint is not registered")
        authkey = self._parse_worker_authkey_token(auth_token)
        conn = None
        try:
            conn = MPClient(address=address, family=family, authkey=authkey)
            conn.send(dict(payload or {}))
            if not conn.poll(max(0.1, float(timeout_seconds or 30.0))):
                raise TimeoutError("ipc worker timeout")
            out = conn.recv()
            if not isinstance(out, dict):
                raise RuntimeError("invalid ipc response")
            return dict(out or {})
        except FileNotFoundError as exc:
            engine_id = str(reg.get("engine_id") or "").strip() or "unknown"
            raise RuntimeError(
                f"worker IPC endpoint is unavailable for engine '{engine_id}' at '{address}'; "
                "worker process may not be running"
            ) from exc
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _require_ipc_registration(self, engine_id: str, *, command_label: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        reg = self._find_registration(eid)
        if not reg:
            raise ValueError(f"engine '{eid}' is not registered")
        if str(reg.get("worker_transport") or "").strip().lower() != "ipc":
            raise ValueError(f"{command_label} is only supported for ipc transport")
        return reg

    @staticmethod
    def _registration_allowed_tool_names(reg: Dict[str, Any]) -> Optional[set[str]]:
        tool_access = dict(reg.get("tool_access") or {})
        allowed = {
            str(item or "").strip()
            for item in list(tool_access.get("allowed_tool_names") or [])
            if str(item or "").strip()
        }
        return allowed or None

    @staticmethod
    def _registration_advertised_tool_names(reg: Dict[str, Any]) -> Optional[set[str]]:
        tool_access = dict(reg.get("tool_access") or {})
        advertised = {
            str(item or "").strip()
            for item in list(tool_access.get("advertised_tool_names") or [])
            if str(item or "").strip()
        }
        return advertised or None

    @staticmethod
    def _registration_hidden_allowed_tool_names(reg: Dict[str, Any]) -> Optional[set[str]]:
        tool_access = dict(reg.get("tool_access") or {})
        hidden = {
            str(item or "").strip()
            for item in list(tool_access.get("hidden_allowed_tool_names") or [])
            if str(item or "").strip()
        }
        return hidden or None

    @staticmethod
    def _tools_view_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[ToolsView]:
        row = dict(payload or {})
        if not row:
            return None
        return ToolsView(
            view_id=str(row.get("view_id") or "").strip() or "hosted-tools-view",
            mode=str(row.get("mode") or "").strip() or "advertised",
            allowed_tools=set(str(item or "").strip() for item in list(row.get("allowed_tools") or []) if str(item or "").strip()),
            advertised_tools=set(str(item or "").strip() for item in list(row.get("advertised_tools") or []) if str(item or "").strip()),
            hidden_allowed_tools=set(
                str(item or "").strip() for item in list(row.get("hidden_allowed_tools") or []) if str(item or "").strip()
            ),
            disabled_tools=set(str(item or "").strip() for item in list(row.get("disabled_tools") or []) if str(item or "").strip()),
        )

    @staticmethod
    def _registration_tool_routes(reg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        tool_access = dict(reg.get("tool_access") or {})
        routes = dict(tool_access.get("tool_routes") or {})
        out: Dict[str, Dict[str, Any]] = {}
        for raw_name, raw_meta in routes.items():
            name = str(raw_name or "").strip()
            if not name:
                continue
            out[name] = dict(raw_meta or {})
        return out

    @staticmethod
    def _registration_toolbox_id(reg: Dict[str, Any]) -> str:
        bundle = dict(reg.get("bundle") or {})
        return str(bundle.get("toolbox_id") or bundle.get("bundle_id") or "").strip()

    def _toolbox_executor_registrations(self, toolbox_id: str) -> List[Dict[str, Any]]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            return []
        rows: List[Dict[str, Any]] = []
        for row in self._read_engines():
            reg = dict(row or {})
            if str(reg.get("executor_kind") or "").strip() != "toolbox_executor_v1":
                continue
            if self._registration_toolbox_id(reg) != tid:
                continue
            rows.append(reg)
        return rows

    def _cleanup_toolbox_bundle_root(self, reg: Dict[str, Any]) -> None:
        bundle = dict(reg.get("bundle") or {})
        raw = str(bundle.get("bundle_root") or "").strip()
        if not raw:
            return
        root = Path(raw).expanduser().resolve()
        allowed_root = (self.hosting_root / "toolbox_bundles").resolve()
        try:
            if root != allowed_root and allowed_root not in root.parents:
                return
        except Exception:
            return
        shutil.rmtree(root, ignore_errors=True)

    def _retire_toolbox_registration(self, engine_id: str) -> None:
        reg = self._find_registration(engine_id)
        if reg:
            try:
                self.shutdown(engine_id, timeout_seconds=2.0)
            except Exception:
                pass
            self.remove_registration(engine_id)
            self._cleanup_toolbox_bundle_root(reg)

    def _merge_toolbox_auto_requests(
        self,
        *,
        toolbox_id: str,
        incoming_requests: Optional[List[Dict[str, Any]]] = None,
        remove_keys: Optional[List[str]] = None,
    ) -> Tuple[List[Any], Dict[str, Any], Dict[str, Any]]:
        from .toolbox_harness import ToolboxAutoAssignmentRequest

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        persisted_requests = [
            ToolboxAutoAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(toolbox_row.get("requests") or [])
        ]
        merged: Dict[str, ToolboxAutoAssignmentRequest] = {
            req.stable_key(): req for req in persisted_requests
        }
        for row in list(incoming_requests or []):
            req = ToolboxAutoAssignmentRequest.from_runtime_dict(dict(row or {}))
            merged[req.stable_key()] = req
        for key in list(remove_keys or []):
            merged.pop(str(key or "").strip(), None)
        return list(merged.values()), state, toolboxes

    def _merge_toolbox_manual_requests(
        self,
        *,
        toolbox_id: str,
        incoming_requests: Optional[List[Dict[str, Any]]] = None,
        remove_keys: Optional[List[str]] = None,
    ) -> Tuple[List[Any], Dict[str, Any], Dict[str, Any]]:
        from .toolbox_harness import ToolboxManualAssignmentRequest

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        persisted_requests = [
            ToolboxManualAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(toolbox_row.get("manual_requests") or [])
        ]
        merged: Dict[str, ToolboxManualAssignmentRequest] = {
            req.stable_key(): req for req in persisted_requests
        }
        for row in list(incoming_requests or []):
            req = ToolboxManualAssignmentRequest.from_runtime_dict(dict(row or {}))
            merged[req.stable_key()] = req
        for key in list(remove_keys or []):
            merged.pop(str(key or "").strip(), None)
        return list(merged.values()), state, toolboxes

    @staticmethod
    def _normalize_intrinsic_tool_names(
        names: List[str],
        *,
        include_guides: bool = False,
    ) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        base_names: List[str] = []
        for item in list(names or []):
            name = str(item or "").strip()
            if not name:
                continue
            if name not in seen:
                seen.add(name)
                out.append(name)
            if not name.endswith("_guide"):
                base_names.append(name)
        if include_guides:
            for base in base_names:
                guide_name = f"{base}_guide"
                if guide_name not in seen:
                    seen.add(guide_name)
                    out.append(guide_name)
        return out

    @staticmethod
    def _registration_sandbox_profile_id(reg: Dict[str, Any]) -> str:
        return str(dict(reg.get("bundle") or {}).get("sandbox_profile_id") or "default").strip() or "default"

    def _route_toolbox_registration(self, *, toolbox_id: str, tool_name: str, command_label: str) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        name = str(tool_name or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        if not name:
            raise ValueError("tool_name is required")
        matches: List[Dict[str, Any]] = []
        for reg in self._toolbox_executor_registrations(tid):
            allowed = self._registration_allowed_tool_names(reg)
            if allowed is not None and name in allowed:
                matches.append(reg)
        if not matches:
            raise PermissionError(f"tool_not_allowed:{name}")
        if len(matches) > 1:
            raise RuntimeError(f"toolbox_route_ambiguous:{tid}:{name}")
        return self._require_toolbox_executor_registration(
            str(matches[0].get("engine_id") or ""),
            command_label=command_label,
        )

    def _require_toolbox_executor_registration(self, engine_id: str, *, command_label: str) -> Dict[str, Any]:
        reg = self._require_ipc_registration(engine_id, command_label=command_label)
        executor_kind = str(reg.get("executor_kind") or "").strip()
        if executor_kind and executor_kind != "toolbox_executor_v1":
            raise ValueError(f"{command_label} is only supported for toolbox executors")
        return reg

    def _build_engine_spawn_spec(self, *, engine_id: str, config_path: str, model_path: str) -> Dict[str, Any]:
        python = self._engine_python_executable()
        ok, err_detail = self._check_module_available(python, "mp13_engine")
        if not ok:
            return {
                "error": (
                    f"mp13_engine is not available in Python '{python}': {err_detail}. "
                    "Set MP13_ENGINE_PYTHON to a Python that has mp13_engine installed."
                ),
                "error_kind": "engine_not_available",
            }
        transport = "ipc"
        worker_auth_token = secrets.token_urlsafe(24)
        worker_auth_header = "X-MP13-Host-Token"
        ipc_family, ipc_address = self._allocate_ipc_address(engine_id)
        endpoint = "ipc://local"
        command = [
            python,
            "-m",
            "hosting.engine_worker_ipc",
            "--ipc-family",
            str(ipc_family),
            "--ipc-address",
            str(ipc_address),
        ]
        return {
            "command": command,
            "cwd": None,
            "endpoint": endpoint,
            "worker_auth_token": worker_auth_token,
            "worker_auth_header": worker_auth_header,
            "worker_transport": transport,
            "worker_ipc_family": ipc_family,
            "worker_ipc_address": ipc_address,
            "env": {
                "MP13_ENGINE_CONFIG_PATH": str(config_path),
                "MP13_ENGINE_ID": str(engine_id),
                "MP13_MODEL_PATH": str(model_path),
                "MP13_ENGINE_HOST_TOKEN": worker_auth_token,
                "MP13_ENGINE_HOST_TOKEN_HEADER": worker_auth_header,
                "MP13_ENGINE_TRANSPORT": transport,
            },
        }

    def _build_generic_spawn_spec(
        self,
        *,
        engine_id: str,
        config_path: str,
        config_payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        cfg = dict(config_payload or {})
        spawn_cfg = dict(cfg.get("spawn") or {}) if isinstance(cfg.get("spawn"), dict) else {}
        cmd_raw = cfg.get("worker_command")
        if not (isinstance(cmd_raw, list) and cmd_raw):
            cmd_raw = spawn_cfg.get("command")
        if not (isinstance(cmd_raw, list) and cmd_raw):
            return {
                "error": "generic worker config is missing worker_command/spawn.command",
                "error_kind": "generic_worker_command_missing",
            }
        command = [str(x) for x in list(cmd_raw) if str(x).strip()]
        if not command:
            return {
                "error": "generic worker command resolved to empty",
                "error_kind": "generic_worker_command_missing",
            }
        selected = Path(str(config_path or "")).expanduser().resolve()
        config_dir = selected.parent
        cwd_raw = cfg.get("worker_cwd") or spawn_cfg.get("cwd")
        cwd = None
        if str(cwd_raw or "").strip():
            try:
                cwd = str(self._resolve_path_token(str(cwd_raw), config_dir=config_dir))
            except Exception:
                cwd = str(cwd_raw)
        env: Dict[str, Any] = {}
        worker_env = cfg.get("worker_env")
        spawn_env = spawn_cfg.get("env")
        if isinstance(worker_env, dict):
            env.update({str(k): str(v) for k, v in worker_env.items()})
        if isinstance(spawn_env, dict):
            env.update({str(k): str(v) for k, v in spawn_env.items()})
        env.setdefault("MP13_ENGINE_CONFIG_PATH", str(config_path))
        env.setdefault("MP13_ENGINE_ID", str(engine_id))
        return {
            "command": command,
            "cwd": cwd,
            "env": env,
            "worker_transport": str(cfg.get("worker_transport") or "").strip() or None,
            "worker_ipc_family": str(cfg.get("worker_ipc_family") or "").strip() or None,
            "worker_ipc_address": str(cfg.get("worker_ipc_address") or "").strip() or None,
            "worker_auth_token": str(cfg.get("worker_auth_token") or "").strip() or None,
            "worker_auth_header": str(cfg.get("worker_auth_header") or "").strip() or None,
        }

    def connect_from_config(self, *, config_path: str, engine_id: Optional[str] = None, model_path: Optional[str] = None) -> Dict[str, Any]:
        progress_events: List[Dict[str, Any]] = [
            self._progress_event("connect.resolve_config", "running", "Resolving engine config"),
        ]
        selected = self._resolve_json_config_path(config_path)
        cfg = self._merge_default_and_selected_config(config_path)
        if not isinstance(cfg, dict):
            cfg = {}
        base_name = self._safe_config_name(Path(selected).stem or "engine")
        requested = self._safe_config_name(engine_id) if str(engine_id or "").strip() else ""
        eid = self._next_engine_id(requested or base_name)
        worker_class = self._classify_connect_worker_class(
            config_path=config_path,
            payload={"model_path": model_path},
        )

        effective_model_path: Optional[str] = None
        if worker_class == "generic":
            progress_events.append(
                self._progress_event("connect.resolve_model", "skipped", "Generic worker profile does not require model selection")
            )
        else:
            configured_model = (
                ((cfg.get("engine_params") or {}).get("base_model_path") if isinstance(cfg.get("engine_params"), dict) else None)
                or cfg.get("base_model_path")
                or cfg.get("model")
                or cfg.get("base_model_name_or_path")
            )
            effective_model_path = str(model_path or configured_model or "").strip() or None
            if not effective_model_path:
                progress_events.append(
                    self._progress_event("connect.resolve_model", "needs_input", "No model path configured")
                )
                return {
                    "status": "needs_model",
                    "stage": "needs_model",
                    "engine_id": eid,
                    "config_path": str(selected),
                    "models": self.models_from_config(config_path),
                    "message": "Config loaded but no model is configured. Select a model folder and connect again.",
                    "progress_events": progress_events,
                }
            progress_events.append(
                self._progress_event("connect.resolve_model", "completed", "Model selected", model_path=effective_model_path)
            )
        progress_events.append(
            self._progress_event("connect.build_spawn_spec", "running", "Preparing engine spawn spec")
        )
        if worker_class == "generic":
            spawn_spec = self._build_generic_spawn_spec(
                engine_id=eid,
                config_path=str(selected),
                config_payload=cfg,
            )
        else:
            spawn_spec = self._build_engine_spawn_spec(
                engine_id=eid,
                config_path=str(selected),
                model_path=str(effective_model_path),
            )
        if spawn_spec.get("error"):
            progress_events.append(
                self._progress_event("connect.build_spawn_spec", "failed", str(spawn_spec.get("error") or "spawn spec failed"))
            )
            return {
                "status": "failed",
                "stage": "failed",
                "engine_id": eid,
                "config_path": str(selected),
                "model_path": effective_model_path,
                "worker_class": worker_class,
                "reason": str(spawn_spec.get("error_kind") or "engine_spawn_error"),
                "message": str(spawn_spec.get("error") or "engine spawn spec build failed"),
                "progress_events": progress_events,
            }
        progress_events.append(
            self._progress_event("connect.build_spawn_spec", "completed", "Spawn spec built")
        )
        progress_events.append(
            self._progress_event("connect.spawn_engine", "running", "Starting engine process")
        )
        try:
            rec = self.spawn(
                engine_id=eid,
                command=list(spawn_spec.get("command") or []),
                cwd=spawn_spec.get("cwd"),
                env=dict(spawn_spec.get("env") or {}),
                worker_auth_token=str(spawn_spec.get("worker_auth_token") or "").strip() or None,
                worker_auth_header=str(spawn_spec.get("worker_auth_header") or "").strip() or None,
                worker_ipc_family=str(spawn_spec.get("worker_ipc_family") or "").strip() or None,
                worker_ipc_address=str(spawn_spec.get("worker_ipc_address") or "").strip() or None,
                worker_profile_class=worker_class,
            )
            progress_events.append(
                self._progress_event("connect.spawn_engine", "completed", "Engine started", engine_id=eid)
            )
            return {
                "status": "ok",
                "stage": "completed",
                "engine_id": eid,
                "config_path": str(selected),
                "model_path": effective_model_path,
                "worker_class": worker_class,
                "managed_engine": rec,
                "progress_events": progress_events,
            }
        except Exception as e:
            progress_events.append(
                self._progress_event("connect.spawn_engine", "failed", str(e))
            )
            return {
                "status": "failed",
                "stage": "failed",
                "engine_id": eid,
                "config_path": str(selected),
                "model_path": effective_model_path,
                "worker_class": worker_class,
                "reason": "spawn_failed",
                "message": str(e),
                "progress_events": progress_events,
            }

    def get_control_config(self) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        engine_policies = dict(cfg.get("engine_traffic_policies") or {})
        return {
            "daemon_version": DAEMON_VERSION,
            "capabilities": self.daemon_capabilities(),
            "ssh_key": cfg.get("ssh_key"),
            "require_auth": bool(cfg.get("require_auth", False)),
            "access_profile": dict(cfg.get("access_profile") or {"connectivity_mode": "local_only"}),
            "endpoint_mode_default": str(cfg.get("endpoint_mode_default") or "shared"),
            "lifecycle_profile": self._normalize_lifecycle_profile(cfg.get("lifecycle_profile")),
            "lifecycle_policy": self._normalize_lifecycle_policy(
                self._normalize_lifecycle_profile(cfg.get("lifecycle_profile")),
                dict(cfg.get("lifecycle_policy") or {}),
            ),
            "config_store_mode": str(cfg.get("config_store_mode") or "store_only"),
            "claim_acl_policy": self._claim_acl_policy(control),
            "traffic_policy": self._normalize_traffic_policy(dict(cfg.get("traffic_policy") or {})),
            "engine_traffic_policies": {
                str(k): self._normalize_traffic_policy(dict(v or {}))
                for k, v in engine_policies.items()
            },
            "engine_traffic_policies_count": len(engine_policies),
            "keys_count": len(dict(auth.get("keys") or {})),
            "sessions_count": len(dict(auth.get("sessions") or {})),
        }

    @staticmethod
    def _endpoint_mode_default(cfg: Dict[str, Any]) -> str:
        raw = str(cfg.get("endpoint_mode_default") or "shared").strip().lower()
        return "exclusive" if raw == "exclusive" else "shared"

    @staticmethod
    def _normalize_lifecycle_profile(profile: Any) -> str:
        raw = str(profile or LIFECYCLE_PROFILE_DETACHED).strip().lower()
        if raw in VALID_LIFECYCLE_PROFILES:
            return raw
        return LIFECYCLE_PROFILE_DETACHED

    @staticmethod
    def _default_lifecycle_policy_for_profile(profile: str) -> Dict[str, Any]:
        p = str(profile or LIFECYCLE_PROFILE_DETACHED).strip().lower()
        if p == LIFECYCLE_PROFILE_FOREGROUND:
            return {
                "on_terminal_disconnect": "stop_daemon",
                "terminal_control_enabled": True,
                "owner_disconnect_shutdown": True,
            }
        if p == LIFECYCLE_PROFILE_SERVICE:
            return {
                "on_terminal_disconnect": "keep_daemon_running",
                "terminal_control_enabled": False,
                "owner_disconnect_shutdown": False,
            }
        return {
            "on_terminal_disconnect": "keep_daemon_running",
            "terminal_control_enabled": True,
            "owner_disconnect_shutdown": False,
        }

    @classmethod
    def _normalize_lifecycle_policy(cls, profile: str, policy: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(cls._default_lifecycle_policy_for_profile(profile))
        incoming = dict(policy or {})
        disconnect = str(incoming.get("on_terminal_disconnect") or "").strip().lower()
        if disconnect in {"stop_daemon", "keep_daemon_running"}:
            out["on_terminal_disconnect"] = disconnect
        if "terminal_control_enabled" in incoming:
            out["terminal_control_enabled"] = bool(incoming.get("terminal_control_enabled"))
        if "owner_disconnect_shutdown" in incoming:
            out["owner_disconnect_shutdown"] = bool(incoming.get("owner_disconnect_shutdown"))
        return out

    def get_lifecycle_policy_effective(self) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        profile = self._normalize_lifecycle_profile(cfg.get("lifecycle_profile"))
        policy = self._normalize_lifecycle_policy(profile, dict(cfg.get("lifecycle_policy") or {}))
        return {
            "profile": profile,
            "policy": policy,
            "effective": {
                "daemon_survives_terminal_disconnect": (
                    str(policy.get("on_terminal_disconnect") or "") == "keep_daemon_running"
                ),
                "terminal_control_enabled": bool(policy.get("terminal_control_enabled", True)),
                "owner_disconnect_shutdown": bool(policy.get("owner_disconnect_shutdown", False)),
            },
        }

    def resolve_actor_id_from_session_token(self, token: str) -> Optional[str]:
        tok = str(token or "").strip()
        if not tok:
            return None
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        self._prune_expired_sessions(auth)
        sessions = dict(auth.get("sessions") or {})
        session = dict(sessions.get(tok) or {})
        if not session or bool(session.get("revoked", False)):
            return None
        key_id = str(session.get("key_id") or "").strip()
        if not key_id:
            return None
        return self._actor_id_from_session_key(key_id)

    def is_actor_exclusive_endpoint_owner(self, actor_id: str) -> bool:
        aid = str(actor_id or "").strip()
        if not aid:
            return False
        control = self._read_control()
        endpoint = dict(control.get("endpoint_claim") or {})
        owner = str(endpoint.get("exclusive_owner") or "").strip()
        return bool(owner and owner == aid)

    @staticmethod
    def _validate_require_auth_disabled_safe_profile(cfg: Dict[str, Any]) -> None:
        auth = dict(cfg.get("auth") or {})
        keys = dict(auth.get("keys") or {})
        sessions = dict(auth.get("sessions") or {})
        challenges = dict(auth.get("challenges") or {})
        access_profile = dict(cfg.get("access_profile") or {})
        connectivity_mode = str(access_profile.get("connectivity_mode") or "local_only").strip().lower()
        if connectivity_mode != "local_only":
            raise PermissionError("require_auth_false_only_supported_for_local_only_connectivity")
        if EngineHostService._endpoint_mode_default(cfg) != "exclusive":
            raise PermissionError("require_auth_false_requires_exclusive_endpoint_mode")
        if sessions or challenges:
            raise PermissionError("require_auth_false_requires_no_active_sessions_or_challenges")
        if not keys:
            return
        if len(keys) > 1:
            raise PermissionError("require_auth_false_requires_single_admin_key_profile")
        for meta in keys.values():
            role = str((meta or {}).get("role") or "").strip().lower()
            if role != ROLE_ADMIN:
                raise PermissionError("require_auth_false_requires_admin_only_keys")

    def assert_runtime_policy_safe(self) -> None:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        if bool(cfg.get("require_auth", False)):
            return
        self._validate_require_auth_disabled_safe_profile(cfg)

    def set_control_config(
        self,
        *,
        ssh_key: Optional[str] = None,
        require_auth: Optional[bool] = None,
        access_profile: Optional[Dict[str, Any]] = None,
        endpoint_mode_default: Optional[str] = None,
        lifecycle_profile: Optional[str] = None,
        lifecycle_policy: Optional[Dict[str, Any]] = None,
        traffic_policy: Optional[Dict[str, Any]] = None,
        engine_traffic_policies: Optional[Dict[str, Dict[str, Any]]] = None,
        claim_acl_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        if ssh_key is not None:
            cfg["ssh_key"] = str(ssh_key).strip() if ssh_key else None
        cfg["access_profile"] = dict(cfg.get("access_profile") or {"connectivity_mode": "local_only"})
        cfg["endpoint_mode_default"] = self._endpoint_mode_default(cfg)
        if access_profile is not None:
            cfg["access_profile"] = dict(cfg.get("access_profile") or {}) | dict(access_profile or {})
            cfg["access_profile"]["connectivity_mode"] = str(
                cfg["access_profile"].get("connectivity_mode") or "local_only"
            ).strip().lower()
        if endpoint_mode_default is not None:
            mode_raw = str(endpoint_mode_default or "").strip().lower()
            if mode_raw not in {"exclusive", "shared"}:
                raise ValueError("endpoint_mode_default must be 'exclusive' or 'shared'")
            cfg["endpoint_mode_default"] = mode_raw
        else:
            cfg["endpoint_mode_default"] = self._endpoint_mode_default(cfg)
        if (require_auth is not None and not bool(require_auth)) or (
            require_auth is None and not bool(cfg.get("require_auth", False))
        ):
            cfg["endpoint_mode_default"] = "exclusive"
        previous_profile = self._normalize_lifecycle_profile(cfg.get("lifecycle_profile"))
        current_profile = previous_profile
        if lifecycle_profile is not None:
            profile_raw = str(lifecycle_profile or "").strip().lower()
            if profile_raw not in VALID_LIFECYCLE_PROFILES:
                raise ValueError(
                    "lifecycle_profile must be one of: "
                    "foreground_terminal_bound, detached_user_process, service_managed"
                )
            current_profile = profile_raw
        cfg["lifecycle_profile"] = current_profile
        profile_changed = current_profile != previous_profile
        existing_policy = {} if profile_changed else dict(cfg.get("lifecycle_policy") or {})
        cfg["lifecycle_policy"] = self._normalize_lifecycle_policy(
            current_profile,
            existing_policy | dict(lifecycle_policy or {}),
        )
        if require_auth is not None:
            requested_require_auth = bool(require_auth)
            if not requested_require_auth:
                self._validate_require_auth_disabled_safe_profile(cfg)
            cfg["require_auth"] = requested_require_auth
        if not bool(cfg.get("require_auth", False)):
            self._validate_require_auth_disabled_safe_profile(cfg)
        cfg.setdefault("config_store_mode", "store_only")
        cfg.setdefault("auth", {"keys": {}, "sessions": {}})
        cfg.setdefault("engine_traffic_policies", {})
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
            "daemon_version": DAEMON_VERSION,
            "capabilities": self.daemon_capabilities(),
            "ssh_key": cfg.get("ssh_key"),
            "require_auth": bool(cfg.get("require_auth", False)),
            "access_profile": dict(cfg.get("access_profile") or {"connectivity_mode": "local_only"}),
            "endpoint_mode_default": str(cfg.get("endpoint_mode_default") or "shared"),
            "lifecycle_profile": self._normalize_lifecycle_profile(cfg.get("lifecycle_profile")),
            "lifecycle_policy": self._normalize_lifecycle_policy(
                self._normalize_lifecycle_profile(cfg.get("lifecycle_profile")),
                dict(cfg.get("lifecycle_policy") or {}),
            ),
            "config_store_mode": str(cfg.get("config_store_mode") or "store_only"),
            "claim_acl_policy": self._claim_acl_policy(control),
            "traffic_policy": self._normalize_traffic_policy(dict(cfg.get("traffic_policy") or {})),
            "engine_traffic_policies": {
                str(k): self._normalize_traffic_policy(dict(v or {}))
                for k, v in engine_policies.items()
            },
            "engine_traffic_policies_count": len(engine_policies),
            "keys_count": len(dict(auth.get("keys") or {})),
            "sessions_count": len(dict(auth.get("sessions") or {})),
        }

    def inspect_engine_capabilities(self, engine_id: str, endpoint: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        reg = self._require_ipc_registration(eid, command_label="inspect-capabilities")
        out = self._ipc_call(reg=reg, payload={"kind": "hello", "engine_id": eid}, timeout_seconds=5.0)
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "inspect_failed"))
        return {
            "engine_id": eid,
            "endpoint": str(reg.get("endpoint") or "ipc://local"),
            "checked_at": time.time(),
            "supported": {
                "health": True,
                "capabilities": True,
                "inference": True,
                "ws": False,
                "rpc": True,
                "async_rpc": bool(out.get("async_rpc", True)),
                "cancellation": bool(out.get("cancellation", True)),
            },
            "worker": dict(out or {}),
        }

    def toolbox_describe(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        tid = str(toolbox_id or "").strip()
        if not eid and not tid:
            raise ValueError("engine_id or toolbox_id is required")
        if tid and not eid:
            regs = self._toolbox_executor_registrations(tid)
            if not regs:
                raise ValueError(f"toolbox '{tid}' has no registered sandbox executors")
            tool_names: set[str] = set()
            advertised_tool_names: set[str] = set()
            hidden_allowed_tool_names: set[str] = set()
            sandbox_profile_ids: set[str] = set()
            engine_ids: List[str] = []
            for reg in regs:
                engine_ids.append(str(reg.get("engine_id") or ""))
                for name in list(self._registration_allowed_tool_names(reg) or set()):
                    tool_names.add(name)
                for name in list(self._registration_advertised_tool_names(reg) or set()):
                    advertised_tool_names.add(name)
                for name in list(self._registration_hidden_allowed_tool_names(reg) or set()):
                    hidden_allowed_tool_names.add(name)
                sandbox_profile_ids.add(str(dict(reg.get("bundle") or {}).get("sandbox_profile_id") or "default"))
            return {
                "status": "ok",
                "toolbox_id": tid,
                "engine_ids": [eid for eid in engine_ids if eid],
                "all_registered_tool_names": sorted(tool_names),
                "allowed_tool_names": sorted(tool_names),
                "advertised_tool_names": sorted(advertised_tool_names or tool_names),
                "hidden_allowed_tool_names": sorted(hidden_allowed_tool_names),
                "sandbox_profile_ids": sorted([pid for pid in sandbox_profile_ids if pid]),
                "executor_kind": "toolbox_executor_v1",
                "mode": "sandbox",
                "parallel_execution": {
                    "async_within_executor": True,
                    "sandbox_pool": len(engine_ids) > 1,
                },
            }
        reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-describe")
        out = self._ipc_call(
            reg=reg,
            payload={"kind": "rpc_call", "engine_id": eid, "method": "toolbox.describe", "params": {}},
            timeout_seconds=float(timeout_seconds or 10.0),
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "toolbox_describe_failed"))
        result = dict(out or {})
        result.setdefault("engine_id", eid)
        result.setdefault("executor_kind", str(reg.get("executor_kind") or "toolbox_executor_v1"))
        result.setdefault("bundle", dict(reg.get("bundle") or {}))
        result.setdefault("tool_access", dict(reg.get("tool_access") or {}))
        result.setdefault("all_registered_tool_names", sorted(list(self._registration_allowed_tool_names(reg) or set())))
        result.setdefault("allowed_tool_names", sorted(list(self._registration_allowed_tool_names(reg) or set())))
        result.setdefault("advertised_tool_names", sorted(list(self._registration_advertised_tool_names(reg) or set())))
        result.setdefault("hidden_allowed_tool_names", sorted(list(self._registration_hidden_allowed_tool_names(reg) or set())))
        result.pop("tool_names", None)
        return result

    def toolbox_gate(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        tool_name: str,
        tools_view: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        tid = str(toolbox_id or "").strip()
        name = str(tool_name or "").strip()
        view = self._tools_view_from_payload(tools_view)
        if not name:
            raise ValueError("tool_name is required")
        if not eid and not tid:
            raise ValueError("engine_id or toolbox_id is required")
        if tid and not eid:
            regs = self._toolbox_executor_registrations(tid)
            if not regs:
                return {
                    "status": "ok",
                    "toolbox_id": tid,
                    "tool_name": name,
                    "outcome": "unavailable_backend",
                    "reason": "toolbox_executor_missing",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
            allowed_for_toolbox: set[str] = set()
            for item in regs:
                allowed_for_toolbox.update(self._registration_allowed_tool_names(item) or set())
            if view is not None and name in allowed_for_toolbox and not view.is_allowed(name):
                return {
                    "status": "ok",
                    "toolbox_id": tid,
                    "tool_name": name,
                    "outcome": "denied",
                    "reason": "blocked_in_scope",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
            try:
                reg = self._route_toolbox_registration(toolbox_id=tid, tool_name=name, command_label="toolbox-gate")
            except PermissionError:
                return {
                    "status": "ok",
                    "toolbox_id": tid,
                    "tool_name": name,
                    "outcome": "denied",
                    "reason": "tool_not_allowed",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
            eid = str(reg.get("engine_id") or "").strip()
        else:
            reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-gate")
            allowed_tool_names = self._registration_allowed_tool_names(reg)
            if allowed_tool_names is not None and name not in allowed_tool_names:
                return {
                    "status": "ok",
                    "engine_id": eid,
                    "tool_name": name,
                    "outcome": "denied",
                    "reason": "tool_not_allowed",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
            if view is not None and allowed_tool_names is not None and name in allowed_tool_names and not view.is_allowed(name):
                return {
                    "status": "ok",
                    "engine_id": eid,
                    "toolbox_id": tid or self._registration_toolbox_id(reg),
                    "tool_name": name,
                    "outcome": "denied",
                    "reason": "blocked_in_scope",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
        return {
            "status": "ok",
            "engine_id": eid,
            "toolbox_id": tid or self._registration_toolbox_id(reg),
            "tool_name": name,
            "outcome": "allowed",
            "reason": "allowed",
            "executable": True,
            "requires_confirmation": False,
            "backend": "sandbox",
        }

    def toolbox_execute(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        tool_call: Dict[str, Any],
        timeout_seconds: float = 30.0,
        tools_view: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        call = dict(tool_call or {})
        tool_name = str(call.get("name") or "").strip()
        view = self._tools_view_from_payload(tools_view)
        if not tool_name:
            raise ValueError("tool_call.name is required")
        tid = str(toolbox_id or "").strip()
        if not eid and not tid:
            raise ValueError("engine_id or toolbox_id is required")
        if tid and not eid:
            gate = self.toolbox_gate(toolbox_id=tid, tool_name=tool_name, tools_view=tools_view)
            if str(gate.get("outcome") or "").strip().lower() != "allowed":
                reason = str(gate.get("reason") or gate.get("outcome") or "denied")
                raise PermissionError(f"{reason}:{tool_name}")
            reg = self._route_toolbox_registration(toolbox_id=tid, tool_name=tool_name, command_label="toolbox-execute")
            eid = str(reg.get("engine_id") or "").strip()
        else:
            reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-execute")
            allowed_tool_names = self._registration_allowed_tool_names(reg)
            if allowed_tool_names is not None and tool_name not in allowed_tool_names:
                raise PermissionError(f"tool_not_allowed:{tool_name}")
            if view is not None and allowed_tool_names is not None and tool_name in allowed_tool_names and not view.is_allowed(tool_name):
                raise PermissionError(f"blocked_in_scope:{tool_name}")
        out = self._ipc_call(
            reg=reg,
            payload={
                "kind": "rpc_call",
                "engine_id": eid,
                "method": "toolbox.execute",
                "params": {"tool_call": call},
            },
            timeout_seconds=float(timeout_seconds or 30.0),
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "toolbox_execute_failed"))
        result = dict(out or {})
        result.setdefault("engine_id", eid)
        return result

    def toolbox_cancel(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        tool_name: str = "",
        tool_call_id: str = "",
        timeout_seconds: float = 8.0,
        respawn: bool = True,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        tid = str(toolbox_id or "").strip()
        name = str(tool_name or "").strip()
        call_id = str(tool_call_id or "").strip()
        if not eid and not tid:
            raise ValueError("engine_id or toolbox_id is required")

        target_regs: List[Dict[str, Any]] = []
        if tid:
            if name:
                try:
                    target_regs = [self._route_toolbox_registration(toolbox_id=tid, tool_name=name, command_label="toolbox-cancel")]
                except PermissionError:
                    return {
                        "status": "ok",
                        "toolbox_id": tid,
                        "tool_name": name,
                        "outcome": "noop",
                        "reason": "tool_not_allowed",
                        "canceled_engine_ids": [],
                        "failed_engine_ids": [],
                        "repaired_toolbox_ids": [],
                    }
            else:
                target_regs = list(self._toolbox_executor_registrations(tid))
            if not target_regs:
                return {
                    "status": "ok",
                    "toolbox_id": tid,
                    "tool_name": name or None,
                    "outcome": "noop",
                    "reason": "toolbox_executor_missing",
                    "canceled_engine_ids": [],
                    "failed_engine_ids": [],
                    "repaired_toolbox_ids": [],
                }
        else:
            reg = dict(self._find_registration(eid) or {})
            if not reg:
                return {
                    "status": "ok",
                    "engine_id": eid,
                    "outcome": "noop",
                    "reason": "engine_not_found",
                    "canceled_engine_ids": [],
                    "failed_engine_ids": [],
                    "repaired_toolbox_ids": [],
                }
            target_regs = [reg]

        canceled_engine_ids: List[str] = []
        failed_engine_ids: List[str] = []
        shutdown_results: Dict[str, Dict[str, Any]] = {}
        target_toolbox_ids: set[str] = set()
        for reg in target_regs:
            target_engine_id = str(dict(reg or {}).get("engine_id") or "").strip()
            if not target_engine_id:
                continue
            reg_toolbox_id = self._registration_toolbox_id(dict(reg or {}))
            if reg_toolbox_id:
                target_toolbox_ids.add(reg_toolbox_id)
            shutdown_out = dict(self.shutdown(target_engine_id, timeout_seconds=float(timeout_seconds or 8.0)) or {})
            shutdown_results[target_engine_id] = shutdown_out
            if bool(shutdown_out.get("alive")) or str(shutdown_out.get("status") or "").strip() == "stop_failed":
                failed_engine_ids.append(target_engine_id)
            else:
                canceled_engine_ids.append(target_engine_id)

        repair_out: Dict[str, Any] = {}
        repaired_toolbox_ids: List[str] = []
        for target_toolbox_id in sorted(target_toolbox_ids):
            with self._locked_toolbox(target_toolbox_id):
                state = self._read_toolboxes()
                toolboxes = dict(state.get("toolboxes") or {})
                toolbox_row = dict(toolboxes.get(target_toolbox_id) or {})
                if toolbox_row:
                    toolbox_row = self._append_toolbox_cancel_event(
                        toolbox_row,
                        engine_ids=canceled_engine_ids,
                        tool_name=name or None,
                        tool_call_id=call_id or None,
                        respawn=respawn,
                        non_restartable=self._toolbox_tool_non_restartable(toolbox_row, name),
                    )
                    toolboxes[target_toolbox_id] = toolbox_row
                    state["toolboxes"] = toolboxes
                    self._write_toolboxes(state)
                if respawn:
                    repair_piece = dict(
                        self.toolbox_repair(
                            toolbox_ids=[target_toolbox_id],
                            only_inconsistent=False,
                            details=False,
                        )
                        or {}
                    )
                    if repair_piece:
                        repaired_toolbox_ids.extend(
                            [
                                str(item or "").strip()
                                for item in list(repair_piece.get("repaired_toolbox_ids") or [])
                                if str(item or "").strip()
                            ]
                        )
                        if not repair_out:
                            repair_out = dict(repair_piece)
                        else:
                            repair_out.setdefault("repaired_toolbox_ids", [])
                            repair_out["repaired_toolbox_ids"] = sorted(
                                {
                                    *list(repair_out.get("repaired_toolbox_ids") or []),
                                    *list(repair_piece.get("repaired_toolbox_ids") or []),
                                }
                            )

        return {
            "status": "ok",
            "engine_id": eid or None,
            "toolbox_id": tid or None,
            "tool_name": name or None,
            "tool_call_id": call_id or None,
            "respawn": bool(respawn),
            "outcome": (
                "canceled_and_repaired"
                if canceled_engine_ids and repaired_toolbox_ids
                else "canceled"
                if canceled_engine_ids
                else "noop"
                if not failed_engine_ids
                else "partial_failure"
            ),
            "canceled_engine_ids": sorted(canceled_engine_ids),
            "failed_engine_ids": sorted(failed_engine_ids),
            "repaired_toolbox_ids": sorted(repaired_toolbox_ids),
            "shutdown_results": shutdown_results,
            "repair": repair_out,
        }

    def _wait_for_toolbox_executor_ready(
        self,
        engine_id: str,
        *,
        timeout_seconds: float = 8.0,
        poll_seconds: float = 0.1,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        try:
            reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-ready")
        except Exception as exc:
            raise ToolboxRolloutError(
                f"toolbox_executor_missing:{eid}",
                code="toolbox_executor_missing",
                details={
                    "failure_phase": "spawned",
                    "engine_id": eid,
                    "reason": str(exc),
                },
            ) from exc
        deadline = time.time() + max(0.1, float(timeout_seconds or 8.0))
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                desc = self.toolbox_describe(engine_id=eid, timeout_seconds=min(2.0, max(0.2, float(timeout_seconds or 8.0))))
                allowed = self._registration_allowed_tool_names(reg)
                reported = {
                    str(item or "").strip()
                    for item in list(
                        dict(desc or {}).get("all_registered_tool_names")
                        or []
                    )
                    if str(item or "").strip()
                }
                if allowed is not None and reported != allowed:
                    raise ToolboxRolloutError(
                        f"toolbox_executor_inventory_mismatch:{eid}",
                        code="toolbox_executor_inventory_mismatch",
                        details={
                            "failure_phase": "inventory_verified",
                            "engine_id": eid,
                            "expected_tool_names": sorted(allowed),
                            "actual_tool_names": sorted(reported),
                        },
                    )
                return desc
            except Exception as exc:
                last_error = exc
                time.sleep(max(0.05, float(poll_seconds or 0.1)))
        if isinstance(last_error, ToolboxRolloutError):
            raise ToolboxRolloutError(
                str(last_error),
                code=last_error.code,
                details=dict(last_error.details or {}),
            ) from last_error
        raise ToolboxRolloutError(
            f"toolbox_executor_not_ready:{eid}:{last_error}",
            code="toolbox_executor_not_ready",
            details={
                "failure_phase": "ready",
                "engine_id": eid,
                "timeout_seconds": float(timeout_seconds or 8.0),
                "reason": str(last_error or ""),
            },
        )

    def _ensure_toolbox_assignments_ready(
        self,
        assignments: List[Any],
        *,
        timeout_seconds: float = 8.0,
    ) -> Dict[str, Dict[str, Any]]:
        from .toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        ready: Dict[str, Dict[str, Any]] = {}
        environment_manager = ToolboxEnvironmentManager(self.hosting_root)
        for item in list(assignments or []):
            reg = dict(getattr(item, "registration", None) or {})
            engine_id = str(reg.get("engine_id") or "").strip()
            if not engine_id:
                continue
            started_at = time.time()
            try:
                desc = self._wait_for_toolbox_executor_ready(engine_id, timeout_seconds=timeout_seconds)
            except ToolboxRolloutError as exc:
                bundle = dict(reg.get("bundle") or {})
                details = dict(exc.details or {})
                details.setdefault("toolbox_id", str(bundle.get("toolbox_id") or getattr(item, "toolbox_id", "") or ""))
                details.setdefault(
                    "sandbox_profile_id",
                    str(bundle.get("sandbox_profile_id") or getattr(getattr(item, "sandbox_profile", None), "normalized_profile_id", lambda: "")() or ""),
                )
                details.setdefault("bundle_revision", str(bundle.get("bundle_revision") or ""))
                details.setdefault("engine_id", engine_id)
                raise ToolboxRolloutError(str(exc), code=exc.code, details=details) from exc
            ready_at = time.time()
            tool_names = [
                str(name or "").strip()
                for name in list(
                    dict(desc or {}).get("all_registered_tool_names")
                    or []
                )
                if str(name or "").strip()
            ]
            environment = dict(reg.get("environment") or {})
            receipt_verification_status = None
            install_execution_status = None
            if environment:
                spec = ToolboxEnvironmentSpec.from_dict(environment)
                metadata = environment_manager.read_environment_metadata(spec)
                install_execution_status = str(dict(metadata.get("install_execution") or {}).get("status") or "").strip() or None
                receipt_verification_status = str(
                    dict(metadata.get("install_receipt_verification") or {}).get("status") or ""
                ).strip() or None
                if install_execution_status == "ok" and receipt_verification_status != "ok":
                    raise ToolboxRolloutError(
                        f"environment receipt verification not ready for {engine_id}",
                        code="toolbox_environment_receipt_unverified",
                        details={
                            "engine_id": engine_id,
                            "install_execution_status": install_execution_status,
                            "install_receipt_verification_status": receipt_verification_status,
                            "toolbox_id": str(dict(reg.get("bundle") or {}).get("toolbox_id") or getattr(item, "toolbox_id", "") or ""),
                            "sandbox_profile_id": str(
                                dict(reg.get("bundle") or {}).get("sandbox_profile_id")
                                or getattr(getattr(item, "sandbox_profile", None), "normalized_profile_id", lambda: "")()
                                or ""
                            ),
                        },
                    )
            ready[engine_id] = {
                "engine_id": engine_id,
                "ready": True,
                "ready_at": ready_at,
                "warmup_ms": int(max(0.0, (ready_at - started_at) * 1000.0)),
                "tool_inventory_ok": True,
                "tool_count": len(tool_names),
                "all_registered_tool_names": tool_names,
                "install_execution_status": install_execution_status,
                "install_receipt_verification_status": receipt_verification_status,
            }
        return ready

    def toolbox_register_auto(
        self,
        *,
        toolbox_id: str,
        requests: List[Dict[str, Any]],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        return self._run_locked_toolbox_call(
            tid,
            self._toolbox_register_auto_impl,
            toolbox_id=tid,
            requests=list(requests or []),
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

    def _toolbox_register_auto_impl(
        self,
        *,
        toolbox_id: str,
        requests: List[Dict[str, Any]],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        from .toolbox_harness import (
            ToolboxAutoAssignmentRequest,
            SandboxProfileSpec,
            ToolboxBundleStager,
            ToolboxManualAssignmentRequest,
            ToolboxSandboxOrchestrator,
        )

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        if not list(requests or []):
            raise ValueError("requests are required")
        merged_requests, state, toolboxes = self._merge_toolbox_auto_requests(
            toolbox_id=tid,
            incoming_requests=list(requests or []),
        )
        toolbox_row_existing = dict(toolboxes.get(tid) or {})
        manual_requests = [
            ToolboxManualAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(toolbox_row_existing.get("manual_requests") or [])
        ]
        intrinsics_row = dict(toolbox_row_existing.get("intrinsics") or {})
        intrinsic_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsics_row.get("names") or []) if str(item or "").strip()],
            include_guides=bool(intrinsics_row.get("with_intrinsic_guides", False)),
        )
        intrinsic_profile = SandboxProfileSpec.from_dict(dict(intrinsics_row.get("sandbox_profile") or {})) if intrinsic_names else None
        with_intrinsic_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
        existing_profiles = dict(dict(toolboxes.get(tid) or {}).get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(
            toolbox_row_existing,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        orchestrator = ToolboxSandboxOrchestrator(
            service=self,
            stager=ToolboxBundleStager(self.hosting_root),
            python_executable=runtime.get("python_executable"),
        )
        assignments = orchestrator.spawn_assignments(
            toolbox_id=tid,
            requests=merged_requests,
            manual_requests=manual_requests,
            intrinsic_tool_names=intrinsic_names,
            intrinsic_profile=intrinsic_profile,
            with_intrinsic_guides=with_intrinsic_guides,
            worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
        )
        try:
            ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
        except Exception:
            for item in assignments:
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    self._retire_toolbox_registration(engine_id)
            self._cleanup_unused_toolbox_environments(state)
            raise

        new_profiles: Dict[str, Dict[str, Any]] = {}
        spawned_engine_ids: List[str] = []
        replaced_engine_ids: List[str] = []
        for item in assignments:
            profile_id = item.sandbox_profile.normalized_profile_id()
            reg = dict(item.registration or {})
            engine_id = str(reg.get("engine_id") or "").strip()
            if engine_id:
                spawned_engine_ids.append(engine_id)
            old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
            bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
            if old_engine_id and old_engine_id != engine_id:
                replaced_engine_ids.append(old_engine_id)
            profile_requests = [
                req.to_runtime_dict()
                for req in merged_requests
                if req.sandbox_profile.normalized_profile_id() == profile_id
            ]
            new_profiles[profile_id] = {
                "sandbox_profile": item.sandbox_profile.to_dict(),
                "requests": profile_requests,
                "engine_id": engine_id,
                "bundle_revision": bundle_revision,
                "environment": dict(reg.get("environment") or {}),
                "rollout": dict(ready_rollout.get(engine_id) or {}),
                "rollout_history": self._append_toolbox_rollout_history(
                    dict(existing_profiles.get(profile_id) or {}),
                    rollout=dict(ready_rollout.get(engine_id) or {}),
                    action="register_auto",
                    engine_id=engine_id,
                    bundle_revision=bundle_revision,
                    replaced_engine_id=old_engine_id,
                ),
            }

        for profile_id, old_engine_id in old_regs_by_profile.items():
            if profile_id not in new_profiles and old_engine_id:
                replaced_engine_ids.append(old_engine_id)

        for old_engine_id in replaced_engine_ids:
            self._retire_toolbox_registration(old_engine_id)

        toolbox_row = {
            "toolbox_id": tid,
            "requests": [req.to_runtime_dict() for req in merged_requests],
            "manual_requests": [req.to_runtime_dict() for req in manual_requests],
            "profiles": new_profiles,
            "runtime": runtime,
        }
        if intrinsic_names:
            toolbox_row["intrinsics"] = {
                "names": intrinsic_names,
                "sandbox_profile": (intrinsic_profile or SandboxProfileSpec(profile_id="default")).to_dict(),
                "with_intrinsic_guides": with_intrinsic_guides,
            }
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "request_count": len(merged_requests),
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "removed_environment_keys": removed_environment_keys,
        }

    def toolbox_unregister_auto(
        self,
        *,
        toolbox_id: str,
        tool_keys: List[str],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        return self._run_locked_toolbox_call(
            tid,
            self._toolbox_unregister_auto_impl,
            toolbox_id=tid,
            tool_keys=list(tool_keys or []),
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

    def _toolbox_unregister_auto_impl(
        self,
        *,
        toolbox_id: str,
        tool_keys: List[str],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        from .toolbox_harness import (
            SandboxProfileSpec,
            ToolboxBundleStager,
            ToolboxManualAssignmentRequest,
            ToolboxSandboxOrchestrator,
        )

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        keys = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        if not keys:
            raise ValueError("tool_keys are required")

        merged_requests, state, toolboxes = self._merge_toolbox_auto_requests(
            toolbox_id=tid,
            remove_keys=keys,
        )
        toolbox_row_existing = dict(toolboxes.get(tid) or {})
        manual_requests = [
            ToolboxManualAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(toolbox_row_existing.get("manual_requests") or [])
        ]
        intrinsics_row = dict(toolbox_row_existing.get("intrinsics") or {})
        intrinsic_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsics_row.get("names") or []) if str(item or "").strip()],
            include_guides=bool(intrinsics_row.get("with_intrinsic_guides", False)),
        )
        intrinsic_profile = SandboxProfileSpec.from_dict(dict(intrinsics_row.get("sandbox_profile") or {})) if intrinsic_names else None
        with_intrinsic_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
        existing_profiles = dict(dict(toolboxes.get(tid) or {}).get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(
            toolbox_row_existing,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )
        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        replaced_engine_ids: List[str] = []
        spawned_engine_ids: List[str] = []
        new_profiles: Dict[str, Dict[str, Any]] = {}

        if merged_requests or manual_requests or intrinsic_names:
            orchestrator = ToolboxSandboxOrchestrator(
                service=self,
                stager=ToolboxBundleStager(self.hosting_root),
                python_executable=runtime.get("python_executable"),
            )
            assignments = orchestrator.spawn_assignments(
                toolbox_id=tid,
                requests=merged_requests,
                manual_requests=manual_requests,
                intrinsic_tool_names=intrinsic_names,
                intrinsic_profile=intrinsic_profile,
                with_intrinsic_guides=with_intrinsic_guides,
                worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
            )
            try:
                ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
            except Exception:
                for item in assignments:
                    reg = dict(item.registration or {})
                    engine_id = str(reg.get("engine_id") or "").strip()
                    if engine_id:
                        self._retire_toolbox_registration(engine_id)
                self._cleanup_unused_toolbox_environments(state)
                raise
            for item in assignments:
                profile_id = item.sandbox_profile.normalized_profile_id()
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    spawned_engine_ids.append(engine_id)
                old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
                bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
                if old_engine_id and old_engine_id != engine_id:
                    replaced_engine_ids.append(old_engine_id)
                profile_requests = [
                    req.to_runtime_dict()
                    for req in merged_requests
                    if req.sandbox_profile.normalized_profile_id() == profile_id
                ]
                new_profiles[profile_id] = {
                    "sandbox_profile": item.sandbox_profile.to_dict(),
                    "requests": profile_requests,
                    "engine_id": engine_id,
                    "bundle_revision": bundle_revision,
                    "environment": dict(reg.get("environment") or {}),
                    "rollout": dict(ready_rollout.get(engine_id) or {}),
                    "rollout_history": self._append_toolbox_rollout_history(
                        dict(existing_profiles.get(profile_id) or {}),
                        rollout=dict(ready_rollout.get(engine_id) or {}),
                        action="unregister_auto",
                        engine_id=engine_id,
                        bundle_revision=bundle_revision,
                        replaced_engine_id=old_engine_id,
                    ),
                }
            for profile_id, old_engine_id in old_regs_by_profile.items():
                if profile_id not in new_profiles and old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
            toolboxes[tid] = {
                "toolbox_id": tid,
                "requests": [req.to_runtime_dict() for req in merged_requests],
                "manual_requests": [req.to_runtime_dict() for req in manual_requests],
                "profiles": new_profiles,
                "runtime": runtime,
                **(
                    {
                        "intrinsics": {
                            "names": intrinsic_names,
                            "sandbox_profile": (intrinsic_profile or SandboxProfileSpec(profile_id="default")).to_dict(),
                            "with_intrinsic_guides": with_intrinsic_guides,
                        }
                    }
                    if intrinsic_names
                    else {}
                ),
            }
        else:
            ready_rollout = {}
            for old_engine_id in old_regs_by_profile.values():
                if old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
            toolboxes.pop(tid, None)

        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "remaining_request_count": len(merged_requests),
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "removed_tool_keys": keys,
            "toolbox_removed": not merged_requests and not manual_requests and not intrinsic_names,
            "removed_environment_keys": removed_environment_keys,
        }

    def toolbox_register_intrinsics(
        self,
        *,
        toolbox_id: str,
        intrinsic_tool_names: List[str],
        include_guides: bool = False,
        sandbox_profile: Optional[Dict[str, Any]] = None,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        return self._run_locked_toolbox_call(
            tid,
            self._toolbox_register_intrinsics_impl,
            toolbox_id=tid,
            intrinsic_tool_names=list(intrinsic_tool_names or []),
            include_guides=include_guides,
            sandbox_profile=dict(sandbox_profile or {}) if isinstance(sandbox_profile, dict) else None,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

    def _toolbox_register_intrinsics_impl(
        self,
        *,
        toolbox_id: str,
        intrinsic_tool_names: List[str],
        include_guides: bool = False,
        sandbox_profile: Optional[Dict[str, Any]] = None,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        from .toolbox_harness import SandboxProfileSpec, ToolboxBundleStager, ToolboxSandboxOrchestrator

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()],
            include_guides=bool(include_guides),
        )
        if not names:
            raise ValueError("intrinsic_tool_names are required")
        merged_requests, state, toolboxes = self._merge_toolbox_auto_requests(toolbox_id=tid)
        toolbox_row_existing = dict(toolboxes.get(tid) or {})
        existing_profiles = dict(toolbox_row_existing.get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(
            toolbox_row_existing,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )
        existing_intrinsics = dict(toolbox_row_existing.get("intrinsics") or {})
        existing_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(existing_intrinsics.get("names") or []) if str(item or "").strip()],
            include_guides=bool(existing_intrinsics.get("with_intrinsic_guides", False)),
        )
        merged_names = self._normalize_intrinsic_tool_names(existing_names + names, include_guides=False)
        intrinsic_profile = SandboxProfileSpec.from_dict(
            dict(sandbox_profile or existing_intrinsics.get("sandbox_profile") or {})
        )
        with_intrinsic_guides = bool(include_guides or existing_intrinsics.get("with_intrinsic_guides", False))

        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        orchestrator = ToolboxSandboxOrchestrator(
            service=self,
            stager=ToolboxBundleStager(self.hosting_root),
            python_executable=runtime.get("python_executable"),
        )
        assignments = orchestrator.spawn_assignments(
            toolbox_id=tid,
            requests=merged_requests,
            intrinsic_tool_names=merged_names,
            intrinsic_profile=intrinsic_profile,
            with_intrinsic_guides=with_intrinsic_guides,
            worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
        )
        try:
            ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
        except Exception:
            for item in assignments:
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    self._retire_toolbox_registration(engine_id)
            self._cleanup_unused_toolbox_environments(state)
            raise

        new_profiles: Dict[str, Dict[str, Any]] = {}
        spawned_engine_ids: List[str] = []
        replaced_engine_ids: List[str] = []
        for item in assignments:
            profile_id = item.sandbox_profile.normalized_profile_id()
            reg = dict(item.registration or {})
            engine_id = str(reg.get("engine_id") or "").strip()
            if engine_id:
                spawned_engine_ids.append(engine_id)
            old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
            bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
            if old_engine_id and old_engine_id != engine_id:
                replaced_engine_ids.append(old_engine_id)
            profile_requests = [
                req.to_runtime_dict()
                for req in merged_requests
                if req.sandbox_profile.normalized_profile_id() == profile_id
            ]
            new_profiles[profile_id] = {
                "sandbox_profile": item.sandbox_profile.to_dict(),
                "requests": profile_requests,
                "engine_id": engine_id,
                "bundle_revision": bundle_revision,
                "environment": dict(reg.get("environment") or {}),
                "rollout": dict(ready_rollout.get(engine_id) or {}),
                "rollout_history": self._append_toolbox_rollout_history(
                    dict(existing_profiles.get(profile_id) or {}),
                    rollout=dict(ready_rollout.get(engine_id) or {}),
                    action="register_intrinsics",
                    engine_id=engine_id,
                    bundle_revision=bundle_revision,
                    replaced_engine_id=old_engine_id,
                ),
            }

        for profile_id, old_engine_id in old_regs_by_profile.items():
            if profile_id not in new_profiles and old_engine_id:
                replaced_engine_ids.append(old_engine_id)
        for old_engine_id in replaced_engine_ids:
            self._retire_toolbox_registration(old_engine_id)

        toolboxes[tid] = {
            "toolbox_id": tid,
            "requests": [req.to_runtime_dict() for req in merged_requests],
            "profiles": new_profiles,
            "runtime": runtime,
            "intrinsics": {
                "names": merged_names,
                "sandbox_profile": intrinsic_profile.to_dict(),
                "with_intrinsic_guides": with_intrinsic_guides,
            },
        }
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "intrinsic_tool_names": merged_names,
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "removed_environment_keys": removed_environment_keys,
        }

    def toolbox_unregister_intrinsics(
        self,
        *,
        toolbox_id: str,
        intrinsic_tool_names: List[str],
        include_guides: bool = False,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        return self._run_locked_toolbox_call(
            tid,
            self._toolbox_unregister_intrinsics_impl,
            toolbox_id=tid,
            intrinsic_tool_names=list(intrinsic_tool_names or []),
            include_guides=include_guides,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

    def _toolbox_unregister_intrinsics_impl(
        self,
        *,
        toolbox_id: str,
        intrinsic_tool_names: List[str],
        include_guides: bool = False,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        from .toolbox_harness import SandboxProfileSpec, ToolboxBundleStager, ToolboxSandboxOrchestrator

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        remove_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()],
            include_guides=bool(include_guides),
        )
        if not remove_names:
            raise ValueError("intrinsic_tool_names are required")
        merged_requests, state, toolboxes = self._merge_toolbox_auto_requests(toolbox_id=tid)
        toolbox_row_existing = dict(toolboxes.get(tid) or {})
        existing_profiles = dict(toolbox_row_existing.get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(
            toolbox_row_existing,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )
        existing_intrinsics = dict(toolbox_row_existing.get("intrinsics") or {})
        current_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(existing_intrinsics.get("names") or []) if str(item or "").strip()],
            include_guides=bool(existing_intrinsics.get("with_intrinsic_guides", False)),
        )
        remaining_names = [name for name in current_names if name not in set(remove_names)]
        intrinsic_profile = SandboxProfileSpec.from_dict(dict(existing_intrinsics.get("sandbox_profile") or {}))
        with_intrinsic_guides = bool(existing_intrinsics.get("with_intrinsic_guides", False))

        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        replaced_engine_ids: List[str] = []
        spawned_engine_ids: List[str] = []
        new_profiles: Dict[str, Dict[str, Any]] = {}
        if merged_requests or remaining_names:
            orchestrator = ToolboxSandboxOrchestrator(
                service=self,
                stager=ToolboxBundleStager(self.hosting_root),
                python_executable=runtime.get("python_executable"),
            )
            assignments = orchestrator.spawn_assignments(
                toolbox_id=tid,
                requests=merged_requests,
                intrinsic_tool_names=remaining_names,
                intrinsic_profile=intrinsic_profile if remaining_names else None,
                with_intrinsic_guides=with_intrinsic_guides,
                worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
            )
            try:
                ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
            except Exception:
                for item in assignments:
                    reg = dict(item.registration or {})
                    engine_id = str(reg.get("engine_id") or "").strip()
                    if engine_id:
                        self._retire_toolbox_registration(engine_id)
                self._cleanup_unused_toolbox_environments(state)
                raise
            for item in assignments:
                profile_id = item.sandbox_profile.normalized_profile_id()
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    spawned_engine_ids.append(engine_id)
                old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
                bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
                if old_engine_id and old_engine_id != engine_id:
                    replaced_engine_ids.append(old_engine_id)
                profile_requests = [
                    req.to_runtime_dict()
                    for req in merged_requests
                    if req.sandbox_profile.normalized_profile_id() == profile_id
                ]
                new_profiles[profile_id] = {
                    "sandbox_profile": item.sandbox_profile.to_dict(),
                    "requests": profile_requests,
                    "engine_id": engine_id,
                    "bundle_revision": bundle_revision,
                    "environment": dict(reg.get("environment") or {}),
                    "rollout": dict(ready_rollout.get(engine_id) or {}),
                    "rollout_history": self._append_toolbox_rollout_history(
                        dict(existing_profiles.get(profile_id) or {}),
                        rollout=dict(ready_rollout.get(engine_id) or {}),
                        action="unregister_intrinsics",
                        engine_id=engine_id,
                        bundle_revision=bundle_revision,
                        replaced_engine_id=old_engine_id,
                    ),
                }
            for profile_id, old_engine_id in old_regs_by_profile.items():
                if profile_id not in new_profiles and old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
            toolboxes[tid] = {
                "toolbox_id": tid,
                "requests": [req.to_runtime_dict() for req in merged_requests],
                "profiles": new_profiles,
                "runtime": runtime,
                **(
                    {
                        "intrinsics": {
                            "names": remaining_names,
                            "sandbox_profile": intrinsic_profile.to_dict(),
                            "with_intrinsic_guides": with_intrinsic_guides,
                        }
                    }
                    if remaining_names
                    else {}
                ),
            }
        else:
            ready_rollout = {}
            for old_engine_id in old_regs_by_profile.values():
                if old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
            toolboxes.pop(tid, None)
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "remaining_intrinsic_tool_names": remaining_names,
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "toolbox_removed": not merged_requests and not remaining_names,
            "removed_environment_keys": removed_environment_keys,
        }

    def toolbox_register_manual(
        self,
        *,
        toolbox_id: str,
        requests: List[Dict[str, Any]],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        return self._run_locked_toolbox_call(
            tid,
            self._toolbox_register_manual_impl,
            toolbox_id=tid,
            requests=list(requests or []),
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

    def _toolbox_register_manual_impl(
        self,
        *,
        toolbox_id: str,
        requests: List[Dict[str, Any]],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        from .toolbox_harness import (
            SandboxProfileSpec,
            ToolboxBundleStager,
            ToolboxManualAssignmentRequest,
            ToolboxSandboxOrchestrator,
        )

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        if not list(requests or []):
            raise ValueError("requests are required")
        manual_requests, state, toolboxes = self._merge_toolbox_manual_requests(
            toolbox_id=tid,
            incoming_requests=list(requests or []),
        )
        toolbox_row_existing = dict(toolboxes.get(tid) or {})
        auto_requests = [
            ToolboxAutoAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(toolbox_row_existing.get("requests") or [])
        ]
        intrinsics_row = dict(toolbox_row_existing.get("intrinsics") or {})
        intrinsic_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsics_row.get("names") or []) if str(item or "").strip()],
            include_guides=bool(intrinsics_row.get("with_intrinsic_guides", False)),
        )
        intrinsic_profile = SandboxProfileSpec.from_dict(dict(intrinsics_row.get("sandbox_profile") or {})) if intrinsic_names else None
        with_intrinsic_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
        existing_profiles = dict(dict(toolboxes.get(tid) or {}).get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(
            toolbox_row_existing,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )
        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        orchestrator = ToolboxSandboxOrchestrator(
            service=self,
            stager=ToolboxBundleStager(self.hosting_root),
            python_executable=runtime.get("python_executable"),
        )
        assignments = orchestrator.spawn_assignments(
            toolbox_id=tid,
            requests=auto_requests,
            manual_requests=manual_requests,
            intrinsic_tool_names=intrinsic_names,
            intrinsic_profile=intrinsic_profile,
            with_intrinsic_guides=with_intrinsic_guides,
            worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
        )
        try:
            ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
        except Exception:
            for item in assignments:
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    self._retire_toolbox_registration(engine_id)
            self._cleanup_unused_toolbox_environments(state)
            raise

        new_profiles: Dict[str, Dict[str, Any]] = {}
        spawned_engine_ids: List[str] = []
        replaced_engine_ids: List[str] = []
        for item in assignments:
            profile_id = item.sandbox_profile.normalized_profile_id()
            reg = dict(item.registration or {})
            engine_id = str(reg.get("engine_id") or "").strip()
            if engine_id:
                spawned_engine_ids.append(engine_id)
            old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
            bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
            if old_engine_id and old_engine_id != engine_id:
                replaced_engine_ids.append(old_engine_id)
            profile_auto_requests = [
                req.to_runtime_dict()
                for req in auto_requests
                if req.sandbox_profile.normalized_profile_id() == profile_id
            ]
            profile_manual_requests = [
                req.to_runtime_dict()
                for req in manual_requests
                if req.sandbox_profile.normalized_profile_id() == profile_id
            ]
            new_profiles[profile_id] = {
                "sandbox_profile": item.sandbox_profile.to_dict(),
                "requests": profile_auto_requests,
                "manual_requests": profile_manual_requests,
                "engine_id": engine_id,
                "bundle_revision": bundle_revision,
                "environment": dict(reg.get("environment") or {}),
                "rollout": dict(ready_rollout.get(engine_id) or {}),
                "rollout_history": self._append_toolbox_rollout_history(
                    dict(existing_profiles.get(profile_id) or {}),
                    rollout=dict(ready_rollout.get(engine_id) or {}),
                    action="register_manual",
                    engine_id=engine_id,
                    bundle_revision=bundle_revision,
                    replaced_engine_id=old_engine_id,
                ),
            }

        for profile_id, old_engine_id in old_regs_by_profile.items():
            if profile_id not in new_profiles and old_engine_id:
                replaced_engine_ids.append(old_engine_id)
        for old_engine_id in replaced_engine_ids:
            self._retire_toolbox_registration(old_engine_id)

        toolboxes[tid] = {
            "toolbox_id": tid,
            "requests": [req.to_runtime_dict() for req in auto_requests],
            "manual_requests": [req.to_runtime_dict() for req in manual_requests],
            "profiles": new_profiles,
            "runtime": runtime,
            **(
                {
                    "intrinsics": {
                        "names": intrinsic_names,
                        "sandbox_profile": (intrinsic_profile or SandboxProfileSpec(profile_id="default")).to_dict(),
                        "with_intrinsic_guides": with_intrinsic_guides,
                    }
                }
                if intrinsic_names
                else {}
            ),
        }
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "request_count": len(manual_requests),
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "removed_environment_keys": removed_environment_keys,
        }

    def toolbox_unregister_manual(
        self,
        *,
        toolbox_id: str,
        tool_keys: List[str],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        return self._run_locked_toolbox_call(
            tid,
            self._toolbox_unregister_manual_impl,
            toolbox_id=tid,
            tool_keys=list(tool_keys or []),
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )

    def _toolbox_unregister_manual_impl(
        self,
        *,
        toolbox_id: str,
        tool_keys: List[str],
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        from .toolbox_harness import (
            SandboxProfileSpec,
            ToolboxAutoAssignmentRequest,
            ToolboxBundleStager,
            ToolboxManualAssignmentRequest,
            ToolboxSandboxOrchestrator,
        )

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        keys = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        if not keys:
            raise ValueError("tool_keys are required")
        manual_requests, state, toolboxes = self._merge_toolbox_manual_requests(
            toolbox_id=tid,
            remove_keys=keys,
        )
        toolbox_row_existing = dict(toolboxes.get(tid) or {})
        auto_requests = [
            ToolboxAutoAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(toolbox_row_existing.get("requests") or [])
        ]
        intrinsics_row = dict(toolbox_row_existing.get("intrinsics") or {})
        intrinsic_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsics_row.get("names") or []) if str(item or "").strip()],
            include_guides=bool(intrinsics_row.get("with_intrinsic_guides", False)),
        )
        intrinsic_profile = SandboxProfileSpec.from_dict(dict(intrinsics_row.get("sandbox_profile") or {})) if intrinsic_names else None
        with_intrinsic_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
        existing_profiles = dict(dict(toolboxes.get(tid) or {}).get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(
            toolbox_row_existing,
            python_executable=python_executable,
            worker_profile_class=worker_profile_class,
        )
        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        replaced_engine_ids: List[str] = []
        spawned_engine_ids: List[str] = []
        new_profiles: Dict[str, Dict[str, Any]] = {}
        if auto_requests or manual_requests or intrinsic_names:
            orchestrator = ToolboxSandboxOrchestrator(
                service=self,
                stager=ToolboxBundleStager(self.hosting_root),
                python_executable=runtime.get("python_executable"),
            )
            assignments = orchestrator.spawn_assignments(
                toolbox_id=tid,
                requests=auto_requests,
                manual_requests=manual_requests,
                intrinsic_tool_names=intrinsic_names,
                intrinsic_profile=intrinsic_profile,
                with_intrinsic_guides=with_intrinsic_guides,
                worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
            )
            try:
                ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
            except Exception:
                for item in assignments:
                    reg = dict(item.registration or {})
                    engine_id = str(reg.get("engine_id") or "").strip()
                    if engine_id:
                        self._retire_toolbox_registration(engine_id)
                self._cleanup_unused_toolbox_environments(state)
                raise
            for item in assignments:
                profile_id = item.sandbox_profile.normalized_profile_id()
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    spawned_engine_ids.append(engine_id)
                old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
                bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
                if old_engine_id and old_engine_id != engine_id:
                    replaced_engine_ids.append(old_engine_id)
                profile_auto_requests = [
                    req.to_runtime_dict()
                    for req in auto_requests
                    if req.sandbox_profile.normalized_profile_id() == profile_id
                ]
                profile_manual_requests = [
                    req.to_runtime_dict()
                    for req in manual_requests
                    if req.sandbox_profile.normalized_profile_id() == profile_id
                ]
                new_profiles[profile_id] = {
                    "sandbox_profile": item.sandbox_profile.to_dict(),
                    "requests": profile_auto_requests,
                    "manual_requests": profile_manual_requests,
                    "engine_id": engine_id,
                    "bundle_revision": bundle_revision,
                    "environment": dict(reg.get("environment") or {}),
                    "rollout": dict(ready_rollout.get(engine_id) or {}),
                    "rollout_history": self._append_toolbox_rollout_history(
                        dict(existing_profiles.get(profile_id) or {}),
                        rollout=dict(ready_rollout.get(engine_id) or {}),
                        action="unregister_manual",
                        engine_id=engine_id,
                        bundle_revision=bundle_revision,
                        replaced_engine_id=old_engine_id,
                    ),
                }
            for profile_id, old_engine_id in old_regs_by_profile.items():
                if profile_id not in new_profiles and old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
            toolboxes[tid] = {
                "toolbox_id": tid,
                "requests": [req.to_runtime_dict() for req in auto_requests],
                "manual_requests": [req.to_runtime_dict() for req in manual_requests],
                "profiles": new_profiles,
                "runtime": runtime,
                **(
                    {
                        "intrinsics": {
                            "names": intrinsic_names,
                            "sandbox_profile": (intrinsic_profile or SandboxProfileSpec(profile_id="default")).to_dict(),
                            "with_intrinsic_guides": with_intrinsic_guides,
                        }
                    }
                    if intrinsic_names
                    else {}
                ),
            }
        else:
            ready_rollout = {}
            for old_engine_id in old_regs_by_profile.values():
                if old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
            toolboxes.pop(tid, None)
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "remaining_request_count": len(manual_requests),
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "removed_tool_keys": keys,
            "toolbox_removed": not auto_requests and not manual_requests and not intrinsic_names,
            "removed_environment_keys": removed_environment_keys,
        }

    def _find_registration(self, engine_id: str) -> Optional[Dict[str, Any]]:
        eid = str(engine_id or "").strip()
        for row in self._read_engines():
            if str(row.get("engine_id") or "") == eid:
                return dict(row)
        return None

    def _probe_registration_reachability(
        self,
        item: Dict[str, Any],
        *,
        timeout_seconds: float = 0.35,
    ) -> Dict[str, Any]:
        checked_at = time.time()
        transport = str(item.get("worker_transport") or "").strip().lower()
        if transport != "ipc":
            return {
                "reachable": False,
                "checked_at": checked_at,
                "transport": transport or None,
                "probe": "unsupported_transport",
                "error": "reachability_probe_not_supported",
            }
        try:
            out = self._ipc_call(
                reg=item,
                payload={"kind": "hello", "engine_id": str(item.get("engine_id") or "")},
                timeout_seconds=max(0.1, float(timeout_seconds or 0.35)),
            )
            return {
                "reachable": str(out.get("status") or "").strip().lower() == "ok",
                "checked_at": checked_at,
                "transport": "ipc",
                "probe": "hello",
            }
        except Exception as exc:
            return {
                "reachable": False,
                "checked_at": checked_at,
                "transport": "ipc",
                "probe": "hello",
                "error": str(exc),
            }

    def discover_running(
        self,
        *,
        prune_stale: bool = True,
        include_progress: bool = False,
        include_reachability: bool = True,
        reachability_timeout_seconds: float = 0.35,
    ) -> Any:
        progress_events: List[Dict[str, Any]] = [
            self._progress_event("discover.read_registry", "running", "Reading managed engine registry"),
        ]
        rows = self._read_engines()
        out: List[Dict[str, Any]] = []
        stale_ids: List[str] = []
        now = time.time()
        reachable_count = 0
        for row in rows:
            item = dict(row)
            pid = int(item.get("pid") or 0)
            alive = self._pid_alive(pid)
            item["alive"] = alive
            item["uptime_seconds"] = max(0.0, now - float(item.get("spawned_at") or now))
            item["reachable"] = False
            if alive and include_reachability:
                reachability = self._probe_registration_reachability(
                    item,
                    timeout_seconds=reachability_timeout_seconds,
                )
                item["reachable"] = bool(reachability.get("reachable", False))
                item["reachability"] = reachability
                if item["reachable"]:
                    reachable_count += 1
            out.append(item)
            if not alive:
                stale_ids.append(str(item.get("engine_id") or ""))
        if prune_stale and stale_ids:
            keep = [r for r in rows if str(r.get("engine_id") or "") not in set(stale_ids)]
            self._write_engines(keep)
            out = [x for x in out if x.get("alive")]
        out.sort(key=lambda x: str(x.get("engine_id") or ""))
        progress_events.append(
            self._progress_event(
                "discover.complete",
                "completed",
                "Discovery complete",
                engines=len(out),
                reachable=reachable_count,
                stale_pruned=len(stale_ids) if prune_stale else 0,
            )
        )
        if include_progress:
            return {
                "status": "ok",
                "stage": "completed",
                "engines": out,
                "progress_events": progress_events,
            }
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
        env: Optional[Dict[str, str]] = None,
        worker_auth_token: Optional[str] = None,
        worker_auth_header: Optional[str] = None,
        worker_ipc_family: Optional[str] = None,
        worker_ipc_address: Optional[str] = None,
        worker_profile_class: Optional[str] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        sandbox_runtime: Optional[Dict[str, Any]] = None,
        executor_kind: Optional[str] = None,
        bundle: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, Any]] = None,
        tool_access: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
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
            "env": {str(k): str(v) for k, v in dict(env or {}).items()},
            "spawned_at": time.time(),
            "owner_host_pid": os.getpid(),
            "source": str(source or "engine_host_spawned"),
            "endpoint": "ipc://local",
            "worker_auth_token": str(worker_auth_token or "").strip() or None,
            "worker_auth_header": str(worker_auth_header or "").strip() or None,
            "worker_transport": "ipc",
            "worker_ipc_family": str(worker_ipc_family or "").strip() or None,
            "worker_ipc_address": str(worker_ipc_address or "").strip() or None,
            "worker_profile_class": self._normalize_worker_profile_class(worker_profile_class),
            "sandbox_policy": dict(sandbox_policy or {}) if isinstance(sandbox_policy, dict) else None,
            "sandbox_runtime": dict(sandbox_runtime or {}) if isinstance(sandbox_runtime, dict) else None,
            "executor_kind": str(executor_kind or "").strip() or None,
            "bundle": dict(bundle or {}) if isinstance(bundle, dict) else None,
            "environment": dict(environment or {}) if isinstance(environment, dict) else None,
            "tool_access": dict(tool_access or {}) if isinstance(tool_access, dict) else None,
            "capabilities": dict(capabilities or {}) if isinstance(capabilities, dict) else None,
            "log_path": str(self._engine_log_path(eid)),
        }
        rows = [r for r in self._read_engines() if str(r.get("engine_id") or "") != eid]
        rows.append(record)
        self._write_engines(rows)
        return record

    def _sandbox_fs_for_engine(self, engine_id: str) -> BrokeredFilesystem:
        reg = self._find_registration(str(engine_id or "").strip())
        if not reg:
            raise ValueError("engine_id is not registered")
        policy = WorkerSandboxPolicy.from_mapping(dict(reg.get("sandbox_policy") or {}))
        if not policy.enabled:
            raise PermissionError("sandbox_not_enabled")
        if not bool(policy.brokered_io.filesystem):
            raise PermissionError("brokered_filesystem_disabled")
        return BrokeredFilesystem(policy)

    def _sandbox_http_for_engine(self, engine_id: str) -> HostBrokeredHttpClient:
        reg = self._find_registration(str(engine_id or "").strip())
        if not reg:
            raise ValueError("engine_id is not registered")
        policy = WorkerSandboxPolicy.from_mapping(dict(reg.get("sandbox_policy") or {}))
        if not policy.enabled:
            raise PermissionError("sandbox_not_enabled")
        if not bool(policy.brokered_io.http):
            raise PermissionError("brokered_http_disabled")
        return HostBrokeredHttpClient(policy)

    def spawn(
        self,
        *,
        engine_id: str,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        worker_auth_token: Optional[str] = None,
        worker_auth_header: Optional[str] = None,
        worker_ipc_family: Optional[str] = None,
        worker_ipc_address: Optional[str] = None,
        worker_profile_class: Optional[str] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        executor_kind: Optional[str] = None,
        bundle: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, Any]] = None,
        tool_access: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not list(command or []):
            raise ValueError("command is required")
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        allocated_family, allocated_address = self._allocate_ipc_address(eid)
        ipc_family = str(worker_ipc_family or "").strip() or allocated_family
        ipc_address = str(worker_ipc_address or "").strip() or allocated_address
        auth_token = str(worker_auth_token or "").strip() or secrets.token_urlsafe(24)
        auth_header = str(worker_auth_header or "").strip() or "X-MP13-Host-Token"
        base_cmd = [str(x) for x in list(command or []) if str(x).strip()]
        if "--ipc-family" not in base_cmd:
            base_cmd.extend(["--ipc-family", ipc_family])
        if "--ipc-address" not in base_cmd:
            base_cmd.extend(["--ipc-address", ipc_address])
        merged_env = dict(os.environ) | {str(k): str(v) for k, v in dict(env or {}).items()}
        merged_env["MP13_ENGINE_HOST_TOKEN"] = auth_token
        merged_env["MP13_ENGINE_HOST_TOKEN_HEADER"] = auth_header
        merged_env["MP13_ENGINE_TRANSPORT"] = "ipc"
        merged_env["MP13_WORKER_IPC_FAMILY"] = ipc_family
        merged_env["MP13_WORKER_IPC_ADDRESS"] = ipc_address
        if str(executor_kind or "").strip() == "toolbox_executor_v1":
            merged_env["MP13_TOOLBOX_EXECUTOR_ENGINE_ID"] = eid
            merged_env["MP13_HOSTING_ENGINES_STATE_FILE"] = str(self.engines_state_file)
            merged_env["MP13_HOSTING_CONTROL_STATE_FILE"] = str(self.control_state_file)
        log_path = self._engine_log_path(str(engine_id or ""))
        normalized_sandbox = WorkerSandboxPolicy.from_mapping(sandbox_policy)
        launched = launch_worker_process(
            WorkerLaunchRequest(
                engine_id=eid,
                command=base_cmd,
                cwd=Path(str(cwd)).expanduser().resolve() if cwd else None,
                env=merged_env,
                log_path=log_path,
                sandbox_policy=normalized_sandbox,
            )
        )
        persisted_env = {str(k): str(v) for k, v in dict(env or {}).items()}
        for key in [
            "MP13_ENGINE_HOST_TOKEN",
            "MP13_ENGINE_HOST_TOKEN_HEADER",
            "MP13_ENGINE_TRANSPORT",
            "MP13_WORKER_IPC_FAMILY",
            "MP13_WORKER_IPC_ADDRESS",
        ]:
            persisted_env[key] = str(launched.persisted_env.get(key) or merged_env.get(key) or "")
        if str(executor_kind or "").strip() == "toolbox_executor_v1":
            for key in [
                "MP13_TOOLBOX_EXECUTOR_ENGINE_ID",
                "MP13_HOSTING_ENGINES_STATE_FILE",
                "MP13_HOSTING_CONTROL_STATE_FILE",
            ]:
                persisted_env[key] = str(launched.persisted_env.get(key) or merged_env.get(key) or "")
        return self.register_spawned(
            engine_id=eid,
            pid=int(launched.pid),
            command=list(launched.command),
            cwd=cwd,
            env=persisted_env,
            worker_auth_token=auth_token,
            worker_auth_header=auth_header,
            worker_ipc_family=ipc_family,
            worker_ipc_address=ipc_address,
            worker_profile_class=worker_profile_class,
            sandbox_policy=normalized_sandbox.to_dict(),
            sandbox_runtime=dict(launched.runtime),
            executor_kind=executor_kind,
            bundle=bundle,
            environment=environment,
            tool_access=tool_access,
            capabilities=capabilities,
        )

    def remove_registration(self, engine_id: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        rows = self._read_engines()
        kept = [r for r in rows if str(r.get("engine_id") or "") != eid]
        changed = len(kept) != len(rows)
        if changed:
            self._write_engines(kept)
        return {"engine_id": eid, "removed": changed}

    def sandbox_fs_list(self, *, engine_id: str, root_id: str, relative_path: Optional[str] = None) -> Dict[str, Any]:
        return self._sandbox_fs_for_engine(engine_id).list_dir(root_id=root_id, relative_path=relative_path)

    def sandbox_fs_read_text(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: str,
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        return self._sandbox_fs_for_engine(engine_id).read_text(
            root_id=root_id,
            relative_path=relative_path,
            encoding=encoding,
        )

    def sandbox_fs_write_text(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: str,
        text: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> Dict[str, Any]:
        return self._sandbox_fs_for_engine(engine_id).write_text(
            root_id=root_id,
            relative_path=relative_path,
            text=text,
            encoding=encoding,
            create_parents=create_parents,
        )

    def sandbox_fs_mkdir(
        self,
        *,
        engine_id: str,
        root_id: str,
        relative_path: str,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> Dict[str, Any]:
        return self._sandbox_fs_for_engine(engine_id).mkdir(
            root_id=root_id,
            relative_path=relative_path,
            parents=parents,
            exist_ok=exist_ok,
        )

    def sandbox_fs_stat(self, *, engine_id: str, root_id: str, relative_path: Optional[str] = None) -> Dict[str, Any]:
        return self._sandbox_fs_for_engine(engine_id).stat(root_id=root_id, relative_path=relative_path)

    def sandbox_http_fetch(
        self,
        *,
        engine_id: str,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body_b64: str = "",
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 1024 * 1024,
    ) -> Dict[str, Any]:
        out = self._sandbox_http_for_engine(engine_id).fetch(
            url=url,
            method=method,
            headers=headers,
            body_b64=body_b64,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
        )
        return {"engine_id": str(engine_id or ""), **dict(out or {})}

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
        env = {str(k): str(v) for k, v in dict(entry.get("env") or {}).items()}
        worker_auth_token = str(entry.get("worker_auth_token") or "").strip() or None
        worker_auth_header = str(entry.get("worker_auth_header") or "").strip() or None
        worker_ipc_family = str(entry.get("worker_ipc_family") or "").strip() or None
        worker_ipc_address = str(entry.get("worker_ipc_address") or "").strip() or None
        worker_profile_class = str(entry.get("worker_profile_class") or "").strip() or None
        sandbox_policy_raw = dict(entry.get("sandbox_policy") or {})
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
        normalized_sandbox = WorkerSandboxPolicy.from_mapping(sandbox_policy_raw)
        launched = launch_worker_process(
            WorkerLaunchRequest(
                engine_id=eid,
                command=command,
                cwd=Path(str(cwd)).expanduser().resolve() if cwd else None,
                env=(dict(os.environ) | env),
                log_path=self._engine_log_path(eid),
                sandbox_policy=normalized_sandbox,
            )
        )
        reg = self.register_spawned(
            engine_id=eid,
            pid=int(launched.pid),
            command=command,
            cwd=str(cwd) if cwd else None,
            env=env,
            worker_auth_token=worker_auth_token,
            worker_auth_header=worker_auth_header,
            worker_ipc_family=worker_ipc_family,
            worker_ipc_address=worker_ipc_address,
            worker_profile_class=worker_profile_class,
            sandbox_policy=normalized_sandbox.to_dict(),
            sandbox_runtime=dict(launched.runtime),
        )
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
        transport = str(reg.get("worker_transport") or "").strip().lower()
        if transport != "ipc":
            self._metrics_proxy_finish(
                eid,
                failed=True,
                error_message="ipc transport is required",
                method=m,
                path=req_path,
                started_at=req_started_at,
            )
            raise ValueError("ipc transport is required")
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
        worker_auth_header = str(reg.get("worker_auth_header") or "").strip()
        worker_auth_token = str(reg.get("worker_auth_token") or "").strip()
        if worker_auth_header and worker_auth_token:
            # Host-controlled channel proof. Client headers cannot override this.
            req_headers[worker_auth_header] = worker_auth_token
        try:
            out = self._proxy_request_via_ipc(
                reg=reg,
                engine_id=eid,
                method=m,
                path=req_path,
                query=query,
                headers=req_headers,
                body_b64=str(body_b64 or ""),
                timeout_seconds=timeout_seconds,
            )
            raw = base64.b64decode(str(out.get("body_b64") or "")) if str(out.get("body_b64") or "") else b""
            lim = min(
                max(1024, int(max_response_bytes or 1024 * 1024)),
                max(1024, int(traffic_policy.get("max_response_bytes") or (1024 * 1024))),
            )
            truncated = len(raw) > lim
            if truncated:
                raw = raw[:lim]
                out["body_b64"] = base64.b64encode(raw).decode("ascii")
                out["body_size"] = len(raw)
                out["truncated"] = True
            self._metrics_proxy_finish(
                eid,
                status_code=int(out.get("status_code") or 500),
                response_bytes=len(raw),
                http_error=bool(int(out.get("status_code") or 500) >= 400),
                failed=False,
                method=m,
                path=req_path,
                started_at=req_started_at,
                truncated=bool(out.get("truncated")),
                request_bytes=len(body_raw),
            )
            return out
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

    def proxy_rpc_call(
        self,
        *,
        engine_id: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        meth = str(method or "").strip()
        if not meth:
            raise ValueError("method is required")
        reg = self._require_ipc_registration(eid, command_label="proxy-rpc")
        out = self._ipc_call(
            reg=reg,
            payload={"kind": "rpc_call", "engine_id": eid, "method": meth, "params": dict(params or {})},
            timeout_seconds=timeout_seconds,
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "rpc_call_failed"))
        return dict(out or {})

    def proxy_rpc_open(
        self,
        *,
        engine_id: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_id: str,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        meth = str(method or "").strip()
        if not meth:
            raise ValueError("method is required")
        req_id = str(request_id or "").strip()
        if not req_id:
            raise ValueError("request_id is required")
        reg = self._require_ipc_registration(eid, command_label="proxy-rpc")
        out = self._ipc_call(
            reg=reg,
            payload={
                "kind": "stream_open",
                "engine_id": eid,
                "method": meth,
                "params": dict(params or {}),
                "request_id": req_id,
            },
            timeout_seconds=timeout_seconds,
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "rpc_open_failed"))
        return {"status": "ok", "engine_id": eid, "stream_id": str(out.get("stream_id") or ""), "request_id": req_id}

    def proxy_rpc_send(
        self,
        *,
        engine_id: str,
        stream_id: str,
        message: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        sid = str(stream_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        if not sid:
            raise ValueError("stream_id is required")
        reg = self._require_ipc_registration(eid, command_label="proxy-rpc")
        out = self._ipc_call(
            reg=reg,
            payload={"kind": "stream_send", "engine_id": eid, "stream_id": sid, "message": dict(message or {})},
            timeout_seconds=timeout_seconds,
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "rpc_send_failed"))
        return dict(out or {})

    def proxy_rpc_recv(
        self,
        *,
        engine_id: str,
        stream_id: str,
        timeout_seconds: float = 2.0,
        max_items: int = 64,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        sid = str(stream_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        if not sid:
            raise ValueError("stream_id is required")
        reg = self._require_ipc_registration(eid, command_label="proxy-rpc")
        out = self._ipc_call(
            reg=reg,
            payload={
                "kind": "stream_recv",
                "engine_id": eid,
                "stream_id": sid,
                "timeout_seconds": float(timeout_seconds or 2.0),
                "max_items": int(max_items or 64),
            },
            timeout_seconds=max(1.0, float(timeout_seconds or 2.0) + 1.0),
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "rpc_recv_failed"))
        return dict(out or {})

    def proxy_rpc_close(
        self,
        *,
        engine_id: str,
        stream_id: str,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        sid = str(stream_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        if not sid:
            raise ValueError("stream_id is required")
        reg = self._require_ipc_registration(eid, command_label="proxy-rpc")
        out = self._ipc_call(
            reg=reg,
            payload={"kind": "stream_close", "engine_id": eid, "stream_id": sid},
            timeout_seconds=timeout_seconds,
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "rpc_close_failed"))
        return dict(out or {})

    def proxy_stream_open(
        self,
        *,
        engine_id: str,
        tool: str = "run-inference",
        arguments: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        args = dict(arguments or {})
        req_id = str(args.get("request_id") or "").strip() or secrets.token_hex(12)
        out = self.proxy_rpc_open(
            engine_id=str(engine_id or ""),
            method=str(tool or "run-inference"),
            params=args,
            request_id=req_id,
            timeout_seconds=timeout_seconds,
        )
        out["worker_transport"] = "ipc"
        return out

    def proxy_stream_send(
        self,
        *,
        engine_id: str,
        stream_id: str,
        message: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        return self.proxy_rpc_send(
            engine_id=str(engine_id or ""),
            stream_id=str(stream_id or ""),
            message=dict(message or {}),
            timeout_seconds=timeout_seconds,
        )

    def proxy_stream_recv(
        self,
        *,
        engine_id: str,
        stream_id: str,
        timeout_seconds: float = 2.0,
        max_items: int = 64,
    ) -> Dict[str, Any]:
        return self.proxy_rpc_recv(
            engine_id=str(engine_id or ""),
            stream_id=str(stream_id or ""),
            timeout_seconds=float(timeout_seconds or 2.0),
            max_items=int(max_items or 64),
        )

    def proxy_stream_close(
        self,
        *,
        engine_id: str,
        stream_id: str,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        return self.proxy_rpc_close(
            engine_id=str(engine_id or ""),
            stream_id=str(stream_id or ""),
            timeout_seconds=timeout_seconds,
        )

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
        exclusive: Optional[bool] = None,
        force_override: bool = False,
        force_override_reason: Optional[str] = None,
        force_override_emergency: bool = False,
        actor_id: Optional[str] = None,
        peer_host: Optional[str] = None,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        bid = str(actor_id or "").strip() or self._normalize_backend_id(backend_id)
        if not eid:
            raise ValueError("engine_id is required")
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        effective_exclusive = bool(exclusive) if exclusive is not None else (self._endpoint_mode_default(cfg) == "exclusive")
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
        reason = self._normalize_force_override_reason(force_override_reason)
        emergency = bool(force_override_emergency)
        if bool(force_override):
            if reason not in VALID_FORCE_OVERRIDE_REASONS:
                self._append_claim_audit_event(
                    control,
                    event_type="claim_deny",
                    command="claim-engine",
                    scope="engine",
                    resource_kind="engine",
                    resource_id=eid,
                    actor_id=bid,
                    decision="deny",
                    code="force_override_reason_required",
                    transition=None,
                    mode="exclusive" if effective_exclusive else "shared",
                    peer_host=peer_host,
                    owners_before=owners_before,
                    owners_after=owners_before,
                    details={"force_override": True, "allowed_reasons": sorted(list(VALID_FORCE_OVERRIDE_REASONS))},
                    severity="high",
                )
                self._write_control(control)
                out = self._deny_payload(
                    "force_override_reason_required",
                    "force override reason is required",
                    engine_id=eid,
                    backend_id=bid,
                    allowed_reasons=sorted(list(VALID_FORCE_OVERRIDE_REASONS)),
                )
                out.update({"engine_id": eid, "backend_id": bid, "mode": "exclusive" if effective_exclusive else "shared"})
                return out
            if emergency and reason not in EMERGENCY_FORCE_OVERRIDE_REASONS:
                self._append_claim_audit_event(
                    control,
                    event_type="claim_deny",
                    command="claim-engine",
                    scope="engine",
                    resource_kind="engine",
                    resource_id=eid,
                    actor_id=bid,
                    decision="deny",
                    code="force_override_emergency_reason_invalid",
                    transition=None,
                    mode="exclusive" if effective_exclusive else "shared",
                    peer_host=peer_host,
                    owners_before=owners_before,
                    owners_after=owners_before,
                    details={"force_override": True, "force_override_emergency": True, "force_override_reason": reason},
                    severity="high",
                )
                self._write_control(control)
                out = self._deny_payload(
                    "force_override_emergency_reason_invalid",
                    "force override emergency reason is invalid",
                    engine_id=eid,
                    backend_id=bid,
                    allowed_emergency_reasons=sorted(list(EMERGENCY_FORCE_OVERRIDE_REASONS)),
                )
                out.update({"engine_id": eid, "backend_id": bid, "mode": "exclusive" if effective_exclusive else "shared"})
                return out
        if effective_exclusive:
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
                    severity="normal",
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
                        severity="normal",
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
        self._clear_ownership_change_notice(control, bid)
        if transition == "force_override" and displaced:
            self._record_ownership_change_notices(
                control,
                displaced_owners=displaced,
                replaced_by=bid,
                scope="engine",
                resource_kind="engine",
                resource_id=eid,
                reason=reason or None,
                emergency=emergency,
                peer_host=peer_host,
                command="claim-engine",
            )
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
            mode="exclusive" if effective_exclusive else "shared",
            peer_host=peer_host,
            owners_before=owners_before,
            owners_after=list(claim.get("owners") or []),
            details={
                "orphan_owners": orphan_owners,
                "force_override": bool(force_override),
                "force_override_reason": reason or None,
                "force_override_emergency": emergency,
            },
            severity="high" if bool(force_override) else "normal",
        )
        self._write_control(control)
        return {
            "scope": "engine",
            "engine_id": eid,
            "backend_id": bid,
            "mode": "exclusive" if effective_exclusive else "shared",
            "owners": list(claim.get("owners") or []),
            "exclusive_owner": claim.get("exclusive_owner"),
            "displaced_backends": displaced,
            "revoked_tokens": revoked,
            "transition": transition,
            "force_override_reason": reason or None,
            "force_override_emergency": emergency if bool(force_override) else False,
        }

    def claim_endpoint(
        self,
        *,
        backend_id: Optional[str],
        exclusive: Optional[bool] = None,
        force_override: bool = False,
        force_override_reason: Optional[str] = None,
        force_override_emergency: bool = False,
        actor_id: Optional[str] = None,
        peer_host: Optional[str] = None,
    ) -> Dict[str, Any]:
        bid = str(actor_id or "").strip() or self._normalize_backend_id(backend_id)
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        effective_exclusive = bool(exclusive) if exclusive is not None else (self._endpoint_mode_default(cfg) == "exclusive")
        endpoint = dict(control.get("endpoint_claim") or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        owners_before = [str(x or "").strip() for x in list(endpoint.get("owners") or []) if str(x or "").strip()]
        active_owners, orphan_owners = self._active_and_orphan_owners(control, owners_before)
        owners = set(active_owners)
        displaced: List[str] = []
        revoked = 0
        transition = "claimed"
        reason = self._normalize_force_override_reason(force_override_reason)
        emergency = bool(force_override_emergency)
        if bool(force_override):
            if reason not in VALID_FORCE_OVERRIDE_REASONS:
                self._append_claim_audit_event(
                    control,
                    event_type="claim_deny",
                    command="claim-endpoint",
                    scope="endpoint",
                    resource_kind="endpoint",
                    resource_id="*",
                    actor_id=bid,
                    decision="deny",
                    code="force_override_reason_required",
                    transition=None,
                    mode="exclusive" if effective_exclusive else "shared",
                    peer_host=peer_host,
                    owners_before=owners_before,
                    owners_after=owners_before,
                    details={"force_override": True, "allowed_reasons": sorted(list(VALID_FORCE_OVERRIDE_REASONS))},
                    severity="high",
                )
                self._write_control(control)
                out = self._deny_payload(
                    "force_override_reason_required",
                    "force override reason is required",
                    backend_id=bid,
                    allowed_reasons=sorted(list(VALID_FORCE_OVERRIDE_REASONS)),
                )
                out.update({"scope": "endpoint", "backend_id": bid, "mode": "exclusive" if effective_exclusive else "shared"})
                return out
            if emergency and reason not in EMERGENCY_FORCE_OVERRIDE_REASONS:
                self._append_claim_audit_event(
                    control,
                    event_type="claim_deny",
                    command="claim-endpoint",
                    scope="endpoint",
                    resource_kind="endpoint",
                    resource_id="*",
                    actor_id=bid,
                    decision="deny",
                    code="force_override_emergency_reason_invalid",
                    transition=None,
                    mode="exclusive" if effective_exclusive else "shared",
                    peer_host=peer_host,
                    owners_before=owners_before,
                    owners_after=owners_before,
                    details={"force_override": True, "force_override_emergency": True, "force_override_reason": reason},
                    severity="high",
                )
                self._write_control(control)
                out = self._deny_payload(
                    "force_override_emergency_reason_invalid",
                    "force override emergency reason is invalid",
                    backend_id=bid,
                    allowed_emergency_reasons=sorted(list(EMERGENCY_FORCE_OVERRIDE_REASONS)),
                )
                out.update({"scope": "endpoint", "backend_id": bid, "mode": "exclusive" if effective_exclusive else "shared"})
                return out
        if effective_exclusive:
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
                    severity="normal",
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
                        severity="normal",
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
        self._clear_ownership_change_notice(control, bid)
        if transition == "force_override" and displaced:
            self._record_ownership_change_notices(
                control,
                displaced_owners=displaced,
                replaced_by=bid,
                scope="endpoint",
                resource_kind="endpoint",
                resource_id="*",
                reason=reason or None,
                emergency=emergency,
                peer_host=peer_host,
                command="claim-endpoint",
            )
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
            mode="exclusive" if effective_exclusive else "shared",
            peer_host=peer_host,
            owners_before=owners_before,
            owners_after=list(endpoint.get("owners") or []),
            details={
                "orphan_owners": orphan_owners,
                "force_override": bool(force_override),
                "force_override_reason": reason or None,
                "force_override_emergency": emergency,
            },
            severity="high" if bool(force_override) else "normal",
        )
        self._write_control(control)
        return {
            "scope": "endpoint",
            "backend_id": bid,
            "mode": "exclusive" if effective_exclusive else "shared",
            "owners": list(endpoint.get("owners") or []),
            "exclusive_owner": endpoint.get("exclusive_owner"),
            "displaced_backends": displaced,
            "revoked_tokens": revoked,
            "transition": transition,
            "force_override_reason": reason or None,
            "force_override_emergency": emergency if bool(force_override) else False,
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
        exclusive: Optional[bool] = None,
        force_override: bool = False,
        force_override_reason: Optional[str] = None,
        force_override_emergency: bool = False,
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
                force_override_reason=force_override_reason,
                force_override_emergency=force_override_emergency,
                actor_id=actor_id,
                peer_host=peer_host,
            )
        bid = str(actor_id or "").strip() or self._normalize_backend_id(backend_id)
        rkey = self._resource_key(rkind, rid)
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        effective_exclusive = bool(exclusive) if exclusive is not None else (self._endpoint_mode_default(cfg) == "exclusive")
        claims = dict(control.get("resource_claims") or {})
        claim = dict(claims.get(rkey) or {"owners": [], "exclusive_owner": None, "claimed_at": 0.0})
        owners_before = [str(x or "").strip() for x in list(claim.get("owners") or []) if str(x or "").strip()]
        active_owners, orphan_owners = self._active_and_orphan_owners(control, owners_before)
        owners = set(active_owners)
        displaced: List[str] = []
        revoked = 0
        transition = "claimed"
        reason = self._normalize_force_override_reason(force_override_reason)
        emergency = bool(force_override_emergency)
        if bool(force_override):
            if reason not in VALID_FORCE_OVERRIDE_REASONS:
                self._append_claim_audit_event(
                    control,
                    event_type="claim_deny",
                    command="claim-resource",
                    scope="resource",
                    resource_kind=rkind,
                    resource_id=rid,
                    actor_id=bid,
                    decision="deny",
                    code="force_override_reason_required",
                    transition=None,
                    mode="exclusive" if effective_exclusive else "shared",
                    peer_host=peer_host,
                    owners_before=owners_before,
                    owners_after=owners_before,
                    details={"force_override": True, "allowed_reasons": sorted(list(VALID_FORCE_OVERRIDE_REASONS))},
                    severity="high",
                )
                self._write_control(control)
                out = self._deny_payload(
                    "force_override_reason_required",
                    "force override reason is required",
                    resource_kind=rkind,
                    resource_id=rid,
                    backend_id=bid,
                    allowed_reasons=sorted(list(VALID_FORCE_OVERRIDE_REASONS)),
                )
                out.update({"scope": "resource", "resource_kind": rkind, "resource_id": rid, "backend_id": bid, "mode": "exclusive" if effective_exclusive else "shared"})
                return out
            if emergency and reason not in EMERGENCY_FORCE_OVERRIDE_REASONS:
                self._append_claim_audit_event(
                    control,
                    event_type="claim_deny",
                    command="claim-resource",
                    scope="resource",
                    resource_kind=rkind,
                    resource_id=rid,
                    actor_id=bid,
                    decision="deny",
                    code="force_override_emergency_reason_invalid",
                    transition=None,
                    mode="exclusive" if effective_exclusive else "shared",
                    peer_host=peer_host,
                    owners_before=owners_before,
                    owners_after=owners_before,
                    details={"force_override": True, "force_override_emergency": True, "force_override_reason": reason},
                    severity="high",
                )
                self._write_control(control)
                out = self._deny_payload(
                    "force_override_emergency_reason_invalid",
                    "force override emergency reason is invalid",
                    resource_kind=rkind,
                    resource_id=rid,
                    backend_id=bid,
                    allowed_emergency_reasons=sorted(list(EMERGENCY_FORCE_OVERRIDE_REASONS)),
                )
                out.update({"scope": "resource", "resource_kind": rkind, "resource_id": rid, "backend_id": bid, "mode": "exclusive" if effective_exclusive else "shared"})
                return out
        if effective_exclusive:
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
                    severity="normal",
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
                        severity="normal",
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
        self._clear_ownership_change_notice(control, bid)
        if transition == "force_override" and displaced:
            self._record_ownership_change_notices(
                control,
                displaced_owners=displaced,
                replaced_by=bid,
                scope="resource",
                resource_kind=rkind,
                resource_id=rid,
                reason=reason or None,
                emergency=emergency,
                peer_host=peer_host,
                command="claim-resource",
            )
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
            mode="exclusive" if effective_exclusive else "shared",
            peer_host=peer_host,
            owners_before=owners_before,
            owners_after=list(claim.get("owners") or []),
            details={
                "orphan_owners": orphan_owners,
                "force_override": bool(force_override),
                "force_override_reason": reason or None,
                "force_override_emergency": emergency,
            },
            severity="high" if bool(force_override) else "normal",
        )
        self._write_control(control)
        return {
            "scope": "resource",
            "resource_kind": rkind,
            "resource_id": rid,
            "backend_id": bid,
            "mode": "exclusive" if effective_exclusive else "shared",
            "owners": list(claim.get("owners") or []),
            "exclusive_owner": claim.get("exclusive_owner"),
            "displaced_backends": displaced,
            "revoked_tokens": revoked,
            "transition": transition,
            "force_override_reason": reason or None,
            "force_override_emergency": emergency if bool(force_override) else False,
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
