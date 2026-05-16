"""State helpers for the engine host service."""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from ..secure_state import secure_state_status
from .constants import LIFECYCLE_PROFILE_DETACHED


class StateMixin:
    def _ensure_engine_runtime_state(self) -> None:
        if hasattr(self, "_runtime_engines"):
            return
        self._runtime_engines_lock = threading.RLock()
        self._runtime_engines: List[Dict[str, Any]] = []

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
        self._ensure_engine_runtime_state()
        with self._runtime_engines_lock:
            return [
                self._normalize_engine_registration(dict(row or {}))
                for row in list(self._runtime_engines or [])
                if isinstance(row, dict)
            ]

    def _write_engines(self, rows: List[Dict[str, Any]]) -> None:
        self._ensure_engine_runtime_state()
        with self._runtime_engines_lock:
            self._runtime_engines = [
                self._normalize_engine_registration(dict(row or {}))
                for row in list(rows or [])
                if isinstance(row, dict)
            ]

    @staticmethod
    def _registration_binding_id(engine_id: str, config_path: str) -> str:
        import hashlib

        raw = f"{str(engine_id or '').strip()}|{str(config_path or '').strip()}"
        digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return f"binding-{digest}"

    @staticmethod
    def _registration_path_value(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            return str(Path(raw).expanduser().resolve())
        except Exception:
            return raw

    def _normalize_engine_registration(self, row: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(row or {})
        engine_id = str(record.get("engine_id") or "").strip()
        worker_id = str(record.get("worker_id") or "").strip() or engine_id
        if worker_id:
            record["worker_id"] = worker_id
        model_instance_id = str(record.get("model_instance_id") or "").strip() or engine_id
        if model_instance_id:
            record["model_instance_id"] = model_instance_id

        env = dict(record.get("env") or {}) if isinstance(record.get("env"), dict) else {}
        model_path = str(record.get("model_path") or env.get("MP13_MODEL_PATH") or "").strip()
        config_path = str(record.get("config_path") or env.get("MP13_ENGINE_CONFIG_PATH") or "").strip()
        canonical_model_path = str(record.get("canonical_model_path") or "").strip() or self._registration_path_value(model_path)
        canonical_config_path = str(record.get("canonical_config_path") or "").strip() or self._registration_path_value(config_path)

        has_loaded_models_field = "loaded_models" in record
        loaded_models = [dict(item or {}) for item in list(record.get("loaded_models") or []) if isinstance(item, dict)]
        if not has_loaded_models_field and not loaded_models and model_path and model_instance_id:
            loaded_models = [
                {
                    "model_instance_id": model_instance_id,
                    "engine_id": engine_id,
                    "model_path": model_path,
                    "canonical_model_path": canonical_model_path,
                    "loaded_at": float(record.get("spawned_at") or time.time()),
                    "runtime_profile": str(record.get("worker_profile_class") or "").strip() or "model",
                    "config_binding_ids": [],
                }
            ]

        has_config_bindings_field = "config_bindings" in record
        config_bindings = [dict(item or {}) for item in list(record.get("config_bindings") or []) if isinstance(item, dict)]
        if not has_config_bindings_field and not config_bindings and config_path and engine_id:
            binding_id = self._registration_binding_id(engine_id, canonical_config_path or config_path)
            config_bindings = [
                {
                    "config_binding_id": binding_id,
                    "engine_id": engine_id,
                    "model_instance_id": model_instance_id,
                    "config_path": config_path,
                    "canonical_config_path": canonical_config_path,
                    "created_at": float(record.get("spawned_at") or time.time()),
                }
            ]
            for model in loaded_models:
                ids = [str(x) for x in list(model.get("config_binding_ids") or []) if str(x).strip()]
                if binding_id not in ids:
                    ids.append(binding_id)
                model["config_binding_ids"] = ids

        if loaded_models:
            record["loaded_models"] = loaded_models
        else:
            record.setdefault("loaded_models", [])
        if config_bindings:
            record["config_bindings"] = config_bindings
        else:
            record.setdefault("config_bindings", [])
        if model_path:
            record.setdefault("model_path", model_path)
        if canonical_model_path:
            record.setdefault("canonical_model_path", canonical_model_path)
        if config_path:
            record.setdefault("config_path", config_path)
        if canonical_config_path:
            record.setdefault("canonical_config_path", canonical_config_path)
        return record

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

    def hosting_secure_state_status(self) -> Dict[str, Any]:
        layout = self._control_layout()
        bootstrap_root = self.hosting_root / "bootstrap"
        files = {
            "access_control": layout["access_control"],
            "keys": layout["keys"],
            "sessions": layout["sessions"],
            "challenges": layout["challenges"],
            "runtime_state": layout["runtime_state"],
            "auth_audit": layout["auth_audit"],
            "claim_audit": layout["claim_audit"],
            "bootstrap_state": bootstrap_root / "bootstrap_state.json",
            "client_key_map": bootstrap_root / "client_key_map.json",
        }
        statuses = {name: secure_state_status(path) for name, path in files.items()}
        encrypted_count = sum(1 for row in statuses.values() if bool(row.get("encrypted")))
        locked_count = sum(1 for row in statuses.values() if bool(row.get("locked")))
        plaintext_count = sum(1 for row in statuses.values() if str(row.get("state") or "") == "plaintext")
        return {
            "status": "ok",
            "hosting_root": str(self.hosting_root),
            "encryption_enabled": False,
            "daemon_secure_state_read_enabled": False,
            "startup_env_names": ["MP13_SECURE_STATE_KEY", "MP13_HOSTING_SECURE_STATE_KEY"],
            "files": statuses,
            "summary": {
                "encrypted_count": encrypted_count,
                "locked_count": locked_count,
                "plaintext_count": plaintext_count,
                "missing_count": sum(1 for row in statuses.values() if str(row.get("state") or "") == "missing"),
            },
        }

    def hosting_setup_summary(self) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        keys = dict(auth.get("keys") or {})
        admin_key_ids = sorted(
            str(key_id).strip()
            for key_id, meta in keys.items()
            if str(key_id).strip() and str((meta or {}).get("role") or "").strip().lower() == "admin"
        )
        access_profile = dict(cfg.get("access_profile") or {})
        return {
            "status": "ok",
            "hosting_root": str(self.hosting_root),
            "configured": bool((self.hosting_root / "access_control.json").exists() or keys),
            "connectivity_mode": str(access_profile.get("connectivity_mode") or "local_only"),
            "endpoint_mode_default": str(cfg.get("endpoint_mode_default") or "exclusive"),
            "lifecycle_profile": self._normalize_lifecycle_profile(cfg.get("lifecycle_profile")),
            "lifecycle_policy": self._normalize_lifecycle_policy(
                self._normalize_lifecycle_profile(cfg.get("lifecycle_profile")),
                dict(cfg.get("lifecycle_policy") or {}),
            ),
            "require_auth": bool(cfg.get("require_auth", False)),
            "admin_key_count": len(admin_key_ids),
            "admin_key_ids": admin_key_ids,
            "keys_count": len(keys),
            "sessions_count": len(dict(auth.get("sessions") or {})),
            "secure_state": self.hosting_secure_state_status(),
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
