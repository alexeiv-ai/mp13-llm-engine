"""Control configuration and lifecycle policy helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .constants import (
    DAEMON_VERSION,
    LIFECYCLE_PROFILE_DETACHED,
    LIFECYCLE_PROFILE_FOREGROUND,
    LIFECYCLE_PROFILE_SERVICE,
    ROLE_ADMIN,
    VALID_LIFECYCLE_PROFILES,
)


class ControlMixin:
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
        if ControlMixin._endpoint_mode_default(cfg) != "exclusive":
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
