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
