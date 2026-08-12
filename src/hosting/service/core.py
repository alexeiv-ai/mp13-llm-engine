"""Core utility helpers for the engine host service."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .._process_utils import pid_alive


class CoreMixin:
    @staticmethod
    def _pid_alive(pid: int) -> bool:
        return pid_alive(pid)

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
            "auth_session_validate": True,
            "auth_session_adopt": True,
            "auth_session_list": True,
            "auth_audit_list": True,
            "hosting_setup_status_v1": True,
            "model_runtime_status_v1": True,
            "secure_state_status_v1": True,
            "hosted_operations_v1": True,
            "hosted_result_artifacts_v1": True,
            "approval_callback_leases_v1": True,
            "explicit_capability_provider_identity_v1": True,
            "capability_authority_leases_v1": True,
            "package_artifact_ingress_v1": True,
            "package_locks_v1": True,
            "environment_management_v1": True,
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
