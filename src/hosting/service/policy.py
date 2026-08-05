"""Command authorization and daemon claim policy helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .constants import (
    EMERGENCY_FORCE_OVERRIDE_REASONS,
    ROLE_ADMIN,
    ROLE_CONFIG_EDITOR,
    ROLE_MODEL_USER_WITH_MODEL_CONTROL,
    ROLE_WORKER_USER,
    VALID_FORCE_OVERRIDE_REASONS,
)


class PolicyMixin:
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
        # Bootstrap: allow first-key provisioning only for local-only access.
        if c in {"auth-upsert-key", "auth-status"} and keys_count == 0:
            if self._requires_ssh_binding(cfg):
                self._metrics_auth_denied("zero_key_bootstrap_local_only")
                raise PermissionError("zero_key_bootstrap_local_only")
            return
        if c in {"auth-issue-session"}:
            # Session issuance authenticates with key_id/key_secret in payload.
            return
        if c in {"auth-begin-challenge", "auth-complete-challenge"}:
            # Challenge issuance/completion perform their own key-based verification.
            return
        if c in {"auth-validate-session", "auth-renew-session"}:
            # Validation/renewal authenticate by proving possession of the token
            # being checked.
            return
        token = self._extract_session_token(payload)
        if not token:
            self._metrics_auth_denied("session_token_required")
            raise PermissionError("session_token_required")
        presented_ssh_binding = dict((payload or {}).get("_ssh_session_binding") or {})
        if c in {
            "discover-running",
            "spawn",
            "workflow-js-environment-spec",
            "workflow-js-ensure",
            "workflow-js-execute",
            "workflow-js-action-describe",
            "workflow-js-action-execute",
            "workflow-js-instance-create",
            "workflow-js-instance-execute",
            "workflow-js-instance-close",
            "workflow-js-instance-list",
            "workflow-js-resources",
            "workflow-js-set-capacity",
            "workflow-js-stream-open",
            "workflow-js-event-subscribe",
            "workflow-js-stream-send",
            "workflow-js-stream-close",
            "workflow-python-environment-spec",
            "workflow-python-prepare-environment",
            "workflow-python-lock-environment",
            "workflow-python-verify-environment",
            "workflow-python-install-environment",
            "workflow-python-verify-install-receipt",
            "sandbox-state-snapshot",
            "sandbox-state-restore",
            "workflow-artifact-recovery-inspect",
            "workflow-artifact-recovery-claim",
            "workflow-artifact-recovery-cleanup",
            "workflow-python-ensure",
            "workflow-python-execute",
            "workflow-python-action-describe",
            "workflow-python-action-execute",
            "workflow-python-instance-create",
            "workflow-python-instance-execute",
            "workflow-python-instance-close",
            "workflow-python-instance-list",
            "workflow-python-resources",
            "workflow-python-set-capacity",
            "workflow-python-stream-open",
            "workflow-python-event-subscribe",
            "workflow-python-stream-send",
            "workflow-python-stream-close",
            "get-registration",
            "shutdown",
            "ensure-running",
            "unload-model",
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
            "auth-status",
            "hosting-setup-status",
            "hosting-secure-state-status",
            "auth-revoke-key",
            "auth-list-keys",
            "auth-list-sessions",
            "list-live-consumers",
            "auth-list-issued-tokens",
            "auth-audit-list",
            "auth-validate-session",
            "auth-renew-session",
            "auth-revoke-session",
            "host-capability-session-register",
            "host-capability-session-list",
            "host-capability-session-close",
            "host-capability-session-renew",
            "host-capability-session-revoke",
            "host-capability-audit-list",
            "host-metrics",
            "op-start",
            "op-status",
            "op-cancel",
            "set-endpoint-mode-override",
            "get-endpoint-mode-effective",
            "get-lifecycle-policy-effective",
            "toolbox-describe",
            "toolbox-gate",
            "toolbox-execute",
            "hosted-operation-status",
            "hosted-operation-result",
            "hosted-operation-cancel",
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
            "unload-model",
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
            "workflow-python-ensure",
            "workflow-python-execute",
            "workflow-python-resources",
            "workflow-python-set-capacity",
            "workflow-python-stream-open",
            "workflow-python-event-subscribe",
            "workflow-python-stream-send",
            "workflow-python-stream-close",
            "workflow-artifact-recovery-inspect",
            "workflow-artifact-recovery-claim",
            "workflow-artifact-recovery-cleanup",
            "workflow-js-ensure",
            "workflow-js-resources",
            "workflow-js-set-capacity",
            "workflow-js-stream-open",
            "workflow-js-event-subscribe",
            "workflow-js-stream-send",
            "workflow-js-stream-close",
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
