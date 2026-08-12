"""Authentication helper methods for the engine host service."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._process_utils import hidden_subprocess_kwargs
from ..client_realm import secret_record_path
from .constants import (
    DAEMON_VERSION,
    ROLE_ADMIN,
    ROLE_CONFIG_EDITOR,
    ROLE_DIAGNOSTIC_USER,
    ROLE_MODEL_USER,
    ROLE_MODEL_USER_WITH_MODEL_CONTROL,
    ROLE_TRANSPORT,
    ROLE_WORKER_USER,
    VALID_AUTH_ROLES,
    ROLE_DEPENDENCY_APPROVER,
)

_WORKFLOW_PYTHON_COMMANDS = {
    "workflow-python-environment-spec",
    "workflow-python-prepare-environment",
    "workflow-python-lock-environment",
    "workflow-python-verify-environment",
    "workflow-python-install-environment",
    "workflow-python-verify-install-receipt",
    "sandbox-state-snapshot",
    "sandbox-state-restore",
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
    "workflow-python-stream-status",
    "workflow-python-stream-send",
    "workflow-python-stream-close",
}

_WORKFLOW_PYTHON_OBSERVE_COMMANDS = {
    "workflow-python-environment-spec",
    "workflow-python-verify-environment",
    "workflow-python-verify-install-receipt",
    "sandbox-state-snapshot",
    "workflow-python-action-describe",
    "workflow-python-instance-list",
    "workflow-python-resources",
    "workflow-python-event-subscribe",
    "workflow-python-stream-status",
}

_TOOLBOX_TEMPLATE_CONSUMER_COMMANDS = {
    "environment-template-list",
    "environment-template-describe",
}

_TOOLBOX_TEMPLATE_ADMIN_COMMANDS = {
    "environment-template-construct",
    "environment-template-activate",
    "environment-template-replace",
    "environment-template-deprecate",
    "environment-template-revoke",
    "environment-template-prewarm",
}

_TOOLBOX_DEFINITION_COMMANDS = {
    "toolbox-get-definition",
    "toolbox-plan-definition",
    "toolbox-confirm-definition-plan",
    "toolbox-apply-definition",
}

_TOOLBOX_DEPENDENCY_APPROVAL_COMMANDS = {
    "toolbox-approve-confirmed-definition-plan",
}

_TOOLBOX_ENVIRONMENT_ADMIN_COMMANDS = {
    "environment-remove",
    "environment-reference-list",
    "environment-reference-release",
    "environment-execution-begin",
    "environment-execution-end",
    "environment-gc",
}

_PACKAGE_UPLOAD_COMMANDS = {
    "package-artifact-upload-begin",
    "package-artifact-upload-chunk",
    "package-artifact-upload-status",
    "package-artifact-upload-cancel",
    "package-artifact-upload-commit",
}

_PACKAGE_LOCK_COMMANDS = {"package-lock-create"}

_WORKFLOW_JS_COMMANDS = {
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
    "workflow-js-stream-status",
    "workflow-js-stream-send",
    "workflow-js-stream-close",
}

_WORKFLOW_JS_OBSERVE_COMMANDS = {
    "workflow-js-environment-spec",
    "workflow-js-action-describe",
    "workflow-js-instance-list",
    "workflow-js-resources",
    "workflow-js-event-subscribe",
    "workflow-js-stream-status",
}

_WORKFLOW_ARTIFACT_COMMANDS = {
    "workflow-artifact-recovery-inspect",
    "workflow-artifact-recovery-claim",
    "workflow-artifact-recovery-cleanup",
}

_WORKFLOW_ARTIFACT_OBSERVE_COMMANDS = {
    "workflow-artifact-recovery-inspect",
}

_HOST_CAPABILITY_SESSION_COMMANDS = {
    "host-capability-session-register",
    "host-capability-session-list",
    "host-capability-session-close",
    "host-capability-session-renew",
    "host-capability-session-revoke",
    "host-capability-audit-list",
}


class AuthMixin:
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
        verifier_identity = f"mp13-key-{hashlib.sha256(kid.encode('utf-8')).hexdigest()[:24]}"
        try:
            with tempfile.TemporaryDirectory(prefix="host_auth_") as td:
                tdp = Path(td)
                data_file = tdp / "challenge.txt"
                sig_file = tdp / "challenge.sig"
                allowed_file = tdp / "allowed_signers"
                data_file.write_text(ch, encoding="utf-8")
                sig_file.write_text(sig, encoding="utf-8")
                allowed_file.write_text(f"{verifier_identity} {pub}\n", encoding="utf-8")
                proc = subprocess.run(  # noqa: S603
                    [
                        "ssh-keygen",
                        "-Y",
                        "verify",
                        "-f",
                        str(allowed_file),
                        "-I",
                        verifier_identity,
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
                    **hidden_subprocess_kwargs(),
                )
                return int(proc.returncode) == 0
        except Exception:
            return False

    def _extract_session_token(self, payload: Optional[Dict[str, Any]]) -> str:
        p = dict(payload or {})
        token = str(p.get("session_token") or p.get("auth_token") or "").strip()
        return token

    def _local_private_key_custody_metadata(self) -> List[Dict[str, Any]]:
        keys_file = (self.hosting_root / "keyring" / "keys.json").resolve()
        if not keys_file.exists():
            return []
        try:
            payload = json.loads(keys_file.read_text(encoding="utf-8"))
        except Exception:
            return []
        rows: List[Dict[str, Any]] = []
        default_config_dir = self.hosting_root.parent.resolve()
        for key_id, row_value in sorted(dict(payload.get("keys") or {}).items()):
            row = dict(row_value or {})
            secret_id = str(row.get("private_key_secret_id") or "").strip()
            secret_realm = str(row.get("private_key_secret_realm") or "default").strip() or "default"
            secret_path = None
            secret_exists = None
            secret_encryption = None
            secret_protection = str(row.get("private_key_protection") or "").strip() or None
            export_path_raw = str(row.get("private_key_export_path") or "").strip()
            export_path = Path(export_path_raw).expanduser().resolve() if export_path_raw else None
            if secret_id:
                secret_path = secret_record_path(
                    (default_config_dir / "hosting_client" / secret_realm).resolve(),
                    secret_id,
                )
                secret_exists = secret_path.exists()
                if secret_exists:
                    try:
                        secret_payload = json.loads(secret_path.read_text(encoding="utf-8"))
                        secret_encryption = str(secret_payload.get("encryption") or "").strip() or None
                        metadata = dict(secret_payload.get("metadata") or {})
                        secret_protection = str(metadata.get("private_key_protection") or secret_protection or "").strip() or None
                    except Exception:
                        secret_encryption = None
            key_origin = str(row.get("key_origin") or row.get("key_source") or "imported").strip().lower()
            public_key_source = str(row.get("public_key_source") or key_origin or "unknown").strip()
            private_key_storage = str(row.get("private_key_storage") or "").strip()
            warning = str(row.get("private_key_warning") or "").strip()
            if not private_key_storage:
                if str(row.get("private_key_openssh") or "").strip():
                    private_key_storage = "embedded_keyring"
                elif secret_id:
                    private_key_storage = "client_realm_secret"
                elif key_origin == "generated":
                    private_key_storage = "unknown_generated_location"
                else:
                    private_key_storage = "not_managed"
            if private_key_storage == "embedded_keyring" and not warning:
                warning = "Generated private key is still embedded in keys.json; export/move it or rotate it."
            if private_key_storage == "client_realm_secret" and secret_id and not bool(secret_exists):
                warning = f"Expected client realm secret record is missing: {secret_path}"
            if private_key_storage == "exported_file" and export_path and export_path.exists() and not warning:
                warning = "Generated private key remains in exported-file custody; hand it off to a client realm or purge it."
            if private_key_storage == "terminal_output" and not warning:
                warning = "Generated private key was emitted to terminal output and is not stored by hosting."
            if private_key_storage == "not_managed":
                continue
            rows.append(
                {
                    "key_id": str(key_id),
                    "role": str(row.get("role") or "").strip() or None,
                    "key_origin": key_origin,
                    "public_key_source": public_key_source,
                    "private_key_storage": private_key_storage,
                    "private_key_secret_id": secret_id or None,
                    "private_key_secret_realm": secret_realm if secret_id else None,
                    "private_key_secret_path": str(secret_path) if secret_path else None,
                    "private_key_secret_exists": secret_exists if secret_id else None,
                    "private_key_secret_encryption": secret_encryption if secret_id else None,
                    "private_key_export_path": str(export_path) if export_path else None,
                    "private_key_export_exists": bool(export_path and export_path.exists()) if export_path else None,
                    "private_key_handoff_recorded": bool(row.get("private_key_handoff_recorded")),
                    "private_key_terminal_output": bool(row.get("private_key_terminal_output")),
                    "private_key_protection": secret_protection,
                    "private_key_warning": warning or None,
                }
            )
        return rows

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
        if r == ROLE_DEPENDENCY_APPROVER:
            return {"control"}
        return set()


    @staticmethod
    def _commands_allowed_for_role(role: str) -> set[str]:
        r = str(role or "").strip().lower()
        all_non_bootstrap = {
            "discover-running",
            "spawn",
            *_WORKFLOW_JS_COMMANDS,
            *_WORKFLOW_PYTHON_COMMANDS,
            *_WORKFLOW_ARTIFACT_COMMANDS,
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
            "toolbox-describe",
            "toolbox-describe-refresh",
            "toolbox-gate",
            "toolbox-execute",
            "hosted-operation-status",
            "hosted-operation-resolve-request",
            "hosted-operation-result",
            "hosted-operation-cancel",
            "toolbox-gc",
            *_PACKAGE_UPLOAD_COMMANDS,
            *_PACKAGE_LOCK_COMMANDS,
            *_TOOLBOX_DEFINITION_COMMANDS,
            *_TOOLBOX_TEMPLATE_CONSUMER_COMMANDS,
            *_TOOLBOX_TEMPLATE_ADMIN_COMMANDS,
            "toolbox-references",
            "toolbox-consistency",
            "toolbox-review-snapshot",
            "toolbox-repair",
            "toolbox-reconcile",
            "get-control-config",
            "auth-status",
            "hosting-setup-status",
            "hosting-secure-state-status",
            "model-runtime-status",
            "auth-list-keys",
            "auth-list-sessions",
            "list-live-consumers",
            "auth-list-issued-tokens",
            "auth-audit-list",
            "auth-validate-session",
            "auth-renew-session",
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
            "op-cancel",
            "set-endpoint-mode-override",
            "get-endpoint-mode-effective",
            "get-lifecycle-policy-effective",
            *_HOST_CAPABILITY_SESSION_COMMANDS,
        }
        if r == ROLE_ADMIN:
            return all_non_bootstrap | _TOOLBOX_DEPENDENCY_APPROVAL_COMMANDS | _TOOLBOX_ENVIRONMENT_ADMIN_COMMANDS
        if r == ROLE_DEPENDENCY_APPROVER:
            return {
                *_TOOLBOX_DEPENDENCY_APPROVAL_COMMANDS,
                *_PACKAGE_LOCK_COMMANDS,
                "auth-status",
                "auth-validate-session",
                "auth-renew-session",
            }
        if r == ROLE_CONFIG_EDITOR:
            return {
                "discover-running",
                "spawn",
                *_WORKFLOW_JS_COMMANDS,
                *_WORKFLOW_PYTHON_COMMANDS,
                *_WORKFLOW_ARTIFACT_COMMANDS,
                "get-registration",
                "shutdown",
                "ensure-running",
                "unload-model",
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
                "toolbox-describe-refresh",
                "toolbox-gate",
                "toolbox-execute",
                "hosted-operation-status",
                "hosted-operation-resolve-request",
                "hosted-operation-result",
                "hosted-operation-cancel",
                "toolbox-gc",
                *_PACKAGE_UPLOAD_COMMANDS,
                *_TOOLBOX_DEFINITION_COMMANDS,
                *_TOOLBOX_TEMPLATE_CONSUMER_COMMANDS,
                "toolbox-references",
                "toolbox-consistency",
                "toolbox-review-snapshot",
                "toolbox-repair",
                "toolbox-reconcile",
                "host-metrics",
                "list-configs",
                "create-config",
                "models-from-config",
                "connect-from-config",
                "get-control-config",
                "get-lifecycle-policy-effective",
                *_HOST_CAPABILITY_SESSION_COMMANDS,
                "auth-status",
                "hosting-setup-status",
                "hosting-secure-state-status",
                "model-runtime-status",
            }
        if r == ROLE_WORKER_USER:
            return {
                "discover-running",
                *_WORKFLOW_JS_COMMANDS,
                *_WORKFLOW_PYTHON_COMMANDS,
                *_WORKFLOW_ARTIFACT_COMMANDS,
                "get-registration",
                "shutdown",
                "ensure-running",
                "unload-model",
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
                "toolbox-describe-refresh",
                "toolbox-gate",
                "toolbox-execute",
                "hosted-operation-status",
                "hosted-operation-resolve-request",
                "hosted-operation-result",
                "hosted-operation-cancel",
                "toolbox-gc",
                *_PACKAGE_UPLOAD_COMMANDS,
                *_TOOLBOX_DEFINITION_COMMANDS,
                *_TOOLBOX_TEMPLATE_CONSUMER_COMMANDS,
                "toolbox-references",
                "toolbox-consistency",
                "toolbox-review-snapshot",
                "toolbox-repair",
                "toolbox-reconcile",
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
                *_HOST_CAPABILITY_SESSION_COMMANDS,
                "auth-status",
                "model-runtime-status",
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
                "model-runtime-status",
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
                "model-runtime-status",
            }
        if r == ROLE_DIAGNOSTIC_USER:
            return {
                "discover-running",
                "get-registration",
                "claim-status",
                "resource-claim-status",
                "inspect-capabilities",
                *_WORKFLOW_JS_OBSERVE_COMMANDS,
                *_WORKFLOW_PYTHON_OBSERVE_COMMANDS,
                *_WORKFLOW_ARTIFACT_OBSERVE_COMMANDS,
                "logs-tail",
                "logs-follow",
                "sandbox-fs-list",
                "sandbox-fs-read-text",
                "sandbox-fs-stat",
                "toolbox-describe",
                "toolbox-describe-refresh",
                "toolbox-gate",
                "toolbox-get-definition",
                *_TOOLBOX_TEMPLATE_CONSUMER_COMMANDS,
                "toolbox-references",
                "toolbox-consistency",
                "toolbox-review-snapshot",
                "hosted-operation-status",
                "hosted-operation-resolve-request",
                "hosted-operation-result",
                "host-metrics",
                "get-control-config",
                "get-lifecycle-policy-effective",
                "auth-status",
                "hosting-setup-status",
                "hosting-secure-state-status",
                "model-runtime-status",
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

    def auth_status(self, session_token: Optional[str] = None, presented_ssh_binding: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        self._prune_expired_sessions(auth)
        self._prune_expired_challenges(auth)
        keys = dict(auth.get("keys") or {})
        sessions = dict(auth.get("sessions") or {})
        challenges = dict(auth.get("challenges") or {})
        private_key_custody = self._local_private_key_custody_metadata()
        res = {
            "daemon_version": DAEMON_VERSION,
            "capabilities": self.daemon_capabilities(),
            "require_auth": bool(cfg.get("require_auth", False)),
            "config_store_mode": str(cfg.get("config_store_mode") or "store_only"),
            "keys_count": len(keys),
            "sessions_count": len(sessions),
            "challenges_count": len(challenges),
            "roles": sorted(list({str((v or {}).get("role") or "") for v in keys.values() if isinstance(v, dict)})),
            "local_private_key_custody": private_key_custody,
        }
        if session_token:
            try:
                # We do not strictly enforce scope here, just try to identify the caller.
                session = dict(sessions.get(str(session_token).strip()) or {})
                if session and not bool(session.get("revoked", False)):
                    res["caller_key_id"] = str(session.get("key_id") or "")
                    res["caller_role"] = str(session.get("role") or "")
            except Exception:
                pass
        return res

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
                "public_key": str(m.get("public_key") or ""),
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
                "auth_method": str(m.get("auth_method") or ""),
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

    def auth_validate_session(
        self,
        *,
        token: str,
        scope: str = "control",
        expected_key_id: Optional[str] = None,
        check_ssh_binding: bool = True,
        presented_ssh_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tok = str(token or "").strip()
        scope_norm = str(scope or "control").strip().lower() or "control"
        if scope_norm not in {"control", "config", "traffic"}:
            raise ValueError("scope must be 'control', 'config', or 'traffic'")
        if not tok:
            return {
                "valid": False,
                "reason": "token_required",
                "scope": scope_norm,
                "ssh_bound": False,
            }
        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        auth = dict(cfg.get("auth") or {})
        self._prune_expired_sessions(auth)
        sessions = dict(auth.get("sessions") or {})
        raw_session = dict(sessions.get(tok) or {})
        if not raw_session:
            return {
                "valid": False,
                "reason": "missing_or_invalid_session_token",
                "token_preview": self._token_preview(tok),
                "scope": scope_norm,
                "ssh_bound": False,
            }
        if bool(raw_session.get("revoked", False)):
            return {
                "valid": False,
                "reason": "session_revoked",
                "token_preview": self._token_preview(tok),
                "key_id": str(raw_session.get("key_id") or ""),
                "scope": str(raw_session.get("scope") or ""),
                "ssh_bound": bool(dict(raw_session.get("ssh_binding") or {})),
            }
        expected = str(expected_key_id or "").strip()
        actual_key_id = str(raw_session.get("key_id") or "").strip()
        if expected and actual_key_id != expected:
            return {
                "valid": False,
                "reason": "key_id_mismatch",
                "token_preview": self._token_preview(tok),
                "key_id": actual_key_id,
                "auth_method": str(raw_session.get("auth_method") or ""),
                "role": str(raw_session.get("role") or ""),
                "scope": str(raw_session.get("scope") or ""),
                "expires_at": float(raw_session.get("expires_at") or 0.0),
                "ssh_bound": bool(dict(raw_session.get("ssh_binding") or {})),
            }
        if check_ssh_binding:
            try:
                session = self._validate_session(
                    control,
                    tok,
                    required_scope=scope_norm,
                    presented_ssh_binding=dict(presented_ssh_binding or {}),
                )
            except PermissionError as exc:
                return {
                    "valid": False,
                    "reason": str(exc or "invalid_session"),
                    "token_preview": self._token_preview(tok),
                    "key_id": actual_key_id,
                    "auth_method": str(raw_session.get("auth_method") or ""),
                    "role": str(raw_session.get("role") or ""),
                    "scope": str(raw_session.get("scope") or ""),
                    "expires_at": float(raw_session.get("expires_at") or 0.0),
                    "ssh_bound": bool(dict(raw_session.get("ssh_binding") or {})),
                }
        else:
            session = raw_session
            key_role = str(session.get("role") or "").strip().lower()
            session_scope = str(session.get("scope") or "").strip().lower()
            if key_role not in VALID_AUTH_ROLES:
                return {
                    "valid": False,
                    "reason": "invalid_role",
                    "token_preview": self._token_preview(tok),
                    "key_id": actual_key_id,
                    "auth_method": str(raw_session.get("auth_method") or ""),
                    "role": key_role,
                    "scope": session_scope,
                    "expires_at": float(raw_session.get("expires_at") or 0.0),
                    "ssh_bound": bool(dict(raw_session.get("ssh_binding") or {})),
                }
            if key_role != ROLE_ADMIN and (scope_norm not in self._role_allowed_scopes(key_role) or session_scope != scope_norm):
                return {
                    "valid": False,
                    "reason": "insufficient_scope",
                    "token_preview": self._token_preview(tok),
                    "key_id": actual_key_id,
                    "auth_method": str(raw_session.get("auth_method") or ""),
                    "role": key_role,
                    "scope": session_scope,
                    "expires_at": float(raw_session.get("expires_at") or 0.0),
                    "ssh_bound": bool(dict(raw_session.get("ssh_binding") or {})),
                }
        expires_at = float(session.get("expires_at") or 0.0)
        now = time.time()
        return {
            "valid": True,
            "reason": "ok",
            "token_preview": self._token_preview(tok),
            "key_id": str(session.get("key_id") or ""),
            "actor_key_id": str(session.get("key_id") or ""),
            "auth_method": str(session.get("auth_method") or ""),
            "role": str(session.get("role") or ""),
            "scope": str(session.get("scope") or ""),
            "requested_scope": scope_norm,
            "issued_at": float(session.get("issued_at") or 0.0),
            "expires_at": expires_at,
            "ttl_remaining_seconds": max(0, int(expires_at - now)) if expires_at > 0 else None,
            "allowed_configs": list(session.get("allowed_configs") or []),
            "allowed_engines": list(session.get("allowed_engines") or []),
            "ssh_bound": bool(dict(session.get("ssh_binding") or {})),
            "ssh_binding": dict(session.get("ssh_binding") or {}),
        }

    def auth_renew_session(
        self,
        *,
        token: str,
        scope: str = "control",
        ttl_seconds: int = 900,
        presented_ssh_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tok = str(token or "").strip()
        if not tok:
            raise ValueError("token is required")
        scope_norm = str(scope or "control").strip().lower() or "control"
        if scope_norm not in {"control", "config", "traffic"}:
            raise ValueError("scope must be 'control', 'config', or 'traffic'")

        control = self._read_control()
        cfg = dict(control.get("control_config") or {})
        if not bool(cfg.get("require_auth", False)):
            raise PermissionError("require_auth_disabled_disallows_session_commands")
        auth = dict(cfg.get("auth") or {})
        self._prune_expired_sessions(auth)

        session = self._validate_session(
            control,
            tok,
            required_scope=scope_norm,
            presented_ssh_binding=dict(presented_ssh_binding or {}),
        )

        sessions = dict(auth.get("sessions") or {})
        current = dict(sessions.get(tok) or session or {})
        if not current:
            raise PermissionError("missing_or_invalid_session_token")
        now = time.time()
        ttl = max(60, min(int(ttl_seconds or 900), 24 * 3600))
        expires_at = now + ttl
        current["expires_at"] = expires_at
        current["renewed_at"] = now
        sessions[tok] = current
        auth["sessions"] = sessions
        cfg["auth"] = auth
        control["control_config"] = cfg
        self._write_control(control)
        return {
            "status": "ok",
            "token_preview": self._token_preview(tok),
            "key_id": str(current.get("key_id") or ""),
            "role": str(current.get("role") or ""),
            "auth_method": str(current.get("auth_method") or ""),
            "scope": str(current.get("scope") or ""),
            "expires_at": expires_at,
            "ttl_seconds": ttl,
            "ttl_remaining_seconds": ttl,
            "renewed_at": now,
            "ssh_bound": bool(dict(current.get("ssh_binding") or {})),
            "ssh_binding": dict(current.get("ssh_binding") or {}),
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
            "public_key": pubkey if method == "public_key" else "",
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
        auth_method = str(key_meta.get("auth_method") or "").strip().lower()
        sessions[token] = {
            "key_id": str(key_id or ""),
            "role": role,
            "auth_method": auth_method,
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
            "key_id": str(key_id or ""),
            "scope": scope_norm,
            "role": role,
            "auth_method": auth_method,
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
        
        # If exact token not found, try matching by token_preview (for interactive CLI support)
        if not existed:
            matched_keys = [k for k in sessions.keys() if self._token_preview(k) == tok]
            if len(matched_keys) == 1:
                tok = matched_keys[0]
                existed = True
            elif len(matched_keys) > 1:
                raise ValueError("Ambiguous token preview: multiple sessions match.")
                
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
