from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional

from .sandbox.policy import WorkerSandboxPolicy


def _preview(request: Dict[str, Any]) -> Dict[str, Any]:
    return dict(dict(request or {}).get("argument_preview") or {})


def _policy(policy: Any) -> WorkerSandboxPolicy:
    if isinstance(policy, WorkerSandboxPolicy):
        return policy
    return WorkerSandboxPolicy.from_mapping(dict(policy or {}) if isinstance(policy, dict) else {})


def host_capability_approval_check_fs_path(
    request: Dict[str, Any],
    sandbox_policy: Any,
    *,
    access: str = "read",
    scoped_root: Optional[str] = None,
    allow_empty_relative_path: bool = False,
) -> Dict[str, Any]:
    """Validate an approval request preview for a brokered filesystem method."""
    preview = _preview(request)
    root_id = str(preview.get("root_id") or "").strip()
    relative_path = str(preview.get("relative_path") or "").strip().replace("\\", "/")
    if not root_id:
        return {"allowed": False, "reason": "root_id_required", "root_id": root_id, "relative_path": relative_path}
    if not relative_path and not allow_empty_relative_path:
        return {"allowed": False, "reason": "relative_path_required", "root_id": root_id, "relative_path": relative_path}
    if relative_path.startswith("/") or ":" in relative_path:
        return {"allowed": False, "reason": "absolute_path_denied", "root_id": root_id, "relative_path": relative_path}
    policy = _policy(sandbox_policy)
    if not policy.enabled:
        return {"allowed": False, "reason": "sandbox_disabled", "root_id": root_id, "relative_path": relative_path}
    if not policy.brokered_io.filesystem:
        return {"allowed": False, "reason": "brokered_filesystem_disabled", "root_id": root_id, "relative_path": relative_path}
    rule = None
    for item in list(policy.filesystem_rules or []):
        if str(item.root_id or "").strip() == root_id:
            rule = item
            break
    if rule is None:
        return {"allowed": False, "reason": "unknown_root_id", "root_id": root_id, "relative_path": relative_path}
    if str(access or "").strip().lower() not in {str(item or "").strip().lower() for item in list(rule.access or [])}:
        return {"allowed": False, "reason": "fs_access_denied", "root_id": root_id, "relative_path": relative_path}
    root = Path(str(rule.path or "")).expanduser().resolve()
    target = (root / relative_path).resolve() if relative_path else root
    try:
        resolved_relative = target.relative_to(root).as_posix()
    except Exception:
        return {
            "allowed": False,
            "reason": "path_traversal_denied",
            "root_id": root_id,
            "relative_path": relative_path,
            "root_path": str(root),
            "resolved_path": str(target),
        }
    if scoped_root:
        scoped = (root / str(scoped_root or "").strip().replace("\\", "/")).resolve()
        try:
            target.relative_to(scoped)
        except Exception:
            return {
                "allowed": False,
                "reason": "outside_approved_scope",
                "root_id": root_id,
                "relative_path": resolved_relative,
                "root_path": str(root),
                "resolved_path": str(target),
                "scoped_root": str(scoped),
            }
    return {
        "allowed": True,
        "reason": "",
        "root_id": root_id,
        "relative_path": resolved_relative,
        "root_path": str(root),
        "resolved_path": str(target),
    }


def host_capability_approval_check_http_fetch(request: Dict[str, Any], sandbox_policy: Any) -> Dict[str, Any]:
    """Validate an approval request preview for brokered `http.fetch`."""
    preview = _preview(request)
    url = str(preview.get("url") or "").strip()
    method = str(preview.get("method") or "GET").strip().upper() or "GET"
    policy = _policy(sandbox_policy)
    if not policy.enabled:
        return {"allowed": False, "reason": "sandbox_disabled", "url": url, "method": method}
    if not policy.brokered_io.http:
        return {"allowed": False, "reason": "brokered_http_disabled", "url": url, "method": method}
    if str(policy.network.mode or "").strip().lower() != "brokered_only":
        return {"allowed": False, "reason": "brokered_http_requires_network_mode_brokered_only", "url": url, "method": method}
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        return {"allowed": False, "reason": "brokered_http_scheme_not_allowed", "url": url, "method": method}
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return {"allowed": False, "reason": "brokered_http_host_required", "url": url, "method": method}
    allowed_hosts = {str(item or "").strip().lower() for item in list(policy.network.allow_hosts or []) if str(item or "").strip()}
    if allowed_hosts and host not in allowed_hosts:
        return {"allowed": False, "reason": f"brokered_http_host_not_allowed:{host}", "url": url, "method": method, "host": host}
    prefixes = [str(item or "").strip() for item in list(policy.network.allow_url_prefixes or []) if str(item or "").strip()]
    if prefixes and not any(url.startswith(prefix) for prefix in prefixes):
        return {"allowed": False, "reason": "brokered_http_url_not_allowed", "url": url, "method": method, "host": host}
    return {"allowed": True, "reason": "", "url": urllib.parse.urlunsplit(parsed), "method": method, "host": host}


def service_broker_method_policy_hint(method: str) -> Dict[str, Any]:
    """Return registry-owned approval-policy hints for a service-broker method."""
    from .sandbox.service_broker_registry import service_broker_method_policy_hint as _policy_hint

    return _policy_hint(str(method or ""))


def host_capability_approval_check_service_broker_request(
    request: Dict[str, Any],
    sandbox_policy: Any,
    *,
    scoped_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a service-broker approval request using registry policy hints."""
    method = str(dict(request or {}).get("method") or "").strip()
    hint = service_broker_method_policy_hint(method)
    kind = str(hint.get("kind") or "").strip().lower()
    if kind == "filesystem":
        return host_capability_approval_check_fs_path(
            request,
            sandbox_policy,
            access=str(hint.get("access") or "read"),
            scoped_root=scoped_root,
            allow_empty_relative_path=bool(hint.get("allow_empty_relative_path", False)),
        )
    if kind == "http":
        return host_capability_approval_check_http_fetch(request, sandbox_policy)
    return {"allowed": False, "reason": f"unsupported_service_broker_method:{method}", "method": method}
