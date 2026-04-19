"""Claim ownership and audit helpers for the engine host service."""
from __future__ import annotations

import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

from .constants import EMERGENCY_FORCE_OVERRIDE_REASONS, VALID_FORCE_OVERRIDE_REASONS


class ClaimsMixin:
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
