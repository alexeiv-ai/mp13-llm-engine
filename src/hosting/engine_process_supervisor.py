"""
Managed local-engine process supervisor.

Phase-2 scaffold:
- Tracks spawned local engine worker metadata in a small JSON state file.
- Provides discovery for currently running managed processes.
- Supports best-effort removal of stale entries.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import subprocess
import time
from threading import Lock
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._process_utils import hidden_subprocess_kwargs, pid_alive, terminate_process_tree

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = Path.home() / ".mp13-llm" / "hosting" / "state"
DEFAULT_STATE_FILE = DEFAULT_STATE_DIR / "managed_engines.json"


class EngineProcessSupervisor:
    """Persisted metadata tracker for managed engine worker processes."""

    def __init__(self, state_file: Optional[Path] = None, control_settings: Optional[Dict[str, Any]] = None):
        self.state_file = (state_file or DEFAULT_STATE_FILE).expanduser().resolve()
        self.control_settings: Dict[str, Any] = dict(control_settings or {})
        self._lock = Lock()
        self._claims_by_engine: Dict[str, Dict[str, Any]] = {}
        self._endpoint_claim: Dict[str, Any] = {"owners": set(), "exclusive_owner": None, "claimed_at": 0.0}
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._tokens_by_engine: Dict[str, set[str]] = {}
        self._resource_claims: Dict[str, Dict[str, Any]] = {}
        self._resource_tokens: Dict[str, Dict[str, Any]] = {}
        self._resource_tokens_by_key: Dict[str, set[str]] = {}

    def _read_state(self) -> List[Dict[str, Any]]:
        if not self.state_file.exists():
            return []
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            items = data.get("engines") if isinstance(data, dict) else []
            return items if isinstance(items, list) else []
        except Exception as exc:
            logger.warning("Failed to read managed engine state file: %s", exc)
            return []

    def _write_state(self, entries: List[Dict[str, Any]]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "engines": entries,
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        return pid_alive(pid)

    def list_registered(self) -> List[Dict[str, Any]]:
        return self._read_state()

    @staticmethod
    def _normalize_backend_id(backend_id: Optional[str]) -> str:
        raw = str(backend_id or "").strip()
        return raw or "backend:unknown"

    def _revoke_tokens_for_engine_nolock(self, engine_id: str) -> int:
        token_ids = set(self._tokens_by_engine.get(str(engine_id), set()))
        revoked = 0
        for token in token_ids:
            if token in self._tokens:
                self._tokens.pop(token, None)
                revoked += 1
        self._tokens_by_engine.pop(str(engine_id), None)
        return revoked

    def _revoke_all_tokens_nolock(self) -> int:
        revoked = len(self._tokens)
        self._tokens.clear()
        self._tokens_by_engine.clear()
        return revoked

    @staticmethod
    def _resource_key(resource_kind: str, resource_id: str) -> str:
        return f"{str(resource_kind or '').strip().lower()}:{str(resource_id or '').strip()}"

    def _revoke_tokens_for_resource_nolock(self, resource_key: str) -> int:
        token_ids = set(self._resource_tokens_by_key.get(str(resource_key), set()))
        revoked = 0
        for token in token_ids:
            if token in self._resource_tokens:
                self._resource_tokens.pop(token, None)
                revoked += 1
        self._resource_tokens_by_key.pop(str(resource_key), None)
        return revoked

    def _revoke_all_resource_tokens_nolock(self) -> int:
        revoked = len(self._resource_tokens)
        self._resource_tokens.clear()
        self._resource_tokens_by_key.clear()
        return revoked

    def _engine_access_denial_reason_nolock(self, engine_id: str, backend_id: str) -> Optional[Dict[str, Any]]:
        endpoint_exclusive = self._endpoint_claim.get("exclusive_owner")
        if endpoint_exclusive and endpoint_exclusive != backend_id:
            return {
                "denied_reason": "endpoint_exclusive_conflict",
                "endpoint_exclusive_owner": endpoint_exclusive,
            }
        claim = self._claims_by_engine.get(str(engine_id))
        if not claim:
            return None
        claim_exclusive = claim.get("exclusive_owner")
        if claim_exclusive:
            if claim_exclusive == backend_id:
                return None
            return {
                "denied_reason": "engine_exclusive_conflict",
                "engine_exclusive_owner": claim_exclusive,
            }
        owners = set(claim.get("owners") or set())
        if not owners or backend_id in owners:
            return None
        return {
            "denied_reason": "engine_shared_claim_not_member",
            "engine_owners": sorted(list(owners)),
        }

    def claim_engine(self, engine_id: str, *, backend_id: Optional[str], exclusive: bool = False) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        bid = self._normalize_backend_id(backend_id)
        if not eid:
            raise ValueError("engine_id is required")
        with self._lock:
            claim = self._claims_by_engine.get(eid) or {"owners": set(), "exclusive_owner": None, "claimed_at": 0.0}
            owners = set(claim.get("owners") or set())
            displaced: List[str] = []
            revoked = 0
            if exclusive:
                displaced = sorted([o for o in owners if o != bid])
                claim["owners"] = {bid}
                claim["exclusive_owner"] = bid
                claim["claimed_at"] = time.time()
                revoked = self._revoke_tokens_for_engine_nolock(eid)
            else:
                previous_exclusive = str(claim.get("exclusive_owner") or "")
                if previous_exclusive and previous_exclusive != bid:
                    displaced = [previous_exclusive]
                    revoked = self._revoke_tokens_for_engine_nolock(eid)
                claim["exclusive_owner"] = None
                owners.add(bid)
                claim["owners"] = owners
                claim["claimed_at"] = time.time()
            self._claims_by_engine[eid] = claim
            return {
                "scope": "engine",
                "engine_id": eid,
                "backend_id": bid,
                "mode": "exclusive" if exclusive else "shared",
                "owners": sorted(list(claim.get("owners") or set())),
                "exclusive_owner": claim.get("exclusive_owner"),
                "displaced_backends": displaced,
                "revoked_tokens": revoked,
            }

    def claim_endpoint(self, *, backend_id: Optional[str], exclusive: bool = False) -> Dict[str, Any]:
        bid = self._normalize_backend_id(backend_id)
        with self._lock:
            displaced: List[str] = []
            revoked = 0
            endpoint_owners = set(self._endpoint_claim.get("owners") or set())
            if exclusive:
                displaced = sorted([o for o in endpoint_owners if o != bid])
                self._endpoint_claim = {"owners": {bid}, "exclusive_owner": bid, "claimed_at": time.time()}
                self._claims_by_engine.clear()
                revoked = self._revoke_all_tokens_nolock() + self._revoke_all_resource_tokens_nolock()
            else:
                previous_exclusive = str(self._endpoint_claim.get("exclusive_owner") or "")
                if previous_exclusive and previous_exclusive != bid:
                    displaced = [previous_exclusive]
                    revoked = self._revoke_all_tokens_nolock() + self._revoke_all_resource_tokens_nolock()
                endpoint_owners.add(bid)
                self._endpoint_claim = {
                    "owners": endpoint_owners,
                    "exclusive_owner": None,
                    "claimed_at": time.time(),
                }
            return {
                "scope": "endpoint",
                "backend_id": bid,
                "mode": "exclusive" if exclusive else "shared",
                "owners": sorted(list(self._endpoint_claim.get("owners") or set())),
                "exclusive_owner": self._endpoint_claim.get("exclusive_owner"),
                "displaced_backends": displaced,
                "revoked_tokens": revoked,
            }

    def get_claim_status(self, engine_id: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        with self._lock:
            engine_claim = self._claims_by_engine.get(eid) or {"owners": set(), "exclusive_owner": None, "claimed_at": 0.0}
            endpoint_claim = self._endpoint_claim or {"owners": set(), "exclusive_owner": None, "claimed_at": 0.0}
            return {
                "engine_id": eid,
                "engine_claim": {
                    "owners": sorted(list(engine_claim.get("owners") or set())),
                    "exclusive_owner": engine_claim.get("exclusive_owner"),
                    "claimed_at": float(engine_claim.get("claimed_at") or 0.0),
                },
                "endpoint_claim": {
                    "owners": sorted(list(endpoint_claim.get("owners") or set())),
                    "exclusive_owner": endpoint_claim.get("exclusive_owner"),
                    "claimed_at": float(endpoint_claim.get("claimed_at") or 0.0),
                },
                "issued_tokens": len(self._tokens_by_engine.get(eid, set())),
            }

    def issue_token(self, engine_id: str, *, backend_id: Optional[str]) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        bid = self._normalize_backend_id(backend_id)
        if not eid:
            raise ValueError("engine_id is required")
        with self._lock:
            denied = self._engine_access_denial_reason_nolock(eid, bid)
            if denied is not None:
                return {
                    "status": "denied",
                    "engine_id": eid,
                    "backend_id": bid,
                    "token": None,
                    **denied,
                }
            token = secrets.token_urlsafe(24)
            self._tokens[token] = {
                "engine_id": eid,
                "backend_id": bid,
                "issued_at": time.time(),
            }
            self._tokens_by_engine.setdefault(eid, set()).add(token)
            return {
                "status": "ok",
                "engine_id": eid,
                "backend_id": bid,
                "token": token,
                "issued_at": self._tokens[token]["issued_at"],
            }

    def claim_resource(
        self,
        resource_kind: str,
        resource_id: str,
        *,
        backend_id: Optional[str],
        exclusive: bool = False,
    ) -> Dict[str, Any]:
        rkind = str(resource_kind or "").strip().lower()
        rid = str(resource_id or "").strip()
        if not rkind:
            raise ValueError("resource_kind is required")
        if not rid:
            raise ValueError("resource_id is required")
        if rkind == "engine":
            return self.claim_engine(rid, backend_id=backend_id, exclusive=exclusive)

        bid = self._normalize_backend_id(backend_id)
        rkey = self._resource_key(rkind, rid)
        with self._lock:
            claim = self._resource_claims.get(rkey) or {"owners": set(), "exclusive_owner": None, "claimed_at": 0.0}
            owners = set(claim.get("owners") or set())
            displaced: List[str] = []
            revoked = 0
            if exclusive:
                displaced = sorted([o for o in owners if o != bid])
                claim["owners"] = {bid}
                claim["exclusive_owner"] = bid
                claim["claimed_at"] = time.time()
                revoked = self._revoke_tokens_for_resource_nolock(rkey)
            else:
                previous_exclusive = str(claim.get("exclusive_owner") or "")
                if previous_exclusive and previous_exclusive != bid:
                    displaced = [previous_exclusive]
                    revoked = self._revoke_tokens_for_resource_nolock(rkey)
                claim["exclusive_owner"] = None
                owners.add(bid)
                claim["owners"] = owners
                claim["claimed_at"] = time.time()
            self._resource_claims[rkey] = claim
            return {
                "scope": "resource",
                "resource_kind": rkind,
                "resource_id": rid,
                "backend_id": bid,
                "mode": "exclusive" if exclusive else "shared",
                "owners": sorted(list(claim.get("owners") or set())),
                "exclusive_owner": claim.get("exclusive_owner"),
                "displaced_backends": displaced,
                "revoked_tokens": revoked,
            }

    def get_resource_claim_status(self, resource_kind: str, resource_id: str) -> Dict[str, Any]:
        rkind = str(resource_kind or "").strip().lower()
        rid = str(resource_id or "").strip()
        if not rkind:
            raise ValueError("resource_kind is required")
        if not rid:
            raise ValueError("resource_id is required")
        if rkind == "engine":
            return self.get_claim_status(rid)
        rkey = self._resource_key(rkind, rid)
        with self._lock:
            claim = self._resource_claims.get(rkey) or {"owners": set(), "exclusive_owner": None, "claimed_at": 0.0}
            endpoint_claim = self._endpoint_claim or {"owners": set(), "exclusive_owner": None, "claimed_at": 0.0}
            return {
                "resource_kind": rkind,
                "resource_id": rid,
                "resource_claim": {
                    "owners": sorted(list(claim.get("owners") or set())),
                    "exclusive_owner": claim.get("exclusive_owner"),
                    "claimed_at": float(claim.get("claimed_at") or 0.0),
                },
                "endpoint_claim": {
                    "owners": sorted(list(endpoint_claim.get("owners") or set())),
                    "exclusive_owner": endpoint_claim.get("exclusive_owner"),
                    "claimed_at": float(endpoint_claim.get("claimed_at") or 0.0),
                },
                "issued_tokens": len(self._resource_tokens_by_key.get(rkey, set())),
            }

    def issue_resource_token(self, resource_kind: str, resource_id: str, *, backend_id: Optional[str]) -> Dict[str, Any]:
        rkind = str(resource_kind or "").strip().lower()
        rid = str(resource_id or "").strip()
        if not rkind:
            raise ValueError("resource_kind is required")
        if not rid:
            raise ValueError("resource_id is required")
        if rkind == "engine":
            issued = self.issue_token(rid, backend_id=backend_id)
            return {
                **issued,
                "resource_kind": "engine",
                "resource_id": rid,
            }

        bid = self._normalize_backend_id(backend_id)
        rkey = self._resource_key(rkind, rid)
        with self._lock:
            endpoint_exclusive = self._endpoint_claim.get("exclusive_owner")
            if endpoint_exclusive and endpoint_exclusive != bid:
                return {
                    "status": "denied",
                    "resource_kind": rkind,
                    "resource_id": rid,
                    "backend_id": bid,
                    "token": None,
                    "denied_reason": "endpoint_exclusive_conflict",
                    "endpoint_exclusive_owner": endpoint_exclusive,
                }
            claim = self._resource_claims.get(rkey)
            if claim:
                exclusive_owner = claim.get("exclusive_owner")
                if exclusive_owner and exclusive_owner != bid:
                    return {
                        "status": "denied",
                        "resource_kind": rkind,
                        "resource_id": rid,
                        "backend_id": bid,
                        "token": None,
                        "denied_reason": "resource_exclusive_conflict",
                        "resource_exclusive_owner": exclusive_owner,
                    }
                owners = set(claim.get("owners") or set())
                if owners and bid not in owners:
                    return {
                        "status": "denied",
                        "resource_kind": rkind,
                        "resource_id": rid,
                        "backend_id": bid,
                        "token": None,
                        "denied_reason": "resource_shared_claim_not_member",
                        "resource_owners": sorted(list(owners)),
                    }
            token = secrets.token_urlsafe(24)
            self._resource_tokens[token] = {
                "resource_kind": rkind,
                "resource_id": rid,
                "resource_key": rkey,
                "backend_id": bid,
                "issued_at": time.time(),
            }
            self._resource_tokens_by_key.setdefault(rkey, set()).add(token)
            return {
                "status": "ok",
                "resource_kind": rkind,
                "resource_id": rid,
                "backend_id": bid,
                "token": token,
                "issued_at": self._resource_tokens[token]["issued_at"],
            }

    def validate_resource_token(self, resource_kind: str, resource_id: str, token: str) -> bool:
        rkind = str(resource_kind or "").strip().lower()
        rid = str(resource_id or "").strip()
        t = str(token or "").strip()
        if not rkind or not rid or not t:
            return False
        if rkind == "engine":
            return self.validate_token(rid, t)
        with self._lock:
            meta = self._resource_tokens.get(t)
            if not meta:
                return False
            return (
                str(meta.get("resource_kind") or "") == rkind
                and str(meta.get("resource_id") or "") == rid
            )

    def validate_token(self, engine_id: str, token: str) -> bool:
        eid = str(engine_id or "").strip()
        t = str(token or "").strip()
        if not eid or not t:
            return False
        with self._lock:
            meta = self._tokens.get(t)
            return bool(meta and str(meta.get("engine_id") or "") == eid)

    def register_spawned(
        self,
        *,
        engine_id: str,
        pid: int,
        command: List[str],
        cwd: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        entries = self._read_state()
        now = time.time()
        record = {
            "engine_id": str(engine_id),
            "pid": int(pid),
            "command": [str(x) for x in (command or [])],
            "cwd": str(cwd) if cwd else None,
            "spawned_at": now,
            "owner_backend_pid": os.getpid(),
            "source": "backend_spawned",
            "endpoint": str(endpoint).strip() if endpoint else None,
        }
        entries = [e for e in entries if str(e.get("engine_id")) != record["engine_id"]]
        entries.append(record)
        self._write_state(entries)
        return record

    def spawn_process(
        self,
        *,
        engine_id: str,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Spawn a detached worker process and register it."""
        if not command:
            raise ValueError("command is required")
        proc = subprocess.Popen(  # noqa: S603,S607
            [str(x) for x in command],
            cwd=str(cwd) if cwd else None,
            env=(dict(os.environ) | {str(k): str(v) for k, v in (env or {}).items()}),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **hidden_subprocess_kwargs(),
        )
        return self.register_spawned(
            engine_id=engine_id,
            pid=int(proc.pid),
            command=[str(x) for x in command],
            cwd=cwd,
            endpoint=endpoint,
        )

    def remove_registration(self, engine_id: str) -> None:
        entries = self._read_state()
        out = [e for e in entries if str(e.get("engine_id")) != str(engine_id)]
        if len(out) != len(entries):
            self._write_state(out)
        with self._lock:
            self._claims_by_engine.pop(str(engine_id), None)
            self._revoke_tokens_for_engine_nolock(str(engine_id))

    def get_registration(self, engine_id: str) -> Optional[Dict[str, Any]]:
        for item in self._read_state():
            if str(item.get("engine_id")) == str(engine_id):
                return item
        return None

    def shutdown_managed(self, engine_id: str, *, timeout_seconds: float = 8.0) -> Dict[str, Any]:
        """Best-effort shutdown by PID for a registered managed engine."""
        entry = self.get_registration(engine_id)
        if not entry:
            return {"status": "not_found", "engine_id": engine_id, "alive": False}

        pid = int(entry.get("pid") or 0)
        if pid <= 0:
            self.remove_registration(engine_id)
            return {"status": "invalid_pid", "engine_id": engine_id, "alive": False}

        if not self._pid_alive(pid):
            self.remove_registration(engine_id)
            return {"status": "already_stopped", "engine_id": engine_id, "pid": pid, "alive": False}

        termination = terminate_process_tree(pid, timeout_seconds=timeout_seconds)
        alive = self._pid_alive(pid)
        if not alive:
            self.remove_registration(engine_id)
        return {
            "status": "stopped" if not alive else "stop_failed",
            "engine_id": engine_id,
            "pid": pid,
            "alive": alive,
            "termination": termination,
        }

    def ensure_running(self, engine_id: str) -> Dict[str, Any]:
        """
        Ensure a managed worker registration has a live process.

        If the registered pid is dead and command metadata exists, respawn using
        the stored command/cwd and update registration.
        """
        entry = self.get_registration(engine_id)
        if not entry:
            return {"status": "not_found", "engine_id": str(engine_id), "alive": False}

        eid = str(entry.get("engine_id") or engine_id)
        pid = int(entry.get("pid") or 0)
        command = [str(x) for x in list(entry.get("command") or []) if str(x).strip()]
        cwd = entry.get("cwd")
        endpoint = entry.get("endpoint")

        if pid > 0 and self._pid_alive(pid):
            return {
                "status": "running",
                "engine_id": eid,
                "pid": pid,
                "alive": True,
                "endpoint": endpoint,
            }

        if not command:
            return {
                "status": "cannot_respawn",
                "engine_id": eid,
                "pid": pid,
                "alive": False,
                "reason": "missing_command_metadata",
                "endpoint": endpoint,
            }

        proc = subprocess.Popen(  # noqa: S603,S607
            command,
            cwd=str(cwd) if cwd else None,
            env=dict(os.environ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **hidden_subprocess_kwargs(),
        )
        record = self.register_spawned(
            engine_id=eid,
            pid=int(proc.pid),
            command=command,
            cwd=str(cwd) if cwd else None,
            endpoint=str(endpoint) if endpoint else None,
        )
        return {
            "status": "respawned",
            "engine_id": eid,
            "previous_pid": pid,
            "pid": int(record.get("pid") or 0),
            "alive": bool(record.get("pid")),
            "endpoint": record.get("endpoint"),
        }

    def discover_running(self, *, prune_stale: bool = True) -> List[Dict[str, Any]]:
        entries = self._read_state()
        running: List[Dict[str, Any]] = []
        stale_seen = False
        for item in entries:
            try:
                pid = int(item.get("pid") or 0)
            except Exception:
                pid = 0
            alive = self._pid_alive(pid)
            enriched = dict(item)
            enriched["alive"] = bool(alive)
            running.append(enriched)
            if not alive:
                stale_seen = True

        if prune_stale and stale_seen:
            self._write_state([x for x in running if bool(x.get("alive"))])
            running = [x for x in running if bool(x.get("alive"))]

        running.sort(key=lambda x: str(x.get("engine_id") or ""))
        return running
