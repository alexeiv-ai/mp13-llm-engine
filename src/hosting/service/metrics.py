"""Runtime metrics helpers for the engine host service."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


class MetricsMixin:
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

    def get_host_metrics(self, session_token: Optional[str] = None) -> Dict[str, Any]:
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
        try:
            auth_status = dict(self.auth_status(session_token=session_token) or {})  # type: ignore[attr-defined]
        except Exception as exc:
            snapshot["auth_status_error"] = str(exc)
        else:
            snapshot["auth_status"] = auth_status
            snapshot["auth_status_error"] = None
            snapshot["require_auth"] = bool(auth_status.get("require_auth", False))
            snapshot["keys_count"] = int(auth_status.get("keys_count") or 0)
            snapshot["sessions_count"] = int(auth_status.get("sessions_count") or 0)
        return snapshot
