"""Runtime metrics helpers for the engine host service."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


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

    @classmethod
    def _process_resource_snapshot(cls, pid: int) -> Dict[str, Any]:
        target = int(pid or 0)
        base: Dict[str, Any] = {
            "pid": target,
            "cpu_percent": None,
            "memory_mb": None,
            "memory_kind": None,
            "gpu_vram_mb": None,
            "gpu_allocated_mb": None,
            "gpu_devices": [],
            "gpu_vram_source": None,
            "process_resource_source": None,
        }
        if target <= 0:
            return base
        if sys.platform.startswith("win"):
            base.update(cls._process_resource_snapshot_windows(target))
        elif sys.platform.startswith("linux"):
            base.update(cls._process_resource_snapshot_linux(target))
        else:
            # macOS/BSD do not expose portable stdlib process CPU/RSS APIs.
            # Keep these fields explicit N/A instead of launching ps/top.
            base["process_resource_source"] = "not_available_stdlib"
        return base

    @classmethod
    def _process_cpu_cache(cls) -> Dict[int, Dict[str, float]]:
        cache = getattr(cls, "_process_resource_cpu_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(cls, "_process_resource_cpu_cache", cache)
        return cache

    @classmethod
    def _cpu_percent_from_sample(cls, pid: int, proc_seconds: float, wall_seconds_basis: float) -> float:
        now = time.time()
        cache = cls._process_cpu_cache()
        previous = dict(cache.get(pid) or {})
        cache[pid] = {"time": now, "proc_seconds": proc_seconds, "basis": wall_seconds_basis}
        prev_time = float(previous.get("time") or 0.0)
        prev_proc = float(previous.get("proc_seconds") or 0.0)
        prev_basis = float(previous.get("basis") or wall_seconds_basis or 1.0)
        elapsed = now - prev_time
        if prev_time <= 0.0 or elapsed <= 0.0:
            return 0.0
        basis = max(1.0, wall_seconds_basis or prev_basis or 1.0)
        return round(max(0.0, ((proc_seconds - prev_proc) / elapsed) * 100.0 / basis), 1)

    @classmethod
    def _process_resource_snapshot_windows(cls, pid: int) -> Dict[str, Any]:
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            PROCESS_VM_READ = 0x0010

            class FILETIME(ctypes.Structure):
                _fields_ = [
                    ("dwLowDateTime", wintypes.DWORD),
                    ("dwHighDateTime", wintypes.DWORD),
                ]

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            kernel32.GetProcessTimes.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
                ctypes.POINTER(FILETIME),
            ]
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, int(pid))
            if not handle:
                return {"process_resource_source": "win32_unavailable"}
            try:
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                memory_mb = None
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    memory_mb = round(float(counters.WorkingSetSize) / (1024.0 * 1024.0), 1)

                creation = FILETIME()
                exit_time = FILETIME()
                kernel = FILETIME()
                user = FILETIME()
                cpu = None
                if kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel), ctypes.byref(user)):
                    kernel_ticks = (int(kernel.dwHighDateTime) << 32) + int(kernel.dwLowDateTime)
                    user_ticks = (int(user.dwHighDateTime) << 32) + int(user.dwLowDateTime)
                    proc_seconds = float(kernel_ticks + user_ticks) / 10_000_000.0
                    cpu = cls._cpu_percent_from_sample(int(pid), proc_seconds, float(os.cpu_count() or 1))
                return {
                    "cpu_percent": cpu,
                    "memory_mb": memory_mb,
                    "memory_kind": "working_set",
                    "process_resource_source": "win32",
                }
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return {"process_resource_source": "win32_error"}

    @classmethod
    def _process_resource_snapshot_linux(cls, pid: int) -> Dict[str, Any]:
        try:
            stat_text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
            stat_end = stat_text.rfind(")")
            fields = stat_text[stat_end + 2 :].split()
            utime = int(fields[11])
            stime = int(fields[12])
            sysconf = getattr(os, "sysconf")
            ticks = sysconf("SC_CLK_TCK")
            proc_seconds = float(utime + stime) / float(ticks or 100)
            cpu = cls._cpu_percent_from_sample(int(pid), proc_seconds, float(os.cpu_count() or 1))
            status_text = (Path("/proc") / str(pid) / "status").read_text(encoding="utf-8")
            memory_mb = None
            for line in status_text.splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        memory_mb = round(float(parts[1]) / 1024.0, 1)
                    break
            return {
                "cpu_percent": cpu,
                "memory_mb": memory_mb,
                "memory_kind": "rss",
                "process_resource_source": "procfs",
            }
        except Exception:
            return {"process_resource_source": "procfs_unavailable"}

    def _registered_worker_resource_rows(self) -> List[Dict[str, Any]]:
        try:
            discovered = self.discover_running(  # type: ignore[attr-defined]
                prune_stale=False,
                include_progress=False,
                include_reachability=True,
                reachability_timeout_seconds=0.75,
            )
            rows = [dict(row or {}) for row in list(discovered or []) if isinstance(row, dict)]
        except Exception:
            try:
                rows = [dict(row or {}) for row in list(self._read_engines() or []) if isinstance(row, dict)]  # type: ignore[attr-defined]
            except Exception:
                return []
        out: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for row in rows:
            pid = int(row.get("pid") or 0)
            if pid <= 0 or pid in seen:
                continue
            seen.add(pid)
            snap = self._process_resource_snapshot(pid)
            worker_resources = dict(row.get("process_resources") or {})
            for key in ("gpu_vram_mb", "gpu_allocated_mb", "gpu_devices", "gpu_vram_source", "gpu_vram_pending"):
                if worker_resources.get(key) not in (None, "", []):
                    snap[key] = worker_resources.get(key)
            snap["engine_id"] = str(row.get("engine_id") or "")
            snap["kind"] = self._describe_registration_kind(row) if hasattr(self, "_describe_registration_kind") else None
            out.append(snap)
        return out

    @staticmethod
    def _resource_summary_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        known_cpu = [float(row.get("cpu_percent") or 0.0) for row in rows if row.get("cpu_percent") is not None]
        known_memory = [float(row.get("memory_mb") or 0.0) for row in rows if row.get("memory_mb") is not None]
        known_vram = [float(row.get("gpu_vram_mb") or 0.0) for row in rows if row.get("gpu_vram_mb") is not None]
        known_allocated = [
            float(row.get("gpu_allocated_mb") or 0.0)
            for row in rows
            if row.get("gpu_allocated_mb") is not None
        ]
        pending_vram = any(bool(row.get("gpu_vram_pending")) for row in rows)
        return {
            "workers_count": len(rows),
            "worker_cpu_percent": round(sum(known_cpu), 1) if known_cpu else None,
            "worker_memory_mb": round(sum(known_memory), 1) if known_memory else None,
            "worker_gpu_vram_mb": round(sum(known_vram), 1) if known_vram else None,
            "worker_gpu_allocated_mb": round(sum(known_allocated), 1) if known_allocated else None,
            "worker_gpu_vram_pending": pending_vram and not known_vram,
        }

    def get_host_metrics(self, session_token: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_metrics_initialized()
        with self._metrics_lock:
            assert isinstance(self._runtime_metrics, dict)
            snapshot = json.loads(json.dumps(self._runtime_metrics))
        snapshot["pid"] = os.getpid()
        snapshot["runtime_scope"] = "process"
        snapshot["recommended_mode"] = "daemon"
        snapshot["timestamp"] = time.time()
        worker_rows = self._registered_worker_resource_rows()
        snapshot["worker_processes"] = worker_rows
        snapshot["resource_summary"] = self._resource_summary_from_rows(worker_rows)
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
