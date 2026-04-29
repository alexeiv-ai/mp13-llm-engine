"""Engine lifecycle and registration helpers for the engine host service."""
from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..sandbox import (
    BrokeredFilesystem,
    HostBrokeredHttpClient,
    WorkerLaunchRequest,
    WorkerSandboxPolicy,
    launch_worker_process,
)


class EnginesMixin:
    @staticmethod
    def _launch_worker_process(request: WorkerLaunchRequest):
        return launch_worker_process(request)

    def _next_engine_id(self, base_name: str) -> str:
        existing = {str(x.get("engine_id") or "") for x in self._read_engines()}
        if base_name not in existing:
            return base_name
        idx = 2
        while f"{base_name}_{idx}" in existing:
            idx += 1
        return f"{base_name}_{idx}"

    def _check_module_available(self, python: str, module_name: str) -> Tuple[bool, str]:
        """
        Check whether engine runtime symbols are importable by *python*.
        """
        from ..engine_discovery import is_engine_available
        # We assume module_name is "mp13_engine" which is the only one used here
        return is_engine_available(python)

    def _check_module_discoverable(self, python: str, module_name: str) -> Tuple[bool, str]:
        """
        Lightweight module check for UX surfaces (e.g., list-configs).
        """
        from ..engine_discovery import is_engine_discoverable
        # We assume module_name is "mp13_engine" which is the only one used here
        return is_engine_discoverable(python)

    def _engine_python_executable(self) -> str:
        python = os.environ.get("MP13_ENGINE_PYTHON", "").strip()
        return python or sys.executable

    def _build_engine_spawn_spec(self, *, engine_id: str, config_path: str, model_path: str) -> Dict[str, Any]:
        python = self._engine_python_executable()
        ok, err_detail = self._check_module_available(python, "mp13_engine")
        if not ok:
            return {
                "error": (
                    f"mp13_engine is not available in Python '{python}': {err_detail}. "
                    "Set MP13_ENGINE_PYTHON to a Python that has mp13_engine installed."
                ),
                "error_kind": "engine_not_available",
            }
        transport = "ipc"
        worker_auth_token = secrets.token_urlsafe(24)
        worker_auth_header = "X-MP13-Host-Token"
        ipc_family, ipc_address = self._allocate_ipc_address(engine_id)
        import socket
        endpoint = f"ipc://{socket.gethostname()}"
        command = [
            python,
            "-m",
            "hosting.engine_worker_ipc",
            "--ipc-family",
            str(ipc_family),
            "--ipc-address",
            str(ipc_address),
        ]
        return {
            "command": command,
            "cwd": None,
            "endpoint": endpoint,
            "worker_auth_token": worker_auth_token,
            "worker_auth_header": worker_auth_header,
            "worker_transport": transport,
            "worker_ipc_family": ipc_family,
            "worker_ipc_address": ipc_address,
            "env": {
                "MP13_ENGINE_CONFIG_PATH": str(config_path),
                "MP13_ENGINE_ID": str(engine_id),
                "MP13_MODEL_PATH": str(model_path),
                "MP13_ENGINE_HOST_TOKEN": worker_auth_token,
                "MP13_ENGINE_HOST_TOKEN_HEADER": worker_auth_header,
                "MP13_ENGINE_TRANSPORT": transport,
            },
        }

    def _build_generic_spawn_spec(
        self,
        *,
        engine_id: str,
        config_path: str,
        config_payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        cfg = dict(config_payload or {})
        spawn_cfg = dict(cfg.get("spawn") or {}) if isinstance(cfg.get("spawn"), dict) else {}
        cmd_raw = cfg.get("worker_command")
        if not (isinstance(cmd_raw, list) and cmd_raw):
            cmd_raw = spawn_cfg.get("command")
        if not (isinstance(cmd_raw, list) and cmd_raw):
            return {
                "error": "generic worker config is missing worker_command/spawn.command",
                "error_kind": "generic_worker_command_missing",
            }
        command = [str(x) for x in list(cmd_raw) if str(x).strip()]
        if not command:
            return {
                "error": "generic worker command resolved to empty",
                "error_kind": "generic_worker_command_missing",
            }
        selected = Path(str(config_path or "")).expanduser().resolve()
        config_dir = selected.parent
        cwd_raw = cfg.get("worker_cwd") or spawn_cfg.get("cwd")
        cwd = None
        if str(cwd_raw or "").strip():
            try:
                cwd = str(self._resolve_path_token(str(cwd_raw), config_dir=config_dir))
            except Exception:
                cwd = str(cwd_raw)
        env: Dict[str, Any] = {}
        worker_env = cfg.get("worker_env")
        spawn_env = spawn_cfg.get("env")
        if isinstance(worker_env, dict):
            env.update({str(k): str(v) for k, v in worker_env.items()})
        if isinstance(spawn_env, dict):
            env.update({str(k): str(v) for k, v in spawn_env.items()})
        env.setdefault("MP13_ENGINE_CONFIG_PATH", str(config_path))
        env.setdefault("MP13_ENGINE_ID", str(engine_id))
        return {
            "command": command,
            "cwd": cwd,
            "env": env,
            "worker_transport": str(cfg.get("worker_transport") or "").strip() or None,
            "worker_ipc_family": str(cfg.get("worker_ipc_family") or "").strip() or None,
            "worker_ipc_address": str(cfg.get("worker_ipc_address") or "").strip() or None,
            "worker_auth_token": str(cfg.get("worker_auth_token") or "").strip() or None,
            "worker_auth_header": str(cfg.get("worker_auth_header") or "").strip() or None,
        }

    def _wait_for_worker_rpc_ready(
        self,
        reg: Dict[str, Any],
        *,
        timeout_seconds: float = 600.0,
        poll_interval_seconds: float = 0.5,
    ) -> Dict[str, Any]:
        deadline = time.time() + max(1.0, float(timeout_seconds or 600.0))
        interval = max(0.1, min(float(poll_interval_seconds or 0.5), 5.0))
        last_error = ""
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            pid = int(reg.get("pid") or 0)
            if pid > 0 and not self._pid_alive(pid):
                raise RuntimeError(f"worker process exited before RPC became ready (pid={pid})")
            try:
                out = self._ipc_call(
                    reg=reg,
                    payload={"kind": "hello", "engine_id": str(reg.get("engine_id") or "")},
                    timeout_seconds=min(5.0, max(0.25, interval)),
                )
                if str(out.get("status") or "").strip().lower() == "ok":
                    return {
                        "status": "ok",
                        "attempts": attempts,
                        "ready_at": time.time(),
                        "worker": dict(out or {}),
                    }
                last_error = str(out.get("message") or out.get("status") or "worker_not_ready")
            except Exception as exc:
                last_error = str(exc)
            time.sleep(interval)
        raise TimeoutError(f"worker RPC did not become ready within {float(timeout_seconds or 600.0):.1f}s: {last_error}")

    def connect_from_config(self, *, config_path: str, engine_id: Optional[str] = None, model_path: Optional[str] = None) -> Dict[str, Any]:
        progress_events: List[Dict[str, Any]] = [
            self._progress_event("connect.resolve_config", "running", "Resolving engine config"),
        ]
        selected = self._resolve_json_config_path(config_path)
        cfg = self._merge_default_and_selected_config(config_path)
        if not isinstance(cfg, dict):
            cfg = {}
        base_name = self._safe_config_name(Path(selected).stem or "engine")
        requested = self._safe_config_name(engine_id) if str(engine_id or "").strip() else ""
        eid = self._next_engine_id(requested or base_name)
        worker_class = self._classify_connect_worker_class(
            config_path=config_path,
            payload={"model_path": model_path},
        )

        effective_model_path: Optional[str] = None
        if worker_class == "generic":
            progress_events.append(
                self._progress_event("connect.resolve_model", "skipped", "Generic worker profile does not require model selection")
            )
        else:
            configured_model = (
                ((cfg.get("engine_params") or {}).get("base_model_path") if isinstance(cfg.get("engine_params"), dict) else None)
                or cfg.get("base_model_path")
                or cfg.get("model")
                or cfg.get("base_model_name_or_path")
            )
            effective_model_path = str(model_path or configured_model or "").strip() or None
            if not effective_model_path:
                progress_events.append(
                    self._progress_event("connect.resolve_model", "needs_input", "No model path configured")
                )
                return {
                    "status": "needs_model",
                    "stage": "needs_model",
                    "engine_id": eid,
                    "config_path": str(selected),
                    "models": self.models_from_config(config_path),
                    "message": "Config loaded but no model is configured. Select a model folder and connect again.",
                    "progress_events": progress_events,
                }
            progress_events.append(
                self._progress_event("connect.resolve_model", "completed", "Model selected", model_path=effective_model_path)
            )
        progress_events.append(
            self._progress_event("connect.build_spawn_spec", "running", "Preparing engine spawn spec")
        )
        if worker_class == "generic":
            spawn_spec = self._build_generic_spawn_spec(
                engine_id=eid,
                config_path=str(selected),
                config_payload=cfg,
            )
        else:
            spawn_spec = self._build_engine_spawn_spec(
                engine_id=eid,
                config_path=str(selected),
                model_path=str(effective_model_path),
            )
        if spawn_spec.get("error"):
            progress_events.append(
                self._progress_event("connect.build_spawn_spec", "failed", str(spawn_spec.get("error") or "spawn spec failed"))
            )
            return {
                "status": "failed",
                "stage": "failed",
                "engine_id": eid,
                "config_path": str(selected),
                "model_path": effective_model_path,
                "worker_class": worker_class,
                "reason": str(spawn_spec.get("error_kind") or "engine_spawn_error"),
                "message": str(spawn_spec.get("error") or "engine spawn spec build failed"),
                "progress_events": progress_events,
            }
        progress_events.append(
            self._progress_event("connect.build_spawn_spec", "completed", "Spawn spec built")
        )
        progress_events.append(
            self._progress_event("connect.spawn_engine", "running", "Starting engine process")
        )
        try:
            rec = self.spawn(
                engine_id=eid,
                command=list(spawn_spec.get("command") or []),
                cwd=spawn_spec.get("cwd"),
                env=dict(spawn_spec.get("env") or {}),
                worker_auth_token=str(spawn_spec.get("worker_auth_token") or "").strip() or None,
                worker_auth_header=str(spawn_spec.get("worker_auth_header") or "").strip() or None,
                worker_ipc_family=str(spawn_spec.get("worker_ipc_family") or "").strip() or None,
                worker_ipc_address=str(spawn_spec.get("worker_ipc_address") or "").strip() or None,
                worker_profile_class=worker_class,
            )
            progress_events.append(
                self._progress_event("connect.spawn_engine", "completed", "Engine started", engine_id=eid)
            )
            ready: Optional[Dict[str, Any]] = None
            if worker_class != "generic" and str(rec.get("worker_transport") or "").strip().lower() == "ipc":
                progress_events.append(
                    self._progress_event(
                        "connect.worker_ready",
                        "running",
                        "Loading model and waiting for worker RPC readiness",
                        engine_id=eid,
                    )
                )
                try:
                    ready = self._wait_for_worker_rpc_ready(
                        dict(rec),
                        timeout_seconds=float(cfg.get("worker_ready_timeout_seconds") or 600.0),
                    )
                    progress_events.append(
                        self._progress_event(
                            "connect.worker_ready",
                            "completed",
                            "Worker RPC is ready",
                            engine_id=eid,
                            attempts=int(ready.get("attempts") or 0),
                        )
                    )
                except Exception as exc:
                    progress_events.append(
                        self._progress_event("connect.worker_ready", "failed", str(exc), engine_id=eid)
                    )
                    return {
                        "status": "failed",
                        "stage": "failed",
                        "engine_id": eid,
                        "config_path": str(selected),
                        "model_path": effective_model_path,
                        "worker_class": worker_class,
                        "reason": "worker_not_ready",
                        "message": str(exc),
                        "managed_engine": rec,
                        "progress_events": progress_events,
                    }
            return {
                "status": "ok",
                "stage": "completed",
                "engine_id": eid,
                "config_path": str(selected),
                "model_path": effective_model_path,
                "worker_class": worker_class,
                "managed_engine": rec,
                "worker_ready": ready,
                "progress_events": progress_events,
            }
        except Exception as e:
            progress_events.append(
                self._progress_event("connect.spawn_engine", "failed", str(e))
            )
            return {
                "status": "failed",
                "stage": "failed",
                "engine_id": eid,
                "config_path": str(selected),
                "model_path": effective_model_path,
                "worker_class": worker_class,
                "reason": "spawn_failed",
                "message": str(e),
                "progress_events": progress_events,
            }

    def inspect_engine_capabilities(self, engine_id: str, endpoint: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        reg = self._require_ipc_registration(eid, command_label="inspect-capabilities")
        out = self._ipc_call(reg=reg, payload={"kind": "hello", "engine_id": eid}, timeout_seconds=5.0)
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "inspect_failed"))
        return {
            "engine_id": eid,
            "endpoint": str(reg.get("endpoint") or "ipc://local"),
            "checked_at": time.time(),
            "supported": {
                "health": True,
                "capabilities": True,
                "inference": True,
                "ws": False,
                "rpc": True,
                "async_rpc": bool(out.get("async_rpc", True)),
                "cancellation": bool(out.get("cancellation", True)),
            },
            "worker": dict(out or {}),
        }

    def _find_registration(self, engine_id: str) -> Optional[Dict[str, Any]]:
        eid = str(engine_id or "").strip()
        for row in self._read_engines():
            if str(row.get("engine_id") or "") == eid:
                return dict(row)
        return None

    def _probe_registration_reachability(
        self,
        item: Dict[str, Any],
        *,
        timeout_seconds: float = 0.35,
    ) -> Dict[str, Any]:
        checked_at = time.time()
        transport = str(item.get("worker_transport") or "").strip().lower()
        if transport != "ipc":
            return {
                "reachable": False,
                "checked_at": checked_at,
                "transport": transport or None,
                "probe": "unsupported_transport",
                "error": "reachability_probe_not_supported",
            }
        try:
            out = self._ipc_call(
                reg=item,
                payload={"kind": "hello", "engine_id": str(item.get("engine_id") or "")},
                timeout_seconds=max(0.1, float(timeout_seconds or 0.35)),
            )
            return {
                "reachable": str(out.get("status") or "").strip().lower() == "ok",
                "checked_at": checked_at,
                "transport": "ipc",
                "probe": "hello",
            }
        except Exception as exc:
            return {
                "reachable": False,
                "checked_at": checked_at,
                "transport": "ipc",
                "probe": "hello",
                "error": str(exc),
            }

    def discover_running(
        self,
        *,
        prune_stale: bool = True,
        include_progress: bool = False,
        include_reachability: bool = True,
        reachability_timeout_seconds: float = 0.35,
    ) -> Any:
        progress_events: List[Dict[str, Any]] = [
            self._progress_event("discover.read_registry", "running", "Reading managed engine registry"),
        ]
        rows = self._read_engines()
        out: List[Dict[str, Any]] = []
        stale_ids: List[str] = []
        now = time.time()
        reachable_count = 0
        for row in rows:
            item = dict(row)
            pid = int(item.get("pid") or 0)
            alive = self._pid_alive(pid)
            item["alive"] = alive
            item["uptime_seconds"] = max(0.0, now - float(item.get("spawned_at") or now))
            item["reachable"] = False
            if alive and include_reachability:
                reachability = self._probe_registration_reachability(
                    item,
                    timeout_seconds=reachability_timeout_seconds,
                )
                item["reachable"] = bool(reachability.get("reachable", False))
                item["reachability"] = reachability
                if item["reachable"]:
                    reachable_count += 1
            out.append(item)
            if not alive:
                stale_ids.append(str(item.get("engine_id") or ""))
        if prune_stale and stale_ids:
            keep = [r for r in rows if str(r.get("engine_id") or "") not in set(stale_ids)]
            self._write_engines(keep)
            out = [x for x in out if x.get("alive")]
        out.sort(key=lambda x: str(x.get("engine_id") or ""))
        progress_events.append(
            self._progress_event(
                "discover.complete",
                "completed",
                "Discovery complete",
                engines=len(out),
                reachable=reachable_count,
                stale_pruned=len(stale_ids) if prune_stale else 0,
            )
        )
        if include_progress:
            return {
                "status": "ok",
                "stage": "completed",
                "engines": out,
                "progress_events": progress_events,
            }
        return out

    def get_registration(self, engine_id: str) -> Optional[Dict[str, Any]]:
        return self._find_registration(engine_id)

    def register_spawned(
        self,
        *,
        engine_id: str,
        pid: int,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        worker_auth_token: Optional[str] = None,
        worker_auth_header: Optional[str] = None,
        worker_ipc_family: Optional[str] = None,
        worker_ipc_address: Optional[str] = None,
        worker_profile_class: Optional[str] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        sandbox_runtime: Optional[Dict[str, Any]] = None,
        executor_kind: Optional[str] = None,
        bundle: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, Any]] = None,
        tool_access: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        source: str = "engine_host_spawned",
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        record = {
            "engine_id": eid,
            "pid": int(pid or 0),
            "command": [str(x) for x in (command or [])],
            "cwd": str(cwd) if cwd else None,
            "env": {str(k): str(v) for k, v in dict(env or {}).items()},
            "spawned_at": time.time(),
            "owner_host_pid": os.getpid(),
            "source": str(source or "engine_host_spawned"),
            "endpoint": "ipc://local",
            "worker_auth_token": str(worker_auth_token or "").strip() or None,
            "worker_auth_header": str(worker_auth_header or "").strip() or None,
            "worker_transport": "ipc",
            "worker_ipc_family": str(worker_ipc_family or "").strip() or None,
            "worker_ipc_address": str(worker_ipc_address or "").strip() or None,
            "worker_profile_class": self._normalize_worker_profile_class(worker_profile_class),
            "sandbox_policy": dict(sandbox_policy or {}) if isinstance(sandbox_policy, dict) else None,
            "sandbox_runtime": dict(sandbox_runtime or {}) if isinstance(sandbox_runtime, dict) else None,
            "executor_kind": str(executor_kind or "").strip() or None,
            "bundle": dict(bundle or {}) if isinstance(bundle, dict) else None,
            "environment": dict(environment or {}) if isinstance(environment, dict) else None,
            "tool_access": dict(tool_access or {}) if isinstance(tool_access, dict) else None,
            "capabilities": dict(capabilities or {}) if isinstance(capabilities, dict) else None,
            "log_path": str(self._engine_log_path(eid)),
        }
        rows = [r for r in self._read_engines() if str(r.get("engine_id") or "") != eid]
        rows.append(record)
        self._write_engines(rows)
        return record

    def _sandbox_fs_for_engine(self, engine_id: str) -> BrokeredFilesystem:
        reg = self._find_registration(str(engine_id or "").strip())
        if not reg:
            raise ValueError("engine_id is not registered")
        policy = WorkerSandboxPolicy.from_mapping(dict(reg.get("sandbox_policy") or {}))
        if not policy.enabled:
            raise PermissionError("sandbox_not_enabled")
        if not bool(policy.brokered_io.filesystem):
            raise PermissionError("brokered_filesystem_disabled")
        return BrokeredFilesystem(policy)

    def _sandbox_http_for_engine(self, engine_id: str) -> HostBrokeredHttpClient:
        reg = self._find_registration(str(engine_id or "").strip())
        if not reg:
            raise ValueError("engine_id is not registered")
        policy = WorkerSandboxPolicy.from_mapping(dict(reg.get("sandbox_policy") or {}))
        if not policy.enabled:
            raise PermissionError("sandbox_not_enabled")
        if not bool(policy.brokered_io.http):
            raise PermissionError("brokered_http_disabled")
        return HostBrokeredHttpClient(policy)

    def spawn(
        self,
        *,
        engine_id: str,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        worker_auth_token: Optional[str] = None,
        worker_auth_header: Optional[str] = None,
        worker_ipc_family: Optional[str] = None,
        worker_ipc_address: Optional[str] = None,
        worker_profile_class: Optional[str] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        executor_kind: Optional[str] = None,
        bundle: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, Any]] = None,
        tool_access: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not list(command or []):
            raise ValueError("command is required")
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        allocated_family, allocated_address = self._allocate_ipc_address(eid)
        ipc_family = str(worker_ipc_family or "").strip() or allocated_family
        ipc_address = str(worker_ipc_address or "").strip() or allocated_address
        auth_token = str(worker_auth_token or "").strip() or secrets.token_urlsafe(24)
        auth_header = str(worker_auth_header or "").strip() or "X-MP13-Host-Token"
        base_cmd = [str(x) for x in list(command or []) if str(x).strip()]
        if "--ipc-family" not in base_cmd:
            base_cmd.extend(["--ipc-family", ipc_family])
        if "--ipc-address" not in base_cmd:
            base_cmd.extend(["--ipc-address", ipc_address])
        merged_env = dict(os.environ) | {str(k): str(v) for k, v in dict(env or {}).items()}
        merged_env["MP13_ENGINE_HOST_TOKEN"] = auth_token
        merged_env["MP13_ENGINE_HOST_TOKEN_HEADER"] = auth_header
        merged_env["MP13_ENGINE_TRANSPORT"] = "ipc"
        merged_env["MP13_WORKER_IPC_FAMILY"] = ipc_family
        merged_env["MP13_WORKER_IPC_ADDRESS"] = ipc_address
        if str(executor_kind or "").strip() == "toolbox_executor":
            merged_env["MP13_TOOLBOX_EXECUTOR_ENGINE_ID"] = eid
            merged_env["MP13_HOSTING_ENGINES_STATE_FILE"] = str(self.engines_state_file)
            merged_env["MP13_HOSTING_CONTROL_STATE_FILE"] = str(self.control_state_file)
        log_path = self._engine_log_path(str(engine_id or ""))
        normalized_sandbox = WorkerSandboxPolicy.from_mapping(sandbox_policy)
        launched = self._launch_worker_process(
            WorkerLaunchRequest(
                engine_id=eid,
                command=base_cmd,
                cwd=Path(str(cwd)).expanduser().resolve() if cwd else None,
                env=merged_env,
                log_path=log_path,
                sandbox_policy=normalized_sandbox,
            )
        )
        persisted_env = {str(k): str(v) for k, v in dict(env or {}).items()}
        for key in [
            "MP13_ENGINE_HOST_TOKEN",
            "MP13_ENGINE_HOST_TOKEN_HEADER",
            "MP13_ENGINE_TRANSPORT",
            "MP13_WORKER_IPC_FAMILY",
            "MP13_WORKER_IPC_ADDRESS",
        ]:
            persisted_env[key] = str(launched.persisted_env.get(key) or merged_env.get(key) or "")
        if str(executor_kind or "").strip() == "toolbox_executor":
            for key in [
                "MP13_TOOLBOX_EXECUTOR_ENGINE_ID",
                "MP13_HOSTING_ENGINES_STATE_FILE",
                "MP13_HOSTING_CONTROL_STATE_FILE",
            ]:
                persisted_env[key] = str(launched.persisted_env.get(key) or merged_env.get(key) or "")
        return self.register_spawned(
            engine_id=eid,
            pid=int(launched.pid),
            command=list(launched.command),
            cwd=cwd,
            env=persisted_env,
            worker_auth_token=auth_token,
            worker_auth_header=auth_header,
            worker_ipc_family=ipc_family,
            worker_ipc_address=ipc_address,
            worker_profile_class=worker_profile_class,
            sandbox_policy=normalized_sandbox.to_dict(),
            sandbox_runtime=dict(launched.runtime),
            executor_kind=executor_kind,
            bundle=bundle,
            environment=environment,
            tool_access=tool_access,
            capabilities=capabilities,
        )

    def remove_registration(self, engine_id: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        rows = self._read_engines()
        kept = [r for r in rows if str(r.get("engine_id") or "") != eid]
        changed = len(kept) != len(rows)
        if changed:
            self._write_engines(kept)
        return {"engine_id": eid, "removed": changed}

    def shutdown(self, engine_id: str, *, timeout_seconds: float = 8.0) -> Dict[str, Any]:
        entry = self._find_registration(engine_id)
        if not entry:
            return {"status": "not_found", "engine_id": str(engine_id), "alive": False}
        pid = int(entry.get("pid") or 0)
        eid = str(entry.get("engine_id") or engine_id)
        if pid <= 0:
            self.remove_registration(eid)
            return {"status": "invalid_pid", "engine_id": eid, "alive": False}
        if not self._pid_alive(pid):
            self.remove_registration(eid)
            return {"status": "already_stopped", "engine_id": eid, "pid": pid, "alive": False}
        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.time() + max(0.1, float(timeout_seconds))
            while time.time() < deadline:
                if not self._pid_alive(pid):
                    break
                time.sleep(0.1)
        except Exception:
            pass
        if self._pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        alive = self._pid_alive(pid)
        if not alive:
            self.remove_registration(eid)
        return {"status": "stopped" if not alive else "stop_failed", "engine_id": eid, "pid": pid, "alive": alive}

    def ensure_running(self, engine_id: str) -> Dict[str, Any]:
        entry = self._find_registration(engine_id)
        if not entry:
            return {"status": "not_found", "engine_id": str(engine_id), "alive": False}
        eid = str(entry.get("engine_id") or engine_id)
        pid = int(entry.get("pid") or 0)
        command = [str(x) for x in list(entry.get("command") or []) if str(x).strip()]
        cwd = entry.get("cwd")
        endpoint = entry.get("endpoint")
        env = {str(k): str(v) for k, v in dict(entry.get("env") or {}).items()}
        worker_auth_token = str(entry.get("worker_auth_token") or "").strip() or None
        worker_auth_header = str(entry.get("worker_auth_header") or "").strip() or None
        worker_ipc_family = str(entry.get("worker_ipc_family") or "").strip() or None
        worker_ipc_address = str(entry.get("worker_ipc_address") or "").strip() or None
        worker_profile_class = str(entry.get("worker_profile_class") or "").strip() or None
        sandbox_policy_raw = dict(entry.get("sandbox_policy") or {})
        if pid > 0 and self._pid_alive(pid):
            return {"status": "running", "engine_id": eid, "pid": pid, "alive": True, "endpoint": endpoint}
        if not command:
            return {
                "status": "cannot_respawn",
                "engine_id": eid,
                "pid": pid,
                "alive": False,
                "reason": "missing_command_metadata",
                "endpoint": endpoint,
            }
        normalized_sandbox = WorkerSandboxPolicy.from_mapping(sandbox_policy_raw)
        launched = self._launch_worker_process(
            WorkerLaunchRequest(
                engine_id=eid,
                command=command,
                cwd=Path(str(cwd)).expanduser().resolve() if cwd else None,
                env=(dict(os.environ) | env),
                log_path=self._engine_log_path(eid),
                sandbox_policy=normalized_sandbox,
            )
        )
        reg = self.register_spawned(
            engine_id=eid,
            pid=int(launched.pid),
            command=command,
            cwd=str(cwd) if cwd else None,
            env=env,
            worker_auth_token=worker_auth_token,
            worker_auth_header=worker_auth_header,
            worker_ipc_family=worker_ipc_family,
            worker_ipc_address=worker_ipc_address,
            worker_profile_class=worker_profile_class,
            sandbox_policy=normalized_sandbox.to_dict(),
            sandbox_runtime=dict(launched.runtime),
        )
        return {
            "status": "respawned",
            "engine_id": eid,
            "previous_pid": pid,
            "pid": int(reg.get("pid") or 0),
            "alive": True,
            "endpoint": reg.get("endpoint"),
        }
