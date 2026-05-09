"""Engine lifecycle and registration helpers for the engine host service."""
from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .._process_utils import terminate_process_tree
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
        existing = set()
        for row in self._read_engines():
            existing.add(str(row.get("engine_id") or ""))
            existing.add(str(row.get("worker_id") or ""))
            existing.add(str(row.get("model_instance_id") or ""))
            for binding in list(row.get("config_bindings") or []):
                existing.add(str((binding or {}).get("engine_id") or ""))
        if base_name not in existing:
            return base_name
        idx = 2
        while f"{base_name}_{idx}" in existing:
            idx += 1
        return f"{base_name}_{idx}"

    @staticmethod
    def _canonical_path_value(value: Optional[str]) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            return str(Path(raw).expanduser().resolve())
        except Exception:
            return raw

    @staticmethod
    def _process_image_path(pid: int) -> str:
        p = int(pid or 0)
        if p <= 0:
            return ""
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            open_process = kernel32.OpenProcess
            open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            open_process.restype = wintypes.HANDLE
            query_full_process_image_name = kernel32.QueryFullProcessImageNameW
            query_full_process_image_name.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.LPWSTR,
                ctypes.POINTER(wintypes.DWORD),
            ]
            query_full_process_image_name.restype = wintypes.BOOL
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL

            handle = open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, p)
            if not handle:
                return ""
            try:
                size = wintypes.DWORD(32768)
                buf = ctypes.create_unicode_buffer(size.value)
                if not query_full_process_image_name(handle, 0, buf, ctypes.byref(size)):
                    return ""
                return str(buf.value or "").strip()
            finally:
                close_handle(handle)
        try:
            return str(Path(f"/proc/{p}/exe").resolve())
        except Exception:
            return ""

    def _registration_pid_matches_command(self, item: Dict[str, Any], pid: int) -> Dict[str, Any]:
        command = list(item.get("command") or [])
        expected = str(command[0] if command else "").strip()
        if not expected:
            return {"matches": True}
        expected_path = Path(expected)
        if not expected_path.is_absolute():
            return {"matches": True, "expected_executable": expected}
        actual = self._process_image_path(pid)
        if not actual:
            return {"matches": True, "expected_executable": expected}
        expected_norm = os.path.normcase(os.path.abspath(expected))
        actual_norm = os.path.normcase(os.path.abspath(actual))
        matches = expected_norm == actual_norm
        return {
            "matches": matches,
            "expected_executable": expected,
            "actual_executable": actual,
            "reason": None if matches else "pid_reused_by_different_process",
        }

    @staticmethod
    def _reachability_indicates_missing_ipc_endpoint(reachability: Dict[str, Any]) -> bool:
        error = str((reachability or {}).get("error") or "").strip().lower()
        return "worker ipc endpoint is unavailable" in error

    def _worker_id_for_model(self, model_path: str) -> str:
        stem = self._safe_config_name(Path(str(model_path or "model")).name or "model")
        digest = hashlib.sha256(str(model_path or "").encode("utf-8", errors="ignore")).hexdigest()[:10]
        return self._next_engine_id(f"worker-{stem}-{digest}")

    def _model_instance_id_for_model(self, model_path: str, requested: str = "") -> str:
        if requested:
            return self._next_engine_id(requested)
        stem = self._safe_config_name(Path(str(model_path or "model")).name or "model")
        digest = hashlib.sha256(str(model_path or "").encode("utf-8", errors="ignore")).hexdigest()[:10]
        return self._next_engine_id(f"model-{stem}-{digest}")

    def _config_binding_id(self, engine_id: str, canonical_config_path: str) -> str:
        return self._registration_binding_id(engine_id, canonical_config_path)

    def _runtime_profile_for_connect(self, *, worker_class: str, config_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "worker_profile_class": self._normalize_worker_profile_class(worker_class),
            "python": self._engine_python_executable() if worker_class == "model" else None,
        }

    def _compatible_runtime_profile(self, reg: Dict[str, Any], profile: Dict[str, Any]) -> bool:
        if self._normalize_worker_profile_class(str(reg.get("worker_profile_class") or "")) != self._normalize_worker_profile_class(str(profile.get("worker_profile_class") or "")):
            return False
        wanted_python = str(profile.get("python") or "").strip()
        if not wanted_python:
            return True
        env = dict(reg.get("env") or {}) if isinstance(reg.get("env"), dict) else {}
        existing_python = str(env.get("MP13_ENGINE_PYTHON") or "").strip()
        return not existing_python or existing_python == wanted_python

    def _find_reusable_model_worker(
        self,
        *,
        canonical_model_path: str,
        runtime_profile: Dict[str, Any],
        reachability_timeout_seconds: float = 0.35,
    ) -> Optional[Dict[str, Any]]:
        target_model = str(canonical_model_path or "").strip()
        if not target_model:
            return None
        for row in self._read_engines():
            reg = dict(row or {})
            if self._normalize_worker_profile_class(str(reg.get("worker_profile_class") or "")) != "model":
                continue
            pid = int(reg.get("pid") or 0)
            if pid <= 0 or not self._pid_alive(pid):
                continue
            if str(reg.get("worker_transport") or "").strip().lower() != "ipc":
                continue
            models = list(reg.get("loaded_models") or [])
            if not any(str((model or {}).get("canonical_model_path") or "").strip() == target_model for model in models):
                continue
            reachability = self._probe_registration_reachability(reg, timeout_seconds=reachability_timeout_seconds)
            if bool(reachability.get("reachable", False)):
                reg["reachable"] = True
                reg["reachability"] = dict(reachability or {})
                return reg
        return None

    def _find_reusable_config_binding_worker(
        self,
        *,
        canonical_config_path: str,
        runtime_profile: Dict[str, Any],
        reachability_timeout_seconds: float = 0.35,
    ) -> Optional[Dict[str, Any]]:
        target_config = str(canonical_config_path or "").strip()
        if not target_config:
            return None
        for row in self._read_engines():
            reg = dict(row or {})
            if not self._compatible_runtime_profile(reg, runtime_profile):
                continue
            pid = int(reg.get("pid") or 0)
            if pid <= 0 or not self._pid_alive(pid):
                continue
            if str(reg.get("worker_transport") or "").strip().lower() != "ipc":
                continue
            bindings = [dict(item or {}) for item in list(reg.get("config_bindings") or []) if isinstance(item, dict)]
            if not any(str(binding.get("canonical_config_path") or "").strip() == target_config for binding in bindings):
                continue
            reachability = self._probe_registration_reachability(reg, timeout_seconds=reachability_timeout_seconds)
            if bool(reachability.get("reachable", False)):
                reg["reachable"] = True
                reg["reachability"] = dict(reachability or {})
                return reg
        return None

    def _find_idle_model_worker(
        self,
        *,
        runtime_profile: Dict[str, Any],
        target_worker_id: str = "",
        reachability_timeout_seconds: float = 0.35,
    ) -> Optional[Dict[str, Any]]:
        wanted = str(target_worker_id or "").strip()
        candidates: List[Dict[str, Any]] = []
        for row in self._read_engines():
            reg = dict(row or {})
            worker_id = str(reg.get("worker_id") or reg.get("engine_id") or "").strip()
            if wanted and wanted not in {worker_id, str(reg.get("engine_id") or "").strip()}:
                continue
            if not self._compatible_runtime_profile(reg, runtime_profile):
                continue
            if str(reg.get("worker_transport") or "").strip().lower() != "ipc":
                continue
            if list(reg.get("loaded_models") or []):
                continue
            pid = int(reg.get("pid") or 0)
            if pid <= 0 or not self._pid_alive(pid):
                continue
            reachability = self._probe_registration_reachability(reg, timeout_seconds=reachability_timeout_seconds)
            if bool(reachability.get("reachable", False)):
                reg["reachable"] = True
                reg["reachability"] = dict(reachability or {})
                candidates.append(reg)
        if not candidates:
            return None
        candidates.sort(key=lambda item: float(item.get("spawned_at") or 0.0), reverse=True)
        return candidates[0]

    def _model_instance_for_engine_id(self, reg: Dict[str, Any], engine_id: str) -> str:
        eid = str(engine_id or "").strip()
        if not eid:
            return str(reg.get("model_instance_id") or reg.get("engine_id") or "").strip()
        if eid == str(reg.get("engine_id") or "").strip():
            return str(reg.get("model_instance_id") or reg.get("engine_id") or "").strip() or eid
        for binding in list(reg.get("config_bindings") or []):
            row = dict(binding or {})
            if str(row.get("engine_id") or "").strip() == eid:
                return str(row.get("model_instance_id") or reg.get("model_instance_id") or reg.get("engine_id") or "").strip() or eid
        for model in list(reg.get("loaded_models") or []):
            row = dict(model or {})
            if str(row.get("engine_id") or "").strip() == eid or str(row.get("model_instance_id") or "").strip() == eid:
                return str(row.get("model_instance_id") or eid).strip()
        return eid

    def _upsert_config_binding(
        self,
        *,
        worker_id: str,
        engine_id: str,
        model_instance_id: str,
        config_path: str,
        canonical_config_path: str,
    ) -> Tuple[Dict[str, Any], bool]:
        rows = self._read_engines()
        updated: List[Dict[str, Any]] = []
        changed = False
        binding_id = self._config_binding_id(engine_id, canonical_config_path or config_path)
        binding = {
            "config_binding_id": binding_id,
            "engine_id": str(engine_id),
            "model_instance_id": str(model_instance_id),
            "config_path": str(config_path),
            "canonical_config_path": str(canonical_config_path or config_path),
            "created_at": time.time(),
        }
        result: Dict[str, Any] = {}
        for row in rows:
            reg = dict(row or {})
            if str(reg.get("worker_id") or reg.get("engine_id") or "") != str(worker_id):
                updated.append(reg)
                continue
            bindings = [dict(item or {}) for item in list(reg.get("config_bindings") or []) if isinstance(item, dict)]
            existing_idx = next((idx for idx, item in enumerate(bindings) if str(item.get("engine_id") or "") == str(engine_id)), -1)
            if existing_idx >= 0:
                binding = dict(bindings[existing_idx])
            else:
                bindings.append(binding)
                reg["config_bindings"] = bindings
                for model in list(reg.get("loaded_models") or []):
                    if str((model or {}).get("model_instance_id") or "") == str(model_instance_id):
                        ids = [str(x) for x in list((model or {}).get("config_binding_ids") or []) if str(x).strip()]
                        if binding_id not in ids:
                            ids.append(binding_id)
                        model["config_binding_ids"] = ids
                changed = True
            result = reg
            updated.append(reg)
        if changed:
            self._write_engines(updated)
        return result, changed

    def _finalize_model_registration(
        self,
        rec: Dict[str, Any],
        *,
        worker_id: str,
        model_instance_id: str,
        config_path: str,
        canonical_config_path: str,
        model_path: str,
        canonical_model_path: str,
    ) -> Dict[str, Any]:
        eid = str(rec.get("engine_id") or model_instance_id).strip()
        wid = str(worker_id or eid).strip()
        mid = str(model_instance_id or eid).strip()
        binding_id = self._config_binding_id(eid, canonical_config_path or config_path)
        updated = dict(rec or {})
        updated["worker_id"] = wid
        updated["model_instance_id"] = mid
        updated["model_path"] = str(model_path or "")
        updated["canonical_model_path"] = str(canonical_model_path or model_path or "")
        updated["config_path"] = str(config_path or "")
        updated["canonical_config_path"] = str(canonical_config_path or config_path or "")
        updated["loaded_models"] = [
            {
                "model_instance_id": mid,
                "engine_id": eid,
                "model_path": str(model_path or ""),
                "canonical_model_path": str(canonical_model_path or model_path or ""),
                "loaded_at": float(updated.get("spawned_at") or time.time()),
                "runtime_profile": str(updated.get("worker_profile_class") or "model"),
                "config_binding_ids": [binding_id],
            }
        ]
        updated["config_bindings"] = [
            {
                "config_binding_id": binding_id,
                "engine_id": eid,
                "model_instance_id": mid,
                "config_path": str(config_path or ""),
                "canonical_config_path": str(canonical_config_path or config_path or ""),
                "created_at": float(updated.get("spawned_at") or time.time()),
            }
        ]
        rows = []
        replaced = False
        for row in self._read_engines():
            if str(row.get("engine_id") or "") == eid:
                rows.append(updated)
                replaced = True
            else:
                rows.append(dict(row or {}))
        if replaced:
            self._write_engines(rows)
        return updated

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

    def connect_from_config(
        self,
        *,
        config_path: str,
        engine_id: Optional[str] = None,
        model_path: Optional[str] = None,
        force_new_worker: bool = False,
        launch_policy: Optional[str] = None,
        target_worker_id: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        progress_events: List[Dict[str, Any]] = [
            self._progress_event("connect.resolve_config", "running", "Resolving engine config"),
        ]

        def _emit_progress(event: Dict[str, Any]) -> None:
            if not callable(progress_callback):
                return
            try:
                progress_callback(dict(event or {}))
            except Exception:
                pass

        _emit_progress(progress_events[-1])
        selected = self._resolve_json_config_path(config_path)
        cfg = self._merge_default_and_selected_config(config_path)
        if not isinstance(cfg, dict):
            cfg = {}
        base_name = self._safe_config_name(Path(selected).stem or "engine")
        requested = self._safe_config_name(engine_id) if str(engine_id or "").strip() else ""
        requested_worker_id = str(target_worker_id or "").strip()
        worker_class = self._classify_connect_worker_class(
            config_path=config_path,
            payload={"model_path": model_path},
        )
        launch = str(launch_policy or "").strip().lower()
        fresh_worker = bool(force_new_worker) or launch in {"fresh_worker", "force_new_worker", "new_worker"}
        canonical_config_path = self._canonical_path_value(str(selected))
        eid = self._next_engine_id(requested or base_name) if worker_class == "generic" else ""

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
            effective_model_raw = str(model_path or configured_model or "").strip()
            effective_model_path = (
                self._resolve_model_path_from_config_value(effective_model_raw, config_path=config_path, cfg=cfg)
                if effective_model_raw
                else None
            )
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
            canonical_model_path = self._canonical_path_value(effective_model_path)
            runtime_profile = self._runtime_profile_for_connect(worker_class=worker_class, config_payload=cfg)
            if not fresh_worker:
                progress_events.append(
                    self._progress_event("connect.reconcile_worker", "running", "Searching for reusable worker")
                )
                reusable = self._find_reusable_config_binding_worker(
                    canonical_config_path=canonical_config_path,
                    runtime_profile=runtime_profile,
                )
                reusable = reusable or self._find_reusable_model_worker(
                    canonical_model_path=canonical_model_path,
                    runtime_profile=runtime_profile,
                )
                if reusable is not None:
                    models = [dict(item or {}) for item in list(reusable.get("loaded_models") or []) if isinstance(item, dict)]
                    binding_for_config = next(
                        (
                            dict(binding or {})
                            for binding in list(reusable.get("config_bindings") or [])
                            if str((binding or {}).get("canonical_config_path") or "").strip() == canonical_config_path
                        ),
                        {},
                    )
                    model_row = next(
                        (
                            model for model in models
                            if str(model.get("canonical_model_path") or "").strip() == canonical_model_path
                        ),
                        {},
                    )
                    model_instance_id = str(model_row.get("model_instance_id") or reusable.get("model_instance_id") or reusable.get("engine_id") or "").strip()
                    existing_binding = {}
                    if not requested:
                        existing_binding = next(
                            (
                                dict(binding or {})
                                for binding in list(reusable.get("config_bindings") or [])
                            if str((binding or {}).get("model_instance_id") or "").strip() == model_instance_id
                            and str((binding or {}).get("canonical_config_path") or "").strip() == canonical_config_path
                        ),
                            {},
                        )
                    if binding_for_config:
                        model_instance_id = str(binding_for_config.get("model_instance_id") or model_instance_id).strip()
                        existing_binding = binding_for_config
                    binding_engine_id = (
                        str(existing_binding.get("engine_id") or "").strip()
                        or self._next_engine_id(requested or base_name)
                    )
                    updated_reg, attached = self._upsert_config_binding(
                        worker_id=str(reusable.get("worker_id") or reusable.get("engine_id") or ""),
                        engine_id=binding_engine_id,
                        model_instance_id=model_instance_id,
                        config_path=str(selected),
                        canonical_config_path=canonical_config_path,
                    )
                    progress_events.append(
                        self._progress_event(
                            "connect.reconcile_worker",
                            "completed",
                            "Reused existing worker",
                            worker_id=str(reusable.get("worker_id") or reusable.get("engine_id") or ""),
                            engine_id=binding_engine_id,
                            model_instance_id=model_instance_id,
                            attached_config_binding=bool(attached),
                        )
                    )
                    return {
                        "status": "reused" if not attached else "attached",
                        "stage": "completed",
                        "reconciled": True,
                        "spawned": False,
                        "worker_id": str(reusable.get("worker_id") or reusable.get("engine_id") or ""),
                        "engine_id": binding_engine_id,
                        "model_instance_id": model_instance_id,
                        "config_binding_id": self._config_binding_id(binding_engine_id, canonical_config_path),
                        "config_path": str(selected),
                        "canonical_config_path": canonical_config_path,
                        "model_path": effective_model_path,
                        "canonical_model_path": canonical_model_path,
                        "worker_class": worker_class,
                        "managed_engine": updated_reg or reusable,
                        "progress_events": progress_events,
                    }
                idle_worker = self._find_idle_model_worker(
                    runtime_profile=runtime_profile,
                    target_worker_id=requested_worker_id,
                )
                if idle_worker is not None:
                    worker_id = str(idle_worker.get("worker_id") or idle_worker.get("engine_id") or "").strip()
                    model_instance_id = requested or worker_id or self._model_instance_id_for_model(str(effective_model_path))
                    worker_out = self._ipc_call(
                        reg=idle_worker,
                        payload={
                            "kind": "rpc_call",
                            "engine_id": worker_id,
                            "method": "model.load",
                            "params": {
                                "model_instance_id": model_instance_id,
                                "model_path": str(effective_model_path),
                                "config_path": str(selected),
                            },
                        },
                        timeout_seconds=float(cfg.get("worker_ready_timeout_seconds") or 600.0),
                    )
                    if str(worker_out.get("status") or "").strip().lower() == "error":
                        raise RuntimeError(str(worker_out.get("message") or "model_load_failed"))
                    updated_reg = self._finalize_model_registration(
                        dict(idle_worker),
                        worker_id=worker_id,
                        model_instance_id=model_instance_id,
                        config_path=str(selected),
                        canonical_config_path=canonical_config_path,
                        model_path=str(effective_model_path),
                        canonical_model_path=canonical_model_path,
                    )
                    progress_events.append(
                        self._progress_event(
                            "connect.reconcile_worker",
                            "completed",
                            "Loaded model into existing idle worker",
                            worker_id=worker_id,
                            engine_id=model_instance_id,
                            model_instance_id=model_instance_id,
                        )
                    )
                    return {
                        "status": "loaded_existing_worker",
                        "stage": "completed",
                        "reconciled": True,
                        "spawned": False,
                        "worker_id": worker_id,
                        "engine_id": model_instance_id,
                        "model_instance_id": model_instance_id,
                        "config_binding_id": self._config_binding_id(model_instance_id, canonical_config_path),
                        "config_path": str(selected),
                        "canonical_config_path": canonical_config_path,
                        "model_path": effective_model_path,
                        "canonical_model_path": canonical_model_path,
                        "worker_class": worker_class,
                        "managed_engine": updated_reg,
                        "worker": dict(worker_out or {}),
                        "progress_events": progress_events,
                    }
                progress_events.append(
                    self._progress_event("connect.reconcile_worker", "completed", "No reusable worker found")
                )
            else:
                progress_events.append(
                    self._progress_event("connect.reconcile_worker", "skipped", "Fresh worker launch requested")
                )
            model_instance_id = self._model_instance_id_for_model(str(effective_model_path), requested=requested)
            worker_id = self._worker_id_for_model(str(effective_model_path))
            eid = model_instance_id
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
                "worker_id": eid,
                "model_instance_id": eid,
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
            if worker_class != "generic":
                rec = self._finalize_model_registration(
                    dict(rec),
                    worker_id=worker_id,
                    model_instance_id=model_instance_id,
                    config_path=str(selected),
                    canonical_config_path=canonical_config_path,
                    model_path=str(effective_model_path),
                    canonical_model_path=self._canonical_path_value(effective_model_path),
                )
            progress_events.append(
                self._progress_event(
                    "connect.spawn_engine",
                    "completed",
                    "Engine started",
                    engine_id=eid,
                    log_path=str(rec.get("log_path") or ""),
                )
            )
            _emit_progress(progress_events[-1])
            ready: Optional[Dict[str, Any]] = None
            if worker_class != "generic" and str(rec.get("worker_transport") or "").strip().lower() == "ipc":
                progress_events.append(
                    self._progress_event(
                        "connect.worker_ready",
                        "running",
                        "Loading model and waiting for worker RPC readiness",
                        engine_id=eid,
                        log_path=str(rec.get("log_path") or ""),
                    )
                )
                _emit_progress(progress_events[-1])
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
                            log_path=str(rec.get("log_path") or ""),
                            attempts=int(ready.get("attempts") or 0),
                        )
                    )
                    _emit_progress(progress_events[-1])
                except Exception as exc:
                    progress_events.append(
                        self._progress_event("connect.worker_ready", "failed", str(exc), engine_id=eid, log_path=str(rec.get("log_path") or ""))
                    )
                    _emit_progress(progress_events[-1])
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
                "reconciled": False,
                "spawned": True,
                "worker_id": str(rec.get("worker_id") or rec.get("engine_id") or eid),
                "engine_id": eid,
                "model_instance_id": str(rec.get("model_instance_id") or eid),
                "config_binding_id": (
                    self._config_binding_id(eid, canonical_config_path)
                    if worker_class != "generic"
                    else None
                ),
                "config_path": str(selected),
                "canonical_config_path": canonical_config_path,
                "model_path": effective_model_path,
                "canonical_model_path": self._canonical_path_value(effective_model_path),
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
                "worker_id": eid,
                "model_instance_id": eid,
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
            if str(row.get("engine_id") or "") == eid or str(row.get("worker_id") or "") == eid or str(row.get("model_instance_id") or "") == eid:
                return dict(row)
            for binding in list(row.get("config_bindings") or []):
                if str((binding or {}).get("engine_id") or "").strip() == eid:
                    out = dict(row)
                    out["_route_engine_id"] = eid
                    out["_route_model_instance_id"] = str((binding or {}).get("model_instance_id") or row.get("model_instance_id") or row.get("engine_id") or "").strip()
                    out["_route_config_binding_id"] = str((binding or {}).get("config_binding_id") or "").strip()
                    return out
            for model in list(row.get("loaded_models") or []):
                if str((model or {}).get("engine_id") or "").strip() == eid or str((model or {}).get("model_instance_id") or "").strip() == eid:
                    out = dict(row)
                    out["_route_engine_id"] = eid
                    out["_route_model_instance_id"] = str((model or {}).get("model_instance_id") or eid).strip()
                    return out
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

    @staticmethod
    def _describe_registration_state(
        item: Dict[str, Any],
        *,
        alive: bool,
        include_reachability: bool,
    ) -> str:
        if not alive:
            return "stopped"
        if include_reachability and not bool(item.get("reachable", False)):
            return "unreachable"
        return "running"

    @staticmethod
    def _describe_registration_kind(item: Dict[str, Any]) -> str:
        executor_kind = str(item.get("executor_kind") or "").strip()
        worker_class = str(item.get("worker_profile_class") or "").strip().lower()
        command_text = " ".join(str(x) for x in list(item.get("command") or [])).lower()
        env = {str(k): str(v) for k, v in dict(item.get("env") or {}).items()}
        sandbox = WorkerSandboxPolicy.from_mapping(dict(item.get("sandbox_policy") or {}))

        is_toolbox = (
            executor_kind == "toolbox_executor"
            or "hosting.toolbox_executor_ipc" in command_text
            or "MP13_TOOLBOX_EXECUTOR_ENGINE_ID" in env
            or isinstance(item.get("tool_access"), dict)
        )
        if is_toolbox:
            return "tools sandbox" if sandbox.enabled else "tools worker"

        is_model = (
            worker_class == "model"
            or "MP13_MODEL_PATH" in env
            or "hosting.engine_worker_ipc" in command_text
        )
        if is_model:
            return "sandboxed model instance" if sandbox.enabled else "model instance"

        if worker_class == "generic":
            return "sandboxed worker" if sandbox.enabled else "generic worker"

        return "sandboxed worker" if sandbox.enabled else "worker"

    def _query_worker_reported_resources(self, item: Dict[str, Any]) -> Dict[str, Any]:
        kind = self._describe_registration_kind(item)
        if "model" not in kind:
            return {}
        try:
            out = self._ipc_call(
                reg=item,
                payload={
                    "kind": "rpc_call",
                    "method": "get-engine-status",
                    "params": {},
                    "engine_id": str(item.get("engine_id") or ""),
                },
                timeout_seconds=1.0,
            )
        except Exception:
            return {}
        if str(out.get("status") or "").strip().lower() != "ok":
            return {}
        result = dict(out.get("result") or {})
        data = result.get("data") if isinstance(result.get("data"), dict) else result
        gpu_info = (data or {}).get("gpu_info") if isinstance(data, dict) else None
        if not isinstance(gpu_info, list):
            return {}
        reserved_mb = 0.0
        allocated_mb = 0.0
        devices: List[str] = []
        for row in gpu_info:
            if not isinstance(row, dict):
                continue
            device_id = row.get("device_id")
            if device_id is not None:
                devices.append(f"cuda:{device_id}")
            try:
                reserved_mb += float(row.get("memory_reserved_gb") or 0.0) * 1024.0
            except Exception:
                pass
            try:
                allocated_mb += float(row.get("memory_allocated_gb") or 0.0) * 1024.0
            except Exception:
                pass
        return {
            "gpu_vram_mb": round(reserved_mb, 1),
            "gpu_allocated_mb": round(allocated_mb, 1),
            "gpu_devices": devices,
            "gpu_vram_source": "worker_torch_cuda",
        }

    def _prune_old_stopped_worker_logs(self, *, max_age_seconds: float = 3 * 24 * 60 * 60) -> None:
        logs_dir = self._logs_dir()
        try:
            if not logs_dir.exists():
                return
        except Exception:
            return
        running_log_paths = set()
        try:
            for row in self._read_engines():
                reg = dict(row or {})
                pid = int(reg.get("pid") or 0)
                if pid > 0 and self._pid_alive(pid):
                    log_path = str(reg.get("log_path") or "").strip()
                    if log_path:
                        try:
                            running_log_paths.add(str(Path(log_path).expanduser().resolve()))
                        except Exception:
                            running_log_paths.add(log_path)
        except Exception:
            running_log_paths = set()
        cutoff = time.time() - max(0.0, float(max_age_seconds or 0.0))
        try:
            candidates = list(logs_dir.glob("*.log"))
        except Exception:
            return
        for path in candidates:
            try:
                resolved = str(path.expanduser().resolve())
                if resolved in running_log_paths:
                    continue
                if float(path.stat().st_mtime) >= cutoff:
                    continue
                path.unlink()
            except Exception:
                continue

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
            age_seconds = max(0.0, now - float(item.get("spawned_at") or now))
            alive = self._pid_alive(pid)
            if alive:
                identity = self._registration_pid_matches_command(item, pid)
                if identity:
                    item["pid_identity"] = identity
                if not bool(identity.get("matches", True)):
                    alive = False
            item["alive"] = alive
            item["uptime_seconds"] = max(0.0, now - float(item.get("spawned_at") or now))
            item["process_resources"] = self._process_resource_snapshot(pid)
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
                    worker_resources = self._query_worker_reported_resources(item)
                    if worker_resources:
                        merged_resources = dict(item.get("process_resources") or {})
                        merged_resources.update(worker_resources)
                        item["process_resources"] = merged_resources
                elif (
                    age_seconds > 30.0
                    and self._reachability_indicates_missing_ipc_endpoint(reachability)
                ):
                    item["stale_reason"] = "worker_ipc_endpoint_unavailable"
                    alive = False
                    item["alive"] = False
            item["state"] = self._describe_registration_state(
                item,
                alive=alive,
                include_reachability=include_reachability,
            )
            item["kind"] = self._describe_registration_kind(item)
            item["sandbox"] = WorkerSandboxPolicy.from_mapping(
                dict(item.get("sandbox_policy") or {})
            ).summary()
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
        self._prune_old_stopped_worker_logs()
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
        termination = terminate_process_tree(pid, timeout_seconds=timeout_seconds)
        alive = self._pid_alive(pid)
        if not alive:
            self.remove_registration(eid)
        return {
            "status": "stopped" if not alive else "stop_failed",
            "engine_id": eid,
            "worker_id": str(entry.get("worker_id") or eid),
            "pid": pid,
            "alive": alive,
            "termination": termination,
        }

    def unload_model(self, engine_id: str, *, timeout_seconds: float = 30.0, shutdown_all: bool = False) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        entry = self._find_registration(eid)
        if not entry:
            return {"status": "not_found", "engine_id": eid}
        worker_id = str(entry.get("worker_id") or entry.get("engine_id") or "").strip()
        model_instance_id = self._model_instance_for_engine_id(entry, eid)
        binding_id = str(entry.get("_route_config_binding_id") or "").strip()

        def _worker_model_ids(reg: Dict[str, Any]) -> List[str]:
            described = self._ipc_call(
                reg=reg,
                payload={
                    "kind": "rpc_call",
                    "engine_id": worker_id or eid,
                    "method": "model.describe",
                    "params": {},
                },
                timeout_seconds=min(10.0, max(1.0, float(timeout_seconds or 30.0))),
            )
            if str(described.get("status") or "").strip().lower() == "error":
                raise RuntimeError(str(described.get("message") or "model_describe_failed"))
            result = dict(described.get("result") or {}) if isinstance(described.get("result"), dict) else {}
            return [
                str((item or {}).get("model_instance_id") or (item or {}).get("engine_id") or "").strip()
                for item in list(result.get("loaded_models") or [])
                if isinstance(item, dict)
            ]

        if bool(shutdown_all):
            reg = self._require_ipc_registration(worker_id or eid, command_label="unload-model")
            worker_out = self._ipc_call(
                reg=reg,
                payload={
                    "kind": "rpc_call",
                    "engine_id": worker_id or eid,
                    "method": "model.unload",
                    "params": {"shutdown_all": True},
                },
                timeout_seconds=timeout_seconds,
            )
            if str(worker_out.get("status") or "").strip().lower() == "error":
                raise RuntimeError(str(worker_out.get("message") or "model_unload_failed"))
            remaining_worker_models = [mid for mid in _worker_model_ids(reg) if mid]
            if remaining_worker_models:
                raise RuntimeError(f"worker still reports loaded models after shutdown_all: {', '.join(remaining_worker_models)}")
            updated: List[Dict[str, Any]] = []
            for row in self._read_engines():
                reg_row = dict(row or {})
                if str(reg_row.get("worker_id") or reg_row.get("engine_id") or "") != worker_id:
                    updated.append(reg_row)
                    continue
                reg_row["loaded_models"] = []
                reg_row["config_bindings"] = []
                reg_row.pop("model_path", None)
                reg_row.pop("canonical_model_path", None)
                reg_row.pop("config_path", None)
                reg_row.pop("canonical_config_path", None)
                updated.append(reg_row)
            self._write_engines(updated)
            return {
                "status": "unloaded",
                "engine_id": eid,
                "worker_id": worker_id,
                "shutdown_all": True,
                "worker_still_running": True,
                "remaining_model_count": 0,
                "worker": dict(worker_out or {}),
            }

        bindings_for_model = [
            dict(item or {})
            for item in list(entry.get("config_bindings") or [])
            if isinstance(item, dict)
            and str((item or {}).get("model_instance_id") or "").strip() == model_instance_id
        ]
        unload_engine_instance = not binding_id or len(bindings_for_model) <= 1
        worker_out: Dict[str, Any] = {}
        if unload_engine_instance:
            reg = self._require_ipc_registration(worker_id or eid, command_label="unload-model")
            worker_out = self._ipc_call(
                reg=reg,
                payload={
                    "kind": "rpc_call",
                    "engine_id": model_instance_id,
                    "method": "model.unload",
                    "params": {"model_instance_id": model_instance_id},
                },
                timeout_seconds=timeout_seconds,
            )
            if str(worker_out.get("status") or "").strip().lower() == "error":
                message = str(worker_out.get("message") or "model_unload_failed")
                if "not found" not in message.lower():
                    raise RuntimeError(message)
            remaining_worker_models = [mid for mid in _worker_model_ids(reg) if mid]
            if model_instance_id in remaining_worker_models:
                raise RuntimeError(f"worker still reports model '{model_instance_id}' after unload")

        rows = self._read_engines()
        updated: List[Dict[str, Any]] = []
        removed_binding = False
        remaining_models = 0
        for row in rows:
            reg = dict(row or {})
            if str(reg.get("worker_id") or reg.get("engine_id") or "") != worker_id:
                updated.append(reg)
                continue
            bindings = [dict(item or {}) for item in list(reg.get("config_bindings") or []) if isinstance(item, dict)]
            if binding_id:
                new_bindings = [b for b in bindings if str(b.get("config_binding_id") or "") != binding_id]
                removed_binding = len(new_bindings) != len(bindings)
                reg["config_bindings"] = new_bindings
                for model in list(reg.get("loaded_models") or []):
                    ids = [str(x) for x in list((model or {}).get("config_binding_ids") or []) if str(x).strip() and str(x).strip() != binding_id]
                    model["config_binding_ids"] = ids
                if not unload_engine_instance:
                    updated.append(reg)
                    remaining_models = len(list(reg.get("loaded_models") or []))
                    continue
            models = [dict(item or {}) for item in list(reg.get("loaded_models") or []) if isinstance(item, dict)]
            new_models = [m for m in models if str(m.get("model_instance_id") or "") != model_instance_id]
            reg["loaded_models"] = new_models
            reg["config_bindings"] = [
                b for b in bindings
                if str(b.get("model_instance_id") or "") != model_instance_id
            ]
            remaining_models = len(new_models)
            if not new_models:
                reg.pop("model_path", None)
                reg.pop("canonical_model_path", None)
                reg.pop("config_path", None)
                reg.pop("canonical_config_path", None)
            updated.append(reg)
        self._write_engines(updated)
        if binding_id and not unload_engine_instance:
            return {
                "status": "unloaded",
                "engine_id": eid,
                "worker_id": worker_id,
                "model_instance_id": model_instance_id,
                "config_binding_id": binding_id,
                "removed_binding": removed_binding,
                "worker_still_running": True,
                "remaining_model_count": remaining_models,
            }
        return {
            "status": "unloaded",
            "engine_id": eid,
            "worker_id": worker_id,
            "model_instance_id": model_instance_id,
            "config_binding_id": binding_id or None,
            "removed_binding": removed_binding if binding_id else None,
            "worker_still_running": True,
            "remaining_model_count": remaining_models,
            "worker": dict(worker_out or {}),
        }

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
        self._prune_old_stopped_worker_logs()
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
