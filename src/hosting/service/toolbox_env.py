"""Toolbox environment, consistency, repair, and GC helpers."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class ToolboxEnvironmentMixin:
    def toolbox_environment_description_list(self) -> Dict[str, Any]:
        from ..toolbox_harness import ToolboxEnvironmentManager

        state = self._read_toolboxes()
        envs = dict(state.get("environment_descriptions") or {})
        out: Dict[str, Any] = {}
        for name, row in envs.items():
            normalized = ToolboxEnvironmentManager.normalize_environment_description(dict(row or {}), name=str(name or ""))
            out[str(normalized["name"])] = normalized
        if "base" not in out:
            out["base"] = ToolboxEnvironmentManager.normalize_environment_description({}, name="base")
        return {"status": "ok", "environment_descriptions": out}

    def toolbox_environment_description_get(self, name: str) -> Dict[str, Any]:
        from ..toolbox_harness import ToolboxEnvironmentManager

        env_name = str(name or "").strip() or "base"
        state = self._read_toolboxes()
        envs = dict(state.get("environment_descriptions") or {})
        payload = dict(envs.get(env_name) or ({}) if env_name == "base" else envs.get(env_name) or {})
        if not payload and env_name != "base":
            raise ValueError(f"environment description '{env_name}' not found")
        return ToolboxEnvironmentManager.normalize_environment_description(payload, name=env_name)

    def toolbox_environment_description_effective_get(self, name: str) -> Dict[str, Any]:
        from ..toolbox_harness import ToolboxEnvironmentManager

        env_name = str(name or "").strip() or "base"
        state = self._read_toolboxes()
        envs = dict(state.get("environment_descriptions") or {})
        if env_name != "base" and env_name not in envs:
            raise ValueError(f"environment description '{env_name}' not found")
        return ToolboxEnvironmentManager.resolve_environment_description(envs, name=env_name)

    def toolbox_environment_description_upsert(
        self,
        *,
        name: str,
        base_env_name: Optional[str] = None,
        extra_packages: Optional[List[str]] = None,
        allow_online_install: bool = False,
    ) -> Dict[str, Any]:
        from ..toolbox_harness import ToolboxEnvironmentManager

        env_name = str(name or "").strip()
        if not env_name:
            raise ValueError("name is required")
        if env_name == "base" and base_env_name:
            raise ValueError("base environment cannot inherit from another environment")
        state = self._read_toolboxes()
        envs = dict(state.get("environment_descriptions") or {})
        if base_env_name and str(base_env_name).strip() not in envs and str(base_env_name).strip() != "base":
            raise ValueError(f"base environment '{base_env_name}' not found")
        normalized = ToolboxEnvironmentManager.normalize_environment_description(
            {
                "base_env_name": str(base_env_name or "").strip() or None,
                "extra_packages": list(extra_packages or []),
                "allow_online_install": bool(allow_online_install),
            },
            name=env_name,
        )
        envs[env_name] = normalized
        state["environment_descriptions"] = envs
        self._write_toolboxes(state)
        return {"status": "ok", "environment_description": normalized}

    def toolbox_environment_description_clone(
        self,
        *,
        source_name: str,
        target_name: str,
        extra_packages: Optional[List[str]] = None,
        allow_online_install: Optional[bool] = None,
    ) -> Dict[str, Any]:
        from ..toolbox_harness import ToolboxEnvironmentManager

        src = str(source_name or "").strip()
        dst = str(target_name or "").strip()
        if not src:
            raise ValueError("source_name is required")
        if not dst:
            raise ValueError("target_name is required")
        if src == dst:
            raise ValueError("target_name must differ from source_name")
        source = self.toolbox_environment_description_get(src)
        state = self._read_toolboxes()
        envs = dict(state.get("environment_descriptions") or {})
        if dst in envs:
            raise ValueError(f"environment description '{dst}' already exists")
        normalized = ToolboxEnvironmentManager.normalize_environment_description(
            {
                "base_env_name": src,
                "extra_packages": list(extra_packages if extra_packages is not None else list(source.get("extra_packages") or [])),
                "allow_online_install": bool(source.get("allow_online_install", False))
                if allow_online_install is None
                else bool(allow_online_install),
            },
            name=dst,
        )
        envs[dst] = normalized
        state["environment_descriptions"] = envs
        self._write_toolboxes(state)
        return {"status": "ok", "environment_description": normalized}

    def toolbox_environment_resolve_requirements(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env = self.toolbox_environment_description_get(environment_name)
        effective_env = self.toolbox_environment_description_effective_get(environment_name)
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        selected = {str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()}
        required_packages: List[str] = []
        seen: set[str] = set()
        for req in list(toolbox_row.get("requests") or []):
            row = dict(req or {})
            key = f"{str(row.get('module_name') or '').strip()}:{str(row.get('callable_name') or '').strip()}"
            profile = dict(row.get("sandbox_profile") or {})
            if str(profile.get("environment_name") or "base").strip() != str(environment_name or "base").strip():
                continue
            if selected and key not in selected:
                continue
            for pkg in list(profile.get("required_imports") or []):
                name = str(pkg or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    required_packages.append(name)
        for req in list(toolbox_row.get("manual_requests") or []):
            row = dict(req or {})
            key = f"manual:{str(row.get('module_name') or '').strip()}:{str(row.get('callable_name') or '').strip()}"
            profile = dict(row.get("sandbox_profile") or {})
            if str(profile.get("environment_name") or "base").strip() != str(environment_name or "base").strip():
                continue
            if selected and key not in selected:
                continue
            for pkg in list(profile.get("required_imports") or []):
                name = str(pkg or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    required_packages.append(name)
        extra_packages = [
            str(item or "").strip()
            for item in list(effective_env.get("effective_extra_packages") or [])
            if str(item or "").strip()
        ]
        missing_packages = [pkg for pkg in required_packages if pkg not in set(extra_packages)]
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": str(environment_name or "base").strip() or "base",
            "environment_lineage": list(effective_env.get("lineage") or []),
            "required_packages": required_packages,
            "configured_extra_packages": [str(item or "").strip() for item in list(env.get("extra_packages") or []) if str(item or "").strip()],
            "effective_extra_packages": extra_packages,
            "missing_packages": missing_packages,
        }

    @staticmethod
    def _toolbox_profile_required_packages(
        profile_row: Optional[Dict[str, Any]],
        *,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        selected = {str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()}
        required_packages: List[str] = []
        seen: set[str] = set()
        matched_tool_keys: List[str] = []
        for req in list(dict(profile_row or {}).get("requests") or []):
            row = dict(req or {})
            key = f"{str(row.get('module_name') or '').strip()}:{str(row.get('callable_name') or '').strip()}"
            if selected and key not in selected:
                continue
            if key:
                matched_tool_keys.append(key)
            profile = dict(row.get("sandbox_profile") or {})
            for pkg in list(profile.get("required_imports") or []):
                name = str(pkg or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    required_packages.append(name)
        for req in list(dict(profile_row or {}).get("manual_requests") or []):
            row = dict(req or {})
            key = f"manual:{str(row.get('module_name') or '').strip()}:{str(row.get('callable_name') or '').strip()}"
            if selected and key not in selected:
                continue
            if key:
                matched_tool_keys.append(key)
            profile = dict(row.get("sandbox_profile") or {})
            for pkg in list(profile.get("required_imports") or []):
                name = str(pkg or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    required_packages.append(name)
        return {
            "required_packages": required_packages,
            "tool_keys": matched_tool_keys,
        }

    @staticmethod
    def _toolbox_runtime_defaults(
        toolbox_row: Optional[Dict[str, Any]],
        *,
        python_executable: Optional[str] = None,
        worker_profile_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        existing = dict(dict(toolbox_row or {}).get("runtime") or {})
        python_value = str(
            python_executable
            if python_executable is not None
            else existing.get("python_executable") or ""
        ).strip() or None
        worker_value = str(
            worker_profile_class
            if worker_profile_class is not None
            else existing.get("worker_profile_class") or "generic"
        ).strip() or "generic"
        return {
            **existing,
            "python_executable": python_value,
            "worker_profile_class": worker_value,
        }

    @staticmethod
    def _append_toolbox_cancel_event(
        toolbox_row: Optional[Dict[str, Any]],
        *,
        engine_ids: Sequence[str],
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        respawn: bool = True,
        non_restartable: bool = False,
    ) -> Dict[str, Any]:
        row = dict(toolbox_row or {})
        runtime = dict(row.get("runtime") or {})
        events = list(runtime.get("cancel_events") or [])
        events.append(
            {
                "timestamp": time.time(),
                "engine_ids": [
                    str(item or "").strip()
                    for item in list(engine_ids or [])
                    if str(item or "").strip()
                ],
                "tool_name": str(tool_name or "").strip() or None,
                "tool_call_id": str(tool_call_id or "").strip() or None,
                "respawn": bool(respawn),
                "non_restartable": bool(non_restartable),
            }
        )
        if len(events) > 25:
            events = events[-25:]
        runtime["cancel_events"] = events
        row["runtime"] = runtime
        return row

    def _toolbox_uses_environment_name(self, toolbox_row: Optional[Dict[str, Any]], environment_name: str) -> bool:
        env_name = str(environment_name or "base").strip() or "base"
        row = dict(toolbox_row or {})
        for req in list(row.get("requests") or []):
            profile = dict(dict(req or {}).get("sandbox_profile") or {})
            if str(profile.get("environment_name") or "base").strip() == env_name:
                return True
        for req in list(row.get("manual_requests") or []):
            profile = dict(dict(req or {}).get("sandbox_profile") or {})
            if str(profile.get("environment_name") or "base").strip() == env_name:
                return True
        intrinsics = dict(row.get("intrinsics") or {})
        profile = dict(intrinsics.get("sandbox_profile") or {})
        names = [
            str(item or "").strip()
            for item in list(intrinsics.get("names") or [])
            if str(item or "").strip()
        ]
        if names and str(profile.get("environment_name") or "base").strip() == env_name:
            return True
        return False

    def _toolbox_uses_environment_dependency(self, toolbox_row: Optional[Dict[str, Any]], environment_name: str) -> bool:
        env_name = str(environment_name or "base").strip() or "base"
        row = dict(toolbox_row or {})
        names_to_check: List[str] = []
        for req in list(row.get("requests") or []):
            profile = dict(dict(req or {}).get("sandbox_profile") or {})
            name = str(profile.get("environment_name") or "base").strip() or "base"
            if name:
                names_to_check.append(name)
        for req in list(row.get("manual_requests") or []):
            profile = dict(dict(req or {}).get("sandbox_profile") or {})
            name = str(profile.get("environment_name") or "base").strip() or "base"
            if name:
                names_to_check.append(name)
        intrinsics = dict(row.get("intrinsics") or {})
        profile = dict(intrinsics.get("sandbox_profile") or {})
        intrinsic_names = [str(item or "").strip() for item in list(intrinsics.get("names") or []) if str(item or "").strip()]
        if intrinsic_names:
            names_to_check.append(str(profile.get("environment_name") or "base").strip() or "base")
        for candidate in names_to_check:
            try:
                effective = self.toolbox_environment_description_effective_get(candidate)
            except Exception:
                continue
            if env_name in {str(item or "").strip() for item in list(effective.get("lineage") or []) if str(item or "").strip()}:
                return True
        return False

    @staticmethod
    def _toolbox_tool_non_restartable(toolbox_row: Optional[Dict[str, Any]], tool_name: str) -> bool:
        target = str(tool_name or "").strip()
        if not target:
            return False
        row = dict(toolbox_row or {})
        for req in list(row.get("requests") or []):
            request = dict(req or {})
            if str(request.get("callable_name") or "").strip() == target:
                return bool(request.get("non_restartable", False))
        for req in list(row.get("manual_requests") or []):
            request = dict(req or {})
            fn = dict(dict(request.get("tool_definition") or {}).get("function") or {})
            if str(fn.get("name") or "").strip() == target:
                return bool(request.get("non_restartable", False))
        return False

    @staticmethod
    def _toolbox_tool_metadata(toolbox_row: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        row = dict(toolbox_row or {})
        intrinsics = dict(row.get("intrinsics") or {})
        hidden_intrinsics = {
            str(item or "").strip()
            for item in list(intrinsics.get("hidden_tool_names") or row.get("hidden_intrinsic_tool_names") or [])
            if str(item or "").strip()
        }
        metadata: Dict[str, Dict[str, Any]] = {}
        for req in list(row.get("requests") or []):
            request = dict(req or {})
            tool_name = str(request.get("callable_name") or "").strip()
            if not tool_name:
                continue
            metadata[tool_name] = {
                "callback_signature": dict(request.get("callback_signature") or {}) or None,
                "non_restartable": bool(request.get("non_restartable", False)),
                "hidden": False,
            }
            if isinstance(request.get("concurrency"), dict) and request.get("concurrency"):
                metadata[tool_name]["concurrency"] = dict(request["concurrency"])
        for req in list(row.get("manual_requests") or []):
            request = dict(req or {})
            fn = dict(dict(request.get("tool_definition") or {}).get("function") or {})
            tool_name = str(fn.get("name") or "").strip()
            if not tool_name:
                continue
            metadata[tool_name] = {
                "callback_signature": dict(request.get("callback_signature") or {}) or None,
                "non_restartable": bool(request.get("non_restartable", False)),
                "hidden": bool(request.get("hidden", False)),
            }
            concurrency = (
                request.get("concurrency")
                or dict(dict(request.get("tool_definition") or {}).get("function") or {}).get("concurrency")
                or dict(request.get("tool_definition") or {}).get("concurrency")
                or {}
            )
            if isinstance(concurrency, dict) and concurrency:
                metadata[tool_name]["concurrency"] = dict(concurrency)
        for tool_name in list(intrinsics.get("names") or row.get("intrinsic_tool_names") or []):
            name = str(tool_name or "").strip()
            if not name:
                continue
            metadata.setdefault(
                name,
                {
                    "callback_signature": None,
                    "non_restartable": False,
                    "hidden": name in hidden_intrinsics,
                },
            )
        return metadata

    def _rebuild_toolbox_from_persisted_state(
        self,
        *,
        toolbox_id: str,
        toolbox_row: Dict[str, Any],
        state: Dict[str, Any],
        action: str,
    ) -> Dict[str, Any]:
        from ..toolbox_harness import (
            SandboxProfileSpec,
            ToolboxAutoAssignmentRequest,
            ToolboxBundleStager,
            ToolboxManualAssignmentRequest,
            ToolboxSandboxOrchestrator,
        )

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        row = dict(toolbox_row or {})
        auto_requests = [
            ToolboxAutoAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(row.get("requests") or [])
        ]
        manual_requests = [
            ToolboxManualAssignmentRequest.from_runtime_dict(dict(item or {}))
            for item in list(row.get("manual_requests") or [])
        ]
        intrinsics_row = dict(row.get("intrinsics") or {})
        intrinsic_names = self._normalize_intrinsic_tool_names(
            [str(item or "").strip() for item in list(intrinsics_row.get("names") or []) if str(item or "").strip()],
            include_guides=bool(intrinsics_row.get("with_intrinsic_guides", False)),
        )
        intrinsic_profile = (
            SandboxProfileSpec.from_dict(dict(intrinsics_row.get("sandbox_profile") or {}))
            if intrinsic_names
            else None
        )
        with_intrinsic_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
        existing_profiles = dict(row.get("profiles") or {})
        runtime = self._toolbox_runtime_defaults(row)
        old_regs_by_profile: Dict[str, str] = {}
        for reg in self._toolbox_executor_registrations(tid):
            old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

        spawned_engine_ids: List[str] = []
        replaced_engine_ids: List[str] = []
        new_profiles: Dict[str, Dict[str, Any]] = {}
        ready_rollout: Dict[str, Dict[str, Any]] = {}

        if auto_requests or manual_requests or intrinsic_names:
            orchestrator = ToolboxSandboxOrchestrator(
                service=self,
                stager=ToolboxBundleStager(self.hosting_root),
                python_executable=runtime.get("python_executable"),
            )
            assignments = orchestrator.spawn_assignments(
                toolbox_id=tid,
                requests=auto_requests,
                manual_requests=manual_requests,
                intrinsic_tool_names=intrinsic_names,
                intrinsic_profile=intrinsic_profile,
                with_intrinsic_guides=with_intrinsic_guides,
                worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
            )
            try:
                ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
            except Exception:
                for item in assignments:
                    reg = dict(item.registration or {})
                    engine_id = str(reg.get("engine_id") or "").strip()
                    if engine_id:
                        self._retire_toolbox_registration(engine_id)
                self._cleanup_unused_toolbox_environments(state)
                raise
            for item in assignments:
                profile_id = item.sandbox_profile.normalized_profile_id()
                reg = dict(item.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                if engine_id:
                    spawned_engine_ids.append(engine_id)
                old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
                bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
                if old_engine_id and old_engine_id != engine_id:
                    replaced_engine_ids.append(old_engine_id)
                profile_auto_requests = [
                    req.to_runtime_dict()
                    for req in auto_requests
                    if req.sandbox_profile.normalized_profile_id() == profile_id
                ]
                profile_manual_requests = [
                    req.to_runtime_dict()
                    for req in manual_requests
                    if req.sandbox_profile.normalized_profile_id() == profile_id
                ]
                profile_row = {
                    "sandbox_profile": item.sandbox_profile.to_dict(),
                    "requests": profile_auto_requests,
                    "manual_requests": profile_manual_requests,
                    "engine_id": engine_id,
                    "bundle_revision": bundle_revision,
                    "environment": dict(reg.get("environment") or {}),
                    "rollout": dict(ready_rollout.get(engine_id) or {}),
                    "rollout_history": self._append_toolbox_rollout_history(
                        dict(existing_profiles.get(profile_id) or {}),
                        rollout=dict(ready_rollout.get(engine_id) or {}),
                        action=action,
                        engine_id=engine_id,
                        bundle_revision=bundle_revision,
                        replaced_engine_id=old_engine_id,
                    ),
                }
                new_profiles[profile_id] = profile_row
            for profile_id, old_engine_id in old_regs_by_profile.items():
                if profile_id not in new_profiles and old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)
        else:
            for old_engine_id in old_regs_by_profile.values():
                if old_engine_id:
                    replaced_engine_ids.append(old_engine_id)
            for old_engine_id in replaced_engine_ids:
                self._retire_toolbox_registration(old_engine_id)

        updated_row: Dict[str, Any] = {
            "toolbox_id": tid,
            "requests": [req.to_runtime_dict() for req in auto_requests],
            "manual_requests": [req.to_runtime_dict() for req in manual_requests],
            "profiles": new_profiles,
            "runtime": runtime,
        }
        if intrinsic_names:
            updated_row["intrinsics"] = {
                "names": intrinsic_names,
                "sandbox_profile": (intrinsic_profile or SandboxProfileSpec(profile_id="default")).to_dict(),
                "with_intrinsic_guides": with_intrinsic_guides,
            }
        return {
            "toolbox_row": updated_row,
            "profiles": new_profiles,
            "spawned_engine_ids": spawned_engine_ids,
            "ready_engine_ids": list(ready_rollout.keys()),
            "rollout": ready_rollout,
            "replaced_engine_ids": replaced_engine_ids,
            "toolbox_removed": not auto_requests and not manual_requests and not intrinsic_names,
        }

    def toolbox_environment_apply(
        self,
        *,
        environment_name: str,
        toolbox_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        env_name = str(environment_name or "base").strip() or "base"
        _ = self.toolbox_environment_description_get(env_name)
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        selected = {
            str(item or "").strip()
            for item in list(toolbox_ids or [])
            if str(item or "").strip()
        }
        affected_ids: List[str] = []
        for tid, row in toolboxes.items():
            toolbox_id = str(tid or "").strip()
            if selected and toolbox_id not in selected:
                continue
            if self._toolbox_uses_environment_dependency(dict(row or {}), env_name):
                affected_ids.append(toolbox_id)
        affected_ids.sort()
        rebuilt: Dict[str, Any] = {}
        for tid in affected_ids:
            result = self._rebuild_toolbox_from_persisted_state(
                toolbox_id=tid,
                toolbox_row=dict(toolboxes.get(tid) or {}),
                state=state,
                action="apply_environment",
            )
            if bool(result.get("toolbox_removed")):
                toolboxes.pop(tid, None)
            else:
                toolboxes[tid] = dict(result.get("toolbox_row") or {})
            rebuilt[tid] = {
                "profiles": dict(result.get("profiles") or {}),
                "spawned_engine_ids": list(result.get("spawned_engine_ids") or []),
                "ready_engine_ids": list(result.get("ready_engine_ids") or []),
                "rollout": dict(result.get("rollout") or {}),
                "replaced_engine_ids": list(result.get("replaced_engine_ids") or []),
            }
            state["toolboxes"] = toolboxes
            self._write_toolboxes(state)
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        return {
            "status": "ok",
            "environment_name": env_name,
            "affected_toolbox_ids": affected_ids,
            "toolboxes": rebuilt,
            "removed_environment_keys": removed_environment_keys,
        }

    def toolbox_environment_realize(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from ..toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        effective_env = self.toolbox_environment_description_effective_get(env_name)
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        realized_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            required_packages = list(packages.get("required_packages") or [])
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            effective_extra = [
                str(item or "").strip()
                for item in list(effective_env.get("effective_extra_packages") or [])
                if str(item or "").strip()
            ]
            missing_packages = [pkg for pkg in required_packages if pkg not in set(effective_extra)]
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.realize_environment(
                spec,
                environment_description=effective_env,
                required_packages=required_packages,
                missing_packages=missing_packages,
                toolbox_id=tid,
                sandbox_profile_id=str(profile_id or "").strip(),
                tool_keys=matched_tool_keys,
            )
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["realization"] = dict(metadata.get("realization") or {})
            profiles[str(profile_id)] = profile_row
            realized_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": realized_profiles,
        }

    def toolbox_environment_sync_description(
        self,
        *,
        toolbox_id: str,
        source_environment_name: str,
        target_environment_name: Optional[str] = None,
        tool_keys: Optional[List[str]] = None,
        apply: bool = False,
        realize: bool = False,
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        source_name = str(source_environment_name or "base").strip() or "base"
        target_name = str(target_environment_name or "").strip() or source_name
        resolved = self.toolbox_environment_resolve_requirements(
            toolbox_id=tid,
            environment_name=source_name,
            tool_keys=tool_keys,
        )
        missing_packages = [
            str(item or "").strip()
            for item in list(resolved.get("missing_packages") or [])
            if str(item or "").strip()
        ]
        source_desc = self.toolbox_environment_description_get(source_name)
        direct_packages = [
            str(item or "").strip()
            for item in list(source_desc.get("extra_packages") or [])
            if str(item or "").strip()
        ]
        merged_packages: List[str] = []
        seen: set[str] = set()
        for item in direct_packages + missing_packages:
            if item and item not in seen:
                seen.add(item)
                merged_packages.append(item)

        if target_name == source_name:
            env_result = self.toolbox_environment_description_upsert(
                name=source_name,
                base_env_name=str(source_desc.get("base_env_name") or "").strip() or None,
                extra_packages=merged_packages,
                allow_online_install=bool(source_desc.get("allow_online_install", False)),
            )
        else:
            env_result = self.toolbox_environment_description_clone(
                source_name=source_name,
                target_name=target_name,
                extra_packages=merged_packages,
                allow_online_install=bool(source_desc.get("allow_online_install", False)),
            )

        apply_result: Optional[Dict[str, Any]] = None
        realize_result: Optional[Dict[str, Any]] = None
        if apply:
            apply_result = self.toolbox_environment_apply(
                environment_name=target_name,
                toolbox_ids=[tid],
            )
        if realize:
            realize_result = self.toolbox_environment_realize(
                toolbox_id=tid,
                environment_name=target_name,
                tool_keys=tool_keys,
            )
        return {
            "status": "ok",
            "toolbox_id": tid,
            "source_environment_name": source_name,
            "target_environment_name": target_name,
            "resolved": resolved,
            "environment_description": dict(env_result.get("environment_description") or {}),
            "updated_direct_extra_packages": merged_packages,
            "apply_result": apply_result,
            "realize_result": realize_result,
        }

    def toolbox_environment_prepare_install(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from ..toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        effective_env = self.toolbox_environment_description_effective_get(env_name)
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        planned_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            required_packages = list(packages.get("required_packages") or [])
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            effective_extra = [
                str(item or "").strip()
                for item in list(effective_env.get("effective_extra_packages") or [])
                if str(item or "").strip()
            ]
            missing_packages = [pkg for pkg in required_packages if pkg not in set(effective_extra)]
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.prepare_install_plan(
                spec,
                environment_description=effective_env,
                required_packages=required_packages,
                missing_packages=missing_packages,
                toolbox_id=tid,
                sandbox_profile_id=str(profile_id or "").strip(),
                tool_keys=matched_tool_keys,
            )
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["realization"] = dict(metadata.get("realization") or {})
            profile_row["environment"]["install_plan"] = dict(metadata.get("install_plan") or {})
            profiles[str(profile_id)] = profile_row
            planned_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": planned_profiles,
        }

    def toolbox_environment_lock_install(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from ..toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        locked_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.lock_install_plan(spec)
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["install_plan"] = dict(metadata.get("install_plan") or {})
            profile_row["environment"]["install_lock"] = dict(metadata.get("install_lock") or {})
            profiles[str(profile_id)] = profile_row
            locked_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": locked_profiles,
        }

    def toolbox_environment_resolve_install_lock(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
        allow_resolution: bool = False,
    ) -> Dict[str, Any]:
        from ..toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        resolved_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.resolve_install_lock(
                spec,
                allow_resolution=bool(allow_resolution),
            )
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["install_plan"] = dict(metadata.get("install_plan") or {})
            profile_row["environment"]["install_resolution"] = dict(metadata.get("install_resolution") or {})
            profile_row["environment"]["resolved_install_lock"] = dict(metadata.get("resolved_install_lock") or {})
            profiles[str(profile_id)] = profile_row
            resolved_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": resolved_profiles,
        }

    def toolbox_environment_verify_install_lock(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from ..toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        verified_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.verify_install_lock(spec)
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["install_plan"] = dict(metadata.get("install_plan") or {})
            profile_row["environment"]["install_lock"] = dict(metadata.get("install_lock") or {})
            profile_row["environment"]["resolved_install_lock"] = dict(metadata.get("resolved_install_lock") or {})
            profile_row["environment"]["install_lock_verification"] = dict(metadata.get("install_lock_verification") or {})
            profiles[str(profile_id)] = profile_row
            verified_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": verified_profiles,
        }

    def toolbox_environment_verify_install_receipt(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from ..toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        verified_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.verify_install_receipt(spec)
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["install_lock"] = dict(metadata.get("install_lock") or {})
            profile_row["environment"]["resolved_install_lock"] = dict(metadata.get("resolved_install_lock") or {})
            profile_row["environment"]["install_lock_verification"] = dict(metadata.get("install_lock_verification") or {})
            profile_row["environment"]["install_receipt"] = dict(metadata.get("install_receipt") or {})
            profile_row["environment"]["install_receipt_verification"] = dict(metadata.get("install_receipt_verification") or {})
            profiles[str(profile_id)] = profile_row
            verified_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": verified_profiles,
        }

    def toolbox_environment_execute_install(
        self,
        *,
        toolbox_id: str,
        environment_name: str,
        tool_keys: Optional[List[str]] = None,
        allow_execution: bool = False,
    ) -> Dict[str, Any]:
        from ..toolbox_harness import ToolboxEnvironmentManager, ToolboxEnvironmentSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        env_name = str(environment_name or "base").strip() or "base"
        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        toolbox_row = dict(toolboxes.get(tid) or {})
        if not toolbox_row:
            raise ValueError(f"toolbox '{tid}' not found")
        manager = ToolboxEnvironmentManager(self.hosting_root)
        selected = [str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()]
        executed_profiles: Dict[str, Dict[str, Any]] = {}
        profiles = dict(toolbox_row.get("profiles") or {})
        for profile_id, raw_profile in profiles.items():
            profile_row = dict(raw_profile or {})
            sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
            if str(sandbox_profile.get("environment_name") or "base").strip() != env_name:
                continue
            packages = self._toolbox_profile_required_packages(profile_row, tool_keys=selected or None)
            matched_tool_keys = list(packages.get("tool_keys") or [])
            if selected and not matched_tool_keys:
                continue
            environment = dict(profile_row.get("environment") or {})
            spec = ToolboxEnvironmentSpec.from_dict(environment)
            metadata = manager.execute_install_plan(
                spec,
                allow_execution=bool(allow_execution),
            )
            profile_row["environment"] = dict(environment)
            profile_row["environment"]["install_plan"] = dict(metadata.get("install_plan") or {})
            profile_row["environment"]["install_lock"] = dict(metadata.get("install_lock") or {})
            profile_row["environment"]["install_resolution"] = dict(metadata.get("install_resolution") or {})
            profile_row["environment"]["resolved_install_lock"] = dict(metadata.get("resolved_install_lock") or {})
            profile_row["environment"]["install_lock_verification"] = dict(metadata.get("install_lock_verification") or {})
            profile_row["environment"]["install_execution"] = dict(metadata.get("install_execution") or {})
            profile_row["environment"]["install_receipt"] = dict(metadata.get("install_receipt") or {})
            profile_row["environment"]["install_receipt_verification"] = dict(metadata.get("install_receipt_verification") or {})
            profiles[str(profile_id)] = profile_row
            executed_profiles[str(profile_id)] = {
                "environment": dict(profile_row.get("environment") or {}),
                "tool_keys": matched_tool_keys,
            }
        toolbox_row["profiles"] = profiles
        toolboxes[tid] = toolbox_row
        state["toolboxes"] = toolboxes
        self._write_toolboxes(state)
        return {
            "status": "ok",
            "toolbox_id": tid,
            "environment_name": env_name,
            "profiles": executed_profiles,
        }

    def _cleanup_unused_toolbox_environments(self, state: Optional[Dict[str, Any]] = None) -> List[str]:
        payload = dict(state or self._read_toolboxes() or {})
        toolboxes = dict(payload.get("toolboxes") or {})
        referenced_keys: set[str] = set()
        referenced_paths: set[str] = set()
        env_roots = [
            (self.hosting_root / "toolbox_venvs").resolve(),
            (self.hosting_root / "runtime_envs").resolve(),
        ]
        for toolbox_row in toolboxes.values():
            profiles = dict(dict(toolbox_row or {}).get("profiles") or {})
            for profile_row in profiles.values():
                environment = dict(dict(profile_row or {}).get("environment") or {})
                venv_key = str(environment.get("venv_key") or "").strip()
                if venv_key:
                    referenced_keys.add(venv_key)
                raw_path = str(environment.get("venv_path") or "").strip()
                if not raw_path:
                    continue
                try:
                    resolved = Path(raw_path).expanduser().resolve()
                except Exception:
                    continue
                for env_root in env_roots:
                    try:
                        if resolved == env_root or env_root in resolved.parents:
                            referenced_paths.add(str(resolved))
                    except Exception:
                        continue
        for reg in self._read_engines():
            environment = dict(dict(reg or {}).get("environment") or {})
            venv_key = str(environment.get("venv_key") or "").strip()
            if venv_key:
                referenced_keys.add(venv_key)
            raw_path = str(environment.get("venv_path") or "").strip()
            if not raw_path:
                continue
            try:
                resolved = Path(raw_path).expanduser().resolve()
            except Exception:
                continue
            for env_root in env_roots:
                try:
                    if resolved == env_root or env_root in resolved.parents:
                        referenced_paths.add(str(resolved))
                except Exception:
                    continue
        removed: List[str] = []
        for env_root in env_roots:
            if not env_root.exists():
                continue
            for child in env_root.iterdir():
                if not child.is_dir():
                    continue
                try:
                    resolved_child = str(child.expanduser().resolve())
                except Exception:
                    resolved_child = ""
                if child.name in referenced_keys or (resolved_child and resolved_child in referenced_paths):
                    continue
                shutil.rmtree(child, ignore_errors=True)
                removed.append(child.name)
        return removed

    def _toolbox_reference_report(
        self,
        state: Optional[Dict[str, Any]] = None,
        *,
        include_reachability: bool = False,
        reachability_timeout_seconds: float = 0.35,
    ) -> Dict[str, Any]:
        payload = dict(state or self._read_toolboxes() or {})
        toolboxes = dict(payload.get("toolboxes") or {})
        referenced_engine_ids = self._referenced_toolbox_engine_ids(payload)
        referenced_env_keys: set[str] = set()
        referenced_env_roots: set[str] = set()
        referenced_env_key_reasons: Dict[str, List[Dict[str, Any]]] = {}
        referenced_env_root_reasons: Dict[str, List[Dict[str, Any]]] = {}
        env_roots = [
            (self.hosting_root / "toolbox_venvs").resolve(),
            (self.hosting_root / "runtime_envs").resolve(),
        ]
        profiles_by_toolbox: Dict[str, Any] = {}
        for toolbox_id, raw_toolbox in toolboxes.items():
            toolbox_row = dict(raw_toolbox or {})
            profiles = dict(toolbox_row.get("profiles") or {})
            out_profiles: Dict[str, Any] = {}
            for profile_id, raw_profile in profiles.items():
                profile_row = dict(raw_profile or {})
                environment = dict(profile_row.get("environment") or {})
                venv_key = str(environment.get("venv_key") or "").strip()
                if venv_key:
                    referenced_env_keys.add(venv_key)
                    referenced_env_key_reasons.setdefault(venv_key, []).append(
                        {
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "kind": "profile_environment",
                        }
                    )
                raw_env_path = str(environment.get("venv_path") or "").strip()
                if raw_env_path:
                    try:
                        resolved_env_path = Path(raw_env_path).expanduser().resolve()
                    except Exception:
                        resolved_env_path = None
                    if resolved_env_path is not None:
                        for env_root in env_roots:
                            try:
                                if not (resolved_env_path == env_root or env_root in resolved_env_path.parents):
                                    continue
                                env_root_value = str(resolved_env_path)
                                referenced_env_roots.add(env_root_value)
                                referenced_env_root_reasons.setdefault(env_root_value, []).append(
                                    {
                                        "toolbox_id": str(toolbox_id),
                                        "sandbox_profile_id": str(profile_id),
                                        "kind": "profile_environment",
                                        "venv_key": venv_key or None,
                                    }
                                )
                            except Exception:
                                pass
                out_profiles[str(profile_id)] = {
                    "engine_id": str(profile_row.get("engine_id") or "").strip() or None,
                    "bundle_revision": str(profile_row.get("bundle_revision") or "").strip() or None,
                    "environment": environment,
                    "sandbox_profile": dict(profile_row.get("sandbox_profile") or {}),
                    "request_count": len(list(profile_row.get("requests") or [])),
                    "rollout": dict(profile_row.get("rollout") or {}),
                }
            profiles_by_toolbox[str(toolbox_id)] = {
                "profiles": out_profiles,
                "runtime": dict(toolbox_row.get("runtime") or {}),
            }

        live_toolbox_regs: Dict[str, Dict[str, Any]] = {}
        stale_engine_ids: List[str] = []
        referenced_bundle_roots: set[str] = set()
        referenced_bundle_root_reasons: Dict[str, List[Dict[str, Any]]] = {}
        for reg in self._read_engines():
            row = dict(reg or {})
            if str(row.get("executor_kind") or "").strip() != "toolbox_executor":
                continue
            engine_id = str(row.get("engine_id") or "").strip()
            bundle = dict(row.get("bundle") or {})
            bundle_root = str(bundle.get("bundle_root") or "").strip()
            is_referenced = engine_id in referenced_engine_ids
            if bundle_root and is_referenced:
                try:
                    resolved_bundle_root = str(Path(bundle_root).expanduser().resolve())
                    referenced_bundle_roots.add(resolved_bundle_root)
                    referenced_bundle_root_reasons.setdefault(resolved_bundle_root, []).append(
                        {
                            "engine_id": engine_id,
                            "toolbox_id": self._registration_toolbox_id(row) or None,
                            "sandbox_profile_id": self._registration_sandbox_profile_id(row) or None,
                            "kind": "live_registration",
                        }
                    )
                except Exception:
                    pass
            live_toolbox_regs[engine_id] = {
                "toolbox_id": self._registration_toolbox_id(row) or None,
                "sandbox_profile_id": self._registration_sandbox_profile_id(row),
                "bundle_root": bundle_root or None,
                "environment": dict(row.get("environment") or {}),
                "allowed_tool_names": sorted(list(self._registration_allowed_tool_names(row) or set())),
                "referenced": is_referenced,
            }
            if include_reachability and engine_id:
                reachability = self._probe_registration_reachability(
                    row,
                    timeout_seconds=reachability_timeout_seconds,
                )
                live_toolbox_regs[engine_id]["reachable"] = bool(reachability.get("reachable", False))
                live_toolbox_regs[engine_id]["reachability"] = dict(reachability or {})
            if engine_id and not is_referenced:
                stale_engine_ids.append(engine_id)

        bundles_root = (self.hosting_root / "toolbox_bundles").resolve()
        stale_bundle_roots: List[str] = []
        if bundles_root.exists():
            for child in bundles_root.iterdir():
                if not child.is_dir():
                    continue
                if not self._bundle_directory_is_referenced(
                    child,
                    referenced_bundle_roots=referenced_bundle_roots,
                ):
                    stale_bundle_roots.append(child.name)
        stale_environment_keys: List[str] = []
        for env_root in env_roots:
            if not env_root.exists():
                continue
            for child in env_root.iterdir():
                if not child.is_dir():
                    continue
                try:
                    resolved_child = str(child.expanduser().resolve())
                except Exception:
                    resolved_child = ""
                if child.name not in referenced_env_keys and resolved_child not in referenced_env_roots:
                    stale_environment_keys.append(child.name)

        return {
            "status": "ok",
            "toolboxes": profiles_by_toolbox,
            "live_registrations": live_toolbox_regs,
            "referenced_engine_ids": sorted(referenced_engine_ids),
            "referenced_environment_keys": sorted(referenced_env_keys),
            "referenced_environment_roots": sorted(referenced_env_roots),
            "referenced_environment_key_reasons": {
                str(key): list(rows)
                for key, rows in sorted(referenced_env_key_reasons.items())
            },
            "referenced_environment_root_reasons": {
                str(key): list(rows)
                for key, rows in sorted(referenced_env_root_reasons.items())
            },
            "referenced_bundle_root_reasons": {
                str(key): list(rows)
                for key, rows in sorted(referenced_bundle_root_reasons.items())
            },
            "stale_engine_ids": sorted(stale_engine_ids),
            "stale_bundle_roots": sorted(stale_bundle_roots),
            "stale_environment_keys": sorted(stale_environment_keys),
            "summary": {
                "toolbox_count": len(profiles_by_toolbox),
                "live_registration_count": len(live_toolbox_regs),
                "referenced_engine_count": len(referenced_engine_ids),
                "stale_engine_count": len(stale_engine_ids),
                "stale_bundle_count": len(stale_bundle_roots),
                "stale_environment_count": len(stale_environment_keys),
            },
        }

    @staticmethod
    def _bundle_directory_is_referenced(
        directory: Path,
        *,
        referenced_bundle_roots: set[str],
    ) -> bool:
        try:
            base = directory.expanduser().resolve()
        except Exception:
            return False
        for raw in set(referenced_bundle_roots or set()):
            try:
                ref = Path(raw).expanduser().resolve()
            except Exception:
                continue
            if ref == base:
                return True
            try:
                ref.relative_to(base)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _toolbox_profile_expected_tool_names(
        toolbox_row: Optional[Dict[str, Any]],
        profile_id: str,
    ) -> List[str]:
        from ..toolbox_harness import (
            ToolboxAutoAssignmentRequest,
            ToolboxManualAssignmentRequest,
        )

        toolbox_payload = dict(toolbox_row or {})
        profile_key = str(profile_id or "").strip()
        names: set[str] = set()

        for raw_request in list(toolbox_payload.get("requests") or []):
            try:
                req = ToolboxAutoAssignmentRequest.from_runtime_dict(dict(raw_request or {}))
            except Exception:
                continue
            if req.sandbox_profile.normalized_profile_id() != profile_key:
                continue
            tool_name = str(req.callable_name or "").strip()
            if tool_name:
                names.add(tool_name)

        for raw_request in list(toolbox_payload.get("manual_requests") or []):
            try:
                req = ToolboxManualAssignmentRequest.from_runtime_dict(dict(raw_request or {}))
            except Exception:
                continue
            if req.sandbox_profile.normalized_profile_id() != profile_key:
                continue
            fn = dict(dict(req.tool_definition or {}).get("function") or {})
            tool_name = str(fn.get("name") or "").strip()
            if tool_name:
                names.add(tool_name)

        intrinsics_row = dict(toolbox_payload.get("intrinsics") or {})
        if intrinsics_row:
            intrinsic_profile = dict(intrinsics_row.get("sandbox_profile") or {})
            intrinsic_profile_id = str(intrinsic_profile.get("profile_id") or "").strip() or "default"
            if intrinsic_profile_id == profile_key:
                include_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
                intrinsic_names = [
                    str(item or "").strip()
                    for item in list(intrinsics_row.get("names") or [])
                    if str(item or "").strip()
                ]
                names.update(
                    self._normalize_intrinsic_tool_names(
                        intrinsic_names,
                        include_guides=include_guides,
                    )
                )
        return sorted(names)

    def toolbox_consistency(self) -> Dict[str, Any]:
        state = self._read_toolboxes()
        refs = self._toolbox_reference_report(
            state,
            include_reachability=True,
        )
        toolboxes = dict(state.get("toolboxes") or {})
        live_regs = {
            str(engine_id or "").strip(): dict(row or {})
            for engine_id, row in dict(refs.get("live_registrations") or {}).items()
            if str(engine_id or "").strip()
        }
        referenced_engine_ids = {
            str(item or "").strip()
            for item in list(refs.get("referenced_engine_ids") or [])
            if str(item or "").strip()
        }
        issues: List[Dict[str, Any]] = []

        for toolbox_id, raw_toolbox in toolboxes.items():
            toolbox_row = dict(raw_toolbox or {})
            profiles = dict(toolbox_row.get("profiles") or {})
            for profile_id, raw_profile in profiles.items():
                profile_row = dict(raw_profile or {})
                engine_id = str(profile_row.get("engine_id") or "").strip()
                sandbox_profile = dict(profile_row.get("sandbox_profile") or {})
                expected_profile_id = str(sandbox_profile.get("profile_id") or "").strip() or str(profile_id or "").strip()
                environment = dict(profile_row.get("environment") or {})
                venv_key = str(environment.get("venv_key") or "").strip()
                venv_path = str(environment.get("venv_path") or "").strip()
                expected_tool_names = self._toolbox_profile_expected_tool_names(toolbox_row, str(profile_id or ""))

                if not engine_id:
                    issues.append(
                        {
                            "issue": "missing_profile_engine_id",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                        }
                    )
                    continue
                if engine_id not in referenced_engine_ids:
                    issues.append(
                        {
                            "issue": "unreferenced_profile_engine_id",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                        }
                    )
                live = dict(live_regs.get(engine_id) or {})
                if not live:
                    issues.append(
                        {
                            "issue": "missing_live_registration",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                        }
                    )
                    continue
                if live.get("reachable") is False:
                    reachability = dict(live.get("reachability") or {})
                    issues.append(
                        {
                            "issue": "live_registration_unreachable",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                            "probe": str(reachability.get("probe") or "").strip() or None,
                            "reachability_error": str(reachability.get("error") or "").strip() or None,
                        }
                    )
                live_toolbox_id = str(live.get("toolbox_id") or "").strip()
                if live_toolbox_id and live_toolbox_id != str(toolbox_id):
                    issues.append(
                        {
                            "issue": "registration_toolbox_id_mismatch",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                            "expected_toolbox_id": str(toolbox_id),
                            "actual_toolbox_id": live_toolbox_id,
                        }
                    )
                live_profile_id = str(live.get("sandbox_profile_id") or "").strip()
                if live_profile_id and live_profile_id != expected_profile_id:
                    issues.append(
                        {
                            "issue": "registration_profile_id_mismatch",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                            "expected_sandbox_profile_id": expected_profile_id,
                            "actual_sandbox_profile_id": live_profile_id,
                        }
                    )
                live_allowed = sorted(
                    [
                        str(item or "").strip()
                        for item in list(live.get("allowed_tool_names") or [])
                        if str(item or "").strip()
                    ]
                )
                if live_allowed != expected_tool_names:
                    issues.append(
                        {
                            "issue": "registration_allowed_tool_names_mismatch",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                            "expected_tool_names": expected_tool_names,
                            "actual_tool_names": live_allowed,
                        }
                    )

                if venv_key and not venv_path:
                    issues.append(
                        {
                            "issue": "environment_path_missing",
                            "toolbox_id": str(toolbox_id),
                            "sandbox_profile_id": str(profile_id),
                            "engine_id": engine_id,
                            "venv_key": venv_key,
                        }
                    )
                elif venv_key and venv_path:
                    try:
                        env_path = Path(venv_path).expanduser().resolve()
                    except Exception:
                        env_path = None
                    if env_path is None or not env_path.exists():
                        issues.append(
                            {
                                "issue": "environment_path_missing_on_disk",
                                "toolbox_id": str(toolbox_id),
                                "sandbox_profile_id": str(profile_id),
                                "engine_id": engine_id,
                                "venv_key": venv_key,
                                "venv_path": venv_path,
                            }
                        )
                    else:
                        metadata_path = env_path / "environment.json"
                        if not metadata_path.exists():
                            issues.append(
                                {
                                    "issue": "environment_metadata_missing",
                                    "toolbox_id": str(toolbox_id),
                                    "sandbox_profile_id": str(profile_id),
                                    "engine_id": engine_id,
                                    "venv_key": venv_key,
                                    "venv_path": str(env_path),
                                }
                            )

        return {
            "status": "ok",
            "issue_count": len(issues),
            "issues": issues,
            "references": refs,
            "summary": {
                "issue_count": len(issues),
                "toolbox_count": len(toolboxes),
                "referenced_engine_count": len(referenced_engine_ids),
            },
        }

    def toolbox_review_snapshot(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        scoped_ids = {
            str(item or "").strip()
            for item in list(toolbox_ids or [])
            if str(item or "").strip()
        }
        references = self.toolbox_references()
        consistency = self.toolbox_consistency()
        consistency_references = dict(consistency.get("references") or {})
        if consistency_references:
            references = {
                **references,
                "live_registrations": dict(consistency_references.get("live_registrations") or dict(references.get("live_registrations") or {})),
                "summary": {
                    **dict(references.get("summary") or {}),
                    **dict(consistency_references.get("summary") or {}),
                },
            }
        if scoped_ids:
            filtered_toolboxes = {
                str(k): dict(v or {})
                for k, v in dict(references.get("toolboxes") or {}).items()
                if str(k or "").strip() in scoped_ids
            }
            filtered_issues = [
                dict(item or {})
                for item in list(consistency.get("issues") or [])
                if str(dict(item or {}).get("toolbox_id") or "").strip() in scoped_ids
            ]
            references = {
                **references,
                "toolboxes": filtered_toolboxes,
                "summary": {
                    **dict(references.get("summary") or {}),
                    "toolbox_count": len(filtered_toolboxes),
                },
            }
            consistency = {
                **consistency,
                "issue_count": len(filtered_issues),
                "issues": filtered_issues,
                "summary": {
                    **dict(consistency.get("summary") or {}),
                    "issue_count": len(filtered_issues),
                    "toolbox_count": len(filtered_toolboxes),
                },
            }
        issues = [dict(item or {}) for item in list(consistency.get("issues") or [])]
        issues_by_toolbox: Dict[str, List[Dict[str, Any]]] = {}
        for item in issues:
            toolbox_id = str(item.get("toolbox_id") or "").strip()
            issues_by_toolbox.setdefault(toolbox_id, []).append(item)

        toolbox_rows: Dict[str, Dict[str, Any]] = {}
        live_regs = dict(references.get("live_registrations") or {})
        for toolbox_id, raw_toolbox in dict(references.get("toolboxes") or {}).items():
            toolbox_row = dict(raw_toolbox or {})
            profile_rows: List[Dict[str, Any]] = []
            for profile_id, raw_profile in dict(toolbox_row.get("profiles") or {}).items():
                profile_row = dict(raw_profile or {})
                rollout = dict(profile_row.get("rollout") or {})
                environment = dict(profile_row.get("environment") or {})
                engine_id = str(profile_row.get("engine_id") or "").strip()
                live_reg = dict(live_regs.get(engine_id) or {})
                tool_names = [
                    str(item or "").strip()
                    for item in list(
                        rollout.get("all_registered_tool_names")
                        or live_reg.get("all_registered_tool_names")
                        or live_reg.get("allowed_tool_names")
                        or []
                    )
                    if str(item or "").strip()
                ]
                advertised_tool_names = [
                    str(item or "").strip()
                    for item in list(
                        rollout.get("advertised_tool_names")
                        or live_reg.get("advertised_tool_names")
                        or []
                    )
                    if str(item or "").strip()
                ]
                hidden_allowed_tool_names = [
                    str(item or "").strip()
                    for item in list(
                        rollout.get("hidden_allowed_tool_names")
                        or live_reg.get("hidden_allowed_tool_names")
                        or []
                    )
                    if str(item or "").strip()
                ]
                profile_rows.append(
                    {
                        "sandbox_profile_id": str(dict(profile_row.get("sandbox_profile") or {}).get("profile_id") or profile_id or "").strip(),
                        "environment_name": str(environment.get("environment_name") or "").strip() or None,
                        "all_registered_tool_names": sorted(tool_names),
                        "advertised_tool_names": sorted(advertised_tool_names),
                        "hidden_allowed_tool_names": sorted(hidden_allowed_tool_names),
                        "engine_id": engine_id or None,
                        "reachable": live_reg.get("reachable"),
                        "ready": bool(rollout.get("ready", False)),
                        "warmup_ms": int(rollout.get("warmup_ms") or 0),
                    }
                )
            profile_rows.sort(key=lambda item: (str(item.get("environment_name") or ""), str(item.get("sandbox_profile_id") or "")))
            toolbox_issues = list(issues_by_toolbox.get(str(toolbox_id or "").strip()) or [])
            toolbox_rows[str(toolbox_id)] = {
                "profile_count": len(profile_rows),
                "profiles": profile_rows,
                "issue_count": len(toolbox_issues),
                "issue_names": sorted(
                    {
                        str(item.get("issue") or "").strip()
                        for item in toolbox_issues
                        if str(item.get("issue") or "").strip()
                    }
                ),
                "live_registration_count": sum(
                    1 for item in profile_rows if str(item.get("engine_id") or "").strip()
                ),
            }

        issue_count = len(issues)
        return {
            "status": "ok",
            "toolbox_ids": sorted(scoped_ids),
            "toolboxes": toolbox_rows,
            "issues": issues,
            "recommended_action": "reconcile" if issue_count > 0 else "observe",
            "references_summary": dict(references.get("summary") or {}),
            "consistency_summary": {
                **dict(consistency.get("summary") or {}),
                "issue_count": issue_count,
            },
            "summary": {
                "toolbox_count": int(dict(dict(references or {}).get("summary") or {}).get("toolbox_count") or 0),
                "issue_count": issue_count,
                "stale_engine_count": int(dict(dict(references or {}).get("summary") or {}).get("stale_engine_count") or 0),
                "stale_bundle_count": int(dict(dict(references or {}).get("summary") or {}).get("stale_bundle_count") or 0),
                "stale_environment_count": int(dict(dict(references or {}).get("summary") or {}).get("stale_environment_count") or 0),
            },
        }

    def toolbox_repair(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
    ) -> Dict[str, Any]:
        from ..toolbox_harness import (
            SandboxProfileSpec,
            ToolboxAutoAssignmentRequest,
            ToolboxBundleStager,
            ToolboxManualAssignmentRequest,
            ToolboxSandboxOrchestrator,
        )

        state = self._read_toolboxes()
        toolboxes = dict(state.get("toolboxes") or {})
        consistency = self.toolbox_consistency()
        issues = list(consistency.get("issues") or [])
        inconsistent_toolbox_ids = sorted(
            {
                str(item.get("toolbox_id") or "").strip()
                for item in issues
                if str(item.get("toolbox_id") or "").strip()
            }
        )
        requested_toolbox_ids = [
            str(item or "").strip()
            for item in list(toolbox_ids or [])
            if str(item or "").strip()
        ]
        if requested_toolbox_ids:
            if only_inconsistent:
                inconsistent_set = set(inconsistent_toolbox_ids)
                target_toolbox_ids = [
                    item for item in requested_toolbox_ids
                    if item in inconsistent_set
                ]
            else:
                target_toolbox_ids = requested_toolbox_ids
        elif only_inconsistent:
            target_toolbox_ids = inconsistent_toolbox_ids
        else:
            target_toolbox_ids = sorted(str(item or "").strip() for item in toolboxes.keys() if str(item or "").strip())

        repaired: Dict[str, Any] = {}
        skipped: Dict[str, str] = {}

        with self._locked_toolboxes(target_toolbox_ids):
            state = self._read_toolboxes()
            toolboxes = dict(state.get("toolboxes") or {})

            for tid in target_toolbox_ids:
                toolbox_row_existing = dict(toolboxes.get(tid) or {})
                if not toolbox_row_existing:
                    skipped[tid] = "toolbox_not_found"
                    continue
                runtime = dict(toolbox_row_existing.get("runtime") or {})
                auto_requests = [
                    ToolboxAutoAssignmentRequest.from_runtime_dict(dict(item or {}))
                    for item in list(toolbox_row_existing.get("requests") or [])
                ]
                manual_requests = [
                    ToolboxManualAssignmentRequest.from_runtime_dict(dict(item or {}))
                    for item in list(toolbox_row_existing.get("manual_requests") or [])
                ]
                intrinsics_row = dict(toolbox_row_existing.get("intrinsics") or {})
                intrinsic_names = self._normalize_intrinsic_tool_names(
                    [str(item or "").strip() for item in list(intrinsics_row.get("names") or []) if str(item or "").strip()],
                    include_guides=bool(intrinsics_row.get("with_intrinsic_guides", False)),
                )
                intrinsic_profile = (
                    SandboxProfileSpec.from_dict(dict(intrinsics_row.get("sandbox_profile") or {}))
                    if intrinsic_names
                    else None
                )
                with_intrinsic_guides = bool(intrinsics_row.get("with_intrinsic_guides", False))
                existing_profiles = dict(toolbox_row_existing.get("profiles") or {})
                old_regs_by_profile: Dict[str, str] = {}
                for reg in self._toolbox_executor_registrations(tid):
                    old_regs_by_profile[self._registration_sandbox_profile_id(reg)] = str(reg.get("engine_id") or "").strip()

                if not auto_requests and not manual_requests and not intrinsic_names:
                    skipped[tid] = "toolbox_has_no_persisted_requests"
                    continue

                orchestrator = ToolboxSandboxOrchestrator(
                    service=self,
                    stager=ToolboxBundleStager(self.hosting_root),
                    python_executable=str(runtime.get("python_executable") or "").strip() or None,
                )
                assignments = orchestrator.spawn_assignments(
                    toolbox_id=tid,
                    requests=auto_requests,
                    manual_requests=manual_requests,
                    intrinsic_tool_names=intrinsic_names,
                    intrinsic_profile=intrinsic_profile,
                    with_intrinsic_guides=with_intrinsic_guides,
                    worker_profile_class=str(runtime.get("worker_profile_class") or "generic"),
                )
                try:
                    ready_rollout = self._ensure_toolbox_assignments_ready(assignments, timeout_seconds=8.0)
                except Exception:
                    for item in assignments:
                        reg = dict(item.registration or {})
                        engine_id = str(reg.get("engine_id") or "").strip()
                        if engine_id:
                            self._retire_toolbox_registration(engine_id)
                    self._cleanup_unused_toolbox_environments(state)
                    raise

                new_profiles: Dict[str, Dict[str, Any]] = {}
                spawned_engine_ids: List[str] = []
                replaced_engine_ids: List[str] = []
                for item in assignments:
                    profile_id = item.sandbox_profile.normalized_profile_id()
                    reg = dict(item.registration or {})
                    engine_id = str(reg.get("engine_id") or "").strip()
                    if engine_id:
                        spawned_engine_ids.append(engine_id)
                    old_engine_id = str(old_regs_by_profile.get(profile_id) or "").strip()
                    bundle_revision = str(dict(reg.get("bundle") or {}).get("bundle_revision") or "")
                    if old_engine_id and old_engine_id != engine_id:
                        replaced_engine_ids.append(old_engine_id)
                    profile_auto_requests = [
                        req.to_runtime_dict()
                        for req in auto_requests
                        if req.sandbox_profile.normalized_profile_id() == profile_id
                    ]
                    profile_manual_requests = [
                        req.to_runtime_dict()
                        for req in manual_requests
                        if req.sandbox_profile.normalized_profile_id() == profile_id
                    ]
                    new_profiles[profile_id] = {
                        "sandbox_profile": item.sandbox_profile.to_dict(),
                        "requests": profile_auto_requests,
                        "manual_requests": profile_manual_requests,
                        "engine_id": engine_id,
                        "bundle_revision": bundle_revision,
                        "environment": dict(reg.get("environment") or {}),
                        "rollout": dict(ready_rollout.get(engine_id) or {}),
                        "rollout_history": self._append_toolbox_rollout_history(
                            dict(existing_profiles.get(profile_id) or {}),
                            rollout=dict(ready_rollout.get(engine_id) or {}),
                            action="repair",
                            engine_id=engine_id,
                            bundle_revision=bundle_revision,
                            replaced_engine_id=old_engine_id,
                        ),
                    }
                for profile_id, old_engine_id in old_regs_by_profile.items():
                    if profile_id not in new_profiles and old_engine_id:
                        replaced_engine_ids.append(old_engine_id)
                for old_engine_id in replaced_engine_ids:
                    self._retire_toolbox_registration(old_engine_id)

                toolboxes[tid] = {
                    "toolbox_id": tid,
                    "requests": [req.to_runtime_dict() for req in auto_requests],
                    "manual_requests": [req.to_runtime_dict() for req in manual_requests],
                    "profiles": new_profiles,
                    "runtime": runtime,
                    **(
                        {
                            "intrinsics": {
                                "names": intrinsic_names,
                                "sandbox_profile": (intrinsic_profile or SandboxProfileSpec(profile_id="default")).to_dict(),
                                "with_intrinsic_guides": with_intrinsic_guides,
                            }
                        }
                        if intrinsic_names
                        else {}
                    ),
                }
                repaired[tid] = {
                    "profiles": new_profiles,
                    "spawned_engine_ids": spawned_engine_ids,
                    "ready_engine_ids": list(ready_rollout.keys()),
                    "rollout": ready_rollout,
                    "replaced_engine_ids": replaced_engine_ids,
                }

            state["toolboxes"] = toolboxes
            self._write_toolboxes(state)
            removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        result = {
            "status": "ok",
            "requested_toolbox_ids": requested_toolbox_ids,
            "target_toolbox_ids": target_toolbox_ids,
            "inconsistent_toolbox_ids": inconsistent_toolbox_ids,
            "repaired_toolbox_ids": sorted(
                str(item or "").strip()
                for item in repaired.keys()
                if str(item or "").strip()
            ),
            "skipped_toolbox_ids": sorted(
                str(item or "").strip()
                for item in skipped.keys()
                if str(item or "").strip()
            ),
            "removed_environment_keys": removed_environment_keys,
            "outcome": "repaired" if repaired else "noop",
            "summary": {
                "requested_toolbox_count": len(requested_toolbox_ids),
                "target_toolbox_count": len(target_toolbox_ids),
                "repaired_toolbox_count": len(repaired),
                "skipped_toolbox_count": len(skipped),
                "removed_environment_count": len(removed_environment_keys),
            },
        }
        if details:
            result["repaired"] = repaired
            result["skipped"] = skipped
        return result

    def toolbox_reconcile(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
    ) -> Dict[str, Any]:
        requested_toolbox_ids = [
            str(item or "").strip()
            for item in list(toolbox_ids or [])
            if str(item or "").strip()
        ]
        before = self.toolbox_consistency()
        repair = self.toolbox_repair(
            toolbox_ids=toolbox_ids,
            only_inconsistent=only_inconsistent,
        )
        gc_out = self.toolbox_gc()
        after = self.toolbox_consistency()
        repair_requested_toolbox_ids = list(dict(repair or {}).get("requested_toolbox_ids") or requested_toolbox_ids)
        repair_inconsistent_toolbox_ids = list(dict(repair or {}).get("inconsistent_toolbox_ids") or [])
        repaired_toolbox_ids = sorted(
            str(item or "").strip()
            for item in dict(dict(repair or {}).get("repaired") or {}).keys()
            if str(item or "").strip()
        )
        repair_target_toolbox_ids = list(dict(repair or {}).get("target_toolbox_ids") or repaired_toolbox_ids)
        skipped_toolbox_ids = sorted(
            str(item or "").strip()
            for item in dict(dict(repair or {}).get("skipped") or {}).keys()
            if str(item or "").strip()
        )
        result = {
            "status": "ok",
            "toolbox_ids": requested_toolbox_ids,
            "requested_toolbox_ids": repair_requested_toolbox_ids,
            "target_toolbox_ids": repair_target_toolbox_ids,
            "inconsistent_toolbox_ids": repair_inconsistent_toolbox_ids,
            "repaired_toolbox_ids": repaired_toolbox_ids,
            "skipped_toolbox_ids": skipped_toolbox_ids,
            "removed_engine_ids": list(dict(gc_out or {}).get("removed_engine_ids") or []),
            "removed_bundle_roots": list(dict(gc_out or {}).get("removed_bundle_roots") or []),
            "removed_environment_keys": list(dict(gc_out or {}).get("removed_environment_keys") or []),
            "outcome": (
                "repaired"
                if repaired_toolbox_ids
                else "noop"
            ),
            "summary": {
                "before_issue_count": int(dict(before or {}).get("issue_count") or 0),
                "after_issue_count": int(dict(after or {}).get("issue_count") or 0),
                "removed_engine_count": len(list(dict(gc_out or {}).get("removed_engine_ids") or [])),
                "removed_bundle_count": len(list(dict(gc_out or {}).get("removed_bundle_roots") or [])),
                "removed_environment_count": len(list(dict(gc_out or {}).get("removed_environment_keys") or [])),
                "repaired_toolbox_count": len(repaired_toolbox_ids),
                "requested_toolbox_count": len(repair_requested_toolbox_ids),
                "target_toolbox_count": len(repair_target_toolbox_ids),
            },
        }
        if details:
            result["before"] = before
            result["repair"] = repair
            result["gc"] = gc_out
            result["after"] = after
        return result

    @staticmethod
    def _referenced_toolbox_engine_ids(state: Optional[Dict[str, Any]] = None) -> set[str]:
        payload = dict(state or {})
        toolboxes = dict(payload.get("toolboxes") or {})
        referenced: set[str] = set()
        for toolbox_row in toolboxes.values():
            profiles = dict(dict(toolbox_row or {}).get("profiles") or {})
            for profile_row in profiles.values():
                engine_id = str(dict(profile_row or {}).get("engine_id") or "").strip()
                if engine_id:
                    referenced.add(engine_id)
        return referenced

    def _cleanup_stale_toolbox_registrations(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = dict(state or self._read_toolboxes() or {})
        referenced_engine_ids = self._referenced_toolbox_engine_ids(payload)
        removed: List[str] = []
        removed_bundle_roots: List[str] = []
        removed_details: List[Dict[str, Any]] = []
        for reg in self._read_engines():
            row = dict(reg or {})
            if str(row.get("executor_kind") or "").strip() != "toolbox_executor":
                continue
            engine_id = str(row.get("engine_id") or "").strip()
            if not engine_id or engine_id in referenced_engine_ids:
                continue
            bundle = dict(row.get("bundle") or {})
            raw_root = str(bundle.get("bundle_root") or "").strip()
            bundle_name = ""
            if raw_root:
                try:
                    bundle_name = Path(raw_root).expanduser().resolve().name
                except Exception:
                    bundle_name = ""
            self._retire_toolbox_registration(engine_id)
            removed.append(engine_id)
            if bundle_name:
                removed_bundle_roots.append(bundle_name)
            removed_details.append(
                {
                    "engine_id": engine_id,
                    "toolbox_id": self._registration_toolbox_id(row) or None,
                    "sandbox_profile_id": self._registration_sandbox_profile_id(row) or None,
                    "bundle_root": raw_root or None,
                    "bundle_name": bundle_name or None,
                    "reason": "unreferenced_live_registration",
                }
            )
        return {
            "removed_engine_ids": removed,
            "removed_bundle_roots": removed_bundle_roots,
            "removed_registration_details": removed_details,
        }

    def _cleanup_unused_toolbox_bundles(self) -> Dict[str, Any]:
        bundles_root = (self.hosting_root / "toolbox_bundles").resolve()
        removed: List[str] = []
        removed_details: List[Dict[str, Any]] = []
        if not bundles_root.exists():
            return {"removed_bundle_roots": removed, "removed_bundle_details": removed_details}
        referenced_roots: set[str] = set()
        for reg in self._read_engines():
            row = dict(reg or {})
            if str(row.get("executor_kind") or "").strip() != "toolbox_executor":
                continue
            bundle = dict(row.get("bundle") or {})
            raw = str(bundle.get("bundle_root") or "").strip()
            if not raw:
                continue
            try:
                referenced_roots.add(str(Path(raw).expanduser().resolve()))
            except Exception:
                continue
        for child in bundles_root.iterdir():
            if not child.is_dir():
                continue
            if self._bundle_directory_is_referenced(
                child,
                referenced_bundle_roots=referenced_roots,
            ):
                continue
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child.name)
            removed_details.append(
                {
                    "bundle_name": child.name,
                    "bundle_root": str(child),
                    "reason": "unreferenced_bundle_directory",
                }
            )
        return {
            "removed_bundle_roots": removed,
            "removed_bundle_details": removed_details,
        }

    def toolbox_gc(self) -> Dict[str, Any]:
        state = self._read_toolboxes()
        stale = self._cleanup_stale_toolbox_registrations(state)
        bundle_gc = self._cleanup_unused_toolbox_bundles()
        removed_bundle_roots = list(stale.get("removed_bundle_roots") or [])
        removed_bundle_roots.extend(list(bundle_gc.get("removed_bundle_roots") or []))
        removed_environment_keys = self._cleanup_unused_toolbox_environments(state)
        removed_environment_details = [
            {
                "venv_key": str(item or "").strip(),
                "reason": "unreferenced_environment_directory",
            }
            for item in list(removed_environment_keys or [])
            if str(item or "").strip()
        ]
        return {
            "status": "ok",
            "removed_engine_ids": list(stale.get("removed_engine_ids") or []),
            "removed_bundle_roots": list(dict.fromkeys([item for item in removed_bundle_roots if str(item or "").strip()])),
            "removed_environment_keys": removed_environment_keys,
            "removed_registration_details": list(stale.get("removed_registration_details") or []),
            "removed_bundle_details": list(bundle_gc.get("removed_bundle_details") or []),
            "removed_environment_details": removed_environment_details,
            "summary": {
                "removed_engine_count": len(list(stale.get("removed_engine_ids") or [])),
                "removed_bundle_count": len(list(dict.fromkeys([item for item in removed_bundle_roots if str(item or "").strip()]))),
                "removed_environment_count": len(removed_environment_keys),
            },
        }

    def toolbox_references(self) -> Dict[str, Any]:
        return self._toolbox_reference_report()

    @staticmethod
    def _append_toolbox_rollout_history(
        existing_profile_row: Optional[Dict[str, Any]],
        *,
        rollout: Dict[str, Any],
        action: str,
        engine_id: str,
        bundle_revision: str,
        replaced_engine_id: str = "",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        history = list(dict(existing_profile_row or {}).get("rollout_history") or [])
        entry = {
            "action": str(action or "").strip() or "rollout",
            "engine_id": str(engine_id or "").strip(),
            "bundle_revision": str(bundle_revision or "").strip(),
            "replaced_engine_id": str(replaced_engine_id or "").strip() or None,
            "ready": bool(dict(rollout or {}).get("ready", False)),
            "ready_at": dict(rollout or {}).get("ready_at"),
            "warmup_ms": int(dict(rollout or {}).get("warmup_ms") or 0),
        }
        history.append(entry)
        max_items = max(1, int(limit or 10))
        if len(history) > max_items:
            history = history[-max_items:]
        return history
