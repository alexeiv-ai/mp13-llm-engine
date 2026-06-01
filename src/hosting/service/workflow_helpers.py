from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..sandbox.python_runtime import HostedPythonRuntimeBase, HostedPythonRuntimeManager
from ..sandbox.js_runtime import HostedJsRuntimeBase
from ..sandbox.runtime_base import HostedPoolKey, HostedRequestLifecycle, HostedWorkerSlot
from ..sandbox.runtime_pool import HostedProcessPoolRegistry
from ..sandbox.workflow_python_contract import workflow_python_node_not_implemented_response


class WorkflowHelperMixin:
    def _workflow_python_runtime_manager(self) -> HostedPythonRuntimeManager:
        return HostedPythonRuntimeManager(self.hosting_root)

    def _workflow_js_runtime_base(self) -> HostedJsRuntimeBase:
        return HostedJsRuntimeBase(self.hosting_root)

    def _workflow_python_pool_registry(self) -> HostedProcessPoolRegistry:
        registry = getattr(self, "_workflow_python_runtime_pools", None)
        if registry is None:
            registry = HostedProcessPoolRegistry()
            setattr(self, "_workflow_python_runtime_pools", registry)
        return registry

    def _workflow_python_stream_base(self) -> HostedPythonRuntimeBase:
        base = getattr(self, "_workflow_python_stream_base_runtime", None)
        if base is None:
            base = HostedPythonRuntimeBase(self.hosting_root)
            base.pool_registry = self._workflow_python_pool_registry()
            setattr(self, "_workflow_python_stream_base_runtime", base)
        return base

    @staticmethod
    def _workflow_python_profile(profile: str) -> str:
        value = str(profile or "helper").strip().lower() or "helper"
        if value not in {"helper", "node"}:
            raise ValueError("profile must be 'helper' or 'node'")
        return value

    def _workflow_python_node_unavailable(
        self,
        *,
        request: Optional[Dict[str, Any]] = None,
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return workflow_python_node_not_implemented_response(
            environment_key=str(environment_key or ""),
            engine_id=str(engine_id or ""),
            request=dict(request or {}),
        )

    def workflow_python_environment_spec(
        self,
        *,
        profile: str = "helper",
        environment_name: str = "workflow-python-helper",
        python: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prof = self._workflow_python_profile(profile)
        return self._workflow_python_runtime_manager().environment_spec(
            environment_name=environment_name,
            profile=prof,
            python_policy=dict(python or {}),
            sandbox_policy=dict(sandbox_policy or {}),
        )

    def workflow_python_prepare_environment(
        self,
        *,
        environment_name: str = "workflow-python-helper",
        python: Optional[Dict[str, Any]] = None,
        package_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._workflow_python_runtime_manager().prepare_install(
            environment_name=environment_name,
            python_policy=dict(python or {}),
            package_id=package_id,
            workflow_id=workflow_id,
        )

    def workflow_python_lock_environment(self, *, environment: Dict[str, Any]) -> Dict[str, Any]:
        return self._workflow_python_runtime_manager().lock_install(environment=dict(environment or {}))

    def workflow_python_verify_environment(self, *, environment: Dict[str, Any]) -> Dict[str, Any]:
        return self._workflow_python_runtime_manager().verify_install_lock(environment=dict(environment or {}))

    def workflow_python_install_environment(self, *, environment: Dict[str, Any], allow_execution: bool = False) -> Dict[str, Any]:
        return self._workflow_python_runtime_manager().execute_install(
            environment=dict(environment or {}),
            allow_execution=bool(allow_execution),
        )

    def workflow_python_verify_install_receipt(self, *, environment: Dict[str, Any]) -> Dict[str, Any]:
        return self._workflow_python_runtime_manager().verify_install_receipt(environment=dict(environment or {}))

    def workflow_python_default_engine_id(self, *, environment_key: str) -> str:
        key = str(environment_key or "").strip()
        return f"workflow-python-{key[:16]}" if key else "workflow-python-helper"

    @staticmethod
    def _workflow_js_profile(profile: str) -> str:
        value = str(profile or "helper").strip().lower() or "helper"
        if value != "helper":
            raise ValueError("workflow_js currently supports only profile='helper'")
        return value

    def workflow_js_environment_spec(
        self,
        *,
        profile: str = "helper",
        environment_name: str = "workflow-js-helper",
        node: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        node_policy = dict(node or {})
        return self._workflow_js_runtime_base().environment_spec(
            profile=prof,
            environment_name=environment_name,
            node_policy=node_policy,
            sandbox_policy=sandbox_policy,
        )

    def _workflow_python_pool_key(self, environment_key: str) -> HostedPoolKey:
        return HostedPoolKey(sandbox_kind="workflow_python", environment_key=str(environment_key or "").strip())

    def _workflow_js_pool_key(self, environment_key: str) -> HostedPoolKey:
        return HostedPoolKey(sandbox_kind="workflow_js", environment_key=str(environment_key or "").strip())

    def _workflow_python_worker_slot(self, *, engine_id: str, environment_key: str, capacity: int) -> HostedWorkerSlot:
        reg = self.get_registration(engine_id)
        pid = int(dict(reg or {}).get("pid") or 0) or None
        return HostedWorkerSlot(
            engine_id=str(engine_id or "").strip(),
            environment_key=str(environment_key or "").strip(),
            capacity=max(1, int(capacity or 1)),
            pid=pid,
            status="registered" if reg else "unknown",
        )

    def _workflow_js_worker_slot(self, *, engine_id: str, environment_key: str, capacity: int) -> HostedWorkerSlot:
        reg = self.get_registration(engine_id)
        pid = int(dict(reg or {}).get("pid") or 0) or None
        return HostedWorkerSlot(
            engine_id=str(engine_id or "").strip(),
            environment_key=str(environment_key or "").strip(),
            capacity=max(1, int(capacity or 1)),
            pid=pid,
            status="registered" if reg else "unknown",
        )

    def _workflow_python_registration_environment_key(self, engine_id: Optional[str]) -> str:
        eid = str(engine_id or "").strip()
        if not eid:
            return ""
        reg = dict(self.get_registration(eid) or {})
        env = dict(reg.get("environment") or {})
        caps = dict(reg.get("capabilities") or {})
        return str(env.get("environment_key") or caps.get("environment_key") or "").strip()

    def _workflow_js_registration_environment_key(self, engine_id: Optional[str]) -> str:
        eid = str(engine_id or "").strip()
        if not eid:
            return ""
        reg = dict(self.get_registration(eid) or {})
        env = dict(reg.get("environment") or {})
        caps = dict(reg.get("capabilities") or {})
        return str(env.get("environment_key") or caps.get("environment_key") or "").strip()

    def _workflow_python_effective_environment_key(
        self,
        *,
        environment_key: Optional[str],
        engine_id: Optional[str],
        derived_environment_key: str,
        spec_was_explicit: bool = False,
    ) -> Dict[str, Any]:
        requested_key = str(environment_key or "").strip()
        registration_key = self._workflow_python_registration_environment_key(engine_id)
        derived_key = str(derived_environment_key or "").strip()
        if requested_key and registration_key and requested_key != registration_key:
            return {
                "status": "error",
                "reason": "environment_key_mismatch",
                "environment_key": requested_key,
                "registration_environment_key": registration_key,
            }
        if requested_key and spec_was_explicit and derived_key and requested_key != derived_key:
            return {
                "status": "error",
                "reason": "environment_key_mismatch",
                "environment_key": requested_key,
                "derived_environment_key": derived_key,
            }
        key = requested_key or registration_key or derived_key
        return {
            "status": "ok",
            "environment_key": key,
            "registration_environment_key": registration_key or None,
            "derived_environment_key": derived_key or None,
        }

    def _annotate_workflow_python_registration(
        self,
        *,
        engine_id: str,
        profile: str,
        environment_key: str,
        environment: Dict[str, Any],
    ) -> None:
        eid = str(engine_id or "").strip()
        if not eid:
            return
        rows = self._read_engines()
        changed = False
        for row in rows:
            if str(row.get("engine_id") or "").strip() != eid:
                continue
            env_row = dict(row.get("environment") or {})
            env_row.update(dict(environment or {}))
            env_row["environment_key"] = str(environment_key or "").strip() or None
            env_row["workflow_runtime_kind"] = "workflow_python"
            env_row["workflow_profile"] = str(profile or "helper").strip() or "helper"
            row["environment"] = env_row
            capabilities = dict(row.get("capabilities") or {})
            capabilities.update(
                {
                    "workflow_python": True,
                    "workflow_python_profile": str(profile or "helper").strip() or "helper",
                    "environment_key": str(environment_key or "").strip() or None,
                }
            )
            row["capabilities"] = capabilities
            changed = True
        if changed:
            self._write_engines(rows)

    def _annotate_workflow_js_registration(
        self,
        *,
        engine_id: str,
        profile: str,
        environment_key: str,
        environment: Dict[str, Any],
    ) -> None:
        eid = str(engine_id or "").strip()
        if not eid:
            return
        rows = self._read_engines()
        changed = False
        for row in rows:
            if str(row.get("engine_id") or "").strip() != eid:
                continue
            env_row = dict(row.get("environment") or {})
            env_row.update(dict(environment or {}))
            env_row["environment_key"] = str(environment_key or "").strip() or None
            env_row["workflow_runtime_kind"] = "workflow_js"
            env_row["workflow_profile"] = str(profile or "helper").strip() or "helper"
            row["environment"] = env_row
            capabilities = dict(row.get("capabilities") or {})
            capabilities.update(
                {
                    "workflow_js": True,
                    "workflow_js_profile": str(profile or "helper").strip() or "helper",
                    "environment_key": str(environment_key or "").strip() or None,
                }
            )
            row["capabilities"] = capabilities
            changed = True
        if changed:
            self._write_engines(rows)

    def workflow_js_default_engine_id(self, *, environment_key: str) -> str:
        key = str(environment_key or "").strip()
        return f"workflow-js-{key[:16]}" if key else "workflow-js-helper"

    def ensure_workflow_js(
        self,
        *,
        profile: str = "helper",
        environment_name: str = "workflow-js-helper",
        environment_key: Optional[str] = None,
        node: Optional[Dict[str, Any]] = None,
        node_executable: Optional[str] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        engine_id: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        node_policy = dict(node or {})
        if node_executable:
            node_policy.setdefault("node_executable", str(node_executable or ""))
        env = self.workflow_js_environment_spec(
            profile=prof,
            environment_name=environment_name,
            node=node_policy,
            sandbox_policy=sandbox_policy,
        )
        derived_key = str(env.get("environment_key") or "").strip()
        requested_key = str(environment_key or "").strip()
        if requested_key and requested_key != derived_key:
            return {
                "status": "error",
                "reason": "environment_key_mismatch",
                "environment_key": requested_key,
                "derived_environment_key": derived_key,
            }
        eid = str(engine_id or "").strip() or self.workflow_js_default_engine_id(environment_key=derived_key)
        existing = self.get_registration(eid)
        if existing:
            ensured = self.ensure_running(eid)
            self._annotate_workflow_js_registration(engine_id=eid, profile=prof, environment_key=derived_key, environment=dict(env.get("environment") or {}))
            pool = self._workflow_python_pool_registry().get_or_create(self._workflow_js_pool_key(derived_key), desired_capacity=capacity)
            pool.ensure_worker(lambda _key, cap: self._workflow_js_worker_slot(engine_id=eid, environment_key=derived_key, capacity=cap))
            return {"status": "ok", "outcome": "already_registered", "profile": prof, "engine_id": eid, "environment_key": derived_key, "environment": dict(env.get("environment") or {}), "ensure": dict(ensured or {})}
        spawned = self.spawn_workflow_js_helper(
            engine_id=eid,
            node_executable=node_executable or node_policy.get("node_executable"),
            capacity=capacity,
            sandbox_policy=sandbox_policy,
            worker_profile_class=worker_profile_class,
        )
        self._annotate_workflow_js_registration(engine_id=eid, profile=prof, environment_key=derived_key, environment=dict(env.get("environment") or {}))
        pool = self._workflow_python_pool_registry().get_or_create(self._workflow_js_pool_key(derived_key), desired_capacity=capacity)
        pool.ensure_worker(lambda _key, cap: self._workflow_js_worker_slot(engine_id=eid, environment_key=derived_key, capacity=cap))
        return {"status": "ok", "outcome": "spawned", "profile": prof, "engine_id": eid, "environment_key": derived_key, "environment": dict(env.get("environment") or {}), "spawn": dict(spawned or {})}

    def workflow_js_resources(
        self,
        *,
        profile: str = "helper",
        environment_name: str = "workflow-js-helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        node: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        env = self.workflow_js_environment_spec(profile=prof, environment_name=environment_name, node=node, sandbox_policy=sandbox_policy)
        derived_key = str(env.get("environment_key") or "").strip()
        registration_key = self._workflow_js_registration_environment_key(engine_id)
        requested_key = str(environment_key or "").strip()
        effective_key = requested_key or registration_key or derived_key
        if requested_key and registration_key and requested_key != registration_key:
            return {"status": "error", "reason": "environment_key_mismatch", "environment_key": requested_key, "registration_environment_key": registration_key}
        eid = str(engine_id or "").strip() or self.workflow_js_default_engine_id(environment_key=effective_key)
        resources = self.workflow_js_helper_resources(engine_id=eid)
        pool = self._workflow_python_pool_registry().get(self._workflow_js_pool_key(effective_key))
        return {**dict(resources or {}), "profile": prof, "engine_id": eid, "environment_key": effective_key, "environment": dict(env.get("environment") or {}), "workflow_pool": pool.resources() if pool is not None else None}

    def set_workflow_js_capacity(
        self,
        *,
        profile: str = "helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        capacity: int,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        effective_key = str(environment_key or "").strip() or self._workflow_js_registration_environment_key(engine_id)
        eid = str(engine_id or "").strip() or self.workflow_js_default_engine_id(environment_key=effective_key)
        out = self.set_workflow_js_helper_capacity(engine_id=eid, capacity=capacity)
        if effective_key:
            self._workflow_python_pool_registry().get_or_create(self._workflow_js_pool_key(effective_key), desired_capacity=capacity).set_capacity(capacity)
        return {**dict(out or {}), "profile": prof, "environment_key": effective_key or None}

    def cancel_workflow_js_request(
        self,
        *,
        profile: str = "helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request_id: str,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        effective_key = str(environment_key or "").strip() or self._workflow_js_registration_environment_key(engine_id)
        eid = str(engine_id or "").strip() or self.workflow_js_default_engine_id(environment_key=effective_key)
        out = self.cancel_workflow_js_helper_request(engine_id=eid, request_id=request_id)
        pool = self._workflow_python_pool_registry().get(self._workflow_js_pool_key(effective_key))
        if pool is not None and "workflow_pool_cancel" not in dict(out or {}):
            out["workflow_pool_cancel"] = pool.cancel_request(request_id)
        return {**dict(out or {}), "profile": prof, "environment_key": effective_key or None}

    def workflow_js_request_status(
        self,
        *,
        profile: str = "helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request_id: str,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        effective_key = str(environment_key or "").strip() or self._workflow_js_registration_environment_key(engine_id)
        if not effective_key:
            return {"status": "not_found", "request_id": str(request_id or "").strip(), "profile": prof, "environment_key": None}
        out = self._workflow_python_pool_registry().request_status(self._workflow_js_pool_key(effective_key), request_id)
        return {**dict(out or {}), "profile": prof, "environment_key": effective_key}

    def ensure_workflow_python(
        self,
        *,
        profile: str = "helper",
        environment_name: str = "workflow-python-helper",
        environment_key: Optional[str] = None,
        python: Optional[Dict[str, Any]] = None,
        python_executable: Optional[str] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        engine_id: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        prof = self._workflow_python_profile(profile)
        if prof != "helper":
            return self._workflow_python_node_unavailable(environment_key=environment_key, engine_id=engine_id)
        env = self.workflow_python_environment_spec(
            profile=prof,
            environment_name=environment_name,
            python=python,
            sandbox_policy=sandbox_policy,
        )
        derived_key = str(env.get("environment_key") or "").strip()
        requested_key = str(environment_key or "").strip()
        if requested_key and requested_key != derived_key:
            return {
                "status": "error",
                "reason": "environment_key_mismatch",
                "environment_key": requested_key,
                "derived_environment_key": derived_key,
            }
        eid = str(engine_id or "").strip() or self.workflow_python_default_engine_id(environment_key=derived_key)
        existing = self.get_registration(eid)
        if existing:
            ensured = self.ensure_running(eid)
            self._annotate_workflow_python_registration(
                engine_id=eid,
                profile=prof,
                environment_key=derived_key,
                environment=dict(env.get("environment") or {}),
            )
            pool = self._workflow_python_pool_registry().get_or_create(
                self._workflow_python_pool_key(derived_key),
                desired_capacity=capacity,
            )
            pool.ensure_worker(lambda _key, cap: self._workflow_python_worker_slot(engine_id=eid, environment_key=derived_key, capacity=cap))
            return {
                "status": "ok",
                "outcome": "already_registered",
                "profile": prof,
                "engine_id": eid,
                "environment_key": derived_key,
                "environment": dict(env.get("environment") or {}),
                "ensure": dict(ensured or {}),
            }
        spawned = self._spawn_workflow_python_helper_worker(
            engine_id=eid,
            python_executable=python_executable,
            capacity=capacity,
            sandbox_policy=sandbox_policy,
            worker_profile_class=worker_profile_class,
        )
        self._annotate_workflow_python_registration(
            engine_id=eid,
            profile=prof,
            environment_key=derived_key,
            environment=dict(env.get("environment") or {}),
        )
        pool = self._workflow_python_pool_registry().get_or_create(
            self._workflow_python_pool_key(derived_key),
            desired_capacity=capacity,
        )
        pool.ensure_worker(lambda _key, cap: self._workflow_python_worker_slot(engine_id=eid, environment_key=derived_key, capacity=cap))
        return {
            "status": "ok",
            "outcome": "spawned",
            "profile": prof,
            "engine_id": eid,
            "environment_key": derived_key,
            "environment": dict(env.get("environment") or {}),
            "spawn": dict(spawned or {}),
        }

    def execute_workflow_python(
        self,
        *,
        profile: str = "helper",
        environment_name: str = "workflow-python-helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prof = self._workflow_python_profile(profile)
        req = dict(request or {})
        if prof != "helper":
            return self._workflow_python_node_unavailable(request=req, environment_key=environment_key, engine_id=engine_id)
        py = dict(req.get("python") or {})
        if environment_name:
            py.setdefault("environment_name", str(environment_name or "workflow-python-helper"))
        req["python"] = py
        ensured = self.ensure_workflow_python(
            profile=prof,
            environment_name=str(py.get("environment_name") or environment_name or "workflow-python-helper"),
            environment_key=environment_key,
            python=py,
            capacity=capacity,
            sandbox_policy=sandbox_policy,
            engine_id=engine_id,
        )
        if str(ensured.get("status") or "") != "ok":
            return ensured
        pool = self._workflow_python_pool_registry().get_or_create(
            self._workflow_python_pool_key(str(ensured.get("environment_key") or "")),
            desired_capacity=capacity,
        )
        lifecycle = HostedRequestLifecycle(
            request_id=str(req.get("request_id") or "").strip() or "workflow-python-sync",
            environment_key=str(ensured.get("environment_key") or ""),
            sandbox_kind="workflow_python",
            profile=prof,
            engine_id=str(ensured["engine_id"]),
            submitted_at=time.time(),
        )
        scheduled = pool.submit_request(
            lifecycle,
            factory=lambda _key, cap: self._workflow_python_worker_slot(
                engine_id=str(ensured["engine_id"]),
                environment_key=str(ensured.get("environment_key") or ""),
                capacity=cap,
            ),
        )
        if str(scheduled.get("status") or "") != "ok":
            return {
                "status": "error",
                "ok": False,
                "profile": prof,
                "engine_id": str(ensured["engine_id"]),
                "environment_key": str(ensured.get("environment_key") or ""),
                "reason": str(scheduled.get("reason") or "capacity_exceeded"),
                "metrics": {"workflow_pool": pool.resources(), "request": dict(scheduled.get("request") or {})},
            }
        out = self.proxy_rpc_call(
            engine_id=str(ensured["engine_id"]),
            method="execute_workflow_python_helper",
            params={**req, "_workflow_python_facade_execute": True},
            timeout_seconds=float(dict(req.get("limits") or {}).get("timeout_ms") or 30000) / 1000.0 + 5.0,
        )
        result = dict(out.get("result") or out or {})
        finished = pool.finish_request(
            lifecycle.request_id,
            status="ok" if bool(result.get("ok", False)) else "error",
            reason=str(result.get("reason") or "") or None,
        )
        return {
            "status": "ok" if bool(result.get("ok", False)) else "error",
            "ok": bool(result.get("ok", False)),
            "profile": prof,
            "engine_id": str(ensured["engine_id"]),
            "environment_key": str(ensured.get("environment_key") or ""),
            "output": result.get("result"),
            "result": result,
            "metrics": {
                "workflow_pool": pool.resources(),
                "request": dict(finished.get("request") or lifecycle.to_dict()),
            },
        }

    def workflow_python_resources(
        self,
        *,
        profile: str = "helper",
        environment_name: str = "workflow-python-helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        python: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prof = self._workflow_python_profile(profile)
        if prof != "helper":
            return self._workflow_python_node_unavailable(environment_key=environment_key, engine_id=engine_id)
        spec_was_explicit = bool(dict(python or {}) or dict(sandbox_policy or {}))
        env = self.workflow_python_environment_spec(
            profile=prof,
            environment_name=environment_name,
            python=python,
            sandbox_policy=sandbox_policy,
        )
        derived_key = str(env.get("environment_key") or "").strip()
        resolved = self._workflow_python_effective_environment_key(
            environment_key=environment_key,
            engine_id=engine_id,
            derived_environment_key=derived_key,
            spec_was_explicit=spec_was_explicit,
        )
        if str(resolved.get("status") or "") != "ok":
            return resolved
        effective_key = str(resolved.get("environment_key") or "").strip()
        eid = str(engine_id or "").strip() or self.workflow_python_default_engine_id(environment_key=effective_key)
        resources = self.workflow_python_helper_resources(engine_id=eid)
        pool = self._workflow_python_pool_registry().get(self._workflow_python_pool_key(effective_key))
        return {
            **dict(resources or {}),
            "profile": prof,
            "engine_id": eid,
            "environment_key": effective_key,
            "environment": dict(env.get("environment") or {}),
            "workflow_pool": pool.resources() if pool is not None else None,
        }

    def set_workflow_python_capacity(
        self,
        *,
        profile: str = "helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        capacity: int,
    ) -> Dict[str, Any]:
        prof = self._workflow_python_profile(profile)
        if prof != "helper":
            return self._workflow_python_node_unavailable(environment_key=environment_key, engine_id=engine_id)
        effective_key = str(environment_key or "").strip() or self._workflow_python_registration_environment_key(engine_id)
        eid = str(engine_id or "").strip() or self.workflow_python_default_engine_id(environment_key=effective_key)
        out = self.set_workflow_python_helper_capacity(engine_id=eid, capacity=capacity)
        if effective_key:
            self._workflow_python_pool_registry().get_or_create(
                self._workflow_python_pool_key(effective_key),
                desired_capacity=capacity,
            ).set_capacity(capacity)
        return {**dict(out or {}), "profile": prof, "environment_key": effective_key or None}

    def cancel_workflow_python_request(
        self,
        *,
        profile: str = "helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request_id: str,
    ) -> Dict[str, Any]:
        prof = self._workflow_python_profile(profile)
        if prof != "helper":
            return self._workflow_python_node_unavailable(environment_key=environment_key, engine_id=engine_id)
        effective_key = str(environment_key or "").strip() or self._workflow_python_registration_environment_key(engine_id)
        eid = str(engine_id or "").strip() or self.workflow_python_default_engine_id(environment_key=effective_key)
        out = self.cancel_workflow_python_helper_request(engine_id=eid, request_id=request_id)
        pool = self._workflow_python_pool_registry().get(self._workflow_python_pool_key(effective_key))
        if pool is not None and "workflow_pool_cancel" not in dict(out or {}):
            out["workflow_pool_cancel"] = pool.cancel_request(request_id)
        return {**dict(out or {}), "profile": prof, "environment_key": effective_key or None}

    def workflow_python_request_status(
        self,
        *,
        profile: str = "helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request_id: str,
    ) -> Dict[str, Any]:
        prof = self._workflow_python_profile(profile)
        if prof != "helper":
            return self._workflow_python_node_unavailable(environment_key=environment_key, engine_id=engine_id, request={"request_id": request_id})
        effective_key = str(environment_key or "").strip() or self._workflow_python_registration_environment_key(engine_id)
        if not effective_key:
            return {"status": "not_found", "request_id": str(request_id or "").strip(), "profile": prof, "environment_key": None}
        out = self._workflow_python_pool_registry().request_status(self._workflow_python_pool_key(effective_key), request_id)
        return {**dict(out or {}), "profile": prof, "environment_key": effective_key}

    def workflow_python_stream_open(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-python-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        python: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
    ) -> Dict[str, Any]:
        prof = self._workflow_python_profile(profile)
        req = dict(request or {})
        py = dict(python or req.get("python") or {})
        env = self.workflow_python_environment_spec(
            profile=prof,
            environment_name=environment_name,
            python=py,
            sandbox_policy=sandbox_policy,
        )
        derived_key = str(env.get("environment_key") or "").strip()
        requested_key = str(environment_key or "").strip()
        if requested_key and requested_key != derived_key:
            return {
                "status": "error",
                "reason": "environment_key_mismatch",
                "environment_key": requested_key,
                "derived_environment_key": derived_key,
            }
        effective_key = requested_key or derived_key
        request_id = str(req.get("request_id") or "").strip() or f"workflow-python-{prof}-{int(time.time() * 1000)}"
        base = self._workflow_python_stream_base()
        opened = base.stream_open(
            environment_key=effective_key,
            request_id=request_id,
            profile=prof,
            desired_capacity=capacity,
            factory=lambda _key, cap: self._workflow_python_worker_slot(
                engine_id=str(engine_id or self.workflow_python_default_engine_id(environment_key=effective_key)),
                environment_key=effective_key,
                capacity=cap,
            ),
        )
        if str(opened.get("status") or "") != "ok":
            return {**dict(opened or {}), "profile": prof, "environment_key": effective_key}
        if prof == "node":
            pending = self._workflow_python_node_unavailable(request={**req, "request_id": request_id}, environment_key=effective_key, engine_id=engine_id)
            base.stream_emit(
                stream_id=str(opened.get("stream_id") or ""),
                event_type="log",
                payload={"logs": dict(pending.get("logs") or {})},
            )
            base.stream_emit(
                stream_id=str(opened.get("stream_id") or ""),
                event_type="error",
                payload={"error": dict(pending.get("error") or {}), "response": pending},
            )
            base.stream_emit(
                stream_id=str(opened.get("stream_id") or ""),
                event_type="done",
                payload={"status": "error", "reason": pending.get("reason")},
            )
            base.finish_request(
                environment_key=effective_key,
                request_id=request_id,
                status="error",
                reason=str(pending.get("reason") or "workflow_python_node_profile_not_implemented"),
            )
        return {
            **dict(opened or {}),
            "profile": prof,
            "environment_key": effective_key,
            "environment": dict(env.get("environment") or {}),
        }

    def workflow_python_stream_recv(self, *, stream_id: str, max_items: int = 64) -> Dict[str, Any]:
        return dict(self._workflow_python_stream_base().stream_recv(stream_id=stream_id, max_items=max_items))

    def workflow_python_stream_send(self, *, stream_id: str, message: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return dict(self._workflow_python_stream_base().stream_send(stream_id=stream_id, message=dict(message or {})))

    def workflow_python_stream_close(self, *, stream_id: str) -> Dict[str, Any]:
        return dict(self._workflow_python_stream_base().stream_close(stream_id=stream_id))

    @staticmethod
    def workflow_js_helper_default_sandbox_policy() -> Dict[str, Any]:
        return {
            "sandbox": {
                "enabled": True,
                "profile": "workflow_js_helper_v1",
                "process": {
                    "allow_subprocess": False,
                },
                "network": {
                    "mode": "disabled",
                },
                "brokered_io": {
                    "filesystem": False,
                    "http": False,
                    "subprocess": False,
                },
            }
        }

    @staticmethod
    def workflow_python_helper_default_sandbox_policy() -> Dict[str, Any]:
        return {
            "sandbox": {
                "enabled": True,
                "profile": "workflow_python_helper_v1",
                "process": {
                    "allow_subprocess": False,
                },
                "network": {
                    "mode": "disabled",
                },
                "brokered_io": {
                    "filesystem": False,
                    "http": False,
                    "subprocess": False,
                },
            }
        }

    def spawn_workflow_js_helper(
        self,
        *,
        engine_id: str = "workflow-js-helper",
        node_executable: Optional[str] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip() or "workflow-js-helper"
        call_capacity = max(1, min(int(capacity or 1), 256))
        env = {
            "MP13_WORKER_CONTRACT": "hosting.workflow_helper.worker.v1",
            "MP13_WORKFLOW_HELPER_WORKER_ID": eid,
            "MP13_ENGINE_ID": eid,
            "MP13_WORKFLOW_JS_HELPER_CAPACITY": str(call_capacity),
        }
        src_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = str(os.environ.get("PYTHONPATH") or "").strip()
        env["PYTHONPATH"] = src_root if not existing_pythonpath else os.pathsep.join([src_root, existing_pythonpath])
        node = str(node_executable or "").strip()
        if node:
            env["MP13_WORKFLOW_JS_NODE"] = node
        policy = dict(sandbox_policy or self.workflow_js_helper_default_sandbox_policy())
        return self.spawn(
            engine_id=eid,
            command=[sys.executable, "-m", "hosting.workflow_js_helper_ipc"],
            env=env,
            worker_profile_class=str(worker_profile_class or "generic").strip() or "generic",
            sandbox_policy=policy,
            executor_kind="workflow_js_helper",
            capabilities={
                "workflow_js_helper": True,
                "execution_contract": "hosting.workflow_helper.worker.v1",
                "sandbox_profile": "workflow_js_helper_v1",
                "capacity": call_capacity,
            },
        )

    def workflow_js_helper_resources(self, *, engine_id: str = "workflow-js-helper") -> Dict[str, Any]:
        eid = str(engine_id or "").strip() or "workflow-js-helper"
        out = self.proxy_rpc_call(
            engine_id=eid,
            method="worker.resources",
            params={},
            timeout_seconds=10.0,
        )
        result = self._enrich_workflow_js_helper_resources(dict(out.get("result") or out or {}))
        return self._attach_workflow_js_alias_pool(engine_id=eid, result=result)

    def _attach_workflow_js_alias_pool(self, *, engine_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(result or {})
        environment_key = self._workflow_js_registration_environment_key(engine_id)
        if not environment_key:
            return out
        pool = self._workflow_python_pool_registry().get(self._workflow_js_pool_key(environment_key))
        out["workflow_runtime_kind"] = "workflow_js"
        out["workflow_profile"] = "helper"
        out["environment_key"] = environment_key
        out["workflow_pool"] = pool.resources() if pool is not None else None
        return out

    def _enrich_workflow_js_helper_resources(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(resources or {})
        pool = dict(result.get("node_pool") or {})
        generic_pool = dict(result.get("pool") or {})
        nodes = []
        total_cpu = 0.0
        total_mem = 0.0
        known_cpu = False
        known_mem = False
        snapshot_fn = getattr(self, "_process_resource_snapshot", None)
        for raw_node in list(pool.get("node_processes") or []):
            node = dict(raw_node or {})
            pid = int(node.get("pid") or 0)
            if pid > 0 and callable(snapshot_fn):
                try:
                    metrics = dict(snapshot_fn(pid) or {})
                except Exception:
                    metrics = {}
                if metrics.get("cpu_percent") is not None:
                    known_cpu = True
                    total_cpu += float(metrics.get("cpu_percent") or 0.0)
                if metrics.get("memory_mb") is not None:
                    known_mem = True
                    total_mem += float(metrics.get("memory_mb") or 0.0)
                node["resources"] = metrics
            nodes.append(node)
        if nodes:
            pool["node_processes"] = nodes
            pool["active_request_ids"] = [
                str(dict(row or {}).get("active_request_id") or "").strip()
                for row in nodes
                if str(dict(row or {}).get("active_request_id") or "").strip()
            ]
            generic_pool["processes"] = nodes
            generic_pool["active_request_ids"] = list(pool["active_request_ids"])
        if known_cpu:
            pool["node_cpu_percent"] = round(total_cpu, 1)
            result["node_cpu_percent"] = round(total_cpu, 1)
        if known_mem:
            pool["node_memory_mb"] = round(total_mem, 1)
            result["node_memory_mb"] = round(total_mem, 1)
        generic_pool.setdefault("process_count", int(pool.get("node_process_count") or 0))
        generic_pool.setdefault("active_process_count", int(pool.get("active_node_process_count") or 0))
        generic_pool.setdefault("idle_process_count", int(pool.get("idle_node_process_count") or 0))
        generic_pool.setdefault("active_request_ids", list(pool.get("active_request_ids") or []))
        generic_pool.setdefault("processes", list(pool.get("node_processes") or []))
        result["node_pool"] = pool
        result["pool"] = generic_pool
        return result

    def set_workflow_js_helper_capacity(self, *, engine_id: str = "workflow-js-helper", capacity: int) -> Dict[str, Any]:
        eid = str(engine_id or "").strip() or "workflow-js-helper"
        out = self.proxy_rpc_call(
            engine_id=eid,
            method="workflow_js_helper.set_capacity",
            params={"capacity": max(1, min(int(capacity or 1), 256))},
            timeout_seconds=10.0,
        )
        result = self._enrich_workflow_js_helper_resources(dict(out.get("result") or out or {}))
        environment_key = self._workflow_js_registration_environment_key(eid)
        if environment_key:
            self._workflow_python_pool_registry().get_or_create(self._workflow_js_pool_key(environment_key), desired_capacity=capacity).set_capacity(capacity)
        return self._attach_workflow_js_alias_pool(engine_id=eid, result=result)

    def cancel_workflow_js_helper_request(self, *, engine_id: str = "workflow-js-helper", request_id: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip() or "workflow-js-helper"
        out = self.proxy_rpc_call(
            engine_id=eid,
            method="workflow_js_helper.cancel_request",
            params={"request_id": str(request_id or "").strip()},
            timeout_seconds=10.0,
        )
        result = dict(out.get("result") or out or {})
        environment_key = self._workflow_js_registration_environment_key(eid)
        if environment_key:
            pool = self._workflow_python_pool_registry().get(self._workflow_js_pool_key(environment_key))
            if pool is not None:
                result["workflow_pool_cancel"] = pool.cancel_request(request_id)
        return self._attach_workflow_js_alias_pool(engine_id=eid, result=result)

    def spawn_workflow_python_helper(
        self,
        *,
        engine_id: str = "workflow-python-helper",
        python_executable: Optional[str] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        python_policy: Dict[str, Any] = {}
        if python_executable:
            python_policy["bootstrap_python_executable"] = str(python_executable or "").strip()
        ensured = self.ensure_workflow_python(
            profile="helper",
            environment_name="workflow-python-helper",
            python=python_policy,
            python_executable=python_executable,
            capacity=capacity,
            sandbox_policy=sandbox_policy,
            engine_id=engine_id,
            worker_profile_class=worker_profile_class,
        )
        spawn_result = dict(ensured.get("spawn") or {})
        if spawn_result:
            return {
                **spawn_result,
                "workflow_runtime_kind": "workflow_python",
                "workflow_profile": "helper",
                "environment_key": ensured.get("environment_key"),
                "environment": dict(ensured.get("environment") or {}),
                "workflow_ensure": ensured,
            }
        return ensured

    def _spawn_workflow_python_helper_worker(
        self,
        *,
        engine_id: str = "workflow-python-helper",
        python_executable: Optional[str] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip() or "workflow-python-helper"
        call_capacity = max(1, min(int(capacity or 1), 256))
        env = {
            "MP13_WORKER_CONTRACT": "hosting.workflow_helper.worker.v1",
            "MP13_WORKFLOW_HELPER_WORKER_ID": eid,
            "MP13_ENGINE_ID": eid,
            "MP13_WORKFLOW_PYTHON_HELPER_CAPACITY": str(call_capacity),
        }
        src_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = str(os.environ.get("PYTHONPATH") or "").strip()
        env["PYTHONPATH"] = src_root if not existing_pythonpath else os.pathsep.join([src_root, existing_pythonpath])
        py = str(python_executable or "").strip()
        if py:
            env["MP13_WORKFLOW_PYTHON"] = py
        policy = dict(sandbox_policy or self.workflow_python_helper_default_sandbox_policy())
        return self.spawn(
            engine_id=eid,
            command=[sys.executable, "-m", "hosting.workflow_python_helper_ipc"],
            env=env,
            worker_profile_class=str(worker_profile_class or "generic").strip() or "generic",
            sandbox_policy=policy,
            executor_kind="workflow_python_helper",
            capabilities={
                "workflow_python_helper": True,
                "execution_contract": "hosting.workflow_helper.worker.v1",
                "sandbox_profile": "workflow_python_helper_v1",
                "capacity": call_capacity,
            },
        )

    def workflow_python_helper_resources(self, *, engine_id: str = "workflow-python-helper") -> Dict[str, Any]:
        eid = str(engine_id or "").strip() or "workflow-python-helper"
        out = self.proxy_rpc_call(
            engine_id=eid,
            method="worker.resources",
            params={},
            timeout_seconds=10.0,
        )
        result = self._enrich_workflow_python_helper_resources(dict(out.get("result") or out or {}))
        return self._attach_workflow_python_alias_pool(engine_id=eid, result=result)

    def _attach_workflow_python_alias_pool(self, *, engine_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(result or {})
        environment_key = self._workflow_python_registration_environment_key(engine_id)
        if not environment_key:
            return out
        pool = self._workflow_python_pool_registry().get(self._workflow_python_pool_key(environment_key))
        out["workflow_runtime_kind"] = "workflow_python"
        out["workflow_profile"] = "helper"
        out["environment_key"] = environment_key
        out["workflow_pool"] = pool.resources() if pool is not None else None
        return out

    def _enrich_workflow_python_helper_resources(self, resources: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(resources or {})
        pool = dict(result.get("pool") or {})
        processes = []
        total_cpu = 0.0
        total_mem = 0.0
        known_cpu = False
        known_mem = False
        snapshot_fn = getattr(self, "_process_resource_snapshot", None)
        for raw_proc in list(pool.get("processes") or []):
            proc = dict(raw_proc or {})
            pid = int(proc.get("pid") or 0)
            if pid > 0 and callable(snapshot_fn):
                try:
                    metrics = dict(snapshot_fn(pid) or {})
                except Exception:
                    metrics = {}
                if metrics.get("cpu_percent") is not None:
                    known_cpu = True
                    total_cpu += float(metrics.get("cpu_percent") or 0.0)
                if metrics.get("memory_mb") is not None:
                    known_mem = True
                    total_mem += float(metrics.get("memory_mb") or 0.0)
                proc["resources"] = metrics
            processes.append(proc)
        if processes:
            pool["processes"] = processes
            pool["active_request_ids"] = [
                str(dict(row or {}).get("active_request_id") or "").strip()
                for row in processes
                if str(dict(row or {}).get("active_request_id") or "").strip()
            ]
        if known_cpu:
            pool["cpu_percent"] = round(total_cpu, 1)
            result["python_cpu_percent"] = round(total_cpu, 1)
        if known_mem:
            pool["memory_mb"] = round(total_mem, 1)
            result["python_memory_mb"] = round(total_mem, 1)
        result["pool"] = pool
        return result

    def set_workflow_python_helper_capacity(self, *, engine_id: str = "workflow-python-helper", capacity: int) -> Dict[str, Any]:
        eid = str(engine_id or "").strip() or "workflow-python-helper"
        out = self.proxy_rpc_call(
            engine_id=eid,
            method="workflow_python_helper.set_capacity",
            params={"capacity": max(1, min(int(capacity or 1), 256))},
            timeout_seconds=10.0,
        )
        result = self._enrich_workflow_python_helper_resources(dict(out.get("result") or out or {}))
        environment_key = self._workflow_python_registration_environment_key(eid)
        if environment_key:
            self._workflow_python_pool_registry().get_or_create(
                self._workflow_python_pool_key(environment_key),
                desired_capacity=capacity,
            ).set_capacity(capacity)
        return self._attach_workflow_python_alias_pool(engine_id=eid, result=result)

    def cancel_workflow_python_helper_request(self, *, engine_id: str = "workflow-python-helper", request_id: str) -> Dict[str, Any]:
        eid = str(engine_id or "").strip() or "workflow-python-helper"
        out = self.proxy_rpc_call(
            engine_id=eid,
            method="workflow_python_helper.cancel_request",
            params={"request_id": str(request_id or "").strip()},
            timeout_seconds=10.0,
        )
        result = dict(out.get("result") or out or {})
        environment_key = self._workflow_python_registration_environment_key(eid)
        if environment_key:
            pool = self._workflow_python_pool_registry().get(self._workflow_python_pool_key(environment_key))
            if pool is not None:
                result["workflow_pool_cancel"] = pool.cancel_request(request_id)
        return self._attach_workflow_python_alias_pool(engine_id=eid, result=result)
