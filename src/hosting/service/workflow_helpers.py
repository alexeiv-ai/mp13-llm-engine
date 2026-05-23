from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional


class WorkflowHelperMixin:
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
        out = self.proxy_rpc_call(
            engine_id=str(engine_id or "").strip() or "workflow-js-helper",
            method="worker.resources",
            params={},
            timeout_seconds=10.0,
        )
        return self._enrich_workflow_js_helper_resources(dict(out.get("result") or out or {}))

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
        out = self.proxy_rpc_call(
            engine_id=str(engine_id or "").strip() or "workflow-js-helper",
            method="workflow_js_helper.set_capacity",
            params={"capacity": max(1, min(int(capacity or 1), 256))},
            timeout_seconds=10.0,
        )
        return self._enrich_workflow_js_helper_resources(dict(out.get("result") or out or {}))

    def cancel_workflow_js_helper_request(self, *, engine_id: str = "workflow-js-helper", request_id: str) -> Dict[str, Any]:
        out = self.proxy_rpc_call(
            engine_id=str(engine_id or "").strip() or "workflow-js-helper",
            method="workflow_js_helper.cancel_request",
            params={"request_id": str(request_id or "").strip()},
            timeout_seconds=10.0,
        )
        return dict(out.get("result") or out or {})

    def spawn_workflow_python_helper(
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
        out = self.proxy_rpc_call(
            engine_id=str(engine_id or "").strip() or "workflow-python-helper",
            method="worker.resources",
            params={},
            timeout_seconds=10.0,
        )
        return self._enrich_workflow_python_helper_resources(dict(out.get("result") or out or {}))

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
        out = self.proxy_rpc_call(
            engine_id=str(engine_id or "").strip() or "workflow-python-helper",
            method="workflow_python_helper.set_capacity",
            params={"capacity": max(1, min(int(capacity or 1), 256))},
            timeout_seconds=10.0,
        )
        return self._enrich_workflow_python_helper_resources(dict(out.get("result") or out or {}))

    def cancel_workflow_python_helper_request(self, *, engine_id: str = "workflow-python-helper", request_id: str) -> Dict[str, Any]:
        out = self.proxy_rpc_call(
            engine_id=str(engine_id or "").strip() or "workflow-python-helper",
            method="workflow_python_helper.cancel_request",
            params={"request_id": str(request_id or "").strip()},
            timeout_seconds=10.0,
        )
        return dict(out.get("result") or out or {})
