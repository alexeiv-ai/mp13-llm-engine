from __future__ import annotations

import sys
import os
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..sandbox.python_runtime import HostedPythonRuntimeBase, HostedPythonRuntimeManager
from ..sandbox.js_runtime import HostedJsRuntimeBase
from ..sandbox.artifacts import HostedArtifactManager, artifact_safe_name
from ..sandbox.broker_http import BrokeredHttpClient
from ..sandbox.host_api import HostApiRegistry, fs_root_args_schema, fs_write_text_args_schema
from ..sandbox.policy import WorkerSandboxPolicy
from ..sandbox.runtime_base import HostedPoolKey, HostedRequestLifecycle, HostedWorkerSlot, hosted_log_summary
from ..sandbox.runtime_pool import HostedProcessPoolRegistry
from ..sandbox.workflow_js_node_runtime import WorkflowJsNodeRuntimeRegistry
from ..sandbox.workflow_python_node_runtime import WorkflowPythonNodeRuntimeRegistry
from ..sandbox.workflow_python_contract import (
    normalize_workflow_python_node_request,
    validate_workflow_python_node_request,
    workflow_python_node_contract,
    workflow_python_node_not_implemented_response,
)


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

    def _workflow_js_stream_base(self) -> HostedJsRuntimeBase:
        base = getattr(self, "_workflow_js_stream_base_runtime", None)
        if base is None:
            base = HostedJsRuntimeBase(self.hosting_root)
            base.pool_registry = self._workflow_python_pool_registry()
            setattr(self, "_workflow_js_stream_base_runtime", base)
        return base

    def _workflow_python_node_runtime_registry(self) -> WorkflowPythonNodeRuntimeRegistry:
        registry = getattr(self, "_workflow_python_node_runtime_registry_instance", None)
        if registry is None:
            registry = WorkflowPythonNodeRuntimeRegistry()
            setattr(self, "_workflow_python_node_runtime_registry_instance", registry)
        return registry

    def _workflow_js_node_runtime_registry(self) -> WorkflowJsNodeRuntimeRegistry:
        registry = getattr(self, "_workflow_js_node_runtime_registry_instance", None)
        if registry is None:
            registry = WorkflowJsNodeRuntimeRegistry()
            setattr(self, "_workflow_js_node_runtime_registry_instance", registry)
        return registry

    def _workflow_python_node_recycle_changed_environment(
        self,
        *,
        environment_name: str,
        environment_key: str,
    ) -> Dict[str, Any]:
        name = str(environment_name or "workflow-python-node").strip() or "workflow-python-node"
        key = str(environment_key or "").strip()
        if not key:
            return {"status": "skipped", "reason": "environment_key_missing", "stopped_count": 0}
        seen = getattr(self, "_workflow_python_node_environment_keys_by_name", None)
        if seen is None:
            seen = {}
            setattr(self, "_workflow_python_node_environment_keys_by_name", seen)
        lock = getattr(self, "_workflow_python_node_environment_keys_lock", None)
        if lock is None:
            lock = threading.Lock()
            setattr(self, "_workflow_python_node_environment_keys_lock", lock)
        with lock:
            previous = str(dict(seen).get(name) or "").strip()
            seen[name] = key
        if previous and previous != key:
            return self._workflow_python_node_runtime_registry().recycle_idle(
                environment_key=previous,
                reason="environment_identity_changed",
            )
        return {"status": "ok", "environment_name": name, "environment_key": key, "previous_environment_key": previous or None, "stopped_count": 0}

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

    @staticmethod
    def _workflow_python_node_response_from_execution(
        *,
        execution: Dict[str, Any],
        request: Dict[str, Any],
        environment_key: str,
        engine_id: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized = normalize_workflow_python_node_request(dict(request or {}))
        result = dict(execution or {})
        ok = bool(result.get("ok", False))
        reason = str(result.get("reason") or "").strip()
        detail = dict(result.get("detail") or {}) if isinstance(result.get("detail"), dict) else {}
        limits = dict(normalized.get("limits") or {})
        logs = hosted_log_summary(
            stdout=str(result.get("stdout") or ""),
            stderr=str(result.get("stderr") or ""),
            max_bytes=int(limits.get("output_limit_bytes") or 4096),
        )
        status = "ok" if ok else ("canceled" if reason == "workflow_sandbox_canceled" else "error")
        artifact_rows = list(result.get("artifacts") or []) if isinstance(result.get("artifacts"), list) else []
        artifact_store = (
            {
                "status": "ok",
                "kind": "local",
                "reason": None,
                "message": "artifact refs were minted from host-provided workflow Python output paths",
            }
            if artifact_rows
            else {
                "status": "unavailable",
                "reason": "artifact_store_no_refs",
                "message": "no host-minted artifact refs were created for this response",
            }
        )
        return {
            "status": status,
            "ok": ok,
            "profile": "node",
            "environment_key": str(environment_key or "").strip() or None,
            "engine_id": str(engine_id or "").strip() or None,
            "request_id": str(normalized.get("request_id") or "").strip() or None,
            "reason": None if ok else (reason or "workflow_python_node_execution_failed"),
            "error": None
            if ok
            else {
                "code": reason or "workflow_python_node_execution_failed",
                "message": str(detail.get("message") or reason or "workflow Python node execution failed"),
                "detail": detail,
            },
            "output": result.get("output") if ok else None,
            "state_patch": dict(result.get("state_patch") or {}) or None,
            "artifacts": artifact_rows,
            "artifact_store": artifact_store,
            "progress": dict(result.get("progress") or {}) or None,
            "logs": logs,
            "metrics": dict(metrics or {}),
            "audit": {
                "package_id": str(normalized.get("package_id") or "").strip() or None,
                "workflow_id": str(normalized.get("workflow_id") or "").strip() or None,
                "package_source_digest": str(normalized.get("package_source_digest") or "").strip() or None,
                "module_sha256": str(normalized.get("module_sha256") or "").strip() or None,
                "provenance": dict(normalized.get("provenance") or {}),
                "runtime": {
                    "python_executable": str(dict(normalized.get("python") or {}).get("python_executable") or sys.executable),
                    "runtime_kind": "workflow_python",
                    "profile": "node",
                },
            },
            "contract": workflow_python_node_contract(),
        }

    @staticmethod
    def _workflow_python_node_has_dependency_intent(python: Dict[str, Any]) -> bool:
        py = dict(python or {})
        package_pins = {
            str(key or "").strip(): str(value or "").strip()
            for key, value in dict(py.get("package_pins") or {}).items()
            if str(key or "").strip() and str(value or "").strip()
        }
        uv = py.get("uv")
        uv_intent = bool(uv) if isinstance(uv, dict) else bool(py.get("uv_enabled") or py.get("pyproject_toml") or py.get("uv_lock"))
        return bool(package_pins or str(py.get("dependency_lock_hash") or "").strip() or uv_intent)

    @staticmethod
    def _workflow_python_node_has_uv_intent(python: Dict[str, Any]) -> bool:
        py = dict(python or {})
        uv = py.get("uv")
        return bool(uv) if isinstance(uv, dict) else bool(py.get("uv_enabled") or py.get("pyproject_toml") or py.get("uv_lock"))

    @staticmethod
    def _workflow_python_with_project_artifact_input(request: Dict[str, Any]) -> Dict[str, Any]:
        req = dict(request or {})
        mode = str(req.get("execution_mode") or dict(req.get("python") or {}).get("execution_mode") or "").strip().lower()
        if mode != "project":
            return req
        project = dict(req.get("project") or {})
        ref = str(project.get("ref") or project.get("root_ref") or "").strip()
        if not ref:
            return req
        root_input = str(project.get("root_input") or project.get("input") or "project").strip() or "project"
        artifacts = [dict(row or {}) for row in list(req.get("artifact_inputs") or []) if isinstance(row, dict)]
        if not any(str(row.get("name") or "").strip() == root_input for row in artifacts):
            artifacts.append(
                {
                    "name": root_input,
                    "ref": ref,
                    "path_mask": str(project.get("path_mask") or project.get("mask") or "*").strip() or "*",
                    "recursive": True if "recursive" not in project else bool(project.get("recursive")),
                }
            )
        project.setdefault("root_input", root_input)
        req["project"] = project
        req["artifact_inputs"] = artifacts
        return req

    def _workflow_python_node_dependency_environment_check(
        self,
        *,
        request: Dict[str, Any],
        python: Dict[str, Any],
        environment: Dict[str, Any],
        environment_key: str,
        engine_id: str,
    ) -> Optional[Dict[str, Any]]:
        if not self._workflow_python_node_has_dependency_intent(python):
            return None
        try:
            verified = self.workflow_python_verify_install_receipt(environment=dict(environment or {}))
            install_status = dict(verified.get("install_status") or {})
        except Exception as exc:
            install_status = {
                "install_plan_status": "missing" if "install_plan_missing" in str(exc) else "error",
                "install_receipt_verification_status": "not_checked",
                "uv_install_plan_status": "missing" if "install_plan_missing" in str(exc) else "error",
                "uv_install_execution_status": "not_executed",
                "uv_install_receipt_status": "missing",
                "uv_install_receipt_verification_status": "not_checked",
                "reason": str(exc),
            }
        if self._workflow_python_node_has_uv_intent(python):
            if str(install_status.get("uv_install_plan_status") or "").strip() in {"", "missing"}:
                reason = "workflow_python_environment_not_prepared"
            elif (
                str(install_status.get("uv_install_execution_status") or "").strip() != "ok"
                or str(install_status.get("uv_install_receipt_status") or "").strip() != "ok"
                or str(install_status.get("uv_install_receipt_verification_status") or "").strip() != "ok"
            ):
                reason = "workflow_python_environment_unverified"
            else:
                reason = ""
        elif str(install_status.get("install_plan_status") or "").strip() in {"", "missing"}:
            reason = "workflow_python_environment_not_prepared"
        elif (
            str(install_status.get("install_execution_status") or "").strip() != "ok"
            or str(install_status.get("install_receipt_status") or "").strip() != "ok"
            or str(install_status.get("install_receipt_verification_status") or "").strip() != "ok"
        ):
            reason = "workflow_python_environment_unverified"
        else:
            reason = ""
        if not reason:
            selected = self._workflow_python_runtime_manager().select_runtime_python(
                environment=dict(environment or {}),
                bootstrap_python_executable=str(python.get("bootstrap_python_executable") or python.get("python_executable") or "").strip() or None,
                fallback_python_executable=str(python.get("fallback_python_executable") or python.get("python_executable") or "").strip() or None,
            )
            return {"status": "ok", "runtime": dict(selected or {}), "install_status": install_status}
        return self._workflow_python_node_response_from_execution(
            execution={
                "ok": False,
                "reason": reason,
                "detail": {
                    "message": "dependency-bearing workflow Python node requests require a prepared and verified runtime environment",
                    "environment_key": str(environment_key or "").strip() or None,
                    "install_status": install_status,
                },
            },
            request={**dict(request or {}), "python": dict(python or {})},
            environment_key=environment_key,
            engine_id=engine_id,
        )

    def _workflow_python_artifact_root(self) -> Path:
        root = Path(self.hosting_root).expanduser().resolve() / "workflow_artifacts"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _workflow_python_artifact_manager(self, *, sandbox_policy: Optional[Dict[str, Any]] = None) -> HostedArtifactManager:
        sandbox = dict(dict(sandbox_policy or {}).get("sandbox") or sandbox_policy or {})
        configured = sandbox.get("artifact_roots")
        if isinstance(configured, dict):
            rows = [{"name": key, "path": value} for key, value in configured.items()]
        else:
            rows = [dict(row or {}) for row in list(configured or []) if isinstance(row, dict)]
        roots: Dict[str, Path] = {}
        for row in rows:
            alias = artifact_safe_name(row.get("name") or row.get("root_id") or row.get("alias"), fallback="")
            if alias:
                roots[alias] = Path(str(row.get("path") or "")).expanduser().resolve()
        return HostedArtifactManager(artifact_root=self._workflow_python_artifact_root(), artifact_roots=roots)

    def _workflow_python_prepare_node_artifacts(
        self,
        *,
        request: Dict[str, Any],
        request_id: str,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._workflow_python_artifact_manager(sandbox_policy=sandbox_policy).prepare(
            request=dict(request or {}),
            request_id=str(request_id or ""),
        )

    def _workflow_python_collect_node_artifacts(
        self,
        context: Dict[str, Any],
        *,
        request_id: str,
        runtime_artifacts: Optional[list[Dict[str, Any]]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> list[Dict[str, Any]]:
        return self._workflow_python_artifact_manager(sandbox_policy=sandbox_policy).collect(
            dict(context or {}),
            request_id=str(request_id or ""),
            runtime_artifacts=list(runtime_artifacts or []),
        )

    def _workflow_python_cleanup_node_artifacts(
        self,
        context: Optional[Dict[str, Any]],
        *,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if context is None:
            return {"status": "skipped", "reason": "artifact_context_missing"}
        return self._workflow_python_artifact_manager(sandbox_policy=sandbox_policy).cleanup_run(dict(context or {}))

    @staticmethod
    def _workflow_python_node_host_root(
        artifact_context: Optional[Dict[str, Any]],
        *,
        root_id: str,
        write: bool = False,
    ) -> Path:
        child = dict(dict(artifact_context or {}).get("child_context") or artifact_context or {})
        inputs = dict(child.get("inputs") or {})
        outputs = dict(child.get("outputs") or {})
        rid = str(root_id or "").strip()
        if not rid:
            raise PermissionError("root_id_required")
        source = outputs if write else {**outputs, **inputs}
        raw = str(source.get(rid) or "").strip()
        if not raw:
            raise PermissionError(f"artifact_root_unavailable:{rid}")
        return Path(raw).expanduser().resolve()

    @staticmethod
    def _workflow_python_node_host_path(root: Path, relative_path: Any = None) -> Path:
        rel = str(relative_path or "").replace("\\", "/").strip("/")
        target = (root / rel).expanduser().resolve() if rel else root
        if target != root and root not in target.parents:
            raise PermissionError("artifact_path_escape")
        return target

    def _workflow_python_node_host_dispatcher(
        self,
        *,
        request: Dict[str, Any],
        artifact_context: Optional[Dict[str, Any]],
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ):
        child = dict(dict(artifact_context or {}).get("child_context") or artifact_context or {})
        input_roots = sorted(str(key) for key in dict(child.get("inputs") or {}).keys())
        output_roots = sorted(str(key) for key in dict(child.get("outputs") or {}).keys())
        sandbox = dict(dict(sandbox_policy or {}).get("sandbox") or sandbox_policy or {})
        host_api_policy = sandbox.get("host_api") if isinstance(sandbox.get("host_api"), dict) else {}
        namespace_policy = dict(host_api_policy.get("namespaces") or {})
        artifact_fs_enabled = bool(host_api_policy.get("enabled", True))
        http_namespace_enabled = bool(host_api_policy.get("enabled", True))
        for key in ("fs", "artifact_fs"):
            if key in host_api_policy:
                artifact_fs_enabled = bool(host_api_policy.get(key))
            if key in namespace_policy:
                artifact_fs_enabled = bool(namespace_policy.get(key))
        for key in ("http", "http_fetch"):
            if key in host_api_policy:
                http_namespace_enabled = bool(host_api_policy.get(key))
            if key in namespace_policy:
                http_namespace_enabled = bool(namespace_policy.get(key))
        worker_policy = WorkerSandboxPolicy.from_mapping(dict(sandbox_policy or {}))
        http_enabled = (
            http_namespace_enabled
            and bool(worker_policy.enabled)
            and bool(worker_policy.brokered_io.http)
            and str(worker_policy.network.mode or "").strip().lower() == "brokered_only"
        )
        registry = HostApiRegistry(
            contract="hosting.workflow_python.node.host_api.v1",
            request_id=str(dict(request or {}).get("request_id") or ""),
            roots={
                "readable": sorted(set(input_roots + output_roots)),
                "writable": output_roots,
            },
            policy={
                "artifact_fs": artifact_fs_enabled,
                "namespaces": {
                    "fs": artifact_fs_enabled,
                    "http": http_enabled,
                    "subprocess": False,
                    "custom_functions": False,
                },
                "http": http_enabled,
                "subprocess": False,
                "custom_functions": False,
            },
        )

        def _root_and_target(args: Dict[str, Any], *, write: bool = False) -> tuple[str, Path, Path, Any]:
            root_id = str(args.get("root_id") or "").strip()
            relative_path = args.get("relative_path")
            root = self._workflow_python_node_host_root(artifact_context, root_id=root_id, write=write)
            target = self._workflow_python_node_host_path(root, relative_path)
            return root_id, root, target, relative_path

        if artifact_fs_enabled:
            registry.register(
                "fs.list",
                namespace="fs",
                description="List direct children under a declared artifact input or output root.",
                args_schema=fs_root_args_schema(),
                result_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "root_id": {"type": "string"},
                        "relative_path": {"type": "string"},
                        "entries": {"type": "array", "items": {"type": "object"}},
                    },
                },
                permissions=["artifact.read"],
                handler=lambda args: _fs_list(args),
            )

            registry.register(
                "fs.read_text",
                namespace="fs",
                description="Read UTF text from a declared artifact input or output root.",
                args_schema=fs_root_args_schema(text=True),
                result_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "root_id": {"type": "string"},
                        "relative_path": {"type": "string"},
                        "text": {"type": "string"},
                        "encoding": {"type": "string"},
                    },
                },
                permissions=["artifact.read"],
                handler=lambda args: _fs_read_text(args),
            )

            registry.register(
                "fs.write_text",
                namespace="fs",
                description="Write UTF text under a declared artifact output root.",
                args_schema=fs_write_text_args_schema(),
                result_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "root_id": {"type": "string"},
                        "relative_path": {"type": "string"},
                        "bytes": {"type": "integer"},
                        "encoding": {"type": "string"},
                    },
                },
                permissions=["artifact.write"],
                handler=lambda args: _fs_write_text(args),
            )

            registry.register(
                "fs.mkdir",
                namespace="fs",
                description="Create a directory under a declared artifact output root.",
                args_schema=fs_root_args_schema(mkdir=True),
                result_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "root_id": {"type": "string"},
                        "relative_path": {"type": "string"},
                    },
                },
                permissions=["artifact.write"],
                handler=lambda args: _fs_mkdir(args),
            )

            registry.register(
                "fs.stat",
                namespace="fs",
                description="Return metadata for a path under a declared artifact input or output root.",
                args_schema=fs_root_args_schema(),
                result_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "root_id": {"type": "string"},
                        "relative_path": {"type": "string"},
                        "exists": {"type": "boolean"},
                        "type": {"type": "string"},
                        "size": {"type": ["integer", "null"]},
                        "mtime": {"type": "number"},
                    },
                },
                permissions=["artifact.read"],
                handler=lambda args: _fs_stat(args),
            )

        if http_enabled:
            registry.register(
                "http.fetch",
                namespace="http",
                description="Fetch an HTTP(S) URL through the host broker using this request's sandbox network policy.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "method": {"type": "string", "default": "GET"},
                        "headers": {"type": "object", "additionalProperties": {"type": "string"}},
                        "body_b64": {"type": "string", "default": ""},
                        "timeout_seconds": {"type": "number", "default": 30.0},
                        "max_response_bytes": {"type": "integer", "default": 1048576},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
                result_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "url": {"type": "string"},
                        "status_code": {"type": "integer"},
                        "headers": {"type": "object"},
                        "body_b64": {"type": "string"},
                        "body_size": {"type": "integer"},
                        "truncated": {"type": "boolean"},
                    },
                },
                permissions=["http.fetch"],
                handler=lambda args: _http_fetch(args),
            )

        def _fs_list(args: Dict[str, Any]) -> Dict[str, Any]:
            root_id, _root, target, relative_path = _root_and_target(args)
            if not target.exists():
                raise FileNotFoundError(str(target))
            if not target.is_dir():
                raise NotADirectoryError(str(target))
            return {
                "status": "ok",
                "root_id": root_id,
                "relative_path": str(relative_path or ""),
                "entries": [
                    {
                        "name": child_path.name,
                        "type": "dir" if child_path.is_dir() else "file",
                        "size": child_path.stat().st_size if child_path.is_file() else None,
                    }
                    for child_path in sorted(target.iterdir(), key=lambda item: item.name)
                ],
            }

        def _fs_read_text(args: Dict[str, Any]) -> Dict[str, Any]:
            root_id, _root, target, relative_path = _root_and_target(args)
            encoding = str(args.get("encoding") or "utf-8")
            return {
                "status": "ok",
                "root_id": root_id,
                "relative_path": str(relative_path or ""),
                "text": target.read_text(encoding=encoding),
                "encoding": encoding,
            }

        def _fs_write_text(args: Dict[str, Any]) -> Dict[str, Any]:
            root_id, _root, target, relative_path = _root_and_target(args, write=True)
            encoding = str(args.get("encoding") or "utf-8")
            text = str(args.get("text") or "")
            if bool(args.get("create_parents", True)):
                target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding=encoding)
            return {
                "status": "ok",
                "root_id": root_id,
                "relative_path": str(relative_path or ""),
                "bytes": len(text.encode(encoding, errors="replace")),
                "encoding": encoding,
            }

        def _fs_mkdir(args: Dict[str, Any]) -> Dict[str, Any]:
            root_id, _root, target, relative_path = _root_and_target(args, write=True)
            target.mkdir(parents=bool(args.get("parents", True)), exist_ok=bool(args.get("exist_ok", True)))
            return {"status": "ok", "root_id": root_id, "relative_path": str(relative_path or "")}

        def _fs_stat(args: Dict[str, Any]) -> Dict[str, Any]:
            root_id, _root, target, relative_path = _root_and_target(args)
            stat = target.stat()
            return {
                "status": "ok",
                "root_id": root_id,
                "relative_path": str(relative_path or ""),
                "exists": True,
                "type": "dir" if target.is_dir() else "file",
                "size": stat.st_size if target.is_file() else None,
                "mtime": stat.st_mtime,
            }

        def _http_fetch(args: Dict[str, Any]) -> Dict[str, Any]:
            headers = {
                str(key): str(value)
                for key, value in dict(args.get("headers") or {}).items()
                if str(key or "").strip()
            }
            out = BrokeredHttpClient(worker_policy).fetch(
                url=str(args.get("url") or ""),
                method=str(args.get("method") or "GET"),
                headers=headers,
                body_b64=str(args.get("body_b64") or ""),
                timeout_seconds=float(args.get("timeout_seconds") or 30.0),
                max_response_bytes=int(args.get("max_response_bytes") or 1024 * 1024),
            )
            return {"status": "ok", **dict(out or {})}

        def _dispatch(call: Dict[str, Any]) -> Dict[str, Any]:
            return registry.dispatch(dict(call or {}))

        return _dispatch

    def _workflow_python_node_artifact_error(
        self,
        *,
        request: Dict[str, Any],
        environment_key: str,
        engine_id: str,
        error: Exception,
    ) -> Dict[str, Any]:
        return self._workflow_python_node_response_from_execution(
            execution={
                "ok": False,
                "reason": "workflow_python_artifact_error",
                "detail": {"message": str(error)},
            },
            request=request,
            environment_key=environment_key,
            engine_id=engine_id,
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
        value = str(profile or "node").strip().lower() or "node"
        if value != "node":
            raise ValueError("workflow_js supports only profile='node'")
        return value

    def workflow_js_environment_spec(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        javascript_policy = {**dict(node or {}), **dict(javascript or {})}
        return self._workflow_js_runtime_base().environment_spec(
            profile=prof,
            environment_name=environment_name,
            javascript_policy=javascript_policy,
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
            env_row["workflow_profile"] = str(profile or "node").strip() or "node"
            row["environment"] = env_row
            capabilities = dict(row.get("capabilities") or {})
            capabilities.update(
                {
                    "workflow_js": True,
                    "workflow_js_profile": str(profile or "node").strip() or "node",
                    "environment_key": str(environment_key or "").strip() or None,
                }
            )
            row["capabilities"] = capabilities
            changed = True
        if changed:
            self._write_engines(rows)

    def workflow_js_default_engine_id(self, *, environment_key: str) -> str:
        key = str(environment_key or "").strip()
        return f"workflow-js-{key[:16]}" if key else "workflow-js-node"

    @staticmethod
    def _workflow_js_node_response_from_execution(
        *,
        execution: Dict[str, Any],
        request: Dict[str, Any],
        environment_key: str,
        engine_id: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        req = dict(request or {})
        result = dict(execution or {})
        ok = bool(result.get("ok", False))
        reason = str(result.get("reason") or "").strip()
        detail = dict(result.get("detail") or {}) if isinstance(result.get("detail"), dict) else {}
        limits = dict(req.get("limits") or {})
        logs = hosted_log_summary(
            stdout=str(result.get("stdout") or ""),
            stderr=str(result.get("stderr") or ""),
            max_bytes=int(limits.get("output_limit_bytes") or 4096),
        )
        status = "ok" if ok else ("canceled" if reason == "workflow_sandbox_canceled" else "error")
        artifact_rows = list(result.get("artifacts") or []) if isinstance(result.get("artifacts"), list) else []
        artifact_store = (
            {
                "status": "ok",
                "kind": "local",
                "reason": None,
                "message": "artifact refs were minted from host-provided workflow JavaScript output paths",
            }
            if artifact_rows
            else {
                "status": "unavailable",
                "reason": "artifact_store_no_refs",
                "message": "no host-minted artifact refs were created for this response",
            }
        )
        return {
            "status": status,
            "ok": ok,
            "profile": "node",
            "engine_id": str(engine_id or "").strip() or None,
            "environment_key": str(environment_key or "").strip() or None,
            "request_id": str(req.get("request_id") or "").strip() or None,
            "reason": None if ok else (reason or "workflow_js_node_execution_failed"),
            "error": None
            if ok
            else {
                "code": reason or "workflow_js_node_execution_failed",
                "message": str(detail.get("message") or reason or "workflow JavaScript node execution failed"),
                "detail": detail,
            },
            "output": result.get("output") if ok else None,
            "state_patch": dict(result.get("state_patch") or {}) or None,
            "artifacts": artifact_rows,
            "artifact_store": artifact_store,
            "progress": dict(result.get("progress") or {}) or None,
            "logs": logs,
            "metrics": dict(metrics or {}),
            "audit": {
                "package_id": str(req.get("package_id") or "").strip() or None,
                "workflow_id": str(req.get("workflow_id") or "").strip() or None,
                "package_source_digest": str(req.get("package_source_digest") or "").strip() or None,
                "module_sha256": str(req.get("module_sha256") or "").strip() or None,
                "provenance": dict(req.get("provenance") or {}),
                "runtime": {
                    **dict(result.get("runtime") or {}),
                    "runtime_kind": "workflow_js",
                    "profile": "node",
                    "engine": "quickjs",
                },
            },
        }

    @staticmethod
    def _workflow_js_node_validation_error(
        *,
        request: Dict[str, Any],
        environment_key: str,
        engine_id: str,
        reason: str,
        message: str,
    ) -> Dict[str, Any]:
        return WorkflowHelperMixin._workflow_js_node_response_from_execution(
            execution={"ok": False, "reason": reason, "detail": {"message": message}},
            request=request,
            environment_key=environment_key,
            engine_id=engine_id,
        )

    def ensure_workflow_js(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        engine_id: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        js_policy = {**dict(node or {}), **dict(javascript or {})}
        env = self.workflow_js_environment_spec(
            profile=prof,
            environment_name=environment_name,
            javascript=js_policy,
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
        pool = self._workflow_python_pool_registry().get_or_create(self._workflow_js_pool_key(derived_key), desired_capacity=capacity)
        pool.ensure_worker(lambda _key, cap: HostedWorkerSlot(engine_id=eid, environment_key=derived_key, capacity=cap, status="node_runtime"))
        return {"status": "ok", "outcome": "ready", "profile": prof, "engine_id": eid, "environment_key": derived_key, "environment": dict(env.get("environment") or {}), "workflow_pool": pool.resources()}

    def workflow_js_resources(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        env = self.workflow_js_environment_spec(profile=prof, environment_name=environment_name, node=node, javascript=javascript, sandbox_policy=sandbox_policy)
        derived_key = str(env.get("environment_key") or "").strip()
        registration_key = self._workflow_js_registration_environment_key(engine_id)
        requested_key = str(environment_key or "").strip()
        effective_key = requested_key or registration_key or derived_key
        if requested_key and registration_key and requested_key != registration_key:
            return {"status": "error", "reason": "environment_key_mismatch", "environment_key": requested_key, "registration_environment_key": registration_key}
        eid = str(engine_id or "").strip() or self.workflow_js_default_engine_id(environment_key=effective_key)
        pool = self._workflow_python_pool_registry().get(self._workflow_js_pool_key(effective_key))
        runtime_resources = self._workflow_js_node_runtime_registry().resources()
        return {
            "status": "ok",
            "profile": prof,
            "engine_id": eid,
            "environment_key": effective_key,
            "environment": dict(env.get("environment") or {}),
            "workflow_pool": pool.resources() if pool is not None else None,
            "node_runtime": runtime_resources,
            "workflow_js_capacity": int(dict(dict(pool.resources() if pool is not None else {}).get("metrics") or {}).get("desired_capacity") or 0),
            "workflow_js_active_calls": int(dict(dict(pool.resources() if pool is not None else {}).get("metrics") or {}).get("active_calls") or 0),
        }

    def set_workflow_js_capacity(
        self,
        *,
        profile: str = "node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        capacity: int,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        effective_key = str(environment_key or "").strip() or self._workflow_js_registration_environment_key(engine_id)
        eid = str(engine_id or "").strip() or self.workflow_js_default_engine_id(environment_key=effective_key)
        pool = self._workflow_python_pool_registry().get_or_create(self._workflow_js_pool_key(effective_key), desired_capacity=capacity)
        actual_capacity = pool.set_capacity(capacity)
        if effective_key:
            self._workflow_python_pool_registry().get_or_create(self._workflow_js_pool_key(effective_key), desired_capacity=capacity).set_capacity(actual_capacity)
        return {"status": "ok", "profile": prof, "engine_id": eid, "environment_key": effective_key or None, "capacity": actual_capacity, "workflow_pool": pool.resources()}

    def cancel_workflow_js_request(
        self,
        *,
        profile: str = "node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request_id: str,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        effective_key = str(environment_key or "").strip() or self._workflow_js_registration_environment_key(engine_id)
        eid = str(engine_id or "").strip() or self.workflow_js_default_engine_id(environment_key=effective_key)
        worker_cancel = self._workflow_js_node_runtime_registry().cancel(request_id)
        pool = self._workflow_python_pool_registry().get(self._workflow_js_pool_key(effective_key))
        pool_cancel = pool.cancel_request(request_id) if pool is not None else None
        return {
            "status": "ok",
            "profile": prof,
            "engine_id": eid,
            "environment_key": effective_key or None,
            "request_id": str(request_id or "").strip(),
            "canceled": bool(dict(worker_cancel or {}).get("canceled") or dict(pool_cancel or {}).get("status") == "ok"),
            "worker_cancel": dict(worker_cancel or {}),
            "workflow_pool_cancel": dict(pool_cancel or {}) if pool_cancel is not None else None,
        }

    def workflow_js_request_status(
        self,
        *,
        profile: str = "node",
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

    def execute_workflow_js(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        req = dict(request or {})
        js = {**dict(javascript or {}), **dict(req.get("javascript") or {})}
        ensured = self.ensure_workflow_js(
            profile=prof,
            environment_name=environment_name,
            environment_key=environment_key,
            node=dict(node or req.get("node") or {}),
            javascript=js,
            capacity=capacity,
            sandbox_policy=sandbox_policy,
            engine_id=engine_id,
        )
        if str(ensured.get("status") or "") != "ok":
            return ensured
        pool = self._workflow_python_pool_registry().get_or_create(
            self._workflow_js_pool_key(str(ensured.get("environment_key") or "")),
            desired_capacity=capacity,
        )
        lifecycle = HostedRequestLifecycle(
            request_id=str(req.get("request_id") or "").strip() or "workflow-js-sync",
            environment_key=str(ensured.get("environment_key") or ""),
            sandbox_kind="workflow_js",
            profile=prof,
            engine_id=str(ensured["engine_id"]),
            submitted_at=time.time(),
        )
        scheduled = pool.submit_request(
            lifecycle,
            factory=lambda _key, cap: HostedWorkerSlot(engine_id=str(ensured["engine_id"]), environment_key=str(ensured.get("environment_key") or ""), capacity=cap, status="node_runtime"),
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
        required = ["module_source", "module_sha256", "package_id", "workflow_id", "package_source_digest"]
        missing = [name for name in required if not str(req.get(name) or "").strip()]
        if missing:
            result = {"ok": False, "reason": "workflow_js_node_invalid_request", "detail": {"message": f"missing required fields: {', '.join(missing)}"}}
        else:
            artifact_context: Optional[Dict[str, Any]] = None
            if req.get("artifact_inputs") or req.get("artifact_outputs"):
                try:
                    artifact_context = self._workflow_python_prepare_node_artifacts(
                        request=req,
                        request_id=lifecycle.request_id,
                        sandbox_policy=sandbox_policy,
                    )
                except Exception as exc:
                    artifact_context = None
                    result = {"ok": False, "reason": "workflow_js_artifact_error", "detail": {"message": str(exc)}}
                else:
                    result = {}
            else:
                result = {}
            if not result:
                def _record_js_event(event_type: str, payload: Dict[str, Any]) -> None:
                    pool.record_stream_event(
                        lifecycle.request_id,
                        {"kind": event_type, "request_id": lifecycle.request_id, "timestamp_ms": int(time.time() * 1000), **dict(payload or {})},
                    )

                result = self._workflow_js_node_runtime_registry().execute(
                    {
                        **req,
                        "request_id": lifecycle.request_id,
                        "environment_key": str(ensured.get("environment_key") or ""),
                        "javascript": js,
                        "artifact_context": dict((artifact_context or {}).get("child_context") or {}),
                    },
                    python_executable=str(js.get("python_executable") or "").strip() or None,
                    on_event=_record_js_event,
                    host_dispatcher=self._workflow_python_node_host_dispatcher(
                        request={**req, "request_id": lifecycle.request_id},
                        artifact_context=artifact_context,
                        sandbox_policy=sandbox_policy,
                    ),
                )
                if artifact_context is not None and bool(result.get("ok", False)):
                    try:
                        result["artifacts"] = self._workflow_python_collect_node_artifacts(
                            artifact_context,
                            request_id=lifecycle.request_id,
                            runtime_artifacts=list(result.get("artifacts") or []),
                            sandbox_policy=sandbox_policy,
                        )
                    except Exception as exc:
                        result = {"ok": False, "reason": "workflow_js_artifact_error", "detail": {"message": str(exc)}, "artifacts": []}
                elif artifact_context is None:
                    result["artifacts"] = []
                if artifact_context is not None:
                    self._workflow_python_cleanup_node_artifacts(artifact_context, sandbox_policy=sandbox_policy)
        status = "ok" if bool(result.get("ok", False)) else "error"
        reason = str(result.get("reason") or "") or None
        if reason == "workflow_sandbox_timeout":
            status = "timeout"
        elif reason == "workflow_sandbox_canceled":
            status = "canceled"
        finished = pool.finish_request(
            lifecycle.request_id,
            status=status,
            reason=reason,
        )
        return self._workflow_js_node_response_from_execution(
            execution=result,
            request={**dict(request or {}), "request_id": lifecycle.request_id, "javascript": js},
            environment_key=str(ensured.get("environment_key") or ""),
            engine_id=str(ensured["engine_id"]),
            metrics={
                "workflow_pool": pool.resources(),
                "request": dict(finished.get("request") or lifecycle.to_dict()),
            },
        )

    def workflow_js_stream_open(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        req = dict(request or {})
        js = {**dict(javascript or {}), **dict(req.get("javascript") or {})}
        ensured = self.ensure_workflow_js(
            profile=prof,
            environment_name=environment_name,
            environment_key=environment_key,
            node=dict(node or req.get("node") or {}),
            javascript=js,
            capacity=capacity,
            sandbox_policy=sandbox_policy,
            engine_id=engine_id,
        )
        if str(ensured.get("status") or "") != "ok":
            return ensured
        request_id = str(req.get("request_id") or "").strip() or f"workflow-js-stream-{int(time.time() * 1000)}"
        limits = dict(req.get("limits") or {})
        try:
            max_events = max(1, min(int(limits.get("stream_max_events") or 256), 10000))
        except Exception:
            max_events = 256
        base = self._workflow_js_stream_base()
        opened = base.stream_open(
            environment_key=str(ensured.get("environment_key") or ""),
            request_id=request_id,
            profile=prof,
            desired_capacity=capacity,
            max_events=max_events,
            factory=lambda _key, cap: HostedWorkerSlot(
                engine_id=str(ensured["engine_id"]),
                environment_key=str(ensured.get("environment_key") or ""),
                capacity=cap,
                status="node_runtime",
            ),
        )
        if str(opened.get("status") or "") != "ok":
            return {**dict(opened or {}), "profile": prof, "environment_key": str(ensured.get("environment_key") or "")}
        thread = threading.Thread(
            target=self._workflow_js_run_node_stream,
            kwargs={
                "stream_id": str(opened.get("stream_id") or ""),
                "environment_key": str(ensured.get("environment_key") or ""),
                "engine_id": str(ensured["engine_id"]),
                "request": {**req, "request_id": request_id, "javascript": js},
                "sandbox_policy": sandbox_policy,
            },
            name=f"workflow-js-node-stream-{request_id}",
            daemon=True,
        )
        thread.start()
        return {
            **dict(opened or {}),
            "profile": prof,
            "engine_id": str(ensured["engine_id"]),
            "environment": dict(ensured.get("environment") or {}),
        }

    def _workflow_js_run_node_stream(
        self,
        *,
        stream_id: str,
        environment_key: str,
        engine_id: str,
        request: Dict[str, Any],
        sandbox_policy: Optional[Dict[str, Any]],
    ) -> None:
        base = self._workflow_js_stream_base()

        def _emit_js_event(event_type: str, payload: Dict[str, Any]) -> None:
            if event_type == "console":
                base.stream_emit(
                    stream_id=stream_id,
                    event_type="stdout",
                    payload={
                        "text": str(dict(payload or {}).get("message") or ""),
                        "level": str(dict(payload or {}).get("level") or "log"),
                    },
                )
                return
            if event_type == "host_call":
                base.stream_emit(stream_id=stream_id, event_type="host_call", payload=dict(payload or {}))
                return
            base.stream_emit(stream_id=stream_id, event_type=event_type, payload=dict(payload or {}))

        artifact_context: Optional[Dict[str, Any]] = None
        if request.get("artifact_inputs") or request.get("artifact_outputs"):
            try:
                artifact_context = self._workflow_python_prepare_node_artifacts(
                    request=request,
                    request_id=str(request.get("request_id") or ""),
                    sandbox_policy=sandbox_policy,
                )
            except Exception as exc:
                response = self._workflow_js_node_response_from_execution(
                    execution={"ok": False, "reason": "workflow_js_artifact_error", "detail": {"message": str(exc)}},
                    request=request,
                    environment_key=environment_key,
                    engine_id=engine_id,
                )
                base.stream_emit(stream_id=stream_id, event_type="error", payload={"error": dict(response.get("error") or {}), "response": response})
                base.stream_emit(stream_id=stream_id, event_type="done", payload={"status": "error", "reason": response.get("reason")})
                session = getattr(base, "_streams", {}).get(str(stream_id or "").strip())
                if session is not None:
                    session.closed = True
                base.finish_request(environment_key=environment_key, request_id=str(request.get("request_id") or ""), status="error", reason="workflow_js_artifact_error")
                return

        result = self._workflow_js_node_runtime_registry().execute(
            {
                **dict(request or {}),
                "environment_key": environment_key,
                "artifact_context": dict((artifact_context or {}).get("child_context") or {}),
            },
            python_executable=str(dict(request.get("javascript") or {}).get("python_executable") or "").strip() or None,
            on_event=_emit_js_event,
            host_dispatcher=self._workflow_python_node_host_dispatcher(
                request=dict(request or {}),
                artifact_context=artifact_context,
                sandbox_policy=sandbox_policy,
            ),
        )
        if artifact_context is not None and bool(result.get("ok", False)):
            try:
                result["artifacts"] = self._workflow_python_collect_node_artifacts(
                    artifact_context,
                    request_id=str(request.get("request_id") or ""),
                    runtime_artifacts=list(result.get("artifacts") or []),
                    sandbox_policy=sandbox_policy,
                )
            except Exception as exc:
                result = {
                    "ok": False,
                    "reason": "workflow_js_artifact_error",
                    "detail": {"message": str(exc)},
                    "stdout": str(result.get("stdout") or ""),
                    "stderr": str(result.get("stderr") or ""),
                    "artifacts": [],
                }
        else:
            result["artifacts"] = []
        if artifact_context is not None:
            self._workflow_python_cleanup_node_artifacts(artifact_context, sandbox_policy=sandbox_policy)

        status_snapshot = base.request_status(environment_key=environment_key, request_id=str(request.get("request_id") or ""))
        response = self._workflow_js_node_response_from_execution(
            execution=result,
            request=request,
            environment_key=environment_key,
            engine_id=engine_id,
            metrics={"request": dict(status_snapshot.get("request") or {})},
        )
        base.stream_emit(stream_id=stream_id, event_type="log", payload={"logs": dict(response.get("logs") or {})})
        logs = dict(response.get("logs") or {})
        if str(logs.get("stdout") or ""):
            base.stream_emit(stream_id=stream_id, event_type="stdout", payload={"text": str(logs.get("stdout") or ""), "truncated": bool(logs.get("stdout_truncated"))})
        if str(logs.get("stderr") or ""):
            base.stream_emit(stream_id=stream_id, event_type="stderr", payload={"text": str(logs.get("stderr") or ""), "truncated": bool(logs.get("stderr_truncated"))})
        if bool(response.get("ok", False)):
            if isinstance(response.get("progress"), dict):
                base.stream_emit(stream_id=stream_id, event_type="progress", payload=dict(response.get("progress") or {}))
            for artifact in list(response.get("artifacts") or []):
                if isinstance(artifact, dict):
                    base.stream_emit(stream_id=stream_id, event_type="artifact", payload=dict(artifact or {}))
            base.stream_emit(
                stream_id=stream_id,
                event_type="result",
                payload={
                    "output": response.get("output"),
                    "state_patch": response.get("state_patch"),
                    "artifacts": list(response.get("artifacts") or []),
                    "metrics": dict(response.get("metrics") or {}),
                },
            )
            base.stream_emit(stream_id=stream_id, event_type="done", payload={"status": "ok"})
            session = getattr(base, "_streams", {}).get(str(stream_id or "").strip())
            if session is not None:
                session.closed = True
            base.finish_request(environment_key=environment_key, request_id=str(request.get("request_id") or ""), status="ok")
            return
        if str(response.get("reason") or "") == "workflow_sandbox_canceled":
            session = getattr(base, "_streams", {}).get(str(stream_id or "").strip())
            if session is not None and bool(getattr(session, "closed", False)):
                base.finish_request(
                    environment_key=environment_key,
                    request_id=str(request.get("request_id") or ""),
                    status="canceled",
                    reason=str(response.get("reason") or "workflow_sandbox_canceled"),
                )
                return
            if session is not None and not bool(getattr(session, "canceled", False)):
                base.stream_emit(
                    stream_id=stream_id,
                    event_type="canceled",
                    payload={"request_id": str(request.get("request_id") or ""), "reason": "workflow_sandbox_canceled"},
                )
                session.canceled = True
            base.stream_emit(stream_id=stream_id, event_type="done", payload={"status": "canceled", "reason": response.get("reason")})
            if session is not None:
                session.closed = True
            base.finish_request(environment_key=environment_key, request_id=str(request.get("request_id") or ""), status="canceled", reason=str(response.get("reason") or "workflow_sandbox_canceled"))
            return
        base.stream_emit(stream_id=stream_id, event_type="error", payload={"error": dict(response.get("error") or {}), "response": response})
        base.stream_emit(stream_id=stream_id, event_type="done", payload={"status": "error", "reason": response.get("reason")})
        session = getattr(base, "_streams", {}).get(str(stream_id or "").strip())
        if session is not None:
            session.closed = True
        reason = str(response.get("reason") or "workflow_js_node_execution_failed")
        base.finish_request(
            environment_key=environment_key,
            request_id=str(request.get("request_id") or ""),
            status="timeout" if reason == "workflow_sandbox_timeout" else "error",
            reason=reason,
        )

    def workflow_js_event_subscribe(self, *, stream_id: str, max_items: int = 64) -> Dict[str, Any]:
        return dict(self._workflow_js_stream_base().event_subscribe(stream_id=stream_id, max_items=max_items))

    def workflow_js_stream_send(self, *, stream_id: str, message: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = self._workflow_js_stream_base()
        msg = dict(message or {})
        out = dict(base.stream_send(stream_id=stream_id, message=msg))
        if bool(out.get("accepted")) and str(msg.get("action") or "").strip() == "cancel":
            session = getattr(base, "_streams", {}).get(str(stream_id or "").strip())
            if session is not None:
                out["worker_cancel"] = self.cancel_workflow_js_request(
                    profile=str(getattr(session, "profile", "") or "node"),
                    environment_key=str(getattr(session, "environment_key", "") or ""),
                    request_id=str(getattr(session, "request_id", "") or ""),
                )
                if bool(dict(out.get("worker_cancel") or {}).get("canceled")) and not bool(getattr(session, "closed", False)):
                    base.stream_emit(stream_id=stream_id, event_type="done", payload={"status": "canceled", "reason": "workflow_sandbox_canceled"})
                    session.closed = True
        return out

    def workflow_js_stream_close(self, *, stream_id: str) -> Dict[str, Any]:
        return dict(self._workflow_js_stream_base().stream_close(stream_id=stream_id))

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
        if prof == "node" and str(environment_name or "") == "workflow-python-helper":
            environment_name = "workflow-python-node"
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
        if prof == "node":
            req = self._workflow_python_with_project_artifact_input(req)
        if prof == "node" and str(environment_name or "") == "workflow-python-helper":
            environment_name = "workflow-python-node"
        py = dict(req.get("python") or {})
        if environment_name:
            py.setdefault("environment_name", str(environment_name or "workflow-python-helper"))
        req["python"] = py
        if prof == "node":
            validation = validate_workflow_python_node_request(req)
            if str(validation.get("status") or "") != "ok":
                return self._workflow_python_node_response_from_execution(
                    execution={
                        "ok": False,
                        "reason": "workflow_python_node_invalid_request",
                        "detail": {"missing_request_fields": list(validation.get("missing") or [])},
                    },
                    request=req,
                    environment_key=str(environment_key or ""),
                    engine_id=str(engine_id or ""),
                )
            env = self.workflow_python_environment_spec(
                profile=prof,
                environment_name=str(py.get("environment_name") or environment_name or "workflow-python-node"),
                python=py,
                sandbox_policy=sandbox_policy,
            )
            derived_key = str(env.get("environment_key") or "").strip()
            requested_key = str(environment_key or "").strip()
            if requested_key and requested_key != derived_key:
                return {
                    "status": "error",
                    "ok": False,
                    "profile": prof,
                    "reason": "environment_key_mismatch",
                    "environment_key": requested_key,
                    "derived_environment_key": derived_key,
                }
            effective_key = requested_key or derived_key
            eid = str(engine_id or "").strip() or self.workflow_python_default_engine_id(environment_key=effective_key)
            dependency_error = self._workflow_python_node_dependency_environment_check(
                request=req,
                python=py,
                environment=dict(env.get("environment") or {}),
                environment_key=effective_key,
                engine_id=eid,
            )
            if dependency_error is not None:
                if str(dependency_error.get("status") or "") != "ok":
                    return dependency_error
                selected_runtime = dict(dependency_error.get("runtime") or {})
                if str(selected_runtime.get("python_executable") or "").strip():
                    py["python_executable"] = str(selected_runtime.get("python_executable") or "").strip()
                    req["python"] = py
            node_runtime_recycle = self._workflow_python_node_recycle_changed_environment(
                environment_name=str(py.get("environment_name") or environment_name or "workflow-python-node"),
                environment_key=effective_key,
            )
            pool = self._workflow_python_pool_registry().get_or_create(
                self._workflow_python_pool_key(effective_key),
                desired_capacity=capacity,
            )
            lifecycle = HostedRequestLifecycle(
                request_id=str(req.get("request_id") or "").strip() or "workflow-python-node-sync",
                environment_key=effective_key,
                sandbox_kind="workflow_python",
                profile=prof,
                engine_id=eid,
                submitted_at=time.time(),
            )
            scheduled = pool.submit_request(
                lifecycle,
                factory=lambda _key, cap: HostedWorkerSlot(
                    engine_id=eid,
                    environment_key=effective_key,
                    capacity=cap,
                    status="node_runtime",
                ),
            )
            if str(scheduled.get("status") or "") != "ok":
                return {
                    "status": "error",
                    "ok": False,
                    "profile": prof,
                    "engine_id": eid,
                    "environment_key": effective_key,
                    "reason": str(scheduled.get("reason") or "capacity_exceeded"),
                    "metrics": {"workflow_pool": pool.resources(), "request": dict(scheduled.get("request") or {})},
                }

            artifact_context: Optional[Dict[str, Any]] = None
            if req.get("artifact_inputs") or req.get("artifact_outputs"):
                try:
                    artifact_context = self._workflow_python_prepare_node_artifacts(
                        request=req,
                        request_id=lifecycle.request_id,
                        sandbox_policy=sandbox_policy,
                    )
                except Exception as exc:
                    finished = pool.finish_request(
                        lifecycle.request_id,
                        status="error",
                        reason="workflow_python_artifact_error",
                    )
                    response = self._workflow_python_node_artifact_error(
                        request={**dict(request or {}), "request_id": lifecycle.request_id, "python": py},
                        environment_key=effective_key,
                        engine_id=eid,
                        error=exc,
                    )
                    response["metrics"] = {
                        "workflow_pool": pool.resources(),
                        "request": dict(finished.get("request") or lifecycle.to_dict()),
                    }
                    return response

            def _record_node_event(event_type: str, payload: Dict[str, Any]) -> None:
                pool.record_stream_event(
                    lifecycle.request_id,
                    {"kind": event_type, "request_id": lifecycle.request_id, "timestamp_ms": int(time.time() * 1000), **dict(payload or {})},
                )

            result = self._workflow_python_node_runtime_registry().execute(
                {
                    **req,
                    "request_id": lifecycle.request_id,
                    "environment_key": effective_key,
                    "python": py,
                    "artifact_context": dict((artifact_context or {}).get("child_context") or {}),
                },
                python_executable=str(py.get("python_executable") or "").strip() or None,
                on_event=_record_node_event,
                host_dispatcher=self._workflow_python_node_host_dispatcher(
                    request={**req, "request_id": lifecycle.request_id},
                    artifact_context=artifact_context,
                    sandbox_policy=sandbox_policy,
                ),
                max_idle=int(pool.resources().get("metrics", {}).get("desired_capacity") or capacity or 1),
            )
            if artifact_context is not None and bool(result.get("ok", False)):
                try:
                    result["artifacts"] = self._workflow_python_collect_node_artifacts(
                        artifact_context,
                        request_id=lifecycle.request_id,
                        runtime_artifacts=list(result.get("artifacts") or []),
                        sandbox_policy=sandbox_policy,
                    )
                except Exception as exc:
                    result = {
                        "ok": False,
                        "reason": "workflow_python_artifact_error",
                        "detail": {"message": str(exc)},
                        "stdout": str(result.get("stdout") or ""),
                        "stderr": str(result.get("stderr") or ""),
                        "artifacts": [],
                    }
            else:
                result["artifacts"] = []
            if artifact_context is not None:
                self._workflow_python_cleanup_node_artifacts(artifact_context, sandbox_policy=sandbox_policy)
            status = "ok" if bool(result.get("ok", False)) else "error"
            reason = str(result.get("reason") or "") or None
            if reason == "workflow_sandbox_timeout":
                status = "timeout"
            elif reason == "workflow_sandbox_canceled":
                status = "canceled"
            output_bytes = None
            try:
                output_bytes = len(json.dumps(result.get("output"), ensure_ascii=False).encode("utf-8"))
            except Exception:
                output_bytes = None
            finished = pool.finish_request(
                lifecycle.request_id,
                status=status,
                reason=reason,
                output_bytes=output_bytes,
            )
            node_runtime_trim = self._workflow_python_node_runtime_registry().trim_idle(
                environment_key=effective_key,
                max_idle=int(pool.resources().get("metrics", {}).get("desired_capacity") or capacity or 1),
            )
            return self._workflow_python_node_response_from_execution(
                execution=result,
                request={**dict(request or {}), "request_id": lifecycle.request_id, "python": py},
                environment_key=effective_key,
                engine_id=eid,
                metrics={
                    "workflow_pool": pool.resources(),
                    "request": dict(finished.get("request") or lifecycle.to_dict()),
                    "node_runtime_recycle": node_runtime_recycle,
                    "node_runtime_trim": node_runtime_trim,
                },
            )
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
        try:
            reg = self.get_registration(str(ensured.get("engine_id") or ""))
            if reg:
                self._wait_for_worker_rpc_ready(reg, timeout_seconds=5.0, poll_interval_seconds=0.05)
        except Exception:
            pass
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
        metrics = {
            "workflow_pool": pool.resources(),
            "request": dict(finished.get("request") or lifecycle.to_dict()),
        }
        return {
            "status": "ok" if bool(result.get("ok", False)) else "error",
            "ok": bool(result.get("ok", False)),
            "profile": prof,
            "engine_id": str(ensured["engine_id"]),
            "environment_key": str(ensured.get("environment_key") or ""),
            "output": result.get("result"),
            "result": result,
            "metrics": metrics,
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
        if prof == "node" and str(environment_name or "") == "workflow-python-helper":
            environment_name = "workflow-python-node"
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
        if prof == "node":
            pool = self._workflow_python_pool_registry().get(self._workflow_python_pool_key(effective_key))
            node_runtime_recycle = self._workflow_python_node_runtime_registry().recycle_unhealthy_idle(
                environment_key=effective_key,
            )
            runtime_resources = self._workflow_python_node_runtime_registry().resources()
            processes = []
            total_cpu = 0.0
            total_mem = 0.0
            known_cpu = False
            known_mem = False
            snapshot_fn = getattr(self, "_process_resource_snapshot", None)
            active_ids = set()
            if pool is not None:
                for worker in list(pool.resources().get("metrics", {}).get("workers", []) or []):
                    for request_id in list(dict(worker or {}).get("active_request_ids") or []):
                        active_ids.add(str(request_id or "").strip())
            active_processes = []
            for proc in list(runtime_resources.get("processes") or []):
                row = dict(proc or {})
                request_id = str(row.get("request_id") or "").strip()
                if active_ids and request_id not in active_ids:
                    continue
                pid = int(row.get("pid") or 0)
                metrics: Dict[str, Any] = {}
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
                row["resources"] = metrics
                active_processes.append(row)
            idle_processes = [
                dict(row or {})
                for row in list(runtime_resources.get("idle_processes") or [])
                if str(dict(row or {}).get("runtime_key") or "").split("|", 3)[1:2] == [effective_key]
            ]
            processes = [*active_processes, *idle_processes]
            pool_resources = pool.resources() if pool is not None else None
            pool_metrics = dict(dict(pool_resources or {}).get("metrics") or {})
            active_request_ids = []
            for worker_row in list(pool_metrics.get("workers") or []):
                for item in list(dict(worker_row or {}).get("active_request_ids") or []):
                    value = str(item or "").strip()
                    if value and value not in active_request_ids:
                        active_request_ids.append(value)
            return {
                "status": "ok",
                "profile": prof,
                "engine_id": eid,
                "environment_key": effective_key,
                "environment": dict(env.get("environment") or {}),
                "workflow_pool": pool_resources,
                "node_runtime": {
                    **dict(runtime_resources or {}),
                    "recycle": node_runtime_recycle,
                    "processes": processes,
                    "cpu_percent": round(total_cpu, 1) if known_cpu else None,
                    "memory_mb": round(total_mem, 1) if known_mem else None,
                },
                "workflow_python_capacity": int(pool_metrics.get("desired_capacity") or 0),
                "workflow_python_active_calls": int(pool_metrics.get("active_calls") or 0),
                "workflow_python_available_slots": int(pool_metrics.get("available_slots") or 0),
                "workflow_python_active_request_ids": active_request_ids,
                "workflow_python_process_count": len(processes),
                "workflow_python_active_process_count": len([row for row in active_processes if bool(dict(row or {}).get("alive"))]),
                "workflow_python_idle_process_count": len([row for row in idle_processes if bool(dict(row or {}).get("alive"))]),
                "workflow_python_pids": [int(dict(row or {}).get("pid") or 0) for row in processes if int(dict(row or {}).get("pid") or 0) > 0],
                "workflow_python_processes": processes,
                "workflow_python_cpu_percent": round(total_cpu, 1) if known_cpu else None,
                "workflow_python_memory_mb": round(total_mem, 1) if known_mem else None,
            }
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
        effective_key = str(environment_key or "").strip() or self._workflow_python_registration_environment_key(engine_id)
        eid = str(engine_id or "").strip() or self.workflow_python_default_engine_id(environment_key=effective_key)
        if prof == "node":
            pool = self._workflow_python_pool_registry().get_or_create(
                self._workflow_python_pool_key(effective_key),
                desired_capacity=capacity,
            )
            actual_capacity = pool.set_capacity(capacity)
            node_runtime_trim = self._workflow_python_node_runtime_registry().trim_idle(
                environment_key=effective_key,
                max_idle=actual_capacity,
            )
            return {
                "status": "ok",
                "profile": prof,
                "engine_id": eid,
                "environment_key": effective_key or None,
                "capacity": actual_capacity,
                "workflow_pool": pool.resources(),
                "node_runtime_trim": node_runtime_trim,
            }
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
        effective_key = str(environment_key or "").strip() or self._workflow_python_registration_environment_key(engine_id)
        eid = str(engine_id or "").strip() or self.workflow_python_default_engine_id(environment_key=effective_key)
        if prof == "node":
            worker_cancel = self._workflow_python_node_runtime_registry().cancel(request_id)
            pool = self._workflow_python_pool_registry().get(self._workflow_python_pool_key(effective_key))
            pool_cancel = pool.cancel_request(request_id) if pool is not None else None
            return {
                "status": "ok",
                "profile": prof,
                "engine_id": eid,
                "environment_key": effective_key or None,
                "request_id": str(request_id or "").strip(),
                "canceled": bool(dict(worker_cancel or {}).get("canceled") or dict(pool_cancel or {}).get("status") == "ok"),
                "worker_cancel": dict(worker_cancel or {}),
                "workflow_pool_cancel": dict(pool_cancel or {}) if pool_cancel is not None else None,
            }
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
        if prof == "node":
            req = self._workflow_python_with_project_artifact_input(req)
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
        eid = str(engine_id or "").strip() or self.workflow_python_default_engine_id(environment_key=effective_key)
        if prof == "node":
            py.setdefault("environment_name", str(environment_name or "workflow-python-node"))
            validation = validate_workflow_python_node_request({**req, "request_id": request_id, "python": py})
            if str(validation.get("status") or "") != "ok":
                return self._workflow_python_node_response_from_execution(
                    execution={
                        "ok": False,
                        "reason": "workflow_python_node_invalid_request",
                        "detail": {"missing_request_fields": list(validation.get("missing") or [])},
                    },
                    request={**req, "request_id": request_id, "python": py},
                    environment_key=effective_key,
                    engine_id=eid,
                )
            dependency_error = self._workflow_python_node_dependency_environment_check(
                request={**req, "request_id": request_id},
                python=py,
                environment=dict(env.get("environment") or {}),
                environment_key=effective_key,
                engine_id=eid,
            )
            if dependency_error is not None:
                if str(dependency_error.get("status") or "") != "ok":
                    return dependency_error
                selected_runtime = dict(dependency_error.get("runtime") or {})
                if str(selected_runtime.get("python_executable") or "").strip():
                    py["python_executable"] = str(selected_runtime.get("python_executable") or "").strip()
            node_runtime_recycle = self._workflow_python_node_recycle_changed_environment(
                environment_name=str(py.get("environment_name") or environment_name or "workflow-python-node"),
                environment_key=effective_key,
            )
        else:
            node_runtime_recycle = {"status": "skipped", "reason": "non_node_profile", "stopped_count": 0}
        base = self._workflow_python_stream_base()
        limits = dict(req.get("limits") or {})
        try:
            max_events = max(1, min(int(limits.get("stream_max_events") or 256), 10000))
        except Exception:
            max_events = 256
        opened = base.stream_open(
            environment_key=effective_key,
            request_id=request_id,
            profile=prof,
            desired_capacity=capacity,
            max_events=max_events,
            factory=lambda _key, cap: self._workflow_python_worker_slot(
                engine_id=eid,
                environment_key=effective_key,
                capacity=cap,
            ),
        )
        if str(opened.get("status") or "") != "ok":
            return {**dict(opened or {}), "profile": prof, "environment_key": effective_key}
        if prof == "node":
            stream_id = str(opened.get("stream_id") or "")
            thread = threading.Thread(
                target=self._workflow_python_run_node_stream,
                kwargs={
                    "stream_id": stream_id,
                    "environment_key": effective_key,
                    "engine_id": eid,
                    "request": {**req, "request_id": request_id, "python": py},
                    "sandbox_policy": sandbox_policy,
                    "capacity": capacity,
                    "node_runtime_recycle": node_runtime_recycle,
                },
                name=f"workflow-python-node-stream-{request_id}",
                daemon=True,
            )
            thread.start()
        return {
            **dict(opened or {}),
            "profile": prof,
            "engine_id": eid,
            "environment_key": effective_key,
            "environment": dict(env.get("environment") or {}),
        }

    def _workflow_python_run_node_stream(
        self,
        *,
        stream_id: str,
        environment_key: str,
        engine_id: str,
        request: Dict[str, Any],
        sandbox_policy: Optional[Dict[str, Any]],
        capacity: int,
        node_runtime_recycle: Optional[Dict[str, Any]] = None,
    ) -> None:
        base = self._workflow_python_stream_base()
        def _emit_node_event(event_type: str, payload: Dict[str, Any]) -> None:
            base.stream_emit(stream_id=stream_id, event_type=event_type, payload=dict(payload or {}))

        artifact_context: Optional[Dict[str, Any]] = None
        if request.get("artifact_inputs") or request.get("artifact_outputs"):
            try:
                artifact_context = self._workflow_python_prepare_node_artifacts(
                    request=request,
                    request_id=str(request.get("request_id") or ""),
                    sandbox_policy=sandbox_policy,
                )
            except Exception as exc:
                response = self._workflow_python_node_artifact_error(
                    request=request,
                    environment_key=environment_key,
                    engine_id=engine_id,
                    error=exc,
                )
                base.stream_emit(stream_id=stream_id, event_type="error", payload={"error": dict(response.get("error") or {}), "response": response})
                base.stream_emit(stream_id=stream_id, event_type="done", payload={"status": "error", "reason": response.get("reason")})
                session = getattr(base, "_streams", {}).get(str(stream_id or "").strip())
                if session is not None:
                    session.closed = True
                base.finish_request(
                    environment_key=environment_key,
                    request_id=str(request.get("request_id") or ""),
                    status="error",
                    reason="workflow_python_artifact_error",
                )
                return

        result = self._workflow_python_node_runtime_registry().execute(
            {
                **dict(request or {}),
                "environment_key": environment_key,
                "artifact_context": dict((artifact_context or {}).get("child_context") or {}),
            },
            python_executable=str(dict(request.get("python") or {}).get("python_executable") or "").strip() or None,
            on_event=_emit_node_event,
            host_dispatcher=self._workflow_python_node_host_dispatcher(
                request=dict(request or {}),
                artifact_context=artifact_context,
                sandbox_policy=sandbox_policy,
            ),
            max_idle=capacity,
        )
        if artifact_context is not None and bool(result.get("ok", False)):
            try:
                result["artifacts"] = self._workflow_python_collect_node_artifacts(
                    artifact_context,
                    request_id=str(request.get("request_id") or ""),
                    runtime_artifacts=list(result.get("artifacts") or []),
                    sandbox_policy=sandbox_policy,
                )
            except Exception as exc:
                result = {
                    "ok": False,
                    "reason": "workflow_python_artifact_error",
                    "detail": {"message": str(exc)},
                    "stdout": str(result.get("stdout") or ""),
                    "stderr": str(result.get("stderr") or ""),
                    "artifacts": [],
                }
        else:
            result["artifacts"] = []
        if artifact_context is not None:
            self._workflow_python_cleanup_node_artifacts(artifact_context, sandbox_policy=sandbox_policy)
        status_snapshot = base.request_status(environment_key=environment_key, request_id=str(request.get("request_id") or ""))
        response = self._workflow_python_node_response_from_execution(
            execution=result,
            request=request,
            environment_key=environment_key,
            engine_id=engine_id,
            metrics={
                "request": dict(status_snapshot.get("request") or {}),
                "node_runtime_recycle": dict(node_runtime_recycle or {}),
            },
        )
        base.stream_emit(stream_id=stream_id, event_type="log", payload={"logs": dict(response.get("logs") or {})})
        logs = dict(response.get("logs") or {})
        if str(logs.get("stdout") or ""):
            base.stream_emit(stream_id=stream_id, event_type="stdout", payload={"text": str(logs.get("stdout") or ""), "truncated": bool(logs.get("stdout_truncated"))})
        if str(logs.get("stderr") or ""):
            base.stream_emit(stream_id=stream_id, event_type="stderr", payload={"text": str(logs.get("stderr") or ""), "truncated": bool(logs.get("stderr_truncated"))})
        if bool(response.get("ok", False)):
            if isinstance(response.get("progress"), dict):
                base.stream_emit(stream_id=stream_id, event_type="progress", payload=dict(response.get("progress") or {}))
            for artifact in list(response.get("artifacts") or []):
                if isinstance(artifact, dict):
                    base.stream_emit(stream_id=stream_id, event_type="artifact", payload=dict(artifact or {}))
            base.stream_emit(
                stream_id=stream_id,
                event_type="result",
                payload={
                    "output": response.get("output"),
                    "state_patch": response.get("state_patch"),
                    "artifacts": list(response.get("artifacts") or []),
                    "metrics": dict(response.get("metrics") or {}),
                },
            )
            base.stream_emit(stream_id=stream_id, event_type="done", payload={"status": "ok"})
            session = getattr(base, "_streams", {}).get(str(stream_id or "").strip())
            if session is not None:
                session.closed = True
            base.finish_request(environment_key=environment_key, request_id=str(request.get("request_id") or ""), status="ok")
            return
        if str(response.get("reason") or "") == "workflow_sandbox_canceled":
            session = getattr(base, "_streams", {}).get(str(stream_id or "").strip())
            if session is not None and bool(getattr(session, "closed", False)):
                base.finish_request(
                    environment_key=environment_key,
                    request_id=str(request.get("request_id") or ""),
                    status="canceled",
                    reason=str(response.get("reason") or "workflow_sandbox_canceled"),
                )
                return
            if session is not None and not bool(getattr(session, "canceled", False)):
                base.stream_emit(
                    stream_id=stream_id,
                    event_type="canceled",
                    payload={"request_id": str(request.get("request_id") or ""), "reason": "workflow_sandbox_canceled"},
                )
                session.canceled = True
            base.stream_emit(stream_id=stream_id, event_type="done", payload={"status": "canceled", "reason": response.get("reason")})
            session = getattr(base, "_streams", {}).get(str(stream_id or "").strip())
            if session is not None:
                session.closed = True
            base.finish_request(
                environment_key=environment_key,
                request_id=str(request.get("request_id") or ""),
                status="canceled",
                reason=str(response.get("reason") or "workflow_sandbox_canceled"),
            )
            return
        base.stream_emit(
            stream_id=stream_id,
            event_type="error",
            payload={"error": dict(response.get("error") or {}), "response": response},
        )
        base.stream_emit(stream_id=stream_id, event_type="done", payload={"status": "error", "reason": response.get("reason")})
        session = getattr(base, "_streams", {}).get(str(stream_id or "").strip())
        if session is not None:
            session.closed = True
        base.finish_request(
            environment_key=environment_key,
            request_id=str(request.get("request_id") or ""),
            status="error",
            reason=str(response.get("reason") or "workflow_python_node_execution_failed"),
        )

    def workflow_python_event_subscribe(self, *, stream_id: str, max_items: int = 64) -> Dict[str, Any]:
        return dict(self._workflow_python_stream_base().event_subscribe(stream_id=stream_id, max_items=max_items))

    def workflow_python_stream_send(self, *, stream_id: str, message: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = self._workflow_python_stream_base()
        msg = dict(message or {})
        out = dict(base.stream_send(stream_id=stream_id, message=msg))
        if bool(out.get("accepted")) and str(msg.get("action") or "").strip() == "cancel":
            session = getattr(base, "_streams", {}).get(str(stream_id or "").strip())
            if session is not None:
                out["worker_cancel"] = self.cancel_workflow_python_request(
                    profile=str(getattr(session, "profile", "") or "node"),
                    environment_key=str(getattr(session, "environment_key", "") or ""),
                    request_id=str(getattr(session, "request_id", "") or ""),
                )
                if bool(dict(out.get("worker_cancel") or {}).get("canceled")) and not bool(getattr(session, "closed", False)):
                    base.stream_emit(
                        stream_id=stream_id,
                        event_type="done",
                        payload={"status": "canceled", "reason": "workflow_sandbox_canceled"},
                    )
                    session.closed = True
        return out

    def workflow_python_stream_close(self, *, stream_id: str) -> Dict[str, Any]:
        return dict(self._workflow_python_stream_base().stream_close(stream_id=stream_id))

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
