from __future__ import annotations

import sys
import os
import json
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..operation_contract import (
    HostedExecutionKind,
    HostedOperationLifecycle,
    HostedOperationSelector,
    hosted_execution_fingerprint,
)
from ..sandbox.python_runtime import HostedPythonRuntimeBase, HostedPythonRuntimeManager
from ..sandbox.js_runtime import HostedJsRuntimeBase
from ..sandbox.artifacts import HostedArtifactManager, artifact_safe_name
from ..sandbox.host_capabilities import (
    HostCapabilityBroker,
    HostCapabilityDescriptor,
    HostCapabilityMethod,
    HostCapabilityProviderCall,
    HostCapabilityProviderUnavailable,
    HostCapabilityProviderRef,
    HostCapabilitySession,
    default_group_path,
)
from ..sandbox.host_api import HostApiRegistry
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

WORKFLOW_ACTION_MANIFEST_CONTRACT = "hosting.sandbox.action_manifest.v1"
WORKFLOW_ACTION_DISCOVERY_CONTRACT = "hosting.sandbox.action_discovery.v1"


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

    def _append_host_capability_audit_event(self, event: Dict[str, Any]) -> None:
        control = self._read_control()
        rows = list(control.get("host_capability_audit_events") or [])
        row = dict(event or {})
        context = dict(row.get("context") or {})
        provider = dict(row.get("provider") or {})
        rows.append(
            {
                "schema_version": 1,
                "event_id": secrets.token_urlsafe(10),
                "timestamp": time.time(),
                "event_type": str(row.get("event_type") or "host_capability_event"),
                "result": str(row.get("result") or "") or None,
                "reason": str(row.get("reason") or "") or None,
                "approval_id": str(row.get("approval_id") or "") or None,
                "call_id": str(row.get("call_id") or "") or None,
                "host_call_id": str(row.get("host_call_id") or "") or None,
                "provider_call_id": str(row.get("provider_call_id") or "") or None,
                "method": str(row.get("method") or "") or None,
                "request_id": str(context.get("request_id") or row.get("request_id") or "") or None,
                "workflow_id": str(context.get("workflow_id") or row.get("workflow_id") or "") or None,
                "instance_id": str(context.get("instance_id") or row.get("instance_id") or "") or None,
                "node_id": str(context.get("node_id") or row.get("node_id") or "") or None,
                "cursor_id": str(context.get("cursor_id") or row.get("cursor_id") or "") or None,
                "context_id": str(context.get("context_id") or row.get("context_id") or "") or None,
                "branch_id": str(context.get("branch_id") or row.get("branch_id") or "") or None,
                "session_tree_id": str(context.get("session_tree_id") or row.get("session_tree_id") or "") or None,
                "actor": str(context.get("actor") or row.get("actor") or "") or None,
                "package_id": str(context.get("package_id") or "") or None,
                "provider_id": str(provider.get("provider_id") or row.get("provider_id") or "") or None,
                "provider": provider,
                "approval": dict(row.get("approval") or {}),
                "argument_keys": list(row.get("argument_keys") or []),
                "argument_preview": dict(row.get("argument_preview") or {}),
                "decision": dict(row.get("decision") or {}),
            }
        )
        if len(rows) > 500:
            rows = rows[-500:]
        control["host_capability_audit_events"] = rows
        self._write_control(control)

    def host_capability_audit_list(
        self,
        *,
        workflow_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        request_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        method: Optional[str] = None,
        approval_id: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        control = self._read_control()
        rows = [dict(row or {}) for row in list(control.get("host_capability_audit_events") or []) if isinstance(row, dict)]

        def _matches(row: Dict[str, Any]) -> bool:
            if workflow_id is not None and str(row.get("workflow_id") or "") != str(workflow_id or ""):
                return False
            if instance_id is not None and str(row.get("instance_id") or "") != str(instance_id or ""):
                return False
            if request_id is not None and str(row.get("request_id") or "") != str(request_id or ""):
                return False
            if provider_id is not None and str(row.get("provider_id") or dict(row.get("provider") or {}).get("provider_id") or "") != str(provider_id or ""):
                return False
            if method is not None and str(row.get("method") or "") != str(method or ""):
                return False
            if approval_id is not None and str(row.get("approval_id") or "") != str(approval_id or ""):
                return False
            ts = float(row.get("timestamp") or 0.0)
            if since is not None and ts < float(since):
                return False
            if until is not None and ts > float(until):
                return False
            return True

        filtered = [row for row in rows if _matches(row)]
        filtered.sort(key=lambda item: float(item.get("timestamp") or 0.0), reverse=True)
        start = max(0, int(offset or 0))
        count = max(1, min(int(limit or 100), 1000))
        return {
            "status": "ok",
            "events": filtered[start : start + count],
            "count": len(filtered[start : start + count]),
            "total": len(filtered),
            "limit": count,
            "offset": start,
        }

    @staticmethod
    def _host_capability_sessions_for_broker(sessions: Optional[list[HostCapabilitySession]]) -> list[HostCapabilitySession]:
        return [session for session in list(sessions or []) if isinstance(session, HostCapabilitySession)]

    @staticmethod
    def _toolbox_tool_name_for_host_capability(session: HostCapabilitySession, call: HostCapabilityProviderCall) -> str:
        method = dict(session.methods or {}).get(call.method)
        descriptor = method.descriptor if method is not None else None
        metadata = dict(getattr(descriptor, "metadata", {}) or {}) if descriptor is not None else {}
        toolbox = dict(metadata.get("toolbox") or {})
        return str(toolbox.get("tool_name") or call.method.split(".", 1)[-1] or "").strip()

    @staticmethod
    def _toolbox_provider_result(payload: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(payload or {})
        if str(row.get("contract") or "") == "hosting.operation_status":
            if str(row.get("lifecycle") or "") != "terminal_success":
                raise HostCapabilityProviderUnavailable(
                    detail={
                        "reason": str(row.get("reason") or row.get("lifecycle") or "toolbox_operation_failed"),
                        "operation": dict(row.get("operation") or {}),
                    }
                )
            row = dict(row.get("result") or {})
        tool_call = dict(row.get("tool_call") or {})
        if "result" in tool_call:
            raw = tool_call.get("result")
            if isinstance(raw, dict):
                return dict(raw)
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except Exception:
                    return {"result": raw}
                if isinstance(parsed, dict):
                    return dict(parsed)
                return {"result": parsed}
        if isinstance(row.get("result"), dict):
            return dict(row.get("result") or {})
        return row

    @staticmethod
    def _host_capability_client_session_binding(session: HostCapabilitySession) -> Dict[str, Any]:
        binding = dict(session.binding or {})
        callback_binding = dict(binding.get("callback_binding") or {}) if isinstance(binding.get("callback_binding"), dict) else {}
        if callback_binding:
            return callback_binding
        if {"address", "session_token"}.issubset(binding.keys()):
            return binding
        return {}

    def _host_capability_client_session_provider_response(
        self,
        session: HostCapabilitySession,
        call: HostCapabilityProviderCall,
    ) -> Dict[str, Any]:
        binding = dict(session.binding or {})
        callback = binding.get("callback")
        if callable(callback):
            from ..callable_surface import bind_host_capability_provider_callback

            return bind_host_capability_provider_callback(callback)(call.to_dict())

        callback_binding = self._host_capability_client_session_binding(session)
        if not callback_binding:
            raise HostCapabilityProviderUnavailable(
                detail={"provider_id": session.session_id, "provider_kind": session.provider_kind, "reason": "callback_binding_missing"}
            )
        try:
            from ..callable_surface import HOST_CAPABILITY_PROVIDER_CALLBACK_NAME
            from ..toolbox_executor_ipc import _invoke_callback_binding

            response = _invoke_callback_binding(
                callback_binding,
                callback_name=str(binding.get("callback_name") or HOST_CAPABILITY_PROVIDER_CALLBACK_NAME),
                payload=call.to_dict(),
                context=call.context.to_dict(),
            )
        except RuntimeError as exc:
            raise HostCapabilityProviderUnavailable(
                detail={
                    "provider_id": session.session_id,
                    "provider_kind": session.provider_kind,
                    "reason": str(exc).split(":", 1)[0] or "callback_invoke_failed",
                }
            ) from exc
        result = response.get("result")
        if isinstance(result, dict) and result.get("provider_call_id"):
            return dict(result)
        return {
            "status": "ok",
            "provider_call_id": call.provider_call_id,
            "result": dict(result or {}) if isinstance(result, dict) else {"result": result},
        }

    def _host_capability_service_broker_response(
        self,
        session: HostCapabilitySession,
        call: HostCapabilityProviderCall,
    ) -> Dict[str, Any]:
        from ..sandbox.service_broker_registry import invoke_service_broker_method

        engine_id = str(call.context.engine_id or dict(session.binding or {}).get("engine_id") or "").strip()
        if not engine_id:
            raise HostCapabilityProviderUnavailable(
                detail={"provider_id": session.session_id, "provider_kind": session.provider_kind, "reason": "engine_id_required"}
            )
        result = invoke_service_broker_method(
            self,
            engine_id=engine_id,
            method=call.method,
            arguments=dict(call.arguments or {}),
            callback_context=dict(dict(call.arguments or {}).get("callback_context") or {})
            if isinstance(dict(call.arguments or {}).get("callback_context"), dict) else None,
        )
        return {
            "status": "ok",
            "provider_call_id": call.provider_call_id,
            "result": dict(result or {}),
        }

    def _host_capability_provider_invoker(self, session: HostCapabilitySession, call: HostCapabilityProviderCall) -> Dict[str, Any]:
        provider_kind = str(session.provider_kind or "").strip()
        if provider_kind == "client_session":
            return self._host_capability_client_session_provider_response(session, call)
        if provider_kind == "service_broker":
            return self._host_capability_service_broker_response(session, call)
        if provider_kind != "toolbox_session":
            raise HostCapabilityProviderUnavailable(
                detail={"provider_id": session.session_id, "provider_kind": session.provider_kind}
            )
        binding = dict(session.binding or {})
        tool_name = self._toolbox_tool_name_for_host_capability(session, call)
        if not tool_name:
            raise ValueError("toolbox_host_capability_tool_name_required")
        timeout_seconds = float(binding.get("timeout_seconds") or binding.get("provider_timeout_seconds") or 30.0)
        out = self.toolbox_execute(
            engine_id=str(binding.get("engine_id") or "").strip(),
            toolbox_id=str(binding.get("toolbox_id") or "").strip(),
            execution_request_id=f"hostcap:{call.provider_call_id}",
            tool_call={
                "id": call.provider_call_id,
                "name": tool_name,
                "arguments": dict(call.arguments or {}),
            },
            timeout_seconds=timeout_seconds,
            tools_view=dict(binding.get("tools_view") or {}) if isinstance(binding.get("tools_view"), dict) else None,
            callback_binding=dict(binding.get("callback_binding") or {}) if isinstance(binding.get("callback_binding"), dict) else None,
        )
        return {
            "status": "ok",
            "provider_call_id": call.provider_call_id,
            "result": self._toolbox_provider_result(dict(out or {})),
        }

    def _host_capability_approval_requester_from_binding(self, binding: Optional[Dict[str, Any]]) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
        row = dict(binding or {})
        callback_binding = dict(row.get("callback_binding") or row)
        if not callback_binding:
            return None

        def _request_approval(payload: Dict[str, Any]) -> Dict[str, Any]:
            from ..callable_surface import HOST_CAPABILITY_APPROVAL_CALLBACK_NAME
            from ..toolbox_executor_ipc import _invoke_callback_binding

            response = _invoke_callback_binding(
                callback_binding,
                callback_name=HOST_CAPABILITY_APPROVAL_CALLBACK_NAME,
                payload=dict(payload or {}),
                context=dict(dict(payload or {}).get("context") or {}),
            )
            return dict(response.get("result") or response or {})

        return _request_approval

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
        artifact_recovery = dict(result.get("artifact_recovery") or {}) if isinstance(result.get("artifact_recovery"), dict) else None
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
            "artifact_recovery": artifact_recovery,
            "progress": dict(result.get("progress") or {}) or None,
            "logs": logs,
            "metrics": dict(metrics or {}),
            "audit": {
                "package_id": str(normalized.get("package_id") or "").strip() or None,
                "workflow_id": str(normalized.get("workflow_id") or "").strip() or None,
                "package_source_digest": str(normalized.get("package_source_digest") or "").strip() or None,
                "module_sha256": str(normalized.get("module_sha256") or "").strip() or None,
                "provenance": dict(normalized.get("provenance") or {}),
                "action": dict(dict(request or {}).get("_workflow_action_context") or {}) or None,
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
        mode = str(
            req.get("execution_mode")
            or dict(req.get("python") or {}).get("execution_mode")
            or dict(req.get("javascript") or {}).get("execution_mode")
            or ""
        ).strip().lower()
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

    @staticmethod
    def _workflow_action_raw_manifest(request: Dict[str, Any]) -> Dict[str, Any]:
        req = dict(request or {})
        raw = req.get("action_manifest")
        if raw is None:
            raw = req.get("actions")
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, list):
            return {"actions": list(raw)}
        return {}

    @staticmethod
    def _workflow_action_entrypoint_from_request(request: Dict[str, Any], *, runtime: str) -> Dict[str, Any]:
        req = dict(request or {})
        mode = str(req.get("execution_mode") or dict(req.get(runtime) or {}).get("execution_mode") or "").strip().lower()
        project = dict(req.get("project") or {})
        if mode == "project" or project:
            return {
                "kind": "project",
                "module": str(project.get("entrypoint") or project.get("module") or "").strip(),
                "callable": str(project.get("callable") or project.get("function") or req.get("export_name") or req.get("operation") or "run").strip() or "run",
            }
        if mode == "snippet":
            return {"kind": "snippet"}
        export_name = str(req.get("export_name") or req.get("operation") or "run").strip() or "run"
        return {"kind": "export", "export_name": export_name}

    @staticmethod
    def _workflow_action_rows(raw: Dict[str, Any]) -> list[Dict[str, Any]]:
        actions = raw.get("actions")
        if isinstance(actions, dict):
            return [{**dict(value or {}), "name": str(key)} for key, value in actions.items() if isinstance(value, dict)]
        return [dict(row or {}) for row in list(actions or []) if isinstance(row, dict)]

    @classmethod
    def _workflow_normalize_action_manifest(cls, request: Dict[str, Any], *, runtime: str) -> Dict[str, Any]:
        req = dict(request or {})
        raw = cls._workflow_action_raw_manifest(req)
        rows = cls._workflow_action_rows(raw)
        default_action = str(raw.get("default_action") or raw.get("default") or req.get("default_action") or "run").strip() or "run"
        if not rows:
            rows = [{"name": default_action, "title": "Run", "entrypoint": cls._workflow_action_entrypoint_from_request(req, runtime=runtime)}]
        actions: list[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            name = str(row.get("name") or row.get("id") or row.get("action") or "").strip()
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            entrypoint = dict(row.get("entrypoint") or {})
            if not entrypoint:
                if row.get("project") and isinstance(row.get("project"), dict):
                    project = dict(row.get("project") or {})
                    entrypoint = {
                        "kind": "project",
                        "module": str(project.get("entrypoint") or project.get("module") or "").strip(),
                        "callable": str(project.get("callable") or project.get("function") or row.get("callable") or "run").strip() or "run",
                    }
                elif row.get("operation"):
                    entrypoint = {"kind": "export", "export_name": str(row.get("operation") or "").strip(), "operation": str(row.get("operation") or "").strip()}
                elif row.get("export_name") or row.get("callable"):
                    entrypoint = {"kind": "export", "export_name": str(row.get("export_name") or row.get("callable") or "").strip()}
                else:
                    entrypoint = {"kind": "export", "export_name": name}
            action = {
                "name": name,
                "title": str(row.get("title") or row.get("label") or name).strip() or name,
                "description": str(row.get("description") or "").strip(),
                "allowed": bool(row.get("allowed", True)),
                "advertised": bool(row.get("advertised", True)),
                "hidden_allowed": bool(row.get("hidden_allowed", False)),
                "disabled": bool(row.get("disabled", False)),
                "gated": bool(row.get("gated", False)),
                "entrypoint": entrypoint,
                "input_schema": dict(row.get("input_schema") or row.get("args_schema") or row.get("parameters") or {}),
                "result_schema": dict(row.get("result_schema") or row.get("returns_schema") or {}),
                "approval": dict(row.get("approval") or row.get("approval_policy") or {}),
                "permissions": list(row.get("permissions") or []),
                "metadata": dict(row.get("metadata") or {}),
            }
            actions.append(action)
        if not any(row["name"] == default_action for row in actions):
            default_action = actions[0]["name"] if actions else "run"
        return {
            "status": "ok",
            "contract": WORKFLOW_ACTION_MANIFEST_CONTRACT,
            "runtime": str(runtime or "").strip(),
            "default_action": default_action,
            "actions": actions,
        }

    @classmethod
    def _workflow_action_manifest_card_view(cls, request: Dict[str, Any], *, runtime: str, include_hidden: bool = False) -> Dict[str, Any]:
        manifest = cls._workflow_normalize_action_manifest(request, runtime=runtime)
        actions = []
        for action in list(manifest.get("actions") or []):
            if not bool(action.get("advertised", True)):
                continue
            if bool(action.get("hidden_allowed", False)) and not bool(include_hidden):
                continue
            actions.append(dict(action))
        return {**manifest, "actions": actions, "count": len(actions), "card_facing": True}

    @classmethod
    def _workflow_selected_action_name(cls, request: Dict[str, Any]) -> str:
        req = dict(request or {})
        selected = req.get("action_name")
        if selected is None:
            selected = req.get("action")
        if isinstance(selected, dict):
            selected = selected.get("name") or selected.get("id")
        return str(selected or "").strip()

    @classmethod
    def _workflow_request_with_action(cls, request: Dict[str, Any], *, runtime: str) -> Dict[str, Any]:
        req = dict(request or {})
        name = cls._workflow_selected_action_name(req)
        if not name:
            return req
        manifest = cls._workflow_normalize_action_manifest(req, runtime=runtime)
        action = next((dict(row or {}) for row in list(manifest.get("actions") or []) if str(row.get("name") or "") == name), None)
        if action is None:
            return {
                **req,
                "_workflow_action_error": {
                    "reason": "workflow_action_not_found",
                    "detail": {"action_name": name, "available_actions": [str(row.get("name") or "") for row in list(manifest.get("actions") or [])]},
                },
            }
        if not bool(action.get("allowed", True)) or bool(action.get("disabled", False)):
            return {
                **req,
                "_workflow_action_error": {
                    "reason": "workflow_action_not_available",
                    "detail": {"action_name": name, "disabled": bool(action.get("disabled", False)), "allowed": bool(action.get("allowed", True))},
                },
            }
        entrypoint = dict(action.get("entrypoint") or {})
        routed = dict(req)
        routed["_workflow_action_context"] = {
            "name": name,
            "manifest_contract": WORKFLOW_ACTION_MANIFEST_CONTRACT,
            "entrypoint": entrypoint,
        }
        kind = str(entrypoint.get("kind") or entrypoint.get("type") or "").strip().lower()
        if kind == "snippet":
            routed["execution_mode"] = "snippet"
        elif kind == "project":
            project = dict(routed.get("project") or {})
            module_name = str(entrypoint.get("module") or entrypoint.get("entrypoint") or "").strip()
            callable_name = str(entrypoint.get("callable") or entrypoint.get("function") or entrypoint.get("export_name") or "run").strip() or "run"
            if module_name:
                project["entrypoint"] = module_name
            project["callable"] = callable_name
            routed["project"] = project
            routed["execution_mode"] = "project"
            routed["export_name"] = callable_name
        else:
            export_name = str(entrypoint.get("export_name") or entrypoint.get("operation") or entrypoint.get("callable") or name).strip() or name
            routed["export_name"] = export_name
            if entrypoint.get("operation"):
                routed["operation"] = str(entrypoint.get("operation") or "").strip()
        return routed

    @staticmethod
    def _workflow_action_discovery_raw(request: Dict[str, Any]) -> Dict[str, Any]:
        req = dict(request or {})
        for key in ("action_discovery", "dynamic_action_discovery", "dynamic_actions"):
            raw = req.get(key)
            if isinstance(raw, dict):
                return dict(raw)
        return {}

    @classmethod
    def _workflow_action_discovery_enabled(cls, request: Dict[str, Any], dynamic: bool) -> bool:
        return bool(dynamic) or bool(cls._workflow_action_discovery_raw(request))

    @classmethod
    def _workflow_request_with_action_discovery(cls, request: Dict[str, Any], *, runtime: str) -> Dict[str, Any]:
        req = dict(request or {})
        discovery = cls._workflow_action_discovery_raw(req)
        entrypoint = dict(discovery.get("entrypoint") or {})
        if not entrypoint:
            entrypoint = {
                "kind": "export",
                "export_name": str(discovery.get("export_name") or discovery.get("callable") or "describe_actions").strip()
                or "describe_actions",
            }

        routed = dict(req)
        for key in ("action", "action_name", "_workflow_action_context", "_workflow_action_error"):
            routed.pop(key, None)

        if "payload" in discovery:
            routed["payload"] = discovery.get("payload")
        elif "discovery_payload" in routed:
            routed["payload"] = routed.get("discovery_payload")

        routed["_workflow_action_discovery_context"] = {
            "contract": WORKFLOW_ACTION_DISCOVERY_CONTRACT,
            "entrypoint": entrypoint,
        }
        kind = str(entrypoint.get("kind") or entrypoint.get("type") or "export").strip().lower()
        if kind == "snippet":
            routed["execution_mode"] = "snippet"
        elif kind == "project":
            project = dict(routed.get("project") or {})
            module_name = str(entrypoint.get("module") or entrypoint.get("entrypoint") or "").strip()
            callable_name = str(entrypoint.get("callable") or entrypoint.get("function") or entrypoint.get("export_name") or "describe_actions").strip()
            if module_name:
                project["entrypoint"] = module_name
            project["callable"] = callable_name or "describe_actions"
            routed["project"] = project
            routed["execution_mode"] = "project"
            routed["export_name"] = project["callable"]
        else:
            export_name = str(
                entrypoint.get("export_name")
                or entrypoint.get("operation")
                or entrypoint.get("callable")
                or "describe_actions"
            ).strip() or "describe_actions"
            routed["export_name"] = export_name
            if entrypoint.get("operation"):
                routed["operation"] = str(entrypoint.get("operation") or "").strip()
        return routed

    @classmethod
    def _workflow_action_manifest_from_discovery_response(
        cls,
        response: Dict[str, Any],
        *,
        request: Dict[str, Any],
        runtime: str,
        include_hidden: bool = False,
    ) -> Dict[str, Any]:
        resp = dict(response or {})
        if not bool(resp.get("ok")):
            return {
                "status": "error",
                "ok": False,
                "contract": WORKFLOW_ACTION_MANIFEST_CONTRACT,
                "runtime": str(runtime or "").strip(),
                "dynamic": True,
                "reason": str(resp.get("reason") or "workflow_action_discovery_failed"),
                "error": dict(resp.get("error") or {}) if isinstance(resp.get("error"), dict) else {},
                "discovery": {
                    "contract": WORKFLOW_ACTION_DISCOVERY_CONTRACT,
                    "status": "error",
                    "request_id": resp.get("request_id"),
                },
            }

        output = resp.get("output")
        if isinstance(output, dict) and "action_manifest" in output:
            raw = output.get("action_manifest")
        else:
            raw = output
        if isinstance(raw, list):
            manifest = {"actions": list(raw)}
        elif isinstance(raw, dict):
            manifest = dict(raw)
        else:
            return {
                "status": "error",
                "ok": False,
                "contract": WORKFLOW_ACTION_MANIFEST_CONTRACT,
                "runtime": str(runtime or "").strip(),
                "dynamic": True,
                "reason": "workflow_action_discovery_invalid_output",
                "discovery": {
                    "contract": WORKFLOW_ACTION_DISCOVERY_CONTRACT,
                    "status": "error",
                    "request_id": resp.get("request_id"),
                },
            }
        context = dict(dict(request or {}).get("_workflow_action_discovery_context") or {})
        view = cls._workflow_action_manifest_card_view(
            {**dict(request or {}), "action_manifest": manifest},
            runtime=runtime,
            include_hidden=include_hidden,
        )
        return {
            **view,
            "dynamic": True,
            "discovery": {
                "contract": WORKFLOW_ACTION_DISCOVERY_CONTRACT,
                "status": "ok",
                "request_id": resp.get("request_id"),
                "entrypoint": dict(context.get("entrypoint") or {}),
            },
        }

    @staticmethod
    def _workflow_operation_result(status: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(status or {})
        if str(row.get("contract") or "") != "hosting.operation_status":
            return row
        result = row.get("result")
        if isinstance(result, dict):
            return dict(result)
        return {
            "status": "error",
            "reason": str(row.get("reason") or row.get("lifecycle") or "workflow_result_unavailable"),
            "operation": dict(row.get("operation") or {}),
            "result_ref": dict(row.get("result_ref") or {}) or None,
            "result_omission": dict(row.get("result_omission") or {}) or None,
        }

    def workflow_python_action_describe(
        self,
        *,
        request: Optional[Dict[str, Any]] = None,
        include_hidden: bool = False,
        dynamic: bool = False,
        profile: str = "node",
        environment_name: str = "workflow-python-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
        instance_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        req = dict(request or {})
        if not self._workflow_action_discovery_enabled(req, dynamic):
            return self._workflow_action_manifest_card_view(req, runtime="python", include_hidden=include_hidden)
        discovery_req = self._workflow_request_with_action_discovery(req, runtime="python")
        iid = str(instance_id or req.get("instance_id") or "").strip()
        if iid:
            response = self.workflow_python_instance_execute(
                instance_id=iid,
                request=discovery_req,
                profile=profile,
                environment_name=environment_name,
                environment_key=environment_key,
                engine_id=engine_id,
                capacity=capacity,
                sandbox_policy=sandbox_policy,
                host_capability_sessions=host_capability_sessions,
                approval_requester=approval_requester,
            )
        else:
            response = self.execute_workflow_python(
                profile=profile,
                environment_name=environment_name,
                environment_key=environment_key,
                engine_id=engine_id,
                request=discovery_req,
                capacity=capacity,
                sandbox_policy=sandbox_policy,
                host_capability_sessions=host_capability_sessions,
                approval_requester=approval_requester,
            )
        return self._workflow_action_manifest_from_discovery_response(
            self._workflow_operation_result(response),
            request=discovery_req,
            runtime="python",
            include_hidden=include_hidden,
        )

    def workflow_js_action_describe(
        self,
        *,
        request: Optional[Dict[str, Any]] = None,
        include_hidden: bool = False,
        dynamic: bool = False,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
        instance_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        req = dict(request or {})
        if not self._workflow_action_discovery_enabled(req, dynamic):
            return self._workflow_action_manifest_card_view(req, runtime="javascript", include_hidden=include_hidden)
        discovery_req = self._workflow_request_with_action_discovery(req, runtime="javascript")
        iid = str(instance_id or req.get("instance_id") or "").strip()
        if iid:
            response = self.workflow_js_instance_execute(
                instance_id=iid,
                request=discovery_req,
                profile=profile,
                environment_name=environment_name,
                environment_key=environment_key,
                engine_id=engine_id,
                node=node,
                javascript=javascript,
                capacity=capacity,
                sandbox_policy=sandbox_policy,
                host_capability_sessions=host_capability_sessions,
                approval_requester=approval_requester,
            )
        else:
            response = self.execute_workflow_js(
                profile=profile,
                environment_name=environment_name,
                environment_key=environment_key,
                engine_id=engine_id,
                request=discovery_req,
                node=node,
                javascript=javascript,
                capacity=capacity,
                sandbox_policy=sandbox_policy,
                host_capability_sessions=host_capability_sessions,
                approval_requester=approval_requester,
            )
        return self._workflow_action_manifest_from_discovery_response(
            self._workflow_operation_result(response),
            request=discovery_req,
            runtime="javascript",
            include_hidden=include_hidden,
        )

    def execute_workflow_python_action(
        self,
        *,
        action_name: str,
        request: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        req = {**dict(request or {}), "action_name": str(action_name or "").strip()}
        return self.execute_workflow_python(request=req, **kwargs)

    def execute_workflow_js_action(
        self,
        *,
        action_name: str,
        request: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        req = {**dict(request or {}), "action_name": str(action_name or "").strip()}
        return self.execute_workflow_js(request=req, **kwargs)

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

    def workflow_artifact_recovery_inspect(
        self,
        *,
        request_id: str,
        names: Optional[list[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._workflow_python_artifact_manager(sandbox_policy=sandbox_policy).recovery_candidates(
            request_id=str(request_id or ""),
            names=list(names or []),
        )

    def workflow_artifact_recovery_claim(
        self,
        *,
        request_id: str,
        names: Optional[list[str]] = None,
        target_id: str = "",
        instance_id: str = "",
        patch_absolute_paths: bool = False,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._workflow_python_artifact_manager(sandbox_policy=sandbox_policy).claim_recovery_artifacts(
            request_id=str(request_id or ""),
            names=list(names or []),
            target_id=str(target_id or ""),
            instance_id=str(instance_id or ""),
            patch_absolute_paths=bool(patch_absolute_paths),
        )

    def workflow_artifact_recovery_cleanup(
        self,
        *,
        request_id: str,
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        manager = self._workflow_python_artifact_manager(sandbox_policy=sandbox_policy)
        return manager.cleanup_run({"run_root": str(manager.run_root_for_request(str(request_id or "")))})

    def _workflow_artifact_recovery_notice(
        self,
        *,
        request_id: str,
        artifact_context: Optional[Dict[str, Any]],
        reason: str = "",
        instance_id: str = "",
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if artifact_context is None:
            return None
        notice = self.workflow_artifact_recovery_inspect(
            request_id=str(request_id or ""),
            sandbox_policy=sandbox_policy,
        )
        if instance_id and not notice.get("instance_id"):
            notice["instance_id"] = str(instance_id or "").strip() or None
        notice["reason"] = str(reason or "") or None
        return notice

    def _workflow_python_node_host_dispatcher(
        self,
        *,
        request: Dict[str, Any],
        artifact_context: Optional[Dict[str, Any]],
        engine_id: str = "",
        sandbox_policy: Optional[Dict[str, Any]] = None,
        event_emitter: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        audit_emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
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
        host_api_enabled = bool(host_api_policy.get("enabled", True))
        http_enabled = (
            http_namespace_enabled
            and bool(worker_policy.enabled)
            and bool(worker_policy.brokered_io.http)
            and str(worker_policy.network.mode or "").strip().lower() == "brokered_only"
        )
        workflow_id = str(dict(request or {}).get("workflow_id") or "")
        instance_id = str(dict(request or {}).get("instance_id") or "")
        request_id = str(dict(request or {}).get("request_id") or "")
        effective_engine_id = str(engine_id or dict(request or {}).get("engine_id") or "").strip()
        state_policy = host_api_policy.get("state")
        state_namespace_enabled = bool(namespace_policy.get("state", False))

        def _state_scope_enabled(scope: str) -> bool:
            if not bool(host_api_policy.get("enabled", True)):
                return False
            normalized = str(scope or "").strip().lower()
            if isinstance(state_policy, dict):
                if normalized in state_policy:
                    return bool(state_policy.get(normalized))
                if normalized == "backend":
                    return False
                return bool(state_policy.get("enabled", state_namespace_enabled))
            if state_policy is not None:
                return bool(state_policy) and normalized != "backend"
            return state_namespace_enabled and normalized != "backend"

        state_scopes: list[str] = []
        if _state_scope_enabled("backend"):
            state_scopes.append("backend")
        if workflow_id and _state_scope_enabled("workflow"):
            state_scopes.append("workflow")
        if instance_id and _state_scope_enabled("instance"):
            state_scopes.append("instance")
        state_available = bool(state_scopes)
        disabled_namespaces: set[str] = set()
        if not artifact_fs_enabled:
            disabled_namespaces.add("fs")
        if not http_namespace_enabled:
            disabled_namespaces.add("http")

        def _approval_requester(payload: Dict[str, Any]) -> Dict[str, Any]:
            if approval_requester is None:
                return {"status": "denied", "approved": False, "decision": "deny", "reason": "approval_requester_unavailable"}
            from ..callable_surface import host_capability_approval_request

            normalized = host_capability_approval_request(dict(payload or {}))
            return approval_requester(normalized)

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
                    "state": state_available,
                    "subprocess": False,
                    "custom_functions": False,
                },
                "http": http_enabled,
                "state": {"available": state_available, "scopes": list(state_scopes)},
                "subprocess": False,
                "custom_functions": False,
            },
        )

        def _state_notice(action: str, result: Dict[str, Any]) -> None:
            if event_emitter is None:
                return
            try:
                event_emitter(
                    "state_notice",
                    {
                        "action": action,
                        "scope": result.get("scope"),
                        "key": result.get("key"),
                        "version": result.get("version"),
                        "request_id": request_id or None,
                        "workflow_id": workflow_id or None,
                        "instance_id": instance_id or None,
                    },
                )
            except Exception:
                pass

        def _state_call(scope: str, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
            normalized_scope = str(scope or "").strip().lower()
            if normalized_scope not in state_scopes:
                raise PermissionError(f"state_scope_unavailable:{normalized_scope}")
            row = dict(args or {})
            if action == "get":
                return self.sandbox_state_get(
                    scope=normalized_scope,
                    key=str(row.get("key") or ""),
                    workflow_id=workflow_id,
                    instance_id=instance_id,
                    request_id=request_id,
                )
            if action == "set":
                result = self.sandbox_state_set(
                    scope=normalized_scope,
                    key=str(row.get("key") or ""),
                    value=row.get("value"),
                    workflow_id=workflow_id,
                    instance_id=instance_id,
                    request_id=request_id,
                    expected_version=row.get("expected_version") if "expected_version" in row else None,
                )
                _state_notice("set", result)
                return result
            if action == "list":
                return self.sandbox_state_list(
                    scope=normalized_scope,
                    prefix=str(row.get("prefix") or ""),
                    workflow_id=workflow_id,
                    instance_id=instance_id,
                    request_id=request_id,
                )
            if action == "delete":
                result = self.sandbox_state_delete(
                    scope=normalized_scope,
                    key=str(row.get("key") or ""),
                    workflow_id=workflow_id,
                    instance_id=instance_id,
                    request_id=request_id,
                    expected_version=row.get("expected_version") if "expected_version" in row else None,
                )
                _state_notice("delete", result)
                return result
            raise RuntimeError(f"unsupported_state_action:{action}")

        def _state_args_schema(action: str) -> Dict[str, Any]:
            if action == "set":
                return {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {},
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                }
            if action == "list":
                return {
                    "type": "object",
                    "properties": {"prefix": {"type": "string", "default": ""}},
                    "additionalProperties": False,
                }
            return {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "expected_version": {"type": "integer"},
                },
                "required": ["key"],
                "additionalProperties": False,
            }

        def _state_result_schema(action: str) -> Dict[str, Any]:
            properties: Dict[str, Any] = {
                "status": {"type": "string"},
                "scope": {"type": "string"},
            }
            if action == "list":
                properties.update(
                    {
                        "prefix": {"type": "string"},
                        "keys": {"type": "array", "items": {"type": "string"}},
                        "entries": {"type": "array", "items": {"type": "object"}},
                    }
                )
            elif action == "get":
                properties.update(
                    {
                        "key": {"type": "string"},
                        "exists": {"type": "boolean"},
                        "value": {},
                        "version": {"type": "integer"},
                        "updated_at": {"type": ["number", "null"]},
                    }
                )
            else:
                properties.update(
                    {
                        "key": {"type": "string"},
                        "version": {"type": "integer"},
                        "updated_at": {"type": "number"},
                    }
                )
                if action == "delete":
                    properties["existed"] = {"type": "boolean"}
            return {"type": "object", "properties": properties}

        def _state_capability_methods() -> list[HostCapabilityMethod]:
            methods: list[HostCapabilityMethod] = []
            provider_id = "builtin.workflow_node_state"
            for scope in state_scopes:
                for action in ("get", "set", "list", "delete"):
                    access = "read" if action in {"get", "list"} else "write"
                    method_name = f"state.{scope}.{action}"
                    descriptor = HostCapabilityDescriptor(
                        name=method_name,
                        namespace="state",
                        group_path=default_group_path(method_name),
                        description=f"{action.title()} {scope} sandbox state.",
                        args_schema=_state_args_schema(action),
                        result_schema=_state_result_schema(action),
                        permissions=[f"state.{scope}.{access}"],
                        scope_requirements=[{"scope": f"state.{scope}", "access": access}],
                        provider=HostCapabilityProviderRef(
                            provider_id=provider_id,
                            kind="builtin",
                            owner="service",
                            visibility="request",
                        ),
                    )
                    methods.append(
                        HostCapabilityMethod(
                            descriptor=descriptor,
                            handler=lambda args, _scope=scope, _action=action: _state_call(_scope, _action, args),
                        )
                    )
            return methods

        broker = HostCapabilityBroker(
            request_id=request_id,
            workflow_id=workflow_id,
            package_id=str(dict(request or {}).get("package_id") or ""),
            instance_id=instance_id,
            engine_id=effective_engine_id,
            runtime_kind="workflow_node",
            policy=dict(registry.policy or {}),
            roots=dict(registry.roots or {}),
            event_emitter=event_emitter,
            audit_emitter=audit_emitter or self._append_host_capability_audit_event,
            provider_invoker=self._host_capability_provider_invoker,
            approval_requester=_approval_requester if approval_requester is not None else None,
            allowed_namespaces=set() if not host_api_enabled else None,
            disabled_namespaces=disabled_namespaces,
            state_info={
                "available": state_available,
                "scopes": list(state_scopes),
                "provider_id": "builtin.workflow_node_state" if state_available else None,
            },
        )
        state_methods = _state_capability_methods()
        if state_methods:
            broker.register_builtin_provider(
                provider_id="builtin.workflow_node_state",
                owner="service",
                methods=state_methods,
            )
        capability_sessions = self._host_capability_sessions_for_broker(host_capability_sessions)
        for session in capability_sessions:
            broker.register_session(session)

        def _dispatch(call: Dict[str, Any]) -> Dict[str, Any]:
            return broker.dispatch(dict(call or {}))

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
        artifact_recovery = dict(result.get("artifact_recovery") or {}) if isinstance(result.get("artifact_recovery"), dict) else None
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
            "artifact_recovery": artifact_recovery,
            "progress": dict(result.get("progress") or {}) or None,
            "logs": logs,
            "metrics": dict(metrics or {}),
            "audit": {
                "package_id": str(req.get("package_id") or "").strip() or None,
                "workflow_id": str(req.get("workflow_id") or "").strip() or None,
                "package_source_digest": str(req.get("package_source_digest") or "").strip() or None,
                "module_sha256": str(req.get("module_sha256") or "").strip() or None,
                "provenance": dict(req.get("provenance") or {}),
                "action": dict(req.get("_workflow_action_context") or {}) or None,
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

    def _cancel_workflow_js_runtime(
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
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        req = dict(request or {})
        request_id = str(req.get("request_id") or "").strip()
        if not request_id:
            raise ValueError("request_id is required for durable hosted execution")
        normalized = self._workflow_request_with_action(req, runtime="javascript")
        js_policy = {
            **dict(node or {}),
            **dict(javascript or {}),
            **dict(normalized.get("javascript") or {}),
        }
        environment = self.workflow_js_environment_spec(
            profile=prof,
            environment_name=environment_name,
            node=dict(node or {}),
            javascript=js_policy,
            sandbox_policy=sandbox_policy,
        )
        derived_key = str(environment.get("environment_key") or "").strip()
        requested_key = str(environment_key or "").strip()
        effective_key = requested_key or derived_key
        eid = str(engine_id or "").strip() or self.workflow_js_default_engine_id(environment_key=effective_key)
        selector = HostedOperationSelector(kind="engine_id", id=eid)
        fingerprint = hosted_execution_fingerprint(
            {
                "execution_kind": HostedExecutionKind.WORKFLOW_JS.value,
                "selector": selector.to_dict(),
                "profile": prof,
                "environment_name": str(environment_name or "workflow-js-node"),
                "environment_key": effective_key,
                "environment": dict(environment.get("environment") or {}),
                "request": normalized,
                "node": dict(node or {}),
                "javascript": js_policy,
                "sandbox_policy": dict(sandbox_policy or {}),
                "capacity": max(1, int(capacity or 1)),
            }
        )
        prepared = self._hosted_operations.prepare(
            owner_actor_id=str(owner_actor_id or "service:local").strip() or "service:local",
            execution_kind=HostedExecutionKind.WORKFLOW_JS,
            selector=selector,
            namespace=f"workflow_js:{eid}",
            request_id=request_id,
            fingerprint=fingerprint,
            metadata={
                "engine_id": eid,
                "environment_key": effective_key,
                "profile": prof,
                "request_id": request_id,
                "runtime": "javascript",
            },
        )
        action = str(prepared.get("action") or "")
        status = dict(prepared.get("status") or {})
        if action in {"conflict", "forgotten", "replay"}:
            return status
        if action == "capacity":
            raise RuntimeError("hosted_operation_capacity_exceeded")
        operation_id = str(dict(status.get("operation") or {}).get("operation_id") or "")
        if action == "attach":
            return self._hosted_operations.wait_for_terminal(
                operation_id=operation_id,
                timeout_seconds=float(dict(normalized.get("limits") or {}).get("timeout_ms") or 30_000) / 1000.0,
            )
        claimed = self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
        if str(claimed.get("lifecycle") or "") != HostedOperationLifecycle.RUNNING.value:
            return claimed
        try:
            result = self._execute_workflow_js_runtime(
                profile=prof,
                environment_name=environment_name,
                environment_key=environment_key,
                engine_id=engine_id,
                request=req,
                node=node,
                javascript=javascript,
                capacity=capacity,
                sandbox_policy=sandbox_policy,
                host_capability_sessions=host_capability_sessions,
                approval_requester=approval_requester,
            )
        except Exception as exc:
            result = {"status": "error", "reason": str(exc) or "workflow_js_execute_failed", "error_type": type(exc).__name__}
        result_status = str(dict(result or {}).get("status") or "").strip().lower()
        reason = str(dict(result or {}).get("reason") or "").strip()
        lifecycle = (
            HostedOperationLifecycle.TERMINAL_CANCELLATION
            if result_status == "canceled" or reason == "workflow_sandbox_canceled"
            else HostedOperationLifecycle.TERMINAL_SUCCESS
            if result_status == "ok" and dict(result or {}).get("ok", True) is not False
            else HostedOperationLifecycle.TERMINAL_FAILURE
        )
        return self._hosted_operations.finish(
            operation_id=operation_id,
            lifecycle=lifecycle,
            envelope=dict(result or {}),
            reason=reason,
        )

    def _execute_workflow_js_runtime(
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
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        req = dict(request or {})
        runtime_instance_id = str(req.pop("_runtime_instance_id", "") or "").strip()
        req = self._workflow_request_with_action(req, runtime="javascript")
        req = self._workflow_python_with_project_artifact_input(req)
        action_error = dict(req.pop("_workflow_action_error", {}) or {})
        if action_error:
            return self._workflow_js_node_response_from_execution(
                execution={
                    "ok": False,
                    "reason": str(action_error.get("reason") or "workflow_action_invalid"),
                    "detail": dict(action_error.get("detail") or {}),
                },
                request=req,
                environment_key=str(environment_key or ""),
                engine_id=str(engine_id or ""),
            )
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
        required = ["module_sha256", "package_id", "workflow_id", "package_source_digest"]
        if str(req.get("execution_mode") or js.get("execution_mode") or "").strip().lower() != "project":
            required.insert(0, "module_source")
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

                def _record_js_broker_event(event_type: str, payload: Dict[str, Any]) -> None:
                    if str(event_type or "") != "host_call":
                        _record_js_event(event_type, payload)

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
                        engine_id=str(ensured["engine_id"]),
                        sandbox_policy=sandbox_policy,
                        event_emitter=_record_js_broker_event,
                        host_capability_sessions=host_capability_sessions,
                        approval_requester=approval_requester,
                    ),
                    instance_id=runtime_instance_id,
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
                if artifact_context is not None and bool(result.get("ok", False)):
                    self._workflow_python_cleanup_node_artifacts(artifact_context, sandbox_policy=sandbox_policy)
                elif artifact_context is not None:
                    result["artifact_recovery"] = self._workflow_artifact_recovery_notice(
                        request_id=lifecycle.request_id,
                        artifact_context=artifact_context,
                        reason=str(result.get("reason") or ""),
                        instance_id=str(req.get("instance_id") or runtime_instance_id or ""),
                        sandbox_policy=sandbox_policy,
                    )
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
            request={**req, "request_id": lifecycle.request_id, "javascript": js},
            environment_key=str(ensured.get("environment_key") or ""),
            engine_id=str(ensured["engine_id"]),
            metrics={
                "workflow_pool": pool.resources(),
                "request": dict(finished.get("request") or lifecycle.to_dict()),
            },
        )

    def workflow_js_instance_create(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        instance_id: Optional[str] = None,
        replace: bool = False,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        req = self._workflow_python_with_project_artifact_input(dict(request or {}))
        js = {**dict(javascript or {}), **dict(req.get("javascript") or {})}
        required = ["module_sha256", "package_id", "workflow_id", "package_source_digest"]
        if str(req.get("execution_mode") or js.get("execution_mode") or "").strip().lower() != "project":
            required.insert(0, "module_source")
        missing = [name for name in required if not str(req.get(name) or "").strip()]
        if missing:
            return {
                "status": "error",
                "ok": False,
                "profile": prof,
                "reason": "workflow_js_node_invalid_request",
                "detail": {"message": f"missing required fields: {', '.join(missing)}"},
            }
        ensured = self.ensure_workflow_js(
            profile=prof,
            environment_name=environment_name,
            environment_key=environment_key,
            node=dict(node or req.get("node") or {}),
            javascript=js,
            capacity=1,
            sandbox_policy=sandbox_policy,
            engine_id=engine_id,
        )
        if str(ensured.get("status") or "") != "ok":
            return ensured
        created = self._workflow_js_node_runtime_registry().create_instance(
            {
                **req,
                "environment_key": str(ensured.get("environment_key") or ""),
                "javascript": js,
            },
            python_executable=str(js.get("python_executable") or "").strip() or None,
            instance_id=str(instance_id or "").strip(),
            replace=replace,
        )
        if str(created.get("status") or "") == "ok":
            created.update(
                {
                    "profile": prof,
                    "engine_id": str(ensured.get("engine_id") or ""),
                    "environment_key": str(ensured.get("environment_key") or ""),
                }
            )
        return dict(created or {})

    def workflow_js_instance_execute(
        self,
        *,
        instance_id: str,
        request: Optional[Dict[str, Any]] = None,
        profile: str = "node",
        environment_name: str = "workflow-js-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        node: Optional[Dict[str, Any]] = None,
        javascript: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        iid = str(instance_id or "").strip()
        if not iid:
            return {"status": "error", "ok": False, "reason": "instance_id_required"}
        req = dict(request or {})
        req.setdefault("instance_id", iid)
        req["_runtime_instance_id"] = iid
        return self.execute_workflow_js(
            profile=profile,
            environment_name=environment_name,
            environment_key=environment_key,
            engine_id=engine_id,
            request=req,
            node=node,
            javascript=javascript,
            capacity=capacity,
            sandbox_policy=sandbox_policy,
            host_capability_sessions=host_capability_sessions,
            approval_requester=approval_requester,
            owner_actor_id=owner_actor_id,
        )

    def workflow_js_instance_close(self, *, instance_id: str, reason: str = "client_requested") -> Dict[str, Any]:
        return dict(self._workflow_js_node_runtime_registry().close_instance(instance_id, reason=reason))

    def workflow_js_instance_list(self) -> Dict[str, Any]:
        return dict(self._workflow_js_node_runtime_registry().list_instances())

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
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        prof = self._workflow_js_profile(profile)
        req = self._workflow_python_with_project_artifact_input(dict(request or {}))
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
                "host_capability_sessions": host_capability_sessions,
                "approval_requester": approval_requester,
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
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        base = self._workflow_js_stream_base()
        live_stdout_seen = False

        def _emit_js_event(event_type: str, payload: Dict[str, Any]) -> None:
            nonlocal live_stdout_seen
            if event_type == "console":
                base.stream_emit(
                    stream_id=stream_id,
                    event_type="stdout",
                    payload={
                        "text": str(dict(payload or {}).get("message") or ""),
                        "level": str(dict(payload or {}).get("level") or "log"),
                    },
                )
                live_stdout_seen = True
                return
            if event_type == "host_call":
                base.stream_emit(stream_id=stream_id, event_type="host_call", payload=dict(payload or {}))
                return
            base.stream_emit(stream_id=stream_id, event_type=event_type, payload=dict(payload or {}))

        def _emit_js_broker_event(event_type: str, payload: Dict[str, Any]) -> None:
            if str(event_type or "") != "host_call":
                _emit_js_event(event_type, payload)

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
                engine_id=engine_id,
                sandbox_policy=sandbox_policy,
                event_emitter=_emit_js_broker_event,
                host_capability_sessions=host_capability_sessions,
                approval_requester=approval_requester,
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
        if artifact_context is not None and bool(result.get("ok", False)):
            self._workflow_python_cleanup_node_artifacts(artifact_context, sandbox_policy=sandbox_policy)
        elif artifact_context is not None:
            result["artifact_recovery"] = self._workflow_artifact_recovery_notice(
                request_id=str(request.get("request_id") or ""),
                artifact_context=artifact_context,
                reason=str(result.get("reason") or ""),
                instance_id=str(request.get("instance_id") or ""),
                sandbox_policy=sandbox_policy,
            )

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
        if str(logs.get("stdout") or "") and not live_stdout_seen:
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
                out["worker_cancel"] = self._cancel_workflow_js_runtime(
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
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        prof = self._workflow_python_profile(profile)
        req = dict(request or {})
        request_id = str(req.get("request_id") or "").strip()
        if not request_id:
            raise ValueError("request_id is required for durable hosted execution")
        normalized = self._workflow_request_with_action(req, runtime="python")
        py_policy = dict(normalized.get("python") or {})
        effective_environment_name = str(
            py_policy.get("environment_name")
            or ("workflow-python-node" if prof == "node" and environment_name == "workflow-python-helper" else environment_name)
            or "workflow-python-helper"
        )
        environment = self.workflow_python_environment_spec(
            profile=prof,
            environment_name=effective_environment_name,
            python=py_policy,
            sandbox_policy=sandbox_policy,
        )
        derived_key = str(environment.get("environment_key") or "").strip()
        requested_key = str(environment_key or "").strip()
        effective_key = requested_key or derived_key
        eid = str(engine_id or "").strip() or self.workflow_python_default_engine_id(environment_key=effective_key)
        selector = HostedOperationSelector(kind="engine_id", id=eid)
        fingerprint = hosted_execution_fingerprint(
            {
                "execution_kind": HostedExecutionKind.WORKFLOW_PYTHON.value,
                "selector": selector.to_dict(),
                "profile": prof,
                "environment_name": effective_environment_name,
                "environment_key": effective_key,
                "environment": dict(environment.get("environment") or {}),
                "request": normalized,
                "python": py_policy,
                "sandbox_policy": dict(sandbox_policy or {}),
                "capacity": max(1, int(capacity or 1)),
            }
        )
        prepared = self._hosted_operations.prepare(
            owner_actor_id=str(owner_actor_id or "service:local").strip() or "service:local",
            execution_kind=HostedExecutionKind.WORKFLOW_PYTHON,
            selector=selector,
            namespace=f"workflow_python:{eid}",
            request_id=request_id,
            fingerprint=fingerprint,
            metadata={
                "engine_id": eid,
                "environment_key": effective_key,
                "profile": prof,
                "request_id": request_id,
                "runtime": "python",
            },
        )
        action = str(prepared.get("action") or "")
        status = dict(prepared.get("status") or {})
        if action in {"conflict", "forgotten", "replay"}:
            return status
        if action == "capacity":
            raise RuntimeError("hosted_operation_capacity_exceeded")
        operation_id = str(dict(status.get("operation") or {}).get("operation_id") or "")
        if action == "attach":
            return self._hosted_operations.wait_for_terminal(
                operation_id=operation_id,
                timeout_seconds=float(dict(normalized.get("limits") or {}).get("timeout_ms") or 30_000) / 1000.0,
            )
        claimed = self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
        if str(claimed.get("lifecycle") or "") != HostedOperationLifecycle.RUNNING.value:
            return claimed
        try:
            result = self._execute_workflow_python_runtime(
                profile=prof,
                environment_name=environment_name,
                environment_key=environment_key,
                engine_id=engine_id,
                request=req,
                capacity=capacity,
                sandbox_policy=sandbox_policy,
                host_capability_sessions=host_capability_sessions,
                approval_requester=approval_requester,
            )
        except Exception as exc:
            result = {"status": "error", "reason": str(exc) or "workflow_python_execute_failed", "error_type": type(exc).__name__}
        result_status = str(dict(result or {}).get("status") or "").strip().lower()
        reason = str(dict(result or {}).get("reason") or "").strip()
        lifecycle = (
            HostedOperationLifecycle.TERMINAL_CANCELLATION
            if result_status == "canceled" or reason == "workflow_sandbox_canceled"
            else HostedOperationLifecycle.TERMINAL_SUCCESS
            if result_status == "ok" and dict(result or {}).get("ok", True) is not False
            else HostedOperationLifecycle.TERMINAL_FAILURE
        )
        return self._hosted_operations.finish(
            operation_id=operation_id,
            lifecycle=lifecycle,
            envelope=dict(result or {}),
            reason=reason,
        )

    def _execute_workflow_python_runtime(
        self,
        *,
        profile: str = "helper",
        environment_name: str = "workflow-python-helper",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        prof = self._workflow_python_profile(profile)
        req = dict(request or {})
        runtime_instance_id = str(req.pop("_runtime_instance_id", "") or "").strip()
        req = self._workflow_request_with_action(req, runtime="python")
        action_error = dict(req.pop("_workflow_action_error", {}) or {})
        if action_error:
            if prof == "node":
                return self._workflow_python_node_response_from_execution(
                    execution={
                        "ok": False,
                        "reason": str(action_error.get("reason") or "workflow_action_invalid"),
                        "detail": dict(action_error.get("detail") or {}),
                    },
                    request=req,
                    environment_key=str(environment_key or ""),
                    engine_id=str(engine_id or ""),
                )
            return {
                "status": "error",
                "ok": False,
                "profile": prof,
                "reason": str(action_error.get("reason") or "workflow_action_invalid"),
                "detail": dict(action_error.get("detail") or {}),
            }
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
                        request={**req, "request_id": lifecycle.request_id, "python": py},
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

            def _record_node_broker_event(event_type: str, payload: Dict[str, Any]) -> None:
                if str(event_type or "") != "host_call":
                    _record_node_event(event_type, payload)

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
                    engine_id=eid,
                    sandbox_policy=sandbox_policy,
                    event_emitter=_record_node_broker_event,
                    host_capability_sessions=host_capability_sessions,
                    approval_requester=approval_requester,
                ),
                max_idle=int(pool.resources().get("metrics", {}).get("desired_capacity") or capacity or 1),
                instance_id=runtime_instance_id,
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
            if artifact_context is not None and bool(result.get("ok", False)):
                self._workflow_python_cleanup_node_artifacts(artifact_context, sandbox_policy=sandbox_policy)
            elif artifact_context is not None:
                result["artifact_recovery"] = self._workflow_artifact_recovery_notice(
                    request_id=lifecycle.request_id,
                    artifact_context=artifact_context,
                    reason=str(result.get("reason") or ""),
                    instance_id=str(req.get("instance_id") or runtime_instance_id or ""),
                    sandbox_policy=sandbox_policy,
                )
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
                request={**req, "request_id": lifecycle.request_id, "python": py},
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

    def workflow_python_instance_create(
        self,
        *,
        profile: str = "node",
        environment_name: str = "workflow-python-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        instance_id: Optional[str] = None,
        replace: bool = False,
    ) -> Dict[str, Any]:
        prof = self._workflow_python_profile(profile)
        if prof != "node":
            return {"status": "error", "ok": False, "reason": "workflow_python_instance_requires_node_profile"}
        req = self._workflow_python_with_project_artifact_input(dict(request or {}))
        if str(environment_name or "") == "workflow-python-helper":
            environment_name = "workflow-python-node"
        py = dict(req.get("python") or {})
        py.setdefault("environment_name", str(environment_name or "workflow-python-node"))
        req["python"] = py
        validation = validate_workflow_python_node_request(req)
        if str(validation.get("status") or "") != "ok":
            return {
                "status": "error",
                "ok": False,
                "profile": prof,
                "reason": "workflow_python_node_invalid_request",
                "detail": {"missing_request_fields": list(validation.get("missing") or [])},
            }
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
        created = self._workflow_python_node_runtime_registry().create_instance(
            {
                **req,
                "environment_key": effective_key,
                "python": py,
            },
            python_executable=str(py.get("python_executable") or "").strip() or None,
            instance_id=str(instance_id or "").strip(),
            replace=replace,
        )
        if str(created.get("status") or "") == "ok":
            created.update({"profile": prof, "engine_id": eid, "environment_key": effective_key})
            self._workflow_python_pool_registry().get_or_create(
                self._workflow_python_pool_key(effective_key),
                desired_capacity=1,
            )
        return dict(created or {})

    def workflow_python_instance_execute(
        self,
        *,
        instance_id: str,
        request: Optional[Dict[str, Any]] = None,
        profile: str = "node",
        environment_name: str = "workflow-python-node",
        environment_key: Optional[str] = None,
        engine_id: Optional[str] = None,
        capacity: int = 1,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        iid = str(instance_id or "").strip()
        if not iid:
            return {"status": "error", "ok": False, "reason": "instance_id_required"}
        req = dict(request or {})
        req.setdefault("instance_id", iid)
        req["_runtime_instance_id"] = iid
        return self.execute_workflow_python(
            profile=profile,
            environment_name=environment_name,
            environment_key=environment_key,
            engine_id=engine_id,
            request=req,
            capacity=capacity,
            sandbox_policy=sandbox_policy,
            host_capability_sessions=host_capability_sessions,
            approval_requester=approval_requester,
            owner_actor_id=owner_actor_id,
        )

    def workflow_python_instance_close(self, *, instance_id: str, reason: str = "client_requested") -> Dict[str, Any]:
        return dict(self._workflow_python_node_runtime_registry().close_instance(instance_id, reason=reason))

    def workflow_python_instance_list(self) -> Dict[str, Any]:
        return dict(self._workflow_python_node_runtime_registry().list_instances())

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

    def _cancel_workflow_python_runtime(
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

    def _cancel_workflow_operation(self, *, record: Dict[str, Any], reason: str) -> Dict[str, Any]:
        row = dict(record or {})
        operation = dict(row.get("operation") or {})
        metadata = dict(row.get("metadata") or {})
        operation_id = str(operation.get("operation_id") or "").strip()
        owner_actor_id = str(row.get("owner_actor_id") or "").strip()
        lifecycle = HostedOperationLifecycle(str(row.get("lifecycle") or ""))
        if lifecycle in {
            HostedOperationLifecycle.QUEUED,
            HostedOperationLifecycle.INTERRUPTED_BEFORE_DISPATCH,
        }:
            canceled = self._hosted_operations.cancel_before_dispatch(
                operation_id=operation_id,
                reason=str(reason or "canceled_before_dispatch"),
            )
            if canceled is not None:
                return canceled
        if lifecycle in {
            HostedOperationLifecycle.TERMINAL_SUCCESS,
            HostedOperationLifecycle.TERMINAL_FAILURE,
            HostedOperationLifecycle.TERMINAL_CANCELLATION,
            HostedOperationLifecycle.FORGOTTEN,
        }:
            return self._hosted_operations.status(ref=operation, owner_actor_id=owner_actor_id)
        runtime = str(metadata.get("runtime") or "").strip()
        kwargs = {
            "profile": str(metadata.get("profile") or ("node" if runtime == "javascript" else "helper")),
            "environment_key": str(metadata.get("environment_key") or "") or None,
            "engine_id": str(metadata.get("engine_id") or "") or None,
            "request_id": str(operation.get("request_id") or ""),
        }
        try:
            if runtime == "javascript":
                canceled_runtime = self._cancel_workflow_js_runtime(**kwargs)
            elif runtime == "python":
                canceled_runtime = self._cancel_workflow_python_runtime(**kwargs)
            else:
                raise ValueError("stored workflow operation runtime is invalid")
        except Exception as exc:
            canceled_runtime = {
                "status": "error",
                "canceled": False,
                "reason": str(exc) or "workflow_cancel_failed",
                "error_type": type(exc).__name__,
            }
        canceled = bool(dict(canceled_runtime or {}).get("canceled"))
        terminal_lifecycle = (
            HostedOperationLifecycle.TERMINAL_CANCELLATION
            if canceled
            else HostedOperationLifecycle.TERMINAL_FAILURE
        )
        terminal_reason = str(reason or "client_requested") if canceled else str(
            dict(canceled_runtime or {}).get("reason") or "workflow_cancel_target_not_active"
        )
        return self._hosted_operations.finish(
            operation_id=operation_id,
            lifecycle=terminal_lifecycle,
            envelope=dict(canceled_runtime or {}),
            reason=terminal_reason,
        )

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
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
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
                    "host_capability_sessions": host_capability_sessions,
                    "approval_requester": approval_requester,
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
        host_capability_sessions: Optional[list[HostCapabilitySession]] = None,
        approval_requester: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        base = self._workflow_python_stream_base()
        def _emit_node_event(event_type: str, payload: Dict[str, Any]) -> None:
            base.stream_emit(stream_id=stream_id, event_type=event_type, payload=dict(payload or {}))

        def _emit_node_broker_event(event_type: str, payload: Dict[str, Any]) -> None:
            if str(event_type or "") != "host_call":
                _emit_node_event(event_type, payload)

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
                engine_id=engine_id,
                sandbox_policy=sandbox_policy,
                event_emitter=_emit_node_broker_event,
                host_capability_sessions=host_capability_sessions,
                approval_requester=approval_requester,
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
        if artifact_context is not None and bool(result.get("ok", False)):
            self._workflow_python_cleanup_node_artifacts(artifact_context, sandbox_policy=sandbox_policy)
        elif artifact_context is not None:
            result["artifact_recovery"] = self._workflow_artifact_recovery_notice(
                request_id=str(request.get("request_id") or ""),
                artifact_context=artifact_context,
                reason=str(result.get("reason") or ""),
                instance_id=str(request.get("instance_id") or ""),
                sandbox_policy=sandbox_policy,
            )
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
                out["worker_cancel"] = self._cancel_workflow_python_runtime(
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
