"""Hosted toolbox reference facade and pending mutation builder."""
from __future__ import annotations

import inspect
import secrets
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from mp13_engine.mp13_toolbox import ToolBoxRef, ToolsView

from .bundle_models import SandboxProfileSpec, ToolboxBundleFile
from .callbacks import _HostedToolCallbackRelay, _request_hosted_tool_approval_with_timeout
from .tools_view import (
    _apply_tool_constraints_in_view,
    _approval_timeout_seconds,
    _approve_tool_in_view,
    _coerce_approval_decision,
    _extract_scope_constraints,
    _merge_scope_ref_into_callback_context,
    _persist_approved_tool,
    _persist_scope_constraints,
    _resolve_scope_ref_from_callback_context,
    serialize_tools_view,
)


class HostedToolBoxRef:
    def __init__(
        self,
        *,
        toolbox_id: str,
        host: Any,
        python_executable: Optional[str] = None,
        worker_profile_class: str = "generic",
    ) -> None:
        self.toolbox_id = str(toolbox_id or "").strip()
        if not self.toolbox_id:
            raise ValueError("toolbox_id_required")
        self.host = host
        self.python_executable = str(python_executable or "").strip() or None
        self.worker_profile_class = str(worker_profile_class or "generic").strip() or "generic"

    @property
    def ref_name(self) -> str:
        return self.toolbox_id

    def _host_descriptor(self) -> Dict[str, Any]:
        host = self.host
        host_type = type(host).__name__
        descriptor: Dict[str, Any] = {
            "host_type": host_type,
        }
        if hasattr(host, "control_settings"):
            descriptor["kind"] = "control_channel"
            descriptor["control_settings"] = dict(getattr(host, "control_settings", {}) or {})
            return descriptor
        engines_state_file = getattr(host, "engines_state_file", None)
        control_state_file = getattr(host, "control_state_file", None)
        if engines_state_file is not None or control_state_file is not None:
            descriptor["kind"] = "service"
            descriptor["engines_state_file"] = str(engines_state_file) if engines_state_file is not None else None
            descriptor["control_state_file"] = str(control_state_file) if control_state_file is not None else None
            return descriptor
        descriptor["kind"] = "opaque"
        return descriptor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "toolbox_id": self.toolbox_id,
            "python_executable": self.python_executable,
            "worker_profile_class": self.worker_profile_class,
            "host": self._host_descriptor(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        *,
        host: Any = None,
    ) -> "HostedToolBoxRef":
        row = dict(payload or {})
        resolved_host = host
        if resolved_host is None:
            host_row = dict(row.get("host") or {})
            kind = str(host_row.get("kind") or "").strip().lower()
            if kind == "control_channel":
                from ..engine_host_channel import EngineHostControlChannel

                resolved_host = EngineHostControlChannel(dict(host_row.get("control_settings") or {}))
            elif kind == "service":
                from ..service.host_service import EngineHostService

                engines_state_raw = str(host_row.get("engines_state_file") or "").strip()
                control_state_raw = str(host_row.get("control_state_file") or "").strip()
                resolved_host = EngineHostService(
                    engines_state_file=Path(engines_state_raw) if engines_state_raw else None,
                    control_state_file=Path(control_state_raw) if control_state_raw else None,
                )
            else:
                raise ValueError("host_required_for_hosted_toolbox_ref_deserialization")
        return cls(
            toolbox_id=str(row.get("toolbox_id") or "").strip(),
            host=resolved_host,
            python_executable=str(row.get("python_executable") or "").strip() or None,
            worker_profile_class=str(row.get("worker_profile_class") or "generic").strip() or "generic",
        )

    def mutate(self) -> "PendingHostedToolboxRef":
        return PendingHostedToolboxRef(self)

    def register_auto_callable(
        self,
        *,
        relative_path: str,
        content: str,
        module_name: str,
        callable_name: str,
        environment_name: str = "base",
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        activate: bool = True,
        hidden: bool = False,
        non_restartable: bool = False,
        guide_content: Optional[Dict[str, List[str]]] = None,
        guide_description: Optional[str] = None,
        callback_signature: Optional[Dict[str, Any]] = None,
        concurrency: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        request = {
            "files": [
                ToolboxBundleFile(
                    relative_path=str(relative_path or "").strip(),
                    content=str(content or ""),
                ).to_runtime_dict()
            ],
            "module_name": str(module_name or "").strip(),
            "callable_name": str(callable_name or "").strip(),
            "sandbox_profile": SandboxProfileSpec(
                environment_name=str(environment_name or "base").strip() or "base",
                required_imports=[str(item or "").strip() for item in list(required_imports or []) if str(item or "").strip()],
                sandbox_policy=dict(sandbox_policy or {}),
            ).to_dict(),
            "activate": bool(activate),
            "hidden": bool(hidden),
            "non_restartable": bool(non_restartable),
            "guide_content": dict(guide_content or {}) or None,
            "guide_description": str(guide_description or "").strip() or None,
            "callback_signature": dict(callback_signature or {}) or None,
        }
        if isinstance(concurrency, dict) and concurrency:
            request["concurrency"] = dict(concurrency)
        return dict(
            self.host.toolbox_register_auto(
                toolbox_id=self.toolbox_id,
                requests=[request],
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def add_auto_callable(self, **kwargs: Any) -> Dict[str, Any]:
        return self.register_auto_callable(**kwargs)

    def register_python_callable(
        self,
        implementation: Any,
        *,
        environment_name: str = "base",
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        activate: bool = True,
        hidden: bool = False,
        non_restartable: bool = False,
        guide_content: Optional[Dict[str, List[str]]] = None,
        guide_description: Optional[str] = None,
        callback_signature: Optional[Dict[str, Any]] = None,
        concurrency: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        module = inspect.getmodule(implementation)
        module_name = str(getattr(implementation, "__module__", "") or getattr(module, "__name__", "") or "").strip()
        if not module_name:
            raise ValueError("callable_module_name_required")
        callable_name = str(getattr(implementation, "__name__", "") or "").strip()
        if not callable_name:
            raise ValueError("callable_name_required")
        source_path = inspect.getsourcefile(implementation) or getattr(module, "__file__", None)
        if not source_path:
            raise ValueError("callable_source_file_required")
        source_file = Path(str(source_path)).expanduser().resolve()
        if not source_file.exists():
            raise ValueError("callable_source_file_missing")
        return self.register_auto_callable(
            relative_path=source_file.name,
            content=source_file.read_text(encoding="utf-8"),
            module_name=module_name,
            callable_name=callable_name,
            environment_name=environment_name,
            required_imports=required_imports,
            sandbox_policy=sandbox_policy,
            activate=activate,
            hidden=hidden,
            non_restartable=non_restartable,
            guide_content=guide_content,
            guide_description=guide_description,
            callback_signature=callback_signature,
            concurrency=concurrency,
        )

    def add_python_callable(self, implementation: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.register_python_callable(implementation, **kwargs)

    def register_manual_tool(
        self,
        tool_definition: Dict[str, Any],
        implementation: Any,
        *,
        environment_name: str = "base",
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        hidden: bool = False,
        non_restartable: bool = False,
        callback_signature: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        module = inspect.getmodule(implementation)
        module_name = str(getattr(implementation, "__module__", "") or getattr(module, "__name__", "") or "").strip()
        if not module_name:
            raise ValueError("callable_module_name_required")
        callable_name = str(getattr(implementation, "__name__", "") or "").strip()
        if not callable_name:
            raise ValueError("callable_name_required")
        source_path = inspect.getsourcefile(implementation) or getattr(module, "__file__", None)
        if not source_path:
            raise ValueError("callable_source_file_required")
        source_file = Path(str(source_path)).expanduser().resolve()
        if not source_file.exists():
            raise ValueError("callable_source_file_missing")
        return dict(
            self.host.toolbox_register_manual(
                toolbox_id=self.toolbox_id,
                requests=[
                    {
                        "files": [
                            ToolboxBundleFile(
                                relative_path=source_file.name,
                                content=source_file.read_text(encoding="utf-8"),
                            ).to_runtime_dict()
                        ],
                        "module_name": module_name,
                        "callable_name": callable_name,
                        "tool_definition": dict(tool_definition or {}),
                        "sandbox_profile": SandboxProfileSpec(
                            environment_name=str(environment_name or "base").strip() or "base",
                            required_imports=[str(item or "").strip() for item in list(required_imports or []) if str(item or "").strip()],
                            sandbox_policy=dict(sandbox_policy or {}),
                        ).to_dict(),
                        "hidden": bool(hidden),
                        "non_restartable": bool(non_restartable),
                        "callback_signature": dict(callback_signature or {}) or None,
                    }
                ],
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def add_manual_tool(self, tool_definition: Dict[str, Any], implementation: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.register_manual_tool(tool_definition, implementation, **kwargs)

    def unregister_manual_tool(self, *, module_name: str, callable_name: str) -> Dict[str, Any]:
        key = f"manual:{str(module_name or '').strip()}:{str(callable_name or '').strip()}"
        return dict(
            self.host.toolbox_unregister_manual(
                toolbox_id=self.toolbox_id,
                tool_keys=[key],
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def remove_manual_tool(self, *, module_name: str, callable_name: str) -> Dict[str, Any]:
        return self.unregister_manual_tool(module_name=module_name, callable_name=callable_name)

    def unregister_auto_callable(self, *, module_name: str, callable_name: str) -> Dict[str, Any]:
        key = f"{str(module_name or '').strip()}:{str(callable_name or '').strip()}"
        return dict(
            self.host.toolbox_unregister_auto(
                toolbox_id=self.toolbox_id,
                tool_keys=[key],
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def remove_auto_callable(self, *, module_name: str, callable_name: str) -> Dict[str, Any]:
        return self.unregister_auto_callable(module_name=module_name, callable_name=callable_name)

    def register_intrinsic_tools(
        self,
        intrinsic_tool_names: Sequence[str],
        *,
        include_guides: bool = False,
        environment_name: str = "base",
        sandbox_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_register_intrinsics(
                toolbox_id=self.toolbox_id,
                intrinsic_tool_names=[str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()],
                include_guides=bool(include_guides),
                sandbox_profile=SandboxProfileSpec(
                    environment_name=str(environment_name or "base").strip() or "base",
                    sandbox_policy=dict(sandbox_policy or {}),
                ).to_dict(),
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def add_intrinsic_tools(self, intrinsic_tool_names: Sequence[str], **kwargs: Any) -> Dict[str, Any]:
        return self.register_intrinsic_tools(intrinsic_tool_names, **kwargs)

    def environment_descriptions(self) -> Dict[str, Any]:
        return dict(self.host.toolbox_environment_description_list() or {})

    def list_environment_descriptions(self) -> Dict[str, Any]:
        return self.environment_descriptions()

    def upsert_environment_description(
        self,
        *,
        name: str,
        base_env_name: Optional[str] = None,
        extra_packages: Optional[Sequence[str]] = None,
        allow_online_install: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_description_upsert(
                name=str(name or "").strip(),
                base_env_name=str(base_env_name or "").strip() or None,
                extra_packages=[str(item or "").strip() for item in list(extra_packages or []) if str(item or "").strip()],
                allow_online_install=bool(allow_online_install),
            )
            or {}
        )

    def clone_environment_description(
        self,
        *,
        source_name: str,
        target_name: str,
        extra_packages: Optional[Sequence[str]] = None,
        allow_online_install: Optional[bool] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_description_clone(
                source_name=str(source_name or "").strip(),
                target_name=str(target_name or "").strip(),
                extra_packages=[str(item or "").strip() for item in list(extra_packages or []) if str(item or "").strip()] if extra_packages is not None else None,
                allow_online_install=allow_online_install,
            )
            or {}
        )

    def resolve_environment_requirements(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_resolve_requirements(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def apply_environment_description(
        self,
        *,
        environment_name: str,
        toolbox_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_apply(
                environment_name=str(environment_name or "base").strip() or "base",
                toolbox_ids=[str(item or "").strip() for item in list(toolbox_ids or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def realize_environment(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_realize(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def sync_environment_description(
        self,
        *,
        source_environment_name: str,
        target_environment_name: Optional[str] = None,
        tool_keys: Optional[Sequence[str]] = None,
        apply: bool = False,
        realize: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_sync_description(
                toolbox_id=self.toolbox_id,
                source_environment_name=str(source_environment_name or "base").strip() or "base",
                target_environment_name=str(target_environment_name or "").strip() or None,
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
                apply=bool(apply),
                realize=bool(realize),
            )
            or {}
        )

    def prepare_environment_install(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_prepare_install(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def lock_environment_install(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_lock_install(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def resolve_environment_install_lock(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
        allow_resolution: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_resolve_install_lock(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
                allow_resolution=bool(allow_resolution),
            )
            or {}
        )

    def verify_environment_install_lock(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_verify_install_lock(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def verify_environment_install_receipt(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_verify_install_receipt(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
            )
            or {}
        )

    def execute_environment_install(
        self,
        *,
        environment_name: str,
        tool_keys: Optional[Sequence[str]] = None,
        allow_execution: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_environment_execute_install(
                toolbox_id=self.toolbox_id,
                environment_name=str(environment_name or "base").strip() or "base",
                tool_keys=[str(item or "").strip() for item in list(tool_keys or []) if str(item or "").strip()] or None,
                allow_execution=bool(allow_execution),
            )
            or {}
        )

    def unregister_intrinsic_tools(
        self,
        intrinsic_tool_names: Sequence[str],
        *,
        include_guides: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_unregister_intrinsics(
                toolbox_id=self.toolbox_id,
                intrinsic_tool_names=[str(item or "").strip() for item in list(intrinsic_tool_names or []) if str(item or "").strip()],
                include_guides=bool(include_guides),
                python_executable=self.python_executable,
                worker_profile_class=self.worker_profile_class,
            )
            or {}
        )

    def remove_intrinsic_tools(
        self,
        intrinsic_tool_names: Sequence[str],
        *,
        include_guides: bool = False,
    ) -> Dict[str, Any]:
        return self.unregister_intrinsic_tools(intrinsic_tool_names, include_guides=include_guides)

    def describe(self, *, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_describe(
                toolbox_id=self.toolbox_id,
                timeout_seconds=float(timeout_seconds or 10.0),
            )
            or {}
        )

    def gate(self, *, tool_name: str, tools_view: Optional[ToolsView] = None) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_gate(
                toolbox_id=self.toolbox_id,
                tool_name=str(tool_name or "").strip(),
                tools_view=serialize_tools_view(tools_view),
            )
            or {}
        )

    def list_tools(self, *, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        return self.describe(timeout_seconds=timeout_seconds)

    def execute(
        self,
        *,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 30.0,
        tools_view: Optional[ToolsView] = None,
        callback_processor: Optional[Callable[..., Any]] = None,
        callback_context: Any = None,
        scope_ref: Optional[ToolBoxRef] = None,
        tool_call_id: str = "",
        execution_request_id: str = "",
        host_api_approval: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute one hosted tool call through the current toolbox routing.

        Approval persistence note:
        - gated tools can trigger `callback_processor` with callback name
          `tool_requires_confirmation`
        - `allow_once` affects only this call
        - `add_to_scope` persists only when `callback_context` carries a
          durable scope target such as:
          - `{"toolbox_ref": some_toolbox_ref}`
          - `{"cursor": some_cursor_with_toolbox_ref}`
        - without that durable scope target, `add_to_scope` only mutates the
          current execution view for this call
        """
        name = str(tool_name or "").strip()
        if not name:
            raise ValueError("tool_name_required")
        durable_request_id = str(execution_request_id or "").strip()
        if not durable_request_id:
            raise ValueError("execution_request_id is required for durable hosted execution")
        callback_context = _merge_scope_ref_into_callback_context(callback_context, scope_ref)
        call_id = str(tool_call_id or "").strip() or secrets.token_hex(12)
        requested_tools_view = tools_view
        tools_view_payload = serialize_tools_view(requested_tools_view)
        gate_out: Dict[str, Any] = {}
        outcome = ""
        if hasattr(self.host, "toolbox_gate"):
            gate_out = self.gate(tool_name=name, tools_view=requested_tools_view)
            outcome = str(gate_out.get("outcome") or "").strip().lower()
        if outcome and outcome != "allowed":
            if outcome == "gated_requires_confirmation":
                approval_result = _request_hosted_tool_approval_with_timeout(
                    processor=callback_processor,
                    toolbox_id=self.toolbox_id,
                    tool_name=name,
                    tool_call_id=call_id,
                    tool_arguments=dict(arguments or {}),
                    callback_context=callback_context,
                    gate_payload=gate_out,
                    tools_view=requested_tools_view,
                    timeout_seconds=_approval_timeout_seconds(callback_context),
                )
                decision = _coerce_approval_decision(approval_result)
                scope_constraints = _extract_scope_constraints(approval_result, name)
                if decision == "allow_once":
                    updated_view = _approve_tool_in_view(requested_tools_view, name, mutate=False)
                    updated_view = _apply_tool_constraints_in_view(
                        updated_view,
                        name,
                        scope_constraints,
                        mutate=True,
                    )
                    tools_view_payload = serialize_tools_view(updated_view)
                elif decision == "add_to_scope":
                    _approve_tool_in_view(requested_tools_view, name, mutate=True)
                    _apply_tool_constraints_in_view(
                        requested_tools_view,
                        name,
                        scope_constraints,
                        mutate=True,
                    )
                    scope_ref = _resolve_scope_ref_from_callback_context(callback_context)
                    _persist_approved_tool(scope_ref, name)
                    _persist_scope_constraints(scope_ref, name, scope_constraints)
                    tools_view_payload = serialize_tools_view(requested_tools_view)
                else:
                    return {
                        "status": "ok",
                        "tool_call": {
                            "id": call_id,
                            "name": name,
                            "arguments": dict(arguments or {}),
                            "result": None,
                            "error": f"Execution gated: denied - {str(gate_out.get('reason') or outcome).strip() or outcome}:{name}",
                            "raw": None,
                            "model_format": None,
                            "parse_errors": [],
                            "action": [],
                        },
                    }
            else:
                return {
                    "status": "ok",
                    "tool_call": {
                        "id": call_id,
                        "name": name,
                        "arguments": dict(arguments or {}),
                        "result": None,
                        "error": f"Execution gated: {outcome} - {str(gate_out.get('reason') or outcome).strip() or outcome}:{name}",
                        "raw": None,
                        "model_format": None,
                        "parse_errors": [],
                        "action": [],
                    },
                }
        callback_binding = None
        if callable(callback_processor):
            if not hasattr(self, "_callback_relay"):
                self._callback_relay = _HostedToolCallbackRelay()
            signature = None
            try:
                tool_meta = dict(self.describe().get("tool_metadata") or {}).get(name) or {}
                signature = dict(tool_meta.get("callback_signature") or {}) or None
            except Exception:
                signature = None
            callback_binding = self._callback_relay.bind_session(
                processor=callback_processor,
                toolbox_id=self.toolbox_id,
                tool_name=name,
                tool_call_id=call_id,
                tool_arguments=dict(arguments or {}),
                callback_signature=signature,
                user_context=callback_context,
            )
        try:
            execute_kwargs = {
                "toolbox_id": self.toolbox_id,
                "tool_call": {
                    "id": call_id,
                    "name": name,
                    "arguments": dict(arguments or {}),
                },
                "timeout_seconds": float(timeout_seconds or 30.0),
                "tools_view": tools_view_payload,
                "callback_binding": dict(callback_binding or {}) or None,
            }
            execute_kwargs["execution_request_id"] = durable_request_id
            if isinstance(host_api_approval, dict):
                execute_kwargs["host_api_approval"] = dict(host_api_approval or {})
            return dict(
                self.host.toolbox_execute(
                    **execute_kwargs,
                )
                or {}
            )
        finally:
            if callback_binding and hasattr(self, "_callback_relay"):
                self._callback_relay.release_session(str(callback_binding.get("session_token") or ""))

    def cancel(
        self,
        *,
        tool_name: str = "",
        tool_call_id: str = "",
        timeout_seconds: float = 8.0,
        respawn: bool = True,
        request_id: str = "",
    ) -> Dict[str, Any]:
        cancel_kwargs = {
            "toolbox_id": self.toolbox_id,
            "tool_name": str(tool_name or "").strip(),
            "tool_call_id": str(tool_call_id or "").strip(),
            "timeout_seconds": float(timeout_seconds or 8.0),
            "respawn": bool(respawn),
        }
        if str(request_id or "").strip():
            cancel_kwargs["request_id"] = str(request_id).strip()
        return dict(self.host.toolbox_cancel(**cancel_kwargs) or {})



class PendingHostedToolboxRef:
    def __init__(self, base_ref: HostedToolBoxRef) -> None:
        self.base_ref = base_ref
        self._pending_auto_requests: List[Dict[str, Any]] = []
        self._pending_manual_requests: List[Dict[str, Any]] = []

    def register_auto_callable(
        self,
        *,
        relative_path: str,
        content: str,
        module_name: str,
        callable_name: str,
        environment_name: str = "base",
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        activate: bool = True,
        hidden: bool = False,
        non_restartable: bool = False,
        guide_content: Optional[Dict[str, List[str]]] = None,
        guide_description: Optional[str] = None,
        callback_signature: Optional[Dict[str, Any]] = None,
        concurrency: Optional[Dict[str, Any]] = None,
    ) -> "PendingHostedToolboxRef":
        request = {
            "files": [
                ToolboxBundleFile(
                    relative_path=str(relative_path or "").strip(),
                    content=str(content or ""),
                ).to_runtime_dict()
            ],
            "module_name": str(module_name or "").strip(),
            "callable_name": str(callable_name or "").strip(),
            "sandbox_profile": SandboxProfileSpec(
                environment_name=str(environment_name or "base").strip() or "base",
                required_imports=[str(item or "").strip() for item in list(required_imports or []) if str(item or "").strip()],
                sandbox_policy=dict(sandbox_policy or {}),
            ).to_dict(),
            "activate": bool(activate),
            "hidden": bool(hidden),
            "non_restartable": bool(non_restartable),
            "guide_content": dict(guide_content or {}) or None,
            "guide_description": str(guide_description or "").strip() or None,
            "callback_signature": dict(callback_signature or {}) or None,
        }
        if isinstance(concurrency, dict) and concurrency:
            request["concurrency"] = dict(concurrency)
        self._pending_auto_requests.append(request)
        return self

    def add_auto_callable(self, **kwargs: Any) -> "PendingHostedToolboxRef":
        return self.register_auto_callable(**kwargs)

    def register_python_callable(
        self,
        implementation: Any,
        *,
        environment_name: str = "base",
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        activate: bool = True,
        hidden: bool = False,
        non_restartable: bool = False,
        guide_content: Optional[Dict[str, List[str]]] = None,
        guide_description: Optional[str] = None,
        callback_signature: Optional[Dict[str, Any]] = None,
        concurrency: Optional[Dict[str, Any]] = None,
    ) -> "PendingHostedToolboxRef":
        module = inspect.getmodule(implementation)
        module_name = str(getattr(implementation, "__module__", "") or getattr(module, "__name__", "") or "").strip()
        if not module_name:
            raise ValueError("callable_module_name_required")
        callable_name = str(getattr(implementation, "__name__", "") or "").strip()
        if not callable_name:
            raise ValueError("callable_name_required")
        source_path = inspect.getsourcefile(implementation) or getattr(module, "__file__", None)
        if not source_path:
            raise ValueError("callable_source_file_required")
        source_file = Path(str(source_path)).expanduser().resolve()
        if not source_file.exists():
            raise ValueError("callable_source_file_missing")
        return self.register_auto_callable(
            relative_path=source_file.name,
            content=source_file.read_text(encoding="utf-8"),
            module_name=module_name,
            callable_name=callable_name,
            environment_name=environment_name,
            required_imports=required_imports,
            sandbox_policy=sandbox_policy,
            activate=activate,
            hidden=hidden,
            non_restartable=non_restartable,
            guide_content=guide_content,
            guide_description=guide_description,
            callback_signature=callback_signature,
            concurrency=concurrency,
        )

    def add_python_callable(self, implementation: Any, **kwargs: Any) -> "PendingHostedToolboxRef":
        return self.register_python_callable(implementation, **kwargs)

    def register_manual_tool(
        self,
        tool_definition: Dict[str, Any],
        implementation: Any,
        *,
        environment_name: str = "base",
        required_imports: Optional[Sequence[str]] = None,
        sandbox_policy: Optional[Dict[str, Any]] = None,
        hidden: bool = False,
        non_restartable: bool = False,
        callback_signature: Optional[Dict[str, Any]] = None,
    ) -> "PendingHostedToolboxRef":
        module = inspect.getmodule(implementation)
        module_name = str(getattr(implementation, "__module__", "") or getattr(module, "__name__", "") or "").strip()
        if not module_name:
            raise ValueError("callable_module_name_required")
        callable_name = str(getattr(implementation, "__name__", "") or "").strip()
        if not callable_name:
            raise ValueError("callable_name_required")
        source_path = inspect.getsourcefile(implementation) or getattr(module, "__file__", None)
        if not source_path:
            raise ValueError("callable_source_file_required")
        source_file = Path(str(source_path)).expanduser().resolve()
        if not source_file.exists():
            raise ValueError("callable_source_file_missing")
        
        request = {
            "files": [
                ToolboxBundleFile(
                    relative_path=source_file.name,
                    content=source_file.read_text(encoding="utf-8"),
                ).to_runtime_dict()
            ],
            "module_name": module_name,
            "callable_name": callable_name,
            "tool_definition": dict(tool_definition or {}),
            "sandbox_profile": SandboxProfileSpec(
                environment_name=str(environment_name or "base").strip() or "base",
                required_imports=[str(item or "").strip() for item in list(required_imports or []) if str(item or "").strip()],
                sandbox_policy=dict(sandbox_policy or {}),
            ).to_dict(),
            "hidden": bool(hidden),
            "non_restartable": bool(non_restartable),
            "callback_signature": dict(callback_signature or {}) or None,
        }
        self._pending_manual_requests.append(request)
        return self

    def add_manual_tool(self, tool_definition: Dict[str, Any], implementation: Any, **kwargs: Any) -> "PendingHostedToolboxRef":
        return self.register_manual_tool(tool_definition, implementation, **kwargs)

    def resolve_sandbox(self) -> HostedToolBoxRef:
        if self._pending_auto_requests:
            self.base_ref.host.toolbox_register_auto(
                toolbox_id=self.base_ref.toolbox_id,
                requests=list(self._pending_auto_requests),
                python_executable=self.base_ref.python_executable,
                worker_profile_class=self.base_ref.worker_profile_class,
            )
        if self._pending_manual_requests:
            self.base_ref.host.toolbox_register_manual(
                toolbox_id=self.base_ref.toolbox_id,
                requests=list(self._pending_manual_requests),
                python_executable=self.base_ref.python_executable,
                worker_profile_class=self.base_ref.worker_profile_class,
            )
        self._pending_auto_requests.clear()
        self._pending_manual_requests.clear()
        return self.base_ref

SandboxedToolboxFacade = HostedToolBoxRef
