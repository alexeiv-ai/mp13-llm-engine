"""Hosted toolbox reference facade for complete definitions and execution."""
from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from mp13_engine.mp13_toolbox import ToolBoxRef, ToolsView

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
    def __init__(self, *, toolbox_id: str, host: Any) -> None:
        self.toolbox_id = str(toolbox_id or "").strip()
        if not self.toolbox_id:
            raise ValueError("toolbox_id_required")
        self.host = host

    @property
    def ref_name(self) -> str:
        return self.toolbox_id

    def _host_descriptor(self) -> Dict[str, Any]:
        host = self.host
        descriptor: Dict[str, Any] = {"host_type": type(host).__name__}
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
        return {"toolbox_id": self.toolbox_id, "host": self._host_descriptor()}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], *, host: Any = None) -> "HostedToolBoxRef":
        row = dict(payload or {})
        if {"python_executable", "worker_profile_class"} & set(row):
            raise ValueError("legacy_toolbox_runtime_selector_rejected")
        resolved_host = host
        if resolved_host is None:
            host_row = dict(row.get("host") or {})
            kind = str(host_row.get("kind") or "").strip().lower()
            if kind == "control_channel":
                from ..engine_host_channel import EngineHostControlChannel

                resolved_host = EngineHostControlChannel(dict(host_row.get("control_settings") or {}))
            elif kind == "service":
                from ..service.host_service import EngineHostService
                from ..hosting_configuration import load_hosting_configuration

                engines_state_raw = str(host_row.get("engines_state_file") or "").strip()
                mp13_config_raw = str(host_row.get("mp13_config_file") or "").strip()
                resolved_host = EngineHostService(
                    engines_state_file=Path(engines_state_raw) if engines_state_raw else None,
                    hosting_configuration=load_hosting_configuration(
                        Path(mp13_config_raw) if mp13_config_raw else None
                    ),
                )
            else:
                raise ValueError("host_required_for_hosted_toolbox_ref_deserialization")
        return cls(toolbox_id=str(row.get("toolbox_id") or "").strip(), host=resolved_host)

    def get_definition(self, *, operator_details: bool = False) -> Dict[str, Any]:
        return dict(self.host.toolbox_get_definition(
            toolbox_id=self.toolbox_id,
            operator_details=bool(operator_details),
        ) or {})

    def plan_definition(
        self,
        definition: Dict[str, Any],
        *,
        request_id: str,
        operator_details: bool = False,
        ttl_ms: int = 15 * 60 * 1000,
    ) -> Dict[str, Any]:
        return dict(self.host.toolbox_plan_definition(
            definition=dict(definition or {}),
            request_id=str(request_id or "").strip(),
            operator_details=bool(operator_details),
            ttl_ms=int(ttl_ms),
        ) or {})

    def plan_tool_changes(
        self,
        changes: list[Dict[str, Any]],
        *,
        expected_revision: str | None,
        request_id: str,
        operator_details: bool = False,
    ) -> Dict[str, Any]:
        return dict(self.host.toolbox_plan_tool_changes(
            toolbox_id=self.toolbox_id,
            expected_revision=expected_revision,
            changes=[dict(item) for item in changes],
            request_id=str(request_id or "").strip(),
            operator_details=bool(operator_details),
        ) or {})

    def confirm_definition_plan(
        self,
        *,
        plan_id: str,
        environment_choices: list[Dict[str, Any]],
        request_id: str,
    ) -> Dict[str, Any]:
        return dict(self.host.toolbox_confirm_definition_plan(
            plan_id=str(plan_id or "").strip(),
            environment_choices=[dict(item) for item in environment_choices],
            request_id=str(request_id or "").strip(),
        ) or {})

    def apply_definition(
        self,
        *,
        plan_id: str,
        confirmation_ref: str,
        request_id: str,
        dependency_approval_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        return dict(self.host.toolbox_apply_definition(
            plan_id=str(plan_id or "").strip(),
            confirmation_ref=str(confirmation_ref or "").strip(),
            request_id=str(request_id or "").strip(),
            dependency_approval_ref=dependency_approval_ref,
        ) or {})

    def list_environment_templates(self) -> Dict[str, Any]:
        return dict(self.host.toolbox_template_list() or {})

    def describe_environment_template(
        self, *, template_id: str, template_digest: Optional[str] = None
    ) -> Dict[str, Any]:
        return dict(self.host.toolbox_template_describe(
            template_id=str(template_id or "").strip(),
            template_digest=str(template_digest or "").strip() or None,
        ) or {})

    def describe(self, *, timeout_seconds: float = 10.0) -> Dict[str, Any]:
        return dict(self.host.toolbox_describe(
            toolbox_id=self.toolbox_id,
            timeout_seconds=float(timeout_seconds or 10.0),
        ) or {})

    def gate(self, *, tool_name: str, tools_view: Optional[ToolsView] = None) -> Dict[str, Any]:
        return dict(self.host.toolbox_gate(
            toolbox_id=self.toolbox_id,
            tool_name=str(tool_name or "").strip(),
            tools_view=serialize_tools_view(tools_view),
        ) or {})

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
                        updated_view, name, scope_constraints, mutate=True
                    )
                    tools_view_payload = serialize_tools_view(updated_view)
                elif decision == "add_to_scope":
                    _approve_tool_in_view(requested_tools_view, name, mutate=True)
                    _apply_tool_constraints_in_view(
                        requested_tools_view, name, scope_constraints, mutate=True
                    )
                    durable_scope = _resolve_scope_ref_from_callback_context(callback_context)
                    _persist_approved_tool(durable_scope, name)
                    _persist_scope_constraints(durable_scope, name, scope_constraints)
                    tools_view_payload = serialize_tools_view(requested_tools_view)
                else:
                    return self._gated_result(call_id, name, arguments, "denied", gate_out)
            else:
                return self._gated_result(call_id, name, arguments, outcome, gate_out)
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
            execute_kwargs: Dict[str, Any] = {
                "toolbox_id": self.toolbox_id,
                "tool_call": {"id": call_id, "name": name, "arguments": dict(arguments or {})},
                "timeout_seconds": float(timeout_seconds or 30.0),
                "tools_view": tools_view_payload,
                "callback_binding": dict(callback_binding or {}) or None,
                "execution_request_id": durable_request_id,
            }
            if isinstance(host_api_approval, dict):
                execute_kwargs["host_api_approval"] = dict(host_api_approval)
            return dict(self.host.toolbox_execute(**execute_kwargs) or {})
        finally:
            if callback_binding and hasattr(self, "_callback_relay"):
                self._callback_relay.release_session(str(callback_binding.get("session_token") or ""))

    @staticmethod
    def _gated_result(
        call_id: str,
        name: str,
        arguments: Optional[Dict[str, Any]],
        outcome: str,
        gate_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        reason = str(gate_out.get("reason") or outcome).strip() or outcome
        return {
            "status": "ok",
            "tool_call": {
                "id": call_id,
                "name": name,
                "arguments": dict(arguments or {}),
                "result": None,
                "error": f"Execution gated: {outcome} - {reason}:{name}",
                "raw": None,
                "model_format": None,
                "parse_errors": [],
                "action": [],
            },
        }

    def cancel(
        self,
        *,
        operation_ref: Dict[str, Any],
        reason: str = "client_requested",
        timeout_seconds: float = 8.0,
        respawn: bool = True,
    ) -> Dict[str, Any]:
        return dict(self.host.hosted_operation_cancel(
            ref=dict(operation_ref or {}),
            reason=str(reason or "client_requested").strip() or "client_requested",
            timeout_seconds=float(timeout_seconds or 8.0),
            respawn=bool(respawn),
        ) or {})


SandboxedToolboxFacade = HostedToolBoxRef
