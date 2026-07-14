"""Application-injected bridge for canonical ToolSearch control operations."""

from __future__ import annotations

import asyncio
import inspect
import re
import uuid
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional


ControlDispatch = Callable[..., Dict[str, Any] | Awaitable[Dict[str, Any]]]
TOOL_CONTEXT_CONTROL_CALLBACK = "tool_context_control_dispatch"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value or {})
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dict(dumped or {}) if isinstance(dumped, Mapping) else {}
    return {}


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def make_tool_context_control_handlers(
    control_dispatch: ControlDispatch,
    *,
    workspace_id: str = "",
) -> Dict[str, Callable[..., Awaitable[Dict[str, Any]]]]:
    """Build engine control handlers backed by an application-owned dispatcher."""

    if not callable(control_dispatch):
        raise TypeError("control_dispatch must be callable")

    async def toolbox_search_and_scope(
        *,
        tool_call: Any,
        cursor: Any,
        tools_view: Any,
        callback_processor: Optional[Callable[..., Any]] = None,
        callback_context: Any = None,
    ) -> Dict[str, Any]:
        arguments = _dict(getattr(tool_call, "arguments", None))
        query = _clean(arguments.get("query"))
        reason = _clean(arguments.get("reason"))
        if not query or not reason:
            raise ValueError("query and reason are required")
        supplied_context = _dict(callback_context)
        context_id = _clean(supplied_context.get("context_id"))
        cursor_id = _clean(supplied_context.get("cursor_id") or getattr(cursor, "context_id", ""))
        if not context_id or not cursor_id:
            raise RuntimeError("application callback context must bind context_id and cursor_id")
        bound_context = {
            **supplied_context,
            "context_id": context_id,
            "cursor_id": cursor_id,
        }
        common = {
            "workspace_id": _clean(workspace_id),
            "context_id": context_id,
            "cursor_id": cursor_id,
        }

        async def dispatch(method: str, method_arguments: Dict[str, Any]) -> Dict[str, Any]:
            response = await _await(
                control_dispatch(
                    method=method,
                    arguments={**common, **method_arguments},
                    context=bound_context,
                )
            )
            payload = _dict(response)
            if payload.get("status") != "ok":
                raise RuntimeError(_clean(payload.get("message") or payload.get("reason")) or "tool control failed")
            return _dict(payload.get("result"))

        max_tools = max(1, min(int(arguments.get("max_tools") or 4), 8))
        search = await dispatch(
            "tools.catalog.search",
            {
                "query": query,
                "provider_id": _clean(arguments.get("provider_id")),
                "group": _clean(arguments.get("group")),
                "risk": _clean(arguments.get("risk")),
                "limit": max_tools * 2,
            },
        )
        selected = []
        for item in list(search.get("items") or []):
            row = _dict(item)
            canonical_id = _clean(row.get("canonical_id"))
            if not canonical_id or canonical_id == "control:toolbox_search_and_scope":
                continue
            if canonical_id.startswith("server:") and "@" not in canonical_id:
                continue
            if _clean(row.get("configuration_state")) in {"unconfigured", "invalid"}:
                continue
            if _clean(row.get("availability") or "available") != "available":
                continue
            selected.append(canonical_id)
            if len(selected) >= max_tools:
                break
        if not selected:
            return {
                "status": "no_candidates",
                "query": query,
                "candidate_digest": _clean(search.get("candidate_digest")),
                "selected_members": [],
            }

        call_id = re.sub(r"[^A-Za-z0-9_.:-]+", "-", _clean(getattr(tool_call, "id", "")) or uuid.uuid4().hex).strip("-")
        proposal_result = await dispatch(
            "tools.scope.propose",
            {
                "operation_id": f"toolscope:search-{call_id}"[:128],
                "operation": "add",
                "lifetime": _clean(arguments.get("lifetime")) or "next_round",
                "reason": reason,
                "expected_view_revision": _clean(getattr(tools_view, "view_digest", "")),
                "requested_member_ids": selected,
            },
        )
        proposal = _dict(proposal_result.get("proposal"))
        if not proposal:
            raise RuntimeError("tool control proposal was not returned")
        if not callable(callback_processor):
            return {
                "status": "proposal_requires_approval",
                "query": query,
                "selected_members": selected,
                "proposal": proposal,
            }
        approval = await _await(
            callback_processor(
                callback_name="tool_requires_confirmation",
                payload={
                    "kind": "tool_approval_request",
                    "tool_name": "toolbox_search_and_scope",
                    "tool_call_id": _clean(getattr(tool_call, "id", "")),
                    "reason": reason,
                    "arguments": {
                        "selected_members": selected,
                        "candidate_digest": _clean(search.get("candidate_digest")),
                    },
                },
                context=bound_context,
            )
        )
        decision = _clean(_dict(approval).get("decision")).lower() or "deny"
        if decision not in {"allow_once", "add_to_scope"}:
            return {"status": "proposal_denied", "decision": decision, "selected_members": selected}
        applied = await dispatch(
            "tools.scope.apply",
            {"proposal": proposal, "approval_decision": decision},
        )
        return {
            "status": "scope_applied",
            "decision": decision,
            "selected_members": selected,
            "proposal": proposal,
            **applied,
        }

    return {"toolbox_search_and_scope": toolbox_search_and_scope}


def make_tool_context_control_handlers_from_binding(
    callback_binding: Mapping[str, Any],
    *,
    workspace_id: str = "",
) -> Dict[str, Callable[..., Awaitable[Dict[str, Any]]]]:
    """Build handlers whose canonical dispatcher lives in another process."""

    binding = dict(callback_binding or {})
    if not _clean(binding.get("address")) or not _clean(binding.get("session_token")):
        raise ValueError("tool context callback binding is incomplete")

    async def dispatch(*, method: str, arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        from hosting.toolbox_executor_ipc import _invoke_callback_binding

        response = await asyncio.to_thread(
            _invoke_callback_binding,
            binding,
            callback_name=TOOL_CONTEXT_CONTROL_CALLBACK,
            payload={
                "method": _clean(method),
                "arguments": dict(arguments or {}),
                "context": dict(context or {}),
            },
            context=dict(context or {}),
        )
        result = response.get("result")
        return _dict(result)

    return make_tool_context_control_handlers(dispatch, workspace_id=workspace_id)


class ToolContextControlCallbackTransport:
    """Application-side authenticated callback binding for a remote chat round."""

    def __init__(self, control_dispatch: ControlDispatch) -> None:
        if not callable(control_dispatch):
            raise TypeError("control_dispatch must be callable")
        from hosting.toolbox.callbacks import _HostedToolCallbackRelay

        self._relay = _HostedToolCallbackRelay()
        self._binding = self._relay.bind_session(
            processor=self._processor,
            toolbox_id="application-tool-context",
            tool_name="toolbox_search_and_scope",
            tool_call_id="control-transport",
            callback_signature={
                "callbacks": [
                    {
                        "name": TOOL_CONTEXT_CONTROL_CALLBACK,
                        "payload_type": "object",
                    }
                ]
            },
        )
        self._control_dispatch = control_dispatch

    @property
    def binding(self) -> Dict[str, Any]:
        return dict(self._binding)

    def _processor(self, *, callback_name: str, payload: Any, context: Any) -> Any:
        if _clean(callback_name) != TOOL_CONTEXT_CONTROL_CALLBACK:
            return {"status": "error", "message": "unsupported_tool_context_callback"}
        request = _dict(payload)
        return self._control_dispatch(
            method=_clean(request.get("method")),
            arguments=_dict(request.get("arguments")),
            context=_dict(request.get("context")),
        )

    def close(self) -> None:
        self._relay.release_session(_clean(self._binding.get("session_token")))


__all__ = [
    "TOOL_CONTEXT_CONTROL_CALLBACK",
    "ToolContextControlCallbackTransport",
    "make_tool_context_control_handlers",
    "make_tool_context_control_handlers_from_binding",
]
