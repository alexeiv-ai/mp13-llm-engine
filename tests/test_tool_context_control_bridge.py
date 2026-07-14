from types import SimpleNamespace

import pytest

from app.tool_context_control_bridge import make_tool_context_control_handlers


@pytest.mark.asyncio
async def test_toolsearch_bridge_uses_canonical_propose_approval_apply_path():
    calls = []
    approvals = []

    async def dispatch(*, method, arguments, context):
        calls.append((method, dict(arguments), dict(context)))
        if method == "tools.catalog.search":
            return {
                "status": "ok",
                "result": {
                    "candidate_digest": "digest-1",
                    "items": [
                        {"canonical_id": "control:toolbox_search_and_scope", "availability": "available"},
                        {"canonical_id": "server:openai/file_search", "availability": "available", "configuration_state": "unconfigured"},
                        {"canonical_id": "server:openai/file_search@docs", "availability": "available", "configuration_state": "configured"},
                        {"canonical_id": "local:project_files.search", "availability": "available", "configuration_state": "not_required"},
                    ],
                },
            }
        if method == "tools.scope.propose":
            return {"status": "ok", "result": {"proposal": {"operation_id": arguments["operation_id"], "requested_member_ids": arguments["requested_member_ids"]}}}
        if method == "tools.scope.apply":
            return {"status": "ok", "result": {"receipt": {"status": "applied", "operation_id": arguments["proposal"]["operation_id"]}}}
        raise AssertionError(method)

    async def approve(*, callback_name, payload, context):
        approvals.append((callback_name, payload, context))
        return {"decision": "allow_once"}

    handler = make_tool_context_control_handlers(dispatch, workspace_id="workspace-1")["toolbox_search_and_scope"]
    result = await handler(
        tool_call=SimpleNamespace(id="call-1", arguments={"query": "docs", "reason": "Find docs", "max_tools": 2}),
        cursor=SimpleNamespace(context_id="engine-cursor"),
        tools_view=SimpleNamespace(view_digest="view-1"),
        callback_processor=approve,
        callback_context={"context_id": "chat-1", "cursor_id": "cursor-1", "source_turn_id": "turn-1"},
    )

    assert [item[0] for item in calls] == ["tools.catalog.search", "tools.scope.propose", "tools.scope.apply"]
    assert calls[1][1]["requested_member_ids"] == [
        "server:openai/file_search@docs",
        "local:project_files.search",
    ]
    assert calls[1][1]["expected_view_revision"] == "view-1"
    assert all(item[1]["context_id"] == "chat-1" and item[1]["cursor_id"] == "cursor-1" for item in calls)
    assert approvals[0][0] == "tool_requires_confirmation"
    assert approvals[0][1]["kind"] == "tool_approval_request"
    assert result["status"] == "scope_applied"
    assert result["receipt"]["status"] == "applied"


@pytest.mark.asyncio
async def test_toolsearch_bridge_denial_does_not_apply_and_missing_binding_fails_closed():
    methods = []

    def dispatch(*, method, arguments, context):
        methods.append(method)
        if method == "tools.catalog.search":
            return {"status": "ok", "result": {"items": [{"canonical_id": "local:echo", "availability": "available"}]}}
        return {"status": "ok", "result": {"proposal": {"operation_id": "toolscope:search-call-2"}}}

    handler = make_tool_context_control_handlers(dispatch)["toolbox_search_and_scope"]
    denied = await handler(
        tool_call=SimpleNamespace(id="call-2", arguments={"query": "echo", "reason": "Need echo"}),
        cursor=SimpleNamespace(context_id="engine-cursor"),
        tools_view=SimpleNamespace(view_digest="view-2"),
        callback_processor=lambda **_kwargs: {"decision": "deny"},
        callback_context={"context_id": "chat-2", "cursor_id": "cursor-2"},
    )
    assert denied["status"] == "proposal_denied"
    assert methods == ["tools.catalog.search", "tools.scope.propose"]

    with pytest.raises(RuntimeError, match="must bind context_id and cursor_id"):
        await handler(
            tool_call=SimpleNamespace(id="call-3", arguments={"query": "echo", "reason": "Need echo"}),
            cursor=SimpleNamespace(context_id="engine-cursor"),
            tools_view=SimpleNamespace(view_digest="view-3"),
            callback_context={},
        )
