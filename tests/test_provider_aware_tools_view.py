from __future__ import annotations

import pytest

from app.context_cursor import ChatContext
from app.engine_session import EngineSession, InferenceParams
from mp13_engine.mp13_toolbox import Toolbox
from mp13_engine.mp13_toolbox import ToolsScope, ToolsView
from mp13_engine.tool_round import normalize_server_tool_events


def test_tools_view_round_trips_provider_tools_and_resolution_metadata():
    view = ToolsView(
        view_id="context:ctx-main",
        mode="advertised",
        allowed_tools={"project_search"},
        advertised_tools={"project_search"},
        hidden_allowed_tools=set(),
        disabled_tools=set(),
        server_tools=[{"type": "web_search"}],
        view_digest="sha256:" + "1" * 64,
        profile_id="research",
        profile_revision=3,
        scope_stack=[{"operation": "add", "source": "user"}],
        unavailable_members=[
            {"member_id": "server:grok/x_search@default", "state": "incompatible"}
        ],
    )

    payload = view.to_dict()
    assert payload["advertised_tools"] == ["project_search"]
    assert payload["server_tools"] == [{"type": "web_search"}]
    assert payload["view_digest"] == "sha256:" + "1" * 64
    assert payload["profile_id"] == "research"


def test_mp13_request_builder_consumes_each_application_resolved_provider_view():
    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    context = ChatContext(session, chat_session=chat_session, toolbox=Toolbox())
    cursor = context.active_cursor
    cursor.add_user("research across the active provider")
    provider_tools = {
        "openai": {"type": "web_search"},
        "grok": {"type": "x_search"},
        "openai_compatible": {"type": "web_lookup"},
    }

    for index, (provider_id, server_tool) in enumerate(provider_tools.items(), 1):
        inactive = [
            {
                "member_id": f"server:{other_provider}/{other_tool['type']}@default",
                "state": "incompatible",
            }
            for other_provider, other_tool in provider_tools.items()
            if other_provider != provider_id
        ]
        view = ToolsView(
            view_id="context:ctx-portable",
            mode="advertised",
            allowed_tools=set(),
            advertised_tools=set(),
            hidden_allowed_tools=set(),
            disabled_tools=set(),
            server_tools=[server_tool],
            view_digest="sha256:" + str(index) * 64,
            profile_id="portable-research",
            profile_revision=4,
            unavailable_members=inactive,
        )
        cursor.set_tools_view(view)

        request, _adapters, consumed_view = cursor.build_inference_request()

        assert request["tools"] == [server_tool]
        assert request["tool_view_digest"] == view.view_digest
        assert consumed_view is view
        assert consumed_view.profile_id == "portable-research"
        assert consumed_view.unavailable_members == inactive

    assert cursor.tools_view is view


def test_tools_scope_round_trips_an_opaque_canonical_layer_for_cursor_replay():
    scope = ToolsScope(
        advertise_tools={"project_search"},
        canonical_layer={
            "operation_id": "toolscope:add-001",
            "operation": "add",
            "layer_digest": "sha256:" + "2" * 64,
        },
    ).clean()
    restored = ToolsScope.from_dict(scope.to_dict())
    assert restored.advertise_tools == {"project_search"}
    assert restored.canonical_layer == scope.canonical_layer
    assert restored.is_noop() is False


def _canonical_scope(operation_id: str, tool_name: str) -> ToolsScope:
    return ToolsScope(
        advertise_tools={tool_name},
        canonical_layer={
            "operation_id": operation_id,
            "operation": "add",
            "layer_digest": "sha256:" + operation_id[-1] * 64,
        },
    ).clean()


def test_cursor_fork_inherits_canonical_scope_then_diverges_independently():
    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    context = ChatContext(session, chat_session=chat_session, toolbox=Toolbox())
    original = context.active_cursor
    original.add_user("start")
    original.apply_tools_scope(
        "add",
        _canonical_scope("toolscope:fork-1", "project_search"),
        command_text="add project search",
    )
    original.add_assistant("scope established")

    fork = original.clone()
    fork.apply_tools_scope(
        "add",
        _canonical_scope("toolscope:fork-2", "web_search"),
        command_text="add web search on fork",
    )

    assert [
        scope.canonical_layer["operation_id"]
        for scope in original.get_effective_tools_scopes()
    ] == ["toolscope:fork-1"]
    assert [
        scope.canonical_layer["operation_id"]
        for scope in fork.get_effective_tools_scopes()
    ] == ["toolscope:fork-1", "toolscope:fork-2"]


def test_session_reload_replays_canonical_scope_layer():
    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    context = ChatContext(session, chat_session=chat_session, toolbox=Toolbox())
    cursor = context.active_cursor
    cursor.add_user("persist this scope")
    cursor.apply_tools_scope(
        "add",
        _canonical_scope("toolscope:reload-3", "project_search"),
        command_text="persist project search",
    )
    chat_session.last_active_turn = cursor.head

    restored = EngineSession.from_dict(session.to_dict_prop)
    restored_chat = restored.conversations[0]
    scopes = restored.get_effective_tools_scopes(restored_chat.last_active_turn)

    assert len(scopes) == 1
    assert scopes[0].advertise_tools == {"project_search"}
    assert scopes[0].canonical_layer["operation_id"] == "toolscope:reload-3"


def test_scope_stack_set_add_targeted_pop_and_reset_preserve_context_root():
    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    toolbox = Toolbox()
    for name in ("context_search", "project_search", "web_search"):
        def _tool(_name=name):
            return {"tool": _name}

        _tool.__name__ = name
        ok, message = toolbox.add_tool_callable(_tool, activate=True)
        assert ok, message
    context = ChatContext(session, chat_session=chat_session, toolbox=toolbox)
    context.toolbox_ref.set_scope(ToolsScope(advertise_tools={"context_search"}))
    cursor = context.active_cursor
    cursor.add_user("layer tools")

    cursor.apply_tools_scope(
        "set",
        _canonical_scope("toolscope:set-4", "project_search"),
        command_text="set project tools",
    )
    cursor.apply_tools_scope(
        "add",
        _canonical_scope("toolscope:add-5", "web_search"),
        command_text="add web search",
    )
    add_stack_id = str(cursor.current_turn.cmd[-1].data.get("stack_id") or "")
    assert [
        scope.canonical_layer["operation_id"]
        for scope in cursor.get_effective_tools_scopes()
    ] == ["toolscope:set-4", "toolscope:add-5"]

    cursor.apply_tools_scope("pop", stack_id=add_stack_id, command_text="pop web search")
    assert [
        scope.canonical_layer["operation_id"]
        for scope in cursor.get_effective_tools_scopes()
    ] == ["toolscope:set-4"]
    assert "context_search" in cursor.get_tools_view().advertised_tools

    cursor.apply_tools_scope("reset", command_text="reset branch tools")
    assert cursor.get_effective_tools_scopes() == []
    assert "context_search" in cursor.get_tools_view().advertised_tools


def test_next_round_metadata_and_server_events_survive_session_reload():
    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    context = ChatContext(session, chat_session=chat_session, toolbox=Toolbox())
    cursor = context.active_cursor
    cursor.add_user("use temporary search")
    scope = _canonical_scope("toolscope:round-6", "web_search")
    scope.canonical_layer.update(
        {"lifetime": "next_round", "expires_after_round": 1, "profile_revision": 7}
    )
    cursor.apply_tools_scope("add", scope, command_text="use search next round")
    cursor.add_assistant(
        "done",
        server_tool_events=[
            {
                "schema_version": "server_tool.event.v1",
                "kind": "server_tool_call",
                "provider_id": "openai",
                "tool_id": "web_search",
                "item_type": "web_search_call",
                "status": "completed",
            }
        ],
    )
    chat_session.last_active_turn = cursor.head

    restored = EngineSession.from_dict(session.to_dict_prop)
    restored_turn = restored.conversations[0].last_active_turn
    restored_scope = restored.get_effective_tools_scopes(restored_turn)[0]
    assert restored_scope.canonical_layer["lifetime"] == "next_round"
    assert restored_scope.canonical_layer["expires_after_round"] == 1
    assert restored_scope.canonical_layer["profile_revision"] == 7
    assert restored_turn.data["assistant"]["server_tool_events"] == [
        {
            "schema_version": "server_tool.event.v1",
            "kind": "server_tool_call",
            "provider_id": "openai",
            "tool_id": "web_search",
            "item_type": "web_search_call",
            "status": "completed",
        }
    ]


def test_server_event_contract_rejects_provider_native_payload_fields():
    with pytest.raises(ValueError, match="unsupported fields: raw"):
        normalize_server_tool_events(
            [
                {
                    "provider_id": "openai",
                    "tool_id": "web_search",
                    "item_type": "web_search_call",
                    "raw": {"query": "must not persist"},
                }
            ]
        )
