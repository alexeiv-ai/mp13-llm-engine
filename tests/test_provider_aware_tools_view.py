from __future__ import annotations

from app.context_cursor import ChatContext
from app.engine_session import EngineSession, InferenceParams
from mp13_engine.mp13_toolbox import Toolbox
from mp13_engine.mp13_toolbox import ToolsScope, ToolsView


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
