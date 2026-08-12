from __future__ import annotations

import contextlib
import io
import re
import threading
import time
from typing import Any, Dict, List

import app
from app.hosted_toolbox_api import (
    HostedToolExecutionRouter,
    HostedToolboxAttachment,
    attach_existing_hosted_toolbox,
    create_hosted_control_channel,
    create_hosted_toolbox_executor,
    create_hosted_toolbox_ref,
    is_hosted_tool_call_canceled,
    should_resubmit_hosted_tool_call,
)
from app.hosted_tool_runtime import execute_tool_round_on_cursor, summarize_canceled_tool_calls
from app.hosted_tool_visibility import summarize_effective_tool_view
from app.mp13chat_tools_cli import LightweightToolsCliHandler
from app.context_cursor import ChatContext
from app.engine_session import EngineSession, InferenceParams
from hosting.engine_host_channel import EngineHostControlChannel
from hosting import HostedToolBoxRef
from hosting.toolbox_harness import ToolboxExecutionHarness
from mp13_engine.mp13_config import InferenceResponse, ToolCall, ToolCallBlock
from mp13_engine.mp13_tools_parser import DEFAULT_PROFILE
from mp13_engine.mp13_toolbox import Toolbox, ToolsScope, ToolsView


def test_app_package_exports_hosted_attach_helpers() -> None:
    assert app.attach_existing_hosted_toolbox is attach_existing_hosted_toolbox
    assert app.create_hosted_control_channel is create_hosted_control_channel
    assert app.create_hosted_toolbox_ref is create_hosted_toolbox_ref
    assert app.create_hosted_toolbox_executor is create_hosted_toolbox_executor
    assert not hasattr(app, "register_hosted_tool_callable")
    assert app.HostedToolboxAttachment is HostedToolboxAttachment


def test_create_hosted_toolbox_ref_returns_public_hosted_ref() -> None:
    class _FakeHost:
        pass

    ref = create_hosted_toolbox_ref(
        host=_FakeHost(),
        toolbox_id="user-tools",
    )

    assert isinstance(ref, HostedToolBoxRef)
    assert ref.toolbox_id == "user-tools"
    assert ref.ref_name == "user-tools"


def test_create_hosted_toolbox_executor_returns_sandbox_harness() -> None:
    class _FakeChannel:
        pass

    harness = create_hosted_toolbox_executor(
        control_channel=_FakeChannel(),
        toolbox_id="user-tools",
        engine_ids=["sandbox-a", "sandbox-b"],
    )

    assert isinstance(harness, ToolboxExecutionHarness)
    assert harness.config.mode == "sandbox"
    assert harness.config.sandbox_toolbox_id == "user-tools"
    assert harness.config.sandbox_engine_ids == ["sandbox-a", "sandbox-b"]


def test_create_hosted_control_channel_builds_local_channel_settings() -> None:
    channel = create_hosted_control_channel(
        engines_state_file="managed_engines.json",
        control_state_file="control_state.json"
        timeout_seconds=9.5,
        auto_bootstrap=False,
    )

    assert isinstance(channel, EngineHostControlChannel)
    assert channel.control_settings["engine_host_state_file"] == "managed_engines.json"
    assert channel.control_settings["engine_host_control_state_file"] == "control_state.json"
    assert channel.control_settings["engine_host_timeout_seconds"] == 9.5
    assert channel.control_settings["engine_host_daemon_auto_bootstrap"] is False


def test_attach_existing_hosted_toolbox_returns_wrapper_ready_attachment() -> None:
    attachment = attach_existing_hosted_toolbox(
        toolbox_id="user-tools",
        engines_state_file="managed_engines.json",
        control_state_file="control_state.json"
        timeout_seconds=9.5,
        auto_bootstrap=False,
    )

    assert isinstance(attachment, HostedToolboxAttachment)
    assert isinstance(attachment.control_channel, EngineHostControlChannel)
    assert isinstance(attachment.toolbox_ref, HostedToolBoxRef)
    assert isinstance(attachment.executor, ToolboxExecutionHarness)
    assert attachment.toolbox_ref.toolbox_id == "user-tools"
    assert attachment.executor.config.mode == "sandbox"
    assert attachment.executor.config.sandbox_toolbox_id == "user-tools"
    assert attachment.summary["mode"] == "sandbox"
    assert attachment.summary["advertised_tool_names"] == []


def test_hosted_cancel_helpers_interpret_non_restartable_default_retry_policy() -> None:
    canceled_call = {"name": "hello_tool", "error": "Execution canceled: sandbox_recycled:hello_tool"}
    failed_call = {"name": "hello_tool", "error": "Execution failed: RuntimeError - boom"}

    assert is_hosted_tool_call_canceled(canceled_call) is True
    assert should_resubmit_hosted_tool_call(canceled_call, non_restartable=False) is True
    assert should_resubmit_hosted_tool_call(canceled_call, non_restartable=True) is False
    assert is_hosted_tool_call_canceled(failed_call) is False
    assert should_resubmit_hosted_tool_call(failed_call, non_restartable=False) is False


def test_hosted_tool_execution_router_switches_between_native_and_hosted_modes() -> None:
    class _FakeChannel:
        def toolbox_describe(self, **kwargs):
            return {
                "status": "ok",
                "all_registered_tool_names": ["hello_remote", "hidden_remote"],
                "advertised_tool_names": ["hello_remote"],
                "hidden_allowed_tool_names": ["hidden_remote"],
            }

    native_toolbox = object()
    router = HostedToolExecutionRouter()

    native_executor = router.active_executor(native_toolbox)
    assert isinstance(native_executor, ToolboxExecutionHarness)
    assert native_executor.native_toolbox is native_toolbox
    assert native_executor.config.mode == "native"

    hosted_executor = router.configure_hosted_execution(
        control_channel=_FakeChannel(),
        toolbox_id="user-tools",
        engine_ids=["sandbox-a"],
    )
    active_hosted = router.active_executor(native_toolbox)
    assert active_hosted is hosted_executor
    assert active_hosted.native_toolbox is native_toolbox
    assert active_hosted.config.mode == "sandbox"
    assert active_hosted.config.sandbox_toolbox_id == "user-tools"
    assert router.hosted_advertised_tool_names() == ["hello_remote"]
    summary = router.hosted_toolbox_summary()
    assert summary is not None
    assert summary["all_registered_tool_names"] == ["hello_remote", "hidden_remote"]
    assert summary["advertised_tool_names"] == ["hello_remote"]
    assert summary["hidden_allowed_tool_names"] == ["hidden_remote"]

    router.clear_hosted_execution()
    native_again = router.active_executor(native_toolbox)
    assert isinstance(native_again, ToolboxExecutionHarness)
    assert native_again is not hosted_executor
    assert native_again.config.mode == "native"
    assert router.hosted_advertised_tool_names() is None


def test_hosted_tool_execution_router_prefers_explicit_advertised_tool_names() -> None:
    class _FakeChannel:
        def toolbox_describe(self, **kwargs):
            return {"status": "ok", "all_registered_tool_names": ["wrong_name"]}

    router = HostedToolExecutionRouter()
    router.configure_hosted_execution(
        control_channel=_FakeChannel(),
        toolbox_id="user-tools",
        advertised_tool_names=["SimpleCalc", "ProjectFilePeek"],
    )

    assert router.hosted_advertised_tool_names() == ["SimpleCalc", "ProjectFilePeek"]
    summary = router.hosted_toolbox_summary()
    assert summary is not None
    assert summary["all_registered_tool_names"] == ["SimpleCalc", "ProjectFilePeek"]
    assert summary["source"] == "explicit"


def test_execute_tool_round_on_cursor_uses_hosted_executor_and_creates_tryout_branch() -> None:
    class _FakeChannel:
        def toolbox_execute(self, **kwargs):
            tool_call = dict(kwargs.get("tool_call") or {})
            return {
                "status": "ok",
                "tool_call": {
                    **tool_call,
                    "result": '{"greeting":"hi Sam"}',
                },
            }

    events: list[str] = []

    async def _action_handler(execute_stage: str, **kwargs):
        events.append(str(execute_stage))
        return None

    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    toolbox = Toolbox()
    context = ChatContext(session, chat_session=chat_session, toolbox=toolbox)
    cursor = context.active_cursor
    cursor.add_user("hello")

    router = HostedToolExecutionRouter()
    executor = router.configure_hosted_execution(
        control_channel=_FakeChannel(),
        toolbox_id="user-tools",
    )
    active_executor = router.active_executor(toolbox)

    response = InferenceResponse(
        chunkType="streaming_chunk",
        prompt_index=0,
        tool_blocks=[
            ToolCallBlock(
                raw_block='<tool_call>{"name":"hello_remote","arguments":{"name":"Sam"}}</tool_call>'
            )
        ],
    )

    result = __import__("asyncio").run(
        execute_tool_round_on_cursor(
            cursor=cursor,
            final_response_items=[response],
            responses_in_progress={0: ""},
            parser_profile=DEFAULT_PROFILE,
            tool_executor=active_executor,
            action_handler=_action_handler,
        )
    )

    assert executor is active_executor
    assert result.had_tool_blocks is True
    assert result.executed is True
    assert result.scheduled_auto_iteration is True
    assert result.aborted is False
    assert result.tool_result_cursor_id
    scoped_cursor = context.active_cursor
    tool_results = (scoped_cursor.current_turn.data or {}).get("tool_results") or []
    assert tool_results
    call = tool_results[0].calls[0]
    assert call.name == "hello_remote"
    assert call.result == '{"greeting":"hi Sam"}'
    assert events == ["calls_parsed", "call_starting", "call_finished", "all_finished"]


def test_execute_tool_round_on_cursor_allows_parallel_hosted_calls() -> None:
    class _FakeChannel:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.in_flight = 0
            self.max_in_flight = 0
            self.execute_calls: list[str] = []

        def toolbox_execute(self, **kwargs):
            tool_call = dict(kwargs.get("tool_call") or {})
            name = str(tool_call.get("name") or "").strip()
            with self._lock:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
                self.execute_calls.append(name)
            try:
                time.sleep(0.10)
                return {
                    "status": "ok",
                    "tool_call": {
                        **tool_call,
                        "result": f'{{"tool":"{name}"}}',
                    },
                }
            finally:
                with self._lock:
                    self.in_flight -= 1

    events: list[str] = []

    async def _action_handler(execute_stage: str, **kwargs):
        events.append(str(execute_stage))
        return None

    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    toolbox = Toolbox()
    context = ChatContext(session, chat_session=chat_session, toolbox=toolbox)
    cursor = context.active_cursor
    cursor.add_user("hello")

    channel = _FakeChannel()
    router = HostedToolExecutionRouter()
    active_executor = router.configure_hosted_execution(
        control_channel=channel,
        toolbox_id="user-tools",
    )

    response = InferenceResponse(
        chunkType="streaming_chunk",
        prompt_index=0,
        tool_blocks=[
            ToolCallBlock(raw_block='<tool_call>{"name":"first_remote","arguments":{}}</tool_call>'),
            ToolCallBlock(raw_block='<tool_call>{"name":"second_remote","arguments":{}}</tool_call>'),
        ],
    )

    result = __import__("asyncio").run(
        execute_tool_round_on_cursor(
            cursor=cursor,
            final_response_items=[response],
            responses_in_progress={0: ""},
            parser_profile=DEFAULT_PROFILE,
            tool_executor=active_executor,
            action_handler=_action_handler,
        )
    )

    assert result.executed is True
    assert sorted(channel.execute_calls) == ["first_remote", "second_remote"]
    assert channel.max_in_flight >= 2
    assert events[0] == "calls_parsed"
    assert events[-1] == "all_finished"
    assert events.count("call_starting") == 2
    assert events.count("call_finished") == 2


def test_hosted_round_records_server_control_and_local_results_once() -> None:
    local_effects: list[str] = []
    control_effects: list[str] = []
    approvals: list[str] = []

    def local_echo(value: str) -> dict:
        local_effects.append(value)
        return {"echo": value}

    async def control_handler(*, tool_call: ToolCall, callback_processor, **kwargs: Any) -> dict:
        control_effects.append(str(tool_call.arguments.get("query") or ""))
        decision = callback_processor(
            callback_name="tool_requires_confirmation",
            payload={"tool_name": tool_call.name, "arguments": dict(tool_call.arguments)},
            context={"tool_call_id": tool_call.id},
        )
        return {"status": "scope_applied", "decision": decision["decision"]}

    def approval_callback(*, callback_name: str, payload: Dict[str, Any], context: Any) -> dict:
        approvals.append(callback_name)
        return {"decision": "allow_once"}

    async def action_handler(execute_stage: str, **kwargs: Any) -> None:
        return None

    toolbox = Toolbox()
    ok, message = toolbox.add_tool_callable(local_echo, activate=True)
    assert ok, message
    session = EngineSession()
    chat_session = session.add_conversation(inference_defaults=InferenceParams(), initial_params={})
    context = ChatContext(session, chat_session=chat_session, toolbox=toolbox)
    cursor = context.active_cursor
    cursor.add_user("hello")
    request_turn = cursor.current_turn
    response = InferenceResponse(
        chunkType="streaming_chunk",
        prompt_index=0,
        tool_blocks=[
            ToolCallBlock(raw_block='<tool_call>{"name":"toolbox_search_and_scope","arguments":{"query":"docs"}}</tool_call>'),
            ToolCallBlock(raw_block='<tool_call>{"name":"local_echo","arguments":{"value":"ok"}}</tool_call>'),
        ],
    )

    result = __import__("asyncio").run(
        execute_tool_round_on_cursor(
            cursor=cursor,
            final_response_items=[response],
            responses_in_progress={0: ""},
            parser_profile=DEFAULT_PROFILE,
            tool_executor=toolbox,
            action_handler=action_handler,
            tools_view=ToolsView(
                view_id="mixed-control-round",
                mode="advertised",
                allowed_tools={"toolbox_search_and_scope", "local_echo"},
                advertised_tools={"toolbox_search_and_scope", "local_echo"},
                hidden_allowed_tools=set(),
                disabled_tools=set(),
                gated_tools=set(),
            ),
            callback_processor=approval_callback,
            control_tool_handlers={"toolbox_search_and_scope": control_handler},
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
    )

    assert result.scheduled_auto_iteration is True
    assert result.had_server_tool_events is True
    assert result.server_events_recorded is True
    assert control_effects == ["docs"]
    assert approvals == ["tool_requires_confirmation"]
    assert local_effects == ["ok"]
    calls = [
        call
        for block in (context.active_cursor.current_turn.data or {})["tool_results"]
        for call in block.calls
    ]
    assert [call.name for call in calls] == ["toolbox_search_and_scope", "local_echo"]
    assert calls[0].result == {"status": "scope_applied", "decision": "allow_once"}
    assert calls[0].error is None
    assert calls[1].result is not None
    assert request_turn.data["assistant"]["server_tool_events"] == [
        {
            "schema_version": "server_tool.event.v1",
            "kind": "server_tool_call",
            "provider_id": "openai",
            "tool_id": "web_search",
            "item_type": "web_search_call",
            "status": "completed",
        }
    ]


def test_hosted_round_records_server_event_without_local_execution() -> None:
    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(), initial_params={}
    )
    context = ChatContext(session, chat_session=chat_session, toolbox=Toolbox())
    cursor = context.active_cursor
    cursor.add_user("search remotely")
    request_turn = cursor.current_turn

    async def action_handler(execute_stage: str, **kwargs: Any) -> None:
        raise AssertionError("event-only rounds must not execute local tools")

    result = __import__("asyncio").run(
        execute_tool_round_on_cursor(
            cursor=cursor,
            final_response_items=[],
            responses_in_progress={0: "Found it"},
            parser_profile=DEFAULT_PROFILE,
            tool_executor=Toolbox(),
            action_handler=action_handler,
            server_tool_events=[
                {
                    "provider_id": "grok",
                    "tool_id": "x_search",
                    "item_type": "x_search_call",
                    "status": "completed",
                }
            ],
        )
    )

    assert result.had_tool_blocks is False
    assert result.executed is False
    assert result.server_events_recorded is True
    assert request_turn.data["assistant"]["content"] == "Found it"
    assert request_turn.data["assistant"]["server_tool_events"][0] == {
        "schema_version": "server_tool.event.v1",
        "kind": "server_tool_call",
        "provider_id": "grok",
        "tool_id": "x_search",
        "item_type": "x_search_call",
        "status": "completed",
    }


def test_execute_tool_round_on_cursor_forwards_callback_processor() -> None:
    class _FakeExecutor:
        def __init__(self) -> None:
            self.calls: list[Dict[str, Any]] = []

        async def execute_request_tools(self, **kwargs):
            self.calls.append(dict(kwargs))
            response = list(kwargs.get("final_response_items") or [])[0]
            block = list(response.tool_blocks or [])[0]
            call = list(block.calls or [])[0]
            call.result = '{"status":"ok"}'

    async def _action_handler(execute_stage: str, **kwargs):
        return None

    def _callback_processor(*, callback_name: str, payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
        return {"callback_name": callback_name}

    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    context = ChatContext(session, chat_session=chat_session, toolbox=Toolbox())
    cursor = context.active_cursor
    cursor.add_user("hello")

    executor = _FakeExecutor()
    response = InferenceResponse(
        chunkType="streaming_chunk",
        prompt_index=0,
        tool_blocks=[
            ToolCallBlock(
                calls=[
                    ToolCall(
                        id="call-1",
                        name="remote_tool",
                        arguments={},
                    )
                ]
            )
        ],
    )

    result = __import__("asyncio").run(
        execute_tool_round_on_cursor(
            cursor=cursor,
            final_response_items=[response],
            responses_in_progress={0: ""},
            parser_profile=DEFAULT_PROFILE,
            tool_executor=executor,
            action_handler=_action_handler,
            callback_processor=_callback_processor,
            callback_context={"origin": "runtime"},
        )
    )

    assert result.executed is True
    assert len(executor.calls) == 1
    payload = dict(executor.calls[0])
    assert payload["callback_processor"] is _callback_processor
    forwarded_context = dict(payload["callback_context"] or {})
    assert forwarded_context["origin"] == "runtime"
    assert forwarded_context["cursor"] is cursor
    assert forwarded_context["toolbox_ref"] is context.toolbox_ref


def test_execute_tool_round_on_cursor_forwards_host_api_approval() -> None:
    class _FakeExecutor:
        def __init__(self) -> None:
            self.calls: list[Dict[str, Any]] = []

        async def execute_request_tools(self, **kwargs):
            self.calls.append(dict(kwargs))
            response = list(kwargs.get("final_response_items") or [])[0]
            block = list(response.tool_blocks or [])[0]
            call = list(block.calls or [])[0]
            call.result = '{"status":"ok"}'

    async def _action_handler(execute_stage: str, **kwargs):
        return None

    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    context = ChatContext(session, chat_session=chat_session, toolbox=Toolbox())
    cursor = context.active_cursor
    cursor.add_user("hello")

    executor = _FakeExecutor()
    response = InferenceResponse(
        chunkType="streaming_chunk",
        prompt_index=0,
        tool_blocks=[
            ToolCallBlock(
                calls=[
                    ToolCall(
                        id="call-1",
                        name="remote_tool",
                        arguments={},
                    )
                ]
            )
        ],
    )

    result = __import__("asyncio").run(
        execute_tool_round_on_cursor(
            cursor=cursor,
            final_response_items=[response],
            responses_in_progress={0: ""},
            parser_profile=DEFAULT_PROFILE,
            tool_executor=executor,
            action_handler=_action_handler,
            host_api_approval={"mode": "always"},
        )
    )

    assert result.executed is True
    assert len(executor.calls) == 1
    assert executor.calls[0]["host_api_approval"] == {"mode": "always"}


def test_execute_tool_round_on_cursor_reports_canceled_resubmit_guidance() -> None:
    class _FakeChannel:
        def toolbox_execute(self, **kwargs):
            raise RuntimeError("toolbox_executor_missing:user-tools")

    events: list[str] = []

    async def _action_handler(execute_stage: str, **kwargs):
        events.append(str(execute_stage))
        return None

    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    toolbox = Toolbox()
    context = ChatContext(session, chat_session=chat_session, toolbox=toolbox)
    cursor = context.active_cursor
    cursor.add_user("hello")

    router = HostedToolExecutionRouter()
    active_executor = router.configure_hosted_execution(
        control_channel=_FakeChannel(),
        toolbox_id="user-tools",
    )

    response = InferenceResponse(
        chunkType="streaming_chunk",
        prompt_index=0,
        tool_blocks=[
            ToolCallBlock(raw_block='<tool_call>{"name":"restartable_remote","arguments":{}}</tool_call>'),
            ToolCallBlock(raw_block='<tool_call>{"name":"sticky_remote","arguments":{}}</tool_call>'),
        ],
    )

    result = __import__("asyncio").run(
        execute_tool_round_on_cursor(
            cursor=cursor,
            final_response_items=[response],
            responses_in_progress={0: ""},
            parser_profile=DEFAULT_PROFILE,
            tool_executor=active_executor,
            action_handler=_action_handler,
            non_restartable_tool_names=["sticky_remote"],
        )
    )

    assert result.executed is True
    assert sorted(result.canceled_tool_names) == ["restartable_remote", "sticky_remote"]
    assert result.resubmittable_tool_names == ["restartable_remote"]
    scoped_cursor = context.active_cursor
    tool_results = (scoped_cursor.current_turn.data or {}).get("tool_results") or []
    summary = summarize_canceled_tool_calls(
        tool_results,
        non_restartable_tool_names=["sticky_remote"],
    )
    assert sorted(summary["canceled_tool_names"]) == ["restartable_remote", "sticky_remote"]
    assert summary["resubmittable_tool_names"] == ["restartable_remote"]
    assert events[0] == "calls_parsed"
    assert events[-1] == "all_finished"


def test_execute_tool_round_on_cursor_preserves_blocked_and_hidden_hosted_semantics() -> None:
    class _FakeChannel:
        def __init__(self) -> None:
            self.gate_calls: list[Dict[str, Any]] = []
            self.execute_calls: list[Dict[str, Any]] = []

        def toolbox_describe(self, **kwargs):
            return {
                "status": "ok",
                "all_registered_tool_names": ["visible_remote", "hidden_remote"],
                "advertised_tool_names": ["visible_remote"],
                "hidden_allowed_tool_names": ["hidden_remote"],
            }

        def toolbox_gate(self, **kwargs):
            self.gate_calls.append(dict(kwargs))
            view = dict(kwargs.get("tools_view") or {})
            allowed = {
                str(item or "").strip()
                for item in list(view.get("allowed_tools") or [])
                if str(item or "").strip()
            }
            name = str(kwargs.get("tool_name") or "").strip()
            if name not in allowed:
                return {
                    "status": "ok",
                    "outcome": "denied",
                    "reason": "blocked_in_scope",
                    "tool_name": name,
                }
            return {
                "status": "ok",
                "outcome": "allowed",
                "reason": "allowed",
                "tool_name": name,
            }

        def toolbox_execute(self, **kwargs):
            self.execute_calls.append(dict(kwargs))
            tool_call = dict(kwargs.get("tool_call") or {})
            return {
                "status": "ok",
                "tool_call": {
                    **tool_call,
                    "result": '{"payload":"hidden-ok"}',
                },
            }

    events: list[str] = []

    async def _action_handler(execute_stage: str, **kwargs):
        events.append(str(execute_stage))
        return None

    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    context = ChatContext(session, chat_session=chat_session, toolbox=Toolbox())
    cursor = context.active_cursor
    cursor.add_user("hello")

    router = HostedToolExecutionRouter()
    executor = router.configure_hosted_execution(
        control_channel=_FakeChannel(),
        toolbox_id="user-tools",
    )
    active_executor = router.active_executor(context.toolbox)
    assert executor is active_executor

    tools_view = ToolsView(
        view_id="turn-1",
        mode="advertised",
        allowed_tools={"hidden_remote"},
        advertised_tools=set(),
        hidden_allowed_tools={"hidden_remote"},
        disabled_tools={"visible_remote"},
    )

    response = InferenceResponse(
        chunkType="streaming_chunk",
        prompt_index=0,
        tool_blocks=[
            ToolCallBlock(raw_block='<tool_call>{"name":"visible_remote","arguments":{}}</tool_call>'),
            ToolCallBlock(raw_block='<tool_call>{"name":"hidden_remote","arguments":{}}</tool_call>'),
        ],
    )

    result = __import__("asyncio").run(
        execute_tool_round_on_cursor(
            cursor=cursor,
            final_response_items=[response],
            responses_in_progress={0: ""},
            parser_profile=DEFAULT_PROFILE,
            tool_executor=active_executor,
            action_handler=_action_handler,
            tools_view=tools_view,
        )
    )

    assert result.executed is True
    scoped_cursor = context.active_cursor
    tool_results = (scoped_cursor.current_turn.data or {}).get("tool_results") or []
    assert tool_results
    calls = [block.calls[0] for block in tool_results if getattr(block, "calls", None)]
    assert [call.name for call in calls] == ["visible_remote", "hidden_remote"]
    assert str(calls[0].error or "") == "Execution gated: denied - blocked_in_scope:visible_remote"
    assert calls[0].result is None
    assert calls[1].result == '{"payload":"hidden-ok"}'
    assert calls[1].error is None

    summary = router.hosted_toolbox_summary()
    assert summary is not None
    effective = summarize_effective_tool_view(
        tools_view,
        hosted_advertised_tool_names=summary.get("advertised_tool_names"),
        hosted_hidden_allowed_tool_names=summary.get("hidden_allowed_tool_names"),
    )
    assert effective["effective_advertised_tools"] == []
    assert effective["effective_hidden_allowed_tools"] == ["hidden_remote"]
    assert effective["hosted_visible_tools"] == []
    assert effective["hosted_gated_tools"] == ["visible_remote"]
    assert effective["hosted_hidden_allowed_tools"] == ["hidden_remote"]

    channel = active_executor.control_channel
    assert len(channel.gate_calls) == 2
    assert len(channel.execute_calls) == 1
    assert channel.execute_calls[0]["tool_call"]["name"] == "hidden_remote"
    assert events[0] == "calls_parsed"
    assert events[-1] == "all_finished"
    assert events.count("call_starting") == 2
    assert events.count("call_finished") == 2


def test_execute_tool_round_on_cursor_auto_forwards_scope_target_for_add_to_scope() -> None:
    class _FakeChannel:
        def __init__(self) -> None:
            self.execute_calls: list[Dict[str, Any]] = []

        def toolbox_describe(self, **kwargs):
            return {
                "status": "ok",
                "all_registered_tool_names": ["dangerous_remote"],
                "advertised_tool_names": ["dangerous_remote"],
                "hidden_allowed_tool_names": [],
            }

        def toolbox_gate(self, **kwargs):
            view = dict(kwargs.get("tools_view") or {})
            name = str(kwargs.get("tool_name") or "").strip()
            gated = set(view.get("gated_tools") or [])
            allowed = set(view.get("allowed_tools") or [])
            if name in gated and name not in allowed:
                return {
                    "status": "ok",
                    "outcome": "gated_requires_confirmation",
                    "reason": "gated_requires_confirmation",
                    "tool_name": name,
                    "requires_confirmation": True,
                }
            return {
                "status": "ok",
                "outcome": "allowed",
                "reason": "allowed",
                "tool_name": name,
            }

        def toolbox_execute(self, **kwargs):
            self.execute_calls.append(dict(kwargs))
            tool_call = dict(kwargs.get("tool_call") or {})
            return {
                "status": "ok",
                "tool_call": {
                    **tool_call,
                    "result": '{"name":"%s"}' % str(tool_call.get("name") or ""),
                },
            }

    seen_callbacks: list[str] = []

    async def _action_handler(execute_stage: str, **kwargs):
        return None

    def _callback_processor(*, callback_name: str, payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
        seen_callbacks.append(callback_name)
        return {
            "decision": "add_to_scope",
            "scope_constraints": {
                "dangerous_remote": {
                    "domains": {
                        "filesystem": {
                            "implied_root": "docs",
                            "allowed_roots": ["docs"],
                        }
                    }
                }
            },
        }

    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    context = ChatContext(session, chat_session=chat_session, toolbox=Toolbox())
    assert context.toolbox_ref is not None
    context.toolbox_ref.set_scope(ToolsScope(gated_tools={"dangerous_remote"}))
    cursor = context.active_cursor
    cursor.add_user("hello")

    router = HostedToolExecutionRouter()
    active_executor = router.configure_hosted_execution(
        control_channel=_FakeChannel(),
        toolbox_id="user-tools",
    )

    tools_view = ToolsView(
        view_id="turn-gated",
        mode="advertised",
        allowed_tools=set(),
        advertised_tools={"dangerous_remote"},
        hidden_allowed_tools=set(),
        disabled_tools=set(),
        gated_tools={"dangerous_remote"},
    )
    response = InferenceResponse(
        chunkType="streaming_chunk",
        prompt_index=0,
        tool_blocks=[
            ToolCallBlock(raw_block='<tool_call>{"name":"dangerous_remote","arguments":{}}</tool_call>'),
            ToolCallBlock(raw_block='<tool_call>{"name":"dangerous_remote","arguments":{}}</tool_call>'),
        ],
    )

    result = __import__("asyncio").run(
        execute_tool_round_on_cursor(
            cursor=cursor,
            final_response_items=[response],
            responses_in_progress={0: ""},
            parser_profile=DEFAULT_PROFILE,
            tool_executor=active_executor,
            action_handler=_action_handler,
            tools_view=tools_view,
            callback_processor=_callback_processor,
        )
    )

    assert result.executed is True
    assert seen_callbacks == ["tool_requires_confirmation"]
    assert context.toolbox_ref.scope.gated_tools == set()
    assert context.toolbox_ref.scope.tool_constraints["dangerous_remote"] == {
        "domains": {
            "filesystem": {
                "implied_root": "docs",
                "allowed_roots": ["docs"],
            }
        }
    }
    channel = active_executor.control_channel
    assert len(channel.execute_calls) == 2


def test_tools_scope_partial_constraints_replay_cleanly_after_targeted_pop() -> None:
    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    toolbox = Toolbox()

    def search_files(name_mask: str = "*") -> dict:
        return {"ok": True, "name_mask": name_mask}

    ok, msg = toolbox.add_tool_callable(search_files, activate=True)
    assert ok, msg

    context = ChatContext(session, chat_session=chat_session, toolbox=toolbox)
    cursor = context.active_cursor
    cursor.add_user("hello")

    scope_a = ToolsScope(
        advertise_tools={"search_files"},
        tool_constraints={
            "search_files": {
                "domains": {
                    "filesystem": {
                        "implied_root": "docs",
                        "allowed_roots": ["docs"],
                    }
                },
                "argument_policy": {
                    "implied_args": {"root_path": "docs"},
                    "normalizers": {"root_path": "path_under_implied_root"},
                },
            }
        },
    )
    cursor.apply_tools_scope("add", scope_a, command_text="scope A")
    stack_id_a = str(cursor.current_turn.cmd[-1].data.get("stack_id") or "")
    assert stack_id_a

    scope_b = ToolsScope(
        tool_constraints={
            "search_files": {
                "argument_policy": {
                    "locked_args": ["root_path"],
                }
            }
        }
    )
    cursor.apply_tools_scope("add", scope_b, command_text="scope B")

    merged_view = cursor.get_tools_view()
    assert merged_view is not None
    assert merged_view.get_constraints("search_files") == {
        "domains": {
            "filesystem": {
                "implied_root": "docs",
                "allowed_roots": ["docs"],
            }
        },
        "argument_policy": {
            "implied_args": {"root_path": "docs"},
            "normalizers": {"root_path": "path_under_implied_root"},
            "locked_args": ["root_path"],
        },
    }

    cursor.apply_tools_scope("pop", command_text="pop A", stack_id=stack_id_a)

    popped_view = cursor.get_tools_view()
    assert popped_view is not None
    assert popped_view.get_constraints("search_files") == {
        "argument_policy": {
            "locked_args": ["root_path"],
        }
    }


def test_tools_scope_cli_accepts_advertised_alias_and_gated_field() -> None:
    from app.mp13chat_tools_cli import parse_scope_cli_args

    scope = parse_scope_cli_args("advertised=ProjectFilePeek gated=ProjectFilePeek")

    assert scope is not None
    assert scope.advertise_tools == {"ProjectFilePeek"}
    assert scope.gated_tools == {"ProjectFilePeek"}


def test_tools_commands_present_hosted_visible_hidden_and_gated_states() -> None:
    class _PromptSession:
        async def prompt_async(self, *_args, **_kwargs):
            raise AssertionError("prompt_async should not be used by /t enum or /t scope show")

    def visible_remote(name: str = "world") -> dict:
        """Visible hosted tool."""
        return {"tool": "visible_remote", "name": name}

    def hidden_remote(name: str = "world") -> dict:
        """Hidden hosted tool."""
        return {"tool": "hidden_remote", "name": name}

    def gated_remote(name: str = "world") -> dict:
        """Gated hosted tool."""
        return {"tool": "gated_remote", "name": name}

    def _strip_ansi(text: str) -> str:
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    session = EngineSession()
    chat_session = session.add_conversation(
        inference_defaults=InferenceParams(),
        initial_params={},
    )
    toolbox = Toolbox()
    ok, msg = toolbox.add_tool_callable(visible_remote, activate=True)
    assert ok, msg
    ok, msg = toolbox.add_tool_callable(hidden_remote, activate=True)
    assert ok, msg
    ok, msg = toolbox.add_tool_callable(gated_remote, activate=True)
    assert ok, msg
    ok, msg = toolbox.set_hidden(["hidden_remote"], True)
    assert ok, msg

    context = ChatContext(session, chat_session=chat_session, toolbox=toolbox)
    cursor = context.active_cursor
    cursor.add_user("hello")
    hosted_summary = {
        "status": "ok",
        "all_registered_tool_names": ["visible_remote", "hidden_remote", "gated_remote"],
        "advertised_tool_names": ["visible_remote", "gated_remote"],
        "hidden_allowed_tool_names": ["hidden_remote"],
    }
    handler = LightweightToolsCliHandler(
        get_toolbox=lambda: toolbox,
        get_toolbox_ref=lambda: None,
        get_hosted_summary=lambda: hosted_summary,
        print_help=lambda: None,
        external_tool_handler=lambda **kwargs: None,
        get_current_config=lambda: {},
        get_search_scope=lambda: {},
    )

    pt_session = _PromptSession()

    scope_stdout = io.StringIO()
    with contextlib.redirect_stdout(scope_stdout):
        scoped_cursor, suppress = __import__("asyncio").run(
            handler.handle_tools_command(
                "scope set disabled=gated_remote",
                cursor,
                pt_session,
            )
        )

    assert suppress is True
    rendered_scope = _strip_ansi(scope_stdout.getvalue())
    assert "Tool scope stack (oldest -> newest):" in rendered_scope
    assert "Advertised tools: visible_remote" in rendered_scope
    assert "Hidden but allowed: hidden_remote" in rendered_scope
    assert "Disabled tools: gated_remote" in rendered_scope
    assert "Hosted execution: active" in rendered_scope
    assert "Hosted-visible tools: visible_remote" in rendered_scope
    assert "Hosted hidden-allowed tools: hidden_remote" in rendered_scope
    assert "Hosted route-gated tools: gated_remote" in rendered_scope

    enum_stdout = io.StringIO()
    with contextlib.redirect_stdout(enum_stdout):
        _, suppress = __import__("asyncio").run(
            handler.handle_tools_command(
                "",
                scoped_cursor,
                pt_session,
            )
        )

    assert suppress is True
    rendered_enum = _strip_ansi(enum_stdout.getvalue())
    assert "Index" in rendered_enum
    assert "visible_remote" in rendered_enum
    assert "Yes" in rendered_enum and "hosted" in rendered_enum
    assert "hidden_remote" in rendered_enum
    assert "hosted-hidden" in rendered_enum
    assert "gated_remote" in rendered_enum
    assert "No" in rendered_enum and "gated" in rendered_enum
