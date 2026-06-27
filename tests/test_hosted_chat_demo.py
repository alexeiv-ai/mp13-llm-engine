from __future__ import annotations

import shutil
from pathlib import Path

from hosting import HOST_CAPABILITY_APPROVAL_CALLBACK_NAME

from app.hosted_chat_demo import (
    HostedChatDemoRuntime,
    build_hosted_chat_demo_plan,
    hosted_demo_non_restartable_tool_names,
    hosted_demo_tool_round_options,
    make_hosted_demo_callback_processor,
)


def test_build_hosted_chat_demo_plan_produces_two_distinct_profiles() -> None:
    tmp_path = (Path.cwd() / ".tmp_test_hosted_chat_demo_plan").resolve()
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        plan = build_hosted_chat_demo_plan(
            toolbox_id="chat-hosted-demo",
            project_root=tmp_path,
        )

        assert plan.toolbox_id == "chat-hosted-demo"
        assert plan.local_tool_names == ["SimpleCalc", "ProjectFilePeek", "ExampleHttpPeek"]
        assert len(plan.auto_requests) == 3

        calc_request = plan.auto_requests[0]
        file_request = plan.auto_requests[1]
        http_request = plan.auto_requests[2]

        assert calc_request["callable_name"] == "SimpleCalc"
        assert calc_request["non_restartable"] is False
        assert calc_request["sandbox_policy"] == {"sandbox": {"enabled": True}}

        assert file_request["callable_name"] == "ProjectFilePeek"
        assert file_request["non_restartable"] is False
        assert "tool_constraints_view" in str(file_request["content"])
        assert "resolve_filesystem_root" in str(file_request["content"])
        fs_rules = file_request["sandbox_policy"]["sandbox"]["filesystem"]["rules"]
        assert fs_rules == [
            {
                "root_id": "project_ro",
                "path": str(tmp_path.resolve()),
                "access": ["read"],
            }
        ]
        assert file_request["sandbox_policy"]["sandbox"]["brokered_io"]["filesystem"] is True
        assert http_request["callable_name"] == "ExampleHttpPeek"
        assert http_request["non_restartable"] is False
        assert http_request["sandbox_policy"]["sandbox"]["brokered_io"]["http"] is True
        assert http_request["sandbox_policy"]["sandbox"]["network"] == {
            "mode": "brokered_only",
            "allow_url_prefixes": ["https://example.com/"],
        }
        assert len(plan.suggested_prompts) >= 5
        assert hosted_demo_non_restartable_tool_names(plan) == []
        assert hosted_demo_tool_round_options(plan) == {"non_restartable_tool_names": []}
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_hosted_project_file_peek_source_executes_with_context_fs() -> None:
    plan = build_hosted_chat_demo_plan(
        toolbox_id="chat-hosted-demo",
        project_root=Path.cwd(),
    )
    file_request = next(req for req in plan.auto_requests if req["callable_name"] == "ProjectFilePeek")
    namespace: dict[str, object] = {}
    exec(str(file_request["content"]), namespace)

    class _Fs:
        calls: list[dict[str, object]]

        def __init__(self) -> None:
            self.calls = []

        def read_text(self, **kwargs):
            self.calls.append(dict(kwargs))
            return {"text": "abcdef"}

    class _Context:
        def __init__(self) -> None:
            self.fs = _Fs()

    context = _Context()
    result = namespace["ProjectFilePeek"](
        relative_path="src/app/mp13chat.py",
        root_path="path/to/project",
        max_chars=3,
        context=context,
    )

    assert result == "src/app/mp13chat.py\n---\nabc"
    assert context.fs.calls == [{"root_id": "project_ro", "relative_path": "src/app/mp13chat.py"}]


def test_make_hosted_demo_callback_processor_returns_scoped_project_file_peek_approval() -> None:
    callback = make_hosted_demo_callback_processor(project_file_peek_root="src/app")

    denied = callback(
        callback_name="tool_requires_confirmation",
        payload={"tool_name": "SimpleCalc"},
        context=None,
    )
    assert denied == {"decision": "deny"}

    approved = callback(
        callback_name="tool_requires_confirmation",
        payload={"tool_name": "ProjectFilePeek"},
        context=None,
    )
    assert approved == {
        "decision": "add_to_scope",
        "scope_constraints": {
            "ProjectFilePeek": {
                "domains": {
                    "filesystem": {
                        "implied_root": "src/app",
                        "allowed_roots": ["src/app"],
                        "allow_explicit_root_override": False,
                    }
                },
                "argument_policy": {
                    "implied_args": {"root_path": "src/app"},
                    "locked_args": ["root_path"],
                    "normalizers": {"root_path": "path_under_implied_root"},
                },
            }
        },
    }


def test_hosted_demo_tool_round_options_exposes_host_api_approval() -> None:
    plan = build_hosted_chat_demo_plan(
        toolbox_id="chat-hosted-demo",
        project_root=Path.cwd(),
    )
    runtime = HostedChatDemoRuntime(
        service=object(),
        toolbox_ref=object(),
        plan=plan,
        callback_processor=lambda **_kwargs: {"decision": "deny"},
        host_api_approval={"mode": "always"},
    )

    options = hosted_demo_tool_round_options(runtime)

    assert options["non_restartable_tool_names"] == []
    assert callable(options["callback_processor"])
    assert options["host_api_approval"] == {"mode": "always"}


def test_make_hosted_demo_callback_processor_prints_auto_approval(capsys) -> None:
    class _Context:
        tool_call_id = "call-1"

    callback = make_hosted_demo_callback_processor(project_file_peek_root="src/app")
    out = callback(
        callback_name="tool_requires_confirmation",
        payload={"tool_name": "ProjectFilePeek", "tool_arguments": {"relative_path": "mp13chat.py"}},
        context=_Context(),
    )

    assert out["decision"] == "add_to_scope"
    captured = capsys.readouterr()
    assert "Auto-approved ProjectFilePeek with scoped root: src/app" in captured.out
    assert "relative_path" in captured.out


def test_make_hosted_demo_callback_processor_prints_denial_for_other_tools(capsys) -> None:
    callback = make_hosted_demo_callback_processor(project_file_peek_root="src/app")
    out = callback(
        callback_name="tool_requires_confirmation",
        payload={"tool_name": "SimpleCalc", "tool_arguments": {"expr": "2+2"}},
        context=None,
    )

    assert out == {"decision": "deny"}
    captured = capsys.readouterr()
    assert "Denied gated tool SimpleCalc" in captured.out


def test_make_hosted_demo_callback_processor_handles_host_api_approval(capsys) -> None:
    class _Context:
        tool_name = "ProjectFilePeek"

    callback = make_hosted_demo_callback_processor(project_file_peek_root="src/app")

    allowed = callback(
        callback_name=HOST_CAPABILITY_APPROVAL_CALLBACK_NAME,
        payload={"method": "fs.read_text", "approval_id": "approval-1"},
        context=_Context(),
    )

    assert allowed["decision"] == "allow_once"
    assert allowed["approval_id"] == "approval-1"

    denied = callback(
        callback_name=HOST_CAPABILITY_APPROVAL_CALLBACK_NAME,
        payload={"method": "fs.write_text", "approval_id": "approval-2"},
        context=_Context(),
    )

    assert denied["decision"] == "deny"
    assert denied["approval_id"] == "approval-2"
    assert denied["reason"] == "unsupported_demo_host_api_method"
    captured = capsys.readouterr()
    assert "Auto-approved fs.read_text" in captured.out
    assert "Denied unsupported method fs.write_text" in captured.out
