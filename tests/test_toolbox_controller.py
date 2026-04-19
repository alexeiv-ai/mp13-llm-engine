from __future__ import annotations

import pytest

from app.mp13chat_tools_cli import parse_cli_targets, parse_pop_target_options, parse_scope_cli_args
from app.session_cursor_toolbox import normalize_scope_tool_names, tool_wildcard_groups
from app.session_toolbox_controller import SessionToolboxController
from mp13_engine.mp13_toolbox import ToolBoxRef, Toolbox, ToolsScope


def _toolbox_with_user_tools() -> Toolbox:
    toolbox = Toolbox()
    toolbox.tools = {
        "SearchWeb": {
            "_type": "callable",
            "function": {"name": "SearchWeb", "description": "Search the web."},
        },
        "WriteFile": {
            "_type": "external",
            "function": {"name": "WriteFile", "description": "Write a file."},
        },
    }
    toolbox.active_tool_names = ["SearchWeb", "WriteFile"]
    return toolbox


def test_parse_scope_cli_args_supports_gated_and_aliases() -> None:
    scope = parse_scope_cli_args('m=silent a=search s=calc d=db g=deploy label="review flow"')

    assert scope is not None
    assert scope.mode == "silent"
    assert scope.advertise_tools == {"search"}
    assert scope.silent_tools == {"calc"}
    assert scope.disabled_tools == {"db"}
    assert scope.gated_tools == {"deploy"}
    assert scope.label == "review flow"


def test_bridge_normalizes_scope_names_without_terminal_colors() -> None:
    toolbox = _toolbox_with_user_tools()
    scope = ToolsScope(advertise_tools={"search"}, disabled_tools={"missing"}).clean()

    normalized, warnings = normalize_scope_tool_names(scope, toolbox)

    assert normalized.advertise_tools == {"SearchWeb"}
    assert normalized.disabled_tools == set()
    assert warnings == ["Tool 'missing' not recognized for scope."]


@pytest.mark.asyncio
async def test_cli_target_parser_supports_wildcard_groups_and_numeric_selection() -> None:
    targets, warnings = await parse_cli_targets(
        "2,*c",
        ["SearchWeb", "WriteFile"],
        allow_wildcard=True,
        wildcard_values=["SearchWeb", "WriteFile"],
        wildcard_groups={"*c": ["SearchWeb"]},
    )

    assert targets == ["WriteFile", "SearchWeb"]
    assert warnings == []


def test_pop_target_parser_handles_cmd_flag() -> None:
    assert parse_pop_target_options("--cmd stack-1") == ("stack-1", True)
    assert parse_pop_target_options("stack-2") == ("stack-2", False)


def test_wildcard_groups_split_callable_external_types() -> None:
    toolbox = _toolbox_with_user_tools()

    assert tool_wildcard_groups(toolbox) == {
        "*c": ["SearchWeb"],
        "*e": ["WriteFile"],
    }


@pytest.mark.asyncio
async def test_controller_global_mutates_toolbox_ref_without_cli_io() -> None:
    toolbox = _toolbox_with_user_tools()
    ref = ToolBoxRef(toolbox)
    controller = SessionToolboxController(
        get_toolbox=lambda: toolbox,
        get_toolbox_ref=lambda: ref,
        get_hosted_summary=lambda: None,
        get_current_config=lambda: {},
        get_search_scope=lambda: {},
        prompt_user_fn=lambda prompt: _async_value(""),
        prompt_multiline_fn=lambda prompt: _async_value(""),
        interactive_edit_fn=lambda name, context: _async_value((True, "edited")),
        get_external_tool_handler_fn=lambda: (lambda **kwargs: _async_value("")),
    )

    result = await controller.cmd_global(None, "silent")

    assert ref.scope.mode == "silent"
    assert [(msg.level, msg.text) for msg in result.messages] == [
        ("info", "Context tools mode set to 'silent'.")
    ]


async def _async_value(value):
    return value
