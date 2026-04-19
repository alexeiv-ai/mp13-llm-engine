from __future__ import annotations

from app.hosted_tool_visibility import annotate_tool_listing, summarize_effective_tool_view
from mp13_engine.mp13_toolbox import ToolsView


def test_summarize_effective_tool_view_applies_hosted_filter() -> None:
    view = ToolsView(
        view_id="v1",
        mode="advertised",
        allowed_tools={"SimpleCalc", "ProjectFilePeek", "scriptable_calculator"},
        advertised_tools={"SimpleCalc", "ProjectFilePeek", "scriptable_calculator"},
        hidden_allowed_tools=set(),
        disabled_tools=set(),
    )

    summary = summarize_effective_tool_view(
        view,
        hosted_advertised_tool_names=["SimpleCalc", "ProjectFilePeek"],
    )

    assert summary["effective_advertised_tools"] == ["ProjectFilePeek", "SimpleCalc"]
    assert summary["effective_gated_tools"] == []
    assert summary["hosted_gated_tools"] == ["scriptable_calculator"]
    assert summary["hosted_execution"] is True


def test_annotate_tool_listing_marks_hosted_gated_tools() -> None:
    view = ToolsView(
        view_id="v1",
        mode="advertised",
        allowed_tools={"SimpleCalc", "scriptable_calculator"},
        advertised_tools={"SimpleCalc", "scriptable_calculator"},
        hidden_allowed_tools=set(),
        disabled_tools=set(),
    )
    rows = annotate_tool_listing(
        [
            ("SimpleCalc", "calc", "callable", True, False, False, False),
            ("scriptable_calculator", "intrinsic", "intrinsic", True, False, False, False),
        ],
        tools_view=view,
        hosted_advertised_tool_names=["SimpleCalc"],
    )

    assert rows[0]["availability"] == "Yes"
    assert rows[0]["via"] == "hosted"
    assert rows[1]["availability"] == "No"
    assert rows[1]["via"] == "hosted-gated"


def test_summarize_effective_tool_view_preserves_hosted_hidden_allowed_tools() -> None:
    view = ToolsView(
        view_id="v1",
        mode="advertised",
        allowed_tools={"SimpleCalc", "HiddenPeek"},
        advertised_tools={"SimpleCalc"},
        hidden_allowed_tools={"HiddenPeek"},
        disabled_tools=set(),
    )

    summary = summarize_effective_tool_view(
        view,
        hosted_advertised_tool_names=["SimpleCalc"],
        hosted_hidden_allowed_tool_names=["HiddenPeek"],
    )

    assert summary["effective_advertised_tools"] == ["SimpleCalc"]
    assert summary["effective_hidden_allowed_tools"] == ["HiddenPeek"]
    assert summary["effective_gated_tools"] == []
    assert summary["hosted_hidden_allowed_tools"] == ["HiddenPeek"]
    assert summary["hosted_gated_tools"] == []


def test_annotate_tool_listing_marks_hosted_hidden_allowed_tools() -> None:
    view = ToolsView(
        view_id="v1",
        mode="advertised",
        allowed_tools={"SimpleCalc", "HiddenPeek"},
        advertised_tools={"SimpleCalc"},
        hidden_allowed_tools={"HiddenPeek"},
        disabled_tools=set(),
    )
    rows = annotate_tool_listing(
        [
            ("SimpleCalc", "calc", "callable", True, False, False, False),
            ("HiddenPeek", "hidden", "callable", True, True, False, False),
        ],
        tools_view=view,
        hosted_advertised_tool_names=["SimpleCalc"],
        hosted_hidden_allowed_tool_names=["HiddenPeek"],
    )

    assert rows[0]["availability"] == "Yes"
    assert rows[0]["via"] == "hosted"
    assert rows[1]["availability"] == "Yes"
    assert rows[1]["via"] == "hosted-hidden"


def test_summarize_effective_tool_view_marks_scope_disabled_hosted_tool_as_gated() -> None:
    view = ToolsView(
        view_id="v1",
        mode="advertised",
        allowed_tools={"visible_remote", "hidden_remote"},
        advertised_tools={"visible_remote"},
        hidden_allowed_tools={"hidden_remote"},
        disabled_tools={"gated_remote"},
    )

    summary = summarize_effective_tool_view(
        view,
        hosted_advertised_tool_names=["visible_remote", "gated_remote"],
        hosted_hidden_allowed_tool_names=["hidden_remote"],
    )

    assert summary["hosted_visible_tools"] == ["visible_remote"]
    assert summary["hosted_hidden_allowed_tools"] == ["hidden_remote"]
    assert summary["hosted_gated_tools"] == ["gated_remote"]


def test_summarize_effective_tool_view_preserves_local_gated_tools_separately() -> None:
    view = ToolsView(
        view_id="v1",
        mode="advertised",
        allowed_tools={"visible_remote"},
        advertised_tools={"visible_remote", "gated_remote"},
        hidden_allowed_tools=set(),
        disabled_tools=set(),
        gated_tools={"gated_remote"},
    )

    summary = summarize_effective_tool_view(
        view,
        hosted_advertised_tool_names=["visible_remote", "gated_remote"],
    )

    assert summary["effective_advertised_tools"] == ["gated_remote", "visible_remote"]
    assert summary["effective_gated_tools"] == ["gated_remote"]
    assert summary["hosted_gated_tools"] == []


def test_annotate_tool_listing_marks_confirmation_gated_tools() -> None:
    view = ToolsView(
        view_id="v1",
        mode="advertised",
        allowed_tools={"visible_remote"},
        advertised_tools={"visible_remote", "gated_remote"},
        hidden_allowed_tools=set(),
        disabled_tools=set(),
        gated_tools={"gated_remote"},
    )
    rows = annotate_tool_listing(
        [
            ("visible_remote", "visible", "callable", True, False, False, False),
            ("gated_remote", "gated", "callable", True, False, False, False),
        ],
        tools_view=view,
        hosted_advertised_tool_names=["visible_remote", "gated_remote"],
    )

    assert rows[0]["availability"] == "Yes"
    assert rows[0]["via"] == "hosted"
    assert rows[1]["availability"] == "No"
    assert rows[1]["via"] == "gated"
