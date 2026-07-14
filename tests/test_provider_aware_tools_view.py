from __future__ import annotations

from mp13_engine.mp13_toolbox import ToolsView


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
