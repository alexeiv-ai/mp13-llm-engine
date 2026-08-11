from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from app.hosted_chat_demo import setup_hosted_chat_demo, shutdown_hosted_chat_demo
from hosting.service.host_service import EngineHostService
from hosting.toolbox_admin import HostedToolboxAdmin
from mp13_engine.mp13_toolbox import Toolbox


class _FakeHost:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.consistency_payload = {
            "status": "ok",
            "issue_count": 1,
            "issues": [{"toolbox_id": "toolbox-a", "issue": "missing_live_registration"}],
        }

    def toolbox_references(self):
        self.calls.append(("references", {}))
        return {"status": "ok", "summary": {"toolbox_count": 1}}

    def toolbox_consistency(self):
        self.calls.append(("consistency", {}))
        return dict(self.consistency_payload)

    def toolbox_reconcile(self, *, request_id, toolbox_ids=None, only_inconsistent=True, details=False):
        self.calls.append(
            (
                "reconcile",
                {
                    "toolbox_ids": list(toolbox_ids or []),
                    "request_id": request_id,
                    "only_inconsistent": bool(only_inconsistent),
                    "details": bool(details),
                },
            )
        )
        return {"status": "ok", "summary": {"before_issue_count": 1, "after_issue_count": 0}}

    def toolbox_gc(self, *, request_id):
        self.calls.append(("gc", {"request_id": request_id}))
        return {"status": "ok", "summary": {"removed_engine_count": 0}}


def test_hosted_toolbox_admin_startup_reconcile_uses_defaults() -> None:
    host = _FakeHost()
    admin = HostedToolboxAdmin(host=host, default_toolbox_ids=["toolbox-a"])

    out = admin.startup_reconcile(request_id="startup-1")

    assert out["status"] == "ok"
    assert host.calls == [
        ("reconcile", {"request_id": "startup-1", "toolbox_ids": ["toolbox-a"], "only_inconsistent": True, "details": False}),
    ]


def test_hosted_toolbox_admin_review_snapshot_filters_and_recommends_reconcile() -> None:
    host = _FakeHost()
    admin = HostedToolboxAdmin(host=host, default_toolbox_ids=["toolbox-a"])

    out = admin.review_snapshot()

    assert out["status"] == "ok"
    assert out["toolbox_ids"] == ["toolbox-a"]
    assert out["recommended_action"] == "reconcile"
    assert out["summary"]["issue_count"] == 1
    assert out["toolboxes"] == {}
    assert host.calls == [
        ("references", {}),
        ("consistency", {}),
    ]


def test_hosted_toolbox_admin_auto_repair_runs_reconcile_only_when_needed() -> None:
    host = _FakeHost()
    admin = HostedToolboxAdmin(host=host, default_toolbox_ids=["toolbox-a"])

    out = admin.auto_repair_if_needed(request_id="repair-1")

    assert out["action"] == "reconcile"
    assert out["issue_count"] == 1
    assert host.calls == [
        ("consistency", {}),
        ("reconcile", {"request_id": "repair-1", "toolbox_ids": ["toolbox-a"], "only_inconsistent": True, "details": False}),
    ]


def test_hosted_toolbox_admin_auto_repair_can_noop_and_gc() -> None:
    host = _FakeHost()
    host.consistency_payload = {"status": "ok", "issue_count": 0, "issues": []}
    admin = HostedToolboxAdmin(host=host, default_toolbox_ids=["toolbox-a"])

    out = admin.auto_repair_if_needed(request_id="gc-1", gc_on_noop=True)

    assert out["action"] == "noop"
    assert out["issue_count"] == 0
    assert dict(out["gc"])["status"] == "ok"
    assert host.calls == [
        ("consistency", {}),
        ("gc", {"request_id": "gc-1"}),
    ]
