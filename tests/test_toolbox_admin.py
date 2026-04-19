from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from app.hosted_chat_demo import setup_hosted_chat_demo, shutdown_hosted_chat_demo
from hosting.engine_host_service import EngineHostService
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

    def toolbox_reconcile(self, *, toolbox_ids=None, only_inconsistent=True, details=False):
        self.calls.append(
            (
                "reconcile",
                {
                    "toolbox_ids": list(toolbox_ids or []),
                    "only_inconsistent": bool(only_inconsistent),
                    "details": bool(details),
                },
            )
        )
        return {"status": "ok", "summary": {"before_issue_count": 1, "after_issue_count": 0}}

    def toolbox_gc(self):
        self.calls.append(("gc", {}))
        return {"status": "ok", "summary": {"removed_engine_count": 0}}


def test_hosted_toolbox_admin_startup_reconcile_uses_defaults() -> None:
    host = _FakeHost()
    admin = HostedToolboxAdmin(host=host, default_toolbox_ids=["toolbox-a"])

    out = admin.startup_reconcile()

    assert out["status"] == "ok"
    assert host.calls == [
        ("reconcile", {"toolbox_ids": ["toolbox-a"], "only_inconsistent": True, "details": False}),
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

    out = admin.auto_repair_if_needed()

    assert out["action"] == "reconcile"
    assert out["issue_count"] == 1
    assert host.calls == [
        ("consistency", {}),
        ("reconcile", {"toolbox_ids": ["toolbox-a"], "only_inconsistent": True, "details": False}),
    ]


def test_hosted_toolbox_admin_auto_repair_can_noop_and_gc() -> None:
    host = _FakeHost()
    host.consistency_payload = {"status": "ok", "issue_count": 0, "issues": []}
    admin = HostedToolboxAdmin(host=host, default_toolbox_ids=["toolbox-a"])

    out = admin.auto_repair_if_needed(gc_on_noop=True)

    assert out["action"] == "noop"
    assert out["issue_count"] == 0
    assert dict(out["gc"])["status"] == "ok"
    assert host.calls == [
        ("consistency", {}),
        ("gc", {}),
    ]


def test_hosted_toolbox_admin_review_snapshot_with_real_hosted_demo_toolbox() -> None:
    root = Path(".tmp_toolbox_admin_real").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    runtime = None
    try:
        original_spawn = EngineHostService.spawn
        original_ready = EngineHostService._ensure_toolbox_assignments_ready

        def _fake_spawn(self, **kwargs: Any):
            engine_id = str(kwargs.get("engine_id") or "")
            sandbox_policy = dict(kwargs.get("sandbox_policy") or {})
            bundle = dict(kwargs.get("bundle") or {})
            environment = dict(kwargs.get("environment") or {})
            tool_access = dict(kwargs.get("tool_access") or {})
            return self.register_spawned(
                engine_id=engine_id,
                pid=1000 + len(self._read_engines()),
                command=list(kwargs.get("command") or []),
                cwd=kwargs.get("cwd"),
                env=dict(kwargs.get("env") or {}),
                worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
                worker_ipc_address=str(root / f"{engine_id}.sock") if sys.platform != "win32" else rf"\\.\pipe\{engine_id}",
                worker_profile_class=kwargs.get("worker_profile_class"),
                sandbox_policy=sandbox_policy,
                executor_kind=kwargs.get("executor_kind"),
                bundle=bundle,
                environment=environment,
                tool_access=tool_access,
                capabilities=dict(kwargs.get("capabilities") or {}),
            )

        def _fake_ready(self, assignments, *, timeout_seconds=8.0):
            ready = {}
            for item in list(assignments or []):
                reg = dict(getattr(item, "registration", None) or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                tool_access = dict(reg.get("tool_access") or {})
                tool_names = [str(x or "").strip() for x in list(tool_access.get("allowed_tool_names") or []) if str(x or "").strip()]
                advertised_tool_names = [str(x or "").strip() for x in list(tool_access.get("advertised_tool_names") or []) if str(x or "").strip()]
                hidden_allowed_tool_names = [str(x or "").strip() for x in list(tool_access.get("hidden_allowed_tool_names") or []) if str(x or "").strip()]
                ready[engine_id] = {
                    "engine_id": engine_id,
                    "ready": True,
                    "ready_at": 0.0,
                    "warmup_ms": 0,
                    "tool_inventory_ok": True,
                    "tool_count": len(tool_names),
                    "all_registered_tool_names": tool_names,
                    "advertised_tool_names": advertised_tool_names,
                    "hidden_allowed_tool_names": hidden_allowed_tool_names,
                }
            return ready

        EngineHostService.spawn = _fake_spawn  # type: ignore[assignment]
        EngineHostService._ensure_toolbox_assignments_ready = _fake_ready  # type: ignore[assignment]

        runtime = setup_hosted_chat_demo(
            toolbox=Toolbox(),
            hosting_root=root,
            project_root=Path.cwd(),
            toolbox_id="toolbox-admin-demo",
        )
        runtime.service._probe_registration_reachability = lambda item, timeout_seconds=0.35: {"reachable": True, "probe": "hello"}  # type: ignore[method-assign]
        admin = HostedToolboxAdmin(host=runtime.service, default_toolbox_ids=["toolbox-admin-demo"])

        out = admin.review_snapshot()

        assert out["status"] == "ok"
        assert out["toolbox_ids"] == ["toolbox-admin-demo"]
        assert out["recommended_action"] == "observe"
        assert out["summary"]["toolbox_count"] == 1
        assert out["summary"]["issue_count"] == 0
        assert out["summary"]["stale_bundle_count"] == 0
        toolbox_row = dict(dict(out.get("toolboxes") or {})["toolbox-admin-demo"])
        assert int(toolbox_row.get("issue_count") or 0) == 0
        profiles = list(toolbox_row.get("profiles") or [])
        assert len(profiles) == 3
        for profile in profiles:
            profile_row = dict(profile or {})
            assert list(profile_row.get("all_registered_tool_names") or [])
            assert "advertised_tool_names" in profile_row
            assert "hidden_allowed_tool_names" in profile_row
    finally:
        EngineHostService.spawn = original_spawn  # type: ignore[assignment]
        EngineHostService._ensure_toolbox_assignments_ready = original_ready  # type: ignore[assignment]
        if runtime is not None:
            shutdown_hosted_chat_demo(runtime)
        shutil.rmtree(root, ignore_errors=True)
