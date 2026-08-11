from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

import pytest


_PROCESS_TEST_MODULES = {
    "test_hosting_r6_restart_healing.py",
    "test_hosting_r7_acceptance.py",
    "test_hosting_toolbox_sandbox.py",
    "test_hosting_worker_sandbox.py",
    "test_hosting_worker_sandbox_windows_live.py",
    "test_workflow_helper_service.py",
    "test_workflow_js_node_runtime.py",
    "test_workflow_python_helper_ipc.py",
    "test_workflow_python_node_worker_ipc.py",
}
_NATIVE_TEST_MODULES = {
    "test_hosting_toolbox_target.py",
    "test_hosting_toolbox_target_workflow.py",
    "test_hosting_worker_sandbox_windows_live.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign stable lane markers from the test's module boundary."""
    for item in items:
        module_name = Path(str(item.fspath)).name
        if module_name in _NATIVE_TEST_MODULES:
            item.add_marker("native")
        elif module_name in _PROCESS_TEST_MODULES:
            item.add_marker("process")
        else:
            item.add_marker("fast")


@pytest.fixture(autouse=True)
def _close_process_test_host_services(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Give every real-process test unconditional service/runtime cleanup."""
    if request.node.get_closest_marker("process") is None:
        yield
        return
    from hosting.service.host_service import EngineHostService

    original_init = EngineHostService.__init__
    services: list[EngineHostService] = []

    def _tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        services.append(self)

    monkeypatch.setattr(EngineHostService, "__init__", _tracked_init)
    try:
        yield
    finally:
        for service in reversed(services):
            try:
                service.close()
            except Exception:
                pass


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _ensure_pytest_temp_root() -> None:
    # Keep pytest temp artifacts outside the repo root by default.
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root.parent / ".mp13_pytest",
        Path(tempfile.gettempdir()) / "mp13_pytest",
        root / ".tmp_pytest",
    ]
    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(base))
            return
        except PermissionError:
            continue


_ensure_src_on_path()
_ensure_pytest_temp_root()
