from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from hosting.service.host_service import EngineHostService as _EngineHostService, ToolboxRolloutError
from hosting.operation_contract import HostedExecutionKind, HostedOperationSelector, hosted_execution_fingerprint
from hosting.sandbox.toolbox_runtime import HostedToolboxRuntimeBase
from hosting.daemon import EngineHostDaemon
from hosting.engine_host_channel import EngineHostControlChannel
from hosting import toolbox_executor_ipc
from hosting.toolbox_harness import (
    HostedToolBoxRef,
    SandboxProfileSpec,
    ToolboxAutoAssignmentRequest,
    ToolboxManualAssignmentRequest,
    ToolboxBundleFile,
    ToolboxBundleAutoTool,
    ToolboxEnvironmentManager,
    ToolboxBundleSpec,
    ToolboxBundleStager,
    ToolboxBundleTool,
    ToolboxExecutionHarness,
    ToolboxHarnessConfig,
    SandboxedToolboxFacade,
    ToolboxSandboxAssignment,
    ToolboxSandboxOrchestrator,
    ToolboxWorkerStartupSpec,
    is_canceled_tool_error,
    load_toolbox_from_manifest,
    should_resubmit_canceled_tool_call,
)
from mp13_engine.mp13_config import InferenceResponse, ToolCall, ToolCallBlock
from mp13_engine.mp13_toolbox import ToolBoxRef, Toolbox, ToolsScope, ToolsView
from mp13_engine.mp13_tools_parser import DEFAULT_PROFILE


class EngineHostService(_EngineHostService):
    """Exercise toolbox worker behavior beneath the durable-operation envelope."""

    _test_request_sequence = 0

    @staticmethod
    def _worker_result(status):
        row = dict(status or {})
        return dict(row.get("result") or {}) if row.get("contract") == "hosting.operation_status" else row

    def toolbox_execute(self, **kwargs):
        if not str(kwargs.get("execution_request_id") or "").strip():
            type(self)._test_request_sequence += 1
            kwargs["execution_request_id"] = f"toolbox-worker-behavior-{type(self)._test_request_sequence}"
        return self._worker_result(super().toolbox_execute(**kwargs))

    def _test_toolbox_status(self, *, environment_key: str, request_id: str):
        return self._toolbox_runtime_base().request_status(
            environment_key=environment_key,
            request_id=request_id,
        )

    def _test_cancel_toolbox(
        self,
        *,
        request_id: str,
        engine_id: str = "",
        toolbox_id: str = "",
        tool_name: str = "",
        tool_call_id: str = "",
        timeout_seconds: float = 8.0,
        respawn: bool = True,
    ):
        selector = HostedOperationSelector(
            kind="toolbox_id" if toolbox_id else "engine_id",
            id=toolbox_id or engine_id,
        )
        prepared = self._hosted_operations.prepare(
            owner_actor_id="service:local",
            execution_kind=HostedExecutionKind.TOOLBOX,
            selector=selector,
            namespace=f"toolbox:{toolbox_id}" if toolbox_id else f"engine:{engine_id}",
            request_id=request_id,
            fingerprint=hosted_execution_fingerprint({"test_request_id": request_id}),
            metadata={
                "engine_id": engine_id,
                "toolbox_id": toolbox_id,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
            },
        )
        operation_id = prepared["status"]["operation"]["operation_id"]
        self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
        record = self._hosted_operations.get_by_operation_id(operation_id)
        return self._cancel_toolbox_operation(
            record=record,
            timeout_seconds=timeout_seconds,
            respawn=respawn,
        )


def _tool_definition(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "delay": {"type": "number"},
                },
                "required": [],
            },
        },
    }


def _scratch_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"mp13-{prefix}"))


def _capture_permission_error(fn) -> str:
    try:
        fn()
    except PermissionError as exc:
        return str(exc)
    raise AssertionError("Expected PermissionError")


def test_toolbox_bundle_stager_writes_manifest_and_files() -> None:
    root = _scratch_dir("bundle-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-alpha",
                files=[ToolboxBundleFile(relative_path="demo_tools.py", content="def hello(name='x'):\n    return {'name': name}\n")],
                tools=[ToolboxBundleTool(definition=_tool_definition("hello_tool"), entrypoint="demo_tools:hello")],
                dependency_lock_hash="lock-123",
            )
        )
        manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
        assert manifest["bundle_id"] == "bundle-alpha"
        assert manifest["toolbox_id"] == "bundle-alpha"
        assert manifest["sandbox_profile"]["profile_id"] == "default"
        assert manifest["executor_kind"] == "toolbox_executor"
        assert staged.registration_bundle()["manifest_hash"] == manifest["manifest_hash"]
        assert staged.registration_environment()["venv_mutable"] is False
        assert staged.registration_tool_access()["allowed_tool_names"] == ["hello_tool"]
        assert staged.registration_tool_access()["advertised_tool_names"] == ["hello_tool"]
        assert staged.registration_tool_access()["hidden_allowed_tool_names"] == []
        assert (staged.bundle_root / "files" / "demo_tools.py").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_bundle_stager_supports_intrinsic_only_revision() -> None:
    root = _scratch_dir("bundle-intrinsic-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-intrinsic",
                with_intrinsics=True,
                with_intrinsic_guides=True,
                intrinsic_tool_names=["symbolic_algebra", "symbolic_algebra_guide"],
            )
        )
        manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
        assert manifest["with_intrinsics"] is True
        assert manifest["with_intrinsic_guides"] is True
        assert manifest["intrinsic_tool_names"] == ["symbolic_algebra", "symbolic_algebra_guide"]
        assert staged.registration_tool_access()["allowed_tool_names"] == [
            "symbolic_algebra",
            "symbolic_algebra_guide",
        ]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_load_toolbox_from_manifest_supports_intrinsic_only_revision() -> None:
    root = _scratch_dir("load-intrinsic-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-load-intrinsic",
                with_intrinsics=True,
                with_intrinsic_guides=True,
                intrinsic_tool_names=["symbolic_algebra", "symbolic_algebra_guide"],
            )
        )
        toolbox, manifest = load_toolbox_from_manifest(staged.manifest_path)

        assert manifest["intrinsic_tool_names"] == ["symbolic_algebra", "symbolic_algebra_guide"]
        calc_call = ToolCall(
            name="symbolic_algebra",
            arguments={"expr": "2 + 2", "variables": [], "operation": "simplify"},
        )
        calc_out = asyncio.run(toolbox.execute(calc_call))
        assert "4" in str(calc_out or "")

        guide_call = ToolCall(name="symbolic_algebra_guide", arguments={"topic": "help"})
        guide_out = asyncio.run(toolbox.execute(guide_call))
        assert "symbolic_algebra" in str(guide_out or "")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_bundle_stager_supports_auto_callable_discovery() -> None:
    root = _scratch_dir("bundle-auto-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-auto",
                files=[
                    ToolboxBundleFile(
                        relative_path="auto_tools.py",
                        content=(
                            "def hello_auto(name: str = 'world'):\n"
                            "    \"\"\"Return a greeting.\n\n"
                            "    Args:\n"
                            "        name (str): Name to greet.\n"
                            "    \"\"\"\n"
                            "    return {'greeting': f'hi {name}'}\n"
                        ),
                    )
                ],
                auto_tools=[ToolboxBundleAutoTool(module_name="auto_tools", callable_name="hello_auto")],
            )
        )
        manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
        assert manifest["auto_tools"] == [
            {
                "name": "hello_auto",
                "module_name": "auto_tools",
                "callable_name": "hello_auto",
                "activate": True,
                "hidden": False,
                "non_restartable": False,
                "guide_content": None,
                "guide_description": None,
                "callback_signature": None,
            }
        ]
        assert staged.registration_tool_access()["allowed_tool_names"] == ["hello_auto"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_bundle_stager_carries_toolbox_and_profile_metadata() -> None:
    root = _scratch_dir("bundle-profile-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-profile",
                toolbox_id="logical-toolbox",
                sandbox_profile=SandboxProfileSpec(
                    profile_id="net-open",
                    required_imports=["requests"],
                    sandbox_policy={"sandbox": {"enabled": True}},
                ),
                files=[
                    ToolboxBundleFile(
                        relative_path="profile_tools.py",
                        content="def profiled_tool(name='x'):\n    return {'name': name}\n",
                    )
                ],
                auto_tools=[ToolboxBundleAutoTool(module_name="profile_tools", callable_name="profiled_tool")],
            )
        )
        manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
        assert manifest["toolbox_id"] == "logical-toolbox"
        assert manifest["sandbox_profile"]["profile_id"] == "net-open"
        assert manifest["sandbox_profile"]["required_imports"] == ["requests"]
        assert staged.registration_bundle()["toolbox_id"] == "logical-toolbox"
        assert staged.registration_bundle()["sandbox_profile_id"] == "net-open"
        assert staged.registration_tool_access()["tool_routes"]["profiled_tool"] == {
            "toolbox_id": "logical-toolbox",
            "sandbox_profile_id": "net-open",
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_bundle_stager_tracks_hidden_user_tool_membership() -> None:
    root = _scratch_dir("bundle-hidden-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-hidden",
                files=[ToolboxBundleFile(relative_path="demo_tools.py", content="def hello(name='x'):\n    return {'name': name}\n")],
                tools=[ToolboxBundleTool(definition=_tool_definition("hello_tool"), entrypoint="demo_tools:hello", hidden=True)],
            )
        )

        manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
        assert manifest["hidden_tool_names"] == ["hello_tool"]
        assert staged.registration_tool_access()["allowed_tool_names"] == ["hello_tool"]
        assert staged.registration_tool_access()["advertised_tool_names"] == []
        assert staged.registration_tool_access()["hidden_allowed_tool_names"] == ["hello_tool"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_load_toolbox_from_manifest_restores_hidden_user_tool_names() -> None:
    root = _scratch_dir("load-hidden-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-load-hidden",
                files=[ToolboxBundleFile(relative_path="demo_tools.py", content="def hello(name='x'):\n    return {'name': name}\n")],
                tools=[ToolboxBundleTool(definition=_tool_definition("hello_tool"), entrypoint="demo_tools:hello", hidden=True)],
            )
        )

        toolbox, _ = load_toolbox_from_manifest(staged.manifest_path)

        assert toolbox.hidden_tool_names == ["hello_tool"]
        gate = toolbox.gate_call("hello_tool")
        assert gate.outcome == "allowed"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_sandbox_profile_spec_derives_stable_profile_id() -> None:
    profile = SandboxProfileSpec(
        required_imports=["requests", "numpy", "requests"],
        sandbox_policy={"sandbox": {"enabled": True, "network": {"mode": "brokered_only"}}},
    )
    profile_id = profile.normalized_profile_id()
    assert profile_id.startswith("profile-")
    assert profile.normalized_required_imports() == ["requests", "numpy"]


def test_native_toolbox_gate_call_reports_denied_and_allowed() -> None:
    toolbox = Toolbox()
    ok, msg = toolbox.add_tool_external(_tool_definition("hello_tool"), lambda name="x": {"name": name}, activate=True)
    assert ok, msg
    allowed = toolbox.gate_call("hello_tool")
    denied = toolbox.gate_call("missing_tool")
    assert allowed.outcome == "allowed"
    assert allowed.reason == "allowed"
    assert allowed.executable is True
    assert denied.outcome == "denied"
    assert denied.reason == "tool_not_defined"
    assert denied.executable is False


def test_native_toolbox_build_view_and_gate_call_support_gated_tools() -> None:
    toolbox = Toolbox()
    ok, msg = toolbox.add_tool_external(_tool_definition("hello_tool"), lambda name="x": {"name": name}, activate=True)
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"hello_tool"},
                gated_tools={"hello_tool"},
            )
        ]
    )

    assert view.advertised_tools == {"hello_tool"}
    assert view.allowed_tools == set()
    assert view.gated_tools == {"hello_tool"}
    assert view.disabled_tools == set()

    gated = toolbox.gate_call("hello_tool", tools_view=view)
    assert gated.outcome == "gated_requires_confirmation"
    assert gated.reason == "gated_requires_confirmation"
    assert gated.executable is False
    assert gated.requires_confirmation is True


def test_native_user_guide_is_not_implicitly_gated_with_paired_tool() -> None:
    toolbox = Toolbox()

    def _search_files(name_mask: str = "*") -> Dict[str, object]:
        return {"ok": True, "name_mask": name_mask}

    ok, msg = toolbox.add_tool_callable(
        _search_files,
        activate=True,
        guide_content={"usage": ["Use this tool to search under the current scoped root."]},
    )
    assert ok, msg
    toolbox.activate_tool("_search_files")

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"_search_files", "_search_files_guide"},
                gated_tools={"_search_files"},
            )
        ]
    )

    tool_gate = toolbox.gate_call("_search_files", tools_view=view)
    guide_gate = toolbox.gate_call("_search_files_guide", tools_view=view)

    assert tool_gate.outcome == "gated_requires_confirmation"
    assert guide_gate.outcome == "allowed"

    guide_call = ToolCall(name="_search_files_guide", arguments={"topic": "usage"})
    guide_out = asyncio.run(toolbox.execute(guide_call, tools_view=view))

    assert guide_call.error is None
    assert "scoped root" in str(guide_out or "")


def test_native_user_guide_can_be_explicitly_gated_by_name() -> None:
    toolbox = Toolbox()

    def _search_files(name_mask: str = "*") -> Dict[str, object]:
        return {"ok": True, "name_mask": name_mask}

    ok, msg = toolbox.add_tool_callable(
        _search_files,
        activate=True,
        guide_content={"usage": ["Use this tool to search under the current scoped root."]},
    )
    assert ok, msg
    toolbox.activate_tool("_search_files")

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"_search_files", "_search_files_guide"},
                gated_tools={"_search_files_guide"},
            )
        ]
    )

    guide_gate = toolbox.gate_call("_search_files_guide", tools_view=view)
    assert guide_gate.outcome == "gated_requires_confirmation"

    guide_call = ToolCall(name="_search_files_guide", arguments={"topic": "usage"})
    guide_out = asyncio.run(toolbox.execute(guide_call, tools_view=view))

    assert guide_out is None
    assert guide_call.error == "Error: Tool '_search_files_guide' requires confirmation before execution."


def test_native_toolbox_build_view_merges_tool_constraints() -> None:
    toolbox = Toolbox()
    ok, msg = toolbox.add_tool_external(_tool_definition("search_files"), lambda name_mask="*": {"ok": True}, activate=True)
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"search_files"},
                tool_constraints={
                    "search_files": {
                        "domains": {
                            "filesystem": {
                                "implied_root": "docs",
                                "allowed_roots": ["docs"],
                            }
                        }
                    }
                },
            ),
            ToolsScope(
                tool_constraints={
                    "search_files": {
                        "domains": {
                            "filesystem": {
                                "implied_root": "src",
                                "allowed_roots": ["src"],
                            }
                        }
                    }
                }
            ),
        ]
    )

    assert view.advertised_tools == {"search_files"}
    assert view.get_constraints("search_files") == {
        "domains": {
            "filesystem": {
                "implied_root": "src",
                "allowed_roots": ["src"],
            }
        }
    }


def test_native_toolbox_build_view_shallow_merges_partial_tool_constraints() -> None:
    toolbox = Toolbox()
    ok, msg = toolbox.add_tool_external(_tool_definition("search_files"), lambda name_mask="*": {"ok": True}, activate=True)
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
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
            ),
            ToolsScope(
                tool_constraints={
                    "search_files": {
                        "argument_policy": {
                            "locked_args": ["root_path"],
                        }
                    }
                }
            ),
        ]
    )

    assert view.get_constraints("search_files") == {
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


def test_native_toolbox_build_view_allows_later_scope_to_clear_tool_constraints() -> None:
    toolbox = Toolbox()
    ok, msg = toolbox.add_tool_external(_tool_definition("search_files"), lambda name_mask="*": {"ok": True}, activate=True)
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"search_files"},
                tool_constraints={
                    "search_files": {
                        "domains": {
                            "filesystem": {
                                "implied_root": "docs",
                                "allowed_roots": ["docs"],
                            }
                        }
                    }
                },
            ),
            ToolsScope(
                tool_constraints={
                    "search_files": None,
                }
            ),
        ]
    )

    assert view.get_constraints("search_files") == {}


def test_native_toolbox_execute_applies_implied_args_from_scope_constraints() -> None:
    toolbox = Toolbox()
    captured: dict[str, object] = {}

    def _search_files(name_mask: str, root_path: str = "") -> Dict[str, object]:
        captured["name_mask"] = name_mask
        captured["root_path"] = root_path
        return {"name_mask": name_mask, "root_path": root_path}

    ok, msg = toolbox.add_tool_external(_tool_definition("search_files"), _search_files, activate=True)
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"search_files"},
                tool_constraints={
                    "search_files": {
                        "argument_policy": {
                            "implied_args": {"root_path": "docs"},
                            "locked_args": ["root_path"],
                        }
                    }
                },
            )
        ]
    )

    call = ToolCall(name="search_files", arguments={"name_mask": "*.md"})
    out = asyncio.run(toolbox.execute(call, tools_view=view))

    assert call.error is None
    assert json.loads(str(out or "{}")) == {"name_mask": "*.md", "root_path": "docs"}
    assert captured == {"name_mask": "*.md", "root_path": "docs"}


def test_native_toolbox_execute_rejects_locked_arg_override_from_scope_constraints() -> None:
    toolbox = Toolbox()
    ok, msg = toolbox.add_tool_external(
        _tool_definition("search_files"),
        lambda name_mask="*", root_path="": {"name_mask": name_mask, "root_path": root_path},
        activate=True,
    )
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"search_files"},
                tool_constraints={
                    "search_files": {
                        "argument_policy": {
                            "implied_args": {"root_path": "docs"},
                            "locked_args": ["root_path"],
                        }
                    }
                },
            )
        ]
    )

    call = ToolCall(name="search_files", arguments={"name_mask": "*.md", "root_path": "src"})
    out = asyncio.run(toolbox.execute(call, tools_view=view))

    assert out is None
    assert call.error == "Error executing tool 'search_files': PermissionError - Tool 'search_files' argument 'root_path' is locked by scope constraints."


def test_native_toolbox_execute_normalizes_path_under_implied_root() -> None:
    toolbox = Toolbox()
    captured: dict[str, object] = {}

    def _search_files(name_mask: str, root_path: str = "") -> Dict[str, object]:
        captured["root_path"] = root_path
        return {"name_mask": name_mask, "root_path": root_path}

    ok, msg = toolbox.add_tool_external(_tool_definition("search_files"), _search_files, activate=True)
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"search_files"},
                tool_constraints={
                    "search_files": {
                        "domains": {
                            "filesystem": {
                                "implied_root": "docs/api",
                                "allowed_roots": ["docs/api"],
                                "allow_explicit_root_override": True,
                            }
                        },
                        "argument_policy": {
                            "normalizers": {"root_path": "path_under_implied_root"},
                        },
                    }
                },
            )
        ]
    )

    call = ToolCall(name="search_files", arguments={"name_mask": "*.md", "root_path": r"docs\api\guides\..\reference"})
    out = asyncio.run(toolbox.execute(call, tools_view=view))

    assert call.error is None
    assert json.loads(str(out or "{}")) == {"name_mask": "*.md", "root_path": "docs/api/reference"}
    assert captured["root_path"] == "docs/api/reference"


def test_native_toolbox_execute_rejects_path_outside_allowed_scoped_root() -> None:
    toolbox = Toolbox()
    ok, msg = toolbox.add_tool_external(
        _tool_definition("search_files"),
        lambda name_mask="*", root_path="": {"name_mask": name_mask, "root_path": root_path},
        activate=True,
    )
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"search_files"},
                tool_constraints={
                    "search_files": {
                        "domains": {
                            "filesystem": {
                                "implied_root": "docs/api",
                                "allowed_roots": ["docs/api"],
                                "allow_explicit_root_override": True,
                            }
                        },
                        "argument_policy": {
                            "normalizers": {"root_path": "path_under_implied_root"},
                        },
                    }
                },
            )
        ]
    )

    call = ToolCall(name="search_files", arguments={"name_mask": "*.md", "root_path": "../secret"})
    out = asyncio.run(toolbox.execute(call, tools_view=view))

    assert out is None
    assert call.error == "Error executing tool 'search_files': PermissionError - Tool 'search_files' argument 'root_path' is outside the allowed scoped roots."


def test_native_toolbox_execute_rejects_url_outside_allowed_scoped_prefix() -> None:
    toolbox = Toolbox()
    ok, msg = toolbox.add_tool_external(
        _tool_definition("fetch_url"),
        lambda url="": {"url": url},
        activate=True,
    )
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"fetch_url"},
                tool_constraints={
                    "fetch_url": {
                        "domains": {
                            "network": {
                                "implied_url_prefix": "https://example.com/api/",
                                "allowed_url_prefixes": ["https://example.com/api/"],
                                "allow_explicit_url_override": True,
                            }
                        },
                        "argument_policy": {
                            "normalizers": {"url": "url_under_implied_prefix"},
                        },
                    }
                },
            )
        ]
    )

    call = ToolCall(name="fetch_url", arguments={"url": "https://example.org/api/test"})
    out = asyncio.run(toolbox.execute(call, tools_view=view))

    assert out is None
    assert call.error == "Error executing tool 'fetch_url': PermissionError - Tool 'fetch_url' argument 'url' is outside the allowed scoped URL prefixes."


def test_native_toolbox_execute_injects_resolved_tool_constraints_into_kwargs_tools() -> None:
    toolbox = Toolbox()
    captured: dict[str, object] = {}

    def _search_files(name_mask: str, root_path: str = "", **kwargs: Any) -> Dict[str, object]:
        captured["root_path"] = root_path
        captured["tool_constraints"] = kwargs.get("tool_constraints")
        captured["tools_view"] = kwargs.get("tools_view")
        captured["tool_call_name"] = getattr(kwargs.get("tool_call"), "name", None)
        return {
            "root_path": root_path,
            "constraint_root": dict(
                dict(dict(kwargs.get("tool_constraints") or {}).get("domains") or {}).get("filesystem") or {}
            ).get("implied_root"),
        }

    ok, msg = toolbox.add_tool_external(_tool_definition("search_files"), _search_files, activate=True)
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"search_files"},
                tool_constraints={
                    "search_files": {
                        "domains": {
                            "filesystem": {
                                "implied_root": "docs/api",
                                "allowed_roots": ["docs/api"],
                            }
                        },
                        "argument_policy": {
                            "implied_args": {"root_path": "docs/api"},
                            "locked_args": ["root_path"],
                        },
                    }
                },
            )
        ]
    )

    call = ToolCall(name="search_files", arguments={"name_mask": "*.md"})
    out = asyncio.run(toolbox.execute(call, tools_view=view))

    assert call.error is None
    assert json.loads(str(out or "{}")) == {"root_path": "docs/api", "constraint_root": "docs/api"}
    assert captured["root_path"] == "docs/api"
    assert captured["tool_constraints"] == {
        "domains": {
            "filesystem": {
                "implied_root": "docs/api",
                "allowed_roots": ["docs/api"],
            }
        },
        "argument_policy": {
            "implied_args": {"root_path": "docs/api"},
            "locked_args": ["root_path"],
        },
    }
    assert isinstance(captured["tools_view"], ToolsView)
    assert captured["tool_call_name"] == "search_files"


def test_native_toolbox_execute_injects_tool_constraints_helper_into_kwargs_tools() -> None:
    toolbox = Toolbox()
    captured: dict[str, object] = {}

    def _search_files(name_mask: str, root_path: str = "", **kwargs: Any) -> Dict[str, object]:
        helper = kwargs.get("tool_constraints_view")
        captured["helper"] = helper
        return {
            "filesystem_root": helper.get_domain("filesystem").get("implied_root"),
            "implied_root_arg": helper.get_implied_arg("root_path"),
            "root_path_locked": helper.is_arg_locked("root_path"),
            "root_path_normalizer": helper.get_normalizer("root_path"),
            "raw_root_path": root_path,
        }

    ok, msg = toolbox.add_tool_external(_tool_definition("search_files"), _search_files, activate=True)
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"search_files"},
                tool_constraints={
                    "search_files": {
                        "domains": {
                            "filesystem": {
                                "implied_root": "docs/api",
                                "allowed_roots": ["docs/api"],
                            }
                        },
                        "argument_policy": {
                            "implied_args": {"root_path": "docs/api"},
                            "locked_args": ["root_path"],
                            "normalizers": {"root_path": "path_under_implied_root"},
                        },
                    }
                },
            )
        ]
    )

    call = ToolCall(name="search_files", arguments={"name_mask": "*.md"})
    out = asyncio.run(toolbox.execute(call, tools_view=view))

    assert call.error is None
    assert json.loads(str(out or "{}")) == {
        "filesystem_root": "docs/api",
        "implied_root_arg": "docs/api",
        "root_path_locked": True,
        "root_path_normalizer": "path_under_implied_root",
        "raw_root_path": "docs/api",
    }
    assert captured["helper"].__class__.__name__ == "ToolConstraintsView"


def test_native_toolbox_execute_tool_constraints_helper_resolves_filesystem_root_and_url() -> None:
    toolbox = Toolbox()

    def _scoped_tool(**kwargs: Any) -> Dict[str, object]:
        helper = kwargs.get("tool_constraints_view")
        return {
            "default_root": helper.resolve_filesystem_root(),
            "normalized_root": helper.resolve_filesystem_root(r"docs\api\guides\..\reference"),
            "default_url": helper.resolve_url(),
        }

    ok, msg = toolbox.add_tool_external(_tool_definition("scoped_tool"), _scoped_tool, activate=True)
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"scoped_tool"},
                tool_constraints={
                    "scoped_tool": {
                        "domains": {
                            "filesystem": {
                                "implied_root": "docs/api",
                                "allowed_roots": ["docs/api"],
                                "allow_explicit_root_override": True,
                            },
                            "network": {
                                "implied_url_prefix": "https://example.com/api/",
                                "allowed_url_prefixes": ["https://example.com/api/"],
                                "allow_explicit_url_override": True,
                            },
                        },
                        "argument_policy": {
                            "implied_args": {
                                "root_path": "docs/api",
                                "url": "https://example.com/api/",
                            },
                            "normalizers": {
                                "root_path": "path_under_implied_root",
                                "url": "url_under_implied_prefix",
                            },
                        },
                    }
                },
            )
        ]
    )

    call = ToolCall(name="scoped_tool", arguments={})
    out = asyncio.run(toolbox.execute(call, tools_view=view))

    assert call.error is None
    assert json.loads(str(out or "{}")) == {
        "default_root": "docs/api",
        "normalized_root": "docs/api/reference",
        "default_url": "https://example.com/api/",
    }


def test_native_toolbox_execute_tool_constraints_helper_rejects_locked_override() -> None:
    toolbox = Toolbox()

    def _scoped_tool(**kwargs: Any) -> Dict[str, object]:
        helper = kwargs.get("tool_constraints_view")
        return {
            "locked_root_error": _capture_permission_error(lambda: helper.resolve_filesystem_root("src")),
        }

    ok, msg = toolbox.add_tool_external(_tool_definition("scoped_tool_locked"), _scoped_tool, activate=True)
    assert ok, msg

    view = toolbox.build_view(
        [
            ToolsScope(
                advertise_tools={"scoped_tool_locked"},
                tool_constraints={
                    "scoped_tool_locked": {
                        "domains": {
                            "filesystem": {
                                "implied_root": "docs/api",
                                "allowed_roots": ["docs/api"],
                                "allow_explicit_root_override": True,
                            }
                        },
                        "argument_policy": {
                            "implied_args": {"root_path": "docs/api"},
                            "locked_args": ["root_path"],
                            "normalizers": {"root_path": "path_under_implied_root"},
                        },
                    }
                },
            )
        ]
    )

    call = ToolCall(name="scoped_tool_locked", arguments={})
    out = asyncio.run(toolbox.execute(call, tools_view=view))

    assert call.error is None
    assert json.loads(str(out or "{}")) == {
        "locked_root_error": "Tool 'scoped_tool_locked' argument 'root_path' is locked by scope constraints.",
    }


def test_toolbox_execution_harness_executes_request_tools_via_hosted_toolbox() -> None:
    class _FakeChannel:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def toolbox_gate(self, **kwargs):
            self.calls.append({"gate": dict(kwargs)})
            return {
                "status": "ok",
                "outcome": "allowed",
                "reason": "allowed",
                "tool_name": kwargs.get("tool_name"),
            }

        def toolbox_execute(self, **kwargs):
            self.calls.append(dict(kwargs))
            tool_call = dict(kwargs.get("tool_call") or {})
            return {
                "contract": "hosting.operation_status",
                "api_status": "ok",
                "lifecycle": "terminal_success",
                "operation": {"contract": "hosting.operation_ref", "operation_id": "op-test"},
                "result": {
                    "status": "ok",
                    "tool_call": {
                        **tool_call,
                        "result": json.dumps({"greeting": f"hi {dict(tool_call.get('arguments') or {}).get('name', 'world')}"}),
                    },
                },
            }

    events: list[str] = []

    async def _action_handler(execute_stage: str, **kwargs):
        events.append(str(execute_stage))
        return None

    response = InferenceResponse(
        chunkType="streaming_chunk",
        prompt_index=0,
        tool_blocks=[
            ToolCallBlock(
                raw_block='<tool_call>{"name":"hello_remote","arguments":{"name":"Sam"}}</tool_call>'
            )
        ],
    )
    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_toolbox_id="user-tools"),
        control_channel=_FakeChannel(),
    )

    asyncio.run(
        harness.execute_request_tools(
            parser_profile=DEFAULT_PROFILE,
            final_response_items=[response],
            action_handler=_action_handler,
            serial_execution=True,
        )
    )

    block = list(response.tool_blocks or [])[0]
    call = list(block.calls or [])[0]
    assert call.name == "hello_remote"
    assert json.loads(str(call.result or "")) == {"greeting": "hi Sam"}
    assert call.execution_envelope["contract"] == "hosting.operation_status"
    assert call.execution_envelope["operation"]["operation_id"] == "op-test"
    assert events == ["calls_parsed", "call_starting", "call_finished", "all_finished"]


def test_toolbox_execution_harness_reports_hosted_gate_denial_before_execute() -> None:
    class _FakeChannel:
        def __init__(self) -> None:
            self.executed = False

        def toolbox_gate(self, **kwargs):
            return {
                "status": "ok",
                "outcome": "denied",
                "reason": "tool_not_allowed",
                "tool_name": kwargs.get("tool_name"),
            }

        def toolbox_execute(self, **kwargs):
            self.executed = True
            return {"status": "ok", "tool_call": dict(kwargs.get("tool_call") or {})}

    channel = _FakeChannel()
    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_toolbox_id="user-tools"),
        control_channel=channel,
    )

    call = asyncio.run(
        harness.execute_calls(
            [ToolCall(name="scriptable_calculator", arguments={"expr": "12 + 3 * 5"})],
            parallel=False,
        )
    )[0]

    assert str(call.error or "") == "Execution gated: denied - tool_not_allowed:scriptable_calculator"
    assert call.result is None
    assert channel.executed is False


def test_toolbox_execution_harness_forwards_tools_view_to_hosted_gate_and_execute() -> None:
    class _FakeChannel:
        def __init__(self) -> None:
            self.gate_calls: list[dict[str, object]] = []
            self.execute_calls: list[dict[str, object]] = []

        def toolbox_gate(self, **kwargs):
            self.gate_calls.append(dict(kwargs))
            return {
                "status": "ok",
                "outcome": "allowed",
                "reason": "allowed",
                "tool_name": kwargs.get("tool_name"),
            }

        def toolbox_execute(self, **kwargs):
            self.execute_calls.append(dict(kwargs))
            tool_call = dict(kwargs.get("tool_call") or {})
            return {"status": "ok", "tool_call": tool_call | {"result": json.dumps({"ok": True})}}

    view = ToolsView(
        view_id="req-1",
        mode="advertised",
        allowed_tools={"hello_remote"},
        advertised_tools={"hello_remote"},
        hidden_allowed_tools=set(),
        disabled_tools={"blocked_tool"},
    )
    channel = _FakeChannel()
    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_toolbox_id="user-tools"),
        control_channel=channel,
    )

    call = asyncio.run(
        harness.execute_calls(
            [ToolCall(name="hello_remote", arguments={})],
            parallel=False,
            native_execute_kwargs={"tools_view": view},
        )
    )[0]

    assert call.error is None
    assert json.loads(str(call.result or "{}")) == {"ok": True}
    assert channel.gate_calls[0]["tools_view"] == {
        "view_id": "req-1",
        "mode": "advertised",
        "allowed_tools": ["hello_remote"],
        "advertised_tools": ["hello_remote"],
        "hidden_allowed_tools": [],
        "disabled_tools": ["blocked_tool"],
        "gated_tools": [],
    }
    assert channel.execute_calls[0]["tools_view"] == channel.gate_calls[0]["tools_view"]


def test_toolbox_execution_harness_approval_allow_once_executes_gated_tool() -> None:
    class _FakeChannel:
        def __init__(self) -> None:
            self.gate_calls: list[dict[str, object]] = []
            self.execute_calls: list[dict[str, object]] = []

        def toolbox_gate(self, **kwargs):
            self.gate_calls.append(dict(kwargs))
            view = dict(kwargs.get("tools_view") or {})
            gated = set(view.get("gated_tools") or [])
            name = str(kwargs.get("tool_name") or "").strip()
            if name in gated:
                return {
                    "status": "ok",
                    "outcome": "gated_requires_confirmation",
                    "reason": "gated_requires_confirmation",
                    "tool_name": name,
                    "requires_confirmation": True,
                }
            return {"status": "ok", "outcome": "allowed", "reason": "allowed", "tool_name": name}

        def toolbox_execute(self, **kwargs):
            self.execute_calls.append(dict(kwargs))
            tool_call = dict(kwargs.get("tool_call") or {})
            return {"status": "ok", "tool_call": tool_call | {"result": json.dumps({"ok": True})}}

    decisions: list[dict[str, Any]] = []

    def _processor(*, callback_name: str, payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
        decisions.append({"callback_name": callback_name, "payload": dict(payload or {}), "tool_name": context.tool_name})
        return {"decision": "allow_once"}

    view = ToolsView(
        view_id="req-approve-once",
        mode="advertised",
        allowed_tools=set(),
        advertised_tools={"dangerous_remote"},
        hidden_allowed_tools=set(),
        disabled_tools=set(),
        gated_tools={"dangerous_remote"},
    )
    channel = _FakeChannel()
    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_toolbox_id="user-tools"),
        control_channel=channel,
    )

    call = asyncio.run(
        harness.execute_calls(
            [ToolCall(name="dangerous_remote", arguments={"path": "x"})],
            parallel=False,
            native_execute_kwargs={"tools_view": view},
            callback_processor=_processor,
        )
    )[0]

    assert call.error is None
    assert json.loads(str(call.result or "{}")) == {"ok": True}
    assert decisions == [
        {
            "callback_name": "tool_requires_confirmation",
            "payload": {
                "kind": "tool_approval_request",
                "decision_options": ["deny", "allow_once", "add_to_scope"],
                "tool_name": "dangerous_remote",
                "tool_call_id": call.id,
                "tool_arguments": {"path": "x"},
                "gate": {
                    "outcome": "gated_requires_confirmation",
                    "reason": "gated_requires_confirmation",
                    "requires_confirmation": True,
                },
                "tools_view": {
                    "view_id": "req-approve-once",
                    "mode": "advertised",
                    "allowed_tools": [],
                    "advertised_tools": ["dangerous_remote"],
                    "hidden_allowed_tools": [],
                    "disabled_tools": [],
                    "gated_tools": ["dangerous_remote"],
                },
            },
            "tool_name": "dangerous_remote",
        }
    ]
    assert channel.execute_calls[0]["tools_view"] == {
        "view_id": "req-approve-once",
        "mode": "advertised",
        "allowed_tools": ["dangerous_remote"],
        "advertised_tools": ["dangerous_remote"],
        "hidden_allowed_tools": [],
        "disabled_tools": [],
        "gated_tools": [],
    }
    assert view.gated_tools == {"dangerous_remote"}
    assert view.allowed_tools == set()


def test_toolbox_execution_harness_approval_add_to_scope_mutates_future_calls() -> None:
    class _FakeChannel:
        def __init__(self) -> None:
            self.gate_calls: list[dict[str, object]] = []
            self.execute_calls: list[dict[str, object]] = []

        def toolbox_gate(self, **kwargs):
            self.gate_calls.append(dict(kwargs))
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
            return {"status": "ok", "outcome": "allowed", "reason": "allowed", "tool_name": name}

        def toolbox_execute(self, **kwargs):
            self.execute_calls.append(dict(kwargs))
            tool_call = dict(kwargs.get("tool_call") or {})
            return {"status": "ok", "tool_call": tool_call | {"result": json.dumps({"name": tool_call.get("name")})}}

    seen_callbacks: list[str] = []

    def _processor(*, callback_name: str, payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
        seen_callbacks.append(callback_name)
        return {"decision": "add_to_scope"}

    toolbox = Toolbox()
    scope_ref = ToolBoxRef(toolbox=toolbox, scope=ToolsScope(gated_tools={"dangerous_remote"}))
    view = ToolsView(
        view_id="req-add-scope",
        mode="advertised",
        allowed_tools=set(),
        advertised_tools={"dangerous_remote"},
        hidden_allowed_tools=set(),
        disabled_tools=set(),
        gated_tools={"dangerous_remote"},
    )
    channel = _FakeChannel()
    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_toolbox_id="user-tools"),
        control_channel=channel,
    )

    calls = asyncio.run(
        harness.execute_calls(
            [
                ToolCall(name="dangerous_remote", arguments={}),
                ToolCall(name="dangerous_remote", arguments={}),
            ],
            parallel=False,
            native_execute_kwargs={"tools_view": view},
            callback_processor=_processor,
            callback_context={"toolbox_ref": scope_ref},
        )
    )

    assert [call.error for call in calls] == [None, None]
    assert seen_callbacks == ["tool_requires_confirmation"]
    assert view.gated_tools == set()
    assert view.allowed_tools == {"dangerous_remote"}
    assert scope_ref.scope.gated_tools == set()
    assert len(channel.execute_calls) == 2
    assert channel.gate_calls[1]["tools_view"]["gated_tools"] == []
    assert channel.gate_calls[1]["tools_view"]["allowed_tools"] == ["dangerous_remote"]


def test_toolbox_execution_harness_approval_add_to_scope_persists_scope_constraints() -> None:
    class _FakeChannel:
        def __init__(self) -> None:
            self.execute_calls: list[dict[str, object]] = []

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
            return {"status": "ok", "outcome": "allowed", "reason": "allowed", "tool_name": name}

        def toolbox_execute(self, **kwargs):
            self.execute_calls.append(dict(kwargs))
            tool_call = dict(kwargs.get("tool_call") or {})
            return {"status": "ok", "tool_call": tool_call | {"result": json.dumps({"ok": True})}}

    def _processor(*, callback_name: str, payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
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

    scope_ref = ToolBoxRef(toolbox=Toolbox(), scope=ToolsScope(gated_tools={"dangerous_remote"}))
    view = ToolsView(
        view_id="req-constraints",
        mode="advertised",
        allowed_tools=set(),
        advertised_tools={"dangerous_remote"},
        hidden_allowed_tools=set(),
        disabled_tools=set(),
        gated_tools={"dangerous_remote"},
    )
    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_toolbox_id="user-tools"),
        control_channel=_FakeChannel(),
    )

    call = asyncio.run(
        harness.execute_calls(
            [ToolCall(name="dangerous_remote", arguments={})],
            parallel=False,
            native_execute_kwargs={"tools_view": view},
            callback_processor=_processor,
            callback_context={"toolbox_ref": scope_ref},
        )
    )[0]

    assert call.error is None
    assert view.get_constraints("dangerous_remote") == {
        "domains": {
            "filesystem": {
                "implied_root": "docs",
                "allowed_roots": ["docs"],
            }
        }
    }
    assert scope_ref.scope.tool_constraints["dangerous_remote"] == {
        "domains": {
            "filesystem": {
                "implied_root": "docs",
                "allowed_roots": ["docs"],
            }
        }
    }


def test_hosted_toolbox_ref_execute_scope_ref_convenience_persists_add_to_scope() -> None:
    class _FakeHost:
        def __init__(self) -> None:
            self.execute_calls: list[dict[str, object]] = []

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
            return {"status": "ok", "outcome": "allowed", "reason": "allowed", "tool_name": name}

        def toolbox_execute(self, **kwargs):
            self.execute_calls.append(dict(kwargs))
            tool_call = dict(kwargs.get("tool_call") or {})
            return {"status": "ok", "tool_call": tool_call | {"result": json.dumps({"ok": True})}}

    def _processor(*, callback_name: str, payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
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

    scope_ref = ToolBoxRef(toolbox=Toolbox(), scope=ToolsScope(gated_tools={"dangerous_remote"}))
    view = ToolsView(
        view_id="ref-constraints",
        mode="advertised",
        allowed_tools=set(),
        advertised_tools={"dangerous_remote"},
        hidden_allowed_tools=set(),
        disabled_tools=set(),
        gated_tools={"dangerous_remote"},
    )
    ref = HostedToolBoxRef(toolbox_id="user-tools", host=_FakeHost())

    out = ref.execute(
        tool_name="dangerous_remote",
        execution_request_id="exec-dangerous-remote",
        arguments={},
        tools_view=view,
        callback_processor=_processor,
        scope_ref=scope_ref,
    )

    tool_call = dict(out.get("tool_call") or {})
    assert tool_call.get("error") is None
    assert view.gated_tools == set()
    assert scope_ref.scope.gated_tools == set()
    assert scope_ref.scope.tool_constraints["dangerous_remote"] == {
        "domains": {
            "filesystem": {
                "implied_root": "docs",
                "allowed_roots": ["docs"],
            }
        }
    }


def test_toolbox_execution_harness_approval_timeout_defaults_to_deny() -> None:
    class _FakeChannel:
        def toolbox_gate(self, **kwargs):
            return {
                "status": "ok",
                "outcome": "gated_requires_confirmation",
                "reason": "gated_requires_confirmation",
                "tool_name": kwargs.get("tool_name"),
                "requires_confirmation": True,
            }

        def toolbox_execute(self, **kwargs):
            raise AssertionError("toolbox_execute should not run after timed-out approval")

    def _processor(*, callback_name: str, payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
        time.sleep(0.10)
        return {"decision": "allow_once"}

    view = ToolsView(
        view_id="req-timeout",
        mode="advertised",
        allowed_tools=set(),
        advertised_tools={"dangerous_remote"},
        hidden_allowed_tools=set(),
        disabled_tools=set(),
        gated_tools={"dangerous_remote"},
    )
    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_toolbox_id="user-tools"),
        control_channel=_FakeChannel(),
    )

    call = asyncio.run(
        harness.execute_calls(
            [ToolCall(name="dangerous_remote", arguments={})],
            parallel=False,
            native_execute_kwargs={"tools_view": view},
            callback_processor=_processor,
            callback_context={"approval_timeout_seconds": 0.01},
        )
    )[0]

    assert call.result is None
    assert str(call.error or "") == "Execution gated: denied - gated_requires_confirmation:dangerous_remote"


def test_toolbox_environment_manager_derives_stable_environment_identity() -> None:
    from mp13_engine.mp13_intrinsics_metadata import intrinsic_dependency_profile_id

    root = _scratch_dir("env-spec-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-env-spec",
                sandbox_profile=SandboxProfileSpec(
                    required_imports=["requests", "numpy", "requests"],
                    sandbox_policy={"sandbox": {"enabled": True}},
                ),
                with_intrinsics=True,
                intrinsic_tool_names=["symbolic_algebra"],
            )
        )
        manager = ToolboxEnvironmentManager(root)
        spec = manager.environment_spec_for_bundle(staged)

        assert spec.venv_key
        assert spec.intrinsics_profile_id == intrinsic_dependency_profile_id(["symbolic_algebra"])
        assert spec.required_imports == ["requests", "numpy", "sympy"]
        assert spec.venv_path.endswith(spec.venv_key)
        assert "toolbox_venvs" in spec.venv_path.replace("\\", "/")
        assert spec.python_executable
        assert staged.registration_environment(spec)["venv_key"] == spec.venv_key
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_environment_manager_reuses_environment_for_same_profile() -> None:
    root = _scratch_dir("env-reuse-")
    try:
        stager = ToolboxBundleStager(root)
        first = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-env-reuse",
                sandbox_profile=SandboxProfileSpec(
                    required_imports=["requests"],
                    sandbox_policy={"sandbox": {"enabled": True}},
                ),
                files=[
                    ToolboxBundleFile(
                        relative_path="alpha.py",
                        content="def alpha_tool():\n    return {'tool': 'alpha'}\n",
                    )
                ],
                auto_tools=[ToolboxBundleAutoTool(module_name="alpha", callable_name="alpha_tool")],
            )
        )
        second = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-env-reuse",
                sandbox_profile=SandboxProfileSpec(
                    required_imports=["requests"],
                    sandbox_policy={"sandbox": {"enabled": True}},
                ),
                files=[
                    ToolboxBundleFile(
                        relative_path="alpha.py",
                        content="def alpha_tool():\n    return {'tool': 'alpha-v2'}\n",
                    ),
                    ToolboxBundleFile(
                        relative_path="beta.py",
                        content="def beta_tool():\n    return {'tool': 'beta'}\n",
                    ),
                ],
                auto_tools=[
                    ToolboxBundleAutoTool(module_name="alpha", callable_name="alpha_tool"),
                    ToolboxBundleAutoTool(module_name="beta", callable_name="beta_tool"),
                ],
            )
        )
        manager = ToolboxEnvironmentManager(root)

        first_spec = manager.ensure_for_bundle(first)
        second_spec = manager.ensure_for_bundle(second)

        assert first_spec.venv_key == second_spec.venv_key
        assert first_spec.venv_path == second_spec.venv_path
        assert Path(first_spec.python_executable).exists()
        assert (Path(first_spec.venv_path) / "pyvenv.cfg").exists()
        metadata = json.loads((Path(first_spec.venv_path) / "environment.json").read_text(encoding="utf-8"))
        assert metadata["venv_key"] == first_spec.venv_key
        assert metadata["python_executable"] == first_spec.python_executable
        assert metadata["required_imports"] == ["requests"]
    finally:
        shutil.rmtree(root, ignore_errors=True)

def test_toolbox_environment_runtime_python_uses_venv_when_no_packages_are_planned() -> None:
    root = _scratch_dir("env-runtime-python-noop-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-runtime-python-fallback",
                files=[ToolboxBundleFile(relative_path="demo_tools.py", content="def hello(name='x'):\n    return {'name': name}\n")],
                tools=[ToolboxBundleTool(definition=_tool_definition("hello_tool"), entrypoint="demo_tools:hello")],
            )
        )
        manager = ToolboxEnvironmentManager(root)
        spec = manager.ensure_for_bundle(staged)

        chosen = manager.runtime_python_executable(spec, bootstrap_python_executable="python-bootstrap")

        assert Path(chosen).resolve() == Path(spec.python_executable).resolve()
        metadata = json.loads((Path(spec.venv_path) / "environment.json").read_text(encoding="utf-8"))
        assert metadata["python_executable"] == spec.python_executable
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_environment_runtime_python_uses_bootstrap_until_dependency_install_verified() -> None:
    root = _scratch_dir("env-runtime-python-bootstrap-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-runtime-python-bootstrap",
                sandbox_profile=SandboxProfileSpec(required_imports=["requests"]),
                files=[ToolboxBundleFile(relative_path="demo_tools.py", content="def hello(name='x'):\n    return {'name': name}\n")],
                tools=[ToolboxBundleTool(definition=_tool_definition("hello_tool"), entrypoint="demo_tools:hello")],
            )
        )
        manager = ToolboxEnvironmentManager(root)
        spec = manager.ensure_for_bundle(staged)
        manager.realize_environment(
            spec,
            environment_description={
                "name": "base",
                "extra_packages": ["requests"],
                "effective_extra_packages": ["requests"],
                "allow_online_install": False,
                "effective_allow_online_install": False,
            },
            required_packages=["requests"],
        )

        chosen = manager.runtime_python_executable(spec, bootstrap_python_executable="python-bootstrap")

        assert chosen == "python-bootstrap"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_environment_runtime_python_uses_venv_after_verified_install() -> None:
    root = _scratch_dir("env-runtime-python-verified-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-runtime-python-verified",
                files=[ToolboxBundleFile(relative_path="demo_tools.py", content="def hello(name='x'):\n    return {'name': name}\n")],
                tools=[ToolboxBundleTool(definition=_tool_definition("hello_tool"), entrypoint="demo_tools:hello")],
            )
        )
        manager = ToolboxEnvironmentManager(root)
        spec = manager.ensure_for_bundle(staged)
        env_root = Path(spec.venv_path)
        metadata = json.loads((env_root / "environment.json").read_text(encoding="utf-8"))
        metadata["install_execution"] = {"status": "ok"}
        metadata["install_receipt_verification"] = {"status": "ok"}
        (env_root / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        chosen = manager.runtime_python_executable(spec, bootstrap_python_executable="python-bootstrap")

        assert Path(chosen).resolve() == Path(spec.python_executable).resolve()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_environment_manager_realizes_workflow_python_helper_environment() -> None:
    root = _scratch_dir("env-workflow-python-helper-")
    try:
        manager = ToolboxEnvironmentManager(root)
        metadata = manager.realize_workflow_python_helper_environment(
            policy={
                "sandbox_kind": "workflow_python_helper",
                "import_allowlist": ["json"],
                "package_pins": {},
            },
            package_id="pkg-demo",
            workflow_id="config/demo",
            package_source_digest="digest-demo",
            helper_source_sha256="a" * 64,
            helper_source_path="dynamic/helpers/demo.py",
            bootstrap_python_executable="python-bootstrap",
        )

        env_path = Path(str(metadata.get("venv_path") or "")).expanduser().resolve()
        assert (env_path / "pyvenv.cfg").exists()
        assert "runtime_envs" in str(env_path).replace("\\", "/")
        assert metadata["toolbox_runtime_hash"] == "workflow-python-helper-v1"
        assert metadata["intrinsics_profile_id"] == "workflow_python_helper"
        assert metadata["required_imports"] == ["json"]
        assert Path(metadata["runtime_python_executable"]).resolve() == env_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        assert metadata["runtime_python_source"] == "venv"
        assert metadata["runtime_python_selection"]["mode"] == "venv"
        assert metadata["workflow_python_helper"]["package_id"] == "pkg-demo"
        assert metadata["workflow_python_helper"]["workflow_id"] == "config/demo"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_environment_manager_resolves_environment_inheritance() -> None:
    resolved = ToolboxEnvironmentManager.resolve_environment_description(
        {
            "base": {
                "name": "base",
                "base_env_name": None,
                "extra_packages": ["numpy"],
                "allow_online_install": False,
            },
            "math-env": {
                "name": "math-env",
                "base_env_name": "base",
                "extra_packages": ["sympy", "numpy"],
                "allow_online_install": True,
            },
        },
        name="math-env",
    )

    assert resolved["name"] == "math-env"
    assert resolved["base_env_name"] == "base"
    assert resolved["extra_packages"] == ["sympy", "numpy"]
    assert resolved["effective_extra_packages"] == ["sympy", "numpy"]
    assert resolved["effective_allow_online_install"] is True
    assert resolved["lineage"] == ["math-env", "base"]


def test_toolbox_sandbox_orchestrator_groups_requests_by_profile() -> None:
    root = _scratch_dir("orchestrator-group-")
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        orchestrator = ToolboxSandboxOrchestrator(
            service=svc,
            stager=ToolboxBundleStager(root),
            python_executable=sys.executable,
        )
        requests = [
            ToolboxAutoAssignmentRequest(
                files=[ToolboxBundleFile(relative_path="alpha.py", content="def alpha_tool():\n    return {'tool': 'alpha'}\n")],
                module_name="alpha",
                callable_name="alpha_tool",
                sandbox_profile=SandboxProfileSpec(required_imports=["requests"], sandbox_policy={"sandbox": {"enabled": True}}),
            ),
            ToolboxAutoAssignmentRequest(
                files=[ToolboxBundleFile(relative_path="beta.py", content="def beta_tool():\n    return {'tool': 'beta'}\n")],
                module_name="beta",
                callable_name="beta_tool",
                sandbox_profile=SandboxProfileSpec(required_imports=["requests"], sandbox_policy={"sandbox": {"enabled": True}}),
            ),
            ToolboxAutoAssignmentRequest(
                files=[ToolboxBundleFile(relative_path="gamma.py", content="def gamma_tool():\n    return {'tool': 'gamma'}\n")],
                module_name="gamma",
                callable_name="gamma_tool",
                sandbox_profile=SandboxProfileSpec(profile_id="isolated", required_imports=["sympy"], sandbox_policy={"sandbox": {"enabled": True}}),
            ),
        ]

        assignments = orchestrator.build_assignments(toolbox_id="toolbox-auto-assign", requests=requests)

        assert len(assignments) == 2
        grouped = {item.sandbox_profile.normalized_profile_id(): item for item in assignments}
        auto_profile_id = [pid for pid in grouped.keys() if pid != "isolated"][0]
        assert sorted([item.callable_name for item in grouped[auto_profile_id].bundle_spec.auto_tools]) == ["alpha_tool", "beta_tool"]
        assert [item.callable_name for item in grouped["isolated"].bundle_spec.auto_tools] == ["gamma_tool"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_runtime_base_adds_shared_environment_identity() -> None:
    root = _scratch_dir("toolbox-runtime-base-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="toolbox-runtime-default",
                toolbox_id="toolbox-runtime",
                dependency_lock_hash="lock-a",
                files=[ToolboxBundleFile(relative_path="demo.py", content="def demo():\n    return 1\n")],
                auto_tools=[ToolboxBundleAutoTool(module_name="demo", callable_name="demo")],
            )
        )
        env_spec = ToolboxEnvironmentManager(root).environment_spec_for_bundle(staged)

        out = HostedToolboxRuntimeBase().registration_environment(
            environment=staged.registration_environment(env_spec),
            toolbox_id="toolbox-runtime",
            sandbox_profile_id="default",
            bundle_revision=str(staged.manifest.get("bundle_revision") or ""),
            sandbox_policy={"sandbox": {"enabled": True}},
        )

        assert out["environment_key"]
        assert out["environment_key_full"]
        assert out["environment_root_kind"] == "toolbox_venvs"
        assert out["environment_consumer_kind"] == "toolbox_executor"
        assert out["environment_identity"]["runtime"]["runtime_kind"] == "toolbox_executor"
        assert out["environment_identity"]["runtime"]["profile"] == "default"
        assert out["environment_identity"]["dependency_lock_hash"] == "lock-a"
    finally:
        shutil.rmtree(root, ignore_errors=True)




def test_toolbox_execute_records_shared_hosted_pool_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _scratch_dir("toolbox-hosted-pool-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    reg = svc.spawn(
        engine_id="toolbox-hosted-pool",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        worker_profile_class="generic",
        executor_kind="toolbox_executor",
        bundle={"toolbox_id": "toolbox-hosted", "sandbox_profile_id": "default"},
        environment={"environment_key": "toolbox-env-key"},
        tool_access={"allowed_tool_names": ["demo_tool"], "advertised_tool_names": ["demo_tool"]},
        capabilities={"capacity": 2},
    )

    def fake_ipc_call(**_kwargs):
        return {"status": "ok", "tool_call": {"id": "call-demo-1", "result": "demo-ok"}}

    monkeypatch.setattr(svc, "_ipc_call", fake_ipc_call)
    try:
        out = svc.toolbox_execute(
            engine_id="toolbox-hosted-pool",
            execution_request_id="exec-call-demo-1",
            tool_call={"id": "call-demo-1", "name": "demo_tool", "arguments": {}},
            timeout_seconds=2.0,
        )
        status = svc._test_toolbox_status(
            environment_key="toolbox-env-key",
            request_id=str(out["request_id"]),
        )
    finally:
        svc.shutdown("toolbox-hosted-pool", timeout_seconds=2.0)
        svc.remove_registration("toolbox-hosted-pool")
        shutil.rmtree(root, ignore_errors=True)

    assert int(reg.get("pid") or 0)
    assert out["status"] == "ok"
    assert out["environment_key"] == "toolbox-env-key"
    assert out["hosted_pool"]["metrics"]["desired_capacity"] == 2
    assert out["hosted_pool"]["metrics"]["recent_requests"][-1]["request_id"] == out["request_id"]
    assert out["request_id"] != out["tool_call_id"]
    assert out["hosted_pool"]["metrics"]["recent_requests"][-1]["operation_id"] == "demo_tool"
    assert out["hosted_pool"]["metrics"]["recent_requests"][-1]["status"] == "ok"
    assert status["status"] == "ok"
    assert status["request"]["status"] == "ok"
    assert status["source"] in {"active", "recent"}


def test_toolbox_execute_returns_all_settled_error_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _scratch_dir("toolbox-all-settled-error-")
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        svc.register_spawned(
            engine_id="toolbox-all-settled-error",
            pid=1234,
            command=[sys.executable, "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "missing.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-all-settled-error",
            executor_kind="toolbox_executor",
            bundle={"toolbox_id": "toolbox-all-settled", "sandbox_profile_id": "default"},
            environment={"environment_key": "toolbox-all-settled-env"},
            tool_access={"allowed_tool_names": ["demo_tool"]},
        )

        def fail_ipc(**_kwargs):
            raise RuntimeError("worker failed after admission")

        monkeypatch.setattr(svc, "_ipc_call", fail_ipc)
        out = svc.toolbox_execute(
            engine_id="toolbox-all-settled-error",
            execution_request_id="exec-call-all-settled-error",
            tool_call={"id": "call-all-settled-error", "name": "demo_tool", "arguments": {}},
        )

        assert out["status"] == "error"
        assert out["reason"] == "worker failed after admission"
        assert out["tool_call_id"] == "call-all-settled-error"
        assert out["request"]["request_id"] == out["request_id"]
        assert out["request_id"] != out["tool_call_id"]
        assert out["request"]["status"] == "error"
        assert out["diagnostics"]["request"]["reason"] == "worker failed after admission"
        assert out["diagnostics"]["pool"]["metrics"]["recent_requests"][-1]["status"] == "error"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_cancel_marks_recycled_sibling_requests_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _scratch_dir("toolbox-sandbox-recycled-")
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        svc.register_spawned(
            engine_id="toolbox-sandbox-recycled",
            pid=1234,
            command=[sys.executable, "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "missing.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-sandbox-recycled",
            executor_kind="toolbox_executor",
            bundle={"toolbox_id": "toolbox-recycled", "sandbox_profile_id": "default"},
            environment={"environment_key": "toolbox-recycled-env"},
            tool_access={"allowed_tool_names": ["demo_tool"]},
            capabilities={"capacity": 2},
        )
        base = svc._toolbox_runtime_base()
        for request_id in ["call-recycled-target", "call-recycled-sibling"]:
            base.submit_request(
                environment_key="toolbox-recycled-env",
                request_id=request_id,
                profile="default",
                factory=lambda _key, cap: HostedToolboxRuntimeBase.worker_slot(
                    engine_id="toolbox-sandbox-recycled",
                    environment_key="toolbox-recycled-env",
                    capacity=cap,
                    status="registered",
                ),
                desired_capacity=2,
                operation_id="demo_tool",
            )
        monkeypatch.setattr(svc, "shutdown", lambda *_args, **_kwargs: {"status": "ok", "alive": False})
        status = svc._test_cancel_toolbox(
            engine_id="toolbox-sandbox-recycled",
            tool_name="demo_tool",
            tool_call_id="call-recycled-target",
            request_id="call-recycled-target",
            respawn=False,
        )
        out = dict(status["result"])

        sibling = base.request_status(environment_key="toolbox-recycled-env", request_id="call-recycled-sibling")
        assert out["sandbox_recycled_request_ids"]["toolbox-sandbox-recycled"] == ["call-recycled-sibling"]
        assert sibling["request"]["status"] == "error"
        assert sibling["request"]["reason"] == "sandbox_recycled"
    finally:
        shutil.rmtree(root, ignore_errors=True)




def test_toolbox_execute_forwards_host_api_approval_to_worker_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _scratch_dir("toolbox-host-api-approval-forward-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    reg = svc.spawn(
        engine_id="toolbox-host-api-approval-forward",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        worker_profile_class="generic",
        executor_kind="toolbox_executor",
        bundle={"toolbox_id": "toolbox-forward", "sandbox_profile_id": "default"},
        environment={"environment_key": "toolbox-env-key"},
        tool_access={"allowed_tool_names": ["demo_tool"], "advertised_tool_names": ["demo_tool"]},
    )
    seen: list[dict] = []

    def fake_ipc_call(**kwargs):
        seen.append(dict(kwargs))
        return {"status": "ok", "tool_call": {"id": "call-demo-approval", "name": "demo_tool", "result": "{}"}}

    monkeypatch.setattr(svc, "_ipc_call", fake_ipc_call)
    try:
        svc.toolbox_execute(
            engine_id="toolbox-host-api-approval-forward",
            execution_request_id="exec-call-demo-approval",
            tool_call={"id": "call-demo-approval", "name": "demo_tool", "arguments": {}},
            host_api_approval={"mode": "always"},
        )
    finally:
        svc.shutdown("toolbox-host-api-approval-forward", timeout_seconds=2.0)
        svc.remove_registration("toolbox-host-api-approval-forward")
        shutil.rmtree(root, ignore_errors=True)

    params = dict(dict(seen[0]["payload"])["params"])
    assert params["host_api_approval"] == {"mode": "always"}


def test_toolbox_cancel_marks_shared_hosted_pool_request_canceled() -> None:
    root = _scratch_dir("toolbox-hosted-cancel-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    svc.spawn(
        engine_id="toolbox-hosted-cancel",
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        worker_profile_class="generic",
        executor_kind="toolbox_executor",
        bundle={"toolbox_id": "toolbox-hosted-cancel", "sandbox_profile_id": "default"},
        environment={"environment_key": "toolbox-cancel-env"},
        tool_access={"allowed_tool_names": ["demo_tool"], "advertised_tool_names": ["demo_tool"]},
        capabilities={"capacity": 2},
    )
    base = svc._toolbox_runtime_base()
    base.submit_request(
        environment_key="toolbox-cancel-env",
        request_id="call-cancel-1",
        profile="default",
        factory=lambda _key, cap: HostedToolboxRuntimeBase.worker_slot(
            engine_id="toolbox-hosted-cancel",
            environment_key="toolbox-cancel-env",
            capacity=cap,
            status="registered",
        ),
        desired_capacity=2,
        operation_id="demo_tool",
    )

    try:
        terminal = svc._test_cancel_toolbox(
            engine_id="toolbox-hosted-cancel",
            tool_name="demo_tool",
            tool_call_id="call-cancel-1",
            request_id="call-cancel-1",
            timeout_seconds=2.0,
            respawn=False,
        )
        out = dict(terminal["result"])
        status = base.request_status(environment_key="toolbox-cancel-env", request_id="call-cancel-1")
    finally:
        svc.shutdown("toolbox-hosted-cancel", timeout_seconds=2.0)
        svc.remove_registration("toolbox-hosted-cancel")
        shutil.rmtree(root, ignore_errors=True)

    assert out["outcome"] == "canceled"
    assert out["hosted_pool_cancels"]["toolbox-hosted-cancel"]["status"] == "ok"
    assert status["status"] == "ok"
    assert status["request"]["status"] == "canceled"
    assert status["source"] in {"active", "recent"}


def test_load_toolbox_from_manifest_supports_auto_callable_discovery() -> None:
    root = _scratch_dir("load-auto-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-load-auto",
                files=[
                    ToolboxBundleFile(
                        relative_path="auto_tools.py",
                        content=(
                            "def hello_auto(name: str = 'world'):\n"
                            "    \"\"\"Return a greeting.\n\n"
                            "    Args:\n"
                            "        name (str): Name to greet.\n"
                            "    \"\"\"\n"
                            "    return {'greeting': f'hi {name}'}\n"
                        ),
                    )
                ],
                auto_tools=[ToolboxBundleAutoTool(module_name="auto_tools", callable_name="hello_auto")],
            )
        )
        toolbox, manifest = load_toolbox_from_manifest(staged.manifest_path)

        assert manifest["auto_tools"][0]["name"] == "hello_auto"
        assert toolbox.get_tool("hello_auto")["function"]["parameters"]["properties"]["name"]["type"] == "string"
        call = ToolCall(name="hello_auto", arguments={"name": "Sam"})
        out = asyncio.run(toolbox.execute(call))
        assert "hi Sam" in str(out or "")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_bundle_startup_spec_env_writes_spec_file() -> None:
    root = _scratch_dir("startup-spec-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-startup",
                files=[ToolboxBundleFile(relative_path="demo_tools.py", content="def hello(name='x'):\n    return {'name': name}\n")],
                tools=[ToolboxBundleTool(definition=_tool_definition("hello_tool"), entrypoint="demo_tools:hello")],
            )
        )
        env = staged.worker_env_with_startup_spec(
            worker_id="toolbox-startup-1",
            sandbox_id="sandbox-startup",
            scratch_root=root / "scratch-runtime",
        )
        spec_path = Path(str(env["MP13_TOOLBOX_WORKER_SPEC_PATH"])).resolve()
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        assert payload["worker_id"] == "toolbox-startup-1"
        assert payload["sandbox_id"] == "sandbox-startup"
        assert payload["manifest_path"] == str(staged.manifest_path)
        assert payload["scratch_root"] == str((root / "scratch-runtime").resolve())
        assert payload["ipc_family"] == ("AF_PIPE" if os.name == "nt" else "AF_UNIX")
        assert env["MP13_TOOLBOX_MANIFEST_PATH"] == str(staged.manifest_path)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_worker_startup_spec_defaults_platform_ipc_family() -> None:
    spec = ToolboxWorkerStartupSpec(
        worker_id="toolbox-startup-default",
        sandbox_id="sandbox-default",
        toolbox_revision="rev-a",
        manifest_path="manifest.json",
        scratch_root="scratch",
    )
    payload = spec.to_dict()
    restored = ToolboxWorkerStartupSpec.from_dict(
        {
            "worker_id": "toolbox-startup-default",
            "sandbox_id": "sandbox-default",
            "toolbox_revision": "rev-a",
            "manifest_path": "manifest.json",
            "scratch_root": "scratch",
        }
    )
    expected = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
    assert spec.ipc_family == expected
    assert payload["ipc_family"] == expected
    assert restored.ipc_family == expected


def test_engine_host_service_allocate_ipc_address_uses_unix_socket_on_posix(monkeypatch) -> None:
    monkeypatch.setattr("hosting.service.proxy.os.name", "posix")
    monkeypatch.setattr("hosting.service.proxy.tempfile.gettempdir", lambda: "/tmp")

    family, address = EngineHostService._allocate_ipc_address("toolbox/linux demo")

    assert family == "AF_UNIX"
    assert str(address).endswith(".sock")
    assert "toolbox_linux_demo" in str(address)
    assert len(str(address).encode("utf-8")) < 108


def test_engine_host_service_allocate_ipc_address_bounds_unix_socket_length(monkeypatch) -> None:
    monkeypatch.setattr("hosting.service.proxy.os.name", "posix")
    monkeypatch.setattr("hosting.service.proxy.tempfile.gettempdir", lambda: "/tmp")

    family, address = EngineHostService._allocate_ipc_address("toolbox-" + ("very-long-name-" * 12))

    assert family == "AF_UNIX"
    assert str(address).endswith(".sock")
    assert len(str(address).encode("utf-8")) < 108


def test_engine_host_service_allocate_ipc_address_uses_named_pipe_on_windows(monkeypatch) -> None:
    monkeypatch.setattr("hosting.service.proxy.os.name", "nt")

    family, address = EngineHostService._allocate_ipc_address("toolbox/win demo")

    assert family == "AF_PIPE"
    assert str(address).startswith(r"\\.\pipe\mp13-host-toolbox_win_demo-")


def test_toolbox_execute_denies_unknown_tool_before_worker_call(monkeypatch) -> None:
    root = _scratch_dir("service-")
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        svc.register_spawned(
            engine_id="toolbox1",
            pid=1234,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "missing.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-missing-toolbox",
            executor_kind="toolbox_executor",
            tool_access={"allowed_tool_names": ["hello_tool"]},
        )

        called = {"count": 0}

        def _fake_ipc_call(*args, **kwargs):
            called["count"] += 1
            return {"status": "ok"}

        monkeypatch.setattr(svc, "_ipc_call", _fake_ipc_call)

        with pytest.raises(PermissionError, match="tool_not_allowed:blocked_tool"):
            svc.toolbox_execute(
                engine_id="toolbox1",
                tool_call={"name": "blocked_tool", "arguments": {}},
            )
        assert called["count"] == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_execute_denies_blocked_in_scope_before_worker_call(monkeypatch) -> None:
    root = _scratch_dir("service-scope-")
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        svc.register_spawned(
            engine_id="toolbox1",
            pid=1234,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "missing.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-missing-toolbox",
            executor_kind="toolbox_executor",
            tool_access={"allowed_tool_names": ["hello_tool"]},
        )

        called = {"count": 0}

        def _fake_ipc_call(*args, **kwargs):
            called["count"] += 1
            return {"status": "ok"}

        monkeypatch.setattr(svc, "_ipc_call", _fake_ipc_call)

        with pytest.raises(PermissionError, match="blocked_in_scope:hello_tool"):
            svc.toolbox_execute(
                engine_id="toolbox1",
                tool_call={"name": "hello_tool", "arguments": {}},
                tools_view={
                    "view_id": "turn-1",
                    "mode": "advertised",
                    "allowed_tools": [],
                    "advertised_tools": [],
                    "hidden_allowed_tools": [],
                    "disabled_tools": ["hello_tool"],
                    "gated_tools": [],
                },
            )
        assert called["count"] == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_execute_denies_gated_requires_confirmation_before_worker_call(monkeypatch) -> None:
    root = _scratch_dir("service-gated-exec-")
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        svc.register_spawned(
            engine_id="toolbox1",
            pid=1234,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "missing.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-missing-toolbox",
            executor_kind="toolbox_executor",
            tool_access={"allowed_tool_names": ["hello_tool"]},
        )

        called = {"count": 0}

        def _fake_ipc_call(*args, **kwargs):
            called["count"] += 1
            return {"status": "ok"}

        monkeypatch.setattr(svc, "_ipc_call", _fake_ipc_call)

        with pytest.raises(PermissionError, match="gated_requires_confirmation:hello_tool"):
            svc.toolbox_execute(
                engine_id="toolbox1",
                tool_call={"name": "hello_tool", "arguments": {}},
                tools_view={
                    "view_id": "turn-1",
                    "mode": "advertised",
                    "allowed_tools": [],
                    "advertised_tools": ["hello_tool"],
                    "hidden_allowed_tools": [],
                    "disabled_tools": [],
                    "gated_tools": ["hello_tool"],
                },
            )
        assert called["count"] == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)




def test_toolbox_cancel_returns_noop_when_target_is_missing() -> None:
    root = _scratch_dir("service-cancel-missing-")
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )

        terminal = svc._test_cancel_toolbox(toolbox_id="missing-box", request_id="cancel-missing")
        out = dict(terminal["result"])

        assert terminal["lifecycle"] == "terminal_failure"
        assert out["outcome"] == "cancel_failed"
        assert out["reason"] == "toolbox_executor_missing"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_assignment_requests_round_trip_non_restartable_flag() -> None:
    auto_request = ToolboxAutoAssignmentRequest.from_runtime_dict(
        {
            "files": [{"relative_path": "demo.py", "content": "def alpha():\n    return 'x'\n"}],
            "module_name": "demo",
            "callable_name": "alpha",
            "sandbox_profile": {"profile_id": "default", "environment_name": "base", "required_imports": [], "sandbox_policy": {}},
            "activate": True,
            "hidden": False,
            "non_restartable": True,
        }
    )
    manual_request = ToolboxManualAssignmentRequest.from_runtime_dict(
        {
            "files": [{"relative_path": "demo_manual.py", "content": "def beta():\n    return 'y'\n"}],
            "module_name": "demo_manual",
            "callable_name": "beta",
            "tool_definition": _tool_definition("beta"),
            "sandbox_profile": {"profile_id": "default", "environment_name": "base", "required_imports": [], "sandbox_policy": {}},
            "hidden": False,
            "non_restartable": True,
        }
    )

    assert auto_request.non_restartable is True
    assert auto_request.to_runtime_dict()["non_restartable"] is True
    assert auto_request.to_auto_tool().to_dict()["non_restartable"] is True
    assert manual_request.non_restartable is True
    assert manual_request.to_runtime_dict()["non_restartable"] is True
    assert manual_request.to_bundle_tool().to_dict()["non_restartable"] is True


def test_toolbox_gate_reports_denied_and_allowed_outcomes() -> None:
    root = Path(".tmp_service_gate_test").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        svc.register_spawned(
            engine_id="toolbox1",
            pid=1234,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "missing.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-missing-toolbox",
            executor_kind="toolbox_executor",
            bundle={"toolbox_id": "demo-box"},
            tool_access={"allowed_tool_names": ["hello_tool"]},
        )

        denied = svc.toolbox_gate(toolbox_id="demo-box", tool_name="blocked_tool")
        allowed = svc.toolbox_gate(toolbox_id="demo-box", tool_name="hello_tool")

        assert denied["outcome"] == "denied"
        assert denied["reason"] == "tool_not_allowed"
        assert denied["executable"] is False
        assert allowed["outcome"] == "allowed"
        assert allowed["reason"] == "allowed"
        assert allowed["executable"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_gate_respects_request_scoped_tools_view() -> None:
    root = _scratch_dir("service-gate-view-")
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        svc.register_spawned(
            engine_id="toolbox1",
            pid=1234,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "missing.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-missing-toolbox",
            executor_kind="toolbox_executor",
            bundle={"toolbox_id": "demo-box"},
            tool_access={
                "allowed_tool_names": ["hello_tool", "hidden_tool"],
                "advertised_tool_names": ["hello_tool"],
                "hidden_allowed_tool_names": ["hidden_tool"],
            },
        )

        denied = svc.toolbox_gate(
            toolbox_id="demo-box",
            tool_name="hello_tool",
            tools_view={
                "view_id": "turn-1",
                "mode": "advertised",
                "allowed_tools": ["hidden_tool"],
                "advertised_tools": [],
                "hidden_allowed_tools": ["hidden_tool"],
                "disabled_tools": ["hello_tool"],
                "gated_tools": [],
            },
        )

        assert denied["outcome"] == "denied"
        assert denied["reason"] == "blocked_in_scope"
        assert denied["executable"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_gate_reports_gated_requires_confirmation_from_request_view() -> None:
    root = _scratch_dir("service-gate-gated-")
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        svc.register_spawned(
            engine_id="toolbox1",
            pid=1234,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "missing.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-missing-toolbox",
            executor_kind="toolbox_executor",
            bundle={"toolbox_id": "demo-box"},
            tool_access={"allowed_tool_names": ["hello_tool"]},
        )

        gated = svc.toolbox_gate(
            toolbox_id="demo-box",
            tool_name="hello_tool",
            tools_view={
                "view_id": "turn-1",
                "mode": "advertised",
                "allowed_tools": [],
                "advertised_tools": ["hello_tool"],
                "hidden_allowed_tools": [],
                "disabled_tools": [],
                "gated_tools": ["hello_tool"],
            },
        )

        assert gated["outcome"] == "gated_requires_confirmation"
        assert gated["reason"] == "gated_requires_confirmation"
        assert gated["executable"] is False
        assert gated["requires_confirmation"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)




def test_native_toolbox_harness_executes_calls_in_parallel() -> None:
    state = {"active": 0, "max_active": 0}
    toolbox = Toolbox()

    async def sleeper(name: str = "", delay: float = 0.05) -> dict:
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        try:
            await asyncio.sleep(delay)
            return {"name": name}
        finally:
            state["active"] -= 1

    ok, msg = toolbox.add_tool_external(_tool_definition("sleep_tool"), sleeper, activate=True)
    assert ok, msg
    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="native"),
        native_toolbox=toolbox,
    )

    calls = [
        ToolCall(name="sleep_tool", arguments={"name": "a", "delay": 0.05}),
        ToolCall(name="sleep_tool", arguments={"name": "b", "delay": 0.05}),
        ToolCall(name="sleep_tool", arguments={"name": "c", "delay": 0.05}),
    ]
    out = asyncio.run(harness.execute_calls(calls, parallel=True, max_concurrency=2))

    assert state["max_active"] == 2
    assert [json.loads(str(item.result or "{}"))["name"] for item in out] == ["a", "b", "c"]


def test_sandbox_harness_round_robins_pool_members() -> None:
    class _FakeChannel:
        def __init__(self) -> None:
            self.engine_ids: list[str] = []

        def toolbox_execute(
            self,
            *,
            engine_id: str,
            tool_call: dict,
            timeout_seconds: float = 30.0,
            tools_view: Optional[Dict[str, Any]] = None,
            callback_binding: Optional[Dict[str, Any]] = None,
        ) -> dict:
            self.engine_ids.append(engine_id)
            row = dict(tool_call or {})
            row["result"] = json.dumps({"engine_id": engine_id})
            return {"status": "ok", "tool_call": row}

    channel = _FakeChannel()
    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_engine_ids=["tbx-a", "tbx-b"]),
        control_channel=channel,
    )

    calls = [
        ToolCall(name="hello_tool", arguments={"name": "x"}),
        ToolCall(name="hello_tool", arguments={"name": "y"}),
        ToolCall(name="hello_tool", arguments={"name": "z"}),
    ]
    out = asyncio.run(harness.execute_calls(calls, parallel=False))

    assert channel.engine_ids == ["tbx-a", "tbx-b", "tbx-a"]
    assert json.loads(str(out[1].result or "{}"))["engine_id"] == "tbx-b"


def test_sandbox_harness_forwards_host_api_approval() -> None:
    class _FakeChannel:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def toolbox_gate(self, **_kwargs):
            return {"outcome": "allowed", "reason": "allowed"}

        def toolbox_execute(self, **kwargs):
            self.calls.append(dict(kwargs))
            row = dict(kwargs["tool_call"])
            row["result"] = "{}"
            return {"status": "ok", "tool_call": row}

    channel = _FakeChannel()
    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_engine_ids=["tbx-a"]),
        control_channel=channel,
    )

    out = asyncio.run(
        harness.execute_calls(
            [ToolCall(name="hello_tool", arguments={})],
            parallel=False,
            host_api_approval={"mode": "always"},
        )
    )

    assert out[0].error is None
    assert channel.calls[0]["host_api_approval"] == {"mode": "always"}


def test_hosted_toolbox_ref_execute_forwards_host_api_approval() -> None:
    class _FakeHost:
        def __init__(self) -> None:
            self.execute_calls: list[dict] = []

        def toolbox_gate(self, **_kwargs):
            return {"outcome": "allowed", "reason": "allowed"}

        def toolbox_execute(self, **kwargs):
            self.execute_calls.append(dict(kwargs))
            return {"status": "ok", "tool_call": dict(kwargs["tool_call"])}

    host = _FakeHost()
    ref = HostedToolBoxRef(toolbox_id="toolbox-ref-approval", host=host)

    out = ref.execute(
        tool_name="hello_tool",
        execution_request_id="exec-hello-tool-approval",
        arguments={"name": "Sam"},
        host_api_approval={"mode": "always"},
    )

    assert out["status"] == "ok"
    assert host.execute_calls[0]["host_api_approval"] == {"mode": "always"}


def test_sandbox_harness_normalizes_missing_executor_into_canceled_tool_error() -> None:
    class _FakeChannel:
        def toolbox_execute(self, **kwargs):
            raise RuntimeError("toolbox_executor_missing:toolbox-demo")

    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_toolbox_id="toolbox-demo"),
        control_channel=_FakeChannel(),
    )

    call = ToolCall(name="hello_tool", arguments={})
    out = asyncio.run(harness.execute_calls([call], parallel=False))

    assert len(out) == 1
    assert out[0].result is None
    assert out[0].error == "Execution canceled: sandbox_recycled:hello_tool"
    assert is_canceled_tool_error(out[0]) is True
    assert should_resubmit_canceled_tool_call(out[0], non_restartable=False) is True
    assert should_resubmit_canceled_tool_call(out[0], non_restartable=True) is False


def test_sandbox_harness_passes_queue_full_as_per_call_error_with_envelope() -> None:
    class _FakeChannel:
        def toolbox_execute(self, **_kwargs):
            return {
                "status": "error",
                "outcome": "error",
                "reason": "queue_full",
                "request_id": "exec-queue-full",
                "diagnostics": {"pool": {"metrics": {"queue_depth": 0}}},
            }

    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_toolbox_id="toolbox-queue-full"),
        control_channel=_FakeChannel(),
    )

    out = asyncio.run(harness.execute_calls([ToolCall(id="model-1", name="hello_tool")], parallel=True))

    assert len(out) == 1
    assert out[0].error == "Execution failed: queue_full"
    assert out[0].execution_envelope["reason"] == "queue_full"
    assert out[0].execution_envelope["diagnostics"]["pool"]["metrics"]["queue_depth"] == 0


def test_sandbox_harness_all_settles_transport_failure_with_successful_sibling() -> None:
    class _FakeChannel:
        def toolbox_execute(self, **kwargs):
            tool_call = dict(kwargs.get("tool_call") or {})
            if tool_call.get("name") == "fail_tool":
                raise ConnectionError("connection reset by peer")
            return {"status": "ok", "tool_call": {**tool_call, "result": "ok"}}

    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_engine_ids=["toolbox-a"]),
        control_channel=_FakeChannel(),
    )

    out = asyncio.run(
        harness.execute_calls(
            [ToolCall(id="fail", name="fail_tool"), ToolCall(id="ok", name="ok_tool")],
            parallel=True,
            max_concurrency=2,
        )
    )

    assert len(out) == 2
    assert out[0].error
    assert out[0].execution_envelope["reason"] in {"transport_error", "sandbox_recycled"}
    assert out[1].error is None
    assert out[1].result == "ok"


def test_sandbox_harness_separates_duplicate_model_ids_from_execution_request_ids() -> None:
    execution_ids: list[str] = []

    class _FakeChannel:
        def toolbox_execute(self, **kwargs):
            execution_ids.append(str(kwargs.get("execution_request_id") or ""))
            tool_call = dict(kwargs.get("tool_call") or {})
            return {"status": "ok", "tool_call": {**tool_call, "result": "ok"}}

    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_engine_ids=["toolbox-a"]),
        control_channel=_FakeChannel(),
    )

    out = asyncio.run(
        harness.execute_calls(
            [ToolCall(id="duplicate", name="first"), ToolCall(id="duplicate", name="second")],
            parallel=True,
            max_concurrency=2,
        )
    )

    assert len(set(execution_ids)) == 2
    assert len({str(item.execution_envelope["request_id"]) for item in out}) == 2
    assert [item.id for item in out] == ["duplicate", "duplicate"]


def test_sandbox_harness_describe_normalizes_parallel_execution_before_execution() -> None:
    class _FakeChannel:
        def toolbox_describe(self, **_kwargs):
            return {"status": "ok", "parallel_execution": {"supported": True, "effective_max_concurrency": 2}}

    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_engine_ids=["toolbox-a"]),
        control_channel=_FakeChannel(),
    )

    described = asyncio.run(harness.describe())
    parallel = described["parallel_execution"]

    assert parallel == {
        "supported": True,
        "async_within_executor": True,
        "sandbox_pool": False,
        "effective_max_concurrency": 2,
        "queue_policy": "bounded",
        "queue_depth": 0,
        "queue_timeout_seconds": 0.0,
        "active_calls": 0,
        "queued_calls": 0,
        "worker_process_count": 0,
        "execution_model": "threaded_worker",
    }






def test_hosted_toolbox_ref_execute_approval_add_to_scope_updates_scope_and_view() -> None:
    class _FakeHost:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Dict[str, Any]]] = []

        def toolbox_describe(self, **kwargs):
            self.calls.append(("toolbox_describe", dict(kwargs)))
            return {
                "status": "ok",
                "tool_metadata": {
                    "dangerous_auto": {
                        "callback_signature": None,
                        "non_restartable": False,
                        "hidden": False,
                    }
                },
            }

        def toolbox_gate(self, **kwargs):
            self.calls.append(("toolbox_gate", dict(kwargs)))
            view = dict(kwargs.get("tools_view") or {})
            gated = set(view.get("gated_tools") or [])
            allowed = set(view.get("allowed_tools") or [])
            name = str(kwargs.get("tool_name") or "").strip()
            if name in gated and name not in allowed:
                return {
                    "status": "ok",
                    "outcome": "gated_requires_confirmation",
                    "reason": "gated_requires_confirmation",
                    "tool_name": name,
                    "requires_confirmation": True,
                }
            return {"status": "ok", "outcome": "allowed", "reason": "allowed", "tool_name": name}

        def toolbox_execute(self, **kwargs):
            self.calls.append(("toolbox_execute", dict(kwargs)))
            tool_call = dict(kwargs.get("tool_call") or {})
            return {"status": "ok", "tool_call": tool_call | {"result": json.dumps({"ok": True})}}

    decisions: list[str] = []

    def _processor(*, callback_name: str, payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
        decisions.append(callback_name)
        return {"decision": "add_to_scope"}

    host = _FakeHost()
    ref = HostedToolBoxRef(toolbox_id="hosted-ref", host=host)
    scope_ref = ToolBoxRef(toolbox=Toolbox(), scope=ToolsScope(gated_tools={"dangerous_auto"}))
    view = ToolsView(
        view_id="turn-approval",
        mode="advertised",
        allowed_tools=set(),
        advertised_tools={"dangerous_auto"},
        hidden_allowed_tools=set(),
        disabled_tools=set(),
        gated_tools={"dangerous_auto"},
    )

    out = ref.execute(
        tool_name="dangerous_auto",
        execution_request_id="exec-dangerous-auto",
        arguments={"name": "Sam"},
        tools_view=view,
        callback_processor=_processor,
        callback_context={"toolbox_ref": scope_ref},
        tool_call_id="call-approval-1",
    )

    tool_row = dict(out.get("tool_call") or {})
    assert json.loads(str(tool_row.get("result") or "{}")) == {"ok": True}
    assert decisions == ["tool_requires_confirmation"]
    assert view.gated_tools == set()
    assert view.allowed_tools == {"dangerous_auto"}
    assert scope_ref.scope.gated_tools == set()
    execute_call = [payload for name, payload in host.calls if name == "toolbox_execute"][0]
    assert execute_call["tools_view"] == {
        "view_id": "turn-approval",
        "mode": "advertised",
        "allowed_tools": ["dangerous_auto"],
        "advertised_tools": ["dangerous_auto"],
        "hidden_allowed_tools": [],
        "disabled_tools": [],
        "gated_tools": [],
    }


def test_hosted_toolbox_ref_serializes_and_deserializes_with_control_channel() -> None:
    from hosting.engine_host_channel import EngineHostControlChannel

    channel = EngineHostControlChannel(
        {
            "engine_host_ssh_target": "user@example-host",
            "control_ssh_key": "C:/keys/id_ed25519",
            "engine_host_daemon_auto_bootstrap": False,
        }
    )
    ref = HostedToolBoxRef(
        toolbox_id="remote-ref",
        host=channel,
    )

    payload = ref.to_dict()
    restored = HostedToolBoxRef.from_dict(payload)

    assert payload["toolbox_id"] == "remote-ref"
    assert dict(payload["host"])["kind"] == "control_channel"
    assert isinstance(restored.host, EngineHostControlChannel)
    assert restored.toolbox_id == "remote-ref"
    assert set(payload) == {"toolbox_id", "host"}


def test_hosted_toolbox_ref_serializes_and_deserializes_with_service() -> None:
    root = Path(".tmp_hosted_ref_serde").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        ref = HostedToolBoxRef(
            toolbox_id="service-ref",
            host=svc,
        )

        payload = ref.to_dict()
        restored = HostedToolBoxRef.from_dict(payload)

        assert dict(payload["host"])["kind"] == "service"
        assert isinstance(restored.host, _EngineHostService)
        assert restored.toolbox_id == "service-ref"
        assert str(restored.host.engines_state_file).endswith("managed_engines.json")
    finally:
        shutil.rmtree(root, ignore_errors=True)




def test_toolbox_executor_context_fs_wrapper_uses_host_call(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    def _fake_invoke(method: str, arguments: dict, **_kwargs: Any) -> dict:
        calls.append((str(method), dict(arguments)))
        return {"text": "hello"}

    monkeypatch.setattr(toolbox_executor_ipc, "_invoke_host_call", _fake_invoke)
    ctx = toolbox_executor_ipc.ToolboxExecutionContext(
        engine_id="toolbox-a",
        toolbox_id="toolbox-demo",
        tool_name="peek_tool",
        tool_call_id="call-123",
        tool_arguments={"path": "a.txt"},
        callback_binding={"session_token": "tok", "address": "addr", "family": "AF_UNIX"},
    )

    out = ctx.fs.read_text(root_id="rw", relative_path="a.txt")
    described = ctx.host.describe()

    assert out == {"text": "hello"}
    described_methods = [row["name"] for row in described["host_capabilities"]["methods"]]
    assert "fs.read_text" in described_methods
    assert described["host_capabilities"]["providers"][0]["kind"] == "service_broker"
    assert calls[0][0] == "fs.read_text"
    payload = calls[0][1]
    assert payload["engine_id"] == "toolbox-a"
    assert payload["root_id"] == "rw"
    assert payload["relative_path"] == "a.txt"
    assert payload["encoding"] == "utf-8"
    callback_context = dict(payload["callback_context"])
    assert callback_context["engine_id"] == "toolbox-a"
    assert callback_context["toolbox_id"] == "toolbox-demo"
    assert callback_context["tool_name"] == "peek_tool"
    assert callback_context["tool_call_id"] == "call-123"
    assert callback_context["tool_arguments"] == {"path": "a.txt"}
    surface = dict(callback_context["callable_surface"])
    assert surface["contract"] == "hosting.toolbox.brokered_io.call_surface.v1"
    assert surface["method"] == "fs.read_text"
    assert surface["identity"]["provider_kind"] == "toolbox_session"
    assert surface["identity"]["provider_id"] == "toolbox-a"
    assert surface["identity"]["toolbox_id"] == "toolbox-demo"
    assert surface["identity"]["session_id"] == "call-123"
    assert surface["bridge_policy"]["namespaces"]["fs"] is True


def test_toolbox_executor_context_fs_wrapper_uses_host_capability_approval(monkeypatch) -> None:
    dispatches: list[dict] = []

    def _fake_dispatch(_binding: dict, *, callback_name: str, payload: Any, context: dict) -> dict:
        dispatches.append({"callback_name": callback_name, "payload": dict(payload or {}), "context": dict(context or {})})
        return {"result": {"status": "ok", "result": {"text": "approved"}}}

    monkeypatch.setattr(toolbox_executor_ipc, "_invoke_callback_binding", _fake_dispatch)

    ctx = toolbox_executor_ipc.ToolboxExecutionContext(
        engine_id="toolbox-approval",
        toolbox_id="toolbox-demo",
        tool_name="peek_tool",
        tool_call_id="call-approval",
        tool_arguments={"path": "a.txt"},
        callback_binding={"session_token": "tok", "address": "addr", "family": "AF_UNIX"},
        host_api_approval={"mode": "always"},
    )

    out = ctx.fs.read_text(root_id="rw", relative_path="a.txt")

    assert out == {"text": "approved"}
    assert dispatches[0]["callback_name"] == toolbox_executor_ipc.HOST_CAPABILITY_DISPATCH_CALLBACK_NAME
    assert dispatches[0]["payload"]["method"] == "fs.read_text"
    assert dispatches[0]["payload"]["approval"] == {"mode": "always"}
    assert dispatches[0]["payload"]["arguments"]["engine_id"] == "toolbox-approval"
    assert dispatches[0]["context"]["tool_call_id"] == "call-approval"


def test_toolbox_executor_context_fs_wrapper_denies_host_capability_approval(monkeypatch) -> None:
    def _fake_dispatch(_binding: dict, *, callback_name: str, payload: Any, context: dict) -> dict:
        return {"result": {"status": "error", "reason": "host_call_approval_denied", "message": "approval denied"}}

    monkeypatch.setattr(toolbox_executor_ipc, "_invoke_callback_binding", _fake_dispatch)

    ctx = toolbox_executor_ipc.ToolboxExecutionContext(
        engine_id="toolbox-deny",
        toolbox_id="toolbox-demo",
        tool_name="peek_tool",
        tool_call_id="call-deny",
        callback_binding={"session_token": "tok", "address": "addr", "family": "AF_UNIX"},
        host_api_approval={"mode": "always"},
    )

    with pytest.raises(Exception, match="approval denied"):
        ctx.fs.read_text(root_id="rw", relative_path="a.txt")


def test_toolbox_execute_dispatches_host_capability_in_parent_and_audits(monkeypatch) -> None:
    from hosting.callable_surface import HOST_CAPABILITY_APPROVAL_CALLBACK_NAME, HOST_CAPABILITY_DISPATCH_CALLBACK_NAME
    from hosting.toolbox.callbacks import _HostedToolCallbackRelay
    from hosting.toolbox_executor_ipc import _invoke_callback_binding

    root = _scratch_dir("toolbox-parent-host-api-")
    project_root = root / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "a.txt").write_text("parent-owned", encoding="utf-8")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    svc.register_spawned(
        engine_id="toolbox-parent-host-api",
        pid=os.getpid(),
        command=[sys.executable, "-m", "hosting.toolbox_executor_ipc"],
        executor_kind="toolbox_executor",
        sandbox_policy={
            "sandbox": {
                "enabled": True,
                "filesystem": {"rules": [{"root_id": "project", "path": str(project_root), "access": ["read"]}]},
                "brokered_io": {"filesystem": True, "http": False, "subprocess": False},
            }
        },
        bundle={"toolbox_id": "toolbox-parent-host-api"},
        tool_access={"allowed_tool_names": ["peek"]},
    )
    caller_relay = _HostedToolCallbackRelay()
    approvals: list[dict] = []
    try:
        caller_binding = caller_relay.bind_session(
            processor=lambda **kwargs: approvals.append(dict(kwargs.get("payload") or {})) or {"decision": "allow_once", "approved": True}
            if kwargs.get("callback_name") == HOST_CAPABILITY_APPROVAL_CALLBACK_NAME
            else {"status": "error", "message": "unexpected_callback"},
            toolbox_id="toolbox-parent-host-api",
            tool_name="peek",
            tool_call_id="call-parent-host-api",
            tool_arguments={},
        )

        def _fake_ipc_call(*, reg: dict, payload: dict, timeout_seconds: float) -> dict:
            params = dict(dict(payload or {}).get("params") or {})
            service_binding = dict(params.get("callback_binding") or {})
            response = _invoke_callback_binding(
                service_binding,
                callback_name=HOST_CAPABILITY_DISPATCH_CALLBACK_NAME,
                payload={
                    "method": "fs.read_text",
                    "arguments": {
                        "engine_id": "toolbox-parent-host-api",
                        "root_id": "project",
                        "relative_path": "a.txt",
                        "encoding": "utf-8",
                        "callback_context": {
                            "tool_call_id": "call-parent-host-api",
                            "toolbox_id": "toolbox-parent-host-api",
                            "tool_name": "peek",
                        },
                    },
                    "approval": {"mode": "always"},
                },
                context={"engine_id": "toolbox-parent-host-api", "tool_call_id": "call-parent-host-api"},
            )
            return {
                "status": "ok",
                "tool_call": {
                    **dict(params.get("tool_call") or {}),
                    "result": json.dumps(dict(response.get("result") or {}).get("result") or {}),
                },
            }

        monkeypatch.setattr(svc, "_ipc_call", _fake_ipc_call)

        out = svc.toolbox_execute(
            engine_id="toolbox-parent-host-api",
            execution_request_id="exec-call-parent-host-api",
            tool_call={"id": "call-parent-host-api", "name": "peek", "arguments": {}},
            callback_binding=caller_binding,
            host_api_approval={"mode": "always"},
        )

        tool_call = dict(out.get("tool_call") or {})
        assert json.loads(str(tool_call.get("result") or "{}"))["text"] == "parent-owned"
        assert approvals[0]["method"] == "fs.read_text"
        assert approvals[0]["argument_keys"] == ["callback_context", "encoding", "engine_id", "relative_path", "root_id"]
        assert approvals[0]["argument_preview"]["root_id"] == "project"
        assert approvals[0]["argument_preview"]["relative_path"] == "a.txt"
        assert "arguments" not in approvals[0]
        audit = svc.host_capability_audit_list(request_id="call-parent-host-api", method="fs.read_text")
        assert audit["total"] == 1
        assert audit["events"][0]["result"] == "approved"
        assert audit["events"][0]["argument_preview"]["relative_path"] == "a.txt"
        assert audit["events"][0]["provider"]["kind"] == "service_broker"
    finally:
        caller_relay.release_session(str(locals().get("caller_binding", {}).get("session_token") or ""))
        shutil.rmtree(root, ignore_errors=True)


def test_sandboxed_toolbox_facade_execute_does_not_serialize_callback_user_context() -> None:
    class _FakeHost:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def toolbox_execute(self, **kwargs):
            self.calls.append(("execute", dict(kwargs)))
            return {"status": "ok", "tool_call": {"name": kwargs["tool_call"]["name"], "result": "{}"}}

    host = _FakeHost()
    facade = SandboxedToolboxFacade(toolbox_id="facade-box", host=host)
    callback_context = {"origin": "chat", "lock": threading.Lock()}

    out = facade.execute(
        tool_name="hello_auto",
        execution_request_id="exec-callback-user-context",
        arguments={"name": "Sam"},
        callback_processor=lambda **kwargs: {"decision": "deny"},
        callback_context=callback_context,
    )

    assert dict(out.get("tool_call") or {}).get("name") == "hello_auto"
    assert len(host.calls) == 1
    payload = host.calls[0][1]
    binding = dict(payload.get("callback_binding") or {})
    assert binding["session_token"]
    assert binding["address"]
    assert binding["contract"] == "hosting.toolbox.callbacks.v2"
    assert "user_context" not in binding


def test_ensure_toolbox_assignments_ready_returns_rollout_metadata(monkeypatch) -> None:
    root = _scratch_dir("ready-rollout-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    monkeypatch.setattr(
        svc,
        "_wait_for_toolbox_executor_ready",
        lambda engine_id, timeout_seconds=8.0: {"status": "ok", "all_registered_tool_names": ["alpha_tool"]},
    )
    env_root = root / "toolbox_venvs" / "ready-rollout"
    env_root.mkdir(parents=True, exist_ok=True)
    (env_root / "environment.json").write_text(
        json.dumps(
            {
                "venv_key": "ready-rollout",
                "venv_path": str(env_root),
                "python_executable": sys.executable,
                "environment_name": "base",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    assignment = ToolboxSandboxAssignment(
        toolbox_id="toolbox-ready",
        sandbox_profile=SandboxProfileSpec(profile_id="fs-only"),
        bundle_spec=ToolboxBundleSpec(
            bundle_id="toolbox-ready-fs-only",
            toolbox_id="toolbox-ready",
            sandbox_profile=SandboxProfileSpec(profile_id="fs-only"),
            auto_tools=[ToolboxBundleAutoTool(module_name="alpha", callable_name="alpha_tool")],
        ),
        registration={
            "engine_id": "toolbox-ready-1",
            "environment": {
                "venv_key": "ready-rollout",
                "venv_path": str(env_root),
                "python_executable": sys.executable,
                "environment_name": "base",
            },
        },
    )

    rollout = svc._ensure_toolbox_assignments_ready([assignment], timeout_seconds=1.0)

    assert list(rollout.keys()) == ["toolbox-ready-1"]
    assert rollout["toolbox-ready-1"]["ready"] is True
    assert rollout["toolbox-ready-1"]["warmup_ms"] >= 0
    assert rollout["toolbox-ready-1"]["tool_inventory_ok"] is True
    assert rollout["toolbox-ready-1"]["tool_count"] == 1
    assert rollout["toolbox-ready-1"]["all_registered_tool_names"] == ["alpha_tool"]
    assert rollout["toolbox-ready-1"]["install_execution_status"] is None
    assert rollout["toolbox-ready-1"]["install_receipt_verification_status"] is None


def test_ensure_toolbox_assignments_ready_requires_verified_install_receipt(monkeypatch) -> None:
    root = _scratch_dir("ready-rollout-receipt-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    monkeypatch.setattr(
        svc,
        "_wait_for_toolbox_executor_ready",
        lambda engine_id, timeout_seconds=8.0: {"status": "ok", "all_registered_tool_names": ["alpha_tool"]},
    )
    env_root = root / "toolbox_venvs" / "ready-rollout-receipt"
    env_root.mkdir(parents=True, exist_ok=True)
    (env_root / "environment.json").write_text(
        json.dumps(
            {
                "venv_key": "ready-rollout-receipt",
                "venv_path": str(env_root),
                "python_executable": sys.executable,
                "environment_name": "base",
                "install_execution": {"status": "ok"},
                "install_receipt_verification": {"status": "mismatch"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    assignment = ToolboxSandboxAssignment(
        toolbox_id="toolbox-ready",
        sandbox_profile=SandboxProfileSpec(profile_id="fs-only"),
        bundle_spec=ToolboxBundleSpec(
            bundle_id="toolbox-ready-fs-only",
            toolbox_id="toolbox-ready",
            sandbox_profile=SandboxProfileSpec(profile_id="fs-only"),
            auto_tools=[ToolboxBundleAutoTool(module_name="alpha", callable_name="alpha_tool")],
        ),
        registration={
            "engine_id": "toolbox-ready-1",
            "bundle": {"toolbox_id": "toolbox-ready", "sandbox_profile_id": "fs-only"},
            "environment": {
                "venv_key": "ready-rollout-receipt",
                "venv_path": str(env_root),
                "python_executable": sys.executable,
                "environment_name": "base",
            },
        },
    )

    with pytest.raises(ToolboxRolloutError) as exc_info:
        svc._ensure_toolbox_assignments_ready([assignment], timeout_seconds=1.0)
    exc = exc_info.value
    assert exc.code == "toolbox_environment_receipt_unverified"
    assert dict(exc.details)["install_execution_status"] == "ok"
    assert dict(exc.details)["install_receipt_verification_status"] == "mismatch"


def test_wait_for_toolbox_executor_ready_requires_inventory_match(monkeypatch) -> None:
    root = Path(".tmp_ready_mismatch").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        svc.register_spawned(
            engine_id="toolbox-mismatch",
            pid=1234,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "missing.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-ready-mismatch",
            executor_kind="toolbox_executor",
            tool_access={"allowed_tool_names": ["alpha_tool"]},
        )
        monkeypatch.setattr(
            svc,
            "_toolbox_describe_live",
            lambda engine_id="", toolbox_id="", timeout_seconds=10.0: {"status": "ok", "all_registered_tool_names": ["wrong_tool"]},
        )
        with pytest.raises(ToolboxRolloutError) as exc_info:
            svc._wait_for_toolbox_executor_ready("toolbox-mismatch", timeout_seconds=0.15, poll_seconds=0.01)
        exc = exc_info.value
        assert exc.code == "toolbox_executor_inventory_mismatch"
        assert dict(exc.details)["failure_phase"] == "inventory_verified"
        assert dict(exc.details)["engine_id"] == "toolbox-mismatch"
        assert dict(exc.details)["expected_tool_names"] == ["alpha_tool"]
        assert dict(exc.details)["actual_tool_names"] == ["wrong_tool"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_executor_host_call_rpc_uses_host_dispatch(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    def _fake_invoke(method: str, arguments: dict, **_kwargs: Any) -> dict:
        calls.append((str(method), dict(arguments)))
        return {"entries": []}

    monkeypatch.setattr(toolbox_executor_ipc, "_invoke_host_call", _fake_invoke)

    out = asyncio.run(
        toolbox_executor_ipc._rpc_call(
            "host.call",
            {"method": "fs.list", "arguments": {"engine_id": "toolbox-b", "root_id": "rw"}},
        )
    )

    assert out == {"status": "ok", "result": {"entries": []}}
    assert calls == [("fs.list", {"engine_id": "toolbox-b", "root_id": "rw"})]


def test_toolbox_executor_manifest_path_prefers_startup_spec(monkeypatch) -> None:
    root = _scratch_dir("spec-path-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-spec-path",
                files=[ToolboxBundleFile(relative_path="demo_tools.py", content="def hello(name='x'):\n    return {'name': name}\n")],
                tools=[ToolboxBundleTool(definition=_tool_definition("hello_tool"), entrypoint="demo_tools:hello")],
            )
        )
        env = staged.worker_env_with_startup_spec(worker_id="toolbox-spec-path")
        monkeypatch.setenv("MP13_TOOLBOX_WORKER_SPEC_PATH", env["MP13_TOOLBOX_WORKER_SPEC_PATH"])
        monkeypatch.delenv("MP13_TOOLBOX_MANIFEST_PATH", raising=False)
        monkeypatch.setattr(toolbox_executor_ipc, "_startup_spec", None)
        path = toolbox_executor_ipc._manifest_path()
        assert path == staged.manifest_path.resolve()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_executor_host_service_prefers_startup_spec_metadata(monkeypatch) -> None:
    root = _scratch_dir("spec-hosting-")
    try:
        stager = ToolboxBundleStager(root)
        staged = stager.stage_bundle(
            ToolboxBundleSpec(
                bundle_id="bundle-spec-hosting",
                files=[ToolboxBundleFile(relative_path="demo_tools.py", content="def hello(name='x'):\n    return {'name': name}\n")],
                tools=[ToolboxBundleTool(definition=_tool_definition("hello_tool"), entrypoint="demo_tools:hello")],
            )
        )
        engines_state = (root / "engines.json").resolve()
        control_state = (root / "control.json").resolve()
        env = staged.worker_env_with_startup_spec(
            worker_id="toolbox-spec-hosting",
            engines_state_file=engines_state,
            control_state_file=control_state,
        )
        monkeypatch.setenv("MP13_TOOLBOX_WORKER_SPEC_PATH", env["MP13_TOOLBOX_WORKER_SPEC_PATH"])
        monkeypatch.setenv("MP13_HOSTING_ENGINES_STATE_FILE", str((root / "wrong-engines.json").resolve()))
        monkeypatch.setenv("MP13_HOSTING_CONTROL_STATE_FILE", str((root / "wrong-control.json").resolve()))
        monkeypatch.setattr(toolbox_executor_ipc, "_startup_spec", None)

        captured: dict[str, object] = {}

        class _FakeService:
            def __init__(self, *, engines_state_file=None, control_state_file=None, **_kwargs):
                captured["engines_state_file"] = engines_state_file
                captured["control_state_file"] = control_state_file

        monkeypatch.setattr("hosting.service.host_service.EngineHostService", _FakeService)
        _ = toolbox_executor_ipc._host_service()

        assert captured["engines_state_file"] == engines_state
        assert captured["control_state_file"] == control_state
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_executor_ipc_end_to_end() -> None:
    root = _scratch_dir("live-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    stager = ToolboxBundleStager(root)
    staged = stager.stage_bundle(
        ToolboxBundleSpec(
            bundle_id="bundle-live",
            files=[
                ToolboxBundleFile(
                    relative_path="demo_tools.py",
                    content=(
                        "def hello(name='world'):\n"
                        "    return {'greeting': f'hi {name}'}\n"
                    ),
                )
            ],
            tools=[ToolboxBundleTool(definition=_tool_definition("hello_tool"), entrypoint="demo_tools:hello")],
        )
    )

    reg = svc.spawn(
        engine_id="toolbox-live",
        command=staged.worker_command(python_executable=sys.executable),
        env=staged.worker_env(),
        worker_profile_class="generic",
        executor_kind="toolbox_executor",
        bundle=staged.registration_bundle(),
        environment=staged.registration_environment(),
        tool_access=staged.registration_tool_access(),
        capabilities={"brokered_filesystem": False, "brokered_http": False, "dynamic_reload": False},
    )
    assert reg["executor_kind"] == "toolbox_executor"
    try:
        deadline = time.time() + 8.0
        last_error: Exception | None = None
        desc = None
        while time.time() < deadline:
            try:
                desc = svc._toolbox_describe_live(engine_id="toolbox-live", timeout_seconds=2.0)
                break
            except Exception as exc:  # pragma: no cover - startup polling
                last_error = exc
                time.sleep(0.1)
        if desc is None:
            raise AssertionError(f"toolbox executor did not become ready: {last_error}")

        assert "hello_tool" in list(desc.get("all_registered_tool_names") or [])
        exec_out = svc.toolbox_execute(
            engine_id="toolbox-live",
            execution_request_id="exec-toolbox-live",
            tool_call={"name": "hello_tool", "arguments": {"name": "Sam"}},
            timeout_seconds=5.0,
        )
        tool_row = dict(exec_out.get("tool_call") or {})
        assert json.loads(str(tool_row.get("result") or "{}"))["greeting"] == "hi Sam"
    finally:
        try:
            os.kill(int(reg.get("pid") or 0), signal.SIGTERM)
            time.sleep(0.2)
        except Exception:
            pass
        _ = svc.remove_registration("toolbox-live")
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_executor_ipc_end_to_end_with_brokered_fs_callback() -> None:
    root = _scratch_dir("live-callback-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    data_root = root / "sandbox-data"
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "name.txt").write_text("callback-ok", encoding="utf-8")
    stager = ToolboxBundleStager(root)
    staged = stager.stage_bundle(
        ToolboxBundleSpec(
            bundle_id="bundle-live-callback",
            files=[
                ToolboxBundleFile(
                    relative_path="demo_tools.py",
                    content=(
                        "def read_name(**kwargs):\n"
                        "    ctx = kwargs['context']\n"
                        "    return ctx.fs.read_text(root_id='rw', relative_path='name.txt')\n"
                    ),
                )
            ],
            tools=[ToolboxBundleTool(definition=_tool_definition("read_name_tool"), entrypoint="demo_tools:read_name")],
        )
    )

    reg = svc.spawn(
        engine_id="toolbox-live-callback",
        command=staged.worker_command(python_executable=sys.executable),
        env=staged.worker_env(),
        worker_profile_class="generic",
        sandbox_policy={
            "sandbox": {
                "enabled": True,
                "filesystem": {
                    "rules": [
                        {
                            "root_id": "rw",
                            "path": str(data_root),
                            "access": ["read", "write"],
                        }
                    ]
                },
                "brokered_io": {"filesystem": True, "http": False, "subprocess": False},
            }
        },
        executor_kind="toolbox_executor",
        bundle=staged.registration_bundle(),
        environment=staged.registration_environment(),
        tool_access=staged.registration_tool_access(),
        capabilities={"brokered_filesystem": True, "brokered_http": False, "dynamic_reload": False},
    )
    try:
        deadline = time.time() + 8.0
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                _ = svc._toolbox_describe_live(engine_id="toolbox-live-callback", timeout_seconds=2.0)
                break
            except Exception as exc:  # pragma: no cover - startup polling
                last_error = exc
                time.sleep(0.1)
        else:
            raise AssertionError(f"toolbox executor did not become ready: {last_error}")

        direct_reg = svc.get_registration("toolbox-live-callback")
        direct_relay, direct_binding = svc._toolbox_host_capability_dispatch_binding(
            engine_id="toolbox-live-callback",
            toolbox_id="bundle-live-callback",
            tool_name="read_name_tool",
            tool_call_id="call-live-host-direct",
            tool_arguments={},
            sandbox_policy=dict(dict(direct_reg or {}).get("sandbox_policy") or {}),
        )
        try:
            host_out = svc._ipc_call(  # type: ignore[attr-defined]
                reg=direct_reg,
                payload={
                    "kind": "rpc_call",
                    "engine_id": "toolbox-live-callback",
                    "method": "host.call",
                    "params": {
                        "method": "fs.read_text",
                        "arguments": {"root_id": "rw", "relative_path": "name.txt"},
                        "callback_binding": direct_binding,
                    },
                },
                timeout_seconds=5.0,
            )
            assert dict(host_out.get("result") or {})["text"] == "callback-ok"
        finally:
            direct_relay.release_session(str(direct_binding.get("session_token") or ""))

        exec_out = svc.toolbox_execute(
            engine_id="toolbox-live-callback",
            execution_request_id="exec-call-live-fs-1",
            tool_call={"id": "call-live-fs-1", "name": "read_name_tool", "arguments": {}},
            timeout_seconds=5.0,
        )
        tool_row = dict(exec_out.get("tool_call") or {})
        parsed = json.loads(str(tool_row.get("result") or "{}"))
        assert parsed["text"] == "callback-ok"
        callback_context = dict(parsed["callback_context"])
        assert callback_context["engine_id"] == "toolbox-live-callback"
        assert callback_context["toolbox_id"] == "bundle-live-callback"
        assert callback_context["tool_name"] == "read_name_tool"
        assert callback_context["tool_call_id"] == "call-live-fs-1"
        assert callback_context["tool_arguments"] == {}
        assert callback_context["callback_signature"] is None
        assert callback_context["user_context"] is None
        surface = dict(callback_context["callable_surface"])
        assert surface["contract"] == "hosting.toolbox.brokered_io.call_surface.v1"
        assert surface["method"] == "fs.read_text"
        assert surface["identity"]["provider_id"] == "toolbox-live-callback"
        assert surface["identity"]["toolbox_id"] == "bundle-live-callback"
        assert surface["identity"]["session_id"] == "call-live-fs-1"
        assert surface["bridge_policy"]["namespaces"]["fs"] is True
        assert surface["bridge_policy"]["namespaces"]["http"] is False
    finally:
        try:
            os.kill(int(reg.get("pid") or 0), signal.SIGTERM)
            time.sleep(0.2)
        except Exception:
            pass
        _ = svc.remove_registration("toolbox-live-callback")
        shutil.rmtree(root, ignore_errors=True)




def test_hosted_toolbox_execution_harness_forwards_callback_processor() -> None:
    class _FakeChannel:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def toolbox_gate(self, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "ok", "outcome": "allowed"}

        def toolbox_describe(self, *, toolbox_id: str = "", engine_id: str = "") -> Dict[str, Any]:
            return {
                "status": "ok",
                "toolbox_id": toolbox_id or "toolbox-callback-harness",
                "all_registered_tool_names": ["callback_harness_tool"],
                "tool_metadata": {
                    "callback_harness_tool": {
                        "callback_signature": {"callbacks": [{"name": "echo_name", "payload_type": "object"}]},
                        "non_restartable": False,
                        "hidden": False,
                    }
                },
            }

        def toolbox_execute(self, **kwargs: Any) -> Dict[str, Any]:
            self.calls.append(dict(kwargs))
            return {"status": "ok", "tool_call": dict(kwargs.get("tool_call") or {}, result=json.dumps({"status": "ok"}))}

    channel = _FakeChannel()
    harness = ToolboxExecutionHarness(
        config=ToolboxHarnessConfig(mode="sandbox", sandbox_toolbox_id="toolbox-callback-harness"),
        control_channel=channel,
    )

    def _processor(*, callback_name: str, payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
        return {"value": dict(payload or {}).get("value"), "tool_call_id": context.tool_call_id}

    executed = asyncio.run(
        harness.execute_calls(
            [
                ToolCall(
                    id="call-harness-1",
                    name="callback_harness_tool",
                    arguments={"name": "Ava"},
                )
            ],
            callback_processor=_processor,
            callback_context={"origin": "harness"},
        )
    )

    tool_call = executed[0]
    assert json.loads(str(tool_call.result or "{}")) == {"status": "ok"}
    assert len(channel.calls) == 1
    payload = dict(channel.calls[0])
    assert dict(payload.get("tool_call") or {})["id"] == "call-harness-1"
    binding = dict(payload.get("callback_binding") or {})
    assert binding["contract"] == "hosting.toolbox.callbacks.v2"
    assert binding["session_token"]
    assert binding["address"]
    assert binding["family"] in {"AF_PIPE", "AF_UNIX"}




def test_toolbox_executor_ipc_end_to_end_with_intrinsic_tools_only() -> None:
    root = _scratch_dir("live-intrinsic-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    stager = ToolboxBundleStager(root)
    staged = stager.stage_bundle(
        ToolboxBundleSpec(
            bundle_id="bundle-live-intrinsic",
            with_intrinsics=True,
            with_intrinsic_guides=True,
            intrinsic_tool_names=["symbolic_algebra", "symbolic_algebra_guide"],
        )
    )

    reg = svc.spawn(
        engine_id="toolbox-live-intrinsic",
        command=staged.worker_command(python_executable=sys.executable),
        env=staged.worker_env(),
        worker_profile_class="generic",
        executor_kind="toolbox_executor",
        bundle=staged.registration_bundle(),
        environment=staged.registration_environment(),
        tool_access=staged.registration_tool_access(),
        capabilities={"brokered_filesystem": False, "brokered_http": False, "dynamic_reload": False},
    )
    try:
        deadline = time.time() + 8.0
        last_error: Exception | None = None
        desc = None
        while time.time() < deadline:
            try:
                desc = svc._toolbox_describe_live(engine_id="toolbox-live-intrinsic", timeout_seconds=2.0)
                break
            except Exception as exc:  # pragma: no cover - startup polling
                last_error = exc
                time.sleep(0.1)
        if desc is None:
            raise AssertionError(f"toolbox executor did not become ready: {last_error}")

        assert "symbolic_algebra" in list(desc.get("all_registered_tool_names") or [])
        assert "symbolic_algebra_guide" in list(desc.get("all_registered_tool_names") or [])

        exec_out = svc.toolbox_execute(
            engine_id="toolbox-live-intrinsic",
            execution_request_id="exec-toolbox-live-intrinsic",
            tool_call={
                "name": "symbolic_algebra",
                "arguments": {"expr": "2 + 2", "variables": [], "operation": "simplify"},
            },
            timeout_seconds=5.0,
        )
        tool_row = dict(exec_out.get("tool_call") or {})
        assert "4" in str(tool_row.get("result") or "")
    finally:
        try:
            os.kill(int(reg.get("pid") or 0), signal.SIGTERM)
            time.sleep(0.2)
        except Exception:
            pass
        _ = svc.remove_registration("toolbox-live-intrinsic")
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_executor_ipc_end_to_end_with_auto_callable_discovery() -> None:
    root = _scratch_dir("live-auto-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    stager = ToolboxBundleStager(root)
    staged = stager.stage_bundle(
        ToolboxBundleSpec(
            bundle_id="bundle-live-auto",
            files=[
                ToolboxBundleFile(
                    relative_path="auto_tools.py",
                    content=(
                        "def hello_auto(name: str = 'world'):\n"
                        "    \"\"\"Return a greeting.\n\n"
                        "    Args:\n"
                        "        name (str): Name to greet.\n"
                        "    \"\"\"\n"
                        "    return {'greeting': f'hi {name}'}\n"
                    ),
                )
            ],
            auto_tools=[ToolboxBundleAutoTool(module_name="auto_tools", callable_name="hello_auto")],
        )
    )

    reg = svc.spawn(
        engine_id="toolbox-live-auto",
        command=staged.worker_command(python_executable=sys.executable),
        env=staged.worker_env(),
        worker_profile_class="generic",
        executor_kind="toolbox_executor",
        bundle=staged.registration_bundle(),
        environment=staged.registration_environment(),
        tool_access=staged.registration_tool_access(),
        capabilities={"brokered_filesystem": False, "brokered_http": False, "dynamic_reload": False},
    )
    try:
        deadline = time.time() + 8.0
        last_error: Exception | None = None
        desc = None
        while time.time() < deadline:
            try:
                desc = svc._toolbox_describe_live(engine_id="toolbox-live-auto", timeout_seconds=2.0)
                break
            except Exception as exc:  # pragma: no cover - startup polling
                last_error = exc
                time.sleep(0.1)
        if desc is None:
            raise AssertionError(f"toolbox executor did not become ready: {last_error}")

        assert "hello_auto" in list(desc.get("all_registered_tool_names") or [])
        exec_out = svc.toolbox_execute(
            engine_id="toolbox-live-auto",
            execution_request_id="exec-toolbox-live-auto",
            tool_call={"name": "hello_auto", "arguments": {"name": "Sam"}},
            timeout_seconds=5.0,
        )
        tool_row = dict(exec_out.get("tool_call") or {})
        assert "hi Sam" in str(tool_row.get("result") or "")
    finally:
        try:
            os.kill(int(reg.get("pid") or 0), signal.SIGTERM)
            time.sleep(0.2)
        except Exception:
            pass
        _ = svc.remove_registration("toolbox-live-auto")
        shutil.rmtree(root, ignore_errors=True)
