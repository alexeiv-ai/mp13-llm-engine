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

from hosting.service.host_service import EngineHostService, ToolboxRolloutError
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
                "status": "ok",
                "tool_call": {
                    **tool_call,
                    "result": json.dumps({"greeting": f"hi {dict(tool_call.get('arguments') or {}).get('name', 'world')}"}),
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
    ref = HostedToolBoxRef(toolbox_id="user-tools", host=_FakeHost(), python_executable="python.exe")

    out = ref.execute(
        tool_name="dangerous_remote",
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
        assert spec.intrinsics_profile_id == "symbolic_math"
        assert spec.required_imports == ["requests", "numpy"]
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


def test_toolbox_orchestrator_spawn_uses_shared_environment_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _scratch_dir("orchestrator-runtime-base-")
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        spawned: list[dict] = []

        def fake_spawn(**kwargs):
            spawned.append(dict(kwargs))
            return {
                "engine_id": kwargs["engine_id"],
                "executor_kind": kwargs["executor_kind"],
                "environment": dict(kwargs["environment"]),
            }

        monkeypatch.setattr(svc, "spawn", fake_spawn)
        orchestrator = ToolboxSandboxOrchestrator(
            service=svc,
            stager=ToolboxBundleStager(root),
            python_executable=sys.executable,
        )
        assignments = orchestrator.spawn_assignments(
            toolbox_id="toolbox-runtime-spawn",
            requests=[
                ToolboxAutoAssignmentRequest(
                    files=[ToolboxBundleFile(relative_path="demo.py", content="def demo():\n    return {'ok': True}\n")],
                    module_name="demo",
                    callable_name="demo",
                    sandbox_profile=SandboxProfileSpec(sandbox_policy={"sandbox": {"enabled": True}}),
                )
            ],
        )

        assert len(assignments) == 1
        assert spawned[0]["executor_kind"] == "toolbox_executor"
        env = dict(spawned[0]["environment"])
        assert env["environment_key"]
        assert env["environment_identity"]["runtime"]["runtime_kind"] == "toolbox_executor"
        assert assignments[0].registration["environment"]["environment_key"] == env["environment_key"]
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
            tool_call={"id": "call-demo-1", "name": "demo_tool", "arguments": {}},
            timeout_seconds=2.0,
        )
        status = svc.toolbox_request_status(
            engine_id="toolbox-hosted-pool",
            request_id="call-demo-1",
        )
    finally:
        svc.shutdown("toolbox-hosted-pool", timeout_seconds=2.0)
        svc.remove_registration("toolbox-hosted-pool")
        shutil.rmtree(root, ignore_errors=True)

    assert int(reg.get("pid") or 0)
    assert out["status"] == "ok"
    assert out["environment_key"] == "toolbox-env-key"
    assert out["hosted_pool"]["metrics"]["desired_capacity"] == 2
    assert out["hosted_pool"]["metrics"]["recent_requests"][-1]["request_id"] == "call-demo-1"
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
            tool_call={"id": "call-all-settled-error", "name": "demo_tool", "arguments": {}},
        )

        assert out["status"] == "error"
        assert out["reason"] == "worker failed after admission"
        assert out["tool_call_id"] == "call-all-settled-error"
        assert out["request"]["request_id"] == "call-all-settled-error"
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
        out = svc.toolbox_cancel(
            engine_id="toolbox-sandbox-recycled",
            tool_name="demo_tool",
            tool_call_id="call-recycled-target",
            respawn=False,
        )

        sibling = base.request_status(environment_key="toolbox-recycled-env", request_id="call-recycled-sibling")
        assert out["sandbox_recycled_request_ids"]["toolbox-sandbox-recycled"] == ["call-recycled-sibling"]
        assert sibling["request"]["status"] == "error"
        assert sibling["request"]["reason"] == "sandbox_recycled"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_real_local_control_path_overlaps_actual_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hosting.daemon.security._tighten_windows_acl", lambda *_args, **_kwargs: None)
    root = _scratch_dir("toolbox-real-control-concurrency-")
    pid_file = root / "daemon.pid"
    daemon = EngineHostDaemon(
        pid_file=pid_file,
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    daemon._execute_startup_worker_recovery = lambda: {"status": "ok"}  # type: ignore[method-assign]
    daemon_errors: list[BaseException] = []

    def run_daemon() -> None:
        try:
            asyncio.run(daemon.run())
        except BaseException as exc:  # pragma: no cover - startup diagnostics
            daemon_errors.append(exc)

    daemon_thread = threading.Thread(target=run_daemon, daemon=True)
    daemon_thread.start()
    channel = EngineHostControlChannel(
        {
            "engine_host_daemon_pid_file": str(pid_file),
            "engine_host_daemon_auto_bootstrap": False,
        }
    )
    ref = HostedToolBoxRef(
        toolbox_id="toolbox-real-control-concurrency",
        host=channel,
        python_executable=sys.executable,
    )
    try:
        last_error = None
        for _ in range(400):
            try:
                if pid_file.exists() and channel.discover_running() is not None:
                    break
            except Exception as exc:
                last_error = exc
                time.sleep(0.05)
        else:
            raise AssertionError(
                f"daemon did not become reachable: {last_error}; errors={daemon_errors}; alive={daemon_thread.is_alive()}; "
                f"ready={daemon._local_listener_ready.is_set()}; listener_error={daemon._local_listener_error}; pid={daemon.pid_file.read()}"
            )

        registration = ref.register_auto_callable(
            relative_path="real_parallel_tools.py",
            content=(
                "import time\n"
                "def sleep_tool(delay=0.2, label=''):\n"
                "    time.sleep(float(delay))\n"
                "    return {'label': label, 'delay': delay}\n"
            ),
            module_name="real_parallel_tools",
            callable_name="sleep_tool",
            concurrency={"mode": "parallel", "max_concurrency": 2},
        )
        engine_id = str(list(registration.get("ready_engine_ids") or [""])[0] or "")
        assert engine_id

        async def run_calls() -> list[dict]:
            first = asyncio.create_task(
                asyncio.to_thread(
                    ref.execute,
                    tool_name="sleep_tool",
                    arguments={"delay": 0.2, "label": "a"},
                    tool_call_id="real-call-a",
                )
            )
            await asyncio.sleep(0.05)
            status_started = time.perf_counter()
            status = await asyncio.to_thread(
                channel.toolbox_request_status,
                engine_id=engine_id,
                request_id="real-call-a",
            )
            status_elapsed = time.perf_counter() - status_started
            assert status["status"] == "ok"
            assert status["request"]["request_id"] == "real-call-a"
            assert status_elapsed < 0.25
            second = asyncio.create_task(
                asyncio.to_thread(
                    ref.execute,
                    tool_name="sleep_tool",
                    arguments={"delay": 0.2, "label": "b"},
                    tool_call_id="real-call-b",
                )
            )
            return await asyncio.gather(first, second)

        started = time.perf_counter()
        results = asyncio.run(run_calls())
        elapsed = time.perf_counter() - started

        assert all(result["status"] == "ok" for result in results)
        assert {dict(result["tool_call"]).get("id") for result in results} == {"real-call-a", "real-call-b"}
        assert elapsed < 0.38
    finally:
        try:
            channel.stop_daemon(reason="test_complete", requested_by="test_real_local_control_path_overlaps_actual_tool_calls")
        except Exception:
            pass
        if daemon_thread.is_alive() and daemon._loop is not None and daemon._stop_event is not None:
            daemon._loop.call_soon_threadsafe(daemon._stop_event.set)
        daemon_thread.join(timeout=10.0)
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
        out = svc.toolbox_cancel(
            engine_id="toolbox-hosted-cancel",
            tool_name="demo_tool",
            tool_call_id="call-cancel-1",
            timeout_seconds=2.0,
            respawn=False,
        )
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


def test_toolbox_cancel_routes_targeted_profile_and_repairs_toolbox(monkeypatch) -> None:
    root = _scratch_dir("service-cancel-")
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        svc.register_spawned(
            engine_id="toolbox-cancel-alpha",
            pid=1111,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "alpha.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-toolbox-cancel-alpha",
            executor_kind="toolbox_executor",
            bundle={"toolbox_id": "toolbox-cancel", "sandbox_profile_id": "alpha"},
            tool_access={"allowed_tool_names": ["alpha_tool"]},
        )
        svc.register_spawned(
            engine_id="toolbox-cancel-beta",
            pid=2222,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "beta.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-toolbox-cancel-beta",
            executor_kind="toolbox_executor",
            bundle={"toolbox_id": "toolbox-cancel", "sandbox_profile_id": "beta"},
            tool_access={"allowed_tool_names": ["beta_tool"]},
        )
        state = svc._read_toolboxes()
        state["toolboxes"] = {
            "toolbox-cancel": {
                "toolbox_id": "toolbox-cancel",
                "requests": [
                    {
                        "files": [{"relative_path": "beta.py", "content": "def beta_tool():\n    return 'ok'\n"}],
                        "module_name": "beta",
                        "callable_name": "beta_tool",
                        "sandbox_profile": {"profile_id": "beta", "environment_name": "base", "required_imports": [], "sandbox_policy": {}},
                        "activate": True,
                        "hidden": False,
                        "non_restartable": True,
                    }
                ],
                "manual_requests": [],
                "profiles": {},
                "runtime": {},
            }
        }
        svc._write_toolboxes(state)

        shutdown_calls: list[tuple[str, float]] = []
        repair_calls: list[dict[str, Any]] = []

        def _fake_shutdown(engine_id: str, *, timeout_seconds: float = 8.0) -> Dict[str, Any]:
            shutdown_calls.append((engine_id, timeout_seconds))
            return {"status": "stopped", "engine_id": engine_id, "alive": False}

        def _fake_repair(*, toolbox_ids=None, only_inconsistent: bool = True, details: bool = False) -> Dict[str, Any]:
            repair_calls.append(
                {
                    "toolbox_ids": list(toolbox_ids or []),
                    "only_inconsistent": only_inconsistent,
                    "details": details,
                }
            )
            return {"status": "ok", "repaired_toolbox_ids": ["toolbox-cancel"]}

        monkeypatch.setattr(svc, "shutdown", _fake_shutdown)
        monkeypatch.setattr(svc, "toolbox_repair", _fake_repair)

        out = svc.toolbox_cancel(
            toolbox_id="toolbox-cancel",
            tool_name="beta_tool",
            tool_call_id="call-beta-1",
            timeout_seconds=3.5,
        )

        assert shutdown_calls == [("toolbox-cancel-beta", 3.5)]
        assert repair_calls == [
            {
                "toolbox_ids": ["toolbox-cancel"],
                "only_inconsistent": False,
                "details": False,
            }
        ]
        assert out["outcome"] == "canceled_and_repaired"
        assert out["canceled_engine_ids"] == ["toolbox-cancel-beta"]
        assert out["repaired_toolbox_ids"] == ["toolbox-cancel"]
        updated_state = svc._read_toolboxes()
        cancel_events = list(dict(dict(updated_state.get("toolboxes") or {}).get("toolbox-cancel") or {}).get("runtime", {}).get("cancel_events") or [])
        assert len(cancel_events) == 1
        assert cancel_events[0]["tool_name"] == "beta_tool"
        assert cancel_events[0]["tool_call_id"] == "call-beta-1"
        assert cancel_events[0]["non_restartable"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_cancel_returns_noop_when_target_is_missing() -> None:
    root = _scratch_dir("service-cancel-missing-")
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )

        out = svc.toolbox_cancel(toolbox_id="missing-box")

        assert out["outcome"] == "noop"
        assert out["reason"] == "toolbox_executor_missing"
        assert out["canceled_engine_ids"] == []
        assert out["repaired_toolbox_ids"] == []
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


def test_toolbox_describe_separates_allowed_and_advertised_visibility() -> None:
    root = _scratch_dir("service-describe-")
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
            bundle={"toolbox_id": "demo-box", "sandbox_profile_id": "default"},
            tool_access={
                "allowed_tool_names": ["hello_tool", "hidden_tool"],
                "advertised_tool_names": ["hello_tool"],
                "hidden_allowed_tool_names": ["hidden_tool"],
            },
        )

        desc = svc.toolbox_describe(toolbox_id="demo-box")

        assert desc["allowed_tool_names"] == ["hello_tool", "hidden_tool"]
        assert desc["advertised_tool_names"] == ["hello_tool"]
        assert desc["hidden_allowed_tool_names"] == ["hidden_tool"]
        assert desc["all_registered_tool_names"] == ["hello_tool", "hidden_tool"]
        assert desc["all_registered_tool_names"] == ["hello_tool", "hidden_tool"]
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
    ]
    out = asyncio.run(harness.execute_calls(calls, parallel=True))

    assert state["max_active"] >= 2
    assert [json.loads(str(item.result or "{}"))["name"] for item in out] == ["a", "b"]


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


def test_sandboxed_toolbox_facade_shapes_requests_for_host_api() -> None:
    class _FakeHost:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def toolbox_register_auto(self, **kwargs):
            self.calls.append(("register", dict(kwargs)))
            return {"status": "ok"}

        def toolbox_unregister_auto(self, **kwargs):
            self.calls.append(("unregister", dict(kwargs)))
            return {"status": "ok"}

        def toolbox_register_intrinsics(self, **kwargs):
            self.calls.append(("register_intrinsics", dict(kwargs)))
            return {"status": "ok"}

        def toolbox_unregister_intrinsics(self, **kwargs):
            self.calls.append(("unregister_intrinsics", dict(kwargs)))
            return {"status": "ok"}

        def toolbox_register_manual(self, **kwargs):
            self.calls.append(("register_manual", dict(kwargs)))
            return {"status": "ok"}

        def toolbox_unregister_manual(self, **kwargs):
            self.calls.append(("unregister_manual", dict(kwargs)))
            return {"status": "ok"}

        def toolbox_environment_description_upsert(self, **kwargs):
            self.calls.append(("toolbox_environment_description_upsert", dict(kwargs)))
            return {"status": "ok", "environment_description": dict(kwargs)}

        def toolbox_environment_description_list(self, **kwargs):
            self.calls.append(("toolbox_environment_description_list", dict(kwargs)))
            return {"status": "ok", "environment_descriptions": {"base": {"name": "base"}}}

        def toolbox_environment_description_clone(self, **kwargs):
            self.calls.append(("toolbox_environment_description_clone", dict(kwargs)))
            return {"status": "ok", "environment_description": dict(kwargs)}

        def toolbox_environment_resolve_requirements(self, **kwargs):
            self.calls.append(("toolbox_environment_resolve_requirements", dict(kwargs)))
            return {"status": "ok"}

        def toolbox_environment_apply(self, **kwargs):
            self.calls.append(("toolbox_environment_apply", dict(kwargs)))
            return {"status": "ok", "affected_toolbox_ids": list(kwargs.get("toolbox_ids") or [])}

        def toolbox_environment_realize(self, **kwargs):
            self.calls.append(("toolbox_environment_realize", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id")}

        def toolbox_environment_sync_description(self, **kwargs):
            self.calls.append(("toolbox_environment_sync_description", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id")}

        def toolbox_environment_prepare_install(self, **kwargs):
            self.calls.append(("toolbox_environment_prepare_install", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id")}

        def toolbox_environment_lock_install(self, **kwargs):
            self.calls.append(("toolbox_environment_lock_install", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id")}

        def toolbox_environment_resolve_install_lock(self, **kwargs):
            self.calls.append(("toolbox_environment_resolve_install_lock", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id")}

        def toolbox_environment_verify_install_lock(self, **kwargs):
            self.calls.append(("toolbox_environment_verify_install_lock", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id")}

        def toolbox_environment_verify_install_receipt(self, **kwargs):
            self.calls.append(("toolbox_environment_verify_install_receipt", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id")}

        def toolbox_environment_execute_install(self, **kwargs):
            self.calls.append(("toolbox_environment_execute_install", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id")}

        def toolbox_describe(self, **kwargs):
            self.calls.append(("describe", dict(kwargs)))
            return {"status": "ok", "all_registered_tool_names": ["hello_auto"]}

        def toolbox_execute(self, **kwargs):
            self.calls.append(("execute", dict(kwargs)))
            return {"status": "ok", "tool_call": {"name": "hello_auto"}}

    host = _FakeHost()
    facade = SandboxedToolboxFacade(toolbox_id="facade-box", host=host, python_executable="python.exe")

    _ = facade.register_auto_callable(
        relative_path="facade_tools.py",
        content="def hello_auto(name='world'):\n    return {'greeting': f'hi {name}'}\n",
        module_name="facade_tools",
        callable_name="hello_auto",
        required_imports=["requests"],
        sandbox_policy={"sandbox": {"enabled": True}},
    )
    _ = facade.describe(timeout_seconds=3.0)
    _ = facade.execute(tool_name="hello_auto", arguments={"name": "Sam"}, timeout_seconds=4.0)
    _ = facade.register_intrinsic_tools(["symbolic_algebra"], include_guides=True, sandbox_policy={"sandbox": {"enabled": True}})
    _ = facade.unregister_intrinsic_tools(["symbolic_algebra"], include_guides=True)
    _ = facade.register_manual_tool(
        _tool_definition("manual_tool"),
        test_sandboxed_toolbox_facade_shapes_requests_for_host_api,
        required_imports=["numpy"],
        sandbox_policy={"sandbox": {"enabled": True}},
    )
    _ = facade.unregister_manual_tool(module_name=__name__, callable_name="test_sandboxed_toolbox_facade_shapes_requests_for_host_api")
    _ = facade.upsert_environment_description(name="math-env", base_env_name="base", extra_packages=["numpy", "sympy"])
    _ = facade.clone_environment_description(source_name="math-env", target_name="math-env-v2", extra_packages=["numpy", "sympy", "pandas"])
    _ = facade.environment_descriptions()
    _ = facade.resolve_environment_requirements(environment_name="math-env", tool_keys=["facade_tools:hello_auto"])
    _ = facade.apply_environment_description(environment_name="math-env", toolbox_ids=["facade-box"])
    _ = facade.realize_environment(environment_name="math-env", tool_keys=["facade_tools:hello_auto"])
    _ = facade.sync_environment_description(
        source_environment_name="math-env",
        target_environment_name="math-env-v2",
        tool_keys=["facade_tools:hello_auto"],
        apply=True,
        realize=True,
    )
    _ = facade.prepare_environment_install(environment_name="math-env", tool_keys=["facade_tools:hello_auto"])
    _ = facade.lock_environment_install(environment_name="math-env", tool_keys=["facade_tools:hello_auto"])
    _ = facade.resolve_environment_install_lock(environment_name="math-env", tool_keys=["facade_tools:hello_auto"], allow_resolution=True)
    _ = facade.verify_environment_install_lock(environment_name="math-env", tool_keys=["facade_tools:hello_auto"])
    _ = facade.verify_environment_install_receipt(environment_name="math-env", tool_keys=["facade_tools:hello_auto"])
    _ = facade.execute_environment_install(environment_name="math-env", tool_keys=["facade_tools:hello_auto"], allow_execution=True)
    _ = facade.unregister_auto_callable(module_name="facade_tools", callable_name="hello_auto")

    register_payload = host.calls[0][1]
    assert register_payload["toolbox_id"] == "facade-box"
    assert register_payload["python_executable"] == "python.exe"
    request = list(register_payload["requests"])[0]
    assert request["module_name"] == "facade_tools"
    assert request["callable_name"] == "hello_auto"
    assert request["sandbox_profile"]["required_imports"] == ["requests"]
    assert request["sandbox_profile"]["sandbox_policy"] == {"sandbox": {"enabled": True}}
    assert host.calls[1] == ("describe", {"toolbox_id": "facade-box", "timeout_seconds": 3.0})
    assert host.calls[2][0] == "execute"
    assert host.calls[2][1]["toolbox_id"] == "facade-box"
    assert host.calls[2][1]["tool_call"]["name"] == "hello_auto"
    assert host.calls[2][1]["tool_call"]["arguments"] == {"name": "Sam"}
    assert host.calls[2][1]["timeout_seconds"] == 4.0
    assert host.calls[2][1]["tools_view"] is None
    assert host.calls[2][1]["callback_binding"] is None
    assert host.calls[2][1]["tool_call"]["id"]
    assert host.calls[3][0] == "register_intrinsics"
    assert host.calls[3][1]["toolbox_id"] == "facade-box"
    assert host.calls[3][1]["intrinsic_tool_names"] == ["symbolic_algebra"]
    assert host.calls[3][1]["include_guides"] is True
    assert host.calls[3][1]["sandbox_profile"]["sandbox_policy"] == {"sandbox": {"enabled": True}}
    assert str(host.calls[3][1]["sandbox_profile"]["profile_id"]).startswith("profile-")
    assert host.calls[3][1]["python_executable"] == "python.exe"
    assert host.calls[3][1]["worker_profile_class"] == "generic"
    assert host.calls[4] == (
        "unregister_intrinsics",
        {
            "toolbox_id": "facade-box",
            "intrinsic_tool_names": ["symbolic_algebra"],
            "include_guides": True,
            "python_executable": "python.exe",
            "worker_profile_class": "generic",
        },
    )
    assert host.calls[5][0] == "register_manual"
    assert host.calls[5][1]["toolbox_id"] == "facade-box"
    manual_request = list(host.calls[5][1]["requests"])[0]
    assert manual_request["tool_definition"]["function"]["name"] == "manual_tool"
    assert manual_request["sandbox_profile"]["required_imports"] == ["numpy"]
    assert host.calls[5][1]["python_executable"] == "python.exe"
    assert host.calls[6] == (
        "unregister_manual",
        {
            "toolbox_id": "facade-box",
            "tool_keys": [f"manual:{__name__}:test_sandboxed_toolbox_facade_shapes_requests_for_host_api"],
            "python_executable": "python.exe",
            "worker_profile_class": "generic",
        },
    )
    assert host.calls[7] == (
        "toolbox_environment_description_upsert",
        {
            "name": "math-env",
            "base_env_name": "base",
            "extra_packages": ["numpy", "sympy"],
            "allow_online_install": False,
        },
    )
    assert host.calls[8] == (
        "toolbox_environment_description_clone",
        {
            "source_name": "math-env",
            "target_name": "math-env-v2",
            "extra_packages": ["numpy", "sympy", "pandas"],
            "allow_online_install": None,
        },
    )
    assert host.calls[9] == ("toolbox_environment_description_list", {})
    assert host.calls[10] == (
        "toolbox_environment_resolve_requirements",
        {
            "toolbox_id": "facade-box",
            "environment_name": "math-env",
            "tool_keys": ["facade_tools:hello_auto"],
        },
    )
    assert host.calls[11] == (
        "toolbox_environment_apply",
        {
            "environment_name": "math-env",
            "toolbox_ids": ["facade-box"],
        },
    )
    assert host.calls[12] == (
        "toolbox_environment_realize",
        {
            "toolbox_id": "facade-box",
            "environment_name": "math-env",
            "tool_keys": ["facade_tools:hello_auto"],
        },
    )
    assert host.calls[13] == (
        "toolbox_environment_sync_description",
        {
            "toolbox_id": "facade-box",
            "source_environment_name": "math-env",
            "target_environment_name": "math-env-v2",
            "tool_keys": ["facade_tools:hello_auto"],
            "apply": True,
            "realize": True,
        },
    )
    assert host.calls[14] == (
        "toolbox_environment_prepare_install",
        {
            "toolbox_id": "facade-box",
            "environment_name": "math-env",
            "tool_keys": ["facade_tools:hello_auto"],
        },
    )
    assert host.calls[15] == (
        "toolbox_environment_lock_install",
        {
            "toolbox_id": "facade-box",
            "environment_name": "math-env",
            "tool_keys": ["facade_tools:hello_auto"],
        },
    )
    assert host.calls[16] == (
        "toolbox_environment_resolve_install_lock",
        {
            "toolbox_id": "facade-box",
            "environment_name": "math-env",
            "tool_keys": ["facade_tools:hello_auto"],
            "allow_resolution": True,
        },
    )
    assert host.calls[17] == (
        "toolbox_environment_verify_install_lock",
        {
            "toolbox_id": "facade-box",
            "environment_name": "math-env",
            "tool_keys": ["facade_tools:hello_auto"],
        },
    )
    assert host.calls[18] == (
        "toolbox_environment_verify_install_receipt",
        {
            "toolbox_id": "facade-box",
            "environment_name": "math-env",
            "tool_keys": ["facade_tools:hello_auto"],
        },
    )
    assert host.calls[19] == (
        "toolbox_environment_execute_install",
        {
            "toolbox_id": "facade-box",
            "environment_name": "math-env",
            "tool_keys": ["facade_tools:hello_auto"],
            "allow_execution": True,
        },
    )
    assert host.calls[20] == (
        "unregister",
        {
            "toolbox_id": "facade-box",
            "tool_keys": ["facade_tools:hello_auto"],
            "python_executable": "python.exe",
            "worker_profile_class": "generic",
        },
    )


def test_hosted_toolbox_ref_aliases_and_ref_style_methods_shape_requests() -> None:
    class _FakeHost:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Dict[str, Any]]] = []

        def toolbox_register_auto(self, **kwargs):
            self.calls.append(("toolbox_register_auto", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id")}

        def toolbox_unregister_auto(self, **kwargs):
            self.calls.append(("toolbox_unregister_auto", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id")}

        def toolbox_register_intrinsics(self, **kwargs):
            self.calls.append(("toolbox_register_intrinsics", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id")}

        def toolbox_unregister_intrinsics(self, **kwargs):
            self.calls.append(("toolbox_unregister_intrinsics", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id")}

        def toolbox_describe(self, **kwargs):
            self.calls.append(("toolbox_describe", dict(kwargs)))
            return {"status": "ok", "all_registered_tool_names": ["hello_auto"]}

        def toolbox_execute(self, **kwargs):
            self.calls.append(("toolbox_execute", dict(kwargs)))
            return {"status": "ok", "tool_call": {"name": "hello_auto"}}

        def toolbox_cancel(self, **kwargs):
            self.calls.append(("toolbox_cancel", dict(kwargs)))
            return {"status": "ok", "toolbox_id": kwargs.get("toolbox_id"), "outcome": "canceled_and_repaired"}

        def toolbox_gate(self, **kwargs):
            self.calls.append(("toolbox_gate", dict(kwargs)))
            return {"status": "ok", "outcome": "allowed", "tool_name": kwargs.get("tool_name")}

        def toolbox_environment_description_list(self, **kwargs):
            self.calls.append(("toolbox_environment_description_list", dict(kwargs)))
            return {"status": "ok", "environment_descriptions": {}}

    ref = HostedToolBoxRef(toolbox_id="hosted-ref", host=_FakeHost(), python_executable="python.exe")
    tools_view = ToolsView(
        view_id="turn-1",
        mode="advertised",
        allowed_tools={"hello_auto"},
        advertised_tools={"hello_auto"},
        hidden_allowed_tools=set(),
        disabled_tools={"blocked_tool"},
    )
    assert ref.ref_name == "hosted-ref"
    _ = ref.add_auto_callable(
        relative_path="facade_tools.py",
        content="def hello_auto(name='world'):\n    return {'greeting': f'hi {name}'}\n",
        module_name="facade_tools",
        callable_name="hello_auto",
        non_restartable=True,
    )
    _ = ref.remove_auto_callable(module_name="facade_tools", callable_name="hello_auto")
    _ = ref.add_intrinsic_tools(["symbolic_algebra"])
    _ = ref.remove_intrinsic_tools(["symbolic_algebra"])
    _ = ref.list_tools()
    _ = ref.gate(tool_name="hello_auto", tools_view=tools_view)
    _ = ref.execute(tool_name="hello_auto", tools_view=tools_view)
    _ = ref.cancel(tool_name="hello_auto", tool_call_id="call-1", timeout_seconds=4.0, respawn=False)
    _ = ref.list_environment_descriptions()

    calls = ref.host.calls
    assert calls[0][0] == "toolbox_register_auto"
    assert calls[0][1]["requests"][0]["non_restartable"] is True
    assert calls[1][0] == "toolbox_unregister_auto"
    assert calls[2][0] == "toolbox_register_intrinsics"
    assert calls[3][0] == "toolbox_unregister_intrinsics"
    assert calls[4] == ("toolbox_describe", {"toolbox_id": "hosted-ref", "timeout_seconds": 10.0})
    assert calls[5] == (
        "toolbox_gate",
        {
            "toolbox_id": "hosted-ref",
            "tool_name": "hello_auto",
            "tools_view": {
                "view_id": "turn-1",
                "mode": "advertised",
                "allowed_tools": ["hello_auto"],
                "advertised_tools": ["hello_auto"],
                "hidden_allowed_tools": [],
                "disabled_tools": ["blocked_tool"],
                "gated_tools": [],
            },
        },
    )
    assert calls[6] == (
        "toolbox_gate",
        {
            "toolbox_id": "hosted-ref",
            "tool_name": "hello_auto",
            "tools_view": {
                "view_id": "turn-1",
                "mode": "advertised",
                "allowed_tools": ["hello_auto"],
                "advertised_tools": ["hello_auto"],
                "hidden_allowed_tools": [],
                "disabled_tools": ["blocked_tool"],
                "gated_tools": [],
            },
        },
    )
    assert calls[7][0] == "toolbox_execute"
    assert calls[7][1]["toolbox_id"] == "hosted-ref"
    assert calls[7][1]["tool_call"]["name"] == "hello_auto"
    assert calls[7][1]["tool_call"]["arguments"] == {}
    assert calls[7][1]["timeout_seconds"] == 30.0
    assert calls[7][1]["tools_view"] == {
        "view_id": "turn-1",
        "mode": "advertised",
        "allowed_tools": ["hello_auto"],
        "advertised_tools": ["hello_auto"],
        "hidden_allowed_tools": [],
        "disabled_tools": ["blocked_tool"],
        "gated_tools": [],
    }
    assert calls[7][1]["callback_binding"] is None
    assert calls[7][1]["tool_call"]["id"]
    assert calls[8] == (
        "toolbox_cancel",
        {
            "toolbox_id": "hosted-ref",
            "tool_name": "hello_auto",
            "tool_call_id": "call-1",
            "timeout_seconds": 4.0,
            "respawn": False,
        },
    )
    assert calls[9] == ("toolbox_environment_description_list", {})


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
    ref = HostedToolBoxRef(toolbox_id="hosted-ref", host=host, python_executable="python.exe")
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
        python_executable="python-demo",
        worker_profile_class="generic",
    )

    payload = ref.to_dict()
    restored = HostedToolBoxRef.from_dict(payload)

    assert payload["toolbox_id"] == "remote-ref"
    assert dict(payload["host"])["kind"] == "control_channel"
    assert isinstance(restored.host, EngineHostControlChannel)
    assert restored.toolbox_id == "remote-ref"
    assert restored.python_executable == "python-demo"


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
        assert isinstance(restored.host, EngineHostService)
        assert restored.toolbox_id == "service-ref"
        assert str(restored.host.engines_state_file).endswith("managed_engines.json")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_sandboxed_toolbox_facade_register_python_callable_reads_module_source(tmp_path: Path) -> None:
    module_path = tmp_path / "facade_source_mod.py"
    module_path.write_text(
        "def facade_source(name='world'):\n"
        "    return {'greeting': f'hi {name}'}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_source_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _FakeHost:
        def __init__(self) -> None:
            self.request: dict | None = None

        def toolbox_register_auto(self, **kwargs):
            self.request = dict(kwargs)
            return {"status": "ok"}

    host = _FakeHost()
    facade = SandboxedToolboxFacade(toolbox_id="facade-source", host=host)

    _ = facade.register_python_callable(
        module.facade_source,
        required_imports=["requests"],
        sandbox_policy={"sandbox": {"enabled": True}},
    )

    assert host.request is not None
    req = list(host.request["requests"])[0]
    assert req["module_name"] == "facade_source_mod"
    assert req["callable_name"] == "facade_source"
    assert req["files"][0]["relative_path"] == "facade_source_mod.py"
    assert "def facade_source" in req["files"][0]["content"]


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
    facade = SandboxedToolboxFacade(toolbox_id="facade-box", host=host, python_executable="python.exe")
    callback_context = {"origin": "chat", "lock": threading.Lock()}

    out = facade.execute(
        tool_name="hello_auto",
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
            "toolbox_describe",
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
                desc = svc.toolbox_describe(engine_id="toolbox-live", timeout_seconds=2.0)
                break
            except Exception as exc:  # pragma: no cover - startup polling
                last_error = exc
                time.sleep(0.1)
        if desc is None:
            raise AssertionError(f"toolbox executor did not become ready: {last_error}")

        assert "hello_tool" in list(desc.get("all_registered_tool_names") or [])
        exec_out = svc.toolbox_execute(
            engine_id="toolbox-live",
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
                _ = svc.toolbox_describe(engine_id="toolbox-live-callback", timeout_seconds=2.0)
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


def test_hosted_toolbox_execute_routes_generic_callback_with_context() -> None:
    root = _scratch_dir("hosted-callback-context-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "callback_context_mod.py"
    module_path.write_text(
        "def callback_context_tool(name: str = 'world', **kwargs):\n"
        "    result = kwargs['callbacks'].invoke('echo_name', {'value': name})\n"
        "    return {'callback': result}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("callback_context_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-callback-context",
        host=svc,
        python_executable=sys.executable,
    )
    seen: List[Dict[str, Any]] = []

    def _processor(*, callback_name: str, payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
        seen.append(
            {
                "callback_name": callback_name,
                "payload": dict(payload or {}),
                "toolbox_id": context.toolbox_id,
                "tool_name": context.tool_name,
                "tool_call_id": context.tool_call_id,
                "tool_arguments": dict(context.tool_arguments or {}),
                "user_context": context.user_context,
            }
        )
        return {
            "echo": dict(payload or {}).get("value"),
            "tool_name": context.tool_name,
            "tool_call_id": context.tool_call_id,
            "user_context": context.user_context,
        }

    try:
        created = facade.register_python_callable(
            module.callback_context_tool,
            callback_signature={"callbacks": [{"name": "echo_name", "payload_type": "object"}]},
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        described = facade.describe()
        assert dict(described.get("tool_metadata") or {}).get("callback_context_tool") == {
            "callback_signature": {"callbacks": [{"name": "echo_name", "payload_type": "object"}]},
            "non_restartable": False,
            "hidden": False,
        }

        out = facade.execute(
            tool_name="callback_context_tool",
            arguments={"name": "Sam"},
            callback_processor=_processor,
            callback_context={"origin": "test"},
            tool_call_id="call-ctx-1",
        )
        tool_row = dict(out.get("tool_call") or {})
        parsed = json.loads(str(tool_row.get("result") or "{}"))
        assert parsed["callback"]["echo"] == "Sam"
        assert parsed["callback"]["tool_call_id"] == "call-ctx-1"
        assert parsed["callback"]["user_context"] == {"origin": "test"}
        assert seen == [
            {
                "callback_name": "echo_name",
                "payload": {"value": "Sam"},
                "toolbox_id": "toolbox-callback-context",
                "tool_name": "callback_context_tool",
                "tool_call_id": "call-ctx-1",
                "tool_arguments": {"name": "Sam"},
                "user_context": {"origin": "test"},
            }
        ]
    finally:
        removed = facade.unregister_auto_callable(module_name="callback_context_mod", callable_name="callback_context_tool")
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-callback-context"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
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


def test_hosted_toolbox_callbacks_run_concurrently() -> None:
    root = _scratch_dir("hosted-callback-concurrency-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "callback_parallel_mod.py"
    module_path.write_text(
        "import concurrent.futures\n"
        "def callback_parallel_tool(**kwargs):\n"
        "    callbacks = kwargs['callbacks']\n"
        "    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:\n"
        "        fut_a = pool.submit(callbacks.invoke, 'slow', {'value': 'a'})\n"
        "        fut_b = pool.submit(callbacks.invoke, 'slow', {'value': 'b'})\n"
        "        return {'results': [fut_a.result(), fut_b.result()]}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("callback_parallel_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-callback-concurrency",
        host=svc,
        python_executable=sys.executable,
    )

    def _processor(*, callback_name: str, payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
        time.sleep(0.20)
        return {
            "callback_name": callback_name,
            "value": dict(payload or {}).get("value"),
            "tool_call_id": context.tool_call_id,
        }

    try:
        created = facade.register_python_callable(
            module.callback_parallel_tool,
            callback_signature={"callbacks": [{"name": "slow", "payload_type": "object"}]},
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])

        started = time.monotonic()
        out = facade.execute(
            tool_name="callback_parallel_tool",
            arguments={},
            callback_processor=_processor,
            tool_call_id="call-par-1",
        )
        elapsed = time.monotonic() - started
        tool_row = dict(out.get("tool_call") or {})
        parsed = json.loads(str(tool_row.get("result") or "{}"))
        values = sorted(item["value"] for item in list(parsed.get("results") or []))
        assert values == ["a", "b"]
        assert elapsed < 0.38
    finally:
        removed = facade.unregister_auto_callable(module_name="callback_parallel_mod", callable_name="callback_parallel_tool")
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-callback-concurrency"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


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
                desc = svc.toolbox_describe(engine_id="toolbox-live-intrinsic", timeout_seconds=2.0)
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
                desc = svc.toolbox_describe(engine_id="toolbox-live-auto", timeout_seconds=2.0)
                break
            except Exception as exc:  # pragma: no cover - startup polling
                last_error = exc
                time.sleep(0.1)
        if desc is None:
            raise AssertionError(f"toolbox executor did not become ready: {last_error}")

        assert "hello_auto" in list(desc.get("all_registered_tool_names") or [])
        exec_out = svc.toolbox_execute(
            engine_id="toolbox-live-auto",
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


def test_toolbox_service_routes_calls_across_multiple_sandbox_profiles() -> None:
    root = _scratch_dir("live-routed-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    stager = ToolboxBundleStager(root)

    alpha = stager.stage_bundle(
        ToolboxBundleSpec(
            bundle_id="bundle-alpha-routed",
            toolbox_id="toolbox-routed",
            sandbox_profile=SandboxProfileSpec(profile_id="fs-only"),
            files=[
                ToolboxBundleFile(
                    relative_path="alpha_tools.py",
                    content=(
                        "def alpha_tool(name: str = 'world'):\n"
                        "    \"\"\"Alpha tool.\n\n"
                        "    Args:\n"
                        "        name (str): Name input.\n"
                        "    \"\"\"\n"
                        "    return {'tool': 'alpha', 'name': name}\n"
                    ),
                )
            ],
            auto_tools=[ToolboxBundleAutoTool(module_name="alpha_tools", callable_name="alpha_tool")],
        )
    )
    beta = stager.stage_bundle(
        ToolboxBundleSpec(
            bundle_id="bundle-beta-routed",
            toolbox_id="toolbox-routed",
            sandbox_profile=SandboxProfileSpec(profile_id="net-open"),
            files=[
                ToolboxBundleFile(
                    relative_path="beta_tools.py",
                    content=(
                        "def beta_tool(name: str = 'world'):\n"
                        "    \"\"\"Beta tool.\n\n"
                        "    Args:\n"
                        "        name (str): Name input.\n"
                        "    \"\"\"\n"
                        "    return {'tool': 'beta', 'name': name}\n"
                    ),
                )
            ],
            auto_tools=[ToolboxBundleAutoTool(module_name="beta_tools", callable_name="beta_tool")],
        )
    )

    reg_alpha = svc.spawn(
        engine_id="toolbox-routed-alpha",
        command=alpha.worker_command(python_executable=sys.executable),
        env=alpha.worker_env(),
        worker_profile_class="generic",
        executor_kind="toolbox_executor",
        bundle=alpha.registration_bundle(),
        environment=alpha.registration_environment(),
        tool_access=alpha.registration_tool_access(),
        capabilities={"brokered_filesystem": False, "brokered_http": False, "dynamic_reload": False},
    )
    reg_beta = svc.spawn(
        engine_id="toolbox-routed-beta",
        command=beta.worker_command(python_executable=sys.executable),
        env=beta.worker_env(),
        worker_profile_class="generic",
        executor_kind="toolbox_executor",
        bundle=beta.registration_bundle(),
        environment=beta.registration_environment(),
        tool_access=beta.registration_tool_access(),
        capabilities={"brokered_filesystem": False, "brokered_http": False, "dynamic_reload": False},
    )
    try:
        deadline = time.time() + 8.0
        last_error: Exception | None = None
        desc = None
        while time.time() < deadline:
            try:
                _ = svc.toolbox_describe(engine_id="toolbox-routed-alpha", timeout_seconds=2.0)
                _ = svc.toolbox_describe(engine_id="toolbox-routed-beta", timeout_seconds=2.0)
                desc = svc.toolbox_describe(toolbox_id="toolbox-routed", timeout_seconds=2.0)
                break
            except Exception as exc:  # pragma: no cover - startup polling
                last_error = exc
                time.sleep(0.1)
        if desc is None:
            raise AssertionError(f"toolbox routing did not become ready: {last_error}")

        assert sorted(list(desc.get("all_registered_tool_names") or [])) == ["alpha_tool", "beta_tool"]
        assert sorted(list(desc.get("sandbox_profile_ids") or [])) == ["fs-only", "net-open"]

        out_alpha = svc.toolbox_execute(
            toolbox_id="toolbox-routed",
            tool_call={"name": "alpha_tool", "arguments": {"name": "A"}},
            timeout_seconds=5.0,
        )
        row_alpha = dict(out_alpha.get("tool_call") or {})
        assert '"tool": "alpha"' in str(row_alpha.get("result") or "")
        assert out_alpha["engine_id"] == "toolbox-routed-alpha"

        out_beta = svc.toolbox_execute(
            toolbox_id="toolbox-routed",
            tool_call={"name": "beta_tool", "arguments": {"name": "B"}},
            timeout_seconds=5.0,
        )
        row_beta = dict(out_beta.get("tool_call") or {})
        assert '"tool": "beta"' in str(row_beta.get("result") or "")
        assert out_beta["engine_id"] == "toolbox-routed-beta"
    finally:
        for reg, eid in ((reg_alpha, "toolbox-routed-alpha"), (reg_beta, "toolbox-routed-beta")):
            try:
                os.kill(int(reg.get("pid") or 0), signal.SIGTERM)
                time.sleep(0.2)
            except Exception:
                pass
            _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_sandbox_orchestrator_spawns_and_routes_multi_profile_toolbox() -> None:
    root = _scratch_dir("orchestrator-live-")
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
            files=[
                ToolboxBundleFile(
                    relative_path="alpha_assign.py",
                    content=(
                        "def alpha_assign(name: str = 'world'):\n"
                        "    \"\"\"Alpha assign.\n\n"
                        "    Args:\n"
                        "        name (str): Name input.\n"
                        "    \"\"\"\n"
                        "    return {'tool': 'alpha_assign', 'name': name}\n"
                    ),
                )
            ],
            module_name="alpha_assign",
            callable_name="alpha_assign",
            sandbox_profile=SandboxProfileSpec(
                required_imports=["requests"],
                sandbox_policy={"sandbox": {"enabled": True, "brokered_io": {"http": True}}},
            ),
        ),
        ToolboxAutoAssignmentRequest(
            files=[
                ToolboxBundleFile(
                    relative_path="beta_assign.py",
                    content=(
                        "def beta_assign(name: str = 'world'):\n"
                        "    \"\"\"Beta assign.\n\n"
                        "    Args:\n"
                        "        name (str): Name input.\n"
                        "    \"\"\"\n"
                        "    return {'tool': 'beta_assign', 'name': name}\n"
                    ),
                )
            ],
            module_name="beta_assign",
            callable_name="beta_assign",
            sandbox_profile=SandboxProfileSpec(
                profile_id="math-only",
                required_imports=["sympy"],
                sandbox_policy={"sandbox": {"enabled": True}},
            ),
        ),
    ]

    assignments = orchestrator.spawn_assignments(toolbox_id="toolbox-assigned", requests=requests)
    try:
        assert len(assignments) == 2
        deadline = time.time() + 8.0
        last_error: Exception | None = None
        desc = None
        engine_ids = [str(dict(item.registration or {}).get("engine_id") or "") for item in assignments]
        while time.time() < deadline:
            try:
                for eid in engine_ids:
                    if eid:
                        _ = svc.toolbox_describe(engine_id=eid, timeout_seconds=2.0)
                desc = svc.toolbox_describe(toolbox_id="toolbox-assigned", timeout_seconds=2.0)
                if sorted(list(desc.get("all_registered_tool_names") or [])) == ["alpha_assign", "beta_assign"]:
                    break
            except Exception as exc:  # pragma: no cover - startup polling
                last_error = exc
            time.sleep(0.1)
        if desc is None:
            raise AssertionError(f"assigned toolbox did not become ready: {last_error}")

        out_alpha = svc.toolbox_execute(
            toolbox_id="toolbox-assigned",
            tool_call={"name": "alpha_assign", "arguments": {"name": "A"}},
            timeout_seconds=5.0,
        )
        assert '"tool": "alpha_assign"' in str(dict(out_alpha.get("tool_call") or {}).get("result") or "")

        out_beta = svc.toolbox_execute(
            toolbox_id="toolbox-assigned",
            tool_call={"name": "beta_assign", "arguments": {"name": "B"}},
            timeout_seconds=5.0,
        )
        assert '"tool": "beta_assign"' in str(dict(out_beta.get("tool_call") or {}).get("result") or "")
    finally:
        for item in assignments:
            reg = dict(item.registration or {})
            eid = str(reg.get("engine_id") or "")
            if reg:
                try:
                    os.kill(int(reg.get("pid") or 0), signal.SIGTERM)
                    time.sleep(0.2)
                except Exception:
                    pass
            if eid:
                _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_register_auto_persists_membership_and_replaces_profile_executor() -> None:
    root = _scratch_dir("register-auto-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    try:
        first = svc.toolbox_register_auto(
            toolbox_id="toolbox-persist",
            requests=[
                {
                    "files": [
                        {
                            "relative_path": "persist_alpha.py",
                            "content": (
                                "def persist_alpha(name: str = 'world'):\n"
                                "    \"\"\"Persist alpha.\n\n"
                                "    Args:\n"
                                "        name (str): Name input.\n"
                                "    \"\"\"\n"
                                "    return {'tool': 'persist_alpha', 'name': name}\n"
                            ),
                        }
                    ],
                    "module_name": "persist_alpha",
                    "callable_name": "persist_alpha",
                    "sandbox_profile": {
                        "required_imports": ["requests"],
                        "sandbox_policy": {"sandbox": {"enabled": True}},
                    },
                }
            ],
            python_executable=sys.executable,
        )
        first_engine_ids = list(first.get("spawned_engine_ids") or [])
        assert len(first_engine_ids) == 1

        second = svc.toolbox_register_auto(
            toolbox_id="toolbox-persist",
            requests=[
                {
                    "files": [
                        {
                            "relative_path": "persist_beta.py",
                            "content": (
                                "def persist_beta(name: str = 'world'):\n"
                                "    \"\"\"Persist beta.\n\n"
                                "    Args:\n"
                                "        name (str): Name input.\n"
                                "    \"\"\"\n"
                                "    return {'tool': 'persist_beta', 'name': name}\n"
                            ),
                        }
                    ],
                    "module_name": "persist_beta",
                    "callable_name": "persist_beta",
                    "sandbox_profile": {
                        "required_imports": ["requests"],
                        "sandbox_policy": {"sandbox": {"enabled": True}},
                    },
                }
            ],
            python_executable=sys.executable,
        )
        second_engine_ids = list(second.get("spawned_engine_ids") or [])
        assert len(second_engine_ids) == 1
        assert second_engine_ids[0] != first_engine_ids[0]
        assert first_engine_ids[0] in list(second.get("replaced_engine_ids") or [])
        assert list(second.get("ready_engine_ids") or []) == second_engine_ids
        assert dict(second.get("rollout") or {}).get(second_engine_ids[0], {}).get("ready") is True

        deadline = time.time() + 8.0
        last_error: Exception | None = None
        desc = None
        while time.time() < deadline:
            try:
                _ = svc.toolbox_describe(engine_id=second_engine_ids[0], timeout_seconds=2.0)
                desc = svc.toolbox_describe(toolbox_id="toolbox-persist", timeout_seconds=2.0)
                if sorted(list(desc.get("all_registered_tool_names") or [])) == ["persist_alpha", "persist_beta"]:
                    break
            except Exception as exc:  # pragma: no cover - startup polling
                last_error = exc
            time.sleep(0.1)
        if desc is None:
            raise AssertionError(f"persistent toolbox did not become ready: {last_error}")

        out_alpha = svc.toolbox_execute(
            toolbox_id="toolbox-persist",
            tool_call={"name": "persist_alpha", "arguments": {"name": "A"}},
            timeout_seconds=5.0,
        )
        assert '"tool": "persist_alpha"' in str(dict(out_alpha.get("tool_call") or {}).get("result") or "")

        out_beta = svc.toolbox_execute(
            toolbox_id="toolbox-persist",
            tool_call={"name": "persist_beta", "arguments": {"name": "B"}},
            timeout_seconds=5.0,
        )
        assert '"tool": "persist_beta"' in str(dict(out_beta.get("tool_call") or {}).get("result") or "")

        state_path = root / "state" / "toolbox_sandboxes.json"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        toolbox_row = dict(dict(payload.get("toolboxes") or {}).get("toolbox-persist") or {})
        assert len(list(toolbox_row.get("requests") or [])) == 2
        profiles = dict(toolbox_row.get("profiles") or {})
        assert len(profiles) == 1
        only_profile = next(iter(profiles.values()))
        assert only_profile["engine_id"] == second_engine_ids[0]
        assert dict(only_profile.get("rollout") or {}).get("ready") is True
        assert int(dict(only_profile.get("rollout") or {}).get("warmup_ms") or 0) >= 0
        history = list(only_profile.get("rollout_history") or [])
        assert len(history) == 2
        assert history[0]["action"] == "register_auto"
        assert history[1]["action"] == "register_auto"
        assert history[1]["engine_id"] == second_engine_ids[0]
        assert history[1]["replaced_engine_id"] == first_engine_ids[0]
        env_row = dict(only_profile.get("environment") or {})
        assert str(env_row.get("venv_key") or "").strip()
        assert Path(str(env_row.get("venv_path") or "")).exists()
        reg = dict(svc._find_registration(second_engine_ids[0]) or {})
        command = list(reg.get("command") or [])
        assert command
        assert Path(str(command[0] or "")).resolve() == Path(str(env_row.get("python_executable") or "")).resolve()
    finally:
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-persist"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_sandboxed_toolbox_facade_runs_end_to_end_against_host_service() -> None:
    root = _scratch_dir("facade-live-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-live",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        created = facade.register_auto_callable(
            relative_path="facade_live.py",
            content=(
                "def facade_live(name: str = 'world'):\n"
                "    \"\"\"Facade live tool.\n\n"
                "    Args:\n"
                "        name (str): Name input.\n"
                "    \"\"\"\n"
                "    return {'tool': 'facade_live', 'name': name}\n"
            ),
            module_name="facade_live",
            callable_name="facade_live",
            required_imports=["requests"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])

        desc = facade.describe(timeout_seconds=5.0)
        assert list(desc.get("all_registered_tool_names") or []) == ["facade_live"]

        out = facade.execute(tool_name="facade_live", arguments={"name": "Z"}, timeout_seconds=5.0)
        assert '"tool": "facade_live"' in str(dict(out.get("tool_call") or {}).get("result") or "")

        removed = facade.unregister_auto_callable(module_name="facade_live", callable_name="facade_live")
        assert removed["toolbox_removed"] is True
        assert list(removed.get("ready_engine_ids") or []) == []
    finally:
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-live"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_sandboxed_toolbox_facade_register_python_callable_runs_end_to_end() -> None:
    root = _scratch_dir("facade-pycall-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_pycall_mod.py"
    module_path.write_text(
        "def facade_pycall(name: str = 'world'):\n"
        "    \"\"\"Facade Python callable.\n\n"
        "    Args:\n"
        "        name (str): Name input.\n"
        "    \"\"\"\n"
        "    return {'tool': 'facade_pycall', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_pycall_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-pycall",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        created = facade.register_python_callable(
            module.facade_pycall,
            required_imports=["requests"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])

        desc = facade.describe(timeout_seconds=5.0)
        assert list(desc.get("all_registered_tool_names") or []) == ["facade_pycall"]

        out = facade.execute(tool_name="facade_pycall", arguments={"name": "Q"}, timeout_seconds=5.0)
        assert '"tool": "facade_pycall"' in str(dict(out.get("tool_call") or {}).get("result") or "")
    finally:
        removed = facade.unregister_auto_callable(module_name="facade_pycall_mod", callable_name="facade_pycall")
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-pycall"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_sandboxed_toolbox_facade_register_intrinsic_tools_runs_end_to_end() -> None:
    root = _scratch_dir("facade-intrinsic-live-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-intrinsic",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        created = facade.register_intrinsic_tools(
            ["symbolic_algebra"],
            include_guides=True,
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])

        desc = facade.describe(timeout_seconds=5.0)
        assert "symbolic_algebra" in list(desc.get("all_registered_tool_names") or [])
        assert "symbolic_algebra_guide" in list(desc.get("all_registered_tool_names") or [])

        out = facade.execute(
            tool_name="symbolic_algebra",
            arguments={"expr": "2 + 3", "variables": [], "operation": "simplify"},
            timeout_seconds=5.0,
        )
        assert '"5"' in str(dict(out.get("tool_call") or {}).get("result") or "") or "5" in str(dict(out.get("tool_call") or {}).get("result") or "")

        removed = facade.unregister_intrinsic_tools(["symbolic_algebra"], include_guides=True)
        assert removed["toolbox_removed"] is True
    finally:
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-intrinsic"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_sandboxed_toolbox_facade_register_manual_tool_runs_end_to_end() -> None:
    root = _scratch_dir("facade-manual-live-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_manual_mod.py"
    module_path.write_text(
        "def facade_manual_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_manual', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_manual_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-manual",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        created = facade.register_manual_tool(
            _tool_definition("facade_manual_tool"),
            module.facade_manual_impl,
            required_imports=["requests"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])

        desc = facade.describe(timeout_seconds=5.0)
        assert list(desc.get("all_registered_tool_names") or []) == ["facade_manual_tool"]

        out = facade.execute(tool_name="facade_manual_tool", arguments={"name": "M"}, timeout_seconds=5.0)
        assert '"tool": "facade_manual"' in str(dict(out.get("tool_call") or {}).get("result") or "")

        removed = facade.unregister_manual_tool(module_name="facade_manual_mod", callable_name="facade_manual_impl")
        assert removed["toolbox_removed"] is True
    finally:
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-manual"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_sandboxed_toolbox_facade_environment_description_and_resolution_live() -> None:
    root = _scratch_dir("facade-env-live-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_mod.py"
    module_path.write_text(
        "def facade_env_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        upserted = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
        )
        assert upserted["environment_description"]["name"] == "math-env"

        created = facade.register_python_callable(
            module.facade_env_impl,
            environment_name="math-env",
            required_imports=["numpy", "sympy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        created_profiles = dict(created.get("profiles") or {})
        created_profile = next(iter(created_profiles.values()))
        initial_engine_id = str(created_profile.get("engine_id") or "")
        initial_env_hash = str(dict(created_profile.get("environment") or {}).get("environment_description_hash") or "")
        assert initial_env_hash

        envs = facade.environment_descriptions()
        assert "math-env" in dict(envs.get("environment_descriptions") or {})

        resolved = facade.resolve_environment_requirements(
            environment_name="math-env",
            tool_keys=["facade_env_mod:facade_env_impl"],
        )
        assert resolved["required_packages"] == ["numpy", "sympy"]
        assert resolved["configured_extra_packages"] == ["numpy"]
        assert resolved["missing_packages"] == ["sympy"]

        updated_env = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy", "sympy"],
        )
        assert updated_env["environment_description"]["extra_packages"] == ["numpy", "sympy"]
        applied = facade.apply_environment_description(environment_name="math-env", toolbox_ids=["toolbox-facade-env"])
        assert applied["affected_toolbox_ids"] == ["toolbox-facade-env"]
        rebuilt = dict(applied.get("toolboxes") or {}).get("toolbox-facade-env") or {}
        assert list(rebuilt.get("ready_engine_ids") or [])
        rebuilt_profiles = dict(rebuilt.get("profiles") or {})
        rebuilt_profile = next(iter(rebuilt_profiles.values()))
        rebuilt_engine_id = str(rebuilt_profile.get("engine_id") or "")
        rebuilt_env_hash = str(dict(rebuilt_profile.get("environment") or {}).get("environment_description_hash") or "")
        assert rebuilt_engine_id == initial_engine_id
        assert rebuilt_env_hash and rebuilt_env_hash != initial_env_hash

        resolved_after_apply = facade.resolve_environment_requirements(
            environment_name="math-env",
            tool_keys=["facade_env_mod:facade_env_impl"],
        )
        assert resolved_after_apply["missing_packages"] == []

        cloned = facade.clone_environment_description(
            source_name="math-env",
            target_name="math-env-v2",
            extra_packages=["numpy", "sympy"],
        )
        assert cloned["environment_description"]["base_env_name"] == "math-env"

        updated = facade.register_python_callable(
            module.facade_env_impl,
            environment_name="math-env-v2",
            required_imports=["numpy", "sympy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(updated.get("ready_engine_ids") or [])
        profiles = dict(updated.get("profiles") or {})
        profile = next(iter(profiles.values()))
        env_row = dict(profile.get("environment") or {})
        assert env_row["environment_name"] == "math-env-v2"
        assert str(env_row.get("environment_description_hash") or "").strip()
    finally:
        removed = facade.unregister_auto_callable(module_name="facade_env_mod", callable_name="facade_env_impl")
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_apply_rebuilds_toolbox_using_derived_environment_when_base_changes() -> None:
    root = _scratch_dir("facade-env-derived-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_derived_mod.py"
    module_path.write_text(
        "def facade_env_derived_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_derived', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_derived_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-derived",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="shared-base",
            base_env_name="base",
            extra_packages=["numpy"],
        )
        _ = facade.upsert_environment_description(
            name="math-derived",
            base_env_name="shared-base",
            extra_packages=["sympy"],
        )
        created = facade.register_python_callable(
            module.facade_env_derived_impl,
            environment_name="math-derived",
            required_imports=["numpy", "sympy", "pandas"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        created_profiles = dict(created.get("profiles") or {})
        created_profile = next(iter(created_profiles.values()))
        initial_hash = str(dict(created_profile.get("environment") or {}).get("environment_description_hash") or "")
        assert initial_hash

        before = facade.resolve_environment_requirements(
            environment_name="math-derived",
            tool_keys=["facade_env_derived_mod:facade_env_derived_impl"],
        )
        assert before["effective_extra_packages"] == ["sympy", "numpy"]
        assert before["missing_packages"] == ["pandas"]
        assert before["environment_lineage"] == ["math-derived", "shared-base", "base"]

        _ = facade.upsert_environment_description(
            name="shared-base",
            base_env_name="base",
            extra_packages=["numpy", "pandas"],
        )
        applied = facade.apply_environment_description(
            environment_name="shared-base",
            toolbox_ids=["toolbox-facade-env-derived"],
        )
        assert applied["affected_toolbox_ids"] == ["toolbox-facade-env-derived"]
        rebuilt = dict(applied.get("toolboxes") or {}).get("toolbox-facade-env-derived") or {}
        rebuilt_profiles = dict(rebuilt.get("profiles") or {})
        rebuilt_profile = next(iter(rebuilt_profiles.values()))
        rebuilt_hash = str(dict(rebuilt_profile.get("environment") or {}).get("environment_description_hash") or "")
        assert rebuilt_hash and rebuilt_hash != initial_hash

        after = facade.resolve_environment_requirements(
            environment_name="math-derived",
            tool_keys=["facade_env_derived_mod:facade_env_derived_impl"],
        )
        assert after["effective_extra_packages"] == ["sympy", "numpy", "pandas"]
        assert after["missing_packages"] == []
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_derived_mod",
            callable_name="facade_env_derived_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-derived"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_realize_writes_metadata_plan_for_linked_profile() -> None:
    root = _scratch_dir("facade-env-realize-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_realize_mod.py"
    module_path.write_text(
        "def facade_env_realize_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_realize', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_realize_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-realize",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
            allow_online_install=False,
        )
        created = facade.register_python_callable(
            module.facade_env_realize_impl,
            environment_name="math-env",
            required_imports=["numpy", "sympy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        realized = facade.realize_environment(
            environment_name="math-env",
            tool_keys=["facade_env_realize_mod:facade_env_realize_impl"],
        )
        profiles = dict(realized.get("profiles") or {})
        assert profiles
        profile_row = next(iter(profiles.values()))
        env_row = dict(profile_row.get("environment") or {})
        realization = dict(env_row.get("realization") or {})
        assert realization["mode"] == "metadata_only"
        assert realization["required_packages"] == ["numpy", "sympy"]
        assert realization["effective_extra_packages"] == ["numpy"]
        assert realization["planned_packages"] == ["numpy", "sympy"]
        assert realization["missing_packages"] == ["sympy"]
        assert realization["allow_online_install"] is False
        assert str(realization.get("provenance_hash") or "").strip()

        env_path = Path(str(env_row.get("venv_path") or "")).expanduser().resolve()
        metadata = json.loads((env_path / "environment.json").read_text(encoding="utf-8"))
        assert dict(metadata.get("realization") or {}).get("provenance_hash") == realization["provenance_hash"]

        state = json.loads((root / "state" / "toolbox_sandboxes.json").read_text(encoding="utf-8"))
        persisted = dict(dict(dict(state.get("toolboxes") or {}).get("toolbox-facade-env-realize") or {}).get("profiles") or {})
        persisted_profile = next(iter(persisted.values()))
        assert dict(dict(persisted_profile.get("environment") or {}).get("realization") or {}).get("planned_packages") == ["numpy", "sympy"]
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_realize_mod",
            callable_name="facade_env_realize_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-realize"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_sync_description_updates_existing_env_with_missing_packages() -> None:
    root = _scratch_dir("facade-env-sync-update-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_sync_update_mod.py"
    module_path.write_text(
        "def facade_env_sync_update_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_sync_update', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_sync_update_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-sync-update",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
        )
        created = facade.register_python_callable(
            module.facade_env_sync_update_impl,
            environment_name="math-env",
            required_imports=["numpy", "sympy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])

        synced = facade.sync_environment_description(
            source_environment_name="math-env",
            tool_keys=["facade_env_sync_update_mod:facade_env_sync_update_impl"],
        )
        assert synced["source_environment_name"] == "math-env"
        assert synced["target_environment_name"] == "math-env"
        assert synced["updated_direct_extra_packages"] == ["numpy", "sympy"]
        assert dict(synced.get("environment_description") or {}).get("extra_packages") == ["numpy", "sympy"]
        assert dict(synced.get("resolved") or {}).get("missing_packages") == ["sympy"]

        envs = facade.environment_descriptions()
        assert dict(dict(envs.get("environment_descriptions") or {}).get("math-env") or {}).get("extra_packages") == ["numpy", "sympy"]
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_sync_update_mod",
            callable_name="facade_env_sync_update_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-sync-update"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_sync_description_can_clone_apply_and_realize() -> None:
    root = _scratch_dir("facade-env-sync-clone-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_sync_clone_mod.py"
    module_path.write_text(
        "def facade_env_sync_clone_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_sync_clone', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_sync_clone_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-sync-clone",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
        )
        created = facade.register_python_callable(
            module.facade_env_sync_clone_impl,
            environment_name="math-env",
            required_imports=["numpy", "sympy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])

        cloned = facade.sync_environment_description(
            source_environment_name="math-env",
            target_environment_name="math-env-v2",
            tool_keys=["facade_env_sync_clone_mod:facade_env_sync_clone_impl"],
            apply=True,
            realize=True,
        )
        assert cloned["target_environment_name"] == "math-env-v2"
        assert dict(cloned.get("environment_description") or {}).get("base_env_name") == "math-env"
        assert dict(cloned.get("environment_description") or {}).get("extra_packages") == ["numpy", "sympy"]
        apply_result = dict(cloned.get("apply_result") or {})
        assert apply_result["affected_toolbox_ids"] == []
        realize_result = dict(cloned.get("realize_result") or {})
        assert realize_result["profiles"] == {}

        envs = facade.environment_descriptions()
        assert "math-env-v2" in dict(envs.get("environment_descriptions") or {})
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_sync_clone_mod",
            callable_name="facade_env_sync_clone_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-sync-clone"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_prepare_install_writes_requirements_and_plan() -> None:
    root = _scratch_dir("facade-env-install-plan-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_install_plan_mod.py"
    module_path.write_text(
        "def facade_env_install_plan_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_install_plan', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_install_plan_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-install-plan",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
            allow_online_install=True,
        )
        created = facade.register_python_callable(
            module.facade_env_install_plan_impl,
            environment_name="math-env",
            required_imports=["numpy", "sympy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])

        planned = facade.prepare_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_plan_mod:facade_env_install_plan_impl"],
        )
        profiles = dict(planned.get("profiles") or {})
        assert profiles
        profile_row = next(iter(profiles.values()))
        env_row = dict(profile_row.get("environment") or {})
        install_plan = dict(env_row.get("install_plan") or {})
        assert install_plan["mode"] == "plan_only"
        assert install_plan["planned_packages"] == ["numpy", "sympy"]
        assert install_plan["missing_packages"] == ["sympy"]
        assert install_plan["can_execute_online"] is True
        assert install_plan["install_command"][1:4] == ["-m", "pip", "install"]

        env_path = Path(str(env_row.get("venv_path") or "")).expanduser().resolve()
        requirements_path = env_path / "requirements-planned.txt"
        assert requirements_path.exists()
        assert requirements_path.read_text(encoding="utf-8").splitlines() == ["numpy", "sympy"]

        metadata = json.loads((env_path / "environment.json").read_text(encoding="utf-8"))
        assert dict(metadata.get("install_plan") or {}).get("requirements_relpath") == "requirements-planned.txt"

        state = json.loads((root / "state" / "toolbox_sandboxes.json").read_text(encoding="utf-8"))
        persisted = dict(dict(dict(state.get("toolboxes") or {}).get("toolbox-facade-env-install-plan") or {}).get("profiles") or {})
        persisted_profile = next(iter(persisted.values()))
        assert dict(dict(persisted_profile.get("environment") or {}).get("install_plan") or {}).get("planned_packages") == ["numpy", "sympy"]
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_install_plan_mod",
            callable_name="facade_env_install_plan_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-install-plan"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_execute_install_records_blocked_when_not_enabled() -> None:
    root = _scratch_dir("facade-env-install-blocked-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_install_blocked_mod.py"
    module_path.write_text(
        "def facade_env_install_blocked_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_install_blocked', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_install_blocked_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-install-blocked",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
            allow_online_install=True,
        )
        created = facade.register_python_callable(
            module.facade_env_install_blocked_impl,
            environment_name="math-env",
            required_imports=["numpy", "sympy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        _ = facade.prepare_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_blocked_mod:facade_env_install_blocked_impl"],
        )
        executed = facade.execute_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_blocked_mod:facade_env_install_blocked_impl"],
            allow_execution=False,
        )
        profiles = dict(executed.get("profiles") or {})
        profile_row = next(iter(profiles.values()))
        execution = dict(dict(profile_row.get("environment") or {}).get("install_execution") or {})
        assert execution["status"] == "blocked"
        assert execution["reason"] == "execution_not_enabled"
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_install_blocked_mod",
            callable_name="facade_env_install_blocked_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-install-blocked"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_resolve_install_lock_requires_locked_plan() -> None:
    root = _scratch_dir("facade-env-install-resolve-needs-lock-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_install_resolve_needs_lock_mod.py"
    module_path.write_text(
        "def facade_env_install_resolve_needs_lock_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_install_resolve_needs_lock', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_install_resolve_needs_lock_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-install-resolve-needs-lock",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
            allow_online_install=True,
        )
        created = facade.register_python_callable(
            module.facade_env_install_resolve_needs_lock_impl,
            environment_name="math-env",
            required_imports=["numpy", "sympy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        _ = facade.prepare_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_resolve_needs_lock_mod:facade_env_install_resolve_needs_lock_impl"],
        )
        resolved = facade.resolve_environment_install_lock(
            environment_name="math-env",
            tool_keys=["facade_env_install_resolve_needs_lock_mod:facade_env_install_resolve_needs_lock_impl"],
            allow_resolution=True,
        )
        profiles = dict(resolved.get("profiles") or {})
        profile_row = next(iter(profiles.values()))
        resolution = dict(dict(profile_row.get("environment") or {}).get("install_resolution") or {})
        assert resolution["status"] == "blocked"
        assert resolution["reason"] == "install_lock_required"
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_install_resolve_needs_lock_mod",
            callable_name="facade_env_install_resolve_needs_lock_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-install-resolve-needs-lock"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_execute_install_requires_locked_plan() -> None:
    root = _scratch_dir("facade-env-install-needs-lock-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_install_needs_lock_mod.py"
    module_path.write_text(
        "def facade_env_install_needs_lock_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_install_needs_lock', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_install_needs_lock_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-install-needs-lock",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
            allow_online_install=True,
        )
        created = facade.register_python_callable(
            module.facade_env_install_needs_lock_impl,
            environment_name="math-env",
            required_imports=["numpy", "sympy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        _ = facade.prepare_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_needs_lock_mod:facade_env_install_needs_lock_impl"],
        )
        executed = facade.execute_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_needs_lock_mod:facade_env_install_needs_lock_impl"],
            allow_execution=True,
        )
        profiles = dict(executed.get("profiles") or {})
        profile_row = next(iter(profiles.values()))
        execution = dict(dict(profile_row.get("environment") or {}).get("install_execution") or {})
        assert execution["status"] == "blocked"
        assert execution["reason"] == "install_lock_required"
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_install_needs_lock_mod",
            callable_name="facade_env_install_needs_lock_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-install-needs-lock"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_execute_install_records_simulated_success(monkeypatch: pytest.MonkeyPatch) -> None:
    root = _scratch_dir("facade-env-install-exec-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_install_exec_mod.py"
    module_path.write_text(
        "def facade_env_install_exec_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_install_exec', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_install_exec_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _FakeCompleted:
        def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    captured: dict[str, Any] = {}

    def _fake_run(*args, **kwargs):
        cmd = list(args[0] or [])
        captured.setdefault("commands", []).append(cmd)
        if "--report" in cmd:
            report_path = Path(cmd[cmd.index("--report") + 1])
            report_path.write_text(
                json.dumps(
                    {
                        "install": [
                            {"metadata": {"name": "numpy", "version": "1.0"}},
                            {"metadata": {"name": "sympy", "version": "2.0"}},
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return _FakeCompleted(stdout="resolved")
        if cmd[-1:] == ["freeze"]:
            return _FakeCompleted(stdout="numpy==1.0\nsympy==2.0\n")
        return _FakeCompleted(stdout="installed")

    monkeypatch.setattr("hosting.toolbox_harness.subprocess.run", _fake_run)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-install-exec",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
            allow_online_install=True,
        )
        created = facade.register_python_callable(
            module.facade_env_install_exec_impl,
            environment_name="math-env",
            required_imports=["numpy", "sympy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        _ = facade.prepare_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_exec_mod:facade_env_install_exec_impl"],
        )
        locked = facade.lock_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_exec_mod:facade_env_install_exec_impl"],
        )
        resolved = facade.resolve_environment_install_lock(
            environment_name="math-env",
            tool_keys=["facade_env_install_exec_mod:facade_env_install_exec_impl"],
            allow_resolution=True,
        )
        resolved_profiles = dict(resolved.get("profiles") or {})
        resolved_profile = next(iter(resolved_profiles.values()))
        resolved_lock = dict(dict(resolved_profile.get("environment") or {}).get("resolved_install_lock") or {})
        assert resolved_lock["resolved_packages"] == ["numpy==1.0", "sympy==2.0"]
        assert str(resolved_lock.get("resolved_lock_hash") or "").strip()
        locked_profiles = dict(locked.get("profiles") or {})
        locked_profile = next(iter(locked_profiles.values()))
        install_lock = dict(dict(locked_profile.get("environment") or {}).get("install_lock") or {})
        assert str(install_lock.get("install_lock_hash") or "").strip()
        verified = facade.verify_environment_install_lock(
            environment_name="math-env",
            tool_keys=["facade_env_install_exec_mod:facade_env_install_exec_impl"],
        )
        verified_profiles = dict(verified.get("profiles") or {})
        verified_profile = next(iter(verified_profiles.values()))
        verification = dict(dict(verified_profile.get("environment") or {}).get("install_lock_verification") or {})
        assert verification["status"] == "ok"
        executed = facade.execute_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_exec_mod:facade_env_install_exec_impl"],
            allow_execution=True,
        )
        profiles = dict(executed.get("profiles") or {})
        profile_row = next(iter(profiles.values()))
        execution = dict(dict(profile_row.get("environment") or {}).get("install_execution") or {})
        assert execution["status"] == "ok"
        assert execution["executed"] is True
        assert execution["returncode"] == 0
        assert execution["stdout"] == "installed"
        assert execution["install_lock_hash"] == install_lock["install_lock_hash"]
        assert execution["resolved_lock_hash"] == resolved_lock["resolved_lock_hash"]
        commands = list(captured.get("commands") or [])
        assert "--report" in commands[0]
        assert commands[1][-1].endswith("requirements-resolved.txt")
        assert commands[2][-1] == "freeze"
        receipt = dict(dict(profile_row.get("environment") or {}).get("install_receipt") or {})
        assert receipt["status"] == "ok"
        assert receipt["packages"] == ["numpy==1.0", "sympy==2.0"]
        assert str(receipt.get("packages_hash") or "").strip()
        execution_receipt_verification = dict(dict(profile_row.get("environment") or {}).get("install_receipt_verification") or {})
        assert execution_receipt_verification["status"] == "ok"
        assert execution_receipt_verification["missing_package_names"] == []
        assert execution_receipt_verification["lock_source"] == "resolved_install_lock"
        receipt_verified = facade.verify_environment_install_receipt(
            environment_name="math-env",
            tool_keys=["facade_env_install_exec_mod:facade_env_install_exec_impl"],
        )
        receipt_profiles = dict(receipt_verified.get("profiles") or {})
        receipt_profile = next(iter(receipt_profiles.values()))
        receipt_verification = dict(dict(receipt_profile.get("environment") or {}).get("install_receipt_verification") or {})
        assert receipt_verification["status"] == "ok"
        assert receipt_verification["missing_package_names"] == []
        assert receipt_verification["lock_source"] == "resolved_install_lock"

        env_path = Path(str(dict(profile_row.get("environment") or {}).get("venv_path") or "")).expanduser().resolve()
        metadata = json.loads((env_path / "environment.json").read_text(encoding="utf-8"))
        assert dict(metadata.get("install_execution") or {}).get("status") == "ok"
        assert dict(metadata.get("install_receipt") or {}).get("packages") == ["numpy==1.0", "sympy==2.0"]
        assert dict(metadata.get("install_receipt_verification") or {}).get("status") == "ok"
        assert dict(metadata.get("resolved_install_lock") or {}).get("resolved_packages") == ["numpy==1.0", "sympy==2.0"]
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_install_exec_mod",
            callable_name="facade_env_install_exec_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-install-exec"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_verify_lock_detects_stale_plan_and_blocks_execution() -> None:
    root = _scratch_dir("facade-env-install-stale-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_install_stale_mod.py"
    module_path.write_text(
        "def facade_env_install_stale_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_install_stale', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_install_stale_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-install-stale",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
            allow_online_install=True,
        )
        created = facade.register_python_callable(
            module.facade_env_install_stale_impl,
            environment_name="math-env",
            required_imports=["numpy", "sympy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        _ = facade.prepare_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_stale_mod:facade_env_install_stale_impl"],
        )
        locked = facade.lock_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_stale_mod:facade_env_install_stale_impl"],
        )
        profiles = dict(locked.get("profiles") or {})
        profile_row = next(iter(profiles.values()))
        env_path = Path(str(dict(profile_row.get("environment") or {}).get("venv_path") or "")).expanduser().resolve()
        metadata = json.loads((env_path / "environment.json").read_text(encoding="utf-8"))
        metadata["install_plan"]["planned_packages"] = ["numpy", "sympy", "pandas"]
        (env_path / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        verified = facade.verify_environment_install_lock(
            environment_name="math-env",
            tool_keys=["facade_env_install_stale_mod:facade_env_install_stale_impl"],
        )
        verified_profiles = dict(verified.get("profiles") or {})
        verified_profile = next(iter(verified_profiles.values()))
        verification = dict(dict(verified_profile.get("environment") or {}).get("install_lock_verification") or {})
        assert verification["status"] == "stale"
        assert verification["reason"] == "install_lock_hash_mismatch"

        executed = facade.execute_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_stale_mod:facade_env_install_stale_impl"],
            allow_execution=True,
        )
        executed_profiles = dict(executed.get("profiles") or {})
        executed_profile = next(iter(executed_profiles.values()))
        execution = dict(dict(executed_profile.get("environment") or {}).get("install_execution") or {})
        assert execution["status"] == "blocked"
        assert execution["reason"] == "install_lock_hash_mismatch"
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_install_stale_mod",
            callable_name="facade_env_install_stale_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-install-stale"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_verify_receipt_detects_missing_locked_package() -> None:
    root = _scratch_dir("facade-env-install-receipt-stale-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_install_receipt_stale_mod.py"
    module_path.write_text(
        "def facade_env_install_receipt_stale_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_install_receipt_stale', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_install_receipt_stale_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-install-receipt-stale",
        host=svc,
        python_executable=sys.executable,
    )
    captured: Dict[str, Any] = {}

    def _fake_run(command, **kwargs):
        captured.setdefault("commands", []).append(list(command))

        class _Result:
            def __init__(self, *, returncode: int, stdout: str = "", stderr: str = "") -> None:
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        if list(command)[-1] == "freeze":
            return _Result(returncode=0, stdout="numpy==1.0\nsympy==2.0\n")
        return _Result(returncode=0, stdout="installed\n")

    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
            allow_online_install=True,
        )
        created = facade.register_python_callable(
            module.facade_env_install_receipt_stale_impl,
            environment_name="math-env",
            required_imports=["numpy", "sympy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        _ = facade.prepare_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_receipt_stale_mod:facade_env_install_receipt_stale_impl"],
        )
        _ = facade.lock_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_receipt_stale_mod:facade_env_install_receipt_stale_impl"],
        )
        import importlib

        toolbox_harness_module = importlib.import_module("hosting.toolbox_harness")

        original_run = toolbox_harness_module.subprocess.run
        toolbox_harness_module.subprocess.run = _fake_run
        try:
            executed = facade.execute_environment_install(
                environment_name="math-env",
                tool_keys=["facade_env_install_receipt_stale_mod:facade_env_install_receipt_stale_impl"],
                allow_execution=True,
            )
        finally:
            toolbox_harness_module.subprocess.run = original_run

        profiles = dict(executed.get("profiles") or {})
        profile_row = next(iter(profiles.values()))
        env_path = Path(str(dict(profile_row.get("environment") or {}).get("venv_path") or "")).expanduser().resolve()
        metadata = json.loads((env_path / "environment.json").read_text(encoding="utf-8"))
        metadata["install_receipt"]["packages"] = ["numpy==1.0"]
        (env_path / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        verified = facade.verify_environment_install_receipt(
            environment_name="math-env",
            tool_keys=["facade_env_install_receipt_stale_mod:facade_env_install_receipt_stale_impl"],
        )
        verified_profiles = dict(verified.get("profiles") or {})
        verified_profile = next(iter(verified_profiles.values()))
        verification = dict(dict(verified_profile.get("environment") or {}).get("install_receipt_verification") or {})
        assert verification["status"] == "mismatch"
        assert verification["missing_package_names"] == ["sympy"]
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_install_receipt_stale_mod",
            callable_name="facade_env_install_receipt_stale_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-install-receipt-stale"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_verify_lock_detects_stale_resolved_lock_and_blocks_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _scratch_dir("facade-env-install-resolved-stale-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_install_resolved_stale_mod.py"
    module_path.write_text(
        "def facade_env_install_resolved_stale_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_install_resolved_stale', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_install_resolved_stale_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _FakeCompleted:
        def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(*args, **kwargs):
        cmd = list(args[0] or [])
        if "--report" in cmd:
            report_path = Path(cmd[cmd.index("--report") + 1])
            report_path.write_text(
                json.dumps(
                    {"install": [{"metadata": {"name": "numpy", "version": "1.0"}}]},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return _FakeCompleted(stdout="resolved")
        return _FakeCompleted(stdout="installed")

    monkeypatch.setattr("hosting.toolbox_harness.subprocess.run", _fake_run)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-install-resolved-stale",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
            allow_online_install=True,
        )
        created = facade.register_python_callable(
            module.facade_env_install_resolved_stale_impl,
            environment_name="math-env",
            required_imports=["numpy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        _ = facade.prepare_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_resolved_stale_mod:facade_env_install_resolved_stale_impl"],
        )
        _ = facade.lock_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_resolved_stale_mod:facade_env_install_resolved_stale_impl"],
        )
        resolved = facade.resolve_environment_install_lock(
            environment_name="math-env",
            tool_keys=["facade_env_install_resolved_stale_mod:facade_env_install_resolved_stale_impl"],
            allow_resolution=True,
        )
        resolved_profiles = dict(resolved.get("profiles") or {})
        resolved_profile = next(iter(resolved_profiles.values()))
        env_path = Path(str(dict(resolved_profile.get("environment") or {}).get("venv_path") or "")).expanduser().resolve()
        metadata = json.loads((env_path / "environment.json").read_text(encoding="utf-8"))
        metadata["resolved_install_lock"]["resolved_lock_hash"] = "stale-lock-hash"
        (env_path / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        verified = facade.verify_environment_install_lock(
            environment_name="math-env",
            tool_keys=["facade_env_install_resolved_stale_mod:facade_env_install_resolved_stale_impl"],
        )
        verified_profiles = dict(verified.get("profiles") or {})
        verified_profile = next(iter(verified_profiles.values()))
        verification = dict(dict(verified_profile.get("environment") or {}).get("install_lock_verification") or {})
        assert verification["status"] == "stale"
        assert verification["reason"] == "resolved_lock_hash_mismatch"
        assert verification["resolved_lock_status"] == "stale"
        assert verification["resolved_reason"] == "resolved_lock_hash_mismatch"

        executed = facade.execute_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_resolved_stale_mod:facade_env_install_resolved_stale_impl"],
            allow_execution=True,
        )
        executed_profiles = dict(executed.get("profiles") or {})
        executed_profile = next(iter(executed_profiles.values()))
        execution = dict(dict(executed_profile.get("environment") or {}).get("install_execution") or {})
        assert execution["status"] == "blocked"
        assert execution["reason"] == "resolved_lock_hash_mismatch"
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_install_resolved_stale_mod",
            callable_name="facade_env_install_resolved_stale_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-install-resolved-stale"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_verify_receipt_refuses_stale_resolved_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _scratch_dir("facade-env-install-receipt-lock-stale-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_install_receipt_lock_stale_mod.py"
    module_path.write_text(
        "def facade_env_install_receipt_lock_stale_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_install_receipt_lock_stale', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_install_receipt_lock_stale_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _FakeCompleted:
        def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(*args, **kwargs):
        cmd = list(args[0] or [])
        if "--report" in cmd:
            report_path = Path(cmd[cmd.index("--report") + 1])
            report_path.write_text(
                json.dumps(
                    {"install": [{"metadata": {"name": "numpy", "version": "1.0"}}]},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return _FakeCompleted(stdout="resolved")
        if cmd[-1:] == ["freeze"]:
            return _FakeCompleted(stdout="numpy==1.0\n")
        return _FakeCompleted(stdout="installed")

    monkeypatch.setattr("hosting.toolbox_harness.subprocess.run", _fake_run)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-install-receipt-lock-stale",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
            allow_online_install=True,
        )
        created = facade.register_python_callable(
            module.facade_env_install_receipt_lock_stale_impl,
            environment_name="math-env",
            required_imports=["numpy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        _ = facade.prepare_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_receipt_lock_stale_mod:facade_env_install_receipt_lock_stale_impl"],
        )
        _ = facade.lock_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_receipt_lock_stale_mod:facade_env_install_receipt_lock_stale_impl"],
        )
        _ = facade.resolve_environment_install_lock(
            environment_name="math-env",
            tool_keys=["facade_env_install_receipt_lock_stale_mod:facade_env_install_receipt_lock_stale_impl"],
            allow_resolution=True,
        )
        executed = facade.execute_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_receipt_lock_stale_mod:facade_env_install_receipt_lock_stale_impl"],
            allow_execution=True,
        )
        executed_profiles = dict(executed.get("profiles") or {})
        executed_profile = next(iter(executed_profiles.values()))
        env_path = Path(str(dict(executed_profile.get("environment") or {}).get("venv_path") or "")).expanduser().resolve()
        metadata = json.loads((env_path / "environment.json").read_text(encoding="utf-8"))
        metadata["resolved_install_lock"]["resolved_lock_hash"] = "stale-lock-hash"
        (env_path / "environment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        verified = facade.verify_environment_install_receipt(
            environment_name="math-env",
            tool_keys=["facade_env_install_receipt_lock_stale_mod:facade_env_install_receipt_lock_stale_impl"],
        )
        verified_profiles = dict(verified.get("profiles") or {})
        verified_profile = next(iter(verified_profiles.values()))
        env_row = dict(verified_profile.get("environment") or {})
        lock_verification = dict(env_row.get("install_lock_verification") or {})
        receipt_verification = dict(env_row.get("install_receipt_verification") or {})
        assert lock_verification["status"] == "stale"
        assert lock_verification["reason"] == "resolved_lock_hash_mismatch"
        assert receipt_verification["status"] == "stale"
        assert receipt_verification["reason"] == "resolved_lock_hash_mismatch"
        assert receipt_verification["lock_verification_status"] == "stale"
        assert receipt_verification["lock_source"] == "resolved_install_lock"
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_install_receipt_lock_stale_mod",
            callable_name="facade_env_install_receipt_lock_stale_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-install-receipt-lock-stale"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_environment_verify_lock_detects_tampered_resolution_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _scratch_dir("facade-env-install-report-stale-")
    module_dir = root / "callables"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_path = module_dir / "facade_env_install_report_stale_mod.py"
    module_path.write_text(
        "def facade_env_install_report_stale_impl(name: str = 'world'):\n"
        "    return {'tool': 'facade_env_install_report_stale', 'name': name}\n",
        encoding="utf-8",
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("facade_env_install_report_stale_mod", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _FakeCompleted:
        def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(*args, **kwargs):
        cmd = list(args[0] or [])
        if "--report" in cmd:
            report_path = Path(cmd[cmd.index("--report") + 1])
            report_path.write_text(
                json.dumps(
                    {"install": [{"metadata": {"name": "numpy", "version": "1.0"}}]},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return _FakeCompleted(stdout="resolved")
        return _FakeCompleted(stdout="installed")

    monkeypatch.setattr("hosting.toolbox_harness.subprocess.run", _fake_run)

    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    facade = SandboxedToolboxFacade(
        toolbox_id="toolbox-facade-env-install-report-stale",
        host=svc,
        python_executable=sys.executable,
    )
    try:
        _ = facade.upsert_environment_description(
            name="math-env",
            base_env_name="base",
            extra_packages=["numpy"],
            allow_online_install=True,
        )
        created = facade.register_python_callable(
            module.facade_env_install_report_stale_impl,
            environment_name="math-env",
            required_imports=["numpy"],
            sandbox_policy={"sandbox": {"enabled": True}},
        )
        assert list(created.get("ready_engine_ids") or [])
        _ = facade.prepare_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_report_stale_mod:facade_env_install_report_stale_impl"],
        )
        _ = facade.lock_environment_install(
            environment_name="math-env",
            tool_keys=["facade_env_install_report_stale_mod:facade_env_install_report_stale_impl"],
        )
        resolved = facade.resolve_environment_install_lock(
            environment_name="math-env",
            tool_keys=["facade_env_install_report_stale_mod:facade_env_install_report_stale_impl"],
            allow_resolution=True,
        )
        resolved_profiles = dict(resolved.get("profiles") or {})
        resolved_profile = next(iter(resolved_profiles.values()))
        env_path = Path(str(dict(resolved_profile.get("environment") or {}).get("venv_path") or "")).expanduser().resolve()
        metadata = json.loads((env_path / "environment.json").read_text(encoding="utf-8"))
        report_path = Path(str(dict(metadata.get("resolved_install_lock") or {}).get("report_path") or "")).expanduser().resolve()
        report_path.write_text(
            json.dumps({"install": [{"metadata": {"name": "numpy", "version": "9.9"}}]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        verified = facade.verify_environment_install_lock(
            environment_name="math-env",
            tool_keys=["facade_env_install_report_stale_mod:facade_env_install_report_stale_impl"],
        )
        verified_profiles = dict(verified.get("profiles") or {})
        verified_profile = next(iter(verified_profiles.values()))
        verification = dict(dict(verified_profile.get("environment") or {}).get("install_lock_verification") or {})
        assert verification["status"] == "stale"
        assert verification["reason"] == "resolved_lock_report_hash_mismatch"
        assert verification["resolved_reason"] == "resolved_lock_report_hash_mismatch"
        assert str(verification.get("resolved_report_sha256") or "").strip()
        assert str(verification.get("expected_resolved_report_sha256") or "").strip()
    finally:
        removed = facade.unregister_auto_callable(
            module_name="facade_env_install_report_stale_mod",
            callable_name="facade_env_install_report_stale_impl",
        )
        assert removed["toolbox_removed"] is True
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-facade-env-install-report-stale"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_unregister_auto_rebuilds_profile_and_removes_tool() -> None:
    root = _scratch_dir("unregister-auto-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    try:
        created = svc.toolbox_register_auto(
            toolbox_id="toolbox-unregister",
            requests=[
                {
                    "files": [
                        {
                            "relative_path": "keep_tool.py",
                            "content": (
                                "def keep_tool(name: str = 'world'):\n"
                                "    \"\"\"Keep tool.\n\n"
                                "    Args:\n"
                                "        name (str): Name input.\n"
                                "    \"\"\"\n"
                                "    return {'tool': 'keep_tool', 'name': name}\n"
                            ),
                        }
                    ],
                    "module_name": "keep_tool",
                    "callable_name": "keep_tool",
                    "sandbox_profile": {
                        "required_imports": ["requests"],
                        "sandbox_policy": {"sandbox": {"enabled": True}},
                    },
                },
                {
                    "files": [
                        {
                            "relative_path": "drop_tool.py",
                            "content": (
                                "def drop_tool(name: str = 'world'):\n"
                                "    \"\"\"Drop tool.\n\n"
                                "    Args:\n"
                                "        name (str): Name input.\n"
                                "    \"\"\"\n"
                                "    return {'tool': 'drop_tool', 'name': name}\n"
                            ),
                        }
                    ],
                    "module_name": "drop_tool",
                    "callable_name": "drop_tool",
                    "sandbox_profile": {
                        "required_imports": ["requests"],
                        "sandbox_policy": {"sandbox": {"enabled": True}},
                    },
                },
            ],
            python_executable=sys.executable,
        )
        old_engine_id = str(list(created.get("spawned_engine_ids") or [None])[0] or "")
        removed = svc.toolbox_unregister_auto(
            toolbox_id="toolbox-unregister",
            tool_keys=["drop_tool:drop_tool"],
            python_executable=sys.executable,
        )
        new_engine_id = str(list(removed.get("spawned_engine_ids") or [None])[0] or "")
        assert new_engine_id
        assert new_engine_id != old_engine_id
        assert old_engine_id in list(removed.get("replaced_engine_ids") or [])

        deadline = time.time() + 8.0
        last_error: Exception | None = None
        desc = None
        while time.time() < deadline:
            try:
                _ = svc.toolbox_describe(engine_id=new_engine_id, timeout_seconds=2.0)
                desc = svc.toolbox_describe(toolbox_id="toolbox-unregister", timeout_seconds=2.0)
                if list(desc.get("all_registered_tool_names") or []) == ["keep_tool"]:
                    break
            except Exception as exc:  # pragma: no cover - startup polling
                last_error = exc
            time.sleep(0.1)
        if desc is None:
            raise AssertionError(f"unregistered toolbox did not become ready: {last_error}")

        out_keep = svc.toolbox_execute(
            toolbox_id="toolbox-unregister",
            tool_call={"name": "keep_tool", "arguments": {"name": "K"}},
            timeout_seconds=5.0,
        )
        assert '"tool": "keep_tool"' in str(dict(out_keep.get("tool_call") or {}).get("result") or "")
        state_payload = json.loads((root / "state" / "toolbox_sandboxes.json").read_text(encoding="utf-8"))
        toolbox_row = dict(dict(state_payload.get("toolboxes") or {}).get("toolbox-unregister") or {})
        profile_row = next(iter(dict(toolbox_row.get("profiles") or {}).values()))
        history = list(profile_row.get("rollout_history") or [])
        assert history[-1]["action"] == "unregister_auto"
        assert history[-1]["engine_id"] == new_engine_id
        assert history[-1]["replaced_engine_id"] == old_engine_id

        with pytest.raises(PermissionError, match="tool_not_allowed:drop_tool"):
            svc.toolbox_execute(
                toolbox_id="toolbox-unregister",
                tool_call={"name": "drop_tool", "arguments": {"name": "D"}},
                timeout_seconds=5.0,
            )
    finally:
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-unregister"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_unregister_auto_removes_last_toolbox_state() -> None:
    root = _scratch_dir("unregister-last-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    try:
        created = svc.toolbox_register_auto(
            toolbox_id="toolbox-last-remove",
            requests=[
                {
                    "files": [
                        {
                            "relative_path": "lonely_tool.py",
                            "content": (
                                "def lonely_tool(name: str = 'world'):\n"
                                "    \"\"\"Lonely tool.\n\n"
                                "    Args:\n"
                                "        name (str): Name input.\n"
                                "    \"\"\"\n"
                                "    return {'tool': 'lonely_tool', 'name': name}\n"
                            ),
                        }
                    ],
                    "module_name": "lonely_tool",
                    "callable_name": "lonely_tool",
                    "sandbox_profile": {
                        "required_imports": ["requests"],
                        "sandbox_policy": {"sandbox": {"enabled": True}},
                    },
                }
            ],
            python_executable=sys.executable,
        )
        old_engine_id = str(list(created.get("spawned_engine_ids") or [None])[0] or "")
        old_profile = next(iter(dict(created.get("profiles") or {}).values()))
        old_env = dict(old_profile.get("environment") or {})
        old_venv_path = Path(str(old_env.get("venv_path") or "")).resolve()
        old_venv_key = str(old_env.get("venv_key") or "").strip()
        removed = svc.toolbox_unregister_auto(
            toolbox_id="toolbox-last-remove",
            tool_keys=["lonely_tool:lonely_tool"],
            python_executable=sys.executable,
        )
        assert removed["toolbox_removed"] is True
        assert removed["remaining_request_count"] == 0
        assert old_engine_id in list(removed.get("replaced_engine_ids") or [])
        assert old_venv_key in list(removed.get("removed_environment_keys") or [])

        state_path = root / "state" / "toolbox_sandboxes.json"
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        assert "toolbox-last-remove" not in dict(payload.get("toolboxes") or {})
        assert not old_venv_path.exists()
        with pytest.raises(ValueError, match="has no registered sandbox executors"):
            svc.toolbox_describe(toolbox_id="toolbox-last-remove", timeout_seconds=1.0)
    finally:
        for reg in svc.discover_running(prune_stale=False, include_reachability=False):
            eid = str(dict(reg or {}).get("engine_id") or "")
            if eid.startswith("toolbox-last-remove"):
                try:
                    svc.shutdown(eid, timeout_seconds=2.0)
                except Exception:
                    _ = svc.remove_registration(eid)
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_register_auto_rolls_back_new_assignments_when_readiness_fails(monkeypatch) -> None:
    root = _scratch_dir("register-auto-fail-")
    svc = EngineHostService(
        engines_state_file=root / "managed_engines.json",
        control_state_file=root / "access_control.json",
    )
    retired: list[str] = []
    cleanup_states: list[dict] = []

    def _fake_spawn_assignments(
        self,
        *,
        toolbox_id: str,
        requests: list,
        manual_requests=None,
        intrinsic_tool_names=None,
        intrinsic_profile=None,
        with_intrinsic_guides: bool = False,
        worker_profile_class: str = "generic",
    ):
        return [
            ToolboxSandboxAssignment(
                toolbox_id=toolbox_id,
                sandbox_profile=SandboxProfileSpec(profile_id="fs-only"),
                bundle_spec=ToolboxBundleSpec(
                    bundle_id=f"{toolbox_id}-fs-only",
                    toolbox_id=toolbox_id,
                    sandbox_profile=SandboxProfileSpec(profile_id="fs-only"),
                    auto_tools=[ToolboxBundleAutoTool(module_name="alpha", callable_name="alpha_tool")],
                ),
                registration={
                    "engine_id": "toolbox-fail-fs",
                    "bundle": {"bundle_revision": "rev-a"},
                    "environment": {"venv_key": "venv-a"},
                },
            )
        ]

    def _fake_ready(assignments, *, timeout_seconds: float = 8.0):
        raise RuntimeError("not ready")

    monkeypatch.setattr(ToolboxSandboxOrchestrator, "spawn_assignments", _fake_spawn_assignments)
    monkeypatch.setattr(svc, "_ensure_toolbox_assignments_ready", _fake_ready)
    monkeypatch.setattr(svc, "_retire_toolbox_registration", lambda engine_id: retired.append(str(engine_id)))
    monkeypatch.setattr(
        svc,
        "_cleanup_unused_toolbox_environments",
        lambda state=None: cleanup_states.append(dict(state or {})) or [],
    )

    try:
        with pytest.raises(RuntimeError, match="not ready"):
            svc.toolbox_register_auto(
                toolbox_id="toolbox-fail",
                requests=[
                    {
                        "files": [{"relative_path": "alpha.py", "content": "def alpha_tool():\n    return {'tool': 'alpha'}\n"}],
                        "module_name": "alpha",
                        "callable_name": "alpha_tool",
                        "sandbox_profile": {"profile_id": "fs-only"},
                    }
                ],
                python_executable=sys.executable,
            )

        assert retired == ["toolbox-fail-fs"]
        assert len(cleanup_states) == 1
        assert not (root / "state" / "toolbox_sandboxes.json").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_gc_reconciles_stale_registrations_and_artifacts() -> None:
    root = Path(".tmp_toolbox_gc").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        keep_bundle = (root / "toolbox_bundles" / "keep-bundle").resolve()
        stale_bundle = (root / "toolbox_bundles" / "stale-bundle").resolve()
        keep_bundle.mkdir(parents=True, exist_ok=True)
        stale_bundle.mkdir(parents=True, exist_ok=True)
        keep_env = (root / "toolbox_venvs" / "keep-env").resolve()
        stale_env = (root / "toolbox_venvs" / "stale-env").resolve()
        keep_env.mkdir(parents=True, exist_ok=True)
        stale_env.mkdir(parents=True, exist_ok=True)

        svc.register_spawned(
            engine_id="toolbox-keep",
            pid=1111,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "keep.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-toolbox-keep",
            executor_kind="toolbox_executor",
            bundle={
                "toolbox_id": "toolbox-gc",
                "bundle_root": str(keep_bundle),
            },
            environment={"venv_key": "keep-env", "venv_path": str(keep_env)},
            tool_access={"allowed_tool_names": ["keep_tool"]},
        )
        svc.register_spawned(
            engine_id="toolbox-stale",
            pid=2222,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "stale.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-toolbox-stale",
            executor_kind="toolbox_executor",
            bundle={
                "toolbox_id": "toolbox-gc",
                "bundle_root": str(stale_bundle),
            },
            environment={"venv_key": "stale-env", "venv_path": str(stale_env)},
            tool_access={"allowed_tool_names": ["stale_tool"]},
        )
        svc._write_toolboxes(
            {
                "version": 1,
                "environment_descriptions": {},
                "toolboxes": {
                    "toolbox-gc": {
                        "toolbox_id": "toolbox-gc",
                        "profiles": {
                            "default": {
                                "engine_id": "toolbox-keep",
                                "environment": {"venv_key": "keep-env", "venv_path": str(keep_env)},
                            }
                        },
                    }
                },
            }
        )

        gc_out = svc.toolbox_gc()

        assert gc_out["status"] == "ok"
        assert gc_out["removed_engine_ids"] == ["toolbox-stale"]
        assert "stale-bundle" in list(gc_out.get("removed_bundle_roots") or [])
        assert "stale-env" in list(gc_out.get("removed_environment_keys") or [])
        assert gc_out["removed_registration_details"] == [
            {
                "engine_id": "toolbox-stale",
                "toolbox_id": "toolbox-gc",
                "sandbox_profile_id": "default",
                "bundle_root": str(stale_bundle),
                "bundle_name": "stale-bundle",
                "reason": "unreferenced_live_registration",
            }
        ]
        assert gc_out["removed_bundle_details"] == []
        assert gc_out["removed_environment_details"] == [
            {
                "venv_key": "stale-env",
                "reason": "unreferenced_environment_directory",
            }
        ]
        assert keep_bundle.exists()
        assert not stale_bundle.exists()
        assert keep_env.exists()
        assert not stale_env.exists()
        assert svc.get_registration("toolbox-keep") is not None
        assert svc.get_registration("toolbox-stale") is None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_gc_preserves_referenced_runtime_envs_and_removes_stale_runtime_envs() -> None:
    root = Path(".tmp_runtime_env_gc").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        keep_env = (root / "runtime_envs" / "keep-runtime-env").resolve()
        stale_env = (root / "runtime_envs" / "stale-runtime-env").resolve()
        keep_env.mkdir(parents=True, exist_ok=True)
        stale_env.mkdir(parents=True, exist_ok=True)

        svc.register_spawned(
            engine_id="workflow-js-node-keep",
            pid=1111,
            command=["python", "-m", "hosting.workflow_js_node_worker_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "helper.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-helper-keep",
            executor_kind="workflow_js_node",
            environment={
                "venv_key": "keep-runtime-env",
                "venv_path": str(keep_env),
                "environment_root_kind": "runtime_envs",
                "environment_consumer_kind": "workflow_js_node",
            },
        )

        removed = svc._cleanup_unused_toolbox_environments({"version": 1, "toolboxes": {}})

        assert "stale-runtime-env" in removed
        assert keep_env.exists()
        assert not stale_env.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_references_reports_referenced_and_stale_artifacts() -> None:
    root = Path(".tmp_toolbox_refs").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        keep_bundle = (root / "toolbox_bundles" / "keep-bundle").resolve()
        stale_bundle = (root / "toolbox_bundles" / "stale-bundle").resolve()
        keep_bundle.mkdir(parents=True, exist_ok=True)
        stale_bundle.mkdir(parents=True, exist_ok=True)
        keep_env = (root / "toolbox_venvs" / "keep-env").resolve()
        stale_env = (root / "toolbox_venvs" / "stale-env").resolve()
        keep_env.mkdir(parents=True, exist_ok=True)
        stale_env.mkdir(parents=True, exist_ok=True)

        svc.register_spawned(
            engine_id="toolbox-keep",
            pid=1111,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "keep.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-toolbox-keep2",
            executor_kind="toolbox_executor",
            bundle={
                "toolbox_id": "toolbox-refs",
                "bundle_root": str(keep_bundle),
                "sandbox_profile_id": "default",
            },
            environment={"venv_key": "keep-env", "venv_path": str(keep_env)},
            tool_access={"allowed_tool_names": ["keep_tool"]},
        )
        svc.register_spawned(
            engine_id="toolbox-stale",
            pid=2222,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "stale.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-toolbox-stale2",
            executor_kind="toolbox_executor",
            bundle={
                "toolbox_id": "toolbox-refs",
                "bundle_root": str(stale_bundle),
                "sandbox_profile_id": "stale",
            },
            environment={"venv_key": "stale-env", "venv_path": str(stale_env)},
            tool_access={"allowed_tool_names": ["stale_tool"]},
        )
        svc._write_toolboxes(
            {
                "version": 1,
                "environment_descriptions": {},
                "toolboxes": {
                    "toolbox-refs": {
                        "toolbox_id": "toolbox-refs",
                        "profiles": {
                            "default": {
                                "engine_id": "toolbox-keep",
                                "bundle_revision": "rev-1",
                                "sandbox_profile": {"profile_id": "default"},
                                "requests": [],
                                "environment": {"venv_key": "keep-env", "venv_path": str(keep_env)},
                                "rollout": {},
                            }
                        },
                    }
                },
            }
        )

        refs = svc.toolbox_references()

        assert refs["status"] == "ok"
        assert refs["referenced_engine_ids"] == ["toolbox-keep"]
        assert refs["referenced_environment_keys"] == ["keep-env"]
        assert refs["referenced_environment_roots"] == [str(keep_env)]
        assert refs["stale_engine_ids"] == ["toolbox-stale"]
        assert refs["stale_bundle_roots"] == ["stale-bundle"]
        assert refs["stale_environment_keys"] == ["stale-env"]
        assert dict(refs["live_registrations"]["toolbox-keep"])["referenced"] is True
        assert dict(refs["live_registrations"]["toolbox-stale"])["referenced"] is False
        assert refs["referenced_environment_key_reasons"]["keep-env"] == [
            {
                "toolbox_id": "toolbox-refs",
                "sandbox_profile_id": "default",
                "kind": "profile_environment",
            }
        ]
        assert refs["referenced_environment_root_reasons"][str(keep_env)] == [
            {
                "toolbox_id": "toolbox-refs",
                "sandbox_profile_id": "default",
                "kind": "profile_environment",
                "venv_key": "keep-env",
            }
        ]
        assert refs["referenced_bundle_root_reasons"][str(keep_bundle)] == [
            {
                "engine_id": "toolbox-keep",
                "toolbox_id": "toolbox-refs",
                "sandbox_profile_id": "default",
                "kind": "live_registration",
            }
        ]
        assert "toolbox-refs" in dict(refs["toolboxes"] or {})
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_consistency_reports_profile_registration_and_environment_mismatches(monkeypatch) -> None:
    root = Path(".tmp_toolbox_consistency").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        monkeypatch.setattr(
            svc,
            "_probe_registration_reachability",
            lambda item, timeout_seconds=0.35: {"reachable": True, "probe": "hello"},
        )
        bundle_root = (root / "toolbox_bundles" / "bundle-a").resolve()
        bundle_root.mkdir(parents=True, exist_ok=True)
        env_root = (root / "toolbox_venvs" / "env-a").resolve()
        env_root.mkdir(parents=True, exist_ok=True)

        svc.register_spawned(
            engine_id="toolbox-live",
            pid=3333,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "live.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-toolbox-consistency",
            executor_kind="toolbox_executor",
            bundle={
                "toolbox_id": "toolbox-other",
                "bundle_root": str(bundle_root),
                "sandbox_profile_id": "profile-other",
            },
            environment={"venv_key": "env-a", "venv_path": str(env_root)},
            tool_access={"allowed_tool_names": ["wrong_tool"]},
        )
        svc._write_toolboxes(
            {
                "version": 1,
                "environment_descriptions": {},
                "toolboxes": {
                    "toolbox-demo": {
                        "toolbox_id": "toolbox-demo",
                        "requests": [
                            {
                                "files": [],
                                "module_name": "demo_mod",
                                "callable_name": "demo_tool",
                                "sandbox_profile": {"profile_id": "profile-a"},
                                "activate": True,
                                "guide_content": None,
                                "guide_description": None,
                            }
                        ],
                        "profiles": {
                            "profile-a": {
                                "engine_id": "toolbox-live",
                                "bundle_revision": "rev-a",
                                "sandbox_profile": {"profile_id": "profile-a"},
                                "requests": [
                                    {
                                        "files": [],
                                        "module_name": "demo_mod",
                                        "callable_name": "demo_tool",
                                        "sandbox_profile": {"profile_id": "profile-a"},
                                        "activate": True,
                                        "guide_content": None,
                                        "guide_description": None,
                                    }
                                ],
                                "environment": {"venv_key": "env-a", "venv_path": str(env_root)},
                                "rollout": {},
                            }
                        },
                    }
                },
            }
        )

        out = svc.toolbox_consistency()

        assert out["status"] == "ok"
        issues = list(out["issues"] or [])
        issue_names = {str(item.get("issue") or "") for item in issues}
        assert "registration_toolbox_id_mismatch" in issue_names
        assert "registration_profile_id_mismatch" in issue_names
        assert "registration_allowed_tool_names_mismatch" in issue_names
        assert "environment_metadata_missing" in issue_names
        assert out["issue_count"] == len(issues)
        assert dict(out["references"] or {})["referenced_engine_ids"] == ["toolbox-live"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_review_snapshot_filters_and_recommends_reconcile(monkeypatch) -> None:
    root = Path(".tmp_toolbox_review_snapshot").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        monkeypatch.setattr(
            svc,
            "_probe_registration_reachability",
            lambda item, timeout_seconds=0.35: {"reachable": True, "probe": "hello"},
        )
        bundle_root = (root / "toolbox_bundles" / "bundle-a").resolve()
        bundle_root.mkdir(parents=True, exist_ok=True)
        env_root = (root / "toolbox_venvs" / "env-a").resolve()
        env_root.mkdir(parents=True, exist_ok=True)

        svc.register_spawned(
            engine_id="toolbox-live",
            pid=3333,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "live.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-toolbox-review-snapshot",
            executor_kind="toolbox_executor",
            bundle={
                "toolbox_id": "toolbox-other",
                "bundle_root": str(bundle_root),
                "sandbox_profile_id": "profile-other",
            },
            environment={"venv_key": "env-a", "venv_path": str(env_root)},
            tool_access={"allowed_tool_names": ["wrong_tool"]},
        )
        svc._write_toolboxes(
            {
                "version": 1,
                "environment_descriptions": {},
                "toolboxes": {
                    "toolbox-demo": {
                        "toolbox_id": "toolbox-demo",
                        "requests": [
                            {
                                "files": [],
                                "module_name": "demo_mod",
                                "callable_name": "demo_tool",
                                "sandbox_profile": {"profile_id": "profile-a"},
                                "activate": True,
                                "guide_content": None,
                                "guide_description": None,
                            }
                        ],
                        "profiles": {
                            "profile-a": {
                                "engine_id": "toolbox-live",
                                "bundle_revision": "rev-a",
                                "sandbox_profile": {"profile_id": "profile-a"},
                                "requests": [
                                    {
                                        "files": [],
                                        "module_name": "demo_mod",
                                        "callable_name": "demo_tool",
                                        "sandbox_profile": {"profile_id": "profile-a"},
                                        "activate": True,
                                        "guide_content": None,
                                        "guide_description": None,
                                    }
                                ],
                                "environment": {"venv_key": "env-a", "venv_path": str(env_root)},
                                "rollout": {},
                            }
                        },
                    }
                },
            }
        )

        out = svc.toolbox_review_snapshot(toolbox_ids=["toolbox-demo"])

        assert out["status"] == "ok"
        assert out["toolbox_ids"] == ["toolbox-demo"]
        assert out["recommended_action"] == "reconcile"
        assert dict(out["summary"])["issue_count"] == 4
        assert list(dict(out["toolboxes"]).keys()) == ["toolbox-demo"]
        assert int(dict(out["toolboxes"])["toolbox-demo"]["issue_count"]) == 4
        assert len(list(out["issues"] or [])) == 4
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_consistency_and_review_snapshot_report_unreachable_live_registration(monkeypatch) -> None:
    root = Path(".tmp_toolbox_liveness").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        monkeypatch.setattr(
            svc,
            "_probe_registration_reachability",
            lambda item, timeout_seconds=0.35: {
                "reachable": False,
                "probe": "hello",
                "error": "worker process may not be running",
            },
        )
        bundle_root = (root / "toolbox_bundles" / "bundle-a").resolve()
        bundle_root.mkdir(parents=True, exist_ok=True)
        env_root = (root / "toolbox_venvs" / "env-a").resolve()
        env_root.mkdir(parents=True, exist_ok=True)
        (env_root / "environment.json").write_text("{}", encoding="utf-8")

        svc.register_spawned(
            engine_id="toolbox-live",
            pid=3333,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "live.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-toolbox-liveness",
            executor_kind="toolbox_executor",
            bundle={
                "toolbox_id": "toolbox-demo",
                "bundle_root": str(bundle_root),
                "sandbox_profile_id": "profile-a",
            },
            environment={"venv_key": "env-a", "venv_path": str(env_root)},
            tool_access={"allowed_tool_names": ["demo_tool"]},
        )
        svc._write_toolboxes(
            {
                "version": 1,
                "environment_descriptions": {},
                "toolboxes": {
                    "toolbox-demo": {
                        "toolbox_id": "toolbox-demo",
                        "requests": [
                            {
                                "files": [],
                                "module_name": "demo_mod",
                                "callable_name": "demo_tool",
                                "sandbox_profile": {"profile_id": "profile-a"},
                                "activate": True,
                                "guide_content": None,
                                "guide_description": None,
                            }
                        ],
                        "profiles": {
                            "profile-a": {
                                "engine_id": "toolbox-live",
                                "bundle_revision": "rev-a",
                                "sandbox_profile": {"profile_id": "profile-a"},
                                "requests": [
                                    {
                                        "files": [],
                                        "module_name": "demo_mod",
                                        "callable_name": "demo_tool",
                                        "sandbox_profile": {"profile_id": "profile-a"},
                                        "activate": True,
                                        "guide_content": None,
                                        "guide_description": None,
                                    }
                                ],
                                "environment": {"venv_key": "env-a", "venv_path": str(env_root)},
                                "rollout": {},
                            }
                        },
                    }
                },
            }
        )

        consistency = svc.toolbox_consistency()
        issues = list(consistency.get("issues") or [])
        assert consistency["issue_count"] == 1
        assert issues[0]["issue"] == "live_registration_unreachable"
        assert issues[0]["toolbox_id"] == "toolbox-demo"
        assert issues[0]["engine_id"] == "toolbox-live"
        live_reg = dict(dict(consistency["references"])["live_registrations"]["toolbox-live"])
        assert live_reg["reachable"] is False
        assert "reachability" in live_reg

        review = svc.toolbox_review_snapshot(toolbox_ids=["toolbox-demo"])
        assert review["recommended_action"] == "reconcile"
        assert review["summary"]["issue_count"] == 1
        profile = dict(review["toolboxes"]["toolbox-demo"])["profiles"][0]
        assert profile["engine_id"] == "toolbox-live"
        assert profile["reachable"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_references_do_not_mark_live_parent_bundle_dirs_stale() -> None:
    root = Path(".tmp_toolbox_refs_nested").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        profile_root = (root / "toolbox_bundles" / "toolbox-demo-profile-a").resolve()
        bundle_root = (profile_root / "rev-a").resolve()
        profile_root.mkdir(parents=True, exist_ok=True)
        bundle_root.mkdir(parents=True, exist_ok=True)

        svc.register_spawned(
            engine_id="toolbox-live",
            pid=3333,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "live.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-toolbox-review-nested",
            executor_kind="toolbox_executor",
            bundle={
                "toolbox_id": "toolbox-demo",
                "bundle_root": str(bundle_root),
                "sandbox_profile_id": "profile-a",
            },
            environment={},
            tool_access={"allowed_tool_names": ["demo_tool"]},
        )
        svc._write_toolboxes(
            {
                "version": 1,
                "environment_descriptions": {},
                "toolboxes": {
                    "toolbox-demo": {
                        "toolbox_id": "toolbox-demo",
                        "profiles": {
                            "profile-a": {
                                "engine_id": "toolbox-live",
                                "bundle_revision": "rev-a",
                                "sandbox_profile": {"profile_id": "profile-a"},
                                "requests": [],
                                "environment": {},
                                "rollout": {},
                            }
                        },
                    }
                },
            }
        )

        refs = svc.toolbox_references()

        assert refs["stale_bundle_roots"] == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_references_and_gc_preserve_referenced_environment_by_path() -> None:
    root = Path(".tmp_toolbox_refs_env_path").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        keep_env = (root / "toolbox_venvs" / "custom-env-dir").resolve()
        stale_env = (root / "toolbox_venvs" / "stale-env-dir").resolve()
        keep_env.mkdir(parents=True, exist_ok=True)
        stale_env.mkdir(parents=True, exist_ok=True)
        bundle_root = (root / "toolbox_bundles" / "bundle-a").resolve()
        bundle_root.mkdir(parents=True, exist_ok=True)

        svc.register_spawned(
            engine_id="toolbox-live",
            pid=3333,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "live.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-toolbox-env-path",
            executor_kind="toolbox_executor",
            bundle={
                "toolbox_id": "toolbox-demo",
                "bundle_root": str(bundle_root),
                "sandbox_profile_id": "profile-a",
            },
            environment={"venv_key": "env-key-a", "venv_path": str(keep_env)},
            tool_access={"allowed_tool_names": ["demo_tool"]},
        )
        svc._write_toolboxes(
            {
                "version": 1,
                "environment_descriptions": {},
                "toolboxes": {
                    "toolbox-demo": {
                        "toolbox_id": "toolbox-demo",
                        "profiles": {
                            "profile-a": {
                                "engine_id": "toolbox-live",
                                "bundle_revision": "rev-a",
                                "sandbox_profile": {"profile_id": "profile-a"},
                                "requests": [],
                                "environment": {"venv_key": "env-key-a", "venv_path": str(keep_env)},
                                "rollout": {},
                            }
                        },
                    }
                },
            }
        )

        refs = svc.toolbox_references()
        gc_out = svc.toolbox_gc()

        assert refs["referenced_environment_keys"] == ["env-key-a"]
        assert refs["referenced_environment_roots"] == [str(keep_env)]
        assert refs["referenced_environment_key_reasons"]["env-key-a"] == [
            {
                "toolbox_id": "toolbox-demo",
                "sandbox_profile_id": "profile-a",
                "kind": "profile_environment",
            }
        ]
        assert refs["referenced_environment_root_reasons"][str(keep_env)] == [
            {
                "toolbox_id": "toolbox-demo",
                "sandbox_profile_id": "profile-a",
                "kind": "profile_environment",
                "venv_key": "env-key-a",
            }
        ]
        assert refs["stale_environment_keys"] == ["stale-env-dir"]
        assert keep_env.exists()
        assert not stale_env.exists()
        assert gc_out["removed_environment_keys"] == ["stale-env-dir"]
        assert gc_out["removed_environment_details"] == [
            {
                "venv_key": "stale-env-dir",
                "reason": "unreferenced_environment_directory",
            }
        ]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_repair_rebuilds_inconsistent_toolbox_from_persisted_state(monkeypatch) -> None:
    root = Path(".tmp_toolbox_repair").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        old_bundle = (root / "toolbox_bundles" / "old-bundle").resolve()
        new_bundle = (root / "toolbox_bundles" / "new-bundle").resolve()
        old_bundle.mkdir(parents=True, exist_ok=True)
        new_bundle.mkdir(parents=True, exist_ok=True)
        old_env = (root / "toolbox_venvs" / "old-env").resolve()
        new_env = (root / "toolbox_venvs" / "new-env").resolve()
        old_env.mkdir(parents=True, exist_ok=True)
        new_env.mkdir(parents=True, exist_ok=True)

        svc.register_spawned(
            engine_id="toolbox-old",
            pid=4444,
            command=["python", "-m", "hosting.toolbox_executor_ipc"],
            worker_ipc_family="AF_UNIX" if sys.platform != "win32" else "AF_PIPE",
            worker_ipc_address=str(root / "old.sock") if sys.platform != "win32" else r"\\.\pipe\mp13-toolbox-old",
            executor_kind="toolbox_executor",
            bundle={
                "toolbox_id": "toolbox-demo",
                "bundle_root": str(old_bundle),
                "sandbox_profile_id": "profile-a",
            },
            environment={"venv_key": "old-env", "venv_path": str(old_env)},
            tool_access={"allowed_tool_names": ["wrong_tool"]},
        )
        svc._write_toolboxes(
            {
                "version": 1,
                "environment_descriptions": {},
                "toolboxes": {
                    "toolbox-demo": {
                        "toolbox_id": "toolbox-demo",
                        "requests": [
                            {
                                "files": [],
                                "module_name": "demo_mod",
                                "callable_name": "demo_tool",
                                "sandbox_profile": {"profile_id": "profile-a"},
                                "activate": True,
                                "guide_content": None,
                                "guide_description": None,
                            }
                        ],
                        "profiles": {
                            "profile-a": {
                                "engine_id": "toolbox-old",
                                "bundle_revision": "rev-old",
                                "sandbox_profile": {"profile_id": "profile-a"},
                                "requests": [
                                    {
                                        "files": [],
                                        "module_name": "demo_mod",
                                        "callable_name": "demo_tool",
                                        "sandbox_profile": {"profile_id": "profile-a"},
                                        "activate": True,
                                        "guide_content": None,
                                        "guide_description": None,
                                    }
                                ],
                                "environment": {"venv_key": "old-env", "venv_path": str(old_env)},
                                "rollout": {},
                            }
                        },
                        "runtime": {"python_executable": "python-demo", "worker_profile_class": "generic"},
                    }
                },
            }
        )

        retired: list[str] = []
        monkeypatch.setattr(svc, "_retire_toolbox_registration", lambda engine_id: retired.append(str(engine_id)))

        class _FakeReg(dict):
            pass

        class _FakeBundle:
            def __init__(self) -> None:
                self.bundle_root = new_bundle

            def registration_bundle(self) -> dict:
                return {
                    "toolbox_id": "toolbox-demo",
                    "bundle_root": str(new_bundle),
                    "bundle_revision": "rev-new",
                    "sandbox_profile_id": "profile-a",
                }

        class _FakeAssignment:
            def __init__(self) -> None:
                self.sandbox_profile = SandboxProfileSpec(profile_id="profile-a")
                self.staged_bundle = _FakeBundle()
                self.registration = {
                    "engine_id": "toolbox-new",
                    "bundle": self.staged_bundle.registration_bundle(),
                    "environment": {"venv_key": "new-env", "venv_path": str(new_env)},
                }

        def _fake_spawn_assignments(self, **kwargs):
            assert kwargs["toolbox_id"] == "toolbox-demo"
            return [_FakeAssignment()]

        monkeypatch.setattr(ToolboxSandboxOrchestrator, "spawn_assignments", _fake_spawn_assignments)
        monkeypatch.setattr(
            svc,
            "_ensure_toolbox_assignments_ready",
            lambda assignments, timeout_seconds=8.0: {
                "toolbox-new": {
                    "ready": True,
                    "ready_at": time.time(),
                    "warmup_ms": 5,
                    "tool_inventory_ok": True,
                    "tool_count": 1,
                    "all_registered_tool_names": ["demo_tool"],
                }
            },
        )

        out = svc.toolbox_repair()

        assert out["status"] == "ok"
        assert out["requested_toolbox_ids"] == []
        assert out["target_toolbox_ids"] == ["toolbox-demo"]
        assert out["repaired_toolbox_ids"] == ["toolbox-demo"]
        assert out["skipped_toolbox_ids"] == []
        assert out["outcome"] == "repaired"
        assert retired == ["toolbox-old"]
        state = svc._read_toolboxes()
        profile = dict(dict(dict(state["toolboxes"])["toolbox-demo"])["profiles"])["profile-a"]
        assert profile["engine_id"] == "toolbox-new"
        assert dict(profile["environment"])["venv_key"] == "new-env"
        history = list(profile.get("rollout_history") or [])
        assert history and history[-1]["action"] == "repair"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_reconcile_chains_consistency_repair_and_gc(monkeypatch) -> None:
    root = Path(".tmp_toolbox_reconcile").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        calls: list[tuple[str, object]] = []

        consistency_states = [
            {"status": "ok", "issue_count": 1, "issues": [{"toolbox_id": "toolbox-demo", "issue": "missing_live_registration"}], "references": {}},
            {"status": "ok", "issue_count": 0, "issues": [], "references": {}},
        ]

        def _fake_consistency():
            calls.append(("consistency", None))
            return consistency_states.pop(0)

        def _fake_repair(*, toolbox_ids=None, only_inconsistent=True):
            calls.append(("repair", {"toolbox_ids": toolbox_ids, "only_inconsistent": only_inconsistent}))
            return {"status": "ok", "repaired": {"toolbox-demo": {}}, "skipped": {}, "removed_environment_keys": []}

        def _fake_gc():
            calls.append(("gc", None))
            return {"status": "ok", "removed_engine_ids": ["toolbox-stale"], "removed_bundle_roots": ["stale-bundle"], "removed_environment_keys": ["stale-env"]}

        monkeypatch.setattr(svc, "toolbox_consistency", _fake_consistency)
        monkeypatch.setattr(svc, "toolbox_repair", _fake_repair)
        monkeypatch.setattr(svc, "toolbox_gc", _fake_gc)

        out = svc.toolbox_reconcile(toolbox_ids=["toolbox-demo"], only_inconsistent=False)

        assert out["status"] == "ok"
        assert out["toolbox_ids"] == ["toolbox-demo"]
        assert out["requested_toolbox_ids"] == ["toolbox-demo"]
        assert out["target_toolbox_ids"] == ["toolbox-demo"]
        assert out["repaired_toolbox_ids"] == ["toolbox-demo"]
        assert out["removed_engine_ids"] == ["toolbox-stale"]
        assert out["outcome"] == "repaired"
        assert dict(out["summary"]) == {
            "before_issue_count": 1,
            "after_issue_count": 0,
            "removed_engine_count": 1,
            "removed_bundle_count": 1,
            "removed_environment_count": 1,
            "repaired_toolbox_count": 1,
            "requested_toolbox_count": 1,
            "target_toolbox_count": 1,
        }
        assert calls == [
            ("consistency", None),
            ("repair", {"toolbox_ids": ["toolbox-demo"], "only_inconsistent": False}),
            ("gc", None),
            ("consistency", None),
        ]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_repair_skips_requested_toolboxes_when_only_inconsistent_and_no_issues(monkeypatch) -> None:
    root = Path(".tmp_toolbox_repair_requested_skip").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        svc._write_toolboxes(
            {
                "version": 1,
                "environment_descriptions": {},
                "toolboxes": {
                    "toolbox-demo": {
                        "toolbox_id": "toolbox-demo",
                        "requests": [],
                        "manual_requests": [],
                        "profiles": {},
                    }
                },
            }
        )

        monkeypatch.setattr(
            svc,
            "toolbox_consistency",
            lambda: {"status": "ok", "issue_count": 0, "issues": [], "summary": {"issue_count": 0}},
        )

        out = svc.toolbox_repair(toolbox_ids=["toolbox-demo"], only_inconsistent=True)

        assert out["status"] == "ok"
        assert out["requested_toolbox_ids"] == ["toolbox-demo"]
        assert out["target_toolbox_ids"] == []
        assert out["inconsistent_toolbox_ids"] == []
        assert out["repaired_toolbox_ids"] == []
        assert out["skipped_toolbox_ids"] == []
        assert out["outcome"] == "noop"
        assert dict(out["summary"] or {})["repaired_toolbox_count"] == 0
        assert dict(out["summary"] or {})["requested_toolbox_count"] == 1
        assert dict(out["summary"] or {})["target_toolbox_count"] == 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_repair_can_include_details_when_requested(monkeypatch) -> None:
    root = Path(".tmp_toolbox_repair_details").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        monkeypatch.setattr(
            svc,
            "toolbox_consistency",
            lambda: {"status": "ok", "issue_count": 0, "issues": [], "summary": {"issue_count": 0}},
        )
        monkeypatch.setattr(svc, "_cleanup_unused_toolbox_environments", lambda state=None: [])

        out = svc.toolbox_repair(toolbox_ids=["toolbox-demo"], only_inconsistent=True, details=True)

        assert "repaired" in out
        assert "skipped" in out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_repair_serializes_concurrent_rebuilds_for_same_toolbox(monkeypatch) -> None:
    root = Path(".tmp_toolbox_repair_serialized").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        svc._write_toolboxes(
            {
                "version": 1,
                "environment_descriptions": {},
                "toolboxes": {
                    "toolbox-demo": {
                        "toolbox_id": "toolbox-demo",
                        "requests": [
                            {
                                "files": [],
                                "module_name": "demo_mod",
                                "callable_name": "demo_tool",
                                "sandbox_profile": {"profile_id": "profile-a"},
                                "activate": True,
                                "guide_content": None,
                                "guide_description": None,
                            }
                        ],
                        "profiles": {
                            "profile-a": {
                                "engine_id": "toolbox-old",
                                "bundle_revision": "rev-old",
                                "sandbox_profile": {"profile_id": "profile-a"},
                                "requests": [
                                    {
                                        "files": [],
                                        "module_name": "demo_mod",
                                        "callable_name": "demo_tool",
                                        "sandbox_profile": {"profile_id": "profile-a"},
                                        "activate": True,
                                        "guide_content": None,
                                        "guide_description": None,
                                    }
                                ],
                                "environment": {"venv_key": "old-env", "venv_path": str(root / "venv-old")},
                                "rollout": {},
                            }
                        },
                        "runtime": {"python_executable": "python-demo", "worker_profile_class": "generic"},
                    }
                },
            }
        )

        monkeypatch.setattr(
            svc,
            "toolbox_consistency",
            lambda: {
                "status": "ok",
                "issue_count": 1,
                "issues": [{"toolbox_id": "toolbox-demo", "issue": "missing_live_registration"}],
                "summary": {"issue_count": 1},
            },
        )
        monkeypatch.setattr(svc, "_cleanup_unused_toolbox_environments", lambda state=None: [])
        monkeypatch.setattr(svc, "_retire_toolbox_registration", lambda engine_id: None)

        inflight = 0
        max_inflight = 0
        inflight_lock = threading.Lock()

        class _FakeBundle:
            def registration_bundle(self) -> dict:
                return {
                    "toolbox_id": "toolbox-demo",
                    "bundle_root": str(root / "toolbox-bundles" / "demo"),
                    "bundle_revision": "rev-new",
                    "sandbox_profile_id": "profile-a",
                }

        class _FakeAssignment:
            def __init__(self) -> None:
                self.sandbox_profile = SandboxProfileSpec(profile_id="profile-a")
                self.staged_bundle = _FakeBundle()
                self.registration = {
                    "engine_id": "toolbox-new",
                    "bundle": self.staged_bundle.registration_bundle(),
                    "environment": {"venv_key": "new-env", "venv_path": str(root / "venv-new")},
                }

        def _fake_spawn_assignments(self, **kwargs):
            nonlocal inflight, max_inflight
            assert kwargs["toolbox_id"] == "toolbox-demo"
            with inflight_lock:
                inflight += 1
                max_inflight = max(max_inflight, inflight)
            try:
                time.sleep(0.10)
                return [_FakeAssignment()]
            finally:
                with inflight_lock:
                    inflight -= 1

        monkeypatch.setattr(ToolboxSandboxOrchestrator, "spawn_assignments", _fake_spawn_assignments)
        monkeypatch.setattr(
            svc,
            "_ensure_toolbox_assignments_ready",
            lambda assignments, timeout_seconds=8.0: {
                "toolbox-new": {
                    "ready": True,
                    "ready_at": time.time(),
                    "warmup_ms": 5,
                    "tool_inventory_ok": True,
                    "tool_count": 1,
                    "all_registered_tool_names": ["demo_tool"],
                }
            },
        )

        results: list[dict] = []

        def _run_repair() -> None:
            results.append(
                svc.toolbox_repair(toolbox_ids=["toolbox-demo"], only_inconsistent=False)
            )

        first = threading.Thread(target=_run_repair)
        second = threading.Thread(target=_run_repair)
        first.start()
        second.start()
        first.join()
        second.join()

        assert len(results) == 2
        assert max_inflight == 1
        assert all(result["status"] == "ok" for result in results)
        assert all(result["repaired_toolbox_ids"] == ["toolbox-demo"] for result in results)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_toolbox_reconcile_can_include_details_when_requested(monkeypatch) -> None:
    root = Path(".tmp_toolbox_reconcile_details").resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    try:
        svc = EngineHostService(
            engines_state_file=root / "managed_engines.json",
            control_state_file=root / "access_control.json",
        )
        monkeypatch.setattr(svc, "toolbox_consistency", lambda: {"status": "ok", "issue_count": 0, "issues": [], "references": {}})
        monkeypatch.setattr(
            svc,
            "toolbox_repair",
            lambda *, toolbox_ids=None, only_inconsistent=True: {
                "status": "ok",
                "requested_toolbox_ids": list(toolbox_ids or []),
                "target_toolbox_ids": [],
                "inconsistent_toolbox_ids": [],
                "repaired": {},
                "skipped": {},
                "removed_environment_keys": [],
            },
        )
        monkeypatch.setattr(
            svc,
            "toolbox_gc",
            lambda: {
                "status": "ok",
                "removed_engine_ids": [],
                "removed_bundle_roots": [],
                "removed_environment_keys": [],
            },
        )

        out = svc.toolbox_reconcile(toolbox_ids=["toolbox-demo"], details=True)

        assert "before" in out
        assert "repair" in out
        assert "gc" in out
        assert "after" in out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_sandboxed_toolbox_facade_builder_api_batches_requests() -> None:
    class _FakeHost:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def toolbox_register_auto(self, **kwargs):
            self.calls.append(("register_auto", dict(kwargs)))
            return {"status": "ok"}
            
        def toolbox_register_manual(self, **kwargs):
            self.calls.append(("register_manual", dict(kwargs)))
            return {"status": "ok"}

    host = _FakeHost()
    facade = SandboxedToolboxFacade(toolbox_id="facade-box", host=host, python_executable="python.exe")

    builder = facade.mutate()
    builder.register_auto_callable(
        relative_path="facade_tools1.py",
        content="def hello_auto1(): pass",
        module_name="facade_tools1",
        callable_name="hello_auto1"
    )
    builder.register_auto_callable(
        relative_path="facade_tools2.py",
        content="def hello_auto2(): pass",
        module_name="facade_tools2",
        callable_name="hello_auto2"
    )
    
    # Assert no backend calls made yet
    assert len(host.calls) == 0
    
    resolved_facade = builder.resolve_sandbox()
    
    # Assert it returns the base ref
    assert resolved_facade is facade
    
    # Assert 1 backend call made with 2 batched requests
    assert len(host.calls) == 1
    call_name, call_kwargs = host.calls[0]
    assert call_name == "register_auto"
    assert call_kwargs["toolbox_id"] == "facade-box"
    assert len(call_kwargs["requests"]) == 2
    
    req1 = call_kwargs["requests"][0]
    assert req1["module_name"] == "facade_tools1"
    assert req1["callable_name"] == "hello_auto1"
    
    req2 = call_kwargs["requests"][1]
    assert req2["module_name"] == "facade_tools2"
    assert req2["callable_name"] == "hello_auto2"

