from __future__ import annotations

import inspect

import pytest

from hosting import engine_host_cli
from hosting.daemon.local_ipc import EngineHostDaemon
from hosting.engine_host_channel import EngineHostControlChannel
from hosting.service.host_service import EngineHostService
from hosting.toolbox.bundle_models import ToolboxAutoAssignmentRequestV2
from hosting.toolbox.hosted_ref import HostedToolBoxRef


OLD_METHODS = {
    "mutate",
    "register_auto_callable",
    "add_auto_callable",
    "register_python_callable",
    "add_python_callable",
    "register_manual_tool",
    "add_manual_tool",
    "unregister_auto_callable",
    "remove_auto_callable",
    "unregister_manual_tool",
    "remove_manual_tool",
    "register_intrinsic_tools",
    "add_intrinsic_tools",
    "unregister_intrinsic_tools",
    "remove_intrinsic_tools",
    "resolve_sandbox",
    "environment_descriptions",
    "list_environment_descriptions",
    "upsert_environment_description",
    "clone_environment_description",
    "resolve_environment_requirements",
    "apply_environment_description",
    "realize_environment",
    "sync_environment_description",
    "prepare_environment_install",
    "lock_environment_install",
    "resolve_environment_install_lock",
    "verify_environment_install_lock",
    "execute_environment_install",
    "verify_environment_install_receipt",
}

OLD_COMMANDS = {
    "toolbox-register-auto",
    "toolbox-unregister-auto",
    "toolbox-register-manual",
    "toolbox-unregister-manual",
    "toolbox-register-intrinsics",
    "toolbox-unregister-intrinsics",
    "toolbox-environment-list",
    "toolbox-environment-upsert",
    "toolbox-environment-clone",
    "toolbox-environment-resolve",
    "toolbox-environment-apply",
    "toolbox-environment-realize",
    "toolbox-environment-sync",
    "toolbox-environment-prepare-install",
    "toolbox-environment-lock-install",
    "toolbox-environment-resolve-install-lock",
    "toolbox-environment-verify-install-lock",
    "toolbox-environment-execute-install",
    "toolbox-environment-verify-install-receipt",
}


def test_old_hosted_reference_and_pending_builder_are_absent() -> None:
    assert not OLD_METHODS & set(dir(HostedToolBoxRef))
    import hosting.toolbox as toolbox

    assert not hasattr(toolbox, "PendingHostedToolboxRef")
    with pytest.raises(TypeError):
        HostedToolBoxRef(toolbox_id="tb", host=object(), python_executable="python")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="legacy_toolbox_runtime_selector_rejected"):
        HostedToolBoxRef.from_dict(
            {"toolbox_id": "tb", "python_executable": "python"}, host=object()
        )


def test_old_service_channel_and_command_routes_are_absent() -> None:
    old_transport_methods = {
        "toolbox_register_auto",
        "toolbox_unregister_auto",
        "toolbox_register_manual",
        "toolbox_unregister_manual",
        "toolbox_register_intrinsics",
        "toolbox_unregister_intrinsics",
        "toolbox_environment_description_list",
        "toolbox_environment_description_get",
        "toolbox_environment_description_effective_get",
        "toolbox_environment_description_upsert",
        "toolbox_environment_description_clone",
        "toolbox_environment_resolve_requirements",
        "toolbox_environment_apply",
        "toolbox_environment_realize",
        "toolbox_environment_sync_description",
        "toolbox_environment_prepare_install",
        "toolbox_environment_lock_install",
        "toolbox_environment_resolve_install_lock",
        "toolbox_environment_verify_install_lock",
        "toolbox_environment_execute_install",
        "toolbox_environment_verify_install_receipt",
    }
    assert not old_transport_methods & set(dir(EngineHostService))
    assert not old_transport_methods & set(dir(EngineHostControlChannel))
    assert not OLD_COMMANDS & set(engine_host_cli.EXAMPLES_BY_COMMAND)
    daemon_source = inspect.getsource(EngineHostDaemon)
    cli_source = inspect.getsource(engine_host_cli)
    assert all(command not in daemon_source for command in OLD_COMMANDS)
    assert all(command not in cli_source for command in OLD_COMMANDS)
    for role in ("admin", "config_editor", "worker_user", "diagnostic_user"):
        assert not OLD_COMMANDS & EngineHostService._commands_allowed_for_role(role)  # noqa: SLF001
    assert not hasattr(EngineHostService, "toolbox_template_publish")
    assert not hasattr(EngineHostControlChannel, "toolbox_template_publish")
    assert "toolbox-template-publish" not in engine_host_cli.EXAMPLES_BY_COMMAND
    assert "toolbox-template-publish" not in inspect.getsource(EngineHostDaemon)


def test_definition_request_rejects_legacy_runtime_fields() -> None:
    request = {
        "files": [{"relative_path": "demo.py", "content_b64": "cGFzcwo="}],
        "module_name": "demo",
        "callable_name": "hello",
        "dependency": {
            "mode": "auto",
            "template_id": None,
            "declared_imports": [],
            "package_requirements": [],
        },
        "sandbox_policy": {},
        "activate": True,
        "hidden": False,
        "non_restartable": False,
        "guide_content": None,
        "guide_description": None,
        "callback_signature": None,
        "concurrency": None,
    }
    for field, value in (
        ("environment_name", "base"),
        ("required_imports", ["numpy"]),
        ("profile_id", "consumer-selected"),
        ("python_executable", "python"),
    ):
        with pytest.raises(ValueError, match="unknown_fields"):
            ToolboxAutoAssignmentRequestV2.from_dict({**request, field: value})


def test_no_ambient_or_mutable_description_fallback_remains() -> None:
    from hosting.toolbox import orchestration
    from hosting.service import host_service

    source = inspect.getsource(orchestration.ToolboxSandboxOrchestrator.spawn_assignments)
    assert "hermetic_toolbox_environment_builder_required" in source
    assert "ensure_for_bundle" not in source
    assert "toolbox_environment_description" not in source
    assert "ToolboxEnvironmentMixin" not in inspect.getsource(host_service)
