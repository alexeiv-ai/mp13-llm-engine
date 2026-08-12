from pathlib import Path

import pytest

from hosting import transport_admin_setup_api as admin_setup


@pytest.mark.parametrize(
    ("platform_name", "suffix", "expected"),
    [
        ("windows", ".ps1", "Set-Service -Name sshd"),
        ("unix", ".sh", "systemctl enable --now ssh.service"),
        ("macos", ".sh", "systemctl enable --now ssh.service"),
    ],
)
def test_admin_setup_plan_is_platform_specific(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    suffix: str,
    expected: str,
) -> None:
    monkeypatch.setattr(admin_setup, "_admin_setup_platform", lambda: platform_name)

    result = admin_setup.plan_transport_admin_setup(
        admin_setup.TransportAdminSetupRequest(
            enable_ssh_service=True,
            enable_firewall=True,
            enable_user_linger=True,
            target_user="operator",
        )
    )

    assert result["status"] == "dry_run"
    assert result["execute"] is False
    assert result["script_suffix"] == suffix
    assert expected in result["script"]
    assert "operator" in result["script"] or platform_name == "windows"


def test_admin_setup_execute_requires_explicit_authority() -> None:
    with pytest.raises(PermissionError, match="execute=True"):
        admin_setup.execute_transport_admin_setup({"execute": False})


@pytest.mark.parametrize(
    ("platform_name", "expected_method", "expected_program"),
    [
        ("windows", "windows_uac", "powershell"),
        ("macos", "macos_authorization", "osascript"),
    ],
)
def test_admin_setup_elevation_command_is_explicit(
    platform_name: str,
    expected_method: str,
    expected_program: str,
) -> None:
    command, method = admin_setup._admin_setup_elevation_command(
        Path("admin setup.ps1"), platform_name=platform_name
    )

    assert method == expected_method
    assert command[0] == expected_program
