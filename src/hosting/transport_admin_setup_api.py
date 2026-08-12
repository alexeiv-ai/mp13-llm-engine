"""Stable API for explicit elevated transport host setup."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


def _admin_setup_platform() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "unix"


def _admin_setup_script(args: Any, *, platform_name: str) -> Tuple[str, str, list[str]]:
    enable_ssh = bool(getattr(args, "admin_setup_enable_ssh_service", True))
    enable_firewall = bool(getattr(args, "admin_setup_enable_firewall", False))
    enable_linger = bool(getattr(args, "admin_setup_enable_user_linger", False))
    target_user = str(getattr(args, "admin_setup_target_user", "") or "").strip()
    followups: list[str] = []
    if platform_name == "windows":
        lines = [
            "$ErrorActionPreference = 'Stop'",
            "Write-Host 'mp13 hosting admin setup: Windows OpenSSH/service checks'",
        ]
        if enable_ssh:
            lines.extend(
                [
                    "$cap = Get-WindowsCapability -Online -Name "
                    "'OpenSSH.Server~~~~0.0.1.0' -ErrorAction SilentlyContinue",
                    "if ($cap -and $cap.State -ne 'Installed') { "
                    "Add-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0' }",
                    "Set-Service -Name sshd -StartupType Automatic",
                    "Start-Service sshd",
                ]
            )
        if enable_firewall:
            lines.extend(
                [
                    "$rule = Get-NetFirewallRule -Name 'mp13-hosting-sshd' "
                    "-ErrorAction SilentlyContinue",
                    "if (-not $rule) { New-NetFirewallRule -Name 'mp13-hosting-sshd' "
                    "-DisplayName 'mp13 Hosting OpenSSH Server' -Enabled True "
                    "-Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 }",
                ]
            )
        else:
            followups.append(
                "Firewall rule was not requested; remote SSH may still be blocked "
                "by Windows Firewall or network policy."
            )
        if enable_linger:
            followups.append(
                "Windows user daemon auto-start is not configured here; use Task "
                "Scheduler or service-managed hosting setup."
            )
        lines.append("Write-Host 'mp13 hosting admin setup complete'")
        return "\r\n".join(lines) + "\r\n", ".ps1", followups

    target_user_expr = shlex.quote(target_user) if target_user else "${SUDO_USER:-$USER}"
    lines = [
        "#!/bin/sh",
        "set -eu",
        "echo 'mp13 hosting admin setup: SSH/service checks'",
    ]
    if enable_ssh:
        lines.extend(
            [
                "if command -v systemctl >/dev/null 2>&1; then",
                "  if systemctl list-unit-files ssh.service >/dev/null 2>&1; then",
                "    systemctl enable --now ssh.service",
                "  elif systemctl list-unit-files sshd.service >/dev/null 2>&1; then",
                "    systemctl enable --now sshd.service",
                "  else",
                "    echo 'No ssh.service or sshd.service unit was found; install/enable "
                "OpenSSH server using the platform package manager.'",
                "  fi",
                "elif command -v service >/dev/null 2>&1; then",
                "  service ssh start 2>/dev/null || service sshd start 2>/dev/null || "
                "echo 'Could not start ssh/sshd through service(8).'",
                "else",
                "  echo 'No supported service manager was found; enable OpenSSH server manually.'",
                "fi",
            ]
        )
    if enable_firewall:
        lines.extend(
            [
                "if command -v ufw >/dev/null 2>&1; then",
                "  ufw allow OpenSSH || ufw allow 22/tcp",
                "elif command -v firewall-cmd >/dev/null 2>&1; then",
                "  firewall-cmd --add-service=ssh --permanent",
                "  firewall-cmd --reload",
                "else",
                "  echo 'No supported firewall helper was found; allow TCP/22 manually "
                "if remote SSH is blocked.'",
                "fi",
            ]
        )
    else:
        followups.append(
            "Firewall changes were not requested; remote SSH may still be blocked "
            "by host or network policy."
        )
    if enable_linger:
        lines.extend(
            [
                "if command -v loginctl >/dev/null 2>&1; then",
                f"  loginctl enable-linger {target_user_expr}",
                "else",
                "  echo 'loginctl is unavailable; configure user daemon persistence "
                "manually for this platform.'",
                "fi",
            ]
        )
    elif platform_name == "macos":
        followups.append(
            "macOS daemon auto-start is not configured here; use a LaunchAgent "
            "or service-managed hosting setup."
        )
    else:
        followups.append(
            "User daemon linger was not requested; detached user daemons may stop "
            "after logout on some systemd hosts."
        )
    lines.append("echo 'mp13 hosting admin setup complete'")
    return "\n".join(lines) + "\n", ".sh", followups


def _write_admin_setup_script(script: str, suffix: str) -> Path:
    temp_dir = Path(tempfile.gettempdir()) / "mp13-hosting-admin-setup"
    temp_dir.mkdir(parents=True, exist_ok=True)
    script_path = (temp_dir / f"admin_setup_{time.time_ns()}{suffix}").resolve()
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o700)
    return script_path


def _admin_setup_elevation_command(
    script_path: Path, *, platform_name: str
) -> Tuple[list[str], str]:
    if platform_name == "windows":
        quoted_path = str(script_path).replace("'", "''")
        return [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "Start-Process -FilePath PowerShell "
            f"-ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','{quoted_path}' "
            "-Verb RunAs -Wait",
        ], "windows_uac"
    if platform_name == "macos":
        quoted = shlex.quote(str(script_path))
        return [
            "osascript",
            "-e",
            f'do shell script "/bin/sh {quoted}" with administrator privileges',
        ], "macos_authorization"
    if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and shutil.which(
        "pkexec"
    ):
        return ["pkexec", "/bin/sh", str(script_path)], "pkexec"
    if shutil.which("sudo"):
        return ["sudo", "/bin/sh", str(script_path)], "sudo"
    raise RuntimeError(
        "No supported elevation tool found. Install pkexec/sudo or run the generated "
        "script as root."
    )


@dataclass(frozen=True)
class TransportAdminSetupRequest:
    enable_ssh_service: bool = True
    enable_firewall: bool = False
    enable_user_linger: bool = False
    target_user: str = ""
    execute: bool = False


class _Args:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.admin_setup_enable_ssh_service = bool(data.get("enable_ssh_service", True))
        self.admin_setup_enable_firewall = bool(data.get("enable_firewall", False))
        self.admin_setup_enable_user_linger = bool(data.get("enable_user_linger", False))
        self.admin_setup_target_user = str(data.get("target_user") or "").strip()


def _data(request: TransportAdminSetupRequest | Dict[str, Any]) -> Dict[str, Any]:
    return asdict(request) if isinstance(request, TransportAdminSetupRequest) else dict(request or {})


def plan_transport_admin_setup(request: TransportAdminSetupRequest | Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = _data(request or {})
    platform_name = _admin_setup_platform()
    script, suffix, followups = _admin_setup_script(_Args(data), platform_name=platform_name)
    return {
        "status": "dry_run",
        "action": "transport_admin_setup_plan",
        "platform": platform_name,
        "script_suffix": suffix,
        "script": script,
        "followups": followups,
        "execute": False,
    }


def execute_transport_admin_setup(request: TransportAdminSetupRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _data(request)
    if not bool(data.get("execute", False)):
        raise PermissionError("execute=True is required for elevated transport admin setup")
    plan = plan_transport_admin_setup(data)
    script_path = _write_admin_setup_script(str(plan.get("script") or ""), str(plan.get("script_suffix") or ".sh"))
    command, method = _admin_setup_elevation_command(script_path, platform_name=str(plan.get("platform") or ""))
    completed = subprocess.run(command, check=False)
    return {
        **plan,
        "status": "ok" if int(completed.returncode) == 0 else "elevation_failed",
        "action": "transport_admin_setup_execute",
        "execute": True,
        "script_file": str(script_path),
        "elevation_method": method,
        "returncode": int(completed.returncode),
    }


__all__ = [
    "TransportAdminSetupRequest",
    "plan_transport_admin_setup",
    "execute_transport_admin_setup",
]
