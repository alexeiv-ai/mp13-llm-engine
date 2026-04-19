"""Daemon state file security helpers."""
from __future__ import annotations

import getpass
import json
import logging
import os
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _legacy_daemon_attr(name: str, fallback: Any) -> Any:
    module = sys.modules.get("hosting.engine_host_daemon")
    return getattr(module, name, fallback) if module is not None else fallback


def _current_windows_account_name() -> str:
    try:
        proc = subprocess.run(  # noqa: S603
            ["whoami"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        raw = str(proc.stdout or "").strip()
        if raw:
            return raw
    except Exception:
        pass
    domain = str(os.environ.get("USERDOMAIN") or "").strip()
    user = str(os.environ.get("USERNAME") or getpass.getuser() or "").strip()
    if domain and user:
        return f"{domain}\\{user}"
    return user


def _tighten_windows_acl(path: Path, *, is_dir: bool) -> None:
    principal = _legacy_daemon_attr("_current_windows_account_name", _current_windows_account_name)()
    if not principal:
        logger.warning("unable to determine current Windows account for ACL hardening")
        return
    grant_suffix = "(OI)(CI)F" if is_dir else "F"
    cmd = [
        "icacls",
        str(path),
        "/inheritance:r",
        "/grant:r",
        f"{principal}:{grant_suffix}",
        "SYSTEM:F" if not is_dir else "SYSTEM:(OI)(CI)F",
        "Administrators:F" if not is_dir else "Administrators:(OI)(CI)F",
    ]
    proc = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if int(proc.returncode) != 0:
        stderr = str(proc.stderr or "").strip()
        logger.warning("ACL hardening failed for %s: %s", path, stderr or "icacls error")


def _secure_state_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        _tighten_windows_acl(path.parent, is_dir=True)
        return
    os.chmod(path.parent, 0o700)


def _secure_path(path: Path) -> None:
    if os.name == "nt":
        _tighten_windows_acl(path, is_dir=False)
        return
    os.chmod(path, 0o600)


def _atomic_write_secure_json(path: Path, payload: Dict[str, Any]) -> None:
    _secure_state_parent_dir(path)
    raw = json.dumps(payload, indent=2)
    if os.name == "nt":
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(raw, encoding="utf-8")
        _secure_path(tmp_path)
        os.replace(tmp_path, path)
        _secure_path(path)
        return
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(raw)
        os.replace(tmp_name, path)
        _secure_path(path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        raise
