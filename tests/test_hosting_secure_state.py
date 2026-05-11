from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest

from hosting.daemon import EngineHostDaemon
from hosting.secure_state import (
    SecureStateFormatError,
    SecureStateLockedError,
    decrypt_secure_json_file,
    encrypt_secure_json_file,
    read_secure_json,
    rotate_secure_json_file,
    secure_state_status,
    write_secure_json,
)


def test_secure_state_json_encrypt_decrypt_rotate_and_detect(tmp_path: Path) -> None:
    path = tmp_path / "backend_users.json"
    write_secure_json(path, {"users": {"alice": {"role": "admin"}}}, encrypt=False)
    plaintext_status = secure_state_status(path)
    assert plaintext_status["state"] == "plaintext"
    assert plaintext_status["encrypted"] is False

    encrypted_status = encrypt_secure_json_file(path, key="pw1", metadata={"owner": "gui"})
    assert encrypted_status["state"] == "encrypted"
    assert encrypted_status["encrypted"] is True
    assert encrypted_status["metadata"]["owner"] == "gui"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["kind"] == "mp13.secure_state.json"
    assert "alice" not in path.read_text(encoding="utf-8")

    with pytest.raises(SecureStateLockedError):
        read_secure_json(path)
    assert read_secure_json(path, key="pw1")["users"]["alice"]["role"] == "admin"

    rotated = rotate_secure_json_file(path, old_key="pw1", new_key="pw2")
    assert rotated["state"] == "encrypted"
    with pytest.raises(SecureStateLockedError):
        read_secure_json(path, key="pw1")
    assert read_secure_json(path, key="pw2")["users"]["alice"]["role"] == "admin"

    decrypted_status = decrypt_secure_json_file(path, key="pw2")
    assert decrypted_status["state"] == "plaintext"
    assert read_secure_json(path)["users"]["alice"]["role"] == "admin"


def test_secure_state_can_fail_closed_on_plaintext(tmp_path: Path) -> None:
    path = tmp_path / "backend_users.json"
    write_secure_json(path, {"users": {}}, encrypt=False)
    with pytest.raises(SecureStateFormatError, match="plaintext_secure_state_disallowed"):
        read_secure_json(path, allow_plaintext=False)


def test_daemon_reports_hosting_owned_secure_state_status(tmp_path: Path) -> None:
    daemon = EngineHostDaemon(
        port=0,
        pid_file=tmp_path / "daemon.pid",
        engines_state_file=tmp_path / "managed_engines.json",
        control_state_file=tmp_path / "access_control.json",
    )
    daemon.svc.set_control_config(require_auth=False)

    result = daemon.svc.hosting_setup_summary()

    assert result["status"] == "ok"
    assert result["hosting_root"] == str(tmp_path)
    assert result["secure_state"]["daemon_secure_state_read_enabled"] is False
    assert "access_control" in result["secure_state"]["files"]
    assert result["secure_state"]["files"]["access_control"]["state"] == "plaintext"

    raw = json.dumps({"seq": 1, "cmd": "hosting-secure-state-status", "payload": {}})
    dispatched = asyncio.run(daemon._dispatch(raw, peer_host="127.0.0.1"))  # noqa: SLF001
    assert dispatched["ok"] is True
    assert dispatched["result"]["daemon_secure_state_read_enabled"] is False
