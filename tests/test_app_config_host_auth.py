from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "app" / "config.py"
_SPEC = importlib.util.spec_from_file_location("app_config_module", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load config module from {_MODULE_PATH}")
app_config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(app_config)


def _run_main(argv: list[str]) -> int:
    old = list(sys.argv)
    try:
        sys.argv = argv
        return int(app_config.main())
    finally:
        sys.argv = old


def test_host_auth_status_and_upsert_key(tmp_path: Path, capsys) -> None:
    control_state = tmp_path / "access_control.json"

    rc = _run_main(
        [
            "mp13config",
            "--host-control-state-file",
            str(control_state),
            "--host-auth-upsert-key",
            "mgmt1",
            "--host-auth-role",
            "admin",
            "--host-auth-secret",
            "secret1",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["key_id"] == "mgmt1"

    rc = _run_main(
        [
            "mp13config",
            "--host-control-state-file",
            str(control_state),
            "--host-auth-status",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert int(payload["keys_count"]) >= 1


def test_host_auth_issue_session_prints_remote_shared_secret_guidance(tmp_path: Path, capsys) -> None:
    control_state = tmp_path / "access_control.json"

    rc = _run_main(
        [
            "mp13config",
            "--host-control-state-file",
            str(control_state),
            "--host-auth-upsert-key",
            "mgmt1",
            "--host-auth-role",
            "admin",
            "--host-auth-secret",
            "secret1",
        ]
    )
    assert rc == 0
    _ = capsys.readouterr()

    payload = json.loads(control_state.read_text(encoding="utf-8"))
    cfg = dict(payload.get("control_config") or {})
    cfg["require_auth"] = True
    cfg["access_profile"] = {"connectivity_mode": "ssh_tunnel_only"}
    payload["control_config"] = cfg
    control_state.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rc = _run_main(
        [
            "mp13config",
            "--host-control-state-file",
            str(control_state),
            "--host-auth-issue-session",
            "mgmt1",
            "--host-auth-secret",
            "secret1",
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "local_only for shared-secret keys" in out
