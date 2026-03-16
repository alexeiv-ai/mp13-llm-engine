"""
Interactive/non-interactive hosting access setup and reconfiguration utility.

Usage examples:
  python -m hosting.hosting_config --interactive
  python -m hosting.hosting_config --mode local_only --key-source import --admin-key-id admin-main --admin-public-key-file C:\\keys\\admin.pub
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .engine_host_service import EngineHostService


VALID_CONNECTIVITY_MODES = {"local_only", "ssh_tunnel_only", "truly_remote"}
VALID_ENDPOINT_MODES = {"exclusive", "shared"}
VALID_LIFECYCLE_PROFILES = {
    "foreground_terminal_bound",
    "detached_user_process",
    "service_managed",
}
VALID_KEY_SOURCES = {"generate", "import"}


def _default_paths() -> Tuple[Path, Path]:
    try:
        from mp13_engine.mp13_config_paths import (  # type: ignore
            get_default_config_dir,
            get_hosting_control_state_path,
        )

        config_dir = Path(get_default_config_dir()).expanduser().resolve()
        control_state = Path(get_hosting_control_state_path()).expanduser().resolve()
        return config_dir, control_state
    except Exception:
        config_dir = (Path.home() / ".mp13-llm").expanduser().resolve()
        control_state = (config_dir / "backend" / "engine_host_control.json").resolve()
        return config_dir, control_state


def _hosting_root(default_config_dir: Path) -> Path:
    return (default_config_dir / "Hosting").resolve()


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if not path.exists():
            return dict(default)
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else dict(default)
    except Exception:
        return dict(default)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_mode(value: str, default: str) -> str:
    v = str(value or "").strip().lower()
    return v if v in VALID_CONNECTIVITY_MODES else default


def _normalize_endpoint_mode(value: str, default: str) -> str:
    v = str(value or "").strip().lower()
    return v if v in VALID_ENDPOINT_MODES else default


def _normalize_lifecycle_profile(value: str, default: str) -> str:
    v = str(value or "").strip().lower()
    return v if v in VALID_LIFECYCLE_PROFILES else default


def _bool_prompt(question: str, default: bool) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = input(question + suffix).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"y", "yes", "1", "true"}


def _prompt_choice(question: str, valid: set[str], default: str) -> str:
    raw = input(f"{question} ({', '.join(sorted(valid))}) [{default}]: ").strip().lower()
    if not raw:
        return default
    return raw if raw in valid else default


def _ensure_dirs(hosting_root: Path) -> Dict[str, Path]:
    paths = {
        "root": hosting_root,
        "keyring": hosting_root / "keyring",
        "private": hosting_root / "keyring" / "private",
        "audit": hosting_root / "audit",
        "state": hosting_root / "state",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _import_public_key(*, public_key_file: Optional[str], public_key_inline: Optional[str]) -> str:
    if public_key_inline:
        return str(public_key_inline).strip()
    if public_key_file:
        p = Path(public_key_file).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"public key file not found: {p}")
        return str(p.read_text(encoding="utf-8")).strip()
    raise ValueError("public key is required (provide --admin-public-key-file or --admin-public-key)")


def _generate_keypair(
    *,
    key_id: str,
    private_dir: Path,
    passphrase: Optional[str],
) -> Tuple[Path, Path]:
    def _run_ssh_keygen(dest_private: Path) -> None:
        cmd = [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-C",
            key_id,
            "-f",
            str(dest_private),
            "-N",
            str(passphrase or ""),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30.0)  # noqa: S603
        if int(proc.returncode) != 0:
            stderr = str(proc.stderr or "").strip()
            raise RuntimeError(f"ssh-keygen failed: {stderr or 'unknown error'}")

    private_path = (private_dir / f"{key_id}_ed25519").resolve()
    public_path = Path(str(private_path) + ".pub")
    if private_path.exists() or public_path.exists():
        raise ValueError(f"key files already exist for key_id={key_id}: {private_path}")
    private_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate in a local temporary directory first, then copy into Hosting keyring.
    # This avoids ssh-keygen write failures on some mapped/network filesystems.
    tmp_error: Optional[Exception] = None
    tmpdir = Path(tempfile.mkdtemp(prefix="hosting_keygen_")).resolve()
    try:
        tmp_private = (tmpdir / f"{key_id}_ed25519").resolve()
        tmp_public = Path(str(tmp_private) + ".pub")
        _run_ssh_keygen(tmp_private)
        if not tmp_private.exists() or not tmp_public.exists():
            raise RuntimeError("ssh-keygen did not produce expected key files")
        shutil.copy2(str(tmp_private), str(private_path))
        shutil.copy2(str(tmp_public), str(public_path))
    except Exception as exc:
        tmp_error = exc
    finally:
        # Some Windows OpenSSH builds set key ACLs that can block recursive delete.
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Fallback to direct generation if temp path cannot be used in constrained runtime.
    if not private_path.exists() or not public_path.exists():
        try:
            _run_ssh_keygen(private_path)
        except Exception as direct_exc:
            if tmp_error is not None:
                raise RuntimeError(f"{direct_exc}; temp_keygen_fallback_error={tmp_error}") from direct_exc
            raise

    if not private_path.exists() or not public_path.exists():
        raise RuntimeError("failed to persist generated key files")
    return private_path, public_path


def _build_access_control_payload(
    *,
    connectivity_mode: str,
    endpoint_mode: str,
    lifecycle_profile: str,
    require_auth: bool,
    admin_key_id: str,
    admin_key_origin: str,
) -> Dict[str, Any]:
    now = time.time()
    return {
        "version": 1,
        "updated_at": now,
        "access_profile": {"connectivity_mode": connectivity_mode},
        "endpoint_mode_default": endpoint_mode,
        "lifecycle_profile": lifecycle_profile,
        "require_auth": bool(require_auth),
        "bootstrap_admin_key_id": admin_key_id,
        "bootstrap_admin_key_origin": admin_key_origin,
    }


def _safe_require_auth(
    *,
    connectivity_mode: str,
    endpoint_mode: str,
    requested: Optional[bool],
) -> bool:
    if requested is None:
        return True
    val = bool(requested)
    if val:
        return True
    # Safe-only profile for unauth mode.
    if connectivity_mode == "local_only" and endpoint_mode == "exclusive":
        return False
    raise ValueError(
        "require_auth=false is only allowed for local_only connectivity with exclusive endpoint mode"
    )


def _write_audit_event(audit_file: Path, event: Dict[str, Any]) -> None:
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(event or {}), ensure_ascii=False)
    with audit_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _migrate_legacy_key_files(
    *,
    default_config_dir: Path,
    hosting_root: Path,
    audit_file: Path,
    migrations_file: Path,
) -> Dict[str, Any]:
    candidates = [
        (default_config_dir / "backend" / "host_auth_keys.json").resolve(),
        (default_config_dir / "backend" / "engine_host_auth_keys.json").resolve(),
        (hosting_root / "keys.json").resolve(),
    ]
    migrated: list[Dict[str, Any]] = []
    for source in candidates:
        if not source.exists() or not source.is_file():
            continue
        target = Path(str(source) + ".migrated")
        if target.exists():
            idx = 1
            while True:
                alt = Path(str(source) + f".migrated.{idx}")
                if not alt.exists():
                    target = alt
                    break
                idx += 1
        source.rename(target)
        evt = {
            "timestamp": time.time(),
            "event": "legacy_key_file_renamed",
            "source": str(source),
            "target": str(target),
        }
        migrated.append(evt)
        _write_audit_event(audit_file, evt)
    if migrated:
        payload = _read_json(migrations_file, {"version": 1, "migrations": []})
        rows = list(payload.get("migrations") or [])
        rows.extend(migrated)
        payload["version"] = 1
        payload["updated_at"] = time.time()
        payload["migrations"] = rows
        _write_json(migrations_file, payload)
    return {"migrated_count": len(migrated), "migrated": migrated}


def run_setup(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_dir, default_control_state_path = _default_paths()
    if str(args.default_config_dir or "").strip():
        default_config_dir = Path(str(args.default_config_dir)).expanduser().resolve()
    control_state_path = default_control_state_path
    if str(args.control_state_file or "").strip():
        control_state_path = Path(str(args.control_state_file)).expanduser().resolve()
    hosting_root = _hosting_root(default_config_dir)
    dirs = _ensure_dirs(hosting_root)
    access_file = dirs["root"] / "access_control.json"
    keys_file = dirs["keyring"] / "keys.json"
    mappings_file = dirs["state"] / "client_key_map.json"
    bootstrap_state_file = dirs["state"] / "bootstrap_state.json"
    audit_file = dirs["audit"] / "setup_audit.jsonl"
    migrations_file = dirs["keyring"] / "migrations.json"

    interactive = bool(args.interactive)
    mode = _normalize_mode(args.mode, "local_only")
    endpoint_mode = _normalize_endpoint_mode(args.endpoint_mode, "exclusive")
    lifecycle_profile = _normalize_lifecycle_profile(args.lifecycle_profile, "detached_user_process")
    key_source = str(args.key_source or "").strip().lower() or "import"
    if key_source not in VALID_KEY_SOURCES:
        key_source = "import"
    admin_key_id = str(args.admin_key_id or "").strip() or "admin-main"

    if interactive:
        mode = _prompt_choice("Connectivity mode", VALID_CONNECTIVITY_MODES, mode)
        endpoint_mode = _prompt_choice("Endpoint mode", VALID_ENDPOINT_MODES, endpoint_mode)
        lifecycle_profile = _prompt_choice(
            "Lifecycle profile",
            VALID_LIFECYCLE_PROFILES,
            lifecycle_profile,
        )
        key_source = _prompt_choice("Admin key source", VALID_KEY_SOURCES, key_source)
        entered_key_id = input(f"Admin key_id [{admin_key_id}]: ").strip()
        if entered_key_id:
            admin_key_id = entered_key_id

    require_auth = _safe_require_auth(
        connectivity_mode=mode,
        endpoint_mode=endpoint_mode,
        requested=args.require_auth,
    )
    if interactive and _bool_prompt("Enable require_auth?", require_auth) != require_auth:
        # Interactive toggle must still obey safe rules.
        require_auth = _safe_require_auth(
            connectivity_mode=mode,
            endpoint_mode=endpoint_mode,
            requested=not require_auth,
        )

    migration_result = _migrate_legacy_key_files(
        default_config_dir=default_config_dir,
        hosting_root=hosting_root,
        audit_file=audit_file,
        migrations_file=migrations_file,
    )

    admin_public_key = ""
    admin_private_key_path: Optional[Path] = None
    admin_public_key_path: Optional[Path] = None
    export_private = bool(args.export_private_key)
    export_private_path = (
        Path(str(args.export_private_key_path)).expanduser().resolve()
        if str(args.export_private_key_path or "").strip()
        else None
    )

    if key_source == "generate":
        passphrase = str(args.key_passphrase or "")
        if interactive and not args.key_passphrase:
            if _bool_prompt("Protect generated private key with passphrase?", False):
                passphrase = input("Passphrase: ")
        generated_private, generated_public = _generate_keypair(
            key_id=admin_key_id,
            private_dir=dirs["private"],
            passphrase=passphrase or None,
        )
        admin_private_key_path = generated_private
        admin_public_key_path = generated_public
        admin_public_key = str(generated_public.read_text(encoding="utf-8")).strip()
        if interactive:
            export_private = _bool_prompt("Export generated private key for client use?", export_private)
        if export_private and export_private_path is not None:
            export_private_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(generated_private), str(export_private_path))
    else:
        admin_public_key = _import_public_key(
            public_key_file=args.admin_public_key_file,
            public_key_inline=args.admin_public_key,
        )

    svc = EngineHostService(control_state_file=control_state_path)
    _ = svc.auth_upsert_key(
        key_id=admin_key_id,
        auth_method="public_key",
        public_key=admin_public_key,
        role="admin",
        disabled=False,
    )
    _ = svc.set_control_config(
        require_auth=require_auth,
        access_profile={"connectivity_mode": mode},
        endpoint_mode_default=endpoint_mode,
        lifecycle_profile=lifecycle_profile,
    )

    keyring = _read_json(keys_file, {"version": 1, "keys": {}})
    keys = dict(keyring.get("keys") or {})
    keys[admin_key_id] = {
        "role": "admin",
        "auth_method": "public_key",
        "public_key": admin_public_key,
        "private_key_managed_path": str(admin_private_key_path) if admin_private_key_path else None,
        "private_key_exported": bool(export_private),
        "private_key_export_path": str(export_private_path) if export_private_path else None,
        "updated_at": time.time(),
    }
    keyring["version"] = 1
    keyring["updated_at"] = time.time()
    keyring["keys"] = keys
    _write_json(keys_file, keyring)

    _write_json(
        access_file,
        _build_access_control_payload(
            connectivity_mode=mode,
            endpoint_mode=endpoint_mode,
            lifecycle_profile=lifecycle_profile,
            require_auth=require_auth,
            admin_key_id=admin_key_id,
            admin_key_origin=key_source,
        ),
    )
    _write_json(
        mappings_file,
        {
            "version": 1,
            "updated_at": time.time(),
            "clients": [
                {
                    "client_id": "default-admin-client",
                    "key_id": admin_key_id,
                    "role": "admin",
                    "engine_host_session_scope": "control",
                    "engine_host_session_ttl_seconds": 900,
                    "connectivity_mode": mode,
                    "notes": [
                        "Set engine_host_key_id in client profile/config",
                        "Issue short-lived session token for runtime access",
                    ],
                }
            ],
        },
    )
    _write_json(
        bootstrap_state_file,
        {
            "version": 1,
            "updated_at": time.time(),
            "setup": {
                "connectivity_mode": mode,
                "endpoint_mode_default": endpoint_mode,
                "lifecycle_profile": lifecycle_profile,
                "require_auth": require_auth,
                "admin_key_id": admin_key_id,
                "key_source": key_source,
            },
            "files": {
                "control_state_file": str(control_state_path),
                "access_control_file": str(access_file),
                "keys_file": str(keys_file),
                "client_mapping_file": str(mappings_file),
                "audit_file": str(audit_file),
            },
            "legacy_migration": migration_result,
        },
    )
    _write_audit_event(
        audit_file,
        {
            "timestamp": time.time(),
            "event": "hosting_config_applied",
            "connectivity_mode": mode,
            "endpoint_mode_default": endpoint_mode,
            "lifecycle_profile": lifecycle_profile,
            "require_auth": require_auth,
            "admin_key_id": admin_key_id,
            "key_source": key_source,
        },
    )
    return {
        "status": "ok",
        "hosting_root": str(hosting_root),
        "control_state_file": str(control_state_path),
        "access_control_file": str(access_file),
        "keys_file": str(keys_file),
        "client_mapping_file": str(mappings_file),
        "bootstrap_state_file": str(bootstrap_state_file),
        "connectivity_mode": mode,
        "endpoint_mode_default": endpoint_mode,
        "lifecycle_profile": lifecycle_profile,
        "require_auth": require_auth,
        "admin_key_id": admin_key_id,
        "legacy_migration": migration_result,
        "admin_public_key_path": str(admin_public_key_path) if admin_public_key_path else None,
        "admin_private_key_path": str(admin_private_key_path) if admin_private_key_path else None,
        "private_key_exported": bool(export_private),
        "private_key_export_path": str(export_private_path) if export_private_path else None,
    }


def run_doctor(args: argparse.Namespace) -> Dict[str, Any]:
    default_config_dir, default_control_state_path = _default_paths()
    if str(args.default_config_dir or "").strip():
        default_config_dir = Path(str(args.default_config_dir)).expanduser().resolve()
    control_state_path = default_control_state_path
    if str(args.control_state_file or "").strip():
        control_state_path = Path(str(args.control_state_file)).expanduser().resolve()
    hosting_root = _hosting_root(default_config_dir)
    issues: list[Dict[str, Any]] = []
    checks: list[Dict[str, Any]] = []

    def _record(name: str, ok: bool, details: Optional[Dict[str, Any]] = None) -> None:
        entry = {"check": name, "ok": bool(ok), "details": dict(details or {})}
        checks.append(entry)
        if not ok:
            issues.append(entry)

    try:
        proc = subprocess.run(  # noqa: S603
            ["ssh-keygen", "-h"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        _record("ssh_dependency", proc.returncode in {0, 1})
    except Exception as exc:
        _record("ssh_dependency", False, {"error": str(exc)})

    _record("default_config_dir_exists", default_config_dir.exists(), {"path": str(default_config_dir)})
    _record("hosting_root_exists", hosting_root.exists(), {"path": str(hosting_root)})
    _record("control_state_exists", control_state_path.exists(), {"path": str(control_state_path)})

    # Write-check in hosting root if present.
    if hosting_root.exists():
        probe = hosting_root / ".doctor_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            _record("hosting_root_writable", True, {"path": str(hosting_root)})
        except Exception as exc:
            _record("hosting_root_writable", False, {"path": str(hosting_root), "error": str(exc)})
    else:
        _record("hosting_root_writable", False, {"path": str(hosting_root), "error": "missing_directory"})

    try:
        svc = EngineHostService(control_state_file=control_state_path)
        cfg = svc.get_control_config()
        _record("control_config_readable", True, {"require_auth": bool(cfg.get("require_auth", False))})
        try:
            svc.assert_runtime_policy_safe()
            _record("runtime_policy_safe", True)
        except Exception as exc:
            _record("runtime_policy_safe", False, {"error": str(exc)})
    except Exception as exc:
        _record("control_config_readable", False, {"error": str(exc)})

    return {
        "status": "ok" if not issues else "issues_found",
        "issues_count": len(issues),
        "checks": checks,
        "issues": issues,
        "default_config_dir": str(default_config_dir),
        "hosting_root": str(hosting_root),
        "control_state_file": str(control_state_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Configure hosting access and keyring state")
    p.add_argument("--default-config-dir", default="", help="Override default config root directory")
    p.add_argument("--control-state-file", default="", help="Override engine host control state JSON path")
    p.add_argument("--doctor", action="store_true", help="Run diagnostics without mutating configuration")
    p.add_argument("--interactive", action="store_true", help="Run interactive setup wizard")
    p.add_argument("--mode", default="local_only", choices=sorted(VALID_CONNECTIVITY_MODES))
    p.add_argument("--endpoint-mode", default="exclusive", choices=sorted(VALID_ENDPOINT_MODES))
    p.add_argument("--lifecycle-profile", default="detached_user_process", choices=sorted(VALID_LIFECYCLE_PROFILES))
    p.add_argument(
        "--require-auth",
        dest="require_auth",
        action="store_true",
        default=None,
        help="Enable daemon auth requirement",
    )
    p.add_argument(
        "--no-require-auth",
        dest="require_auth",
        action="store_false",
        help="Disable daemon auth requirement (safe-profile only)",
    )
    p.add_argument("--key-source", default="import", choices=sorted(VALID_KEY_SOURCES))
    p.add_argument("--admin-key-id", default="admin-main")
    p.add_argument("--admin-public-key-file", default="")
    p.add_argument("--admin-public-key", default="")
    p.add_argument("--key-passphrase", default="")
    p.add_argument("--export-private-key", action="store_true", default=False)
    p.add_argument("--export-private-key-path", default="")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run_doctor(args) if bool(args.doctor) else run_setup(args)
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
