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


CONNECTIVITY_INTENT_GUIDANCE: Dict[str, Dict[str, str]] = {
    "local_only": {
        "intent": "Single host usage with no off-host clients.",
        "provides": "Lowest setup overhead. Optional no-auth is possible only in strict safe profile.",
        "precautions": "Keep loopback-only bind and exclusive endpoint mode when auth is disabled.",
    },
    "ssh_tunnel_only": {
        "intent": "Remote operators connect through SSH tunnel while daemon stays loopback-bound.",
        "provides": "Remote reachability without direct non-loopback daemon exposure.",
        "precautions": "Require auth, maintain SSH key hygiene, and verify tunnel endpoint controls.",
    },
    "truly_remote": {
        "intent": "Persistent direct/proxied remote serving for multiple remote clients.",
        "provides": "Full remote operations with role separation and explicit ingress controls.",
        "precautions": "Require auth, enforce strict role boundaries, and apply firewall/proxy policy.",
    },
}


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
        control_state = (config_dir / "hosting" / "access_control.json").resolve()
        return config_dir, control_state


def _hosting_root(default_config_dir: Path) -> Path:
    return (default_config_dir / "hosting").resolve()


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


def _prompt_menu(question: str, options: Dict[str, str], default: str) -> str:
    print(question)
    for key, label in options.items():
        print(f"  {key}) {label}")
    raw = input(f"Select [{default}]: ").strip().lower()
    if not raw:
        return default
    return raw if raw in options else default


def _wizard_choice_prompt(
    *,
    title: str,
    valid: set[str],
    current: str,
    allow_skip: bool = True,
) -> Tuple[str, str]:
    print(f"{title}")
    print(f"  options: {', '.join(sorted(valid))}")
    nav = " prev=p, skip=s, enter=keep/current" if allow_skip else " prev=p, enter=keep/current"
    raw = input(f"  current={current}; choose value ({nav}): ").strip().lower()
    if raw in {"p", "prev"}:
        return "prev", current
    if allow_skip and raw in {"s", "skip"}:
        return "skip", current
    if not raw:
        return "next", current
    if raw in valid:
        return "next", raw
    print(f"  invalid choice '{raw}', keeping current value")
    return "next", current


def _wizard_bool_prompt(*, title: str, current: bool, allow_skip: bool = True) -> Tuple[str, bool]:
    nav = " prev=p, skip=s, enter=keep/current" if allow_skip else " prev=p, enter=keep/current"
    raw = input(f"{title} current={'yes' if current else 'no'} ({nav}): ").strip().lower()
    if raw in {"p", "prev"}:
        return "prev", current
    if allow_skip and raw in {"s", "skip"}:
        return "skip", current
    if not raw:
        return "next", current
    if raw in {"y", "yes", "1", "true"}:
        return "next", True
    if raw in {"n", "no", "0", "false"}:
        return "next", False
    print(f"  invalid boolean '{raw}', keeping current value")
    return "next", current


def _wizard_text_prompt(
    *,
    title: str,
    current: str,
    allow_skip: bool = True,
) -> Tuple[str, str]:
    nav = " prev=p, skip=s, enter=keep/current" if allow_skip else " prev=p, enter=keep/current"
    raw = input(f"{title} current={current} ({nav}): ").strip()
    if raw.lower() in {"p", "prev"}:
        return "prev", current
    if allow_skip and raw.lower() in {"s", "skip"}:
        return "skip", current
    if not raw:
        return "next", current
    return "next", raw


def _summarize_existing_config(
    *,
    control_state_path: Path,
    access_file: Path,
    keys_file: Path,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "exists": False,
        "connectivity_mode": "local_only",
        "endpoint_mode_default": "exclusive",
        "lifecycle_profile": "detached_user_process",
        "require_auth": True,
        "admin_key_id": "admin-main",
        "admin_key_count": 0,
    }
    access_payload = _read_json(access_file, {})
    if "control_config" in access_payload and isinstance(access_payload.get("control_config"), dict):
        access_payload = dict(access_payload.get("control_config") or {})
    keys_payload = _read_json(keys_file, {"keys": {}})
    keys = dict(keys_payload.get("keys") or {})
    summary["admin_key_count"] = len([k for _, k in keys.items() if str((k or {}).get("role") or "") == "admin"])
    if access_payload:
        ap = dict(access_payload.get("access_profile") or {})
        summary["connectivity_mode"] = _normalize_mode(
            str(ap.get("connectivity_mode") or summary["connectivity_mode"]),
            summary["connectivity_mode"],
        )
        summary["endpoint_mode_default"] = _normalize_endpoint_mode(
            str(access_payload.get("endpoint_mode_default") or summary["endpoint_mode_default"]),
            summary["endpoint_mode_default"],
        )
        summary["lifecycle_profile"] = _normalize_lifecycle_profile(
            str(access_payload.get("lifecycle_profile") or summary["lifecycle_profile"]),
            summary["lifecycle_profile"],
        )
        summary["require_auth"] = bool(access_payload.get("require_auth", summary["require_auth"]))
        summary["exists"] = True
    try:
        svc = EngineHostService(control_state_file=control_state_path)
        cfg = dict(svc.get_control_config() or {})
        ap = dict(cfg.get("access_profile") or {})
        summary["connectivity_mode"] = _normalize_mode(
            str(ap.get("connectivity_mode") or summary["connectivity_mode"]),
            summary["connectivity_mode"],
        )
        summary["endpoint_mode_default"] = _normalize_endpoint_mode(
            str(cfg.get("endpoint_mode_default") or summary["endpoint_mode_default"]),
            summary["endpoint_mode_default"],
        )
        summary["lifecycle_profile"] = _normalize_lifecycle_profile(
            str(cfg.get("lifecycle_profile") or summary["lifecycle_profile"]),
            summary["lifecycle_profile"],
        )
        summary["require_auth"] = bool(cfg.get("require_auth", summary["require_auth"]))
        summary["exists"] = True
    except Exception:
        pass
    return summary


def _print_intent_guidance(mode: str) -> None:
    g = dict(CONNECTIVITY_INTENT_GUIDANCE.get(mode) or {})
    print(f"Intent `{mode}`:")
    print(f"  - usage: {str(g.get('intent') or 'n/a')}")
    print(f"  - value: {str(g.get('provides') or 'n/a')}")
    print(f"  - precautions: {str(g.get('precautions') or 'n/a')}")


def _apply_permission_hardening(paths: list[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"attempted": [], "errors": []}
    for p in paths:
        try:
            if p.exists():
                mode = 0o700 if p.is_dir() else 0o600
                p.chmod(mode)
                out["attempted"].append({"path": str(p), "mode": oct(mode)})
        except Exception as exc:
            out["errors"].append({"path": str(p), "error": str(exc)})
    return out


def _ensure_dirs(hosting_root: Path) -> Dict[str, Path]:
    paths = {
        "root": hosting_root,
        "keyring": hosting_root / "keyring",
        "audit": hosting_root / "audit",
        "state": hosting_root / "state",
        "bootstrap": hosting_root / "bootstrap",
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
    passphrase: Optional[str],
) -> Tuple[str, str]:
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

    private_text = ""
    public_text = ""
    tmpdir = Path(tempfile.mkdtemp(prefix="hosting_keygen_")).resolve()
    try:
        tmp_private = (tmpdir / f"{key_id}_ed25519").resolve()
        tmp_public = Path(str(tmp_private) + ".pub")
        _run_ssh_keygen(tmp_private)
        if not tmp_private.exists() or not tmp_public.exists():
            raise RuntimeError("ssh-keygen did not produce expected key files")
        private_text = str(tmp_private.read_text(encoding="utf-8")).strip()
        public_text = str(tmp_public.read_text(encoding="utf-8")).strip()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if not private_text or not public_text:
        raise RuntimeError("failed to generate importable key material")
    return private_text, public_text


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


def _store_importable_key_record(
    *,
    keys_file: Path,
    key_id: str,
    role: str,
    auth_method: str,
    public_key: str,
    private_key_openssh: Optional[str] = None,
    key_source: Optional[str] = None,
) -> None:
    payload = _read_json(keys_file, {"version": 1, "keys": {}})
    keys = dict(payload.get("keys") or {})
    existing = dict(keys.get(key_id) or {})
    row = {
        "role": str(role or "").strip(),
        "auth_method": str(auth_method or "").strip(),
        "public_key": str(public_key or "").strip(),
    }
    if private_key_openssh:
        row["private_key_openssh"] = str(private_key_openssh).strip()
    if key_source:
        row["key_source"] = str(key_source).strip()
    preserved = {
        str(k): v
        for k, v in existing.items()
        if str(k) not in {"role", "auth_method", "public_key", "private_key_openssh", "key_source"}
    }
    keys[str(key_id)] = preserved | row
    payload["version"] = 1
    payload["updated_at"] = time.time()
    payload["keys"] = keys
    _write_json(keys_file, payload)


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
    mappings_file = dirs["bootstrap"] / "client_key_map.json"
    bootstrap_state_file = dirs["bootstrap"] / "bootstrap_state.json"
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
    key_action = "replace"
    permission_action = "none"
    setup_scope = "fresh_setup"
    setup_notes: list[str] = []
    permission_result: Dict[str, Any] = {"attempted": [], "errors": []}

    existing_summary = _summarize_existing_config(
        control_state_path=control_state_path,
        access_file=access_file,
        keys_file=keys_file,
    )

    if interactive:
        assumed_intent = _normalize_mode(existing_summary.get("connectivity_mode", mode), mode)
        print("\n=== Hosting Access Setup/Reconfigure ===")
        print(f"Assumed user intent: {assumed_intent}")
        _print_intent_guidance(assumed_intent)
        print("\nCurrent config snapshot:")
        print(f"  - config exists: {'yes' if existing_summary.get('exists') else 'no'}")
        print(f"  - connectivity_mode: {existing_summary.get('connectivity_mode')}")
        print(f"  - endpoint_mode_default: {existing_summary.get('endpoint_mode_default')}")
        print(f"  - lifecycle_profile: {existing_summary.get('lifecycle_profile')}")
        print(f"  - require_auth: {bool(existing_summary.get('require_auth'))}")
        print(f"  - bootstrap admin key_id: {existing_summary.get('admin_key_id')}")
        print(f"  - admin key entries: {existing_summary.get('admin_key_count')}")

        mode = assumed_intent
        endpoint_mode = _normalize_endpoint_mode(
            str(existing_summary.get("endpoint_mode_default") or endpoint_mode),
            endpoint_mode,
        )
        lifecycle_profile = _normalize_lifecycle_profile(
            str(existing_summary.get("lifecycle_profile") or lifecycle_profile),
            lifecycle_profile,
        )
        require_auth_seed = bool(existing_summary.get("require_auth", True))
        admin_key_id = str(existing_summary.get("admin_key_id") or admin_key_id)
        if bool(existing_summary.get("exists")):
            key_action = "keep_existing"

        workflow_choice = _prompt_menu(
            "\nChoose workflow path:",
            {
                "1": "Adjust within current intent (config tweaks, key handling, permission hardening)",
                "2": "Complete reconfiguration under a new intent (full guided steps)",
            },
            "1" if bool(existing_summary.get("exists")) else "2",
        )
        if workflow_choice == "2":
            setup_scope = "full_reconfigure_new_intent"
            print("\nConnectivity intent choices:")
            for k in sorted(VALID_CONNECTIVITY_MODES):
                _print_intent_guidance(k)
            mode = _prompt_choice("Connectivity mode", VALID_CONNECTIVITY_MODES, mode)
            setup_notes.append("Full reconfigure selected: all grouped steps reviewed.")
        else:
            setup_scope = "adjust_within_intent"
            setup_notes.append("Within-intent adjustment selected.")

        grouped_steps = [
            "endpoint_mode",
            "lifecycle_profile",
            "require_auth",
            "key_action",
            "key_source",
            "admin_key_id",
            "permission_action",
        ]
        step_idx = 0
        current_require_auth = _safe_require_auth(
            connectivity_mode=mode,
            endpoint_mode=endpoint_mode,
            requested=args.require_auth if args.require_auth is not None else require_auth_seed,
        )
        print("\nGrouped configuration steps (type `p` for previous step, `s` to skip current step):")
        while step_idx < len(grouped_steps):
            step = grouped_steps[step_idx]
            if step == "endpoint_mode":
                print("\n[Group: Access envelope]")
                cmd, val = _wizard_choice_prompt(
                    title="Step 1: Endpoint mode",
                    valid=VALID_ENDPOINT_MODES,
                    current=endpoint_mode,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next":
                    endpoint_mode = val
                    current_require_auth = _safe_require_auth(
                        connectivity_mode=mode,
                        endpoint_mode=endpoint_mode,
                        requested=current_require_auth,
                    )
                step_idx += 1
                continue
            if step == "lifecycle_profile":
                cmd, val = _wizard_choice_prompt(
                    title="Step 2: Lifecycle profile",
                    valid=VALID_LIFECYCLE_PROFILES,
                    current=lifecycle_profile,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next":
                    lifecycle_profile = val
                step_idx += 1
                continue
            if step == "require_auth":
                print(
                    "Step 3: Require auth\n"
                    "  - value: protects multi-user and remote/tunnel workflows.\n"
                    "  - constraint: no-auth allowed only for local_only + exclusive."
                )
                cmd, val = _wizard_bool_prompt(
                    title="Enable require_auth?",
                    current=current_require_auth,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next":
                    current_require_auth = _safe_require_auth(
                        connectivity_mode=mode,
                        endpoint_mode=endpoint_mode,
                        requested=val,
                    )
                step_idx += 1
                continue
            if step == "key_action":
                print("\n[Group: Key management]")
                cmd, val = _wizard_choice_prompt(
                    title="Step 4: Key handling action",
                    valid={"keep_existing", "replace"},
                    current=key_action,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next":
                    key_action = val
                step_idx += 1
                continue
            if step == "key_source":
                if key_action == "keep_existing":
                    step_idx += 1
                    continue
                cmd, val = _wizard_choice_prompt(
                    title="Step 5: Key source for replacement",
                    valid=VALID_KEY_SOURCES,
                    current=key_source,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next":
                    key_source = val
                step_idx += 1
                continue
            if step == "admin_key_id":
                cmd, val = _wizard_text_prompt(
                    title="Step 6: Admin key_id",
                    current=admin_key_id,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next" and str(val).strip():
                    admin_key_id = str(val).strip()
                step_idx += 1
                continue
            if step == "permission_action":
                print("\n[Group: Permission hardening]")
                print("  - none: keep filesystem permissions unchanged")
                print("  - tighten: best-effort chmod on Hosting folders/files")
                cmd, val = _wizard_choice_prompt(
                    title="Step 7: Permission action",
                    valid={"none", "tighten"},
                    current=permission_action,
                )
                if cmd == "prev":
                    step_idx = max(0, step_idx - 1)
                    continue
                if cmd == "next":
                    permission_action = val
                step_idx += 1
                continue

        require_auth = current_require_auth
        print("\nPlanned result:")
        print(f"  - workflow: {setup_scope}")
        print(f"  - connectivity_mode: {mode}")
        print(f"  - endpoint_mode_default: {endpoint_mode}")
        print(f"  - lifecycle_profile: {lifecycle_profile}")
        print(f"  - require_auth: {require_auth}")
        print(f"  - key_action: {key_action}")
        if key_action != "keep_existing":
            print(f"  - key_source: {key_source}")
        print(f"  - admin_key_id: {admin_key_id}")
        print(f"  - permission_action: {permission_action}")
        _print_intent_guidance(mode)
        if not _bool_prompt("Apply this configuration now?", True):
            raise RuntimeError("interactive setup cancelled by user")
    else:
        require_auth = _safe_require_auth(
            connectivity_mode=mode,
            endpoint_mode=endpoint_mode,
            requested=args.require_auth,
        )

    migration_result = _migrate_legacy_key_files(
        default_config_dir=default_config_dir,
        hosting_root=hosting_root,
        audit_file=audit_file,
        migrations_file=migrations_file,
    )

    admin_public_key = ""
    admin_private_key_text: Optional[str] = None
    admin_public_key_path: Optional[Path] = None
    export_private = bool(args.export_private_key)
    export_private_path = (
        Path(str(args.export_private_key_path)).expanduser().resolve()
        if str(args.export_private_key_path or "").strip()
        else None
    )

    if key_action == "keep_existing":
        keyring_existing = _read_json(keys_file, {"keys": {}})
        existing_keys = dict(keyring_existing.get("keys") or {})
        row = dict(existing_keys.get(admin_key_id) or {})
        admin_public_key = str(row.get("public_key") or "").strip()
        if not admin_public_key:
            raise ValueError(
                f"key_action=keep_existing requested but key_id={admin_key_id} has no existing public key"
            )
        key_source = "import"
    else:
        if key_source == "generate":
            passphrase = str(args.key_passphrase or "")
            if interactive and not args.key_passphrase:
                if _bool_prompt("Protect generated private key with passphrase?", False):
                    passphrase = input("Passphrase: ")
            generated_private, generated_public = _generate_keypair(
                key_id=admin_key_id,
                passphrase=passphrase or None,
            )
            admin_private_key_text = generated_private
            admin_public_key = str(generated_public).strip()
            if interactive:
                export_private = _bool_prompt("Export generated private key for client use?", export_private)
            if export_private and export_private_path is not None:
                export_private_path.parent.mkdir(parents=True, exist_ok=True)
                export_private_path.write_text(str(generated_private), encoding="utf-8")
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
    _store_importable_key_record(
        keys_file=keys_file,
        key_id=admin_key_id,
        role="admin",
        auth_method="public_key",
        public_key=admin_public_key,
        private_key_openssh=admin_private_key_text,
        key_source=key_source,
    )

    if permission_action == "tighten":
        permission_result = _apply_permission_hardening(
            [
                dirs["root"],
                dirs["keyring"],
                dirs["audit"],
                dirs["state"],
                dirs["bootstrap"],
                access_file,
                keys_file,
                mappings_file,
                bootstrap_state_file,
                audit_file,
            ]
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
                "setup_scope": setup_scope,
                "setup_notes": setup_notes,
                "connectivity_mode": mode,
                "endpoint_mode_default": endpoint_mode,
                "lifecycle_profile": lifecycle_profile,
                "require_auth": require_auth,
                "admin_key_id": admin_key_id,
                "key_source": key_source,
                "key_action": key_action,
                "permission_action": permission_action,
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
        "admin_private_key_path": None,
        "private_key_exported": bool(export_private),
        "private_key_export_path": str(export_private_path) if export_private_path else None,
        "setup_scope": setup_scope,
        "key_action": key_action,
        "permission_action": permission_action,
        "permission_result": permission_result,
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

    def _record(
        name: str,
        ok: bool,
        details: Optional[Dict[str, Any]] = None,
        *,
        blocking: bool = True,
    ) -> None:
        entry = {"check": name, "ok": bool(ok), "details": dict(details or {})}
        checks.append(entry)
        if (not ok) and bool(blocking):
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

    # Readiness probe for Windows/mapped-path keygen behavior.
    # This check is non-blocking for baseline setup because key import remains valid,
    # but it must be reviewed before rotation-heavy hardening work.
    key_probe_dir = (hosting_root / "keyring").resolve()
    key_probe_private = (key_probe_dir / ".doctor_keygen_probe_ed25519").resolve()
    key_probe_public = Path(str(key_probe_private) + ".pub")
    key_probe_details: Dict[str, Any] = {"path": str(key_probe_dir), "blocking": False}
    key_probe_ok = False
    try:
        key_probe_dir.mkdir(parents=True, exist_ok=True)
        key_probe_private.unlink(missing_ok=True)
        key_probe_public.unlink(missing_ok=True)
        probe = subprocess.run(  # noqa: S603
            [
                "ssh-keygen",
                "-t",
                "ed25519",
                "-C",
                "doctor-probe",
                "-f",
                str(key_probe_private),
                "-N",
                "",
            ],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        key_probe_details["returncode"] = int(probe.returncode)
        if int(probe.returncode) == 0 and key_probe_private.exists() and key_probe_public.exists():
            key_probe_ok = True
        else:
            key_probe_details["stderr"] = str(probe.stderr or "").strip()
    except Exception as exc:
        key_probe_details["error"] = str(exc)
    finally:
        key_probe_private.unlink(missing_ok=True)
        key_probe_public.unlink(missing_ok=True)
    _record("ssh_keygen_host_path_probe", key_probe_ok, key_probe_details, blocking=False)

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
