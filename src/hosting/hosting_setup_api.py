"""Stable host-local hosting setup API.

This is the integration contract for backend bootstrap/materialization code.
It is host-local only: callers must be running on the machine whose hosting
configuration files are being inspected or changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

from . import hosting_config_cli as _cli
from mp13_engine.mp13_config_paths import (
    DEFAULT_CATEGORY_DIRS,
    HOSTING_CATEGORY_ROOT_KEYS,
    get_default_config_path,
    load_json_config,
    resolve_config_paths,
)


HOSTING_SETUP_CONTRACT = "hosting.setup.v1"
HOSTING_SETUP_RESULT_CONTRACT = "hosting.setup.result.v1"
_ROOT_FIELDS = tuple(sorted(HOSTING_CATEGORY_ROOT_KEYS))


@dataclass(frozen=True)
class LocalHostingSetupRequest:
    host_local: bool = True
    default_config_dir: Optional[Path] = None
    hosting_root: Optional[Path] = None
    contract: str = HOSTING_SETUP_CONTRACT
    operation: str = ""
    mp13_config_file: Optional[Path] = None
    roots: Optional[Dict[str, str]] = None
    hosting_configuration: Optional[Dict[str, Any]] = None
    expected_config_revision: str = ""
    expected_hosting_revision: str = ""
    allow_nonempty_destinations: bool = False
    allow_cross_volume: bool = False
    confirm: bool = False
    mode: str = "local_only"
    endpoint_mode: str = "exclusive"
    lifecycle_profile: str = "detached_user_process"
    require_auth: Optional[bool] = None
    usage_intent: str = "single_admin"
    key_action: str = "replace"
    key_source: str = "generate"
    admin_key_id: str = "admin-main"
    admin_public_key: str = ""
    admin_public_key_file: str = ""
    generated_key_passphrase: str = ""
    client_secret_password: str = ""
    permission_action: str = "none"
    print_private_key_handoff: bool = False
    confirm_reset: bool = False


def _data(request: LocalHostingSetupRequest | Dict[str, Any] | None) -> Dict[str, Any]:
    if request is None:
        return {}
    return asdict(request) if isinstance(request, LocalHostingSetupRequest) else dict(request or {})


def _args(request: LocalHostingSetupRequest | Dict[str, Any] | None) -> Any:
    data = _data(request)
    args = _cli._build_parser().parse_args([])
    args.interactive = False
    args.json_output = False
    args.no_interactive = True
    for key, value in data.items():
        if value is None:
            continue
        if key == "hosting_root":
            if not data.get("default_config_dir"):
                setattr(args, "default_config_dir", str(Path(value).expanduser().resolve().parent))
            continue
        if isinstance(value, Path):
            value = str(value)
        setattr(args, key, value)
    return args


def _require_host_local(data: Dict[str, Any]) -> None:
    if not bool(data.get("host_local", True)):
        raise PermissionError("hosting setup API is host-local only")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _revision(value: Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _config_file(data: Dict[str, Any]) -> Path:
    explicit = data.get("mp13_config_file")
    if explicit:
        return Path(str(explicit)).expanduser().resolve()
    default_dir = data.get("default_config_dir")
    if default_dir:
        return Path(str(default_dir)).expanduser().resolve() / "mp13_config.json"
    return get_default_config_path().expanduser().resolve()


def _hosting_configuration_file(config_file: Path) -> Path:
    return config_file.parent / "hosting" / "hosting_config.json"


def _journal_file(config_file: Path) -> Path:
    return config_file.parent / "hosting" / ".hosting_setup_journal.json"


def _read_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    value = load_json_config(path)
    if value is None:
        raise ValueError(f"configuration_json_invalid:{path.name}")
    return dict(value)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n"
    descriptor = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        try:
            parent_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            parent_fd = -1
        if parent_fd >= 0:
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
            finally:
                os.close(parent_fd)
    finally:
        if temp.exists():
            temp.unlink()


@contextmanager
def _setup_lock(config_file: Path) -> Iterator[None]:
    lock = config_file.parent / "hosting" / ".hosting_setup.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("hosting_setup_locked") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def _recover_root_update(config_file: Path) -> Optional[str]:
    journal_path = _journal_file(config_file)
    if not journal_path.exists():
        return None
    journal = _read_mapping(journal_path)
    phase = str(journal.get("phase") or "")
    top_path = Path(str(journal.get("top_path") or config_file)).resolve()
    hosting_path = Path(str(journal.get("hosting_path") or _hosting_configuration_file(config_file))).resolve()
    previous_top = dict(journal.get("previous_top") or {})
    previous_hosting = dict(journal.get("previous_hosting") or {})
    target_top = dict(journal.get("target_top") or {})
    target_hosting = dict(journal.get("target_hosting") or {})
    write_hosting = bool(journal.get("write_hosting", False))
    if phase == "prepared":
        journal_path.unlink(missing_ok=True)
        return "discarded_prepared"
    if phase == "top_level_written":
        _atomic_write_json(top_path, previous_top)
        if previous_hosting:
            _atomic_write_json(hosting_path, previous_hosting)
        elif hosting_path.exists():
            hosting_path.unlink()
        journal_path.unlink(missing_ok=True)
        return "rolled_back_top_level"
    if phase in {"hosting_written", "committed"}:
        _atomic_write_json(top_path, target_top)
        if write_hosting:
            _atomic_write_json(hosting_path, target_hosting)
        journal_path.unlink(missing_ok=True)
        return "completed_target"
    raise ValueError("hosting_setup_journal_phase_invalid")


def _daemon_active(hosting_root: Path) -> bool:
    pid_path = hosting_root / "state" / "daemon.pid"
    if not pid_path.exists():
        return False
    try:
        payload = _read_mapping(pid_path)
        pid = int(payload.get("pid") or 0)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _root_change_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    if str(data.get("contract") or HOSTING_SETUP_CONTRACT) != HOSTING_SETUP_CONTRACT:
        raise ValueError("hosting_setup_contract_unsupported")
    config_file = _config_file(data)
    current = _read_mapping(config_file)
    current_dirs = dict(DEFAULT_CATEGORY_DIRS)
    if isinstance(current.get("category_dirs"), dict):
        current_dirs.update(current["category_dirs"])
    requested = dict(data.get("roots") or {})
    unknown = sorted(set(requested) - set(_ROOT_FIELDS))
    if unknown:
        raise ValueError(f"hosting_setup_roots_unknown:{','.join(unknown)}")
    logical = {key: str(requested.get(key) or current_dirs[key]) for key in _ROOT_FIELDS}
    candidate = dict(current)
    candidate_dirs = dict(current.get("category_dirs") or {})
    candidate_dirs.update(logical)
    candidate["category_dirs"] = candidate_dirs
    resolved_candidate, _ = resolve_config_paths(candidate, cwd=config_file.parent, config_path=config_file)
    resolved_current, _ = resolve_config_paths(current, cwd=config_file.parent, config_path=config_file)
    resolved = {key: str(resolved_candidate["category_dirs"][key]) for key in _ROOT_FIELDS}
    current_resolved = {key: str(resolved_current["category_dirs"][key]) for key in _ROOT_FIELDS}
    checks = []
    allow_nonempty = bool(data.get("allow_nonempty_destinations", False))
    allow_cross_volume = bool(data.get("allow_cross_volume", False))
    for key in _ROOT_FIELDS:
        destination = Path(resolved[key])
        current_path = Path(current_resolved[key])
        parent = destination if destination.exists() else destination.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        writable = parent.exists() and os.access(parent, os.W_OK)
        nonempty = destination.exists() and destination.is_dir() and any(destination.iterdir())
        changed = destination != current_path
        cross_volume = bool(changed and current_path.anchor and destination.anchor and current_path.anchor.lower() != destination.anchor.lower())
        free_bytes = shutil.disk_usage(parent).free if parent.exists() else 0
        checks.append(
            {
                "root": key,
                "writable": writable,
                "nonempty": nonempty,
                "changed": changed,
                "cross_volume": cross_volume,
                "free_bytes": free_bytes,
                "ok": bool(writable and (not changed or allow_nonempty or not nonempty) and (allow_cross_volume or not cross_volume)),
            }
        )
    active = _daemon_active(Path(current_resolved["hosting_root_dir"]))
    if active and any(bool(row["changed"]) for row in checks):
        checks.append({"root": "daemon", "active": True, "ok": False})
    hosting_file = _hosting_configuration_file(config_file)
    hosting_configuration = _read_mapping(hosting_file)
    return {
        "contract": HOSTING_SETUP_RESULT_CONTRACT,
        "status": "planned",
        "host_local": True,
        "would_write": False,
        "mp13_config_file": str(config_file),
        "hosting_config_file": str(hosting_file),
        "logical_roots": logical,
        "resolved_roots": resolved,
        "current_resolved_roots": current_resolved,
        "config_revision": _revision(current),
        "hosting_revision": _revision(hosting_configuration),
        "preflight": checks,
        "ok": all(bool(row.get("ok")) for row in checks),
    }


def _apply_root_change(data: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(data.get("confirm", False)):
        raise PermissionError("confirm=True is required to apply hosting roots")
    config_file = _config_file(data)
    with _setup_lock(config_file):
        recovered = _recover_root_update(config_file)
        plan = _root_change_plan(data)
        if not bool(plan["ok"]):
            raise PermissionError("hosting_setup_preflight_failed")
        current = _read_mapping(config_file)
        hosting_file = _hosting_configuration_file(config_file)
        current_hosting = _read_mapping(hosting_file)
        expected_config = str(data.get("expected_config_revision") or "")
        expected_hosting = str(data.get("expected_hosting_revision") or "")
        if expected_config and expected_config != _revision(current):
            raise RuntimeError("hosting_setup_config_revision_conflict")
        if expected_hosting and expected_hosting != _revision(current_hosting):
            raise RuntimeError("hosting_setup_hosting_revision_conflict")
        target = dict(current)
        dirs = dict(target.get("category_dirs") or {})
        dirs.update(dict(plan["logical_roots"]))
        target["category_dirs"] = dirs
        write_hosting = data.get("hosting_configuration") is not None
        target_hosting = dict(data.get("hosting_configuration") or current_hosting)
        journal = {
            "contract": "hosting.setup.journal.v1",
            "phase": "prepared",
            "top_path": str(config_file),
            "hosting_path": str(hosting_file),
            "previous_top": current,
            "previous_hosting": current_hosting,
            "target_top": target,
            "target_hosting": target_hosting,
            "write_hosting": write_hosting,
        }
        journal_path = _journal_file(config_file)
        _atomic_write_json(journal_path, journal)
        _atomic_write_json(config_file, target)
        journal["phase"] = "top_level_written"
        _atomic_write_json(journal_path, journal)
        if write_hosting:
            _atomic_write_json(hosting_file, target_hosting)
        journal["phase"] = "hosting_written"
        _atomic_write_json(journal_path, journal)
        for value in plan["resolved_roots"].values():
            Path(str(value)).mkdir(parents=True, exist_ok=True)
        journal["phase"] = "committed"
        _atomic_write_json(journal_path, journal)
        journal_path.unlink(missing_ok=True)
        return {
            **plan,
            "status": "applied",
            "would_write": True,
            "config_revision": _revision(target),
            "hosting_revision": _revision(target_hosting),
            "journal_state": "committed",
            "recovery": recovered,
        }


def plan_local_hosting_setup(request: LocalHostingSetupRequest | Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = _data(request)
    _require_host_local(data)
    if data.get("roots") is not None or data.get("mp13_config_file") or str(data.get("operation") or "") == "plan":
        return _root_change_plan(data)
    args = _args(data)
    paths = _cli._resolve_paths(args, create_dirs=False)
    mode = _cli._normalize_mode(str(getattr(args, "mode", "") or "local_only"), "local_only")
    endpoint_mode = _cli._normalize_endpoint_mode(str(getattr(args, "endpoint_mode", "") or "exclusive"), "exclusive")
    lifecycle_profile = _cli._normalize_lifecycle_profile(
        str(getattr(args, "lifecycle_profile", "") or "detached_user_process"),
        "detached_user_process",
    )
    require_auth = _cli._safe_require_auth(
        connectivity_mode=mode,
        endpoint_mode=endpoint_mode,
        requested=getattr(args, "require_auth", None),
    )
    summary = _cli._summarize_existing_config(
        control_state_path=paths["control_state_path"],
        access_file=paths["access_file"],
        keys_file=paths["keys_file"],
    )
    probe = _cli._probe_current_files(
        control_state_path=paths["control_state_path"],
        access_file=paths["access_file"],
        keys_file=paths["keys_file"],
        mappings_file=paths["mappings_file"],
        bootstrap_state_file=paths["bootstrap_state_file"],
        audit_file=paths["audit_file"],
    )
    return {
        "status": "planned",
        "host_local": True,
        "would_write": False,
        "hosting_root": str(paths["hosting_root"]),
        "control_state_file": str(paths["control_state_path"]),
        "access_control_file": str(paths["access_file"]),
        "keys_file": str(paths["keys_file"]),
        "connectivity_mode": mode,
        "endpoint_mode_default": endpoint_mode,
        "lifecycle_profile": lifecycle_profile,
        "require_auth": require_auth,
        "key_action": str(getattr(args, "key_action", "") or "replace"),
        "key_source": str(getattr(args, "key_source", "") or "generate"),
        "admin_key_id": str(getattr(args, "admin_key_id", "") or "admin-main"),
        "current_summary": summary,
        "current_state": _cli._classify_config_state(summary, probe),
    }


def apply_local_hosting_setup(request: LocalHostingSetupRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _data(request)
    _require_host_local(data)
    if data.get("roots") is not None or data.get("mp13_config_file") or str(data.get("operation") or "") == "apply":
        return _apply_root_change(data)
    return _cli.run_setup(_args(data))


def inspect_local_hosting_setup(request: LocalHostingSetupRequest | Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = _data(request)
    _require_host_local(data)
    if data.get("roots") is not None or data.get("mp13_config_file") or str(data.get("operation") or "") == "inspect":
        plan = _root_change_plan(data)
        return {**plan, "status": "ok", "recovery_pending": _journal_file(_config_file(data)).exists()}
    args = _args(data)
    return {
        "status": "ok",
        "host_local": True,
        "setup": _cli.run_status(args),
        "doctor": _cli.run_doctor(args),
    }


def get_local_hosting_setup_status(request: LocalHostingSetupRequest | Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = _data(request)
    _require_host_local(data)
    if data.get("roots") is not None or data.get("mp13_config_file") or str(data.get("operation") or "") == "status":
        plan = _root_change_plan(data)
        return {**plan, "status": "ok", "recovery_pending": _journal_file(_config_file(data)).exists()}
    args = _args(data)
    paths = _cli._resolve_paths(args, create_dirs=False)
    from .service.host_service import EngineHostService

    svc = EngineHostService(
        control_state_file=paths["control_state_path"],
    )
    summary = _cli._summarize_existing_config(
        control_state_path=paths["control_state_path"],
        access_file=paths["access_file"],
        keys_file=paths["keys_file"],
    )
    probe = _cli._probe_current_files(
        control_state_path=paths["control_state_path"],
        access_file=paths["access_file"],
        keys_file=paths["keys_file"],
        mappings_file=paths["mappings_file"],
        bootstrap_state_file=paths["bootstrap_state_file"],
        audit_file=paths["audit_file"],
    )
    return {
        "status": "ok",
        "host_local": True,
        "hosting_root": str(paths["hosting_root"]),
        "setup_summary": summary,
        "setup_state": _cli._classify_config_state(summary, probe),
        "probe": probe,
        "hosting_api_summary": svc.hosting_setup_summary(),
    }


def reset_local_hosting_setup(request: LocalHostingSetupRequest | Dict[str, Any]) -> Dict[str, Any]:
    data = _data(request)
    _require_host_local(data)
    if data.get("roots") is not None or data.get("mp13_config_file") or str(data.get("operation") or "") == "reset":
        if not bool(data.get("confirm", False) or data.get("confirm_reset", False)):
            raise PermissionError("confirm=True is required to reset hosting roots")
        reset_data = {
            **data,
            "operation": "apply",
            "confirm": True,
            "roots": {key: str(DEFAULT_CATEGORY_DIRS[key]) for key in _ROOT_FIELDS},
            "hosting_configuration": None,
            "allow_nonempty_destinations": True,
            "allow_cross_volume": True,
        }
        result = _apply_root_change(reset_data)
        return {**result, "status": "reset", "packages_environments_preserved": True}
    if not bool(data.get("confirm_reset", False)):
        raise PermissionError("confirm_reset=True is required to reset hosting setup")
    args = _args(data)
    paths = _cli._resolve_paths(args, create_dirs=False)
    return _cli._reset_access_configuration(paths)


__all__ = [
    "LocalHostingSetupRequest",
    "plan_local_hosting_setup",
    "apply_local_hosting_setup",
    "inspect_local_hosting_setup",
    "get_local_hosting_setup_status",
    "reset_local_hosting_setup",
]
