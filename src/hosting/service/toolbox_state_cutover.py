"""Local operator cutover that archives version-1 toolbox state without translation."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..daemon.pidfile import DaemonPidFile
from .operation_repository import _exclusive_process_file_lock, _replace_with_bounded_retries
from .toolbox_state_v2 import AtomicJsonToolboxStateV2Repository


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_NEWER_TOOLBOX_STATE_NAMES = (
    "toolbox_sandboxes_v2.json",
    "toolbox_definition_plans.json",
    "toolbox_host_configurations.json",
    "toolbox_dependency_approvals.json",
    "toolbox_definition_confirmations.json",
    "toolbox_definition_candidates.json",
)


class ToolboxStateArchiveError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_bounded_retries(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _resolved_operator_root(value: str) -> Path:
    raw = str(value or "").strip()
    candidate = Path(raw).expanduser()
    if not raw or not candidate.is_absolute():
        raise ToolboxStateArchiveError("toolbox_archive_hosting_root_must_be_absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ToolboxStateArchiveError("toolbox_archive_hosting_root_unavailable") from exc
    if candidate.is_symlink() or not resolved.is_dir() or candidate.absolute() != resolved:
        raise ToolboxStateArchiveError("toolbox_archive_hosting_root_must_be_exact_resolved_directory")
    return resolved


def _inventory(root: Path, sources: list[Path]) -> list[dict[str, Any]]:
    files: list[Path] = []
    for source in sources:
        if source.is_symlink():
            raise ToolboxStateArchiveError("toolbox_archive_symlink_denied")
        if source.is_file():
            files.append(source)
        elif source.is_dir():
            for path in source.rglob("*"):
                if path.is_symlink():
                    raise ToolboxStateArchiveError("toolbox_archive_symlink_denied")
                if path.is_file():
                    files.append(path)
    out: list[dict[str, Any]] = []
    for path in sorted(set(files)):
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ToolboxStateArchiveError("toolbox_archive_path_escape") from exc
        out.append(
            {
                "source_relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return out


def archive_toolbox_state_v1(
    *,
    hosting_root: str,
    expected_state_sha256: str,
    acknowledge_version_1_archive: bool,
) -> dict[str, Any]:
    if not bool(acknowledge_version_1_archive):
        raise PermissionError("toolbox_state_v1_archive_acknowledgement_required")
    expected = str(expected_state_sha256 or "").strip()
    if not _DIGEST_RE.fullmatch(expected):
        raise ToolboxStateArchiveError("toolbox_archive_expected_digest_invalid")
    release_commit = str(os.environ.get("MP13_RELEASE_COMMIT") or "").strip().lower()
    if not _COMMIT_RE.fullmatch(release_commit):
        raise ToolboxStateArchiveError("toolbox_archive_release_commit_unavailable")
    root = _resolved_operator_root(hosting_root)
    state_root = (root / "state").resolve()
    source = (state_root / "toolbox_sandboxes.json").resolve()
    v2_path = (state_root / "toolbox_sandboxes_v2.json").resolve()
    if source.parent != state_root or not source.is_file():
        raise ToolboxStateArchiveError("toolbox_archive_state_file_unavailable")
    for pid_name in ("daemon.pid", "daemon_http.pid"):
        if DaemonPidFile(state_root / pid_name).is_alive():
            raise ToolboxStateArchiveError("toolbox_archive_daemon_running")
    lock_path = source.with_suffix(source.suffix + ".lock")
    with _exclusive_process_file_lock(lock_path):
        newer_state = [name for name in _NEWER_TOOLBOX_STATE_NAMES if (state_root / name).exists()]
        if newer_state:
            raise ToolboxStateArchiveError("toolbox_archive_newer_state_present")
        actual = _sha256(source)
        if actual != expected:
            raise ToolboxStateArchiveError("toolbox_archive_state_digest_mismatch")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ToolboxStateArchiveError("toolbox_archive_state_corrupt") from exc
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ToolboxStateArchiveError("toolbox_archive_state_version_invalid")
        sources = [source]
        bundles_root = (root / "toolbox_bundles").resolve()
        if bundles_root.exists():
            if bundles_root.parent != root:
                raise ToolboxStateArchiveError("toolbox_archive_bundle_root_invalid")
            sources.append(bundles_root)
        inventory = _inventory(root, sources)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive_root = (
            root / "archive" / "toolbox-state-v1" / f"{timestamp}-{actual.removeprefix('sha256:')[:16]}"
        ).resolve()
        if archive_root.exists():
            raise ToolboxStateArchiveError("toolbox_archive_target_exists")
        archive_root.mkdir(parents=True, exist_ok=False)
        incomplete = archive_root / "INCOMPLETE"
        incomplete.write_text("version-1 toolbox state archive in progress\n", encoding="utf-8")
        moved: list[tuple[Path, Path]] = []
        try:
            _atomic_json(
                archive_root / "inventory.json",
                {
                    "contract": "hosting.toolbox.state_v1_archive_inventory.v1",
                    "archive_scope": "legacy_toolbox_state_only",
                    "shared_package_environment_state_archived": False,
                    "hosting_root": str(root),
                    "state_sha256": actual,
                    "parent_release_commit": release_commit,
                    "files": inventory,
                },
            )
            for original in sources:
                destination = archive_root / "payload" / original.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(original, destination)
                moved.append((destination, original))
            _atomic_json(
                archive_root / "receipt.json",
                {
                    "contract": "hosting.toolbox.state_v1_archive_receipt.v1",
                    "status": "complete",
                    "archive_scope": "legacy_toolbox_state_only",
                    "shared_package_environment_state_archived": False,
                    "state_sha256": actual,
                    "parent_release_commit": release_commit,
                    "inventory_count": len(inventory),
                },
            )
            incomplete.unlink()
            _fsync_directory(archive_root)
            _fsync_directory(archive_root.parent)
            repository = AtomicJsonToolboxStateV2Repository(v2_path, legacy_path=source)
            initialized = repository.initialize_empty()
            return {
                "status": "ok",
                "archive_root": str(archive_root),
                "state_sha256": actual,
                "parent_release_commit": release_commit,
                "inventory_count": len(inventory),
                "version_2_state": initialized,
            }
        except Exception:
            if not v2_path.exists():
                for archived, original in reversed(moved):
                    original.parent.mkdir(parents=True, exist_ok=True)
                    if archived.exists():
                        os.replace(archived, original)
                shutil.rmtree(archive_root, ignore_errors=True)
            raise


__all__ = ["ToolboxStateArchiveError", "archive_toolbox_state_v1"]
