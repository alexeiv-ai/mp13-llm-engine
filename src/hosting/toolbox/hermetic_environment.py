"""Toolbox-only immutable environment identity and hermetic build inputs."""
from __future__ import annotations

import hashlib
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Mapping, Sequence

from .._process_utils import hidden_subprocess_kwargs
from .catalog import (
    ToolboxLockedDistributionSpec,
    normalize_distribution_name,
    normalize_import_root,
)
from .identity import ENVIRONMENT_IDENTITY_DOMAIN, environment_identity, require_digest
from .target import (
    SUPPORTED_PYTHON_ABI,
    detect_current_toolbox_target,
    validate_target_platform,
    wheel_is_compatible,
)


TOOLBOX_ENVIRONMENT_KEY_DOMAIN = ENVIRONMENT_IDENTITY_DOMAIN
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_ARTIFACT_FILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,255}")
_ENVIRONMENT_RECEIPT_CONTRACT = "hosting.toolbox.hermetic_environment_receipt.v1"
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def _environment_process_lock(path: Path, *, timeout_seconds: float = 120.0) -> Generator[None, None, None]:
    """Serialize threads and processes on a stable environment-key sidecar."""

    lock_name = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(lock_name, threading.Lock())
    if not thread_lock.acquire(timeout=max(0.1, timeout_seconds)):
        raise TimeoutError("toolbox_environment_lock_timeout")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                deadline = time.monotonic() + max(0.1, timeout_seconds)
                while True:
                    handle.seek(0)
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("toolbox_environment_lock_timeout")
                        time.sleep(0.05)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if sys.platform == "win32":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        thread_lock.release()


def _strict_fields(row: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    unknown = sorted(set(row) - fields)
    missing = sorted(fields - set(row))
    if unknown:
        raise ValueError(f"{label}_unknown_fields:{','.join(unknown)}")
    if missing:
        raise ValueError(f"{label}_missing_fields:{','.join(missing)}")


def _id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value.strip()):
        raise ValueError(f"{label}_invalid")
    return value.strip()


@dataclass(frozen=True, order=True)
class ToolboxLockedArtifactSpec:
    """One immutable install artifact selected from an administrator source."""

    distribution_name: str
    version: str
    source_id: str
    filename: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "distribution_name", normalize_distribution_name(self.distribution_name))
        object.__setattr__(self, "version", _id(self.version, label="locked_artifact_version"))
        object.__setattr__(self, "source_id", _id(self.source_id, label="locked_artifact_source_id"))
        filename = str(self.filename or "").strip()
        if not _ARTIFACT_FILE_RE.fullmatch(filename) or Path(filename).name != filename:
            raise ValueError("locked_artifact_filename_invalid")
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "sha256", require_digest(self.sha256, label="locked_artifact_sha256"))
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise ValueError("locked_artifact_size_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution_name": self.distribution_name,
            "version": self.version,
            "source_id": self.source_id,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxLockedArtifactSpec":
        row = dict(payload or {})
        _strict_fields(
            row,
            {"distribution_name", "version", "source_id", "filename", "sha256", "size_bytes"},
            label="locked_artifact",
        )
        return cls(**row)


@dataclass(frozen=True)
class ResolvedToolboxEnvironmentInput:
    """Complete host-derived input; never accepts a venv or environment name."""

    template_id: str
    template_digest: str
    runtime_version: str
    runtime_artifact_digest: str
    python_abi: str
    platform: str
    complete_lock_digest: str
    complete_lock: tuple[ToolboxLockedDistributionSpec, ...]
    locked_artifacts: tuple[ToolboxLockedArtifactSpec, ...]
    custom_resolved_lock_digest: str | None
    isolation_policy_version: str
    resolved_import_roots: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "template_id", _id(self.template_id, label="resolved_template_id"))
        object.__setattr__(self, "template_digest", require_digest(self.template_digest, label="resolved_template_digest"))
        object.__setattr__(self, "runtime_version", _id(self.runtime_version, label="resolved_runtime_version"))
        object.__setattr__(self, "runtime_artifact_digest", require_digest(self.runtime_artifact_digest, label="resolved_runtime_artifact_digest"))
        if self.python_abi != SUPPORTED_PYTHON_ABI:
            raise ValueError("resolved_python_abi_invalid")
        validate_target_platform(self.platform, label="resolved_platform")
        object.__setattr__(self, "complete_lock_digest", require_digest(self.complete_lock_digest, label="resolved_complete_lock_digest"))
        lock = tuple(self.complete_lock)
        if any(not isinstance(item, ToolboxLockedDistributionSpec) for item in lock):
            raise ValueError("resolved_complete_lock_item_invalid")
        if not lock or tuple(sorted(lock)) != lock or len({item.name for item in lock}) != len(lock):
            raise ValueError("resolved_complete_lock_invalid")
        object.__setattr__(self, "complete_lock", lock)
        artifacts = tuple(self.locked_artifacts)
        if any(not isinstance(item, ToolboxLockedArtifactSpec) for item in artifacts):
            raise ValueError("resolved_locked_artifact_item_invalid")
        if not artifacts or tuple(sorted(artifacts)) != artifacts:
            raise ValueError("resolved_locked_artifacts_invalid")
        artifact_pairs = [(item.distribution_name, item.version) for item in artifacts]
        lock_pairs = [(item.name, item.version) for item in lock]
        if artifact_pairs != lock_pairs:
            raise ValueError("resolved_locked_artifacts_incomplete")
        object.__setattr__(self, "locked_artifacts", artifacts)
        if self.custom_resolved_lock_digest is not None:
            object.__setattr__(
                self,
                "custom_resolved_lock_digest",
                require_digest(self.custom_resolved_lock_digest, label="resolved_custom_lock_digest"),
            )
        object.__setattr__(
            self,
            "isolation_policy_version",
            _id(self.isolation_policy_version, label="resolved_isolation_policy_version"),
        )
        roots = tuple(sorted(normalize_import_root(item) for item in self.resolved_import_roots))
        if len(set(roots)) != len(roots):
            raise ValueError("resolved_import_roots_duplicate")
        object.__setattr__(self, "resolved_import_roots", roots)

    @property
    def environment_key(self) -> str:
        return environment_identity(
            runtime_identity={
                "runtime_version": self.runtime_version,
                "runtime_artifact_digest": self.runtime_artifact_digest,
                "python_abi": self.python_abi,
                "platform": self.platform,
            },
            template_lock_digest=self.complete_lock_digest,
            custom_lock_digest=self.custom_resolved_lock_digest,
            isolation_policy={"version": self.isolation_policy_version},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "template_digest": self.template_digest,
            "runtime_version": self.runtime_version,
            "runtime_artifact_digest": self.runtime_artifact_digest,
            "python_abi": self.python_abi,
            "platform": self.platform,
            "complete_lock_digest": self.complete_lock_digest,
            "complete_lock": [item.to_dict() for item in self.complete_lock],
            "locked_artifacts": [item.to_dict() for item in self.locked_artifacts],
            "custom_resolved_lock_digest": self.custom_resolved_lock_digest,
            "isolation_policy_version": self.isolation_policy_version,
            "resolved_import_roots": list(self.resolved_import_roots),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResolvedToolboxEnvironmentInput":
        row = dict(payload or {})
        fields = {
            "template_id", "template_digest", "runtime_version",
            "runtime_artifact_digest", "python_abi", "platform",
            "complete_lock_digest", "complete_lock", "locked_artifacts", "custom_resolved_lock_digest",
            "isolation_policy_version", "resolved_import_roots",
        }
        _strict_fields(row, fields, label="resolved_toolbox_environment")
        if not isinstance(row["complete_lock"], Sequence) or isinstance(row["complete_lock"], (str, bytes, bytearray)):
            raise ValueError("resolved_complete_lock_invalid")
        if not isinstance(row["resolved_import_roots"], Sequence) or isinstance(row["resolved_import_roots"], (str, bytes, bytearray)):
            raise ValueError("resolved_import_roots_invalid")
        if not isinstance(row["locked_artifacts"], Sequence) or isinstance(row["locked_artifacts"], (str, bytes, bytearray)):
            raise ValueError("resolved_locked_artifacts_invalid")
        return cls(
            **{
                **row,
                "complete_lock": tuple(ToolboxLockedDistributionSpec.from_dict(item) for item in row["complete_lock"]),
                "locked_artifacts": tuple(ToolboxLockedArtifactSpec.from_dict(item) for item in row["locked_artifacts"]),
                "resolved_import_roots": tuple(row["resolved_import_roots"]),
            }
        )


@dataclass(frozen=True)
class HermeticToolboxEnvironmentSpec:
    resolved: ResolvedToolboxEnvironmentInput
    environment_key: str
    environment_root: str
    python_executable: str

    def __post_init__(self) -> None:
        if not isinstance(self.resolved, ResolvedToolboxEnvironmentInput):
            raise ValueError("hermetic_environment_resolved_invalid")
        if self.environment_key != self.resolved.environment_key:
            raise ValueError("hermetic_environment_key_mismatch")
        root = Path(self.environment_root).expanduser().resolve()
        python = Path(self.python_executable).expanduser().resolve()
        expected = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if python != expected:
            raise ValueError("hermetic_environment_python_path_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved.to_dict(),
            "environment_key": self.environment_key,
            "environment_root": self.environment_root,
            "python_executable": self.python_executable,
        }


class HermeticToolboxEnvironmentResolver:
    """Pure toolbox resolver; physical building is supplied by the next slice."""

    def __init__(self, hosting_root: Path):
        self.hosting_root = Path(hosting_root).expanduser().resolve()
        self.environments_root = (self.hosting_root / "environments" / "content").resolve()

    def environment_spec(
        self, resolved: ResolvedToolboxEnvironmentInput | Mapping[str, Any]
    ) -> HermeticToolboxEnvironmentSpec:
        model = (
            resolved
            if isinstance(resolved, ResolvedToolboxEnvironmentInput)
            else ResolvedToolboxEnvironmentInput.from_dict(resolved)
        )
        root = (self.environments_root / model.environment_key.removeprefix("sha256:")).resolve()
        python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        return HermeticToolboxEnvironmentSpec(
            resolved=model,
            environment_key=model.environment_key,
            environment_root=str(root),
            python_executable=str(python),
        )


class HermeticToolboxEnvironmentBuildError(RuntimeError):
    """Bounded physical-build failure safe for operation diagnostics."""

    def __init__(self, code: str, summary: str):
        self.code = _id(code, label="hermetic_build_error_code")
        text = str(summary or "").strip()
        if not text or len(text.encode("utf-8")) > 512:
            raise ValueError("hermetic_build_error_summary_invalid")
        self.summary = text
        super().__init__(self.code)


class PythonEnvironmentBuilder:
    """Target-host offline builder for independently materialized toolbox venvs."""

    def __init__(
        self,
        hosting_root: Path,
        *,
        artifact_sources: Mapping[str, Path],
        environment_root: Path | None = None,
        gc_grace_ms: int = 24 * 60 * 60 * 1000,
        build_timeout_seconds: int = 300,
    ) -> None:
        self.resolver = HermeticToolboxEnvironmentResolver(hosting_root)
        if environment_root is not None:
            self.resolver.environments_root = Path(environment_root).expanduser().resolve() / "content"
        self.environments_root = self.resolver.environments_root
        self.locks_root = self.environments_root / ".locks"
        self.candidates_root = self.environments_root / ".candidates"
        self.quarantine_root = self.environments_root / ".quarantine"
        self.references_path = self.environments_root / "references.json"
        self.references_lock = self.locks_root / "references.lock"
        self.artifact_sources = {
            _id(source_id, label="artifact_source_id"): Path(path).expanduser().resolve()
            for source_id, path in dict(artifact_sources or {}).items()
        }
        self.verified_artifact_paths: dict[tuple[str, str], Path] = {}
        if isinstance(gc_grace_ms, bool) or not isinstance(gc_grace_ms, int) or gc_grace_ms < 1:
            raise ValueError("environment_gc_grace_ms_invalid")
        self.gc_grace_ms = gc_grace_ms
        if (
            isinstance(build_timeout_seconds, bool)
            or not isinstance(build_timeout_seconds, int)
            or build_timeout_seconds < 60
            or build_timeout_seconds > 1_800
        ):
            raise ValueError("environment_build_timeout_seconds_invalid")
        self.build_timeout_seconds = build_timeout_seconds

    def configure_verified_artifact_paths(
        self, artifacts: Mapping[tuple[str, str], Path]
    ) -> None:
        exact: dict[tuple[str, str], Path] = {}
        for raw_key, raw_path in dict(artifacts or {}).items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise ValueError("verified_artifact_key_invalid")
            key = (
                _id(raw_key[0], label="verified_artifact_source_id"),
                str(raw_key[1] or "").strip(),
            )
            path = Path(raw_path).expanduser().resolve()
            if not key[1] or path.name != key[1] or not path.is_file():
                raise ValueError("verified_artifact_path_invalid")
            exact[key] = path
        if not exact:
            raise ValueError("verified_artifact_paths_required")
        if self.verified_artifact_paths and self.verified_artifact_paths != exact:
            raise ValueError("verified_artifact_paths_already_configured")
        self.verified_artifact_paths = exact

    def extend_verified_artifact_paths(
        self, artifacts: Mapping[tuple[str, str], Path]
    ) -> None:
        merged = dict(self.verified_artifact_paths)
        for raw_key, raw_path in dict(artifacts or {}).items():
            key = (
                _id(raw_key[0], label="verified_artifact_source_id"),
                str(raw_key[1] or "").strip(),
            )
            path = Path(raw_path).expanduser().resolve()
            if not key[1] or path.name != key[1] or not path.is_file():
                raise ValueError("verified_artifact_path_invalid")
            if key in merged and merged[key] != path:
                raise ValueError("verified_artifact_path_conflict")
            merged[key] = path
        if not merged:
            raise ValueError("verified_artifact_paths_required")
        self.verified_artifact_paths = merged

    @staticmethod
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
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HermeticToolboxEnvironmentBuildError(
                "environment_receipt_invalid", "The environment verification receipt is missing or invalid."
            ) from exc
        if not isinstance(payload, dict):
            raise HermeticToolboxEnvironmentBuildError(
                "environment_receipt_invalid", "The environment verification receipt is missing or invalid."
            )
        return payload

    def _run(self, python: Path, arguments: Sequence[str], *, code: str, summary: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [str(python), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=self.build_timeout_seconds,
            env=environment,
            **hidden_subprocess_kwargs(),
        )
        if completed.returncode != 0:
            raise HermeticToolboxEnvironmentBuildError(code, summary)
        return completed

    @staticmethod
    def _validate_target(resolved: ResolvedToolboxEnvironmentInput) -> None:
        current = detect_current_toolbox_target()
        current_version = ".".join(str(item) for item in sys.version_info[:3])
        if (
            resolved.python_abi != current.python_abi
            or resolved.platform != current.platform
            or resolved.runtime_version != current_version
        ):
            raise HermeticToolboxEnvironmentBuildError(
                "environment_runtime_target_mismatch",
                "The resolved runtime identity does not match this materialization host.",
            )
        incompatible = [
            artifact.filename
            for artifact in resolved.locked_artifacts
            if not wheel_is_compatible(artifact.filename, current)
        ]
        if incompatible:
            raise HermeticToolboxEnvironmentBuildError(
                "environment_artifact_target_mismatch",
                f"Locked wheel '{incompatible[0]}' is incompatible with this materialization host.",
            )

    def _artifact_paths(self, resolved: ResolvedToolboxEnvironmentInput) -> tuple[Path, ...]:
        paths: list[Path] = []
        for artifact in resolved.locked_artifacts:
            if self.verified_artifact_paths:
                path = self.verified_artifact_paths.get(
                    (artifact.source_id, artifact.filename)
                )
                if path is None:
                    raise HermeticToolboxEnvironmentBuildError(
                        "environment_artifact_source_denied",
                        "A locked artifact is absent from the verified source set.",
                    )
            else:
                source = self.artifact_sources.get(artifact.source_id)
                if source is None:
                    raise HermeticToolboxEnvironmentBuildError(
                        "environment_artifact_source_denied",
                        "A locked artifact source is not configured on this runtime host.",
                    )
                path = (source / artifact.filename).resolve()
                try:
                    path.relative_to(source)
                except ValueError as exc:
                    raise HermeticToolboxEnvironmentBuildError(
                        "environment_artifact_path_denied", "A locked artifact escaped its configured source."
                    ) from exc
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise HermeticToolboxEnvironmentBuildError(
                    "environment_artifact_unavailable", "A locked artifact is unavailable on this runtime host."
                ) from exc
            if len(data) != artifact.size_bytes or f"sha256:{hashlib.sha256(data).hexdigest()}" != artifact.sha256:
                raise HermeticToolboxEnvironmentBuildError(
                    "environment_artifact_verification_failed",
                    "A locked artifact did not match its approved digest and size.",
                )
            paths.append(path)
        return tuple(paths)

    @staticmethod
    def _receipt_payload(
        spec: HermeticToolboxEnvironmentSpec,
        *,
        verified_at_ms: int,
    ) -> dict[str, Any]:
        return {
            "contract": _ENVIRONMENT_RECEIPT_CONTRACT,
            "state": "verified",
            "environment_key": spec.environment_key,
            "resolved": spec.resolved.to_dict(),
            "artifact_digests": sorted(item.sha256 for item in spec.resolved.locked_artifacts),
            "installed_distributions": {
                item.name: item.version for item in spec.resolved.complete_lock
            },
            "verified_import_roots": list(spec.resolved.resolved_import_roots),
            "verified_at_ms": verified_at_ms,
            "system_site_packages": False,
            "installer": "venv-pip-offline-no-deps",
        }

    def _validate_published(self, spec: HermeticToolboxEnvironmentSpec) -> dict[str, Any]:
        root = Path(spec.environment_root)
        python = Path(spec.python_executable)
        receipt = self._read_json(root / "verification-receipt.json")
        resolved_receipt = dict(receipt.get("resolved") or {})
        resolved_expected = spec.resolved.to_dict()
        physical_fields = {
            "runtime_version", "runtime_artifact_digest", "python_abi", "platform",
            "complete_lock_digest", "complete_lock", "locked_artifacts",
            "custom_resolved_lock_digest", "isolation_policy_version",
        }
        if (
            receipt.get("contract") != _ENVIRONMENT_RECEIPT_CONTRACT
            or receipt.get("state") != "verified"
            or receipt.get("environment_key") != spec.environment_key
            or {key: resolved_receipt.get(key) for key in physical_fields}
            != {key: resolved_expected[key] for key in physical_fields}
            or receipt.get("artifact_digests")
            != sorted(item.sha256 for item in spec.resolved.locked_artifacts)
            or receipt.get("installed_distributions")
            != {item.name: item.version for item in spec.resolved.complete_lock}
            or receipt.get("system_site_packages") is not False
            or receipt.get("installer") != "venv-pip-offline-no-deps"
            or not python.is_file()
        ):
            raise HermeticToolboxEnvironmentBuildError(
                "environment_receipt_mismatch", "The published environment does not match its resolved identity."
            )
        config = (root / "pyvenv.cfg").read_text(encoding="utf-8").lower()
        if "include-system-site-packages = false" not in config:
            raise HermeticToolboxEnvironmentBuildError(
                "environment_site_packages_inherited", "The toolbox environment inherits ambient site packages."
            )
        return receipt

    def _probe_imports(self, python: Path, import_roots: tuple[str, ...]) -> None:
        probe_script = (
            "import importlib,json,sys;"
            "r=json.loads(sys.argv[1]);"
            "[importlib.import_module(x) for x in r];"
            "sys.stdout.write(json.dumps(r))"
        )
        probed = self._run(
            python,
            ["-c", probe_script, json.dumps(list(import_roots))],
            code="environment_import_probe_failed",
            summary="A resolved import root failed under the final environment interpreter.",
        )
        if tuple(json.loads(probed.stdout or "[]")) != import_roots:
            raise HermeticToolboxEnvironmentBuildError(
                "environment_import_probe_failed",
                "A resolved import root failed under the final environment interpreter.",
            )

    def _build_candidate(self, spec: HermeticToolboxEnvironmentSpec, candidate: Path) -> None:
        self._validate_target(spec.resolved)
        artifact_paths = self._artifact_paths(spec.resolved)
        candidate.mkdir(parents=True, exist_ok=False)
        venv.EnvBuilder(with_pip=True, system_site_packages=False, clear=True).create(str(candidate))
        python = candidate / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self._run(
            python,
            ["-m", "pip", "install", "--disable-pip-version-check", "--no-index", "--no-deps", *map(str, artifact_paths)],
            code="environment_locked_install_failed",
            summary="The offline locked artifact installation failed.",
        )
        expected = {item.name: item.version for item in spec.resolved.complete_lock}
        verification_script = (
            "import importlib.metadata as m,json,sys;"
            "e=json.loads(sys.argv[1]);"
            "g={k:m.version(k) for k in e};"
            "sys.stdout.write(json.dumps(g,sort_keys=True));"
            "raise SystemExit(0 if g==e else 7)"
        )
        checked = self._run(
            python,
            ["-c", verification_script, json.dumps(expected, sort_keys=True)],
            code="environment_lock_receipt_failed",
            summary="The final interpreter did not contain the complete resolved distribution lock.",
        )
        if json.loads(checked.stdout or "{}") != expected:
            raise HermeticToolboxEnvironmentBuildError(
                "environment_lock_receipt_failed",
                "The final interpreter did not contain the complete resolved distribution lock.",
            )
        self._probe_imports(python, spec.resolved.resolved_import_roots)
        candidate_spec = HermeticToolboxEnvironmentSpec(
            resolved=spec.resolved,
            environment_key=spec.environment_key,
            environment_root=str(candidate),
            python_executable=str(python),
        )
        self._atomic_json(
            candidate / "verification-receipt.json",
            self._receipt_payload(candidate_spec, verified_at_ms=int(time.time() * 1000)),
        )

    def materialize_environment(
        self,
        resolved: ResolvedToolboxEnvironmentInput | Mapping[str, Any],
        *,
        reference_id: str,
        add_reference: bool = True,
    ) -> HermeticToolboxEnvironmentSpec:
        model = resolved if isinstance(resolved, ResolvedToolboxEnvironmentInput) else ResolvedToolboxEnvironmentInput.from_dict(resolved)
        reference = _id(reference_id, label="environment_reference_id")
        spec = self.resolver.environment_spec(model)
        key = spec.environment_key.removeprefix("sha256:")
        lock_path = self.locks_root / f"{key}.lock"
        with _environment_process_lock(lock_path):
            root = Path(spec.environment_root)
            if root.exists():
                self._validate_published(spec)
                self._probe_imports(
                    Path(spec.python_executable), spec.resolved.resolved_import_roots
                )
            else:
                candidate = self.candidates_root / f"{key}.{os.getpid()}.{uuid.uuid4().hex}"
                try:
                    self._build_candidate(spec, candidate)
                    root.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(candidate, root)
                    self._validate_published(spec)
                except Exception as exc:
                    if candidate.exists():
                        self.quarantine_root.mkdir(parents=True, exist_ok=True)
                        quarantined = self.quarantine_root / f"{key}.{int(time.time() * 1000)}.{uuid.uuid4().hex}"
                        os.replace(candidate, quarantined)
                        code = exc.code if isinstance(exc, HermeticToolboxEnvironmentBuildError) else "environment_build_failed"
                        self._atomic_json(quarantined / "failure.json", {"code": code, "environment_key": spec.environment_key})
                    if isinstance(exc, HermeticToolboxEnvironmentBuildError):
                        raise
                    raise HermeticToolboxEnvironmentBuildError(
                        "environment_build_failed", "The hermetic toolbox environment build failed."
                    ) from exc
        if add_reference:
            self._add_reference(spec.environment_key, reference)
        return spec

    def verified_environment(
        self, resolved: ResolvedToolboxEnvironmentInput | Mapping[str, Any]
    ) -> HermeticToolboxEnvironmentSpec:
        spec = self.resolver.environment_spec(resolved)
        self._validate_published(spec)
        return spec

    def _read_references_unlocked(self) -> dict[str, Any]:
        if not self.references_path.exists():
            return {"contract": "hosting.toolbox.environment_references.v1", "environments": {}}
        row = self._read_json(self.references_path)
        if set(row) != {"contract", "environments"} or row["contract"] != "hosting.toolbox.environment_references.v1" or not isinstance(row["environments"], dict):
            raise HermeticToolboxEnvironmentBuildError(
                "environment_references_invalid", "The environment reference index is invalid."
            )
        return row

    def _add_reference(self, environment_key: str, reference_id: str) -> None:
        with _environment_process_lock(self.references_lock):
            state = self._read_references_unlocked()
            refs = state["environments"].setdefault(environment_key, {})
            refs[reference_id] = int(time.time() * 1000)
            self._atomic_json(self.references_path, state)

    def release_reference(self, *, environment_key: str, reference_id: str) -> None:
        key = require_digest(environment_key, label="environment_key")
        reference = _id(reference_id, label="environment_reference_id")
        with _environment_process_lock(self.references_lock):
            state = self._read_references_unlocked()
            refs = state["environments"].get(key, {})
            refs.pop(reference, None)
            if not refs:
                state["environments"].pop(key, None)
            self._atomic_json(self.references_path, state)

    def remove_environment(self, *, environment_key: str) -> str:
        """Remove one exact unreferenced environment, never a logical/glob path."""

        key = require_digest(environment_key, label="environment_key")
        root = (self.environments_root / key.removeprefix("sha256:")).resolve()
        if root.parent != self.environments_root:
            raise HermeticToolboxEnvironmentBuildError(
                "environment_path_invalid", "The exact environment digest escaped the cache root."
            )
        with _environment_process_lock(self.references_lock):
            state = self._read_references_unlocked()
            if dict(state["environments"].get(key) or {}):
                raise HermeticToolboxEnvironmentBuildError(
                    "environment_references_present",
                    "The environment still has one or more active references.",
                )
            if not root.exists():
                return "already_absent"
            if not root.is_dir():
                raise HermeticToolboxEnvironmentBuildError(
                    "environment_path_invalid", "The exact environment target is not a directory."
                )
            with _environment_process_lock(self.locks_root / f"{key.removeprefix('sha256:')}.lock"):
                shutil.rmtree(root)
            return "removed"

    def garbage_collect(
        self,
        *,
        now_ms: int | None = None,
        protected_environment_keys: Sequence[str] = (),
        maximum_cache_bytes: int | None = None,
        maximum_cache_artifacts: int | None = None,
    ) -> tuple[str, ...]:
        current = int(time.time() * 1000) if now_ms is None else int(now_ms)
        protected = {
            require_digest(item, label="protected_environment_key")
            for item in protected_environment_keys
        }
        if maximum_cache_bytes is not None and int(maximum_cache_bytes) < 1:
            raise ValueError("maximum_cache_bytes_invalid")
        if maximum_cache_artifacts is not None and int(maximum_cache_artifacts) < 1:
            raise ValueError("maximum_cache_artifacts_invalid")
        removed: list[str] = []
        with _environment_process_lock(self.references_lock):
            state = self._read_references_unlocked()
            referenced = set(state["environments"])
            for root in self.environments_root.iterdir() if self.environments_root.exists() else ():
                if not root.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", root.name):
                    continue
                key = f"sha256:{root.name}"
                if key in referenced or key in protected:
                    continue
                receipt = self._read_json(root / "verification-receipt.json")
                verified_at = int(receipt.get("verified_at_ms", current))
                if current - verified_at < self.gc_grace_ms:
                    continue
                with _environment_process_lock(self.locks_root / f"{root.name}.lock"):
                    shutil.rmtree(root)
                    removed.append(key)
        return tuple(sorted(removed))

    def required_environment_readiness(
        self, resolved_inputs: Sequence[ResolvedToolboxEnvironmentInput]
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for resolved in resolved_inputs:
            try:
                spec = self.verified_environment(resolved)
                items.append({"template_id": resolved.template_id, "environment_key": spec.environment_key, "state": "ready", "code": "environment_verified"})
            except HermeticToolboxEnvironmentBuildError as exc:
                items.append({"template_id": resolved.template_id, "environment_key": resolved.environment_key, "state": "degraded", "code": exc.code})
        ready = bool(items) and all(item["state"] == "ready" for item in items)
        return {"state": "ready" if ready else "degraded", "code": "required_environments_ready" if ready else "required_environment_unavailable", "environments": items}


__all__ = [
    "HermeticToolboxEnvironmentBuildError",
    "PythonEnvironmentBuilder",
    "HermeticToolboxEnvironmentResolver",
    "HermeticToolboxEnvironmentSpec",
    "ResolvedToolboxEnvironmentInput",
    "ToolboxLockedArtifactSpec",
    "TOOLBOX_ENVIRONMENT_KEY_DOMAIN",
]
