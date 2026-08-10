"""Resolve release-owned built-in intent from bounded read-only wheelhouses."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

from packaging.utils import InvalidWheelFilename, parse_wheel_filename

from .._process_utils import hidden_subprocess_kwargs
from .catalog import ToolboxLockedDistributionSpec, normalize_distribution_name
from .hermetic_environment import ToolboxLockedArtifactSpec
from .host_project_config import ToolboxBuiltinIntent, ToolboxHostProjectConfiguration
from .identity import identity_digest
from .target import wheel_is_compatible


@dataclass(frozen=True)
class ResolvedBuiltinWheelClosure:
    template_id: str
    lock_digest: str
    locked_distributions: tuple[ToolboxLockedDistributionSpec, ...]
    locked_artifacts: tuple[ToolboxLockedArtifactSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "lock_digest": self.lock_digest,
            "locked_distributions": [item.to_dict() for item in self.locked_distributions],
            "locked_artifacts": [item.to_dict() for item in self.locked_artifacts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResolvedBuiltinWheelClosure":
        row = dict(payload or {})
        if set(row) != {
            "template_id", "lock_digest", "locked_distributions", "locked_artifacts"
        }:
            raise ValueError("resolved_builtin_closure_fields_invalid")
        from .catalog import ToolboxLockedDistributionSpec

        return cls(
            template_id=str(row["template_id"]),
            lock_digest=str(row["lock_digest"]),
            locked_distributions=tuple(
                ToolboxLockedDistributionSpec.from_dict(item)
                for item in row["locked_distributions"]
            ),
            locked_artifacts=tuple(
                ToolboxLockedArtifactSpec.from_dict(item)
                for item in row["locked_artifacts"]
            ),
        )


@dataclass(frozen=True)
class BuiltinWheelResolutionResult:
    status: str
    config_revision: str
    source_set_revision: str
    target: str
    closures: tuple[ResolvedBuiltinWheelClosure, ...]
    diagnostics: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "config_revision": self.config_revision,
            "source_set_revision": self.source_set_revision,
            "target": self.target,
            "closures": [item.to_dict() for item in self.closures],
            "diagnostics": [dict(item) for item in self.diagnostics],
        }


class AirgapBuiltinWheelResolver:
    """Use pip's resolver without indexes, installs, or source distributions."""

    def __init__(
        self,
        configuration: ToolboxHostProjectConfiguration,
        *,
        artifact_sources: Mapping[str, Path],
        verified_artifacts: Mapping[str, Mapping[str, Path]] | None = None,
    ) -> None:
        self.configuration = configuration
        self.sources = {
            source.source_id: Path(artifact_sources[source.source_id]).expanduser().resolve()
            for source in configuration.sources
            if source.kind == "airgap_store" and source.source_id in artifact_sources
        }
        self.verified_artifacts = {
            str(source_id): {
                str(filename): Path(path).expanduser().resolve()
                for filename, path in dict(artifacts).items()
            }
            for source_id, artifacts in dict(verified_artifacts or {}).items()
        }

    @staticmethod
    def _diagnostic(intent: ToolboxBuiltinIntent, code: str) -> dict[str, str]:
        summaries = {
            "required_template_requirements_missing": "The built-in intent has no package requirement roots.",
            "required_template_source_unavailable": "The configured source mode is not available to the air-gap resolver.",
            "required_template_wheel_missing": "No complete compatible exact wheel closure is available.",
            "required_template_resolution_invalid": "The configured wheel closure is invalid.",
            "required_template_resolution_bounds_exceeded": "The resolved wheel closure exceeds configured bounds.",
        }
        return {
            "template_id": intent.template_id,
            "code": code,
            "summary": summaries[code],
        }

    @staticmethod
    def _report_path(raw_url: str) -> Path:
        parsed = urlsplit(raw_url)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError("resolved_artifact_url_invalid")
        raw = url2pathname(unquote(parsed.path))
        if os.name == "nt" and re.match(r"^/[A-Za-z]:", raw):
            raw = raw[1:]
        return Path(raw).resolve()

    @staticmethod
    def _allowed(name: str, namespaces: tuple[str, ...]) -> bool:
        return any(
            item == "*"
            or name == item.removesuffix(".*")
            or (item.endswith(".*") and name.startswith(item[:-1]))
            for item in namespaces
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()

    def _source_for(self, path: Path, distribution_name: str) -> tuple[str, int]:
        for source in self.configuration.sources:
            verified = self.verified_artifacts.get(source.source_id, {})
            if verified.get(path.name) == path:
                if not self._allowed(distribution_name, source.allowed_package_namespaces):
                    raise ValueError("resolved_artifact_source_invalid")
                return source.source_id, source.maximum_download_bytes
            root = self.sources.get(source.source_id)
            if root is None:
                continue
            try:
                path.relative_to(root)
            except ValueError:
                continue
            if path.parent != root or not self._allowed(distribution_name, source.allowed_package_namespaces):
                raise ValueError("resolved_artifact_source_invalid")
            return source.source_id, source.maximum_download_bytes
        raise ValueError("resolved_artifact_source_invalid")

    def resolve_requirements(
        self, *, template_id: str, package_requirements: tuple[str, ...]
    ) -> ResolvedBuiltinWheelClosure:
        logical_template_id = str(template_id or "").strip()
        if not logical_template_id:
            raise ValueError("resolved_template_id_required")
        requirements = tuple(str(item or "").strip() for item in package_requirements)
        if not requirements or any(not item for item in requirements):
            raise RuntimeError("required_template_requirements_missing")
        wheelhouses = [
            self.sources[item.source_id]
            for item in self.configuration.sources
            if item.source_id in self.sources
        ]
        wheelhouses.extend(
            path.parent
            for source in self.configuration.sources
            for path in self.verified_artifacts.get(source.source_id, {}).values()
        )
        wheelhouses = list(dict.fromkeys(wheelhouses))
        if not wheelhouses or self.configuration.resolution.mode not in {
            "air_gapped", "prefer_airgap", "online"
        }:
            raise RuntimeError("required_template_source_unavailable")
        with tempfile.TemporaryDirectory(prefix="mp13-toolbox-resolve-") as temporary:
            report = Path(temporary) / "report.json"
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--disable-pip-version-check",
                "--no-input",
                "--no-index",
                "--only-binary=:all:",
                "--report",
                str(report),
            ]
            for root in wheelhouses:
                command.extend(("--find-links", str(root)))
            command.extend(requirements)
            environment = {
                **os.environ,
                "PIP_CONFIG_FILE": os.devnull,
                "PIP_NO_INPUT": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            }
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.configuration.resolution.timeout_seconds,
                check=False,
                env=environment,
                **hidden_subprocess_kwargs(),
            )
            if completed.returncode != 0 or not report.is_file():
                raise RuntimeError("required_template_wheel_missing")
            try:
                payload = json.loads(report.read_text(encoding="utf-8"))
                installs = list(payload["install"])
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise RuntimeError("required_template_resolution_invalid") from exc

        distributions: list[ToolboxLockedDistributionSpec] = []
        artifacts: list[ToolboxLockedArtifactSpec] = []
        source_bytes: dict[str, int] = {}
        try:
            for row in installs:
                metadata = dict(row["metadata"])
                name = normalize_distribution_name(metadata["name"])
                version = str(metadata["version"])
                path = self._report_path(row["download_info"]["url"])
                if not path.is_file() or path.suffix.lower() != ".whl" or not wheel_is_compatible(path.name):
                    raise ValueError("resolved_artifact_wheel_invalid")
                wheel_name, wheel_version, _build, _tags = parse_wheel_filename(path.name)
                if normalize_distribution_name(wheel_name) != name or str(wheel_version) != version:
                    raise ValueError("resolved_artifact_metadata_mismatch")
                source_id, source_maximum = self._source_for(path, name)
                size = path.stat().st_size
                source_bytes[source_id] = source_bytes.get(source_id, 0) + size
                if source_bytes[source_id] > source_maximum:
                    raise OverflowError
                digest = self._sha256(path)
                distributions.append(ToolboxLockedDistributionSpec(name=name, version=version))
                artifacts.append(
                    ToolboxLockedArtifactSpec(
                        distribution_name=name,
                        version=version,
                        source_id=source_id,
                        filename=path.name,
                        sha256=digest,
                        size_bytes=size,
                    )
                )
        except (InvalidWheelFilename, KeyError, TypeError, ValueError, OSError) as exc:
            raise RuntimeError("required_template_resolution_invalid") from exc
        except OverflowError as exc:
            raise RuntimeError("required_template_resolution_bounds_exceeded") from exc
        distributions.sort()
        artifacts.sort()
        if (
            not distributions
            or len(distributions) > self.configuration.resolution.maximum_artifacts
            or sum(item.size_bytes for item in artifacts) > self.configuration.resolution.maximum_bytes
            or len({item.name for item in distributions}) != len(distributions)
        ):
            raise RuntimeError("required_template_resolution_bounds_exceeded")
        lock = tuple(distributions)
        artifact_tuple = tuple(artifacts)
        return ResolvedBuiltinWheelClosure(
            template_id=logical_template_id,
            lock_digest=identity_digest(
                "hosting.toolbox.builtin_lock.v1",
                {
                    "target": self.configuration.target.name,
                    "distributions": [item.to_dict() for item in lock],
                    "artifacts": [item.to_dict() for item in artifact_tuple],
                },
            ),
            locked_distributions=lock,
            locked_artifacts=artifact_tuple,
        )

    def _resolve_one(self, intent: ToolboxBuiltinIntent) -> ResolvedBuiltinWheelClosure:
        return self.resolve_requirements(
            template_id=intent.template_id,
            package_requirements=intent.package_requirements,
        )

    def resolve(self) -> BuiltinWheelResolutionResult:
        closures: list[ResolvedBuiltinWheelClosure] = []
        diagnostics: list[dict[str, str]] = []
        required_failed = False
        for intent in self.configuration.builtins:
            try:
                closures.append(self._resolve_one(intent))
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                code = (
                    "required_template_wheel_missing"
                    if isinstance(exc, subprocess.TimeoutExpired)
                    else str(exc)
                )
                diagnostics.append(self._diagnostic(intent, code))
                required_failed = required_failed or intent.required
        if required_failed:
            closures = []
        return BuiltinWheelResolutionResult(
            status="not_ready" if required_failed else ("degraded" if diagnostics else "resolved"),
            config_revision=self.configuration.config_revision,
            source_set_revision=self.configuration.source_set_revision,
            target=self.configuration.target.name,
            closures=tuple(closures),
            diagnostics=tuple(diagnostics),
        )


__all__ = [
    "AirgapBuiltinWheelResolver",
    "BuiltinWheelResolutionResult",
    "ResolvedBuiltinWheelClosure",
]
