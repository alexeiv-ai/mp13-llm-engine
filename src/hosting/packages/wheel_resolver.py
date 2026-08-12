"""Bounded offline Python wheel resolution over the generic package index."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

from packaging.utils import InvalidWheelFilename, parse_wheel_filename

from .._process_utils import hidden_subprocess_kwargs
from ..toolbox.builtin_resolver import ResolvedBuiltinWheelClosure
from ..toolbox.catalog import ToolboxLockedDistributionSpec, normalize_distribution_name
from ..toolbox.hermetic_environment import ToolboxLockedArtifactSpec
from ..toolbox.identity import identity_digest
from ..toolbox.target import ToolboxTargetIdentity, wheel_is_compatible
from .manager import PackageArtifactManager


class GenericPackageWheelResolver:
    """Use pip only as an offline solver over daemon-indexed generic CAS bytes."""

    def __init__(
        self,
        manager: PackageArtifactManager,
        *,
        target: ToolboxTargetIdentity,
        timeout_seconds: int = 600,
        maximum_artifacts: int = 2048,
    ) -> None:
        self.manager = manager
        self.target = target
        self.timeout_seconds = timeout_seconds
        self.maximum_artifacts = maximum_artifacts
        self.artifacts = {
            source_id: manager.source_artifacts(source_id)
            for source_id in manager.policy.allowed_source_ids
            if source_id in manager.sources and manager.sources[source_id].enabled
        }

    @staticmethod
    def _report_path(raw_url: str) -> Path:
        parsed = urlsplit(raw_url)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError("resolved_artifact_url_invalid")
        raw = url2pathname(unquote(parsed.path))
        if os.name == "nt" and len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
        return Path(raw).resolve()

    def _source_for(self, path: Path) -> str:
        matches = [
            source_id
            for source_id, artifacts in self.artifacts.items()
            if artifacts.get(path.name) == path
        ]
        if len(matches) != 1:
            raise ValueError("resolved_artifact_source_invalid")
        return matches[0]

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        return "sha256:" + digest.hexdigest()

    def resolve_requirements(
        self, *, template_id: str, package_requirements: Sequence[str]
    ) -> ResolvedBuiltinWheelClosure:
        requirements = tuple(str(item or "").strip() for item in package_requirements)
        if not requirements or any(not item for item in requirements):
            raise RuntimeError("required_template_requirements_missing")
        wheelhouses = list(dict.fromkeys(
            path.parent
            for source_id in self.manager.policy.allowed_source_ids
            for path in self.artifacts.get(source_id, {}).values()
        ))
        if not wheelhouses:
            raise RuntimeError("required_template_source_unavailable")
        with tempfile.TemporaryDirectory(prefix="mp13-package-resolve-") as temporary:
            report = Path(temporary) / "report.json"
            command = [
                sys.executable, "-m", "pip", "install", "--dry-run",
                "--ignore-installed", "--disable-pip-version-check", "--no-input",
                "--no-index", "--only-binary=:all:", "--report", str(report),
            ]
            for root in wheelhouses:
                command.extend(("--find-links", str(root)))
            command.extend(requirements)
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
                check=False,
                env={
                    **os.environ,
                    "PIP_CONFIG_FILE": os.devnull,
                    "PIP_NO_INPUT": "1",
                    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                },
                **hidden_subprocess_kwargs(),
            )
            if completed.returncode != 0 or not report.is_file():
                raise RuntimeError("required_template_wheel_missing")
            try:
                installs = list(json.loads(report.read_text(encoding="utf-8"))["install"])
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise RuntimeError("required_template_resolution_invalid") from exc

        distributions: list[ToolboxLockedDistributionSpec] = []
        artifacts: list[ToolboxLockedArtifactSpec] = []
        try:
            for row in installs:
                metadata = dict(row["metadata"])
                name = normalize_distribution_name(metadata["name"])
                version = str(metadata["version"])
                path = self._report_path(row["download_info"]["url"])
                wheel_name, wheel_version, _build, _tags = parse_wheel_filename(path.name)
                if (
                    not path.is_file()
                    or not wheel_is_compatible(path.name, self.target)
                    or normalize_distribution_name(wheel_name) != name
                    or str(wheel_version) != version
                ):
                    raise ValueError("resolved_artifact_metadata_mismatch")
                size = path.stat().st_size
                if size > self.manager.policy.max_artifact_bytes:
                    raise OverflowError
                source_id = self._source_for(path)
                digest = self._sha256(path)
                if self.manager.artifact_path(digest).resolve() != path.resolve() and not self.manager.artifact_path(digest).is_file():
                    raise ValueError("resolved_artifact_cas_missing")
                distributions.append(ToolboxLockedDistributionSpec(name=name, version=version))
                artifacts.append(ToolboxLockedArtifactSpec(
                    distribution_name=name,
                    version=version,
                    source_id=source_id,
                    filename=path.name,
                    sha256=digest,
                    size_bytes=size,
                ))
        except (InvalidWheelFilename, KeyError, TypeError, ValueError, OSError) as exc:
            raise RuntimeError("required_template_resolution_invalid") from exc
        except OverflowError as exc:
            raise RuntimeError("required_template_resolution_bounds_exceeded") from exc
        distributions.sort()
        artifacts.sort()
        if (
            not artifacts
            or len(artifacts) > self.maximum_artifacts
            or len({item.name for item in distributions}) != len(distributions)
        ):
            raise RuntimeError("required_template_resolution_bounds_exceeded")
        return ResolvedBuiltinWheelClosure(
            template_id=str(template_id),
            lock_digest=identity_digest(
                "hosting.package.python_lock.v1",
                {
                    "target": self.target.name,
                    "distributions": [item.to_dict() for item in distributions],
                    "artifacts": [item.to_dict() for item in artifacts],
                },
            ),
            locked_distributions=tuple(distributions),
            locked_artifacts=tuple(artifacts),
        )


__all__ = ["GenericPackageWheelResolver"]
