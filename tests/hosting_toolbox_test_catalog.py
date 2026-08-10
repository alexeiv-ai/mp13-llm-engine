"""Explicit realized-template fixtures for planner/catalog unit tests only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hosting.toolbox.catalog import (
    ToolboxEnvironmentTemplateSpec,
    ToolboxLockedDistributionSpec,
    ToolboxTemplateProvenance,
)
from hosting.toolbox.identity import identity_digest
from hosting.toolbox.target import SUPPORTED_TARGET_PLATFORMS


CORE_DISTRIBUTIONS = (
    ("annotated-types", "0.7.0"),
    ("mp13-engine", "0.9.0"),
    ("packaging", "26.0"),
    ("pydantic", "2.12.5"),
    ("pydantic-core", "2.41.5"),
    ("typing-extensions", "4.15.0"),
    ("typing-inspection", "0.4.2"),
)
COMPUTE_DISTRIBUTIONS = tuple(
    sorted(
        {
            *CORE_DISTRIBUTIONS,
            ("mpmath", "1.3.0"),
            ("numexpr", "2.14.1"),
            ("numpy", "2.4.3"),
            ("sympy", "1.14.0"),
        }
    )
)


@dataclass(frozen=True)
class TestTemplateRelease:
    template: ToolboxEnvironmentTemplateSpec
    manifest_signature: str
    artifact: dict[str, object]

    def artifact_reference(self) -> dict[str, object]:
        return dict(self.artifact)


@dataclass(frozen=True)
class TestTemplateCatalog:
    releases: tuple[TestTemplateRelease, ...]

    @property
    def templates(self) -> tuple[ToolboxEnvironmentTemplateSpec, ...]:
        return tuple(item.template for item in self.releases)

    def release(self, template_id: str) -> TestTemplateRelease:
        return next(item for item in self.releases if item.template.template_id == template_id)


def _release(
    template_id: str,
    distributions: tuple[tuple[str, str], ...],
    roots: tuple[str, ...],
) -> TestTemplateRelease:
    lock = tuple(ToolboxLockedDistributionSpec(name=name, version=version) for name, version in distributions)
    lock_digest = identity_digest(
        "hosting.toolbox.test.lock.v1", [item.to_dict() for item in lock]
    )
    manifest_digest = identity_digest(
        "hosting.toolbox.test.manifest.v1", {"template_id": template_id, "lock": lock_digest}
    )
    template = ToolboxEnvironmentTemplateSpec(
        template_id=template_id,
        python_requires=">=3.12,<3.13",
        python_abis=("cp312",),
        runtime_kind="toolbox_python",
        worker_protocol_version="1.0",
        platforms=tuple(sorted(SUPPORTED_TARGET_PLATFORMS)),
        locked_distributions=lock,
        exposed_import_roots=roots,
        lock_digest=lock_digest,
        parent_worker_artifact_digest=identity_digest(
            "hosting.toolbox.test.worker.v1", {"version": "0.9.0"}
        ),
        isolation_policy_version="compute-only-v1",
        provenance=ToolboxTemplateProvenance(
            source="test-only-realized-template",
            revision="fixture-v1",
            manifest_digest=manifest_digest,
            signing_key_id="test-fixture-key",
        ),
    )
    artifact = {
        "source_id": "test-fixture-source",
        "filename": f"{template_id.replace('-', '_')}_fixture-1.0-py3-none-any.whl",
        "sha256": identity_digest("hosting.toolbox.test.artifact.v1", template_id),
        "size_bytes": 1,
    }
    return TestTemplateRelease(template=template, manifest_signature="s" * 64, artifact=artifact)


def realized_test_catalog() -> TestTemplateCatalog:
    return TestTemplateCatalog(
        releases=(
            _release(
                "core",
                CORE_DISTRIBUTIONS,
                ("hosting", "mp13_engine", "packaging", "pydantic"),
            ),
            _release(
                "py-compute",
                COMPUTE_DISTRIBUTIONS,
                (
                    "hosting", "mp13_engine", "mpmath", "numexpr", "numpy",
                    "packaging", "pydantic", "sympy",
                ),
            ),
        )
    )


def publish_realized_test_catalog(service: Any) -> None:
    """Publish explicit test-only releases into a service catalog."""
    for release in realized_test_catalog().releases:
        service.toolbox_template_publish(
            template=release.template.to_dict(),
            artifact_references=[release.artifact_reference()],
            manifest_signature=release.manifest_signature,
            activate=True,
            actor_id="test:realized-template-fixture",
        )


__all__ = [
    "TestTemplateCatalog",
    "TestTemplateRelease",
    "publish_realized_test_catalog",
    "realized_test_catalog",
]
