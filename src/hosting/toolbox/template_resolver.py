"""Shared pure template resolution for isolated hosted Python consumer classes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mp13_engine.mp13_intrinsics_metadata import intrinsic_dependency_metadata

from .bundle_models import ToolboxBundleFile
from .catalog import ToolboxEnvironmentTemplateSpec
from .dependency_analysis import (
    ToolboxResolvedDependencies,
    ToolboxSourceAnalysis,
    analyze_toolbox_bundle_imports,
    resolve_toolbox_dependencies,
    select_toolbox_environment_template,
)
from .identity import identity_digest, require_digest
from .shipped_templates import compute_only_sandbox_policy


TEMPLATE_RUNTIME_BINDING_DOMAIN = "hosting.toolbox.template_runtime_binding.v1"
SUPPORTED_TEMPLATE_CONSUMERS = frozenset(
    {
        "toolbox",
        "workflow_python_node",
        "workflow_python_snippet",
        "workflow_python_helper",
    }
)
_RUNTIME_FAMILY = {
    "toolbox": "toolbox_executor",
    "workflow_python_node": "workflow_python_node",
    "workflow_python_snippet": "workflow_python_node",
    "workflow_python_helper": "workflow_python_helper",
}


@dataclass(frozen=True)
class VerifiedTemplateCandidate:
    template: ToolboxEnvironmentTemplateSpec
    template_digest: str
    environment_digest: str
    python_abi: str
    platform: str

    def __post_init__(self) -> None:
        if not isinstance(self.template, ToolboxEnvironmentTemplateSpec):
            raise ValueError("verified_template_candidate_template_invalid")
        require_digest(self.template_digest, label="verified_template_digest")
        require_digest(self.environment_digest, label="verified_environment_digest")
        if not self.python_abi or not self.platform:
            raise ValueError("verified_template_candidate_target_invalid")


@dataclass(frozen=True)
class HostedTemplateRuntimeBinding:
    consumer_kind: str
    runtime_family: str
    template_id: str
    template_digest: str
    lock_digest: str
    environment_digest: str
    python_abi: str
    platform: str
    sandbox_policy_digest: str
    binding_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "consumer_kind": self.consumer_kind,
            "runtime_family": self.runtime_family,
            "template_id": self.template_id,
            "template_digest": self.template_digest,
            "lock_digest": self.lock_digest,
            "environment_digest": self.environment_digest,
            "python_abi": self.python_abi,
            "platform": self.platform,
            "sandbox_policy_digest": self.sandbox_policy_digest,
            "binding_id": self.binding_id,
        }


@dataclass(frozen=True)
class HostedTemplateResolution:
    analysis: ToolboxSourceAnalysis
    dependencies: ToolboxResolvedDependencies
    binding: HostedTemplateRuntimeBinding

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding": self.binding.to_dict(),
            "imports": [item.to_dict() for item in self.analysis.imports],
            "diagnostics": [item.to_dict() for item in self.analysis.diagnostics],
            "requirements": [item.to_dict() for item in self.dependencies.requirements],
        }


def resolve_verified_template_environment(
    *,
    consumer_kind: str,
    files: Sequence[ToolboxBundleFile | Mapping[str, Any]],
    candidates: Sequence[VerifiedTemplateCandidate],
    python_abi: str,
    platform: str,
    declared_imports: Sequence[str] = (),
    package_requirements: Sequence[str] = (),
    intrinsic_names: Sequence[str] = (),
    allowed_template_ids: Sequence[str] | None = None,
    sandbox_policy: Mapping[str, Any] | None = None,
) -> HostedTemplateResolution:
    consumer = str(consumer_kind or "").strip()
    if consumer not in SUPPORTED_TEMPLATE_CONSUMERS:
        raise ValueError("template_consumer_kind_invalid")
    intrinsic = intrinsic_dependency_metadata(intrinsic_names)
    effective_imports = tuple(sorted(set(declared_imports) | set(intrinsic["import_roots"])))
    effective_requirements = tuple(
        sorted(set(package_requirements) | set(intrinsic["package_requirements"]))
    )
    analysis = analyze_toolbox_bundle_imports(files, declared_imports=effective_imports)
    dependencies = resolve_toolbox_dependencies(
        analysis,
        package_requirements=effective_requirements,
    )
    exact_candidates = [
        item for item in candidates
        if item.python_abi == python_abi and item.platform == platform
    ]
    if not exact_candidates:
        raise ValueError("verified_template_target_unavailable")
    selection = select_toolbox_environment_template(
        dependencies,
        [item.template for item in exact_candidates],
        python_abi=python_abi,
        platform=platform,
        allowed_template_ids=allowed_template_ids,
    )
    if selection.mode != "template" or selection.custom_delta:
        raise ValueError("verified_template_custom_delta_required")
    candidate = next(item for item in exact_candidates if item.template == selection.template)
    effective_sandbox = dict(sandbox_policy or compute_only_sandbox_policy())
    sandbox_digest = identity_digest(
        "hosting.toolbox.effective_sandbox_policy.v1", effective_sandbox
    )
    binding_payload = {
        "consumer_kind": consumer,
        "runtime_family": _RUNTIME_FAMILY[consumer],
        "template_id": candidate.template.template_id,
        "template_digest": candidate.template_digest,
        "lock_digest": candidate.template.lock_digest,
        "environment_digest": candidate.environment_digest,
        "python_abi": python_abi,
        "platform": platform,
        "sandbox_policy_digest": sandbox_digest,
    }
    return HostedTemplateResolution(
        analysis=analysis,
        dependencies=dependencies,
        binding=HostedTemplateRuntimeBinding(
            **binding_payload,
            binding_id=identity_digest(TEMPLATE_RUNTIME_BINDING_DOMAIN, binding_payload),
        ),
    )


__all__ = [
    "HostedTemplateResolution",
    "HostedTemplateRuntimeBinding",
    "SUPPORTED_TEMPLATE_CONSUMERS",
    "VerifiedTemplateCandidate",
    "resolve_verified_template_environment",
]
