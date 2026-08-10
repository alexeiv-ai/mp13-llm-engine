"""Construct immutable built-in template candidates from verified resolution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .builtin_resolver import ResolvedBuiltinWheelClosure
from .catalog import ToolboxEnvironmentTemplateSpec, ToolboxTemplateProvenance
from .host_project_config import ToolboxBuiltinIntent
from .identity import require_digest
from .target import ToolboxTargetIdentity


@dataclass(frozen=True)
class ResolvedBuiltinTemplateCandidate:
    template: ToolboxEnvironmentTemplateSpec
    artifact_references: tuple[dict[str, Any], ...]
    manifest_signature: str
    source_bundle_id: str


def resolved_builtin_template_candidate(
    *,
    intent: ToolboxBuiltinIntent,
    closure: ResolvedBuiltinWheelClosure,
    target: ToolboxTargetIdentity,
    evidence: Mapping[str, Any],
) -> ResolvedBuiltinTemplateCandidate:
    if intent.template_id != closure.template_id:
        raise ValueError("builtin_template_intent_closure_mismatch")
    if intent.sandbox_policy != "compute-only":
        raise ValueError("builtin_template_sandbox_policy_invalid")
    row = dict(evidence or {})
    fields = {
        "bundle_id", "manifest_digest", "source_id", "source_set_revision",
        "target", "signing_key_id", "signature", "artifact_digests",
    }
    if set(row) != fields or row["target"] != target.name:
        raise ValueError("builtin_template_evidence_invalid")
    closure_digests = {item.sha256 for item in closure.locked_artifacts}
    if not closure_digests.issubset(set(row["artifact_digests"])):
        raise ValueError("builtin_template_evidence_incomplete")
    runtime_artifact = next(
        (
            item for item in closure.locked_artifacts
            if item.distribution_name == "mp13-engine"
        ),
        None,
    )
    if runtime_artifact is None:
        raise ValueError("required_template_runtime_artifact_missing")
    manifest_digest = require_digest(
        row["manifest_digest"], label="builtin_template_manifest_digest"
    )
    template = ToolboxEnvironmentTemplateSpec(
        template_id=intent.template_id,
        python_requires=">=3.12,<3.13",
        python_abis=(target.python_abi,),
        runtime_kind="toolbox_python",
        worker_protocol_version="1.0",
        platforms=(target.platform,),
        locked_distributions=closure.locked_distributions,
        exposed_import_roots=intent.imports,
        lock_digest=closure.lock_digest,
        parent_worker_artifact_digest=runtime_artifact.sha256,
        isolation_policy_version="compute-only-v1",
        provenance=ToolboxTemplateProvenance(
            source=f"signed-airgap:{row['source_id']}",
            revision=str(row["bundle_id"]),
            manifest_digest=manifest_digest,
            signing_key_id=str(row["signing_key_id"]),
        ),
    )
    references = tuple(
        {
            "source_id": item.source_id,
            "filename": item.filename,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in closure.locked_artifacts
    )
    signature = str(row["signature"] or "")
    if not signature:
        raise ValueError("builtin_template_signature_missing")
    return ResolvedBuiltinTemplateCandidate(
        template=template,
        artifact_references=references,
        manifest_signature=signature,
        source_bundle_id=str(row["bundle_id"]),
    )


__all__ = ["ResolvedBuiltinTemplateCandidate", "resolved_builtin_template_candidate"]
