"""Strict loader for the two immutable toolbox templates shipped by the parent."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, Mapping, Sequence

from ..sandbox.policy import WorkerSandboxPolicy
from .catalog import ToolboxEnvironmentTemplateSpec, ToolboxLockedDistributionSpec
from .identity import canonical_json_bytes, identity_digest


SHIPPED_CATALOG_CONTRACT = "hosting.toolbox.shipped_catalog.v1"
SHIPPED_LOCK_CONTRACT = "hosting.toolbox.distribution_lock.v1"
SHIPPED_CATALOG_RESOURCE = "pkg:hosting.resources/toolbox_templates/catalog.json"
SHIPPED_TEMPLATE_IDS = ("core", "py-compute")
LOCK_IDENTITY_DOMAIN = "hosting.toolbox.shipped_distribution_lock.v1"
MANIFEST_IDENTITY_DOMAIN = "hosting.toolbox.shipped_template_manifest.v1"
WORKER_ARTIFACT_IDENTITY_DOMAIN = "hosting.toolbox.parent_worker_artifact.v1"

_FORBIDDEN_CAPABILITY_KEYS = frozenset(
    {
        "allow_execution",
        "allow_resolution",
        "approval",
        "approval_ref",
        "approved",
        "artifact_roots",
        "brokered_io",
        "capabilities",
        "filesystem",
        "host_api_permissions",
        "network",
        "sandbox",
        "sandbox_policy",
        "subprocess",
    }
)

_COMPUTE_ONLY_POLICY = {
    "policy_id": "compute-only",
    "sandbox_required": True,
    "filesystem_read_roots": [],
    "filesystem_write_roots": [],
    "artifact_roots": [],
    "network": False,
    "subprocess": False,
    "brokered_io": {"filesystem": False, "http": False, "subprocess": False},
    "host_api_permissions": [],
}


def compute_only_sandbox_policy() -> dict[str, Any]:
    return copy.deepcopy(_COMPUTE_ONLY_POLICY)


def compute_only_worker_policy() -> WorkerSandboxPolicy:
    policy = WorkerSandboxPolicy.from_mapping(
        {
            "sandbox": {
                "enabled": True,
                "profile": "compute-only",
                "filesystem": {"default_access": "deny", "rules": []},
                "artifact_roots": {},
                "process": {"allow_subprocess": False, "inherit_parent_handles": False},
                "network": {"mode": "disabled"},
                "brokered_io": {"filesystem": False, "http": False, "subprocess": False},
            }
        }
    )
    if (
        not policy.enabled
        or policy.filesystem_rules
        or policy.artifact_roots
        or policy.process.allow_subprocess
        or policy.process.inherit_parent_handles
        or policy.network.mode != "disabled"
        or policy.brokered_io.filesystem
        or policy.brokered_io.http
        or policy.brokered_io.subprocess
    ):
        raise ValueError("compute_only_policy_invalid")
    return policy


def _strict_fields(row: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    unknown = sorted(set(row) - fields)
    missing = sorted(fields - set(row))
    if unknown:
        raise ValueError(f"{label}_unknown_fields:{','.join(unknown)}")
    if missing:
        raise ValueError(f"{label}_missing_fields:{','.join(missing)}")


def _reject_capability_metadata(value: Any, *, path: str = "resource") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key)
            if name in _FORBIDDEN_CAPABILITY_KEYS:
                raise ValueError(f"shipped_package_capability_metadata_denied:{path}.{name}")
            _reject_capability_metadata(nested, path=f"{path}.{name}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_capability_metadata(nested, path=f"{path}[{index}]")


def _read_json_resource(name: str) -> tuple[dict[str, Any], bytes]:
    resource = resources.files("hosting.resources.toolbox_templates").joinpath(name)
    raw = resource.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"shipped_template_resource_invalid:{name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"shipped_template_resource_invalid:{name}")
    _reject_capability_metadata(payload)
    return payload, raw


@dataclass(frozen=True)
class ShippedToolboxTemplateRelease:
    template: ToolboxEnvironmentTemplateSpec
    lock_resource: str
    artifact_sha256: str
    artifact_size_bytes: int
    artifact_source_id: str
    manifest_signature: str

    def artifact_reference(self) -> dict[str, Any]:
        return {
            "source_id": self.artifact_source_id,
            "filename": self.lock_resource,
            "sha256": self.artifact_sha256,
            "size_bytes": self.artifact_size_bytes,
        }


@dataclass(frozen=True)
class ShippedToolboxCatalog:
    revision: str
    releases: tuple[ShippedToolboxTemplateRelease, ...]
    compute_only_policy: Mapping[str, Any]
    resource: str = SHIPPED_CATALOG_RESOURCE

    @property
    def templates(self) -> tuple[ToolboxEnvironmentTemplateSpec, ...]:
        return tuple(item.template for item in self.releases)

    def release(self, template_id: str) -> ShippedToolboxTemplateRelease:
        for item in self.releases:
            if item.template.template_id == template_id:
                return item
        raise ValueError("shipped_template_not_found")


def _load_lock(template_id: str, resource_name: str) -> tuple[tuple[ToolboxLockedDistributionSpec, ...], str, str, int]:
    try:
        payload, raw = _read_json_resource(resource_name)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("required_template_lock_invalid") from exc
    _strict_fields(payload, {"contract", "template_id", "distributions"}, label="shipped_template_lock")
    if payload["contract"] != SHIPPED_LOCK_CONTRACT or payload["template_id"] != template_id:
        raise ValueError("shipped_template_lock_identity_invalid")
    if not isinstance(payload["distributions"], list):
        raise ValueError("shipped_template_lock_distributions_invalid")
    distributions = tuple(ToolboxLockedDistributionSpec.from_dict(item) for item in payload["distributions"])
    if tuple(sorted(distributions)) != distributions:
        raise ValueError("shipped_template_lock_order_invalid")
    lock_digest = identity_digest(
        LOCK_IDENTITY_DOMAIN,
        {"template_id": template_id, "distributions": [item.to_dict() for item in distributions]},
    )
    artifact_digest = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    return distributions, lock_digest, artifact_digest, len(raw)


def load_shipped_toolbox_catalog() -> ShippedToolboxCatalog:
    payload, _ = _read_json_resource("catalog.json")
    fields = {
        "contract", "revision", "python_requires", "python_abis", "platforms",
        "runtime_kind", "worker_protocol_version", "isolation_policy_version",
        "artifact_source_id", "signing_key_id", "templates",
    }
    _strict_fields(payload, fields, label="shipped_template_catalog")
    if payload["contract"] != SHIPPED_CATALOG_CONTRACT:
        raise ValueError("shipped_template_catalog_contract_invalid")
    if not isinstance(payload["templates"], list) or len(payload["templates"]) != 2:
        raise ValueError("shipped_template_catalog_templates_invalid")
    entries = []
    for item in payload["templates"]:
        if not isinstance(item, dict):
            raise ValueError("shipped_template_catalog_entry_invalid")
        _strict_fields(
            item,
            {"template_id", "lock_resource", "exposed_import_roots"},
            label="shipped_template_catalog_entry",
        )
        entries.append(item)
    if tuple(item["template_id"] for item in entries) != SHIPPED_TEMPLATE_IDS:
        raise ValueError("shipped_template_ids_invalid")
    worker_artifact_digest = identity_digest(
        WORKER_ARTIFACT_IDENTITY_DOMAIN,
        {
            "distribution": "mp13-engine",
            "version": "0.9.0",
            "worker_protocol_version": payload["worker_protocol_version"],
            "catalog_revision": payload["revision"],
        },
    )
    releases: list[ShippedToolboxTemplateRelease] = []
    for entry in entries:
        distributions, lock_digest, artifact_digest, artifact_size = _load_lock(
            entry["template_id"], entry["lock_resource"]
        )
        manifest_payload = {
            "template_id": entry["template_id"],
            "python_requires": payload["python_requires"],
            "python_abis": payload["python_abis"],
            "runtime_kind": payload["runtime_kind"],
            "worker_protocol_version": payload["worker_protocol_version"],
            "platforms": payload["platforms"],
            "locked_distributions": [item.to_dict() for item in distributions],
            "exposed_import_roots": entry["exposed_import_roots"],
            "lock_digest": lock_digest,
            "parent_worker_artifact_digest": worker_artifact_digest,
            "isolation_policy_version": payload["isolation_policy_version"],
            "artifact": {
                "source_id": payload["artifact_source_id"],
                "filename": entry["lock_resource"],
                "sha256": artifact_digest,
                "size_bytes": artifact_size,
            },
        }
        manifest_digest = identity_digest(MANIFEST_IDENTITY_DOMAIN, manifest_payload)
        template = ToolboxEnvironmentTemplateSpec.from_dict(
            {
                **{key: manifest_payload[key] for key in (
                    "template_id", "python_requires", "python_abis", "runtime_kind",
                    "worker_protocol_version", "platforms", "locked_distributions",
                    "exposed_import_roots", "lock_digest", "parent_worker_artifact_digest",
                    "isolation_policy_version",
                )},
                "provenance": {
                    "source": SHIPPED_CATALOG_RESOURCE,
                    "revision": payload["revision"],
                    "manifest_digest": manifest_digest,
                    "signing_key_id": payload["signing_key_id"],
                },
            }
        )
        signature = base64.urlsafe_b64encode(
            hashlib.sha512(canonical_json_bytes(manifest_payload)).digest()
        ).decode("ascii").rstrip("=")
        releases.append(
            ShippedToolboxTemplateRelease(
                template=template,
                lock_resource=entry["lock_resource"],
                artifact_sha256=artifact_digest,
                artifact_size_bytes=artifact_size,
                artifact_source_id=payload["artifact_source_id"],
                manifest_signature=signature,
            )
        )
    compute_only_worker_policy()
    return ShippedToolboxCatalog(
        revision=payload["revision"],
        releases=tuple(releases),
        compute_only_policy=compute_only_sandbox_policy(),
    )


__all__ = [
    "SHIPPED_CATALOG_RESOURCE",
    "SHIPPED_TEMPLATE_IDS",
    "ShippedToolboxCatalog",
    "ShippedToolboxTemplateRelease",
    "compute_only_sandbox_policy",
    "compute_only_worker_policy",
    "load_shipped_toolbox_catalog",
]
