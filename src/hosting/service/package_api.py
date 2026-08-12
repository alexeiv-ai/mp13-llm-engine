"""Service facade for worker-neutral package operations."""
from __future__ import annotations

import threading
from typing import Any, Dict

from ..packages import PackageArtifactManager, PackagePolicy, PackageSource


class PackageApiMixin:
    _package_manager_guard = threading.Lock()

    @property
    def _package_manager(self) -> PackageArtifactManager:
        current = getattr(self, "_package_manager_instance", None)
        if current is not None:
            return current
        with self._package_manager_guard:
            current = getattr(self, "_package_manager_instance", None)
            if current is not None:
                return current
            current = self._build_package_manager()
            self._package_manager_instance = current
            return current

    def _build_package_manager(self) -> PackageArtifactManager:
        package = dict(self.hosting_configuration.package_management)
        sources = {
            str(source_id): PackageSource.from_dict(
                {
                    "contract": PackageSource.CONTRACT,
                    "source_id": str(source_id),
                    **dict(raw),
                }
            )
            for source_id, raw in dict(package.get("sources") or {}).items()
        }
        raw_policy = dict(package.get("dependency_policy") or {})
        policy = PackagePolicy.from_dict(
            {
                "contract": PackagePolicy.CONTRACT,
                "policy_id": str(raw_policy.get("policy_id") or "default"),
                "revision": int(raw_policy.get("revision") or 1),
                "allowed_source_ids": list(raw_policy.get("allowed_source_ids") or sorted(sources)),
                "allowed_platforms": list(raw_policy.get("allowed_platforms") or ["*"]),
                "allowed_runtimes": list(raw_policy.get("allowed_runtimes") or ["python", "javascript"]),
                "max_artifact_bytes": int(raw_policy.get("max_artifact_bytes") or 64 * 1024 * 1024),
                "require_sha256": bool(raw_policy.get("require_sha256", True)),
                "optional_verifier": raw_policy.get("optional_verifier"),
            }
        )
        return PackageArtifactManager(
            artifact_root=self.hosting_configuration.resolved_paths["artifact_root"],
            lock_root=self.hosting_configuration.resolved_paths["lock_root"],
            scratch_root=self.hosting_configuration.resolved_paths["scratch_root"],
            sources=sources,
            credentials=dict(package.get("credentials") or {}),
            policy=policy,
            configuration_revision=self.hosting_configuration_revision,
        )

    def package_artifact_upload_begin(self, **payload: Any) -> Dict[str, Any]:
        return self._package_manager.begin(**payload)

    def package_artifact_upload_chunk(self, **payload: Any) -> Dict[str, Any]:
        return self._package_manager.chunk(**payload)

    def package_artifact_upload_status(self, **payload: Any) -> Dict[str, Any]:
        return self._package_manager.status(**payload)

    def package_artifact_upload_cancel(self, **payload: Any) -> Dict[str, Any]:
        return self._package_manager.cancel(**payload)

    def package_artifact_upload_commit(self, **payload: Any) -> Dict[str, Any]:
        return self._package_manager.commit(**payload)

    def package_lock_create(self, **payload: Any) -> Dict[str, Any]:
        return self._package_manager.create_lock(**payload)
