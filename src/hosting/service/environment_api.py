"""Service facade for worker-neutral environment lifecycle operations."""
from __future__ import annotations

import threading
from typing import Any, Dict, Mapping

from ..environments import (
    EnvironmentManager,
    EnvironmentRequest,
    EnvironmentTemplate,
    ManifestEnvironmentBuilder,
)


class EnvironmentApiMixin:
    _environment_manager_guard = threading.Lock()

    @property
    def _environment_manager(self) -> EnvironmentManager:
        current = getattr(self, "_environment_manager_instance", None)
        if current is not None:
            return current
        with self._environment_manager_guard:
            current = getattr(self, "_environment_manager_instance", None)
            if current is None:
                current = self._build_environment_manager()
                self._environment_manager_instance = current
            return current

    def _build_environment_manager(self) -> EnvironmentManager:
        environment = dict(self.hosting_configuration.environment_management)
        retention = dict(environment.get("retention") or {})
        builders = {
            "python-manifest-v1": ManifestEnvironmentBuilder(builder_id="python-manifest-v1", runtime_kind="python"),
            "javascript-manifest-v1": ManifestEnvironmentBuilder(builder_id="javascript-manifest-v1", runtime_kind="javascript"),
        }
        return EnvironmentManager(
            environment_root=self.hosting_configuration.resolved_paths["environment_root"],
            scratch_root=self.hosting_configuration.resolved_paths["scratch_root"],
            package_lock_root=self.hosting_configuration.resolved_paths["lock_root"],
            configuration_revision=self.hosting_configuration_revision,
            builders=builders,
            retention_seconds=int(retention.get("unused_seconds") or 0),
        )

    def environment_template_list(self, **payload: Any) -> Dict[str, Any]:
        return self._environment_manager.list_templates(include_revoked=bool(payload.get("include_revoked", False)))

    def environment_template_describe(self, **payload: Any) -> Dict[str, Any]:
        revision = payload.get("revision")
        return self._environment_manager.describe_template(
            template_id=str(payload.get("template_id") or ""),
            revision=int(revision) if revision is not None else None,
        )

    def environment_template_construct(self, **payload: Any) -> Dict[str, Any]:
        template_value = payload.get("template")
        raw: Mapping[str, Any] = (
            template_value if isinstance(template_value, Mapping) else payload
        )
        return self._environment_manager.put_template(
            EnvironmentTemplate.from_dict(dict(raw))
        )

    def environment_template_activate(self, **payload: Any) -> Dict[str, Any]:
        return self._set_environment_template_state(payload, "active")

    def environment_template_replace(self, **payload: Any) -> Dict[str, Any]:
        result = self.environment_template_construct(**payload)
        return self._environment_manager.set_template_state(
            template_id=str(result["template_id"]), revision=int(result["revision"]), state_value="active"
        )

    def environment_template_deprecate(self, **payload: Any) -> Dict[str, Any]:
        return self._set_environment_template_state(payload, "deprecated")

    def environment_template_revoke(self, **payload: Any) -> Dict[str, Any]:
        return self._set_environment_template_state(payload, "revoked")

    def _set_environment_template_state(self, payload: Mapping[str, Any], state: str) -> Dict[str, Any]:
        return self._environment_manager.set_template_state(
            template_id=str(payload.get("template_id") or ""),
            revision=int(payload.get("revision") or 0),
            state_value=state,
        )

    def environment_ensure(self, **payload: Any) -> Dict[str, Any]:
        request_value = payload.get("request")
        raw: Mapping[str, Any] = (
            request_value if isinstance(request_value, Mapping) else payload
        )
        return self._environment_manager.ensure(EnvironmentRequest.from_dict(dict(raw)))

    def environment_template_prewarm(self, **payload: Any) -> Dict[str, Any]:
        return self.environment_ensure(**payload)

    def environment_reference_release(self, **payload: Any) -> Dict[str, Any]:
        return self._environment_manager.release(reference_id=str(payload.get("reference_id") or ""))

    def environment_reference_list(self, **payload: Any) -> Dict[str, Any]:
        return self._environment_manager.list_references(cursor=str(payload.get("cursor") or ""), limit=int(payload.get("limit") or 100))

    def environment_execution_begin(self, **payload: Any) -> Dict[str, Any]:
        return self._environment_manager.execution_begin(environment_id=str(payload.get("environment_id") or ""), execution_id=str(payload.get("execution_id") or ""))

    def environment_execution_end(self, **payload: Any) -> Dict[str, Any]:
        return self._environment_manager.execution_end(execution_id=str(payload.get("execution_id") or ""))

    def environment_remove(
        self,
        *,
        environment_id: str,
        request_id: str,
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        return self._environment_remove_start(
            environment_id=environment_id,
            request_id=request_id,
            owner_actor_id=owner_actor_id,
        )

    def environment_gc(self, **payload: Any) -> Dict[str, Any]:
        return self._environment_manager.gc()
