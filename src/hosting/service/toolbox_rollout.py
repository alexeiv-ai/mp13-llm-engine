"""Resolved toolbox definition rollout with atomic active-route publication."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..operation_contract import TOOLBOX_DEFINITION_APPLY_COMMITTED_PHASES, HostedOperationLifecycle
from ..toolbox.bundle_models import ResolvedToolboxSandboxAssignment
from ..toolbox.definition_planner import ToolboxDefinitionPlanDraft
from ..toolbox.orchestration import ToolboxSandboxOrchestrator
from ..toolbox.staging import ToolboxBundleStager
from .operation_repository import _replace_with_bounded_retries


class ToolboxDefinitionRolloutCoordinator:
    def __init__(self, service: Any):
        self.service = service

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _progress(
        self,
        operation_id: str,
        *,
        phase: str,
        code: str,
        summary: str,
        cancellable: bool,
        completed_units: int | None = None,
        total_units: int | None = None,
    ) -> dict[str, Any]:
        return self.service._hosted_operations.update_progress(
            operation_id=operation_id,
            progress={
                "phase": phase,
                "code": code,
                "completed_units": completed_units,
                "total_units": total_units,
                "updated_at_ms": self._now_ms(),
                "summary": summary,
                "cancellable": cancellable,
            },
        )

    def _orchestrator(self) -> ToolboxSandboxOrchestrator:
        factory = getattr(self.service, "_toolbox_rollout_orchestrator_factory", None)
        if callable(factory):
            return factory()
        return ToolboxSandboxOrchestrator(
            service=self.service,
            stager=ToolboxBundleStager(self.service.hosting_root),
        )

    def _write_operator_details(self, operation_id: str, payload: Mapping[str, Any]) -> None:
        root = (Path(self.service.hosting_root) / "state" / "toolbox_rollout_operator_details").resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{operation_id}.json"
        fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=root)
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

    @staticmethod
    def _non_restartable_by_name(assignment: ResolvedToolboxSandboxAssignment) -> dict[str, bool]:
        manifest = assignment.bundle_spec.manifest_payload()
        values: dict[str, bool] = {}
        for item in [*list(manifest.get("tools") or []), *list(manifest.get("auto_tools") or [])]:
            values[str(item.get("name") or "").strip()] = bool(item.get("non_restartable", False))
        for name in list(manifest.get("active_intrinsic_tool_names") or []):
            values[str(name or "").strip()] = False
        return {key: value for key, value in values.items() if key}

    def _runtime_payload(
        self,
        *,
        draft: ToolboxDefinitionPlanDraft,
        assignments: Sequence[ResolvedToolboxSandboxAssignment],
        old_snapshot: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        old_profiles = dict(dict(old_snapshot or {}).get("profiles") or {})
        profiles: dict[str, Any] = {}
        routes: dict[str, Any] = {}
        environment_references: list[str] = []
        for assignment in assignments:
            profile = assignment.resolved_profile
            manifest = assignment.bundle_spec.manifest_payload()
            if assignment.classification == "reused":
                source_id = str(assignment.active_profile_id or profile.profile_id)
                active = dict(old_profiles.get(source_id) or {})
                if not active:
                    raise RuntimeError("reused_profile_runtime_missing")
                engine_id = str(active.get("engine_id") or "")
                manifest_hash = str(active.get("manifest_hash") or "")
            else:
                reg = dict(assignment.registration or {})
                engine_id = str(reg.get("engine_id") or "").strip()
                manifest_hash = str(dict(reg.get("bundle") or {}).get("manifest_hash") or "").strip()
                if not engine_id:
                    raise RuntimeError("candidate_registration_missing")
            if len(manifest_hash) == 64 and all(character in "0123456789abcdef" for character in manifest_hash):
                manifest_hash = f"sha256:{manifest_hash}"
            tool_names = sorted(self._non_restartable_by_name(assignment))
            reference = f"toolbox:{draft.definition.toolbox_id}:{profile.profile_id}:{draft.definition.revision}"
            profiles[profile.profile_id] = {
                "profile": profile.to_dict(),
                "manifest_hash": manifest_hash,
                "engine_id": engine_id,
                "tool_names": tool_names,
                "environment_reference": reference,
            }
            environment_references.append(reference)
            for name, non_restartable in self._non_restartable_by_name(assignment).items():
                if name in routes:
                    raise RuntimeError("toolbox_route_duplicate")
                routes[name] = {
                    "profile_id": profile.profile_id,
                    "engine_id": engine_id,
                    "non_restartable": non_restartable,
                }
        return profiles, routes, sorted(environment_references)

    def _drain_old(self, old_snapshot: Mapping[str, Any] | None, active_engine_ids: set[str]) -> dict[str, Any]:
        old_engine_ids = {
            str(dict(item or {}).get("engine_id") or "").strip()
            for item in dict(dict(old_snapshot or {}).get("profiles") or {}).values()
        } - {""}
        retiring = sorted(old_engine_ids - active_engine_ids)
        retired: list[str] = []
        pending: list[str] = []
        if retiring:
            self.service.set_toolbox_registration_routing_states({item: "retired" for item in retiring})
        for engine_id in retiring:
            reg = dict(self.service._find_registration(engine_id) or {})
            environment_key = self.service._toolbox_registration_environment_key(reg) if reg else ""
            resources = self.service._toolbox_runtime_base().resources(environment_key) if environment_key else {}
            if int(dict(resources.get("metrics") or {}).get("active_calls") or 0) > 0:
                pending.append(engine_id)
                continue
            self.service._retire_toolbox_registration(engine_id)
            retired.append(engine_id)
        return {"retired_engine_ids": retired, "drain_pending_engine_ids": pending}

    def apply(
        self,
        *,
        draft: ToolboxDefinitionPlanDraft,
        profile_changes: Sequence[Mapping[str, Any]],
        confirmation_result: Mapping[str, Any] | None = None,
        operation_id: str,
    ) -> dict[str, Any]:
        tid = draft.definition.toolbox_id
        repository = self.service._hosted_operations
        operation = repository.get_by_operation_id(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        if operation["lifecycle"] == HostedOperationLifecycle.QUEUED.value:
            repository.mark_dispatch_claimed(operation_id=operation_id)
        old_snapshot = self.service._toolbox_state_v2.get(tid)
        assignments: list[ResolvedToolboxSandboxAssignment] = []
        candidates: list[str] = []
        published = False
        operator_details: dict[str, Any] = {"toolbox_id": tid, "candidate_engine_ids": []}
        try:
            repository.merge_metadata(
                operation_id=operation_id,
                metadata={"toolbox_id": tid, "definition_revision": draft.definition.revision},
            )
            self._progress(
                operation_id,
                phase="validation",
                code="definition_apply_validated",
                summary="The pinned definition plan is valid.",
                cancellable=True,
            )
            orchestrator = self._orchestrator()
            assignments = orchestrator.build_resolved_assignments(
                toolbox_id=tid,
                profiles=draft.profiles,
                bundles=draft.bundles,
                profile_changes=[dict(item) for item in profile_changes],
            )
            changed = [item for item in assignments if item.classification != "reused"]
            self._progress(
                operation_id,
                phase="environment_build",
                code="definition_apply_environment_build",
                summary="Required verified environments are being acquired.",
                cancellable=True,
                completed_units=0,
                total_units=len(changed),
            )
            self._progress(
                operation_id,
                phase="staging",
                code="definition_apply_staging",
                summary="Changed toolbox profiles are being staged.",
                cancellable=True,
                completed_units=0,
                total_units=len(changed),
            )
            assignments = orchestrator.spawn_resolved_assignments(
                toolbox_id=tid,
                definition_revision=draft.definition.revision,
                assignments=assignments,
            )
            candidates = [
                str(dict(item.registration or {}).get("engine_id") or "").strip()
                for item in assignments
                if item.classification != "reused" and item.registration
            ]
            operator_details["candidate_engine_ids"] = candidates
            repository.merge_metadata(operation_id=operation_id, metadata={"candidate_engine_ids": candidates})
            self._progress(
                operation_id,
                phase="warmup",
                code="definition_apply_warmup",
                summary="Candidate workers are being verified.",
                cancellable=True,
                completed_units=0,
                total_units=len(candidates),
            )
            ready = self.service._ensure_toolbox_assignments_ready(assignments)
            operator_details["readiness"] = ready
            profiles, routes, references = self._runtime_payload(
                draft=draft,
                assignments=assignments,
                old_snapshot=old_snapshot,
            )
            self._progress(
                operation_id,
                phase="publication",
                code="definition_apply_publication",
                summary="The complete toolbox revision is being published.",
                cancellable=False,
                completed_units=0,
                total_units=1,
            )
            snapshot = self.service._toolbox_state_v2.publish(
                toolbox_id=tid,
                expected_revision=draft.definition.expected_revision,
                definition=draft.definition.to_dict(),
                profiles=profiles,
                tool_routes=routes,
                environment_references=references,
                published_at_ms=self._now_ms(),
            )
            published = True
            active_engine_ids = {str(item["engine_id"]) for item in profiles.values()}
            if active_engine_ids:
                self.service.set_toolbox_registration_routing_states(
                    {engine_id: "active" for engine_id in active_engine_ids}
                )
            self._progress(
                operation_id,
                phase="draining",
                code="definition_apply_draining",
                summary="Replaced workers are being drained.",
                cancellable=False,
            )
            drain = self._drain_old(old_snapshot, active_engine_ids)
            operator_details["drain"] = drain
            self._progress(
                operation_id,
                phase="cleanup",
                code="definition_apply_cleanup",
                summary="Rollout cleanup is complete.",
                cancellable=False,
            )
            self._write_operator_details(operation_id, operator_details)
            result = {
                "contract": "hosting.toolbox.definition_apply_result",
                "status": "ok",
                "code": "definition_apply_succeeded",
                "toolbox_id": tid,
                "active_revision": snapshot["active_revision"],
                "active_tool_names": sorted(snapshot["tool_routes"]),
                "accepted_tool_keys": list(dict(confirmation_result or {}).get("accepted_tool_keys") or []),
                "skipped_tools": list(dict(confirmation_result or {}).get("skipped_tools") or []),
                "preserved_active_tool_keys": list(dict(confirmation_result or {}).get("preserved_active_tool_keys") or []),
                "removed_tool_keys": list(dict(confirmation_result or {}).get("removed_tool_keys") or []),
                "package_mutations": list(dict(confirmation_result or {}).get("package_mutations") or []),
                "rollout_summary": {
                    "reused_profiles": sum(item.classification == "reused" for item in assignments),
                    "changed_profiles": len(candidates),
                    "retired_profiles": len(drain["retired_engine_ids"]),
                    "drain_pending_profiles": len(drain["drain_pending_engine_ids"]),
                },
                "user_projection": {
                    "code": "definition_apply_succeeded",
                    "summary": "The toolbox definition is active.",
                },
                "operator_details_available": True,
            }
            return repository.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                envelope=result,
            )
        except Exception as exc:
            current = repository.get_by_operation_id(operation_id)
            if current and current["lifecycle"] == HostedOperationLifecycle.TERMINAL_CANCELLATION.value:
                return repository.status(ref=current["operation"], owner_actor_id=current["owner_actor_id"])
            cleanup: list[str] = []
            if not published:
                for engine_id in candidates:
                    self.service._retire_toolbox_registration(engine_id)
                    cleanup.append(engine_id)
            operator_details.update(
                {
                    "published": published,
                    "cleanup_engine_ids": cleanup,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            self._write_operator_details(operation_id, operator_details)
            code = "definition_apply_post_commit_cleanup_pending" if published else "definition_apply_failed"
            active = self.service._toolbox_state_v2.get(tid)
            return repository.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "contract": "hosting.toolbox.definition_apply_result",
                    "status": "error",
                    "code": code,
                    "toolbox_id": tid,
                    "active_revision": dict(active or {}).get("active_revision"),
                    "diagnostics": [{"code": code, "summary": "The toolbox rollout did not complete cleanly."}],
                    "operator_details_available": True,
                },
                reason=code,
            )

    def recover(self) -> dict[str, Any]:
        """Reconcile registrations and interrupted definition applies from active v2 truth."""

        state = self.service._toolbox_state_v2.read()
        snapshots = dict(state.get("toolboxes") or {})
        active_engine_ids = {
            str(dict(route or {}).get("engine_id") or "").strip()
            for snapshot in snapshots.values()
            for route in dict(dict(snapshot or {}).get("tool_routes") or {}).values()
        } - {""}
        activated: list[str] = []
        retired: list[str] = []
        candidates_removed: list[str] = []
        for reg in list(self.service._read_engines()):
            if str(reg.get("executor_kind") or "") != "toolbox_executor":
                continue
            engine_id = str(reg.get("engine_id") or "").strip()
            routing_state = str(reg.get("routing_state") or "active")
            if engine_id in active_engine_ids:
                if routing_state != "active":
                    self.service.set_toolbox_registration_routing_states({engine_id: "active"})
                    activated.append(engine_id)
                continue
            if routing_state == "candidate":
                self.service._retire_toolbox_registration(engine_id)
                candidates_removed.append(engine_id)
                continue
            toolbox_id = self.service._registration_toolbox_id(reg)
            if toolbox_id in snapshots and routing_state != "retired":
                self.service.set_toolbox_registration_routing_states({engine_id: "retired"})
                retired.append(engine_id)
                environment_key = self.service._toolbox_registration_environment_key(reg)
                resources = self.service._toolbox_runtime_base().resources(environment_key) if environment_key else {}
                if int(dict(resources.get("metrics") or {}).get("active_calls") or 0) == 0:
                    self.service._retire_toolbox_registration(engine_id)

        recovered_operations: list[str] = []
        failed_operations: list[str] = []
        for row in self.service._hosted_operations.active_records(
            execution_kind="toolbox_definition_apply"
        ):
            operation = dict(row.get("operation") or {})
            metadata = dict(row.get("metadata") or {})
            operation_id = str(operation.get("operation_id") or "")
            toolbox_id = str(metadata.get("toolbox_id") or "")
            revision = str(metadata.get("definition_revision") or "")
            snapshot = dict(snapshots.get(toolbox_id) or {})
            progress_phase = str(dict(row.get("progress") or {}).get("phase") or "")
            if (
                revision
                and snapshot.get("active_revision") == revision
                and progress_phase in TOOLBOX_DEFINITION_APPLY_COMMITTED_PHASES
            ):
                self.service._hosted_operations.finish(
                    operation_id=operation_id,
                    lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                    envelope={
                        "contract": "hosting.toolbox.definition_apply_result",
                        "status": "ok",
                        "code": "definition_apply_recovered_after_publication",
                        "toolbox_id": toolbox_id,
                        "active_revision": revision,
                        "active_tool_names": sorted(dict(snapshot.get("tool_routes") or {})),
                        "user_projection": {
                            "code": "definition_apply_recovered_after_publication",
                            "summary": "The published toolbox definition was recovered.",
                        },
                    },
                )
                recovered_operations.append(operation_id)
            else:
                self.service._cleanup_toolbox_definition_apply_candidates(record=row)
                self.service._hosted_operations.finish(
                    operation_id=operation_id,
                    lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                    envelope={
                        "contract": "hosting.toolbox.definition_apply_result",
                        "status": "error",
                        "code": "definition_apply_interrupted_before_publication",
                        "toolbox_id": toolbox_id,
                        "active_revision": snapshot.get("active_revision"),
                        "diagnostics": [
                            {
                                "code": "definition_apply_interrupted_before_publication",
                                "summary": "An interrupted candidate rollout was cleaned up.",
                            }
                        ],
                    },
                    reason="definition_apply_interrupted_before_publication",
                )
                failed_operations.append(operation_id)
        return {
            "status": "ok",
            "activated_engine_ids": sorted(activated),
            "retired_engine_ids": sorted(retired),
            "removed_candidate_engine_ids": sorted(candidates_removed),
            "recovered_operation_ids": sorted(recovered_operations),
            "failed_operation_ids": sorted(failed_operations),
        }


__all__ = ["ToolboxDefinitionRolloutCoordinator"]
