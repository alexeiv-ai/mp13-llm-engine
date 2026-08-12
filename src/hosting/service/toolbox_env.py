"""Route-based toolbox consistency, recovery, repair, references, and GC."""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..operation_contract import (
    HostedExecutionKind,
    HostedOperationLifecycle,
    HostedOperationProgress,
    HostedOperationSelector,
    hosted_execution_fingerprint,
)
from ..toolbox.identity import require_digest


class ToolboxMaintenanceMixin:
    @staticmethod
    def _contains_environment_digest(value: Any, environment_digest: str) -> bool:
        if isinstance(value, str):
            return value == environment_digest
        if isinstance(value, Mapping):
            return any(
                ToolboxMaintenanceMixin._contains_environment_digest(item, environment_digest)
                for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(
                ToolboxMaintenanceMixin._contains_environment_digest(item, environment_digest)
                for item in value
            )
        return False

    def _environment_removal_blockers(
        self, *, environment_digest: str, operation_id: str = ""
    ) -> list[str]:
        key = require_digest(environment_digest, label="environment_digest")
        blockers: set[str] = set()
        configuration = getattr(self, "_toolbox_host_project_config", None)
        if configuration is not None and key in set(configuration.retention.protected_digests):
            blockers.add("protected")

        for snapshot in self._toolbox_v2_snapshots().values():
            for raw in dict(snapshot.get("profiles") or {}).values():
                if str(dict(dict(raw or {}).get("profile") or {}).get("environment_key") or "") == key:
                    blockers.add("active")
        for registration in self._toolbox_v2_registrations().values():
            if (
                str(registration.get("routing_state") or "") == "candidate"
                and self._toolbox_registration_environment_key(registration) == key
            ):
                blockers.add("candidate")

        now_ms = int(time.time() * 1000)
        for plan in self._toolbox_definition_plans.list(now_ms=now_ms):
            if self._contains_environment_digest(plan.to_dict(), key):
                blockers.add("plan")
        confirmation_state = self._toolbox_confirmations._read()  # noqa: SLF001 - same service boundary
        for raw in dict(confirmation_state.get("receipts") or {}).values():
            if int(dict(raw or {}).get("expires_at_ms") or 0) > now_ms and self._contains_environment_digest(raw, key):
                blockers.add("confirmation")
        for record in self._hosted_operations.active_records():
            current_id = str(dict(dict(record or {}).get("operation") or {}).get("operation_id") or "")
            if current_id != str(operation_id or "") and self._contains_environment_digest(
                dict(record.get("metadata") or {}), key
            ):
                blockers.add("operation")

        builder = getattr(self, "_hermetic_toolbox_environment_builder", None)
        if builder is not None:
            references = dict(builder._read_references_unlocked().get("environments") or {})  # noqa: SLF001
            for reference_id in dict(references.get(key) or {}):
                if str(reference_id).startswith("template:"):
                    blockers.add("built_in")
                else:
                    blockers.add("reference")
        return [item for item in (
            "active", "candidate", "plan", "confirmation", "operation",
            "built_in", "protected", "reference",
        ) if item in blockers]

    def toolbox_environment_remove(
        self,
        *,
        environment_digest: str,
        request_id: str,
        owner_actor_id: str = "service:local",
    ) -> dict[str, Any]:
        key = require_digest(environment_digest, label="environment_digest")
        rid = str(request_id or "").strip()
        if not rid:
            raise ValueError("toolbox_environment_remove_request_id_required")
        configuration = getattr(self, "_toolbox_host_project_config", None)
        if configuration is None:
            raise ValueError("toolbox_host_project_configuration_required")
        fingerprint = hosted_execution_fingerprint(
            {
                "execution_kind": HostedExecutionKind.TOOLBOX_ENVIRONMENT_REMOVE.value,
                "configuration_revision": self.hosting_configuration_revision,
                "environment_digest": key,
                "config_revision": configuration.config_revision,
                "source_set_revision": configuration.source_set_revision,
            }
        )
        owner = self._operation_owner(owner_actor_id)
        prepared = self._hosted_operations.prepare(
            owner_actor_id=owner,
            execution_kind=HostedExecutionKind.TOOLBOX_ENVIRONMENT_REMOVE,
            selector=HostedOperationSelector(kind="environment_digest", id=key),
            namespace=f"environment_remove:{key}",
            request_id=rid,
            fingerprint=fingerprint,
            metadata={
                "configuration_revision": self.hosting_configuration_revision,
                "environment_digest": key,
                "config_revision": configuration.config_revision,
                "source_set_revision": configuration.source_set_revision,
            },
        )
        status = prepared.get("status")
        if status is None:
            raise RuntimeError("hosted_operation_capacity")
        if prepared["action"] != "dispatch":
            return dict(status)
        operation_id = str(status["operation"]["operation_id"])
        thread = threading.Thread(
            target=self._run_toolbox_environment_remove,
            kwargs={"operation_id": operation_id, "environment_digest": key},
            name=f"environment-remove-{operation_id[-8:]}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "contract": "hosting.toolbox.environment_remove_result.v1",
                    "status": "error",
                    "code": "environment_remove_dispatch_failed",
                },
                reason="environment_remove_dispatch_failed",
            )
            raise
        return dict(status)

    def _run_toolbox_environment_remove(
        self, *, operation_id: str, environment_digest: str
    ) -> None:
        self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)

        def progress(
            phase: str,
            code: str,
            summary: str,
            *,
            completed_units: int | None = None,
            total_units: int | None = None,
            cancellable: bool,
        ) -> None:
            self._hosted_operations.update_progress(
                operation_id=operation_id,
                progress=HostedOperationProgress(
                    phase=phase,
                    code=code,
                    completed_units=completed_units,
                    total_units=total_units,
                    updated_at_ms=int(time.time() * 1000),
                    summary=summary,
                    cancellable=cancellable,
                ),
            )

        try:
            progress(
                "validation", "environment_remove_validated",
                "The exact environment digest and retention revision were validated.",
                completed_units=1, total_units=1, cancellable=True,
            )
            progress(
                "reference_check", "environment_remove_references_checked",
                "Active, candidate, plan, receipt, operation, and built-in references were checked.",
                completed_units=1, total_units=1, cancellable=True,
            )
            blockers = self._environment_removal_blockers(
                environment_digest=environment_digest, operation_id=operation_id
            )
            if blockers:
                return self._hosted_operations.finish(
                    operation_id=operation_id,
                    lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                    envelope={
                        "contract": "hosting.toolbox.environment_remove_result.v1",
                        "status": "blocked",
                        "code": "environment_removal_blocked",
                        "environment_digest": environment_digest,
                        "blocking_reference_kinds": blockers,
                    },
                )
            builder = getattr(self, "_hermetic_toolbox_environment_builder", None)
            if builder is None:
                raise ValueError("toolbox_environment_builder_unconfigured")
            progress(
                "removal", "environment_remove_started",
                "The exact unreferenced environment is being removed.",
                completed_units=0, total_units=1, cancellable=False,
            )
            result = builder.remove_environment(environment_key=environment_digest)
            progress(
                "cleanup", "environment_remove_complete",
                "The exact environment removal completed.",
                completed_units=1, total_units=1, cancellable=False,
            )
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                envelope={
                    "contract": "hosting.toolbox.environment_remove_result.v1",
                    "status": result,
                    "code": f"environment_{result}",
                    "environment_digest": environment_digest,
                },
            )
        except Exception as exc:
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "contract": "hosting.toolbox.environment_remove_result.v1",
                    "status": "error",
                    "code": "environment_remove_failed",
                    "environment_digest": environment_digest,
                    "diagnostics": [{"code": "environment_remove_failed", "summary": str(exc)}],
                },
                reason="environment_remove_failed",
            )
    def _toolbox_v2_snapshots(self) -> Dict[str, Dict[str, Any]]:
        return {
            str(toolbox_id): dict(snapshot or {})
            for toolbox_id, snapshot in dict(self._toolbox_state_v2.read().get("toolboxes") or {}).items()
        }

    def _toolbox_v2_registrations(self) -> Dict[str, Dict[str, Any]]:
        return {
            str(row.get("engine_id") or "").strip(): dict(row)
            for row in self._read_engines()
            if str(row.get("executor_kind") or "").strip() == "toolbox_executor"
            and str(row.get("engine_id") or "").strip()
        }

    def _active_candidate_engine_ids(self) -> set[str]:
        return {
            str(item or "").strip()
            for record in self._hosted_operations.active_records()
            for item in list(dict(record.get("metadata") or {}).get("candidate_engine_ids") or [])
            if str(item or "").strip()
        }

    def _toolbox_reference_report(self) -> Dict[str, Any]:
        snapshots = self._toolbox_v2_snapshots()
        registrations = self._toolbox_v2_registrations()
        active_engine_ids = {
            str(profile.get("engine_id") or "").strip()
            for snapshot in snapshots.values()
            for profile in dict(snapshot.get("profiles") or {}).values()
            if str(dict(profile or {}).get("engine_id") or "").strip()
        }
        toolboxes: Dict[str, Any] = {}
        referenced_bundle_roots: set[str] = set()
        for toolbox_id, snapshot in snapshots.items():
            profiles: Dict[str, Any] = {}
            for profile_id, raw_profile in dict(snapshot.get("profiles") or {}).items():
                profile = dict(raw_profile or {})
                engine_id = str(profile.get("engine_id") or "").strip()
                registration = dict(registrations.get(engine_id) or {})
                bundle_root = str(dict(registration.get("bundle") or {}).get("bundle_root") or "").strip()
                if bundle_root:
                    try:
                        referenced_bundle_roots.add(str(Path(bundle_root).expanduser().resolve()))
                    except Exception:
                        pass
                profiles[str(profile_id)] = {
                    "engine_id": engine_id,
                    "manifest_hash": profile.get("manifest_hash"),
                    "environment_reference": profile.get("environment_reference"),
                    "tool_names": list(profile.get("tool_names") or []),
                    "registration_present": bool(registration),
                    "routing_state": str(registration.get("routing_state") or "") or None,
                }
            toolboxes[toolbox_id] = {
                "active_revision": snapshot.get("active_revision"),
                "published_at_ms": snapshot.get("published_at_ms"),
                "profiles": profiles,
                "tool_routes": dict(snapshot.get("tool_routes") or {}),
                "environment_references": list(snapshot.get("environment_references") or []),
            }
        candidates = sorted(
            engine_id
            for engine_id, row in registrations.items()
            if str(row.get("routing_state") or "") == "candidate"
        )
        retired = sorted(
            engine_id
            for engine_id, row in registrations.items()
            if str(row.get("routing_state") or "") == "retired"
        )
        stale = sorted(set(registrations) - active_engine_ids)
        bundles_root = (self.hosting_root / "toolbox_bundles").resolve()
        stale_bundles: List[str] = []
        if bundles_root.exists():
            for directory in bundles_root.rglob("manifest.json"):
                root = str(directory.parent.resolve())
                if root not in referenced_bundle_roots:
                    stale_bundles.append(root)
        return {
            "status": "ok",
            "contract": "hosting.toolbox.references.v2",
            "toolboxes": toolboxes,
            "registrations": registrations,
            "candidate_engine_ids": candidates,
            "retired_engine_ids": retired,
            "stale_engine_ids": stale,
            "stale_bundle_roots": sorted(set(stale_bundles)),
            "summary": {
                "toolbox_count": len(toolboxes),
                "active_registration_count": len(active_engine_ids & set(registrations)),
                "candidate_registration_count": len(candidates),
                "retired_registration_count": len(retired),
                "stale_engine_count": len(stale),
                "stale_bundle_count": len(set(stale_bundles)),
                "stale_environment_count": 0,
            },
        }

    def toolbox_references(self) -> Dict[str, Any]:
        return self._toolbox_reference_report()

    def toolbox_consistency(self) -> Dict[str, Any]:
        references = self._toolbox_reference_report()
        registrations = dict(references["registrations"])
        issues: List[Dict[str, Any]] = []
        routed_engine_ids: set[str] = set()
        for toolbox_id, raw_snapshot in dict(references["toolboxes"]).items():
            snapshot = dict(raw_snapshot or {})
            profiles = dict(snapshot.get("profiles") or {})
            routes = dict(snapshot.get("tool_routes") or {})
            for profile_id, raw_profile in profiles.items():
                profile = dict(raw_profile or {})
                engine_id = str(profile.get("engine_id") or "").strip()
                routed_engine_ids.add(engine_id)
                registration = dict(registrations.get(engine_id) or {})
                if not registration:
                    issues.append({
                        "toolbox_id": toolbox_id,
                        "profile_id": profile_id,
                        "engine_id": engine_id,
                        "issue": "missing_active_registration",
                    })
                    continue
                if str(registration.get("routing_state") or "") != "active":
                    issues.append({
                        "toolbox_id": toolbox_id,
                        "profile_id": profile_id,
                        "engine_id": engine_id,
                        "issue": "active_registration_not_routable",
                    })
                if self._registration_toolbox_id(registration) != toolbox_id:
                    issues.append({
                        "toolbox_id": toolbox_id,
                        "profile_id": profile_id,
                        "engine_id": engine_id,
                        "issue": "registration_toolbox_mismatch",
                    })
                expected_names = {
                    name
                    for name, route in routes.items()
                    if str(dict(route or {}).get("profile_id") or "") == profile_id
                }
                allowed = self._registration_allowed_tool_names(registration)
                if expected_names != allowed:
                    issues.append({
                        "toolbox_id": toolbox_id,
                        "profile_id": profile_id,
                        "engine_id": engine_id,
                        "issue": "active_inventory_mismatch",
                    })
            for tool_name, raw_route in routes.items():
                route = dict(raw_route or {})
                profile = dict(profiles.get(str(route.get("profile_id") or "")) or {})
                if not profile or str(profile.get("engine_id") or "") != str(route.get("engine_id") or ""):
                    issues.append({
                        "toolbox_id": toolbox_id,
                        "tool_name": tool_name,
                        "issue": "route_profile_mismatch",
                    })
        for engine_id, registration in registrations.items():
            if str(registration.get("routing_state") or "") == "active" and engine_id not in routed_engine_ids:
                issues.append({
                    "toolbox_id": self._registration_toolbox_id(registration) or None,
                    "engine_id": engine_id,
                    "issue": "orphan_active_registration",
                })
        return {
            "status": "ok",
            "contract": "hosting.toolbox.consistency.v2",
            "issue_count": len(issues),
            "consistent": not issues,
            "issues": issues,
            "references": references,
            "summary": {
                "toolbox_count": len(dict(references.get("toolboxes") or {})),
                "issue_count": len(issues),
            },
        }

    def toolbox_review_snapshot(
        self, *, toolbox_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        selected = {
            str(item or "").strip()
            for item in list(toolbox_ids or [])
            if str(item or "").strip()
        }
        consistency = self.toolbox_consistency()
        references = dict(consistency["references"])
        all_toolboxes = dict(references.get("toolboxes") or {})
        if not selected:
            selected = set(all_toolboxes)
        issues = [
            dict(item or {})
            for item in list(consistency.get("issues") or [])
            if str(dict(item or {}).get("toolbox_id") or "") in selected
        ]
        toolboxes: Dict[str, Any] = {}
        for toolbox_id in sorted(selected):
            snapshot = dict(all_toolboxes.get(toolbox_id) or {})
            if not snapshot:
                continue
            toolbox_issues = [item for item in issues if item.get("toolbox_id") == toolbox_id]
            toolboxes[toolbox_id] = {
                "active_revision": snapshot.get("active_revision"),
                "profile_count": len(dict(snapshot.get("profiles") or {})),
                "profiles": [
                    {"profile_id": profile_id, **dict(profile or {})}
                    for profile_id, profile in sorted(dict(snapshot.get("profiles") or {}).items())
                ],
                "tool_names": sorted(dict(snapshot.get("tool_routes") or {})),
                "issue_count": len(toolbox_issues),
                "issue_names": sorted({str(item.get("issue") or "") for item in toolbox_issues}),
            }
        return {
            "status": "ok",
            "contract": "hosting.toolbox.review.v2",
            "toolbox_ids": sorted(selected),
            "toolboxes": toolboxes,
            "issues": issues,
            "recommended_action": "reconcile" if issues else "observe",
            "summary": {
                "toolbox_count": len(toolboxes),
                "issue_count": len(issues),
                **{
                    key: int(dict(references.get("summary") or {}).get(key) or 0)
                    for key in ("stale_engine_count", "stale_bundle_count", "stale_environment_count")
                },
            },
        }

    def _toolbox_repair_now(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
    ) -> Dict[str, Any]:
        self.recover_toolbox_definition_rollouts()
        snapshots = self._toolbox_v2_snapshots()
        selected = {
            str(item or "").strip()
            for item in list(toolbox_ids or snapshots)
            if str(item or "").strip()
        }
        before = self.toolbox_consistency()
        inconsistent = {
            str(item.get("toolbox_id") or "")
            for item in list(before.get("issues") or [])
            if str(item.get("toolbox_id") or "")
        }
        if only_inconsistent:
            selected &= inconsistent
        registrations = self._toolbox_v2_registrations()
        desired: Dict[str, str] = {}
        unresolved: List[Dict[str, Any]] = []
        from .toolbox_runtime_identity import runtime_binding_digest
        for toolbox_id in sorted(selected):
            for profile_id, raw_profile in dict(dict(snapshots.get(toolbox_id) or {}).get("profiles") or {}).items():
                profile_row = dict(raw_profile or {})
                engine_id = str(profile_row.get("engine_id") or "").strip()
                registration = dict(registrations.get(engine_id) or {})
                expected_digest = str(profile_row.get("runtime_binding_digest") or "").strip()
                actual_digest = str(registration.get("runtime_binding_digest") or "").strip()
                if registration and not actual_digest:
                    bundle = dict(registration.get("bundle") or {})
                    environment = dict(registration.get("environment") or {})
                    actual_digest = runtime_binding_digest(
                        toolbox_id=toolbox_id,
                        profile_id=profile_id,
                        manifest_hash=str(bundle.get("manifest_hash") or profile_row.get("manifest_hash") or ""),
                        environment_reference=str(
                            environment.get("environment_reference")
                            or profile_row.get("environment_reference")
                            or ""
                        ),
                        engine_id=engine_id,
                        definition_revision=str(dict(snapshots.get(toolbox_id) or {}).get("active_revision") or ""),
                    )
                if engine_id in registrations and (not expected_digest or actual_digest == expected_digest):
                    if str(registrations[engine_id].get("routing_state") or "") != "active":
                        desired[engine_id] = "active"
                else:
                    unresolved.append({
                        "toolbox_id": toolbox_id,
                        "profile_id": profile_id,
                        "issue": (
                            "runtime_binding_mismatch"
                            if registration
                            else "definition_reapply_required"
                        ),
                    })
        if desired:
            self.set_toolbox_registration_routing_states(desired)
        protected_candidates = self._active_candidate_engine_ids()
        retired: List[str] = []
        for engine_id, registration in registrations.items():
            toolbox_id = self._registration_toolbox_id(registration)
            state = str(registration.get("routing_state") or "")
            if state == "candidate" and engine_id not in protected_candidates:
                self._retire_toolbox_registration(engine_id)
                retired.append(engine_id)
            elif state == "active" and toolbox_id in selected and engine_id not in desired and all(
                str(dict(profile or {}).get("engine_id") or "") != engine_id
                for profile in dict(dict(snapshots.get(toolbox_id) or {}).get("profiles") or {}).values()
            ):
                self.set_toolbox_registration_routing_states({engine_id: "retired"})
        after = self.toolbox_consistency()
        result: Dict[str, Any] = {
            "status": "ok",
            "contract": "hosting.toolbox.repair.v2",
            "repaired_toolbox_ids": sorted(selected - {item["toolbox_id"] for item in unresolved}),
            "reapply_required": unresolved,
            "reactivated_engine_ids": sorted(desired),
            "removed_candidate_engine_ids": sorted(retired),
            "before_issue_count": int(before.get("issue_count") or 0),
            "after_issue_count": int(after.get("issue_count") or 0),
        }
        if details:
            result["before"] = before
            result["after"] = after
        return result

    def _toolbox_gc_now(self) -> Dict[str, Any]:
        recovery = self.recover_toolbox_definition_rollouts()
        references = self._toolbox_reference_report()
        registrations = self._toolbox_v2_registrations()
        protected_candidates = self._active_candidate_engine_ids()
        active_engine_ids = {
            str(profile.get("engine_id") or "")
            for snapshot in self._toolbox_v2_snapshots().values()
            for profile in dict(snapshot.get("profiles") or {}).values()
        }
        removed_engines: List[str] = []
        for engine_id, registration in registrations.items():
            state = str(registration.get("routing_state") or "")
            if engine_id in active_engine_ids or (state == "candidate" and engine_id in protected_candidates):
                continue
            environment_key = self._toolbox_registration_environment_key(registration)
            resources = self._toolbox_runtime_base().resources(environment_key) if environment_key else {}
            if int(dict(resources.get("metrics") or {}).get("active_calls") or 0) > 0:
                continue
            self._retire_toolbox_registration(engine_id)
            removed_engines.append(engine_id)
        removed_bundles: List[str] = []
        for raw in list(references.get("stale_bundle_roots") or []):
            path = Path(str(raw)).expanduser().resolve()
            bundles_root = (self.hosting_root / "toolbox_bundles").resolve()
            if bundles_root in path.parents and path.is_dir():
                shutil.rmtree(path)
                removed_bundles.append(str(path))
        removed_environments: List[str] = []
        builder = getattr(self, "_hermetic_toolbox_environment_builder", None)
        configuration = getattr(self, "_toolbox_host_project_config", None)
        if builder is not None:
            retention = configuration.retention if configuration is not None else None
            active_references = {
                str(item or "")
                for snapshot in self._toolbox_v2_snapshots().values()
                for item in list(snapshot.get("environment_references") or [])
            }
            index = builder._read_references_unlocked()  # noqa: SLF001 - same service boundary
            for environment_key, refs in list(dict(index.get("environments") or {}).items()):
                for reference_id in list(dict(refs or {})):
                    if reference_id not in active_references:
                        builder.release_reference(
                            environment_key=environment_key,
                            reference_id=reference_id,
                        )
            removed_environments = list(
                builder.garbage_collect(
                    protected_environment_keys=(retention.protected_digests if retention else ()),
                    maximum_cache_bytes=(retention.maximum_cache_bytes if retention else None),
                    maximum_cache_artifacts=(retention.maximum_cache_artifacts if retention else None),
                )
            )
        return {
            "status": "ok",
            "contract": "hosting.toolbox.gc.v2",
            "recovery": recovery,
            "removed_engine_ids": sorted(removed_engines),
            "removed_bundle_roots": sorted(removed_bundles),
            "removed_environment_keys": sorted(removed_environments),
            "summary": {
                "removed_engine_count": len(removed_engines),
                "removed_bundle_count": len(removed_bundles),
                "removed_environment_count": len(removed_environments),
            },
            "retention": (
                {
                    "config_revision": configuration.config_revision,
                    "source_set_revision": configuration.source_set_revision,
                    **configuration.retention.to_dict(),
                }
                if configuration is not None
                else None
            ),
        }

    def _toolbox_reconcile_now(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
    ) -> Dict[str, Any]:
        before = self.toolbox_consistency()
        repair = self._toolbox_repair_now(
            toolbox_ids=toolbox_ids,
            only_inconsistent=only_inconsistent,
            details=details,
        )
        gc = self._toolbox_gc_now()
        after = self.toolbox_consistency()
        result: Dict[str, Any] = {
            "status": "ok",
            "contract": "hosting.toolbox.reconcile.v2",
            "repair": repair,
            "gc": gc,
            "summary": {
                "before_issue_count": int(before.get("issue_count") or 0),
                "after_issue_count": int(after.get("issue_count") or 0),
            },
        }
        if details:
            result["before"] = before
            result["after"] = after
        return result

    def _toolbox_maintenance_start(
        self,
        *,
        action: str,
        request_id: str,
        toolbox_ids: Optional[List[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        maintenance_action = str(action or "").strip()
        if maintenance_action not in {"gc", "repair", "reconcile"}:
            raise ValueError("toolbox_maintenance_action_invalid")
        rid = str(request_id or "").strip()
        if not rid:
            raise ValueError("toolbox_maintenance_request_id_required")
        selected = sorted(
            {
                str(item or "").strip()
                for item in list(toolbox_ids or [])
                if str(item or "").strip()
            }
        )
        configuration = getattr(self, "_toolbox_host_project_config", None)
        fingerprint = hosted_execution_fingerprint(
            {
                "execution_kind": HostedExecutionKind.TOOLBOX_MAINTENANCE.value,
                "configuration_revision": self.hosting_configuration_revision,
                "action": maintenance_action,
                "toolbox_ids": selected,
                "only_inconsistent": bool(only_inconsistent),
                "details": bool(details),
                "config_revision": (
                    configuration.config_revision if configuration is not None else None
                ),
                "source_set_revision": (
                    configuration.source_set_revision if configuration is not None else None
                ),
            }
        )
        owner = self._operation_owner(owner_actor_id)
        prepared = self._hosted_operations.prepare(
            owner_actor_id=owner,
            execution_kind=HostedExecutionKind.TOOLBOX_MAINTENANCE,
            selector=HostedOperationSelector(kind="host_scope", id="toolbox-host"),
            namespace="toolbox_maintenance:toolbox-host",
            request_id=rid,
            fingerprint=fingerprint,
            metadata={
                "configuration_revision": self.hosting_configuration_revision,
                "action": maintenance_action,
                "toolbox_ids": selected,
                "only_inconsistent": bool(only_inconsistent),
                "details": bool(details),
            },
        )
        status = prepared.get("status")
        if status is None:
            raise RuntimeError("hosted_operation_capacity")
        dispatch = prepared["action"] == "dispatch"
        if (
            not dispatch
            and status.get("lifecycle")
            == HostedOperationLifecycle.INTERRUPTED_AFTER_DISPATCH_UNKNOWN.value
        ):
            operation_id = str(status["operation"]["operation_id"])
            resumed = self._hosted_operations.requeue_interrupted_after_dispatch(
                operation_id=operation_id
            )
            if resumed is not None:
                status = resumed
                dispatch = True
        if not dispatch:
            return dict(status)
        operation_id = str(status["operation"]["operation_id"])
        thread = threading.Thread(
            target=self._run_toolbox_maintenance,
            kwargs={
                "operation_id": operation_id,
                "action": maintenance_action,
                "toolbox_ids": selected,
                "only_inconsistent": bool(only_inconsistent),
                "details": bool(details),
            },
            name=f"toolbox-maintenance-{maintenance_action}-{operation_id[-8:]}",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "status": "error",
                    "code": "toolbox_maintenance_dispatch_failed",
                    "summary": "The toolbox maintenance worker could not be started.",
                },
                reason="toolbox_maintenance_dispatch_failed",
            )
            raise
        return dict(status)

    def toolbox_repair(
        self,
        *,
        request_id: str,
        toolbox_ids: Optional[List[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
        apply: bool = False,
        mutation_authorized: bool = False,
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        if not apply:
            snapshot = self.toolbox_review_snapshot(toolbox_ids=toolbox_ids)
            return {**snapshot, "recommended_action": "repair" if snapshot["summary"]["issue_count"] else "observe", "mutation_applied": False}
        if not mutation_authorized:
            raise PermissionError("toolbox_repair_mutation_not_authorized")
        return self._toolbox_maintenance_start(
            action="repair",
            request_id=request_id,
            toolbox_ids=toolbox_ids,
            only_inconsistent=only_inconsistent,
            details=details,
            owner_actor_id=owner_actor_id,
        )

    def toolbox_gc(
        self, *, request_id: str, owner_actor_id: str = "service:local"
    ) -> Dict[str, Any]:
        return self._toolbox_maintenance_start(
            action="gc", request_id=request_id, owner_actor_id=owner_actor_id
        )

    def toolbox_reconcile(
        self,
        *,
        request_id: str,
        toolbox_ids: Optional[List[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        return self._toolbox_maintenance_start(
            action="reconcile",
            request_id=request_id,
            toolbox_ids=toolbox_ids,
            only_inconsistent=only_inconsistent,
            details=details,
            owner_actor_id=owner_actor_id,
        )

    def _run_toolbox_maintenance(
        self,
        *,
        operation_id: str,
        action: str,
        toolbox_ids: List[str],
        only_inconsistent: bool,
        details: bool,
    ) -> None:
        self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)

        def checkpoint(
            phase: str,
            code: str,
            completed_units: int | None,
            total_units: int | None,
            summary: str,
            cancellable: bool,
        ) -> None:
            self._hosted_operations.update_progress(
                operation_id=operation_id,
                progress=HostedOperationProgress(
                    phase=phase,
                    code=code,
                    completed_units=completed_units,
                    total_units=total_units,
                    updated_at_ms=int(time.time() * 1000),
                    summary=summary,
                    cancellable=cancellable,
                ),
            )

        try:
            checkpoint(
                "validation", "toolbox_maintenance_validated", 1, 1,
                "The maintenance action and bounded selection were validated.", True,
            )
            checkpoint(
                "recovery", "toolbox_maintenance_recovering", 0, 1,
                "Interrupted toolbox rollout state is being reconciled before maintenance.", False,
            )
            self.recover_toolbox_definition_rollouts()
            checkpoint(
                "recovery", "toolbox_maintenance_recovered", 1, 1,
                "Interrupted toolbox rollout state was reconciled.", False,
            )
            if action == "repair":
                checkpoint("repair", "toolbox_repair_started", 0, 1, "Selected toolbox state is being repaired.", False)
                result = self._toolbox_repair_now(
                    toolbox_ids=toolbox_ids,
                    only_inconsistent=only_inconsistent,
                    details=details,
                )
                checkpoint("repair", "toolbox_repair_completed", 1, 1, "Selected toolbox state was repaired.", False)
            elif action == "gc":
                checkpoint("gc", "toolbox_gc_started", 0, 1, "Unreferenced toolbox resources are being reclaimed.", False)
                result = self._toolbox_gc_now()
                checkpoint("gc", "toolbox_gc_completed", 1, 1, "Unreferenced toolbox resources were reclaimed.", False)
            else:
                before = self.toolbox_consistency()
                checkpoint("repair", "toolbox_reconcile_repair_started", 0, 1, "Selected toolbox state is being repaired.", False)
                repair = self._toolbox_repair_now(
                    toolbox_ids=toolbox_ids,
                    only_inconsistent=only_inconsistent,
                    details=details,
                )
                checkpoint("repair", "toolbox_reconcile_repair_completed", 1, 1, "Selected toolbox state was repaired.", False)
                checkpoint("gc", "toolbox_reconcile_gc_started", 0, 1, "Unreferenced toolbox resources are being reclaimed.", False)
                gc = self._toolbox_gc_now()
                checkpoint("gc", "toolbox_reconcile_gc_completed", 1, 1, "Unreferenced toolbox resources were reclaimed.", False)
                after = self.toolbox_consistency()
                result = {
                    "status": "ok",
                    "contract": "hosting.toolbox.reconcile.v2",
                    "repair": repair,
                    "gc": gc,
                    "summary": {
                        "before_issue_count": int(before.get("issue_count") or 0),
                        "after_issue_count": int(after.get("issue_count") or 0),
                    },
                }
                if details:
                    result["before"] = before
                    result["after"] = after
            checkpoint("cleanup", "toolbox_maintenance_cleanup_completed", 1, 1, "Maintenance cleanup is complete.", False)
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                envelope={
                    "contract": "hosting.toolbox.maintenance_result.v1",
                    "status": "ok",
                    "code": "toolbox_maintenance_completed",
                    "action": action,
                    "maintenance_result": result,
                },
            )
        except Exception:
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "status": "error",
                    "code": "toolbox_maintenance_failed",
                    "summary": "Toolbox maintenance failed before a complete terminal result.",
                },
                reason="toolbox_maintenance_failed",
            )
