"""Route-based toolbox consistency, recovery, repair, references, and GC."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


class ToolboxMaintenanceMixin:
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

    def toolbox_repair(
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
        for toolbox_id in sorted(selected):
            for profile_id, raw_profile in dict(dict(snapshots.get(toolbox_id) or {}).get("profiles") or {}).items():
                engine_id = str(dict(raw_profile or {}).get("engine_id") or "").strip()
                if engine_id in registrations:
                    if str(registrations[engine_id].get("routing_state") or "") != "active":
                        desired[engine_id] = "active"
                else:
                    unresolved.append({
                        "toolbox_id": toolbox_id,
                        "profile_id": profile_id,
                        "issue": "definition_reapply_required",
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

    def toolbox_gc(self) -> Dict[str, Any]:
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
        if builder is not None:
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
            removed_environments = list(builder.garbage_collect())
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
        }

    def toolbox_reconcile(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
    ) -> Dict[str, Any]:
        before = self.toolbox_consistency()
        repair = self.toolbox_repair(
            toolbox_ids=toolbox_ids,
            only_inconsistent=only_inconsistent,
            details=details,
        )
        gc = self.toolbox_gc()
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
