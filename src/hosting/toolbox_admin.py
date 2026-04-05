from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HostedToolboxAdmin:
    host: Any
    default_toolbox_ids: List[str] = field(default_factory=list)

    def _normalize_toolbox_ids(self, toolbox_ids: Optional[List[str]] = None) -> Optional[List[str]]:
        values = [
            str(item or "").strip()
            for item in list(toolbox_ids if toolbox_ids is not None else self.default_toolbox_ids)
            if str(item or "").strip()
        ]
        return values or None

    def references(self) -> Dict[str, Any]:
        return dict(self.host.toolbox_references() or {})

    def consistency(self) -> Dict[str, Any]:
        return dict(self.host.toolbox_consistency() or {})

    def review_snapshot(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        scoped_ids = set(self._normalize_toolbox_ids(toolbox_ids) or [])
        if hasattr(self.host, "toolbox_review_snapshot"):
            return dict(
                self.host.toolbox_review_snapshot(
                    toolbox_ids=sorted(scoped_ids),
                )
                or {}
            )
        references = dict(self.references() or {})
        consistency = dict(self.consistency() or {})
        if scoped_ids:
            filtered_issues = [
                dict(item or {})
                for item in list(consistency.get("issues") or [])
                if str(dict(item or {}).get("toolbox_id") or "").strip() in scoped_ids
            ]
            filtered_toolboxes = {
                str(k): dict(v or {})
                for k, v in dict(references.get("toolboxes") or {}).items()
                if str(k or "").strip() in scoped_ids
            }
            consistency = {
                **consistency,
                "issue_count": len(filtered_issues),
                "issues": filtered_issues,
            }
            references = {
                **references,
                "toolboxes": filtered_toolboxes,
                "summary": {
                    **dict(references.get("summary") or {}),
                    "toolbox_count": len(filtered_toolboxes),
                },
            }
        issues = [dict(item or {}) for item in list(consistency.get("issues") or [])]
        issues_by_toolbox: Dict[str, List[Dict[str, Any]]] = {}
        for item in issues:
            toolbox_id = str(item.get("toolbox_id") or "").strip()
            issues_by_toolbox.setdefault(toolbox_id, []).append(item)
        toolbox_rows: Dict[str, Dict[str, Any]] = {}
        for toolbox_id, raw_toolbox in dict(references.get("toolboxes") or {}).items():
            toolbox_row = dict(raw_toolbox or {})
            profile_rows = []
            for profile_id, raw_profile in dict(toolbox_row.get("profiles") or {}).items():
                profile_row = dict(raw_profile or {})
                rollout = dict(profile_row.get("rollout") or {})
                environment = dict(profile_row.get("environment") or {})
                profile_rows.append(
                    {
                        "sandbox_profile_id": str(dict(profile_row.get("sandbox_profile") or {}).get("profile_id") or profile_id or "").strip(),
                        "environment_name": str(environment.get("environment_name") or "").strip() or None,
                        "all_registered_tool_names": [
                            str(item or "").strip()
                            for item in list(rollout.get("all_registered_tool_names") or [])
                            if str(item or "").strip()
                        ],
                        "engine_id": str(profile_row.get("engine_id") or "").strip() or None,
                        "ready": bool(rollout.get("ready", False)),
                        "warmup_ms": int(rollout.get("warmup_ms") or 0),
                    }
                )
            toolbox_issues = list(issues_by_toolbox.get(str(toolbox_id or "").strip()) or [])
            toolbox_rows[str(toolbox_id)] = {
                "profile_count": len(profile_rows),
                "profiles": profile_rows,
                "issue_count": len(toolbox_issues),
                "issue_names": sorted(
                    {
                        str(item.get("issue") or "").strip()
                        for item in toolbox_issues
                        if str(item.get("issue") or "").strip()
                    }
                ),
                "live_registration_count": sum(
                    1 for item in profile_rows if str(item.get("engine_id") or "").strip()
                ),
            }
        issue_count = len(issues)
        recommended_action = "reconcile" if issue_count > 0 else "observe"
        return {
            "status": "ok",
            "toolbox_ids": sorted(scoped_ids),
            "toolboxes": toolbox_rows,
            "issues": issues,
            "recommended_action": recommended_action,
            "references_summary": dict(references.get("summary") or {}),
            "consistency_summary": {
                **dict(consistency.get("summary") or {}),
                "issue_count": issue_count,
            },
            "summary": {
                "toolbox_count": int(dict(dict(references or {}).get("summary") or {}).get("toolbox_count") or 0),
                "issue_count": issue_count,
                "stale_engine_count": int(dict(dict(references or {}).get("summary") or {}).get("stale_engine_count") or 0),
                "stale_bundle_count": int(dict(dict(references or {}).get("summary") or {}).get("stale_bundle_count") or 0),
                "stale_environment_count": int(dict(dict(references or {}).get("summary") or {}).get("stale_environment_count") or 0),
            },
        }

    def startup_reconcile(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
        only_inconsistent: bool = True,
        details: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.host.toolbox_reconcile(
                toolbox_ids=self._normalize_toolbox_ids(toolbox_ids),
                only_inconsistent=bool(only_inconsistent),
                details=bool(details),
            )
            or {}
        )

    def periodic_consistency_check(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        consistency = dict(self.consistency() or {})
        scoped_ids = set(self._normalize_toolbox_ids(toolbox_ids) or [])
        if not scoped_ids:
            return {
                "status": "ok",
                "issue_count": int(consistency.get("issue_count") or 0),
                "toolbox_ids": [],
                "consistency": consistency,
            }
        issues = [
            dict(item or {})
            for item in list(consistency.get("issues") or [])
            if str(dict(item or {}).get("toolbox_id") or "").strip() in scoped_ids
        ]
        return {
            "status": "ok",
            "issue_count": len(issues),
            "toolbox_ids": sorted(scoped_ids),
            "consistency": {
                **consistency,
                "issue_count": len(issues),
                "issues": issues,
            },
        }

    def auto_repair_if_needed(
        self,
        *,
        toolbox_ids: Optional[List[str]] = None,
        gc_on_noop: bool = False,
        details: bool = False,
    ) -> Dict[str, Any]:
        scoped_ids = self._normalize_toolbox_ids(toolbox_ids)
        check = self.periodic_consistency_check(toolbox_ids=scoped_ids)
        issue_count = int(check.get("issue_count") or 0)
        if issue_count <= 0:
            gc_out: Dict[str, Any] = {}
            if gc_on_noop:
                gc_out = dict(self.host.toolbox_gc() or {})
            return {
                "status": "ok",
                "action": "noop",
                "toolbox_ids": list(scoped_ids or []),
                "issue_count": 0,
                "consistency": dict(check.get("consistency") or {}),
                "gc": gc_out,
            }
        reconcile = dict(
            self.host.toolbox_reconcile(
                toolbox_ids=scoped_ids,
                only_inconsistent=True,
                details=bool(details),
            )
            or {}
        )
        return {
            "status": "ok",
            "action": "reconcile",
            "toolbox_ids": list(scoped_ids or []),
            "issue_count": issue_count,
            "consistency": dict(check.get("consistency") or {}),
            "reconcile": reconcile,
        }
