"""Toolbox runtime routing, execution, and registration orchestration."""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mp13_engine.mp13_toolbox import ToolsView

from ..callable_surface import HOST_CAPABILITY_APPROVAL_CALLBACK_NAME, HOST_CAPABILITY_DISPATCH_CALLBACK_NAME, host_capability_approval_request
from ..operation_contract import (
    HostedExecutionKind,
    HostedOperationLifecycle,
    HostedOperationProgress,
    HostedOperationSelector,
    hosted_execution_fingerprint,
)
from ..sandbox.host_capabilities import HostCapabilityBroker
from ..sandbox.service_broker_registry import service_broker_host_capability_session
from ..toolbox.callbacks import _HostedToolCallbackRelay
from .errors import ToolboxRolloutError


class ToolboxRuntimeMixin:
    def _toolbox_definition_planning_context(self) -> Dict[str, Any]:
        from ..toolbox.catalog import ToolboxEnvironmentTemplateSpec, normalize_distribution_name
        from ..toolbox.dependency_policy import ToolboxDependencyPolicy
        from ..toolbox.identity import identity_digest
        from ..toolbox.target import detect_current_toolbox_target

        catalog = self._toolbox_template_catalog.read()
        active_templates = []
        for template_id, template_digest in sorted(dict(catalog.get("active") or {}).items()):
            entry = next(
                (
                    item for item in list(catalog.get("entries") or [])
                    if item["template_id"] == template_id
                    and item["template_digest"] == template_digest
                    and item["lifecycle"] == "active"
                ),
                None,
            )
            if entry is not None:
                active_templates.append(ToolboxEnvironmentTemplateSpec.from_dict(entry["template"]))
        if not active_templates:
            raise ValueError("toolbox_builtins_not_ready")
        templates = tuple(active_templates)
        catalog_revision = str(catalog["catalog_revision"])
        current_target = detect_current_toolbox_target()
        python_abi = self._toolbox_required_python_abi or current_target.python_abi
        platform = self._toolbox_required_platform or current_target.platform
        configured = getattr(self, "_configured_toolbox_dependency_policy", None)
        if configured is None:
            allowed_packages = sorted(
                {
                    normalize_distribution_name(distribution.name)
                    for template in templates
                    for distribution in template.locked_distributions
                }
            )
            policy_payload = {
                "allowed_template_ids": sorted(template.template_id for template in templates),
                "allowed_targets": [f"{python_abi}-{platform}"],
                "package_allowlist": allowed_packages,
                "package_denylist": [],
                "allow_custom": False,
                "custom_requires_approval": True,
                "online_resolution_allowed": False,
                "allowed_index_origins": [],
            }
            configured = ToolboxDependencyPolicy(
                revision=identity_digest("hosting.toolbox.package_policy.v1", policy_payload),
                **policy_payload,
            )
        return {
            "templates": templates,
            "catalog": catalog,
            "catalog_revision": catalog_revision,
            "policy": configured,
            "python_abi": python_abi,
            "platform": platform,
            "target": current_target.name,
            "configuration": getattr(self, "_toolbox_host_project_config", None),
            "runtime_identity": {
                "version": ".".join(str(item) for item in sys.version_info[:3]),
                "artifact_digest": templates[0].parent_worker_artifact_digest,
            },
        }

    @staticmethod
    def _toolbox_custom_delta_digest(draft: Any) -> str:
        from ..toolbox.identity import identity_digest

        return identity_digest(
            "hosting.toolbox.custom_delta_set.v1",
            sorted(
                item.custom_resolved_lock_digest
                for item in draft.profiles
                if item.custom_resolved_lock_digest is not None
            ),
        )

    @staticmethod
    def _validate_definition_policy(draft: Any, context: Dict[str, Any]) -> None:
        from packaging.requirements import Requirement
        from ..toolbox.catalog import normalize_distribution_name
        from ..toolbox.dependency_policy import ToolboxDependencyPolicyError

        policy = context["policy"]
        target = f"{context['python_abi']}-{context['platform']}"
        if target not in policy.allowed_targets:
            raise ToolboxDependencyPolicyError("target_denied", "The runtime target is not allowed.")
        for profile in draft.profiles:
            if profile.template_id not in policy.allowed_template_ids:
                raise ToolboxDependencyPolicyError("template_denied", "The selected template is not allowed.")
        requirements = [
            requirement
            for request in (*draft.definition.auto_requests, *draft.definition.manual_requests)
            for requirement in request.dependency.package_requirements
        ]
        distributions = {normalize_distribution_name(Requirement(item).name) for item in requirements}
        if distributions & set(policy.package_denylist):
            raise ToolboxDependencyPolicyError("package_denied", "A requested package is denied.")
        if policy.package_allowlist and not distributions <= set(policy.package_allowlist):
            raise ToolboxDependencyPolicyError("package_not_allowlisted", "A requested package is not allowlisted.")
        if draft.custom_environment_count and not policy.allow_custom:
            raise ToolboxDependencyPolicyError("custom_environment_denied", "Custom environments are disabled.")

    def toolbox_get_definition(
        self,
        *,
        toolbox_id: str,
        operator_details: bool = False,
        owner_actor_id: str = "service:local",
        authority_id: str = "authority:local",
    ) -> Dict[str, Any]:
        from ..toolbox.bundle_models import ToolboxDefinitionSpec

        tid = str(toolbox_id or "").strip()
        if not tid:
            raise ValueError("toolbox_id_required")
        snapshot = self._toolbox_state_v2.get(tid)
        if snapshot is None:
            definition = ToolboxDefinitionSpec.from_dict(
                {
                    "contract": "hosting.toolbox.definition",
                    "toolbox_id": tid,
                    "expected_revision": None,
                    "auto_requests": [],
                    "manual_requests": [],
                    "intrinsics": {"names": [], "include_guides": False, "sandbox_policy": {}},
                }
            ).to_dict()
            active_revision = None
            routes: Dict[str, Any] = {}
            history: List[Dict[str, Any]] = []
        else:
            definition = dict(snapshot["definition"])
            definition["expected_revision"] = None
            active_revision = snapshot["active_revision"]
            routes = dict(snapshot["tool_routes"])
            history = [
                {
                    "revision": item["revision"],
                    "published_at_ms": item["published_at_ms"],
                    "profile_count": item["profile_count"],
                    "tool_count": item["tool_count"],
                }
                for item in list(snapshot["rollout_history"])[-32:]
            ]
        return {
            "contract": "hosting.toolbox.definition_snapshot",
            "tool_runtime_id": "runtime-local",
            "toolbox_id": tid,
            "active_revision": active_revision,
            "definition": definition,
            "active_tools": sorted(routes),
            "rollout": history,
            "diagnostics": [],
            "user_projection": {
                "state": "ready",
                "code": "toolbox_definition_active",
                "summary": "The active toolbox definition is available.",
            },
        }

    def toolbox_plan_definition(
        self,
        *,
        definition: Dict[str, Any],
        request_id: str,
        operator_details: bool = False,
        owner_actor_id: str = "service:local",
        authority_id: str = "authority:local",
        ttl_ms: int = 15 * 60 * 1000,
    ) -> Dict[str, Any]:
        from ..toolbox.bundle_models import ToolboxDefinitionSpec

        model = ToolboxDefinitionSpec.from_dict(definition)
        actor = str(owner_actor_id or "").strip()
        rid = str(request_id or "").strip()
        fingerprint = hosted_execution_fingerprint(
            {
                "definition": model.to_dict(),
                "configuration_revision": self.hosting_configuration_revision,
                "operator_details": bool(operator_details),
                "ttl_ms": int(ttl_ms),
                "authority_id": str(authority_id or "").strip(),
            }
        )
        prepared = self._hosted_operations.prepare(
            owner_actor_id=actor,
            execution_kind=HostedExecutionKind.TOOLBOX_DEFINITION_PLAN,
            selector={"kind": "toolbox_id", "id": model.toolbox_id},
            namespace=f"toolbox-definition-plan:{model.toolbox_id}",
            request_id=rid,
            fingerprint=fingerprint,
            metadata={
                "toolbox_id": model.toolbox_id,
                "configuration_revision": self.hosting_configuration_revision,
                "retain_terminal_result": True,
            },
        )
        status = dict(prepared.get("status") or {})
        if prepared.get("action") != "dispatch":
            return status
        operation_id = str(dict(status["operation"])["operation_id"])
        worker = threading.Thread(
            target=self._run_toolbox_definition_plan,
            kwargs={
                "operation_id": operation_id,
                "definition": model.to_dict(),
                "operator_details": bool(operator_details),
                "owner_actor_id": actor,
                "authority_id": str(authority_id or "").strip(),
                "ttl_ms": int(ttl_ms),
            },
            name=f"toolbox-definition-plan-{operation_id[:12]}",
            daemon=True,
        )
        worker.start()
        return status

    def _run_toolbox_definition_plan(self, *, operation_id: str, **kwargs: Any) -> Dict[str, Any]:
        self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
        try:
            result = self._build_toolbox_definition_plan(**kwargs)
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                envelope=result,
            )
        except Exception as exc:
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "contract": "hosting.toolbox.definition_plan_failure.v1",
                    "status": "failed",
                    "code": "toolbox_definition_plan_failed",
                },
            )

    def _plan_generic_toolbox_environments(
        self,
        *,
        definition: Any,
        environment_mutations: Any,
        platform: str,
        consumer_revision: int,
    ) -> tuple[Any, ...]:
        from ..environments.contracts import EnvironmentRequest
        from ..packages.contracts import PackageLock
        from ..toolbox.identity import identity_digest
        from .toolbox_plans import ToolboxPlannedEnvironmentRecord

        planned = []
        for environment in environment_mutations:
            if all(item.change == "removed" for item in environment.tool_mutations):
                continue
            for alternative in environment.alternatives:
                generic_artifacts = []
                generic_dependencies = []
                import_request_id = identity_digest(
                    "hosting.toolbox.plan_package_import.v1",
                    {
                        "toolbox_id": definition.toolbox_id,
                        "definition_revision": definition.revision,
                        "environment_id": environment.environment_id,
                        "alternative_id": alternative.alternative_id,
                        "configuration_revision": self.hosting_configuration_revision,
                    },
                )
                for artifact in alternative.artifacts:
                    generic_artifacts.append(self._package_manager.import_verified_file(
                        source_id=artifact.source_id,
                        path=self._toolbox_artifact_store.object_path(
                            artifact.artifact_digest
                        ),
                        expected_digest=artifact.artifact_digest,
                        actor_id="service:toolbox-planner",
                        request_id=import_request_id,
                    ))
                    generic_dependencies.append({
                        "name": artifact.distribution,
                        "version": artifact.version,
                        "artifact_id": artifact.artifact_digest,
                    })
                lock = PackageLock.from_dict(self._package_manager.create_lock(
                    lock_id="tbx-" + alternative.alternative_id.removeprefix("sha256:"),
                    revision=consumer_revision,
                    runtime_kind="python",
                    platform=platform,
                    artifacts=generic_artifacts,
                    dependencies=generic_dependencies,
                ))
                request = EnvironmentRequest.from_dict({
                    "contract": EnvironmentRequest.CONTRACT,
                    "request_id": identity_digest(
                        "hosting.toolbox.environment_request_id.v1",
                        {
                            "toolbox_id": definition.toolbox_id,
                            "definition_revision": definition.revision,
                            "environment_id": environment.environment_id,
                            "alternative_id": alternative.alternative_id,
                            "package_lock_digest": lock.lock_digest,
                        },
                    ),
                    "consumer_kind": "toolbox",
                    "consumer_id": definition.toolbox_id,
                    "revision": consumer_revision,
                    "template_id": environment.base_template_id,
                    "template_revision": 1,
                    "package_lock_digest": lock.lock_digest,
                    "runtime_kind": "python",
                    "platform": platform,
                    "configuration_revision": self.hosting_configuration_revision,
                })
                planned.append(ToolboxPlannedEnvironmentRecord(
                    environment_id=environment.environment_id,
                    alternative_id=alternative.alternative_id,
                    package_lock=lock,
                    environment_request=request,
                ))
        return tuple(planned)

    def _build_toolbox_definition_plan(
        self,
        *,
        definition: Dict[str, Any],
        operator_details: bool = False,
        owner_actor_id: str = "service:local",
        authority_id: str = "authority:local",
        ttl_ms: int = 15 * 60 * 1000,
    ) -> Dict[str, Any]:
        from ..toolbox.bundle_models import ToolboxDefinitionSpec, ToolboxPlanPins
        from ..toolbox.definition_planner import (
            ActiveToolboxProfileSnapshot,
            build_toolbox_environment_mutations,
            plan_toolbox_definition,
        )
        from ..toolbox.identity import identity_digest
        from .toolbox_definition_resolution import ConfiguredToolboxPlanResolver

        model = ToolboxDefinitionSpec.from_dict(definition)
        active = self._toolbox_state_v2.get(model.toolbox_id)
        active_revision = dict(active or {}).get("active_revision")
        if model.expected_revision != active_revision:
            from .toolbox_state_v2 import ToolboxRevisionConflictError

            raise ToolboxRevisionConflictError("toolbox_revision_conflict")
        context = self._toolbox_definition_planning_context()
        configuration = context["configuration"]
        if configuration is None:
            raise ValueError("toolbox_host_project_configuration_required")
        draft = plan_toolbox_definition(
            model,
            templates=context["templates"],
            python_abi=context["python_abi"],
            platform=context["platform"],
            runtime_identity=context["runtime_identity"],
        )
        self._validate_definition_policy(draft, context)
        active_profiles = []
        for profile_id, row in dict(dict(active or {}).get("profiles") or {}).items():
            profile = dict(row["profile"])
            active_profiles.append(
                ActiveToolboxProfileSnapshot(
                    profile_id=profile_id,
                    manifest_hash=row["manifest_hash"],
                    environment_key=profile["environment_key"],
                    sandbox_policy_digest=identity_digest(
                        "hosting.toolbox.sandbox_policy.v1", profile["sandbox_policy"]
                    ),
                    assigned_tool_keys=tuple(profile["assigned_tool_keys"]),
                )
            )
        resolver = ConfiguredToolboxPlanResolver(
            configuration=configuration,
            artifact_store=self._toolbox_artifact_store,
            catalog_state=context["catalog"],
        )
        candidates = resolver.candidates_for_draft(draft)
        active_environments = resolver.active_environments(active)
        approval_required = bool(
            draft.custom_environment_count and context["policy"].custom_requires_approval
        )
        environment_mutations = build_toolbox_environment_mutations(
            active_definition=ToolboxDefinitionSpec.from_dict(
                self.toolbox_get_definition(toolbox_id=model.toolbox_id)["definition"]
            ),
            draft=draft,
            candidates=candidates,
            active_environments=active_environments,
            dependency_approval_required=approval_required,
        )
        # Environment references require an integer revision. Derive a stable,
        # JSON-safe value from the immutable definition identity rather than the
        # bounded rollout history, which intentionally truncates after 32 rows.
        consumer_revision = int(model.revision.removeprefix("sha256:")[:13], 16) + 1
        planned_environments = self._plan_generic_toolbox_environments(
            definition=model,
            environment_mutations=environment_mutations,
            platform=context["platform"],
            consumer_revision=consumer_revision,
        )
        pins = ToolboxPlanPins(
            active_definition_revision=active_revision,
            target=context["target"],
            configuration_revision=self.hosting_configuration_revision,
            catalog_revision=context["catalog_revision"],
            host_config_revision=configuration.config_revision,
            dependency_policy_revision=context["policy"].revision,
            source_set_revision=configuration.source_set_revision,
        )
        now_ms = int(time.time() * 1000)
        record = self._toolbox_definition_plans.create(
            draft,
            active_definition=ToolboxDefinitionSpec.from_dict(
                self.toolbox_get_definition(toolbox_id=model.toolbox_id)["definition"]
            ),
            pins=pins,
            environment_mutations=environment_mutations,
            planned_environments=planned_environments,
            active_profiles=active_profiles,
            now_ms=now_ms,
            ttl_ms=int(ttl_ms),
            owner_actor_id=str(owner_actor_id or "").strip(),
            authority_id=str(authority_id or "").strip(),
        )
        state = "confirmation_required"
        code = "toolbox_definition_confirmation_required"
        return {
            "contract": "hosting.toolbox.definition_plan.v2",
            "plan_id": record.plan_id,
            "toolbox_id": record.toolbox_id,
            "definition_hash": record.proposed_definition.revision,
            "expected_revision": record.proposed_definition.expected_revision,
            "pins": record.pins.to_dict(),
            "expires_at_ms": record.expires_at_ms,
            "can_apply": False,
            "confirmation_required": True,
            "approval_required": approval_required,
            "environment_mutations": [item.to_dict() for item in environment_mutations],
            "profile_diff": {
                classification: sum(
                    item["classification"] == classification for item in record.profile_changes
                )
                for classification in ("reused", "added", "replaced", "removed")
            },
            "diagnostics": [],
            "user_projection": {
                "state": state,
                "code": code,
                "summary": (
                    "Review and confirm the exact package alternatives before apply."
                ),
            },
        }

    def toolbox_confirm_definition_plan(
        self,
        *,
        plan_id: str,
        environment_choices: List[Dict[str, Any]],
        request_id: str,
        owner_actor_id: str = "service:local",
        authority_id: str = "authority:local",
    ) -> Dict[str, Any]:
        from ..toolbox.definition_planner import ToolboxEnvironmentConfirmationChoice

        now_ms = int(time.time() * 1000)
        plan = self._toolbox_definition_plans.get(plan_id, now_ms=now_ms)
        actor = str(owner_actor_id or "").strip()
        authority = str(authority_id or "").strip()
        if plan.owner_actor_id != actor or plan.authority_id != authority:
            raise PermissionError("toolbox_definition_plan_not_found")
        choices = tuple(
            ToolboxEnvironmentConfirmationChoice.from_dict(item)
            for item in environment_choices
        )
        fingerprint = hosted_execution_fingerprint(
            {
                "plan_id": plan.plan_id,
                "choices": [item.to_dict() for item in choices],
                "authority_id": authority,
                "configuration_revision": plan.pins.configuration_revision,
            }
        )
        prepared = self._hosted_operations.prepare(
            owner_actor_id=actor,
            execution_kind=HostedExecutionKind.TOOLBOX_DEFINITION_CONFIRMATION,
            selector={"kind": "toolbox_id", "id": plan.toolbox_id},
            namespace=f"toolbox-definition-confirmation:{plan.toolbox_id}",
            request_id=str(request_id or "").strip(),
            fingerprint=fingerprint,
            metadata={
                "toolbox_id": plan.toolbox_id,
                "plan_id": plan.plan_id,
                "configuration_revision": plan.pins.configuration_revision,
            },
        )
        status = dict(prepared.get("status") or {})
        if prepared.get("action") != "dispatch":
            return status
        operation_id = str(dict(status["operation"])["operation_id"])
        worker = threading.Thread(
            target=self._run_toolbox_definition_confirmation,
            kwargs={
                "operation_id": operation_id,
                "plan_id": plan.plan_id,
                "choices": [item.to_dict() for item in choices],
                "owner_actor_id": actor,
                "authority_id": authority,
            },
            name=f"toolbox-definition-confirm-{operation_id[:12]}",
            daemon=True,
        )
        worker.start()
        return status

    def _pin_confirmed_toolbox_resolutions(
        self,
        *,
        plan: Any,
        draft: Any,
        reduction: Any,
        context: Dict[str, Any],
    ) -> tuple[Any, Dict[str, Any]]:
        from ..toolbox.bundle_models import ResolvedToolboxProfileSpec
        from ..toolbox.catalog import ToolboxEnvironmentTemplateSpec, ToolboxLockedDistributionSpec
        from ..toolbox.definition_planner import ToolboxDefinitionPlanDraft
        from ..toolbox.hermetic_environment import (
            ResolvedToolboxEnvironmentInput,
            ToolboxLockedArtifactSpec,
        )

        selected = {
            item["environment_id"]: item["alternative_id"]
            for item in reduction.selected_alternatives
        }
        catalog = dict(context["catalog"] or {})
        profiles = []
        bundles = []
        resolved_environments: Dict[str, Any] = {}
        for profile, bundle in zip(draft.profiles, draft.bundles, strict=True):
            matching = [
                offer for offer in plan.environment_mutations
                if set(profile.assigned_tool_keys).issubset(
                    {item.tool_key for item in offer.tool_mutations}
                )
            ]
            if len(matching) != 1:
                raise ValueError("toolbox_confirmation_profile_offer_mismatch")
            offer = matching[0]
            alternative = next(
                item for item in offer.alternatives
                if item.alternative_id == selected.get(offer.environment_id)
            )
            entry = next(
                dict(item) for item in list(catalog.get("entries") or [])
                if item.get("template_digest") == offer.base_template_revision
                and item.get("template_id") == offer.base_template_id
                and item.get("lifecycle") == "active"
            )
            template = ToolboxEnvironmentTemplateSpec.from_dict(entry["template"])
            locked_artifacts = []
            for artifact in alternative.artifacts:
                path = self._toolbox_artifact_store.object_path(artifact.artifact_digest)
                if not path.is_file():
                    raise ValueError("toolbox_confirmation_artifact_missing")
                locked_artifacts.append(ToolboxLockedArtifactSpec(
                    distribution_name=artifact.distribution,
                    version=artifact.version,
                    source_id=artifact.source_id,
                    filename=artifact.wheel_filename,
                    sha256=artifact.artifact_digest,
                    size_bytes=path.stat().st_size,
                ))
            custom_lock = (
                alternative.lock_digest
                if profile.custom_resolved_lock_digest is not None else None
            )
            resolved = ResolvedToolboxEnvironmentInput(
                template_id=template.template_id,
                template_digest=offer.base_template_revision,
                runtime_version=str(context["runtime_identity"]["version"]),
                runtime_artifact_digest=template.parent_worker_artifact_digest,
                python_abi=context["python_abi"],
                platform=context["platform"],
                complete_lock_digest=template.lock_digest,
                complete_lock=tuple(sorted(
                    ToolboxLockedDistributionSpec(item.distribution, item.version)
                    for item in alternative.artifacts
                )),
                locked_artifacts=tuple(sorted(locked_artifacts)),
                custom_resolved_lock_digest=custom_lock,
                isolation_policy_version=template.isolation_policy_version,
                resolved_import_roots=profile.resolved_import_roots,
            )
            pinned_profile = ResolvedToolboxProfileSpec(
                environment_key=resolved.environment_key,
                template_id=profile.template_id,
                template_lock_digest=profile.template_lock_digest,
                custom_resolved_lock_digest=custom_lock,
                sandbox_policy=profile.sandbox_policy,
                assigned_tool_keys=profile.assigned_tool_keys,
                resolved_import_roots=profile.resolved_import_roots,
            )
            bundle.resolved_profile = pinned_profile
            bundle.dependency_lock_hash = pinned_profile.effective_lock_digest
            profiles.append(pinned_profile)
            bundles.append(bundle)
            resolved_environments[pinned_profile.profile_id] = resolved.to_dict()
        pinned = ToolboxDefinitionPlanDraft(
            definition=draft.definition,
            profiles=tuple(profiles),
            bundles=tuple(bundles),
            custom_environment_count=sum(
                item.custom_resolved_lock_digest is not None for item in profiles
            ),
        )
        return pinned, resolved_environments

    def _run_toolbox_definition_confirmation(
        self,
        *,
        operation_id: str,
        plan_id: str,
        choices: List[Dict[str, Any]],
        owner_actor_id: str,
        authority_id: str,
    ) -> Dict[str, Any]:
        from ..toolbox.definition_planner import (
            plan_toolbox_definition,
            reduce_toolbox_confirmation,
        )

        self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
        try:
            now_ms = int(time.time() * 1000)
            plan = self._toolbox_definition_plans.get(plan_id, now_ms=now_ms)
            if plan.owner_actor_id != owner_actor_id or plan.authority_id != authority_id:
                raise PermissionError("toolbox_definition_plan_not_found")
            reduction = reduce_toolbox_confirmation(
                active_definition=plan.active_definition,
                proposed_definition=plan.proposed_definition,
                environment_mutations=plan.environment_mutations,
                choices=choices,
            )
            context = self._toolbox_definition_planning_context()
            configuration = context["configuration"]
            if configuration is None or (
                self.hosting_configuration_revision != plan.pins.configuration_revision
                or context["catalog_revision"] != plan.pins.catalog_revision
                or configuration.config_revision != plan.pins.host_config_revision
                or configuration.source_set_revision != plan.pins.source_set_revision
                or context["policy"].revision != plan.pins.dependency_policy_revision
                or context["target"] != plan.pins.target
            ):
                raise ValueError("toolbox_definition_plan_pins_changed")
            confirmed_draft = plan_toolbox_definition(
                reduction.effective_definition,
                templates=context["templates"],
                python_abi=context["python_abi"],
                platform=context["platform"],
                runtime_identity=context["runtime_identity"],
            )
            self._validate_definition_policy(confirmed_draft, context)
            confirmed_draft, resolved_environments = self._pin_confirmed_toolbox_resolutions(
                plan=plan,
                draft=confirmed_draft,
                reduction=reduction,
                context=context,
            )
            confirmation_ref, receipt = self._toolbox_confirmations.create(
                plan_id=plan.plan_id,
                toolbox_id=plan.toolbox_id,
                owner_actor_id=owner_actor_id,
                authority_id=authority_id,
                choices=choices,
                reduction=reduction,
                confirmed_draft=confirmed_draft.to_persisted_dict(),
                resolved_environments=resolved_environments,
                now_ms=now_ms,
                expires_at_ms=plan.expires_at_ms,
            )
            result = {
                "contract": "hosting.toolbox.definition_confirmation_result.v1",
                "status": "confirmed",
                "confirmation_ref": confirmation_ref,
                "plan_id": plan.plan_id,
                "expires_at_ms": receipt.expires_at_ms,
                **reduction.to_dict(),
            }
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                envelope=result,
            )
        except Exception as exc:
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "contract": "hosting.toolbox.definition_confirmation_failure.v1",
                    "status": "failed",
                    "code": "toolbox_definition_confirmation_failed",
                },
            )

    @staticmethod
    def _toolbox_confirmation_approval_binding(plan: Any, receipt: Any) -> tuple[str, str]:
        from ..toolbox.identity import identity_digest

        selected = {
            item["environment_id"]: item["alternative_id"]
            for item in receipt.reduction["selected_alternatives"]
        }
        exact = []
        for offer in plan.environment_mutations:
            alternative_id = selected.get(offer.environment_id)
            alternative = next(
                (item for item in offer.alternatives if item.alternative_id == alternative_id),
                None,
            )
            if alternative is None:
                raise ValueError("toolbox_confirmation_alternative_not_offered")
            exact.append(alternative.to_dict())
        exact_resolution_digest = identity_digest(
            "hosting.toolbox.confirmation.exact_resolution.v1",
            {
                "plan_id": plan.plan_id,
                "confirmation_ref_digest": receipt.confirmation_ref_digest,
                "choices_digest": receipt.choices_digest,
                "effective_definition_revision": receipt.reduction["effective_definition_revision"],
                "alternatives": exact,
            },
        )
        pins_digest = identity_digest(
            "hosting.toolbox.confirmation.plan_pins.v1", plan.pins.to_dict()
        )
        return exact_resolution_digest, pins_digest

    def toolbox_approve_confirmed_definition_plan(
        self,
        *,
        confirmation_ref: str,
        approver_actor_id: str,
        dependency_approver_authorized: bool = False,
    ) -> Dict[str, Any]:
        if not dependency_approver_authorized:
            raise PermissionError("dependency_approver_authorization_required")
        now_ms = int(time.time() * 1000)
        receipt = self._toolbox_confirmations.get_for_approval(
            confirmation_ref, now_ms=now_ms
        )
        record = self._toolbox_definition_plans.get(receipt.plan_id, now_ms=now_ms)
        context = self._toolbox_definition_planning_context()
        configuration = context["configuration"]
        if configuration is None or (
            self.hosting_configuration_revision != record.pins.configuration_revision
            or context["catalog_revision"] != record.pins.catalog_revision
            or context["policy"].revision != record.pins.dependency_policy_revision
            or configuration.config_revision != record.pins.host_config_revision
            or configuration.source_set_revision != record.pins.source_set_revision
            or context["target"] != record.pins.target
            or not context["policy"].allow_custom
            or not context["policy"].custom_requires_approval
        ):
            raise PermissionError("dependency_approval_invalid")
        if not bool(receipt.reduction["dependency_approval_required"]):
            raise ValueError("dependency_approval_not_required")
        exact_resolution_digest, pins_digest = self._toolbox_confirmation_approval_binding(
            record, receipt
        )
        return self._toolbox_dependency_approvals.mint(
            owner_actor_id=receipt.owner_actor_id,
            authority_id=receipt.authority_id,
            approver_actor_id=str(approver_actor_id or "").strip(),
            toolbox_id=record.toolbox_id,
            plan_id=record.plan_id,
            confirmation_ref_digest=receipt.confirmation_ref_digest,
            effective_definition_revision=receipt.reduction["effective_definition_revision"],
            exact_resolution_digest=exact_resolution_digest,
            plan_pins_digest=pins_digest,
            now_ms=now_ms,
            expires_at_ms=min(record.expires_at_ms, now_ms + 60 * 60 * 1000),
        )

    def toolbox_apply_definition(
        self,
        *,
        plan_id: str,
        confirmation_ref: str,
        request_id: str,
        dependency_approval_ref: Optional[str] = None,
        owner_actor_id: str = "service:local",
        authority_id: str = "authority:local",
    ) -> Dict[str, Any]:
        from ..toolbox.definition_planner import (
            ActiveToolboxProfileSnapshot,
            ToolboxDefinitionPlanDraft,
            classify_toolbox_profiles,
        )
        from ..toolbox.identity import identity_digest

        if dependency_approval_ref is not None and not isinstance(dependency_approval_ref, str):
            raise ValueError("dependency_approval_ref_must_be_opaque_string")
        rid = str(request_id or "").strip()
        if not rid or len(rid) > 128 or any(ord(character) < 32 or ord(character) > 126 for character in rid):
            raise ValueError("toolbox_apply_request_id_invalid")
        now_ms = int(time.time() * 1000)
        record = self._toolbox_definition_plans.get(plan_id, now_ms=now_ms)
        actor = str(owner_actor_id or "").strip()
        authority = str(authority_id or "").strip()
        if record.owner_actor_id != actor or record.authority_id != authority:
            raise PermissionError("toolbox_definition_plan_not_found")
        receipt = self._toolbox_confirmations.get(
            confirmation_ref,
            owner_actor_id=actor,
            authority_id=authority,
            now_ms=now_ms,
        )
        if receipt.plan_id != record.plan_id:
            raise ValueError("toolbox_confirmation_plan_mismatch")
        draft = ToolboxDefinitionPlanDraft.from_persisted_dict(receipt.confirmed_draft)
        model = draft.definition
        active = self._toolbox_state_v2.get(model.toolbox_id)
        if dict(active or {}).get("active_revision") != model.expected_revision:
            from .toolbox_state_v2 import ToolboxRevisionConflictError

            raise ToolboxRevisionConflictError("toolbox_revision_conflict")
        context = self._toolbox_definition_planning_context()
        configuration = context["configuration"]
        if configuration is None or (
            self.hosting_configuration_revision != record.pins.configuration_revision
            or context["catalog_revision"] != record.pins.catalog_revision
            or context["policy"].revision != record.pins.dependency_policy_revision
            or configuration.config_revision != record.pins.host_config_revision
            or configuration.source_set_revision != record.pins.source_set_revision
            or context["target"] != record.pins.target
        ):
            raise ValueError("toolbox_definition_plan_pins_changed")
        exact_resolution_digest, pins_digest = self._toolbox_confirmation_approval_binding(
            record, receipt
        )
        approval_identity = None
        if bool(receipt.reduction["dependency_approval_required"]):
            if not dependency_approval_ref:
                raise PermissionError("dependency_approval_required")
            self._toolbox_dependency_approvals.validate_and_consume(
                approval_ref=dependency_approval_ref,
                owner_actor_id=actor,
                authority_id=authority,
                toolbox_id=model.toolbox_id,
                plan_id=record.plan_id,
                confirmation_ref_digest=receipt.confirmation_ref_digest,
                effective_definition_revision=model.revision,
                exact_resolution_digest=exact_resolution_digest,
                plan_pins_digest=pins_digest,
                request_id=rid,
                now_ms=now_ms,
            )
            approval_identity = identity_digest(
                "hosting.toolbox.dependency_approval_ref.v1", dependency_approval_ref
            )
        elif dependency_approval_ref:
            raise ValueError("dependency_approval_not_required")
        fingerprint = hosted_execution_fingerprint(
            {
                "toolbox_id": model.toolbox_id,
                "definition_revision": model.revision,
                "expected_revision": model.expected_revision,
                "plan_id": record.plan_id,
                "confirmation_ref_digest": receipt.confirmation_ref_digest,
                "exact_resolution_digest": exact_resolution_digest,
                "plan_pins_digest": pins_digest,
                "approval_identity": approval_identity,
                "catalog_revision": record.pins.catalog_revision,
                "package_policy_revision": record.pins.dependency_policy_revision,
                "configuration_revision": record.pins.configuration_revision,
            }
        )
        prepared = self._hosted_operations.prepare(
            owner_actor_id=actor,
            execution_kind=HostedExecutionKind.TOOLBOX_DEFINITION_APPLY,
            selector={"kind": "toolbox_id", "id": model.toolbox_id},
            namespace=f"toolbox-definition:{model.toolbox_id}",
            request_id=rid,
            fingerprint=fingerprint,
            metadata={
                "toolbox_id": model.toolbox_id,
                "definition_revision": model.revision,
                "plan_id": record.plan_id,
                "configuration_revision": record.pins.configuration_revision,
            },
        )
        action = str(prepared.get("action") or "")
        status = dict(prepared.get("status") or {})
        if action != "dispatch":
            return status
        operation_id = str(dict(status.get("operation") or {}).get("operation_id") or "")
        active_profiles = []
        for profile_id, row in dict(dict(active or {}).get("profiles") or {}).items():
            profile = dict(row["profile"])
            active_profiles.append(
                ActiveToolboxProfileSnapshot(
                    profile_id=profile_id,
                    manifest_hash=row["manifest_hash"],
                    environment_key=profile["environment_key"],
                    sandbox_policy_digest=identity_digest(
                        "hosting.toolbox.sandbox_policy.v1", profile["sandbox_policy"]
                    ),
                    assigned_tool_keys=tuple(profile["assigned_tool_keys"]),
                )
            )
        profile_changes = classify_toolbox_profiles(draft, active_profiles)
        selected_alternatives = {
            item["environment_id"]: item["alternative_id"]
            for item in receipt.reduction["selected_alternatives"]
        }
        planned_environment_records = {
            item.environment_id: item.to_dict()
            for item in record.planned_environments
            if selected_alternatives.get(item.environment_id) == item.alternative_id
        }
        if set(planned_environment_records) != {item.profile_id for item in draft.profiles}:
            raise ValueError("toolbox_planned_environment_selection_incomplete")
        self._hosted_operations.update_progress(
            operation_id=operation_id,
            progress={
                "phase": "validation",
                "code": "definition_apply_queued",
                "completed_units": 0,
                "total_units": None,
                "updated_at_ms": int(time.time() * 1000),
                "summary": "The toolbox definition is queued for validation.",
                "cancellable": True,
            },
        )
        worker = threading.Thread(
            target=self._apply_resolved_toolbox_definition,
            kwargs={
                "draft": draft,
                "profile_changes": [dict(item) for item in profile_changes],
                "confirmation_result": dict(receipt.reduction),
                "resolved_environments": dict(receipt.resolved_environments),
                "planned_environment_records": planned_environment_records,
                "operation_id": operation_id,
            },
            name=f"toolbox-definition-apply-{operation_id[:12]}",
            daemon=True,
        )
        worker.start()
        return status

    def _apply_resolved_toolbox_definition(
        self,
        *,
        draft: Any,
        profile_changes: List[Dict[str, Any]],
        confirmation_result: Optional[Dict[str, Any]] = None,
        resolved_environments: Optional[Dict[str, Any]] = None,
        planned_environment_records: Optional[Dict[str, Any]] = None,
        operation_id: str,
    ) -> Dict[str, Any]:
        from .toolbox_rollout import ToolboxDefinitionRolloutCoordinator

        return self._run_locked_toolbox_call(
            str(draft.definition.toolbox_id),
            ToolboxDefinitionRolloutCoordinator(self).apply,
            draft=draft,
            profile_changes=profile_changes,
            confirmation_result=dict(confirmation_result or {}),
            resolved_environments=dict(resolved_environments or {}),
            planned_environment_records=(
                dict(planned_environment_records)
                if planned_environment_records is not None
                else None
            ),
            operation_id=str(operation_id or "").strip(),
        )

    def recover_toolbox_definition_rollouts(self) -> Dict[str, Any]:
        from .toolbox_rollout import ToolboxDefinitionRolloutCoordinator

        return ToolboxDefinitionRolloutCoordinator(self).recover()

    def _cleanup_toolbox_definition_apply_candidates(self, *, record: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(dict(record or {}).get("metadata") or {})
        candidates = sorted(
            {
                str(item or "").strip()
                for item in list(metadata.get("candidate_engine_ids") or [])
                if str(item or "").strip()
            }
        )
        toolbox_id = str(metadata.get("toolbox_id") or "").strip()
        active = self._active_toolbox_v2_snapshot(toolbox_id) if toolbox_id else None
        active_engine_ids = {
            str(dict(route or {}).get("engine_id") or "").strip()
            for route in dict(dict(active or {}).get("tool_routes") or {}).values()
        }
        cleaned: list[str] = []
        for engine_id in candidates:
            if engine_id in active_engine_ids:
                continue
            self._retire_toolbox_registration(engine_id)
            cleaned.append(engine_id)
        return {"status": "complete", "candidate_count": len(cleaned)}

    def toolbox_definition_apply_operator_details(
        self,
        *,
        operation_id: str,
        operator_authorized: bool,
    ) -> Dict[str, Any]:
        if not bool(operator_authorized):
            raise PermissionError("toolbox_operator_details_denied")
        oid = str(operation_id or "").strip()
        if not oid or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-" for character in oid):
            raise ValueError("operation_id_invalid")
        path = (
            self.hosting_root / "state" / "toolbox_rollout_operator_details" / f"{oid}.json"
        ).resolve()
        try:
            path.relative_to((self.hosting_root / "state" / "toolbox_rollout_operator_details").resolve())
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("toolbox_operator_details_unavailable") from exc
        if not isinstance(payload, dict):
            raise ValueError("toolbox_operator_details_unavailable")
        return payload

    @staticmethod
    def _toolbox_operation_namespace(*, engine_id: str = "", toolbox_id: str = "") -> str:
        tid = str(toolbox_id or "").strip()
        eid = str(engine_id or "").strip()
        if tid:
            return f"toolbox:{tid}"
        if eid:
            return f"engine:{eid}"
        raise ValueError("engine_id or toolbox_id is required")

    @staticmethod
    def _registration_allowed_tool_names(reg: Dict[str, Any]) -> Optional[set[str]]:
        tool_access = dict(reg.get("tool_access") or {})
        allowed = {
            str(item or "").strip()
            for item in list(tool_access.get("allowed_tool_names") or [])
            if str(item or "").strip()
        }
        return allowed or None

    @staticmethod
    def _registration_advertised_tool_names(reg: Dict[str, Any]) -> Optional[set[str]]:
        tool_access = dict(reg.get("tool_access") or {})
        advertised = {
            str(item or "").strip()
            for item in list(tool_access.get("advertised_tool_names") or [])
            if str(item or "").strip()
        }
        return advertised or None

    @staticmethod
    def _registration_hidden_allowed_tool_names(reg: Dict[str, Any]) -> Optional[set[str]]:
        tool_access = dict(reg.get("tool_access") or {})
        hidden = {
            str(item or "").strip()
            for item in list(tool_access.get("hidden_allowed_tool_names") or [])
            if str(item or "").strip()
        }
        return hidden or None

    @staticmethod
    def _tools_view_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[ToolsView]:
        row = dict(payload or {})
        if not row:
            return None
        return ToolsView(
            view_id=str(row.get("view_id") or "").strip() or "hosted-tools-view",
            mode=str(row.get("mode") or "").strip() or "advertised",
            allowed_tools=set(str(item or "").strip() for item in list(row.get("allowed_tools") or []) if str(item or "").strip()),
            advertised_tools=set(str(item or "").strip() for item in list(row.get("advertised_tools") or []) if str(item or "").strip()),
            hidden_allowed_tools=set(
                str(item or "").strip() for item in list(row.get("hidden_allowed_tools") or []) if str(item or "").strip()
            ),
            disabled_tools=set(str(item or "").strip() for item in list(row.get("disabled_tools") or []) if str(item or "").strip()),
            gated_tools=set(str(item or "").strip() for item in list(row.get("gated_tools") or []) if str(item or "").strip()),
            tool_constraints={
                str(tool_name or "").strip(): json.loads(json.dumps(dict(item or {})))
                for tool_name, item in dict(row.get("tool_constraints") or {}).items()
                if str(tool_name or "").strip() and isinstance(item, dict)
            },
        )

    @staticmethod
    def _registration_tool_routes(reg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        tool_access = dict(reg.get("tool_access") or {})
        routes = dict(tool_access.get("tool_routes") or {})
        out: Dict[str, Dict[str, Any]] = {}
        for raw_name, raw_meta in routes.items():
            name = str(raw_name or "").strip()
            if not name:
                continue
            out[name] = dict(raw_meta or {})
        return out

    @staticmethod
    def _registration_toolbox_id(reg: Dict[str, Any]) -> str:
        bundle = dict(reg.get("bundle") or {})
        return str(bundle.get("toolbox_id") or bundle.get("bundle_id") or "").strip()

    @staticmethod
    def _callback_context_payload(context: Any) -> Dict[str, Any]:
        return {
            "engine_id": str(getattr(context, "engine_id", "") or "").strip() or None,
            "toolbox_id": str(getattr(context, "toolbox_id", "") or "").strip() or None,
            "tool_name": str(getattr(context, "tool_name", "") or "").strip() or None,
            "tool_call_id": str(getattr(context, "tool_call_id", "") or "").strip() or None,
            "tool_arguments": dict(getattr(context, "tool_arguments", {}) or {}),
        }

    def _toolbox_host_capability_dispatch_binding(
        self,
        *,
        engine_id: str,
        toolbox_id: str,
        tool_name: str,
        tool_call_id: str,
        tool_arguments: Dict[str, Any],
        sandbox_policy: Dict[str, Any],
        callback_binding: Optional[Dict[str, Any]] = None,
        host_api_approval: Optional[Dict[str, Any]] = None,
    ) -> Tuple[_HostedToolCallbackRelay, Dict[str, Any]]:
        original_binding = dict(callback_binding or {}) if isinstance(callback_binding, dict) else {}
        eid = str(engine_id or "").strip()
        relay = _HostedToolCallbackRelay()

        def _forward_callback(*, callback_name: str, payload: Any, context: Any) -> Dict[str, Any]:
            if not original_binding:
                return {"status": "error", "message": "callback_binding_missing"}
            from ..toolbox_executor_ipc import _invoke_callback_binding

            response = _invoke_callback_binding(
                original_binding,
                callback_name=str(callback_name or "").strip(),
                payload=payload,
                context=self._callback_context_payload(context),
            )
            return dict(response.get("result") or {}) if isinstance(response.get("result"), dict) else {"result": response.get("result")}

        def _approval_requester(payload: Dict[str, Any]) -> Dict[str, Any]:
            if not original_binding:
                return {"status": "denied", "approved": False, "decision": "deny", "reason": "approval_requester_unavailable"}
            from ..toolbox_executor_ipc import _invoke_callback_binding

            response = _invoke_callback_binding(
                original_binding,
                callback_name=HOST_CAPABILITY_APPROVAL_CALLBACK_NAME,
                payload=host_capability_approval_request(dict(payload or {})),
                context=dict(dict(payload or {}).get("context") or {}),
            )
            return dict(response.get("result") or response or {})

        def _dispatch_host_capability(payload: Dict[str, Any], context: Any) -> Dict[str, Any]:
            row = dict(payload or {})
            method = str(row.get("method") or "").strip()
            arguments = dict(row.get("arguments") or {}) if isinstance(row.get("arguments"), dict) else {}
            approval = dict(row.get("approval") or host_api_approval or {}) if isinstance(row.get("approval") or host_api_approval, dict) else {}
            callback_context = dict(arguments.get("callback_context") or {}) if isinstance(arguments.get("callback_context"), dict) else {}
            broker = HostCapabilityBroker(
                request_id=str(callback_context.get("tool_call_id") or getattr(context, "tool_call_id", "") or tool_call_id or ""),
                workflow_id=str(callback_context.get("workflow_id") or ""),
                package_id=str(callback_context.get("package_id") or ""),
                instance_id=str(callback_context.get("instance_id") or ""),
                engine_id=eid,
                consumer_id=eid,
                runtime_kind="toolbox_worker",
                policy=dict(sandbox_policy or {}),
                provider_invoker=self._host_capability_provider_invoker,
                approval_requester=_approval_requester if approval else None,
                audit_emitter=self._append_host_capability_audit_event,
            )
            broker.register_session(
                service_broker_host_capability_session(
                    session_id=f"{eid}.service_broker",
                    provider_id="builtin.service_broker",
                    owner="service",
                    visibility="consumer",
                    scope={"consumer_id": eid},
                    approval=approval,
                    binding={"engine_id": eid},
                )
            )
            result = broker.dispatch({"method": method, "arguments": arguments})
            return {"status": "ok", "result": dict(result or {})}

        def _processor(*, callback_name: str, payload: Any, context: Any) -> Dict[str, Any]:
            name = str(callback_name or "").strip()
            if name == HOST_CAPABILITY_DISPATCH_CALLBACK_NAME:
                return _dispatch_host_capability(dict(payload or {}) if isinstance(payload, dict) else {}, context)
            return _forward_callback(callback_name=name, payload=payload, context=context)

        binding = relay.bind_session(
            processor=_processor,
            toolbox_id=str(toolbox_id or "").strip(),
            tool_name=str(tool_name or "").strip(),
            tool_call_id=str(tool_call_id or "").strip(),
            tool_arguments=dict(tool_arguments or {}),
            callback_signature={"callbacks": [{"name": HOST_CAPABILITY_DISPATCH_CALLBACK_NAME, "payload_type": "object"}]},
            user_context=None,
        )
        return relay, binding

    def _toolbox_executor_registrations(self, toolbox_id: str) -> List[Dict[str, Any]]:
        tid = str(toolbox_id or "").strip()
        if not tid:
            return []
        rows: List[Dict[str, Any]] = []
        for row in self._read_engines():
            reg = dict(row or {})
            if str(reg.get("executor_kind") or "").strip() != "toolbox_executor":
                continue
            if self._registration_toolbox_id(reg) != tid:
                continue
            rows.append(reg)
        return rows

    def _active_toolbox_v2_snapshot(self, toolbox_id: str) -> Optional[Dict[str, Any]]:
        repository = getattr(self, "_toolbox_state_v2", None)
        if repository is None:
            return None
        try:
            return repository.get(str(toolbox_id or "").strip())
        except Exception as exc:
            # The deprecated procedural adapter remains isolated until Phase 6
            # removes it. Definition APIs call the strict repository directly
            # and reject version 1; legacy execution may still route its own
            # registrations during that temporary compatibility window.
            from .toolbox_state_v2 import LegacyToolboxStateError

            if isinstance(exc, LegacyToolboxStateError):
                return None
            raise

    def _active_toolbox_v2_registrations(self, toolbox_id: str) -> Optional[List[Dict[str, Any]]]:
        snapshot = self._active_toolbox_v2_snapshot(toolbox_id)
        if snapshot is None:
            return None
        engine_ids = sorted(
            {
                str(dict(route or {}).get("engine_id") or "").strip()
                for route in dict(snapshot.get("tool_routes") or {}).values()
            }
            - {""}
        )
        registrations: List[Dict[str, Any]] = []
        for engine_id in engine_ids:
            reg = dict(self._find_registration(engine_id) or {})
            if not reg or self._registration_toolbox_id(reg) != str(toolbox_id or "").strip():
                raise RuntimeError(f"toolbox_active_route_registration_missing:{toolbox_id}:{engine_id}")
            registrations.append(reg)
        return registrations

    @staticmethod
    def _definition_tool_metadata(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        definition = dict(dict(snapshot or {}).get("definition") or {})
        metadata: Dict[str, Dict[str, Any]] = {}
        for request in list(definition.get("auto_requests") or []):
            row = dict(request or {})
            name = str(row.get("callable_name") or "").strip()
            if name:
                metadata[name] = {
                    "tool_definition": {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": str(row.get("guide_description") or ""),
                        },
                    },
                    "hidden": bool(row.get("hidden", False)),
                    "non_restartable": bool(row.get("non_restartable", False)),
                    "callback_signature": row.get("callback_signature"),
                    "concurrency": row.get("concurrency"),
                }
        for request in list(definition.get("manual_requests") or []):
            row = dict(request or {})
            tool_definition = dict(row.get("tool_definition") or {})
            name = str(dict(tool_definition.get("function") or {}).get("name") or "").strip()
            if name:
                metadata[name] = {
                    "tool_definition": tool_definition,
                    "hidden": bool(row.get("hidden", False)),
                    "non_restartable": bool(row.get("non_restartable", False)),
                    "callback_signature": row.get("callback_signature"),
                    "concurrency": row.get("concurrency"),
                }
        for name in list(dict(definition.get("intrinsics") or {}).get("names") or []):
            normalized = str(name or "").strip()
            if normalized:
                metadata.setdefault(normalized, {"hidden": False, "non_restartable": False})
        return metadata

    def _cleanup_toolbox_bundle_root(self, reg: Dict[str, Any]) -> None:
        bundle = dict(reg.get("bundle") or {})
        raw = str(bundle.get("bundle_root") or "").strip()
        if not raw:
            return
        root = Path(raw).expanduser().resolve()
        allowed_root = (self.hosting_root / "toolbox_bundles").resolve()
        try:
            if root != allowed_root and allowed_root not in root.parents:
                return
        except Exception:
            return
        shutil.rmtree(root, ignore_errors=True)

    def _retire_toolbox_registration(self, engine_id: str) -> None:
        reg = self._find_registration(engine_id)
        if reg:
            env = dict(reg.get("env") or {})
            spec_path = str(env.get("MP13_TOOLBOX_WORKER_SPEC_PATH") or "").strip()
            scratch_root = str(
                dict(reg.get("bundle") or {}).get("scratch_root")
                or env.get("MP13_TOOLBOX_SCRATCH_ROOT")
                or ""
            ).strip()
            try:
                self.shutdown(engine_id, timeout_seconds=2.0)
            except Exception:
                pass
            self.remove_registration(engine_id)
            self._cleanup_toolbox_bundle_root(reg)
            for raw_path, allowed_parent in (
                (spec_path, (self.hosting_root / "state" / "toolbox_worker_specs").resolve()),
                (scratch_root, (self.hosting_root / "toolbox_scratch").resolve()),
            ):
                if not raw_path:
                    continue
                try:
                    path = Path(raw_path).expanduser().resolve()
                    parent = Path(allowed_parent).resolve()
                    if path != parent and parent not in path.parents:
                        continue
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
                except Exception:
                    pass


    @staticmethod
    def _registration_sandbox_profile_id(reg: Dict[str, Any]) -> str:
        return str(dict(reg.get("bundle") or {}).get("sandbox_profile_id") or "default").strip() or "default"

    def _route_toolbox_registration(self, *, toolbox_id: str, tool_name: str, command_label: str) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        name = str(tool_name or "").strip()
        if not tid:
            raise ValueError("toolbox_id is required")
        if not name:
            raise ValueError("tool_name is required")
        snapshot = self._active_toolbox_v2_snapshot(tid)
        if snapshot is not None:
            route = dict(dict(snapshot.get("tool_routes") or {}).get(name) or {})
            if not route:
                raise PermissionError(f"tool_not_allowed:{name}")
            engine_id = str(route.get("engine_id") or "").strip()
            profile_id = str(route.get("profile_id") or "").strip()
            reg = dict(self._find_registration(engine_id) or {})
            bundle = dict(reg.get("bundle") or {})
            if (
                not reg
                or self._registration_toolbox_id(reg) != tid
                or str(bundle.get("resolved_profile_id") or bundle.get("sandbox_profile_id") or "") != profile_id
            ):
                raise RuntimeError(f"toolbox_active_route_registration_mismatch:{tid}:{name}")
            return self._require_toolbox_executor_registration(engine_id, command_label=command_label)
        matches: List[Dict[str, Any]] = []
        for reg in self._toolbox_executor_registrations(tid):
            if str(reg.get("routing_state") or "active") != "active":
                continue
            allowed = self._registration_allowed_tool_names(reg)
            if allowed is not None and name in allowed:
                matches.append(reg)
        if not matches:
            raise PermissionError(f"tool_not_allowed:{name}")
        if len(matches) > 1:
            raise RuntimeError(f"toolbox_route_ambiguous:{tid}:{name}")
        return self._require_toolbox_executor_registration(
            str(matches[0].get("engine_id") or ""),
            command_label=command_label,
        )

    def _require_toolbox_executor_registration(self, engine_id: str, *, command_label: str) -> Dict[str, Any]:
        reg = self._require_ipc_registration(engine_id, command_label=command_label)
        executor_kind = str(reg.get("executor_kind") or "").strip()
        if executor_kind and executor_kind != "toolbox_executor":
            raise ValueError(f"{command_label} is only supported for toolbox executors")
        return reg

    def _toolbox_runtime_base(self) -> HostedToolboxRuntimeBase:
        from ..sandbox.toolbox_runtime import HostedToolboxRuntimeBase

        base = getattr(self, "_hosted_toolbox_runtime_base", None)
        if not isinstance(base, HostedToolboxRuntimeBase):
            base = HostedToolboxRuntimeBase()
            setattr(self, "_hosted_toolbox_runtime_base", base)
        return base

    @staticmethod
    def _toolbox_registration_environment_key(reg: Dict[str, Any]) -> str:
        env = dict(dict(reg or {}).get("environment") or {})
        caps = dict(dict(reg or {}).get("capabilities") or {})
        return str(env.get("environment_key") or caps.get("environment_key") or dict(reg or {}).get("engine_id") or "").strip()

    @staticmethod
    def _toolbox_registration_capacity(reg: Dict[str, Any]) -> int:
        caps = dict(dict(reg or {}).get("capabilities") or {})
        for key in ("capacity", "max_concurrency", "max_parallel_calls"):
            try:
                value = int(caps.get(key) or 0)
            except Exception:
                value = 0
            if value > 0:
                return max(1, min(value, 1024))
        return 256

    @classmethod
    def _toolbox_registration_described_capacity(cls, reg: Dict[str, Any]) -> int:
        """Return configured capacity for discovery before a runtime pool exists."""
        caps = dict(dict(reg or {}).get("capabilities") or {})
        for key in ("capacity", "max_concurrency", "max_parallel_calls"):
            if key not in caps or caps.get(key) is None:
                continue
            try:
                value = int(caps.get(key))
            except Exception:
                return cls._toolbox_registration_capacity(reg)
            return max(0, min(value, 1024))
        return cls._toolbox_registration_capacity(reg)

    @staticmethod
    def _toolbox_registration_queue_config(reg: Dict[str, Any]) -> Dict[str, Any]:
        caps = dict(dict(reg or {}).get("capabilities") or {})
        policy = str(caps.get("queue_policy") or caps.get("concurrency_queue_policy") or "bounded").strip().lower()
        if policy not in {"bounded", "fail_fast"}:
            policy = "bounded"

        depth = 32
        for key in ("queue_depth", "max_queue_depth"):
            if key not in caps or caps.get(key) is None:
                continue
            try:
                depth = int(caps.get(key))
            except Exception:
                depth = 32
            break

        timeout = 30.0
        for key in ("queue_timeout_seconds", "queue_wait_timeout_seconds"):
            if key not in caps or caps.get(key) is None:
                continue
            try:
                timeout = float(caps.get(key))
            except Exception:
                timeout = 30.0
            break
        return {
            "queue_policy": policy,
            "queue_depth": max(0, min(depth, 4096)),
            "queue_timeout_seconds": max(0.0, min(timeout, 3600.0)),
        }

    def _toolbox_tool_concurrency_policy(
        self,
        *,
        toolbox_id: str,
        tool_name: str,
        call: Dict[str, Any],
    ) -> Dict[str, Any]:
        tid = str(toolbox_id or "").strip()
        name = str(tool_name or "").strip()
        snapshot = self._active_toolbox_v2_snapshot(tid) if tid else None
        metadata = dict(self._definition_tool_metadata(snapshot).get(name) or {})
        raw = dict(metadata.get("concurrency") or {})
        mode = str(raw.get("mode") or "parallel").strip().lower()
        if mode not in {"parallel", "serial", "keyed", "exclusive"}:
            mode = "parallel"
        group = str(raw.get("group") or "").strip()
        if mode in {"serial", "keyed"} and not group:
            group = name or "tool"
        if mode == "exclusive" and not group:
            group = "toolbox"
        arguments = dict(call.get("arguments") or {}) if isinstance(call.get("arguments"), dict) else {}
        resource_key = str(raw.get("resource_key") or "").strip()
        key_arguments = raw.get("key_arguments") or raw.get("resource_key_arguments")
        if not resource_key:
            key_argument = str(raw.get("key_argument") or raw.get("resource_key_argument") or "").strip()
            if key_argument:
                key_arguments = [key_argument]
            if isinstance(key_arguments, str):
                key_arguments = [key_arguments]
            if mode == "keyed":
                values: List[Any] = []
                for key in list(key_arguments or []):
                    current: Any = arguments
                    for part in str(key or "").split("."):
                        if isinstance(current, dict):
                            current = current.get(part)
                        else:
                            current = None
                    values.append(current)
                if not values:
                    values = [arguments.get("resource_key", arguments.get("key", "__missing__"))]
                resource_key = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        try:
            max_concurrency = int(raw.get("max_concurrency") or 0)
        except Exception:
            max_concurrency = 0
        if mode == "serial":
            max_concurrency = 1
        if max_concurrency > 0 and not group:
            group = name or "tool"
        return {
            "mode": mode,
            "group": group,
            "resource_key": resource_key,
            "max_concurrency": max(0, max_concurrency),
            "decision": "compatibility_default" if not raw else "declared",
        }

    def _toolbox_worker_slot(self, *, reg: Dict[str, Any], environment_key: str, capacity: int) -> Any:
        from ..sandbox.toolbox_runtime import HostedToolboxRuntimeBase

        return HostedToolboxRuntimeBase.worker_slot(
            engine_id=str(dict(reg or {}).get("engine_id") or "").strip(),
            environment_key=str(environment_key or "").strip(),
            capacity=capacity,
            pid=int(dict(reg or {}).get("pid") or 0) or None,
            status="registered",
        )

    def _toolbox_pool_resources(self, reg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        environment_key = self._toolbox_registration_environment_key(reg)
        if not environment_key:
            return None
        resources = self._toolbox_runtime_base().resources(environment_key)
        return dict(resources or {}) if str(dict(resources or {}).get("status") or "") != "not_found" else None

    def toolbox_describe(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        operator_details: bool = False,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        """Return a bounded persisted/registration view without contacting workers."""

        eid = str(engine_id or "").strip()
        tid = str(toolbox_id or "").strip()
        if tid:
            return self._toolbox_describe_live(
                engine_id=eid,
                toolbox_id=tid,
                timeout_seconds=float(timeout_seconds or 10.0),
            )
        if not eid:
            raise ValueError("engine_id or toolbox_id is required")
        reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-describe")
        allowed = sorted(self._registration_allowed_tool_names(reg) or set())
        advertised = sorted(self._registration_advertised_tool_names(reg) or set())
        return {
            "status": "ok",
            "engine_id": eid,
            "executor_kind": "toolbox_executor",
            "mode": "sandbox",
            "cache": "registration",
            "bundle": dict(reg.get("bundle") or {}),
            "all_registered_tool_names": allowed,
            "allowed_tool_names": allowed,
            "advertised_tool_names": advertised,
            "hidden_allowed_tool_names": sorted(self._registration_hidden_allowed_tool_names(reg) or set()),
            "user_projection": {
                "state": "ready" if allowed or advertised else "starting",
                "code": "toolbox_runtime_cached",
                "summary": "The persisted toolbox runtime view is available.",
            },
        }

    def _toolbox_describe_live(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        tid = str(toolbox_id or "").strip()
        if not eid and not tid:
            raise ValueError("engine_id or toolbox_id is required")
        if tid and not eid:
            regs = self._active_toolbox_v2_registrations(tid)
            snapshot = self._active_toolbox_v2_snapshot(tid)
            if regs is None:
                raise ValueError(f"toolbox '{tid}' has no active definition")
            if not regs and snapshot is None:
                raise ValueError(f"toolbox '{tid}' has no registered sandbox executors")
            tool_names: set[str] = set()
            advertised_tool_names: set[str] = set()
            hidden_allowed_tool_names: set[str] = set()
            sandbox_profile_ids: set[str] = set()
            engine_ids: List[str] = []
            hosted_pools: Dict[str, Any] = {}
            parallel_rows: List[Dict[str, Any]] = []
            for reg in regs:
                reg_engine_id = str(reg.get("engine_id") or "")
                engine_ids.append(reg_engine_id)
                pool = self._toolbox_pool_resources(reg)
                if pool is not None and reg_engine_id:
                    hosted_pools[reg_engine_id] = pool
                registration_queue = self._toolbox_registration_queue_config(reg)
                registration_capacity = self._toolbox_registration_described_capacity(reg)
                metrics = dict(pool.get("metrics") or {}) if pool is not None else {}
                parallel_rows.append(
                    {
                        "effective_max_concurrency": int(metrics["desired_capacity"])
                        if pool is not None and "desired_capacity" in metrics
                        else registration_capacity,
                        "queue_policy": str(metrics["queue_policy"])
                        if pool is not None and "queue_policy" in metrics
                        else registration_queue["queue_policy"],
                        "queue_depth": int(metrics["queue_depth"])
                        if pool is not None and "queue_depth" in metrics
                        else registration_queue["queue_depth"],
                        "queue_timeout_seconds": float(metrics["queue_timeout_seconds"])
                        if pool is not None and "queue_timeout_seconds" in metrics
                        else registration_queue["queue_timeout_seconds"],
                        "active_calls": int(metrics.get("active_calls") or 0),
                        "queued_calls": int(metrics.get("queued_calls") or 0),
                        "worker_process_count": int(metrics.get("worker_count") or 0),
                    }
                )
                for name in list(self._registration_allowed_tool_names(reg) or set()):
                    tool_names.add(name)
                for name in list(self._registration_advertised_tool_names(reg) or set()):
                    advertised_tool_names.add(name)
                for name in list(self._registration_hidden_allowed_tool_names(reg) or set()):
                    hidden_allowed_tool_names.add(name)
                sandbox_profile_ids.add(str(dict(reg.get("bundle") or {}).get("sandbox_profile_id") or "default"))
            if snapshot is not None:
                tool_names = set(dict(snapshot.get("tool_routes") or {}))
                advertised_tool_names.intersection_update(tool_names)
                hidden_allowed_tool_names.intersection_update(tool_names)
            return {
                "status": "ok",
                "toolbox_id": tid,
                "all_registered_tool_names": sorted(tool_names),
                "allowed_tool_names": sorted(tool_names),
                "advertised_tool_names": sorted(advertised_tool_names or tool_names),
                "hidden_allowed_tool_names": sorted(hidden_allowed_tool_names),
                "tool_metadata": self._definition_tool_metadata(snapshot),
                "executor_kind": "toolbox_executor",
                "mode": "sandbox",
                "parallel_execution": {
                    "async_within_executor": True,
                    "sandbox_pool": len(engine_ids) > 1,
                    "supported": True,
                    "effective_max_concurrency": max(
                        [int(row["effective_max_concurrency"]) for row in parallel_rows] or [0]
                    ),
                    "queue_policy": "bounded"
                    if any(row["queue_policy"] == "bounded" for row in parallel_rows)
                    else "fail_fast",
                    "queue_depth": max(
                        [int(row["queue_depth"]) for row in parallel_rows] or [0]
                    ),
                    "queue_timeout_seconds": max(
                        [float(row["queue_timeout_seconds"]) for row in parallel_rows] or [0.0]
                    ),
                    "active_calls": sum(
                        [int(row["active_calls"]) for row in parallel_rows]
                    ),
                    "queued_calls": sum(
                        [int(row["queued_calls"]) for row in parallel_rows]
                    ),
                    "worker_process_count": sum(
                        [int(row["worker_process_count"]) for row in parallel_rows]
                    ),
                    "execution_model": "threaded_worker",
                },
                "user_projection": {
                    "state": "ready",
                    "code": "toolbox_runtime_ready",
                    "summary": "The toolbox runtime is ready.",
                },
            }
        reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-describe")
        out = self._ipc_call(
            reg=reg,
            payload={"kind": "rpc_call", "engine_id": eid, "method": "toolbox.describe", "params": {}},
            timeout_seconds=float(timeout_seconds or 10.0),
        )
        if str(out.get("status") or "").strip().lower() == "error":
            raise RuntimeError(str(out.get("message") or "toolbox_describe_failed"))
        result = dict(out or {})
        result.setdefault("engine_id", eid)
        result.setdefault("executor_kind", str(reg.get("executor_kind") or "toolbox_executor"))
        result.setdefault("bundle", dict(reg.get("bundle") or {}))
        result.setdefault("tool_access", dict(reg.get("tool_access") or {}))
        result.setdefault("all_registered_tool_names", sorted(list(self._registration_allowed_tool_names(reg) or set())))
        result.setdefault("allowed_tool_names", sorted(list(self._registration_allowed_tool_names(reg) or set())))
        result.setdefault("advertised_tool_names", sorted(list(self._registration_advertised_tool_names(reg) or set())))
        result.setdefault("hidden_allowed_tool_names", sorted(list(self._registration_hidden_allowed_tool_names(reg) or set())))
        pool = self._toolbox_pool_resources(reg)
        if pool is not None:
            result.setdefault("hosted_pool", pool)
            result.setdefault("toolbox_pool", pool)
        metrics = dict(pool.get("metrics") or {}) if pool is not None else {}
        registration_capacity = self._toolbox_registration_described_capacity(reg)
        registration_queue = self._toolbox_registration_queue_config(reg)
        parallel = dict(result.get("parallel_execution") or {})
        effective_max = (
            int(metrics["desired_capacity"])
            if pool is not None and "desired_capacity" in metrics
            else int(parallel.get("effective_max_concurrency") or 0) or registration_capacity
        )
        queue_depth = (
            int(metrics["queue_depth"])
            if pool is not None and "queue_depth" in metrics
            else int(parallel.get("queue_depth") or 0) or registration_queue["queue_depth"]
        )
        queue_timeout = (
            float(metrics["queue_timeout_seconds"])
            if pool is not None and "queue_timeout_seconds" in metrics
            else float(parallel.get("queue_timeout_seconds") or 0.0) or registration_queue["queue_timeout_seconds"]
        )
        parallel.update(
            {
                "supported": bool(parallel.get("supported", True)),
                "async_within_executor": bool(parallel.get("async_within_executor", True)),
                "sandbox_pool": bool(parallel.get("sandbox_pool", False)),
                "effective_max_concurrency": effective_max,
                "queue_policy": str(parallel.get("queue_policy") or metrics.get("queue_policy") or registration_queue["queue_policy"]),
                "queue_depth": queue_depth,
                "queue_timeout_seconds": queue_timeout,
                "active_calls": int(parallel.get("active_calls") or metrics.get("active_calls") or 0),
                "queued_calls": int(parallel.get("queued_calls") or metrics.get("queued_calls") or 0),
                "worker_process_count": int(parallel.get("worker_process_count") or metrics.get("worker_count") or 0),
                "execution_model": str(parallel.get("execution_model") or "threaded_worker"),
            }
        )
        result["parallel_execution"] = parallel
        result.pop("tool_names", None)
        return result

    def toolbox_describe_refresh(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        request_id: str,
        owner_actor_id: str = "service:local",
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        """Schedule an explicit live worker describe through hosted operations."""

        eid = str(engine_id or "").strip()
        tid = str(toolbox_id or "").strip()
        if bool(eid) == bool(tid):
            raise ValueError("engine_id_or_toolbox_id_required")
        selector = HostedOperationSelector(kind="engine_id" if eid else "toolbox_id", id=eid or tid)
        owner = str(owner_actor_id or "service:local").strip() or "service:local"
        prepared = self._hosted_operations.prepare(
            owner_actor_id=owner,
            execution_kind=HostedExecutionKind.TOOLBOX_DESCRIBE_REFRESH,
            selector=selector,
            namespace=f"toolbox_describe_refresh:{selector.kind}:{selector.id}",
            request_id=str(request_id or "").strip(),
            fingerprint=hosted_execution_fingerprint(
                {
                    "execution_kind": HostedExecutionKind.TOOLBOX_DESCRIBE_REFRESH.value,
                    "configuration_revision": self.hosting_configuration_revision,
                    "selector": selector.to_dict(),
                }
            ),
            metadata={
                "configuration_revision": self.hosting_configuration_revision,
                "engine_id": eid,
                "toolbox_id": tid,
                "retain_terminal_result": True,
            },
        )
        status = dict(prepared.get("status") or {})
        if prepared.get("action") != "dispatch":
            return status
        operation_id = str(dict(status.get("operation") or {}).get("operation_id") or "")
        threading.Thread(
            target=self._run_toolbox_describe_refresh,
            kwargs={
                "operation_id": operation_id,
                "engine_id": eid,
                "toolbox_id": tid,
                "timeout_seconds": float(timeout_seconds or 10.0),
            },
            name=f"toolbox-describe-refresh-{operation_id[:12]}",
            daemon=True,
        ).start()
        return status

    def _run_toolbox_describe_refresh(
        self,
        *,
        operation_id: str,
        engine_id: str,
        toolbox_id: str,
        timeout_seconds: float,
    ) -> None:
        self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
        try:
            self._hosted_operations.update_progress(
                operation_id=operation_id,
                progress=HostedOperationProgress(
                    phase="validation",
                    code="toolbox_describe_refresh_validated",
                    completed_units=0,
                    total_units=1,
                    updated_at_ms=int(time.time() * 1000),
                    summary="The live toolbox refresh request is valid.",
                    cancellable=True,
                ),
            )
            self._hosted_operations.update_progress(
                operation_id=operation_id,
                progress=HostedOperationProgress(
                    phase="refresh",
                    code="toolbox_describe_refresh_started",
                    completed_units=0,
                    total_units=1,
                    updated_at_ms=int(time.time() * 1000),
                    summary="The worker is being queried for a live toolbox description.",
                    cancellable=False,
                ),
            )
            result = self._toolbox_describe_live(
                engine_id=engine_id,
                toolbox_id=toolbox_id,
                timeout_seconds=timeout_seconds,
            )
            self._hosted_operations.update_progress(
                operation_id=operation_id,
                progress=HostedOperationProgress(
                    phase="cleanup",
                    code="toolbox_describe_refresh_completed",
                    completed_units=1,
                    total_units=1,
                    updated_at_ms=int(time.time() * 1000),
                    summary="The live toolbox description is ready.",
                    cancellable=False,
                ),
            )
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_SUCCESS,
                envelope={
                    "contract": "hosting.toolbox.describe_refresh_result.v1",
                    "status": "ok",
                    "code": "toolbox_describe_refresh_completed",
                    "description": result,
                },
            )
        except Exception as exc:
            self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "contract": "hosting.toolbox.describe_refresh_result.v1",
                    "status": "error",
                    "code": "toolbox_describe_refresh_failed",
                    "diagnostics": [{"code": "toolbox_describe_refresh_failed", "summary": str(exc)}],
                },
                reason="toolbox_describe_refresh_failed",
            )

    def toolbox_gate(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        tool_name: str,
        tools_view: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        tid = str(toolbox_id or "").strip()
        name = str(tool_name or "").strip()
        view = self._tools_view_from_payload(tools_view)
        if not name:
            raise ValueError("tool_name is required")
        if not eid and not tid:
            raise ValueError("engine_id or toolbox_id is required")
        if tid and not eid:
            v2_snapshot = self._active_toolbox_v2_snapshot(tid)
            regs = (
                self._active_toolbox_v2_registrations(tid)
                if v2_snapshot is not None
                else self._toolbox_executor_registrations(tid)
            )
            if not regs and v2_snapshot is None:
                return {
                    "status": "ok",
                    "toolbox_id": tid,
                    "tool_name": name,
                    "outcome": "unavailable_backend",
                    "reason": "toolbox_executor_missing",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
            allowed_for_toolbox: set[str] = (
                set(dict(v2_snapshot.get("tool_routes") or {}))
                if v2_snapshot is not None
                else set()
            )
            if v2_snapshot is None:
                for item in regs:
                    allowed_for_toolbox.update(self._registration_allowed_tool_names(item) or set())
            if view is not None and name in allowed_for_toolbox and view.is_gated(name):
                return {
                    "status": "ok",
                    "toolbox_id": tid,
                    "tool_name": name,
                    "outcome": "gated_requires_confirmation",
                    "reason": "gated_requires_confirmation",
                    "executable": False,
                    "requires_confirmation": True,
                    "backend": "sandbox",
                }
            if view is not None and name in allowed_for_toolbox and not view.is_allowed(name):
                return {
                    "status": "ok",
                    "toolbox_id": tid,
                    "tool_name": name,
                    "outcome": "denied",
                    "reason": "blocked_in_scope",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
            try:
                reg = self._route_toolbox_registration(toolbox_id=tid, tool_name=name, command_label="toolbox-gate")
            except PermissionError:
                return {
                    "status": "ok",
                    "toolbox_id": tid,
                    "tool_name": name,
                    "outcome": "denied",
                    "reason": "tool_not_allowed",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
            eid = str(reg.get("engine_id") or "").strip()
        else:
            reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-gate")
            allowed_tool_names = self._registration_allowed_tool_names(reg)
            if allowed_tool_names is not None and name not in allowed_tool_names:
                return {
                    "status": "ok",
                    "engine_id": eid,
                    "tool_name": name,
                    "outcome": "denied",
                    "reason": "tool_not_allowed",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
            if view is not None and allowed_tool_names is not None and name in allowed_tool_names and view.is_gated(name):
                return {
                    "status": "ok",
                    "engine_id": eid,
                    "toolbox_id": tid or self._registration_toolbox_id(reg),
                    "tool_name": name,
                    "outcome": "gated_requires_confirmation",
                    "reason": "gated_requires_confirmation",
                    "executable": False,
                    "requires_confirmation": True,
                    "backend": "sandbox",
                }
            if view is not None and allowed_tool_names is not None and name in allowed_tool_names and not view.is_allowed(name):
                return {
                    "status": "ok",
                    "engine_id": eid,
                    "toolbox_id": tid or self._registration_toolbox_id(reg),
                    "tool_name": name,
                    "outcome": "denied",
                    "reason": "blocked_in_scope",
                    "executable": False,
                    "requires_confirmation": False,
                    "backend": "sandbox",
                }
        result = {
            "status": "ok",
            "engine_id": eid,
            "toolbox_id": tid or self._registration_toolbox_id(reg),
            "tool_name": name,
            "outcome": "allowed",
            "reason": "allowed",
            "executable": True,
            "requires_confirmation": False,
            "backend": "sandbox",
        }
        return result

    def toolbox_execute(
        self,
        *,
        engine_id: str = "",
        toolbox_id: str = "",
        tool_call: Dict[str, Any],
        timeout_seconds: float = 30.0,
        tools_view: Optional[Dict[str, Any]] = None,
        callback_binding: Optional[Dict[str, Any]] = None,
        host_api_approval: Optional[Dict[str, Any]] = None,
        execution_request_id: str = "",
        owner_actor_id: str = "service:local",
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        call = dict(tool_call or {})
        tool_name = str(call.get("name") or "").strip()
        view = self._tools_view_from_payload(tools_view)
        if not tool_name:
            raise ValueError("tool_call.name is required")
        tid = str(toolbox_id or "").strip()
        if not eid and not tid:
            raise ValueError("engine_id or toolbox_id is required")
        if tid and not eid:
            gate = self.toolbox_gate(toolbox_id=tid, tool_name=tool_name, tools_view=tools_view)
            if str(gate.get("outcome") or "").strip().lower() != "allowed":
                reason = str(gate.get("reason") or gate.get("outcome") or "denied")
                raise PermissionError(f"{reason}:{tool_name}")
            reg = self._route_toolbox_registration(toolbox_id=tid, tool_name=tool_name, command_label="toolbox-execute")
            eid = str(reg.get("engine_id") or "").strip()
        else:
            reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-execute")
            allowed_tool_names = self._registration_allowed_tool_names(reg)
            if allowed_tool_names is not None and tool_name not in allowed_tool_names:
                raise PermissionError(f"tool_not_allowed:{tool_name}")
            if view is not None and allowed_tool_names is not None and tool_name in allowed_tool_names and view.is_gated(tool_name):
                raise PermissionError(f"gated_requires_confirmation:{tool_name}")
            if view is not None and allowed_tool_names is not None and tool_name in allowed_tool_names and not view.is_allowed(tool_name):
                raise PermissionError(f"blocked_in_scope:{tool_name}")
        environment_key = self._toolbox_registration_environment_key(reg)
        capacity = self._toolbox_registration_capacity(reg)
        toolbox_identity = tid or self._registration_toolbox_id(reg)
        concurrency = self._toolbox_tool_concurrency_policy(
            toolbox_id=toolbox_identity,
            tool_name=tool_name,
            call=call,
        )
        queue_config = self._toolbox_registration_queue_config(reg)
        model_tool_call_id = str(call.get("id") or call.get("tool_call_id") or "").strip() or f"tool-call-{uuid.uuid4().hex}"
        request_id = str(execution_request_id or "").strip()
        if not request_id:
            raise ValueError("execution_request_id is required for durable hosted execution")
        receipt_namespace = self._toolbox_operation_namespace(
            engine_id=eid if not tid else "",
            toolbox_id=tid,
        )
        selector = HostedOperationSelector(
            kind="toolbox_id" if tid else "engine_id",
            id=toolbox_identity if tid else eid,
        )
        fingerprint = hosted_execution_fingerprint(
            {
                "execution_kind": HostedExecutionKind.TOOLBOX.value,
                "configuration_revision": self.hosting_configuration_revision,
                "selector": selector.to_dict(),
                "tool": {
                    "name": tool_name,
                    "arguments": call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                },
                "policy": {
                    "tools_view": dict(tools_view or {}) if isinstance(tools_view, dict) else None,
                    "host_api_approval": dict(host_api_approval or {}) if isinstance(host_api_approval, dict) else None,
                    "sandbox_policy": dict(reg.get("sandbox_policy") or {}),
                    "sandbox_profile_id": self._registration_sandbox_profile_id(reg),
                    "concurrency": dict(concurrency),
                },
            }
        )
        prepared = self._hosted_operations.prepare(
            owner_actor_id=str(owner_actor_id or "service:local").strip() or "service:local",
            execution_kind=HostedExecutionKind.TOOLBOX,
            selector=selector,
            namespace=receipt_namespace,
            request_id=request_id,
            fingerprint=fingerprint,
            metadata={
                "configuration_revision": self.hosting_configuration_revision,
                "engine_id": eid,
                "toolbox_id": toolbox_identity,
                "tool_name": tool_name,
                "tool_call_id": model_tool_call_id,
                "environment_key": environment_key,
                "retain_terminal_result": bool(dict(reg.get("sandbox_policy") or {}).get("retain_terminal_result")),
            },
        )
        operation_action = str(prepared.get("action") or "")
        prepared_status = dict(prepared.get("status") or {})
        if operation_action in {"conflict", "forgotten", "replay"}:
            return prepared_status
        if operation_action == "capacity":
            raise RuntimeError("hosted_operation_capacity_exceeded")
        operation_id = str(dict(prepared_status.get("operation") or {}).get("operation_id") or "")
        if operation_action == "attach":
            # Duplicate submission is an observation, never a synchronous
            # attachment to the original worker. The caller can watch the
            # returned durable snapshot and retrieve its terminal result.
            return prepared_status
        base = self._toolbox_runtime_base()

        def _persist_terminal(envelope: Dict[str, Any], lifecycle: str) -> Dict[str, Any]:
            if tid:
                envelope = dict(envelope)
                for key in (
                    "engine_id",
                    "environment_key",
                    "worker_id",
                    "request",
                    "diagnostics",
                    "hosted_pool",
                    "toolbox_pool",
                    "profile_id",
                    "sandbox_profile_id",
                    "package_path",
                    "installer_output",
                ):
                    envelope.pop(key, None)
                outcome = str(envelope.get("outcome") or envelope.get("status") or "").strip().lower()
                if lifecycle == HostedOperationLifecycle.TERMINAL_SUCCESS.value:
                    state, code, summary = "succeeded", "toolbox_execution_succeeded", "The toolbox call succeeded."
                elif lifecycle == HostedOperationLifecycle.TERMINAL_CANCELLATION.value or outcome == "canceled":
                    state, code, summary = "canceled", "toolbox_execution_canceled", "The toolbox call was canceled."
                elif outcome == "timeout":
                    state, code, summary = "failed", "toolbox_execution_timeout", "The toolbox call timed out."
                else:
                    state, code, summary = "failed", "toolbox_execution_failed", "The toolbox call failed."
                envelope["user_projection"] = {"state": state, "code": code, "summary": summary}
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=lifecycle,
                envelope=envelope,
                reason=str(envelope.get("reason") or "").strip(),
            )

        # A bounded cached describe no longer serves as a readiness probe. For
        # a freshly routed toolbox call, wait for the concrete IPC endpoint
        # before dispatching the first call; duplicate submissions above never
        # enter this path. Direct engine-id callers retain the legacy behavior
        # because they explicitly selected and manage that executor.
        if tid and str(reg.get("worker_ipc_address") or "").strip():
            try:
                self._wait_for_toolbox_executor_ready(
                    eid,
                    # Honor the caller's already-bounded execution deadline.
                    # Cold CPython startup can exceed eight seconds on WSL and
                    # other constrained hosts even though the executor is healthy.
                    timeout_seconds=float(timeout_seconds or 30.0),
                )
            except Exception as exc:
                return _persist_terminal({
                    "status": "error",
                    "outcome": "error",
                    "reason": "toolbox_executor_not_ready",
                    "error": str(exc),
                    "engine_id": eid,
                    "toolbox_id": tid or self._registration_toolbox_id(reg),
                    "tool_name": tool_name,
                    "tool_call_id": model_tool_call_id,
                    "request_id": request_id,
                }, "terminal_failure")

        scheduled = base.submit_request(
            environment_key=environment_key,
            request_id=request_id,
            profile=self._registration_sandbox_profile_id(reg),
            factory=lambda _key, cap: self._toolbox_worker_slot(reg=reg, environment_key=environment_key, capacity=cap),
            desired_capacity=capacity,
            operation_id=tool_name,
            input_bytes=len(json.dumps(call, ensure_ascii=False).encode("utf-8", errors="replace")),
            queue_policy=str(queue_config.get("queue_policy") or "bounded"),
            queue_depth=int(queue_config.get("queue_depth") or 0),
            queue_timeout_seconds=float(queue_config.get("queue_timeout_seconds") or 0.0),
            concurrency=concurrency,
        )
        if str(scheduled.get("status") or "") != "ok":
            pool_snapshot = base.resources(environment_key)
            request_snapshot = dict(scheduled.get("request") or {})
            failure_reason = str(scheduled.get("reason") or "capacity_exceeded")
            return _persist_terminal({
                "status": "error",
                "outcome": "error",
                "reason": failure_reason,
                "error": failure_reason,
                "engine_id": eid,
                "toolbox_id": tid or self._registration_toolbox_id(reg),
                "tool_name": tool_name,
                "tool_call_id": model_tool_call_id,
                "request_id": request_id,
                "environment_key": environment_key,
                "worker_id": request_snapshot.get("worker_id"),
                "retry_count": 0,
                "admission": request_snapshot.get("admission") or "rejected",
                "concurrency": dict(concurrency),
                "request": request_snapshot,
                "diagnostics": {
                    "request": request_snapshot,
                    "concurrency": dict(concurrency),
                    "pool": pool_snapshot,
                },
                "hosted_pool": pool_snapshot,
            }, "terminal_failure")
        dispatch_claim = base.claim_dispatch(environment_key=environment_key, request_id=request_id)
        if str(dispatch_claim.get("status") or "") != "ok":
            request_snapshot = dict(dispatch_claim.get("request") or {})
            pool_snapshot = base.resources(environment_key)
            return _persist_terminal({
                "status": "error",
                "outcome": "canceled",
                "reason": str(request_snapshot.get("reason") or "canceled"),
                "error": str(request_snapshot.get("reason") or "canceled"),
                "engine_id": eid,
                "toolbox_id": tid or self._registration_toolbox_id(reg),
                "tool_name": tool_name,
                "tool_call_id": model_tool_call_id,
                "request_id": request_id,
                "environment_key": environment_key,
                "worker_id": request_snapshot.get("worker_id"),
                "retry_count": 0,
                "admission": request_snapshot.get("admission") or "canceled",
                "concurrency": dict(concurrency),
                "request": request_snapshot,
                "diagnostics": {
                    "request": request_snapshot,
                    "concurrency": dict(concurrency),
                    "pool": pool_snapshot,
                },
                "hosted_pool": pool_snapshot,
            }, "terminal_cancellation")
        finished = False
        dispatch_relay: Optional[_HostedToolCallbackRelay] = None
        dispatch_binding: Optional[Dict[str, Any]] = None
        try:
            dispatch_relay, dispatch_binding = self._toolbox_host_capability_dispatch_binding(
                engine_id=eid,
                toolbox_id=tid or self._registration_toolbox_id(reg),
                tool_name=tool_name,
                tool_call_id=model_tool_call_id,
                tool_arguments=call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                sandbox_policy=dict(reg.get("sandbox_policy") or {}),
                callback_binding=dict(callback_binding or {}) if isinstance(callback_binding, dict) else None,
                host_api_approval=dict(host_api_approval or {}) if isinstance(host_api_approval, dict) else None,
            )
            durable_dispatch = self._hosted_operations.mark_dispatch_claimed(operation_id=operation_id)
            if str(durable_dispatch.get("lifecycle") or "") != HostedOperationLifecycle.RUNNING.value:
                base.cancel_request(environment_key=environment_key, request_id=request_id)
                return durable_dispatch
            out = self._ipc_call(
                reg=reg,
                payload={
                    "kind": "rpc_call",
                    "engine_id": eid,
                    "method": "toolbox.execute",
                    "params": {
                        "tool_call": call,
                        "callback_binding": dict(dispatch_binding or {}) if isinstance(dispatch_binding, dict) else None,
                        "host_api_approval": dict(host_api_approval or {}) if isinstance(host_api_approval, dict) else None,
                    },
                },
                timeout_seconds=float(timeout_seconds or 30.0),
            )
            if str(out.get("status") or "").strip().lower() == "error":
                reason = str(out.get("message") or "toolbox_execute_failed")
                finish_out = base.finish_request(environment_key=environment_key, request_id=request_id, status="error", reason=reason)
                finished = True
                request_snapshot = dict(finish_out.get("request") or {})
                pool_snapshot = base.resources(environment_key)
                return _persist_terminal({
                    "status": "error",
                    "outcome": "error",
                    "reason": reason,
                    "error": str(out.get("message") or reason),
                    "engine_id": eid,
                    "toolbox_id": tid or self._registration_toolbox_id(reg),
                    "tool_name": tool_name,
                    "tool_call_id": model_tool_call_id,
                    "request_id": request_id,
                    "environment_key": environment_key,
                    "worker_id": request_snapshot.get("worker_id"),
                    "retry_count": 0,
                    "admission": request_snapshot.get("admission") or "admitted",
                    "concurrency": dict(concurrency),
                    "request": request_snapshot,
                    "diagnostics": {
                        "request": request_snapshot,
                        "concurrency": dict(concurrency),
                        "pool": pool_snapshot,
                    },
                    "hosted_pool": pool_snapshot,
                }, "terminal_failure")
            result = dict(out or {})
            output_bytes = len(json.dumps(result, ensure_ascii=False, default=str).encode("utf-8", errors="replace"))
            finish_out = base.finish_request(environment_key=environment_key, request_id=request_id, status="ok", output_bytes=output_bytes)
            finished = True
            request_snapshot = dict(finish_out.get("request") or {})
            pool_snapshot = base.resources(environment_key)
            result.setdefault("engine_id", eid)
            result.setdefault("toolbox_id", tid or self._registration_toolbox_id(reg))
            result.setdefault("tool_name", tool_name)
            result.setdefault("tool_call_id", model_tool_call_id)
            result.setdefault("request_id", request_id)
            result.setdefault("environment_key", environment_key)
            result.setdefault("worker_id", request_snapshot.get("worker_id"))
            result.setdefault("retry_count", 0)
            result.setdefault("admission", request_snapshot.get("admission") or "admitted")
            result.setdefault("concurrency", dict(concurrency))
            result.setdefault("request", request_snapshot)
            result.setdefault(
                "diagnostics",
                {
                    "request": request_snapshot,
                    "concurrency": dict(concurrency),
                    "pool": pool_snapshot,
                },
            )
            result.setdefault("hosted_pool", pool_snapshot)
            result.setdefault("toolbox_pool", result["hosted_pool"])
            return _persist_terminal(result, "terminal_success")
        except Exception as exc:
            reason = "toolbox_execute_timeout" if isinstance(exc, TimeoutError) else str(exc) or "toolbox_execute_failed"
            finish_status = "timeout" if isinstance(exc, TimeoutError) else "error"
            if not finished:
                finish_out = base.finish_request(
                    environment_key=environment_key,
                    request_id=request_id,
                    status=finish_status,
                    reason=reason,
                )
            else:
                finish_out = base.request_status(environment_key=environment_key, request_id=request_id)
            request_snapshot = dict(finish_out.get("request") or {})
            pool_snapshot = base.resources(environment_key)
            return _persist_terminal({
                "status": "timeout" if isinstance(exc, TimeoutError) else "error",
                "outcome": "timeout" if isinstance(exc, TimeoutError) else "error",
                "reason": reason,
                "error": str(exc) or reason,
                "error_type": type(exc).__name__,
                "engine_id": eid,
                "toolbox_id": tid or self._registration_toolbox_id(reg),
                "tool_name": tool_name,
                "tool_call_id": model_tool_call_id,
                "request_id": request_id,
                "environment_key": environment_key,
                "worker_id": request_snapshot.get("worker_id"),
                "retry_count": 0,
                "admission": request_snapshot.get("admission") or "admitted",
                "concurrency": dict(concurrency),
                "request": request_snapshot,
                "diagnostics": {
                    "request": request_snapshot,
                    "concurrency": dict(concurrency),
                    "pool": pool_snapshot,
                },
                "hosted_pool": pool_snapshot,
            }, "terminal_failure")
        finally:
            if dispatch_relay is not None and dispatch_binding:
                dispatch_relay.release_session(str(dispatch_binding.get("session_token") or ""))

    def _cancel_toolbox_operation(
        self,
        *,
        record: Dict[str, Any],
        reason: str = "client_requested",
        timeout_seconds: float = 8.0,
        respawn: bool = True,
    ) -> Dict[str, Any]:
        row = dict(record or {})
        operation = dict(row.get("operation") or {})
        metadata = dict(row.get("metadata") or {})
        operation_id = str(operation.get("operation_id") or "").strip()
        owner_actor_id = str(row.get("owner_actor_id") or "").strip()
        eid = str(metadata.get("engine_id") or "").strip()
        tid = str(metadata.get("toolbox_id") or "").strip()
        name = str(metadata.get("tool_name") or "").strip()
        model_tool_call_id = str(metadata.get("tool_call_id") or "").strip()
        call_id = str(operation.get("request_id") or "").strip()
        if not eid and not tid:
            raise ValueError("stored toolbox operation selector is invalid")
        lifecycle = HostedOperationLifecycle(str(row.get("lifecycle") or ""))
        if lifecycle in {
            HostedOperationLifecycle.QUEUED,
            HostedOperationLifecycle.INTERRUPTED_BEFORE_DISPATCH,
        }:
            canceled = self._hosted_operations.cancel_before_dispatch(
                operation_id=operation_id,
                reason=str(reason or "canceled_before_dispatch"),
            )
            if canceled is not None:
                return canceled
        if lifecycle in {
            HostedOperationLifecycle.TERMINAL_SUCCESS,
            HostedOperationLifecycle.TERMINAL_FAILURE,
            HostedOperationLifecycle.TERMINAL_CANCELLATION,
            HostedOperationLifecycle.INTERRUPTED_AFTER_DISPATCH_UNKNOWN,
            HostedOperationLifecycle.FORGOTTEN,
        }:
            return self._hosted_operations.status(ref=operation, owner_actor_id=owner_actor_id)

        def _cancel_failure(failure_reason: str) -> Dict[str, Any]:
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
                envelope={
                    "status": "error",
                    "outcome": "cancel_failed",
                    "reason": failure_reason,
                    "engine_id": eid or None,
                    "toolbox_id": tid or None,
                    "tool_name": name or None,
                    "request_id": call_id,
                },
                reason=failure_reason,
            )

        target_regs: List[Dict[str, Any]] = []
        if eid:
            reg = dict(self._find_registration(eid) or {})
            if not reg:
                return _cancel_failure("engine_not_found")
            target_regs = [reg]
        elif tid:
            if name:
                try:
                    target_regs = [self._route_toolbox_registration(toolbox_id=tid, tool_name=name, command_label="hosted-operation-cancel")]
                except PermissionError:
                    return _cancel_failure("tool_not_allowed")
            else:
                target_regs = list(self._toolbox_executor_registrations(tid))
            if not target_regs:
                return _cancel_failure("toolbox_executor_missing")

        canceled_engine_ids: List[str] = []
        failed_engine_ids: List[str] = []
        shutdown_results: Dict[str, Dict[str, Any]] = {}
        hosted_pool_cancels: Dict[str, Dict[str, Any]] = {}
        sandbox_recycled_request_ids: Dict[str, List[str]] = {}
        canceled_request_ids: Dict[str, List[str]] = {}
        target_toolbox_ids: set[str] = set()
        base = self._toolbox_runtime_base()
        for reg in target_regs:
            target_engine_id = str(dict(reg or {}).get("engine_id") or "").strip()
            if not target_engine_id:
                continue
            environment_key = self._toolbox_registration_environment_key(dict(reg or {}))
            pool = base.pool_registry.get(base.pool_key(environment_key)) if environment_key else None
            sibling_request_ids: List[str] = []
            target_request_status: Dict[str, Any] = {}
            if pool is not None and call_id:
                target_request_status = dict(pool.request_status(call_id).get("request") or {})
                for worker in list(pool.workers or []):
                    if str(worker.engine_id or "").strip() != target_engine_id:
                        continue
                    sibling_request_ids.extend(
                        [
                            str(active_request_id or "").strip()
                            for active_request_id in list(worker.active_request_ids or [])
                            if str(active_request_id or "").strip() and str(active_request_id or "").strip() != call_id
                        ]
                    )
            sibling_request_ids = sorted(set(sibling_request_ids))
            if environment_key and call_id:
                hosted_pool_cancels[target_engine_id] = dict(base.cancel_request(environment_key=environment_key, request_id=call_id))
                if str(hosted_pool_cancels[target_engine_id].get("status") or "") == "ok":
                    canceled_request_ids[target_engine_id] = [call_id]
            elif environment_key:
                canceled_requests: List[str] = []
                if pool is not None:
                    for worker in list(pool.workers or []):
                        if str(worker.engine_id or "").strip() != target_engine_id:
                            continue
                        for active_request_id in list(worker.active_request_ids or []):
                            cancel_out = dict(base.cancel_request(environment_key=environment_key, request_id=str(active_request_id or "")))
                            if str(cancel_out.get("status") or "") == "ok":
                                canceled_requests.append(str(active_request_id or ""))
                hosted_pool_cancels[target_engine_id] = {
                    "status": "ok" if canceled_requests else "not_found",
                    "environment_key": environment_key,
                    "canceled_request_ids": canceled_requests,
                }
                if canceled_requests:
                    canceled_request_ids[target_engine_id] = sorted(set(canceled_requests))
            reg_toolbox_id = self._registration_toolbox_id(dict(reg or {}))
            if reg_toolbox_id:
                target_toolbox_ids.add(reg_toolbox_id)
            queued_call_canceled = bool(
                call_id
                and str(target_request_status.get("status") or "") == "queued"
                and str(hosted_pool_cancels.get(target_engine_id, {}).get("status") or "") == "ok"
            )
            if queued_call_canceled:
                shutdown_out = {
                    "status": "ok",
                    "alive": False,
                    "skipped": True,
                    "reason": "queued_request_canceled",
                }
            else:
                shutdown_out = dict(self.shutdown(target_engine_id, timeout_seconds=float(timeout_seconds or 8.0)) or {})
            shutdown_results[target_engine_id] = shutdown_out
            if bool(shutdown_out.get("alive")) or str(shutdown_out.get("status") or "").strip() == "stop_failed":
                failed_engine_ids.append(target_engine_id)
            else:
                if not queued_call_canceled:
                    canceled_engine_ids.append(target_engine_id)
            if not queued_call_canceled and sibling_request_ids:
                recycled: List[str] = []
                for sibling_request_id in sibling_request_ids:
                    finished = base.finish_request(
                        environment_key=environment_key,
                        request_id=sibling_request_id,
                        status="error",
                        reason="sandbox_recycled",
                    )
                    if str(finished.get("status") or "") == "ok":
                        recycled.append(sibling_request_id)
                if recycled:
                    sandbox_recycled_request_ids[target_engine_id] = recycled

        repair_out: Dict[str, Any] = {}
        repaired_toolbox_ids: List[str] = []
        for target_toolbox_id in sorted(target_toolbox_ids):
            with self._locked_toolbox(target_toolbox_id):
                if respawn:
                    repair_piece = dict(
                        self._toolbox_repair_now(
                            toolbox_ids=[target_toolbox_id],
                            only_inconsistent=False,
                            details=False,
                        )
                        or {}
                    )
                    if repair_piece:
                        repaired_toolbox_ids.extend(
                            [
                                str(item or "").strip()
                                for item in list(repair_piece.get("repaired_toolbox_ids") or [])
                                if str(item or "").strip()
                            ]
                        )
                        if not repair_out:
                            repair_out = dict(repair_piece)
                        else:
                            repair_out.setdefault("repaired_toolbox_ids", [])
                            repair_out["repaired_toolbox_ids"] = sorted(
                                {
                                    *list(repair_out.get("repaired_toolbox_ids") or []),
                                    *list(repair_piece.get("repaired_toolbox_ids") or []),
                                }
                            )

        result = {
            "status": "ok",
            "engine_id": eid or None,
            "toolbox_id": tid or None,
            "tool_name": name or None,
            "tool_call_id": model_tool_call_id or None,
            "request_id": call_id or None,
            "respawn": bool(respawn),
            "outcome": (
                "canceled_and_repaired"
                if (canceled_engine_ids or canceled_request_ids) and repaired_toolbox_ids
                else "canceled"
                if canceled_engine_ids or canceled_request_ids
                else "noop"
                if not failed_engine_ids
                else "partial_failure"
            ),
            "canceled_engine_ids": sorted(canceled_engine_ids),
            "failed_engine_ids": sorted(failed_engine_ids),
            "canceled_request_ids": {key: sorted(value) for key, value in sorted(canceled_request_ids.items())},
            "sandbox_recycled_request_ids": {
                key: sorted(value) for key, value in sorted(sandbox_recycled_request_ids.items())
            },
            "repaired_toolbox_ids": sorted(repaired_toolbox_ids),
            "shutdown_results": shutdown_results,
            "hosted_pool_cancels": hosted_pool_cancels,
            "repair": repair_out,
        }
        if canceled_engine_ids or canceled_request_ids:
            return self._hosted_operations.finish(
                operation_id=operation_id,
                lifecycle=HostedOperationLifecycle.TERMINAL_CANCELLATION,
                envelope=result,
                reason=str(reason or "client_requested"),
            )
        failure_reason = "cancel_partial_failure" if failed_engine_ids else "cancel_target_not_active"
        return self._hosted_operations.finish(
            operation_id=operation_id,
            lifecycle=HostedOperationLifecycle.TERMINAL_FAILURE,
            envelope=result,
            reason=failure_reason,
        )

    def _wait_for_toolbox_executor_ready(
        self,
        engine_id: str,
        *,
        timeout_seconds: float = 8.0,
        poll_seconds: float = 0.1,
    ) -> Dict[str, Any]:
        eid = str(engine_id or "").strip()
        if not eid:
            raise ValueError("engine_id is required")
        try:
            reg = self._require_toolbox_executor_registration(eid, command_label="toolbox-ready")
        except Exception as exc:
            raise ToolboxRolloutError(
                f"toolbox_executor_missing:{eid}",
                code="toolbox_executor_missing",
                details={
                    "failure_phase": "spawned",
                    "engine_id": eid,
                    "reason": str(exc),
                },
            ) from exc
        deadline = time.time() + max(0.1, float(timeout_seconds or 8.0))
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            try:
                desc = self._toolbox_describe_live(engine_id=eid, timeout_seconds=min(2.0, max(0.2, float(timeout_seconds or 8.0))))
                allowed = self._registration_allowed_tool_names(reg)
                reported = {
                    str(item or "").strip()
                    for item in list(
                        dict(desc or {}).get("all_registered_tool_names")
                        or []
                    )
                    if str(item or "").strip()
                }
                if allowed is not None and reported != allowed:
                    raise ToolboxRolloutError(
                        f"toolbox_executor_inventory_mismatch:{eid}",
                        code="toolbox_executor_inventory_mismatch",
                        details={
                            "failure_phase": "inventory_verified",
                            "engine_id": eid,
                            "expected_tool_names": sorted(allowed),
                            "actual_tool_names": sorted(reported),
                        },
                    )
                return desc
            except Exception as exc:
                last_error = exc
                time.sleep(max(0.05, float(poll_seconds or 0.1)))
        if isinstance(last_error, ToolboxRolloutError):
            raise ToolboxRolloutError(
                str(last_error),
                code=last_error.code,
                details=dict(last_error.details or {}),
            ) from last_error
        raise ToolboxRolloutError(
            f"toolbox_executor_not_ready:{eid}:{last_error}",
            code="toolbox_executor_not_ready",
            details={
                "failure_phase": "ready",
                "engine_id": eid,
                "timeout_seconds": float(timeout_seconds or 8.0),
                "reason": str(last_error or ""),
            },
        )

    def _ensure_toolbox_assignments_ready(
        self,
        assignments: List[Any],
        *,
        timeout_seconds: float = 8.0,
    ) -> Dict[str, Dict[str, Any]]:
        from ..toolbox_harness import EnvironmentRuntimeAdapter, ToolboxEnvironmentSpec

        ready: Dict[str, Dict[str, Any]] = {}
        environment_manager = EnvironmentRuntimeAdapter(self.hosting_root)
        for item in list(assignments or []):
            reg = dict(getattr(item, "registration", None) or {})
            engine_id = str(reg.get("engine_id") or "").strip()
            if not engine_id:
                continue
            started_at = time.time()
            try:
                desc = self._wait_for_toolbox_executor_ready(engine_id, timeout_seconds=timeout_seconds)
            except ToolboxRolloutError as exc:
                bundle = dict(reg.get("bundle") or {})
                details = dict(exc.details or {})
                details.setdefault("toolbox_id", str(bundle.get("toolbox_id") or getattr(item, "toolbox_id", "") or ""))
                details.setdefault(
                    "sandbox_profile_id",
                    str(bundle.get("sandbox_profile_id") or getattr(getattr(item, "sandbox_profile", None), "normalized_profile_id", lambda: "")() or ""),
                )
                details.setdefault("bundle_revision", str(bundle.get("bundle_revision") or ""))
                details.setdefault("engine_id", engine_id)
                raise ToolboxRolloutError(str(exc), code=exc.code, details=details) from exc
            ready_at = time.time()
            tool_names = [
                str(name or "").strip()
                for name in list(
                    dict(desc or {}).get("all_registered_tool_names")
                    or []
                )
                if str(name or "").strip()
            ]
            environment = dict(reg.get("environment") or {})
            receipt_verification_status = None
            install_execution_status = None
            if str(reg.get("routing_state") or "") == "candidate":
                expected_names = [
                    str(name or "").strip()
                    for name in list(dict(reg.get("tool_access") or {}).get("allowed_tool_names") or [])
                    if str(name or "").strip()
                ]
                if len(tool_names) != len(set(tool_names)) or set(tool_names) != set(expected_names):
                    raise ToolboxRolloutError(
                        f"toolbox inventory mismatch for {engine_id}",
                        code="toolbox_candidate_inventory_mismatch",
                        details={"engine_id": engine_id, "failure_phase": "inventory"},
                    )
                bundle = dict(reg.get("bundle") or {})
                profile = getattr(item, "resolved_profile", None)
                expected_profile_id = str(getattr(profile, "profile_id", "") or "")
                expected_environment_key = str(getattr(profile, "environment_key", "") or "")
                if (
                    not expected_profile_id
                    or bundle.get("resolved_profile_id") != expected_profile_id
                    or environment.get("environment_key") != expected_environment_key
                    or environment.get("verification_state") != "verified"
                    or environment.get("verification_receipt_contract")
                    != "hosting.environment_receipt.v1"
                ):
                    raise ToolboxRolloutError(
                        f"toolbox candidate metadata mismatch for {engine_id}",
                        code="toolbox_candidate_metadata_mismatch",
                        details={"engine_id": engine_id, "failure_phase": "metadata"},
                    )
                try:
                    receipt = self._environment_manager.receipt(
                        environment_id=expected_environment_key
                    )
                except Exception as exc:
                    raise ToolboxRolloutError(
                        f"toolbox environment receipt unavailable for {engine_id}",
                        code="toolbox_environment_receipt_unverified",
                        details={"engine_id": engine_id, "failure_phase": "environment_receipt"},
                    ) from exc
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("contract") != "hosting.environment_receipt.v1"
                    or receipt.get("environment_id") != expected_environment_key
                    or receipt.get("configuration_revision")
                    != self.hosting_configuration_revision
                ):
                    raise ToolboxRolloutError(
                        f"toolbox environment receipt mismatch for {engine_id}",
                        code="toolbox_environment_receipt_unverified",
                        details={"engine_id": engine_id, "failure_phase": "environment_receipt"},
                    )
                receipt_verification_status = "ok"
                install_execution_status = "ok"
            elif environment:
                spec = ToolboxEnvironmentSpec.from_dict(environment)
                metadata = environment_manager.read_environment_metadata(spec)
                install_execution_status = str(dict(metadata.get("install_execution") or {}).get("status") or "").strip() or None
                receipt_verification_status = str(
                    dict(metadata.get("install_receipt_verification") or {}).get("status") or ""
                ).strip() or None
                if install_execution_status == "ok" and receipt_verification_status != "ok":
                    raise ToolboxRolloutError(
                        f"environment receipt verification not ready for {engine_id}",
                        code="toolbox_environment_receipt_unverified",
                        details={
                            "engine_id": engine_id,
                            "install_execution_status": install_execution_status,
                            "install_receipt_verification_status": receipt_verification_status,
                            "toolbox_id": str(dict(reg.get("bundle") or {}).get("toolbox_id") or getattr(item, "toolbox_id", "") or ""),
                            "sandbox_profile_id": str(
                                dict(reg.get("bundle") or {}).get("sandbox_profile_id")
                                or getattr(getattr(item, "sandbox_profile", None), "normalized_profile_id", lambda: "")()
                                or ""
                            ),
                        },
                    )
            ready[engine_id] = {
                "engine_id": engine_id,
                "ready": True,
                "ready_at": ready_at,
                "warmup_ms": int(max(0.0, (ready_at - started_at) * 1000.0)),
                "tool_inventory_ok": True,
                "tool_count": len(tool_names),
                "all_registered_tool_names": tool_names,
                "install_execution_status": install_execution_status,
                "install_receipt_verification_status": receipt_verification_status,
            }
        return ready
