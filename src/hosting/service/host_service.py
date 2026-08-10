"""
Standalone engine host service logic.

This module is intentionally backend-agnostic: it only manages engine-host
process lifecycle and generic control-plane state (claims/tokens/resources).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .auth import AuthMixin
from .claims import ClaimsMixin
from .constants import (
    DEFAULT_CONTROL_STATE_FILE,
    DEFAULT_ENGINES_STATE_FILE,
    VALID_AUTH_ROLES,
)
from .configs import ConfigMixin
from .control import ControlMixin
from .core import CoreMixin
from .engines import EnginesMixin
from .errors import ToolboxRolloutError
from .logs import LogsMixin
from .metrics import MetricsMixin
from .model_runtime import ModelRuntimeMixin
from ..model_runtime_contract import ModelRuntimeIdentity
from .hosted_operations import HostedOperationsMixin
from .operation_repository import AtomicJsonHostedOperationRepository, LegacyOperationRepositoryError
from .result_artifacts import TerminalResultArtifactStore
from .policy import PolicyMixin
from .proxy import ProxyMixin
from .sandbox_api import SandboxApiMixin
from .state import StateMixin
from .toolbox_env import ToolboxMaintenanceMixin
from .toolbox_catalog import ToolboxTemplateCatalogMixin
from ..toolbox.hermetic_environment import HermeticToolboxEnvironmentBuilder
from .toolbox_materialization import (
    HermeticToolboxTemplateMaterializer,
    ToolboxTemplateMaterializer,
    UnconfiguredToolboxTemplateMaterializer,
)
from .toolbox_runtime import ToolboxRuntimeMixin
from .toolbox_state_v2 import AtomicJsonToolboxStateV2Repository
from .toolbox_plans import AtomicJsonToolboxDefinitionPlanRepository
from .toolbox_approvals import AtomicJsonToolboxDependencyApprovalRepository
from ..toolbox.dependency_policy import ToolboxDependencyPolicy
from ..toolbox.host_project_config import (
    ToolboxHostProjectConfiguration,
)
from ..toolbox.target import detect_current_toolbox_target
from .workflow_helpers import WorkflowHelperMixin


class EngineHostService(CoreMixin, MetricsMixin, StateMixin, ConfigMixin, ControlMixin, AuthMixin, ClaimsMixin, PolicyMixin, EnginesMixin, ProxyMixin, SandboxApiMixin, LogsMixin, ToolboxMaintenanceMixin, ToolboxTemplateCatalogMixin, ToolboxRuntimeMixin, WorkflowHelperMixin, HostedOperationsMixin, ModelRuntimeMixin):
    """Engine host service for terminal-command control."""
    _metrics_lock = threading.Lock()
    _runtime_metrics: Optional[Dict[str, Any]] = None
    _toolbox_lock_guard = threading.Lock()
    _toolbox_locks: Dict[str, threading.RLock] = {}
    _operation_repository_guard = threading.Lock()
    _operation_repositories: Dict[str, AtomicJsonHostedOperationRepository] = {}

    def __init__(
        self,
        *,
        engines_state_file: Optional[Path] = None,
        control_state_file: Optional[Path] = None,
        operation_retention_seconds: Optional[float] = None,
        operation_tombstone_seconds: Optional[float] = None,
        operation_max_count: Optional[int] = None,
        operation_max_tombstones: Optional[int] = None,
        operation_max_inline_result_bytes: Optional[int] = None,
        toolbox_template_materializer: Optional[ToolboxTemplateMaterializer] = None,
        toolbox_artifact_sources: Optional[Dict[str, Path]] = None,
        toolbox_required_python_abi: Optional[str] = None,
        toolbox_required_platform: Optional[str] = None,
        toolbox_host_project_configuration: Optional[Mapping[str, Any]] = None,
        model_runtime_identity: Optional[Dict[str, Any] | ModelRuntimeIdentity] = None,
        toolbox_dependency_policy: Optional[Dict[str, Any] | ToolboxDependencyPolicy] = None,
    ):
        self.engines_state_file = (engines_state_file or DEFAULT_ENGINES_STATE_FILE).expanduser().resolve()
        raw_control = (control_state_file or DEFAULT_CONTROL_STATE_FILE).expanduser().resolve()
        if raw_control.suffix:
            self.hosting_root = raw_control.parent.resolve()
            self.control_state_file = self.hosting_root / "access_control.json"
        else:
            self.hosting_root = raw_control.resolve()
            self.control_state_file = self.hosting_root / "access_control.json"
        self._runtime_engines_lock = threading.RLock()
        self._runtime_engines: list[Dict[str, Any]] = []
        self._toolbox_artifact_sources = {
            str(source_id): Path(path).expanduser().resolve()
            for source_id, path in dict(toolbox_artifact_sources or {}).items()
        }
        self._toolbox_host_project_config = (
            ToolboxHostProjectConfiguration.from_dict(toolbox_host_project_configuration)
            if toolbox_host_project_configuration is not None
            else None
        )
        current_target = detect_current_toolbox_target()
        configured_abi = ""
        configured_platform = ""
        if self._toolbox_host_project_config is not None:
            configured_abi = self._toolbox_host_project_config.target.python_abi
            configured_platform = self._toolbox_host_project_config.target.platform
            if toolbox_required_python_abi and toolbox_required_python_abi != configured_abi:
                raise ValueError("toolbox_required_python_abi_conflict")
            if toolbox_required_platform and toolbox_required_platform != configured_platform:
                raise ValueError("toolbox_required_platform_conflict")
            if toolbox_artifact_sources is not None:
                configured_sources = {
                    item.source_id
                    for item in self._toolbox_host_project_config.sources
                    if item.kind == "airgap_store"
                }
                if not configured_sources.issubset(set(toolbox_artifact_sources)):
                    raise ValueError("toolbox_artifact_sources_incomplete")
        if toolbox_template_materializer is not None and toolbox_artifact_sources is not None:
            raise ValueError("toolbox_materializer_configuration_conflict")
        self._hermetic_toolbox_environment_builder = (
            HermeticToolboxEnvironmentBuilder(
                self.hosting_root,
                artifact_sources=self._toolbox_artifact_sources,
                gc_grace_ms=(
                    self._toolbox_host_project_config.retention.artifact_cache_grace_seconds * 1000
                    if self._toolbox_host_project_config is not None
                    else 24 * 60 * 60 * 1000
                ),
                build_timeout_seconds=(
                    self._toolbox_host_project_config.resolution.timeout_seconds
                    if self._toolbox_host_project_config is not None
                    else 300
                ),
            )
            if toolbox_artifact_sources is not None
            else None
        )
        if toolbox_template_materializer is not None:
            self._toolbox_template_materializer = toolbox_template_materializer
        elif self._hermetic_toolbox_environment_builder is not None:
            self._toolbox_template_materializer = HermeticToolboxTemplateMaterializer(
                self._hermetic_toolbox_environment_builder
            )
        else:
            self._toolbox_template_materializer = UnconfiguredToolboxTemplateMaterializer()
        selected_abi = str(configured_abi or toolbox_required_python_abi or current_target.python_abi).strip()
        selected_platform = str(
            configured_platform or toolbox_required_platform or current_target.platform
        ).strip()
        if selected_abi != current_target.python_abi or selected_platform != current_target.platform:
            raise ValueError("toolbox_required_target_cross_target")
        self._toolbox_target = current_target
        self._toolbox_required_python_abi = selected_abi
        self._toolbox_required_platform = selected_platform
        self._toolbox_state_v2 = AtomicJsonToolboxStateV2Repository(
            self.hosting_root / "state" / "toolbox_sandboxes_v2.json",
            legacy_path=self.hosting_root / "state" / "toolbox_sandboxes.json",
        )
        self._toolbox_definition_plans = AtomicJsonToolboxDefinitionPlanRepository(
            self.hosting_root / "state" / "toolbox_definition_plans.json"
        )
        self._toolbox_dependency_approvals = AtomicJsonToolboxDependencyApprovalRepository(
            self.hosting_root / "state" / "toolbox_dependency_approvals.json"
        )
        self._configured_toolbox_dependency_policy = (
            toolbox_dependency_policy
            if isinstance(toolbox_dependency_policy, ToolboxDependencyPolicy)
            else ToolboxDependencyPolicy.from_dict(toolbox_dependency_policy)
            if toolbox_dependency_policy is not None
            else None
        )
        self._model_runtime_identity = (
            model_runtime_identity
            if isinstance(model_runtime_identity, ModelRuntimeIdentity)
            else ModelRuntimeIdentity.from_dict(model_runtime_identity)
            if model_runtime_identity is not None
            else None
        )

        def _float_setting(value: Optional[float], env_name: str, default: float) -> float:
            if value is not None:
                return float(value)
            try:
                return float(os.environ.get(env_name, default))
            except (TypeError, ValueError):
                return default

        def _int_setting(value: Optional[int], env_name: str, default: int) -> int:
            if value is not None:
                return int(value)
            try:
                return int(os.environ.get(env_name, default))
            except (TypeError, ValueError):
                return default

        self._hosted_operation_options = {
            "receipt_retention_seconds": _float_setting(
                operation_retention_seconds,
                "MP13_HOSTED_OPERATION_RETENTION_SECONDS",
                7 * 24 * 3600,
            ),
            "tombstone_retention_seconds": _float_setting(
                operation_tombstone_seconds,
                "MP13_HOSTED_OPERATION_TOMBSTONE_SECONDS",
                14 * 24 * 3600,
            ),
            "max_receipts": _int_setting(operation_max_count, "MP13_HOSTED_OPERATION_MAX_COUNT", 10_000),
            "max_tombstones": _int_setting(
                operation_max_tombstones,
                "MP13_HOSTED_OPERATION_MAX_TOMBSTONES",
                20_000,
            ),
            "max_inline_result_bytes": _int_setting(
                operation_max_inline_result_bytes,
                "MP13_HOSTED_OPERATION_MAX_INLINE_RESULT_BYTES",
                64 * 1024,
            ),
        }
        self._ensure_metrics_initialized()
        self._toolbox_startup = None
        if self._toolbox_host_project_config is not None:
            self._toolbox_startup = self.initialize_configured_toolbox_templates(
                configuration=self._toolbox_host_project_config,
                request_id_prefix=f"host-startup-{self._toolbox_host_project_config.config_revision}",
            )

    @property
    def _hosted_operations(self) -> AtomicJsonHostedOperationRepository:
        state_root = (self.hosting_root / "state").resolve()
        path = (state_root / "hosted_operations.json").resolve()
        legacy_path = (state_root / "toolbox_execution_receipts.json").resolve()
        if legacy_path.exists() and not path.exists():
            raise LegacyOperationRepositoryError(
                "legacy hosted-operation receipt schema is unsupported; "
                "run hosting-receipt-ledger-cutover after confirming the replay window is clear"
            )
        key = str(path)
        with self._operation_repository_guard:
            repository = self._operation_repositories.get(key)
            if repository is None:
                artifact_store = TerminalResultArtifactStore(
                    state_root / "hosted_operation_results",
                    ttl_seconds=float(self._hosted_operation_options["receipt_retention_seconds"]),
                )
                repository = AtomicJsonHostedOperationRepository(
                    path,
                    result_artifact_store=artifact_store,
                    **self._hosted_operation_options,
                )
                self._operation_repositories[key] = repository
            return repository

    def close(self) -> None:
        node_registry = getattr(self, "_workflow_python_node_runtime_registry_instance", None)
        if node_registry is not None:
            try:
                node_registry.shutdown()
            except Exception:
                pass
            try:
                setattr(self, "_workflow_python_node_runtime_registry_instance", None)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
