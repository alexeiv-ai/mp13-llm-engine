"""
Standalone engine host service logic.

This module is intentionally backend-agnostic: it only manages engine-host
process lifecycle and generic control-plane state (claims/tokens/resources).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .auth import AuthMixin
from .claims import ClaimsMixin
from .constants import (
    DEFAULT_ENGINES_STATE_FILE,
    VALID_AUTH_ROLES,
)
from ..hosting_configuration import HostingConfiguration
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
from .package_api import PackageApiMixin
from .proxy import ProxyMixin
from .sandbox_api import SandboxApiMixin
from .state import StateMixin
from .toolbox_env import ToolboxMaintenanceMixin
from .toolbox_catalog import ToolboxTemplateCatalogMixin
from .toolbox_materialization import (
    HermeticToolboxTemplateMaterializer,
    ToolboxTemplateMaterializer,
    UnconfiguredToolboxTemplateMaterializer,
)
from .toolbox_runtime import ToolboxRuntimeMixin
from .toolbox_state_v2 import AtomicJsonToolboxStateV2Repository
from .toolbox_plans import AtomicJsonCompleteToolboxDefinitionPlanRepository
from .toolbox_host_config_state import AtomicJsonToolboxHostConfigurationRepository
from .toolbox_artifact_store import AtomicToolboxArtifactStore
from .toolbox_approvals import AtomicJsonToolboxDependencyApprovalRepository
from .toolbox_confirmations import AtomicJsonToolboxConfirmationRepository
from ..toolbox.target import detect_current_toolbox_target
from .workflow_helpers import WorkflowHelperMixin


class EngineHostService(CoreMixin, MetricsMixin, StateMixin, ConfigMixin, ControlMixin, AuthMixin, ClaimsMixin, PolicyMixin, PackageApiMixin, EnginesMixin, ProxyMixin, SandboxApiMixin, LogsMixin, ToolboxMaintenanceMixin, ToolboxTemplateCatalogMixin, ToolboxRuntimeMixin, WorkflowHelperMixin, HostedOperationsMixin, ModelRuntimeMixin):
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
        hosting_configuration: HostingConfiguration,
        operation_retention_seconds: Optional[float] = None,
        operation_tombstone_seconds: Optional[float] = None,
        operation_max_count: Optional[int] = None,
        operation_max_tombstones: Optional[int] = None,
        operation_max_inline_result_bytes: Optional[int] = None,
        toolbox_template_materializer: Optional[ToolboxTemplateMaterializer] = None,
        toolbox_required_python_abi: Optional[str] = None,
        toolbox_required_platform: Optional[str] = None,
        model_runtime_identity: Optional[Dict[str, Any] | ModelRuntimeIdentity] = None,
    ):
        if not isinstance(hosting_configuration, HostingConfiguration):
            raise TypeError("hosting_configuration_required")
        self.hosting_configuration = hosting_configuration
        self.hosting_configuration_revision = hosting_configuration.revision
        self.hosting_root = Path(hosting_configuration.resolved_paths["scratch_root"]).parent.resolve()
        self.control_state_file = self.hosting_root / "state" / "control_state.json"
        self.engines_state_file = (
            engines_state_file or self.hosting_root / "state" / DEFAULT_ENGINES_STATE_FILE.name
        ).expanduser().resolve()
        self._runtime_engines_lock = threading.RLock()
        self._runtime_engines: list[Dict[str, Any]] = []
        self._toolbox_artifact_sources: Dict[str, Path] = {}
        self._toolbox_host_project_config = None
        self._toolbox_trust_public_keys = None
        self._toolbox_source_credentials: Dict[str, str] = {}
        current_target = detect_current_toolbox_target()
        configured_abi = ""
        configured_platform = ""
        self._hermetic_toolbox_environment_builder = None
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
        self._toolbox_definition_plans = AtomicJsonCompleteToolboxDefinitionPlanRepository(
            self.hosting_root / "state" / "toolbox_definition_plans.json"
        )
        self._toolbox_host_config_revisions = AtomicJsonToolboxHostConfigurationRepository(
            self.hosting_root / "state" / "toolbox_host_configurations.json"
        )
        self._toolbox_dependency_approvals = AtomicJsonToolboxDependencyApprovalRepository(
            self.hosting_root / "state" / "toolbox_dependency_approvals.json"
        )
        self._toolbox_confirmations = AtomicJsonToolboxConfirmationRepository(
            self.hosting_root / "state" / "toolbox_definition_confirmations.json"
        )
        self._configured_toolbox_dependency_policy = None
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
        self._toolbox_setup_operation = None
        self._toolbox_config_transition = None
        self._toolbox_artifact_store = AtomicToolboxArtifactStore(
            self.hosting_root / "toolbox_artifact_store"
        )
        self._toolbox_verified_artifacts: dict[str, dict[str, Path]] = {}
        self._toolbox_artifact_ingestion_diagnostic = None
        if self._toolbox_host_project_config is not None:
            self._toolbox_config_transition = self._toolbox_host_config_revisions.apply(
                self._toolbox_host_project_config
            )
            if self._toolbox_config_transition["changed"]:
                active_digests = set(self._toolbox_template_catalog.read()["active"].values())
                self._toolbox_config_transition["invalidated_plans"] = (
                    self._toolbox_definition_plans.invalidate_all()
                )
                self._toolbox_config_transition["invalidated_materialization_receipts"] = (
                    self._toolbox_materialization_receipts.retain_template_digests(active_digests)
                )
            self._toolbox_startup = {
                "status": "pending",
                "config_revision": self._toolbox_host_project_config.config_revision,
                "source_set_revision": self._toolbox_host_project_config.source_set_revision,
                "target": self._toolbox_host_project_config.target.name,
                "closures": [],
                "diagnostics": [],
                "published": [],
                "operations": [],
            }

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
        for attribute in (
            "_workflow_python_node_runtime_registry_instance",
            "_workflow_js_node_runtime_registry_instance",
        ):
            node_registry = getattr(self, attribute, None)
            if node_registry is None:
                continue
            try:
                node_registry.shutdown()
            except Exception:
                pass
            try:
                setattr(self, attribute, None)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
