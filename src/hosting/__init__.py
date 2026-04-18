"""
mp13 engine hosting package.

Public API — stdlib only at module level, no heavy imports (torch/transformers etc.).

Key classes and functions available for import:
    EngineHostService         — file-backed process lifecycle and control-plane state
    EngineHostDaemon          — local IPC daemon server for control commands
    DaemonPidFile             — read/write/probe the daemon PID file
    DEFAULT_DAEMON_PORT       — legacy compatibility port metadata (19876)
    run_daemon_foreground()   — start daemon blocking in foreground
    start_daemon_background() — spawn daemon as detached background process
    LocalSocketConnection     — persistent local daemon connection (IPC with legacy TCP fallback)
    SSHRelayConnection        — persistent SSH subprocess running --relay
    BaseConnection            — abstract base for connection strategies
    ConnectionError           — raised on unrecoverable connection failure
    EngineHostControlChannel  — command channel with daemon connection + subprocess fallback
    EngineProcessSupervisor   — in-process persisted tracker for managed worker processes
    WorkerSandboxPolicy       — worker sandbox policy schema
"""
from __future__ import annotations

from .engine_host_service import EngineHostService
from .engine_host_daemon import (
    EngineHostDaemon,
    DaemonPidFile,
    DEFAULT_DAEMON_PORT,
    run_daemon_foreground,
    start_daemon_background,
)
from .engine_host_connection import (
    BaseConnection,
    LocalSocketConnection,
    SSHRelayConnection,
    ConnectionError,
)
from .engine_host_channel import EngineHostControlChannel
from .engine_process_supervisor import EngineProcessSupervisor
from .client_realm import (
    CLIENT_REALM_ROOT_SUBDIR,
    VALID_SECRET_RECORD_ENCRYPTION,
    SecretRecord,
    FileSecretStore,
    get_default_client_realm_root,
    client_realm_layout,
    ensure_client_realm_dirs,
    secret_record_path,
    managed_key_path,
    write_client_access,
    read_client_access,
    write_client_profile,
    read_client_profile,
    list_client_profiles,
    iter_secret_ids,
    append_client_audit_event,
    list_client_audit_events,
    materialize_secret_file,
    resolve_client_profile_control_settings,
)
from .transport_bootstrap import (
    TRANSPORT_BOOTSTRAP_KIND,
    make_transport_bootstrap_bundle,
    validate_transport_bootstrap_bundle,
    write_transport_bootstrap_bundle,
    read_transport_bootstrap_bundle,
    import_transport_bootstrap_bundle,
    validate_client_transport_profile,
)
from .sandbox import WorkerSandboxPolicy
from .toolbox_harness import (
    ToolboxBundleFile,
    ToolboxBundleAutoTool,
    ToolboxBundleTool,
    SandboxProfileSpec,
    ToolboxEnvironmentSpec,
    ToolboxEnvironmentManager,
    ToolboxAutoAssignmentRequest,
    ToolboxManualAssignmentRequest,
    ToolboxSandboxAssignment,
    ToolboxBundleSpec,
    StagedToolboxBundle,
    ToolboxBundleStager,
    ToolboxSandboxOrchestrator,
    ToolboxHarnessConfig,
    ToolboxExecutionHarness,
    HostedToolBoxRef,
    SandboxedToolboxFacade,
)
from .toolbox_admin import HostedToolboxAdmin

__all__ = [
    "EngineHostService",
    "EngineHostDaemon",
    "DaemonPidFile",
    "DEFAULT_DAEMON_PORT",
    "run_daemon_foreground",
    "start_daemon_background",
    "BaseConnection",
    "LocalSocketConnection",
    "SSHRelayConnection",
    "ConnectionError",
    "EngineHostControlChannel",
    "EngineProcessSupervisor",
    "CLIENT_REALM_ROOT_SUBDIR",
    "VALID_SECRET_RECORD_ENCRYPTION",
    "SecretRecord",
    "FileSecretStore",
    "get_default_client_realm_root",
    "client_realm_layout",
    "ensure_client_realm_dirs",
    "secret_record_path",
    "managed_key_path",
    "write_client_access",
    "read_client_access",
    "write_client_profile",
    "read_client_profile",
    "list_client_profiles",
    "iter_secret_ids",
    "append_client_audit_event",
    "list_client_audit_events",
    "materialize_secret_file",
    "resolve_client_profile_control_settings",
    "TRANSPORT_BOOTSTRAP_KIND",
    "make_transport_bootstrap_bundle",
    "validate_transport_bootstrap_bundle",
    "write_transport_bootstrap_bundle",
    "read_transport_bootstrap_bundle",
    "import_transport_bootstrap_bundle",
    "validate_client_transport_profile",
    "WorkerSandboxPolicy",
    "ToolboxBundleFile",
    "ToolboxBundleAutoTool",
    "ToolboxBundleTool",
    "SandboxProfileSpec",
    "ToolboxEnvironmentSpec",
    "ToolboxEnvironmentManager",
    "ToolboxAutoAssignmentRequest",
    "ToolboxManualAssignmentRequest",
    "ToolboxSandboxAssignment",
    "ToolboxBundleSpec",
    "StagedToolboxBundle",
    "ToolboxBundleStager",
    "ToolboxSandboxOrchestrator",
    "ToolboxHarnessConfig",
    "ToolboxExecutionHarness",
    "HostedToolBoxRef",
    "SandboxedToolboxFacade",
    "HostedToolboxAdmin",
]
