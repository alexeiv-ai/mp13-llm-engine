"""
mp13 engine hosting package.

Public API — stdlib only at module level, no heavy imports (torch/transformers etc.).

Key classes and functions available for import:
    EngineHostService         — file-backed process lifecycle and control-plane state
    EngineHostDaemon          — local IPC daemon server for control commands
    DaemonPidFile             — read/write/probe the daemon PID file
    DEFAULT_DAEMON_PORT       — daemon default port metadata (19876)
    run_daemon_foreground()   — start daemon blocking in foreground
    start_daemon_background() — spawn daemon as detached background process
    LocalSocketConnection     — persistent local daemon connection over IPC metadata
    SSHRelayConnection        — persistent SSH subprocess running --relay-wrapper
    BaseConnection            — abstract base for connection strategies
    ConnectionError           — raised on unrecoverable connection failure
    EngineHostControlChannel  — command channel requiring daemon control, with diagnostic-only fallback
    EngineProcessSupervisor   — in-process persisted tracker for managed worker processes
    WorkerSandboxPolicy       — worker sandbox policy schema
"""
from __future__ import annotations

from .service.host_service import EngineHostService
from .daemon import (
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
    HANDOFF_PAYLOAD_KIND,
    VALID_SECRET_RECORD_ENCRYPTION,
    SecretRecord,
    FileSecretStore,
    create_private_key_handoff_text,
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
    store_private_key_handoff_in_realm,
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
from .secure_state import (
    SECURE_STATE_ALGORITHM,
    SECURE_STATE_ENV_NAMES,
    SECURE_STATE_KIND,
    SECURE_STATE_VERSION,
    SecureStateError,
    SecureStateFormatError,
    SecureStateLockedError,
    decrypt_json_payload,
    decrypt_secure_json_file,
    encrypt_json_payload,
    encrypt_secure_json_file,
    is_secure_state_envelope,
    read_secure_json,
    rotate_secure_json_file,
    secure_state_key_available,
    secure_state_status,
    write_secure_json,
)
from .sandbox import WorkerSandboxPolicy
from .toolbox_harness import (
    ToolboxBundleFile,
    ToolboxBundleAutoTool,
    ToolboxBundleTool,
    SandboxProfileSpec,
    ToolboxEnvironmentSpec,
    ToolboxEnvironmentManager,
    RuntimeEnvironmentManager,
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
from .engine_discovery import (
    is_engine_available,
    is_engine_discoverable,
)

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
    "HANDOFF_PAYLOAD_KIND",
    "VALID_SECRET_RECORD_ENCRYPTION",
    "SecretRecord",
    "FileSecretStore",
    "create_private_key_handoff_text",
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
    "store_private_key_handoff_in_realm",
    "TRANSPORT_BOOTSTRAP_KIND",
    "make_transport_bootstrap_bundle",
    "validate_transport_bootstrap_bundle",
    "write_transport_bootstrap_bundle",
    "read_transport_bootstrap_bundle",
    "import_transport_bootstrap_bundle",
    "validate_client_transport_profile",
    "SECURE_STATE_ALGORITHM",
    "SECURE_STATE_ENV_NAMES",
    "SECURE_STATE_KIND",
    "SECURE_STATE_VERSION",
    "SecureStateError",
    "SecureStateFormatError",
    "SecureStateLockedError",
    "decrypt_json_payload",
    "decrypt_secure_json_file",
    "encrypt_json_payload",
    "encrypt_secure_json_file",
    "is_secure_state_envelope",
    "read_secure_json",
    "rotate_secure_json_file",
    "secure_state_key_available",
    "secure_state_status",
    "write_secure_json",
    "WorkerSandboxPolicy",
    "ToolboxBundleFile",
    "ToolboxBundleAutoTool",
    "ToolboxBundleTool",
    "SandboxProfileSpec",
    "ToolboxEnvironmentSpec",
    "ToolboxEnvironmentManager",
    "RuntimeEnvironmentManager",
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
    "is_engine_available",
    "is_engine_discoverable",
]
