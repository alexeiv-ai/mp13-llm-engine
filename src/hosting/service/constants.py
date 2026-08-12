"""Constants and default paths for the engine host service."""
from __future__ import annotations

from pathlib import Path


def _default_state_dir() -> Path:
    # Keep hosting bootstrap lightweight: avoid importing mp13_engine package
    # during module import to prevent unrelated heavy dependency side-effects.
    return (Path.home() / ".mp13-llm" / "hosting" / "state").expanduser().resolve()


def _default_hosting_root() -> Path:
    return (Path.home() / ".mp13-llm" / "hosting").expanduser().resolve()


DEFAULT_STATE_DIR = _default_state_dir()
DEFAULT_HOSTING_ROOT = _default_hosting_root()
DEFAULT_ENGINES_STATE_FILE = DEFAULT_STATE_DIR / "managed_engines.json"
DEFAULT_HOSTING_CONFIG_FILE = DEFAULT_HOSTING_ROOT / "hosting_config.json"
DEFAULT_CONTROL_STATE_FILE = DEFAULT_STATE_DIR / "control_state.json"
DAEMON_VERSION = "3.0.0"

ROLE_ADMIN = "admin"
ROLE_CONFIG_EDITOR = "config_editor"
ROLE_WORKER_USER = "worker_user"
ROLE_MODEL_USER_WITH_MODEL_CONTROL = "model_user_with_model_control"
ROLE_MODEL_USER = "model_user"
ROLE_DIAGNOSTIC_USER = "diagnostic_user"
ROLE_TRANSPORT = "transport"
ROLE_DEPENDENCY_APPROVER = "dependency_approver"

LIFECYCLE_PROFILE_FOREGROUND = "foreground_terminal_bound"
LIFECYCLE_PROFILE_DETACHED = "detached_user_process"
LIFECYCLE_PROFILE_SERVICE = "service_managed"
VALID_LIFECYCLE_PROFILES = {
    LIFECYCLE_PROFILE_FOREGROUND,
    LIFECYCLE_PROFILE_DETACHED,
    LIFECYCLE_PROFILE_SERVICE,
}

VALID_AUTH_ROLES = {
    ROLE_ADMIN,
    ROLE_CONFIG_EDITOR,
    ROLE_WORKER_USER,
    ROLE_MODEL_USER_WITH_MODEL_CONTROL,
    ROLE_MODEL_USER,
    ROLE_DIAGNOSTIC_USER,
    ROLE_TRANSPORT,
    ROLE_DEPENDENCY_APPROVER,
}

VALID_FORCE_OVERRIDE_REASONS = {
    "stale_owner_unreachable",
    "owner_malicious",
    "security_incident",
    "policy_recovery",
}
EMERGENCY_FORCE_OVERRIDE_REASONS = {
    "stale_owner_unreachable",
    "owner_malicious",
    "security_incident",
}
