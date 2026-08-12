"""Strict, host-local repository for the unified hosting configuration."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping

from mp13_engine.mp13_config_paths import PathResolver


HOSTING_CONFIGURATION_CONTRACT = "hosting.configuration.v3"
_LOCKS_GUARD = threading.Lock()
_LOCKS: Dict[str, threading.RLock] = {}


class HostingConfigurationError(ValueError):
    """A stable validation error which never includes rejected values."""

    def __init__(self, code: str, field: str = "") -> None:
        self.code = str(code)
        self.field = str(field)
        super().__init__(f"{self.code}:{self.field}" if self.field else self.code)


def _mapping(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise HostingConfigurationError("hosting_configuration_type_invalid", field)
    return dict(value)


def _exact_keys(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise HostingConfigurationError("hosting_configuration_key_unknown", f"{field}.{unknown[0]}")


def _string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise HostingConfigurationError("hosting_configuration_type_invalid", field)
    return value.strip()


def _optional_bool(value: Mapping[str, Any], key: str, field: str) -> None:
    if key in value and not isinstance(value[key], bool):
        raise HostingConfigurationError("hosting_configuration_type_invalid", f"{field}.{key}")


def _optional_int(value: Mapping[str, Any], key: str, field: str, *, minimum: int = 0) -> None:
    if key not in value:
        return
    raw = value[key]
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        raise HostingConfigurationError("hosting_configuration_type_invalid", f"{field}.{key}")


def _logical_path(value: Any, field: str, resolver: PathResolver, label: str) -> tuple[str, str]:
    logical = _string(value, field)
    if not logical.startswith(f"@{label}"):
        raise HostingConfigurationError("hosting_configuration_path_invalid", field)
    try:
        resolved = resolver.resolve(logical)
    except (TypeError, ValueError) as exc:
        raise HostingConfigurationError("hosting_configuration_path_invalid", field) from exc
    return logical, str(resolved)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _revision(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class HostingConfiguration:
    contract: str
    control: Mapping[str, Any]
    package_management: Mapping[str, Any]
    environment_management: Mapping[str, Any]
    resolved_paths: Mapping[str, str]
    revision: str

    def logical_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "control": _thaw(self.control),
            "package_management": _thaw(self.package_management),
            "environment_management": _thaw(self.environment_management),
        }

    def inspect(self, *, local_admin: bool = False) -> Dict[str, Any]:
        package = _thaw(self.package_management)
        sources = dict(package.get("sources") or {})
        source_health = {
            source_id: {"configured": True, "enabled": bool(dict(source).get("enabled", True))}
            for source_id, source in sources.items()
        }
        result: Dict[str, Any] = {
            "contract": self.contract,
            "revision": self.revision,
            "configuration_health": {"status": "ok", "code": "hosting_configuration_ready"},
            "logical_roots": {
                "artifact_root": package["artifact_root"],
                "lock_root": package["lock_root"],
                "environment_root": self.environment_management["environment_root"],
                "scratch_root": self.environment_management["scratch_root"],
            },
            "source_availability": source_health,
            "environment_health": {"status": "ok", "code": "environment_configuration_ready"},
        }
        if local_admin:
            result["resolved_paths"] = dict(self.resolved_paths)
        return result


def parse_hosting_configuration(payload: Mapping[str, Any], resolver: PathResolver) -> HostingConfiguration:
    data = _mapping(payload, "configuration")
    _exact_keys(data, {"contract", "control", "package_management", "environment_management"}, "configuration")
    contract = _string(data.get("contract"), "contract")
    if contract != HOSTING_CONFIGURATION_CONTRACT:
        raise HostingConfigurationError("hosting_configuration_unsupported", "contract")

    control = _mapping(data.get("control"), "control")
    _exact_keys(control, {"authentication", "roles", "session_policy", "audit"}, "control")
    authentication = _mapping(control.get("authentication"), "control.authentication")
    _exact_keys(
        authentication,
        {"require_auth", "connectivity_mode", "endpoint_mode", "ssh_key_ref"},
        "control.authentication",
    )
    _optional_bool(authentication, "require_auth", "control.authentication")
    if "connectivity_mode" in authentication:
        mode = _string(authentication["connectivity_mode"], "control.authentication.connectivity_mode")
        if mode not in {"local_only", "lan", "remote"}:
            raise HostingConfigurationError("hosting_configuration_value_invalid", "control.authentication.connectivity_mode")
    if "endpoint_mode" in authentication:
        endpoint = _string(authentication["endpoint_mode"], "control.authentication.endpoint_mode")
        if endpoint not in {"exclusive", "shared"}:
            raise HostingConfigurationError("hosting_configuration_value_invalid", "control.authentication.endpoint_mode")
    if authentication.get("require_auth") is False and authentication.get("connectivity_mode", "local_only") != "local_only":
        raise HostingConfigurationError("hosting_configuration_policy_conflict", "control.authentication.require_auth")
    roles = _mapping(control.get("roles"), "control.roles")
    for role_id, role in roles.items():
        role_data = _mapping(role, f"control.roles.{role_id}")
        _exact_keys(role_data, {"permissions"}, f"control.roles.{role_id}")
        permissions = role_data.get("permissions", [])
        if not isinstance(permissions, list) or any(not isinstance(item, str) for item in permissions):
            raise HostingConfigurationError("hosting_configuration_type_invalid", f"control.roles.{role_id}.permissions")
    session_policy = _mapping(control.get("session_policy"), "control.session_policy")
    _exact_keys(session_policy, {"ttl_seconds", "idle_timeout_seconds", "max_sessions_per_key"}, "control.session_policy")
    for key in session_policy:
        _optional_int(session_policy, key, "control.session_policy", minimum=1)
    audit = _mapping(control.get("audit"), "control.audit")
    _exact_keys(audit, {"event_limit", "retention_seconds"}, "control.audit")
    for key in audit:
        _optional_int(audit, key, "control.audit", minimum=1)

    package = _mapping(data.get("package_management"), "package_management")
    _exact_keys(package, {"artifact_root", "lock_root", "sources", "credentials", "dependency_policy", "verification"}, "package_management")
    artifact_logical, artifact_resolved = _logical_path(package.get("artifact_root"), "package_management.artifact_root", resolver, "packages")
    lock_logical, lock_resolved = _logical_path(package.get("lock_root"), "package_management.lock_root", resolver, "packages")
    sources = _mapping(package.get("sources"), "package_management.sources")
    credentials = _mapping(package.get("credentials"), "package_management.credentials")
    for credential_id, credential in credentials.items():
        if not isinstance(credential, (str, dict)):
            raise HostingConfigurationError("hosting_configuration_type_invalid", f"package_management.credentials.{credential_id}")
        if isinstance(credential, dict):
            _exact_keys(credential, {"provider", "key"}, f"package_management.credentials.{credential_id}")
            _string(credential.get("provider"), f"package_management.credentials.{credential_id}.provider")
            _string(credential.get("key"), f"package_management.credentials.{credential_id}.key")
    for source_id, source in sources.items():
        source_data = _mapping(source, f"package_management.sources.{source_id}")
        _exact_keys(source_data, {"kind", "location", "credential_ref", "enabled", "priority"}, f"package_management.sources.{source_id}")
        _string(source_data.get("kind"), f"package_management.sources.{source_id}.kind")
        _string(source_data.get("location"), f"package_management.sources.{source_id}.location")
        _optional_bool(source_data, "enabled", f"package_management.sources.{source_id}")
        _optional_int(source_data, "priority", f"package_management.sources.{source_id}")
        credential_ref = source_data.get("credential_ref")
        if credential_ref is not None:
            ref = _string(credential_ref, f"package_management.sources.{source_id}.credential_ref")
            if ref not in credentials:
                raise HostingConfigurationError("hosting_configuration_credential_policy_conflict", f"package_management.sources.{source_id}.credential_ref")
    dependency_policy = _mapping(package.get("dependency_policy"), "package_management.dependency_policy")
    _exact_keys(dependency_policy, {"allow_prereleases", "allow_sdists", "allowed_packages", "denied_packages", "max_dependencies"}, "package_management.dependency_policy")
    _optional_bool(dependency_policy, "allow_prereleases", "package_management.dependency_policy")
    _optional_bool(dependency_policy, "allow_sdists", "package_management.dependency_policy")
    _optional_int(dependency_policy, "max_dependencies", "package_management.dependency_policy", minimum=1)
    for key in ("allowed_packages", "denied_packages"):
        if key in dependency_policy and (not isinstance(dependency_policy[key], list) or any(not isinstance(item, str) for item in dependency_policy[key])):
            raise HostingConfigurationError("hosting_configuration_type_invalid", f"package_management.dependency_policy.{key}")
    verification = _mapping(package.get("verification"), "package_management.verification")
    _exact_keys(verification, {"hash_algorithm", "verifier"}, "package_management.verification")
    if _string(verification.get("hash_algorithm"), "package_management.verification.hash_algorithm").lower() != "sha256":
        raise HostingConfigurationError("hosting_configuration_value_invalid", "package_management.verification.hash_algorithm")
    if "verifier" in verification and verification["verifier"] is not None and not isinstance(verification["verifier"], str):
        raise HostingConfigurationError("hosting_configuration_type_invalid", "package_management.verification.verifier")

    environment = _mapping(data.get("environment_management"), "environment_management")
    _exact_keys(environment, {"environment_root", "scratch_root", "retention", "cache"}, "environment_management")
    environment_logical, environment_resolved = _logical_path(environment.get("environment_root"), "environment_management.environment_root", resolver, "environments")
    scratch_logical, scratch_resolved = _logical_path(environment.get("scratch_root"), "environment_management.scratch_root", resolver, "hosting")
    retention = _mapping(environment.get("retention"), "environment_management.retention")
    _exact_keys(retention, {"unused_seconds", "receipt_seconds"}, "environment_management.retention")
    for key in retention:
        _optional_int(retention, key, "environment_management.retention")
    cache = _mapping(environment.get("cache"), "environment_management.cache")
    _exact_keys(cache, {"enabled", "max_bytes"}, "environment_management.cache")
    _optional_bool(cache, "enabled", "environment_management.cache")
    _optional_int(cache, "max_bytes", "environment_management.cache", minimum=1)

    normalized = {
        "contract": contract,
        "control": control,
        "package_management": {**package, "artifact_root": artifact_logical, "lock_root": lock_logical},
        "environment_management": {**environment, "environment_root": environment_logical, "scratch_root": scratch_logical},
    }
    return HostingConfiguration(
        contract=contract,
        control=_freeze(normalized["control"]),
        package_management=_freeze(normalized["package_management"]),
        environment_management=_freeze(normalized["environment_management"]),
        resolved_paths=MappingProxyType({
            "artifact_root": artifact_resolved,
            "lock_root": lock_resolved,
            "environment_root": environment_resolved,
            "scratch_root": scratch_resolved,
        }),
        revision=_revision(normalized),
    )


class HostingConfigurationRepository:
    """The sole locked reader/writer for ``hosting_config.json``."""

    def __init__(self, path: Path, resolver: PathResolver) -> None:
        self.path = Path(path).expanduser().resolve()
        self.resolver = resolver

    @contextmanager
    def _locked(self) -> Iterator[None]:
        key = str(self.path)
        with _LOCKS_GUARD:
            lock = _LOCKS.setdefault(key, threading.RLock())
        with lock:
            yield

    def read(self) -> HostingConfiguration:
        with self._locked():
            if not self.path.exists():
                raise HostingConfigurationError("hosting_configuration_missing")
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise HostingConfigurationError("hosting_configuration_invalid") from exc
            return parse_hosting_configuration(payload, self.resolver)

    def write(self, payload: Mapping[str, Any]) -> HostingConfiguration:
        configuration = parse_hosting_configuration(payload, self.resolver)
        with self._locked():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            descriptor = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(configuration.logical_dict(), handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, self.path)
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
            finally:
                temp.unlink(missing_ok=True)
        return configuration

    def delete(self) -> None:
        """Remove the authority during an explicit local rollback/reset."""
        with self._locked():
            self.path.unlink(missing_ok=True)


__all__ = [
    "HOSTING_CONFIGURATION_CONTRACT",
    "HostingConfiguration",
    "HostingConfigurationError",
    "HostingConfigurationRepository",
    "parse_hosting_configuration",
]
