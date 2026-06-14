from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

PlatformSupport = Literal["supported", "partial", "unsupported", "planned_supported"]
FsAccess = Literal["read", "write", "execute"]
IntegrityLevel = Literal["untrusted", "low", "medium"]
NetworkMode = Literal["disabled", "direct", "brokered_only"]


def _support(value: Any, default: PlatformSupport) -> PlatformSupport:
    raw = str(value or "").strip().lower()
    if raw in {"supported", "partial", "unsupported", "planned_supported"}:
        return raw  # type: ignore[return-value]
    return default


def _integrity(value: Any, default: IntegrityLevel = "low") -> IntegrityLevel:
    raw = str(value or "").strip().lower()
    if raw in {"untrusted", "low", "medium"}:
        return raw  # type: ignore[return-value]
    return default


def _network_mode(value: Any, default: NetworkMode = "disabled") -> NetworkMode:
    raw = str(value or "").strip().lower()
    if raw in {"disabled", "direct", "brokered_only"}:
        return raw  # type: ignore[return-value]
    return default


def _fs_access_list(items: Any) -> List[FsAccess]:
    out: List[FsAccess] = []
    for item in list(items or []):
        raw = str(item or "").strip().lower()
        if raw in {"read", "write", "execute"} and raw not in out:
            out.append(raw)  # type: ignore[arg-type]
    return out


def _artifact_roots(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        rows = [{"name": key, "path": path} for key, path in value.items()]
    else:
        rows = [dict(row or {}) for row in list(value or []) if isinstance(row, dict)]
    out: Dict[str, str] = {}
    for row in rows:
        raw_name = str(row.get("name") or row.get("root_id") or row.get("alias") or "").strip()
        name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw_name).strip("._")
        path = str(row.get("path") or "").strip()
        if name and path:
            out[name] = path
    return dict(sorted(out.items()))


@dataclass
class SandboxFsRule:
    path: str
    root_id: Optional[str] = None
    access: List[FsAccess] = field(default_factory=list)
    windows_status: PlatformSupport = "partial"
    linux_status: PlatformSupport = "supported"

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "SandboxFsRule":
        raw = dict(data or {})
        status = dict(raw.get("platform_status") or {})
        return cls(
            root_id=str(raw.get("root_id") or "").strip() or None,
            path=str(raw.get("path") or "").strip(),
            access=_fs_access_list(raw.get("access")),
            windows_status=_support(status.get("windows"), "partial"),
            linux_status=_support(status.get("linux"), "supported"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_id": self.root_id,
            "path": self.path,
            "access": list(self.access),
            "platform_status": {
                "windows": self.windows_status,
                "linux": self.linux_status,
            },
        }


@dataclass
class SandboxProcessPolicy:
    allow_subprocess: bool = False
    inherit_parent_handles: bool = False
    windows_allow_subprocess_status: PlatformSupport = "partial"
    windows_inherit_parent_handles_status: PlatformSupport = "supported"
    linux_allow_subprocess_status: PlatformSupport = "supported"
    linux_inherit_parent_handles_status: PlatformSupport = "supported"

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "SandboxProcessPolicy":
        raw = dict(data or {})
        status = dict(raw.get("platform_status") or {})
        win = dict(status.get("windows") or {})
        lin = dict(status.get("linux") or {})
        return cls(
            allow_subprocess=bool(raw.get("allow_subprocess", False)),
            inherit_parent_handles=bool(raw.get("inherit_parent_handles", False)),
            windows_allow_subprocess_status=_support(win.get("allow_subprocess"), "partial"),
            windows_inherit_parent_handles_status=_support(win.get("inherit_parent_handles"), "supported"),
            linux_allow_subprocess_status=_support(lin.get("allow_subprocess"), "supported"),
            linux_inherit_parent_handles_status=_support(lin.get("inherit_parent_handles"), "supported"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allow_subprocess": bool(self.allow_subprocess),
            "inherit_parent_handles": bool(self.inherit_parent_handles),
            "platform_status": {
                "windows": {
                    "allow_subprocess": self.windows_allow_subprocess_status,
                    "inherit_parent_handles": self.windows_inherit_parent_handles_status,
                },
                "linux": {
                    "allow_subprocess": self.linux_allow_subprocess_status,
                    "inherit_parent_handles": self.linux_inherit_parent_handles_status,
                },
            },
        }


@dataclass
class SandboxNetworkPolicy:
    mode: NetworkMode = "disabled"
    allow_hosts: List[str] = field(default_factory=list)
    allow_url_prefixes: List[str] = field(default_factory=list)
    windows_disabled_status: PlatformSupport = "partial"
    windows_host_allowlist_status: PlatformSupport = "unsupported"
    windows_url_allowlist_status: PlatformSupport = "unsupported"
    linux_disabled_status: PlatformSupport = "supported"
    linux_host_allowlist_status: PlatformSupport = "unsupported"
    linux_url_allowlist_status: PlatformSupport = "unsupported"

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "SandboxNetworkPolicy":
        raw = dict(data or {})
        status = dict(raw.get("platform_status") or {})
        win = dict(status.get("windows") or {})
        lin = dict(status.get("linux") or {})
        return cls(
            mode=_network_mode(raw.get("mode"), "disabled"),
            allow_hosts=[str(x).strip() for x in list(raw.get("allow_hosts") or []) if str(x).strip()],
            allow_url_prefixes=[str(x).strip() for x in list(raw.get("allow_url_prefixes") or []) if str(x).strip()],
            windows_disabled_status=_support(win.get("disabled"), "partial"),
            windows_host_allowlist_status=_support(win.get("allow_hosts"), "unsupported"),
            windows_url_allowlist_status=_support(win.get("allow_url_prefixes"), "unsupported"),
            linux_disabled_status=_support(lin.get("disabled"), "supported"),
            linux_host_allowlist_status=_support(lin.get("allow_hosts"), "unsupported"),
            linux_url_allowlist_status=_support(lin.get("allow_url_prefixes"), "unsupported"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "allow_hosts": list(self.allow_hosts),
            "allow_url_prefixes": list(self.allow_url_prefixes),
            "platform_status": {
                "windows": {
                    "disabled": self.windows_disabled_status,
                    "allow_hosts": self.windows_host_allowlist_status,
                    "allow_url_prefixes": self.windows_url_allowlist_status,
                },
                "linux": {
                    "disabled": self.linux_disabled_status,
                    "allow_hosts": self.linux_host_allowlist_status,
                    "allow_url_prefixes": self.linux_url_allowlist_status,
                },
            },
        }


@dataclass
class WindowsSandboxPolicy:
    restricted_token: bool = True
    integrity_level: IntegrityLevel = "low"
    job_object: bool = True

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "WindowsSandboxPolicy":
        raw = dict(data or {})
        return cls(
            restricted_token=bool(raw.get("restricted_token", True)),
            integrity_level=_integrity(raw.get("integrity_level"), "low"),
            job_object=bool(raw.get("job_object", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "restricted_token": bool(self.restricted_token),
            "integrity_level": self.integrity_level,
            "job_object": bool(self.job_object),
        }


@dataclass
class BrokeredIoPolicy:
    filesystem: bool = True
    http: bool = True
    subprocess: bool = False

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "BrokeredIoPolicy":
        raw = dict(data or {})
        return cls(
            filesystem=bool(raw.get("filesystem", True)),
            http=bool(raw.get("http", True)),
            subprocess=bool(raw.get("subprocess", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filesystem": bool(self.filesystem),
            "http": bool(self.http),
            "subprocess": bool(self.subprocess),
        }


@dataclass
class WorkerSandboxPolicy:
    enabled: bool = False
    profile: str = "generic_worker_v1"
    filesystem_rules: List[SandboxFsRule] = field(default_factory=list)
    artifact_roots: Dict[str, str] = field(default_factory=dict)
    process: SandboxProcessPolicy = field(default_factory=SandboxProcessPolicy)
    network: SandboxNetworkPolicy = field(default_factory=SandboxNetworkPolicy)
    windows: WindowsSandboxPolicy = field(default_factory=WindowsSandboxPolicy)
    brokered_io: BrokeredIoPolicy = field(default_factory=BrokeredIoPolicy)

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "WorkerSandboxPolicy":
        raw = dict(data or {})
        sandbox = dict(raw.get("sandbox") or raw)
        filesystem = dict(sandbox.get("filesystem") or {})
        platform_policy = dict(sandbox.get("platform_policy") or {})
        return cls(
            enabled=bool(sandbox.get("enabled", False)),
            profile=str(sandbox.get("profile") or "generic_worker_v1").strip() or "generic_worker_v1",
            filesystem_rules=[
                SandboxFsRule.from_mapping(item)
                for item in list(filesystem.get("rules") or [])
                if isinstance(item, dict)
            ],
            artifact_roots=_artifact_roots(sandbox.get("artifact_roots")),
            process=SandboxProcessPolicy.from_mapping(sandbox.get("process")),
            network=SandboxNetworkPolicy.from_mapping(sandbox.get("network")),
            windows=WindowsSandboxPolicy.from_mapping(platform_policy.get("windows")),
            brokered_io=BrokeredIoPolicy.from_mapping(sandbox.get("brokered_io")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sandbox": {
                "enabled": bool(self.enabled),
                "profile": self.profile,
                "platform_policy": {
                    "windows": self.windows.to_dict(),
                },
                "filesystem": {
                    "default_access": "deny",
                    "rules": [item.to_dict() for item in self.filesystem_rules],
                },
                "artifact_roots": dict(self.artifact_roots),
                "process": self.process.to_dict(),
                "network": self.network.to_dict(),
                "brokered_io": self.brokered_io.to_dict(),
            }
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "profile": self.profile,
            "filesystem_rules_count": len(self.filesystem_rules),
            "artifact_roots": dict(self.artifact_roots),
            "brokered_filesystem": bool(self.brokered_io.filesystem),
            "brokered_http": bool(self.brokered_io.http),
            "allow_subprocess": bool(self.process.allow_subprocess),
            "inherit_parent_handles": bool(self.process.inherit_parent_handles),
            "network_mode": self.network.mode,
            "windows": self.windows.to_dict(),
        }
