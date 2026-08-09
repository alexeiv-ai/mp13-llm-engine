"""Toolbox bundle, sandbox, assignment, and harness data models."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from packaging.requirements import InvalidRequirement, Requirement

from .common import _sha256_text, _stable_json
from .catalog import normalize_import_root
from .identity import canonical_json_bytes, definition_revision, require_digest, resolved_profile_identity


TOOLBOX_DEFINITION_CONTRACT = "hosting.toolbox.definition"
_DEFINITION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_TOOL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}")


def _strict_model_fields(row: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    unknown = sorted(set(row) - fields)
    missing = sorted(fields - set(row))
    if unknown:
        raise ValueError(f"{label}_unknown_fields:{','.join(unknown)}")
    if missing:
        raise ValueError(f"{label}_missing_fields:{','.join(missing)}")


def _canonical_mapping(value: Any, *, label: str, nullable: bool = False) -> dict[str, Any] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}_object_required")
    return json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))


def _strict_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label}_boolean_required")
    return value


@dataclass(frozen=True)
class ToolboxDependencyRequest:
    mode: str
    template_id: str | None
    declared_imports: tuple[str, ...]
    package_requirements: tuple[str, ...]

    def __post_init__(self) -> None:
        mode = str(self.mode or "").strip()
        if mode not in {"auto", "template", "custom"}:
            raise ValueError("dependency_mode_invalid")
        object.__setattr__(self, "mode", mode)
        template = str(self.template_id or "").strip() or None
        if mode == "auto" and template is not None:
            raise ValueError("dependency_auto_template_forbidden")
        if mode in {"template", "custom"} and (
            template is None or not _DEFINITION_ID_RE.fullmatch(template)
        ):
            raise ValueError("dependency_template_id_required")
        object.__setattr__(self, "template_id", template)
        imports = tuple(sorted(normalize_import_root(item) for item in self.declared_imports))
        if len(set(imports)) != len(imports):
            raise ValueError("dependency_declared_imports_duplicate")
        object.__setattr__(self, "declared_imports", imports)
        normalized_requirements: list[str] = []
        for raw in self.package_requirements:
            try:
                requirement = Requirement(str(raw or "").strip())
            except InvalidRequirement as exc:
                raise ValueError("dependency_package_requirement_invalid") from exc
            if requirement.url or requirement.marker:
                raise ValueError("dependency_package_requirement_unsupported")
            normalized_requirements.append(str(requirement))
        requirements = tuple(sorted(normalized_requirements))
        if len(set(requirements)) != len(requirements):
            raise ValueError("dependency_package_requirements_duplicate")
        if mode == "custom" and not requirements:
            raise ValueError("dependency_custom_requirement_required")
        object.__setattr__(self, "package_requirements", requirements)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "template_id": self.template_id,
            "declared_imports": list(self.declared_imports),
            "package_requirements": list(self.package_requirements),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxDependencyRequest":
        row = dict(payload or {})
        _strict_model_fields(
            row,
            {"mode", "template_id", "declared_imports", "package_requirements"},
            label="toolbox_dependency",
        )
        for field_name in ("declared_imports", "package_requirements"):
            if not isinstance(row[field_name], Sequence) or isinstance(row[field_name], (str, bytes, bytearray)):
                raise ValueError(f"dependency_{field_name}_list_required")
        return cls(
            mode=row["mode"],
            template_id=row["template_id"],
            declared_imports=tuple(row["declared_imports"]),
            package_requirements=tuple(row["package_requirements"]),
        )


@dataclass(frozen=True)
class ToolboxIntrinsicSelection:
    names: tuple[str, ...]
    include_guides: bool
    sandbox_policy: Mapping[str, Any]

    def __post_init__(self) -> None:
        names = tuple(sorted(str(item or "").strip() for item in self.names))
        if any(not _TOOL_NAME_RE.fullmatch(item) for item in names) or len(set(names)) != len(names):
            raise ValueError("definition_intrinsic_names_invalid")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "include_guides", _strict_bool(self.include_guides, label="definition_include_guides"))
        object.__setattr__(self, "sandbox_policy", _canonical_mapping(self.sandbox_policy, label="definition_intrinsic_sandbox_policy"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "include_guides": self.include_guides,
            "sandbox_policy": dict(self.sandbox_policy),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxIntrinsicSelection":
        row = dict(payload or {})
        _strict_model_fields(row, {"names", "include_guides", "sandbox_policy"}, label="toolbox_intrinsics")
        if not isinstance(row["names"], Sequence) or isinstance(row["names"], (str, bytes, bytearray)):
            raise ValueError("definition_intrinsic_names_list_required")
        return cls(
            names=tuple(row["names"]),
            include_guides=row["include_guides"],
            sandbox_policy=row["sandbox_policy"],
        )


@dataclass
class ToolboxBundleFile:
    relative_path: str
    content: str

    def normalized_path(self) -> str:
        raw = str(self.relative_path or "").replace("\\", "/").strip("/")
        if not raw or raw.startswith("../") or "/../" in f"/{raw}/":
            raise ValueError("bundle_file_path_invalid")
        return raw

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relative_path": self.normalized_path(),
            "content_sha256": _sha256_text(str(self.content or "")),
        }

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "relative_path": self.normalized_path(),
            "content": str(self.content or ""),
        }

    @classmethod
    def from_runtime_dict(cls, payload: Dict[str, Any]) -> "ToolboxBundleFile":
        row = dict(payload or {})
        return cls(
            relative_path=str(row.get("relative_path") or "").strip(),
            content=str(row.get("content") or ""),
        )

    @classmethod
    def from_definition_dict(cls, payload: Mapping[str, Any]) -> "ToolboxBundleFile":
        row = dict(payload or {})
        _strict_model_fields(row, {"relative_path", "content"}, label="toolbox_definition_file")
        if not isinstance(row["relative_path"], str) or not isinstance(row["content"], str):
            raise ValueError("toolbox_definition_file_strings_required")
        model = cls(relative_path=row["relative_path"], content=row["content"])
        path = model.normalized_path()
        segments = path.split("/")
        if any(not item or item in {".", ".."} for item in segments):
            raise ValueError("bundle_file_path_invalid")
        return model


@dataclass
class ToolboxBundleTool:
    definition: Dict[str, Any]
    entrypoint: str
    hidden: bool = False
    non_restartable: bool = False
    callback_signature: Optional[Dict[str, Any]] = None
    concurrency: Optional[Dict[str, Any]] = None

    def tool_name(self) -> str:
        fn = dict(self.definition.get("function") or {})
        name = str(fn.get("name") or "").strip()
        if not name:
            raise ValueError("tool_name_required")
        return name

    def to_dict(self) -> Dict[str, Any]:
        row = {
            "name": self.tool_name(),
            "definition": dict(self.definition or {}),
            "entrypoint": str(self.entrypoint or "").strip(),
            "hidden": bool(self.hidden),
            "non_restartable": bool(self.non_restartable),
            "callback_signature": dict(self.callback_signature or {}) or None,
        }
        if isinstance(self.concurrency, dict) and self.concurrency:
            row["concurrency"] = dict(self.concurrency)
        return row


@dataclass
class ToolboxBundleAutoTool:
    module_name: str
    callable_name: str
    activate: bool = True
    hidden: bool = False
    non_restartable: bool = False
    guide_content: Optional[Dict[str, List[str]]] = None
    guide_description: Optional[str] = None
    callback_signature: Optional[Dict[str, Any]] = None
    concurrency: Optional[Dict[str, Any]] = None

    def normalized_module_name(self) -> str:
        raw = str(self.module_name or "").strip()
        if not raw:
            raise ValueError("auto_tool_module_name_required")
        return raw

    def normalized_callable_name(self) -> str:
        raw = str(self.callable_name or "").strip()
        if not raw:
            raise ValueError("auto_tool_callable_name_required")
        return raw

    def tool_name(self) -> str:
        return self.normalized_callable_name()

    def to_dict(self) -> Dict[str, Any]:
        row = {
            "name": self.tool_name(),
            "module_name": self.normalized_module_name(),
            "callable_name": self.normalized_callable_name(),
            "activate": bool(self.activate),
            "hidden": bool(self.hidden),
            "non_restartable": bool(self.non_restartable),
            "guide_content": dict(self.guide_content or {}) or None,
            "guide_description": str(self.guide_description or "").strip() or None,
            "callback_signature": dict(self.callback_signature or {}) or None,
        }
        if isinstance(self.concurrency, dict) and self.concurrency:
            row["concurrency"] = dict(self.concurrency)
        return row


@dataclass
class SandboxProfileSpec:
    profile_id: str = ""
    environment_name: str = ""
    required_imports: List[str] = field(default_factory=list)
    sandbox_policy: Dict[str, Any] = field(default_factory=dict)

    def normalized_profile_id(self) -> str:
        raw = str(self.profile_id or "").strip()
        if raw:
            return raw
        return f"profile-{self.profile_fingerprint()[:12]}"

    def normalized_required_imports(self) -> List[str]:
        imports: List[str] = []
        seen: set[str] = set()
        for item in list(self.required_imports or []):
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                imports.append(name)
        return imports

    def profile_fingerprint(self) -> str:
        payload = {
            "environment_name": str(self.environment_name or "").strip() or "base",
            "required_imports": self.normalized_required_imports(),
            "sandbox_policy": dict(self.sandbox_policy or {}),
        }
        return _sha256_text(_stable_json(payload))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.normalized_profile_id(),
            "environment_name": str(self.environment_name or "").strip() or "base",
            "required_imports": self.normalized_required_imports(),
            "sandbox_policy": dict(self.sandbox_policy or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SandboxProfileSpec":
        row = dict(payload or {})
        return cls(
            profile_id=str(row.get("profile_id") or "").strip(),
            environment_name=str(row.get("environment_name") or "base").strip() or "base",
            required_imports=[str(item or "").strip() for item in list(row.get("required_imports") or []) if str(item or "").strip()],
            sandbox_policy=dict(row.get("sandbox_policy") or {}),
        )


@dataclass
class ToolboxAutoAssignmentRequest:
    files: List[ToolboxBundleFile]
    module_name: str
    callable_name: str
    sandbox_profile: SandboxProfileSpec = field(default_factory=SandboxProfileSpec)
    activate: bool = True
    hidden: bool = False
    non_restartable: bool = False
    guide_content: Optional[Dict[str, List[str]]] = None
    guide_description: Optional[str] = None
    callback_signature: Optional[Dict[str, Any]] = None
    concurrency: Optional[Dict[str, Any]] = None

    def to_auto_tool(self) -> ToolboxBundleAutoTool:
        return ToolboxBundleAutoTool(
            module_name=str(self.module_name or "").strip(),
            callable_name=str(self.callable_name or "").strip(),
            activate=bool(self.activate),
            hidden=bool(self.hidden),
            non_restartable=bool(self.non_restartable),
            guide_content=dict(self.guide_content or {}) or None,
            guide_description=str(self.guide_description or "").strip() or None,
            callback_signature=dict(self.callback_signature or {}) or None,
            concurrency=dict(self.concurrency or {}) or None,
        )

    def stable_key(self) -> str:
        return f"{str(self.module_name or '').strip()}:{str(self.callable_name or '').strip()}"

    def to_runtime_dict(self) -> Dict[str, Any]:
        row = {
            "files": [item.to_runtime_dict() for item in list(self.files or [])],
            "module_name": str(self.module_name or "").strip(),
            "callable_name": str(self.callable_name or "").strip(),
            "sandbox_profile": self.sandbox_profile.to_dict(),
            "activate": bool(self.activate),
            "hidden": bool(self.hidden),
            "non_restartable": bool(self.non_restartable),
            "guide_content": dict(self.guide_content or {}) or None,
            "guide_description": str(self.guide_description or "").strip() or None,
            "callback_signature": dict(self.callback_signature or {}) or None,
        }
        if isinstance(self.concurrency, dict) and self.concurrency:
            row["concurrency"] = dict(self.concurrency)
        return row

    @classmethod
    def from_runtime_dict(cls, payload: Dict[str, Any]) -> "ToolboxAutoAssignmentRequest":
        row = dict(payload or {})
        return cls(
            files=[ToolboxBundleFile.from_runtime_dict(dict(item or {})) for item in list(row.get("files") or [])],
            module_name=str(row.get("module_name") or "").strip(),
            callable_name=str(row.get("callable_name") or "").strip(),
            sandbox_profile=SandboxProfileSpec.from_dict(dict(row.get("sandbox_profile") or {})),
            activate=bool(row.get("activate", True)),
            hidden=bool(row.get("hidden", False)),
            non_restartable=bool(row.get("non_restartable", False)),
            guide_content=dict(row.get("guide_content") or {}) or None,
            guide_description=str(row.get("guide_description") or "").strip() or None,
            callback_signature=dict(row.get("callback_signature") or {}) or None,
            concurrency=dict(row.get("concurrency") or {}) or None,
        )


@dataclass
class ToolboxManualAssignmentRequest:
    files: List[ToolboxBundleFile]
    module_name: str
    callable_name: str
    tool_definition: Dict[str, Any]
    sandbox_profile: SandboxProfileSpec = field(default_factory=SandboxProfileSpec)
    hidden: bool = False
    non_restartable: bool = False
    callback_signature: Optional[Dict[str, Any]] = None

    def to_bundle_tool(self) -> ToolboxBundleTool:
        return ToolboxBundleTool(
            definition=dict(self.tool_definition or {}),
            entrypoint=f"{str(self.module_name or '').strip()}:{str(self.callable_name or '').strip()}",
            hidden=bool(self.hidden),
            non_restartable=bool(self.non_restartable),
            callback_signature=dict(self.callback_signature or {}) or None,
        )

    def stable_key(self) -> str:
        return f"manual:{str(self.module_name or '').strip()}:{str(self.callable_name or '').strip()}"

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "files": [item.to_runtime_dict() for item in list(self.files or [])],
            "module_name": str(self.module_name or "").strip(),
            "callable_name": str(self.callable_name or "").strip(),
            "tool_definition": dict(self.tool_definition or {}),
            "sandbox_profile": self.sandbox_profile.to_dict(),
            "hidden": bool(self.hidden),
            "non_restartable": bool(self.non_restartable),
            "callback_signature": dict(self.callback_signature or {}) or None,
        }

    @classmethod
    def from_runtime_dict(cls, payload: Dict[str, Any]) -> "ToolboxManualAssignmentRequest":
        row = dict(payload or {})
        return cls(
            files=[ToolboxBundleFile.from_runtime_dict(dict(item or {})) for item in list(row.get("files") or [])],
            module_name=str(row.get("module_name") or "").strip(),
            callable_name=str(row.get("callable_name") or "").strip(),
            tool_definition=dict(row.get("tool_definition") or {}),
            sandbox_profile=SandboxProfileSpec.from_dict(dict(row.get("sandbox_profile") or {})),
            hidden=bool(row.get("hidden", False)),
            non_restartable=bool(row.get("non_restartable", False)),
            callback_signature=dict(row.get("callback_signature") or {}) or None,
        )


@dataclass
class ToolboxSandboxAssignment:
    toolbox_id: str
    sandbox_profile: SandboxProfileSpec
    bundle_spec: "ToolboxBundleSpec"
    staged_bundle: Optional["StagedToolboxBundle"] = None
    registration: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ResolvedToolboxProfileSpec:
    environment_key: str
    template_id: str
    template_lock_digest: str
    custom_resolved_lock_digest: str | None
    sandbox_policy: Mapping[str, Any]
    assigned_tool_keys: tuple[str, ...]
    resolved_import_roots: tuple[str, ...]
    profile_id: str = ""

    def __post_init__(self) -> None:
        environment_key = require_digest(self.environment_key, label="resolved_profile_environment_key")
        template_lock = require_digest(self.template_lock_digest, label="resolved_profile_template_lock_digest")
        custom_lock = (
            require_digest(self.custom_resolved_lock_digest, label="resolved_profile_custom_lock_digest")
            if self.custom_resolved_lock_digest is not None
            else None
        )
        template_id = str(self.template_id or "").strip()
        if not _DEFINITION_ID_RE.fullmatch(template_id):
            raise ValueError("resolved_profile_template_id_invalid")
        sandbox = _canonical_mapping(self.sandbox_policy, label="resolved_profile_sandbox_policy")
        assigned = tuple(sorted(str(item or "").strip() for item in self.assigned_tool_keys))
        if not assigned or any(not item for item in assigned) or len(set(assigned)) != len(assigned):
            raise ValueError("resolved_profile_assigned_tool_keys_invalid")
        roots = tuple(sorted(normalize_import_root(item) for item in self.resolved_import_roots))
        if len(set(roots)) != len(roots):
            raise ValueError("resolved_profile_import_roots_duplicate")
        expected_profile_id = resolved_profile_identity(
            environment_identity=environment_key,
            sandbox_policy=sandbox,
        )
        if self.profile_id and self.profile_id != expected_profile_id:
            raise ValueError("resolved_profile_id_mismatch")
        object.__setattr__(self, "environment_key", environment_key)
        object.__setattr__(self, "template_id", template_id)
        object.__setattr__(self, "template_lock_digest", template_lock)
        object.__setattr__(self, "custom_resolved_lock_digest", custom_lock)
        object.__setattr__(self, "sandbox_policy", sandbox)
        object.__setattr__(self, "assigned_tool_keys", assigned)
        object.__setattr__(self, "resolved_import_roots", roots)
        object.__setattr__(self, "profile_id", expected_profile_id)

    @property
    def effective_lock_digest(self) -> str:
        return self.custom_resolved_lock_digest or self.template_lock_digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "environment_key": self.environment_key,
            "template_id": self.template_id,
            "template_lock_digest": self.template_lock_digest,
            "custom_resolved_lock_digest": self.custom_resolved_lock_digest,
            "sandbox_policy": dict(self.sandbox_policy),
            "assigned_tool_keys": list(self.assigned_tool_keys),
            "resolved_import_roots": list(self.resolved_import_roots),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResolvedToolboxProfileSpec":
        row = dict(payload or {})
        fields = {
            "profile_id", "environment_key", "template_id", "template_lock_digest",
            "custom_resolved_lock_digest", "sandbox_policy", "assigned_tool_keys",
            "resolved_import_roots",
        }
        _strict_model_fields(row, fields, label="resolved_toolbox_profile")
        return cls(
            **{
                **row,
                "assigned_tool_keys": tuple(row["assigned_tool_keys"]),
                "resolved_import_roots": tuple(row["resolved_import_roots"]),
            }
        )


@dataclass
class ResolvedToolboxSandboxAssignment:
    toolbox_id: str
    resolved_profile: ResolvedToolboxProfileSpec
    bundle_spec: "ToolboxBundleSpec"
    classification: str
    active_profile_id: str | None = None
    staged_bundle: Optional["StagedToolboxBundle"] = None
    registration: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.toolbox_id = str(self.toolbox_id or "").strip()
        if not self.toolbox_id:
            raise ValueError("resolved_assignment_toolbox_id_required")
        if self.classification not in {"reused", "added", "replaced"}:
            raise ValueError("resolved_assignment_classification_invalid")
        if self.bundle_spec.normalized_toolbox_id() != self.toolbox_id:
            raise ValueError("resolved_assignment_toolbox_mismatch")
        if self.bundle_spec.resolved_profile != self.resolved_profile:
            raise ValueError("resolved_assignment_profile_mismatch")
        if self.classification == "replaced" and not str(self.active_profile_id or "").strip():
            raise ValueError("resolved_assignment_active_profile_required")


@dataclass
class ToolboxBundleSpec:
    bundle_id: str
    toolbox_id: Optional[str] = None
    sandbox_profile: Optional[SandboxProfileSpec] = None
    files: List[ToolboxBundleFile] = field(default_factory=list)
    tools: List[ToolboxBundleTool] = field(default_factory=list)
    auto_tools: List[ToolboxBundleAutoTool] = field(default_factory=list)
    with_intrinsics: bool = False
    with_intrinsic_guides: bool = False
    intrinsic_tool_names: List[str] = field(default_factory=list)
    active_intrinsic_tool_names: List[str] = field(default_factory=list)
    hidden_intrinsic_tool_names: List[str] = field(default_factory=list)
    hidden_tool_names: List[str] = field(default_factory=list)
    dependency_lock_hash: Optional[str] = None
    resolved_profile: Optional[ResolvedToolboxProfileSpec] = None

    def normalized_bundle_id(self) -> str:
        raw = str(self.bundle_id or "").strip()
        if not raw:
            raise ValueError("bundle_id_required")
        return raw

    def normalized_toolbox_id(self) -> str:
        raw = str(self.toolbox_id or "").strip()
        return raw or self.normalized_bundle_id()

    def normalized_intrinsic_tool_names(self) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in list(self.intrinsic_tool_names or []):
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    @staticmethod
    def _normalize_name_list(items: Sequence[Any]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for item in list(items or []):
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def manifest_payload(self) -> Dict[str, Any]:
        bundle_id = self.normalized_bundle_id()
        toolbox_id = self.normalized_toolbox_id()
        sandbox_profile = (self.sandbox_profile or SandboxProfileSpec(profile_id="default")).to_dict()
        tools = [item.to_dict() for item in self.tools]
        auto_tools = [item.to_dict() for item in self.auto_tools]
        intrinsic_tool_names = self.normalized_intrinsic_tool_names()
        if not tools and not auto_tools and not intrinsic_tool_names:
            raise ValueError("bundle_tools_required")
        files = [item.to_dict() for item in self.files]
        active_intrinsic_tool_names = self._normalize_name_list(
            self.active_intrinsic_tool_names if self.active_intrinsic_tool_names else intrinsic_tool_names
        )
        hidden_intrinsic_tool_names = self._normalize_name_list(self.hidden_intrinsic_tool_names)
        hidden_tool_names = self._normalize_name_list(
            list(self.hidden_tool_names)
            + [item.tool_name() for item in list(self.tools or []) if bool(getattr(item, "hidden", False))]
            + [item.tool_name() for item in list(self.auto_tools or []) if bool(getattr(item, "hidden", False))]
        )
        manifest_input = {
            "bundle_id": bundle_id,
            "toolbox_id": toolbox_id,
            "sandbox_profile": sandbox_profile,
            "tools": tools,
            "auto_tools": auto_tools,
            "files": files,
            "with_intrinsics": bool(self.with_intrinsics or bool(intrinsic_tool_names)),
            "with_intrinsic_guides": bool(self.with_intrinsic_guides),
            "intrinsic_tool_names": intrinsic_tool_names,
            "active_intrinsic_tool_names": active_intrinsic_tool_names,
            "hidden_intrinsic_tool_names": hidden_intrinsic_tool_names,
            "hidden_tool_names": hidden_tool_names,
            "dependency_lock_hash": str(self.dependency_lock_hash or "").strip() or None,
        }
        if self.resolved_profile is not None:
            if not isinstance(self.resolved_profile, ResolvedToolboxProfileSpec):
                raise ValueError("bundle_resolved_profile_invalid")
            manifest_input["resolved_profile"] = self.resolved_profile.to_dict()
        manifest_hash = _sha256_text(_stable_json(manifest_input))
        bundle_revision = manifest_hash[:16]
        return {
            "executor_kind": "toolbox_executor",
            "bundle_id": bundle_id,
            "toolbox_id": toolbox_id,
            "sandbox_profile": sandbox_profile,
            "bundle_revision": bundle_revision,
            "manifest_hash": manifest_hash,
            "dependency_lock_hash": str(self.dependency_lock_hash or "").strip() or None,
            **(
                {"resolved_profile": self.resolved_profile.to_dict()}
                if self.resolved_profile is not None
                else {}
            ),
            "tools": tools,
            "auto_tools": auto_tools,
            "files": files,
            "with_intrinsics": bool(self.with_intrinsics or bool(intrinsic_tool_names)),
            "with_intrinsic_guides": bool(self.with_intrinsic_guides),
            "intrinsic_tool_names": intrinsic_tool_names,
            "active_intrinsic_tool_names": active_intrinsic_tool_names,
            "hidden_intrinsic_tool_names": hidden_intrinsic_tool_names,
            "hidden_tool_names": hidden_tool_names,
        }


@dataclass
class ToolboxWorkerStartupSpec:
    worker_id: str
    sandbox_id: str
    toolbox_revision: str
    manifest_path: str
    scratch_root: str
    engines_state_file: Optional[str] = None
    control_state_file: Optional[str] = None
    venv_path: Optional[str] = None
    ipc_family: str = field(default_factory=lambda: "AF_PIPE" if os.name == "nt" else "AF_UNIX")
    ipc_address: str = ""
    auth_token_env: str = "MP13_ENGINE_HOST_TOKEN"
    execution_contract: str = "hosting.toolbox.worker.v1"
    callback_contract: str = "hosting.toolbox.callbacks.v1"
    policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        default_ipc_family = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
        return {
            "worker_id": str(self.worker_id or "").strip(),
            "sandbox_id": str(self.sandbox_id or "").strip(),
            "toolbox_revision": str(self.toolbox_revision or "").strip(),
            "manifest_path": str(self.manifest_path or "").strip(),
            "scratch_root": str(self.scratch_root or "").strip(),
            "engines_state_file": str(self.engines_state_file or "").strip() or None,
            "control_state_file": str(self.control_state_file or "").strip() or None,
            "venv_path": str(self.venv_path or "").strip() or None,
            "ipc_family": str(self.ipc_family or default_ipc_family).strip() or default_ipc_family,
            "ipc_address": str(self.ipc_address or "").strip(),
            "auth_token_env": str(self.auth_token_env or "MP13_ENGINE_HOST_TOKEN").strip() or "MP13_ENGINE_HOST_TOKEN",
            "execution_contract": str(self.execution_contract or "hosting.toolbox.worker.v1").strip() or "hosting.toolbox.worker.v1",
            "callback_contract": str(self.callback_contract or "hosting.toolbox.callbacks.v1").strip() or "hosting.toolbox.callbacks.v1",
            "policy": dict(self.policy or {}),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ToolboxWorkerStartupSpec":
        row = dict(payload or {})
        default_ipc_family = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
        return cls(
            worker_id=str(row.get("worker_id") or "").strip(),
            sandbox_id=str(row.get("sandbox_id") or "").strip(),
            toolbox_revision=str(row.get("toolbox_revision") or "").strip(),
            manifest_path=str(row.get("manifest_path") or "").strip(),
            scratch_root=str(row.get("scratch_root") or "").strip(),
            engines_state_file=str(row.get("engines_state_file") or "").strip() or None,
            control_state_file=str(row.get("control_state_file") or "").strip() or None,
            venv_path=str(row.get("venv_path") or "").strip() or None,
            ipc_family=str(row.get("ipc_family") or default_ipc_family).strip() or default_ipc_family,
            ipc_address=str(row.get("ipc_address") or "").strip(),
            auth_token_env=str(row.get("auth_token_env") or "MP13_ENGINE_HOST_TOKEN").strip() or "MP13_ENGINE_HOST_TOKEN",
            execution_contract=str(row.get("execution_contract") or "hosting.toolbox.worker.v1").strip() or "hosting.toolbox.worker.v1",
            callback_contract=str(row.get("callback_contract") or "hosting.toolbox.callbacks.v1").strip() or "hosting.toolbox.callbacks.v1",
            policy=dict(row.get("policy") or {}),
        )

    def write_json(self, path: Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target


@dataclass
class ToolboxEnvironmentSpec:
    venv_key: str
    venv_path: str
    python_executable: str = ""
    environment_name: str = "base"
    environment_description_hash: str = ""
    venv_lock_hash: Optional[str] = None
    toolbox_runtime_hash: str = "toolbox-executor-v1"
    intrinsics_profile_id: str = "none"
    required_imports: List[str] = field(default_factory=list)
    dependency_lock_hash: Optional[str] = None
    environment_root_kind: str = "toolbox_venvs"
    environment_consumer_kind: str = "toolbox_executor"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venv_key": str(self.venv_key or "").strip(),
            "venv_path": str(self.venv_path or "").strip(),
            "python_executable": str(self.python_executable or "").strip(),
            "environment_name": str(self.environment_name or "base").strip() or "base",
            "environment_description_hash": str(self.environment_description_hash or "").strip() or None,
            "venv_lock_hash": str(self.venv_lock_hash or "").strip() or None,
            "toolbox_runtime_hash": str(self.toolbox_runtime_hash or "toolbox-executor-v1").strip() or "toolbox-executor-v1",
            "intrinsics_profile_id": str(self.intrinsics_profile_id or "none").strip() or "none",
            "required_imports": [str(item or "").strip() for item in list(self.required_imports or []) if str(item or "").strip()],
            "dependency_lock_hash": str(self.dependency_lock_hash or "").strip() or None,
            "environment_root_kind": str(self.environment_root_kind or "toolbox_venvs").strip() or "toolbox_venvs",
            "environment_consumer_kind": str(self.environment_consumer_kind or "toolbox_executor").strip() or "toolbox_executor",
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ToolboxEnvironmentSpec":
        row = dict(payload or {})
        return cls(
            venv_key=str(row.get("venv_key") or "").strip(),
            venv_path=str(row.get("venv_path") or "").strip(),
            python_executable=str(row.get("python_executable") or "").strip(),
            environment_name=str(row.get("environment_name") or "base").strip() or "base",
            environment_description_hash=str(row.get("environment_description_hash") or "").strip() or None,
            venv_lock_hash=str(row.get("venv_lock_hash") or "").strip() or None,
            toolbox_runtime_hash=str(row.get("toolbox_runtime_hash") or "toolbox-executor-v1").strip() or "toolbox-executor-v1",
            intrinsics_profile_id=str(row.get("intrinsics_profile_id") or "none").strip() or "none",
            required_imports=[str(item or "").strip() for item in list(row.get("required_imports") or []) if str(item or "").strip()],
            dependency_lock_hash=str(row.get("dependency_lock_hash") or "").strip() or None,
            environment_root_kind=str(row.get("environment_root_kind") or "toolbox_venvs").strip() or "toolbox_venvs",
            environment_consumer_kind=str(row.get("environment_consumer_kind") or "toolbox_executor").strip() or "toolbox_executor",
        )

@dataclass(frozen=True)
class ToolboxAutoAssignmentRequestV2:
    files: tuple[ToolboxBundleFile, ...]
    module_name: str
    callable_name: str
    dependency: ToolboxDependencyRequest
    sandbox_policy: Mapping[str, Any]
    activate: bool
    hidden: bool
    non_restartable: bool
    guide_content: Mapping[str, Any] | None
    guide_description: str | None
    callback_signature: Mapping[str, Any] | None
    concurrency: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        files = tuple(sorted(self.files, key=lambda item: item.normalized_path()))
        if not files or len({item.normalized_path() for item in files}) != len(files):
            raise ValueError("auto_request_files_invalid")
        object.__setattr__(self, "files", files)
        module = str(self.module_name or "").strip()
        callable_name = str(self.callable_name or "").strip()
        if not module or not _TOOL_NAME_RE.fullmatch(callable_name):
            raise ValueError("auto_request_entrypoint_invalid")
        object.__setattr__(self, "module_name", module)
        object.__setattr__(self, "callable_name", callable_name)
        if not isinstance(self.dependency, ToolboxDependencyRequest):
            raise ValueError("auto_request_dependency_invalid")
        object.__setattr__(self, "sandbox_policy", _canonical_mapping(self.sandbox_policy, label="auto_request_sandbox_policy"))
        for field_name in ("activate", "hidden", "non_restartable"):
            object.__setattr__(self, field_name, _strict_bool(getattr(self, field_name), label=f"auto_request_{field_name}"))
        for field_name in ("guide_content", "callback_signature", "concurrency"):
            object.__setattr__(self, field_name, _canonical_mapping(getattr(self, field_name), label=f"auto_request_{field_name}", nullable=True))
        description = None if self.guide_description is None else str(self.guide_description).strip()
        object.__setattr__(self, "guide_description", description or None)

    @property
    def stable_key(self) -> str:
        return f"{self.module_name}:{self.callable_name}"

    @property
    def advertised_name(self) -> str:
        return self.callable_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [item.to_runtime_dict() for item in self.files],
            "module_name": self.module_name,
            "callable_name": self.callable_name,
            "dependency": self.dependency.to_dict(),
            "sandbox_policy": dict(self.sandbox_policy),
            "activate": self.activate,
            "hidden": self.hidden,
            "non_restartable": self.non_restartable,
            "guide_content": dict(self.guide_content) if self.guide_content is not None else None,
            "guide_description": self.guide_description,
            "callback_signature": dict(self.callback_signature) if self.callback_signature is not None else None,
            "concurrency": dict(self.concurrency) if self.concurrency is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxAutoAssignmentRequestV2":
        row = dict(payload or {})
        fields = {"files", "module_name", "callable_name", "dependency", "sandbox_policy", "activate", "hidden", "non_restartable", "guide_content", "guide_description", "callback_signature", "concurrency"}
        _strict_model_fields(row, fields, label="toolbox_auto_request_v2")
        if not isinstance(row["files"], Sequence) or isinstance(row["files"], (str, bytes, bytearray)):
            raise ValueError("auto_request_files_list_required")
        return cls(
            files=tuple(ToolboxBundleFile.from_definition_dict(item) for item in row["files"]),
            module_name=row["module_name"],
            callable_name=row["callable_name"],
            dependency=ToolboxDependencyRequest.from_dict(row["dependency"]),
            sandbox_policy=row["sandbox_policy"],
            activate=row["activate"],
            hidden=row["hidden"],
            non_restartable=row["non_restartable"],
            guide_content=row["guide_content"],
            guide_description=row["guide_description"],
            callback_signature=row["callback_signature"],
            concurrency=row["concurrency"],
        )


@dataclass(frozen=True)
class ToolboxManualAssignmentRequestV2:
    files: tuple[ToolboxBundleFile, ...]
    module_name: str
    callable_name: str
    tool_definition: Mapping[str, Any]
    dependency: ToolboxDependencyRequest
    sandbox_policy: Mapping[str, Any]
    hidden: bool
    non_restartable: bool
    callback_signature: Mapping[str, Any] | None
    concurrency: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        files = tuple(sorted(self.files, key=lambda item: item.normalized_path()))
        if not files or len({item.normalized_path() for item in files}) != len(files):
            raise ValueError("manual_request_files_invalid")
        object.__setattr__(self, "files", files)
        module = str(self.module_name or "").strip()
        callable_name = str(self.callable_name or "").strip()
        if not module or not callable_name:
            raise ValueError("manual_request_entrypoint_invalid")
        object.__setattr__(self, "module_name", module)
        object.__setattr__(self, "callable_name", callable_name)
        definition = _canonical_mapping(self.tool_definition, label="manual_request_tool_definition")
        advertised = str(dict(dict(definition).get("function") or {}).get("name") or "").strip()
        if dict(definition).get("type") != "function" or not _TOOL_NAME_RE.fullmatch(advertised):
            raise ValueError("manual_request_tool_definition_invalid")
        object.__setattr__(self, "tool_definition", definition)
        if not isinstance(self.dependency, ToolboxDependencyRequest):
            raise ValueError("manual_request_dependency_invalid")
        object.__setattr__(self, "sandbox_policy", _canonical_mapping(self.sandbox_policy, label="manual_request_sandbox_policy"))
        for field_name in ("hidden", "non_restartable"):
            object.__setattr__(self, field_name, _strict_bool(getattr(self, field_name), label=f"manual_request_{field_name}"))
        for field_name in ("callback_signature", "concurrency"):
            object.__setattr__(self, field_name, _canonical_mapping(getattr(self, field_name), label=f"manual_request_{field_name}", nullable=True))

    @property
    def stable_key(self) -> str:
        return f"manual:{self.module_name}:{self.callable_name}"

    @property
    def advertised_name(self) -> str:
        return str(dict(self.tool_definition["function"])["name"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [item.to_runtime_dict() for item in self.files],
            "module_name": self.module_name,
            "callable_name": self.callable_name,
            "tool_definition": dict(self.tool_definition),
            "dependency": self.dependency.to_dict(),
            "sandbox_policy": dict(self.sandbox_policy),
            "hidden": self.hidden,
            "non_restartable": self.non_restartable,
            "callback_signature": dict(self.callback_signature) if self.callback_signature is not None else None,
            "concurrency": dict(self.concurrency) if self.concurrency is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxManualAssignmentRequestV2":
        row = dict(payload or {})
        fields = {"files", "module_name", "callable_name", "tool_definition", "dependency", "sandbox_policy", "hidden", "non_restartable", "callback_signature", "concurrency"}
        _strict_model_fields(row, fields, label="toolbox_manual_request_v2")
        if not isinstance(row["files"], Sequence) or isinstance(row["files"], (str, bytes, bytearray)):
            raise ValueError("manual_request_files_list_required")
        return cls(
            files=tuple(ToolboxBundleFile.from_definition_dict(item) for item in row["files"]),
            module_name=row["module_name"],
            callable_name=row["callable_name"],
            tool_definition=row["tool_definition"],
            dependency=ToolboxDependencyRequest.from_dict(row["dependency"]),
            sandbox_policy=row["sandbox_policy"],
            hidden=row["hidden"],
            non_restartable=row["non_restartable"],
            callback_signature=row["callback_signature"],
            concurrency=row["concurrency"],
        )


@dataclass(frozen=True)
class ToolboxDefinitionSpec:
    toolbox_id: str
    expected_revision: str | None
    auto_requests: tuple[ToolboxAutoAssignmentRequestV2, ...]
    manual_requests: tuple[ToolboxManualAssignmentRequestV2, ...]
    intrinsics: ToolboxIntrinsicSelection
    contract: str = TOOLBOX_DEFINITION_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != TOOLBOX_DEFINITION_CONTRACT:
            raise ValueError("toolbox_definition_contract_invalid")
        toolbox_id = str(self.toolbox_id or "").strip()
        if not _DEFINITION_ID_RE.fullmatch(toolbox_id):
            raise ValueError("toolbox_definition_id_invalid")
        object.__setattr__(self, "toolbox_id", toolbox_id)
        if self.expected_revision is not None:
            object.__setattr__(self, "expected_revision", require_digest(self.expected_revision, label="definition_expected_revision"))
        autos = tuple(sorted(self.auto_requests, key=lambda item: item.stable_key))
        manuals = tuple(sorted(self.manual_requests, key=lambda item: item.stable_key))
        if any(not isinstance(item, ToolboxAutoAssignmentRequestV2) for item in autos):
            raise ValueError("toolbox_definition_auto_request_invalid")
        if any(not isinstance(item, ToolboxManualAssignmentRequestV2) for item in manuals):
            raise ValueError("toolbox_definition_manual_request_invalid")
        if len({item.stable_key for item in autos}) != len(autos):
            raise ValueError("toolbox_definition_duplicate_auto_stable_key")
        if len({item.stable_key for item in manuals}) != len(manuals):
            raise ValueError("toolbox_definition_duplicate_manual_stable_key")
        object.__setattr__(self, "auto_requests", autos)
        object.__setattr__(self, "manual_requests", manuals)
        if not isinstance(self.intrinsics, ToolboxIntrinsicSelection):
            raise ValueError("toolbox_definition_intrinsics_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "toolbox_id": self.toolbox_id,
            "expected_revision": self.expected_revision,
            "auto_requests": [item.to_dict() for item in self.auto_requests],
            "manual_requests": [item.to_dict() for item in self.manual_requests],
            "intrinsics": self.intrinsics.to_dict(),
        }

    @property
    def revision(self) -> str:
        return definition_revision(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolboxDefinitionSpec":
        row = dict(payload or {})
        fields = {"contract", "toolbox_id", "expected_revision", "auto_requests", "manual_requests", "intrinsics"}
        _strict_model_fields(row, fields, label="toolbox_definition")
        for field_name in ("auto_requests", "manual_requests"):
            if not isinstance(row[field_name], Sequence) or isinstance(row[field_name], (str, bytes, bytearray)):
                raise ValueError(f"toolbox_definition_{field_name}_list_required")
        return cls(
            contract=row["contract"],
            toolbox_id=row["toolbox_id"],
            expected_revision=row["expected_revision"],
            auto_requests=tuple(ToolboxAutoAssignmentRequestV2.from_dict(item) for item in row["auto_requests"]),
            manual_requests=tuple(ToolboxManualAssignmentRequestV2.from_dict(item) for item in row["manual_requests"]),
            intrinsics=ToolboxIntrinsicSelection.from_dict(row["intrinsics"]),
        )

@dataclass
class ToolboxHarnessConfig:
    mode: str = "native"
    sandbox_toolbox_id: Optional[str] = None
    sandbox_engine_ids: List[str] = field(default_factory=list)
    sandbox_selection: str = "round_robin"
    max_concurrency: Optional[int] = None
