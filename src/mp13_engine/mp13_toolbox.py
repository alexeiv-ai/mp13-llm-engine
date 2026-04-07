# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json
import time
import asyncio
import inspect
import importlib
import re, copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Union, Set, Sequence
from typing import TYPE_CHECKING

# Import the Tool model for validation
from .mp13_config import RegisteredTool, Tool, ToolCall, ToolCallBlock, InferenceResponse
from .mp13_tools_parser import UnifiedToolIO

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession
    from .mp13_state import MP13State


def _get_tools_builtin_module():
    return importlib.import_module(".mp13_tools_builtin", __package__)


def _get_intrinsics_registry() -> Dict[str, RegisteredTool]:
    return _get_tools_builtin_module().INTRINSICS_REGISTRY


def _normalize_tool_constraints(
    raw: Optional[Dict[str, Any]],
    *,
    allow_clear: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tool_name, payload in dict(raw or {}).items():
        name = str(tool_name or "").strip()
        if not name:
            continue
        if payload is None and allow_clear:
            out[name] = None
            continue
        if not isinstance(payload, dict):
            continue
        out[name] = copy.deepcopy(dict(payload))
    return out


def _merge_tool_constraint_payload(base: Optional[Dict[str, Any]], overlay: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge tool constraint payloads shallowly enough for scope-stack composition.

    Rules:
    - top-level keys merge by section name
    - nested dict sections merge recursively
    - lists and scalars are replaced by the overlay value
    """
    left = dict(base or {})
    right = dict(overlay or {})
    out: Dict[str, Any] = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[str(key)] = _merge_tool_constraint_payload(
                dict(out.get(key) or {}),
                dict(value or {}),
            )
            continue
        out[str(key)] = copy.deepcopy(value)
    return out


def _normalize_scoped_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    text = raw.replace("\\", "/")
    drive_match = re.match(r"^(?P<drive>[A-Za-z]:)(?:/|$)", text)
    drive = (drive_match.group("drive").lower() if drive_match else "")
    remainder = text[len(drive):] if drive else text
    is_absolute = bool(drive) or remainder.startswith("/")
    parts: List[str] = []
    for part in remainder.split("/"):
        bit = part.strip()
        if not bit or bit == ".":
            continue
        if bit == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            elif not is_absolute:
                parts.append(bit)
            continue
        parts.append(bit)
    normalized = "/".join(parts)
    if drive:
        return f"{drive}/" + normalized if normalized else f"{drive}/"
    if is_absolute:
        return f"/{normalized}" if normalized else "/"
    return normalized


def _path_is_within(candidate: str, allowed_root: str) -> bool:
    candidate_norm = _normalize_scoped_path(candidate)
    root_norm = _normalize_scoped_path(allowed_root)
    if not root_norm:
        return False
    cand_abs = bool(re.match(r"^(?:[a-z]:/|/)", candidate_norm, re.IGNORECASE))
    root_abs = bool(re.match(r"^(?:[a-z]:/|/)", root_norm, re.IGNORECASE))
    if cand_abs != root_abs:
        return False
    cand_drive = re.match(r"^(?P<drive>[a-z]:)/", candidate_norm, re.IGNORECASE)
    root_drive = re.match(r"^(?P<drive>[a-z]:)/", root_norm, re.IGNORECASE)
    if bool(cand_drive) != bool(root_drive):
        return False
    if cand_drive and root_drive and cand_drive.group("drive").lower() != root_drive.group("drive").lower():
        return False
    return candidate_norm == root_norm or candidate_norm.startswith(root_norm.rstrip("/") + "/")


def _normalize_scoped_url(value: Any) -> str:
    return str(value or "").strip()


def _url_is_within(candidate: str, allowed_prefix: str) -> bool:
    cand = _normalize_scoped_url(candidate)
    prefix = _normalize_scoped_url(allowed_prefix)
    return bool(cand and prefix and cand.startswith(prefix))


def _apply_argument_normalizer(
    *,
    tool_name: str,
    arg_name: str,
    current_value: Any,
    normalizer_name: str,
    constraints: Dict[str, Any],
) -> Any:
    key = str(arg_name or "").strip()
    name = str(normalizer_name or "").strip()
    if not key or not name:
        return current_value
    if name == "path_under_implied_root":
        filesystem = dict(dict(constraints.get("domains") or {}).get("filesystem") or {})
        implied_root = _normalize_scoped_path(filesystem.get("implied_root"))
        allowed_roots = [
            _normalize_scoped_path(item)
            for item in list(filesystem.get("allowed_roots") or [])
            if _normalize_scoped_path(item)
        ]
        allow_override = bool(filesystem.get("allow_explicit_root_override", True))
        resolved = _normalize_scoped_path(current_value)
        if not resolved:
            resolved = implied_root
        if not resolved:
            return current_value
        if implied_root and not allow_override and resolved != implied_root:
            raise PermissionError(f"Tool '{tool_name}' argument '{key}' must stay under the implied root '{implied_root}'.")
        effective_roots = allowed_roots or ([implied_root] if implied_root else [])
        if effective_roots and not any(_path_is_within(resolved, root) for root in effective_roots):
            raise PermissionError(f"Tool '{tool_name}' argument '{key}' is outside the allowed scoped roots.")
        return resolved
    if name == "url_under_implied_prefix":
        network = dict(dict(constraints.get("domains") or {}).get("network") or {})
        implied_prefix = _normalize_scoped_url(network.get("implied_url_prefix"))
        allowed_prefixes = [
            _normalize_scoped_url(item)
            for item in list(network.get("allowed_url_prefixes") or [])
            if _normalize_scoped_url(item)
        ]
        allow_override = bool(network.get("allow_explicit_url_override", True))
        resolved = _normalize_scoped_url(current_value)
        if not resolved:
            resolved = implied_prefix
        if not resolved:
            return current_value
        if implied_prefix and not allow_override and not _url_is_within(resolved, implied_prefix):
            raise PermissionError(f"Tool '{tool_name}' argument '{key}' must stay under the implied URL prefix.")
        effective_prefixes = allowed_prefixes or ([implied_prefix] if implied_prefix else [])
        if effective_prefixes and not any(_url_is_within(resolved, prefix) for prefix in effective_prefixes):
            raise PermissionError(f"Tool '{tool_name}' argument '{key}' is outside the allowed scoped URL prefixes.")
        return resolved
    return current_value


def _resolved_tool_arguments(
    tool_name: str,
    tool_arguments: Any,
    tools_view: Optional["ToolsView"],
) -> Dict[str, Any]:
    if not isinstance(tool_arguments, dict):
        return dict(tool_arguments or {})
    resolved: Dict[str, Any] = dict(tool_arguments)
    if not tools_view:
        return resolved
    constraints = tools_view.get_constraints(tool_name)
    argument_policy = dict(constraints.get("argument_policy") or {})
    implied_args = dict(argument_policy.get("implied_args") or {})
    locked_args = {
        str(name).strip()
        for name in list(argument_policy.get("locked_args") or [])
        if str(name).strip()
    }
    normalizers = {
        str(name).strip(): str(kind).strip()
        for name, kind in dict(argument_policy.get("normalizers") or {}).items()
        if str(name).strip() and str(kind).strip()
    }
    for arg_name, implied_value in implied_args.items():
        key = str(arg_name or "").strip()
        if not key:
            continue
        if key in locked_args and key in resolved and resolved.get(key) != implied_value:
            raise PermissionError(f"Tool '{tool_name}' argument '{key}' is locked by scope constraints.")
        if key not in resolved:
            resolved[key] = copy.deepcopy(implied_value)
    for arg_name, normalizer_name in normalizers.items():
        resolved[arg_name] = _apply_argument_normalizer(
            tool_name=tool_name,
            arg_name=arg_name,
            current_value=resolved.get(arg_name),
            normalizer_name=normalizer_name,
            constraints=constraints,
        )
    return resolved


@dataclass
class ToolsScope:
    """
    Describes a scoped mutation of tool permissions/visibility.

    Attributes:
        mode: Optional override of the default toolbox mode. Supported values:
              "advertised", "silent", "disabled".
        advertise_tools: Names that must be advertised (even if hidden globally).
        silent_tools: Names that must stay enabled but hidden from the LLM.
        disabled_tools: Names that must be disabled for this scope.
        gated_tools: Names that require explicit confirmation before execution.
        tool_constraints: Per-tool dynamic contextual narrowing payloads.
    """
    mode: Optional[str] = None
    advertise_tools: Set[str] = field(default_factory=set)
    silent_tools: Set[str] = field(default_factory=set)
    disabled_tools: Set[str] = field(default_factory=set)
    gated_tools: Set[str] = field(default_factory=set)
    tool_constraints: Dict[str, Optional[Dict[str, Any]]] = field(default_factory=dict)
    label: Optional[str] = None

    DEFAULT_MODE = "*"
    VALID_MODES = {"advertised", "silent", "disabled", DEFAULT_MODE}

    def clean(self) -> "ToolsScope":
        """Normalizes tool name casing and removes empty strings."""
        def _normalize(items: Set[str]) -> Set[str]:
            return {name.strip() for name in items if isinstance(name, str) and name.strip()}

        self.advertise_tools = _normalize(self.advertise_tools)
        self.silent_tools = _normalize(self.silent_tools)
        self.disabled_tools = _normalize(self.disabled_tools)
        self.gated_tools = _normalize(self.gated_tools)
        self.tool_constraints = _normalize_tool_constraints(self.tool_constraints, allow_clear=True)
        if self.mode and self.mode not in self.VALID_MODES:
            raise ValueError(f"ToolsScope.mode '{self.mode}' is invalid. Allowed: {sorted(self.VALID_MODES)}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "advertise_tools": sorted(list(self.advertise_tools)),
            "silent_tools": sorted(list(self.silent_tools)),
            "disabled_tools": sorted(list(self.disabled_tools)),
            "gated_tools": sorted(list(self.gated_tools)),
            "tool_constraints": copy.deepcopy(self.tool_constraints),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ToolsScope":
        if not data:
            return cls()
        return cls(
            mode=data.get("mode"),
            advertise_tools=set(data.get("advertise_tools", data.get("advertise", [])) or []),
            silent_tools=set(data.get("silent_tools", data.get("silent", [])) or []),
            disabled_tools=set(data.get("disabled_tools", data.get("disabled", [])) or []),
            gated_tools=set(data.get("gated_tools", data.get("gated", [])) or []),
            tool_constraints=_normalize_tool_constraints(data.get("tool_constraints"), allow_clear=True),
            label=data.get("label"),
        ).clean()

    def describe(self) -> str:
        """Returns a concise string description for logging/debugging."""
        bits = []
        if self.mode:
            bits.append(f"mode={self.mode}")
        if self.advertise_tools:
            bits.append(f"adv={','.join(sorted(self.advertise_tools))}")
        if self.silent_tools:
            bits.append(f"silent={','.join(sorted(self.silent_tools))}")
        if self.disabled_tools:
            bits.append(f"disabled={','.join(sorted(self.disabled_tools))}")
        if self.gated_tools:
            bits.append(f"gated={','.join(sorted(self.gated_tools))}")
        if self.tool_constraints:
            bits.append(f"constraints={','.join(sorted(self.tool_constraints.keys()))}")
        return " | ".join(bits) if bits else "no-op scope"

    def is_noop(self) -> bool:
        return not (self.mode or self.advertise_tools or self.silent_tools or self.disabled_tools or self.gated_tools or self.tool_constraints)


@dataclass
class ToolsView:
    """Materialized permissions/advertisement view for a particular turn/request."""
    view_id: str
    mode: str
    allowed_tools: Set[str]
    advertised_tools: Set[str]
    hidden_allowed_tools: Set[str]
    disabled_tools: Set[str]
    gated_tools: Set[str] = field(default_factory=set)
    tool_constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Ensures that upon deserialization from a dict (where these might be lists),
        the fields are converted back to sets for runtime efficiency."""
        self.allowed_tools = set(self.allowed_tools)
        self.advertised_tools = set(self.advertised_tools)
        self.hidden_allowed_tools = set(self.hidden_allowed_tools)
        self.disabled_tools = set(self.disabled_tools)
        self.gated_tools = set(self.gated_tools)
        self.tool_constraints = _normalize_tool_constraints(self.tool_constraints)

    def should_advertise(self, tool_name: str) -> bool:
        return tool_name in self.advertised_tools

    def is_allowed(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools

    def is_disabled(self, tool_name: str) -> bool:
        return tool_name in self.disabled_tools

    def is_gated(self, tool_name: str) -> bool:
        return tool_name in self.gated_tools

    def get_constraints(self, tool_name: str) -> Dict[str, Any]:
        return copy.deepcopy(dict(self.tool_constraints.get(str(tool_name or "").strip()) or {}))


@dataclass(frozen=True)
class ToolConstraintsView:
    """Read-only helper wrapper around the resolved per-tool constraint payload."""
    tool_name: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_name", str(self.tool_name or "").strip())
        object.__setattr__(self, "payload", _normalize_tool_constraints({self.tool_name or "_": self.payload}).get(self.tool_name or "_", {}))

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.payload)

    def get_domain(self, domain_name: str) -> Dict[str, Any]:
        domains = dict(self.payload.get("domains") or {})
        return copy.deepcopy(dict(domains.get(str(domain_name or "").strip()) or {}))

    def get_argument_policy(self) -> Dict[str, Any]:
        return copy.deepcopy(dict(self.payload.get("argument_policy") or {}))

    def get_implied_arg(self, arg_name: str, default: Any = None) -> Any:
        implied = dict(self.get_argument_policy().get("implied_args") or {})
        key = str(arg_name or "").strip()
        if not key:
            return default
        value = implied.get(key, default)
        return copy.deepcopy(value)

    def is_arg_locked(self, arg_name: str) -> bool:
        key = str(arg_name or "").strip()
        if not key:
            return False
        locked = {str(name).strip() for name in list(self.get_argument_policy().get("locked_args") or []) if str(name).strip()}
        return key in locked

    def get_normalizer(self, arg_name: str) -> Optional[str]:
        key = str(arg_name or "").strip()
        if not key:
            return None
        normalizers = {
            str(name).strip(): str(value).strip()
            for name, value in dict(self.get_argument_policy().get("normalizers") or {}).items()
            if str(name).strip() and str(value).strip()
        }
        return normalizers.get(key)

    def resolve_argument(self, arg_name: str, value: Any = None) -> Any:
        key = str(arg_name or "").strip()
        if not key:
            return value
        resolved = copy.deepcopy(value)
        implied_value = self.get_implied_arg(key, None)
        if self.is_arg_locked(key) and resolved is not None and implied_value is not None and resolved != implied_value:
            raise PermissionError(f"Tool '{self.tool_name}' argument '{key}' is locked by scope constraints.")
        if resolved is None and implied_value is not None:
            resolved = implied_value
        normalizer = self.get_normalizer(key)
        if normalizer:
            resolved = _apply_argument_normalizer(
                tool_name=self.tool_name,
                arg_name=key,
                current_value=resolved,
                normalizer_name=normalizer,
                constraints=self.payload,
            )
        return copy.deepcopy(resolved)

    def resolve_filesystem_root(self, value: Any = None, *, arg_name: str = "root_path") -> str:
        resolved = self.resolve_argument(arg_name, value)
        return str(resolved or "")

    def resolve_url(self, value: Any = None, *, arg_name: str = "url") -> str:
        resolved = self.resolve_argument(arg_name, value)
        return str(resolved or "")


@dataclass
class ToolsAccess:
    """
    Lightweight wrapper around Toolbox to memoize/reuse a ToolsView.
    Useful for serialization and execution time lookups.
    """
    toolbox: "Toolbox"
    scopes: List[ToolsScope] = field(default_factory=list)
    label: Optional[str] = None
    _view: Optional[ToolsView] = None

    def get_view(self) -> ToolsView:
        if not self._view:
            self._view = self.toolbox.build_view(self.scopes, label=self.label)
        return self._view


@dataclass
class ToolBoxRef:
    """
    Tracks the toolbox instance paired with a persistent ToolsScope snapshot.
    Subclasses can override `_scope_updated` to hook into persistence flows.
    """
    toolbox: "Toolbox"
    scope: ToolsScope = field(default_factory=ToolsScope)

    def snapshot_scope(self) -> ToolsScope:
        """Return a deep copy of the tracked scope for safe mutation."""
        return copy.deepcopy(self.scope)

    def set_scope(self, scope: Optional[ToolsScope]) -> ToolsScope:
        """Replace the stored scope and dispatch persistence hooks."""
        self.scope = (scope or ToolsScope()).clean()
        self._scope_updated()
        return self.scope

    def mutate_scope(self, mutator: Callable[[ToolsScope], ToolsScope]) -> ToolsScope:
        """Apply a mutation function to the stored scope."""
        base = self.snapshot_scope()
        updated = mutator(base) if mutator else base
        return self.set_scope(updated or ToolsScope())

    def _scope_updated(self) -> None:
        """Template method for subclasses that need to persist scope changes."""
        return


@dataclass
class ToolCallGate:
    outcome: str
    tool_name: str
    reason: str
    executable: bool
    requires_confirmation: bool = False
    backend: str = "native"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": str(self.outcome or "").strip() or "denied",
            "tool_name": str(self.tool_name or "").strip(),
            "reason": str(self.reason or "").strip() or "denied",
            "executable": bool(self.executable),
            "requires_confirmation": bool(self.requires_confirmation),
            "backend": str(self.backend or "native").strip() or "native",
        }


class Toolbox:
    """Manages tool definitions for the chat application."""
    _VALID_GLOBAL_MODES = {"advertised", "silent", "disabled"}

    def __init__(self, with_intrinsics: bool = False, with_intrinsic_guides: bool = False):
        self.with_intrinsics = bool(with_intrinsics)
        self.with_intrinsic_guides = bool(with_intrinsic_guides)
        self.tools: Dict[str, Dict[str, Any]] = {}  # User-defined tools from JSON
#        self.prompt_header: Optional[str] = None
#        self.prompt_footer: Optional[str] = None
#        self.tool_footers: Dict[str, str] = {}
        self.active_tool_names: List[str] = []
        self.hidden_tool_names: List[str] = []

        self.intrinsic_overrides: Dict[str, Dict[str, Any]] = {}
        self.hidden_intrinsic_tool_names: List[str] = []

        # New additions for intrinsic tools
        self.intrinsic_tools: Dict[str, Dict[str, Any]] = {}
        self.intrinsic_tool_callables: Dict[str, Callable[..., Any]] = {}
        self.active_intrinsic_tool_names: List[str] = []
        self.user_tool_callables: Dict[str, Callable[..., Any]] = {}
        self.global_tools_mode: str = "advertised"  # advertised | silent | disabled
        self._view_seq: int = 0

        if self.with_intrinsics:
            self._initialize_intrinsic_tools(include_guides=self.with_intrinsic_guides)
        self._create_default_state() # Initialize with default state

    def _normalize_mode(self, mode: Optional[str]) -> str:
        """Validates and normalizes toolbox-wide tool modes."""
        if not mode:
            return "advertised"
        if mode not in self._VALID_GLOBAL_MODES:
            raise ValueError(f"Invalid toolbox mode '{mode}'. Valid: {sorted(self._VALID_GLOBAL_MODES)}")
        return mode

    def set_global_tools_mode(self, mode: str) -> None:
        """Sets the global tools mode (advertised, silent, disabled)."""
        normalized = self._normalize_mode(mode)
        self.global_tools_mode = normalized

    def create_access(self, scopes: Optional[List[ToolsScope]] = None, label: Optional[str] = None) -> ToolsAccess:
        """Creates a ToolsAccess wrapper for the provided scopes."""
        normalized_scopes = [scope.clean() for scope in (scopes or [])]
        return ToolsAccess(toolbox=self, scopes=normalized_scopes, label=label)

    def _resolve_intrinsic_targets(self, intrinsic_names: Optional[Union[str, Sequence[str]]]) -> Tuple[Optional[Set[str]], List[str]]:
        if intrinsic_names is None:
            return None, []
        targets = {str(n).strip() for n in ([intrinsic_names] if isinstance(intrinsic_names, str) else intrinsic_names) if str(n).strip()}
        if not targets:
            return set(), []
        registry = _get_intrinsics_registry()
        known: Set[str] = set()
        for container in registry.values():
            known.add(container.name)
            if container.guide_definition:
                known.add(container.guide_definition["function"]["name"])
        missing = sorted([name for name in targets if name not in known])
        valid_targets = {name for name in targets if name in known}
        return valid_targets, missing

    def available_intrinsics(self, include_guides: bool = True) -> List[Dict[str, Any]]:
        """
        Returns discoverable intrinsic tool metadata for registration UX.
        """
        items: List[Dict[str, Any]] = []
        for container in _get_intrinsics_registry().values():
            base_def = container.definition or {}
            base_fn = base_def.get("function", {})
            items.append({
                "name": container.name,
                "is_guide": False,
                "parent": None,
                "has_guide": bool(container.guide_definition and container.guide_content),
                "description": base_fn.get("description", ""),
            })
            if include_guides and container.guide_definition and container.guide_content:
                guide_name = container.guide_definition.get("function", {}).get("name")
                if guide_name:
                    items.append({
                        "name": guide_name,
                        "is_guide": True,
                        "parent": container.name,
                        "has_guide": False,
                        "description": container.guide_definition.get("function", {}).get("description", ""),
                    })
        return sorted(items, key=lambda x: x["name"])

    def _initialize_intrinsic_tools(
        self,
        *,
        include_guides: bool = False,
        intrinsic_names: Optional[Union[str, Sequence[str]]] = None,
    ):
        """Defines the schema and callables for built-in tools."""
        targets, _ = self._resolve_intrinsic_targets(intrinsic_names)
        # Load intrinsic tools from the central registry and unwrap them.
        for name, tool_container in _get_intrinsics_registry().items():
            guide_name = tool_container.guide_definition["function"]["name"] if tool_container.guide_definition else None
            if targets is not None and tool_container.name not in targets and (not guide_name or guide_name not in targets):
                continue
            include_base = (targets is None and tool_container.name not in self.intrinsic_tools) or (targets is not None and tool_container.name in targets)
            if include_base:
                base_definition = copy.deepcopy(dict(tool_container.definition or {}))
                if tool_container.guide_content:
                    base_definition["guide_content"] = copy.deepcopy(dict(tool_container.guide_content))
                self.intrinsic_tools[tool_container.name] = base_definition
                self.intrinsic_tool_callables[tool_container.name] = tool_container.implementation
            if tool_container.guide_definition and tool_container.guide_content:
                include_this_guide = bool(include_guides)
                if guide_name and targets is not None and guide_name in targets:
                    include_this_guide = True
                if include_this_guide:
                    guide_name = tool_container.guide_definition["function"]["name"]
                    guide_definition = copy.deepcopy(dict(tool_container.guide_definition or {}))
                    guide_definition["guide_content"] = copy.deepcopy(dict(tool_container.guide_content or {}))
                    self.intrinsic_tools[guide_name] = guide_definition

    def from_dict(self, data: Dict[str, Any], search_scope: Optional[Dict[str, Any]] = None, external_handler: Optional[Callable[..., Any]] = None): # noqa
        """Loads tool state from a dictionary and re-links callables."""
        self.with_intrinsics = bool(data.get("with_intrinsics", self.with_intrinsics))
        self.with_intrinsic_guides = bool(data.get("with_intrinsic_guides", self.with_intrinsic_guides))
        loaded_intrinsic_tools = data.get("loaded_intrinsic_tools")

        if loaded_intrinsic_tools is None:
            inferred: Set[str] = set(data.get("active_intrinsic_tools", []) or [])
            inferred.update(data.get("hidden_intrinsic_tools", []) or [])
            intrinsic_overrides = data.get("intrinsic_overrides", {}) or {}
            for base_name in intrinsic_overrides.keys():
                inferred.add(base_name)
                inferred.add(f"{base_name}_guide")
            loaded_intrinsic_tools = sorted(inferred) if inferred else None
            if inferred and "with_intrinsics" not in data:
                self.with_intrinsics = True
            if any(str(name).endswith("_guide") for name in inferred) and "with_intrinsic_guides" not in data:
                self.with_intrinsic_guides = True

        self.intrinsic_tools = {}
        self.intrinsic_tool_callables = {}
        if self.with_intrinsics:
            self._initialize_intrinsic_tools(
                include_guides=self.with_intrinsic_guides,
                intrinsic_names=loaded_intrinsic_tools,
            )

        self.tools = data.get("tools", {})
#        self.prompt_header = data.get("prompt_header")
#        self.prompt_footer = data.get("prompt_footer")
#        self.tool_footers = data.get("tool_footers", {})
        self.intrinsic_overrides = data.get("intrinsic_overrides", {})
        self.active_tool_names = data.get("active_tools", [])
        self.hidden_tool_names = data.get("hidden_tools", [])
        self.active_intrinsic_tool_names = data.get("active_intrinsic_tools", list(self.intrinsic_tools.keys()))
        self.hidden_intrinsic_tool_names = data.get("hidden_intrinsic_tools", [])
        self.active_intrinsic_tool_names = [n for n in self.active_intrinsic_tool_names if n in self.intrinsic_tools]
        self.hidden_intrinsic_tool_names = [n for n in self.hidden_intrinsic_tool_names if n in self.intrinsic_tools]
        self.global_tools_mode = self._normalize_mode(data.get("global_tools_mode", self.global_tools_mode))

        # Re-link callables for user-defined tools and determine their type.
        # This logic is now more robustly handled during ChatSession deserialization,
        # where the correct search_scope (from initial_params) is available.
        self.relink_user_tool_callables(search_scope=search_scope, external_handler=external_handler)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the current toolbox state to a dictionary."""
        return {
#            "prompt_header": self.prompt_header,
#            "prompt_footer": self.prompt_footer,
#            "tool_footers": self.tool_footers,
            "tools": self.tools,
            "intrinsic_overrides": self.intrinsic_overrides,
            "active_tools": self.active_tool_names,
            "hidden_tools": self.hidden_tool_names,
            "active_intrinsic_tools": self.active_intrinsic_tool_names,
            "hidden_intrinsic_tools": self.hidden_intrinsic_tool_names,
            "loaded_intrinsic_tools": sorted(list(self.intrinsic_tools.keys())),
            "with_intrinsics": self.with_intrinsics,
            "with_intrinsic_guides": self.with_intrinsic_guides,
            "global_tools_mode": self.global_tools_mode,
        }

    def relink_user_tool_callables(self, search_scope: Optional[Dict[str, Any]] = None, external_handler: Optional[Callable[..., Any]] = None) -> None:
        """Re-links callables for user-defined tools after loading."""
        self.user_tool_callables.clear()
        for name in self.tools.keys():
            tool_def = self.tools[name]
            original_type = tool_def.get("_type")

            if original_type == "callable" and search_scope and name in search_scope and callable(search_scope[name]):
                self.user_tool_callables[name] = search_scope[name]
                tool_def["_type"] = "callable"
            elif original_type == "external" and external_handler:
                self.user_tool_callables[name] = external_handler
                tool_def["_type"] = "external"
            else:
                # If the implementation cannot be found, it becomes unresolved.
                if original_type == "callable":
                    print(f"Warning: Callable tool '{name}' is unresolved. Its Python implementation was not found.")
                elif original_type == "external":
                    print(f"Warning: External tool '{name}' is unresolved because no external_handler was provided.")
                
                # The tool definition exists, but its implementation is not found.
                tool_def["_type"] = "unresolved"



    def _create_default_state(self):
        """Initializes the toolbox to a default, empty state."""
        self.tools = {}
#        self.prompt_header = None
#        self.prompt_footer = None
#        self.tool_footers = {}
        self.intrinsic_overrides = {}
        self.active_tool_names = []
        self.hidden_tool_names = []
        # Default to all intrinsic tools being active
        self.active_intrinsic_tool_names = list(self.intrinsic_tools.keys())
        self.hidden_intrinsic_tool_names = [] # Default to none being hidden
        self.global_tools_mode = "advertised"
        self._view_seq = 0

    def list_tools(self) -> List[Tuple[str, str, str, bool, bool, bool, bool]]:
        """Returns a list of (name, description, type, is_active, is_hidden, is_guide, is_modified) tuples."""
        managed_tools: Dict[str, Dict[str, Any]] = {}

        # User-defined tools
        for name, definition in self.tools.items():
            managed_tools[name] = {
                "description": definition.get("function", {}).get("description", ""),
                "type": definition.get("_type", "external"), # Use the resolved type
                "is_active": name in self.active_tool_names,
                "is_hidden": name in self.hidden_tool_names,
                "is_intrinsic": False,
                "is_guide": False,
                "is_modified": False, # User tools are always "modified" by definition, but marker is for intrinsics
            }

        # Intrinsic tools
        if self.with_intrinsics:
            for tool_container in _get_intrinsics_registry().values():
                guide_name = tool_container.guide_definition["function"]["name"] if tool_container.guide_definition else None
                has_base = tool_container.name in self.intrinsic_tools
                has_guide = bool(guide_name and guide_name in self.intrinsic_tools)
                if not has_base and not has_guide:
                    continue
                # Process main tool
                if has_base and tool_container.name not in managed_tools:
                    override = self.intrinsic_overrides.get(tool_container.name, {})
                    base_desc = tool_container.definition.get("function", {}).get("description", "No description.")
                    managed_tools[tool_container.name] = {
                        "description": override.get("description", base_desc),
                        "type": "intrinsic",
                        "is_active": tool_container.name in self.active_intrinsic_tool_names,
                        "is_hidden": tool_container.name in self.hidden_intrinsic_tool_names,
                        "is_intrinsic": True,
                        "is_guide": False,
                        "is_modified": tool_container.name in self.intrinsic_overrides,
                    }
                # Process guide tool if it exists
                if tool_container.guide_definition:
                    if guide_name in self.intrinsic_tools and guide_name not in managed_tools:
                        # Guides can't be modified directly, but their content comes from the parent tool's override
                        parent_override = self.intrinsic_overrides.get(tool_container.name, {})
                        base_guide_desc = tool_container.guide_definition.get("function", {}).get("description", "No description.")
                        managed_tools[guide_name] = {
                            "description": parent_override.get("guide_description", base_guide_desc),
                            "type": "intrinsic",
                            "is_active": guide_name in self.active_intrinsic_tool_names,
                            "is_hidden": guide_name in self.hidden_intrinsic_tool_names,
                            "is_intrinsic": True,
                            "is_guide": True,
                            "is_modified": tool_container.name in self.intrinsic_overrides, # Guide is modified if parent is
                        }
        
        # Add user-defined guides to the list
        for name, definition in self.tools.items():
            if "guide_definition" in definition:
                guide_name = definition["guide_definition"]["function"]["name"]
                if guide_name not in managed_tools:
                     managed_tools[guide_name] = {
                        "description": definition["guide_definition"]["function"].get("description", "No description."),
                        "type": "callable",
                        "is_active": guide_name in self.active_tool_names,
                        "is_hidden": guide_name in self.hidden_tool_names,
                        "is_intrinsic": False,
                        "is_guide": True,
                        "is_modified": False,
                    }

        tool_list = []
        for name, data in managed_tools.items():
            tool_list.append((name, data['description'], data['type'], data['is_active'], data['is_hidden'], data['is_guide'], data['is_modified']))
        return sorted(tool_list, key=lambda x: x[0])

    def _is_hidden(self, name: str) -> bool:
        if name in self.intrinsic_tools or name in self.intrinsic_tool_callables:
            return name in self.hidden_intrinsic_tool_names
        return name in self.hidden_tool_names

    def _active_tool_names(self) -> Set[str]:
        """Returns the set of tool names currently marked as active (user + intrinsic)."""
        active = set(self.active_tool_names or [])
        active.update(self.active_intrinsic_tool_names or [])
        for tool_name in list(self.active_tool_names or []):
            tool_def = self.tools.get(tool_name) or {}
            guide_name = str(dict(dict(tool_def).get("guide_definition") or {}).get("function", {}).get("name") or "").strip()
            if guide_name:
                active.add(guide_name)
        return active

    def _registered_tool_names(self) -> Set[str]:
        """Returns all currently registered tool/guide names in the toolbox namespace."""
        names: Set[str] = set(self.tools.keys())
        names.update(self.intrinsic_tools.keys())
        for tool_def in self.tools.values():
            guide_name = tool_def.get("guide_definition", {}).get("function", {}).get("name")
            if isinstance(guide_name, str) and guide_name:
                names.add(guide_name)
        return names

    def _guide_content_for_tool_name(self, name: str) -> Optional[Dict[str, Any]]:
        guide_name = str(name or "").strip()
        if not guide_name:
            return None
        direct_intrinsic = dict(self.intrinsic_tools.get(guide_name) or {})
        direct_content = dict(direct_intrinsic.get("guide_content") or {})
        if direct_content:
            return copy.deepcopy(direct_content)
        for tool_def in list(self.tools.values()) + list(self.intrinsic_tools.values()):
            guide_def = dict(tool_def.get("guide_definition") or {})
            linked_name = str(dict(guide_def.get("function") or {}).get("name") or "").strip()
            if linked_name != guide_name:
                continue
            content = dict(tool_def.get("guide_content") or {}) or dict(guide_def.get("guide_content") or {})
            if content:
                return copy.deepcopy(content)
        return None

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Gets the full definition of a tool by name."""
        direct = self.tools.get(name) or self.intrinsic_tools.get(name)
        if direct:
            return direct
        for tool_def in self.tools.values():
            guide_def = dict(tool_def.get("guide_definition") or {})
            guide_name = str(dict(guide_def.get("function") or {}).get("name") or "").strip()
            if guide_name == str(name or "").strip():
                return guide_def
        return None

    def build_view(self, scopes: Optional[List[ToolsScope]] = None, label: Optional[str] = None) -> ToolsView:
        """
        Builds a ToolsView by applying the supplied scopes on top of the global toolbox mode.
        """
        scopes = scopes or []
        cleaned_scopes = [scope.clean() for scope in scopes]
        normalized_mode = self._normalize_mode(self.global_tools_mode)
        effective_mode = normalized_mode
        for scope in cleaned_scopes:
            if scope.mode == ToolsScope.DEFAULT_MODE:
                effective_mode = normalized_mode
                continue
            if scope.mode:
                effective_mode = scope.mode

        active_names = self._active_tool_names()
        visible_names: Set[str] = set()
        hidden_names: Set[str] = set()
        disabled_names: Set[str] = set(active_names if effective_mode == "disabled" else [])
        gated_names: Set[str] = set()
        effective_constraints: Dict[str, Dict[str, Any]] = {}

        if effective_mode != "disabled":
            for name in active_names:
                if effective_mode == "advertised" and not self._is_hidden(name):
                    visible_names.add(name)
                else:
                    hidden_names.add(name)

        def resolve_targets(targets: Set[str]) -> Set[str]:
            if not targets:
                return set()
            if "*" in targets:
                return set(active_names)
            return {t for t in targets if t in active_names}

        def apply_status(names: Set[str], status: str, scope_index: int) -> None:
            resolved = resolve_targets(names)
            if not resolved:
                return
            if status == "disabled":
                disabled_names.update(resolved)
                visible_names.difference_update(resolved)
                hidden_names.difference_update(resolved)
                gated_names.difference_update(resolved)
                return
            resolved -= disabled_names
            if not resolved:
                return
            if status == "advertised":
                visible_names.update(resolved)
                hidden_names.difference_update(resolved)
                return
            if status == "silent":
                hidden_names.update(resolved)
                visible_names.difference_update(resolved)
                return
            raise ValueError(f"unsupported_tool_status:{status}")

        for idx, scope in enumerate(cleaned_scopes, start=1):
            apply_status(scope.disabled_tools, "disabled", idx)
            apply_status(scope.advertise_tools, "advertised", idx)
            apply_status(scope.silent_tools, "silent", idx)
            gated_names.update(resolve_targets(scope.gated_tools) - disabled_names)
            for tool_name, payload in dict(scope.tool_constraints or {}).items():
                name = str(tool_name or "").strip()
                if name not in active_names or name in disabled_names:
                    continue
                if payload is None:
                    effective_constraints.pop(name, None)
                    continue
                if isinstance(payload, dict):
                    effective_constraints[name] = _merge_tool_constraint_payload(
                        effective_constraints.get(name),
                        dict(payload),
                    )

        allowed: Set[str] = set()
        advertised: Set[str] = set()
        disabled: Set[str] = set()
        for name in active_names:
            if name in disabled_names:
                disabled.add(name)
                continue
            if name in visible_names:
                advertised.add(name)
            if name not in gated_names:
                allowed.add(name)

        hidden_allowed = (allowed - advertised)
        self._view_seq += 1
        view_id = label or f"tools-view-{self._view_seq}"

        disabled.update(set(active_names) - allowed)
        disabled.difference_update(gated_names)

        return ToolsView(
            view_id=view_id,
            mode=effective_mode,
            allowed_tools=allowed,
            advertised_tools=advertised,
            hidden_allowed_tools=hidden_allowed,
            disabled_tools=disabled,
            gated_tools=gated_names,
            tool_constraints=effective_constraints,
        )

    def resolve_tool_link(self, name: str, search_scope: Optional[Dict[str, Any]] = None, external_handler: Optional[Callable[..., Any]] = None) -> Tuple[bool, str]: # noqa
        """
        Attempts to fix an 'unresolved' tool. The resolution strategy depends on the arguments provided:
        - To fix as 'callable' only: Provide `search_scope`, omit `external_handler`. Fails if not found in scope.
        - To fix as 'external' only: Provide `external_handler`, omit `search_scope`.
        - To fix as 'callable' with 'external' as a fallback: Provide both `search_scope` and `external_handler`.
        """
        tool_def = self.tools.get(name)
        if not tool_def:
            return False, f"Tool '{name}' not found."
        if tool_def.get("_type") != "unresolved":
            return False, f"Tool '{name}' is not unresolved. Its type is '{tool_def.get('_type', 'unknown')}'."
        if not search_scope and not external_handler:
            return False, "Resolution failed: Either a search_scope or an external_handler must be provided."

        # Preserve the original active/shown state before attempting to fix.
        was_active = name in self.active_tool_names # noqa
        was_hidden = name in self.hidden_tool_names

        # Priority 1: Attempt to re-link to a Python callable if scope is provided.
        if search_scope and name in search_scope and callable(search_scope[name]):
            self.user_tool_callables[name] = search_scope[name]
            tool_def["_type"] = "callable"
            msg = f"Tool '{name}' has been successfully re-linked to its Python function."
        # Priority 2: Fallback to converting to an external tool if a handler is provided.
        elif external_handler:
            self.user_tool_callables[name] = external_handler
            tool_def["_type"] = "external"
            msg = f"Tool '{name}' has been converted to an external tool."
            if search_scope: # This means the callable was not found, and we fell back.
                msg = f"Could not find a Python function for '{name}'. " + msg
        else:
            return False, f"Cannot fix tool '{name}'. No Python function found in scope and no external_handler provided."

        # Restore the original active/shown state
        if was_active and name not in self.active_tool_names: self.active_tool_names.append(name) # noqa
        if not was_hidden and name in self.hidden_tool_names: self.hidden_tool_names.remove(name)

        return True, msg

    def _update_tool_internal(self, name: str, new_definition: Dict[str, Any], external_handler: Callable[..., Any]) -> Tuple[bool, str]:
        """Internal method to update or create a tool. Overwrites if exists."""
        # For tools defined via the editor or JSON, we register the default handler
        # (which prompts the user for input) as their implementation.
        # We reuse add_tool_external for validation and saving.
        return self.add_tool_external(new_definition, external_handler, activate=None, allow_override=True)
    
    def delete_tool(self, names: Union[str, List[str]]) -> Tuple[bool, str]:
        """Deletes one or more user-defined tools."""
        if isinstance(names, str):
            names = [names]

        deleted_count = 0
        errors = []
        for name in names:
            if name in self.intrinsic_tools:
                errors.append(f"Cannot delete intrinsic tool '{name}'.")
                continue
            if name not in self.tools:
                errors.append(f"Tool '{name}' not found.")
                continue
            
            del self.tools[name]
            if name in self.active_tool_names: self.active_tool_names.remove(name)
            if name in self.hidden_tool_names: self.hidden_tool_names.remove(name)
            if name in self.user_tool_callables: del self.user_tool_callables[name]
            # Also remove any associated tool footer

            deleted_count += 1

        msg = f"Successfully deleted {deleted_count} tool(s)."
        if errors: msg += f"\nErrors:\n- " + "\n- ".join(errors)
        return deleted_count > 0, msg

    async def interactive_edit_tool(self, pt_session: "PromptSession", external_handler: Callable[..., Any], tool_name_to_edit: Optional[str] = None, search_scope: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        Interactively creates or edits a tool definition using a sequential dialog.
        If tool_name_to_edit is None, it creates a new tool.
        """
        is_intrinsic_edit = tool_name_to_edit and tool_name_to_edit in self.intrinsic_tools
        is_create_mode = tool_name_to_edit is None
        is_callable_tool = tool_name_to_edit and tool_name_to_edit in self.user_tool_callables

        if is_intrinsic_edit:
            # Load the base definition and apply any existing overrides
            base_def = self.intrinsic_tools[tool_name_to_edit]
            override_def = self.intrinsic_overrides.get(tool_name_to_edit, {})
            
            # For intrinsic tools, we only allow editing description and guide content.
            # We create a temporary structure for the editor.
            temp_def = {
                "description": override_def.get("description", base_def.get("function", {}).get("description", "")), # type: ignore
                "guide_content": override_def.get("guide_content", {})
            }
            # If the base tool has a guide and there's no override content yet, pre-populate it.
            if not temp_def["guide_content"]:
                parent_tool_name = tool_name_to_edit.removesuffix("_guide")
                registry = _get_intrinsics_registry()
                if parent_tool_name in registry and registry[parent_tool_name].guide_content:
                    temp_def["guide_content"] = {
                        k: copy.deepcopy(v)
                        for k, v in dict(registry[parent_tool_name].guide_content or {}).items()
                        if isinstance(v, list)
                    }

            original_name = tool_name_to_edit
        elif is_create_mode:
            temp_def = {
                "type": "function",
                "function": {
                    "name": "", "description": "",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                },
                "guide_content": {} # Initialize empty guide content
            }
            original_name = None
        else:
            current_def = self.get_tool(tool_name_to_edit)
            if not current_def:
                return False, f"Tool '{tool_name_to_edit}' not found." # type: ignore
            temp_def = copy.deepcopy(current_def)
            if "guide_content" not in temp_def:
                temp_def["guide_content"] = {} # Ensure it exists # type: ignore
            original_name = tool_name_to_edit

        def build_field_list():
            fields = []
            if is_intrinsic_edit:
                # Simplified editor for intrinsics
                fields.append({'display': 'Function Description', 'path': ('description',), 'type': 'str'}) # type: ignore
            else:
                # Full editor for user-defined tools
                fields.append({'display': 'Function Name', 'path': ('function', 'name'), 'type': 'str'})
                fields.append({'display': 'Function Description', 'path': ('function', 'description'), 'type': 'str'})

                if is_callable_tool:
                    fields.append({'display': 'Function Body (Reference)', 'type': 'info', 'value': f'<Python Callable: {self.user_tool_callables[tool_name_to_edit].__module__}.{tool_name_to_edit}>'})
                    # Parameters are not editable for callable tools
                
                param_props = temp_def['function']['parameters']['properties']
                for i, (param_name, param_schema) in enumerate(param_props.items()):
                    fields.append({'display': f'Parameter [{param_name}] > Name', 'type': 'param_name', 'old_name': param_name})
                    fields.append({'display': f'Parameter [{param_name}] > Type', 'path': ('function', 'parameters', 'properties', param_name, 'type'), 'type': 'str'})
                    fields.append({'display': f'Parameter [{param_name}] > Description', 'path': ('function', 'parameters', 'properties', param_name, 'description'), 'type': 'str'})
                    fields.append({'display': f'Parameter [{param_name}] > Required (y/n)', 'type': 'param_required', 'param_name': param_name})
                
                # Only allow parameter editing for non-callable tools
                if not is_callable_tool:
                    fields.append({'display': 'Add New Parameter', 'type': 'action', 'action': 'add_param'})
                    if param_props:
                        fields.append({'display': 'Remove a Parameter', 'type': 'action', 'action': 'remove_param'})
            
            # Guide editing fields
            fields.append({'display': 'Create/Edit Guide', 'type': 'action', 'action': 'edit_guide'})
            if temp_def.get("guide_content"):
                fields.append({'display': 'Remove Guide', 'type': 'action', 'action': 'remove_guide'})
            return fields

        fields = build_field_list()
        current_index = 0

        print("\n--- Interactive Tool Editor ---")
        print("Enter new value for the current field.")
        print("Commands: [+] next | [-] prev | [.] save & exit | [~] cancel without saving")

        while True:
            field = fields[current_index]
            display_name = field['display']
            field_type = field['type']

            # Get current value for display
            current_value_str = ''
            if field_type == 'str':
                path = field['path']
                obj = temp_def
                try:
                    for key in path: obj = obj[key]
                    current_value_str = str(obj)
                except (KeyError, TypeError): pass
            elif field_type == 'param_name':
                current_value_str = field['old_name']
            elif field_type == 'info':
                current_value_str = field['value']
                prompt_text = f"{display_name}: {current_value_str}"
            elif field_type == 'param_required':
                current_value_str = 'y' if field['param_name'] in temp_def['function']['parameters']['required'] else 'n'

            # Handle actions
            if field_type == 'action':
                prompt_text = f"{display_name}? (y/n) "
                user_input = (await pt_session.prompt_async(prompt_text)).strip().lower()
                
                action_taken = False
                if user_input == 'y':
                    if field['action'] == 'add_param':
                        new_param_name = (await pt_session.prompt_async("  Enter new parameter name: ")).strip()
                        if new_param_name and new_param_name not in temp_def['function']['parameters']['properties']:
                            temp_def['function']['parameters']['properties'][new_param_name] = {'type': 'string', 'description': ''}
                            fields = build_field_list()
                            for i, f in enumerate(fields):
                                if f.get('type') == 'param_name' and f.get('old_name') == new_param_name:
                                    current_index = i
                                    break
                            action_taken = True
                        elif not new_param_name:
                            print("  Name cannot be empty.")
                        else:
                            print(f"  Parameter '{new_param_name}' already exists.")
                    elif field['action'] == 'remove_param':
                        param_to_remove = (await pt_session.prompt_async("  Enter parameter name to remove: ")).strip()
                        if param_to_remove in temp_def['function']['parameters']['properties']:
                            del temp_def['function']['parameters']['properties'][param_to_remove]
                            if param_to_remove in temp_def['function']['parameters']['required']:
                                temp_def['function']['parameters']['required'].remove(param_to_remove)
                            fields = build_field_list()
                            current_index = min(current_index, len(fields) - 1)
                            action_taken = True
                        else:
                            print(f"  Parameter '{param_to_remove}' not found.")
                
                elif user_input == 'y' and field['action'] == 'edit_guide':
                    print("\n--- Guide Editor ---")
                    print("Manage topics and their content. Type '~' on a new line to finish editing a topic.")
                    guide_content = temp_def.get("guide_content", {})
                    while True:
                        existing_topics = ", ".join(guide_content.keys()) or "None"
                        print(f"Current topics: {existing_topics}")
                        topic_to_edit = (await pt_session.prompt_async("Enter topic name to edit/create (or '.' to exit guide editor): ")).strip()
                        if topic_to_edit == '.': break
                        if not topic_to_edit: continue

                        if topic_to_edit in guide_content and (await pt_session.prompt_async(f"Topic '{topic_to_edit}' exists. Delete it? (y/n) [n]: ")).lower() == 'y':
                            del guide_content[topic_to_edit]
                            print(f"Topic '{topic_to_edit}' deleted.")
                            continue

                        print(f"Editing topic '{topic_to_edit}'. Enter content lines. Type '~' on a new line to finish.")
                        current_content = guide_content.get(topic_to_edit, [])
                        if current_content:
                            print("--- Current Content ---")
                            for line in current_content: print(f"  {line}")
                            print("-----------------------")
                            if (await pt_session.prompt_async("Clear existing content? (y/n) [n]: ")).lower() == 'y':
                                current_content = []

                        new_lines = []
                        while True:
                            line_input = await pt_session.prompt_async(f"  ({len(current_content) + len(new_lines) + 1})> ")
                            if line_input.strip() == '~': break
                            new_lines.append(line_input)
                        
                        guide_content[topic_to_edit] = current_content + new_lines
                        print(f"Topic '{topic_to_edit}' updated with {len(guide_content[topic_to_edit])} lines.")
                    
                    temp_def["guide_content"] = guide_content
                    # After editing guide, rebuild fields and stay on the same action
                    fields = build_field_list()
                    action_taken = True                

                elif user_input == 'y' and field['action'] == 'remove_guide':
                    if (await pt_session.prompt_async("Are you sure you want to remove the entire guide? (y/n) [n]: ")).lower() == 'y':
                        temp_def.pop("guide_definition", None)
                        temp_def.pop("guide_content", None)
                        print("Guide removed.")
                    else:
                        print("Guide removal cancelled.")
                    fields = build_field_list()
                    current_index = min(current_index, len(fields) - 1)
                    action_taken = True

                if not action_taken: # If 'n' or action failed, just move to next field
                    current_index = (current_index + 1) % len(fields)
                continue

            # Handle info fields (not editable)
            if field_type == 'info':
                print(prompt_text)
                current_index = (current_index + 1) % len(fields)
                continue

            # Handle regular field prompts
            display_default = current_value_str
            if len(display_default) > 50: display_default = display_default[:47] + "..."
            prompt_text = f"{display_name} [{display_default}]: "
            user_input = (await pt_session.prompt_async(prompt_text)).strip()

            if user_input == '.': break
            if user_input == '~': return False, "Edit cancelled."
            if user_input == '+':
                current_index = (current_index + 1) % len(fields)
                continue
            if user_input == '-':
                current_index = (current_index - 1 + len(fields)) % len(fields)
                continue
            
            # If user just presses enter (empty input), keep the current value and move on.
            if not user_input:
                current_index = (current_index + 1) % len(fields)
                continue

            # Handle "" or '' input to mean an empty string value.
            value_to_set = user_input
            if user_input in ['""', "''"]:
                value_to_set = ""

            # Update value
            if field_type == 'param_name':
                old_name = field['old_name']
                new_name = value_to_set
                if old_name != new_name and new_name:
                    props = temp_def['function']['parameters']['properties']
                    if new_name in props:
                        print(f"Error: Parameter name '{new_name}' already exists.")
                    else:
                        props[new_name] = props.pop(old_name)
                        reqs = temp_def['function']['parameters']['required']
                        if old_name in reqs:
                            reqs.remove(old_name)
                            reqs.append(new_name)
                        fields = build_field_list()
            elif field_type == 'param_required':
                param_name = field['param_name']
                reqs = temp_def['function']['parameters']['required']
                if value_to_set.lower() == 'y':
                    if param_name not in reqs: reqs.append(param_name)
                elif value_to_set.lower() == 'n':
                    if param_name in reqs: reqs.remove(param_name)
            elif field_type == 'str':
                path = field['path']
                obj = temp_def
                for key in path[:-1]: obj = obj[key]
                obj[path[-1]] = value_to_set

                # --- NEW: Check for existing callable when setting function name in create mode ---
                if is_create_mode and path == ('function', 'name'):
                    # Check the provided scope (or fallback to local globals) for a callable with this name
                    if search_scope:
                        potential_func = search_scope.get(value_to_set)
                        if potential_func and callable(potential_func):
                            confirm = (await pt_session.prompt_async(f"Callable function '{value_to_set}' found. Register it automatically? (y/n) [y]: ")).strip().lower()
                            if confirm in ['y', 'yes', '']:
                                # Use add_tool_callable to perform the registration.
                                # This will create the schema, register the callable, and save.
                                success, msg = self.add_tool_callable(potential_func, search_scope=search_scope)
                                # The interactive session is now complete, so we return.
                                return success, msg

            # Move to next field
            current_index = (current_index + 1) % len(fields)

        # --- Save Logic ---
        if is_intrinsic_edit:
            # For intrinsics, we save to the overrides dictionary
            self.intrinsic_overrides[original_name] = {k: v for k, v in temp_def.items() if k != 'tool_footer'} # type: ignore
            return True, f"Intrinsic tool '{original_name}' override saved successfully."

        # --- Save Logic for user-defined tools ---
        final_name = temp_def['function']['name']
        if not final_name:
            return False, "Save cancelled: Function Name cannot be empty."

        # Clean up temp fields from schema that are not part of the official Tool definition
        for param_schema in temp_def['function']['parameters']['properties'].values():
            param_schema.pop('name', None)
            param_schema.pop('required_flag', None)

        if not is_create_mode and original_name != final_name:
            self.delete_tool(original_name) # type: ignore
        
        # Check for existence before creating
        if is_create_mode and final_name in self.tools:
            return False, f"Tool '{final_name}' already exists. Use 'edit' to modify it."

        # --- Auto-generate guide definition if content exists ---
        if temp_def.get("guide_content"):
            guide_name = f"{final_name}_guide"
            temp_def["guide_definition"] = {
                "type": "function",
                "function": {
                    "name": guide_name,
                    "description": f"Provides detailed guidance on using the {final_name} tool. Use topic='help' to see all topics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "The guidance topic to retrieve.", "enum": ["help"] + sorted(list(temp_def["guide_content"].keys()))},
                            "search": {"type": "string", "description": "An optional substring to filter results."}
                        },
                        "required": ["topic"]
                    }
                }
            }
            print(f"Auto-generated guide definition for '{guide_name}'.")
        
        success, msg = self._update_tool_internal(final_name, temp_def, external_handler=external_handler)
        
        if success:
            formatted_json = json.dumps(temp_def, indent=2)
            return True, f"Tool '{final_name}' saved successfully.\n{formatted_json}"
        else:
            return False, msg

    def update_tool_from_json_string(self, tool_name: str, json_string: str, external_handler: Callable[..., Any], allow_create: bool = False, search_scope: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Updates a tool from a raw JSON string."""
        if tool_name in self.intrinsic_tools:
            return False, f"Cannot update intrinsic tool '{tool_name}'."
        if tool_name in self.user_tool_callables:
            return False, f"Cannot update tool '{tool_name}' from JSON as it is a registered Python callable. Edit its description or guide content via the interactive editor."
        if not allow_create and tool_name not in self.tools:
            return False, f"Tool '{tool_name}' not found. Cannot update." # type: ignore
        
        try:
            new_definition = json.loads(json_string)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON provided: {e}"
        
        # The internal update function handles validation and saving
        success, msg = self._update_tool_internal(tool_name, new_definition, external_handler)

        # After updating, if the tool has no callable, assign the default handler.
        if success and tool_name not in self.user_tool_callables:
            self.user_tool_callables[tool_name] = external_handler

        return success, msg

    def add_tool_external(self, tool_definition: Dict[str, Any], implementation: Callable[..., Any], activate: Optional[bool] = False, allow_override: bool = False) -> Tuple[bool, str]: # noqa
        """
        Registers a tool with an explicit definition and a Python callable for execution.
        The tool definition is saved to tools.json, but the callable is only registered at runtime.
        This is the "manual" method, giving full control over the definition.
        The implementation callable does not need annotations.
        """
        try:
            # Use Pydantic for robust validation of the provided definition
            validated_tool = Tool.model_validate(tool_definition)
            tool_name = validated_tool.function.name
        except Exception as e:
            return False, f"Tool definition validation failed: {e}"

        # --- NEW: Inspect the handler to see if it accepts **kwargs and track required args ---
        accepts_kwargs = False
        internal_arg_names = {"_tool_args_issue"}
        required_args: List[str] = []
        try:
            sig = inspect.signature(implementation)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            for name, param in sig.parameters.items():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if name in internal_arg_names:
                    continue
                if param.default is inspect._empty:
                    required_args.append(name)
        except (ValueError, TypeError):
            # Some callables (like certain built-ins) can't be inspected. Assume they don't accept kwargs.
            accepts_kwargs = False
            required_args = []
        
        is_update = allow_override and tool_name in self.tools
        was_active = is_update and tool_name in self.active_tool_names

        if not allow_override and tool_name in self.tools:
            return False, f"Tool '{tool_name}' already exists. Use allow_override=True to replace it."

        if not callable(implementation):
            return False, f"Provided implementation for '{tool_name}' is not a callable Python function."

        # Save the definition to the in-memory dictionary
        self.tools[tool_name] = tool_definition

        # Register the callable for the current session
        self.user_tool_callables[tool_name] = implementation
        # Store whether it accepts kwargs
        tool_definition["_accepts_kwargs"] = accepts_kwargs
        tool_definition["_required_args"] = required_args

        # Activation Logic
        should_be_active = (activate is None and is_update and was_active) or (activate is True)
        is_active_now = tool_name in self.active_tool_names

        if should_be_active and not is_active_now:
            self.active_tool_names.append(tool_name)
        elif not should_be_active and is_active_now:
            self.active_tool_names.remove(tool_name)

        # Also handle hidden status when explicitly activating
        if activate is True and tool_name in self.hidden_tool_names:
            self.hidden_tool_names.remove(tool_name)


        return True, f"External tool '{tool_name}' registered and saved successfully."

    def add_tool_callable(
        self,
        func_or_name: str | Callable[..., Any] | Sequence[str],
        search_scope: Optional[Dict[str, Any]] = None,
        activate: Optional[bool] = False,
        *,
        is_intrinsic: bool = False,
        include_guides: bool = False,
        guide_content: Optional[Dict[str, List[str]]] = None,
        guide_description: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Register a callable tool, or intrinsic tool(s) when `is_intrinsic=True`.
        For user callables, optional `guide_content` can auto-create `<tool>_guide`.
        """
        if is_intrinsic:
            if callable(func_or_name):
                return False, "For intrinsic registration, provide intrinsic name(s), not a callable object."
            targets_input: Optional[Union[str, Sequence[str]]] = func_or_name
            targets, missing = self._resolve_intrinsic_targets(targets_input)
            existing_names = self._registered_tool_names()
            requested_names = set(targets or set())
            collisions = sorted([name for name in requested_names if name in existing_names])
            if collisions:
                return False, f"Intrinsic registration blocked due to existing tool name(s): {', '.join(collisions)}"
            existing = set(self.intrinsic_tools.keys())
            self.with_intrinsics = True
            self.with_intrinsic_guides = bool(include_guides)
            self._initialize_intrinsic_tools(include_guides=include_guides, intrinsic_names=targets_input)
            added = [name for name in self.intrinsic_tools.keys() if name not in existing]
            if activate:
                for name in added:
                    if name not in self.active_intrinsic_tool_names:
                        self.active_intrinsic_tool_names.append(name)
            if missing:
                return True, f"Registered {len(added)} intrinsic tool entries. Unknown intrinsic names skipped: {', '.join(missing)}"
            if targets is not None and not added:
                return False, "No intrinsic tools were added for the requested names."
            return True, f"Registered {len(added)} intrinsic tool entries."

        func: Optional[Callable[..., Any]] = None
        if isinstance(func_or_name, str):
            if not search_scope:
                return False, "A search_scope must be provided when adding a tool by name."
            func = search_scope.get(func_or_name)
            if not func:
                return False, f"Function '{func_or_name}' not found in the provided scope."
        elif isinstance(func_or_name, (list, tuple, set)):
            return False, "For non-intrinsic registration, provide a single callable or function name string."
        else:
            func = func_or_name

        if not callable(func):
            return False, f"Provided object '{func_or_name}' is not a callable function."

        tool_name = func.__name__
        existing_names = self._registered_tool_names()
        if tool_name in existing_names:
            return False, f"Tool name '{tool_name}' already exists."

        docstring = inspect.getdoc(func) or "No description provided."
        signature = inspect.signature(func)

        def _parse_param_descriptions(doc: str) -> Dict[str, str]:
            descriptions = {}
            try:
                args_section = doc.split("Args:")[1].split("Returns:")[0]
            except IndexError:
                return {}

            current_param = None
            for line in args_section.split('\n'):
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"(\w+)\s*\(.*\):\s*(.*)", line)
                if match:
                    param_name, description = match.groups()
                    descriptions[param_name] = description.strip()
                    current_param = param_name
                elif current_param and line:
                    descriptions[current_param] += " " + line.strip()
            return descriptions

        param_descriptions = _parse_param_descriptions(docstring)
        properties: Dict[str, Dict[str, str]] = {}
        required: List[str] = []
        internal_arg_names = {"_tool_args_issue"}
        for param in signature.parameters.values():
            if param.name in ("self", "cls"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.name in internal_arg_names:
                continue
            param_type = "string"
            if param.annotation in [int, float]:
                param_type = "number"
            if param.annotation is bool:
                param_type = "boolean"

            param_desc = param_descriptions.get(param.name, "")
            properties[param.name] = {"type": param_type, "description": param_desc}
            if param.default is inspect.Parameter.empty:
                required.append(param.name)

        tool_def: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": docstring,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
            "_type": "callable",
        }

        if guide_content:
            guide_name = f"{tool_name}_guide"
            if guide_name in existing_names:
                return False, f"Guide name '{guide_name}' already exists."
            topics = sorted(list(guide_content.keys()))
            tool_def["guide_content"] = guide_content
            tool_def["guide_definition"] = {
                "type": "function",
                "function": {
                    "name": guide_name,
                    "description": guide_description or f"Provides detailed guidance on using the {tool_name} tool. Use topic='help' to see all topics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "The guidance topic to retrieve.", "enum": ["help"] + topics},
                            "search": {"type": "string", "description": "An optional substring to filter results."},
                        },
                        "required": ["topic"],
                    },
                },
            }

        success, msg = self.add_tool_external(tool_def, func, activate, allow_override=False)
        if success:
            return True, f"Callable tool '{tool_name}' registered successfully."
        return False, msg

    def activate_tool(self, names: Union[str, List[str]]) -> Tuple[bool, str]:
        """Activates one or more tools."""
        if isinstance(names, str):
            names = [names]

        activated_count = 0
        errors = []
        names_to_process = set(names)
        primary_targets = set(names)

        for name in names:
            if name in self.intrinsic_tools:
                changed = False
                if name not in self.active_intrinsic_tool_names: 
                    self.active_intrinsic_tool_names.append(name)
                    changed = True
                if changed:
                    activated_count += 1
                # Also activate its guide if it exists
                guide_name = f"{name}_guide"
                if guide_name in self.intrinsic_tools: names_to_process.add(guide_name)
            elif name in self.tools:
                changed = False
                if name not in self.active_tool_names:
                    self.active_tool_names.append(name)
                    changed = True
                tool_def = self.get_tool(name)
                if tool_def and "guide_definition" in tool_def:
                    guide_name = tool_def["guide_definition"]["function"]["name"]
                    names_to_process.add(guide_name)
                if changed:
                    activated_count += 1
            else:
                errors.append(f"Tool '{name}' not found.")
        
        # Process the full set including guides
        extra_targets = names_to_process - primary_targets
        for name in extra_targets:
            if name in self.intrinsic_tools:
                if name not in self.active_intrinsic_tool_names: self.active_intrinsic_tool_names.append(name)
            elif name in self.tools:
                if name not in self.active_tool_names: self.active_tool_names.append(name)


        msg = f"Activated {activated_count} tool(s)."
        if errors: msg += f"\nErrors:\n- " + "\n- ".join(errors)
        return activated_count > 0, msg

    def deactivate_tool(self, names: Union[str, List[str]]) -> Tuple[bool, str]:
        """Deactivates one or more tools."""
        if isinstance(names, str):
            names = [names]

        deactivated_count = 0
        errors = []
        names_to_process = set(names)

        for name in names:
            if name in self.intrinsic_tools:
                if name in self.active_intrinsic_tool_names: 
                    self.active_intrinsic_tool_names.remove(name)
                if name not in self.hidden_intrinsic_tool_names:
                    self.hidden_intrinsic_tool_names.append(name)
                deactivated_count += 1
                guide_name = f"{name}_guide"
                if guide_name in self.intrinsic_tools: names_to_process.add(guide_name)
            elif name in self.tools:
                if name in self.active_tool_names:
                    self.active_tool_names.remove(name)
                if name not in self.hidden_tool_names:
                    self.hidden_tool_names.append(name)
                    deactivated_count += 1
                    tool_def = self.get_tool(name)
                    if tool_def and "guide_definition" in tool_def:
                        guide_name = tool_def["guide_definition"]["function"]["name"]
                        names_to_process.add(guide_name)
            else:
                errors.append(f"Tool '{name}' not found.")

        msg = f"Deactivated {deactivated_count} tool(s)."
        if errors: msg += f"\nErrors:\n- " + "\n- ".join(errors)
        return deactivated_count > 0, msg

    def set_hidden(self, names: Union[str, List[str]], is_hidden: bool) -> Tuple[bool, str]:
        """Sets the 'is_hidden' status of one or more tools."""
        if isinstance(names, str):
            names = [names]

        changed_count = 0
        errors = []
        action = "hidden" if is_hidden else "shown"

        for name in names:
            is_intrinsic = name in self.intrinsic_tools
            is_user_tool = name in self.tools

            if not is_intrinsic and not is_user_tool:
                errors.append(f"Tool '{name}' not found.")
                continue

            target_list = self.hidden_intrinsic_tool_names if is_intrinsic else self.hidden_tool_names
            
            if is_hidden: # if is_hidden is True, remove from shown list
                if name not in target_list:
                    target_list.append(name)
                    changed_count += 1
            else: # is_hidden is False, add to shown list
                if name in target_list:
                    target_list.remove(name)
                    changed_count += 1
        
        msg = f"Successfully set {changed_count} tool(s) to '{action}'."
        if errors: msg += f"\nErrors:\n- " + "\n- ".join(errors)
        return changed_count > 0, msg

    def get_tools_for_inference(self, tools_view: Optional[ToolsView] = None) -> Optional[Dict[str, Any]]:
        """
        Returns a dictionary containing the  list of tool definitions
        for all tools marked as 'shown'. Returns None if no tools are shown.
        """
        tools_view = tools_view or self.build_view()
        if not tools_view.advertised_tools:
            return None

        shown_defs = []

        for name in sorted(tools_view.advertised_tools):
            tool_def = self.get_tool(name)
            if not tool_def:
                continue

            if name in self.tools:
                engine_formatted_def = {
                    "type": tool_def.get("type", "function"),
                    "function": tool_def.get("function", {}),
                }
                shown_defs.append(engine_formatted_def)
            else:
                # Intrinsic tools already conform to the tool schema.
                shown_defs.append(tool_def)

        if not shown_defs:
            return None

        return {
            "for_dump": shown_defs,
        }

    def is_executable(self, name: str, tools_view: Optional[ToolsView] = None) -> bool:
        """Checks if a tool has a callable implementation (intrinsic, user-defined, or guide)."""
        if tools_view and (tools_view.is_disabled(name) or tools_view.is_gated(name) or not tools_view.is_allowed(name)):
            return False
        if self.global_tools_mode == "disabled" and not tools_view:
            return False
        if self._guide_content_for_tool_name(name):
            return True
        if name in self.intrinsic_tool_callables:
            # This covers both python functions and the default_handler for interactive tools.
            return True
        if name in self.user_tool_callables:
            if self.tools.get(name, {}).get("_type") == "unresolved":
                return False # Explicitly disable unresolved tools
            return True
        for tool_def in self.tools.values():
            if tool_def.get("guide_definition", {}).get("function", {}).get("name") == name:
                return True
        return False

    def gate_call(self, name: str, tools_view: Optional[ToolsView] = None) -> ToolCallGate:
        tool_name = str(name or "").strip()
        if not tool_name:
            return ToolCallGate(
                outcome="denied",
                tool_name="",
                reason="tool_name_required",
                executable=False,
            )
        tool_def = self.get_tool(tool_name)
        if not tool_def:
            return ToolCallGate(
                outcome="denied",
                tool_name=tool_name,
                reason="tool_not_defined",
                executable=False,
            )
        if tools_view and not tools_view.is_allowed(tool_name):
            if tools_view.is_gated(tool_name):
                return ToolCallGate(
                    outcome="gated_requires_confirmation",
                    tool_name=tool_name,
                    reason="gated_requires_confirmation",
                    executable=False,
                    requires_confirmation=True,
                )
            return ToolCallGate(
                outcome="denied",
                tool_name=tool_name,
                reason="blocked_in_scope",
                executable=False,
            )
        if self.global_tools_mode == "disabled" and not tools_view:
            return ToolCallGate(
                outcome="denied",
                tool_name=tool_name,
                reason="all_tools_disabled",
                executable=False,
            )
        is_intrinsic = tool_name in self.intrinsic_tools
        if not tools_view:
            is_active = (is_intrinsic and tool_name in self.active_intrinsic_tool_names) or (
                not is_intrinsic and tool_name in self.active_tool_names
            )
            if not is_active:
                return ToolCallGate(
                    outcome="denied",
                    tool_name=tool_name,
                    reason="tool_not_active",
                    executable=False,
                )
        if self.is_executable(tool_name, tools_view=tools_view):
            return ToolCallGate(
                outcome="allowed",
                tool_name=tool_name,
                reason="allowed",
                executable=True,
            )
        return ToolCallGate(
            outcome="unavailable_backend",
            tool_name=tool_name,
            reason="tool_has_no_executable_implementation",
            executable=False,
        )

    async def execute(self, tool_call: ToolCall, tools_view: Optional[ToolsView] = None, **kwargs: Any) -> Optional[str]:
        """
        Finds and executes the implementation for a tool call.
        - On success, returns the serialized result as a string.
        - On failure, sets `tool_call.error` and returns None.

        If the tool's callable accepts `**kwargs`, the following are injected:
        - `toolbox`: The Toolbox instance.
        - `tool_def`: The tool's definition dictionary.
        - `tool_call`: The ToolCall object being executed.
        - `tools_view`: The resolved ToolsView for this execution, when present.
        - `tool_constraints`: The resolved per-tool constraint payload for this execution.
        - `tool_constraints_view`: Helper wrapper around the resolved per-tool constraint payload.
        """
        tool_name = tool_call.name
        gate = self.gate_call(tool_name, tools_view=tools_view)
        if gate.outcome != "allowed":
            if gate.reason == "tool_not_defined":
                tool_call.error = f"Error: Tool '{tool_name}' is not defined."
            elif gate.reason == "blocked_in_scope":
                tool_call.error = f"Error: Tool '{tool_name}' is not permitted in the current scope."
            elif gate.reason == "gated_requires_confirmation":
                tool_call.error = f"Error: Tool '{tool_name}' requires confirmation before execution."
            elif gate.reason == "all_tools_disabled":
                tool_call.error = "Error: All tools are currently disabled."
            elif gate.reason == "tool_not_active":
                tool_call.error = f"Error: Tool '{tool_name}' is not active."
            else:
                tool_call.error = f"Error: Tool '{tool_name}' is defined but has no executable implementation."
            return None

        tool_def = self.get_tool(tool_name)
        callable_func: Optional[Callable[..., Any]] = None
        is_intrinsic = tool_name in self.intrinsic_tools

        if is_intrinsic:
            callable_func = self.intrinsic_tool_callables.get(tool_name)
        elif tool_name in self.user_tool_callables:
            callable_func = self.user_tool_callables.get(tool_name)
        else:
            guide_content = self._guide_content_for_tool_name(tool_name)
            if guide_content:
                return self._execute_static_guide(tool_call, guide_content)

        if not callable_func:
            guide_content = self._guide_content_for_tool_name(tool_name)
            if guide_content:
                return self._execute_static_guide(tool_call, guide_content)
            tool_call.error = f"Error: Tool '{tool_name}' is defined but has no executable implementation."
            return None

        try:
            # Check if the handler was registered as accepting **kwargs
            accepts_kwargs = tool_def.get("_accepts_kwargs", False)
            resolved_tool_arguments = _resolved_tool_arguments(tool_name, tool_call.arguments, tools_view)

            # --- NEW: Logic to handle malformed arguments ---
            # Detect if the arguments payload is a result of parser salvage/wrapping.
            # This is heuristic-based, checking for a single '_non_parsed' or '_string_value' key.
            is_malformed_payload = False
            if len(resolved_tool_arguments) == 1 and ('_non_parsed' in resolved_tool_arguments or '_string_value' in resolved_tool_arguments):
                is_malformed_payload = True

            call_kwargs = {}
            tool_args_issue_payload = None
            if isinstance(resolved_tool_arguments, dict):
                if "tool_args_issue" in resolved_tool_arguments:
                    tool_args_issue_payload = resolved_tool_arguments.get("tool_args_issue")
                internal_keys = [k for k in resolved_tool_arguments.keys() if k.startswith("_")]
                if internal_keys:
                    tool_args_issue_payload = tool_args_issue_payload or {
                        k: resolved_tool_arguments.get(k) for k in internal_keys
                    }

            if not accepts_kwargs and is_malformed_payload:
                # If the payload is malformed and the function is strict about its arguments,
                # pass the entire raw dictionary under the special 'tool_args_issue' key.
                # This allows the tool to attempt recovery instead of failing with a TypeError.
                call_kwargs['tool_args_issue'] = resolved_tool_arguments
            else:
                # For well-formed calls, only pass tool_call arguments unless **kwargs are accepted.
                if accepts_kwargs:
                    # Merge execute() kwargs with tool_call arguments; tool_call takes precedence.
                    final_args = kwargs.copy()
                    final_args.update(resolved_tool_arguments)
                    if tool_args_issue_payload is not None:
                        final_args.pop("tool_args_issue", None)
                        for k in list(final_args.keys()):
                            if k.startswith("_"):
                                final_args.pop(k, None)
                    call_kwargs.update(final_args)
                    resolved_constraints = copy.deepcopy(
                        tools_view.get_constraints(tool_name) if tools_view else {}
                    )
                    call_kwargs['toolbox'] = self
                    call_kwargs['tool_def'] = tool_def
                    call_kwargs['tool_call'] = tool_call
                    call_kwargs['tools_view'] = tools_view
                    call_kwargs['tool_constraints'] = resolved_constraints
                    call_kwargs['tool_constraints_view'] = ToolConstraintsView(
                        tool_name=tool_name,
                        payload=resolved_constraints,
                    )
                    if tool_args_issue_payload is not None:
                        call_kwargs["tool_args_issue"] = tool_args_issue_payload
                else:
                    # Strict signature: pass only the model-provided tool arguments.
                    if isinstance(resolved_tool_arguments, dict):
                        cleaned_args = {
                            k: v for k, v in resolved_tool_arguments.items()
                            if not k.startswith("_") and k != "tool_args_issue"
                        }
                        call_kwargs.update(cleaned_args)
                    else:
                        call_kwargs.update(resolved_tool_arguments)

            # Check if the function is async and call it accordingly
            is_async_func = inspect.iscoroutinefunction(callable_func)

            try:
                if is_async_func:
                    result: Any = await callable_func(**call_kwargs)
                else:
                    # Run sync function in a thread to avoid blocking the event loop
                    result: Any = await asyncio.to_thread(callable_func, **call_kwargs)

            except TypeError as e:
                # Catch TypeErrors specifically, which are often caused by unexpected keyword arguments
                # when the model provides a malformed tool call.
                if "unexpected keyword argument" in str(e):
                    tool_call.error = "Syntax error or unrecognized arguments format, pls correct and retry."
                    tool_call.action.append(ToolCall.Retry)
                    return None
                # Re-raise other TypeErrors (e.g., missing required arguments) to be caught below.
                raise e

            # --- Intelligent Serialization of Tool Results ---
            if isinstance(result, str):
                return result
            try:
                # For other types (dict, list, int, etc.), serialize to a JSON string.
                return json.dumps(result, indent=2)
            except TypeError:
                # Fallback for non-serializable objects.
                return str(result)
        except Exception as e:
            tool_call.error = f"Error executing tool '{tool_name}': {type(e).__name__} - {e}"
            return None

    def _execute_static_guide(self, tool_call: ToolCall, guide_content: Dict[str, Any]) -> str:
        """Executes a static content-backed guide."""
        def _query_guide_content(content_map: Dict[str, Any], topic: str, search: Optional[str] = None) -> List[str]:
            topics = sorted(list(content_map.keys()))
            if topic not in topics and topic != "all":
                raise ValueError(f"Invalid topic '{topic}'. Available topics are: {topics}")

            topics_to_process = topics if topic == "all" else [topic]
            results: List[str] = []
            search_lower = search.lower() if search else None

            for t in topics_to_process:
                items = content_map.get(t, [])
                if callable(items):
                    items = items()
                if not isinstance(items, list):
                    continue
                filtered_items = [item for item in items if not search_lower or (isinstance(item, str) and search_lower in item.lower())]
                if filtered_items:
                    results.append(f"--- {t.replace('_', ' ').title()} ---")
                    results.extend(filtered_items)

            return results or ["No results found for your search criteria."]

        if not guide_content:
            return "Error: This guide has no content."
        topic = tool_call.arguments.get("topic", "help")
        search = tool_call.arguments.get("search")
        return str(_query_guide_content(guide_content, topic, search))

    async def execute_request_tools(
        self,
        parser_profile: "ParserProfile", # type: ignore
        final_response_items: List["InferenceResponse"],
        action_handler: Callable[..., Any],
        serial_execution: bool = False,
        *,
        tools_view: Optional[ToolsView] = None,
        context: Optional[Any] = None,
        tool_retries_max: Optional[int] = None,
        tool_retries_left: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Parses and executes all tool calls from a batch inference response.
        This method mutates the ToolCall and ToolCallBlock objects in place with results or errors.

        Args:
            parser_profile: The ParserProfile object to use for parsing tool blocks.
            final_response_items: A list of the final InferenceResponse objects from the engine.
            action_handler: An async callable invoked at different execution stages.
                            It receives `execute_stage` and the objects relevant to that stage.
            serial_execution: If True, executes tool calls sequentially. If False (default), executes them in parallel.
            tools_view: The security/permissions view for this request.
            context: Optional context associated with this tool execution round.
            tool_retries_max: Maximum retries allowed for auto tool execution rounds (if any).
            tool_retries_left: Remaining retries for the current auto tool execution round (if any).
            **kwargs: Additional arguments to pass to the `action_handler` and `execute` methods.
                      The `action_handler` will receive:
                      - `final_response_items`: All response items from the engine.
                      - `current_response_item`: (during execution) The response item for the current tool call.
                      - `tool_call_block`: (during execution) The block for the current tool call.
                      - `context`: Chat round / context object.
                      - `tools_view`: The permissions/visibility context for this tool round.
                      - `tool_retries_max`: Retry ceiling for tool execution auto rounds.
                      - `tool_retries_left`: Remaining retries for tool execution auto rounds.
        """
        # --- 1. Parse all blocks first ---
        all_blocks_to_parse: List[ToolCallBlock] = []
        for response_item in final_response_items:
            if response_item.tool_blocks and len(response_item.tool_blocks) > 0:
                # Correctly propagate the prompt_index from the response item to each of its tool blocks.
                for block in response_item.tool_blocks:
                    if block.prompt_index is None:
                        block.prompt_index = response_item.prompt_index
                all_blocks_to_parse.extend(response_item.tool_blocks)

        if not all_blocks_to_parse:
            return
        
        parser = UnifiedToolIO(profile=parser_profile)
        parser.parse_collected_blocks(all_blocks_to_parse)

        parsed_kwargs: Dict[str, Any] = {
            **kwargs,
            'context': context,
            'final_response_items': final_response_items,
            'current_response_item': None,
            'parser': parser,
            'tool_call': None,
            'tool_call_block': None,
            'tools_view': tools_view,
            'tool_retries_max': tool_retries_max,
            'tool_retries_left': tool_retries_left,
            'serial_execution': serial_execution,
        }

        def _needs_recovery_args(tool_call: ToolCall) -> bool:
            if isinstance(tool_call.arguments, dict):
                if "tool_args_issue" in tool_call.arguments:
                    return True
                if "_non_parsed" in tool_call.arguments or "_string_value" in tool_call.arguments:
                    return True
                if any(k.startswith("_") for k in tool_call.arguments.keys()):
                    return True
            return False

        # Preflight: flag malformed calls that the tool cannot handle using tool definition metadata only.
        for response_item in final_response_items:
            for block in (response_item.tool_blocks or []):
                for tool_call in (block.calls or []):
                    if not _needs_recovery_args(tool_call):
                        continue
                    tool_def = self.get_tool(tool_call.name)
                    if not tool_def:
                        continue
                    accepts_kwargs = bool(tool_def.get("_accepts_kwargs", False))
                    required_args = tool_def.get("_required_args")
                    if required_args is None:
                        required_args = tool_def.get("function", {}).get("parameters", {}).get("required", []) or []
                    if (not accepts_kwargs) or required_args:
                        tool_call.error = (
                            f"Error executing tool '{tool_call.name}': malformed or truncated tool call arguments. "
                            "Please resend a complete tool call with valid arguments."
                        )
                        if ToolCall.KeepRaw not in tool_call.action:
                            tool_call.action.append(ToolCall.KeepRaw)
                        if ToolCall.Ignore not in tool_call.action:
                            tool_call.action.append(ToolCall.Ignore)

        await action_handler(
            execute_stage='calls_parsed', 
            **parsed_kwargs
        )

        # --- 2. Define the core execution logic for a single tool call ---
        async def _execute_and_handle(tc: ToolCall, act_kwargs: Dict[str, Any]):
            try:
                # Invoke handler before execution
                await action_handler(execute_stage='call_starting', tool_call=tc, **act_kwargs)

                # The `execute` method handles finding the callable and running it.
                # It returns an error string if the tool is not executable.
                exec_kwargs = dict(act_kwargs)
                exec_kwargs.setdefault("tools_view", tools_view)
                result = await self.execute(tool_call=tc, **exec_kwargs)
                # The `execute` method now sets tc.error directly on failure and returns None.
                # On success, it returns the serialized result string.
                if result is not None:
                    tc.result = result
            except Exception as e:
                if not tc.error: # Only set if not already set by the execute method
                    tc.error = f"Execution failed: {type(e).__name__} - {e}"
            finally:
                # Invoke the action handler at the end of the execution attempt, with the result/error populated
                await action_handler(execute_stage='call_finished', tool_call=tc, **act_kwargs)

        # --- 3. Execute tasks sequentially or in parallel ---
        if serial_execution:
            for response_item in final_response_items:
                for block in (response_item.tool_blocks or []):
                    if not block.calls and not block.is_incomplete:
                        block.error_block = "Tool calls list is empty."
                        if ToolCall.KeepRaw not in (block.action_block or []):
                            block.action_block = list(block.action_block or [])
                            block.action_block.append(ToolCall.KeepRaw)
                        continue
                    if ToolCall.Ignore in block.action_block:
                        continue
                    # In serial mode, we still create tasks but await them one by one.
                    # This keeps the _execute_and_handle logic consistent.
                    for tool_call in block.calls: # type: ignore
                        if ToolCall.Ignore in tool_call.action:
                            continue
                        action_kwargs = {
                            **kwargs, 
                            'context': context,
                            'final_response_items': final_response_items, 
                            'current_response_item': response_item, 
                            'parser': parser,
                            'tool_call_block': block,
                            'tools_view': tools_view,
                            'tool_retries_max': tool_retries_max,
                            'tool_retries_left': tool_retries_left,
                            'serial_execution': serial_execution,
                        }
                        task = asyncio.create_task(_execute_and_handle(tool_call, action_kwargs))
                        await task # Await the single task before creating the next one.
        else: # Parallel execution
            tasks = []
            for response_item in final_response_items:
                for block in (response_item.tool_blocks or []):
                    if not block.calls and not block.is_incomplete:
                        block.error_block = "Tool calls list is empty."
                        if ToolCall.KeepRaw not in (block.action_block or []):
                            block.action_block = list(block.action_block or [])
                            block.action_block.append(ToolCall.KeepRaw)
                        continue
                    if ToolCall.Ignore in block.action_block:
                        continue
                    for tool_call in block.calls:
                        if ToolCall.Ignore in tool_call.action:
                            continue
                        action_kwargs = {
                            **kwargs, 
                            'context': context,
                            'final_response_items': final_response_items, 
                            'current_response_item': response_item, 
                            'parser': parser,
                            'tool_call_block': block,
                            'tools_view': tools_view,
                            'tool_retries_max': tool_retries_max,
                            'tool_retries_left': tool_retries_left,
                            'serial_execution': serial_execution,
                        }
                        tasks.append(_execute_and_handle(tool_call, action_kwargs))
            if tasks:
                await asyncio.gather(*tasks)

        # --- 4. Final handler invocation ---
        await action_handler(
            execute_stage='all_finished',
            **parsed_kwargs,
        )
