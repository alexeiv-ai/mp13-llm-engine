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
import re, copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Union, Set
from typing import TYPE_CHECKING

# Import the Tool model for validation
from .mp13_config import RegisteredTool, Tool, ToolCall, ToolCallBlock, InferenceResponse
from .mp13_tools_builtin  import  INTRINSICS_REGISTRY, Guide
from .mp13_tools_parser import UnifiedToolIO

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession
    from .mp13_state import MP13State


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
    """
    mode: Optional[str] = None
    advertise_tools: Set[str] = field(default_factory=set)
    silent_tools: Set[str] = field(default_factory=set)
    disabled_tools: Set[str] = field(default_factory=set)
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
        if self.mode and self.mode not in self.VALID_MODES:
            raise ValueError(f"ToolsScope.mode '{self.mode}' is invalid. Allowed: {sorted(self.VALID_MODES)}")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "advertise_tools": sorted(list(self.advertise_tools)),
            "silent_tools": sorted(list(self.silent_tools)),
            "disabled_tools": sorted(list(self.disabled_tools)),
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
        return " | ".join(bits) if bits else "no-op scope"

    def is_noop(self) -> bool:
        return not (self.mode or self.advertise_tools or self.silent_tools or self.disabled_tools)


@dataclass
class ToolsView:
    """Materialized permissions/advertisement view for a particular turn/request."""
    view_id: str
    mode: str
    allowed_tools: Set[str]
    advertised_tools: Set[str]
    hidden_allowed_tools: Set[str]
    disabled_tools: Set[str]

    def __post_init__(self):
        """Ensures that upon deserialization from a dict (where these might be lists),
        the fields are converted back to sets for runtime efficiency."""
        self.allowed_tools = set(self.allowed_tools)
        self.advertised_tools = set(self.advertised_tools)
        self.hidden_allowed_tools = set(self.hidden_allowed_tools)
        self.disabled_tools = set(self.disabled_tools)

    def should_advertise(self, tool_name: str) -> bool:
        return tool_name in self.advertised_tools

    def is_allowed(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools


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


class Toolbox:
    """Manages tool definitions for the chat application."""
    _VALID_GLOBAL_MODES = {"advertised", "silent", "disabled"}

    def __init__(self):
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

        self._initialize_intrinsic_tools()
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

    def _initialize_intrinsic_tools(self):
        """Defines the schema and callables for built-in tools."""
        # Load intrinsic tools from the central registry and unwrap them.
        for name, tool_container in INTRINSICS_REGISTRY.items():
            self.intrinsic_tools[tool_container.name] = tool_container.definition
            self.intrinsic_tool_callables[tool_container.name] = tool_container.implementation
            if tool_container.guide_definition and tool_container.guide_implementation:
                guide_name = tool_container.guide_definition["function"]["name"]
                self.intrinsic_tools[guide_name] = tool_container.guide_definition
                self.intrinsic_tool_callables[guide_name] = tool_container.guide_implementation

    def from_dict(self, data: Dict[str, Any], search_scope: Optional[Dict[str, Any]] = None, external_handler: Optional[Callable[..., Any]] = None): # noqa
        """Loads tool state from a dictionary and re-links callables."""
        self.tools = data.get("tools", {})
#        self.prompt_header = data.get("prompt_header")
#        self.prompt_footer = data.get("prompt_footer")
#        self.tool_footers = data.get("tool_footers", {})
        self.intrinsic_overrides = data.get("intrinsic_overrides", {})
        self.active_tool_names = data.get("active_tools", [])
        self.hidden_tool_names = data.get("hidden_tools", [])
        self.active_intrinsic_tool_names = data.get("active_intrinsic_tools", list(self.intrinsic_tools.keys()))
        self.hidden_intrinsic_tool_names = data.get("hidden_intrinsic_tools", [])
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
        for tool_container in INTRINSICS_REGISTRY.values():
            # Process main tool
            if tool_container.name not in managed_tools:
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
                guide_name = tool_container.guide_definition["function"]["name"]
                if guide_name not in managed_tools:
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
        return active

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Gets the full definition of a tool by name."""
        return self.tools.get(name) or self.intrinsic_tools.get(name)

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
        status_priority = {"silent": 1, "advertised": 2, "disabled": 3}
        per_tool_status: Dict[str, Tuple[int, int, str]] = {}

        if effective_mode != "disabled":
            for name in active_names:
                if effective_mode == "advertised" and not self._is_hidden(name):
                    per_tool_status[name] = (0, status_priority["advertised"], "advertised")
                else:
                    per_tool_status[name] = (0, status_priority["silent"], "silent")

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
            for name in resolved:
                new_rank = scope_index * 10 + status_priority[status]
                previous = per_tool_status.get(name)
                prev_rank = previous[0] * 10 + previous[1] if previous else -1
                if new_rank >= prev_rank:
                    per_tool_status[name] = (scope_index, status_priority[status], status)

        for idx, scope in enumerate(cleaned_scopes, start=1):
            apply_status(scope.disabled_tools, "disabled", idx)
            apply_status(scope.advertise_tools, "advertised", idx)
            apply_status(scope.silent_tools, "silent", idx)

        allowed: Set[str] = set()
        advertised: Set[str] = set()
        disabled: Set[str] = set()
        for name in active_names:
            entry = per_tool_status.get(name)
            if not entry or entry[2] == "disabled":
                disabled.add(name)
                continue
            allowed.add(name)
            if entry[2] == "advertised":
                advertised.add(name)

        hidden_allowed = allowed - advertised
        self._view_seq += 1
        view_id = label or f"tools-view-{self._view_seq}"

        disabled.update(set(active_names) - allowed)

        return ToolsView(
            view_id=view_id,
            mode=effective_mode,
            allowed_tools=allowed,
            advertised_tools=advertised,
            hidden_allowed_tools=hidden_allowed,
            disabled_tools=disabled,
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
        return self.add_tool_external(new_definition, external_handler, activate=False, allow_override=True)
    
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
            if name in self.tool_footers:
                del self.tool_footers[name]
                print(f"Removed footer for deleted tool '{name}'.")

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
                "tool_footer": self.tool_footers.get(tool_name_to_edit, ""),
                "guide_content": override_def.get("guide_content", {})
            }
            # If the base tool has a guide and there's no override content yet, pre-populate it.
            if not temp_def["guide_content"]:
                parent_tool_name = tool_name_to_edit.removesuffix("_guide")
                if parent_tool_name in INTRINSICS_REGISTRY and INTRINSICS_REGISTRY[parent_tool_name].guide_implementation:
                    guide_impl = INTRINSICS_REGISTRY[parent_tool_name].guide_implementation
                    if hasattr(guide_impl, "_content_map"):
                        temp_def["guide_content"] = {k: v for k, v in guide_impl._content_map.items() if isinstance(v, list)}

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
                fields.append({'display': 'Tool Footer', 'path': ('tool_footer',), 'type': 'str'}) # type: ignore
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
            tool_footer = temp_def.get("tool_footer") # type: ignore
            if tool_footer: self.tool_footers[original_name] = tool_footer # type: ignore
            elif original_name in self.tool_footers: del self.tool_footers[original_name] # type: ignore
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

    def add_tool_external(self, tool_definition: Dict[str, Any], implementation: Callable[..., Any], activate: bool = True, allow_override: bool = False) -> Tuple[bool, str]: # noqa
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

        if activate:
            if tool_name not in self.active_tool_names:
                self.active_tool_names.append(tool_name)
            # When a tool is added and activated, also make it shown by default (remove from hidden).
            if tool_name in self.hidden_tool_names:
                self.hidden_tool_names.remove(tool_name)

        return True, f"External tool '{tool_name}' registered and saved successfully."

    def add_tool_callable(self, func_or_name: str | Callable[..., Any], search_scope: Optional[Dict[str, Any]] = None, activate: bool = True) -> Tuple[bool, str]:
        """
        Introspects a Python function to create and register a tool definition.
        This is the "automatic" method. It requires a well-annotated function.
        Can be called with a function object or its name (if search_scope is provided).
        """
        func: Optional[Callable[..., Any]] = None
        if isinstance(func_or_name, str):
            if not search_scope:
                return False, "A search_scope must be provided when adding a tool by name."
            func = search_scope.get(func_or_name)
            if not func:
                return False, f"Function '{func_or_name}' not found in the provided scope."
        else:
            func = func_or_name

        if not callable(func):
            return False, f"Provided object '{func_or_name}' is not a callable function."

        tool_name = func.__name__
        docstring = inspect.getdoc(func) or "No description provided."
        signature = inspect.signature(func)

        # --- Helper to parse parameter descriptions from the docstring ---
        def _parse_param_descriptions(doc: str) -> Dict[str, str]:
            descriptions = {}
            try:
                # Isolate the 'Args:' section
                args_section = doc.split("Args:")[1].split("Returns:")[0]
            except IndexError:
                return {}

            current_param = None
            for line in args_section.split('\n'):
                line = line.strip()
                if not line: continue

                match = re.match(r"(\w+)\s*\(.*\):\s*(.*)", line)
                if match:
                    param_name, description = match.groups()
                    descriptions[param_name] = description.strip()
                    current_param = param_name
                elif current_param and line:
                    descriptions[current_param] += " " + line.strip()
            return descriptions

        param_descriptions = _parse_param_descriptions(docstring)
        properties = {}
        required = []
        internal_arg_names = {"_tool_args_issue"}
        for param in signature.parameters.values():
            if param.name in ('self', 'cls'): continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.name in internal_arg_names:
                continue
            # Basic type mapping
            param_type = "string" # Default
            if param.annotation in [int, float]: param_type = "number"
            if param.annotation is bool: param_type = "boolean"

            param_desc = param_descriptions.get(param.name, "")
            properties[param.name] = {"type": param_type, "description": param_desc}
            if param.default is inspect.Parameter.empty:
                required.append(param.name)

        tool_def = {"type": "function", "function": {"name": tool_name, "description": docstring, "parameters": {"type": "object", "properties": properties, "required": required}}}
        # --- Explicitly set the type so it's saved to JSON ---
        tool_def["_type"] = "callable"

        # Allow overriding an existing definition when re-registering a callable
        success, msg = self.add_tool_external(tool_def, func, activate, allow_override=True)
        if success:
            # Explicitly activate the tool if it's not already active.
            if tool_name not in self.active_tool_names:
                self.active_tool_names.append(tool_name)
            return True, f"Callable tool '{tool_name}' registered successfully."
        else:
            return False, msg # Pass the original error message through

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
        if tools_view and not tools_view.is_allowed(name):
            return False
        if self.global_tools_mode == "disabled" and not tools_view:
            return False
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

    async def execute(self, tool_call: ToolCall, tools_view: Optional[ToolsView] = None, **kwargs: Any) -> Optional[str]:
        """
        Finds and executes the implementation for a tool call.
        - On success, returns the serialized result as a string.
        - On failure, sets `tool_call.error` and returns None.

        If the tool's callable accepts `**kwargs`, the following are injected:
        - `toolbox`: The Toolbox instance.
        - `tool`: The tool's definition dictionary.
        - `tool_call`: The ToolCall object being executed.
        """
        tool_name = tool_call.name
        tool_def = self.get_tool(tool_name)

        # Find the callable implementation based on the tool's type.
        callable_func: Optional[Callable[..., Any]] = None
        is_intrinsic = tool_name in self.intrinsic_tools

        if not tool_def:
            tool_call.error = f"Error: Tool '{tool_name}' is not defined."
            return None

        # --- NEW: Check if the tool is active before execution ---
        if tools_view:
            if not tools_view.is_allowed(tool_name):
                tool_call.error = f"Error: Tool '{tool_name}' is not permitted in the current scope."
                return None
        else:
            if self.global_tools_mode == "disabled":
                tool_call.error = "Error: All tools are currently disabled."
                return None
            is_active = (is_intrinsic and tool_name in self.active_intrinsic_tool_names) or \
                        (not is_intrinsic and tool_name in self.active_tool_names)
            if not is_active:
                tool_call.error = f"Error: Tool '{tool_name}' is not active."
                return None

        if is_intrinsic:
            callable_func = self.intrinsic_tool_callables.get(tool_name)
        elif tool_name in self.user_tool_callables:
            callable_func = self.user_tool_callables.get(tool_name)
        else:
            # Check if it's a user-defined guide, which is a special case.
            for t_def in self.tools.values():
                if t_def.get("guide_definition", {}).get("function", {}).get("name") == tool_name:
                    return self._execute_user_guide(tool_call)

        if not callable_func:
            tool_call.error = f"Error: Tool '{tool_name}' is defined but has no executable implementation."
            return None

        try:
            # Check if the handler was registered as accepting **kwargs
            accepts_kwargs = tool_def.get("_accepts_kwargs", False)

            # --- NEW: Logic to handle malformed arguments ---
            # Detect if the arguments payload is a result of parser salvage/wrapping.
            # This is heuristic-based, checking for a single '_non_parsed' or '_string_value' key.
            is_malformed_payload = False
            if len(tool_call.arguments) == 1 and ('_non_parsed' in tool_call.arguments or '_string_value' in tool_call.arguments):
                is_malformed_payload = True

            call_kwargs = {}
            tool_args_issue_payload = None
            if isinstance(tool_call.arguments, dict):
                if "tool_args_issue" in tool_call.arguments:
                    tool_args_issue_payload = tool_call.arguments.get("tool_args_issue")
                internal_keys = [k for k in tool_call.arguments.keys() if k.startswith("_")]
                if internal_keys:
                    tool_args_issue_payload = tool_args_issue_payload or {
                        k: tool_call.arguments.get(k) for k in internal_keys
                    }

            if not accepts_kwargs and is_malformed_payload:
                # If the payload is malformed and the function is strict about its arguments,
                # pass the entire raw dictionary under the special 'tool_args_issue' key.
                # This allows the tool to attempt recovery instead of failing with a TypeError.
                call_kwargs['tool_args_issue'] = tool_call.arguments
            else:
                # For well-formed calls, only pass tool_call arguments unless **kwargs are accepted.
                if accepts_kwargs:
                    # Merge execute() kwargs with tool_call arguments; tool_call takes precedence.
                    final_args = kwargs.copy()
                    final_args.update(tool_call.arguments)
                    if tool_args_issue_payload is not None:
                        final_args.pop("tool_args_issue", None)
                        for k in list(final_args.keys()):
                            if k.startswith("_"):
                                final_args.pop(k, None)
                    call_kwargs.update(final_args)
                    call_kwargs['toolbox'] = self
                    call_kwargs['tool_def'] = tool_def
                    # --- NEW: Inject the tool_call object itself ---
                    call_kwargs['tool_call'] = tool_call
                    if tool_args_issue_payload is not None:
                        call_kwargs["tool_args_issue"] = tool_args_issue_payload
                else:
                    # Strict signature: pass only the model-provided tool arguments.
                    if isinstance(tool_call.arguments, dict):
                        cleaned_args = {
                            k: v for k, v in tool_call.arguments.items()
                            if not k.startswith("_") and k != "tool_args_issue"
                        }
                        call_kwargs.update(cleaned_args)
                    else:
                        call_kwargs.update(tool_call.arguments)

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

    def _execute_user_guide(self, tool_call: ToolCall) -> str:
        """Executes a user-defined guide tool."""
        for tool_def in self.tools.values():
            if tool_def.get("guide_definition", {}).get("function", {}).get("name") == tool_call.name:
                guide_content = tool_def.get("guide_content", {})
                if not guide_content:
                    return "Error: This guide has no content."
                
                guide_obj = Guide(guide_content)
                topic = tool_call.arguments.get("topic", "help")
                search = tool_call.arguments.get("search")
                return str(guide_obj.query(topic, search))
        return f"Error: Could not find implementation for user-defined guide '{tool_call.name}'."

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
