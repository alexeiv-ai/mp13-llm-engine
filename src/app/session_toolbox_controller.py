from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from .context_cursor import ChatCursor
from .hosted_tool_visibility import annotate_tool_listing, summarize_effective_tool_view
from .session_cursor_toolbox import collect_tools_scope_entries, normalize_scope_tool_names
from mp13_engine.mp13_toolbox import ToolBoxRef, Toolbox, ToolsScope, ToolsView


@dataclass
class ControllerMessage:
    text: str
    level: str = "info"


@dataclass
class CommandResult:
    cursor: Optional[ChatCursor] = None
    handled: bool = True
    messages: List[ControllerMessage] = field(default_factory=list)
    listing_rows: Optional[List[Dict[str, Any]]] = None
    listing_tools: Optional[List[str]] = None
    scope_entries: Optional[List[Tuple[Optional[str], ToolsScope]]] = None
    scope_effective: Optional[Dict[str, Any]] = None
    tools_view: Optional[ToolsView] = None

    def add(self, text: str, level: str = "info") -> "CommandResult":
        self.messages.append(ControllerMessage(text=text, level=level))
        return self


PromptFn = Callable[[str], Awaitable[str]]
MultilinePromptFn = Callable[[str], Awaitable[str]]
InteractiveEditFn = Callable[[Optional[str], Dict[str, Any]], Awaitable[Tuple[bool, str]]]
ExternalHandlerFn = Callable[[], Callable[..., Awaitable[str]]]


@dataclass
class SessionToolboxController:
    get_toolbox: Callable[[], Optional[Toolbox]]
    get_toolbox_ref: Callable[[], Optional[ToolBoxRef]]
    get_hosted_summary: Callable[[], Optional[Dict[str, Any]]]
    get_current_config: Callable[[], Optional[Dict[str, Any]]]
    get_search_scope: Callable[[], Mapping[str, Any]]
    prompt_user_fn: PromptFn
    prompt_multiline_fn: MultilinePromptFn
    interactive_edit_fn: InteractiveEditFn
    get_external_tool_handler_fn: ExternalHandlerFn

    def _result(self, cursor: Optional[ChatCursor]) -> CommandResult:
        return CommandResult(cursor=cursor, handled=True)

    def _toolbox_or_error(self, cursor: ChatCursor) -> Tuple[Optional[Toolbox], CommandResult]:
        result = self._result(cursor)
        toolbox = self.get_toolbox()
        if not toolbox:
            result.add("Error: Toolbox not initialized.", "error")
            return None, result
        return toolbox, result

    def _hosted_advertised_tool_names(self) -> List[str]:
        hosted = self.get_hosted_summary() or {}
        return [
            str(item or "").strip()
            for item in list(hosted.get("advertised_tool_names") or [])
            if str(item or "").strip()
        ]

    def _hosted_hidden_allowed_tool_names(self) -> List[str]:
        hosted = self.get_hosted_summary() or {}
        return [
            str(item or "").strip()
            for item in list(hosted.get("hidden_allowed_tool_names") or [])
            if str(item or "").strip()
        ]

    def _scope_summary(self, tools_view: ToolsView, entries: List[Tuple[Optional[str], ToolsScope]]) -> Dict[str, Any]:
        return summarize_effective_tool_view(
            tools_view,
            hosted_advertised_tool_names=self._hosted_advertised_tool_names(),
            hosted_hidden_allowed_tool_names=self._hosted_hidden_allowed_tool_names(),
        )

    async def cmd_enum(self, cursor: ChatCursor) -> CommandResult:
        toolbox, result = self._toolbox_or_error(cursor)
        if not toolbox:
            return result
        tools = toolbox.list_tools()
        if not tools:
            result.listing_tools = []
            return result.add("No tools defined. Use '/t new' to add one.", "warning")
        tools_view = cursor.get_tools_view() or cursor.refresh_tools_view()
        result.listing_rows = annotate_tool_listing(
            tools,
            tools_view=tools_view,
            hosted_advertised_tool_names=self._hosted_advertised_tool_names(),
            hosted_hidden_allowed_tool_names=self._hosted_hidden_allowed_tool_names(),
        )
        result.listing_tools = [str(row["name"]) for row in result.listing_rows]
        return result

    async def cmd_new(self, cursor: ChatCursor) -> CommandResult:
        result = self._result(cursor)
        success, msg = await self.interactive_edit_fn(None, {"search_scope": dict(self.get_search_scope())})
        result.add(msg, "info" if success else "error")
        return result

    async def cmd_modify(self, cursor: ChatCursor, target_arg: str, last_enumerated_tools: List[str]) -> CommandResult:
        toolbox, result = self._toolbox_or_error(cursor)
        if not toolbox:
            return result
        if not target_arg.strip():
            return result.add("Usage: /t modify [g/]<name|num>", "warning")
        target_arg = target_arg.strip()
        edit_guide = False
        if target_arg.lower().startswith("g/"):
            edit_guide = True
            target_arg = target_arg[2:]
        if target_arg.isdigit():
            idx = int(target_arg) - 1
            if 0 <= idx < len(last_enumerated_tools):
                tool_name_to_edit = last_enumerated_tools[idx]
            else:
                return result.add(f"Invalid tool number: {target_arg}. Use '/t list'.", "error")
        else:
            tool_name_to_edit = target_arg
        if edit_guide:
            tool_def = toolbox.get_tool(tool_name_to_edit)
            if tool_def and "guide_definition" in tool_def:
                tool_name_to_edit = tool_def["guide_definition"]["function"]["name"]
            elif f"{tool_name_to_edit}_guide" in toolbox.intrinsic_tools:
                tool_name_to_edit = f"{tool_name_to_edit}_guide"
            else:
                return result.add(f"Error: Could not find a guide function for tool '{tool_name_to_edit}'.", "error")
        success, msg = await self.interactive_edit_fn(
            tool_name_to_edit,
            {"search_scope": dict(self.get_search_scope())},
        )
        return result.add(msg, "info" if success else "error")

    async def cmd_replace(self, cursor: ChatCursor, tool_name: str) -> CommandResult:
        toolbox, result = self._toolbox_or_error(cursor)
        if not toolbox:
            return result
        if not tool_name:
            return result.add("Usage: /t replace <name|num>", "warning")
        json_string = await self.prompt_multiline_fn(
            f"Enter the full JSON definition for '{tool_name}'. Type END_JSON on a new line to finish."
        )
        if not json_string:
            return result.add("Update cancelled.", "warning")
        success, msg = toolbox.update_tool_from_json_string(
            tool_name,
            json_string,
            external_handler=self.get_external_tool_handler_fn(),
            search_scope=dict(self.get_search_scope()),
        )
        return result.add(msg, "info" if success else "error")

    async def cmd_print(self, cursor: ChatCursor, tool_name: str) -> CommandResult:
        toolbox, result = self._toolbox_or_error(cursor)
        if not toolbox:
            return result
        if not tool_name:
            return result.add("Usage: /t print <name|num>", "warning")
        tool_def = toolbox.get_tool(tool_name)
        if tool_def:
            result.add(f"\n--- Tool Definition: {tool_name} ---")
            result.add(json.dumps(tool_def, indent=2))
            result.add("---")
        else:
            result.add(f"Tool '{tool_name}' not found.", "error")
        return result

    async def cmd_tool_state(self, cursor: ChatCursor, action: str, tool_names: List[str]) -> CommandResult:
        toolbox, result = self._toolbox_or_error(cursor)
        if not toolbox:
            return result
        if not tool_names:
            msg = "No valid tools specified for unregister." if action == "unregister" else "No valid tools specified."
            return result.add(msg, "error")
        if action == "activate":
            success, msg = await asyncio.to_thread(toolbox.activate_tool, tool_names)
        elif action == "deactivate":
            success, msg = await asyncio.to_thread(toolbox.deactivate_tool, tool_names)
        elif action in {"hide", "show"}:
            success, msg = await asyncio.to_thread(toolbox.set_hidden, tool_names, action == "hide")
        else:
            success, msg = await asyncio.to_thread(toolbox.delete_tool, tool_names)
        return result.add(msg, "info" if success else "error")

    async def cmd_save(self, cursor: ChatCursor, save_path_str: str) -> CommandResult:
        toolbox, result = self._toolbox_or_error(cursor)
        if not toolbox:
            return result
        if save_path_str:
            target_path = Path(save_path_str).expanduser().resolve()
        else:
            tools_path = (self.get_current_config() or {}).get("tools_config_path")
            if not tools_path:
                return result.add("No tools config path configured.", "error")
            target_path = Path(tools_path)
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(toolbox.to_dict(), f, indent=2)
            result.add(f"Toolbox state saved to {target_path}")
        except Exception as exc:
            result.add(f"Error saving toolbox state: {exc}", "error")
        return result

    async def cmd_load(self, cursor: ChatCursor, load_arg: str) -> CommandResult:
        toolbox, result = self._toolbox_or_error(cursor)
        if not toolbox:
            return result
        if not load_arg:
            return result.add("Usage: /t load <file_path|json_string>", "warning")
        try:
            if load_arg.strip().startswith("{"):
                data = json.loads(load_arg)
                toolbox.from_dict(
                    data,
                    search_scope=dict(self.get_search_scope()),
                    external_handler=self.get_external_tool_handler_fn(),
                )
                result.add("Toolbox state loaded from JSON string.")
            else:
                with open(Path(load_arg).expanduser().resolve(), "r", encoding="utf-8") as f:
                    toolbox.from_dict(
                        json.load(f),
                        search_scope=dict(self.get_search_scope()),
                        external_handler=self.get_external_tool_handler_fn(),
                    )
                result.add(f"Toolbox state loaded from {load_arg}")
        except Exception as exc:
            result.add(f"Error loading toolbox state: {exc}", "error")
        return result

    async def cmd_fix(self, cursor: ChatCursor, tool_to_fix: str) -> CommandResult:
        toolbox, result = self._toolbox_or_error(cursor)
        if not toolbox:
            return result
        if not tool_to_fix:
            return result.add("Usage: /t fix <name|num>", "warning")
        choice = (
            await self.prompt_user_fn(
                "\nHow do you want to fix the unresolved tool?\n"
                "  1. Try callable first, falling back to external. (Default)\n"
                "  2. Try callable only.\n"
                "  3. Convert to an external tool.\n"
                "Enter choice (1, 2, or 3) [1]: "
            )
        ).strip()
        if choice == "3":
            success, msg = toolbox.resolve_tool_link(
                tool_to_fix,
                search_scope=None,
                external_handler=self.get_external_tool_handler_fn(),
            )
        elif choice == "2":
            success, msg = toolbox.resolve_tool_link(
                tool_to_fix,
                search_scope=dict(self.get_search_scope()),
                external_handler=None,
            )
        elif choice in {"1", ""}:
            success, msg = toolbox.resolve_tool_link(
                tool_to_fix,
                search_scope=dict(self.get_search_scope()),
                external_handler=self.get_external_tool_handler_fn(),
            )
        else:
            success, msg = False, "Invalid choice. Fix cancelled."
        return result.add(msg, "info" if success else "error")

    async def cmd_global(self, cursor: ChatCursor, resolved_mode: str) -> CommandResult:
        result = self._result(cursor)
        toolbox_ref = self.get_toolbox_ref()
        if not toolbox_ref:
            return result.add("Error: Toolbox scope context unavailable.", "error")

        def _update(scope: ToolsScope) -> ToolsScope:
            scope.mode = resolved_mode
            return scope

        toolbox_ref.mutate_scope(_update)
        result.add(f"Context tools mode set to '{resolved_mode}'.")
        return result

    async def cmd_scope_show(self, cursor: ChatCursor) -> CommandResult:
        result = self._result(cursor)
        tools_view = cursor.get_tools_view()
        if not tools_view:
            return result.add("No active tools context available.", "warning")
        entries = collect_tools_scope_entries(cursor)
        result.tools_view = tools_view
        result.scope_entries = entries
        result.scope_effective = self._scope_summary(tools_view, entries)
        return result

    async def cmd_scope_apply(
        self,
        cursor: ChatCursor,
        action: str,
        scope_obj: Optional[ToolsScope],
        *,
        command_text: str,
        stack_id: Optional[str] = None,
    ) -> CommandResult:
        toolbox, result = self._toolbox_or_error(cursor)
        if not toolbox:
            return result
        try:
            if action in {"set", "add"}:
                if not scope_obj:
                    return result.add("No valid scope options provided. Example: mode=silent advertise=search gated=db", "error")
                normalized_scope, warnings = normalize_scope_tool_names(scope_obj, toolbox)
                for warning in warnings:
                    result.add(warning, "warning")
                if normalized_scope.is_noop():
                    return result.add("Scope has no valid settings; command ignored.", "warning")
                cursor.apply_tools_scope(action, normalized_scope, command_text=command_text)
            elif action == "pop":
                cursor.apply_tools_scope("pop", None, command_text=command_text, stack_id=stack_id)
            else:
                return result.add("Unsupported scope action.", "error")
        except ValueError as exc:
            return result.add(str(exc), "error")

        tools_view = cursor.get_tools_view()
        if tools_view:
            entries = collect_tools_scope_entries(cursor)
            result.tools_view = tools_view
            result.scope_entries = entries
            result.scope_effective = self._scope_summary(tools_view, entries)
        return result
