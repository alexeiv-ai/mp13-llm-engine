from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .context_cursor import ChatCursor
from .engine_session import Colors
from .session_cursor_toolbox import all_tool_names, tool_wildcard_groups
from .session_toolbox_controller import CommandResult, ControllerMessage, SessionToolboxController
from mp13_engine.mp13_toolbox import ToolBoxRef, Toolbox, ToolsScope


def split_tool_arg_list(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_scope_cli_args(arg_str: str) -> Optional[ToolsScope]:
    arg_str = arg_str.strip()
    if not arg_str:
        return None
    mode = None
    advertise: Set[str] = set()
    silent: Set[str] = set()
    disabled: Set[str] = set()
    gated: Set[str] = set()
    label: Optional[str] = None
    for token in shlex.split(arg_str):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.lower().strip()
        value = value.strip()
        if not value:
            continue
        if key in {"mode", "m"}:
            mode = value.lower()
        elif key in {"advertise", "advertised", "adv", "a"}:
            advertise.update(split_tool_arg_list(value))
        elif key in {"silent", "hide", "s"}:
            silent.update(split_tool_arg_list(value))
        elif key in {"disabled", "deny", "d"}:
            disabled.update(split_tool_arg_list(value))
        elif key in {"gated", "gate", "g"}:
            gated.update(split_tool_arg_list(value))
        elif key in {"label", "name", "l"}:
            label = value
    scope = ToolsScope(
        mode=mode,
        advertise_tools=advertise,
        silent_tools=silent,
        disabled_tools=disabled,
        gated_tools=gated,
        label=label,
    ).clean()
    return None if scope.is_noop() else scope


def parse_pop_target_options(arg_text: str) -> Tuple[Optional[str], bool]:
    stack_id: Optional[str] = None
    force_cmd: bool = False
    if not arg_text:
        return stack_id, force_cmd
    tokens = shlex.split(arg_text)
    if tokens and tokens[0] in {"--cmd", "-c"}:
        force_cmd = True
        tokens = tokens[1:]
    if tokens:
        stack_id = tokens[0]
    return stack_id, force_cmd


async def parse_cli_targets(
    targets_str: str,
    enumerated_list: List[Any],
    name_key: Optional[str] = None,
    *,
    allow_wildcard: bool = False,
    wildcard_values: Optional[List[str]] = None,
    wildcard_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[List[str], List[str]]:
    resolved_names: List[str] = []
    warnings: List[str] = []
    if not targets_str:
        return [], warnings

    normalized_groups: Dict[str, List[str]] = {}
    if allow_wildcard:
        default_values = wildcard_values or [
            item[name_key] if name_key and isinstance(item, dict) else item
            for item in enumerated_list
        ]
        if default_values:
            normalized_groups["*"] = list(dict.fromkeys(default_values))
    if wildcard_groups:
        for key, values in wildcard_groups.items():
            if not values:
                continue
            normalized_groups[key.lower()] = list(dict.fromkeys(values))

    targets = [t.strip() for t in targets_str.split(",") if t.strip()]
    for target in targets:
        lowered = target.lower()
        if allow_wildcard and lowered in normalized_groups:
            for value in normalized_groups[lowered]:
                if value and value not in resolved_names:
                    resolved_names.append(value)
            continue
        try:
            idx = int(target) - 1
            if 0 <= idx < len(enumerated_list):
                item = enumerated_list[idx]
                resolved_names.append(item[name_key] if name_key and isinstance(item, dict) else item)
            else:
                warnings.append(f"Warning: Invalid number '{target}' ignored (out of range).")
        except ValueError:
            resolved_names.append(target)
    return resolved_names, warnings


def print_tools_cli_help() -> None:
    print(f"{Colors.HEADER}--- /t (Tools) Commands ---{Colors.RESET}")
    print("  /t e[num]                  List registered tools.")
    print("  /t h[ide]/sh[ow] <name|num|*|*i|*c|*e,...>  Hide or reveal tools (*i intrinsic, *c callable, *e external).")
    print("  /t a[ctivate]/d[eactivate] <name|num|*...>  Enable or disable tools (same wildcard support).")
    print("  /t u[nregister] <name|num|*...>            Remove tools permanently.")
    print("  /t g[lobal] <a|s|d>         Set tools mode: advertised, silent, or disabled.")
    print("  /t sc[ope] s[et] m[ode]=... a=foo s=bar d=baz  Record a stacked override (mode=* resets to default).")
    print("  /t sc[ope] a[dd] ...        Same syntax as 'set'; pushes a later layer (newest wins).")
    print("  /t sc[ope] p[op] [--cmd] [pop_id|cmd_id|gen_id|anchor_id]  Undo the latest or targeted scope layer.")
    print("  /t sc[ope] [gen_id]         Show current tool view summary for a turn with related commands pop_ids.")
    print("  /t sa[ve]/l[oad]/f[ix]/p[rint]/n[ew]/m[odify] ... (see /help).")


async def default_external_tool_handler(
    async_input_fn: Callable[[str], Awaitable[str]],
    **kwargs: Any,
) -> str:
    toolbox = kwargs.pop("toolbox", None)
    tool = kwargs.pop("tool", None)
    if not tool or not isinstance(tool, dict) or not toolbox:
        return "Error: Interactive handler was called without a valid 'tool' definition object."

    tool_name = tool.get("function", {}).get("name", "unknown_tool")
    tool_args_str = json.dumps(kwargs)
    print(f"\n{Colors.TOOL}--- Tool Call Requires Your Input ---{Colors.RESET}")
    print(f"  {Colors.TOOL}Tool:{Colors.RESET} {tool_name}")
    print(f"  {Colors.TOOL}Arguments:{Colors.RESET} {tool_args_str}")
    print(f"{Colors.TOOL}-------------------------------------------------{Colors.RESET}")
    return await async_input_fn(f"Enter result for {tool_name}: ")


@dataclass
class LightweightToolsCliHandler:
    get_toolbox: Callable[[], Optional[Toolbox]]
    get_toolbox_ref: Callable[[], Optional[ToolBoxRef]]
    get_hosted_summary: Callable[[], Optional[Dict[str, Any]]]
    print_help: Callable[[], None]
    external_tool_handler: Callable[..., Awaitable[str]]
    get_current_config: Callable[[], Optional[Dict[str, Any]]]
    get_search_scope: Callable[[], Mapping[str, Any]]
    last_enumerated_tools: List[str] = field(default_factory=list)

    async def _prompt_multiline(self, prompt_text: str, pt_session: Any) -> str:
        print(prompt_text)
        json_lines = []
        while True:
            line = await pt_session.prompt_async("")
            if line.strip() == "END_JSON":
                break
            json_lines.append(line)
        return "\n".join(json_lines)

    async def _interactive_edit(self, tool_name: Optional[str], context: Dict[str, Any], pt_session: Any) -> Tuple[bool, str]:
        toolbox = self.get_toolbox()
        if not toolbox:
            return False, "Error: Toolbox not initialized."
        return await toolbox.interactive_edit_tool(
            pt_session,
            self.external_tool_handler,
            tool_name_to_edit=tool_name,
            search_scope=dict(context.get("search_scope") or self.get_search_scope()),
        )

    def _controller(self, pt_session: Any) -> SessionToolboxController:
        return SessionToolboxController(
            get_toolbox=self.get_toolbox,
            get_toolbox_ref=self.get_toolbox_ref,
            get_hosted_summary=self.get_hosted_summary,
            get_current_config=self.get_current_config,
            get_search_scope=self.get_search_scope,
            prompt_user_fn=pt_session.prompt_async,
            prompt_multiline_fn=lambda prompt_text: self._prompt_multiline(prompt_text, pt_session),
            interactive_edit_fn=lambda tool_name, context: self._interactive_edit(tool_name, context, pt_session),
            get_external_tool_handler_fn=lambda: self.external_tool_handler,
        )

    def _render_message(self, text: str, level: str) -> None:
        if level == "error":
            print(f"{Colors.ERROR}{text}{Colors.RESET}")
        elif level == "warning":
            print(f"{Colors.TOOL_WARNING}{text}{Colors.RESET}")
        else:
            print(text)

    def _render_listing(self, result: CommandResult) -> None:
        rows = result.listing_rows or []
        if not rows:
            return
        max_name_len = max(len(str(row["name"])) for row in rows) if rows else 30
        name_col_width = max(30, max_name_len + 9)
        print(f"{'Index':<7} {'Name':<{name_col_width}} {'Avail':<8} {'Via':<8} {'Type':<12} {'Description'}")
        print(f"{'-'*5:<7} {'-'*(name_col_width-2):<{name_col_width}} {'-'*5:<8} {'-'*3:<8} {'-'*10:<12} {'-'*58}")
        for idx, row in enumerate(rows):
            desc = str(row["description"])
            desc_trunc = (desc[:57] + "...") if len(desc) > 57 else desc
            tool_type = str(row["tool_type"])
            type_display = f"{Colors.ERROR}{'Unresolved':<12}{Colors.RESET}" if tool_type == "unresolved" else f"{tool_type.capitalize():<12}"
            name_display = f"  └─ {row['name']}" if row["is_guide"] else f"{'*' if row['is_modified'] and not row['is_guide'] else ' '} {row['name']}"
            print(f"  {idx+1:<5} {name_display:<{name_col_width}} {str(row['availability']):<8} {str(row['via']):<8} {type_display:<12} '{desc_trunc}'")

    def _render_scope_summary(self, result: CommandResult) -> None:
        if result.scope_entries is None or result.scope_effective is None or result.tools_view is None:
            return
        entries = result.scope_entries
        if entries:
            print(f"{Colors.SYSTEM}Tool scope stack (oldest -> newest):{Colors.RESET}")
            for idx, (stack_id, scope) in enumerate(entries, start=1):
                label = f"{stack_id}: " if stack_id else ""
                print(f"  {idx}. {label}{scope.describe()}")
        else:
            print(f"{Colors.SYSTEM}No active tool scopes. Using context toolbox defaults.{Colors.RESET}")
        effective = result.scope_effective
        print(f"{Colors.SYSTEM}Tools mode:{Colors.RESET} {result.tools_view.mode}")
        print(f"{Colors.SYSTEM}Advertised tools:{Colors.RESET} {', '.join(effective['effective_advertised_tools']) or '<none>'}")
        print(f"{Colors.SYSTEM}Hidden but allowed:{Colors.RESET} {', '.join(effective['effective_hidden_allowed_tools']) or '<none>'}")
        print(f"{Colors.SYSTEM}Gated tools:{Colors.RESET} {', '.join(effective['effective_gated_tools']) or '<none>'}")
        print(f"{Colors.SYSTEM}Disabled tools:{Colors.RESET} {', '.join(effective['disabled_tools']) or '<none>'}")
        if self.get_hosted_summary():
            print(f"{Colors.SYSTEM}Hosted execution:{Colors.RESET} active")
            print(f"{Colors.SYSTEM}Hosted-visible tools:{Colors.RESET} {', '.join(effective['hosted_visible_tools']) or '<none>'}")
            print(f"{Colors.SYSTEM}Hosted hidden-allowed tools:{Colors.RESET} {', '.join(effective['hosted_hidden_allowed_tools']) or '<none>'}")
            print(f"{Colors.SYSTEM}Hosted route-gated tools:{Colors.RESET} {', '.join(effective['hosted_gated_tools']) or '<none>'}")

    def _render_result(self, result: CommandResult) -> Tuple[ChatCursor, bool]:
        if result.listing_tools is not None:
            self.last_enumerated_tools.clear()
            self.last_enumerated_tools.extend(result.listing_tools)
        self._render_listing(result)
        for message in result.messages:
            self._render_message(message.text, message.level)
        self._render_scope_summary(result)
        return result.cursor, result.handled

    def _resolve_mode(self, sub_args: str) -> Optional[str]:
        arg = sub_args.strip().lower()
        mode_alias = {
            "a": "advertised", "adv": "advertised", "advertised": "advertised",
            "s": "silent", "sil": "silent", "silent": "silent",
            "d": "disabled", "dis": "disabled", "disabled": "disabled",
        }
        return mode_alias.get(arg, arg)

    async def _resolve_single_tool_target(self, sub_args: str) -> Tuple[str, List[str]]:
        tool_names, warnings = await parse_cli_targets(sub_args.strip(), self.last_enumerated_tools)
        return (tool_names[0] if tool_names else ""), warnings

    async def handle_tools_command(self, args_str: str, cursor: ChatCursor, pt_session: Any) -> Tuple[ChatCursor, bool]:
        toolbox = self.get_toolbox()
        if not toolbox:
            print(f"{Colors.ERROR}Error: Toolbox not initialized.{Colors.RESET}")
            return cursor, True
        controller = self._controller(pt_session)
        all_names = all_tool_names(toolbox)
        wildcard_groups = tool_wildcard_groups(toolbox)
        stripped_args = args_str.strip()
        if stripped_args in {"?", "help"}:
            self.print_help()
            return cursor, True

        parts = args_str.split(" ", 1)
        sub_cmd_full = parts[0].lower()
        sub_args = parts[1] if len(parts) > 1 else ""
        sub_cmd_map = {
            "": "enum", "e": "enum", "enum": "enum",
            "n": "new", "new": "new",
            "m": "modify", "modify": "modify",
            "r": "replace", "replace": "replace",
            "p": "print", "print": "print",
            "a": "activate", "activate": "activate",
            "d": "deactivate", "deactivate": "deactivate",
            "h": "hide", "hide": "hide", "hidden": "hide",
            "show": "show", "sh": "show",
            "u": "unregister", "unregister": "unregister",
            "f": "fix", "fix": "fix",
            "save": "save", "sa": "save",
            "load": "load",
            "scope": "scope", "sc": "scope",
            "global": "global", "g": "global", "gl": "global", "mode": "global",
        }
        sub_cmd = sub_cmd_map.get(sub_cmd_full) or sub_cmd_map.get(sub_cmd_full[0] if sub_cmd_full else "")
        if not sub_cmd_full.strip():
            sub_cmd = "enum"
        if not sub_cmd:
            print(f"{Colors.ERROR}Unknown tools command: '{sub_cmd_full}'.{Colors.RESET}")
            print("Valid options are: e[num], n[ew], m[odify], r[eplace], p[rint], a[ctivate], d[eactivate], u[nregister], h[idden], f[ix], sa[ve], l[oad], sc[ope], g[lobal].")
            print("Type '/help' for more details.")
            return cursor, True

        result: CommandResult
        if sub_cmd == "enum":
            result = await controller.cmd_enum(cursor)
        elif sub_cmd == "new":
            result = await controller.cmd_new(cursor)
        elif sub_cmd == "modify":
            result = await controller.cmd_modify(cursor, sub_args, self.last_enumerated_tools)
        elif sub_cmd in {"replace", "print", "fix"}:
            if not sub_args.strip():
                usage = f"Usage: /t {sub_cmd} <name|num>"
                result = CommandResult(cursor=cursor).add(usage, "warning")
            else:
                tool_name, warnings = await self._resolve_single_tool_target(sub_args)
                if sub_cmd == "replace":
                    result = await controller.cmd_replace(cursor, tool_name)
                elif sub_cmd == "print":
                    result = await controller.cmd_print(cursor, tool_name)
                else:
                    result = await controller.cmd_fix(cursor, tool_name)
                for warning in reversed(warnings):
                    result.messages.insert(0, ControllerMessage(warning, "warning"))
        elif sub_cmd in {"activate", "deactivate", "hide", "show", "unregister"}:
            if not sub_args.strip():
                result = CommandResult(cursor=cursor).add(f"Usage: /t {sub_cmd} <name|num|*|*i|*c|*e,...>", "warning")
            else:
                tool_names, warnings = await parse_cli_targets(
                    sub_args.strip(),
                    self.last_enumerated_tools,
                    allow_wildcard=True,
                    wildcard_values=all_names,
                    wildcard_groups=wildcard_groups,
                )
                result = await controller.cmd_tool_state(cursor, sub_cmd, tool_names)
                for warning in reversed(warnings):
                    result.messages.insert(0, ControllerMessage(warning, "warning"))
        elif sub_cmd == "save":
            result = await controller.cmd_save(cursor, sub_args.strip())
        elif sub_cmd == "load":
            result = await controller.cmd_load(cursor, sub_args.strip())
        elif sub_cmd == "global":
            arg = sub_args.strip().lower()
            resolved_mode = self._resolve_mode(sub_args)
            if arg in {"?", "help"} or not arg or resolved_mode not in {"advertised", "silent", "disabled"}:
                result = CommandResult(cursor=cursor).add("Usage: /t g[lobal] <a|s|d> (advertised|silent|disabled)", "warning")
            else:
                result = await controller.cmd_global(cursor, resolved_mode)
        elif sub_cmd == "scope":
            result = await self._handle_scope(controller, cursor, sub_args)
        else:
            self.print_help()
            return cursor, True
        return self._render_result(result)

    async def _handle_scope(
        self,
        controller: SessionToolboxController,
        cursor: ChatCursor,
        sub_args: str,
    ) -> CommandResult:
        scope_args = sub_args.strip()
        if scope_args in {"?", "help"}:
            self.print_help()
            return CommandResult(cursor=cursor)
        action = "show"
        remainder = ""
        if scope_args:
            action_token, _, remainder = scope_args.partition(" ")
            action_token = action_token.lower()
            remainder = remainder.strip()
            action = {
                "set": "set", "s": "set",
                "add": "add", "a": "add",
                "pop": "pop", "p": "pop",
                "reset": "reset", "r": "reset",
                "show": "show", "status": "show", "": "show",
            }.get(action_token, "show")
            if action == "show" and "=" in action_token:
                action, remainder = "set", scope_args
        command_text = f"/t scope {scope_args}" if scope_args else "/t scope"
        if action == "show":
            return await controller.cmd_scope_show(cursor)
        if action in {"set", "add"}:
            if not remainder or remainder in {"?", "help"}:
                verb = "set" if action == "set" else "add"
                hint = " (use mode=* to reset to defaults)" if action == "set" else ""
                return CommandResult(cursor=cursor).add(
                    f"Usage: /t scope {verb} m[ode]=... a[dvertise|d]=... s[ilent]=... d[isabled]=... g[ated]=...{hint}",
                    "warning",
                )
            scope_obj = parse_scope_cli_args(remainder)
            return await controller.cmd_scope_apply(cursor, action, scope_obj, command_text=command_text)
        if action == "pop":
            stack_id, _ = parse_pop_target_options(remainder)
            return await controller.cmd_scope_apply(cursor, "pop", None, command_text=command_text, stack_id=stack_id)
        if action == "reset":
            return CommandResult(cursor=cursor).add("Use '/t scope set mode=*' to reset to the default tools mode.")
        return CommandResult(cursor=cursor).add("Usage: /t scope s[et]|a[dd]|p[op] [options]", "warning")
