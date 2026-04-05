from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .context_cursor import ChatCursor
from .engine_session import Colors, Command, Turn
from .hosted_tool_visibility import annotate_tool_listing, summarize_effective_tool_view
from mp13_engine.mp13_toolbox import ToolBoxRef, Toolbox, ToolsScope, ToolsView


def _normalized_name_set(items: Optional[Iterable[Any]]) -> set[str]:
    return {
        str(item or "").strip()
        for item in list(items or [])
        if str(item or "").strip()
    }


def all_tool_names(toolbox: Toolbox) -> List[str]:
    try:
        return [entry[0] for entry in toolbox.list_tools()]
    except Exception:
        return []


def tool_wildcard_groups(toolbox: Toolbox) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    if not toolbox:
        return groups
    intrinsic_names = sorted(toolbox.intrinsic_tools.keys()) if getattr(toolbox, "intrinsic_tools", None) else []
    if intrinsic_names:
        groups["*i"] = intrinsic_names
    callable_names = sorted(
        name for name, definition in getattr(toolbox, "tools", {}).items()
        if definition.get("_type") == "callable"
    )
    external_names = sorted(
        name for name, definition in getattr(toolbox, "tools", {}).items()
        if definition.get("_type") == "external"
    )
    if callable_names:
        groups["*c"] = callable_names
    if external_names:
        groups["*e"] = external_names
    return groups


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
        elif key in {"advertise", "adv", "a"}:
            advertise.update(split_tool_arg_list(value))
        elif key in {"silent", "hide", "s"}:
            silent.update(split_tool_arg_list(value))
        elif key in {"disabled", "deny", "d"}:
            disabled.update(split_tool_arg_list(value))
        elif key in {"label", "name", "l"}:
            label = value
    scope = ToolsScope(
        mode=mode,
        advertise_tools=advertise,
        silent_tools=silent,
        disabled_tools=disabled,
        label=label,
    ).clean()
    return None if scope.is_noop() else scope


def normalize_scope_tool_names(scope: ToolsScope, toolbox: Toolbox) -> Tuple[ToolsScope, List[str]]:
    known_names = sorted(set(toolbox.tools.keys()) | set(toolbox.intrinsic_tools.keys()))
    lower_map = {name.lower(): name for name in known_names}

    def resolve_name(raw: str) -> Optional[str]:
        if raw == "*":
            return "*"
        key = raw.lower()
        if key in lower_map:
            return lower_map[key]
        prefix_matches = [name for name in known_names if name.lower().startswith(key)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        non_guides = [name for name in prefix_matches if not name.lower().endswith("_guide")]
        if len(non_guides) == 1:
            return non_guides[0]
        return None

    warnings: List[str] = []

    def normalize_set(names: Set[str]) -> Set[str]:
        normalized: Set[str] = set()
        for raw in names:
            resolved = resolve_name(raw)
            if resolved:
                normalized.add(resolved)
            else:
                warnings.append(f"{Colors.TOOL_WARNING}Warning: Tool '{raw}' not recognized for scope.{Colors.RESET}")
        return normalized

    return (
        ToolsScope(
            mode=scope.mode,
            advertise_tools=normalize_set(scope.advertise_tools),
            silent_tools=normalize_set(scope.silent_tools),
            disabled_tools=normalize_set(scope.disabled_tools),
            label=scope.label,
        ).clean(),
        warnings,
    )


def collect_tools_scope_entries(cursor: ChatCursor) -> List[Tuple[Optional[str], ToolsScope]]:
    if not cursor or not cursor.current_turn:
        return []
    path: List[Turn] = cursor.session.get_active_path_for_llm(cursor.current_turn)
    all_ops: List[Tuple[Turn, Command]] = []
    for turn in path:
        for cmd in getattr(turn, "cmd", []) or []:
            if cmd.cmd_type == Command.STATE_CHANGE and cmd.data.get("change") == "tools_scope":
                all_ops.append((turn, cmd))
    filtered_ops: List[Tuple[Turn, Command]] = []
    for _, cmd in all_ops:
        op = (cmd.data.get("op") or "add").lower()
        if op != "pop":
            filtered_ops.append((cursor.current_turn, cmd))
            continue
        target_id = cmd.data.get("stack_id")
        if not target_id:
            for idx in range(len(filtered_ops) - 1, -1, -1):
                if (filtered_ops[idx][1].data.get("op") or "add").lower() in {"set", "add"}:
                    filtered_ops.pop(idx)
                    break
            continue
        removed = False
        for idx, (_, candidate_cmd) in enumerate(filtered_ops):
            if candidate_cmd.data.get("stack_id") == target_id and candidate_cmd.data.get("change") == "tools_scope":
                filtered_ops.pop(idx)
                removed = True
                break
        if removed:
            continue

    entries: List[Tuple[Optional[str], ToolsScope]] = []
    for _, cmd in filtered_ops:
        op = (cmd.data.get("op") or "add").lower()
        scope_payload = cmd.data.get("scope")
        scope_obj = ToolsScope.from_dict(scope_payload) if scope_payload else None
        stack_id = cmd.data.get("stack_id")
        if op == "add":
            entries.append((stack_id, scope_obj or ToolsScope()))
        elif op == "set":
            entries = [(stack_id, scope_obj)] if scope_obj and not scope_obj.is_noop() else []
        elif op == "pop":
            if entries:
                entries.pop()
        elif op == "reset":
            entries = []
    return entries


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
) -> List[str]:
    resolved_names: List[str] = []
    if not targets_str:
        return []

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
                print(f"Warning: Invalid number '{target}' ignored (out of range).")
        except ValueError:
            resolved_names.append(target)
    return resolved_names


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

    def _hosted_advertised_tool_names(self) -> List[str]:
        hosted = self.get_hosted_summary() or {}
        return [
            str(item or "").strip()
            for item in list(hosted.get("advertised_tool_names") or hosted.get("all_registered_tool_names") or [])
            if str(item or "").strip()
        ]

    def _hosted_hidden_allowed_tool_names(self) -> List[str]:
        hosted = self.get_hosted_summary() or {}
        return [
            str(item or "").strip()
            for item in list(hosted.get("hidden_allowed_tool_names") or [])
            if str(item or "").strip()
        ]

    def _print_scope_summary(self, tools_view: ToolsView, entries: List[Tuple[Optional[str], ToolsScope]]) -> None:
        if entries:
            print(f"{Colors.SYSTEM}Tool scope stack (oldest -> newest):{Colors.RESET}")
            for idx, (stack_id, scope) in enumerate(entries, start=1):
                label = f"{stack_id}: " if stack_id else ""
                print(f"  {idx}. {label}{scope.describe()}")
        else:
            print(f"{Colors.SYSTEM}No active tool scopes. Using context toolbox defaults.{Colors.RESET}")
        effective = summarize_effective_tool_view(
            tools_view,
            hosted_advertised_tool_names=self._hosted_advertised_tool_names(),
            hosted_hidden_allowed_tool_names=self._hosted_hidden_allowed_tool_names(),
        )
        print(f"{Colors.SYSTEM}Tools mode:{Colors.RESET} {tools_view.mode}")
        print(f"{Colors.SYSTEM}Advertised tools:{Colors.RESET} {', '.join(effective['effective_advertised_tools']) or '<none>'}")
        print(f"{Colors.SYSTEM}Hidden but allowed:{Colors.RESET} {', '.join(effective['effective_hidden_allowed_tools']) or '<none>'}")
        print(f"{Colors.SYSTEM}Disabled tools:{Colors.RESET} {', '.join(effective['disabled_tools']) or '<none>'}")
        if self.get_hosted_summary():
            print(f"{Colors.SYSTEM}Hosted execution:{Colors.RESET} active")
            print(f"{Colors.SYSTEM}Hosted-visible tools:{Colors.RESET} {', '.join(effective['hosted_visible_tools']) or '<none>'}")
            print(f"{Colors.SYSTEM}Hosted hidden-allowed tools:{Colors.RESET} {', '.join(effective['hosted_hidden_allowed_tools']) or '<none>'}")
            print(f"{Colors.SYSTEM}Hosted-gated tools:{Colors.RESET} {', '.join(effective['hosted_gated_tools']) or '<none>'}")

    async def handle_tools_command(self, args_str: str, cursor: ChatCursor, pt_session: Any) -> Tuple[ChatCursor, bool]:
        toolbox = self.get_toolbox()
        toolbox_ref = self.get_toolbox_ref()
        if not toolbox:
            print(f"{Colors.ERROR}Error: Toolbox not initialized.{Colors.RESET}")
            return cursor, True
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

        if sub_cmd == "enum":
            tools = toolbox.list_tools()
            self.last_enumerated_tools.clear()
            if not tools:
                print(f"{Colors.TOOL_WARNING}No tools defined. Use '/t new' to add one.{Colors.RESET}")
            else:
                tools_view = cursor.get_tools_view() or cursor.refresh_tools_view()
                rows = annotate_tool_listing(
                    tools,
                    tools_view=tools_view,
                    hosted_advertised_tool_names=self._hosted_advertised_tool_names(),
                    hosted_hidden_allowed_tool_names=self._hosted_hidden_allowed_tool_names(),
                )
                max_name_len = max(len(t[0]) for t in tools) if tools else 30
                name_col_width = max(30, max_name_len + 9)
                print(f"{'Index':<7} {'Name':<{name_col_width}} {'Avail':<8} {'Via':<8} {'Type':<12} {'Description'}")
                print(f"{'-'*5:<7} {'-'*(name_col_width-2):<{name_col_width}} {'-'*5:<8} {'-'*3:<8} {'-'*10:<12} {'-'*58}")
                for idx, row in enumerate(rows):
                    desc_trunc = (str(row['description'])[:57] + "...") if len(str(row["description"])) > 57 else str(row["description"])
                    tool_type = str(row["tool_type"])
                    type_display = f"{Colors.ERROR}{'Unresolved':<12}{Colors.RESET}" if tool_type == "unresolved" else f"{tool_type.capitalize():<12}"
                    name_display = f"  └─ {row['name']}" if row["is_guide"] else f"{'*' if row['is_modified'] and not row['is_guide'] else ' '} {row['name']}"
                    print(f"  {idx+1:<5} {name_display:<{name_col_width}} {str(row['availability']):<8} {str(row['via']):<8} {type_display:<12} '{desc_trunc}'")
                    self.last_enumerated_tools.append(str(row["name"]))
            return cursor, True
        if sub_cmd == "new":
            success, msg = await toolbox.interactive_edit_tool(
                pt_session,
                self.external_tool_handler,
                tool_name_to_edit=None,
                search_scope=dict(self.get_search_scope()),
            )
            print(msg)
            return cursor, True
        if sub_cmd == "modify":
            if not sub_args.strip():
                print(f"Usage: /t modify {Colors.CYAN}[g/]<name|num>{Colors.RESET}")
                return cursor, True
            target_arg = sub_args.strip()
            edit_guide = False
            if target_arg.lower().startswith("g/"):
                edit_guide = True
                target_arg = target_arg[2:]
            tool_name_to_edit = ""
            if target_arg.isdigit():
                idx = int(target_arg) - 1
                if 0 <= idx < len(self.last_enumerated_tools):
                    tool_name_to_edit = self.last_enumerated_tools[idx]
                else:
                    print(f"{Colors.ERROR}Invalid tool number: {target_arg}. Use '/t list'.{Colors.RESET}")
                    return cursor, True
            else:
                tool_name_to_edit = target_arg
            if edit_guide:
                tool_def = toolbox.get_tool(tool_name_to_edit)
                if tool_def and "guide_definition" in tool_def:
                    tool_name_to_edit = tool_def["guide_definition"]["function"]["name"]
                elif f"{tool_name_to_edit}_guide" in toolbox.intrinsic_tools:
                    tool_name_to_edit = f"{tool_name_to_edit}_guide"
                else:
                    print(f"{Colors.ERROR}Error: Could not find a guide function for tool '{tool_name_to_edit}'.{Colors.RESET}")
                    return cursor, True
            success, msg = await toolbox.interactive_edit_tool(
                pt_session,
                self.external_tool_handler,
                tool_name_to_edit=tool_name_to_edit,
                search_scope=dict(self.get_search_scope()),
            )
            print(msg)
            return cursor, True
        if sub_cmd == "replace":
            if not sub_args.strip():
                print(f"Usage: /t replace {Colors.CYAN}<name|num>{Colors.RESET}")
                return cursor, True
            if sub_args.isdigit():
                idx = int(sub_args) - 1
                if 0 <= idx < len(self.last_enumerated_tools):
                    tool_name_to_update = self.last_enumerated_tools[idx]
                else:
                    print(f"{Colors.ERROR}Invalid tool number: {sub_args}. Use '/t list'.{Colors.RESET}")
                    return cursor, True
            else:
                tool_name_to_update = sub_args.strip()
            print(f"Enter the full JSON definition for '{tool_name_to_update}'. Type END_JSON on a new line to finish.")
            json_lines = []
            while True:
                line = await pt_session.prompt_async("")
                if line.strip() == "END_JSON":
                    break
                json_lines.append(line)
            json_string = "\n".join(json_lines)
            if not json_string:
                print("Update cancelled.")
                return cursor, True
            success, msg = toolbox.update_tool_from_json_string(
                tool_name_to_update,
                json_string,
                external_handler=self.external_tool_handler,
                search_scope=dict(self.get_search_scope()),
            )
            print(msg)
            return cursor, True
        if sub_cmd == "print":
            if not sub_args.strip():
                print(f"Usage: /t print {Colors.CYAN}<name|num>{Colors.RESET}")
                return cursor, True
            if sub_args.isdigit():
                idx = int(sub_args) - 1
                if 0 <= idx < len(self.last_enumerated_tools):
                    tool_name_to_print = self.last_enumerated_tools[idx]
                else:
                    print(f"{Colors.ERROR}Invalid tool number: {sub_args}. Use '/t list'.{Colors.RESET}")
                    return cursor, True
            else:
                tool_name_to_print = sub_args.strip()
            tool_def = toolbox.get_tool(tool_name_to_print)
            if tool_def:
                print(f"\n--- Tool Definition: {tool_name_to_print} ---")
                print(json.dumps(tool_def, indent=2))
                print("---")
            else:
                print(f"{Colors.ERROR}Tool '{tool_name_to_print}' not found.{Colors.RESET}")
            return cursor, True
        if sub_cmd in {"activate", "deactivate", "hide", "show", "unregister"}:
            if not sub_args.strip():
                print(f"Usage: /t {sub_cmd} {Colors.CYAN}<name|num|*|*i|*c|*e,...>{Colors.RESET}")
                return cursor, True
            tool_names = await parse_cli_targets(
                sub_args.strip(),
                self.last_enumerated_tools,
                allow_wildcard=True,
                wildcard_values=all_names,
                wildcard_groups=wildcard_groups,
            )
            if not tool_names:
                print(f"{Colors.ERROR}No valid tools specified.{Colors.RESET}" if sub_cmd != "unregister" else "No valid tools specified for unregister.")
                return cursor, True
            if sub_cmd == "activate":
                success, msg = await asyncio.to_thread(toolbox.activate_tool, tool_names)
            elif sub_cmd == "deactivate":
                success, msg = await asyncio.to_thread(toolbox.deactivate_tool, tool_names)
            elif sub_cmd in {"hide", "show"}:
                success, msg = await asyncio.to_thread(toolbox.set_hidden, tool_names, sub_cmd == "hide")
            else:
                success, msg = await asyncio.to_thread(toolbox.delete_tool, tool_names)
            print(msg)
            return cursor, True
        if sub_cmd == "save":
            save_path_str = sub_args.strip()
            if save_path_str:
                target_path = Path(save_path_str).expanduser().resolve()
            else:
                tools_path = (self.get_current_config() or {}).get("tools_config_path")
                if not tools_path:
                    print(f"{Colors.ERROR}No tools config path configured.{Colors.RESET}")
                    return cursor, True
                target_path = Path(tools_path)
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    json.dump(toolbox.to_dict(), f, indent=2)
                print(f"Toolbox state saved to {target_path}")
            except Exception as e:
                print(f"Error saving toolbox state: {e}")
            return cursor, True
        if sub_cmd == "load":
            load_arg = sub_args.strip()
            if not load_arg:
                print(f"Usage: /t load {Colors.CYAN}<file_path|json_string>{Colors.RESET}")
                return cursor, True
            try:
                if load_arg.strip().startswith("{"):
                    data = json.loads(load_arg)
                    toolbox.from_dict(data, search_scope=dict(self.get_search_scope()), external_handler=self.external_tool_handler)
                    print("Toolbox state loaded from JSON string.")
                else:
                    with open(Path(load_arg).expanduser().resolve(), "r", encoding="utf-8") as f:
                        toolbox.from_dict(json.load(f), search_scope=dict(self.get_search_scope()), external_handler=self.external_tool_handler)
                    print(f"Toolbox state loaded from {load_arg}")
            except Exception as e:
                print(f"Error loading toolbox state: {e}")
            return cursor, True
        if sub_cmd == "fix":
            if not sub_args.strip():
                print(f"Usage: /t fix {Colors.CYAN}<name|num>{Colors.RESET}")
                return cursor, True
            tool_names = await parse_cli_targets(sub_args.strip(), self.last_enumerated_tools)
            if not tool_names:
                print(f"{Colors.ERROR}No valid tool specified.{Colors.RESET}")
                return cursor, True
            tool_to_fix = tool_names[0]
            print(f"\nHow do you want to fix the unresolved tool '{Colors.CYAN}{tool_to_fix}{Colors.RESET}'?")
            print(f"  1. Try to re-link as a {Colors.BOLD}'callable'{Colors.RESET} Python function, falling back to 'external' if not found. (Default)")
            print(f"  2. Try to re-link as a {Colors.BOLD}'callable'{Colors.RESET} Python function {Colors.ERROR}only{Colors.RESET}. The command will fail if the function is not found.")
            print(f"  3. Convert to an {Colors.BOLD}'external'{Colors.RESET} tool, using the console input handler.")
            choice = (await pt_session.prompt_async("Enter choice (1, 2, or 3) [1]: ")).strip()
            if choice == "3":
                success, msg = toolbox.resolve_tool_link(tool_to_fix, search_scope=None, external_handler=self.external_tool_handler)
            elif choice == "2":
                success, msg = toolbox.resolve_tool_link(tool_to_fix, search_scope=dict(self.get_search_scope()), external_handler=None)
            elif choice in {"1", ""}:
                success, msg = toolbox.resolve_tool_link(tool_to_fix, search_scope=dict(self.get_search_scope()), external_handler=self.external_tool_handler)
            else:
                success, msg = False, "Invalid choice. Fix cancelled."
            print(msg)
            return cursor, True
        if sub_cmd == "global":
            arg = sub_args.strip().lower()
            if arg in {"?", "help"} or not arg:
                print(f"Usage: /t g[lobal] {Colors.CYAN}<a|s|d>{Colors.RESET} (advertised|silent|disabled)")
                return cursor, True
            mode_alias = {
                "a": "advertised", "adv": "advertised", "advertised": "advertised",
                "s": "silent", "sil": "silent", "silent": "silent",
                "d": "disabled", "dis": "disabled", "disabled": "disabled",
            }
            resolved_mode = mode_alias.get(arg, arg)
            if resolved_mode not in {"advertised", "silent", "disabled"}:
                print(f"Usage: /t g[lobal] {Colors.CYAN}<a|s|d>{Colors.RESET} (advertised|silent|disabled)")
                return cursor, True
            if not toolbox_ref:
                print(f"{Colors.ERROR}Error: Toolbox scope context unavailable.{Colors.RESET}")
                return cursor, True
            def _update(scope: ToolsScope) -> ToolsScope:
                scope.mode = resolved_mode
                return scope
            toolbox_ref.mutate_scope(_update)
            print(f"{Colors.SYSTEM}Context tools mode set to '{resolved_mode}'.{Colors.RESET}")
            return cursor, True
        if sub_cmd == "scope":
            scope_args = sub_args.strip()
            if scope_args in {"?", "help"}:
                self.print_help()
                return cursor, True
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
                tools_view = cursor.get_tools_view()
                if not tools_view:
                    print(f"{Colors.SYSTEM}No active tools context available.{Colors.RESET}")
                else:
                    self._print_scope_summary(tools_view, collect_tools_scope_entries(cursor))
                return cursor, True
            if action in {"set", "add"}:
                if not remainder or remainder in {"?", "help"}:
                    verb = "set" if action == "set" else "add"
                    hint = " (use mode=* to reset to defaults)" if action == "set" else ""
                    print(f"Usage: /t scope {verb} m[ode]=... a[dvertised]=... s[ilent]=... d[isabled]=...{hint}")
                    return cursor, True
                scope_obj = parse_scope_cli_args(remainder)
                if not scope_obj:
                    print(f"{Colors.ERROR}No valid scope options provided. Example: mode=silent advertise=search{Colors.RESET}")
                    return cursor, True
                normalized_scope, warnings = normalize_scope_tool_names(scope_obj, toolbox)
                for warning in warnings:
                    print(warning)
                if normalized_scope.is_noop():
                    print(f"{Colors.TOOL_WARNING}Scope has no valid settings; command ignored.{Colors.RESET}")
                    return cursor, True
                cursor.apply_tools_scope(action, normalized_scope, command_text=command_text)
            elif action == "pop":
                stack_id, _ = parse_pop_target_options(remainder)
                try:
                    cursor.apply_tools_scope("pop", None, command_text=command_text, stack_id=stack_id)
                except ValueError as exc:
                    print(f"{Colors.ERROR}{exc}{Colors.RESET}")
                    return cursor, True
            elif action == "reset":
                print(f"{Colors.SYSTEM}Use '/t scope set mode=*' to reset to the default tools mode.{Colors.RESET}")
                return cursor, True
            else:
                print("Usage: /t scope s[et]|a[dd]|p[op] [options]")
                print(f"{Colors.SYSTEM}Tip: Use '/t scope set mode=*' to reset to defaults.{Colors.RESET}")
                return cursor, True
            if action in {"set", "add", "pop"}:
                tools_view = cursor.get_tools_view()
                if tools_view:
                    self._print_scope_summary(tools_view, collect_tools_scope_entries(cursor))
                return cursor, True
        self.print_help()
        return cursor, True
