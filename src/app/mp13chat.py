# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: MIT
import sys
import atexit
import threading 
import asyncio
import json
import os
import re
import shlex
import time, datetime
import codecs
import signal
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set, AsyncIterator, Mapping, Sequence, Callable
import argparse
import traceback
import logging
import prompt_toolkit
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from dataclasses import dataclass

# ------------------------------------------------------------------------------
# How to add custom tools or command wrappers around mp13chat
#
# Thin wrappers can preload application-specific tools before starting the chat
# loop. Example:
#
#     from src.app.mp13chat import register_tool_callable, main
#     
#     def get_weather(city: str) -> str:
#         """Return a formatted weather summary for the requested city."""
#         ...
#     
#     register_tool_callable(get_weather)
#     
#     if __name__ == "__main__":
#         asyncio.run(main())
#
# `register_tool_callable()` ensures the shared Toolbox exists and registers the
# callable definition (marking it active by default). For interactive/external
# handlers, use `register_tool_external(tool_definition, handler)` instead.
# The helpers can be called multiple times before invoking `main()` to preload
# a suite of application tools.
#
# Custom command workflows currently require wrapping or delegating to
# `handle_command()` directly; there is no dedicated plug-in interface for new
# slash commands. Wrapper applications can intercept user input before handing
# it to `handle_command()` and either short-circuit or post-process the results.
# ------------------------------------------------------------------------------

# --- Fix for UnicodeEncodeError on Windows ---
# Reconfigure stdout/stderr to use UTF-8 encoding if they don't already.
# This prevents errors when printing characters not supported by the default
# console codepage (e.g., cp1252).
if sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        # This works in most cases, especially when running from a standard terminal.
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (TypeError, AttributeError):
        # Fallback for environments where reconfigure is not available or fails.
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Adjust import as needed for your project structure
from mp13_engine import logger as engine_logger
from mp13_engine.mp13_engine_api import handle_call_tool, inference_stream_to_dict_stream
from mp13_engine.mp13_config import EngineMode, ChunkType, InferenceResponse, InferenceRequest, MP13Response, ParserProfile, IfExistsEnum
from mp13_engine.mp13_tools_parser import UnifiedToolIO, ToolsParserHelper
from mp13_engine.mp13_config import ToolCallBlock, ToolCall
from mp13_engine.mp13_config_paths import (
    DEFAULT_CHAT_CONFIG,
    DEFAULT_CATEGORY_DIRS,
    get_default_config_dir,
    get_default_config_path,
    load_json_config,
    load_effective_config,
    load_merged_config,
    resolve_config_paths,
    resolve_custom_config_path,
    resolve_engine_inputs,
    save_json_config,
    extract_engine_params,
    build_engine_init_payload,
)
from .engine_session import EngineSession, Turn, Command, ChatSession, InferenceParams, Colors
from .context_cursor import ChatCursor, ChatContext, ChatContextScope, StreamDisplayContext, StreamDisplayPlan, ChatForks
from mp13_engine.mp13_toolbox import Toolbox, ToolsScope, ToolsView

# --- Constants and globals ---
APP_NAME = "mp13chat"
DEFAULT_CONFIG_DIR = get_default_config_dir()
DEFAULT_CONFIG_FILE = get_default_config_path()
DEFAULT_SESSIONS_DIR_NAME = "sessions"
DEFAULT_ADAPTERS_DIR_NAME = "adapters"

EFFECTIVE_CONFIG_FILE_PATH: Path = DEFAULT_CONFIG_FILE # Will be updated if --config is used

# --- Global State ---
current_config: Optional[Dict[str, Any]] = None
conversation_template: Optional[ChatSession] = None
chat_scope: Optional[ChatContextScope] = None
session_control: Optional["SessionControl"] = None 
toolbox: Optional[Toolbox] = None
DEBUG_MODE_ENABLED: bool = False # New global flag for debug mode
LAST_ENUMERATED_ADAPTERS: List[Dict[str, Any]] = [] # For selection by number from engine {'name': str, 'details': dict}
LAST_ENUMERATED_TOOLS: List[str] = []
LAST_ENUMERATED_LOCAL_ADAPTERS: List[str] = [] # For selection by number from local disk
LAST_ENUMERATED_SESSIONS: Dict[str, str] = {} # Maps hierarchical number string to full path
LAST_EXCEPTION_TRACEBACK: Optional[str] = None # To store the last exception traceback
REPLAY_CANCELLATION_EVENT = asyncio.Event()
# Global flag to track engine initialization status ---
ENGINE_INITIALIZED_SUCCESSFULLY = False
ENGINE_PARSER_PROFILE: Optional[ParserProfile] = None

TOOL_AUTO_TRYOUT_KIND = "auto_tool_tryout"
CONTINUE_AUTO_TRYOUT_KIND = "auto_continue_tryout"

DEFAULT_CHAT_PARAMS: Dict[str, Any] = {
    "stream": True,
    "cache": None,
    "return_prompt": None,
    "generation_config_template": {},
    "max_new_tokens": None,
    "no_tools_parse": False,
    "auto_retry_truncated": False,
    "suppress_full_response": False,
    "results_as_user_role": False,
    "pack_results_as_one_role": False,
    "advertised_tools": [],
    "silent_tools": [],
    "disabled_tools": [],
    "auto_tool_retry_limit": 5,
    "auto_continue_retry_limit": 10,
    "global_tools_mode": "advertised",
    "tools_config_path": "mp13tools.json",
}

# ---------------------------------- A Tool definition example ------------------------------

# Tool guidance for built-in examples:
# - The docstring below feeds directly into the advertised tool description; keep it concise and phrased for LLM instructions.
# - `kwargs` exposes helper fields: `toolbox`, `tool`, `tool_call`, `tools_view`, `context`, `tool_retries_max`, `tool_retries_left`,
#   plus any executor passthrough (e.g., `pt_session`, `final_response_items`, etc.).
# - You can mutate the passed `tool_call`: rewrite actions (e.g., ToolCall.KeepRaw to preserve non normalized call format),
#   assign value directly to `tool_call.result` if you want to bypass default handling of return vs exception.
# - Raising will surface as a formatted tool call.error, while returning a value yields a standard tool_result
#   payload that prefers tool.result unless action has ToolCall.Retry to prefer call.error if both are present.
def SimpleCalc(
    expr: Optional[str] = None, 
    **kwargs
):
    """
    A simple calculator that evaluates a Python numerical expression string.

    Args:
        expr (str): The mathematical expression to evaluate. Example: '2 + 2 * 10'. It does NOT support 'import' statements

    Returns:
        The numerical result of the expression or an error message.
    """
    expression_to_eval = expr
    tool_args_issue = kwargs.get('tool_args_issue')
    tool_call = kwargs.get('tool_call')
    recovery_used = False
    if not expression_to_eval and tool_args_issue:
        # Standard recovery path for malformed arguments.
        raw_val = tool_args_issue.get('_non_parsed') or tool_args_issue.get('_string_value')
        recovered_expr = None
        if isinstance(raw_val, str):
            try:
                # Direct string search for the expression is more robust than parsing broken JSON.
                expr_marker = '"expr": "'
                start_index = raw_val.rfind(expr_marker)
                if start_index != -1:
                    start_index += len(expr_marker)
                    esc = False
                    end_index = -1
                    for i in range(start_index, len(raw_val)):
                        ch = raw_val[i]
                        if esc:
                            esc = False
                            continue
                        if ch == "\\":
                            esc = True
                            continue
                        if ch == '"':
                            end_index = i
                            break
                    if end_index > start_index:
                        recovered_expr = raw_val[start_index:end_index]
                    else:
                        recovered_expr = raw_val[start_index:]
                else: # Fallback to old regex if direct search fails
                    unescaped_val = codecs.decode(raw_val, 'unicode_escape')
                    quoted_strings = re.findall(r'"((?:\\.|[^"\\])*)"', unescaped_val)
                    if quoted_strings:
                        recovered_expr = quoted_strings[-1]
                    else:
                        recovered_expr = unescaped_val
            except Exception:
                recovered_expr = raw_val
        elif isinstance(raw_val, dict):
             recovered_expr = raw_val.get('expr')
        
        if recovered_expr:
            expression_to_eval = recovered_expr
            recovery_used = True
            if tool_call and 'KeepRaw' not in tool_call.action:
                tool_call.action.append('KeepRaw')

    if not expression_to_eval:
        return "Error: Expression not provided. Use 'expr' or provide a raw string argument."
    
    expr_text = str(expression_to_eval).strip()
    # Provide clearer feedback instead of a generic SyntaxError when statements are present.
    if re.search("(?m)^\\s*(from|import)\\b", expr_text):
        return ("Error evaluating expression: import statements are not supported. "
                "Provide a single mathematical expression (e.g., '2 + 2').")
    if ';' in expr_text or re.search(r'(?<![<>!=])=(?![=])', expr_text):
        return ("Error evaluating expression: assignments or multi-step scripts are not supported. "
                "Use 'scriptable_calculator' for multi-line calculations.")

    try:
        # For safety, use a more restricted eval environment if possible,
        # but for this simple calculator, standard eval is used as per original.
        return eval(expr_text)
    except SyntaxError as e:
        if recovery_used:
            return ("Error evaluating expression: malformed or truncated tool call arguments. "
                    "Please resend the tool call with a complete 'expr' string.")
        error_str = tool_call.normalize_error(e) if tool_call else str(e)
        return (f"Error evaluating expression: {error_str}. The input must be a single Python expression "
                "without statements or assignments.")
    except Exception as e:
        if recovery_used:
            return ("Error evaluating expression: malformed or truncated tool call arguments. "
                    "Please resend the tool call with a complete 'expr' string.")
        error_str = tool_call.normalize_error(e) if tool_call else str(e)
        return f"Error evaluating expression: {error_str}"

# ---------------------------------- END Tool definition example ------------------------------


# ------------ Helper methods -------------
def batch_list(data: list, num_batches: int) -> list[list]:
    """Splits a list into a specified number of roughly equal-sized batches."""
    if not data or num_batches <= 0:
        return []
    # If more batches than items, each item becomes a batch.
    if num_batches >= len(data):
        return [[item] for item in data]
    
    k, m = divmod(len(data), num_batches)
    return [data[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(num_batches)]


def _generate_try_out_name(prefix: str = "tryout", *, width: int = 6) -> str:
    """Return a short, timestamp-derived try-out name suffix."""
    suffix = str(int(time.time() * 1000))[-width:]
    return f"{prefix}_{suffix}"


async def _find_resurrectable_anchors(cursor: ChatCursor):
    """Finds and lists all unique anchor names from the session history."""
    if not cursor.context:
        print(f"{Colors.ERROR}No active context available.{Colors.RESET}")
        return

    print(f"{Colors.SYSTEM}Scanning session for past try_out anchors...{Colors.RESET}")
    all_anchor_turns = cursor.context.find_all_turns_with_anchor_data()

    if not all_anchor_turns:
        print("No past try-out anchors found in session history.")
        return

    found_anchors = {}
    for turn in all_anchor_turns:
        meta = turn.data.get("$try_out", {})
        name = meta.get("anchor")
        if name:
            # Store first occurrence's turn for context
            if name not in found_anchors:
                found_anchors[name] = {
                    "kind": meta.get("kind", "try_out"),
                    "first_seen": turn.gen_id_or_parent,
                }

    if not found_anchors:
        print("No past try-out anchors with names found.")
        return

    print("Found resurrectable try-out anchors:")
    # Use a temporary dictionary for enumeration to avoid conflicting with other commands
    enumerated_anchors = {}
    for i, (name, data) in enumerate(found_anchors.items()):
        kind = data['kind']
        turn_id = data['first_seen']
        print(f"  {i+1}. {name} (kind: {kind}, first seen near {turn_id})")
        enumerated_anchors[str(i+1)] = name

    # Store the enumeration in a global or pass it directly
    # For now, we are just printing and the user has to use the name
    print(f"\nUse {Colors.CYAN}/try --resurrect <num|name>{Colors.RESET} to bring an anchor back into the active context.")
    # To enable selection by number, we would store `enumerated_anchors` in a global
    # that the --resurrect handler can access. For now, this is a simplified demo.
    global LAST_ENUMERATED_SESSIONS
    LAST_ENUMERATED_SESSIONS = enumerated_anchors


def _find_try_out_anchor_name(turn: Optional[Turn]) -> Optional[str]:
    """Return the anchor label stored on this turn, if any."""
    if not turn:
        return None
    try_meta = getattr(turn, "data", {}).get("$try_out") or {}
    if isinstance(try_meta, dict):
        anchor_name = try_meta.get("anchor")
        if anchor_name:
            return str(anchor_name)
    return None


def _is_active_try_out_anchor(anchor_name: Optional[str], cursor: Optional[ChatCursor]) -> bool:
    if not anchor_name:
        return False
    ctx = getattr(cursor, "context", None) if cursor else None
    if not ctx:
        ctx = _active_chat_context()
    if not ctx:
        return False
    scope = getattr(cursor, "scope", None) if cursor else None
    try:
        anchor = ctx.get_try_out_anchor(anchor_name, scope=scope) if scope else ctx.get_try_out_anchor(anchor_name)
        return anchor is not None
    except Exception:
        return False


def _cursor_labels_for_turn(turn: Optional[Turn], cursor: Optional[ChatCursor]) -> List[str]:
    """Return any registered cursor handle labels that currently point at `turn`."""
    if not turn or not cursor:
        return []
    ctx = getattr(cursor, "context", None) or _active_chat_context()
    if not ctx:
        return []
    try:
        if hasattr(ctx, "owns_turn") and not ctx.owns_turn(turn):
            return []
    except Exception:
        return []

    scope = getattr(cursor, "scope", None)
    if scope and getattr(scope, "cursors_snapshot", None):
        items = scope.cursors_snapshot()
    else:
        items = ctx.cursors_snapshot()

    labels: List[str] = []
    for handle, cur in items:
        if cur and getattr(cur, "head", None) is turn:
            head_id = getattr(cur.head, "gen_id_or_parent", None)
            label = f"{handle}(gen_id={head_id})" if head_id else handle
            labels.append(label)

    if getattr(cursor, "context_id", None) and cursor.current_turn is turn:
        head_id = getattr(turn, "gen_id_or_parent", None)
        label = f"{cursor.context_id}(gen_id={head_id})" if head_id else cursor.context_id
        if label not in labels:
            labels.append(label)

    seen = set()
    ordered: List[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _format_context_markers(turn: Optional[Turn], cursor: Optional[ChatCursor]) -> List[str]:
    """Return contextual markers (cursor/try-out) for display surfaces."""
    markers: List[str] = []
    for label in _cursor_labels_for_turn(turn, cursor):
        markers.append(f"cursor:{label}")
    anchor_name = _find_try_out_anchor_name(turn)
    if anchor_name:
        markers.append(f"try:{anchor_name}")
    else:
        node = turn
        while node:
            try_meta = getattr(node, "data", {}).get("$try_out") or {}
            if isinstance(try_meta, dict):
                role = try_meta.get("role")
                anchor = try_meta.get("anchor")
                if role == "try" and anchor:
                    if _is_active_try_out_anchor(str(anchor), cursor):
                        markers.append(f"try:{anchor}")
                    break
                if role == "main":
                    break
            node = getattr(node, "parent", None)
    return markers


def _format_context_marker(marker: str) -> str:
    """Colorize context markers by type (cursor vs try-out)."""
    if marker.startswith("cursor:"):
        color = Colors.BRIGHT_CYAN
    elif marker.startswith("try:"):
        color = Colors.SYSTEM
    else:
        color = Colors.SYSTEM
    return f"{color}[{marker}]{Colors.RESET}"

def _append_summary_context_flags(summary: str, turn: Turn, cursor: ChatCursor) -> str:
    """Attach context markers to summary-style strings."""
    markers = _format_context_markers(turn, cursor)
    if not markers:
        return summary
    colored = " ".join(_format_context_marker(m) for m in markers)
    return f"{summary} {colored}"


def _summary_context_markers(turn: Optional[Turn], cursor: Optional[ChatCursor]) -> str:
    """Return colored context markers for summary lines."""
    markers = _format_context_markers(turn, cursor)
    if not markers:
        return ""
    colored = " ".join(_format_context_marker(m) for m in markers)
    return f" {colored}"



def _ensure_global_toolbox() -> Toolbox:
    """Create the shared toolbox if it does not yet exist and return it."""
    global toolbox
    if toolbox is None:
        toolbox = Toolbox()
    return toolbox


def _ensure_chat_session_toolbox(chat_session: ChatSession) -> Optional[Toolbox]:
    """Ensure the provided ChatSession references the active toolbox instance."""
    global toolbox
    if not chat_session:
        return toolbox
    active_toolbox = _ensure_global_toolbox()
    if chat_session.toolbox and chat_session.toolbox is not active_toolbox:
        chat_session.toolbox = active_toolbox
    elif chat_session.toolbox is None:
        chat_session.toolbox = active_toolbox
    return chat_session.toolbox

def _active_toolbox_ref() -> Optional["ToolBoxRef"]:
    context = _active_chat_context()
    if context and getattr(context, "toolbox_ref", None):
        return context.toolbox_ref
    return None

def _active_toolbox() -> Optional[Toolbox]:
    ref = _active_toolbox_ref()
    if ref:
        return ref.toolbox
    return toolbox


def get_global_toolbox() -> Toolbox:
    """
    Public helper for thin wrappers that need to inspect or mutate the shared toolbox.
    Ensures the toolbox exists before returning it.
    """
    return _ensure_global_toolbox()


def register_tool_callable(
    func: Callable[..., Any],
    *,
    activate: bool = True,
) -> Tuple[bool, str]:
    """
    Register a Python callable as a tool before launching the chat loop.

    Example:
        from mp13chat import register_tool_callable

        def current_time(city: str) -> str:
            ...

        register_tool_callable(current_time)
    """
    toolbox_obj = _ensure_global_toolbox()
    return toolbox_obj.add_tool_callable(func, activate=activate)


def register_tool_external(
    tool_definition: Dict[str, Any],
    handler: Callable[..., Any],
    *,
    activate: bool = True,
    allow_override: bool = False,
) -> Tuple[bool, str]:
    """
    Register an external (interactive) tool definition that forwards to the provided handler.
    """
    toolbox_obj = _ensure_global_toolbox()
    return toolbox_obj.add_tool_external(
        tool_definition,
        handler,
        activate=activate,
        allow_override=allow_override,
    )

def _reset_chat_context(session: EngineSession, conversation: ChatSession, active_turn: Optional[Turn] = None) -> ChatCursor:
    """(Re)build ChatContext and return a cursor pinned to active_turn."""
    global chat_scope, toolbox
    _ensure_chat_session_toolbox(conversation)
    _sync_inference_defaults_from_initial(conversation)
    _apply_tool_params_to_toolbox(conversation.initial_params)
    _apply_parser_profile_flags(conversation)
    context = ChatContext(
        session,
        chat_session=conversation,
        toolbox=toolbox,
    )
    chat_scope = context.create_scope(label="chat", auto_mark_active=False)
    if active_turn:
        try:
            root_cursor = (
                chat_scope.register_cursor_for_turn(active_turn, make_active=True)
                if chat_scope else context.register_cursor_for_turn(active_turn, make_active=True)
            )
            if root_cursor:
                _scope_for_cursor(root_cursor)
                return root_cursor
        except Exception:
            pass
    root_cursor = chat_scope.active_cursor() if chat_scope else context.active_cursor
    _scope_for_cursor(root_cursor)
    return root_cursor

def _update_active_conversation_state(cursor: Optional[ChatCursor]) -> None:
    """Persist the active conversation index and head turn on the owning session."""
    if not cursor:
        return
    chat_session = cursor.chat_session
    session = cursor.session
    if chat_session and chat_session in session.conversations:
        session.last_converation = session.conversations.index(chat_session)
        chat_session.last_active_turn = cursor.current_turn or chat_session.root_turn

def _resolve_target_turn_for_conversation(session: EngineSession, chat_session: ChatSession) -> Tuple[Optional[Turn], Optional[str]]:
    """
    Resolve the best target turn for a conversation using its stored last_active_turn,
    falling back to the main branch leaf and returning an optional warning message.
    """
    idx = session.conversations.index(chat_session) if chat_session in session.conversations else None
    main_leaf = session.get_main_branch_leaf(idx) if idx is not None else None
    preferred = getattr(chat_session, "last_active_turn", None)
    if preferred:
        try:
            if any(t is preferred for t in session._get_all_turns(chat_session)):
                return preferred, None
        except Exception:
            pass
        warning = f"{Colors.TOOL_WARNING}Warning: Could not resolve stored active turn for conversation {idx + 1 if idx is not None else '?'}; defaulting to main branch.{Colors.RESET}"
        return main_leaf or chat_session.root_turn, warning
    return main_leaf or chat_session.root_turn, None


def _seed_chat_session_flags(chat_session: ChatSession, values: Optional[Dict[str, Any]] = None) -> None:
    chat_session.initial_params = chat_session.initial_params or {}
    seeds = _normalize_chat_params(values if values is not None else _conversation_param_defaults())
    mapping = {
        "stream": seeds.get("stream"),
        "cache_override": seeds.get("cache"),
        "return_prompt_mode": seeds.get("return_prompt"),
        "generation_config_template": seeds.get("generation_config_template"),
        "max_new_tokens_override": seeds.get("max_new_tokens"),
        "no_tools_parse": seeds.get("no_tools_parse"),
        "auto_retry_truncated": seeds.get("auto_retry_truncated"),
        "suppress_full_response": seeds.get("suppress_full_response"),
        "results_as_user_role": seeds.get("results_as_user_role"),
        "pack_results_as_one_role": seeds.get("pack_results_as_one_role"),
        "advertised_tools": copy.deepcopy(seeds.get("advertised_tools")),
        "silent_tools": copy.deepcopy(seeds.get("silent_tools")),
        "disabled_tools": copy.deepcopy(seeds.get("disabled_tools")),
        "auto_tool_retry_limit": seeds.get("auto_tool_retry_limit"),
        "auto_continue_retry_limit": seeds.get("auto_continue_retry_limit"),
    }
    for key, val in mapping.items():
        if key in chat_session.initial_params:
            continue
        chat_session.initial_params[key] = copy.deepcopy(val)

def _summarize_param_resets(current_params: Dict[str, Any], default_params: Dict[str, Any], max_items: int = 12) -> List[str]:
    normalized_current = copy.deepcopy(current_params)
    normalized_default = copy.deepcopy(default_params)
    if normalized_current.get("system_message") is None:
        normalized_current["system_message"] = ""
    if normalized_default.get("system_message") is None:
        normalized_default["system_message"] = ""
    if "stream" not in normalized_current and "streaming" in normalized_current:
        normalized_current["stream"] = normalized_current.pop("streaming")
    if "stream" not in normalized_default and "streaming" in normalized_default:
        normalized_default["stream"] = normalized_default.pop("streaming")
    normalized_current.pop("engine_base_model_name", None)
    normalized_default.pop("engine_base_model_name", None)
    changed_keys = [
        key for key in sorted(set(normalized_current.keys()) | set(normalized_default.keys()))
        if normalized_current.get(key, None) != normalized_default.get(key, None)
    ]
    diffs: List[str] = []
    for key in changed_keys[:max_items]:
        cur_val = normalized_current.get(key, None)
        def_val = normalized_default.get(key, None)
        cur_display = _clip_long_message(str(cur_val), max_len=60)
        def_display = _clip_long_message(str(def_val), max_len=60)
        diffs.append(f"{key}: {cur_display} -> {def_display}")
    remaining = len(changed_keys) - len(diffs)
    if remaining > 0:
        diffs.append(f"... (+{remaining} more)")
    return diffs

def _summarize_effective_state_resets(
    current_system: Optional[str],
    target_system: Optional[str],
    current_adapters: List[str],
    target_adapters: List[str],
    *,
    default_system: Optional[str] = None,
) -> List[str]:
    diffs: List[str] = []
    cur_sys = _format_system_message_display(current_system, default_value=default_system)
    tgt_sys = _format_system_message_display(target_system, default_value=default_system)
    if cur_sys != tgt_sys:
        diffs.append(f"system_message: {cur_sys} -> {tgt_sys}")
    cur_adapters = ", ".join(current_adapters) if current_adapters else "__base__"
    tgt_adapters = ", ".join(target_adapters) if target_adapters else "__base__"
    if cur_adapters != tgt_adapters:
        diffs.append(f"active_adapters: {cur_adapters} -> {tgt_adapters}")
    return diffs

def _record_loaded_adapter_commands(
    cursor: ChatCursor,
    loaded_adapters: List[Dict[str, Any]],
    *,
    command_label: str,
) -> int:
    """Record load-adapter commands on the current turn for session consistency."""
    recorded = 0
    for adapter in loaded_adapters:
        name = adapter.get("name")
        path = adapter.get("root_path") or adapter.get("checkpoint_path")
        if not name or not path:
            continue
        payload = {
            "adapter_name": name,
            "adapter_path": path,
            "if_exists": IfExistsEnum.IGNORE,
        }
        response = {"status": "success", "data": {"adapter_name": name, "adapter_path": path}}
        cursor.save_api_command("load-adapter", payload, command_text=f"{command_label} {name}", response=response)
        recorded += 1
    return recorded

def _sync_inference_defaults_from_initial(chat_session: ChatSession) -> None:
    """Align inference_defaults from the chat session initial params."""
    params = getattr(chat_session, "initial_params", {}) or {}
    inf = chat_session.inference_defaults or InferenceParams()
    inf.stream = bool(params.get("stream", params.get("streaming", inf.stream)))
    inf.cache = params.get("cache_override", inf.cache)
    inf.return_prompt = params.get("return_prompt_mode", inf.return_prompt)
    inf.generation_config = copy.deepcopy(params.get("generation_config_template") or inf.generation_config or {})
    max_new = params.get("max_new_tokens") if params.get("max_new_tokens") is not None else params.get("max_new_tokens_override")
    if max_new is not None:
        if inf.generation_config is None:
            inf.generation_config = {}
        inf.generation_config["max_new_tokens"] = max_new
    inf.suppress_full_response = bool(params.get("suppress_full_response", inf.suppress_full_response))
    inf.no_tools_parse = bool(params.get("no_tools_parse", inf.no_tools_parse))
    chat_session.inference_defaults = inf


def _apply_parser_profile_flags(chat_session: ChatSession) -> None:
    profile = getattr(chat_session, "parser_profile", None)
    if not profile:
        return
    params = getattr(chat_session, "initial_params", {}) or {}
    if "results_as_user_role" in params:
        profile.results_as_user_role = bool(params.get("results_as_user_role"))
    if "pack_results_as_one_role" in params:
        profile.pack_results_as_one_role = bool(params.get("pack_results_as_one_role"))


def _current_param_snapshot() -> Dict[str, Any]:
    base = _conversation_param_defaults()
    context = _active_chat_context()
    if context:
        snapshot = copy.deepcopy(context.get_param_snapshot())
        base.update(snapshot)
        return base
    if current_config and isinstance(current_config, dict):
        return base
    return _default_new_conversation_params()


def _get_param_value(key: str, default: Any = None) -> Any:
    context = _active_chat_context()
    if context:
        return context.get_param(key, default)
    return DEFAULT_CHAT_PARAMS.get(key, default)


def _set_chat_param_value(key: str, value: Any, *, command_text: Optional[str] = None) -> None:
    context = _active_chat_context()
    if not context:
        try:
            context = _require_current_cursor().context
        except Exception:
            context = None
    if not context:
        return
    normalized_value = copy.deepcopy(value)
    context.set_param(key, normalized_value, command_text=command_text)


def _set_cursor_reset_metrics(cursor: Optional[ChatCursor], enabled: bool, *, command_text: Optional[str] = None) -> None:
    target = cursor or (_require_current_cursor() if chat_scope else None)
    if not target or not hasattr(target, "reset_metrics_override"):
        return
    target.reset_metrics_override = bool(enabled)


def _cursor_reset_metrics_enabled(cursor: Optional[ChatCursor] = None) -> bool:
    target = cursor or (_require_current_cursor() if chat_scope else None)
    if not target or not hasattr(target, "reset_metrics_override"):
        return False
    return bool(getattr(target, "reset_metrics_override", False))


def _generation_config_template_snapshot() -> Dict[str, Any]:
    template = _get_param_value("generation_config_template") or {}
    return copy.deepcopy(template)


def _auto_retry_enabled() -> bool:
    context = _active_chat_context()
    if context:
        return bool(context.get_param("auto_retry_truncated", False))
    return bool(DEFAULT_CHAT_PARAMS.get("auto_retry_truncated", False))

def _auto_tool_retry_limit() -> int:
    context = _active_chat_context()
    if context:
        return int(context.get_param("auto_tool_retry_limit", DEFAULT_CHAT_PARAMS.get("auto_tool_retry_limit", 3)))
    return int(DEFAULT_CHAT_PARAMS.get("auto_tool_retry_limit", 3))

def _auto_continue_retry_limit() -> int:
    context = _active_chat_context()
    if context:
        return int(context.get_param("auto_continue_retry_limit", DEFAULT_CHAT_PARAMS.get("auto_continue_retry_limit", 10)))
    return int(DEFAULT_CHAT_PARAMS.get("auto_continue_retry_limit", 10))

def _auto_anchor_name(kind: str, cursor: ChatCursor) -> str:
    """Build a deterministic anchor name for auto try-outs based on turn lineage."""
    turn = cursor.current_turn
    label = None
    if turn:
        label = getattr(turn, "gen_id", None) or getattr(turn, "gen_id_or_parent", None)
        if not label:
            parent = getattr(turn, "parent", None)
            if parent:
                try:
                    idx = list(getattr(parent, "turns", []) or []).index(turn)
                except Exception:
                    idx = None
                parent_label = getattr(parent, "gen_id", None) or getattr(parent, "gen_id_or_parent", None)
                if parent_label:
                    label = f"{parent_label}_child{idx}" if idx is not None else parent_label
    if not label:
        label = cursor.display_id()
    return f"{kind}_{label}"


def _max_new_tokens_value() -> Optional[int]:
    value = _get_param_value("max_new_tokens")
    return value if value is not None else None


def _current_streaming_enabled() -> bool:
    value = _get_param_value("stream")
    return bool(value) if value is not None else True


def _current_cache_override() -> Optional[str]:
    return _get_param_value("cache")


def _current_return_prompt_mode() -> Optional[str]:
    return _get_param_value("return_prompt")


def _current_no_tools_parse() -> bool:
    return bool(_get_param_value("no_tools_parse"))


def _current_suppress_full_response() -> bool:
    return bool(_get_param_value("suppress_full_response"))


def _bootstrap_cursor_for_session(session: EngineSession, *, conversation: Optional[ChatSession] = None) -> ChatCursor:
    """Initialize chat scope for `session` and land on its main leaf."""
    target_index = 0
    if conversation and conversation in session.conversations:
        target_index = session.conversations.index(conversation)
    elif not conversation:
        if 0 <= getattr(session, "last_converation", 0) < session.get_conversations_count():
            target_index = session.last_converation
    chat_session = conversation or session.get_conversation(target_index)
    _ensure_chat_session_toolbox(chat_session)
    target_turn, warning_msg = _resolve_target_turn_for_conversation(session, chat_session)
    if chat_session in session.conversations:
        session.last_converation = session.conversations.index(chat_session)
    root_cursor = _reset_chat_context(session, chat_session, target_turn or chat_session.root_turn)
    if warning_msg:
        print(warning_msg)
    context = _active_chat_context()
    if target_turn is not root_cursor.current_turn and context:
        try:
            scope = _scope_for_cursor(root_cursor)
            rebound = (
                scope.register_cursor_for_turn(target_turn, make_active=True)
                if scope else context.register_cursor_for_turn(target_turn, make_active=True)
            )
            if rebound:
                root_cursor = rebound
        except Exception:
            pass
    _set_active_cursor(root_cursor)
    _scope_for_cursor(root_cursor)
    _update_active_conversation_state(root_cursor)
    return root_cursor


def _require_current_cursor() -> ChatCursor:
    if chat_scope:
        return chat_scope.active_cursor()
    raise RuntimeError("No active cursor is set.")

def _active_chat_context() -> Optional[ChatContext]:
    return chat_scope.context if chat_scope else None

def _log_cursor_warning(message: str, exc: Optional[Exception] = None) -> None:
    if not DEBUG_MODE_ENABLED:
        return
    suffix = f" ({exc})" if exc else ""
    print(f"{Colors.TOOL_WARNING}Cursor warning: {message}{suffix}{Colors.RESET}")

def _set_active_cursor(cursor: Optional[ChatCursor], *, transient: bool = False) -> Optional[ChatCursor]:
    if not cursor:
        return None
    if chat_scope:
        if (
            not transient
            and chat_scope.active_cursor_override is None
            and (chat_scope.active_cursor_ref is cursor or chat_scope.active_cursor_id == cursor.context_id)
        ):
            return cursor
        try:
            if transient:
                chat_scope.set_active_override(cursor)
                return cursor
            chat_scope.set_active_override(None)
            return chat_scope.set_active_cursor(cursor)
        except Exception as exc:
            _log_cursor_warning("failed to set active cursor", exc)
            return cursor
    return cursor

def _scope_for_cursor(cursor: Optional[ChatCursor]) -> Optional[ChatContextScope]:
    if chat_scope and (cursor is None or cursor.context is chat_scope.context):
        if cursor and cursor.scope is not chat_scope:
            cursor.scope = chat_scope
        return chat_scope
    if cursor and getattr(cursor, "scope", None):
        return cursor.scope
    if cursor and cursor.context:
        return cursor.context.default_scope
    return None

def _require_live_chat_scope(cursor: Optional[ChatCursor] = None) -> ChatContextScope:
    if not chat_scope:
        raise RuntimeError("Chat scope is not initialized.")
    if cursor and cursor.context and cursor.context is not chat_scope.context:
        raise RuntimeError("Cursor context does not match active chat scope.")
    return chat_scope

def _resolve_context_for_scope(
    scope: Optional[ChatContextScope],
    ctx: Optional[ChatContext] = None,
    cursor: Optional[ChatCursor] = None,
) -> Optional[ChatContext]:
    if scope:
        return scope.context
    if ctx:
        return ctx
    if cursor and cursor.context:
        return cursor.context
    return None

def _resolve_bag_dict(
    scope: Optional[ChatContextScope],
    ctx: Optional[ChatContext] = None,
    cursor: Optional[ChatCursor] = None,
) -> Optional[Dict[str, Any]]:
    if scope:
        return scope.bag_dict
    context = _resolve_context_for_scope(scope, ctx, cursor)
    if context and isinstance(getattr(context, "bag_dict", None), dict):
        return context.bag_dict
    return None


def _cleanup_cursor_registry(scope: Optional[ChatContextScope], ctx: Optional[ChatContext]) -> int:
    """Drop duplicate cursors and stale batch-hub handles when safe."""
    if not scope or not ctx:
        return 0
    protected: Set[str] = set()
    if scope.active_cursor_id:
        protected.add(scope.active_cursor_id)
    for cursor in (scope.active_cursor_ref, scope.active_cursor_override):
        if cursor and cursor.context_id:
            protected.add(cursor.context_id)
    try:
        for anchor in scope.try_out_anchors_snapshot():
            if anchor.origin_cursor_id:
                protected.add(anchor.origin_cursor_id)
            for handle in anchor.try_out_cursor_ids or []:
                protected.add(handle)
    except Exception:
        pass
    batch_hubs: Set[Turn] = set()
    bag_dict = _resolve_bag_dict(scope, ctx)
    if bag_dict:
        for entry in bag_dict.get("batch_stack", []) or []:
            for handle in entry.get("fork_handles") or []:
                protected.add(handle)
            for cursor in entry.get("owner_cursors") or []:
                if cursor and getattr(cursor, "context_id", None):
                    protected.add(cursor.context_id)
            hub_cursor = entry.get("hub_cursor")
            if hub_cursor and getattr(hub_cursor, "context_id", None):
                protected.add(hub_cursor.context_id)
            hub_turn = entry.get("hub_turn")
            if hub_turn:
                batch_hubs.add(hub_turn)
    anchors = []
    try:
        anchors = scope.try_out_anchors_snapshot()
    except Exception:
        anchors = []
    dropped = 0
    grouped: Dict[Union[int, str], List[Tuple[str, ChatCursor]]] = {}
    for handle, cursor in scope.cursors_snapshot():
        head = cursor.current_turn if cursor else None
        if not head:
            continue
        key = getattr(head, "gen_id", None) or getattr(head, "gen_id_or_parent", None) or id(head)
        grouped.setdefault(key, []).append((handle, cursor))
    for items in grouped.values():
        if len(items) <= 1:
            continue
        keep_handle = None
        if scope.active_cursor_id and any(handle == scope.active_cursor_id for handle, _ in items):
            keep_handle = scope.active_cursor_id
        if not keep_handle:
            keep_handle = items[0][0]
        for handle, _ in items:
            if handle == keep_handle:
                continue
            try:
                for anchor in anchors:
                    if getattr(anchor, "origin_cursor_id", None) == handle:
                        anchor.origin_cursor_id = None
                    if getattr(anchor, "try_out_cursor_ids", None) and handle in anchor.try_out_cursor_ids:
                        anchor.try_out_cursor_ids = [h for h in anchor.try_out_cursor_ids if h != handle]
                scope.drop_cursor(handle)
                dropped += 1
            except Exception:
                pass
    for handle, cursor in scope.cursors_snapshot():
        if handle in protected:
            continue
        head = cursor.current_turn if cursor else None
        if not head or head in batch_hubs:
            continue
        if getattr(head, "turn_type", None) != Turn.BATCH:
            continue
        try:
            scope.drop_cursor(handle)
            dropped += 1
        except Exception:
            pass
    return dropped

def _chat_session_template() -> ChatSession:
    global conversation_template    
    if not conversation_template:
        raise RuntimeError("No conversation_template is set.")
    return conversation_template

def _get_console_log_level() -> int:
    """Gets the current level of the console handler."""
    for handler in engine_logger.handlers:
        if handler.name == "console_handler":
            return handler.level
    # If no console handler, printing is safe as it won't be redundant.
    return logging.CRITICAL + 1

def _clip_long_message(msg: str, max_len: int = 80, clip_len: int = 35) -> str:
    """Clips a long message to 'start...end' format."""
    if len(msg) > max_len:
        start = msg[:clip_len]
        end = msg[-clip_len:]
        return f"{start}...{end}"
    return msg


def _format_replay_command_preview(
    command_text: Optional[str],
    *,
    stack_id: Optional[str] = None,
    extra_text: Optional[str] = None,
    max_len: int = 80,
) -> str:
    prefix = f"[{stack_id}] " if stack_id else ""
    base = (command_text or "").strip()
    if base:
        combined = f"{prefix}{base}"
    else:
        combined = prefix.rstrip()
    if extra_text:
        combined = f"{combined} {extra_text}".strip()
    return _clip_long_message(combined, max_len=max_len)

def _fork_entry_cursor(entry: Any) -> Optional[ChatCursor]:
    """Resolve a ChatCursor from a batch entry using ChatForks + cursor_idx."""
    if isinstance(entry, dict):
        fork_obj = entry.get("fork")
        cursor_idx = entry.get("cursor_idx")
        if fork_obj and cursor_idx is not None and cursor_idx < len(getattr(fork_obj, "cursors", []) or []):
            active = getattr(fork_obj, "active_cursors", None)
            if active and cursor_idx < len(active) and active[cursor_idx]:
                return active[cursor_idx]
            if getattr(fork_obj, "cursors", None) and cursor_idx < len(fork_obj.cursors):
                return fork_obj.cursors[cursor_idx]
            return None
    return None

def _fork_entry_original_index(entry: Any, default_idx: int = 0) -> int:
    """Resolve the original prompt index for a batch entry using ChatForks metadata."""
    if isinstance(entry, dict):
        if "original_index" in entry:
            return entry.get("original_index", default_idx)
        fork_obj = entry.get("fork")
        cursor_idx = entry.get("cursor_idx")
        prompt_indices = getattr(fork_obj, "prompt_indices", []) if fork_obj else []
        if fork_obj and cursor_idx is not None and cursor_idx < len(prompt_indices):
            return prompt_indices[cursor_idx]
    return default_idx

def _drop_batch_entry_cursors(
    entry: Dict[str, Any],
    *,
    ctx: Optional[ChatContext],
    keep_cursor: Optional[ChatCursor],
    scope: Optional[ChatContextScope] = None,
) -> int:
    if not scope:
        raise RuntimeError("Chat scope is required to drop batch entry cursors.")
    dropped = 0
    forks = entry.get("forks") or []
    for fork in forks:
        fork_cursors = list(getattr(fork, "cursors", []) or [])
        kept = []
        for c in fork_cursors:
            if keep_cursor and c is keep_cursor:
                kept.append(c)
                continue
            try:
                scope.drop_cursor(c)
            except Exception:
                pass
            dropped += 1
        if kept:
            fork.cursors = kept
            fork.main_cursor = kept[0]
        else:
            try:
                fork.drop_cursors()
            except Exception:
                pass
    return dropped

def _drop_batch_entry_handles(
    entry: Dict[str, Any],
    *,
    ctx: Optional[ChatContext],
    keep_cursor: Optional[ChatCursor],
    scope: Optional[ChatContextScope] = None,
) -> int:
    if not scope:
        raise RuntimeError("Chat scope is required to drop batch entry handles.")
    handles = entry.get("fork_handles") or []
    if not handles:
        return 0
    keep_handle = getattr(keep_cursor, "context_id", None) if keep_cursor else None
    dropped = 0
    for handle in list(handles):
        if keep_handle and handle == keep_handle:
            continue
        try:
            scope.drop_cursor(handle)
        except Exception:
            pass
        dropped += 1
    return dropped

def _ensure_registered_cursor(cursor: Optional[ChatCursor]) -> Optional[ChatCursor]:
    """Ensure the cursor is registered with its context when available."""
    if not cursor:
        return None
    ctx = cursor.context
    if not ctx:
        return cursor
    try:
        head = cursor.current_turn or cursor.head
        if head:
            head_type = getattr(head, "turn_type", None)
            if head_type in {Turn.BATCH, Turn.FORK, Turn.RESERVED}:
                return cursor
            try:
                if hasattr(ctx, "owns_turn") and not ctx.owns_turn(head):
                    return cursor
            except Exception:
                pass
            existing = ctx.find_cursor_for_turn(head)
            if existing:
                return existing
    except Exception:
        pass
    try:
        scope = _require_live_chat_scope(cursor)
        return scope.register_cursor_if_needed(cursor, make_active=False)
    except Exception:
        return cursor

def _pop_batch_stack_entry(
    ctx: Optional[ChatContext],
    batch_gen_id: Optional[str],
    *,
    scope: Optional[ChatContextScope] = None,
) -> Optional[Dict[str, Any]]:
    bag_dict = _resolve_bag_dict(scope, ctx)
    if not bag_dict:
        return None
    batch_stack = bag_dict.get("batch_stack", [])
    if not batch_stack:
        return None
    if not batch_gen_id:
        return batch_stack.pop()
    for idx in range(len(batch_stack) - 1, -1, -1):
        entry = batch_stack[idx]
        hub_cursor = entry.get("hub_cursor")
        hub_turn = entry.get("hub_turn")
        gen_id = None
        if hub_cursor:
            gen_id = getattr(hub_cursor.current_turn, "gen_id", None)
        if not gen_id and hub_turn:
            gen_id = getattr(hub_turn, "gen_id", None)
        if gen_id == batch_gen_id:
            return batch_stack.pop(idx)
    return None

def _conversation_index(session: EngineSession, chat_session: Optional[ChatSession]) -> Optional[int]:
    if not chat_session:
        return None
    try:
        return session.conversations.index(chat_session)
    except ValueError:
        return None

def _conversation_label(session: EngineSession, chat_session: Optional[ChatSession]) -> str:
    idx = _conversation_index(session, chat_session)
    prefix = f"Conversation {idx + 1}" if idx is not None else "Conversation"
    title = (getattr(chat_session, "title", "") or "").strip()
    return f"{prefix}: {title}" if title else prefix


def _engine_parser_profile() -> Optional[ParserProfile]:
    """Return a copy of the parser profile reported by the engine during init, if known."""
    if ENGINE_PARSER_PROFILE is None:
        return None
    try:
        return copy.deepcopy(ENGINE_PARSER_PROFILE)
    except Exception:
        return ENGINE_PARSER_PROFILE


def _effective_parser_profile(cursor: Optional[ChatCursor]) -> Optional[ParserProfile]:
    """Resolve the best parser profile for reconstructing tool block text."""
    profile = None
    if cursor:
        profile = getattr(cursor, "parser_profile", None)
        if not profile and cursor.chat_session:
            profile = cursor.chat_session.parser_profile
    return profile or _engine_parser_profile()


def _suppress_truncated_tool_blocks(
    text: str,
    tool_blocks: Optional[List[ToolCallBlock]],
    *,
    cursor: Optional[ChatCursor],
) -> Tuple[str, Optional[List[ToolCallBlock]]]:
    if not tool_blocks:
        return text, tool_blocks
    print(f"{Colors.TOOL_WARNING}Warning: tool calls are suppressed for truncated response.{Colors.RESET}")
    profile = _effective_parser_profile(cursor)
    reconstructed = ToolsParserHelper.reconstruct_text_with_blocks(text, tool_blocks, profile=profile)
    block_lines: List[str] = []
    for block in tool_blocks:
        insert_content = block.raw_block or block.normalized_block or ""
        effective_profile = profile
        if effective_profile:
            start_marker = block.start_marker or (effective_profile.block_start[0] if effective_profile.block_start else "")
            if block.end_marker:
                end_marker = block.end_marker
            elif block.hard_stop_marker or block.is_incomplete:
                end_marker = ""
            else:
                end_marker = effective_profile.block_end[0] if effective_profile.block_end else ""
            if start_marker and not insert_content.startswith(start_marker):
                insert_content = f"{start_marker}{insert_content}"
            if end_marker and not insert_content.endswith(end_marker):
                insert_content = f"{insert_content}{end_marker}"
        if insert_content:
            block_lines.append(insert_content)
    if block_lines:
        print(f"{Colors.TOOL_WARNING}Suppressed tool block(s):{Colors.RESET}")
        print(f"{Colors.LLM_CONTENT}{'\n'.join(block_lines)}{Colors.RESET}")
    return reconstructed, None


def _normalize_chat_params(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = copy.deepcopy(DEFAULT_CHAT_PARAMS)
    if not params:
        return base
    for key, value in params.items():
        if key == "streaming" and "stream" not in params:
            base["stream"] = copy.deepcopy(value)
            continue
        base[key] = copy.deepcopy(value)
    return base


def _default_new_conversation_params() -> Dict[str, Any]:
    return _normalize_chat_params()


def _conversation_param_defaults() -> Dict[str, Any]:
    config_params = None
    if current_config and isinstance(current_config, dict):
        config_params = current_config.get("inference_params")
    return _normalize_chat_params(config_params)


def _resolve_tool_targets(values: Optional[Sequence[str]], toolbox: Toolbox) -> Set[str]:
    if not values:
        return set()
    tool_names = set(_all_tool_names(toolbox))
    wildcard_groups = _tool_wildcard_groups(toolbox)
    resolved: Set[str] = set()
    for val in values:
        if not isinstance(val, str):
            continue
        candidate = val.strip()
        if not candidate:
            continue
        lower = candidate.lower()
        if lower in wildcard_groups:
            resolved.update(wildcard_groups[lower])
            continue
        if candidate == "*":
            resolved.update(tool_names)
            continue
        if candidate in tool_names:
            resolved.add(candidate)
    return resolved


def _apply_tool_params_to_toolbox(initial_params: Dict[str, Any]) -> None:
    global toolbox
    initial_params = initial_params or {}
    active_toolbox = _ensure_global_toolbox()

    # default state of the toolbox as most permissive.
    global_mode = initial_params.get("global_tools_mode", "advertised")
    active_toolbox.set_global_tools_mode(global_mode)

    advertised = _resolve_tool_targets(initial_params.get("advertised_tools"), active_toolbox)
    silent = _resolve_tool_targets(initial_params.get("silent_tools"), active_toolbox)
    disabled = _resolve_tool_targets(initial_params.get("disabled_tools"), active_toolbox)

    active_tool_names = set(active_toolbox.active_tool_names)
    hidden_tool_names = set(active_toolbox.hidden_tool_names)

    active_tool_names -= disabled
    hidden_tool_names -= disabled

    active_tool_names |= silent
    hidden_tool_names |= silent

    active_tool_names |= advertised
    hidden_tool_names -= advertised

    active_toolbox.active_tool_names = sorted(active_tool_names)
    active_toolbox.hidden_tool_names = sorted(hidden_tool_names)

def _tool_call_has_error(calls_or_blocks: List[Any]) -> bool:
    """Return True if any tool call or block reports a structured error."""
    if not calls_or_blocks:
        return False
    for item in calls_or_blocks:
        if isinstance(item, ToolCallBlock):
            if getattr(item, "error_block", None):
                return True
            for c in getattr(item, "calls", []) or []:
                if getattr(c, "error", None):
                    return True
        elif getattr(item, "error", None):
            return True
    return False

def _tool_retry_counters(cursor: Optional[ChatCursor]) -> Tuple[Optional[int], Optional[int]]:
    """Return (max, remaining) retries for auto-tool anchors, falling back to defaults."""
    anchor_limit = _auto_tool_retry_limit()
    if cursor and cursor.context:
        scope = _scope_for_cursor(cursor)
        if scope:
            anchor = scope.find_active_anchor("auto_tool", cursor)
        else:
            anchor = cursor.context.find_active_anchor("auto_tool", cursor)
        if anchor:
            return anchor.retry_limit, anchor.retries_remaining
    return anchor_limit, anchor_limit

def _tool_blocks_have_results(blocks: List[ToolCallBlock]) -> bool:
    """Check whether any block carries executable results or errors."""
    if not blocks:
        return False
    for block in blocks:
        if getattr(block, "parse_errors", None):
            return True
        if getattr(block, "error_block", None):
            return True
        for call in getattr(block, "calls", []) or []:
            if getattr(call, "result", None) is not None or getattr(call, "error", None) or getattr(call, "parse_errors", None):
                return True
    return False

def _tool_blocks_have_abort(blocks: List[ToolCallBlock]) -> bool:
    """Return True if any block/call requests abort."""
    if not blocks:
        return False
    for block in blocks:
        block_actions = list(getattr(block, "action_block", []) or [])
        if ToolCall.Abort in block_actions:
            return True
        for call in getattr(block, "calls", []) or []:
            call_actions = list(getattr(call, "action", []) or [])
            # A single call requesting abort aborts the whole round, regardless of block-level overrides.
            if ToolCall.Abort in call_actions:
                return True
    return False

def _resolve_non_placeholder_cursor(cursor: ChatCursor) -> ChatCursor:
    """Walk up to the nearest non-placeholder turn."""
    resolved = cursor
    while resolved.current_turn and resolved.current_turn.IsPlaceholderLike:
        parent_candidate = resolved.parent_cursor()
        if not parent_candidate:
            break
        resolved = parent_candidate
    return resolved

def _retry_target_cursor(cursor: ChatCursor) -> Optional[ChatCursor]:
    """
    Return the nearest ancestor cursor that carries user/tool input (non-placeholder).
    This guards retry from landing on the root placeholder/off:root nodes.
    """
    probe = cursor
    while probe and probe.current_turn:
        turn = probe.current_turn
        if not turn.IsPlaceholderLike and (turn.data.get("user") or turn.data.get("tool_results")):
            return probe
        probe = probe.parent_cursor()
    return None

def _collect_turn_subtree(root: Turn) -> List[Turn]:
    """Return a list containing `root` and all descendant turns."""
    stack: List[Turn] = [root]
    collected: List[Turn] = []
    while stack:
        node = stack.pop()
        collected.append(node)
        stack.extend(list(getattr(node, "turns", []) or []))
    return collected

def _prune_descendants_after_turn(turn: Turn, session: EngineSession) -> int:
    """
    Remove all descendant turns beneath `turn` and drop associated commands.
    Returns the number of Turn nodes removed.
    """
    children = list(getattr(turn, "turns", []) or [])
    if not children:
        return 0
    removed_nodes: List[Turn] = []
    for child in children:
        removed_nodes.extend(_collect_turn_subtree(child))
    for node in removed_nodes:
        if node.parent and node in node.parent.turns:
            node.parent.turns.remove(node)
        node.parent = None
    turn.turns.clear()
    if removed_nodes and hasattr(session, "commands_history"):
        removed_ids = {id(node) for node in removed_nodes}
        session.commands_history = [
            cmd for cmd in session.commands_history
            if getattr(cmd, "parent", None) is None or id(cmd.parent) not in removed_ids
        ]
    return len(removed_nodes)

def _clear_turn_outputs_for_retry(turn: Turn) -> None:
    """Remove assistant/tool response payloads so the turn can be retried."""
    preserve_keys = {"user", "tool_results"}
    # Rebuild the data map to guarantee assistant metadata is dropped.
    turn.data = {k: v for k, v in (turn.data or {}).items() if k in preserve_keys}
    turn.metrics = {}
    turn.was_truncated = False
    turn.was_canceled = False

def _is_echo_command(cmd: Command) -> bool:
    """Check if a command is an echo command."""
    if not cmd or cmd.cmd_type != Command.COMMAND:
        return False
    if cmd.data.get("$Action") == "echo":
        return True
    command = cmd.data.get("command", "") or ""
    return command.strip().startswith("/echo")

def _normalize_system_message_value(raw_value: Optional[str], value_kind: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalize legacy/system message markers into structured form.
    Returns a tuple of (value, kind) where kind is one of {default, remove, None}.
    """
    kind = (value_kind or "").lower() or None
    value = raw_value
    if not kind and isinstance(value, str):
        if value == "<>":
            kind = "remove"
            value = None
        elif value == "<def>":
            kind = "default"
            value = None
    if kind not in {None, "default", "remove"}:
        kind = None
    return value, kind

def _format_system_message_value(raw_value: Optional[str], value_kind: Optional[str] = None) -> str:
    """Return a human-readable description for a system message payload."""
    value, kind = _normalize_system_message_value(raw_value, value_kind)
    if kind == "default":
        return "<default>"
    if kind == "remove":
        return "<None>"
    if value is None:
        return "<None>"
    if value == "":
        return "''"
    return f"'{_clip_long_message(str(value), max_len=80)}'"

def _format_system_message_display(
    raw_value: Optional[str],
    *,
    value_kind: Optional[str] = None,
    default_value: Optional[str] = None,
) -> str:
    """Format a system message with default/empty/None distinctions."""
    if value_kind:
        display = _format_system_message_value(raw_value, value_kind)
        if display.startswith("<default") and default_value is not None:
            display = f"<default:{_format_system_message_value(default_value)}>"
        return display
    if raw_value == default_value:
        return f"<default:{_format_system_message_value(raw_value)}>"
    return _format_system_message_value(raw_value)

def _format_state_change_annotations(node: Turn, *, show_logs: bool = False) -> List[str]:
    """Returns human-readable annotations for state changes and commands on a node."""
    annotations: List[str] = []
    for cmd in getattr(node, "cmd", []):
        handled = False
        if cmd.cmd_type == Command.STATE_CHANGE and cmd.data.get("change") == "adapters_command":
            op = cmd.data.get("op", "set")
            adapters_field = cmd.data.get("adapters")
            stack_id = cmd.data.get("stack_id")
            if op == "pop":
                suffix = f" ({stack_id})" if stack_id else ""
                annotations.append(f"[{cmd.gen_id or 'pending'}] adapters pop{suffix}")
            else:
                prefix = f"[{stack_id}] " if stack_id else ""
                if isinstance(adapters_field, str):
                    adapters = [adapters_field]
                else:
                    adapters = list(adapters_field or [])
                if not adapters:
                    adapters = ["__base__"]
                adapter_display = ", ".join(adapters)
                annotations.append(f"{prefix}[{cmd.gen_id or 'pending'}] adapters {op} -> {adapter_display}")
            handled = True
        elif cmd.cmd_type == Command.PARAM_CHANGE and cmd.data.get("change") == "system_message":
            op = cmd.data.get("op") or ("pop" if cmd.data.get("value") is None and cmd.data.get("new_value") is None else "set")
            if op == "pop":
                stack_id = cmd.data.get("stack_id")
                suffix = f" ({stack_id})" if stack_id else ""
                annotations.append(f"[{cmd.gen_id or 'pending'}] system pop{suffix}")
            else:
                stack_id = cmd.data.get("stack_id")
                prefix = f"[{stack_id}] " if stack_id else ""
                raw_value = cmd.data.get("value", cmd.data.get("new_value"))
                value_display = _format_system_message_value(raw_value, cmd.data.get("value_kind"))
                annotations.append(f"{prefix}[{cmd.gen_id or 'pending'}] system {op} -> {value_display}")
            handled = True
        elif cmd.cmd_type == Command.STATE_CHANGE and cmd.data.get("change") == "tools_scope":
            op = (cmd.data.get("op") or "add").lower()
            is_root_ctx = bool(cmd.data.get("context_scope"))
            scope_desc = ""
            scope_payload = cmd.data.get("scope")
            if scope_payload:
                try:
                    scope_obj = ToolsScope.from_dict(scope_payload)
                    scope_desc = scope_obj.describe()
                except Exception:
                    scope_desc = str(scope_payload)
            if op == "pop":
                stack_id = cmd.data.get("stack_id")
                suffix = f" ({stack_id})" if stack_id else ""
                prefix = "root context tools" if is_root_ctx else "tools"
                annotations.append(f"[{cmd.gen_id or 'pending'}] {prefix} pop{suffix}")
            else:
                stack_id = cmd.data.get("stack_id")
                prefix = f"[{stack_id}] " if stack_id else ""
                label = "root context tools" if is_root_ctx else "tools"
                text = f"{prefix}[{cmd.gen_id or 'pending'}] {label} {op}"
                if scope_desc:
                    text += f" -> {scope_desc}"
                annotations.append(text)
            handled = True

        if not handled and cmd.cmd_type in (Command.STATE_CHANGE, Command.PARAM_CHANGE):
            data_str = _clip_long_message(json.dumps(cmd.data))
            annotations.append(f"[{cmd.gen_id or 'pending'}] {cmd.cmd_type.lower()} -> {data_str}")
        elif cmd.cmd_type == Command.COMMAND:
            command_text = cmd.data.get("command", cmd.data.get("text", "unknown_command"))
            clipped = _clip_long_message(command_text, max_len=120, clip_len=50)
            suffix = ""
            cmd_lower = str(command_text).strip().lower()
            if cmd.api_name == "get-aggregate-metrics":
                tracked_line = EngineSession.format_selected_engine_metrics(getattr(cmd, "data", {}) or {})
                if tracked_line:
                    suffix = f" -> {tracked_line}"
            annotations.append(f"[{cmd.gen_id or 'pending'}] cmd -> {clipped}{suffix}")
        elif show_logs and cmd.cmd_type == Command.LOG:
            log_text = cmd.data.get("text", cmd.data.get("command", "unknown_log"))
            clipped = _clip_long_message(log_text, max_len=120, clip_len=50)
            annotations.append(f"[{cmd.gen_id or 'pending'}] log -> {clipped}")
    return annotations


def _split_tool_arg_list(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _parse_scope_cli_args(arg_str: str) -> Optional[ToolsScope]:
    """
    Parses CLI arguments for /t scope set/add commands.
    Example: /t scope set mode=silent advertise=search silent=calc disabled=db
    """
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
            advertise.update(_split_tool_arg_list(value))
        elif key in {"silent", "hide", "s"}:
            silent.update(_split_tool_arg_list(value))
        elif key in {"disabled", "deny", "d"}:
            disabled.update(_split_tool_arg_list(value))
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


def _normalize_scope_tool_names(scope: ToolsScope, toolbox: Toolbox) -> Tuple[ToolsScope, List[str]]:
    """
    Resolves user-entered tool names against the toolbox, supporting '*' wildcards
    and case-insensitive unique prefixes. Returns normalized scope plus warnings.
    """
    if not toolbox:
        return scope, []

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
        if len(prefix_matches) > 1:
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

    normalized_scope = ToolsScope(
        mode=scope.mode,
        advertise_tools=normalize_set(scope.advertise_tools),
        silent_tools=normalize_set(scope.silent_tools),
        disabled_tools=normalize_set(scope.disabled_tools),
        label=scope.label,
    ).clean()

    return normalized_scope, warnings


def _collect_tools_scope_entries(cursor: ChatCursor) -> List[Tuple[Optional[str], ToolsScope]]:
    """Return effective tools scope stack entries with stack_ids."""
    if not cursor or not cursor.current_turn:
        return []
    session = cursor.session
    path: List[Turn] = session.get_active_path_for_llm(cursor.current_turn)
    if not path:
        return []

    all_ops: List[Tuple[Turn, Command]] = []
    for turn in path:
        for cmd in getattr(turn, "cmd", []) or []:
            if cmd.cmd_type == Command.STATE_CHANGE and cmd.data.get("change") == "tools_scope":
                all_ops.append((turn, cmd))

    filtered_ops: List[Tuple[Turn, Command]] = []
    for _, cmd in all_ops:
        op_type = (cmd.data.get("op") or "add").lower()
        if op_type != "pop":
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

def _print_tools_scope_summary(cursor: ChatCursor, tools_view: ToolsView, entries: List[Tuple[Optional[str], ToolsScope]]) -> None:
    """Prints the current tools scope stack and resolved permissions."""
    if entries:
        print(f"{Colors.SYSTEM}Tool scope stack (oldest -> newest):{Colors.RESET}")
        for idx, (stack_id, scope) in enumerate(entries, start=1):
            label = f"{stack_id}: " if stack_id else ""
            print(f"  {idx}. {label}{scope.describe()}")
    else:
        print(f"{Colors.SYSTEM}No active tool scopes. Using context toolbox defaults.{Colors.RESET}")

    advertised = ", ".join(sorted(tools_view.advertised_tools)) or "<none>"
    hidden_allowed = ", ".join(sorted(tools_view.hidden_allowed_tools)) or "<none>"
    disabled = ", ".join(sorted(tools_view.disabled_tools)) or "<none>"
    print(f"{Colors.SYSTEM}Tools mode:{Colors.RESET} {tools_view.mode}")
    print(f"{Colors.SYSTEM}Advertised tools:{Colors.RESET} {advertised}")
    print(f"{Colors.SYSTEM}Hidden but allowed:{Colors.RESET} {hidden_allowed}")
    print(f"{Colors.SYSTEM}Disabled tools:{Colors.RESET} {disabled}")


def _format_tools_scope_header(cursor: ChatCursor) -> str:
    """Returns a one-line summary of the current tools view."""
    base = f"{Colors.SYSTEM}Tools Scope:{Colors.RESET} "
    if not cursor.toolbox:
        return f"{base}{Colors.SYSTEM}<toolbox unavailable>{Colors.RESET}"
    try:
        tools_view = cursor.get_tools_view()
        if not tools_view:
            return f"{base}{Colors.SYSTEM}<no tools view>{Colors.RESET}"
    except Exception:
        return f"{base}{Colors.SYSTEM}<error computing view>{Colors.RESET}"

    advertised = ", ".join(sorted(tools_view.advertised_tools)) or "<none>"
    hidden = ", ".join(sorted(tools_view.hidden_allowed_tools)) or "<none>"
    disabled = ", ".join(sorted(tools_view.disabled_tools)) or "<none>"
    return (
        f"{base}{Colors.SYSTEM}"
        f"mode={tools_view.mode}; adv={advertised}; hidden={hidden}; disabled={disabled}"
        f"{Colors.RESET}"
    )

def _print_try_out_summary(cursor: ChatCursor) -> None:
    """Prints tracked try-out anchors and registered cursors for the active context."""
    ctx = cursor.context
    if not ctx:
        return
    scope = _scope_for_cursor(cursor)
    if not scope:
        return
    anchors = scope.try_out_anchors_snapshot()
    if anchors:
        print(f"{Colors.SYSTEM}Try-outs:{Colors.RESET} " + ", ".join(
            f"{a.anchor_name}({a.kind}, branches={len(a.try_out_turns)})" for a in anchors
        ))
    registered = scope.cursors_snapshot()
    if registered:
        cursor_labels = []
        for handle, cur in registered:
            label = handle
            head_id = getattr(cur.head, "gen_id_or_parent", None) if cur and cur.head else None
            if head_id:
                label = f"{handle}(gen_id={head_id})"
            cursor_labels.append(f"{Colors.SYSTEM}{label}{Colors.RESET}")
        print(f"{Colors.SYSTEM}Cursors:{Colors.RESET} " + ", ".join(cursor_labels))


def _print_tools_scope_header(cursor: ChatCursor) -> None:
    line = _format_tools_scope_header(cursor)
    if line:
        print(line)
    _print_try_out_summary(cursor)


def _set_active_fallback(ctx: ChatContext, scope: Optional[ChatContextScope]) -> ChatCursor:
    """Ensure the context has an active cursor, preferring 'main' or first registered."""
    if not scope:
        raise RuntimeError("Chat scope is required for active cursor fallback.")
    try:
        return scope.active_cursor()
    except Exception:
        pass
    registered = scope.cursors_snapshot()
    if not registered:
        raise RuntimeError("ChatContext has no registered cursors.")
    handles = [h for h, _ in registered]
    target = "main" if any(h == "main" for h in handles) else handles[0]
    return scope.set_active_cursor(target)


def _handle_switch_command(args_str: str, cursor: ChatCursor) -> Tuple[ChatCursor, bool]:
    """Handle /sw command: list, switch, add by gen_id, or drop tracked cursors."""
    ctx = cursor.context
    if not ctx:
        print(f"{Colors.ERROR}No active chat context; cannot switch cursors.{Colors.RESET}")
        return cursor, True
    scope = _require_live_chat_scope(cursor)

    tokens = [t for t in args_str.split() if t]
    if not tokens:
        # List cursors
        print(f"{Colors.SYSTEM}Tracked cursors:{Colors.RESET}")
        for handle, cur in scope.cursors_snapshot():
            active_handle = scope.active_cursor_id()
            marker = "*" if handle == active_handle else " "
            head_id = cur.head.gen_id if cur.head else "N/A"
            print(f" {marker} {handle} -> {head_id}")
        return cursor, True

    if tokens[0] in ("--drop", "-d"):
        if len(tokens) < 2:
            print(f"{Colors.ERROR}Usage: /sw --drop <cursor_name>{Colors.RESET}")
            return cursor, True
        name = tokens[1]
        try:
            scope.drop_cursor(name)
            try:
                active = _set_active_fallback(ctx, scope)
                cursor = active
            except Exception:
                pass
            print(f"{Colors.SYSTEM}Dropped cursor '{name}'.{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.ERROR}Could not drop cursor '{name}': {e}{Colors.RESET}")
        return cursor, True

    if tokens[0] in ("--gen_id", "-g"):
        if len(tokens) < 2:
            print(f"{Colors.ERROR}Usage: /sw --gen_id <turn_gen_id> [alias]{Colors.RESET}")
            return cursor, True
        gen_id = tokens[1]
        alias = tokens[2] if len(tokens) > 2 else None
        try:
            target_cursor = cursor.cursor_for_gen_id(gen_id)
        except KeyError:
            print(f"{Colors.ERROR}Turn '{gen_id}' not found in this session.{Colors.RESET}")
            return cursor, True
        except ValueError as err:
            print(f"{Colors.ERROR}{err}{Colors.RESET}")
            return cursor, True
        if not target_cursor or not target_cursor.current_turn:
            print(f"{Colors.ERROR}Turn '{gen_id}' not found in this session.{Colors.RESET}")
            return cursor, True

        # Avoid duplicate if a cursor already points at this gen_id
        existing = scope.find_cursor_by_gen_id(gen_id)
        if existing:
            print(f"{Colors.SYSTEM}Cursor already exists for turn '{gen_id}' (alias '{existing.label}').{Colors.RESET}")
            cursor = scope.set_active_cursor(existing)
            return cursor, True

        # Register/alias the resolved cursor
        try:
            new_cursor = scope.adopt_cursor(target_cursor, alias=alias, make_active=False)
        except Exception as exc:
            print(f"{Colors.ERROR}Failed to register cursor for '{gen_id}': {exc}{Colors.RESET}")
            return cursor, True

        scope.set_active_cursor(new_cursor)
        cursor = new_cursor
        print(f"{Colors.SYSTEM}Added cursor for '{gen_id}' as '{cursor.label}'.{Colors.RESET}")
        return cursor, True

    # Switch by alias or gen_id (auto-rebuild if needed)
    target = tokens[0]
    try:
        cursor = scope.set_active_cursor(target)
        print(f"{Colors.SYSTEM}Active cursor set to '{target}' (turn {cursor.head.gen_id if cursor.head else 'N/A'}).{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.ERROR}Could not activate cursor '{target}': {e}{Colors.RESET}")
    return cursor, True

# --- SessionControl Class ---
class SessionControl:
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, name: str) -> Path:
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
        if not safe_name:
            safe_name = f"invalid_name_{int(time.time())}"
        return self.sessions_dir / f"{safe_name}.json"

    def save(self, session: EngineSession, *, path: Optional[Union[str, Path]] = None):
        session_path = Path(path).expanduser().resolve() if path else self._get_session_path(session.name)
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session.serialize(file_path=session_path)
        print(f"{Colors.SYSTEM}Session '{session.name}' saved to {session_path}{Colors.RESET}")

    def load(self, name: str) -> Optional[EngineSession]:
        # Allow loading from a direct path
        if os.path.exists(name):
            session_path = Path(name)
        else:
            session_path = self._get_session_path(name)

        if not session_path.exists():
            print(f"{Colors.ERROR}Session '{name}' not found at {session_path}{Colors.RESET}")
            return None
        # print(f"Session '{name}' loaded from {session_path}") # Printed by caller after screen clear
        return EngineSession.deserialize(source=session_path)

    def list_sessions(self) -> List[str]:
        return sorted([p.stem for p in self.sessions_dir.glob("*.json")])

    def exists(self, name: str) -> bool:
        return self._get_session_path(name).exists()

def _store_exception_traceback_if_clear(exc: BaseException):
    """Stores the traceback of an exception only if the global store is empty."""
    global LAST_EXCEPTION_TRACEBACK
    if not LAST_EXCEPTION_TRACEBACK:
        LAST_EXCEPTION_TRACEBACK = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(f"\n{Colors.ERROR}An error occurred: {exc}{Colors.RESET}\n{Colors.DIM}Use '/f ex' to see the full traceback.{Colors.RESET}")

# --- API Call Helper ---
async def call_api(tool_name: str, arguments: Optional[Dict[str, Any]] = None):
    """A helper to call the engine API. Returns a dict for most calls and a dict with a stream or error for run-inference."""
    global LAST_EXCEPTION_TRACEBACK
    # Reset traceback before each call to capture the most relevant error.
    LAST_EXCEPTION_TRACEBACK = None
    api_args = arguments or {}
    response_from_api = await handle_call_tool(tool_name, api_args)
    
    # All API calls now return an MP13Response object.
    if isinstance(response_from_api, MP13Response):
        # Handle streaming responses for 'run-inference'
        if response_from_api.stream is not None:
            return {"stream": inference_stream_to_dict_stream(response_from_api.stream)}
        # Handle synchronous responses
        response_dict = response_from_api.model_dump(exclude_none=True)
        if response_dict.get("status") == "error" and response_dict.get("details", {}).get("full_traceback"):
            LAST_EXCEPTION_TRACEBACK = response_dict["details"]["full_traceback"]
            print(f"\n{Colors.ERROR}An error occurred: {response_dict.get('message', 'Engine error')}{Colors.RESET}\n{Colors.DIM}Use '/f ex' to see the full traceback.{Colors.RESET}")
        return response_dict
    # Fallback for unexpected return types
    else:
        return {"status": "error", "message": f"Unexpected API response type: {type(response_from_api).__name__}"}

def _apply_config_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    base = json.loads(json.dumps(DEFAULT_CHAT_CONFIG))
    merged = dict(base)
    merged.update(config)
    merged["engine_params"] = extract_engine_params(merged)
    merged["inference_params"] = _normalize_chat_params(merged.get("inference_params"))
    return merged


def load_config(default_config_path: Path, custom_config_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    global EFFECTIVE_CONFIG_FILE_PATH
    EFFECTIVE_CONFIG_FILE_PATH = custom_config_path or default_config_path
    merged, _, _ = load_merged_config(
        default_config_path=default_config_path,
        custom_config_path=custom_config_path,
    )
    if not merged:
        return None
    resolved, resolver = resolve_config_paths(
        merged,
        config_path=EFFECTIVE_CONFIG_FILE_PATH,
        cwd=Path.cwd(),
    )
    config = resolve_engine_inputs(resolved, resolver)
    config = _apply_config_defaults(config)
    EngineSession.current_config = config
    return config

def save_config(config_data: Dict[str, Any], save_to_path: Path):
    normalized = dict(config_data)
    normalized["inference_params"] = _normalize_chat_params(config_data.get("inference_params"))
    save_json_config(normalized, save_to_path)
    print(f"{Colors.SYSTEM}Configuration saved to {save_to_path}{Colors.RESET}")

def _input_with_default(prompt_text: str, default_value: Any, current_value: Optional[Any] = None) -> str:
    display_value = current_value if current_value is not None else default_value
    user_input = input(f"{prompt_text} [{display_value}]: ").strip()
    return user_input or str(display_value)

def _input_with_default_optional(prompt_text: str, default_value: Optional[Any], current_value: Optional[Any] = None) -> Optional[str]:
    display_value = current_value if current_value is not None else (default_value if default_value is not None else "")
    user_input = input(f"{prompt_text} [{display_value}]: ").strip()
    if not user_input: # User pressed enter
        return str(display_value) if display_value else None # Return current or default, or None if both empty
    return user_input

def prompt_for_config(config_to_update: Optional[Dict[str, Any]] = None, save_to_path: Path = DEFAULT_CONFIG_FILE, *, prompt_for_name: bool = False) -> Dict[str, Any]:
    print(f"{Colors.HEADER}--- MP13 Chat Configuration ---{Colors.RESET}")
    # Initialize with defaults, then override with config_to_update if provided
    defaults = dict(_apply_config_defaults({}))
    defaults.update({
        "category_dirs": DEFAULT_CATEGORY_DIRS,
        "inference_params": _conversation_param_defaults(),
    })
    
    current_values = {**defaults, **(config_to_update or {})}
    current_values["inference_params"] = _normalize_chat_params(current_values.get("inference_params"))

    global EFFECTIVE_CONFIG_FILE_PATH
    if prompt_for_name:
        default_display = str(save_to_path)
        if save_to_path == DEFAULT_CONFIG_FILE:
            default_display = DEFAULT_CONFIG_FILE.name
        config_name = _input_with_default("Config file name (relative names are stored under the default config dir)", default_display, current_values.get("_config_name", default_display))
        if config_name:
            candidate_path = Path(config_name)
            if not candidate_path.is_absolute():
                candidate_path = DEFAULT_CONFIG_DIR / candidate_path
            save_to_path = candidate_path.expanduser().resolve()
            EFFECTIVE_CONFIG_FILE_PATH = save_to_path

    category_dirs = dict(current_values.get("category_dirs") or DEFAULT_CATEGORY_DIRS)
    models_root_dir = _input_with_default("Enter models root directory", category_dirs.get("models_root_dir", ""), category_dirs.get("models_root_dir"))
    adapters_root_dir = _input_with_default("Enter adapters root directory", category_dirs.get("adapters_root_dir", ""), category_dirs.get("adapters_root_dir"))
    sessions_root_dir = _input_with_default("Enter sessions root directory", category_dirs.get("sessions_root_dir", ""), category_dirs.get("sessions_root_dir"))
    data_root_dir = _input_with_default("Enter data root directory", category_dirs.get("data_root_dir", ""), category_dirs.get("data_root_dir"))
    tools_root_dir = _input_with_default("Enter tools root directory", category_dirs.get("tools_root_dir", ""), category_dirs.get("tools_root_dir"))
    logs_root_dir = _input_with_default("Enter logs root directory", category_dirs.get("logs_root_dir", ""), category_dirs.get("logs_root_dir"))
    engine_params = dict(current_values.get("engine_params") or {})
    base_model_path = _input_with_default("Enter base model path (prefix with hf: for remote IDs)", engine_params.get("base_model_path", ""), engine_params.get("base_model_path"))
    base_model_dtype = _input_with_default("Enter base model dtype (auto, bfloat16, float16, float32)", engine_params.get("base_model_dtype", "auto"), engine_params.get("base_model_dtype"))
    attn_implementation = _input_with_default("Enter attention implementation (auto, flash_attention_2, sdpa, eager)", engine_params.get("attn_implementation", "auto"), engine_params.get("attn_implementation"))
    quant_prompt = "Enter quantization method (none, hqq, eetq)"
    quantize_bits_method = (_input_with_default(quant_prompt, engine_params.get("quantize_bits", "none"), engine_params.get("quantize_bits"))).lower()
    default_context_size_str = _input_with_default_optional("Enter default context size (e.g., 4096, or empty for model default)", engine_params.get("default_context_size"), engine_params.get("default_context_size"))
    default_context_size = int(default_context_size_str) if default_context_size_str and default_context_size_str.isdigit() else None
    default_max_new_tokens_str = _input_with_default("Enter default max new tokens for engine", engine_params.get("default_max_new_tokens", 8192), engine_params.get("default_max_new_tokens"))
    default_max_new_tokens = int(default_max_new_tokens_str) if default_max_new_tokens_str.isdigit() else engine_params.get("default_max_new_tokens", 8192)
    default_system_message = _input_with_default_optional("Enter default system message for new sessions (can be empty)", engine_params.get("default_system_message", ""), engine_params.get("default_system_message"))

    config_data: Dict[str, Any] = current_values.copy() # Start with current/default values
    if quantize_bits_method not in ("none", "hqq", "eetq"):
        print(f"{Colors.TOOL_WARNING}Invalid quantization method '{quantize_bits_method}'. Defaulting to 'none'.{Colors.RESET}")
        quantize_bits_method = "none"

    engine_params["quantize_bits"] = quantize_bits_method

    if quantize_bits_method == "hqq":
        engine_params["hqq_bits"] = int(_input_with_default("  HQQ: Bits (2,3,4,8)", engine_params.get("hqq_bits", 4), engine_params.get("hqq_bits")))
        engine_params["hqq_group_size"] = int(_input_with_default("  HQQ: Group size (e.g., 64)", engine_params.get("hqq_group_size", 64), engine_params.get("hqq_group_size")))
        engine_params["hqq_quant_zero"] = (_input_with_default("  HQQ: Quant zero (True/False)", str(engine_params.get("hqq_quant_zero", True)), str(engine_params.get("hqq_quant_zero")))).lower() == 'true'
        engine_params["hqq_quant_scale"] = (_input_with_default("  HQQ: Quant scale (True/False)", str(engine_params.get("hqq_quant_scale", False)), str(engine_params.get("hqq_quant_scale")))).lower() == 'true'
        engine_params["hqq_axis"] = int(_input_with_default("  HQQ: Axis (0 or 1)", engine_params.get("hqq_axis", 1), engine_params.get("hqq_axis")))

    tools_config_path_str = _input_with_default("Enter tools JSON file path (relative uses tools root)", engine_params.get("tools_config_path", "mp13tools.json"), engine_params.get("tools_config_path"))
    tools_config_path = tools_config_path_str
    
    use_cache_str = _input_with_default("Enable KV Caching (True/False)", str(engine_params.get("use_cache", True)), str(engine_params.get("use_cache", "True"))).lower()
    use_cache = use_cache_str == 'true'

    # Default to True for the prompt if not present in current_values
    use_torch_compile_str = _input_with_default("Enable torch.compile for faster inference (True/False)", str(engine_params.get("use_torch_compile", True)), str(engine_params.get("use_torch_compile", "True"))).lower()
    use_torch_compile = use_torch_compile_str == 'true'

    static_kv_cache_str = _input_with_default("Enable static GPU KV cache (True/False)", str(engine_params.get("static_kv_cache", False)), str(engine_params.get("static_kv_cache", "True"))).lower()
    static_kv_cache = static_kv_cache_str == 'true'

    concurrent_generate_str = _input_with_default("Number of concurrent requests for the engine (1 for disabled)", engine_params.get("concurrent_generate", 4), engine_params.get("concurrent_generate"))
    concurrent_generate = int(concurrent_generate_str) if concurrent_generate_str.isdigit() and int(concurrent_generate_str) >= 1 else 1

    engine_params.update({
        "base_model_path": base_model_path,
        "base_model_dtype": base_model_dtype,
        "default_context_size": default_context_size,
        "default_max_new_tokens": default_max_new_tokens,
        "default_system_message": default_system_message,
        "quantize_bits": quantize_bits_method,
        "attn_implementation": attn_implementation,
        "tools_config_path": tools_config_path,
        "use_cache": use_cache,
        "device_map": "auto",
        "trust_remote_code": True,
        "use_torch_compile": use_torch_compile,
        "static_kv_cache": static_kv_cache,
        "concurrent_generate": concurrent_generate
    })
    config_data["engine_params"] = engine_params

    category_dirs.update({
        "models_root_dir": models_root_dir,
        "adapters_root_dir": adapters_root_dir,
        "sessions_root_dir": sessions_root_dir,
        "data_root_dir": data_root_dir,
        "tools_root_dir": tools_root_dir,
        "logs_root_dir": logs_root_dir,
    })
    config_data["category_dirs"] = category_dirs

    config_data["inference_params"] = _normalize_chat_params(config_data.get("inference_params"))
    save_config(config_data, save_to_path)
    resolved_config, _ = resolve_config_paths(
        config_data,
        cwd=Path.cwd(),
        config_path=save_to_path,
    )
    Path(resolved_config["adapters_root_dir"]).mkdir(parents=True, exist_ok=True)
    Path(resolved_config["sessions_save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(resolved_config["tools_config_path"]).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    return config_data


def _interactive_edit_conversation_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    current = _normalize_chat_params(params) if params else _conversation_param_defaults()
    hints = {
        "stream": "Stream responses",
        "cache": "Cache key override",
        "return_prompt": "Return prompt from engine",
        "generation_config_template": "Generation config JSON",
        "max_new_tokens": "Max new tokens (None to unset)",
        "no_tools_parse": "Disable tool parsing",
        "results_as_user_role": "Emit tool results as user role",
        "pack_results_as_one_role": "Pack tool results into one message",
        "advertised_tools": "Tools to advertise to model (*, *i, *c, *e or names)",
        "silent_tools": "Tools enabled but hidden (*, *i, *c, *e or names)",
        "disabled_tools": "Tools disabled (*, *i, *c, *e or names)",
        "auto_retry_truncated": "Auto retry on truncation",
        "suppress_full_response": "Skip full response printing",
        "auto_tool_retry_limit": "Retries for failed tool calls",
        "auto_continue_retry_limit": "Retries for auto-continue flows",
        "global_tools_mode": "Global tools mode (advertised, silent, disabled)",
        "tools_config_path": "Path to the JSON file defining tools",
    }
    print(f"{Colors.HEADER}--- New Conversation Parameters ---{Colors.RESET}")
    print(json.dumps(current, indent=2))
    while True:
        target = input("Param to change (enter to finish): ").strip()
        if not target:
            break
        key = target
        if key not in current:
            print(f"{Colors.ERROR}Unknown param '{key}'.{Colors.RESET}")
            continue
        hint = hints.get(key, "")
        prompt_text = f"Enter value for {key}"
        if hint:
            prompt_text += f" ({hint})"
        existing_val = current.get(key)
        raw_val = input(f"{prompt_text} [{existing_val}]: ").strip()
        if not raw_val:
            continue
        try:
            if key in {"stream", "no_tools_parse", "auto_retry_truncated", "suppress_full_response", "results_as_user_role", "pack_results_as_one_role"}:
                current[key] = raw_val.lower() in {"true", "1", "yes", "y"}
            elif key in {"max_new_tokens"}:
                current[key] = int(raw_val) if raw_val.lower() != "none" else None
            elif key in {"generation_config_template"}:
                current[key] = json.loads(raw_val)
            elif key in {"advertised_tools", "silent_tools", "disabled_tools"}:
                current[key] = [p.strip() for p in raw_val.split(",") if p.strip()]
            elif key in {"auto_tool_retry_limit", "auto_continue_retry_limit"}:
                current[key] = int(raw_val)
            elif key == "global_tools_mode":
                if raw_val.lower() in {"advertised", "silent", "disabled", "a", "s", "d"}:
                    mode_map = {"a": "advertised", "s": "silent", "d": "disabled"}
                    current[key] = mode_map.get(raw_val.lower(), raw_val.lower())
                else:
                    print(f"{Colors.ERROR}Invalid mode. Must be one of advertised, silent, disabled.{Colors.RESET}")
            else:
                current[key] = raw_val
        except Exception as exc:
            print(f"{Colors.ERROR}Could not parse value: {exc}{Colors.RESET}")
    return current

# --- Engine Lifecycle Functions ---
async def initialize_mp13_engine(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Initializes the engine and prints status based on the --dump-init flag. Returns the init response on success."""
    global ENGINE_INITIALIZED_SUCCESSFULLY
    print(f"{Colors.SYSTEM}Initializing MP13 Engine...{Colors.RESET}")
    engine_config_payload = build_engine_init_payload(config)
    
    init_resp = await call_api("initialize-engine", engine_config_payload)
    if init_resp.get("status") != "success":
        error_message = init_resp.get('message', 'Unknown error')
        print(f"{Colors.ERROR}Failed to initialize engine: {error_message}{Colors.RESET}")
        # Print any partial details if available
        if init_resp.get("details"):
            print(f"{Colors.DIM}{json.dumps(init_resp.get('details'), indent=2)}{Colors.RESET}")
        return None

    instance_id = (init_resp.get("data") or {}).get("instance_id")
    if instance_id:
        set_default_resp = await call_api("set-default-engine", {"instance_id": instance_id})
        if set_default_resp.get("status") != "success":
            print(f"{Colors.TOOL_WARNING}Warning: Failed to set default engine to '{instance_id}': {set_default_resp.get('message', 'Unknown error')}{Colors.RESET}")
    else:
        print(f"{Colors.TOOL_WARNING}Warning: Engine init response missing instance_id; default engine not set explicitly.{Colors.RESET}")

    engine_data = init_resp.get("data")
    model_path = ""
    effective_dtype = ""
    quant_method = ""

    if engine_data and 'global_config' in engine_data:
        global_config = engine_data['global_config']
        model_path = global_config.get('other_config', {}).get('base_model_name_or_path', '')
        effective_dtype = global_config.get('effective_torch_dtype', '')
        quant_method = global_config.get('effective_quantization_method', '')
        quant_precision = global_config.get('effective_quant_precision', '')

    quant_info = ""
    if quant_method and quant_method != "none":
        quant_precision_suffix = f"/{quant_precision}" if quant_precision else ""
        quant_info = f" {quant_method}{quant_precision_suffix}"
    
    path_info = f": {model_path}" if model_path else ""

    print(f"{Colors.SYSTEM}Engine initialized successfully{path_info} [{effective_dtype}{quant_info}]{Colors.RESET}")
    ENGINE_INITIALIZED_SUCCESSFULLY = True # Set flag on success

    # --- Store the base model name for UI display ---

    # --- Print the detailed initialization report based on the --dump-init flag ---
    init_details = init_resp.get("data", {})
    if init_details:
        if warnings := init_details.get("warnings"): # type: ignore
            if _get_console_log_level() > logging.WARNING:
                print(f"{Colors.BRIGHT_YELLOW}Initialization Warnings:{Colors.RESET}")
                for warning in warnings:
                    print(f"  - {warning}")
        
        # Check the global flag set by argparse
        if DUMP_INIT_ENABLED:
            if patches := init_details.get("applied_patches"):
                print(f"{Colors.SYSTEM}Applied Patches & Checks:{Colors.RESET}")
                for patch in patches:
                    print(f"  - {patch}")
            if effective_config := init_details.get("global_config"): # data field is the init_report
                print(f"{Colors.HEADER}--- Effective Engine Configuration ---{Colors.RESET}")
                print(f"{Colors.DIM}{json.dumps(effective_config, indent=2, default=str)}{Colors.RESET}")

    mode_set_resp = await call_api("check-set-mode", {"mode": EngineMode.INFERENCE.value, "force": False}) # type: ignore
    if not (mode_set_resp.get("status") == "success" and mode_set_resp.get("data", {}).get("effective_mode") == EngineMode.INFERENCE.value):
        print(f"{Colors.ERROR}Failed to set engine to INFERENCE mode: {mode_set_resp.get('message')}{Colors.RESET}")
    else:
        print(f"{Colors.SYSTEM}Engine mode confirmed as INFERENCE.{Colors.RESET}")
    return init_resp

async def shutdown_mp13_engine():
    # --- NEW: Check if engine was ever initialized before attempting shutdown ---
    if not ENGINE_INITIALIZED_SUCCESSFULLY:
        return
    print(f"\n{Colors.SYSTEM}Shutting down MP13 Engine...{Colors.RESET}")
    resp = await call_api("shutdown-engine", {"shutdown_all": True})
    if resp.get("status") == "success":
        print(f"{Colors.SYSTEM}Engine shut down successfully.{Colors.RESET}")
    else:
        print(f"{Colors.ERROR}Error shutting down engine: {resp.get('message', 'Unknown error')}{Colors.RESET}")

# --- Adapter and Engine Helper Functions ---
def _resolve_adapter_root_path(adapter_path: Path, adapters_root: Path) -> Optional[Path]:
    """Best-effort resolve for an adapter root by walking up to metadata.json."""
    if adapter_path.is_file():
        adapter_path = adapter_path.parent
    current = adapter_path
    while True:
        try:
            if (current / "metadata.json").is_file():
                return current
        except Exception:
            pass
        if current == adapters_root or current.parent == current:
            break
        current = current.parent
    return None

def _normalize_path_key(path_obj: Path) -> str:
    try:
        return str(path_obj.resolve())
    except Exception:
        return str(path_obj)

def _filter_adapter_entries(
    adapters: List[Dict[str, Any]],
    adapters_root: Path,
    *,
    include_checkpoints: bool,
) -> List[Dict[str, Any]]:
    if include_checkpoints:
        filtered: List[Dict[str, Any]] = []
        seen_foreign: Set[str] = set()
        for info in adapters:
            is_foreign = bool(info.get("is_foreign"))
            path = Path(info.get("path", ""))
            root_hint = info.get("root_path")
            root_hint_path = Path(root_hint) if root_hint else None
            if is_foreign:
                root = root_hint_path or _resolve_adapter_root_path(path, adapters_root) or path
                key = _normalize_path_key(root)
                if key in seen_foreign:
                    continue
                seen_foreign.add(key)
                info = dict(info)
                info["path"] = str(root)
                info["root_path"] = str(root)
                filtered.append(info)
            else:
                root = root_hint_path or _resolve_adapter_root_path(path, adapters_root)
                if root:
                    info = dict(info)
                    info["root_path"] = str(root)
                filtered.append(info)
        return filtered

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for info in adapters:
        path = Path(info.get("path", ""))
        root_hint = info.get("root_path")
        root_hint_path = Path(root_hint) if root_hint else None
        root = root_hint_path or _resolve_adapter_root_path(path, adapters_root)
        key = _normalize_path_key(root or path)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(info)

    result: List[Dict[str, Any]] = []
    for key in order:
        entries = grouped[key]
        root: Optional[Path] = None
        for entry in entries:
            entry_root_hint = entry.get("root_path")
            entry_root_hint_path = Path(entry_root_hint) if entry_root_hint else None
            root = entry_root_hint_path or _resolve_adapter_root_path(Path(entry.get("path", "")), adapters_root)
            if root:
                break
        chosen = None
        if root:
            root_key = _normalize_path_key(root)
            for entry in entries:
                entry_path = Path(entry.get("root_path") or entry.get("path", ""))
                if _normalize_path_key(entry_path) == root_key:
                    chosen = entry
                    break
        if chosen is None:
            chosen = entries[0]
        chosen = dict(chosen)
        if root:
            chosen["root_path"] = str(root)
            chosen["path"] = str(root)
        result.append(chosen)
    return result

async def list_adapters_from_engine(*, include_incompatible: bool = False, include_checkpoints: bool = False):
    global LAST_ENUMERATED_ADAPTERS
    LAST_ENUMERATED_ADAPTERS.clear()
    if not current_config or "adapters_root_dir" not in current_config:
        print("Configuration not loaded. Cannot determine adapters root directory.")
        return
    adapters_root = Path(current_config["adapters_root_dir"]).expanduser().resolve()
    payload = {"root_folder": str(adapters_root), "include_incompatible": include_incompatible}
    resp = await call_api("list-all-adapters", payload)
    if resp.get("status") != "success":
        print(f"Failed to get adapter list from engine: {resp.get('message')}")
        return
    data = resp.get("data") or {}
    adapters = data.get("adapters", [])
    adapters = _filter_adapter_entries(adapters, adapters_root, include_checkpoints=include_checkpoints)
    for adapter_info in adapters:
        name = adapter_info.get("name") or adapter_info.get("alias") or "<unknown>"
        path = Path(adapter_info.get("path", ""))
        rel_path = None
        try:
            rel_path = str(path.resolve().relative_to(adapters_root))
        except Exception:
            rel_path = str(path)
        LAST_ENUMERATED_ADAPTERS.append({"name": name, "info": adapter_info, "rel_path": rel_path, "abs_path": str(path)})

def _print_loaded_adapters(cursor: Optional[ChatCursor], *, include_header: bool = True) -> None:
    """Print a unified adapter list with __base__ at index 0 and active markers."""
    effective_adapters: List[str] = []
    try:
        if cursor:
            effective_adapters = cursor.get_effective_adapters()
    except Exception:
        effective_adapters = []
    if not effective_adapters:
        effective_adapters = ["__base__"]

    if include_header:
        print("Loaded adapters in engine (0 = __base__, use numbers in /a commands):")

    def _active_marker(name: str) -> str:
        return f" {Colors.BRIGHT_CYAN}*(active)*{Colors.RESET}" if name in effective_adapters else ""

    print(f"  0. __base__{_active_marker('__base__')}")
    if LAST_ENUMERATED_ADAPTERS:
        header = f"{'Idx':>4} {'Name':<30} {'Prec':<8} {'Info':<17} Path"
        print(header)
    for idx, adapter_data in enumerate(LAST_ENUMERATED_ADAPTERS, start=1):
        name = adapter_data.get("name", "<unknown>")
        adapter_details = adapter_data.get("info", {}) or {}
        marker = _active_marker(name)
        quant_raw = (
            adapter_details.get("base_model_quant")
            or (adapter_details.get("metadata") or {}).get("base_model_effective_dtype_at_init")
            or (current_config.get("effective_torch_dtype") if current_config else None)
            or "N/A"
        )
        quant = str(quant_raw)
        meta_safe = adapter_details.get("metadata") or {}
        if quant.lower().startswith("hqq") and meta_safe:
            bits = meta_safe.get("precision_info", {}).get("base_model_quantization_config", {}).get("hqq_bits")
            if bits and not any(ch.isdigit() for ch in quant):
                quant = f"HQQ-i{bits}"
        flag = "non-compat"
        if adapter_details.get("is_loaded"):
            flag = "*(loaded)*"
        elif adapter_details.get("is_new"):
            flag = "new"
        elif adapter_details.get("is_foreign"):
            flag = "foreign"
        elif adapter_details.get("is_compatible"):
            path = Path(adapter_details.get("path", ""))
            try:
                mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
                flag = mtime.strftime("%Y-%m-%d %H:%M")
            except Exception:
                flag = "last"
        rel_path = adapter_data.get("rel_path") or adapter_data.get("abs_path") or ""
        print(f"{idx:>4} {name:<30} {quant:<8} {flag:<15} {rel_path}{marker}")

def _normalize_adapter_name_list(names: Optional[Union[str, List[str]]]) -> List[str]:
    if names is None:
        return []
    if isinstance(names, str):
        raw = [names]
    else:
        raw = list(names)
    normalized: List[str] = []
    seen: set[str] = set()
    for item in raw:
        trimmed = (item or "").strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        normalized.append(trimmed)
    return normalized

def _parse_adapter_list_flags(arg_text: str) -> Tuple[str, bool, bool]:
    include_all = False
    include_checkpoints = False
    if not arg_text:
        return "", include_all, include_checkpoints
    try:
        tokens = shlex.split(arg_text)
    except Exception:
        return arg_text.strip(), include_all, include_checkpoints
    include_all = any(tok in {"--all", "-a"} for tok in tokens)
    include_checkpoints = any(tok in {"--checkpoints", "-c"} for tok in tokens)
    cleaned = " ".join(tok for tok in tokens if tok not in {"--all", "-a", "--checkpoints", "-c"})
    return cleaned.strip(), include_all, include_checkpoints

def _parse_pop_target_options(arg_text: str) -> Tuple[Optional[str], bool]:
    """
    Parses shared pop options: pop [--cmd] [gen_id|anchor|cmd_id].
    Returns (identifier, force_cmd_flag).
    """
    stack_id: Optional[str] = None
    force_cmd: bool = False
    if not arg_text:
        return stack_id, force_cmd
    tokens = shlex.split(arg_text)
    # Syntax: [--cmd] <id>
    if tokens and tokens[0] in {"--cmd", "-c"}:
        force_cmd = True
        tokens = tokens[1:]
    if tokens:
        stack_id = tokens[0]
    return stack_id, force_cmd

def apply_adapter_operation(
    cursor: ChatCursor,
    operation: str,
    adapter_names: Optional[List[str]],
    user_command: str,
    *,
    stack_id: Optional[str] = None,
) -> ChatCursor:
    """
    Applies a logical adapter stack operation to the session tree and prints the resulting state.
    """
    adapter_payload: Optional[List[str]] = None
    if operation in {"set", "add"}:
        normalized = _normalize_adapter_name_list(adapter_names)
        if operation == "set" and not normalized:
            normalized = ["__base__"]
        adapter_payload = normalized
    cursor.apply_adapter_operation(
        operation,
        adapter_payload,
        command_text=user_command,
        stack_id=stack_id,
    )
    effective_adapters = cursor.get_effective_adapters()
    adapters_display = ", ".join(effective_adapters) if effective_adapters else "__base__"
    if operation == "add":
        print(f"{Colors.SYSTEM}Adapter(s) added. Active set: {adapters_display}{Colors.RESET}")
    elif operation == "set":
        print(f"{Colors.SYSTEM}Active adapter(s) set to: {adapters_display}{Colors.RESET}")
    else:
        print(f"{Colors.SYSTEM}Adapter stack pop recorded. Active set: {adapters_display}{Colors.RESET}")
    return cursor

def _collect_adapter_scope_entries(cursor: ChatCursor) -> List[Tuple[Optional[str], List[str]]]:
    """Return effective adapter stack entries with stack_ids."""
    if not cursor or not cursor.current_turn:
        return []
    session = cursor.session
    path: List[Turn] = session.get_active_path_for_llm(cursor.current_turn)
    if not path:
        return []

    all_ops: List[Tuple[Turn, Command]] = []
    for turn in path:
        for cmd in getattr(turn, "cmd", []) or []:
            if cmd.cmd_type == Command.STATE_CHANGE and cmd.data.get("change") == "adapters_command":
                all_ops.append((turn, cmd))

    filtered_ops: List[Tuple[Turn, Command]] = []
    for _, cmd in all_ops:
        op_type = (cmd.data.get("op") or "set").lower()
        if op_type != "pop":
            filtered_ops.append((cursor.current_turn, cmd))
            continue

        target_id = cmd.data.get("stack_id")
        if not target_id:
            for idx in range(len(filtered_ops) - 1, -1, -1):
                if (filtered_ops[idx][1].data.get("op") or "set").lower() in {"set", "add"}:
                    filtered_ops.pop(idx)
                    break
            continue

        removed = False
        for idx, (_, candidate_cmd) in enumerate(filtered_ops):
            if candidate_cmd.data.get("stack_id") == target_id and candidate_cmd.data.get("change") == "adapters_command":
                filtered_ops.pop(idx)
                removed = True
                break
        if removed:
            continue

    entries: List[Tuple[Optional[str], List[str]]] = []
    for _, cmd in filtered_ops:
        op = (cmd.data.get("op") or "set").lower()
        adapters = cmd.data.get("adapters") if "adapters" in cmd.data else cmd.data.get("value")
        if isinstance(adapters, str):
            adapter_list = [adapters]
        else:
            adapter_list = list(adapters or [])
        stack_id = cmd.data.get("stack_id")
        if op == "add":
            entries.append((stack_id, adapter_list))
        elif op == "set":
            entries = [(stack_id, adapter_list)] if adapter_list else []
        elif op == "pop":
            if entries:
                entries.pop()
        elif op == "reset":
            entries = []
    return entries

def _print_adapter_scope_summary(cursor: ChatCursor, entries: List[Tuple[Optional[str], List[str]]]) -> None:
    if entries:
        print(f"{Colors.SYSTEM}Adapter stack (oldest -> newest):{Colors.RESET}")
        for idx, (stack_id, adapters) in enumerate(entries, start=1):
            label = f"{stack_id}: " if stack_id else ""
            adapters_display = ", ".join(adapters) if adapters else "__base__"
            print(f"  {idx}. {label}{adapters_display}")
    else:
        print(f"{Colors.SYSTEM}No active adapter stack. Using __base__.{Colors.RESET}")
    effective_adapters = cursor.get_effective_adapters()
    adapters_display = ", ".join(effective_adapters) if effective_adapters else "__base__"
    print(f"{Colors.SYSTEM}Effective adapters:{Colors.RESET} {adapters_display}")

async def unload_adapter_from_engine(cursor: ChatCursor, adapter_name_to_unload: str) -> ChatCursor:
    resp = await call_api("unload-adapter", {"adapter_name": adapter_name_to_unload})
    cursor.save_api_command("unload-adapter", {"adapter_name": adapter_name_to_unload}, command_text=f"/a u {adapter_name_to_unload}", response=resp)
    if resp.get("status") == "success":
        print(f"{Colors.SYSTEM}Adapter '{adapter_name_to_unload}' unloaded successfully from engine.{Colors.RESET}")
        effective_adapters = cursor.get_effective_adapters()
        if adapter_name_to_unload in effective_adapters:
            new_active = [name for name in effective_adapters if name != adapter_name_to_unload]
            if not new_active:
                new_active = ["__base__"]
            cursor.apply_adapter_operation("set", new_active, command_text=f"/a u {adapter_name_to_unload} (auto)")
            print(f"{Colors.SYSTEM}Active adapter list updated after unload. Now: {', '.join(new_active)}{Colors.RESET}")
    else:
        print(f"{Colors.ERROR}Failed to unload adapter '{adapter_name_to_unload}': {resp.get('message')}{Colors.RESET}")
    return cursor

async def load_adapter_to_engine(path_or_name: str, cursor: ChatCursor, *, include_incompatible: bool = False):
    global current_config
    if not current_config:
        print("Configuration not loaded. Cannot determine adapters root directory.")
        return

    adapters_root = Path(current_config["adapters_root_dir"])
    resolved_absolute: Optional[Path] = None
    suggested_logical_name: Optional[str] = None

    # Allow numeric selection from last enumeration (which may include incompatible if requested).
    resolved_entry = None
    try:
        if path_or_name.startswith("#"):
            idx = int(path_or_name[1:])
        else:
            idx = int(path_or_name)
        if idx > 0 and (idx - 1) < len(LAST_ENUMERATED_ADAPTERS):
            resolved_entry = LAST_ENUMERATED_ADAPTERS[idx - 1]
    except Exception:
        resolved_entry = None

    if resolved_entry:
        resolved_absolute = Path(resolved_entry["abs_path"]).expanduser().resolve()
        suggested_logical_name = resolved_entry["info"].get("name") or resolved_entry.get("name") or resolved_absolute.name
    else:
        input_path_obj = Path(path_or_name)
        resolved_absolute = (adapters_root / input_path_obj).expanduser().resolve() if not input_path_obj.is_absolute() else input_path_obj.expanduser().resolve()
        suggested_logical_name = resolved_absolute.name
    
    adapter_root_for_replay = resolved_absolute
    if adapter_root_for_replay.is_file():
        adapter_root_for_replay = adapter_root_for_replay.parent
    
    if not suggested_logical_name or suggested_logical_name == "adapter_config.json":
        suggested_logical_name = adapter_root_for_replay.name

    replay_adapter_subpath = None
    try:
        replay_adapter_subpath = str(adapter_root_for_replay.relative_to(adapters_root))
    except Exception:
        replay_adapter_subpath = None

    command_path_for_log = replay_adapter_subpath or str(resolved_absolute)

    print(f"{Colors.SYSTEM}Attempting to load adapter '{suggested_logical_name}' from path: {resolved_absolute}{Colors.RESET}")
    if not resolved_absolute.exists():
        print(f"{Colors.ERROR}Error: Resolved adapter path does not exist: {resolved_absolute}{Colors.RESET}")
        return

    api_payload = {
        "adapter_name": suggested_logical_name,
        "adapter_path": str(resolved_absolute),
        "if_exists": IfExistsEnum.IGNORE,
    }
    if include_incompatible:
        api_payload["include_incompatible"] = True
    replay_payload = {
        "adapter_name": suggested_logical_name,
        "adapter_path": str(resolved_absolute),
        "replay_adapter_subpath": replay_adapter_subpath,
        "replay_adapters_root": str(adapters_root),
        "replay_loaded_via_checkpoint": bool(resolved_absolute.is_file()),
        "if_exists": IfExistsEnum.IGNORE,
    }
    resp = await call_api("load-adapter", api_payload)
    cursor.save_api_command("load-adapter", replay_payload, command_text=f"/a l {command_path_for_log}", response=resp)

    if resp.get("status") == "success":
        details = resp.get("data", {})
        loaded_adapter_name_from_engine = details.get("adapter_name") 
        if loaded_adapter_name_from_engine: # type: ignore
            print(f"{Colors.SYSTEM}Engine reported adapter '{loaded_adapter_name_from_engine}' loaded successfully.{Colors.RESET}")
        else:
            print(f"{Colors.SYSTEM}Adapter loaded (path sent: {resolved_absolute}), but name couldn't be confirmed. Check /a e.{Colors.RESET}")
    else:
        print(f"{Colors.ERROR}Failed to load adapter: {resp.get('message')}{Colors.RESET}")
        if resp.get("details"): print(f"  Details: {json.dumps(resp.get('details'), indent=2)}")
    return cursor

async def query_adapters_from_engine(cursor: ChatCursor, adapter_name_or_num: Optional[str] = None) -> ChatCursor:
    """Queries and prints details for one or all adapters from the engine. This is a read-only operation."""
    if adapter_name_or_num:
        target_name = ""
        try:
            idx = int(adapter_name_or_num)
            if idx == 0:
                target_name = "__base__"
            elif idx > 0 and (idx - 1) < len(LAST_ENUMERATED_ADAPTERS):
                target_name = LAST_ENUMERATED_ADAPTERS[idx - 1]["name"]
            else:
                target_name = ""
        except (ValueError, IndexError):
            target_name = adapter_name_or_num
        if not target_name:
            print(f"{Colors.ERROR}Could not resolve adapter from '{adapter_name_or_num}'.{Colors.RESET}")
            return cursor

        resp = await call_api("get-adapter-details", {"adapter_name": target_name})
        cursor.save_api_command("get-adapter-details", {"adapter_name": target_name}, command_text=f"/a q {adapter_name_or_num}", response=resp)
        if resp.get("status") != "success":
            print(f"{Colors.ERROR}Failed to get details for adapter '{target_name}': {resp.get('message')}{Colors.RESET}")
            return cursor
        
        adapter_details_dict = resp.get("data", {})
        print(f"\n{Colors.HEADER}--- Adapter Details ---{Colors.RESET}")
        marker = " (active)" if adapter_details_dict.get("is_active") else ""
        print(f"\n{Colors.BOLD}{Colors.WHITE}Adapter: {target_name}{marker}{Colors.RESET}")
        print(f"  {Colors.SYSTEM}Type:{Colors.RESET} {adapter_details_dict.get('type', 'N/A')}")
        print(f"  {Colors.SYSTEM}Root Path:{Colors.RESET} {adapter_details_dict.get('root_path', 'N/A')}")
        if checkpoint_path := adapter_details_dict.get("checkpoint_path"):
            print(f"  {Colors.SYSTEM}Checkpoint:{Colors.RESET} {checkpoint_path}")
        if metadata := adapter_details_dict.get("metadata"):
            try:
                display_pretty = json.dumps(metadata, indent=2)
            except Exception:
                display_pretty = str(metadata)
            print(f"  {Colors.SYSTEM}Metadata:{Colors.RESET} {display_pretty}")
        config_dump = adapter_details_dict.get('config') or adapter_details_dict.get('peft_config_on_model') # Support old and new key
        if config_dump:
            try:
                config_pretty = json.dumps(json.loads(config_dump) if isinstance(config_dump, str) else config_dump, indent=2)
            except Exception:
                config_pretty = str(config_dump) # Fallback
            print(f"  {Colors.SYSTEM}Config:{Colors.RESET} {config_pretty}")
        else:
            print(f"  {Colors.SYSTEM}Config:{Colors.RESET} Not available")
        return cursor
    else:
        # Use engine's loaded-adapters API (no disk scan) and display columns similar to /a e.
        resp = await call_api("get-loaded-adapters")
        cursor.save_api_command("get-loaded-adapters", {}, command_text="/a q", response=resp)
        if resp.get("status") != "success":
            print(f"{Colors.ERROR}Failed to get adapter list: {resp.get('message')}{Colors.RESET}")
            return cursor

        all_adapters = resp.get("data", {}).get("adapters", [])
        if not all_adapters:
            print("No adapters currently loaded in the engine.")
            return cursor

        print(f"\n{Colors.HEADER}--- All Loaded Adapters ---{Colors.RESET}")
        header = f"{'Idx':>3} {'Name':<30} {'Prec':<8} {'Info':<15} Path"
        print(header)
        for idx, adapter_info in enumerate(all_adapters, start=1):
            name = adapter_info.get("name") or "<unknown>"
            quant = adapter_info.get("base_model_quant") or (adapter_info.get("metadata") or {}).get("base_model_effective_dtype_at_init") or "N/A"
            flag = "*(active)*" if adapter_info.get("is_active") else "*(loaded)*"
            rel_path = adapter_info.get("root_path") or adapter_info.get("checkpoint_path") or ""
            print(f"{idx:>4} {name:<30} {str(quant):<8} {flag:<15} {rel_path}")
        print(f"{Colors.HEADER}---{Colors.RESET}")
        return cursor

def list_local_adapters_from_disk():
    global LAST_ENUMERATED_LOCAL_ADAPTERS, current_config
    LAST_ENUMERATED_LOCAL_ADAPTERS.clear()
    if not current_config or "adapters_root_dir" not in current_config: # type: ignore
        print(f"{Colors.ERROR}Adapters root directory not configured.{Colors.RESET}"); return
    
    adapters_root = Path(current_config["adapters_root_dir"]) # type: ignore # No print here
    if not adapters_root.is_dir():
        # print(f"Adapters root directory '{adapters_root}' not found."); # Suppressed, caller handles
        return

    # print(f"Available adapters in '{adapters_root}' (use '/a l @<num_here>' or '/a l <name>'):") # Suppressed
    # found_adapters = False; idx = 0 # No longer needed here
    for item in sorted(adapters_root.iterdir()):
        if item.is_dir():
            # idx += 1 # No longer needed here
            # print(f"  {idx}. {item.name}") # Suppressed
            LAST_ENUMERATED_LOCAL_ADAPTERS.append(item.name)
            # found_adapters = True # No longer needed here
    # if not found_adapters: print("  No adapter directories found.") # Suppressed

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_current_session_compact(
    cursor: ChatCursor,
    clear_first: bool = True,
    *,
    start_at_turn: Optional[Turn] = None,
    depth: Optional[int] = None,
    branch_path: Optional[List[Turn]] = None,
):
    """
    Prints a compact, linear history of the active conversation branch.
    This is a dedicated function for the '/s hs' command.
    """
    session = cursor.session
    chat_session = cursor.chat_session
    if clear_first:
        clear_screen()
    print(f"{Colors.HEADER}--- {_conversation_label(session, chat_session)} ({session.name}) ---{Colors.RESET}")
    conv_idx = _conversation_index(session, chat_session)
    if conv_idx is not None:
        print(f"{Colors.SYSTEM}Conversation Index:{Colors.RESET} {conv_idx + 1}")

    target_turn = cursor.current_turn or start_at_turn
    sys_msg = cursor.effective_system_message(target_turn)
    default_sys = None
    if cursor.chat_session and isinstance(getattr(cursor.chat_session, "initial_params", None), dict):
        default_sys = cursor.chat_session.initial_params.get("system_message")
    if default_sys is None:
        default_sys = current_config.get("default_system_message", "") if current_config else None
    sys_msg_display = _format_system_message_display(sys_msg, default_value=default_sys)
    print(f"{Colors.SYSTEM}System Message:{Colors.RESET} {Colors.SYSTEM}{sys_msg_display}{Colors.RESET}")

    effective_adapters = cursor.get_effective_adapters(target_turn)
    active_adapters_display = ", ".join(effective_adapters) if effective_adapters else "__base__"
    print(f"{Colors.SYSTEM}Active Adapter(s):{Colors.RESET} {Colors.SYSTEM}{active_adapters_display}{Colors.RESET}")
    _print_tools_scope_header(cursor)

    print(f"{Colors.SYSTEM}Created:{Colors.RESET} {Colors.SYSTEM}{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session.creation_timestamp))}{Colors.RESET}")

    if branch_path is not None:
        if not branch_path:
            print(f"{Colors.HEADER}--- (No turns in active branch) ---{Colors.RESET}")
            return
        branch_set = set(branch_path)
        working_cursor = cursor.clone_at(target_turn)
        items = list(working_cursor.iter_spine_tree(
            include_forks=True,
            limit_to_active_branch=True,
            detours_first=False))
        label_width = 18
        id_width = 0
        for it in items:
            node = it.cursor.current_turn
            if not node:
                continue
            if it.relation == "spine":
                if node not in branch_set:
                    continue
            else:
                parent = getattr(node, "parent", None)
                if parent not in branch_set:
                    continue
            id_width = max(id_width, len(str(getattr(node, "gen_id_or_parent", "N/A"))))
        if id_width <= 0:
            id_width = 1
        seen = set()
        for it in items:
            node_cursor = it.cursor
            node = node_cursor.current_turn
            if not node:
                continue
            if it.relation == "spine":
                if node not in branch_set:
                    continue
            else:
                parent = getattr(node, "parent", None)
                if parent not in branch_set:
                    continue
            key = getattr(node, "gen_id", None) or f"obj:{id(node)}"
            is_peek = bool(it.is_peek)
            is_collected = bool(it.collected) or (it.relation == "fork" and node in branch_set)
            if is_peek and is_collected:
                continue
            if key in seen:
                continue
            label = node_cursor.display_id(turn=node, active_cursor=cursor)
            active_mark = f" {Colors.BRIGHT_CYAN}*(active)*{Colors.RESET}" if node is target_turn else ""
            summary = node_cursor.summarize_turn(node_cursor)
            is_on_active_path = node in branch_set
            if it.relation == "fork":
                peek_kind = str(getattr(it, "peek_kind", "") or "").lower()
                if "inactive" in peek_kind and "inactive" not in summary:
                    summary = f"{summary} (inactive)"
            if not is_on_active_path and not re.search(r'\binactive\b', summary, re.IGNORECASE):
                summary = f"{summary} (inactive)"
            summary_markers = _summary_context_markers(node, node_cursor)
            is_archived = node.is_archived or re.search(r'\barchived\b', summary)
            if it.relation == "fork":
                line = f"{label:<{label_width}} [{getattr(node,'gen_id','N/A')}] {summary}{active_mark}"
            else:
                line = f"{label:<{label_width}} [{node.gen_id_or_parent}] {summary}{active_mark}"
            for marker in _format_state_change_annotations(node, show_logs=False):
                print(f"    {Colors.SYSTEM}{marker}{Colors.RESET}")
            if is_archived or not is_on_active_path or re.search(r'\binactive\b', summary, re.IGNORECASE):
                print(f"{Colors.DIM}{line}{Colors.RESET}{summary_markers}") # type: ignore
            else:
                print(f"{line}{summary_markers}")
            seen.add(key)
        print(f"{Colors.HEADER}---{Colors.RESET}")
        return

    # Pre-calculate items to determine column widths for alignment
    items = list(cursor.iter_spine_tree(
        include_forks=True,
        limit_to_active_branch=True,
        detours_first=False))
    if not items:
        print(f"{Colors.HEADER}--- (No turns in active branch) ---{Colors.RESET}")
        return

    active_set = set(cursor.active_path_for_llm())
    seen = set()

    label_width = 18

    for it in items:
        node_cursor = it.cursor
        node = node_cursor.current_turn
        if not _within_history_scope(node, start_at_turn, depth):
            continue
        context_cursor = it.context_cursor or cursor
        key = getattr(node, "gen_id", None) or f"obj:{id(node)}"
        # Suppress duplicate “peek + collected” lines: if iterator flags exist, use them;
        # otherwise, treat (relation=='fork' and node in active_set) as collected peek.
        is_peek = bool(it.is_peek)
        is_collected = bool(it.collected) or (it.relation == "fork" and node in active_set)
        if is_peek and is_collected:
            continue
        label = node_cursor.display_id(turn=node, active_cursor=context_cursor)
        active_mark = f" {Colors.BRIGHT_CYAN}*(active)*{Colors.RESET}" if node is target_turn else ""

        summary = context_cursor.summarize_turn(node_cursor)
        if is_peek:
            peek_kind = str(getattr(it, "peek_kind", "") or "").lower()
            if "inactive" in peek_kind and "inactive" not in summary:
                summary = f"{summary} (inactive)"
        summary_markers = _summary_context_markers(node, node_cursor)
        for marker in _format_state_change_annotations(node, show_logs=False):
            print(f"    {Colors.SYSTEM}{marker}{Colors.RESET}")
        if is_peek:
            is_archived = node.is_archived or re.search(r'\barchived\b', summary)
            line = f"{label:<{label_width}} [{node.gen_id_or_parent}] {summary}{active_mark}"
            # Dim archived or inactive peeks
            if is_archived or re.search(r'\binactive\b', summary):
                print(f"{Colors.DIM}{line}{Colors.RESET}{summary_markers}") # type: ignore
            else:
                print(f"{line}{summary_markers}")
        else:
            is_archived = node.is_archived or re.search(r'\barchived\b', summary)
            line = f"{label:<{label_width}} [{node.gen_id_or_parent}] {summary}{active_mark}"
            # Dim inactive spine items by checking for the 'inactive' flag word.
            if is_archived or re.search(r'\binactive\b', summary):
                print(f"{Colors.DIM}{line}{Colors.RESET}{summary_markers}") # type: ignore
            else:
                print(f"{line}{summary_markers}")
        seen.add(key)

    print(f"{Colors.HEADER}---{Colors.RESET}")

def _print_turn_content( # noqa
    cursor: ChatCursor,
    display_id: str,
    prefix: str,
    content_prefix: str,
    is_archived_parent: bool,
    has_forks: bool,
    is_on_active_path: bool,
    active_path_for_llm: Optional[set] = None,
    show_logs: Optional[bool] = False,
    suppress_archived_content: bool = False,
    active_turn: Optional[Turn] = None,
    role_preview_chars: Optional[int] = None,
):
    """
    A dedicated helper to print the formatted content of a single turn.
    This function is non-recursive and is called by both the linear and forked history views.
    """
    turn = cursor.current_turn
    parser_profile = cursor.parser_profile

    def _role_preview_text(value: Any) -> str:
        if role_preview_chars is None:
            return str(value)
        flat = " ".join(str(value).split())
        if len(flat) > role_preview_chars:
            return flat[:role_preview_chars].rstrip() + "..."
        return flat

    def _clip_text(value: Any, max_len: int) -> str:
        flat = " ".join(str(value).split())
        if len(flat) > max_len:
            return flat[: max(0, max_len - 3)].rstrip() + "..."
        return flat

    def _format_tool_args(args: Any, max_len: int = 60) -> str:
        if args is None:
            return ""
        if isinstance(args, dict):
            visible_args = {k: v for k, v in args.items() if not k.startswith('_') and k != 'tool_args_issue'}
            parts = [f"{k}={_clip_text(v, 24)}" for k, v in visible_args.items()]
            args_text = " ".join([p for p in parts if p])
        elif isinstance(args, (list, tuple)):
            parts = [_clip_text(v, 24) for v in args]
            args_text = " ".join([p for p in parts if p])
        else:
            args_text = str(args)
        return _clip_text(args_text, max_len)

    def _tool_call_bracket(call: ToolCall) -> str:
        args_text = _format_tool_args(call.arguments, max_len=60)
        if args_text:
            return f"[{call.name} {args_text}]"
        return f"[{call.name}]"

    def _iter_tool_calls_from_payload(payload: List[Any]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for item in payload:
            if isinstance(item, ToolCallBlock):
                calls.extend([call for call in item.calls if isinstance(call, ToolCall)])
            elif isinstance(item, ToolCall):
                calls.append(item)
            elif isinstance(item, dict) and item.get("name"):
                calls.append(
                    ToolCall(
                        name=item.get("name", ""),
                        arguments=item.get("arguments") or {},
                        result=item.get("result"),
                        error=item.get("error"),
                    )
                )
        return calls

    # NOTE: The chat UI still inspects the underlying Turn for command/state
    # payloads because EngineSession stores adapters, tool blocks, and metrics
    # on the Turn itself. Cursor helpers only steer navigation/labels.

    is_current_turn_archived = cursor.is_archived() or is_archived_parent

    # --- Logic for dimming or suppressing archived content ---
    if is_current_turn_archived:
        if suppress_archived_content:
            header_text = f"--- Turn {display_id} ({turn.gen_id_or_parent}) (archived) "
            print(f"\n{prefix}{Colors.DIM}{header_text:->55}{Colors.RESET}")
            return
        content_prefix = f"{Colors.DIM}{content_prefix}"

    fork_indicator = f" {Colors.DIM}🔱{Colors.RESET}" if has_forks else ""

    branch_marker = ""
    parent_cursor = cursor.parent_cursor()
    sibling_index = cursor.sibling_index()
    if parent_cursor and sibling_index is not None:
        sibling_number = sibling_index + 1
        parent_type = parent_cursor.turn_type()
        if parent_type == Turn.FORK:
            branch_marker = f" {Colors.DIM}(fork:{sibling_number}){Colors.RESET}"
        elif parent_type == Turn.BATCH:
            branch_marker = f" {Colors.DIM}(prompt:{sibling_number}){Colors.RESET}"
        elif sibling_number > 1:
            try_out_number = sibling_number - 1
            branch_marker = f" {Colors.DIM}(try_out:{try_out_number}){Colors.RESET}"

    off_path_marker = ""
    if active_path_for_llm and turn not in active_path_for_llm and turn.turn_type is None:
        off_path_marker = f" {Colors.DIM}(off_path){Colors.RESET}"

    header_color = Colors.HEADER if is_on_active_path else Colors.DIM
    def _flag(text: str) -> str:
        return f"{Colors.BRIGHT_YELLOW}({text}){Colors.RESET}{header_color}"

    archived_marker = f" {Colors.DIM}(archived){Colors.RESET}" if is_current_turn_archived else ""
    active_marker = f" {Colors.BRIGHT_CYAN}*(active)*{Colors.RESET}" if active_turn and turn is active_turn else ""
    root_context_marker = _flag("root_ctx") if getattr(turn, "root_context", False) else ""
    continued_marker = _flag("continued") if turn.do_continue else ""
    truncated_marker = _flag("truncated") if turn.was_truncated else ""
    canceled_marker = _flag("canceled") if turn.was_canceled else ""
    adapters_list = _extract_turn_adapters(turn)
    adapters_marker = f" (Adapters: {_format_adapter_label(adapters_list)})" if adapters_list is not None else " (Adapters: __base__)"
    context_markers = [_format_context_marker(m) for m in _format_context_markers(turn, cursor)]
    context_marker_text = ""
    if context_markers:
        context_marker_text = " " + " ".join(f"{m}{header_color}" for m in context_markers)

    if turn.turn_type == Turn.CHAT or turn.turn_type is None:
        all_markers = branch_marker + off_path_marker
        status_markers = f"{continued_marker}{truncated_marker}{canceled_marker}{root_context_marker}"
        header_text = f"--- Turn {display_id} ({turn.gen_id_or_parent}){all_markers}{archived_marker}{status_markers}{adapters_marker}{context_marker_text} "
        print(f"\n{prefix}{header_color}{header_text:->55}{active_marker}{Colors.RESET}")

    for marker in _format_state_change_annotations(turn, show_logs=show_logs):
        print(f"{content_prefix}{Colors.SYSTEM}  {marker}{Colors.RESET}")

    # --- Print additional command data (logs, API calls, legacy states) ---
    for cmd in getattr(turn, "cmd", []):
        if cmd.cmd_type == Command.PARAM_CHANGE and cmd.data.get("change") == "system_message":
            continue
        if cmd.cmd_type == Command.STATE_CHANGE and cmd.data.get("change") == "adapters_command":
            continue
        if cmd.cmd_type == Command.COMMAND:
            continue
        cmd_data_str = ""
        if cmd.cmd_type == Command.ADAPTERS_STATE:
            cmd_data_str = f"Adapters State: {cmd.data.get('state_type')} = {cmd.data.get('value')}"
        elif cmd.api_name in ["load-adapter", "unload-adapter", "set-active-adapter"]:
            cmd_data_str = cmd.data.get('command', f"API call: {cmd.api_name}")
        elif show_logs and cmd.cmd_type in [Command.LOG, Command.TURN]:
            if cmd.cmd_type == Command.LOG:
                log_text = cmd.data.get('text') or cmd.data.get('command', '')
                cmd_data_str = log_text
            elif cmd.cmd_type == Command.TURN:
                turn_parent = cmd.parent
                turn_type_str = turn_parent.turn_type if turn_parent and turn_parent.turn_type else "placeholder"
                turn_gen_id_str = f" (gen_id: {turn_parent.gen_id})" if turn_parent and turn_parent.gen_id else ""
                parent_turn_id_str = (
                    cursor.display_id(turn_parent.parent, active_cursor=cursor)
                    if turn_parent and turn_parent.parent
                    else "root"
                )
                cmd_data_str = f"Added '{turn_type_str}' turn{turn_gen_id_str} to '{parent_turn_id_str}'"
            elif cmd.api_name:
                cmd_data_str = f"API Call: {cmd.api_name}"

        if cmd_data_str:
            print(f"{content_prefix}{Colors.DIM}  [{cmd.cmd_type}] {cmd_data_str}{Colors.RESET}")

    # --- Print all user-side messages (any key except 'assistant') ---
    if isinstance(turn, Turn) and turn.turn_type != Turn.FORK:
        for role, payload in (turn.data or {}).items():
            if role in {"assistant", "$RequestParams", "$try_out"}:
                continue

            if role == "user":
                user_message_content = payload.get("content")
                if user_message_content:
                    continued_flag = f" {Colors.DIM}(continued){Colors.RESET}" if turn.do_continue else ""
                    if role_preview_chars is not None:
                        snippet = _role_preview_text(user_message_content)
                        print(f"{content_prefix}{Colors.YOU_HEADER}You{continued_flag}{root_context_marker}:{Colors.RESET} {Colors.YOU_CONTENT}{snippet}{Colors.RESET}")
                    else:
                        print(f"{content_prefix}{Colors.YOU_HEADER}You{continued_flag}{root_context_marker}:{Colors.RESET}")
                        print(f"{content_prefix}{Colors.YOU_CONTENT}{user_message_content}{Colors.RESET}")

            elif role == "tool_results":
                if not payload:
                    continue
                tool_calls = _iter_tool_calls_from_payload(payload)
                if role_preview_chars is not None:
                    if tool_calls:
                        first = tool_calls[0]
                        result_value = first.result if first.result is not None else first.error
                        result_text = "" if result_value is None else str(result_value)
                        snippet = _role_preview_text(result_text)
                        bracket = _tool_call_bracket(first)
                        message = f"{snippet} {bracket}".strip()
                        more_suffix = f" (+{len(tool_calls) - 1} more)" if len(tool_calls) > 1 else ""
                    else:
                        first = payload[0]
                        snippet = _role_preview_text(str(first))
                        message = snippet
                        more_suffix = f" (+{len(payload) - 1} more)" if len(payload) > 1 else ""
                    continued_label = " (continued)" if turn.do_continue else ""
                    print(f"\n{content_prefix}{Colors.TOOL}Tool Results{continued_label}:{Colors.RESET} {message}{more_suffix}")
                    continue
                if turn.do_continue:
                    print(f"\n{content_prefix}{Colors.TOOL}Tool Results (continued):{Colors.RESET}")
                else:
                    print(f"\n{content_prefix}{Colors.TOOL}Tool Results:{Colors.RESET}")

                if tool_calls:
                    for tool_call in tool_calls:
                        result_value = tool_call.result if tool_call.result is not None else tool_call.error
                        result_text = "" if result_value is None else str(result_value)
                        is_error = tool_call.error is not None
                        content_color = Colors.ERROR if is_error else Colors.RESET
                        lines = result_text.splitlines() or [""]
                        bracket = _tool_call_bracket(tool_call)
                        for i, line in enumerate(lines):
                            if i == 0:
                                line_out = f"{line} {bracket}".strip()
                            else:
                                line_out = line
                            print(f"{content_prefix}  {content_color}{line_out}{Colors.RESET}")
                else:
                    for tool_call in payload:
                        content_to_print = str(tool_call)
                        is_error = str(content_to_print).lower().startswith("error")
                        content_color = Colors.ERROR if is_error else Colors.RESET
                        for line in content_to_print.splitlines():
                            print(f"{content_prefix}  {content_color}{line}{Colors.RESET}")

            else:  # Handle any other custom role
                content_to_print = ""
                if isinstance(payload, dict):
                    # For dicts, try to get 'content', otherwise serialize the whole dict.
                    content_to_print = payload.get("content", json.dumps(payload, indent=2))
                elif isinstance(payload, str):
                    content_to_print = payload
                else: # For other types like lists, etc.
                    content_to_print = str(payload)

                if content_to_print:
                    role_display = role.replace("_", " ").title()
                    if role_preview_chars is not None:
                        snippet = _role_preview_text(content_to_print)
                        print(f"\n{content_prefix}{Colors.TOOL}{role_display}:{Colors.RESET} {snippet}")
                        continue
                    if role.startswith("$") and not show_logs:
                        # For /s hf, show only length for special roles
                        content_len = len(str(content_to_print))
                        print(f"\n{content_prefix}{Colors.TOOL}{role_display}: (len: {content_len}){Colors.RESET}")
                    else:
                        # For /s hfl or normal roles, show full content
                        print(f"\n{content_prefix}{Colors.TOOL}{role_display}:{Colors.RESET}")
                        for line in str(content_to_print).splitlines():
                            print(f"{content_prefix}  {line}{Colors.RESET}")

    # --- Print Assistant Message ---
    if isinstance(turn, Turn) and (assistant_msg := turn.data.get("assistant", None)) and turn.turn_type != Turn.FORK:
        msg = assistant_msg
        text_part = msg.get('content', '')
        blocks_part = msg.get('tool_blocks')
        was_truncated = turn.was_truncated
        was_canceled = turn.was_canceled
        is_continuation = turn.do_continue
        llm_label = msg.get("model_name") or _resolve_llm_display_name(cursor, turn)

        has_content_to_print = bool(text_part) or bool(blocks_part)
        if has_content_to_print:
            if role_preview_chars is not None:
                summary_bits: List[str] = []
                if text_part:
                    summary_bits.append(_role_preview_text(text_part))
                if blocks_part:
                    summary_bits.append("[tool calls]")
                summary_text = " ".join(summary_bits) if summary_bits else "<empty>"
                print(f"{content_prefix}{Colors.LLM_HEADER}{llm_label}:{Colors.RESET} {summary_text}")
            else:
                print(f"{content_prefix}{Colors.LLM_HEADER}{llm_label}:{Colors.RESET}")

        if role_preview_chars is None and text_part:
            indented_text = '\n'.join([f"{content_prefix}{Colors.LLM_CONTENT}{line}{Colors.RESET}" for line in text_part.splitlines()])
            print(indented_text)

        if role_preview_chars is None and blocks_part:
            if parser_profile:
                parser = UnifiedToolIO(profile=parser_profile)
                rehydrated_blocks = []
                for block_data in blocks_part:
                    if isinstance(block_data, dict):
                        rehydrated_blocks.append(ToolCallBlock.from_dict(block_data))
                blocks_part = rehydrated_blocks if rehydrated_blocks else blocks_part
                parser.parse_collected_blocks(blocks_part)

            for block in blocks_part:
                if not isinstance(block, ToolCallBlock):
                    print(f"{content_prefix}  {Colors.TOOL_WARNING}(Warning: Expected ToolCallBlock, got {type(block)}. Showing raw data.){Colors.RESET}")
                    print(f"{content_prefix}  {Colors.TOOL_ARGS}{str(block)}{Colors.RESET}")
                    continue

                if parser_profile:
                    if block.calls:
                        serialized_calls: List[str] = ToolsParserHelper.serialize_blocks(
                            profile=parser_profile, blocks=[block], is_result=False
                        )
                    else:
                        serialized_calls = [block.raw_block]
                    for serialized_block in serialized_calls:
                        for line in serialized_block.splitlines():
                            print(f"{content_prefix}  {Colors.TOOL_ARGS}{line}{Colors.RESET}")
                else:
                    print(f"{content_prefix}  {Colors.TOOL_WARNING}(Tool profile not loaded, showing raw block){Colors.RESET}")
                    print(f"{content_prefix}  {Colors.TOOL_ARGS}{getattr(block, 'raw_block', str(block))}{Colors.RESET}")

        if is_continuation:
            print(f"{content_prefix}{Colors.DIM}(Response continued from previous){Colors.RESET}")

    # --- Print Metrics ---
    if isinstance(turn, Turn) and turn.metrics:
        per_item_metrics = _build_metric_parts(turn.metrics)
        aggregate_parts: List[str] = []
        if (total_in := turn.metrics.get("total_input_tokens")) is not None: aggregate_parts.append(f"Total In: {total_in}")
        if (total_out := turn.metrics.get("total_output_tokens")) is not None: aggregate_parts.append(f"Total Out: {total_out}")
        if (total_gen_time := turn.metrics.get("total_generation_duration_sec")) is not None: aggregate_parts.append(f"Total GenTime: {total_gen_time:.1f}s")
        if (overall_tps := turn.metrics.get("overall_tps")) is not None: aggregate_parts.append(f"Overall TPS: {overall_tps:.1f}")
        if (avg_ttft := turn.metrics.get("avg_time_to_first_token_sec")) is not None: aggregate_parts.append(f"Avg Latency: {avg_ttft * 1000:.0f}ms")
        if (prompts := turn.metrics.get("total_prompts_processed")) is not None: aggregate_parts.append(f"Prompts: {prompts}")
        combined_parts = per_item_metrics + aggregate_parts
        if combined_parts:
            print(f"{content_prefix}{Colors.METRICS}  {' | '.join(combined_parts)}{Colors.RESET}")

def print_current_session_info(cursor: ChatCursor, # noqa
                               show_messages: bool = False, show_forks: bool = False,
                               show_logs: bool = False, clear_first: bool = True,
                               role_preview_chars: Optional[int] = None,
                               active_only: bool = False,
                               start_at_turn: Optional[Turn] = None,
                               depth: Optional[int] = None,
                               show_all_turns: bool = False,
                               branch_path: Optional[List[Turn]] = None):
    """Print session header/meta and optionally the turn/messages view.""" # noqa
    session = cursor.session
    chat_session = cursor.chat_session
    header_label = _conversation_label(session, chat_session)
    show_full_tree = (show_all_turns or show_forks) and not active_only

    if show_messages:
        # The `clear_first` flag controls whether the screen is cleared.
        if clear_first:
            clear_screen()
        print(f"{Colors.HEADER}--- {header_label} ({session.name}) ---{Colors.RESET}")
    else: # Incremental update, e.g. after new session
        print(f"\n{Colors.HEADER}--- Switched to {header_label} ({session.name}) ---{Colors.RESET}")

    conv_idx = _conversation_index(session, chat_session)
    if conv_idx is not None:
        print(f"{Colors.SYSTEM}Conversation Index:{Colors.RESET} {conv_idx + 1}")

    target_turn = cursor.current_turn or start_at_turn
    sys_msg = cursor.effective_system_message(target_turn)
    default_sys = None
    if cursor.chat_session and isinstance(getattr(cursor.chat_session, "initial_params", None), dict):
        default_sys = cursor.chat_session.initial_params.get("system_message")
    if default_sys is None:
        default_sys = current_config.get("default_system_message", "") if current_config else None
    sys_msg_display = _format_system_message_display(sys_msg, default_value=default_sys)
    print(f"{Colors.SYSTEM}System Message:{Colors.RESET} {Colors.SYSTEM}{sys_msg_display}{Colors.RESET}")
    
    effective_adapters = cursor.get_effective_adapters(target_turn)
    active_adapters_display = ", ".join(effective_adapters) if effective_adapters else "__base__"
    print(f"{Colors.SYSTEM}Active Adapter(s):{Colors.RESET} {Colors.SYSTEM}{active_adapters_display}{Colors.RESET}")
    _print_tools_scope_header(cursor)

    if not show_messages:
        print(f"{Colors.HEADER}---{Colors.RESET}")
        return

    print(f"{Colors.SYSTEM}Created:{Colors.RESET} {Colors.SYSTEM}{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session.creation_timestamp))}{Colors.RESET}")
    print(f"\n{Colors.HEADER}--- Messages ---{Colors.RESET}")

    if show_full_tree:
        # Shared tree walk for /s h and /s hf; show_all_turns expands the walk beyond the active path.
        active_path_for_llm = set(cursor.active_path_for_llm())
        active_spine = set(cursor.active_path())
        active_turn = target_turn
        if cursor.current_turn:
            for item in cursor.iter_spine_tree(
                include_forks=True,
                limit_to_active_branch=False,
                detours_first=True,
            ):
                turn_cursor = item.cursor
                turn = turn_cursor.current_turn
                if not turn:
                    continue
                if not _within_history_scope(turn, start_at_turn, depth):
                    continue
                on_spine = (item.relation == "spine")
                indent = "" if on_spine else "  " * item.depth
                connector = "" if on_spine else ("└─" if item.is_last_sibling else "├─")
                prefix = f"{indent}{connector}"
                content_prefix = prefix.replace("├─", "│ ").replace("└─", "  ")
                has_children = _history_has_children(turn, start_at_turn, depth)
                _print_turn_content(
                    cursor=turn_cursor,
                    display_id=turn_cursor.display_id(turn=turn, active_cursor=cursor),
                    prefix=prefix,
                    content_prefix=content_prefix,
                    is_archived_parent=turn.is_archived,
                    has_forks=has_children,
                    is_on_active_path=turn in active_spine,
                    active_path_for_llm=active_path_for_llm,
                    show_logs=show_logs,
                    suppress_archived_content=False,
                    active_turn=active_turn,
                    role_preview_chars=role_preview_chars,
                )
    else:
        # --- /s h: stick to iter_spine_tree (spine only), then peek siblings manually ---
        if active_only:
            path_nodes = branch_path or cursor.active_path_for_llm()
            for node in path_nodes:
                if branch_path is None and not _within_history_scope(node, start_at_turn, depth):
                    continue
                node_cursor = cursor.clone_at(node)
                display_id = node_cursor.display_id(turn=node, active_cursor=cursor)
                _print_turn_content(
                    cursor=node_cursor,
                    display_id=display_id,
                    prefix="",
                    content_prefix="",
                    is_archived_parent=node.is_archived,
                    has_forks=False,
                    is_on_active_path=True,
                    active_path_for_llm=set(),
                    suppress_archived_content=True,
                    show_logs=show_logs,
                    active_turn=target_turn,
                    role_preview_chars=role_preview_chars,
                )
            return
        for it in cursor.iter_spine_tree(
                include_forks=True,
                limit_to_active_branch=True,
                detours_first=True):
            node_cursor = it.cursor
            node = node_cursor.current_turn
            if not node:
                continue
            if not _within_history_scope(node, start_at_turn, depth):
                continue
            context_cursor = it.context_cursor or cursor
            if it.relation == "spine":
                display_id = node_cursor.display_id(turn=node, active_cursor=context_cursor)
                _print_turn_content(
                    cursor=node_cursor,
                    display_id=display_id, prefix="", content_prefix="",
                    is_archived_parent=node.is_archived, has_forks=False,
                    is_on_active_path=True, active_path_for_llm=set(), # type: ignore
                    suppress_archived_content=True,
                    show_logs=show_logs,
                    active_turn=target_turn,
                    role_preview_chars=role_preview_chars,
                )
            else:  # 'fork' peek (one-liner)
                if not getattr(it, "collected", False):
                    label = node_cursor.display_id(active_cursor=context_cursor)
                    # The prefix should only contain info not in the suffix.
                    # 'peek_kind' (TryOut/Fork) is contextual and useful. 'main'/'inactive' are in the suffix.
                    kind_label = f"({getattr(it, 'peek_kind', 'try out')})"

                    summary = context_cursor.summarize_turn(node_cursor).replace("\n", " ")
                    summary_markers = _summary_context_markers(node, node_cursor)
                    # Always dim the one-line summaries for forks and try-outs.
                    print(f"  {Colors.DIM}{kind_label} {label} [{getattr(node,'gen_id','N/A')}] {summary}{Colors.RESET}{summary_markers}")
        return

    print(f"{Colors.HEADER}---{Colors.RESET}")

def _print_single_turn(cursor: ChatCursor) -> None:
    """Print a single turn's content without moving the active cursor."""
    turn = cursor.current_turn
    if not turn:
        print(f"{Colors.TOOL_WARNING}No active turn to show.{Colors.RESET}")
        return
    active_path = set(cursor.active_path_for_llm())
    display_id = cursor.display_id(active_cursor=cursor)
    has_forks = bool(turn.turns)
    _print_turn_content(
        cursor=cursor,
        display_id=display_id,
        prefix="",
        content_prefix="",
        is_archived_parent=turn.is_archived,
        has_forks=has_forks,
        is_on_active_path=turn in active_path,
        active_path_for_llm=active_path,
        suppress_archived_content=False,
        show_logs=False,
        active_turn=turn,
        role_preview_chars=None,
    )
    print(f"{Colors.HEADER}---{Colors.RESET}")

def print_session_tree_summary(
    cursor: ChatCursor,
    title: str = "",
    active: Optional[Turn] = None,
    start_at_turn: Optional[Turn] = None,
    depth: Optional[int] = None,
    show_logs: bool = False,
) -> None:
    """Spine-left tree printer leveraging ChatCursor.iter_spine_tree()."""
    target_turn = active or cursor.current_turn
    if not target_turn:
        print(f"--- Threaded / Spine Tree View: {title or getattr(cursor.session, 'name', '')} ---\n(empty)")
        return

    if cursor.context:
        working_cursor = cursor.clone_at(target_turn)
    else:
        context = _active_chat_context()
    if not cursor.context and context:
        try:
            scope = _scope_for_cursor(cursor)
            if scope:
                working_cursor = scope.active_cursor().clone_at(target_turn)
            else:
                working_cursor = context.active_cursor.clone_at(target_turn)
        except Exception:
            working_cursor = cursor
    elif not cursor.context:
        working_cursor = cursor

    session = working_cursor.session
    active_turn = active if active is not None else (start_at_turn or target_turn)
    view_title = title or getattr(session, 'name', '')
    print(f"{Colors.HEADER}--- Threaded / Spine Tree View: {view_title} ---{Colors.RESET}")
    sys_msg = working_cursor.effective_system_message(active_turn)
    default_sys = None
    if working_cursor.chat_session and isinstance(getattr(working_cursor.chat_session, "initial_params", None), dict):
        default_sys = working_cursor.chat_session.initial_params.get("system_message")
    if default_sys is None:
        default_sys = current_config.get("default_system_message", "") if current_config else None
    sys_display = _format_system_message_display(sys_msg, default_value=default_sys)
    print(f"{Colors.SYSTEM}System Message:{Colors.RESET} {Colors.SYSTEM}{sys_display}{Colors.RESET}")
    effective_adapters = working_cursor.get_effective_adapters(active_turn)
    adapters_display = ", ".join(effective_adapters) if effective_adapters else "__base__"
    print(f"{Colors.SYSTEM}Active Adapter(s):{Colors.RESET} {Colors.SYSTEM}{adapters_display}{Colors.RESET}")
    _print_tools_scope_header(working_cursor)

    active_cursor_for_labels = cursor
    if active_turn and cursor.current_turn is not active_turn:
        if cursor.context:
            active_cursor_for_labels = cursor.clone_at(active_turn)
        else:
            context = _active_chat_context()
        if not cursor.context and context:
            try:
                scope = _scope_for_cursor(cursor)
                if scope:
                    active_cursor_for_labels = scope.active_cursor().clone_at(active_turn)
                else:
                    active_cursor_for_labels = context.active_cursor.clone_at(active_turn)
            except Exception:
                pass

    for item in working_cursor.iter_spine_tree(
        include_forks=True,
        limit_to_active_branch=False,
        detours_first=True,
    ):
        turn_cursor = item.cursor
        current_turn = turn_cursor.current_turn
        if not _within_history_scope(current_turn, start_at_turn, depth):
            continue
        context_cursor = item.context_cursor or active_cursor_for_labels
        label = turn_cursor.display_id(active_cursor=context_cursor)
        label = f"{label} [{current_turn.gen_id_or_parent}]"
        meta = context_cursor.summarize_turn(turn_cursor)
        meta_markers = _summary_context_markers(current_turn, turn_cursor)
        active_mark = f" {Colors.BRIGHT_CYAN}*(active)*{Colors.RESET}" if current_turn is active_turn else ""
        is_archived = current_turn.is_archived or re.search(r'\barchived\b', meta)
        connector = "└─" if item.is_last_sibling else "├─"
        indent = ("  " * item.depth) if item.relation == "fork" else ""
        line = f"{indent}{connector}{label} {meta}{active_mark}"
        for marker in _format_state_change_annotations(current_turn, show_logs=show_logs):
            print(f"{indent}    {Colors.SYSTEM}{marker}{Colors.RESET}")
        if is_archived:
            print(f"{Colors.DIM}{line}{Colors.RESET}{meta_markers}")
        else:
            print(f"{line}{meta_markers}")

# --- Chat Loop and Commands ---
def print_help():
    C, H, R = Colors.CYAN, Colors.HEADER, Colors.RESET
    def _p(cmd, desc): print(f"  {H}{cmd:<40}{R} {desc}")
    def _ps(cmd, desc): print(f"    {C}{cmd:<38}{R} {desc}")

    print(f"\n{H}--- MP13 Playground Chat Commands ---{R}")
    print("Type your message and press Enter to chat. Commands start with / and can be abbreviated.")
    print(f"Use {C}<name|num>{R} to select items by name or by number from the last list.")
    print(f"Use a comma-separated list {C}<item1,item2,...>{R} for commands that accept multiple selections.")

    print(f"\n{H}General Commands{R}")
    _p("/help or /?", "Show this help message.")
    _p("/q[uit]", "Quit the chat application.")
    _p("/cls", "Clear screen and wipe the active conversation (keeps loaded adapters).")
    _p("/cg", "Continue generation of a possibly truncated model response.")
    _p("/g[enerate] b[atch|bc|ao]", "Enter batch mode as single (b) or many concurrent (bc) or for adapters override (ao).")
    _p("/cb[close_batches] [gen_id]", "Close a batch hub by gen_id (or most recent) and drop its batch cursors.")
    _p("/-[x]", "Delete last x user turns from the active branch. Default is 1.") # noqa
    _p("/r [gen_id]", "Retry inference for the active turn or a specific gen_id.")
    _p("/try [--list|--find|--resurrect <name|num>]", "Manage try-outs. No args starts a new one.")
    _p("/try [anchor_name]", "Start a new named try-out.")
    _p("/mt [anchor_name] [try_index]", "Controls or prints main_thread flag on current or said branch.")
    _p("/ct [anchor_name] [--m a[ll]|[none]|<idx>]", "Closes all try outs of an anchor.")
    _p("/sw[itch] [alias|gen_id|--gen_id id [alias]|--drop alias]", "List cursors or switch/add/drop tracked cursors.")
    _p("/br[anch] --gen_id <id> [alias]", "Resurrect a branch cursor for a historical turn.")

    print(f"\n{H}System Message{R} {C}(/sm){R} (No args prints current by section.)")
    _ps("s[et] <text|''|<>|<def>>", "Set system message. ''=empty, <>=remove, <def>=default.")
    _ps("a[dd] <text>", "Append a blob to the current system message stack.")
    _ps("pop [--cmd] [pop_id|cmd_id|gen_id|anchor_id]", "Undo latest set/add or target a specific turn/command id.")
    _ps("sc[ope] [gen_id]", "Show effective system message scope for a turn (default active).")

    print(f"\n{H}Adapter Management{R} {C}(/a){R}")
    _ps("e[num] [--all] [--checkpoints]", "Enumerate loaded adapters in the engine (0=__base__, numbers follow listing).")
    _ps("l[oad] [name|path|#num] [--all] [--checkpoints]", "Load an adapter into the engine (global; does not change active adapters).")
    _ps("u[nload] [name|num,...]", "Unload one or more adapters from the engine. No args for interactive.")
    _ps("sc[ope] s[et] [name|num,...]", "Set active adapters on the current branch (0 selects __base__).")
    _ps("sc[ope] a[dd] [name|num,...]", "Add adapter(s) to the active stack on this branch. __base__/0 is not allowed here.")
    _ps("sc[ope] p[op] [--cmd] [pop_id|cmd_id|gen_id|anchor_id]", "Undo latest set/add or target a specific id.")
    _ps("sc[ope] [gen_id]", "Show effective adapter stack for active or specified turn.")
    _ps("q[uery] [name|num]", "Query details of loaded adapter(s). No args for interactive.")

    print(f"\n{H}Session Management{R} {C}(/s){R}")
    _ps("n[ew]/sa[ve]/l[oad]/e[num]", "Session file management (/s new|load will drop the current session).")
    _ps("a[dd] [--title <text>]", "Add a new conversation (based on the current one) and switch to it.")
    _ps("c[onversation] <idx> [set|del|insert]", "Switch, delete, or insert a new conversation at a 1-based index.")
    _ps("c[onversation] <idx> [h|hl|hf|hfl|hs|hfs|hfsl|ch|chl] [gen_id] [--rounds N]", "Run history/log helpers for that conversation (1-based).")
    _ps("re[play] <idx> [--from ...]", "Re-run a branch from another conversation (use '/s replay ?' for syntax).")
    _ps("hs [gen_id] [--rounds N]", "Compact branch summary ending at gen_id; rounds limit upward steps.")
    _ps("h[istory]/hl [gen_id] [--rounds N]", "Active branch view; rounds limits upward steps (l includes logs).")
    _ps("hf/hfl [gen_id] [--rounds N]", "Full tree dump ending at gen_id; rounds caps depth above it.")
    _ps("hfs/hfsl [gen_id|*] [--rounds N]", "Compact summary ending at gen_id; rounds caps depth (ignored for '*').")
    _ps("p[rompt] [gen_id]", "Print LLM prompt path for the active or specified turn.")
    _ps("ch/chl [gen_id] [--rounds N]", "Command history ending at gen_id; rounds caps depth above it.")
    _ps("t[urn] <up|down|next|prev|close|main|sh[ow]|gen_id>", "Navigate relative to the active turn or jump/show a gen_id.")
    _ps("t m[ain_thread] [gen_id] [on|off]", "Control main_thread flag for current on gen_id branch.")
    _ps("t root_ctx [gen_id] [on|off]", "Inspect or toggle the root-context flag for a turn (default=current).")
    _ps("t arch[ive] [gen_id] [on|off]", "Inspect or toggle a turn's archive flag without moving the cursor.")
    _ps("t t[rim] [gen_id]", "Delete a turn (and its branch) for the active or specified gen_id.")

    print(f"\n{H}Tool Management{R} {C}(/t){R} (* modified intrinsic)")
    _ps("e[num]", "List available tools and their active status.")
    _ps("h[ide]/sh[ow] <name|num|*|*i|*c|*e,...>", "Hide or reveal tools (*=all, *i intrinsic, *c callable, *e external).")
    _ps("a[ctivate]/d[eactivate] <name|num|*|*i|*c|*e,...>", "Enable or disable tools (same wildcard support).")
    _ps("u[nregister] <name|num|*...>", "Remove tools permanently (wildcards allowed).")
    _ps("f[ix] <name|num>", "Fix an 'unresolved' tool by re-linking to a callable or converting to external.")
    _ps("n[ew]", "Interactively create a new tool definition (callable or interactive).")
    _ps("m[odify] [g/]<name|num>", "Interactively modify a tool. Use 'g/' prefix for a tool's guide.")
    _ps("p[rint] <name|num>", "Print the JSON definition of a tool.")
    # _ps("ph[eader] [text|'']", "Set or print the global prompt header. Use '' to remove.")
    # _ps("pf[ooter] [text|'']", "Set or print the global prompt footer. Use '' to remove.")
    # _ps("f[ooter] <name|num> [text|'']", "Set or print a specific tool's footer. Use '' to remove.")
    _ps("sa[ve] [path]", "Save toolbox state to a file. Defaults to config path.")
    _ps("l[oad] [path|json]", "Load toolbox state from a file or raw JSON string.")
    _ps("r[eplace] <name|num>", "Replace a tool's definition from a raw JSON string.")
    _ps("g[lobal] <a|s|d>", "Set the context root (default) tools mode (advertised, silent, or disabled).")
    _ps("sc[ope] s[et] m[ode]=... a[dvertised]=... s[ilent]=... d[isabled]=...", "Record a stacked override. Use mode=* to reset to the context default mode.")
    _ps("sc[ope] a[dd] ...", "Same syntax as 'set' (no mode); pushes a new additive layer so that several commands can be combined.")
    _ps("sc[ope] p[op] [--cmd] [pop_id|cmd_id|gen_id|anchor_id]", "Undo the latest scope layer or target a specific id.")
    _ps("sc[ope] [gen_id]", "Show the resolved tool permissions for the active or specified turn with related commands pop_ids.")

    print(f"\n{H}Engine & Debugging{R}")
    _p("/rl [<conv_idx>]", "Remove all LOG commands from a conversation or the entire session.")
    _p("/eng[ine] [status|metrics [reset]]", "Display engine status or metrics. 'reset' flags next request.")
    _p("/config [-r|-p]", "Print current config. -r reconfigures all (restart). -p edits new conversation defaults only.")
    _p("/fp[c|r][-repr] [-s[ave]] [text]", "Format prompt. c=continue, r=root. -repr for special chars. -save to store response.")
    _p("/tk[-repr] <text to tokenize>", "Count tokens in provided, possibly escaped, text.")
    _p("/f[lags] [show|...]", "Manage client-side flags for the session. Use '/f show' for details.")
    _p("/raw <prompt_text>", "Send a raw, unaltered prompt with escaped chars (e.g., \\n).")
    _p("/addmsg <role> <content>", "Manually add a message turn (e.g., role='tool').")
    _p("/e[cho] [--gen_id <id>] [--r] [text]", "Manage inline echo notes on a turn (default replaces on current turn).")

    print("---")


def _print_tools_cli_help() -> None:
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


def _print_adapter_cli_help() -> None:
    print(f"{Colors.HEADER}--- /a (Adapters) Commands ---{Colors.RESET}")
    print("  /a e[num] [--all] [--checkpoints]  Enumerate adapters (loaded + available). Indexes: 0=__base__, then 1..N.")
    print("  /a l[oad] <name|path|#num> [--all] [--checkpoints]  Load adapter into the engine (global). #num refers to '/a e' list. --all includes incompatible.")
    print("  /a u[nload] <name|num>      Unload an adapter from the engine (indexes above).")
    print("  /a sc[ope] s[et] <name|num>...  Set active adapter(s) on the current branch. Using '__base__' or '0' disables all other adapters and must be used alone.")
    print("  /a sc[ope] a[dd] <name|num>...  Add adapter(s) to the active set on this branch (indexes above, __base__ cannot be used with add).")
    print("  /a sc[ope] p[op] [--cmd] [pop_id|cmd_id|gen_id|anchor_id]  Pop (revert) adapter stack command.")
    print("  /a sc[ope] [gen_id]         Show effective adapter stack for active or specified turn with related commands pop_ids.")
    print("  /a q[uery] [name|num]       Show loaded adapters (or details for one) using '/a e' info.")


def _print_session_cli_help() -> None:
    print(f"{Colors.HEADER}--- /s (Session) Commands ---{Colors.RESET}")
    print("  /s n[ew]/sa[ve]/l[oad]/e[num]  Manage session files (/s new|load will drop the current session).")
    print("  /s sync                        Align the active conversation with the current engine config.")
    print("  /s h[l] [gen_id] [--rounds N]  Show one-branch ending at gen_id; rounds limits upward steps.")
    print("  /s hs [gen_id] [--rounds N]    Compact branch summary ending at gen_id, rounds limit upward steps.")
    print("  /s hf/hfl [gen_id] [--rounds N]   Full tree summaries ending at gen_id; rounds caps depth above it.")
    print("  /s hfs[l] [gen_id|*] [--rounds N] Compact summary ending at gen_id; rounds caps depth (ignored for '*').")
    print("  /s ch/chl [gen_id] [--rounds N] Command history ending at gen_id; rounds caps depth above it.")
    print("  /s a[dd] [--title <text>]      Add a new conversation and switch to it.")
    print("  /s c[onv] <idx> [set|del|insert]   Switch/delete/insert conversation at an index.")
    print("  /s c[onv] <idx> [h|hl|hf|hfl|hs|hfs|hfsl|ch|chl] [gen_id] [--rounds N]  Conversation <idx> history helpers.")
    print("  /s re[play] <idx> [...]        Replay a branch (use '/s replay ?' for syntax).")
    print("  /s p[rompt] [gen_id]           Print LLM prompt path for active turn or a specific gen_id.")
    print("  /s t[urn] <target>             Navigate or toggle flags. Use '/s t ?' for details.")


def _print_replay_cli_help() -> None:
    print(f"{Colors.HEADER}--- /s replay Syntax ---{Colors.RESET}")
    print("  /s replay <conversation_index> one-branch [options]")
    print("    Replays a single branch from the source conversation (auto-generated turns are skipped).")
    print("    --from GEN_ID       Start replaying from the specified turn (optional).")
    print("    --root_ctx          Mark the current destination turn as root_ctx before replay.")
    print("    --debug             Show verbose replay trace (turn traversal, batch details).")
    print("\n  /s replay <conversation_index> all-down [options]")
    print("    Replays the entire conversation from the top down, preserving structure (auto-generated turns are skipped).")
    print("    --from GEN_ID       Start from a root_ctx turn only (enforced).")
    print("    --root_ctx          Mark the current destination turn as root_ctx before replay.")
    print("    --debug             Show verbose replay trace (turn traversal, batch details).")
    print(f"\n{Colors.SYSTEM}Example:{Colors.RESET} /s replay 1 all-down --from g_12 --root_ctx")


def _is_descendant_of(node: Optional[Turn], ancestor: Optional[Turn]) -> bool:
    """Return True if `node` is a descendant of `ancestor` in the turn tree."""
    if not node or not ancestor:
        return False
    current = getattr(node, "parent", None)
    while current is not None:
        if current is ancestor:
            return True
        current = getattr(current, "parent", None)
    return False


def _depth_from_start(node: Optional[Turn], start: Optional[Turn]) -> Optional[int]:
    """Return depth from start -> node (0 = start). None if node not under start."""
    if not node or not start:
        return None
    depth = 0
    current = node
    while current is not None:
        if current is start:
            return depth
        current = getattr(current, "parent", None)
        depth += 1
    return None


def _within_history_scope(node: Optional[Turn], start: Optional[Turn], depth: Optional[int]) -> bool:
    """Return True if node is within start subtree and depth limit."""
    if not start:
        return True
    dist = _depth_from_start(node, start)
    if dist is None:
        return False
    if depth is None:
        return True
    return dist <= depth


def _history_has_children(
    node: Optional[Turn],
    start: Optional[Turn],
    depth: Optional[int],
) -> bool:
    if not node:
        return False
    has_children = bool(getattr(node, "cmd", None) or getattr(node, "turns", None))
    if not has_children:
        return False
    if start and depth is not None:
        dist = _depth_from_start(node, start)
        if dist is None or dist >= depth:
            return False
    return True


def _history_scope_start_turn(target_turn: Optional[Turn], rounds: Optional[int]) -> Optional[Turn]:
    """Return the virtual history start based on rounds up from target_turn (None => root)."""
    if not target_turn:
        return None
    if rounds is None:
        cur = target_turn
        while getattr(cur, "parent", None):
            cur = cur.parent
        return cur
    cur = target_turn
    for _ in range(max(0, rounds)):
        parent = getattr(cur, "parent", None)
        if parent is None:
            break
        cur = parent
    return cur

def _history_branch_path_from_start(
    cursor: ChatCursor,
    target_turn: Optional[Turn],
    start_turn: Optional[Turn],
) -> List[Turn]:
    """Return active-path slice from start_turn to focus_turn (or full path if no start)."""
    target = target_turn or cursor.current_turn
    if not target:
        return []
    path = cursor.session.get_active_path_for_llm(target)
    if not start_turn:
        return path
    try:
        start_idx = path.index(start_turn)
    except ValueError:
        return path
    return path[start_idx:]

def _resolve_history_scope(
    cursor: ChatCursor,
    gen_id: Optional[str],
    rounds: Optional[int],
) -> Tuple[Optional[ChatCursor], Optional[Turn]]:
    """Resolve a history cursor and scope-start turn."""
    history_cursor, target_turn = _history_cursor_for_gen_id(cursor, gen_id)
    if not history_cursor:
        return None, None
    target_turn = target_turn or history_cursor.current_turn
    scope_start = _history_scope_start_turn(target_turn, rounds)
    return history_cursor, scope_start


def _parse_history_args(arg_str: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Parse history args, returning (gen_id, rounds, error_msg)."""
    if not arg_str:
        return None, None, None
    tokens = shlex.split(arg_str)
    gen_id: Optional[str] = None
    rounds: Optional[int] = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--rounds="):
            raw = token.split("=", 1)[1]
        elif token in ("--rounds"):
            if i + 1 >= len(tokens):
                return gen_id, None, "Missing value for --rounds."
            i += 1
            raw = tokens[i]
        else:
            if gen_id is None:
                gen_id = token
            i += 1
            continue
        try:
            rounds_val = int(raw)
        except ValueError:
            return gen_id, None, f"Invalid rounds value '{raw}'."
        if rounds_val < 0:
            return gen_id, None, "Rounds must be >= 0."
        rounds = rounds_val
        i += 1
        continue
    return gen_id, rounds, None


def _history_cursor_for_gen_id(
    cursor: ChatCursor,
    gen_id: Optional[str],
) -> Tuple[Optional[ChatCursor], Optional[Turn]]:
    """
    Resolve a cursor pinned to gen_id within the active conversation.

    Returns a tuple of (resolved_cursor, start_turn). start_turn is set when
    a gen_id is supplied so callers can render from that node.
    """
    if not gen_id:
        return cursor, None
    try:
        resolved = cursor.cursor_for_gen_id(gen_id)
        return resolved, resolved.current_turn
    except KeyError:
        print(f"{Colors.ERROR}Turn with gen_id '{gen_id}' not found in this conversation.{Colors.RESET}")
    except ValueError as err:
        print(f"{Colors.ERROR}{err}{Colors.RESET}")
    return None, None


def _print_all_conversation_summaries(
    base_cursor: ChatCursor,
    show_logs: bool = False,
    *,
    depth: Optional[int] = None,
) -> None:
    """Print tree summaries for every conversation in the current session."""
    session = base_cursor.session
    conversations = session.conversations
    if not conversations:
        print(f"{Colors.TOOL_WARNING}No conversations available in this session.{Colors.RESET}")
        return
    temp_contexts: List[ChatContext] = []
    try:
        for idx, conv in enumerate(conversations, start=1):
            if conv is base_cursor.chat_session:
                working_cursor = base_cursor
            else:
                _ensure_chat_session_toolbox(conv)
                temp_ctx = ChatContext(session, chat_session=conv, toolbox=toolbox)
                temp_contexts.append(temp_ctx)
                working_cursor = temp_ctx.active_cursor
            title = f"{_conversation_label(session, conv)} ({session.name})"
            start_turn = conv.root_turn if depth is not None else None
            print_session_tree_summary(
                working_cursor,
                title=title,
                active=working_cursor.current_turn,
                start_at_turn=start_turn,
                depth=depth,
                show_logs=show_logs,
            )
    finally:
        temp_contexts.clear()


def _parser_profile_label(conv: ChatSession) -> str:
    profile = conv.parser_profile
    if isinstance(profile, dict):
        return str(profile.get("key") or profile.get("name") or "N/A")
    if profile is None:
        return "N/A"
    key = getattr(profile, "key", None)
    name = getattr(profile, "name", None)
    return str(key or name or "N/A")


def _collect_simple_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    simple: Dict[str, Any] = {}
    for key, value in settings.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            simple[key] = value
    return simple


def _preferred_summary_turn(
    conversation: ChatSession,
    *,
    active_cursor: Optional[ChatCursor] = None,
    use_active: bool = False,
) -> Optional[Turn]:
    """Choose a summary turn for conversation listings."""
    if use_active and active_cursor and active_cursor.chat_session is conversation:
        turn = active_cursor.current_turn
        if turn and getattr(turn, "IsPlaceholderLike", False):
            return turn.parent or (turn.turns[0] if getattr(turn, "turns", None) else turn)
        return turn or conversation.root_turn

    turn = conversation.root_turn
    if turn and getattr(turn, "IsPlaceholderLike", False):
        children = getattr(turn, "turns", []) or []
        if children:
            turn = children[0]
    return turn


def _print_conversation_summary(
    session: EngineSession,
    conversation: ChatSession,
    index: int,
    *,
    active: bool,
    single: bool = False,
) -> None:
    active_marker = f" {Colors.BRIGHT_CYAN}*(active)*{Colors.RESET}" if active else ""
    title = (getattr(conversation, "title", "") or "").strip()
    header_base = f"Conversation {index}"
    if title:
        header_base = f"{header_base}: {title}"
    header = f"{header_base}{active_marker}"
    if single:
        print(f"\n{Colors.HEADER}--- {header} ---{Colors.RESET}")
    else:
        print(f"\n{Colors.HEADER}{header}:{Colors.RESET}")
    engine_name = (
        conversation.engine_config.get("base_model_name")
        or conversation.engine_config.get("base_model_name_or_path")
        or "N/A"
    )
    parser_label = _parser_profile_label(conversation)
    num_nodes = session.count_nodes_in_tree(conversation.root_turn)
    last_turn = session.get_main_branch_leaf(index)
    last_turn_label = session.get_display_id(last_turn) if last_turn else "N/A"
    last_turn_gen = last_turn.gen_id_or_parent if last_turn else "N/A"
    print(f"  ID: {conversation.id}")
    print(f"  Engine Name: {engine_name}")
    print(f"  Parser Profile: {parser_label}")
    print(f"  Turn Nodes: {num_nodes}")
    print(f"  Last Main Turn: {last_turn_label} ({last_turn_gen})")

    simple_params = _collect_simple_settings(conversation.initial_params)
    if simple_params:
        print(f"  Settings:")
        for key in sorted(simple_params.keys()):
            value = simple_params[key]
            value_display = _clip_long_message(str(value)) if isinstance(value, str) else value
            print(f"    {key}: {value_display}")

    defaults_data: Dict[str, Any] = {}
    if hasattr(conversation.inference_defaults, "serialize"):
        defaults_data = conversation.inference_defaults.serialize()
    elif hasattr(conversation.inference_defaults, "__dict__"):
        defaults_data = dict(conversation.inference_defaults.__dict__)
    simple_defaults = _collect_simple_settings(defaults_data)
    if simple_defaults:
        print(f"  Inference Defaults:")
        for key in sorted(simple_defaults.keys()):
            print(f"    {key}: {simple_defaults[key]}")


def _conversation_cursor_for_index(
    session: EngineSession,
    conversation: ChatSession,
    base_cursor: ChatCursor,
) -> Tuple[ChatCursor, Optional[ChatContext]]:
    if conversation is base_cursor.chat_session:
        return base_cursor, None
    _ensure_chat_session_toolbox(conversation)
    _sync_inference_defaults_from_initial(conversation)
    _apply_tool_params_to_toolbox(conversation.initial_params)
    temp_context = ChatContext(session, chat_session=conversation, toolbox=toolbox)
    target_turn, _warning = _resolve_target_turn_for_conversation(session, conversation)
    if target_turn and target_turn is not temp_context.active_cursor.current_turn:
        rebound = temp_context.register_cursor_for_turn(target_turn, make_active=True)
        if rebound:
            return rebound, temp_context
    return temp_context.active_cursor, temp_context


def _print_command_history(
    context: ChatContext,
    include_logs: bool,
    *,
    conversation: Optional[ChatSession] = None,
    start_at_turn: Optional[Turn] = None,
    depth: Optional[int] = None,
) -> None:
    title_suffix = " (including logs)" if include_logs else ""
    session = context.session
    scope_label = (
        f" for Conversation {session.conversations.index(conversation)}"
        if conversation and conversation in session.conversations
        else ""
    )
    print(f"\n--- Command History for Session: {session.name}{scope_label}{title_suffix} ---")

    start_node = conversation.root_turn if conversation else context.root_turn
    
    history = context.get_scope_commands(start_node=start_node, include_logs=include_logs)

    if start_at_turn or depth is not None:
        turn_lookup: Dict[str, Turn] = {}
        if conversation:
            turn_lookup = {
                t.gen_id: t
                for t in session._get_all_turns(conversation)
                if getattr(t, "gen_id", None)
            }

        def _within_scope(cmd: Command) -> bool:
            parent = cmd.parent
            parent_turn: Optional[Turn] = None
            if isinstance(parent, Turn):
                parent_turn = parent
            elif isinstance(parent, str):
                parent_turn = turn_lookup.get(parent)
            if parent_turn is None:
                return start_at_turn is None
            return _within_history_scope(parent_turn, start_at_turn, depth)

        history = [cmd for cmd in history if _within_scope(cmd)]

    if not history:
        print(f"{Colors.TOOL_WARNING}No commands recorded for this scope.{Colors.RESET}")
        print("---")
        return

    print(f"{'Time':<10} {'Turn':<10} {'Type':<12} {'Text':<54} {'ID':<10} {'Facts'}")
    print(f"{'-'*10} {'-'*10} {'-'*12} {'-'*54} {'-'*10} {'-'*20}")
    TYPE_ABBREVIATIONS = {
        "COMMAND": "cmd",
        "PARAM_CHANGE": "param",
        "STATE_CHANGE": "state",
        "ADAPTERS_STATE": "adapt",
        "CHAT": "chat",
    }
    for cmd in history:
        ts = time.strftime('%H:%M:%S', time.localtime(cmd.timestamp)) # type: ignore
        item_id = cmd.gen_id
        turn_label = ""
        item_type = TYPE_ABBREVIATIONS.get(cmd.cmd_type.upper(), cmd.cmd_type.lower())
        facts: List[str] = []

        turn_parent = cmd.parent
        parent_turn = turn_parent if isinstance(turn_parent, Turn) else None
        parent_gen_id: Optional[str] = None
        if isinstance(turn_parent, Turn):
            parent_gen_id = turn_parent.gen_id or getattr(turn_parent, "gen_id_or_parent", None)
            try:
                turn_label = session.get_display_id(turn_parent)
            except Exception:
                turn_label = ""
        elif isinstance(turn_parent, str):
            parent_gen_id = turn_parent
            turn_label = f"(del:{turn_parent.split('_')[-1]})"

        if turn_label:
            facts.append(f"display_id={turn_label}")

        display_text = ""
        if cmd.cmd_type in {Command.COMMAND, Command.LOG}:
            base_text = cmd.data.get('command', '')
            if cmd.cmd_type == Command.LOG and cmd.data.get("echo"):
                facts.append("echo")
                base_text = cmd.data.get("text") or base_text
                display_text = f"{Colors.ECHO}[echo]{Colors.RESET} {base_text}"
            else:
                display_text = base_text
        elif cmd.cmd_type == Command.PARAM_CHANGE:
            display_text = f"{cmd.data.get('change')} -> {cmd.data.get('new_value')}"
        elif cmd.cmd_type == Command.STATE_CHANGE:
            change = cmd.data.get('change')
            if change == 'tools_scope':
                scope_data = cmd.data.get('scope')
                if isinstance(scope_data, dict):
                    try:
                        scope_obj = ToolsScope.from_dict(scope_data)
                        display_text = f"tools_scope = {scope_obj.describe()}"
                    except Exception as e:
                        display_text = f"tools_scope = {scope_data} (Error: {e})"
                else:
                    display_text = f"tools_scope = {scope_data}"
            else:
                value = cmd.data.get('value')
                display_text = f"{change} = {value}"
        elif cmd.cmd_type == Command.ADAPTERS_STATE:
            display_text = f"State: {cmd.data.get('state_type')} = {cmd.data.get('value')}"
        elif cmd.cmd_type == Command.TURN:
            fork_index = -1
            if parent_turn:
                if parent_turn.is_archived:
                    facts.append("archived")
                
                if parent_turn.parent and len(parent_turn.parent.turns) > 1:
                    try:
                        # Find the fork index relative to its siblings
                        fork_index = parent_turn.parent.turns.index(parent_turn) + 1
                        facts.append(f"fork:{fork_index}")
                    except ValueError:
                        # Fallback for safety, though it shouldn't happen
                        if not parent_turn.main_thread and parent_turn.turn_type in [Turn.FORK, Turn.BATCH]:
                             facts.append("fork")
                elif not parent_turn.main_thread and parent_turn.turn_type in [Turn.FORK, Turn.BATCH]:
                    facts.append("fork")

            method = cmd.data.get('method', '')
            prompt = cmd.data.get('prompt', '')

            # Default text
            display_text = f"[{method}] {prompt}" if method else prompt

            # If parentage exists, enhance the text
            if parent_turn and parent_turn.parent and parent_turn.parent.gen_id:
                # If prompt is just a gen_id (e.g., "[g_20]"), it's redundant.
                is_redundant_prompt = prompt.startswith('[g_') and prompt.endswith(']')

                if is_redundant_prompt:
                    display_text = f"[{method}] : off:[{parent_turn.parent.gen_id}]"
                else:
                    display_text = f"[{method}] {prompt} : off:[{parent_turn.parent.gen_id}]"

            # Fallback for inferred parent if not set via parent_turn
            if not parent_gen_id:
                inferred_parent = cmd.data.get("parent_gen_id") or ""
                if inferred_parent:
                    display_text = f"{display_text} [{inferred_parent}]"
        else:
            display_text = cmd.data.get('command', '')

        facts_str = ", ".join(facts)
        text_str = str(display_text)
        print(f"{ts:<10} {str(parent_gen_id or 'N/A'):<10} {item_type:<12} {text_str:<54.54} {str(item_id or 'N/A'):<10} {facts_str}")
    print("---")

def _delete_conversation_at_index(session: EngineSession, index: int) -> Optional[ChatCursor]:

    # Expects zero-based index (user input is converted beforehand).
    if session.get_conversations_count() <= 1:
        print(f"{Colors.ERROR}Cannot delete the only conversation in the session.{Colors.RESET}")
        return None
    if index < 0 or index >= session.get_conversations_count():
        print(f"{Colors.ERROR}Invalid conversation index '{index + 1}'.{Colors.RESET}")
        return None

    cur_cursor = _require_current_cursor()
    cur_index = -1
    if cur_cursor.chat_session:
        cur_index = session.conversations.index(cur_cursor.chat_session)

    removed = session.conversations.pop(index)
    print(f"{Colors.SYSTEM}Deleted conversation {index + 1} (ID: {removed.id}). Remaining: {len(session.conversations)}.{Colors.RESET}")
    if cur_index >= 0 and cur_index != index:
        return cur_cursor
     
    if cur_index < 0:
        # should not happen, adding new conv seem as right action here.
        loaded_chat_session = session.add_conversation(parser_profile=_engine_parser_profile())
        _seed_chat_session_flags(loaded_chat_session)
        cur_index = 0
    elif cur_index == index:
        # switching to right or last conversation as active.
        cur_index = min(index, session.get_conversations_count() - 1)

    target_conversation = session.get_conversation(cur_index)
    cur_cursor = _bootstrap_cursor_for_session(session, conversation=target_conversation)
    print_current_session_compact(cur_cursor, clear_first=False)
    print(f"{Colors.SYSTEM}Switched to conversation {cur_index + 1}.{Colors.RESET}")
    return cur_cursor

def _insert_conversation_at_index(
    session: EngineSession,
    index: int,
    template: ChatSession,
    title: Optional[str] = None,
) -> ChatCursor:
    # Expects zero-based index (user input is converted beforehand).
    if index < 0 or index > session.get_conversations_count():
        print(f"{Colors.ERROR}Invalid insert index '{index + 1}'.{Colors.RESET}")
        index = session.get_conversations_count()
    new_conv = session.insert_conversation(
        index, 
        template, 
        title=title)
    _seed_chat_session_flags(new_conv)

    print(f"{Colors.SYSTEM}Inserted new conversation at index {index + 1} using template settings (history is not copied).{Colors.RESET}")
    if title:
        print(f"{Colors.SYSTEM}Conversation title set to '{title}'.{Colors.RESET}")
    return _bootstrap_cursor_for_session(session, conversation=new_conv)

def _run_conversation_history_action(
    action: str,
    target_conversation: ChatSession,
    base_cursor: ChatCursor,
    args: str,
) -> None:
    conv_cursor, temp_context = _conversation_cursor_for_index(base_cursor.session, target_conversation, base_cursor)
    try:
        normalized = action.lower()
        gen_arg, rounds, err = _parse_history_args(args)
        if err:
            print(f"{Colors.ERROR}{err}{Colors.RESET}")
            return
        if normalized in ["hfs", "hfsl"]:
            show_logs = (normalized == "hfsl")
            target_cursor, scope_start = _resolve_history_scope(conv_cursor, gen_arg, rounds)
            depth_limit = rounds
            if target_cursor:
                print_session_tree_summary(
                    target_cursor,
                    active=target_cursor.current_turn,
                    start_at_turn=scope_start,
                    depth=depth_limit,
                    show_logs=show_logs,
                )
        elif normalized == "hs":
            target_cursor, scope_start = _resolve_history_scope(conv_cursor, gen_arg, rounds)
            if target_cursor:
                branch_path = _history_branch_path_from_start(target_cursor, target_cursor.current_turn, scope_start)
                print_current_session_compact(
                    target_cursor,
                    clear_first=False,
                    start_at_turn=scope_start,
                    branch_path=branch_path,
                )
        elif normalized == "hl":
            target_cursor, scope_start = _resolve_history_scope(conv_cursor, gen_arg, rounds)
            if target_cursor:
                branch_path = _history_branch_path_from_start(target_cursor, target_cursor.current_turn, scope_start)
                print_current_session_info(
                    target_cursor,
                    show_messages=True,
                    show_logs=True,
                    clear_first=False,
                    active_only=True,
                    start_at_turn=scope_start,
                    branch_path=branch_path,
                )
        elif normalized == "h":
            target_cursor, scope_start = _resolve_history_scope(conv_cursor, gen_arg, rounds)
            if target_cursor:
                branch_path = _history_branch_path_from_start(target_cursor, target_cursor.current_turn, scope_start)
                print_current_session_info(
                    target_cursor,
                    show_messages=True,
                    show_logs=False,
                    clear_first=False,
                    active_only=True,
                    start_at_turn=scope_start,
                    branch_path=branch_path,
                )
        elif normalized == "hf":
            target_cursor, scope_start = _resolve_history_scope(conv_cursor, gen_arg, rounds)
            depth_limit = rounds
            if target_cursor:
                print_current_session_info(
                    target_cursor,
                    show_messages=True,
                    show_logs=False,
                    clear_first=False,
                    start_at_turn=scope_start,
                    depth=depth_limit,
                    show_all_turns=True,
                )
        elif normalized == "hfl":
            target_cursor, scope_start = _resolve_history_scope(conv_cursor, gen_arg, rounds)
            depth_limit = rounds
            if target_cursor:
                print_current_session_info(
                    target_cursor,
                    show_messages=True,
                    show_logs=True,
                    clear_first=False,
                    start_at_turn=scope_start,
                    depth=depth_limit,
                    show_all_turns=True,
                )
        elif normalized == "ch":
            target_cursor, scope_start = _resolve_history_scope(conv_cursor, gen_arg, rounds)
            depth_limit = rounds
            if target_cursor:
                _print_command_history(
                    temp_context or _active_chat_context(),
                    include_logs=False,
                    conversation=target_conversation,
                    start_at_turn=scope_start,
                    depth=depth_limit,
                )
        elif normalized == "chl":
            target_cursor, scope_start = _resolve_history_scope(conv_cursor, gen_arg, rounds)
            depth_limit = rounds
            if target_cursor:
                _print_command_history(
                    temp_context or _active_chat_context(),
                    include_logs=True,
                    conversation=target_conversation,
                    start_at_turn=scope_start,
                    depth=depth_limit,
                )
        else:
            print(f"{Colors.ERROR}Unknown conversation history action '{action}'.{Colors.RESET}")
    finally:
        temp_context = None


def _print_system_message_cli_help() -> None:
    print(f"{Colors.HEADER}--- /sm (System Message) Commands ---{Colors.RESET}")
    print("  /sm s[et] <text|''|<>|<def>>   Set system message (''=empty, <> remove), <def> chat template default.")
    print("  /sm a[dd] <text>               Append frgamnet to current system message stack.")
    print("  /sm pop [--cmd] [pop_id|cmd_id|gen_id|anchor_id]  Remove last set/add layer or target a specific turn/command.")
    print("     - No args: drop the most recent set/add.")
    print("     - pop_id: drop by generated stack_id.")
    print("     - gen_id/anchor_id: drop everything after that turn (inclusive).")
    print("     - cmd_id: drop only that system-message command.")
    print("  /sm sc[ope] [gen_id]           Show effective system message, segments, and related commands pop_ids.")
    print("  /sm                            Show current system message segments.")


def _all_tool_names(toolbox: Toolbox) -> List[str]:
    try:
        return [entry[0] for entry in toolbox.list_tools()]
    except Exception:
        return []


def _tool_wildcard_groups(toolbox: Toolbox) -> Dict[str, List[str]]:
    """
    Returns wildcard shortcut groups for tool selection.
    *i -> intrinsic tools, *c -> callable tools, *e -> external tools.
    """
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

async def _parse_cli_targets(
    targets_str: str,
    enumerated_list: List[Any],
    name_key: Optional[str] = None,
    *,
    allow_wildcard: bool = False,
    wildcard_values: Optional[List[str]] = None,
    wildcard_groups: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[str]:
    """
    Parses a comma-separated string of names or numbers into a list of resolved names.
    - targets_str: The comma-separated input string (e.g., "1, my_item, 3").
    - enumerated_list: The list used for number-based selection (e.g., LAST_ENUMERATED_ADAPTERS).
    - name_key: If the list contains dicts, this is the key to get the name (e.g., "name").
    """
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

    targets = [t.strip() for t in targets_str.split(',') if t.strip()]
    for target in targets:
        if not target:
            continue
        lowered = target.lower()
        if allow_wildcard and lowered in normalized_groups:
            for value in normalized_groups[lowered]:
                if value and value not in resolved_names:
                    resolved_names.append(value)
            continue
        try:
            # Try to parse as a 1-based index first
            idx = int(target) - 1
            if 0 <= idx < len(enumerated_list):
                item = enumerated_list[idx]
                resolved_names.append(item[name_key] if name_key and isinstance(item, dict) else item)
            else:
                print(f"Warning: Invalid number '{target}' ignored (out of range).")
        except ValueError:
            # If not a number, treat it as a name
            resolved_names.append(target)
    return resolved_names

async def _prompt_for_selection(
    item_list: List[str], # The actual values to return upon selection
    item_names_for_prompt: List[str], # Names to display in the prompt
    prompt_message: str,
    allow_multiple: bool = False,
    start_index: int = 1,
) -> Union[None, str, List[str]]:
    """
    Displays a numbered list of items and prompts the user for selection.
    Returns the selected item_list value(s), or None if cancelled.
    """
    if not item_list:
        print(f"No items available for selection.")
        return None

    print(f"\n--- {prompt_message} ---")
    for i, name_to_display in enumerate(item_names_for_prompt):
        print(f"  {i + start_index}. {name_to_display}")
    print("---")

    while True:
        try:
            if allow_multiple:
                user_choice_str = await asyncio.to_thread(input, f"Enter number(s) (e.g., {start_index} or {start_index},{start_index+2}) or press Enter to cancel: ")
            else:
                user_choice_str = await asyncio.to_thread(input, f"Enter number or press Enter to cancel: ")
        except KeyboardInterrupt:
            print("\nSelection cancelled by user (Ctrl+C).")
            return None
        except EOFError:
            print("\nSelection cancelled (EOF).")
            return None

        if not user_choice_str.strip():
            print("Selection cancelled.")
            return None

        try:
            parts = user_choice_str.split(',')
            selected_indices: List[int] = []
            for part in parts:
                num = int(part.strip())
                idx = num - start_index
                if 0 <= idx < len(item_list):
                    selected_indices.append(idx)
                else:
                    raise ValueError("Number out of range.")
            if not selected_indices:
                raise ValueError("No valid numbers entered.")

            if allow_multiple:
                return [item_list[i] for i in selected_indices]
            elif len(selected_indices) == 1: # Ensure only one selection if not allow_multiple
                return item_list[selected_indices[0]]
            else: # User entered multiple numbers when only one was expected
                raise ValueError("Multiple selections are not allowed here.")

        except ValueError as e:
            print(f"Invalid input: {e}. Please enter number(s) from {start_index} to {start_index + len(item_list) - 1}{' (comma-separated for multiple)' if allow_multiple else ''}, or Enter to cancel.")

async def _resolve_adapter_targets_1based(
    raw_targets: str,
    prompt_message: str,
    allow_multiple: bool,
    *,
    normalize: bool = True,
    cursor: Optional[ChatCursor] = None,
) -> Optional[List[str]]:
    """
    Resolves adapter targets from loaded adapters (0 = __base__, then 1..N from engine get-loaded-adapters).
    Returns a list of adapter names or None if cancelled.
    """
    resp = await call_api("get-loaded-adapters", {})
    loaded = resp.get("data", {}).get("adapters", []) if resp.get("status") == "success" else []
    selectable_names = ["__base__"] + [item.get("name") for item in loaded]

    def _print_table():
        print(f"\n{Colors.HEADER}--- {prompt_message} ---{Colors.RESET}")
        header = f"{'Idx':>3} {'Name':<30} {'Prec':<8} Path"
        print(header)
        print(f"  0. __base__")
        for idx, item in enumerate(loaded, start=1):
            quant = item.get("base_model_quant") or (item.get("metadata") or {}).get("base_model_effective_dtype_at_init") or "N/A"
            path = item.get("root_path") or item.get("checkpoint_path") or ""
            name = item.get("name") or "<unknown>"
            print(f"{idx:>4} {name:<30} {str(quant):<8} {path}")
        print("---")

    if not raw_targets.strip():
        if not selectable_names:
            print(f"{Colors.TOOL_WARNING}No adapters available to select from.{Colors.RESET}")
            return None
        _print_table()
        try:
            user_input = await asyncio.to_thread(input, "Enter number(s) (comma-separated) or press Enter to cancel: ")
        except Exception:
            return None
        raw_targets = user_input or ""
        if not raw_targets.strip():
            return None

    resolved_names: List[str] = []
    parts = [t.strip() for t in raw_targets.split(',') if t.strip()]
    for part in parts:
        try:
            idx = int(part.lstrip("#"))
            if idx == 0:
                resolved_names.append("__base__")
            elif 0 < idx < len(selectable_names):
                resolved_names.append(selectable_names[idx])
            else:
                print(f"Warning: Invalid adapter number '{part}' ignored (out of range).")
        except ValueError:
            resolved_names.append(part)

    return _normalize_adapter_name_list(resolved_names) if normalize else resolved_names


async def _handle_adapter_command(args_str: str, cursor: ChatCursor, pt_session: "PromptSession") -> Tuple[ChatCursor, bool]:
    global LAST_ENUMERATED_ADAPTERS, LAST_ENUMERATED_LOCAL_ADAPTERS, current_config
    user_input = f"/a {args_str}"
    if args_str.strip() in {"?", "help"}:
        _print_adapter_cli_help()
        return cursor, True
    parts = args_str.split(" ", 1)
    sub_cmd_full = parts[0].lower()
    sub_args = parts[1] if len(parts) > 1 else ""

    sub_cmd_map = {
        "e": "enum", "enum": "enum",
        "l": "load", "load": "load",
        "u": "unload", "unload": "unload",
        "s": "set_active", "set": "set_active",
        "act": "set_active", "activate": "set_active",
        "a": "add_active", "add": "add_active",
        "p": "pop_active", "pop": "pop_active",
        "q": "query", "query": "query",
        "sc": "scope", "scope": "scope",
    }
    sub_cmd = sub_cmd_map.get(sub_cmd_full) or sub_cmd_map.get(sub_cmd_full[0] if sub_cmd_full else "")

    if not sub_cmd_full.strip():
        sub_cmd = "enum"
        cursor.log_command(user_input)
    elif not sub_cmd:
        print(f"{Colors.ERROR}Unknown adapter command: '{sub_cmd_full}'.{Colors.RESET}")
        print("Valid options are: e[num], l[oad], u[nload], sc[ope], s[et], a[dd], p[op], q[uery].")
        return cursor, True

    if sub_cmd == "enum":
        _, include_all, include_checkpoints = _parse_adapter_list_flags(sub_args)
        cursor.log_command(user_input)
        effective_adapters = cursor.get_effective_adapters()
        adapters_display = ", ".join(effective_adapters) if effective_adapters else "__base__"
        print(f"{Colors.SYSTEM}Active adapter(s) for current turn:{Colors.RESET} {adapters_display}")
        await list_adapters_from_engine(include_incompatible=include_all, include_checkpoints=include_checkpoints)
        _print_loaded_adapters(cursor)
    elif sub_cmd == "load":
        target_to_load = sub_args.strip()

        include_checkpoints = False
        if target_to_load:
            target_to_load, include_all, include_checkpoints = _parse_adapter_list_flags(target_to_load)
        else:
            include_all = False

        # Non-interactive load (with arguments)
        if target_to_load:
            if target_to_load.startswith("#"):
                try:
                    idx = int(target_to_load[1:])
                except ValueError:
                    print(f"{Colors.ERROR}Invalid adapter number: {target_to_load}{Colors.RESET}")
                    return cursor, True
                await load_adapter_to_engine(f"{idx}", cursor, include_incompatible=include_all)
            else:
                await load_adapter_to_engine(target_to_load, cursor, include_incompatible=include_all)
            return cursor, True

        # Interactive load (no arguments): reuse enum list
        await list_adapters_from_engine(include_incompatible=include_all, include_checkpoints=include_checkpoints)
        if not LAST_ENUMERATED_ADAPTERS:
            print("No adapters available to load.")
            return cursor, True
        print(f"Adapters in '{current_config['adapters_root_dir']}' (from /a e):")
        header = f"{'Idx':>3} {'Name':<30} {'Prec':<8} {'Info':<15} Path"
        print(header)
        for idx, info in enumerate(LAST_ENUMERATED_ADAPTERS, start=1):
            info_dict = info.get("info", {}) if isinstance(info.get("info"), dict) else {}
            quant = info_dict.get("base_model_quant", info.get("base_model_quant"))
            meta = info_dict.get("metadata") or {}
            quant = quant or meta.get("base_model_effective_dtype_at_init") or "N/A"
            rel_path = info.get("rel_path") or info.get("abs_path") or info.get("path", "")
            flag = "non-compat"
            if info_dict.get("is_loaded"):
                flag = "*(loaded)*"
            elif info_dict.get("is_new"):
                flag = "new"
            elif info_dict.get("is_foreign"):
                flag = "foreign"
            elif info_dict.get("is_compatible"):
                path_obj = Path(info_dict.get("path", ""))
                try:
                    mtime = datetime.datetime.fromtimestamp(path_obj.stat().st_mtime)
                    flag = mtime.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    flag = "last"
            print(f"{idx:>4} {info['name']:<30} {str(quant):<8} {flag:<15} {rel_path}")
        try:
            response = await pt_session.prompt_async("Enter adapter number to load (or press Enter to cancel): ")
            response = response.strip()
            if not response:
                print("Adapter load cancelled.")
                return cursor, True
            if response.startswith("#"):
                response = response[1:]
            idx = int(response)
            await load_adapter_to_engine(f"{idx}", cursor, include_incompatible=include_all)
        except ValueError:
            print("Invalid input. Please enter a number.")
        return cursor, True
    elif sub_cmd == "unload":
        resolved_names = await _resolve_adapter_targets_1based(sub_args, "Select adapter(s) to unload", allow_multiple=True, cursor=cursor)
        if resolved_names is None:
            print(f"{Colors.SYSTEM}Adapter unload cancelled.{Colors.RESET}")
            return cursor, True
        if not resolved_names:
            print(f"{Colors.ERROR}No valid adapters specified for unload. Use '/a u <name|num>, ...' or select interactively.{Colors.RESET}")
            return cursor, True
        for name in resolved_names:
            if name == "__base__":
                print(f"{Colors.SYSTEM}Info: '__base__' is an alias for the base model and cannot be unloaded.{Colors.RESET}")
                continue
            await unload_adapter_from_engine(cursor, name)

    elif sub_cmd == "scope":
        scope_args = sub_args.strip()
        if scope_args in {"?", "help"}:
            _print_adapter_cli_help()
            return cursor, True
        if not scope_args:
            action = "show"
            remainder = ""
        else:
            action_token, _, remainder = scope_args.partition(" ")
            action_token = action_token.lower()
            remainder = remainder.strip()
            action_map = {
                "set": "set", "s": "set",
                "add": "add", "a": "add",
                "pop": "pop", "p": "pop",
                "show": "show", "status": "show", "": "show",
            }
            action = action_map.get(action_token)
            if not action:
                action = "show"
                remainder = scope_args
        command_text = f"/a scope {scope_args}".strip()
        if action == "show":
            target_cursor = cursor
            gen_id_arg = remainder.strip()
            if gen_id_arg:
                try:
                    target_cursor = cursor.cursor_for_gen_id(gen_id_arg)
                except KeyError:
                    print(f"{Colors.ERROR}Turn with gen_id '{gen_id_arg}' not found.{Colors.RESET}")
                    return cursor, True
                except ValueError as err:
                    print(f"{Colors.ERROR}{err}{Colors.RESET}")
                    return cursor, True
            entries = _collect_adapter_scope_entries(target_cursor)
            _print_adapter_scope_summary(target_cursor, entries)
            return cursor, True
        if action == "set":
            resolved_names = await _resolve_adapter_targets_1based(remainder, "Select adapter(s) to set as active", allow_multiple=True, cursor=cursor)
            if resolved_names is None:
                print(f"{Colors.SYSTEM}Adapter selection cancelled.{Colors.RESET}")
                return cursor, True
            if "__base__" in resolved_names and len(resolved_names) > 1:
                print(f"{Colors.ERROR}The '__base__' adapter cannot be set with other adapters. Use '/a scope set 0' to disable all adapters.{Colors.RESET}")
                return cursor, True
            apply_adapter_operation(cursor, "set", resolved_names, command_text)
            return cursor, True
        if action == "add":
            resolved_names = await _resolve_adapter_targets_1based(remainder, "Select adapter(s) to add to active", allow_multiple=True, cursor=cursor)
            if resolved_names is None:
                print(f"{Colors.SYSTEM}Adapter add cancelled.{Colors.RESET}")
                return cursor, True
            if not resolved_names:
                print(f"{Colors.ERROR}No valid adapters specified for add. Use '/a scope add <name|num>, ...' or select interactively.{Colors.RESET}")
                return cursor, True
            if "__base__" in resolved_names:
                print(f"{Colors.ERROR}'__base__' cannot be added. Use '/a scope set 0' to reset to the base adapter or pop to revert.{Colors.RESET}")
                return cursor, True
            apply_adapter_operation(cursor, "add", resolved_names, command_text)
            return cursor, True
        if action == "pop":
            stack_id, _ = _parse_pop_target_options(remainder)
            try:
                apply_adapter_operation(
                    cursor,
                    "pop",
                    None,
                    command_text,
                    stack_id=stack_id,
                )
            except ValueError as exc:
                print(f"{Colors.ERROR}{exc}{Colors.RESET}")
            return cursor, True
        print(f"{Colors.ERROR}Unknown /a scope action '{action}'.{Colors.RESET}")
        return cursor, True

    elif sub_cmd == "set_active":
        resolved_names = await _resolve_adapter_targets_1based(sub_args, "Select adapter(s) to set as active", allow_multiple=True, cursor=cursor)
        if resolved_names is None:
            print(f"{Colors.SYSTEM}Adapter selection cancelled.{Colors.RESET}")
            return cursor, True
        
        if "__base__" in resolved_names and len(resolved_names) > 1:
            print(f"{Colors.ERROR}The '__base__' adapter cannot be set with other adapters. Use '/a s 0' to disable all adapters.{Colors.RESET}")
            return cursor, True

        apply_adapter_operation(cursor, "set", resolved_names, user_input)
        return cursor, True
    elif sub_cmd == "add_active":
        resolved_names = await _resolve_adapter_targets_1based(sub_args, "Select adapter(s) to add to active", allow_multiple=True, cursor=cursor)
        if resolved_names is None:
            print(f"{Colors.SYSTEM}Adapter add cancelled.{Colors.RESET}")
            return cursor, True
        if not resolved_names:
            print(f"{Colors.ERROR}No valid adapters specified for add. Use '/a a <name|num>, ...' or select interactively.{Colors.RESET}")
            return cursor, True
        if "__base__" in resolved_names:
            print(f"{Colors.ERROR}'__base__' cannot be added. Use '/a s 0' to reset to the base adapter or pop to revert.{Colors.RESET}")
            return cursor, True
        apply_adapter_operation(cursor, "add", resolved_names, user_input)
        return cursor, True
    elif sub_cmd == "pop_active":
        stack_id, _ = _parse_pop_target_options(sub_args.strip())
        try:
            apply_adapter_operation(
                cursor,
                "pop",
                None,
                user_input,
                stack_id=stack_id,
            )
        except ValueError as exc:
            print(f"{Colors.ERROR}{exc}{Colors.RESET}")
        return cursor, True

    elif sub_cmd == "query":
        cursor.log_command(user_input)
        arg = sub_args.strip()
        if not arg:
            # No prompt; show loaded adapters only.
            await query_adapters_from_engine(cursor, None)
            return cursor, True

        # Map numeric selection to loaded adapters only.
        target_name = arg
        try:
            idx = int(arg.lstrip("#"))
            if idx <= 0:
                target_name = "__base__"
            else:
                resp = await call_api("get-loaded-adapters", {})
                adapters = resp.get("data", {}).get("adapters", []) if resp.get("status") == "success" else []
                if 0 < idx <= len(adapters):
                    target_name = adapters[idx - 1].get("name")
        except Exception:
            pass

        if target_name == "__base__":
            print(f"{Colors.SYSTEM}Info: '__base__' is an alias for referencing base model only inference mode. It has no specific configuration to query.{Colors.RESET}")
            return cursor, True

        await query_adapters_from_engine(cursor, target_name)
    return cursor, True

async def _handle_tools_command(args_str: str, cursor: ChatCursor, pt_session: "PromptSession") -> Tuple[ChatCursor, bool]:
    global LAST_ENUMERATED_TOOLS
    toolbox = _active_toolbox()
    toolbox_ref = _active_toolbox_ref()
    if not toolbox:
        print(f"{Colors.ERROR}Error: Toolbox not initialized.{Colors.RESET}")
        return cursor, True
    all_tool_names = _all_tool_names(toolbox)
    tool_wildcards = _tool_wildcard_groups(toolbox)

    stripped_args = args_str.strip()
    if stripped_args in {"?", "help"}:
        _print_tools_cli_help()
        return cursor, True

    parts = args_str.split(" ", 1)
    sub_cmd_full = parts[0].lower()
    sub_args = parts[1] if len(parts) > 1 else ""

    sub_cmd_map = {
        "e": "enum", "n": "new", "m": "modify", "r": "replace", 
        "a": "activate", "d": "deactivate", "p": "print", "u": "unregister",
        "sa": "save", "save": "save", "h": "hide", "hide": "hide", "hidden": "hide",
        "show": "show", "sh": "show",
        "load": "load", "f": "fix",
        "scope": "scope", "sc": "scope",
        "global": "global", "g": "global", "gl": "global", "mode": "global",
    }
    sub_cmd = sub_cmd_map.get(sub_cmd_full) or sub_cmd_map.get(sub_cmd_full[0] if sub_cmd_full else "")

    if not sub_cmd_full.strip():
        sub_cmd = "enum"
    
    # --- Handle commands that can have a tool name as the first argument --- #TBD
    if not sub_cmd:
        print(f"{Colors.ERROR}Unknown tools command: '{sub_cmd_full}'.{Colors.RESET}")
        print("Valid options are: e[num], n[ew], m[odify], r[eplace], p[rint], a[ctivate], d[eactivate], u[nregister], h[idden], f[ix], sa[ve], l[oad], sc[ope], g[lobal].")
        print("Type '/help' for more details.")
        return cursor, True

    if sub_cmd == "enum":
        tools = toolbox.list_tools()
        LAST_ENUMERATED_TOOLS.clear() 
        if not tools:
            print(f"{Colors.TOOL_WARNING}No tools defined. Use '/t new' to add one.{Colors.RESET}")
        else:
            # Adjust name width based on the longest tool name
            max_name_len = max(len(t[0]) for t in tools) if tools else 30
            name_col_width = max(30, max_name_len + 9) # Add padding for guide/modified marker
            print(f"{'Index':<7} {'Name':<{name_col_width}} {'Hidden':<8} {'Active':<8} {'Type':<12} {'Description'}")
            print(f"{'-'*5:<7} {'-'*(name_col_width-2):<{name_col_width}} {'-'*6:<8} {'-'*6:<8} {'-'*10:<12} {'-'*58}")
            for idx, (name, description, tool_type, is_active, is_hidden, is_guide, is_modified) in enumerate(tools):
                desc_trunc = (description[:57] + '...') if len(description) > 57 else description
                modified_marker = "*" if is_modified and not is_guide else " "
                
                if tool_type == "unresolved":
                    type_display = f"{Colors.ERROR}{'Unresolved':<12}{Colors.RESET}"
                else:
                    type_display = f'{tool_type.capitalize():<12}'

                name_display = f"  └─ {name}" if is_guide else f"{modified_marker} {name}"
                hidden_marker = "Yes" if is_hidden else "No"
                active_marker = "Yes" if is_active else "No"
                print(f"  {idx+1:<5} {name_display:<{name_col_width}} {hidden_marker:<8} {active_marker:<8} {type_display:<12} '{desc_trunc}'")
                LAST_ENUMERATED_TOOLS.append(name)
    elif sub_cmd == "new":
        # Call the interactive editor with no tool name
        success, msg = await toolbox.interactive_edit_tool(pt_session, external_tool_handler, tool_name_to_edit=None, search_scope=globals())
        print(msg)
    elif sub_cmd == "modify":
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
            try:
                idx = int(target_arg) - 1
                if 0 <= idx < len(LAST_ENUMERATED_TOOLS):
                    tool_name_to_edit = LAST_ENUMERATED_TOOLS[idx]
                else: print(f"{Colors.ERROR}Invalid tool number: {target_arg}. Use '/t list'.{Colors.RESET}"); return cursor, True
            except ValueError: print(f"Invalid number format: {target_arg}."); return True
        else:
            tool_name_to_edit = target_arg

        if edit_guide:
            # Find the corresponding guide name
            tool_def = toolbox.get_tool(tool_name_to_edit)
            # For user-defined tools, the guide is part of the main tool's definition.
            # The editor handles this by asking to edit the guide.
            # For intrinsics, we need to find the separate guide tool name.
            if tool_def and "guide_definition" in tool_def:
                 tool_name_to_edit = tool_def["guide_definition"]["function"]["name"]
            elif f"{tool_name_to_edit}_guide" in toolbox.intrinsic_tools:
                 tool_name_to_edit = f"{tool_name_to_edit}_guide"
            else:
                print(f"{Colors.ERROR}Error: Could not find a guide function for tool '{tool_name_to_edit}'.{Colors.RESET}")
                return cursor, True

        success, msg = await toolbox.interactive_edit_tool(pt_session, external_tool_handler, tool_name_to_edit=tool_name_to_edit, search_scope=globals())
        print(msg)
    elif sub_cmd == "replace":
        if not sub_args.strip():
            print(f"Usage: /t replace {Colors.CYAN}<name|num>{Colors.RESET}")
            return cursor, True
        
        tool_name_to_update = ""
        if sub_args.isdigit():
            try:
                idx = int(sub_args) - 1
                if 0 <= idx < len(LAST_ENUMERATED_TOOLS):
                    tool_name_to_update = LAST_ENUMERATED_TOOLS[idx]
                else: print(f"{Colors.ERROR}Invalid tool number: {sub_args}. Use '/t list'.{Colors.RESET}"); return cursor, True
            except ValueError: print(f"{Colors.ERROR}Invalid number format: {sub_args}.{Colors.RESET}"); return cursor, True
        else:
            tool_name_to_update = sub_args.strip()

        print(f"Enter the full JSON definition for '{tool_name_to_update}'. Type END_JSON on a new line to finish.")
        json_lines = []
        while True:
            line = await pt_session.prompt_async("")
            if line.strip() == "END_JSON": break
            json_lines.append(line)
        json_string = "\n".join(json_lines)

        if not json_string:
            print("Update cancelled.")
            return cursor, True

        success, msg = toolbox.update_tool_from_json_string(tool_name_to_update, json_string, external_handler=external_tool_handler, search_scope=globals())
        print(msg)
    elif sub_cmd == "print":
        if not sub_args.strip():
            print(f"Usage: /t print {Colors.CYAN}<name|num>{Colors.RESET}")
            return cursor, True
        
        tool_name_to_print = ""
        if sub_args.isdigit():
            try:
                idx = int(sub_args) - 1
                if 0 <= idx < len(LAST_ENUMERATED_TOOLS):
                    tool_name_to_print = LAST_ENUMERATED_TOOLS[idx]
                else:
                    print(f"{Colors.ERROR}Invalid tool number: {sub_args}. Use '/t list'.{Colors.RESET}")
                    return cursor, True
            except ValueError:
                print(f"{Colors.ERROR}Invalid number format: {sub_args}.{Colors.RESET}")
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
    elif sub_cmd in ["activate", "deactivate"]:
        if not sub_args.strip(): 
            print(f"Usage: /t {sub_cmd} {Colors.CYAN}<name|num|*|*i|*c|*e,...>{Colors.RESET}")
            return cursor, True
        tool_names = await _parse_cli_targets(
            sub_args.strip(),
            LAST_ENUMERATED_TOOLS,
            allow_wildcard=True,
            wildcard_values=all_tool_names,
            wildcard_groups=tool_wildcards,
        )
        if not tool_names:
            print(f"{Colors.ERROR}No valid tools specified.{Colors.RESET}")
            return cursor, True

        success, msg = await asyncio.to_thread(toolbox.activate_tool if sub_cmd == "activate" else toolbox.deactivate_tool, tool_names)
        print(msg)
    elif sub_cmd == "save":
        save_path_str = sub_args.strip()
        if save_path_str:
            target_path = Path(save_path_str).expanduser().resolve()
        else:
            tools_path = (current_config or {}).get("tools_config_path")
            if not tools_path:
                print(f"{Colors.ERROR}No tools config path configured.{Colors.RESET}")
                return cursor, True
            target_path = Path(tools_path) # type: ignore
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            toolbox_state = toolbox.to_dict()
            with open(target_path, "w") as f:
                json.dump(toolbox_state, f, indent=2)
            print(f"Toolbox state saved to {target_path}")
        except Exception as e:
            print(f"Error saving toolbox state: {e}")
    elif sub_cmd == "load":
        load_arg = sub_args.strip()
        if not load_arg:
            print(f"Usage: /t load {Colors.CYAN}<file_path|json_string>{Colors.RESET}")
            return cursor, True
        
        try:
            if load_arg.strip().startswith('{'): # Assume it's a JSON string
                data = json.loads(load_arg)
                toolbox.from_dict(data, search_scope=globals(), external_handler=external_tool_handler)
                print("Toolbox state loaded from JSON string.")
            else: # Assume it's a file path
                with open(Path(load_arg).expanduser().resolve(), "r") as f:
                    toolbox.from_dict(json.load(f), search_scope=globals(), external_handler=external_tool_handler)
                print(f"Toolbox state loaded from {load_arg}")
        except Exception as e:
            print(f"Error loading toolbox state: {e}")
    elif sub_cmd in ["hide", "show"]:
        if not sub_args.strip():
            print(f"Usage: /t {sub_cmd} {Colors.CYAN}<name|num|*|*i|*c|*e,...>{Colors.RESET}")
            return cursor, True
        tool_names = await _parse_cli_targets(
            sub_args.strip(),
            LAST_ENUMERATED_TOOLS,
            allow_wildcard=True,
            wildcard_values=all_tool_names,
            wildcard_groups=tool_wildcards,
        )
        if not tool_names:
            print(f"{Colors.ERROR}No valid tools specified.{Colors.RESET}")
            return cursor, True
        hide_flag = (sub_cmd == "hide")
        success, msg = await asyncio.to_thread(toolbox.set_hidden, tool_names, hide_flag)
        print(msg)
    elif sub_cmd == "unregister":
        if not sub_args.strip():
            print(f"Usage: /t unregister {Colors.CYAN}<name|num|*|*i|*c|*e,...>{Colors.RESET}")
            return cursor, True

        # This command now correctly handles single or multiple targets.
        tool_names = await _parse_cli_targets(
            sub_args.strip(),
            LAST_ENUMERATED_TOOLS,
            allow_wildcard=True,
            wildcard_values=all_tool_names,
            wildcard_groups=tool_wildcards,
        )
        if not tool_names:
            print("No valid tools specified for unregister.")
            return cursor, True
        success, msg = await asyncio.to_thread(toolbox.delete_tool, tool_names)
        print(msg) # This will now print the summary message for all deletions
    elif sub_cmd == "fix":
        if not sub_args.strip():
            print(f"Usage: /t fix {Colors.CYAN}<name|num>{Colors.RESET}")
            return cursor, True
        tool_names = await _parse_cli_targets(sub_args.strip(), LAST_ENUMERATED_TOOLS)
        if not tool_names:
            print(f"{Colors.ERROR}No valid tool specified.{Colors.RESET}")
            return cursor, True
        
        tool_to_fix = tool_names[0]
        print(f"\nHow do you want to fix the unresolved tool '{Colors.CYAN}{tool_to_fix}{Colors.RESET}'?")
        print(f"  1. Try to re-link as a {Colors.BOLD}'callable'{Colors.RESET} Python function, falling back to 'external' if not found. (Default)") 
        print(f"  2. Try to re-link as a {Colors.BOLD}'callable'{Colors.RESET} Python function {Colors.ERROR}only{Colors.RESET}. The command will fail if the function is not found.")
        print(f"  3. Convert to an {Colors.BOLD}'external'{Colors.RESET} tool, using the console input handler.")
        choice = (await pt_session.prompt_async("Enter choice (1, 2, or 3) [1]: ")).strip()

        if choice == '3':
            # Fix as external
            success, msg = toolbox.resolve_tool_link(tool_to_fix, search_scope=None, external_handler=external_tool_handler)
        elif choice == '2':
            # Fix as callable only (no fallback)
            success, msg = toolbox.resolve_tool_link(tool_to_fix, search_scope=globals(), external_handler=None)
        elif choice in ['1', '']:
            # Default: Fix as callable with external as a fallback
            success, msg = toolbox.resolve_tool_link(tool_to_fix, search_scope=globals(), external_handler=external_tool_handler) 
        else:
            success, msg = False, "Invalid choice. Fix cancelled."

        print(msg) # Print the result from the toolbox method
    elif sub_cmd == "global":
        arg = sub_args.strip().lower()
        if arg in {"?", "help"} or not arg:
            print(f"Usage: /t g[lobal] {Colors.CYAN}<a|s|d>{Colors.RESET} (advertised|silent|disabled)")
            return cursor, True
        mode_alias = {"a": "advertised", "adv": "advertised", "advertised": "advertised",
                      "s": "silent", "sil": "silent", "silent": "silent",
                      "d": "disabled", "dis": "disabled", "disabled": "disabled"}
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
    elif sub_cmd == "scope":
        scope_args = sub_args.strip()
        if scope_args in {"?", "help"}:
            _print_tools_cli_help()
            return cursor, True
        if not scope_args:
            action = "show"
            remainder = ""
        else:
            action_token, _, remainder = scope_args.partition(" ")
            action_token = action_token.lower()
            remainder = remainder.strip()
            scope_action_map = {
                "set": "set", "s": "set",
                "add": "add", "a": "add",
                "pop": "pop", "p": "pop",
                "reset": "reset", "r": "reset",
                "show": "show", "status": "show", "": "show",
            }
            action = scope_action_map.get(action_token)
            if not action and "=" in action_token:
                # No verb was provided; default to 'set' when the first token looks like an option.
                action = "set"
                remainder = scope_args
            elif not action:
                action = "show"
                remainder = scope_args
        command_text = f"/t scope {scope_args}" if scope_args else "/t scope"
        if action == "show":
            target_cursor = cursor
            gen_id_arg = remainder.strip()
            if gen_id_arg:
                try:
                    target_cursor = cursor.cursor_for_gen_id(gen_id_arg)
                except KeyError:
                    print(f"{Colors.ERROR}Turn with gen_id '{gen_id_arg}' not found.{Colors.RESET}")
                    return cursor, True
                except ValueError as err:
                    print(f"{Colors.ERROR}{err}{Colors.RESET}")
                    return cursor, True
            tools_view = target_cursor.get_tools_view()
            if not tools_view:
                print(f"{Colors.SYSTEM}No active tools context available.{Colors.RESET}")
            else:
                entries = _collect_tools_scope_entries(target_cursor)
                _print_tools_scope_summary(target_cursor, tools_view, entries)
        elif action in {"set", "add"}:
            if not remainder or remainder in {"?", "help"}:
                verb = "set" if action == "set" else "add"
                hint = " (use mode=* to reset to defaults)" if action == "set" else ""
                print(f"Usage: /t scope {verb} m[ode]=... a[dvertised]=... s[ilent]=... d[isabled]=...{hint}")
                return cursor, True
            scope_obj = _parse_scope_cli_args(remainder)
            if not scope_obj:
                print(f"{Colors.ERROR}No valid scope options provided. Example: mode=silent advertise=search{Colors.RESET}")
                return cursor, True
            normalized_scope, warnings = _normalize_scope_tool_names(scope_obj, toolbox)
            for warning in warnings:
                print(warning)
            if normalized_scope.is_noop():
                print(f"{Colors.TOOL_WARNING}Scope has no valid settings; command ignored.{Colors.RESET}")
                return cursor, True
            cursor.apply_tools_scope(action, normalized_scope, command_text=command_text)
        elif action == "pop":
            stack_id, _ = _parse_pop_target_options(remainder)
            try:
                cursor.apply_tools_scope(
                    "pop",
                    None,
                    command_text=command_text,
                    stack_id=stack_id,
                )
            except ValueError as exc:
                print(f"{Colors.ERROR}{exc}{Colors.RESET}")
                return cursor, True
        elif action == "reset":
            print(f"{Colors.SYSTEM}Use '/t scope set mode=*' to reset to the default tools mode.{Colors.RESET}")
            return cursor, True
        else:
            print(f"Usage: /t scope s[et]|a[dd]|p[op] [options]")
            print(f"{Colors.SYSTEM}Tip: Use '/t scope set mode=*' to reset to defaults.{Colors.RESET}")
            return cursor, True
        if action in {"set", "add", "pop"}:
            tools_view = cursor.get_tools_view()
            if tools_view:
                entries = _collect_tools_scope_entries(cursor)
                _print_tools_scope_summary(cursor, tools_view, entries)

    return cursor, True

async def async_input(prompt: str) -> str:
    """Asynchronously read input from the console."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,  # Use the default thread pool
        lambda: input(prompt)
    )

async def external_tool_handler(**kwargs: Any) -> str:
    """
    This async function serves as the default implementation for tools created
    interactively that are not bound to an existing Python function.
    It prompts the user in the console to provide the result for the tool call.
    It receives a kwargs bundle containing 'toolbox', 'tool', and the tool's arguments.
    """
    # Pop the context arguments from kwargs; the rest are the tool's arguments.
    toolbox = kwargs.pop("toolbox", None)
    tool = kwargs.pop("tool", None)

    if not tool or not isinstance(tool, dict) or not toolbox:
        return "Error: Interactive handler was called without a valid 'tool' definition object."

    tool_name = tool.get("function", {}).get("name", "unknown_tool")
    tool_args_str = json.dumps(kwargs) # kwargs are the arguments for the tool call

    print(f"\n{Colors.TOOL}--- Tool Call Requires Your Input ---{Colors.RESET}")
    print(f"  {Colors.TOOL}Tool:{Colors.RESET} {tool_name}")
    print(f"  {Colors.TOOL}Arguments:{Colors.RESET} {tool_args_str}")
    print(f"{Colors.TOOL}-------------------------------------------------{Colors.RESET}")

    user_content_input = await async_input(f"Enter result for {tool_name}: ")
    return user_content_input


async def _tool_execution_action_handler(
    execute_stage: str,
    serial_execution: bool,
    final_response_items: List[InferenceResponse],      # more than one for batch responses
    tool_call: Optional[ToolCall] = None,   # call, only during execution
    tool_call_block: Optional[ToolCallBlock] = None, # The block this call belongs to
    current_response_item: Optional["InferenceResponse"] = None, # The item this call belongs to
    **kwargs: Any
    )-> bool| None:
    """
    An action handler passed to `execute_request_tools` to print execution progress.
    It's called at different stages of the execution process.
    During parsing stage it returns either confirmation or override for serial_execution.
    """

    if execute_stage == 'calls_parsed':
        # --- Stage 1: After all raw blocks have been parsed ---
        # In batch mode, the "round" is less clear. Let's use a generic header.
        print(f"\n{Colors.TOOL}--- Tool Execution Phase ---{Colors.RESET}")

        # Check for and print any parsing errors found in the blocks.
        is_batch = len(final_response_items) > 1
        for response_item in final_response_items:
            if response_item.tool_blocks and len(response_item.tool_blocks) > 0:
                for block in response_item.tool_blocks or  []:
                    prompt_index_for_log = block.prompt_index if block.prompt_index is not None else "N/A"
                    batch_context = f" (from prompt {prompt_index_for_log + 1})" if is_batch and isinstance(prompt_index_for_log, int) else ""
                    # The prompt_index on the block should be set by the caller.
                    # Let's double-check.
                    if block.parse_errors:
                        print(f"{Colors.TOOL_WARNING}Warning: A tool block had parsing errors{batch_context}:{Colors.RESET}")
                        print(f"{Colors.DIM}  Block: {block.raw_block[:100]}...{Colors.RESET}")
                        for err in block.parse_errors:
                            # Application logic: if a block fails to parse, the definitive action is to keep its raw content.
                            # This overrides any other default actions.
                            block.action_block = [ToolCall.KeepRaw]
                            print(f"{Colors.DIM}  - {err}{Colors.RESET}")
                    elif not block.calls:
                        # If there are no parsing errors but the block is empty,
                        # it means the model generated an empty tool block (e.g., "[TOOL_CALLS][]").
                        # We should strip it.
                        block.action_block = [ToolCall.KeepRaw]
                        print(f"{Colors.TOOL_WARNING}Warning: A tool block is empty {batch_context}{Colors.RESET}")
            
                    for call in block.calls or []:
                        if call.parse_errors:
                            if ToolCall.KeepRaw not in call.action:
                                call.action.append(ToolCall.KeepRaw)
                            print(f"{Colors.TOOL_WARNING}Warning: A tool call within a block had parsing errors{batch_context}:{Colors.RESET}")
                            print(f"{Colors.DIM}  Call (raw): {call.raw[:100] if call.raw else 'N/A'}...{Colors.RESET}")
                            for err in call.parse_errors:
                                print(f"{Colors.DIM}  - {err}{Colors.RESET}")
        # If needed, this can override original parallel execution intent
        return serial_execution

    if execute_stage == 'call_starting' and tool_call:
        tool_name = tool_call.name or "unnamed_tool"
        display_args = tool_call.arguments
        tool_args_note = ""
        recovered_preview = ""
        if isinstance(tool_call.arguments, dict) and "tool_args_issue" in tool_call.arguments:
            raw_issue = tool_call.arguments.get("tool_args_issue")
            raw_len = None
            if isinstance(raw_issue, dict):
                raw_val = raw_issue.get("_non_parsed") or raw_issue.get("_string_value")
                if isinstance(raw_val, str):
                    raw_len = len(raw_val)
                    expr_marker = '"expr": "'
                    start_index = raw_val.rfind(expr_marker)
                    if start_index != -1:
                        start_index += len(expr_marker)
                        esc = False
                        end_index = -1
                        for i in range(start_index, len(raw_val)):
                            ch = raw_val[i]
                            if esc:
                                esc = False
                                continue
                            if ch == "\\":
                                esc = True
                                continue
                            if ch == '"':
                                end_index = i
                                break
                        if end_index > start_index:
                            recovered_preview = raw_val[start_index:end_index]
                        else:
                            recovered_preview = raw_val[start_index:]
            if raw_len is not None:
                tool_args_note = f" (tool_args_issue len={raw_len})"
            else:
                tool_args_note = " (tool_args_issue present)"
        if isinstance(display_args, dict):
            display_args = {k: v for k, v in display_args.items() if not k.startswith('_') and k != 'tool_args_issue'}
        tool_args_str = json.dumps(display_args) if isinstance(display_args, dict) else str(display_args)
        if recovered_preview:
            for marker in ["</s>", "<tool_call", "</tool_call>", "[TOOL_CALLS", "[/TOOL_CALLS]"]:
                idx = recovered_preview.find(marker)
                if idx != -1:
                    recovered_preview = recovered_preview[:idx]
            recovered_preview = recovered_preview.replace("\n", " ").strip()
            if len(recovered_preview) > 80:
                recovered_preview = recovered_preview[:77].rstrip() + "..."
        
        # This logic is now part of the 'starting' phase, before execution.
        # We can check if the tool is executable to decide which header to print.
        toolbox = _active_toolbox()
        prompt_index = tool_call_block.prompt_index if tool_call_block and tool_call_block.prompt_index is not None else 0
        
        # Build a context string for logging, including batch and block index if relevant.
        context_parts = []
        if len(final_response_items) > 1:
            context_parts.append(f"prompt {prompt_index + 1}/{len(final_response_items)}")
        if not tool_call_block is None and not current_response_item is None and len(current_response_item.tool_blocks or []) > 1:
            block_idx = current_response_item.tool_blocks.index(tool_call_block)
            context_parts.append(f"block {block_idx + 1}/{len(current_response_item.tool_blocks)}")
        batch_context = f" ({', '.join(context_parts)})" if context_parts else ""

        tools_view: Optional[ToolsView] = kwargs.get("tools_view")

        if toolbox and toolbox.is_executable(tool_name, tools_view=tools_view):
            print(f"\n  {Colors.TOOL}--- Executing Tool: {tool_name}{batch_context} ---{Colors.RESET}")
            print(f"    {Colors.TOOL}Tool:     {Colors.RESET} {tool_name}")
            print(f"    {Colors.TOOL_ARGS}Arguments:{Colors.RESET} {Colors.DIM}{tool_args_str}{tool_args_note}{Colors.RESET}")
            if recovered_preview:
                print(f"    {Colors.TOOL_ARGS}Recovered expr preview:{Colors.RESET} {Colors.DIM}{recovered_preview}{Colors.RESET}")
            print(f"  {Colors.TOOL}--------------------------------{Colors.RESET}")
        else:
            # This case handles when the model calls a tool that is not registered or active.
            # The actual error message will be generated by `toolbox.execute` and printed in the 'call_finished' stage.
            print(f"\n  {Colors.TOOL_WARNING}--- Tool Call Not Found or access denied {batch_context} ---{Colors.RESET}")
            print(f"    {Colors.TOOL_WARNING}Tool:     {Colors.RESET} {tool_name}")
            print(f"    {Colors.TOOL_ARGS}Arguments:{Colors.RESET} {Colors.DIM}{tool_args_str}{tool_args_note}{Colors.RESET}")
            if recovered_preview:
                print(f"    {Colors.TOOL_ARGS}Recovered expr preview:{Colors.RESET} {Colors.DIM}{recovered_preview}{Colors.RESET}")
            print(f"  {Colors.TOOL_WARNING}-----------------------{Colors.RESET}")

    elif execute_stage == 'call_finished' and tool_call:
        if tool_call.error:
            print(f"    {Colors.ERROR}Result:   {Colors.RESET} {Colors.ERROR}{tool_call.error}{Colors.RESET}")
        elif tool_call.result is not None:
            # Pretty print JSON results
            result_str = str(tool_call.result)
            try:
                # Try to load and dump to pretty-print if it's a JSON string
                parsed_json = json.loads(result_str)
                pretty_result = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                print(f"    {Colors.TOOL}Result:{Colors.RESET}")
                for line in pretty_result.splitlines():
                    print(f"      {line}")
            except (json.JSONDecodeError, TypeError):
                # Fallback for non-JSON string results
                print(f"    {Colors.TOOL}Result:   {Colors.RESET} {result_str}")
        else:
            # This case can happen if the tool call was malformed from the start (e.g., no name)
            if not tool_call.error:
                print(f"    {Colors.TOOL_WARNING}Result:   {Colors.RESET} (No result or error recorded)")

    elif execute_stage == 'all_finished':
        # This stage is useful for final logging or summary, but we don't print anything here for now.
        pass

    return None

async def _run_batch_request(
    request_payload: Dict[str, Any],
    batch_forks: List[Dict[str, Any]],
    batch_index: int,
    total_batches_in_round: int,
    batch_round: int,
    llm_name: str,
    parent_cursor: ChatCursor,
    console_lock: asyncio.Lock,
    total_prompts_for_batch: int,
) -> Dict[str, Any]:
    """
    Shared helper: submit a batch payload, stream responses, and return fan-out data.
    Used by both live chat batch handling and replay batch handling.
    """
    try:
        api_response = await call_api("run-inference", request_payload)
        was_canceled_by_user = False

        if not (isinstance(api_response, dict) and "stream" in api_response):
            error_msg = api_response.get("message", "Unknown API error during batch processing")
            async with console_lock:
                print(f"\n{Colors.ERROR}API error processing batch {batch_index}: {error_msg}{Colors.RESET}")
            return {
                "batch_index": batch_index,
                "metrics": {},
                "final_chunks": {
                    i: {
                        "chunkType": ChunkType.ERROR.value,
                        "error": error_msg,
                        "is_final_chunk": True,
                    }
                    for i in range(len(batch_forks))
                },
            }

        api_iterator = api_response.get("stream") if isinstance(api_response, dict) else None
        final_chunks_for_items: Dict[int, Dict[str, Any]] = {}
        final_metrics: Dict[str, Any] = {}
        stream_was_properly_closed = False

        if api_iterator:
            per_prompt_contexts: Dict[int, StreamDisplayContext] = {}
            for idx_in_batch, fork in enumerate(batch_forks):
                original_idx = _fork_entry_original_index(fork, idx_in_batch)
                per_prompt_contexts[idx_in_batch] = StreamDisplayContext(
                    prompt_index=idx_in_batch,
                    original_prompt_index=original_idx,
                    total_original_prompts=total_prompts_for_batch,
                    batch_index=batch_index,
                    total_batches=total_batches_in_round,
                    show_batch_header=True,
                    show_prompt_echo=True,
                    show_response_banner=True,
                    show_turn_counter=False,
                    response_label="Response",
                )
            display_plan = StreamDisplayPlan(
                per_prompt=per_prompt_contexts,
                default_context=StreamDisplayContext(
                    show_batch_header=True,
                    show_prompt_echo=True,
                    show_response_banner=True,
                    show_turn_counter=False,
                    response_label="Response",
                ),
            )
            try:
                stream_result = await _consume_inference_stream(
                    api_iterator,
                    display_plan=display_plan,
                    llm_name=llm_name,
                    adapters_for_display=[],
                    cursor=parent_cursor,
                    console_lock=console_lock,
                )
            finally:
                await api_iterator.aclose()

            final_chunks_for_items = stream_result.final_chunks
            final_metrics = stream_result.stream_metrics or {}
            was_canceled_by_user = stream_result.was_canceled
            stream_was_properly_closed = stream_result.stream_completed or stream_result.was_canceled
        else:
            final_chunks_for_items = {}
            final_metrics = {}

        if was_canceled_by_user:
            for i in range(len(batch_forks)):
                chunk = final_chunks_for_items.get(i, {}) or {}
                chunk.setdefault("chunkType", ChunkType.STREAMING_ENDED.value)
                chunk.setdefault("is_final_chunk", True)
                chunk.setdefault("prompt_index", i)
                chunk["was_canceled"] = True
                final_chunks_for_items[i] = chunk

        if not was_canceled_by_user and not stream_was_properly_closed:
            async with console_lock:
                print(f"\n{Colors.ERROR}Error: The engine stream for batch {batch_index} terminated unexpectedly. Check engine logs for a full traceback.{Colors.RESET}")
            for i in range(len(batch_forks)):
                if i not in final_chunks_for_items:
                    final_chunks_for_items[i] = {
                        "chunkType": ChunkType.ERROR.value,
                        "error": "Engine stream ended unexpectedly.",
                        "is_final_chunk": True,
                        "prompt_index": i,
                    }
        return {
            "batch_index": batch_index,
            "metrics": final_metrics,
            "final_chunks": final_chunks_for_items,
            "text_by_prompt": stream_result.text_by_prompt if api_iterator else {},
            "request_payload": request_payload,
            "prompt_index_map": [_fork_entry_original_index(fork, idx) for idx, fork in enumerate(batch_forks)],
        }

    except Exception as e:
        was_canceled_by_user = False
        if isinstance(e, asyncio.CancelledError):
            was_canceled_by_user = True
        async with console_lock:
            print(f"\n{Colors.ERROR}An unexpected error occurred processing batch {batch_index}: {e}{Colors.RESET}")
        return {
            "batch_index": batch_index,
            "metrics": {},
            "final_chunks": {
                i: {
                    "chunkType": ChunkType.ERROR.value,
                    "error": str(e),
                    "is_final_chunk": True,
                    "was_canceled": was_canceled_by_user,
                }
                for i in range(len(batch_forks))
            },
            "text_by_prompt": {},
        }

async def _apply_batch_results_to_children(
    prepared_forks: List[Dict[str, Any]],
    final_by_prompt: Dict[int, InferenceResponse],
    final_chunks: Dict[int, Dict[str, Any]],
    text_by_prompt: Dict[int, str],
    source_to_dest_cursor_map: Optional[Dict[Turn, Dict[str, Any]]] = None,
    *,
    tools_view: Optional[ToolsView] = None,
    allow_auto_retry: bool = False,
    next_active_forks: Optional[List[Dict[str, Any]]] = None,
    batch_context_cursor: Optional[ChatCursor] = None,
    replay_mode: bool = False,
) -> Dict[Turn, ChatCursor]:
    """
    Apply assistant/tool results and metrics to each child cursor after a batch replay or live batch chunk.
    If allow_auto_retry is True, truncated responses queue continuation turns into next_active_forks.
    """
    def _replay_debug_enabled(cur: Optional[ChatCursor]) -> bool:
        try:
            ctx = cur.context if cur else None
            scope = _scope_for_cursor(cur)
            bag_dict = _resolve_bag_dict(scope, ctx)
            return bool(bag_dict and bag_dict.get("replay_debug"))
        except Exception:
            return False

    def _sync_fork_cursor(
        fork_obj: Optional[ChatForks],
        cursor_idx: Optional[int],
        cursor_obj: Optional[ChatCursor],
    ) -> None:
        if not fork_obj or cursor_idx is None or cursor_obj is None:
            return
        try:
            fork_obj.update_active_cursor(cursor_idx, cursor_obj)
        except Exception:
            pass

    def _record_source_mapping(
        source_turn: Optional[Turn],
        dest_cursor: Optional[ChatCursor] = None,
    ) -> None:
        if not source_turn or source_to_dest_cursor_map is None:
            return
        if source_turn in source_to_dest_cursor_map:
            return
        if not dest_cursor or not getattr(dest_cursor, "current_turn", None):
            return
        source_to_dest_cursor_map[source_turn] = _make_replay_mapping_entry(source_turn, dest_cursor)

    child_cursor_map: Dict[Turn, ChatCursor] = {}
    batch_entries: List[Dict[str, Any]] = []

    # First pass: collect per-child metadata and normalize tool blocks.
    for fork in prepared_forks:
        child_cursor = _fork_entry_cursor(fork)
        if not child_cursor:
            continue
        source_turn = fork.get("source_turn") if isinstance(fork, dict) else None
        child_turn = source_turn if source_turn else child_cursor.current_turn
        fork_obj: Optional[ChatForks] = fork.get("fork")
        cursor_idx = fork.get("cursor_idx")
        if fork_obj:
            fork_obj.cursor_meta = fork_obj.cursor_meta or {}
            if not fork_obj.prompt_indices:
                fork_obj.prompt_indices = list(range(len(fork_obj.cursors)))
        if cursor_idx is None:
            if _replay_debug_enabled(child_cursor):
                print(f"{Colors.TOOL_WARNING}DEBUG: missing cursor_idx for fork entry; skipping.{Colors.RESET}")
            continue
        original_idx = fork_obj.prompt_indices[cursor_idx] if fork_obj and cursor_idx < len(fork_obj.prompt_indices) else fork.get("original_index", _fork_entry_original_index(fork))
        response_obj = final_by_prompt.get(original_idx)
        chunk_meta = final_chunks.get(original_idx, {}) if isinstance(final_chunks, dict) else {}
        text = text_by_prompt.get(original_idx) or ""
        tool_blocks = None
        if response_obj and response_obj.tool_blocks:
            tool_blocks = response_obj.tool_blocks
        elif chunk_meta.get("tool_blocks"):
            tool_blocks = chunk_meta.get("tool_blocks")
        if tool_blocks:
            normalized_blocks: List[ToolCallBlock] = []
            for tb in tool_blocks:
                if isinstance(tb, ToolCallBlock):
                    normalized_blocks.append(tb)
                elif isinstance(tb, dict):
                    normalized_blocks.append(ToolCallBlock.from_dict(tb))
            tool_blocks = normalized_blocks
            if response_obj is not None and tool_blocks and not response_obj.tool_blocks:
                response_obj.tool_blocks = tool_blocks

        was_canceled = bool(getattr(response_obj, "was_canceled", False) or chunk_meta.get("was_canceled"))
        was_truncated = bool(getattr(response_obj, "was_truncated", False) or chunk_meta.get("was_truncated"))
        is_error = (
            getattr(response_obj, "chunkType", None) == ChunkType.ERROR
            or (chunk_meta.get("chunkType") == ChunkType.ERROR.value if chunk_meta else False)
            or bool(chunk_meta.get("error") or chunk_meta.get("error_message") or getattr(response_obj, "error", None))
        )
        suppress_tools = was_truncated or was_canceled or is_error
        source_has_assistant = False
        try:
            if source_turn and isinstance(getattr(source_turn, "data", None), dict):
                source_has_assistant = bool(source_turn.data.get("assistant"))
        except Exception:
            source_has_assistant = False
        if tool_blocks and was_truncated:
            text, tool_blocks = _suppress_truncated_tool_blocks(
                text,
                tool_blocks,
                cursor=child_cursor,
            )
            if response_obj:
                try:
                    response_obj.tool_blocks = None
                except Exception:
                    pass
                try:
                    response_obj.response_text = text
                except Exception:
                    pass
        elif tool_blocks and is_error:
            for block in tool_blocks:
                block.action_block.extend([ToolCall.KeepRaw, ToolCall.Ignore])

        if _replay_debug_enabled(child_cursor):
            try:
                tool_count = len(tool_blocks or [])
                print(f"{Colors.SYSTEM}DEBUG: batch apply idx={original_idx} turn={child_cursor.display_id()} text_len={len(text)} tools={tool_count} suppress={suppress_tools}{Colors.RESET}")
            except Exception:
                pass

        batch_entries.append({
            "original_idx": original_idx,
            "child_cursor": child_cursor,
            "child_turn": child_turn,
            "source_turn": source_turn,
            "fork_obj": fork_obj,
            "response_obj": response_obj,
            "chunk_meta": chunk_meta,
            "text": text,
            "tool_blocks": tool_blocks,
            "was_canceled": was_canceled,
            "was_truncated": was_truncated,
            "is_error": is_error,
            "suppress_tools": suppress_tools,
            "cursor_idx": cursor_idx,
        })

    # Second pass: execute all eligible tool blocks together using the batch hub context.
    response_items: List[InferenceResponse] = [entry["response_obj"] for entry in batch_entries if entry["response_obj"]]
    any_tool_blocks_present = any(entry.get("tool_blocks") for entry in batch_entries)

    if any_tool_blocks_present and response_items:
        active_toolbox = _active_toolbox()
        if active_toolbox:
            context_cursor = batch_context_cursor
            if not context_cursor:
                for entry in batch_entries:
                    fork_obj = entry.get("fork_obj")
                    if fork_obj and fork_obj.batch_hub:
                        context_cursor = fork_obj.batch_hub
                        break
            if not context_cursor and prepared_forks:
                context_cursor = _fork_entry_cursor(prepared_forks[0])
            parser_profile = None
            if context_cursor:
                parser_profile = context_cursor.parser_profile or None  # type: ignore
                if not parser_profile and context_cursor.chat_session:
                    parser_profile = context_cursor.chat_session.parser_profile  # type: ignore
            if not parser_profile:
                parser_profile = _engine_parser_profile()
            tool_retries_max, tool_retries_left = _tool_retry_counters(context_cursor)
            try:
                await active_toolbox.execute_request_tools(
                    parser_profile=parser_profile,
                    final_response_items=response_items,
                    action_handler=_tool_execution_action_handler,
                    serial_execution=True,
                    tools_view=tools_view or (context_cursor.get_tools_view() if context_cursor else None),
                    context=context_cursor,
                    tool_retries_max=tool_retries_max,
                    tool_retries_left=tool_retries_left,
                )
            except Exception as exc:
                print(f"{Colors.TOOL_WARNING}Replay: batch tool execution skipped due to error: {exc}{Colors.RESET}")
        else:
            print(f"{Colors.TOOL_WARNING}Replay: toolbox unavailable; tool blocks skipped for batch round.{Colors.RESET}")

    # Third pass: apply assistant content, tool results, and auto-retry decisions per child.
    for entry in batch_entries:
        child_cursor = _fork_entry_cursor(entry) or entry["child_cursor"]
        child_turn = entry["child_turn"]
        source_turn = entry.get("source_turn")
        fork_obj = entry.get("fork_obj")
        cursor_idx = entry.get("cursor_idx")
        original_idx = entry["original_idx"]
        response_obj = entry["response_obj"]
        chunk_meta = entry["chunk_meta"]
        text = entry["text"]
        tool_blocks = entry["tool_blocks"]
        was_canceled = entry["was_canceled"]
        was_truncated = entry["was_truncated"]
        is_error = entry["is_error"]
        suppress_tools = entry["suppress_tools"]
        cursor_meta = fork_obj.cursor_meta.setdefault(child_cursor.id, {}) if fork_obj and child_cursor else {}
        if fork_obj and cursor_idx is None:
            if _replay_debug_enabled(child_cursor):
                print(f"{Colors.TOOL_WARNING}DEBUG: missing cursor_idx for fork entry; skipping.{Colors.RESET}")
            continue

        pending_tool_followup = False

        # Avoid writing a second assistant message onto a turn that already has one.
        has_assistant = bool(child_cursor.current_turn and (child_cursor.current_turn.data or {}).get("assistant"))
        if has_assistant and not tool_blocks:
            continue

        if text or tool_blocks or (was_canceled and not has_assistant):
            if child_cursor.current_turn and child_cursor.current_turn.IsPlaceholderLike:
                # TBD-DESIGN: batch child cursors should already target a user/tool turn;
                # a placeholder here means cursor tracking is inconsistent.
                raise ValueError("Batch apply cannot add assistant to a placeholder turn.")
            try:
                if child_cursor.current_turn and child_cursor.current_turn.IsEmpty:
                    raise ValueError("Replay target turn is a placeholder without user/tool input.")
                child_cursor.add_assistant(
                    text or "",
                    tool_blocks=tool_blocks,
                    archived=is_error or was_canceled,
                    was_truncated=was_truncated,
                    was_canceled=was_canceled,
                )
                if _replay_debug_enabled(child_cursor):
                    try:
                        label = child_cursor.display_id() if child_cursor else "None"
                        print(f"{Colors.SYSTEM}DEBUG: added assistant on {label} (tools={len(tool_blocks or [])}).{Colors.RESET}")
                    except Exception:
                        pass
            except Exception as exc:
                try:
                    head = child_cursor.current_turn if child_cursor else None
                    head_id = head.gen_id_or_parent if head else "None"
                    head_type = getattr(head, "turn_type", None)
                    head_empty = bool(getattr(head, "IsEmpty", False))
                    head_has_response = bool(getattr(head, "HasResponse", False))
                    head_user = bool((head.data or {}).get("user")) if head else False
                    head_tools = bool((head.data or {}).get("tool_results")) if head else False
                    fork_head = None
                    if fork_obj and cursor_idx is not None and cursor_idx < len(getattr(fork_obj, "cursors", []) or []):
                        fork_head = fork_obj.cursors[cursor_idx].current_turn
                    fork_id = fork_head.gen_id_or_parent if fork_head else "None"
                    fork_ids = []
                    if fork_obj:
                        for idx, cur in enumerate(getattr(fork_obj, "cursors", []) or []):
                            try:
                                fork_ids.append(f"{idx}:{cur.current_turn.gen_id_or_parent}")
                            except Exception:
                                fork_ids.append(f"{idx}:<err>")
                    tool_blocks_len = len(tool_blocks or [])
                    leaf_id = None
                    if child_turn:
                        try:
                            leaf = child_cursor.session.get_last_turn_on_branch(child_turn)
                            leaf_id = leaf.gen_id_or_parent if leaf else None
                        except Exception:
                            leaf_id = "<err>"
                        if _replay_debug_enabled(child_cursor):
                            print(
                                f"{Colors.TOOL_WARNING}Replay: assistant target unresolved for child "
                                f"{child_cursor.display_id() if child_cursor else 'None'}; skipping "
                                f"(head={head_id} empty={head_empty} has_resp={head_has_response}).{Colors.RESET}"
                            )
                    has_results = _tool_blocks_have_results(tool_blocks) if tool_blocks else False
                    print(
                        f"{Colors.TOOL_WARNING}Batch: debug add_assistant failed: "
                        f"child={child_cursor.display_id() if child_cursor else 'None'} "
                        f"head={head_id} type={head_type} empty={head_empty} has_resp={head_has_response} "
                        f"user={head_user} tool_results={head_tools} fork_head={fork_id} "
                        f"leaf={leaf_id} forks={fork_ids} "
                        f"idx={cursor_idx} orig={original_idx} tools={tool_blocks_len} "
                        f"has_results={has_results} suppress={suppress_tools}{Colors.RESET}"
                    )
                except Exception:
                    pass
                target_turn = child_turn or child_cursor.current_turn
                target_id = target_turn.gen_id_or_parent if target_turn else "None"
                print(f"{Colors.TOOL_WARNING}Batch: failed to apply assistant response on batch child {target_id}: {exc}{Colors.RESET}")
        elif replay_mode and source_has_assistant and not has_assistant and not tool_blocks:
            # Abort replay for this branch when the assistant response was suppressed.
            try:
                if child_cursor.current_turn:
                    child_cursor.current_turn.is_archived = True
            except Exception:
                pass
            target_turn = child_turn or child_cursor.current_turn
            target_id = target_turn.gen_id_or_parent if target_turn else "None"
            print(f"{Colors.TOOL_WARNING}Replay: suppressed assistant for batch child {target_id}; branch archived.{Colors.RESET}")

        if tool_blocks and not suppress_tools:
            if _tool_blocks_have_abort(tool_blocks):
                print(f"{Colors.TOOL_WARNING}Tool round aborted by action after execution; skipping results for {child_turn.gen_id_or_parent}.{Colors.RESET}")
                continue
            if _tool_blocks_have_results(tool_blocks):
                try:
                    head = child_cursor.current_turn if child_cursor else None
                    head_id = head.gen_id_or_parent if head else "None"
                    if _replay_debug_enabled(child_cursor):
                        print(
                            f"{Colors.SYSTEM}Batch: tool follow-up prep child={child_cursor.display_id()} "
                            f"head={head_id} idx={cursor_idx} orig={original_idx} tools={len(tool_blocks)}{Colors.RESET}"
                        )
                except Exception:
                    pass
                ctx = child_cursor.context
                scope = _require_live_chat_scope(child_cursor)
                anchor = scope.find_active_anchor("auto_tool", child_cursor)
                if not anchor:
                    anchor = scope.start_try_out_anchor(
                        _auto_anchor_name("auto_tool", child_cursor),
                        child_cursor.head,
                        kind="auto_tool",
                        retry_limit=_auto_tool_retry_limit(),
                        origin_cursor=child_cursor,
                    )
                if anchor:
                    cursor_meta["auto_tool_anchor"] = anchor.anchor_name
                tool_has_error = _tool_call_has_error(tool_blocks)
                decrement_retry = tool_has_error
                if anchor and decrement_retry and anchor.retries_remaining <= 0:
                    warning = f"{Colors.TOOL_WARNING}Tool retries exhausted; returning to main without launching a new try_out.{Colors.RESET}"
                    print(warning)
                    try:
                        closed_cursor = _ensure_registered_cursor(
                            scope.close_try_out_anchor(anchor.anchor_name, dist_mode="keep")
                        )
                        if closed_cursor:
                            _set_active_cursor(closed_cursor)
                    except Exception:
                        pass
                    cursor_meta["auto_tool_anchor"] = None
                    _drain_and_close_auto_tryouts(cursor=child_cursor, close_anchors=False)
                elif anchor:
                    if decrement_retry:
                        anchor.retries_remaining -= 1
                    anchor_turn_for_retry = child_cursor.current_turn or anchor.anchor_turn or child_cursor.head
                    main_cursor_for_try, tryout_cursor = child_cursor.add_try_out(
                        anchor=anchor,
                        anchor_turn=anchor_turn_for_retry,
                        keep_in_main=True,
                    )
                    if fork_obj and cursor_idx is not None:
                        fork_obj.set_main_placeholder(cursor_idx, main_cursor_for_try)
                    try_turn = getattr(tryout_cursor, "head", None)
                    if try_turn:
                        child_cursor = tryout_cursor
                    if fork_obj and cursor_idx is not None:
                        fork_obj.update_active_cursor(cursor_idx, child_cursor)
                    main_turn = getattr(main_cursor_for_try, "head", None)
                    if try_turn:
                        try_turn.main_thread = True
                    if main_turn:
                        main_turn.main_thread = True
                    child_cursor.add_tool_results(tool_blocks)
                    child_cursor.set_auto(True)
                    _sync_fork_cursor(fork_obj, cursor_idx, child_cursor)
                    if _replay_debug_enabled(child_cursor):
                        try:
                            head = child_cursor.current_turn if child_cursor else None
                            head_id = head.gen_id_or_parent if head else "None"
                            print(
                                f"{Colors.SYSTEM}Batch: tool results added child={child_cursor.display_id()} "
                                f"head={head_id} idx={cursor_idx} orig={original_idx}{Colors.RESET}"
                            )
                        except Exception:
                            pass
                    if child_cursor.head:
                        child_cursor.head.main_thread = True
                    _record_source_mapping(source_turn, child_cursor)
                    if next_active_forks is not None:
                        pending_tool_followup = True
                        if _replay_debug_enabled(child_cursor):
                            try:
                                label = child_cursor.display_id() if child_cursor else "None"
                                print(f"{Colors.SYSTEM}DEBUG: queued tool follow-up on {label}.{Colors.RESET}")
                            except Exception:
                                pass
                        next_active_forks.append({
                            "fork": fork_obj,
                            "cursor_idx": cursor_idx if cursor_idx is not None else 0,
                            "original_index": original_idx,
                            "source_turn": source_turn,
                        })
            else:
                tool_results_cursor = child_cursor.add_tool_results(tool_blocks)
                tool_results_cursor.set_auto(True)
                _sync_fork_cursor(fork_obj, cursor_idx, tool_results_cursor)
                if _replay_debug_enabled(tool_results_cursor):
                    try:
                        head = tool_results_cursor.current_turn if tool_results_cursor else None
                        head_id = head.gen_id_or_parent if head else "None"
                        print(
                            f"{Colors.SYSTEM}Batch: tool results added (no-anchor) child={tool_results_cursor.display_id()} "
                            f"head={head_id} idx={cursor_idx} orig={original_idx}{Colors.RESET}"
                        )
                    except Exception:
                        pass
                    _record_source_mapping(source_turn, tool_results_cursor)
                if next_active_forks is not None and not was_truncated and not was_canceled and not is_error:
                    pending_tool_followup = True
                    if _replay_debug_enabled(child_cursor):
                        try:
                            label = tool_results_cursor.display_id() if tool_results_cursor else "None"
                            print(f"{Colors.SYSTEM}DEBUG: queued tool follow-up on {label}.{Colors.RESET}")
                        except Exception:
                            pass
                    next_active_forks.append({
                        "fork": fork_obj,
                        "cursor_idx": cursor_idx if cursor_idx is not None else 0,
                        "original_index": original_idx,
                        "source_turn": source_turn,
                    })

        metrics_payload = ChatCursor.update_response_metrics(response=response_obj) if response_obj else {}
        if metrics_payload:
            child_cursor.update_metrics(metrics_payload)

        pending_auto_retry = False
        if was_truncated and not was_canceled and allow_auto_retry and next_active_forks is not None:
            prior_leaf = child_cursor.current_turn
            new_cursor_for_fork, scheduled = await _handle_auto_continuation(child_cursor)
            if scheduled:
                # The helper already requested the iteration via the context.
                # We just need to ensure this fork's new cursor is what's used in the next batch round.
                _sync_fork_cursor(fork_obj, cursor_idx, new_cursor_for_fork)
                next_active_forks.append({
                    "fork": fork_obj,
                    "cursor_idx": cursor_idx if cursor_idx is not None else 0,
                    "original_index": original_idx,
                    "source_turn": source_turn,
                })
            # Preserve mapping for the original source child on its leaf before continuation is added
            _record_source_mapping(source_turn, child_cursor)
                # Ensure descendants of this child resolve beneath the truncated leaf, not the batch hub
            try:
                child_cursor_map[source_turn] = child_cursor.snapshot_at(prior_leaf)
            except Exception:
                child_cursor_map[source_turn] = child_cursor
            pending_auto_retry = True
        if pending_auto_retry:
            _drain_and_close_auto_tryouts(child_cursor, close_anchors=False)
            continue

        if pending_tool_followup:
            _record_source_mapping(source_turn, child_cursor)
            if source_turn:
                child_cursor_map[source_turn] = child_cursor
            continue

        stabilized_cursor = _normalize_cursor_after_auto_iters(child_cursor)
        if not stabilized_cursor:
            try:
                if child_cursor.current_turn:
                    child_cursor.current_turn.is_archived = True
                    print(f"{Colors.TOOL_WARNING}Replay: auto-rounds cleanup failed on {child_cursor.current_turn.gen_id_or_parent}; skipping branch.{Colors.RESET}")
            except Exception:
                pass
            continue
        _record_source_mapping(source_turn, stabilized_cursor)
        _sync_fork_cursor(fork_obj, cursor_idx, stabilized_cursor)
        if source_turn:
            child_cursor_map[source_turn] = stabilized_cursor
    return child_cursor_map

async def _process_batch_results(
    sorted_results: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    active_forks: List[Dict[str, Any]],
    *,
    tools_view: Optional[ToolsView],
    batch_round: int,
    is_concurrent: bool,
    wall_clock_time: float,
    cursor: ChatCursor,
    batch_forks: Optional[List[ChatForks]] = None,
    source_to_dest_cursor_map: Optional[Dict[Turn, Dict[str, Any]]] = None,
    allow_auto_retry: bool = True,
    replay_mode: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Shared batch post-processing: applies assistant/tool results, auto-retry, and aggregates metrics.
    Returns (next_active_forks, batch_metrics_to_store).
    """
    # Build aggregate structures
    all_final_response_items: List[Tuple[int, InferenceResponse]] = []
    all_final_chunks: Dict[int, Dict[str, Any]] = {}
    text_by_prompt: Dict[int, str] = {}
    per_batch_metrics: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []

    for _, res in sorted_results:
        final_chunks = res.get("final_chunks", {})
        prompt_index_map: List[int] = res.get("prompt_index_map") or []
        streamed_text = res.get("text_by_prompt", {}) or {}
        for idx, text in streamed_text.items():
            mapped_idx = prompt_index_map[idx] if idx < len(prompt_index_map) else idx
            if text:
                text_by_prompt[mapped_idx] = text_by_prompt.get(mapped_idx, "") + text
        for idx, chunk in final_chunks.items():
            mapped_idx = prompt_index_map[idx] if idx < len(prompt_index_map) else idx
            all_final_chunks[mapped_idx] = chunk
        metrics = res.get("metrics", {}) or {}
        per_batch_metrics.append((res.get("batch_index", -1), metrics, res.get("request_payload", {})))
        for idx, chunk in final_chunks.items():
            try:
                resp = InferenceResponse(**chunk)
                mapped_idx = prompt_index_map[idx] if idx < len(prompt_index_map) else idx
                all_final_response_items.append((mapped_idx, resp))
            except Exception as e:
                print(f"{Colors.ERROR}Error processing batch results: {e}{Colors.RESET}")

    # Build final_by_prompt map
    final_by_prompt: Dict[int, InferenceResponse] = {}
    for mapped_idx, resp in all_final_response_items:
        final_by_prompt[mapped_idx] = resp

    # Validate concatenated chunk_text against response_text when provided.
    for mapped_idx, resp in final_by_prompt.items():
        if resp.response_text:
            combined = text_by_prompt.get(mapped_idx, "")
            if combined != resp.response_text and not (resp.tool_blocks or getattr(resp, "was_truncated", False)):
                print(f"{Colors.TOOL_WARNING}Warning: response_text mismatch for prompt {mapped_idx}. chunk_text len={len(combined)} response_text len={len(resp.response_text)}.{Colors.RESET}")

    next_active_forks: List[Dict[str, Any]] = []
    await _apply_batch_results_to_children(
        prepared_forks=active_forks,
        final_by_prompt=final_by_prompt,
        final_chunks=all_final_chunks,
        text_by_prompt=text_by_prompt,
        source_to_dest_cursor_map=source_to_dest_cursor_map,
        tools_view=tools_view,
        allow_auto_retry=allow_auto_retry,
        next_active_forks=next_active_forks,
        batch_context_cursor=cursor,
        replay_mode=replay_mode,
    )

    # Aggregate batch-level metrics
    batch_metrics_to_store: Dict[str, Any] = {}
    if all_final_response_items:
        total_input_tokens_batch = sum(resp.input_tokens or 0 for _, resp in all_final_response_items)
        total_output_tokens_batch = sum(resp.output_tokens or 0 for _, resp in all_final_response_items)
        total_gen_duration_batch = sum(resp.generation_duration_sec or 0.0 for _, resp in all_final_response_items)
        overall_tps_batch = 0.0
        if is_concurrent and wall_clock_time > 0:
            overall_tps_batch = total_output_tokens_batch / wall_clock_time
        elif not is_concurrent and total_gen_duration_batch > 0:
            overall_tps_batch = total_output_tokens_batch / total_gen_duration_batch
        all_ttfts = [resp.time_to_first_token_sec for _, resp in all_final_response_items if resp.time_to_first_token_sec is not None]
        avg_ttft_batch = sum(all_ttfts) / len(all_ttfts) if all_ttfts else None
        batch_metrics_to_store = {
            "total_input_tokens": total_input_tokens_batch,
            "total_output_tokens": total_output_tokens_batch,
            "total_generation_duration_sec": total_gen_duration_batch,
            "overall_tps": overall_tps_batch,
            "avg_time_to_first_token_sec": avg_ttft_batch,
        }
        if batch_forks:
            first_hub_cursor = batch_forks[0].batch_hub
            if not first_hub_cursor:
                raise RuntimeError("Batch fork missing batch_hub cursor for metrics update.")
            batch_payload = ChatCursor.update_response_metrics(metrics=batch_metrics_to_store)
            first_hub_cursor.update_metrics(batch_payload)

    return next_active_forks, batch_metrics_to_store
def _build_metric_parts(metrics: Mapping[str, Any]) -> List[str]:
    parts: List[str] = []
    if (in_tok := metrics.get("input_tokens")) is not None:
        parts.append(f"In: {in_tok}")
    if (out_tok := metrics.get("output_tokens")) is not None:
        parts.append(f"Out: {out_tok}")
    if (gen_time := metrics.get("generation_duration_sec")) is not None:
        parts.append(f"GenTime: {gen_time:.1f}s")
    if (ttft := metrics.get("time_to_first_token_sec")) is not None:
        if ttft < metrics.get("generation_duration_sec", float("inf")):
            parts.append(f"Latency: {ttft * 1000:.0f}ms")
    if (tps := metrics.get("tokens_per_second")) is not None:
        parts.append(f"TPS: {tps:.1f}")
    if (cache_metric := metrics.get("cache_metric")):
        parts.append(f"Cache: {cache_metric}")
    if (cache_warming := metrics.get("cache_warming")):
        parts.append(f"Warm-up: {cache_warming}")
    if metrics.get("was_truncated"):
        parts.append(f"{Colors.BRIGHT_YELLOW}Truncated: Yes{Colors.METRICS}")
    if (tool_blocks_count := metrics.get("tool_blocks_count")) is not None:
        parts.append(f"Tool Blocks: {tool_blocks_count}")
    if (tool_blocks_tokens := metrics.get("tool_blocks_tokens")) is not None:
        parts.append(f"Tool Tokens: {tool_blocks_tokens}")
    return parts


def _print_per_item_metrics(chunk_data: Dict[str, Any]):
    """Helper to print a formatted line of per-item metrics."""
    metrics_parts = _build_metric_parts(chunk_data)
    if metrics_parts:
        print(f"{Colors.METRICS}  {' | '.join(metrics_parts)}{Colors.RESET}")

def _normalize_adapter_list(raw_value: Any) -> List[str]:
    """Convert adapter metadata into a normalized list of strings."""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value if str(item)]
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        return [stripped] if stripped else []
    return []


def _format_adapter_label(values: List[str]) -> str:
    return ", ".join(values) if values else "__base__"


def _extract_turn_adapters(turn: Optional[Turn]) -> Optional[List[str]]:
    if not turn or not getattr(turn, "data", None):
        return None
    adapters_payload = turn.data.get("$Adapters")
    if adapters_payload is None:
        return None
    if isinstance(adapters_payload, list):
        return [str(item) for item in adapters_payload if str(item)]
    if isinstance(adapters_payload, str):
        cleaned = adapters_payload.strip()
        return [cleaned] if cleaned else None
    return None


def _resolve_llm_display_name(cursor: ChatCursor, turn: Optional[Turn] = None) -> str:
    try:
        conversation = cursor.context.chat_session
        if conversation:
            name = conversation.initial_params.get("engine_base_model_name")
            if name:
                return str(name)
    except Exception:
        pass
    return "Assistant"


def _extract_response_adapters(response_item: Optional[InferenceResponse]) -> List[str]:
    if response_item is None:
        return []
    adapters_list = getattr(response_item, "active_adapters", None)
    if isinstance(adapters_list, list):
        return [str(item) for item in adapters_list if str(item)]
    adapters_field = response_item.adapters
    if not adapters_field:
        return []
    if isinstance(adapters_field, str):
        return [segment.strip() for segment in adapters_field.split(",") if segment.strip()]
    return []


def _record_response_adapters(cursor: ChatCursor, response_item: Optional[InferenceResponse]) -> None:
    adapters = _extract_response_adapters(response_item)
    if adapters:
        cursor.record_adapters(adapters)


def _extract_error_details(response_item: InferenceResponse) -> Dict[str, Any]:
    """Prepare a stable payload for cursor.error recordings."""
    details: Dict[str, Any] = {}
    if response_item.prompt_index is not None:
        details["prompt_index"] = response_item.prompt_index
    if response_item.was_canceled is not None:
        details["was_canceled"] = response_item.was_canceled
    if response_item.was_truncated is not None:
        details["was_truncated"] = response_item.was_truncated
    if response_item.full_traceback:
        details["full_traceback"] = response_item.full_traceback
    return details


class _ConsolePrinter:
    """Small helper to centralize stdout writes with optional locking."""

    def __init__(self, lock: Optional[asyncio.Lock] = None) -> None:
        self._lock = lock

    async def write(self, text: str, *, flush: bool = False) -> None:
        if self._lock:
            async with self._lock:
                sys.stdout.write(text)
                if flush or text.endswith("\n"):
                    sys.stdout.flush()
        else:
            sys.stdout.write(text)
            if flush or text.endswith("\n"):
                sys.stdout.flush()

    async def writeln(self, text: str = "") -> None:
        await self.write(f"{text}\n")


@dataclass
class InferenceStreamResult:
    final_responses: List[InferenceResponse]
    final_chunks: Dict[int, Dict[str, Any]]
    stream_metrics: Dict[str, Any]
    was_canceled: bool
    was_truncated: bool
    stream_completed: bool
    text_by_prompt: Dict[int, str]


async def _render_prompt_started(
    ctx: StreamDisplayContext,
    chunk_data: Dict[str, Any],
    *,
    printer: _ConsolePrinter,
    llm_name: str,
    adapters_label: str,
    cursor: ChatCursor,
) -> None:
    prompt_idx = ctx.original_prompt_index
    if prompt_idx is None:
        prompt_idx = chunk_data.get("prompt_index", ctx.prompt_index) or 0

    if ctx.show_override_banner and ctx.override_total:
        await printer.write(
            f"\n  {Colors.TOOL}--- Response for Item {prompt_idx + 1}/{ctx.override_total} "
            f"(Adapters: {adapters_label}) ---{Colors.RESET}\n"
        )

    if ctx.show_batch_header:
        total_prompts = ctx.total_original_prompts or ctx.override_total or (prompt_idx + 1)
        header_text = f"--- Response for Prompt {prompt_idx + 1}/{total_prompts}"
        if ctx.total_batches and ctx.batch_index is not None:
            header_text += f" (from Batch {ctx.batch_index + 1}/{ctx.total_batches}, Adapters: {adapters_label}) "
        else:
            header_text += f" (Adapters: {adapters_label}) "
        await printer.write(f"\n{Colors.HEADER}{header_text:->58}{Colors.RESET}\n")

    if ctx.show_prompt_echo and chunk_data.get("prompt"):
        await printer.write(
            f"\n{Colors.YOU_HEADER}Prompt:{Colors.RESET} "
            f"{Colors.YOU_CONTENT}{chunk_data.get('prompt')}{Colors.RESET}"
        )

    if ctx.show_response_banner:
        scope = _scope_for_cursor(cursor)
        bag_dict = _resolve_bag_dict(scope, cursor=cursor)
        replay_start_time = bag_dict.get("replay_start_time") if bag_dict else None
        start_time = replay_start_time or time.time()
        elapsed_seconds = int(time.time() - start_time)
        if ctx.show_turn_counter:
            render_cursor = ctx.cursor_override or cursor
            chat_turns_count = render_cursor.user_turns_count()
            await printer.write(
                f"\n{Colors.LLM_HEADER}Turn {chat_turns_count + 1}->"
                f"{llm_name} [{adapters_label}]:{Colors.RESET}\n{Colors.LLM_CONTENT}",
                flush=True,
            )
        else:
            await printer.write(
                f"\n{Colors.LLM_HEADER}{llm_name} [{adapters_label}]:{Colors.RESET}\n{Colors.LLM_CONTENT}",
                flush=True,
            )


async def _consume_inference_stream(
    api_iterator: AsyncIterator[Dict[str, Any]],
    *,
    display_plan: StreamDisplayPlan,
    llm_name: str,
    adapters_for_display: List[str],
    cursor: ChatCursor,
    console_lock: Optional[asyncio.Lock] = None,
    drop_stream_metrics: bool = False,
) -> InferenceStreamResult:
    printer = _ConsolePrinter(console_lock)
    final_chunks: Dict[int, Dict[str, Any]] = {}
    final_responses: List[InferenceResponse] = []
    final_metrics: Dict[str, Any] = {}
    was_canceled = False
    was_truncated = False
    stream_completed = False
    text_by_prompt: Dict[int, str] = {}

    async for chunk_data in api_iterator:
        chunk_type = chunk_data.get("chunkType")
        if chunk_data.get("was_canceled"):
            was_canceled = True
            await printer.write(f"\n{Colors.BRIGHT_YELLOW}Request was canceled.{Colors.RESET}\n")
        if chunk_data.get("was_truncated"):
            was_truncated = True

        if chunk_type == ChunkType.PROMPT_STARTED.value:
            prompt_index = chunk_data.get("prompt_index", 0)
            ctx = display_plan.resolve(prompt_index)
            adapters_label = ctx.adapters_label or _format_adapter_label(
                _normalize_adapter_list(chunk_data.get("adapters")) or adapters_for_display
            )
            await _render_prompt_started(
                ctx,
                chunk_data,
                printer=printer,
                llm_name=llm_name,
                adapters_label=adapters_label,
                cursor=cursor,
            )
            continue

        if chunk_type == ChunkType.STREAMING_CHUNK.value:
            prompt_index = chunk_data.get("prompt_index", 0)
            token = chunk_data.get("chunk_text", "")
            if token:
                await printer.write(token, flush=True)
                text_by_prompt[prompt_index] = text_by_prompt.get(prompt_index, "") + token
            if chunk_data.get("is_final_chunk"):
                await printer.write(f"{Colors.RESET}\n")
                chunk_copy = dict(chunk_data)
                active_adapters = _normalize_adapter_list(chunk_copy.get("adapters")) or adapters_for_display
                if active_adapters:
                    chunk_copy["active_adapters"] = active_adapters
                final_chunks[prompt_index] = chunk_copy
                final_responses.append(InferenceResponse(**chunk_copy))
                _print_per_item_metrics(chunk_copy)
            continue

        if chunk_type == ChunkType.STREAMING_ENDED.value:
            await printer.write(f"{Colors.RESET}\n")
            metrics_payload = {k: v for k, v in chunk_data.items() if k != "chunkType"}
            if drop_stream_metrics:
                keep_keys = {"cache_queued", "in_flight_req", "mem_allocated", "mem_reserved"}
                final_metrics = {k: metrics_payload[k] for k in keep_keys if k in metrics_payload}
            else:
                final_metrics = metrics_payload
            stream_completed = True
            break

        if chunk_type == ChunkType.ERROR.value:
            await printer.write(f"\n{Colors.ERROR}Error during stream: {chunk_data.get('error')}{Colors.RESET}\n")
            prompt_index = chunk_data.get("prompt_index", 0)
            chunk_copy = dict(chunk_data)
            chunk_copy.setdefault("is_final_chunk", True)
            active_adapters = _normalize_adapter_list(chunk_copy.get("adapters")) or adapters_for_display
            if active_adapters:
                chunk_copy["active_adapters"] = active_adapters
            final_chunks[prompt_index] = chunk_copy
            final_responses.append(InferenceResponse(**chunk_copy))
            stream_completed = True
            continue

    return InferenceStreamResult(
        final_responses=final_responses,
        final_chunks=final_chunks,
        stream_metrics=final_metrics,
        was_canceled=was_canceled,
        was_truncated=was_truncated,
        stream_completed=stream_completed,
        text_by_prompt=text_by_prompt,
    )

async def _handle_general_batch_generation(
    cursor: ChatCursor,
    pt_session: "PromptSession",
    is_concurrent: bool,
    allow_auto_retry: bool = True,
    *,
    prepared_prompts: Optional[List[str]] = None,
    override_adapters: Optional[List[str]] = None,
    command_text_override: Optional[str] = None,
) -> Tuple[ChatCursor, bool]:
    """Handles both sequential (`/g b`) and concurrent (`/g bc`) batch generation.
    This function now returns the cursor and suppression flag.
    
    This function implements a "forking" and "looping" mechanism to handle
    tool calls within a batch. Each prompt in the initial batch is treated as a
    separate conversational "fork".
    - Forks that return a final text response are considered "completed".
    - Forks that return tool calls are considered "active". Their tool calls are
      executed, the results are appended to their history, and they are added to
      a new batch for the next round of inference.
    This process repeats until all forks have completed.
    """    
    global LAST_EXCEPTION_TRACEBACK
    # This is now handled by the new looping logic.

    def _finalize(suppress: bool = True) -> Tuple[ChatCursor, bool]:
        return cursor, suppress

    # --- NEW: Distinguish between modes ---
    mode_name = "Concurrent Batch" if is_concurrent else "Batch"
    batch_prompts: List[str] = []
    if prepared_prompts is not None:
        batch_prompts = list(prepared_prompts)
    else:
        # --- FIX: Use a loop that correctly handles multiline input ---
        # The original loop was not correctly handling multiline input.
        while True:
            try:
                prompt_ui = [('class:prompt.you', f"{mode_name} Prompt {len(batch_prompts) + 1}: ")]
                prompt = await pt_session.prompt_async(prompt_ui)
                if not prompt.strip():
                    break
                batch_prompts.append(prompt.strip())
            except (KeyboardInterrupt, EOFError):
                print("\nBatch generation cancelled.")
                return _finalize(True)

    if not batch_prompts:
        print("No prompts entered. Exiting batch mode.")
        return _finalize(True)

    if prepared_prompts is None:
        print(f"{Colors.SYSTEM}--- {mode_name} Generation Mode ---{Colors.RESET}")
        print(f"Collected {len(batch_prompts)} prompts. Starting generation.")

    # --- NEW: Ask for concurrency level only in concurrent mode ---
    concurrency_level = 1
    if is_concurrent:
        concurrency_level_str = await pt_session.prompt_async("Enter number of concurrent requests: ")
        try:
            concurrency_level = int(concurrency_level_str)
            if concurrency_level <= 0: raise ValueError
        except (ValueError, TypeError):
            print("Invalid number. Batch generation cancelled.")
            return _finalize(True)

    gen_config_for_payload = _generation_config_template_snapshot()
    max_new_override = _max_new_tokens_value()
    if max_new_override:
        gen_config_for_payload["max_new_tokens"] = max_new_override
    console_lock = asyncio.Lock()
    llm_name = cursor.context.chat_session.initial_params.get("engine_base_model_name") or "LLM"
    tools_view: Optional[ToolsView] = cursor.get_tools_view()
    active_tools_for_request = cursor.get_active_tools(tools_view)

    async def process_batch_request(
        request_payload: Dict[str, Any],
        batch_forks: List[Dict[str, Any]],
        batch_index: int,
        total_batches_in_round: int,
        batch_round: int
    ) -> Dict[str, Any]:
        """Shared batch submit + stream fan-out helper."""
        return await _run_batch_request(
            request_payload=request_payload,
            batch_forks=batch_forks,
            batch_index=batch_index,
            total_batches_in_round=total_batches_in_round,
            batch_round=batch_round,
            llm_name=llm_name,
            parent_cursor=cursor,
            console_lock=console_lock,
            total_prompts_for_batch=len(batch_prompts),
        )
    
    # --- Main Batch Processing Loop ---
    try:
        # --- Create a "try out" node for the batch and populate it ---
        batch_forks: List[ChatForks] = []
        active_forks: List[Dict[str, Any]] = []
        batch_groups = batch_list(batch_prompts, concurrency_level)
        owner_cursors: List[ChatCursor] = []

        parent_cursor = cursor
        fork_index = 0
        for i, batch_group in enumerate(batch_groups): # type: ignore
            if not batch_group:
                continue

            batch_holder = parent_cursor

            # This shows how to isolate batches from the main branch so they will stay hidden
            # unless some prompts are explicitly promoted to main-thread
            #if is_concurrent:
            #    main_cursor, try_cursor = parent_cursor.add_try_out()
            #    parent_cursor = main_cursor
            #    batch_holder = try_cursor

            chunk_override_adapters = None
            if override_adapters:
                chunk_override_adapters = override_adapters[fork_index:fork_index + len(batch_group)]

            try:
                fork = batch_holder.add_batch(
                    prompts=batch_group,
                    command_text=command_text_override or f"/{'g bc' if is_concurrent else 'g b'}",
                    make_active=False,
                    adopt_into_context=True,
                    override_adapters=chunk_override_adapters,
                )
            except ValueError as e:
                print(f"{Colors.ERROR}Error: {e}{Colors.RESET}")
                print(f"{Colors.DIM}Hint: A single batch ('/g b') cannot be created as a sibling to another batch. Use '/g bc' to create a concurrent fork.{Colors.RESET}")
                return _finalize(True)

            batch_forks.append(fork)
            owner_cursors.append(batch_holder)
            fork.prompt_indices = list(range(fork_index, fork_index + len(fork.cursors)))
            fork.cursor_meta = fork.cursor_meta or {}
            for idx, child_cursor in enumerate(fork.cursors):
                fork.cursor_meta.setdefault(child_cursor.id, {})
                active_forks.append({
                    "fork": fork,
                    "cursor_idx": idx,
                    "original_index": fork.prompt_indices[idx],
                })
            fork_index += len(fork.cursors)

        if batch_forks and cursor.context:
            try:
                hub_cursor = batch_forks[0].batch_hub
                hub_turn = hub_cursor.current_turn if hub_cursor else None
                if hub_turn:
                    fork_handles: List[str] = []
                    for fork in batch_forks:
                        for fork_cursor in getattr(fork, "cursors", []) or []:
                            handle = getattr(fork_cursor, "context_id", None)
                            if handle:
                                fork_handles.append(handle)
                    scope = _scope_for_cursor(cursor)
                    bag_dict = _resolve_bag_dict(scope, cursor=cursor)
                    if bag_dict is not None:
                        bag_dict.setdefault("batch_stack", []).append({
                        "hub_cursor": hub_cursor,
                        "hub_turn": hub_turn,
                        "forks": batch_forks,
                        "owner_cursors": owner_cursors,
                        "fork_handles": fork_handles,
                        })
            except Exception:
                pass

        batch_round = 0

        while active_forks:
            batch_round += 1
            print(f"\n{Colors.SYSTEM}--- Batch Round {batch_round}: Processing {len(active_forks)} active fork(s) ---{Colors.RESET}")

            # --- Create InferenceRequest payloads and asyncio tasks for the current active_forks ---
            tasks = []
            requests_for_batch: List[Dict[str, Any]] = []
            batch_groups = batch_list(active_forks, concurrency_level)

            for i, batch_group in enumerate(batch_groups): # type: ignore

                batch_cursors: List[ChatCursor] = []
                for fork in batch_group:
                    entry_cursor = _fork_entry_cursor(fork)
                    if not entry_cursor:
                        continue
                    if not entry_cursor.context and cursor.context:
                        try:
                            entry_cursor.bind_context(cursor.context)
                        except Exception:
                            pass
                    if entry_cursor:
                        batch_cursors.append(entry_cursor)

                turns_for_chunk = [c.current_turn for c in batch_cursors if c.current_turn]
                if not turns_for_chunk:
                    continue

                if not batch_cursors:
                    continue
                inference_payload, _, _ = batch_cursors[0].build_inference_request(
                    batch=batch_cursors,
                    request_id_prefix=f"batch_round_{batch_round+1}",
                    manual_continue=False,
                    include_tools=True,
                )
                if override_adapters:
                    adapters_for_chunk: List[str] = []
                    for fork in batch_group:
                        entry_cursor = _fork_entry_cursor(fork)
                        turn = entry_cursor.current_turn if entry_cursor else None
                        adapters_for_chunk.append(getattr(turn, "data", {}).get("adapter_override_name"))
                    if any(adapters_for_chunk):
                        inference_payload["override_adapters"] = adapters_for_chunk

                requests_for_batch.append(inference_payload)
                # Pass batch_round to the processing function for better logging
                task = asyncio.create_task(process_batch_request(
                    inference_payload, batch_group, 
                    batch_index=i, total_batches_in_round=len(batch_groups),
                    batch_round=batch_round 
                ))
 
                tasks.append(task)

            # --- Gather results for the current batch round ---
            start_time = time.monotonic()
            if any(req.get("reset_metrics") for req in requests_for_batch):
                print(f"{Colors.SYSTEM}Note: This batch request includes 'reset_metrics=True'.{Colors.RESET}")
            try:
                results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                print(f"\n{Colors.SYSTEM}Batch round cancelled by user. Cleaning up...{Colors.RESET}")
                # Propagate cancellation to any remaining tasks
                for task in tasks: task.cancel()
                raise # Re-raise to be caught by the outer try/except
            wall_clock_time = time.monotonic() - start_time

            print(f"\n{Colors.SYSTEM}--- Batch Round {batch_round} Generation Complete ---{Colors.RESET}")

            # --- Associate results with their original requests ---
            valid_results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            for i, res in enumerate(results_from_gather):
                if isinstance(res, dict) and not isinstance(res, Exception):
                    # Pair the result with the request that generated it
                    valid_results.append((requests_for_batch[i], res))

            # Sort by original chunk index to maintain order
            sorted_results = sorted(valid_results, key=lambda r: r[1].get("batch_index", -1))

            # --- Per-Batch Metrics (for this round) ---
            if len(sorted_results) > 1: # Only show per-batch if there's more than one
                print(f"\n{Colors.METRICS}--- Per-Batch Metrics ---{Colors.RESET}")
                for _, res in sorted_results: # type: ignore
                    metrics = res.get("metrics", {}) # type: ignore
                    if metrics: # type: ignore
                        batch_idx = res.get("batch_index", -1)
                        # Use the newly added request_payload to get the prompt count
                        request_payload_from_res = res.get("request_payload", {})
                        num_prompts_in_chunk = len(request_payload_from_res.get("messages_list", []))
                        metrics_line = f"  Batch {batch_idx + 1}/{len(batch_groups)} ({num_prompts_in_chunk} prompts):"
                        if (total_in := metrics.get("total_input_tokens")) is not None: metrics_line += f" In: {total_in}"
                        if (total_out := metrics.get("total_output_tokens")) is not None: metrics_line += f" Out: {total_out}"
                        if (duration := metrics.get("total_generation_duration_sec")) is not None: metrics_line += f" GenTime: {duration:.1f}s"
                        if (tps := metrics.get("overall_tps")) is not None: metrics_line += f" TPS: {tps:.1f}"
                        if (latency := metrics.get("avg_time_to_first_token_sec")) is not None: metrics_line += f" Latency: {latency * 1000:.0f}ms"
                        if cache_queued := metrics.get("cache_queued"): metrics_line += f" Queued: {cache_queued}"
                        if (in_flight := metrics.get("in_flight_req")) is not None: metrics_line += f" In-flight: {in_flight}"
                        if (mem_alloc := metrics.get("mem_allocated")) is not None: metrics_line += f" Mem(A): {mem_alloc:.0f}MB"
                        if (mem_rsvd := metrics.get("mem_reserved")) is not None: metrics_line += f" Mem(R): {mem_rsvd:.0f}MB"
                        print(f"{Colors.METRICS}{metrics_line}{Colors.RESET}")
                print(f"{Colors.METRICS}---------------------------{Colors.RESET}")

            # --- Cumulative Metrics (for this round) ---
            total_prompts_processed = 0
            total_input_tokens = 0
            total_output_tokens = 0 # This will be recalculated from final_response_items
            cache_queued: Optional[str] = None
            mem_alloc_final, mem_rsvd_final = None, None
            total_tool_blocks = 0
            total_tool_blocks_tokens = 0
            in_flight_req: Optional[int] = None

            # --- FIX: Re-aggregate metrics from the final chunks to ensure accuracy ---
            # The metrics from the last STREAMING_ENDED chunk are only for that chunk, not the whole round.
            all_final_response_items_this_round: List[InferenceResponse] = []
            for _, res in sorted_results: # type: ignore
                final_chunks_for_batch = res.get("final_chunks", {})
                for item_chunk in sorted(final_chunks_for_batch.values(), key=lambda x: x.get("prompt_index", -1)):
                    all_final_response_items_this_round.append(InferenceResponse(**item_chunk))

                request_payload_from_res = res.get("request_payload", {})
                total_prompts_processed += len(request_payload_from_res.get("messages_list", []))
                metrics = res.get("metrics", {})
                if metrics: 
                    total_input_tokens += metrics.get("total_input_tokens") or 0
                    total_output_tokens += metrics.get("total_output_tokens") or 0
                    if metrics.get("cache_queued"):
                        cache_queued = metrics.get("cache_queued")
                    if metrics.get("total_tool_blocks") is not None:
                        total_tool_blocks += metrics.get("total_tool_blocks", 0)
                    if metrics.get("total_tool_blocks_tokens") is not None:
                        total_tool_blocks_tokens += metrics.get("total_tool_blocks_tokens", 0)
                    # Capture the in-flight count from the last available chunk's metrics
                    if metrics.get("in_flight_req") is not None:
                        in_flight_req = metrics.get("in_flight_req")
                    # Capture memory from the last chunk
                    if metrics.get("mem_allocated") is not None: mem_alloc_final = metrics.get("mem_allocated")
                    if metrics.get("mem_reserved") is not None: mem_rsvd_final = metrics.get("mem_reserved")

            # Recalculate total output tokens from the actual final items
            total_output_tokens = sum(item.output_tokens or 0 for item in all_final_response_items_this_round)
            
            if total_prompts_processed > 0:
                metrics_line = f"Metrics (Cumulative Total): Prompts: {total_prompts_processed} In: {total_input_tokens} Out: {total_output_tokens}"
                if is_concurrent:
                    if wall_clock_time > 0:
                        metrics_line += f" WallTime: {wall_clock_time:.1f}s"
                        if total_output_tokens > 0: metrics_line += f" TPS: {total_output_tokens / wall_clock_time:.1f}"
                else: # Sequential # type: ignore
                    total_gen_duration_seq = sum(res.get("metrics", {}).get("total_generation_duration_sec", 0.0) for _, res in sorted_results)
                    if total_gen_duration_seq > 0:
                        metrics_line += f" GenTime: {total_gen_duration_seq:.1f}s"
                        if total_output_tokens > 0: metrics_line += f" TPS: {total_output_tokens / total_gen_duration_seq:.1f}"
                if cache_queued: 
                    metrics_line += f" Queued: {cache_queued}"
                if total_tool_blocks > 0:
                    metrics_line += f" Tool Blocks: {total_tool_blocks}"
                if total_tool_blocks_tokens > 0:
                    metrics_line += f" TBlk Tokens: {total_tool_blocks_tokens}"
                if mem_alloc_final is not None: metrics_line += f" Mem(A): {mem_alloc_final:.0f}MB"
                if mem_rsvd_final is not None: metrics_line += f" Mem(R): {mem_rsvd_final:.0f}MB"
                if in_flight_req is not None:
                    metrics_line += f" In-flight: {in_flight_req}"
                print(f"\n{Colors.METRICS}{metrics_line}{Colors.RESET}")

            # --- Process results and prepare for the next round ---
            next_active_forks, _ = await _process_batch_results(
                sorted_results=sorted_results,
                active_forks=active_forks,
                tools_view=tools_view,
                batch_round=batch_round,
                is_concurrent=is_concurrent,
                wall_clock_time=wall_clock_time,
                cursor=cursor,
                batch_forks=batch_forks,
                allow_auto_retry=allow_auto_retry and _auto_retry_enabled(),
            )

            active_forks = next_active_forks
            if not active_forks:
                print(f"\n{Colors.SYSTEM}--- All Batch Forks Complete ---{Colors.RESET}")

        first_batch_turn_created = batch_forks[0].batch_hub if batch_forks else None # type: ignore

        if batch_forks:
            hub_cursor = batch_forks[0].batch_hub
            placeholder = cursor.resolve_batch_main_placeholder(hub_cursor=hub_cursor)
            if placeholder:
                try:
                    if cursor.context:
                        scope = _scope_for_cursor(cursor)
                        if scope:
                            rebound = scope.register_cursor_for_turn(placeholder, make_active=True)
                        else:
                            context = _resolve_context_for_scope(scope, cursor=cursor)
                            rebound = context.register_cursor_for_turn(placeholder, make_active=True) if context else None
                        if rebound:
                            cursor = rebound
                except Exception:
                    pass
            try:
                scope = _scope_for_cursor(cursor)
                ctx = _resolve_context_for_scope(scope, cursor=cursor)
                hub_turn = hub_cursor.current_turn if hub_cursor else None
                entry = _pop_batch_stack_entry(ctx, getattr(hub_turn, "gen_id", None), scope=scope)
                if entry and scope:
                    dropped = _drop_batch_entry_handles(entry, ctx=ctx, keep_cursor=cursor, scope=scope)
                    if dropped == 0:
                        dropped = _drop_batch_entry_cursors(entry, ctx=ctx, keep_cursor=cursor, scope=scope)
                    hub_handle = getattr(entry.get("hub_cursor"), "context_id", None)
                    keep_handle = getattr(cursor, "context_id", None)
                    if hub_handle and hub_handle != keep_handle:
                        try:
                            scope.drop_cursor(hub_handle)
                            dropped += 1
                        except Exception:
                            pass
                    if ctx:
                        _cleanup_cursor_registry(scope, ctx)
            except Exception:
                pass

        # Update the original command turn's metadata to reflect completion.
        # Aggregate total metrics for the entire batch and store in first_batch_turn_created
        if first_batch_turn_created: # type: ignore
            total_input_tokens_batch = sum(item.input_tokens or 0 for item in all_final_response_items_this_round)
            total_output_tokens_batch = sum(item.output_tokens or 0 for item in all_final_response_items_this_round)
            total_gen_duration_batch = sum(item.generation_duration_sec or 0.0 for item in all_final_response_items_this_round)
            
            # For overall TPS, use the wall_clock_time if concurrent, else sum of gen_duration
            overall_tps_batch = 0.0
            if is_concurrent and wall_clock_time > 0:
                overall_tps_batch = total_output_tokens_batch / wall_clock_time
            elif not is_concurrent and total_gen_duration_batch > 0:
                overall_tps_batch = total_output_tokens_batch / total_gen_duration_batch

            # Average TTFT
            all_ttfts = [item.time_to_first_token_sec for item in all_final_response_items_this_round if item.time_to_first_token_sec is not None]
            avg_ttft_batch = sum(all_ttfts) / len(all_ttfts) if all_ttfts else None

            batch_metrics_to_store = {
                "total_input_tokens": total_input_tokens_batch,
                "total_output_tokens": total_output_tokens_batch,
                "total_generation_duration_sec": total_gen_duration_batch,
                "overall_tps": overall_tps_batch,
                "avg_time_to_first_token_sec": avg_ttft_batch,
                "total_prompts_processed": total_prompts_processed,
                "total_tool_blocks": total_tool_blocks,
                "total_tool_blocks_tokens": total_tool_blocks_tokens,
                "cache_queued": cache_queued,
                "in_flight_req": in_flight_req,
                "mem_allocated": mem_alloc_final,
                "mem_reserved": mem_rsvd_final,
            }
            batch_metrics_payload = ChatCursor.update_response_metrics(metrics=batch_metrics_to_store)
            cursor.update_metrics(batch_metrics_payload)

        for batch_fork in batch_forks: # type: ignore
            batch_cursor = batch_fork.batch_hub
            batch_turn = batch_cursor.current_turn if batch_cursor else None
            if not batch_cursor or not batch_turn:
                continue
            original_command = (
                batch_cursor.get_data().get("command")
                or command_text_override
                or f"/{'g bc' if is_concurrent else 'g b'}"
            )
            batch_cursor.get_data()["command"] = f"{original_command}: Generated {len(batch_turn.turns)} responses." # type: ignore

    except (KeyboardInterrupt, asyncio.CancelledError) as e:
        print(f"\n{mode_name} generation cancelled by user. Notifying engine and cleaning up tasks... ({type(e).__name__})")
        # Cancel all in-flight asyncio tasks for this batch
        cancelled_count = 0
        tasks = [] # Ensure tasks is defined
        if 'requests_for_batch' not in locals(): # type: ignore
            requests_for_batch = []
            
        if 'tasks' in locals(): # type: ignore
            for i, task in enumerate(tasks): # type: ignore
                if not task.done():
                    task.cancel()
                    # Also notify the engine for each request that was in flight
                    request_id_to_cancel = requests_for_batch[i].get("request_id")
                    if request_id_to_cancel:
                        await call_api("cancel-request", {"request_id": request_id_to_cancel}, suppress=True)
                    cancelled_count += 1
            if cancelled_count > 0:
                await asyncio.gather(*tasks, return_exceptions=True) # type: ignore
        await _wait_for_engine_ready()
    except Exception as e:
        # --- FIX: Store exception if not already set by a sub-task ---
        _store_exception_traceback_if_clear(e)

    # Return True to suppress the main loop's prompt if no follow-up is needed.
    return _finalize(True) # The new logic is self-contained and never needs a follow-up from the main loop.

def _compare_and_print_session_diffs(
    source_session: ChatSession,
    dest_session: ChatSession,
    *,
    header: str = "Warning: Replay environment differs from source session.",
    include_engine_config: bool = False,
) -> bool:
    """Compares two ChatSession objects and prints a summary of differences."""
    diffs: List[str] = []
    source_params = source_session.initial_params or {}
    dest_params = dest_session.initial_params or {}

    # 1. Engine Name
    source_engine = source_params.get('engine_base_model_name')
    dest_engine = dest_params.get('engine_base_model_name')
    if source_engine != dest_engine:
        diffs.append(f"  - Engine Name: '{source_engine or 'N/A'}' -> '{dest_engine or 'N/A'}'")

    # 2. Parser Profile
    source_parser = source_session.parser_profile.key if source_session.parser_profile else "N/A"
    dest_parser = dest_session.parser_profile.key if dest_session.parser_profile else "N/A"
    if source_parser != dest_parser:
        diffs.append(f"  - Parser Profile: '{source_parser}' -> '{dest_parser}'")

    # 3. Settings (inference_defaults)
    source_inf = source_session.inference_defaults
    dest_inf = dest_session.inference_defaults
    inf_diffs = []
    if source_inf and dest_inf:
        # Compare key inference settings
        if source_inf.stream != dest_inf.stream:
            inf_diffs.append(f"    - stream: {source_inf.stream} -> {dest_inf.stream}")
        if source_inf.cache != dest_inf.cache:
            inf_diffs.append(f"    - cache: '{source_inf.cache}' -> '{dest_inf.cache}'")
        if source_inf.return_prompt != dest_inf.return_prompt:
            inf_diffs.append(f"    - return_prompt: {source_inf.return_prompt} -> {dest_inf.return_prompt}")
        if source_inf.generation_config != dest_inf.generation_config:
            # Full diff is too verbose, just note the difference.
            inf_diffs.append(f"    - generation_config differs.")
    elif source_inf != dest_inf:
        inf_diffs.append("Inference defaults object missing or mismatch.")
    
    if inf_diffs:
        diffs.append("  - Inference Settings differ:")
        diffs.extend(inf_diffs)

    # 4. Initial Parameters
    param_diffs = []
    keys_to_check = {
        'system_message', 'advertised_tools', 'silent_tools', 'disabled_tools',
        'auto_tool_retry_limit', 'auto_continue_retry_limit', 'results_as_user_role',
        'pack_results_as_one_role', 'max_new_tokens_override'
    }
    # Use sorted list for deterministic output
    for key in sorted(list(keys_to_check)):
        s_val = source_params.get(key)
        d_val = dest_params.get(key)
        if key == "system_message":
            if s_val is None:
                s_val = ""
            if d_val is None:
                d_val = ""
        if s_val != d_val:
            # Use json.dumps for a consistent, readable representation
            s_repr = json.dumps(s_val)
            d_repr = json.dumps(d_val)
            if len(s_repr) > 40: s_repr = f"{s_repr[:37]}..."
            if len(d_repr) > 40: d_repr = f"{d_repr[:37]}..."
            param_diffs.append(f"    - {key}: {s_repr} -> {d_repr}")

    if param_diffs:
        diffs.append("  - Initial Parameters differ:")
        diffs.extend(param_diffs)

    if include_engine_config:
        source_engine_config = source_session.engine_config or {}
        dest_engine_config = dest_session.engine_config or {}
        engine_config_diffs = []
        for key in sorted(set(source_engine_config.keys()) | set(dest_engine_config.keys())):
            s_val = source_engine_config.get(key)
            d_val = dest_engine_config.get(key)

            if key == "model_layout":
                try:
                    # The string representation from str() uses single quotes, but json.loads wants double quotes.
                    s_val_parsed = json.loads(s_val.replace("'", "\""))
                    d_val_parsed = json.loads(d_val.replace("'", "\""))
                    s_val = json.dumps(s_val_parsed, sort_keys=True)
                    d_val = json.dumps(d_val_parsed, sort_keys=True)
                except (json.JSONDecodeError, AttributeError):
                    pass # Fallback to original string comparison if parsing fails

            if s_val != d_val:
                s_repr = json.dumps(s_val)
                d_repr = json.dumps(d_val)
                if len(s_repr) > 40: s_repr = f"{s_repr[:37]}..."
                if len(d_repr) > 40: d_repr = f"{d_repr[:37]}..."
                engine_config_diffs.append(f"    - {key}: {s_repr} -> {d_repr}")
        if engine_config_diffs:
            diffs.append("  - Engine Config differs:")
            diffs.extend(engine_config_diffs)

    if diffs:
        print(f"{Colors.TOOL_WARNING}{header}{Colors.RESET}")
        for d in diffs:
            print(d)
        return True
    return False


def _effective_engine_name_for_session(chat_session: Optional[ChatSession]) -> str:
    if not chat_session:
        return "N/A"
    params = getattr(chat_session, "initial_params", {}) or {}
    name = params.get("engine_base_model_name")
    if not name:
        engine_config = getattr(chat_session, "engine_config", {}) or {}
        name = engine_config.get("base_model_name") or engine_config.get("base_model_name_or_path")
        if not name:
            other_config = engine_config.get("other_config")
            if isinstance(other_config, dict):
                name = other_config.get("base_model_name_or_path") or other_config.get("base_model_name")
    if not name:
        return "N/A"
    try:
        return str(name)
    except Exception:
        return "N/A"


def _effective_replay_llm_name(cursor: Optional[ChatCursor]) -> str:
    baseline = _engine_baseline_session()
    name = _effective_engine_name_for_session(baseline)
    if name == "N/A" and cursor:
        name = _effective_engine_name_for_session(cursor.chat_session)
    return name if name != "N/A" else "LLM"


def _engine_baseline_session() -> Optional[ChatSession]:
    if conversation_template:
        return conversation_template
    if chat_scope:
        try:
            return chat_scope.active_cursor().chat_session
        except Exception:
            return None
    return None


def _mark_session_sync_required(chat_session: Optional[ChatSession], required: bool) -> None:
    if chat_session:
        setattr(chat_session, "sync_required", required)


def _mark_session_adapter_sync_required(
    chat_session: Optional[ChatSession],
    required: bool,
    missing_adapters: Optional[List[str]] = None,
) -> None:
    if chat_session:
        setattr(chat_session, "adapter_sync_required", required)
        if missing_adapters is not None:
            setattr(chat_session, "adapter_sync_missing", list(missing_adapters))


def _session_requires_engine_sync(chat_session: Optional[ChatSession]) -> bool:
    return bool(chat_session and getattr(chat_session, "sync_required", False))


def _session_requires_adapter_sync(chat_session: Optional[ChatSession]) -> bool:
    return bool(chat_session and getattr(chat_session, "adapter_sync_required", False))


def _session_requires_sync(chat_session: Optional[ChatSession]) -> bool:
    return _session_requires_engine_sync(chat_session) or _session_requires_adapter_sync(chat_session)


def _warn_session_sync_required(chat_session: Optional[ChatSession] = None) -> None:
    if not chat_session:
        try:
            chat_session = _require_current_cursor().chat_session
        except Exception:
            chat_session = None
    if not chat_session:
        return
    needs_engine = _session_requires_engine_sync(chat_session)
    needs_adapters = _session_requires_adapter_sync(chat_session)
    if needs_engine and needs_adapters:
        print(
            f"{Colors.TOOL_WARNING}Warning: This conversation is read-only until you run "
            f"/s sync to align it with the current engine and load required adapters.{Colors.RESET}"
        )
    elif needs_engine:
        print(
            f"{Colors.TOOL_WARNING}Warning: This conversation is read-only until you run "
            f"/s sync to align it with the current engine.{Colors.RESET}"
        )
    elif needs_adapters:
        print(
            f"{Colors.TOOL_WARNING}Warning: This conversation is read-only until you run "
            f"/s sync to load the required adapters.{Colors.RESET}"
        )


def _expected_loaded_adapters_for_cursor(cursor: Optional[ChatCursor]) -> Set[str]:
    if not cursor:
        return set()
    chat_session = cursor.chat_session
    if not chat_session:
        return set()
    active_turn = cursor.current_turn or chat_session.root_turn
    if not active_turn:
        return set()
    try:
        path = cursor.session.get_active_path_for_llm(active_turn)
    except Exception:
        path = []
    loaded: Set[str] = set()
    for turn in path or []:
        for cmd in getattr(turn, "cmd", []) or []:
            api_name = getattr(cmd, "api_name", None)
            if api_name not in {"load-adapter", "unload-adapter"}:
                continue
            params = getattr(cmd, "api_params", None) or {}
            name = None
            if isinstance(params, dict):
                name = params.get("adapter_name")
            if not name:
                data = getattr(cmd, "data", {}) or {}
                if isinstance(data, dict):
                    name = data.get("adapter_name")
            if not name:
                continue
            name_str = str(name)
            if api_name == "load-adapter":
                loaded.add(name_str)
            else:
                loaded.discard(name_str)
    return loaded


async def _check_adapter_sync_status(cursor: Optional[ChatCursor]) -> bool:
    if not cursor or not cursor.chat_session:
        return False
    expected = _expected_loaded_adapters_for_cursor(cursor)
    if not expected:
        _mark_session_adapter_sync_required(cursor.chat_session, False, [])
        return False
    resp = await call_api("get-loaded-adapters", {})
    if not isinstance(resp, dict) or resp.get("status") != "success":
        print(f"{Colors.TOOL_WARNING}Warning: Could not verify loaded adapters from engine.{Colors.RESET}")
        return False
    loaded_entries = resp.get("data", {}).get("adapters", []) if isinstance(resp.get("data"), dict) else []
    loaded_names = {str(entry.get("name")) for entry in loaded_entries if entry.get("name")}
    missing = sorted(expected - loaded_names)
    required = bool(missing)
    _mark_session_adapter_sync_required(cursor.chat_session, required, missing)
    if required:
        print(f"{Colors.TOOL_WARNING}Warning: Missing adapters for this conversation: {', '.join(missing)}{Colors.RESET}")
    return required


async def _check_engine_sync_status(cursor: Optional[ChatCursor]) -> bool:
    engine_session = _engine_baseline_session()
    chat_session = cursor.chat_session if cursor else None
    has_diffs = False
    if chat_session and engine_session:
        has_diffs = _compare_and_print_session_diffs(
            engine_session,
            chat_session,
            header="Warning: Engine config differs from loaded session.",
            include_engine_config=True,
        )
        _mark_session_sync_required(chat_session, has_diffs)
    adapter_diffs = await _check_adapter_sync_status(cursor)
    if has_diffs or adapter_diffs:
        _warn_session_sync_required(chat_session)
    return has_diffs or adapter_diffs


def _sync_chat_session_from_engine(chat_session: ChatSession, engine_session: ChatSession) -> None:
    chat_session.engine_config = copy.deepcopy(engine_session.engine_config) if engine_session.engine_config else {}
    chat_session.engine_warnings = list(engine_session.engine_warnings or [])
    chat_session.parser_profile = copy.deepcopy(engine_session.parser_profile)
    chat_session.inference_defaults = copy.deepcopy(engine_session.inference_defaults)
    chat_session.initial_params = copy.deepcopy(engine_session.initial_params or {})


async def _sync_loaded_adapters_for_cursor(cursor: ChatCursor) -> bool:
    if not current_config or "adapters_root_dir" not in current_config:
        print(f"{Colors.ERROR}Configuration not loaded. Cannot determine adapters root directory.{Colors.RESET}")
        return False
    if not cursor or not cursor.chat_session:
        print(f"{Colors.ERROR}No active conversation to sync adapters.{Colors.RESET}")
        return False
    expected = _expected_loaded_adapters_for_cursor(cursor)
    if not expected:
        _mark_session_adapter_sync_required(cursor.chat_session, False, [])
        print(f"{Colors.SYSTEM}No adapters recorded on the active path.{Colors.RESET}")
        return True
    resp = await call_api("get-loaded-adapters", {})
    if not isinstance(resp, dict) or resp.get("status") != "success":
        print(f"{Colors.ERROR}Failed to query loaded adapters from engine.{Colors.RESET}")
        return False
    loaded_entries = resp.get("data", {}).get("adapters", []) if isinstance(resp.get("data"), dict) else []
    loaded_names = {str(entry.get("name")) for entry in loaded_entries if entry.get("name")}
    missing = sorted(expected - loaded_names)
    if not missing:
        _mark_session_adapter_sync_required(cursor.chat_session, False, [])
        print(f"{Colors.SYSTEM}Adapters already synced for the active path.{Colors.RESET}")
        return True
    adapters_root_dir = current_config.get("adapters_root_dir")
    print(f"{Colors.SYSTEM}Syncing {len(missing)} adapter(s) by name from {adapters_root_dir}.{Colors.RESET}")
    all_ok = True
    for name in missing:
        payload = {
            "adapter_name": name,
            "adapter_path": adapters_root_dir,
            "if_exists": IfExistsEnum.IGNORE,
        }
        response = await call_api("load-adapter", payload)
        if response.get("status") != "success":
            all_ok = False
            print(f"{Colors.ERROR}Failed to load adapter '{name}' during sync: {response.get('message')}{Colors.RESET}")
    if all_ok:
        _mark_session_adapter_sync_required(cursor.chat_session, False, [])
    return all_ok

async def _handle_session_command(
    cursor: ChatCursor,
    args_str: str,
    user_input: str,
    pt_session: "PromptSession",
) -> Tuple[Optional[EngineSession], bool, Optional[ChatCursor]]:
    global session_control, LAST_ENUMERATED_SESSIONS, current_config
    parts = args_str.split(" ", 1)
    command_full_text = f"/s {args_str}"
    sub_cmd_full = parts[0].lower().strip()
    if sub_cmd_full in {"?", "help"}:
        _print_session_cli_help()
        return None, True, None
    sub_args = parts[1].strip() if len(parts) > 1 else ""
    
    sub_cmd_map = {
        "n": "new", "s": "save", "l": "load", "e": "enum", "enum": "enum",
        "sync": "sync",
        "h": "history", "history": "history", "hs": "history_summary", "hl": "history_logs",
        "hf": "history_full", "hfl": "history_full_logs",
        "hfs": "history_full_summary", "hfsl": "history_full_summary_logs",
        "ch": "commands_history", "chl": "commands_history_logs",
        "p": "prompt", "prompt": "prompt", "t": "turn", "m": "main",
        "r": "replay", "replay": "replay",
        "c": "conversation", "conversation": "conversation",
        "a": "add", "add": "add",
    }
    sub_cmd = sub_cmd_map.get(sub_cmd_full)

    if not sub_cmd_full:  # User typed just /s, now means "show history"
        cursor = _require_current_cursor()
        print_current_session_info(cursor, show_messages=True, show_forks=False, clear_first=True)
        return None, True, None
    
    if not sub_cmd:
        print(f"{Colors.ERROR}Unknown session command: '{sub_cmd_full}'. Options: n, s, l, e, h, hl, hs, hf, hfl, hfs, hfsl, ch, chl, p, t, m, a.{Colors.RESET}")
        return None, True, None

    if sub_cmd == "new":
        cursor = _require_current_cursor()
        cursor.log_command(command_full_text)
        new_name = sub_args.strip() or f"untitled_{int(time.time())}"
        template = _chat_session_template()
        current_params = _current_param_snapshot()
        default_params = copy.deepcopy(getattr(template, "initial_params", {}) or {})
        reset_diffs = _summarize_param_resets(current_params, default_params)
        current_system = cursor.effective_system_message()
        target_system = default_params.get("system_message")
        current_adapters = cursor.get_effective_adapters() or ["__base__"]
        target_adapters = ["__base__"]
        state_diffs = _summarize_effective_state_resets(
            current_system,
            target_system,
            current_adapters,
            target_adapters,
            default_system=default_params.get("system_message"),
        )
        loaded_adapters_resp = await call_api("get-loaded-adapters")
        loaded_adapters = loaded_adapters_resp.get("data", {}).get("adapters", []) # noqa
        if session_control.exists(new_name) and not new_name.startswith("untitled_"): # type: ignore
            if input(f"Session '{new_name}' already exists. Overwrite? (y/N): ").lower() != 'y':
                return None, True, None
        # Retain system message when creating new session
        new_session = EngineSession(name=new_name)
        new_session.default_system_message  = current_config.get("default_system_message", "") 
        new_chat_session = new_session.insert_conversation(-1, template)
        _seed_chat_session_flags(new_chat_session)
        cursor = _bootstrap_cursor_for_session(new_session, conversation=new_chat_session)
        _set_active_cursor(cursor)
        if loaded_adapters:
            recorded = _record_loaded_adapter_commands(cursor, loaded_adapters, command_label="/s new (record load-adapter)")
            if recorded:
                print(f"{Colors.SYSTEM}Recorded {recorded} loaded adapter(s) on new session root.{Colors.RESET}")
        if reset_diffs or state_diffs:
            print(f"{Colors.SYSTEM}Reset settings for new session:{Colors.RESET}")
            for line in reset_diffs:
                print(f"  - {line}")
            for line in state_diffs:
                print(f"  - {line}")
        print_current_session_info(cursor, show_messages=False) # noqa
        return new_session, True, cursor
    elif sub_cmd == "add":
        cursor = _require_current_cursor()
        cursor.log_command(command_full_text)
        session = cursor.session
        template = _chat_session_template()
        current_params = _current_param_snapshot()
        default_params = copy.deepcopy(getattr(template, "initial_params", {}) or {})
        reset_diffs = _summarize_param_resets(current_params, default_params)
        current_system = cursor.effective_system_message()
        target_system = default_params.get("system_message")
        current_adapters = cursor.get_effective_adapters() or ["__base__"]
        target_adapters = ["__base__"]
        state_diffs = _summarize_effective_state_resets(
            current_system,
            target_system,
            current_adapters,
            target_adapters,
            default_system=default_params.get("system_message"),
        )
        loaded_adapters_resp = await call_api("get-loaded-adapters")
        loaded_adapters = loaded_adapters_resp.get("data", {}).get("adapters", []) # noqa
        title_arg: Optional[str] = None
        if sub_args.strip():
            try:
                parsed = shlex.split(sub_args)
                if parsed:
                    title_arg = " ".join(parsed)
            except ValueError:
                title_arg = sub_args.strip()
        new_chat_session = session.insert_conversation(
            index=-1,
            template=template,
            title=title_arg)
        _seed_chat_session_flags(new_chat_session)
        clear_screen()
        cursor = _bootstrap_cursor_for_session(session, conversation=new_chat_session)
        _set_active_cursor(cursor)
        conv_num = session.conversations.index(new_chat_session) + 1
        title_display = f" '{title_arg}'" if title_arg else ""
        print(f"{Colors.SYSTEM}Added conversation {conv_num}{title_display} (ID: {new_chat_session.id}).{Colors.RESET}")
        if loaded_adapters:
            recorded = _record_loaded_adapter_commands(cursor, loaded_adapters, command_label="/s add (record load-adapter)")
            if recorded:
                print(f"{Colors.SYSTEM}Recorded {recorded} loaded adapter(s) on new conversation root.{Colors.RESET}")
        if reset_diffs or state_diffs:
            print(f"{Colors.SYSTEM}Reset settings for new conversation:{Colors.RESET}")
            for line in reset_diffs:
                print(f"  - {line}")
            for line in state_diffs:
                print(f"  - {line}")
        print_current_session_info(cursor, show_messages=False, clear_first=False)
        return session, True, cursor
    elif sub_cmd == "save": # type: ignore
        save_target = sub_args.strip()
        cursor = _require_current_cursor()
        cursor.log_command(command_full_text)
        session = cursor.session
        _update_active_conversation_state(cursor)
        if not save_target:
            # It's a path
            session_control.save(session) # type: ignore
        elif '.json' in save_target or os.path.sep in save_target or (os.altsep and os.altsep in save_target):
            # It's a path
            target_path = Path(save_target).expanduser().resolve()
            session.name = target_path.stem # Update session name to match file
            session_control.save(session, path=target_path) # type: ignore
        else:
            # It's a name
            session.name = save_target # type: ignore
            session_control.save(session)
        return None, True, None
    elif sub_cmd == "sync":
        cursor = _require_current_cursor()
        cursor.log_command(command_full_text)
        chat_session = cursor.chat_session
        if not chat_session:
            print(f"{Colors.ERROR}No active conversation to sync.{Colors.RESET}")
            return None, True, None
        sync_arg = sub_args.strip()
        if sync_arg:
            print(f"{Colors.TOOL_WARNING}Ignoring sync target '{sync_arg}'; '/s sync' now syncs engine and adapters.{Colors.RESET}")
        do_engine = True
        do_adapters = True

        if do_engine:
            engine_session = _engine_baseline_session()
            if not engine_session:
                print(f"{Colors.ERROR}Engine baseline is not available; cannot sync.{Colors.RESET}")
                return None, True, None
            _sync_chat_session_from_engine(chat_session, engine_session)
            active_turn = cursor.current_turn or chat_session.root_turn
            cursor = _reset_chat_context(cursor.session, chat_session, active_turn)
            _set_active_cursor(cursor)
            _mark_session_sync_required(chat_session, False)
            print(f"{Colors.SYSTEM}Conversation synced to current engine settings.{Colors.RESET}")
            print_current_session_info(cursor, show_messages=False, clear_first=False)

        if do_adapters:
            ok = await _sync_loaded_adapters_for_cursor(cursor)
            if ok:
                _mark_session_adapter_sync_required(chat_session, False, [])
                print(f"{Colors.SYSTEM}Adapter sync complete.{Colors.RESET}")
            else:
                await _check_adapter_sync_status(cursor)
        return None, True, cursor
    elif sub_cmd == "load": # type: ignore
        cursor = _require_current_cursor()
        cursor.log_command(command_full_text)
        name_to_load_input = sub_args.strip()

        if not name_to_load_input: # User typed just /s l
            saved_sessions = session_control.list_sessions() # type: ignore # Does not print
            LAST_ENUMERATED_SESSIONS.clear()
            # For interactive mode, we still need to populate the dict with simple numbers
            for i, name in enumerate(saved_sessions):
                LAST_ENUMERATED_SESSIONS[str(i+1)] = str(session_control._get_session_path(name)) # type: ignore

            if not LAST_ENUMERATED_SESSIONS:
                print("No saved sessions found.")
                return None, True, None
            
            selected_name_from_prompt = await _prompt_for_selection(
                item_list=list(LAST_ENUMERATED_SESSIONS.values()),
                item_names_for_prompt=[Path(p).stem for p in LAST_ENUMERATED_SESSIONS.values()],
                prompt_message="Select session to load",
                allow_multiple=False
            ) 

            if not selected_name_from_prompt or not isinstance(selected_name_from_prompt, str): # type: ignore
                print("Session load cancelled.")
                return None, True, None
            name_to_load_str = selected_name_from_prompt
        else:
            name_to_load_str = name_to_load_input

        loaded_session: Optional[EngineSession] = None # type: ignore
        # Check if it's a path first
        if '.json' in name_to_load_str or os.path.sep in name_to_load_str or (os.altsep and os.altsep in name_to_load_str):
            target_path = Path(name_to_load_str).expanduser().resolve()
            if target_path.exists():
                loaded_session = session_control.load(str(target_path)) # type: ignore
            else:
                print(f"{Colors.ERROR}Session file not found at path: {target_path}{Colors.RESET}")
        # Check if it's a hierarchical number (e.g., 1.2)
        elif '.' in name_to_load_str and all(part.isdigit() for part in name_to_load_str.split('.')):
            if not LAST_ENUMERATED_SESSIONS:
                # If the user tries to load by number but no enumeration has happened yet.
                print(f"{Colors.ERROR}No sessions have been enumerated. Use '/s enum' first to load by number.{Colors.RESET}")
                return None, True, None
            try:
                # Find the path from the last enumeration
                path_to_load = LAST_ENUMERATED_SESSIONS[name_to_load_str]
                loaded_session = session_control.load(path_to_load) # type: ignore
            except (KeyError, IndexError):
                print(f"Invalid session number: {name_to_load_str}. Use '/s enum'.")
        # Check if it's a simple number
        elif name_to_load_str.isdigit():
            if not LAST_ENUMERATED_SESSIONS:
                # If the user tries to load by number but no enumeration has happened yet.
                print(f"{Colors.ERROR}No sessions have been enumerated. Use '/s enum' first to load by number.{Colors.RESET}")
                return None, True, None
            try:
                # Find the path from the last enumeration using the simple number as key
                path_to_load = LAST_ENUMERATED_SESSIONS[name_to_load_str]
                loaded_session = session_control.load(path_to_load) # type: ignore
            except (KeyError, IndexError):
                print(f"Invalid session number: {name_to_load_str}. Use '/s enum'.")
        else:
            loaded_session = session_control.load(name_to_load_str) # type: ignore

        if loaded_session:
            # --- FIX: Ensure a conversation exists before trying to access it ---
            if loaded_session.get_conversations_count() == 0:
                # If the loaded session has no conversations, create a default one.
                print(f"{Colors.TOOL_WARNING}Loaded session '{loaded_session.name}' was empty. Creating a new conversation tree.{Colors.RESET}")
                loaded_chat_session = loaded_session.add_conversation(
                    parser_profile=_engine_parser_profile())
                _seed_chat_session_flags(loaded_chat_session)
            else:
                # Otherwise, pick the last active conversation if available.
                target_index = loaded_session.last_converation if 0 <= loaded_session.last_converation < loaded_session.get_conversations_count() else 0
                if target_index != loaded_session.last_converation:
                    print(f"{Colors.TOOL_WARNING}Stored conversation index {loaded_session.last_converation + 1} is out of range; defaulting to 1.{Colors.RESET}")
                loaded_chat_session = loaded_session.get_conversation(target_index)
            # --- END FIX ---
            cursor = _bootstrap_cursor_for_session(loaded_session, conversation=loaded_chat_session)
            _set_active_cursor(cursor)
            print_current_session_compact(cursor, clear_first=False)
            active_conv_idx = _conversation_index(loaded_session, loaded_chat_session)
            if active_conv_idx is not None:
                print(f"{Colors.SYSTEM}Active conversation: {active_conv_idx + 1}{Colors.RESET}")
            await _check_engine_sync_status(cursor)
            if loaded_session.active_adapters_on_save:
                print(f"{Colors.TOOL_WARNING}Warning: This session had these adapters loaded on save: {loaded_session.active_adapters_on_save}{Colors.RESET}")
            print("  ---")
            return loaded_session, True, cursor
        # else: error already printed by session_control.load or #num parsing
        return None, True, None # Return current if load failed
    elif sub_cmd == "enum":
        cursor = _require_current_cursor()
        cursor.log_command(command_full_text)
        # This is a client-side file system operation, no API call.
        start_path_str = sub_args.strip()
        start_path = Path(start_path_str).expanduser().resolve() if start_path_str else Path(current_config["sessions_save_dir"])

        if not start_path.is_dir():
            print(f"Error: Directory not found: {start_path}")
            return None, True, None

        LAST_ENUMERATED_SESSIONS.clear() # This will now map number -> path
        print(f"Sessions in '{start_path}' (use '/s l <num>' or '/s l <path>'):")

        def _walk_dir(dir_path: Path, prefix: str = ""):
            items = sorted(list(dir_path.iterdir()))
            local_counter = 0
            for item in items:
                if item.is_dir():
                    local_counter += 1
                    new_prefix = f"{prefix}{local_counter}."
                    print(f"{Colors.DIM}{'  ' * len(prefix.split('.'))}{prefix}{local_counter} {item.name}/{Colors.RESET}")
                    _walk_dir(item, new_prefix)
                elif item.suffix == '.json':
                    local_counter += 1
                    num_id = f"{prefix}{local_counter}"
                    print(f"{'  ' * len(prefix.split('.'))}{num_id} {item.stem}")
                    LAST_ENUMERATED_SESSIONS[num_id] = str(item.resolve())
        
        _walk_dir(start_path)
        return None, True, None
    elif sub_cmd == "history":  # /s h - Dumps current session info and active turns
        base_cursor = _require_current_cursor()
        base_cursor.log_command(command_full_text)
        gen_arg, rounds, err = _parse_history_args(sub_args)
        if err:
            print(f"{Colors.ERROR}{err}{Colors.RESET}")
            return None, True, None
        history_cursor, scope_start = _resolve_history_scope(base_cursor, gen_arg, rounds)
        if not history_cursor:
            return None, True, None
        branch_path = _history_branch_path_from_start(history_cursor, history_cursor.current_turn, scope_start)
        print_current_session_info(
            history_cursor,
            show_messages=True,
            show_forks=False,
            clear_first=False,
            active_only=True,
            start_at_turn=scope_start,
            branch_path=branch_path,
        )
        return None, True, None
    elif sub_cmd == "history_logs": # /s hl
        base_cursor = _require_current_cursor()
        base_cursor.log_command(command_full_text)
        gen_arg, rounds, err = _parse_history_args(sub_args)
        if err:
            print(f"{Colors.ERROR}{err}{Colors.RESET}")
            return None, True, None
        history_cursor, scope_start = _resolve_history_scope(base_cursor, gen_arg, rounds)
        if not history_cursor:
            return None, True, None
        branch_path = _history_branch_path_from_start(history_cursor, history_cursor.current_turn, scope_start)
        print_current_session_info(
            history_cursor,
            show_messages=True,
            show_logs=True,
            clear_first=False,
            active_only=True,
            start_at_turn=scope_start,
            branch_path=branch_path,
        )
        return None, True, None
    elif sub_cmd == "history_full": # /s hf
        base_cursor = _require_current_cursor()
        base_cursor.log_command(command_full_text)
        gen_arg, rounds, err = _parse_history_args(sub_args)
        if err:
            print(f"{Colors.ERROR}{err}{Colors.RESET}")
            return None, True, None
        history_cursor, scope_start = _resolve_history_scope(base_cursor, gen_arg, rounds)
        depth_limit = rounds
        if not history_cursor:
            return None, True, None
        print_current_session_info(
            history_cursor,
            show_messages=True,
            show_logs=False,
            clear_first=False,
            start_at_turn=scope_start,
            depth=depth_limit,
            show_all_turns=True,
        )
        return None, True, None
    elif sub_cmd == "history_full_logs": # /s hfl
        base_cursor = _require_current_cursor()
        base_cursor.log_command(command_full_text)
        gen_arg, rounds, err = _parse_history_args(sub_args)
        if err:
            print(f"{Colors.ERROR}{err}{Colors.RESET}")
            return None, True, None
        history_cursor, scope_start = _resolve_history_scope(base_cursor, gen_arg, rounds)
        depth_limit = rounds
        if not history_cursor:
            return None, True, None
        print_current_session_info(
            history_cursor,
            show_messages=True,
            show_logs=True,
            clear_first=False,
            start_at_turn=scope_start,
            depth=depth_limit,
            show_all_turns=True,
        )
        return None, True, None
    elif sub_cmd == "history_summary": # /s hs
        base_cursor = _require_current_cursor()
        base_cursor.log_command(command_full_text)
        gen_arg, rounds, err = _parse_history_args(sub_args)
        if err:
            print(f"{Colors.ERROR}{err}{Colors.RESET}")
            return None, True, None
        history_cursor, scope_start = _resolve_history_scope(base_cursor, gen_arg, rounds)
        if not history_cursor:
            return None, True, None
        branch_path = _history_branch_path_from_start(history_cursor, history_cursor.current_turn, scope_start)
        print_current_session_compact(
            history_cursor,
            clear_first=False,
            start_at_turn=scope_start,
            branch_path=branch_path,
        )
        return None, True, None
    elif sub_cmd in ["history_full_summary", "history_full_summary_logs"]: # /s hfs or /s hfsl
        show_logs = (sub_cmd == "history_full_summary_logs")
        base_cursor = _require_current_cursor()
        base_cursor.log_command(command_full_text)
        gen_arg, rounds, err = _parse_history_args(sub_args)
        if err:
            print(f"{Colors.ERROR}{err}{Colors.RESET}")
            return None, True, None
        if gen_arg == "*":
            if rounds is not None:
                print(f"{Colors.TOOL_WARNING}Ignoring --rounds for '*'; summaries show full trees per conversation.{Colors.RESET}")
            _print_all_conversation_summaries(base_cursor, show_logs=show_logs, depth=None)
            return None, True, None
        history_cursor, scope_start = _resolve_history_scope(base_cursor, gen_arg, rounds)
        depth_limit = rounds
        if not history_cursor:
            return None, True, None
        print_session_tree_summary(
            history_cursor,
            active=history_cursor.current_turn,
            start_at_turn=scope_start,
            depth=depth_limit,
            show_logs=show_logs,
        ) # noqa
        return None, True, None
    elif sub_cmd in ["commands_history", "commands_history_logs"]: # /s ch or /s chl
        cursor = _require_current_cursor()
        cursor.log_command(command_full_text)
        include_logs = (sub_cmd == "commands_history_logs")
        gen_arg, rounds, err = _parse_history_args(sub_args)
        if err:
            print(f"{Colors.ERROR}{err}{Colors.RESET}")
            return None, True, None
        history_cursor, scope_start = _resolve_history_scope(cursor, gen_arg, rounds)
        depth_limit = rounds
        if not history_cursor:
            return None, True, None
        _print_command_history(
            _active_chat_context(),
            include_logs,
            conversation=history_cursor.chat_session,
            start_at_turn=scope_start,
            depth=depth_limit,
        )
        return None, True, None
    elif sub_cmd == "prompt": # /s p[rompt]
        cursor = _require_current_cursor()
        cursor.log_command(command_full_text) # noqa
        gen_arg = sub_args.strip() or None
        prompt_cursor, _clip_turn = _history_cursor_for_gen_id(cursor, gen_arg)
        if not prompt_cursor:
            return None, True, None
        session = prompt_cursor.session
        parser_profile = prompt_cursor.parser_profile
        debug_prompt_list = session.get_llm_messages(
            current_turn=prompt_cursor.current_turn,
            parser=UnifiedToolIO(parser_profile) if parser_profile else None, # type: ignore
            debug_format=True,
            include_content=True) # Request content for the new column
    
        print(f"\n{Colors.HEADER}--- LLM Prompt Path (Debug) ---{Colors.RESET}")
        if not debug_prompt_list:
            print("  (No turns in prompt path)")
        else:
            segments_dict, explicit_none = prompt_cursor.get_system_message_segments()
            print("  System Message:")
            if explicit_none:
                print("    <removed (no system prompt will be sent)>")
            elif not segments_dict:
                print("    <None>")
            else:
                for key, value in segments_dict.items():
                    value_display = _clip_long_message(value) if value else "<empty>"
                    print(f"    {key}: '{value_display}'")
            tools_line = _format_tools_scope_header(prompt_cursor)
            print(f"  {tools_line}")
            # --- NEW: Table formatting ---
            print(f"  {'ID':<10} {'gen_id':<15} {'Msgs':<25} {'Content':<22} {'Flags':<20} {'Info'}")
            print(f"  {'-'*10} {'-'*15} {'-'*25} {'-'*22} {'-'*20} {'-'*30}")

            for item in debug_prompt_list:
                flags = item.get("flags", [])
                flags_str = f"[{','.join(flags)}]" if flags else ""
                msgs = item.get('msgs', [])
    
                # Add tool_calls and tool_results to the message list for display
                if item.get("has_tool_calls"): msgs.append("tool_calls")
                if item.get("has_tool_results"): msgs.append("tool_results")
    
                messages_str = f"[{','.join(msgs)}]" if msgs else ""
    
                info_list = item.get('info', [])
                info_str = info_list[0] if info_list else ""
    
                id_str = item.get('id', 'N/A') # This is the display_id
                gen_id_str = item.get('gen_id') or 'N/A' # This is the raw gen_id from the turn

                # New content preview column
                content_list = item.get('content_preview', [])
                content_preview_str = ""
                if content_list:
                    first_content = content_list[0].replace('\n', ' ').strip()
                    content_preview_str = (first_content[:19] + '…') if len(first_content) > 20 else first_content
    
                # For placeholders, the raw gen_id will be 'N/A'.
                # The `get_llm_messages` debug output doesn't directly expose the node object to get gen_id_or_parent.
                # However, the `id` (display_id) already contains the correct hierarchical path.
                # The `gen_id` column should show the node's actual gen_id or 'off:...' for placeholders.
                # The logic in `get_llm_messages` needs to provide this. Let's assume it does and just display it.
                print(f"  {id_str:<10} {gen_id_str:<15} {messages_str:<25} {content_preview_str:<22} {flags_str:<20} {info_str}")
        print(f"{Colors.HEADER}----------------------------------{Colors.RESET}")
        return None, True, None
        
    elif sub_cmd == "turn": # /s t
        cursor = _require_current_cursor()
        cursor.log_command(command_full_text)
        target_arg_raw = sub_args.strip()
        target_arg = target_arg_raw.lower()
        moved = False
        nav_cursor = cursor
        try:
            if not target_arg:
                print(f"{Colors.SYSTEM}Usage: /s t <up|down|next|prev|close|main|trim|gen_id>{Colors.RESET}")
                print(f"{Colors.SYSTEM}       /s t root_ctx [gen_id] [on|off]{Colors.RESET}")
                print(f"{Colors.SYSTEM}       /s t arch[ive] [gen_id] [on|off]{Colors.RESET}")
                print(f"{Colors.SYSTEM}       /s t sh[ow] [gen_id]{Colors.RESET}")
                return None, True, None
            first_token = target_arg.split(" ", 1)[0]

            if target_arg.startswith("c"): # close
                previous_label = nav_cursor.display_id()
                if nav_cursor.close_branch():
                    moved = True
                    print(f"{Colors.SYSTEM}Closed branch fork for [{previous_label}].{Colors.RESET}")
                else:
                    print(f"{Colors.TOOL_WARNING}Could not close branch from [{previous_label}].{Colors.RESET}")
            elif target_arg.startswith("d"): # down
                child_cursor = nav_cursor.child_cursor()
                if child_cursor:
                    nav_cursor = child_cursor
                    moved = True
                else:
                    print(f"{Colors.TOOL_WARNING}Cannot move down, no child turns.{Colors.RESET}")
            elif target_arg.startswith("u"): # up
                parent_cursor = nav_cursor.parent_cursor()
                if parent_cursor:
                    nav_cursor = parent_cursor
                    moved = True
                else:
                    print(f"{Colors.TOOL_WARNING}Cannot move up, already at the root.{Colors.RESET}")
            elif target_arg.startswith("n"): # next
                next_cursor = nav_cursor.next_sibling_cursor()
                if next_cursor:
                    nav_cursor = next_cursor
                    moved = True
                else:
                    print(f"{Colors.TOOL_WARNING}Cannot move next, this is the last sibling or no siblings exist.{Colors.RESET}")
            elif target_arg.startswith("p"): # prev
                prev_cursor = nav_cursor.prev_sibling_cursor()
                if prev_cursor:
                    nav_cursor = prev_cursor
                    moved = True
                else:
                    print(f"{Colors.TOOL_WARNING}Cannot move prev, this is the first sibling or no siblings exist.{Colors.RESET}")
            elif target_arg.startswith("m"): # main
                main_cursor = nav_cursor.main_leaf_cursor()
                if main_cursor:
                    nav_cursor = main_cursor
                    moved = True
                    print(f"{Colors.SYSTEM}Moved to the leaf of the main branch.{Colors.RESET}")
                else:
                    print(f"{Colors.ERROR}Could not find the main branch leaf.{Colors.RESET}")
            elif first_token in {"sh", "show"}:
                try:
                    parts = shlex.split(target_arg_raw)
                except ValueError as err:
                    print(f"{Colors.ERROR}Could not parse show arguments: {err}{Colors.RESET}")
                    return None, True, None
                gen_id = parts[1] if len(parts) > 1 else None
                target_cursor = nav_cursor
                if gen_id:
                    try:
                        target_cursor = nav_cursor.cursor_for_gen_id(gen_id)
                    except KeyError:
                        print(f"{Colors.ERROR}Turn with gen_id '{gen_id}' not found.{Colors.RESET}")
                        return None, True, None
                    except ValueError as err:
                        print(f"{Colors.ERROR}{err}{Colors.RESET}")
                        return None, True, None
                _print_single_turn(target_cursor)
                return None, True, None
            elif first_token == "m" or first_token.startswith("main_thread"):
                try:
                    parts = shlex.split(target_arg_raw)
                    gen_id = None
                    state = None
                    if len(parts) > 1:
                        if parts[1].lower() in ["on", "off"]:
                            state = parts[1].lower()
                        else:
                            gen_id = parts[1]
                            if len(parts) > 2 and parts[2].lower() in ["on", "off"]:
                                state = parts[2].lower()
                    
                    target_cursor = cursor.cursor_for_gen_id(gen_id) if gen_id else cursor
                    closest_fork_cursor = target_cursor.find_closest_fork()

                    if state:
                        closest_fork_cursor.set_main_thread(state == "on")
                        print(f"Main thread flag for turn '{closest_fork_cursor.display_id()}' set to {state}.")
                    else:
                        is_main = closest_fork_cursor.is_main_thread()
                        print(f"Main thread flag for turn '{closest_fork_cursor.display_id()}' is {'on' if is_main else 'off'}.")
                except Exception as e:
                    print(f"{Colors.ERROR}Error processing '/s t m': {e}{Colors.RESET}")
                return None, True, None
            elif first_token in {"root", "root_ctx", "r"}: # root_ctx
                try:
                    root_parts = shlex.split(target_arg_raw)
                except ValueError as err:
                    print(f"{Colors.ERROR}Could not parse root_ctx arguments: {err}{Colors.RESET}")
                    return None, True, None

                target_cursor_for_root = nav_cursor
                gen_id_arg = None
                state_arg = None

                if len(root_parts) > 1:
                    candidate = root_parts[1]
                    if candidate.lower() in ["on", "off"]:
                        state_arg = candidate.lower()
                    else:
                        gen_id_arg = candidate
                        if len(root_parts) > 2 and root_parts[2].lower() in ["on", "off"]:
                            state_arg = root_parts[2].lower()

                if gen_id_arg:
                    try:
                        target_cursor_for_root = nav_cursor.cursor_for_gen_id(gen_id_arg)
                    except KeyError:
                        print(f"{Colors.ERROR}Error: Turn with gen_id '{gen_id_arg}' not found.{Colors.RESET}")
                        return None, True, None
                    except ValueError as err:
                        print(f"{Colors.ERROR}{err}{Colors.RESET}")
                        return None, True, None

                turn_label = target_cursor_for_root.display_id(
                    turn=target_cursor_for_root.current_turn,
                    active_cursor=nav_cursor,
                )
                if state_arg is None:
                    print(f"{Colors.SYSTEM}Turn '{turn_label}' root_context is: {target_cursor_for_root.current_turn.root_context}{Colors.RESET}")
                else:
                    new_state = (state_arg == "on")
                    if target_cursor_for_root.current_turn.root_context != new_state:
                        target_cursor_for_root.current_turn.root_context = new_state
                        print(f"{Colors.SYSTEM}Set root_context for turn '{turn_label}' to {new_state}.{Colors.RESET}")
                    else:
                        print(f"{Colors.TOOL_WARNING}Turn '{turn_label}' already has root_context set to {new_state}.{Colors.RESET}")

                return None, True, None
            elif first_token in {"arch", "archive", "a"}:
                try:
                    arch_parts = shlex.split(target_arg_raw)
                except ValueError as err:
                    print(f"{Colors.ERROR}Could not parse archive arguments: {err}{Colors.RESET}")
                    return None, True, None

                target_cursor_for_arch = nav_cursor
                gen_id_arg = None
                state_arg = None

                if len(arch_parts) > 1:
                    candidate = arch_parts[1]
                    if candidate.lower() in ["on", "off"]:
                        state_arg = candidate.lower()
                    else:
                        gen_id_arg = candidate
                        if len(arch_parts) > 2 and arch_parts[2].lower() in ["on", "off"]:
                            state_arg = arch_parts[2].lower()

                if gen_id_arg:
                    try:
                        target_cursor_for_arch = nav_cursor.cursor_for_gen_id(gen_id_arg)
                    except KeyError:
                        print(f"{Colors.ERROR}Error: Turn with gen_id '{gen_id_arg}' not found.{Colors.RESET}")
                        return None, True, None
                    except ValueError as err:
                        print(f"{Colors.ERROR}{err}{Colors.RESET}")
                        return None, True, None

                turn_label = target_cursor_for_arch.display_id(
                    turn=target_cursor_for_arch.current_turn,
                    active_cursor=nav_cursor,
                )
                if state_arg is None:
                    print(f"{Colors.SYSTEM}Turn '{turn_label}' archived flag is: {target_cursor_for_arch.current_turn.is_archived}{Colors.RESET}")
                else:
                    new_state = (state_arg == "on")
                    if target_cursor_for_arch.current_turn.is_archived != new_state:
                        target_cursor_for_arch.current_turn.is_archived = new_state
                        print(f"{Colors.SYSTEM}Set archived flag for turn '{turn_label}' to {new_state}.{Colors.RESET}")
                    else:
                        print(f"{Colors.TOOL_WARNING}Turn '{turn_label}' archive flag is already {new_state}.{Colors.RESET}")

                return None, True, None
            elif first_token == "t" or first_token.startswith("trim"): # trim
                try:
                    trim_parts = shlex.split(target_arg_raw)
                except ValueError as err:
                    print(f"{Colors.ERROR}Could not parse trim arguments: {err}{Colors.RESET}")
                    return None, True, None
                trim_gen_id = trim_parts[1] if len(trim_parts) > 1 else None
                target_cursor_for_trim = nav_cursor
                if trim_gen_id:
                    try:
                        target_cursor_for_trim = nav_cursor.cursor_for_gen_id(trim_gen_id)
                    except KeyError:
                        print(f"{Colors.ERROR}Error: Turn with gen_id '{trim_gen_id}' not found.{Colors.RESET}")
                        return None, True, None
                    except ValueError as err:
                        print(f"{Colors.ERROR}{err}{Colors.RESET}")
                        return None, True, None
                new_cursor_for_trim, commands_to_reverse, success, message = target_cursor_for_trim.trim_turns(1)
                if success:
                    print(f"{Colors.SYSTEM}{message}{Colors.RESET}")
                    if commands_to_reverse:
                        print(f"{Colors.TOOL_WARNING}Warning: The trimmed branch contained state changes that may need to be manually reversed.{Colors.RESET}")
                        for cmd in commands_to_reverse:
                            if cmd.cmd_type == Command.PARAM_CHANGE:
                                print(f"  - Revert '{cmd.data.get('change')}' from '{cmd.data.get('new_value')}' to '{cmd.data.get('old_value', '<unknown>')}'")
                            elif cmd.cmd_type == Command.STATE_CHANGE:
                                print(f"  - Revert state '{cmd.data.get('change')}' from '{cmd.data.get('value')}'")
                    if new_cursor_for_trim:
                        nav_cursor = new_cursor_for_trim
                        cursor = new_cursor_for_trim
                        _set_active_cursor(new_cursor_for_trim)
                else:
                    print(f"{Colors.ERROR}{message}{Colors.RESET}")
                return None, True, nav_cursor
            else: # gen_id
                try:
                    nav_cursor = nav_cursor.cursor_for_gen_id(target_arg_raw)
                    moved = True
                except KeyError:
                    print(f"{Colors.ERROR}Turn with gen_id '{target_arg}' not found.{Colors.RESET}")
                except ValueError as err:
                    print(f"{Colors.ERROR}{err}{Colors.RESET}")
        except Exception as e:
            print(f"Error during traversal: {e}")
            _store_exception_traceback_if_clear(e)

        if moved and nav_cursor.current_turn is not cursor.current_turn:
            cursor = nav_cursor
            _set_active_cursor(nav_cursor)
            active_turn = cursor.current_turn
            print_session_tree_summary(cursor, active=active_turn, show_logs=False) # noqa
            new_display_id = cursor.display_id(active_cursor=cursor) # noqa
            print(f"{Colors.SYSTEM}Active turn is now: '{new_display_id}' (gen_id: {active_turn.gen_id_or_parent}){Colors.RESET}")
            return None, True, cursor # Return the new turn as active
        return None, True, None

    elif sub_cmd == "conversation": # /s c
        base_cursor = _require_current_cursor()
        _update_active_conversation_state(base_cursor)
        base_cursor.log_command(command_full_text)
        session = base_cursor.session
        conversations = session.conversations
        if not conversations:
            print(f"{Colors.ERROR}No conversations available in this session.{Colors.RESET}")
            return None, True, None
        stripped = sub_args.strip()
        if not stripped:
            print(f"\n--- Chat Conversations in Session: {session.name} ---")
            for idx, conv in enumerate(conversations, start=1):
                _print_conversation_summary(session, conv, idx, active=(conv is base_cursor.chat_session), single=False)
                conv_cursor, temp_context = _conversation_cursor_for_index(session, conv, base_cursor)
                target_cursor = conv_cursor
                preferred_turn = _preferred_summary_turn(conv, use_active=False)
                context_for_turns = temp_context or _active_chat_context()
                if preferred_turn:
                    try:
                        target_cursor = conv_cursor.clone_at(preferred_turn)
                    except Exception:
                        target_cursor = conv_cursor.clone_at(conv.root_turn)
                display_cursor = target_cursor.clone_at(target_cursor.current_turn) if target_cursor.current_turn else target_cursor
                if preferred_turn:
                    # Limit active path display to the chosen summary turn
                    display_cursor.active_path_for_llm = lambda: [preferred_turn]  # type: ignore
                print_current_session_info(
                    display_cursor,
                    show_messages=True,
                    show_forks=False,
                    show_logs=False,
                    clear_first=False,
                    role_preview_chars=80,
                    active_only=True,
                )
            print("---")
            return None, True, None

        try:
            tokens = shlex.split(sub_args)
        except ValueError as err:
            print(f"{Colors.ERROR}Could not parse arguments: {err}{Colors.RESET}")
            return None, True, None

        title_update: Optional[str] = None
        cleaned_tokens: List[str] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--title":
                if i + 1 < len(tokens):
                    title_update = tokens[i + 1]
                    i += 2
                    continue
                title_update = ""
                i += 1
                continue
            if tok.startswith("--title="):
                title_update = tok.split("=", 1)[1]
                i += 1
                continue
            cleaned_tokens.append(tok)
            i += 1
        tokens = cleaned_tokens

        if not tokens:
            print(f"{Colors.ERROR}Conversation index required. Use '/s c' to list options.{Colors.RESET}")
            return None, True, None

        index_token = tokens.pop(0)
        try:
            conv_index = int(index_token) - 1
        except ValueError:
            print(f"{Colors.ERROR}Invalid conversation index '{index_token}'.{Colors.RESET}")
            return None, True, None

        if conv_index < 0 or conv_index >= session.get_conversations_count():
            print(f"{Colors.ERROR}Conversation index {conv_index + 1} out of range.{Colors.RESET}")
            return None, True, None

        target_conversation = session.get_conversation(conv_index)
        if title_update is not None:
            title_display = title_update if title_update else "<empty>"
            target_conversation.title = title_update
            print(f"{Colors.SYSTEM}Conversation {conv_index + 1} title set to '{title_display}'.{Colors.RESET}")
        if not tokens:
            conv_cursor, temp_context = _conversation_cursor_for_index(session, target_conversation, base_cursor)
            _print_conversation_summary(session, target_conversation, conv_index + 1, active=(target_conversation is base_cursor.chat_session), single=True)
            context_for_turns = temp_context or _active_chat_context()
            preferred_turn = _preferred_summary_turn(
                target_conversation,
                use_active=True,
                active_cursor=base_cursor if target_conversation is base_cursor.chat_session else None,
            )
            warn_msg = None
            target_cursor = conv_cursor
            if preferred_turn:
                try:
                    target_cursor = conv_cursor.clone_at(preferred_turn)
                except Exception:
                    target_cursor = conv_cursor.clone_at(target_conversation.root_turn)
            if warn_msg and (target_conversation is base_cursor.chat_session):
                print(warn_msg)
            display_cursor = target_cursor.clone_at(target_cursor.current_turn) if target_cursor.current_turn else target_cursor
            if preferred_turn:
                display_cursor.active_path_for_llm = lambda: [preferred_turn]  # type: ignore
            print_current_session_info(
                display_cursor,
                show_messages=True,
                show_forks=False,
                show_logs=False,
                clear_first=False,
                role_preview_chars=80,
                active_only=True,
            )
            return None, True, None

        action_token = tokens.pop(0).lower()
        remainder = " ".join(tokens)

        if action_token in {"set", "s"}:
            cursor = _bootstrap_cursor_for_session(session, conversation=target_conversation)
            _set_active_cursor(cursor)
            _print_conversation_summary(session, target_conversation, conv_index + 1, active=True, single=True)
            print(f"{Colors.SYSTEM}Switched to conversation {conv_index + 1}.{Colors.RESET}")
            print_current_session_info(
                cursor,
                show_messages=True,
                show_forks=False,
                clear_first=False,
                role_preview_chars=80,
                active_only=True,
            )
            await _check_engine_sync_status(cursor)
            _update_active_conversation_state(cursor)
            return None, True, cursor
        elif action_token in {"del", "d", "delete"}:
            new_cursor = _delete_conversation_at_index(session, conv_index)
            if new_cursor:
                cursor = new_cursor
                _set_active_cursor(cursor)
                _update_active_conversation_state(cursor)
            return None, True, cursor
        elif action_token in {"insert", "i"}:
            new_cursor = _insert_conversation_at_index(session, conv_index, _chat_session_template(), title=title_update)
            cursor = new_cursor
            _set_active_cursor(cursor)
            _update_active_conversation_state(cursor)
            return None, True, cursor
        elif action_token in {"h", "hfs", "hfsl", "hs", "hf", "hfl", "hl", "ch", "chl"}:
            _run_conversation_history_action(action_token, target_conversation, base_cursor, remainder)
            return None, True, None
        else:
            print(f"{Colors.ERROR}Unknown conversation action '{action_token}'.{Colors.RESET}")
            return None, True, None
    elif sub_cmd in {"replay", "re"} or sub_cmd == "r":
        cursor = _require_current_cursor()
        cursor.log_command(command_full_text)
        if _session_requires_sync(cursor.chat_session):
            _warn_session_sync_required()
            return None, True, None
        new_cursor = await _handle_replay_command(sub_args, cursor, pt_session)
        return None, True, new_cursor
    return None, True, None

async def _handle_replay_command(args_str: str, cursor: ChatCursor, pt_session: "PromptSession") -> Optional[ChatCursor]:
    """Handles the /s replay command with modes and argparse."""
    class NoExitArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            # The default argparse behavior on error is to print usage and exit.
            # We override it to raise an exception that can be caught locally.
            raise argparse.ArgumentError(None, message)

    parser = NoExitArgumentParser(prog="/s replay", description="Replay a conversation.", add_help=False)
    subparsers = parser.add_subparsers(dest="mode", required=False)

    # one-branch mode
    parser_one_branch = subparsers.add_parser("one-branch", help="Replay a single branch.", add_help=False)
    parser_one_branch.add_argument("conversation_index", type=int, help="The 1-based index of the conversation to replay.")
    parser_one_branch.add_argument("--from", dest="from_gen_id", help="Start replaying from the specified turn.")
    parser_one_branch.add_argument("--root_ctx", dest="root_ctx", nargs="?", const=True, default=False, help="Mark the destination cursor as root_ctx before replay (optional flag).")
    parser_one_branch.add_argument("--debug", action="store_true", dest="debug", help="Show verbose replay trace.")

    # all-down mode
    parser_all_down = subparsers.add_parser("all-down", help="Replay the entire conversation.", add_help=False)
    parser_all_down.add_argument("conversation_index", type=int, help="The 1-based index of the conversation to replay.")
    parser_all_down.add_argument("--from", dest="from_gen_id", help="Start replaying from the specified turn (must have root_ctx).")
    parser_all_down.add_argument("--root_ctx", dest="root_ctx", nargs="?", const=True, default=False, help="Mark the destination cursor as root_ctx before replay (optional flag).")
    parser_all_down.add_argument("--debug", action="store_true", dest="debug", help="Show verbose replay trace.")
    
    # Help handling
    if not args_str.strip() or args_str.strip() == '?':
        _print_replay_cli_help()
        return None

    def _normalize_tokens(raw_tokens: List[str]) -> List[str]:
        modes = {"one-branch", "all-down"}
        if not raw_tokens:
            return []
        mode: Optional[str] = None
        conv_token: Optional[str] = None
        rest: List[str] = []

        if raw_tokens[0] in modes:
            mode = raw_tokens[0]
            if len(raw_tokens) > 1:
                conv_token = raw_tokens[1]
                rest = raw_tokens[2:]
        elif len(raw_tokens) >= 2 and raw_tokens[1] in modes:
            mode = raw_tokens[1]
            conv_token = raw_tokens[0]
            rest = raw_tokens[2:]
        else:
            # Try to find conversation index in first two tokens
            first, second = raw_tokens[0], raw_tokens[1] if len(raw_tokens) > 1 else ""
            if first.isdigit():
                conv_token = first
                rest = raw_tokens[1:]
            elif second.isdigit():
                conv_token = second
                # If first looks like mode, use it; otherwise keep as rest
                if first in modes:
                    mode = first
                    rest = raw_tokens[2:]
                else:
                    rest = raw_tokens[2:]
            else:
                conv_token = first
                rest = raw_tokens[1:]
        if not mode:
            mode = "all-down"
        args_list = [mode]
        if conv_token:
            args_list.append(conv_token)
        args_list.extend(rest)
        return args_list

    try:
        tokens = shlex.split(args_str)
        if not tokens:
            _print_replay_cli_help()
            return None

        args_to_parse = _normalize_tokens(tokens)
        args = parser.parse_args(args_to_parse)

    except SystemExit:
        # Argparse prints its own error, so we just return to prevent a crash.
        return None
    except (ValueError, argparse.ArgumentError) as e:
        print(f"{Colors.ERROR}Error parsing arguments: {e}{Colors.RESET}")
        _print_replay_cli_help()
        return None

    # If no mode was provided (backward compat), default to all-down
    if not hasattr(args, 'mode') or not args.mode:
        args.mode = 'all-down'
        
    driver_index = args.conversation_index - 1

    if not (0 <= driver_index < cursor.session.get_conversations_count()):
        print(f"{Colors.ERROR}Invalid conversation index '{args.conversation_index}'.{Colors.RESET}")
        return None

    driver_conversation = cursor.session.get_conversation(driver_index)
    
    from_gen_id = getattr(args, 'from_gen_id', None)
    
    if args.mode == "one-branch":
        if from_gen_id:
            driver_start_turn = cursor.session.get_turn_by_gen_id(from_gen_id, driver_conversation)
        else:
            driver_start_turn = driver_conversation.root_turn
    else: # all-down
        if from_gen_id:
            driver_start_turn = cursor.session.get_turn_by_gen_id(from_gen_id, driver_conversation)
            if driver_start_turn and not getattr(driver_start_turn, "root_context", False):
                print(f"{Colors.ERROR}All-down replay with --from requires the source turn to have root_ctx set.{Colors.RESET}")
                return None
        else:
            driver_start_turn = driver_conversation.root_turn


    if not driver_start_turn:
        print(f"{Colors.ERROR}Could not resolve replay start turn for conversation {driver_index + 1}.{Colors.RESET}")
        return None

    if not _active_chat_context():
        print(f"{Colors.ERROR}Replay requires an active chat context to resolve the destination cursor.{Colors.RESET}")
        return None

    target_cursor = _require_current_cursor()

    # Compare source and destination environments before starting replay
    if driver_conversation and target_cursor.chat_session:
        _compare_and_print_session_diffs(driver_conversation, target_cursor.chat_session)

    if args.root_ctx:
        if args.root_ctx is not True:
            print(f"{Colors.TOOL_WARNING}Ignoring root_ctx target '{args.root_ctx}'; flag now only marks the current destination turn.{Colors.RESET}")
        if target_cursor.current_turn:
            target_cursor.set_root_context(True)
            print(f"{Colors.SYSTEM}Destination turn marked as root_ctx on {target_cursor.display_id()}.{Colors.RESET}")

    mode_label = "one-branch" if args.mode == "one-branch" else "all-down"
    engine_name = _effective_engine_name_for_session(target_cursor.chat_session)
    print(
        f"{Colors.SYSTEM}Replaying conversation {args.conversation_index} in mode: "
        f"{mode_label} (engine: {engine_name}).{Colors.RESET}"
    )

    driver_context = ChatContext(cursor.session, chat_session=driver_conversation, toolbox=toolbox)
    driver_cursor = driver_context.register_cursor_for_turn(driver_start_turn, make_active=False) or driver_context.active_cursor

    dest_cursor_after_replay = None
    ctx = cursor.context
    scope = _scope_for_cursor(cursor)
    bag_dict = _resolve_bag_dict(scope, ctx)
    if bag_dict is not None:
        bag_dict["replay_start_time"] = time.time()
        bag_dict["replay_debug"] = bool(getattr(args, "debug", False))
    try:
        if args.mode == "one-branch":
            dest_cursor_after_replay = await _replay_session_branch(
                driver_cursor=driver_cursor,
                dest_cursor=target_cursor,
                pt_session=pt_session,
                replay_debug=bool(getattr(args, "debug", False)),
            )
        elif args.mode == "all-down":
            dest_cursor_after_replay = await _replay_all_down(
                driver_cursor=driver_cursor,
                dest_cursor=target_cursor,
                pt_session=pt_session,
                replay_debug=bool(getattr(args, "debug", False)),
            )
    finally:
        if bag_dict is not None:
            bag_dict.pop("replay_start_time", None)
            bag_dict.pop("replay_debug", None)

    # After replay, clear any pending auto-iteration flag to prevent the main loop from continuing.
    context = _active_chat_context()
    if context:
        scope = _require_live_chat_scope(cursor)
        while True:
            if not scope.consume_auto_iteration():
                break
        dropped = _cleanup_cursor_registry(scope, context)
        if dropped:
            print(f"{Colors.SYSTEM}Replay: dropped {dropped} cursor handle(s).{Colors.RESET}")

    return dest_cursor_after_replay.descend_to_leaf() if dest_cursor_after_replay else None


async def handle_command(
    user_input: str,
    cursor: ChatCursor,
    pt_session: "PromptSession",
) -> Tuple[bool, bool, Optional[ChatCursor]]: # type: ignore
    global current_config, session_control, toolbox

    if not session_control:
        print("Error: Session manager not initialized.")
        return True, False, None # type: ignore
    
    command_full = user_input[1:].strip()
    parts = command_full.split(" ", 1) # noqa
    cmd_prefix = parts[0].lower()
    # args_str = parts[1] if len(parts) > 1 else "" # Handled by sub-handlers

    live_scope_required = not (
        cmd_prefix in {"help", "?"}
        or cmd_prefix.startswith("q")
        or cmd_prefix.startswith("config")
        or cmd_prefix.startswith("rl")
        or (cmd_prefix.startswith("s") and not (cmd_prefix.startswith("sy") or cmd_prefix.startswith("sm")))
    )
    if live_scope_required:
        try:
            _require_live_chat_scope(cursor)
        except Exception as exc:
            print(f"{Colors.ERROR}{exc}{Colors.RESET}")
            return True, False, None

    if _session_requires_sync(cursor.chat_session):
        if cmd_prefix in {"cg", "g", "generate", "r", "retry", "raw"}:
            _warn_session_sync_required()
            return True, False, None

    if cmd_prefix == "help" or cmd_prefix == "?": 
        print_help()
        base_cursor = cursor
        base_cursor.log_command(user_input)

        return True, False, None
    elif cmd_prefix.startswith("q"):
        return True, True, None # /q or /quit
    elif cmd_prefix.startswith("cg"): # Manual continue
        base_cursor = cursor
        base_cursor.add_continuation_turn()
        base_cursor.log_command(user_input)

        return False, False, None # Trigger inference
    elif cmd_prefix.startswith("cb"):
        base_cursor = cursor
        cb_args_match = re.match(r"cb(?:lose_batches)?\s*(.*)", command_full, re.IGNORECASE)
        cb_args = cb_args_match.group(1).strip() if cb_args_match else ""
        batch_gen_id = cb_args.split()[0] if cb_args else None
        ctx = base_cursor.context
        scope = _require_live_chat_scope(base_cursor)
        entry = _pop_batch_stack_entry(ctx, batch_gen_id, scope=scope)
        base_cursor.log_command(user_input)
        if not entry:
            print(f"{Colors.ERROR}No batch fork found to close.{Colors.RESET}")
            return True, False, None
        owner_cursors = entry.get("owner_cursors") or []
        candidates: List[ChatCursor] = []
        for c in owner_cursors + [base_cursor]:
            if c and c not in candidates:
                candidates.append(c)
        closed_cursor: Optional[ChatCursor] = None
        if batch_gen_id:
            for candidate in candidates:
                closed_cursor = candidate.close_batches_by_gen_id(batch_gen_id, make_active=False)
                if closed_cursor:
                    break
        else:
            for candidate in candidates:
                closed_cursor = candidate.close_batches(make_active=False)
                if closed_cursor:
                    break
        if not closed_cursor:
            bag_dict = _resolve_bag_dict(scope, ctx)
            if bag_dict is not None:
                bag_dict.setdefault("batch_stack", []).append(entry)
            print(f"{Colors.ERROR}No batch fork found to close.{Colors.RESET}")
            return True, False, None
        if ctx:
            try:
                scope.set_active_cursor(closed_cursor)
            except Exception:
                pass
        dropped = _drop_batch_entry_handles(entry, ctx=ctx, keep_cursor=closed_cursor, scope=scope)
        if dropped == 0:
            dropped = _drop_batch_entry_cursors(entry, ctx=ctx, keep_cursor=closed_cursor, scope=scope)
        _set_active_cursor(closed_cursor)
        print(f"{Colors.SYSTEM}Closed batch forks. Dropped {dropped} batch cursor(s).{Colors.RESET}")
        return True, False, closed_cursor
    elif cmd_prefix.startswith("g"):
        gen_args_match = re.match(r"g(?:generate)?\s*(.*)", command_full, re.IGNORECASE)
        gen_args = gen_args_match.group(1).strip() if gen_args_match else ""
        base_cursor = cursor
        if gen_args.startswith("ao"):
            sub_args = gen_args[len("ao"):].strip()
            selected_overrides = await _resolve_adapter_targets_1based(
                sub_args,
                "Select one or more adapters for adapter-override batch",
                allow_multiple=True,
                normalize=False,
                cursor=base_cursor,
            )
            if selected_overrides:
                override_prompt = await pt_session.prompt_async("Enter prompt for adapter override batch: ")
                if override_prompt.strip():
                    prompt_list = [override_prompt] * len(selected_overrides)
                    new_cursor, suppress = await _handle_general_batch_generation(
                        base_cursor,
                        pt_session,
                        is_concurrent=False,
                        allow_auto_retry=True,
                        prepared_prompts=prompt_list,
                        override_adapters=list(selected_overrides),
                        command_text_override=f"/g ao (adapters: {', '.join(selected_overrides)})",
                    )
                    return suppress, False, new_cursor
                else:
                    print("Adapter override cancelled; no prompt entered.")
                    return True, False, None
            else:
                print(f"{Colors.SYSTEM}Adapter override cancelled.{Colors.RESET}")
                return True, False, None
        if gen_args.startswith("bc"):
            new_cursor, suppress = await _handle_general_batch_generation(base_cursor, pt_session, is_concurrent=True, command_text_override=command_full)
            return suppress, False, new_cursor
        elif gen_args.startswith("b"):
            new_cursor, suppress = await _handle_general_batch_generation(base_cursor, pt_session, is_concurrent=False, command_text_override=command_full)
            return suppress, False, new_cursor
        else:
            print(f"{Colors.ERROR}Unknown generate command. Try '/g b', '/g bc', or '/g ao'.{Colors.RESET}")
            return True, False, None
    elif cmd_prefix.startswith("a") and not cmd_prefix.startswith("addmsg"):
        adapter_args_match = re.match(r"a(?:dapter)?\s*(.*)", command_full, re.IGNORECASE)
        adapter_args = adapter_args_match.group(1).strip() if adapter_args_match else ""
        base_cursor = cursor
        new_cursor, suppress = await _handle_adapter_command(adapter_args, base_cursor, pt_session)
        return suppress, False, new_cursor
    elif cmd_prefix.startswith("sw"):
        sw_args_match = re.match(r"sw(?:itch)?\s*(.*)", command_full, re.IGNORECASE)
        sw_args = sw_args_match.group(1).strip() if sw_args_match else ""
        new_cursor, suppress = _handle_switch_command(sw_args, cursor)
        return suppress, False, new_cursor
    elif cmd_prefix.startswith("br"):
        br_args_match = re.match(r"br(?:anch)?\s*(.*)", command_full, re.IGNORECASE)
        br_args = br_args_match.group(1).strip() if br_args_match else ""
        parts = [p for p in br_args.split() if p]
        if len(parts) < 2 or parts[0] not in ("--gen_id", "-g"):
            print(f"{Colors.ERROR}Usage: /br --gen_id <turn_gen_id> [alias]{Colors.RESET}")
            return True, False, None
        gen_id = parts[1]
        alias = parts[2] if len(parts) > 2 else None
        ctx = cursor.context
        if not ctx:
            print(f"{Colors.ERROR}No active chat context; cannot resurrect branch.{Colors.RESET}")
            return True, False, None
        try:
            scope = _require_live_chat_scope(cursor)
            resurrected = scope.resurrect_cursor_for_gen_id(gen_id, alias=alias, make_active=True)
            _set_active_cursor(resurrected)
            print(f"{Colors.SYSTEM}Resurrected cursor for '{gen_id}' as '{resurrected.label}'.{Colors.RESET}")
            return True, False, resurrected
        except Exception as exc:
            print(f"{Colors.ERROR}Failed to resurrect '{gen_id}': {exc}{Colors.RESET}")
            return True, False, None
    elif cmd_prefix.startswith("try"):
        try_args_match = re.match(r"try\s*(.*)", command_full, re.IGNORECASE)
        try_args = try_args_match.group(1).strip() if try_args_match else ""
        base_cursor = cursor
        
        # Sub-command parsing
        if try_args.startswith("--"):
            parts = try_args.split(" ", 1)
            sub_command = parts[0]
            sub_args = parts[1] if len(parts) > 1 else ""

            if sub_command == "--list":
                ctx = base_cursor.context
                if not ctx:
                    print(f"{Colors.ERROR}No active context to list try-out anchors.{Colors.RESET}")
                else:
                    scope = _require_live_chat_scope(base_cursor)
                    anchors = scope.try_out_anchors_snapshot()
                    if not anchors:
                        print("No currently active try-out anchors.")
                    else:
                        print("Active try-out anchors:")
                        for i, anchor in enumerate(anchors):
                            print(f"  {i+1}. {anchor.anchor_name} (kind: {anchor.kind}, branches: {len(anchor.try_out_turns)}, at: {anchor.anchor_turn.gen_id})")
            
            elif sub_command == "--find":
                await _find_resurrectable_anchors(base_cursor)

            elif sub_command == "--resurrect":
                ctx = base_cursor.context
                if not ctx:
                    print(f"{Colors.ERROR}No active context available.{Colors.RESET}")
                    return True, False, base_cursor
                if not sub_args:
                    print(f"Usage: /try --resurrect <name|num>")
                    return True, False, base_cursor
                
                target_name = ""
                if sub_args in LAST_ENUMERATED_SESSIONS:
                    target_name = LAST_ENUMERATED_SESSIONS[sub_args]
                else:
                    target_name = sub_args
                
                try:
                    scope = _require_live_chat_scope(base_cursor)
                    resurrected = scope.resurrect_try_out_anchor(target_name)
                    if resurrected:
                        print(f"Successfully resurrected anchor '{resurrected.anchor_name}'. It is now active.")
                        print("You can navigate to its branches using /s t <gen_id>.")
                except ValueError as e:
                    print(f"{Colors.ERROR}{e}{Colors.RESET}")

            else:
                print(f"{Colors.ERROR}Unknown subcommand for /try: {sub_command}{Colors.RESET}")
                print("Usage: /try [--list|--find|--resurrect <name>|new_anchor_name]")

        else: # Default behavior: start a new try_out
            anchor_name = try_args or _generate_try_out_name()
            ctx = base_cursor.context
            if not ctx:
                print(f"{Colors.ERROR}No active context to start try-out anchor.{Colors.RESET}")
            else:
                try:
                    scope = _require_live_chat_scope(base_cursor)
                    anchor = scope.start_try_out_anchor(
                        anchor_name,
                        base_cursor.head,
                        origin_cursor=base_cursor,
                    )
                    _, try_cursor = base_cursor.add_try_out(anchor=anchor)
                    try:
                        scope.set_active_cursor(try_cursor)
                    except Exception:
                        pass
                    _set_active_cursor(try_cursor)
                    print(f"Started new try-out anchor '{anchor_name}' and switched to new branch: {try_cursor.display_id()}")
                except ValueError as e:
                    print(f"{Colors.ERROR}{e}{Colors.RESET}")
        return True, False, _require_current_cursor()
    elif cmd_prefix.startswith("mt"):
        mt_args_match = re.match(r"mt\s*(.*)", command_full, re.IGNORECASE)
        mt_args = mt_args_match.group(1).strip() if mt_args_match else ""
        base_cursor = cursor
        parts = mt_args.split()
        if len(parts) == 2:
            anchor_name, try_index_str = parts
            try:
                try_index = int(try_index_str) -1
                scope = _require_live_chat_scope(base_cursor)
                anchor = scope.get_try_out_anchor(anchor_name)
                if not anchor or not (0 <= try_index < len(anchor.try_out_turns)):
                    print("Invalid anchor name or try index.")
                else:
                    turn = anchor.try_out_turns[try_index]
                    print(f"Main thread flag for try-out {try_index + 1} of anchor '{anchor_name}' is {'on' if turn.main_thread else 'off'}.")
            except (ValueError, IndexError):
                print("Invalid try index.")
        else:
            print(f"Main thread flag for current turn is {'on' if base_cursor.is_main_thread() else 'off'}.")
        return True, False, None
    elif cmd_prefix.startswith("ct"):
        ct_args_match = re.match(r"ct\s*(.*)", command_full, re.IGNORECASE)
        ct_args = ct_args_match.group(1).strip() if ct_args_match else ""
        base_cursor = cursor
        
        parts = ct_args.split()
        anchor_name = parts[0] if parts else None
        
        dist_mode = "keep"
        main_thread_index = None

        if "--m" in parts:
            m_index = parts.index("--m")
            if m_index + 1 < len(parts):
                dist_mode_arg = parts[m_index+1]
                if dist_mode_arg == "all":
                    dist_mode = "all"
                elif dist_mode_arg == "none":
                    dist_mode = "none"
                else:
                    try:
                        main_thread_index = int(dist_mode_arg) - 1
                        dist_mode = "index"
                    except ValueError:
                        print("Invalid value for --m flag.")
                        return True, False, None
        
        try:
            ctx = base_cursor.context
            scope = _require_live_chat_scope(base_cursor)
            if not ctx:
                print(f"{Colors.ERROR}No active context to close try-out anchors.{Colors.RESET}")
            elif not anchor_name:
                print("Please specify an anchor name.")
            else:
                closed_cursor = _ensure_registered_cursor(
                    scope.close_try_out_anchor(
                        anchor_name,
                        dist_mode=dist_mode,
                        main_thread_index=main_thread_index,
                    )
                )
                command_cursor = closed_cursor or base_cursor
                command_cursor.command_event(
                    user_input,
                    metadata={
                        "$Action": "close_try_out",
                        "$ActionArgs": {
                            "anchor_name": anchor_name,
                            "dist_mode": dist_mode,
                            "main_thread_index": main_thread_index,
                        },
                    },
                )
                _set_active_cursor(command_cursor)
                print(f"Closed try-out anchor '{anchor_name}'.")
        except ValueError as e:
            print(f"{Colors.ERROR}{e}{Colors.RESET}")

        return True, False, _require_current_cursor()
    elif cmd_prefix.startswith("s") and not (cmd_prefix.startswith("sy") or cmd_prefix.startswith("sm")):
        session_args_match = re.match(r"s(?:ession)?\s*(.*)", command_full, re.IGNORECASE)
        session_args = session_args_match.group(1).strip() if session_args_match else ""
        base_cursor = cursor
        new_session, suppress, new_cursor = await _handle_session_command(
            base_cursor,
            session_args,
            user_input,
            pt_session,
        )
        if new_session:
            # The session object itself might be new, but cursor logic is handled by new_cursor
            pass
        return suppress, False, new_cursor
    elif cmd_prefix.startswith("-"): # /-[x]
        num_to_delete_str = command_full[1:].strip()
        num_to_delete = 1
        if num_to_delete_str and num_to_delete_str.isdigit():
            num_to_delete = int(num_to_delete_str)
        if num_to_delete <= 0:
            print("Number of turns to delete must be a positive number.")
            return True, False, None

        # --- Delegate trimming logic via ChatCursor helper ---
        new_active_cursor, commands_to_reverse, success, message = cursor.trim_turns(num_to_delete)
        if success:
            print(f"{Colors.SYSTEM}{message}{Colors.RESET}")
            if commands_to_reverse:
                print(f"{Colors.TOOL_WARNING}Warning: The trimmed branch contained state changes that may need to be manually reversed.{Colors.RESET}")
                for cmd in commands_to_reverse:
                    if cmd.cmd_type == Command.PARAM_CHANGE:
                        print(f"  - Revert '{cmd.data.get('change')}' from '{cmd.data.get('new_value')}' to '{cmd.data.get('old_value', '<unknown>')}'")
                    elif cmd.cmd_type == Command.STATE_CHANGE:
                        print(f"  - Revert state '{cmd.data.get('change')}' from '{cmd.data.get('value')}'")
            if new_active_cursor:
                cursor = new_active_cursor
                _set_active_cursor(new_active_cursor)
        else:
            print(f"{Colors.ERROR}{message}{Colors.RESET}")
        cursor.log_command(user_input)
        return True, False, new_active_cursor
    elif cmd_prefix.startswith("cls"): # clear session
        clear_screen()
        session = cursor.session
        active_chat_session = cursor.chat_session or (session.conversations[0] if session.conversations else None)
        if not active_chat_session:
            print(f"{Colors.ERROR}No active conversation to clear.{Colors.RESET}")
            return True, False, None

        try:
            conv_index = session.conversations.index(active_chat_session)
        except ValueError:
            conv_index = 0

        # 1. Gather state from the branch we are about to reset
        retained_segments_dict, retained_system_removed = cursor.get_system_message_segments()
        retained_system_pairs = list(retained_segments_dict.items())
        loaded_adapters_resp = await call_api("get-loaded-adapters")
        loaded_adapters = loaded_adapters_resp.get("data", {}).get("adapters", []) # noqa
        retained_active_adapters = cursor.get_effective_adapters()

        template = _chat_session_template()
        current_params = _current_param_snapshot()
        default_params = copy.deepcopy(getattr(template, "initial_params", {}) or {})
        reset_diffs = _summarize_param_resets(current_params, default_params)
        current_system = cursor.effective_system_message()
        target_system = current_system
        current_adapters = cursor.get_effective_adapters() or ["__base__"]
        target_adapters = ["__base__"]
        state_diffs = _summarize_effective_state_resets(
            current_system,
            target_system,
            current_adapters,
            target_adapters,
            default_system=default_params.get("system_message"),
        )

        # 2. Create a fresh conversation and replace the current slot
        new_chat_session = session.conversation_from_template(
            template=template,
            title=getattr(active_chat_session, "title", None),
        )
        _seed_chat_session_flags(new_chat_session)
        session.conversations[conv_index] = new_chat_session

        cursor = _bootstrap_cursor_for_session(session, conversation=new_chat_session)
        _set_active_cursor(cursor)

        # 3. Reapply reconstructed state on the new conversation
        print(f"{Colors.SYSTEM}Reconstructing effective state...{Colors.RESET}")
        if reset_diffs or state_diffs:
            print(f"{Colors.SYSTEM}Reset settings for cleared conversation:{Colors.RESET}")
            for line in reset_diffs:
                print(f"  - {line}")
            for line in state_diffs:
                print(f"  - {line}")
        base_cursor = cursor
        if loaded_adapters:
            recorded = _record_loaded_adapter_commands(base_cursor, loaded_adapters, command_label="/cls (record load-adapter)")
            if recorded:
                print(f"  - {Colors.DIM}Recorded {recorded} loaded adapter(s) on cleared conversation root.{Colors.RESET}")
        if retained_system_removed:
            base_cursor.apply_system_message("set", None, command_text="/cls (retained system message)")
            print(f"  - {Colors.DIM}System message state retained as <removed>.{Colors.RESET}")
        elif retained_system_pairs:
            for idx, (_, segment_text) in enumerate(retained_system_pairs):
                op = "set" if idx == 0 else "add"
                base_cursor.apply_system_message(op, segment_text, command_text="/cls (retained system message)")
            print(f"  - {Colors.DIM}System Message retained ({len(retained_system_pairs)} segment(s)).{Colors.RESET}")
        if loaded_adapters:
            base_cursor.add_adapters_state("loaded_adapters", [a.get("name") for a in loaded_adapters])
            print(f"  - {Colors.DIM}{len(loaded_adapters)} Loaded Adapter(s) retained.{Colors.RESET}")
        else:
            print(f"  - {Colors.DIM}No loaded adapters reported by engine.{Colors.RESET}")
        base_cursor.log_command("/cls executed")
        print(f"{Colors.SYSTEM}Conversation {conv_index} reset to a clean slate.{Colors.RESET}")
        print_current_session_info(cursor)
        return True, False, cursor
    elif cmd_prefix.startswith("raw"):
        raw_args_match = re.match(r"raw\s*(.*)", command_full, re.IGNORECASE)
        raw_prompt_escaped = raw_args_match.group(1) if raw_args_match else ""
        # This command is self-contained and will handle its own API call and printing.
        base_cursor = cursor
        new_cursor = await _handle_raw_command(raw_prompt_escaped, base_cursor)
        return True, False, new_cursor
    elif cmd_prefix.startswith("addmsg"): # type: ignore
        msg_parts = command_full.split(" ", 2)
        if len(msg_parts) < 3:
            print("Usage: /addmsg <role> <content>")
            return True, False, None
        role, content = msg_parts[1], msg_parts[2]
        base_cursor = cursor
        base_cursor.command_event(user_input, metadata={"role": role, "content": content})
        base_cursor.add_message(role, content)

        print(f"Added manual '{role}' message turn.")
        return True, False, None
    elif cmd_prefix.startswith("r"):
        retry_parts = command_full.split(" ", 1)
        retry_gen_id = retry_parts[1].strip() if len(retry_parts) > 1 else ""
        base_cursor = cursor
        target_cursor = base_cursor
        if retry_gen_id:
            try:
                target_cursor = base_cursor.cursor_for_gen_id(retry_gen_id)
            except KeyError:
                print(f"{Colors.ERROR}Error: Turn with gen_id '{retry_gen_id}' not found.{Colors.RESET}")
                return True, False, None
            except ValueError as err:
                print(f"{Colors.ERROR}{err}{Colors.RESET}")
                return True, False, None

        resolved_cursor = _retry_target_cursor(target_cursor) or _resolve_non_placeholder_cursor(target_cursor)
        target_turn = resolved_cursor.current_turn if resolved_cursor else None
        if not target_turn:
            print(f"{Colors.ERROR}No turn available to retry.{Colors.RESET}")
            return True, False, None
        if not target_turn.data.get("user") and not target_turn.data.get("tool_results"):
            print(f"{Colors.ERROR}Retry is only supported on user/tool-result turns.{Colors.RESET}")
            return True, False, None

        # 1. Prune the conversation tree of the old response and subsequent turns.
        removed_descendants = _prune_descendants_after_turn(target_turn, cursor.session)
        if removed_descendants:
            print(f"{Colors.SYSTEM}Removed {removed_descendants} descendant turn(s) beneath {resolved_cursor.display_id()}.{Colors.RESET}")

        # 2. Clear stale assistant output/metrics from the target turn itself.
        _clear_turn_outputs_for_retry(target_turn)
        if target_turn.is_archived:
            target_turn.is_archived = False
        
        # 3. Log the /retry command and set the active cursor.
        resolved_cursor.log_command(user_input)
        _set_active_cursor(resolved_cursor)
        
        print(f"{Colors.SYSTEM}Retrying turn {resolved_cursor.display_id()}...{Colors.RESET}")

        # 4. Return control to the main loop to execute inference.
        # This ensures the auto-execution loop (`while should_continue:`) is engaged.
        return False, False, resolved_cursor
    elif cmd_prefix.startswith("echo"):
        usage_msg = "Usage: /echo [--gen_id <gen_id>] [--r] [text]"
        try:
            tokens = shlex.split(command_full)
        except ValueError as err:
            print(f"{Colors.ERROR}Error parsing /echo arguments: {err}{Colors.RESET}")
            print(usage_msg)
            return True, False, None

        args = tokens[1:] if tokens and tokens[0].lstrip().startswith("/echo") else tokens
        gen_id_arg: Optional[str] = None
        remove_only = False
        note_tokens: List[str] = []
        idx = 0
        while idx < len(args):
            part = args[idx]
            if part == "--gen_id":
                if idx + 1 >= len(args):
                    print(usage_msg)
                    return True, False, None
                gen_id_arg = args[idx + 1]
                idx += 2
                continue
            if part == "--r":
                remove_only = True
                idx += 1
                continue
            note_tokens = args[idx:]
            break
        note_text = " ".join(note_tokens).strip()

        base_cursor = cursor
        if gen_id_arg:
            try:
                target_cursor = base_cursor.cursor_for_gen_id(gen_id_arg)
            except KeyError:
                print(f"{Colors.ERROR}Error: Turn with gen_id '{gen_id_arg}' not found.{Colors.RESET}")
                return True, False, None
            except ValueError as err:
                print(f"{Colors.ERROR}{err}{Colors.RESET}")
                return True, False, None
        else:
            target_cursor = base_cursor

        target_turn = target_cursor.current_turn if target_cursor else None
        if not target_turn:
            print(f"{Colors.ERROR}No turn available for /echo.{Colors.RESET}")
            return True, False, None
        if not remove_only and not note_text:
            print(usage_msg)
            return True, False, None

        try:
            removed_cmds = target_cursor.remove_commands_in_turn(target_turn, _is_echo_command)
        except ValueError as err:
            print(f"{Colors.ERROR}{err}{Colors.RESET}")
            return True, False, None

        turn_label = target_cursor.display_id(target_turn)
        if remove_only:
            if removed_cmds:
                print(f"{Colors.SYSTEM}Removed {len(removed_cmds)} echo command(s) from turn {turn_label}.{Colors.RESET}")
            else:
                print(f"{Colors.TOOL_WARNING}No echo commands found on turn {turn_label}.{Colors.RESET}")
            return True, False, None

        try:
            target_cursor.set_replace_command_in_turn(
                target_turn,
                user_input,
                _is_echo_command,
                metadata={"$Action": "echo", "$ActionArgs": {"text": note_text}, "text": note_text},
                insert_at=0,
            )
        except ValueError as err:
            print(f"{Colors.ERROR}{err}{Colors.RESET}")
            return True, False, None

        suffix = f" {Colors.DIM}(turn {turn_label}){Colors.RESET}" if gen_id_arg else ""
        print(f"{Colors.ECHO}[echo]{Colors.RESET} {note_text}{suffix}")
        return True, False, None
    elif cmd_prefix.startswith("sm") or cmd_prefix.startswith("systemmessage"):
        sys_args_match = re.match(r"(?:sm|systemmessage)(\s+.*)?", command_full, re.IGNORECASE)
        raw_args = sys_args_match.group(1) if sys_args_match else None

        def _print_system_message_segments(target: ChatCursor):
            segments, explicit_none = target.get_system_message_segments()
            if explicit_none:
                print("System message: <removed (no system prompt)>")
                return
            if not segments:
                print("System message: <None>")
                return
            for key, value in segments.items():
                preview = _clip_long_message(value) if value else ""
                preview_display = preview if preview else "<empty>"
                print(f"{key}: '{preview_display}'")

        def _print_system_message_scope(target: ChatCursor):
            segments, explicit_none = target.get_system_message_segments()
            effective = target.effective_system_message()
            default_sys = None
            if target.chat_session and isinstance(getattr(target.chat_session, "initial_params", None), dict):
                default_sys = target.chat_session.initial_params.get("system_message")
            if default_sys is None:
                default_sys = current_config.get("default_system_message", "") if current_config else None
            if explicit_none:
                eff_display = "<None>"
            else:
                eff_display = _format_system_message_display(effective, default_value=default_sys)
            print(f"Effective system message: {eff_display}")
            if segments and not explicit_none:
                print("Segments (oldest → newest):")
                for key, value in segments.items():
                    preview = _clip_long_message(value) if value else ""
                    preview_display = preview if preview else "<empty>"
                    print(f"  {key}: '{preview_display}'")

            # Show command history along the active path
            path = target.active_path_for_llm() if target else []
            history = []
            for turn in path:
                for cmd in getattr(turn, "cmd", []) or []:
                    if cmd.cmd_type == Command.PARAM_CHANGE and cmd.data.get("change") == "system_message":
                        history.append((turn, cmd))
            if not history:
                print("Commands: none on active path.")
                return
            print("Commands on active path:")
            for idx, (turn, cmd) in enumerate(history, start=1):
                op = cmd.data.get("op") or "set"
                val = cmd.data.get("value", cmd.data.get("new_value"))
                preview_display = _format_system_message_value(val, cmd.data.get("value_kind"))
                stack_id = cmd.data.get("stack_id")
                meta_bits = []
                if stack_id:
                    meta_bits.append(f"stack_id={stack_id}")
                meta = f" [{', '.join(meta_bits)}]" if meta_bits else ""
                turn_id = getattr(turn, "gen_id", None) or getattr(turn, "gen_id_or_parent", "N/A")
                cmd_id = cmd.gen_id or "N/A"
                print(f"  {idx}. op={op} val='{preview_display}' turn={turn_id} cmd_id={cmd_id}{meta}")

        def _resolve_sm_cursor(gen_id_arg: str) -> Optional[ChatCursor]:
            if not gen_id_arg:
                return cursor
            try:
                return cursor.cursor_for_gen_id(gen_id_arg)
            except KeyError:
                print(f"{Colors.ERROR}Turn with gen_id '{gen_id_arg}' not found.{Colors.RESET}")
            except ValueError as err:
                print(f"{Colors.ERROR}{err}{Colors.RESET}")
            return None

        if raw_args is None or not raw_args.strip():
            _print_system_message_segments(cursor)
            return True, False, None
        if raw_args.strip() in {"?", "help"}:
            _print_system_message_cli_help()
            return True, False, None
        sys_args = raw_args.strip()
        parts = sys_args.split(" ", 1)
        action_token = parts[0].lower()
        remainder = parts[1].strip() if len(parts) > 1 else ""
        operation = None
        value_arg = None

        if action_token in {"status", "stat", "sc", "scope"}:
            target_cursor = _resolve_sm_cursor(remainder)
            if target_cursor:
                _print_system_message_scope(target_cursor)
            return True, False, None
        if action_token in ("s", "set"):
            operation = "set"
            value_arg = remainder
        elif action_token in ("a", "add"):
            operation = "add"
            value_arg = remainder
        elif action_token in ("p", "pop"):
            operation = "pop"
        else:
            operation = "set"
            value_arg = sys_args

        def _interpret_value(text: str) -> Tuple[Optional[str], Optional[str]]:
            token = text.strip()
            if token == "''":
                return "", None
            if token == "<>":
                return None, "remove"
            if token.lower() == "<def>":
                return None, "default"
            return text, None

        if operation in {"set", "add"}:
            if not value_arg:
                print(f"{Colors.ERROR}No text provided for '/sm {operation}'.{Colors.RESET}")
                return True, False, None
            parsed_value, value_kind = _interpret_value(value_arg)
            if operation == "add" and (value_kind is not None or parsed_value in ("", None)):
                print(f"{Colors.ERROR}Empty/default system messages can only be set, not added.{Colors.RESET}")
                return True, False, None
            base_cursor = cursor
            base_cursor.apply_system_message(
                operation,
                parsed_value,
                command_text=user_input,
                value_kind=value_kind,
            )

            display_value = _format_system_message_value(parsed_value, value_kind)
            verb = "Appended" if operation == "add" else "Set"
            print(f"{Colors.SYSTEM}{verb} system message segment: {display_value}{Colors.RESET}")
        else: # pop
            base_cursor = cursor
            stack_id, _ = _parse_pop_target_options(remainder)
            try:
                base_cursor.apply_system_message(
                    "pop",
                    None,
                    command_text=user_input,
                    stack_id=stack_id,
                )
            except ValueError as exc:
                print(f"{Colors.ERROR}{exc}{Colors.RESET}")
                return True, False, None

            print(f"{Colors.SYSTEM}System message stack pop recorded.{Colors.RESET}")
        return True, False, None
    elif cmd_prefix.startswith("config"):
        parts = command_full.split()
        flags = {p for p in parts[1:] if p.startswith("-")}
        reconfig_flag = "-r" in flags
        params_only_flag = "-p" in flags

        if reconfig_flag:
            # This is /config -r, so reconfigure
            print("Reconfiguring. Restart required for engine changes.")
            current_config = prompt_for_config(
                config_to_update=current_config,
                save_to_path=EFFECTIVE_CONFIG_FILE_PATH,
                prompt_for_name=True,
            )
            updated_params = _interactive_edit_conversation_params(current_config.get("inference_params"))
            current_config["inference_params"] = _normalize_chat_params(updated_params)
            save_config(current_config, EFFECTIVE_CONFIG_FILE_PATH)
            # Re-init classes that depend on config paths. Toolbox is now loaded manually.
            session_control = SessionControl(Path(current_config["sessions_save_dir"]))
            toolbox = Toolbox() # Re-initialize toolbox
            toolbox.from_dict(json.load(open(current_config["tools_config_path"])), search_scope=globals(), external_handler=external_tool_handler) # Reload from new path
            base_cursor = cursor
            base_cursor.log_command(user_input)

            print("Config updated. Please restart.")
        elif params_only_flag:
            base_cursor = cursor
            base_cursor.log_command(user_input)
            print("\n--- Current inference_params ---")
            print(json.dumps(current_config.get("inference_params", _conversation_param_defaults()), indent=2))
            updated = _interactive_edit_conversation_params(current_config.get("inference_params"))
            current_config["inference_params"] = _normalize_chat_params(updated)
            save_config(current_config, EFFECTIVE_CONFIG_FILE_PATH)
            print("Conversation defaults updated. No restart required.")
        else:
            # This is just /config, so print current config
            base_cursor = cursor
            base_cursor.log_command(user_input)

            print("\n--- Current Configuration ---")
            print(json.dumps(current_config, indent=2))
            print("---")
        return True, False, None

    elif cmd_prefix.startswith("fp"): # Handles /fp, /fpc, /fpr and their -repr variants
        # Matches fp, fpc, fpr, with an optional -repr flag.
        format_args_match = re.match(r"fp[cr]?(-repr)?\s*(.*)", command_full, re.IGNORECASE | re.DOTALL)
        use_repr_flag = bool(format_args_match.group(1)) # type: ignore
        text_to_format = format_args_match.group(2).strip() if format_args_match else "" # type: ignore
        
        gen_id = None
        args_list = text_to_format.split(" ", 2)

        if len(args_list) >= 2 and args_list[0] == "--gen_id":
            gen_id = args_list[1]
            text_to_format = args_list[2] if len(args_list) > 2 else ""

        base_cursor = cursor
        if gen_id:
            try:
                base_cursor = cursor.cursor_for_gen_id(gen_id)
                # This is a temporary live-context cursor for formatting; it does not change the active turn.
                print(f"{Colors.SYSTEM}Formatting prompt from context of turn {gen_id}.{Colors.RESET}")
            except (KeyError, ValueError) as e:
                print(f"{Colors.ERROR}Could not find turn for gen_id '{gen_id}': {e}{Colors.RESET}")
                return True, False, None
        
        new_cursor, suppress = await _handle_format_prompt_command(text_to_format, use_repr_flag, base_cursor, user_input)
        return suppress, False, new_cursor
    elif cmd_prefix.startswith("tk"):
        tokenize_args_match = re.match(r"tk(-repr)?\s*(.*)", command_full, re.IGNORECASE | re.DOTALL) # noqa
        use_repr_flag = bool(tokenize_args_match.group(1)) # type: ignore
        text_to_tokenize = tokenize_args_match.group(2).strip() if tokenize_args_match else "" # type: ignore
        base_cursor = cursor
        new_cursor, suppress = await _handle_tokenize_command(text_to_tokenize, use_repr_flag, base_cursor, user_input)
        return suppress, False, new_cursor
    elif cmd_prefix.startswith("t"): # Must be checked AFTER tk
        tools_args_match = re.match(r"t(?:ools)?\s*(.*)", command_full, re.IGNORECASE) # type: ignore
        tools_args = tools_args_match.group(1).strip() if tools_args_match else "" # type: ignore
        base_cursor = cursor
        new_cursor, suppress = await _handle_tools_command(tools_args, base_cursor, pt_session)
        return suppress, False, new_cursor
    elif cmd_prefix.startswith("f"): # flags
        flags_args_match = re.match(r"f(?:lags)?\s*(.*)", command_full, re.IGNORECASE)
        flags_args = flags_args_match.group(1).strip() if flags_args_match else "" # type: ignore
        # The /flags command can result in either a PARAM_CHANGE or a STATE_CHANGE.
        # The _handle_flags_command function will now be responsible for creating the appropriate turn.
        # We no longer create a turn here.
        base_cursor = cursor
        new_cursor, suppress = await _handle_flags_command(flags_args, base_cursor, user_input)
        return suppress, False, new_cursor
    elif cmd_prefix.startswith("rl"):
        cursor = _require_current_cursor()
        cursor.log_command(command_full_text)
        session = cursor.session
        conv_idx: Optional[int] = None
        if sub_args and sub_args.strip():
            try:
                conv_idx = int(sub_args.strip()) - 1
            except ValueError:
                print(f"{Colors.ERROR}Invalid conversation index '{sub_args.strip()}'.{Colors.RESET}")
                return None, True, None
        
        try:
            removed_count = session.remove_logs(conv_idx)
            if conv_idx is not None:
                print(f"{Colors.SYSTEM}Removed {removed_count} log commands from conversation {conv_idx + 1}.{Colors.RESET}")
            else:
                print(f"{Colors.SYSTEM}Removed {removed_count} log commands from all conversations.{Colors.RESET}")
        except IndexError:
            print(f"{Colors.ERROR}Conversation index '{sub_args.strip()}' is out of range.{Colors.RESET}")

        return None, True, None

    elif cmd_prefix.startswith("eng"): # Covers /engine, /eng
        engine_args_match = re.match(r"eng(?:ine)?\s*(.*)", command_full, re.IGNORECASE)
        engine_args = engine_args_match.group(1).strip().lower() if engine_args_match else "" # type: ignore

        if engine_args.startswith("m"): # metrics
            # NEW: handle metrics subcommands
            metrics_args_match = re.match(r"m(?:etrics)?\s*(.*)", engine_args, re.IGNORECASE)
            metrics_args = metrics_args_match.group(1).strip().lower() if metrics_args_match else "" # type: ignore
            if metrics_args.startswith("r"): # reset
                base_cursor = cursor
                base_cursor.command_event(
                    user_input,
                    metadata={"$Action": "reset_metrics", "$ActionArgs": {"enabled": True}},
                )
                _set_cursor_reset_metrics(base_cursor, True, command_text=user_input)
                print(f"{Colors.SYSTEM}Metrics will be reset on the next inference request.{Colors.RESET}")
            else: # get metrics
                print("Getting aggregate engine metrics...")
                base_cursor = cursor
                resp = await call_api("get-aggregate-metrics")
                cmd_obj = base_cursor.save_api_command(
                    "get-aggregate-metrics",
                    {"reset_metrics": False},
                    command_text=user_input,
                    response=resp,
                )
                if resp.get("status") == "success":
                    print("\n--- Aggregate Engine Metrics ---")
                    print(json.dumps(resp.get("data", {}), indent=2, default=str))
                    print("------------------------------\n")
                else:
                    print(f"Failed to get aggregate metrics: {resp.get('message')}")
        else: # Default to status # type: ignore
            base_cursor = cursor
            base_cursor.log_command(user_input)
            print("Getting engine status...")
            resp = await call_api("get-engine-status")
            base_cursor.save_api_command("get-engine-status", {}, command_text=user_input, response=resp)
            if resp.get("status") == "success":
                print("\n--- Engine Status ---")
                print(json.dumps(resp.get("data", {}), indent=2, default=str))
                print("---------------------\n")
            else:
                print(f"Failed to get engine status: {resp.get('message')}")
        return True, False, None
    else:
        base_cursor = cursor
        base_cursor.log_command(user_input)

        print(f"{Colors.ERROR}Unknown command: {parts[0]}. Type /help.{Colors.RESET}")
    
    return True, False, None

async def _wait_for_engine_ready():
    """Helper function to poll the engine until the inference component is ready."""
    print("Waiting for engine to be ready...")
    while True:
        await asyncio.sleep(1)  # Poll every second
        status_resp = await call_api("get-engine-status")
        if status_resp.get("status") == "success":
            inference_status = status_resp.get("data", {}).get("inference_component_status", {}).get("status")
            # The engine is considered "ready" for the next prompt if it's either in the 'ready' state
            # or in a recoverable 'error' state. The next API call will handle the error recovery.
            # The CANCELLATION_TIMEOUT state is transient, so we just continue polling.
            if inference_status in ["ready", "error"]:
                print(f"Engine is ready (status: {inference_status}). You can enter your next prompt.")
                break
        else:
            print("\nError getting engine status while waiting. Proceeding with caution.")
            break

async def _handle_raw_command(args_str: str, cursor: ChatCursor) -> ChatCursor:
    """Handles the /raw command using a temporary try-out branch."""
    global LAST_EXCEPTION_TRACEBACK

    # Ensure cursor has a context before cloning
    if not cursor.context:
        context = _active_chat_context()
        if context:
            cursor.bind_context(context)
        else:
            print(f"{Colors.ERROR}Cannot send raw prompt: no active chat context.{Colors.RESET}")
            return cursor

    raw_prompt_escaped = args_str
    if not raw_prompt_escaped:
        print("Usage: /raw <prompt text with escaped characters>")
        print("Example: /raw A user asks: What is 2+2?\\nAssistant: ")
        return cursor

    try:
        raw_prompt_unescaped = codecs.decode(raw_prompt_escaped, 'unicode_escape')
    except Exception as e:
        print(f"Error decoding escaped string: {e}")
        return cursor

    print(f"{Colors.SYSTEM}Sending raw prompt (unescaped):\n---\n{raw_prompt_unescaped}\n---{Colors.RESET}")

    # 1) Create a temporary try-out branch for raw prompts.
    _, try_cursor = cursor.add_try_out()

    gen_config_for_payload = _generation_config_template_snapshot()
    max_new_override = _max_new_tokens_value()
    if max_new_override:
        gen_config_for_payload["max_new_tokens"] = max_new_override

    inference_payload = {
        "request_id": try_cursor.get_request_id("chat_raw"),
        "raw_list": [raw_prompt_unescaped],
        "generation_config": gen_config_for_payload,
        "stream": _current_streaming_enabled(),
    }
    if _current_suppress_full_response():
        inference_payload["suppress_full_response"] = True
    active_adapters_for_raw = try_cursor.get_effective_adapters()
    inference_payload["active_adapters"] = active_adapters_for_raw
    current_return_prompt = _current_return_prompt_mode()
    if current_return_prompt:
        inference_payload["return_prompt"] = current_return_prompt

    if _cursor_reset_metrics_enabled(try_cursor):
        inference_payload["reset_metrics"] = True
        print(f"{Colors.SYSTEM}Note: This raw inference request includes 'reset_metrics=True'.{Colors.RESET}")

    print(f"{Colors.SYSTEM}LLM thinking (raw prompt)...{Colors.RESET}")

    try:
        sys.stdout.write(f"\n{Colors.LLM_HEADER}LLM:{Colors.RESET}\n{Colors.LLM_CONTENT}")
        sys.stdout.flush()

        api_response = await call_api("run-inference", inference_payload)
        try_cursor.save_api_command("run-inference", inference_payload, command_text=f"/raw {args_str}", response=api_response)

        if not (isinstance(api_response, dict) and "stream" in api_response):
            print(f"\n{Colors.ERROR}Error from inference API: {api_response.get('message', 'Unknown error')}{Colors.RESET}")
            return cursor

        api_iterator = api_response.get("stream")

        final_metrics = {}
        final_chunk_for_item_metrics = None

        if api_iterator:
            async for chunk_data in api_iterator:
                if chunk_data.get("chunkType") == ChunkType.ERROR.value:
                    print(f"\n{Colors.ERROR}Error during stream: {chunk_data.get('message')}{Colors.RESET}\n")
                    break

                if chunk_data.get("chunkType") == ChunkType.STREAMING_ENDED.value:
                    sys.stdout.write(f"{Colors.RESET}\n")
                    final_metrics = {
                        "total_input_tokens": chunk_data.get("total_input_tokens"),
                        "total_output_tokens": chunk_data.get("total_output_tokens"),
                        "total_generation_duration_sec": chunk_data.get("total_generation_duration_sec"),
                        "overall_tps": chunk_data.get("overall_tps"),
                        "avg_time_to_first_token_sec": chunk_data.get("avg_time_to_first_token_sec"),
                        "cache_queued": chunk_data.get("cache_queued"),
                        "in_flight_req": chunk_data.get("in_flight_req"),
                        "mem_allocated": chunk_data.get("mem_allocated"),
                        "mem_reserved": chunk_data.get("mem_reserved"),
                    }
                    agg_subset = {
                        key: final_metrics.get(key)
                        for key in ("cache_queued", "in_flight_req", "mem_allocated", "mem_reserved")
                        if final_metrics.get(key) is not None
                    }
                    try_cursor.update_metrics(ChatCursor.update_response_metrics(metrics=agg_subset))
                    break
                elif chunk_data.get("chunkType") == ChunkType.ERROR.value:
                    sys.stdout.write(f"\n{Colors.ERROR}Error during stream: {chunk_data.get('error')}{Colors.RESET}\n")
                    break

                if chunk_data.get("chunkType") == ChunkType.STREAMING_CHUNK.value:
                    if token := chunk_data.get("chunk_text", ""):
                        sys.stdout.write(token)
                        sys.stdout.flush()

                        if chunk_data.get("is_final_chunk"):
                            final_chunk_for_item_metrics = chunk_data
                            metrics_payload = ChatCursor.update_response_metrics(metrics=chunk_data)
                            try_cursor.update_metrics(metrics_payload)
        else:
            print(f"\n{Colors.ERROR}Error: Expected a streaming response for /raw command but received a synchronous one.{Colors.RESET}")

        if final_metrics and any(v is not None for v in final_metrics.values()):
            elapsed_seconds_metrics = int(time.time() - try_cursor.context.session.creation_timestamp)
            metrics_line = f"Metrics:"
            if (total_in := final_metrics.get("total_input_tokens")) is not None: metrics_line += f" In: {total_in}"
            if (total_out := final_metrics.get("total_output_tokens")) is not None: metrics_line += f" Out: {total_out}"
            if (duration := final_metrics.get("total_generation_duration_sec")) is not None: metrics_line += f" GenTime: {duration:.1f}s"
            if (tps := final_metrics.get("overall_tps")) is not None: metrics_line += f" TPS: {tps:.1f}"
            if (latency := final_metrics.get("avg_time_to_first_token_sec")) is not None: metrics_line += f" Latency: {latency * 1000:.0f}ms"

            if final_chunk_for_item_metrics:
                if (cache_metric := final_chunk_for_item_metrics.get("cache_metric")):
                    metrics_line += f" Cache: {cache_metric}"
                if (cache_warming := final_chunk_for_item_metrics.get("cache_warming")):
                    metrics_line += f" Warm-up: {cache_warming}"
                if final_chunk_for_item_metrics.get("was_truncated"):
                    metrics_line += f" {Colors.BRIGHT_YELLOW}Truncated: Yes{Colors.METRICS}"

            if cache_queued := final_metrics.get("cache_queued"): metrics_line += f" Queued: {cache_queued}"
            if in_flight := final_metrics.get("in_flight_req"): metrics_line += f" In-flight: {in_flight}"
            if (mem_alloc := final_metrics.get("mem_allocated")) is not None: metrics_line += f" Mem(A): {mem_alloc:.0f}MB"
            if (mem_rsvd := final_metrics.get("mem_reserved")) is not None: metrics_line += f" Mem(R): {mem_rsvd:.0f}MB"
            print(f"{Colors.METRICS}[+{elapsed_seconds_metrics}s] {metrics_line}{Colors.RESET}")

        if inference_payload.get("reset_metrics"):
            _set_cursor_reset_metrics(try_cursor, False)

        print(f"\n{Colors.SYSTEM}(Raw prompt response not added to session history){Colors.RESET}")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nInference cancelled by user. Notifying engine...")
        request_id_to_cancel = inference_payload.get("request_id")
        cancel_resp = await call_api("cancel-request", {"request_id": request_id_to_cancel} if request_id_to_cancel else {})
        print(f"Engine notified of cancellation (Status: {cancel_resp.get('status', 'unknown')}, Msg: {cancel_resp.get('message', 'N/A')}).")
        await _wait_for_engine_ready()
    except Exception as e:
        _store_exception_traceback_if_clear(e)

    try_cursor.close_branch()
    return try_cursor.context.active_cursor if try_cursor.context else try_cursor

async def _handle_format_prompt_command(args_str: str, use_repr: bool, cursor: ChatCursor, user_input: str) -> Tuple[ChatCursor, bool]:
    """Handles the /fp and /fpc commands by formatting a detached prompt preview."""
    command_part = user_input.split(" ", 1)[0]
    is_continue = "fpc" in command_part
    is_root = "fpr" in command_part
    save_response = False
    stripped_args = args_str.strip()
    while stripped_args:
        flag_match = re.match(r"^-(?:s|save)(?:\s+|$)", stripped_args, re.IGNORECASE)
        if not flag_match:
            break
        save_response = True
        stripped_args = stripped_args[flag_match.end():]
    args_str = stripped_args

    # Ensure cursor has a context before cloning
    if not cursor.context:
        context = _active_chat_context()
        if context:
            cursor.bind_context(context)
        else:
            print(f"{Colors.ERROR}Cannot format prompt: no active chat context.{Colors.RESET}")
            return cursor, True

    format_turn = Turn(
        turn_type=Turn.CHAT if (args_str or is_continue) else None,
        metadata={"user": {"role": "user", "content": args_str or ""}} if (args_str or is_continue) else None,
        do_continue=is_continue,
        root_context=is_root,
    )
    format_turn.parent = cursor.current_turn
    format_cursor = cursor.snapshot_at(format_turn)

    # Build the payload using the standard helper function on the detached turn.
    payload, _, _ = format_cursor.build_inference_request(
        request_id_prefix="format",
        manual_continue=is_continue,
    )

    # Extract only the necessary parts for the format-inference-prompt API.
    format_payload = {}
    if payload.get("messages_list"):
        format_payload["messages_list"] = payload.get("messages_list")
    if payload.get("tools"):
        format_payload["tools"] = payload.get("tools")
    if payload.get("do_continue"):
        format_payload["do_continue"] = payload.get("do_continue")
        
    # Call the API and log it.
    resp = await call_api("format-inference-prompt", format_payload)
    cursor.save_api_command(
        "format-inference-prompt",
        format_payload,
        command_text=user_input,
        response=resp if save_response else None,
    )

    # Print the formatted prompt from the response.
    if resp.get("status") in ["success"]:
        result = resp.get("data", {})
        formatted_prompts = result.get("formatted_prompts", [])
        token_counts = result.get("prompt_token_counts", [])
 
        if formatted_prompts:
            token_count_info = ""
            if token_counts and len(token_counts) > 0:
                token_count_info = f" | Tokens: {token_counts[0]}"
 
            header_text = f"Formatted Prompt (repr)" if use_repr else "Formatted Prompt"
            print(f"\n--- {header_text}{token_count_info} ---")
 
            if use_repr:
                print(repr(formatted_prompts[0]))
            else:
                print(formatted_prompts[0])

            print("---------------------------------\n")
        else:
            print("Engine returned no formatted prompts.")
    else:
        print(f"Error formatting prompt: {resp.get('message', 'Unknown error')}")

    # Conversation stays on the main branch
    return cursor, True # Always suppress LLM call

async def _handle_tokenize_command(args_str: str, use_repr: bool, cursor: ChatCursor, user_input: str) -> Tuple[ChatCursor, bool]:
    """Handles the /tk command."""
    cursor.log_command(user_input)

    if not args_str:
        print("Usage: /tk[-repr] <text to tokenize>")
        return cursor, True

    payload = {
        "text": args_str,
        "is_repr": use_repr,
    }
    resp = await call_api("count-tokens", payload)
    cursor.save_api_command("count-tokens", payload, command_text=f"/tk {'-repr ' if use_repr else ''}{args_str}", response=resp)
    if resp.get("status") == "success":
        result = resp.get("data", {})
        token_count = result.get("token_count")
        text_processed = result.get("text_processed", args_str)
        
        print(f"\n--- Token Count ---")
        print(f"  Token Count: {token_count}")
        if use_repr:
            print(f"  Text (decoded from repr): {text_processed}")
        print(f"-------------------\n")
    else:
        print(f"Error counting tokens: {resp.get('message', 'Unknown error')}")
    return cursor, True

async def _handle_flags_command(args_str: str, cursor: ChatCursor, user_input: str) -> Tuple[ChatCursor, bool]:
    """Handles the /flags command and its subcommands."""
    global LAST_EXCEPTION_TRACEBACK, DEBUG_MODE_ENABLED, current_config

    parts = args_str.split(" ", 1)
    sub_cmd = parts[0].lower().strip() if parts and parts[0] else ""
    sub_args = parts[1].strip() if len(parts) > 1 else ""
    sub_args_lower = sub_args.lower()
    context = cursor.context or _active_chat_context()

    # Quick help: /flags ?  or /flags help
    if sub_cmd in ["?", "help"]:
        print("--- /flags help ---")
        print("Usage: /flags <subcommand> [value]")
        print("")
        print("Available subcommands and allowed values:")
        print("  show                  - Display current flag values.")
        print("  stream | s [on|off|toggle] - Enable/disable or toggle streaming of model output.")
        print("  maxnew [<number>|off|reset|default] - Override max_new_tokens for this session or reset to engine default.")
        print("  debug | d [n|e|w|i|a] - Set console log level (none, error, warn, info, all).")
        print("  cache | c [dynamic|static|offloaded|dynamic_reset|static_reset|default|off|no_cache] - Override KV cache routing for this session.\n"
              "    - 'no_cache' disables KV cache for the session.\n"
              "    - 'default' clears the override and uses engine defaults.\n"
              "    - 'static' requires static_kv_cache to be enabled in engine config.")
        print("  prompt | p [off|full|last] - Prompt response mode: 'full' returns whole prompt batches, 'last' returns only the most recent batch, 'off' disables returning prompts.")
        print("  exception | ex         - Print the last recorded exception traceback (if any).")
        print("  tr [on|off]            - Automatically retry truncated responses.")
        print("")
        print("Examples:")
        print("  /f s off                 -> disable streaming")
        print("  /flags maxnew 256        -> set max new tokens to 256 for this session")
        print("-----------------------")
        return cursor, True

    if sub_cmd in ("d", "debug"):
        level_str = sub_args_lower
        log_level_map = {
            "n": logging.CRITICAL + 1,
            "e": logging.ERROR,
            "w": logging.WARNING,
            "i": logging.INFO,
            "a": logging.DEBUG,
            "none": logging.CRITICAL + 1,
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "all": logging.DEBUG,
        }
        if level_str in log_level_map:
            new_level = log_level_map[level_str]
            _set_console_log_level(new_level)
            DEBUG_MODE_ENABLED = new_level <= logging.DEBUG
            display_name = "NONE" if new_level > logging.CRITICAL else logging.getLevelName(new_level)
            print(f"Console log level set to: {display_name}")
        else:
            print("Usage: /f d[ebug] <n[one]|e[rror]|w[arn]|i[nfo]|a[ll]>")
        return cursor, True

    if not sub_cmd or sub_cmd == "show":
        print("--- Current Flags ---")
        print(f"  Streaming: {'ON' if _current_streaming_enabled() else 'OFF'}")
        max_new_value = _max_new_tokens_value()
        mnt_override_display = str(max_new_value) if max_new_value is not None else "Engine Default"
        print(f"  Max New Tokens Override: {mnt_override_display}")
        cache_value = _current_cache_override()
        cache_override_display = cache_value.capitalize() if cache_value else "Engine Default"
        print(f"  Cache Override: {cache_override_display}")
        print(f"  Debug Mode: {'ON' if DEBUG_MODE_ENABLED else 'OFF'}")
        return_prompt_value = _current_return_prompt_mode() or "off"
        print(f"  Prompt Response Mode: {return_prompt_value}")
        print(f"  Auto-retry Truncated: {'ON' if _auto_retry_enabled() else 'OFF'}")
        print(f"  Auto-continue Retry Limit: {_auto_continue_retry_limit()}")
        print(f"  Auto-tool Retry Limit: {_auto_tool_retry_limit()}")
        print("---------------------")
        cursor.log_command(user_input)
        return cursor, True

    if sub_cmd in ("stream", "s"):
        context = cursor.context or _active_chat_context()
        if not context:
            print("Error: No active context available.")
            return cursor, True
        original_value = _current_streaming_enabled()
        if not sub_args_lower:
            print(f"Streaming mode is currently {'ON' if _current_streaming_enabled() else 'OFF'}.")
            return cursor, True
        if sub_args_lower == "on":
            new_value = True
        elif sub_args_lower == "off":
            new_value = False
        elif sub_args_lower == "toggle":
            new_value = not original_value
        else:
            print(f"Invalid argument for stream flag: '{sub_args}'. Use 'on', 'off', or 'toggle'.")
            return cursor, True
        _set_chat_param_value("stream", new_value, command_text=user_input)
        print(f"Streaming mode {'enabled' if new_value else 'disabled'}.")
        return cursor, True

    if sub_cmd == "maxnew":
        context = cursor.context or _active_chat_context()
        if not context:
            print("Error: No active context available.")
            return cursor, True
        original_value = _max_new_tokens_value()
        if not sub_args_lower:
            mnt_override_display = str(original_value) if original_value is not None else "Engine Default"
            print(f"Max New Tokens Override: {mnt_override_display}")
            return cursor, True
        new_value: Optional[int] = None
        if sub_args_lower in {"off", "reset", "default"}:
            print("Max new tokens override disabled. Using engine default.")
        else:
            try:
                size = int(sub_args)
                if size > 0:
                    new_value = size
                    print(f"Max new tokens override for this session set to: {size}")
                else:
                    print("Error: Max new tokens must be a positive number.")
                    return cursor, True
            except ValueError:
                print(f"Error: Invalid context size '{sub_args}'. Must be a number.")
                return cursor, True
        _set_chat_param_value("max_new_tokens", new_value, command_text=user_input)
        return cursor, True

    if sub_cmd in ("cache", "c"):
        context = cursor.context or _active_chat_context()
        if not context:
            print("Error: No active context available.")
            return cursor, True
        original_value = _current_cache_override()
        valid_options = ["dynamic", "static", "offloaded", "dynamic_reset", "static_reset", "default", "off", "no_cache"]
        if not sub_args_lower or sub_args_lower not in valid_options:
            print(f"Invalid argument for cache flag: '{sub_args}'. Use one of: {', '.join(valid_options)}.")
            return cursor, True
        if sub_args_lower == "default":
            new_value = None
            print("Cache override disabled. Using engine default routing.")
        elif sub_args_lower == "no_cache":
            new_value = "no_cache"
            print("Cache override for this session set to: no_cache. KV cache will not be used.")
        elif sub_args_lower in ["static", "static_reset"]:
            if current_config and not current_config.get("static_kv_cache", True):
                print(f"{Colors.ERROR}Error: Cannot set cache override to '{sub_args_lower}' because static KV cache is disabled in the engine configuration.{Colors.RESET}")
                return cursor, True
            new_value = sub_args_lower
            print(f"Cache override for this session set to: {sub_args_lower}")
        else:
            new_value = sub_args_lower
            print(f"Cache override for this session set to: {sub_args_lower}")
        _set_chat_param_value("cache", new_value, command_text=user_input)
        return cursor, True

    if sub_cmd in ("prompt", "p"):
        context = cursor.context or _active_chat_context()
        if not context:
            print("Error: No active context available.")
            return cursor, True
        original_value = _current_return_prompt_mode()
        valid_options = ["off", "full", "last"]
        if not sub_args_lower:
            print(f"Return prompt mode is currently: {original_value or 'off'}")
            return cursor, True
        if sub_args_lower in valid_options:
            if sub_args_lower == "off":
                new_value = None
                print("Return prompt mode is OFF.")
            else:
                new_value = sub_args_lower
                print(f"Return prompt mode set to: {sub_args_lower}")
        else:
            print(f"Invalid argument for prompt flag: '{sub_args}'. Use one of: {', '.join(valid_options)}.")
            return cursor, True
        _set_chat_param_value("return_prompt", new_value, command_text=user_input)
        return cursor, True

    if sub_cmd in ("exception", "ex"):
        if LAST_EXCEPTION_TRACEBACK:
            print("\n--- Last Recorded Exception Traceback ---")
            print(LAST_EXCEPTION_TRACEBACK)
            print("---------------------------------------\n")
        else:
            print("No exception traceback recorded yet.")
        return cursor, True

    if sub_cmd == "tr":
        context = cursor.context or _active_chat_context()
        if not context:
            print("Error: No active context available.")
            return cursor, True
        original_value = _auto_retry_enabled()
        if not sub_args_lower:
            print(f"Auto-retry truncated is currently {'ON' if original_value else 'OFF'}.")
            return cursor, True
        if sub_args_lower == "on":
            new_value = True
            print("Auto-retry truncated enabled.")
        elif sub_args_lower == "off":
            new_value = False
            print("Auto-retry truncated disabled.")
        else:
            print(f"Invalid argument for tr flag: '{sub_args}'. Use 'on' or 'off'.")
            return cursor, True
        _set_chat_param_value("auto_retry_truncated", new_value, command_text=user_input)
        return cursor, True

    if sub_cmd == "trc":
        context = cursor.context or _active_chat_context()
        if not context:
            print("Error: No active context available.")
            return cursor, True
        current_val = _get_param_value("auto_continue_retry_limit", DEFAULT_CHAT_PARAMS.get("auto_continue_retry_limit", 10))
        if not sub_args_lower:
            print(f"Auto-continue retry limit is currently {current_val}.")
            return cursor, True
        try:
            new_value = int(sub_args_lower)
            if new_value < 0:
                raise ValueError
        except ValueError:
            print("Invalid value for trc. Provide a non-negative integer.")
            return cursor, True
        _set_chat_param_value("auto_continue_retry_limit", new_value, command_text=user_input)
        print(f"Auto-continue retry limit set to {new_value}.")
        return cursor, True

    if sub_cmd == "trt":
        context = cursor.context or _active_chat_context()
        if not context:
            print("Error: No active context available.")
            return cursor, True
        current_val = _get_param_value("auto_tool_retry_limit", DEFAULT_CHAT_PARAMS.get("auto_tool_retry_limit", 3))
        if not sub_args_lower:
            print(f"Auto-tool retry limit is currently {current_val}.")
            return cursor, True
        try:
            new_value = int(sub_args_lower)
            if new_value < 0:
                raise ValueError
        except ValueError:
            print("Invalid value for trt. Provide a non-negative integer.")
            return cursor, True
        _set_chat_param_value("auto_tool_retry_limit", new_value, command_text=user_input)
        print(f"Auto-tool retry limit set to {new_value}.")
        return cursor, True

    print(f"Unknown flags command: '{sub_cmd}'. Available: stream, maxnew, debug, cache, tr, trc, trt, show, p[rompt], ex[ception].")
    return cursor, True


async def _handle_auto_continuation(cursor: ChatCursor) -> Tuple[ChatCursor, bool]:
    """
    Handles the logic for auto-continuing a truncated response.
    Checks for an existing 'auto_cont' anchor and reuses it, otherwise creates a new one.
    Returns the new cursor and a boolean indicating if an iteration was scheduled.
    """
    if not _auto_retry_enabled():
        return cursor, False

    ctx = cursor.context
    if not ctx:
        return cursor, False
    scope = _require_live_chat_scope(cursor)

    def _cursor_on_anchor_branch(cur: ChatCursor, anchor) -> ChatCursor:
        """Keep auto-continue retries on the anchor's try_out branch."""
        target_try_turn = None
        try:
            target_try_turn = next((t for t in anchor.try_out_turns or [] if t), None)
        except Exception:
            target_try_turn = None
        if not target_try_turn:
            return cur
        # Prefer a registered try-out cursor to avoid rebinding.
        try:
            candidate = scope.resolve_try_out_cursor(anchor, prefer_latest=True)
            if candidate:
                ancestry: Set[Turn] = set()
                node = getattr(candidate, "current_turn", None)
                while node:
                    ancestry.add(node)
                    node = getattr(node, "parent", None)
                if target_try_turn in ancestry:
                    return candidate
        except Exception:
            pass
        ancestry: Set[Turn] = set()
        node = getattr(cur, "current_turn", None)
        while node:
            ancestry.add(node)
            node = getattr(node, "parent", None)
        if target_try_turn in ancestry:
            return cur
        try:
            origin_count = cur.real_user_turns()
            rebuilt = scope.register_cursor_for_turn(target_try_turn, make_active=False)
            if rebuilt:
                rebuilt.is_fork = True
                rebuilt.origin_user_turn_index = origin_count
                return rebuilt.descend_to_leaf()
            # Fallback: rebind the existing cursor if we cannot build a new one.
            cur.rebind_to_turn(target_try_turn)
            if not cur.is_fork:
                cur.is_fork = True
                cur.origin_user_turn_index = origin_count
            return cur.descend_to_leaf()
        except Exception:
            return cur

    # Check if we are already in a continuation loop for this branch
    existing_anchor = scope.find_active_anchor(CONTINUE_AUTO_TRYOUT_KIND, cursor)
    if existing_anchor:
        cursor = _cursor_on_anchor_branch(cursor, existing_anchor)
    
    new_cursor = None

    if existing_anchor:
        if existing_anchor.retries_remaining > 0:
            # We are in a retry loop, just add a continuation turn to the current branch
            existing_anchor.retries_remaining -= 1
            new_cursor = cursor.add_continuation_turn()
            new_cursor.set_auto(True)
            cursor.set_main_thread(True)
            new_cursor.set_main_thread(True)
            print(f"\n{Colors.SYSTEM}Response truncated again. Continuing on same try_out branch... ({existing_anchor.retries_remaining} retries left){Colors.RESET}")
        else:
            # Retries exhausted
            print(f"{Colors.TOOL_WARNING}Continuation retries exhausted for anchor '{existing_anchor.anchor_name}'.{Colors.RESET}")
            try:
                closed_cursor = _ensure_registered_cursor(
                    scope.close_try_out_anchor(existing_anchor.anchor_name, dist_mode="keep")
                )
                if closed_cursor:
                    cursor = closed_cursor
            except Exception as e:
                print(f"{Colors.ERROR}Error closing exhausted anchor: {e}{Colors.RESET}")
            rebound_ctx = _drain_and_close_auto_tryouts(cursor, close_anchors=True, context_override=ctx)
            try:
                if rebound_ctx:
                    context = _resolve_context_for_scope(scope, ctx=rebound_ctx)
                    cursor = scope.active_cursor()
            except Exception:
                pass
            return cursor, False
    else:
        # This is the first truncation, start a new anchor and try_out
        limit = _auto_continue_retry_limit()
        if limit > 0:
            anchor_name = _auto_anchor_name(CONTINUE_AUTO_TRYOUT_KIND, cursor)
            try:
                cont_anchor = scope.start_try_out_anchor(
                    anchor_name,
                    cursor.head,
                    kind=CONTINUE_AUTO_TRYOUT_KIND,
                    retry_limit=limit,
                    origin_cursor=cursor,
                )
                cont_anchor.retries_remaining -= 1
                _, tryout_cursor = cursor.add_try_out(anchor=cont_anchor)
                cursor = tryout_cursor
                new_cursor = cursor.add_continuation_turn()
                cursor.set_auto(True)
                new_cursor.set_auto(True)
                cursor.set_main_thread(True)
                new_cursor.set_main_thread(True)
                print(f"\n{Colors.SYSTEM}Response was truncated. Auto-retrying on new try_out branch... ({cont_anchor.retries_remaining} retries left){Colors.RESET}")
            except Exception as e:
                print(f"{Colors.ERROR}Failed to start auto-continuation try_out: {e}{Colors.RESET}")
                return cursor, False
        else: # limit is 0, so disabled
            return cursor, False

    # Schedule the next iteration if we have a new cursor
    if new_cursor and new_cursor.context:
        if scope:
            scope.request_auto_iteration()
        else:
            new_cursor.context.request_auto_iteration()
        return new_cursor, True

    return cursor, False

def _drain_and_close_auto_tryouts(
    cursor: Optional[ChatCursor],
    *,
    close_anchors: bool = True,
    context_override: Optional["ChatContext"] = None,
) -> Optional["ChatContext"]:
    """
    Best-effort helper to consume pending auto-iteration counters and, when requested,
    close any remaining auto_tool/auto_cont try-out anchors so completed branches
    return to the main thread.
    """
    ctx = context_override or (cursor.context if cursor else None)
    if not ctx:
        return None
    scope = _require_live_chat_scope(cursor)
    try:
        while True:
            if not scope.consume_auto_iteration():
                break
    except Exception:
        pass
    if close_anchors and cursor and cursor.current_turn:
        ancestors: Set[Turn] = set()
        node: Optional[Turn] = cursor.current_turn
        while node:
            ancestors.add(node)
            node = getattr(node, "parent", None)
        try:
            closed_cursor = _ensure_registered_cursor(
                scope.close_try_out_anchors_by_kind(
                    ["auto_tool", "auto_cont"],
                    dist_mode="keep",
                    anchor_scope=ancestors,
                )
            )
            if closed_cursor:
                _set_active_cursor(closed_cursor)
            else:
                try:
                    _set_active_cursor(scope.active_cursor())
                except Exception:
                    pass
        except Exception:
            pass
    # Align the global/current cursor to the context's active cursor after cleanup.
    if ctx:
        try:
            active = scope.active_cursor()
            if active:
                _set_active_cursor(active)
        except Exception:
            pass
    return ctx


def _normalize_cursor_after_auto_iters(
    cursor: Optional[ChatCursor],
    *,
    prefer_session_main_leaf: bool = False,
) -> Optional[ChatCursor]:
    """
    Rebinds a cursor after auto-try-out cleanup so future turns attach to the intended branch.
    """
    if not cursor:
        return None
    ctx_hint = getattr(cursor, "context", None)
    context = _active_chat_context()
    if not ctx_hint and context and getattr(cursor, "session", None) is context.session:
        if context.chat_session is None or getattr(cursor, "chat_session", None) is None:
            ctx_hint = context
        elif context.chat_session is getattr(cursor, "chat_session"):
            ctx_hint = context
    ctx = _drain_and_close_auto_tryouts(cursor, close_anchors=True, context_override=ctx_hint)
    scope = _require_live_chat_scope(cursor)
    if not ctx:
        return None
    try:
        active_for_scope = scope.active_cursor()
    except Exception:
        active_for_scope = None
    if not active_for_scope:
        try:
            bag_dict = _resolve_bag_dict(scope, ctx)
            if bag_dict and bag_dict.get("replay_debug"):
                print(f"{Colors.TOOL_WARNING}DEBUG: auto-iter cleanup failed; no active cursor.{Colors.RESET}")
        except Exception:
            pass
        return None
    normalized = active_for_scope
    try:
        normalized = normalized.descend_to_leaf()
    except Exception:
        pass
    try:
        bag_dict = _resolve_bag_dict(scope, ctx)
        if bag_dict and bag_dict.get("replay_debug"):
            label = normalized.display_id() if normalized else "None"
            print(f"{Colors.SYSTEM}DEBUG: auto-iter cleanup resolved to {label}.{Colors.RESET}")
    except Exception:
        pass
    return normalized

async def _execute_inference_round(
    *,
    cursor: ChatCursor,
    pt_session: "PromptSession",
    inference_payload: Dict[str, Any],
    active_adapters_for_turn: Optional[List[str]],
    tools_view_for_request: Optional[ToolsView],
    is_override_prompt: bool,
    override_adapters_for_response: Optional[List[str]],
    is_manual_continue: bool,
    llm_name_override: Optional[str] = None,
) -> bool:
    """Shared inference-and-stream handling used by chat loop and replay helpers.

    Returns True when the caller should immediately continue to the next loop iteration.
    """
    # Auto-mode intent (tools + continue):
    # - At most one auto_tool anchor per branch; tool calls create a try_out branch for results. Errors decrement retries; when exhausted we warn, close the anchor, and return to main.
    # - At most one auto_continue anchor per branch; first truncation starts it, later truncations on that branch reuse it. On exhaustion we warn, close the anchor, and return to main.
    # - Only the first auto do_continue spawns a try_out; subsequent auto retries stay on that try_out until the response finishes or retries expire.
    # - Truncated responses skip tool execution but can still auto-continue. Non-truncated responses with tool calls run tools even inside an auto_continue branch.
    # - Prompt path rule: only first-child (main) paths pull promoted right-sibling branches.
    # - Cleanup after auto rounds: close auto anchors, drop their try_out cursors, and rebind to the context's active cursor so main continues from the latest placeholder.
    _set_active_cursor(cursor, transient=True)

    responses_in_progress: Dict[int, str] = {}
    final_response_item_objects: List[InferenceResponse] = []
    accumulated_input_tokens, accumulated_output_tokens = 0, 0
    total_gen_duration_sum: float = 0.0
    first_token_latencies: List[float] = []
    accumulated_tool_blocks_count, accumulated_tool_block_tokens, in_flight_req = 0, 0, None
    mem_alloc, mem_rsvd = None, None
    cache_queued: Optional[str] = None

    final_metrics: Dict[str, Any] = {}
    auto_iteration_scheduled = False
    active_tools = None
    active_cursor_for_tools = _require_current_cursor()
    if tools_view_for_request:
        active_tools = active_cursor_for_tools.get_active_tools(tools_view_for_request)
    else:
        active_tools = active_cursor_for_tools.get_active_tools()

    if active_tools:
        print(f"{Colors.SYSTEM}Sending {len(active_tools['for_dump'])} active tool definition(s) to the model.{Colors.RESET}")

    if inference_payload.get("reset_metrics"):
        print(f"{Colors.SYSTEM}Note: This inference request includes 'reset_metrics=True'.{Colors.RESET}")

    try:
        was_canceled = False
        was_truncated = False
        llm_name = llm_name_override or cursor.context.chat_session.initial_params.get("engine_base_model_name") or "LLM"
        api_response = None
        try:
            api_response = await call_api("run-inference", inference_payload)  # type: ignore
        except Exception as e:  # type: ignore
            print(f"\n{Colors.ERROR}Error calling inference API: {e}{Colors.RESET}")
            _store_exception_traceback_if_clear(e)
            return True

        if not (isinstance(api_response, dict) and "stream" in api_response):
            print(f"\n{Colors.ERROR}Error from inference API: {api_response.get('message', 'Unknown error')}{Colors.RESET}")
            return True

        adapters_for_display = override_adapters_for_response or active_adapters_for_turn or []
        if not adapters_for_display:
            adapters_for_display = ["__base__"]
        display_plan = StreamDisplayPlan(
            per_prompt={
                0: StreamDisplayContext(
                    prompt_index=0,
                    show_override_banner=is_override_prompt,
                    override_total=len(override_adapters_for_response or []) if is_override_prompt else None,
                )
            },
            default_context=StreamDisplayContext(),
        )

        api_iterator = api_response.get("stream") if isinstance(api_response, dict) else None
        primary_response_item: Optional[InferenceResponse] = None
        if api_iterator:
            try:
                stream_result = await _consume_inference_stream(
                    api_iterator,
                    display_plan=display_plan,
                    llm_name=llm_name,
                    adapters_for_display=adapters_for_display,
                    cursor=cursor,
                )
            finally:
                await api_iterator.aclose()

            final_response_item_objects = stream_result.final_responses
            final_metrics = stream_result.stream_metrics or {}
            was_canceled = stream_result.was_canceled
            was_truncated = stream_result.was_truncated
            error_response_item:Optional[InferenceResponse] = None 

            if final_response_item_objects:
                if any(bool(item.was_canceled) for item in final_response_item_objects):
                    was_canceled = True
                if any(bool(item.was_truncated) for item in final_response_item_objects):
                    was_truncated = True
            responses_in_progress = stream_result.text_by_prompt
            primary_response_item = final_response_item_objects[0] if final_response_item_objects else None
            for response_item in final_response_item_objects:
                if not error_response_item and response_item.chunkType == ChunkType.ERROR:
                    error_response_item = response_item
                    
                extra_metrics: Dict[str, Any] = {}
                for agg_key in ("cache_queued", "in_flight_req", "mem_allocated", "mem_reserved"):
                    if final_metrics.get(agg_key) is not None:
                        extra_metrics[agg_key] = final_metrics[agg_key]
                metrics_payload = ChatCursor.update_response_metrics(
                    response=response_item,
                    extra=extra_metrics or None,
                )
                if metrics_payload:
                    cursor.update_metrics(metrics_payload)
            if final_metrics:
                accumulated_input_tokens = final_metrics.get("total_input_tokens", accumulated_input_tokens)
                accumulated_output_tokens = final_metrics.get("total_output_tokens", accumulated_output_tokens)
                total_gen_duration_sum = final_metrics.get("total_generation_duration_sec", total_gen_duration_sum)
                avg_ttft = final_metrics.get("avg_time_to_first_token_sec")
                if avg_ttft is not None:
                    first_token_latencies = [avg_ttft]
                if final_metrics.get("total_tool_blocks") is not None:
                    accumulated_tool_blocks_count = final_metrics.get("total_tool_blocks", accumulated_tool_blocks_count)
                if final_metrics.get("total_tool_blocks_tokens") is not None:
                    accumulated_tool_block_tokens = final_metrics.get("total_tool_blocks_tokens", accumulated_tool_block_tokens)
                if final_metrics.get("cache_queued"):
                    cache_queued = final_metrics.get("cache_queued")
                if final_metrics.get("in_flight_req") is not None:
                    in_flight_req = final_metrics.get("in_flight_req")
                if final_metrics.get("mem_allocated") is not None:
                    mem_alloc = final_metrics.get("mem_allocated")
                if final_metrics.get("mem_reserved") is not None:
                    mem_rsvd = final_metrics.get("mem_reserved")
        else:
            return True

        pt_session.app.invalidate()

        if final_metrics and (accumulated_input_tokens > 0 or accumulated_output_tokens > 0):
            elapsed_seconds_metrics = int(time.time() - cursor.session.creation_timestamp)
            metrics_line = "Metrics:"
            if accumulated_input_tokens is not None:
                metrics_line += f" In: {accumulated_input_tokens}"
            if accumulated_output_tokens is not None:
                metrics_line += f" Out: {accumulated_output_tokens}"

            overall_tps_stream = accumulated_output_tokens / total_gen_duration_sum if total_gen_duration_sum > 0 else 0.0
            if total_gen_duration_sum > 0:
                metrics_line += f" GenTime: {total_gen_duration_sum:.1f}s"
            if overall_tps_stream > 0:
                metrics_line += f" TPS: {overall_tps_stream:.1f}"
            if first_token_latencies:
                avg_ttft_stream = sum(first_token_latencies) / len(first_token_latencies)  # type: ignore
                metrics_line += f" Latency: {avg_ttft_stream * 1000:.0f}ms"

            final_tool_blocks_count = sum(item.tool_blocks_count or 0 for item in final_response_item_objects)
            final_tool_blocks_tokens = sum(item.tool_blocks_tokens or 0 for item in final_response_item_objects)

            if final_response_item_objects:
                first_final_item = final_response_item_objects[0]
                if (cache_metric := first_final_item.cache_metric):
                    metrics_line += f" Cache: {cache_metric}"
                if (cache_warming := first_final_item.cache_warming):
                    metrics_line += f" Warm-up: {cache_warming}"

            if final_tool_blocks_count > 0:
                metrics_line += f" Tool Blocks: {final_tool_blocks_count}"
            if final_tool_blocks_tokens > 0:
                metrics_line += f" Tool Tokens: {final_tool_blocks_tokens}"
            if cache_queued:
                metrics_line += f" Queued: {cache_queued}"
            if mem_alloc is not None:
                metrics_line += f" Mem(A): {mem_alloc:.0f}MB"
            if mem_rsvd is not None:
                metrics_line += f" Mem(R): {mem_rsvd:.0f}MB"
            if in_flight_req is not None:
                metrics_line += f" In-flight: {in_flight_req}"
            sys.stdout.write(f"{Colors.METRICS}[+{elapsed_seconds_metrics}s] {metrics_line}{Colors.RESET}\n")

        if inference_payload.get("reset_metrics"):
            _set_cursor_reset_metrics(cursor, False)

        any_tool_blocks = False
        all_tool_blocks: List[ToolCallBlock] = []
        for item in final_response_item_objects:
            if item.tool_blocks and len(item.tool_blocks) > 0:
                any_tool_blocks = True
                all_tool_blocks.extend(item.tool_blocks)


        if was_truncated and any_tool_blocks:
            for block in all_tool_blocks:
                block.action_block.extend([ToolCall.KeepRaw, ToolCall.Ignore])

        active_cursor = _require_current_cursor()

        if error_response_item:
            content_text = responses_in_progress.get(0, "") or (getattr(error_response_item, "chunk_text", None) or "")
            effective_was_canceled = was_canceled or bool(error_response_item.was_canceled)
            effective_was_truncated = bool(error_response_item.was_truncated)
            tool_blocks_for_error = error_response_item.tool_blocks
            if effective_was_truncated and tool_blocks_for_error:
                content_text, tool_blocks_for_error = _suppress_truncated_tool_blocks(
                    content_text,
                    tool_blocks_for_error,
                    cursor=active_cursor,
                )
            active_cursor.add_assistant(
                content=content_text,
                tool_blocks=tool_blocks_for_error,
                was_truncated=effective_was_truncated,
                was_canceled=effective_was_canceled,
                archived=True,
            )
            _record_response_adapters(active_cursor, error_response_item)
            active_cursor.record_error(
                error_response_item.error or "Unknown inference error",
                details=_extract_error_details(error_response_item),
            )
            return True

        elif not was_truncated and any_tool_blocks:
            active_chat_session = active_cursor.chat_session
            profile_to_use = active_chat_session.parser_profile if active_chat_session else _engine_parser_profile()
            active_toolbox = _active_toolbox()

            final_text_response = responses_in_progress.get(0, "")
            active_cursor.add_assistant(
                content=final_text_response or "",
                tool_blocks=all_tool_blocks,
                archived=False,
                do_continue=is_manual_continue
            )
            _record_response_adapters(active_cursor, primary_response_item)

            if active_toolbox:
                tool_retries_max, tool_retries_left = _tool_retry_counters(active_cursor)
                await active_toolbox.execute_request_tools(
                    parser_profile=profile_to_use,
                    final_response_items=final_response_item_objects,
                    action_handler=_tool_execution_action_handler,
                    serial_execution=True,
                    tools_view=tools_view_for_request,
                    pt_session=pt_session,
                    context=active_cursor.current_turn,
                    tool_retries_max=tool_retries_max,
                    tool_retries_left=tool_retries_left,
                )
            else:
                print(f"{Colors.TOOL_WARNING}Tool execution skipped: toolbox unavailable.{Colors.RESET}")

            if _tool_blocks_have_abort(all_tool_blocks):
                print(f"{Colors.TOOL_WARNING}Tool round aborted by action; skipping execution/results.{Colors.RESET}")
                return False

            tool_anchor = None
            if _tool_blocks_have_results(all_tool_blocks):
                active_cursor_for_tools = _require_current_cursor()
                scope = _scope_for_cursor(active_cursor_for_tools)
                if scope:
                    tool_anchor = scope.find_active_anchor("auto_tool", active_cursor_for_tools)
                else:
                    tool_anchor = active_cursor_for_tools.context.find_active_anchor(
                        "auto_tool",
                        active_cursor_for_tools,
                    )
                if not tool_anchor:
                    if scope:
                        tool_anchor = scope.start_try_out_anchor(
                            _auto_anchor_name("auto_tool", active_cursor_for_tools),
                            active_cursor_for_tools.head,
                            kind="auto_tool",
                            retry_limit=_auto_tool_retry_limit(),
                            origin_cursor=active_cursor_for_tools,
                        )
                    else:
                        tool_anchor = active_cursor_for_tools.context.start_try_out_anchor(
                            _auto_anchor_name("auto_tool", active_cursor_for_tools),
                            active_cursor_for_tools.head,
                            "auto_tool",
                            retry_limit=_auto_tool_retry_limit(),
                            origin_cursor=active_cursor_for_tools,
                        )

            if not tool_anchor:
                return False
            tool_has_error = _tool_call_has_error(all_tool_blocks)
            decrement_retry = tool_has_error

            if decrement_retry and tool_anchor.retries_remaining <= 0:
                warning = f"{Colors.TOOL_WARNING}Tool retries exhausted; returning to main without launching a new try_out.{Colors.RESET}"
                print(warning)
                try:
                    if scope:
                        closed_cursor = _ensure_registered_cursor(
                            scope.close_try_out_anchor(tool_anchor.anchor_name, dist_mode="keep")
                        )
                    else:
                        closed_cursor = _ensure_registered_cursor(
                            active_cursor_for_tools.context.close_try_out_anchor(
                                tool_anchor.anchor_name,
                                dist_mode="keep",
                            )
                        )
                    if closed_cursor:
                        _set_active_cursor(closed_cursor)
                except Exception:
                    pass
                _drain_and_close_auto_tryouts(cursor=active_cursor_for_tools, close_anchors=False)
                try:
                    scope = _scope_for_cursor(active_cursor_for_tools)
                    _set_active_cursor(scope.active_cursor() if scope else active_cursor_for_tools.context.active_cursor_for_scope(None))
                except Exception:
                    pass
                return False

            if not decrement_retry or tool_anchor.retries_remaining > 0:
                if decrement_retry:
                    tool_anchor.retries_remaining -= 1
                anchor_turn_for_retry = active_cursor_for_tools.current_turn or tool_anchor.anchor_turn or active_cursor_for_tools.head
                main_cursor_for_try, tryout_cursor = active_cursor_for_tools.add_try_out(
                    anchor=tool_anchor,
                    anchor_turn=anchor_turn_for_retry,
                    keep_in_main=True,
                )
                try:
                    tryout_cursor.set_main_thread(True)
                    main_cursor_for_try.set_main_thread(True)
                except Exception:
                    if tryout_cursor.head:
                        tryout_cursor.head.main_thread = True
                    if main_cursor_for_try.head:
                        main_cursor_for_try.head.main_thread = True
                active_cursor_for_tools = tryout_cursor
                tool_results_cursor = active_cursor_for_tools.add_tool_results(all_tool_blocks)
                tool_results_cursor.set_auto(True)
                try:
                    tool_results_cursor.set_main_thread(True)
                except Exception:
                    if tool_results_cursor.head:
                        tool_results_cursor.head.main_thread = True
                _set_active_cursor(_ensure_registered_cursor(tool_results_cursor) or tool_results_cursor)
                try:
                    scope = _scope_for_cursor(tool_results_cursor)
                    if scope:
                        scope.set_active_cursor(tool_results_cursor)
                    else:
                        tool_results_cursor.context.set_active_cursor(tool_results_cursor)
                except Exception:
                    pass
                print(f"{Colors.DIM}(Anchored tool try-out branch: {tool_results_cursor.display_id()}){Colors.RESET}")
                cursor_for_iteration = _require_current_cursor()
                if cursor_for_iteration.context:
                    scope = _scope_for_cursor(cursor_for_iteration)
                    if scope:
                        scope.request_auto_iteration()
                    else:
                        cursor_for_iteration.context.request_auto_iteration()
                    auto_iteration_scheduled = True
                return True

            warning = f"{Colors.TOOL_WARNING}Tool retries exhausted after repeated errors; skipping latest tool result and returning to main.{Colors.RESET}"
            print(warning)
            ancestors: Set[Turn] = set()
            node = active_cursor_for_tools.current_turn
            while node:
                ancestors.add(node)
                node = getattr(node, "parent", None)
            scope = _scope_for_cursor(active_cursor_for_tools)
            if scope:
                closed_cursor = _ensure_registered_cursor(
                    scope.close_try_out_anchors_by_kind(
                        ["auto_tool"],
                        dist_mode="keep",
                        anchor_scope=ancestors or None,
                    )
                )
            else:
                closed_cursor = _ensure_registered_cursor(
                    active_cursor_for_tools.context.close_try_out_anchors_by_kind(
                        ["auto_tool"],
                        dist_mode="keep",
                        anchor_scope=ancestors or None,
                    )
                )
            if closed_cursor:
                _set_active_cursor(closed_cursor)
                _drain_and_close_auto_tryouts(cursor=active_cursor_for_tools, close_anchors=False)
                try:
                    scope = _scope_for_cursor(active_cursor_for_tools)
                    _set_active_cursor(scope.active_cursor() if scope else active_cursor_for_tools.context.active_cursor_for_scope(None))
                except Exception:
                    pass

        elif was_truncated :
            final_text = responses_in_progress.get(0, "")
            tool_blocks_for_turn = all_tool_blocks if any_tool_blocks else None
            if tool_blocks_for_turn:
                final_text, tool_blocks_for_turn = _suppress_truncated_tool_blocks(
                    final_text,
                    tool_blocks_for_turn,
                    cursor=active_cursor,
                )
            active_cursor.add_assistant(
                final_text,
                tool_blocks=tool_blocks_for_turn,
                was_truncated=True,
                do_continue=is_manual_continue,
            )
            _record_response_adapters(active_cursor, primary_response_item)
            
            # Centralized continuation logic
            new_cursor, scheduled = await _handle_auto_continuation(active_cursor)
            if scheduled:
                _set_active_cursor(new_cursor)
                auto_iteration_scheduled = True
                return True # Inform the loop to continue

        elif is_override_prompt:
            final_text = responses_in_progress.get(0, "")
            active_cursor.add_assistant(
                final_text,
                tool_blocks=None,
                was_canceled=was_canceled,
                was_truncated=was_truncated,
                archived=was_canceled,
            )
            _record_response_adapters(active_cursor, primary_response_item)
        elif responses_in_progress:
            final_text = responses_in_progress.get(0, "")
            active_cursor.add_assistant(
                final_text,
                tool_blocks=None,
                was_canceled=was_canceled,
                was_truncated=was_truncated,
                archived=was_canceled,
            )
            _record_response_adapters(active_cursor, primary_response_item)
            # If no further tool calls were produced, close any pending auto-tool anchors
            # and normalize the cursor to the main placeholder leaf.
            try:
                _drain_and_close_auto_tryouts(cursor=active_cursor, close_anchors=True)
            except Exception:
                pass
        else:
            sys.stdout.write(f"\n{Colors.ERROR}(Warning: No response or empty stream; ending auto round){Colors.RESET}\n")
            ctx = None
            try:
                ctx = _drain_and_close_auto_tryouts(cursor=_require_current_cursor(), close_anchors=True)
            except Exception:
                ctx = None
            if ctx:
                try:
                    scope = _require_live_chat_scope(_require_current_cursor())
                    _set_active_cursor(scope.active_cursor())
                except Exception:
                    pass
            return False

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nClient-side interruption detected. Waiting for engine to confirm status...")
        await _wait_for_engine_ready()
    except Exception as e:
        traceback.print_exc()
        _store_exception_traceback_if_clear(e)
    finally:
        # Ensure auto try-out anchors are cleaned up before returning control.
        if not auto_iteration_scheduled:
            ctx = None
            try:
                ctx = _require_current_cursor().context
            except Exception:
                ctx = None
            if ctx:
                try:
                    # Keep the context's active cursor aligned with the one we just used.
                    scope = _require_live_chat_scope(_require_current_cursor())
                    scope.set_active_override(None)
                    scope.set_active_cursor(_require_current_cursor())
                except Exception as exc:
                    _log_cursor_warning("failed to align active cursor after round", exc)
                try:
                    active_turn = _require_current_cursor().current_turn
                except Exception:
                    active_turn = None
                ancestors: Set[Turn] = set()
                node = active_turn
                while node:
                    ancestors.add(node)
                    node = getattr(node, "parent", None)
                closed_cursor = _ensure_registered_cursor(
                    scope.close_try_out_anchors_by_kind(
                        ["auto_tool", "auto_cont"],
                        dist_mode="keep",
                        anchor_scope=ancestors or None,
                    )
                )
                if closed_cursor:
                    _set_active_cursor(closed_cursor)
    return False

async def _replay_format_prompt_command(cmd: Command, dest_cursor: ChatCursor) -> ChatCursor:
    """
    Replay helper for /fp: reruns the live handler against the destination cursor
    so the payload is rebuilt from the dest context (not the source).
    """
    if not dest_cursor:
        return dest_cursor

    command_text = str((cmd.data or {}).get("command") or "/fp").strip()
    api_params = getattr(cmd, "api_params", {}) or {}
    use_repr_flag = False
    text_to_format = ""
    gen_id: Optional[str] = None

    if isinstance(api_params, dict):
        gen_id = api_params.get("gen_id")
        use_repr_flag = bool(api_params.get("use_repr"))

    base_cursor = dest_cursor
    if gen_id:
        try:
            base_cursor = dest_cursor.cursor_for_gen_id(gen_id)
            print(f"{Colors.SYSTEM}Formatting prompt from context of turn {gen_id}.{Colors.RESET}")
        except (KeyError, ValueError) as e:
            print(f"{Colors.ERROR}Could not find turn for gen_id '{gen_id}': {e}{Colors.RESET}")
            return dest_cursor

    new_cursor, _ = await _handle_format_prompt_command(text_to_format, use_repr_flag, base_cursor, command_text)
    return new_cursor or dest_cursor


async def _replay_raw_command(cmd: Command, dest_cursor: ChatCursor) -> ChatCursor:
    """
    Replay helper for /raw: reruns the live handler on the destination cursor,
    using the raw text from the recorded command string.
    """
    api_params = getattr(cmd, "api_params", {}) or {}
    raw_list = api_params.get("raw_list") if isinstance(api_params, dict) else None
    raw_args = raw_list[0] if isinstance(raw_list, list) and raw_list else ""
    if not raw_args:
        command_text = str((cmd.data or {}).get("command") or "/raw").strip()
        if command_text.startswith("/"):
            match = re.match(r"/raw\s*(.*)", command_text, re.IGNORECASE | re.DOTALL)
            raw_args = match.group(1) if match else ""
        else:
            raw_args = command_text
    if not raw_args:
        print(f"{Colors.ERROR}Replay: cannot reconstruct /raw text; command was '{(cmd.data or {}).get('command')}'.{Colors.RESET}")
        return dest_cursor
    return await _handle_raw_command(raw_args, dest_cursor)


async def _mirror_turn_state(
    source_cursor: ChatCursor,
    dest_cursor: ChatCursor,
    *,
    replay_mode: bool = True,
    replay_debug: bool = False,
) -> ChatCursor:
    """Mirrors state-changing commands from a source turn to a destination cursor."""
    if not source_cursor.current_turn or not dest_cursor:
        return dest_cursor

    dest_session = dest_cursor.session
    source_turn = source_cursor.current_turn
    source_is_placeholder = bool(source_turn and source_turn.IsPlaceholderLike)

    def _same_replay_base_model() -> bool:
        try:
            source_session = source_cursor.chat_session
            dest_session_local = dest_cursor.chat_session
            if not source_session or not dest_session_local:
                return False
            source_params = getattr(source_session, "initial_params", {}) or {}
            dest_params = getattr(dest_session_local, "initial_params", {}) or {}
            source_model = source_params.get("engine_base_model_name")
            dest_model = dest_params.get("engine_base_model_name")
            return bool(source_model) and source_model == dest_model
        except Exception:
            return False

    def _append_command_copy(
        cmd_obj: Command,
        override_params: Optional[Dict[str, Any]] = None,
        override_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Command]:
        """Clone a command onto the destination turn without altering cursor stacks."""
        nonlocal dest_cursor
        if not dest_cursor.current_turn:
            return None
        metadata = copy.deepcopy(getattr(cmd_obj, "data", {}) or {})
        if override_metadata:
            metadata.update(override_metadata)
        if source_is_placeholder:
            clone = Command(
                cmd_type=cmd_obj.cmd_type,
                metadata=metadata,
                api_name=copy.deepcopy(getattr(cmd_obj, "api_name", None)),
                api_params=copy.deepcopy(
                    override_params if override_params is not None else getattr(cmd_obj, "api_params", None)
                ),
            )
            clone.timestamp = getattr(cmd_obj, "timestamp", time.time())
            new_turn = dest_session._add_command(clone, dest_cursor.current_turn)
            try:
                if dest_cursor.context:
                    scope = _scope_for_cursor(dest_cursor)
                    rebound = (
                        scope.register_cursor_for_turn(new_turn, make_active=False)
                        if scope else dest_cursor.context.register_cursor_for_turn(new_turn, make_active=False)
                    )
                    if rebound:
                        dest_cursor = rebound
            except Exception:
                pass
            return new_turn.cmd[-1] if getattr(new_turn, "cmd", None) else None
        if override_metadata:
            clone = dest_cursor.add_command_copy(cmd_obj, override_params=override_params)
            if clone is not None:
                try:
                    clone.data.update(metadata)
                except Exception:
                    pass
            return clone
        return dest_cursor.add_command_copy(cmd_obj, override_params=override_params)

    def _format_state_args(cmd_obj: Command) -> str:
        """Serialize state/param command payloads for logging."""
        payload: Dict[str, Any] = {}
        data = getattr(cmd_obj, "data", {}) or {}
        api_params = getattr(cmd_obj, "api_params", None)
        if data:
            payload["data"] = data
        if api_params:
            payload["api_params"] = api_params
        if not payload:
            return ""
        try:
            serialized = json.dumps(payload, default=str)
        except Exception:
            serialized = str(payload)
        return f" args={serialized}"

    # Log the replay source unless this routine is being reused for live routing.
    source_gen_id = source_cursor.current_turn.gen_id
    if replay_mode and source_gen_id:
        dest_cursor.log_command(f"replay: {source_gen_id}", metadata={"replay_source_gen_id": source_gen_id})

    commands_to_mirror = []
    replayable_commands = {Command.STATE_CHANGE, Command.PARAM_CHANGE, Command.COMMAND}
    for cmd in source_cursor.current_turn.cmd:
        if cmd.cmd_type not in replayable_commands:
            continue
        commands_to_mirror.append(cmd)
    if commands_to_mirror and replay_mode and replay_debug:
        try:
            labels = [getattr(cmd, "data", {}).get("command") or cmd.cmd_type for cmd in commands_to_mirror]
            print(f"{Colors.SYSTEM}Replay: mirroring {len(commands_to_mirror)} command(s) on dest turn {dest_cursor.display_id()}: {labels}{Colors.RESET}")
        except Exception:
            pass

    for cmd in commands_to_mirror:
        change = None
        try:
            cmd_text = str(cmd.data.get("command", "")).strip().lower()
            if replay_mode and (cmd_text.startswith("/f ") or cmd_text == "/f" or cmd_text.startswith("/flags")):
                print(f"{Colors.TOOL_WARNING}Replay: skipping flags command '{cmd.data.get('command')}'.{Colors.RESET}")
                continue
            change = cmd.data.get("change")
            op = cmd.data.get("op")
            scope_value = cmd.data.get("scope") or cmd.data.get("value")
            adapter_value = cmd.data.get("adapters") if "adapters" in cmd.data else cmd.data.get("value")
            if isinstance(adapter_value, str):
                adapter_value = [adapter_value]
            system_value = (
                cmd.data.get("value")
                if cmd.data.get("change") == "system_message"
                else (cmd.data.get("new_value") if cmd.cmd_type == Command.PARAM_CHANGE else cmd.data.get("value"))
            )
            command_text = cmd.data.get("command", f"/s replay {cmd.cmd_type}:{cmd.gen_id or ''}".strip())
            args_suffix = _format_state_args(cmd) if cmd.cmd_type in (Command.STATE_CHANGE, Command.PARAM_CHANGE) else ""
            stack_id = cmd.data.get("stack_id")

            if cmd.cmd_type == Command.COMMAND and cmd.api_name:
                if replay_mode and cmd.api_name == "load-adapter":
                    replay_data = getattr(cmd, "api_params", {}) or {}
                    adapter_name = replay_data.get("adapter_name")
                    original_path = replay_data.get("adapter_path")
                    replay_subpath = replay_data.get("replay_adapter_subpath")
                    replay_root = replay_data.get("replay_adapters_root")

                    loaded_via_checkpoint = bool(replay_data.get("replay_loaded_via_checkpoint"))
                    if not loaded_via_checkpoint and original_path:
                        path_obj = Path(original_path)
                        if path_obj.suffix and replay_subpath is None:
                            if replay_root:
                                try:
                                    path_obj.relative_to(Path(replay_root))
                                except Exception:
                                    loaded_via_checkpoint = True
                            else:
                                loaded_via_checkpoint = True

                    path_to_load = None
                    adapters_root_dir = (current_config.get("adapters_root_dir") if current_config else None)

                    if not adapters_root_dir:
                        print(f"{Colors.ERROR}Adapters root directory not configured. Cannot replay load-adapter.{Colors.RESET}")
                        continue

                    if loaded_via_checkpoint and _same_replay_base_model():
                        if original_path:
                            candidate = Path(original_path)
                            if candidate.exists() and candidate.is_file():
                                path_to_load = str(candidate)
                            else:
                                print(f"{Colors.TOOL_WARNING}Warning: Direct checkpoint path '{original_path}' not found during replay. Skipping.{Colors.RESET}")
                                continue
                        else:
                            print(f"{Colors.TOOL_WARNING}Warning: Missing checkpoint path for adapter replay. Skipping.{Colors.RESET}")
                            continue
                    else:
                        if not adapter_name:
                            print(f"{Colors.TOOL_WARNING}Warning: Missing adapter name during replay. Skipping.{Colors.RESET}")
                            continue
                        current_adapters_root = Path(adapters_root_dir)
                        if current_adapters_root.exists() and current_adapters_root.is_dir():
                            path_to_load = str(current_adapters_root)
                        else:
                            print(f"{Colors.TOOL_WARNING}Warning: Adapters root '{current_adapters_root}' not found during replay. Skipping.{Colors.RESET}")
                            continue

                    if not path_to_load:
                        print(f"{Colors.TOOL_WARNING}Warning: Could not find a path for adapter '{adapter_name}' during replay. Skipping.{Colors.RESET}")
                        continue
                    
                    # include_incompatible is always False for replay
                    api_payload = {
                        "adapter_name": adapter_name,
                        "adapter_path": path_to_load,
                        "include_incompatible": False,
                        "if_exists": IfExistsEnum.IGNORE,
                    }

                    record_payload = dict(api_payload)
                    if replay_subpath:
                        record_payload["replay_adapter_subpath"] = replay_subpath
                    elif adapter_name:
                        record_payload["replay_adapter_subpath"] = adapter_name
                    if adapters_root_dir:
                        record_payload["replay_adapters_root"] = str(adapters_root_dir)
                    record_payload["replay_loaded_via_checkpoint"] = bool(loaded_via_checkpoint)
                    cmd_copy = _append_command_copy(
                        cmd,
                        override_params=record_payload,
                    )
                    response = await call_api("load-adapter", api_payload)

                    if response.get("status") == "success":
                        details = response.get("data", {}) or {}
                        loaded_name = details.get("adapter_name") or adapter_name or "<unknown>"
                        endpoint = (
                            details.get("checkpoint_path")
                            or details.get("root_path")
                            or details.get("adapter_path")
                            or api_payload.get("adapter_path")
                            or "N/A"
                        )
                        if cmd_copy and endpoint:
                            cmd_copy.data["command"] = f"/a l {endpoint}"
                        no_op = None
                        if "already_loaded" in details:
                            no_op = bool(details.get("already_loaded"))
                        elif "was_loaded" in details:
                            no_op = not bool(details.get("was_loaded"))
                        elif "loaded" in details:
                            no_op = not bool(details.get("loaded"))
                        if no_op is True:
                            print(f"{Colors.SYSTEM}Replay: Adapter '{loaded_name}' already loaded (no-op). Endpoint: '{endpoint}'{Colors.RESET}")
                        else:
                            print(f"{Colors.SYSTEM}Replay: Loaded adapter '{loaded_name}' from '{endpoint}'{Colors.RESET}")
                    else:
                        print(f"{Colors.ERROR}Failed to re-load adapter '{adapter_name}' during replay: {response.get('message')}{Colors.RESET}")
                    continue
                
                if replay_mode:
                    print(f"{Colors.ECHO}[echo]{Colors.RESET} {command_text}")
                if replay_mode and cmd.api_name == "format-inference-prompt":
                    dest_cursor = await _replay_format_prompt_command(cmd, dest_cursor)
                    continue
                if replay_mode and cmd.api_name == "run-inference" and isinstance(getattr(cmd, "api_params", None), dict) and cmd.api_params.get("raw_list"):
                    dest_cursor = await _replay_raw_command(cmd, dest_cursor)
                    continue
                
                api_params = copy.deepcopy(getattr(cmd, "api_params", {}) or {})
                cmd_copy = _append_command_copy(cmd, override_params=api_params)
                response = await call_api(cmd.api_name, api_params)
                if replay_mode:
                    try:
                        _print_replay_api_response(cmd.api_name, cmd.data or {}, response, dest_cursor)
                    except Exception:
                        pass
                if cmd.api_name == "get-aggregate-metrics" and isinstance(response, dict):
                    if response.get("status") == "success" and cmd_copy and isinstance(response.get("data"), dict):
                        cmd_copy.data["$Response"] = copy.deepcopy(response.get("data") or {})
            elif cmd.cmd_type == Command.COMMAND:
                _append_command_copy(cmd)
                command_text = cmd.data.get('command', '')
                cmd_text_lower = str(command_text).strip().lower()

                if cmd.data.get("$Action") == "reset_metrics":
                    try:
                        _set_cursor_reset_metrics(dest_cursor, True, command_text=command_text)
                        if replay_mode:
                            print(f"{Colors.SYSTEM}Metrics will be reset on the next inference request.{Colors.RESET}")
                    except Exception:
                        pass
                
                action_name = cmd.data.get("$Action")
                is_echo = action_name == "echo"

                if action_name == "close_try_out":
                    action_args = cmd.data.get("$ActionArgs") or {}
                    anchor_name = action_args.get("anchor_name")
                    dist_mode = action_args.get("dist_mode") or "keep"
                    main_thread_index = action_args.get("main_thread_index")
                    if anchor_name and dest_cursor.context:
                        try:
                            scope = _scope_for_cursor(dest_cursor)
                            if not scope:
                                scope = dest_cursor.context.default_scope
                            updated_cursor = scope.close_try_out_anchor(
                                anchor_name,
                                dist_mode=dist_mode,
                                main_thread_index=main_thread_index,
                            )
                            if updated_cursor:
                                dest_cursor = updated_cursor
                            if replay_mode:
                                print(f"{Colors.SYSTEM}Replay: closed try-out anchor '{anchor_name}'.{Colors.RESET}")
                        except Exception as exc:
                            if replay_mode:
                                print(f"{Colors.TOOL_WARNING}Replay: failed to close try-out anchor '{anchor_name}': {exc}{Colors.RESET}")
                elif cmd_text_lower.startswith("/ct"):
                    ct_args = command_text.split(" ", 1)[1].strip() if " " in command_text else ""
                    parts = [p for p in ct_args.split() if p]
                    anchor_name = parts[0] if parts else None
                    dist_mode = "keep"
                    main_thread_index = None
                    if "--m" in parts:
                        m_index = parts.index("--m")
                        if m_index + 1 < len(parts):
                            dist_mode_arg = parts[m_index + 1]
                            if dist_mode_arg == "all":
                                dist_mode = "all"
                            elif dist_mode_arg == "none":
                                dist_mode = "none"
                            else:
                                try:
                                    main_thread_index = int(dist_mode_arg) - 1
                                    dist_mode = "index"
                                except ValueError:
                                    dist_mode = "keep"
                    if anchor_name and dest_cursor.context:
                        try:
                            scope = _scope_for_cursor(dest_cursor)
                            if not scope:
                                scope = dest_cursor.context.default_scope
                            updated_cursor = scope.close_try_out_anchor(
                                anchor_name,
                                dist_mode=dist_mode,
                                main_thread_index=main_thread_index,
                            )
                            if updated_cursor:
                                dest_cursor = updated_cursor
                            if replay_mode:
                                print(f"{Colors.SYSTEM}Replay: closed try-out anchor '{anchor_name}'.{Colors.RESET}")
                        except Exception as exc:
                            if replay_mode:
                                print(f"{Colors.TOOL_WARNING}Replay: failed to close try-out anchor '{anchor_name}': {exc}{Colors.RESET}")

                if replay_mode:
                    if is_echo:
                        action_args = cmd.data.get("$ActionArgs") or {}
                        note_text = action_args.get("text") or cmd.data.get('text') or command_text or ''
                        if note_text:
                            print(f"{Colors.ECHO}[echo]{Colors.RESET} {note_text}")
                    elif command_text:
                        print(f"{Colors.SYSTEM}Replay: {command_text}{Colors.RESET}")
            elif change == "adapters_command" and op:
                dest_cursor.apply_adapter_operation(
                    op,
                    adapter_value,
                    command_text=command_text,
                    stack_id=stack_id,
                )
                if replay_mode:
                    adapters_display = ", ".join(adapter_value) if isinstance(adapter_value, list) else adapter_value
                    preview = _format_replay_command_preview(
                        command_text,
                        stack_id=stack_id,
                        extra_text=adapters_display,
                    )
                    print(f"{Colors.SYSTEM}Replay: adapters {op} -> {preview}{Colors.RESET}")
            elif change == "system_message" and op:
                value_kind = cmd.data.get("value_kind")
                dest_cursor.apply_system_message(
                    op,
                    system_value,
                    command_text=command_text,
                    value_kind=value_kind,
                    stack_id=stack_id,
                )
                if replay_mode:
                    preview = _format_replay_command_preview(command_text, stack_id=stack_id)
                    print(f"{Colors.SYSTEM}Replay: system message {op} -> {preview}{Colors.RESET}")
            elif change == "tools_scope" and op:
                scope = ToolsScope.from_dict(scope_value) if isinstance(scope_value, dict) else None
                if cmd.data.get("context_scope") and dest_cursor.context and dest_cursor.context.toolbox_ref:
                    dest_cursor.context.toolbox_ref.set_scope(scope or ToolsScope())
                else:
                    dest_cursor.apply_tools_scope(
                        op,
                        scope,
                        command_text=command_text,
                        stack_id=stack_id,
                    )
                if replay_mode:
                    preview = _format_replay_command_preview(command_text, stack_id=stack_id)
                    print(f"{Colors.SYSTEM}Replay: tools scope {op} -> {preview}{Colors.RESET}")
            elif change and cmd.cmd_type in (Command.STATE_CHANGE, Command.PARAM_CHANGE):
                # Generic parameter/state change (covers /f flags including retry limits, maxnew, stream, etc.)
                payload_key = "new_value" if cmd.cmd_type == Command.PARAM_CHANGE else "value"
                new_val = cmd.data.get(payload_key)
                _set_chat_param_value(change, new_val, command_text=command_text)
                if replay_mode:
                    print(f"{Colors.SYSTEM}Replay: set '{change}' -> {new_val}{args_suffix}.{Colors.RESET}")
        except Exception as exc:
            msg = f"Replay: failed to mirror command '{change or cmd.cmd_type}': {exc}"
            if replay_mode:
                raise RuntimeError(msg) from exc
            print(f"{Colors.TOOL_WARNING}{msg}{Colors.RESET}")

    return dest_cursor


def _print_replay_api_response(api_name: str, cmd_data: Dict[str, Any], response: Any, dest_cursor: ChatCursor) -> None:
    """Mirror the user-facing prints for API commands during replay."""
    status = response.get("status") if isinstance(response, dict) else None
    data = response.get("data", {}) if isinstance(response, dict) else {}
    cmd_text = str(cmd_data.get("command", "")).lower()
    api_params = cmd_data.get("api_params", {}) or {}

    if api_name == "get-aggregate-metrics":
        if status == "success":
            print("\n--- Aggregate Engine Metrics ---")
            print(json.dumps(data, indent=2, default=str))
            print("------------------------------\n")
        else:
            print(f"Failed to get aggregate metrics: {response.get('message') if isinstance(response, dict) else response}")
        return

    if api_name == "get-engine-status":
        if status == "success":
            print("\n--- Engine Status ---")
            print(json.dumps(data, indent=2, default=str))
            print("---------------------\n")
        else:
            print(f"Failed to get engine status: {response.get('message') if isinstance(response, dict) else response}")
        return

    if api_name == "unload-adapter":
        adapter_name = api_params.get("adapter_name") or cmd_data.get("adapter_name")
        if status == "success":
            print(f"{Colors.SYSTEM}Adapter '{adapter_name}' unloaded successfully from engine.{Colors.RESET}")
        else:
            print(f"{Colors.ERROR}Failed to unload adapter '{adapter_name}': {response.get('message') if isinstance(response, dict) else response}{Colors.RESET}")
        return

    if api_name == "load-adapter":
        adapter_name = api_params.get("adapter_name") or cmd_data.get("adapter_name")
        if status == "success":
            loaded_name = (data or {}).get("adapter_name") or adapter_name
            if loaded_name:
                print(f"{Colors.SYSTEM}Engine reported adapter '{loaded_name}' loaded successfully.{Colors.RESET}")
            else:
                print(f"{Colors.SYSTEM}Adapter loaded (path sent: {api_params.get('adapter_path','<unknown>')}), but name couldn't be confirmed. Check /a e.{Colors.RESET}")
        else:
            print(f"{Colors.ERROR}Failed to load adapter: {response.get('message') if isinstance(response, dict) else response}{Colors.RESET}")
            if isinstance(response, dict) and response.get("details"):
                try:
                    print(f"  Details: {json.dumps(response.get('details'), indent=2)}")
                except Exception:
                    pass
        return

    if api_name == "get-adapter-details":
        target_name = api_params.get("adapter_name") or cmd_data.get("adapter_name") or "<unknown>"
        if status != "success":
            print(f"{Colors.ERROR}Failed to get details for adapter '{target_name}': {response.get('message') if isinstance(response, dict) else response}{Colors.RESET}")
            return
        adapter_details_dict = data or {}
        print(f"\n{Colors.HEADER}--- Adapter Details ---{Colors.RESET}")
        marker = " (active)" if adapter_details_dict.get("is_active") else ""
        print(f"\n{Colors.BOLD}{Colors.WHITE}Adapter: {target_name}{marker}{Colors.RESET}")
        print(f"  {Colors.SYSTEM}Type:{Colors.RESET} {adapter_details_dict.get('type', 'N/A')}")
        print(f"  {Colors.SYSTEM}Root Path:{Colors.RESET} {adapter_details_dict.get('root_path', 'N/A')}")
        if checkpoint_path := adapter_details_dict.get("checkpoint_path"):
            print(f"  {Colors.SYSTEM}Checkpoint:{Colors.RESET} {checkpoint_path}")
        config_dump = adapter_details_dict.get('config') or adapter_details_dict.get('peft_config_on_model')
        if config_dump:
            try:
                config_pretty = json.dumps(json.loads(config_dump) if isinstance(config_dump, str) else config_dump, indent=2)
            except Exception:
                config_pretty = str(config_dump)
            print(f"  {Colors.SYSTEM}Config:{Colors.RESET} {config_pretty}")
        else:
            print(f"  {Colors.SYSTEM}Config:{Colors.RESET} Not available")
        return

    if api_name == "get-loaded-adapters":
        if status != "success":
            print(f"{Colors.ERROR}Failed to get adapter list: {response.get('message') if isinstance(response, dict) else response}{Colors.RESET}")
            return
        all_adapters = data.get("adapters", []) if isinstance(data, dict) else []
        if not all_adapters:
            print("No adapters currently loaded in the engine.")
            return
        print(f"\n{Colors.HEADER}--- All Loaded Adapters ---{Colors.RESET}")
        for adapter_info in all_adapters:
            name = adapter_info.get("name")
            marker = " (active)" if adapter_info.get("is_active") else ""
            print(f"\n{Colors.BOLD}{Colors.WHITE}Adapter: {name}{marker}{Colors.RESET}")
            print(f"  {Colors.SYSTEM}Type:{Colors.RESET} {adapter_info.get('type', 'N/A')}")
            print(f"  {Colors.SYSTEM}Root Path:{Colors.RESET} {adapter_info.get('root_path', 'N/A')}")
            if checkpoint_path := adapter_info.get("checkpoint_path"):
                print(f"  {Colors.SYSTEM}Checkpoint:{Colors.RESET} {checkpoint_path}")
        print(f"{Colors.HEADER}---{Colors.RESET}")
        return

    if api_name == "count-tokens":
        use_repr = "-repr" in cmd_text
        if status == "success":
            result = data or {}
            token_count = result.get("token_count")
            text_processed = result.get("text_processed")
            print(f"\n--- Token Count ---")
            print(f"  Token Count: {token_count}")
            if use_repr and text_processed is not None:
                print(f"  Text (decoded from repr): {text_processed}")
            print(f"-------------------\n")
        else:
            print(f"Error counting tokens: {response.get('message', 'Unknown error') if isinstance(response, dict) else response}")
        return

    if api_name == "format-inference-prompt":
        use_repr = "-repr" in cmd_text or cmd_text.startswith("/fpr")
        if status in ["success"]:
            formatted_prompts = (data or {}).get("formatted_prompts", [])
            token_counts = (data or {}).get("prompt_token_counts", [])
            if formatted_prompts:
                token_count_info = ""
                if token_counts and len(token_counts) > 0:
                    token_count_info = f" | Tokens: {token_counts[0]}"
                header_text = "Formatted Prompt (repr)" if use_repr else "Formatted Prompt"
                print(f"\n--- {header_text}{token_count_info} ---")
                if use_repr:
                    print(repr(formatted_prompts[0]))
                else:
                    print(formatted_prompts[0])
                print("\n------------------------------\n")
            else:
                print(f"{Colors.TOOL_WARNING}No formatted prompt returned by engine.{Colors.RESET}")
        else:
            print(f"{Colors.ERROR}format-inference-prompt failed: {response.get('message','Unknown error') if isinstance(response, dict) else response}{Colors.RESET}")
        return

async def _prepare_replay_batch_turn(
    source_turn: Turn,
    dest_parent_cursor: ChatCursor,
    driver_cursor: ChatCursor,
    pt_session: "PromptSession",
    source_to_dest_cursor_map: Dict[Turn, Dict[str, Any]],
    *,
    replay_mode: bool = True,
    skip_is_auto: bool = True,
    replay_debug: bool = False,
) -> Tuple[ChatCursor, List[Turn], List[Dict[str, Any]], int]:
    """
    Prepare batch scaffolding and child cursors for replay without running inference.
    Returns (batch_hub_cursor, kept_children, active_forks, concurrency_level).
    """
    batch_children: List[Turn] = [
        child for child in (getattr(source_turn, "turns", []) or [])
        if isinstance((getattr(child, "data", None) or {}).get("user"), dict)
    ]
    if len(batch_children) > 1:
        # Preserve prompt order using creation sequence (gen_id counter).
        batch_children.sort(key=lambda t: getattr(t, "sort_id", 0))
    kept_children: List[Turn] = [
        child for child in batch_children
        if not (skip_is_auto and getattr(child, "is_auto", False))
    ]

    prompts: List[str] = []
    override_adapters: List[str] = []

    for child in kept_children:
        user_payload = (child.data or {}).get("user") or {}
        prompts.append(user_payload.get("content", "") if isinstance(user_payload, dict) else "")
        override_adapters.append((child.data or {}).get("adapter_override_name"))

    if not kept_children:
        return dest_parent_cursor, [], [], 0

    command_text = (source_turn.data or {}).get("command") or "/g b"
    try:
        fork = dest_parent_cursor.add_batch(
            command_text=command_text,
            prompts=prompts,
            override_adapters=override_adapters if any(override_adapters) else None,
            make_active=False,
            adopt_into_context=False,
        )
        fork.prompt_indices = list(range(len(fork.cursors)))
        fork.cursor_meta = fork.cursor_meta or {}
        batch_hub_cursor = getattr(fork, "batch_hub", None)
        chat_turns = [c.current_turn for c in getattr(fork, "cursors", [])]
        if not batch_hub_cursor or not batch_hub_cursor.current_turn or not chat_turns:
            raise RuntimeError("Batch creation returned no hub or children.")
    except Exception as exc:
        if replay_mode:
            print(f"{Colors.TOOL_WARNING}Replay: could not mirror batch turn: {exc}{Colors.RESET}")
        return dest_parent_cursor, [], [], 0

    dest_batch_cursor = batch_hub_cursor
    batch_turn = dest_batch_cursor.current_turn if dest_batch_cursor else None

    # Mirror any other commands from the source batch turn itself.
    if replay_mode and replay_debug:
        print(f"{Colors.SYSTEM}DEBUG: replay: mirroring commands from source batch turn {source_turn.gen_id_or_parent} to dest {dest_batch_cursor.display_id()}{Colors.RESET}")

    await _mirror_turn_state(
        source_cursor=driver_cursor.clone_at(source_turn),
        dest_cursor=dest_batch_cursor,
        replay_mode=replay_mode,
        replay_debug=replay_debug,
    )

    if source_turn.gen_id and batch_turn and dest_batch_cursor:
        source_to_dest_cursor_map.setdefault(
            source_turn, _make_replay_mapping_entry(source_turn, dest_batch_cursor)
        )

    # Seed source->dest mappings for batch children before replaying results.
    # This keeps replay parent resolution anchored on the correct child, not the batch hub.
    for idx, child in enumerate(kept_children):
        if idx < len(getattr(fork, "cursors", []) or []):
            child_cursor = fork.cursors[idx]
            if child_cursor and child_cursor.current_turn:
                source_to_dest_cursor_map.setdefault(
                    child, _make_replay_mapping_entry(child, child_cursor)
                )

    prepared_children: List[Tuple[Turn, ChatCursor]] = []
    for idx, child in enumerate(kept_children):
        dest_child_cursor = dest_parent_cursor.clone_at(chat_turns[idx]) if idx < len(chat_turns) else dest_batch_cursor
        mirrored_cursor = await _mirror_turn_state(
            source_cursor=driver_cursor.clone_at(child),
            dest_cursor=dest_child_cursor,
            replay_mode=replay_mode,
            replay_debug=replay_debug,
        )
        if mirrored_cursor.current_turn and getattr(mirrored_cursor.current_turn, "IsStructural", False):
            try:
                mirrored_cursor = mirrored_cursor.descend_to_leaf()
            except Exception as exc:
                raise RuntimeError(
                    f"Replay batch child failed to descend to leaf: src={child.gen_id_or_parent} "
                    f"dest={mirrored_cursor.display_id()} err={exc}"
                ) from exc
        if replay_mode:
            try:
                user_payload = (child.data or {}).get("user") or {}
                turn_idx = mirrored_cursor.user_turns_count()
                print(f"{Colors.YOU_HEADER}Turn {turn_idx}->You:{Colors.RESET} {Colors.YOU_CONTENT}{user_payload.get('content','')}{Colors.RESET}")
            except Exception as exc:
                if replay_debug:
                    print(f"{Colors.TOOL_WARNING}Replay: failed to print user turn label: {exc}{Colors.RESET}")
        prepared_children.append((child, mirrored_cursor))

    if not prepared_children:
        return dest_batch_cursor, [], [], 0

    active_forks: List[Dict[str, Any]] = []
    for idx, _ in enumerate(prepared_children):
        active_forks.append({
            "original_index": idx,
            "auto_tool_anchor": None,
            "fork": fork,
            "cursor_idx": idx,
            "source_turn": kept_children[idx] if idx < len(kept_children) else None,
        })

    # Each batch node represents one request (live /g b, /g bc, /g ao split into nodes already).
    concurrency_level = 1 if active_forks else 0
    return dest_batch_cursor, kept_children, active_forks, concurrency_level


def _build_replay_child_cursor_map(
    kept_children: List[Turn],
    dest_batch_cursor: ChatCursor,
    dest_parent_cursor: ChatCursor,
    source_to_dest_cursor_map: Dict[Turn, Dict[str, Any]],
) -> Dict[Turn, ChatCursor]:
    child_cursor_map: Dict[Turn, ChatCursor] = {}
    for child in kept_children:
        entry = source_to_dest_cursor_map.get(child) if source_to_dest_cursor_map else None
        dest_cursor_for_child = entry.get("cursor") if isinstance(entry, dict) else None
        if not dest_cursor_for_child or not getattr(dest_cursor_for_child, "current_turn", None):
            raise RuntimeError(
                f"Replay cursor mapping missing for batch child {child.gen_id_or_parent}; "
                "expected a destination cursor for child replay."
            )
        child_cursor_map[child] = dest_cursor_for_child
    return child_cursor_map


def _make_replay_mapping_entry(source_turn: Turn, dest_cursor: ChatCursor) -> Dict[str, Any]:
    anchor_turn = getattr(dest_cursor, "current_turn", None)
    if anchor_turn:
        try:
            source_is_placeholder = bool(
                getattr(source_turn, "IsPlaceholderLike", False)
                or (getattr(source_turn, "data", None) or {}).get("$try_out")
            )
        except Exception:
            source_is_placeholder = False
        if getattr(anchor_turn, "IsPlaceholderLike", False) and not source_is_placeholder:
            anchor_turn = getattr(anchor_turn, "parent", None) or anchor_turn
    return {"cursor": dest_cursor, "anchor": anchor_turn}


async def _replay_all_down(
    driver_cursor: ChatCursor,
    dest_cursor: ChatCursor,
    pt_session: "PromptSession",
    *,
    replay_mode: bool = True,
    replay_debug: bool = False,
) -> ChatCursor:
    driver_context = ChatContext(driver_cursor.session, chat_session=driver_cursor.chat_session, toolbox=toolbox)
    scope_turns = driver_context.get_scope_turns(start_node=driver_cursor.current_turn, suppress_auto=True)
    if replay_debug and scope_turns:
        try:
            ids = [t.gen_id_or_parent for t in scope_turns]
            print(f"{Colors.SYSTEM}DEBUG: get_scope_turns returned {len(ids)} turn(s): {ids}{Colors.RESET}")
        except Exception as exc:
            print(f"{Colors.TOOL_WARNING}Replay: failed to render scope turn ids: {exc}{Colors.RESET}")

    if not scope_turns:
        if replay_mode:
            print(f"{Colors.TOOL_WARNING}Replay: no turns found in scope for all-down mode.{Colors.RESET}")
        return dest_cursor

    source_to_dest_cursor_map: Dict[Turn, Dict[str, Any]] = {}
    branch_map: Dict[Turn, ChatCursor] = {}
    ignored_branches: Set[Turn] = set()
    failed_branches: Dict[Turn, str] = {}
    handled_turns: Set[Turn] = set()

    def _cursor_for(dest: ChatCursor, turn: Optional[Turn]) -> ChatCursor:
        if not turn:
            return _ensure_registered_cursor(dest) or dest
        if dest and dest.context:
            rebound = dest.context.find_cursor_for_turn(turn)
            if rebound:
                return rebound
        try:
            cloned = dest.clone_at(turn)
        except Exception as exc:
            raise RuntimeError(
                f"Replay failed to clone cursor at turn {getattr(turn, 'gen_id_or_parent', 'N/A')}: {exc}"
            ) from exc
        return _ensure_registered_cursor(cloned) or cloned

    def _spawn_try_out_cursor(src_turn: Turn, dest_parent: ChatCursor) -> Tuple[Optional[ChatCursor], Optional[ChatCursor]]:
        anchor = None
        try_meta = (getattr(src_turn, "data", {}) or {}).get("$try_out", {}) if getattr(src_turn, "data", None) else {}
        anchor_name = try_meta.get("anchor")
        anchor_kind = try_meta.get("kind", "try_out")
        if anchor_name and dest_parent.context:
            ctx = dest_parent.context
            try:
                scope = _require_live_chat_scope(dest_parent)
                anchor = scope.get_try_out_anchor(anchor_name) or scope.start_try_out_anchor(
                    anchor_name,
                    dest_parent.current_turn,
                    kind=anchor_kind,
                    origin_cursor=dest_parent,
                )
            except Exception as exc:
                anchor = None  # TODO: anchor recreation failed; try-out will be unanchored.
                if replay_mode:
                    print(f"{Colors.TOOL_WARNING}Replay: failed to rebuild try-out anchor {anchor_name}: {exc}{Colors.RESET}")
        try:
            main_cursor, try_cursor = dest_parent.add_try_out(anchor=anchor, anchor_turn=dest_parent.current_turn)
            return main_cursor, try_cursor
        except Exception as exc:
            if replay_mode:
                print(f"{Colors.TOOL_WARNING}Replay: failed to mirror try-out for {src_turn.gen_id_or_parent}: {exc}{Colors.RESET}")
            return None, None

    def _record_mapping(src_turn: Turn, dest_cursor: Optional[ChatCursor]) -> None:
        if not src_turn or not dest_cursor or not getattr(dest_cursor, "current_turn", None):
            return
        if src_turn in source_to_dest_cursor_map:
            return
        source_to_dest_cursor_map[src_turn] = _make_replay_mapping_entry(src_turn, dest_cursor)

    def _validate_mapping(src_turn: Turn, dest_turn: Optional[Turn]) -> None:
        if not src_turn or not dest_turn:
            raise RuntimeError(f"Replay mapping missing for source {getattr(src_turn, 'gen_id_or_parent', 'N/A')}.")
        src_parent = getattr(src_turn, "parent", None)
        expected_parent = _find_mapped_ancestor(src_parent) if src_parent else None
        actual_parent = getattr(dest_turn, "parent", None)
        if expected_parent:
            try:
                if not _is_descendant_of(dest_turn, expected_parent):
                    raise RuntimeError(
                        "Replay mapping parent mismatch: "
                        f"src={src_turn.gen_id_or_parent} "
                        f"dest={dest_turn.gen_id_or_parent} "
                        f"src_parent={getattr(src_parent, 'gen_id_or_parent', 'N/A')} "
                        f"expected_parent={getattr(expected_parent, 'gen_id_or_parent', 'N/A')} "
                        f"actual_parent={getattr(actual_parent, 'gen_id_or_parent', 'N/A')}"
                    )
            except Exception as exc:
                raise RuntimeError(
                    "Replay mapping parent check failed: "
                    f"src={src_turn.gen_id_or_parent} "
                    f"dest={dest_turn.gen_id_or_parent} "
                    f"expected_parent={getattr(expected_parent, 'gen_id_or_parent', 'N/A')} "
                    f"err={exc}"
                ) from exc

    def _update_branch(root_turn: Turn, cursor: Optional[ChatCursor]) -> None:
        if cursor and cursor.current_turn:
            branch_map[root_turn] = cursor
        else:
            failed_branches.setdefault(root_turn, "destination cursor unavailable")
            ignored_branches.add(root_turn)
            # TODO: recover branch mapping when destination cursor disappears unexpectedly.

    def _mark_failed_branch(root_turn: Turn, reason: str) -> None:
        failed_branches.setdefault(root_turn, reason)
        ignored_branches.add(root_turn)

    def _find_mapped_ancestor(turn: Optional[Turn]) -> Optional[Turn]:
        current = turn
        while current:
            entry = source_to_dest_cursor_map.get(current) if source_to_dest_cursor_map else None
            anchor = entry.get("anchor") if isinstance(entry, dict) else None
            if anchor:
                return anchor
            current = getattr(current, "parent", None)
        return None

    source_root = getattr(driver_cursor.chat_session, "root_turn", None)
    dest_root = getattr(dest_cursor.chat_session, "root_turn", None)
    if source_root and dest_root and source_root not in source_to_dest_cursor_map:
        try:
            # Use a dedicated root cursor to avoid drift from active cursor mutations.
            root_cursor = dest_cursor.clone_at(dest_root)
            source_to_dest_cursor_map[source_root] = _make_replay_mapping_entry(source_root, root_cursor)
        except Exception:
            pass

    root_branch = scope_turns[0].branch_root()
    branch_map[root_branch] = dest_cursor

    for source_turn in scope_turns:
        if REPLAY_CANCELLATION_EVENT.is_set():
            if replay_mode:
                print(f"{Colors.TOOL_WARNING}Replay canceled by user.{Colors.RESET}")
            break
        if getattr(source_turn, "is_auto", False):
            continue
        if source_turn in handled_turns:
            continue

        branch_root = source_turn.branch_root()
        if branch_root in failed_branches:
            if replay_mode:
                print(f"{Colors.TOOL_WARNING}Replay: skipping failed branch {branch_root.gen_id_or_parent}: {failed_branches[branch_root]}{Colors.RESET}")
            continue
        if branch_root in ignored_branches:
            continue

        if branch_root not in branch_map and source_turn.is_try_out_turn():
            parent = getattr(source_turn, "parent", None)
            parent_root = parent.branch_root() if parent else None  # type: ignore
            parent_cursor = branch_map.get(parent_root) if parent_root else None
            if not parent_cursor:
                _mark_failed_branch(branch_root, "missing parent cursor for try-out replay")
                continue
            mapped_parent = _find_mapped_ancestor(parent)
            if parent and not mapped_parent:
                _mark_failed_branch(branch_root, "missing mapped ancestor for try-out parent")
                continue
            parent_cursor = _cursor_for(parent_cursor, mapped_parent)  # type: ignore
            main_cursor, try_cursor = _spawn_try_out_cursor(source_turn, parent_cursor)
            if not try_cursor:
                _mark_failed_branch(branch_root, "failed to create try-out cursor")
                continue
            if parent_root:
                branch_map[parent_root] = main_cursor or parent_cursor
            branch_map[branch_root] = try_cursor

        dest_branch_cursor = branch_map.get(branch_root)
        if not dest_branch_cursor:
            _mark_failed_branch(branch_root, "missing destination branch cursor")
            continue

        parent = getattr(source_turn, "parent", None)
        mapped_parent = _find_mapped_ancestor(parent)
        if parent and not mapped_parent:
            if branch_root in failed_branches:
                if replay_mode:
                    print(f"{Colors.TOOL_WARNING}Replay: missing mapped ancestor for {parent.gen_id_or_parent}; skipping failed branch.{Colors.RESET}")
                continue
            print(f"{Colors.ERROR}Replay error: missing mapped ancestor for {parent.gen_id_or_parent}. Aborting replay.{Colors.RESET}")
            return dest_cursor
        effective_parent = mapped_parent
        effective_cursor = dest_branch_cursor
        if not getattr(source_turn, "IsPlaceholderLike", False) and mapped_parent and dest_branch_cursor.current_turn:
            try:
                leaf_cursor = dest_branch_cursor.descend_to_leaf()
                if leaf_cursor and leaf_cursor.current_turn and _is_descendant_of(leaf_cursor.current_turn, mapped_parent):
                    effective_cursor = leaf_cursor
                    effective_parent = leaf_cursor.current_turn
                elif _is_descendant_of(dest_branch_cursor.current_turn, mapped_parent):
                    effective_parent = dest_branch_cursor.current_turn
            except Exception:
                pass
        if effective_cursor and effective_parent and getattr(effective_cursor, "current_turn", None) is effective_parent:
            dest_cursor_for_turn = effective_cursor
        else:
            dest_cursor_for_turn = _cursor_for(effective_cursor, effective_parent)
        if mapped_parent and dest_branch_cursor.current_turn and dest_cursor_for_turn.current_turn:
            if not _is_descendant_of(dest_branch_cursor.current_turn, mapped_parent):
                # Parent mapping indicates a branch jump; re-anchor this branch to the mapped parent.
                branch_map[branch_root] = dest_cursor_for_turn

        # Guard against mis-anchored destination cursors before replaying this turn.
        if effective_parent and dest_cursor_for_turn.current_turn is not effective_parent:
            err = (
                "Replay cursor anchor mismatch: "
                f"src={source_turn.gen_id_or_parent} "
                f"src_parent={getattr(parent, 'gen_id_or_parent', 'N/A')} "
                f"expected_parent={effective_parent.gen_id_or_parent} "
                f"cursor_at={dest_cursor_for_turn.current_turn.gen_id_or_parent}"
            )
            _mark_failed_branch(branch_root, err)
            if replay_mode:
                print(f"{Colors.ERROR}Replay error: {err}{Colors.RESET}")
            return dest_cursor

        try_meta = (getattr(source_turn, "data", None) or {}).get("$try_out")
        if getattr(source_turn, "IsPlaceholderLike", False) and isinstance(try_meta, dict):
            try_kind = try_meta.get("kind")
            if try_kind in {"auto_tool", TOOL_AUTO_TRYOUT_KIND, "auto_cont", CONTINUE_AUTO_TRYOUT_KIND}:
                if mapped_parent:
                    dest_placeholder_turn = None
                    anchor_name = try_meta.get("anchor")
                    try:
                        candidates = [
                            child for child in (getattr(mapped_parent, "turns", []) or [])
                            if getattr(child, "IsPlaceholderLike", False)
                        ]
                        if anchor_name:
                            for child in candidates:
                                child_try = (getattr(child, "data", None) or {}).get("$try_out") or {}
                                if child_try.get("anchor") == anchor_name:
                                    dest_placeholder_turn = child
                                    break
                        if not dest_placeholder_turn:
                            for child in candidates:
                                child_try = (getattr(child, "data", None) or {}).get("$try_out") or {}
                                child_kind = child_try.get("kind")
                                if child_kind in {"auto_tool", TOOL_AUTO_TRYOUT_KIND, "auto_cont", CONTINUE_AUTO_TRYOUT_KIND}:
                                    dest_placeholder_turn = child
                                    break
                    except Exception as exc:
                        dest_placeholder_turn = None
                        if replay_mode and replay_debug:
                            print(f"{Colors.TOOL_WARNING}Replay: failed to resolve auto-tool placeholder under {mapped_parent.gen_id_or_parent}: {exc}{Colors.RESET}")
                    if not dest_placeholder_turn:
                        err = (
                            "Replay mapping missing auto-tool placeholder: "
                            f"src={source_turn.gen_id_or_parent} "
                            f"parent={mapped_parent.gen_id_or_parent}"
                        )
                        _mark_failed_branch(branch_root, err)
                        if replay_mode:
                            print(f"{Colors.ERROR}Replay error: {err}{Colors.RESET}")
                        return dest_cursor
                    _record_mapping(source_turn, dest_cursor_for_turn)
                    try:
                        _validate_mapping(source_turn, dest_placeholder_turn)
                    except Exception as exc:
                        _mark_failed_branch(branch_root, str(exc))
                        if replay_mode:
                            print(f"{Colors.ERROR}Replay error: {exc}{Colors.RESET}")
                        return dest_cursor
                handled_turns.add(source_turn)
                continue
            anchor_name = try_meta.get("anchor")
            anchor_kind = try_meta.get("kind", "try_out")
            if anchor_name:
                dest_placeholder_turn = None
                dest_placeholder_cursor = None

                source_is_try = False
                try:
                    source_is_try = source_turn.is_try_out_turn()
                except Exception as exc:
                    source_is_try = False
                    if replay_mode and replay_debug:
                        print(f"{Colors.TOOL_WARNING}Replay: failed to read try-out flag for {source_turn.gen_id_or_parent}: {exc}{Colors.RESET}")

                if mapped_parent:
                    try:
                        anchor_children = [
                            child for child in (getattr(mapped_parent, "turns", []) or [])
                            if isinstance((getattr(child, "data", None) or {}).get("$try_out"), dict)
                            and (getattr(child, "data", None) or {}).get("$try_out", {}).get("anchor") == anchor_name
                        ]
                        if anchor_children:
                            if source_is_try:
                                dest_placeholder_turn = anchor_children[1] if len(anchor_children) > 1 else anchor_children[0]
                            else:
                                dest_placeholder_turn = anchor_children[0]
                    except Exception as exc:
                        dest_placeholder_turn = None
                        if replay_mode and replay_debug:
                            print(f"{Colors.TOOL_WARNING}Replay: failed to locate try-out placeholder under {mapped_parent.gen_id_or_parent}: {exc}{Colors.RESET}")

                if not dest_placeholder_turn and mapped_parent:
                    scope = _require_live_chat_scope(dest_cursor_for_turn)
                    ctx = dest_cursor_for_turn.context if dest_cursor_for_turn else None
                    anchor = None
                    try:
                        anchor = scope.get_try_out_anchor(anchor_name)
                    except Exception as exc:
                        anchor = None
                        if replay_mode and replay_debug:
                            print(f"{Colors.TOOL_WARNING}Replay: failed to read try-out anchor {anchor_name}: {exc}{Colors.RESET}")
                    if not anchor and ctx and dest_cursor_for_turn and dest_cursor_for_turn.current_turn:
                        try:
                            anchor = scope.start_try_out_anchor(
                                anchor_name,
                                dest_cursor_for_turn.current_turn,
                                kind=anchor_kind,
                                origin_cursor=dest_cursor_for_turn,
                            )
                        except Exception as exc:
                            anchor = None
                            if replay_mode:
                                print(f"{Colors.TOOL_WARNING}Replay: failed to start try-out anchor {anchor_name}: {exc}{Colors.RESET}")

                    try:
                        temp_cursor = dest_cursor_for_turn.clone_at(mapped_parent) if dest_cursor_for_turn else None
                        if temp_cursor:
                            main_cursor, try_cursor = temp_cursor.add_try_out(
                                anchor=anchor,
                                anchor_turn=mapped_parent,
                            )
                            candidate_cursor = try_cursor if source_is_try else main_cursor
                            if candidate_cursor and candidate_cursor.current_turn:
                                dest_placeholder_turn = candidate_cursor.current_turn
                                dest_placeholder_cursor = candidate_cursor
                    except Exception as exc:
                        if replay_mode:
                            print(f"{Colors.TOOL_WARNING}Replay: failed to recreate try-out placeholder: {exc}{Colors.RESET}")

                if dest_placeholder_turn and not dest_placeholder_cursor:
                    try:
                        if dest_cursor_for_turn and dest_cursor_for_turn.context:
                            scope = _require_live_chat_scope(dest_cursor_for_turn)
                            dest_placeholder_cursor = (
                                scope.register_cursor_for_turn(dest_placeholder_turn, make_active=False)
                            )
                        if not dest_placeholder_cursor and dest_cursor_for_turn:
                            dest_placeholder_cursor = dest_cursor_for_turn.clone_at(dest_placeholder_turn)
                    except Exception as exc:
                        if replay_mode:
                            print(f"{Colors.TOOL_WARNING}Replay: failed to bind cursor for try-out placeholder: {exc}{Colors.RESET}")
                        dest_placeholder_cursor = dest_cursor_for_turn

                if dest_placeholder_turn:
                    _record_mapping(source_turn, dest_placeholder_cursor or dest_cursor_for_turn)
                    try:
                        _validate_mapping(source_turn, dest_placeholder_turn)
                    except Exception as exc:
                        _mark_failed_branch(branch_root, str(exc))
                        if replay_mode:
                            print(f"{Colors.ERROR}Replay error: {exc}{Colors.RESET}")
                        return dest_cursor
                    _update_branch(branch_root, dest_placeholder_cursor or dest_branch_cursor)
                    handled_turns.add(source_turn)
                    continue

        if getattr(source_turn, "turn_type", None) == Turn.BATCH:
            parent_turn = getattr(source_turn, "parent", None)
            sibling_candidates = [source_turn]
            if parent_turn:
                sibling_candidates = [
                    t for t in getattr(parent_turn, "turns", []) or []
                    if getattr(t, "turn_type", None) == Turn.BATCH
                ]
            sibling_batches = [
                t for t in sibling_candidates
                if t in scope_turns and not getattr(t, "is_auto", False) and t not in handled_turns
            ]
            if not sibling_batches:
                sibling_batches = [source_turn]

            prepared_batches: List[Dict[str, Any]] = []
            for batch_turn in sibling_batches:
                dest_batch_cursor, kept_children, active_forks, concurrency_level = await _prepare_replay_batch_turn(
                    source_turn=batch_turn,
                    dest_parent_cursor=dest_cursor_for_turn,
                    driver_cursor=driver_cursor,
                    pt_session=pt_session,
                    source_to_dest_cursor_map=source_to_dest_cursor_map,
                    replay_mode=replay_mode,
                    replay_debug=replay_debug,
                )
                prepared_batches.append({
                    "source_turn": batch_turn,
                    "dest_batch_cursor": dest_batch_cursor,
                    "kept_children": kept_children,
                    "active_forks": active_forks,
                    "concurrency_level": concurrency_level,
                })

            def _replay_batch_concurrency(command_text: str, batch_count: int) -> int:
                text = (command_text or "").strip().lower()
                if not text.startswith("/g"):
                    return 1
                rest = text[2:].lstrip()
                if rest.startswith("generate"):
                    rest = rest[len("generate"):].lstrip()
                if rest.startswith("bc"):
                    return max(1, batch_count)
                return 1

            all_active_forks: List[Dict[str, Any]] = []
            batch_node_count = len(prepared_batches)
            global_offset = 0
            for batch in prepared_batches:
                active_forks = batch.get("active_forks") or []
                if not active_forks:
                    continue
                fork_obj = active_forks[0].get("fork") if active_forks else None
                if fork_obj and getattr(fork_obj, "cursors", None):
                    fork_obj.prompt_indices = list(range(global_offset, global_offset + len(fork_obj.cursors)))
                for entry in active_forks:
                    cursor_idx = entry.get("cursor_idx")
                    local_idx = cursor_idx if cursor_idx is not None else entry.get("original_index", 0)
                    entry["original_index"] = global_offset + (local_idx or 0)
                    all_active_forks.append(entry)
                if fork_obj and getattr(fork_obj, "cursors", None):
                    global_offset += len(fork_obj.cursors)
                else:
                    global_offset += len(active_forks)

            if all_active_forks:
                first_command = ""
                if prepared_batches:
                    source_turn = prepared_batches[0].get("source_turn")
                    if source_turn:
                        first_command = (getattr(source_turn, "data", {}) or {}).get("command", "")
                concurrency_level = _replay_batch_concurrency(first_command, batch_node_count)
                await _run_batch_rounds_for_replay(
                    active_forks=all_active_forks,
                    parent_cursor=dest_cursor_for_turn,
                    concurrency_level=concurrency_level,
                    tools_view=dest_cursor_for_turn.get_tools_view(),
                    llm_name=_effective_replay_llm_name(dest_cursor_for_turn),
                    source_to_dest_cursor_map=source_to_dest_cursor_map,
                    replay_debug=replay_debug,
                )

            for batch in prepared_batches:
                batch_turn = batch["source_turn"]
                dest_batch_cursor = batch["dest_batch_cursor"]
                kept_children = batch["kept_children"]
                batch_branch_root = batch_turn.branch_root()
                dest_batch_turn = getattr(dest_batch_cursor, "current_turn", None)
                _record_mapping(batch_turn, dest_batch_cursor)
                try:
                    _validate_mapping(batch_turn, dest_batch_turn)
                except Exception as exc:
                    _mark_failed_branch(batch_branch_root, str(exc))
                    if replay_mode:
                        print(f"{Colors.ERROR}Replay error: {exc}{Colors.RESET}")
                    return dest_cursor
                _update_branch(batch_branch_root, dest_batch_cursor)
                child_cursor_map = _build_replay_child_cursor_map(
                    kept_children=kept_children,
                    dest_batch_cursor=dest_batch_cursor,
                    dest_parent_cursor=dest_cursor_for_turn,
                    source_to_dest_cursor_map=source_to_dest_cursor_map,
                )
                for child_turn, child_cursor in (child_cursor_map or {}).items():
                    child_root = child_turn.branch_root()
                    dest_child_turn = getattr(child_cursor, "current_turn", None)
                    _record_mapping(child_turn, child_cursor)
                    try:
                        _validate_mapping(child_turn, dest_child_turn)
                    except Exception as exc:
                        _mark_failed_branch(child_root, str(exc))
                        if replay_mode:
                            print(f"{Colors.ERROR}Replay error: {exc}{Colors.RESET}")
                        return dest_cursor
                    _update_branch(child_root, child_cursor)
                    handled_turns.add(child_turn)
                handled_turns.add(batch_turn)
            continue

        if getattr(source_turn, "turn_type", None) == Turn.FORK:
            try:
                fork_cursor = await _mirror_turn_state(
                    source_cursor=driver_cursor.clone_at(source_turn),
                    dest_cursor=dest_cursor_for_turn,
                    replay_mode=replay_mode,
                    replay_debug=replay_debug,
                )
                dest_mapped = getattr(fork_cursor, "current_turn", None)
                _record_mapping(source_turn, fork_cursor)
                try:
                    _validate_mapping(source_turn, dest_mapped)
                except Exception as exc:
                    _mark_failed_branch(branch_root, str(exc))
                    if replay_mode:
                        print(f"{Colors.ERROR}Replay error: {exc}{Colors.RESET}")
                    return dest_cursor
                _update_branch(branch_root, fork_cursor or dest_branch_cursor)
                continue
            except Exception as exc:
                _mark_failed_branch(branch_root, f"replay command failed: {exc}")
                if replay_mode:
                    print(f"{Colors.TOOL_WARNING}Replay: aborting branch {branch_root.gen_id_or_parent}: {exc}{Colors.RESET}")
                continue

        try:
            new_dest_cursor, executed, _, mapped_turn = await _replay_turn(
                driver_cursor=driver_cursor.clone_at(source_turn),
                dest_cursor=dest_cursor_for_turn,
                pt_session=pt_session,
                replay_mode=replay_mode,
                replay_debug=replay_debug,
            )
            if executed:
                dest_mapped = mapped_turn or getattr(new_dest_cursor, "current_turn", None)
                _record_mapping(source_turn, new_dest_cursor)
                try:
                    _validate_mapping(source_turn, dest_mapped)
                except Exception as exc:
                    _mark_failed_branch(branch_root, str(exc))
                    if replay_mode:
                        print(f"{Colors.ERROR}Replay error: {exc}{Colors.RESET}")
                    return dest_cursor
            _update_branch(branch_root, new_dest_cursor or dest_branch_cursor)
        except Exception as exc:
            _mark_failed_branch(branch_root, f"replay command failed: {exc}")
            if replay_mode:
                print(f"{Colors.TOOL_WARNING}Replay: aborting branch {branch_root.gen_id_or_parent}: {exc}{Colors.RESET}")
            continue

    if replay_mode:
        print(f"{Colors.SYSTEM}Replay 'all-down' traversal completed.{Colors.RESET}")

    # Preserve the source active leaf so branch-scoped state (system/tools/adapters)
    # remains effective on the replayed session.
    source_active = driver_cursor.current_turn
    mapped_active = _find_mapped_ancestor(source_active)
    if mapped_active:
        try:
            scope = _scope_for_cursor(dest_cursor)
            active_cursor = (
                scope.register_cursor_for_turn(mapped_active, make_active=False)
                if scope else dest_cursor.context.register_cursor_for_turn(mapped_active, make_active=False)
            )
            _set_active_cursor(active_cursor or dest_cursor.clone_at(mapped_active))
        except Exception as exc:
            if replay_mode:
                print(f"{Colors.ERROR}Replay error: failed to set active cursor to mapped source leaf: {exc}{Colors.RESET}")
            return dest_cursor

    try:
        conv_index = dest_cursor.session.conversations.index(dest_cursor.chat_session)
        new_leaf_turn = dest_cursor.session.get_main_branch_leaf(conv_index)
        if new_leaf_turn:
            scope = _scope_for_cursor(dest_cursor)
            new_cursor = (
                scope.register_cursor_for_turn(new_leaf_turn, make_active=False)
                if scope else dest_cursor.context.register_cursor_for_turn(new_leaf_turn, make_active=False)
            )
            if new_cursor:
                return new_cursor
    except Exception as exc:
        if replay_mode:
            print(f"{Colors.ERROR}Replay error: failed to resolve main branch leaf: {exc}{Colors.RESET}")
        return dest_cursor

    return dest_cursor


async def _run_batch_rounds_for_replay(
    active_forks: List[Dict[str, Any]],
    *,
    parent_cursor: ChatCursor,
    concurrency_level: int,
    tools_view: Optional[ToolsView],
    llm_name: str,
    source_to_dest_cursor_map: Dict[Turn, Dict[str, Any]],
    replay_debug: bool = False,
) -> None:
    """Drive batch rounds for replay using the live batch post-processor for parity."""
    batch_round = 0
    console_lock = asyncio.Lock()

    while active_forks:
        if REPLAY_CANCELLATION_EVENT.is_set():
            print(f"{Colors.TOOL_WARNING}Replay canceled by user.{Colors.RESET}")
            break
        batch_round += 1
        print(f"\n{Colors.SYSTEM}Replay Batch Round {batch_round}: {len(active_forks)} fork(s){Colors.RESET}")

        tasks = []
        requests_for_batch: List[Dict[str, Any]] = []
        batch_groups = batch_list(active_forks, concurrency_level)

        for i, batch_group in enumerate(batch_groups): # type: ignore
            batch_cursors: List[ChatCursor] = []
            for fork in batch_group:
                entry_cursor = _fork_entry_cursor(fork)
                if not entry_cursor or not entry_cursor.current_turn:
                    if replay_debug:
                        print(f"{Colors.TOOL_WARNING}DEBUG: missing cursor for replay batch fork; skipping entry.{Colors.RESET}")
                    continue
                if not entry_cursor.context:
                    raise RuntimeError("Replay batch: entry cursor missing context.")
                if entry_cursor:
                    fork_obj = fork.get("fork")
                    cursor_idx = fork.get("cursor_idx")
                    batch_cursors.append(entry_cursor)
            if not batch_cursors:
                continue
            try:
                inference_payload, _, _ = batch_cursors[0].build_inference_request(
                    batch=batch_cursors,
                    request_id_prefix=f"replay_batch_r{batch_round}",
                    manual_continue=False,
                    include_tools=True,
                )
                adapters_for_chunk: List[str] = []
                for fork in batch_group:
                    entry_cursor = _fork_entry_cursor(fork)
                    adapters_for_chunk.append(entry_cursor.get_data().get("adapter_override_name") if entry_cursor else None)
                if any(adapters_for_chunk):
                    inference_payload["override_adapters"] = adapters_for_chunk
            except Exception as exc:
                print(f"{Colors.ERROR}Replay batch: failed to build request for batch {i+1}: {exc}{Colors.RESET}")
                continue

            requests_for_batch.append(inference_payload)
            task = asyncio.create_task(_run_batch_request(
                request_payload=inference_payload,
                batch_forks=batch_group,
                batch_index=i,
                total_batches_in_round=len(batch_groups),
                batch_round=batch_round,
                llm_name=llm_name,
                parent_cursor=parent_cursor,
                console_lock=console_lock,
                total_prompts_for_batch=len(batch_cursors),
            ))
            tasks.append(task)

        start_time = time.monotonic()
        try:
            results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            print(f"{Colors.ERROR}Replay batch canceled by user (Ctrl+C).{Colors.RESET}")
            raise
        wall_clock_time = time.monotonic() - start_time

        valid_results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for idx, res in enumerate(results_from_gather):
            if isinstance(res, dict) and not isinstance(res, Exception) and idx < len(requests_for_batch):
                valid_results.append((requests_for_batch[idx], res))
            elif isinstance(res, Exception):
                print(f"{Colors.ERROR}Replay batch round {batch_round} batch {idx+1} failed: {res}{Colors.RESET}")

        sorted_results = sorted(valid_results, key=lambda r: r[1].get("batch_index", -1))

        next_active_forks, _ = await _process_batch_results(
            sorted_results=sorted_results,
            active_forks=active_forks,
            tools_view=tools_view,
            batch_round=batch_round,
            is_concurrent=concurrency_level > 1,
            wall_clock_time=wall_clock_time,
            cursor=parent_cursor,
            source_to_dest_cursor_map=source_to_dest_cursor_map,
            allow_auto_retry=True,
            replay_mode=True,
        )

        active_forks = next_active_forks


async def _run_replay_auto_rounds(
    dest_cursor: ChatCursor,
    pt_session: "PromptSession",
    *,
    adapter_override: Optional[List[str]] = None,
    request_id_prefix: str = "replay",
    include_tools: bool = True,
) -> ChatCursor:
    """
    Replay driver that mirrors the live chat loop's auto-iteration handling for a turn.
    """
    _require_live_chat_scope(dest_cursor)
    active_cursor = _ensure_registered_cursor(dest_cursor) or dest_cursor
    _set_active_cursor(active_cursor)

    def _needs_inference(cursor: Optional[ChatCursor]) -> bool:
        """Return True when the cursor head is waiting for an assistant response."""
        if not cursor or not cursor.current_turn:
            return False
        turn = cursor.current_turn
        if getattr(turn, "HasResponse", False):
            return False
        turn_payload = getattr(turn, "data", {}) or {}
        if turn_payload.get("user") or turn_payload.get("tool_results"):
            return True
        return bool(getattr(turn, "do_continue", False))

    while True:
        ctx = active_cursor.context if active_cursor else None
        scope = _require_live_chat_scope(active_cursor) if active_cursor else _require_live_chat_scope()
        if not _needs_inference(active_cursor):
            if ctx:
                try:
                    while True:
                        if not scope.consume_auto_iteration():
                            break
                except Exception as exc:
                    print(f"{Colors.TOOL_WARNING}Auto-rounds: failed to consume auto-iteration: {exc}{Colors.RESET}")
            break
        try:
            inference_payload, active_adapters_for_turn, tools_view_for_request = active_cursor.build_inference_request(
                request_id_prefix=request_id_prefix,
                manual_continue=bool(getattr(active_cursor.current_turn, "do_continue", False)),
                include_tools=include_tools,
            )
            if adapter_override:
                inference_payload["override_adapters"] = adapter_override
        except Exception as exc:
            print(f"{Colors.ERROR}Failed to build inference request: {exc}{Colors.RESET}")
            return active_cursor

        should_continue_loop = await _execute_inference_round(
            cursor=active_cursor,
            pt_session=pt_session,
            inference_payload=inference_payload,
            active_adapters_for_turn=active_adapters_for_turn,
            tools_view_for_request=tools_view_for_request,
            is_override_prompt=bool(adapter_override),
            override_adapters_for_response=adapter_override,
            is_manual_continue=bool(getattr(active_cursor.current_turn, "do_continue", False)),
            llm_name_override=_effective_replay_llm_name(active_cursor),
        )

        current_active = _require_current_cursor()
        active_cursor = _ensure_registered_cursor(current_active) or current_active

        ctx = active_cursor.context if active_cursor else None
        if should_continue_loop:
            if ctx:
                try:
                    scope.consume_auto_iteration()
                except Exception as exc:
                    print(f"{Colors.TOOL_WARNING}Auto-rounds: failed to consume auto-iteration after round: {exc}{Colors.RESET}")
            continue

        auto_iteration_requested = bool(scope.consume_auto_iteration()) if ctx else False
        if not auto_iteration_requested:
            break

    final_cursor = _normalize_cursor_after_auto_iters(active_cursor, prefer_session_main_leaf=True)
    if final_cursor:
        return final_cursor
    try:
        if active_cursor and active_cursor.current_turn:
            active_cursor.current_turn.is_archived = True
            print(f"{Colors.ERROR}Auto-rounds cleanup failed; archiving turn {active_cursor.current_turn.gen_id_or_parent}.{Colors.RESET}")
    except Exception as exc:
        print(f"{Colors.TOOL_WARNING}Auto-rounds cleanup: failed to archive turn: {exc}{Colors.RESET}")
    ctx = active_cursor.context if active_cursor else None
    if ctx:
        try:
            return scope.active_cursor()
        except Exception as exc:
            print(f"{Colors.TOOL_WARNING}Auto-rounds cleanup: failed to resolve active cursor: {exc}{Colors.RESET}")
    raise RuntimeError("Auto-rounds cleanup failed: no active cursor to continue.")


async def _run_turn_auto_rounds(
    dest_cursor: ChatCursor,
    pt_session: "PromptSession",
    *,
    adapter_override: Optional[List[str]] = None,
    request_id_prefix: str = "chat",
    include_tools: bool = True,
) -> ChatCursor:
    """Unified turn executor shared by live chat (new mode) and replay."""
    return await _run_replay_auto_rounds(
        dest_cursor=dest_cursor,
        pt_session=pt_session,
        adapter_override=adapter_override,
        request_id_prefix=request_id_prefix,
        include_tools=include_tools,
    )


async def _replay_turn(
    driver_cursor: ChatCursor,
    dest_cursor: ChatCursor,
    pt_session: "PromptSession",
    *,
    replay_mode: bool = True,
    replay_debug: bool = False,
) -> Tuple[ChatCursor, bool, bool, Optional[Turn]]:
    _require_live_chat_scope(dest_cursor)
    def _ensure_user_target(cursor: ChatCursor) -> ChatCursor:
        """Make sure we are parked on a non-structural leaf before adding a user turn."""
        if not cursor.current_turn:
            return cursor
        if getattr(cursor.current_turn, "IsStructural", False):
            try:
                cursor = cursor.descend_to_leaf()
            except Exception as exc:
                raise RuntimeError(
                    f"Replay failed to descend from structural turn {cursor.display_id()}: {exc}"
                ) from exc
        if cursor.current_turn and getattr(cursor.current_turn, "IsStructural", False):
            try:
                non_structural_child = next(
                    (
                        t for t in getattr(cursor.current_turn, "turns", []) or []
                        if not getattr(t, "IsStructural", False)
                    ),
                    None,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Replay failed to enumerate children for {cursor.display_id()}: {exc}"
                ) from exc
            if non_structural_child:
                try:
                    if cursor.context:
                        rebound = cursor.context.find_cursor_for_turn(non_structural_child)
                        cursor = rebound or cursor.clone_at(non_structural_child)
                    else:
                        cursor = cursor.clone_at(non_structural_child)
                except Exception as exc:
                    raise RuntimeError(
                        f"Replay failed to rebind cursor to child {non_structural_child.gen_id_or_parent}: {exc}"
                    ) from exc
        return cursor

    # Ensure destination cursor is context-bound; adopt if missing
    context = _active_chat_context()
    if not dest_cursor.context and context and dest_cursor.current_turn:
        try:
            scope = _scope_for_cursor(dest_cursor)
            rebound = (
                scope.register_cursor_for_turn(dest_cursor.current_turn, make_active=False)
                if scope else context.register_cursor_for_turn(dest_cursor.current_turn, make_active=False)
            )
            if rebound:
                dest_cursor = rebound
        except Exception as exc:
            if replay_mode:
                print(f"{Colors.TOOL_WARNING}Replay: failed to bind destination cursor to context: {exc}{Colors.RESET}")
    active_turn = driver_cursor.current_turn
    source_turn_cursor = driver_cursor.clone_at(active_turn)

    turn_data = source_turn_cursor.current_turn.data or {}
    has_user = bool(turn_data.get("user"))
    has_tool_results = bool(turn_data.get("tool_results"))
    has_other_roles = any(role for role in turn_data.keys() if role not in ["assistant", "user", "tool_results"])
    has_commands = bool(source_turn_cursor.current_turn.cmd)
    
    handled_descendants = False
    mapped_turn: Optional[Turn] = None

    if not (has_user or source_turn_cursor.current_turn.do_continue or has_tool_results or has_other_roles or has_commands):
        if replay_mode and replay_debug:
            print(f"{Colors.TOOL_WARNING}Replay skipped turn {active_turn.gen_id_or_parent}: empty turn.{Colors.RESET}")
        return dest_cursor, True, handled_descendants, mapped_turn

    if replay_mode and replay_debug:
        print(f"{Colors.SYSTEM}→ Replaying turn {active_turn.gen_id_or_parent} into {dest_cursor.display_id()}...{Colors.RESET}")

    if not dest_cursor.context:
        if replay_mode: print(f"{Colors.ERROR}Replay aborted: destination cursor context is unavailable.{Colors.RESET}")
        return dest_cursor, False, handled_descendants, mapped_turn

    is_archived_turn = source_turn_cursor.is_archived()
    if replay_mode and replay_debug:
        try:
            turn = source_turn_cursor.current_turn
            data_keys = sorted(list((turn.data or {}).keys())) if turn and getattr(turn, "data", None) else []
            print(
                f"{Colors.SYSTEM}DEBUG: replay src turn {active_turn.gen_id_or_parent} "
                f"type={getattr(turn, 'turn_type', None)} archived={bool(getattr(turn, 'is_archived', False))} "
                f"canceled={bool(getattr(turn, 'was_canceled', False))} "
                f"truncated={bool(getattr(turn, 'was_truncated', False))} "
                f"data_keys={data_keys}{Colors.RESET}"
            )
        except Exception as exc:
            print(f"{Colors.TOOL_WARNING}Replay: failed to render turn debug info: {exc}{Colors.RESET}")
    if is_archived_turn:
        if replay_mode:
            print(f"{Colors.TOOL_WARNING}Replay: applying commands from archived turn {active_turn.gen_id_or_parent}, but skipping LLM round.{Colors.RESET}")
        dest_cursor = await _mirror_turn_state(source_turn_cursor, dest_cursor, replay_mode=replay_mode, replay_debug=replay_debug)
        mapped_turn = dest_cursor.current_turn
        try:
            return dest_cursor.descend_to_leaf(), True, handled_descendants, mapped_turn
        except Exception as exc:
            if replay_mode:
                print(f"{Colors.TOOL_WARNING}Replay: failed to descend from archived turn {active_turn.gen_id_or_parent}: {exc}{Colors.RESET}")
            return dest_cursor, True, handled_descendants, mapped_turn

    adapter_override = source_turn_cursor.current_turn.data.get("adapter_override_name")

    if has_user:
        src_user = turn_data.get("user") or {}
        user_content = src_user.get("content", "") if isinstance(src_user, dict) else ""
        dest_cursor = _ensure_user_target(dest_cursor)
        if dest_cursor.current_turn and getattr(dest_cursor.current_turn, "IsStructural", False):
            try:
                dest_cursor.add_user(user_content)
                if replay_mode:
                    print(f"{Colors.TOOL_WARNING}Replay: created chat child under structural turn {dest_cursor.display_id()} to place user.{Colors.RESET}")
            except Exception as exc:
                if replay_mode:
                    print(f"{Colors.TOOL_WARNING}Replay: cannot place user under structural turn {dest_cursor.display_id()}: {exc}{Colors.RESET}")
                return dest_cursor, False, handled_descendants, mapped_turn
        
        dest_cursor.add_user(user_content)
        mapped_turn = dest_cursor.current_turn
        if replay_mode:
            try:
                turn_idx = dest_cursor.user_turns_count()
                print(f"{Colors.YOU_HEADER}Turn {turn_idx}->You:{Colors.RESET} {Colors.YOU_CONTENT}{user_content}{Colors.RESET}")
            except Exception as exc:
                if replay_debug:
                    print(f"{Colors.TOOL_WARNING}Replay: failed to print user turn label: {exc}{Colors.RESET}")
        
    elif has_tool_results:
        raw_results = turn_data.get("tool_results") or []
        results_blocks: List[ToolCallBlock] = []
        if raw_results and isinstance(raw_results[0], ToolCallBlock):
            results_blocks = list(raw_results)
        elif raw_results and isinstance(raw_results[0], ToolCall):
            results_blocks = [ToolCallBlock(raw_block="", calls=list(raw_results))]
        # Ensure we are not parked on a structural turn (e.g., batch hub) before adding tool results
        if dest_cursor.current_turn and getattr(dest_cursor.current_turn, "turn_type", None) == Turn.BATCH:
            try:
                dest_cursor = dest_cursor.descend_to_leaf()
            except Exception as exc:
                if replay_mode:
                    print(f"{Colors.TOOL_WARNING}Replay: failed to descend from batch hub: {exc}{Colors.RESET}")
            if dest_cursor.current_turn and getattr(dest_cursor.current_turn, "turn_type", None) == Turn.BATCH:
                next_turn = None
                try:
                    next_turn = next(
                        (t for t in getattr(dest_cursor.current_turn, "turns", []) or [] if getattr(t, "turn_type", None) != Turn.BATCH),
                        None,
                    )
                except Exception as exc:
                    next_turn = None
                    if replay_mode:
                        print(f"{Colors.TOOL_WARNING}Replay: failed to enumerate batch hub children: {exc}{Colors.RESET}")
                if next_turn:
                    try:
                        if dest_cursor.context:
                            scope = _scope_for_cursor(dest_cursor)
                            dest_cursor = (
                                scope.register_cursor_for_turn(next_turn, make_active=False)
                                if scope else dest_cursor.context.register_cursor_for_turn(next_turn, make_active=False)
                            ) or dest_cursor.clone_at(next_turn)
                        else:
                            dest_cursor = dest_cursor.clone_at(next_turn)
                    except Exception as exc:
                        if replay_mode:
                            print(f"{Colors.TOOL_WARNING}Replay: failed to rebind from batch hub to child {next_turn.gen_id_or_parent}: {exc}{Colors.RESET}")
        if dest_cursor.current_turn and getattr(dest_cursor.current_turn, "turn_type", None) == Turn.BATCH:
            print(f"{Colors.TOOL_WARNING}Replay: tool results skipped; destination cursor stuck on batch hub {dest_cursor.display_id()}.{Colors.RESET}")
            return dest_cursor, False, handled_descendants, mapped_turn
        if _tool_blocks_have_abort(results_blocks):
            print(f"{Colors.TOOL_WARNING}Replay: tool round aborted by action; skipping results.{Colors.RESET}")
            return dest_cursor, False, handled_descendants, mapped_turn
        dest_cursor.add_tool_results(results_blocks)
        dest_cursor.set_auto(True)
        if replay_mode:
            print(f"\n{Colors.TOOL}Replaying Tool Results...{Colors.RESET}")
            for block in results_blocks:
                if getattr(block, "error_block", None):
                    print(f"  - {Colors.ERROR}{block.error_block}{Colors.RESET}")
                for tool_call in getattr(block, "calls", []) or []:
                    if getattr(tool_call, 'result', None) is not None:
                        print(f"  - {Colors.DIM}{tool_call.result}{Colors.RESET}")
                    elif getattr(tool_call, 'error', None) is not None:
                        print(f"  - {Colors.ERROR}{tool_call.error}{Colors.RESET}")
    elif source_turn_cursor.current_turn.do_continue:
        if dest_cursor.current_turn and not getattr(dest_cursor.current_turn, "was_truncated", False):
            if replay_mode:
                print(f"{Colors.TOOL_WARNING}Replay: skipping do_continue on non-truncated turn {dest_cursor.display_id()}.{Colors.RESET}")
            return dest_cursor, False, handled_descendants, mapped_turn
        dest_cursor.add_continuation_turn()

    dest_cursor = await _mirror_turn_state(source_turn_cursor, dest_cursor, replay_mode=replay_mode, replay_debug=replay_debug)
    if not mapped_turn:
        mapped_turn = dest_cursor.current_turn

    dest_label = None
    try:
        if dest_cursor.current_turn and getattr(dest_cursor.current_turn, "gen_id_or_parent", None):
            dest_label = dest_cursor.current_turn.gen_id_or_parent
    except Exception as exc:
        dest_label = None
        if replay_debug:
            print(f"{Colors.TOOL_WARNING}Replay: failed to read destination turn label: {exc}{Colors.RESET}")
    suffix = f" (dest {dest_label})" if dest_label else ""
    if replay_mode and replay_debug:
        print(f"{Colors.SYSTEM}→ Replaying turn {active_turn.gen_id_or_parent}{suffix}...{Colors.RESET}")

    if has_user or has_tool_results or source_turn_cursor.current_turn.do_continue:
        dest_cursor = await _run_replay_auto_rounds(
            dest_cursor=dest_cursor,
            pt_session=pt_session,
            adapter_override=[adapter_override] if adapter_override else None,
            request_id_prefix="replay",
        )
    
    return dest_cursor.descend_to_leaf(), True, handled_descendants, mapped_turn

async def _replay_session_branch(
    driver_cursor: ChatCursor,
    dest_cursor: ChatCursor,
    pt_session: "PromptSession",
    *,
    replay_mode: bool = True,
    replay_debug: bool = False,
) -> ChatCursor:
    """Replays a branch of a session, starting from driver_cursor, into dest_cursor."""
    if not driver_cursor or not driver_cursor.current_turn:
        print(f"{Colors.ERROR}Replay source is invalid.{Colors.RESET}")
        return dest_cursor

    candidate_turns = driver_cursor.active_path_for_llm()

    if not candidate_turns:
        print(f"{Colors.ERROR}Replay source does not have any turns to process.{Colors.RESET}")
        return dest_cursor

    candidate_turns = [t for t in candidate_turns if not getattr(t, 'is_auto', False)]

    if not candidate_turns:
        print(f"{Colors.TOOL_WARNING}Replay source has no user turns beneath the selected cursor.{Colors.RESET}")
        return dest_cursor

    if replay_mode:
        engine_name = _effective_engine_name_for_session(dest_cursor.chat_session)
        print(
            f"{Colors.SYSTEM}Replaying conversation starting at turn "
            f"{driver_cursor.current_turn.gen_id_or_parent} (engine: {engine_name}).{Colors.RESET}"
        )
    
    executed = 0
    for active_turn in candidate_turns:
        if REPLAY_CANCELLATION_EVENT.is_set():
            if replay_mode:
                print(f"{Colors.TOOL_WARNING}Replay canceled by user.{Colors.RESET}")
            break
        dest_cursor, turn_executed, _, _ = await _replay_turn(
            driver_cursor=driver_cursor.clone_at(active_turn),
            dest_cursor=dest_cursor,
            pt_session=pt_session,
            replay_mode=replay_mode,
            replay_debug=replay_debug,
        )
        if turn_executed:
            executed += 1
    
    if replay_mode:
        print(f"{Colors.SYSTEM}Replay completed. {executed} turns processed.{Colors.RESET}")

    return dest_cursor


async def _process_turn_and_descendants(cursor: ChatCursor, pt_session: "PromptSession", *, adapter_override: Optional[List[str]] = None) -> ChatCursor:
    """
    Takes a cursor pointing to a turn with fresh input and handles the entire
    resulting exchange (including multi-round tool calls) until the model is done.
    Returns the final cursor position.
    """
    return await _run_turn_auto_rounds(
        dest_cursor=cursor,
        pt_session=pt_session,
        adapter_override=adapter_override,
        request_id_prefix="chat",
        include_tools=True,
    )


async def _handle_live_prompt(user_input: str, cursor: ChatCursor, pt_session: "PromptSession") -> ChatCursor:
    """Adds the user's message and calls the processing engine."""
    cursor.add_user(user_input)
    final_cursor = await _process_turn_and_descendants(cursor, pt_session, adapter_override=None)
    return final_cursor


async def chat_loop(
    init_resp: Dict[str, Any],
    *,
    startup_session_path: Optional[str] = None,
    startup_replay_specs: Optional[List[str]] = None,
    startup_save_spec: Optional[str] = None,
    startup_quit: bool = False,
):
    global current_config, session_control, toolbox, ENGINE_PARSER_PROFILE, conversation_template

    if not current_config or not session_control:
        print("Critical error: Config/session manager not init.")
        return

    # --- NEW: Populate initial ChatSession with engine info from startup ---
    init_data = init_resp.get("data", {}) or {}
    engine_config_for_session = init_data.get("global_config", {}) or {}
    engine_config_summary = {k: v for k, v in engine_config_for_session.items() if not isinstance(v, dict)}
    engine_other_config = engine_config_for_session.get("other_config", {}) or {}
    engine_warnings_for_session = init_data.get("warnings", [])
    initial_system_message = current_config.get("default_system_message", "")
    no_tools_parse_flag = bool(engine_config_for_session.get("no_tools_parse", False))
    auto_retry_flag = _auto_retry_enabled()
    suppress_full_response_flag = bool(engine_config_for_session.get("suppress_full_response", _current_suppress_full_response()))
    seed_values = _current_param_snapshot()
    seed_values["no_tools_parse"] = no_tools_parse_flag
    seed_values["auto_retry_truncated"] = auto_retry_flag
    seed_values["suppress_full_response"] = suppress_full_response_flag

    # Capture current inference params for the new session
    inference_defaults_for_new_session = InferenceParams(
        stream=seed_values.get("stream", True),
        return_prompt=seed_values.get("return_prompt"),
        cache=seed_values.get("cache"),
        generation_config=seed_values.get("generation_config_template") or {},
        no_tools_parse=no_tools_parse_flag,
        suppress_full_response=suppress_full_response_flag,
    )
    initial_params = {
        "stream": seed_values.get("stream"),
        "cache_override": seed_values.get("cache"),
        "return_prompt_mode": seed_values.get("return_prompt"),
        "generation_config_template": seed_values.get("generation_config_template"),
        "max_new_tokens_override": seed_values.get("max_new_tokens"),
        "system_message": initial_system_message,
        "auto_retry_truncated": auto_retry_flag,
        "suppress_full_response": suppress_full_response_flag,
        "no_tools_parse": no_tools_parse_flag,
        "auto_tool_retry_limit": seed_values.get("auto_tool_retry_limit"),
        "auto_continue_retry_limit": seed_values.get("auto_continue_retry_limit"),
    }

    parser_profile: Optional[ParserProfile] = None
    if profile_dict := engine_other_config.get("tool_parser_profile"):
        parser_profile = ParserProfile(**profile_dict)
        ENGINE_PARSER_PROFILE = parser_profile

    base_model_name = engine_config_for_session.get("base_model_name")
    if not base_model_name:
        base_model_path = engine_other_config.get("base_model_name_or_path")
        if base_model_path:
            base_model_name = Path(base_model_path).name
    if base_model_name:
        initial_params["engine_base_model_name"] = base_model_name

    initial_session = EngineSession() # Start with a default session
    initial_chat_session = initial_session.add_conversation(
        engine_config=engine_config_summary,
        engine_warnings=engine_warnings_for_session,
        parser_profile=parser_profile,
        initial_params=initial_params, # type: ignore
        inference_defaults=inference_defaults_for_new_session,
    )
    _seed_chat_session_flags(initial_chat_session, values=seed_values)
    conversation_template = initial_session.conversation_from_template(initial_chat_session)
    cursor = _bootstrap_cursor_for_session(initial_session, conversation=initial_chat_session)
    _set_active_cursor(cursor)
 
    print(f"MP13 Playground Chat. Type /help for commands.")
    print_current_session_info(cursor)
    
    # Style for the prompt_toolkit prompt itself
    pt_style = Style.from_dict({
        '': 'bold lightgreen', # Default input text color
        'prompt.you': 'lightgreen',
        'prompt.override': 'yellow',
        'prompt.role': 'gray', # Use a dim color instead of the 'dim' attribute
    })
    key_bindings = KeyBindings()

    @key_bindings.add("enter")
    def _(event):
        """When enter is pressed, submit the input."""
        event.current_buffer.validate_and_handle()

    @key_bindings.add('escape', 'enter')
    def _(event):
        """When Alt+enter is pressed, insert a newline."""
        event.current_buffer.insert_text("\n")

    # --- FIX: Custom Ctrl+C handler to integrate with asyncio signal handling ---
    # This prevents prompt_toolkit from raising KeyboardInterrupt and instead
    # calls our async cancellation logic directly, unifying signal handling.
    @key_bindings.add("c-c", eager=True)
    def _(event):
        """Custom handler for Ctrl+C."""
        REPLAY_CANCELLATION_EVENT.set()
        print("Ctrl+C detected. Current replay (if any) will be canceled.")

    pt_session = PromptSession(style=pt_style, key_bindings=key_bindings)

    loaded_session_path: Optional[Path] = None
    if startup_session_path:
        sessions_dir = Path(current_config["sessions_save_dir"]) if current_config else Path.cwd()
        input_path = Path(startup_session_path)
        if input_path.is_absolute():
            resolved_path = input_path.expanduser().resolve()
        else:
            resolved_path = (sessions_dir / input_path).expanduser().resolve()
        if not resolved_path.exists() and resolved_path.suffix != ".json":
            resolved_path = resolved_path.with_suffix(".json")
        if not resolved_path.exists():
            print(f"{Colors.ERROR}Session file not found at {resolved_path}{Colors.RESET}")
            loaded_session = None
        else:
            loaded_session_path = resolved_path
            loaded_session = session_control.load(str(resolved_path)) # type: ignore
        if loaded_session:
            if loaded_session.get_conversations_count() == 0:
                print(f"{Colors.TOOL_WARNING}Loaded session '{loaded_session.name}' was empty. Creating a new conversation tree.{Colors.RESET}")
                loaded_chat_session = loaded_session.add_conversation(
                    parser_profile=_engine_parser_profile())
                _seed_chat_session_flags(loaded_chat_session)
            else:
                target_index = loaded_session.last_converation if 0 <= loaded_session.last_converation < loaded_session.get_conversations_count() else 0
                if target_index != loaded_session.last_converation:
                    print(f"{Colors.TOOL_WARNING}Stored conversation index {loaded_session.last_converation + 1} is out of range; defaulting to 1.{Colors.RESET}")
                loaded_chat_session = loaded_session.get_conversation(target_index)
            cursor = _bootstrap_cursor_for_session(loaded_session, conversation=loaded_chat_session)
            _set_active_cursor(cursor)
            print_current_session_compact(cursor, clear_first=False)
            active_conv_idx = _conversation_index(loaded_session, loaded_chat_session)
            if active_conv_idx is not None:
                print(f"{Colors.SYSTEM}Active conversation: {active_conv_idx + 1}{Colors.RESET}")
            await _check_engine_sync_status(cursor)
        else:
            print(f"{Colors.ERROR}Failed to load session from '{startup_session_path}'.{Colors.RESET}")

    replay_specs: List[str] = []
    for spec in (startup_replay_specs or []):
        replay_specs.extend([s for s in spec.split(",") if s.strip()])
    replay_ran = False
    if replay_specs:
        session = _require_current_cursor().session
        has_all = any(s.strip().lower() == "all" for s in replay_specs)
        if has_all and len(replay_specs) > 1:
            print(f"{Colors.TOOL_WARNING}Note: '--replay all' ignores other replay specs.{Colors.RESET}")
        if has_all:
            total_sources = session.get_conversations_count()
            print(f"{Colors.SYSTEM}Replaying all conversations: {total_sources} source(s).{Colors.RESET}")
            for src_idx in range(1, total_sources + 1):
                new_chat_session = session.insert_conversation(
                    index=-1,
                    template=_chat_session_template(),
                )
                _seed_chat_session_flags(new_chat_session)
                dest_idx = session.conversations.index(new_chat_session) + 1
                target_conversation = session.get_conversation(dest_idx - 1)
                cursor = _bootstrap_cursor_for_session(session, conversation=target_conversation)
                _set_active_cursor(cursor)
                await handle_command("/cls", cursor, pt_session)
                await _handle_replay_command(f"{src_idx} all-down", _require_current_cursor(), pt_session)
            replay_ran = True
        else:
            for spec in replay_specs:
                match = re.match(r"^(\d+)(?:-(\d+))?$", spec.strip())
                if not match:
                    print(f"{Colors.ERROR}Invalid --replay value '{spec}'. Use <source>[-dest] or 'all'.{Colors.RESET}")
                    continue
                src_idx = int(match.group(1))
                dest_idx = int(match.group(2)) if match.group(2) else None
                if src_idx < 1 or src_idx > session.get_conversations_count():
                    print(f"{Colors.ERROR}Replay source index {src_idx} out of range (1..{session.get_conversations_count()}).{Colors.RESET}")
                    continue
                replay_ready = True
                if dest_idx is not None:
                    if dest_idx < 1 or dest_idx > session.get_conversations_count():
                        print(f"{Colors.ERROR}Replay dest index {dest_idx} out of range (1..{session.get_conversations_count()}).{Colors.RESET}")
                        replay_ready = False
                if replay_ready:
                    if dest_idx is None:
                        new_chat_session = session.insert_conversation(
                            index=-1,
                            template=_chat_session_template(),
                        )
                        _seed_chat_session_flags(new_chat_session)
                        dest_idx = session.conversations.index(new_chat_session) + 1
                    target_conversation = session.get_conversation(dest_idx - 1)
                    cursor = _bootstrap_cursor_for_session(session, conversation=target_conversation)
                    _set_active_cursor(cursor)
                    await handle_command("/cls", cursor, pt_session)
                    await _handle_replay_command(f"{src_idx} all-down", _require_current_cursor(), pt_session)
                    replay_ran = True

    if startup_save_spec is not None and replay_ran:
        save_target = startup_save_spec.strip()
        session = _require_current_cursor().session
        if not save_target:
            if loaded_session_path:
                session_control.save(session, path=loaded_session_path) # type: ignore
            else:
                session_control.save(session) # type: ignore
        elif save_target.endswith(".json") or os.path.sep in save_target or (os.altsep and os.altsep in save_target):
            session_control.save(session, path=save_target) # type: ignore
        else:
            session.name = save_target # type: ignore
            session_control.save(session) # type: ignore

    if startup_quit:
        return

    while True:
        try:
            scope = _require_live_chat_scope(_require_current_cursor())
            context = _active_chat_context()
            if context:
                auto_iteration_requested = bool(scope.consume_auto_iteration())
            else:
                auto_iteration_requested = False

            if not auto_iteration_requested:
                try:
                    prompt_fragments = []
                    cursor_for_prompt = _normalize_cursor_after_auto_iters(_require_current_cursor(), prefer_session_main_leaf=True) or _require_current_cursor()
                    _set_active_cursor(cursor_for_prompt)
                    
                    chat_turns_count = cursor_for_prompt.user_turns_count()
                    prompt_fragments.append(('class:prompt.you', f"Turn {chat_turns_count + 1}->You: "))
                    
                    user_input_raw = await pt_session.prompt_async(prompt_fragments, key_bindings=key_bindings, enable_suspend=True)
                except (KeyboardInterrupt, EOFError):
                    print("\nInput cancelled. Type /q to quit or continue.")
                    continue

                user_input = user_input_raw.strip()
                if not user_input:
                    continue
                REPLAY_CANCELLATION_EVENT.clear()

                is_command = user_input.startswith("/")

                if is_command:
                    base_cursor = _normalize_cursor_after_auto_iters(_require_current_cursor(), prefer_session_main_leaf=True) or _require_current_cursor()
                    _set_active_cursor(base_cursor)
                    suppress_llm_call, should_exit, new_cursor = await handle_command(user_input_raw, base_cursor, pt_session)
                    if new_cursor:
                        _set_active_cursor(new_cursor)

                    if should_exit:
                        break
                    
                    if suppress_llm_call:
                        continue
                    else: # Command wants to trigger inference
                        cursor_for_iteration = _require_current_cursor()
                        if cursor_for_iteration.context:
                            scope = _scope_for_cursor(cursor_for_iteration)
                            if scope:
                                scope.request_auto_iteration()
                            else:
                                cursor_for_iteration.context.request_auto_iteration()
                        auto_iteration_requested = True # Fall through to the processing block
                
                elif user_input: # Is a user message
                    base_cursor = _normalize_cursor_after_auto_iters(_require_current_cursor(), prefer_session_main_leaf=True) or _require_current_cursor()
                    _set_active_cursor(base_cursor)
                    if _session_requires_sync(base_cursor.chat_session):
                        _warn_session_sync_required()
                        continue
                    cursor = await _handle_live_prompt(user_input, base_cursor, pt_session)
                    _set_active_cursor(cursor)
                    LAST_EXCEPTION_TRACEBACK = None
                    continue # Finished processing, loop for next prompt
                
                else: # Empty input
                    continue

            # This block handles auto-iteration from tools/commands or fall-through from a command
            if auto_iteration_requested:
                active_cursor = _require_current_cursor()
                cursor = await _process_turn_and_descendants(active_cursor, pt_session)
                _set_active_cursor(cursor)

        except Exception as e:
            _store_exception_traceback_if_clear(e)
            traceback.print_exc()
            await _wait_for_engine_ready()
        finally:
            is_manual_continue = False
 
# --- Main Execution ---

# --- Custom Colored Formatter for Console ---
class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt):
        super().__init__(fmt)
        self.FORMATS = {
            logging.DEBUG: Colors.LOG_DEBUG + fmt + Colors.RESET,
            logging.INFO: Colors.LOG_INFO + fmt + Colors.RESET,
            logging.WARNING: Colors.LOG_WARNING + fmt + Colors.RESET,
            logging.ERROR: Colors.LOG_ERROR + fmt + Colors.RESET,
            logging.CRITICAL: f"{Colors.BOLD}{Colors.LOG_ERROR}" + fmt + Colors.RESET,
        }

    def setLevel(self, level):
        # This method is not used for changing level, but required by the interface
        super().setLevel(level)


    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self._fmt)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def _set_console_log_level(level: int):
    """Finds the console handler and sets its level."""
    for handler in engine_logger.handlers:
        if handler.name == "console_handler":
            handler.setLevel(level)
            return

TERMINAL_BG_SET = False

def _attempt_set_dark_terminal_background() -> None:
    """Best-effort OSC 11 background set; ignored by unsupported terminals."""
    global TERMINAL_BG_SET
    if not sys.stdout.isatty():
        return
    try:
        sys.stdout.write("\033]11;#101010\007")
        sys.stdout.flush()
        TERMINAL_BG_SET = True
    except Exception:
        pass

def _attempt_reset_terminal_background() -> None:
    """Best-effort OSC 111 background reset; ignored by unsupported terminals."""
    if not TERMINAL_BG_SET or not sys.stdout.isatty():
        return
    try:
        sys.stdout.write("\033]111\007")
        sys.stdout.flush()
    except Exception:
        pass

atexit.register(_attempt_reset_terminal_background)

async def main_logic():
    global current_config, session_control, toolbox, EFFECTIVE_CONFIG_FILE_PATH, DUMP_INIT_ENABLED

    parser = argparse.ArgumentParser(description=f"{APP_NAME} - MP13 Playground Chat")
    parser.add_argument(
        "--log",
        type=str,
        default="warning",
        choices=["none", "error", "warning", "info", "all"],
        help="Set initial console logging level. Default: warning."
    )

    parser.add_argument(
        "--logfile",
        type=str,
        default="auto",
        help="Path to log file. 'none' disables file logging. 'auto' creates a temp file. Default: auto."
    )

    parser.add_argument(
        "--config", dest="config_path", type=str, default=None,
        help=f"Path to a custom configuration file (merged with default: {DEFAULT_CONFIG_FILE})"
    )
    parser.add_argument(
        "--base-model", dest="base_model_override", type=str, default=None,
        help="Path to base model (category-relative unless prefixed with ./ or ../)"
    )
    parser.add_argument(
        "--reconfigure", action="store_true", dest="force_reconfigure",
        help="Force interactive reconfiguration before loading the engine. Updates the target config file."
    )
    parser.add_argument(
        "--quantize_bits", dest="quantize_bits_override", type=str, default=None,
        choices=["none", "hqq", "eetq"],
        help="Override quantize_bits from config for this session only (none, hqq, or eetq)."
    )
    parser.add_argument(
        "--no-torch-compile", action="store_true",
        help="Disable torch.compile() for the base model, which is enabled by default. Overrides config file setting."
    )
    parser.add_argument(
        "--dump-init", action="store_true",
        help="Enable detailed dumping of applied patches and effective engine configuration upon initialization."
    )
    parser.add_argument(
        "--sess", dest="startup_session_path", type=str, default=None,
        help="Session path to load after engine initialization (absolute or relative to sessions dir)."
    )
    parser.add_argument(
        "--replay", dest="startup_replay_specs", action="append", default=None,
        help="Replay all-down from source with optional dest: <source>[-dest] or 'all'. Can be repeated."
    )
    parser.add_argument(
        "--save", dest="startup_save_spec", nargs="?", const="", default=None,
        help="Save after replay. Optional name/path; if omitted, save to loaded session path."
    )
    parser.add_argument(
        "--quit", dest="startup_quit", action="store_true",
        help="Quit after replay (and optional save)."
    )
    parser.add_argument(
        "config_path_name",
        nargs="?",
        default=None,
        help="Optional config file path or name. Relative names resolve under the default config dir.",
    )
    args = parser.parse_args()
    DUMP_INIT_ENABLED = args.dump_init # Set the global flag

    _attempt_set_dark_terminal_background()


    # --- Configure Logging ---
    log_level_map = {
        "error": logging.ERROR, "warning": logging.WARNING, "info": logging.INFO,
        "debug": logging.DEBUG, "all": logging.DEBUG, "none": logging.CRITICAL + 1
    }
    console_log_level = log_level_map.get(args.log.lower(), logging.WARNING)

    # Get the root logger
    engine_logger.setLevel(logging.DEBUG) # Set the lowest level on the logger itself
    engine_logger.propagate = False # Prevent messages from being passed to the root logger

    # Clear existing handlers to avoid duplicates if this is re-run
    engine_logger.handlers.clear()

    # Formatter strings
    log_format_string = '%(asctime)s - %(levelname)-8s - %(message)s'

    # --- File Handler ---
    if args.logfile.lower() != "none":
        if args.logfile.lower() == "auto":
            import tempfile
            temp_log_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".log", prefix="mp13chat_")
            log_file_path = temp_log_file.name
            temp_log_file.close()
        else:
            log_file_path = Path(args.logfile).expanduser().resolve()
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG) # File logging is always at DEBUG
        file_handler.setFormatter(logging.Formatter(log_format_string))
        engine_logger.addHandler(file_handler)
        print(f"Logging to file: {log_file_path}")

    # --- Console Handler ---
    if console_log_level <= logging.CRITICAL:
        console_handler = logging.StreamHandler()
        console_handler.set_name("console_handler") # Give it a name to find it later
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(ColoredFormatter(log_format_string))
        engine_logger.addHandler(console_handler)
        engine_logger.info(f"Console log level set to: {logging.getLevelName(console_log_level)}")

    print(f"Welcome to {APP_NAME}!")

    # --- Configuration Loading Logic ---
    cli_config_provided = bool(args.config_path or args.config_path_name)
    custom_config_arg = args.config_path or args.config_path_name
    custom_config_path = resolve_custom_config_path(
        custom_config_arg,
        DEFAULT_CONFIG_DIR,
    )

    default_config_source_path = DEFAULT_CONFIG_FILE
    default_config = load_json_config(default_config_source_path)
    if default_config is None:
        print(f"No default configuration found at {default_config_source_path}. Prompting for initial setup...")
        prompt_for_config(
            save_to_path=DEFAULT_CONFIG_FILE,
            prompt_for_name=False,
        )

    if custom_config_path and not custom_config_path.exists():
        print(f"{Colors.TOOL_WARNING}Custom config not found at {custom_config_path}; continuing with defaults.{Colors.RESET}")

    current_config, config_resolver, ok = load_effective_config(
        default_config_path=default_config_source_path,
        custom_config_path=custom_config_path,
        cwd=Path.cwd(),
    )
    current_config = _apply_config_defaults(current_config or {})

    if args.force_reconfigure:
        print("Forcing reconfiguration (from command line)...")
        current_config = prompt_for_config(
            config_to_update=current_config,
            save_to_path=EFFECTIVE_CONFIG_FILE_PATH,
            prompt_for_name=not cli_config_provided,
        )
        current_config, _, ok = load_effective_config(
            default_config_path=default_config_source_path,
            custom_config_path=custom_config_path,
            cwd=Path.cwd(),
        )
        current_config = _apply_config_defaults(current_config or {})

    if not ok or current_config is None:
        print(f"Error: Configuration could not be loaded or created at '{EFFECTIVE_CONFIG_FILE_PATH}'. Exiting.")
        sys.exit(1)

    if args.base_model_override:
        if config_resolver:
            resolved = resolve_engine_inputs({"base_model_path": args.base_model_override}, config_resolver)
            current_config["base_model_path"] = resolved.get("base_model_path", args.base_model_override)
        else:
            current_config["base_model_path"] = args.base_model_override
        print(f"{Colors.SYSTEM}Base model overridden from command line: {current_config['base_model_path']}{Colors.RESET}")

    if not current_config.get("base_model_path"):
        print(f"{Colors.TOOL_WARNING}Base model path not configured.{Colors.RESET}")
        model_input = input("Enter base model path (or press Enter to quit): ").strip()
        if not model_input:
            print("No base model path provided. Exiting.")
            return
        if config_resolver:
            resolved = resolve_engine_inputs({"base_model_path": model_input}, config_resolver)
            current_config["base_model_path"] = resolved.get("base_model_path", model_input)
        else:
            current_config["base_model_path"] = model_input

    print(f"Using configuration from: {EFFECTIVE_CONFIG_FILE_PATH}")

    session_control = SessionControl(Path(current_config["sessions_save_dir"]))
    
    # --- NEW: Manual Toolbox Initialization and Loading ---
    toolbox = Toolbox()
    tools_file = Path(current_config["tools_config_path"])
    if tools_file.exists():
        try:
            with open(tools_file, "r") as f:
                toolbox.from_dict(json.load(f), search_scope=globals(), external_handler=external_tool_handler)
            print(f"Toolbox state loaded from {tools_file}")
        except Exception as e:
            print(f"Warning: Could not load tools from {tools_file}. Starting with an empty toolbox. Error: {e}")
    # --- END NEW ---
    
    config_for_engine_init = current_config.copy()
    if args.quantize_bits_override:
        print(f"Overriding quantize_bits for this session to: {args.quantize_bits_override} (from command line)")
        config_for_engine_init["quantize_bits"] = args.quantize_bits_override

    # Set default for use_torch_compile, then check for override
    config_for_engine_init["use_torch_compile"] = config_for_engine_init.get("use_torch_compile", True)
    if args.no_torch_compile:
        print("Disabling torch.compile for this session (from command line).")
        config_for_engine_init["use_torch_compile"] = False
        if current_config: current_config["use_torch_compile"] = False # Also update current_config for /config command
    
    init_resp = await initialize_mp13_engine(config_for_engine_init)
    if not init_resp: print("Exiting: engine init fail."); return

    Path(current_config["adapters_root_dir"]).mkdir(parents=True, exist_ok=True)
    Path(current_config["sessions_save_dir"]).mkdir(parents=True, exist_ok=True)

    # The main logic is now wrapped in a try/finally to ensure shutdown is always called
    # within the same event loop.
    try:
        await chat_loop(
            init_resp,
            startup_session_path=args.startup_session_path,
            startup_replay_specs=args.startup_replay_specs,
            startup_save_spec=args.startup_save_spec,
            startup_quit=args.startup_quit,
        )
    except asyncio.CancelledError:
        # This can happen if the main task is cancelled during shutdown.
        # It's a normal part of the process.
        print("\nChat loop task was cancelled.")
    except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
        print("\nChat loop interrupted by user. Proceeding to shutdown.")
    except Exception as e:
        # This is the final safety net.
        print(f"\nAn unexpected error occurred in the chat loop: {e}")
        _store_exception_traceback_if_clear(e)
        print(traceback.format_exc())


async def main():
    """
    Main entry point that sets up signal handling and runs the primary application logic.
    """
    loop = asyncio.get_running_loop()

    # This function needs to be defined here to be in the scope of the key binding.
    async def _cancel_engine_op_async(): # noqa
        """Asynchronous wrapper to call the cancel-request tool."""
        print(f"\n{Colors.SYSTEM}Ctrl+C detected. Sending cancellation request to engine...{Colors.RESET}")
        try:
            # We call the API but don't need to display the full response here.
            # The primary goal is to send the signal. The prompt will be redrawn.
            await call_api("cancel-request")
            # A brief pause can allow the engine to process the cancellation
            # before the next prompt is displayed.
            await asyncio.sleep(0.1)
            REPLAY_CANCELLATION_EVENT.set()
        except Exception as e:
            print(f"{Colors.ERROR}Error sending cancellation request: {e}{Colors.RESET}")

    async def _suppressible_signal_handler():
        """
        Handles SIGINT by scheduling cancellation and then awaiting a dummy
        future to suppress the KeyboardInterrupt from being raised.
        """
        # Schedule the cancellation task to run on the event loop.
        asyncio.create_task(_cancel_engine_op_async())
        # Await a new, immediately resolved future. This is the key step
        # that consumes the signal and prevents the event loop from raising
        # KeyboardInterrupt in the main task.
        await asyncio.Future()

    # Set up the signal handler for SIGINT (Ctrl+C).
    # loop.add_signal_handler is the idiomatic way for asyncio apps.
    try:
        loop.add_signal_handler(
            signal.SIGINT,
            lambda: asyncio.create_task(_suppressible_signal_handler())
        )
    except (AttributeError, NotImplementedError):
        # Windows fallback: install a sync handler that schedules the async cancel.
        def _win_sigint_handler(signum, frame):
            try:
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(_cancel_engine_op_async())
                )
            except RuntimeError:
                # Loop not running yet or already closed; do nothing.
                pass
        signal.signal(signal.SIGINT, _win_sigint_handler)

    try:
        await main_logic()
    finally:
        # Ensure shutdown is called when the main logic exits for any reason.
        await shutdown_mp13_engine()
        _attempt_reset_terminal_background()
        print("MP13 Playground Chat closed.")

if __name__ == "__main__":
    try:
        DUMP_INIT_ENABLED = False # Default value before parsing args
        asyncio.run(main())
    except KeyboardInterrupt:
        # This will now only catch Ctrl+C if it happens during the initial,
        # non-async part of main() or if the signal handler setup fails.
        print("\nApplication interrupted during startup or shutdown. Exiting.")
    except Exception as e:
        print(f"Critical error: {e}"); traceback.print_exc()
