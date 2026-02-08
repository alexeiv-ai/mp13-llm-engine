# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import copy
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union, Literal, Set

from .engine_session import Colors, InferenceParams, Turn, EngineSession, ChatSession, Command, ReentrantWriterFairRWLock
from mp13_engine.mp13_toolbox import Toolbox, ToolsScope, ToolsView, ToolsAccess, ToolBoxRef
from mp13_engine.mp13_tools_parser import UnifiedToolIO
from mp13_engine.mp13_config import InferenceResponse, ParserProfile

CHAT_PARAM_NAMES: Tuple[str, ...] = (
    "stream",
    "cache",
    "return_prompt",
    "generation_config_template",
    "max_new_tokens",
    "no_tools_parse",
    "auto_retry_truncated",
    "suppress_full_response",
    "results_as_user_role",
    "pack_results_as_one_role",
    "advertised_tools",
    "silent_tools",
    "disabled_tools",
    "auto_tool_retry_limit",
    "auto_continue_retry_limit",
)

CHAT_FLAG_FIELDS: Set[str] = {
    "stream",
    "cache",
    "return_prompt",
    "generation_config_template",
    "max_new_tokens_override",
    "no_tools_parse",
    "auto_retry_truncated",
    "suppress_full_response",
    "results_as_user_role",
    "pack_results_as_one_role",
    "advertised_tools",
    "silent_tools",
    "disabled_tools",
    "auto_tool_retry_limit",
    "auto_continue_retry_limit",
}

PARAM_CHANGE_FIELDS: Set[str] = {
    "max_new_tokens_override",
}

PARAM_NAME_TO_CHANGE: Dict[str, str] = {
    "stream": "stream",
    "cache": "cache",
    "return_prompt": "return_prompt",
    "generation_config_template": "generation_config_template",
    "max_new_tokens": "max_new_tokens_override",
    "no_tools_parse": "no_tools_parse",
    "auto_retry_truncated": "auto_retry_truncated",
    "suppress_full_response": "suppress_full_response",
    "results_as_user_role": "results_as_user_role",
    "pack_results_as_one_role": "pack_results_as_one_role",
    "advertised_tools": "advertised_tools",
    "silent_tools": "silent_tools",
    "disabled_tools": "disabled_tools",
    "auto_tool_retry_limit": "auto_tool_retry_limit",
    "auto_continue_retry_limit": "auto_continue_retry_limit",
}

PARAM_CHANGE_TO_NAME: Dict[str, str] = {
    actual: public for public, actual in PARAM_NAME_TO_CHANGE.items()
}

_TRYOUT_KIND_NORMALIZATION: Dict[str, str] = {
    "auto_cont": "auto_continue_tryout",
    "auto_continue": "auto_continue_tryout",
    "auto_continue_tryout": "auto_continue_tryout",
    "auto_tool": "auto_tool_tryout",
    "auto_tool_tryout": "auto_tool_tryout",
}
_TRYOUT_KIND_SUFFIX = "_tryout"

def _normalize_tryout_kind(kind: Optional[str]) -> Set[str]:
    """Return all synonymous labels for a try-out kind (base + *_tryout variants)."""
    normalized: Set[str] = set()
    if not kind:
        return normalized
    normalized.add(kind)
    if kind.endswith(_TRYOUT_KIND_SUFFIX):
        base = kind[: -len(_TRYOUT_KIND_SUFFIX)]
        if base:
            normalized.add(base)
    else:
        normalized.add(f"{kind}{_TRYOUT_KIND_SUFFIX}")
    canonical = _TRYOUT_KIND_NORMALIZATION.get(kind)
    if not canonical and kind.endswith(_TRYOUT_KIND_SUFFIX):
        canonical = _TRYOUT_KIND_NORMALIZATION.get(kind[: -len(_TRYOUT_KIND_SUFFIX)])
    if canonical:
        normalized.add(canonical)
        if canonical.endswith(_TRYOUT_KIND_SUFFIX):
            base = canonical[: -len(_TRYOUT_KIND_SUFFIX)]
            if base:
                normalized.add(base)
        else:
            normalized.add(f"{canonical}{_TRYOUT_KIND_SUFFIX}")
    return normalized

"""
ChatContext & ChatCursor helpers for EngineSession integrations.
"""



@dataclass
class TryOutAnchor:
    """Represents an anchored try-out session."""
    anchor_name: str
    anchor_turn: Turn
    kind: str
    try_out_turns: List[Turn] = field(default_factory=list)
    try_out_cursor_ids: List[str] = field(default_factory=list)
    retries_remaining: int = 5
    retry_limit: int = 5
    origin_cursor_id: Optional[str] = None
    owner_scope: Optional["ChatContextScope"] = None


@dataclass
class ChatContextScope:
    """Runtime scope for chat-local state within a shared ChatContext."""
    context: "ChatContext"
    scope_id: str
    label: Optional[str] = None
    auto_mark_active: bool = True
    active_cursor_id: Optional[str] = None
    active_cursor_ref: Optional["ChatCursor"] = None
    active_cursor_override: Optional["ChatCursor"] = None
    pending_auto_iterations: int = 0
    bag_dict: Dict[str, Any] = field(default_factory=dict)
    is_disposed: bool = False

    def active_cursor(self) -> "ChatCursor":
        override = self.active_cursor_override
        if override and self.context.has_cursor(override) and getattr(override, "scope", None) is self:
            return override
        if override:
            self.active_cursor_override = None
        ref = self.active_cursor_ref
        if ref and self.context.has_cursor(ref) and getattr(ref, "scope", None) is self:
            return ref
        if self.active_cursor_id:
            cursor = self.context.get_cursor(self.active_cursor_id)
            if cursor:
                self.active_cursor_ref = cursor
                return cursor
        cursor = self.context._resolve_cursor(None, scope=self)
        self.active_cursor_ref = cursor
        return cursor

    def set_active_cursor(self, cursor_or_id: Union[str, "ChatCursor"]) -> "ChatCursor":
        return self.context.set_active_cursor(cursor_or_id, scope=self)

    def set_active_override(self, cursor: Optional["ChatCursor"]) -> None:
        if cursor and cursor.scope is not self:
            cursor.scope = self
        self.active_cursor_override = cursor

    def start_try_out_anchor(
        self,
        anchor_name: str,
        anchor_turn: Turn,
        *,
        kind: str = "try_out",
        retry_limit: int = 5,
        origin_cursor: Optional["ChatCursor"] = None,
    ) -> TryOutAnchor:
        return self.context.start_try_out_anchor(
            anchor_name,
            anchor_turn,
            kind=kind,
            retry_limit=retry_limit,
            origin_cursor=origin_cursor,
            scope=self,
        )

    def close_try_out_anchor(
        self,
        anchor_name: str,
        *,
        dist_mode: Optional[str] = "keep",
        main_thread_index: Optional[int] = None,
    ) -> Optional["ChatCursor"]:
        return self.context.close_try_out_anchor(
            anchor_name,
            dist_mode=dist_mode,
            main_thread_index=main_thread_index,
            scope=self,
        )

    def close_try_out_anchors_by_kind(
        self,
        kinds: Sequence[str],
        *,
        dist_mode: Optional[str] = "keep",
        anchor_scope: Optional[Set["Turn"]] = None,
    ) -> Optional["ChatCursor"]:
        return self.context.close_try_out_anchors_by_kind(
            kinds,
            dist_mode=dist_mode,
            anchor_scope=anchor_scope,
            scope=self,
        )

    def find_active_anchor(self, kind: str, cursor: Optional["ChatCursor"] = None) -> Optional[TryOutAnchor]:
        return self.context.find_active_anchor(kind, cursor=cursor, scope=self)

    def get_try_out_anchor(self, anchor_name: str) -> Optional[TryOutAnchor]:
        return self.context.get_try_out_anchor(anchor_name, scope=self)

    def request_auto_iteration(self, count: int = 1) -> None:
        self.context.request_auto_iteration(count=count, scope=self)

    def consume_auto_iteration(self) -> bool:
        return self.context.consume_auto_iteration(scope=self)

    def register_cursor_for_turn(
        self,
        target_turn: Turn,
        *,
        alias: Optional[str] = None,
        make_active: bool = False,
    ) -> Optional["ChatCursor"]:
        return self.context.register_cursor_for_turn(
            target_turn,
            alias=alias,
            make_active=make_active,
            scope=self,
        )

    def register_cursor_if_needed(
        self,
        cursor: "ChatCursor",
        *,
        alias: Optional[str] = None,
        make_active: bool = False,
    ) -> "ChatCursor":
        return self.context.register_cursor_if_needed(
            cursor,
            alias=alias,
            make_active=make_active,
            scope=self,
        )

    def adopt_cursor(
        self,
        cursor: "ChatCursor",
        alias: Optional[str] = None,
        *,
        make_active: bool = False,
    ) -> "ChatCursor":
        return self.context.adopt_cursor(
            cursor,
            alias=alias,
            make_active=make_active,
            scope=self,
        )

    def resurrect_cursor_for_gen_id(
        self,
        gen_id: str,
        *,
        alias: Optional[str] = None,
        make_active: bool = False,
    ) -> "ChatCursor":
        return self.context.resurrect_cursor_for_gen_id(
            gen_id,
            alias=alias,
            make_active=make_active,
            scope=self,
        )

    def try_out_anchors_snapshot(self) -> List[TryOutAnchor]:
        return self.context.try_out_anchors_snapshot(scope=self)

    def cursors_snapshot(self) -> List[Tuple[str, "ChatCursor"]]:
        """Snapshot of registered cursors limited to this scope."""
        cursors = self.context.cursors_snapshot()
        return [(handle, cursor) for handle, cursor in cursors if getattr(cursor, "scope", None) is self]

    def active_cursor_id(self) -> Optional[str]:
        return self.context.active_cursor_id(self)

    def drop_cursor(self, cursor_or_id: Union[str, "ChatCursor"]) -> None:
        cursor: Optional["ChatCursor"]
        if isinstance(cursor_or_id, ChatCursor):
            cursor = cursor_or_id
        else:
            cursor = self.context.get_cursor(cursor_or_id)
        if cursor is None:
            raise KeyError(f"Unknown cursor id '{cursor_or_id}'.")
        if getattr(cursor, "scope", None) is not self:
            raise ValueError("Cursor does not belong to this scope.")
        self.context.drop_cursor(cursor)

    def resolve_try_out_cursor(
        self,
        anchor: TryOutAnchor,
        *,
        prefer_latest: bool = True,
    ) -> Optional["ChatCursor"]:
        cursor = self.context.resolve_try_out_cursor(anchor, prefer_latest=prefer_latest)
        if cursor and getattr(cursor, "scope", None) is self:
            return cursor
        return None

    def find_cursor_by_gen_id(self, gen_id: Optional[str]) -> Optional["ChatCursor"]:
        cursor = self.context.find_cursor_by_gen_id(gen_id)
        if cursor and getattr(cursor, "scope", None) is self:
            return cursor
        return None

    def resurrect_try_out_anchor(self, anchor_name: str) -> Optional[TryOutAnchor]:
        return self.context.resurrect_try_out_anchor(anchor_name, scope=self)

class ChatContextToolBoxRef(ToolBoxRef):
    """Context-aware ToolBoxRef that persists changes on the root turn."""

    def __init__(self, context: "ChatContext", toolbox: Toolbox):
        self._context = context
        initial_scope = context._load_context_tools_scope()
        super().__init__(toolbox=toolbox, scope=initial_scope)

    def _scope_updated(self) -> None:
        self._context._persist_context_tools_scope(self.scope)

# ------------------------------ ChatCursor ---------------------------------
class ChatCursor:
    """Stateful view over a single progressing branch of a conversation.

    Key behaviors:
    - Tracks `root` and moving `head` turns inside `session`.
    - add_batch(...) returns a ChatForks (multiple concurrent batches supported).
    - StartFork()/CloseFork() manage fork branches via an internal stack.
    - real-user-turn counter (non-automation) is stable for forks.
    - clone() makes a shallow copy preserving global knobs.
    - push/pop helpers for system message and active adapter names.
    - save_api_call(...) records engine call signatures for replay.
    """

    class WalkItem(NamedTuple):
        """
        Snapshot produced by `iter_spine_tree`.

        `cursor` and `context_cursor` are lightweight clones detached from the
        active cursor – consumers should treat them as read-only throwaways.
        The iterators never mutate global cursor state; callers that need to
        walk away from the yielded node should make their own clones.
        """

        cursor: "ChatCursor"
        depth: int
        relation: Literal["spine", "fork"]
        is_last_sibling: bool
        has_forks: bool
        context_cursor: Optional["ChatCursor"]
        is_peek: bool
        peek_kind: Optional[str]
        collected: bool

    # ---- construction -----------------------------------------------------
    def __init__(
        self,
        context: "ChatContext",
        head: Turn|None = None,
        *,
        scope: Optional["ChatContextScope"] = None,
        origin_user_turn_index: Optional[int] = None,
        is_fork: bool = False,
        gen_config: Optional[Dict[str, Any]] = None,
        streaming: Optional[bool] = None,
        cache: Optional[str] = None,
        return_prompt: Optional[str] = None,
        adapter_override_for_next_turn: Optional[List[str]] = None,
    ) -> None:
        self.context: Optional["ChatContext"] = context
        self.session: EngineSession = context.session
        self.context_id: Optional[str] = None
        self.root: Turn = context.root_turn
        self.head: Turn = head or self.root
        self.scope: Optional["ChatContextScope"] = scope or getattr(context, "default_scope", None)

        self.is_fork: bool = is_fork
        # stack frames remember prior cursor states and fork/batch origins # noqa
        # frame schema examples:
        #   {"kind":"fork", "cursor": <ChatCursor clone>, "origin_head": <Turn>}
        #   {"kind":"batch", "cursor": <ChatCursor clone>, "batch_hub": <ChatCursor>}
        self._stack: List[Dict[str, Any]] = []
        self._pending_fork_split: bool = False

        # counters
        self.origin_user_turn_index: int = (
            origin_user_turn_index
            if origin_user_turn_index is not None
            else _count_real_user_turns_on_path(self.root, self.head)
        )

        # global knobs
        self.gen_config: Dict[str, Any] = dict(gen_config or {})
        self.stream_override: Optional[bool] = streaming
        self.cache_override: Optional[str] = cache
        self.return_prompt_override: Optional[str] = return_prompt
        self.generation_config_template_override: Optional[Dict[str, Any]] = None
        self.max_new_tokens_override: Optional[int] = None
        self.no_tools_parse_override: Optional[bool] = None
        self.auto_retry_truncated_override: Optional[bool] = None
        self.suppress_full_response_override: Optional[bool] = None
        self.reset_metrics_override: bool = False

        # one-shot adapter override
        self.adapter_override_for_next_turn: Optional[List[str]] = (
            list(adapter_override_for_next_turn) if adapter_override_for_next_turn else None
        )

        self._tools_view_override: Optional[ToolsView] = None

        # concurrent forks created from this cursor
        self.forks: List[ChatForks] = []

    @property
    def is_registered(self) -> bool:
        """Return True when this cursor is tracked by its context."""
        return bool(self.context and self.context.has_cursor(self))

    @property
    def is_disposed(self) -> bool:
        """Return True when this cursor is orphaned."""
        return not self.context

    @property
    def is_closed(self) -> bool:
        """Return True when the current head has a complete user+assistant exchange."""
        return bool(self.head and (not self.head.IsEmpty and self.head.HasResponse))

    # ---- internal helpers -------------------------------------------------
    def _reset_param_overrides(self) -> None:
        self.stream_override = None
        self.cache_override = None
        self.return_prompt_override = None
        self.generation_config_template_override = None
        self.max_new_tokens_override = None
        self.no_tools_parse_override = None
        self.auto_retry_truncated_override = None
        self.suppress_full_response_override = None
        self.reset_metrics_override = False

    def _assign_head(self, new_head: Turn) -> None:
        previous_head = getattr(self, "head", None)
        self.head = new_head
        if previous_head is not new_head:
            self._reset_param_overrides()

    def _mark_active(self) -> None:
        # TBC: Explicit scope.set_active_cursor(...) avoids cross-scope drift,
        # makes intent clear for callers, and removes hidden side effects.
        if self.context and self.scope and self.scope.auto_mark_active:
            self.context.set_active_cursor(self, scope=self.scope)

    def clone_at(self, turn: Turn, *, resolve_existing: bool = False) -> "ChatCursor":
        """Clone a cursor pinned to `turn` without changing active state."""
        if not self.context:
            raise RuntimeError("ChatCursor requires a ChatContext to resolve alternate turns.")
        if resolve_existing:
            resolved = self.context.find_cursor_for_turn(turn)
            if resolved:
                return resolved
        clone = self.clone()
        clone._assign_head(turn)
        return clone

    def rebind_to_turn(self, turn: Turn, *, allow_unowned: bool = False) -> "ChatCursor":
        """Move this cursor's head in-place to `turn`.

        Deprecated: prefer ChatContext.register_cursor_for_turn(...) or
        ChatContext.find_cursor_for_turn(...) to avoid mutating existing cursors
        that may be referenced elsewhere. Use rebind only as a last-resort fallback
        when no valid registered cursor can be resolved or built.
        """
        if not turn:
            raise ValueError("rebind_to_turn requires a target turn.")
        if self.context and not allow_unowned and not self.context.owns_turn(turn):
            raise ValueError("Target turn is not managed by this ChatContext.")
        self._assign_head(turn)
        return self

    def snapshot_at(self, turn: Turn) -> "ChatCursor":
        """Create a detached cursor snapshot at `turn` without context lookups."""
        clone = self.clone()
        clone._assign_head(turn)
        return clone

    # ---- turn metadata -----------------------------------------------------
    def is_root_context(self) -> bool:
        return bool(getattr(self.head, "root_context", False))

    def set_root_context(self, state: bool) -> None:
        if state:
            self.head.root_context = True
        else:
            setattr(self.head, "root_context", False)

    def is_archived(self) -> bool:
        return bool(getattr(self.head, "is_archived", False))

    def set_archived(self, state: bool) -> None:
        self.head.is_archived = bool(state)

    def find_closest_fork(self) -> "ChatCursor":
        cursor = self.clone()
        while cursor.head.parent:
            if cursor.head.parent.turn_type in [Turn.FORK, Turn.BATCH]:
                return cursor
            cursor.head = cursor.head.parent
        return cursor

    def is_main_thread(self) -> bool:
        """Gets the main_thread flag of the current turn."""
        return self.head.main_thread

    def set_main_thread(self, state: bool) -> None:
        """Sets the main_thread flag on the current turn."""
        self.head.main_thread = state

    def is_auto(self) -> bool:
        """Gets the is_aotu flag of the current turn."""
        return self.head.is_auto

    def set_auto(self, is_auto: bool) -> None:
        """Set or clear the is_auto flag on the current user turn."""
        self.head.is_auto = is_auto

    def turn_type(self) -> Optional[str]:
        return getattr(self.head, "turn_type", None)

    def gen_id_or_parent(self) -> str:
        return getattr(self.head, "gen_id_or_parent", None) or (self.head.gen_id or "N/A")

    def sibling_index(self) -> Optional[int]:
        """Return zero-based index of this head among parent's children."""
        parent = getattr(self.head, "parent", None)
        if not parent:
            return None
        siblings: Sequence[Turn] = list(parent.turns or [])
        try:
            return siblings.index(self.head)
        except ValueError:
            return None

    def child_cursors(self) -> List["ChatCursor"]:
        """Return clones for each child turn to avoid exposing Turn pointers."""
        children: Sequence[Turn] = list(getattr(self.head, "turns", []) or [])
        return [self.clone_at(child) for child in children]

    def trim_turns(
        self,
        levels_to_trim: int,
    ) -> Tuple[Optional["ChatCursor"], List[Command], bool, str]:
        """
        Delegate to EngineSession.trim_up() while returning a cursor for the new head.
        """
        if self.context:
            self.context._ensure_turn_in_scope(self.head)

        start_trim_node = self.head
        if getattr(start_trim_node, "IsEmpty", False) and getattr(start_trim_node, "parent", None):
            start_trim_node = start_trim_node.parent

        trimmed_turns: Set[Turn] = set()

        if self.context:
            root_turn = self.context.root_turn
            path = self.session.get_active_path(start_trim_node)
            if root_turn not in path:
                return None, [], False, "Cannot trim outside the context root."
            scoped_path = path[path.index(root_turn):]
            if levels_to_trim > 0:
                user_turns = 0
                for turn in reversed(scoped_path):
                    if turn.data.get("user") and not turn.is_auto:
                        user_turns += 1
                    if user_turns >= levels_to_trim:
                        break
                if user_turns < levels_to_trim:
                    return None, [], False, "Cannot trim beyond the context root."


        if levels_to_trim == 0:
            trimmed_turns = set(self.head.get_all_descendants())
        else:
            path_to_active = self.session.get_active_path(start_trim_node)
            if self.context and self.context.root_turn in path_to_active:
                path_to_active = path_to_active[path_to_active.index(self.context.root_turn):]
            path_to_active.reverse()
            turn_to_delete = None
            user_turns_counted = 0
            for turn in path_to_active:
                if turn.data.get("user") and not turn.is_auto:
                    user_turns_counted += 1
                if user_turns_counted >= levels_to_trim:
                    turn_to_delete = turn
                    break
            if turn_to_delete:
                if self.context and self.context.root_turn is turn_to_delete:
                    root_cleared = True
                if turn_to_delete.parent and turn_to_delete.parent.turn_type == Turn.BATCH:
                    trimmed_turns = set(turn_to_delete.parent.get_all_descendants())
                else:
                    trimmed_turns = {turn_to_delete, *turn_to_delete.get_all_descendants()}

        new_turn, commands_to_reverse, success, message = self.session.trim_up(
            active_turn=self.head,
            levels_to_trim=levels_to_trim,
        )
        new_cursor: Optional["ChatCursor"] = None
        if success and new_turn:
            if self.context:
                new_cursor = self.context.register_cursor_for_turn(new_turn, make_active=False)
            if not new_cursor:
                new_cursor = self.clone_at(new_turn)
        if success and self.context and trimmed_turns:
            keep_handles: Set[str] = set()
            if new_cursor and new_cursor.context_id:
                keep_handles.add(new_cursor.context_id)
            self.context._drop_tracked_items_for_trim(trimmed_turns, keep_handles=keep_handles)
        if success and self.context:
            root_turn = self.context.root_turn
            if root_turn:
                root_turn.root_context = True
        return new_cursor, list(commands_to_reverse or []), success, message

    # ---- navigation helpers -------------------------------------------------
    def parent_cursor(self) -> Optional["ChatCursor"]:
        parent = getattr(self.head, "parent", None)
        if not parent:
            return None
        if self.context:
            return self.context.register_cursor_for_turn(parent, make_active=False) or self.clone_at(parent)
        return self.clone_at(parent)

    def child_cursor(self, index: int = 0) -> Optional["ChatCursor"]:
        children: Sequence[Turn] = list(getattr(self.head, "turns", []) or [])
        if not children or index < 0 or index >= len(children):
            return None
        if self.context:
            return self.context.register_cursor_for_turn(children[index], make_active=False) or self.clone_at(children[index])
        return self.clone_at(children[index])

    def sibling_cursor(self, offset: int) -> Optional["ChatCursor"]:
        parent = getattr(self.head, "parent", None)
        if not parent:
            return None
        siblings: Sequence[Turn] = list(parent.turns or [])
        if not siblings:
            return None
        try:
            idx = siblings.index(self.head)
        except ValueError:
            return None
        target_idx = idx + offset
        if target_idx < 0 or target_idx >= len(siblings):
            return None
        if self.context:
            return self.context.register_cursor_for_turn(siblings[target_idx], make_active=False) or self.clone_at(siblings[target_idx])
        return self.clone_at(siblings[target_idx])

    def next_sibling_cursor(self) -> Optional["ChatCursor"]:
        return self.sibling_cursor(1)

    def prev_sibling_cursor(self) -> Optional["ChatCursor"]:
        return self.sibling_cursor(-1)

    def main_leaf_cursor(self, conversation_index: Optional[int] = None) -> Optional["ChatCursor"]:
        idx = conversation_index
        if idx is None:
            chat = self.context.chat_session if self.context and self.context.chat_session else self.chat_session
            if chat and chat in self.session.conversations:
                idx = self.session.conversations.index(chat)
        turn = self.session.get_main_branch_leaf(idx) if idx is not None else self.session.get_main_branch_leaf()
        if not turn:
            return None
        if self.context:
            return self.context.register_cursor_for_turn(turn, make_active=False) or self.clone_at(turn)
        return self.clone_at(turn)

    def cursor_for_gen_id(self, gen_id: str, chat: Optional[ChatSession] = None) -> "ChatCursor":
        if not gen_id:
            raise ValueError("cursor_for_gen_id requires a gen_id value.")
        target_chat = chat or self.chat_session
        if target_chat is None:
            raise RuntimeError("cursor_for_gen_id requires an associated chat session.")
        turn = self.session.get_turn_by_gen_id(gen_id, target_chat)
        if not turn:
            raise KeyError(f"Turn '{gen_id}' was not found in the current chat session.")
        if self.context and not self.context.owns_turn(turn):
            raise ValueError(f"Turn '{gen_id}' is not within the active ChatContext scope.")
        return self.clone_at(turn)

    # ---- basic props ------------------------------------------------------
    @property
    def id(self) -> str:
        return self.head.gen_id or "N/A"

    def path(self) -> List[Turn]:
        return list(self.session.get_active_path(self.head))

    @property
    def label(self) -> str:
        """Stable human-readable identifier (falls back to context id when available)."""
        if self.context_id:
            return self.context_id
        return self.id

    def display_id(
        self,
        turn: Optional[Turn] = None,
        *,
        active_cursor: Optional["ChatCursor"] = None,
    ) -> str:
        """Pretty display label using the session's formatting."""
        target = turn or self.head
        if not target:
            return "N/A"

        if active_cursor:
            active_turn = active_cursor.current_turn
        else:
            active_turn = self.current_turn

        return self.session.get_display_id(target, active=active_turn)
    @property
    def current_turn(self) -> Turn:
        """Read-only handle for the cursor's head turn."""
        return self.head

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Effective LLM messages (system + history) for the current head."""
        return self.session.get_llm_messages(self.head)

    @property
    def active_adapters(self) -> List[str]:
        """Current adapter stack hint tracked locally."""
        return self.session.get_effective_adapters(self.head)

    @property
    def system_message(self) -> Optional[str]:
        return self.effective_system_message()

    @property
    def toolbox(self) -> Optional[Toolbox]:
        if self.context:
            return self.context.toolbox
        return None

    @property
    def chat_session(self) -> Optional[ChatSession]:
        if self.context:
            return self.context.chat_session
        return None

    def get_initial_param(self, key: str, default: Any = None) -> Any:
        chat = self.chat_session
        if chat and getattr(chat, "initial_params", None):
            return chat.initial_params.get(key, default)
        return default

    @property
    def parser_profile(self) -> Optional[ParserProfile]:
        chat = self.chat_session
        if chat:
            return chat.parser_profile
        return None

    @property
    def tool_parser_profile(self) -> Optional[ParserProfile]:
        return self.parser_profile

    def bind_context(self, context: Optional["ChatContext"], *, context_id: Optional[str] = None) -> None:
        """Attach the cursor to a ChatContext registry."""
        self.context = context
        if context_id:
            self.context_id = context_id

    def dispose(self) -> None:
        """Remove this cursor from its context registry and orphan."""
        if self.context:
            self.context.drop_cursor(self)

    @property
    def tools_view(self) -> Optional[ToolsView]:
        return self._tools_view_override

    def set_tools_view(self, view: Optional[ToolsView]) -> None:
        self._tools_view_override = view

    def real_user_turns(self) -> int:
        if self.is_fork:
            return self.origin_user_turn_index
        return _count_real_user_turns_on_path(self.root, self.head)


    # ---- session-aware state helpers --------------------------------------
    def apply_system_message(
        self,
        operation: str,
        value: Optional[str] = None,
        *,
        command_text: Optional[str] = None,
        label: Optional[str] = None,
        value_kind: Optional[str] = None,
        stack_id: Optional[str] = None,
    ) -> "ChatCursor":
        updated = self.session.add_system_message_command(
            self.head,
            operation,
            value,
            command_text,
            label,
            value_kind=value_kind,
            stack_id=stack_id,
        )
        self._assign_head(updated)
        self._mark_active()
        return self

    def resolve_stack_id(self, target_id: str, *, change_key: str) -> str:
        """Resolve a pop target into a stack_id for the given change key."""
        return self.session._resolve_stack_id(self.head, target_id, change_key=change_key)

    def resolve_system_stack_id(self, target_id: str) -> str:
        return self.resolve_stack_id(target_id, change_key="system_message")

    def resolve_adapter_stack_id(self, target_id: str) -> str:
        return self.resolve_stack_id(target_id, change_key="adapters_command")

    def resolve_tools_stack_id(self, target_id: str) -> str:
        return self.resolve_stack_id(target_id, change_key="tools_scope")

    def apply_adapter_operation(
        self,
        operation: str,
        adapters: Optional[Sequence[str]] = None,
        *,
        command_text: Optional[str] = None,
        stack_id: Optional[str] = None,
    ) -> "ChatCursor":
        adapter_list = list(adapters) if adapters is not None else None
        updated = self.session.add_adapter_command(
            self.head,
            operation,
            adapter_list,
            command_text,
            stack_id=stack_id,
        )
        self._assign_head(updated)
        self._mark_active()
        return self

    def apply_tools_scope(
        self,
        operation: str,
        scope: Optional[ToolsScope] = None,
        *,
        command_text: Optional[str] = None,
        stack_id: Optional[str] = None,
    ) -> "ChatCursor":
        """
        Mutate the tools scope stack for the active turn.
        - "set" replaces the entire stack with the provided scope (or clears it if empty).
          If the scope omits mode, the toolbox global mode still applies (often re-advertising all active tools).
        - "add" pushes a new scope layer evaluated after the existing ones.
        - "pop"/"reset" unwind the stack.
        """
        updated = self.session.add_tools_scope_command(
            self.head,
            operation,
            scope,
            command_text,
            stack_id=stack_id,
        )
        # invalidate memoized view; caller can refresh via ChatContext.refresh_tools_view
        self._tools_view_override = None
        self._assign_head(updated)
        self._mark_active()
        return self

    def log_command(self, command_text: str, *, metadata: Optional[Dict[str, Any]] = None) -> Command:
        new_turn = self.session.add_log_command(self.head, command_text, metadata)
        self._assign_head(new_turn)
        self._mark_active()
        return new_turn.cmd[-1]

    def command_event(self, command_text: str, *, metadata: Optional[Dict[str, Any]] = None) -> Command:
        """Record a client-side command using Command.COMMAND for replayability."""
        new_turn = self.session.add_client_command(self.head, command_text, metadata)
        self._assign_head(new_turn)
        self._mark_active()
        return new_turn.cmd[-1]

    def set_replace_command_in_turn(
        self,
        target_turn: Turn,
        command_text: str,
        predicate: Callable[[Command], bool],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        insert_at: Optional[int] = None,
    ) -> Command:
        """Replace or set a client-side command on a specific turn without moving the cursor."""
        if target_turn is None:
            raise ValueError("Cannot add a command to an empty turn.")
        if self.context and not self.context.owns_turn(target_turn):
            raise ValueError("Target turn is outside of the active chat context.")
        payload = {"command": command_text}
        if metadata:
            payload.update(metadata)
        new_command = Command(cmd_type=Command.COMMAND, metadata=payload)
        added = self.session.replace_command_in_turn(
            target_turn,
            predicate,
            new_command,
            insert_at=insert_at,
        )
        if not added:
            raise RuntimeError("Failed to add command to target turn.")
        return added

    def remove_commands_in_turn(
        self,
        target_turn: Optional[Turn],
        predicate: Callable[[Command], bool],
    ) -> List[Command]:
        """Remove commands from a turn that satisfy `predicate` (no cursor movement)."""
        if target_turn is None:
            return []
        if self.context and not self.context.owns_turn(target_turn):
            raise ValueError("Target turn is outside of the active chat context.")
        return self.session.remove_commands_in_turn(target_turn, predicate)

    def add_command_copy(
        self,
        cmd_obj: Command,
        *,
        override_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Command]:
        """Clone a command onto the current head without altering cursor stacks."""
        if not self.head:
            return None
        clone = Command(
            cmd_type=cmd_obj.cmd_type,
            metadata=copy.deepcopy(getattr(cmd_obj, "data", {}) or {}),
            api_name=copy.deepcopy(getattr(cmd_obj, "api_name", None)),
            api_params=copy.deepcopy(
                override_params if override_params is not None else getattr(cmd_obj, "api_params", None)
            ),
        )
        clone.timestamp = getattr(cmd_obj, "timestamp", time.time())
        return self.session.add_command_to_turn(clone, self.head)

    def get_request_id(self, prefix: str = "") -> str:
        target = self.head
        return self.session.get_request_id(target, prefix)

    def get_effective_adapters(self, turn: Optional[Turn] = None) -> List[str]:
        target = turn or self.head
        return list(self.session.get_effective_adapters(target))

    def get_system_message_segments(self) -> Tuple[Dict[str, str], bool]:
        target = self.head
        return self.session.get_system_message_segments(target)

    def effective_system_message(self, turn: Optional[Turn] = None) -> Optional[str]:
        target = turn or self.head
        return self.session.get_effective_system_message(target)

    def update_metrics(self, metrics: Mapping[str, Any]) -> None:
        if not metrics:
            return
        if not self.head:
            raise RuntimeError("ChatCursor.update_metrics requires a head turn.")
        self.session.add_update_metrics(self.head, dict(metrics))

    def get_data(self) -> Mapping[str, Any]:
        if not self.head or not hasattr(self.head, "data"):
            raise RuntimeError("ChatCursor.get_data requires an attached head turn with data.")
        return self.head.data

    def user_turns_count(self, start: Optional[Turn] = None) -> int:
        target = start or self.head
        return self.session.user_turns_count(start_node=target)

    def param_change(
        self,
        change_type: str,
        new_value: Any,
        *,
        command_text: Optional[str] = None,
    ) -> Command:
        new_turn = self.session.add_param_change(self.head, change_type, new_value, command_text)
        self._assign_head(new_turn)
        self._mark_active()
        return new_turn.cmd[-1]

    def state_change(
        self,
        change_type: str,
        value: Any,
        *,
        command_text: Optional[str] = None,
    ) -> Command:
        new_turn = self.session.add_state_change(self.head, change_type, value, command_text)
        self._assign_head(new_turn)
        self._mark_active()
        return new_turn.cmd[-1]

    # ---- fork lifecycle ---------------------------------------------------
    def _close_branch_node(
        self,
        target: Optional[Turn] = None,
    ) -> bool:
        """Close the branch containing ``target`` (defaults to cursor head)."""
        target_turn = target or self.head
        if not target_turn:
            return False
        placeholder = self.session.close_branch(target_turn)
        if placeholder is None:
            return False
        self._assign_head(placeholder)
        return True

    # ---- chat building ----------------------------------------------------
    def add_user(
        self,
        text: str,
        *,
        archived: bool = False,
        meta: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> "ChatCursor":
        """Append a user message under current head.
        - If StartFork() was called, this becomes a branch and the cursor
          returned continues on the fork (stack holds the original).
        - Otherwise, advances this cursor on the main/fork branch.
        """
        meta = meta or {}
        pending_branch = self._pending_fork_split
        new_turn: Turn = self.session.add_user(
            content=text,
            current_turn=self.head,
            archived=archived, # noqa
            **extra,
        )

        if pending_branch:
            # consume flag and switch this cursor to the fork branch view
            self._pending_fork_split = False
            # Fork branches inherit the mainline real-user count at split # noqa
            self._assign_head(new_turn)
            self.is_fork = True
            self.origin_user_turn_index = self.real_user_turns()  # inherited snapshot
        else:
            self._assign_head(new_turn)

        return self


    def add_message(self, role: str, content: Any, **kwargs: Any) -> "ChatCursor":
        """Add an arbitrary chat message and update the cursor head."""
        new_turn = self.session.add_message(role, content, current_turn=self.head, **kwargs)
        self._assign_head(new_turn)
        return self

    def add_continuation_turn(self, **kwargs: Any) -> "ChatCursor":
        """Insert a continuation placeholder beneath the current head and advance."""
        new_turn = self.session.add_continuation_turn(self.head, **kwargs)
        self._assign_head(new_turn)
        return self

    def add_assistant(
        self,
        content: Optional[str],
        *,
        tool_blocks: Optional[Sequence[Any]] = None,
        archived: bool = False,
        was_truncated: bool = False,
        was_canceled: bool = False,
        **extra: Any,
    ) -> Turn:
        target = self.head
        self.session.add_assistant(
            content,
            tool_blocks=list(tool_blocks) if tool_blocks is not None else None,
            archived=archived,
            was_truncated=was_truncated,
            was_canceled=was_canceled,
            destination_turn=target,
            **extra,
        )
        return target

    def add_adapters_state(self, state_type: str, value: Any) -> Turn:
        """Record adapter state reconstruction metadata on the current head."""
        updated = self.session.add_adapters_state(self.head, state_type, value)
        self._assign_head(updated)
        return updated

    def add_tool_results(self, tool_blocks: Sequence[Any]) -> "ChatCursor":
        new_turn = self.session.add_tool_results(
            tool_blocks=list(tool_blocks),  # noqa
            current_turn=self.head,
        )
        self._assign_head(new_turn)
        return self

    def add_try_out(
        self,
        *,
        anchor: Optional["TryOutAnchor"] = None,
        anchor_turn: Optional[Turn] = None,
        keep_in_main: bool = False,
        convert_existing: bool = False,
    ) -> Tuple["ChatCursor", "ChatCursor"]:
        """Create placeholders for a try-out branch and return cursors to track them."""
        anchor_turn = anchor_turn or self.head
        if anchor_turn is None:
            raise ValueError("Cannot create try-out without a valid anchor turn.")

        new_anchor_name = anchor.anchor_name if anchor else None
        existing_meta = (getattr(anchor_turn, "data", None) or {}).get("$try_out")
        existing_anchor = existing_meta.get("anchor") if isinstance(existing_meta, dict) else None

        if existing_anchor and new_anchor_name and existing_anchor != new_anchor_name:
            if getattr(anchor_turn, "turns", None):
                raise ValueError(
                    f"Cannot create try-out '{new_anchor_name}' on a turn already attached to '{existing_anchor}'."
                )
            # Grow down under this node to avoid $try_out metadata clashes.
            if anchor_turn.turn_type is None:
                try:
                    self.session._mark_placeholder_anchor(anchor_turn, Turn.HOLD)
                except Exception:
                    pass
            main_turn = Turn(turn_type=None, metadata={})
            main_turn.main_thread = True
            try_turn = Turn(turn_type=None, metadata={})
            try_turn.main_thread = bool(keep_in_main)
            self.session._add_chat_turn(anchor_turn, main_turn)
            self.session._add_chat_turn(anchor_turn, try_turn)
        else:
            main_turn, try_turn = self.session.add_try_out(
                anchor_turn,
                keep_in_main=keep_in_main,
                convert_existing=convert_existing,
            )
        if keep_in_main:
            try_turn.main_thread = True
            main_turn.main_thread = True

        if anchor:
            try_meta = {"anchor": anchor.anchor_name, "kind": anchor.kind, "role": "try"}
            main_meta = {"anchor": anchor.anchor_name, "kind": anchor.kind, "role": "main"}
            try:
                try_turn.data.setdefault("$try_out", {}).update(try_meta)
                main_turn.data.setdefault("$try_out", {}).update(main_meta)
            except Exception:
                pass
        # Advance current cursor to the new main placeholder
        self._assign_head(main_turn)
        main_cursor = self

        trial_cursor = self.clone()
        trial_cursor._assign_head(try_turn)
        trial_cursor.is_fork = True
        trial_cursor.origin_user_turn_index = self.real_user_turns()
        trial_cursor.context_id = None
        context_for_branch = self.context
        if context_for_branch:
            trial_cursor = context_for_branch.adopt_cursor(trial_cursor, make_active=False, scope=self.scope)
            if anchor:
                anchor.try_out_turns.append(try_turn)
                if trial_cursor.context_id:
                    anchor.try_out_cursor_ids.append(trial_cursor.context_id)
        return main_cursor, trial_cursor

    def close_branch(
        self,
        *,
        keep_in_main: bool = False,
        the_only_main: bool = False,
    ) -> bool:
        """
        Close a temporary try-out branch created via add_try_out and return the
        cursor that should remain active (typically the original main cursor).
        """
        branch_root = self.head
        parent = branch_root.parent if branch_root else None
        success = self._close_branch_node()

        if keep_in_main and branch_root:
            branch_root.main_thread = True
        if the_only_main and parent and branch_root:
            try:
                self.session.promote_tryouts_to_main(parent, [branch_root])
            except Exception:
                pass

        return success

    def add_update_metrics(self, metrics: Dict[str, Any], *, skip_none: bool = True) -> Dict[str, Any]:
        """Attach fresh metrics to the cursor head."""
        payload = {k: v for k, v in metrics.items() if not skip_none or v is not None}
        if payload:
            self.session.add_update_metrics(self.head, payload)
        return payload

    _PER_ITEM_METRIC_FIELDS: Tuple[str, ...] = (
        "input_tokens",
        "output_tokens",
        "generation_duration_sec",
        "time_to_first_token_sec",
        "tokens_per_second",
        "cache_metric",
        "cache_warming",
        "was_truncated",
        "tool_blocks_count",
        "tool_blocks_tokens",
    )
    _AGGREGATE_METRIC_FIELDS: Tuple[str, ...] = (
        "total_input_tokens",
        "total_output_tokens",
        "total_generation_duration_sec",
        "overall_tps",
        "avg_time_to_first_token_sec",
        "total_tool_blocks",
        "total_tool_blocks_tokens",
        "cache_queued",
        "in_flight_req",
        "mem_allocated",
        "mem_reserved",
    )
    _KNOWN_METRIC_FIELDS: Tuple[str, ...] = _PER_ITEM_METRIC_FIELDS + _AGGREGATE_METRIC_FIELDS + ("total_prompts_processed",)

    @staticmethod
    def update_response_metrics(
        response: Optional["InferenceResponse"] = None,
        *,
        metrics: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build/update the response metrics dict, dropping None values."""
        payload: Dict[str, Any] = {}
        if response is not None:
            for field in ChatCursor._PER_ITEM_METRIC_FIELDS + ChatCursor._AGGREGATE_METRIC_FIELDS:
                value = getattr(response, field, None)
                if value is not None:
                    payload[field] = value
        if metrics:
            for key in ChatCursor._KNOWN_METRIC_FIELDS:
                if key in metrics:
                    value = metrics.get(key)
                    if value is not None:
                        payload[key] = value
        if extra:
            for key, value in extra.items():
                if value is not None:
                    payload[key] = value
        return payload

    @staticmethod
    def build_chunk_metrics(
        chunk_data: Mapping[str, Any],
        *,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tracked_fields = [
            "input_tokens",
            "output_tokens",
            "generation_duration_sec",
            "time_to_first_token_sec",
            "tokens_per_second",
            "cache_metric",
            "cache_warming",
            "was_truncated",
            "tool_blocks_count",
            "tool_blocks_tokens",
            "total_input_tokens",
            "total_output_tokens",
            "total_generation_duration_sec",
            "overall_tps",
            "avg_time_to_first_token_sec",
            "total_prompts_processed",
            "total_tool_blocks",
            "total_tool_blocks_tokens",
            "cache_queued",
            "in_flight_req",
            "mem_allocated",
            "mem_reserved",
        ]
        metrics = {field: chunk_data.get(field) for field in tracked_fields if field in chunk_data}
        if extra:
            metrics.update(extra)
        return {k: v for k, v in metrics.items() if v is not None}

    # ---- batching / forking ----------------------------------------------
    def add_batch(
        self,
        prompts: Sequence[str],
        *,
        command_text: str = "/g b",
        override_adapters: Optional[Sequence[str]] = None,
        adopt_into_context: bool = True,
        make_active: bool = True,
        **extra: Any,
    ) -> "ChatForks":
        """Create a batch fork and optionally adopt/activate cursors in the context."""
        batch_hub, chat_turns = self.session.add_batch_turn(
            command_text=command_text,
            prompts=list(prompts),
            current_turn=self.head,
            override_adapters=list(override_adapters) if override_adapters else None,
            **extra,
        )
        hub_cursor = self.clone_at(batch_hub)

        # build child cursors that inherit the current mainline user count
        origin_count = self.real_user_turns()
        cursors: List[ChatCursor] = []
        for t in chat_turns:
            child = self.clone()
            child._assign_head(t)
            child.is_fork = True
            child.origin_user_turn_index = origin_count
            child.context_id = None
            child.bind_context(self.context)
            cursors.append(child)
        chosen = cursors[0]
        fork = ChatForks(parent=self, cursors=cursors, main_cursor=chosen, batch_hub=hub_cursor)
        fork.active_cursors = list(cursors)
        fork.main_placeholders = [None] * len(cursors)
        self.forks.append(fork)

        # push a batch frame so _close_all_batches() can find its hub without params
        self._stack.append({
            "kind": "batch",
            "cursor": self.clone(),
            "batch_hub": hub_cursor,
        })

        if adopt_into_context and self.context:
            for idx, child in enumerate(fork.cursors, start=1):
                alias = f"{self.label}:fork{idx}"
                self.context.adopt_cursor(child, alias=alias, make_active=False, scope=self.scope)
            # Only activate the main cursor if this batch sits on the main branch.
            if make_active and fork.main_cursor and getattr(hub_cursor.current_turn, "main_thread", False):
                self.context.set_active_cursor(fork.main_cursor, scope=self.scope)

        return fork

    def close_batches(
        self,
        *,
        make_active: bool = True,
    ) -> Optional["ChatCursor"]:
        closed = self._close_all_batches()
        if not closed:
            return None
        # Reassign this cursor's head; do not create a new tracked cursor.
        self._assign_head(closed.head)
        if self.context and make_active:
            try:
                self.context.set_active_cursor(self, scope=self.scope)
            except Exception:
                pass
        return self

    def resolve_batch_main_placeholder(
        self,
        *,
        hub_cursor: Optional["ChatCursor"] = None,
    ) -> Optional["Turn"]:
        """Resolve the main-branch placeholder for the most recent (or specified) batch without popping stack frames."""
        hub_turn: Optional[Turn] = None
        if hub_cursor:
            hub_turn = hub_cursor.current_turn
        else:
            for idx in range(len(self._stack) - 1, -1, -1):
                frame = self._stack[idx]
                if frame.get("kind") != "batch":
                    continue
                frame_cursor = frame.get("batch_hub")
                if frame_cursor:
                    hub_turn = frame_cursor.current_turn
                break
        if not hub_turn:
            return None
        return self.session.close_all_batches(hub_turn)

    def close_batches_by_gen_id(
        self,
        hub_gen_id: str,
        *,
        make_active: bool = True,
    ) -> Optional["ChatCursor"]:
        """Close a specific batch frame by hub gen_id using this cursor's stack."""
        if not hub_gen_id:
            return None
        matched_idx = None
        hub_cursor: Optional[ChatCursor] = None
        for idx in range(len(self._stack) - 1, -1, -1):
            frame = self._stack[idx]
            if frame.get("kind") != "batch":
                continue
            frame_cursor = frame.get("batch_hub")
            frame_turn = frame_cursor.current_turn if frame_cursor else None
            if frame_turn and getattr(frame_turn, "gen_id", None) == hub_gen_id:
                matched_idx = idx
                hub_cursor = frame_cursor
                break
        if matched_idx is None or not hub_cursor:
            return None
        self._stack.pop(matched_idx)
        placeholder = self.session.close_all_batches(hub_cursor.current_turn)
        if placeholder is None:
            return None
        self._assign_head(placeholder)
        if self.context and make_active:
            try:
                self.context.set_active_cursor(self, scope=self.scope)
            except Exception:
                pass
        return self

    def assign_as_main(self) -> None:
        # Promote this branch to main_thread and record the intent for replay.
        self.session.promote_tryouts_to_main(self.head.parent, [self.head])
        self.session.add_state_change(
            self.head,
            "main_thread",
            True,
            command_text="assign_as_main",
        )

    # ---- closing batches (stack-driven) -----------------------------------
    def _close_all_batches(self) -> Optional["ChatCursor"]:
        """Close the batches associated with the most-recent batch frame on the stack.
        Pops frames until it finds a 'batch' frame and uses its hub with session.close_all_batches.
        Returns a cursor positioned at the resulting main placeholder.
        """
        # Walk backwards to find last batch origin
        idx = len(self._stack) - 1
        found_idx = None
        while idx >= 0:
            if self._stack[idx].get("kind") == "batch":
                found_idx = idx
                break
            idx -= 1
        if found_idx is None:
            return None

        frame = self._stack.pop(found_idx)
        hub_cursor = frame.get("batch_hub")
        hub_turn = hub_cursor.current_turn if hub_cursor else None
        placeholder = self.session.close_all_batches(hub_turn)  # hub provided by session
        if placeholder is None:
            return None
        restored = self.clone()
        restored._assign_head(placeholder)
        restored.is_fork = False
        restored.context_id = None
        restored.bind_context(self.context)
        return restored

    # ---- engine calls & replay -------------------------------------------
    def save_api_command(
        self,
        api_name: str,
        params: Dict[str, Any],
        *,
        command_text: str,
        response: Optional[Any] = None,
    ) -> Command:
        new_turn = self.session.add_api_command(self.head, api_name, params, command_text)
        self._assign_head(new_turn)
        cmd = new_turn.cmd[-1]
        if response is not None:
            if isinstance(response, dict) and response.get("status") not in (None, "success"):
                cmd.data["$Error"] = {
                    "message": response.get("message"),
                    "details": copy.deepcopy(response.get("details")),
                }
            elif isinstance(response, dict) and response.get("status") == "success":
                cmd.data["$Response"] = copy.deepcopy(response.get("data"))
            else:
                cmd.data["$Error"] = {"message": str(response)}
        return cmd
    
    def save_log_command(self, command_text: str, *, metadata: Optional[Dict[str, Any]] = None) -> Command:
        new_turn = self.session.add_log_command(self.head, command_text, metadata)
        self._assign_head(new_turn)
        return new_turn.cmd[-1]

    def save_state_command(self, change_type: str, value: Any,  command_text: Optional[str] = None) -> Command:
        new_turn = self.session.add_state_change(self.head, change_type, value, command_text) 
        self._assign_head(new_turn)
        return new_turn.cmd[-1]

    def save_param_command(self, change_type: str, new_value: Any,  command_text: Optional[str] = None) -> Command:
        new_turn = self.session.add_param_change(self.head, change_type, new_value, command_text)
        self._assign_head(new_turn)
        return new_turn.cmd[-1]

    # ---- cloning ----------------------------------------------------------
    def clone(self) -> "ChatCursor":
        if not self.context:
            raise RuntimeError("ChatCursor.clone requires an associated ChatContext.")
        c = ChatCursor(
            self.context,
            self.head,
            scope=self.scope,
            origin_user_turn_index=self.origin_user_turn_index,
            is_fork=self.is_fork,
            gen_config=dict(self.gen_config),
            streaming=self.stream_override,
            cache=self.cache_override,
            return_prompt=self.return_prompt_override,
            adapter_override_for_next_turn=(list(self.adapter_override_for_next_turn)
                                            if self.adapter_override_for_next_turn else None),
        )
        c._tools_view_override = self._tools_view_override
        c.generation_config_template_override = (
            copy.deepcopy(self.generation_config_template_override)
            if self.generation_config_template_override is not None
            else None
        )
        c.max_new_tokens_override = self.max_new_tokens_override
        c.no_tools_parse_override = self.no_tools_parse_override
        c.auto_retry_truncated_override = self.auto_retry_truncated_override
        c.suppress_full_response_override = self.suppress_full_response_override
        c.reset_metrics_override = self.reset_metrics_override
        # Do not copy _stack; clones start with a fresh fork stack by design
        c.context_id = None
        return c

    def get_effective_tools_scopes(self) -> List[ToolsScope]:
        target = self.head
        return self.session.get_effective_tools_scopes(target)

    def get_tools_access(
        self,
        *,
        label: Optional[str] = None,
    ) -> Optional[ToolsAccess]:
        toolbox = self.toolbox
        target = self.head
        if not toolbox or target is None:
            return None
        resolved_label = (
            label
            or getattr(target, "gen_id", None)
            or getattr(target, "gen_id_or_parent", None)
        )
        return self.session.get_tools_access(
            toolbox,
            target,
            label=resolved_label,
        )

    def get_tools_view(
        self,
        *,
        label: Optional[str] = None,
    ) -> Optional[ToolsView]:
        access = self.get_tools_access(label=label)
        if not access:
            return None
        view = access.get_view()
        self.set_tools_view(view)
        return view

    def get_active_tools(self, view: Optional[ToolsView] = None) -> Optional[Any]:
        toolbox = self.toolbox
        if not toolbox:
            return None
        active_view = view or self.get_tools_view()
        if active_view:
            return toolbox.get_tools_for_inference(active_view)
        return toolbox.get_tools_for_inference()

    def active_path(self) -> List[Turn]:
        """Return the active spine ending at the cursor's head."""
        return list(self.session.get_active_path(self.head))

    def active_path_for_llm(self) -> List[Turn]:
        """Return the spine used for LLM formatting for the cursor's head."""
        return list(self.session.get_active_path_for_llm(self.head))

    def descend_to_leaf(self) -> "ChatCursor":
        """Move the cursor to the leaf of its current branch."""
        leaf = self.session.get_last_turn_on_branch(self.head)
        if leaf:
            self._assign_head(leaf)
            self._mark_active()
        return self

    def promote_tryouts(self, promote_list: Sequence[Turn], *, parent: Optional[Turn] = None) -> List[Turn]:
        target_parent = parent or self.head
        return self.session.promote_tryouts_to_main(parent=target_parent, promote_list=list(promote_list))

    def refresh_tools_view(self, *, label: Optional[str] = None) -> Optional[ToolsView]:
        if not self.context or not self.context.toolbox:
            return None
        access = self.session.get_tools_access(
            self.context.toolbox,
            self.head,
            label=label or (self.head.gen_id if self.head.gen_id else None),
        )
        view = access.get_view()
        self.set_tools_view(view)
        return view

    def _cursor_inference_defaults(self) -> InferenceParams:
        if self.context:
            return self.context.inference_defaults
        raise ValueError("Orphaned cursor instance")

    def _resolve_streaming_flag(self) -> bool:
        if self.context:
            return self.context._resolve_streaming_flag()
        return bool(self._cursor_inference_defaults().stream)

    def _base_flag_settings(self) -> Dict[str, Any]:
        params = self.chat_session.initial_params if self.chat_session else {}
        generation_template = params.get("generation_config_template")
        defaults = self._cursor_inference_defaults()
        if generation_template is None:
            generation_template = copy.deepcopy(defaults.generation_config or {})
        max_new_seed = params.get("max_new_tokens")
        if max_new_seed is None:
            max_new_seed = params.get("max_new_tokens_override")
        return {
            "stream": params.get("stream", params.get("streaming", self._resolve_streaming_flag())),
            "cache": params.get("cache_override"),
            "return_prompt": params.get("return_prompt_mode"),
            "generation_config_template": copy.deepcopy(generation_template),
            "max_new_tokens_override": max_new_seed,
            "no_tools_parse": bool(params.get("no_tools_parse", False)),
            "auto_retry_truncated": bool(params.get("auto_retry_truncated", False)),
            "suppress_full_response": bool(params.get("suppress_full_response", False)),
        }

    def _collect_flag_settings(self, defaults: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(defaults)
        if not self.head:
            return result
        path: List[Turn] = self.session.get_active_path_for_llm(self.head)
        for turn in path:
            for cmd in turn.cmd:
                change = cmd.data.get("change")
                if not change:
                    continue
                if change not in CHAT_FLAG_FIELDS:
                    continue
                if cmd.cmd_type == Command.STATE_CHANGE:
                    value = copy.deepcopy(cmd.data.get("value"))
                elif cmd.cmd_type == Command.PARAM_CHANGE:
                    value = copy.deepcopy(cmd.data.get("new_value"))
                else:
                    continue
                result[change] = value
        return result

    def resolve_flag_settings(self) -> Dict[str, Any]:
        defaults = self._base_flag_settings()
        return self._collect_flag_settings(defaults)

    def build_inference_request(
        self,
        *,
        batch: Optional[Sequence["ChatCursor"]] = None,
        request_id_prefix: str = "",
        overrides: Optional[InferenceParams] = None,
        manual_continue: bool = False,
        include_tools: bool = True,
    ) -> Tuple[Dict[str, Any], List[str], Optional[ToolsView]]:
        cursors = list(batch) if batch else [self]
        turns = [c.current_turn for c in cursors if c.current_turn]
        if not turns:
            raise ValueError("build_inference_request requires at least one valid turn.")

        session = self.session
        chat_session = self.chat_session or (self.context.chat_session if self.context else None)
        if not chat_session:
            raise ValueError("build_inference_request requires live context.")

        parser_profile = chat_session.parser_profile if chat_session else None
        parser = UnifiedToolIO(parser_profile) if parser_profile else None
        messages_list: List[List[Dict[str, Any]]] = []
        adapter_sets: List[Tuple[str, ...]] = []
        has_adapter_override = False

        for cursor in cursors:
            turn = cursor.current_turn
            messages = session.get_llm_messages(
                current_turn=turn,
                parser=parser,
            )
            messages_list.append(messages)
            adapter_sets.append(tuple(session.get_effective_adapters(turn)))
            if turn:
                override_name = (turn.data or {}).get("adapter_override_name")
                if override_name is not None:
                    has_adapter_override = True

        selected_adapters: List[str] = list(adapter_sets[0]) if adapter_sets and adapter_sets[0] else ["__base__"]
        if not has_adapter_override and len({aset for aset in adapter_sets}) > 1:
            first_set = adapter_sets[0]
            conflict_idx = next(
                (idx for idx, aset in enumerate(adapter_sets) if aset != first_set),
                None,
            )
            conflict_set = adapter_sets[conflict_idx] if conflict_idx is not None else ()
            first_display = list(first_set) if first_set else ["__base__"]
            conflict_display = list(conflict_set) if conflict_set else ["__base__"]
            raise ValueError(
                "Mixed adapter states detected in this request; "
                f"first prompt adapters={first_display}, "
                f"conflicting adapters at index {conflict_idx}={conflict_display}."
            )

        flag_settings = self._compose_request_flags()
        inference_payload = self._materialize_request_params(flag_settings)

        inference_defaults = chat_session.inference_defaults if chat_session else None
        if overrides:
            serialized_overrides = overrides.serialize(defaults=inference_defaults)
            inference_payload.update(serialized_overrides)

        first_turn = turns[0]
        inference_payload["request_id"] = session.get_request_id(first_turn, request_id_prefix) if first_turn else str(uuid.uuid4())
        inference_payload["messages_list"] = messages_list
        inference_payload["active_adapters"] = selected_adapters

        tools_view: Optional[ToolsView] = None
        if include_tools:
            primary_cursor = cursors[0]
            tools_view = primary_cursor.get_tools_view()
            if tools_view is None:
                tools_view = primary_cursor.refresh_tools_view()
            active_tools = primary_cursor.get_active_tools(tools_view)
            filtered_tools = self._filter_tools_payload(active_tools, tools_view)
            if filtered_tools:
                inference_payload["tools"] = filtered_tools

        if manual_continue or any(turn.do_continue for turn in turns if turn):
            inference_payload["do_continue"] = True

        self._record_request_metadata(turns[0], inference_payload)

        return inference_payload, selected_adapters, tools_view

    def _compose_request_flags(self) -> Dict[str, Any]:
        base_flags: Dict[str, Any] = {}
        if self.context:
            base_flags = copy.deepcopy(self.resolve_flag_settings())
        if self.stream_override is not None:
            base_flags["stream"] = self.stream_override
        if self.return_prompt_override is not None:
            base_flags["return_prompt"] = self.return_prompt_override
        if self.cache_override is not None:
            base_flags["cache"] = self.cache_override
        if self.gen_config:
            base_flags["generation_config"] = copy.deepcopy(self.gen_config)
        if self.generation_config_template_override is not None:
            base_flags["generation_config_template"] = copy.deepcopy(self.generation_config_template_override)
        if self.max_new_tokens_override is not None:
            base_flags["max_new_tokens_override"] = self.max_new_tokens_override
        if self.no_tools_parse_override is not None:
            base_flags["no_tools_parse"] = bool(self.no_tools_parse_override)
        if self.auto_retry_truncated_override is not None:
            base_flags["auto_retry_truncated"] = bool(self.auto_retry_truncated_override)
        if self.suppress_full_response_override is not None:
            base_flags["suppress_full_response"] = bool(self.suppress_full_response_override)
        if self.reset_metrics_override:
            base_flags["reset_metrics"] = True
        return base_flags

    def _materialize_request_params(self, flags: Dict[str, Any]) -> Dict[str, Any]:
        request_params: Dict[str, Any] = {}

        generation_template = copy.deepcopy(flags.get("generation_config_template") or {})
        generation_overrides = copy.deepcopy(flags.get("generation_config") or {})

        max_new_tokens = flags.get("max_new_tokens_override")
        if max_new_tokens is not None:
            generation_overrides["max_new_tokens"] = max_new_tokens

        if generation_template or generation_overrides:
            generation_template.update(generation_overrides)
            if generation_template:
                request_params["generation_config"] = generation_template

        stream_value = flags.get("stream")
        request_params["stream"] = bool(stream_value) if stream_value is not None else True

        if flags.get("return_prompt") is not None:
            request_params["return_prompt"] = flags.get("return_prompt")

        cache_value = flags.get("cache")
        if cache_value not in (None, "", [], {}):
            request_params["cache"] = cache_value

        if bool(flags.get("suppress_full_response")):
            request_params["suppress_full_response"] = True

        if bool(flags.get("no_tools_parse")):
            request_params["no_tools_parse"] = True

        if flags.get("reset_metrics"):
            request_params["reset_metrics"] = True

        return request_params

    def _filter_tools_payload(
        self,
        tools_payload: Optional[Dict[str, Any]],
        tools_view: Optional[ToolsView],
    ) -> Optional[Dict[str, Any]]:
        if not tools_payload:
            return None
        if not tools_view or not tools_view.advertised_tools:
            return None
        allowed = set(tools_view.advertised_tools)
        if not allowed:
            return None
        filtered: List[Dict[str, Any]] = []
        for entry in tools_payload.get("for_dump", []):
            name = entry.get("function", {}).get("name") or entry.get("name")
            if name and name in allowed:
                filtered.append(entry)
        if not filtered:
            return None
        return {"for_dump": filtered}

    def _record_request_metadata(self, turn: Optional[Turn], payload: Dict[str, Any]) -> None:
        if not turn:
            return
        if turn.data is None:
            turn.data = {}
        sanitized_payload = copy.deepcopy(payload)
        sanitized_payload.pop("messages_list", None)

        tools_payload = sanitized_payload.get("tools")
        tool_names: List[str] = []
        if isinstance(tools_payload, dict):
            for entry in tools_payload.get("for_dump", []):
                if not isinstance(entry, dict):
                    continue
                name = entry.get("function", {}).get("name") if isinstance(entry.get("function"), dict) else None
                if not name:
                    name = entry.get("name")
                if name:
                    tool_names.append(name)
        elif isinstance(tools_payload, list):
            for entry in tools_payload:
                if isinstance(entry, dict):
                    name = entry.get("function", {}).get("name") if isinstance(entry.get("function"), dict) else None
                    if not name:
                        name = entry.get("name")
                    if name:
                        tool_names.append(name)
                elif isinstance(entry, str):
                    tool_names.append(entry)
        if tool_names:
            sanitized_payload["tools"] = tool_names
        else:
            sanitized_payload.pop("tools", None)

        turn.data["$RequestParams"] = sanitized_payload

    def record_adapters(
        self,
        adapters: Sequence[str],
    ) -> None:
        if not adapters:
            return
        target = self.head
        if target is None:
            return
        if target.data is None:
            target.data = {}
        target.data["$Adapters"] = list(adapters)

    def record_error(
        self,
        message: Optional[str],
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Attach engine error metadata to the given turn."""
        if not message:
            return
        target_turn = self.head
        if target_turn is None:
            return
        if target_turn.data is None:
            target_turn.data = {}
        payload: Dict[str, Any] = {"message": message}
        if details:
            payload["details"] = copy.deepcopy(details)
        target_turn.data["$Error"] = payload

    def iter_spine_tree(
        self,
        *,
        include_forks: bool = True,
        include_archived: bool = True,
        move_main_child_last_on_fork: bool = True,
        limit_to_active_branch: bool = False,
        detours_first: bool = False,
    ) -> Iterable["ChatCursor.WalkItem"]:
        """
        Yield cursor-aware walk items mirroring EngineSession.iter_spine_tree.

        Consumers should iterate once and immediately process each WalkItem;
        both `cursor` and `context_cursor` are clones of the active cursor and
        should be treated as short-lived helpers for rendering labels or
        summaries. They are intentionally detached from the global cursor so
        iteration cannot accidentally move the interactive head.
        """
        if not self.context:
            raise RuntimeError("ChatCursor.iter_spine_tree requires a ChatContext.")
        session_items = self.session.iter_spine_tree_snapshot(
            active=self.head,
            include_forks=include_forks,
            include_archived=include_archived,
            move_main_child_last_on_fork=move_main_child_last_on_fork,
            limit_to_active_branch=limit_to_active_branch,
            detours_first=detours_first,
        )
        for item in session_items:
            node_cursor = self.clone_at(item.node)
            ctx_cursor = (
                self.clone_at(item.context_for_labels)
                if getattr(item, "context_for_labels", None)
                else None
            )
            yield ChatCursor.WalkItem(
                cursor=node_cursor,
                depth=item.depth,
                relation=item.relation,
                is_last_sibling=item.is_last_sibling,
                has_forks=item.has_forks,
                context_cursor=ctx_cursor,
                is_peek=getattr(item, "is_peek", False),
                peek_kind=getattr(item, "peek_kind", None),
                collected=getattr(item, "collected", False),
            )

    def summarize_turn(
        self,
        target: Optional[Union["ChatCursor", Turn]] = None,
        *,
        max_len: int = 80,
    ) -> str:
        """Proxy to EngineSession.summarize_turn accepting cursors or raw turns."""
        if target is None:
            node = self.head
        elif isinstance(target, ChatCursor):
            node = target.head
        else:
            node = target
        if node is None:
            raise ValueError("summarize_turn requires a valid target turn.")
        return self.session.summarize_turn(node, active=self.head, max_len=max_len)


# ------------------------------ ChatForks ----------------------------------
@dataclass
class ChatForks:
    """Tracker for a single batch fork.

    - `cursors`: per-prompt ChatCursor list.
    - `main_cursor`: designated inline branch (can be reassigned or inlined).
    - `batch_hub`: cursor pinned to the FORK/BATCH hub from session.add_batch_turn.
    - `completed_ids`: track completion per cursor head id.
    """
    parent: ChatCursor
    cursors: List[ChatCursor]
    main_cursor: Optional[ChatCursor] = None
    batch_hub: Optional[ChatCursor] = None
    completed_ids: set = field(default_factory=set)
    anchor_id: Optional[str] = None
    cursor_meta: Optional[Dict[str, Dict[str, Any]]] = None
    prompt_indices: List[int] = field(default_factory=list)
    active_cursor: Optional[ChatCursor] = None
    active_cursors: List[Optional[ChatCursor]] = field(default_factory=list)
    main_placeholders: List[Optional[ChatCursor]] = field(default_factory=list)

    def inline_main(self) -> None:
        if self.main_cursor is not None:
            self.main_cursor.assign_as_main()

    def update_cursor(self, idx: int, cursor: ChatCursor) -> None:
        if idx < 0 or idx >= len(self.cursors):
            return
        self.cursors[idx] = cursor
        if idx == 0 and self.main_cursor is not None:
            self.main_cursor = cursor

    def update_active_cursor(self, idx: int, cursor: ChatCursor) -> None:
        if idx < 0:
            return
        if idx >= len(self.active_cursors):
            self.active_cursors.extend([None] * (idx + 1 - len(self.active_cursors)))
        self.active_cursors[idx] = cursor
        self.update_cursor(idx, cursor)

    def set_main_placeholder(self, idx: int, cursor: ChatCursor) -> None:
        if idx < 0:
            return
        if idx >= len(self.main_placeholders):
            self.main_placeholders.extend([None] * (idx + 1 - len(self.main_placeholders)))
        self.main_placeholders[idx] = cursor

    def get_active_cursor(self, idx: int) -> Optional[ChatCursor]:
        if idx < 0 or idx >= len(self.active_cursors):
            return None
        return self.active_cursors[idx]

    def drop_cursors(self) -> None:
        self.cursors = []
        self.completed_ids.clear()
        self.active_cursor = None
        self.active_cursors = []
        self.main_placeholders = []

    def complete(self, cursor: ChatCursor) -> None:
        self.completed_ids.add(cursor.id)

    def all_done(self) -> bool:
        return len(self.completed_ids) == len(self.cursors)


# ------------------------------ ChatContext ---------------------------------
class ChatContext:
    """
    High-level orchestrator that keeps EngineSession/ChatSession state in sync
    with multiple ChatCursor objects. The chat CLI (app.mp13chat) can track
    branches, temporary try-outs, and concurrent batches simply by holding on
    to ChatCursor handles that this context owns.
    """

    def __init__(
        self,
        session: EngineSession,
        chat_session: Optional[ChatSession] = None,
        *,
        conversation_index: int = 0,
        toolbox: Optional[Toolbox] = None,
        request_overrides: Optional[InferenceParams] = None,
    ) -> None:
        self.session = session
        self.chat_session = chat_session or self._resolve_chat_session(conversation_index)
        self.root_turn: Turn = self.chat_session.root_turn
        if toolbox:
            if self.chat_session.toolbox is None or self.chat_session.toolbox is not toolbox:
                self.chat_session.toolbox = toolbox
        self.toolbox_ref: Optional[ChatContextToolBoxRef] = None
        if self.chat_session.toolbox:
            self.toolbox_ref = ChatContextToolBoxRef(self, self.chat_session.toolbox)
        self.inference_defaults: InferenceParams = self.chat_session.inference_defaults or InferenceParams()
        self.request_overrides: InferenceParams = request_overrides or InferenceParams()

        self._cursor_seq: int = 0
        self._scope_seq: int = 0
        self._rw_lock = ReentrantWriterFairRWLock()
        self._scopes: Dict[str, ChatContextScope] = {}
        self._cursors: Dict[str, ChatCursor] = {}
        self._active_cursor_id: Optional[str] = None
        self._try_out_anchors: Dict[str, TryOutAnchor] = {}
        self._default_scope: ChatContextScope = ChatContextScope(
            context=self,
            scope_id="default",
            label="default",
            auto_mark_active=True,
        )
        self._scopes[self._default_scope.scope_id] = self._default_scope

        root_cursor = self._create_root_cursor()
        self._register_cursor(root_cursor, alias="main", make_active=True, scope=self._default_scope)

    @property
    def toolbox(self) -> Optional[Toolbox]:
        if self.toolbox_ref:
            return self.toolbox_ref.toolbox
        return None

    @property
    def default_scope(self) -> ChatContextScope:
        return self._default_scope

    @property
    def bag_dict(self) -> Dict[str, Any]:
        return self._default_scope.bag_dict

    @property
    def pending_auto_iterations(self) -> int:
        return self._default_scope.pending_auto_iterations

    @pending_auto_iterations.setter
    def pending_auto_iterations(self, value: int) -> None:
        self._default_scope.pending_auto_iterations = int(value)

    def create_scope(
        self,
        *,
        label: Optional[str] = None,
        auto_mark_active: bool = True,
        make_default: bool = False,
    ) -> ChatContextScope:
        self._scope_seq += 1
        scope_id = f"scope_{self._scope_seq}"
        scope = ChatContextScope(
            context=self,
            scope_id=scope_id,
            label=label,
            auto_mark_active=auto_mark_active,
        )
        with self._rw_lock.write_lock():
            self._scopes[scope_id] = scope
            if make_default:
                self._default_scope = scope
        return scope

    def get_scope(self, scope: Optional[ChatContextScope]) -> ChatContextScope:
        return scope or self._default_scope

    def dispose_scope(self, scope: ChatContextScope, *, drop_cursors: bool = False) -> None:
        if not scope or scope.is_disposed:
            return
        with self._rw_lock.write_lock():
            for name, anchor in list(self._try_out_anchors.items()):
                if anchor.owner_scope is scope:
                    self._try_out_anchors.pop(name, None)
            if drop_cursors:
                for handle, cursor in list(self._cursors.items()):
                    if getattr(cursor, "scope", None) is scope:
                        self._cursors.pop(handle, None)
            self._scopes.pop(scope.scope_id, None)
            scope.is_disposed = True

    def request_auto_iteration(self, count: int = 1, *, scope: Optional[ChatContextScope] = None) -> None:
        if count <= 0:
            return
        target_scope = self.get_scope(scope)
        target_scope.pending_auto_iterations += count

    def consume_auto_iteration(self, *, scope: Optional[ChatContextScope] = None) -> bool:
        target_scope = self.get_scope(scope)
        if target_scope.pending_auto_iterations <= 0:
            return False
        target_scope.pending_auto_iterations -= 1
        return True

    # ---- cursor helpers ----------------------------------------------------
    def _resolve_chat_session(self, conversation_index: int) -> ChatSession:
        if self.session.conversations:
            if 0 <= conversation_index < len(self.session.conversations):
                return self.session.get_conversation(conversation_index)
            return self.session.conversations[0]
        return self.session.add_conversation()

    def _create_root_cursor(self) -> ChatCursor:
        cursor = ChatCursor(
            self,
            self.root_turn,
            scope=self._default_scope,
            gen_config=self._resolve_generation_config(),
            streaming=None,
            cache=None,
            return_prompt=None,
            adapter_override_for_next_turn=self._resolve_adapter_override(),
        )
        return cursor

    def _relative_path_from_root(self, target_turn: Optional[Turn]) -> Optional[List[Turn]]:
        if not target_turn:
            return None
        path = self.session.get_active_path(target_turn)
        if not path:
            return None
        try:
            start_idx = path.index(self.root_turn)
        except ValueError:
            return None
        return path[start_idx:]

    def owns_turn(self, target_turn: Optional[Turn]) -> bool:
        """Return True when `target_turn` belongs to this context's subtree."""
        return self._relative_path_from_root(target_turn) is not None

    def _ensure_turn_in_scope(self, target_turn: Turn) -> None:
        if not self.owns_turn(target_turn):
            label = getattr(target_turn, "gen_id_or_parent", "unknown")
            raise ValueError(f"Turn '{label}' is not managed by this ChatContext.")

    def _drop_tracked_items_for_trim(self, trimmed_turns: Set[Turn], *, keep_handles: Optional[Set[str]] = None) -> None:
        if not trimmed_turns:
            return
        keep_handles = keep_handles or set()

        with self._rw_lock.read_lock():
            cursor_items = list(self._cursors.items())
            anchor_items = list(self._try_out_anchors.items())

        drop_cursors: List[ChatCursor] = []
        cursor_by_handle = {handle: cursor for handle, cursor in cursor_items}
        for handle, cursor in cursor_items:
            if handle in keep_handles:
                continue
            if cursor.head in trimmed_turns:
                drop_cursors.append(cursor)

        for cursor in drop_cursors:
            try:
                self.drop_cursor(cursor)
            except Exception:
                pass

        for anchor_name, anchor in anchor_items:
            anchor_turn = anchor.anchor_turn
            try_out_turns = list(anchor.try_out_turns or [])
            if anchor_turn in trimmed_turns or any(t in trimmed_turns for t in try_out_turns):
                for handle in list(anchor.try_out_cursor_ids or []):
                    if handle in keep_handles:
                        continue
                    cursor = cursor_by_handle.get(handle)
                    if cursor:
                        try:
                            self.drop_cursor(cursor)
                        except Exception:
                            pass
                with self._rw_lock.write_lock():
                    self._try_out_anchors.pop(anchor_name, None)

    def _is_context_scope_command(self, cmd: Command) -> bool:
        return (
            cmd.cmd_type == Command.STATE_CHANGE
            and cmd.data.get("change") == "tools_scope"
            and bool(cmd.data.get("context_scope"))
        )

    def _load_context_tools_scope(self) -> ToolsScope:
        root_cmds = list(getattr(self.root_turn, "cmd", []) or [])
        context_cmds = [cmd for cmd in root_cmds if self._is_context_scope_command(cmd)]
        if not context_cmds:
            return ToolsScope()
        scope_payload = context_cmds[-1].data.get("scope")
        if scope_payload:
            try:
                return ToolsScope.from_dict(scope_payload)
            except Exception:
                pass
        return ToolsScope()

    def _persist_context_tools_scope(self, scope: ToolsScope) -> None:
        if not self.root_turn:
            return
        def _match(cmd: Command) -> bool:
            return self._is_context_scope_command(cmd)
        def _is_echo(cmd: Command) -> bool:
            if not cmd or cmd.cmd_type != Command.COMMAND:
                return False
            if cmd.data.get("$Action") == "echo":
                return True
            command_text = cmd.data.get("command", "") or ""
            return command_text.strip().startswith("/echo")
        payload = {
            "change": "tools_scope",
            "op": "set",
            "scope": scope.to_dict(),
            "context_scope": True,
        }
        new_cmd = Command(cmd_type=Command.STATE_CHANGE, metadata=payload)
        commands = list(getattr(self.root_turn, "cmd", []) or [])
        insert_at = 1 if commands and _is_echo(commands[0]) else 0
        self.session.replace_command_in_turn(self.root_turn, _match, new_cmd, insert_at=insert_at)

    def _resolve_generation_config(self) -> Dict[str, Any]:
        if self.request_overrides.generation_config:
            gen_config = copy.deepcopy(self.request_overrides.generation_config)
        elif self.inference_defaults.generation_config:
            gen_config = copy.deepcopy(self.inference_defaults.generation_config)
        else:
            gen_config = {}
        return gen_config

    def _resolve_streaming_flag(self) -> bool:
        snapshot = self._context_flag_snapshot()
        value = snapshot.get("stream")
        if value is None:
            return self._default_stream_flag()
        return bool(value)

    def _resolve_adapter_override(self) -> Optional[List[str]]:
        if self.request_overrides.override_adapters:
            return list(self.request_overrides.override_adapters)
        return None

    def _initial_active_adapters(self) -> Optional[List[str]]:
        if self.request_overrides.active_adapters:
            return list(self.request_overrides.active_adapters)
        if self.inference_defaults.active_adapters:
            return list(self.inference_defaults.active_adapters)
        return None

    def _default_generation_template(self) -> Dict[str, Any]:
        if self.request_overrides.generation_config:
            return copy.deepcopy(self.request_overrides.generation_config)
        if self.inference_defaults.generation_config:
            return copy.deepcopy(self.inference_defaults.generation_config)
        return {}

    def _default_stream_flag(self) -> bool:
        if self.request_overrides.stream is not None:
            return bool(self.request_overrides.stream)
        return bool(self.inference_defaults.stream)

    def _default_cache_value(self) -> Optional[str]:
        if self.request_overrides.cache not in (None, "", [], {}):
            return self.request_overrides.cache
        value = self.inference_defaults.cache
        if value in (None, "", [], {}):
            return None
        return value

    def _default_return_prompt(self) -> Optional[str]:
        if self.request_overrides.return_prompt not in (None, "", [], {}):
            return self.request_overrides.return_prompt
        value = self.inference_defaults.return_prompt
        if value in (None, "", [], {}):
            return None
        return value

    def _default_no_tools_parse_flag(self) -> bool:
        if self.request_overrides.no_tools_parse:
            return True
        return bool(self.inference_defaults.no_tools_parse)

    def _default_suppress_full_response_flag(self) -> bool:
        if self.request_overrides.suppress_full_response:
            return True
        return bool(self.inference_defaults.suppress_full_response)

    def _context_flag_defaults(self) -> Dict[str, Any]:
        params = getattr(self.chat_session, "initial_params", None) or {}
        max_seed = params.get("max_new_tokens")
        if max_seed is None:
            max_seed = params.get("max_new_tokens_override")
        base: Dict[str, Any] = {
            "stream": params.get("stream", params.get("streaming", self._default_stream_flag())),
            "cache": params.get("cache_override", self._default_cache_value()),
            "return_prompt": params.get("return_prompt_mode", self._default_return_prompt()),
            "generation_config_template": copy.deepcopy(params.get("generation_config_template") or {}),
            "max_new_tokens_override": max_seed,
            "no_tools_parse": params.get("no_tools_parse", self._default_no_tools_parse_flag()),
            "auto_retry_truncated": params.get("auto_retry_truncated", False),
            "suppress_full_response": params.get("suppress_full_response", self._default_suppress_full_response_flag()),
            "results_as_user_role": params.get("results_as_user_role", False),
            "pack_results_as_one_role": params.get("pack_results_as_one_role", False),
            "advertised_tools": copy.deepcopy(params.get("advertised_tools") or []),
            "silent_tools": copy.deepcopy(params.get("silent_tools") or []),
            "disabled_tools": copy.deepcopy(params.get("disabled_tools") or []),
            "auto_tool_retry_limit": params.get("auto_tool_retry_limit"),
            "auto_continue_retry_limit": params.get("auto_continue_retry_limit"),
        }
        max_tokens = base.get("max_new_tokens_override")
        if max_tokens is not None:
            try:
                base["max_new_tokens_override"] = int(max_tokens)
            except (TypeError, ValueError):
                base["max_new_tokens_override"] = None
        for limit_key in ("auto_tool_retry_limit", "auto_continue_retry_limit"):
            limit_val = base.get(limit_key)
            if limit_val is None:
                continue
            try:
                base[limit_key] = int(limit_val)
            except (TypeError, ValueError):
                base[limit_key] = None
        for flag in CHAT_FLAG_FIELDS:
            base.setdefault(flag, None)
        return base

    def _context_flag_snapshot(self) -> Dict[str, Any]:
        snapshot = copy.deepcopy(self._context_flag_defaults())
        root_cmds = list(getattr(self.root_turn, "cmd", []) or [])
        for cmd in root_cmds:
            change = cmd.data.get("change")
            if not change or change not in CHAT_FLAG_FIELDS:
                continue
            if cmd.cmd_type == Command.PARAM_CHANGE:
                value = copy.deepcopy(cmd.data.get("new_value"))
            else:
                value = copy.deepcopy(cmd.data.get("value"))
            snapshot[change] = value
        return snapshot

    def _normalize_param_key(self, name: str) -> str:
        if not name:
            raise KeyError("Parameter name cannot be empty.")
        if name in PARAM_NAME_TO_CHANGE:
            return PARAM_NAME_TO_CHANGE[name]
        if name in CHAT_FLAG_FIELDS:
            return name
        raise KeyError(f"Unknown chat parameter '{name}'.")

    def _normalize_param_value(self, key: str, value: Any) -> Any:
        if key == "stream":
            return None if value is None else bool(value)
        if key == "cache":
            if value is None:
                return None
            if isinstance(value, str):
                trimmed = value.strip()
                return trimmed or None
            return value
        if key == "return_prompt":
            if value is None:
                return None
            if isinstance(value, str):
                trimmed = value.strip()
                return trimmed or None
            return value
        if key == "generation_config_template":
            return copy.deepcopy(value) if value is not None else None
        if key == "max_new_tokens_override":
            if value in (None, ""):
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if parsed > 0 else None
        if key in {"no_tools_parse", "auto_retry_truncated", "suppress_full_response", "results_as_user_role", "pack_results_as_one_role"}:
            return None if value is None else bool(value)
        if key in {"advertised_tools", "silent_tools", "disabled_tools"}:
            if value is None:
                return []
            if isinstance(value, str):
                return [p.strip() for p in value.split(",") if p.strip()]
            try:
                return [str(item).strip() for item in value if str(item).strip()]
            except TypeError:
                return []
        if key in {"auto_tool_retry_limit", "auto_continue_retry_limit"}:
            if value in (None, ""):
                return None
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            return parsed if parsed >= 0 else None
        return copy.deepcopy(value)

    def get_param(self, name: str, default: Any = None) -> Any:
        try:
            change_key = self._normalize_param_key(name)
        except KeyError:
            return default
        snapshot = self._context_flag_snapshot()
        if change_key not in snapshot:
            return default
        value = snapshot[change_key]
        if value is None:
            return default
        return copy.deepcopy(value)

    def get_param_snapshot(self) -> Dict[str, Any]:
        snapshot = self._context_flag_snapshot()
        public_view: Dict[str, Any] = {}
        for public_name in CHAT_PARAM_NAMES:
            change_key = PARAM_NAME_TO_CHANGE[public_name]
            value = snapshot.get(change_key)
            public_view[public_name] = copy.deepcopy(value)
        return public_view

    def get_scope_turns(self, start_node: Optional[Union[Turn, ChatCursor, str]] = None, suppress_auto: bool = False, history_only: bool = False) -> List[Turn]:
        """
        Returns a list of turns within the scope of start_node, ordered chronologically.
        """
        turn = None
        if isinstance(start_node, Turn):
            turn = start_node
        elif isinstance(start_node, ChatCursor):
            turn = start_node.head
        elif isinstance(start_node, str):
            turn = self.session.get_turn_by_gen_id(start_node, self.chat_session)
        
        start_turn = turn or self.root_turn
        if not start_turn:
            return []
        
        return self.session.get_scope_turns(start_turn, suppress_auto, history_only)

    def get_scope_commands(self, start_node: Optional[Union[Turn, ChatCursor, str]] = None, suppress_auto: bool = False, include_logs: bool = False) -> List[Command]:
        """
        Returns a list of commands within the scope of start_node, ordered chronologically.
        """
        turn = None
        if isinstance(start_node, Turn):
            turn = start_node
        elif isinstance(start_node, ChatCursor):
            turn = start_node.head
        elif isinstance(start_node, str):
            turn = self.session.get_turn_by_gen_id(start_node, self.chat_session)
        
        start_turn = turn or self.root_turn
        if not start_turn:
            return []

        return self.session.get_scope_commands(start_turn, suppress_auto, include_logs)

    def set_param(self, name: str, value: Any, *, command_text: Optional[str] = None) -> None:
        change_key = self._normalize_param_key(name)
        normalized_value = self._normalize_param_value(change_key, value)
        cmd_type = Command.PARAM_CHANGE if change_key in PARAM_CHANGE_FIELDS else Command.STATE_CHANGE
        self.session.reconcile_state_command(
            self.root_turn,
            change_key,
            normalized_value,
            command_text=command_text,
            cmd_type=cmd_type,
        )

    def _next_cursor_name(self) -> str:
        self._cursor_seq += 1
        return f"cursor_{self._cursor_seq}"

    def _default_cursor_alias(self, cursor: ChatCursor) -> str:
        """Derive a stable alias, preferring gen_id over display_id for speed."""
        base = (cursor.head.gen_id_or_parent if cursor.head else None) or "cursor"
        with self._rw_lock.read_lock():
            existing_handles = set(self._cursors.keys())
        if base in existing_handles:
            suffix = 1
            candidate = f"{base}_{suffix}"
            while candidate in existing_handles:
                suffix += 1
                candidate = f"{base}_{suffix}"
            base = candidate
        return base

    def _register_cursor(
        self,
        cursor: ChatCursor,
        alias: Optional[str] = None,
        *,
        make_active: bool = False,
        scope: Optional[ChatContextScope] = None,
    ) -> ChatCursor:
        if self.has_cursor(cursor) and (alias is None or alias == cursor.context_id):
            if scope and cursor.scope is not scope:
                cursor.scope = scope
            if make_active:
                self.set_active_cursor(cursor, scope=scope)
            return cursor
        fallback_alias = self._default_cursor_alias(cursor)
        handle = alias or cursor.context_id or fallback_alias or self._next_cursor_name()
        cursor.bind_context(self, context_id=handle)
        if scope:
            cursor.scope = scope
        elif cursor.scope is None:
            cursor.scope = self._default_scope
        became_active = False
        with self._rw_lock.write_lock():
            target_scope = self.get_scope(scope)
            self._cursors[handle] = cursor
            if make_active:
                target_scope.active_cursor_id = handle
                target_scope.active_cursor_ref = cursor
                target_scope.active_cursor_override = None
                if target_scope is self._default_scope:
                    self._active_cursor_id = handle
                became_active = True
            elif target_scope.active_cursor_id is None:
                target_scope.active_cursor_id = handle
                target_scope.active_cursor_ref = cursor
                target_scope.active_cursor_override = None
                if target_scope is self._default_scope:
                    self._active_cursor_id = handle
                became_active = True
        return cursor

    def _resolve_cursor(
        self,
        cursor_or_id: Optional[Union[str, ChatCursor]],
        *,
        scope: Optional[ChatContextScope] = None,
    ) -> ChatCursor:
        if cursor_or_id is None:
            target_scope = self.get_scope(scope)
            ref = target_scope.active_cursor_ref
            if ref and self.has_cursor(ref) and getattr(ref, "scope", None) is target_scope:
                return ref
            with self._rw_lock.read_lock():
                active_id = target_scope.active_cursor_id
            if not active_id and target_scope is not self._default_scope:
                with self._rw_lock.read_lock():
                    cursor_items = list(self._cursors.items())
                for handle, cursor in cursor_items:
                    if getattr(cursor, "scope", None) is target_scope:
                        active_id = handle
                        break
                if active_id:
                    with self._rw_lock.write_lock():
                        target_scope.active_cursor_id = active_id
            if not active_id and target_scope is self._default_scope:
                with self._rw_lock.read_lock():
                    active_id = self._active_cursor_id
            if active_id is None:
                raise RuntimeError("ChatContext has no active cursor.")
            with self._rw_lock.read_lock():
                cursor = self._cursors.get(active_id) if active_id else None
            if not cursor:
                raise KeyError(f"Unknown cursor id '{active_id}'.")
            if getattr(cursor, "scope", None) is target_scope:
                target_scope.active_cursor_ref = cursor
            return cursor
        if isinstance(cursor_or_id, ChatCursor):
            return cursor_or_id
        with self._rw_lock.read_lock():
            cursor = self._cursors.get(cursor_or_id)
        if cursor:
            return cursor
        # Attempt to rebuild a cursor for this gen_id if it belongs to this context.
        turn = self.session.get_turn_by_gen_id(cursor_or_id, self.chat_session)
        if turn and self.owns_turn(turn):
            rebuilt = self._build_cursor_for_turn(turn)
            if rebuilt:
                return self._register_cursor(rebuilt, alias=cursor_or_id, make_active=False)
        raise KeyError(f"Unknown cursor id '{cursor_or_id}'.")

    # ---- cursor registry ---------------------------------------------------
    @property
    def active_cursor(self) -> ChatCursor:
        return self._resolve_cursor(None, scope=self._default_scope)

    def active_cursor_for_scope(self, scope: Optional[ChatContextScope] = None) -> ChatCursor:
        return self._resolve_cursor(None, scope=scope)

    def set_active_cursor(
        self,
        cursor_or_id: Union[str, ChatCursor],
        *,
        scope: Optional[ChatContextScope] = None,
    ) -> ChatCursor:
        cursor = self._resolve_cursor(cursor_or_id, scope=scope)
        if scope and cursor.scope is not scope:
            cursor.scope = scope
        if cursor.context_id:
            target_scope = self.get_scope(scope)
            with self._rw_lock.write_lock():
                target_scope.active_cursor_id = cursor.context_id
                target_scope.active_cursor_ref = cursor
                target_scope.active_cursor_override = None
                if target_scope is self._default_scope:
                    self._active_cursor_id = cursor.context_id
        return cursor

    def active_cursor_id(self, scope: Optional[ChatContextScope] = None) -> Optional[str]:
        target_scope = self.get_scope(scope)
        return target_scope.active_cursor_id or (self._active_cursor_id if target_scope is self._default_scope else None)

    def resolve_try_out_cursor(
        self,
        anchor: TryOutAnchor,
        *,
        prefer_latest: bool = True,
    ) -> Optional[ChatCursor]:
        """Resolve a live try-out cursor for an anchor, if still registered."""
        if not anchor:
            return None
        handles = list(anchor.try_out_cursor_ids or [])
        if prefer_latest:
            handles = list(reversed(handles))
        with self._rw_lock.read_lock():
            for handle in handles:
                if not handle:
                    continue
                cursor = self._cursors.get(handle)
                if cursor:
                    return cursor
        return None

    def get_cursor(self, handle: str) -> Optional[ChatCursor]:
        """Return a registered cursor by handle, if present."""
        if not handle:
            return None
        with self._rw_lock.read_lock():
            return self._cursors.get(handle)

    def has_cursor(self, cursor: ChatCursor) -> bool:
        """Return True when the cursor is registered in this context by identity."""
        if cursor.context is not self:
            return False
        with self._rw_lock.read_lock():
            if cursor.context_id and self._cursors.get(cursor.context_id) is cursor:
                return True
            cursors = list(self._cursors.values())
        return any(c is cursor for c in cursors)

    def cursor_handle(self, cursor: Optional[ChatCursor]) -> Optional[str]:
        """Return the registered handle for a cursor when tracked by this context."""
        if not cursor or cursor.context is not self:
            return None
        handle = cursor.context_id
        with self._rw_lock.read_lock():
            if handle and self._cursors.get(handle) is cursor:
                return handle
            items = list(self._cursors.items())
        for key, value in items:
            if value is cursor:
                return key
        return None

    def is_same_cursor(self, left: Optional[Union[str, ChatCursor]], right: Optional[Union[str, ChatCursor]]) -> bool:
        """Return True when two cursor refs resolve to the same registered handle."""
        if left is None or right is None:
            return False
        if isinstance(left, ChatCursor) and isinstance(right, ChatCursor):
            if left is right:
                return True
            left_handle = self.cursor_handle(left)
            right_handle = self.cursor_handle(right)
            return bool(left_handle and left_handle == right_handle)
        if isinstance(left, str) and isinstance(right, str):
            return left == right
        if isinstance(left, str) and isinstance(right, ChatCursor):
            return left == self.cursor_handle(right)
        if isinstance(left, ChatCursor) and isinstance(right, str):
            return right == self.cursor_handle(left)
        return False

    def find_cursor_by_gen_id(self, gen_id: Optional[str]) -> Optional[ChatCursor]:
        """Find a registered cursor by its head gen_id without rebuilding."""
        if not gen_id:
            return None
        with self._rw_lock.read_lock():
            cursors = list(self._cursors.values())
        for cursor in cursors:
            if cursor.head and cursor.head.gen_id == gen_id:
                return cursor
        return None

    def register_cursor_for_turn(
        self,
        target_turn: Turn,
        *,
        alias: Optional[str] = None,
        make_active: bool = False,
        scope: Optional[ChatContextScope] = None,
    ) -> Optional[ChatCursor]:
        """Build and register a cursor for a turn without turn-based lookup."""
        if not target_turn:
            return None
        self._ensure_turn_in_scope(target_turn)
        existing = self.find_cursor_by_gen_id(getattr(target_turn, "gen_id", None))
        if existing:
            if scope:
                existing.scope = scope
            if make_active:
                self.set_active_cursor(existing, scope=scope)
            return existing
        rebuilt = self._build_cursor_for_turn(target_turn, scope=scope)
        if not rebuilt:
            return None
        return self._register_cursor(rebuilt, alias=alias, make_active=make_active, scope=scope)

    def register_cursor_if_needed(
        self,
        cursor: ChatCursor,
        *,
        alias: Optional[str] = None,
        make_active: bool = False,
        scope: Optional[ChatContextScope] = None,
    ) -> ChatCursor:
        """Register a cursor by identity when it is not already tracked."""
        if self.has_cursor(cursor):
            if make_active:
                self.set_active_cursor(cursor, scope=scope)
            return cursor
        if cursor.context is not self:
            cursor.bind_context(self, context_id=None)
        if scope:
            cursor.scope = scope
        return self._register_cursor(cursor, alias=alias, make_active=make_active, scope=scope)

    def drop_cursor(self, cursor_or_id: Union[str, ChatCursor]) -> None:
        cursor = self._resolve_cursor(cursor_or_id)
        with self._rw_lock.write_lock():
            handle = cursor.context_id
            if handle is None:
                handle = next((key for key, value in self._cursors.items() if value is cursor), None)
            if handle and handle in self._cursors:
                del self._cursors[handle]
                for scope in self._scopes.values():
                    if scope.active_cursor_id == handle:
                        replacement = next(
                            (h for h, cur in self._cursors.items() if getattr(cur, "scope", None) is scope),
                            None,
                        )
                        scope.active_cursor_id = replacement
                        scope.active_cursor_ref = self._cursors.get(replacement) if replacement else None
                    if scope.active_cursor_ref is cursor:
                        scope.active_cursor_ref = None
                    if scope.active_cursor_override is cursor:
                        scope.active_cursor_override = None
                if self._active_cursor_id == handle:
                    self._active_cursor_id = self._default_scope.active_cursor_id
        cursor.context_id = None
        cursor.context = None
        cursor.scope = None

    def adopt_cursor(
        self,
        cursor: ChatCursor,
        alias: Optional[str] = None,
        *,
        make_active: bool = False,
        scope: Optional[ChatContextScope] = None,
    ) -> ChatCursor:
        """Register an externally constructed cursor (for example from replay)."""
        return self._register_cursor(cursor, alias=alias, make_active=make_active, scope=scope)

    def resurrect_cursor_for_gen_id(
        self,
        gen_id: str,
        *,
        alias: Optional[str] = None,
        make_active: bool = False,
        scope: Optional[ChatContextScope] = None,
    ) -> ChatCursor:
        """Register a new cursor for a historical turn by gen_id."""
        if not gen_id:
            raise ValueError("gen_id is required.")
        existing = self.find_cursor_by_gen_id(gen_id)
        if existing:
            if make_active:
                self.set_active_cursor(existing, scope=scope)
            return existing
        turn = self.session.get_turn_by_gen_id(gen_id, self.chat_session)
        if not turn:
            raise KeyError(f"Turn '{gen_id}' was not found in the current chat session.")
        if not self.owns_turn(turn):
            raise ValueError(f"Turn '{gen_id}' is not within the active ChatContext scope.")
        rebuilt = self._build_cursor_for_turn(turn, scope=scope)
        if not rebuilt:
            raise RuntimeError(f"Failed to rebuild cursor for '{gen_id}'.")
        return self._register_cursor(rebuilt, alias=alias, make_active=make_active, scope=scope)

    def find_cursor_for_turn(self, target_turn: Optional[Turn]) -> Optional[ChatCursor]:
        if not target_turn:
            return None
        self._ensure_turn_in_scope(target_turn)
        with self._rw_lock.read_lock():
            cursor_items = list(self._cursors.items())
        for handle, cursor in cursor_items:
            if cursor.head is target_turn:
                return cursor
        return None

    def _blank_cursor(self) -> ChatCursor:
        cursor = ChatCursor(
            self,
            self.root_turn,
            scope=self._default_scope,
            origin_user_turn_index=_count_real_user_turns_on_path(self.root_turn, self.root_turn),
            is_fork=False,
            gen_config=self._resolve_generation_config(),
            streaming=None,
            cache=None,
            return_prompt=None,
            adapter_override_for_next_turn=self._resolve_adapter_override(),
        )
        return cursor

    def _build_cursor_for_turn(
        self,
        target_turn: Turn,
        *,
        scope: Optional[ChatContextScope] = None,
    ) -> Optional[ChatCursor]:
        if not target_turn:
            return None
        path = self._relative_path_from_root(target_turn)
        if not path:
            raise ValueError("Target turn is not reachable from this ChatContext root.")
        
        # With history-aware state calculation, we no longer need to replay
        # commands to build the cursor's state. We just point it to the turn.
        cursor = self._blank_cursor()
        if scope:
            cursor.scope = scope
        cursor._assign_head(target_turn)
        cursor.origin_user_turn_index = _count_real_user_turns_on_path(self.root_turn, target_turn)
        return cursor

    def find_active_anchor(
        self,
        kind: str,
        cursor: Optional["ChatCursor"] = None,
        *,
        scope: Optional[ChatContextScope] = None,
    ) -> Optional[TryOutAnchor]:
        """Find an active anchor of a specific kind, optionally scoped to a cursor's lineage."""
        normalized = _normalize_tryout_kind(kind)
        with self._rw_lock.read_lock():
            anchors = [
                a for a in self._try_out_anchors.values()
                if a.kind in normalized and (scope is None or a.owner_scope is scope)
            ]
        if not anchors:
            return None

        if cursor and cursor.current_turn:
            ancestry: Set[Turn] = set()
            node = cursor.current_turn
            while node:
                ancestry.add(node)
                node = getattr(node, "parent", None)
            for anchor in anchors:
                anchor_turn = anchor.anchor_turn
                if anchor_turn and (anchor_turn in ancestry or any(t in ancestry for t in anchor.try_out_turns)):
                    return anchor

        # No lineage match; avoid cross-branch reuse.
        return None

    def get_try_out_anchor(
        self,
        anchor_name: str,
        *,
        scope: Optional[ChatContextScope] = None,
        allow_foreign_scope: bool = False,
    ) -> Optional[TryOutAnchor]:
        """Return a tracked try-out anchor by name, if present."""
        with self._rw_lock.read_lock():
            anchor = self._try_out_anchors.get(anchor_name)
        if anchor and scope and anchor.owner_scope is not scope and not allow_foreign_scope:
            return None
        return anchor

    def cursors_snapshot(self) -> List[Tuple[str, ChatCursor]]:
        """Snapshot of registered (handle, cursor) pairs."""
        with self._rw_lock.read_lock():
            return list(self._cursors.items())

    def try_out_anchors_snapshot(self, *, scope: Optional[ChatContextScope] = None) -> List[TryOutAnchor]:
        """Snapshot of tracked try-out anchors."""
        with self._rw_lock.read_lock():
            anchors = list(self._try_out_anchors.values())
        if scope is None:
            return anchors
        return [anchor for anchor in anchors if anchor.owner_scope is scope]

    def start_try_out_anchor(
        self,
        anchor_name: str,
        anchor_turn: Turn,
        kind: str = "try_out",
        retry_limit: int = 5,
        origin_cursor: Optional["ChatCursor"] = None,
        scope: Optional[ChatContextScope] = None,
    ) -> TryOutAnchor:
        """Creates a new try-out anchor."""
        with self._rw_lock.read_lock():
            if anchor_name in self._try_out_anchors:
                raise ValueError(f"Anchor with name '{anchor_name}' already exists.")

        origin_cursor_id = None
        if origin_cursor and origin_cursor.context is self:
            if not origin_cursor.context_id:
                try:
                    self.register_cursor_if_needed(origin_cursor, make_active=False, scope=scope)
                except Exception:
                    pass
            origin_cursor_id = origin_cursor.context_id
        target_scope = self.get_scope(scope)
        if not origin_cursor_id:
            origin_cursor_id = target_scope.active_cursor_id or self._active_cursor_id
        anchor = TryOutAnchor(
            anchor_name=anchor_name,
            anchor_turn=anchor_turn,
            kind=kind,
            retries_remaining=retry_limit,
            retry_limit=retry_limit,
            origin_cursor_id=origin_cursor_id,
            owner_scope=target_scope,
        )
        with self._rw_lock.write_lock():
            if anchor_name in self._try_out_anchors:
                raise ValueError(f"Anchor with name '{anchor_name}' already exists.")
            self._try_out_anchors[anchor_name] = anchor
        return anchor

    def close_try_out_anchor(
        self,
        anchor_name: str,
        dist_mode: Optional[str] = "keep",
        main_thread_index: Optional[int] = None,
        *,
        scope: Optional[ChatContextScope] = None,
    ) -> Optional["ChatCursor"]:
        """
        Close all try-out branches tied to an anchor and optionally control
        main_thread promotion behavior.
        """
        with self._rw_lock.read_lock():
            anchor = self._try_out_anchors.get(anchor_name)
        if not anchor:
            raise ValueError(f"Anchor with name '{anchor_name}' does not exist.")
        if scope and anchor.owner_scope is not scope:
            raise ValueError(f"Anchor '{anchor_name}' is not owned by the active scope.")

        branches: List[Turn] = list(anchor.try_out_turns or [])
        if not branches:
            with self._rw_lock.write_lock():
                self._try_out_anchors.pop(anchor_name, None)
            return None

        origin_cursor: Optional[ChatCursor] = None
        if anchor.origin_cursor_id:
            with self._rw_lock.read_lock():
                origin_cursor = self._cursors.get(anchor.origin_cursor_id)
        if not origin_cursor:
            origin_cursor = self.resolve_try_out_cursor(anchor, prefer_latest=False)
        mode = (dist_mode or "").lower() if dist_mode is not None else None
        promote_targets: Optional[List[Turn]] = None
        if mode:
            valid_modes = {"keep", "all", "none", "index"}
            if mode not in valid_modes:
                raise ValueError(f"Invalid distribution mode '{dist_mode}'. Expected one of {sorted(valid_modes)}.")

            if mode == "index":
                if main_thread_index is None:
                    raise ValueError("main_thread_index is required when dist_mode is 'index'.")
                if main_thread_index < 0 or main_thread_index >= len(branches):
                    raise ValueError("main_thread_index out of range for available try-outs.")
                promote_targets = [branches[main_thread_index]]
            elif mode == "all":
                promote_targets = branches
            elif mode == "none":
                promote_targets = []

            parent_for_promotion: Optional[Turn] = branches[0].parent if branches else None
            if promote_targets is not None and parent_for_promotion:
                try:
                    self.session.promote_tryouts_to_main(parent_for_promotion, promote_targets)
                except Exception:
                    # Promotion failure should not block branch closure.
                    pass

        placeholder_turn: Optional[Turn] = None
        for branch_turn in branches:
            keep_flag = bool(getattr(branch_turn, "main_thread", False))
            the_only_main = bool(promote_targets and branch_turn is promote_targets[0] and mode == "index")
            try:
                closed_turn = self.session.close_branch(branch_turn)
                if keep_flag:
                    branch_turn.main_thread = True
                if the_only_main and branch_turn.parent:
                    try:
                        self.session.promote_tryouts_to_main(branch_turn.parent, [branch_turn])
                    except Exception:
                        pass
                if closed_turn and not placeholder_turn:
                    placeholder_turn = closed_turn
            except Exception:
                # Keep closing other branches even if one fails.
                continue

        with self._rw_lock.write_lock():
            self._try_out_anchors.pop(anchor_name, None)
        target_cursor = origin_cursor
        if placeholder_turn:
            try:
                existing = self.find_cursor_for_turn(placeholder_turn)
                if existing:
                    target_cursor = existing
                elif origin_cursor and origin_cursor.context is self and self.owns_turn(placeholder_turn):
                    target_cursor = origin_cursor.rebind_to_turn(placeholder_turn)
                else:
                    target_cursor = self.register_cursor_for_turn(
                        placeholder_turn,
                        make_active=False,
                        scope=scope,
                    ) or target_cursor
            except Exception:
                pass

        origin_handle = anchor.origin_cursor_id
        with self._rw_lock.read_lock():
            cursor_map = {handle: self._cursors.get(handle) for handle in (anchor.try_out_cursor_ids or [])}
        for handle in list(anchor.try_out_cursor_ids or []):
            if origin_handle and handle == origin_handle:
                continue
            cursor = cursor_map.get(handle)
            if cursor:
                try:
                    self.drop_cursor(cursor)
                except Exception:
                    pass

        if target_cursor:
            self.set_active_cursor(target_cursor, scope=scope)
        return target_cursor

    def find_all_turns_with_anchor_data(self) -> List[Turn]:
        """Scans the entire active conversation for turns with '$try_out' metadata."""
        # This is a conceptual helper. The actual implementation might need to call a
        # method on the underlying EngineSession to search all turns efficiently.
        all_turns = self.session._get_all_turns(self.chat_session) # Assuming _get_all_turns exists and is available
        return [t for t in all_turns if t.data and "$try_out" in t.data]

    def resurrect_try_out_anchor(
        self,
        anchor_name: str,
        *,
        scope: Optional[ChatContextScope] = None,
    ) -> Optional[TryOutAnchor]:
        """
        Finds all turns related to a (presumably closed) anchor and "resurrects" it
        into the active runtime context.
        """
        with self._rw_lock.read_lock():
            existing_anchor = self._try_out_anchors.get(anchor_name)
        if existing_anchor:
            #print(f"Warning: Anchor '{anchor_name}' is already active in the runtime context.")
            return existing_anchor

        # 1. Find all turns in the session that are part of this anchor
        all_turns = self.session._get_all_turns(self.chat_session)
        related_turns = [
            t for t in all_turns
            if t.data and isinstance(t.data.get("$try_out"), dict) and t.data["$try_out"].get("anchor") == anchor_name
        ]

        if not related_turns:
            raise ValueError(f"No turns found for anchor '{anchor_name}' in the session history.")

        # 2. Re-create the TryOutAnchor object
        # Find the original anchor_turn (parent of the placeholders) and kind
        anchor_turn: Optional[Turn] = None
        kind: str = "try_out"
        placeholder_turns: List[Turn] = []

        for turn in related_turns:
            # The placeholder turns are direct children of the anchor_turn
            if turn.parent and not anchor_turn:
                # Check if this turn's parent is the anchor point
                is_placeholder = False
                for sibling in turn.parent.turns:
                    if sibling is not turn and sibling.data and isinstance(sibling.data.get("$try_out"), dict) and sibling.data["$try_out"].get("anchor") == anchor_name:
                        is_placeholder = True
                        break
                if is_placeholder:
                    anchor_turn = turn.parent

            if turn.data and isinstance(turn.data.get("$try_out"), dict):
                kind = turn.data["$try_out"].get("kind", kind)
                # A placeholder turn is one that has a '$try_out' role
                if turn.data["$try_out"].get("role") in ("main", "try"):
                    placeholder_turns.append(turn)

        if not anchor_turn:
            # Fallback for finding anchor_turn if the first method fails
            for turn in placeholder_turns:
                if turn.parent:
                    anchor_turn = turn.parent
                    break
            if not anchor_turn:
                raise ValueError(f"Could not determine the original anchor point for '{anchor_name}'.")

        resurrected_anchor = TryOutAnchor(
            anchor_name=anchor_name,
            anchor_turn=anchor_turn,
            kind=kind,
            # We only add the placeholder turns that were direct children of the anchor
            try_out_turns=[t for t in placeholder_turns if t.parent is anchor_turn],
            owner_scope=scope or self._default_scope,
        )

        # 3. Add it to the runtime context
        with self._rw_lock.write_lock():
            if anchor_name in self._try_out_anchors:
                return self._try_out_anchors[anchor_name]
            self._try_out_anchors[anchor_name] = resurrected_anchor
        return resurrected_anchor

    def close_try_out_anchors_by_kind(
        self,
        kinds: Sequence[str],
        dist_mode: Optional[str] = "keep",
        *,
        anchor_scope: Optional[Set["Turn"]] = None,
        scope: Optional[ChatContextScope] = None,
    ) -> Optional["ChatCursor"]:
        normalized_kinds: Set[str] = set()
        for kind in kinds:
            normalized_kinds.update(_normalize_tryout_kind(kind))
        with self._rw_lock.read_lock():
            anchor_items = list(self._try_out_anchors.items())
        anchors_to_close = [
            name
            for name, anchor in anchor_items
            if anchor.kind in normalized_kinds
            and (anchor_scope is None or anchor.anchor_turn in anchor_scope)
            and (scope is None or anchor.owner_scope is scope)
        ]
        last_cursor: Optional["ChatCursor"] = None
        for anchor_name in anchors_to_close:
            try:
                last_cursor = self.close_try_out_anchor(
                    anchor_name,
                    dist_mode=dist_mode,
                    scope=scope,
                )
            except Exception:
                # Best-effort cleanup; continue closing other anchors.
                continue
        return last_cursor

# ----------------------- streaming display helpers ------------------------

@dataclass
class StreamDisplayContext:
    """Hints describing how to render streaming output for a prompt."""

    prompt_index: int = 0
    original_prompt_index: Optional[int] = None
    total_original_prompts: Optional[int] = None
    batch_index: Optional[int] = None
    total_batches: Optional[int] = None
    adapters_label: Optional[str] = None
    show_batch_header: bool = False
    show_prompt_echo: bool = True
    show_response_banner: bool = True
    show_turn_counter: bool = True
    response_label: Optional[str] = None
    show_override_banner: bool = False
    override_total: Optional[int] = None
    cursor_override: Optional["ChatCursor"] = None


@dataclass
class StreamDisplayPlan:
    """Maps prompt indexes to display contexts with a sensible default."""

    per_prompt: Dict[int, StreamDisplayContext] = field(default_factory=dict)
    default_context: StreamDisplayContext = field(default_factory=StreamDisplayContext)

    def resolve(self, prompt_index: int) -> StreamDisplayContext:
        return self.per_prompt.get(prompt_index, self.default_context)


# ------------------------------ utilities ---------------------------------

def _count_real_user_turns_on_path(root: Turn, head: Turn) -> int:
    """Count user (non-automation) turns on the path [root..head].
    Relies on Turn.data structure used by your session.
    """
    path = list(_walk_path(root, head))
    count = 0
    for t in path:
        d = t.data or {}
        role = d.get("role") or (d.get("user") or {}).get("role")
        user_block = d.get("user") or {}
        is_auto_flag = bool(getattr(t, "is_auto", False))
        if role == "user" and not is_auto_flag:
            count += 1
    return count


def _walk_path(root: Turn, head: Turn) -> Iterable[Turn]:
    out: List[Turn] = []
    node = head
    while node is not None:
        out.append(node)
        if node is root:
            break
        node = node.parent
        if node is None:
            break
    out.reverse()
    return out
