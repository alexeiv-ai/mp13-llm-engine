# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: MIT
from __future__ import annotations
import time
import os
import json
import uuid
import functools
import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple, Generator, Union, Iterable, Sequence, NamedTuple, Literal, Mapping
import copy
import heapq
from pathlib import Path
from dataclasses import dataclass, field, replace, is_dataclass, asdict
import threading
from contextlib import contextmanager
from collections import defaultdict


# Import for rehydrating tool call objects
from mp13_engine.mp13_config import ToolCallBlock, ToolCall, ParserProfile
from mp13_engine.mp13_tools_parser import ToolsParserHelper
from mp13_engine.mp13_toolbox import Toolbox, ToolsScope, ToolsAccess, ToolsView
from mp13_engine.mp13_tools_parser import UnifiedToolIO

def _rehydrate_tool_call_block(block: ToolCallBlock) -> ToolCallBlock:
    if isinstance(block, ToolCallBlock):
        return block
    if isinstance(block, dict):
        try:
            return ToolCallBlock.from_dict(block)
        except Exception:
            return block
    return block

def _rehydrate_tool_payload_list(items: List[Any]) -> List[Any]:
    if not isinstance(items, list):
        return items
    rehydrated: List[Any] = []
    for item in items:
        if isinstance(item, ToolCallBlock):
            rehydrated.append(_rehydrate_tool_call_block(item))
            continue
        if isinstance(item, dict) and "raw_block" in item:
            try:
                rehydrated.append(ToolCallBlock.from_dict(item))
            except Exception:
                rehydrated.append(item)
            continue
        if isinstance(item, ToolCall):
            rehydrated.append(item)
            continue
        if isinstance(item, dict) and item.get("name"):
            try:
                rehydrated.append(ToolCall.from_dict(item))
            except Exception:
                rehydrated.append(item)
            continue
        rehydrated.append(item)
    return rehydrated

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    BRIGHT_BLUE = "\033[94m"    
    CYAN = "\033[36m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_BLACK = "\033[90m"  # Grey
    BRIGHT_YELLOW = "\033[93m"
    WHITE = "\033[97m"
    LOG_DEBUG = "\033[2m"  # DIM
    LOG_INFO = "\033[2m"   # DIM
    LOG_WARNING = "\033[33;20m" # YELLOW
    LOG_ERROR = "\033[31;20m" # RED

    YOU_HEADER = f"{BOLD}{GREEN}"
    YOU_CONTENT = f"{BOLD}{GREEN}"
    LLM_CONTENT = f"{BOLD}{YELLOW}"
    LLM_HEADER = f"{YELLOW}"
    SYSTEM = f"{CYAN}"
    TOOL = f"{CYAN}"
    TOOL_ARGS = f"{DIM}{CYAN}"
    TOOL_WARNING = f"{DIM}{RED}"
    METRICS = f"{DIM}{BRIGHT_BLACK}"
    ERROR = f"{BOLD}{RED}"
    SUCCESS = f"{BOLD}{GREEN}"
    HEADER = f"{BOLD}{WHITE}"
    ECHO = f"{BOLD}{CYAN}"

@dataclass
class InferenceParams:
    """
    Holds default parameters for an inference request within a ChatSession.
    These are session-level defaults that can be overridden by individual requests.
    """
    stream: bool = True
    return_prompt: Optional[str] = None
    generation_config: Dict[str, Any] = field(default_factory=dict)
    cache: Optional[str] = None
    reset_metrics: bool = False
    override_adapters: Optional[List[str]] = None
    active_adapters: Optional[List[str]] = None
    suppress_full_response: bool = False
    no_tools_parse: bool = False

    def serialize(self, defaults: Optional[InferenceParams] = None) -> Dict[str, Any]:
        """
        Serializes to a dictionary, omitting None/empty values.
        If `defaults` is provided, its values are used for any fields that are empty in `self`.
        """
        from dataclasses import fields

        result = {}
        self_data = asdict(self)
        defaults_data = asdict(defaults) if defaults else {}

        for f in fields(self):
            field_name = f.name
            self_value = self_data.get(field_name)

            # Determine if self_value is "empty"
            is_self_value_empty = self_value is None or (isinstance(self_value, (dict, list)) and not self_value)

            final_value = self_value
            if is_self_value_empty and defaults:
                final_value = defaults_data.get(field_name)

            # Check if the final value is non-empty before adding to result
            is_final_value_empty = final_value is None or (isinstance(final_value, (dict, list)) and not final_value)

            if not is_final_value_empty:
                result[field_name] = final_value

        return result


class Command:
    """Represents a non-conversational command or state change in the session tree."""
    # Command types
    ADAPTERS_STATE: str = "adapters_state" # For recording reconstructed state
    COMMAND: str = "command"
    LOG: str = "log" # For client-side commands that don't call the engine API
    PARAM_CHANGE: str = "param_change" # For direct API parameter changes (e.g., system message)
    STATE_CHANGE: str = "state_change" # For implicit behavior changes (e.g., flags, tool activation)
    TURN: str = "turn" # For logging a conversational turn itself as a command

    def __init__(self, cmd_type: str, metadata: Optional[Dict[str, Any]] = None, api_name: Optional[str] = None, api_params: Optional[Dict[str, Any]] = None):
        self.gen_id: Optional[str] = None
        self.parent: Optional[Union[Turn, str]] = None
        self.cmd_type: str = cmd_type
        self.data: Dict[str, Any] = metadata or {}
        self.timestamp: float = time.time()
        # --- NEW: Fields for storing engine API call details ---
        self.api_name: Optional[str] = api_name
        self.api_params: Optional[Dict[str, Any]] = api_params

    def to_dict(self, include_children: bool = True) -> Dict[str, Any]:
        """Serializes the Command to a dictionary for saving."""
        data = {
            "gen_id": self.gen_id,
            # Store parent as gen_id string if it's a Turn object, otherwise store as is (it's already a string or None)
            "parent_gen_id": self.parent.gen_id if isinstance(self.parent, Turn) else self.parent,
            "cmd_type": self.cmd_type,
            "timestamp": self.timestamp,
        }
        if self.data:
            data["data"] = self.data
        if self.api_name:
            data["api_name"] = self.api_name
        if self.api_params:
            data["api_params"] = self.api_params
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Command":
        """Deserializes a dictionary into a Command object."""
        cmd = cls(
            cmd_type=data.get("cmd_type", data.get("turn_type", cls.COMMAND)),
            metadata=data.get("data"),
            api_name=data.get("api_name"),
            api_params=data.get("api_params")
        )
        # The parent is stored as a gen_id string. It will be re-linked to a Turn object
        # by the session's deserialization logic if the parent turn is found.
        cmd.gen_id = data.get("gen_id")
        cmd.parent = data.get("parent_gen_id")
        cmd.timestamp = data.get("timestamp", time.time())
        return cmd

    @property
    def sort_id(self) -> int:
        """Returns the integer part of the gen_id for stable sorting."""
        if self.gen_id:
            parts = self.gen_id.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                return int(parts[-1])
        return 0 # Fallback for nodes without a standard gen_id

class Turn:
    """Represents a conversational exchange round that includes  both user and assistant parts in the session tree."""
    CHAT: str = "chat"
    BATCH: str = "batch" # A parent for forked CHAT turns
    FORK: str = "fork" # A parent for BATCH turns to manage depth
    HOLD: str = "hold" # Placeholder converted to a stateful anchor
    RESERVED: str = "reserved" # Placeholder reserved as a structural anchor

    def __init__(self, # type: ignore
                 turn_type: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 is_archived: bool = False,
                 main_thread: bool = False,
                 do_continue: bool = False,
                 was_truncated: bool = False,
                 was_canceled: bool = False,
                 metrics: Optional[Dict[str, Any]] = None,
                 root_context: bool = False
                 ):
        # A Turn represents one conversational exchange. The `data` dictionary holds the
        # content of this exchange.
        #
        # A turn can contain input from either a user (`data["user"]`) or from
        # tool execution results (`data["tool_results"]`). These are semantically
        # equivalent inputs for the next assistant response.
        #
        # A turn may lack an input if it is a placeholder for a future command,
        # a `do_continue` turn for extending a response, or a `BATCH`
        # turn where the user inputs are located in its child nodes.

        self.gen_id: Optional[str] = None # Will be assigned by the session for CHAT/BATCH nodes
        self.parent: Optional[Turn] = None # The parent Turn node

        self.turn_type: Optional[str] = turn_type
        self.data: Dict[str, Any] = metadata or {} # Renamed from metadata
        self.is_archived: bool = is_archived
        self.main_thread: bool = main_thread # New flag for main thread
        self.do_continue: bool = do_continue # New flag for continuation turns
        self.was_truncated: bool = was_truncated # New flag for truncated responses
        self.was_canceled: bool = was_canceled # New flag for canceled responses
        self.root_context: bool = root_context # New flag for virtual root
        self.is_auto: bool = False

        self.metrics: Dict[str, Any] = metrics or {} # Initialize metrics
        # Child containers
        self.cmd: List[Command] = []   # This is a list of commands executed for this node vefore turn_type was assigned.
        self.turns: List[Turn] = []    # List of child CHAT or BATCH turns. This is part of the Turn tree structure.

        self.timestamp: float = time.time()

    @property
    def HasResponse(self) -> bool:
        """Checks if the turn has an assistant response or tool results."""
        return "assistant" in self.data

    @property
    def IsEmpty(self) -> bool:
        """Checks if the turn is an empty placeholder."""
        return self.turn_type is None

    @property
    def IsPlaceholderLike(self) -> bool:
        """Returns True for placeholders or reserved placeholder anchors."""
        return self.turn_type is None or self.turn_type in (Turn.HOLD, Turn.RESERVED)

    @property
    def IsFork(self) -> bool:
        """Checks if the turn is a FORK hub."""
        return self.turn_type == Turn.FORK

    @property
    def IsBatch(self) -> bool:
        """Checks if the turn is a BATCH hub."""
        return self.turn_type == Turn.BATCH

    @property
    def IsStructural(self) -> bool:
        """Checks if the turn is a structural node (FORK or BATCH)."""
        return self.IsFork or self.IsBatch

    @property
    def gen_id_or_parent(self) -> str:
        """
        Returns a displayable ID for the turn.
        - If gen_id exists, returns it.
        - If it's a placeholder, returns 'off:<parent_gen_id>'.
        - If it's a root placeholder, returns 'off:root'.
        """
        if self.gen_id:
            return self.gen_id
        if self.parent and self.parent.gen_id:
            return f"off:{self.parent.gen_id}"
        if self.parent is None:
            # This is a root turn. In a multi-conversation session, we might need more context.
            # For now, 'off:root' is a clear indicator.
            return "off:root"
        return "off:unknown"

    # helpers for the "first child is the active branch" rule
    def is_first_child(self) -> bool:
        if not self.parent:
            return True
        if not self.parent.turns:
            return False
        return len(self.parent.turns) != 0 and self.parent.turns[0] is self

    def branch_root(self) -> "Turn":
        """Return the root of this branch by walking up first-child links."""
        node = self
        while node.parent and node.is_first_child():
            node = node.parent
        return node

    def is_try_out_turn(self) -> bool:
        """Return True when this turn is not the first child of its parent."""
        if not self.parent:
            return False
        return not self.is_first_child()
        
    def right_siblings(self) -> list["Turn"]:
        if not self.parent:
            return []
        sibs = list(self.parent.turns or [])
        if not self in sibs:
            return []
        idx = sibs.index(self)
        return [t for t in sibs[idx + 1:] if isinstance(t, Turn)]

    def to_dict(self, include_children: bool = True) -> Dict[str, Any]:
        """Serializes the Turn to a dictionary for saving."""
        data = {
            "gen_id": self.gen_id,
            "turn_type": self.turn_type,
            "timestamp": self.timestamp,
            "_parent_gen_id": self.parent.gen_id if self.parent else None,
        }
        if self.data:
            data["data"] = self.data

        if include_children:
            if self.cmd:
                data["cmd"] = [child.to_dict() for child in self.cmd]
            if self.turns:
                data["turns"] = [child.gen_id for child in self.turns if child.gen_id]
        if self.is_archived: # Only serialize if True
            data["is_archived"] = self.is_archived
        if self.main_thread: # Only serialize if True
            data["main_thread"] = self.main_thread
        if self.do_continue: # Only serialize if True
            data["do_continue"] = self.do_continue
        if self.was_canceled: # Only serialize if True
            data["was_canceled"] = self.was_canceled
        if self.was_truncated: # Only serialize if True
            data["was_truncated"] = self.was_truncated
        if self.root_context: # Only serialize if True
            data["root_context"] = self.root_context
        if self.is_auto: # Only serialize if True
            data["is_auto"] = self.is_auto
        if self.metrics: # Serialize metrics if not empty
            data["metrics"] = self.metrics
        
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Turn:
        """Deserializes a dictionary into a Turn object. Parent is for migration only."""
        turn = cls(
            turn_type=data.get("turn_type"),
            metadata=data.get("data")
        )
        turn.gen_id = data.get("gen_id")
        turn.timestamp = data.get("timestamp", time.time())
        # Commands are now deserialized directly from the nested list
        turn.cmd = [Command.from_dict(cmd_data) for cmd_data in data.get("cmd", [])]
        # After creating the command objects, set their parent to this turn
        for cmd_obj in turn.cmd:
            cmd_obj.parent = turn
        turn.turns = data.get("turns", []) # type: ignore # Will be re-linked
        turn.is_archived = data.get("is_archived", False)
        turn.main_thread = data.get("main_thread", False)
        turn.do_continue = data.get("do_continue", False)
        turn.was_truncated = data.get("was_truncated", False)
        turn.was_canceled = data.get("was_canceled", False)
        turn.root_context = data.get("root_context", False)
        turn.is_auto = data.get("is_auto", False)
        turn.metrics = data.get("metrics", {}) # Deserialize metrics

        # --- Rehydrate ToolCallBlock objects from dictionaries on load (compatibility) ---
        if assistant_msg := turn.data.get("assistant"):
            if "tool_blocks" in assistant_msg:
                rehydrated_blocks = []
                for block_data in assistant_msg.get("tool_blocks", []):
                    if isinstance(block_data, dict):
                        rehydrated_blocks.append(ToolCallBlock.from_dict(block_data))
                    elif isinstance(block_data, ToolCallBlock):
                        rehydrated_blocks.append(_rehydrate_tool_call_block(block_data))
                if rehydrated_blocks:
                    assistant_msg["tool_blocks"] = rehydrated_blocks

        if "tool_results" in turn.data and isinstance(turn.data.get("tool_results"), list):
            turn.data["tool_results"] = _rehydrate_tool_payload_list(turn.data["tool_results"])

        return turn    
    @property
    def sort_id(self) -> int:
        """Returns the integer part of the gen_id for stable sorting."""
        if self.gen_id:
            parts = self.gen_id.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                return int(parts[-1])
        return 0 # Fallback for nodes without a standard gen_id

    def get_all_descendants(self) -> Generator[Turn, None, None]:
        """A generator that yields all descendant turns of this turn."""
        # This method is useful for checking if a turn is an ancestor of another.
        q: List[Turn] = list(self.turns) # type: ignore
        visited = set(self.turns) # type: ignore
        while q:
            current = q.pop(0)
            yield current
            for child in current.turns:
                if child not in visited:
                    q.append(child)
                    visited.add(child)

class ChatSession:
    """Represents a single conversation tree within an EngineSession."""
    def __init__(self,
                 id: Optional[str] = None,
                 ancestor: Optional[str] = None,
                 engine_config: Optional[Dict[str, Any]] = None,
                 engine_warnings: Optional[List[str]] = None,
                 parser_profile: Optional[ParserProfile] = None,
                 inference_defaults: Optional[InferenceParams] = None,
                 initial_params: Optional[Dict[str, Any]] = None,
                 root_turn: Optional[Turn] = None,
                 title: Optional[str] = None,
                 last_active_turn: Optional[Turn] = None):
        self.id: str = id or str(uuid.uuid4())
        self.ancestor: Optional[str] = ancestor
        self.engine_config: Dict[str, Any] = engine_config or {}
        self.parser_profile: Optional[ParserProfile] = None
        self.toolbox: Optional[Toolbox] = None
        self.title: str = title or ""
        self.inference_defaults: InferenceParams = inference_defaults or InferenceParams()
        self.engine_warnings: List[str] = engine_warnings or []
        self.initial_params: Dict[str, Any] = initial_params or {}
        profile_value = copy.deepcopy(parser_profile) if parser_profile is not None else None
        if isinstance(profile_value, dict):
            try:
                profile_value = ParserProfile(**profile_value)
            except Exception:
                profile_value = None
        if isinstance(profile_value, ParserProfile):
            self.parser_profile = profile_value
        # The root_turn is the entry point into the conversation tree for this session.
        # Root must stay as a structural anchor; it should never be repurposed into CHAT/BATCH/etc.
        self.root_turn: Turn = root_turn or Turn(turn_type=Turn.RESERVED)
        if self.root_turn.turn_type != Turn.RESERVED:
            self.root_turn.turn_type = Turn.RESERVED
        # Ensure the root turn is explicitly marked as a root context anchor.
        if self.root_turn and not getattr(self.root_turn, "root_context", False):
            self.root_turn.root_context = True

        self.last_active_turn: Optional[Turn] = last_active_turn or self.root_turn

    def to_dict(self) -> Dict[str, Any]:
        # Ensure initial_params with dataclasses are serialized correctly
        serialized_initial_params = {}
        for key, value in self.initial_params.items():
            if is_dataclass(value):
                serialized_initial_params[key] = asdict(value)  # type: ignore[arg-type]
            elif isinstance(value, ToolsScope):
                serialized_initial_params[key] = value.to_dict()
            elif isinstance(value, ToolsAccess):
                tools_view = value.get_view()
                if tools_view:
                    serialized_initial_params[key] = asdict(tools_view)
            else:
                serialized_initial_params[key] = value
        # root_turn is serialized as part of the main nodes dictionary
        return {
            "id": self.id,
            "ancestor": self.ancestor,
            "engine_config": self.engine_config,
            "initial_params": serialized_initial_params,
            "parser_profile": self.parser_profile,
            "inference_defaults": self.inference_defaults.serialize(),
            "engine_warnings": self.engine_warnings,
            "title": self.title,
            # Add a reference to the root turn's ID for deserialization
            "root_turn_gen_id": self.root_turn.gen_id if self.root_turn else None,
            "last_active_turn_gen_id": self.last_active_turn.gen_id if self.last_active_turn else None,
        }


class ReentrantWriterFairRWLock:
    def __init__(self):
        self._turnstile = threading.Lock()
        self._room_empty = threading.Lock()
        self._readers_lock = threading.Lock()

        self._readers = 0
        self._reader_owners = defaultdict(int)
        self._writer_owner = None
        self._writer_depth = 0

    def _tid(self):
        return threading.get_ident()

    def is_write_locked_by_current(self) -> bool:
        return self._writer_owner == self._tid()

    def is_read_locked_by_current(self) -> bool:
        return self._reader_owners.get(self._tid(), 0) > 0

    def can_acquire_write(self) -> bool:
        """True iff acquiring a write lock won't violate the no read->write upgrade rule."""
        tid = self._tid()
        return self._writer_owner == tid or self._reader_owners.get(tid, 0) == 0

    @contextmanager
    def read_lock(self):
        tid = self._tid()

        # writer reentrancy: writer may also read
        if self._writer_owner == tid:
            self._reader_owners[tid] += 1
            try:
                yield
            finally:
                self._reader_owners[tid] -= 1
            return

        # reader reentrancy: avoid turnstile to prevent self-deadlock
        if self._reader_owners.get(tid, 0) > 0:
            with self._readers_lock:
                self._readers += 1
                self._reader_owners[tid] += 1
            try:
                yield
            finally:
                with self._readers_lock:
                    self._readers -= 1
                    self._reader_owners[tid] -= 1
                    if self._readers == 0:
                        self._room_empty.release()
            return

        with self._turnstile:
            pass

        with self._readers_lock:
            self._readers += 1
            self._reader_owners[tid] += 1
            if self._readers == 1:
                self._room_empty.acquire()

        try:
            yield
        finally:
            with self._readers_lock:
                self._readers -= 1
                self._reader_owners[tid] -= 1
                if self._readers == 0:
                    self._room_empty.release()

    @contextmanager
    def write_lock(self):
        tid = self._tid()

        # reentrant write
        if self._writer_owner == tid:
            self._writer_depth += 1
            try:
                yield
            finally:
                self._writer_depth -= 1
            return

        # forbid upgrade
        if self._reader_owners.get(tid, 0) > 0:
            raise RuntimeError("read->write upgrade is not supported")

        self._turnstile.acquire()
        self._room_empty.acquire()
        self._writer_owner = tid
        self._writer_depth = 1

        try:
            yield
        finally:
            self._writer_depth -= 1
            if self._writer_depth == 0:
                self._writer_owner = None
                self._room_empty.release()
                self._turnstile.release()

def _with_read_lock(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._rw_lock.read_lock():
            return method(self, *args, **kwargs)
    return wrapper

def _with_write_lock(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._rw_lock.write_lock():
            return method(self, *args, **kwargs)
    return wrapper

class EngineSession:
    """
    Manages the state of a chat conversation using a tree of Turn objects.
    A session can contain multiple independent conversation trees.

    Locking contract (ReentrantWriterFairRWLock):
    - Read locks allow concurrent readers and are reentrant on the same thread.
    - Write locks are exclusive and reentrant on the same thread.
    - Read -> write upgrades are NOT supported and will raise.
    - Write -> read is supported (writer reentrancy for reads).
    - The lock protects session-owned collections and structure:
        - conversation list, turn tree links, Turn.cmd, commands_history.
    - Turn/Command content (e.g., Turn.data payloads) are NOT protected by the lock.
    """
    def __init__(self, name: str = f"untitled_{int(time.time())}", system_message: Optional[str] = None):

        self.id = str(uuid.uuid4()) # New unique ID for the session object itself
        self.name = name
        self.default_system_message: Optional[str] = system_message or "" # type: ignore
        self.conversations: List[ChatSession] = []
        self._last_gen_id_counter: int = 0
        self._last_cmd_id_counter: int = 0
        self._last_stack_id_counter: int = 0
        self.creation_timestamp: float = time.time()
        self.commands_history: List[Command] = []
        self.active_adapters_on_save: List[str] = []
        self.last_converation: int = 0
        self._rw_lock = ReentrantWriterFairRWLock()

    @contextmanager
    def read_lock(self):
        with self._rw_lock.read_lock():
            yield

    @contextmanager
    def write_lock(self):
        with self._rw_lock.write_lock():
            yield

    def can_acquire_write_lock(self) -> bool:
        """Return False if current thread holds a read lock (upgrade would error)."""
        return self._rw_lock.can_acquire_write()

    def is_read_locked_by_current(self) -> bool:
        return self._rw_lock.is_read_locked_by_current()

    def is_write_locked_by_current(self) -> bool:
        return self._rw_lock.is_write_locked_by_current()

    class _CustomEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (set, frozenset)):
                try:
                    return sorted(o)
                except TypeError:
                    return list(o)
            # The `is_dataclass` check handles ToolCall, ToolCallBlock, and any other dataclasses.
            # `asdict` recursively converts them to dictionaries, which is exactly what we need for JSON serialization.
            # If an object has a `to_dict` method, use it for serialization.
            # This applies to Turn, Command, ToolsScope, and ToolsView.
            to_dict_method = getattr(o, "to_dict", None)
            if callable(to_dict_method):
                return to_dict_method()  # type: ignore[misc]
            # For other dataclasses without a custom method, use asdict.
            if is_dataclass(o):
                return asdict(o)
            return super().default(o)

    @staticmethod
    def _is_tool_block_list(value: Any) -> bool:
        if not isinstance(value, list):
            return False
        if not value:
            return True
        return all(isinstance(item, ToolCallBlock) for item in value)

    @property
    def to_dict_prop(self) -> Dict[str, Any]:
        """Serializes the session to a dictionary."""

        with self._rw_lock.read_lock():
            # We need a temporary ID map to handle turns that might not have a gen_id yet (placeholders).
            all_turns = list(self._get_all_turns())
            node_to_temp_id = {node: f"temp_{i}" for i, node in enumerate(all_turns)}
            
            tool_block_lists: Dict[str, List[ToolCallBlock]] = {}
            tool_block_ref_by_list_id: Dict[int, str] = {}

            def _register_tool_block_list(list_obj: List[ToolCallBlock]) -> str:
                list_id = id(list_obj)
                existing = tool_block_ref_by_list_id.get(list_id)
                if existing:
                    return existing
                ref_id = f"tbl_{len(tool_block_lists) + 1}"
                tool_block_ref_by_list_id[list_id] = ref_id
                tool_block_lists[ref_id] = list_obj
                return ref_id

            serialized_nodes = {}
            for node in all_turns:
                temp_id = node_to_temp_id[node]
                node_dict = node.to_dict()
                if "data" in node_dict and isinstance(node_dict["data"], dict):
                    original_data = node.data or {}
                    data_copy = copy.deepcopy(node_dict["data"])

                    assistant = original_data.get("assistant")
                    if isinstance(assistant, dict):
                        tool_blocks = assistant.get("tool_blocks")
                        if self._is_tool_block_list(tool_blocks):
                            ref_id = _register_tool_block_list(tool_blocks)
                            if isinstance(data_copy.get("assistant"), dict):
                                data_copy["assistant"].pop("tool_blocks", None)
                                data_copy["assistant"]["tool_blocks_ref"] = ref_id

                    tool_results = original_data.get("tool_results")
                    if self._is_tool_block_list(tool_results):
                        ref_id = _register_tool_block_list(tool_results)
                        data_copy.pop("tool_results", None)
                        data_copy["tool_results_ref"] = ref_id

                    node_dict["data"] = data_copy
                # Replace parent object with its temporary ID for serialization
                if node.parent:
                    node_dict["_parent_gen_id"] = node_to_temp_id.get(node.parent)
                # Replace child objects with their temporary IDs
                if "turns" in node_dict:
                    node_dict["turns"] = [node_to_temp_id.get(child) for child in node.turns if child in node_to_temp_id] # type: ignore
                serialized_nodes[temp_id] = node_dict

            # Build chat_sessions data and include a reference to the node map so
            # that deserialization can locate the ChatSession root node reliably.
            chat_sessions_serialized: List[Dict[str, Any]] = []
            for c in self.conversations:
                cs_dict = c.to_dict()
                root = getattr(c, "root_turn", None)
                if root in node_to_temp_id:
                    cs_dict["root_turn_temp_id"] = node_to_temp_id.get(root)
                # Prefer to store the gen_id if available for easier lookup by older loaders
                if getattr(root, "gen_id", None):
                    cs_dict["root_turn_gen_id"] = getattr(root, "gen_id")
                last_active = getattr(c, "last_active_turn", None)
                if last_active in node_to_temp_id:
                    cs_dict["last_active_turn_temp_id"] = node_to_temp_id.get(last_active)
                if getattr(last_active, "gen_id", None):
                    cs_dict["last_active_turn_gen_id"] = getattr(last_active, "gen_id")
                chat_sessions_serialized.append(cs_dict)

            data = {
                "id": self.id,
                "name": self.name,
                "schema_version": "4.5", # New schema version for stack_id support
                "conversation_roots": [node_to_temp_id.get(c.root_turn) for c in self.conversations if c.root_turn in node_to_temp_id], # type: ignore
                "nodes": serialized_nodes,
                "tool_block_lists": tool_block_lists,
                "last_gen_id_counter": self._last_gen_id_counter,
                "last_cmd_id_counter": self._last_cmd_id_counter,
                "last_stack_id_counter": self._last_stack_id_counter,
                "chat_sessions": chat_sessions_serialized,
                "creation_timestamp": self.creation_timestamp,
                "commands_history": [cmd.to_dict() for cmd in self.commands_history],
                "last_converation": self.last_converation,
            }
            return data

    @_with_read_lock
    def serialize(self, file_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """
        Serializes the session object.
        - If file_path is provided, writes to the file and returns None.
        - If file_path is None, returns the session as a JSON string.
        """
        data_to_serialize = self.to_dict_prop
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_serialize, f, indent=2, cls=self._CustomEncoder)
            return None
        else:
            return json.dumps(data_to_serialize, indent=2, cls=self._CustomEncoder)

    async def async_serialize(self, file_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        return await asyncio.to_thread(self.serialize, file_path)

    @classmethod
    def deserialize(cls, source: Union[str, Path]) -> "EngineSession":
        """
        Deserializes a session object from a JSON file path or a JSON string.
        """
        if isinstance(source, str) and source.strip().startswith('{'):
            try:
                data = json.loads(source)
                return cls.from_dict(data)
            except json.JSONDecodeError as e:
                raise ValueError("The provided string is not valid JSON.") from e
        else:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineSession":
        """Deserializes a dictionary into a ChatSession object."""
        session_name = data.get("name", f"loaded_{int(time.time())}")
        session = cls(name=session_name)
        session.id = data.get("id", str(uuid.uuid4()))
        session.creation_timestamp = data.get("creation_timestamp", time.time())
        session.active_adapters_on_save = data.get("active_adapters_on_save", [])
        session._last_gen_id_counter = data.get("last_gen_id_counter", 0)
        session._last_cmd_id_counter = data.get("last_cmd_id_counter", 0)
        session._last_stack_id_counter = data.get("last_stack_id_counter", 0)
        raw_commands_history = data.get("commands_history", [])
        if not isinstance(raw_commands_history, list):
            raw_commands_history = []
        session.last_converation = data.get("last_converation", 0)

        schema_version = data.get("schema_version")

        # Handle malformed 4.3 files that are missing the 'nodes' key.
        if schema_version in ["4.3", "4.4", "4.5"] and "nodes" not in data:
            #print(f"Warning: Session file '{session.name}' has schema {schema_version} but is missing the 'nodes' dictionary. Loading as an empty session.")
            return session

        if schema_version not in ["4.2", "4.3", "4.4", "4.5"] or "nodes" not in data:
            raise ValueError("Session file is not in a supported schema version (4.2, 4.3, 4.4, 4.5) or is corrupted.")

        # 1. Create all Turn and Command objects from the flat dictionary
        nodes_by_temp_id: Dict[str, Union[Turn, Command]] = {}
        for node_id, node_data in data["nodes"].items():
            if "cmd_type" in node_data:
                nodes_by_temp_id[node_id] = Command.from_dict(node_data)
            else:
                nodes_by_temp_id[node_id] = Turn.from_dict(node_data)

        # 2. Re-link object references (parents and children)
        gen_map: Dict[str, Turn] = {}
        for maybe_turn in nodes_by_temp_id.values():
            if isinstance(maybe_turn, Turn):
                gid = getattr(maybe_turn, "gen_id", None)
                if gid is not None:
                    gen_map[gid] = maybe_turn

        def _resolve_node_by_identifier(identifier: Optional[str]) -> Optional[Union[Turn, Command]]:
            if not identifier: return None
            node = nodes_by_temp_id.get(identifier)
            if node is not None: return node
            return gen_map.get(identifier)

        for temp_id, node in nodes_by_temp_id.items():
            raw_data = data.get("nodes", {}).get(temp_id, {})
            if isinstance(node, Turn):
                parent_id = raw_data.get("_parent_gen_id")
                parent_node = _resolve_node_by_identifier(parent_id)
                if isinstance(parent_node, Turn):
                    node.parent = parent_node

                child_ids = raw_data.get("turns", []) or []
                node.turns = [child for child_id in child_ids if (child := _resolve_node_by_identifier(child_id)) and isinstance(child, Turn)]

        tool_block_lists = data.get("tool_block_lists", {}) or {}
        tool_block_refs: Dict[str, List[ToolCallBlock]] = {}
        for ref_id, raw_list in tool_block_lists.items():
            if not isinstance(raw_list, list):
                continue
            rehydrated_list: List[ToolCallBlock] = []
            for item in raw_list:
                if isinstance(item, ToolCallBlock):
                    rehydrated_list.append(_rehydrate_tool_call_block(item))
                elif isinstance(item, dict):
                    try:
                        rehydrated_list.append(ToolCallBlock.from_dict(item))
                    except Exception:
                        continue
            tool_block_refs[ref_id] = rehydrated_list

        if tool_block_refs:
            for node in nodes_by_temp_id.values():
                if not isinstance(node, Turn):
                    continue
                payload = node.data
                if not isinstance(payload, dict):
                    continue
                assistant = payload.get("assistant")
                if isinstance(assistant, dict):
                    ref_id = assistant.get("tool_blocks_ref")
                    if ref_id and ref_id in tool_block_refs:
                        assistant["tool_blocks"] = tool_block_refs[ref_id]
                        assistant.pop("tool_blocks_ref", None)
                    if "tool_blocks" in assistant and isinstance(assistant.get("tool_blocks"), list):
                        assistant["tool_blocks"] = _rehydrate_tool_payload_list(assistant["tool_blocks"])

                ref_id = payload.get("tool_results_ref")
                if ref_id and ref_id in tool_block_refs:
                    payload["tool_results"] = tool_block_refs[ref_id]
                    payload.pop("tool_results_ref", None)
                if "tool_results" in payload and isinstance(payload.get("tool_results"), list):
                    payload["tool_results"] = _rehydrate_tool_payload_list(payload["tool_results"])

        # 3. Reconstruct ChatSession objects
        for cs_data in data.get("chat_sessions", []):
            root_gen_id = cs_data.pop("root_turn_gen_id", None)
            root_temp_id = cs_data.pop("root_turn_temp_id", None)
            last_active_gen_id = cs_data.pop("last_active_turn_gen_id", None)
            last_active_temp_id = cs_data.pop("last_active_turn_temp_id", None)
            title = cs_data.pop("title", None)

            root_turn_candidate = _resolve_node_by_identifier(root_gen_id) or _resolve_node_by_identifier(root_temp_id)
            root_turn = root_turn_candidate if isinstance(root_turn_candidate, Turn) else Turn(turn_type=None)

            # Handle migration for 4.2 -> 4.3 key rename
            inference_defaults_data = cs_data.pop("inference_defaults", cs_data.pop("inference_params", {}))
            inference_defaults = InferenceParams(**inference_defaults_data)
            parser_profile_data = cs_data.pop("parser_profile", None)

            cs = ChatSession(
                root_turn=root_turn,
                inference_defaults=inference_defaults,
                title=title,
                parser_profile=parser_profile_data,
                **cs_data,
            )
            session._ensure_conversation_root_initialized(cs)
            session.conversations.append(cs)

            last_active_candidate = _resolve_node_by_identifier(last_active_gen_id) or _resolve_node_by_identifier(last_active_temp_id)
            if isinstance(last_active_candidate, Turn):
                cs.last_active_turn = last_active_candidate

            if cs.toolbox and "tools_access" in cs.initial_params and isinstance(cs.initial_params["tools_access"], dict):
                cs.initial_params["tools_access"] = cs.toolbox.create_access(scopes=[], label="rehydrated")
        
        # 4. Rebuild commands_history using Turn.cmd references when possible.
        all_turns_by_id = {t.gen_id: t for t in nodes_by_temp_id.values() if isinstance(t, Turn) and t.gen_id}
        cmd_by_id: Dict[str, Command] = {}
        for turn in all_turns_by_id.values():
            for cmd in getattr(turn, "cmd", []) or []:
                if cmd.gen_id:
                    cmd_by_id[cmd.gen_id] = cmd

        rebuilt_history: List[Command] = []
        for cmd_data in raw_commands_history:
            gen_id = cmd_data.get("gen_id") if isinstance(cmd_data, dict) else None
            cmd = cmd_by_id.get(gen_id) if gen_id else None
            if cmd is None:
                cmd = Command.from_dict(cmd_data)
            if isinstance(cmd.parent, str) and (turn := all_turns_by_id.get(cmd.parent)):
                cmd.parent = turn
            rebuilt_history.append(cmd)
        session.commands_history = rebuilt_history

        return session
    
    def get_request_id(self, turn: Turn, prefix: str = "") -> str:
        """Generates a unique request ID for an API call associated with a turn."""
        return f"{prefix}_{os.getpid()}_{time.time()}_{turn.gen_id_or_parent}"

    @_with_read_lock
    def commands_history_snapshot(self) -> List[Command]:
        return list(self.commands_history)

    @_with_write_lock
    def remove_commands_from_history(self, removed: Sequence[Command]) -> int:
        if not removed:
            return 0
        removed_ids = {id(cmd) for cmd in removed if cmd is not None}
        if not removed_ids:
            return 0
        before = len(self.commands_history)
        self.commands_history = [cmd for cmd in self.commands_history if id(cmd) not in removed_ids]
        return before - len(self.commands_history)

    @_with_write_lock
    def remove_command_from_history(self, cmd: Optional[Command]) -> bool:
        if cmd is None:
            return False
        try:
            self.commands_history.remove(cmd)
            return True
        except ValueError:
            return False

    @_with_write_lock
    def append_command_to_history(self, cmd: Optional[Command]) -> None:
        if cmd is None:
            return
        self.commands_history.append(cmd)

    @_with_read_lock
    def user_turns_count(self, start_node: Turn) -> int:
            """
            Returns the number of non-placeholder and not archived user turns
            from the given node to the root of the conversation tree.
            """
            count = 0
            current: Optional[Turn] = start_node
            while current is not None:
                if current.turn_type == Turn.CHAT and current.data and "user" in current.data:
                    count += 1
                current = current.parent
            return count

    @_with_read_lock
    def count_nodes_in_tree(self, start_node: Turn) -> int:
        """Counts all Turn nodes in the tree starting from start_node."""
        if not start_node:
            return 0
        count = 0
        q: List[Turn] = [start_node]
        visited = {start_node}
        while q:
            current = q.pop(0)
            count += 1
            for child in current.turns:
                if child not in visited:
                    q.append(child) # type: ignore
                    visited.add(child)
        return count

    def _get_all_turns(self, chat_session: Optional["ChatSession"] = None) -> Generator[Turn, None, None]:
        """A generator that yields all turns in the session tree.

        If `chat_session` is provided, traversal is scoped to that session's root only.
        If `chat_session` is None, traverse all conversations in the session (backwards compatible).
        Traversal is breadth-first and guards against cycles.
        """
        with self._rw_lock.read_lock():
            roots: List[Turn]
            if chat_session is None:
                roots = [cs.root_turn for cs in self.conversations]
            else:
                roots = [chat_session.root_turn]

            for conv_root in roots:
                if conv_root is None:
                    continue
                q: List[Turn] = [conv_root]
                visited = {conv_root}
                while q:
                    current = q.pop(0)
                    yield current
                    for child in getattr(current, "turns", []) or []:
                        if child not in visited:
                            q.append(child)  # type: ignore
                            visited.add(child)

    @_with_read_lock
    def _find_turn_by_anchor_name(self, anchor_name: str, chat_session: Optional["ChatSession"] = None) -> Optional[Turn]:
        """Finds the first turn matching a given try_out anchor name."""
        if not anchor_name:
            return None
        for turn in self._get_all_turns(chat_session):
            try_meta = turn.data.get("$try_out", {})
            if isinstance(try_meta, dict) and try_meta.get("anchor") == anchor_name:
                return turn
        return None

    @_with_read_lock
    def get_turn_by_gen_id(self, gen_id: str, chat_session: Optional["ChatSession"] = None) -> Optional[Turn]:
        """Finds a turn by its unique gen_id by traversing the tree.

        Args:
            gen_id: generation id to search for (e.g., 'g_1').
            chat_session: optional ChatSession to scope the search. If None, searches all conversations.
        """
        if not gen_id:
            return None
        for turn in self._get_all_turns(chat_session):
            if getattr(turn, "gen_id", None) == gen_id:
                return turn
        return None

    @_with_read_lock
    def _get_chat_session_for_turn(self, turn: Turn) -> Optional[ChatSession]:
        for conv in self.conversations:
            # Check if active_turn is the root of this conversation
            if conv.root_turn is turn:
                return conv
            # Check if active_turn is a descendant of the root of this conversation
            q: List[Turn] = [conv.root_turn]
            visited = {conv.root_turn}
            found = False
            while q:
                current = q.pop(0)
                if current is turn:
                    return conv
                for child in getattr(current, "turns", []) or []:
                    if child not in visited:
                        q.append(child) # type: ignore
                        visited.add(child)
        return None

    @_with_read_lock
    def get_scope_turns(self, start_turn: Turn, suppress_auto: bool = False, history_only: bool = False) -> List[Turn]:
        """
        Returns a list of turns within the scope of start_turn, ordered by command
        history for turns that appear in commands_history. Turns without commands
        are ordered after their parents, using TURN command metadata to place siblings.
        If history_only is True, returns only the turns that have commands associated with them.
        """
        q = [start_turn]
        scope_turns_set = {start_turn}
        # BFS to get all descendants
        head = 0
        while head < len(q):
            current = q[head]
            head += 1
            for child in getattr(current, 'turns', []):
                if child not in scope_turns_set:
                    scope_turns_set.add(child)
                    q.append(child)

        def _is_suppressed_auto(turn: Turn) -> bool:
            if getattr(turn, "is_auto", False):
                return True
            return False

        if suppress_auto:
            scope_turns_set = {t for t in scope_turns_set if not _is_suppressed_auto(t)}
        turns_by_id: Dict[str, Turn] = {
            t.gen_id: t for t in scope_turns_set if getattr(t, "gen_id", None)
        }

        cmd_turns_ordered: List[Turn] = []
        seen_turn_ids: set[str] = set()
        for cmd in self.commands_history:
            parent_turn = None
            if isinstance(cmd.parent, Turn):
                parent_turn = cmd.parent
            elif isinstance(cmd.parent, str):
                parent_turn = turns_by_id.get(cmd.parent)
            
            if parent_turn and parent_turn in scope_turns_set and parent_turn.gen_id not in seen_turn_ids:
                cmd_turns_ordered.append(parent_turn)
                if parent_turn.gen_id:
                    seen_turn_ids.add(parent_turn.gen_id)

        if history_only:
            # Ensure start_turn is always the first item in the results
            if start_turn in cmd_turns_ordered:
                cmd_turns_ordered.remove(start_turn)
            cmd_turns_ordered.insert(0, start_turn)
            return cmd_turns_ordered

        # Index turns that appear in command history by their first occurrence.
        cmd_index_map: Dict[Turn, int] = {}
        for idx, turn in enumerate(cmd_turns_ordered):
            cmd_index_map[turn] = idx

        # Build a stable tree-order index for deterministic tie-breaking.
        tree_order: List[Turn] = []
        visited_tree: set[Turn] = set()
        queue: List[Turn] = [start_turn]
        while queue:
            current = queue.pop(0)
            if current in visited_tree or current not in scope_turns_set:
                continue
            visited_tree.add(current)
            tree_order.append(current)
            for child in getattr(current, "turns", []) or []:
                if child in scope_turns_set and child not in visited_tree:
                    queue.append(child)
        tree_index_map = {t: i for i, t in enumerate(tree_order)}

        # Capture sibling ordering using command-history when available, otherwise tree order.
        child_order_by_parent: Dict[Turn, List[Turn]] = {}
        for parent_turn in scope_turns_set:
            children = [child for child in getattr(parent_turn, "turns", []) or [] if child in scope_turns_set]
            if not children:
                continue
            def _child_key(child: Turn) -> tuple:
                cmd_index = cmd_index_map.get(child, 10**9)
                tree_index = tree_index_map.get(child, 10**9)
                return (cmd_index, tree_index, child.sort_id)
            children_sorted = sorted(children, key=_child_key)
            child_order_by_parent[parent_turn] = children_sorted

        # Topological ordering with constraints:
        # - parent precedes child
        # - sibling order derived from TURN command metadata (if available)
        # - among eligible nodes, prefer command-history order
        nodes = list(scope_turns_set)
        edges: Dict[Turn, List[Turn]] = {t: [] for t in nodes}
        indegree: Dict[Turn, int] = {t: 0 for t in nodes}

        for t in nodes:
            parent = getattr(t, "parent", None)
            if parent in scope_turns_set:
                edges[parent].append(t)
                indegree[t] += 1

        for parent, children in child_order_by_parent.items():
            for prev, curr in zip(children, children[1:]):
                edges[prev].append(curr)
                indegree[curr] += 1

        def _priority_key(turn: Turn) -> tuple:
            cmd_index = cmd_index_map.get(turn, 10**9)
            if turn is start_turn:
                cmd_index = -1
            tree_index = tree_index_map.get(turn, 10**9)
            return (cmd_index, tree_index, turn.sort_id)

        heap: List[tuple] = []
        for t in nodes:
            if indegree[t] == 0:
                heapq.heappush(heap, (_priority_key(t), t))

        ordered: List[Turn] = []
        while heap:
            _, t = heapq.heappop(heap)
            if t in ordered:
                continue
            ordered.append(t)
            for nxt in edges.get(t, []):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(heap, (_priority_key(nxt), nxt))

        if len(ordered) != len(scope_turns_set):
            # Fallback to tree order for any leftover nodes.
            for t in tree_order:
                if t not in ordered and t in scope_turns_set:
                    ordered.append(t)
            for t in scope_turns_set:
                if t not in ordered:
                    ordered.append(t)

        # Ensure start_turn is always the first item in the results
        if start_turn in ordered:
            ordered.remove(start_turn)
        ordered.insert(0, start_turn)

        return ordered

    @_with_read_lock
    def get_scope_commands(self, start_turn: Turn, suppress_auto: bool = False, include_logs: bool = False) -> List[Command]:
        scope_turns = self.get_scope_turns(start_turn, suppress_auto)
        scope_turn_ids = {t.gen_id for t in scope_turns if t.gen_id}

        scope_commands = []
        for cmd in self.commands_history:
            if not include_logs and cmd.cmd_type == Command.LOG:
                continue
            
            parent_id = None
            if isinstance(cmd.parent, Turn):
                parent_id = cmd.parent.gen_id
            elif isinstance(cmd.parent, str):
                parent_id = cmd.parent
            
            if parent_id in scope_turn_ids:
                scope_commands.append(cmd)
                
        return scope_commands

    def add_update_metrics(self, target_turn: Turn, metrics: Dict[str, Any]) -> None:
        """
        Adds or updates metrics for a given Turn object.
        """
        if not target_turn:
            raise ValueError("Cannot add/update metrics without a valid target turn.")
        target_turn.metrics.update(metrics)


    @_with_write_lock
    def _add_command(self, new_command: Command, current_turn: Turn) -> Turn:
        """Helper to add a Command node and set it as the active turn."""
        if not isinstance(current_turn, Turn):
            raise ValueError("Cannot add a command without a valid active turn.")

        # A turn is considered "closed" if it has an assistant response.
        # A turn with no data at all is also considered "open" (it's a placeholder).
        is_closed = self._is_closed_turn(current_turn)

        if is_closed:
            # The current turn is complete. Find or create a placeholder for the next turn.
            # A placeholder is a turn with turn_type of None.
            existing_placeholder = next((t for t in current_turn.turns if t.turn_type is None and not t.data), None)

            if existing_placeholder:
                # An open placeholder already exists, add the command to it.
                self._add_command_to_turn(new_command, existing_placeholder)
                return existing_placeholder

            # No suitable placeholder exists, so create a new one.
            # This is the correct place to create a new turn for a command following a completed turn.
            new_placeholder_turn = self._add_chat_turn(current_turn, Turn(turn_type=None))
            self._add_command_to_turn(new_command, new_placeholder_turn)
            return new_placeholder_turn

        else:
            # The current turn is open. Only add a LOG command if it's a true placeholder.
            # Do not add logs to a turn that has a user prompt but is awaiting a response.
            if new_command.cmd_type == Command.LOG and not current_turn.IsEmpty:
                raise ValueError("Cannot add a log command to a turn that is awaiting a response.")
            self._add_command_to_turn(new_command, current_turn)
            return current_turn

    @_with_write_lock
    def _add_chat_turn(self, parent_turn:Turn, new_turn: Turn) -> Turn:
        """Helper to add a CHAT turn and update both active turn pointers."""
        # A gen_id is assigned only when a turn becomes non-placeholder.
        if new_turn.turn_type is not None:
            self._ensure_turn_has_gen_id(new_turn)

        # If we're adding a non-placeholder and the parent already has a second+ placeholder
        # (a try-out slot), reuse it instead of appending.
        if new_turn.turn_type is not None and parent_turn and parent_turn.turns:
            # Prefer the first placeholder among indices ≥ 1
            for idx in range(1, len(parent_turn.turns)):
                slot = parent_turn.turns[idx]
                if self._is_convertible_placeholder(slot):
                    # Repurpose this placeholder in place
                    slot.turn_type = new_turn.turn_type
                    slot.data = new_turn.data or {}
                    slot.is_archived = getattr(new_turn, "is_archived", False)
                    slot.do_continue = getattr(new_turn, "do_continue", False)
                    slot.was_truncated = getattr(new_turn, "was_truncated", False)
                    slot.metrics = getattr(new_turn, "metrics", {}) or {}
                    # Keep main_thread flag as it semantically belongs to the slot
                    slot.gen_id = new_turn.gen_id
                    slot.is_auto = new_turn.is_auto
                    return slot

        # Otherwise, append normally
        new_turn.parent = parent_turn
        if parent_turn:
            parent_turn.turns.append(new_turn)
        return new_turn

    @_with_write_lock
    def _ensure_turn_has_gen_id(self, turn: Turn) -> None:
        """
        Central helper for assigning turn gen_ids.
        Ensures every gen_id comes from the single session counter.
        """
        if not turn.gen_id:
            self._last_gen_id_counter += 1
            turn.gen_id = f"g_{self._last_gen_id_counter}"

    @_with_write_lock
    def _mark_placeholder_anchor(self, turn: Turn, new_type: str) -> Turn:
        """
        Converts an empty placeholder into a structural anchor (HOLD/RESERVED) and
        ensures it has a gen_id so its commands participate in state collection.
        """
        if new_type not in (Turn.HOLD, Turn.RESERVED):
            raise ValueError(f"Unsupported placeholder anchor type '{new_type}'.")
        if turn.turn_type is None:
            turn.turn_type = new_type
            self._ensure_turn_has_gen_id(turn)
        elif turn.turn_type == new_type and not turn.gen_id:
            self._ensure_turn_has_gen_id(turn)
        return turn

    @_with_write_lock
    def _promote_placeholder_parent_if_needed(self, turn: Optional[Turn]) -> None:
        """
        Ensure placeholders that already gained children have stable ids for command replay.
        Do not change them into RESERVED anchors; keep them convertible (HOLD/None).
        """
        if turn and self._is_convertible_placeholder(turn) and getattr(turn, "turns", []):
            self._ensure_turn_has_gen_id(turn)

    # ---------------------------
    # TRY OUT & BRANCH MANAGEMENT
    # ---------------------------
    @_with_write_lock
    def add_try_out(
        self,
        base: "Turn",
        *,
        keep_in_main: bool = False,
        convert_existing: bool = False,
    ) -> Tuple["Turn", "Turn"]:
        """
        Create a try-out placeholder per the separated semantics.

        Returns:
            (main_branch_turn, tryout_placeholder)

        Rules:
        • If `base` is a placeholder and NOT the root: add both the main continuation
            placeholder and the try-out as SIBLINGS under base.parent (so `base` remains a placeholder).
        • If `base` is the ROOT placeholder: go DOWN — create first child (main placeholder)
            and second child (try-out placeholder) under the root. Root can remain a placeholder.
        • If `base` is NOT a placeholder: go DOWN — ensure/create first child as the main continuation,
            then create a try-out as the second+ child.
        
        If convert_existing is True:
        • Find the first non-placeholder turn walking up from `base`.
        • Require that turn to be closed (has a response); otherwise raise.
        • If it has commands, move them into a RESERVED placeholder inserted above it.
        • If it already has siblings, wrap it with a RESERVED placeholder as well.
        • Ensure its effective parent has a main placeholder as the first child and
          the found turn as the second child (the try-out).
        """
        if convert_existing:
            if base is None:
                raise ValueError("add_try_out(convert_existing): base must not be None.")

            def _is_descendant(node: Optional["Turn"], ancestor: Optional["Turn"]) -> bool:
                if not node or not ancestor:
                    return False
                current = node
                while current:
                    if current is ancestor:
                        return True
                    current = getattr(current, "parent", None)
                return False

            found = base
            while found and getattr(found, "IsPlaceholderLike", False):
                found = found.parent

            if not found:
                raise ValueError("add_try_out(convert_existing): no non-placeholder turn found from base.")

            is_closed = self._is_closed_turn(found)
            if not is_closed:
                raise ValueError(
                    f"add_try_out(convert_existing): target turn '{self.get_display_id(found)}' is not closed."
                )

            effective_parent = found.parent
            if not effective_parent:
                raise ValueError("add_try_out(convert_existing): cannot convert a root turn into a try-out.")

            siblings = list(getattr(effective_parent, "turns", []) or [])
            has_siblings = any(child is not found for child in siblings)
            non_trivial_cmds = [
                cmd for cmd in list(getattr(found, "cmd", []) or [])
                if cmd.cmd_type not in (Command.LOG, Command.TURN)
            ]
            has_cmds = bool(non_trivial_cmds)

            if has_cmds or has_siblings:
                wrapper = Turn(turn_type=Turn.RESERVED, metadata={})
                self._ensure_turn_has_gen_id(wrapper)

                # Replace found with wrapper in the parent's child list.
                try:
                    idx = effective_parent.turns.index(found)
                except ValueError:
                    idx = None
                if idx is not None:
                    effective_parent.turns[idx] = wrapper
                else:
                    effective_parent.turns.append(wrapper)
                wrapper.parent = effective_parent

                # Move commands onto the wrapper in original order.
                if has_cmds:
                    retained: List[Command] = []
                    for cmd in list(found.cmd):
                        if cmd.cmd_type in (Command.LOG, Command.TURN):
                            retained.append(cmd)
                        else:
                            cmd.parent = wrapper
                            wrapper.cmd.append(cmd)
                    found.cmd = retained

                # Reparent the found turn under the wrapper.
                found.parent = wrapper
                wrapper.turns = [found]
                effective_parent = wrapper

            # Normalize children so the main placeholder is first and found is second.
            kids = list(getattr(effective_parent, "turns", []) or [])
            if found in kids:
                kids.remove(found)

            main_node: Optional[Turn] = None
            if kids:
                first = kids[0]
                if first.IsPlaceholderLike:
                    main_node = first
                else:
                    main_node = Turn(turn_type=None, metadata={})
                    main_node.main_thread = True
                    main_node.parent = effective_parent
                    kids.insert(0, main_node)
            else:
                main_node = Turn(turn_type=None, metadata={})
                main_node.main_thread = True
                main_node.parent = effective_parent
                kids.insert(0, main_node)

            if kids[0] is not main_node:
                if main_node in kids:
                    kids.remove(main_node)
                kids.insert(0, main_node)

            main_node.main_thread = True
            found.main_thread = bool(keep_in_main)
            found.parent = effective_parent

            if found in kids:
                kids.remove(found)
            kids.insert(1 if len(kids) >= 1 else 0, found)

            effective_parent.turns = kids
            tryout_leaf = found
            if getattr(base, "IsPlaceholderLike", False) and _is_descendant(base, found):
                tryout_leaf = base
            else:
                tryout_leaf = self.get_last_turn_on_branch(found)
            return (main_node, tryout_leaf)

        def _needs_hold_conversion(t: "Turn") -> bool:
            if t.turn_type is not None:
                return False
            return any(
                cmd.cmd_type in {Command.STATE_CHANGE, Command.PARAM_CHANGE, Command.ADAPTERS_STATE}
                for cmd in getattr(t, "cmd", [])
            )

        force_grow_down = _needs_hold_conversion(base)
        if force_grow_down:
            base = self._mark_placeholder_anchor(base, Turn.HOLD)

        # Helper: ensure the "first child" placeholder exists (or use real node if already present)
        def _ensure_first_child_main(p: "Turn") -> "Turn":
            if p.turns:
                first = p.turns[0]
                # Make sure it's marked as main continuation
                setattr(first, "main_thread", True)
                return first
            # Create a placeholder first child
            main_ph = Turn(turn_type=None, metadata={})
            setattr(main_ph, "main_thread", True)
            self._add_chat_turn(p, main_ph)
            return main_ph

        # Helper: create a new try-out placeholder under parent p
        def _create_tryout_under(p: "Turn") -> "Turn":
            tr = Turn(turn_type=None, metadata={})
            setattr(tr, "main_thread", bool(keep_in_main))
            self._add_chat_turn(p, tr)
            return tr

        # CASE 1: base is a placeholder
        if self._is_convertible_placeholder(base) and not force_grow_down:
            # Subcase: root placeholder → go DOWN under root, root remains placeholder
            if base.parent is None:
                parent = base
                # First child = main continuation, second child = try-out
                main_node = _ensure_first_child_main(parent)
                tryout_node = _create_tryout_under(parent)
                return (main_node, tryout_node)

            # Subcase: non-root placeholder → the placeholder *itself* is the main continuation.
            # It must ALREADY be the first child by construction; otherwise, the parent
            # has been advanced directly (e.g., by a BATCH) and this call is invalid.
            parent = base.parent
            setattr(base, "main_thread", True)
            if parent.turns and parent.turns[0] is not base:
                # Allow out-of-order placeholders (e.g., after /trim) to branch without failing.
                # Keep it marked as main, and proceed with the try_out creation.
                try:
                    parent.turns.index(base)
                except ValueError:
                    pass
            tryout_node = _create_tryout_under(parent)
            return (base, tryout_node)

        # CASE 2: base is not a placeholder → go DOWN under `base`
        parent = base
        main_node = _ensure_first_child_main(parent)
        tryout_node = _create_tryout_under(parent)
        return (main_node, tryout_node)

    @_with_write_lock
    def promote_tryouts_to_main(self, parent: "Turn", promote_list: List["Turn"]) -> List["Turn"]:
        """
        Promote one or more try-out branches under `parent` to main-thread status.

        Args:
            parent: The common parent node whose children will be adjusted.
            promote_list: A list of Turn objects (try-out roots) that should become main threads.
                        An empty list means "un-promote all" (clear main_thread flags).

        Returns:
            The list of successfully promoted Turn nodes (subset of promote_list).
        """
        if not parent:
            raise ValueError("promote_tryouts_to_main: parent must not be None")

        kids: List["Turn"] = list(getattr(parent, "turns", []))
        if not kids:
            return []

        def _key(t: "Turn") -> str:
            return getattr(t, "gen_id", f"obj:{id(t)}")

        promote_keys = {_key(x) for x in promote_list}
        promoted: List["Turn"] = []

        # --- NEW: Context-aware promotion logic ---

        if parent.turn_type == Turn.BATCH:
            # Case 1: The parent is a BATCH node.
            is_direct_continuation_batch = self._is_first_child(parent)
            # Ensure the structural main child stays marked as main.
            if kids:
                kids[0].main_thread = True
            for i, child in enumerate(kids):
                is_promoted = _key(child) in promote_keys
                # First child always remains main; additional children are opt-in via promote_list.
                if i == 0:
                    child.main_thread = True
                    promoted.append(child)
                    continue
                child.main_thread = is_promoted
                if is_promoted:
                    promoted.append(child)
            
            # For a BATCH node, its main_thread status reflects if it contains a main child.
            parent.main_thread = bool(kids)

        else:
            # Case 2: Standard try-out branch (parent is not a BATCH).
            # The `add_try_out` method is responsible for setting the main_thread flag on the
            # main placeholder (the first child). This method should only adjust the flags
            # of the try-out siblings based on the promote_list.
            for child in kids:
                is_promoted = _key(child) in promote_keys
                if is_promoted and not child.main_thread:
                    child.main_thread = True
                    promoted.append(child)
                elif not is_promoted and child.main_thread and child is not kids[0]:
                    # Demote other try-out siblings that are not being promoted.
                    # Crucially, do not demote the main placeholder (kids[0]).
                    child.main_thread = False
            
            # For a standard try-out, do not mark the parent. Its status is independent.

        return promoted


    @_with_write_lock
    def _add_user_with_parent(self, content: str, parent_turn: Turn, archived: bool = False, **kwargs) -> Turn:
        """
        Private helper to add a user message as a child of a specific current turn.
        This is the core method for creating user turns in batch or forked scenarios.

        Args:
            content: The user's message content.
            parent_turn: The parent Turn object.
            archived: Whether the turn should be archived.
            **kwargs: Additional metadata for the message.

        Returns:
            The newly created Turn.
        """
        if not parent_turn:
            raise ValueError("Cannot add user message: a valid parent_turn must be provided.")

        keep_in_main = kwargs.pop("keep_in_main", False)
        message = {"role": "user", "content": content}
        message.update(kwargs)
        new_turn = Turn(
            turn_type=Turn.CHAT,
            metadata={"user": message},
            is_archived=archived,
            main_thread=keep_in_main,
        )
        # Add the new turn as a child of the specified parent and log the command.
        added_turn = self._add_chat_turn(parent_turn, new_turn)
        new_command = Command(Command.TURN, {"prompt": f"[{added_turn.gen_id}]", "method": "_add_user_with_parent"})
        self._annotate_turn_command(new_command, added_turn)
        self._add_command_to_turn(new_command, added_turn)
        return added_turn
    
    @_with_write_lock
    def add_user(self, content: str, current_turn: Turn, archived: bool = False, **kwargs) -> Turn:
        """
        Adds a user message to the current conversational flow. This method handles
        the logic for interactive chat, populating placeholders or creating new turns
        as needed. For batch or direct parent assignment, use `_add_user_with_parent`.

        Args:
            parent_turn: If provided, the new turn will be a child of this turn,
                         overriding the default behavior of using the current current turn.
        """
        message = {"role": "user", "content": content}
        message.update(kwargs)

        if current_turn.IsStructural:
            # If the current turn is a FORK or BATCH hub, always add the new user message
            # as a new child turn (a new fork/branch).
            return self._add_user_with_parent(
                content,
                parent_turn=current_turn,
                archived=archived,
                **kwargs,
            )

        # take over the placeholder (open or HOLD)
        elif self._is_convertible_placeholder(current_turn):
            # The active turn is an empty placeholder. Populate it with the user message.
            current_turn.data["user"] = message
            # Set turn_type and assign gen_id now that it's a real turn.
            if current_turn.turn_type in (None, Turn.HOLD):
                current_turn.turn_type = Turn.CHAT
                # Assign gen_id via central helper since _add_chat_turn was already called.
                self._ensure_turn_has_gen_id(current_turn)
            new_command = Command(Command.TURN, {"prompt": f"[{current_turn.gen_id}]", "method": "add_user"})
            self._annotate_turn_command(new_command, current_turn)
            self._add_command_to_turn(new_command, current_turn)
            current_turn.is_archived = archived
            return current_turn
        elif current_turn.turn_type == Turn.RESERVED:
            # RESERVED anchors cannot be repurposed; grow a real child turn instead.
            keep_in_main = kwargs.pop(
                "keep_in_main",
                bool(getattr(current_turn, "main_thread", False) or getattr(current_turn, "root_context", False) or not current_turn.turns),
            )
            return self._add_user_with_parent(
                content,
                parent_turn=current_turn,
                archived=archived,
                keep_in_main=keep_in_main,
                **kwargs,
            )
        else:
            if not current_turn.HasResponse:
                raise ValueError("Cannot add another user message until previous turn gets a response.")
            
            # Standard linear conversation: create a new turn as a child of the active one.
            new_turn = Turn(turn_type=Turn.CHAT, metadata={"user": message}, is_archived=archived)
            added_turn = self._add_chat_turn(current_turn, new_turn)
            new_command = Command(Command.TURN, {"prompt": f"[{added_turn.gen_id}]", "method": "add_user"})
            self._annotate_turn_command(new_command, added_turn)
            self._add_command_to_turn(new_command, added_turn)
            return added_turn

    @_with_write_lock
    def _ensure_cmd_has_gen_id(self, command: Command) -> None:
        """Assigns a new `gen_id` to a command if it doesn't have one."""
        if command.gen_id:
            return
        self._last_cmd_id_counter += 1
        new_id = self._last_cmd_id_counter
        
        prefix = "cmd"
        type_lower = (command.cmd_type or "").lower()
        if type_lower == Command.COMMAND.lower():
            prefix = "c"
        elif type_lower == Command.LOG.lower():
            prefix = "l"
        elif type_lower == Command.PARAM_CHANGE.lower():
            prefix = "p"
        elif type_lower == Command.ADAPTERS_STATE.lower():
            prefix = "as"
        elif type_lower == Command.STATE_CHANGE.lower():
            prefix = "s"
        elif type_lower == Command.TURN.lower():
            prefix = "t"
        
        command.gen_id = f"{prefix}_{new_id}"

    def _next_stack_id(self) -> str:
        """Return a new stack id for set/add stack entries."""
        self._last_stack_id_counter += 1
        return f"pop_{self._last_stack_id_counter}"

    def _register_stack_id(self, stack_id: Optional[str]) -> None:
        """Advance the stack counter to preserve monotonicity when ids are replayed."""
        if not stack_id:
            return
        match = re.match(r"^pop_(\d+)$", str(stack_id))
        if not match:
            return
        try:
            value = int(match.group(1))
        except ValueError:
            return
        if value > self._last_stack_id_counter:
            self._last_stack_id_counter = value

    def _find_chat_session_for_turn(self, active_turn: Optional[Turn]) -> Optional["ChatSession"]:
        """Locate the chat session that owns active_turn."""
        if not active_turn:
            return None
        for conv in self.conversations:
            if conv.root_turn is active_turn:
                return conv
            q: List[Turn] = [conv.root_turn]
            visited = {conv.root_turn}
            while q:
                current = q.pop(0)
                if current is active_turn:
                    return conv
                for child in getattr(current, "turns", []) or []:
                    if child not in visited:
                        q.append(child)  # type: ignore
                        visited.add(child)
        return None

    def _resolve_stack_id(self, current_turn: Turn, stack_id: str, *, change_key: str) -> str:
        """
        Resolve a pop target (stack_id/cmd_id/turn_gen_id/anchor) into a stack_id.
        Raises if no matching stack entry can be found on the active path.
        """
        original_id = stack_id
        if not stack_id:
            raise ValueError("stack_id is required for targeted pop.")
        path: List[Turn] = self.get_active_path_for_llm(current_turn) if current_turn else []
        if not path:
            raise ValueError("No active path available for pop resolution.")

        push_ops = {"set", "add"}
        candidates: List[Tuple[Turn, Command]] = []
        for turn in path:
            for cmd in getattr(turn, "cmd", []) or []:
                if cmd.data.get("change") == change_key and cmd.data.get("op") in push_ops:
                    candidates.append((turn, cmd))

        # 1) Direct stack_id match
        for _, cmd in candidates:
            if cmd.data.get("stack_id") == stack_id:
                return stack_id

        # 2) Command id match
        for _, cmd in candidates:
            if cmd.gen_id == stack_id:
                resolved = cmd.data.get("stack_id")
                if not resolved:
                    raise ValueError(f"Command '{original_id}' has no stack_id for change '{change_key}'.")
                return resolved

        # 3) Turn gen_id or anchor match (use latest push on that turn)
        chat_session = self._find_chat_session_for_turn(current_turn)
        target_turn = self.get_turn_by_gen_id(stack_id, chat_session)
        if not target_turn:
            target_turn = self._find_turn_by_anchor_name(stack_id, chat_session)
        if target_turn and target_turn in path:
            for cmd in reversed(getattr(target_turn, "cmd", []) or []):
                if cmd.data.get("change") == change_key and cmd.data.get("op") in push_ops:
                    resolved = cmd.data.get("stack_id")
                    if not resolved:
                        raise ValueError(f"Turn '{original_id}' has no stack_id for change '{change_key}'.")
                    return resolved
            raise ValueError(f"No stack entry found on turn '{original_id}' for change '{change_key}'.")

        raise ValueError(f"Pop target '{original_id}' not found on active path for change '{change_key}'.")

    def _annotate_turn_command(self, command: Command, added_turn: Turn) -> None:
        """Attach parent and sibling placement metadata for TURN commands."""
        if not command or not added_turn:
            return
        data = command.data if isinstance(command.data, dict) else {}
        if added_turn.gen_id:
            data.setdefault("turn_gen_id", added_turn.gen_id)
        parent = getattr(added_turn, "parent", None)
        if parent and parent.gen_id:
            data.setdefault("parent_gen_id", parent.gen_id)
            try:
                data.setdefault("sibling_index", list(parent.turns or []).index(added_turn))
            except ValueError:
                pass
        command.data = data

    @staticmethod
    def _is_closed_turn(turn: Optional["Turn"]) -> bool:
        """Return True for turns that should be treated as closed for branching."""
        if not turn:
            return False
        if getattr(turn, "IsStructural", False) or turn.turn_type == Turn.RESERVED:
            return True
        return (not turn.IsEmpty) and turn.HasResponse

    @_with_write_lock
    def _add_command_to_turn(self, new_command: Command, target_turn: Turn) -> Command:
        """
        Adds a new Command node to a specific target Turn.
        This is an advanced method for directly manipulating the tree, used for
        cases like batch generation where the target is not the active turn.

        Args:
            cmd_type: The type of the command (e.g., Command.LOG, Command.TURN).
            metadata: The data payload for the command.
            target_turn: The Turn object to which the command should be attached.
        """
        if not isinstance(target_turn, Turn):
            raise ValueError(f"Cannot add command to an invalid target turn of type {type(target_turn)}.")

        self._ensure_turn_has_gen_id(target_turn)
        self._ensure_cmd_has_gen_id(new_command)

        new_command.parent = target_turn
        target_turn.cmd.append(new_command)
        self.commands_history.append(new_command)
        return new_command

    @_with_write_lock
    def add_command_to_turn(self, new_command: Command, target_turn: Turn) -> Command:
        return self._add_command_to_turn(new_command, target_turn)

    @_with_write_lock
    def remove_commands_in_turn(self, target_turn: Optional[Turn], predicate) -> List[Command]:
        if target_turn is None:
            return []
        commands = getattr(target_turn, "cmd", None)
        if not commands:
            return []
        removed: List[Command] = []
        retained: List[Command] = []
        for cmd in commands:
            if predicate(cmd):
                removed.append(cmd)
            else:
                retained.append(cmd)
        if not removed:
            return []
        commands[:] = retained
        self.remove_commands_from_history(removed)
        return removed

    @_with_write_lock
    def replace_command_in_turn(
        self,
        target_turn: Optional[Turn],
        predicate,
        new_command: Command,
        *,
        insert_at: Optional[int] = None,
    ) -> Optional[Command]:
        if target_turn is None:
            return None
        commands = getattr(target_turn, "cmd", None)
        if commands is None:
            target_turn.cmd = []
            commands = target_turn.cmd
        for cmd in list(commands):
            if predicate(cmd):
                commands.remove(cmd)
                self.remove_command_from_history(cmd)
                break
        if insert_at is None:
            return self._add_command_to_turn(new_command, target_turn)
        insert_index = max(0, min(insert_at, len(commands)))
        self._ensure_turn_has_gen_id(target_turn)
        self._ensure_cmd_has_gen_id(new_command)
        new_command.parent = target_turn
        commands.insert(insert_index, new_command)
        self.append_command_to_history(new_command)
        return new_command

    @_with_write_lock
    def reconcile_state_command(
        self,
        target_turn: Optional[Turn],
        change: str,
        value: Any,
        *,
        command_text: Optional[str] = None,
        cmd_type: str = Command.STATE_CHANGE,
    ) -> Optional[Command]:
        if target_turn is None:
            return None
        def _match(existing: Command) -> bool:
            return (
                existing.cmd_type in (Command.STATE_CHANGE, Command.PARAM_CHANGE)
                and existing.data.get("change") == change
            )
        self.remove_commands_in_turn(target_turn, _match)
        if value is None:
            return None
        payload_key = "value" if cmd_type == Command.STATE_CHANGE else "new_value"
        metadata = {"change": change, payload_key: copy.deepcopy(value)}
        if command_text:
            metadata["command"] = command_text
        new_command = Command(cmd_type=cmd_type, metadata=metadata)
        return self._add_command_to_turn(new_command, target_turn)

    @staticmethod
    def _is_convertible_placeholder(turn: "Turn") -> bool:
        """
        Returns True for placeholders that may still be repurposed into CHAT/BATCH/FORK.
        RESERVED anchors are intentionally excluded to keep them immutable.
        """
        return turn.turn_type in (None, Turn.HOLD)

    @_with_write_lock
    def add_batch_turn(
        self,
        command_text: str,
        prompts: List[str],
        current_turn: Turn, # The logical parent for the operation
        override_adapters: Optional[List[str]] = None,
    ) -> Tuple[Turn, List[Turn]]:
        """
        Adds a BATCH turn.
        
        Args:
            command_text: The batch command text (e.g., "/g b" or "/g bc" or "/g ao").
            prompts: The list of user prompts to seed the batch.
            current_turn: The logical parent turn where the batch will attach.
            override_adapters: Optional per-prompt adapter overrides.

        - If `current_turn` is an empty placeholder, it is repurposed into the BATCH turn
          (covers both main and try-out placeholders).
        - If `current_turn` is not a placeholder:
          * If this is the first real child under `current_turn`, this BATCH becomes
            the natural main continuation; its first prompt is marked `main_thread`
            and can later be re-promoted via state_change.
          * Additional BATCH siblings are allowed; main_thread inclusion is governed
            by explicit promotion and HasResponse gating in active-path construction.
        """
        if not prompts:
            raise ValueError("Cannot add a batch turn with no prompts.")
        if current_turn is None:
            raise ValueError("A valid current_turn must be provided.")

        batch_metadata = {
            "command": command_text,
            "batch_count": len(prompts),
        }
        if override_adapters:
            batch_metadata["override_adapters"] = list(override_adapters)

        if self._is_convertible_placeholder(current_turn):
            batch_turn = current_turn
            batch_turn.turn_type = Turn.BATCH
            batch_turn.data.update(batch_metadata)
            self._ensure_turn_has_gen_id(batch_turn)
        else:
            batch_turn = self._add_chat_turn(
                parent_turn=current_turn, new_turn=Turn(turn_type=Turn.BATCH, metadata=batch_metadata))

        new_command = Command(Command.TURN, {"prompt": command_text, "method": "add_batch_turn"})
        self._annotate_turn_command(new_command, batch_turn)
        self._add_command_to_turn(new_command, batch_turn)

        all_chat_turns: List[Turn] = []
        # Automatically mark the first child as main only when the batch itself sits on the main spine.
        batch_is_main = bool(batch_turn.parent and batch_turn.parent.turns and batch_turn.parent.turns[0] is batch_turn)
        for i, prompt in enumerate(prompts):
            chat_turn = self._add_user_with_parent(prompt, parent_turn=batch_turn)
            chat_turn.main_thread = bool(batch_is_main and i == 0)
            all_chat_turns.append(chat_turn)
            if override_adapters and i < len(override_adapters):
                chat_turn.data["adapter_override_name"] = override_adapters[i]

        # Mirror "has a main child" onto the BATCH node to simplify collectors.
        setattr(batch_turn, "main_thread", batch_is_main)

        return batch_turn, all_chat_turns

    @_with_write_lock
    def add_assistant(self, content: Optional[str], tool_blocks: Optional[List[ToolCallBlock]] = None, archived: bool = False, was_truncated: bool = False, was_canceled: bool = False, destination_turn: Optional[Turn] = None, **kwargs):
        message: Optional[Dict[str, Any]] = None
        if content:
            message = {"role": "assistant", "content": content}
            message.update(kwargs)
        if tool_blocks:
            if message is None: message = {"role": "assistant"}
            message["tool_blocks"] = tool_blocks

        target_turn = destination_turn
        if not target_turn:
            raise ValueError("Cannot add assistant message: a valid destination_turn must be provided.")
        
        is_closed_for_assistant = (
            target_turn.IsEmpty
            or target_turn.HasResponse
            or target_turn.IsStructural
            or target_turn.turn_type == Turn.RESERVED
        )

        if not target_turn: # type: ignore
            raise ValueError("Cannot add assistant message, no valid target turn could be determined.")

        # The active turn should be the one waiting for a response.
        # It's an error if it's a placeholder or already has an assistant message.
        if is_closed_for_assistant:
            raise ValueError(f"Cannot add assistant message to turn '{self.get_display_id(target_turn)}' which is closed for assistant. Target Turn: {target_turn.to_dict(include_children=False)}")

        target_turn.data["assistant"] = message
        if archived: target_turn.is_archived = True
        if was_truncated: target_turn.was_truncated = True
        if was_canceled: target_turn.was_canceled = True

    @_with_write_lock
    def add_continuation_turn(self, current_turn: Turn, archived: bool = False, **kwargs) -> Turn:
        """
        Creates a new, empty child turn to prepare for a continued generation.
        This is called before inference for commands like /cg.
        It mirrors the `add_user` signature and routes to it with empty content.
        """
        if current_turn.IsStructural:
            raise ValueError(f"Cannot add a continuation turn directly to a structural turn of type '{current_turn.turn_type}'.")

        # Inherit the main_thread status from the parent turn.
        kwargs_with_main = {"keep_in_main": current_turn.main_thread, **kwargs}
        continuation_turn = self.add_user("", current_turn, archived, **kwargs_with_main)
        continuation_turn.do_continue = True
        return continuation_turn

    @_with_write_lock
    def add_fork_hub_and_main_placeholder(self, current_turn: Turn) -> Tuple[Turn, Turn]:
        """
        Takes the current_turn placeholder or adds a FORK node to act as a fork hub.
        The original branch will continue as its first child.

        Returns:
            A tuple containing (the fork hub turn, the main branch placeholder turn).
        """

        fork_turn = Turn(Turn.FORK)
        if current_turn.turn_type is None: # repurpose a placeholder
            fork_turn = current_turn
            fork_turn.turn_type = Turn.FORK
            fork_turn.timestamp = time.time()
        else: # grow current branch
            fork_turn.parent = current_turn
            current_turn.turns.append(fork_turn)

        self._ensure_turn_has_gen_id(fork_turn)

        main_placeholder = Turn(turn_type=None, main_thread=True)
        self._add_chat_turn(fork_turn, main_placeholder)
        return fork_turn, main_placeholder

    @_with_write_lock
    def add_tool_results(self, tool_blocks: List[ToolCallBlock], current_turn: Turn) -> Turn:
        """
        Creates a new turn for tool results (ToolCallBlock list). Populates placeholders
        when possible; otherwise grows a child turn on the active branch.
        """

        if current_turn.IsFork:
            new_turn = Turn(turn_type=Turn.CHAT, metadata={"tool_results": tool_blocks})
            return self._add_chat_turn(current_turn, new_turn)
        else:
            if self._is_convertible_placeholder(current_turn):
                current_turn.data["tool_results"] = tool_blocks # noqa
                if current_turn.turn_type in (None, Turn.HOLD):
                    current_turn.turn_type = Turn.CHAT
                    self._ensure_turn_has_gen_id(current_turn)
                return current_turn
            else:
                if current_turn.IsStructural:
                    raise ValueError(f"Cannot add tool results directly to a structural turn of type '{current_turn.turn_type}'.")
                new_turn = Turn(turn_type=Turn.CHAT, metadata={"tool_results": tool_blocks})
                added_turn = self._add_chat_turn(current_turn, new_turn)

        return added_turn

    @_with_write_lock
    def add_message(self, role: str, content: Any, current_turn: Turn, is_auto: bool = False, **kwargs) -> Turn:
        """
        Adds a message with any role to the conversation.
        - If role is 'assistant', it delegates to `add_assistant`.
        - For other roles, it creates a new turn or populates a placeholder,
          storing the content in `turn.data[role]`.
        """
        if role == 'assistant':
            # `add_assistant` expects content to be a string and handles its own logic.
            # We pass the current_turn as the destination.
            self.add_assistant(content=str(content), destination_turn=current_turn, **kwargs)
            if is_auto:
                current_turn.is_auto = is_auto
            return current_turn

        message = {"role": role, "content": content} if content is not None else {}
        message.update(kwargs)

        new_turn = Turn(turn_type=Turn.CHAT, metadata={role: message})
        new_turn.is_auto = is_auto
        turn_to_log_on = None
        if current_turn.IsFork:
            # If the current turn is a FORK hub, always add the new message
            # as a new child turn (a new fork/branch).
            turn_to_log_on = self._add_chat_turn(current_turn, new_turn)
        elif current_turn.turn_type == Turn.RESERVED:
            # Do not repurpose RESERVED anchors; create a child turn for real content.
            turn_to_log_on = self._add_chat_turn(current_turn, new_turn)
        elif current_turn.IsEmpty or role.startswith("$"):
            current_turn.data[role] = message
            if current_turn.turn_type is None:
                current_turn.turn_type = Turn.CHAT
                self._ensure_turn_has_gen_id(current_turn)
            if is_auto:
                current_turn.is_auto = is_auto
            turn_to_log_on = current_turn
        else:
            if current_turn.IsStructural:
                raise ValueError(f"Cannot add message directly to a structural turn of type '{current_turn.turn_type}'.")
            if not current_turn.HasResponse:
                raise ValueError("Cannot add another user-side message until the previous turn gets a response.")
            
            new_turn = Turn(turn_type=Turn.CHAT, metadata={role: message})
            new_turn.is_auto = is_auto
            turn_to_log_on = self._add_chat_turn(current_turn, new_turn)

        if turn_to_log_on and turn_to_log_on.gen_id:
            new_command = Command(Command.TURN, {"prompt": f"[{turn_to_log_on.gen_id}]", "method": "add_message", "role": role})
            self._annotate_turn_command(new_command, turn_to_log_on)
            self._add_command_to_turn(new_command, turn_to_log_on)


        return turn_to_log_on
    
    @_with_write_lock
    def add_param_change(self, current_turn: Turn, change_type: str, new_value: Any, command_text: Optional[str] = None) -> Turn: # type: ignore
        """Adds a non-chat turn to record a direct inference parameter change (e.g., system message)."""
        old_value = None
        if change_type == "system_message":
            old_value = self.get_effective_system_message(current_turn)
        data = {"change": change_type, "old_value": old_value, "new_value": new_value}
        if command_text:
            data["command"] = command_text
        return self._add_command(Command(cmd_type=Command.PARAM_CHANGE, metadata=data), current_turn)

    @_with_write_lock
    def add_state_change(self, current_turn: Turn, change_type: str, value: Any, command_text: Optional[str] = None) -> Turn:
        """Adds a non-chat turn to record an implicit behavior change (e.g., flags, tool activation)."""
        data = {"change": change_type, "value": value}
        if command_text:
            data["command"] = command_text
        new_command = Command(cmd_type=Command.STATE_CHANGE, metadata=data)
        # The command is attached to the current_turn, which might be a placeholder.
        # The method returns the (potentially new) turn that now holds the command.
        return self._add_command(new_command, current_turn)

    @_with_write_lock
    def add_adapter_command(
        self,
        current_turn: Turn,
        operation: str,
        adapters: Optional[List[str]] = None,
        command_text: Optional[str] = None,
        stack_id: Optional[str] = None,
        validate_cmd_id: bool = True,
    ) -> Turn:
        """
        Records an adapter stack operation (set/add/pop) on the session tree.
        These commands are replayed via get_effective_adapters() when traversing branches.
        stack_id is normally generated; pass it only to preserve ids during replay.
        """
        if operation not in {"set", "add", "pop"}:
            raise ValueError(f"Unsupported adapter operation '{operation}'.")
        payload: Dict[str, Any] = {"change": "adapters_command", "op": operation}
        if operation == "pop" and stack_id and validate_cmd_id:
            stack_id = self._resolve_stack_id(current_turn, stack_id, change_key="adapters_command")
        if adapters is not None:
            payload["adapters"] = list(adapters)
        if operation in {"set", "add"}:
            if stack_id:
                self._register_stack_id(stack_id)
                payload["stack_id"] = stack_id
            else:
                payload["stack_id"] = self._next_stack_id()
        if command_text:
            payload["command"] = command_text
        if operation == "pop":
            if stack_id:
                payload["stack_id"] = stack_id
        new_command = Command(cmd_type=Command.STATE_CHANGE, metadata=payload)
        return self._add_command(new_command, current_turn)

    @_with_write_lock
    def add_tools_scope_command(
        self,
        current_turn: Turn,
        operation: str,
        scope: Optional[ToolsScope] = None,
        command_text: Optional[str] = None,
        stack_id: Optional[str] = None,
        validate_cmd_id: bool = True,
    ) -> Turn:
        """
        Records a tools scope mutation (set/add/pop/reset) so that tool visibility/execution can be
        reconstructed for each branch.
        stack_id is normally generated; pass it only to preserve ids during replay.
        """
        if operation not in {"set", "add", "pop", "reset"}:
            raise ValueError(f"Unsupported tools scope operation '{operation}'.")
        payload: Dict[str, Any] = {"change": "tools_scope", "op": operation}
        if operation == "pop" and stack_id and validate_cmd_id:
            stack_id = self._resolve_stack_id(current_turn, stack_id, change_key="tools_scope")
        if scope and not scope.is_noop():
            payload["scope"] = scope.to_dict()
        if operation in {"set", "add"}:
            if stack_id:
                self._register_stack_id(stack_id)
                payload["stack_id"] = stack_id
            else:
                payload["stack_id"] = self._next_stack_id()
        if command_text:
            payload["command"] = command_text
        if operation == "pop":
            if stack_id:
                payload["stack_id"] = stack_id
        new_command = Command(cmd_type=Command.STATE_CHANGE, metadata=payload)
        return self._add_command(new_command, current_turn)

    @_with_write_lock
    def reapply_state_change_commands(self, target_turn: Turn, commands: Sequence[Command]) -> Tuple[Turn, List[str]]:
        """
        Replays adapter/system-message state commands onto the specified target turn.
        """
        if not target_turn:
            raise ValueError("Cannot reapply commands without a valid target turn.")
        applied: List[str] = []
        for cmd in commands:
            try:
                if cmd.cmd_type == Command.STATE_CHANGE and cmd.data.get("change") == "adapters_command":
                    op = cmd.data.get("op")
                    if not op:
                        continue
                    adapters = cmd.data.get("adapters")
                    if isinstance(adapters, str):
                        adapters_list = [adapters]
                    elif isinstance(adapters, list):
                        adapters_list = [str(a) for a in adapters if isinstance(a, str)]
                    else:
                        adapters_list = None
                    target_turn = self.add_adapter_command(
                        target_turn,
                        op,
                        adapters_list,
                        command_text=cmd.data.get("command", "reapplied adapter state"),
                        stack_id=cmd.data.get("stack_id"),
                        validate_cmd_id=False,
                    )
                    applied.append(f"adapters:{op}")
                elif cmd.cmd_type == Command.PARAM_CHANGE and cmd.data.get("change") == "system_message":
                    op = cmd.data.get("op")
                    inferred_op = op or ("set" if cmd.data.get("new_value", cmd.data.get("value")) is not None else "set")
                    value = cmd.data.get("value", cmd.data.get("new_value"))
                    target_turn = self.add_system_message_command(
                        target_turn,
                        inferred_op,
                        value,
                        command_text=cmd.data.get("command", "reapplied system message"),
                        validate_cmd_id=False,
                        value_kind=cmd.data.get("value_kind"),
                        stack_id=cmd.data.get("stack_id"),
                    )
                    applied.append(f"system:{inferred_op}")
                elif cmd.cmd_type == Command.STATE_CHANGE and cmd.data.get("change") == "tools_scope":
                    op = (cmd.data.get("op") or "add").lower()
                    scope_data = cmd.data.get("scope")
                    scope_obj = ToolsScope.from_dict(scope_data) if scope_data else None
                    target_turn = self.add_tools_scope_command(
                        target_turn,
                        op,
                        scope_obj,
                        command_text=cmd.data.get("command", "reapplied tools scope"),
                        stack_id=cmd.data.get("stack_id"),
                        validate_cmd_id=False,
                    )
                    applied.append(f"tools:{op}")
            except Exception:
                continue
        return target_turn, applied

    @_with_read_lock
    def _ensure_cmd_id_on_path(self, current_turn: Turn, cmd_id: str, *, change_key: str) -> None:
        """
        Validate that a command id exists on the direct active branch (root -> current_turn),
        stopping at the conversation root or a turn marked as root_context.
        """
        path: List[Turn] = self.get_active_path_for_llm(current_turn) if current_turn else []
        for turn in path:
            for cmd in getattr(turn, "cmd", []) or []:
                if cmd.gen_id == cmd_id and cmd.data.get("change") == change_key:
                    return
            if getattr(turn, "root_context", False):
                break
        raise ValueError(f"Command id '{cmd_id}' not found on active branch for change '{change_key}'.")

    @_with_write_lock
    def add_system_message_command(
        self,
        current_turn: Turn,
        operation: str,
        value: Optional[str] = None,
        command_text: Optional[str] = None,
        label: Optional[str] = None,
        *,
        validate_cmd_id: bool = True,
        value_kind: Optional[str] = None,
        stack_id: Optional[str] = None,
    ) -> Turn:
        """
        Records a structured system message operation (set/add/pop) on the session tree.
        stack_id is normally generated; pass it only to preserve ids during replay.
        """
        if operation not in {"set", "add", "pop"}:
            raise ValueError(f"Unsupported system message operation '{operation}'.")
        payload: Dict[str, Any] = {"change": "system_message", "op": operation}
        if operation == "pop" and stack_id and validate_cmd_id:
            stack_id = self._resolve_stack_id(current_turn, stack_id, change_key="system_message")
        if operation in {"set", "add"}:
            payload["value"] = value
            if value_kind:
                payload["value_kind"] = value_kind
            if stack_id:
                self._register_stack_id(stack_id)
                payload["stack_id"] = stack_id
            else:
                payload["stack_id"] = self._next_stack_id()
        if command_text:
            payload["command"] = command_text
        if operation == "pop":
            if stack_id:
                payload["stack_id"] = stack_id
        new_command = Command(cmd_type=Command.PARAM_CHANGE, metadata=payload)
        destination = self._add_command(new_command, current_turn)
        return destination

    @_with_write_lock
    def add_adapters_state(self, current_turn: Turn, state_type: str, value: Any) -> Turn:
        """Adds a command to record a reconstructed state from a previous session."""
        data = {"state_type": state_type, "value": value}
        # This state is global, so attach it to the first conversation's root.
        if self.conversations:
            new_command = Command(cmd_type=Command.ADAPTERS_STATE, metadata=data)
            self._add_command_to_turn(new_command, current_turn)
        return current_turn

    @_with_write_lock
    def add_log_command(self, current_turn: Turn, command_text: str, extra: Optional[Dict[str, Any]] = None)  -> Turn:
        """
        Adds a LOG command, which represents a client-side action that does not
        call the engine API. This is for logging and history purposes.
        This command is always attached to the current turn and never creates a new one.
        """
        payload: Dict[str, Any] = {"command": command_text}
        if extra:
            payload.update(extra)
        new_command = Command(cmd_type=Command.LOG, metadata=payload)
        self._add_command_to_turn(new_command, current_turn)
        return current_turn

    @_with_write_lock
    def add_client_command(self, current_turn: Turn, command_text: str, extra: Optional[Dict[str, Any]] = None) -> Turn:
        """
        Adds a COMMAND entry for a client-side action (replay-friendly).
        This is similar to add_log_command but uses Command.COMMAND to ensure
        it participates in replay/mirroring flows.
        """
        payload: Dict[str, Any] = {"command": command_text}
        if extra:
            payload.update(extra)
        new_command = Command(cmd_type=Command.COMMAND, metadata=payload)
        self._add_command_to_turn(new_command, current_turn)
        return current_turn

    @_with_write_lock
    def add_api_command(self, current_turn: Turn, api_name: str, api_params: Dict[str, Any], command_text: str) -> Turn:

        """
        Adds a command that represents an engine API call, performing special
        processing on parameters for replayability.
        """
        params_copy = copy.deepcopy(api_params)

        # 1. Handle `messages_list` replacement with turn IDs
        if "messages_list" in params_copy or "raw_list" in params_copy:
            # This assumes the messages_list was generated from the active path.
            # We get the path and extract the gen_ids.
            if current_turn:
                path_for_llm = self.get_active_path_for_llm(current_turn)
                turn_ids = [turn.gen_id for turn in path_for_llm if turn.gen_id]
                # Replace the content with the list of turn IDs.
                params_copy["messages_list"] = {"turns": turn_ids}

        new_command = Command(cmd_type=Command.COMMAND, metadata={"command": command_text}, api_name=api_name, api_params=params_copy)
        return self._add_command(new_command, current_turn)

    @_with_read_lock
    def get_conversations_count(self) -> int:
        """Returns the number of root conversation trees in the session."""
        return len(self.conversations)

    @_with_read_lock
    def get_conversation(self, index: int) -> ChatSession:
        """Returns the ChatSession at the specified index."""
        if 0 <= index < len(self.conversations):
            return self.conversations[index]
        raise IndexError("Conversation index out of range.")

    @_with_read_lock
    def get_main_branch_leaf(self, conversation_index: int = 0) -> Optional[Turn]:
        """
        Finds the leaf node of the main branch (always following the first child)
        for a given conversation tree.
        """
        if not (0 <= conversation_index < len(self.conversations)):
            return None
        
        root_turn = self.conversations[conversation_index].root_turn
        return self.get_last_turn_on_branch(root_turn)

    @_with_write_lock
    def _ensure_conversation_root_initialized(self, chat_session: ChatSession) -> None:
        """Ensure the conversation root is a RESERVED anchor with a stable gen_id."""
        if not chat_session or not getattr(chat_session, "root_turn", None):
            return
        root = chat_session.root_turn
        if root.turn_type != Turn.RESERVED:
            root.turn_type = Turn.RESERVED
        if not getattr(root, "root_context", False):
            root.root_context = True
        self._ensure_turn_has_gen_id(root)

    @_with_write_lock
    def add_conversation(self,
                          engine_config: Optional[Dict[str, Any]] = None,
                          engine_warnings: Optional[List[str]] = None,
                          parser_profile: Optional[ParserProfile] = None,
                          inference_defaults: Optional[InferenceParams] = None,
                          initial_params: Optional[Dict[str, Any]] = None,
                          title: Optional[str] = None,
                         ) -> ChatSession:
        """Creates and adds a new ChatSession, returning it."""
        new_session = ChatSession(
            engine_config=engine_config,
            engine_warnings=engine_warnings,
            parser_profile=parser_profile,
            inference_defaults=inference_defaults,
            initial_params=initial_params,
            title=title
        )
        self._ensure_conversation_root_initialized(new_session)
        self.conversations.append(new_session)
        return new_session

    def conversation_from_template(self, template: ChatSession, title: Optional[str] = None) -> ChatSession:
        engine_config = copy.deepcopy(template.engine_config) if template else {}
        engine_warnings = list(template.engine_warnings or []) if template else []
        inference_defaults = copy.deepcopy(template.inference_defaults) if template else InferenceParams()
        initial_params = copy.deepcopy(template.initial_params) if template else {}
        parser_profile = copy.deepcopy(template.parser_profile) if template and template.parser_profile else None
        new_conv = ChatSession(
            engine_config=engine_config,
            engine_warnings=engine_warnings,
            parser_profile=parser_profile,
            initial_params=initial_params,
            inference_defaults=inference_defaults,
            title=title if title is not None else getattr(template, "title", None),
        )
        if new_conv.root_turn and not getattr(new_conv.root_turn, "root_context", False):
            new_conv.root_turn.root_context = True
        return new_conv
    
    @_with_write_lock
    def insert_conversation(
                        self,
                        index: Optional[int] = None,
                        template: Optional[ChatSession] = None,
                        engine_config: Optional[Dict[str, Any]] = None,
                        engine_warnings: Optional[List[str]] = None,
                        parser_profile: Optional[ParserProfile] = None,
                        inference_defaults: Optional[InferenceParams] = None,
                        initial_params: Optional[Dict[str, Any]] = None,
                        title: Optional[str] = None
                        ) -> ChatSession: 
        """Creates and adds a new ChatSession, returning it."""
        if template:
            new_session = self.conversation_from_template(template=template, title=title)
        else:
            new_session = ChatSession(
                engine_config=engine_config,
                engine_warnings=engine_warnings,
                parser_profile=parser_profile,
                inference_defaults=inference_defaults,
                initial_params=initial_params,
                title=title if title is not None else getattr(template, "title", None),
            )

        self._ensure_conversation_root_initialized(new_session) 
        if not index or index < 0:
            self.conversations.append(new_session)
        else:
            self.conversations.insert(index, new_session)
        
        return new_session


    @_with_read_lock
    def get_last_turn_on_branch(self, start_node: Turn) -> Turn:
        """
        Finds the 'active' leaf of a branch starting from `start_node`.
        The active leaf is the last turn in a linear sequence of descendants.
        """
        # This is a recursive helper to find the leaf.
        def _find_leaf(node: Turn) -> Turn:
            if not node.turns:
                return node
            return _find_leaf(node.turns[0])
        return _find_leaf(start_node)

    @_with_read_lock
    def _default_system_message_segments(self) -> List[Tuple[str, str]]:
        """Returns the baseline segment list derived from the session default."""
        segments: List[Tuple[str, str]] = []
        if self.default_system_message is not None:
            segments.append(("default", self.default_system_message))
        return segments

    @_with_read_lock
    def get_system_message_segments(self, active_turn: Optional[Turn], chat_session: Optional[ChatSession] = None) -> Tuple[Dict[str, str], bool]:
        """
        Returns a tuple of (segments_dict, explicitly_removed_flag) representing the effective
        structured system message for the active branch. The dictionary preserves insertion order.
        This version includes the new 'pop <target>' logic.
        """
        if not active_turn:
            return {}, False

        path: List[Turn] = self.get_active_path_for_llm(active_turn)
        if not path:
            return {}, False

        # 1. Collect all relevant commands from the history path, enforcing 'op' presence.
        all_ops: List[Tuple[Turn, Command]] = []
        for turn in path:
            for cmd in turn.cmd:
                if cmd.cmd_type == Command.PARAM_CHANGE and cmd.data.get("change") == "system_message":
                    if "op" not in cmd.data:
                        raise ValueError(f"System message command {cmd.gen_id or 'N/A'} on turn {turn.gen_id_or_parent} is missing 'op' field.")
                    all_ops.append((turn, cmd))

        # 2. Process pops inline to build the effective command list.
        filtered_ops: List[Tuple[Turn, Command]] = []
        for turn, cmd in all_ops:
            op_type = cmd.data["op"]
            if op_type != "pop":
                filtered_ops.append((turn, cmd))
                continue

            target_id = cmd.data.get("stack_id")
            if not target_id:
                # simple pop: drop the most recent set/add
                for idx in range(len(filtered_ops) - 1, -1, -1):
                    if filtered_ops[idx][1].data["op"] in {"set", "add"}:
                        filtered_ops.pop(idx)
                        break
                continue

            removed = False
            for idx, (_, candidate_cmd) in enumerate(filtered_ops):
                if candidate_cmd.data.get("stack_id") == target_id and candidate_cmd.data.get("change") == "system_message":
                    filtered_ops.pop(idx)
                    removed = True
                    break
            if removed:
                continue

        valid_ops: List[Command] = [cmd for _, cmd in filtered_ops]

        # 3. Process the filtered operations to build the final state
        segments: List[Tuple[str, str]] = self._default_system_message_segments()
        explicit_none = False

        active_ops: List[Dict[str, Any]] = []
        for cmd in valid_ops:
            op_type = cmd.data["op"]
            raw_value = cmd.data.get("value") if "value" in cmd.data else cmd.data.get("new_value")
            raw_kind = (cmd.data.get("value_kind") or "").lower() or None

            # Backwards compatibility for old sentinel-based commands
            if not raw_kind and isinstance(raw_value, str):
                if raw_value == "<>":
                    raw_kind = "remove"
                    raw_value = None
                elif raw_value == "<def>":
                    raw_kind = "default"
                    raw_value = None

            active_ops.append({
                "op": op_type,
                "value": raw_value,
                "stack_id": cmd.data.get("stack_id"),
                "value_kind": raw_kind
            })

        # 5. Build final segment list from the active operations
        for op in active_ops:
            op_type = op["op"]
            value_kind = op.get("value_kind")
            if op_type == "set":
                value = op.get("value")
                # Special markers: default and remove
                if value_kind == "default":
                    explicit_none = False
                    segments = self._default_system_message_segments()
                    continue
                if value_kind == "remove" or value is None:
                    explicit_none = True
                    segments = []
                else:
                    explicit_none = False
                    label = op.get("stack_id") or f"segment_{len(segments) + 1}"
                    segments = [(label, value)]
            elif op_type == "add":
                if value_kind:
                    # Ignore add operations that refer to special markers; they are invalid.
                    continue
                value = op.get("value")
                if value is None:
                    continue
                explicit_none = False
                label = op.get("stack_id") or f"segment_{len(segments) + 1}"
                segments.append((label, value))

        ordered_segments: Dict[str, str] = {}
        if not explicit_none:
            for idx, (label, text) in enumerate(segments):
                key = label or f"segment_{idx + 1}"
                candidate = key
                suffix = 2
                while candidate in ordered_segments:
                    candidate = f"{key}_{suffix}"
                    suffix += 1
                ordered_segments[candidate] = text

        return ordered_segments, explicit_none

    @_with_read_lock
    def _serialize_system_message_segments(self, segments: Mapping[str, str]) -> Optional[str]:
        if not segments:
            return None
        parts: List[str] = []
        for _, value in segments.items():
            value_text = value if value is not None else ""
            parts.append(value_text)
        return "\n\n".join(parts)

    @_with_read_lock
    def get_effective_system_message(self, active_turn: Optional[Turn]) -> Optional[str]:
        """
        Finds the system message that applies to the currently active turn, combining
        stacked operations along the active branch (including try-outs).
        """
        if not active_turn:
            return self._serialize_system_message_segments(dict(self._default_system_message_segments()))

        # Find which conversation the active_turn belongs to
        chat_session = None
        for conv in self.conversations:
            # Check if active_turn is the root of this conversation
            if conv.root_turn is active_turn:
                chat_session = conv
                break
            # Check if active_turn is a descendant of the root of this conversation
            q: List[Turn] = [conv.root_turn]
            visited = {conv.root_turn}
            found = False
            while q:
                current = q.pop(0)
                if current is active_turn:
                    chat_session = conv
                    found = True
                    break
                for child in getattr(current, "turns", []) or []:
                    if child not in visited:
                        q.append(child) # type: ignore
                        visited.add(child)
            if found:
                break
        
        segments, explicit_none = self.get_system_message_segments(active_turn, chat_session)
        if explicit_none:
            return None
        return self._serialize_system_message_segments(segments)

    @_with_read_lock
    def get_effective_adapters(self, active_turn: Optional[Turn]) -> List[str]:
        """
        Calculates the effective adapter stack for a given turn by replaying commands
        from the session history, including the new 'pop <target>' logic.
        """
        if not active_turn:
            return []

        path: List[Turn] = self.get_active_path_for_llm(active_turn)
        if not path:
            return []

        # 1. Collect all relevant commands
        all_ops: List[Tuple[Turn, Command]] = []
        for turn in path:
            for cmd in turn.cmd:
                if cmd.cmd_type == Command.STATE_CHANGE and cmd.data.get("change") == "adapters_command":
                    all_ops.append((turn, cmd))

        # 2. Process pops inline to build the effective command list.
        filtered_ops: List[Tuple[Turn, Command]] = []
        for turn, cmd in all_ops:
            op_type = cmd.data.get("op")
            if op_type != "pop":
                filtered_ops.append((turn, cmd))
                continue

            target_id = cmd.data.get("stack_id")
            if not target_id:
                # simple pop: drop the most recent set/add
                for idx in range(len(filtered_ops) - 1, -1, -1):
                    if filtered_ops[idx][1].data.get("op") in {"set", "add"}:
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

        valid_ops: List[Command] = [cmd for _, cmd in filtered_ops]
        
        # 3. Process the filtered operations to build the final adapter stack
        adapter_stack: List[List[str]] = []

        for cmd in valid_ops:
            op = cmd.data.get("op")
            if op == "set":
                adapter_stack = [cmd.data.get("adapters", [])]
            elif op == "add":
                adapter_stack.append(cmd.data.get("adapters", []))
            elif op == "pop":
                # This is a simple pop, as targeted pops were already handled
                if adapter_stack:
                    adapter_stack.pop()

        # The effective adapters are the top of the stack
        if not adapter_stack:
            return [] 
        
        final_adapters = adapter_stack[-1]
        # Ensure __base__ is returned for an empty list, but allow an empty list if it was explicitly set
        if not final_adapters:
             # Check if the last operation was an explicit set to empty
            last_op = valid_ops[-1] if valid_ops else None
            if last_op and last_op.data.get("op") == "set" and last_op.data.get("adapters") == []:
                return []
            return ["__base__"]


        return final_adapters


    @_with_read_lock
    def _compute_spine_path(self, active: Turn) -> List[Turn]:
        """Return the path [root ... active] via .parent only."""
        path: List[Turn] = []
        cur = active
        seen = set()
        while cur is not None:
            if cur in seen:
                raise RuntimeError("_compute_spine_path: cycle detected.")
            seen.add(cur)
            path.insert(0, cur)
            cur = getattr(cur, "parent", None)
        return path

    def replace_turn_message(self, current_turn: Turn, role: str, content: Any) -> Turn:
        """
        Replaces the content of a specific role within the current turn's data.

        Args:
            current_turn: The turn to modify.
            role: The data key (e.g., 'user', 'tool_results') to replace.
            content: The new content for the specified role.
        """
        if not current_turn or current_turn.IsStructural:
            raise ValueError("Cannot replace message on a structural or non-existent turn.")
        if role not in current_turn.data:
            raise ValueError(f"Role '{role}' not found in the data for the current turn.")
        current_turn.data[role] = content
        return current_turn

    @_with_write_lock
    def remove_logs(self, conversation_index: Optional[int] = None) -> int:
        """
        Removes all LOG commands from a specific conversation or the entire session.
        """
        chat_session = None
        if conversation_index is not None:
            if not (0 <= conversation_index < len(self.conversations)):
                raise IndexError("Conversation index out of range.")
            chat_session = self.get_conversation(conversation_index)

        log_cmd_ids = set()
        turns_to_process = self._get_all_turns(chat_session)

        for turn in turns_to_process:
            if not hasattr(turn, "cmd") or not turn.cmd:
                continue

            original_cmd_count = len(turn.cmd)
            logs_in_turn = [cmd for cmd in turn.cmd if cmd.cmd_type == Command.LOG]
            
            if not logs_in_turn:
                continue

            for cmd in logs_in_turn:
                if cmd.gen_id:
                    log_cmd_ids.add(cmd.gen_id)

            turn.cmd = [cmd for cmd in turn.cmd if cmd.cmd_type != Command.LOG]

        if not log_cmd_ids:
            return 0

        if hasattr(self, "commands_history"):
            self.commands_history = [
                cmd for cmd in self.commands_history if cmd.gen_id not in log_cmd_ids
            ]

        return len(log_cmd_ids)

    class WalkItem(NamedTuple):
        node: "Turn"
        depth: int # noqa
        relation: Literal["spine", "fork"] # noqa
        is_last_sibling: bool # noqa
        has_forks: bool # noqa
        context_for_labels: "Turn" # noqa
        is_peek: bool; peek_kind: Optional[str]; collected: bool # noqa

    def iter_spine_tree(
        self,
        active: "Turn",
        include_forks: bool = True,
        include_archived: bool = True,
        move_main_child_last_on_fork: bool = True, # noqa
        limit_to_active_branch: bool = False,
        detours_first: bool = False,
    ) -> Iterable["WalkItem"]:
        """
            Unified spine iterator.

            Modes:
            • Active-branch (/s h): limit_to_active_branch=True
                - Emits exactly the nodes used in /s p (get_active_path_for_llm).
                - Also emits one-hop sibling peeks (relation="fork") for each of those nodes.
                - Does NOT recurse into forks (peeks only), and ignores include_forks flag.

            • Full-tree (/s hfs): include_forks=True, limit_to_active_branch=False
                - Full DFS from root, relation="spine" for physical first-child chain,
                    relation="fork" for all other branches, recursing into all children.

            All other combinations remain backward-compatible and conservative.
        """        
        with self._rw_lock.read_lock():
            WalkItem = self.WalkItem  # alias
            spine = self._compute_spine_path(active)
            on_spine = lambda n: n in spine
            SPINE: Literal["spine"] = "spine"
            FORK: Literal["fork"] = "fork"

            # --------- Mode A: /s h (active-branch only + peeks) ----------
            if limit_to_active_branch:
                prompt_nodes = self.get_active_path_for_llm(active)
                if not prompt_nodes:
                    return
                active_set = set(prompt_nodes)
                yielded: set[str] = set()   # for non-peek (spine) items only; peeks may duplicate on purpose

                def _key(t: "Turn") -> str:
                    return t.gen_id or f"obj:{id(t)}"

                # Helper: does this child (or its subtree) contribute to active path?
                def _subtree_contains_active(t: "Turn") -> bool:
                    # cheap check first
                    if t in active_set:
                        return True
                    # climb from every active node to see if we pass through t
                    for a in prompt_nodes:
                        n = a
                        while n:
                            if n is t:
                                return True
                            n = n.parent
                    return False

                # Emit each prompt node once; then emit peeks for its OFF-PATH CHILDREN,
                # in the children's natural order (no detours-first here).
                for i, p in enumerate(prompt_nodes):
                    if (not include_archived) and getattr(p, "is_archived", False):
                        continue
                    k = _key(p)
                    if k not in yielded:
                        yielded.add(k)
                        yield WalkItem(
                            node=p,
                            depth=0,
                            relation=SPINE,
                            is_last_sibling=(i == len(prompt_nodes) - 1),
                            has_forks=False,
                            context_for_labels=active,
                            is_peek=False, peek_kind=None, collected=False,
                        )

                    # Peeks: OFF-PATH CHILDREN OF p (true fork point), in natural order.
                    children = list(getattr(p, "turns", []) or [])
                    for ch in children:
                        if ch in active_set:
                            # keep the natural position by emitting a peek for the path child too,
                            # but mark it collected=True so printers can choose different styling.
                            is_main_tryout = getattr(ch, "main_thread", False)
                            pk_kind = "Fork" if getattr(p, "turn_type", None) == Turn.FORK else "TryOut"
                            if pk_kind == "TryOut":
                                pk_kind = f"TryOut, main" if is_main_tryout else f"TryOut, inactive"
                            yield WalkItem(
                                node=ch,
                                depth=1,
                                relation=FORK,  # displayed as a one-liner peek
                                is_last_sibling=False,
                                has_forks=False,
                                context_for_labels=active,
                                is_peek=True, peek_kind=pk_kind, collected=True,
                            )
                            continue
                        # off-path child
                        if getattr(ch, "IsStructural", False):
                            # peek its immediate REAL children
                            for c in list(getattr(ch, "turns", []) or []):
                                if c.IsPlaceholderLike:
                                    continue
                                is_main_tryout = getattr(c, "main_thread", False)
                                pk_kind = "Fork" if getattr(ch, "turn_type", None) == Turn.FORK else "TryOut"
                                if pk_kind == "TryOut": pk_kind = f"TryOut, main" if is_main_tryout else f"TryOut, inactive"
                                yield WalkItem(
                                    node=c,
                                    depth=1,
                                    relation=FORK,
                                    is_last_sibling=False,
                                    has_forks=False,
                                    context_for_labels=active,
                                    is_peek=True, peek_kind=pk_kind, collected=_subtree_contains_active(c),
                                )
                        else:
                            is_main_tryout = getattr(ch, "main_thread", False)
                            pk_kind = "Fork" if getattr(p, "turn_type", None) == Turn.FORK else "TryOut"
                            if pk_kind == "TryOut": pk_kind = f"TryOut, main" if is_main_tryout else f"TryOut, inactive"
                            yield WalkItem(
                                node=ch,
                                depth=1,
                                relation=FORK,
                                is_last_sibling=False,
                                has_forks=False,
                                context_for_labels=active,
                                is_peek=True, peek_kind=pk_kind, collected=_subtree_contains_active(ch),
                            )
                return

            # --------- Mode B: /s hfs (full tree) ----------
            # We require include_forks=True to descend fully.
            root = spine[0] if spine else active

            def children_of(node: "Turn") -> List["Turn"]:
                kids: List["Turn"] = list(getattr(node, "turns", []) or [])
                # Optional cosmetic tweak for FORK hubs
                if node.turn_type == Turn.FORK and move_main_child_last_on_fork and kids:
                    try:
                        if getattr(kids[0], "main_thread", False):
                            main = kids.pop(0)
                            kids.append(main)
                    except Exception:
                        pass
                if not kids:
                    return kids
                # If we want to mirror LLM prompt ordering in displays, show detours before spine child
                if detours_first:
                    spine_child = [k for k in kids if k in spine]
                    off_spine   = [k for k in kids if k not in spine]
                    return off_spine + spine_child
                return kids

            def walk(
                node: "Turn",
                depth: int,
                relation: Literal["spine", "fork"],
                context_for_labels: "Turn",
            ):
                if (not include_archived) and getattr(node, "is_archived", False):
                    return

                # Yield the current node itself
                has_forks = any(not on_spine(ch) for ch in (getattr(node, "turns", []) or []))
                yield WalkItem(node, depth,
                            relation=relation,
                            is_last_sibling=False,
                            has_forks=has_forks,
                            context_for_labels=context_for_labels,
                            is_peek=False, peek_kind=None, collected=True)

                # Recurse into children
                if include_forks or relation == SPINE:
                    kids = children_of(node)
                    for i, child in enumerate(kids):
                        rel: Literal["spine", "fork"] = SPINE if on_spine(child) else FORK
                        # Keep labels anchored to the *global* active node; not per-item context
                        ctx = active
                        child_depth = depth if rel == SPINE else depth + 1
                        last = (i == len(kids) - 1)
                        first = True
                        for item in walk(child, child_depth, rel, ctx):
                            if first:
                                yield item._replace(is_last_sibling=last)  # type: ignore
                                first = False
                            else:
                                yield item

            # Start walk; use `active` as label context for consistent IDs
            yield from walk(root, depth=0, relation=SPINE, context_for_labels=active)

    @_with_read_lock
    def iter_spine_tree_snapshot(
        self,
        active: "Turn",
        include_forks: bool = True,
        include_archived: bool = True,
        move_main_child_last_on_fork: bool = True, # noqa
        limit_to_active_branch: bool = False,
        detours_first: bool = False,
    ) -> List["WalkItem"]:
        return list(
            self.iter_spine_tree(
                active=active,
                include_forks=include_forks,
                include_archived=include_archived,
                move_main_child_last_on_fork=move_main_child_last_on_fork,
                limit_to_active_branch=limit_to_active_branch,
                detours_first=detours_first,
            )
        )

    async def async_iter_spine_tree_snapshot(
        self,
        active: "Turn",
        include_forks: bool = True,
        include_archived: bool = True,
        move_main_child_last_on_fork: bool = True, # noqa
        limit_to_active_branch: bool = False,
        detours_first: bool = False,
    ) -> List["WalkItem"]:
        return await asyncio.to_thread(
            self.iter_spine_tree_snapshot,
            active,
            include_forks,
            include_archived,
            move_main_child_last_on_fork,
            limit_to_active_branch,
            detours_first,
        )

    @_with_read_lock
    def get_display_id(self, turn: "Turn",
                    active: Optional["Turn"] = None) -> str:
        """
        Display ID relative to ACTIVE path, with compressed first-child chains.

        Format:
        • On active path:            X-Y
        • Off active path:           X-A.[1[.D]][.K[.K2...]]
            where:
            - X   = conversation index (1 for now)
            - A   = index (1-based) of the LCA on the active path
            - [1[.D]] represents a compressed chain of first-child continuations
                        starting at the LCA (or first non-structural step below it):
                · If there is at least one first-child hop, append ".1"
                · If the chain length > 1, append ".{length}" (so .1.2, .1.3, ...)
            - [.K...] are any subsequent non-first-child hops (1-based child indices)
                or structural-child hops (BATCH/FORK/try-out children are always explicit)

        Rationale:
        - Pure "continue" chains (first-child under non-structural parents) should not
            explode into .1.1.1...; they are rendered as a single ".1" plus a depth counter.
        - Structural hubs (BATCH/FORK) and second+ children always create explicit segments.

        Notes:
        - This only changes the DISPLAY. The underlying tree remains unchanged.
        """
        # ---------- Build ACTIVE path root→active ----------
        if active is None:
            active = turn
        active_path = []

        # The active turn should always be marked, even if it's the root.
        is_the_active_turn = (turn is active)

        n = active
        while n and getattr(n, "parent", None):
            n = n.parent
        root = n or active

        ancestry = set()
        tmp = active
        while tmp:
            ancestry.add(tmp)
            tmp = tmp.parent

        n = root
        while n:
            active_path.append(n)
            if n is active:
                break
            kids = list(getattr(n, "turns", []) or [])
            next_on_path = None
            for k in kids:
                if k in ancestry:
                    next_on_path = k
                    break
            n = next_on_path

        if not active_path:
            return getattr(turn, "gen_id", None) or "?"
        
        X = 0 
        for index, conv in enumerate(self.conversations):
            if conv.root_turn is root:  
                X = index+1
                    
        pos = {node: i + 1 for i, node in enumerate(active_path)}

        # ---------- On-path? Return X-Y ----------
        if turn in pos:
            return f"{X}-{pos[turn]}"

        # ---------- Find LCA of turn wrt active path ----------
        anc = turn
        while anc and anc not in pos:
            anc = getattr(anc, "parent", None)
        if anc is None:
            # Shouldn't happen; fall back
            return f"{X}-{pos.get(root, 1)}"

        A = pos[anc]
        segments: list[str] = []

        # ---------- Gather chain from LCA→turn ----------
        chain = []
        n = turn
        while n is not None and n is not anc:
            chain.append(n)
            n = getattr(n, "parent", None)
        chain.reverse()  # from child-of-LCA down to `turn`

        # Helpers to detect structural hubs
        def _is_structural(node: "Turn") -> bool:
            tt = getattr(node, "turn_type", None)
            # Treat FORK / BATCH as structural; extend here if you have explicit try-out flags.
            from_types = getattr(type(node), "__dict__", {})
            try:
                # Try to compare against Turn.FORK/Turn.BATCH if available
                TurnCls = type(node)
                return tt in (TurnCls.FORK, TurnCls.BATCH)
            except Exception:
                # Fallback: heuristics
                return bool(getattr(node, "IsFork", False) or getattr(node, "IsBatch", False))

        # ---------- Compressed-segment construction ----------
        parent = anc
        compress_run = 0         # length of consecutive first-child under NON-structural parents
        started = False          # becomes True after first structural/non-first segment is appended

        def _flush_compress():
            nonlocal compress_run, segments, started
            if compress_run > 0:
                # first-child chain exists → ".1"
                segments.append("1")
                # if the chain length > 1 → also append ".{length}"
                if compress_run > 1:
                    segments.append(str(compress_run))
                compress_run = 0
                started = True

        for node in chain:
            kids = list(getattr(parent, "turns", []) or [])
            try:
                idx = kids.index(node) + 1
            except ValueError:
                idx = 1

            if _is_structural(parent):
                # Structural hubs always create an explicit segment (even idx==1)
                _flush_compress()
                segments.append(str(idx))
                started = True
            else:
                if idx == 1:
                    # First-child continue under non-structural parent → compress
                    compress_run += 1
                else:
                    # Second+ child: terminate any compress run, then append explicit idx
                    _flush_compress()
                    segments.append(str(idx))
                    started = True

            parent = node

        # Flush any trailing continue chain
        _flush_compress()

        return f"{X}-{A}" + (("." + ".".join(segments)) if segments else "")


    # ----------------------- Renderless summaries ----------------------------
    def _take_text(self, s: Any, n: int = 40, is_strict: bool = False) -> str | None:
        if s is None:
            return None

        # Try to get content from a 'content' key if s is a dict
        if isinstance(s, dict):
            content = s.get("content")
        else:
            content = None

        # If no 'content' key, use the string representation of the payload itself
        if content is None:
            try:
                content = getattr(s, "content", None)  # NEW: Check for attribute "s"
            except:
                pass

        if content is None:
            return None
        else:
            text = str(content)

        if is_strict:
            return text

        s_clean = text.replace("\n", " ").strip()
        return s_clean if len(s_clean) <= n else s_clean[: max(0, n - 1)] + "…"

    def _fmt_jsonish(self, obj: Any, *, max_len: int = 80) -> str:
        try:
            s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            s = str(obj)
        return self._take_text(s, n=max_len) or ""

    @staticmethod
    def _is_echo_command(cmd: Command) -> bool:
        """Check if a command is an echo command."""
        if not cmd or cmd.cmd_type != Command.COMMAND:
            return False
        command = cmd.data.get("command", "") or ""
        return command.strip().startswith("/echo")

    @staticmethod
    def format_selected_engine_metrics(data: Dict[str, Any]) -> Optional[str]:
        metrics = data.get("$Response")
        if not isinstance(metrics, dict):
            return None

        def _fmt(val: Any) -> str:
            if isinstance(val, float):
                return f"{val:.2f}".rstrip("0").rstrip(".")
            return str(val)

        parts = []
        if (req := metrics.get("tracked_history_request_count")) is not None:
            parts.append(f"REQ={_fmt(req)}")
        if (tps := metrics.get("tracked_throughput_tps")) is not None:
            parts.append(f"TPS={_fmt(tps)}")
        if (busy := metrics.get("tracked_total_busy_time_sec")) is not None:
            parts.append(f"BUSY_SEC={_fmt(busy)}")
        if (peak := metrics.get("peak_tracked_gpu_mem_reserved_inference_gb")) is not None:
            parts.append(f"PEAK_RSV_MEM_GB={_fmt(peak)}")

        if not parts:
            return None
        return f"Last Tracked: [{', '.join(parts)}]"

    @_with_read_lock
    def summarize_turn(self, node: "Turn", *, active: "Turn", max_len: int = 80) -> str:
        """One-line summary used by /s hfs. Prefers assistant reply text; shows tool activity compactly."""
        # Structural nodes first
        ttype = getattr(node, "turn_type", None) # noqa
        data: Dict[str, Any] = getattr(node, "data", {}) or {}
 
        # --- NEW: Add status flags to the end of the summary ---
        flags = []
        if getattr(node, "is_archived", False): flags.append("archived")
        if getattr(node, "was_canceled", False): flags.append("canceled")
        if getattr(node, "was_truncated", False): flags.append("truncated")
        if getattr(node, "root_context", False): flags.append("root_ctx")
        if getattr(node, "do_continue", False): flags.append("continued")

        # --- NEW: More specific fork/try-out labeling ---
        is_main_thread = hasattr(node, "main_thread") and node.main_thread
        parent = getattr(node, "parent", None)
        if parent:
            try:
                child_index = parent.turns.index(node) + 1
            except (ValueError, AttributeError):
                child_index = 0

            if parent.turn_type == Turn.FORK:
                flags.append(f"fork_{child_index}")
            elif parent.turn_type == Turn.BATCH:
                flags.append(f"prompt_{child_index}") # This is correct for BATCH children
            elif child_index > 1:
                try_out_number = child_index - 1
                flags.append(f"try_out_{try_out_number}")

        if is_main_thread: flags.append("main")
        if parent and not is_main_thread and node is not active:
            is_try_out_sibling = len(parent.turns) > 1
            if parent.turn_type == Turn.BATCH or is_try_out_sibling:
                flags.append("inactive")

        # Check for echo commands
        for cmd in getattr(node, "cmd", []):
            if self._is_echo_command(cmd):
                flags.append("echo")
                break

        metrics_facts = []
        if node.metrics:
            if in_tok := node.metrics.get("input_tokens"): metrics_facts.append(f"in:{in_tok}")
            if out_tok := node.metrics.get("output_tokens"): metrics_facts.append(f"out:{out_tok}")
            if tps := node.metrics.get("tokens_per_second"): metrics_facts.append(f"tps:{tps:.1f}")
            elif overall_tps := node.metrics.get("overall_tps"): metrics_facts.append(f"tps:{overall_tps:.1f}")
            if gen_time := node.metrics.get("generation_duration_sec"): metrics_facts.append(f"gt:{gen_time:.1f}s")
            elif total_gen_duration := node.metrics.get("total_generation_duration_sec"): metrics_facts.append(f"tgt:{total_gen_duration:.1f}s")
            if cache_metric := node.metrics.get("cache_metric"): metrics_facts.append(f"cache:{cache_metric}")
            if was_truncated := node.metrics.get("was_truncated"): metrics_facts.append(f"trunc:{was_truncated}")

        tool_facts = []
        assistant_summary_override = None
        calls_facts = []
        tool_call_summaries: List[str] = []
        if assistant_msg := data.get("assistant"):
            if tool_blocks := assistant_msg.get("tool_blocks"):
                tool_facts.append(f"tools:{len(tool_blocks)}")
                if text_content := self._take_text(assistant_msg, is_strict=True):
                    if text_content.strip():
                        # Only show assistant text; tool/error snippets stay in tool facts.
                        assistant_summary_override = f"'{self._ellipsize(text_content, n=20)}'"

                # Summarize tool activity in the facts suffix instead of assistant text.
                tool_call_count = sum(len(b.calls) for b in tool_blocks if isinstance(b, ToolCallBlock))
                calls_facts.append(f"tools:{len(tool_blocks)}")
                calls_facts.append(f"calls:{tool_call_count}")
                # Flag if any parse/call errors exist.
                any_err = any(
                    (isinstance(b, ToolCallBlock) and (b.parse_errors or any(getattr(c, 'error', None) or (isinstance(c, dict) and c.get('error')) for c in b.calls)))
                    for b in tool_blocks
                )
                if any_err:
                    calls_facts.append("tool_err:1")
                for block in tool_blocks:
                    if not isinstance(block, ToolCallBlock):
                        continue
                    if not block.calls:
                        tool_call_summaries.append("<empty tool block>")
                        continue
                    for call in block.calls:
                        tool_call_summaries.append(self._format_tool_call_summary(call))
        if "tool_results" in data:
            calls_facts.append(f"results:{len(data['tool_results'])}")
        
        all_facts = flags + calls_facts # Combine all facts
        flags_suffix = f" {Colors.DIM}({', '.join(all_facts)}){Colors.RESET}" if all_facts else ""

        # --- NEW: Handle placeholders explicitly ---
        if ttype is None:
            return f"(placeholder){flags_suffix}" # noqa

        if ttype == Turn.FORK:
            main_children = sum(1 for child in node.turns if getattr(child, "main_thread", False))
            fork_children = len(node.turns) - main_children # noqa
            return f"<FORK> {{main:{main_children}, forks:{fork_children}}}{flags_suffix}"

        if ttype == Turn.BATCH:
            cmd = data.get("command") or data.get("cmd") or data.get("note") or "Batch"
            count = data.get("batch_count") or data.get("count") or "N/A"
            return f"<BATCH> {cmd} ({count}){flags_suffix}"

        # --- NEW: Dynamic role summarization ---
        # Discover roles present in the turn's data, excluding special/internal keys.
        suppressed_roles = {"$RequestParams","$try_out"}
        present_roles = [role for role in data.keys() if role not in suppressed_roles]
        
        if not present_roles:
            return "(no content)" + flags_suffix

        # Prioritize 'user' and 'assistant' for a more conversational summary if they exist.
        summary_parts = []
        if 'user' in present_roles:
            summary_parts.append(f"user: '{self._ellipsize(data['user'].get('content', ''))}'")
            present_roles.remove('user')

        if 'tool_results' in present_roles and isinstance(data['tool_results'], list):
            tool_results_list = data['tool_results']
            calls = list(self._iter_tool_calls(tool_results_list))
            if calls:
                max_show = 2
                shown = [self._format_tool_result_summary(call) for call in calls[:max_show]]
                extra = f" (+{len(calls) - max_show} more)" if len(calls) > max_show else ""
                summary_parts.append(f"tool_results: '{', '.join(shown)}{extra}'")
            elif tool_results_list:
                # Fallback for unexpected formats (try error/raw blocks before raw object strings)
                fallback_items: List[str] = []
                for item in tool_results_list:
                    if isinstance(item, ToolCallBlock):
                        if item.error_block and item.error_block.strip():
                            fallback_items.append(self._ellipsize(item.error_block))
                        elif item.raw_block and item.raw_block.strip():
                            fallback_items.append(self._ellipsize(item.raw_block))
                        elif item.normalized_block and item.normalized_block.strip():
                            fallback_items.append(self._ellipsize(item.normalized_block))
                        else:
                            fallback_items.append("<empty tool block>")
                    else:
                        fallback_items.append(self._ellipsize(str(item)))
                if fallback_items:
                    max_show = 2
                    shown = fallback_items[:max_show]
                    extra = f" (+{len(fallback_items) - max_show} more)" if len(fallback_items) > max_show else ""
                    summary_parts.append(f"tool_results: '{', '.join(shown)}{extra}'")
            present_roles.remove('tool_results')

        if tool_call_summaries:
            max_show = 2
            shown = tool_call_summaries[:max_show]
            extra = f" (+{len(tool_call_summaries) - max_show} more)" if len(tool_call_summaries) > max_show else ""
            summary_parts.append(f"tool_calls: '{', '.join(shown)}{extra}'")

        if 'assistant' in present_roles:
            if assistant_summary_override:
                summary_parts.append(f"assistant: '{assistant_summary_override}'")
            else:
                assistant_content = (data.get('assistant') or {}).get('content', '')
                if assistant_content:
                    summary_parts.append(f"assistant: '{self._ellipsize(assistant_content)}'")
            present_roles.remove('assistant')

        # Add any remaining roles.
        for role in sorted(present_roles):
            payload = data[role]
            # If payload is a dict with 'content', summarize that. Otherwise, summarize the whole payload.
            if isinstance(payload, dict) and 'content' in payload:
                summary_parts.append(f"{role}: '{self._ellipsize(str(payload.get('content')))}'")
            else:
                summary_parts.append(f"{role}: '{self._ellipsize(str(payload))}'")

        summary = " → ".join(summary_parts)
        return summary.replace("\n", " ") + flags_suffix

    @_with_write_lock
    def close_branch(self, current: "Turn") -> "Turn":
        """
        Close only the *immediate* try-out by returning the corresponding main-path node.

        Walk up until we find a parent where `current` is NOT the first child.
        Return that parent's FIRST child (the main continuation).
        If the entire chain to the root consists of first-children, we're on the original
        main branch -> raise, unless it's a direct batch continuation. This is now more batch-aware.
        """

        n = current
        while n and n.parent: # type: ignore
            parent = n.parent # type: ignore
            # --- REVISED BATCH-AWARE LOGIC ---
            # If the parent is a BATCH node, it means we are closing a prompt's branch (e.g., g_2 or g_3's branch).
            if parent.turn_type == Turn.BATCH:
                # If the current branch is NOT the main thread of the batch,
                # "closing" it means returning to the BATCH hub itself.
                # This allows the main loop to see that a non-main fork has completed.
                # The final `close_all_batches` will then correctly find the true main continuation.
                if not n.main_thread:
                    return parent # Signal to caller this was a non-main fork.
                # If it IS the main thread of the batch, we are on the main path.
                # We need to find the actual main continuation for the whole operation.
                return self.close_all_batches(parent)

            # --- REVISED LOGIC for non-batch forks (e.g., tool result try-outs) ---
            # This handles closing a temporary branch like a tool result.
            if not self._is_first_child(n):
                kids = list(parent.turns or [])
                # Verification: Ensure the parent has a first child to return to.
                if not kids:
                    raise RuntimeError(f"close_branch: Inconsistent state. Node '{self.get_display_id(n)}' ({n.gen_id_or_parent}) is not a first child, but its parent has no children.")

                # This is the "dive-in" logic. Find the main sibling (first child)
                # and return the leaf of that branch.
                main_node = kids[0] # type: ignore
                return self.get_last_turn_on_branch(main_node)
            n = n.parent # type: ignore

        # We never found a level where we were a second+ child -> already on very original main branch
        # Instead of raising an error, return the current turn. This makes the operation idempotent
        # and safer if called incorrectly on a main branch.
        # raise RuntimeError(f"close_branch: Cannot close branch starting from '{self.get_display_id(current)} ({current.gen_id_or_parent})' as it is already on the main branch.")
        return current

    @_with_write_lock
    def close_all_batches(self, first_batch_turn_created: Optional[Turn]) -> Optional[Turn]:
        """
        Structural helper used by chat logic to land on main-branch placeholder
        after batches complete. If multiple BATCH hubs are present, they can
        coexist
        """
        if not first_batch_turn_created:
            return None

        # --- REVISED: Handle direct batch continuation vs. try-out ---
        # Case 1: The batch was a direct continuation (not a try-out). It will be the first child of its parent.
        if first_batch_turn_created.turn_type == Turn.BATCH and self._is_first_child(first_batch_turn_created):
            # This was a direct batch. The new active turn is the leaf of its main branch.
            # Find the main child (first prompt) of this batch turn.
            main_prompt_turn = next((t for t in first_batch_turn_created.turns if t.main_thread), None)
            if main_prompt_turn:
                return self.get_last_turn_on_branch(main_prompt_turn)
            else:
                # Fallback: return the batch turn itself if no main child is found (should not happen).
                return first_batch_turn_created
        else:
            # Case 2: This was a try-out batch. The main continuation is the leaf of the main placeholder branch.
            p = first_batch_turn_created.parent
            if not p:
                return None # Should not happen if tree is valid

            if p and p.turns:
                main_placeholder = p.turns[0]
                if main_placeholder:
                    # Dive in to find the leaf of the main branch.
                    return self.get_last_turn_on_branch(main_placeholder)
            # If there's no parent or the parent has no children, something is structurally wrong.
            return None # Return None to signal failure to the caller.

    @_with_read_lock
    def get_active_path(self, start_node: Turn, stop_on_root_context: bool = False) -> List[Turn]:
        """
        Returns the list of turns from the root to the start_node, representing the active conversational path.
        This version walks strictly via parents (no sideways 'continuation' jumps), which ensures finiteness.
        If stop_on_root_context is True, the path will stop at the first ancestor that has `root_context=True`.

        """
        path: List[Turn] = []
        current: Optional[Turn] = start_node
        visited = set()

        while current is not None:
            if current in visited:
                # Structural cycle guard
                raise RuntimeError("get_active_path: cycle detected while ascending the tree.")
            visited.add(current)
            path.insert(0, current)
            if stop_on_root_context and current.root_context:
                break
            current = current.parent

        return path

    def _get_node_id(self, node: "Turn") -> str:
        """Returns a stable ID for a turn, using gen_id if available or a memory address fallback."""
        if getattr(node, 'gen_id', None):
            return node.gen_id
        return f"obj:{id(node)}"

    @_with_read_lock
    def _find_latest_turn_with_response(self, node: "Turn") -> Optional["Turn"]:
        """
        Walks the subtree rooted at `node` (depth-first, natural child order) and returns
        the last descendant that has an assistant reply. Useful when the branch's active
        leaf is a placeholder with no response.
        """
        latest: Optional["Turn"] = node if getattr(node, "HasResponse", False) else None
        stack: List["Turn"] = list(reversed(getattr(node, "turns", []) or []))

        while stack:
            current = stack.pop()
            if getattr(current, "HasResponse", False):
                latest = current
            if getattr(current, "turns", None):
                stack.extend(reversed(current.turns))

        return latest

    @_with_read_lock
    def get_active_path_for_llm(self, current_turn: "Turn") -> list["Turn"]:
        """
        Builds an ordered list of nodes for prompt construction using a fully recursive
        traversal that includes `main_thread` sibling branches at all levels.

        Rules (compact/spec-like):
        - Ascend from `current_turn` through parents, assembling oldest→newest.
        - Ascent halts at FORK hubs or nodes marked `root_context`.
        - Only while on the first-child (main) path of a parent do we collect promoted siblings
          (`main_thread=True`): those siblings are visited before the current node, using their
          active leaf; they are included if that leaf has a response or is placeholder-like,
          otherwise the most recent descendant with a response is used.
        - When starting from a non-first child (e.g., a batch try-out branch), no promoted-sibling
          detours are traversed for that level; the walk simply ascends.
        - Each turn is emitted once (visited set). Final ordering is conversational chronology.
        """
        if not current_turn:
            return []
        visited_ids = set()
        # The recursive helper returns the path in the correct final order (oldest to newest).
        return self._get_active_path_recursive(current_turn, visited_ids)

    @_with_read_lock
    def _get_active_path_recursive(self, node: Optional["Turn"], visited_ids: set[str]) -> list["Turn"]:
        """
        A recursive helper that constructs the LLM path in post-order traversal,
        ensuring oldest nodes come first.
        """
        if node is None or self._get_node_id(node) in visited_ids:
            return []

        # Stop traversal at FORK boundaries or explicit root contexts.
        if node.IsFork or node.root_context:
            node_id = self._get_node_id(node)
            if node_id not in visited_ids:
                visited_ids.add(node_id)
                return [node]
            return []

        # 1. Recursively build the path for the parent first. This establishes the base order.
        path = self._get_active_path_recursive(node.parent, visited_ids)

        # After the parent path is built, `visited_ids` is populated with all ancestors.
        # Now, process the current node if it hasn't been visited via another detour.
        node_id = self._get_node_id(node)
        if node_id in visited_ids:
            # This can happen if the node was already pulled in as part of a different detour.
            return path

        # 2. When on the first-child (main) path, traverse promoted siblings (main_thread=True)
        #    *before* adding the current node, so try-out branches stay chronologically
        #    before the continuation placeholder.
        if node.parent and node.parent.turns and node.parent.turns[0] is node:
            promoted_siblings = [s for s in node.parent.turns[1:] if getattr(s, "main_thread", False)]

            for sibling in promoted_siblings:
                sibling_id = self._get_node_id(sibling)
                if sibling_id in visited_ids:
                    continue

                leaf = self.get_last_turn_on_branch(sibling)
                has_response = getattr(leaf, "HasResponse", False)
                is_placeholder = getattr(leaf, "IsPlaceholderLike", False)

                if not has_response and not is_placeholder:
                    # If the active leaf is a placeholder, treat it as valid; otherwise try to
                    # locate the last descendant with a response before skipping the detour.
                    leaf = self._find_latest_turn_with_response(sibling) or leaf
                    has_response = getattr(leaf, "HasResponse", False)
                    is_placeholder = getattr(leaf, "IsPlaceholderLike", False)

                if not has_response and not is_placeholder:
                    continue
                detour_path = self._get_active_path_recursive(leaf, visited_ids)
                path.extend(detour_path)

        # 3. Add the current node to the path and mark it as visited.
        visited_ids.add(node_id)
        path.append(node)

        return path

    @_with_read_lock
    def get_effective_tools_scopes(self, current_turn: Optional["Turn"]) -> List[ToolsScope]:
        """
        Replays logged tools_scope commands along the active path for the provided turn.
        Returns the list of scopes currently in effect (LIFO order preserved).

        Behavior notes:
        - "set" replaces the stack; an empty/None scope clears it. A set without a mode leaves
          the toolbox global mode intact (typically "advertised"), so all active tools may remain
          advertised unless later scopes change mode or per-tool flags.
        - "add" appends a scope evaluated after prior layers. If it includes a mode, that mode
          overrides earlier ones for the view; otherwise it inherits the prior effective mode.
        - "pop"/"reset" remove scope layers (pop targets the newest or a specific id).
        """
        path: List["Turn"] = self.get_active_path_for_llm(current_turn) if current_turn else []
        scopes: List[ToolsScope] = []

        # 1) Collect all tools_scope commands along the active LLM path.
        all_ops: List[Tuple["Turn", Command]] = []
        for turn in path:
            for cmd in turn.cmd:
                if cmd.cmd_type == Command.STATE_CHANGE and cmd.data.get("change") == "tools_scope":
                    all_ops.append((turn, cmd))

        # 2) Process pops inline to build the effective command list.
        filtered_ops: List[Tuple["Turn", Command]] = []
        for turn, cmd in all_ops:
            op_type = (cmd.data.get("op") or "add").lower()
            if op_type != "pop":
                filtered_ops.append((turn, cmd))
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

        valid_ops = [cmd for _, cmd in filtered_ops]

        # 4) Replay filtered operations to reconstruct the scope stack.
        for cmd in valid_ops:
            op = (cmd.data.get("op") or "add").lower()
            scope_payload = cmd.data.get("scope")
            scope_obj = ToolsScope.from_dict(scope_payload) if scope_payload else None
            if op == "add":
                scopes.append(scope_obj or ToolsScope())
            elif op == "set":
                scopes = [scope_obj] if scope_obj and not scope_obj.is_noop() else []
            elif op == "pop":
                if scopes:
                    scopes.pop()
            elif op == "reset":
                scopes = []

        return scopes

    @_with_read_lock
    def get_tools_access(
        self,
        toolbox: Toolbox,
        current_turn: Optional["Turn"],
        *,
        label: Optional[str] = None,
    ) -> ToolsAccess:
        scopes = self.get_effective_tools_scopes(current_turn)
        return toolbox.create_access(scopes=scopes, label=label)



    @_with_write_lock
    def trim_up(self, active_turn: Turn, levels_to_trim: int) -> Tuple[Optional[Turn], List[Command], bool, str]:
        """
        Trims the active branch by the specified number of non-auto user turns,
        collecting and replaying parameter and state change commands to the new active turn.

        Args:
            active_turn: The current active turn from which to start trimming.
            levels_to_trim: The number of user turns to trim.

        Returns:
            A tuple of (new_active_turn, commands_to_reverse, success, message).
        """
        if not active_turn:
            return None, [], False, "No active chat turns to trim."

        if levels_to_trim == 0:
            # Clear the current turn in place while preserving its gen_id.
            if getattr(active_turn, "turns", None):
                deleted_ids = {id(node) for node in active_turn.get_all_descendants()}
                self.commands_history = [
                    cmd for cmd in self.commands_history
                    if getattr(cmd, "parent", None) is None or id(cmd.parent) not in deleted_ids
                ]
            self._clear_turn_for_trim0(active_turn)
            return active_turn, [], True, "Cleared current turn."

        # 1. Determine the starting point for trimming. If the active turn is a
        #    placeholder, start from its parent.
        start_trim_node = active_turn
        if start_trim_node.IsEmpty and start_trim_node.parent:
            start_trim_node = start_trim_node.parent

        # 2. Find the turn to delete by ascending and counting non-auto user turns.
        path_to_active = self.get_active_path(start_trim_node)
        path_to_active.reverse()  # Traverse from active turn up to root

        turn_to_delete = None
        user_turns_counted = 0
        for turn in path_to_active:
            if turn.data.get("user") and not turn.is_auto:
                user_turns_counted += 1
            if user_turns_counted >= levels_to_trim:
                turn_to_delete = turn
                break
        
        if not turn_to_delete:
            return active_turn, [], False, f"Cannot trim {levels_to_trim} user turn(s) up from the current turn."
        
        new_parent = turn_to_delete.parent

        if new_parent and new_parent.turn_type == Turn.BATCH:
            # Trimming a direct batch child removes the batch hub in place.
            batch_turn = new_parent
            deleted_descendants = list(batch_turn.get_all_descendants())
            deleted_ids = {id(node) for node in deleted_descendants}
            self.commands_history = [
                cmd for cmd in self.commands_history
                if getattr(cmd, "parent", None) is None or id(cmd.parent) not in deleted_ids
            ]

            commands_to_reverse: List[Command] = []
            state_changing_types = [Command.PARAM_CHANGE, Command.STATE_CHANGE]
            for turn in deleted_descendants:
                for cmd in turn.cmd:
                    if cmd.cmd_type in state_changing_types and cmd not in commands_to_reverse:
                        commands_to_reverse.append(cmd)

            self._soft_clear_turn(batch_turn)
            return batch_turn, commands_to_reverse, True, f"Trimmed batch at '{self.get_display_id(batch_turn)}'."

        if not new_parent:
            # This means we are trimming a child of the root turn.
            # The session will be reset to an empty state.
            root_turn = self.get_conversation(0).root_turn
            
            # Collect all turns to be deleted for command history cleanup
            deleted_subtree_turns = []
            for child in root_turn.turns:
                deleted_subtree_turns.append(child)
                deleted_subtree_turns.extend(list(child.get_all_descendants()))
            
            deleted_ids = {id(node) for node in deleted_subtree_turns}
            self.commands_history = [
                cmd for cmd in self.commands_history
                if getattr(cmd, "parent", None) is None or id(cmd.parent) not in deleted_ids
            ]

            root_turn.turns.clear()
            root_turn.data.clear()
            
            new_placeholder = self._add_chat_turn(root_turn, Turn(turn_type=None))
            return new_placeholder, [], True, "Trimmed all user turns. Session has been reset."

        # 3. Get all turns in the subtree being deleted, including the root of the subtree.
        deleted_subtree_turns = [turn_to_delete] + list(turn_to_delete.get_all_descendants())

        # 4. Find all state-changing commands within the deleted subtree to return for reconciliation.
        commands_to_reverse: List[Command] = []
        state_changing_types = [Command.PARAM_CHANGE, Command.STATE_CHANGE]

        # Iterate through the turns being deleted and collect their state-changing commands.
        for turn in deleted_subtree_turns:
            for cmd in turn.cmd:
                # If it was a state-changing command, it still needs to be flagged for reversal.
                if cmd.cmd_type in state_changing_types and cmd not in commands_to_reverse:
                    commands_to_reverse.append(cmd)

        # 5. Perform the trim.
        new_parent.turns.remove(turn_to_delete)

        # 6. Create a new placeholder turn to make the branch active again.
        new_placeholder = self._add_chat_turn(new_parent, Turn(turn_type=None))

        return new_placeholder, commands_to_reverse, True, f"Trimmed {levels_to_trim} level(s). New active turn is '{self.get_display_id(new_placeholder)}'."

    def _soft_clear_turn(self, turn: Turn) -> None:
        """Clear a turn in place, preserving its gen_id and non-TURN commands."""
        if not turn:
            return
        for child in list(getattr(turn, "turns", []) or []):
            child.parent = None
        turn.turns.clear()
        turn.data = {}
        turn.metrics = {}
        turn.is_archived = False
        turn.main_thread = False
        turn.do_continue = False
        turn.was_truncated = False
        turn.was_canceled = False
        turn.is_auto = False
        if turn.parent is not None:
            turn.root_context = False
        turn.turn_type = Turn.HOLD if turn.parent is not None else turn.turn_type
        turn.cmd = [cmd for cmd in (turn.cmd or []) if cmd.cmd_type != Command.TURN]

    def _clear_turn_for_trim0(self, turn: Turn) -> None:
        """Clear a turn in place for trim(0), removing non-TURN commands."""
        if not turn:
            return
        for child in list(getattr(turn, "turns", []) or []):
            child.parent = None
        turn.turns.clear()
        turn.data = {}
        turn.metrics = {}
        turn.is_archived = False
        turn.main_thread = False
        turn.do_continue = False
        turn.was_truncated = False
        turn.was_canceled = False
        turn.is_auto = False
        if turn.parent is not None:
            turn.root_context = False
            turn.turn_type = Turn.HOLD
        if turn.cmd:
            removed = list(turn.cmd)
            if removed:
                self.remove_commands_from_history(removed)
            turn.cmd = []

    def _ellipsize(self, s: str, n: int = 40) -> str:
        s = (s or "").replace("\n", " ").strip()
        return s if len(s) <= n else s[: max(0, n - 1)] + "…"

    def _iter_tool_calls(self, tool_results_list: List[Any]) -> Iterable[ToolCall]:
        for item in tool_results_list:
            if isinstance(item, ToolCallBlock):
                for call in item.calls:
                    if isinstance(call, ToolCall):
                        yield call
            elif isinstance(item, ToolCall):
                yield item

    def _format_tool_result_summary(self, call: ToolCall, max_len: int = 24) -> str:
        value = call.result if call.result is not None else call.error
        text = "" if value is None else str(value)
        return self._ellipsize(text, n=max_len)

    def _format_tool_call_summary(self, call: Any, max_len: int = 48) -> str:
        name = getattr(call, "name", "") if not isinstance(call, dict) else call.get("name", "")
        args = getattr(call, "arguments", None) if not isinstance(call, dict) else call.get("arguments")
        args_text = ""
        if isinstance(args, dict):
            parts = []
            for key, value in args.items():
                parts.append(f"{key}={self._ellipsize(str(value), n=16)}")
            args_text = " ".join([p for p in parts if p])
        elif args is not None:
            args_text = self._ellipsize(str(args), n=24)
        label = f"{name} {args_text}".strip()
        return self._ellipsize(label, n=max_len)

    # ---------- Message extraction (one pair per turn) ----------

    def _messages_from_turn(self, node: Turn, parser: Optional[UnifiedToolIO] = None, is_strict: bool = False) -> List[Dict[str, str]]:
        """
        Build messages for one turn based on node.data:
        - user (string or dict)
        - tool_results: list (each becomes a tool message)
        - assistant: string or dict (may include tool_blocks and/or text)
        Order: user -> tool_results -> assistant
        """
        profile = getattr(parser, "profile", None) if parser else None
        results_as_user_role = bool(getattr(profile, "results_as_user_role", False))
        data = getattr(node, "data", {}) or {} # type: ignore
        out: List[Dict[str, Any]] = []

        # --- User-side messages (all keys except 'assistant') ---
        # The order is preserved from dict insertion (Python 3.7+)
        for role, payload in data.items():
            if role == "assistant" or role.startswith("$"): # Assistant message is handled separately later, special roles are ignored
                continue
    
            # Handle 'tool_results' which is a list of ToolCall objects
            if role == "tool_results":
                results_serialized = None
                result_role = "user" if results_as_user_role else "tool"

                if payload and isinstance(payload[0], ToolCallBlock):
                    if parser:
                        results_serialized = ToolsParserHelper.serialize_blocks(parser.profile, payload, is_result=True) # type: ignore
                elif payload and isinstance(payload[0], ToolCall):
                    if parser: # type: ignore
                        results_serialized = parser.serialize_calls(payload, is_result=True)
                    else:
                        results = [str(item.result) for item in payload if getattr(item, "result", None)]
                        if results:
                            results_serialized = "[" + ", ".join(results) + "]" if len(results) > 1 else results[0]
                        else:
                            results_serialized = ""
                else: # Legacy format where content might already be a string
                    for item in payload:
                        content = str(item.get("content", ""))
                        if not content: continue
                        if results_as_user_role and parser:
                            start_wrapper = getattr(parser.profile, 'result_wrapper_start', '') or ''
                            end_wrapper = getattr(parser.profile, 'result_wrapper_end', '') or ''
                            if start_wrapper and not content.strip().startswith(start_wrapper):
                                if content.strip(): content = f"{start_wrapper}{content}{end_wrapper}"
                        out.append({"role": result_role, "content": content})
                    continue

                if results_serialized is None:
                    continue

                if isinstance(results_serialized, list):
                    if results_serialized and isinstance(results_serialized[0], dict):
                        result_role = "user" if results_as_user_role else "tool_results"
                        for item in results_serialized:
                            out.append({"role": result_role, **item})
                        continue
                    results_serialized = "\n".join(str(item) for item in results_serialized if str(item) != "")

                if results_serialized is None:
                    continue
                else:
                    # If all results were stripped, emit an empty user-role message to preserve ordering.
                    if results_serialized == "" and results_as_user_role:
                        out.append({"role": "user", "content": ""})
                    elif results_serialized != "":
                        out.append({"role": "user" if results_as_user_role else "tool", "content": results_serialized})
            else: # Handle 'user' and any other custom roles
                text_content = self._take_text(payload, is_strict=is_strict)
                if text_content is not None:
                    # Use the key as the role, but default to 'user' if the payload doesn't specify one.
                    message_role = payload.get("role") if isinstance(payload, dict) else role
                    out.append({"role": message_role, "content": text_content})

        asst = data.get("assistant")
        if isinstance(asst, dict): # Modern format with content and/or tool_blocks
            asst_text = self._take_text(asst, is_strict=is_strict)
            tool_blocks = asst.get("tool_blocks") # noqa
            
            # Always populate assistant content
            assistant_message = {"role": "assistant", "content": asst_text or ""}
 
            if asst_text or tool_blocks:
                if tool_blocks and  isinstance(tool_blocks[0], ToolCallBlock):
                    if parser:
                        # Reconstruct the prompt content with normalized tool calls.
                        # The `reconstruct_prompt_with_tools` method now handles the full reconstruction.
                        # We pass the assistant message (containing the text part) and the tool blocks. # type: ignore
                        reconstructed_msg, _ = parser.reconstruct_prompt_with_tools(assistant_message, tool_blocks) # type: ignore
                        #print("original:")
                        #print(assistant_message["content"])
                        #print("ToolCall:")
                        #print(ToolCallBlock.serialize_calls(parser.profile, tool_blocks, is_result=False))
                        #print("result:")
                        #print(ToolCallBlock.serialize_calls(parser.profile, tool_blocks, is_result=True))
                        #print("reconstructed:")
                        #print(reconstructed_msg["content"])
                        assistant_message = reconstructed_msg
                    else:
                        # If no parser just insert back non normalized blocks that possibly have been cut out
                        asst_text = ToolsParserHelper.reconstruct_text_with_blocks(asst_text, tool_blocks) # type: ignore
                        assistant_message["content"] = asst_text
                        #print("no parser:")
                        #print(asst_text)
                elif tool_blocks:
                    #TBD: not ToolCallBlock[] format, just append?
                    assistant_message["content"] += str(tool_blocks) # type: ignore
                    #print("Raw tool block:")
                    #print(assistant_message["content"])
                
            out.append(assistant_message)

        elif asst: # Legacy format where assistant value is just the text content
            asst_text = self._take_text(asst, is_strict=is_strict) or  ""
            out.append({"role": "assistant", "content": asst_text})

        return out


    def _is_first_child(self, n: "Turn") -> bool:
        """True iff `n` is the first child of its parent (or has no parent)."""
        if not n:
            return True
        return n.is_first_child()

    def _right_siblings(self, n: "Turn") -> list["Turn"]:
        """Return Turn siblings to the right of `n` (in natural insertion order)."""
        if not n or not n.parent:
            return []
        kids = n.parent.turns or []
        try:
            i = kids.index(n)
        except ValueError:
            return []
        return kids[i+1:]

    def _active_leaf(self, n: "Turn") -> "Turn":
        """Return the active leaf of a branch (delegates to existing helper if present)."""
        return self.get_last_turn_on_branch(n)

    @_with_read_lock
    def get_llm_messages(
        self,
        current_turn: "Turn",
        parser: Optional["UnifiedToolIO"] = None,
        debug_format: bool = False,
        keep_all_turns: bool = False,
        include_content: bool = False, # New flag for content preview
    ) -> List[Dict[str, Any]]:
        """
        Materialize chat messages using the *separated* try-out semantics:

        • Uses get_active_path_for_llm() above (which already performs fully-recursive
        detour insertion with FORK clipping).
        • Filters out placeholders and structural nodes (FORK/BATCH) when emitting messages.
        • Deduplicates by node (already handled by the collector).
        """
        if not current_turn:
            return []  # type: ignore

        messages: List[Dict[str, Any]] = []
        debug_rows: List[Dict[str, Any]] = []

        sys = self.get_effective_system_message(current_turn)
        if sys is not None:
            messages.append({"role": "system", "content": sys})
            if debug_format and not include_content:
                debug_rows.append({"id": "system", "gen_id": "N/A", "msgs": ["system"], "info": []})

        path: List["Turn"] = self.get_active_path_for_llm(current_turn)

        def _emit_for(node: "Turn") -> List[Dict[str, Any]]:
            # Reuse your existing low-level extractor; keep behavior identical.
            return self._messages_from_turn(node, parser=parser, is_strict=not debug_format) or []

        for node in path:
            # Skip archived unless explicitly requested
            if node.is_archived and not keep_all_turns:
                if debug_format and not include_content:
                    debug_rows.append({
                        "id": self.get_display_id(node, active=current_turn), # type: ignore
                        "gen_id": getattr(node, "gen_id", "N/A"),
                        "msgs": [],
                        "info": ["excluded(archived)"]
                    })
                continue

            # Skip placeholder-like & structural nodes for the LLM payload
            if node.IsPlaceholderLike or node.IsStructural:
                if debug_format and not include_content:
                    reason = f"excluded({node.turn_type or 'placeholder'})"
                    # Provide minimal trace that a structural/placeholder existed
                    debug_rows.append({
                        "id": self.get_display_id(node, active=current_turn),
                        "gen_id": getattr(node, "gen_id", "N/A"),
                        "msgs": [],
                        "info": [reason]
                    })
                continue

            # Emit user/tool/assistant messages in order
            turn_msgs = _emit_for(node)
            msgs = turn_msgs
            if not msgs and not include_content:
                if debug_format:
                    debug_rows.append({
                        "id": self.get_display_id(node, active=current_turn),
                        "gen_id": getattr(node, "gen_id", "N/A"),
                        "msgs": [],
                        "info": ["no_msgs"]
                    })
                continue

            # --- Add back the internal keys for post-processing ---
            for msg in msgs:
                msg["_node_obj"] = node
                msg["_node_type"] = node.turn_type

            messages.extend(msgs)

            if debug_format and not include_content:
                debug_rows.append({
                    "id": self.get_display_id(node, active=current_turn), # type: ignore
                    "gen_id": getattr(node, "gen_id", "N/A"),
                    "msgs": [
                        m.get("role") or
                        ('tool_calls' if 'tool_blocks' in m else None) or
                        ('tool_results' if 'tool_results' in m else None) or
                        'unknown'
                        for m in msgs if isinstance(m, dict)
                    ],
                    "info": []
                })
            elif debug_format and include_content:
                debug_row = {
                    "id": self.get_display_id(node, active=current_turn),
                    "gen_id": node.gen_id_or_parent,
                    "msgs": [m.get("role") for m in msgs if m.get("role")],
                    "content_preview": [m.get("content") for m in msgs if m.get("content")],
                    "flags": [],
                    "info": []
                }
                if node.is_archived: debug_row["flags"].append("archived")
                if node.was_truncated: debug_row["flags"].append("truncated")
                if node.was_canceled: debug_row["flags"].append("canceled")
                if node.do_continue: debug_row["flags"].append("continued")
                if node.main_thread: debug_row["flags"].append("main")
                elif node.parent and node.parent.turn_type in [Turn.FORK, Turn.BATCH]:
                    debug_row["flags"].append("fork")
                if node.IsPlaceholderLike or node.IsStructural:
                    debug_row["info"].append(f"excluded({node.turn_type or 'placeholder'})")
                debug_rows.append(debug_row)

        # Post-process to handle continuations and remove internal keys
        if debug_format:
            return debug_rows
        else:
            return self._postprocess_llm_messages(messages, parser=parser, keep_all_turns=keep_all_turns)

    async def async_get_llm_messages(
        self,
        current_turn: "Turn",
        parser: Optional["UnifiedToolIO"] = None,
        debug_format: bool = False,
        keep_all_turns: bool = False,
        include_content: bool = False,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self.get_llm_messages,
            current_turn,
            parser,
            debug_format,
            keep_all_turns,
            include_content,
        )


    @_with_read_lock
    def _postprocess_llm_messages(self, messages: List[Dict[str, Any]], parser: Optional[UnifiedToolIO] = None, keep_all_turns: bool = False) -> List[Dict[str, Any]]:
        """
        Sanitizes and post-processes a list of messages to create a clean prompt for the LLM.
        - Removes placeholder and structural nodes.
        - Combines `was_truncated` and `do_continue` message sequences.
        - Removes internal keys like '_node_obj'.

        Args:
            keep_all_turns: If True, placeholder and structural nodes are kept.
        """
        if keep_all_turns:
            # For history views, just remove internal keys but keep all nodes.
            sanitized_messages = []
            for msg in messages:
                if "_node_type" not in msg:
                    clean_msg = {k: v for k, v in msg.items() if not k.startswith('_')}
                    sanitized_messages.append(clean_msg)
                else:
                    sanitized_messages.append(msg)
            return sanitized_messages

        prefix_messages: List[Dict[str, Any]] = []
        node_messages: List[Dict[str, Any]] = []
        for msg in messages:
            node = msg.get("_node_obj")
            if node:
                node_messages.append(msg)
            else:
                prefix_messages.append({k: v for k, v in msg.items() if not k.startswith('_')})

        # 1. First, sanitize the node-backed list to remove structural/placeholder nodes.
        sanitized_for_processing = [
            msg for msg in node_messages 
            if msg.get("_node_type") not in [Turn.FORK, Turn.BATCH] and 
               (msg.get("_node_obj") and not msg.get("_node_obj").IsEmpty)
        ]

        # 2. Now, combine `was_truncated` and `do_continue` sequences on the clean list.
        final_messages = []
        i = 0
        while i < len(sanitized_for_processing):
            current_node = sanitized_for_processing[i].get("_node_obj")
            if not current_node:
                i += 1
                continue
            
            turn_msgs = [m for m in sanitized_for_processing if m.get("_node_obj") == current_node]
            
            if getattr(current_node, 'was_truncated', False):
                combined_assistant_msg = None
                
                for msg in turn_msgs:
                    if msg.get("role") == "assistant":
                        combined_assistant_msg = copy.deepcopy(msg)
                    elif "_node_type" not in msg:
                        final_messages.append(msg)

                # Look ahead for `do_continue` turns
                j = i + len(turn_msgs) # Start looking from the next message
                while j < len(sanitized_for_processing):
                    next_node = sanitized_for_processing[j].get("_node_obj")
                    if next_node and getattr(next_node, 'do_continue', False):
                        continuation_msgs = [m for m in sanitized_for_processing if m.get("_node_obj") == next_node]
                        for msg in continuation_msgs:
                            if msg.get("role") == "assistant" and combined_assistant_msg:
                                # Stitch together assistant content
                                combined_assistant_msg["content"] = (combined_assistant_msg.get("content") or "") + (msg.get("content") or "") # type: ignore
                                # Also stitch tool blocks if they exist
                                if "tool_blocks" in msg and "tool_blocks" in combined_assistant_msg:
                                    combined_assistant_msg["tool_blocks"].extend(msg["tool_blocks"]) # type: ignore
                            elif "_node_type" not in msg:
                                # This handles user/tool messages that might be part of a continuation turn,
                                # although this is not a standard pattern.
                                final_messages.append(msg)
                        j += len(continuation_msgs)
                        # After a continuation, check if the *next* node is the one we are looking for.
                        # If it is, we need to process its messages before breaking the inner loop.
                        # This is the key fix.
                    else: # Not a `do_continue` turn, so the sequence ends.
                        break

                if combined_assistant_msg:
                    final_messages.append(combined_assistant_msg)
                i = j

            else: # Not a truncated turn, just add its messages
                # --- FIX: Ensure all messages from the current turn are added ---
                # The original logic was flawed and could miss messages. This is simpler and more robust.
                # We find all messages belonging to the current node and add them.
                node_msgs_to_add = [m for m in sanitized_for_processing if m.get("_node_obj") == current_node]
                for msg in node_msgs_to_add:
                    final_messages.append(msg)
                # Advance the main loop index by the number of messages we just processed.
                i += len(node_msgs_to_add)

        # 3. Final pass to remove all internal keys before sending to LLM.
        cleaned_messages = [{k: v for k, v in msg.items() if not k.startswith('_')} for msg in final_messages]
        return prefix_messages + cleaned_messages
