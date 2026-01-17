# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
# unified_toolcalls_robust.py
from __future__ import annotations

# --- Path fix for direct script execution ---
import sys
import os
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# --- End of path fix ---

import json, re, uuid
import copy
import logging
from typing import Any, Union, AsyncIterator, List, Dict, Any, Optional, Callable,  Iterable, Tuple

from .mp13_utils import StoringTextIteratorStreamer
from .mp13_config import ToolCall, ToolCallBlock, ParserProfile

DEFAULT_PROFILE = ParserProfile(
    key="unknown",
    block_start=["<tool_call>"], # A reasonable default
    block_end=["</tool_call>"],
    hard_stop_patterns=["</s>"],
    payload_mode="json_obj_or_list",
    name_field="name",
    arguments_field="arguments",
    arguments_may_be_string=True,
    preserve_eos_for_parsing=True,
)

BUILTIN_PROFILES: List[ParserProfile] = [
    ParserProfile(
        key="granite",
        block_start=["<|tool_call|>"],
        block_end=[], # Relies on hard stops
        hard_stop_patterns=["<|start_of_role|>", "<|end_of_text|>"],
        payload_mode="json_list",
        name_field="name",
        arguments_field="arguments",
        arguments_may_be_string=True,
        preserve_eos_for_parsing=True,
    ),
    ParserProfile(
        key="qwen",
        block_start=["<tool_call>"],
        block_end=["</tool_call>"],
        hard_stop_patterns=[],
        payload_mode="json_obj",
        name_field="name",
        arguments_field="arguments",
        arguments_may_be_string=True,
        result_wrapper_start="<tool_response>\n",
        result_wrapper_end="\n</tool_response>",
    ),
    ParserProfile(
        key="mistral",
        block_start=["[TOOL_CALLS]["],
        block_end=["]</s>", "]"], # Order matters: check for longer one first
        hard_stop_patterns=["</s>"],
        payload_mode="json_list_with_id",
        name_field="function.name",
        arguments_field="function.arguments",
        id_field="id",
        arguments_may_be_string=False,
        result_wrapper_start="[TOOL_RESULTS]",
        result_wrapper_end="[/TOOL_RESULTS]",
        result_requires_id=True,
        result_id_key="tool_call_id",
        enforce_tool_call_id_format=True,
        results_as_messages=True,
        results_message_content_key="content",
        emit_tool_calls_field=True,
        end_marker_outside_json_string=True,
        preserve_eos_for_parsing=True,
        # Note: results are emitted as one message per call with role=tool_results,
        # using keys tool_call_id + content (no nested output object) to match Mistral expectation.
    ),
    ParserProfile(
        key="mistral3",
        block_start=["[TOOL_CALLS]"],
        block_end=[], # Tagged calls don't use a dedicated end marker.
        hard_stop_patterns=["</s>"],
        payload_mode="json_list_with_id",
        name_field="function.name",
        arguments_field="function.arguments",
        arguments_may_be_string=False,
        result_wrapper_start="[TOOL_RESULTS]",
        result_wrapper_end="[/TOOL_RESULTS]",
        result_requires_id=False,
        results_as_messages=False,
        emit_tool_calls_field=True,
        end_marker_outside_json_string=True,
        tagged_call_args_marker="[ARGS]",
        preserve_eos_for_parsing=True,
    ),
    ParserProfile(
        key="phi4",
        block_start=["<|assistant|>"],
        block_end=["<|end|>"], # The parser logic will handle the end-of-string case
        hard_stop_patterns=[],
        payload_mode="json_obj_or_list",
        name_field="name",
        arguments_field="arguments",
        arguments_may_be_string=True,
        tools_in_system_prompt=True,
    ),
    ParserProfile(
        key="deepseek-r1",
        block_start=["<｜tool▁call▁begin｜>"],
        block_end=["<｜tool▁call▁end｜>"],
        hard_stop_patterns=["<｜tool▁calls▁end｜>", "<｜end▁of▁sentence｜>"],
        payload_mode="json_obj",
        name_field="function.name",
        arguments_field="function.arguments",
        arguments_may_be_string=False,
        result_wrapper_start="<｜tool▁outputs▁begin｜>",
        result_wrapper_end="<｜tool▁outputs▁end｜>",
        result_requires_id=False,
        tools_in_system_prompt=False,
        preserve_eos_for_parsing=True,
    ),
]

# ============================================================
# Utilities
# ============================================================

def _first_balanced_json_span(s: str) -> Tuple[int, int]:
    """
    Return (start,end) indices of the first balanced JSON object/array or (-1,-1) if none.
    Handles quotes/escapes.
    """
    start = -1
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                if depth == 0:
                    start = i
                depth += 1
            elif ch in "]}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        return start, i + 1
    return -1, -1


def _json_loads_loose(s: str) -> Any:
    s = s.strip()
    # strip code fences if present
    if s.startswith("```"):
        # remove opening fence
        s = s.split("\n", 1)[1] if "\n" in s else ""
        # remove closing fence
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

    def _collect_top_level_objects(txt: str) -> Optional[List[Any]]:
        """Collect multiple top-level JSON objects in a comma-separated stream."""
        objs = []
        idx = 0
        while idx < len(txt):
            start, end = _first_balanced_json_span(txt[idx:])
            if start == -1:
                break
            start += idx
            end += idx
            try:
                objs.append(json.loads(txt[start:end]))
            except Exception:
                return None
            idx = end
            while idx < len(txt) and txt[idx] in " \t\r\n,":
                idx += 1
        return objs if objs else None

    # try direct parse
    try:
        return json.loads(s)
    except Exception:
        # Heuristic: multiple top-level JSON objects separated by commas (common LLM error)
        stripped = s.lstrip()
        if stripped.startswith("{") and re.search(r"}\\s*,\\s*{", stripped):
            try:
                return json.loads(f"[{stripped}]")
            except Exception:
                objs = _collect_top_level_objects(stripped)
                if objs:
                    return objs
        st, en = _first_balanced_json_span(s)
        if st != -1:
            return json.loads(s[st:en])
        raise

def _get_nested(d: Dict[str, Any], dotted: str) -> Any:
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def _set_nested(d: Dict[str, Any], dotted: str, value: Any):
    """Sets a value in a nested dictionary using a dotted key."""
    keys = dotted.split('.')
    for key in keys[:-1]:
        if key not in d or not isinstance(d.get(key), dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def _double_decode_arguments(val: Any) -> Any:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {"_string_value": val}
    return val

def _strip_internal_tool_args(args: Any) -> Any:
    """
    Remove internal recovery/diagnostic keys from tool arguments.
    Best-effort: if args is a JSON string, decode first to avoid double-escaping
    in templates that apply `tojson`.
    """
    if isinstance(args, str):
        try:
            decoded = json.loads(args)
        except Exception:
            return args
        args = decoded
    if isinstance(args, dict):
        return {k: v for k, v in args.items() if not k.startswith('_') and k != 'tool_args_issue'}
    return args

def _clone_profile(key: str) -> ParserProfile:
    for p in BUILTIN_PROFILES:
        if p.key == key:
            # Return a shallow copy to allow overrides
            return ParserProfile(**p.__dict__)
    return DEFAULT_PROFILE

def _template_accepts_tools_arg(chat_template: str) -> bool:
    """
    Best-effort detection of whether a chat template references the `tools`
    variable, meaning `apply_chat_template(..., tools=...)` will have an effect.
    Only looks for the template variable, not static tool tokens.
    """
    if not chat_template:
        return False
    markers = [
        "{% if tools", "{{ tools", "{% for tool in tools", "{%- if tools", "{%- for tool in tools",
        "tools | tojson", "tools|tojson", "tools|json", "tools }}", "tools}}"
    ]
    t = chat_template.lower()
    return any(marker.lower() in t for marker in markers)

def guess_profile_from_template(chat_template: str, model_name: Optional[str]) -> ParserProfile:
    t = chat_template or ""
    m = (model_name or "").lower()

    def _with_template_meta(profile: ParserProfile) -> ParserProfile:
        # Annotate whether apply_chat_template will likely use the tools argument.
        accepts_tools = _template_accepts_tools_arg(t)
        profile.template_accepts_tools_arg = accepts_tools
        if not accepts_tools and not profile.tools_in_system_prompt:
            profile.tool_handling_hint = "Chat template does not reference 'tools'"
        return profile

    # DeepSeek R1-style markers (full-width pipes and U+2581 separators)
    if "<｜tool▁call▁begin｜>" in t:
        return _with_template_meta(_clone_profile("deepseek-r1"))

    if "<|tool_call|>" in t:
        return _with_template_meta(_clone_profile("granite"))
    if "<tools>" in t and "<tool_call>" in t:
        return _with_template_meta(_clone_profile("qwen"))
    if "[TOOL_CALLS]" in t and "[ARGS]" in t:
        return _with_template_meta(_clone_profile("mistral3"))
    if "[AVAILABLE_TOOLS]" in t or "[TOOL_CALLS][" in t:
        return _with_template_meta(_clone_profile("mistral"))
    if "<|tool|>" in t and "<|/tool|>" in t:
        return _with_template_meta(_clone_profile("phi4"))
    if "granite" in m:
        return _with_template_meta(_clone_profile("granite"))
    if m.startswith("qwen"):
        return _with_template_meta(_clone_profile("qwen"))
    if "mistral" in m:
        if "mistral3" in m or "ministral-3" in m or "mistral-3" in m:
            return _with_template_meta(_clone_profile("mistral3"))
        return _with_template_meta(_clone_profile("mistral"))
    if "phi" in m:
        return _with_template_meta(_clone_profile("phi4"))
    if "deepseek-r1" in m:
        return _with_template_meta(_clone_profile("deepseek-r1"))

    return _with_template_meta(DEFAULT_PROFILE)

def _ensure_list(x):
    if isinstance(x, list):
        return x
    return [x]

# ============================================================
# Public API
# ============================================================

class UnifiedToolIO:
    """
    Robust parser/serializer for tool calls.

    Public methods:
    - reset() -> None
      Clear internal stream state and collected blocks.

    - get_collected_blocks() -> List[ToolCallBlock]
      Return parsed tool blocks collected so far (includes an incomplete block if streaming).

    - finalize(tokenizer=None, force_incomplete=None) -> Optional[str]
      Finish stream parsing, optionally forcing incomplete block handling; returns leftover text.

    - parse_stream(...) -> AsyncIterator[str]
      Streaming parser that yields only plain text and stores tool blocks internally.

    - parse_model_output(text, ...) -> List[ToolCallBlock]
      Non-streaming, parse a complete response and extract tool blocks (best-effort recovery on malformed JSON).

    - parse_collected_blocks(blocks) -> None
      Populate ToolCall objects for previously collected ToolCallBlock instances.

    - serialize_calls(calls, is_result=False, block_action=None) -> Union[str, List[Dict[str, Any]]]
      Serialize tool calls or tool results using the active profile.
      
    - reconstruct_prompt_with_tools(original_message, tool_blocks) -> Tuple[Dict[str, str], Optional[Dict[str, str]]]
      Re-insert normalized tool calls into assistant content and build a tool-results message.

    """

    def __init__(self, profile: ParserProfile):
        """
        Initializes the parser with a specific profile.

        Args:
            profile: The ParserProfile object.
        """
        if isinstance(profile, ParserProfile):
            self.profile = profile
        elif isinstance(profile, dict):
            # If a dict is passed, instantiate the ParserProfile object from it.
            self.profile = ParserProfile(**profile)
        else:
            raise TypeError(f"Profile must be a ParserProfile object or a dict, not {type(profile).__name__}")

        self.collected_blocks: List[ToolCallBlock] = []
        self._stream_buffer = ""
        self._in_tool_code_block = False
        self._current_text_offset = 0
        self._current_block_start_pos = -1
        self._current_start_marker: Optional[str] = None
        self._absolute_offset = 0
        self._eos_id: Optional[int] = None
        self._eos_str: Optional[str] = None
        self._force_incomplete_on_stream_end = False
    def reset(self):
        """Resets the internal state of the parser for reuse."""
        self.collected_blocks.clear()
        self._stream_buffer = ""
        self._in_tool_code_block = False
        self._current_block_start_pos = -1
        self._current_text_offset = 0
        self._current_start_marker = None
        self._absolute_offset = 0
        self._eos_id = None
        self._eos_str = None
        self._force_incomplete_on_stream_end = False

    def get_collected_blocks(self) -> List[ToolCallBlock]:
        """
        Returns all tool blocks collected from the stream so far.
        If a block is currently being parsed, it will be included and marked as incomplete.
        """
        blocks_to_return = list(self.collected_blocks)
        if self._in_tool_code_block and self._stream_buffer:
            # Create a final, incomplete block from the buffer
            start_marker_len = len(self._current_start_marker or "")
            payload_start_pos = self._current_block_start_pos + start_marker_len
            payload_end_pos = payload_start_pos + len(self._stream_buffer)
            incomplete_block = ToolCallBlock(
                raw_block=self._stream_buffer,
                normalized_block=None,
                model_format=self.profile.key if self.profile else "unknown",
                block_start_pos=self._current_block_start_pos,
                block_end_pos=payload_end_pos,
                payload_start_pos=payload_start_pos,
                payload_end_pos=payload_end_pos,
                start_marker=self._current_start_marker,
                position_mode="full",
                is_incomplete=True,
                parse_errors=["Stream ended before closing tag was found."]
            )
            blocks_to_return.append(incomplete_block)
        return blocks_to_return

    def finalize(self, tokenizer: Optional[Any] = None, force_incomplete: Optional[bool] = None) -> Optional[str]:
        """
        Finalizes the stream, processing any remaining buffer. It attempts to recover a
        complete tool block from the buffer if the stream ended inside one.
        Returns any leftover text that should be yielded as user-facing content.
        """
        if force_incomplete is None:
            force_incomplete = self._force_incomplete_on_stream_end
        leftover_text = self._finalize_buffer(force_incomplete=force_incomplete)
        if leftover_text and tokenizer:
            # decode the final leftover text to strip any special tokens
            # that were not part of a valid marker.
            return tokenizer.decode(tokenizer.encode(leftover_text, add_special_tokens=False), skip_special_tokens=True)
        return leftover_text
        
    def _cleanup_eos_in_stream_buffer(self):
        # --- Clean all occurrences of EOS from stream buffer content ---
        if self._stream_buffer and self._eos_str:
            self._stream_buffer = self._stream_buffer.replace(self._eos_str, "")
            self._eos_id = None
            self._eos_str = None

    def _strip_sequences(self, text: str, sequences: Optional[Iterable[str]]) -> str:
        if not sequences:
            return text
        for seq in sequences:
            if seq:
                text = text.replace(seq, "")
        return text

    def _tagged_args_marker(self) -> Optional[str]:
        if not self.profile:
            return None
        marker = getattr(self.profile, "tagged_call_args_marker", None)
        if marker:
            return marker
        return None

    def _normalize_payload_text(self, payload_text: str) -> str:
        payload = payload_text.strip()
        if not payload:
            return payload
        start, end = _first_balanced_json_span(payload)
        if start != -1 and end != -1:
            before = payload[:start].strip()
            after = payload[end:].strip()
            # If there is additional structured content after the first balanced JSON (e.g., multiple objects),
            # keep the full payload so downstream loose parsing can capture them.
            if after and re.match(r"^[,\s]*[\[{]", after):
                return payload
            # Otherwise, trim to the balanced span (drops leading/trailing noise/code fences).
            return payload[start:end]
        return payload

    def _find_tagged_payload_end(self, text: str, payload_start_pos: int, args_marker: str) -> Optional[int]:
        """
        Finds the end position of the last args payload in tagged tool calls.
        Returns the absolute end position (exclusive) or None if no valid payload is found.
        """
        scan = text[payload_start_pos:]
        pos = 0
        last_end = -1
        while pos < len(scan):
            args_pos = scan.find(args_marker, pos)
            if args_pos == -1:
                break
            json_pos = args_pos + len(args_marker)
            while json_pos < len(scan) and scan[json_pos].isspace():
                json_pos += 1
            span_start, span_end = _first_balanced_json_span(scan[json_pos:])
            if span_start == -1:
                break
            span_start += json_pos
            span_end += json_pos
            last_end = span_end
            pos = span_end
        if last_end == -1:
            return None
        return payload_start_pos + last_end

    def _parse_tagged_calls(self, payload_text: str, raw_block: str, args_marker: str, call_marker: Optional[str]) -> Tuple[List[ToolCall], List[str]]:
        """
        Parses tagged tool calls: <call_marker>name<args_marker>{...}
        Returns (calls, errors).
        """
        calls: List[ToolCall] = []
        errors: List[str] = []
        if not payload_text:
            return calls, errors
        text = payload_text
        pos = 0
        while pos < len(text):
            # Optional marker between calls.
            if call_marker and text.startswith(call_marker, pos):
                pos += len(call_marker)
            while pos < len(text) and text[pos].isspace():
                pos += 1
            args_pos = text.find(args_marker, pos)
            if args_pos == -1:
                break
            name = text[pos:args_pos].strip()
            pos = args_pos + len(args_marker)
            while pos < len(text) and text[pos].isspace():
                pos += 1
            span_start, span_end = _first_balanced_json_span(text[pos:])
            if span_start == -1:
                errors.append("tagged_args_missing_json")
                break
            span_start += pos
            span_end += pos
            json_text = text[span_start:span_end]
            pos = span_end
            call_errors: List[str] = []
            if not name:
                call_errors.append("missing_tool_name")
                name = "malformed_tool_call"
            try:
                args_val = _json_loads_loose(json_text)
            except Exception as e:
                call_errors.append(f"mistral_tagged_args_parse_error: {type(e).__name__}: {e}")
                args_val = {"_non_parsed": json_text}
            if not isinstance(args_val, dict):
                call_errors.append("arguments_not_dict")
                args_val = {"_non_parsed": args_val}
            calls.append(ToolCall(
                name=name,
                arguments=args_val,
                raw=raw_block,
                model_format=self.profile.key if self.profile else None,
                parse_errors=call_errors,
            ))
        if not calls and args_marker in text:
            errors.append("tagged_calls_unparsed")
        return calls, errors


    def _finalize_buffer(self, upto_pos: Optional[int] = None, force_incomplete: bool = False) -> Optional[str]:
        """
        Internal finalization logic that can operate on the whole buffer or a prefix.
        If `upto_pos` is provided, it processes the buffer up to that position, leaving
        the rest. Otherwise, it processes the entire buffer.
        """
        if not self._in_tool_code_block:
            self._cleanup_eos_in_stream_buffer()
            return self._stream_buffer

        content_to_process = self._stream_buffer[:upto_pos] if upto_pos is not None else self._stream_buffer
        remaining_buffer = self._stream_buffer[upto_pos:] if upto_pos is not None else ""
        eos_seq = [self._eos_str] if self._eos_str else None

        tagged_args_marker = self._tagged_args_marker()
        is_tagged = bool(
            tagged_args_marker
            and self.profile
            and self._current_start_marker in (self.profile.block_start or [])
        )
        if is_tagged and tagged_args_marker:
            tagged_end = self._find_tagged_payload_end(content_to_process, 0, tagged_args_marker)
            if tagged_end is not None:
                recovered_block_content = content_to_process[:tagged_end]
                tail_content = content_to_process[tagged_end:] + remaining_buffer
                normalized_block = self._strip_sequences(recovered_block_content, eos_seq)
                normalized_block = normalized_block if normalized_block != recovered_block_content else None
                start_marker_len = len(self._current_start_marker or "")
                payload_start_pos = self._current_block_start_pos + start_marker_len
                payload_end_pos = payload_start_pos + len(recovered_block_content)
                self.collected_blocks.append(ToolCallBlock(
                    raw_block=recovered_block_content,
                    normalized_block=normalized_block,
                    model_format=self.profile.key,
                    block_start_pos=self._current_block_start_pos,
                    start_marker=self._current_start_marker,
                    block_end_pos=payload_end_pos,
                    payload_start_pos=payload_start_pos,
                    payload_end_pos=payload_end_pos,
                    position_mode="full",
                    is_incomplete=False,
                ))
                self._absolute_offset = payload_end_pos

                self._stream_buffer = tail_content
                self._in_tool_code_block = False
                self._cleanup_eos_in_stream_buffer()
                return self._stream_buffer
            if force_incomplete:
                normalized_block = self._strip_sequences(content_to_process, eos_seq)
                normalized_block = normalized_block if normalized_block != content_to_process else None
                start_marker_len = len(self._current_start_marker or "")
                payload_start_pos = self._current_block_start_pos + start_marker_len
                payload_end_pos = payload_start_pos + len(content_to_process)
                self.collected_blocks.append(ToolCallBlock(
                    raw_block=content_to_process,
                    normalized_block=normalized_block,
                    model_format=self.profile.key,
                    block_start_pos=self._current_block_start_pos,
                    start_marker=self._current_start_marker,
                    block_end_pos=payload_end_pos,
                    payload_start_pos=payload_start_pos,
                    payload_end_pos=payload_end_pos,
                    position_mode="full",
                    is_incomplete=True,
                    parse_errors=["Stream ended before tagged tool call completed."],
                ))
                self._absolute_offset = payload_end_pos
                self._stream_buffer = remaining_buffer
                self._in_tool_code_block = False
                self._cleanup_eos_in_stream_buffer()
                return self._stream_buffer

        # Attempt to recover a complete JSON object/array from the content
        start_char = content_to_process.lstrip()[:1]
        if start_char in "[{":
            # Keep the full sequence of balanced JSON objects, but avoid swallowing trailing user text.
            idx = 0
            last_end = -1
            while True:
                span_start, span_end = _first_balanced_json_span(content_to_process[idx:])
                if span_start == -1:
                    break
                span_start += idx
                span_end += idx
                last_end = span_end
                idx = span_end
                # consume separating commas/whitespace
                while idx < len(content_to_process) and content_to_process[idx] in " \t\r\n,":
                    idx += 1

            if last_end != -1:
                recovered_block_content = content_to_process[:last_end]
                tail_content = content_to_process[last_end:] + remaining_buffer
            else:
                recovered_block_content = content_to_process
                tail_content = remaining_buffer

            normalized_block = self._strip_sequences(recovered_block_content, eos_seq)
            normalized_block = normalized_block if normalized_block != recovered_block_content else None
            start_marker_len = len(self._current_start_marker or "")
            payload_start_pos = self._current_block_start_pos + start_marker_len
            payload_end_pos = payload_start_pos + len(recovered_block_content)
            self.collected_blocks.append(ToolCallBlock(
                raw_block=recovered_block_content,
                normalized_block=normalized_block,
                model_format=self.profile.key,
                block_start_pos=self._current_block_start_pos,
                start_marker=self._current_start_marker,
                block_end_pos=payload_end_pos,
                payload_start_pos=payload_start_pos,
                payload_end_pos=payload_end_pos,
                position_mode="full",
                is_incomplete=False, # Recovered successfully
            ))
            self._absolute_offset = payload_end_pos

            self._stream_buffer = tail_content # The rest is user text
            self._in_tool_code_block = False
            self._cleanup_eos_in_stream_buffer()
            return self._stream_buffer
        else:
            # The content inside the block did not start with a JSON marker.
            # This means it's just user text that was terminated by a hard stop.
            self._in_tool_code_block = False
            self._cleanup_eos_in_stream_buffer()
            return self._stream_buffer # Return the full buffer as it's all user text

    async def parse_stream(
        self,
        logger: logging.Logger,
        token_iterator: AsyncIterator[str],
        streamer: Optional[StoringTextIteratorStreamer] = None,
        mark_incomplete_on_stream_end: bool = False,
    ) -> AsyncIterator[str]:

        """
        Parses a stream of tokens, yielding only plain text.
        Tool call blocks are collected internally and can be retrieved via `get_collected_blocks`.
        """

        if not self.profile or not self.profile.block_start:
            async for token in token_iterator:
                #We do not want EOS/PAD in streaming user output
                eos_str = streamer.eos_token_decoded if streamer is not None else None
                if eos_str:
                    token = token.replace(eos_str, "")
                    streamer.eos_token_detected = None
                    streamer.eos_token_decoded = None
                if token:
                    self._absolute_offset += len(token)
                yield token
            return

        p = self.profile
        self._force_incomplete_on_stream_end = mark_incomplete_on_stream_end
        # Use raw strings for markers, not regex patterns
        start_markers = p.block_start
        end_markers = p.block_end
        hard_stop_markers = p.hard_stop_patterns
        all_markers = start_markers + end_markers + hard_stop_markers
        max_marker_len = max(len(m) for m in all_markers) if all_markers else 0

        async for token_text in token_iterator:
            self._stream_buffer += token_text

            self._eos_id = streamer.eos_token_detected if streamer is not None else None
            self._eos_str = streamer.eos_token_decoded if streamer is not None else None

            while True: # Loop to process buffer until no more markers can be found
                if self._in_tool_code_block:
                    # --- We are inside a tool block, looking for an end ---
                    effective_end_markers = end_markers
                    tagged_args_marker = self._tagged_args_marker()
                    is_tagged = (
                        tagged_args_marker
                        and self.profile
                        and self._current_start_marker in (self.profile.block_start or [])
                    )
                    if is_tagged:
                        effective_end_markers = []
                    if self.profile and self.profile.end_marker_outside_json_string:
                        first_end_pos, end_marker = self._find_earliest_marker_outside_string(self._stream_buffer, effective_end_markers)
                    else:
                        first_end_pos, end_marker = self._find_earliest_marker(self._stream_buffer, effective_end_markers)
                    first_hard_stop_pos, _ = self._find_earliest_marker(self._stream_buffer, hard_stop_markers)

                    found_end = False
                    is_hard_stop = False

                    if is_tagged and tagged_args_marker:
                        tagged_end = self._find_tagged_payload_end(self._stream_buffer, 0, tagged_args_marker)
                        if tagged_end is not None and (first_hard_stop_pos == -1 or tagged_end < first_hard_stop_pos):
                            block_content = self._stream_buffer[:tagged_end]
                            cleaned_block = self._strip_sequences(block_content, [self._eos_str] if self._eos_str else None)
                            normalized_block = cleaned_block if cleaned_block != block_content else None
                            self.collected_blocks.append(ToolCallBlock(
                                raw_block=block_content,
                                normalized_block=normalized_block,
                                model_format=p.key,
                                block_start_pos=self._current_block_start_pos,
                                start_marker=self._current_start_marker,
                                end_marker=None,
                                block_end_pos=(
                                    self._current_block_start_pos
                                    + len(self._current_start_marker or "")
                                    + len(block_content)
                                ),
                                payload_start_pos=self._current_block_start_pos + len(self._current_start_marker or ""),
                                payload_end_pos=(
                                    self._current_block_start_pos
                                    + len(self._current_start_marker or "")
                                    + len(block_content)
                                ),
                                position_mode="full",
                                is_incomplete=False
                            ))
                            self._absolute_offset = self.collected_blocks[-1].block_end_pos or self._absolute_offset
                            self._stream_buffer = self._stream_buffer[tagged_end:]
                            found_end = True

                    if first_end_pos != -1 and (first_hard_stop_pos == -1 or first_end_pos < first_hard_stop_pos):
                        # A regular end tag was found first.
                        block_content = self._stream_buffer[:first_end_pos]
                        cleaned_block = self._strip_sequences(block_content, [self._eos_str] if self._eos_str else None)
                        normalized_block = cleaned_block if cleaned_block != block_content else None
                        self.collected_blocks.append(ToolCallBlock(
                            raw_block=block_content,
                            normalized_block=normalized_block,
                            model_format=p.key,
                            block_start_pos=self._current_block_start_pos,
                            start_marker=self._current_start_marker,
                            end_marker=end_marker,
                            block_end_pos=(
                                self._current_block_start_pos
                                + len(self._current_start_marker or "")
                                + len(block_content)
                                + len(end_marker)
                            ),
                            payload_start_pos=self._current_block_start_pos + len(self._current_start_marker or ""),
                            payload_end_pos=(
                                self._current_block_start_pos
                                + len(self._current_start_marker or "")
                                + len(block_content)
                            ),
                            position_mode="full",
                            is_incomplete=False
                        ))
                        self._absolute_offset = self.collected_blocks[-1].block_end_pos or self._absolute_offset
                        self._stream_buffer = self._stream_buffer[first_end_pos + len(end_marker):]
                        found_end = True
                    elif first_hard_stop_pos != -1:
                        # A hard stop was found. Delegate to the finalizer logic, but only on the
                        # content *before* the hard stop.
                        self._finalize_buffer(upto_pos=first_hard_stop_pos, force_incomplete=True)
                        # The buffer is now updated by _finalize_buffer.
                        found_end = True

                    if found_end:
                        self._in_tool_code_block = False
                        continue # Re-process buffer immediately as it has been modified
                    else:
                        break # No end marker yet, wait for more tokens
                else:
                    # --- We are outside a tool block, looking for a start ---
                    start_pos, start_marker = self._find_earliest_marker(self._stream_buffer, start_markers)
                    if start_pos != -1:
                        text_to_yield = self._stream_buffer[:start_pos]
                        # --- Clean all EOS before yielding ---
                        if text_to_yield:
                            if self._eos_str:
                                original_len = len(text_to_yield)
                                text_to_yield = text_to_yield.replace(self._eos_str, "")
                                # Adjust the offset by the length of the removed EOS string.
                                self._current_text_offset -= (original_len - len(text_to_yield))
                                self._eos_id = None
                                self._eos_str = None
                                # Also reset on the streamer
                                if streamer:
                                    streamer.eos_token_detected = None
                                    streamer.eos_token_decoded = None
                            if text_to_yield:
                                self._absolute_offset += len(text_to_yield)
                                yield text_to_yield
                        self._stream_buffer = self._stream_buffer[start_pos + len(start_marker):]
                        self._current_text_offset += start_pos
                        self._current_block_start_pos = self._absolute_offset
                        self._current_start_marker = start_marker
                        self._absolute_offset += len(start_marker)
                        self._in_tool_code_block = True
                    else:
                        # Yield "safe" text, holding back enough for a potential marker
                        # The buffer length must be strictly greater than the max marker length to yield anything.
                        yield_upto = len(self._stream_buffer) - max_marker_len if len(self._stream_buffer) > max_marker_len else 0
                        if yield_upto > 0:
                            text_to_yield = self._stream_buffer[:yield_upto]
                            # --- Clean all EOS before yielding ---
                            if self._eos_str:
                                before = len(text_to_yield)
                                text_to_yield = text_to_yield.replace(self._eos_str, "")
                                self._current_text_offset -= (before - len(text_to_yield))
                                # Do NOT clear _eos_str or streamer here; we still may need it for the trailing buffer.
                            if text_to_yield:
                                self._absolute_offset += len(text_to_yield)
                                yield text_to_yield
                            self._stream_buffer = self._stream_buffer[len(text_to_yield):]
                            self._current_text_offset += yield_upto
                        break # Wait for more tokens

        # After the loop, yield any remaining plain text in the buffer
        if self._stream_buffer and not self._in_tool_code_block:
            self._cleanup_eos_in_stream_buffer()
            if self._eos_str and streamer:
                streamer.eos_token_detected = None
                streamer.eos_token_decoded = None
            if self._stream_buffer:
                self._absolute_offset += len(self._stream_buffer)
                yield self._stream_buffer
            # After yielding, clear the buffer.
            # This prevents a bug where the same leftover text could be yielded again
            # if finalize() is called on the same parser instance.
            self._stream_buffer = ""

        if self.collected_blocks:
            logger.info(f"[parser-stream] Stream finished. Collected {len(self.collected_blocks)} tool blocks.")

    def parse_collected_blocks(self, blocks: List[ToolCallBlock]) -> None:
        """
        Parses the `raw_block` of each ToolCallBlock in the list to populate its `calls` field.
        This method mutates the ToolCallBlock objects in place.
        """
        for block in blocks:
            if block.is_incomplete:
                continue
            # If calls already exist (e.g., from a prior partial parse), attempt a safer re-parse
            # when the payload looks like concatenated objects but only one call was captured.
            reparsed = False
            if block.calls and self.profile and self.profile.payload_mode in ("json_list", "json_list_with_id", "json_obj_or_list"):
                payload_source = block.normalized_block if block.normalized_block else block.raw_block
                if payload_source and isinstance(payload_source, str) and re.search(r"}\s*,\s*{", payload_source):
                    try:
                        payload = _json_loads_loose(payload_source)
                        calls = self._payload_to_calls(payload, block.raw_block)
                        if len(calls) > len(block.calls):
                            block.calls = [c for c in calls if c]
                            reparsed = True
                    except Exception:
                        pass
            if block.calls and not reparsed:
                continue
            tagged_args_marker = self._tagged_args_marker()
            if self.profile and tagged_args_marker:
                payload_source = block.normalized_block if block.normalized_block else block.raw_block
                if payload_source and tagged_args_marker in payload_source and not payload_source.lstrip().startswith(("{", "[")):
                    call_marker = (self.profile.block_start[0] if self.profile.block_start else None)
                    calls, errors = self._parse_tagged_calls(payload_source, block.raw_block, tagged_args_marker, call_marker)
                    if calls:
                        block.calls.extend(c for c in calls if c)
                        if errors:
                            block.parse_errors.extend(errors)
                        continue
                    if errors:
                        block.parse_errors.extend(errors)
            try:
                payload_source = block.normalized_block if block.normalized_block else block.raw_block
                payload = _json_loads_loose(payload_source)
                # If the parsed payload is not a list but the raw text looks like concatenated objects,
                # try to coerce it into a list to capture multiple calls.
                if not isinstance(payload, list) and isinstance(payload_source, str) and re.search(r"}\s*,\s*{", payload_source):
                    try:
                        payload = json.loads(f"[{payload_source}]")
                    except Exception:
                        collected = _json_loads_loose(payload_source)  # may return list from inner heuristic
                        if isinstance(collected, list):
                            payload = collected
                calls = self._payload_to_calls(payload, block.raw_block)
                block.calls.extend(c for c in calls if c) # Filter out None values
            except Exception:
                # The parser's job is to report errors, not to set actions.
                # The application layer will decide the action based on parse_errors.
                # If the standard loose parser fails, engage the robust salvage parser.
                self._robust_salvage_parser(block)

            if self.profile:
                # Ensure IDs where required, for all successfully parsed calls in the block
                for c in block.calls:
                    if c and self.profile.result_requires_id and not c.id:
                        c.id = self._gen_id()

    def _robust_salvage_parser(self, block: ToolCallBlock):
        """
        A robust, stateful parser that scans through a raw block of text,
        attempting to extract and repair one or more JSON tool calls.
        """
        # This logic is a simplified version of the one in mp13tools.py for demonstration
        # It tries to find JSON objects or lists and parse them.
        # A more complete implementation would handle various malformations.
        
        # For simplicity, we'll use a regex to find potential JSON objects/arrays
        # and try to parse them individually.
        scan_str = (block.normalized_block if block.normalized_block else block.raw_block).strip()
        pos = 0
        errors = []

        while pos < len(scan_str):
            match = re.search(r"[\{\[]", scan_str[pos:])
            if not match:
                remnant = scan_str[pos:].strip()
                if remnant: errors.append(f"Ignoring trailing non-JSON content: '{remnant[:100]}...'")
                break

            next_obj_start = pos + match.start()
            skipped_garbage = scan_str[pos:next_obj_start].strip()
            if skipped_garbage: errors.append(f"Skipping malformed content: '{skipped_garbage[:100]}...'")
            pos = next_obj_start

            # Find the matching end brace/bracket, respecting strings
            start_char = scan_str[pos]
            end_char = '}' if start_char == '{' else ']'
            level, in_string, esc, end_pos = 1, False, False, -1
            for i in range(pos + 1, len(scan_str)):
                char = scan_str[i]
                if in_string:
                    if esc:
                        esc = False
                    elif char == '\\':
                        esc = True
                    elif char == '"':
                        in_string = False
                else:
                    if char == '"':
                        in_string = True
                    elif char == start_char:
                        level += 1
                    elif char == end_char:
                        level -= 1
                        if level == 0:
                            end_pos = i + 1
                            break
            
            potential_json_str = ""
            if end_pos > 0: # Found a balanced segment
                potential_json_str = scan_str[pos:end_pos]
                pos = end_pos
            else: # Truncated object
                errors.append(f"Could not find matching '{end_char}' for '{start_char}' at position {pos}. Attempting to parse as truncated.")
                # Try to fix by appending the missing character
                potential_json_str = scan_str[pos:] + end_char
                pos = len(scan_str) # Consume rest of string

            try:
                data = json.loads(potential_json_str)
                # The payload can be a single call dict or a list of them
                data_list = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
                
                for item in data_list:
                    if isinstance(item, dict):
                        block.calls.extend(self._payload_to_calls(item, block.raw_block))
                    else:
                        errors.append(f"Parsed a valid JSON list, but an item was not a dictionary: {str(item)[:100]}")
            except json.JSONDecodeError as e:
                # --- NEW: Salvage a ToolCall object even from invalid JSON ---
                # Try to find the tool name with a regex, as it's often intact.
                name_match = re.search(r'"name"\s*:\s*"([^"]+)"', potential_json_str)
                salvaged_name = name_match.group(1) if name_match else "malformed_tool_call"
                
                salvaged_call = ToolCall(
                    name=salvaged_name,
                    arguments={"tool_args_issue": {"_non_parsed": potential_json_str}},
                    parse_errors=[f"Failed to parse JSON: {e}"],
                    raw=block.raw_block
                )
                block.calls.append(salvaged_call)

        if errors:
            # Consolidate all parsing errors into the main block's error field.
            block.parse_errors.extend(errors)
        elif not block.calls:
            block.parse_errors.append("robust_salvage_failure: No valid tool calls could be salvaged from the block.")

    # ---------------------------
    # Parsing model -> user text
    # ---------------------------
    def parse_model_output(
        self,
        text: str,
        prompt_index: Optional[int] = None,
        eos_strings: Optional[Iterable[str]] = None,
        mark_truncated: bool = False,
    ) -> List[ToolCallBlock]:
        p = self.profile
        blocks: List[ToolCallBlock] = []
        if not self.profile:
            return blocks

        if not p.block_start:
            return blocks

        current_pos = 0
        while current_pos < len(text):
            # Find the earliest next start marker
            first_start_pos = -1
            first_start_marker = ""
            for marker in p.block_start:
                pos = text.find(marker, current_pos)
                if pos != -1 and (first_start_pos == -1 or pos < first_start_pos): # noqa
                    first_start_pos = pos
                    first_start_marker = marker
            
            if first_start_pos == -1:
                break # No more start markers found

            block_start_pos = first_start_pos
            payload_start_pos = first_start_pos + len(first_start_marker)

            # Find the earliest end marker after the start
            first_end_pos = -1
            first_end_marker = ""
            effective_end_markers = p.block_end
            tagged_args_marker = self._tagged_args_marker()
            is_tagged = bool(tagged_args_marker and first_start_marker in (p.block_start or []))
            if is_tagged:
                effective_end_markers = []
            if effective_end_markers:
                if p.end_marker_outside_json_string:
                    first_end_pos, first_end_marker = self._find_earliest_marker_outside_string(
                        text[payload_start_pos:], effective_end_markers
                    )
                    if first_end_pos != -1:
                        first_end_pos += payload_start_pos
                else:
                    for marker in effective_end_markers:
                        pos = text.find(marker, payload_start_pos) # noqa
                        if pos != -1 and (first_end_pos == -1 or pos < first_end_pos):
                            first_end_pos = pos
                            first_end_marker = marker

            # Find the earliest hard stop pattern
            hard_stop_pos = len(text)
            hard_stop_marker = ""
            for marker in p.hard_stop_patterns:
                pos = text.find(marker, payload_start_pos)
                if pos != -1: # noqa
                    if pos < hard_stop_pos:
                        hard_stop_pos = pos
                        hard_stop_marker = marker

            # Determine the actual end of the block
            payload_end_pos = hard_stop_pos
            block_end_pos = hard_stop_pos
            used_tagged_end = False
            if first_end_pos != -1 and first_end_pos < hard_stop_pos:
                block_end_pos = first_end_pos + len(first_end_marker)

            payload_end_pos = first_end_pos if (first_end_pos != -1 and first_end_pos < hard_stop_pos) else hard_stop_pos
            if is_tagged and tagged_args_marker:
                tagged_end = self._find_tagged_payload_end(text, payload_start_pos, tagged_args_marker)
                if tagged_end is not None and tagged_end <= hard_stop_pos:
                    payload_end_pos = tagged_end
                    block_end_pos = tagged_end
                    used_tagged_end = True
            payload_text = text[payload_start_pos:payload_end_pos]
            payload_text = self._strip_sequences(payload_text, eos_strings)
            normalized_payload = self._normalize_payload_text(payload_text)
            if not normalized_payload:
                normalized_payload = payload_text.strip()

            raw_block = text[payload_start_pos:payload_end_pos]
            block = ToolCallBlock(
                calls=[],
                raw_block=raw_block,
                normalized_block=normalized_payload or raw_block,
                model_format=p.key,
                block_start_pos=block_start_pos,
                block_end_pos=block_end_pos,
                payload_start_pos=payload_start_pos,
                payload_end_pos=payload_end_pos,
                start_marker=first_start_marker,
                end_marker=first_end_marker if first_end_pos != -1 and first_end_pos < hard_stop_pos else None,
                hard_stop_marker=hard_stop_marker if (hard_stop_pos != len(text) and (first_end_pos == -1 or first_end_pos >= hard_stop_pos)) else None,
                position_mode="full",
                prompt_index=prompt_index,
            )

            parse_errors: List[str] = []
            used_hard_stop = ((first_end_pos == -1) or (first_end_pos >= hard_stop_pos)) and not used_tagged_end
            hard_stop_triggered = used_hard_stop and (hard_stop_pos != len(text))
            truncated_at_end = used_hard_stop and (hard_stop_pos == len(text))

            if hard_stop_triggered:
                block.is_incomplete = True
                parse_errors.append("Block terminated by hard stop pattern instead of closing tag.")
            elif truncated_at_end and mark_truncated:
                block.is_incomplete = True
                parse_errors.append("Stream ended before closing tag was found.")

            if parse_errors:
                block.parse_errors.extend(parse_errors)

            blocks.append(block)
            current_pos = block_end_pos

        return blocks

    def _payload_to_calls(self, data: Any, raw_block: str) -> List[ToolCall]:
        p = self.profile
        out: List[Optional[ToolCall]] = []

        def _set_nested(obj: Dict[str, Any], path: str, value: Any) -> None:
            parts = path.split(".")
            node = obj
            for key in parts[:-1]:
                next_node = node.get(key)
                if not isinstance(next_node, dict):
                    next_node = {}
                    node[key] = next_node
                node = next_node
            node[parts[-1]] = value

        def _split_args_list(args_text: str) -> Optional[List[Dict[str, Any]]]:
            candidate = args_text.strip()
            if not candidate.startswith("{") or not candidate.endswith("}"):
                return None
            if "},{" not in candidate and "}, {" not in candidate:
                return None
            try:
                parsed = json.loads(f"[{candidate}]")
            except Exception:
                return None
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                return parsed
            return None

        def make_call(obj: Dict[str, Any]) -> Optional[ToolCall]:
            parse_error = None
            needs_keep_raw = False
            name = None
            args = None
            try:
                name = _get_nested(obj, p.name_field) if "." in p.name_field else obj.get(p.name_field)
                args = _get_nested(obj, p.arguments_field) if "." in p.arguments_field else obj.get(p.arguments_field)
                # Fallback for flat structures (e.g., {"name": "...", "arguments": {...}}) when profile expects nested fields.
                if not name and "name" in obj:
                    name = obj.get("name")
                if args is None and "arguments" in obj:
                    args = obj.get("arguments")
                # Attempt to decode stringified/double-encoded arguments even when arguments_may_be_string is False.
                if isinstance(args, str):
                    decoded_args = _double_decode_arguments(args) if p.arguments_may_be_string else args
                    if isinstance(decoded_args, str):
                        try:
                            decoded_args_json = json.loads(decoded_args)
                            args = decoded_args_json
                        except Exception:
                            args = decoded_args
                    else:
                        args = decoded_args

                if p.arguments_may_be_string:
                    args = _double_decode_arguments(args)
                if not isinstance(args, dict):
                    if args is None:
                        args = {}
                    elif isinstance(args, (str, int, float, bool)):
                        # When a primitive type is found instead of a dict for arguments,
                        # this indicates a malformed response from the LLM. We wrap
                        # the raw value in '_string_value' so the tool function can
                        # attempt to salvage it.
                        args = {"_string_value": str(args)}
                        parse_error = parse_error or "arguments_not_dict"
                        needs_keep_raw = True
                    else:
                        try:
                            json.dumps(args)
                        except Exception:
                            args = {"_non_parsed": str(args)}
                            needs_keep_raw = True
            except Exception as e:
                parse_error = f"call_parse_error: {type(e).__name__}: {e}" # This will be a list now
                # best-effort salvage
                name = name or obj.get("name") or ""
                args = args if isinstance(args, dict) else {"_non_parsed": obj}
                needs_keep_raw = True

            call_id = None
            if p.id_field:
                call_id = _get_nested(obj, p.id_field) if "." in p.id_field else obj.get(p.id_field)
                if not call_id and "id" in obj:
                    call_id = obj.get("id")

            if not name:
                return None # Skip calls without a name

            return ToolCall(
                id=call_id,
                name=name,
                arguments=args,
                result=None,
                raw=raw_block,
                model_format=p.key,
                parse_errors=[parse_error] if parse_error else [],
                action=[ToolCall.KeepRaw] if needs_keep_raw else []
            )

        try:
            if p.payload_mode in ("json_obj", "json_obj_or_list"):
                if isinstance(data, dict):
                    if "." in p.arguments_field:
                        args_raw = _get_nested(data, p.arguments_field)
                    else:
                        args_raw = data.get(p.arguments_field) if p.arguments_field in data else data.get("arguments")
                    if isinstance(args_raw, str):
                        split_args = _split_args_list(args_raw)
                        if split_args:
                            if "." in p.name_field:
                                name_raw = _get_nested(data, p.name_field)
                            else:
                                name_raw = data.get(p.name_field) if p.name_field in data else data.get("name")
                            if name_raw:
                                for item in split_args:
                                    call_obj: Dict[str, Any] = {}
                                    if "." in p.name_field:
                                        _set_nested(call_obj, p.name_field, name_raw)
                                    else:
                                        call_obj[p.name_field] = name_raw
                                    if "." in p.arguments_field:
                                        _set_nested(call_obj, p.arguments_field, item)
                                    else:
                                        call_obj[p.arguments_field] = item
                                    out.append(make_call(call_obj))
                            else:
                                out.append(make_call(data))
                        else:
                            out.append(make_call(data))
                    else:
                        out.append(make_call(data))
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            out.append(make_call(item))
                        else:
                            out.append(ToolCall(
                                name="",
                                arguments={"_non_parsed": item},
                                raw=raw_block,
                                model_format=p.key,
                                parse_errors=["unexpected_list_item_type"]
                            ))
                else:
                    out.append(ToolCall(
                        name="",
                        arguments={"_non_parsed": data},
                        raw=raw_block,
                        model_format=p.key,
                        parse_errors=["unexpected_payload_type"]
                    ))
            elif p.payload_mode in ("json_list", "json_list_with_id"):
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            out.append(make_call(item))
                        else:
                            out.append(ToolCall(
                                name="",
                                arguments={"_non_parsed": item},
                                raw=raw_block,
                                model_format=p.key
                            ))
                elif isinstance(data, dict):
                    # tolerate single-object where list is expected
                    out.append(make_call(data))
                else:
                    out.append(ToolCall(
                        name="",
                        arguments={"_non_parsed": data},
                        raw=raw_block,
                        model_format=p.key,
                        parse_errors=["unexpected_payload_type"]
                    ))
            else:
                out.append(ToolCall(
                    name="",
                    arguments={"_non_parsed": data},
                    raw=raw_block,
                    model_format=p.key
                ))
        except Exception as e:
            out.append(ToolCall(
                name="",
                arguments={"_non_parsed": data},
                raw=raw_block,
                model_format=p.key,
                parse_errors=[f"payload_to_calls_error: {type(e).__name__}: {e}"]
            ))
        return [call for call in out if call is not None]

    def _find_earliest_marker(self, text: str, markers: List[str]) -> Tuple[int, str]:
        """Finds the earliest occurrence of any marker in the text."""
        first_pos = -1
        found_marker = ""
        for marker in markers:
            pos = text.find(marker)
            if pos != -1 and (first_pos == -1 or pos < first_pos):
                first_pos = pos
                found_marker = marker
        return first_pos, found_marker

    def _find_earliest_marker_outside_string(self, text: str, markers: List[str]) -> Tuple[int, str]:
        """
        Finds the earliest occurrence of any marker that is not inside a JSON string.
        This is used to avoid premature closes when markers appear inside quoted payloads.
        """
        if not text or not markers:
            return -1, ""
        earliest_pos = -1
        earliest_marker = ""
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
            for marker in markers:
                if text.startswith(marker, i):
                    if earliest_pos == -1 or i < earliest_pos:
                        earliest_pos = i
                        earliest_marker = marker
            if earliest_pos == i:
                # We found the earliest possible match at this index.
                break
        return earliest_pos, earliest_marker


    def _gen_id(self) -> str:
        if self.profile and getattr(self.profile, "enforce_tool_call_id_format", False):
            return uuid.uuid4().hex[:9]
        return f"call_{uuid.uuid4().hex[:8]}"

    # -------------------------------------------------
    # Serializing Tool Calls (for results or normalization)
    # -------------------------------------------------
    def serialize_calls(self, calls: List[ToolCall], is_result: bool = False, block_action: Optional[List[str]] = None) -> Union[str, List[Dict[str, Any]]]:
        """
        Serializes a list of ToolCall objects based on the profile.
        - If is_result=True, serializes the tool results for the model.
        - If is_result=False, serializes the original tool calls for normalization.
        """
        p = self.profile
        results_as_user_role = bool(getattr(p, "results_as_user_role", False))
        pack_results_as_one_role = bool(getattr(p, "pack_results_as_one_role", False))
        action_override = [a for a in (block_action or []) if a]

        # Helpers to tolerate ToolCall objects or plain dicts
        def _get(obj, name, default=None):
            return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

        def _get_list(obj, name):
            v = _get(obj, name, [])
            if v is None:
                return []
            return v if isinstance(v, list) else [v]

        def _get_parse_errors(obj):
            errs = _get(obj, "parse_errors", [])
            if errs is None:
                return []
            return errs if isinstance(errs, list) else [errs]

        def _effective_actions(obj):
            call_actions = _get_list(obj, "action")
            return action_override or call_actions

        # Action priority (highest first): block actions override per-call actions,
        # and the first present item in this list wins.
        _ACTION_PRIORITY = [
            ToolCall.Strip,
            ToolCall.Ignore,
            ToolCall.Retry,
            ToolCall.KeepRaw,
        ]

        def _primary_action(actions: List[str]) -> Optional[str]:
            for a in _ACTION_PRIORITY:
                if a in actions:
                    return a
            return None

        def _select_content_for_result(call: Any, primary_action: Optional[str]) -> Any:
            """Pick payload honoring action priority."""
            c_error = _get(call, "error", None)
            c_parse_errors = _get_parse_errors(call)
            c_result = _get(call, "result", None)

            if primary_action == ToolCall.Retry:
                return c_error or (", ".join(c_parse_errors) if c_parse_errors else None) or c_result or {}

            if primary_action == ToolCall.KeepRaw:
                raw_val = _get(call, "raw", None)
                if raw_val is not None:
                    return raw_val

            call_actions = _effective_actions(call)
            if not call_actions:
                return c_error or (", ".join(c_parse_errors) if c_parse_errors else None) or c_result or {}

            return c_result if c_result is not None else (c_error or (", ".join(c_parse_errors) if c_parse_errors else {}))

        def _ensure_id(obj):
            cid = _get(obj, "id")
            if cid and isinstance(cid, str) and cid.isalnum() and len(cid) == 9:
                return cid
            return uuid.uuid4().hex[:9]

        tagged_args_marker = self._tagged_args_marker()
        if not is_result and tagged_args_marker:
            primary_block_action = _primary_action(action_override)
            if primary_block_action == ToolCall.Strip:
                return ""
            out_blocks: List[str] = []
            call_marker = p.block_start[0] if p.block_start else ""
            for c in calls:
                call_actions = _get_list(c, "action")
                if ToolCall.KeepRaw in call_actions:
                    raw_payload = _get(c, "raw", "")
                    if raw_payload:
                        if call_marker and raw_payload.strip().startswith(call_marker):
                            return raw_payload
                        return f"{call_marker}{raw_payload}" if call_marker else raw_payload
                    continue
                if ToolCall.Strip in call_actions:
                    continue
                name = _get(c, "name", "")
                if not name:
                    continue
                raw_args = _get(c, "arguments", {})
                cleaned_args = _strip_internal_tool_args(raw_args)
                out_blocks.append(f"{call_marker}{name}{tagged_args_marker}{json.dumps(cleaned_args, ensure_ascii=False)}")
            return "".join(out_blocks)

        out_blocks: List[str] = []
        if not calls:
            return ""

        if is_result:
            # --- Serialize RESULTS to send back to the model ---
            # Controlled by profile flags and effective tool call actions.
            calls_to_serialize = []
            for c in calls:
                eff_actions = _effective_actions(c)
                primary_action = _primary_action(eff_actions)
                if primary_action in {ToolCall.Ignore, ToolCall.Strip}:
                    continue
                # Normalize IDs for strict templates (e.g., Mistral)
                if self.profile and getattr(self.profile, "enforce_tool_call_id_format", False):
                    if isinstance(c, dict):
                        c["id"] = _ensure_id(c)
                    else:
                        c.id = _ensure_id(c)
                if isinstance(c, dict):
                    new_call = dict(c)
                    new_call["action"] = eff_actions
                    calls_to_serialize.append(new_call)
                else:
                    new_call = ToolCall(
                        name=_get(c, "name", ""),
                        arguments=_get(c, "arguments", {}),
                        id=_get(c, "id", None),
                        result=_get(c, "result", None),
                        error=_get(c, "error", None),
                        raw=_get(c, "raw", None),
                        model_format=_get(c, "model_format", None),
                        parse_errors=_get_parse_errors(c),
                        action=eff_actions,
                    )
                    calls_to_serialize.append(new_call)

            if not calls_to_serialize:
                return ""

            # If the profile requests per-call message payloads, return a list of dicts.
            if getattr(p, "results_as_messages", False):
                result_msgs = []
                content_key = getattr(p, "results_message_content_key", None) or getattr(p, "result_payload_key", None) or "content"
                for c in calls_to_serialize:
                    primary_action = _primary_action(_effective_actions(c))
                    content = _select_content_for_result(c, primary_action)
                    if isinstance(content, str):
                        payload_obj = {content_key: content}
                    else:
                        try:
                            payload_obj = {content_key: json.dumps(content, ensure_ascii=False)}
                        except Exception:
                            payload_obj = {content_key: json.dumps(str(content), ensure_ascii=False)}
                    if getattr(p, "result_requires_id", False) and getattr(p, "result_id_key", None):
                        payload_obj[p.result_id_key] = _ensure_id(c)
                    result_msgs.append(payload_obj)
                return result_msgs

            if pack_results_as_one_role:
                # Serialize all results into a single list payload within one block.
                if results_as_user_role:
                    # Each result is individually wrapped and then joined.
                    individual_results = []
                    for c in calls_to_serialize:
                        primary_action = _primary_action(_effective_actions(c))
                        content = _select_content_for_result(c, primary_action)
                        payload_str = json.dumps(content, ensure_ascii=False)
                        start_wrapper = p.result_wrapper_start or ""
                        end_wrapper = p.result_wrapper_end or ""
                        if payload_str and payload_str != '{}' and payload_str != '[]':
                            individual_results.append(f"{start_wrapper}{payload_str}{end_wrapper}")
                    return "\n".join(individual_results)
                else:
                    # Results are serialized as a JSON array with no wrappers.
                    results_list = []
                    for c in calls_to_serialize:
                        primary_action = _primary_action(_effective_actions(c))
                        content = _select_content_for_result(c, primary_action)
                        results_list.append(content)
                    return json.dumps(results_list, ensure_ascii=False)
            else:
                # Default behavior: one block per result.
                results_payloads: List[Any] = []
                for c in calls_to_serialize:
                    primary_action = _primary_action(_effective_actions(c))
                    content = _select_content_for_result(c, primary_action)

                    payload_obj = {}
                    payload_str = ""
                    if results_as_user_role:
                        payload_str = json.dumps(content, ensure_ascii=False)
                        # Only wrap if converting to user role.
                        if payload_str and payload_str != '{}' and payload_str != '[]':
                            start_wrapper = p.result_wrapper_start or ""
                            end_wrapper = p.result_wrapper_end or ""
                            out_blocks.append(f"{start_wrapper}{payload_str}{end_wrapper}")
                    else:  # Original tool role logic
                        # If a payload key is specified (like Mistral's 'content'), wrap the result.
                        if p.result_payload_key:
                            payload_obj[p.result_payload_key] = content
                        else:
                            # Otherwise, the content itself is the payload (if it's a dict)
                            payload_obj = content if isinstance(content, (dict, list, str, int, float, bool)) else {"result": str(content)}
                            if not isinstance(payload_obj, dict):
                                payload_obj = {"result": payload_obj}

                        # Add the call ID if required by the profile.
                        if p.result_requires_id and p.result_id_key:
                            payload_obj[p.result_id_key] = _get(c, "id") or self._gen_id()

                        results_payloads.append(payload_obj)
                        payload_str = json.dumps(payload_obj, ensure_ascii=False)
                        if payload_str and payload_str != '{}' and payload_str != '[]':
                            # For the 'tool' role, we do not use start/end wrappers.
                            out_blocks.append(payload_str)

                if not out_blocks:
                    return "" # If all results were ignored or empty, return nothing.
                if getattr(p, "result_as_single_block_list", False):
                    # Emit a single JSON array of result objects when requested by the profile.
                    return json.dumps(results_payloads, ensure_ascii=False)
                return "\n".join(out_blocks)
        else:
            # --- Serialize original CALLS for normalization ---
            primary_block_action = _primary_action(action_override)
            if primary_block_action == ToolCall.Strip:
                return "" # Strip the entire block

            serialized_call_dicts = []
            for c in calls: # Use original list to respect Strip action during reconstruction
                call_actions = _get_list(c, "action")
                if ToolCall.KeepRaw in call_actions:
                    raw_payload = _get(c, "raw", "")
                    if raw_payload:
                        start_marker = p.block_start[0] if p.block_start else ""
                        end_marker = p.block_end[0] if p.block_end else ""
                        return f"{start_marker}{raw_payload}{end_marker}"
                    continue
                call_dict = {}
                # Use the robust _set_nested helper to handle both flat and nested fields.
                if p.id_field:
                    _set_nested(call_dict, p.id_field, _get(c, "id") or self._gen_id())
                _set_nested(call_dict, p.name_field, _get(c, "name", ""))
                
                # --- FIX: Do not serialize internal recovery keys back to the model ---
                raw_args = _get(c, "arguments", {})
                cleaned_args = _strip_internal_tool_args(raw_args)
                _set_nested(call_dict, p.arguments_field, cleaned_args)

                serialized_call_dicts.append(call_dict)

            # Determine if the payload should be a list or a single object
            is_list_payload = p.payload_mode in ["json_list", "json_list_with_id"] or (p.payload_mode == "json_obj_or_list" and len(serialized_call_dicts) > 1)
            
            payload_str = ""
            if is_list_payload:
                payload_str = json.dumps(serialized_call_dicts, indent=2)
            else: # Single object
                if not serialized_call_dicts: return ""
                payload_str = json.dumps(serialized_call_dicts[0], indent=2)

            # Wrap the payload string with block markers
            start_marker = p.block_start[0] if p.block_start else ""
            end_marker = p.block_end[0] if p.block_end else ""
            return f"{start_marker}{payload_str}{end_marker}"

    def reconstruct_prompt_with_tools(self, original_message: Dict[str, Any], tool_blocks: List[ToolCallBlock]) -> Tuple[Dict[str, str], Optional[Dict[str, str]]]:
        """
        Reconstructs a message history by re-inserting normalized tool calls and creating a new message for tool results.
        This is used to clean up and structure conversation history before sending it to the model.
        """
        if not tool_blocks:
            return original_message, None

        results_as_user_role = bool(getattr(self.profile, "results_as_user_role", False))

        # Keep every block unless it explicitly requests stripping and thus shouldn't be reinserted.
        valid_tool_blocks = [block for block in tool_blocks if ToolCall.Strip in block.action_block or block.calls or block.raw_block]
        
        # The 'content' of the original assistant message is assumed to be the text part,
        # with tool blocks already stripped out.
        reconstructed_content = original_message.get("content") or ""

        # Sort blocks by their intended insertion point, in reverse order
        for block in sorted(valid_tool_blocks, key=lambda b:  b.block_start_pos if b.block_start_pos is not None else -1, reverse=True):
            block_actions = list(block.action_block or [])
            has_strip = ToolCall.Strip in block_actions

            def _first_error_text() -> Optional[str]:
                if block.error_block:
                    return str(block.error_block)
                for c in block.calls:
                    if getattr(c, "error", None):
                        return str(getattr(c, "error"))
                    errs = getattr(c, "parse_errors", None) or []
                    if errs:
                        return ", ".join([str(e) for e in errs if e])
                return None

            if has_strip:
                err_text = _first_error_text()
                if err_text:
                    insert_at = block.block_start_pos if (block.block_start_pos is not None and block.block_start_pos >= 0) else len(reconstructed_content)
                    reconstructed_content = reconstructed_content[:insert_at] + err_text + reconstructed_content[insert_at:]
                continue

            # Check if every call is effectively stripped.
            all_calls_stripped = True
            for c in block.calls:
                actions = block_actions or getattr(c, "action", []) or []
                if ToolCall.Strip not in actions:
                    all_calls_stripped = False
                    break
            if all_calls_stripped:
                err_text = _first_error_text()
                if err_text:
                    insert_at = block.block_start_pos if (block.block_start_pos is not None and block.block_start_pos >= 0) else len(reconstructed_content)
                    reconstructed_content = reconstructed_content[:insert_at] + err_text + reconstructed_content[insert_at:]
                continue
            
            if block.block_start_pos == -1 and ToolCall.KeepRaw not in block_actions:
                # If position is unknown and we are not keeping it raw, we can't reconstruct.
                continue

            start_marker = self.profile.block_start[0] if self.profile.block_start else ""
            end_marker = self.profile.block_end[0] if self.profile.block_end else ""

            def _ensure_wrapped(payload: str) -> str:
                if start_marker and not payload.startswith(start_marker):
                    payload = f"{start_marker}{payload}"
                if end_marker and not payload.endswith(end_marker):
                    payload = f"{payload}{end_marker}"
                return payload

            # Serialize the block back to its "model-generated" format.
            if block.calls:
                normalized_block_str = self.serialize_calls(block.calls, is_result=False, block_action=block_actions)
            elif block.raw_block:
                # Fallback: no parsed calls but we still have the original raw payload.
                normalized_block_str = _ensure_wrapped(block.normalized_block or block.raw_block)
            else:
                continue

            # Insert the normalized block into the text content
            reconstructed_content = reconstructed_content[:block.block_start_pos] + normalized_block_str + reconstructed_content[block.block_start_pos:]

        # The original message now contains the full, normalized text + tool calls
        final_assistant_message = {
            "role": "assistant",
            "content": reconstructed_content
        }

        # Create a new, separate message for the tool results
        all_calls = [call for block in valid_tool_blocks for call in block.calls]
        if not all_calls:
            return final_assistant_message, None

        # Serialize all results into a single string, respecting all new flags
        results_str = ToolsParserHelper.serialize_blocks(
            profile=self.profile,
            blocks=valid_tool_blocks,
            is_result=True
        )

        if not results_str: # If all results were ignored, don't create a tool message
            return final_assistant_message, None

        if isinstance(results_str, list) and (not results_str or isinstance(results_str[0], str)):
            results_str = "\n".join(str(item) for item in results_str if str(item) != "")

        tool_results_message = {
            "role": "user" if results_as_user_role else "tool",
            "content": results_str
        }
        
        # Optionally attach tool_calls to the assistant stub for templates that expect it.
        emit_tool_calls = bool(getattr(self.profile, "emit_tool_calls_field", False))
        if emit_tool_calls and all_calls:
            def _ensure_id(call: ToolCall) -> str:
                cid = getattr(call, "id", None)
                if getattr(self.profile, "enforce_tool_call_id_format", False):
                    if isinstance(cid, str) and len(cid) == 9 and cid.isalnum():
                        return cid
                    return uuid.uuid4().hex[:9]
                return cid or uuid.uuid4().hex[:9]

            tool_calls_payload = []
            for call in all_calls:
                call_id = _ensure_id(call)
                args = _strip_internal_tool_args(getattr(call, "arguments", {}))
                tool_calls_payload.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": getattr(call, "name", ""),
                        "arguments": args
                    }
                })
            if tool_calls_payload:
                final_assistant_message["tool_calls"] = tool_calls_payload

        return final_assistant_message, tool_results_message


class ToolsParserHelper:
    @staticmethod
    def reconstruct_text_without_block(text_content: str, tool_blocks: List["ToolCallBlock"]) -> str:
        """
        Removes multiple tool blocks (markers + payload) from a given text string.
        It processes blocks in reverse order of their start position to avoid index shifting issues.
        """
        if not tool_blocks:
            return text_content

        # Sort blocks by start position in reverse order to avoid index shifting issues
        sorted_blocks = sorted(
            [b for b in tool_blocks if b.block_start_pos >= 0 and b.raw_block],
            key=lambda b: b.block_start_pos,
            reverse=True
        )

        def _strip_with_spans(text_value: str, blocks: List["ToolCallBlock"]) -> str:
            for block in blocks:
                if block.position_mode != "full" or block.block_end_pos is None:
                    continue
                start_pos = block.block_start_pos
                if start_pos < 0:
                    continue
                block_end_pos = block.block_end_pos
                text_value = text_value[:start_pos] + text_value[block_end_pos:]
            return text_value

        text_content = _strip_with_spans(text_content, sorted_blocks)

        return text_content

    @staticmethod
    def reconstruct_text_with_blocks(
        text_content: str | None,
        tool_blocks: List["ToolCallBlock"],
        profile: Optional["ParserProfile"] = None,
    ) -> str:  # type: ignore
        """
        Reinserts tool call blocks back into the text content based on their block_start_pos.
        This is a fallback when a profile-aware reconstruction path is unavailable.
        Prefer UnifiedToolIO.reconstruct_prompt_with_tools when a parser profile is available.
        """
        if text_content is None:
            text_content = ""
        if not tool_blocks:
            return text_content
        # Sort tool blocks by block_start_pos in reverse order.
        sorted_blocks = sorted(
            tool_blocks,
            key=lambda block: block.block_start_pos if block.block_start_pos is not None else -1,
            reverse=True
        )

        for block in sorted_blocks:  # type: ignore
            start_pos = block.block_start_pos
            if ToolCall.Strip in block.action_block:
                continue
            if start_pos is None or start_pos < 0:
                continue
            else:
                insert_content = block.raw_block or block.normalized_block or ""
                effective_profile = profile or DEFAULT_PROFILE
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
                # Insert the block into the text at its original position.
                # The original implementation was flawed for start_pos=0.
                # This corrected logic works for all positions.
                text_content = text_content[:start_pos] + insert_content + text_content[start_pos:]

        return text_content

    @staticmethod
    def serialize_blocks(profile: "ParserProfile", blocks: List["ToolCallBlock"], is_result: bool = False) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Serializes a list of ToolCallBlock objects, either for normalization (is_result=False)
        or for sending results back to the model (is_result=True).
        Delegates the profile-specific logic to UnifiedToolIO.
        """
        if not blocks:
            return []
        parser = UnifiedToolIO(profile=profile)

        if is_result:
            effective_calls: List[ToolCall] = []
            for block in blocks:
                block_actions = list(block.action_block or [])
                # Include block-level error as a synthetic call.
                if block.error_block:
                    effective_calls.append(
                        ToolCall(
                            name="",
                            arguments={},
                            id=None,
                            result=None,
                            error=block.error_block,
                            raw=block.raw_block,
                            model_format=block.model_format,
                            parse_errors=list(block.parse_errors or []),
                            action=block_actions or [ToolCall.KeepRaw],
                        )
                    )
                for call in block.calls:
                    actions = block_actions or getattr(call, "action", []) or []
                    if isinstance(call, ToolCall):
                        cloned = ToolCall(
                            name=call.name,
                            arguments=copy.deepcopy(call.arguments),
                            id=call.id,
                            result=call.result,
                            error=call.error,
                            raw=call.raw or block.raw_block,
                            model_format=call.model_format or block.model_format,
                            parse_errors=list(call.parse_errors),
                            action=list(actions),
                        )
                    else:
                        cloned = copy.deepcopy(call)
                        cloned["action"] = list(actions)
                        cloned.setdefault("raw", block.raw_block)
                        if block.model_format is not None:
                            cloned.setdefault("model_format", block.model_format)
                    effective_calls.append(cloned)

            result_payload = parser.serialize_calls(effective_calls, is_result=True, block_action=None)
            if isinstance(result_payload, list):
                return result_payload
            if result_payload == "":
                return []
            return [result_payload]
        else:
            # For normalization, we serialize each block individually and return the list.
            serialized_blocks: List[str] = []
            for block in blocks:
                serialized_block = parser.serialize_calls(block.calls, is_result=False, block_action=block.action_block)
                if serialized_block:
                    serialized_blocks.append(serialized_block)
            return serialized_blocks


# ============================================================
# Convenience: pick profile by explicit key
# ============================================================

def profile_for(key: str) -> ParserProfile:
    return _clone_profile(key)
