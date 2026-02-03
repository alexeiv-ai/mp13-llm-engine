# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""MP13 Inference Logic"""

import os, asyncio, time, traceback, contextlib
import json, codecs
import threading
import functools
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Union, AsyncIterator, List, Dict, Any, Optional, Callable,  Tuple, Iterator, Sequence, Iterable

import contextlib
from contextlib import nullcontext

import torch
from transformers import GenerationConfig
from transformers.generation.streamers import TextIteratorStreamer
from transformers.generation.stopping_criteria import StoppingCriteriaList 
from transformers.cache_utils import StaticCache
from contextlib import ExitStack
from transformers.tokenization_utils_base import BatchEncoding

from peft import PeftModel 
from peft import PeftMixedModel
from .mp13_state import MP13State, InferenceMetricsHistoryItem
from .mp13_state import EngineModeState, InferenceStatus, ServerStatus, ModeMismatchError, EngineError, AdapterError, InferenceRequestError, RequestResource
from .mp13_config import InferenceRequest, InferenceResponse, ChunkType, ColumnsConfig, Tool, ToolCallBlock, ParserProfile
from .mp13_tools_parser import UnifiedToolIO, ToolsParserHelper
from .mp13_state import MP13State
from .mp13_cache import maybe_apply_static_cache, queue_static_cache_warmup, get_effective_cache_mode
from .mp13_utils import format_prompt_messages, CancellableStoppingCriteria, StoringTextIteratorStreamer
from .mp13_utils import first_module_device, get_modified_generation_config, round_floats
from .mp13_utils import no_static_cuda_launcher

if TYPE_CHECKING:
    from .mp13_engine import MP13Engine

# Sentinel object to detect end of stream from executor
_SENTINEL = object()

class InferenceCancelledError(Exception):
    """Custom exception for cancelled inference."""
    pass

def _get_stop_ids_for_output(
    tokenizer: Any,
    extra_eos_ids: Optional[Union[int, Iterable[int]]] = None,
    include_eos: bool = True,
) -> set:
    stop_ids = set()
    if include_eos:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            stop_ids.add(eos_token_id)
        elif isinstance(eos_token_id, list):
            stop_ids.update(eos_token_id)

        if extra_eos_ids is not None:
            if isinstance(extra_eos_ids, int):
                stop_ids.add(extra_eos_ids)
            else:
                for tid in extra_eos_ids:
                    if isinstance(tid, int):
                        stop_ids.add(tid)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_id, int):
        stop_ids.add(pad_id)
    return stop_ids

def _get_eos_strings(tokenizer: Any, extra_eos_ids: Optional[Union[int, Iterable[int]]] = None) -> List[str]:
    eos_ids = _get_stop_ids_for_output(tokenizer, extra_eos_ids, include_eos=True)
    eos_ids.discard(getattr(tokenizer, "pad_token_id", None))
    eos_strings: List[str] = []
    for tid in eos_ids:
        if isinstance(tid, int):
            eos_strings.append(
                tokenizer.decode(
                    [tid],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
    return eos_strings

def _should_mark_truncated_for_hard_stop(
    blocks: Optional[Iterable[Any]],
    *,
    output_tokens: Optional[int],
    max_new_tokens: Optional[int],
    threshold: int = 5,
) -> bool:
    if output_tokens is None or max_new_tokens is None:
        return False
    if output_tokens < max_new_tokens - threshold:
        return False
    if not blocks:
        return False
    for block in blocks:
        if isinstance(block, dict):
            hard_stop_marker = bool(block.get("hard_stop_marker"))
        else:
            hard_stop_marker = bool(getattr(block, "hard_stop_marker", None))
        if hard_stop_marker:
            return True
    return False

def _strip_eos_strings_from_end(text: str, eos_strings: Iterable[str]) -> str:
    if not text or not eos_strings:
        return text
    out = text
    changed = True
    while changed:
        changed = False
        for s in eos_strings:
            if s and out.endswith(s):
                out = out[:-len(s)]
                changed = True
                break
    return out


def _format_adapter_label(adapters: Optional[Union[str, Sequence[str]]]) -> str:
    """
    Formats adapter identifiers for prompt-level telemetry.
    Returns '__base__' when no adapters are active.
    """
    if adapters is None:
        return "__base__"
    if isinstance(adapters, str):
        return adapters or "__base__"
    if not adapters:
        return "__base__"
    return ", ".join(adapters)

def _ensure_cudagraph_tls():
    """
    Ensure cudagraph_trees TLS is initialized for non-autograd threads.
    This avoids torch._C._is_key_in_tls assertions when generate runs in executor threads.
    """
    try:
        import torch._inductor.cudagraph_trees as cgt
        if not hasattr(cgt.local, "tree_manager_containers"):
            cgt.local.tree_manager_containers = {}
        if not hasattr(cgt.local, "tree_manager_locks"):
            cgt.local.tree_manager_locks = defaultdict(threading.Lock)
    except Exception:
        # Best-effort only; if unavailable, let upstream handle it.
        pass

# --------------------------------------------------------------
# Implementation for PeftMixedModel.generate patch.
# The actual patching is done in mp13_patches.py to centralize monkey-patching.
# --------------------------------------------------------------
_ORIG_GENERATE = PeftMixedModel.generate

def _mixed_generate_like_peft(self, *args, **kwargs):
    """
    An implementation for PeftMixedModel.generate that adds two layers of gating:
    1. First User Generate Gate: For concurrent mode, serializes the very first request
       to ensure one-time compilations complete without race conditions.
    2. Static Cache Gate: For static cache mode, blocks user requests if the initial
       cache warmup is actively running in the background.
    It also handles adapter fan-out for batch inference with multiple adapters.
    """
    if not isinstance(self, PeftMixedModel):
        raise RuntimeError(
            f"_mixed_generate_like_peft expects a PeftMixedModel; got {type(self).__name__}."
        )
    if _ORIG_GENERATE is _mixed_generate_like_peft:
        raise RuntimeError("Original PeftMixedModel.generate is not available (recursive patch detected).")
    # --- Pop engine-specific kwargs to avoid passing them to the original generate ---
    engine_state: MP13State = kwargs.pop("mp13_engine_state", None)
    request: Optional["InferenceRequest"] = kwargs.pop("mp13_request", None)
    req_stream: Optional[torch.cuda.Stream] = kwargs.pop("mp13_cuda_stream", None)
    _is_warmup_call: bool = kwargs.pop("_is_warmup_call", False)
    # for both user and cache warm up threads
    _disable_compile: bool = engine_state.is_warming_cache or _is_warmup_call 

    # --- Gating Logic ---
    acquired_first_user_lock = False
    # These gates protect initial compilation and cache warming from race conditions.
    # They should only apply to user-facing requests, not internal warmup calls.
    if engine_state is not None and not _is_warmup_call:
        # GATE 1: First User Generate (for concurrent_generate > 1)
        # This serializes the very first user request after a mode switch to INFERENCE.
        cg = engine_state.concurrent_generate
        if cg > 1 and not engine_state._first_user_generate_done.is_set():
            # The first thread acquires the lock and proceeds. Others wait for the event.
            if engine_state._first_user_generate_lock.acquire(blocking=False):
                acquired_first_user_lock = True
                engine_state.logger.debug(f"First concurrent user request (req={request.request_id if request else 'N/A'}) acquired lock, proceeding.")
            else:
                engine_state.logger.debug(f"Gating concurrent request (req={request.request_id if request else 'N/A'}) until first user generate completes...")
                engine_state._first_user_generate_done.wait()
                engine_state.logger.debug(f"Concurrent request (req={request.request_id if request else 'N/A'}) un-gated. Proceeding.")

        # GATE 2: Static Cache Warmup
        # This blocks user requests if the *very first* static cache warmup is actively running.
        static_opt_in = bool((engine_state.current_inference_session_config or {}).get("static_kv_cache", False))

    try:
        ad_names = kwargs.pop("adapter_names", None)
        # --------------------------- FAST PATH (no adapters) -------------------
        if ad_names is None:
            effective_mode = get_effective_cache_mode(kwargs)
            final_max_new_tokens_for_log = getattr(kwargs.get("generation_config"), "max_new_tokens", "N/A")
            engine_state.logger.debug(f"KV mode: {effective_mode}, PKV: {type(kwargs.get('past_key_values'))}, MaxNew: {final_max_new_tokens_for_log}")

            cg = engine_state.concurrent_generate if engine_state else 1

            # Resolve stream usage
            stream_ctx = nullcontext()
            if req_stream and torch.cuda.is_available():
                stream_ctx = torch.cuda.stream(req_stream)

            with stream_ctx:
                # Eager fallback during warm-up to avoid FX<->Dynamo contention
                if _disable_compile:
                    import torch._dynamo as dynamo
                    # Use decorator form to support builds where context-manager is not allowed
                    @dynamo.disable
                    def _call():
                        # if using a static cache slot, wait on its ready event on the request stream
                        pkv = kwargs.get("past_key_values", None)
                        if torch.cuda.is_available() and isinstance(pkv, StaticCache):
                            evt = getattr(pkv, "_ready_event", None)
                            if evt is not None and not evt.query():
                                (req_stream or torch.cuda.current_stream()).wait_event(evt)
                        return _ORIG_GENERATE(self, *args, **kwargs)
                    out = _call()
                else:
                    _ensure_cudagraph_tls()
                    launcher_ctx = no_static_cuda_launcher() if (req_stream and os.name == "nt") else nullcontext()
                    with launcher_ctx, torch.inference_mode():
                        # if using a static cache slot, wait on its ready event on the request stream
                        pkv = kwargs.get("past_key_values", None)  # or sub_kwargs in the sub-path
                        if torch.cuda.is_available() and isinstance(pkv, StaticCache):
                            evt = getattr(pkv, "_ready_event", None)
                            if evt is not None and not evt.query():
                                (req_stream or torch.cuda.current_stream()).wait_event(evt)
                        out = _ORIG_GENERATE(self, *args, **kwargs) 
                       
            return out

        # ------------------------- ADAPTER FAN-OUT PATH ------------------------
        inp = kwargs.get("input_ids", args[0] if args else None)
        if inp is None:
            raise ValueError("input_ids required with adapter_names")
        B = inp.size(0)

        # normalise adapter list --------------------------------------------------
        if isinstance(ad_names, str):
            ad_names = [ad_names]
        if len(ad_names) == 1:
            ad_names *= B
        if len(ad_names) != B:
            raise ValueError("len(adapter_names) must equal batch size")

        # The entire sequence of saving state, switching adapters for micro-batches,
        # and restoring state must be atomic to prevent race conditions.
        lock = getattr(self, "_adapter_mutation_lock", None)
        if not lock: # This lock is injected by mp13_patches.py
            raise RuntimeError(f"_adapter_mutation_lock not found on {type(self).__name__}. This is required for safe concurrent adapter switching.")

        with lock:
            # bucket rows that share the same adapter (micro-batch) -------------------
            buckets = {}
            for row, name in enumerate(ad_names):
                buckets.setdefault(name, []).append(row)

            # Directly queries the `active_adapter` attribute from the model (`self`).
            prev = getattr(self, "active_adapter", None) 
            outs = [None] * B

            for name, rows in buckets.items():
                if name in (None, "__base__"):
                    self.disable_adapter_layers()
                else:
                    self.set_adapter(name)
                    self.enable_adapter_layers()

                idx = torch.as_tensor(rows, device=inp.device)
                def _slice_one(t):
                    # only slice tensors that are truly batch-aligned
                    if isinstance(t, torch.Tensor) and t.size(0) == B:
                        return t.index_select(0, idx) if isinstance(t, torch.Tensor) and t.size(0) == B else t
                    return t

                sub_kwargs = {k: _slice_one(v) for k, v in kwargs.items()}
                sub_kwargs["input_ids"] = _slice_one(inp)

                # -- forward the cancel criterion --
                if "stopping_criteria" in kwargs:
                    sub_kwargs["stopping_criteria"] = kwargs["stopping_criteria"]

                # --- Static Cache Routing for Micro-batch ---
                sub_cache_mode = "dynamic" # Default if not routed
                if engine_state and request:
                    sub_batch_size = len(rows)
                    #sub_prompt_len = sub_kwargs["input_ids"].shape[1] # max total length of any request in the batch
                    #sub_kwargs["attention_mask"].shape[1]  # max total length of any request in the batch
                    sub_prompt_len = sub_kwargs["attention_mask"].sum(-1).max() # max effective length i.e. minus padding

                    # Get max_new_tokens from the original generation_config
                    original_gc = kwargs.get("generation_config", self.generation_config)
                    sub_max_new_tokens = getattr(original_gc, "max_new_tokens", 1024)

                    active_adapters_for_sub_batch = [] if name in (None, "__base__") else [name]

                    # This is a dummy accumulator. We don't want to schedule new warmups from inside a fan-out.
                    dummy_deferred_slots = []

                    sub_cache_mode, sub_cache_bucket, sub_eff_new_tokens = maybe_apply_static_cache(
                        engine_state=engine_state,
                        model=self, # The [compiled] PeftMixedModel instance
                        gen_kwargs=sub_kwargs, # Mutated in-place
                        batch_size=sub_batch_size,
                        prompt_len=sub_prompt_len,
                        max_new_tokens=sub_max_new_tokens,
                        request=request,
                        active_adapters_for_request=active_adapters_for_sub_batch,
                        deferred_slots_accumulator=dummy_deferred_slots,
                        is_micro_batch=True # This is the key flag
                    )

                    # Update the generation config for this sub-batch if the cache logic changed max_new_tokens
                    if sub_cache_mode == "static" and sub_eff_new_tokens is not None and sub_eff_new_tokens != sub_max_new_tokens:
                        sub_kwargs["generation_config"] = get_modified_generation_config(original_gc, max_new_tokens=sub_eff_new_tokens)
                # --- End Static Cache Routing ---
                # Per-micro-batch concurrency/cache/stream policy
                cg = engine_state.concurrent_generate if engine_state else 1

                stream_ctx = nullcontext()
                if req_stream and torch.cuda.is_available():
                    stream_ctx = torch.cuda.stream(req_stream)

                effective_mode_for_log = get_effective_cache_mode(sub_kwargs)
                final_max_new_tokens_for_log = getattr(sub_kwargs.get("generation_config"), "max_new_tokens", "N/A")
                engine_state.logger.debug(f"KV mode: {effective_mode_for_log}, PKV: {type(sub_kwargs.get('past_key_values'))}, MaxNew: {final_max_new_tokens_for_log}")
                with stream_ctx:
                    # Eager fallback during warm-up to avoid FX<->Dynamo contention
                    if _disable_compile:
                        import torch._dynamo as dynamo
                        # Use decorator form to support builds where context-manager is not allowed
                        @dynamo.disable
                        def _call():
                            pkv = sub_kwargs.get("past_key_values", None)
                            if torch.cuda.is_available() and isinstance(pkv, StaticCache):
                                evt = getattr(pkv, "_ready_event", None)
                                if evt is not None and not evt.query():
                                    (req_stream or torch.cuda.current_stream()).wait_event(evt)                            
                            return  _ORIG_GENERATE(self, *args, **sub_kwargs) 
                        sub_out = _call()
                    else:
                        _ensure_cudagraph_tls()
                        launcher_ctx = no_static_cuda_launcher() if (req_stream and os.name == "nt") else nullcontext()
                        with launcher_ctx, torch.inference_mode():
                            pkv = sub_kwargs.get("past_key_values", None)
                            if torch.cuda.is_available() and isinstance(pkv, StaticCache):
                                evt = getattr(pkv, "_ready_event", None)
                                if evt is not None and not evt.query():
                                    (req_stream or torch.cuda.current_stream()).wait_event(evt)
                            sub_out = _ORIG_GENERATE(self, *args, **sub_kwargs) 

                # put back in order of input batch
                for j, global_idx in enumerate(rows):
                    outs[global_idx] = sub_out[j]

            # restore original adapter
            if prev in (None, "__base__"):
                self.disable_adapter_layers()
            elif prev is not None:
                self.set_adapter(prev)
                self.enable_adapter_layers()
        return outs
        # list, not tensor
    finally:
        # --- Cleanup for the first concurrent request ---
        if acquired_first_user_lock and engine_state:
            engine_state.logger.debug(f"First concurrent request (req={request.request_id if request else 'N/A'}) finished. Releasing lock and setting event.")
            engine_state._first_user_generate_done.set()
            engine_state._first_user_generate_lock.release()

def _run_generate_with_patches(
    model,
    engine_state: "MP13State",
    cancel_criteria: CancellableStoppingCriteria,   # already built!
    confirmation_event: asyncio.Event,
    loop: asyncio.AbstractEventLoop,
    **gen_kwargs,
):
    engine_state.logger.debug("G-START")
    # ------------------------------------------------------------------ #
    # 1.  Inject our criterion into any existing list
    # ------------------------------------------------------------------ #
    sc_list = gen_kwargs.get("stopping_criteria")
    if sc_list is None:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([cancel_criteria])
    else:
        if not isinstance(sc_list, StoppingCriteriaList):
            sc_list = StoppingCriteriaList([sc_list])
        sc_list.append(cancel_criteria)
        gen_kwargs["stopping_criteria"] = sc_list
        
    # ------------------------------------------------------------------ #
    # 1.1  Ensuring PAD as part of EOS
    # ------------------------------------------------------------------ #
#    gen_cfg = gen_kwargs["generation_config"]
#    gen_cfg = ensure_pad_is_eos(gen_cfg)
#    engine.state.logger.debug(f"PAD token used: {gen_cfg.pad_token_id} ({engine_state.tokenizer.pad_token})")
#    gen_kwargs["generation_config"] = gen_cfg 

    # ------------------------------------------------------------------ #
    # 2. Add engine state to kwargs for the patched generate method
    # ------------------------------------------------------------------ #
    gen_kwargs['mp13_engine_state'] = engine_state

    #engine.state.logger.debug(f"[inference] raw handle id={id(model)} class={type(model).__name__}", time.time())

    if engine_state.logger.isEnabledFor(logging.DEBUG):
        from .mp13_adapter import _unwrap_if_compiled, log_model_state, first_lora_status, adapter_health_AB
        m = _unwrap_if_compiled(engine_state.peft_model)
        active_adapter_name_for_health_check = (engine_state._model_active_set[0]
                                                if engine_state._model_active_set
                                                else None)
        log_model_state(engine_state.logger, m, "before-infer")
        first_lora_status(engine_state.logger, m, "before-infer")
        if active_adapter_name_for_health_check:
            adapter_health_AB(engine_state.logger, m, active_adapter_name_for_health_check)
    

    try:
        # ------------------------------------------------------------------ #
        # 2.  Run generation
        # ------------------------------------------------------------------ #
        
        # --- Preflight: ensure input_ids are within vocab range (helps catch batch>1 padding bugs early)
        try:
            _ids = gen_kwargs.get("input_ids", None)
            if isinstance(_ids, torch.Tensor):
                req_stream = gen_kwargs.get("mp13_cuda_stream")
                if req_stream is not None and _ids.device.type == "cuda":
                    with torch.cuda.stream(req_stream):
                        _min_id = int(_ids.min().item())
                        _max_id = int(_ids.max().item())
                else:
                    _min_id = int(_ids.min().item())
                    _max_id = int(_ids.max().item())
                _vsz = int(getattr(getattr(model, "config", None), "vocab_size", 0) or getattr(model, "vocab_size", 0) or 0)
                if _vsz > 0 and (_min_id < 0 or _max_id >= _vsz):
                    raise RuntimeError(f"Invalid token id in input_ids: min={_min_id}, max={_max_id}, vocab_size={_vsz}. Check padding/collator.")
        except Exception:
            # Re-raise with context so the caller sees a clean error instead of a late device-side assert.
            raise

        output = model.generate(**gen_kwargs)
        return output

    finally:
        engine_state.logger.debug(f"G-END (cancelled={cancel_criteria.cancellation_triggered})")
        # ------------------------------------------------------------------ #
        # 3.  Always drain kernels, release memory, and signal completion
        # ------------------------------------------------------------------ #
        try:
            # The patched model.generate now handles its own fine-grained synchronization.
            # This broad torch.cuda.synchronize() call is redundant.
            engine_state.logger.debug("SYNC-DONE (model.generate handles its own sync)")
        except Exception as e_cleanup:
            engine_state.logger.warning(f"Exception during GPU cleanup in generate thread: {e_cleanup}")
        finally:
            # This MUST be called to unblock the main thread, even if cleanup fails.
            if loop.is_running():
                loop.call_soon_threadsafe(confirmation_event.set)


def _yield_streamed_response_in_chunks(
    prompt_text: str,
    engine_state: "MP13State",
    generated_token_ids: List[int],
    tokenizer: Any,
    item_metrics: Dict[str, Any],
    suppress_full_response: bool,
    prompt_index: int,
    skip_tool_parsing: bool,
    max_new_tokens: Optional[int] = None,
    extra_eos_ids: Optional[Union[int, Iterable[int]]] = None,
    preserve_eos_for_parsing: bool = False,
) -> Iterator[InferenceResponse]:
    """
    Takes a full non-streamed response and yields it in streaming-like chunks of tokens.
    It parses for tool calls and yields text and tool blocks separately unless parsing is disabled.
    This is used to emulate a stream from a batch-processed result. It now decodes with special tokens.
    """
    if not generated_token_ids:
        final_item_data: Dict[str, Any] = {
            "chunkType": ChunkType.STREAMING_CHUNK,
            "prompt_index": prompt_index,
            "chunk_text": "",
            "is_final_chunk": True,
            **item_metrics,
        }
        yield InferenceResponse.model_construct(**final_item_data)
        return

    # 1. Define stop tokens (PAD and EOS) to be removed from the final text.
    stop_ids_to_remove = _get_stop_ids_for_output(
        tokenizer,
        extra_eos_ids,
        include_eos=not preserve_eos_for_parsing,
    )
    stop_ids_all = _get_stop_ids_for_output(
        tokenizer,
        extra_eos_ids,
        include_eos=True,
    )
    stop_token_generated = any(t in stop_ids_all for t in generated_token_ids)
    
    clean_token_ids = [t for t in generated_token_ids if t not in stop_ids_to_remove]

    # 2. Decode the cleaned sequence to text, keeping special tokens for the parser.
    full_generated_text = tokenizer.decode(clean_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    eos_strings = _get_eos_strings(tokenizer, extra_eos_ids) if preserve_eos_for_parsing else []
    response_text_for_output = full_generated_text
    if eos_strings and response_text_for_output and stop_token_generated:
        response_text_for_output = _strip_eos_strings_from_end(response_text_for_output, eos_strings)

    tool_blocks: List[ToolCallBlock] = []
    tool_blocks_for_output: Optional[List[ToolCallBlock]] = None
    mark_incomplete = bool(item_metrics.get("was_truncated"))
    if skip_tool_parsing or mark_incomplete:
        final_text_part = full_generated_text
    else:
        # --- Parse for tool blocks ---
        profile_dict = engine_state.tool_parser_profile
        if not profile_dict:
            raise EngineError("Tool parser profile is not available in the engine state for streaming chunk generation.")
        
        parser_profile = ParserProfile(**profile_dict)
        parser = UnifiedToolIO(profile=parser_profile)
        tool_blocks = parser.parse_model_output(
            full_generated_text,
            prompt_index=prompt_index,
            eos_strings=eos_strings,
            mark_truncated=mark_incomplete,
        )

        # Reconstruct the final text part by removing full tool blocks (markers + payload).
        final_text_part = ToolsParserHelper.reconstruct_text_without_block(full_generated_text, tool_blocks)
        tool_blocks_for_output = tool_blocks
        if tool_blocks and engine_state.logger.isEnabledFor(logging.DEBUG):
            engine_state.logger.debug(
                "Tool strip (emulated stream): blocks=%d len=%d->%d",
                len(tool_blocks), len(full_generated_text), len(final_text_part)
            )

    if eos_strings and final_text_part and stop_token_generated:
        final_text_part = _strip_eos_strings_from_end(final_text_part, eos_strings)

    # --- Recalculate tool metrics from the parsed blocks ---
    # This ensures the metrics are correct for the emulated stream, overriding
    # any potentially incomplete metrics passed in from the non-streaming path.
    if tool_blocks_for_output:
        item_metrics["tool_blocks_count"] = len(tool_blocks_for_output)
        item_metrics["tool_blocks_tokens"] = sum(len(tokenizer.encode(block.raw_block)) for block in tool_blocks_for_output)
        output_tokens = None
        try:
            output_tokens = int(item_metrics.get("output_tokens")) if item_metrics.get("output_tokens") is not None else None
        except Exception:
            output_tokens = None
        if _should_mark_truncated_for_hard_stop(
            tool_blocks_for_output,
            output_tokens=output_tokens,
            max_new_tokens=max_new_tokens,
        ):
            item_metrics["was_truncated"] = True


    # Now, chunk the resulting TEXT part of the string.
    string_chunk_size = 10 # characters
    if not final_text_part: # Handle empty text generation (could still have tool calls)
        if item_metrics.get("was_truncated") is not True:
            item_metrics.pop("was_truncated", None)
        # If there's no text, send a single final chunk with tool calls and metrics.
        final_item_data: Dict[str, Any] = {
            "chunkType": ChunkType.STREAMING_CHUNK,
            "prompt_index": prompt_index,
            "chunk_text": "",
            "is_final_chunk": True,
            **item_metrics,
        }
        if tool_blocks_for_output:
            final_item_data["tool_blocks"] = tool_blocks_for_output
        yield InferenceResponse.model_construct(**final_item_data)
        return

    for i in range(0, len(final_text_part), string_chunk_size):
        chunk_text = final_text_part[i:i + string_chunk_size]
        is_final = (i + string_chunk_size) >= len(final_text_part)

        item_data: Dict[str, Any] = {
            "chunkType": ChunkType.STREAMING_CHUNK,
            "prompt_index": prompt_index,
            "chunk_text": chunk_text,
            "is_final_chunk": is_final,
        }
        if is_final:
            if not suppress_full_response:
                item_data["response_text"] = response_text_for_output
            item_data.update(item_metrics)
            if tool_blocks_for_output:
                item_data["tool_blocks"] = tool_blocks_for_output

        yield InferenceResponse.model_construct(**item_data)

def _yield_full_non_streamed_response(
    engine_state: "MP13State",
    generated_token_ids: List[int],
    tokenizer: Any,
    item_metrics: Dict[str, Any],
    suppress_full_response: bool,
    prompt_index: int,
    was_truncated: bool = False,
    skip_tool_parsing: bool = False,
    max_new_tokens: Optional[int] = None,
    extra_eos_ids: Optional[Union[int, Iterable[int]]] = None,
    preserve_eos_for_parsing: bool = False,
) -> Iterator[InferenceResponse]:
    if not generated_token_ids:
        final_item_data: Dict[str, Any] = {
            "chunkType": ChunkType.STREAMING_CHUNK,
            "prompt_index": prompt_index,
            "chunk_text": "",
            "calls": [],
            "is_final_chunk": True,
            **item_metrics,
        }
        yield InferenceResponse.model_construct(**final_item_data)
        return
    
    # 1. Define stop tokens (PAD and EOS) to be removed from the final text.
    stop_ids_to_remove = _get_stop_ids_for_output(
        tokenizer,
        extra_eos_ids,
        include_eos=not preserve_eos_for_parsing,
    )
    stop_ids_all = _get_stop_ids_for_output(
        tokenizer,
        extra_eos_ids,
        include_eos=True,
    )
    stop_token_generated = any(t in stop_ids_all for t in generated_token_ids)
    
    clean_token_ids = [t for t in generated_token_ids if t not in stop_ids_to_remove]

    # 2. Decode the sequence to text, keeping special tokens for the parser.
    full_generated_text = tokenizer.decode(clean_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    eos_strings = _get_eos_strings(tokenizer, extra_eos_ids) if preserve_eos_for_parsing else []
    response_text_for_output = full_generated_text
    if eos_strings and response_text_for_output and stop_token_generated:
        response_text_for_output = _strip_eos_strings_from_end(response_text_for_output, eos_strings)

    tool_blocks_for_output: Optional[List[ToolCallBlock]] = None
    num_tool_blocks = 0
    num_tool_block_tokens = 0

    mark_incomplete = was_truncated or bool(item_metrics.get("was_truncated"))
    if skip_tool_parsing or mark_incomplete:
        final_text_part = full_generated_text
    else:
        # 3. Run the parser on the raw, decoded text to identify tool blocks.
        tool_blocks: List[ToolCallBlock] = []

        profile_dict = engine_state.tool_parser_profile
        if not profile_dict:
            raise EngineError("Tool parser profile is not available in the engine state.")
        parser_profile = ParserProfile(**profile_dict)
        parser = UnifiedToolIO(profile=parser_profile)
        tool_blocks = parser.parse_model_output(
            full_generated_text,
            prompt_index=prompt_index,
            eos_strings=eos_strings,
            mark_truncated=mark_incomplete,
        )
        # 4. Reconstruct the final text part by removing full tool blocks (markers + payload).
        final_text_part = ToolsParserHelper.reconstruct_text_without_block(full_generated_text, tool_blocks)
        tool_blocks_for_output = tool_blocks
        if tool_blocks and engine_state.logger.isEnabledFor(logging.DEBUG):
            engine_state.logger.debug(
                "Tool strip (non-stream): blocks=%d len=%d->%d",
                len(tool_blocks), len(full_generated_text), len(final_text_part)
            )

        num_tool_blocks = len(tool_blocks)
        num_tool_block_tokens = sum(len(tokenizer.encode(block.raw_block)) for block in tool_blocks)
        item_metrics["tool_blocks_count"] = num_tool_blocks if num_tool_blocks > 0 else None
        item_metrics["tool_blocks_tokens"] = num_tool_block_tokens if num_tool_block_tokens > 0 else None        
        output_tokens = None
        try:
            output_tokens = int(item_metrics.get("output_tokens")) if item_metrics.get("output_tokens") is not None else None
        except Exception:
            output_tokens = None
        if _should_mark_truncated_for_hard_stop(
            tool_blocks_for_output,
            output_tokens=output_tokens,
            max_new_tokens=max_new_tokens,
        ):
            item_metrics["was_truncated"] = True

    if eos_strings and final_text_part and stop_token_generated:
        final_text_part = _strip_eos_strings_from_end(final_text_part, eos_strings)


    if not final_text_part:
        if not item_metrics.get("was_truncated"):
            item_metrics.pop("was_truncated", None)

    final_item_data: Dict[str, Any] = {
        "chunkType": ChunkType.STREAMING_CHUNK,
        "prompt_index": prompt_index,
        "chunk_text": final_text_part,
        "is_final_chunk": True,
        **item_metrics,
    }

    if tool_blocks_for_output:
        final_item_data["tool_blocks"] = tool_blocks_for_output

    if not suppress_full_response:
        final_item_data["response_text"] = response_text_for_output

    yield InferenceResponse.model_construct(**final_item_data)

def _get_prompt_for_response(
    engine: "MP13Engine",
    request: InferenceRequest,
    prompt_index: int,
    full_formatted_prompt: str,
    tokenizer: Any
) -> Optional[str]:
    """
    Determines what prompt text to return in the PROMPT_STARTED chunk based on request.return_prompt.
    This version correctly applies the chat template when needed.
    """
    if not request.return_prompt:
        return None

    if request.return_prompt == "full":
        return full_formatted_prompt


    if request.return_prompt == "last":
        if not request.messages_list or prompt_index >= len(request.messages_list):
            return full_formatted_prompt # Fallback

        messages = request.messages_list[prompt_index]
        # Find the last message that is not from the assistant
        last_non_assistant_msg = None
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                last_non_assistant_msg = msg
                break
        
        if not last_non_assistant_msg:
            return "" # No non-assistant messages found

        # Check if it's the first user message in the history
        user_message_count = sum(1 for m in messages if m.get("role") == "user")

        system_msg = None
        should_strip_system_prompt = False
        if user_message_count == 1 and last_non_assistant_msg.get("role") == "user":
            # This is very first user message need to pass through chat template
            system_msg = next((msg for msg in messages if msg.get("role") == "system"), None)
            if system_msg == None:
                # user really wanted remove user message pass empty to template and strip afterwards
                system_msg = ""  
                should_strip_system_prompt = True
            elif system_msg == "<>" or system_msg == "<def>":
                system_msg = None # means user wanted default so chat template will add one.
        else:
            # Not the first user message, or the last message was not 'user' (e.g., 'tool')
            return last_non_assistant_msg.get("content", "")

        # Convert Tool objects to dicts for format_prompt_messages
        tools_for_template: Optional[List[Union[Dict[str, Any], Callable]]] = None
        tools = request.tools
        if tools:
            tools_for_template = []
            for tool in tools:
                if isinstance(tool, Tool):
                    tools_for_template.append(tool.model_dump())
                elif callable(tool):
                    tools_for_template.append(tool)

        # Re-apply the chat template to just the last user/tool message, but include the system prompt.
        messages_for_last_turn = [system_msg, last_non_assistant_msg] if system_msg else [last_non_assistant_msg]
        formatted = format_prompt_messages(
            engine.state.logger,
            example={"messages": messages_for_last_turn}, 
            columns=ColumnsConfig(messages="messages"), tokenizer=tokenizer,
            tools=tools_for_template if tools_for_template else None, 
            add_generation_prompt=False,
            strip_empty_system_prompt=should_strip_system_prompt,
            empty_system_prompt_template=engine.state.empty_system_prompt_template
        )
        
        return formatted.text

    return None


def to_device_on_stream(encoding: BatchEncoding, device: torch.device, stream: torch.cuda.Stream | None):
    """
    Moves a BatchEncoding to 'device' with non_blocking copies, and crucially enqueues
    those copies on 'stream' (if provided) to avoid default-stream serialization.
    Returns a BatchEncoding (not a plain dict) so attribute access still works.
    """
    if not isinstance(encoding, BatchEncoding):
        # If upstream gave a dict-like, normalize to BatchEncoding once.
        encoding = BatchEncoding(encoding, tensor_type='pt')

    if stream is None:
        # Fall back to regular .to(); attribute access remains intact.
        return encoding.to(device)
    # Prefer a tiny per-stream pinned staging cache for very small inputs.
    # This avoids per-request pin_memory() overhead while keeping H2D async.
    with torch.inference_mode():
        with torch.cuda.stream(stream):
            pinned_cache = getattr(stream, "mp13_pinned_buffers", None)
            pinned_bytes = getattr(stream, "mp13_pinned_bytes", 0)
            pinned_cap   = getattr(stream, "mp13_pinned_bytes_cap", 2 * 1024 * 1024)

            new_items = {}
            for k, t in list(encoding.items()):
                if not isinstance(t, torch.Tensor):
                    continue
                if t.device.type == "cuda":
                    # Ensure lifetime is tied to this stream
                    try:
                        t.record_stream(stream)
                    except Exception:
                        pass
                    new_items[k] = t
                    continue

                # For tiny tensors (<= 64 KiB), reuse a pinned staging tensor.
                nbytes = t.element_size() * t.numel()
                tiny      = (nbytes <= 64 * 1024)
                use_cache = isinstance(pinned_cache, dict) and tiny

                if use_cache:
                    key = (t.dtype, tuple(t.shape))
                    buf = pinned_cache.get(key)
                    if buf is None or buf.numel() != t.numel():
                        if (pinned_bytes + nbytes) <= pinned_cap:
                            buf = torch.empty_like(t, pin_memory=True)
                            pinned_cache[key] = buf
                            try:
                                setattr(stream, "mp13_pinned_bytes", pinned_bytes + nbytes)
                            except Exception:
                                pass
                        else:
                            out = t.to(device, non_blocking=False)
                            try:
                                out.record_stream(stream)
                            except Exception:
                                pass
                            new_items[k] = out
                            continue
                    # CPU copy into pinned buffer, then async H2D on this stream.
                    buf.copy_(t, non_blocking=False)
                    out = buf.to(device, non_blocking=True)
                    try:
                        out.record_stream(stream)
                    except Exception:
                        pass
                    new_items[k] = out
                else:
                    # Larger tensors: occasional one-time pin+H2D
                    tt = t.contiguous()
                    if not tt.is_pinned():
                        try:
                            tt = tt.pin_memory()
                        except RuntimeError:
                            out = tt.to(device, non_blocking=False)
                            try:
                                out.record_stream(stream)
                            except Exception:
                                pass
                            new_items[k] = out
                            continue
                    # For larger tensors we still allocate on device once here
                    out = tt.to(device, non_blocking=True)
                    try:
                        out.record_stream(stream)
                    except Exception:
                        pass
                    new_items[k] = out

            for k, v in new_items.items():
                encoding[k] = v
    return encoding

async def _generate_non_streamed_batch_internal(
    engine,
    request: InferenceRequest,
    prompts_to_process: List[str],
    adapter_names_to_use: List[str],
    pass_adapters_to_generate: bool,
    generation_config: GenerationConfig,
    inference_model: torch.nn.Module, # type: ignore,
    request_tokenizer: Any,
    cancel_event_threadsafe: threading.Event,
    prompt_token_counts: List[int],
    deferred_slots_accumulator: List[Tuple[Tuple[int, int], frozenset[str]]],
    streaming: bool,
    skip_tool_parsing: bool,
    start_index: int,
    batch_durations_collector: List[float],
    req_stream_for_batch: Optional[torch.cuda.Stream],
) -> AsyncIterator[InferenceResponse]:
    """
    Internal helper to process a list of prompts as a non-streaming batch.
    This includes creating non-mixed sub-batches if necessary.
    If streaming=True, yields chunked responses, else yields full responses.
    """
    loop = asyncio.get_running_loop()
    preserve_eos_for_parsing = False
    if not skip_tool_parsing:
        profile_dict = engine.state.tool_parser_profile
        if profile_dict:
            preserve_eos_for_parsing = bool(profile_dict.get("preserve_eos_for_parsing"))
    prompt_batches: List[List[str]] = []
    adapter_name_batches: List[List[str]] = []
    max_batch_size = 8 # TODO: Make configurable

    # Simplified batching logic. PeftMixedModel correctly handles `__base__` within
    # the `adapter_names` list, so the complex splitting logic is not necessary.
    for i in range(0, len(prompts_to_process), max_batch_size):
        prompt_batches.append(prompts_to_process[i:i + max_batch_size])
        # Also slice the corresponding token counts for this batch
        adapter_name_batches.append(adapter_names_to_use[i:i + max_batch_size])


    # This loop processes micro-batches
    for batch_idx, prompt_batch_list in enumerate(prompt_batches):
        if not prompt_batch_list:
            continue
        adapter_names_for_batch = adapter_name_batches[batch_idx]

        target_device = first_module_device(inference_model)
        if len(prompt_batch_list) == 1:
            cpu_inputs = request_tokenizer(
                prompt_batch_list, return_tensors="pt", padding=False
            )
        else:
            cpu_inputs = request_tokenizer(
                prompt_batch_list, return_tensors="pt", padding="longest", pad_to_multiple_of=64
            )
        inputs = to_device_on_stream(cpu_inputs, target_device, req_stream_for_batch)        

        # --- Pass engine state and request for potential micro-batch cache routing ---
        # The patched PeftMixedModel.generate will pop these if it needs them.
        gen_kwargs = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask}
        gen_kwargs['mp13_engine_state'] = engine.state
        if req_stream_for_batch:
            gen_kwargs['mp13_cuda_stream'] = req_stream_for_batch
        gen_kwargs['mp13_request'] = request

        # --- Static KV Cache Logic for Batch ---
        # Determine if we can use static cache for this batch. It's only possible if all prompts
        # in the batch use the same adapter combination.
        unique_adapter_sets = {frozenset(name.split(',')) if isinstance(name, str) else frozenset([name]) for name in adapter_names_for_batch}
        can_use_static_for_batch = len(unique_adapter_sets) <= 1

        current_batch_token_counts = prompt_token_counts[batch_idx * max_batch_size : (batch_idx + 1) * max_batch_size]
        # For stopping criteria and cache routing, we MUST use the post-padding length
        # from the tokenized tensor. This accounts for `pad_to_multiple_of`.
        max_padded_prompt_len = int(inputs.input_ids.size(1))
        
        max_new_from_config = getattr(generation_config, "max_new_tokens", None)

        # Create the stopping criteria here, now that we have the final config and prompt length.
        cancel_criteria = CancellableStoppingCriteria(
            engine.state.logger,
            cancel_event=cancel_event_threadsafe,
            max_new_tokens=max_new_from_config,
            prompt_length=max_padded_prompt_len
        )

        gen_kwargs["generation_config"] = generation_config

        cache_metric_str: str = ""
        cache_warming_str: Optional[str] = None
        deferred_for_this_micro_batch = [] # Local accumulator for this micro-batch
        cache_mode = "dynamic"
        cache_bucket = None
        eff_max_new = max_new_from_config

        if can_use_static_for_batch:
            # All prompts use the same adapter, so we can check for a static cache slot.
            # The adapter combination is the single unique set, or an empty set for base model.
            adapter_combo_for_cache = list(unique_adapter_sets)[0] if unique_adapter_sets else frozenset()
            cache_mode, cache_bucket, eff_max_new = maybe_apply_static_cache(
                engine_state=engine.state, model=inference_model, gen_kwargs=gen_kwargs,
                batch_size=inputs.input_ids.size(0), prompt_len=max_padded_prompt_len, max_new_tokens=max_new_from_config,
                request=request, active_adapters_for_request=list(adapter_combo_for_cache),
                deferred_slots_accumulator=deferred_for_this_micro_batch
            )
            if cache_mode == "static" and eff_max_new is not None and eff_max_new != max_new_from_config:
                gen_kwargs["generation_config"] = get_modified_generation_config(generation_config, max_new_tokens=eff_max_new)
        else:
            engine.state.logger.debug(f"Batch has mixed adapters {unique_adapter_sets}. Forcing dynamic cache.")
            gen_kwargs.pop("past_key_values", None) # Ensure dynamic

        using_static = cache_mode == "static"
        if cache_mode == "static" and cache_bucket:
            cache_metric_str = f"static (B={cache_bucket[0]}, L={cache_bucket[1]})"
        elif cache_mode == "dynamic":
            total_len = max_padded_prompt_len + (eff_max_new if eff_max_new is not None else max_new_from_config or 0)
            cache_metric_str = f"dynamic (L={total_len})"
        elif cache_mode == "offloaded":
            total_len = max_padded_prompt_len + (eff_max_new if eff_max_new is not None else max_new_from_config or 0)
            cache_metric_str = f"offloaded (L={total_len})"

        if deferred_for_this_micro_batch:
            deferred_slots_accumulator.extend(deferred_for_this_micro_batch)
            slot = deferred_for_this_micro_batch[0]
            cache_warming_str = f"(B={slot[0][0]},L={slot[0][1]})"
        #  END cache

        if pass_adapters_to_generate:
            if (hasattr(inference_model, "set_adapter") and hasattr(inference_model, "disable_adapter_layers")):
                gen_kwargs["adapter_names"] = adapter_names_for_batch
            else:
                 raise InferenceRequestError(
                     f"(BUG) the base model cannot be used for a batch containing adapters: {adapter_names_for_batch}."
                 )

        # Capture the final prompt length for slicing AFTER any potential padding from static cache
        prompt_len_for_slicing = gen_kwargs["input_ids"].shape[1]

        batch_gen_start_time = time.monotonic()

        was_canceled = False
        output_sequences = []
        with torch.no_grad(): # type: ignore
            future = loop.run_in_executor(
                engine.state._gen_exec,
                functools.partial(  # wrap the function
                    _run_generate_with_patches,
                    inference_model, engine.state,
                    cancel_criteria, asyncio.Event(), # Dummy confirmation event
                    loop, 
                    **gen_kwargs
                ))
            
            # Poll the thread-safe cancel event from this async task
            async def _cancel_waiter(evt: threading.Event):
                while not evt.is_set(): await asyncio.sleep(0.05)
            
            cancel_wait = asyncio.create_task(_cancel_waiter(cancel_criteria._cancel_event))
            try:
                done, pending = await asyncio.wait({future, cancel_wait},
                                            return_when=asyncio.FIRST_COMPLETED)
                if cancel_wait in done:
                    # cancel the still‑running generate() thread – running futures can't
                    # be killed, but we can set the flag *and* forget the result
                    future.cancel()
                    was_canceled = True
                else:
                    # Future finished, we can cancel the waiter
                    if cancel_wait in pending:
                        cancel_wait.cancel()
                    output_sequences = future.result()  # normal path
            except (KeyboardInterrupt, asyncio.CancelledError) as e: # type: ignore
                engine.state.logger.debug(f"[cancel-debug] Non-streaming batch caught {type(e).__name__}. Cancelling future.")
                if not future.done():
                    future.cancel()
                engine.state.logger.info("Non-streaming batch generation cancelled by client.")
                raise # Re-raise to propagate cancellation

        batch_gen_duration = time.monotonic() - batch_gen_start_time
        batch_durations_collector.append(batch_gen_duration)

        if was_canceled or cancel_criteria.cancellation_triggered:
            engine.state.logger.info(f"Non-streaming batch {batch_idx} was cancelled.")
            raise InferenceCancelledError("Inference was cancelled by user request.")

        # New logic: Mark static cache slot as successfully used
        if using_static and not was_canceled and not cancel_criteria.cancellation_triggered:
            pkv_to_mark = gen_kwargs.get("past_key_values")
            if pkv_to_mark and isinstance(pkv_to_mark, StaticCache):
                pkv_to_mark._is_successfully_used = True

        # The boolean value of a tensor is ambiguous. We must handle it separately.
        # This block is a fallback for cancellation, so we check for empty results.
        sequences_are_empty = False
        if isinstance(output_sequences, torch.Tensor):
            # A tensor from generate() on a batch should not be empty.
            # The boolean check is ambiguous, so we check its number of elements.
            sequences_are_empty = output_sequences.numel() == 0
        elif not output_sequences: # For lists or other sequences
            sequences_are_empty = True

        if sequences_are_empty: # Can happen if cancelled
            # With the improved was_canceled flag, we can be certain.
            if was_canceled:
                engine.state.logger.info(f"Non-streaming batch {batch_idx} was cancelled and yielded no output sequences.")
            else:
                engine.state.logger.warning(f"Non-streaming batch {batch_idx} yielded no output sequences for an unknown reason.")

            for i, prompt_text in enumerate(prompt_batch_list):
                prompt_to_return = _get_prompt_for_response (
                    engine=engine,
                    request=request,
                    prompt_index=start_index + (batch_idx * max_batch_size) + i,
                    full_formatted_prompt=prompt_text,
                    tokenizer=request_tokenizer
                )
                global_prompt_index = start_index + (batch_idx * max_batch_size) + i
                # Add was_canceled metric to the error chunk.
                yield InferenceResponse.model_construct(chunkType=ChunkType.ERROR, prompt_index=global_prompt_index, prompt=prompt_to_return, error="Cancelled by user", was_truncated=was_canceled, was_canceled=True, is_final_chunk=True)
            return
        
        for i in range(len(output_sequences)):
            all_generated_tokens = output_sequences[i][prompt_len_for_slicing:]
            global_prompt_index = start_index + (batch_idx * max_batch_size) + i

            # --- Truncation Detection ---
            was_truncated = False
            if cancel_criteria.max_tokens_triggered:
                batch_size = len(output_sequences)
                if batch_size == 1:
                    # If batch size is 1, max_tokens_triggered is the ultimate truth.
                    was_truncated = True
                else:
                    # For batches > 1, a sequence is truncated if the batch-level token limit was triggered,
                    # AND this specific sequence did not happen to end on an EOS token.
                    eos_token_id = gen_kwargs["generation_config"].eos_token_id
                    stop_ids = set()
                    if isinstance(eos_token_id, int): stop_ids.add(eos_token_id)
                    elif isinstance(eos_token_id, list): stop_ids.update(eos_token_id)

                    # --- Find the last non-padding token ---
                    # In a batch, shorter sequences that finish early are padded to the length of the longest sequence.
                    # We must find the actual last token generated before padding.
                    last_meaningful_token_id = None
                    generated_ids_list = all_generated_tokens.tolist()
                    for token_id in reversed(generated_ids_list):
                        if token_id != request_tokenizer.pad_token_id:
                            last_meaningful_token_id = token_id
                            break

                    # A sequence is considered truncated if the token limit was hit AND its last meaningful token was NOT a stop token.
                    last_token_is_stop = bool(last_meaningful_token_id is not None and last_meaningful_token_id in stop_ids)
                    was_truncated = not last_token_is_stop

            # Determine adapter display name
            adapter_display_name: str
            if request.override_adapters:
                # An override was used. The name for this item is in the batch list.
                adapter_display_name = adapter_names_for_batch[i]
            else:
                # Use the adapters explicitly provided on the request (if any), otherwise the engine's active set.
                label_source: Optional[Union[str, Sequence[str]]] = request.active_adapters if request.active_adapters is not None else adapter_names_to_use
                adapter_display_name = _format_adapter_label(label_source)
            
            prompt_to_return = _get_prompt_for_response(
                engine=engine,
                request=request,
                prompt_index=global_prompt_index, 
                full_formatted_prompt=prompt_batch_list[i],
                tokenizer=request_tokenizer
                )
            prompt_started_payload: Dict[str, Any] = {"chunkType": ChunkType.PROMPT_STARTED, "prompt_index": global_prompt_index, "adapters": adapter_display_name}
            if prompt_to_return is not None:
                prompt_started_payload["prompt"] = prompt_to_return
            yield InferenceResponse.model_construct(**prompt_started_payload)
            
            #eos_token_ids = gen_kwargs["generation_config"].eos_token_id
            #if not isinstance(eos_token_ids, list): eos_token_ids = [eos_token_ids]

            #STRIP PADs left by HF generate()
            generated_tokens = [t for t in all_generated_tokens.tolist() if t != request_tokenizer.pad_token_id]
            if was_truncated and not generated_tokens:
                was_truncated = False

            raw_item_metrics: Dict[str, Any] = {
                "input_tokens": current_batch_token_counts[i],
                "output_tokens": len(generated_tokens),
                "generation_duration_sec": batch_gen_duration,
                "tokens_per_second": len(generated_tokens) / batch_gen_duration if batch_gen_duration > 0 else 0,
                "time_to_first_token_sec": batch_gen_duration, # For batches, TTFT is the whole batch duration
                # Tentative flags; these will be filtered below so False values are not sent
                "was_truncated": was_truncated,
                "was_canceled": was_canceled,
                "cache_metric": cache_metric_str,
                "cache_warming": cache_warming_str
            }

            # Round numeric values first
            rounded = round_floats(raw_item_metrics)

            # Build final item_metrics applying policy:
            # - Include was_truncated / was_canceled only if True
            # - Drop keys with None or empty string/list/dict
            item_metrics: Dict[str, Any] = {}
            for k, v in rounded.items():
                if k in ("was_truncated", "was_canceled"):
                    if v is True:
                        item_metrics[k] = v
                    continue
                if v is None:
                    continue
                if isinstance(v, str) and v == "":
                    continue
                if isinstance(v, (list, dict)) and len(v) == 0:
                    continue
                item_metrics[k] = v

            if streaming:
                for item in _yield_streamed_response_in_chunks(
                            prompt_batch_list[i],
                            engine.state,
                            generated_tokens,
                            request_tokenizer,
                            item_metrics,
                            request.suppress_full_response,
                            global_prompt_index,
                            skip_tool_parsing,
                            max_new_tokens=eff_max_new if eff_max_new is not None else max_new_from_config,
                            extra_eos_ids=getattr(generation_config, "eos_token_id", None),
                            preserve_eos_for_parsing=preserve_eos_for_parsing):
                    yield item
            else:
                for item in _yield_full_non_streamed_response(
                            engine_state=engine.state,
                            generated_token_ids=generated_tokens,
                            tokenizer=request_tokenizer,
                            item_metrics=item_metrics,
                            suppress_full_response=request.suppress_full_response,
                            prompt_index=global_prompt_index,
                            was_truncated=was_truncated,
                            skip_tool_parsing=skip_tool_parsing,
                            max_new_tokens=eff_max_new if eff_max_new is not None else max_new_from_config,
                            extra_eos_ids=getattr(generation_config, "eos_token_id", None),
                            preserve_eos_for_parsing=preserve_eos_for_parsing):
                    yield item

async def _prepare_prompts_for_inference(
    engine, 
    request: InferenceRequest, 
    request_tokenizer: Any
) -> Tuple[List[str], List[int], List[str], List[Dict[str, Any]]]:
    """
    Prepares a list of prompt strings from an InferenceRequest.
    This is the shared logic for both run_inference and format_inference_prompt.
    Returns a tuple of (prompts_to_process, prompt_token_counts, errors, error_details).
    """
    errors: List[str] = []
    error_details: List[Dict[str, Any]] = []

    if request.do_continue:
        engine.state.logger.debug(f"This is a do_continue request for all prompts in batch size: {len(request.messages_list)}")

    # --- Manual Tool Injection if Needed ---
    tools_for_template: Optional[List[Union[Dict[str, Any], Callable]]] = None
    # The request now only contains the serialized toolkit or None.
    if request.tools:
        # Extract the list of tool dictionaries from the ToolsSerialized object.
        tools_for_template = [t.model_dump() for t in request.tools.for_dump]
        if not engine.state._tool_templates and not engine.state._tool_parser_profile:
            engine.state.logger.warning("Tools provided but no tool-supporting chat template found. Manually injecting tool descriptions into system prompt.")
            tools_as_dicts = [t for t in tools_for_template if isinstance(t, dict)]
            if any(not isinstance(t, dict) for t in tools_for_template):
                engine.state.logger.warning("Callable tools cannot be manually injected into the prompt and were ignored.")
            
            if tools_as_dicts and request.messages_list:
                tool_str = json.dumps(tools_as_dicts, indent=2)
                tool_prompt_injection = f"\n\n# Tools\nThe user has access to the following tools. Respond with a `tool_code` block if you need to use them.\n```json\n{tool_str}\n```"
                for i, msg_list in enumerate(request.messages_list):
                    system_msg_found = False
                    for msg in msg_list:
                        if msg.get("role") == "system":
                            msg["content"] += tool_prompt_injection
                            system_msg_found = True
                            break
                    if not system_msg_found:
                        request.messages_list[i].insert(0, {"role": "system", "content": tool_prompt_injection.strip()})
    # --- End Manual Tool Injection ---

    prompts_to_process: List[str] = []
    prompt_token_counts: List[int] = []

    if request.raw_list:
        prompts_to_process = request.raw_list
        # Tokenize each raw prompt to get its length
        prompt_token_counts = [len(request_tokenizer.encode(p)) for p in prompts_to_process]
    elif request.messages_list:
        for i, messages_item in enumerate(request.messages_list):
            try:
                processed_messages = list(messages_item) # Make a mutable copy
                should_strip_system_prompt = False

                # Case 1: User wants to strip the system prompt block entirely.
                # This is triggered by the absence of a system message in the list.
                if not processed_messages or processed_messages[0].get("role") != "system":
                    should_strip_system_prompt = True
                    # To get a predictable template to strip, we must insert a placeholder empty system message.
                    processed_messages.insert(0, {"role": "system", "content": ""})

                # Case 2: User wants model's default system prompt.
                # This must be checked *after* the stripping case.
                elif processed_messages[0].get("role") == "system" and processed_messages[0].get("content") == "<>":
                    processed_messages.pop(0)

                # --- FIX: Rehydrate parser_profile from dict to object ---
                parser_profile_obj = None
                if profile_dict := engine.state.tool_parser_profile:
                    parser_profile_obj = ParserProfile(**profile_dict)

                formatted = format_prompt_messages(
                    engine.state.logger,
                    example={"messages": processed_messages},
                    columns=ColumnsConfig(messages="messages"), # type: ignore
                    parser_profile=parser_profile_obj or engine.state.tool_parser_profile,
                    tokenizer=request_tokenizer,
                    tools=tools_for_template if tools_for_template else None,
                    add_generation_prompt=request.do_continue is None or not request.do_continue,
                    continue_final_message=request.do_continue if request.do_continue else False,
                    strip_empty_system_prompt=should_strip_system_prompt,
                    empty_system_prompt_template=engine.state.empty_system_prompt_template,
                    strip_eos_token=bool(request.do_continue)
                )

                prompts_to_process.append(formatted.text)
                if formatted.token_count is not None:
                    prompt_token_counts.append(formatted.token_count)
                else:
                    prompt_token_counts.append(len(request_tokenizer.encode(formatted.text)))
            except Exception as e_format:
                error_msg = f"Error formatting message seq {i}: {e_format}"
                error_traceback = traceback.format_exc()
                engine.state.logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
                error_details.append(
                    {
                        "prompt_index": i,
                        "error": error_msg,
                        "full_traceback": error_traceback,
                    }
                )
                prompts_to_process.append(f"ERROR: {error_msg}")
                prompt_token_counts.append(0) # Add a zero count for errored prompts
    
    return prompts_to_process, prompt_token_counts, errors, error_details

async def format_inference_prompt_logic(engine: "MP13Engine", request: InferenceRequest, request_tokenizer: Optional[Any] = None) -> Dict[str, Any]:
    """
    Formats an inference request into a list of prompt strings without running generation.
    Returns a dictionary suitable for the `data` field of an MP13Response.
    """
    resource_to_checkin: Optional["RequestResource"] = None
    try:
        if engine.state.engine_mode != EngineModeState.INFERENCE:
            raise ModeMismatchError(f"Cannot format prompt. Engine mode is '{engine.state.engine_mode.value if engine.state.engine_mode else 'UNSET'}', expected INFERENCE.")
        if engine.state.tokenizer is None:
            raise EngineError("Inference failed. Engine tokenizer not available.")

        if request_tokenizer is None:
            resource = await engine.checkout_resource(request.request_id or "format_prompt")
            request_tokenizer = resource.tokenizer
            resource_to_checkin = resource

        # Delegate to the shared prompt preparation function
        prompts_to_process, tokens_count, errors, error_details = await _prepare_prompts_for_inference(engine, request, request_tokenizer)
        
        return {
            "formatted_prompts": prompts_to_process,
            "prompt_token_counts": tokens_count,
            "errors": errors or None,
            "error_details": error_details or None,
        }
    finally:
        if resource_to_checkin:
            await engine.checkin_resource(resource_to_checkin)

async def count_tokens_logic(engine: "MP13Engine", text: str, is_repr: bool, request_tokenizer: Optional[Any] = None) -> Dict[str, Any]:
    """
    Counts the number of tokens in a given text string.
    Returns a dictionary suitable for the `data` field of an MP13Response.
    """
    resource_to_checkin: Optional["RequestResource"] = None
    try:
        if engine.state.engine_mode != EngineModeState.INFERENCE:
            raise ModeMismatchError(f"Cannot count tokens. Engine mode is '{engine.state.engine_mode.value if engine.state.engine_mode else 'UNSET'}', expected INFERENCE.")
        if request_tokenizer is None:
            resource = await engine.checkout_resource("count_tokens_request")
            request_tokenizer = resource.tokenizer
            resource_to_checkin = resource

        if request_tokenizer is None: # Should not happen if engine is initialized
            raise EngineError("Could not acquire a tokenizer to count tokens.")

        text_to_tokenize = text
        if is_repr:
            text_to_tokenize = codecs.decode(text, 'unicode_escape')

        token_ids = request_tokenizer.encode(text_to_tokenize, add_special_tokens=False)
        return {"token_count": len(token_ids), "text_processed": text_to_tokenize}
    finally:
        if resource_to_checkin:
            await engine.checkin_resource(resource_to_checkin)

async def _internal_response_generator(
    start_time_mono,        
    engine: "MP13Engine",
    request: InferenceRequest,
    prompts_to_process: List[str],
    adapter_names_to_use: List[str],
    pass_adapters_to_generate: bool,
    start_time_wall: float,
    inference_model: torch.nn.Module, # type: ignore
    request_tokenizer: Any,    
    prompt_token_counts: List[int],
    cancel_event_threadsafe: threading.Event,
    deferred_slots_accumulator: List[Tuple[Tuple[int, int], frozenset[str]]],
    req_stream: Optional[torch.cuda.Stream]
) -> AsyncIterator[InferenceResponse]:
    """
    The core async generator for inference. It handles batching, streaming vs. non-streaming,
    and yields InferenceResponse objects for each part of the response.
    It concludes by yielding a final item containing aggregate metrics for the entire request.
    """
    loop = asyncio.get_running_loop()
    skip_tool_parsing = bool(engine.state.no_tools_parse or getattr(request, "no_tools_parse", False))

    # --- Model is now passed from the main function to ensure proper timing ---
    if inference_model is None:
        # This case should be rare if base_model exists, but it's a good safeguard.
        raise EngineError("Inference failed. Could not determine an active model.")

    base_gen_config = getattr(inference_model, 'generation_config', None)
    if base_gen_config and hasattr(base_gen_config, 'to_dict'):
        final_gen_config_dict = base_gen_config.to_dict()
    else:
        # Fallback if generation_config is not a GenerationConfig object
        final_gen_config_dict = {}

    final_gen_config_dict.update(engine.state.current_inference_session_config.get('default_generation_config', {}) if engine.state.current_inference_session_config else {})
    final_gen_config_dict.update(request.generation_config or {})
    
    # If do_sample is not specified, default to False.
    if 'do_sample' not in final_gen_config_dict:
        final_gen_config_dict['do_sample'] = False

    # Create a base GenerationConfig from the merged dict, filtering out None values
    base_config_for_modification = GenerationConfig(**{k: v for k, v in final_gen_config_dict.items() if v is not None})
    
    # Use the helper to get the final, clean config. It will handle popping sampling keys if do_sample=False.
    generation_config = get_modified_generation_config(base_config_for_modification)

    # Create the stopping criteria here, now that we have the final config.
    # For the streaming path, we set prompt_length here. For non-streaming,
    # it's set inside _generate_non_streamed_batch_internal.
    cancel_criteria = CancellableStoppingCriteria(
        engine.state.logger,
        cancel_event=cancel_event_threadsafe,
        max_new_tokens=generation_config.max_new_tokens, # type: ignore
        prompt_length=prompt_token_counts[0] if request.stream and prompts_to_process else 0
    )


    def _generation_thread_wrapper(model, engine_state: "MP13State", streamer, cancel_criteria, confirmation_event, loop, **kwargs):
        """
        Wraps the model.generate call to ensure streamer is closed on error, preventing hangs.
        This function is blocking and intended to be run in a separate thread.
        """
        try:
            # Ensure this worker thread uses the model's CUDA device (device_map="cuda:1" etc.)
            p0 = next(model.parameters(), None)
            if p0 is not None and p0.is_cuda:
                torch.cuda.set_device(p0.device.index)

            engine_state.logger.info(f"adapter_names: {kwargs.get('adapter_names', 'none')}")
            _run_generate_with_patches(model, engine_state, cancel_criteria, confirmation_event, loop, **kwargs, streamer=streamer)
        except BaseException as e: # Catch BaseException to handle GeneratorExit, etc.
                # Log the exception from within the thread for debugging.
                # The exception is not re-raised, as the primary goal is to unblock the main loop.
            engine_state.logger.error(f"Exception in generation thread: {type(e).__name__}: {e}", exc_info=True)
            # Avoid logging a full traceback for expected generator exits.
            if not isinstance(e, (GeneratorExit, StopIteration, asyncio.CancelledError, KeyboardInterrupt)):
                engine_state.logger.exception("Unhandled exception in generation thread")
            raise # Re-raise the exception to propagate it to the main event loop
        finally:
            # This is the critical part of the fix. It ensures the streamer's iterator
            # will raise StopIteration, unblocking the `await loop.run_in_executor(...)`
            # call in the main async loop.
            # The confirmation event is now set inside _run_generate_with_patches
            streamer.end()

    # --- Metrics Aggregation ---
    all_final_items_for_metrics: List[InferenceResponse] = []
    micro_batch_durations: List[float] = []

    if request.stream and prompts_to_process:
        # --- STREAMING PATH: Stream first item, then batch the rest for performance ---
        # 1. Process the first item with streaming to get fast TTFT
        first_prompt = prompts_to_process[0]
        first_adapter = adapter_names_to_use[0] if adapter_names_to_use else "__base__"
        
        adapter_display_name: str
        if request.override_adapters:
            adapter_display_name = first_adapter
        else:
            label_source: Optional[Union[str, Sequence[str]]] = request.active_adapters if request.active_adapters is not None else adapter_names_to_use
            adapter_display_name = _format_adapter_label(label_source)

        prompt_to_return = _get_prompt_for_response(
            engine=engine,
            request=request,
            prompt_index=0, # It's always the first item in the streaming path
            full_formatted_prompt=first_prompt,
            tokenizer=request_tokenizer
        )

        prompt_started_payload: Dict[str, Any] = {
            "chunkType": ChunkType.PROMPT_STARTED,
            "prompt_index": 0,
            "adapters": adapter_display_name,
        }
        if prompt_to_return is not None: prompt_started_payload["prompt"] = prompt_to_return
        yield InferenceResponse.model_construct(**prompt_started_payload)

        preserve_eos_for_parsing = False
        if not skip_tool_parsing:
            profile_dict = engine.state.tool_parser_profile
            if profile_dict:
                preserve_eos_for_parsing = bool(profile_dict.get("preserve_eos_for_parsing"))

        target_device = first_module_device(inference_model)
        eos_strings = _get_eos_strings(
            request_tokenizer,
            getattr(generation_config, "eos_token_id", None),
        )
        eos_strip_logged = False
        eos_strip_enabled = False
        eos_tail = ""
        max_eos_len = max((len(s) for s in eos_strings), default=0)
        cpu_inputs = request_tokenizer(
            [first_prompt], return_tensors="pt", padding=False, pad_to_multiple_of=64
        )
        # make H2D copies ride the request stream, not the default
        inputs = to_device_on_stream(cpu_inputs, target_device, req_stream)

        item_input_tokens = prompt_token_counts[0]
        streamer = StoringTextIteratorStreamer(
            engine.state.logger,
            request_tokenizer,
            drop_eos_and_pad=not preserve_eos_for_parsing,
            skip_prompt=True,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
            extra_stop_ids=getattr(generation_config, "eos_token_id", None),
        )

        def _filter_stream_text(text: str) -> str:
            nonlocal eos_tail, eos_strip_logged, eos_strip_enabled
            if not text:
                return text
            if not eos_strip_enabled and streamer.eos_token_seen:
                eos_strip_enabled = True
            if not eos_strip_enabled or not eos_strings:
                return text
            combined = eos_tail + text
            cleaned = _strip_eos_strings_from_end(combined, eos_strings)
            if not eos_strip_logged and cleaned != combined:
                engine.state.logger.debug(
                    "Stripped EOS strings from streaming chunk: %s",
                    eos_strings,
                )
                eos_strip_logged = True
            if max_eos_len <= 1:
                eos_tail = ""
                return cleaned
            keep = max_eos_len - 1
            if len(cleaned) <= keep:
                eos_tail = cleaned
                return ""
            eos_tail = cleaned[-keep:]
            return cleaned[:-keep]
        
        #assert isinstance(request_tokenizer.eos_token_id, (int, list))
        #engine.state.logger.debug(f"tok eos: {request_tokenizer.eos_token_id}, pad: {request_tokenizer.pad_token_id}")
        #engine.state.logger.debug(f"mod eos: {inference_model.generation_config.eos_token_id}, pad: {inference_model.generation_config.pad_token_id}")
        #engine.state.logger.debug(f"gen eos: {generation_config.eos_token_id}, pad: {generation_config.pad_token_id}")

        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "generation_config": generation_config,
        }
        if req_stream:
            gen_kwargs["mp13_cuda_stream"] = req_stream

        gen_kwargs['mp13_request'] = request

        adapter_combo_for_cache = [first_adapter] if first_adapter != "__base__" else []
        # static KV cache (opt-in) for streaming item
        max_new_from_config = getattr(generation_config, "max_new_tokens", None)
        gen_kwargs["generation_config"] = generation_config

        deferred_for_this_stream_item = []
        cache_mode, cache_bucket, eff_max_new = maybe_apply_static_cache(
            engine.state, inference_model, gen_kwargs, batch_size=1,
            prompt_len=item_input_tokens, max_new_tokens=max_new_from_config,
            request=request,
            active_adapters_for_request=adapter_combo_for_cache,
            deferred_slots_accumulator=deferred_for_this_stream_item
        )
        if deferred_for_this_stream_item:
            deferred_slots_accumulator.extend(deferred_for_this_stream_item)

        using_static = cache_mode == "static"
        if using_static and eff_max_new is not None and eff_max_new != max_new_from_config:
            gen_kwargs["generation_config"] = get_modified_generation_config(
                generation_config, max_new_tokens=eff_max_new)
            # Update the criteria again if static cache changed the budget.
            cancel_criteria.set_max_new_tokens(eff_max_new)

        cache_metric_str = ""
        if cache_mode == "static" and cache_bucket:
            cache_metric_str = f"static (B={cache_bucket[0]}, L={cache_bucket[1]})"
        elif cache_mode == "dynamic":
            total_len = item_input_tokens + (eff_max_new if eff_max_new is not None else max_new_from_config or 0)
            cache_metric_str = f"dynamic (L={total_len})"
        elif cache_mode == "offloaded":
            total_len = item_input_tokens + (eff_max_new if eff_max_new is not None else max_new_from_config or 0)
            cache_metric_str = f"offloaded (L={total_len})"
        # END cache

        if pass_adapters_to_generate:
            if (hasattr(inference_model, "set_adapter") and hasattr(inference_model, "disable_adapter_layers")):
                gen_kwargs["adapter_names"] = [first_adapter]
            else:
                 raise InferenceRequestError(f"(BUG) the base model cannot be used for a batch containing  adapters: {first_adapter}.")

        # Dummy confirmation event?
        confirmation_event = asyncio.Event() 
        
        stream_gen_start_time = time.monotonic()
        # Use the robust wrapper for the thread target
        # Package the function and all its arguments for the executor, which is non-blocking.
        background_task_func = functools.partial(
            _generation_thread_wrapper,
            model=inference_model,
            engine_state=engine.state,
            streamer=streamer,
            cancel_criteria=cancel_criteria, confirmation_event=confirmation_event, 
            loop=loop,
            **gen_kwargs
        )
        # Run the blocking generation function in an executor. This returns a Future.
        generation_future = loop.run_in_executor(engine.state._gen_exec, background_task_func)

        first_token_received = False
        produced_text_chunks = False
        ttft_sec = None
        full_response_text_for_item = ""  # Accumulates only the *text* part of the response

        # --- Producer/Consumer for streamer tokens (reused from original) ---
        def _producer():
            try:
                for tok in streamer: # This loop blocks until streamer.end() is called
                    if loop.is_running() and not loop.is_closed():
                        loop.call_soon_threadsafe(q.put_nowait, tok)
                    else:
                        engine.state.logger.info("Producer thread: Event loop is closed, stopping.")
                        break # Exit if loop is closed
            finally:
                if loop.is_running() and not loop.is_closed():
                    loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

        q: asyncio.Queue[Any] = asyncio.Queue()
        threading.Thread(target=_producer, daemon=True).start()

        # This request's specific async cancel event, polled from the thread-safe one
        # This is now the primary cancellation signal for the stream consumer.
        cancel_event_async = asyncio.Event() 
        
        parser: Optional[UnifiedToolIO] = None
        parser_profile: Optional[ParserProfile] = None
        if not skip_tool_parsing:
            profile_dict = engine.state.tool_parser_profile
            if not profile_dict:
                raise EngineError("Tool parser profile is not available in the engine state.")
            parser_profile = ParserProfile(**profile_dict)
            parser = UnifiedToolIO(profile=parser_profile)

        parse_iterator: AsyncIterator[str]
        if parser:
            parse_iterator = parser.parse_stream(
                engine.state.logger,
                _consume_queue_with_cancel(q, cancel_event_async),
                streamer=streamer,
                mark_incomplete_on_stream_end=True,
            )
        else:
            parse_iterator = _consume_queue_with_cancel(q, cancel_event_async)

        async for text_chunk in parse_iterator:
            text_chunk = str(text_chunk)
            text_chunk = _filter_stream_text(text_chunk)
            if not text_chunk:
                continue
            produced_text_chunks = True
            if not first_token_received:
                ttft_sec = time.monotonic() - stream_gen_start_time
                first_token_received = True
            
            yield InferenceResponse.model_construct(chunkType=ChunkType.STREAMING_CHUNK, prompt_index=0, chunk_text=text_chunk, is_final_chunk=False)
            full_response_text_for_item += text_chunk
            
            # Poll the thread-safe cancel event and set the async one if needed
            if not cancel_event_async.is_set() and cancel_criteria._cancel_event.is_set():
                cancel_event_async.set()

        # --- Finalize the parser to handle any leftover buffer ---
        # This is crucial for streams ending on a hard stop token that isn't yielded.
        if parser and (final_text_chunk := parser.finalize(tokenizer=request_tokenizer)):
            final_text_chunk = str(final_text_chunk)
            final_text_chunk = _filter_stream_text(final_text_chunk)
            if final_text_chunk:
                produced_text_chunks = True
                yield InferenceResponse.model_construct(chunkType=ChunkType.STREAMING_CHUNK, prompt_index=0, chunk_text=final_text_chunk, is_final_chunk=False)
                full_response_text_for_item += final_text_chunk

        if eos_tail:
            if eos_strings and eos_strip_enabled:
                tail_text = _strip_eos_strings_from_end(eos_tail, eos_strings)
            else:
                tail_text = eos_tail
            eos_tail = ""
            if tail_text:
                produced_text_chunks = True
                yield InferenceResponse.model_construct(chunkType=ChunkType.STREAMING_CHUNK, prompt_index=0, chunk_text=tail_text, is_final_chunk=False)
                full_response_text_for_item += tail_text

        # After the consumer loop, we must await the background task to ensure it's finished
        # before proceeding, and to catch any exceptions from it.
        try:
            await generation_future
        except (asyncio.CancelledError, KeyboardInterrupt):
            # This is the crucial part for handling Ctrl+C.
            # The await is cancelled, so we don't block.
            # We also try to cancel the future itself, though this may not stop the thread.
            # The cancel_event is what actually stops the thread.
            if not generation_future.done():
                generation_future.cancel()
            engine.state.logger.info("Streaming generation task cancelled by client.")

            # Don’t tear the loop away yet: give the gen thread a beat to hit its finally:
            #  - _run_generate_with_patches finally sets confirmation_event
            #  - _generation_thread_wrapper finally calls streamer.end()
            #try:
            #    await asyncio.wait_for(confirmation_event.wait(), timeout=3.0)
            #except asyncio.TimeoutError:
            #    pass
            # Re-raise to allow the full cancellation chain to complete.
            raise
        except BaseException as e:
            # Catch any other exceptions from the generation thread and propagate to client
            engine.state.logger.exception(f"Unhandled exception from generation thread.")
            yield InferenceResponse.model_construct(
                chunkType=ChunkType.ERROR,
                prompt_index=0,
                error=f"Inference failed: {type(e).__name__}: {e}",
                is_final_chunk=True,
                full_traceback=traceback.format_exc())
            return # Stop the generator

        # After the loop, get the collected blocks and convert to the expected format.
        collected_blocks = parser.get_collected_blocks() if parser else []


        stream_gen_end_time = time.monotonic()
        
        # Get EOS token(s) from the config used for generation
        eos_token_id = gen_kwargs["generation_config"].eos_token_id
        eos_token_ids = set()
        if isinstance(eos_token_id, int): eos_token_ids.add(eos_token_id)
        elif isinstance(eos_token_id, list): eos_token_ids.update(eos_token_id)

        # Mark static cache slot as successfully used for the streamed item
        if using_static and not (cancel_event_async.is_set() or cancel_criteria.cancellation_triggered):
            pkv_to_mark = gen_kwargs.get("past_key_values")
            if pkv_to_mark and isinstance(pkv_to_mark, StaticCache):
                pkv_to_mark._is_successfully_used = True
                # engine.state.logger.debug(f"[cache-route] Marked static slot B={pkv_to_mark.max_batch_size}, L={pkv_to_mark.max_cache_len} as successfully used (streamed).")


        num_text_tokens = len(streamer.generated_ids)
        has_visible_text = produced_text_chunks
        
        first_pass_budget = getattr(gen_kwargs["generation_config"], "max_new_tokens", 0)
        # --- Truncation Detection for Streamed Item ---
        # Streaming is always batch size 1. If the token limit was hit, it was truncated.
        was_truncated = False
        if cancel_criteria.max_tokens_triggered:
            # Since streaming is always batch size 1, max_tokens_triggered is the ultimate truth.
            # The hybrid check (inspecting the last token) is only needed for B > 1,
            # where one sequence might hit EOS while another hits the token limit.
            was_truncated = True

        # --- Tool Block Metrics (from accumulated raw blocks) ---        
        num_tool_blocks = 0
        num_tool_block_tokens = 0
        final_blocks_payload = collected_blocks if collected_blocks else None

        if final_blocks_payload:
            num_tool_blocks = len(collected_blocks)
            for tool_block in final_blocks_payload:
                try:
                    # Count tokens in each tool call block
                    num_tool_block_tokens += len(request_tokenizer(tool_block.raw_block, add_special_tokens=False).input_ids)
                    # The number of calls is just the length of the list.
                    # We could add more validation here if needed.
                except Exception as e:
                    engine.state.logger.warning(f"Could not count tokens for tool call content. Error: {e}. Content: {tool_block.raw_block[:100]}...")
        
        if was_truncated and not has_visible_text and not final_blocks_payload:
            was_truncated = False

        num_tokens_this_stream = num_text_tokens # Tool block tokens are part of the main generation
        # --- End of Token Counting ---

        item_gen_duration = stream_gen_end_time - stream_gen_start_time
        micro_batch_durations.append(item_gen_duration)
        item_tps = num_tokens_this_stream / item_gen_duration if item_gen_duration > 0 else 0

        cache_warming_str: Optional[str] = None
        if deferred_for_this_stream_item:
            slot = deferred_for_this_stream_item[0]
            cache_warming_str = f"(B={slot[0][0]},L={slot[0][1]})"

        # Include a per-item cancellation flag so clients can tell if this streamed item
        # was ended because of cancellation rather than normal completion.
        stream_was_canceled = (cancel_event_async.is_set() or cancel_criteria.cancellation_triggered)
        canceled_after_output = stream_was_canceled and has_visible_text
        effective_was_truncated = was_truncated or canceled_after_output
        response_text_for_item = full_response_text_for_item
        if final_blocks_payload and parser_profile:
            response_text_for_item = ToolsParserHelper.reconstruct_text_with_blocks(
                full_response_text_for_item,
                final_blocks_payload,
                profile=parser_profile,
            )
            if effective_was_truncated:
                num_tool_blocks = 0
                num_tool_block_tokens = 0
        if eos_strings and response_text_for_item and eos_strip_enabled:
            original_response_text_for_item = response_text_for_item
            response_text_for_item = _strip_eos_strings_from_end(response_text_for_item, eos_strings)
            if not eos_strip_logged and response_text_for_item != original_response_text_for_item:
                engine.state.logger.debug(
                    "Stripped EOS strings from streaming response_text: %s",
                    eos_strings,
                )
                eos_strip_logged = True
        raw_stream_metrics: Dict[str, Any] = {
            "input_tokens": item_input_tokens,
            "output_tokens": num_tokens_this_stream,
            "generation_duration_sec": item_gen_duration,
            "tokens_per_second": item_tps,
            "time_to_first_token_sec": ttft_sec,
            "cache_metric": cache_metric_str,
        }

        if effective_was_truncated:
            raw_stream_metrics["was_truncated"] = True
        if  stream_was_canceled:
            raw_stream_metrics["was_canceled"] = stream_was_canceled
        if cache_warming_str:
            raw_stream_metrics["cache_warming"] = cache_warming_str
        if num_tool_blocks:
            raw_stream_metrics["tool_blocks_count"] = num_tool_blocks if num_tool_blocks > 0 else None
            raw_stream_metrics["tool_blocks_tokens"] = num_tool_block_tokens if num_tool_block_tokens > 0 else None

        rounded_stream = round_floats(raw_stream_metrics)
        # Apply same filtering policy as non-streaming: include boolean flags only when True,
        # drop None/empty values. Note: _yield_streamed_response_in_chunks only applies
        # item_metrics to the final chunk, so this ensures flags are only present on final chunk.
        item_metrics: Dict[str, Any] = {}
        for k, v in rounded_stream.items():
            if k in ("was_truncated", "was_canceled"):
                if v is True:
                    item_metrics[k] = v
                continue
            if v is None:
                continue
            if isinstance(v, str) and v == "":
                continue
            if isinstance(v, (list, dict)) and len(v) == 0:
                continue
            item_metrics[k] = v

        final_item_data: Dict[str, Any] = {
            "chunkType": ChunkType.STREAMING_CHUNK,
            "prompt_index": 0,
            "chunk_text": "",
            "is_final_chunk": True,
            **item_metrics
        }
        if not request.suppress_full_response:
            final_item_data["response_text"] = response_text_for_item
        if final_blocks_payload:
            final_item_data["tool_blocks"] = final_blocks_payload

        should_mark_error = canceled_after_output
        if should_mark_error:
            final_item_data["chunkType"] = ChunkType.ERROR
            final_item_data.setdefault("error", "Cancelled before completion")

        final_stream_item = InferenceResponse.model_construct(**final_item_data)
        all_final_items_for_metrics.append(final_stream_item)

        yield final_stream_item

        # 2. Process remaining items in a non-streaming batch (simulate chunking for streaming scenario)
        remaining_prompts = prompts_to_process[1:]
        if remaining_prompts:
            remaining_adapters = adapter_names_to_use[1:]
            async for item in _generate_non_streamed_batch_internal(
                engine, request, remaining_prompts, 
                remaining_adapters, 
                pass_adapters_to_generate,
                generation_config, 
                inference_model, 
                request_tokenizer, 
                cancel_event_threadsafe,
                prompt_token_counts[1:],
                deferred_slots_accumulator, 
                streaming=True,
                skip_tool_parsing=skip_tool_parsing,
                start_index=1,
                batch_durations_collector=micro_batch_durations,
                req_stream_for_batch=req_stream,
            ):
                if item.is_final_chunk:
                    all_final_items_for_metrics.append(item)
                yield item
    else:
        # --- NON-STREAMING PATH: Batch everything ---
        async for item in _generate_non_streamed_batch_internal(
            engine, request, prompts_to_process, 
            adapter_names_to_use,
            pass_adapters_to_generate,
            generation_config,
            inference_model, 
            request_tokenizer,
            cancel_event_threadsafe,
            prompt_token_counts,
            deferred_slots_accumulator, 
            streaming=False,
            skip_tool_parsing=skip_tool_parsing,
            start_index=0,
            batch_durations_collector=micro_batch_durations,
            req_stream_for_batch=req_stream
        ):
            all_final_items_for_metrics.append(item)
            yield item

    # --- After all items are generated, yield a final chunk with aggregate metrics ---
    total_input_tokens = sum(i.input_tokens or 0 for i in all_final_items_for_metrics if i.input_tokens is not None)
    total_output_tokens = sum(i.output_tokens or 0 for i in all_final_items_for_metrics if i.output_tokens is not None)
    total_gen_duration = sum(micro_batch_durations)
    overall_tps = (total_output_tokens / total_gen_duration) if total_gen_duration > 0 else 0.0
    ttfts = [i.time_to_first_token_sec for i in all_final_items_for_metrics if i.time_to_first_token_sec is not None]
    avg_ttft = (sum(ttfts) / len(ttfts)) if ttfts else None
    total_tool_blocks = sum(i.tool_blocks_count or 0 for i in all_final_items_for_metrics if i.tool_blocks_count is not None)
    total_tool_blocks_tokens = sum(i.tool_blocks_tokens or 0 for i in all_final_items_for_metrics if i.tool_blocks_tokens is not None)
    had_prompt_failure = any(
        (item.error is not None)
        or (item.chunkType == ChunkType.ERROR)
        or bool(item.was_canceled)
        for item in all_final_items_for_metrics
    )

    cache_warming_str = None
    if deferred_slots_accumulator:
        slot_strings = [f"(B:{B},L:{L})" for (B, L), _ in deferred_slots_accumulator]
        cache_warming_str = ", ".join(slot_strings)

    cache_queued_str: Optional[str] = None
    # Best-effort read of state for metrics. This is not under a lock, but should be safe
    # for read-only display at the end of a request.
    with contextlib.suppress(Exception):
        queued_slots_for_display = []
        if engine.state._active_signature:
            (B, L), _ = engine.state._active_signature
            queued_slots_for_display.append(f"(B={B},L={L})*") # Star indicates active

        if engine.state._pending_warmup_queue:
            for sig in engine.state._pending_warmup_queue:
                (B, L), _ = sig
                queued_slots_for_display.append(f"(B={B},L={L})")
        if queued_slots_for_display:
            cache_queued_str = ", ".join(queued_slots_for_display)

    in_flight_req_count = engine.adapters_control.get_active_and_pending_requests_count()[0] - 1 # current request is still there

    engine.state._update_gpu_memory_stats() # Ensure stats are fresh
    current_mem_alloc = engine.state.current_gpu_mem_allocated_mb
    current_mem_rsvd = engine.state.current_gpu_mem_reserved_mb
    cache_queued_value = cache_queued_str or ""
    mem_alloc_value = current_mem_alloc if current_mem_alloc is not None else 0.0
    mem_rsvd_value = current_mem_rsvd if current_mem_rsvd is not None else 0.0

    final_metrics_item = InferenceResponse.model_construct(
        chunkType=ChunkType.STREAMING_ENDED,
        is_final_chunk=True,  # This is a special "meta" chunk
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_generation_duration_sec=round_floats(total_gen_duration),
        overall_tps=round_floats(overall_tps),
        avg_time_to_first_token_sec=round_floats(avg_ttft),
        total_tool_blocks=total_tool_blocks or None,
        total_tool_blocks_tokens=total_tool_blocks_tokens or None,
        cache_warming=cache_warming_str,
        cache_queued=cache_queued_value,
        had_error=had_prompt_failure,
        in_flight_req=in_flight_req_count,
        mem_allocated=round_floats(mem_alloc_value),
        mem_reserved=round_floats(mem_rsvd_value),
    )

    # --- Update engine-level aggregate metrics ---
    if hasattr(engine.state, 'aggregate_metrics'):
        was_successful = not had_prompt_failure
        async with engine.state._aggregate_metrics_lock:
            # Record the timestamp of the last completed request for throughput calculation.
            engine.state.aggregate_metrics.last_request_end_time_mono = time.monotonic()

            metrics_obj = engine.state.aggregate_metrics
            metrics_obj.total_requests += 1
            # Check if request was successful (no error in any item)
            if was_successful:
                metrics_obj.total_successful_requests += 1
                if final_metrics_item.total_input_tokens:
                    metrics_obj.total_input_tokens += final_metrics_item.total_input_tokens
                if final_metrics_item.total_output_tokens:
                    metrics_obj.total_output_tokens += final_metrics_item.total_output_tokens
                if final_metrics_item.total_generation_duration_sec:
                    metrics_obj.total_generation_duration_sec += final_metrics_item.total_generation_duration_sec
                if final_metrics_item.total_tool_blocks:
                    metrics_obj.total_tool_blocks += final_metrics_item.total_tool_blocks
                if final_metrics_item.total_tool_blocks_tokens:
                    metrics_obj.total_tool_blocks_tokens += final_metrics_item.total_tool_blocks_tokens
            else:
                metrics_obj.total_failed_requests += 1

            if final_metrics_item.mem_allocated:
                metrics_obj.mem_allocated=final_metrics_item.mem_allocated
            if final_metrics_item.mem_reserved:
                metrics_obj.mem_reserved=final_metrics_item.mem_reserved

            # Create a dedicated history item instead of modifying the response item.
            end_time_mono = time.monotonic()
            end_time_wall = time.time()
            history_item = InferenceMetricsHistoryItem(
                request_id=request.request_id,
                start_time_mono=start_time_mono,
                end_time_mono=end_time_mono,
                end_time_wall=end_time_wall,
                total_input_tokens=final_metrics_item.total_input_tokens,
                total_output_tokens=final_metrics_item.total_output_tokens,
                total_generation_duration_sec=final_metrics_item.total_generation_duration_sec,
                avg_time_to_first_token_sec=final_metrics_item.avg_time_to_first_token_sec,
                was_truncated=any(item.was_truncated for item in all_final_items_for_metrics if item.was_truncated is not None),
                mem_allocated=final_metrics_item.mem_allocated,
                mem_reserved=final_metrics_item.mem_reserved,
                total_tool_blocks=final_metrics_item.total_tool_blocks,
                total_tool_blocks_tokens=final_metrics_item.total_tool_blocks_tokens,
            )
            # --- Add to state history ---
            engine.state._inference_metrics_history.append(history_item)

    yield final_metrics_item

async def _consume_queue_with_cancel(q: asyncio.Queue, cancel_event: asyncio.Event):
    """Helper async generator to consume from a queue until sentinel or cancellation."""
    while True:
        token_task = asyncio.create_task(q.get())
        cancel_task = asyncio.create_task(cancel_event.wait())
        done, pending = await asyncio.wait({token_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)

        if cancel_task in done:
            token_task.cancel() # Cancel the pending q.get()
            break

        # This is the crucial fix: cancel the waiter task if the token task finished.
        if cancel_task in pending:
            cancel_task.cancel()

        token = token_task.result()
        if token is _SENTINEL: break
        yield token

async def run_inference_logic(engine: "MP13Engine", request: InferenceRequest) -> AsyncIterator[InferenceResponse]:
    """
    Main entry point for running inference. This is an async generator that wraps the
    internal generator to provide robust state management.

    """
    if request.reset_metrics:
        # This is a good place to clear history as it's at the very start of the request logic.
        await engine.state.reset_aggregate_metrics()
        engine.state.logger.info("Inference metrics history and aggregates cleared due to request flag.")

    stream_ended_sent = False
    is_cancelled = False
    error_sent = False
    deferred_slots_accumulator: List[Tuple[Tuple[int, int], frozenset[str]]] = []

    adapters_control = engine.adapters_control
    req_set = tuple()
    try:
        # --- Cohort key selection (ordered fairness) ---
        # 1. If `adapter_names` is provided, it's a fan-out request. This is an exclusive "NF" (Non-Fast-path) cohort.
        # 2. If `active_adapters` is an empty list `[]`, use the base model cohort: ("F", tuple()).
        # 3. If `active_adapters` is `None`, use the globally active adapter set.
        # 4. If `active_adapters` is a list of names, use that specific cohort.
        has_fanout = bool(request.override_adapters)
        if has_fanout:
            # Non-fast path (adapter fan-out): cohort = ("NF", mixed_signature), exclusive
            # signature for non-fast path = sorted unique names passed in this request
            ad = request.override_adapters
            if isinstance(ad, str): ad = [ad]
            mixed_sig = adapters_control.get_normalized_adapter_set(ad)
            cohort_key = ("NF", mixed_sig)
        else:
            active_adapters_req = getattr(request, "active_adapters", None)
            if active_adapters_req is None:
                # Use the globally active set if the request doesn't specify.
                req_set = tuple(sorted(await engine.state.active_adapter_names()))
            else:
                req_set = adapters_control.get_normalized_adapter_set(active_adapters_req)
            cohort_key = ("F", req_set)

        # Pass the request to cohort_enter to handle request_id generation and uniqueness.
        # cohort_enter returns a threadsafe cancel event for user requests.
        cancel_event = await adapters_control.cohort_enter(cohort_key, request)

        # --- Check if the request was cancelled while pending in the cohort queue ---
        if cancel_event is not None and cancel_event.is_set():
            engine.state.logger.info(f"Request '{request.request_id}' was cancelled while pending. Terminating.")
            # Yield an error chunk and exit the generator immediately.
            yield InferenceResponse.model_construct(chunkType=ChunkType.ERROR, prompt_index=0, error="Cancelled by user while pending in queue", was_canceled=True, was_truncated=True, is_final_chunk=True)
            # The `finally` block will handle cohort_leave and state cleanup.
            return

        # --------- One-time switch for fast-path cohorts (no restore) ----------
        # Make the model's adapters match the cohort; leave them set for subsequent requests
        if cohort_key[0] == "F":
            async with engine.state._cohort_lock: # type: ignore
                # compute current active set from the model (not just cached)
                current = adapters_control.get_current_active_set_unlocked()
                if current != req_set:
                    engine.state.logger.debug(f"Switching model active adapters from {current} to {req_set}")
                    adapters_control.one_time_switch_to_set(req_set)

        # The core logic is now wrapped and iterated over here.
        # The original `run_inference_logic` has been converted into this generator.
        # Pass the cohort-provided threadsafe cancel event into the core generator so
        # AdaptersControl manages active mapping internally (no external registration needed).
        async for item in _core_inference_generator(engine, request, cancel_event, deferred_slots_accumulator):
            if item.chunkType == ChunkType.ERROR:
                if error_sent:
                    continue
                error_sent = True
            if item.chunkType == ChunkType.STREAMING_ENDED:
                stream_ended_sent = True
            yield item
    except (GeneratorExit, asyncio.CancelledError, InferenceCancelledError):
        # If the generator is closed after the final chunk has been sent,
        # it's a normal closure, not a cancellation.
        if not stream_ended_sent:
            is_cancelled = True
        # Re-raise to allow the runtime to handle the generator closure.
        raise
    except Exception:
        # Any other exception is considered a failure/cancellation.
        is_cancelled = True
        # Re-raise so the API layer can see the error. The internal generator
        # should have already yielded an error chunk.
        raise
    finally:
        # Leave cohort at the end (we do not restore adapters for fast path)
        try:
            await adapters_control.cohort_leave(request)
        except Exception:
            pass
        
        if request.request_id:
            await engine.state.set_inference_complete(request.request_id, cancelled=is_cancelled)

        # Launch the warmup task in the background if needed, but not if the engine is shutting down or in a forced mode switch.
        # We use a lock and re-check to prevent a race condition with the shutdown signal.
        if deferred_slots_accumulator:
            async with engine.state._lock:
                # Check the authoritative server status under lock to prevent race conditions.
                is_shutting_down = engine.state.server_status == ServerStatus.SHUTTING_DOWN
                # The flag is for forced mode switches, which don't change server status.
                prevent_new_tasks = getattr(engine.state, '_prevent_new_background_tasks', False)

                if is_shutting_down or prevent_new_tasks:
                    engine.state.logger.info(f"Skipping cache warmup: engine shutting down ({is_shutting_down}) or new tasks prevented ({prevent_new_tasks}).")
                else:
                    for slot in deferred_slots_accumulator:
                        asyncio.create_task(queue_static_cache_warmup(engine, slot))


async def _core_inference_generator(engine, request: InferenceRequest, cohort_cancel_event: threading.Event, deferred_slots_accumulator: List[Tuple[Tuple[int, int], frozenset[str]]]) -> AsyncIterator[InferenceResponse]:
    """
    The main body of the inference logic, refactored into its own async generator.
    This function contains the logic previously in `run_inference_logic`.
    """
    start_time_wall = time.time()
    start_time_mono = time.monotonic()
    response_notes: list[str] = []
    was_cancelled = False
    total_tokens_generated_for_response = 0
    # A single resource object holds the tokenizer and stream for the request.
    resource: Optional["RequestResource"] = None # type: ignore

    # --- DEBUG: log immediately on entry (no await yet) ---
    #engine.state.logger.debug(f"[DBG] _core_inference_generator ENTER (req={request.request_id or 'N/A'})")

    async def _error_iterator(
        msg: str,
        exc: BaseException | None = None,
        *,
        prompt_index: Optional[int] = None,
        full_traceback: Optional[str] = None
    ):
        # Best-effort state + stats
        try:
            if engine.state.inference_status != InferenceStatus.ERROR:
                await engine.state.set_inference_error(msg)
        except Exception:
            pass

        engine.state._update_gpu_memory_stats()
        
        yield InferenceResponse.model_construct(
            chunkType=ChunkType.ERROR,
            error=msg, is_final_chunk=True,
            prompt_index=prompt_index,
            full_traceback=full_traceback if full_traceback is not None else (traceback.format_exc() if exc else None)
        )

    try:
        resource = await engine.checkout_resource(request.request_id)
        request_tokenizer = resource.tokenizer
        req_stream = resource.stream
        response_notes.append("Using request-local resource (tokenizer/stream) from pool.")
        if req_stream:
            response_notes.append("Using request-local CUDA stream from pool.")

        if engine.state.engine_mode != EngineModeState.INFERENCE:
            raise ModeMismatchError(f"Cannot run inference. engine_mode='{engine.state.engine_mode.name if engine.state.engine_mode else 'UNSET'}', expected INFERENCE.")

        # Use the per-request cancel event provided by cohort_enter (cohort_cancel_event).
        # If None, create a local dummy Event: this makes internal/system requests
        # non-cancellable by external APIs but avoids errors in the generation path.
        if cohort_cancel_event is not None:
            cancel_event_threadsafe = cohort_cancel_event
        else:
            raise EngineError("(internal) Cancel event was not registered.")

        await engine.state.set_inferring(request.request_id)
        engine.state._update_gpu_memory_stats()

        # NOTE: The guard that previously blocked inference during cache warming has been removed.
        # Concurrency is now managed by fine-grained locks on adapter mutation and dedicated
        # CUDA streams for the background warmup process, allowing inference to proceed.
        # Allow inference to start if the engine is READY or if it's warming the cache in the background.

        if engine.state.base_model is None or request_tokenizer is None:
            raise EngineError("Engine not initialized: missing base model and/or tokenizer.")

        # --- Determine the effective list of adapter names for the batch ---
        # This list will be used to determine the inference model and for fan-out.
        # This flag indicates if `adapter_names` should be passed to `model.generate()`.
        # It's true if `request.override_adapters` was explicitly provided (an override).
        # It's false if we're relying on the engine's active adapters.
        # Replace empty/None with the special '__base__' name
        # This list is needed for cache routing even if not passed to generate().
        # --- Handle Fan-Out (duplicating the single input prompt/message list) ---
        # Only perform fan-out if adapter_names were explicitly provided in the request (i.e., it's an override)
        # and there's a single input prompt/message. If multiple prompts are provided, the number of prompts
        # must exactly match the number of adapter_names, which is handled by the model's generate method.

        pass_adapters_to_generate = False
        effective_adapter_names_for_batch: List[str] = []

        if request.override_adapters:
            loaded = await engine.state.get_all_adapter_names_in_model()
            for name in request.override_adapters:
                if not name or name == "__base__":
                    effective_adapter_names_for_batch.append("__base__")
                elif name in loaded:
                    effective_adapter_names_for_batch.append(name)
                else:
                    raise AdapterError(f"Adapter '{name}' is not loaded. Loaded: {loaded}")
            pass_adapters_to_generate = True
            response_notes.append(f"Using request adapters: {effective_adapter_names_for_batch}")
        # No override, use the engine's currently active adapters.
        else:
            effective_adapter_names_for_batch = await engine.state.active_adapter_names()
            response_notes.append(f"Using engine active adapters: {effective_adapter_names_for_batch}")

        num_input_raw = len(request.raw_list) if request.raw_list else 0
        num_input_msgs = len(request.messages_list) if request.messages_list else 0
        if pass_adapters_to_generate and (num_input_raw == 1 or num_input_msgs == 1) and effective_adapter_names_for_batch:
            response_notes.append(f"Fan-out: duplicating single input x{len(effective_adapter_names_for_batch)}")
            if num_input_raw and num_input_msgs:
                raise ValueError("Both raw_list and messages_list present.")
            if request.raw_list:
                request.raw_list = request.raw_list * len(effective_adapter_names_for_batch)
            elif request.messages_list:
                request.messages_list = request.messages_list * len(effective_adapter_names_for_batch)

        # Delegate to the shared prompt preparation function
        prompts_to_process, prompt_token_counts, format_errors, format_error_details = await _prepare_prompts_for_inference(engine, request, request_tokenizer)

        if format_errors:
            if format_error_details:
                indices = [d.get("prompt_index") for d in format_error_details]
                index_list = ", ".join(str(i) for i in indices if i is not None)
                header = f"Prompt format failed for {len(format_error_details)} prompt(s)"
                if index_list:
                    header += f" (indices: {index_list})"
                combined_errors = "\n".join(d.get("error", "") for d in format_error_details if d.get("error"))
                first_traceback = None
                for detail in format_error_details:
                    tb = detail.get("full_traceback")
                    if tb:
                        first_traceback = tb
                        break
                msg = f"{header}: {combined_errors}" if combined_errors else header
                async for x in _error_iterator(
                    msg,
                    prompt_index=format_error_details[0].get("prompt_index"),
                    full_traceback=first_traceback
                ):
                    yield x
            else:
                async for x in _error_iterator(f"Prompt format failed: {format_errors[0]}"):
                    yield x
            return

        inference_model = engine.state.peft_model

        if inference_model is None:
            raise EngineError("Could not resolve an inference model for this request.")

        try:
            internal_iterator = _internal_response_generator(
                start_time_mono=start_time_mono,
                engine=engine, request=request, prompts_to_process=prompts_to_process,
                adapter_names_to_use=effective_adapter_names_for_batch,
                pass_adapters_to_generate=pass_adapters_to_generate,
                start_time_wall=start_time_wall, prompt_token_counts=prompt_token_counts, 
                inference_model=inference_model,
                request_tokenizer=request_tokenizer, cancel_event_threadsafe=cancel_event_threadsafe,
                deferred_slots_accumulator=deferred_slots_accumulator,
                req_stream=req_stream,
            )
            async for item in internal_iterator:
                if item.is_final_chunk and item.total_output_tokens is not None:
                    total_tokens_generated_for_response = item.total_output_tokens
                yield item
        except (KeyboardInterrupt, asyncio.CancelledError):
            was_cancelled = True
            raise
        except (InferenceCancelledError):
            yield InferenceResponse.model_construct(chunkType=ChunkType.ERROR, error="Inference was Canceled ", was_canceled=True, was_truncated=True, is_final_chunk=True)
        except (AdapterError, InferenceRequestError, EngineError, ModeMismatchError) as e_known:
            yield InferenceResponse.model_construct(chunkType=ChunkType.ERROR, error=str(e_known), is_final_chunk=True) # type: ignore
        except Exception as e:
            err = f"Unexpected inference error: {type(e).__name__}: {e}"
            engine.state.logger.error(err, exc_info=True)
            await engine.state.set_inference_error(err)
            yield InferenceResponse.model_construct(chunkType=ChunkType.ERROR, prompt=None, is_final_chunk=True, error=err)

    except (AdapterError, InferenceRequestError, EngineError, ModeMismatchError) as e:
        msg = f"Inference Error: {type(e).__name__}: {e}"
        engine.state.logger.error(f"!!! {msg}")
        async for x in _error_iterator(msg, e): yield x
    except Exception as e_main:
        msg = f"Unexpected Top-Level Error: {type(e_main).__name__}: {e_main}"
        engine.state.logger.critical(msg, exc_info=True)

        with contextlib.suppress(Exception):
            await engine.state.set_inference_error(msg, clear_config=False)

        engine.state._update_gpu_memory_stats()

        async for x in _error_iterator(msg, e_main): yield x

    finally:
        # Always check in the resource if it was checked out.
        if resource:
            await engine.checkin_resource(resource)

        async def _post_log():
            engine.state._update_gpu_memory_stats()
            engine.state.logger.debug(f"Generator lifecycle END (req={request.request_id or 'N/A'}, cancelled={was_cancelled}, tokens={total_tokens_generated_for_response})")

        asyncio.create_task(_post_log())
