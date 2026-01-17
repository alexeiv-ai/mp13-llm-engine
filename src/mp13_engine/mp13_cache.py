# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""MP13 Engine - Static KV Cache Routing and Management."""

import time
import os
import inspect
import torch, threading, asyncio, traceback, contextlib, concurrent.futures
from typing import TYPE_CHECKING, List, Dict, Any, Tuple, Optional, Callable, Set

from transformers import GenerationConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.cache_utils import StaticCache
from .mp13_utils import get_modified_generation_config, low_priority_stream_for, no_cudagraphs
from .mp13_utils import first_module_device_for_sharded_model
from . import mp13_state

if TYPE_CHECKING or not hasattr(torch.nn, 'Module'): # The second part is a simple trick to always execute this block
    from .mp13_state import MP13State
    from .mp13_config import InferenceRequest
    from .mp13_engine import MP13Engine
    from .mp13_adapter import CohortTaskType

def reset_static_cache(engine_state: "MP13State") -> None:
    """Invalidates and frees all static cache slots and clears warmed up trackers."""
    if _STATIC_POOLS:
        engine_state.logger.info("[cache] Resetting all static cache slots and warmed signatures.")
        # Iterate over a copy of keys since we are modifying the dict
        for model_id in list(_STATIC_POOLS.keys()):
            pool = _STATIC_POOLS.pop(model_id)
            pool._clear()
        _STATIC_POOLS.clear()
    _STATIC_POOL_DEVICE_BLOCKLIST.clear()

def reset_compile_warm_tracker(engine_state: "MP13State") -> None:
    """Resets warmed signatures on all static cache pools. Called after a recompile."""
    if _STATIC_POOLS:
        engine_state.logger.info("[cache] Resetting all warmed adapter signatures due to recompile.")
        for pool in _STATIC_POOLS.values():
            pool._clear_warmed_signatures()

# TBD: This function was invoked form adapters but is not currenlty used
def invalidate_warmed_signatures_except_base(engine_state: "MP13State") -> None:
    """Invalidates all warmed adapter signatures, keeping only the base model signature."""
    if _STATIC_POOLS:
        engine_state.logger.info("[cache] Invalidating all warmed adapter signatures except for base model.")
        for pool in _STATIC_POOLS.values():
            pool._clear_warmed_signatures_except_base()

# --- Module-level state for managing the single background worker ---
_warmup_task_future: Optional[concurrent.futures.Future] = None
_warmup_lock = threading.Lock()  # Synchronizes access to _warmup_task_future

# --- End of module-level state ---

# Global pool for static cache objects, keyed by model ID
_STATIC_POOLS: Dict[int, "_StaticCachePool"] = {}
# Track models where static cache is disabled due to device placement risk
_STATIC_POOL_DEVICE_BLOCKLIST: Set[int] = set()

# A fixed set of "known" buckets that the engine supports for static GPU KV cache.
# This can be overridden by the session config `static_kv_known_buckets_map`.
_DEFAULT_KNOWN_BUCKETS_MAP: Dict[int, List[int]] = {
    1: [1024, 4096, 8192, 16384, 32768, 65536], 
    2: [1024, 4096, 8192, 16384, 32768],
    3: [1024, 4096, 8192, 16384],
    4: [1024, 4096, 8192, 16384],
    5: [1024, 4096, 8192],
    6: [1024, 4096, 8192],
    7: [1024, 4096],
    8: [1024, 4096],
}

# A map of bucket length to a fixed prompt padding length.
# This helps stabilize the `max_new_tokens` argument for `torch.compile`.
_DEFAULT_PROMPT_PADDING_MAP = {
    1024: 128,
    2048: 128,
    8192: 512,
    32768: 512,
    65536: 1024,
    131072: 1024,
}

def initialize_cache_session_config(engine_state: "MP13State", static_kv_cache_enabled: bool):
    """Initializes the cache-related parts of the inference session config."""
    if engine_state.current_inference_session_config is None:
        engine_state.current_inference_session_config = {}
    
    # Always set as override
    engine_state.current_inference_session_config["static_kv_cache"] = static_kv_cache_enabled
    
    # --- Static Offload Rule ---
    # Offload if (total_len * batch_size) > 49152.
    # This is implemented by setting a max_len cap for each batch size.
    # L_cap = 49152 / B
    engine_state.current_inference_session_config.setdefault("offloaded_len_cap_by_batch", {
        "1": 65536,
        "2": 49152,
        "3": 32768,
        "4": 32768,
        "5": 16384,
        "6": 16384,
        "7": 12288,
        "8": 12288,
    })
    # Dynamic clear rule. Invalidate static cache if dynamic is used and max_new_tokens > cap.
    # Values are half of the offload caps.
    engine_state.current_inference_session_config.setdefault("dynamic_clear_static_len_cap_by_batch", {
        "1": 49152,
        "2": 32768,
        "3": 32768,
        "4": 32768,
        "5": 16384,
        "6": 16384,
        "7": 8192,
        "8": 8192,
    })

    # Known GPU static buckets
    engine_state.current_inference_session_config.setdefault("static_kv_known_buckets_map", _DEFAULT_KNOWN_BUCKETS_MAP)
    
    # New: Prompt padding map for static cache stability
    engine_state.current_inference_session_config.setdefault("static_kv_prompt_padding_map", _DEFAULT_PROMPT_PADDING_MAP)

    engine_state.logger.info(f"[cache] Initialized default static cache session config (static GPU cache: {static_kv_cache_enabled}).")


class _StaticCachePool:
    def __init__(self, model, device, dtype, known_buckets_map: Dict[int, List[int]]):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.known_buckets_map = known_buckets_map
        self._pool: Dict[Tuple[int, int], StaticCache] = {}  # (B, L) -> StaticCache
        self._warmed_signatures: Dict[Tuple[int, int], Set[frozenset[str]]] = {} # (B, L) -> set of warmed adapter frozensets
        self._lock = threading.RLock() # New lock for thread-safety

    def pick_shape(self, B_needed: int, total_len_needed: int) -> Optional[Tuple[int, int]]:
        """Picks the smallest 'known' bucket that fits the request."""
        if B_needed not in self.known_buckets_map:
            return None
        
        available_L_for_B = sorted(self.known_buckets_map[B_needed])
        chosen_L = next((l for l in available_L_for_B if l >= total_len_needed), None)
        
        if chosen_L is None:
            return None
        return (B_needed, chosen_L)

    def get(self, B: int, L: int) -> StaticCache:
        """Gets a cache object from the pool, creating it if it doesn't exist."""
        key = (B, L)
        if key not in self._pool:
            # This is now the pre-allocation step during warmup.
            self._pool[key] = StaticCache(
                self.model.config, max_batch_size=B, max_cache_len=L, dtype=self.dtype, device=self.device
            )
        return self._pool[key]

    def remove(self, B: int, L: int):
        """Removes a cache object from the pool, e.g., if it's corrupted."""
        key = (B, L)
        with self._lock:
            if key in self._pool:
                del self._pool[key]
            if key in self._warmed_signatures:
                del self._warmed_signatures[key]

    def is_warmed(self, B: int, L: int, adapters_key: frozenset[str]) -> bool:
        """Checks if a specific adapter combination is warmed for a given slot."""
        with self._lock:
            return adapters_key in self._warmed_signatures.get((B, L), set())

    def mark_as_warmed(self, B: int, L: int, adapters_key: frozenset[str]):
        """Marks a specific adapter combination as warmed for a given slot."""
        with self._lock:
            if (B, L) not in self._warmed_signatures:
                self._warmed_signatures[(B, L)] = set()
            self._warmed_signatures[(B, L)].add(adapters_key)

    def _clear_warmed_signatures_except_base(self):
        """Resets all warmed signatures for this pool, keeping only the base model signature."""
        with self._lock:
            base_model_key = frozenset()
            new_warmed_signatures = {}
            for slot_key, signatures in self._warmed_signatures.items():
                if base_model_key in signatures:
                    # If base model was warmed for this slot, keep only that signature
                    new_warmed_signatures[slot_key] = {base_model_key}
            
            self._warmed_signatures = new_warmed_signatures

    def _clear_warmed_signatures(self):
        """Resets all warmed signatures for this pool."""
        with self._lock:
            self._warmed_signatures.clear()

    def _clear(self):
        """Clears all allocated cache objects from the pool."""
        if self._pool:
            # Explicitly delete to help GC
            for cache_obj in self._pool.values():
                del cache_obj
            self._pool.clear()
        self._clear_warmed_signatures()


def _static_cfg(engine_state: "MP13State") -> Tuple:
    sess = engine_state.current_inference_session_config or {}
    opt_in = bool(sess.get("static_kv_cache", False))
    
    # Use the map from config if it exists, otherwise use the hardcoded default.
    known_buckets_map_str_keys = sess.get("static_kv_known_buckets_map", _DEFAULT_KNOWN_BUCKETS_MAP)
    known_buckets_map = {int(k): v for k, v in known_buckets_map_str_keys.items()}

    padding_map_str_keys = sess.get("static_kv_prompt_padding_map", _DEFAULT_PROMPT_PADDING_MAP)
    padding_map = {int(k): int(v) for k, v in padding_map_str_keys.items()}

    offload_cap_by_batch_str_keys = dict(sess.get("offloaded_len_cap_by_batch", {}) or {})
    offload_cap_by_batch = {int(k): v for k, v in offload_cap_by_batch_str_keys.items()}

    dynamic_clear_cap_by_batch_str_keys = dict(sess.get("dynamic_clear_static_len_cap_by_batch", {}) or {})
    dynamic_clear_cap_by_batch = {int(k): v for k, v in dynamic_clear_cap_by_batch_str_keys.items()}

    return opt_in, known_buckets_map, offload_cap_by_batch, padding_map, dynamic_clear_cap_by_batch


def _normalize_device_repr(device_like: Any) -> str:
    """Convert a device spec (torch.device, str, etc.) to a normalized string."""
    try:
        return str(torch.device(device_like))
    except (TypeError, ValueError, RuntimeError):
        return str(device_like or "")


def _static_cache_is_device_safe(engine_state: "MP13State", model) -> bool:
    """
    Returns False when the model spans multiple devices/offload targets, meaning
    a single-device static cache slot would later be consumed by a forward pass
    whose parameters live elsewhere.
    """
    device_map_cfg = (engine_state.global_config or {}).get("device_map")
    if isinstance(device_map_cfg, str):
        if device_map_cfg.lower() == "auto":
            return False
    elif isinstance(device_map_cfg, dict):
        devices = {_normalize_device_repr(dev) for dev in device_map_cfg.values() if dev is not None}
        devices.discard("")
        if len(devices) > 1:
            return False

    hf_device_map = getattr(model, "hf_device_map", None) or getattr(model, "_hf_device_map", None)
    if isinstance(hf_device_map, dict):
        devices = {_normalize_device_repr(dev) for dev in hf_device_map.values() if dev is not None}
        devices.discard("")
        if len(devices) > 1:
            return False

    return True


def _static_pool_for(engine_state: "MP13State", model, create_if_missing: bool = True) -> Optional["_StaticCachePool"]:
    ident = id(model)
    opt_in, known_buckets_map, _, _, _ = _static_cfg(engine_state)
    if not opt_in:
        return None
    if ident in _STATIC_POOL_DEVICE_BLOCKLIST:
        return None
    if not _static_cache_is_device_safe(engine_state, model):
        if ident not in _STATIC_POOL_DEVICE_BLOCKLIST:
            engine_state.logger.info("[cache] Static cache disabled for this model because its parameters are spread across multiple devices/offload targets.")
            existing = _STATIC_POOLS.pop(ident, None)
            if existing:
                existing._clear()
            _STATIC_POOL_DEVICE_BLOCKLIST.add(ident)
        return None
    if ident not in _STATIC_POOLS:
        # The StaticCache object itself must be created on a concrete device.
        # For sharded models, this is typically the device of the first parameter (e.g., cuda:0).
        # The 'auto' device placement is handled by passing `cache_kwargs` to model.generate().
        try:
            p0 = next(model.parameters())
            device, dtype = p0.device, p0.dtype
        except StopIteration:
            device, dtype = torch.device("cpu"), torch.float32
        if create_if_missing:
            _STATIC_POOLS[ident] = _StaticCachePool(model, device, dtype, known_buckets_map)
        else:
            return None
    return _STATIC_POOLS[ident]

def get_effective_cache_mode(kwargs: dict) -> str:
    """Infers the cache mode from generate() kwargs for logging purposes."""
    gcfg = kwargs.get("generation_config")
    if not gcfg:
        raise ValueError("generation_config not found in generate() kwargs for logging.")
    
    if not hasattr(gcfg, "use_cache"):
        # This should ideally not happen if the config is always constructed correctly.
        raise AttributeError("generation_config in generate() kwargs must have a 'use_cache' attribute for logging.")

    if gcfg.use_cache is False:
        return "no_cache"

    if isinstance(kwargs.get("past_key_values"), StaticCache):
        return "static"
    
    if getattr(gcfg, "cache_implementation", None) == "offloaded":
        return "offloaded"
    
    # Default assumption if use_cache is True and not static/offloaded
    return "dynamic"


def route_cache_mode(
    engine_state: "MP13State", 
    model, 
    *, 
    batch_size: int, 
    prompt_len: int, 
    max_new_tokens: int, 
    active_adapters: List[str],
    max_new_tokens_is_user_specified: bool,
    cache_override: Optional[str] = None,
    allow_warmup: bool = True
) -> Tuple[str, Dict, Optional[Tuple[Tuple[int, int], frozenset[str]]]]:
    """
    Decide which cache mode to use for a request with (B, total_len).
    Returns (mode_for_this_request, info_dict, deferred_warmup_task).
    """
    # --- Global and per-request cache disabling ---
    if not engine_state.use_cache:
        return "no_cache", {}, None
    if cache_override == "no_cache":
        return "no_cache", {}, None
    
    opt_in, known_buckets_map, offload_cap_by_B, padding_map, dynamic_clear_cap_by_B = _static_cfg(engine_state)

    # 1. Handle explicit override from the request.
    # The '_reset' variants are handled by the caller (maybe_apply_static_cache).
    if cache_override == "dynamic":
        return "dynamic", {}, None
    if cache_override == "offloaded":
        return "offloaded", {}, None
    if cache_override == "static":
        if not engine_state.is_compiled:
            engine_state.logger.warning("[cache-route] Warning: 'static' cache override requested, but torch.compile is disabled. Falling back to dynamic.")
            return "dynamic", {}, None
        if not opt_in:
            engine_state.logger.warning("[cache-route] Warning: 'static' cache override requested, but the session did not opt into static cache. Continuing with normal routing.")
        # Otherwise, fall through to the static logic below so we can still consider other modes.

    # 2. Default routing logic (if no override)
    B = int(batch_size)
    adapters_key = frozenset(active_adapters)

    # 2a) Offload Rule Check
    total_len = prompt_len + max_new_tokens
    off_cap = int(offload_cap_by_B.get(B, 0)) if offload_cap_by_B else 0
    is_offload_candidate = off_cap and total_len > off_cap

    if is_offload_candidate:
        # Exception to the offload rule: if the request is also a "dynamic clear" candidate
        # AND the static cache is enabled and populated, we should prefer dynamic to clear it.
        clear_threshold = int(dynamic_clear_cap_by_B.get(B, 0)) if dynamic_clear_cap_by_B else 0
        is_dynamic_clear_candidate = clear_threshold and max_new_tokens > clear_threshold

        pool = _static_pool_for(engine_state, model, create_if_missing=False)
        should_prefer_dynamic_to_clear = (
            is_dynamic_clear_candidate and
            opt_in and # Static cache must be enabled
            pool and pool._pool # And populated
        )

        if not should_prefer_dynamic_to_clear:
            # This is a standard offload case.
            engine_state.logger.info(f"[cache-route] Routing to offloaded. B={B}, L={total_len}, Cap={off_cap}")
            return "offloaded", {"offload_cap": off_cap}, None
        else:
            # This is the special case where we fall through to dynamic to clear the cache.
            engine_state.logger.info(f"[cache-route] Offload candidate, but preferring dynamic to clear populated static cache.")
            # Fall through to the static/dynamic logic below.
            pass

    # 2b) Static GPU check is conditional on opt_in AND torch.compile being enabled.
    if opt_in and engine_state.is_compiled:
        pool = _static_pool_for(engine_state, model)
        if pool:
            # Iterate through known buckets and find the first fit.
            available_L_for_B = sorted(known_buckets_map.get(B, []))
            for L_candidate in available_L_for_B:
                potential_slot_key = ((B, L_candidate), adapters_key)

                # --- New check for in-progress warmup ---
                with engine_state._sync_lock: # type: ignore
                    is_active = engine_state._active_signature == potential_slot_key
                    is_pending = potential_slot_key in engine_state._pending_warmup_queue
                    is_already_warming = is_active or is_pending
                if is_already_warming:
                    engine_state.logger.info(f"[cache-route] Warmup for B={B}, L={L_candidate} with adapters {set(adapters_key)} is already active or queued. Using dynamic.")
                    return "dynamic", {}, None # Stop searching, use dynamic.

                # The prompt itself must fit in the candidate bucket.
                if prompt_len > L_candidate:
                    continue

                # Determine the effective prompt length for this bucket, considering padding.
                # This padded length is used to stabilize the graph for torch.compile.
                padded_len_for_bucket = padding_map.get(L_candidate)
                if padded_len_for_bucket and prompt_len <= padded_len_for_bucket:
                    effective_prompt_len = padded_len_for_bucket
                else:
                    effective_prompt_len = prompt_len

                # --- New routing logic based on whether max_new_tokens is user-specified ---
                if not max_new_tokens_is_user_specified:
                    # If user does not specify max_new_tokens, we can't know the generation length.
                    # The most reasonable assumption is that the generation will not exceed the bucket size.
                    # So, we just need to check if the prompt fits, which we already did.
                    # The bucket is a valid candidate.
                    total_needed_for_routing = L_candidate
                else:
                    # CONSERVATIVE: User specified max_new_tokens. We must respect it for routing.
                    generation_buffer_for_routing = max_new_tokens
                    total_needed_for_routing = effective_prompt_len + generation_buffer_for_routing

                if total_needed_for_routing <= L_candidate:
                    # This bucket fits!
                    if pool.is_warmed(B, L_candidate, adapters_key):
                        return "static", {"bucket": (B, L_candidate)}, None
                    else:
                        # Not used yet. Defer warmup only if allowed.
                        if allow_warmup:
                            engine_state.logger.info(f"[cache-route] Deferring warmup for B={B}, L={L_candidate} with adapters {set(adapters_key)}. Using dynamic for now.")
                            return "dynamic", {}, potential_slot_key
                        else:
                            engine_state.logger.info(f"[cache-route] Static slot for B={B}, L={L_candidate} with adapters {set(adapters_key)} is not warm. Using dynamic and skipping warmup due to request type (e.g., fan-out).")
                            return "dynamic", {}, None
            
            # If no bucket fits after checking all candidates, conditionally clear the pool and fall back to dynamic.
            clear_threshold = int(dynamic_clear_cap_by_B.get(B, 0)) if dynamic_clear_cap_by_B else 0
            if clear_threshold and max_new_tokens > clear_threshold:
                engine_state.logger.info(f"[cache-route] Dynamic fallback with large request (max_new_tokens={max_new_tokens} > threshold={clear_threshold}). Invalidating static cache.")
                pool._clear()
            else:
                engine_state.logger.info(f"[cache-route] Dynamic fallback, but request size is within limits (max_new_tokens={max_new_tokens} <= threshold={clear_threshold}). Preserving static cache.")
    elif opt_in and not engine_state.is_compiled:
        # If static cache was opted into, but compile is off, clear any pools to free VRAM.
        ident = id(model)
        if ident in _STATIC_POOLS: _STATIC_POOLS[ident].clear()
    
    # 3) Fallback to dynamic (if opt_in is False, or if opt_in is True but no bucket fit, or compile is off).
    return "dynamic", {}, None


def maybe_apply_static_cache(
    engine_state: "MP13State", 
    model: torch.nn.Module, 
    gen_kwargs: Dict[str, Any], 
    batch_size, 
    prompt_len: int, 
    max_new_tokens: int,
    request: "InferenceRequest",
    active_adapters_for_request: List[str],
    deferred_slots_accumulator: List[Tuple[Tuple[int, int], frozenset[str]]],
    is_micro_batch: bool = False,
) -> Tuple[str, Optional[Tuple[int, int]], Optional[int]]:
    """Mutates gen_kwargs in-place to request a static cache when possible."""
    if batch_size == 0:
        engine_state.logger.warning("[cache-route] Warning: batch_size is 0. Skipping static cache application.")
        gen_kwargs.pop("past_key_values", None) # Ensure no stale cache is present
        return "dynamic", None, max_new_tokens

    cache_override = getattr(request, "cache", None)
    
    # --- New Reset Logic ---
    if cache_override and cache_override.endswith("_reset"):
        engine_state.logger.info(f"[cache-route] Reset requested via cache='{cache_override}'. Resetting static cache.")
        reset_static_cache(engine_state)
        # Use the base mode for routing after reset
        cache_override = cache_override.replace("_reset", "")
    # --- End New Reset Logic ---

    # --- NEW: Global cache check ---
    global_use_cache = engine_state.use_cache
    if not global_use_cache:
        gcfg = gen_kwargs.get("generation_config", getattr(model, "generation_config", GenerationConfig()))
        gcfg = get_modified_generation_config(gcfg, use_cache=False)
        gen_kwargs["generation_config"] = gcfg
        gen_kwargs.pop("past_key_values", None)
        return "no_cache", None, max_new_tokens

    # Determine if warmup should be allowed.
    # It should NOT be allowed if this is a fan-out request.
    # A fan-out request is identified by `request.override_adapters` being present.
    allow_warmup_for_request = not bool(request.override_adapters)

    # Check if max_new_tokens was explicitly provided in the request's generation_config
    req_gen_config = getattr(request, "generation_config", None)
    max_new_tokens_is_user_specified = (
        req_gen_config and "max_new_tokens" in req_gen_config
    )

    mode, info, deferred_slot = route_cache_mode(
        engine_state, 
        model, 
        batch_size=batch_size, 
        prompt_len=prompt_len, 
        max_new_tokens=max_new_tokens, 
        active_adapters=active_adapters_for_request,
        max_new_tokens_is_user_specified=max_new_tokens_is_user_specified or False,
        cache_override=cache_override,
        allow_warmup=allow_warmup_for_request
    )

    # --- NEW: Handle no_cache mode ---
    if mode == "no_cache":
        gcfg = gen_kwargs.get("generation_config", getattr(model, "generation_config", GenerationConfig()))
        gcfg = get_modified_generation_config(gcfg, use_cache=False)
        gen_kwargs["generation_config"] = gcfg
        gen_kwargs.pop("past_key_values", None)
        engine_state.logger.info(f"[cache-route] Caching disabled for this request (global or per-request).")
        return "no_cache", None, max_new_tokens
    # --- END NEW ---

    if mode == "static":
        (B_eff, L_eff) = info["bucket"]
        _, _, _, padding_map, _ = _static_cfg(engine_state)
        padded_prompt_len = padding_map.get(L_eff)
        effective_prompt_len = prompt_len

        # Perform padding if a fixed length is defined for this bucket and inputs are available
        if padded_prompt_len and 'input_ids' in gen_kwargs and gen_kwargs['input_ids'].shape[1] < padded_prompt_len:
            input_ids = gen_kwargs['input_ids']
            attention_mask = gen_kwargs.get('attention_mask')
            pad_len = padded_prompt_len - input_ids.shape[1]
            pad_token_id = getattr(engine_state.tokenizer, "pad_token_id", 0)
            
            padding_tensor = torch.full((input_ids.shape[0], pad_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            gen_kwargs['input_ids'] = torch.cat([padding_tensor, input_ids], dim=1)
            
            if attention_mask is not None:
                padding_attn_mask = torch.zeros((attention_mask.shape[0], pad_len), dtype=attention_mask.dtype, device=attention_mask.device)
                gen_kwargs['attention_mask'] = torch.cat([padding_attn_mask, attention_mask], dim=1)
            
            effective_prompt_len = padded_prompt_len
            engine_state.logger.info(f"[cache-route] Padded prompt from {prompt_len} to {effective_prompt_len} for bucket L={L_eff}.")
        
        allowed_new = L_eff - int(effective_prompt_len)
        original_request = int(max_new_tokens)
        eff_new = int(min(allowed_new, original_request))

        # If override_adapters is used for fan-out, we must not pass the top-level KV cache object.
        # The patched generate method will handle micro-batches, which will use dynamic cache internally.
        # However, we still return True and the effective max_new_tokens to ensure the compiled graph shape is respected.
        # The is_micro_batch flag tells us if this call is for a sub-batch inside the patched generate.
        # If it is, we should NOT disable the KV cache object passing.
        is_fan_out = bool(request.override_adapters) and not is_micro_batch
        if is_fan_out:
            engine_state.logger.info(f"[cache-route] Fan-out request with override_adapters. Using static bucket B={B_eff}, L={L_eff} for shape (align max_new_tokens to {eff_new}), but disabling KV cache object passing.")
            gen_kwargs.pop("past_key_values", None)
        else:
            pool = _static_pool_for(engine_state, model)
            if pool:
                pkv = pool.get(B_eff, L_eff)
                pkv.reset()
                gen_kwargs["past_key_values"] = pkv
                engine_state.logger.info(f"[cache-route] static GPU bucket B={B_eff}, L={L_eff} (align max_new_tokens to stable bucket length: {original_request}->{eff_new})")
        
        return "static", (B_eff, L_eff), eff_new
    
    # Handle offloaded and dynamic cases
    gen_kwargs.pop("past_key_values", None)
    if mode == "offloaded":
        total_len = prompt_len + max_new_tokens
        gcfg = gen_kwargs.get("generation_config", getattr(model, "generation_config", GenerationConfig()))
        gcfg = get_modified_generation_config(gcfg, cache_implementation="offloaded", use_cache=True)
        gen_kwargs["generation_config"] = gcfg
        engine_state.logger.info(f"[cache-route] offloaded for B={batch_size}, L={total_len} (cap={info.get('offload_cap')})")
        return "offloaded", None, max_new_tokens

    # Dynamic fallback is the default
    if deferred_slot:
        if deferred_slot not in deferred_slots_accumulator:
            deferred_slots_accumulator.append(deferred_slot)

    return "dynamic", None, max_new_tokens


async def queue_static_cache_warmup(engine: "MP13Engine", signature: Tuple[Tuple[int, int], frozenset[str]]):
    """
    Queues a cache signature for background warming. Starts the worker if not already running.
    This is the new entry point for triggering background warmups.
    """
    global _warmup_task_future, _warmup_lock, _engine_instance_for_worker

    # This is the critical section that needs to be thread-safe.
    def _queue_if_not_present_sync():
        with engine.state._sync_lock: # type: ignore
            # gate on shutdown or mode switch    
            if engine.state._prevent_new_background_tasks:
                return False

            is_active = engine.state._active_signature == signature
            is_pending = signature in engine.state._pending_warmup_queue
            if is_active or is_pending:
                return False # Already being warmed or in the queue
            engine.state._pending_warmup_queue.append(signature)
            return True

    # Run the synchronous, locked part in a thread to avoid blocking the event loop
    was_added = await asyncio.to_thread(_queue_if_not_present_sync)
    if not was_added:
        return # Don't proceed if it was already in the queue

    engine.state.logger.info(f"[bg-warmup] Queued 1 new unique signature(s) for warmup. Queue size: {len(engine.state._pending_warmup_queue)}")

    with _warmup_lock:
        if _warmup_task_future and not _warmup_task_future.done():
            engine.state.logger.info("[bg-warmup] A warmup task is already active. Newly queued items will be processed by it.")
            return

        if engine.state._bg_exec:
            engine.state.logger.info("[bg-warmup] No active warmup task. Submitting a new background worker.")
            _engine_instance_for_worker = engine # Store engine instance for the worker
            _warmup_task_future = engine.state._bg_exec.submit(_background_warmup_worker, engine)

def _warmup_a_signature_sync(
    model: torch.nn.Module,
    tokenizer: "PreTrainedTokenizerBase",
    engine: "MP13Engine",
    B: int,
    L: Optional[int],
    adapters_to_warm: frozenset[str],
    log_fn: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Warms up a single signature (B, L) for dynamic or static cache.
    This is a blocking function intended for an executor.
    It now mimics the production generation config and stopping criteria more closely
    to ensure torch.compile graphs are correctly cached.
    """
    # --- 1. Build a realistic GenerationConfig, mimicking mp13_infer.py's logic ---
    success = False
    base_gen_config = getattr(model, 'generation_config', None)
    if base_gen_config and hasattr(base_gen_config, 'to_dict'): # type: ignore
        final_gen_config_dict = base_gen_config.to_dict()
    else:
        final_gen_config_dict = {}

    # Merge session defaults
    if engine.state.current_inference_session_config:
        final_gen_config_dict.update(engine.state.current_inference_session_config.get('default_generation_config', {}))

    # --- 2. Create dummy inputs, same as before ---
    sess = engine.state.current_inference_session_config or {}
    static_opt_in, _, _, padding_map, _ = _static_cfg(engine.state)
    
    # For sharded models, inputs must be created on the first device (e.g., cuda:0 for embeddings)
    # The first_module_device_for_sharded_model helper correctly finds this.
    
    device = first_module_device_for_sharded_model(model)
    if log_fn: log_fn(f"Creating dummy tensors on device {device}.")

    padded_prompt_len = padding_map.get(L, 2) if L is not None else 2
    
    gcfg_base_for_pad = getattr(model, "generation_config", GenerationConfig())
    pad_token_id = gcfg_base_for_pad.pad_token_id # type: ignore
    if pad_token_id is None:
        eos_ids = gcfg_base_for_pad.eos_token_id
        if not isinstance(eos_ids, list): eos_ids = [eos_ids]
        pad_token_id = eos_ids[0] if eos_ids and eos_ids[0] is not None else 0

    ids = torch.full((B, padded_prompt_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(ids)
    if padded_prompt_len > 0: ids[:, -1] = 1; attention_mask[:, -1] = 1

    # --- 3. Finalize GenerationConfig with 2 tokens to trigger decoding loop ---
    final_gen_config_dict['max_new_tokens'] = 2
    final_gen_config_dict['min_new_tokens'] = 2
    
    # Use the helper to get the final, clean config.
    base_config_for_modification = GenerationConfig(**{k: v for k, v in final_gen_config_dict.items() if v is not None})
    warmup_gc = get_modified_generation_config(base_config_for_modification, do_sample=False)
    # Defensive: set explicit max/min new tokens and clear length fields to avoid warnings.
    max_new = 2
    warmup_gc.max_new_tokens = max_new  # type: ignore[attr-defined]
    warmup_gc.min_new_tokens = None  # type: ignore[attr-defined]
    warmup_gc.max_length = None  # type: ignore[attr-defined]
    warmup_gc.min_length = None  # type: ignore[attr-defined]
    if log_fn:
        log_fn("[warmup] gen_config lengths: max_new=2, min_new=None, max_length=None, min_length=None")

    # --- 4. Build gen_kwargs, mimicking production path ---
    # Annotate as a general mapping so Pylance doesn't narrow the value type to the
    # union inferred from the literal values (Tensor | GenerationConfig). We need
    # to be able to insert engine state and other runtime-only keys.
    gk: Dict[str, Any] = dict(
        input_ids=ids,
        attention_mask=attention_mask,
        generation_config=warmup_gc,
        max_new_tokens=max_new,
    )

    # --- 4b. Patch device argument compat for HF/Hub models that require it ---
    def _ensure_causal_mask_device_compat(obj: Any):
        fn = getattr(obj, "_prepare_4d_causal_attention_mask_with_cache_position", None)
        if fn is None or getattr(fn, "_mp13_device_compat", False):
            return
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return
        param = sig.parameters.get("device")
        if param is None or param.default is not inspect._empty:
            return

        def _wrapper(*args, **kwargs):
            if kwargs.get("device") is None:
                input_ids = kwargs.get("input_ids")
                if input_ids is None:
                    for a in args:
                        if isinstance(a, torch.Tensor):
                            input_ids = a
                            break
                if input_ids is not None:
                    kwargs["device"] = input_ids.device
            return fn(*args, **kwargs)

        _wrapper._mp13_device_compat = True  # type: ignore[attr-defined]
        setattr(obj, "_prepare_4d_causal_attention_mask_with_cache_position", _wrapper)

    # Apply to all modules to catch nested HF models wrapped by PEFT/mixed models.
    seen: Set[int] = set()
    for candidate in model.modules():
        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)
        _ensure_causal_mask_device_compat(candidate)


    # --- 5. Handle static cache, same as before ---
    pkv = None
    if static_opt_in and L is not None:
        if not _static_cache_is_device_safe(engine.state, model):
            if log_fn: log_fn("[warmup] Static cache disabled due to multi-device/offload placement. Skipping warmup.")
            return False
        # The `cache_kwargs` must be passed inside the `generation_config` object
        # for the `generate` method to recognize and use it.
        gcfg_for_warmup = gk.get("generation_config", GenerationConfig())
        pool = _static_pool_for(engine.state, model, create_if_missing=True)
        if not pool:
            if log_fn: log_fn(f"[warmup] Could not get static pool for model. Skipping warmup for ({B}, {L}).")
            return False
        
        pkv = pool.get(B, L) # type: ignore
        pkv.reset() # CRITICAL: Reset cache state before use.
        if log_fn: log_fn(f"[warmup] Warming graph for static bucket B={B}, L={L} (using persistent pool).")
        gk["past_key_values"] = pkv

    # Signal to patched HF generate that this is cache warming call
    gk["_is_warmup_call"] = True
    gk["mp13_engine_state"] = engine.state # so generate patch can detectt first warm up
    
    # For sharded models, we must pass device="auto" in cache_kwargs to let the model
    # handle the creation of cache tensors on the correct devices.
    # --- 6. Run generation (prefill+decode inside HF loop) ---
    try:
        with torch.no_grad():
            if L is None:
                if log_fn: log_fn(f"Skipping generate call because L is None (dynamic cache).")
                return False

            if torch.cuda.is_available():
                dev = next(model.parameters()).device
                if dev.type == "cuda":
                    torch.cuda.set_device(dev)
                stream = low_priority_stream_for(dev)
                with torch.cuda.stream(stream):
                    with no_cudagraphs():
                        model.generate(**gk)

                # Record an event at the end of warmup work, on the warmup stream
                done_evt = torch.cuda.Event(blocking=False, enable_timing=False)
                stream.record_event(done_evt)
    
                # Later, when you want to mark the slot ready, poll the event (no host sync):
                #ready = done_evt.query()  # True = warmup finished, False = still in flight
                # If you need on-GPU ordering with the inference stream, do:
                #infer_stream.wait_event(done_evt)

                # attach to the slot so inference can fence if it grabs this pkv
                if pkv is not None:
                    setattr(pkv, "_ready_event", done_evt)

            else:
                model.generate(**gk)

            success = True

    finally:
        # Intentionally do not call torch.cuda.empty_cache() here.
        # Calling it while *any* CUDA graph capture is underway can crash
        # the process (allocator internal assert).
        pass
    return success


def _warmup_one_slot_sync_work(
    model: torch.nn.Module,
    tokenizer: "PreTrainedTokenizerBase",
    slot_and_adapters: Tuple[Tuple[int, int], frozenset[str]],
    engine: "MP13Engine"
) -> bool:
    """
    Performs the synchronous part of the warmup: adapter switching and GPU work.
    This function assumes the appropriate cohort lock is already held by the caller.
    """
    (B, L), adapters_to_warm = slot_and_adapters
    adapters_control = engine.adapters_control
    warmup_succeeded = False
    adapters_list = adapters_control.get_normalized_adapter_set(adapters_to_warm)
    current_on_model = adapters_control.get_current_active_set_unlocked()
    if adapters_list != current_on_model:
        engine.state.logger.info(f"[bg-warmup] Switching adapters from {current_on_model} to {adapters_list} for warmup.")
        adapters_control.one_time_switch_to_set(adapters_list)
    try:
        # Call the core blocking warmup logic
        warmup_succeeded = _warmup_a_signature_sync(
            model=model, tokenizer=tokenizer, engine=engine,
            B=B, L=L, adapters_to_warm=adapters_to_warm,
            log_fn=lambda m: engine.state.logger.info(f"[bg-warmup] {m}")
        )
    finally:
        if adapters_list != current_on_model:
            adapters_control.one_time_switch_to_set(current_on_model)
            engine.state.logger.info(f"[bg-warmup] Restored adapters from {adapters_list} to {current_on_model} after warmup.")
    return warmup_succeeded

def _background_warmup_worker(engine: "MP13Engine"):
    """
    Runs in a background thread to process the static cache warmup queue.
    This is a SYNCHRONOUS function. It calls thread-safe wrappers for async state changes.
    """
    global _warmup_task_future, _warmup_lock
    state = engine.state

    # This call now uses the new synchronous, thread-safe wrapper
    state.set_warming_status_from_thread(is_warming=True)
    
    try:
        model = state.peft_model # Get model once
        tokenizer = state.tokenizer_for_warmup
        if not model or not tokenizer:
            state.logger.error("[bg-warmup] ERROR: Model or tokenizer not available in state. Worker exiting.")
            return

        if not hasattr(model, 'set_adapter'):
            state.logger.error(f"[bg-warmup] Model of type {type(model).__name__} does not support adapter switching. Worker exiting.")
            return

        while True:
            signature = None
            try:
                # This part is synchronous and thread-safe (deque is thread-safe for pop/append)
                signature = state._pending_warmup_queue.popleft()
            except IndexError:
                # Queue is empty, worker's job is done for now.
                break # Queue is empty

            # --- Acquire Cohort Lock (if adapters are involved) ---
            (B, L), adapters_to_warm = signature
            adapters_control = engine.adapters_control
            adapters_list = adapters_control.get_normalized_adapter_set(adapters_to_warm)
            
            cohort_key = ("NF", adapters_list)
            enter_future = asyncio.run_coroutine_threadsafe(
                adapters_control.cohort_enter(cohort_key, request="CACHE_WARMUP"),
                state.loop
            )
            enter_future.result() # Block this worker thread until lock is acquired

            # This outer try/finally ensures that the active signature is cleared
            # and the first-warmup event is set, even if the warmup itself fails.
            try:
                state._active_signature = signature
                pool = _static_pool_for(state, model)
                try:
                    # This is the core blocking operation, which is why this runs in a thread.
                    success = _warmup_one_slot_sync_work(model, tokenizer, signature, engine)
                    
                    if success and pool:
                        pool.mark_as_warmed(B, L, frozenset(adapters_list))
                        state.logger.info(f"[bg-warmup] Successfully warmed and marked slot B={B},L={L} for adapters: {set(adapters_list)}")

                except Exception as e:
                    adapters_str = ", ".join(sorted(list(adapters_list))) if adapters_list else "base"
                    # --- If warmup fails, remove the potentially corrupted cache object from the pool ---
                    if pool:
                        try:
                            pool.remove(B, L)
                            state.logger.warning(f"[bg-warmup] Removed potentially corrupted cache slot B={B}, L={L} from pool due to warmup failure.")
                        except Exception as e_remove:
                            state.logger.error(f"[bg-warmup] CRITICAL: Failed to remove corrupted cache slot B={B}, L={L} from pool: {e_remove}")

                    state.logger.error(f"[bg-warmup] Error warming slot B={B},L={L},A=[{adapters_str}]: {e}")
                    state.logger.debug(f"[bg-warmup] Warmup traceback:\n{traceback.format_exc()}")
            finally:
                # CRITICAL: Always mark the slot as "processed" to clear the active signature
                # and allow the next item in the queue to be processed.
                state.set_slot_warmup_complete_from_thread(signature)
                
                # --- Release Cohort Lock ---
                leave_future = asyncio.run_coroutine_threadsafe(
                    adapters_control.cohort_leave(request="CACHE_WARMUP"), state.loop)
                leave_future.result() # Wait for cohort to be left on main loop.

    except Exception as e:
        state.logger.critical(f"[bg-warmup] !!! Error during background warmup task: {e}\n{traceback.format_exc()}")
    finally:
        # This call also uses the new synchronous, thread-safe wrapper
        state.set_warming_status_from_thread(is_warming=False)
        
        # Clear the task handle so a new one can be started next time.
        with _warmup_lock:
            _warmup_task_future = None
            state._active_signature = None
        state.logger.info("[bg-warmup] Background worker finished and task handle cleared.")
