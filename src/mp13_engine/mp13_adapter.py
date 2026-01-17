# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""MP13 Engine - Adapter management functionality."""

import asyncio
from enum import Enum
from collections import deque, defaultdict
import json
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Iterable, Tuple, TYPE_CHECKING, cast, Set
import threading
import torch
from peft import PeftModel, PeftMixedModel, LoraConfig, TaskType, PeftConfig

from .mp13_config import AdapterConfig, AdapterType, IfExistsEnum, InferenceRequest
from . import mp13_state
from .mp13_state import (
    MP13State, ConfigurationError, AdapterError, EngineError,
    EngineModeState
)
from .mp13_utils import find_latest_checkpoint_in_dir
from .mp13_cache import reset_compile_warm_tracker, invalidate_warmed_signatures_except_base

if TYPE_CHECKING:
    from .mp13_config import InferenceRequest

class CohortTaskType(str, Enum):
    """Defines the types of non-inference tasks that can enter a cohort."""
    LOAD_ADAPTER = "load_adapter"
    UNLOAD_ADAPTER = "unload_adapter"
    SET_ADAPTERS = "set_adapters"
    CACHE_WARMUP = "cache_warmup"


# --- DEBUG HELPERS -----------------------------------------------------------
from peft.tuners.tuners_utils import BaseTunerLayer

# ---------- helpers ----------
def _unwrap_if_compiled(m):
    return getattr(m, "_orig_mod", m)

def _active_adapters_compat(m):
    aa = getattr(m, "active_adapters", None)
    # Directly queries the `active_adapters` attribute from the model object `m`.
    aa = getattr(m, "active_adapters", None)
    return aa() if callable(aa) else aa

def quant_display_from_meta(meta: Optional[Dict[str, Any]], quant_method: Optional[str] = None) -> Optional[str]:
    if not meta:
        return None
    precision_info = meta.get("precision_info") or {}
    quant_cfg = precision_info.get("base_model_quantization_config") or precision_info.get("quantization_config") or {}

    def _norm(v: Any) -> Optional[str]:
        return str(v).lower() if v is not None else None

    q_bits = _norm(quant_cfg.get("quantize_bits"))
    q_method = _norm(
        quant_cfg.get("quant_method")
        or quant_cfg.get("method")
        or quant_cfg.get("quant_type")
        or quant_cfg.get("backend")
        or quant_cfg.get("name")
        or quant_cfg.get("load_type")
    )
    hqq_bits = quant_cfg.get("hqq_bits") or (quant_cfg.get("quant_config", {}) or {}).get("weight_quant_params", {}).get("nbits")
    if (q_bits and "hqq" in q_bits) or (q_method and "hqq" in q_method):
        try:
            bits_int = int(hqq_bits) if hqq_bits is not None else None
        except Exception:
            bits_int = None
        return f"HQQ-i{bits_int}" if bits_int is not None else "HQQ"

    load_in_4bit = quant_cfg.get("load_in_4bit")
    bnb_type = _norm(quant_cfg.get("bnb_4bit_quant_type"))
    if load_in_4bit:
        if bnb_type:
            return f"BNB-{bnb_type.upper()}"
        return "BNB-4bit"

    load_in_8bit = quant_cfg.get("load_in_8bit")
    if load_in_8bit:
        return "BNB-i8"

    dtype = (
        precision_info.get("dtype")
        or precision_info.get("precision")
        or precision_info.get("format")
        or precision_info.get("base_model_effective_dtype_at_init")
        or meta.get("base_model_effective_dtype_at_init")
    )
    if dtype:
        dtype_norm = _norm(dtype) or ""
        dtype_norm = dtype_norm.replace("torch.", "")
        if "bfloat16" in dtype_norm or dtype_norm == "bf16":
            return "BP16"
        if "float16" in dtype_norm or dtype_norm == "fp16" or "half" in dtype_norm:
            return "FP16"
        if "float32" in dtype_norm or dtype_norm == "fp32" or dtype_norm == "float":
            return "FP32"
        if "float64" in dtype_norm or dtype_norm == "fp64" or "double" in dtype_norm:
            return "FP64"
        return str(dtype)

    if quant_method:
        return str(quant_method)
    return None

def adapter_health_AB(logger, model, adapter_name, max_print=3):
    from peft.tuners.tuners_utils import BaseTunerLayer
    m = getattr(model, "_orig_mod", model)
    found = 0
    total_A = 0.0
    total_B = 0.0
    samples = []
    skipped_meta = False
    for modname, mod in m.named_modules():
        if isinstance(mod, BaseTunerLayer):
            A = getattr(getattr(mod, "lora_A", {}), adapter_name, None)
            B = getattr(getattr(mod, "lora_B", {}), adapter_name, None)
            if A is not None and B is not None:
                if _has_meta_params(A) or _has_meta_params(B):
                    skipped_meta = True
                    continue
                found += 1
                a = A.weight.data.norm().item()
                b = B.weight.data.norm().item()
                total_A += a
                total_B += b
                if len(samples) < max_print:
                    samples.append((modname, tuple(A.weight.shape), tuple(B.weight.shape), a, b))
    logger.debug(f"[PEFT-DEBUG] [healthAB] adapter={adapter_name} layers={found} total_A_norm={total_A:.6f} total_B_norm={total_B:.6f}")
    if skipped_meta:
        logger.debug("!!! [PEFT-DEBUG] [healthAB] skipped meta LoRA tensors (offloaded model); totals are partial.")
    for s in samples:
        logger.debug(f"   {s}")

def _has_meta_params(model) -> bool:
    try:
        return any(getattr(p, "is_meta", False) for p in model.parameters())
    except Exception:
        return False

def _safetensors_lora_B_norms(logger, ckpt_dir):
    import os
    from safetensors.torch import load_file
    fn = os.path.join(ckpt_dir, "adapter_model.safetensors")
    sd = load_file(fn)
    total = 0.0
    count = 0
    for k, v in sd.items():
        if k.endswith("lora_B.weight") or k.endswith("lora_B.default.weight"):
            n = v.norm().item()
            total += n
            count += 1
            if count <= 5:
                 logger.debug(f"[PEFT-DEBUG] disk B: {k} {tuple(v.shape)} {n}")
    logger.debug(f"[PEFT-DEBUG] disk total_B_norm={total:.6f} across {count} tensors")

def first_lora_status(logger, model, tag=""):
    model = _unwrap_if_compiled(model)
    for name, mod in model.named_modules():
        if isinstance(mod, BaseTunerLayer):
            merged = getattr(mod, "merged", None) 
            # Directly queries the `active_adapter` attribute from a submodule.
            disabled = getattr(mod, "disable_adapters", None)
            active = getattr(mod, "active_adapter", None)
            # scaling can be a dict keyed by adapter name on many PEFT builds
            scaling = getattr(mod, "scaling", None)
            logger.debug(f"[PEFT-DEBUG]{('['+tag+']') if tag else ''} "
                   f"first_lora_layer='{name}' merged={merged} disabled={disabled} "
                   f"active={active} scaling={scaling if isinstance(scaling, dict) else type(scaling).__name__}")
            return
    logger.debug(f"[PEFT-DEBUG]{('['+tag+']') if tag else ''} no BaseTunerLayer found")

def log_model_state(logger, model, tag):
    model = _unwrap_if_compiled(model)
    try:
        logger.debug(f"[PEFT-DEBUG][{tag}] class={type(model).__name__} id={id(model)} "
               f"peft_config={list(getattr(model, 'peft_config', {}).keys())} "
               f"active_adapters={_active_adapters_compat(model)}")
    except Exception as e:
        logger.debug(f"[PEFT-DEBUG][{tag}] state-read-failed: {e}")
        pass
# --- END DEBUG HELPERS -----


def _compile_blocking(state: "MP13State", model: Any, start_time: float) -> Any:
    """
    Unwraps a torch.compiled model and recompiles it.
    This is a blocking function intended to be run in an executor.
    """
    import torch
    from torch._inductor import config as ic
    ic.triton.cudagraphs = False
    ic.triton.cudagraph_trees = False

    # Unwrap compiled wrappers (OptimizedModule) to the original module
    base = getattr(model, "_orig_mod", model)
    try:
        compiled = torch.compile(base, mode="reduce-overhead") #mode="max-autotune-no-cudagraphs")
        # Guard: make sure we really got an nn.Module back
        import torch.nn as nn
        if not isinstance(compiled, nn.Module):
            state.logger.warning("!!! Warning: torch.compile() returned a non-module object. Reverting to original model.", start_time)
            return base  # fall back to the real module to avoid poisoning state with a function
        state.logger.info("Inference model re-compiled successfully.", start_time)
        return compiled
    except Exception as e:
        state.logger.warning(f"!!! Warning: torch.compile() on model re-compilation failed: {e}. Continuing without compilation.", start_time)
        state.logger.warning(traceback.format_exc(), start_time)
        return base

#TBD Expensive for static cache not used currently
async def _recompile_inference_model_if_needed(state: "MP13State", model_to_recompile: Any, start_time: float) -> None:
    """
    Helper to re-compile the persistent inference model after its structure changes.
    """
    # Only recompile the persistent PeftMixedModel for inference, not training models.
    if model_to_recompile is not state.peft_model:
        return
    
    # Check if torch.compile is enabled globally
    if not (state.global_config and state.global_config.get("use_torch_compile", False)):
        return

    if not state._gen_exec:
        state.logger.warning("!!! Warning: Generation executor not available. Cannot re-compile inference model.")
        return

    state.logger.info("Re-compiling inference model due to adapter change...")
    loop = asyncio.get_running_loop()

    old_model_id = id(state.peft_model)
    old_model_ref = state.peft_model # Keep a reference to the old model

    # Run compilation in the executor and update the model in the state
    new_compiled_model = await loop.run_in_executor(
        state._gen_exec,
        _compile_blocking,
        old_model_ref, # Pass the old model to be unwrapped and recompiled
        start_time
    )
    
    # Explicitly replace the model in the state
    state.peft_model = new_compiled_model
    # Now, explicitly delete the old model and clear caches to free up VRAM before warmup
    del old_model_ref
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.ipc_collect()
    state.logger.info("Old compiled model released and CUDA cache cleared before re-warmup.")

    new_model_id = id(new_compiled_model)

    # Reset compile tracker if a new compiled model was created
    if new_model_id != old_model_id:
        state.logger.info("New compiled model instance created. Resetting warmup tracker.")
        reset_compile_warm_trackers(state)


def _normalize_adapter_set(names: Iterable[Optional[str]]) -> Tuple[str, ...]:
    """
    Normalizes an iterable of adapter names into a canonical, sorted tuple.
    '__base__' and None are treated as an empty set.
    """
    if not names:
        return tuple()
    processed_names = {name for name in names if name and name != "__base__"}
    return tuple(sorted(processed_names))



def create_quantization_metadata(engine_quant_config: dict) -> dict:
    """
    Creates a canonical dictionary of quantization metadata from the engine's quantization_config.
    This is the unified function for creating a comparable metadata object, which is stored
    in the project's private metadata.json file.
    """
    if engine_quant_config and 'quant_method' in engine_quant_config:
        return {
            'quant_method': engine_quant_config['quant_method'],
            'quant_config': engine_quant_config.get('quant_config'),
            'skip_modules': engine_quant_config.get('skip_modules', [])
        }
    return {}


def _validate_quantization_config(logger, metadata_from_disk: dict, engine_quant_config: dict):
    """
    Compares adapter's quantization config (from metadata.json) with the engine's config.
    Logs a warning if there is a mismatch.
    """
    adapter_quant_config = metadata_from_disk.get("precision_info", {}).get("base_model_quantization_config")
    source = "metadata.json"

    # Case 1: Engine is not quantized.
    if not engine_quant_config or not engine_quant_config.get('quant_method'):
        if adapter_quant_config:
            logger.warning("Warning: Loading a quantized adapter into a non-quantized model. This is not recommended.")
        return

    # Case 2: Adapter has no quantization metadata.
    if not adapter_quant_config:
        logger.info("Adapter does not have quantization metadata in metadata.json. Assuming compatibility.")
        return

    # Case 3: Both are quantized, compare them.
    # Create a clean, comparable dictionary from the live engine state.
    engine_quant_config_for_comparison = create_quantization_metadata(engine_quant_config)

    if adapter_quant_config != engine_quant_config_for_comparison:
        adapter_method_str = str(adapter_quant_config.get('quant_method', 'N/A'))
        engine_method_str = str(engine_quant_config_for_comparison.get('quant_method', 'N/A'))

        adapter_method = adapter_method_str.split('.')[-1].lower()
        engine_method = engine_method_str.split('.')[-1].lower()
        
        adapter_params = adapter_quant_config.get('quant_config', {}).get('weight_quant_params', {}) or {}
        engine_params = engine_quant_config_for_comparison.get('quant_config', {}).get('weight_quant_params', {}) or {}
        
        adapter_nbits = adapter_params.get('nbits')
        engine_nbits = engine_params.get('nbits')

        warning_msg = ""
        if adapter_method == engine_method and adapter_nbits == engine_nbits:
             warning_msg = (
                "Warning: Adapter was trained with a different quantization setup. "
                "While method and bit-width match, other parameters differ. Performance or behavior may vary."
             )
        else:
            warning_msg = (
                "Warning: Adapter quantization config does not match engine. "
                "Key parameters like method or bit-width are different."
            )
        
        logger.warning(
            f"{warning_msg}\n"
            f"--> Adapter metadata (from {source}): {adapter_quant_config}\n"
            f"--> Engine config: {engine_quant_config_for_comparison}"
        )


class AdaptersControl:
    """Handles adapter-related operations for the MP13 Engine."""
    
    def __init__(self, state: MP13State):
        self.state = state

    # --- Path helpers shared by training and load flows ---
    def _precision_suffix_from_state(self) -> str:
        """Derive a precision/quantization suffix string from current state/config."""
        gcfg = self.state._global_config or {}
        quant_method = gcfg.get("effective_quantization_method") or "none"
        details = self.state.quantization_config or {}
        if quant_method in (None, "none"):
            dt = gcfg.get("effective_torch_dtype") or self.state.requested_torch_dtype or ""
            dt = str(dt).lower()
            if "bfloat16" in dt or dt == "bf16":
                return "bf16"
            if "float16" in dt or dt == "fp16" or "half" in dt:
                return "fp16"
            return dt or "fp32"
        bits = details.get("hqq_bits") or details.get("awq_bits") or details.get("quantize_bits")
        if bits:
            return f"{quant_method}-i{bits}"
        return str(quant_method)

    def build_model_precision_dir_name(self) -> str:
        """Return the directory name combining model base name and precision/quantization info."""
        model_base = self.state.base_model_name_or_path or "model"
        model_base = Path(model_base).name
        return f"{model_base}.{self._precision_suffix_from_state()}"

    def derive_adapter_root_for_storage(self, adapter_name: str, root_hint: Optional[Union[str, Path]]) -> Path:
        """
        Given an adapter name and a user-provided root hint, derive the adapter root folder
        using the model/precision directory convention.
        """
        base = Path(root_hint or adapter_name).expanduser()
        try:
            base = base.resolve()
        except Exception:
            pass

        adapter_norm = adapter_name.replace(".", "_")
        # If the hint already looks like an adapter root (metadata or weights), use it directly.
        if (base / "metadata.json").is_file() or (base / "adapter_model.safetensors").is_file():
            return base
        # If the hint ends with the adapter name and exists (or is empty), treat it as adapter root.
        if base.name.replace(".", "_") == adapter_norm:
            return base

        return (base / self.build_model_precision_dir_name() / adapter_name).resolve()

    def get_active_and_pending_requests_count(self) -> Tuple[int, int]:
        """
        Returns the number of active (inflight) and pending (queued) cohort requests.
        This is a lock-free, synchronous method for speed, accepting minor precision loss.
        """
        state = self.state
        return state._cohort_active_inference_count, state._cohort_pending_count

    def has_active_or_pending_requests(self) -> bool:
        """
        Checks if there are any inference requests that are currently
        active (inflight) or pending in any cohort queue.
        """
        state = self.state
        active_count, pending_count = self.get_active_and_pending_requests_count()
        return active_count > 0 or pending_count > 0
    # --------------------------------------------------------------------------------------
    # Ordered concurrency “cohorts” + one-time adapter switching (implemented at run-level) # noqa: E501
    # --------------------------------------------------------------------------------------
    def _cohort_init_if_needed(self):
        state = self.state
        if not hasattr(state, "_cohort_lock") or state._cohort_lock is None:
            state._cohort_lock = asyncio.Lock()
        if state._cohort_sem is None:
            try:
                cg = int((state.global_config or {}).get("concurrent_generate", 1))
            except Exception:
                cg = 1
            state._cohort_cg = max(1, cg)
            state._cohort_sem = asyncio.Semaphore(state._cohort_cg)
            state._cohort_sem_nf = asyncio.Semaphore(1)
            state._cohort_rotor = state._cohort_rotor if getattr(state, "_cohort_rotor", None) else deque()
            state._cohort_queues = state._cohort_queues if getattr(state, "_cohort_queues", None) else defaultdict(deque)
            # Map of active (inflight) requests. Structure: { cohort_key: { request_id: entry_list } }
            # We keep active entries as mutable lists to allow cancel_request to find and signal them.
            state._cohort_active = state._cohort_active if getattr(state, "_cohort_active", None) else defaultdict(dict)
            state._cohort_active_inference_count = getattr(state, "_cohort_active_inference_count", 0)
            state._cohort_arrival_ctr = getattr(state, "_cohort_arrival_ctr", 0)
            state._cohort_inflight = getattr(state, "_cohort_inflight", 0)
            # Set queue capacity to 5 * concurrent_generate (same for NF)
            state._cohort_sig_cap = max(1, int(state._cohort_cg) * 5)
            state._cohort_current_key = getattr(state, "_cohort_current_key", None)
            # New condition variable for backpressure
            state._cohort_queue_cond = asyncio.Condition(lock=state._cohort_lock)
            # --- New: per-queue head sequence to pick oldest cohort fairly across kinds ---
            # Maps cohort_key -> arrival_seq for the current head item of that queue.
            state._cohort_head_seq = getattr(state, "_cohort_head_seq", {})

    async def _cohort__maybe_adopt_head_unlocked(self):
        state = self.state
        if state._cohort_inflight == 0 and state._cohort_rotor:
            state._cohort_current_key = state._cohort_rotor[0]

    async def _cohort__matching_sem_unlocked(self):
        state = self.state
        kind, _ = state._cohort_current_key or ("F", tuple())
        return state._cohort_sem if kind == "F" else state._cohort_sem_nf

    async def _cohort__plan_releases_unlocked(self):
        """
        Plan which waiters to wake. May rotate to a non-empty queue
        IFF inflight==0. No awaits; must be called WITH _cohort_lock held.
        Returns (sem, to_wake: List[(aid, event)]).
        """
        state = self.state
        # Helper: choose the next cohort by "oldest head" across all non-empty queues.
        def _select_oldest_nonempty_key():
            oldest_key = None
            oldest_seq = None
            for k in state._cohort_rotor:
                q = state._cohort_queues.get(k, deque())
                if not q:
                    continue
                head_seq = state._cohort_head_seq.get(k, None)
                if head_seq is None:
                    # Fallback: treat as very new if missing; should be rare.
                    continue
                if oldest_seq is None or head_seq < oldest_seq:
                    oldest_seq = head_seq
                    oldest_key = k
            return oldest_key

        # If no inflight, we pick the oldest non-empty cohort globally.
        if state._cohort_inflight == 0:
            state._cohort_current_key = _select_oldest_nonempty_key()

        if not state._cohort_current_key:
            return None, []
        sem = await self._cohort__matching_sem_unlocked()
        cap = state._cohort_cg if state._cohort_current_key[0] == "F" else 1
        qsig = state._cohort_queues[state._cohort_current_key]
        to_wake = []
        # qsig entries are stored in multiple shapes. New user-facing entries
        # are stored as lists of length 5: [id, event, has_request, cancel_event, request].
        # When a user-facing entry is selected to run, we move it into the active map.
        cohort_key = state._cohort_current_key
        active_for_key = state._cohort_active[cohort_key] # type: ignore
        # --- Refined gating: while any OTHER queue is non-empty AND we have inflight,
        # do not admit additional *inference* items from this cohort.
        # We still allow non-inference tasks (e.g., CACHE_WARMUP) to pass
        # so warmups and adapter management are never deadlocked behind the gate.
        gate_inference = (
            state._cohort_inflight > 0 and any(
                (k != cohort_key) and state._cohort_queues.get(k, deque())
                for k in state._cohort_rotor # type: ignore
            )
        )

        # qsig entries may be mutated above; we pop and convert as needed.
        # If gating, we will *peek* first: if the head is an inference request, we stop;
        # if it is a non-inference task, we let it through.
        while len(to_wake) < cap and qsig: # type: ignore
            if gate_inference:
                head = qsig[0] # type: ignore
                try:
                    # head may be tuple or list; has_request flag is at index 2
                    head_has_request = bool(head[2]) if isinstance(head, (list, tuple)) and len(head) >= 3 else False
                except Exception:
                    head_has_request = False
                if head_has_request:
                    break  # keep the gate: do not admit more inference from this cohort

            item = qsig.popleft() # type: ignore
            state._cohort_pending_count -= 1
            # If there are enqueue waiters for this cohort, wake one now
            # Notify any task waiting on the condition that a slot has freed up.
            state._cohort_queue_cond.notify() 
            try:
                # The tuple signature is (id, event, has_request, cancel_event, request_obj_or_task_type, arrive_seq)
                if not (isinstance(item, (list, tuple)) and len(item) >= 6):
                    self.state.logger.warning(f"[ADAPTER-WARN] Skipping malformed item in cohort queue: {item}")
                    continue
                item_id, item_evt, has_request, _, _, _ = item
            except (IndexError, ValueError):
                self.state.logger.warning(f"[ADAPTER-WARN] Skipping malformed item in cohort queue (unpacking failed): {item}")
                continue

            # If this was a user-facing request, move it to the active map so
            # cancel_request can find it while it is inflight.
            if has_request:
                # Ensure we keep a mutable container for active entries
                if not isinstance(item, list): # type: ignore
                    item = list(item)
                active_for_key[item_id] = item # type: ignore

            # Handoff to the caller only needs id and event
            to_wake.append((item_id, item_evt, has_request))

        # Update head sequence for this queue after pops
        if qsig: # type: ignore
            next_head = qsig[0] # type: ignore
            if isinstance(next_head, (list, tuple)) and len(next_head) >= 6:
                state._cohort_head_seq[cohort_key] = next_head[5] # type: ignore
        else:
            # queue became empty; remove its head seq
            state._cohort_head_seq.pop(cohort_key, None) # type: ignore

        return sem, to_wake
 
    async def _cohort__maybe_rotate_unlocked(self):
        """
        If current queue is empty and nothing inflight, adopt the oldest non-empty queue.
        Must be called with state._cohort_lock held.
        """
        state = self.state
        if state._cohort_inflight != 0:
            return
        # Current empty? choose oldest non-empty queue globally
        if not state._cohort_current_key or not state._cohort_queues.get(state._cohort_current_key, deque()):
            oldest_key = None
            oldest_seq = None
            for k in state._cohort_rotor:
                q = state._cohort_queues.get(k, deque())
                if not q:
                    continue
                head_seq = state._cohort_head_seq.get(k, None)
                if head_seq is None:
                    continue
                if oldest_seq is None or head_seq < oldest_seq:
                    oldest_seq = head_seq
                    oldest_key = k
            state._cohort_current_key = oldest_key

    async def cohort_enter(
        self, 
        key: Tuple[str, Tuple[str, ...]], 
        request: Optional[Union["InferenceRequest", CohortTaskType]] = None
    ) -> threading.Event:
        self._cohort_init_if_needed()
        state = self.state
        evt = asyncio.Event()
        queue_id: Union[int, str]

        # Acquire the lock to safely modify shared queue state.
        # The lock is associated with the condition variable.
        cond = state._cohort_queue_cond
        try:
            await cond.acquire()
            is_inference_request = isinstance(request, InferenceRequest)
            if is_inference_request:
                # --- Use request_id for user requests ---
                if not request.request_id:
                    # Generate a unique ID if not provided
                    request.request_id = f"<auto>_{os.getpid()}_{int(time.time())}_{state._cohort_arrival_ctr}"
                state._cohort_arrival_ctr += 1
                queue_id = request.request_id

                # Check for uniqueness within the target queue
                if any(item[0] == queue_id for item in state._cohort_queues[key]):
                    cond.release()
                    raise AdapterError(f"Request ID '{queue_id}' is already present in the queue for cohort {key}. Request IDs must be unique per cohort.")
            else:
                # --- Fallback to arrival_id for internal calls (e.g., cache) ---
                queue_id = state._cohort_arrival_ctr
                state._cohort_arrival_ctr += 1

            if key not in state._cohort_rotor:
                state._cohort_rotor.append(key)

            qsig = state._cohort_queues[key]

            # If the queue is at capacity, wait on the condition variable until notified.
            while len(qsig) >= state._cohort_sig_cap:
                await cond.wait()

            # Re-check uniqueness after any waits to avoid a race where another
            # enqueuer inserted the same request_id while we released the lock
            # during cond.wait(). Consistent with the earlier check, raise
            # AdapterError if a duplicate is now present.
            if is_inference_request:
                if any(item[0] == queue_id for item in state._cohort_queues[key]):
                    cond.release()
                    raise AdapterError(f"Request ID '{queue_id}' is already present in the queue for cohort {key}. Request IDs must be unique per cohort.")

            cancel_event = threading.Event()
            # Monotonic arrival sequence for fairness across cohorts
            # Use the global arrival counter (already incremented for uniqueness above).
            arrive_seq = state._cohort_arrival_ctr            

            # The tuple now stores the request object (for InferenceRequest) or the
            # task type string (for CohortTaskType).
            # The `is_inference_request` flag distinguishes them.
            # We use a mutable list for user requests so they can be moved to the active map.
            if is_inference_request:
                qsig.append([queue_id, evt, is_inference_request, cancel_event, request, arrive_seq])
            else: # For CohortTaskType or None
                qsig.append((queue_id, evt, is_inference_request, cancel_event, request, arrive_seq))
            
            # If the queue was previously empty, record its new head arrival sequence.
            if len(qsig) == 1:
                state._cohort_head_seq[key] = arrive_seq

            state._cohort_pending_count += 1
            if state._cohort_inflight == 0:
                # If nothing is inflight, bias rotor head to this key so the planner wakes us now.
                state._cohort_current_key = key            

            # PLAN after adopting/rotating so new arrivals can be selected when inflight==0
            sem, to_wake = await self._cohort__plan_releases_unlocked()
        finally:
            # Ensure the lock is always released.
            cond.release()
        # EXECUTE outside the lock
        if sem is not None and to_wake:
            for _aid, e, has_request in to_wake:
                await sem.acquire() # type: ignore
                async with state._cohort_lock:
                    state._cohort_inflight += 1
                    # Check if the woken item is an inference request and increment the specific counter
                    if has_request:
                        state._cohort_active_inference_count += 1
                    e.set()

        # Wait until our own event is set (slot allocated or cancelled).
        await evt.wait()

        # Return the per-request cancel_event so the caller can use it.
        return cancel_event


    async def cohort_leave(self, request: Optional[Union["InferenceRequest", CohortTaskType]] = None) -> None:
        self._cohort_init_if_needed()
        state = self.state
        # PHASE 1 (plan under lock, including rotate-before-plan)
        async with state._cohort_lock:
            # Decrement inflight counter (caller should have been counted when woken)
            state._cohort_inflight -= 1
            # Decrement inference counter only if the request was an inference request
            is_inference_request = isinstance(request, InferenceRequest)
            if is_inference_request:
                state._cohort_active_inference_count -= 1

            # Release the matching semaphore for the cohort kind
            try:
                (await self._cohort__matching_sem_unlocked()).release()
            except Exception:
                # Defensive: if semaphore state is inconsistent, continue and log later
                pass
            # After dropping inflight, we can rotate to a non-empty key and plan
            sem, to_wake = await self._cohort__plan_releases_unlocked()

            # If a request object was provided, remove it from the active map.
            # We raise if the request id is not found among active requests to
            # help detect lifecycle mismatches. Removal is done under lock.
            if is_inference_request:
                req_id = getattr(request, "request_id", None)
                if req_id is None:
                    raise AdapterError("cohort_leave called with request lacking request_id")
                found = False
                # Search active maps for the request id
                for k, active_map in list(state._cohort_active.items()):
                    if req_id in active_map:
                        try:
                            del active_map[req_id]
                        except Exception:
                            pass
                        found = True
                        break
                if not found:
                    # We already updated inflight & released sem; raise to signal mismatch
                    raise AdapterError(f"cohort_leave: active request id '{req_id}' not found in active map")
        # PHASE 2 (execute outside lock)
        if sem is not None and to_wake:
            for _aid, e, has_request in to_wake:
                await sem.acquire() # type: ignore
                async with state._cohort_lock:
                    state._cohort_inflight += 1
                    # Check if the woken item is an inference request and increment the specific counter
                    if has_request: # to_wake is (id, event, has_request, ...)
                        state._cohort_active_inference_count += 1
                    e.set()

    def get_current_active_set_unlocked(self) -> Tuple[str, ...]:
        model = self.state.peft_model
        current_val = getattr(model, "active_adapters", getattr(model, "active_adapter", None))
        if isinstance(current_val, list):
            return _normalize_adapter_set(current_val)
        if current_val:
            if isinstance(current_val, (tuple, set)):
                return _normalize_adapter_set(list(current_val))
            return _normalize_adapter_set([current_val])
        return tuple()

    async def cancel_request(
        self, 
        request_id: Optional[str] = None,
        cancel_ops: Optional[Union[CohortTaskType, List[CohortTaskType]]] = None,
        cancel_for_adapter_name: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Cancels one or all inference requests.
        - If request_id is given, cancels that specific request.
        - If cancel_for_adapter_name is given, cancels requests using that adapter.
        - If cancel_ops is given, cancels pending background tasks of that type.
        - If no filter is given, cancels ALL active and pending requests.
        Returns a tuple of (pending_cancelled, active_cancelled).
        """
        self._cohort_init_if_needed()
        state = self.state
        pending_cancelled = 0
        active_cancelled = 0

        async with state._cohort_queue_cond:
            # --- Optimization: If a specific request_id is given, check active requests first ---
            if request_id:
                for active_map in state._cohort_active.values():
                    if request_id in active_map:
                        entry = active_map[request_id]
                        if len(entry) > 3 and (cancel_event := entry[3]):
                            cancel_event.set()
                            active_cancelled += 1
                        # Since request IDs are unique, we can stop here.
                        self.state.logger.info(f"Cancel processed: {pending_cancelled} pending and {active_cancelled} active requests signalled.")
                        return pending_cancelled, active_cancelled

            # --- Part 1: Cancel Pending Items ---
            # This now handles user requests by ID, adapter name, and background ops by type.
            ops_to_cancel: Optional[set] = None
            if cancel_ops:
                ops_to_cancel = {cancel_ops} if isinstance(cancel_ops, str) else set(cancel_ops)

            for queue_key, queue in list(state._cohort_queues.items()):
                new_queue = deque()
                items_removed = 0
                for entry in queue:
                    # The tuple signature is (id, event, has_request, cancel_event, request_obj_or_task_type, arrive_seq)
                    if not (isinstance(entry, (tuple, list)) and len(entry) >= 6):
                        new_queue.append(entry)
                        continue

                    item_id, event, has_request, entry_cancel_event, task_type_or_req, _ = entry
                    
                    should_cancel = False
                    # Condition 1: Cancel by request_id
                    if has_request and request_id is not None and item_id == request_id:
                        should_cancel = True
                    # Condition 2: Cancel all user requests (if no other filter is active)
                    elif has_request and request_id is None and not cancel_ops and not cancel_for_adapter_name:
                        should_cancel = True
                    # Condition 3: Cancel by operation type
                    elif not has_request and ops_to_cancel and task_type_or_req in ops_to_cancel:
                        should_cancel = True
                    # Condition 4: Cancel pending inference requests for a specific adapter
                    elif has_request and cancel_for_adapter_name:
                        kind, adapters = queue_key
                        if kind == 'F' and cancel_for_adapter_name in adapters:
                            should_cancel = True

                    if should_cancel:
                        if entry_cancel_event:
                            try: entry_cancel_event.set()
                            except Exception: pass
                        if event:
                            try: event.set()
                            except Exception: pass
                        pending_cancelled += 1
                        items_removed += 1
                        # If we found and cancelled a specific request_id, we can stop early.
                        if request_id is not None:
                            # Add remaining items to new_queue and break the inner loop
                            new_queue.extend(list(queue)[queue.index(entry) + 1:])
                            break
                    else:
                        new_queue.append(entry)

                if items_removed > 0:
                    state._cohort_queues[queue_key] = new_queue
                    # Notify any tasks waiting for queue space that slots have freed up.
                    for _ in range(items_removed):
                        state._cohort_queue_cond.notify()

                # If we found the specific request, we can stop scanning other queues.
                if request_id is not None and pending_cancelled > 0:
                    break

            # If we cancelled a specific pending request, we are done.
            if request_id is not None and pending_cancelled > 0:
                self.state.logger.info(f"Cancel processed: {pending_cancelled} pending and {active_cancelled} active requests signalled.")
                return pending_cancelled, active_cancelled

            # --- Part 2: Cancel Active Requests ---
            # This part is now primarily for the "cancel all" case (request_id is None).
            # The single-ID case is handled above.
            # Active (inflight) requests are tracked in state._cohort_active.
            if request_id:
                # This case is handled by the optimization at the top, but kept for clarity.
                for cq_key, active_map in list(state._cohort_active.items()):
                    if request_id in active_map:
                        entry = active_map[request_id]
                        if len(entry) > 3 and (cancel_event := entry[3]):
                            cancel_event.set()
                            active_cancelled += 1
                        break
            elif cancel_for_adapter_name:
                # Cancel active requests for a specific adapter
                for cq_key, active_map in list(state._cohort_active.items()):
                    kind, adapters = cq_key
                    if kind == 'F' and cancel_for_adapter_name in adapters:
                        for entry in list(active_map.values()):
                            if len(entry) > 3 and (cancel_event := entry[3]):
                                cancel_event.set()
                                active_cancelled += 1
            elif not cancel_ops: # Don't cancel all active if just cancelling pending ops
                # Cancel all active requests
                for cq_key, active_map in list(state._cohort_active.items()):
                    for entry in list(active_map.values()):
                        if len(entry) > 3 and (cancel_event := entry[3]):
                            cancel_event.set()
                            active_cancelled += 1
        
        self.state.logger.info(f"Cancel processed: {pending_cancelled} pending and {active_cancelled} active requests signalled.")
        return pending_cancelled, active_cancelled
    
    def get_normalized_adapter_set(self, names: Iterable[Optional[str]]) -> Tuple[str, ...]:
        return _normalize_adapter_set(names)

    async def get_adapter_names(self, root_folder: str, include_incompatible: bool = False, adapter_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Recursively scan a root folder for adapters. Structure may include arbitrary
        intermediate folders before an adapter root (with metadata.json) and its checkpoints.
        Foreign adapters are folders containing adapter_model.safetensors but no metadata.json.
        """
        root_path = Path(root_folder).expanduser()
        try:
            root_path = root_path.resolve()
        except Exception:
            pass

        # When probing a specific adapter, scope scanning to the canonical model/precision/adapter folder.
        scoped_root: Optional[Path] = None
        if adapter_name:
            scoped_root = self.derive_adapter_root_for_storage(adapter_name, root_path)
            root_path = scoped_root

        if not root_path.exists():
            if adapter_name:
                # For probe mode, return a placeholder entry representing a new adapter slot.
                placeholder = {
                    "index": 1,
                    "name": adapter_name,
                    "alias": adapter_name,
                    "is_foreign": False,
                    "is_compatible": True,
                    "is_loaded": False,
                    "is_new": True,
                    "is_diff_checkpoint": None,
                    "is_diff_folder": None,
                    "loaded_name": None,
                    "base_model_name": Path(self.state.base_model_name_or_path).name if self.state.base_model_name_or_path else None,
                    "base_model_quant": self._precision_suffix_from_state(),
                    "path": str(scoped_root) if scoped_root else str(root_folder),
                    "metadata": None,
                }
                return [placeholder]
            raise AdapterError(f"Adapter root folder '{root_folder}' not found.")

        loaded_adapters = await self.state.get_loaded_adapters_info()
        loaded_names: Set[str] = set(loaded_adapters.keys())
        loaded_cp_map: Dict[Path, str] = {}
        loaded_root_map: Dict[Path, str] = {}
        for lname, linfo in loaded_adapters.items():
            cp = linfo.get("checkpoint_path") or linfo.get("root_path")
            rp = linfo.get("root_path")
            if cp:
                try:
                    loaded_cp_map[Path(cp).resolve()] = lname
                except Exception:
                    loaded_cp_map[Path(cp)] = lname
            if rp:
                try:
                    loaded_root_map[Path(rp).resolve()] = lname
                except Exception:
                    loaded_root_map[Path(rp)] = lname
        engine_base_path = self.state.base_model_name_or_path
        engine_base_path_norm = os.path.normpath(engine_base_path).lower() if engine_base_path else None
        engine_base_name = os.path.basename(engine_base_path_norm) if engine_base_path_norm else None

        engine_quant_cfg = self.state.base_model_quantization_cfg or (self.state._effective_quantization_details if hasattr(self.state, "_effective_quantization_details") else {}) or {}
        engine_method = self.state._effective_quantization_method if hasattr(self.state, "_effective_quantization_method") else None
        engine_bits = None
        engine_dtype = None
        if isinstance(engine_quant_cfg, dict):
            for key in ("quant_method", "method", "quant_type", "backend", "name", "load_type"):
                if engine_quant_cfg.get(key):
                    engine_method = engine_quant_cfg.get(key)
                    break
            for key in ("hqq_bits", "awq_bits", "quantize_bits"):
                if engine_quant_cfg.get(key) is not None:
                    try:
                        engine_bits = float(engine_quant_cfg.get(key))
                    except Exception:
                        pass
            if engine_bits is None:
                if engine_quant_cfg.get("load_in_4bit"):
                    engine_bits = 4
                elif engine_quant_cfg.get("load_in_8bit"):
                    engine_bits = 8
        gcfg = self.state._global_config or {}
        engine_dtype = gcfg.get("effective_torch_dtype") or self.state.requested_torch_dtype

        def _read_metadata(path: Path) -> Optional[Dict[str, Any]]:
            if not path.is_file():
                return None
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.state.logger.warning(f"[ADAPTER-LIST] Failed reading metadata at {path}: {e}")
                return None

        def _extract_quant(meta: Optional[Dict[str, Any]]) -> Optional[str]:
            if not meta:
                return None
            precision_info = meta.get("precision_info") or {}
            quant_cfg = None
            if isinstance(precision_info, dict):
                quant_cfg = precision_info.get("base_model_quantization_config") or precision_info.get("quantization_config")
            if isinstance(quant_cfg, dict):
                if quant_cfg.get("quantize_bits"):
                    return str(quant_cfg.get("quantize_bits"))
                for key in ("quant_method", "method", "quant_type", "backend", "name", "load_type"):
                    val = quant_cfg.get(key)
                    if val:
                        return str(val)
                for key in ("bnb_4bit_quant_type",):
                    val = quant_cfg.get(key)
                    if val:
                        return str(val)
                if quant_cfg.get("load_in_4bit"):
                    return "4bit"
                if quant_cfg.get("load_in_8bit"):
                    return "8bit"
            if isinstance(precision_info, dict):
                for key in ("dtype", "precision", "format", "base_model_effective_dtype_at_init"):
                    val = precision_info.get(key)
                    if val:
                        return str(val)
            # Fallback: saved base model dtype at training time
            dtype_at_init = meta.get("base_model_effective_dtype_at_init")
            if dtype_at_init:
                return str(dtype_at_init)
            return None

        def _quant_signature(meta: Optional[Dict[str, Any]], quant_str: Optional[str]) -> Tuple[Optional[str], Optional[float], Optional[str]]:
            method = None
            bits = None
            dtype = None
            if meta:
                dtype = meta.get("base_model_effective_dtype_at_init")
                precision_info = meta.get("precision_info") or {}
                quant_cfg = precision_info.get("base_model_quantization_config") or precision_info.get("quantization_config") or {}
                if isinstance(quant_cfg, dict):
                    for key in ("quant_method", "method", "quant_type", "backend", "name", "load_type"):
                        if quant_cfg.get(key):
                            method = str(quant_cfg.get(key))
                            break
                    for key in ("hqq_bits", "awq_bits", "quantize_bits"):
                        if quant_cfg.get(key) is not None:
                            try:
                                bits = float(quant_cfg.get(key))
                            except Exception:
                                pass
                    if bits is None:
                        if quant_cfg.get("load_in_4bit"):
                            bits = 4
                        elif quant_cfg.get("load_in_8bit"):
                            bits = 8
            if quant_str:
                qs = str(quant_str).lower()
                if method is None:
                    if "hqq" in qs:
                        method = "hqq"
                    elif "awq" in qs:
                        method = "awq"
                    elif "bnb" in qs:
                        method = "bnb"
                if bits is None:
                    import re
                    m = re.search(r"(\d+)\s*bit", qs)
                    if m:
                        try:
                            bits = float(m.group(1))
                        except Exception:
                            pass
            return method, bits, dtype

        def _compatibility_flags(meta: Optional[Dict[str, Any]], quant_method: Optional[str]) -> Tuple[Optional[bool], Tuple[int, float, float, float], Optional[str]]:
            meta_base_raw = (meta or {}).get("base_model_name_or_path")
            meta_base_norm = os.path.normpath(meta_base_raw).lower() if meta_base_raw else None
            meta_base_name = os.path.basename(meta_base_norm) if meta_base_norm else None
            path_match = bool(engine_base_path_norm and meta_base_norm and meta_base_norm == engine_base_path_norm)
            name_match = bool(engine_base_name and meta_base_name and meta_base_name == engine_base_name)
            if meta is None:
                return None, (99, float("inf"), float("inf"), 0.0), None
            is_compatible = path_match or name_match
            adapter_method, adapter_bits, adapter_dtype = _quant_signature(meta, quant_method)

            # Quant matching score: prefer same method, then closest/higher bits.
            method_penalty = 0 if engine_method and adapter_method and str(engine_method).lower() == str(adapter_method).lower() else 1
            if engine_method is None or adapter_method is None:
                method_penalty = 0 if engine_method == adapter_method else 1

            if engine_bits is not None and adapter_bits is not None:
                bits_distance = abs(adapter_bits - engine_bits)
            else:
                bits_distance = float("inf")
            bits_prefer = -(adapter_bits or 0.0)

            compat_bucket = 0 if path_match else (1 if name_match else 2)
            compat_rank = (compat_bucket, method_penalty, bits_distance if bits_distance != float("inf") else 9999.0, bits_prefer)

            base_display = meta_base_name or meta_base_raw
            return is_compatible, compat_rank, base_display

        def _path_time(path: Path) -> float:
            """
            Return best-effort ordering timestamp:
            1) metadata.json mtime if present.
            2) adapter_model.safetensors mtime if present.
            3) filesystem mtime, then ctime.
            """
            try:
                meta_path = path / "metadata.json"
                if meta_path.is_file():
                    return meta_path.stat().st_mtime

                tensor_path = path / "adapter_model.safetensors" if path.is_dir() else path
                if tensor_path.exists():
                    return tensor_path.stat().st_mtime
                stat = path.stat()
                return stat.st_mtime or stat.st_ctime
            except Exception:
                return 0.0

        def _checkpoint_sort_key(p: Path) -> Tuple[int, Any, float]:
            name = p.name
            if name.startswith("checkpoint-"):
                try:
                    return (3, int(name.split("-")[-1]), _path_time(p))
                except Exception:
                    pass
            is_timestamp = False
            if len(name) > 15 and name[8:9] == "-" and name[15:16] == "-":
                date_part = name[0:8]
                time_part = name[9:15]
                if date_part.isdigit() and time_part.isdigit():
                    is_timestamp = True
            if is_timestamp:
                return (2, name, _path_time(p))
            return (1, name, _path_time(p))

        def _collect_checkpoint_dirs(adapter_root: Path) -> List[Path]:
            checkpoints: List[Path] = []
            try:
                for item in adapter_root.iterdir():
                    if item.is_dir() and (item / "adapter_model.safetensors").is_file():
                        checkpoints.append(item)
            except FileNotFoundError:
                pass
            if (adapter_root / "adapter_model.safetensors").is_file():
                checkpoints.append(adapter_root)
            deduped: List[Path] = []
            seen: Set[Path] = set()
            for p in checkpoints:
                try:
                    rp = p.resolve()
                except Exception:
                    rp = p
                if rp in seen:
                    continue
                seen.add(rp)
                deduped.append(p)
            deduped.sort(key=_checkpoint_sort_key, reverse=True)
            return deduped

        def _looks_like_checkpoint_name(name: str) -> bool:
            if name.startswith("checkpoint-"):
                return True
            if len(name) > 15 and name[8:9] == "-" and name[15:16] == "-":
                date_part = name[0:8]
                time_part = name[9:15]
                if date_part.isdigit() and time_part.isdigit():
                    return True
            return False

        def _infer_adapter_root_for_checkpoint(checkpoint_path: Path) -> Path:
            if not checkpoint_path.is_dir():
                return checkpoint_path
            parent = checkpoint_path.parent
            try:
                if (parent / "metadata.json").is_file():
                    return parent
            except Exception:
                return checkpoint_path
            if parent == root_path:
                return checkpoint_path
            if not _looks_like_checkpoint_name(checkpoint_path.name):
                return checkpoint_path
            sibling_checkpoint_like = 0
            try:
                for child in parent.iterdir():
                    if child.is_dir() and (child / "adapter_model.safetensors").is_file():
                        if _looks_like_checkpoint_name(child.name):
                            sibling_checkpoint_like += 1
                            if sibling_checkpoint_like >= 2:
                                return parent
            except Exception:
                pass
            return checkpoint_path

        def _build_entry(
            adapter_root: Path,
            checkpoint_path: Optional[Path],
            metadata: Optional[Dict[str, Any]],
            is_foreign: bool
        ) -> Tuple[int, Dict[str, Any]]:
            adapter_name = (metadata or {}).get("adapter_name")
            alias = checkpoint_path.name if checkpoint_path else adapter_root.name

            quant_method = _extract_quant(metadata)
            quant_display = quant_display_from_meta(metadata, quant_method)
            if not quant_method and metadata:
                quant_method = metadata.get("base_model_effective_dtype_at_init")
            if is_foreign:
                is_compatible = None
                compat_rank = (99, float("inf"), float("inf"), 0.0)
                base_model_display = None
            else:
                is_compatible, compat_rank, base_model_display = _compatibility_flags(metadata, quant_method)

            loaded_name: Optional[str] = None
            resolved_loaded_cp = None
            resolved_loaded_root = None
            loaded_info = {}
            # Prefer matching by path to recover internal name even if metadata names differ.
            if checkpoint_path:
                try:
                    resolved_loaded_cp = checkpoint_path.resolve()
                except Exception:
                    resolved_loaded_cp = checkpoint_path
                loaded_name = loaded_cp_map.get(resolved_loaded_cp)
            if not loaded_name:
                try:
                    resolved_loaded_root = adapter_root.resolve()
                except Exception:
                    resolved_loaded_root = adapter_root
                loaded_name = loaded_root_map.get(resolved_loaded_root)
            if loaded_name:
                loaded_info = loaded_adapters.get(loaded_name, {})
                lp_cp = loaded_info.get("checkpoint_path") or loaded_info.get("root_path")
                lp_root = loaded_info.get("root_path")
                if lp_cp:
                    try:
                        resolved_loaded_cp = Path(lp_cp).resolve()
                    except Exception:
                        resolved_loaded_cp = Path(lp_cp)
                if lp_root:
                    try:
                        resolved_loaded_root = Path(lp_root).resolve()
                    except Exception:
                        resolved_loaded_root = Path(lp_root)
            entry_cp_path = checkpoint_path or None
            entry_root_path = adapter_root
            if entry_cp_path:
                entry_root_path = _infer_adapter_root_for_checkpoint(entry_cp_path)

            is_loaded = False
            if loaded_name:
                # Treat as loaded only when the checkpoint/root matches the loaded one.
                if entry_cp_path and resolved_loaded_cp:
                    try:
                        is_loaded = entry_cp_path.resolve() == resolved_loaded_cp
                    except Exception:
                        is_loaded = str(entry_cp_path) == str(resolved_loaded_cp)
                elif not entry_cp_path and resolved_loaded_root:
                    try:
                        is_loaded = entry_root_path.resolve() == resolved_loaded_root
                    except Exception:
                        is_loaded = str(entry_root_path) == str(resolved_loaded_root)

            is_diff_checkpoint: Optional[bool]
            if loaded_name and entry_cp_path and resolved_loaded_cp:
                try:
                    is_diff_checkpoint = entry_cp_path.resolve() != resolved_loaded_cp
                except Exception:
                    is_diff_checkpoint = str(entry_cp_path) != str(resolved_loaded_cp)
            else:
                is_diff_checkpoint = None

            is_diff_folder: Optional[bool]
            if loaded_name and entry_root_path and resolved_loaded_root:
                try:
                    is_diff_folder = entry_root_path.resolve() != resolved_loaded_root
                except Exception:
                    is_diff_folder = str(entry_root_path) != str(resolved_loaded_root)
            else:
                is_diff_folder = None

            if not checkpoint_path:
                has_ckpt = find_latest_checkpoint_in_dir(adapter_root)
                is_new = has_ckpt is None
            else:
                is_new = False

            entry = {
                "index": 0,
                "name": loaded_name or adapter_name,
                "alias": alias,
                "is_foreign": is_foreign,
                "is_compatible": is_compatible if not is_foreign else None,
                "is_loaded": is_loaded,
                "is_new": is_new,
                "is_diff_checkpoint": is_diff_checkpoint,
                "is_diff_folder": is_diff_folder,
                "loaded_name": loaded_name,
                "base_model_name": base_model_display,
                "base_model_quant": quant_display,
                "root_path": str(entry_root_path),
                "checkpoint_path": str(entry_cp_path) if entry_cp_path else None,
                "path": str(checkpoint_path or adapter_root),
                "metadata": metadata
            }
            return compat_rank, entry

        entries: List[Tuple[int, Dict[str, Any]]] = []
        stack: List[Path] = [root_path]
        while stack:
            current = stack.pop()
            if not current.is_dir():
                continue

            meta = _read_metadata(current / "metadata.json")
            has_meta = meta is not None
            has_weights_here = (current / "adapter_model.safetensors").is_file()

            if has_meta:
                checkpoints = _collect_checkpoint_dirs(current)
                if checkpoints:
                    for cp in checkpoints:
                        cp_meta = _read_metadata(cp / "metadata.json") if cp != current else None
                        merged_meta = dict(meta or {})
                        if cp_meta:
                            merged_meta.update(cp_meta)
                        entries.append(_build_entry(current, cp if cp != current else current if has_weights_here else None, merged_meta if merged_meta else None, is_foreign=False))
                else:
                    entries.append(_build_entry(current, None, meta, is_foreign=False))
                # Continue scanning deeper for other adapters that may exist under this root
                try:
                    checkpoint_set = set(checkpoints)
                    for child in current.iterdir():
                        if not child.is_dir():
                            continue
                        if child in checkpoint_set:
                            continue
                        # Skip pure checkpoint folders already accounted for
                        if (child / "adapter_model.safetensors").is_file() and not (child / "metadata.json").is_file():
                            continue
                        stack.append(child)
                except FileNotFoundError:
                    pass
            elif has_weights_here:
                entries.append(_build_entry(current, current, None, is_foreign=True))
                try:
                    for child in current.iterdir():
                        if child.is_dir():
                            stack.append(child)
                except FileNotFoundError:
                    pass
            else:
                try:
                    for child in current.iterdir():
                        if child.is_dir():
                            stack.append(child)
                except FileNotFoundError:
                    continue

        proper_precision_dir = self.build_model_precision_dir_name()

        def _sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int, Any, int, float, str]:
            compat_rank, entry = item
            is_foreign = entry.get("is_foreign")
            compat_flag = entry.get("is_compatible")
            # Order: compatible (including unknown/None) -> foreign -> explicitly incompatible
            if not is_foreign and (compat_flag or compat_flag is None):
                group = 0
            elif is_foreign:
                group = 1
            else:
                group = 2

            # Tie-breaker 1: Give priority to paths within the "proper" precision directory.
            path_penalty = 1
            try:
                if proper_precision_dir in Path(entry["path"]).parts:
                    path_penalty = 0
            except Exception:
                pass  # Ignore path parsing errors

            # Tie-breaker 2: Penalize paths containing "hidden" or "saved" directory segments.
            hidden_penalty = 0
            try:
                path_parts = Path(entry["path"]).parts
                if any(part.startswith('_') or part.startswith('.') for part in path_parts):
                    hidden_penalty = 1
            except Exception:
                pass  # Ignore path parsing errors

            mtime = _path_time(Path(entry["path"]))
            name_key = entry["alias"] or entry["name"] or ""
            return (group, path_penalty, compat_rank, hidden_penalty, -mtime, name_key.lower())

        entries.sort(key=_sort_key)

        result: List[Dict[str, Any]] = []
        for idx, (_, entry) in enumerate(entries, start=1):
            entry["index"] = idx
            result.append(entry)
        if include_incompatible:
            return result
        filtered: List[Dict[str, Any]] = []
        for item in result:
            compat_flag = item.get("is_compatible")
            # Treat None as "maybe compatible" and include foreign entries by default;
            # only drop items explicitly marked incompatible.
            if compat_flag is False:
                continue
            filtered.append(item)
        if adapter_name and not filtered:
            placeholder = {
                "index": 1,
                "name": adapter_name,
                "alias": adapter_name,
                "is_foreign": False,
                "is_compatible": True,
                "is_loaded": False,
                "is_new": True,
                "is_diff_checkpoint": None,
                "is_diff_folder": None,
                "loaded_name": None,
                "base_model_name": Path(self.state.base_model_name_or_path).name if self.state.base_model_name_or_path else None,
                "base_model_quant": self._precision_suffix_from_state(),
                "path": str(scoped_root) if scoped_root else str(root_folder),
                "metadata": None,
            }
            filtered.append(placeholder)
        return filtered

    async def load_adapter(self, adapter_config: AdapterConfig):
        """Add an adapter to the engine."""
        # adapter_config.adapter_path is the primary input for path resolution.
        # adapter_config.adapter_name from API is an override or name for new adapter.
        # adapter_config.adapter_type and other LoRA params are for NEW adapters.
        
        if not adapter_config.adapter_path:
            raise ConfigurationError("`adapter_path` is mandatory for load_adapter operation.") # noqa: E501

        target_model: Optional[Union[PeftModel, PeftMixedModel]] = None
        original_training_state = False # Default

        input_path = Path(adapter_config.adapter_path).expanduser()
        try:
            input_path = input_path.resolve()
        except Exception:
            pass

        requested_name_raw = adapter_config.adapter_name
        last_segment = input_path.name
        last_segment_norm = last_segment.replace(".", "_")
        force_new_adapter = bool(getattr(adapter_config, "is_new", False))

        def _precision_suffix() -> str:
            return self._precision_suffix_from_state()

        def _build_precision_dir_name() -> str:
            return self.build_model_precision_dir_name()

        # Determine new vs existing scenario
        subdirs = []
        has_files = False
        if input_path.exists() and input_path.is_dir():
            try:
                entries = list(input_path.iterdir())
                subdirs = [p for p in entries if p.is_dir()]
                has_files = any(p.is_file() for p in entries)
            except FileNotFoundError:
                subdirs = []
        is_new_scenario = force_new_adapter or (not input_path.exists()) or (input_path.is_dir() and len(subdirs) == 0 and not has_files)

        effective_adapter_name_raw = requested_name_raw or last_segment
        effective_adapter_name = effective_adapter_name_raw.replace(".", "_")

        engine_base_path = self.state.base_model_name_or_path
        engine_base_path_norm = os.path.normpath(engine_base_path).lower() if engine_base_path else None
        engine_base_name = os.path.basename(engine_base_path_norm) if engine_base_path_norm else None
        engine_quant_cfg = self.state.base_model_quantization_cfg or (self.state._effective_quantization_details if hasattr(self.state, "_effective_quantization_details") else {}) or {}
        engine_method = self.state._effective_quantization_method if hasattr(self.state, "_effective_quantization_method") else None
        engine_bits = None
        if isinstance(engine_quant_cfg, dict):
            for key in ("quant_method", "method", "quant_type", "backend", "name", "load_type"):
                if engine_quant_cfg.get(key):
                    engine_method = engine_quant_cfg.get(key)
                    break
            for key in ("hqq_bits", "awq_bits", "quantize_bits"):
                if engine_quant_cfg.get(key) is not None:
                    try:
                        engine_bits = float(engine_quant_cfg.get(key))
                    except Exception:
                        pass
            if engine_bits is None:
                if engine_quant_cfg.get("load_in_4bit"):
                    engine_bits = 4
                elif engine_quant_cfg.get("load_in_8bit"):
                    engine_bits = 8

        def _extract_quant_local(meta: Optional[Dict[str, Any]]) -> Optional[str]:
            if not meta:
                return None
            precision_info = meta.get("precision_info") or {}
            quant_cfg = None
            if isinstance(precision_info, dict):
                quant_cfg = precision_info.get("base_model_quantization_config") or precision_info.get("quantization_config")
            if isinstance(quant_cfg, dict):
                if quant_cfg.get("quantize_bits"):
                    return str(quant_cfg.get("quantize_bits"))
                for key in ("quant_method", "method", "quant_type", "backend", "name", "load_type"):
                    val = quant_cfg.get(key)
                    if val:
                        return str(val)
                for key in ("bnb_4bit_quant_type",):
                    val = quant_cfg.get(key)
                    if val:
                        return str(val)
                if quant_cfg.get("load_in_4bit"):
                    return "4bit"
                if quant_cfg.get("load_in_8bit"):
                    return "8bit"
            if isinstance(precision_info, dict):
                for key in ("dtype", "precision", "format", "base_model_effective_dtype_at_init"):
                    val = precision_info.get(key)
                    if val:
                        return str(val)
            return None

        def _quant_signature(meta: Optional[Dict[str, Any]], quant_str: Optional[str]) -> Tuple[Optional[str], Optional[float], Optional[str]]:
            method = None
            bits = None
            dtype = None
            if meta:
                dtype = meta.get("base_model_effective_dtype_at_init")
                precision_info = meta.get("precision_info") or {}
                quant_cfg = precision_info.get("base_model_quantization_config") or precision_info.get("quantization_config") or {}
                if isinstance(quant_cfg, dict):
                    for key in ("quant_method", "method", "quant_type", "backend", "name", "load_type"):
                        if quant_cfg.get(key):
                            method = str(quant_cfg.get(key))
                            break
                    for key in ("hqq_bits", "awq_bits", "quantize_bits"):
                        if quant_cfg.get(key) is not None:
                            try:
                                bits = float(quant_cfg.get(key))
                            except Exception:
                                pass
                    if bits is None:
                        if quant_cfg.get("load_in_4bit"):
                            bits = 4
                        elif quant_cfg.get("load_in_8bit"):
                            bits = 8
            if quant_str:
                qs = str(quant_str).lower()
                if method is None:
                    if "hqq" in qs:
                        method = "hqq"
                    elif "awq" in qs:
                        method = "awq"
                    elif "bnb" in qs:
                        method = "bnb"
                if bits is None:
                    import re
                    m = re.search(r"(\\d+)\\s*bit", qs)
                    if m:
                        try:
                            bits = float(m.group(1))
                        except Exception:
                            pass
            return method, bits, dtype

        def _compatibility_flags(meta: Optional[Dict[str, Any]], quant_method: Optional[str]) -> Tuple[Optional[bool], Tuple[int, float, float, float], Optional[str]]:
            meta_base_raw = (meta or {}).get("base_model_name_or_path")
            meta_base_norm = os.path.normpath(meta_base_raw).lower() if meta_base_raw else None
            meta_base_name = os.path.basename(meta_base_norm) if meta_base_norm else None
            path_match = bool(engine_base_path_norm and meta_base_norm and meta_base_norm == engine_base_path_norm)
            name_match = bool(engine_base_name and meta_base_name and meta_base_name == engine_base_name)
            if meta is None:
                return None, (99, float("inf"), float("inf"), 0.0), None
            is_compatible = path_match or name_match
            adapter_method, adapter_bits, adapter_dtype = _quant_signature(meta, quant_method)

            # Quant matching score: prefer same method, then closest/higher bits.
            method_penalty = 0 if engine_method and adapter_method and str(engine_method).lower() == str(adapter_method).lower() else 1
            if engine_method is None or adapter_method is None:
                method_penalty = 0 if engine_method == adapter_method else 1

            if engine_bits is not None and adapter_bits is not None:
                bits_distance = abs(adapter_bits - engine_bits)
            else:
                bits_distance = float("inf")
            bits_prefer = -(adapter_bits or 0.0)

            compat_bucket = 0 if path_match else (1 if name_match else 2)
            compat_rank = (compat_bucket, method_penalty, bits_distance if bits_distance != float("inf") else 9999.0, bits_prefer)

            base_display = meta_base_name or meta_base_raw
            return is_compatible, (compat_rank[0], method_penalty, bits_distance if bits_distance != float("inf") else 9999.0, bits_prefer), base_display

        def _strip_config_from_meta(meta: Dict[str, Any], cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
            """
            Return a copy of metadata with PEFT config keys removed when they match
            the adapter config stored separately. Keeps true metadata (e.g., precision_info).
            """
            if not meta:
                return {}
            cleaned = dict(meta)
            for k in list(cleaned.keys()):
                if k in cfg_dict and cleaned.get(k) == cfg_dict.get(k):
                    cleaned.pop(k, None)
            return cleaned

        def _enrich_meta_with_config(meta: Dict[str, Any], cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
            """
            Ensure metadata carries a lora_config block with target_modules, falling back to
            the stored adapter config if missing or null.
            """
            meta = dict(meta) if meta else {}
            lora_cfg = meta.get("lora_config") or {}
            if not lora_cfg.get("target_modules") and cfg_dict.get("target_modules"):
                tm = cfg_dict.get("target_modules")
                if isinstance(tm, (set, tuple)):
                    tm = list(tm)
                lora_cfg = dict(lora_cfg)
                lora_cfg["target_modules"] = tm
                if "r" not in lora_cfg and cfg_dict.get("r") is not None:
                    lora_cfg["r"] = cfg_dict.get("r")
                if "alpha" not in lora_cfg and cfg_dict.get("lora_alpha") is not None:
                    lora_cfg["alpha"] = cfg_dict.get("lora_alpha")
                if "dropout" not in lora_cfg and cfg_dict.get("lora_dropout") is not None:
                    lora_cfg["dropout"] = cfg_dict.get("lora_dropout")
                meta["lora_config"] = lora_cfg
            return meta

        # NOTE: Heavy I/O and validation happen before the cohort gate so bad/slow inputs
        # fail fast without blocking other model mutations.
        entered_cohort = False
        evt: Optional[threading.Event] = None
        try:
            self.state.logger.info(f"--- Adding Adapter: API Path='{adapter_config.adapter_path}', API Name='{adapter_config.adapter_name}', API Type='{adapter_config.adapter_type.value if adapter_config.adapter_type else 'N/A'}' ---")

            if self.state.base_model is None:
                raise EngineError("Base model not initialized. Cannot add adapter.")

            adapter_root_path: Path
            actual_checkpoint_path_to_load: Optional[Path] = None
            is_new_adapter_creation = False
            metadata_from_disk: Dict[str, Any] = {}
            source_is_foreign = False
            assumption_context = ""
            metadata_candidate: Optional[Dict[str, Any]] = None

            if is_new_scenario:
                # New adapter request
                if force_new_adapter:
                    adapter_root_path = self.derive_adapter_root_for_storage(effective_adapter_name, input_path)
                else:
                    adapter_root_path = input_path
                    # If a different adapter name was provided, build a structured path <model.prec>/<adapter>
                    if requested_name_raw and requested_name_raw.replace(".", "_") != last_segment_norm:
                        adapter_root_path = input_path / _build_precision_dir_name() / requested_name_raw
                effective_adapter_name_raw_local = effective_adapter_name_raw
                effective_adapter_name = effective_adapter_name_raw_local.replace(".", "_")
                is_new_adapter_creation = True
                await asyncio.to_thread(os.makedirs, adapter_root_path, exist_ok=True)
                if force_new_adapter:
                    assumption_context = f"forced new adapter creation at '{adapter_root_path}'"
                else:
                    assumption_context = f"new adapter creation at '{adapter_root_path}'"
            else:
                # Existing adapter: find checkpoint/metadata
                metadata_candidate: Optional[Dict[str, Any]] = None
                if input_path.is_dir() and (input_path / "adapter_model.safetensors").is_file():
                    adapter_root_path = input_path
                    metadata_candidate = {}
                    root_meta = (input_path / "metadata.json")
                    if root_meta.is_file():
                        with open(root_meta, "r") as f:
                            metadata_candidate.update(json.load(f))
                    actual_checkpoint_path_to_load = input_path
                    if not metadata_candidate or "adapter_name" not in metadata_candidate:
                        source_is_foreign = True
                        effective_adapter_name_raw_local = requested_name_raw or last_segment
                        effective_adapter_name = effective_adapter_name_raw_local.replace(".", "_")
                        assumption_context = f"existing adapter (foreign/missing metadata) direct load at '{actual_checkpoint_path_to_load}'"
                    else:
                        effective_adapter_name_raw_local = metadata_candidate.get("adapter_name", effective_adapter_name_raw)
                        effective_adapter_name = effective_adapter_name_raw_local.replace(".", "_")
                        assumption_context = f"existing adapter direct load at '{actual_checkpoint_path_to_load}'"
                else:
                    requested_norm = requested_name_raw.replace(".", "_") if requested_name_raw else None
                    discovered = await self.get_adapter_names(
                        str(input_path),
                        adapter_name=requested_name_raw if requested_name_raw else None,
                    )
                    candidates = discovered
                    if requested_norm:
                        def _meta_name(entry: Dict[str, Any]) -> Optional[str]:
                            meta = entry.get("metadata") or {}
                            return meta.get("adapter_name") if isinstance(meta, dict) else None

                        matching_meta = []
                        for entry in discovered:
                            meta_name = _meta_name(entry)
                            if meta_name and meta_name.replace(".", "_") == requested_norm:
                                matching_meta.append(entry)
                        if matching_meta:
                            candidates = matching_meta
                    first_viable = next((e for e in candidates if e.get("is_compatible") is not False), None)
                    if not first_viable:
                        raise AdapterError(f"No compatible adapter found while scanning '{input_path}'. Assumption: existing adapter selection.")
                    actual_checkpoint_path_to_load = Path(first_viable["path"])
                    adapter_root_path = actual_checkpoint_path_to_load.parent if first_viable.get("checkpoint_path") else actual_checkpoint_path_to_load
                    metadata_candidate = first_viable.get("metadata") or {}
                    source_is_foreign = bool(first_viable.get("is_foreign"))
                    if source_is_foreign:
                        effective_adapter_name_raw = first_viable.get("alias") or first_viable.get("name") or last_segment
                    else:
                        effective_adapter_name_raw = requested_name_raw or first_viable.get("name") or last_segment
                    effective_adapter_name = effective_adapter_name_raw.replace(".", "_")
                    is_new_adapter_creation = False
                    assumption_context = f"existing adapter selection at '{actual_checkpoint_path_to_load}' (scan root '{input_path}')"

            metadata_from_disk.update(metadata_candidate or {})

            # Precompute quant info for downstream display/compatibility.
            quant_method = _extract_quant_local(metadata_from_disk)
            quant_display = quant_display_from_meta(metadata_from_disk, quant_method)
            is_compatible_precomputed, _, base_display_precomputed = _compatibility_flags(metadata_from_disk, quant_method)
            base_display = base_display_precomputed
            if not quant_display and quant_method:
                quant_display = str(quant_method)

            try:
                loaded_info_all = await self.state.get_loaded_adapters_info()
            except Exception:
                loaded_info_all = {}

            # Handle if_exists policy
            existing_adapter_name_by_path = None
            existing_adapter_info_by_path = None
            if not is_new_scenario:
                for name, info in loaded_info_all.items():
                    existing_cp = info.get("checkpoint_path")
                    existing_root = info.get("root_path")
                    is_same = False
                    if actual_checkpoint_path_to_load and existing_cp:
                        try:
                            is_same = Path(actual_checkpoint_path_to_load).resolve() == Path(existing_cp).resolve()
                        except Exception:
                            is_same = str(actual_checkpoint_path_to_load) == str(existing_cp)
                    elif adapter_root_path and existing_root and not actual_checkpoint_path_to_load and not existing_cp:
                        try:
                            is_same = Path(adapter_root_path).resolve() == Path(existing_root).resolve()
                        except Exception:
                            is_same = str(adapter_root_path) == str(existing_root)

                    if is_same:
                        existing_adapter_name_by_path = name
                        existing_adapter_info_by_path = info
                        break
            
            perform_reload_unload = False
            adapter_to_reload = None
            if existing_adapter_name_by_path and existing_adapter_info_by_path:
                self.state.logger.info(f"Adapter from path '{adapter_config.adapter_path}' is already loaded as '{existing_adapter_name_by_path}'. Handling with if_exists='{adapter_config.if_exists.value}'.")
    
                if adapter_config.if_exists == IfExistsEnum.FAIL:
                    raise AdapterError(f"Adapter from path '{adapter_config.adapter_path}' is already loaded as '{existing_adapter_name_by_path}'. Use if_exists='ignore' or 'reload'.")
    
                if adapter_config.if_exists == IfExistsEnum.IGNORE:
                    return {
                        "adapter_name": existing_adapter_name_by_path,
                        "alias": existing_adapter_info_by_path.get("alias"),
                        "is_loaded": True,
                        "is_new": False,
                        "path": existing_adapter_info_by_path.get("checkpoint_path") or existing_adapter_info_by_path.get("root_path"),
                        "metadata": existing_adapter_info_by_path.get("metadata"),
                        "adapter_type": existing_adapter_info_by_path.get("type"),
                    }

                if adapter_config.if_exists == IfExistsEnum.RELOAD:
                    perform_reload_unload = True
                    adapter_to_reload = existing_adapter_name_by_path
                    self.state.logger.info(f"Reload for '{adapter_to_reload}' has priority. Signalling conflicting requests to cancel.")
                    await self.cancel_request(cancel_for_adapter_name=adapter_to_reload)

            # If another adapter with the same name is already loaded and this one is compatible,
            # assign a unique internal name with a numeric suffix.
            if effective_adapter_name in loaded_info_all and (is_compatible_precomputed or is_compatible_precomputed is None):
                base_name = effective_adapter_name
                counter = 2
                chosen = base_name
                # Avoid suffix if the same checkpoint/root is already loaded.
                existing = loaded_info_all.get(base_name, {})
                existing_cp = existing.get("checkpoint_path") or existing.get("root_path")
                existing_root = existing.get("root_path")
                already_same = False
                if actual_checkpoint_path_to_load and existing_cp:
                    try:
                        already_same = Path(actual_checkpoint_path_to_load).resolve() == Path(existing_cp).resolve()
                    except Exception:
                        already_same = str(actual_checkpoint_path_to_load) == str(existing_cp)
                elif adapter_root_path and existing_root:
                    try:
                        already_same = Path(adapter_root_path).resolve() == Path(existing_root).resolve()
                    except Exception:
                        already_same = str(adapter_root_path) == str(existing_root)
                if not already_same:
                    while chosen in loaded_info_all:
                        chosen = f"{base_name}_{counter}"
                        counter += 1
                    self.state.logger.info(f"Adapter name '{base_name}' already loaded; assigning internal name '{chosen}'.")
                    effective_adapter_name = chosen

            # --- Cache Invalidation Logic on Add ---
    #TBD
    #        async with self.state._lock:
    #            # Add to the list of new adapters for this session if it's a new creation
    #            if is_new_adapter_creation and effective_adapter_name not in self.state._new_adapters_added_in_session:
    #                self.state._new_adapters_added_in_session.append(effective_adapter_name)
    #
    #            last_unloaded = self.state._last_unloaded_adapter_name
    #            # Invalidate if we are loading a different adapter than the one we just unloaded.
    #            # This prevents invalidation if we are simply reloading the same adapter.
    #            if last_unloaded is not None and last_unloaded != effective_adapter_name: # type: ignore
    #                self.state.logger.info(f"[cache-invalidate] Loading adapter '{effective_adapter_name}' which differs from last unloaded ('{last_unloaded}'). Invalidating adapter cache signatures.")
    #                invalidate_warmed_signatures_except_base()
    #            
    #            # Add to load order and clear last unloaded tracker
    #            if effective_adapter_name not in self.state._adapter_load_order:
    #                self.state._adapter_load_order.append(effective_adapter_name)
    #            self.state._last_unloaded_adapter_name = None
    #        # --- End Cache Invalidation ---

            # Determine effective_adapter_type
            effective_adapter_type: Optional[AdapterType] = None

            if not is_new_adapter_creation:
                # Loading existing adapter: type and other metadata come from disk.
                peft_config_from_disk: Optional[Dict[str, Any]] = None
                if actual_checkpoint_path_to_load and (actual_checkpoint_path_to_load / "adapter_config.json").is_file():
                    with open(actual_checkpoint_path_to_load / "adapter_config.json", 'r') as f:
                        peft_config_from_disk = json.load(f)

                root_meta_path = adapter_root_path / "metadata.json"
                if root_meta_path.is_file():
                    with open(root_meta_path, 'r') as f:
                        root_meta = json.load(f)
                        metadata_from_disk.update(root_meta)

                if actual_checkpoint_path_to_load:
                    checkpoint_meta_path = actual_checkpoint_path_to_load / "metadata.json"
                    if checkpoint_meta_path.is_file():
                        with open(checkpoint_meta_path, 'r') as f:
                            checkpoint_meta = json.load(f)
                            metadata_from_disk.update(checkpoint_meta)
                            self.state.logger.info(f"Loaded overriding metadata from checkpoint directory: {checkpoint_meta_path}")

                peft_type_str = metadata_from_disk.get("peft_type") or (peft_config_from_disk or {}).get("peft_type")
                custom_type_str_from_meta = metadata_from_disk.get("adapter_type")

                if custom_type_str_from_meta:
                    try:
                        effective_adapter_type = AdapterType(custom_type_str_from_meta.lower())
                    except ValueError:
                        raise ConfigurationError(f"Unsupported adapter_type '{custom_type_str_from_meta}' found while loading adapter '{effective_adapter_name}' at '{actual_checkpoint_path_to_load}'.")
                elif peft_type_str:
                    try:
                        effective_adapter_type = AdapterType[peft_type_str.upper()]
                    except KeyError:
                        self.state.logger.warning(f"Warning: Unknown peft_type '{peft_type_str}' in adapter_config.json for '{effective_adapter_name}'")

                if not effective_adapter_type:
                    raise ConfigurationError(f"Could not determine adapter type for existing adapter '{effective_adapter_name}' at '{actual_checkpoint_path_to_load}'.")
                self.state.logger.info(f"Loading existing adapter '{effective_adapter_name}'. Effective type from metadata: {effective_adapter_type.value}")

            else: # New adapter creation
                if self.state.engine_mode != EngineModeState.TRAIN:
                    raise ConfigurationError(f"Cannot create a new adapter definition ('{effective_adapter_name}'). Engine is in {self.state.engine_mode.value} mode. Switch to TRAIN mode first.")
                if not adapter_config.adapter_type:
                    raise ConfigurationError("`adapter_type` must be provided in API call when creating a new adapter.")
                effective_adapter_type = adapter_config.adapter_type

                # --- Infer target modules if not provided  ---
                if not adapter_config.target_modules:
                    model_architecture = self.state.base_model.config.model_type.lower()
                    inferred_target_modules = []
                    if "phi3" in model_architecture or "phi-3" in model_architecture:
                        self.state.logger.info("Phi-3 model detected. Inferring LoRA target modules.")
                        inferred_target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
                    elif "qwen2" in model_architecture:
                        self.state.logger.info("Qwen2 model detected. Inferring LoRA target modules.")
                        inferred_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                    else: # Default to Llama-style
                        self.state.logger.info(f"Model architecture '{model_architecture}' not explicitly handled. Defaulting to Llama-style LoRA target modules.")
                        inferred_target_modules = ["q_proj", "v_proj"]

                    self.state.logger.info(f"Inferred target modules: {inferred_target_modules}")
                    adapter_config.target_modules = inferred_target_modules
                await asyncio.to_thread(os.makedirs, adapter_root_path, exist_ok=True)
                self.state.logger.info(f"Creating new adapter. Type from API: {effective_adapter_type.value}. Root dir: {str(adapter_root_path)}")

            # --- Perform validation checks BEFORE the main try/except block for model loading ---
            if not is_new_adapter_creation and metadata_from_disk:
                base_model_from_meta = metadata_from_disk.get("base_model_name_or_path") # From PEFT or custom
                if base_model_from_meta and self.state.base_model_name_or_path:
                    # Compare just the model names (basenames of the paths)
                    meta_model_name = os.path.basename(os.path.normpath(base_model_from_meta))
                    engine_model_name = os.path.basename(os.path.normpath(cast(str, self.state.base_model_name_or_path)))
                    if meta_model_name != engine_model_name: # type: ignore
                        raise ConfigurationError(
                            f"Adapter and engine base model mismatch. "
                            f"Adapter was trained on '{meta_model_name}' (from path '{base_model_from_meta}'), "
                            f"but engine is using '{engine_model_name}' (from path '{self.state.base_model_name_or_path}'). "
                            f"Context: {assumption_context or 'existing adapter load'}, path: {actual_checkpoint_path_to_load or adapter_root_path}."
                        )
                
                # Corrected quantization validation, respecting metadata.json as the source of truth.
                _validate_quantization_config(
                    self.state.logger,
                    metadata_from_disk,
                    self.state.base_model_quantization_cfg or {}
                )

            # Fail fast before blocking the cohort gate for obvious conflicts.
            if effective_adapter_name in await self.state.get_all_adapter_names_in_model():
                raise AdapterError(f"Adapter '{effective_adapter_name}' already exists in the model. Unload it first.")

            # --- Enter the cohort queue ONLY for the mutation window ---
            cohort_key = ("NF", (effective_adapter_name,".load.")) # type: ignore
            evt = await self.cohort_enter(cohort_key, request=CohortTaskType.LOAD_ADAPTER)
            entered_cohort = True
            if evt.is_set():
                raise AdapterError("Canceled adding adapter")

            # Re-check state after waiting in the queue to avoid stomping over a newly added adapter.
            if self.state.base_model is None:
                raise EngineError("Base model not initialized. Cannot add adapter.")
            
            target_model = self.state.peft_model
            if target_model is None:
                raise EngineError("PeftMixedModel not initialized. This is an internal error; the model should be created at engine startup.")

            if perform_reload_unload and adapter_to_reload:
                if adapter_to_reload in await self.state.get_all_adapter_names_in_model():
                    self.state.logger.info(f"Unloading adapter '{adapter_to_reload}' as part of reload.")
                    await asyncio.to_thread(target_model.delete_adapter, adapter_to_reload)
                    await self.state.remove_loaded_adapter_info(adapter_to_reload)
                    async with self.state._lock:
                        self.state._last_unloaded_adapter_name = adapter_to_reload
                        if adapter_to_reload in self.state._adapter_load_order:
                            self.state._adapter_load_order.remove(adapter_to_reload)
                    loaded_info_all = await self.state.get_loaded_adapters_info()
                else:
                    self.state.logger.warning(f"Adapter '{adapter_to_reload}' to be reloaded was not found in the model. Proceeding with load.")

            if effective_adapter_name in await self.state.get_all_adapter_names_in_model():
                raise AdapterError(f"Adapter '{effective_adapter_name}' already exists in the model. Unload it first.")

            try:
                if effective_adapter_type == AdapterType.LORA: # Only LORA is supported now
                    # All adapter management now goes through the persistent inference model. # noqa: E501
                    target_model = self.state.peft_model
                    if target_model is None:
                        raise EngineError("PeftMixedModel not initialized. This is an internal error; the model should be created at engine startup.")
                else:
                    raise ConfigurationError(f"Unsupported adapter type: {effective_adapter_type.value}")

                if target_model is None:
                    raise EngineError(f"Target model for adapter type {effective_adapter_type.value} could not be established.")

                original_training_state = target_model.training
                target_model.eval()

                if not is_new_adapter_creation and actual_checkpoint_path_to_load:
                    # Derive root path for state tracking: prefer the adapter_root_path; if the checkpoint
                    # path equals the root (flat layout), store the parent as the root when possible.
                    root_path_for_state = adapter_root_path
                    if actual_checkpoint_path_to_load == adapter_root_path and adapter_root_path.parent:
                        root_path_for_state = adapter_root_path.parent
                    # For foreign/flat layouts where the parent is a generic folder, keep the checkpoint path as root.
                    if source_is_foreign and actual_checkpoint_path_to_load and root_path_for_state == actual_checkpoint_path_to_load.parent:
                        root_path_for_state = actual_checkpoint_path_to_load

                    # Always (re)load from disk when a checkpoint path is given.
                    # If an adapter with this name already exists, replace it.
                    if effective_adapter_name in target_model.peft_config:
                        self.state.logger.info(
                            f"Replacing adapter '{effective_adapter_name}' from path: {actual_checkpoint_path_to_load}",
                        )
                        await asyncio.to_thread(target_model.delete_adapter, effective_adapter_name)
                    else:
                        self.state.logger.info(
                            f"Loading adapter '{effective_adapter_name}' from path: {actual_checkpoint_path_to_load}",
                        )

                    # TRAIN mode -> load as trainable; INFERENCE -> load frozen
                    #is_trainable_for_load = self.state.engine_mode == EngineModeState.TRAIN
                    low_mem_load = _has_meta_params(target_model)
                    if low_mem_load:
                        self.state.logger.warning(
                            "Adapter load: detected meta parameters in model; enabling low_cpu_mem_usage (assign=True)."
                        )
                    await asyncio.to_thread(
                        target_model.load_adapter,
                        str(actual_checkpoint_path_to_load),
                        effective_adapter_name,
                        is_trainable=False, #is_trainable_for_load, #TBD trainer should take care ot this
                        low_cpu_mem_usage=low_mem_load,
                    )

                    loaded_peft_config_on_model = target_model.peft_config[effective_adapter_name] # Should exist now
                    cfg_dict_for_store = loaded_peft_config_on_model.to_dict()
                    meta_for_store = _strip_config_from_meta(metadata_from_disk, cfg_dict_for_store)
                    meta_for_store = _enrich_meta_with_config(meta_for_store, cfg_dict_for_store)
                    self.state.logger.info(f"Engine.load_adapter: Storing info for EXISTING adapter: Name='{effective_adapter_name}', Root='{adapter_root_path}', Type='{effective_adapter_type.value}'")
                    await self.state.add_loaded_adapter_info(
                        adapter_name=effective_adapter_name,
                        adapter_root_path=str(root_path_for_state),
                        adapter_type=effective_adapter_type.value,
                        checkpoint_path=str(actual_checkpoint_path_to_load),
                        adapter_config_dump=cfg_dict_for_store,
                        base_model_quant=quant_display,
                        base_model_name=base_display,
                        alias=(actual_checkpoint_path_to_load or adapter_root_path).name,
                        metadata=meta_for_store or None,
                        is_foreign=source_is_foreign,
                    )

                    log_model_state(self.state.logger, self.state.peft_model, "after-load_adapter")
                    first_lora_status(self.state.logger, self.state.peft_model, "after-load")
                    _safetensors_lora_B_norms(self.state.logger, actual_checkpoint_path_to_load)                
                    adapter_health_AB(self.state.logger, self.state.peft_model, effective_adapter_name)

                elif is_new_adapter_creation: 
                    # Adding a new adapter definition to the persistent inference model.
                    # is_new_adapter_creation=True implies we are defining from parameters.
                    if target_model is None: 
                        raise EngineError("Target model (PeftMixedModel) not initialized for adding new adapter definition.")

                    peft_internal_config: Optional[LoraConfig] = None
                    if effective_adapter_type == AdapterType.LORA:
                        peft_internal_config = LoraConfig(
                            r=adapter_config.r,
                            lora_alpha=adapter_config.lora_alpha,
                            lora_dropout=adapter_config.lora_dropout,
                            target_modules=adapter_config.target_modules,
                            bias="none",
                            task_type=TaskType.CAUSAL_LM
                        )

                    # Ensure peft_internal_config is not None and is of the correct type (LoraConfig)
                    if not peft_internal_config:
                        raise ConfigurationError(f"Could not generate PEFT internal config for adapter '{effective_adapter_name}' of type '{effective_adapter_type.value}'.")
                    self.state.logger.info(f"Adding new empty adapter '{effective_adapter_name}' (Type: {effective_adapter_type.value}) from parameters to existing model.")
                    await asyncio.to_thread(target_model.add_adapter, effective_adapter_name, peft_internal_config)
                    
                    # For a new adapter, its config is already on the model from get_peft_model or add_adapter.
                    final_config_on_model = target_model.peft_config[effective_adapter_name]
                    if not final_config_on_model: # Should not happen if adapter was added correctly
                        raise AdapterError(f"Failed to retrieve PEFT config for newly added adapter '{effective_adapter_name}' from the model.")
                    self.state.logger.info(f"Engine.load_adapter: Storing info for NEW adapter: Name='{effective_adapter_name}', Root='{adapter_root_path}', Type='{effective_adapter_type.value}'")
                    cfg_dict_for_store = final_config_on_model.to_dict()
                    meta_for_store = _strip_config_from_meta(metadata_from_disk, cfg_dict_for_store)
                    meta_for_store = _enrich_meta_with_config(meta_for_store, cfg_dict_for_store)
                    await self.state.add_loaded_adapter_info(
                        adapter_name=effective_adapter_name,
                        adapter_root_path=str(adapter_root_path),
                        adapter_type=effective_adapter_type.value,
                        checkpoint_path=None, # No checkpoint path when just defining a new adapter
                        adapter_config_dump=cfg_dict_for_store,
                        base_model_quant=quant_display,
                        base_model_name=Path(self.state.base_model_name_or_path).name if self.state.base_model_name_or_path else None,
                        alias=adapter_root_path.name,
                        metadata=meta_for_store or None,
                        is_foreign=source_is_foreign,
                    )
                
                # No explicit recompile: rely on torch.compile guards to retrace if needed.
                #await _recompile_inference_model_if_needed(self.state, target_model, time.time())

                # Prepare detailed response
                # Ensure the adapter name used here is the one that was successfully processed and is on the model.
                # effective_adapter_name should be correct at this point.
                final_peft_config_on_model = target_model.peft_config.get(effective_adapter_name)
                
                peft_config_dict_for_response = None
                if final_peft_config_on_model:
                    peft_config_dict_for_response = final_peft_config_on_model.to_dict()
                    # Convert any sets to lists for JSON serialization
                    for key, value in peft_config_dict_for_response.items():
                        if isinstance(value, set):
                            peft_config_dict_for_response[key] = list(value)

                # self.state.logger.info(f"Engine.load_adapter: Name used for state.add_loaded_adapter_info was: '{effective_adapter_name}'") # Verbose
                # self.state.logger.info(f"Engine.load_adapter: Current state tracked adapters BEFORE returning: {await self.state.get_all_adapter_names_in_model()}") # Verbose
                # Build response aligned with list-all-adapters structure
                if is_new_adapter_creation:
                    return_entry = {
                        "index": 1,
                        "name": metadata_from_disk.get("adapter_name"),
                        "adapter_name": effective_adapter_name,
                        "alias": adapter_root_path.name,
                        "is_foreign": False,
                        "is_compatible": None,
                        "is_loaded": True,
                        "is_new": True,
                        "is_diff_checkpoint": None,
                        "is_diff_folder": None,
                        "base_model_name": Path(self.state.base_model_name_or_path).name if self.state.base_model_name_or_path else None,
                        "base_model_quant": _precision_suffix(),
                        "checkpoint_path": None,
                        "path": str(adapter_root_path),
                        "metadata": metadata_from_disk or None,
                        "adapter_type": effective_adapter_type.value,
                    }
                else:
                    quant_method = quant_method or _extract_quant_local(metadata_from_disk)
                    quant_display = quant_display or quant_display_from_meta(metadata_from_disk, quant_method)
                    is_compatible, _, base_display = _compatibility_flags(metadata_from_disk, quant_method)
                    cp_name = (actual_checkpoint_path_to_load.name if actual_checkpoint_path_to_load and actual_checkpoint_path_to_load != adapter_root_path else None)
                    cp_path_str = str(actual_checkpoint_path_to_load) if actual_checkpoint_path_to_load else None
                    return_entry = {
                        "index": 1,
                        "name": metadata_from_disk.get("adapter_name"),
                        "adapter_name": effective_adapter_name,
                        "alias": (actual_checkpoint_path_to_load or adapter_root_path).name,
                        "is_foreign": source_is_foreign,
                        "is_compatible": is_compatible,
                        "is_loaded": True,
                        "is_new": False,
                        "is_diff_checkpoint": False,
                        "is_diff_folder": False,
                        "base_model_name": base_display,
                        "base_model_quant": quant_display,
                        "checkpoint_path": cp_path_str,
                        "path": str(actual_checkpoint_path_to_load or adapter_root_path),
                        "metadata": metadata_from_disk or None,
                        "adapter_type": effective_adapter_type.value,
                    }

                # Final check to ensure the adapter is indeed in the state's tracking
                checked_adapters = await self.state.get_all_adapter_names_in_model()
                if effective_adapter_name not in checked_adapters:
                    message = f"Adapter '{effective_adapter_name}' processed but not found in state tracking after add_loaded_adapter_info. State: {checked_adapters}"
                    self.state.logger.error(message) # This indicates a deeper issue if it occurs.
                    raise AdapterError({message:message})

                return return_entry
            except Exception as e:
                # Use effective_adapter_name if available, otherwise fallback to original config name or "UNKNOWN"
                name_for_error = effective_adapter_name or adapter_config.adapter_name or "UNKNOWN"
                context_path = str(actual_checkpoint_path_to_load or adapter_root_path if 'adapter_root_path' in locals() else input_path)

                error_msg_str = str(e)
                if isinstance(e, RuntimeError) and "size mismatch" in error_msg_str:
                    # This is a very common error when the base model doesn't match the one the adapter was trained on.
                    # Provide a more user-friendly message.
                    base_model_from_meta = metadata_from_disk.get("base_model_name_or_path", "N/A")
                    engine_base_model = self.state.base_model_name_or_path  or "N/A"
                    
                    error_msg = (
                        f"Failed to add adapter '{name_for_error}' due to a model architecture mismatch. "
                        f"This usually means the adapter was trained on a different base model than the one currently loaded.\n"
                        f"  - Adapter's base model (from metadata): {base_model_from_meta}\n"
                        f"  - Engine's current base model: {engine_base_model}\n"
                        f"Please ensure the correct base model is loaded in the engine before adding this adapter. Original error: {error_msg_str}\n"
                        f"Context: {assumption_context or 'unspecified assumption'}, path considered: {context_path}"
                    )
                else:
                    error_msg = f"Failed to add adapter '{name_for_error}': {type(e).__name__} - {e}. Context: {assumption_context or 'unspecified assumption'}, path considered: {context_path}"
                self.state.logger.critical(f"!!! {error_msg}\n{traceback.format_exc()}")
                raise AdapterError(error_msg) from e
            finally:
                if target_model: # target_model might not be set if error occurs early
                    if original_training_state: target_model.train() # type: ignore
                    else: target_model.eval()
        finally:
            # --- Leave the cohort to allow other requests to proceed ---
            if entered_cohort:
                await self.cohort_leave(CohortTaskType.LOAD_ADAPTER)                

    async def set_active_adapter(self, adapter_names_to_set: Union[None, str, List[str]]) -> Dict[str, Any]:
        """
        Set the active adapter(s) by entering the cohort queue. This is a thread-safe operation
        that waits for its turn to mutate the model state, preventing conflicts with
        concurrent inference requests.
        """
        self.state.logger.info(f"--- Setting Active Adapter(s) to: {adapter_names_to_set} ---")
        if self.state.base_model is None:
            raise EngineError("Base model not initialized. Cannot set active adapter.")

        if not self.state.peft_model:
            raise EngineError("PeftMixedModel not initialized. This is an internal error; the model should be created at engine startup.")

        # Normalize the requested set of adapters
        target_set = self.get_normalized_adapter_set(
            [adapter_names_to_set] if isinstance(adapter_names_to_set, str) else adapter_names_to_set
        )

        # --- Validate that all requested adapters are loaded ---
        loaded_adapters = await self.state.get_loaded_adapters_info()
        for name in target_set:
            if name not in loaded_adapters:
                raise AdapterError(f"Adapter '{name}' not found. Available: {list(loaded_adapters.keys())}")

        # --- Enter the cohort queue to wait for a slot ---
        # This operation is treated as a "fast-path" cohort, similar to an inference request
        # that uses a specific set of adapters.
        cohort_key = ("F", target_set) # type: ignore
        evt = await self.cohort_enter(cohort_key, request=CohortTaskType.SET_ADAPTERS)
        if evt.is_set():
            raise AdapterError("Canceled set active adapters")

        try:
            # --- Perform the one-time switch now that we have the cohort lock ---
            # This logic is now centralized and shared with inference requests.
            async with self.state._cohort_lock: # type: ignore
                current_on_model = self.get_current_active_set_unlocked()
                if current_on_model != target_set:
                    self.state.logger.info(f"Switching model active adapters from {current_on_model} to {target_set}")
                    self.one_time_switch_to_set(target_set)
                else:
                    self.state.logger.info(f"Model active adapters already match target set {target_set}. No switch needed.")

                # --- Update the central state in MP13State ---
                # Determine the type if only one adapter is active.
                single_adapter_type = None
                if len(target_set) == 1:
                    adapter_name = target_set[0]
                    single_adapter_type = loaded_adapters.get(adapter_name, {}).get("type")

                # This is now an async call that updates the state under its own lock.
                await self.state.set_active_adapter_state(list(target_set), single_adapter_type)

        finally:
            # --- Leave the cohort to allow other requests to proceed ---
            await self.cohort_leave(CohortTaskType.SET_ADAPTERS)

        return {
            "active_adapter_names": await self.state.active_adapter_names(),
            "primary_active_adapter_name": await self.state.active_adapter_name(),
            "primary_active_adapter_type": await self.state.active_adapter_type()
        }
    
    def one_time_switch_to_set(self, target_set: Tuple[str, ...]):
        """
        Switch the model's active adapters to target_set under the mutation lock (no restore).
        Supports PeftModel (multi) and PeftMixedModel (multi if supported; falls back to first).
        """
        unwrapped_model = _unwrap_if_compiled(self.state.peft_model)
        lock = getattr(unwrapped_model, "_adapter_mutation_lock", None)
        if not lock:
            raise RuntimeError("Adapter mutation lock not found on model.")
        with lock:  # This is a synchronous method, so `with` is correct here.
            model = self.state.peft_model
            if not target_set:
                # Clear the model's notion of "active adapters" AND disable layers.
                # Use only public/forgiving calls and guard for PEFT version differences.
                # Most PEFT builds accept list[...] for multi and [] to clear
                model.set_adapter([])  # clear active set
                # Also clear in our engine/state bookkeeping
                self.state.active_adapters = ()
                # Fully disable adapter layers
                model.disable_adapter_layers()
                return
            model.set_adapter(list(target_set))
            self.state._model_active_set = target_set
            model.enable_adapter_layers()

    async def unload_adapter(self, adapter_name: str) -> Dict[str, Any]:
        """Remove an adapter from the engine, cancelling any pending requests that use it."""
        self.state.logger.info(f"--- Unloading Adapter: {adapter_name} ---")

        if not self.state.peft_model:
            raise EngineError("PeftMixedModel not initialized.")

        # Check for existence before performing any disruptive actions
        if adapter_name not in await self.state.get_all_adapter_names_in_model():
            self.state.logger.warning(f"Adapter '{adapter_name}' not found in model, unload is a no-op.")
            return {"status": "not_found", "adapter_name": adapter_name}

        # --- Unload Priority: Cancel pending and active requests ---
        # WARNING: This cancels active requests by signaling them, but does not wait for them to terminate.
        # This can lead to race conditions if the adapter is unloaded while still in active use.
        self.state.logger.info(f"Unload for '{adapter_name}' has priority. Signalling conflicting requests to cancel.")
        await self.cancel_request(cancel_for_adapter_name=adapter_name)
        
        # --- Enter cohort queue for exclusive mutation access ---
        cohort_key = ("NF", (adapter_name, ".unload."))
        evt = await self.cohort_enter(cohort_key, request=CohortTaskType.UNLOAD_ADAPTER)
        entered_cohort = True
        if evt.is_set():
            raise AdapterError("Canceled unloading adapter during cohort entry")
        
        try:
            # Re-check state after acquiring cohort
            if adapter_name not in await self.state.get_all_adapter_names_in_model():
                self.state.logger.warning(f"Adapter '{adapter_name}' was not found in the model after acquiring cohort; it may have been unloaded by another request.")
                return {"status": "not_found", "adapter_name": adapter_name}

            target_model = self.state.peft_model
            original_training_state = target_model.training
            target_model.eval()

            try:
                self.state.logger.info(f"Removing adapter '{adapter_name}' from the model.")
                
                async with self.state._lock:
                    if adapter_name in self.state._adapter_load_order:
                        self.state._adapter_load_order.remove(adapter_name)
                    self.state._last_unloaded_adapter_name = adapter_name

                await asyncio.to_thread(target_model.delete_adapter, adapter_name)
                await self.state.remove_loaded_adapter_info(adapter_name)

                log_model_state(self.state.logger, self.state.peft_model, "after-unload_adapter")

            # --- Cache Invalidation Logic on Unload ---
            #TBD
            #        async with self.state._lock:
            #            is_last_loaded = self.state._adapter_load_order and self.state._adapter_load_order[-1] == adapter_name
            #            if not is_last_loaded:
            #                self.state.logger.info(f"[cache-invalidate] Unloading adapter '{adapter_name}' which was not the last loaded. Invalidating adapter cache signatures.")
            #                invalidate_warmed_signatures_except_base()
            #
            #            if adapter_name in self.state._adapter_load_order:
            #                self.state._adapter_load_order.remove(adapter_name)
            #            self.state._last_unloaded_adapter_name = adapter_name

                # No explicit recompile: rely on torch.compile guards to retrace if needed.
                #await _recompile_inference_model_if_needed(self.state, model_to_modify, time.time()) 
                               
                return {"status": "unloaded", "adapter_name": adapter_name}
            
            except Exception as e:
                self.state.logger.error(f"Error during adapter removal: {e}")
                raise AdapterError(f"Failed to unload adapter '{adapter_name}': {e}") from e
            finally:
                if original_training_state: target_model.train() # type: ignore
                else: target_model.eval()
        finally:
            # --- Leave cohort ---
            if entered_cohort:
                await self.cohort_leave(CohortTaskType.UNLOAD_ADAPTER)
