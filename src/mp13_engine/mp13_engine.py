# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""MP13 Engine - Unified training and inference server."""

import logging, types
import importlib.util
# Create a global logger for the engine module
logger = logging.getLogger(__name__)

import asyncio, contextvars, functools, threading

# --- Context-aware Logging Support ---
instance_id_var = contextvars.ContextVar('instance_id', default='system')
_LOGGER_RECONFIGURED = False
_logger_lock = threading.Lock()

class InstanceIdFilter(logging.Filter):
    """A logging filter that injects the instance_id from a context variable."""
    def filter(self, record):
        record.instance_id = instance_id_var.get()
        return True

def set_log_context(func):
    """A decorator to set the instance_id in the logging context for the duration of an async method."""
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Ensure self is an MP13Engine instance with an instance_id
        if hasattr(self, 'instance_id'):
            token = instance_id_var.set(self.instance_id)
            try:
                return await func(self, *args, **kwargs)
            finally:
                instance_id_var.reset(token)
        else:
            return await func(self, *args, **kwargs)
    return wrapper

class EngineLogContextMeta(type):
    """Metaclass to automatically apply the set_log_context decorator to public async methods."""
    def __new__(cls, name, bases, dct):
        for attr_name, attr_value in dct.items():
            if not attr_name.startswith('_') and asyncio.iscoroutinefunction(attr_value):
                dct[attr_name] = set_log_context(attr_value)
        return super().__new__(cls, name, bases, dct)
# --- End Context-aware Logging Support ---

from dataclasses import asdict
import copy, concurrent.futures
import threading, time
import asyncio
import time
import traceback, gc
from typing import Optional, Dict, Any, List, Union, Tuple, cast, Callable, AsyncIterator
from pathlib import Path

import torch

from transformers import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
# Optional (newer transformers) autos/classes - imported lazily when needed.
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig, AwqConfig, HqqConfig, EetqConfig

# Local imports
from .mp13_patches import apply_engine_init_patches, apply_infer_patches

from .mp13_config import ( 
    APIStatus, TrainingConfig, InferenceRequest, InferenceResponse,
    EngineMode, AdapterConfig, GlobalEngineConfig,ColumnsConfig
)

from . import mp13_state
from .mp13_state import (
    MP13State, TrainingStatus, InferenceStatus, ServerStatus, EngineModeState,
    ConfigurationError, DatasetError, TrainingError, EngineError,
    EngineInitializationError, AdapterError, InferenceRequestError, BusyError, ModeMismatchError,
    RequestResource, AggregateInferenceMetrics
)
from .mp13_tools_parser import guess_profile_from_template, profile_for
from .mp13_utils import (
    choose_special_tokens_no_add, format_prompt_messages,
    get_best_device_map, first_module_device, needs_single_gpu,
    inspect_device_layout
)

from .mp13_adapter import AdaptersControl, CohortTaskType, quant_display_from_meta
from .mp13_train import (
    start_training_logic,
    stop_training_logic,
    execute_training_run_logic
)
from .mp13_infer import run_inference_logic, format_inference_prompt_logic, count_tokens_logic
# Import from the new location
from .mp13_cache import initialize_cache_session_config, reset_compile_warm_tracker, reset_static_cache


# HQQ Imports
from hqq.core.quantize import HQQBackend, HQQLinear
from hqq.utils.patching import prepare_for_inference as hqq_prepare_for_inference

def _safe_log_text(value: Any) -> str:
    """Return an ASCII-only string for logging to avoid Windows console encoding errors."""
    try:
        text = value if isinstance(value, str) else repr(value)
    except Exception:
        text = "<unprintable>"

    try:
        return text.encode("ascii", "backslashreplace").decode("ascii")
    except Exception:
        return "<unprintable>"

def _create_quantization_config(
    config: GlobalEngineConfig,
    is_bf16_supported: bool
) -> Tuple[Optional[Union[BitsAndBytesConfig, AwqConfig, HqqConfig, EetqConfig]], str, Optional[str]]:
    """Creates the quantization config object based on the global engine config."""
    quantization_config_obj: Optional[Union[BitsAndBytesConfig, AwqConfig, HqqConfig, EetqConfig]] = None
    effective_quantization_method = "none"
    effective_quant_precision: Optional[str] = None
    cuda_is_available = torch.cuda.is_available()

    if config.quantize_bits == "4":
        logging.info(f"Using 4-bit BitsAndBytes quantization. Type: {config.bnb_4bit_quant_type}, Compute Dtype: {config.bnb_4bit_compute_dtype}")
        bnb_compute_torch_dtype = getattr(torch, config.bnb_4bit_compute_dtype, torch.bfloat16)
        if config.bnb_4bit_compute_dtype == "bfloat16" and not (cuda_is_available and is_bf16_supported):
            logging.warning("bnb_4bit_compute_dtype 'bfloat16' requested but not supported. Falling back to float32 for compute.")
            bnb_compute_torch_dtype = torch.float32
        quantization_config_obj = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_compute_torch_dtype,
            bnb_4bit_use_double_quant=True
        )
        effective_quantization_method = "bnb"
        effective_quant_precision = "I4"
    elif config.quantize_bits == "8":
        logging.info("Using 8-bit BitsAndBytes quantization.")
        quantization_config_obj = BitsAndBytesConfig(load_in_8bit=True)
        effective_quantization_method = "bnb"
        effective_quant_precision = "I8"
    elif config.quantize_bits == "awq":
        logging.info(f"Using AWQ quantization. Bits: {config.awq_bits}, Group Size: {config.awq_group_size}, Zero Point: {config.awq_zero_point}")
        quantization_config_obj = AwqConfig(
            bits=config.awq_bits,
            group_size=config.awq_group_size,
            zero_point=config.awq_zero_point
        )
        effective_quantization_method = "awq"
        effective_quant_precision = f"I{config.awq_bits}"
    elif config.quantize_bits == "hqq":
        logging.info(f"Using HQQ quantization. Bits: {config.hqq_bits}, Group Size: {config.hqq_group_size}, Axis: {config.hqq_axis}")
        quantization_config_obj = HqqConfig(
            nbits=config.hqq_bits,
            group_size=config.hqq_group_size,
            axis=config.hqq_axis,
            compute_dtype="bfloat16"
        )
        effective_quantization_method = "hqq"
        effective_quant_precision = f"I{config.hqq_bits}"
    elif config.quantize_bits == "eetq":
        logging.info("Using EETQ quantization.")
        quantization_config_obj = EetqConfig()
        effective_quantization_method = "eetq"
        effective_quant_precision = "I8"
    
    return quantization_config_obj, effective_quantization_method, effective_quant_precision

def _set_hqq_backend_for_mode(state: "MP13State", model: Any, mode: EngineMode):
    """Sets the HQQ backend based on the desired engine mode."""
    if not state.global_config or state.global_config.get("quantize_bits") != "hqq":
        return

    target_backend = "pytorch" if mode == EngineMode.TRAIN else "default"
    
    # This helper needs to be defined here or imported if it's to be used.
    def hqq_backend_name(m) -> str:
        for mod in m.modules():
            if isinstance(mod, HQQLinear): return str(mod.backend)
        return "<no HQQ layers>"

    current_backend = hqq_backend_name(model)
    
    if current_backend != target_backend:
        state.logger.info(f"Switching HQQ backend from '{current_backend}' to '{target_backend}' for {mode.value} mode.")
        hqq_prepare_for_inference(model, backend=target_backend, verbose=False)
        # Re-materialize meta tensors after backend switch
        for mod in model.modules():
            if not isinstance(mod, HQQLinear) or not hasattr(mod, "meta") or not isinstance(mod.meta, dict):
                continue

            weight_device: Optional[torch.device] = None
            weight_tensor = getattr(mod, "W_q", None)
            if isinstance(weight_tensor, torch.Tensor):
                weight_device = weight_tensor.device
            elif hasattr(mod, "linear_layer"):
                orig_weight = getattr(getattr(mod, "linear_layer"), "weight", None)
                if isinstance(orig_weight, torch.Tensor):
                    weight_device = orig_weight.device

            if weight_device is None:
                continue

            for k, v in list(mod.meta.items()):
                if isinstance(v, torch.Tensor) and v.device != weight_device:
                    mod.meta[k] = v.to(weight_device)
        torch.cuda.synchronize()
        state._hqq_backend = target_backend

# --- Global Frozen Config for Multi-Engine Consistency ---
_FROZEN_ENGINE_CONFIG: Optional[GlobalEngineConfig] = None
_frozen_config_lock = threading.Lock()
# --- End Global Frozen Config ---

class MP13Engine(metaclass=EngineLogContextMeta):
    """
    Unified engine for training and inference.
    Manages a base AutoModelForCausalLM and MixedPeftModel for adapters.
    Operates in either TRAIN or INFERENCE mode.
    """

    def __init__(self, instance_id: str = "default"):
        self.instance_id = instance_id
        self.state = MP13State(logger=logger, instance_id = instance_id)
        self.adapters_control = AdaptersControl(self.state)

        logger.debug(f"[MP13Engine] Initialized new instance: {instance_id}")


    def _check_engine_readiness(self):
        """Raises an error if the engine is not initialized or is shutting down."""
        # Check for initialization
        if self.state.base_model is None:
            raise EngineError("Engine is not initialized. Please call initialize_global_resources first.")
        # Check for shutdown status
        if self.state.server_status == ServerStatus.SHUTTING_DOWN:
            raise EngineError("Engine is shutting down. No new operations are allowed.")

    async def _get_effective_global_config_dict(self) -> Dict[str, Any]:
        """Returns a clean dictionary of the effective global config, omitting None values."""
        # The state's global_config is now the single source of truth for all effective config.
        return self.state.global_config or {}

    async def _check_and_recover_state(self):
        """
        Checks for and recovers from known, non-fatal error states before starting a new operation.
        This makes the engine more self-healing by automatically resetting components from
        recoverable errors (e.g., a cancelled request) upon the next API call.
        """

        #TBD
        return
    
        is_recoverable = True
        if is_recoverable:
            logger.info(f"[Engine Recovery] Acknowledged recoverable inference error: '{inference_error_msg}'. Resetting inference component to READY.")
            await self.state.set_inference_status(InferenceStatus.READY, "Resetting from acknowledged recoverable error.")

    async def initialize_global_resources(
        self,
        config: GlobalEngineConfig
    ):
        # --- Initialization Report Tracking ---
        init_report: Dict[str, Any] = {
            "warnings": [],
            "errors": [],
            "applied_patches": []
        }
        
        # --- Multi-Engine Configuration Consistency Check ---
        global _FROZEN_ENGINE_CONFIG
        with _frozen_config_lock:
            if _FROZEN_ENGINE_CONFIG is not None:
                logger.warning("An engine has already been initialized. Enforcing configuration consistency.")
                frozen_config = _FROZEN_ENGINE_CONFIG

                # Define keys that must be consistent across all engines
                # These relate to global state (patches, environment) set by the first engine
                keys_to_check = [
                    'attn_implementation', 'base_model_torch_dtype', 'use_torch_compile', 'quantize_bits',
                    'bnb_4bit_quant_type', 'bnb_4bit_compute_dtype',
                    'awq_bits', 'awq_group_size', 'awq_zero_point',
                    'hqq_bits', 'hqq_group_size', 'hqq_axis'
                ]

                for key in keys_to_check:
                    frozen_value = getattr(frozen_config, key, None)
                    current_value = getattr(config, key, None)
                    if frozen_value != current_value:
                        warning_msg = (
                            f"Configuration mismatch for '{key}'. The first engine set this to '{frozen_value}', "
                            f"but this engine requested '{current_value}'. Overriding to '{frozen_value}' for consistency."
                        )
                        logger.warning(warning_msg)
                        init_report["warnings"].append(warning_msg)
                        setattr(config, key, frozen_value)
        # --- End Consistency Check ---
        
        # --- One-time Logger Reconfiguration ---
        global _LOGGER_RECONFIGURED
        if config.log_with_instance_id and not _LOGGER_RECONFIGURED:
            with _logger_lock:
                if not _LOGGER_RECONFIGURED:
                    logger.info("Reconfiguring logger to include instance_id.")
                    log_format = f'%(asctime)s [%(levelname)s] [%(instance_id)-{config.log_instance_id_width}s] %(message)s'
                    formatter = logging.Formatter(log_format)
                    
                    # Configure the specific logger, not the root
                    engine_logger = logging.getLogger(__name__)
                    engine_logger.addFilter(InstanceIdFilter())
                    
                    # Find the handler to update, assuming a basic config or a StreamHandler
                    # This is complex; for this app, we assume it's okay to modify handlers on the specific logger
                    if not engine_logger.handlers:
                        # If no handlers, it propagates to root. Add a handler to prevent this and control format.
                        handler = logging.StreamHandler()
                        handler.setFormatter(formatter)
                        engine_logger.addHandler(handler)
                        engine_logger.propagate = False
                    else:
                        for handler in engine_logger.handlers:
                            handler.setFormatter(formatter)
                    
                    _LOGGER_RECONFIGURED = True
                    logger.info("Logger reconfiguration complete.")
        # --- End Logger Reconfiguration ---

        # Unpack config for local use
        base_model_name_or_path = config.base_model_name_or_path
        device_map = config.device_map
        trust_remote_code = config.trust_remote_code
        base_model_torch_dtype = config.base_model_torch_dtype
        quantize_bits = config.quantize_bits
        bnb_4bit_quant_type = config.bnb_4bit_quant_type
        bnb_4bit_compute_dtype = config.bnb_4bit_compute_dtype
        awq_bits = config.awq_bits
        awq_group_size = config.awq_group_size
        awq_zero_point = config.awq_zero_point
        hqq_bits = config.hqq_bits
        hqq_group_size = config.hqq_group_size
        hqq_axis = config.hqq_axis
        initial_engine_mode = config.initial_engine_mode
        concurrent_generate = config.concurrent_generate
        default_context_size = config.default_context_size
        default_max_new_tokens = config.default_max_new_tokens
        attn_implementation = config.attn_implementation
        static_kv_cache = config.static_kv_cache
        use_cache = config.use_cache
        use_torch_compile = config.use_torch_compile
        no_tools_parse = config.no_tools_parse
        disable_custom_pad_ids = config.disable_custom_pad_ids

        # New optional configs
        instance_id_override = config.instance_id
        parser_profile_key_override = config.tools_parser_profile_key
        custom_chat_template = config.custom_chat_template

        if self.state.base_model is not None:
            msg = f"Global resources (base model {base_model_name_or_path}) already initialized. Skipping."
            logger.info(msg)
            return {"message": msg}

        applied_patches = apply_engine_init_patches(logger)
        init_report["applied_patches"].extend(applied_patches)
        apply_infer_patches(logger) 

        logger.info(f"--- Initializing Global Engine Resources ({self.instance_id}) ---")
        run_start_time_wall = time.time()
        # print(f"[MP13Engine:{self.instance_id}] initialize_global_resources: Before self.state.set_server_status(INITIALIZING)") # Verbose
        await self.state.set_server_status(ServerStatus.INITIALIZING, "Starting global resource initialization.")
        # print(f"[MP13Engine:{self.instance_id}] initialize_global_resources: After self.state.set_server_status(INITIALIZING)") # Verbose
        #logger.debug(f"MP13State server status set to INITIALIZING.")

        # --- Handle instance_id override ---
        if instance_id_override:
            logger.info(f"Overriding engine instance ID from '{self.instance_id}' to '{instance_id_override}'.")
            self.instance_id = instance_id_override
            self.state.instance_id = instance_id_override
 
        # Define helper for CUDA calls
        async def _is_cuda_available_async():
            # logger.debug("Checking torch.cuda.is_available() via asyncio.to_thread...") # Verbose
            available = await asyncio.to_thread(torch.cuda.is_available)
            # logger.debug(f"torch.cuda.is_available() returned: {available}") # Verbose
            return available

        async def _cuda_device_count_async():
            # logger.debug("Checking torch.cuda.device_count() via asyncio.to_thread...") # Verbose
            count = await asyncio.to_thread(torch.cuda.device_count)
            # logger.debug(f"torch.cuda.device_count() returned: {count}") # Verbose
            return count

        async def _is_bf16_supported_async():
            # logger.debug("Checking torch.cuda.is_bf16_supported() via asyncio.to_thread...") # Verbose
            supported = await asyncio.to_thread(torch.cuda.is_bf16_supported)
            # logger.debug(f"torch.cuda.is_bf16_supported() returned: {supported}") # Verbose
            return supported

        try:
            cuda_is_available = await _is_cuda_available_async()
            
            if cuda_is_available:
                import gc
                device_count = await _cuda_device_count_async()
                logger.info(f"CUDA available. Devices: {device_count}. Clearing cache and collecting garbage.")
                await asyncio.to_thread(torch.cuda.empty_cache)
                await asyncio.to_thread(gc.collect)
                await asyncio.to_thread(torch.cuda.ipc_collect)

            # First, load just the configuration to inspect it for feature support
            logger.info(f"Loading model config for '{base_model_name_or_path}' to check for feature support...")
            model_config = await asyncio.to_thread(
                lambda: AutoConfig.from_pretrained(
                    base_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                )
            )
            model_type_from_config = getattr(model_config, "model_type", "").lower()
            logger.info(f"Detected model type from config: '{model_type_from_config}'")

            # Mistral3/Ministral3 models are exposed as *conditional generation* in newer Transformers,
            # not as AutoModelForCausalLM. We special-case routing for text-only use here.
            is_mistral3_family = model_type_from_config in ("mistral3", "ministral3")

            # --- Attention Implementation Override ---
            # For certain models, 'auto' might default to a suboptimal 'eager' implementation.
            # This logic attempts to force a better implementation if available.
            models_to_override_attn = ['phi', 'phi3', 'phi3small', 'qwen2']
            if model_type_from_config in models_to_override_attn and attn_implementation and attn_implementation.lower() == "auto":
                is_flash_attn_available = False
                try:
                    if importlib.util.find_spec("flash_attn"):
                        is_flash_attn_available = True
                except Exception:
                    pass  # Ignore any import-related errors

                if is_flash_attn_available:
                    warn_msg = f"Model type '{model_type_from_config}' detected with attn_implementation='auto'. Forcing to 'flash_attention_2' for optimal performance." # type: ignore
                    logger.warning(warn_msg)
                    init_report["warnings"].append(warn_msg)
                    attn_implementation = "flash_attention_2"
                else:
                    # Check if the model supports SDPA before falling back to it.
                    # Some models (like Phi-3) do not support SDPA and would fail.
                    supports_sdpa = getattr(model_config, "_supports_sdpa", True) # Default to True if attr not present
                    if supports_sdpa:
                        warn_msg = f"Model type '{model_type_from_config}' with attn_implementation='auto', but flash-attn is not installed. Using 'sdpa' as a fallback. For best performance, consider `pip install flash-attn --no-build-isolation`." # type: ignore
                        logger.warning(warn_msg)
                        init_report["warnings"].append(warn_msg)
                        attn_implementation = "sdpa"
                    else:
                        warn_msg = f"Model type '{model_type_from_config}' with attn_implementation='auto', but flash-attn is not installed and the model does not support SDPA. The 'eager' implementation will be used, which may be slow." # type: ignore
                        logger.warning(warn_msg)
                        init_report["warnings"].append(warn_msg)
                        # Do not change attn_implementation, let 'auto' resolve to 'eager'.
            # --- End of Attention Override ---

            actual_torch_dtype: Optional[torch.dtype] = None
            # logger.debug(f"Determining actual_torch_dtype. Requested: {base_model_torch_dtype}") # Verbose
            if base_model_torch_dtype == "bfloat16":
                if cuda_is_available:
                    bf16_supported = await _is_bf16_supported_async()
                    if bf16_supported:
                        actual_torch_dtype = torch.bfloat16
                    else:
                        warn_msg = "bfloat16 requested but not supported by CUDA, falling back to float16." # type: ignore
                        logger.warning(warn_msg)
                        init_report["warnings"].append(warn_msg)
                else:
                    warn_msg = "bfloat16 requested but CUDA not available, falling back to float32." # type: ignore
                    logger.warning(warn_msg)
                    init_report["warnings"].append(warn_msg)
            elif base_model_torch_dtype == "float16":
                if cuda_is_available:
                    actual_torch_dtype = torch.float16
                else:
                    logger.warning("float16 requested but CUDA not available, falling back to float32.")
            elif base_model_torch_dtype == "float32":
                actual_torch_dtype = torch.float32
            
            if actual_torch_dtype is None: # Auto or fallback
                # logger.debug("actual_torch_dtype is None, attempting auto-detection.") # Verbose
                if cuda_is_available:
                    bf16_supported_auto = await _is_bf16_supported_async()
                    if bf16_supported_auto: actual_torch_dtype = torch.bfloat16
                    else: actual_torch_dtype = torch.float16 # Fallback to float16 if CUDA available but not bf16
                else: actual_torch_dtype = torch.float32
            if quantize_bits == "eetq" and actual_torch_dtype == torch.bfloat16:
                warn_msg = (
                    "EETQ does not support bfloat16. Overriding torch_dtype to float16 for model loading."
                )
                logger.warning(warn_msg)
                init_report["warnings"].append(warn_msg)
                actual_torch_dtype = torch.float16

            logger.info(f"Determined torch_dtype for base model loading: {actual_torch_dtype} (requested: {base_model_torch_dtype})")

            # --- CPU-specific adjustments (moved here to override auto-detection) ---
            # This block handles both cases: CUDA not available, or user explicitly requests CPU.
            if not cuda_is_available or device_map == "cpu":
                if not cuda_is_available:
                    logger.warning("CUDA not available. Forcing device_map='cpu' and CPU-compatible settings.")
                    device_map = "cpu" # Ensure it's set if auto-detected
                else: # User explicitly requested CPU
                    logger.info("User requested device_map='cpu'. Applying CPU-compatible settings.")

                if actual_torch_dtype != torch.float32:
                    warn_msg = f"CPU mode active. Overriding torch_dtype from '{actual_torch_dtype}' to 'torch.float32'."
                    logger.warning(warn_msg)
                    init_report["warnings"].append(warn_msg)
                    actual_torch_dtype = torch.float32
                if quantize_bits != "none":
                    warn_msg = "CPU mode active. Disabling quantization."
                    logger.warning(warn_msg)
                    init_report["warnings"].append(warn_msg)
                    quantize_bits = "none"

            model_load_kwargs: Dict[str, Any] = {
                "torch_dtype": actual_torch_dtype,
                "device_map": device_map,
                "trust_remote_code": trust_remote_code,
                "low_cpu_mem_usage": True,
                # NOTE: some model classes (e.g. Mistral3ForConditionalGeneration in some builds)
                # do NOT accept `use_cache` as an __init__ kwarg. We set it on config/model after load.
            }
            # Keep a separate flag for later
            requested_use_cache = use_cache

            # Phi-3 model tweak to prevent model sharding with flash_attention_2
            effective_attn_impl_for_check = attn_implementation or "auto"
            if model_load_kwargs.get("device_map") == "auto" and await asyncio.to_thread(needs_single_gpu, base_model_name_or_path, effective_attn_impl_for_check): # type: ignore
                warn_msg = "flash_attention_2 detected on a model that can’t safely shard (e.g., Phi-3 / no SDPA). Forcing single-GPU to avoid blocksparse device mismatches."
                logger.warning(warn_msg)
                init_report["warnings"].append(warn_msg)
                single_gpu_device_map = await asyncio.to_thread(get_best_device_map)
                logger.info(f"Enforcing single GPU device map: {single_gpu_device_map}") # This is info, not a warning
                model_load_kwargs["device_map"] = single_gpu_device_map

            # --- Create Quantization Config using Helper ---
            bf16_supported_for_quant = await _is_bf16_supported_async() if cuda_is_available else False
            quantization_config_obj, effective_quantization_method, effective_quant_precision = _create_quantization_config(
                config, bf16_supported_for_quant
            )
            if quantization_config_obj:
                model_load_kwargs["quantization_config"] = quantization_config_obj

            if attn_implementation and attn_implementation.lower() != "auto":
                model_load_kwargs["attn_implementation"] = attn_implementation

            logger.info(f"Device Map: {device_map}, dtype: {actual_torch_dtype}, attn: {model_load_kwargs.get('attn_implementation', 'auto')}, use_cache: {requested_use_cache}")

            #logger.debug(f"Loading tokenizer for {base_model_name_or_path}...")
            tokenizer_kwargs = {
                "trust_remote_code": trust_remote_code,
                "fix_mistral_regex": True,
            }

            try:
                tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained, base_model_name_or_path, **tokenizer_kwargs
                )
            except TypeError as exc:
                # Some tokenizers pass fix_mistral_regex internally; avoid double-kwarg.
                if "fix_mistral_regex" in str(exc):
                    tokenizer_kwargs.pop("fix_mistral_regex", None)
                    tokenizer = await asyncio.to_thread(
                        AutoTokenizer.from_pretrained, base_model_name_or_path, **tokenizer_kwargs
                    )
                else:
                    raise
            #logger.debug(f"Tokenizer loading via asyncio.to_thread completed.")
            if tokenizer is None:
                raise EngineInitializationError("AutoTokenizer.from_pretrained returned None.")
            tokenizer.padding_side = "left"
            logger.debug("Tokenizer padding side set to 'left'.")
            if custom_chat_template:
                tokenizer.chat_template = custom_chat_template
                logger.info("Applied custom chat template override.")
                if not hasattr(tokenizer, "apply_chat_template"):
                    logger.warning("Custom chat template provided, but tokenizer does not implement apply_chat_template; override will be ignored.")

            # --- NEW: Initialize combined resource pool ---
            self.state._resource_pool.clear()
            self.state._resource_in_use.clear()
            self.state._resource_semaphore = None
            logger.info(f"Initializing resource pool for max concurrency of {concurrent_generate}.")
            self.state._resource_semaphore = asyncio.Semaphore(concurrent_generate)

            # Could extend Qwen context length from 32K as per their claim?
            #model_load_kwargs["max_position_embeddings"]= 131072  # Set desired to 128K 
            #model_load_kwargs["rope_scaling"] ={
            #    "type": "yarn",
            #    "factor": 4.0,
            #    "original_max_position_embeddings": 32768, # Set to original trained size per Qwen
            #}

            logger.info(f"Loading base model: {base_model_name_or_path}")
            if is_mistral3_family:
                # Prefer the explicit class when available (Transformers main / newer builds).
                # Fall back to a Seq2Seq auto if present. Otherwise, raise a clear error.
                def _load_mistral3_model():
                    # Clone kwargs so we can safely mutate for this special-case model.
                    _kwargs = dict(model_load_kwargs)
                    _kwargs.pop("use_cache", None)

                    # Ministral-3-3B-Instruct-2512 ships FP8 weights; Transformers expects fine-grained FP8 integration.
                    # If triton isn't available, we can't load/dequantize these weights in this codepath.
                    triton_available = importlib.util.find_spec("triton") is not None
                    if not triton_available:
                        raise EngineInitializationError(
                            "This checkpoint uses FP8 weights (fine-grained FP8). Loading it via Transformers requires "
                            "`triton` (or a compatible triton-windows build) so Transformers can dequantize to BF16/FP16. "
                            "Alternative: use a BF16 variant of Ministral 3 (Base/Reasoning) instead of the FP8 Instruct checkpoint."
                        )

                    try:
                        from transformers import Mistral3ForConditionalGeneration  # type: ignore
                        from transformers import FineGrainedFP8Config  # type: ignore

                        # Dequantize FP8 weights to BF16 on load (per model card guidance)
                        _kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)

                        return Mistral3ForConditionalGeneration.from_pretrained(
                            base_model_name_or_path,
                            config=model_config,
                            **_kwargs
                        )
                    except Exception:
                        # Fallback: try the auto for conditional/seq2seq generation if the explicit
                        # class isn't importable in this installed transformers build.
                        try:
                            from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM  # type: ignore
                            return AutoModelForSeq2SeqLM.from_pretrained(
                                base_model_name_or_path, **model_load_kwargs
                            )
                        except Exception as e2:
                            raise EngineInitializationError(
                                "This model has model_type='mistral3' and is not compatible with "
                                "AutoModelForCausalLM in this Transformers build. "
                                "Install a newer Transformers (e.g. pinned git rev) that provides "
                                "Mistral3ForConditionalGeneration / AutoModelForSeq2SeqLM."
                            ) from e2

                actual_base_model = await asyncio.to_thread(_load_mistral3_model)
            else:
                actual_base_model = await asyncio.to_thread(
                    lambda: AutoModelForCausalLM.from_pretrained(
                        base_model_name_or_path,
                        config=model_config,
                        **model_load_kwargs,
                    )
                )
            #logger.debug(f"DEBUG: type(actual_base_model)={type(actual_base_model)}, repr={repr(actual_base_model)}")
            if actual_base_model is None:
                raise EngineInitializationError("AutoModelForCausalLM.from_pretrained returned None.")

            # Apply `use_cache` after load (works for both causal + conditional-generation models)
            try:
                if hasattr(actual_base_model, "config") and actual_base_model.config is not None:
                    actual_base_model.config.use_cache = bool(requested_use_cache)
            except Exception:
                pass
 
            # --- Create a static deepcopy of the tokenizer for cache warming ---
            logger.debug("Creating static deepcopy of tokenizer for cache warming.")
            tokenizer_for_warmup = await asyncio.to_thread(copy.deepcopy, tokenizer)

            # --- Initialize generation executor based on concurrency setting ---
            # This executor is used for  inference generation 
            # The number of workers for inference is controlled by `concurrent_generate`.
            # The background cache  warm up task will run on its own executor.
            if concurrent_generate > 1:
                max_workers = concurrent_generate
                logger.info(f"Concurrent generation enabled. Initializing ThreadPoolExecutor with max_workers={max_workers}.")
            else:
                max_workers = 1
                logger.info("Concurrent generation disabled (max_workers=1).")
            self.state._gen_exec = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mp13-infer")

            # --- Create a persistent PeftMixedModel wrapper for inference ---
            if use_torch_compile:
                logger.info("Creating, compiling, and warming up persistent PeftMixedModel wrapper...")
            else:
                logger.info("Creating persistent PeftMixedModel wrapper (compilation disabled)...")

            loop = asyncio.get_running_loop()
            self.state.loop = loop # Pass loop to state for thread-safe callbacks

            def _create_compile_warmup_and_finalize_model():
                # This function runs in the executor thread and performs the entire setup sequence.
                # 1. Create model with a temporary 'bootstrap' adapter
                # Infer target modules for the bootstrap LoRA config
                model_architecture = actual_base_model.config.model_type.lower()
                bootstrap_target_modules: List[str] = []
                if "phi3" in model_architecture or "phi-3" in model_architecture:
                    bootstrap_target_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
                elif "qwen2" in model_architecture:
                    bootstrap_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                else: # Default to Llama-style
                    bootstrap_target_modules = ["q_proj", "v_proj"]
                logger.debug(f"Using bootstrap target modules for PeftMixedModel: {bootstrap_target_modules}")
                # For mistral3-family models we load a ConditionalGeneration/Seq2Seq class, so use SEQ_2_SEQ_LM.
                # For the classic causal LMs keep CAUSAL_LM.
                bootstrap_task_type = TaskType.SEQ_2_SEQ_LM if is_mistral3_family else TaskType.CAUSAL_LM
                bootstrap_lora_config = LoraConfig(
                    r=1, lora_alpha=1, lora_dropout=0.0, bias="none",
                    target_modules=bootstrap_target_modules, task_type=bootstrap_task_type
                )

                peft_model = get_peft_model(actual_base_model, bootstrap_lora_config, adapter_name="bootstrap", mixed=True)
                # --- Direct Lock Injection ---
                # Instead of relying on a global __init__ patch, inject the lock directly after creation.
                # This is more robust against import order issues.
                if not hasattr(peft_model, "_adapter_mutation_lock"):
                    peft_model._adapter_mutation_lock = threading.RLock()
                    logger.debug("Directly injected _adapter_mutation_lock into PeftMixedModel instance.")
                # --- End Direct Lock Injection ---

                peft_model.eval() # Set to eval mode right after creation
                logger.info("PeftMixedModel wrapper created with bootstrap adapter.")

                # 2. Compile the model (if enabled)
                compiled_model = peft_model
                if use_torch_compile:
                    import torch as _torch
                    from torch._inductor import config as ic
                    ic.triton.cudagraphs = True
                    ic.triton.cudagraph_trees = True
                    ic.triton.cudagraph_skip_dynamic_graphs = True
                    #ic.triton.cudagraph_dynamic_shape_warn_limit = None

                    logger.info("Applying torch.compile() to persistent PeftMixedModel for inference...")
                    try:
                        compiled_model = _torch.compile(peft_model, mode="reduce-overhead")
                        logger.info("torch.compile() applied successfully to PeftMixedModel.")

                        # run once per process (or per device) before the very first compiled call
                        # This is to avoid backgound cache warn up crashes doeu to  Cuda graph captures conflict
                        def _prime_blas(device):
                            _torch.cuda.set_device(device)
                            a = _torch.empty((32, 32), device=device, dtype=_torch.bfloat16)
                            b = _torch.empty((32, 32), device=device, dtype=_torch.bfloat16)
                            # cublasLt path is captureable; this first matmul initializes handle/workspace
                            _ = a @ b

                        # 1) make sure this thread is on the model’s device
                        dev = next(compiled_model.parameters()).device
                        if dev.type == "cuda":
                            _torch.cuda.set_device(dev)
                            _prime_blas(dev)

                    except Exception as e:
                        logger.warning(f"torch.compile() on PeftMixedModel failed: {e}. Continuing without compilation.", exc_info=True)
                        
                
                # 3. Reset compile tracker (clears static cache usage flags)
                reset_compile_warm_tracker(self.state)

                # 4. Delete the temporary bootstrap adapter
                if "bootstrap" in compiled_model.peft_config:
                    compiled_model.delete_adapter("bootstrap")
                    logger.debug("Bootstrap adapter removed after compilation.")

                # 5. Final CUDA Synchronization within the thread
                # This is critical. It ensures any async errors from torch.compile() or model
                # manipulation are caught here, before the function returns.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                return compiled_model

            # Prepare special tokens
            overrides = choose_special_tokens_no_add(tokenizer, actual_base_model)

            if not disable_custom_pad_ids:
                # Apply EOS/PAD overrides to the model’s generation config and tokenizer
                gcfg = actual_base_model.generation_config
                if "eos_token_id" in overrides:
                    gcfg.eos_token_id = overrides["eos_token_id"]  # lists are fine
                if "pad_token_id" in overrides:
                    tokenizer.pad_token_id = overrides["pad_token_id"]
                    pad_tok = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)
                    tokenizer.pad_token = pad_tok
                    gcfg.pad_token_id = overrides["pad_token_id"]
                    actual_base_model.config.pad_token_id = tokenizer.pad_token_id

                # If after all that, pad_token is still None, set it to eos_token.
                # This mirrors the behavior of DataCollatorForLanguageModeling and prevents
                # the training process from mutating the shared tokenizer state.
                if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                    logger.warning("Tokenizer pad_token_id is None. Setting to eos_token_id to ensure consistent state.")
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                    gcfg.pad_token_id = tokenizer.eos_token_id
            else:
                suggested_eos_id = overrides.get("eos_token_id")
                suggested_pad_id = overrides.get("pad_token_id")
                
                suggested_eos_token = tokenizer.convert_ids_to_tokens(suggested_eos_id) if suggested_eos_id is not None else "None"
                suggested_pad_token = tokenizer.convert_ids_to_tokens(suggested_pad_id) if suggested_pad_id is not None else "None"
                safe_suggested_eos_token = _safe_log_text(suggested_eos_token)
                safe_suggested_pad_token = _safe_log_text(suggested_pad_token)

                warn_msg = (
                    "Custom PAD/EOS token overrides are disabled by 'disable_custom_pad_ids=True'. "
                    f"Suggested overrides not taken: EOS={safe_suggested_eos_token}, "
                    f"PAD={safe_suggested_pad_token}. "
                    "The model's default special tokens will be used. This may affect trained adapter stability if the model's default PAD token is not well-defined."
                )
                logger.warning(warn_msg)
                init_report["warnings"].append(warn_msg)

            gcfg = actual_base_model.generation_config

            # --- Warnings for PAD token configuration ---
            if tokenizer.pad_token_id is not None:
                all_special_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, 'all_special_ids') else set()
                if tokenizer.pad_token_id not in all_special_ids:
                    pad_tok_for_warning = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)
                    warn_msg = (
                        f"The effective pad token '{_safe_log_text(pad_tok_for_warning)}' (ID: {tokenizer.pad_token_id}) "
                        "is not in the tokenizer's official list of special tokens. This may work but can indicate a misconfiguration."
                    )
                    logger.warning(warn_msg)
                    init_report["warnings"].append(warn_msg)

            pad_tok = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else "<None>"
            safe_pad_tok = _safe_log_text(pad_tok)
            
            # Handle list of EOS tokens for logging
            eos_toks = gcfg.eos_token_id
            if isinstance(eos_toks, int): 
                eos_toks = [eos_toks]
            safe_eos_tok = _safe_log_text([tokenizer.convert_ids_to_tokens(t) for t in (eos_toks or [])])

            logger.info(f"Effective EOS: {safe_eos_tok}, PAD: '{safe_pad_tok}'")
            # Store effective stop IDs/tokens on state for consistent reporting.
            effective_eos_ids = [t for t in (eos_toks or []) if isinstance(t, int)]
            effective_pad_id = tokenizer.pad_token_id if isinstance(tokenizer.pad_token_id, int) else None
            stop_ids = list(dict.fromkeys([*effective_eos_ids, *( [effective_pad_id] if effective_pad_id is not None else [])]))
            stop_tokens = [tokenizer.convert_ids_to_tokens(t) for t in stop_ids] if stop_ids else []
            self.state.effective_eos_token_ids = effective_eos_ids or None
            self.state.effective_pad_token_id = effective_pad_id
            self.state.effective_stop_token_ids = stop_ids or None
            self.state.effective_stop_tokens = stop_tokens or None

            # --- Generate and store the empty system prompt template ---
            empty_system_prompt_template = format_prompt_messages(
                logger,
                example={"messages": [{"role": "system", "content": ""}]},
                columns=ColumnsConfig(messages="messages"),
                tokenizer=tokenizer,
                add_generation_prompt=False
            ).text
            logger.debug(f"Generated empty system prompt template: {_safe_log_text(repr(empty_system_prompt_template))}")


            # --- Post-load Quantization Steps ---
            if effective_quantization_method == "bnb" or effective_quantization_method == "bnb_legacy":
                logger.info(f"Preparing model for k-bit training (BitsAndBytes quantization).")
                # Let the Trainer handle gradient checkpointing based on TrainingConfig.
                # Do not enable it here globally, as it creates a discrepancy.
                actual_base_model = prepare_model_for_kbit_training(
                    actual_base_model, 
                    use_gradient_checkpointing=False # Set to False
                )
                logger.info("Model prepared for k-bit training (GC not enabled during init).")

            # AWQ does not typically require a post-load step like this for PEFT.
            # --- End Post-load Quantization Steps ---

            #logger.debug(f"Effective Device Map: {actual_base_model.hf_device_map}")
            
            # For quantized models, model.dtype might reflect the compute dtype (e.g., bnb_4bit_compute_dtype)
            # or a quantized type for some layers.
            effective_model_dtype_str = str(actual_base_model.dtype)
            quant_config_on_model = getattr(actual_base_model.config, "quantization_config", None)
            logger.info(f"Effective Dtype: {effective_model_dtype_str}")
            logger.info(f"Effective Model Quant Config: {quant_config_on_model}")
            # Log effective attention implementation from the loaded model's config
            effective_attn_impl_from_config = getattr(actual_base_model.config, '_attn_implementation', 'N/A (not specified in config)')
            logger.info(f"Effective attention implementation used by model: {effective_attn_impl_from_config} (requested: {attn_implementation})")
            # Determine effective default context size for the engine
            model_intrinsic_max_len = 2048
            try:
                cfg = getattr(actual_base_model, "config", None)
                if cfg is not None:
                    model_intrinsic_max_len = getattr(cfg, "max_position_embeddings", model_intrinsic_max_len)
                    text_cfg = getattr(cfg, "text_config", None)
                    if text_cfg is not None:
                        model_intrinsic_max_len = getattr(text_cfg, "max_position_embeddings", model_intrinsic_max_len)
                    lang_cfg = getattr(cfg, "language_config", None)
                    if lang_cfg is not None:
                        model_intrinsic_max_len = getattr(lang_cfg, "max_position_embeddings", model_intrinsic_max_len)
                    lm_cfg = getattr(cfg, "language_model_config", None)
                    if lm_cfg is not None:
                        model_intrinsic_max_len = getattr(lm_cfg, "max_position_embeddings", model_intrinsic_max_len)
            except Exception:
                model_intrinsic_max_len = 2048
            logger.info(f"Max supported model context size: {model_intrinsic_max_len}")


            effective_default_ctx_for_engine: int
            if default_context_size is not None and default_context_size > 0: # User provided a specific size
                if default_context_size > model_intrinsic_max_len:
                    logger.warning(f"User-provided default_context_size ({default_context_size}) exceeds model's intrinsic max ({model_intrinsic_max_len}). Capping to model's max.")
                    effective_default_ctx_for_engine = model_intrinsic_max_len
                else:
                    effective_default_ctx_for_engine = default_context_size
                    logger.info(f"Requesting user-provided default_context_size: {effective_default_ctx_for_engine}")
            else: # User provided None, "auto", or 0, so use model's intrinsic max
                effective_default_ctx_for_engine = model_intrinsic_max_len
                #logger.debug(f"Requesting default_context_size from model's intrinsic max: {effective_default_ctx_for_engine}")

            # This tokenizer.model_max_length becomes the authoritative cap for all tokenization
            tokenizer.model_max_length = effective_default_ctx_for_engine

            # Set engine-level defaults on the model's config object
            if hasattr(actual_base_model, 'generation_config'):
                if default_max_new_tokens is not None:
                    actual_base_model.generation_config.max_new_tokens = default_max_new_tokens
                    logger.info(f"Set model's default max_new_tokens to engine default: {default_max_new_tokens}")

            logger.info(f"Effective context size: {tokenizer.model_max_length}")
            
            if not(hasattr(tokenizer, "apply_chat_template")):
                logger.warning("Base Model does not have chat_template")

            # --- Inspect chat templates for tool support ---
            chat_template_names: List[str] = []
            tool_template_names: List[str] = []
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                chat_template_obj = tokenizer.chat_template
                
                def _template_supports_tools(template_str: str) -> bool:
                    tool_keywords = ["tool_call", "<function_call>", "tool_calls", "{% if tools"]
                    return any(keyword in template_str for keyword in tool_keywords)

                if isinstance(chat_template_obj, str):
                    template_name = "custom" if custom_chat_template else "default"
                    chat_template_names.append(template_name)
                    if _template_supports_tools(chat_template_obj):
                        tool_template_names.append(template_name)
                elif isinstance(chat_template_obj, dict):
                    for name, template_str in chat_template_obj.items():
                        chat_template_names.append(name)
                        if _template_supports_tools(template_str):
                            tool_template_names.append(name)
            if custom_chat_template and "custom" not in tool_template_names:
                tool_template_names.append("custom")
            logger.info(f"Found chat templates: {chat_template_names}. Tool-supporting templates: {tool_template_names}")
            
            # --- Guess and store the tool parser profile ---
            if parser_profile_key_override:
                logger.info(f"Using user-provided tool parser profile key: '{parser_profile_key_override}'")
                try:
                    tool_parser_profile = profile_for(parser_profile_key_override)
                except Exception as e:
                    raise ConfigurationError(f"Failed to get profile for key '{parser_profile_key_override}': {e}")
            else:
                tool_parser_profile = guess_profile_from_template(
                    getattr(tokenizer, "chat_template", ""),
                    base_model_name_or_path
                )
                logger.info(f"Guessed tool parser profile: '{tool_parser_profile.key}'")

            engine_mode_state = EngineModeState.INFERENCE if initial_engine_mode == EngineMode.INFERENCE else EngineModeState.TRAIN
            if engine_mode_state == EngineModeState.INFERENCE:
                actual_base_model.eval()
            else:
                # Base model can remain in eval if only adapters are trained.
                actual_base_model.eval() # Default to eval, training mode will set specific model to train
            
            logger.info(f"Base model loaded. Initial mode for engine: {engine_mode_state.value}")

            global_engine_config_dump = {
                "base_model_name_or_path": base_model_name_or_path,
                "initial_engine_mode": initial_engine_mode.value,
                "instance_id": self.instance_id,
                "tools_parser_profile_key": tool_parser_profile.key,
                "no_tools_parse": no_tools_parse,
                "device_map": device_map, "trust_remote_code": trust_remote_code,
                "base_model_torch_dtype": base_model_torch_dtype, # Requested
                # Store new quantization parameters
                "quantize_bits": quantize_bits,
                "bnb_4bit_quant_type": bnb_4bit_quant_type if quantize_bits == "4" else None,
                "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype if quantize_bits == "4" else None,
                "awq_bits": awq_bits if quantize_bits == "awq" else None,
                "awq_group_size": awq_group_size if quantize_bits == "awq" else None,
                "awq_zero_point": awq_zero_point if quantize_bits == "awq" else None,
                "hqq_bits": hqq_bits if quantize_bits == "hqq" else None,
                "hqq_group_size": hqq_group_size if quantize_bits == "hqq" else None,                
                # legacy_quantization_config can also be stored if needed for full transparency
                "concurrent_generate": concurrent_generate,
                "attn_implementation": attn_implementation,
                "use_torch_compile": use_torch_compile,
                "use_cache": use_cache,               
                "static_kv_cache": static_kv_cache,
            }

            # Add the full quantization config object's dict representation for metadata saving
            if quantization_config_obj:
                global_engine_config_dump["quantization_config"] = quantization_config_obj.to_dict()

            # logger.debug(f"About to call self.state.set_global_resources...")
            # Defensive: ensure actual_base_model is a model instance, not a function
            # Only raise error if actual_base_model does not have model attributes
            if not hasattr(actual_base_model, "state_dict"):
                logger.critical(f"actual_base_model is not a valid model instance! type={type(actual_base_model)} repr={repr(actual_base_model)}")
                raise EngineInitializationError("actual_base_model is not a valid model instance. Check for shadowing or import errors.")

            peft_mixed_model_for_inference = await loop.run_in_executor(self.state._gen_exec, _create_compile_warmup_and_finalize_model)
            # --- End of PeftMixedModel creation & compilation ---

            layout = inspect_device_layout(peft_mixed_model_for_inference)
            logger.info(f"Loaded model layout: {layout}")
            is_single_device = layout.get("mode", "") == "single"
            devices = layout.get("devices", "N/A")

            logger.info(f"Base model loaded. Initial mode for engine: {engine_mode_state.value}")

            # --- Assemble the complete, effective global config ---
            # Start with the user-provided config
            effective_global_config_dump = global_engine_config_dump.copy()
            # This structure is what's stored in the state and returned to the user.
            # It contains both user-requested settings and dynamically determined effective settings.
            effective_global_config_dump = {
                # --- Top-level, most important parameters (as requested) ---
                "base_model_name": Path(base_model_name_or_path).name,
                "instance_id": self.instance_id,
                "initial_engine_mode": initial_engine_mode.value,
                "effective_attn_implementation": effective_attn_impl_from_config,
                "effective_torch_dtype": str(actual_torch_dtype),
                "effective_quantization_method": effective_quantization_method,
                "effective_quant_precision": effective_quant_precision,
                "is_compiled": use_torch_compile and hasattr(peft_mixed_model_for_inference, "_orig_mod"),
                "is_static_cache_enabled": static_kv_cache and use_cache and is_single_device and use_torch_compile and hasattr(peft_mixed_model_for_inference, "_orig_mod") and concurrent_generate == 1,
                "use_cache": use_cache,
                "max_model_context_size": model_intrinsic_max_len,
                "engine_default_context_size": tokenizer.model_max_length,
                "engine_default_max_new_tokens": default_max_new_tokens,
                "device_map": device_map,
                "model_layout": str(layout),
                "concurrent_generate": concurrent_generate,
                "no_tools_parse": no_tools_parse,
                "tools_parser_profile_key": tool_parser_profile.key,
            }

            # --- Conditionally add nested dictionary for detailed quantization info ---
            if quantize_bits != "none":
                effective_global_config_dump["quantization_details"] = {
                        k: v for k, v in {
                            "quantize_bits": quantize_bits,
                            "bnb_4bit_quant_type": bnb_4bit_quant_type if quantize_bits == "4" else None,
                            "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype if quantize_bits == "4" else None,
                            "awq_bits": awq_bits if quantize_bits == "awq" else None,
                            "awq_group_size": awq_group_size if quantize_bits == "awq" else None,
                            "awq_zero_point": awq_zero_point if quantize_bits == "awq" else None,
                            "hqq_bits": hqq_bits if quantize_bits == "hqq" else None,
                            "hqq_group_size": hqq_group_size if quantize_bits == "hqq" else None,
                        }.items() if v is not None
                    }

            # --- Nested dictionary for other configuration settings ---
            effective_global_config_dump["other_config"] = {
                "base_model_name_or_path": base_model_name_or_path,
                "trust_remote_code": trust_remote_code,
                "requested_torch_dtype": base_model_torch_dtype,
                "requested_attn_implementation": attn_implementation,
                "use_torch_compile_request": use_torch_compile,
                "static_kv_cache_request": static_kv_cache,
                "tool_parser_profile": asdict(tool_parser_profile) if tool_parser_profile else None,
                "empty_system_prompt_template": empty_system_prompt_template,
                "custom_chat_template": custom_chat_template,
            }
            
            # Augment it with dynamically determined settings
            is_compiled = use_torch_compile and hasattr(peft_mixed_model_for_inference, "_orig_mod")

            # Static cache is only truly enabled if all below are true
            is_static_cache_enabled_for_session =  static_kv_cache and use_cache and is_single_device and use_torch_compile and is_compiled and concurrent_generate == 1
            initialize_cache_session_config(self.state, static_kv_cache_enabled=is_static_cache_enabled_for_session)

            # --- Auto-disable static cache if torch.compile is off or failed ---
            if static_kv_cache and not is_static_cache_enabled_for_session:
                warn_mess = "???"
                if not use_cache:
                    warn_mess = "Static KV cache is suppressed because of global use_cache=False override"
                elif not use_torch_compile:
                    warn_mess = "Static KV cache is suppressed becaue torch.compile is disabled by config."
                elif not is_compiled: 
                    warn_mess = "Static KV cache is suppressed because torch compile is unavailbale or failed."
                elif not is_single_device: 
                    warn_mess = f"Static KV cache is suppressed because model mode '{layout.get("mode", "")}' is not on single GPU or devices='{devices}'."
                elif concurrent_generate > 1:
                    warn_mess = f"Static KV cache is suppressed because concurrent_generate ({concurrent_generate}) is > 1"
                logger.warning(warn_mess)
                init_report["warnings"].append(warn_mess)
            # These are now part of the top-level parameters
            effective_global_config_dump["is_compiled"] = is_compiled
            effective_global_config_dump["is_static_cache_enabled"] = is_static_cache_enabled_for_session
            effective_global_config_dump["chat_templates"] = chat_template_names
            effective_global_config_dump["tool_templates"] = tool_template_names
            effective_global_config_dump["tool_parser_profile"] = asdict(tool_parser_profile) if tool_parser_profile else None
            effective_global_config_dump["empty_system_prompt_template"] = empty_system_prompt_template
            effective_global_config_dump["use_separate_stream"] = True # for concurrent Cuda streams
            # --- End of config assembly ---

            await self.state.set_global_resources(
                base_model=actual_base_model, 
                peft_model=peft_mixed_model_for_inference,
                tokenizer=tokenizer, 
                tokenizer_for_warmup=tokenizer_for_warmup,
                device=str(layout), 
                global_config_dump=effective_global_config_dump, 
                initial_mode=engine_mode_state)
            
            self.state._update_gpu_memory_stats() # Initial memory stats
            # Explicitly set component statuses based on initial mode
            if initial_engine_mode == EngineMode.INFERENCE:
                await self.state.set_inference_status(InferenceStatus.READY, "Engine initialized in INFERENCE mode.")
                await self.state.set_training_status(TrainingStatus.OFFLINE, "Engine initialized in INFERENCE mode.")
            elif initial_engine_mode == EngineMode.TRAIN:
                await self.state.set_training_status(TrainingStatus.READY, "Engine initialized in TRAIN mode.")
                await self.state.set_inference_status(InferenceStatus.OFFLINE, "Engine initialized in TRAIN mode.")
            
            #TODO: migrate onto quant method and details from effective state config
            self.state.base_model_quantization_cfg = quantization_config_obj.to_dict() if quantization_config_obj else None
            
            await self.state.set_active_adapter_state(None, None) # No active adapter initially
            await self.state.set_server_status(ServerStatus.READY, "Global resources initialized.")
            logger.info(f"--- Global Engine Resources Initialized (Total Time: {time.time() - run_start_time_wall:.2f}s) ---")

            # --- Freeze config on first successful init ---
            with _frozen_config_lock:
                if _FROZEN_ENGINE_CONFIG is None:
                    _FROZEN_ENGINE_CONFIG = copy.deepcopy(config)
                    logger.info("First engine initialized. Freezing global configuration settings.")
            # --- End freeze ---
       
            # --- Construct final response using the single source of truth ---
            effective_global_config = await self._get_effective_global_config_dict()
            init_report["global_config"] = effective_global_config
            
            return init_report

        except Exception as e:
            error_msg = f"Global Engine Initialization Failed: {type(e).__name__}: {e}"
            logger.critical(error_msg, exc_info=True)
            init_report["errors"].append(error_msg)
            await self.state.clear_global_resources()
            await self.state.set_server_status(ServerStatus.ERROR, f"Global init failed: {error_msg}") # Call set_server_status after clear
            # Even on failure, return the report
            init_report["message"] = error_msg            
            raise EngineInitializationError({"message":error_msg, "details":init_report}) from e

    async def checkout_resource(self, request_id: str) -> "RequestResource":
        """
        Checks out a combined resource (tokenizer, stream) from the pool.
        This method is the concurrency gate for inference requests.
        If the pool is empty, lazily clones a new tokenizer.
        Waits if max concurrency is reached (i.e., semaphore is not available).
        """
        if not self.state._resource_semaphore:
            raise EngineError("Resource pool is not initialized. Cannot check out a resource.")

        # --- Concurrency Gate and Logging ---
        start_time = time.perf_counter()
        await self.state._resource_semaphore.acquire()
        waited_seconds = time.perf_counter() - start_time
        
        if waited_seconds > 0.01: # Use a small threshold to avoid logging for tiny yields
            logger.debug(f"Inference request '{request_id}' was blocked for {waited_seconds:.3f}s waiting for a resource, now proceeding. (Limit: {self.state.concurrent_generate})")
        # --- End Concurrency Gate ---

        try:
            resource = None
            # Check the pool first. This is fast.
            async with self.state._resource_pool_lock:
                if self.state._resource_pool:
                    resource = self.state._resource_pool.pop(0) # FIFO

            # If pool was empty, lazily create a new one. This is the expensive part.
            if resource is None:
                logger.info(f"Resource pool empty, lazily creating new resource for request '{request_id}'.")
                
                # Create tokenizer
                new_tokenizer = await asyncio.to_thread(copy.deepcopy, self.state.tokenizer)
                
                # Create stream
                new_stream = None
                use_separate_stream = False
                if self.state.global_config:
                    is_enabled_in_config = self.state.global_config.get("use_separate_stream", False)
                    use_separate_stream = is_enabled_in_config and torch.cuda.is_available()
                
                if use_separate_stream:
                    target_device = first_module_device(self.state.peft_model)
                    # Only create a CUDA stream if the target device is a CUDA device.
                    if target_device.type == "cuda":
                        new_stream = torch.cuda.Stream(device=target_device, priority=0)
                        # Attach a small pinned staging buffer cache to the stream for H2D copies of tiny payloads
                        try:
                            new_stream.mp13_pinned_buffers = {}
                            new_stream.mp13_pinned_bytes_cap = 2 * 1024 * 1024  # 2 MiB cap for all cached pinned buffers
                            new_stream.mp13_pinned_bytes = 0
                        except Exception:
                            pass
                    else:
                        self.state.logger.debug(f"Cannot create CUDA stream for device type: {target_device.type}. Skipping stream creation.")
                resource = RequestResource(tokenizer=new_tokenizer, stream=new_stream)

            # Register the instance as in-use.
            async with self.state._resource_pool_lock:
                self.state._resource_in_use.add(resource)
            
            return resource
        except Exception:
            # If something goes wrong after acquiring semaphore, release it to prevent deadlock
            self.state._resource_semaphore.release()
            raise

    async def checkin_resource(self, resource: "RequestResource"):
        """Returns a resource to the pool."""
        should_release_semaphore = False
        async with self.state._resource_pool_lock:
            if resource in self.state._resource_in_use:
                self.state._resource_in_use.remove(resource)
                self.state._resource_pool.append(resource)
                should_release_semaphore = True
        
        if should_release_semaphore and self.state._resource_semaphore:
            self.state._resource_semaphore.release()

    async def shutdown_global_resources(self):
        logger.info(f"--- Shutting Down Global Engine Resources ({self.instance_id}) ---")

        if self.state.base_model is None: 
            logger.info("Global resources already shut down or not initialized.")
            return {"message": "Global resources already shut down."}

        # This is a signal for being scheduled tasks to bail out 
        async with self.state._lock:
            self.state._prevent_new_background_tasks = True 

        await self.state.set_server_status(ServerStatus.SHUTTING_DOWN, "Global shutdown initiated.")

        # --- Proactive Cleanup and Error Flushing ---
        # This is a critical defensive measure. Asynchronous CUDA errors, especially OOM,
        # can be triggered by the garbage collection of tensors that have gone out of scope
        # *before* shutdown was even called. By explicitly collecting garbage and emptying
        # the cache here, we force any such latent errors to surface immediately.
        # We wrap this in a try/except block so that if an OOM is caught, we can log it
        # informatively and continue with the rest of the shutdown, rather than crashing.
        if torch.cuda.is_available():
            import gc
            try:
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Proactively flushed garbage and CUDA cache before shutdown.")
            except RuntimeError as e:
                logger.warning(f"Caught a latent CUDA error during pre-shutdown flush: {e}. This is often an OOM from a prior operation. Continuing with shutdown.")

        # Check training status
        safe_training_statuses = [
            TrainingStatus.OFFLINE, TrainingStatus.READY, TrainingStatus.COMPLETED,
            TrainingStatus.ERROR, TrainingStatus.STOPPED
        ]
        if self.state.training_status not in safe_training_statuses:
            raise BusyError(f"Cannot shutdown. Training operations busy: {self.state.training_status.value}")

        # --- Robust Shutdown Logic for Inference ---
        # This loop waits for all inference and cache warming to complete.
        wait_for_idle_start_time = time.monotonic()
        max_wait_for_idle_sec = 60 # Total time to wait for engine to become idle.
        
        # --- Clear any pending (not yet started) cache warmup requests ---
        if self.state._pending_warmup_queue:
            async with self.state._lock:
                queue_size = len(self.state._pending_warmup_queue)
                if queue_size > 0:
                    logger.info(f"Clearing {queue_size} pending cache warmup request(s) due to shutdown.")
                    self.state._pending_warmup_queue.clear()
        
        sleep_secs = 2
        while self.state.inference_status not in [InferenceStatus.READY, InferenceStatus.OFFLINE, InferenceStatus.ERROR]:
            if time.monotonic() - wait_for_idle_start_time > max_wait_for_idle_sec:
                logger.error(f"Timeout ({max_wait_for_idle_sec}s) waiting for inference to become idle. Forcing shutdown. Final status: {self.state.inference_status.value}")
                break

            current_status = self.state.inference_status
            logger.info(f"Shutdown waiting for idle state. Current inference status: {current_status.value}")

            if current_status == InferenceStatus.WARMING_CACHE:
                logger.info("Cache warming in progress. Waiting for completion...")
                # The background task will eventually change the state. We just wait.
            
            elif current_status == InferenceStatus.INFERRING:
                if self.adapters_control.has_active_or_pending_requests():
                    # Delegate to AdaptersControl to signal all pending and active requests
                    try:
                        pending_cancelled, active_cancelled = await self.adapters_control.cancel_request(
                            request_id=None, cancel_ops= [CohortTaskType.LOAD_ADAPTER, CohortTaskType.UNLOAD_ADAPTER, CohortTaskType.SET_ADAPTERS]
                            )
                        logger.info(f"Cancellation signalled for active requests: pending={pending_cancelled}, active={active_cancelled}. Waiting for cleanup...")
                    except Exception as e_cancel:
                        logger.warning(f"Failed to signal cancellation via AdaptersControl: {e_cancel}")
                else:
                    # This is the "stuck inferring" state. A 'finally' block is likely running.
                    # We just wait for it to complete the state transition.
                    logger.info("Status is 'inferring' but no active requests tracked. Waiting for state to update...")
            
            await asyncio.sleep(sleep_secs) # Polling interval to check the status.
            sleep_secs = 20 # waith for longer next time

        if self.state.inference_status in [InferenceStatus.READY, InferenceStatus.OFFLINE, InferenceStatus.ERROR]:
            logger.info(f"Inference component is idle (status: {self.state.inference_status.value}). Proceeding with shutdown.") # type: ignore
        
        # --- Reset static cache on shutdown ---
        reset_static_cache(self.state)
        
        if self.state._gen_exec:
            logger.info("Shutting down generation executor...")
            await asyncio.to_thread(self.state._gen_exec.shutdown, wait=False, cancel_futures=True)

        if self.state._bg_exec:
            logger.info("Shutting down background executor...")
            await asyncio.to_thread(self.state._bg_exec.shutdown, wait=False, cancel_futures=True)

        # --- Clear resource pool ---
        self.state._resource_pool.clear()
        self.state._resource_in_use.clear()
        self.state._resource_semaphore = None
        self.state._inference_metrics_history.clear()

        if self.state.tokenizer_for_warmup:
            del self.state.tokenizer_for_warmup
            self.state.tokenizer_for_warmup = None
            logger.debug("Deleted static tokenizer for cache warming.")
        self.state.aggregate_metrics = AggregateInferenceMetrics()
        await self.state.clear_global_resources()

#       NB: needs tracking featue for each threading.Thread() call and  _bg_threads member
#        for t in list(self._bg_threads):        # copy in case list mutates
#            if t.is_alive():
#                self.logger.debug("Joining thread %s", t.name)
#                t.join(timeout=1.0)             # wait up to 1 s each
#        self._bg_threads.clear()

        # Explicitly delete large objects to help Python's garbage collector
        # release VRAM before we clear the CUDA cache.
        if self.state.peft_model:
            del self.state.peft_model
            self.state.peft_model = None
        if self.state.base_model:
            del self.state.base_model
            self.state.base_model = None
        if self.state.tokenizer:
            del self.state.tokenizer
            self.state.tokenizer = None

        if torch.cuda.is_available():
            import gc

            # Determine the devices used by the model.
            used_devices = set()
            if self.state.base_model:
                for param in self.state.base_model.parameters():
                    used_devices.add(param.device)

            try:
                # Flush any pending kernels/errors.
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"WARNING: cuda.synchronize raised during shutdown: {e}")

            try:
                gc.collect()

                # Iterate through the used devices and clear the cache.
                for device in used_devices:
                    if device.type == 'cuda':
                        try:
                            with torch.cuda.device(device):
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                                logger.info(f"Garbage collected and CUDA cache/IPCs cleared for device {device}.")
                        except Exception as e:
                            logger.warning(f"Ignoring error during cache clear on device {device}: {e}")

                # Reset peak stats.
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

                logger.info("Garbage collected and CUDA caches/IPCs cleared after model deletion.")
            except Exception as e:
                logger.warning(f"Ignoring error during final cache clear: {e}")


        await self.state.set_server_status(ServerStatus.OFFLINE, "Global resources shut down.")
        logger.info("Global engine resources shut down successfully.")
        return {"message": "Global resources shut down."}

    # --- Subscription Methods ---
    async def subscribe_to_training_status(self, callback: Callable):
        """Subscribes a callback to receive training status updates."""
        async with self.state._lock:
            if callback in self.state._training_status_subscribers:
                logger.warning(f"Callback {getattr(callback, '__name__', 'unknown')} is already subscribed to training status. Ignoring duplicate.")
                return
            self.state._training_status_subscribers.append(callback)
        logger.info(f"Callback {getattr(callback, '__name__', 'unknown')} subscribed to training status.")

    async def unsubscribe_from_training_status(self, callback: Callable):
        """Unsubscribes a callback from training status updates."""
        async with self.state._lock:
            if callback in self.state._training_status_subscribers:
                self.state._training_status_subscribers.remove(callback)
        logger.info(f"Callback {getattr(callback, '__name__', 'unknown')} unsubscribed from training status.")

    async def subscribe_to_inference_status(self, callback: Callable):
        """Subscribes a callback to receive inference status updates."""
        async with self.state._lock:
            if callback in self.state._inference_status_subscribers:
                logger.warning(f"Callback {getattr(callback, '__name__', 'unknown')} is already subscribed to inference status. Ignoring duplicate.")
                return
            self.state._inference_status_subscribers.append(callback)
        logger.info(f"Callback {getattr(callback, '__name__', 'unknown')} subscribed to inference status.")

    async def unsubscribe_from_inference_status(self, callback: Callable):
        """Unsubscribes a callback from inference status updates."""
        async with self.state._lock:
            if callback in self.state._inference_status_subscribers:
                self.state._inference_status_subscribers.remove(callback)
        logger.info(f"Callback {getattr(callback, '__name__', 'unknown')} unsubscribed from inference status.")

    async def subscribe_to_engine_events(self, callback: Callable):
        """Subscribes a callback to receive general engine events (e.g., server status, mode changes)."""
        async with self.state._lock:
            if callback in self.state._engine_event_subscribers:
                logger.info(f"Callback {getattr(callback, '__name__', 'unknown')} is already subscribed to engine events. Ignoring duplicate.", time.time())
                return
            self.state._engine_event_subscribers.append(callback)
        logger.info(f"Callback {getattr(callback, '__name__', 'unknown')} subscribed to engine events.")

    async def unsubscribe_from_engine_events(self, callback: Callable):
        async with self.state._lock:
            if callback in self.state._engine_event_subscribers:
                self.state._engine_event_subscribers.remove(callback) # type: ignore
        logger.info(f"Callback {getattr(callback, '__name__', 'unknown')} unsubscribed from engine events.")


    async def _internal_cancel_current_operation(self, operation_being_cancelled: str, request_id: Optional[str] = None) -> bool:
        """
        Attempts to cancel the currently running operation (training or inference).
        Returns True if cancellation was successfully initiated, False otherwise.
        """
        self._check_engine_readiness()
        logger.debug("CANCEL-REQ")
        logger.info(f"--- Internal: Attempting to cancel {operation_being_cancelled} ({self.instance_id}) ---")
        if operation_being_cancelled == "training" and self.state.training_status == TrainingStatus.TRAINING:
            # This is the new logic for forceful cancellation of training.
            # It sets flags that the training thread and callbacks can see,
            # and then cancels the asyncio task waiting on the thread.
            logger.info("Forcefully cancelling training: setting flags and cancelling task.")
            self.state._graceful_stop_requested = True # Ask callback to stop the trainer loop
            self.state._training_cancelled = True # Prevent post-train cleanup in the thread
            
            if self.state._training_task and not self.state._training_task.done():
                logger.info("Sending cancellation to training task wrapper.")
                self.state._training_task.cancel("Operation cancelled by request.")
                # The task's `except CancelledError` will handle the state change.
                return True
            logger.warning("No active/cancellable training task found, but flags were set.")
            # Even if task is not found, we set flags and can consider it "processed"
            await self.state.set_training_error(f"Cancelled by API request at step {self.state.current_step} (no task found).")
            return True
        elif operation_being_cancelled == "training_prep" and self.state.training_status == TrainingStatus.PREPARING:
            # If PREPARING, there's no async task yet, just reset state
            await self.state.set_training_error(f"Preparation cancelled by API request.")
            return True
        elif operation_being_cancelled == "inference" and request_id:
            logger.info(f"Inference cancellation requested for request ID '{request_id}'. Delegating to AdaptersControl.")
            pending_cancelled, active_cancelled = await self.adapters_control.cancel_request(request_id)
            if pending_cancelled > 0 or active_cancelled > 0:
                return True
            logger.warning(f"Could not find active or pending inference request with ID '{request_id}' to cancel.")
            return False
        elif operation_being_cancelled == "inference" and not request_id:
            logger.error("_internal_cancel_current_operation called for inference without a request_id.")
            return False

        logger.info(f"No active {operation_being_cancelled} operation to cancel.")
        return False

    async def check_set_mode(self, mode: EngineMode, force: bool = False) -> Dict[str, Any]: # noqa: C901
        """
        Sets the engine's operational mode (TRAIN/INFERENCE).
        If force=True, attempts to cancel ongoing operations before switching.
        Returns the effective mode.
        """
        self._check_engine_readiness()
        logger.info(f"--- Received Check/Set Mode to: {mode.value} (force={force}) ({self.instance_id}) ---")
        
        is_forced_switch = False
        if self.state.engine_mode != EngineModeState(mode.value):
            is_busy = (self.state.training_status in [TrainingStatus.PREPARING, TrainingStatus.TRAINING] or
                       self.state.inference_status == InferenceStatus.INFERRING)
            if is_busy and force:
                is_forced_switch = True

        if is_forced_switch:
            self.state._prevent_new_background_tasks = True
            logger.info("Mode switch: Set flag to prevent new background tasks.")

        try:
            current_wall_time = time.time()

            target_mode_state = EngineModeState(mode.value)
            current_mode_state = self.state.engine_mode

            if current_mode_state == target_mode_state:
                logger.info(f"Engine already in {mode.value} mode.")
                # ... (rest of the block is unchanged)
                if target_mode_state == EngineModeState.INFERENCE and self.state.inference_status != InferenceStatus.READY:
                    logger.info(f"Mode is {mode.value}, but inference status is {self.state.inference_status.value}. Setting to READY.")
                    await self.state.set_inference_status(InferenceStatus.READY, "Correcting status: Engine already in INFERENCE mode.")
                elif target_mode_state == EngineModeState.TRAIN and self.state.training_status != TrainingStatus.READY:
                    logger.info(f"Mode is {mode.value}, but training status is {self.state.training_status.value}. Setting to READY.")
                    await self.state.set_training_status(TrainingStatus.READY, "Correcting status: Engine already in TRAIN mode.")
                return {"message": f"Engine already in {mode.value} mode.", "effective_mode": mode.value,}

            # Check if busy
            busy_operation = None
            if self.state.training_status == TrainingStatus.PREPARING:
                busy_operation = "training_prep"
            elif self.state.training_status == TrainingStatus.TRAINING:
                busy_operation = "training"
            elif self.state.inference_status == InferenceStatus.INFERRING:
                busy_operation = "inference"

            if busy_operation:
                if not force:
                    msg = f"Cannot change mode. Engine is busy with {busy_operation}. Use force=true or stop/cancel the operation."
                    logger.error(msg)
                    raise BusyError({"message": msg,
                                     "effective_mode": current_mode_state.value if current_mode_state else "UNKNOWN", 
                                    })
                else:
                    # Flag is already set if is_forced_switch is true
                    logger.info(f"Force mode switch: Attempting to cancel ongoing '{busy_operation}'.")
                    if busy_operation == "inference":
                        await self._internal_cancel_current_operation("inference")
                        cancelled_successfully = True # Assume cancellation signals were sent
                    else:
                        cancelled_successfully = await self._internal_cancel_current_operation(busy_operation)
                    
                    if not cancelled_successfully and busy_operation == "inference": # Special handling for non-cancellable inference
                        msg = f"Force mode switch failed. Active '{busy_operation}' could not be reliably cancelled. Please wait."
                        logger.error(msg)
                        raise BusyError({"message": msg,
                                        "effective_mode": current_mode_state.value if current_mode_state else "UNKNOWN", 
                                        })
                    elif not cancelled_successfully: # For training if it wasn't running or task was gone
                         logger.warning(f"No active {busy_operation} operation found to cancel/already handled, or cancellation failed. Proceeding with mode switch.")
                    else: # Cancellation initiated
                        await asyncio.sleep(0.1) # Give a moment for async cancellation to propagate
                        if busy_operation == "training" and self.state.training_status == TrainingStatus.TRAINING:
                            msg = f"Force mode switch failed. Training cancellation did not result in prompt state change."
                            logger.error(msg)
                            raise EngineError({"message": msg,
                                            "effective_mode": current_mode_state.value if current_mode_state else "UNKNOWN", 
                                             })
                        logger.info(f"Ongoing '{busy_operation}' cancellation initiated. Proceeding with mode switch.")

            # Reset the state of the mode we are LEAVING
            if current_mode_state == EngineModeState.TRAIN:
                # Only cancel the training task if it's still running.
                # This prevents cancelling a completed task during a fast mode switch after training.
                if self.state._training_task and not self.state._training_task.done():
                    logger.info("Mode switch: Cancelling in-progress training task.")
                    self.state._training_task.cancel("Mode switch initiated.")
                    # The task's `except CancelledError` will handle the state change.

                # Restore the active adapters to their state before training started.
                if self.state._adapters_active_before_training is not None:
                    logger.info(f"Restoring active adapters to pre-training state: {self.state._adapters_active_before_training}")
                    await self.set_active_adapter(self.state._adapters_active_before_training)
                    self.state._adapters_active_before_training = None # Clear after restoring

                # Freeze all adapter params so casting utilities treat them as non-trainable
                def _freeze_lora_params(model):
                    for n, p in model.named_parameters():
                        if "lora_" in n:
                            p.requires_grad = False
                _freeze_lora_params(self.state.peft_model)

                await self.state.reset_training()
                logger.info("Training state reset. All trained adapters persist for inference.")
                if self.state.base_model and hasattr(self.state.base_model.config, "use_cache"):
                    logger.info(f"Restoring model.config.use_cache to global config value: {self.state.use_cache} for inference mode.")
                    self.state.base_model.config.use_cache = self.state.use_cache
                    if hasattr(self.state.base_model, "generation_config"):
                        self.state.base_model.generation_config.use_cache = self.state.use_cache
            elif current_mode_state == EngineModeState.INFERENCE:
                await self.state.reset_inference()
                logger.info("Inference state reset due to mode switch.")

            # --- State modifications for ENTERING a mode ---
            if target_mode_state == EngineModeState.TRAIN:
                # Set use_cache=False for training to save VRAM and prevent warnings.
                if self.state.global_config and self.state.global_config.get("quantize_bits") == "hqq":
                    _set_hqq_backend_for_mode(self.state, self.state.base_model, EngineMode.TRAIN)
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                if self.state.base_model and hasattr(self.state.base_model.config, "use_cache"):
                    self.state.base_model.config.use_cache = False
                    if hasattr(self.state.base_model, "generation_config"):
                        self.state.base_model.generation_config.use_cache = False
                    logger.info("Set model.config.use_cache=False for TRAIN mode.")
                self.state._adapters_active_before_training = await self.state.active_adapter_names()
                logger.info(f"Saved active adapters for inference mode: {self.state._adapters_active_before_training}")
                logger.info("Switching to TRAIN mode. Resetting static cache.")
                reset_static_cache(self.state)
            elif target_mode_state == EngineModeState.INFERENCE:
                if self.state.global_config and self.state.global_config.get("quantize_bits") == "hqq":
                    _set_hqq_backend_for_mode(self.state, self.state.base_model, EngineMode.INFERENCE)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            elif target_mode_state == EngineModeState.INFERENCE:
                # --- Re-initialize inference session config when entering INFERENCE mode ---
                if self.state.global_config:
                    # Use the new property to get the effective value
                    initialize_cache_session_config(self.state, self.state.is_static_cache_enabled)


            await self.state.set_engine_mode_state(target_mode_state)

            if target_mode_state == EngineModeState.TRAIN:
                await self.state.set_training_status(TrainingStatus.READY, "Engine mode set to TRAIN, ready for training config.")
                await self.state.set_inference_status(InferenceStatus.OFFLINE, "Engine mode set to TRAIN.")
            elif target_mode_state == EngineModeState.INFERENCE:
                if self.state.peft_model: self.state.peft_model.eval()
                await self.state.set_inference_status(InferenceStatus.READY, "Engine mode set to INFERENCE and ready for requests.")
                await self.state.set_training_status(TrainingStatus.OFFLINE, "Engine mode set to INFERENCE.")

            msg = f"Engine mode successfully set to {mode.value}."
            return {"effective_mode": mode.value, "message": msg}
        finally:
            if is_forced_switch:
                self.state._prevent_new_background_tasks = False
                logger.info("Mode switch: Reset flag for background tasks.")

    def _normalize_cancel_ops(
        self,
        cancel_ops: Optional[Union[CohortTaskType, str, List[Union[CohortTaskType, str]], Tuple[Union[CohortTaskType, str], ...]]]
    ) -> Optional[List[CohortTaskType]]:
        if cancel_ops is None:
            return None
        if isinstance(cancel_ops, (CohortTaskType, str)):
            try:
                return [CohortTaskType(cancel_ops)]
            except ValueError as e:
                raise ConfigurationError(f"Invalid cancel_ops value: {cancel_ops}") from e
        if isinstance(cancel_ops, (list, tuple)):
            normalized: List[CohortTaskType] = []
            for op in cancel_ops:
                try:
                    normalized.append(CohortTaskType(op))
                except ValueError as e:
                    raise ConfigurationError(f"Invalid cancel_ops value: {op}") from e
            return normalized
        raise ConfigurationError("cancel_ops must be a string or list of strings.")

    async def cancel_request(
        self,
        request_id: Optional[str] = None,
        cancel_ops: Optional[Union[CohortTaskType, str, List[Union[CohortTaskType, str]], Tuple[Union[CohortTaskType, str], ...]]] = None,
        cancel_for_adapter_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Attempts to forcefully cancel training or inference operations."""
        """
        Delegates cancellation to the AdaptersControl, which now handles both
        pending (queued) and active (running) inference requests.
        Supports:
        - request_id: cancel a specific request.
        - cancel_ops: cancel pending cohort operations (adapter load/unload/set).
        - cancel_for_adapter_name: cancel requests using a specific adapter.
        """
        self._check_engine_readiness()
        logger.info(f"--- Cancel Request API Call (req_id: {request_id or 'ALL'}) ---")

        # Training cancellation remains separate as it's not cohort-based.
        if self.state.training_status in [TrainingStatus.PREPARING, TrainingStatus.TRAINING]:
            op_to_cancel = "training_prep" if self.state.training_status == TrainingStatus.PREPARING else "training"
            cancelled = await self._internal_cancel_current_operation(op_to_cancel)
            return {"message": f"{op_to_cancel.replace('_', ' ').capitalize()} cancellation processed."} if cancelled else {"message": "No active training to cancel."}

        # Delegate inference cancellation to the cohort-aware method.
        normalized_cancel_ops = self._normalize_cancel_ops(cancel_ops)
        pending_cancelled, active_cancelled = await self.adapters_control.cancel_request(
            request_id=request_id,
            cancel_ops=normalized_cancel_ops,
            cancel_for_adapter_name=cancel_for_adapter_name
        )

        if pending_cancelled == 0 and active_cancelled == 0:
            return {"status": APIStatus.NO_OP.value, "message": "No matching active or pending inference requests found to cancel."}

        return {"message": f"Cancellation signal sent to {pending_cancelled} pending and {active_cancelled} active requests."}

    # --- Training and Inference Delegation ---
    async def start_training(self, training_config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Delegates to mp13_train.start_training_logic"""
        self._check_engine_readiness()
        await self._check_and_recover_state()
        return await start_training_logic(self, training_config_data)

    async def stop_training(self) -> Dict[str, Any]:
        """Delegates to mp13_train.stop_training_logic"""
        self._check_engine_readiness()
        # No recovery check here, as stop should be able to run even if another component is in error.
        return await stop_training_logic(self)

    async def _execute_training_run(self, config: TrainingConfig):
        """Delegates to mp13_train.execute_training_run_logic"""
        return await execute_training_run_logic(self, config)

    async def run_inference(self, request: InferenceRequest) -> AsyncIterator[InferenceResponse]:
        """Delegates to mp13_infer.run_inference_logic"""
        self._check_engine_readiness()
        await self._check_and_recover_state()
        # The run_inference_logic function now handles its own state management (setting status,
        # adding/removing active requests, and handling completion/cancellation).
        # This wrapper just needs to pass through the yielded items and exceptions.
        async for item in run_inference_logic(self, request):
            yield item

    async def format_inference_prompt(self, request: InferenceRequest, request_tokenizer: Optional[Any] = None) -> Dict[str, Any]:
        """Delegates to mp13_infer.format_inference_prompt_logic"""
        self._check_engine_readiness()
        await self._check_and_recover_state()
        # The logic inside format_inference_prompt_logic will handle exceptions
        # and return a dict. The API layer will then format this into a response.
        return await format_inference_prompt_logic(self, request, request_tokenizer=request_tokenizer)

    async def count_tokens(self, text: str, is_repr: bool = False, request_tokenizer: Optional[Any] = None) -> Dict[str, Any]:
        """Delegates to mp13_infer.count_tokens_logic to count tokens in a string."""
        self._check_engine_readiness()
        await self._check_and_recover_state()
        # The logic function will handle resource checkout/checkin if tokenizer is not provided.
        return await count_tokens_logic(self, text, is_repr, request_tokenizer)

    async def get_training_status(self) -> Dict[str, Any]:
        """Returns the current training status payload."""
        return await self.state.get_training_status_dict()

    async def get_inference_status(self) -> Dict[str, Any]:
        """Returns the current inference status payload."""
        return await self.state.get_inference_status_dict()

    async def get_engine_status(self) -> Dict[str, Any]:
        """Returns the server status with GPU and PyTorch details."""
        status_data = await self.state.get_server_status_dict()
        if torch.cuda.is_available():
            try:
                gpu_info_list = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    mem_stats = torch.cuda.memory_stats(i)
                    gpu_info_list.append({
                        "device_id": i, "name": props.name,
                        "memory_allocated_gb": round(mem_stats.get("allocated_bytes.all.current", 0) / 1e9, 2),
                        "memory_reserved_gb": round(mem_stats.get("reserved_bytes.all.current", 0) / 1e9, 2),
                        "memory_total_gb": round(props.total_memory / 1e9, 2),
                    })
                status_data["gpu_info"] = gpu_info_list
            except Exception as gpu_e:
                status_data["gpu_info"] = f"Error retrieving detailed GPU info: {gpu_e}"
        else:
            status_data["gpu_info"] = "CUDA not available"

        status_data["pytorch_version"] = torch.__version__
        return status_data

    async def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Returns aggregate inference metrics."""
        return await self.state.get_aggregate_metrics()

    async def get_loaded_adapters(self) -> List[Dict[str, Any]]:
        """Returns loaded adapter details with active status."""
        loaded_adapters_info = await self.state.get_loaded_adapters_info()
        active_adapter_names = await self.state.active_adapter_names()
        adapters_list = []
        for name, info in loaded_adapters_info.items():
            meta = info.get("metadata") or {}
            quant_display = info.get("base_model_quant") or quant_display_from_meta(meta)
            adapters_list.append({
                "name": name,
                "is_active": name in active_adapter_names,
                "is_loaded": True,
                "is_foreign": info.get("is_foreign", False),
                "type": info.get("type"),
                "root_path": info.get("root_path"),
                "checkpoint_path": info.get("checkpoint_path"),
                "alias": info.get("alias"),
                "base_model_quant": quant_display,
                "base_model_name": info.get("base_model_name"),
                "metadata": meta,
            })
        return adapters_list

    async def get_adapter_details(self, adapter_name: str) -> Dict[str, Any]:
        """Returns detailed info for a specific adapter."""
        loaded_adapters_info = await self.state.get_loaded_adapters_info()
        if adapter_name not in loaded_adapters_info:
            raise AdapterError(f"Adapter '{adapter_name}' not found.")
        adapter_details = dict(loaded_adapters_info[adapter_name])
        active_adapter_names = await self.state.active_adapter_names()
        adapter_details["is_active"] = adapter_name in active_adapter_names
        adapter_details["name"] = adapter_name
        return adapter_details

    # --- Adapter Management Delegation ---
    async def load_adapter(self, adapter_config: AdapterConfig) -> Dict[str, Any]:
        """Adds a new adapter using the AdaptersControl and returns a status dict."""
        self._check_engine_readiness()
        await self._check_and_recover_state()
        return await self.adapters_control.load_adapter(adapter_config)

    async def set_active_adapter(self, adapter_names_to_set: Union[None, str, List[str]]) -> Dict[str, Any]:
        """Sets the active adapter by name using the AdaptersControl and returns a status dict."""
        self._check_engine_readiness()
        await self._check_and_recover_state()
        return await self.adapters_control.set_active_adapter(adapter_names_to_set)

    async def unload_adapter(self, adapter_name: str) -> Dict[str, Any]:
        """Unloads the specified adapter using the AdaptersControl and returns a status dict."""
        self._check_engine_readiness()
        await self._check_and_recover_state()
        return await self.adapters_control.unload_adapter(adapter_name)

    async def list_all_adapters(self, root_folder: str, include_incompatible: bool = False, probe_adapter_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns adapter discovery info for a root folder using AdaptersControl."""
        self._check_engine_readiness()
        await self._check_and_recover_state()
        return await self.adapters_control.get_adapter_names(root_folder, include_incompatible=include_incompatible, adapter_name=probe_adapter_name)
