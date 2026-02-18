# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""State management for MP13 server (training and inference)."""
import os
import logging
import time
import asyncio
import traceback, concurrent.futures
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Set, Deque, Union, Callable, TYPE_CHECKING
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch 
import datetime
from peft import PeftMixedModel, PeftModel 
import threading

from .mp13_utils import round_floats

# Define specific exceptions
class MP13Error(Exception): ...
class ConfigurationError(MP13Error): pass
class DatasetError(MP13Error): pass
class TrainingError(MP13Error): pass
class EngineError(MP13Error): pass
class EngineInitializationError(EngineError): pass
class AdapterError(EngineError): pass
class InferenceRequestError(EngineError): pass
class BusyError(MP13Error): pass
class ModeMismatchError(EngineError): pass

class ServerStatus(Enum):
    """Status of the MP13 engine instance."""
    OFFLINE = "offline" # Before global initialization
    INITIALIZING = "initializing" # Global init in progress
    READY = "ready" # Global resources initialized and engine is idle.
    BUSY = "busy" # Engine is actively processing a long-running task (training/inference).
    ERROR = "error" # Global error or unrecoverable component error.
    SHUTTING_DOWN = "shutting_down"

class EngineModeState(Enum): # Mirrored from config for state tracking
    """Current operational mode of the engine's model."""
    TRAIN = "train"
    INFERENCE = "inference"

class TrainingStatus(Enum):
    """Status of the training operations."""
    OFFLINE = "offline" # Engine is not in TRAIN mode, or training component is not initialized for the mode.
    READY = "ready" # Engine in TRAIN mode, training component idle and ready for start_training.
    PREPARING = "preparing" # Head configured, ready for training start or dataset loading
    TRAINING = "training" # Actively training
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"
    # CANCELLING = "cancelling" # Optional intermediate state

class InferenceStatus(Enum):
    """Status of the inference operations."""
    OFFLINE = "offline" # Engine is not in INFERENCE mode
    READY = "ready" # Engine in INFERENCE mode, configured and ready for inference requests.
    INFERRING = "inferring" # Actively processing an inference request
    WARMING_CACHE = "warming_cache" # Actively warming a static cache slot in the background.
    ERROR = "error"
    CANCELLATION_TIMEOUT = "cancellation_timeout" # Cancellation timed out, awaiting thread exit
    # CANCELLING = "cancelling" # Optional intermediate state

@dataclass(eq=False)
class RequestResource:
    """A container for resources needed for a single inference request."""
    tokenizer: Any
    stream: Optional[torch.cuda.Stream]

@dataclass
class InferenceMetricsHistoryItem:
    """Stores metrics for a single completed inference request for historical analysis."""
    request_id: str
    start_time_mono: float
    end_time_mono: float
    end_time_wall: Optional[float] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    total_generation_duration_sec: Optional[float] = None
    avg_time_to_first_token_sec: Optional[float] = None
    was_truncated: bool = False
    mem_allocated: Optional[float] = None
    mem_reserved: Optional[float] = None
    total_tool_blocks: Optional[int] = None
    total_tool_blocks_tokens: Optional[int] = None


@dataclass
class AggregateInferenceMetrics:
    """Holds aggregate metrics for all inference requests."""
    active_requests: int = 0
    total_requests: int = 0
    total_successful_requests: int = 0
    total_failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_generation_duration_sec: float = 0.0
    total_tool_blocks: int = 0
    total_tool_blocks_tokens: int = 0
    mem_allocated: float = 0.0    
    mem_reserved: float = 0.0    

    def reset(self):
        """Resets all aggregate metrics to their initial values."""
        self.active_requests = 0
        self.total_requests = 0
        self.total_successful_requests = 0
        self.total_failed_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_generation_duration_sec = 0.0
        self.total_tool_blocks = 0
        self.total_tool_blocks_tokens = 0
        self.mem_allocated = 0.0
        self.mem_reserved  = 0.0
        
# Type alias for status history entry
StatusHistoryEntry = Tuple[float, str, int, Optional[float]]  # (timestamp, status_value, step, loss)

class MP13State:
    """
    Manages the state of the MP13 server instance.
    The engine manages a base model and can have PEFT adapters.
    The engine can be in TRAIN or INFERENCE mode.
    """
    def __init__(self, logger: logging.Logger, instance_id: str = "default"):
        self.instance_id = instance_id
        self.logger = logger
        self._lock = asyncio.Lock()  # async-side lock
        self._sync_lock = threading.RLock()  # sync-side lock        
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.run_start_time_wall: Optional[float] = None 
        
        # --- Global Server and Engine State ---
        self._server_status = ServerStatus.OFFLINE
        self._busy_reason = None # More specific reason if server is BUSY
        self._engine_mode: Optional[EngineModeState] = None # Set after global init
        self._shutting_down: bool = False # New flag to indicate engine is in shutdown process
        self._prevent_new_background_tasks: bool = False # gate for warm up or any other tasks during mode switch or shutdown
        
        self._base_model: Optional[Any] = None # The base HuggingFace model
        self._peft_model: Optional[PeftMixedModel] = None # PeftMixedModel for LoRA adapters
        self._tokenizer: Optional[Any] = None # PreTrainedTokenizerBase
        self.tokenizer_for_warmup: Optional[Any] = None
        self._device: Optional[str] = None # Now a string representation
        self._global_config: Optional[Dict[str, Any]] = None # Store GlobalEngineConfig dump
        self.base_model_quantization_cfg:  Optional[Dict[str, Any]] = None  # Used by adapters module. TODO: migrate onto name and details.
        self._effective_quantization_method: Optional[str] = None # Stores "none", "bnb", "awq", "hqq"
        self._effective_quantization_details: Optional[Dict[str, Any]] = None # Stores details of the active quantization
        self._new_adapters_added_in_session: List[str] = [] # New state
        self._adapters_active_before_training: Optional[List[str]] = None # New state
        # Direct state properties for config values used by engine components
        self._chat_templates: List[str] = []
        self._tool_parser_profile: Optional[Any] = None # Will hold ParserProfile object
        self._tool_templates: List[str] = []
        self.empty_system_prompt_template: Optional[str] = None
        self._hqq_backend: Optional[str] = None # New state for HQQ backend
        self._no_tools_parse: bool = False
        self.effective_eos_token_ids: Optional[List[int]] = None
        self.effective_pad_token_id: Optional[int] = None
        self.effective_stop_token_ids: Optional[List[int]] = None
        self.effective_stop_tokens: Optional[List[str]] = None
        # Training component state
        self.training_status: TrainingStatus = TrainingStatus.OFFLINE 
        self.current_training_config: Optional[Dict[str, Any]] = None # Stores TrainingConfig.model_dump(), set by start_training
        self._training_adapter_name: Optional[str] = None # Adapter being trained (derived from output_dir)
        self._training_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None # Task for the training watchdog
        self._graceful_stop_requested: bool = False # Flag for graceful stop
        self._training_cancelled: bool = False # Flag for forceful cancellation
        self._trainer_instance: Optional[Any] = None # To hold the HuggingFace Trainer instance
        self._training_error_message: Optional[str] = None
        self._training_start_time: Optional[float] = None
        self._training_end_time: Optional[float] = None
        self._training_last_update_time: Optional[float] = None
        self._current_step = 0
        self._total_steps = 0
        self._current_epoch = 0.0
        self._loss: Optional[float] = None
        self._learning_rate: Optional[float] = None
        self._grad_norm: Optional[float] = None 
        self._step_times: Deque[float] = deque(maxlen=50)
        self._status_history: Deque[StatusHistoryEntry] = deque(maxlen=30)
        self._adapter_files: List[str] = [] # Checkpoints, final adapter files from training
        self._last_checkpoint_path: Optional[str] = None
        self._final_adapter_path: Optional[str] = None
        self._adapter_name_trained_in_session: Optional[str] = None         
        self._adapter_reports: Dict[str, Optional[Dict[str, Any]]] = {"initial": None, "final": None, "delta": None}
        self._lora_grads_observed_in_training: bool = False
        self.heuristic_settings_summary: Optional[str] = None
        self.training_resource_report: Optional[Dict[str, Any]] = None
        self.effective_training_config: Optional[Dict[str, Any]] = None
        
        self.historical_loss: List[float] = []
        self.historical_lr: List[float] = []
        self.historical_steps: List[int] = []
        self.historical_grad_norm: List[float] = []
        self._last_recorded_log_step: int = -1

        self._dataset_loaded_for_training = False
        # Trainer initialization refers to the HuggingFace Trainer, not just our component init
        self._trainer_initialized_for_training = False 
        self._training_actually_started = False # Loop began
        self._steps_completed_in_training = False
        self._non_zero_loss_reported_in_training = False
        self._save_attempted_in_training = False
        # Inference component state
        self.inference_status: InferenceStatus = InferenceStatus.OFFLINE 
        self.current_inference_session_config: Optional[Dict[str, Any]] = None # Stores InferenceConfig.model_dump() for session defaults
        self.last_inference_error: Optional[str] = None
        self._gen_exec: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._bg_exec: Optional[concurrent.futures.ThreadPoolExecutor] = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mp13-bg-worker")

        # --- state for combined resource pool (tokenizer + stream) ---
        self._resource_pool: List["RequestResource"] = []
        self._resource_pool_lock = asyncio.Lock()
        self._resource_semaphore: Optional[asyncio.Semaphore] = None
        self._resource_in_use: Set["RequestResource"] = set() # Holds in-use resource objects

        # global sliding window metrics
        self._inference_metrics_history: deque = deque(maxlen=100)
        self.aggregate_metrics = AggregateInferenceMetrics()
        self._aggregate_metrics_lock = asyncio.Lock() # Lock for thread-safe updates to aggregate_metrics

        # Cohort mgmt tracks active reuests so this member is only used here and may be removed soon.
        self._active_inference_request_ids: set[str] = set()
        # Active inference cancel events are now managed by AdaptersControl.
        self.is_warming_cache: bool = False

        self._pending_warmup_queue: Deque[Tuple[Tuple[int, int], frozenset[str]]] = deque(maxlen=2) # Max queue size of 2
        self._active_signature: Optional[Tuple[Tuple[int, int], frozenset[str]]] = None # currently warmed up signature
        self._loaded_adapters_info: Dict[str, Dict[str, Any]] = {} # adapter_name -> {"root_path": str, "type": AdapterType, "config": AdapterConfig_dump}
        self._single_active_adapter_type_if_any: Optional[str] = None # Type if only one adapter is active
        
        # New state for cache invalidation logic
        self._adapter_load_order: List[str] = []
        self._last_unloaded_adapter_name: Optional[str] = None
        
        self._all_states_seen: Set[str] = set([TrainingStatus.OFFLINE.value, InferenceStatus.OFFLINE.value, ServerStatus.OFFLINE.value])
        self._state_transition_times: Dict[str, float] = {
            ServerStatus.OFFLINE.value: time.monotonic() # type: ignore
        }
        # print(f"[MP13State] Initialized new state instance: {instance_id}") # Verbose

        # --- Event Subscription Callbacks ---
        self._training_status_subscribers: List[Callable] = []
        self._inference_status_subscribers: List[Callable] = []
        self._engine_event_subscribers: List[Callable] = [] # For server status, mode changes

        # --- GPU Memory Tracking ---
        # Values stored in MB
        self.current_gpu_mem_allocated_mb: float = 0.0
        self.current_gpu_mem_reserved_mb: float = 0.0
        
        # --- Training Peak Tracking (native) ---
        self.peak_gpu_mem_allocated_current_op_mb: float = 0.0
        self.peak_gpu_mem_reserved_current_op_mb: float = 0.0
        self.peak_tracked_gpu_mem_allocated_training_mb: float = 0.0
        self.peak_tracked_gpu_mem_reserved_training_mb: float = 0.0

        # --- Minimal “first-run” gates (used by engine/infer patches) ---
        # 1) In concurrent user mode (dynamic cache): block all other requests until the very first
        #    user generate fully completes (prevents nested FX/Dynamo traces during first compile).
        self._first_user_generate_done: threading.Event = threading.Event()
        self._first_user_generate_lock: threading.Lock = threading.Lock()

        # --- Cohort Concurrency State (managed by AdaptersControl) ---
        self._cohort_lock: Optional[asyncio.Lock] = None
        self._cohort_sem: Optional[asyncio.Semaphore] = None
        self._cohort_sem_nf: Optional[asyncio.Semaphore] = None
        self._cohort_cg: int = 1
        self._cohort_current_key: Optional[Tuple[str, Tuple[str, ...]]] = None
        self._cohort_inflight: int = 0
        self._cohort_pending_count: int = 0
        self._cohort_active_inference_count: int = 0        
        self._cohort_rotor: Optional[Deque[Tuple[str, Tuple[str, ...]]]] = None
        self._cohort_queues: Optional[Dict[Any, Deque[Any]]] = None
        # Map of active (inflight) cohort requests.
        # Structure: { cohort_key: { request_id: entry_list } }
        # Populated by AdaptersControl when a pending user request is woken.
        self._cohort_active: Optional[Dict[Any, Dict[str, Any]]] = None
        self._cohort_queue_cond: Optional[asyncio.Condition] = None
        self._cohort_arrival_ctr: int = 0
        self._cohort_sig_cap: int = 1
        self._model_active_set: Tuple[str, ...] = tuple()
        self._cohort_head_seq: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        
    @property
    def base_model(self): return self._base_model
    
    @property
    def peft_model(self): return self._peft_model
    # Mode-aware property for the PEFT model used by AdaptersControl
    # However, AdaptersControl logic will be more explicit based on engine_mode.
    @property
    def global_config(self): return self._global_config # Expose global_config

    @property
    def engine_mode(self) -> Optional[EngineModeState]: return self._engine_mode
    @property
    def tokenizer(self): return self._tokenizer
    @property
    def tool_parser_profile(self): return self._tool_parser_profile
    @property
    def no_tools_parse(self) -> bool:
        # Reads from the top-level config field
        if self._global_config:
            return bool(self._global_config.get("no_tools_parse", False))
        return False

    @property
    def use_cache(self) -> bool:
        """Returns the effective 'use_cache' boolean flag from the global config."""
        if self._global_config:
            return bool(self._global_config.get("use_cache", True))
        return True # Default to True if config is not present

    @property
    def is_compiled(self) -> bool:
        """Returns the effective 'is_compiled' boolean flag from the global config."""
        if self._global_config:
            return bool(self._global_config.get("is_compiled", False))
        return False

    @property
    def is_static_cache_enabled(self) -> bool:
        """Returns the effective 'is_static_cache_enabled' boolean flag from the global config."""
        if self._global_config:
            return bool(self._global_config.get("is_static_cache_enabled", False))
        return False

    @property
    def concurrent_generate(self) -> int:
        """Returns the 'concurrent_generate' value from the global config."""
        if self._global_config:
            return int(self._global_config.get("concurrent_generate", 1))
        return 1

    @property
    def base_model_name_or_path(self) -> Optional[str]:
        """Returns the 'base_model_name_or_path' from the global config."""
        if self._global_config:
            return self._global_config.get("other_config", {}).get("base_model_name_or_path")
        return None

    @property
    def requested_torch_dtype(self) -> Optional[str]:
        """Returns the 'requested_torch_dtype' from the global config."""
        if self._global_config:
            return self._global_config.get("other_config", {}).get("requested_torch_dtype")
        return None

    @property
    def quantization_config(self) -> Optional[Dict[str, Any]]:
        """Returns the full quantization config dictionary from the global config."""
        # This value is stored at the top level of the effective config dump
        if self._global_config:
            return self._global_config.get("quantization_details")
        return None

    @property
    def device(self): return self._device
    @property
    def server_status(self) -> ServerStatus: return self._server_status
    @property
    def busy_reason(self) -> Optional[str]: return self._busy_reason    
    @property
    def current_step(self): return self._current_step
    @property
    def total_steps(self): return self._total_steps
    @property
    def current_epoch(self): return self._current_epoch
    @property
    def loss(self): return self._loss
    @property
    def learning_rate(self): return self._learning_rate
    @property
    def grad_norm(self): return self._grad_norm # Added property
    @property
    def training_error_message(self): return self._training_error_message

    # --- Active Adapter Properties (No-Lock Synchronous Versions) ---
    def _active_adapter_name_nolock(self) -> Optional[str]:
        if len(self._model_active_set) == 1:
            return self._model_active_set[0]
        return None

    def _get_loaded_adapters_info_nolock(self) -> Dict[str, Dict[str, Any]]:
        return self._loaded_adapters_info.copy()

    def _get_all_adapter_names_in_model_nolock(self) -> List[str]:
        return list(self._loaded_adapters_info.keys())

    # --- Active Adapter Properties (Public Async Versions with Lock) ---
    async def active_adapter_names(self) -> List[str]:
        """Returns a list of currently active adapter names."""
        async with self._lock:
            return list(self._model_active_set)

    async def active_adapter_name(self) -> Optional[str]:
        """Returns the adapter name if exactly one adapter is active, otherwise None."""
        async with self._lock:
            return self._active_adapter_name_nolock()
        
    async def active_adapter_type(self) -> Optional[str]:
        """
        Returns the type of the active adapter if exactly one adapter is active, 
        otherwise None. The type is derived from _single_active_adapter_type_if_any,
        which is set by the engine during set_active_adapter.
        """
        async with self._lock:
            return self._model_active_set
        
    async def get_loaded_adapters_info(self) -> Dict[str, Dict[str, Any]]:
        """Returns a copy of adapter info (name -> {path, type, config_dump})."""
        async with self._lock:
            return self._get_loaded_adapters_info_nolock()

    async def get_all_adapter_names_in_model(self) -> List[str]:
        """Returns all adapter names from the loaded adapters information."""
        async with self._lock:
            return self._get_all_adapter_names_in_model_nolock()

    # --- Other Getters (No-Lock Synchronous Versions) ---
    async def get_adapter_root_path(self, adapter_name: str) -> Optional[str]:
        """Retrieves the stored root path for a given adapter."""
        async with self._lock:
            info = self._loaded_adapters_info.get(adapter_name)
            if info:
                return info.get("root_path") # canonical adapter root
            return None

    async def get_adapter_type(self, adapter_name: str) -> Optional[str]:
        """Retrieves the stored type for a given adapter."""
        async with self._lock:
            info = self._loaded_adapters_info.get(adapter_name)
            if info:
                return info.get("type")
            return None

    # --- Model Getters (Public Async with Lock) ---
    def _add_status_history(self):
        """Adds the current training state to the history deque."""
        try:
            ts = time.monotonic()
            entry: StatusHistoryEntry = (
                round(ts, 2), self.training_status.value if self.training_status else "UNKNOWN_TRAIN_STATUS", self._current_step, self._loss
            )
            if not self._status_history or self._status_history[-1][1:] != entry[1:]:
                self._status_history.append(entry)
        except Exception as e:
            print(f"[Warning] Failed to add status history entry: {e}")

    async def _notify_subscribers_async(self, subscribers_list: List[Callable], event_data: Dict[str, Any]):
        """
        Asynchronously notifies a list of subscribers with the given event_data.

        Each callback is scheduled as a new asyncio task, allowing notifications
        to occur concurrently without blocking the caller or each other.

        Subscriber Callback Guidelines:
        - Callbacks MUST be `async` functions (coroutines).
        - Callbacks should be designed to be non-blocking and complete relatively quickly.
          Avoid long-synchronous operations directly within the callback. If necessary,
          use `await asyncio.to_thread()` for such operations.
        - Callbacks are responsible for their own exception handling (a try-except block
          is included here for logging purposes if a callback fails).
        - Callbacks receive a dictionary (`event_data`) containing the event information.
        - Callbacks should not attempt to directly acquire `MP13State._lock`. If state
          interaction is needed beyond the provided event_data, callbacks should use
          the public `async` methods of `MP13State`, which manage their own locking.
        """
        if self._shutting_down:
            return
        
        if not subscribers_list:
            return
        # Iterate over a copy in case the list is modified by a callback (though unlikely with create_task)
        for callback in list(subscribers_list):
            try:
                # Schedule the callback to run concurrently without awaiting its completion here
                asyncio.create_task(callback(event_data))
            except Exception as e:
                # Log error and potentially remove problematic subscriber after multiple failures
                print(f"[MP13State:{self.instance_id}] Error invoking subscriber {getattr(callback, '__name__', 'unknown')}: {e}")
                traceback.print_exc() # For more detailed debugging

    async def _notify_inference_status(self):
        """Helper to gather and notify subscribers about the current inference status."""
        if self._shutting_down:
            return
        self._update_gpu_memory_stats()
        status_dict = await self.get_inference_status_dict()
        await self._notify_subscribers_async(self._inference_status_subscribers, status_dict)


    def _update_gpu_memory_stats(self):
        if self._shutting_down:
            return

        try:
            cuda_available = False
            try:
                cuda_available = torch.cuda.is_available()
            except (RuntimeError, AttributeError) as e_is_available:
                cuda_available = False
                self.logger.warning(f"Error calling torch.cuda.is_available(): {e_is_available}. Assuming CUDA not available.")

            if cuda_available:
                device_indices_to_query = []
                if self._base_model is not None and hasattr(self._base_model, 'hf_device_map'):
                    all_devices = set(self._base_model.hf_device_map.values())
                    device_indices_to_query = sorted([d for d in all_devices if isinstance(d, int)])
                
                if not device_indices_to_query:
                    try:
                        num_devices = torch.cuda.device_count()
                        device_indices_to_query = list(range(num_devices))
                    except (RuntimeError, AttributeError) as e_device_count:
                        self.logger.warning(f"Failed to get CUDA device_count: {e_device_count}. Assuming 0 devices.")
                        device_indices_to_query = []

                total_current_alloc_bytes = 0
                sum_of_peak_alloc_bytes_current_op = 0
                total_current_reserved_bytes = 0
                sum_of_peak_reserved_bytes_current_op = 0

                for i in device_indices_to_query:
                    try:
                        allocated_on_device = torch.cuda.memory_allocated(i)
                        total_current_alloc_bytes += allocated_on_device

                        peak_alloc_on_device_i_bytes = torch.cuda.max_memory_allocated(i)
                        sum_of_peak_alloc_bytes_current_op += peak_alloc_on_device_i_bytes

                        reserved_on_device = torch.cuda.memory_reserved(i)
                        total_current_reserved_bytes += reserved_on_device

                        peak_reserved_on_device_i_bytes = torch.cuda.max_memory_reserved(i)
                        sum_of_peak_reserved_bytes_current_op += peak_reserved_on_device_i_bytes
                    except (RuntimeError, AttributeError) as e_alloc:
                        self.logger.warning(f"Failed to get memory stats for device {i}: {e_alloc}")
                
                try:
                    bytes_to_mb = 1024 * 1024
                    self.current_gpu_mem_allocated_mb = total_current_alloc_bytes / bytes_to_mb
                    self.current_gpu_mem_reserved_mb = total_current_reserved_bytes / bytes_to_mb
                    self.peak_gpu_mem_allocated_current_op_mb = sum_of_peak_alloc_bytes_current_op / bytes_to_mb
                    self.peak_gpu_mem_reserved_current_op_mb = sum_of_peak_reserved_bytes_current_op / bytes_to_mb

                    if self.training_status == TrainingStatus.TRAINING or self.training_status == TrainingStatus.PREPARING:
                        self.peak_tracked_gpu_mem_allocated_training_mb = max(self.peak_tracked_gpu_mem_allocated_training_mb, self.peak_gpu_mem_allocated_current_op_mb)
                        self.peak_tracked_gpu_mem_reserved_training_mb = max(self.peak_tracked_gpu_mem_reserved_training_mb, self.peak_gpu_mem_reserved_current_op_mb)
                except ZeroDivisionError as e_calc:
                    self.logger.warning(f"Failed during GPU memory calculation (ZeroDivisionError): {e_calc}")
            else:
                self.current_gpu_mem_allocated_mb = 0.0
                self.current_gpu_mem_reserved_mb = 0.0
                self.peak_gpu_mem_allocated_current_op_mb = 0.0
                self.peak_gpu_mem_reserved_current_op_mb = 0.0

        except Exception as e_outer:
            print(f"[MP13State:{self.instance_id}] CRITICAL UNHANDLED EXCEPTION in _update_gpu_memory_stats: {e_outer}")
            traceback.print_exc()
            self.current_gpu_mem_allocated_mb = 0.0
            self.current_gpu_mem_reserved_mb = 0.0
            self.peak_gpu_mem_allocated_current_op_mb = 0.0
            self.peak_gpu_mem_reserved_current_op_mb = 0.0

    def _update_server_status(self):
        """Updates overall server status based on component statuses."""
        # --- Priority 1: Top-level server states that override component states ---
        if self._server_status in [ServerStatus.INITIALIZING, ServerStatus.SHUTTING_DOWN, ServerStatus.OFFLINE]:
            return # Don't change these based on component status

        # --- Priority 2: Check for component error states ---
        is_error = (
            self.training_status == TrainingStatus.ERROR or
            self.inference_status in [InferenceStatus.ERROR, InferenceStatus.CANCELLATION_TIMEOUT]
        )
        if is_error:
            self._server_status = ServerStatus.ERROR
            # _busy_reason should be cleared or set to error reason by the operation that sets the error state.
            self._busy_reason = None
            return

        # --- Priority 3: Check for component busy states ---
        is_busy = (
            self.training_status in [TrainingStatus.PREPARING, TrainingStatus.TRAINING] or
            self.inference_status in [InferenceStatus.INFERRING]
        )
        if is_busy:
            self._server_status = ServerStatus.BUSY
            # _busy_reason should be set by the specific operation that makes the server busy.
            return

        # --- Priority 4: If not busy or in error, it's ready ---
        self._server_status = ServerStatus.READY
        self._busy_reason = None

    async def set_global_resources(
        self, 
        base_model: Any, 
        peft_model: Any,
        tokenizer_for_warmup: Any,
        tokenizer: Any, 
        device: Any, 
        global_config_dump: Dict[str, Any], 
        initial_mode: EngineModeState
    ):
        """Sets all global resources, using the config dump as the source of truth."""
        async with self._lock:
            self._base_model = base_model
            self._tokenizer = tokenizer
            self.tokenizer_for_warmup = tokenizer_for_warmup
            self._device = device
            # The PeftMixedModel for inference is now created at init and is persistent.
            self._peft_model = peft_model
            self._global_config = global_config_dump
            self._engine_mode = initial_mode
            
            # Populate convenience attributes from the single source of truth
            self._effective_quantization_details = global_config_dump.get("quantization_details")
            self._chat_templates = global_config_dump.get("chat_templates", [])
            self._tool_templates = global_config_dump.get("tool_templates", [])
            self._tool_parser_profile = global_config_dump.get("other_config", {}).get("tool_parser_profile")
            self.empty_system_prompt_template = global_config_dump.get("other_config", {}).get("empty_system_prompt_template")
            self._server_status = ServerStatus.READY
            self.logger.info(f"State global resources set. Engine mode: {initial_mode.value}.")

    async def clear_global_resources(self):
        # Part 1: Synchronous state clearing under lock
        async with self._lock:
            # Explicitly delete model references to help GC
            if self._base_model is not None: del self._base_model

            self._base_model = None
            self._peft_model = None
            self._engine_mode = None # Set mode to None *before* resetting components
            if self.tokenizer_for_warmup: del self.tokenizer_for_warmup
            self._tokenizer = None
            self._device = None
            self._gen_exec = None
            self._global_config = None
            self.base_model_quantization_cfg = None
            self._effective_quantization_method = None
            self._chat_templates.clear()
            self._tool_templates.clear()
            self._tool_parser_profile = None # This is now sourced from other_config
            self.empty_system_prompt_template = None            
            # self._engine_mode = None # engine_mode is reset by reset_training/reset_inference if they set status to OFFLINE

            self._model_active_set = tuple()
            self._new_adapters_added_in_session.clear()
            self._single_active_adapter_type_if_any = None
            self._loaded_adapters_info.clear()
            self._pending_warmup_queue.clear()
            # New state clearing
            self._adapter_load_order.clear()
            self._last_unloaded_adapter_name = None
            self.logger.info(f"Core global resource attributes cleared under lock.")

        # Part 2: Call async methods that manage their own locks
        # These methods will also update their respective component statuses (e.g., to OFFLINE)
        await self.reset_training(full_reset=True, shutting_down=True)
        await self.reset_inference(full_reset=True, shutting_down=True)

        # Part 3: Final server status update under lock
        async with self._lock:
            self._server_status = ServerStatus.OFFLINE
            self.logger.info(f"Global resources fully cleared. Server OFFLINE.")

    async def set_engine_mode_state(self, mode: EngineModeState):
        async with self._lock:
            self._engine_mode = mode
            self.logger.info(f"Engine mode set to: {mode.value}")
        self._update_gpu_memory_stats()
        status_dict = await self.get_server_status_dict()
        event_data = {"event_type": "engine_mode_changed", "mode": mode.value, "data": status_dict}
        await self._notify_subscribers_async(self._engine_event_subscribers, event_data)

    async def set_training_status(self, new_status: TrainingStatus, message: Optional[str] = None, force_update: bool = False, expected_current_status: Optional[TrainingStatus] = None):
        # Determine if a notification should be forced even if the status string doesn't change.
        # This is useful for system-level state changes where underlying data might have changed.
        is_system_message = message and (
            "Engine initialized in" in message or
            "Correcting status" in message or
            "Engine mode set to" in message or
            "Training state reset" in message
        )
        force_update = force_update or is_system_message
        async with self._lock:
            if expected_current_status is not None and self.training_status != expected_current_status:
                log_msg = f"Training status transition to {new_status.value} aborted. Expected {expected_current_status.value}, but found {self.training_status.value}."
                self.logger.info(log_msg)
                return # Abort
            if self.training_status != new_status or force_update:
                old_status = self.training_status
                self.training_status = new_status
                if old_status != new_status: # Only log state transitions and history if actually changed
                    self._all_states_seen.add(new_status.value)
                    if new_status.value not in self._state_transition_times:
                        self._state_transition_times[new_status.value] = time.monotonic()
                    self._add_status_history() # Uses self.training_status internally
                self._update_server_status()
                log_msg = f"Training status: {old_status.value} -> {new_status.value}"
                if message: log_msg += f" - {message}"
                self.logger.info(log_msg)
        # Notify subscribers regardless of whether status string changed, if forced or if data might have changed
        self._update_gpu_memory_stats()
        status_dict = await self.get_training_status_dict()
        await self._notify_subscribers_async(self._training_status_subscribers, status_dict)

    async def set_inference_status(self, new_status: InferenceStatus, message: Optional[str] = None, force_update: bool = False, expected_current_status: Optional[InferenceStatus] = None):
        # Determine if a notification should be forced even if the status string doesn't change.
        is_system_message = message and (
            "Engine initialized in" in message or
            "Correcting status" in message or
            "Engine mode set to" in message or
            "Inference state reset" in message
        )
        force_update = force_update or is_system_message
        async with self._lock:
            if expected_current_status is not None and self.inference_status != expected_current_status:
                # log_msg = f"Inference status transition to {new_status.value} aborted. Expected {expected_current_status.value}, but found {self.inference_status.value}."
                # print(f"[MP13State:{self.instance_id}] {log_msg}") # This is too noisy for background tasks. The check is sufficient.
                return # Abort

            if self.inference_status != new_status or force_update:
                old_status = self.inference_status
                self.inference_status = new_status
                
                # Only log actual state transitions and update history if the status truly changed
                # or if it's a forced update ensuring a specific state like READY.
                if old_status != new_status:
                    self._all_states_seen.add(new_status.value)
                    if new_status.value not in self._state_transition_times:
                        self._state_transition_times[new_status.value] = time.monotonic()
                
                self._update_server_status()
                log_msg = f"Inference status: {old_status.value} -> {new_status.value}"
                if message: log_msg += f" - {message}"
                self.logger.info(f"[MP13State:{self.instance_id}] {log_msg}")
            if new_status == InferenceStatus.READY: # Ensure busy reason is cleared when ready
                self._busy_reason = None
        self._update_gpu_memory_stats()
        status_dict = await self.get_inference_status_dict()
        await self._notify_subscribers_async(self._inference_status_subscribers, status_dict)

    async def set_server_status(self, new_status: ServerStatus, message: Optional[str] = None):
        # print(f"[MP13State:{self.instance_id}] Entering set_server_status. Target: {new_status.value}, Message: {message}") # Verbose
        async with self._lock:
            if self._server_status != new_status:
                old_status = self._server_status
                self._server_status = new_status
                if new_status == ServerStatus.SHUTTING_DOWN:
                    self._shutting_down = True
                elif new_status == ServerStatus.OFFLINE: # After shutdown, reset flag
                    self._shutting_down = False
                self._all_states_seen.add(new_status.value)
                if new_status.value not in self._state_transition_times:
                     self._state_transition_times[new_status.value] = time.monotonic()
                log_msg = f"Server status: {old_status.value} -> {new_status.value}"
                if message: log_msg += f" - {message}"
                self.logger.info(log_msg)

        if not self._shutting_down: # Only update/notify if not shutting down
            self._update_gpu_memory_stats()
            status_dict = await self.get_training_status_dict()
            await self._notify_subscribers_async(self._training_status_subscribers, status_dict)
            status_dict = await self.get_server_status_dict()
            # print(f"[MP13State:{self.instance_id}] set_server_status: After get_server_status_dict, Before _notify_subscribers_async") # Verbose
            event_data = {"event_type": "server_status_changed", "status": status_dict}
            await self._notify_subscribers_async(self._engine_event_subscribers, event_data)
            # print(f"[MP13State:{self.instance_id}] set_server_status: After _notify_subscribers_async. Exiting.") # Verbose


    async def reset_training(self, full_reset: bool = False, shutting_down: bool = False):
        async with self._lock:
            self.logger.info(f"Resetting training state (full_reset={full_reset}, shutting_down={shutting_down})...")
            # Status will be set by mode switch or explicitly after reset
            self.current_training_config = None
            self._training_adapter_name = None
            self._current_step = 0
            self._total_steps = 0
            self._current_epoch = 0.0
            self._loss = None
            self._learning_rate = None
            self._grad_norm = None
            self._training_error_message = None
            
            if self._training_task and not self._training_task.done():
                self._training_task.cancel("Training state reset")
            self._training_task = None
            self._trainer_instance = None # Clear trainer instance

            self._graceful_stop_requested = False # Reset flag
            self._training_cancelled = False # Reset flag
            self._training_start_time = None
            self._training_end_time = None
            self._training_last_update_time = None
            self._step_times.clear()
            if full_reset: self._adapter_files.clear() # Only clear fully if global reset
            self._last_checkpoint_path = None
            if full_reset: self._final_adapter_path = None

            self._dataset_loaded_for_training = False
            self._trainer_initialized_for_training = False
            self._training_actually_started = False
            self._steps_completed_in_training = False
            self._non_zero_loss_reported_in_training = False
            self._save_attempted_in_training = False
            
            self.historical_loss.clear()
            self.historical_lr.clear()
            self.historical_steps.clear()
            self.historical_grad_norm.clear()
            self._last_recorded_log_step = -1            
            self.run_start_time_wall = None

            self._status_history.clear()
            self._adapter_reports = {"initial": None, "final": None, "delta": None}
            self._lora_grads_observed_in_training = False
            self.heuristic_settings_summary = None
            self.training_resource_report = None
            self.effective_training_config = None

        # Determine target status
        if shutting_down:
            target_status = TrainingStatus.OFFLINE
            status_message = "Training state reset due to engine shutdown."
        else:
            target_status = TrainingStatus.READY if self._engine_mode == EngineModeState.TRAIN else TrainingStatus.OFFLINE
            status_message = "Training state reset, ready for new config." if target_status == TrainingStatus.READY else "Training state reset, engine not in TRAIN mode."

        await self.set_training_status(target_status, status_message, force_update=True)
        async with self._lock: # Re-acquire lock if needed for _update_server_status
            self._update_server_status()
        self.logger.info(f"Training state reset complete.")

    async def set_training_config(self,
                            training_config_dump: Dict[str, Any], 
                            adapter_name: str, 
                            training_run_output_dir: str, # Added
                            current_wall_time: float):
        """Sets the configuration for the current training run. Assumes reset_training was called before if needed."""
        # This method is now async because it's awaited by the engine.
        async with self._lock:
            if self._engine_mode != EngineModeState.TRAIN:
                raise ModeMismatchError(f"Cannot prepare for training. Engine mode is {self._engine_mode.value if self._engine_mode else 'UNSET'}, expected TRAIN.")
            if self._server_status == ServerStatus.BUSY and self._busy_reason and "inferring" in self._busy_reason.lower():
                 raise BusyError(f"Cannot start training. Server is busy with inference ({self._busy_reason}).")
            
            # self.reset_training() # Engine's start_training will call reset_training before this.
            self.run_start_time_wall = current_wall_time
            self.current_training_config = training_config_dump
            # Ensure the determined output_dir for this run is stored in the config dump if not already there or different
            self.current_training_config["output_dir"] = training_run_output_dir 
            self._training_adapter_name = adapter_name
            # The engine will call set_training_status(PREPARING) after this.
            # If this method needed to set status itself, it would be:
            # await self.set_training_status(TrainingStatus.PREPARING, f"Preparing for training adapter '{adapter_name}'")
            self._busy_reason = f"Training ({TrainingStatus.PREPARING.value}) for adapter {adapter_name}"
            self._update_server_status()
            # training_start_time is set when training *actually* starts (set_training_started)
            self._training_start_time = time.monotonic() # Time when prep starts
            self._training_last_update_time = self._training_start_time

    async def set_dataset_loaded_for_training(self):
        async with self._lock:
            self._dataset_loaded_for_training = True
            #print(f"[MP13State:{self.instance_id}] Dataset loaded for training")
        # If this implies a sub-status change that needs notification, handle it here.
        # For now, assuming it's part of PREPARING.
        # If it needs to trigger a specific notification:
        # self._update_gpu_memory_stats()
        # await self._notify_subscribers_async(self._training_status_subscribers, self.get_training_status_dict())

    async def set_training_started(self): # Renamed from _training_actually_started to be consistent
        if await asyncio.to_thread(torch.cuda.is_available):
            try:
                num_devices = await asyncio.to_thread(torch.cuda.device_count)
                for i in range(num_devices):
                    try:
                        await asyncio.to_thread(torch.cuda.reset_peak_memory_stats, i)
                    except Exception as e_reset_peak_device:
                        self.logger.warning(f"Warning: Failed to reset peak memory stats for device {i}: {e_reset_peak_device}")
                # Using a more concise log, or remove if too verbose for normal operation
                # print(f"[MP13State:{self.instance_id}] Reset peak GPU memory stats for all {num_devices} devices (training).")
            except Exception as e_reset_peak:
                self.logger.warning(f"Warning: Failed to reset peak memory stats for all devices (training): {e_reset_peak}")


        async with self._lock: # Use async lock if operations inside are async
            self._training_start_time = time.monotonic() # Actual start of the training loop
            self._training_actually_started = True
            self._busy_reason = f"Training ({TrainingStatus.TRAINING.value}) adapter {self._training_adapter_name}"
            self._update_server_status()
            self._training_last_update_time = time.monotonic()
        await self.set_training_status(TrainingStatus.TRAINING, "Training loop started") # This will notify


    async def update_training_progress(self, step: int, epoch: float, loss: Optional[float] = None, learning_rate: Optional[float] = None, grad_norm: Optional[float] = None):
        # Lock should be acquired before modifying state and released after notification logic if possible,
        # or use an async lock if _update_gpu_memory_stats or _notify_subscribers_async need it.
        async with self._lock: # Assuming state modification needs to be atomic
            current_time = time.monotonic()
            self._current_step = step
            self._current_epoch = epoch
            if grad_norm is not None: self._grad_norm = grad_norm 
            if loss is not None: self._loss = loss; self._non_zero_loss_reported_in_training = loss > 1e-9
            if learning_rate is not None: self._learning_rate = learning_rate

            if loss is not None and step > self._last_recorded_log_step:
                self.historical_loss.append(loss)
                actual_lr = learning_rate if learning_rate is not None else (self._learning_rate or 0.0)
                self.historical_lr.append(actual_lr)
                actual_gn = grad_norm if grad_norm is not None else np.nan # Use np.nan for missing
                self.historical_grad_norm.append(actual_gn)
                self.historical_steps.append(step)
                self._last_recorded_log_step = step

            if step > 0: self._steps_completed_in_training = True
            if self._training_last_update_time:
                self._step_times.append(current_time - self._training_last_update_time)
            self._training_last_update_time = current_time
            self._add_status_history()
        
        self._update_gpu_memory_stats()
        # get_training_status_dict() will be called inside _notify_subscribers_async via set_training_status
        # or directly if we don't change the status string but want to send data.
        # For progress, we always send new data.
        status_dict = await self.get_training_status_dict()
        await self._notify_subscribers_async(self._training_status_subscribers, status_dict)

    async def set_save_attempted(self, checkpoint_path=None):
        async with self._lock:
            self._save_attempted_in_training = True
            if checkpoint_path: self._last_checkpoint_path = checkpoint_path
            self.logger.info(f"Checkpoint saved to: {checkpoint_path}")
        # This is a progress point, might notify if needed.
        # self._update_gpu_memory_stats()
        # status_dict = await self.get_training_status_dict()
        # await self._notify_subscribers_async(self._training_status_subscribers, status_dict)

    async def set_final_adapter_saved(self, adapter_path: str):
        async with self._lock:
            self._final_adapter_path = adapter_path
            self.logger.info(f"Final adapter saved to: {adapter_path}")
            self._add_files_from_dir_to_list(adapter_path, self._adapter_files, set(self._adapter_files))
        # This is a progress point, might notify if status changes or significant data update.
        # self._update_gpu_memory_stats()
        # status_dict = await self.get_training_status_dict()
        # await self._notify_subscribers_async(self._training_status_subscribers, status_dict)

    async def set_training_completed(self):
        async with self._lock:
            self._training_end_time = time.monotonic()
            self._busy_reason = None
            self._update_server_status()
        await self.set_training_status(TrainingStatus.COMPLETED, "Training completed successfully")

    async def set_training_error(self, error_message):
        async with self._lock:
            self._training_error_message = error_message
            self._training_end_time = self._training_end_time or time.monotonic()
            self._busy_reason = None # Or set to error reason
            self._update_server_status() # This will set server to ERROR
        await self.set_training_status(TrainingStatus.ERROR, f"Error: {error_message}")

    async def set_training_stopped(self, reason="User requested stop"):
        async with self._lock:
            self._training_error_message = f"Training stopped: {reason}" # Store reason in error field
            self._training_end_time = self._training_end_time or time.monotonic()
            self._busy_reason = None
            self._update_server_status()
        await self.set_training_status(TrainingStatus.STOPPED, reason)

    async def set_training_status_externally_cancelled(self, reason: str):
        """Used when cancellation is initiated from outside the training loop (e.g., API call)."""
        async with self._lock:
            if self.training_status == TrainingStatus.TRAINING or self.training_status == TrainingStatus.PREPARING:
                self._training_error_message = f"Cancelled: {reason}" if reason else "Training cancelled externally."
                self._training_end_time = time.monotonic()
                self.logger.info(f"Training status set to {self.training_status.value} due to external cancellation. Reason: {reason}")

    async def add_adapter_files(self, files: List[str]): # From checkpoint saving
        async with self._lock:
            # Ensure no duplicates and preserve order, more robustly
            current_files_set = set(self._adapter_files)
            for f in files:
                if f not in current_files_set:
                    self._adapter_files.append(f) # type: ignore
                    current_files_set.add(f) # type: ignore

    async def set_adapter_report(self, stage: str, report: Dict[str, Any]):
        """Stores a structured adapter report (e.g., initial/final) for status subscribers."""
        stage_key = (stage or "").strip().lower() or "unknown"
        normalized = dict(report)
        lines_value = normalized.get("lines")
        if isinstance(lines_value, list):
            normalized["lines"] = [str(line) for line in lines_value]

        def _sanitize_for_storage(value):
            if value is None:
                return None
            if isinstance(value, dict):
                return {str(k): _sanitize_for_storage(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_sanitize_for_storage(v) for v in value]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return float(value)
            return str(value)

        if "metrics" in normalized:
            normalized["metrics"] = _sanitize_for_storage(normalized["metrics"])

        async with self._lock:
            # Preserve existing keys but allow new ones if needed for diagnostics
            if stage_key not in self._adapter_reports:
                self._adapter_reports[stage_key] = None
            self._adapter_reports[stage_key] = normalized

    async def set_lora_grads_observed(self, observed: bool):
        """Tracks whether LoRA gradients were observed during the latest training run."""
        async with self._lock:
            self._lora_grads_observed_in_training = bool(observed)

    async def set_heuristic_summary(self, summary: str):
        """Stores the heuristic settings summary and notifies subscribers."""
        async with self._lock:
            self.heuristic_settings_summary = summary
        # Trigger a notification so the client gets the summary immediately
        status_dict = await self.get_training_status_dict()
        await self._notify_subscribers_async(self._training_status_subscribers, status_dict)

    async def set_training_resource_report(self, report: Dict[str, Any]):
        """Store detailed hardware/heuristic data for the current run and notify subscribers."""
        async with self._lock:
            self.training_resource_report = round_floats(report)
        status_dict = await self.get_training_status_dict()
        await self._notify_subscribers_async(self._training_status_subscribers, status_dict)

    async def set_effective_training_config(self, report: Dict[str, Any]):
        """Store the resolved training settings pushed to Trainer and notify subscribers."""
        async with self._lock:
            self.effective_training_config = round_floats(report)
        status_dict = await self.get_training_status_dict()
        await self._notify_subscribers_async(self._training_status_subscribers, status_dict)
        
    async def reset_inference(self, full_reset: bool = False, shutting_down: bool = False):
        async with self._lock:
            self.logger.info(f"Resetting inference state (full_reset={full_reset}, shutting_down={shutting_down})...")
            self.current_inference_session_config = None
            if full_reset:
                self._loaded_adapters_info.clear() # User-provided info cleared on full reset
            # Cancellation is handled via per-request threading.Events attached by cohort_enter
            # Clear first-run gates so a new session can prime again.
            try:
                self._first_user_generate_done.clear()
            except Exception:
                pass        

        if shutting_down:
            target_status = InferenceStatus.OFFLINE
            status_message = "Inference state reset due to engine shutdown."
        else:
            target_status = InferenceStatus.READY if self._engine_mode == EngineModeState.INFERENCE else InferenceStatus.OFFLINE
            status_message = "Inference state reset, ready for requests." if target_status == InferenceStatus.READY else "Inference state reset, engine not in INFERENCE mode."
        await self.set_inference_status(target_status, status_message, force_update=True)
        async with self._lock: # Re-acquire lock if needed for _update_server_status
            self._update_server_status()
        #print(f"[MP13State:{self.instance_id}] Inference state reset complete.")

    async def set_active_adapter_state(self, adapter_names: Union[None, str, List[str]], single_adapter_type_if_applicable: Optional[str]):
        """
        Sets the active adapter names and the type if only a single adapter is active.
        Called by the engine after it has configured the underlying models.
        If adapter_names is None, it's treated as an empty list (no active adapters).
        If adapter_names is a string, it's treated as a list with one element.
        """
        async with self._lock:
            processed_adapter_names_list: List[str]
            if isinstance(adapter_names, str):
                processed_adapter_names_list = [adapter_names]
            elif isinstance(adapter_names, (list, tuple)):
                processed_adapter_names_list = adapter_names
            else: # Handles None and any other unexpected types
                processed_adapter_names_list = []
            
            # The new source of truth is the normalized tuple `_model_active_set`
            # Normalize adapter names into a canonical, sorted tuple.
            # '__base__' and None are treated as an empty set.
            active_adapters = tuple()
            if processed_adapter_names_list:
                active_adapters = tuple(sorted(
                    {name for name in processed_adapter_names_list if name and name != "__base__"}))
            
            self._model_active_set = active_adapters
            
            if len(self._model_active_set) == 1:
                self._single_active_adapter_type_if_any = single_adapter_type_if_applicable
            else:
                self._single_active_adapter_type_if_any = None # Explicitly None if multiple or zero
            self.logger.info(f"Active adapters set in state: Names='{self._model_active_set}', SingleActiveType='{self._single_active_adapter_type_if_any}'")
    
    async def add_loaded_adapter_info(
        self,
        adapter_name: str,
        adapter_root_path: Optional[str],
        adapter_type: str,
        checkpoint_path: Optional[str],
        adapter_config_dump: Optional[Dict[str,Any]],
        *,
        base_model_quant: Optional[str] = None,
        base_model_name: Optional[str] = None,
        alias: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_foreign: bool = False
    ):
        """
        Store canonical adapter information keyed by adapter name.
        `path` always refers to the adapter root directory; `checkpoint_path` points to the
        specific checkpoint folder (may equal root); metadata is persisted for later display.
        """
        async with self._lock:
            self._loaded_adapters_info[adapter_name] = {
                "root_path": adapter_root_path,
                "type": adapter_type,
                "checkpoint_path": checkpoint_path,
                "alias": alias,
                "base_model_quant": base_model_quant,
                "base_model_name": base_model_name,
                "metadata": metadata or {},
                "config": adapter_config_dump,
                "is_foreign": is_foreign,
            }
            self.logger.info(f"Adapter info added/updated: '{adapter_name}' (Type: {adapter_type}, Root Path: {adapter_root_path or 'N/A'}, Checkpoint: {checkpoint_path or 'N/A'})")

    async def remove_loaded_adapter_info(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if adapter_name in self._loaded_adapters_info:
                removed_info = self._loaded_adapters_info.pop(adapter_name)
                self.logger.info(f"Adapter info removed: '{adapter_name}'")
                return removed_info
            return None

    async def set_inferring(self, request_id: str):
        """
        Sets the engine to the INFERRING state and adds the request to the active set.
        Concurrency is managed by the resource pool checkout, not here.
        """

        # Update the state under the main lock
        async with self._lock:
            self._active_inference_request_ids.add(request_id)

        # Now that the counter is updated (and the lock is released), set the status.
        # This will notify subscribers.
        await self.set_inference_status(
            InferenceStatus.INFERRING, f"Starting inference for request {request_id}", force_update=True)
    
    async def set_inference_complete(self, request_id: str, cancelled: bool = False):
        """
        Removes a request from the active set and updates the engine status.
        If no requests are left, status becomes READY or WARMING_CACHE. This method is idempotent.
        Concurrency slots are released when resources are checked in.
        """
    
        new_status = self.inference_status # Default to current
    
        async with self._lock:
   
            # This check makes the method idempotent. It only acts if the request is still considered active.
            if request_id in self._active_inference_request_ids:
                self._active_inference_request_ids.remove(request_id)
            # Active cancel events are managed by AdaptersControl; just update active IDs.

            if not self._active_inference_request_ids:
                # If no more requests, status depends on whether we are warming cache
                if self.is_warming_cache:
                    new_status = InferenceStatus.WARMING_CACHE
                else:
                    new_status = InferenceStatus.READY
                
                self.last_inference_error = None
            else:
                # If there are still active requests, status remains INFERRING
                new_status = InferenceStatus.INFERRING
    
        status_msg = f"Inference cancelled for request {request_id}" if cancelled else f"Inference complete for request {request_id}"
        await self.set_inference_status(new_status, status_msg, force_update=True)

    async def set_warming_status(self, is_warming: bool):
        """
        Sets the cache warming flag and updates the inference status if appropriate.
        Does not interrupt an ongoing inference operation.
        """
        if is_warming:
            self.logger.info(f"[bg-warmup] Starting background cache warmup.")
        else:
            self.logger.info(f"[bg-warmup] Background cache warmup finished.")

        new_status = self.inference_status # default
        async with self._lock:
            self.is_warming_cache = is_warming
            if is_warming:
                # Only switch to WARMING_CACHE if we are currently READY.
                if self.inference_status == InferenceStatus.READY:
                    new_status = InferenceStatus.WARMING_CACHE
            else: # Warming is finishing
                # Only switch back to READY if we were in WARMING_CACHE and there are no active inference requests.
                if self.inference_status == InferenceStatus.WARMING_CACHE and not self._active_inference_request_ids:
                    new_status = InferenceStatus.READY
        
        # Use the main setter to notify and update server status
        await self.set_inference_status(new_status, f"Cache warming status changed to {is_warming}", force_update=True)

    def _run_coro_threadsafe(self, coro, timeout=15):
        """Helper to run a coroutine from a sync thread on the main event loop."""
        if not self.loop or not self.loop.is_running():
            self.logger.error(f"Error: Event loop not available or not running for thread-safe call.")
            return

        # Fire-and-forget: Schedule the coroutine and do not wait for its result.
        # This prevents the background thread from blocking on the event loop,
        # which is the cause of the deadlock when the event loop is waiting on
        # the background thread.
        asyncio.run_coroutine_threadsafe(coro, self.loop)
        # By not calling future.result(), we avoid the timeout and the deadlock.
        # The coroutine will run on the loop when it gets a chance.
        return None

    def set_warming_status_from_thread(self, is_warming: bool):
        """Synchronous wrapper for set_warming_status, intended for use by background worker threads."""
        self._run_coro_threadsafe(self.set_warming_status(is_warming))

    def set_slot_warmup_complete_from_thread(self, signature: Tuple[Tuple[int, int], frozenset[str]]):
        """Synchronous wrapper for set_slot_warmup_complete, intended for use by background worker threads.""" 
        self._run_coro_threadsafe(self.set_slot_warmup_complete(signature))
        
    async def set_slot_warmup_complete(self, signature: Tuple[Tuple[int, int], frozenset[str]]):
        """
        Logs the completion of a specific background cache slot warmup.
        If the completed signature was the active one, it is cleared.
        This should be called by the background worker after a slot is successfully warmed.
        """
        async with self._lock:
            # If the completed signature was the active one, clear it.
            if self._active_signature == signature:
                self._active_signature = None
 
        # Logging is done outside the lock to minimize lock hold time.
        shape, adapters = signature
        batch, length = shape
        adapters_str = ", ".join(sorted(list(adapters))) if adapters else "base"
        self.logger.info(f"[bg-warmup] Slot warmup complete for: B={batch}, L={length}, Adapters=[{adapters_str}]")

    async def set_inference_error(self, error_message: str, request_id: Optional[str] = None, clear_config: bool = True):
        """
        Handles an error during an inference request.
        Concurrency slots are released when resources are checked in.
        """

        new_status = self.inference_status # Default to current

        async with self._lock:
            is_concurrent = self.global_config and self.global_config.get("concurrent_generate", 1) > 1

            # Remove the request from the active set.
            if request_id and request_id in self._active_inference_request_ids:
                self._active_inference_request_ids.remove(request_id)
            # Active cancel events are managed by AdaptersControl; nothing to delete here.

            # In concurrent mode, an error in one request doesn't stop the whole engine.
            if is_concurrent:
                # If this was the last active request, the engine becomes ready.
                if not self._active_inference_request_ids:
                    # If no more requests, status depends on whether we are warming cache
                    if self.is_warming_cache:
                        new_status = InferenceStatus.WARMING_CACHE
                    else:
                        new_status = InferenceStatus.READY
                    self.last_inference_error = None # Clear last global error
                else:
                    new_status = InferenceStatus.INFERRING
            else:
                # In single-request mode, any error puts the whole component into an ERROR state.
                new_status = InferenceStatus.ERROR

            self.last_inference_error = error_message
            if clear_config and not is_concurrent: # Only clear session config on non-concurrent error
                self.current_inference_session_config = {}
        
        # Notify subscribers about the status change.
        await self.set_inference_status(new_status, f"Inference error for request {request_id}: {error_message}", force_update=True)

    def _add_files_from_dir_to_list(self, directory_path: str, target_list: List[str], existing_files_set: Set[str]):
        if os.path.isdir(directory_path):
            added_count = 0
            for root, _, files_in_dir in os.walk(directory_path):
                for file_name in files_in_dir:
                    file_path = os.path.join(root, file_name)
                    if file_path not in existing_files_set:
                        target_list.append(file_path)
                        existing_files_set.add(file_path)
                        added_count +=1
            if added_count > 0:
                self.logger.info(f"Added {added_count} files from {directory_path}.")
        else: 
            self.logger.warning(f"Dir not found for adding files: {directory_path}")

    async def add_checkpoint_files(self, checkpoint_dir: str):
        async with self._lock:
            self._add_files_from_dir_to_list(checkpoint_dir, self._adapter_files, set(self._adapter_files))

    # --- Synchronous No-Lock Status Dictionary Getters ---
    async def _get_server_status_dict_nolock(self) -> Dict[str, Any]:
        """Returns a dictionary with the full status of the engine."""
        # Assumes the asyncio.Lock (`self._lock`) is ALREADY HELD by the calling method.
        bytes_in_gb = 1024.0

        model_generation_config = None
        if self._peft_model is not None and hasattr(self._peft_model, 'generation_config'):
            model_generation_config = self._peft_model.generation_config.to_dict()
        
        # Get the full inference status dict first, as it contains the calculated peak memory
        inference_status_dict = await self._get_inference_status_dict_nolock()

        status_dict = round_floats({
            "instance_id": self.instance_id,
            "model_intrinsic_generation_config": model_generation_config,
            "server_status": self._server_status.value,
            "busy_reason": self._busy_reason,
            "engine_mode": self._engine_mode.value if self._engine_mode else None,
            "global_config": self._global_config, # Already a dict, no lock needed to read
            "base_model_class": str(type(self._base_model).__name__) if self._base_model is not None else None,
            "peft_model_class": str(type(self._peft_model).__name__) if self._peft_model is not None else None,
            "tokenizer_class": str(type(self._tokenizer).__name__) if self._tokenizer else None,
            "device": str(self._device) if self._device else None,
            "effective_eos_token_ids": self.effective_eos_token_ids,
            "effective_pad_token_id": self.effective_pad_token_id,
            "effective_stop_token_ids": self.effective_stop_token_ids,
            "effective_stop_tokens": self.effective_stop_tokens,
            "all_loaded_adapter_names": self._get_all_adapter_names_in_model_nolock(),
            "loaded_adapters_info": self._get_loaded_adapters_info_nolock(),
            "active_adapter_names": self._model_active_set,
            "primary_active_adapter_name": self._active_adapter_name_nolock(),
            "primary_active_adapter_type": self._model_active_set,
            "active_model_type": self._determine_active_model_type_status(),
            "training_component_status": self._get_training_status_dict_nolock(),
            "inference_component_status": inference_status_dict,
            # Top-level GPU memory for general server status
            "current_gpu_mem_allocated_gb": self.current_gpu_mem_allocated_mb / bytes_in_gb,
            "current_gpu_mem_reserved_gb": self.current_gpu_mem_reserved_mb / bytes_in_gb,
            "peak_gpu_mem_allocated_current_op_gb": self.peak_gpu_mem_allocated_current_op_mb / bytes_in_gb,
            "peak_gpu_mem_reserved_current_op_gb": self.peak_gpu_mem_reserved_current_op_mb / bytes_in_gb,
            "max_peak_gpu_mem_allocated_lifetime_training_gb": self.peak_tracked_gpu_mem_allocated_training_mb / bytes_in_gb,
            "max_peak_gpu_mem_reserved_lifetime_training_gb": self.peak_tracked_gpu_mem_reserved_training_mb / bytes_in_gb,
            "peak_tracked_gpu_mem_allocated_inference_gb": inference_status_dict.get("metrics", {}).get("peak_tracked_gpu_mem_allocated_inference_gb", 0.0),
            "peak_tracked_gpu_mem_reserved_inference_gb": inference_status_dict.get("metrics", {}).get("peak_tracked_gpu_mem_reserved_inference_gb", 0.0),
        })
        
        return status_dict

    async def get_server_status_dict(self) -> Dict[str, Any]:
        """Public async method to get server status, acquires lock."""
        async with self._lock:
            return await self._get_server_status_dict_nolock()

    def _determine_active_model_type_status(self) -> Optional[str]:
        if self._model_active_set:
            if len(self._model_active_set) == 1:
                return self._single_active_adapter_type_if_any # Type of the single active adapter
            return "mixed" # Multiple active adapters
        elif self._base_model is not None:
            return "BASE"
        return None
    
    def _get_training_status_dict_nolock(self) -> Dict[str, Any]:
        # This is a SYNCHRONOUS method. It assumes the asyncio.Lock (`self._lock`)
        # is ALREADY HELD by the calling ASYNCHRONOUS method (e.g., get_training_status_dict).
        bytes_in_gb = 1024.0
        result = {
            "status": self.training_status.value,
            "training_config_active": bool(self.current_training_config),
            "adapter_being_trained": self._training_adapter_name,
            "current_step": self._current_step,
            "total_steps": self._total_steps,
            "current_epoch": round(self._current_epoch, 2),
            "loss": round(self._loss, 4) if isinstance(self._loss, float) else self._loss,
            "learning_rate": self._learning_rate,
            "grad_norm": round(self._grad_norm, 4) if isinstance(self._grad_norm, float) else self._grad_norm,
            "error_message": self._training_error_message,
            "saved_adapter_files_count": len(self._adapter_files),
            "final_adapter_path": self._final_adapter_path,
            "output_dir": self.current_training_config.get("output_dir") if self.current_training_config else None,
            "progress_flags": {
                "dataset_loaded": self._dataset_loaded_for_training,
                "trainer_initialized": self._trainer_initialized_for_training,
                "training_loop_started": self._training_actually_started,
                "steps_completed": self._steps_completed_in_training,
                "non_zero_loss_reported": self._non_zero_loss_reported_in_training,
                "save_attempted": self._save_attempted_in_training,
                "lora_grads_observed": self._lora_grads_observed_in_training,
            }
        }
        result["current_gpu_mem_allocated_mb"] = round(self.current_gpu_mem_allocated_mb, 2)
        result["current_gpu_mem_reserved_mb"] = round(self.current_gpu_mem_reserved_mb, 2)
        result["peak_gpu_mem_allocated_current_op_mb"] = round(self.peak_gpu_mem_allocated_current_op_mb, 2)
        result["peak_gpu_mem_reserved_current_op_mb"] = round(self.peak_gpu_mem_reserved_current_op_mb, 2)
        result["peak_tracked_gpu_mem_allocated_training_mb"] = round(self.peak_tracked_gpu_mem_allocated_training_mb, 2)
        result["peak_tracked_gpu_mem_reserved_training_mb"] = round(self.peak_tracked_gpu_mem_reserved_training_mb, 2)

        if self._training_start_time:
            result["elapsed_seconds"] = round((self._training_end_time or time.monotonic()) - self._training_start_time, 2)
            if len(self._step_times) > 0:
                avg_step_time = sum(self._step_times) / len(self._step_times)
                result["step_timing_avg_sec"] = round(avg_step_time, 3)
        if self.historical_loss:
            result["historical_metrics_points"] = len(self.historical_loss)
        reports_payload: Dict[str, Optional[Dict[str, Any]]] = {}
        for stage, payload in self._adapter_reports.items():
            if payload is None:
                reports_payload[stage] = None
            else:
                payload_copy = dict(payload)
                if isinstance(payload_copy.get("lines"), list):
                    payload_copy["lines"] = list(payload_copy["lines"])
                reports_payload[stage] = payload_copy
        if reports_payload:
            result["adapter_reports"] = reports_payload
        
        result["heuristic_settings_summary"] = self.heuristic_settings_summary
        if self.training_resource_report is not None:
            result["resource_report"] = self.training_resource_report
        if self.effective_training_config is not None:
            result["effective_training_config"] = self.effective_training_config
            if self.training_resource_report and "param_sources" in self.training_resource_report:
                result.setdefault("effective_training_config", {})
                result["effective_training_config"].setdefault("sources", self.training_resource_report.get("param_sources"))

        return result

    async def get_training_status_dict(self) -> Dict[str, Any]:
        """Public async method to get training status, acquires lock."""
        async with self._lock:
            return self._get_training_status_dict_nolock()

    async def _get_inference_status_dict_nolock(self) -> Dict[str, Any]:
        # Assumes the asyncio.Lock (`self._lock`) is ALREADY HELD by the calling method.
        
        result = {
            "status": self.inference_status.value,
            "inference_session_config_active": bool(self.current_inference_session_config),
            "error_message": self.last_inference_error,
            'is_warming_cache': self.is_warming_cache,
            "default_generation_config": self.current_inference_session_config.get("default_generation_config") if self.current_inference_session_config else None,
            "loaded_adapters_info": self._get_loaded_adapters_info_nolock(),
        }

        metrics = await self.get_aggregate_metrics()
        result["metrics"] = metrics

        return round_floats(result)

    async def reset_aggregate_metrics(self):
        """Clears the inference metrics history and resets aggregate counters."""
        async with self._aggregate_metrics_lock:
            self._inference_metrics_history.clear()
            self.aggregate_metrics.reset()

    async def get_inference_status_dict(self) -> Dict[str, Any]:
        """Public async method to get inference status, acquires lock."""
        async with self._lock:
            return await self._get_inference_status_dict_nolock()

    async def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Returns aggregate inference metrics with rounded floats."""
        # Compute peak from history
        bytes_in_gb = 1024.0
        peak_alloc_hist = 0.0
        peak_rsvd_hist = 0.0
        metrics_dict = {}

        # Data for tracked throughput
        tracked_total_output = 0
        intervals = []
        max_end_time_wall = 0.0

        async with self._aggregate_metrics_lock:
            # Make a copy to avoid modifying the state's raw data
            metrics_dict = self.aggregate_metrics.__dict__.copy()
            history_copy = list(self._inference_metrics_history)
            if history_copy:
                for item in history_copy: # item is InferenceMetricsHistoryItem
                    # For peak memory
                    if item.mem_allocated is not None: peak_alloc_hist = max(peak_alloc_hist, item.mem_allocated)
                    if item.mem_reserved is not None: peak_rsvd_hist = max(peak_rsvd_hist, item.mem_reserved)

                    # For tracked throughput
                    intervals.append((item.start_time_mono, item.end_time_mono))
                    tracked_total_output += item.total_output_tokens or 0
                    if item.end_time_wall is not None:
                        max_end_time_wall = max(max_end_time_wall, item.end_time_wall)

        # --- On-the-fly calculation for TPS metrics (outside lock) ---
        # 1. Average TPS (based on pure model generation time from cumulative aggregates)
        total_output_cumulative = metrics_dict.get("total_output_tokens", 0)
        total_gen_duration = metrics_dict.get("total_generation_duration_sec", 0.0)
        if total_gen_duration > 0:
            metrics_dict["total_avg_tps"] = total_output_cumulative / total_gen_duration
        else:
            metrics_dict["total_avg_tps"] = 0.0

        # 2. Tracked Throughput TPS (based on wall-clock time of requests in history, excluding idle time)
        total_busy_time = 0.0
        max_end_time_mono = 0.0
        if intervals:
            # Sort intervals by start time
            intervals.sort(key=lambda x: x[0])
            max_end_time_mono = max(i[1] for i in intervals)

            merged_start, merged_end = intervals[0]

            for i in range(1, len(intervals)):
                next_start, next_end = intervals[i]
                if next_start <= merged_end:
                    # Overlapping or adjacent interval, merge it
                    merged_end = max(merged_end, next_end)
                else:
                    # Disjoint interval, finalize the previous merged one and start a new one
                    total_busy_time += (merged_end - merged_start)
                    merged_start, merged_end = next_start, next_end

            # Add the last merged interval
            total_busy_time += (merged_end - merged_start)

        if total_busy_time > 0:
            metrics_dict["tracked_throughput_tps"] = tracked_total_output / total_busy_time
        else:
            metrics_dict["tracked_throughput_tps"] = 0.0

        # Add other on-the-fly metrics
        if max_end_time_mono > 0:
            metrics_dict["tracked_window_end_mono_ts"] = max_end_time_mono
        
        if max_end_time_wall > 0:
            metrics_dict["tracked_window_end_utc_ts"] = datetime.datetime.fromtimestamp(max_end_time_wall, tz=datetime.timezone.utc).isoformat()

       # --- Refactor memory metrics in aggregate_metrics ---
        # Convert from MB to GB and rename keys
        if "mem_allocated" in metrics_dict:
            mem_alloc_mb = metrics_dict.pop("mem_allocated")
            metrics_dict["last_mem_allocated_gb"] = mem_alloc_mb / bytes_in_gb if mem_alloc_mb is not None else None
        if "mem_reserved" in metrics_dict:
            mem_rsvd_mb = metrics_dict.pop("mem_reserved")
            metrics_dict["last_mem_reserved_gb"] = mem_rsvd_mb / bytes_in_gb if mem_rsvd_mb is not None else None

        # --- End of on-the-fly calculation ---

        final_metrics = {
            "active_requests": len(self._active_inference_request_ids),
            "tracked_history_request_count": len(history_copy),
            "tracked_throughput_tps": metrics_dict.pop("tracked_throughput_tps", 0.0),
            "tracked_total_busy_time_sec": total_busy_time,
            "tracked_window_end_utc_ts": metrics_dict.pop("tracked_window_end_utc_ts", None),
            "tracked_window_end_mono_ts": metrics_dict.pop("tracked_window_end_mono_ts", None),
            "current_gpu_mem_allocated_gb": self.current_gpu_mem_allocated_mb / bytes_in_gb,
            "current_gpu_mem_reserved_gb":  self.current_gpu_mem_reserved_mb / bytes_in_gb,
            "peak_tracked_gpu_mem_allocated_inference_gb": peak_alloc_hist / bytes_in_gb,
            "peak_tracked_gpu_mem_reserved_inference_gb": peak_rsvd_hist / bytes_in_gb,
            "aggregate_metrics": metrics_dict,
        }

        return round_floats(final_metrics)
