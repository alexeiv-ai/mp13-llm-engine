# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""MP13 Training Logic"""

import datetime # Added for timestamp string
import inspect
import os
import logging
import asyncio
import time
import traceback
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Union, Tuple
from collections import defaultdict
import math
import json
import torch
import torch.nn as nn
import shutil # Added for moving checkpoints
from datasets import load_dataset
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import LoraConfig
from .mp13_prep import get_sft_formatting_func, map_preprocess_sft
from .mp13_utils import generate_training_report
from .mp13_utils import save_adapter_metadata, copy_report_and_metadata_files, save_adapter_package_for_mixed
from .mp13_callbacks import (
    TrainingProgressCallback,
    QuietTrainer,
    CastLoRAWeights,
    LoRAGradMonitorCallback,
)
from .mp13_adapter import create_quantization_metadata, quant_display_from_meta
from . import mp13_state
from .mp13_state import (
    TrainingStatus, EngineModeState, BusyError, AdapterError, ModeMismatchError,
    EngineInitializationError, ConfigurationError, TrainingError
)
from .mp13_config import APIStatus, TrainingConfig
from pathlib import Path


if TYPE_CHECKING:
    from .mp13_engine import MP13Engine


def _create_oom_suggestion_message(
    config: "TrainingConfig",
    params_used: Dict[str, Any],
    effective_max_seq_len: int
) -> str:
    """Creates a detailed error message with suggestions for an OOM error."""
    oom_suggestions = []
    oom_suggestions.append("--- Optimistic Fixes (try one of these) ---")

    if not params_used.get("gradient_checkpointing"):
        oom_suggestions.append("1. Enable gradient checkpointing (biggest memory saver). In your TrainingConfig, set: 'gradient_checkpointing': True")
    
    if params_used.get("per_device_train_batch_size", 1) > 1:
            oom_suggestions.append(f"2. Reduce the batch size. Current is {params_used.get('per_device_train_batch_size')}. In TrainingConfig, set: 'per_device_train_batch_size': {max(1, params_used.get('per_device_train_batch_size', 1) // 2)}")

    if not config.auto_manage_memory:
        oom_suggestions.append("3. Let the engine manage memory automatically. In TrainingConfig, ensure 'auto_manage_memory' is True (or not set).")

    oom_suggestions.append("\n--- Brute-Force Fix (most likely to work) ---")
    oom_suggestions.append("Use this configuration to force minimal memory usage:")
    oom_suggestions.append("  'auto_manage_memory': True,")
    oom_suggestions.append("  'gradient_checkpointing': True,")
    oom_suggestions.append("  'per_device_train_batch_size': 1")
    
    final_error_message = (
        f"CUDA Out of Memory Error.\n\n"
        f"This means the model, context length ({effective_max_seq_len}), and batch size are too large for your GPU's VRAM.\n\n"
        f"--- Parameters used in failed attempt ---\n"
        f"  - per_device_train_batch_size: {params_used.get('per_device_train_batch_size')}\n"
        f"  - gradient_accumulation_steps: {params_used.get('gradient_accumulation_steps')}\n"
        f"  - gradient_checkpointing: {params_used.get('gradient_checkpointing')}\n\n"
        f"--- Suggestions for your next run ---\n" +
        "\n".join(oom_suggestions)
    )
    return final_error_message

def _determine_safe_training_params(
    logger: logging.Logger,
    engine: "MP13Engine",
    training_config: "TrainingConfig",
    effective_max_seq_len: int,
    dataset_size: Optional[int] = None
) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    """
    Analyzes hardware and config to determine safe training parameters.
    Returns a tuple containing:
    - A dictionary with safe training parameters.
    - The final effective sequence length to be used for training.
    - A resource/heuristic report dict for diagnostics.
    """
    from .mp13_utils import get_hardware_report
    logger.info("--- Determining safe training parameters ---")
    report = get_hardware_report(engine.state.base_model)
    logger.info(f"Hardware Report: Device={report.device_type}, Name='{report.device_name}', Available VRAM={report.available_memory_gb:.2f}GB")
    param_sources: Dict[str, str] = {}

    # --- Path 1: CPU Training ---
    if report.device_type == 'cpu':
        logger.warning("Training on CPU. This will be extremely slow. Forcing conservative parameters.")
        safe_params = {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps, # Respect user's GA on CPU
            "gradient_checkpointing": False, # Not applicable for CPU
            "trainer_compute_precision": "fp32" # Mixed precision not available on CPU
        }
        resource_report = {
            "hardware": report.__dict__,
            "final_effective_seq_len": effective_max_seq_len,
            "mode": "cpu"
        }
        return safe_params, effective_max_seq_len, resource_report

    # --- Path 2: GPU Training, Manual Mode ---
    if not training_config.auto_manage_memory:
        logger.info("auto_manage_memory is False. Using user-provided training parameters directly.")
        use_grad_checkpointing = training_config.gradient_checkpointing
        if use_grad_checkpointing is None:
            use_grad_checkpointing = effective_max_seq_len > 4096
            logger.info(f"gradient_checkpointing is 'auto', enabling it due to context length ({effective_max_seq_len}).")

        safe_params = {
            "per_device_train_batch_size": training_config.per_device_train_batch_size,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "gradient_checkpointing": use_grad_checkpointing,
            "trainer_compute_precision": training_config.trainer_compute_precision
        }
        resource_report = {
            "hardware": report.__dict__,
            "final_effective_seq_len": effective_max_seq_len,
            "mode": "manual"
        }
        param_sources["per_device_train_batch_size"] = "user"
        param_sources["gradient_accumulation_steps"] = "user"
        param_sources["gradient_checkpointing"] = "user" if training_config.gradient_checkpointing is not None else "auto"
        param_sources["trainer_compute_precision"] = "user"
        resource_report["param_sources"] = param_sources
        return safe_params, effective_max_seq_len, resource_report

    # --- Path 3: GPU Training with Auto-Management ---
    logger.info("auto_manage_memory is True. Applying heuristics to determine parameters.")
    
    final_effective_seq_len = effective_max_seq_len
    # 1. Apply heuristic context capping
    cap_value = training_config.heuristic_context_cap
    if cap_value and cap_value > 0 and final_effective_seq_len > cap_value:
        logger.info(f"Heuristic is capping training context length from {final_effective_seq_len} down to {cap_value} for efficiency.")
        final_effective_seq_len = cap_value
        param_sources["context_length"] = "heuristic_cap"
    else:
        param_sources["context_length"] = "config_or_model"

    # 2. Decide on Gradient Checkpointing
    use_grad_checkpointing = training_config.gradient_checkpointing
    if use_grad_checkpointing is None: # 'auto' mode
        use_grad_checkpointing = final_effective_seq_len > 4096
        log_msg = "enabled" if use_grad_checkpointing else "disabled"
        logger.info(f"gradient_checkpointing is 'auto', decided to keep it {log_msg} based on context length ({final_effective_seq_len}).")
        param_sources["gradient_checkpointing"] = "auto"
    else:
        logger.info(f"User forced gradient_checkpointing to {use_grad_checkpointing}.")
        param_sources["gradient_checkpointing"] = "user"

    # 3. Estimate Memory Budget
    from .mp13_utils import inspect_device_layout
    layout = inspect_device_layout(engine.state.base_model)
    num_gpus = sum(1 for d in layout.get('devices', set()) if str(d).startswith('cuda'))
    if num_gpus == 0: num_gpus = 1
    logger.info(f"Model layout: {layout.get('mode', 'unknown')} on {num_gpus} GPU(s).")
    if report.devices:
        per_dev_log = ", ".join(
            f"cuda:{d.get('id')} free={float(d.get('free_gb', 0.0)):.2f}GB total={float(d.get('total_gb', 0.0)):.2f}GB"
            for d in report.devices
        )
        logger.info(f"Per-device free memory (pre-alloc): {per_dev_log}")

    safety_margin = 0.90
    available_pool_gb = report.available_memory_gb
    training_budget_gb = available_pool_gb * safety_margin

    # Optimizer estimate based on trainable parameters, not full model size (LoRA-friendly)
    optimizer_mem_per_device_gb = 0.0
    try:
        trainable_params = sum(p.numel() for p in engine.state._peft_model.parameters() if p.requires_grad)  # type: ignore[attr-defined]
        # Approximate AdamW states: param + m + v, assume 4 bytes each -> 12 bytes/param
        optimizer_mem_total_gb = (trainable_params * 12) / (1024**3)
        optimizer_mem_per_device_gb = optimizer_mem_total_gb / max(1, num_gpus)
    except Exception:
        # Fallback to base-model estimate if trainable scan fails
        total_optimizer_mem_gb = (report.model_params_b * 1e9 * 8) / (1024**3)
        optimizer_mem_per_device_gb = total_optimizer_mem_gb / num_gpus

    # If the optimizer estimate is larger than available budget, cap it conservatively
    if optimizer_mem_per_device_gb > available_pool_gb * 0.9:
        logger.info(
            f"Optimizer memory estimate ({optimizer_mem_per_device_gb:.2f}GB) exceeds available pool ({available_pool_gb:.2f}GB). "
            "Capping to 25% of available memory for safety."
        )
        optimizer_mem_per_device_gb = available_pool_gb * 0.25

    training_budget_gb -= optimizer_mem_per_device_gb
    if training_budget_gb <= 0:
        logger.warning(
            "Not enough VRAM after optimizer accounting. Forcing conservative floor budget and enabling gradient checkpointing."
        )
        training_budget_gb = max(0.5, available_pool_gb * 0.25)
        use_grad_checkpointing = True
        param_sources["gradient_checkpointing"] = "forced_for_budget"
    logger.info(
        "Memory Budget: Available(min per device)=%.2fGB, Safety Margin=%.0f%%, "
        "Optimizer Est. (per device)=%.2fGB -> Training Budget=%.2fGB",
        available_pool_gb,
        safety_margin * 100,
        optimizer_mem_per_device_gb,
        training_budget_gb,
    )

    # 4. Estimate Memory per Sample (Heuristic)
    K_grad_checkpointing = 2.0e12
    K_no_grad_checkpointing = 0.7e12
    K = K_grad_checkpointing if use_grad_checkpointing else K_no_grad_checkpointing
    mem_per_sample_gb = (report.model_params_b * (final_effective_seq_len**2)) / K if K > 0 else float('inf')
    logger.info(f"Heuristic Estimate: Memory per sample = {mem_per_sample_gb * 1024:.2f}MB (GC={'On' if use_grad_checkpointing else 'Off'})")

    # 5. Determine Batch Size
    max_possible_batch_size = int(training_budget_gb / mem_per_sample_gb) if mem_per_sample_gb > 1e-9 else 256
    final_batch_size = min(training_config.per_device_train_batch_size, max_possible_batch_size)
    final_batch_size = max(1, final_batch_size)
    param_sources["per_device_train_batch_size"] = "user"
    if training_config.auto_manage_memory and training_config.per_device_train_batch_size <= 1 and max_possible_batch_size > final_batch_size:
        boosted = min(max_possible_batch_size, 8)
        if boosted > final_batch_size:
            logger.info(f"Heuristic increased per_device_train_batch_size from {final_batch_size} to {boosted} to better utilize memory.")
            final_batch_size = boosted
            param_sources["per_device_train_batch_size"] = "heuristic_scaled_up"
    if final_batch_size < training_config.per_device_train_batch_size:
        logger.warning(f"Heuristic lowered per_device_train_batch_size from {training_config.per_device_train_batch_size} to {final_batch_size} to fit in memory.")
        param_sources["per_device_train_batch_size"] = "budget_limited"
    
    # 6. Determine Gradient Accumulation
    target_effective_batch_size = 32
    if dataset_size is not None and dataset_size > 0:
        target_effective_batch_size = min(target_effective_batch_size, max(4, dataset_size))
    final_grad_accum = max(1, target_effective_batch_size // final_batch_size)
    if training_config.gradient_accumulation_steps > final_grad_accum:
        logger.info(f"Using user-provided gradient_accumulation_steps ({training_config.gradient_accumulation_steps}) as it's higher than the heuristic's ({final_grad_accum}).")
        final_grad_accum = training_config.gradient_accumulation_steps
        param_sources["gradient_accumulation_steps"] = "user"
    else:
        param_sources["gradient_accumulation_steps"] = "auto"

    safe_params = {
        "per_device_train_batch_size": final_batch_size,
        "gradient_accumulation_steps": final_grad_accum,
        "gradient_checkpointing": use_grad_checkpointing,
        "trainer_compute_precision": training_config.trainer_compute_precision
    }
    normalized_layout = dict(layout) if isinstance(layout, dict) else layout
    if isinstance(normalized_layout, dict) and isinstance(normalized_layout.get("devices"), set):
        normalized_layout = dict(normalized_layout)
        normalized_layout["devices"] = sorted(normalized_layout["devices"])
    hardware_payload = dict(report.__dict__)
    if isinstance(hardware_payload.get("layout"), dict):
        hardware_payload["layout"] = dict(hardware_payload["layout"])
        if isinstance(hardware_payload["layout"].get("devices"), set):
            hardware_payload["layout"]["devices"] = sorted(hardware_payload["layout"]["devices"])
    resource_report = {
        "hardware": hardware_payload,
        "layout": normalized_layout,
        "safety_margin": safety_margin,
        "available_pool_gb": available_pool_gb,
        "optimizer_mem_per_device_gb": optimizer_mem_per_device_gb,
        "training_budget_gb": training_budget_gb,
        "mem_per_sample_gb": mem_per_sample_gb,
        "target_effective_batch_size": target_effective_batch_size,
        "dataset_size": dataset_size,
        "final_batch_size": final_batch_size,
        "final_grad_accum": final_grad_accum,
        "final_effective_seq_len": final_effective_seq_len,
        "gradient_checkpointing": use_grad_checkpointing,
        "param_sources": param_sources,
    }
    return safe_params, final_effective_seq_len, resource_report




class _TrainerModelShim(nn.Module):
    """
    Proxy that presents the *inner* HF model (which already has PEFT layers injected)
    to Transformers' Trainer. It exposes `config`, forwards `forward`, and implements
    `get_base_model()` so Trainer init doesn't choke (MixedModel lacks it).
    """
    def __init__(self, inner, peft_host: Optional[nn.Module] = None):
        super().__init__()
        self.add_module("core", inner)        # the module Trainer will call forward() on
        object.__setattr__(self, "_inner", inner)
        # host that actually carries adapters/peft_config (e.g., mixed or mixed._orig_mod)
        object.__setattr__(self, "_peft_host", peft_host or inner)
        # Inform Trainer that adapters are already attached to quantized weights.
        self._hf_peft_config_loaded = True

    # ---- Core expectations ----
    def forward(self, *args, **kwargs):
        # Trainer (as of HF 4.44+) may inject bookkeeping kwargs (e.g., num_items_in_batch)
        # that some model forwards (Phi-3) do not accept. Drop them defensively.
        kwargs.pop("num_items_in_batch", None)
        return self._inner(*args, **kwargs)

    # >>> CRITICAL: surface PEFT to the Trainer
    @property
    def peft_config(self):
        return getattr(self._peft_host, "peft_config", {})

    def get_base_model(self):
        # some Trainer code probes this
        return self._inner

    @property
    def config(self):
        return getattr(self._inner, "config", None)

    # Keep train/eval toggles coherent (and avoid recursion)
    def train(self, mode: bool = True):
        self._inner.train(mode)
        return super().train(mode)

    def eval(self):
        self._inner.eval()
        return super().eval()

    # ---- Transparent passthrough for anything else (generate, device, dtype, etc.) ----
    def __getattr__(self, name):
        # First try nn.Module attributes (won't recurse)
        try:
            return super().__getattribute__(name)
        except AttributeError as err:
            # delegate unknown attrs to the inner first, then to peft_host
            inner = object.__getattribute__(self, "_inner")
            if hasattr(inner, name):
                return getattr(inner, name)

            peft_host = object.__getattribute__(self, "_peft_host")
            if hasattr(peft_host, name):
                return getattr(peft_host, name)

            raise err


class _TokenTypeIdsCollator:
    """Wraps a collator to inject token_type_ids for models that require it during training."""
    def __init__(self, base_collator, require_for_model_type: Optional[str]):
        self._base_collator = base_collator
        self._require_for_model_type = require_for_model_type

    def __call__(self, features):
        batch = self._base_collator(features)
        if self._require_for_model_type and "token_type_ids" not in batch:
            input_ids = batch.get("input_ids")
            if input_ids is not None:
                batch["token_type_ids"] = torch.zeros_like(input_ids)
        return batch


async def start_training_logic(engine: "MP13Engine", training_config_data: Dict[str, Any]
                                ) -> Dict[str, Any]:
    engine.state.logger.info(f"--- Received Start Training Request ({engine.instance_id}) ---")
    current_wall_time = time.time()
    if engine.state.base_model is None or engine.state.tokenizer is None:
        raise EngineInitializationError("Global resources (actual base model/tokenizer) not initialized.")
    if engine.state.engine_mode != EngineModeState.TRAIN:
        raise ModeMismatchError(f"Cannot start training. Engine mode is '{engine.state.engine_mode.value if engine.state.engine_mode else 'UNSET'}', expected TRAIN. Call check_set_mode first.")
    allowed_start_statuses = [
        TrainingStatus.READY, 
        TrainingStatus.COMPLETED, 
        TrainingStatus.STOPPED, 
        TrainingStatus.ERROR,
        TrainingStatus.OFFLINE
    ]
    if engine.state.training_status not in allowed_start_statuses:
        raise BusyError(f"Cannot start new training. Training status is '{engine.state.training_status.value}'. Expected one of {', '.join(s.value for s in allowed_start_statuses)}.")
    try:
        config = TrainingConfig(**training_config_data) 
    except Exception as e:
        raise ConfigurationError(f"Invalid TrainingConfig provided: {e}") from e
    adapter_to_train = config.adapter_name_to_train
    loaded_adapters = await engine.state.get_all_adapter_names_in_model()
    if adapter_to_train not in loaded_adapters:
        raise AdapterError(f"Adapter '{adapter_to_train}' is not loaded. Please add the adapter before starting training.")
    await engine.state.reset_training()
    await engine.state.set_training_config(config.model_dump(), adapter_to_train, str(config.output_dir), current_wall_time)
    await engine.state.set_training_status(TrainingStatus.PREPARING, f"Configuration set for adapter '{adapter_to_train}'. Starting training preparation task.")
    try:
        # The Trainer will enable gradient checkpointing based on TrainingArguments.
        # We set model.config.use_cache=False inside the training thread to prevent warnings.
        engine.state._training_task = asyncio.create_task(
            execute_training_run_logic(engine, config)
        )
        engine.state.logger.info(f"--- Training for adapter '{adapter_to_train}' initiated. Output: '{str(config.output_dir)}'. (Prep Time: {time.time() - current_wall_time:.2f}s) ---")
        status_dict = await engine.state.get_training_status_dict()
        return status_dict
    except Exception as e:
        error_msg = f"Failed to start training: {type(e).__name__}: {e}"
        engine.state.logger.critical(f"!!! {error_msg}\n{traceback.format_exc()}")
        await engine.state.set_training_error(error_msg)
        status_dict_err = await engine.state.get_training_status_dict()
        raise TrainingError({"message":error_msg, "details": status_dict_err})

async def stop_training_logic(engine: "MP13Engine")-> Dict[str, Any]:
    engine.state.logger.info(f"--- Received Stop Training Request (Graceful) ({engine.instance_id}) ---")
    if engine.state.training_status != TrainingStatus.TRAINING:
        msg = f"No active training session to stop. Current status: {engine.state.training_status.value}"
        engine.state.logger.info(msg)
        non_active_stoppable_statuses = [
            TrainingStatus.OFFLINE, 
            TrainingStatus.READY, 
            TrainingStatus.COMPLETED, 
            TrainingStatus.ERROR, 
            TrainingStatus.STOPPED
        ]
        if engine.state.training_status in non_active_stoppable_statuses:
            return {"message": msg}
        else:
            raise TrainingError({"message":msg})
    engine.state.logger.info("Setting _graceful_stop_requested = True in engine state for callback.")
    engine.state._graceful_stop_requested = True
    if engine.state._trainer_instance and hasattr(engine.state._trainer_instance, 'request_graceful_stop'):
        engine.state.logger.info("Trainer instance has 'request_graceful_stop'. Calling it.")
        try:
            await asyncio.to_thread(engine.state._trainer_instance.request_graceful_stop)
        except Exception as e_rg_stop:
            engine.state.logger.warning(f"Warning: Error calling custom 'request_graceful_stop' on trainer: {e_rg_stop}")
    else:
        engine.state.logger.info("Trainer instance not found or does not have 'request_graceful_stop'. Relying on callback and _graceful_stop_requested flag.")
    return {"message": "Graceful stop initiated. Trainer will attempt to save and exit upon next callback check."}

async def execute_training_run_logic(engine: "MP13Engine", config: TrainingConfig):
    engine.state.logger.info(f"--- Entering _execute_training_run ({engine.instance_id}) ---")
    if engine.state.training_status != TrainingStatus.PREPARING:
        raise TrainingError(f"Training component not in PREPARING state. Current: {engine.state.training_status.value}. Initialize first.")
    if engine.state.engine_mode != EngineModeState.TRAIN:
        raise ModeMismatchError(f"Cannot run training. Engine mode is {engine.state.engine_mode.value if engine.state.engine_mode else 'UNSET'}, expected TRAIN.")
    if engine.state.tokenizer is None:
        raise TrainingError("Training failed. Tokenizer not available.")
    adapter_to_train = config.adapter_name_to_train
    adapter_type = await engine.state.get_adapter_type(adapter_to_train)
    adapter_current_root_path = await engine.state.get_adapter_root_path(adapter_to_train) # For fallback output dir
    loaded_adapter_info = (await engine.state.get_loaded_adapters_info()).get(adapter_to_train, {})
    adapter_was_foreign = bool(loaded_adapter_info.get("is_foreign", False))
    adapter_loaded_checkpoint_path = loaded_adapter_info.get("checkpoint_path")
    storage_hint = config.output_dir or adapter_current_root_path or adapter_to_train
    training_run_output_dir = engine.adapters_control.derive_adapter_root_for_storage(adapter_to_train, storage_hint)
    config.output_dir = str(training_run_output_dir)
    async with engine.state._lock:
        if engine.state.current_training_config is not None:
            engine.state.current_training_config["output_dir"] = str(training_run_output_dir)
    await asyncio.to_thread(os.makedirs, str(training_run_output_dir), exist_ok=True)
    engine.state.logger.info(f"Effective adapter root for this training run: {training_run_output_dir}")
    engine.state.logger.info(f"Model/precision folder component: {engine.adapters_control.build_model_precision_dir_name()}")
    if adapter_was_foreign:
        engine.state.logger.info(f"Adapter '{adapter_to_train}' is marked as foreign (missing metadata). Training will emit metadata and reset the flag.")

    loop = asyncio.get_running_loop()

    # Instantiate callbacks here, in the async thread, so they capture the correct event loop.
    callbacks_for_trainer: list[TrainerCallback] = [
        TrainingProgressCallback(
            # engine.state._lock, # No longer passing lock directly
            engine.state,
            time.time(),
            engine.state.tokenizer
        )
    ]
    # Add CastLoRAWeights callback if using bf16/fp16 precision
    if config.trainer_compute_precision in ["bf16", "fp16"]:
        target_dtype = torch.bfloat16 if config.trainer_compute_precision == "bf16" else torch.float16
        callbacks_for_trainer.append(CastLoRAWeights(logger=engine.state.logger, dtype=target_dtype))
        engine.state.logger.info(f"Added CastLoRAWeights callback for {config.trainer_compute_precision} precision.")

    grad_monitor_callback = LoRAGradMonitorCallback(logger=engine.state.logger)
    callbacks_for_trainer.append(grad_monitor_callback)


    # ---------------------------------------------------------------------
    # We will log trainable parameters, shapes summary and adaptetr assessment
    #  after activating the adapter, just before creating the Trainer.
    # ---------------------------------------------------------------------

    # Shared utilities 

    def _active_adapter_name(m):
        # Directly queries the `active_adapter` attribute from the model object `m`.
        val = getattr(m, "active_adapter", None) 
        if isinstance(val, str):
            return val
        cfg = getattr(m, "peft_config", None)
        if isinstance(cfg, dict) and len(cfg) == 1:
            return next(iter(cfg.keys()))
        for n, _ in m.named_parameters():
            if ".lora_A." in n:
                return n.split(".lora_A.")[1].split(".")[0]
        return "<none>"

    def _is_meta_tensor(t) -> bool:
        try:
            return bool(getattr(t, "is_meta", False)) or (getattr(t, "device", None) is not None and t.device.type == "meta")
        except Exception:
            return False
    
    def _adapter_targets_summary(peft_model, adapter_name) -> tuple[defaultdict, int, int, int]:
        """
        Returns per-target stats and global totals for the active LoRA adapter.
        Collects sizes from lora_A/lora_B pairs:
        - in_features, out_features, r
        - delta_params (dpl) and base_params (bpl) per site
        - simple headroom proxy (fcp = r / min(in,out))
        """
        from collections import defaultdict

        A, B = {}, {}
        for pname, p in peft_model.named_parameters():
            if f".lora_A.{adapter_name}.weight" in pname:
                pref = pname.split(".lora_A.")[0]
                A[pref] = (p.shape[1], p.shape[0])  # (in, r)
            elif f".lora_B.{adapter_name}.weight" in pname:
                pref = pname.split(".lora_B.")[0]
                B[pref] = (p.shape[0], p.shape[1])  # (out, r)

        grouped = defaultdict(lambda: {
            "cnt": 0, "in": 0, "r": 0, "out": 0,
            "dpl": 0, "bpl": 0, "fcp_sum": 0.0
        })
        total_delta_params = 0
        total_base = 0
        mismatches = 0

        for pref in sorted(set(A) & set(B)):
            in_f, rA = A[pref]
            out_f, rB = B[pref]
            if rA != rB:
                mismatches += 1
                continue
            r = rA
            dpl = (r * in_f) + (out_f * r)
            bpl = out_f * in_f

            tgt = pref.split(".")[-1]  # last token usually q_proj/up_proj/etc
            g = grouped[tgt]
            g["cnt"] += 1
            g["in"] = in_f
            g["r"] = r
            g["out"] = out_f
            g["dpl"] += dpl
            g["bpl"] += bpl
            g["fcp_sum"] += (float(r) / float(max(1, min(in_f, out_f))))

            total_delta_params += dpl
            total_base += bpl

        return grouped, total_delta_params, total_base, mismatches


    def _adapter_dtype_and_mem(peft_model, trainable_only=True, params_override=None) -> tuple[str, float]:
        """
        Returns (dtype_str, mem_mb) for the trainable adapter weights or a provided param-count.
        """
        # majority dtype among requires_grad params
        trainable_dtypes = {}
        total_trainable = 0
        for _, p in peft_model.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            trainable_dtypes[str(p.dtype)] = trainable_dtypes.get(str(p.dtype), 0) + p.numel()
            total_trainable += p.numel()

        dtype_head = "<unknown>"
        if trainable_dtypes:
            dtype_head = max(trainable_dtypes.items(), key=lambda kv: kv[1])[0].replace("torch.", "")

        bytes_per = {"float32": 4, "bfloat16": 2, "float16": 2}.get(dtype_head, 2)

        # if params_override given, estimate memory for just those params
        if params_override is not None:
            mem_mb = (int(params_override) * bytes_per) / (1024.0 * 1024.0)
        else:
            # fallback to all trainables
            mem_mb = (total_trainable * bytes_per) / (1024.0 * 1024.0)

        return dtype_head, mem_mb


    # ---------------------------------------------------------------------
    # 1) Verbose adapter assessment with more behavior classes and ASCII-only
    # ---------------------------------------------------------------------

    def _log_active_adapter_shapes_verbose(logger: logging.Logger, peft_model):
        name = _active_adapter_name(peft_model)
        if name == "<none>":
            logger.info("[PEFT] Behavioral Impact: No active adapter.")
            return {}

        grouped, total_delta, total_base, mismatches = _adapter_targets_summary(peft_model, name)

        cfg = (getattr(peft_model, "peft_config", {}) or {}).get(name)
        r_cfg = getattr(cfg, "r", None) if cfg is not None else None
        alpha = getattr(cfg, "lora_alpha", None) if cfg is not None else None
        a_over_r = (float(alpha) / float(r_cfg)) if (alpha and r_cfg) else 1.0
        lora_dropout = getattr(cfg, "lora_dropout", None)

        if mismatches > 0:
            logger.warning(f"[PEFT] WARNING: {mismatches} adapter site(s) had mismatched ranks and were skipped.")

        if not grouped:
            logger.info(f"[PEFT] Behavioral Impact: Adapter '{name}' has no active sites.")
            return {}

        # Per-target breakdown
        logger.info(f"[PEFT] Adapter '{name}' targets {len(grouped)} module types:")
        for tgt, g in sorted(grouped.items()):
            cnt, in_f, r, out_f = g["cnt"], g["in"], g["r"], g["out"]
            avg_fcp = (g["fcp_sum"] / max(1, cnt))
            dpl = g["dpl"]; bpl = g["bpl"]
            logger.info(f"  - {cnt}x {tgt}  base={out_f}x{in_f}  r={r}  dparams/layer~{((r*in_f)+(out_f*r)):,}  base/layer~{(out_f*in_f):,}  FCP~{avg_fcp:.6f}  dsum={dpl:,}")

        # Subsystem and fine-grained buckets for behavior classes
        attn_set = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}
        mlp_set  = {"gate_proj", "up_proj", "down_proj", "gate_up_proj"}

        buckets = {
            "attn_q": 0, "attn_k": 0, "attn_v": 0, "attn_o": 0, "attn_all": 0,
            "mlp_gate": 0, "mlp_up": 0, "mlp_down": 0, "mlp_all": 0,
            "other": 0
        }
        for tgt, g in grouped.items():
            d = g["dpl"]
            if tgt in attn_set:
                buckets["attn_all"] += d
                if tgt == "q_proj": buckets["attn_q"] += d
                elif tgt == "k_proj": buckets["attn_k"] += d
                elif tgt == "v_proj": buckets["attn_v"] += d
                elif tgt == "o_proj": buckets["attn_o"] += d
            elif tgt in mlp_set:
                buckets["mlp_all"] += d
                if tgt == "gate_proj": buckets["mlp_gate"] += d
                elif tgt == "up_proj": buckets["mlp_up"] += d
                elif tgt == "down_proj": buckets["mlp_down"] += d
            else:
                buckets["other"] += d

        def _pct(x):
            return (100.0 * x / max(1, total_delta))

        # Headline strength and capacity
        pct_delta_vs_base = (100.0 * total_delta / total_base) if total_base > 0 else 0.0
        ASI = pct_delta_vs_base * float(a_over_r if a_over_r else 1.0)  # alpha/r scaled
        # Free capacity proxy: average over all sites
        fcp_global = sum(g["fcp_sum"] for g in grouped.values()) / max(1, sum(g["cnt"] for g in grouped.values()))

        # Memory estimate for just adapter weights
        dtype_head, mem_mb = _adapter_dtype_and_mem(peft_model, trainable_only=True, params_override=total_delta)

        logger.info("[PEFT] Behavioral Impact Assessment:")
        logger.info(f"  - Added Params: {total_delta:,} (~{mem_mb:.2f} MB at {dtype_head})")
        logger.info(f"  - Strength: dparams/base ~= {pct_delta_vs_base:.3f}%  alpha_over_r ~= {a_over_r:.3g}  ASI ~= {ASI:.3f}%")
        logger.info(f"  - Free Capacity Proxy (global): {fcp_global:.6f}  (lower means more rank headroom)")

        # Fine-grained behavior classes
        # These map adapter placement to expected qualitative effects.
        # References: LoRA core paper (rank vs capacity), PEFT docs (alpha/r scaling),
        # and notes that MLP vs Attention placements influence "knowledge vs style" tendencies.
        # See citations at the end of this message.

        classes = []
        attn_pct = _pct(buckets["attn_all"])
        mlp_pct  = _pct(buckets["mlp_all"])

        # Dominance classes
        if attn_pct >= 60.0 and mlp_pct < 30.0:
            classes.append("attention_dominated")
        elif mlp_pct >= 60.0 and attn_pct < 30.0:
            classes.append("mlp_dominated")
        else:
            classes.append("balanced_attn_mlp")

        # Attention micro-classes
        if _pct(buckets["attn_k"] + buckets["attn_v"]) >= 35.0:
            classes.append("routing_focus_kv_heavy")      # stronger focus/selection
        if _pct(buckets["attn_q"]) >= 25.0:
            classes.append("prompt_biasing_q_heavy")      # query shaping
        if _pct(buckets["attn_o"]) >= 25.0:
            classes.append("output_shaping_o_heavy")      # head mixing/output style

        # MLP micro-classes
        if _pct(buckets["mlp_up"] + buckets["mlp_gate"]) >= 35.0:
            classes.append("feature_injection_up_gate")   # content/feature expansion
        if _pct(buckets["mlp_down"]) >= 25.0:
            classes.append("compression_control_down")    # output projection shaping

        if _pct(buckets["other"]) >= 10.0:
            classes.append("misc_targets_present")

        logger.info("  - Behavioral Classes: " + ", ".join(classes))

        # Strength tier text
        if ASI > 5.0:
            strength_tier = "HIGH"
            logger.info("  - Strength Tier: HIGH (expect strong behavioral shifts)")
        elif ASI > 1.0:
            strength_tier = "MEDIUM"
            logger.info("  - Strength Tier: MEDIUM (expect noticeable changes)")
        else:
            strength_tier = "LOW"
            logger.info("  - Strength Tier: LOW (expect subtle changes)")

        if lora_dropout is not None:
            logger.info(f"  - Regularization: lora_dropout={lora_dropout}")

        # Guidance based on capacity proxy
        if fcp_global < 0.002:
            logger.info("  - Note: Very low FCP suggests rank is large vs layer dims; consider lower r or more regularization.")
        elif fcp_global > 0.02:
            logger.info("  - Note: High FCP suggests ample headroom; consider increasing r or broadening target_modules if underfitting.")

        behavior_metrics = {
            "adapter_name": name,
            "added_params": int(total_delta),
            "approx_mem_mb": float(mem_mb),
            "pct_delta_vs_base": float(pct_delta_vs_base),
            "ASI": float(ASI),
            "free_capacity_proxy": float(fcp_global),
            "attn_pct": float(attn_pct),
            "mlp_pct": float(mlp_pct),
            "classes": list(classes),
            "strength_tier": strength_tier,
            "lora_dropout": lora_dropout,
            "mismatched_sites": int(mismatches),
        }
        return behavior_metrics


    # ---------------------------------------------------------------------
    # 2) Adapter snapshot: add missing info
    # ---------------------------------------------------------------------

    def _log_adapter_snapshot(logger: logging.Logger, peft_model):
        """
        Snapshot includes:
        - counts by target module and total sites
        - rank/alpha patterns (detect non-uniform r or alpha)
        - trainable dtype, adapter-only memory estimate
        - optimizer state presence for adapter params (if accessible)
        - running norms of A, B, and implied delta W (Frobenius)
        """
        name = _active_adapter_name(peft_model)
        if name == "<none>":
            logger.info("[PEFT] Adapter: No active adapter.")
            return {}

        grouped, total_delta, total_base, mismatches = _adapter_targets_summary(peft_model, name)
        cfg = (getattr(peft_model, "peft_config", {}) or {}).get(name)
        r = getattr(cfg, "r", None) if cfg is not None else None
        alpha = getattr(cfg, "lora_alpha", None) if cfg is not None else None
        rank_pat = getattr(cfg, "rank_pattern", None) if cfg is not None else None
        alpha_pat = getattr(cfg, "alpha_pattern", None) if cfg is not None else None
        lora_dropout = getattr(cfg, "lora_dropout", None)

        dtype_head, mem_mb = _adapter_dtype_and_mem(peft_model, trainable_only=True, params_override=total_delta)

        if mismatches > 0:
            logger.warning(f"[PEFT] WARNING: {mismatches} mismatched rank site(s) skipped.")

        # Site counts by target
        per_target = {tgt: g["cnt"] for tgt, g in grouped.items()}
        logger.info(f"[PEFT] Adapter name={name} sites_total={sum(per_target.values())} per_target={per_target}")

        # Patterns and hyperparams
        logger.info(f"[PEFT] Hyperparams: r={r} alpha={alpha} dropout={lora_dropout} "
            f"rank_pattern={'yes' if rank_pat else 'no'} alpha_pattern={'yes' if alpha_pat else 'no'}")

        # Memory and dtype
        logger.info(f"[PEFT] Trainable dtype={dtype_head} adapter_params={total_delta:,} approx_mem={mem_mb:.2f} MB")

        # Lightweight norms for health
        # We avoid SVD; compute Frobenius norms of A, B and implied dW = B @ A per site.
        # These norms help sanity check scale growth over time in _log_adapter_progress.
        import math
        sum_normA = 0.0
        sum_normB = 0.0
        sum_normDW = 0.0
        counted = 0

        for pname, p in peft_model.named_parameters():
            if f".lora_A.{name}.weight" in pname or f".lora_B.{name}.weight" in pname:
                if _is_meta_tensor(p):
                    continue
                with torch.no_grad():
                    n = p.norm().item()
                if ".lora_A." in pname: sum_normA += n
                if ".lora_B." in pname: sum_normB += n
                counted += 1

        # Estimate ||dW||_F using per-site norms when both A and B exist:
        # ||BA||_F <= ||B||_F * ||A||_F; use it as a cheap proxy.
        sum_normDW = sum_normA * sum_normB

        logger.info(f"[PEFT] Norms (proxy): normA_sum~{sum_normA:.4f} normB_sum~{sum_normB:.4f} normBA_proxy~{sum_normDW:.4f} counted_tensors={counted}")

        snapshot_metrics = {
            "adapter_name": name,
            "sites_total": sum(per_target.values()),
            "per_target_sites": per_target,
            "trainable_dtype": dtype_head,
            "adapter_params": int(total_delta),
            "approx_mem_mb": float(mem_mb),
            "mismatched_sites": int(mismatches),
            "hyperparams": {
                "r": r,
                "alpha": alpha,
                "lora_dropout": lora_dropout,
                "rank_pattern": bool(rank_pat),
                "alpha_pattern": bool(alpha_pat),
            },
            "norms": {
                "sum_normA": float(sum_normA),
                "sum_normB": float(sum_normB),
                "norm_proxy": float(sum_normDW),
                "counted_tensors": counted,
            },
        }
        return snapshot_metrics


    # ---------------------------------------------------------------------
    # 3) Adapter progress: add remaining capacity trajectory and health
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _log_adapter_progress(logger: logging.Logger, peft_model,sample_layers=0, step=None, label=None):  # noqa: C901
        """
        Progress snapshot of the *loaded* adapter (what it has been through).
        - ASI: alpha_over_r scaled percent delta vs base
        - Remaining capacity: avg(1 - r/min(in,out)) across sites (percent)
        - FCP: avg(r/min(in,out)) across sites (lower => more headroom)
        - Subsystem shares: attention vs MLP
        - Grad norms: average grad norm over LoRA A/B if grads exist
        - Norm health: exact sample ||B@A||_F on up to `sample_layers` sites (CPU).
                    If sample_layers==0, logs a cheap proxy sum_normA * sum_normB.
        """

        name = _active_adapter_name(peft_model)
        if not name or name == "<none>":
            logger.info("[PEFT] Progress: no active adapter")
            return {}

        # Collect LoRA sites and shapes
        A_shapes = {}
        B_shapes = {}
        A_params = {}
        B_params = {}

        for pname, p in peft_model.named_parameters():
            if f".lora_A.{name}.weight" in pname:
                pref = pname.split(".lora_A.")[0]
                A_shapes[pref] = (p.shape[1], p.shape[0])  # (in, r)
                A_params[pref] = p
            elif f".lora_B.{name}.weight" in pname:
                pref = pname.split(".lora_B.")[0]
                B_shapes[pref] = (p.shape[0], p.shape[1])  # (out, r)
                B_params[pref] = p

        # Aggregate per-site stats
        sites = []
        total_delta = 0
        total_base = 0
        rem_sum = 0.0
        fcp_sum = 0.0

        attn_set = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}
        mlp_set = {"gate_proj", "up_proj", "down_proj", "gate_up_proj"}
        attn_d = 0
        mlp_d = 0
        other_d = 0

        for pref in sorted(set(A_shapes) & set(B_shapes)):
            in_f, rA = A_shapes[pref]
            out_f, rB = B_shapes[pref]
            if rA != rB:
                continue
            r = int(rA)
            dpl = (r * in_f) + (out_f * r)
            bpl = out_f * in_f
            total_delta += dpl
            total_base += bpl

            mi = float(max(1, min(in_f, out_f)))
            rem_sum += max(0.0, 1.0 - (r / mi))
            fcp_sum += (r / mi)

            tgt = pref.split(".")[-1]
            if tgt in attn_set:
                attn_d += dpl
            elif tgt in mlp_set:
                mlp_d += dpl
            else:
                other_d += dpl

            sites.append((pref, A_params[pref], B_params[pref], in_f, out_f, r, tgt))

        if not sites:
            logger.info("[PEFT] Progress: adapter has no matched A/B sites")
            return

        # Config for alpha/r scaling
        cfg = (getattr(peft_model, "peft_config", {}) or {}).get(name)
        r_cfg = getattr(cfg, "r", None) if cfg is not None else None
        alpha = getattr(cfg, "lora_alpha", None) if cfg is not None else None
        lora_dropout = getattr(cfg, "lora_dropout", None) if cfg is not None else None
        a_over_r = (float(alpha) / float(r_cfg)) if (alpha and r_cfg) else 1.0

        # Strength and capacity
        pct_delta_vs_base = (100.0 * total_delta / total_base) if total_base > 0 else 0.0
        ASI = pct_delta_vs_base * a_over_r

        rem_pct = 100.0 * (rem_sum / max(1, len(sites)))
        fcp_avg = fcp_sum / max(1, len(sites))

        total_d = float(max(1, attn_d + mlp_d + other_d))
        attn_share = 100.0 * attn_d / total_d
        mlp_share = 100.0 * mlp_d / total_d

        # Norm health: proxy or sampled exact
        sum_normA = 0.0
        sum_normB = 0.0
        for _, pA, pB, *_ in sites:
            try:
                if _is_meta_tensor(pA) or _is_meta_tensor(pB):
                    continue
                sum_normA += pA.norm().item()
                sum_normB += pB.norm().item()
            except Exception:
                pass
        norm_proxy = sum_normA * sum_normB

        ba_fro_sample = None
        if sample_layers and len(sites) > 0:
            k = max(1, min(int(sample_layers), len(sites)))
            picked = sites[:k]  # deterministic selection; change to random.sample for stochastic
            tot = 0.0
            for _, pA, pB, *_ in picked:
                try:
                    if _is_meta_tensor(pA) or _is_meta_tensor(pB):
                        continue
                    A_cpu = pA.detach().to("cpu")
                    B_cpu = pB.detach().to("cpu")
                    BA = B_cpu @ A_cpu
                    tot += BA.norm().item()  # Frobenius norm
                    del BA
                except Exception:
                    pass
            ba_fro_sample = tot

        # Header with optional step/label
        header = "[PEFT] "
        label_part  = None
        if step is not None:
            label_part = f" step={step}"
        if label:
            label_part = f" ({label})"

        # Determine adapter state (new or pretrained) based on B-matrix norms
        is_new_adapter = (ba_fro_sample is not None and ba_fro_sample == 0.0) or (ba_fro_sample is None and norm_proxy == 0.0)
        adapter_state_str = "Adapter is empty (new)" if is_new_adapter else "Adapter is pretrained"
        delta_norm_str = f"Delta Norm (Sampled): {ba_fro_sample:.4f}" if ba_fro_sample is not None else f"Delta Norm (Proxy): {norm_proxy:.4f}"

        # Format the first line of metrics
        logger.info(f"{header}({adapter_state_str}) Progress{label_part}: {delta_norm_str}  Capacity Left: {rem_pct:.1f}%")

        # Format the second line of metrics
        metrics_line = (
            f"  +- ASI: {ASI:.3f}% | Attn Share: {attn_share:.1f}% | MLP Share: {mlp_share:.1f}%"
        )
        logger.info(metrics_line)

        # Hints
        if rem_pct < 10.0:
            logger.info("  Note: remaining capacity under 10 percent; consider increasing rank or broadening targets if underfitting.")
        if ba_fro_sample is None and norm_proxy == 0.0:
            logger.info("  Note: adapter delta near zero; fresh LoRA often starts with B=0 and A random (BA zero at init).")

        progress_metrics = {
            "adapter_name": name,
            "ASI": float(ASI),
            "remaining_capacity_pct": float(rem_pct),
            "fcp_avg": float(fcp_sum / max(1, len(sites))),
            "attn_share_pct": float(attn_share),
            "mlp_share_pct": float(mlp_share),
            "delta_norm": float(ba_fro_sample if ba_fro_sample is not None else norm_proxy),
            "delta_norm_source": "sampled" if ba_fro_sample is not None else "proxy",
            "is_new_adapter": bool(is_new_adapter),
            "sampled_layers": int(sample_layers or 0),
        }
        return progress_metrics

    class _AdapterReportLoggerProxy(logging.Logger):
        """Routes log messages to the primary logger and optionally collects them."""

        def __init__(self, base_logger: logging.Logger, collector: Optional[List[str]] = None):
            self._base_logger = base_logger
            self._collector = collector

        def _record(self, level: str, message: Any, *args, **kwargs):
            log_method = getattr(self._base_logger, level)
            log_method(message, *args, **kwargs)
            if self._collector is None:
                return
            try:
                text = str(message)
            except Exception:
                text = repr(message)
            if args:
                try:
                    text = text % args
                except Exception:
                    text = f"{text} {args}"
            prefix = "" if level == "info" else f"{level.upper()}: "
            self._collector.append(f"{prefix}{text}")

        def info(self, message, *args, **kwargs):
            self._record("info", message, *args, **kwargs)

        def warning(self, message, *args, **kwargs):
            self._record("warning", message, *args, **kwargs)

        def error(self, message, *args, **kwargs):
            self._record("error", message, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._base_logger, name)

    def _sanitize_metrics_structure(obj: Any):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {str(k): _sanitize_metrics_structure(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize_metrics_structure(v) for v in obj]
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, float)):
            return float(obj)
        return str(obj)

    METRIC_DELTA_LABELS = {
        "progress.ASI": "ASI (%)",
        "progress.remaining_capacity_pct": "Remaining Capacity (%)",
        "progress.avg_grad_norm": "Avg Grad Norm",
        "progress.delta_norm": "Delta Norm",
        "behavior.attn_pct": "Attention Share (%)",
        "behavior.mlp_pct": "MLP Share (%)",
        "behavior.free_capacity_proxy": "Free Capacity Proxy",
        "snapshot.norms.norm_proxy": "Norm Proxy",
    }

    def _flatten_metric_dict(data: Optional[Dict[str, Any]], prefix: str = "") -> Dict[str, float]:
        if not isinstance(data, dict):
            return {}
        flat: Dict[str, float] = {}
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(_flatten_metric_dict(value, new_prefix))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[new_prefix] = float(value)
        return flat

    def _format_metric_delta_lines(
        initial_metrics: Optional[Dict[str, Any]],
        final_metrics: Optional[Dict[str, Any]],
    ) -> tuple[List[str], Dict[str, Dict[str, float]]]:
        init_flat = _flatten_metric_dict(initial_metrics)
        final_flat = _flatten_metric_dict(final_metrics)
        lines: List[str] = []
        delta_payload: Dict[str, Dict[str, float]] = {}
        for key, label in METRIC_DELTA_LABELS.items():
            if key not in init_flat or key not in final_flat:
                continue
            start = init_flat[key]
            end = final_flat[key]
            delta = end - start
            delta_payload[label] = {"start": start, "end": end, "delta": delta}
            if abs(delta) < 1e-6:
                continue
            lines.append(f"{label}: {start:.4f} -> {end:.4f} ({delta:+.4f})")
        return lines, delta_payload

    def _publish_adapter_report(
        stage: str,
        step_value: Optional[Union[int, float, str]],
        lines: List[str],
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Stores adapter diagnostics on the engine state for downstream consumers."""
        payload = {
            "stage": stage,
            "step": step_value,
            "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "lines": list(lines),
        }
        if metrics is not None:
            payload["metrics"] = _sanitize_metrics_structure(metrics)
        try:
            asyncio.run_coroutine_threadsafe(
                engine.state.set_adapter_report(stage, payload),
                loop,
            ).result()
        except Exception as exc:
            engine.state.logger.warning(
                "Failed to publish '%s' adapter report: %s", stage, exc
            )

    initial_adapter_metrics: Optional[Dict[str, Any]] = None
    final_adapter_metrics: Optional[Dict[str, Any]] = None


    def blocking_training_logic(passed_callbacks): # Accept callbacks as an argument
        nonlocal initial_adapter_metrics, final_adapter_metrics
        trainer = None
        trainer_temp_output_dir = None
        run_start_time_wall = time.time()
        mixed = engine.state._peft_model           # compiled PeftMixedModel (inference wrapper)
        if mixed is None or getattr(mixed, "base_model", None) is None:
            raise TrainingError("Mixed wrapper not initialized properly.")

        peft_owner = getattr(mixed, "_orig_mod", mixed)  # uncompiled PEFT module that owns adapters

        # This is the HF module that actually has the PEFT layers installed.
        train_core = peft_owner.base_model.model        # your existing inner core

        # train on the real HF base module that has LoRA modules injected
        # IMPORTANT: still train the uncompiled *inner* module for speed/stability,
        # but expose PEFT metadata via peft_host so Trainer sees adapters:
        model_for_training = _TrainerModelShim(train_core, peft_host=peft_owner)

        if model_for_training is None:
            raise TrainingError("Base model is not initialized.")

        engine.state.logger.info(f"Using model ref (type: {type(model_for_training).__name__}) for training.")

        #        if not isinstance(model_for_training, PreTrainedModel):
        #            raise TrainingError(f"Unexpected training model type: {type(model_for_training).__name__}")

        safe_params_for_err: Dict[str, Any] = {}
        effective_max_seq_len_for_err: Optional[int] = None
        initial_max_seq_len: Optional[int] = None
        try:
            engine.state.logger.info("Loading and preprocessing dataset...")
            # Use datasets.load_dataset with 'json' loader for local files
            dataset_path = config.dataset.dataset_path
            if dataset_path.lower().endswith('.json'):
                dataset = load_dataset('json', data_files=dataset_path, split=config.dataset.split)
            else:
                dataset = load_dataset(dataset_path, split=config.dataset.split)
            engine.state.logger.info(f"Loaded {len(dataset)} samples for training.")

            sft_formatting_func = get_sft_formatting_func(
                engine.state.logger,
                formatting=config.dataset.formatting,
                columns=config.dataset.columns,
                tokenizer=engine.state.tokenizer,
                tags=config.dataset.tags
            )
            if sft_formatting_func is None:
                raise ConfigurationError("SFT formatting function could not be determined. Check your config.")

            initial_max_seq_len = config.max_sequence_length
            if not initial_max_seq_len: # If None or 0 from config
                if engine.state.tokenizer and hasattr(engine.state.tokenizer, 'model_max_length'):
                    initial_max_seq_len = engine.state.tokenizer.model_max_length
                else:
                    raise TrainingError("Tokenizer or tokenizer.model_max_length not available for determining sequence length.")
            if engine.state.tokenizer and hasattr(engine.state.tokenizer, 'model_max_length'):
                initial_max_seq_len = min(initial_max_seq_len, engine.state.tokenizer.model_max_length)

            # --- Determine safe training parameters using heuristics ---
            safe_params, final_effective_seq_len, resource_report = _determine_safe_training_params(
                logger=engine.state.logger,
                engine=engine,
                training_config=config,
                effective_max_seq_len=initial_max_seq_len,
                dataset_size=len(dataset)
            )
            safe_params_for_err = dict(safe_params)
            effective_max_seq_len_for_err = final_effective_seq_len
            effective_batch_size_config = safe_params["per_device_train_batch_size"]
            effective_grad_accum_config = safe_params["gradient_accumulation_steps"]
            effective_grad_checkpointing = safe_params["gradient_checkpointing"]
            effective_precision = safe_params["trainer_compute_precision"]
        
            use_fp16 = effective_precision == "fp16"
            use_bf16 = effective_precision == "bf16"
    
            # --- Communicate settings to client ---
            summary_lines = [
                "--- Effective Training Settings ---",
                f"  - Context Length: {final_effective_seq_len}",
                f"  - Device Batch Size: {effective_batch_size_config}",
                f"  - Gradient Accumulation: {effective_grad_accum_config}",
                f"  - >> Effective Batch Size: {effective_batch_size_config * effective_grad_accum_config}",
                f"  - Gradient Checkpointing: {'On' if effective_grad_checkpointing else 'Off'}",
                f"  - Compute Precision: {effective_precision}",
                f"  - Auto-Managed Memory: {'On' if config.auto_manage_memory else 'Off'}",
                f"  - Samples: {len(dataset)}",
            ]
            summary_str = "\n".join(summary_lines)
            engine.state.logger.info(summary_str)
            param_src = resource_report.get("param_sources", {})
            if param_src:
                src_str = ", ".join(f"{k}={v}" for k, v in param_src.items())
                engine.state.logger.info(f"Parameter sources (user|auto|heuristic): {src_str}")
            asyncio.run_coroutine_threadsafe(engine.state.set_heuristic_summary(summary_str), loop).result()
            effective_config_report = {
                "context_length": final_effective_seq_len,
                "per_device_batch_size": effective_batch_size_config,
                "gradient_accumulation": effective_grad_accum_config,
                "effective_batch_size": effective_batch_size_config * effective_grad_accum_config,
                "gradient_checkpointing": bool(effective_grad_checkpointing),
                "precision": effective_precision,
                "auto_manage_memory": bool(config.auto_manage_memory),
                "dataset_size": len(dataset),
                "sources": resource_report.get("param_sources", {}),
            }
            asyncio.run_coroutine_threadsafe(engine.state.set_training_resource_report(resource_report), loop).result()
            asyncio.run_coroutine_threadsafe(engine.state.set_effective_training_config(effective_config_report), loop).result()
            # --- End of parameter determination ---
    
            engine.state.logger.info(f"Effective max_sequence_length for tokenization: {final_effective_seq_len}")
            engine.state.logger.info("Tokenizing dataset for SFT (Supervised Fine-Tuning)...")
    
            tokenized_dataset = dataset.map(
                map_preprocess_sft, # Use top-level helper to avoid hashing warnings
                batched=True,
                remove_columns=dataset.column_names,
                fn_kwargs={
                    "tokenizer_ref": engine.state.tokenizer,
                    "formatting_func_ref": sft_formatting_func,
                    "max_seq_len_ref": final_effective_seq_len, # Use the final capped length
                    "proc_mode_ref": config.dataset.preprocessing_mode
                }
            )
            engine.state.logger.info(f"Dataset tokenization complete. Original size: {len(tokenized_dataset)}")
    
            # --- Filter out oversized samples, aligning with train_lora.py ---
            bad_rows = tokenized_dataset.filter(lambda x: not x["keep"])
            if len(bad_rows) > 0:
                engine.state.logger.warning(f"WARNING: {len(bad_rows)} samples exceed max_sequence_length ({final_effective_seq_len}) and have been removed.")
                tokenized_dataset = tokenized_dataset.filter(lambda x: x["keep"])
    
            # Drop helper columns that some models (e.g., Phi-3) don't accept in forward()
            drop_cols = [c for c in ("length", "keep") if c in tokenized_dataset.column_names]
            if drop_cols:
                engine.state.logger.info(f"Removing helper columns to avoid unexpected kwargs in model forward: {drop_cols}")
                tokenized_dataset = tokenized_dataset.remove_columns(drop_cols)
    
            asyncio.run_coroutine_threadsafe(engine.state.set_dataset_loaded_for_training(), loop).result()
            
            engine.state.logger.info("Setting up the Trainer...")
    
            # --- Calculate max_steps and logging_steps  ---
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
            # Use the *effective* batch size for this calculation
            eff_bs_calc = effective_batch_size_config * effective_grad_accum_config * num_devices
            steps_per_epoch = len(tokenized_dataset) // eff_bs_calc if eff_bs_calc > 0 else 1
            
            if config.max_steps is not None and config.max_steps > 0:
                final_max_steps = config.max_steps
            else:
                final_max_steps = int(steps_per_epoch * config.num_train_epochs)
            if final_max_steps <= 0: final_max_steps = 1
            
            engine.state._total_steps = final_max_steps # Set total_steps in state early
    
            effective_logging_steps = config.logging_steps
            if not (effective_logging_steps > 0 and effective_logging_steps <= final_max_steps) :
                effective_logging_steps = max(1, final_max_steps // 10)
            engine.state.logger.info(f"Calculated effective max_steps: {final_max_steps}, logging_steps: {effective_logging_steps}")
            # --- End of calculation ---
    
            # --- Trainer output directory  ---
            # Trainer saves its checkpoints to a temporary subdirectory
            trainer_temp_output_dir = training_run_output_dir / f"__trainer_temp_{int(run_start_time_wall)}"
            if trainer_temp_output_dir.exists():
                shutil.rmtree(trainer_temp_output_dir)
            trainer_temp_output_dir.mkdir(parents=True, exist_ok=True)
            engine.state.logger.info(f"Trainer internal output_dir for its checkpoints: {trainer_temp_output_dir}")
            # --- End of output directory logic ---
    
    
            # Push Pad token to it
            if engine.state.tokenizer.pad_token_id is None:
                engine.state.logger.warning("Engine tokenizer does not have PAD token configured")
            else:
                model_for_training.config.pad_token_id = engine.state.tokenizer.pad_token_id

            # --- Prepare TrainingArguments ---
            tensorboard_log_dir = trainer_temp_output_dir / "logs"
            if "TENSORBOARD_LOGGING_DIR" not in os.environ:
                os.environ["TENSORBOARD_LOGGING_DIR"] = str(tensorboard_log_dir)
            training_args_dict = {
                "output_dir": str(trainer_temp_output_dir),
                "num_train_epochs": config.num_train_epochs,
                "per_device_train_batch_size": effective_batch_size_config,
                "gradient_accumulation_steps": effective_grad_accum_config,
                "learning_rate": config.learning_rate,
                "max_steps": final_max_steps,
                "lr_scheduler_type": config.lr_scheduler_type,
                "warmup_steps": config.warmup_steps,
                "optim": config.optim,
                "max_grad_norm": config.max_grad_norm,
                "seed": config.seed,
                "save_strategy": "no",
                "save_steps": final_max_steps,
                "save_total_limit": 0,
                "logging_steps": effective_logging_steps,
                "logging_first_step": False,
                "disable_tqdm": config.disable_tqdm if hasattr(config, 'disable_tqdm') else True,
                "log_level": "warning",
                "report_to": config.report_to,
                "gradient_checkpointing": effective_grad_checkpointing,
                "fp16": use_fp16,
                "bf16": use_bf16,
                "dataloader_num_workers": config.dataloader_num_workers,
                "run_name": f"train-{adapter_to_train}",
                "label_smoothing_factor": config.label_smoothing_factor, "resume_from_checkpoint": config.resume_from_checkpoint,
                "gradient_checkpointing_kwargs": {"use_reentrant": False} if effective_grad_checkpointing else None,
                "label_names": ["labels"], # Hardcode use_reentrant to False (modern implementation)
            }

            # Drop unsupported TrainingArguments keys for older transformers installs.
            supported_training_args = set(inspect.signature(TrainingArguments.__init__).parameters)
            supported_training_args.discard("self")
            unsupported_training_args = [k for k in training_args_dict if k not in supported_training_args]
            for key in unsupported_training_args:
                training_args_dict.pop(key, None)
            if unsupported_training_args:
                engine.state.logger.warning(
                    "Removed unsupported TrainingArguments keys: %s",
                    ", ".join(sorted(unsupported_training_args)),
                )
    
            # TBD: we do not use compiled for training, can we pass True here? 
            training_args_dict["remove_unused_columns"] = False
    
            # --- Activate Adapter and Set Mode just before creating Trainer ---
            # This ensures the model is in the correct state right before Trainer takes over.
            
            peft_owner.enable_adapter_layers()
    
            # The adapter definition should already be on the model from the `load-adapter` API call.
            if adapter_to_train not in peft_owner.peft_config:
                # This should not happen if the API flow is correct.
                raise AdapterError(f"Adapter '{adapter_to_train}' definition not found on the model before training.")
    
            # Activate ONLY the adapter we want to train.
            # PEFT's `set_adapter` handles setting `requires_grad` correctly.
            engine.state.logger.info(f"Activating adapter '{adapter_to_train}' for training.")
            peft_owner.set_adapter(adapter_to_train)
    
            # Set the model to training mode.
            model_for_training.train()
    
            engine.state.logger.info("Trainable parameters on training model (after set_adapter):")
            initial_report_lines: List[str] = []
            report_logger = _AdapterReportLoggerProxy(engine.state.logger, initial_report_lines)
            snapshot_metrics = _log_adapter_snapshot(report_logger, train_core)  # Log parameters now that requires_grad is set.
            behavior_metrics = _log_active_adapter_shapes_verbose(report_logger, train_core)
            progress_metrics = _log_adapter_progress(report_logger, train_core, step="pre", sample_layers=8)
            initial_adapter_metrics = {
                "snapshot": snapshot_metrics,
                "behavior": behavior_metrics,
                "progress": progress_metrics,
            }
            initial_adapter_metrics.setdefault("progress", {})["avg_grad_norm"] = 0.0
            _publish_adapter_report("initial", 0, initial_report_lines, metrics=initial_adapter_metrics)
    
            training_args = TrainingArguments(**training_args_dict)
    
            # --- Prevent DataParallel wrapper for single-device models ---
            # The Trainer automatically wraps the model in nn.DataParallel if n_gpu > 1.
            # This is inefficient and causes OOM if the model was already loaded on a single specific device.
            is_sharded = hasattr(model_for_training, "is_parallelizable") and model_for_training.is_parallelizable
            if not is_sharded and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                engine.state.logger.info("Model is on a single device; forcing Trainer's n_gpu=1 to prevent DataParallel wrapping.")
                training_args._n_gpu = 1
    
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=engine.state.tokenizer,
                mlm=False, # For SFT, MLM is False
                pad_to_multiple_of=8,   # optional, good for speed on Ampere/Ada
            )
            model_type = getattr(getattr(model_for_training, "config", None), "model_type", None)
            if model_type == "gemma3":
                data_collator = _TokenTypeIdsCollator(data_collator, require_for_model_type=model_type)
    
            trainer = QuietTrainer( # Use QuietTrainer
                model=model_for_training,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_dataset,
                callbacks=passed_callbacks # Use the passed callbacks
            )
            engine.state._trainer_instance = trainer 
            engine.state._trainer_initialized_for_training = True # Directly set the flag
    
            engine.state.logger.info("Trainer setup complete.")
            asyncio.run_coroutine_threadsafe(engine.state.set_training_started(), loop).result() # Notify that training loop is starting
            engine.state.logger.info("Starting the training process...")
            train_result = trainer.train()
            engine.state.logger.info(f"Training completed. Total steps: {trainer.state.global_step}.")
    
            final_step_value = getattr(trainer.state, "global_step", None)
            final_report_lines: List[str] = []
            post_report_logger = _AdapterReportLoggerProxy(engine.state.logger, final_report_lines)
            snapshot_metrics_final = _log_adapter_snapshot(post_report_logger, train_core)
            behavior_metrics_final = _log_active_adapter_shapes_verbose(post_report_logger, train_core)
            progress_metrics_final = _log_adapter_progress(
                post_report_logger,
                train_core,
                step=final_step_value if final_step_value is not None else "post",
                sample_layers=8,
            )
            final_adapter_metrics = {
                "snapshot": snapshot_metrics_final,
                "behavior": behavior_metrics_final,
                "progress": progress_metrics_final,
            }
            avg_grad_norm_observed = None
            max_grad_norm_observed = None
            if grad_monitor_callback:
                avg_grad_norm_observed = grad_monitor_callback.avg_grad_norm
                max_grad_norm_observed = grad_monitor_callback.max_grad_norm
            if avg_grad_norm_observed is not None:
                final_adapter_metrics.setdefault("progress", {})["avg_grad_norm"] = avg_grad_norm_observed
                final_report_lines.append(
                    f"[PEFT] Avg LoRA grad norm during training (sampled): {avg_grad_norm_observed:.4f}"
                )
            if max_grad_norm_observed is not None:
                final_adapter_metrics.setdefault("progress", {})["max_grad_norm"] = max_grad_norm_observed
            _publish_adapter_report("final", final_step_value, final_report_lines, metrics=final_adapter_metrics)
    
            delta_lines, delta_metrics_payload = _format_metric_delta_lines(
                initial_adapter_metrics, final_adapter_metrics
            )
            if not delta_lines:
                delta_lines = ["No numeric adapter metric deltas detected."]
            _publish_adapter_report(
                "delta",
                final_step_value,
                delta_lines,
                metrics={"delta": delta_metrics_payload} if delta_metrics_payload else None,
            )
    
            grads_seen = False
            grad_steps: List[int] = []
            avg_grad_norm_observed = None
            max_grad_norm_observed = None
            if grad_monitor_callback:
                grads_seen = grad_monitor_callback.observed_gradients
                grad_steps = grad_monitor_callback.steps_with_gradients
                avg_grad_norm_observed = grad_monitor_callback.avg_grad_norm
                max_grad_norm_observed = grad_monitor_callback.max_grad_norm
            lora_weight_change_detected = False
            delta_norm_change = None
            if not grads_seen and initial_adapter_metrics and final_adapter_metrics:
                try:
                    init_progress = initial_adapter_metrics.get("progress", {}) if isinstance(initial_adapter_metrics, dict) else {}
                    final_progress = final_adapter_metrics.get("progress", {}) if isinstance(final_adapter_metrics, dict) else {}
                    init_dn = init_progress.get("delta_norm")
                    final_dn = final_progress.get("delta_norm")
                    if isinstance(init_dn, (int, float)) and isinstance(final_dn, (int, float)):
                        delta_norm_change = abs(float(final_dn) - float(init_dn))
                        lora_weight_change_detected = delta_norm_change > 1e-6
                except Exception:
                    lora_weight_change_detected = False
                    delta_norm_change = None
            progress_warning_emitted = False
            if initial_adapter_metrics and final_adapter_metrics:
                try:
                    init_progress = initial_adapter_metrics.get("progress", {}) if isinstance(initial_adapter_metrics, dict) else {}
                    final_progress = final_adapter_metrics.get("progress", {}) if isinstance(final_adapter_metrics, dict) else {}
                    init_dn = init_progress.get("delta_norm")
                    final_dn = final_progress.get("delta_norm")
                    init_np = init_progress.get("norm_proxy")
                    final_np = final_progress.get("norm_proxy")
                    if isinstance(init_dn, (int, float)) and isinstance(final_dn, (int, float)) and isinstance(init_np, (int, float)) and isinstance(final_np, (int, float)):
                        delta_norm_delta = abs(float(final_dn) - float(init_dn))
                        norm_proxy_delta = abs(float(final_np) - float(init_np))
                        if delta_norm_delta < 1e-4 and norm_proxy_delta < 1.0:
                            loss_delta = None
                            loss_delta_pct = None
                            try:
                                losses = list(engine.state.historical_loss) if engine and engine.state and engine.state.historical_loss else []
                                if len(losses) >= 2 and isinstance(losses[0], (int, float)) and isinstance(losses[-1], (int, float)):
                                    initial_loss = float(losses[0])
                                    final_loss = float(losses[-1])
                                    loss_delta = final_loss - initial_loss
                                    if abs(initial_loss) > 1e-9:
                                        loss_delta_pct = (loss_delta / initial_loss) * 100.0
                            except Exception:
                                loss_delta = None
                                loss_delta_pct = None
                            progress_warning_emitted = True
                            engine.state.logger.warning(
                                "Adapter progress delta below threshold (delta_norm_delta=%.6f, norm_proxy_delta=%.4f%s%s). "
                                "Run may be too short, data may be redundant, or learning rate too low.",
                                delta_norm_delta,
                                norm_proxy_delta,
                                f", loss_delta={loss_delta:.6f}" if loss_delta is not None else "",
                                f", loss_delta_pct={loss_delta_pct:.2f}%" if loss_delta_pct is not None else "",
                            )
                except Exception:
                    progress_warning_emitted = False
            if not grads_seen and lora_weight_change_detected:
                grads_seen = True
            try:
                asyncio.run_coroutine_threadsafe(
                    engine.state.set_lora_grads_observed(grads_seen),
                    loop,
                ).result()
            except Exception as exc:
                engine.state.logger.warning(
                    "Failed to persist LoRA grad observation flag: %s", exc
                )
    
            if grads_seen:
                first_step = grad_steps[0] if grad_steps else "unknown"
                grad_norm_msg = ""
                if avg_grad_norm_observed is not None:
                    grad_norm_msg = f" avg_grad_norm~{avg_grad_norm_observed:.4f}"
                if max_grad_norm_observed is not None:
                    grad_norm_msg += f" max_grad_norm~{max_grad_norm_observed:.4f}"
                if delta_norm_change is not None:
                    grad_norm_msg += f" delta_norm_delta~{delta_norm_change:.6f}"
                engine.state.logger.info(
                    "Preflight: observed non-null gradients on LoRA tensors (first noted at step %s).%s",
                    first_step,
                    grad_norm_msg,
                )
            else:
                engine.state.logger.warning(
                    "Preflight WARNING: did not observe gradients on LoRA tensors; verify adapter path."
                )
            
            # If the training was cancelled, the thread should exit immediately after trainer.train() returns.
            # This prevents it from trying to save models or generate reports when the main app is shutting down.
            if engine.state._training_cancelled:
                engine.state.logger.info("Training was cancelled. Skipping post-training save and report generation.")
                return # Exit the thread cleanly
    
            # --- Adapter-only save (no full base model) ---
            final_saved_checkpoint_path = None
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            final_step_for_name = trainer.state.global_step if hasattr(trainer.state, 'global_step') and trainer.state.global_step > 0 else engine.state._total_steps
            target_final_checkpoint_dir = training_run_output_dir / f"{timestamp_str}-step{final_step_for_name}"
            if adapter_was_foreign and adapter_loaded_checkpoint_path:
                try:
                    prior_cp = Path(adapter_loaded_checkpoint_path).resolve()
                    current_root = Path(training_run_output_dir).resolve()
                    if prior_cp == current_root:
                        engine.state.logger.info(f"Foreign adapter weights lived at root '{prior_cp}'. Saving new checkpoint under '{target_final_checkpoint_dir}' to attach metadata and clear foreign status.")
                    else:
                        engine.state.logger.info(f"Foreign adapter checkpoint was '{prior_cp}'. New checkpoint will be created under '{target_final_checkpoint_dir}'.")
                except Exception:
                    engine.state.logger.info(f"Foreign adapter detected; creating dedicated checkpoint folder '{target_final_checkpoint_dir}' for metadata.")
    
            if target_final_checkpoint_dir.exists():
                shutil.rmtree(target_final_checkpoint_dir)
            target_final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
            # Get the unwrapped model for consistent metadata access
            unwrapped_inference_model = getattr(engine.state._peft_model, "_orig_mod", engine.state._peft_model)
    
            mixed = engine.state._peft_model
            ckpt_dir = save_adapter_package_for_mixed(
                mixed_model=unwrapped_inference_model,
                adapter_name=adapter_to_train,
                adapter_root_dir=str(training_run_output_dir),
                step=trainer.state.global_step or 0, # The model_for_training shim was a proxy to the inner model, which is part of engine.state._peft_model.
                # Saving from the top-level PeftMixedModel ensures keys match what `load_adapter` expects.
                base_model_name_or_path=engine.state.base_model_name_or_path,
                # Optionally pass extra metadata dicts:
                extra_root_meta={
                    "precision": str(getattr(unwrapped_inference_model.config, "dtype", "unknown")),
                    "max_context": getattr(unwrapped_inference_model.config, "max_position_embeddings", None),
                },
                extra_ckpt_meta={"final_loss": float(train_result.metrics.get("loss", 0.0)) if train_result else None, "learning_rate": training_args.learning_rate,},
            )
            final_saved_checkpoint_path = str(ckpt_dir)
            engine.state.logger.info(f"Adapter-only files saved to: {final_saved_checkpoint_path}")
    
            # With the new in-place training, the weights are already in the model.
            # No need to load them back from the saved checkpoint.
            engine.state.logger.info(f"Training was performed in-place on the main model. Adapter '{adapter_to_train}' is already updated.")
    
            if final_saved_checkpoint_path:
                asyncio.run_coroutine_threadsafe(engine.state.set_final_adapter_saved(final_saved_checkpoint_path), loop).result()
                
            # --- (Optional) 1-batch preflight to assert LoRA grads were active during training ---
            # --- Generate reports and metadata in the training_run_output_dir ---
            engine.state.logger.info(f"Generating training report in '{training_run_output_dir}'...")
            generate_training_report(
                engine.state.logger,
                str(training_run_output_dir),
                list(engine.state.historical_steps),
                list(engine.state.historical_loss),
                list(engine.state.historical_lr),
                list(engine.state.historical_grad_norm)
            )
            
            adapter_peft_config_on_model = model_for_training.peft_config.get(adapter_to_train) # type: ignore
            lora_config_dict_for_meta = {}
            if adapter_peft_config_on_model and isinstance(adapter_peft_config_on_model, (LoraConfig)):
                lora_config_dict_for_meta = {
                    "r": getattr(adapter_peft_config_on_model, 'r', None),
                    "lora_alpha": getattr(adapter_peft_config_on_model, 'lora_alpha', None),
                    "lora_dropout": getattr(adapter_peft_config_on_model, 'lora_dropout', None),
                    "target_modules": getattr(adapter_peft_config_on_model, 'target_modules', None),
                }
                # Normalize target_modules to a JSON-friendly list and ensure it is present if the model has it.
                tm = lora_config_dict_for_meta.get("target_modules")
                if tm:
                    if isinstance(tm, set):
                        tm = sorted(list(tm))
                    elif isinstance(tm, tuple):
                        tm = list(tm)
                    lora_config_dict_for_meta["target_modules"] = tm
                elif hasattr(adapter_peft_config_on_model, "target_modules") and adapter_peft_config_on_model.target_modules:
                    tm = adapter_peft_config_on_model.target_modules
                    if isinstance(tm, set):
                        tm = sorted(list(tm))
                    elif isinstance(tm, tuple):
                        tm = list(tm)
                    lora_config_dict_for_meta["target_modules"] = tm
            
            precision_info_for_meta = {
                "adapter_training_compute_precision": config.trainer_compute_precision,
                "base_model_requested_torch_dtype": engine.state.requested_torch_dtype,
                # Create a clean, canonical representation of the quantization config for storage in metadata.json.
                "base_model_quantization_config": create_quantization_metadata(engine.state.base_model_quantization_cfg or engine.state.quantization_config or {}),
                "base_model_effective_dtype_at_init": str(engine.state.base_model.dtype) if engine.state.base_model else None
            }
            save_adapter_metadata(
                str(training_run_output_dir),
                adapter_to_train,
                adapter_type if adapter_type else "UNKNOWN",
                engine.state.base_model_name_or_path,
                training_config=lora_config_dict_for_meta,
                precision_info=precision_info_for_meta
            )
            engine.state.logger.info(f"Training reports and metadata generated in '{training_run_output_dir}'.")
    
            # --- Copy reports and metadata to checkpoint and root adapter folder ---
            if final_saved_checkpoint_path and Path(final_saved_checkpoint_path).is_dir():
                engine.state.logger.info(f"Copying reports/metadata from {training_run_output_dir} to checkpoint dir {final_saved_checkpoint_path}")
                asyncio.run_coroutine_threadsafe(copy_report_and_metadata_files(engine.state.logger, Path(training_run_output_dir), Path(final_saved_checkpoint_path)), loop).result()
    
            adapter_root_path_for_reports_meta_str = asyncio.run_coroutine_threadsafe(engine.state.get_adapter_root_path(adapter_to_train), loop).result()
            if adapter_root_path_for_reports_meta_str:
                adapter_root_for_reports_meta = Path(adapter_root_path_for_reports_meta_str)
                if adapter_root_for_reports_meta.is_dir() and adapter_root_for_reports_meta != Path(training_run_output_dir):
                    engine.state.logger.info(f"Copying latest reports/metadata from {training_run_output_dir} to adapter root {adapter_root_for_reports_meta}")
                    asyncio.run_coroutine_threadsafe(copy_report_and_metadata_files(engine.state.logger, Path(training_run_output_dir), adapter_root_for_reports_meta), loop).result()
                elif adapter_root_for_reports_meta == Path(training_run_output_dir):
                    engine.state.logger.info(f"Adapter root {adapter_root_for_reports_meta} is the same as current run output dir. Reports/metadata already there.")
            else:
                adapter_root_for_reports_meta = Path(training_run_output_dir)
    
            # Refresh loaded adapter info to reflect new checkpoint, config, quant, and ensure non-foreign.
            meta = {}
            try:
                meta_path = Path(training_run_output_dir) / "metadata.json"
                if meta_path.is_file():
                    meta = json.load(open(meta_path, "r"))
            except Exception:
                meta = {}
            quant_display = quant_display_from_meta(meta)
            adapter_root_path_for_add = str(adapter_root_for_reports_meta if adapter_root_for_reports_meta.is_dir() else Path(training_run_output_dir))
            cp_name = Path(final_saved_checkpoint_path).name if final_saved_checkpoint_path else None
            asyncio.run_coroutine_threadsafe(engine.state.add_loaded_adapter_info(
                adapter_name=adapter_to_train,
                adapter_root_path=adapter_root_path_for_add,
                adapter_type=adapter_type if adapter_type else "LORA",
                checkpoint_path=final_saved_checkpoint_path,
                adapter_config_dump=engine.state.current_training_config,
                base_model_quant=quant_display,
                base_model_name=Path(engine.state.base_model_name_or_path).name if engine.state.base_model_name_or_path else None,
                alias=cp_name or Path(adapter_root_path_for_add).name,
                metadata=meta or None,
                is_foreign=False
            ), loop).result()
            # --- End of copying logic ---
            
            # Original: trainer.save_model(str(training_run_output_dir))
            # Original: asyncio.run_coroutine_threadsafe(engine.state.set_final_adapter_saved(str(training_run_output_dir)), loop).result()

            engine.state.logger.info(f"Model saved successfully for adapter '{adapter_to_train}'.")
            fut_status = asyncio.run_coroutine_threadsafe(
                engine.state.set_training_status(TrainingStatus.COMPLETED, "Training completed successfully."),
                loop
            )
            fut_status.result()
            engine.state.logger.info(f"--- Training run completed successfully for adapter '{adapter_to_train}'. Total Time: {time.time() - run_start_time_wall:.2f}s ---")
        except Exception as e:
            error_msg = f"Error during training run: {type(e).__name__}: {e}"

            # Check for CUDA OOM and provide helpful suggestions
            if 'CUDA out of memory' in str(e):
                # The 'safe_params' dict is available here, containing the params that were actually used
                error_msg = _create_oom_suggestion_message(
                    config=config,
                    params_used=safe_params_for_err,
                    effective_max_seq_len=effective_max_seq_len_for_err or 0
                )

            engine.state.logger.error(f"!!! {error_msg}\n{traceback.format_exc()}")
            fut_err = asyncio.run_coroutine_threadsafe(
                engine.state.set_training_error(error_msg),
                loop
            )
            fut_err.result()
            raise # Re-raise the exception to propagate it
        finally:
            # This block ensures the temporary directory is always cleaned up.
            if trainer_temp_output_dir and trainer_temp_output_dir.exists():
                shutil.rmtree(trainer_temp_output_dir, ignore_errors=True)
                engine.state.logger.info(f"Cleaned up temporary trainer directory: {trainer_temp_output_dir}")
    
        # --- Final Cleanup for the training round ---
        # Release references to allow garbage collection
        if 'trainer' in locals() and trainer is not None:
            del trainer
        if 'model_for_training' in locals() and model_for_training is not None:
            del model_for_training
        # Explicitly clear the GPU cache to release memory from optimizer, gradients, etc.
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        engine.state.logger.info("Training round cleanup: Released trainer/model references and cleared CUDA cache.")
    
    try:
        await asyncio.to_thread(blocking_training_logic, callbacks_for_trainer)
    except asyncio.CancelledError:
        # This block now runs when the task is cancelled by the engine.
        # The thread running blocking_training_logic is now "orphaned" but
        # will check the _training_cancelled flag and exit cleanly if it hasn't finished.
        engine.state.logger.info(f"Training task for adapter '{adapter_to_train}' was cancelled by engine request.")
        # CRITICAL: Only set an error if the training wasn't already completed.
        # This prevents a race condition where a mode switch after successful completion
        # cancels this task wrapper and incorrectly flags an error.
        if engine.state.training_status not in [TrainingStatus.COMPLETED, TrainingStatus.STOPPED, TrainingStatus.ERROR, TrainingStatus.OFFLINE]:
            await engine.state.set_training_error("Training run was forcefully cancelled.")
        # Do not re-raise CancelledError, just let the coroutine finish.
    except Exception as e:
        error_msg = f"Error during training execution task: {type(e).__name__}: {e}"
        engine.state.logger.critical(f"!!! {error_msg}\n{traceback.format_exc()}")
        if engine.state.training_status != TrainingStatus.ERROR:
            await engine.state.set_training_error(error_msg)
    finally:
        # Ensure the task reference is cleared on any exit path (success, error, or cancellation)
        # to prevent a mode switch from trying to cancel a completed task.
        if engine.state._training_task is not None:
            engine.state._training_task = None
