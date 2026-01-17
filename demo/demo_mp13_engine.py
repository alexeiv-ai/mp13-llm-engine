"""
Main workflow script for the test adapter demonstration using the MP13 Engine API.
"""

import os
import sys
import json
import shutil
import codecs
import signal
import asyncio
import argparse
import datetime
import logging
import tempfile
import time
import traceback
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Union, Optional, Tuple, AsyncIterator

# --- Fix for UnicodeEncodeError on Windows ---
# Reconfigure stdout/stderr to use UTF-8 encoding if they don't already.
if sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (TypeError, AttributeError):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_root = os.path.join(project_root, "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from mp13_engine import logger
from mp13_engine.mp13_engine_api import handle_call_tool, get_engine_instance_for_direct_use
from mp13_engine.mp13_config_paths import (
    get_default_config_dir,
    get_default_config_path,
    load_json_config,
    load_effective_config,
    load_merged_config,
    resolve_config_paths,
    resolve_custom_config_path,
    resolve_engine_inputs,
)
from mp13_engine.mp13_config import (
    APIStatus, GlobalEngineConfig, TrainingConfig, InferenceRequest, AdapterConfig, AdapterType, ChunkType, MP13Response,
    DatasetFormat, DatasetConfig, ColumnsConfig, DatasetTags, PreprocessingMode, EngineMode
)
from mp13_engine.mp13_state import TrainingStatus

# Global flag to ensure shutdown logic runs only once
_shutdown_initiated = False

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

async def call_api(
    tool_name: str, 
    arguments: Optional[Dict[str, Any]] = None
) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
    if arguments is None:
        arguments = {}

    response_from_api = await handle_call_tool(tool_name, arguments)

    # All API calls now return an MP13Response object.
    if isinstance(response_from_api, MP13Response):
        # Handle streaming responses for 'run-inference'
        if response_from_api.stream:
            # The stream now yields InferenceResponse objects. We create an async generator to convert them to dicts.
            async def chunk_iterator():
                async for item in response_from_api.stream:
                    try:
                        yield item.model_dump(exclude_none=True)
                    except Exception as e:
                        yield {"chunkType": ChunkType.ERROR.value, "error": f"Client-side model_dump error: {e}", "details": {"raw_response": str(item)}}
            return {"stream": chunk_iterator()}

        # Handle synchronous responses
        # Convert the Pydantic model to a dictionary for consistent handling in the demo script
        response_dict = response_from_api.model_dump(exclude_none=True)
        if response_dict.get("status") == "error":
            print(f"[DEMO] API call '{tool_name}' completed with status '{response_dict.get('status')}'. Message: {response_dict.get('message', 'N/A')}")
        return response_dict

    # Fallback for any other unexpected type
    error_msg = f"API call '{tool_name}' returned an unexpected type: {type(response_from_api).__name__}"
    print(f"[DEMO] Error: {error_msg}")
    return {"status": "error", "message": error_msg, "details": {"raw_response": str(response_from_api)}}


def _precision_suffix_from_global_config(global_cfg: Dict[str, Any]) -> str:
    quant_method = global_cfg.get("effective_quantization_method") or "none"
    quant_details = global_cfg.get("quantization_details") or {}
    if quant_method in (None, "none"):
        dt = str(global_cfg.get("effective_torch_dtype") or "").lower()
        if "bfloat16" in dt or dt == "bf16":
            return "bf16"
        if "float16" in dt or dt == "fp16" or "half" in dt:
            return "fp16"
        return dt or "fp32"
    bits = quant_details.get("hqq_bits") or quant_details.get("awq_bits") or quant_details.get("quantize_bits")
    if bits:
        return f"{quant_method}-i{bits}"
    return str(quant_method)

def build_adapter_root_path(global_cfg: Dict[str, Any], adapters_root_dir: str, adapter_name: str) -> Tuple[str, str]:
    model_base = global_cfg.get("base_model_name") or Path(global_cfg.get("other_config", {}).get("base_model_name_or_path", "model")).name
    model_folder = f"{model_base}.{_precision_suffix_from_global_config(global_cfg)}"
    adapter_root = os.path.join(adapters_root_dir, model_folder, adapter_name)
    return model_folder, adapter_root

async def run_test_adapter_training(
    adapter_output_dir: str, 
    adapter_logical_name: str,
    dataset_path: str,
    training_params: Dict[str, Any],
    engine_instance_for_callbacks, # Pass the engine instance for direct subscription
    keep_existing_checkpoints: bool = False
) -> Optional[str]:     

    print("\n--- Configuring and Starting Test Adapter Training ---")

    if os.path.exists(adapter_output_dir):
        if keep_existing_checkpoints:
            print(f"[DEMO] Preserving existing adapter output directory: {adapter_output_dir}")
        else:
            print(f"[DEMO] Cleaning existing adapter output directory: {adapter_output_dir}")
            # Add some resilience to rmtree, e.g. for lingering file handles on Windows
            for _ in range(3): # Try a few times
                try:
                    shutil.rmtree(adapter_output_dir)
                    break
                except OSError as e:
                    print(f"[DEMO] Warning: rmtree failed (attempt): {e}. Retrying in 1s...")
                    time.sleep(1)
            else:
                print(f"[DEMO] Error: Failed to clean existing adapter output directory: {adapter_output_dir} after multiple attempts.")
                return None
    os.makedirs(adapter_output_dir, exist_ok=True)
    if not keep_existing_checkpoints or not os.path.exists(adapter_output_dir):
        print(f"[DEMO] Created adapter output directory: {adapter_output_dir}")
    else:
        print(f"[DEMO] Using adapter output directory: {adapter_output_dir}")

    # 3. Prepare TrainingConfig
    dataset_config = DatasetConfig(
        dataset_path=dataset_path,
        formatting=DatasetFormat.MESSAGES, 
        # For MESSAGES format, the trainer should apply the chat template.
        # PreprocessingMode.APPLY_CHAT_TEMPLATE signals this.
        # Fallback to FULL_TEXT if APPLY_CHAT_TEMPLATE is somehow not defined (defensive).
        preprocessing_mode=PreprocessingMode.APPLY_CHAT_TEMPLATE.value if hasattr(PreprocessingMode, 'APPLY_CHAT_TEMPLATE') else PreprocessingMode.FULL_TEXT.value,
        columns=ColumnsConfig(messages="messages"), 
        tags=DatasetTags() 
    )
    
    train_config_dict = {
        "adapter_name_to_train": adapter_logical_name,
        "training_mode": "sft",
        "output_dir": adapter_output_dir, 
        "dataset": dataset_config.model_dump(),
        "max_sequence_length": training_params.get("train_override_ctx"), # Will be None if "auto" or not provided
        # "per_device_train_batch_size": 1, # Default is 1 in TrainingConfig
        # "gradient_accumulation_steps": 4, # Default is 8 in TrainingConfig
        #"learning_rate": 2e-4,            # Default is 5e-5 in TrainingConfig
        "num_train_epochs": 3.0, 
        "max_steps": training_params.get("steps", -1),
        # "lr_scheduler_type": "linear",      # Default is "cosine" in TrainingConfig
        # "warmup_steps": 0,                  # Default is 10 in TrainingConfig
        # "max_grad_norm": 1.0,               # Default is 1.0 in TrainingConfig
        # "seed": 42,                         # Default is 42 in TrainingConfig
        # Per user request, no intermediate checkpoints. Final save is done in mp13_engine.run_training.
        # "save_strategy": "no", # Default is "epoch" in TrainingConfig. Engine handles final save.
        # "save_steps": 0,       # Default is 0 in TrainingConfig.
        # "save_total_limit": 1, # Default is 1 in TrainingConfig.
        # "optim": "adamw_torch",             # Default is "adamw_torch" in TrainingConfig
        #"gradient_checkpointing": False, # Explicitly False due to observed instability
        "trainer_compute_precision": training_params.get("trainer_precision", "bf16"), "resume_from_checkpoint": None # Not resuming for this demo
    }
    if train_config_dict["max_steps"] > 0:
        train_config_dict["num_train_epochs"] = 999.0 

    try:
        config = TrainingConfig(**train_config_dict)
    except Exception as e:
        print(f"[DEMO] Error creating TrainingConfig: {e}")
        return None

    # With the new API, start-training takes the config directly.
    # The engine must be in TRAIN mode (set in main() before calling this function).
    # The engine's start_training will handle setting its status to PREPARING then TRAINING.
    
    print("[DEMO] Starting training process via API with TrainingConfig...")
    start_response = await call_api("start-training", config.model_dump()) 

    # The start_response from the API layer wraps the engine's response.
    # Engine's start_training returns: {"status": "success", "message": ..., "data": training_status_dict}
    if start_response.get("status") != "success":
        print(f"[DEMO] Failed to start training. API Response: {json.dumps(start_response, indent=2)}")
        return None
    
    print(f"[DEMO] Training initiation reported as '{start_response.get('message', 'N/A')}'. Monitoring progress...")
    # The engine should now be in TRAINING status if initiation was successful.
    
    # --- Event-driven progress monitoring ---
    training_log_history = deque(maxlen=10)
    adapter_report_seen_tokens: Dict[str, Optional[Tuple[Any, Any]]] = {}
    current_progress_details_dict = {} # Stores the latest full status details
    training_complete_event = asyncio.Event()
    final_adapter_path_result: Optional[str] = None # To store the result from the callback
    console_lock = asyncio.Lock()
    resource_info_emitted = False
    heuristic_summary_emitted = False

    async def on_training_status_update_callback(status_details: Dict[str, Any]):
        nonlocal current_progress_details_dict, final_adapter_path_result, resource_info_emitted, heuristic_summary_emitted

        # DEBUG: Print received status details to see if intermediate updates are coming through
        #if status_details.get("status") == TrainingStatus.TRAINING.value:
        #    print(f"[DEMO_CALLBACK_DEBUG] Received training update: Step={status_details.get('current_step')}, Loss={status_details.get('loss')}, LR={status_details.get('learning_rate')}")

        async with console_lock:
            current_progress_details_dict.clear()
            current_progress_details_dict.update(status_details) # Keep latest full details
            adapter_reports = status_details.get("adapter_reports") or {}
            if not resource_info_emitted:
                resource_report = status_details.get("resource_report") or {}
                eff_cfg = status_details.get("effective_training_config") or {}
                sources = eff_cfg.get("sources") or resource_report.get("param_sources") or {}
                hw = resource_report.get("hardware") or {}
                per_devices = hw.get("devices") or []
                if resource_report or eff_cfg:
                    print("\n[DEMO] === Training resource plan ===")
                    if per_devices:
                        min_free = min(float(d.get("free_gb", 0.0)) for d in per_devices)
                        device_count = len(per_devices)
                        print(f"[DEMO][GPU] devices used: {device_count}, min_free={min_free:.2f}GB (budget uses min device)")
                    if per_devices:
                        for dev in per_devices:
                            dev_id = dev.get("id", "?")
                            name = dev.get("name", "gpu")
                            free_gb = float(dev.get("free_gb", 0.0))
                            total_gb = float(dev.get("total_gb", 0.0))
                            print(f"[DEMO][GPU] cuda:{dev_id} ({name}) free={free_gb:.2f}GB / total={total_gb:.2f}GB")
                    budget = resource_report.get("training_budget_gb")
                    if budget is not None:
                        opt_mem = resource_report.get("optimizer_mem_per_device_gb")
                        opt_mem_str = f"{float(opt_mem):.2f}GB" if opt_mem is not None else "n/a"
                        print(f"[DEMO][GPU] Per-device training budget after safety/optimizer: {float(budget):.2f}GB (opt est {opt_mem_str})")
                    if eff_cfg:
                        eff_bs = eff_cfg.get("effective_batch_size")
                        ctx_len = eff_cfg.get("context_length")
                        grad_ckpt = eff_cfg.get("gradient_checkpointing")
                        print(f"[DEMO][CFG] context={ctx_len}, per_device_batch={eff_cfg.get('per_device_batch_size')}, grad_accum={eff_cfg.get('gradient_accumulation')} (effective={eff_bs})")
                        print(f"[DEMO][CFG] grad_checkpointing={'on' if grad_ckpt else 'off'}, precision={eff_cfg.get('precision')}, dataset_size={eff_cfg.get('dataset_size')}")
                    if sources:
                        print("[DEMO][CFG] parameter sources: " + ", ".join(f"{k}={v}" for k, v in sources.items()))
                    resource_info_emitted = True
            if not heuristic_summary_emitted:
                heuristic_summary = status_details.get("heuristic_settings_summary")
                if heuristic_summary:
                    print("\n[DEMO] === Effective training settings ===")
                    print(heuristic_summary)
                    heuristic_summary_emitted = True

            def maybe_emit_adapter_report(stage_key: str):
                report = adapter_reports.get(stage_key)
                if not isinstance(report, dict):
                    return
                token = (report.get("generated_at"), report.get("step"))
                if adapter_report_seen_tokens.get(stage_key) == token:
                    return
                adapter_report_seen_tokens[stage_key] = token
                generated_at = report.get("generated_at", "unknown")
                step_val = report.get("step", "n/a")
                print(f"\n[DEMO] === Adapter report ({stage_key} | step={step_val} | generated_at={generated_at}) ===")
                for line in report.get("lines", []):
                    print(f"[DEMO][ADAPTER] {line}")
                metrics_block = report.get("metrics")
                if stage_key == "delta" and isinstance(metrics_block, dict):
                    delta_metrics = metrics_block.get("delta")
                    if isinstance(delta_metrics, dict) and delta_metrics:
                        print("[DEMO][ADAPTER] -- Delta metrics snapshot --")
                        for label, vals in delta_metrics.items():
                            start = vals.get("start")
                            end = vals.get("end")
                            delta_val = vals.get("delta")
                            try:
                                start_f = f"{float(start):.4f}" if start is not None else "n/a"
                                end_f = f"{float(end):.4f}" if end is not None else "n/a"
                                delta_f = f"{float(delta_val):+.4f}" if delta_val is not None else "n/a"
                                print(f"[DEMO][ADAPTER]   {label}: {start_f} -> {end_f} ({delta_f})")
                            except (TypeError, ValueError):
                                print(f"[DEMO][ADAPTER]   {label}: {vals}")
                training_log_history.append(
                    f"[{datetime.datetime.now().strftime('%H:%M:%S')}] DEMO: Recorded {stage_key} adapter report (step={step_val})."
                )

            maybe_emit_adapter_report("initial")
            maybe_emit_adapter_report("final")
            maybe_emit_adapter_report("delta")

            status_val = status_details.get("status")
            current_step = status_details.get("current_step", 0)
            total_steps = status_details.get("total_steps")

            # Log significant events to history (simplified for now, can be expanded)
            if status_details.get("_event_type_") == "log": # Assuming engine might send specific log events
                log_msg = status_details.get("message", "Log event")
                training_log_history.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ENG: {log_msg}")
            # Log first time PREPARING or TRAINING status is seen from callback
            elif status_val == TrainingStatus.PREPARING.value and not any(TrainingStatus.PREPARING.value in s for s in training_log_history if isinstance(s, str)):
                 log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] DEMO: Status changed to {status_val}"
                 training_log_history.append(log_entry)
                 # print(log_entry) # Optionally print important status changes immediately
            elif status_val == TrainingStatus.TRAINING.value and current_step == 0 and total_steps and not any(f"{TrainingStatus.TRAINING.value}, Step=0/" in s for s in training_log_history if isinstance(s, str)):
                 log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] DEMO: Status {status_val}, Step=0/{total_steps}. Training started."
                 training_log_history.append(log_entry)
                 # print(log_entry) # Optionally print important status changes immediately
                 pass

            if status_val in [TrainingStatus.COMPLETED.value, TrainingStatus.ERROR.value, TrainingStatus.STOPPED.value]:
                if status_val == TrainingStatus.COMPLETED.value:
                    msg = "[DEMO] Training completed successfully via callback!"
                    training_log_history.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
                    print(f"\n{msg}") # Print final message clearly
                    
                    path_from_status = status_details.get("final_adapter_path")
                    training_log_history.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] DEMO: Final adapter path from status: {path_from_status}")
                    #print(f"[DEMO] Final adapter files expected in directory: {path_from_status}")
                    
                    if path_from_status and os.path.exists(os.path.join(path_from_status, "adapter_config.json")):
                        final_adapter_path_result = path_from_status
                        training_log_history.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] DEMO: Confirmed adapter_config.json exists.")
                        print(f"[DEMO] Confirmed 'adapter_config.json' exists in {path_from_status}")
                    else:
                        missing_reason = "final_adapter_path not provided" if not path_from_status else "'adapter_config.json' NOT found"
                        err_msg = f"[DEMO] Error: {missing_reason} in {path_from_status}."
                        training_log_history.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {err_msg}")
                        print(err_msg)
                        final_adapter_path_result = None
                else: # ERROR or STOPPED
                    error_msg_detail = status_details.get('error_message', 'No error message provided.')
                    msg = f"[DEMO] Training ended unexpectedly via callback: Status={status_val}, Error: {error_msg_detail}"
                    training_log_history.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
                    print(f"\n{msg}")
                    final_adapter_path_result = None
                training_complete_event.set()

    # Subscribe to engine events
    if engine_instance_for_callbacks and hasattr(engine_instance_for_callbacks, 'subscribe_to_training_status'):
        await engine_instance_for_callbacks.subscribe_to_training_status(on_training_status_update_callback)
        print("[DEMO] Subscribed to training status updates from the engine.")
    else:
        print("[DEMO] Error: Engine instance not available or does not support status subscription. Cannot use event callbacks.")
        return None

    # Display loop
    is_progress_line_active = False
    last_displayed_message = ""
    start_time_training_loop = None # To calculate ETA and it/s
    last_printed_step = -1
    last_printed_loss_str = "" # Store the string representation of loss
    last_step_time = None
    initial_step_logged = False # To print header once

    async def display_training_output_loop():
        nonlocal is_progress_line_active, last_displayed_message, start_time_training_loop, last_step_time, initial_step_logged
        nonlocal last_printed_step, last_printed_loss_str

        while not training_complete_event.is_set():
            async with console_lock: # Ensure atomic read of shared state
                details_to_render = dict(current_progress_details_dict)

            status_val = details_to_render.get("status")
            current_step_val = details_to_render.get("current_step")
            total_steps = details_to_render.get("total_steps")

            # Primary progress line for TRAINING status
            if status_val == TrainingStatus.TRAINING.value and current_step_val is not None and total_steps is not None:
                if start_time_training_loop is None and current_step_val >= 0: # Start timer on first valid step
                    start_time_training_loop = time.monotonic()
                    last_step_time = start_time_training_loop

                if not initial_step_logged and current_step_val >=0:
                    # Print a header or initial status line
                    print(f"[DEMO] Training started. Monitoring progress (Total Steps: {total_steps})...")
                    initial_step_logged = True

                loss = details_to_render.get("loss", "N/A")
                lr = details_to_render.get("learning_rate", "N/A")
                grad_norm = details_to_render.get("grad_norm", "N/A")

                # Get GPU memory info from details_to_render
                current_gpu_mem = details_to_render.get("current_gpu_mem_allocated_mb", 0.0)
                peak_gpu_mem_op = details_to_render.get("peak_gpu_mem_current_op_mb", 0.0)
                
                loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
                lr_str = f"{lr:.2e}" if isinstance(lr, (float)) else str(lr) # Check for float specifically for scientific notation
                grad_norm_str = f"{grad_norm:.4f}" if isinstance(grad_norm, (int, float)) else str(grad_norm)
                

                # Calculate ETA and it/s
                elapsed_time = time.monotonic() - start_time_training_loop if start_time_training_loop else 0
                iterations_per_second = 0
                eta_str = "-:--:--"

                if current_step_val > 0 and elapsed_time > 0:
                    iterations_per_second = current_step_val / elapsed_time
                    if iterations_per_second > 0:
                        remaining_steps = total_steps - current_step_val
                        eta_seconds = remaining_steps / iterations_per_second
                        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                # Progress bar visualization (simple text based)
                progress_percent = (current_step_val / total_steps) * 100 if total_steps > 0 else 0
                bar_length = 10 # Length of the progress bar (shrunk by 3x from 30)
                filled_length = int(bar_length * current_step_val // total_steps) if total_steps > 0 else 0
                bar = '█' * filled_length + '-' * (bar_length - filled_length)

                # Format similar to tqdm, but with fixed-width fields for alignment
                progress_message = (
                    f"{progress_percent:3.0f}%|{bar}| "
                    f"{current_step_val:>4}/{total_steps:<4} "
                    f"[{str(datetime.timedelta(seconds=int(elapsed_time))):>8}<{eta_str:>8}, "
                    f"{iterations_per_second:6.2f}it/s, "
                    f"Loss:{loss_str:>8}, "
                    f"LR:{lr_str:>9}, " # GPU Mem: CurrentAlloc / PeakForOp
                    f"Mem:{current_gpu_mem:.0f}/{peak_gpu_mem_op:.0f}MB, "
                    f"GradNorm:{grad_norm_str:>8}]"
                )
                
                # Determine if an update should be printed.
                # Print if step has changed, or if loss string has changed.
                should_print = False
                if current_step_val != last_printed_step:
                    should_print = True
                    last_printed_step = current_step_val
                
                if loss_str != last_printed_loss_str and loss != "N/A": # Check if loss string changed
                    should_print = True
                    last_printed_loss_str = loss_str
                
                # Also print if it's the very first message for training (step 0)
                if not initial_step_logged and current_step_val == 0: # Ensure first step 0 is printed
                    should_print = True

                if should_print:
                    print(f"[DEMO] {progress_message}")
                    sys.stdout.flush()
                    last_displayed_message = progress_message # Corrected from display_message
                is_progress_line_active = True
            elif status_val == TrainingStatus.TRAINING.value and (current_step_val is None or total_steps is None) and not initial_step_logged:
                pass # Waiting for first valid step/total_steps from engine callback
            
            # Initial "Preparing..." message
            elif status_val == TrainingStatus.PREPARING.value and not is_progress_line_active:
                prep_message = "[DEMO] Status: preparing..."
                if prep_message != last_displayed_message:
                    print(prep_message)
                    sys.stdout.flush()
                    last_displayed_message = prep_message
                is_progress_line_active = True

            # If training is done or in a non-active state, and we were printing progress,
            # the callback will print the final status. We just reset our flags.
            elif status_val not in [TrainingStatus.TRAINING.value, TrainingStatus.PREPARING.value] and is_progress_line_active:
                is_progress_line_active = False
                last_displayed_message = "" # Reset

            await asyncio.sleep(0.15) # Slightly faster refresh for responsiveness, but not too fast.

    display_task = asyncio.create_task(display_training_output_loop())

    try:
        await asyncio.wait_for(training_complete_event.wait(), timeout=3600) # Max 1 hour for training
    except asyncio.TimeoutError:
        print() # Ensure a new line before the timeout message
        print("[DEMO] Training monitoring timed out via event.")
        training_log_history.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] DEMO: Training timed out.")
        # Attempt to stop training if engine supports it and is still active
        if engine_instance_for_callbacks and hasattr(engine_instance_for_callbacks, 'get_current_training_status_direct'): # Hypothetical direct status check
            if engine_instance_for_callbacks.get_current_training_status_direct().get('status') == TrainingStatus.TRAINING.value:
                 await call_api("stop-training", {}) # Best effort
        final_adapter_path_result = None # Ensure it's None on timeout
        training_complete_event.set() # Allow display_task to finish
    
    await display_task # Ensure display task finishes and cleans up console

    # Unsubscribe
    if engine_instance_for_callbacks and hasattr(engine_instance_for_callbacks, 'unsubscribe_from_training_status'):
        await engine_instance_for_callbacks.unsubscribe_from_training_status(on_training_status_update_callback)
        print("[DEMO] Unsubscribed from training status updates.")

    return final_adapter_path_result

class TestAdapterInference:
    def __init__(
        self,
        adapter_path: str, 
        adapter_name: str, 
        max_length: int = 512,
        gen_config: Dict[str, Any] ={}
    ):
        self.adapter_path = adapter_path
        self.adapter_name = adapter_name 
        self.max_length = max_length
        self.is_initialized_for_inference = False
        self.gen_config = {
            "max_new_tokens": 150,
        }


    async def configure_engine_for_inference(self):
        """Ensures engine is in INFERENCE mode for the upcoming requests."""
        print(f"\n--- Configuring Engine for Inference with adapter '{self.adapter_name}' ---")
        
        # 1. Ensure engine is in INFERENCE mode
        mode_set_resp = await call_api("check-set-mode", {"mode": EngineMode.INFERENCE.value, "force": False})
        if not (mode_set_resp.get("status") == "success" and mode_set_resp.get("data", {}).get("effective_mode") == EngineMode.INFERENCE.value):
            print(f"[DEMO] Failed to set engine to INFERENCE mode. Response: {json.dumps(mode_set_resp, indent=2)}")
            return False
        print("[DEMO] Engine confirmed in INFERENCE mode.")

        self.is_initialized_for_inference = True
        return True

    async def run_inference_request(self, inputs: Union[List[str], List[List[Dict[str, str]]]],
                                   use_messages_format: bool = True,
                                   stream: bool = False,
                                   generation_config_override: Optional[Dict[str, Any]] = None,
                                   active_adapters_override: Optional[List[str]] = None
                                   ) -> Union[Tuple[List[Dict[str, Any]], Dict[str, Any]], AsyncIterator[Dict[str, Any]]]:
        if not self.is_initialized_for_inference:
            print("[DEMO] Error: Inference session not configured (engine mode/active adapter).")
            if stream:
                async def empty_iterator(): yield {"response": "Error: Inference session not configured", "error": "Client-side check", "chunkType": ChunkType.ERROR.value}
                return empty_iterator()
            return [{"response": "Error: Inference session not configured", "error": "Client-side check"}] * len(inputs), {}

        print(f"[DEMO] Running inference for {len(inputs)} inputs. Stream: {stream}.")
        
        # Use override if provided, otherwise use default for verification
        gen_config = generation_config_override or {}

        # GenerationConfig is now part of InferenceRequest
        request_dict = {
            "request_id": f"req_{int(time.time())}",
            "generation_config": gen_config,
            "active_adapters": active_adapters_override,
        } # type: Dict[str, Any]
        request_dict["stream"] = stream

        if use_messages_format:
            request_dict["messages_list"] = inputs
            print(f"[DEMO] Inference request uses 'messages_list' (chat format) for {len(inputs)} inputs.")
        else:
            request_dict["raw_list"] = inputs
            print(f"[DEMO] Inference request uses 'raw_list' (string format) for {len(inputs)} inputs.")

        try:
            request = InferenceRequest(**request_dict)
        except Exception as e:
            print(f"[DEMO] Error creating InferenceRequest: {e}")
            if stream:
                async def error_iterator_req(): yield {"response": f"Error creating request: {e}", "error": str(e), "chunkType": ChunkType.ERROR.value}
                return error_iterator_req()
            return [{"response": f"Error creating request: {e}", "error": str(e)}] * len(inputs), {}

        response_from_call = await call_api("run-inference", request.model_dump())

        # Handle pre-flight errors from the API call itself (e.g., engine not ready)
        if "stream" not in response_from_call:
            error_msg = response_from_call.get("message", "Unknown API error")
            print(f"[DEMO] Error calling run-inference: {error_msg}")
            if stream:
                async def error_iterator(): yield {"chunkType": ChunkType.ERROR.value, "error": error_msg}
                return error_iterator()
            return [{"response_text": f"Error: {error_msg}", "error": error_msg}] * len(inputs), {}

        # If streaming is requested, return the iterator directly.
        if stream:
            return response_from_call["stream"]

        # For non-streaming, consume the iterator to build the final response.
        # This logic correctly aggregates tokens for each item in the batch.
        num_inputs = len(inputs)
        results = [{} for _ in range(num_inputs)]
        response_texts = ["" for _ in range(num_inputs)]
        metrics = {}

        async for chunk in response_from_call["stream"]:
            chunk_type = chunk.get("chunkType")
            prompt_index = chunk.get("prompt_index")

            if chunk_type == ChunkType.STREAMING_CHUNK.value:
                if prompt_index is not None and 0 <= prompt_index < num_inputs:
                    response_texts[prompt_index] += chunk.get("chunk_text", "")
                    if chunk.get("is_final_chunk"):
                        final_chunk_data = {k: v for k, v in chunk.items() if k not in ["chunkType", "chunk_text"]}
                        results[prompt_index].update(final_chunk_data)
                        results[prompt_index]["response_text"] = response_texts[prompt_index]
                        _print_per_item_metrics(prompt_index, num_inputs, chunk)

            elif chunk_type == ChunkType.STREAMING_ENDED.value:
                metrics = {
                    "total_input_tokens": chunk.get("total_input_tokens"),
                    "total_output_tokens": chunk.get("total_output_tokens"),
                    "total_generation_duration_sec": chunk.get("total_generation_duration_sec"),
                    "overall_tps": chunk.get("overall_tps"),
                    "avg_time_to_first_token_sec": chunk.get("avg_time_to_first_token_sec"),
                    "cache_queued": chunk.get("cache_queued"),
                    "in_flight_req": chunk.get("in_flight_req"),
                    "mem_allocated": chunk.get("mem_allocated"),
                    "mem_reserved": chunk.get("mem_reserved"),
                }
                break

            elif chunk_type == ChunkType.ERROR.value:
                error_msg = chunk.get("error", "Unknown engine error during non-streaming inference")
                print(f"[DEMO] Engine returned an error: {error_msg}")
                error_results = [{"response_text": f"Error: {error_msg}", "error": error_msg}] * num_inputs
                return error_results, {}

        # Post-loop check for partial results
        for i in range(num_inputs):
            if "response_text" not in results[i] and response_texts[i]:
                results[i]["response_text"] = response_texts[i]
                results[i]["error"] = "Stream ended before final chunk was received for this item."

        return results, metrics

    def print_metrics(self, model_name: str, metrics: Dict[str, Any]):
        if not metrics or not any(metrics.values()):
            print(f"[DEMO] No metrics available for {model_name}.")
            return

        total_input = metrics.get("total_input_tokens")
        total_output = metrics.get("total_output_tokens")
        duration = metrics.get("total_generation_duration_sec")
        tps = metrics.get("overall_tps")
        latency = metrics.get("avg_time_to_first_token_sec")
        cache_queued = metrics.get("cache_queued")
        in_flight = metrics.get("in_flight_req")
        mem_alloc = metrics.get("mem_allocated")
        mem_rsvd = metrics.get("mem_reserved")

        metrics_line = f"Metrics ({model_name}):"
        if total_input is not None: metrics_line += f" In: {total_input}"
        if total_output is not None: metrics_line += f" Out: {total_output}"
        if duration is not None: metrics_line += f" GenTime: {duration:.1f}s"
        if tps is not None: metrics_line += f" TPS: {tps:.1f}"
        if latency is not None: metrics_line += f" Avg Latency: {latency * 1000:.0f}ms"
        if cache_queued:
            metrics_line += f" Queued: {cache_queued}"
        if in_flight:
            metrics_line += f" In-flight: {in_flight}"
        if mem_alloc is not None: metrics_line += f" Mem(A): {mem_alloc:.0f}MB"
        if mem_rsvd is not None: metrics_line += f" Mem(R): {mem_rsvd:.0f}MB"
        
        print(f"[DEMO] {metrics_line}")

    async def batch_inference_compare(self, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.is_initialized_for_inference:
            raise ValueError("Inference session must be configured (engine mode/active adapter).")

        uses_messages_format = "messages" in test_data[0] if test_data else False
        inputs_for_engine: Union[List[str], List[List[Dict[str, str]]]]
        prompts_data_tracking = []

        if uses_messages_format:
            messages_list_for_engine = []
            for item in test_data:
                messages_for_template = [msg for msg in item["messages"] if msg.get("role") != "assistant"]
                messages_list_for_engine.append(messages_for_template)
                prompts_data_tracking.append({
                    "instruction": next((m["content"] for m in item["messages"] if m["role"] == "user"), ""),
                    "system_message": next((m["content"] for m in item["messages"] if m["role"] == "system"), ""),
                    "expected_output": item.get("expected_output") or next((m["content"] for m in item["messages"] if m["role"] == "assistant"), ""),
                    "format": "messages"
                })
            inputs_for_engine = messages_list_for_engine
        else:
            formatted_prompts = []
            for item in test_data:
                instruction = item.get("instruction", "")
                input_text = item.get("input", "") 
                system_message = item.get("system_message", "")
                expected_output = item.get("expected_output", "")
                prompt = f"User: {instruction}{f'{chr(10)}{input_text}' if input_text else ''}\n\nAssistant: "
                if system_message: 
                    prompt = f"system: {system_message}\n{prompt}"
                formatted_prompts.append(prompt)
                prompts_data_tracking.append({
                    "instruction": instruction, "input_text": input_text, 
                    "system_message": system_message, "expected_output": expected_output,
                    "format": "original"
                })
            inputs_for_engine = formatted_prompts
        
        # --- Run inference for base model (no adapter) ---
        print("\n[DEMO] === Running batch inference for base model (no adapter) ===")
        base_model_results_raw, base_metrics = await self.run_inference_request(
            inputs_for_engine,
            use_messages_format=uses_messages_format, stream=False,
            generation_config_override=self.gen_config,
            active_adapters_override=[]  # Empty list specifies base model
        )
        self.print_metrics("Base Model", base_metrics)

        # --- Run inference with the fine-tuned adapter ---
        print(f"\n[DEMO] === Running batch inference with adapter '{self.adapter_name}' ===")
        adapter_results_raw, adapter_metrics = await self.run_inference_request(
            inputs_for_engine,
            use_messages_format=uses_messages_format, stream=False,
            generation_config_override=self.gen_config,
            active_adapters_override=[self.adapter_name]
        )
        self.print_metrics(f"Adapter '{self.adapter_name}'", adapter_metrics)

        
        results = []
        for i, data_track_item in enumerate(prompts_data_tracking):
            with_adapter_response_item = adapter_results_raw[i] if i < len(adapter_results_raw) else {"response_text": "Error: No response", "error": "Index out of bounds"}
            without_adapter_response_item = base_model_results_raw[i] if i < len(base_model_results_raw) else {"response_text": "Error: No response", "error": "Index out of bounds"}
            
            result = {
                "instruction": data_track_item["instruction"],
                "system_message": data_track_item["system_message"],
                "expected_output": data_track_item["expected_output"],
                "with_adapter": with_adapter_response_item.get("response_text", ""),
                "with_adapter_was_truncated": with_adapter_response_item.get("was_truncated"),
                "with_adapter_error": with_adapter_response_item.get("error"),
                "without_adapter": without_adapter_response_item.get("response_text", ""),
                "without_adapter_was_truncated": without_adapter_response_item.get("was_truncated"),
                "without_adapter_error": without_adapter_response_item.get("error"),
            }
            if data_track_item["format"] == "original":
                result["input"] = data_track_item["input_text"]
            results.append(result)
        return results

def string_similarity(expected_output: str, generated_output: str, max_new_tokens: int) -> float:
    """
    Calculates a similarity score that combines Jaccard index with a length penalty.
    - It compares the generated output against a prefix of the expected output.
    - It penalizes the score if the generated output is much shorter than expected.
    """
    if not expected_output or not generated_output:
        return 0.0

    # --- 1. Calculate Jaccard similarity on the generated prefix ---
    expected_prefix = expected_output[:len(generated_output)]
    s1_lower = expected_prefix.lower().strip()
    s2_lower = generated_output.lower().strip()

    words1 = set(s1_lower.split())
    words2 = set(s2_lower.split())
    if not words1 or not words2: return 0.0
    jaccard_score = len(words1.intersection(words2)) / len(words1.union(words2))

    # --- 2. Calculate a length penalty ---
    # The target length is the smaller of max_new_tokens or the full expected output length.
    target_len = min(len(expected_output), max_new_tokens)
    generated_len = len(generated_output)

    # The penalty is a ratio of generated length to target length, capped at 1.0.
    length_ratio = min(1.0, generated_len / target_len if target_len > 0 else 0)

    # The final score is the Jaccard score scaled by the length ratio.
    return jaccard_score * length_ratio

def truncated_print(message: str, max_chars: int = 120):
    """Print a truncated version of the message if it exceeds max_chars.
    Shows more content at the start and the last few words, separated by '...'.
    """
    if len(message) <= max_chars:
        print(message)
        return

    # Calculate how many characters should appear at the start and the end
    start_part_chars = max_chars - 40  # Reserve 40 chars for ending plus ellipsis

    # Splitting the message into words
    words = message.split()

    # Constructing the truncated message
    first_part = message[:start_part_chars].strip()
    last_part = " ".join(words[-5:]).strip()
    print(f"{first_part} ... {last_part}")

def analyze_results(results: List[Dict[str, Any]]):
    total = len(results)
    if total == 0:
        print("[DEMO] No results to analyze.")
        return False

    adapter_different_count = 0
    adapter_matches_expected_count = 0
    base_matches_expected_count = 0
    
    num_trained_examples = min(5, total) 
    
    # This is the max_new_tokens value used in the batch_inference_compare function.
    max_new_tokens_for_scoring = 150

    print("\n--- Detailed Results ---")
    for i, result in enumerate(results):
        expected = result.get("expected_output", "")
        with_adapter = result.get("with_adapter", "")
        without_adapter = result.get("without_adapter", "")

        print(f"\nExample {i+1} ({'Trained' if i < num_trained_examples else 'Untrained'}):")
        truncated_print(f"  Instruction: {result['instruction']}")
        if result.get("input"): truncated_print(f"  Input: {result['input']}")
        if result.get("system_message"): truncated_print(f"  System: {result['system_message']}")
        truncated_print(f"  Expected: '{expected}'")
        truncated_print(f"  With Adapter: '{with_adapter}' (Similarity: {string_similarity(expected, with_adapter, max_new_tokens_for_scoring):.2f})")
        if result.get('with_adapter_was_truncated'):
            print(f"    \033[93mWarning: Adapter output may have been truncated.\033[0m")
        truncated_print(f"  Without Adapter: '{without_adapter}' (Similarity: {string_similarity(expected, without_adapter, max_new_tokens_for_scoring):.2f})")
        if result.get('without_adapter_was_truncated'):
            print(f"    \033[93mWarning: Base model output may have been truncated.\033[0m")

        if string_similarity(with_adapter, without_adapter, max_new_tokens_for_scoring) < 0.95: 
            adapter_different_count += 1
        
        if i < num_trained_examples: 
            if string_similarity(expected, with_adapter, max_new_tokens_for_scoring) > 0.7:
                adapter_matches_expected_count += 1
            if string_similarity(expected, without_adapter, max_new_tokens_for_scoring) > 0.7:
                base_matches_expected_count += 1
    
    print("\n--- Summary Statistics ---")
    print(f"Total examples: {total}")
    print(f"Adapter produced different output than base model: {adapter_different_count}/{total} ({adapter_different_count/total*100:.1f}%)")
    
    if num_trained_examples > 0:
        print(f"For the {num_trained_examples} 'trained' examples:")
        print(f"  Adapter matched expected output: {adapter_matches_expected_count}/{num_trained_examples} ({adapter_matches_expected_count/num_trained_examples*100:.1f}%)")
        print(f"  Base model matched expected output: {base_matches_expected_count}/{num_trained_examples} ({base_matches_expected_count/num_trained_examples*100:.1f}%)")
    
    success = False
    if num_trained_examples > 0:
        success = (adapter_different_count > 0) and \
                  (adapter_matches_expected_count > base_matches_expected_count) and \
                  (adapter_matches_expected_count / num_trained_examples >= 0.6) 
    elif total > 0 : 
        success = (adapter_different_count > 0)
        print("[DEMO] Note: Verification based only on adapter producing different output, as no 'trained' examples with expected outputs were evaluated.")

    if success:
        print("\n✅ VERIFICATION SUCCESSFUL (based on current criteria)")
    else:
        print("\n❌ VERIFICATION FAILED (based on current criteria)")
    return success

def parse_k_notation(k_string: Optional[str]) -> Optional[int]:
    """Converts K notation (e.g., '2K') or plain integer string to an int. Handles 'auto' and None."""
    if k_string is None or k_string.lower() == "auto":
        return None
    
    k_string_lower = k_string.lower()
    if k_string_lower.endswith('k'):
        try:
            num_part = k_string_lower[:-1]
            val = int(num_part)
            return val * 1024
        except ValueError:
            print(f"[DEMO] Warning: Invalid K notation '{k_string}'. Expected format like '2K', '4K'. Using None.")
            return None
    try: # Allow plain integer as well
        return int(k_string)
    except ValueError:
        print(f"[DEMO] Warning: Invalid context size value '{k_string}'. Expected integer or K notation. Using None.")
        return None

def _print_per_item_metrics(prompt_index: int, total_items: int, chunk_data: Dict[str, Any]):
    """Helper to print a formatted line of per-item metrics, similar to mp13chat."""
    metrics_parts = []
    if (in_tok := chunk_data.get("input_tokens")) is not None:
        metrics_parts.append(f"In: {in_tok}")
    if (out_tok := chunk_data.get("output_tokens")) is not None:
        metrics_parts.append(f"Out: {out_tok}")
    if (gen_time := chunk_data.get("generation_duration_sec")) is not None:
        metrics_parts.append(f"GenTime: {gen_time:.1f}s")
    if (ttft := chunk_data.get("time_to_first_token_sec")) is not None:
        # In non-streaming batch, TTFT is the full generation time, so this check prevents redundant printing.
        if ttft < chunk_data.get("generation_duration_sec", float('inf')):
            metrics_parts.append(f"Latency: {ttft * 1000:.0f}ms")
    if (tps := chunk_data.get("tokens_per_second")) is not None:
        metrics_parts.append(f"TPS: {tps:.1f}")
    if (cache_metric := chunk_data.get("cache_metric")):
        metrics_parts.append(f"Cache: {cache_metric}")
    if (cache_warming := chunk_data.get("cache_warming")):
        metrics_parts.append(f"Warm-up: {cache_warming}")
    if chunk_data.get("was_truncated"):
        metrics_parts.append(f"Truncated: Yes")
    
    if metrics_parts:
        # This print happens during the non-streaming batch verification.
        # A newline helps separate it from the final summary.
        print(f"[DEMO]  Metrics (Item {prompt_index + 1}/{total_items}): {' | '.join(metrics_parts)}")

async def graceful_shutdown(signal_obj: Optional[signal.Signals] = None):
    """Handles graceful shutdown of the application."""
    global _shutdown_initiated
    if _shutdown_initiated:
        return
    _shutdown_initiated = True

    if signal_obj:
        print(f"\n[DEMO] Received signal {signal_obj.name}, initiating graceful shutdown...")
    else:
        print("\n[DEMO] Initiating graceful shutdown...")

    # Attempt to stop training if it's in a stoppable state
    try: # Outer try for the entire "check and stop training" phase
        print("[DEMO] Checking training status for potential stop...")
        try:
            status_resp = await asyncio.wait_for(call_api("get-training-status", {}), timeout=5.0)
            if status_resp.get("status") == "success":
                training_details = status_resp.get("data", {})
                current_training_status = training_details.get("status")
                if current_training_status in [TrainingStatus.TRAINING.value, TrainingStatus.PREPARING.value]:
                    print(f"[DEMO] Training is active ({current_training_status}). Attempting to stop training...")
                    try:
                        stop_resp = await asyncio.wait_for(call_api("stop-training", {}), timeout=5.0)
                        if stop_resp.get("status") == "success":
                            print("[DEMO] Stop-training command issued. Waiting for training to finalize (up to 30s)...")
                            wait_for_stop_timeout = 30  # seconds
                            wait_start_time = asyncio.get_event_loop().time()
                            while asyncio.get_event_loop().time() - wait_start_time < wait_for_stop_timeout:
                                await asyncio.sleep(3) # Check every 3 seconds
                                try:
                                    status_check_resp = await asyncio.wait_for(call_api("get-training-status", {}), timeout=3.0)
                                    if status_check_resp.get("status") == "success":
                                        current_training_status = status_check_resp.get("data", {}).get("status")
                                        print(f"[DEMO] ... training status now: {current_training_status}")
                                        if current_training_status not in [TrainingStatus.TRAINING.value, TrainingStatus.PREPARING.value]:
                                            print("[DEMO] Training has exited active state.")
                                            break
                                    else:
                                        print("[DEMO] ... could not get training status while waiting for stop.")
                                except asyncio.TimeoutError:
                                    print("[DEMO] ... timeout getting training status while waiting for stop.")
                                    break # Break inner loop on timeout
                            else: # Loop completed without break (timeout)
                                print("[DEMO] Timeout waiting for graceful stop. Attempting forceful cancellation...")
                                try:
                                    cancel_resp = await asyncio.wait_for(call_api("cancel-request", {}), timeout=5.0)
                                    if cancel_resp.get("status") == "success":
                                        print("[DEMO] Forceful cancel command issued. Waiting up to 10s for status change...")
                                        # Wait a bit more for the status to reflect cancellation (e.g., ERROR)
                                        cancel_wait_start = asyncio.get_event_loop().time()
                                        while asyncio.get_event_loop().time() - cancel_wait_start < 10:
                                            await asyncio.sleep(2)
                                            status_check_resp = await asyncio.wait_for(call_api("get-training-status", {}), timeout=3.0)
                                            if status_check_resp.get("status") == "success":
                                                current_training_status = status_check_resp.get("data", {}).get("status")
                                                print(f"[DEMO] ... training status after cancel request: {current_training_status}")
                                                if current_training_status not in [TrainingStatus.TRAINING.value, TrainingStatus.PREPARING.value]:
                                                    print("[DEMO] Training has exited active state after forceful cancel.")
                                                    break
                                        else: # Loop finished without break (timeout)
                                            print("[DEMO] Warning: Training did not exit active state after forceful cancel. Proceeding with shutdown, but it may fail.")
                                    else:
                                        print(f"[DEMO] Forceful cancel command failed: {cancel_resp.get('message')}")
                                except asyncio.TimeoutError:
                                    print("[DEMO] Timeout trying to issue forceful cancel command.")
                                except Exception as e_cancel:
                                    print(f"[DEMO] Error during forceful cancel: {e_cancel}")
                        else:
                            print(f"[DEMO] Failed to issue stop-training command: {stop_resp.get('message')}")
                    except asyncio.TimeoutError:
                        print("[DEMO] Timeout trying to issue stop-training command.")
                else:
                    print(f"[DEMO] Training not in an active stoppable state (status: {current_training_status}).")
            else:
                print("[DEMO] Could not get training status to determine if stop is needed.")
        except asyncio.TimeoutError:
            print("[DEMO] Timeout checking training status for potential stop.")
        except Exception as e_check_train:
            print(f"[DEMO] Error during check/stop training phase: {e_check_train}")
    except Exception as e_outer_train_stop_check: # Catch-all for the outer "check and stop training" try
        print(f"[DEMO] Error during the overall training stop/check procedure: {e_outer_train_stop_check}")

    # Always attempt to shut down the engine
    print("[DEMO] Attempting to shut down the global engine...")
    try:
        shutdown_response = await asyncio.wait_for(call_api("shutdown-engine", {"shutdown_all": True}), timeout=70.0)
        if shutdown_response.get("status") == "success":
            print("[DEMO] Global engine shut down successfully.")
        else:
            print(f"[DEMO] Warning: Shutting down global engine reported: {shutdown_response.get('message', 'N/A')}")
    except asyncio.TimeoutError:
        print("[DEMO] Timeout attempting to shut down the global engine.")
    except Exception as e_shutdown:
        # This might happen if the engine is already down or unresponsive
        print(f"[DEMO] Exception during engine shutdown: {e_shutdown} (Engine might be unresponsive or already down)")

    # Cancel all other running asyncio tasks
    # Ensure this runs even if previous steps had issues
    try:
        current_task = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks() if t is not current_task]
        if tasks:
            print(f"[DEMO] Cancelling {len(tasks)} outstanding asyncio tasks...")
            for task in tasks:
                task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                print("[DEMO] Outstanding tasks cancelled.")
            except Exception as e_gather:
                print(f"[DEMO] Error during gathering of cancelled tasks: {e_gather}")
    except Exception as e_cancel_tasks:
        print(f"[DEMO] Error during task cancellation phase: {e_cancel_tasks}")

    if signal_obj:
        print(f"[DEMO] Graceful shutdown due to signal {signal_obj.name} complete. Exiting.")
        # Use os._exit for a more immediate exit after cleanup if sys.exit hangs
        os._exit(130 if signal_obj == signal.SIGINT else 1)
    else:
        print("[DEMO] Graceful shutdown complete.")

    # If not exiting due to signal, normal exit path
    if not signal_obj and sys.platform != "win32": # sys.exit(0) can hang on some non-windows if tasks not fully joined
        pass # Allow script to terminate naturally
    elif not signal_obj:
        sys.exit(0)


async def main_logic():
    """
    Contains the main operational logic of the demo script.
    Signal handling and overarching try/except/finally are managed by main_wrapper.
    """
    global _shutdown_initiated # Allow checking this flag

    parser = argparse.ArgumentParser(description="Run the MP13 test adapter workflow")
    parser.add_argument("--config", type=str, default=None, help="Path or name of a custom config file (merged with default).")
    parser.add_argument("--base-model", type=str, default=None, help="Path to base model (category-relative unless prefixed with ./ or ../)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training and use existing test adapter")
    parser.add_argument("--training-steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--keep-checkpoints", action="store_true", help="Preserve existing checkpoints in the adapter output directory instead of deleting them.")
    parser.add_argument("--train-as-new", action="store_true", help="Force treating the adapter as new (ignore existing checkpoints).")
    
    # --- Model Loading & Quantization Arguments ---
    parser.add_argument("--base-model-dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Torch dtype for loading the base model.")
    
    parser.add_argument("--quantize-bits", type=str, default="none", choices=["none", "hqq", "eetq"], 
                        help="Quantization method for base model. Options: none, hqq, eetq.")
    # HQQ specific (for quantize_bits='hqq')
    parser.add_argument("--hqq-bits", type=int, default=4, choices=[2, 3, 4, 8], help="HQQ bits (if --quantize-bits='hqq').")
    parser.add_argument("--hqq-group-size", type=int, default=64, help="HQQ group size (if --quantize-bits='hqq').")
    parser.add_argument("--hqq-axis", type=int, default=1, choices=[0, 1], help="HQQ axis for quantization (if --quantize-bits='hqq').")

    parser.add_argument("--trainer-precision", type=str, default=None, choices=["bf16", "fp16", "fp32"], help="Compute precision for the HuggingFace Trainer.")
    parser.add_argument("--verification-data", type=str, default="test_mp13_engine_demo_training.json", help="Path to verification data JSON file (category-relative by default)")
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA r value for training")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha value for training (64 is for small training datasets)")
    parser.add_argument("--lora-dropout", type=float, default=None, help="LoRA dropout value for training")
    parser.add_argument("--lora-target-modules", nargs='+', default=None, help="LoRA target modules for training. If not set, modules are inferred from model architecture.")    
    parser.add_argument("--use-cache", type=str_to_bool, default=None, help="Enable KV caching for generation (True|False). Defaults to config when omitted.")
    parser.add_argument("--attn-implementation", type=str, default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"], help="Attention implementation for the model. Default: 'auto'.") # noqa
    parser.add_argument("--no-torch-compile", action="store_true", help="Disable torch.compile() globally for inference.")
    parser.add_argument("--no-tools-parse", action="store_true", help="If present, disables tool block parsing for all inference requests.")
    parser.add_argument("--disable-custom-pad-ids", action="store_true", help="If present, suppress modifying model special tokens during engine initialization.")
    parser.add_argument("--adapters-root", type=str, default=None, help="Root directory where adapters will be stored (model/precision/adapter).")
    parser.add_argument("--adapter-name", type=str, default="test_mp13_adapter", help="Logical adapter name to create/train/use.")
    parser.add_argument("--skip-verification", action="store_true", help="Skip verification step")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode after workflow")
    parser.add_argument("--static-kv-cache", type=str_to_bool, default=None, help="Enable static GPU KV cache for inference (True|False). Defaults to config when omitted.")
    parser.add_argument("--default-ctx", type=str, default="auto", help="Default context size for the engine (e.g., '2K', '4096', 'auto'). 'auto' derives from model. Default: auto.")
    parser.add_argument("--train-override-ctx", type=str, default=None, help="Override context size specifically for training (e.g., '512', '1K', 'auto'). 'auto' uses engine's default. Default: auto.")
    parser.add_argument("--device-map", type=str, default="auto", help="Device map for model loading (e.g., 'auto', 'cpu', or a JSON string like '{\"\":0}'). Default: 'auto'.")
    parser.add_argument("--concurrent-generate", type=int, default=None, help="Number of concurrent generation requests to allow. Defaults to config when omitted.")
    
    parser.add_argument("--log", type=str, default="warning", choices=["error", "warning", "info", "debug", "all", "none"], help="Set console logging level. Log file is always at DEBUG level. 'none' disables console output.")
    args = parser.parse_args()

    custom_config_path = resolve_custom_config_path(
        args.config,
        get_default_config_dir(),
    )

    default_config_source_path = get_default_config_path()
    default_config = load_json_config(default_config_source_path)
    if default_config is None and not custom_config_path:
        print(f"[DEMO] Missing default config at {default_config_source_path}. Run mp13chat --reconfigure to create it.")
        return
    if custom_config_path and not custom_config_path.exists():
        print(f"[DEMO] Warning: Custom config not found at {custom_config_path}; continuing with defaults.")

    resolved_config, resolver, ok = load_effective_config(
        default_config_path=default_config_source_path,
        custom_config_path=custom_config_path,
        cwd=Path.cwd(),
    )
    if not ok or resolved_config is None or resolver is None:
        print(f"[DEMO] Error: Could not load config from {default_config_source_path}.")
        return

    base_model_input = args.base_model or resolved_config.get("base_model_path")
    if not base_model_input:
        print("[DEMO] Warning: Base model path not configured.")
        base_model_input = input("Enter base model path (or press Enter to quit): ").strip()
        if not base_model_input:
            print("[DEMO] No base model path provided. Exiting.")
            return
    runtime_inputs = resolve_engine_inputs({"base_model_path": base_model_input}, resolver)
    abs_base_model_path = runtime_inputs["base_model_path"]
    if not Path(abs_base_model_path).exists():
        print(f"[DEMO] Error: Base model path '{abs_base_model_path}' not found.")
        return

    adapters_root_input = args.adapters_root or resolved_config.get("adapters_root_dir")
    if not adapters_root_input:
        print("[DEMO] Error: Adapters root path not set. Provide --adapters-root or configure adapters_root_dir.")
        return
    runtime_inputs = resolve_engine_inputs({"adapters_root_dir": adapters_root_input}, resolver)
    adapters_root_dir = runtime_inputs["adapters_root_dir"]
    if not Path(adapters_root_dir).exists():
        print(f"[DEMO] Error: Adapters root directory '{adapters_root_dir}' not found.")
        return

    runtime_inputs = resolve_engine_inputs({"dataset_path": args.verification_data}, resolver)
    verification_data_path = runtime_inputs["dataset_path"]
    if not Path(verification_data_path).exists():
        print(f"[DEMO] Error: Verification data file '{verification_data_path}' not found.")
        return

    runtime_inputs = resolve_engine_inputs({"dataset_path": "test_mp13_engine_demo_training.json"}, resolver)
    training_dataset_path = runtime_inputs["dataset_path"]
    if not Path(training_dataset_path).exists():
        print(f"[DEMO] Error: Training data file '{training_dataset_path}' not found.")
        return

    training_defaults = resolved_config.get("training_params") or {}
    effective_training_steps = args.training_steps if args.training_steps is not None else training_defaults.get("training_steps", 100)
    effective_trainer_precision = args.trainer_precision if args.trainer_precision is not None else training_defaults.get("trainer_precision", "bf16")
    effective_lora_r = args.lora_r if args.lora_r is not None else training_defaults.get("lora_r", 8)
    effective_lora_alpha = args.lora_alpha if args.lora_alpha is not None else training_defaults.get("lora_alpha", 64)
    effective_lora_dropout = args.lora_dropout if args.lora_dropout is not None else training_defaults.get("lora_dropout", 0.0)
    if args.lora_target_modules is not None:
        effective_lora_target_modules = args.lora_target_modules
    else:
        tmods = training_defaults.get("lora_target_modules")
        if isinstance(tmods, str):
            effective_lora_target_modules = [m.strip() for m in tmods.split(",") if m.strip()]
        else:
            effective_lora_target_modules = tmods
    effective_train_override_ctx = args.train_override_ctx if args.train_override_ctx is not None else training_defaults.get("train_override_ctx", "auto")

    adapter_logical_name_for_inference = args.adapter_name
    adapter_root_path: Optional[str] = None
    model_precision_folder: Optional[str] = None
    
    # --- Custom Colored Formatter for Console ---
    class ColoredFormatter(logging.Formatter):
        DIM = "\x1b[2m"
        YELLOW = "\x1b[33;20m"
        RED = "\x1b[31;20m"
        BOLD_RED = "\x1b[31;1m"
        RESET = "\x1b[0m"

        def __init__(self, fmt):
            super().__init__(fmt)
            self.FORMATS = {
                logging.DEBUG: self.DIM + fmt + self.RESET,
                logging.INFO: self.DIM + fmt + self.RESET,
                logging.WARNING: self.YELLOW + fmt + self.RESET,
                logging.ERROR: self.RED + fmt + self.RESET,
                logging.CRITICAL: self.BOLD_RED + fmt + self.RESET,
            }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno, self._fmt)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    # --- Configure Logging ---
    log_level_map = {
        "error": logging.ERROR, "warning": logging.WARNING, "info": logging.INFO,
        "debug": logging.DEBUG, "all": logging.DEBUG, "none": None,
    }
    console_log_level = log_level_map.get(args.log.lower(), logging.WARNING)

    # Create a temporary file for logging
    temp_log_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".log", prefix="mp13_demo_")
    log_file_path = temp_log_file.name
    temp_log_file.close() # Close the file so the handler can open it

    # Configure the logger instance imported from the engine
    logger.setLevel(logging.DEBUG) # Set the lowest level on the logger itself
    logger.propagate = False # Prevent messages from being passed to the root logger

    # Formatter strings
    log_format_string = '%(asctime)s - %(levelname)-8s - %(message)s'

    # File handler (always DEBUG)
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format_string))

    # Add handlers to the logger
    logger.addHandler(file_handler)

    if console_log_level is not None:
        # Console handler (level from args)
        # Create a reverse map to get the level name string without using the deprecated function part
        level_to_name_map = {v: k for k, v in log_level_map.items()}
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(ColoredFormatter(log_format_string))
        logger.addHandler(console_handler)
        logger.info(f"Console log level set to: {level_to_name_map.get(console_log_level, 'UNKNOWN').upper()}")

    print(f"[DEMO] Logging to file: {log_file_path}")
    logger.info("--- MP13 Demo Workflow Started ---")
    logger.info(f"Console log level set to: {logging.getLevelName(console_log_level)}")

    print(f"[DEMO] Adapters Root Directory: {adapters_root_dir}")
    print(f"[DEMO] Adapter Logical Name: {adapter_logical_name_for_inference}")

    
    # Parse device_map argument
    device_map_arg = args.device_map
    parsed_device_map: Union[str, Dict[str, Any]]
    if device_map_arg.strip().startswith('{'):
        try:
            parsed_device_map = json.loads(device_map_arg)
        except json.JSONDecodeError:
            print(f"[DEMO] Warning: Invalid JSON for --device-map argument: '{device_map_arg}'. Treating as a string.")
            parsed_device_map = device_map_arg
    else:
        parsed_device_map = device_map_arg

    print(f"[DEMO] Using device_map: {parsed_device_map} for global engine.")

    # Parse context size arguments for engine and training config
    parsed_default_ctx = parse_k_notation(args.default_ctx) # For GlobalEngineConfig.default_context_size
    parsed_train_override_ctx = parse_k_notation(effective_train_override_ctx) # For TrainingConfig.max_sequence_length

    engine_params = resolved_config.get("engine_params") or {}
    global_engine_config_dict = {
        "base_model_name_or_path": abs_base_model_path,
        "device_map": parsed_device_map,
        "initial_engine_mode": EngineMode.INFERENCE, # Start in inference, switch as needed
        "trust_remote_code": True,
        "base_model_torch_dtype": args.base_model_dtype,
        # New quantization parameters
        "quantize_bits": args.quantize_bits,
        "hqq_bits": args.hqq_bits,
        "hqq_group_size": args.hqq_group_size,
        "hqq_axis": args.hqq_axis,
        "default_context_size": parsed_default_ctx, # Engine will derive from model if this is None ("auto")
        "attn_implementation": args.attn_implementation,
        "use_torch_compile": not args.no_torch_compile,
    }
    if args.use_cache is not None:
        global_engine_config_dict["use_cache"] = args.use_cache
    elif "use_cache" in engine_params:
        global_engine_config_dict["use_cache"] = engine_params.get("use_cache")
    if args.static_kv_cache is not None:
        global_engine_config_dict["static_kv_cache"] = args.static_kv_cache
    elif "static_kv_cache" in engine_params:
        global_engine_config_dict["static_kv_cache"] = engine_params.get("static_kv_cache")
    if args.concurrent_generate is not None:
        global_engine_config_dict["concurrent_generate"] = args.concurrent_generate
    elif "concurrent_generate" in engine_params:
        global_engine_config_dict["concurrent_generate"] = engine_params.get("concurrent_generate")
    if args.no_tools_parse:
        global_engine_config_dict['no_tools_parse'] = True
    if args.disable_custom_pad_ids:
        global_engine_config_dict['disable_custom_pad_ids'] = True

    try:
        global_config = GlobalEngineConfig(**global_engine_config_dict)
    except Exception as e:
        print(f"[DEMO] Error creating GlobalEngineConfig: {e}")
        return

    print("\n--- Initializing Global Engine ---")
    init_global_response = await call_api("initialize-engine", global_config.model_dump())
    if init_global_response.get("status") != "success":
        print(f"[DEMO] Failed to initialize global engine: {init_global_response.get('message', 'Unknown error')}")
        return

    instance_id = (init_global_response.get("data") or {}).get("instance_id")
    if instance_id:
        set_default_resp = await call_api("set-default-engine", {"instance_id": instance_id})
        if set_default_resp.get("status") != "success":
            print(f"[DEMO] Warning: Failed to set default engine to '{instance_id}': {set_default_resp.get('message', 'Unknown error')}")
    else:
        print("[DEMO] Warning: Engine init response missing instance_id; default engine not set explicitly.")

    if args.disable_custom_pad_ids:
        print("\n[DEMO] WARNING: `disable_custom_pad_ids` is enabled. This may affect the stability and performance of trained adapters if the base model's token configuration is not optimal.")

    global_cfg_from_init = (init_global_response.get("data") or {}).get("global_config") or {}
    adapter_root_path = adapters_root_dir
    model_precision_folder = None
    if global_cfg_from_init:
        try:
            model_precision_folder, adapter_root_path = build_adapter_root_path(global_cfg_from_init, adapters_root_dir, adapter_logical_name_for_inference)
            print(f"[DEMO] Model-specific adapter folder: {model_precision_folder}")
            print(f"[DEMO] Adapter root path: {adapter_root_path}")
        except Exception as e:
            model_precision_folder = Path(abs_base_model_path).name
            adapter_root_path = os.path.join(adapters_root_dir, model_precision_folder, adapter_logical_name_for_inference)
            print(f"[DEMO] Warning: Failed to derive adapter root from init config ({e}). Falling back to {adapter_root_path}")
    else:
        print("[DEMO] Global config from engine init missing; using adapters root directly.")
    if not adapter_root_path:
        print("[DEMO] Error: Unable to derive adapter root path.")
        return

    # Get engine instance for direct callback subscription if needed
    engine_instance_for_callbacks = None
    if not args.skip_training: # Only needed if training
        engine_instance_for_callbacks = get_engine_instance_for_direct_use()


    path_to_actual_adapter_files: Optional[str] = None
    adapter_name = adapter_logical_name_for_inference
    
    if not args.skip_training:
        print("\n--- Starting Training Phase ---")

        probe_entries: List[Dict[str, Any]] = []
        has_existing_checkpoint = False
        probe_success = False
        try:
            list_args = {"root_folder": adapters_root_dir, "probe": adapter_logical_name_for_inference}
            probe_response = await call_api("list-all-adapters", list_args)
            if probe_response.get("status") == "success":
                probe_success = True
                probe_entries = (probe_response.get("data") or {}).get("adapters") or []
                has_existing_checkpoint = any(not entry.get("is_new", False) for entry in probe_entries)
                if has_existing_checkpoint:
                    print(f"[DEMO] Existing checkpoints detected for adapter '{adapter_logical_name_for_inference}'.")
                else:
                    print(f"[DEMO] No checkpoints found for adapter '{adapter_logical_name_for_inference}'.")
            else:
                print(f"[DEMO] Warning: list-all-adapters probe failed: {json.dumps(probe_response, indent=2)}")
        except Exception as e:
            print(f"[DEMO] Warning: list-all-adapters probe raised an exception: {e}")

        # 1. Set engine mode to TRAIN *before* adding a new adapter definition
        print("[DEMO] Setting engine mode to TRAIN for new adapter definition and training...")
        mode_set_train_resp = await call_api("check-set-mode", {"mode": EngineMode.TRAIN.value, "force": False}) # type: ignore
        if not (mode_set_train_resp.get("status") == "success" and mode_set_train_resp.get("data", {}).get("effective_mode") == EngineMode.TRAIN.value):
            print(f"[DEMO] Failed to set engine mode to TRAIN. Response: {json.dumps(mode_set_train_resp, indent=2)}")
            await call_api("shutdown-engine", {})
            return
        print("[DEMO] Engine mode set to TRAIN.")

        # 2. Add adapter definition to the engine (now that mode is TRAIN)
        print(f"[DEMO] Adding new adapter definition '{adapter_logical_name_for_inference}' to the engine...")
        force_new_adapter = args.train_as_new or (probe_success and not has_existing_checkpoint)
        adapter_config_kwargs = {
            "adapter_path": adapters_root_dir, # Provide the adapters root; engine will build model/precision/adapter folders
            "adapter_name": adapter_logical_name_for_inference,
            "adapter_type": AdapterType.LORA,
            "r": effective_lora_r,
            "lora_alpha": effective_lora_alpha,
            "lora_dropout": effective_lora_dropout,
            "target_modules": effective_lora_target_modules
        }
        if force_new_adapter:
            adapter_config_kwargs["is_new"] = True
            if has_existing_checkpoint and args.train_as_new:
                print(f"[DEMO] Training as new requested; existing checkpoints will be ignored.")
        adapter_config_for_add = AdapterConfig(**adapter_config_kwargs)
        load_adapter_response = await call_api("load-adapter", adapter_config_for_add.model_dump())
        if load_adapter_response.get("status") != "success":
            print(f"[DEMO] Failed to load adapter '{adapter_logical_name_for_inference}'. API Response: {json.dumps(load_adapter_response, indent=2)}")
            await call_api("shutdown-engine", {})
            return

        # Check if the adapter was newly created or loaded from an existing checkpoint
        if load_adapter_response.get("data", {}).get("is_new"):
            print(f"[DEMO] Adapter '{adapter_logical_name_for_inference}' is new. Preparing for fresh training.")
        else:
            print(f"[DEMO] Adapter '{adapter_logical_name_for_inference}' exists. Preparing for further training.")

        # 3. Set the added adapter as active for training (or confirm it's active)
        print(f"[DEMO] Setting adapter '{adapter_logical_name_for_inference}' as active for training...")
        set_active_response = await call_api("set-active-adapter", {"adapter_name": adapter_logical_name_for_inference})
        if set_active_response.get("status") != "success":
            print(f"[DEMO] Failed to set active adapter to '{adapter_logical_name_for_inference}'. API Response: {json.dumps(set_active_response, indent=2)}")
            await call_api("shutdown-engine", {})
            return
        
        # Parameters for the run_test_adapter_training function (excluding LoRA params which are now in AdapterConfig)
        training_run_params = {
            "steps": effective_training_steps,
            "trainer_precision": effective_trainer_precision,
            "train_override_ctx": parsed_train_override_ctx, # Pass the parsed value
        }
        final_checkpoint_path_from_training = await run_test_adapter_training(
            adapter_output_dir=adapter_root_path,
            adapter_logical_name=adapter_logical_name_for_inference,
            dataset_path=training_dataset_path,
            training_params=training_run_params,
            engine_instance_for_callbacks=engine_instance_for_callbacks,
            keep_existing_checkpoints=args.keep_checkpoints
        )
        if not final_checkpoint_path_from_training:
            print("[DEMO] Training failed or did not produce the expected output. Shutting down engine.")
            await call_api("shutdown-engine", {})
            return
        path_to_actual_adapter_files = final_checkpoint_path_from_training
        print(f"[DEMO] Training phase completed. Using adapter files from: {path_to_actual_adapter_files}")
        
        # Explicitly switch to INFERENCE mode if verification or interactive mode is next
        if not args.skip_verification or args.interactive:
            print("[DEMO] Training complete. Switching engine to INFERENCE mode for next phase...")
            mode_set_inference_resp = await call_api("check-set-mode", {"mode": EngineMode.INFERENCE.value, "force": False}) # type: ignore
            if not (mode_set_inference_resp.get("status") == "success" and mode_set_inference_resp.get("data", {}).get("effective_mode") == EngineMode.INFERENCE.value):
                print(f"[DEMO] Failed to set engine to INFERENCE mode after training. Response: {json.dumps(mode_set_inference_resp, indent=2)}")
            else:
                print(f"[DEMO] Engine mode set to INFERENCE. Newly trained adapter '{adapter_logical_name_for_inference}' should be available.")
        print("[DEMO] Training complete. Training configuration will be cleared when switching mode or shutting down.")
    else:
        print("\n--- Skipping Training Phase ---")
        # Use the adapters root; engine will scan for compatible adapters
        path_to_send_to_engine = adapters_root_dir
        print(f"[DEMO] Skipped training. Will attempt to load adapter '{adapter_logical_name_for_inference}' from adapters root: {path_to_send_to_engine}")
        print(f"[DEMO] The engine will attempt to find the latest compatible checkpoint within this path.")

        if not os.path.isdir(path_to_send_to_engine):
            print(f"[DEMO] Error: Adapter root path '{path_to_send_to_engine}' does not exist or is not a directory.")
            await call_api("shutdown-engine", {})
            return
        
        # If training is skipped, we need to explicitly add the adapter to the engine
        # The adapter_name in AdapterConfig can be None, engine will derive it from path.
        # Or we can provide adapter_logical_name_for_inference as an override.
        adapter_config_for_load = AdapterConfig(
            adapter_path=path_to_send_to_engine,
            adapter_name=adapter_logical_name_for_inference
        )
        load_adapter_response = await call_api("load-adapter", adapter_config_for_load.model_dump())
        if load_adapter_response.get("status") != "success":
            print(f"[DEMO] Failed to load pre-trained adapter. API Response: {json.dumps(load_adapter_response, indent=2)}")
            await call_api("shutdown-engine", {})
            return

        adapter_name = (load_adapter_response.get("data", {}) or {}).get("adapter_name")
        if not adapter_name:
            # Fallback to CLI logical name only if engine didn't return one:
            print(f"[DEMO] Warning: No adapter name in response, will use '{adapter_logical_name_for_inference}'")
            adapter_name = adapter_logical_name_for_inference
        elif adapter_name != adapter_logical_name_for_inference:
            print(f"[DEMO] Warning: Engine resolved adapter name to '{adapter_name}', which differs from the initial logical name '{adapter_logical_name_for_inference}'.")

        print(f"[DEMO] Pre-trained adapter '{adapter_logical_name_for_inference}' added to engine.")
        # The actual checkpoint path used by the engine will be in load_adapter_response["data"]["checkpoint_path"]
        path_to_actual_adapter_files = load_adapter_response.get("data", {}).get("checkpoint_path")
        if not path_to_actual_adapter_files:
             print(f"[DEMO] Warning: Engine loaded adapter but did not return specific checkpoint path. Using root: {path_to_send_to_engine}")
             path_to_actual_adapter_files = path_to_send_to_engine # Fallback for TestAdapterInference init
        else:
            print(f"[DEMO] Engine loaded adapter checkpoint from: {path_to_actual_adapter_files}")
    if not path_to_actual_adapter_files: # Should not happen if logic above is correct
        print("[DEMO] CRITICAL Error: Path to adapter files was not determined. Shutting down.")
        await call_api("shutdown-engine", {})
        return

    # Ensure engine is in INFERENCE mode before verification or interactive, if not already set after training.
    # TestAdapterInference.configure_engine_for_inference() will also attempt this.
    if not args.skip_verification or args.interactive:
        current_engine_status_resp = await call_api("get-engine-status", {}) # type: ignore
        current_mode = current_engine_status_resp.get("data", {}).get("engine_mode")
        if current_mode != EngineMode.INFERENCE.value:
            print(f"[DEMO] Engine not in INFERENCE mode (current: {current_mode}). Attempting to switch before verification/interactive.")
            await call_api("check-set-mode", {"mode": EngineMode.INFERENCE.value, "force": False}) # Best effort

    if not args.skip_verification:
        print("\n--- Starting Verification Phase ---")
        inference_handler = TestAdapterInference(
            adapter_path=path_to_actual_adapter_files, # This is now the checkpoint-X path
            adapter_name=adapter_name # The adapter to make active for this session
        )
        # configure_engine_for_inference calls check-set-mode(INFERENCE) and set-active-adapter
        config_success = await inference_handler.configure_engine_for_inference()
        if not config_success:
            print("[DEMO] Failed to configure engine for inference (verification). Shutting down engine.")
            await call_api("shutdown-engine", {})
            return

        with open(verification_data_path, "r") as f:
            verification_data = json.load(f)
        
        print(f"[DEMO] Loaded {len(verification_data)} verification examples.")
        verification_results = await inference_handler.batch_inference_compare(verification_data)

        # Results file path should be in the adapter root directory
        results_file_path = os.path.join(adapter_root_path, "verification_results.json")
        with open(results_file_path, "w") as f:
            json.dump(verification_results, f, indent=2)
        print(f"[DEMO] Verification results saved to: {results_file_path}")

        analyze_results(verification_results)
        # No explicit shutdown-inference-head.
        print("[DEMO] Verification phase completed.")
    else:
        print("\n--- Skipping Verification Phase ---")

    if args.interactive:
        print("\n--- Starting Interactive Mode ---")
        interactive_inference_handler = TestAdapterInference(
            adapter_path=path_to_actual_adapter_files,  # Use the checkpoint-X path
            adapter_name=adapter_logical_name_for_inference
        )
        config_interactive_success = await interactive_inference_handler.configure_engine_for_inference()
        if not config_interactive_success:
            print("[DEMO] Failed to configure engine for interactive inference. Shutting down engine.")
        else:
            print("\nType 'exit' to quit.")
            while True:
                system_message = input("System Message (optional): ")
                user_instruction = input("User Instruction (optional): ")

                if user_instruction.lower() == "exit":
                    break

                # Create the message list based on inputs
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                if user_instruction:
                    messages.append({"role": "user", "content": user_instruction})

                if not messages:
                    print("No valid messages provided. Please enter either a system message or a user instruction.")
                    continue
                
                # Base model response
                print("\n[DEMO] Generating response with base model (streaming)...")
                
                base_iterator = await interactive_inference_handler.run_inference_request(
                    [messages], # type: ignore
                    use_messages_format=True, # type: ignore
                    stream=True,
                    active_adapters_override=[], # Empty list for base model
                    generation_config_override=interactive_inference_handler.gen_config
                )
                
                sys.stdout.write("  Base Model Response: ") # type: ignore
                async for chunk in base_iterator:
                    if chunk.get("chunkType") == ChunkType.PROMPT_STARTED.value:
                        continue
                    elif chunk.get("chunkType") == ChunkType.STREAMING_CHUNK.value:
                        sys.stdout.write(chunk.get("chunk_text", ""))
                        sys.stdout.flush()
                    elif chunk.get("chunkType") == ChunkType.STREAMING_ENDED.value:
                        if chunk.get("was_truncated"):
                            sys.stdout.write("\n    \033[93mWarning: Base model output may have been truncated.\033[0m")
                        sys.stdout.write("\n")
                        interactive_inference_handler.print_metrics("Base Model (Interactive)", chunk)
                        break
                    elif chunk.get("chunkType") == ChunkType.ERROR.value:
                        sys.stdout.write(f"\n  Error during stream: {chunk.get('message')}\n")
                        break

                # Adapter response
                print("\n[DEMO] Generating response with adapter (streaming)...")
                
                adapter_iterator = await interactive_inference_handler.run_inference_request(
                    [messages], # type: ignore
                    use_messages_format=True, # type: ignore
                    stream=True,
                    active_adapters_override=[adapter_logical_name_for_inference],
                    generation_config_override=interactive_inference_handler.gen_config
                )
                
                sys.stdout.write("  Adapter Response: ") # type: ignore
                async for chunk in adapter_iterator:
                    if chunk.get("chunkType") == ChunkType.PROMPT_STARTED.value:
                        continue
                    elif chunk.get("chunkType") == ChunkType.STREAMING_CHUNK.value:
                        sys.stdout.write(chunk.get("chunk_text", ""))
                        sys.stdout.flush()
                    elif chunk.get("chunkType") == ChunkType.STREAMING_ENDED.value:
                        if chunk.get("was_truncated"):
                            sys.stdout.write("\n    \033[93mWarning: Adapter output may have been truncated.\033[0m")
                        sys.stdout.write("\n")
                        interactive_inference_handler.print_metrics(f"Adapter '{adapter_logical_name_for_inference}' (Interactive)", chunk)
                        break
                    elif chunk.get("chunkType") == ChunkType.ERROR.value:
                        sys.stdout.write(f"\n  Error during stream: {chunk.get('message')}\n")
                        break

            print("[DEMO] Exiting interactive mode.")
 
    # The final shutdown is now handled by graceful_shutdown,
    # called from main_wrapper's finally block or signal handler.
    # So, remove the explicit shutdown call from here.
    
async def main_wrapper():
    """
    Wrapper for the main logic that sets up signal handlers and ensures
    graceful shutdown on exit, cancellation, or error.
    """
    loop = asyncio.get_event_loop()

    try:
        await main_logic()
    except (asyncio.CancelledError, KeyboardInterrupt) as e: # KeyboardInterrupt for Ctrl+C during sync input()
        print(f"[DEMO] Main logic interrupted/cancelled ({type(e).__name__}). Ensuring graceful shutdown.")
    except Exception as e:
        print(f"[DEMO] An unhandled exception occurred in main_logic: {type(e).__name__} - {e}")
        traceback.print_exc()
    finally:
        print("[DEMO] Main wrapper's finally block reached. Ensuring graceful shutdown.")
        await graceful_shutdown() # Ensure cleanup happens

if __name__ == "__main__":
    asyncio.run(main_wrapper())
