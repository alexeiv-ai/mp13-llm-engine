# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""MP13 Engine - Custom trainer callbacks and quiet trainer implementation."""
from __future__ import annotations
import asyncio
import json
import os
import time
import math
import logging
import torch
from typing import TYPE_CHECKING
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any

from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerState, TrainerControl, TrainerCallback, DefaultFlowCallback, PrinterCallback, ProgressCallback
from transformers.trainer import Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .mp13_state import MP13State, TrainingStatus

if TYPE_CHECKING:
    from .mp13_state import MP13State

class CastLoRAWeights(TrainerCallback):
    """
    After Accelerate.prepare() finishes, cast *only the LoRA adapter weights*
    (and any other newly-added trainables) to fp16/bf16 so Flash-Attn 2
    never sees a float32 activation. Works with ShardedTrainer / FS-DP.
    This is particularly useful for models like Phi-3.
    """
    def __init__(self, logger:logging.Logger, dtype=torch.bfloat16):
        self.logger = logger
        self.dtype = dtype

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if not model:
            self.logger.warning("[CastLoRAWeights] Warning: Model not found in kwargs for on_train_begin.")
            return
        #self.logger.debug(f"[CastLoRAWeights] Checking trainable parameters against target dtype: {self.dtype}.")
        
        n_params_casted = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.dtype != self.dtype:
                param.data = param.data.to(self.dtype)
                n_params_casted += 1

        if n_params_casted > 0:
            self.logger.info(f"[CastLoRAWeights] Casted {n_params_casted} trainable parameter tensors to {self.dtype}.")


class LoRAGradMonitorCallback(TrainerCallback):
    """
    Lightweight callback that samples gradients on LoRA parameters to make sure
    the adapter being trained actually received gradient updates.
    """

    def __init__(
        self,
        logger: logging.Logger,
        target_substring: str = "lora_",
        probe_interval: int = 1,
        max_hooks: int = 32,
    ) -> None:
        self.logger = logger
        self.target_substring = target_substring
        self.probe_interval = max(1, probe_interval)
        self.max_hooks = max(1, max_hooks)
        self._observed = False
        self._steps_with_gradients: List[int] = []
        self._cached_model: Optional[torch.nn.Module] = None
        self._grad_norm_accum = 0.0
        self._grad_norm_samples = 0
        self._grad_norm_max = 0.0
        self._last_grad_sample = 0.0
        self._hook_handles: List[Any] = []
        self._hook_param_names: List[str] = []
        self._grad_param_refs: List[Tuple[str, torch.nn.Parameter]] = []
        self._observed_since_last_step = False
        self._last_recorded_step = -1

        self._last_logged_percent = -1 # For 10% interval logging
    @property
    def observed_gradients(self) -> bool:
        return self._observed

    @property
    def steps_with_gradients(self) -> List[int]:
        return list(self._steps_with_gradients)

    @property
    def avg_grad_norm(self) -> Optional[float]:
        if self._grad_norm_samples == 0:
            return None
        return self._grad_norm_accum / self._grad_norm_samples

    @property
    def max_grad_norm(self) -> Optional[float]:
        if self._grad_norm_samples == 0:
            return None
        return self._grad_norm_max

    @property
    def last_grad_sample(self) -> Optional[float]:
        if self._grad_norm_samples == 0:
            return None
        return self._last_grad_sample

    def _make_hook(self, param_name: str):
        def _hook(grad):
            if grad is None:
                return
            self._observed = True
            self._observed_since_last_step = True
            try:
                norm_val = float(grad.data.norm().item())
            except Exception:
                norm_val = None
            if norm_val is not None:
                self._grad_norm_accum += norm_val
                self._grad_norm_samples += 1
                self._grad_norm_max = max(self._grad_norm_max, norm_val)
                self._last_grad_sample = norm_val
        return _hook

    def _detach_hooks(self):
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles.clear()
        self._hook_param_names = []

    def _attach_hook(self, model: Optional[torch.nn.Module]):
        self._detach_hooks()
        self._grad_param_refs = []
        if model is None:
            return
        for name, param in model.named_parameters():
            if self.target_substring not in name or not param.requires_grad:
                continue
            handle = param.register_hook(self._make_hook(name))
            self._hook_handles.append(handle)
            self._hook_param_names.append(name)
            self._grad_param_refs.append((name, param))
            if len(self._hook_handles) >= self.max_hooks:
                break
        if not self._hook_handles:
            self.logger.warning("[LoRAGradMonitor] No trainable parameters matched '%s'.", self.target_substring)
        else:
            self.logger.info(
                "[LoRAGradMonitor] Gradient monitoring active on %d parameter(s).",
                len(self._hook_handles),
            )
        if not self._grad_param_refs and self._hook_param_names:
            self._grad_param_refs = [(name, param) for name, param in model.named_parameters() if name in self._hook_param_names]

    def _maybe_refresh_model(self, model: Optional[torch.nn.Module]):
        if model is None:
            return
        if self._cached_model is not model or not self._hook_handles:
            self._cached_model = model
            self._attach_hook(model)

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        self._maybe_refresh_model(kwargs.get("model"))

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        self._detach_hooks()
        self._cached_model = None

    def on_step_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        # In newer transformers, the wrapped model can change after on_train_begin.
        self._maybe_refresh_model(kwargs.get("model"))

    def on_pre_optimizer_step(self, args, state, control, **kwargs):  # type: ignore[override]
        if self._observed:
            return
        self._maybe_refresh_model(kwargs.get("model"))
        if not self._grad_param_refs:
            return
        for name, param in self._grad_param_refs:
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            try:
                norm_val = float(grad.data.norm().item())
            except Exception:
                norm_val = None
            if norm_val is None:
                continue
            self._observed = True
            self._observed_since_last_step = True
            self._grad_norm_accum += norm_val
            self._grad_norm_samples += 1
            self._grad_norm_max = max(self._grad_norm_max, norm_val)
            self._last_grad_sample = norm_val
            break

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if not self._observed_since_last_step:
            return
        step = getattr(state, "global_step", None)
        if step is None:
            step = 0
        if step == self._last_recorded_step:
            return
        if step % self.probe_interval != 0:
            self._observed_since_last_step = False
            return
        self._last_recorded_step = step
        self._observed_since_last_step = False
        self._steps_with_gradients.append(step)

        total_steps = getattr(state, "max_steps", 0)
        if total_steps > 0:
            current_percent = int((step / total_steps) * 10) # 0-10 scale for 10% chunks
            if current_percent > self._last_logged_percent:
                self._last_logged_percent = current_percent
                self.logger.debug(
                    "[LoRAGradMonitor] Observed gradients at step %d / %d (~%d%%).",
                    step, total_steps, current_percent * 10
                )

    def as_status_payload(self) -> Dict[str, Any]:
        return {
            "observed": self._observed,
            "probe_interval": self.probe_interval,
            "last_recorded_step": self._last_recorded_step,
            "steps_with_gradients": list(self._steps_with_gradients),
            "avg_grad_norm": self.avg_grad_norm,
            "max_grad_norm": self.max_grad_norm,
            "samples": self._grad_norm_samples,
            "monitor_params": list(self._hook_param_names),
        }


class QuietTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # The 'callbacks' kwarg passed to super().__init__ will typically be
        # [TrainingProgressCallback_instance] from mp13_train.py.
        super().__init__(*args, **kwargs)
        # After super().__init__(), self.callback_handler is initialized.
        # The Trainer's __init__ logic (and DefaultFlowCallback) might add
        # PrinterCallback and ProgressCallback based on TrainingArguments.
        # We explicitly remove them here to ensure quiet operation,
        # complementing the TrainingArguments settings from mp13_train.py.

        if self.callback_handler:
            # Filter out PrinterCallback and ProgressCallback.
            # We want to keep our custom TrainingProgressCallback and DefaultFlowCallback
            # (which Trainer adds if not present and controls flow but doesn't print itself).
            self.callback_handler.callbacks = [
                cb for cb in self.callback_handler.callbacks
                if not isinstance(cb, (PrinterCallback, ProgressCallback))
            ]

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Override _prepare_inputs to prevent moving inputs to a single device
        when the model is sharded with `device_map="auto"`. This is necessary because
        the default Trainer behavior can cause device mismatches for sharded models.
        This logic is adapted from the ShardedTrainer in train_lora.py.

        NOTE: This override is specifically for **sharded models** (where `is_parallelizable`
        is True). It does NOT prevent the Trainer from wrapping a non-sharded model
        in `torch.nn.DataParallel` if multiple GPUs are detected. That separate issue
        is handled in `mp13_train.py` by setting `training_args._n_gpu = 1`.
        """
        # If model is sharded, inputs should not be moved to a single device.
        # The `accelerate` hooks on the model will handle moving inputs to the correct
        # devices for each layer during the forward pass.
        if hasattr(self.model, "is_parallelizable") and self.model.is_parallelizable:
            # For sharded models, we return inputs as-is (on CPU) and let accelerate handle placement.
            return inputs
        
        # For non-sharded models, use the default behavior.
        return super()._prepare_inputs(inputs)


class TrainingProgressCallback(TrainerCallback):
    def __init__(self, server_state: "MP13State", start_time_wall: float, tokenizer_ref: "PreTrainedTokenizerBase"):
        self.state = server_state
        self.run_start_time_wall = start_time_wall
        self.tokenizer = tokenizer_ref
        self.initial_loss: Optional[float] = None
        self.first_log_call: bool = True
        # Capture the main event loop when the callback is instantiated
        self.main_loop = asyncio.get_event_loop()

    # Note on QuietTrainer._log():
    # The base Trainer.log() method (called during the training loop) dispatches to
    # self.callback_handler.on_log(). If PrinterCallback and ProgressCallback are
    # removed from the handler (as done in QuietTrainer.__init__), Trainer.log()
    # will not cause console output through them.
    # The base Trainer._log() method (called during evaluation/prediction) also
    # eventually calls Trainer.log() if its `iterator` is None (which it should be
    # if TrainingArguments.disable_tqdm=True, set in mp13_train.py).
    # Thus, overriding _log in QuietTrainer is not strictly necessary if the
    # callback filtering and TrainingArguments are correctly set.
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # MP13State.set_training_started() will handle notifications
        asyncio.run_coroutine_threadsafe(self.state.set_training_started(), self.main_loop)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs): # type: ignore
        """Called after each training step for responsive stop requests."""
        if (self.state._graceful_stop_requested or self.state._training_cancelled or self.state._shutting_down) and not control.should_training_stop:
            self.state.logger.info("Graceful stop request detected in on_step_end. Setting control.should_training_stop = True.")
            control.should_training_stop = True
            return # Do not proceed with other actions if stopping

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        """Called when logs are recorded (e.g., at logging_steps)."""
        #server_state.logger.debug(f"[ENGINE_CALLBACK_DEBUG] TrainingProgressCallback.on_log called. Step: {state.global_step}, Logs: {logs}") # DEBUG
        loss = logs.get("loss") if logs else None
        lr = logs.get("learning_rate") if logs else None # type: ignore
        grad_norm = logs.get("grad_norm") if logs else None # Get grad_norm

        # Epoch here might be a float representing progress through current epoch
        if self.first_log_call and loss is not None:
            self.initial_loss = loss
            self.first_log_call = False
            # self.state.logger.info(f"Initial loss estimate captured at step {state.global_step}: {self.initial_loss:.4f}")
        # MP13State.update_training_progress will handle notifications
        if not (self.state._graceful_stop_requested or self.state._training_cancelled or self.state._shutting_down):
            asyncio.run_coroutine_threadsafe(self.state.update_training_progress(state.global_step, state.epoch, loss, lr, grad_norm), self.main_loop)
        else:
            self.state.logger.info("Skipping update_training_progress in on_log due to stop/cancel/shutdown request.")
            control.should_training_stop = True # Ensure stop is propagated

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        saved_model_path = kwargs.get('model_path', os.path.join(args.output_dir, f"checkpoint-{state.global_step}"))
        is_final_configured_save = (args.save_strategy == "steps" and state.global_step == args.save_steps and args.save_steps == state.max_steps)

        # Only log save if not in shutdown process
        if not self.state._shutting_down:
            self.state.logger.info(f"Trainer on_save: Saving final model checkpoint at step {state.global_step} as configured. Path: {saved_model_path}")
        elif args.save_strategy == "no":
            self.state.logger.warning(f"Trainer on_save: Callback triggered with save_strategy='no' at step {state.global_step}. Path: {saved_model_path}. This is unexpected.")
        else:
            # Covers other scenarios: epoch saves, intermediate step saves, or saves from graceful stop.
            self.state.logger.info(f"Trainer on_save: Model checkpoint saved at step {state.global_step}. Path: {saved_model_path}. Strategy: {args.save_strategy}, Configured SaveSteps: {args.save_steps}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs): # type: ignore
        self.state.logger.info("Training loop finished by Trainer.")
        # Only update state if not already in an error/stopped state and not shutting down
        if (self.state._graceful_stop_requested or self.state._training_cancelled) and self.state.training_status not in [TrainingStatus.ERROR, TrainingStatus.STOPPED] and not self.state._shutting_down:
            asyncio.run_coroutine_threadsafe(self.state.set_training_stopped("Training gracefully stopped as confirmed by Trainer on_train_end."), self.main_loop)
            self.state.logger.info("Graceful stop confirmed by on_train_end. Status set to STOPPED.")
        elif self.state.training_status not in [TrainingStatus.STOPPED, TrainingStatus.ERROR]:
            # Do not set COMPLETED here. It will be set after saving.
            self.state.logger.info("Trainer on_train_end: Training loop appears to have finished normally.")
