# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for MP13 engine."""
import os
from dataclasses import dataclass, field
import asyncio
import threading, types
from pathlib import Path
import time, datetime, json, shutil
import torch, gc, logging
import uuid
from typing import Optional, TYPE_CHECKING, Dict, Any, List, Union, Callable
from transformers.generation.stopping_criteria import StoppingCriteria 
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.generation.streamers import TextIteratorStreamer
from transformers import AutoConfig, GenerationConfig
from contextlib import contextmanager
from torch._inductor import config as ic
from safetensors.torch import save_file
from peft import get_peft_model_state_dict
from peft.tuners.lora import LoraConfig

import numpy as np              #for assessment
import matplotlib
matplotlib.use('Agg') # Ensure this is set before importing pyplot
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .mp13_state import MP13State
    from . import mp13_state
    from .mp13_config import ColumnsConfig, DatasetTags

from .mp13_config import ParserProfile

class StoringTextIteratorStreamer(TextIteratorStreamer):
    """TextIteratorStreamer that (optionally) mirrors base prompt-skipping
    for metrics, and lets you control EOS/PAD & special-token handling.

    Args:
        tokenizer: HF tokenizer (first and only positional arg).
        skip_prompt: forward only *generated* tokens (default: True).
        skip_special_tokens: if True, special tokens are not decoded.
        clean_up_tokenization_spaces: pass-through to parent.
        drop_eos_and_pad: if True, remove EOS & PAD from visible output text.
        extra_stop_ids: optional extra token IDs to treat as EOS/PAD for output filtering.
        **kwargs: forwarded to TextIteratorStreamer (e.g. timeout, etc).
    """
    def __init__(
        self,
        logger: logging.Logger,
        tokenizer,
        *,
        skip_prompt: bool = True,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
        drop_eos_and_pad: bool = False,
        extra_stop_ids: Optional[Union[int, List[int], set, tuple]] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            skip_prompt=skip_prompt,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        self.logger = logger
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        # Store raw token ids we *receive*, except the initial prompt chunk if skip_prompt=True.
        self.generated_ids: List[int] = []
        # Remember whether we've seen (and skipped) the first chunk already.
        self._skipped_prompt_chunk: bool = not skip_prompt  # if skip_prompt=False, don't skip anything
        # Token IDs to optionally filter from *visible output* (never affects generated_ids).
        self._stop_ids = self._initialize_stop_ids()
        if extra_stop_ids is not None:
            if isinstance(extra_stop_ids, int):
                self._stop_ids.add(extra_stop_ids)
            else:
                for tid in extra_stop_ids:
                    if isinstance(tid, int):
                        self._stop_ids.add(tid)
        self._drop_eos_and_pad: bool = bool(drop_eos_and_pad or skip_special_tokens)
        # Diagnostics
        self.eos_token_detected: Optional[int] = None
        self.eos_token_decoded: Optional[str] = None

    def _initialize_stop_ids(self) -> set:
        """Pre-calculates the set of stop token IDs from the tokenizer."""
        stop_ids = set()
        if self.tokenizer:
            eos_token_id = self.tokenizer.eos_token_id
            if isinstance(eos_token_id, int):
                stop_ids.add(eos_token_id)
            elif isinstance(eos_token_id, list):
                stop_ids.update(eos_token_id)
            if self.tokenizer.pad_token_id is not None:
                stop_ids.add(self.tokenizer.pad_token_id)
        return stop_ids

    def put(self, value):
        # HF will push the prompt first, then new tokens; `skip_prompt=True` only hides it from *output*.
        # We mirror that for metrics by skipping that first chunk in `generated_ids`.
        flat = value.flatten()

        # Decide what to forward for *visible output*.
        forward_list: List[torch.Tensor] = []
        for tok in flat:
            tok_id = int(tok.item())

            # Diagnostics: capture first EOS/PAD observed (regardless of drops)
            if self.eos_token_detected is None and tok_id in self._stop_ids:
                self.eos_token_detected = tok_id
                self.eos_token_decoded = self.tokenizer.decode(
                    [tok_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
                )
                self.logger.info(f"EOS/PAD token '{self.eos_token_decoded}', position (post-metrics): {len(self.generated_ids)}")

            # Output filtering (applies only to what users *see*).
            if self._drop_eos_and_pad and tok_id in self._stop_ids:
                # Drop from outgoing stream; still eligible for metrics depending on prompt chunk.
                pass
            else:
                forward_list.append(tok.detach().clone())

        # Metrics: store everything *except* the initial prompt chunk if skip_prompt=True.
        if self._skipped_prompt_chunk:
            for tok in flat:
                self.generated_ids.append(int(tok.item()))
        else:
            # This was the prompt chunk; mark as skipped for metrics and do NOT record it.
            self._skipped_prompt_chunk = True

        # Hand off to parent with the (possibly) filtered tensor for output.
        if forward_list:
            out = torch.stack(forward_list) if len(forward_list) > 1 else forward_list[0].unsqueeze(0)
            super().put(out)

class CancellableStoppingCriteria(StoppingCriteria):
    """
    Stops generation as soon as `cancel_event.is_set()` returns True.
    Optionally stops after generating only one new token.
    """
    def __init__(self, logger: logging.Logger, cancel_event: threading.Event, max_new_tokens: Optional[int] = None, prompt_length: int = 0) -> None:
        super().__init__()
        self.logger = logger
        self._cancel_event = cancel_event
        self._max_new_tokens = max_new_tokens  # Optional token limit
        self._prompt_length = prompt_length    # longest unpadded prompt length in tokens for the left padded batch.
        self.cancellation_triggered = False
        self.max_tokens_triggered = False

    def __call__( 
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs
    ) -> Union[bool, torch.BoolTensor]:
        if self._cancel_event.is_set():        # Event.is_set() is the std way 
            self.logger.info(f"-- cancel flag SEEN at step: {input_ids.shape[-1]} --") 
            self.cancellation_triggered = True
            return True                        # tells HF to break the loop
        
        if self._max_new_tokens is not None:
            new_tokens = input_ids.shape[-1] - self._prompt_length
            if new_tokens >= self._max_new_tokens:
                self.max_tokens_triggered = True
                return True        
        return False 

    def set_max_new_tokens(self, max_new_tokens: int):
        """Allows dynamically updating the token limit after initialization."""
        self._max_new_tokens = max_new_tokens

def round_floats(obj: Any, precision: int = 2) -> Any:
    """Recursively rounds float values in a nested data structure (dict, list)."""
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(i, precision) for i in obj]
    return obj

def _clear_gpu_mem():
    """Helper to clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.ipc_collect()

def get_modified_generation_config(original_config: GenerationConfig, **kwargs_to_update: Any) -> GenerationConfig:
    """Creates a safe, mutable copy of a GenerationConfig and applies updates."""
    config_dict = original_config.to_dict()
    config_dict.update(kwargs_to_update)

    # If max_new_tokens is set, prefer it over max_length to avoid conflicts.
    if config_dict.get("max_new_tokens") is not None and config_dict.get("max_length") is not None:
        config_dict.pop("max_length", None)

    # If sampling is disabled, remove sampling-specific parameters to avoid warnings from transformers.
    if config_dict.get("do_sample") is False:
        config_dict.pop("temperature", None)
        config_dict.pop("top_p", None)
        config_dict.pop("top_k", None)
        config_dict.pop("presence_penalty", None)
        config_dict.pop("frequency_penalty", None)

    return GenerationConfig(**config_dict)

def ensure_pad_is_eos(generation_config: GenerationConfig) -> GenerationConfig:
    """
    Modifies a GenerationConfig to ensure the pad_token_id is included
    in the eos_token_id list, making generation stop on PAD tokens.
    Returns the modified config.
    """
    eos_ids = generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = []
    elif isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    # Make a copy to avoid modifying a list that might be shared across configs
    eos_ids = list(eos_ids)

    pid = generation_config.pad_token_id
    if pid is not None and pid not in eos_ids:
        eos_ids.append(pid)

    generation_config.eos_token_id = eos_ids
    return generation_config

# --- Format Prompt Messages ---
@dataclass(frozen=True)
class FormattedPrompt:
    """Container for a formatted prompt and its token count (if available)."""
    text: str
    token_count: Optional[int] = None


def format_prompt_messages(
    logger:logging.Logger,
    example: Dict[str, Any],
    columns: 'ColumnsConfig',
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    parser_profile: Optional[ParserProfile] = None,
    tags: Optional['DatasetTags'] = None, # Keep for fallback
    tools: Optional[List[Union[Dict[str, Any], Callable]]] = None, # Accept callables
    add_generation_prompt: bool = False,  # Key parameter for inference vs. SFT
    continue_final_message : bool = False,
    strip_empty_system_prompt: bool = False,
    empty_system_prompt_template: Optional[str] = None,
    strip_eos_token: bool = False
) -> FormattedPrompt:
    """
    Formats a prompt using the messages format similar to train_lora.py.
    Uses the model's chat template if available, otherwise falls back to a custom format.
    """
    # When continuing a message, we never want to add a new generation prompt.
    if continue_final_message:
        add_generation_prompt = False

    # Initialize actual_messages list
    actual_messages: List[Dict[str, str]] = []

    # --- FIX: Handle both list-of-dicts and dict-with-messages-key ---
    # The 'example' can be the list of messages directly, or a dict containing that list.
    if isinstance(example, list):
        messages_from_col = example
    else:
        # Try to get messages from the 'messages' column key specified in ColumnsConfig
        messages_from_col = example.get(columns.messages) if hasattr(columns, 'messages') and columns.messages else None

    if isinstance(messages_from_col, list):
        actual_messages = [msg for msg in messages_from_col if isinstance(msg, dict)] # Basic validation
    
    if not actual_messages: # If 'messages' column was not found, empty, or invalid
        # Fallback to using prompt/query as a user message if messages not found
        user_content_parts = []
        prompt_key = getattr(columns, 'prompt', None)
        query_key = getattr(columns, 'query', None)

        if prompt_key and example.get(prompt_key):
            user_content_parts.append(str(example.get(prompt_key)))
        if query_key and example.get(query_key):
            user_content_parts.append(str(example.get(query_key)))
        
        final_user_content = "\n".join(user_content_parts).strip()
        if final_user_content:
            user_role_name = getattr(tags, 'user_tag', 'user') if tags else 'user'
            actual_messages = [{"role": user_role_name, "content": final_user_content}]
        else:
            logger.info("Warning: No messages or prompt/query content found in example for format_prompt_messages.")
            return FormattedPrompt(text="", token_count=None)
    
    # Use default tags if not provided
    if tags is None:
        from .mp13_config import DatasetTags 
        tags = DatasetTags()

    # --- NEW: Handle phi-4 style tool injection into system prompt ---
    # --- FIX: Guard against parser_profile being a dict ---
    template_accepts_tools = False
    tool_handling_hint: Optional[str] = None

    if parser_profile and tools:
        # This check is now redundant because the caller in mp13_infer.py
        # ensures it's an object, but it's a good defensive measure.
        if isinstance(parser_profile, dict):
            parser_profile = ParserProfile(**parser_profile)

        template_accepts_tools = getattr(parser_profile, "template_accepts_tools_arg", False)
        tool_handling_hint = getattr(parser_profile, "tool_handling_hint", None)

        if parser_profile.tools_in_system_prompt:
            # Find the first system message and append the tools to it.
            for msg in actual_messages:
                if msg.get("role") == "system":
                    tools_json_str = json.dumps(tools, indent=2)
                    # The template expects a 'tools' key in the message dict.
                    msg["tools"] = tools_json_str
                    break
        else:
            # If the template ignores the `tools` argument, inject a plain-text tools block.
            if tools and not template_accepts_tools:
                tools_json_str = json.dumps(tools, indent=2)
                injected = False
                for msg in actual_messages:
                    if msg.get("role") == "system":
                        msg["content"] = (msg.get("content", "") or "") + f"\n\n# Tools\n{tools_json_str}"
                        injected = True
                        break
                if not injected:
                    actual_messages.insert(0, {"role": "system", "content": f"# Tools\n{tools_json_str}"})
                hint = f" {tool_handling_hint}" if tool_handling_hint else ""
                logger.info("Chat template ignored `tools`; injected tools JSON into system message." + hint)
    # --- END NEW ---

    # Note: tool_call normalization is handled closer to serialization.
    
    # Use tokenizer's chat template if available and valid
    if tokenizer and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        if tools and not template_accepts_tools and not (parser_profile.tools_in_system_prompt if parser_profile else False):
            # Warn when the template will ignore the tools arg so callers know to inject manually.
            hint = f" {tool_handling_hint}" if tool_handling_hint else ""
            logger.warning("Chat template does not appear to consume `tools`; they may be ignored when formatting." + hint)

        try:
            formatted_prompt = tokenizer.apply_chat_template(
                actual_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt, 
                continue_final_message=continue_final_message,
                tools=tools # Pass tools to the template
            )
            formatted_prompt = str(formatted_prompt)
        except Exception as e:
            raise ValueError(f"apply_chat_template failed: {e}") from e
    else:
        # Custom fallback formatting
        formatted_prompt_parts = []
        for i, message_dict in enumerate(actual_messages):
            # Default to 'role' and 'content' if specific tags are not found or 'tags' is None
            role_key = getattr(tags, 'role_tag', 'role')
            content_key = getattr(tags, 'content_tag', 'content')
            
            role = message_dict.get(role_key)
            # Check for content existence. An empty string is valid content.
            content = message_dict.get(content_key)
            
            if not role or content is None:
                logger.warning(f"Warning: Skipping message with missing role or content in fallback: {message_dict}")
                continue
            
            # Build the message part.
            # For continue_final_message, we treat the last message differently.
            is_last_message = (i == len(actual_messages) - 1)

            if role == tags.system_tag:
                # System prompt is always at the start and fully formed.            
                formatted_prompt_parts.append(f"<|system|>\n{content}\n")
            elif role == tags.user_tag:
                formatted_prompt_parts.append(f"<|user|>\n{content}\n")
            elif role == tags.assistant_tag:
                # For an assistant message, we only add the closing marker if it's NOT the last message we are continuing.
                if is_last_message and continue_final_message:
                    formatted_prompt_parts.append(f"<|assistant|>\n{content}") # No closing marker
                else:
                    formatted_prompt_parts.append(f"<|assistant|>\n{content}\n")
            else:
                logger.warning(f"Warning: Unknown role '{role}' in custom message formatting. Using as-is: <|{role}|>")
                formatted_prompt_parts.append(f"<|{role}|>\n{content}\n") # Generic fallback for unknown roles

        formatted_prompt = "\n".join(formatted_prompt_parts)

        if add_generation_prompt:
            formatted_prompt += "<|assistant|>\n"

    # Optional stripping of the placeholder empty system prompt the caller inserted.
    if strip_empty_system_prompt and empty_system_prompt_template:
        if formatted_prompt.startswith(empty_system_prompt_template):
            formatted_prompt = formatted_prompt[len(empty_system_prompt_template):]

    # Optional stripping of trailing EOS added by some templates when continuing a generation.
    if strip_eos_token and tokenizer is not None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token and formatted_prompt.endswith(eos_token):
            formatted_prompt = formatted_prompt[:-len(eos_token)]

    token_count = None
    if tokenizer is not None:
        try:
            token_count = len(tokenizer.encode(formatted_prompt))
        except Exception as e:
            logger.warning(f"Warning: Failed to tokenize formatted prompt for counting: {e}")
            token_count = None

    return FormattedPrompt(text=formatted_prompt, token_count=token_count)

def assess_training_quality(
    losses: List[float], 
    learning_rates: List[float], 
    steps: List[int],
    grad_norms: List[float]
):
    """Generate a more detailed and realistic assessment of training quality."""
    
    assessment = [f"Training Assessment Summary (Data points: {len(losses)}):"]
    if not losses or len(losses) < 2:
        assessment.append("Insufficient data to assess training quality (less than 2 data points).")
        return "\n".join(assessment)

    # --- 1. Critical Failure Checks ---
    has_nan_loss = any(np.isnan(l) for l in losses)
    has_inf_loss = any(np.isinf(l) for l in losses)
    has_nan_grad = any(np.isnan(g) for g in grad_norms)

    if has_nan_loss or has_inf_loss or has_nan_grad:
        verdict = "FAILED"
        assessment.append(f"- Verdict: {verdict}")
        if has_nan_loss or has_inf_loss:
            assessment.append("- Failure Reason: Loss became NaN or Infinity.")
        if has_nan_grad:
            assessment.append("- Failure Reason: Gradient norm became NaN.")
        assessment.append("- Suggestion: This is a critical failure, often caused by an unstable learning rate (too high) or numerical precision issues (e.g., using fp16 with a model that requires bf16). Lower the learning rate significantly or switch to a more stable precision like bf16 or fp32.")
        return "\n".join(assessment)

    # --- 2. Data & Metric Preparation ---
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = initial_loss - final_loss
    loss_reduction_percentage = (loss_reduction / initial_loss) * 100 if initial_loss > 1e-9 else 0
    
    # --- 3. Model Collapse Check ---
    # Check if the last 50% of the run has near-zero loss
    is_collapsed = False
    if len(losses) > 4:
        midpoint_idx = len(losses) // 2
        tail_losses = losses[midpoint_idx:]
        if all(abs(l) < 1e-5 for l in tail_losses):
            is_collapsed = True

    # --- 4. Stability and Plateau Analysis ---
    # Use losses after the first point for stability calculation to be less sensitive to initial drop
    losses_for_stability = losses[1:] if len(losses) > 1 else losses
    
    # Plateau check (on last 25% of data)
    plateaued = False
    if len(losses) > 4:
        last_quarter_idx = int(len(losses) * 0.75)
        recent_losses = losses[last_quarter_idx:]
        if len(recent_losses) > 1:
            loss_std_recent = np.std(recent_losses)
            loss_mean_recent = np.mean(recent_losses)
            # Plateau if relative standard deviation is < 1%
            plateau_variation_coeff = (loss_std_recent / loss_mean_recent) if loss_mean_recent > 1e-9 else 0
            plateaued = plateau_variation_coeff < 0.01

    # Instability check (on all but the first point)
    unstable = False
    if len(losses_for_stability) > 1:
        loss_std_stable = np.std(losses_for_stability)
        loss_mean_stable = np.mean(losses_for_stability)
        # Unstable if relative standard deviation is > 50%
        instability_variation_coeff = (loss_std_stable / loss_mean_stable) if loss_mean_stable > 1e-9 else 0
        unstable = instability_variation_coeff > 0.50

    # --- 5. Build Assessment Sections ---
    assessment.append("\n--- Loss Analysis ---")
    assessment.append(f"- Steps range: {steps[0]} to {steps[-1]}" if steps else "- Steps data unavailable")
    assessment.append(f"- Initial loss: {initial_loss:.4f}")
    assessment.append(f"- Final loss: {final_loss:.4f}")
    assessment.append(f"- Loss reduction: {loss_reduction:.4f} ({loss_reduction_percentage:.2f}%)")

    LOW_LOSS_THRESHOLD = 0.1
    is_low_initial_loss = initial_loss < LOW_LOSS_THRESHOLD

    if is_collapsed:
        assessment.append("- Observation: Loss collapsed to near-zero. This is a strong indicator of model collapse or catastrophic forgetting, not successful learning.")
    elif is_low_initial_loss:
        assessment.append("- Observation: Initial loss is very low, suggesting the base model is already well-aligned with the data. Absolute loss reduction is more meaningful than percentage.")
    elif loss_reduction_percentage > 85:
        assessment.append("- Observation: Excellent loss reduction achieved, indicating a strong learning signal.")
    elif loss_reduction_percentage > 65:
        assessment.append("- Observation: Good loss reduction achieved, indicating the model was learning effectively.")
    elif loss_reduction_percentage > 30:
        assessment.append("- Observation: Moderate loss reduction achieved, showing some learning occurred.")
    elif loss_reduction_percentage > 10:
        assessment.append("- Observation: Questionable loss reduction. The model showed some learning, but it was limited.")
    else:
        assessment.append("- Observation: Minimal or negligible loss reduction. The model may have already been converged, the learning rate may be too low, or the training was too short.")

    assessment.append("\n--- Stability & Convergence ---")
    if unstable:
        assessment.append(f"- Stability: UNSTABLE (loss variation: {instability_variation_coeff:.1%}). High variation can harm model quality.")
    else:
        assessment.append("- Stability: Training loss progression was STABLE.")
    
    if plateaued and not is_collapsed:
        assessment.append("- Convergence: Loss plateaued, suggesting convergence.")
    elif not is_collapsed:
        assessment.append("- Convergence: Loss still decreasing; more training may improve results.")

    # --- 6. Other Metrics ---
    assessment.append("\n--- Other Metrics ---")
    if learning_rates and len(learning_rates) > 1 and any(lr != learning_rates[0] for lr in learning_rates):
        initial_lr = learning_rates[0]
        final_lr = learning_rates[-1]
        decay_info = f"Learning rate decayed from {initial_lr:.2e} to {final_lr:.2e}."
        if initial_lr > 1e-9 and final_lr > 1e-12: # Avoid division by zero or huge numbers from tiny final LRs
            fold_decrease = initial_lr / final_lr if final_lr > 0 else float('inf')
            if fold_decrease > 1.1: # Only mention if it's a meaningful decrease
                decay_info += f" (a decrease of ~{fold_decrease:,.0f}x)"
        assessment.append(f"- {decay_info}")
    elif learning_rates:
        assessment.append(f"- Constant or near-constant learning rate of ~{learning_rates[0]:.2e} was used.")

    valid_grad_norms = [g for g in grad_norms if not np.isnan(g)]
    if valid_grad_norms:
        avg_grad_norm = np.mean(valid_grad_norms)
        # Format the number for readability
        if avg_grad_norm > 10000:
            grad_norm_str = f"{avg_grad_norm:.4e}"
        else:
            grad_norm_str = f"{avg_grad_norm:,.4f}"
        
        grad_norm_assessment = f"Gradient Norm: Averaged {grad_norm_str} during the run."
        if avg_grad_norm > 1000:
            grad_norm_assessment += " (High value suggests potential gradient explosion)."
        assessment.append(f"- {grad_norm_assessment}")
    else:
        assessment.append("- Gradient Norm: No valid gradient norm data was recorded.")

    # --- 7. Final Verdict ---
    verdict = "UNKNOWN"
    if is_collapsed:
        verdict = "WARNING"
    elif unstable:
        verdict = "WARNING"
    elif is_low_initial_loss:
        if loss_reduction > 0.001 and not unstable:
            verdict = "GOOD"
        elif not unstable:
            verdict = "ACCEPTABLE"
        else: # Loss increased or was unstable
            verdict = "POOR"
    elif loss_reduction_percentage < 10:
        verdict = "POOR"
    elif loss_reduction_percentage < 30:
        verdict = "QUESTIONABLE"
    elif loss_reduction_percentage < 65:
        verdict = "MODERATE"
    elif loss_reduction_percentage < 85:
        verdict = "GOOD"
    else: # >= 85
        verdict = "EXCELLENT"
    
    assessment.append("\n--- Final Verdict ---")
    assessment.append(f"- Overall Quality: {verdict}")
    if verdict == "EXCELLENT":
        assessment.append("- Suggestion: The training run appears highly successful. The model learned very effectively and stably.")
    elif verdict == "GOOD":
        if is_low_initial_loss:
            assessment.append("- Suggestion: The training run appears successful. The model refined its knowledge on an already well-aligned dataset.")
        else:
            assessment.append("- Suggestion: The training run appears successful. The model learned effectively and stably.")
    elif verdict == "ACCEPTABLE":
        assessment.append("- Suggestion: The training was stable but showed little improvement, possibly because the model was already converged or the dataset was too easy.")
    elif verdict == "MODERATE":
        assessment.append("- Suggestion: The training was stable and showed moderate learning. Further tuning could yield improvements.")
    elif verdict == "QUESTIONABLE":
        assessment.append("- Suggestion: The training was stable but learning was limited. Consider reviewing hyperparameters or increasing training steps.")
    elif verdict == "WARNING":
        assessment.append("- Suggestion: The training run has warning signs (instability or collapse). Review hyperparameters, especially the learning rate. The resulting adapter may not perform well.")
    elif verdict == "POOR":
        assessment.append("- Suggestion: The model showed very little learning. This could be due to a very low learning rate, a dataset the model has already mastered, or insufficient training steps.")

    return "\n".join(assessment)

def generate_training_report(
    logger: logging.Logger,
    report_dir: str,
    historical_steps: List[Union[int, float]],
    historical_loss: List[float],
    historical_lr: List[float],
    historical_grad_norm: List[float]
) -> str:
    """
    Generates and saves a training metrics plot and an assessment file.

    Args:
        report_dir: The directory where the report files will be saved.
        historical_steps: A list of steps for which metrics were recorded.
        historical_loss: A list of loss values corresponding to the steps.
        historical_lr: A list of learning rate values corresponding to the steps.
        historical_grad_norm: A list of gradient norm values corresponding to the steps.
        logger: The logger instance to use.
    """
    
    #logger.debug(f"Generating training report in: {report_dir}")

    steps = list(historical_steps)
    losses = list(historical_loss)
    learning_rates = list(historical_lr)
    grad_norms = list(historical_grad_norm)
    file_content = "error"

    if not steps or not losses or len(steps) < 2: # Need at least 2 points for a meaningful plot/assessment
        logger.warning("Insufficient historical data (less than 2 points for steps or losses) for training report. Skipping plot/assessment.")
        assessment_path = os.path.join(report_dir, 'training_assessment.txt')
        file_content = "No training metrics (or insufficient data) were collected to generate a report.\n" \
                        "Check logging_steps, max_steps, and ensure historical_loss/lr/steps are populated."
        try:
            os.makedirs(report_dir, exist_ok=True) # Ensure directory exists
            with open(assessment_path, 'w') as f:
                f.write(file_content)
            logger.info(f"Empty/placeholder training assessment saved to {assessment_path}")
        except Exception as e:
            logger.error(f"!!! Failed to save empty training assessment: {e}")
        return file_content

    # Pad learning_rates if its length is less than losses
    if len(learning_rates) < len(losses):
        logger.warning(f"Learning rates data points ({len(learning_rates)}) less than loss data points ({len(losses)}). Padding LR for plot.")
        last_lr = learning_rates[-1] if learning_rates else 0.0
        learning_rates.extend([last_lr] * (len(losses) - len(learning_rates)))
    elif len(learning_rates) > len(losses) and losses: # Ensure losses is not empty
            learning_rates = learning_rates[:len(losses)]

    try:
        os.makedirs(report_dir, exist_ok=True) # Ensure directory exists
        plt.figure(figsize=(12, 8))
        plt.style.use('seaborn-v0_8-darkgrid') # Using a seaborn style

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(steps, losses, marker='o', linestyle='-', color='dodgerblue', label='Loss')
        plt.title('Training Loss Over Steps', fontsize=14)
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot learning rate
        plt.subplot(2, 1, 2)
        if learning_rates:
            plt.plot(steps, learning_rates, marker='.', linestyle='-', color='coral', label='Learning Rate')
        plt.title('Learning Rate Over Steps', fontsize=14)
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Scientific notation for LR
        if learning_rates: plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if not learning_rates:
            plt.gca().text(0.5, 0.5, 'Learning rate data not available.', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        plt.tight_layout(pad=2.0)
        plot_path = os.path.join(report_dir, 'training_metrics.png')
        plt.savefig(plot_path)
        plt.close() # Close the plot to free memory
        #logger.debug(f"Training metrics plot saved to {plot_path}")

        # Use the assess_training_quality function from this module directly
        assessment = assess_training_quality(losses, learning_rates, steps, grad_norms)
        assessment_path = os.path.join(report_dir, 'training_assessment.txt')
        with open(assessment_path, 'w') as f:
            f.write(assessment)
        #logger.debug(f"Training assessment saved to {assessment_path}")
        logger.info(f"\n{assessment}")
        file_content = assessment

    except Exception as e:
        logger.error(f"!!! Error generating training report: {e}", exc_info=True)
        file_content = f"Error:{str(e)}"
        # Consider logging traceback here if needed: import traceback; traceback.print_exc()

    return file_content

def save_adapter_metadata(
    output_dir: str,
    adapter_name: str,
    adapter_type: str, # e.g., "LORA"
    base_model_name_or_path: Optional[str],
    training_config: Optional[Dict[str, Any]] = None, # For LoRA r, alpha etc.
    precision_info: Optional[Dict[str, Any]] = None # For trainer_use_fp16 etc.
):
    """Saves metadata about the adapter and training run."""
    metadata = {
        "adapter_name": adapter_name,
        "adapter_type": adapter_type,
        "base_model_name_or_path": base_model_name_or_path,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if training_config:
        metadata["lora_config"] = {
            "r": training_config.get("r"),
            "alpha": training_config.get("lora_alpha"),
            "dropout": training_config.get("lora_dropout"),
            "target_modules": training_config.get("target_modules"),
        }
    if precision_info:
        metadata["precision_info"] = precision_info
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    #logger.debug(f"Adapter metadata saved to {metadata_path}")

# --- Adapter Path and Metadata Utilities ---

class ResolvedAdapterPathInfo:
    """Holds information about resolved adapter paths."""
    def __init__(self,
                 input_path_str: str, # The original path string from the API
                 adapter_root_path: Path, # Resolved root path for the adapter
                 checkpoint_path_to_load: Optional[Path], # Path to specific checkpoint files to load (if existing)
                 logical_name: str,
                 is_new_adapter_scenario: bool = False):
        self.input_path_str = input_path_str
        self.adapter_root_path = adapter_root_path
        self.checkpoint_path_to_load = checkpoint_path_to_load
        self.logical_name = logical_name
        self.is_new_adapter_scenario = is_new_adapter_scenario

    def __repr__(self):
        return (f"ResolvedAdapterPathInfo(input='{self.input_path_str}', root='{self.adapter_root_path}', "
                f"checkpoint_to_load='{self.checkpoint_path_to_load}', name='{self.logical_name}', new={self.is_new_adapter_scenario})")

def find_latest_checkpoint_in_dir(parent_dir: Path) -> Optional[Path]:
    """Finds the latest checkpoint subdirectory within a given parent directory."""
    potential_checkpoints: List[Path] = []
    if not parent_dir.is_dir():
        return None

    for item in parent_dir.iterdir():
        if item.is_dir():
            # General timestamped format: YYYYMMDD-HHMMSS-...
            # This will match YYYYMMDD-HHMMSS-stepXXX, YYYYMMDD-HHMMSS-best, etc.
            is_timestamp_format = False
            if len(item.name) > 15 and item.name[8] == '-' and item.name[15] == '-': # Ensures YYYYMMDD-HHMMSS-
                date_part = item.name[0:8]
                time_part = item.name[9:15]
                if date_part.isdigit() and time_part.isdigit():
                    is_timestamp_format = True

            is_checkpoint_num = item.name.startswith("checkpoint-")
            
            # A directory is a checkpoint if it contains adapter weights OR adapter_config.
            has_weights = (item / "adapter_model.safetensors").is_file()
            has_cfg     = (item / "adapter_config.json").is_file()
            if (is_timestamp_format or is_checkpoint_num) and (has_weights or has_cfg):
                potential_checkpoints.append(item)
    
    if not potential_checkpoints:
        # Fallback: treat the ROOT as a checkpoint if it directly contains both files
        root_has_weights = (parent_dir / "adapter_model.safetensors").is_file()
        root_has_cfg     = (parent_dir / "adapter_config.json").is_file()
        if root_has_weights and root_has_cfg:
            return parent_dir
        return None

    def sort_key_checkpoint(p: Path):
        name = p.name
        # Priority 0: Hugging Face Trainer's checkpoint-XXX (numeric part for sorting)
        if name.startswith("checkpoint-"):
            try: return (0, int(name.split('-')[-1])) # Sort by number primarily
            except ValueError: return (2, name) # Fallback for checkpoint- non-numeric
        
        # Priority 1: Timestamped format YYYYMMDD-HHMMSS-...
        is_our_timestamp_format_for_sort = False
        if len(name) > 15 and name[8] == '-' and name[15] == '-': # Check for YYYYMMDD-HHMMSS-
            date_part_sort = name[0:8]
            time_part_sort = name[9:15]
            if date_part_sort.isdigit() and time_part_sort.isdigit():
                is_our_timestamp_format_for_sort = True
        
        if is_our_timestamp_format_for_sort:
            return (1, name) # Sort by full name (timestamp first)

        return (2, name) # Fallback for other directory names

    potential_checkpoints.sort(key=sort_key_checkpoint, reverse=True) # Get latest
    return potential_checkpoints[0]

def resolve_adapter_paths_and_name(
    logger: logging.Logger,
    input_path_str: str,
    api_adapter_name_override: Optional[str] = None
) -> ResolvedAdapterPathInfo:
    """
    Resolves adapter root, checkpoint path, and logical name from an input path string.
    - input_path_str: The path provided in the API call (AdapterConfig.adapter_path).
    - api_adapter_name_override: The name provided in the API call (AdapterConfig.adapter_name),
                                 acts as an override for existing or name for new.
    """
    input_path = Path(input_path_str).resolve()
    original_input_path_str = input_path_str # Keep for the final object

    adapter_root_path: Path
    checkpoint_to_load: Optional[Path] = None
    logical_name: str
    is_new_scenario = False

    custom_metadata_file = "metadata.json"
    peft_model_file = "adapter_model.safetensors"

    # Scenario 1: input_path is a direct checkpoint directory
    if input_path.is_dir() and (input_path / peft_model_file).is_file():
        logger.info(f"resolve: Input '{input_path}' is a checkpoint directory (found {peft_model_file}).")
        checkpoint_to_load = input_path
        adapter_root_path = input_path.parent

    # Scenario 2: input_path is an adapter root directory
    # It's a root if it's a directory, does NOT contain the model file itself,
    # but either has metadata.json or contains valid checkpoint subdirectories.
    elif input_path.is_dir() and not (input_path / peft_model_file).is_file() and \
         ((input_path / custom_metadata_file).is_file() or find_latest_checkpoint_in_dir(input_path) is not None):
        logger.info(f"resolve: Input '{input_path}' is an adapter root directory.")
        adapter_root_path = input_path
        checkpoint_to_load = find_latest_checkpoint_in_dir(adapter_root_path)
        if not checkpoint_to_load:
            logger.info(f"resolve: Adapter root '{input_path}' has no loadable checkpoints.")

    # Scenario 3: New adapter, or input_path is a parent dir for a new adapter
    else:
        logger.info(f"resolve: Input '{input_path}' not a direct checkpoint or recognized root. Assuming new adapter context.")
        is_new_scenario = True

        if api_adapter_name_override:
            # If an override name is given, the input path is the parent directory,
            # and the new adapter will be a subdirectory named after the override.
            # If the input path's name already matches the override, then the input path is the root.
            adapter_root_path = input_path / api_adapter_name_override if input_path.name != api_adapter_name_override else input_path
        else:
            # No API name, so the input_path itself is the intended new adapter root path.
            adapter_root_path = input_path
        
        # Crucial check: If the derived adapter_root_path *already* exists and looks like an adapter,
        # it's NOT a new scenario. Re-evaluate.
        if adapter_root_path.is_dir() and (
            (adapter_root_path / custom_metadata_file).is_file() or \
            find_latest_checkpoint_in_dir(adapter_root_path) is not None # Check for checkpoints too
           ):
            logger.info(f"resolve: Derived new adapter root '{adapter_root_path}' seems to be an existing adapter. Re-resolving.")
            # Call recursively with the derived_adapter_root_path as the new input_path_str
            # and preserve the original api_adapter_name_override.
            return resolve_adapter_paths_and_name(logger, str(adapter_root_path), api_adapter_name_override)

    # --- Determine logical_name based on precedence ---
    if is_new_scenario:
        # For a new adapter, the name is the override or derived from the path.
        logical_name = api_adapter_name_override or adapter_root_path.name
        if not logical_name or logical_name in [".", ".."]:
            raise ValueError(f"Cannot derive a valid adapter name from path '{input_path_str}' for a new adapter without an override name.")
    else: # Existing adapter
        name_from_metadata = None
        metadata_path = adapter_root_path / "metadata.json"
        if metadata_path.is_file():
            try:
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                    name_from_metadata = meta.get("adapter_name")
                    if name_from_metadata:
                        logger.info(f"resolve: Found adapter name '{name_from_metadata}' in {custom_metadata_file}.")
            except Exception as e:
                logger.warning(f"resolve: Could not read adapter_name from {metadata_path}: {e}")
        
        # Precedence for existing adapters: API override > metadata name > directory name
        logical_name = api_adapter_name_override or name_from_metadata or adapter_root_path.name

    # --- Sanitize the final logical_name for PEFT compatibility ---
    # This is applied universally because PEFT will reject any name with a dot.
    original_logical_name = logical_name
    if '.' in logical_name:
        logical_name = logical_name.replace('.', '_')
        logger.info(f"resolve: Sanitized adapter name from '{original_logical_name}' to '{logical_name}' for PEFT compatibility.")

    return ResolvedAdapterPathInfo(original_input_path_str, adapter_root_path, checkpoint_to_load, logical_name, is_new_scenario)

def get_checkpoints(adapter_root_path_str: str) -> List[Dict[str, Union[str, float]]]:
    """Returns a list of checkpoint subfolder names and their modification times from an adapter root."""
    adapter_root_path = Path(adapter_root_path_str)
    checkpoints_info = []
    if not adapter_root_path.is_dir():
        return []

    for item in adapter_root_path.iterdir():
        if item.is_dir() and (item / "adapter_config.json").is_file():
            # Could add more sophisticated parsing of name for timestamp if needed
            checkpoints_info.append({
                "name": item.name,
                "modified_time": item.stat().st_mtime 
            })
    
    checkpoints_info.sort(key=lambda x: x["modified_time"], reverse=True) # Newest first
    return checkpoints_info

async def copy_report_and_metadata_files(
    logger: logging.Logger,
    source_dir: Path,
    dest_dir: Path
):
    """
    Asynchronously copies metadata.json, training_assessment.txt, and training_metrics.png
    from source_dir to dest_dir.
    """
    files_to_copy = ["metadata.json", "training_assessment.txt", "training_metrics.png"]
    try:
        await asyncio.to_thread(os.makedirs, dest_dir, exist_ok=True)
    except Exception as e_mkdir:
        logger.warning(f"Failed to create destination directory {dest_dir} for copying reports: {e_mkdir}")
        return

    copied_files_count = 0
    for filename in files_to_copy:
        source_file = source_dir / filename
        dest_file = dest_dir / filename
        if await asyncio.to_thread(source_file.is_file):
            try:
                await asyncio.to_thread(shutil.copy2, source_file, dest_file)
                copied_files_count += 1
            except Exception as e_copy:
                logger.warning(f"Failed to copy {filename} from {source_dir} to {dest_dir}: {e_copy}")
        else:
            logger.info(f"Source report/metadata file {source_file} not found for copying to {dest_dir}.")
    if copied_files_count > 0:
        logger.info(f"Copied {copied_files_count} report/metadata files to {dest_dir}")


#------------------ Helper to ensure eos and pad tokens ------------------
def choose_special_tokens_no_add(tokenizer, model=None):
    """
    Returns dict with possibly:
      - 'eos_token_id': int | list[int]
      - 'pad_token_id': int
    No adding/resizing; picks from existing vocab only.
    """
    # 1) EOS from model if available; else from tokenizer
    e = None
    if model is not None and getattr(model, "generation_config", None) is not None:
        e = model.generation_config.eos_token_id
    if e is None:
        e = tokenizer.eos_token_id

    if isinstance(e, int):
        eos_ids = [e]
    elif isinstance(e, (list, tuple)):
        eos_ids = [x for x in e if isinstance(x, int)]
    else:
        eos_ids = []
    eos_set = set(eos_ids)

    overrides = {}
    if eos_ids:
        overrides["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]

    # 2) If a pad already exists and isn't an EOS, keep it
    pad_id = tokenizer.pad_token_id
    if isinstance(pad_id, int) and pad_id >= 0 and pad_id not in eos_set:
        overrides["pad_token_id"] = pad_id
        return overrides

    # 3) Probe the actual vocab for safe PAD candidates (model-agnostic)
    vocab = tokenizer.get_vocab()  # token -> id
    all_special_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, 'all_special_ids') else set()

    def add_id(tok):
        tid = vocab.get(tok)
        if isinstance(tid, int) and tid >= 0:
            # Return tuple: (token_id, is_officially_special)
            return [(tid, tid in all_special_ids)]
        return []

    candidates_with_confidence = []
    # (a) Common pad tokens (exact match)
    for t in ("<pad>", "[PAD]", "<|pad|>", "<|fim_pad|>", "<fim_pad>"):
        candidates_with_confidence += add_id(t)

    # (b) Phi-3.5 placeholders (if present)
    for t in ("<|placeholder5|>", "<|placeholder4|>", "<|placeholder3|>"):
        candidates_with_confidence += add_id(t)

    # (c) Mistral <SPECIAL_N> tokens
    special_n_tokens = sorted([
        (tok, tid) for tok, tid in vocab.items()
        if (isinstance(tok, str) and tok.startswith("<SPECIAL_") and tok.endswith(">")) or \
           (isinstance(tok, bytes) and tok.startswith(b"<SPECIAL_") and tok.endswith(b">"))
    ])
    candidates_with_confidence += [(tid, tid in all_special_ids) for tok, tid in special_n_tokens]

    # (d) Phi-4 “filler” endoftextNN (NOT the real <|endoftext|>)
    filler_tokens = sorted([
        (tok, tid) for tok, tid in vocab.items()
        if (isinstance(tok, str) and tok.startswith("<|endoftext") and tok.endswith("|>") and tok not in ("<|endoftext|>",)) or \
           (isinstance(tok, bytes) and tok.startswith(b"<|endoftext") and tok.endswith(b"|>") and tok not in (b"<|endoftext|>",))
    ])
    candidates_with_confidence += [(tid, tid in all_special_ids) for tok, tid in filler_tokens]

    # (e) Phi-3-small dummy ids (there are many)
    dummy_tokens = sorted([
        (tok, tid) for tok, tid in vocab.items()
        if (isinstance(tok, str) and tok.startswith("<|dummy_id_")) or (isinstance(tok, bytes) and tok.startswith(b"<|dummy_id_"))
    ])
    candidates_with_confidence += [(tid, tid in all_special_ids) for tok, tid in dummy_tokens]

    # Sort candidates to prioritize those that are officially special
    high_confidence = [tid for tid, is_special in candidates_with_confidence if is_special]
    low_confidence = [tid for tid, is_special in candidates_with_confidence if not is_special]
    
    # Dedup in order, keeping high-confidence candidates first
    seen, ordered = set(), []
    for tid in high_confidence + low_confidence:
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)

    for tid in ordered:
        if tid not in eos_set:
            overrides["pad_token_id"] = tid
            return overrides

    # 4) No safe pad found -> leave pad unset; use attention_mask in training,
    #    and at inference you can set pad=eos at runtime.
    return overrides

def _is_phi3(path: str, cfg=None) -> bool:
    """Checks if a model path or config corresponds to a Phi-3 model (excluding 3.5)."""
    name = path.lower()
    if "phi-3.5" in name:
        return False
    if "phi-3" in name:
        return True
    if cfg is None:
        try: cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        except Exception: return False
    # Fallback for models not named with "phi-3" but have the type
    return getattr(cfg, "model_type", "") == "phi3"

def needs_single_gpu(path: str, attn_impl: str) -> bool:
    """Determines if a model needs to be forced onto a single GPU."""
    if attn_impl != "flash_attention_2": return False
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return _is_phi3(path, cfg) or not getattr(cfg, "_supports_sdpa", True)


def get_best_device_map(reserve_gib: int = 2) -> Dict[Union[int, str], Union[int, str]]:
    """Finds the GPU with the most free memory and returns a device map for it."""
    _clear_gpu_mem()
    best, best_idx = 0, 0
    for idx in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(idx)
        if free > best: best, best_idx = free, idx
    torch.cuda.set_device(best_idx)
    return {"": best_idx}

def low_priority_stream_for(device: torch.device) -> "torch.cuda.Stream":
    # ensure we’re on the right device before creating the stream
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # best-effort: try both known names, else fall back to default priority
    priority = 0  # default priority (safe fallback)
    try:
        # Some builds expose this name
        least, greatest = torch.cuda.get_device_stream_priority_range()  # type: ignore[attr-defined]
        priority = least   # lowest priority
    except Exception:
        try:
            # Others expose this shorter name
            least, greatest = torch.cuda.get_stream_priority_range()      # type: ignore[attr-defined]
            priority = least
        except Exception:
            priority = 0
    return torch.cuda.Stream(priority=priority)

# Disabled cuda graphs context
@contextmanager
def no_cudagraphs():
    old = {
        "cudagraphs": ic.triton.cudagraphs,
        "cudagraph_trees": getattr(ic.triton, "cudagraph_trees", None),
        "cudagraph_skip_dynamic_graphs": getattr(ic.triton, "cudagraph_skip_dynamic_graphs", None),
        "use_static_cuda_launcher": getattr(ic, "use_static_cuda_launcher", None),
        "cudagraph_trees_top": getattr(ic, "cudagraph_trees", None),
    }
    try:
        ic.triton.cudagraphs = False
        if hasattr(ic.triton, "cudagraph_trees"):
            ic.triton.cudagraph_trees = False
        if hasattr(ic.triton, "cudagraph_skip_dynamic_graphs"):
            ic.triton.cudagraph_skip_dynamic_graphs = False
        if hasattr(ic, "use_static_cuda_launcher"):
            ic.use_static_cuda_launcher = False
        if hasattr(ic, "cudagraph_trees"):
            ic.cudagraph_trees = False
        yield
    finally:
        ic.triton.cudagraphs = old["cudagraphs"]
        if old["cudagraph_trees"] is not None:
            ic.triton.cudagraph_trees = old["cudagraph_trees"]
        if old["cudagraph_skip_dynamic_graphs"] is not None:
            ic.triton.cudagraph_skip_dynamic_graphs = old["cudagraph_skip_dynamic_graphs"]
        if old["use_static_cuda_launcher"] is not None:
            ic.use_static_cuda_launcher = old["use_static_cuda_launcher"]
        if old["cudagraph_trees_top"] is not None:
            ic.cudagraph_trees = old["cudagraph_trees_top"]

@contextmanager
def no_static_cuda_launcher():
    """Temporarily disable Inductor's static CUDA launcher (avoids c_long overflow on some streams)."""
    old = getattr(ic, "use_static_cuda_launcher", None)
    try:
        if old is not None:
            ic.use_static_cuda_launcher = False
        yield
    finally:
        if old is not None:
            ic.use_static_cuda_launcher = old

import torch

def inspect_device_layout(model):
    """
    Returns a dict like:
      {
        "mode": "single" | "sharded" | "offloaded" | "unknown",
        "devices": ["cuda:0", "cuda:1", "cpu", "disk", "meta"],
        "source": "hf_device_map" | "param_scan",
      }
    """

    # 1) Try Hugging Face’s authoritative map (set by device_map='auto' / accelerate)
    dm = getattr(model, "hf_device_map", None) or getattr(getattr(model, "model", None), "hf_device_map", None)
    if isinstance(dm, dict):
        # Normalize device labels
        devs = set()
        for v in dm.values():
            if isinstance(v, int):
                devs.add(f"cuda:{v}")
            elif isinstance(v, str):
                devs.add(v)  # e.g., "cpu", "disk", "meta", "cuda:0"
        
        if devs:
            mode = "single"
            if any(d in devs for d in ("disk", "cpu", "meta")) and any(d.startswith("cuda:") for d in devs):
                mode = "offloaded"              # parts on CPU/disk + CUDA
            elif sum(d.startswith("cuda:") for d in devs) > 1:
                mode = "sharded"                # multiple CUDA devices
            elif len(devs) == 1 and next(iter(devs)).startswith("cuda:"):
                mode = "single"
            else:
                mode = "unknown"
            return {"mode": mode, "devices": sorted(list(devs)), "source": "hf_device_map"}

    # 2) Fallback: scan parameters/buffers (works even after torch.compile)
    devs = set()
    try:
        for p in model.parameters():
            # Compiled/PEFT models still expose real parameter devices
            devs.add(str(p.device))
    except Exception:
        pass
    # also scan buffers (some wrappers stash tensors there)
    try:
        for b in model.buffers():
            devs.add(str(b.device))
    except Exception:
        pass
    devs = {d for d in devs if d} or {"unknown"}

    mode = "single"
    n_cuda = sum(d.startswith("cuda:") for d in devs)
    has_cpu_or_meta = any(d.startswith("cpu") or d == "meta" for d in devs)

    if n_cuda > 1:
        mode = "sharded"
    elif has_cpu_or_meta and any(d.startswith("cuda:") for d in devs):
        mode = "offloaded"
    elif len(devs) == 1 and next(iter(devs)).startswith("cuda:"):
        mode = "single"
    else:
        mode = "unknown"

    return {"mode": mode, "devices": sorted(list(devs)), "source": "param_scan"}


# ------------------------------------------------------------------
# Figure out which GPU holds the embedding layer when the model
# has been sharded with device_map="auto".
# (Adapted from verify_lora.py)
# ------------------------------------------------------------------
def first_module_device(model: torch.nn.Module) -> torch.device:
    """
    Return the device that hosts the model's embedding layer
    when the model was loaded with device_map=\"auto\".
    Falls back to the device of the first parameter for other cases.
    """
    # Hugging Face stores a mapping layer_name → "cuda:N"
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        # try common keys first
        for key in ("model.embed_tokens", "embed_tokens"):
            if key in model.hf_device_map:
                return torch.device(model.hf_device_map[key])
        # otherwise just take the first entry in the map
        if model.hf_device_map:
            first_key = next(iter(model.hf_device_map))
            return torch.device(model.hf_device_map[first_key])

    # Fallback for non-sharded models or if map is empty/not present
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu") # Model has no parameters
# ------------------------------------------------------------------
def first_module_device_for_sharded_model(model: torch.nn.Module) -> torch.device:
    """
    Return the device that hosts the model's embedding layer
    when the model was loaded with device_map=\"auto\".
    Falls back to the device of the first parameter for other cases.
    """
    # Hugging Face stores a mapping layer_name → "cuda:N"
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        # try common keys first
        for key in ("model.embed_tokens", "embed_tokens", "transformer.wte"):
            if key in model.hf_device_map:
                return torch.device(model.hf_device_map[key])
        # otherwise just take the first entry in the map
        if model.hf_device_map:
            first_key = next(iter(model.hf_device_map))
            return torch.device(model.hf_device_map[first_key])

    # Fallback for non-sharded models or if map is empty/not present
    return first_module_device(model)

# --- Adapter-only save for PeftMixedModel (works with engine.add_adapter) ---

def save_adapter_package_for_mixed(
    mixed_model,                    # engine.state._peft_model (PeftMixedModel)
    adapter_name: str,              # e.g. "test_mp13_adapter"
    adapter_root_dir: Path,         # e.g. adapters/test_mp13_adapter
    step: int,                      # e.g. trainer.state.global_step
    *,
    base_model_name_or_path: Optional[str] = None,
    extra_root_meta: Optional[Dict[str, Any]] = None,
    extra_ckpt_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Writes an adapter-only checkpoint in the structure expected by engine.add_adapter():
      <adapter_root_dir>/
        metadata.json                   # (optional/custom) root metadata
        <YYYYMMDD-HHMMSS-stepN>/
          adapter_model.safetensors     # adapter weights only
          adapter_config.json           # PEFT config
          metadata.json                 # checkpoint-level metadata (takes precedence)
          README.md

    Returns the checkpoint directory path.
    """
    adapter_root_dir = Path(adapter_root_dir)
    adapter_root_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Compose checkpoint directory name ----
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = adapter_root_dir / f"{ts}-step{step}"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2) Extract adapter-only weights from the PEFT training core ----
    # Use the model that was actually used for training forward/backward
    unwrapped_model =   getattr(mixed_model, "_orig_mod", mixed_model) 
    peft_state = get_peft_model_state_dict(unwrapped_model, adapter_name=adapter_name)
    save_file(peft_state, str(ckpt_dir / "adapter_model.safetensors"))

    # ---- 3) Save adapter_config.json using the actual subclass (e.g., LoraConfig) ----
    # Prefer the config on the training core; fall back to mixed
    # Use the unwrapped model to get the config source
    cfg_src = getattr(unwrapped_model, "peft_config", None)
    if not cfg_src or adapter_name not in cfg_src:
        raise RuntimeError(f"Adapter '{adapter_name}' config not found on training core/mixed model.")
    peft_cfg = cfg_src[adapter_name]
    # Normalize to LoraConfig if the instance was a dict-like wrapper
    if not isinstance(peft_cfg, LoraConfig):
        peft_cfg = LoraConfig(**peft_cfg.to_dict())
    if base_model_name_or_path and not getattr(peft_cfg, "base_model_name_or_path", None):
        peft_cfg.base_model_name_or_path = base_model_name_or_path
    peft_cfg.save_pretrained(str(ckpt_dir))  # writes adapter_config.json

    # ---- 4) Write metadata.json files (optional but your engine reads & merges them) ----
    # Root metadata.json (coarse info shared across checkpoints)
    root_meta: Dict[str, Any] = {
        "adapter_name": adapter_name,
        "adapter_type": "lora",                # your engine reads 'adapter_type'
        "peft_type": peft_cfg.peft_type if hasattr(peft_cfg, "peft_type") else "LORA",
        "base_model_name_or_path": base_model_name_or_path,
        "created_at": ts,
    }
    if extra_root_meta:
        root_meta.update(extra_root_meta)
    # Only (re)write if absent; safe to always write if you prefer
    root_meta_path = adapter_root_dir / "metadata.json"
    try:
        if not root_meta_path.exists():
            root_meta_path.write_text(json.dumps(root_meta, indent=2))
    except Exception:
        # If you prefer root_meta to always reflect latest, replace the condition above
        pass

    # Checkpoint-level metadata.json (highest precedence in your engine)
    ckpt_meta: Dict[str, Any] = {
        "adapter_name": adapter_name,
        "adapter_type": "lora",
        "peft_type": peft_cfg.peft_type if hasattr(peft_cfg, "peft_type") else "LORA",
        "base_model_name_or_path": base_model_name_or_path,
        "step": step,
        "timestamp": ts,
    }
    if extra_ckpt_meta:
        ckpt_meta.update(extra_ckpt_meta)
    (ckpt_dir / "metadata.json").write_text(json.dumps(ckpt_meta, indent=2))

    # ---- 5) README.md (nice to have; your engine doesn’t require it but PEFT usually writes one) ----
    (ckpt_dir / "README.md").write_text(
        f"# Adapter: {adapter_name}\n\n"
        f"- **Type**: LoRA\n"
        f"- **Step**: {step}\n"
        f"- **Created**: {ts}\n"
        f"- **Base**: {base_model_name_or_path or 'unknown'}\n\n"
        f"Files:\n"
        f"- `adapter_config.json`\n"
        f"- `adapter_model.safetensors`\n"
        f"- `metadata.json`\n"
    )

    return ckpt_dir

@dataclass
class HardwareReport:
    """A report on the current hardware status, focusing on GPU memory."""
    device_type: str
    device_name: str
    total_memory_gb: float
    model_memory_gb: float
    # The memory available for the training process to allocate new tensors.
    available_memory_gb: float
    model_params_b: float
    devices: List[Dict[str, float]] = field(default_factory=list)
    layout: Optional[Dict[str, Any]] = None
    device_count: int = 1

def get_hardware_report(model: torch.nn.Module) -> HardwareReport:
    """
    Inspects the hardware (primarily GPU) to get a report on memory and model size.
    Assumes the model is loaded on the target device.
    """
    # Force garbage collection to get a more accurate memory reading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_device = None
    layout: Optional[Dict[str, Any]] = None
    active_gpu_ids: List[int] = []
    if torch.cuda.is_available():
        try:
            layout = inspect_device_layout(model)
            active_gpu_ids = sorted([
                int(str(dev).split(":")[1]) for dev in layout.get("devices", set())
                if str(dev).startswith("cuda:")
            ])
            if not active_gpu_ids:
                active_gpu_ids = list(range(torch.cuda.device_count()))
            # Determine the primary device of the model
            model_device = first_module_device(model)
            if model_device.type != 'cuda':
                model_device = None # Treat as CPU if not on a CUDA device
        except (RuntimeError, StopIteration):
            model_device = None # Fallback if model has no params or other issues

    if model_device:
        per_device: List[Dict[str, float]] = []
        total_allocated = 0.0
        total_reserved = 0.0
        free_pool: List[float] = []
        total_memory_sum = 0.0
        device_name = torch.cuda.get_device_properties(model_device).name
        for idx in active_gpu_ids:
            props = torch.cuda.get_device_properties(idx)
            total_memory_sum += props.total_memory / (1024**3)
            free_memory_bytes, _ = torch.cuda.mem_get_info(idx)
            allocated_memory_bytes = torch.cuda.memory_allocated(idx)
            reserved_memory_bytes = torch.cuda.memory_reserved(idx)
            free_gb = free_memory_bytes / (1024**3)
            allocated_gb = allocated_memory_bytes / (1024**3)
            reserved_gb = reserved_memory_bytes / (1024**3)
            free_pool.append(free_gb)
            total_allocated += allocated_gb
            total_reserved += reserved_gb
            per_device.append({
                "id": idx,
                "name": props.name,
                "total_gb": props.total_memory / (1024**3),
                "free_gb": free_gb,
                "allocated_gb": allocated_gb,
                "reserved_gb": reserved_gb,
            })
        # Use the most constrained device as the available budget for sharded models
        available_memory_gb = min(free_pool) if free_pool else 0.0
        model_params = sum(p.numel() for p in model.parameters())
        return HardwareReport(
            device_type='cuda',
            device_name=device_name,
            total_memory_gb=total_memory_sum,
            # This represents the memory used by the model weights and any other tensors at this point
            model_memory_gb=total_allocated,
            # Use per-device minimum free memory to set a safe budget when sharded
            available_memory_gb=available_memory_gb,
            model_params_b=model_params / 1e9,
            devices=per_device,
            layout=layout,
            device_count=max(1, len(per_device))
        )
    else:
        # CPU fallback
        model_params = sum(p.numel() for p in model.parameters())
        return HardwareReport(
            device_type='cpu',
            device_name='CPU',
            total_memory_gb=0.0,
            model_memory_gb=0.0,
            available_memory_gb=0.0, # Not applicable in the same way for CPU
            model_params_b=model_params / 1e9,
            devices=[],
            layout=layout,
            device_count=1
        )
