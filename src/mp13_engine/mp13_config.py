# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""Configuration classes for language model training and inference."""

from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from typing import List, Optional, Dict, Union, Any, Callable, TYPE_CHECKING, ClassVar, AsyncIterator
import copy
from dataclasses import dataclass, field
import torch
import os
import hashlib
import json
import re
import codecs


# --- General Enums ---

class EngineMode(str, Enum):
    """Mode of the MP13 engine."""
    TRAIN = "train"
    INFERENCE = "inference"

class ReturnPromptEnum(str, Enum):
    """Specifies what part of the prompt to return, if any."""
    FULL = "full"
    LAST = "last"

class ChunkType(str, Enum):
    """Status type for an inference response chunk."""
    PROMPT_STARTED = "prompt_started"
    STREAMING_CHUNK = "streaming_chunk"
    STREAMING_ENDED = "streaming_ended"
    ERROR = "error"

class APIStatus(str, Enum):
    """Standardized status strings for API responses."""
    SUCCESS = "success"
    ERROR = "error"
    NO_OP = "no_op"


class AdapterType(str, Enum):
    """Type of PEFT adapter."""
    LORA = "lora"

class IfExistsEnum(str, Enum):
    """Behavior if an item being added already exists."""
    FAIL = "fail"
    IGNORE = "ignore"
    RELOAD = "reload"

class TrainingMode(str, Enum):
    """Primary training objective."""
    SFT = "sft" # Supervised Fine-Tuning (instruction/response pairs)
    # MLM removed

class PreprocessingMode(str, Enum):
    """Mode for preprocessing data during training."""
    FULL_TEXT = "full_text" # Process the entire text as one sequence
    APPLY_CHAT_TEMPLATE = "apply_chat_template" # Expects structured messages, trainer applies chat template

# --- Dataset Related Enums and Classes ---

class DatasetFormat(str, Enum):
    """Supported dataset formats."""
    MESSAGES = "messages"   # Chat messages format
    TEXT = "text"           # Simple text file (for SFT if structured with prompt/response or single text field)

class DatasetTags(BaseModel):
    """Tags used in specific formats like ShareGPT or messages format."""
    role_tag: str = "role"
    content_tag: str = "content"
    user_tag: str = "user"
    assistant_tag: str = "assistant"
    system_tag: str = "system"

class ColumnsConfig(BaseModel):
    """Column names mapping for datasets."""
    # For SFT (Alpaca/Causal LM)
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input" # Optional secondary input for SFT
    response: Optional[str] = "output"
    # For messages format
    messages: Optional[str] = "messages"
    # For simple text SFT
    text: Optional[str] = "text"

class DatasetConfig(BaseModel):
    """Configuration for the dataset."""
    dataset_path: str = Field(..., description="Path to the dataset (local file or Hugging Face Hub ID)")
    formatting: DatasetFormat = DatasetFormat.MESSAGES
    preprocessing_mode: Optional[str] = PreprocessingMode.FULL_TEXT.value
    subset: Optional[str] = None
    split: Optional[str] = "train"
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    tags: Optional[DatasetTags] = None

# QuantizationConfig and related enums removed

# MLMConfig removed

# --- Global Engine Initialization Configuration ---
class GlobalEngineConfig(BaseModel):
    """Configuration for initializing the global MP13 engine resources."""
    base_model_name_or_path: str = Field(..., description="Path or Hub ID of the base model")
    instance_id: Optional[str] = Field(None, description="Optional instance ID for the engine. If provided, it will override the default.")
    tools_parser_profile_key: Optional[str] = Field(None, description="Key for a specific tool parser profile (e.g., 'mistral', 'qwen'). If None, it's inferred from the model's chat template.")
    custom_chat_template: Optional[str] = Field(None, description="Optional chat template string to override the tokenizer's built-in template.")
    initial_engine_mode: EngineMode = Field(EngineMode.INFERENCE, description="Initial mode for the engine after global initialization.")
    no_tools_parse: bool = Field(False, description="If true, disables tool block parsing for all inference requests handled by this engine instance.")
    disable_custom_pad_ids: bool = Field(False, description="If true, suppress modifying model special tokens (e.g., pad_token_id) during engine initialization. A warning will be logged.")

    device_map: Union[Optional[str], Dict[str, Any]] = Field(
        default="auto",
        description="Device map for model loading (e.g., 'auto', 'cpu', or a dictionary like {'':0})."
    )

    trust_remote_code: bool = Field(True, description="Trust remote code execution for model/tokenizer.")
    base_model_torch_dtype: str = Field(
        "auto",
        description="Torch dtype for loading the base model. Options: 'auto', 'float16', 'bfloat16', 'float32'."
    )
    # --- Quantization ---
    quantize_bits: str = Field(
        "none",
        description="Quantization bits/method for base model. Options: 'none', '4', '8', 'awq', 'hqq', 'eetq'."
    )
    # (Not supported) BitsAndBytes specific (for quantize_bits='4' or '8')
    bnb_4bit_quant_type: str = Field("nf4", description="BitsAndBytes 4-bit quantization type (if quantize_bits='4'). Options: 'nf4', 'fp4'.")
    bnb_4bit_compute_dtype: str = Field("bfloat16", description="BitsAndBytes 4-bit compute dtype (if quantize_bits='4'). Options: 'float32', 'bfloat16', 'float16'.")
    # (Not supported) AWQ specific (for quantize_bits='awq')
    awq_bits: int = Field(4, description="AWQ bits (if quantize_bits='awq'). Typically 4.")
    awq_group_size: int = Field(128, description="AWQ group size (if quantize_bits='awq').") # Common default is 128
    awq_zero_point: bool = Field(True, description="AWQ zero point (if quantize_bits='awq').")
    # HQQ specific (for quantize_bits='hqq')
    hqq_bits: int = Field(4, description="HQQ bits (if quantize_bits='hqq'). Options: 2, 3, 4, 8.")
    hqq_group_size: int = Field(64, description="HQQ group size (if quantize_bits='hqq').")
    hqq_quant_zero: bool = Field(True, description="HQQ quant_zero (if quantize_bits='hqq').")
    hqq_quant_scale: bool = Field(False, description="HQQ quant_scale (if quantize_bits='hqq').")
    hqq_axis: int = Field(1, description="HQQ axis for quantization (if quantize_bits='hqq'). Options: 0, 1.")

    attn_implementation: Optional[str] = Field("auto", description="Attention implementation for the base model (e.g., 'auto', 'sdpa', 'flash_attention_2', 'eager').")    
    default_context_size: Optional[int] = Field(None, description="Default context size for the engine. 'None' or 'auto' means derive from model. User can specify (e.g., 2048) to restrict it. Cannot exceed model's max capability.")
    default_max_new_tokens: int = Field(8192, description="Default value for max_new_tokens for inference requests. Can be overridden by individual requests.")
    use_cache: bool = Field(True, description="Enable KV caching for generation. If False, all caching is disabled.")
    use_torch_compile: bool = Field(True, description="Enable torch.compile() for faster inference. May increase initialization time but improves generation speed. Set to False if you encounter compilation errors.")
    static_kv_cache: bool = Field(
        False,
        description="Enable static GPU KV cache for inference. If False, only dynamic and offloaded caches will be used."
    )
    concurrent_generate: int = Field(4, description="Number of allowed concurrent user requests. Default is 1, if more than one then static cache will be forced to off.")
    log_with_instance_id: bool = Field(False, description="If true, reconfigures the logger to include the instance_id in all messages.")
    log_instance_id_width: int = Field(8, description="The fixed width for the instance_id column in logs, if enabled.")


    @field_validator('device_map', mode='before')
    @classmethod
    def normalize_device_map(cls, v: Union[Optional[str], Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        if v is None: # If input is explicitly None
            return "auto" if torch.cuda.is_available() else "cpu"
        if isinstance(v, str) and v == "auto" and not torch.cuda.is_available():
            return "cpu"
        # If it's a dict, or a str like "cpu", "cuda:0", or "auto" with CUDA, pass it through
        return v

    @field_validator('base_model_torch_dtype')
    @classmethod
    def check_dtype_cuda_support(cls, v: str) -> str:
        allowed_dtypes = ["auto", "float16", "bfloat16", "float32"]
        if v not in allowed_dtypes:
            raise ValueError(f"Invalid base_model_torch_dtype: '{v}'. Must be one of {allowed_dtypes}.")
        if v in ["float16", "bfloat16"] and not torch.cuda.is_available():
            # This is a config-level check; engine might still override if CUDA not found at runtime
            print(f"Warning: base_model_torch_dtype is '{v}' but CUDA is not available. Effective dtype may differ.")
        if v == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            print(f"Warning: base_model_torch_dtype is 'bfloat16' but the current CUDA device does not support bfloat16. Effective dtype may differ.")
        return v

    # No specific validators for new quantize_bits, awq_*, hqq_*, bnb_* here,
    # as the engine will perform more detailed checks and conversions based on quantize_bits.
    # Basic type validation is handled by Pydantic.
        
    @model_validator(mode='after')
    def validate_device_map_and_precision_warnings(self) -> 'GlobalEngineConfig':
        device_map_is_cpu = False
        if isinstance(self.device_map, str):
            device_map_is_cpu = (self.device_map == "cpu")
        # Note: More sophisticated checks for dict device_map implying all CPU are possible but complex.
        # The engine loading logic will ultimately determine behavior.
        
        if device_map_is_cpu:
            if self.base_model_torch_dtype in ["float16", "bfloat16"]:
                print(f"Warning: base_model_torch_dtype is '{self.base_model_torch_dtype}' but device_map is '{self.device_map}'. Precision flag will likely be ignored by transformers, defaulting to float32 on CPU.")
            if self.quantize_bits in ["4", "8", "awq", "hqq", "eetq"]:
                print(f"Warning: Quantization is enabled but device_map is '{self.device_map}'. Quantization will likely be disabled by transformers on CPU.")
        return self
    
    model_config = {
        "extra": "ignore", "validate_assignment": True
    }

# --- Training Configuration (for a specific adapter/training run) ---
class TrainingConfig(BaseModel):
    """Configuration for a training run using a specific adapter on the engine's model."""
    adapter_name_to_train: str = Field(..., description="Name of the adapter (already added to the engine) to be trained.")
    training_mode: TrainingMode = Field(TrainingMode.SFT, description="Training objective (only SFT supported).")

    # --- Dataset ---
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
    max_sequence_length: Optional[int] = Field(None, description="Override sequence length for this training run. 'None' or 'auto' uses engine's default. Cannot exceed engine's default context size or model's max capability.")
    heuristic_context_cap: Optional[int] = Field(
        default=2048,
        description="When auto_manage_memory is True, if the model's context is larger than this value, the heuristic will use this cap for its memory calculations. Set to 0 or -1 to disable capping. Default is 2k to favor stability."
    )

    # --- Training Hyperparameters ---
    output_dir: str = Field(..., description="Directory to save training outputs (checkpoints, logs) for this training run.")
    num_train_epochs: float = Field(1.0, description="Total number of training epochs (can be float).")
    max_steps: Optional[int] = Field(-1, description="Total number of training steps. Overrides num_train_epochs if > 0.")
    per_device_train_batch_size: int = Field(1, description="Batch size per GPU/CPU for training.")
    gradient_accumulation_steps: int = Field(8, description="Number of steps to accumulate gradients.")
    learning_rate: float = Field(5e-5, description="Initial learning rate.")
    lr_scheduler_type: str = Field("cosine", description="Learning rate scheduler type.")
    warmup_steps: int = Field(10, description="Number of warmup steps for the LR scheduler.")
    optim: str = Field("adamw_torch", description="Optimizer type.")
    max_grad_norm: float = Field(1.0, description="Maximum gradient norm for clipping.")
    seed: int = Field(42, description="Random seed for reproducibility.")

    # --- Logging / Saving ---
    logging_steps: int = Field(10, description="Log training metrics every N steps.") # Diff had 0, but context had 10. Keeping 10 as a more sensible default.
    save_strategy: str = Field("epoch", description="Checkpoint saving strategy.")
    save_steps: int = Field(0, description="Save checkpoint every N steps (if save_strategy='steps', ensure >0). If 0 with 'steps', uses epoch-based saving or similar.")
    save_total_limit: Optional[int] = Field(1, description="Maximum number of checkpoints to keep.")
    report_to: List[str] = Field(default_factory=lambda: ["none"], description="Integrations to report results to.")

    # --- Performance / Efficiency (for the training process) ---
    auto_manage_memory: Optional[bool] = Field(
        default=True,
        description="If True, the engine will automatically adjust batch size and gradient accumulation to prevent OOM errors based on context length and hardware. Set to False to disable this and use the provided parameters directly."
    )
    gradient_checkpointing: Optional[bool] = Field(
        default=None,
        description="Use gradient checkpointing to save memory. If None (auto), the engine will decide based on context length and hardware. If True/False, forces the setting."
    )
    dataloader_num_workers: int = Field(default=0 if os.name == 'nt' else 2, description="Number of workers for dataloader (0 recommended on Windows).")
    trainer_compute_precision: str = Field(
        "bf16", 
        description="Compute precision for the HuggingFace Trainer. Options: 'bf16', 'fp16', 'fp32'."
    )

    # --- Resuming ---
    resume_from_checkpoint: Optional[Union[str, bool]] = Field(None, description="Path to a Trainer checkpoint directory (within output_dir) or True to auto-detect last checkpoint.")
    disable_tqdm: bool = Field(False, description="Disable HuggingFace Trainer's tqdm progress bar.")
    # --- Added for DataCollator and TrainingArguments consistency ---
    pad_to_multiple_of: Optional[int] = Field(8, description="Pad sequences to a multiple of this value for efficiency. If None, no such padding.")
    label_smoothing_factor: float = Field(0.0, description="Label smoothing factor for loss calculation during training.")

    @model_validator(mode='after')
    def validate_training_setup(self) -> 'TrainingConfig':
        if self.training_mode != TrainingMode.SFT:
            raise ValueError("Only TrainingMode 'sft' is supported.")
        
        allowed_precisions = ["bf16", "fp16", "fp32"]
        if self.trainer_compute_precision not in allowed_precisions:
            raise ValueError(f"trainer_compute_precision must be one of {allowed_precisions}, got '{self.trainer_compute_precision}'.")

        if self.trainer_compute_precision in ["fp16", "bf16"]:
            if not torch.cuda.is_available():
                # This is a config-level check; Trainer will also check and might error or fallback.
                print(f"Warning: trainer_compute_precision is '{self.trainer_compute_precision}' but CUDA is not available. Effective precision may differ or training may fail.")
            if self.trainer_compute_precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                print(f"Warning: trainer_compute_precision is 'bf16' but the current CUDA device does not support bfloat16. Effective precision may differ or training may fail.")
        
        if self.save_strategy == "steps" and (self.save_steps is None or self.save_steps <= 0):
            print(f"Warning: save_strategy is 'steps' but save_steps is {self.save_steps}. Effective saving might be linked to logging_steps or epoch boundaries by Trainer if not a positive integer.")

        return self
    
    model_config = {
        "extra": "ignore", "validate_assignment": True
    }    

# --- Inference Session Configuration ---
class InferenceConfig(BaseModel):
    """Configuration for an inference session (primarily for default generation params)."""
    default_generation_config: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {},
        description="Default generation parameters for this inference session if no overrides from request."
    )


    @model_validator(mode='after')
    def validate_inference_setup(self) -> 'InferenceConfig':
        return self
    
    model_config = {
        "extra": "ignore", "validate_assignment": True
    }    

# --- Adapter Configuration (for adding new adapters dynamically) ---
class AdapterConfig(BaseModel):
    """Configuration for an adapter to be added to the engine's model."""
    adapter_name: Optional[str] = Field(
        None, 
        description="Unique name for the adapter. Required if 'adapter_path' is not provided. If 'adapter_path' is provided and this is None, the name may be inferred from metadata when loading an existing adapter."
    )
    adapter_type: Optional[AdapterType] = Field(
        None, 
        description="Type of adapter. Required if 'adapter_path' is not provided. If 'adapter_path' is provided and this is None, the type may be inferred from metadata when loading an existing adapter."
    )
    adapter_path: Optional[str] = Field(
        None,
        description="For existing adapters: path to a pre-trained adapter directory. For new adapters (is_new=True): root directory under which the model/precision/adapter folders will be created."
    )
    if_exists: IfExistsEnum = Field(
        IfExistsEnum.FAIL,
        description="Action to take if an adapter with the same path is already loaded. 'fail' (default), 'ignore', 'reload'."
    )
    is_new: bool = Field(
        False,
        description="If true, force creation of a new adapter definition using this config (ignore any existing checkpoints at adapter_path). adapter_path is treated as the adapters root; the engine will create the model/precision/adapter folder."
    )
    
    r: int = Field(8, description="LoRA rank (dimension).")
    lora_alpha: int = Field(16, description="LoRA alpha scaling factor.")
    lora_dropout: float = Field(0.0, description="Dropout probability for LoRA layers.")
    target_modules: Optional[List[str]] = Field(default=None, description="List of module names for LoRA. If None, modules will be inferred from model architecture.")

    @model_validator(mode='after')
    def validate_adapter_definition(self) -> 'AdapterConfig':
        if self.is_new:
            # New adapter request: require name and type so engine can build config.
            if self.adapter_name is None:
                raise ValueError("When 'is_new' is True, 'adapter_name' must be specified.")
            if self.adapter_type is None:
                raise ValueError("When 'is_new' is True, 'adapter_type' must be specified.")
            # adapter_path may point to adapters root and can be missing on disk; no existence check.
            return self

        if self.adapter_path is not None:
            # Case 1: adapter_path is provided.
            # It could be for loading an existing adapter, or for specifying the root directory of a new adapter.
            if self.adapter_type is None:
                # Subcase 1.1: adapter_path is provided, but adapter_type is NOT.
                # This implies loading an existing adapter. The path must exist and be a directory.
                if not os.path.isdir(self.adapter_path):
                    raise ValueError(
                        f"If 'adapter_path' ('{self.adapter_path}') is provided and 'adapter_type' is not specified "
                        f"(implying loading of an existing adapter), 'adapter_path' must be a valid directory. Path not found."
                    )
            # else:
                # Subcase 1.2: adapter_path is provided, AND adapter_type IS specified.
                # This implies defining a new adapter to be created at adapter_path.
                # The directory might not exist yet; the engine is responsible for creating it.
                # No os.path.isdir() check is performed here.
                # adapter_name can be None (engine will infer from path or use a default).
                # r, lora_alpha, etc., will be used for the new adapter's PeftConfig.
        else:
            # Case 2: adapter_path is NOT provided.
            # adapter_name and adapter_type are mandatory.
            if self.adapter_name is None:
                raise ValueError("If 'adapter_path' is not provided (i.e., defining a new adapter), 'adapter_name' must be specified.")
            if self.adapter_type is None:
                raise ValueError("If 'adapter_path' is not provided (i.e., defining a new adapter), 'adapter_type' must be specified.")
        return self
    
# --- Tool/Function Calling Schemas ---
class ToolParameters(BaseModel):
    """Defines the JSON schema for a tool's input parameters."""
    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)

class FunctionDefinition(BaseModel):
    """Describes a function's name, description, and parameters."""
    name: str
    description: str
    parameters: ToolParameters

class Tool(BaseModel):
    """Represents a tool that the model can call."""
    type: str = "function"
    # Replaced generic Dict with a specific Pydantic model for better validation.
    function: FunctionDefinition
    
class ToolsSerialized(BaseModel):
    """Represents a collection of tools and an optional header for an inference request."""
#    prompt_header: Optional[str] = Field(None, description="An optional header to prepend to the entire tool definitions block.")
#    prompt_footer: Optional[str] = Field(None, description="An optional footer to append to the entire tool definitions block.")
    for_dump: List[Tool] = Field(..., description="A list of tools the model may call.")
#    tool_footers: Optional[Dict[str, str]] = Field(None, description="A dictionary mapping tool names to optional footers to append to their individual definitions.")

# --- Inference Request/Response Classes ---

class InferenceRequest(BaseModel):
    """Configuration for a single inference request."""
    raw_list: Optional[List[str]] = Field(None, description="List of raw, pre-formatted prompt strings. Each string is passed to the model as-is without any templating. Mutually exclusive with 'messages_list'.")
    messages_list: Optional[List[List[Dict[str, Any]]]] = Field(None, description="List of message arrays for chat-style inference. The engine will apply the chat template. Mutually exclusive with 'raw_list'.")
    override_adapters: Optional[List[str]] = Field(None, description="List of adapter names to use for each prompt in the batch. Use '__base__' for base model. Length must match prompts/messages_list if provided.")
    active_adapters: Optional[List[str]] = Field(None, description="Explicit set of adapters to activate for this request. None will use the engine's current active set. An empty list forces the base model.")
    tools: Optional[ToolsSerialized] = Field(None, description="A serialized collection of tools and an optional prompt header. If provided, the model may generate tool_calls.")
    generation_config: Optional[Dict[str, Any]] = Field(None, description="Override generation parameters for this request.")
    request_id: Optional[str] = Field(None, description="Optional identifier for this request batch.")
    stream: bool = Field(False, description="Whether to stream the response token by token.")
    return_prompt: Optional[ReturnPromptEnum] = Field(
        None,
        description="Specifies whether to return the formatted prompt in the PROMPT_STARTED chunk. 'full': returns the entire formatted prompt. 'last': returns only the last non-assistant message."
    )
    suppress_full_response: bool = Field(False, description="If true, suppress complete response text in the final streaming chunk for each prompt.")
    reset_metrics: bool = Field(default=False, description="If true, clears the engine's inference metrics history before this request.")
    cache: Optional[str] = Field(
        None,
        description="Force a specific cache implementation for this request. Options: 'dynamic', 'static', 'offloaded', 'dynamic_reset', 'static_reset', 'no_cache'. The '_reset' suffix also clears all static cache slots before the request. Overrides engine's default routing."
    )
    do_continue: bool = Field(
        default=False,
        description="If true, the chat template will be applied with `add_generation_prompt=False`. This is useful for continuing a generation from an existing assistant message."
    )
    no_tools_parse: bool = Field(
        False,
        description="If true, skip parsing tool call blocks for this request and treat all generated output as plain text."
    )

    @model_validator(mode='after')
    def validate_input_format(self) -> 'InferenceRequest':
        if not self.raw_list and not self.messages_list:
            raise ValueError("Either 'raw_list' or 'messages_list' must be provided")
        if self.raw_list and self.messages_list:
            raise ValueError("Only one of 'raw_list' or 'messages_list' should be provided, not both")

        num_raw = len(self.raw_list) if self.raw_list else 0
        num_messages = len(self.messages_list) if self.messages_list else 0
        num_requests = num_raw or num_messages

        if self.override_adapters is not None:
            # If there is more than one prompt, the number of adapters must match.
            if num_requests > 1 and len(self.override_adapters) != num_requests:
                raise ValueError(
                    f"When providing multiple prompts ({num_requests}), 'override_adapters' length ({len(self.override_adapters)}) must match."
                )
            # If there are no prompts, adapter_names should not be provided.
            if num_requests == 0 and self.override_adapters:
                 raise ValueError("'override_adapters' cannot be provided without 'raw_list' or 'messages_list'.")

        return self
    
    model_config = {
        "extra": "forbid", "arbitrary_types_allowed": True
    }

# --- API Response Data Models ---

class FormattedPromptData(BaseModel):
    formatted_prompts: List[str]
    prompt_token_counts: Optional[List[int]] = None
    errors: Optional[List[str]] = None

class TokenCountData(BaseModel):
    token_count: int
    text_processed: str

class AdapterDetailsData(BaseModel):
    name: str
    is_active: bool
    type: Optional[str] = None
    root_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    base_model_quant: Optional[str] = None
    base_model_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    alias: Optional[str] = None
    is_loaded: Optional[bool] = None
    is_foreign: Optional[bool] = None

    model_config = {
        "extra": "allow",
        "arbitrary_types_allowed": True,
    }

class LoadedAdaptersData(BaseModel):
    adapters: List[AdapterDetailsData]

class ToolsListData(BaseModel):
    tools: List[str]

# --- Main API Response Model ---

class MP13Response(BaseModel):
    """Standardized response object for API calls."""
    status: str = Field(..., description="Status of the operation (e.g., 'success', 'error').")
    message: str = Field(..., description="A human-readable message describing the result.")
    details: Dict[str, Any] = Field(default_factory=dict, description="A dictionary containing detailed information about the response.")
    # The `data` field is now a Union of all possible structured response data types.
    # For responses without a specific data model (like shutdown-engine), it will be None.
    data: Optional[Union[
        FormattedPromptData,
        TokenCountData,
        AdapterDetailsData,
        LoadedAdaptersData,
        ToolsListData,
        # Add other response data models here as they are created.
        # For now, other data will be passed as a dict and validated by Pydantic.
        Dict[str, Any] 
    ]] = Field(None, description="The primary data payload of the response, if any.")
    stream: Optional[AsyncIterator['InferenceResponse']] = Field(None, description="An asynchronous iterator for streaming responses.", exclude=True)

    model_config = {
        "arbitrary_types_allowed": True,
    }


class InferenceResponse(BaseModel):
    """
    Represents a single data chunk sent from the engine to the client during inference.
    The 'chunkType' field determines the type of chunk and which other fields are relevant.
    """
    # --- Core Fields (present on most chunks) ---
    chunkType: ChunkType = Field(..., description=f"The type of this chunk. E.g., '{ChunkType.PROMPT_STARTED.value}', '{ChunkType.STREAMING_CHUNK.value}', '{ChunkType.STREAMING_ENDED.value}', '{ChunkType.ERROR.value}'.")
    prompt_index: Optional[int] = Field(None, description="The 0-based index of the prompt this chunk belongs to within the batch.")

    # --- Fields for chunkType=PROMPT_STARTED ---
    prompt: Optional[str] = Field(None, description=f"The formatted prompt text, sent with chunkType='{ChunkType.PROMPT_STARTED.value}' if requested via InferenceRequest.return_prompt.")
    adapters: Optional[str] = Field(None, description="The adapter(s) used for this prompt. '__base__' indicates the base model; multiple adapters are comma-separated.")

    # --- Fields for chunkType=STREAMING_CHUNK ---
    chunk_text: Optional[str] = Field(None, description="A single generated token or text chunk (tool call markup stripped when tool parsing is enabled).")
    tool_blocks: Optional[List['ToolCallBlock']] = Field(None, description="A list of tool call blocks found in the response.")
    is_final_chunk: bool = Field(False, description="Indicates if this is the final content chunk for this item.")

    # --- Per-Item Metrics (sent with the final 'streaming_chunk' for an item) ---
    response_text: Optional[str] = Field(None, description="Full response sent with the final chunk when requested (includes raw tool call markup when tool parsing is enabled).")
    input_tokens: Optional[int] = Field(None, description="Number of input tokens for this prompt.")
    output_tokens: Optional[int] = Field(None, description="Number of output tokens generated for this prompt.")
    generation_duration_sec: Optional[float] = Field(None, description="Time taken to generate response for this prompt (seconds).")
    tokens_per_second: Optional[float] = Field(None, description="Tokens per second for this prompt's generation.")
    time_to_first_token_sec: Optional[float] = Field(None, description="Latency to the first token (seconds).")
    was_canceled: Optional[bool] = Field(None, description="Indicates if the generation was cancelled by the user.")
    was_truncated: Optional[bool] = Field(None, description="Indicates if the generation was likely truncated by hitting max_new_tokens.")
    cache_metric: Optional[str] = Field(None, description="Cache mode used for this item (e.g., 'static (B=1, L=4096)', 'dynamic (L=512)').")
    cache_warming: Optional[str] = Field(None, description="Static cache slots queued as the result of this response such as (B=1,L=8192).")
    tool_blocks_count: Optional[int] = Field(None, description="Number of tool call blocks generated for this prompt.")
    tool_blocks_tokens: Optional[int] = Field(None, description="Number of tokens in the tool call blocks for this prompt.")

    # --- Aggregate Metrics (sent with chunkType=STREAMING_ENDED) ---
    total_input_tokens: Optional[int] = Field(None, description="Total input tokens for the entire batch.")
    total_output_tokens: Optional[int] = Field(None, description="Total output tokens for the entire batch.")
    total_generation_duration_sec: Optional[float] = Field(None, description="Total generation duration for the entire batch.")
    overall_tps: Optional[float] = Field(None, description="Overall tokens per second for the batch.")
    avg_time_to_first_token_sec: Optional[float] = Field(None, description="Average TTFT for the batch.")
    total_tool_blocks: Optional[int] = Field(None, description="Total tool call blocks for the batch.")
    total_tool_blocks_tokens: Optional[int] = Field(None, description="Total tool call tokens for the batch.")
    cache_queued: Optional[str] = Field(None, description="Static cache slots that are pending background warmup, if any.")
    in_flight_req: Optional[int] = Field(None, description="Current count of inference requests still in progress.")
    mem_allocated: Optional[float] = Field(None, description="Currently allocated GPU memory.")
    mem_reserved: Optional[float] = Field(None, description="Currently reserved GPU memory.")
    had_error: Optional[bool] = Field(None, description="Indicates whether any error chunk was emitted for the request.")
    # --- Fields for chunkType=ERROR ---
    error: Optional[str] = Field(None, description="Error message, if an error occurred.")
    full_traceback: Optional[str] = Field(None, description="Full traceback string if an unexpected error occurred (on final chunk).")

    model_config = {
        "arbitrary_types_allowed": True,
    }
# ============================================================
@dataclass
class ParserProfile:
    key: str
    # Markers used by UnifiedToolIO.parse_model_output to find tool blocks in raw model text.
    block_start: List[str] = field(default_factory=list)
    block_end: List[str] = field(default_factory=list)
    # Hard-stop tokens that terminate a block even without block_end (used in stream parsing).
    hard_stop_patterns: List[str] = field(default_factory=list)
    # How to interpret the JSON payload inside a block (drives _payload_to_calls/serialize_calls).
    payload_mode: str = "json_obj_or_list"
    # Dotted path to the tool name and arguments inside the payload (used in _payload_to_calls/_set_nested).
    name_field: str = "name"
    arguments_field: str = "arguments"
    # Optional path for a tool call id inside the payload (parsed and re-serialized when present).
    id_field: Optional[str] = None
    # If True, attempt to JSON-decode string arguments (helps models that double-encode).
    arguments_may_be_string: bool = False
    # If True, tools are injected via system prompt rather than chat_template tools arg (see mp13_utils.build_messages).
    tools_in_system_prompt: bool = False
    # Wrappers used when results are emitted as user-role text blocks (results_as_user_role).
    result_wrapper_start: Optional[str] = None
    result_wrapper_end: Optional[str] = None
    # Require attaching an id to each result payload (checked in serialize_calls when is_result).
    result_requires_id: bool = False
    # Key under which the result payload is stored for tool-role results (non message-style).
    result_payload_key: Optional[str] = None
    # Key name for result ids when required (e.g., Mistral uses tool_call_id).
    result_id_key: Optional[str] = None
    # If True, collapse multiple results into one JSON array block (non message-style).
    result_as_single_block_list: bool = False
    # Hard cap on parsed tool JSON size (used during parsing/safety).
    max_tool_json_bytes: int = 1_000_000
    # Emit tool results as user-role text instead of tool-role (serialize_calls/is_result path).
    results_as_user_role: bool = False
    # Emit multiple results inside one role message instead of per-call messages (non message-style).
    pack_results_as_one_role: bool = False
    # Emit results as a list of per-call message payloads (e.g., tool_results role); handled in serialize_calls.
    results_as_messages: bool = False
    # Content key to use when results_as_messages is True (e.g., "content" for Mistral).
    results_message_content_key: Optional[str] = None
    # If True, attach a `tool_calls` array to the reconstructed assistant stub (for templates that expect it).
    emit_tool_calls_field: bool = False
    # If True, end markers are matched only when not inside JSON string literals.
    end_marker_outside_json_string: bool = False
    # If True, allow EOS tokens to remain in the tools parser input (they will still be stripped from text output).
    preserve_eos_for_parsing: bool = False
    # Metadata: does the chat template reference `tools`? set in guess_profile_from_template.
    template_accepts_tools_arg: bool = False
    tool_handling_hint: Optional[str] = None
    # Enforce 9-char alphanumeric tool_call IDs when required by a template (checked in serialize_calls).
    enforce_tool_call_id_format: bool = False
    # Tagged tool call formats: name is plain text followed by an args marker (e.g., [TOOL_CALLS]name[ARGS]{...}).
    # When set, the parser treats block_start as the call marker and uses this marker to locate JSON args.
    tagged_call_args_marker: Optional[str] = None

@dataclass
class RegisteredTool:
    """
    A container for a tool's definition, implementation, and optional guide.
    This structure allows a tool and its guide to be managed as a single unit.
    """
    name: str
    definition: Dict[str, Any]
    implementation: Callable[..., Any]
    guide_definition: Optional[Dict[str, Any]] = None
    guide_implementation: Optional[Callable[..., Any]] = None
    is_intrinsic: bool = False

# ============================================================
# Unified Tool Call objects
# ============================================================

@dataclass
class ToolCall:
    """Represents a single, parsed tool call with its name and arguments."""
    # Action constants, more than one can be specified. If none is set it pick result if not empty or error field. 
    Ignore: ClassVar[str] = "ignore"    # do include original llm call but do not execute or ignore it's result
    Strip: ClassVar[str] = "strip"      # strip original llm call and ignore result. To avoid empty assistant reponse in prompt,
                                        # the error field if exists will replace tool call in original assistant response.
    KeepRaw: ClassVar[str] = "keep_raw" # unless 'stripped' override tool call normalization in the prompt.
    Retry: ClassVar[str] = "retry"      # explictly says to send llm an error (pick error field over result).
    Abort: ClassVar[str] = "abort"      # stop auto tool execution, do not send result.

    # Fields without default values must come first
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)

    # Fields with default values
    id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    raw: Optional[str] = None
    model_format: Optional[str] = None
    parse_errors: List[str] = field(default_factory=list)
    action: List[str] = field(default_factory=list)

    def normalize_error(self, error: Union[str, Exception]) -> str:
        """
        Cleans up an error string by un-escaping characters for readability.
        It's designed to make errors containing code snippets (which often get escaped)
        easier to read. For example, it turns "\\n" into a newline.
        """
        error_str = str(error)
        try:
            # A simple unescape is often sufficient and safer than complex regex.
            return codecs.decode(error_str, 'unicode_escape')
        except Exception:
            # If unescaping fails (e.g., not a valid escape sequence), return the original.
            return error_str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "id": self.id,
            "result": self.result,
            "error": self.error,
            "raw": self.raw,
            "model_format": self.model_format,
            "parse_errors": list(self.parse_errors) if self.parse_errors else [],
            "action": list(self.action) if self.action else [],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        if isinstance(data, cls):
            return data
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments") or {},
            id=data.get("id"),
            result=data.get("result"),
            error=data.get("error"),
            raw=data.get("raw"),
            model_format=data.get("model_format"),
            parse_errors=list(data.get("parse_errors", []) or []),
            action=list(data.get("action", []) or []),
        )


@dataclass
class ToolCallBlock:
    """
    Represents a contiguous tool-call block found in the model's output text.
    This block may contain one or more individual tool calls.
    """
    # Fields without default values must come first
    raw_block: str = ""

    # Fields with default values
    normalized_block: Optional[str] = None
    calls: List[ToolCall] = field(default_factory=list)
    prompt_index: Optional[int] = None
    model_format: Optional[str] = None
    block_start_pos: int = -1 
    block_end_pos: Optional[int] = None
    payload_start_pos: Optional[int] = None
    payload_end_pos: Optional[int] = None
    start_marker: Optional[str] = None
    end_marker: Optional[str] = None
    hard_stop_marker: Optional[str] = None
    position_mode: Optional[str] = None
    parse_errors: List[str] = field(default_factory=list)
    error_block: Optional[str] = None
    action_block: List[str] = field(default_factory=list)
    is_incomplete: bool = False

    def normalize_error(self, error: Union[str, Exception]) -> str:
        """
        Cleans up an error string by un-escaping characters for readability.
        It's designed to make errors containing code snippets (which often get escaped)
        easier to read. For example, it turns "\\n" into a newline.
        """
        error_str = str(error)
        try:
            # A simple unescape is often sufficient and safer than complex regex.
            return codecs.decode(error_str, 'unicode_escape')
        except Exception:
            # If unescaping fails (e.g., not a valid escape sequence), return the original.
            return error_str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_block": self.raw_block,
            "normalized_block": self.normalized_block,
            "calls": [call.to_dict() if isinstance(call, ToolCall) else call for call in (self.calls or [])],
            "prompt_index": self.prompt_index,
            "model_format": self.model_format,
            "block_start_pos": self.block_start_pos,
            "block_end_pos": self.block_end_pos,
            "payload_start_pos": self.payload_start_pos,
            "payload_end_pos": self.payload_end_pos,
            "start_marker": self.start_marker,
            "end_marker": self.end_marker,
            "hard_stop_marker": self.hard_stop_marker,
            "position_mode": self.position_mode,
            "parse_errors": list(self.parse_errors) if self.parse_errors else [],
            "error_block": self.error_block,
            "action_block": list(self.action_block) if self.action_block else [],
            "is_incomplete": self.is_incomplete,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallBlock":
        if isinstance(data, cls):
            return data
        calls_raw = data.get("calls", []) or []
        calls: List[ToolCall] = []
        for call in calls_raw:
            if isinstance(call, ToolCall):
                calls.append(call)
            elif isinstance(call, dict):
                try:
                    calls.append(ToolCall.from_dict(call))
                except Exception:
                    continue
        return cls(
            raw_block=data.get("raw_block", ""),
            normalized_block=data.get("normalized_block"),
            calls=calls,
            prompt_index=data.get("prompt_index"),
            model_format=data.get("model_format"),
            block_start_pos=data.get("block_start_pos", -1),
            block_end_pos=data.get("block_end_pos"),
            payload_start_pos=data.get("payload_start_pos"),
            payload_end_pos=data.get("payload_end_pos"),
            start_marker=data.get("start_marker"),
            end_marker=data.get("end_marker"),
            hard_stop_marker=data.get("hard_stop_marker"),
            position_mode=data.get("position_mode"),
            parse_errors=list(data.get("parse_errors", []) or []),
            error_block=data.get("error_block"),
            action_block=list(data.get("action_block", []) or []),
            is_incomplete=bool(data.get("is_incomplete", False)),
        )


# Update forward reference
InferenceResponse.model_rebuild()
InferenceRequest.model_rebuild()
