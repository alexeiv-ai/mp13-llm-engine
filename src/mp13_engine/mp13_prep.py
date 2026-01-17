# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""MP13 Engine - Dataset handling and preprocessing utilities."""

import os
import logging 
from typing import Dict, Any, List, Callable, Optional

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Import from local modules
from .mp13_config import DatasetFormat, ColumnsConfig, DatasetTags, PreprocessingMode
from .mp13_state import DatasetError # Keep if used, else remove
from .mp13_utils import format_prompt_messages

def get_sft_formatting_func(logger: logging.Logger, formatting: DatasetFormat, columns: ColumnsConfig, tokenizer: Optional[PreTrainedTokenizerBase] = None, tags: Optional[DatasetTags] = None) -> Callable[[Dict[str, Any]], str]:
    """Returns the appropriate prompt formatting function based on the dataset format."""
    if formatting == DatasetFormat.MESSAGES:
        # format_prompt_messages is designed to handle this.
        # It needs the example dictionary, columns config, tokenizer, and tags.
        # For SFT, the example 'ex' contains the full conversation including assistant's turn.
        # So, add_generation_prompt should be False.
        return lambda ex: format_prompt_messages(logger, ex, columns, tokenizer, tags, add_generation_prompt=False).text
    elif formatting == DatasetFormat.TEXT:
        # For plain text, we might assume a simple concatenation or specific column use.
        # If text column is supposed to contain pre-formatted prompt+response, this is okay.
        # If it's just "text" and needs special handling for SFT, this needs more logic.
        # For now, assume format_prompt_messages can handle it if columns.text is set
        # and the content of that column is the full thing to tokenize.
        # Or, if prompt/response columns are used with TEXT format, it falls back to that.
        logger.info(f"Using SFT formatting for TEXT format. Ensure columns '{columns.text}', or '{columns.prompt}'/'{columns.response}' "
               f"are appropriately set and handled by format_prompt_messages or your data structure. "
               f"add_generation_prompt=False will be used.")
        return lambda ex: format_prompt_messages(logger, ex, columns, tokenizer, tags, add_generation_prompt=False).text
    else:
        # Fallback or error for other SFT formats if any were planned (e.g. Alpaca-specific)
        # The current format_prompt_messages is quite generic for message-like structures.
        # If a non-messages format implies specific prompt/response field usage,
        # format_prompt_messages' fallback to prompt/query/response might cover it.
        logger.warning(f"Warning: SFT formatting for '{formatting}' relies on 'format_prompt_messages'. Ensure data columns align. "
               f"add_generation_prompt=False will be used.")
        return lambda ex: format_prompt_messages(logger, ex, columns, tokenizer, tags, add_generation_prompt=False).text


def preprocess_sft_full_text(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerBase,
    formatting_func: Callable[[Dict[str, Any]], str],
    max_sequence_length: int
    # columns: ColumnsConfig removed, formatting_func encapsulates column access via its closure
) -> Dict[str, List[Any]]:
    """
    Full-text preprocessing for Supervised Fine-Tuning (Causal LM).
    1. Format the entire conversation/text as a single string using formatting_func.
    2. Tokenize the full text.
    3. Set labels to be a copy of input_ids (Trainer's DataCollatorForLanguageModeling with mlm=False handles padding tokens in labels).
    """
    first_col_key = list(examples.keys())[0]
    num_examples = len(examples[first_col_key])
    full_texts = []

    for i in range(num_examples):
        example_dict = {key: examples[key][i] for key in examples.keys()}
        
        full_text = formatting_func(example_dict) # formatting_func has access to columns via its closure
        
        if full_text and tokenizer.eos_token is not None and not full_text.endswith(tokenizer.eos_token):
            full_text += tokenizer.eos_token
            
        full_texts.append(full_text)

    tokenized_outputs = tokenizer(
        full_texts, 
        padding=False, # Do not pad here; let the DataCollator handle dynamic padding per batch.
        truncation=False, # Set to False to align with train_lora.py filtering strategy.
        return_attention_mask=True,
        return_length=True # Required for filtering long sequences in the next step.
    )

    # Add a 'keep' flag to be used for filtering, aligning with train_lora.py's hashable filter approach.
    tokenized_outputs["keep"] = [l <= max_sequence_length for l in tokenized_outputs["length"]]

    return {
        "input_ids": tokenized_outputs["input_ids"],
        "attention_mask": tokenized_outputs["attention_mask"],
        "length": tokenized_outputs["length"], # Return length for filtering.
        "keep": tokenized_outputs["keep"],
        # The 'labels' field is correctly omitted. The DataCollatorForLanguageModeling
        # with mlm=False will automatically create it by cloning 'input_ids'.
    }

def preprocess_sft(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerBase,
    formatting_func: Callable[[Dict[str, Any]], str],
    max_sequence_length: int,
    # columns: ColumnsConfig, # Removed, formatting_func has it
    preprocessing_mode: str = "full_text" # from DatasetConfig
) -> Dict[str, List[Any]]:
    """Preprocess data for Supervised Fine-Tuning."""
    # The `formatting_func` encapsulates the logic for both FULL_TEXT and APPLY_CHAT_TEMPLATE,
    # so no conditional logic is needed here based on preprocessing_mode.
    return preprocess_sft_full_text(
            examples, tokenizer, formatting_func, max_sequence_length #, columns removed
    )

def map_preprocess_sft(
    examples: Dict[str, List[Any]],
    tokenizer_ref: PreTrainedTokenizerBase,
    formatting_func_ref: Callable[[Dict[str, Any]], str],
    max_seq_len_ref: int,
    proc_mode_ref: str
) -> Dict[str, List[Any]]:
    """Top-level helper for dataset.map to avoid hashing complex objects."""
    return preprocess_sft(
        examples,
        tokenizer=tokenizer_ref,
        formatting_func=formatting_func_ref,
        max_sequence_length=max_seq_len_ref,
        preprocessing_mode=proc_mode_ref
    )
