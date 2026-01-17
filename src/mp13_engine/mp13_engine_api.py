# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""
MP13 API - Unified API for a multi-instance, concurrent engine.

This module provides a single entry point (`handle_call_tool`) to manage and interact
with one or more MP13Engine instances running in the same process.

---
### Usage Pattern 1: Single Default Engine
---
This is the simplest workflow, ideal for applications that only need one model.

1.  **Initialize the engine.** The first engine to be created automatically
    becomes the "default". The `instance_id` returned is its unique alias.
    ```python
    # Returns an alias like "mp13_engine"
    init_response = await handle_call_tool(
        "initialize-engine",
        {"base_model_name_or_path": "meta-llama/Llama-2-7b-chat-hf"}
    )
    engine_alias = init_response.data["instance_id"]
    ```

2.  **Use the engine.** Make subsequent calls without specifying the `instance_id`.
    They will automatically target the default engine.
    ```python
    status_response = await handle_call_tool("get-engine-status", {})
    ```

---
### Usage Pattern 2: Multiple Concurrent Engines
---
This pattern allows you to run multiple models or configurations concurrently.

1.  **Initialize multiple engines.** Call `initialize-engine` for each desired
    configuration. It's recommended to provide a friendly `instance_id` (alias)
    to keep track of them. The API will ensure the final alias is unique.
    ```python
    # Initialize a Llama model
    llama_response = await handle_call_tool(
        "initialize-engine",
        {
            "instance_id": "llama_engine",
            "base_model_name_or_path": "meta-llama/Llama-2-7b-chat-hf"
        }
    )
    llama_alias = llama_response.data["instance_id"]

    # Initialize a Phi-3 model
    phi_response = await handle_call_tool(
        "initialize-engine",
        {
            "instance_id": "phi3_engine",
            "base_model_name_or_path": "microsoft/Phi-3-mini-4k-instruct"
        }
    )
    phi_alias = phi_response.data["instance_id"]
    ```

2.  **Target specific engines.** Use the unique alias in the `instance_id` field
    for all subsequent calls to ensure you're targeting the correct engine.
    ```python
    # Run inference on the Phi-3 engine
    await handle_call_tool(
        "run-inference",
        {
            "instance_id": phi_alias,
            "messages_list": [[{"role": "user", "content": "Hello"}]]
        }
    )
    ```

3.  **Manage Instances.** You can list all running engines or change the default.
    ```python
    await handle_call_tool("list-engines", {})
    await handle_call_tool("set-default-engine", {"instance_id": llama_alias})
    ```

**Note on Concurrency:** To run operations on multiple engines truly concurrently,
the application should schedule the `handle_call_tool` calls in separate
`asyncio.Task`s (e.g., using `asyncio.gather`).
"""

import asyncio
import traceback
import threading
from typing import Dict, Any, Union, Optional, AsyncIterator, Tuple

from pydantic import ValidationError

# Use absolute imports for clarity within the package
from .mp13_config import (
    InferenceRequest, AdapterConfig,
    GlobalEngineConfig, EngineMode, MP13Response, APIStatus,
    FormattedPromptData, TokenCountData, AdapterDetailsData, LoadedAdaptersData, ToolsListData,
    ChunkType
)
from .mp13_state import (
    ServerStatus,
    ConfigurationError, TrainingError, EngineError, ModeMismatchError,
    EngineInitializationError, AdapterError, BusyError
)
from .mp13_engine import MP13Engine, logger

# --- Multi-Engine Management Globals ---
_api_lock = threading.Lock()
_ENGINE_INSTANCES: Dict[int, MP13Engine] = {}
_ALIAS_TO_ID: Dict[str, int] = {}
_DEFAULT_ENGINE_ALIAS: Optional[str] = None
# --- End Globals ---


# Helper function for formatting API responses
def _create_response(
    status: str, # "success" or "error"
    message: str,
    data: Optional[Any] = None,
    details: Optional[Dict[str, Any]] = None,
    stream: Optional[Any] = None
) -> MP13Response:
    """Creates a standardized MP13Response object."""
    response_details = details or {}
    if status == APIStatus.ERROR.value and "error_message" in response_details:
        message = response_details["error_message"]
    return MP13Response(
        status=status,
        message=message,
        details=response_details,
        data=data,
        stream=stream
    )

def inference_stream_to_dict_stream(stream: AsyncIterator[Any]) -> AsyncIterator[Dict[str, Any]]:
    """Convert engine-streamed InferenceResponse objects into JSON-friendly dicts."""
    async def _generator():
        async for item in stream:
            try:
                if hasattr(item, "model_dump"):
                    yield item.model_dump(exclude_none=True)
                else:
                    yield item
            except Exception as e:
                yield {
                    "chunkType": ChunkType.ERROR.value,
                    "error": f"Client-side serialization error: {e}",
                    "details": {"raw_response": str(item)},
                }
    return _generator()

def _extract_exception_message_and_details(e: Exception) -> Tuple[str, Dict[str, Any]]:
    """Return (message, details) where details is a dict when available."""
    details: Dict[str, Any] = {}
    message = str(e)
    if hasattr(e, "args") and len(e.args) > 0:
        first = e.args[0]
        if isinstance(first, dict):
            details = first
            message = details.get("message", message)
        elif isinstance(first, str):
            message = first
    return message, details

def get_engine_instance_for_direct_use(alias: Optional[str] = None) -> Optional[MP13Engine]:
    """
    Returns a specific MP13Engine instance for direct use cases like event subscription.
    If alias is None, returns the default engine. Returns None if not found.
    """
    try:
        return _get_engine(alias)
    except EngineError:
        return None

def _get_engine(alias: Optional[str]) -> MP13Engine:
    """
    Resolves an alias (or the default) to a specific MP13Engine instance.
    This function is central to multi-instance routing.
    Raises: EngineError if the requested instance is not found or not available.
    """
    global _DEFAULT_ENGINE_ALIAS, _ALIAS_TO_ID, _ENGINE_INSTANCES
    with _api_lock:
        target_alias = alias
        if target_alias is None:
            target_alias = _DEFAULT_ENGINE_ALIAS
            if target_alias is None:
                raise EngineError("No instance_id (alias) provided and no default engine is set.")

        engine_id = _ALIAS_TO_ID.get(target_alias)
        if engine_id is None:
            raise EngineError(f"Engine instance with alias '{target_alias}' not found.")

        engine = _ENGINE_INSTANCES.get(engine_id)
        if engine is None:
            logger.critical(f"CRITICAL STATE DESYNC: Alias '{target_alias}' exists but ID '{engine_id}' does not map to an engine instance.")
            raise EngineError(f"Engine instance for alias '{target_alias}' is unavailable due to an internal state error.")

        return engine

async def handle_call_tool(
    name: str,
    arguments: Dict[str, Any] | None
) -> MP13Response:
    """
    Handles incoming tool execution requests for both training and inference.
    Routes calls to the correct engine instance based on the 'instance_id' argument (alias).
    """
    global _DEFAULT_ENGINE_ALIAS, _ENGINE_INSTANCES, _ALIAS_TO_ID
    arguments = arguments or {}

    # --- Engine Lifecycle and Management Tools (do not use _get_engine) ---
    if name == "initialize-engine":
        try:
            config = GlobalEngineConfig(**arguments)
            
            with _api_lock:
                # 1. Determine a unique alias
                base_alias = config.instance_id or "mp13_engine"
                final_alias = base_alias
                counter = 2
                while final_alias in _ALIAS_TO_ID:
                    final_alias = f"{base_alias}_{counter}"
                    counter += 1

                if final_alias != base_alias:
                    logger.warning(f"Requested instance_id '{base_alias}' already exists. Using unique alias '{final_alias}'.")

                # 2. Create and store the engine instance
                engine = MP13Engine(instance_id=final_alias)
                engine_id = id(engine)
                _ENGINE_INSTANCES[engine_id] = engine
                _ALIAS_TO_ID[final_alias] = engine_id

                # Automatically set the first engine as the default
                if _DEFAULT_ENGINE_ALIAS is None:
                    _DEFAULT_ENGINE_ALIAS = final_alias
                    logger.info(f"First engine initialized. Setting '{final_alias}' as the default instance.")
            
            logger.info(f"[API][initialize-engine] Creating new engine with alias '{final_alias}' (ID: {engine_id}).")
            
            # 3. Initialize resources (outside the lock)
            init_report = await engine.initialize_global_resources(config)
            
            # Add the final alias to the response data
            init_report["instance_id"] = final_alias
            
            return _create_response(status=APIStatus.SUCCESS.value, message=f"Engine '{final_alias}' initialized successfully.", data=init_report)

        except ValidationError as e:
            error_details = [f"Field '{' -> '.join(map(str, err['loc']))}': {err['msg']}" for err in e.errors()]
            full_error_summary = f"GlobalEngineConfig Validation Error: {'; '.join(error_details)}"
            return _create_response(status=APIStatus.ERROR.value, message=full_error_summary)
        except (EngineInitializationError, ConfigurationError) as e:
            return _create_response(status=APIStatus.ERROR.value, message=f"Initializing engine failed: {e}")
        except Exception as e:
            return _create_response(status=APIStatus.ERROR.value, message=f"Failed to initialize engine ({type(e).__name__}). Check logs.", details={"error_str": str(e)})

    elif name == "shutdown-engine":
        if arguments.get("shutdown_all") is True:
            logger.info("[API][shutdown-engine] Processing 'shutdown_all=True' request...")
            with _api_lock:
                aliases_to_shut = list(_ALIAS_TO_ID.keys())
                if not aliases_to_shut:
                    return _create_response(status=APIStatus.SUCCESS.value, message="No running engines to shut down.")
            
            success_count = 0
            fail_count = 0
            for alias in aliases_to_shut:
                try:
                    engine_to_shut = _get_engine(alias)
                    await engine_to_shut.shutdown_global_resources()
                    logger.info(f"Successfully shut down instance '{alias}'.")
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to shut down instance '{alias}': {e}")
                    fail_count += 1
            
            with _api_lock:
                _ENGINE_INSTANCES.clear()
                _ALIAS_TO_ID.clear()
                _DEFAULT_ENGINE_ALIAS = None

            message = f"Shutdown all complete. Success: {success_count}, Failed: {fail_count}."
            status = APIStatus.SUCCESS.value if fail_count == 0 else APIStatus.ERROR.value
            return _create_response(status=status, message=message)

        alias_to_shut = arguments.get("instance_id")
        if not alias_to_shut:
            return _create_response(status=APIStatus.ERROR.value, message="Missing 'instance_id' (alias) to shut down. To shut down all, please provide 'shutdown_all: true'.")
            
        with _api_lock:
            engine_id = _ALIAS_TO_ID.get(alias_to_shut)
            if engine_id is None or engine_id not in _ENGINE_INSTANCES:
                return _create_response(status=APIStatus.ERROR.value, message=f"Engine with alias '{alias_to_shut}' not found.")
            engine = _ENGINE_INSTANCES[engine_id]
        
        try:
            await engine.shutdown_global_resources()
            
            with _api_lock:
                del _ENGINE_INSTANCES[engine_id]
                del _ALIAS_TO_ID[alias_to_shut]
                if _DEFAULT_ENGINE_ALIAS == alias_to_shut:
                    _DEFAULT_ENGINE_ALIAS = None
                    logger.info(f"Default engine '{alias_to_shut}' was shut down. No default is set.")

            logger.info(f"Engine '{alias_to_shut}' shut down and removed successfully.")
            return _create_response(status=APIStatus.SUCCESS.value, message=f"Engine '{alias_to_shut}' shut down successfully.")
        except Exception as e:
            return _create_response(status=APIStatus.ERROR.value, message=f"Error during shutdown of '{alias_to_shut}': {e}")

    elif name == "list-engines":
        with _api_lock:
            if not _ENGINE_INSTANCES:
                return _create_response(status=APIStatus.SUCCESS.value, message="No engine instances are currently running.", data={"engines": []})

            engine_list = []
            for alias, engine_id in _ALIAS_TO_ID.items():
                engine = _ENGINE_INSTANCES.get(engine_id)
                if engine:
                    status_dict = await engine.get_engine_status()
                    status_dict['instance_id'] = alias
                    status_dict['is_default'] = (alias == _DEFAULT_ENGINE_ALIAS)
                    engine_list.append(status_dict)
            return _create_response(status=APIStatus.SUCCESS.value, message="Engines listed successfully.", data={"engines": engine_list})

    elif name == "set-default-engine":
        alias_to_set = arguments.get("instance_id")
        with _api_lock:
            if alias_to_set is None:
                _DEFAULT_ENGINE_ALIAS = None
                return _create_response(status=APIStatus.SUCCESS.value, message="Default engine has been unset.")
            
            if alias_to_set not in _ALIAS_TO_ID:
                return _create_response(status=APIStatus.ERROR.value, message=f"Cannot set default: alias '{alias_to_set}' not found.")
            
            _DEFAULT_ENGINE_ALIAS = alias_to_set
            return _create_response(status=APIStatus.SUCCESS.value, message=f"Engine '{alias_to_set}' is now the default.")

    # --- All other tools must resolve to a specific engine instance ---
    else:
        try:
            # Pop instance_id so it doesn't get passed to the engine method itself
            instance_alias = arguments.pop("instance_id", None)
            engine = _get_engine(instance_alias)
        except EngineError as e:
            return _create_response(status=APIStatus.ERROR.value, message=str(e))
        except Exception as e:
            return _create_response(status=APIStatus.ERROR.value, message=f"Unexpected error resolving engine instance: {e}")

        # --- Route to the appropriate method on the resolved 'engine' object ---
        try:
            if name == "check-set-mode":
                mode_str = arguments.get("mode")
                if not mode_str: return _create_response(status=APIStatus.ERROR.value, message="Missing 'mode' argument.")
                result = await engine.check_set_mode(EngineMode(mode_str), arguments.get("force", False))
                return _create_response(status=APIStatus.SUCCESS.value, message=result.get("message"), data=result)

            elif name == "get-engine-status":
                status_data = await engine.get_engine_status()
                return _create_response(status=APIStatus.SUCCESS.value, message="Engine status retrieved.", data=status_data)

            elif name == "start-training":
                result = await engine.start_training(arguments)
                return _create_response(status=APIStatus.SUCCESS.value, message=result.get("message", "Training started."), data=result)

            elif name == "stop-training":
                result = await engine.stop_training()
                return _create_response(status=APIStatus.SUCCESS.value, message=result.get("message", "Stop training initiated."), data=result)

            elif name == "get-training-status":
                status_data = await engine.get_training_status()
                return _create_response(status=APIStatus.SUCCESS.value, message="Training status retrieved.", data=status_data)
                
            elif name == "run-inference":
                request = InferenceRequest(**arguments)
                stream = engine.run_inference(request)
                return _create_response(status=APIStatus.SUCCESS.value, message="Inference stream started.", stream=stream)

            elif name == "get-inference-status":
                status_data = await engine.get_inference_status()
                return _create_response(status=APIStatus.SUCCESS.value, message="Inference status retrieved.", data=status_data)

            elif name == "format-inference-prompt":
                request = InferenceRequest(**arguments)
                result_dict = await engine.format_inference_prompt(request)
                return _create_response(status=APIStatus.SUCCESS.value, message="Prompt formatting completed.", data=FormattedPromptData(**result_dict))

            elif name == "count-tokens":
                text_to_count = arguments.get("text")
                if text_to_count is None: return _create_response(status=APIStatus.ERROR.value, message="Missing 'text' argument.")
                result_dict = await engine.count_tokens(text_to_count, arguments.get("is_repr", False))
                return _create_response(status=APIStatus.SUCCESS.value, message="Token count completed.", data=TokenCountData(**result_dict))

            elif name == "cancel-request":
                result_dict = await engine.cancel_request(
                    request_id=arguments.get("request_id"),
                    cancel_ops=arguments.get("cancel_ops"),
                    cancel_for_adapter_name=arguments.get("cancel_for_adapter_name")
                )
                return _create_response(APIStatus.SUCCESS.value, message=result_dict.get("message", "Cancel operation completed."), data=result_dict.get("data"))

            elif name == "load-adapter":
                config = AdapterConfig(**arguments)
                result = await engine.load_adapter(config)
                return _create_response(APIStatus.SUCCESS.value, message=f"Adapter '{result.get('adapter_name')}' loaded.", data=result)

            elif name == "unload-adapter":
                adapter_name = arguments.get("adapter_name")
                if not adapter_name: return _create_response(status=APIStatus.ERROR.value, message="Missing 'adapter_name' argument.")
                result = await engine.unload_adapter(adapter_name)
                return _create_response(APIStatus.SUCCESS.value, message="Unload adapter operation completed.", data=result.get("data"))

            elif name == "set-active-adapter":
                adapter_name_arg = arguments.get("adapter_name")
                if adapter_name_arg is None: return _create_response(status=APIStatus.ERROR.value, message="Missing 'adapter_name' argument.")
                result = await engine.set_active_adapter(adapter_name_arg)
                return _create_response(APIStatus.SUCCESS.value, message=f"Set active adapter operation completed.", data=result)

            elif name == "get-loaded-adapters":
                adapters_list = await engine.get_loaded_adapters()
                return _create_response(status=APIStatus.SUCCESS.value, message="Loaded adapters retrieved.", data=LoadedAdaptersData(adapters=adapters_list))

            elif name == "get-adapter-details":
                adapter_name = arguments.get("adapter_name")
                if not adapter_name: return _create_response(status=APIStatus.ERROR.value, message="Missing 'adapter_name' argument.")
                details = await engine.get_adapter_details(adapter_name)
                return _create_response(status=APIStatus.SUCCESS.value, message="Adapter details retrieved.", data=AdapterDetailsData(**details))

            elif name == "list-all-adapters":
                root_folder = arguments.get("root_folder")
                if not root_folder: return _create_response(status=APIStatus.ERROR.value, message="Missing 'root_folder' argument.")
                result = await engine.list_all_adapters(root_folder, arguments.get("include_incompatible", False), arguments.get("probe"))
                return _create_response(status=APIStatus.SUCCESS.value, message="Adapter discovery completed.", data={"adapters": result})

            elif name == "get-aggregate-metrics":
                metrics = await engine.get_aggregate_metrics()
                return _create_response(status=APIStatus.SUCCESS.value, message="Engine metrics retrieved.", data=metrics)

            elif name == "get-tools":
                 tools = [
                    "initialize-engine", "shutdown-engine", "list-engines", "set-default-engine",
                    "check-set-mode", "cancel-request", "start-training", "stop-training",
                    "get-training-status", "run-inference", "get-inference-status", "format-inference-prompt", 
                    "count-tokens", "load-adapter", "unload-adapter", "set-active-adapter", "get-loaded-adapters",
                    "get-adapter-details", "list-all-adapters", "get-engine-status", "get-aggregate-metrics", "get-tools"
                ]
                 return _create_response(status=APIStatus.SUCCESS.value, message="Available tools listed.", data=ToolsListData(tools=tools))

            else:
                return _create_response(status=APIStatus.ERROR.value, message=f"Unknown tool name '{name}'.")

        except (ValidationError, ConfigurationError, AdapterError, TrainingError, BusyError, EngineError, ModeMismatchError) as e:
            message, details = _extract_exception_message_and_details(e)
            logger.error(f"[API][{name}] Handled error on engine '{engine.instance_id}': {message}")
            return _create_response(status=APIStatus.ERROR.value, message=message, details=details)
        except Exception as e:
            full_tb = traceback.format_exc()
            logger.critical(f"[API][{name}] Unexpected error on engine '{engine.instance_id}': {type(e).__name__}: {e}\n{full_tb}")
            return _create_response(status=APIStatus.ERROR.value, message=f"Unexpected engine error: {type(e).__name__}", details={"full_traceback": full_tb})