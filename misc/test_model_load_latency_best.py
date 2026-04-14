import torch
import transformers
import accelerate
import time
import logging
import os
import gc
import argparse

# --- Custom Logging Formatter for Seconds ---
class SecondsFormatter(logging.Formatter):
    """Custom formatter to show relative time in seconds."""
    start_time = time.time()

    def format(self, record):
        record.relativeSeconds = record.created - self.start_time
        return super().format(record)

# --- Logging Setup ---
log_formatter = SecondsFormatter(
    "[%(asctime)s.%(msecs)03d (%(relativeSeconds).3fs)] %(levelname)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

def log_environment_info(args):
    """Logs information about the environment."""
    logger.info("--- Environment Information ---")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"Transformers Version: {transformers.__version__}")
    logger.info(f"Accelerate Version: {accelerate.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: True")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"     Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    else:
        logger.info("CUDA Available: False")
    logger.info(f"Model Path: {args.model_path}")
    logger.info(f"Compute Dtype: {args.dtype}")
    
    index_found = False
    shard_files = []
    if os.path.isdir(args.model_path):
        for fname in os.listdir(args.model_path):
            if fname.endswith(".index.json"):
                index_found = True
            if fname.endswith(".bin") or fname.endswith(".safetensors"):
                 shard_files.append(fname)
    logger.info(f"Model Sharded: {'Yes (index found)' if index_found else 'No (index not found)'}")
    logger.info(f"Detected Shard Files ({len(shard_files)}): {shard_files[:5]} {'...' if len(shard_files)>5 else ''}")
    logger.info("-----------------------------")

def cleanup_model(model_var):
    """Safely cleans up a model and frees memory."""
    if model_var is not None:
        logger.info(f"Cleaning up model reference (ID: {id(model_var)})...")
        del model_var
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
    else:
        logger.info("No model reference to clean up.")

def time_model_load(test_name: str, device_map_config: any, args):
    """Times the model loading process with specified parameters."""
    logger.info(f"--- Starting Test: {test_name} ---")
    logger.info(f"Parameters: device_map='{device_map_config}'")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    start_time = time.time()
    try:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype_map[args.dtype],
            device_map=device_map_config,
            low_cpu_mem_usage=True,
        )
        
        # Ensure CUDA operations are finished before stopping the timer
        if torch.cuda.is_available() and device_map_config != "cpu":
            torch.cuda.synchronize()
            
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Model loading finished.")
        logger.info(f"SUCCESS: Test '{test_name}' completed in {duration:.3f} seconds.")
        return model, duration

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"FAILED: Test '{test_name}' encountered an error after {duration:.3f} seconds: {e}")
        return None, duration
    finally:
        logger.info(f"--- Finished Test: {test_name} ---")
        print("-" * 60)

def time_cpu_to_gpu_transfer(model, target_device):
    """Times the transfer of a model already in CPU RAM to a target GPU."""
    if model is None:
         logger.warning("Skipping CPU->GPU transfer test: Model object is None.")
         return None

    logger.info(f"--- Starting CPU -> {target_device} Transfer Test ---")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    time.sleep(1)

    start_time = time.time()
    try:
        model.to(target_device)
        
        # Ensure transfer is fully complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"SUCCESS: Model transfer to {target_device} completed in {duration:.3f} seconds.")
        return duration
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"FAILED: Model transfer to {target_device} encountered an error: {e}")
        return None
    finally:
        logger.info(f"--- Finished CPU -> {target_device} Transfer Test ---")
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model load latency with different strategies.")
    parser.add_argument("--model_path", type=str, default="test_models/granite-3.1-2b-instruct/", help="Path to the model directory.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Compute dtype to use (default: bfloat16).")
    parser.add_argument("--target_gpu", type=int, default=0, help="Target GPU index for direct load and transfer tests (default: 0).")
    
    args = parser.parse_args()
    
    log_environment_info(args)

    results = {}

    # Test 1: Direct Single GPU Map
    model_gpu, duration_gpu = time_model_load(
        f"Direct Single GPU Load (cuda:{args.target_gpu})",
        device_map_config={"": args.target_gpu},
        args=args
    )
    results[f"Direct Single GPU Load (cuda:{args.target_gpu})"] = duration_gpu
    cleanup_model(model_gpu)

    # Test 2: Auto Device Map
    model_auto, duration_auto = time_model_load(
        "Auto Device Map Load",
        device_map_config="auto",
        args=args
    )
    results["Auto Device Map Load"] = duration_auto
    cleanup_model(model_auto)

    # Test 3: Load to CPU first, then transfer to GPU
    model_cpu, duration_cpu = time_model_load(
        "Load to CPU First",
        device_map_config="cpu",
        args=args
    )
    results["Load to CPU First"] = duration_cpu

    if model_cpu:
        transfer_duration = time_cpu_to_gpu_transfer(model_cpu, f'cuda:{args.target_gpu}')
        results[f"CPU -> GPU (cuda:{args.target_gpu}) Transfer"] = transfer_duration
        if transfer_duration is not None and duration_cpu is not None:
            results["Total CPU-First Load + Transfer"] = duration_cpu + transfer_duration
        cleanup_model(model_cpu)

    # --- Summary ---
    logger.info("\n--- Comprehensive Test Summary ---")
    for test_name, duration_val in results.items():
        status = f"{duration_val:.3f} seconds" if duration_val is not None else "FAILED or SKIPPED"
        logger.info(f"{test_name}: {status}")
    logger.info("----------------------------------")
    logger.info("Test script finished.")
