import torch
import transformers
import time
import gc
import argparse
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def run_model_vector_transfer_test(model_path, dtype_str, target_device_id):
    """
    Tests an alternative model loading strategy:
    1. Loads the model structure and weights directly to CPU RAM.
    2. Flattens all the model's parameters into a single contiguous vector on CPU.
    3. Transfers this single massive vector to the GPU (often faster than moving many small tensors).
    4. Initializes an empty model structure on the GPU.
    5. Copies the flattened vector data back into the GPU model's parameters.
    """
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    compute_dtype = dtype_map[dtype_str]
    target_device = f'cuda:{target_device_id}'

    # --- Start Timing ---
    overall_start_time = time.time()

    # --- Step 1: Load Model Structure and Weights to CPU ---
    print(f"[{time.time() - overall_start_time:.2f}s] Loading model to CPU from: {model_path}")
    start_time = time.time()
    model_cpu = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=compute_dtype,
        device_map="cpu", # Ensure it loads to CPU
        low_cpu_mem_usage=True # Recommended for large models even on CPU
    )
    print(f"[{time.time() - overall_start_time:.2f}s] CPU model loaded in {time.time() - start_time:.2f}s")

    # --- Step 2: Flatten Parameters (CPU) ---
    print(f"[{time.time() - overall_start_time:.2f}s] Flattening parameters on CPU...")
    start_time = time.time()
    params_vector_cpu = parameters_to_vector(model_cpu.parameters())
    print(f"[{time.time() - overall_start_time:.2f}s] Parameters flattened in {time.time() - start_time:.2f}s")
    print(f"[{time.time() - overall_start_time:.2f}s] Flat parameter vector size: {params_vector_cpu.numel()} elements, {params_vector_cpu.nelement() * params_vector_cpu.element_size() / 1024**3:.3f} GB")

    # --- Step 3: Transfer Flat Vector (CPU -> GPU) ---
    print(f"[{time.time() - overall_start_time:.2f}s] Transferring flat vector CPU -> GPU ({target_device})...")
    start_time = time.time()
    if torch.cuda.is_available(): torch.cuda.synchronize(target_device_id) # Sync before transfer
    params_vector_gpu = params_vector_cpu.to(target_device, non_blocking=False) # Blocking transfer
    if torch.cuda.is_available(): torch.cuda.synchronize(target_device_id) # Sync after transfer
    print(f"[{time.time() - overall_start_time:.2f}s] Flat vector transferred in {time.time() - start_time:.2f}s")

    # --- Clean up CPU vector ---
    print(f"[{time.time() - overall_start_time:.2f}s] Cleaning up CPU parameter vector...")
    del params_vector_cpu
    gc.collect()

    # --- Step 4: Create Model Skeleton (GPU) ---
    print(f"[{time.time() - overall_start_time:.2f}s] Initializing empty model directly on GPU ({target_device}) using config...")
    start_time = time.time()
    config = transformers.AutoConfig.from_pretrained(model_path) # Load config first

    # Use a context manager and from_config to initialize directly on the target device
    model_gpu = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=compute_dtype)
    # Ensure it's on the correct device and dtype AFTER initialization
    model_gpu.to(device=target_device, dtype=compute_dtype)

    print(f"[{time.time() - overall_start_time:.2f}s] Empty model created on GPU in {time.time() - start_time:.2f}s")
    # Ensure garbage collection happens if large intermediate tensors were created
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Step 5: Distribute Vector to Parameters (GPU) ---
    print(f"[{time.time() - overall_start_time:.2f}s] Copying flat vector data into GPU model parameters...")
    start_time = time.time()
    # This copies data from params_vector_gpu into the buffers underlying model_gpu.parameters()
    vector_to_parameters(params_vector_gpu, model_gpu.parameters())
    print(f"[{time.time() - overall_start_time:.2f}s] Parameters populated from vector in {time.time() - start_time:.2f}s")


    # --- Cleanup flat GPU vector ---
    print(f"[{time.time() - overall_start_time:.2f}s] Cleaning up GPU parameter vector...")
    del params_vector_gpu
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


    # --- Verification ---
    print(f"[{time.time() - overall_start_time:.2f}s] Verifying model device...")
    final_device = next(model_gpu.parameters()).device
    print(f"Model is now on device: {final_device}")
    if final_device.type != 'cuda':
        print("!!! ERROR: Model is not on CUDA device after process !!!")

    overall_end_time = time.time()
    print(f"\nTotal time using vector transfer method: {overall_end_time - overall_start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tests an alternative model loading strategy involving flattening model parameters into a single vector for CPU->GPU transfer.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory (required)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Compute dtype to use (default: bfloat16)")
    parser.add_argument("--target_gpu", type=int, default=0, help="Target GPU index for transfer (default: 0)")
    
    args = parser.parse_args()
    
    run_model_vector_transfer_test(args.model_path, args.dtype, args.target_gpu)