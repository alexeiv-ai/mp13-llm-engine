import torch
import time
import gc
import sys
import platform
import argparse

def get_gpu_name(device_id):
    """Gets the name of the specified GPU."""
    try:
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            return torch.cuda.get_device_name(device_id)
        else:
            return "N/A (CUDA not available or invalid device ID)"
    except Exception as e:
        return f"Error getting GPU name: {e}"

def run_transfer_test(device_id, tensor_size_mb, num_iterations):
    """
    Runs a memory transfer test from CPU to the specified GPU.
    This helps measure the latency and bandwidth of the PCIe/NVLink connection.
    """
    if not torch.cuda.is_available():
        print("!!! CUDA is not available in this environment. Skipping test. !!!")
        return None

    if device_id >= torch.cuda.device_count():
        print(f"!!! Error: Device ID {device_id} is invalid. Found {torch.cuda.device_count()} GPUs. !!!")
        return None

    target_device = f'cuda:{device_id}'
    tensor_size_bytes = tensor_size_mb * 1024 * 1024
    # Use float32 for simplicity, size is the main factor here
    num_elements = tensor_size_bytes // 4
    total_data_gb = (tensor_size_bytes * num_iterations) / (1024**3)

    print(f"\n--- Starting CPU to GPU Transfer Test ---")
    print(f"Environment: {platform.system()} {platform.release()} ({'WSL' if 'microsoft' in platform.release().lower() else 'Native Host'})")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Target Device: {target_device} ({get_gpu_name(device_id)})")
    print(f"Tensor Size per Transfer: {tensor_size_mb} MB")
    print(f"Number of Iterations: {num_iterations}")
    print(f"Total Data Transferred: {total_data_gb:.3f} GB")
    print("-----------------------------")

    timings = []
    try:
        for i in range(num_iterations):
            print(f"Running iteration {i + 1}/{num_iterations}...")
            # 1. Create tensor on CPU
            tensor_cpu = torch.randn(num_elements, dtype=torch.float32, device='cpu')

            # 2. Synchronize before timing (ensure previous CUDA work is done)
            torch.cuda.synchronize(target_device)
            gc.collect() # Try to ensure memory is available

            # 3. Time the transfer
            start_time = time.perf_counter()
            tensor_gpu = tensor_cpu.to(target_device, non_blocking=False) # Use blocking transfer
            torch.cuda.synchronize(target_device) # Wait for the transfer to complete on GPU
            end_time = time.perf_counter()

            duration = end_time - start_time
            timings.append(duration)
            print(f"  Iteration {i + 1} time: {duration:.4f} seconds")

            # 4. Clean up immediately to avoid accumulating memory usage
            del tensor_cpu
            del tensor_gpu
            gc.collect()
            if torch.cuda.is_available():
                 torch.cuda.empty_cache()
            time.sleep(0.1) # Small pause

    except Exception as e:
        print(f"\n!!! ERROR during testing: {e} !!!")
        return None

    if not timings:
        print("!!! No successful iterations completed. !!!")
        return None

    # --- Calculate and Print Results ---
    avg_time = sum(timings) / len(timings)
    min_time = min(timings)
    max_time = max(timings)
    # Bandwidth = Bytes / Time
    avg_bandwidth_mbs = tensor_size_bytes / avg_time / (1024**2)
    max_bandwidth_mbs = tensor_size_bytes / min_time / (1024**2) # Max BW corresponds to Min Time
    min_bandwidth_mbs = tensor_size_bytes / max_time / (1024**2) # Min BW corresponds to Max Time

    print("\n--- Test Results ---")
    print(f"Average Transfer Time: {avg_time:.4f} seconds")
    print(f"Minimum Transfer Time: {min_time:.4f} seconds")
    print(f"Maximum Transfer Time: {max_time:.4f} seconds")
    print(f"Average Bandwidth:     {avg_bandwidth_mbs:.2f} MB/s")
    print(f"Peak Bandwidth (Max):  {max_bandwidth_mbs:.2f} MB/s")
    print(f"Minimum Bandwidth:     {min_bandwidth_mbs:.2f} MB/s")
    print("--------------------\n")

    return avg_bandwidth_mbs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measures latency and bandwidth for transferring tensors from CPU to GPU.")
    parser.add_argument("--device_id", type=int, default=0, help="Target GPU device ID (default: 0)")
    parser.add_argument("--size_mb", type=int, default=512, help="Size of the tensor to transfer in MB (default: 512)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of transfer iterations to average (default: 10)")
    args = parser.parse_args()

    run_transfer_test(args.device_id, args.size_mb, args.iterations)