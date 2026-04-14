import time
import argparse
from concurrent.futures import ThreadPoolExecutor

def compute_task(n):
    """
    Performs a computationally intensive task.
    This is used to simulate a heavy CPU workload.
    """
    total = 0
    for i in range(n):
        total += i * i
    return total

def measure_single_threaded_performance(n):
    """Measures the time taken to run the compute task on a single thread."""
    start_time = time.time()
    result = compute_task(n)
    end_time = time.time()
    return end_time - start_time, result

def measure_multi_threaded_performance(n, threads=10):
    """Measures the time taken to run the compute task concurrently across multiple threads."""
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Distribute the workload evenly across the specified number of threads
        futures = [executor.submit(compute_task, n // threads) for _ in range(threads)]
        results = [future.result() for future in futures]
    end_time = time.time()
    total_result = sum(results)
    return end_time - start_time, total_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measures single-threaded vs multi-threaded CPU performance using a simple math task.")
    parser.add_argument("--iterations", type=int, default=10**7, help="Number of iterations for the math task. Higher means heavier CPU load. (default: 10000000)")
    parser.add_argument("--threads", type=int, default=10, help="Number of threads to use for multi-threaded test (default: 10)")
    args = parser.parse_args()

    print(f"Running CPU Performance Test with {args.iterations} iterations.")

    # Measure single-threaded performance
    single_threaded_time, single_result = measure_single_threaded_performance(args.iterations)
    print(f"\nSingle-threaded performance:")
    print(f"Time: {single_threaded_time:.2f} seconds, Result: {single_result}")

    # Measure multi-threaded performance
    multi_threaded_time, multi_result = measure_multi_threaded_performance(args.iterations, args.threads)
    print(f"\nMulti-threaded performance ({args.threads} threads):")
    print(f"Time: {multi_threaded_time:.2f} seconds, Result: {multi_result}")
