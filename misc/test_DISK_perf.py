import os
import time
import argparse
import tempfile
import asyncio
import aiofiles

async def async_read_file_chunk(file_path, start, size):
    """Asynchronously reads a chunk of a file."""
    async with aiofiles.open(file_path, 'rb') as f:
        await f.seek(start)
        return await f.read(size)

async def async_write_file_chunk(file_path, start, data):
    """Asynchronously writes a chunk to a file."""
    async with aiofiles.open(file_path, 'r+b') as f:
        await f.seek(start)
        await f.write(data)

def measure_sync_performance(file_path, task_func, input_data, file_size, chunk_size, is_write=False):
    """Measures performance using synchronous, single-threaded IO."""
    results = []
    for start in range(0, file_size, chunk_size):
        start_time = time.time()
        if is_write:
            task_func(file_path, start, input_data[start:start + min(chunk_size, file_size - start)])
        else:
            task_func(file_path, start, min(chunk_size, file_size - start))
        end_time = time.time()
        results.append(end_time - start_time)
    return sum(results)

async def measure_async_performance(file_path, task_func, input_data, file_size, chunk_size, is_write=False):
    """Measures performance using asynchronous concurrent IO operations."""
    tasks = []
    start_time = time.time()
    
    for start in range(0, file_size, chunk_size):
        if is_write:
            data_slice = input_data[start:start + min(chunk_size, file_size - start)]
            task = asyncio.create_task(task_func(file_path, start, data_slice))
        else:
            task = asyncio.create_task(task_func(file_path, start, min(chunk_size, file_size - start)))
        tasks.append(task)
    
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    return end_time - start_time

async def main():
    parser = argparse.ArgumentParser(description='Measures and compares single-threaded vs asynchronous (multi-task) file read and write performance on disk.')
    parser.add_argument('--file', type=str, required=True, help='Path to the test file to read/write from. Use a reasonably large file for better metrics.')
    parser.add_argument('--percent', type=int, default=10, help='Percent of the file to process in each I/O chunk (default: 10)')
    
    args = parser.parse_args()

    print(f"Test Target File: {args.file}")
    print(f"Percent of file to process per chunk: {args.percent}%")

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"The specified file does not exist: {args.file}")

    # Read full input data once to keep in memory for write tests
    with open(args.file, 'rb') as f:
        input_data = f.read()

    file_size = os.path.getsize(args.file)
    chunk_size = int(file_size * (args.percent / 100))
    # Ensure chunk size is at least 1 byte if the file is very small or percent is tiny
    if chunk_size == 0:
        chunk_size = 1
    num_chunks = (file_size + chunk_size - 1) // chunk_size

    print(f"File Size: {file_size / (1024*1024):.2f} MB")
    print(f"Chunk Size: {chunk_size / (1024*1024):.2f} MB ({num_chunks} chunks)")

    # Read Tests
    single_read_time = measure_sync_performance(args.file, read_file_chunk, None, file_size, chunk_size)
    print(f"\nRead Performance (Single-threaded): {single_read_time:.4f} seconds")

    multi_read_time = await measure_async_performance(args.file, async_read_file_chunk, None, file_size, chunk_size)
    print(f"Read Performance (Async I/O, {num_chunks} tasks): {multi_read_time:.4f} seconds")

    # Write Tests Setup
    temp_dir = os.path.dirname(os.path.abspath(args.file))
    with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(input_data)

    try:
        # Sync write
        single_write_time = measure_sync_performance(temp_file_path, write_file_chunk, input_data, file_size, chunk_size, is_write=True)
        print(f"\nWrite Performance (Single-threaded): {single_write_time:.4f} seconds")

        # Async write
        multi_write_time = await measure_async_performance(temp_file_path, async_write_file_chunk, input_data, file_size, chunk_size, is_write=True)
        print(f"Write Performance (Async I/O, {num_chunks} tasks): {multi_write_time:.4f} seconds")
    finally:
        # Ensure the file is closed and cleaned up by all operations
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def read_file_chunk(file_path, start, size):
    """Synchronously reads a chunk of a file."""
    with open(file_path, 'rb') as f:
        f.seek(start)
        return f.read(size)

def write_file_chunk(file_path, start, data):
    """Synchronously writes a chunk to a file."""
    with open(file_path, 'r+b') as f:
        f.seek(start)
        f.write(data)

if __name__ == "__main__":
    asyncio.run(main())
