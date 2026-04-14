import torch
import triton
import triton.language as tl
import argparse
import sys

@triton.jit
def add_kernel(X_ptr, Y_ptr, Z_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offs, mask=mask, other=0.0)
    tl.store(Z_ptr + offs, x + y, mask=mask)

def run_test(size, device_id):
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Cannot test Triton.")
        return 1
        
    device = f"cuda:{device_id}"
    print(f"Testing Triton JIT compilation on {device} with array size {size}...")
    
    try:
        x = torch.randn(size, device=device, dtype=torch.float16)
        y = torch.randn(size, device=device, dtype=torch.float16)
        z = torch.empty_like(x)
        
        # Calculate grid
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK']),)
        
        # Launch kernel
        add_kernel[grid](x, y, z, n=size, BLOCK=256)
        torch.cuda.synchronize(device)
        
        # Verify
        max_diff = (z - (x + y)).abs().max().item()
        print(f"OK. Maximum difference: {max_diff:.6f}")
        return 0 if max_diff < 1e-4 else 1
    except Exception as e:
        print(f"Error during Triton kernel execution: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description=(
            "OpenAI Triton compiler sanity check.\n\n"
            "Purpose:\n"
            "  This script verifies that the OpenAI Triton compiler is installed correctly\n"
            "  and can successfully JIT compile and execute a custom CUDA kernel on your GPU.\n"
            "  It performs a basic vector addition and verifies the result against PyTorch.\n"
            "  If this fails, you may have incompatible CUDA toolkits, missing headers, or\n"
            "  unsupported driver versions.\n\n"
            "Usage:\n"
            "  python triton_check.py             # Run default test on cuda:0\n"
            "  python triton_check.py --size 4096 # Test with a specific vector size\n"
            "  python triton_check.py --device 1  # Test on a specific GPU"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=1024, help="Size of the 1D tensor to process (default: 1024).")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index to test on (default: 0).")
    
    args = parser.parse_args()
    
    sys.exit(run_test(args.size, args.device))

if __name__ == "__main__":
    main()
