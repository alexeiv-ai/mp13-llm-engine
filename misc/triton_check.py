import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(X_ptr, Y_ptr, Z_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offs, mask=mask, other=0.0)
    tl.store(Z_ptr + offs, x + y, mask=mask)

def test():
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    y = torch.randn(1024, device="cuda", dtype=torch.float16)
    z = torch.empty_like(x)
    add_kernel[(1024 // 256,)](x, y, z, n=1024, BLOCK=256)
    torch.cuda.synchronize()
    print("OK", (z - (x + y)).abs().max().item())

test()
