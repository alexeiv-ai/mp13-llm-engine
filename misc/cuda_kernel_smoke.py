#!/usr/bin/env python
"""Minimal CUDA smoke test for torch runtime/kernel compatibility."""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal torch CUDA kernel smoke test.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0).")
    args = parser.parse_args()

    print(f"torch: {torch.__version__} cuda: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("cuda available: False")
        return 1

    if args.device < 0 or args.device >= torch.cuda.device_count():
        print(f"invalid device index: {args.device} (count={torch.cuda.device_count()})")
        return 1

    device = torch.device(f"cuda:{args.device}")
    name = torch.cuda.get_device_name(args.device)
    cap = torch.cuda.get_device_capability(args.device)
    arch_list = getattr(torch.cuda, "get_arch_list", lambda: "n/a")()

    print(f"device: {name}")
    print(f"capability: {cap}")
    print(f"arch list: {arch_list}")

    try:
        x = torch.arange(1024, device=device, dtype=torch.int64)
        print(f"min ok: {x.min().item()}")
    except Exception as exc:
        print(f"kernel test failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
