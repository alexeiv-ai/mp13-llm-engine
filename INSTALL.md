# Install

This repository uses **Poetry** and supports **two pinned dependency stacks** for local LLM inference and LoRA training.

It also supports optional, platform-dependent performance groups (Triton, FlashAttention 2) and a targeted compatibility group for one specific model family.

---

## Prerequisites

### Required
- **Python** (match the repo’s supported version; commonly 3.12)
- **Poetry** installed and available on PATH
- A working C/C++ build toolchain can be helpful for optional packages on some platforms

### GPU (optional)
If you plan to run on NVIDIA GPUs:
- NVIDIA driver installed (`nvidia-smi` should work)
- A torch CUDA build compatible with your driver/runtime
- CUDA Toolkit is **not** required unless you are building GPU extensions from source

> CPU-only is supported as a reference implementation and for broad compatibility.

---

## Supported dependency stacks

1) **Mainline (recommended)** — PyTorch **2.9.1 + CUDA 12.6** and Transformers **v5** (default `pyproject.toml`).  
   Best for: newest HF codepaths and latest kernels.

2) **Legacy / compatibility** — PyTorch **2.6.0 + CUDA 12.4** and Transformers **4.53.1** (use `pyproject.toml_torch_2.6.0`).  
   Best for: older model repos that break on Transformers v5, or workflows relying on v4-era HQQ behavior.

### Selecting a stack

If you want the default stack, follow the rest of this doc.

If you need the legacy stack, temporarily replace `pyproject.toml`:

```bash
cp pyproject.toml_torch_2.6.0 pyproject.toml
poetry lock --no-update
poetry install
```

---

## Choose your install profile

### A) Minimal (recommended default)

Works on CPU or GPU and avoids GPU-only kernels:

```bash
poetry sync
```

What you get:
- Transformers + baseline runtime
- Attention backend: **SDPA** (no Triton/FlashAttention required)

---

### B) Triton-only (kernel support)

```bash
poetry sync --with triton
```

When to use:
- You need Triton-backed kernels for certain model families/features
- You experiment with HQQ or other kernel paths

---

### C) FlashAttention 2 (optional acceleration)

```bash
poetry sync --with flashattn
```

What you get:
- FlashAttention 2 (where supported) plus the right Triton package

Windows note:
- This repo pins a community wheel URL for a specific Torch/CUDA combo.
- If it doesn’t work on your machine, skip it and use SDPA.

Linux note:
- On Linux, `flash-attn` is typically installed from PyPI.

---

### D) Mistral 3 support (targeted)

> The `mistral3` extra is **only needed for `Ministral-3-3B-Instruct-2512`**.

```bash
poetry sync --with mistral3
```

This is a deliberate deviation from a “single unified dependency set”: it is kept isolated so most users can stay on the minimal profile.

---

### E) xFormers (optional)

xFormers can provide alternative attention kernels on some platforms:

```bash
poetry run pip install -U xformers
```

Binary availability is sensitive to your exact torch/CUDA build. If it fails, use SDPA (default).

---

## CPU-only machines

Recommended:
```bash
poetry sync
```

Avoid:
- `--with flashattn`
- `--with triton`
- xFormers (usually not beneficial on CPU)

If you want a true CPU-only torch build, you’ll need to override the torch selection in your environment / Poetry sources.

---

## Appendix: Building flash-attn on Windows (source build)

Use this only if you cannot use a prebuilt wheel.

Prereqs:
- NVIDIA driver installed and a working CUDA runtime
- CUDA Toolkit matching your torch CUDA (e.g., 12.4 for `+cu124`)
- Visual Studio 2022 Build Tools (C++ + Windows SDK)
- `ninja`

Then inside the Poetry venv:

```bash
poetry run python -m pip install --upgrade pip setuptools wheel
poetry run python -m pip install ninja packaging
poetry run python -m pip install --no-build-isolation --no-deps flash-attn==2.8.0.post2
```
