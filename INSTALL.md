# Install

This repository uses **Poetry** and supports **two pinned dependency stacks** for local LLM inference and LoRA training.

It also supports optional, platform-dependent performance groups (Triton, FlashAttention 2) and a targeted compatibility group for one specific model family.

---

## Quick Start: No Poetry Required

**Just want to try the project without installing Poetry?**

```bash
# 1. Clone and create venv
git clone https://github.com/alexeiv-ai/mp13-llm-engine
cd mp13-llm-engine
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install dependencies directly with pip
pip install --upgrade pip
pip install -e .

# 3. Run the demo or chat
python -m src.app.mp13chat
```

**Limitations of this approach:**
- Doesn't lock to exact versions from `poetry.lock` (may get newer package versions)
- Optional groups (triton, flashattn, mistral3) need manual installation
- No Poetry-specific features (better dependency resolution, reproducible builds)

**For a production-ready installation with locked dependencies, follow the full Poetry-based installation below.**

---

## Prerequisites

### Required
- **Python** (match the repo's supported version; commonly 3.12)
- **Poetry** installed and available on PATH (version 1.8+ recommended, 2.1+ for `poetry sync` command)
- A working C/C++ build toolchain can be helpful for optional packages on some platforms

### Installing Poetry

**Minimum version:** 1.8
**Recommended version:** 2.1+ (for `poetry sync` command and better dependency resolution)

Pick one of the official methods:

```bash
# Windows (PowerShell)
py -m pip install --user "poetry>=2.1"

# macOS / Linux
python3 -m pip install --user "poetry>=2.1"
```

**Upgrading existing Poetry installation:**
```bash
# If installed via pip
python3 -m pip install --user --upgrade "poetry>=2.1"

# If installed via system package manager (not recommended for latest version)
# Better to use pip install instead
```

Verify:

```bash
poetry --version
```

**Note:** If you're stuck with Poetry 1.8.x (e.g., restricted environment), the project will work but use `poetry install` instead of `poetry sync`.

### Linux-specific setup

#### Keyring configuration (headless systems)
On headless Linux systems, Poetry may wait for keyring authentication. Disable it:

```bash
poetry config keyring.enabled false
```

#### Platform-specific lock files
The `poetry.lock` file in this repository may have been generated on a different platform (Windows/Linux/x86_64/ARM64). If you encounter installation errors about platform-incompatible packages, regenerate the lock file on your system:

```bash
# Remove the existing lock file
rm poetry.lock

# Generate a new lock file for your platform
poetry lock --no-update

# Then proceed with installation (see below)
```

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

**Poetry version note:**
- Poetry 2.1+: Use `poetry sync` (faster, more accurate)
- Poetry 1.8-2.0: Use `poetry install` (works on all versions)

Both commands are shown below. Use the one compatible with your Poetry version.

### A) Minimal (recommended default)

Works on CPU or GPU and avoids GPU-only kernels:

```bash
# Poetry 2.1+
poetry sync

# Poetry 1.8-2.0
poetry install
```

What you get:
- Transformers + baseline runtime
- Attention backend: **SDPA** (no Triton/FlashAttention required)

---

### B) Triton-only (kernel support)

```bash
# Poetry 2.1+
poetry sync --with triton

# Poetry 1.8-2.0
poetry install --with triton
```

When to use:
- You need Triton-backed kernels for certain model families/features
- You experiment with HQQ or other kernel paths

---

### C) FlashAttention 2 (optional acceleration)

```bash
# Poetry 2.1+
poetry sync --with flashattn

# Poetry 1.8-2.0
poetry install --with flashattn
```

What you get:
- FlashAttention 2 (where supported) plus the right Triton package

**Windows note:**
- This repo pins a community wheel URL for a specific Torch/CUDA combo.
- If it doesn't work on your machine, skip it and use SDPA.

**Linux x86_64 note:**
- On Linux x86_64, `flash-attn` is typically installed from PyPI.

**Linux ARM64 (aarch64) note:**
- FlashAttention 2 pre-built wheels are not available for ARM64.
- Skip this optional group and use SDPA (default attention backend).
- Advanced users can build from source (see Appendix below).

---

### D) Mistral 3 support (targeted)

> The `mistral3` extra is **only needed for `Ministral-3-3B-Instruct-2512`**.

```bash
# Poetry 2.1+
poetry sync --with mistral3

# Poetry 1.8-2.0
poetry install --with mistral3
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
# Poetry 2.1+
poetry sync

# Poetry 1.8-2.0
poetry install
```

Avoid:
- `--with flashattn`
- `--with triton`
- xFormers (usually not beneficial on CPU)

If you want a true CPU-only torch build, you'll need to override the torch selection in your environment / Poetry sources.

---

## Troubleshooting

### Lock file platform mismatch

**Symptom:** Installation fails with errors like:
```
RuntimeError: Package https://...flash_attn...win_amd64.whl cannot be installed in the current environment
```

**Cause:** The lock file was generated on a different platform (e.g., Windows lock file used on Linux).

**Solution:**
```bash
# Remove the existing lock file
rm poetry.lock

# Regenerate for your platform
poetry lock --no-update

# Install
poetry install  # or poetry sync on Poetry 2.1+
```

### Poetry hangs on headless Linux

**Symptom:** Poetry appears to hang during installation.

**Cause:** Poetry is waiting for keyring authentication.

**Solution:**
```bash
poetry config keyring.enabled false
```

### Old lock file warning

**Symptom:** Poetry warns "The lock file is not up to date with the latest changes in pyproject.toml."

**Solution:**
```bash
poetry lock --no-update
```

### ARM64 Linux (aarch64) flash-attn issues

**Symptom:** flash-attn installation fails on ARM64 systems.

**Solution:** Skip the flashattn group:
```bash
# Install without flash-attn
poetry install --with triton --with mistral3  # omit --with flashattn
```

The default SDPA attention backend works well on ARM64 systems.

### Using --no-root flag

If you encounter issues installing the project itself (e.g., missing README), you can install dependencies without installing the project package:

```bash
poetry install --no-root --with triton --with flashattn --with mistral3
```

**Note:** This means the `mp13chat` and `mp13config` commands won't be available. Run them via:
```bash
python -m src.app.mp13chat
python -m src.app.config
```

### Poetry installed inside venv disappears

**Symptom:** You installed Poetry inside the project's `.venv`, and after running `poetry install`, Poetry is no longer available.

**Why this happens (technical explanation):**

When you run `poetry install`, Poetry does the following:
1. Reads `pyproject.toml` to see what dependencies your project needs
2. Reads `poetry.lock` to see the exact versions to install
3. **Synchronizes** the venv to match exactly what's in the lock file
4. Poetry itself is NOT listed in `pyproject.toml` as a dependency (it shouldn't be - it's a build tool, not a runtime dependency)
5. During sync, Poetry may remove packages not in the lock file - including itself if it was installed in the venv

Think of it like this: **Poetry is the construction crew, not the building**. The crew builds the house (your venv) but doesn't live inside it.

**Solution 1: User-wide Poetry (Recommended for trying projects)**

Install Poetry once in your user directory - it works for all projects:

```bash
# 1. Deactivate any active venv
deactivate

# 2. Install Poetry user-wide (OUTSIDE any venv)
python3 -m pip install --user "poetry>=2.1"

# 3. Ensure it's on PATH (add to ~/.bashrc if needed)
export PATH="$HOME/.local/bin:$PATH"

# 4. Verify
poetry --version

# 5. Use it for any project
cd /path/to/any-project
poetry install  # Works for all projects!
```

**Benefits:**
- ✅ Install once, use for all Poetry projects
- ✅ No global admin privileges needed
- ✅ Isolated from system Python
- ✅ Easy to upgrade: `python3 -m pip install --user --upgrade poetry`

**Solution 2: Use pip directly (No Poetry needed for trying projects)**

If you really don't want to install Poetry, you can install dependencies directly:

```bash
# 1. Create a venv manually
python3 -m venv .venv
source .venv/bin/activate

# 2. Install from the lock file using pip
# First, install poetry-core (just the library, not the CLI)
pip install poetry-core

# 3. Install the project in editable mode
pip install -e .

# 4. Manually install optional groups if needed
pip install triton  # For --with triton
# etc.
```

**Trade-offs:**
- ❌ Doesn't respect lock file exactly (may get newer versions)
- ❌ Optional groups need manual installation
- ❌ Misses Poetry's dependency resolution
- ✅ No Poetry installation needed
- ✅ Works with familiar pip workflow

**Solution 3: pipx (Best for system-wide Poetry)**

If you manage multiple Python tools, use `pipx`:

```bash
# Install pipx (if not already installed)
python3 -m pip install --user pipx
python3 -m pipx ensurepath

# Install Poetry via pipx (isolated from all Python environments)
pipx install poetry

# Verify
poetry --version
```

**Benefits:**
- ✅ Poetry completely isolated in its own environment
- ✅ Never conflicts with any project
- ✅ Easy to upgrade: `pipx upgrade poetry`
- ✅ Works system-wide

**Key principle:** Poetry is a **build/dependency management tool**, not a runtime dependency:
- ✅ User-wide: `pip install --user` (recommended for trying projects)
- ✅ System-wide via pipx: `pipx install poetry` (recommended for regular use)
- ✅ System-wide: requires admin (not recommended)
- ❌ Never inside a project's venv (will cause issues)

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
