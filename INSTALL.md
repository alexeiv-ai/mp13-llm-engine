# Install

This repository uses **Poetry** and supports **three pinned dependency stacks** for local LLM inference and LoRA training.

It also supports optional, platform-dependent performance groups (Triton, FlashAttention 2) and a targeted compatibility group for one specific model family.

---

## Quick Start: No Poetry Required

**Just want to try the project without installing Poetry?**

```bash
# 1. Clone and create venv
git clone https://github.com/alexeiv-ai/mp13-llm-engine
cd mp13-llm-engine
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate.bat on Windows

# 2. Install dependencies directly with pip
pip install --upgrade pip
pip install -e .
# see CONFIGURE.md for full options set
python mp13config.py --init

# 3. Run the demo or chat (you will still be asked for the model path/name unless read CONFIGURE.md)
python mp13chat.py
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
- **Poetry** installed and available on PATH for `poetry sync` command)

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

### Important: Cross-Platform Installation and the `poetry.lock` File

The `poetry.lock` file included in this repository is pre-built for a specific environment to ensure reproducibility for the current/default stack.

**The checked-in `poetry.lock` file is for:**
- **OS:** Windows
- **Architecture:** x86_64
- **CUDA:** 12.6 (for PyTorch 2.9.1+cu126)
- **GPU Family:** NVIDIA Ada Lovelace series and compatible predecessors

**If your environment is different (except for GPU)** (e.g., you are on Linux, macOS, Linux aarch64 or DGX Spark) you **must delete the existing `poetry.lock` file** before installing dependencies. This will allow Poetry to resolve and lock the correct binary packages for your specific platform.

To do this, run the following command **before** `poetry install` or `poetry sync`:

```bash
# For Linux / macOS
rm poetry.lock

# For Windows (Command Prompt)
del poetry.lock
```

After deleting the file, you can proceed with the installation commands, and Poetry will automatically generate a new, correct `poetry.lock` file for your system. On slower links or large solver updates, this can add up to about 10 minutes.

### Platform-Specific Configuration

#### Keyring on Headless Linux
On headless Linux systems, Poetry may hang while waiting for a graphical keyring service. Disable it with this one-time command:

```bash
poetry config keyring.enabled false
```

### GPU (optional)
If you plan to run on NVIDIA GPUs:
- NVIDIA driver installed (`nvidia-smi` should work)
- CUDA Toolkit is **not** required unless you are building GPU extensions from source

> CPU-only is supported as a reference implementation and for broad compatibility.


---

## Supported dependency stacks

1) **Current (recommended)** — PyTorch **2.9.1 + CUDA 12.6** and Transformers **v5** (`pyproject.toml_torch_2.9.1`, mirrored by default `pyproject.toml`).  
   Best for: newer HF codepaths and latest kernels.

2) **Legacy / compatibility** — PyTorch **2.6.0 + CUDA 12.4** and Transformers **4.53.1** (`pyproject.toml_torch_2.6.0`).  
   Best for: broadest compatibility with older model repos and v4-era behavior.


3) **Experimental GB10 / SPARC (sm_121)** — PyTorch **nightly cu130** (aarch64-compatible) and Transformers **v5** (`pyproject.toml_gb10`).  
   Best for very latest models, NVIDIA **GB10** / DGX Spark / other **sm_121** systems where stable cu12.6 wheels fail with `cudaErrorNoKernelImageForDevice`.

### Selecting a stack

If you want the default stack, follow the rest of this doc.

**Lock file note:** The checked-in `poetry.lock` corresponds to the default stack (default `pyproject.toml`, primary Windows environment). Alternate stacks require generating a **local** lock file; do not commit it and ingore in git pull.
One way to do that is to run once:
```bash
git update-index --skip-worktree poetry.lock 
```
To revert (start getting updated lock file from github):
```bash
git update-index --no-skip-worktree poetry.lock 
```

If you need the legacy stack, temporarily replace `pyproject.toml`:

```bash
cp pyproject.toml_torch_2.6.0 pyproject.toml
rm -f poetry.lock
poetry lock --no-update
poetry sync  # or: poetry install (Poetry 1.8-2.0)
```

If you need the GB10 / SPARC stack (experimental sm_121), temporarily replace `pyproject.toml`:

```bash
cp pyproject.toml_gb10 pyproject.toml
rm -f poetry.lock
poetry lock --no-update
poetry sync  # or: poetry install (Poetry 1.8-2.0)
```

### Model compatibility notes by stack

- Most project testing was done with the **legacy** .toml_torch_2.6.0 stack.
- Newer stacks can unlock newer models, but some older models may stop working.
- If an older model fails on newer stacks, try setting `trust_remote_code=false` in the engine default or custom config.
- The **experimental GB10** stack may show additional regressions with older models due to newer torch/runtime paths.
- On Hopper GPUs, the experimental stack allows opportunistically install and enable `flash_attention_3` adding `--with flashattn3` install/sync option; this path is not yet tested.

### Keep one repeatable install command per target box

Once you settle on a specific machine and install profile, write a small command script (`.cmd`/`.ps1`/`.sh`) that always runs the same stack file swap and Poetry flags. This helps you stay aligned with future `pyproject.toml*` updates without repeatedly forgetting fixed arguments or unnecessarily re-creating lock files from scratch.

---

## Choose your install profile

**Poetry version note:**
- Poetry 2.1+: Use `poetry sync` (faster, more accurate)
- Poetry 1.8-2.0: Use `poetry install` (works on all versions)

### 1) Minimal setup (works for most models)

```bash
# Poetry 2.1+
poetry sync

# Poetry 1.8-2.0
poetry install
```

Optional: verify CUDA kernel compatibility (recommended on new GPU/driver stacks):

```bash
# from activated venv
python misc/cuda_kernel_smoke.py

# or without activating venv
poetry run python misc/cuda_kernel_smoke.py
```

---

### 2) Complete setup (enable all optional groups)

```bash
# Poetry 2.1+
poetry sync --with triton --with flashattn --with mistral3

# Poetry 1.8-2.0
poetry install --with triton --with flashattn --with mistral3
```

**Always pass all desired** `--with` groups in  one poetry command.

Optional add-on:

- With experimental setup (`pyproject.toml_gb10`) and on  Hopper GPU, `--with flashattn3` might work (wheel is not pinned).
- `mistral3` is only required for `Ministral-3-3B-Instruct-2512`.

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

---

## Troubleshooting

### "Cannot be installed in the current environment" Error

**Symptom:** Installation fails with an error about a package being for the wrong platform, often mentioning a `.whl` file (e.g., `...win_amd64.whl` on Linux).

**Cause & Solution:** This happens when the `poetry.lock` file is for a different operating system or architecture. Please see the **"Important: Cross-Platform Installation"** section near the top of this document for the solution.

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

### Flash-attn issues

**Symptom:** flash-attn fais at installation or runtime.

**Solution:** Skip the flashattn group:
```bash
# Install without flash-attn
poetry install --with triton --with mistral3  # omit --with flashattn
```

The default SDPA attention backend should work well and may automatically pick internal flash_attention_2 impl.

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

<a id="poetry-in-venv-disappears"></a>
### Poetry installed inside venv disappears

**Symptom:** You installed Poetry inside the project's `.venv`, and after running `poetry install`, Poetry is no longer available.

**Why this happens (technical explanation):**

When you run `poetry install`, Poetry does the following:
1. Reads `pyproject.toml` to see what dependencies your project needs
2. Reads `poetry.lock` to see the exact versions to install
3. **Synchronizes** the venv to match exactly what's in the lock file
4. Poetry itself is NOT listed in `pyproject.toml` as a dependency (it shouldn't be - it's a build tool, not a runtime dependency)
5. During sync, Poetry may remove packages not in the lock file - including itself if it was installed in the venv

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
