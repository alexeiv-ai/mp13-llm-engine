# Copyright (c) 2025 mp13
# Author: alexeiv-ai <188820640+alexeiv-ai@users.noreply.github.com>
# AI-Assistance: Portions of this file were drafted using AI coding tools
# (e.g., ChatGPT, Gemini, Codex) under active human design supervision.
# Contact: Please open an issue or discussion on GitHub.
# SPDX-License-Identifier: Apache-2.0
"""
Centralized module for applying runtime patches (monkey-patching) to third-party libraries
like transformers, peft, etc. This helps to isolate modifications and makes them easier to
track and maintain.
"""
import sys
import importlib.abc
import importlib.machinery
import importlib
import types
import functools
import time
import inspect
import logging
from typing import Callable, List

import warnings
import importlib.util
import torch
from peft import PeftMixedModel
from peft.tuners.lora.hqq import HqqLoraLinear
from hqq.backends.torchao import HQQLinearTorchWeightOnlynt4

# --- Engine init patch Control ---
_engine_patches_applied = False
def apply_engine_init_patches(logger) -> List[str]:
    """
    Applies all necessary monkey-patches. This function should be called once
    at the beginning of the engine's lifecycle.
    Returns a list of applied patch names and checks performed.
    """
    global _engine_patches_applied
    if _engine_patches_applied:
        return ["patches_already_applied"]

    applied = []
    _suppress_known_warnings(logger)
    applied.append("suppress_known_warnings")
    applied.extend(f"check_package:{pkg}" for pkg, available in check_packages(logger).items() if available)

    logger.info("Applying MP13 Engine patches...")

    _patch_mistral_regex_kwarg_collision(logger)
    applied.append("patch_mistral_regex_kwarg_collision")
    _patch_phi3_triton_multigpu(logger) # This needs to run early because of MetaPathFinder
    applied.append("patch_phi3_triton_multigpu")
    _patch_cache_get_max_length(logger)
    applied.append("patch_cache_get_max_length")
    _patch_peft_for_quantized_models(logger)
    applied.append("patch_peft_for_quantized_models")
    _patch_peft_for_concurrency(logger)
    applied.append("patch_peft_for_concurrency")
    
    _engine_patches_applied = True
    logger.info("All MP13 Engine patches applied.")
    return applied

# --- inference mode  patch Control ---
def apply_infer_patches(logger):

    """Applies all patches related to the inference module."""
    current_generate = PeftMixedModel.generate
    if current_generate.__name__ == "_mixed_generate_like_peft" and current_generate is not _mixed_generate_like_peft:
        logger.warning(
            "Unexpected: PeftMixedModel.generate already patched by %s.",
            getattr(current_generate, "__module__", "unknown"),
        )
    try:
        from . import mp13_infer as _infer
        if getattr(_infer, "_ORIG_GENERATE", None) is _mixed_generate_like_peft:
            logger.warning("Unexpected: _ORIG_GENERATE points to _mixed_generate_like_peft (recursive patch).")
    except Exception as exc:
        logger.warning("Unexpected: failed to verify mp13_infer patch state: %s", exc)

    if current_generate.__name__ != '_mixed_generate_like_peft':
        PeftMixedModel.generate = _mixed_generate_like_peft
        logger.info("Patched PeftMixedModel.generate() using implementation from mp13_infer.")

    if HqqLoraLinear.forward.__name__ != '_final_guard_forward':
        HqqLoraLinear.forward = _final_guard_forward
        logger.info("Patched HqqLoraLinear.forward for torchAO device compatibility.")

    if HQQLinearTorchWeightOnlynt4.forward.__name__ != '_bias_fix_hqq_fwd':
        HQQLinearTorchWeightOnlynt4.forward = _bias_fix_hqq_fwd
        logger.info("Patched HQQLinearTorchWeightOnlynt4.forward for torchAO device compatibility.")


def _suppress_known_warnings(logger):
    """Suppresses known, non-critical warnings from various libraries."""
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="Torchmetrics v0.9 introduced a new argument class_reduction")
    warnings.filterwarnings("ignore", message="Support for nested tensors is experimental")
    warnings.filterwarnings("ignore", message="co_lnotab is deprecated, use co_lines instead.", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="fan_in_fan_out is set to False but the target module is `Conv1D`")
    warnings.filterwarnings('ignore', message='MatMul8bitLt: inputs will be cast.*', module='bitsandbytes')
    warnings.filterwarnings('ignore', message='The input hidden states seems to be silently casted.*')
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    
    try:
        import peft.tuners.lora.layer as _ll
        warnings.filterwarnings("ignore", message=r"Unsupported layer type '.*HQQLinearTorchWeightOnlynt4.*'", module=_ll.__name__,)
    except ImportError:
        pass # peft not installed, no need to suppress

def check_packages(logger) -> dict[str, bool]:
    """
    Checks for the availability of optional, performance-critical packages.
    Logs warnings for missing packages and returns their status.
    """
    availability = {}
    
    # 1. CUDA
    availability['cuda'] = torch.cuda.is_available()
    if not availability['cuda']:
        logger.warning("CUDA is not available. The engine will run on CPU, which will be very slow. Quantization and high-performance features are disabled.")
    else:
        logger.info(f"CUDA is available. Found {torch.cuda.device_count()} device(s).")

    # 2. xformers
    if importlib.util.find_spec("xformers"):
        availability['xformers'] = True
        logger.info("xformers is available. Memory-efficient attention will be used if supported by the model.")
    else:
        availability['xformers'] = False
        logger.info("xformers is not available. Performance may be impacted for models that support it.")

    # 3. flash-attn (for flash_attention_2)
    if importlib.util.find_spec("flash_attn"):
        availability['flash_attn'] = True
        logger.info("flash-attn is available. 'flash_attention_2' can be used for optimal performance on supported models.")
    else:
        availability['flash_attn'] = False
        logger.warning("flash-attn is not installed. For best performance, consider `pip install flash-attn --no-build-isolation`.")

    # 4. triton
    if importlib.util.find_spec("triton"):
        availability['triton'] = True
        logger.info("triton is available. It will be used by torch.compile for optimized kernels.")
    else:
        availability['triton'] = False
        logger.warning("triton is not available. torch.compile performance may be reduced.")

    # 5. bitsandbytes
    if importlib.util.find_spec("bitsandbytes"):
        availability['bitsandbytes'] = True
        logger.info("bitsandbytes is available. BNB quantization (4-bit, 8-bit) is supported.")
    else:
        availability['bitsandbytes'] = False
        logger.info("bitsandbytes is not available. 4-bit and 8-bit quantization will not work.")

    return availability

#  --- Inference mode patch functions  ---

# Import the function to be used as a patch from the inference module.
# This is safe as mp13_infer does not import mp13_patches.
from .mp13_infer import _mixed_generate_like_peft

# --- Patch for HqqLoraLinear.forward ---
_orig_hqq_lora_fwd = HqqLoraLinear.forward

def _final_guard_forward(self: HqqLoraLinear, x, *args, **kwargs):
    # ── 1. keep dense bias with the activation’s GPU (fix we added) ──────────
    bl = self.base_layer
    if getattr(bl, "bias", None) is not None and bl.bias.device != x.device:
        bl.bias = torch.nn.Parameter(
            bl.bias.to(x.device, non_blocking=True),
            requires_grad=bl.bias.requires_grad,
        )

    # If no adapters are active, just call the original forward pass.
    # The original forward pass has its own checks for disabled adapters.
    if not self.active_adapters:
        return _orig_hqq_lora_fwd(self, x, *args, **kwargs)

    # ── 2. ensure both LoRA linear sub‑modules sit on *x.device* ─────────────
    adapter_name = self.active_adapters[0]           # MixedModel always sets 0
    for proj in (self.lora_A[adapter_name], self.lora_B[adapter_name]):
        # Move only once: the first parameter dictates the module’s device tag
        if next(proj.parameters()).device != x.device:
            proj.to(x.device, non_blocking=True)

    return _orig_hqq_lora_fwd(self, x, *args, **kwargs)

# --- Patch for HQQLinearTorchWeightOnlynt4.forward ---
_orig_hqq_torchao_fwd = HQQLinearTorchWeightOnlynt4.forward

def _bias_fix_hqq_fwd(self: HQQLinearTorchWeightOnlynt4, x, *args, **kwargs):
    # find the packed‑weight tensor that matmul will use
    if getattr(self, "bias", None) is not None:
        if self.bias.device != x.device:           # x is always on the right GPU
            self.bias = torch.nn.Parameter(
                self.bias.to(x.device, non_blocking=True),
                requires_grad=self.bias.requires_grad,
            )
    return _orig_hqq_torchao_fwd(self, x, *args, **kwargs)

# END --- Inference mode patch functions  ---


# --- Individual engine init Patch Functions ---

def _patch_cache_get_max_length(logger):
    """HOT-FIX: re-add deprecated `get_max_length()` so old model code works"""
    from transformers.cache_utils import Cache
    if not hasattr(Cache, "get_max_length"):
        def _get_max_length(self):
            # Newer API renamed this to get_seq_length / get_max_cache_shape
            try:
                return self.get_seq_length()           # transformers ≥4.49
            except AttributeError:
                # Fallback for even newer cache refactors
                try:
                    return self.get_max_cache_shape()
                except AttributeError:
                    # Last-ditch: infer from first layer
                    layer = next(iter(self.key_cache.values()))
                    return layer.shape[-2] if layer is not None else 0

        Cache.get_max_length = _get_max_length
        logger.info("Patched Cache.get_max_length() back in for compatibility.")

def _patch_mistral_regex_kwarg_collision(logger):
    """HOT-FIX: avoid duplicate fix_mistral_regex kwarg in TokenizersBackend._patch_mistral_regex."""
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend
    except Exception:
        return

    try:
        target = TokenizersBackend._patch_mistral_regex
        target_func = getattr(target, "__func__", target)
        sig = inspect.signature(target_func)
    except Exception:
        return

    if "fix_mistral_regex" not in sig.parameters:
        return

    if getattr(target_func, "__name__", "") == "_patch_mistral_regex_compat":
        return

    orig = target_func

    def _patch_mistral_regex_compat(cls, *args, **kwargs):
        fix_mistral_regex = kwargs.pop("fix_mistral_regex", None)
        if fix_mistral_regex is None:
            fix_mistral_regex = kwargs.pop("_fix_mistral_regex", None)
        try:
            return orig(cls, *args, fix_mistral_regex=fix_mistral_regex, **kwargs)
        except AttributeError as exc:
            if "backend_tokenizer" not in str(exc):
                raise
            if not args:
                raise
            tokenizer = args[0]
            if not hasattr(tokenizer, "pre_tokenizer"):
                raise
            if not (fix_mistral_regex is True or getattr(tokenizer, "fix_mistral_regex", False)):
                return tokenizer
            import tokenizers
            split_pretokenizer = tokenizers.pre_tokenizers.Split(
                pattern=tokenizers.Regex(
                    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                ),
                behavior="isolated",
            )
            current_pretokenizer = tokenizer.pre_tokenizer
            if isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Sequence):
                current_pretokenizer[0] = split_pretokenizer
            else:
                if isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Metaspace):
                    current_pretokenizer = tokenizers.pre_tokenizers.ByteLevel(
                        add_prefix_space=False, use_regex=False
                    )
                tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
                    [
                        split_pretokenizer,
                        current_pretokenizer,
                    ]
                )
            return tokenizer

    TokenizersBackend._patch_mistral_regex = classmethod(_patch_mistral_regex_compat)

    orig_init = TokenizersBackend.__init__
    if getattr(orig_init, "__name__", "") != "_tokenizersbackend_init_compat":
        def _tokenizersbackend_init_compat(self, *args, **kwargs):
            if "fix_mistral_regex" in kwargs and "_fix_mistral_regex" not in kwargs:
                kwargs["_fix_mistral_regex"] = kwargs.pop("fix_mistral_regex")
            return orig_init(self, *args, **kwargs)

        _tokenizersbackend_init_compat.__name__ = "_tokenizersbackend_init_compat"
        TokenizersBackend.__init__ = _tokenizersbackend_init_compat

    logger.info("Patched TokenizersBackend to avoid duplicate fix_mistral_regex kwarg.")

def _patch_peft_for_quantized_models(logger):
    """
    Applies patches to PEFT to allow PeftMixedModel to work correctly with
    BitsAndBytes and HQQ quantized base models.
    """
    try:
        from peft.tuners.mixed import model as _mixed
        from peft.tuners.mixed.model import MixedModel
        from peft import PeftModel, PeftMixedModel

        def _device_for_module(mod):
            """
            Return the device of any quantised Linear/Embedding layer, covering:
            • HQQ <=0.2.3  -> .W_q
            • HQQ >=0.2.4  -> .weight
            • Torch‑AO int4 -> .qweight
            • Anything else -> first buffer, else CPU/GPU default
            """
            for attr in ("qweight", "W_q", "weight"): # NB: literals order matters
                if hasattr(mod, attr):
                    t = getattr(mod, attr)
                    if isinstance(t, torch.Tensor):
                        return t.device
            for buf in mod.buffers(recurse=False):        # generic fallback
                return buf.device
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Patch 1: _replace_module in peft ---

        def _safe_replace_module(self, parent, target_name, new_module, target):
            device = _device_for_module(new_module.base_layer) 
            new_module = new_module.to(device)
            # HQQ weight‑only layers keep a *dense* bias parameter which may still be
            # on the default GPU (cuda:0).  Make sure it rides along:
            bl = new_module.base_layer if hasattr(new_module, "base_layer") else new_module
            if getattr(bl, "bias", None) is not None:   # <-- guard in case the layer is bias‑free
                bl.bias = torch.nn.Parameter(           # re‑wrap so *the parameter itself* is moved
                    bl.bias.to(device, non_blocking=True),
                    requires_grad=bl.bias.requires_grad  # preserve training flag
                )
            setattr(parent, target_name, new_module)

        MixedModel._replace_module = _safe_replace_module
        logger.info("Applied HQQ-aware _replace_module patch to PeftMixedModel.")

        def _peft_get_base_model(self):
            base = getattr(self, "base_model", None)
            if base is None:
                return self
            if hasattr(base, "get_base_model"):
                return base.get_base_model()
            if hasattr(base, "model"):
                return base.model
            return base

        if not hasattr(PeftMixedModel, "get_base_model"):
            PeftMixedModel.get_base_model = _peft_get_base_model
            logger.info("Patched PeftMixedModel.get_base_model for PEFT 0.15 compatibility.")
        if not hasattr(MixedModel, "get_base_model"):
            MixedModel.get_base_model = _peft_get_base_model
            logger.info("Patched MixedModel.get_base_model for PEFT 0.15 compatibility.")

        # --- Patch 1.2: Training side (does not work yet) make LoRA skip empty-parameter modules (e.g. HQQ-wrapped kernels)
        import peft.tuners.lora.model as _lora_model

        _original_replace = _lora_model.LoraModel._replace_module

        def _safe_replace_for_training(self, parent, name, new_module, target):
            try:
                _original_replace(self, parent, name, new_module, target)
            except StopIteration:
                logger.warning(f"[PATCH] Skipping LoRA injection into "
                    f"{new_module.__class__.__name__} (no parameters)")

        _lora_model.LoraModel._replace_module = _safe_replace_for_training
        logger.info("Applied safe _replace_module patch to LoraModel")

        # --- Patch 2: create‑new‑module in peft ---
        from peft import LoraConfig
        from peft.tuners.lora.layer import LoraLayer 
        from hqq.backends.torchao import HQQLinearTorchWeightOnlynt4

        # grab the *exact* module object that MixedModel imported
        lora_pkg: types.ModuleType = importlib.import_module("peft.tuners.lora")
        LoraModel = lora_pkg.LoraModel                    # same class MixedModel sees
        _orig_cnm = LoraModel._create_new_module  

        @classmethod
        def _hqq_bnb_aware_cnm(cls, cfg: LoraConfig, name, target, **kw):
            # HQQ weight‑only branch
            if isinstance(target, HQQLinearTorchWeightOnlynt4):
                from peft.tuners.lora.hqq import HqqLoraLinear
                return HqqLoraLinear(
                    target, name,
                    cfg.r, cfg.lora_alpha, cfg.lora_dropout,
                    cfg.init_lora_weights,
                    getattr(cfg, "use_rslora", False),
                    getattr(cfg, "use_dora",  False),
                    getattr(cfg, "lora_bias", "none"),
                )

            # Bits‑and‑Bytes branch – strip flags so PEFT thinks it's dense
            kw.pop("loaded_in_4bit", None)
            kw.pop("loaded_in_8bit", None)

            # ── call the original in a way that works for *both* forms ──
            if isinstance(_orig_cnm, classmethod):
                return _orig_cnm.__func__(cls, cfg, name, target, **kw)
            else:                           # plain function (PEFT 0.15.x)
                return _orig_cnm(cfg, name, target, **kw)

        LoraModel._create_new_module = _hqq_bnb_aware_cnm  # <- single assignment!
        logger.info("Applied HQQ-HQQLinearTorchWeightOnlynt4 to  _create_new_module  patch to LoraModel.")
        # -------------------------------------------------------------------

        # --- Patch 3: HQQ-aware layer update for correct device placement ---
        _original_lora_update_layer = LoraLayer.update_layer
        
        def _update_layer_hqq(self, *args, **kwargs):
            adapter_name = args[0] if args else kwargs.get("adapter_name")
            _original_lora_update_layer(self, *args, **kwargs)

            dev = _device_for_module(self.base_layer)

            def _move_module(mod):
                try:
                    has_meta = any(getattr(p, "is_meta", False) for p in mod.parameters())
                except Exception:
                    has_meta = False
                if has_meta and hasattr(mod, "to_empty"):
                    return mod.to_empty(device=dev)
                return mod.to(dev, non_blocking=True)

            # move Linear LoRA matrices - re-assign!
            self.lora_A[adapter_name] = _move_module(self.lora_A[adapter_name])
            self.lora_B[adapter_name] = _move_module(self.lora_B[adapter_name])

            # move optional embedding deltas
            if hasattr(self, "lora_embedding_A") and adapter_name in self.lora_embedding_A:
                self.lora_embedding_A[adapter_name] = self.lora_embedding_A[adapter_name].to(dev, non_blocking=True)
                self.lora_embedding_B[adapter_name] = self.lora_embedding_B[adapter_name].to(dev, non_blocking=True)

            # make sure dense bias of the base HQQ layer is on the same GPU
            bl = self.base_layer
            if getattr(bl, "bias", None) is not None and bl.bias.device != dev:
                bl.bias = torch.nn.Parameter(bl.bias.to(dev, non_blocking=True),
                                            requires_grad=bl.bias.requires_grad)
        LoraLayer.update_layer = _update_layer_hqq
        logger.info("Applied HQQ device compatibility patch to LoraLayer.")

        # --- Patch 4: PEFT >= 0.15 compatibility for Phi-3 ---
        if not hasattr(MixedModel, "_update_offload"):
            MixedModel._update_offload = lambda self, *a, **k: None
            logger.info("Applied _update_offload patch to MixedModel for PEFT >= 0.15.")

        # --- Patch 5: Guarantee _hf_hook exists before merge_and_unload() ---
        from peft.tuners.lora.model import LoraModel as LoraModel_for_patch # Avoid name clash

        class _NoOpHook:
            def remove_hook(self, *_, **__): pass
            def detach_hook(self, *_, **__): pass

        def _ensure_hf_hooks(model):
            for m in model.modules():
                if not hasattr(m, "_hf_hook"):
                    m._hf_hook = _NoOpHook()

        def _wrap_merge(cls):
            if not hasattr(cls, "merge_and_unload"): return
            orig = cls.merge_and_unload
            def safe_merge(self, *a, **kw):
                _ensure_hf_hooks(self)
                merged = orig(self, *a, **kw)
                _ensure_hf_hooks(merged)
                return merged
            cls.merge_and_unload = safe_merge

        for _cls in (PeftModel, PeftMixedModel, MixedModel, LoraModel_for_patch):
            _wrap_merge(_cls)
        logger.info("Applied _hf_hook patch for safe merge_and_unload.")
    except ImportError as e:
        logger.error(f"Could not apply PEFT patches for quantized models: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while applying PEFT patches: {e}")

def _patch_phi3_triton_multigpu(logger):
    """HOT-FIX: Phi-3 Triton multi-GPU device mismatch (v3 - Robust)"""
    _PHI3_TRITON_LAYER_SUFFIX = ".triton_blocksparse_attention_layer"

    def _phi3_triton_guard(mod):
        # The target is the BlockSparseFlashAttention class inside the module
        if not hasattr(mod, "BlockSparseFlashAttention"):
            return

        BlockSparseFlashAttentionClass = mod.BlockSparseFlashAttention
        if getattr(BlockSparseFlashAttentionClass, "_device_guarded", False):
            return

        orig_forward = BlockSparseFlashAttentionClass.forward

        @staticmethod
        def guarded_forward(ctx, q, k, v, *a, **kw):
            # This guard ensures the Triton kernel runs on the same CUDA device as the input tensors, if on CUDA.
            if q.device.type == "cuda":
                with torch.cuda.device(q.device.index):
                    return orig_forward(ctx, q, k, v, *a, **kw)
            # If not on a CUDA device, the Triton kernel would fail anyway. Call original without the device context.
            return orig_forward(ctx, q, k, v, *a, **kw)

        BlockSparseFlashAttentionClass.forward = guarded_forward
        BlockSparseFlashAttentionClass._device_guarded = True
        logger.info(f"Applied Triton device guard patch to BlockSparseFlashAttention.forward in {mod.__name__}")


    class _Phi3TritonFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname.endswith(_PHI3_TRITON_LAYER_SUFFIX):
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                if spec and spec.loader:
                    original_exec_module = spec.loader.exec_module # type: ignore
                    def exec_and_patch(module):
                        original_exec_module(module)
                        _phi3_triton_guard(module)
                    spec.loader.exec_module = exec_and_patch # type: ignore
                return spec
            return None

    # Unconditionally insert the finder to ensure it's active.
    if not any(isinstance(finder, _Phi3TritonFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Phi3TritonFinder())
        logger.info("Inserted _Phi3TritonFinder for Phi-3 multi-GPU patch.")

_peft_patched_for_concurrency = False

def _patch_peft_for_concurrency(logger):
    """
    Applies monkey-patches to PEFT model classes to make adapter mutations thread-safe.
    """
    global _peft_patched_for_concurrency
    if _peft_patched_for_concurrency:
        return

    from peft import PeftModel

    def _guard_mutation(fn: Callable) -> Callable:
        """A decorator to wrap a model's method with its adapter mutation lock."""
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            # The 'self' here could be the raw model or a compiled wrapper (e.g., OptimizedModule).
            # We must unwrap it to find the lock on the original model instance.
            unwrapped_self = getattr(self, "_orig_mod", self)
            lock = getattr(unwrapped_self, "_adapter_mutation_lock", None)

            if lock is None:
                logger.error(f"_adapter_mutation_lock not found on unwrapped model {type(unwrapped_self).__name__} for method {fn.__name__}. Proceeding without lock.")
                return fn(self, *args, **kwargs)
            with lock:
                return fn(self, *args, **kwargs)
        return wrapped

    logger.info("Applying concurrency patches to PEFT model classes for thread-safety.")
    # Methods to patch on the PeftModel class (and its subclasses like PeftMixedModel)
    methods_to_patch = [ "set_adapter", "enable_adapter_layers", "disable_adapter_layers", "load_adapter", "delete_adapter", "add_adapter", "merge_adapter", "unmerge_adapter", "merge_and_unload" ]
    for method_name in methods_to_patch:
        if hasattr(PeftModel, method_name) and not getattr(getattr(PeftModel, method_name), "__is_guarded", False):
            guarded_method = _guard_mutation(getattr(PeftModel, method_name))
            guarded_method.__is_guarded = True # type: ignore
            setattr(PeftModel, method_name, guarded_method)
            logger.info(f"Patched PeftModel.{method_name} with mutation guard.")
    _peft_patched_for_concurrency = True

    # --- END Individual engine init Patch Functions ---
