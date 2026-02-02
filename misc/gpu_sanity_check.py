#!/usr/bin/env python
# Minimal standalone sanity check for CUDA vs CPU generation and logits.
import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model


def _resolve_model_from_config(cfg_path: Path) -> Optional[str]:
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # mp13 chat config typically stores base_model_path under engine_params
    engine_params = data.get("engine_params") or {}
    model_path = engine_params.get("base_model_path")
    if isinstance(model_path, str) and model_path.strip():
        return model_path.strip()
    return None


def _set_sdp_mode(math_only: bool) -> None:
    if not torch.cuda.is_available():
        return
    if math_only:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

def _set_tf32_mode(mode: str) -> None:
    if not torch.cuda.is_available():
        return
    if mode == "on":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif mode == "off":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _load_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype: str,
    attn_impl: str,
    device_map: Optional[str],
    low_cpu_mem_usage: bool,
):
    torch_dtype = {
        "auto": "auto",
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(dtype, "auto")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    dm = None if device_map in (None, "", "none") else device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=dm,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    if dm is None:
        model.to(device)
    model.eval()
    return tokenizer, model


def _infer_bootstrap_targets(model) -> list[str]:
    model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
    model_type = model_type.lower()
    if "phi3" in model_type or "phi-3" in model_type:
        return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    if "qwen2" in model_type:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "v_proj"]


def _wrap_with_bootstrap_peft(model, use_seq2seq: bool):
    targets = _infer_bootstrap_targets(model)
    task_type = TaskType.SEQ_2_SEQ_LM if use_seq2seq else TaskType.CAUSAL_LM
    cfg = LoraConfig(
        r=1,
        lora_alpha=1,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        task_type=task_type,
    )
    peft_model = get_peft_model(model, cfg, adapter_name="bootstrap", mixed=True)
    peft_model.eval()
    print(f"[peft] wrapped with bootstrap adapter targets={targets} task_type={task_type.value}")
    if "bootstrap" in peft_model.peft_config:
        peft_model.delete_adapter("bootstrap")
        print("[peft] bootstrap adapter deleted")
    return peft_model


def _configure_special_tokens(tokenizer, model, pad_token: Optional[str], eos_token: Optional[str], add_pad: bool, use_pad_as_eos: bool):
    if pad_token:
        if tokenizer.pad_token != pad_token:
            pad_id = tokenizer.convert_tokens_to_ids(pad_token)
            if pad_id == tokenizer.unk_token_id:
                if add_pad:
                    tokenizer.add_special_tokens({"pad_token": pad_token})
                    model.resize_token_embeddings(len(tokenizer))
                else:
                    print(f"[warn] pad_token '{pad_token}' not in vocab; use --add-pad to add it.")
            else:
                tokenizer.pad_token = pad_token
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    if eos_token:
        if tokenizer.eos_token != eos_token:
            eos_id = tokenizer.convert_tokens_to_ids(eos_token)
            if eos_id == tokenizer.unk_token_id:
                print(f"[warn] eos_token '{eos_token}' not in vocab; leaving as-is.")
            else:
                tokenizer.eos_token = eos_token
        if tokenizer.eos_token_id is not None:
            model.config.eos_token_id = tokenizer.eos_token_id

    if use_pad_as_eos and tokenizer.pad_token_id is not None:
        model.config.eos_token_id = tokenizer.pad_token_id


def _print_special_tokens(tokenizer, model):
    print(
        "[tokens] pad_token="
        f"{tokenizer.pad_token!r} id={tokenizer.pad_token_id} | "
        f"eos_token={tokenizer.eos_token!r} id={tokenizer.eos_token_id}"
    )
    cfg = getattr(model, "config", None)
    if cfg is not None:
        print(f"[config] pad_token_id={getattr(cfg, 'pad_token_id', None)} eos_token_id={getattr(cfg, 'eos_token_id', None)}")


def _logits_stats(model, input_ids, attention_mask):
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, -1, :]
        finite = torch.isfinite(logits).all().item()
        mn = float(logits.min().item())
        mx = float(logits.max().item())
        argmax_id = int(torch.argmax(logits, dim=-1)[0].item())
    return finite, mn, mx, argmax_id


def _generate_ids(model, input_ids, attention_mask, max_new_tokens: int):
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=getattr(model.config, "pad_token_id", None),
            eos_token_id=getattr(model.config, "eos_token_id", None),
        )
    return out[0].tolist()

def _print_input_ids(tokenizer, enc):
    try:
        ids = enc["input_ids"][0].tolist()
        print(f"[inputs] len={len(ids)} head_ids={ids[:32]}")
        print(f"[inputs] text_preview={tokenizer.decode(ids, skip_special_tokens=False)[:200]!r}")
    except Exception as e:
        print(f"[inputs] debug error: {e}")

def _topk_logits(model, input_ids, attention_mask, k: int):
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, -1, :]
        vals, idx = torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1)
        return idx[0].tolist(), vals[0].tolist()

def _weights_stats(model):
    emb = getattr(model, "get_input_embeddings", None)
    if callable(emb):
        w = emb().weight
        with torch.no_grad():
            finite = torch.isfinite(w).all().item()
            mn = float(w.min().item())
            mx = float(w.max().item())
            mean = float(w.mean().item())
        return {"finite": finite, "min": mn, "max": mx, "mean": mean, "shape": tuple(w.shape)}
    return None

def _emb_checksum(model):
    emb = getattr(model, "get_input_embeddings", None)
    if not callable(emb):
        return None
    w = emb().weight
    with torch.no_grad():
        flat = w.flatten()
        n = min(1024, flat.numel())
        head = flat[:n]
        return float(head.sum().item())

def _print_cuda_info():
    if not torch.cuda.is_available():
        print("[cuda] not available")
        return
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"[cuda] device={props.name} capability={props.major}.{props.minor} total_gb={props.total_memory/1e9:.2f}")
    print(f"[cuda] torch_version={torch.__version__} cuda_version={torch.version.cuda} cudnn={torch.backends.cudnn.version()}")
    print(f"[cuda] tf32_matmul={torch.backends.cuda.matmul.allow_tf32} tf32_cudnn={torch.backends.cudnn.allow_tf32}")

def _print_template_info(tokenizer):
    tmpl = getattr(tokenizer, "chat_template", None)
    if not tmpl:
        print("[template] none")
        return
    try:
        import hashlib
        tmpl_text = str(tmpl)
        tmpl_hash = hashlib.sha256(tmpl_text.encode("utf-8")).hexdigest()[:12]
        has_cutoff = "Knowledge Cutoff" in tmpl_text
        print(f"[template] len={len(tmpl_text)} hash={tmpl_hash} has_cutoff={has_cutoff}")
    except Exception as e:
        print(f"[template] error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="GPU/CPU sanity check for model logits and generation.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python gpu_sanity_check.py --model ..\\granite-3.3-2b-instruct --device cuda --chat --system \"\" --prompt \"hi\" --pad-token \"<fim_pad>\"\n"
            "  python gpu_sanity_check.py --config granite.json --device cuda --chat --prompt \"hi\"\n"
            "  python gpu_sanity_check.py --model gpt2 --device cuda\n"
            "  python gpu_sanity_check.py --model ..\\granite-3.3-2b-instruct --device cuda --attn sdpa --chat --system \"\" --prompt \"hi\"\n"
            "  python gpu_sanity_check.py --model ..\\granite-3.3-2b-instruct --device cuda --device-map auto --low-cpu-mem --chat --prompt \"hi\"\n"
        ),
    )
    parser.add_argument("--model", type=str, default=None, help="Model path or HF ID.")
    parser.add_argument("--config", type=str, default=None, help="mp13 config JSON path (reads engine_params.base_model_path).")
    parser.add_argument("--prompt", type=str, default="hi", help="Prompt to test.")
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt for chat templates.")
    parser.add_argument("--chat", action="store_true", help="Apply tokenizer chat template if available.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Primary device for test.")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"], help="Torch dtype.")
    parser.add_argument("--attn", type=str, default="eager", help="Attention implementation (e.g., eager, sdpa).")
    parser.add_argument("--device-map", type=str, default="none", help="Device map for model load (e.g., auto, cuda:0, none).")
    parser.add_argument("--low-cpu-mem", action="store_true", help="Enable low_cpu_mem_usage during model load.")
    parser.add_argument("--max-new", type=int, default=32, help="Max new tokens for generate.")
    parser.add_argument("--math-sdp", action="store_true", help="Force math-only SDPA (disable flash/mem-efficient).")
    parser.add_argument("--tf32", type=str, default="keep", choices=["keep", "on", "off"], help="Control TF32 for matmul/cudnn.")
    parser.add_argument("--topk", type=int, default=0, help="Print top-k logits for the next token.")
    parser.add_argument("--weights", action="store_true", help="Print embedding weight stats.")
    parser.add_argument("--emb-check", action="store_true", help="Print embedding head checksum (first 1024 values).")
    parser.add_argument("--pad-token", type=str, default=None, help="Override tokenizer pad_token.")
    parser.add_argument("--eos-token", type=str, default=None, help="Override tokenizer eos_token.")
    parser.add_argument("--add-pad", action="store_true", help="Add pad_token to vocab if missing (resizes embeddings).")
    parser.add_argument("--use-pad-as-eos", action="store_true", help="Force eos_token_id to match pad_token_id.")
    parser.add_argument("--peft-wrap", action="store_true", help="Wrap base model in PeftMixedModel with bootstrap adapter, then delete it.")
    parser.add_argument("--peft-seq2seq", action="store_true", help="Use SEQ_2_SEQ_LM task type for bootstrap adapter.")
    parser.add_argument("--compare-cpu", action="store_true", help="Also load a CPU copy to compare logits.")
    args = parser.parse_args()

    model_name = args.model
    if not model_name and args.config:
        model_name = _resolve_model_from_config(Path(args.config))
    if not model_name:
        raise SystemExit("Provide --model or --config with engine_params.base_model_path.")

    _set_sdp_mode(args.math_sdp)
    _set_tf32_mode(args.tf32)
    print(f"[info] loading model={model_name} device={args.device} dtype={args.dtype} attn={args.attn}")
    _print_cuda_info()
    tokenizer, model = _load_model_and_tokenizer(
        model_name,
        args.device,
        args.dtype,
        args.attn,
        args.device_map,
        args.low_cpu_mem,
    )
    if args.peft_wrap:
        model = _wrap_with_bootstrap_peft(model, args.peft_seq2seq)
    _configure_special_tokens(tokenizer, model, args.pad_token, args.eos_token, args.add_pad, args.use_pad_as_eos)
    _print_special_tokens(tokenizer, model)
    _print_template_info(tokenizer)

    if args.chat and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if args.system is not None:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(prompt_text, return_tensors="pt")
        print("[info] using chat template prompt")
    else:
        enc = tokenizer(args.prompt, return_tensors="pt")
    _print_input_ids(tokenizer, enc)
    input_ids = enc["input_ids"].to(args.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)

    finite, mn, mx, argmax_id = _logits_stats(model, input_ids, attention_mask)
    print(f"[logits] finite={finite} min={mn:.4f} max={mx:.4f} argmax_id={argmax_id}")
    print(f"[logits] argmax_token={tokenizer.decode([argmax_id], skip_special_tokens=False)}")
    if args.topk and args.topk > 0:
        ids, vals = _topk_logits(model, input_ids, attention_mask, args.topk)
        decoded = [tokenizer.decode([i], skip_special_tokens=False) for i in ids]
        print(f"[logits] topk_ids={ids}")
        print(f"[logits] topk_vals={[round(v, 4) for v in vals]}")
        print(f"[logits] topk_tokens={decoded}")
    if args.weights:
        stats = _weights_stats(model)
        if stats:
            print(f"[weights] emb finite={stats['finite']} min={stats['min']:.4f} max={stats['max']:.4f} mean={stats['mean']:.4f} shape={stats['shape']}")
    if args.emb_check:
        chk = _emb_checksum(model)
        if chk is not None:
            print(f"[weights] emb head_sum={chk:.6f}")

    gen_ids = _generate_ids(model, input_ids, attention_mask, args.max_new)
    print(f"[generate] ids(head)={gen_ids[:20]}")
    print(f"[generate] text={tokenizer.decode(gen_ids, skip_special_tokens=False)}")

    if args.compare_cpu:
        print("[info] loading CPU copy for comparison (fp32).")
        cpu_tokenizer, cpu_model = _load_model_and_tokenizer(
            model_name,
            "cpu",
            "fp32",
            args.attn,
            "none",
            False,
        )
        cpu_enc = cpu_tokenizer(args.prompt, return_tensors="pt")
        cpu_ids = cpu_enc["input_ids"]
        cpu_mask = cpu_enc.get("attention_mask", None)
        c_finite, c_mn, c_mx, c_argmax = _logits_stats(cpu_model, cpu_ids, cpu_mask)
        print(f"[cpu logits] finite={c_finite} min={c_mn:.4f} max={c_mx:.4f} argmax_id={c_argmax}")
        print(f"[cpu logits] argmax_token={cpu_tokenizer.decode([c_argmax], skip_special_tokens=False)}")


if __name__ == "__main__":
    main()
