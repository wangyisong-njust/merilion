"""Marlin kernel smoke test on A100.

Measures decode throughput on the verifier's TEXT path only (no audio,
no speculation) to isolate the Linear/MatMul kernel cost.  Compares:
  - bf16 baseline
  - W4A16 compressed-tensors checkpoint (loaded via HfQuantizer →
    CompressedLinear → Marlin kernel on sm_80+)

If Marlin works on A100, decode tok/s on the W4A16 path should be
~1.5-2× the bf16 baseline (4× weight bandwidth reduction, partially
offset by dequant overhead).  On L40 (sm_89) the Marlin path was 0.3×
bf16, hence this re-test on Ampere.

Usage:
  python bench_marlin_smoke.py \\
      --bf16   /path/to/MERaLiON-2-3B \\
      --w4a16  quant_checkpoints/MERaLiON-2-3B-W4A16-RTN \\
      --steps  256
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_gpu import load_model_gpu


def time_decode(model, processor, n_steps, prompt, device, warmup=5):
    """Measure pure decode tok/s on text_decoder.

    1. Tokenise `prompt`, prefill once.
    2. Loop n_steps single-token greedy decodes; time only the loop.
    """
    from transformers.cache_utils import HybridCache

    tok = processor.tokenizer
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    seq_len = input_ids.shape[1]
    cache_max = seq_len + n_steps + warmup + 4

    td = model.text_decoder
    _dtype = next(p.dtype for p in td.parameters()
                  if p.dtype in (torch.float16, torch.bfloat16))
    kv = HybridCache(td.model.config, max_batch_size=1,
                     max_cache_len=cache_max, dtype=_dtype, device=device)

    with torch.inference_mode():
        # Prefill
        out = td(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=kv, use_cache=True,
            cache_position=torch.arange(0, seq_len, device=device),
            return_dict=True,
        )
        next_tok = int(out.logits[0, -1].argmax())
        cur = seq_len

        # Warmup
        for _ in range(warmup):
            o = td(
                input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=device),
                attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=device),
                past_key_values=kv, use_cache=True,
                cache_position=torch.tensor([cur], device=device),
                return_dict=True,
            )
            next_tok = int(o.logits[0, -1].argmax())
            cur += 1

        # Timed decode
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            o = td(
                input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=device),
                attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=device),
                past_key_values=kv, use_cache=True,
                cache_position=torch.tensor([cur], device=device),
                return_dict=True,
            )
            next_tok = int(o.logits[0, -1].argmax())
            cur += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    return n_steps / dt, dt


def detect_quantized_linears(model):
    """Walk text_decoder, count every leaf-ish module that has a weight-like
    buffer.  We don't filter by class name — that's exactly what we want
    to discover (CompressedLinear / MarlinLinear / etc.).
    """
    from collections import Counter
    cnt = Counter()
    for name, mod in model.text_decoder.named_modules():
        # Treat as leaf if it has any weight-ish attr (weight, weight_packed,
        # qweight, B, qzeros, …) and no Linear-like submodules.
        has_w = any(hasattr(mod, attr) for attr in
                    ("weight", "weight_packed", "qweight", "B", "qzeros"))
        if not has_w:
            continue
        has_linear_child = any(
            type(c).__name__.endswith("Linear") for c in mod.children())
        if has_linear_child:
            continue
        cnt[type(mod).__name__] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16",  required=True, help="Path to bf16 MERaLiON-2-3B")
    ap.add_argument("--w4a16", required=True, help="Path to W4A16 compressed-tensors ckpt")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. " * 8,
                    help="Text prompt to prefill (length affects KV cache reads).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("=" * 64)
    print(f"GPU: {torch.cuda.get_device_name(args.device)}")
    print(f"  CC: sm_{torch.cuda.get_device_capability(args.device)[0]}{torch.cuda.get_device_capability(args.device)[1]}")
    print(f"  steps: {args.steps}")
    print("=" * 64)

    results = {}
    for tag, path in [("bf16", args.bf16), ("w4a16", args.w4a16)]:
        print(f"\n── {tag.upper()} : {path}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)
        if tag == "w4a16":
            # GPTQ-Marlin path: bypass HfQuantizer (optimum→gptqmodel→pcre is
            # blocked); use auto-gptq's from_quantized directly via our loader.
            from load_gptq_marlin import load_meralion2_gptq_marlin
            m, p = load_meralion2_gptq_marlin(
                path, args.bf16, device=args.device, dtype=torch.float16)
        else:
            m, p = load_model_gpu(path, quant="bf16",
                                  flash_attn=True, device=args.device)
        vram_load = torch.cuda.max_memory_allocated(args.device) / 1e9
        cnt = detect_quantized_linears(m)
        print(f"  VRAM after load: {vram_load:.2f} GB")
        print(f"  Linear classes:  {dict(cnt)}")

        tps, dt = time_decode(m, p, args.steps, args.prompt, args.device)
        vram_peak = torch.cuda.max_memory_allocated(args.device) / 1e9
        print(f"  decode: {tps:7.1f} tok/s   ({dt:.2f}s for {args.steps} steps)")
        print(f"  VRAM peak: {vram_peak:.2f} GB")
        results[tag] = {"tps": tps, "dt": dt, "vram_load": vram_load,
                        "vram_peak": vram_peak, "linear_classes": dict(cnt)}
        del m, p
        torch.cuda.empty_cache()

    print("\n" + "=" * 64)
    print(f"{'config':<10} {'tok/s':>10} {'VRAM_load':>12} {'speedup':>10}")
    base = results["bf16"]["tps"]
    for tag in ("bf16", "w4a16"):
        r = results[tag]
        sp = r["tps"] / base
        print(f"{tag:<10} {r['tps']:>10.1f} {r['vram_load']:>11.2f}G {sp:>9.2f}x")
    print("=" * 64)
    if results["w4a16"]["tps"] > base * 1.1:
        print("Marlin kernel: WORKING (>1.1× speedup)")
    elif results["w4a16"]["tps"] > base * 0.9:
        print("Marlin kernel: NEUTRAL (no speedup, no regression)")
    else:
        print("Marlin kernel: REGRESSION — check sm_xx and compressed_tensors lib version")


if __name__ == "__main__":
    main()
