"""Marlin kernel smoke test via vLLM (A100, W4A16 compressed-tensors).

vLLM's compressed-tensors integration auto-dispatches to its Marlin
kernel on sm_80+ when the checkpoint is W4A16 pack-quantized. This
script measures decode tok/s on a fixed text prompt for:
    - bf16 baseline
    - W4A16-RTN (Marlin path)
and reports the speedup.

Requires: vllm + the vllm_plugin_meralion2 plugin so vLLM can load the
multimodal MERaLiON-2 checkpoint.

Usage:
  python bench_marlin_vllm.py \\
      --bf16   /path/to/MERaLiON-2-3B \\
      --w4a16  quant_checkpoints/MERaLiON-2-3B-W4A16-RTN \\
      --max_tokens 256
"""
import argparse
import os
import time

import torch


def bench(model_path, max_tokens, prompt, gpu_mem_util=0.4):
    from vllm import LLM, SamplingParams

    print(f"\n── Loading: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=1024,
        enforce_eager=False,    # let vLLM use CUDA graphs (fastest)
    )

    # Inspect what quantization method vLLM picked.
    try:
        cfg = llm.llm_engine.model_config
        qcfg = cfg.quantization or "none"
        print(f"  vLLM quantization: {qcfg}")
    except Exception as e:
        print(f"  (couldn't read quant cfg: {e})")

    sp_warm = SamplingParams(max_tokens=16, temperature=0.0)
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0,
                        ignore_eos=True)         # force fixed length

    # Warmup
    _ = llm.generate([prompt], sp_warm, use_tqdm=False)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    n_gen = len(out[0].outputs[0].token_ids)
    tps = n_gen / dt
    print(f"  decode: {tps:7.1f} tok/s   ({dt:.2f}s for {n_gen} tokens)")
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM peak: {vram_peak:.2f} GB")

    # Free
    del llm
    torch.cuda.empty_cache()
    return {"tps": tps, "dt": dt, "n_gen": n_gen, "vram_peak": vram_peak,
            "quant": str(qcfg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16",  required=True)
    ap.add_argument("--w4a16", required=True)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. " * 8)
    args = ap.parse_args()

    print("=" * 64)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"  CC: sm_{cap[0]}{cap[1]}")
    print(f"  max_tokens: {args.max_tokens}")
    print("=" * 64)

    results = {
        "bf16":  bench(args.bf16,  args.max_tokens, args.prompt),
        "w4a16": bench(args.w4a16, args.max_tokens, args.prompt),
    }

    print("\n" + "=" * 64)
    print(f"{'config':<10} {'tok/s':>10} {'VRAM(GB)':>10} {'quant':>20} {'speedup':>10}")
    base = results["bf16"]["tps"]
    for tag in ("bf16", "w4a16"):
        r = results[tag]
        sp = r["tps"] / base
        print(f"{tag:<10} {r['tps']:>10.1f} {r['vram_peak']:>10.2f} {r['quant']:>20} {sp:>9.2f}x")
    print("=" * 64)
    if results["w4a16"]["tps"] > base * 1.2:
        print("Marlin kernel: WORKING (>1.2× speedup)")
    elif results["w4a16"]["tps"] > base * 0.9:
        print("Marlin kernel: NEUTRAL (no speedup, no regression)")
    else:
        print("Marlin kernel: REGRESSION — check vLLM logs above for which kernel was chosen")


if __name__ == "__main__":
    main()
