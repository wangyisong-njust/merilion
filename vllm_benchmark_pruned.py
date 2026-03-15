"""vLLM latency benchmark: pruned vs original MERaLiON-2 model.

Usage:
    python vllm_benchmark_pruned.py \
        --pruned meralion_checkpoints/MERaLiON-2-3B-v3-td50-mid4-22 \
        --original /path/to/MERaLiON-2-3B \
        --dataset /path/to/IMDA_PART1_mono_en_30_ASR \
        --num_samples 50
"""
import sys
import os
import time
import json
import gc
import argparse

import numpy as np

# Force vLLM V0 engine — our multimodal registration uses the V0 decorator API
# which is incompatible with the V1 engine's processor factory system.
os.environ.setdefault("VLLM_USE_V1", "0")

# Add vllm_inference to path for PrunedGemma2Model
script_dir = os.path.dirname(os.path.abspath(__file__))
vllm_dir = os.path.join(script_dir, "vllm_inference")
if vllm_dir not in sys.path:
    sys.path.insert(0, vllm_dir)


def build_inputs(sample):
    audio_array = np.asarray(sample["context"]["audio"]["array"], dtype=np.float32)
    sr = sample["context"]["audio"]["sampling_rate"]
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=-1)
    if len(audio_array) / sr < 1:
        audio_array = np.pad(audio_array, (0, sr), 'constant')
    instruction = sample["instruction"]["text"] if isinstance(sample["instruction"], dict) else sample["instruction"]
    prompt = (
        "<start_of_turn>user\n"
        f"Instruction: {instruction} \nFollow the text instruction based on the following audio: <SpeechHere><end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    return {"prompt": prompt, "multi_modal_data": {"audio": [(audio_array, sr)]}}


def benchmark_model(model_path, label, test_subset, sampling_params, num_samples):
    import torch
    from vllm import LLM

    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {label}")
    print(f"Model: {model_path}")
    print(f"{'=' * 60}")

    # Detect AWQ format explicitly — vLLM auto-detection may miss AWQ for custom
    # trust_remote_code models, causing it to load INT4-packed weights as FP16.
    # compressed-tensors (W8A16/W4A16-RTN/FP8) are always auto-detected correctly.
    quant_kwarg = {}
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.exists(cfg_path):
        import json as _json
        with open(cfg_path) as _f:
            _cfg = _json.load(_f)
        if _cfg.get("quantization_config", {}).get("quant_type") == "awq":
            quant_kwarg["quantization"] = "awq"

    t0 = time.time()
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        limit_mm_per_prompt={"audio": 1},
        trust_remote_code=True,
        **quant_kwarg,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Warmup (2 samples)
    print("Warming up...")
    warmup = [build_inputs(test_subset[i]) for i in range(min(2, num_samples))]
    _ = llm.generate(warmup, sampling_params=sampling_params)

    # Benchmark
    print(f"Running {num_samples} samples...")
    all_inputs = [build_inputs(test_subset[i]) for i in range(num_samples)]

    t_start = time.time()
    outputs = llm.generate(all_inputs, sampling_params=sampling_params)
    t_end = time.time()

    total_time = t_end - t_start

    # Per-request prefill/decode breakdown from vLLM metrics
    ttfts = []        # time-to-first-token (prefill latency) per request, seconds
    decode_tps = []   # decode tok/s per request
    total_output_tokens = 0

    for o in outputs:
        n_tokens = len(o.outputs[0].token_ids)
        total_output_tokens += n_tokens
        m = getattr(o, "metrics", None)
        if (m is not None
                and getattr(m, "first_token_time", None) is not None
                and getattr(m, "finished_time", None) is not None
                and getattr(m, "arrival_time", None) is not None):
            ttfts.append(m.first_token_time - m.arrival_time)
            decode_time = m.finished_time - m.first_token_time
            decode_tokens = max(n_tokens - 1, 0)
            if decode_time > 0 and decode_tokens > 0:
                decode_tps.append(decode_tokens / decode_time)

    avg_ttft_ms = np.mean(ttfts) * 1000 if ttfts else None
    avg_decode_tps = np.mean(decode_tps) if decode_tps else None
    # Fallback throughput if metrics unavailable
    total_tps = total_output_tokens / total_time

    print(f"\nResults ({label}):")
    print(f"  Samples:              {num_samples}")
    print(f"  Total output tokens:  {total_output_tokens}  ({total_output_tokens/num_samples:.1f} tok/sample)")
    print(f"  Total time:           {total_time:.1f}s")
    if avg_ttft_ms is not None:
        print(f"  Prefill (TTFT):       {avg_ttft_ms:.0f} ms  (mean per request)")
        print(f"  Decode speed:         {avg_decode_tps:.1f} tok/s  (mean per request)")
    else:
        print(f"  Throughput:           {total_tps:.1f} tok/s  (batch, metrics unavailable)")
    print(f"  Load time:            {load_time:.1f}s")

    # Print first 3 predictions
    for i, o in enumerate(outputs[:3]):
        pred = o.outputs[0].text.removeprefix("<Speaker1>: ").strip()
        print(f"  Sample {i}: {pred[:100]}...")

    # Free GPU memory
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "load_time": load_time,
        "num_samples": num_samples,
        "total_output_tokens": total_output_tokens,
        "tokens_per_sample": total_output_tokens / num_samples,
        "total_time": total_time,
        "avg_ttft_ms": avg_ttft_ms,
        "avg_decode_tps": avg_decode_tps,
        "total_tps": total_tps,
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM latency benchmark for pruned MERaLiON-2")
    parser.add_argument("--pruned", required=True, help="Path to pruned model checkpoint")
    parser.add_argument("--original", default=None, help="Path to original model (for comparison)")
    parser.add_argument("--dataset", required=True, help="Path to IMDA_PART1_mono_en_30_ASR dataset")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples for benchmark")
    parser.add_argument("--output", default="vllm_benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()

    # Register pruned model support
    from vllm import SamplingParams, ModelRegistry
    from pruned_gemma2_vllm import is_pruned_model
    from meralion2_vllm_pruned import (
        MERaLiON2PrunedForConditionalGeneration,
        _register_processor_factory,
    )
    ModelRegistry.register_model(
        "MERaLiON2ForConditionalGeneration",
        MERaLiON2PrunedForConditionalGeneration,
    )
    # Re-run processor factory registration now that ModelRegistry is set up.
    # The module-level call may have failed/skipped if vLLM validated against
    # ModelRegistry before the register_model call above.
    _register_processor_factory()

    # Load test data
    from datasets import load_from_disk
    print(f"Loading test data from {args.dataset}...")
    test_data = load_from_disk(args.dataset)
    test_subset = test_data.shuffle(seed=42).select(range(10500, 10500 + args.num_samples))

    sampling_params = SamplingParams(
        temperature=0.0, top_p=0.9, top_k=50,
        repetition_penalty=1.0, seed=42, max_tokens=128,
        stop=["<end_of_turn>", "<eos>"],
    )

    results = {}

    # Benchmark pruned model
    if os.path.exists(args.pruned):
        results["pruned"] = benchmark_model(
            args.pruned, "pruned", test_subset, sampling_params, args.num_samples)
    else:
        print(f"ERROR: Pruned model not found: {args.pruned}")
        sys.exit(1)

    # Benchmark original model
    if args.original and os.path.exists(args.original):
        results["original"] = benchmark_model(
            args.original, "original", test_subset, sampling_params, args.num_samples)

    # Summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    header = f"  {'model':25s}  {'prefill(ms)':>12}  {'decode(tok/s)':>14}  {'tok/sample':>10}  {'load(s)':>8}"
    print(header)
    print(f"  {'-'*25}  {'-'*12}  {'-'*14}  {'-'*10}  {'-'*8}")
    for label, r in results.items():
        prefill = f"{r['avg_ttft_ms']:.0f}" if r['avg_ttft_ms'] is not None else "N/A"
        decode = f"{r['avg_decode_tps']:.1f}" if r['avg_decode_tps'] is not None else "N/A"
        print(f"  {label:25s}  {prefill:>12}  {decode:>14}  {r['tokens_per_sample']:>10.1f}  {r['load_time']:>8.1f}")

    if "pruned" in results and "original" in results:
        rp, ro = results["pruned"], results["original"]
        if rp["avg_decode_tps"] and ro["avg_decode_tps"]:
            speedup = rp["avg_decode_tps"] / ro["avg_decode_tps"]
            print(f"\n  Decode speedup (pruned / original): {speedup:.2f}x")
        if rp["avg_ttft_ms"] and ro["avg_ttft_ms"]:
            prefill_ratio = ro["avg_ttft_ms"] / rp["avg_ttft_ms"]
            print(f"  Prefill speedup (pruned / original): {prefill_ratio:.2f}x")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
