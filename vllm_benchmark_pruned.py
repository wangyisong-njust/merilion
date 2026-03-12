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

    t0 = time.time()
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        limit_mm_per_prompt={"audio": 1},
        trust_remote_code=True,
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
    avg_time = total_time / num_samples
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tps = total_tokens / total_time

    print(f"\nResults ({label}):")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"  Avg per sample:  {avg_time:.2f}s")
    print(f"  Output tokens:   {total_tokens}")
    print(f"  Tokens/sec:      {tps:.1f}")
    print(f"  Load time:       {load_time:.1f}s")

    # Print first 3 predictions
    for i, o in enumerate(outputs[:3]):
        pred = o.outputs[0].text.removeprefix("<Speaker1>: ").strip()
        print(f"  Sample {i}: {pred[:100]}...")

    # Free GPU memory
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "total_time": total_time,
        "avg_per_sample": avg_time,
        "output_tokens": total_tokens,
        "tokens_per_sec": tps,
        "load_time": load_time,
        "num_samples": num_samples,
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
    from meralion2_vllm_pruned import MERaLiON2PrunedForConditionalGeneration
    ModelRegistry.register_model(
        "MERaLiON2ForConditionalGeneration",
        MERaLiON2PrunedForConditionalGeneration,
    )

    # Load test data
    from datasets import load_from_disk
    print(f"Loading test data from {args.dataset}...")
    test_data = load_from_disk(args.dataset)
    test_subset = test_data.shuffle(seed=42).select(range(10500, 10500 + args.num_samples))

    sampling_params = SamplingParams(
        temperature=0.0, top_p=0.9, top_k=50,
        repetition_penalty=1.0, seed=42, max_tokens=256,
    )

    results = {}

    # Benchmark pruned model
    if os.path.exists(args.pruned):
        results["pruned"] = benchmark_model(
            args.pruned, "pruned-50%-mid4-22", test_subset, sampling_params, args.num_samples)
    else:
        print(f"ERROR: Pruned model not found: {args.pruned}")
        sys.exit(1)

    # Benchmark original model
    if args.original and os.path.exists(args.original):
        results["original"] = benchmark_model(
            args.original, "original", test_subset, sampling_params, args.num_samples)

    # Summary
    print(f"\n{'=' * 60}")
    print("LATENCY BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    for label, r in results.items():
        print(f"  {label:25s}  {r['avg_per_sample']:.2f}s/sample  {r['tokens_per_sec']:.1f} tok/s  load={r['load_time']:.1f}s")

    if "pruned" in results and "original" in results:
        speedup = results["original"]["avg_per_sample"] / results["pruned"]["avg_per_sample"]
        print(f"\n  Speedup: {speedup:.2f}x")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
