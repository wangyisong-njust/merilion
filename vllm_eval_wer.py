"""WER evaluation and sample demonstration for a vLLM-deployed MERaLiON-2 model.

Usage:
    python vllm_eval_wer.py \
        --model meralion_tune_log/MERaLiON-2-3B-v3-td50-mid6-20-tune \
        --dataset /path/to/IMDA_PART1_mono_en_30_ASR \
        --num_samples 500 \
        --num_demo 10 \
        --output wer_results.json
"""
import sys
import os
import json
import gc
import argparse

import numpy as np
import evaluate

os.environ.setdefault("VLLM_USE_V1", "0")

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


def get_reference(sample):
    return sample["other_attributes"]["Transcription"]


def clean_pred(text):
    return text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()


def main():
    parser = argparse.ArgumentParser(description="vLLM WER evaluation for MERaLiON-2")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to IMDA_PART1_mono_en_30_ASR dataset")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of test samples")
    parser.add_argument("--num_demo", type=int, default=10, help="Number of sample outputs to display")
    parser.add_argument("--output", default="wer_results.json", help="Output JSON file")
    args = parser.parse_args()
    args.model = os.path.abspath(args.model)
    args.dataset = os.path.abspath(args.dataset)

    import torch
    from vllm import LLM, SamplingParams
    from vllm import ModelRegistry
    from pruned_gemma2_vllm import is_pruned_model
    from meralion2_vllm_pruned import (
        MERaLiON2PrunedForConditionalGeneration,
        _register_processor_factory,
    )
    ModelRegistry.register_model(
        "MERaLiON2ForConditionalGeneration",
        MERaLiON2PrunedForConditionalGeneration,
    )
    _register_processor_factory()

    from datasets import load_from_disk
    print(f"Loading dataset from {args.dataset}...")
    test_data = load_from_disk(args.dataset)
    # Samples 10500+ avoid overlap with train (0-10000) and val (10000-10500)
    test_subset = test_data.shuffle(seed=42).select(range(10500, 10500 + args.num_samples))

    sampling_params = SamplingParams(
        temperature=0.0, top_p=0.9, top_k=50,
        repetition_penalty=1.0, seed=42, max_tokens=128,
        stop=["<end_of_turn>", "<eos>"],
    )

    print(f"\nLoading model from {args.model}...")
    quant_kwarg = {}
    cfg_path = os.path.join(args.model, "config.json")
    if os.path.exists(cfg_path):
        import json as _json
        with open(cfg_path) as _f:
            if _json.load(_f).get("quantization_config", {}).get("quant_type") == "awq":
                quant_kwarg["quantization"] = "awq"
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        limit_mm_per_prompt={"audio": 1},
        trust_remote_code=True,
        **quant_kwarg,
    )

    print(f"Running inference on {args.num_samples} samples...")
    all_inputs = [build_inputs(test_subset[i]) for i in range(args.num_samples)]
    outputs = llm.generate(all_inputs, sampling_params=sampling_params)

    predictions = [clean_pred(o.outputs[0].text) for o in outputs]
    references = [get_reference(test_subset[i]) for i in range(args.num_samples)]

    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)

    print(f"\n{'='*60}")
    print(f"WER EVALUATION — {args.model}")
    print(f"{'='*60}")
    print(f"  Samples:  {args.num_samples}")
    print(f"  WER:      {wer:.4f}  ({wer*100:.2f}%)")
    print(f"{'='*60}")

    # Sample demonstrations
    print(f"\n--- {args.num_demo} Sample Outputs ---")
    for i in range(min(args.num_demo, args.num_samples)):
        ref = references[i]
        pred = predictions[i]
        sample_wer = wer_metric.compute(predictions=[pred], references=[ref])
        print(f"\n[{i+1}]")
        print(f"  REF:  {ref}")
        print(f"  PRED: {pred}")
        print(f"  WER:  {sample_wer:.3f}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "model": args.model,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "wer": wer,
        "samples": [
            {"reference": references[i], "prediction": predictions[i]}
            for i in range(args.num_samples)
        ],
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
