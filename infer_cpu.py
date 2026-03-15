"""CPU inference for pruned MERaLiON-2 using torchao INT4 weight-only quantization.

torchao int4_weight_only() stores weights as INT4 and dequantizes on-the-fly
during each forward pass using optimized SIMD kernels (AVX-VNNI on x86,
NEON on ARM).  Combined with torch.compile, this achieves real INT4 GEMM
speedup — not just smaller storage.

The pruned model's non-uniform layer dimensions are fully supported since
torchao quantizes each nn.Linear independently.

Start point: the merged pruned BF16 model from merge_lora.py.
The AWQ model (GPU-only kernels) cannot be used for CPU inference.

Install:  pip install torchao

Usage:
    # Single audio file:
    python infer_cpu.py \
        --model meralion_tune_log/MERaLiON-2-3B-v3-td50-mid3-22-tune \
        --audio sample.wav

    # WER + latency benchmark on dataset:
    python infer_cpu.py \
        --model meralion_tune_log/MERaLiON-2-3B-v3-td50-mid3-22-tune \
        --dataset /path/to/IMDA_PART1_mono_en_30_ASR \
        --num_samples 50 --output cpu_results.json

    # FP32 baseline (no quantization):
    python infer_cpu.py --model ... --dataset ... --no_quant
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

SAMPLE_RATE = 16000
CHUNK_SIZE = SAMPLE_RATE * 30
SPEECH_TOKENS_PER_CHUNK = 100
MAX_CHUNKS = 8


def _apply_torchao_int4(model):
    """Apply INT4 weight-only quantization, handling different torchao API versions."""
    # torchao >= 0.3: quantize_() + int4_weight_only()
    try:
        from torchao.quantization import quantize_, int4_weight_only
        quantize_(model, int4_weight_only())
        return
    except ImportError:
        pass

    # torchao 0.1–0.2: Int4WeightOnlyQuantizer
    try:
        from torchao.quantization.quant_api import Int4WeightOnlyQuantizer
        Int4WeightOnlyQuantizer().quantize(model)
        return
    except ImportError:
        pass

    raise RuntimeError(
        "No compatible torchao INT4 API found. "
        "Upgrade with: pip install torchao --upgrade")


def load_model_cpu(model_path: str, int4: bool = True, compile: bool = True):
    """Load pruned model on CPU with torchao INT4 weight-only quantization.

    Args:
        int4:    apply torchao int4_weight_only to all Linear layers
        compile: apply torch.compile for SIMD kernel fusion (recommended)
    """
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    print(f"Loading processor …")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model in FP32 on CPU …")
    t0 = time.time()
    model = MERaLiON2ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    # Clear cache_implementation so DynamicCache can be passed to generate()
    # without conflict. DynamicCache handles non-uniform KV heads in pruned models.
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None
    print(f"  Loaded in {time.time()-t0:.1f}s")

    if int4:
        model = model.to("cpu")   # ensure all tensors are on CPU before INT4 packing
        print("Applying torchao INT4 weight-only quantization …")
        t0 = time.time()
        _apply_torchao_int4(model)
        print(f"  Done in {time.time()-t0:.1f}s")

    if compile:
        print("Compiling with torch.compile (first inference will be slow) …")
        model = torch.compile(model, mode="reduce-overhead")

    return model, processor


def prepare_audio(audio_array: np.ndarray, sample_rate: int, processor):
    """Resample, chunk, extract mel features. Returns (input_features, mask, n_speech_tokens)."""
    import librosa
    fe = processor.feature_extractor
    target_sr = fe.sampling_rate

    if sample_rate != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate,
                                       target_sr=target_sr)
    chunks = []
    for i in range(0, len(audio_array), CHUNK_SIZE):
        chunk = audio_array[i:i + CHUNK_SIZE]
        if len(chunk) < target_sr:
            chunk = np.pad(chunk, (0, target_sr - len(chunk)), 'constant')
        chunks.append(chunk)
    chunks = chunks[:MAX_CHUNKS]

    out = fe(chunks, sampling_rate=target_sr, return_attention_mask=True,
             padding="max_length", return_tensors="pt", do_normalize=True)
    return out.input_features, out.attention_mask, len(chunks) * SPEECH_TOKENS_PER_CHUNK


def transcribe(model, processor, audio_array: np.ndarray, sample_rate: int,
               instruction: str = "Transcribe the speech",
               max_new_tokens: int = 128) -> str:
    """Run ASR inference for a single audio sample."""
    input_features, feature_attention_mask, n_speech = prepare_audio(
        audio_array, sample_rate, processor)

    tokenizer = processor.tokenizer
    speech_token_id = model.config.speech_token_index

    prompt = (
        "<start_of_turn>user\n"
        f"Instruction: {instruction} \n"
        "Follow the text instruction based on the following audio: "
        "<SpeechHere><end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        pos = raw_ids.index(speech_token_id)
    except ValueError:
        raise RuntimeError(
            f"speech_token_id={speech_token_id} not in tokenized prompt. "
            "Verify processor matches model.")

    # Expand the single <SpeechHere> placeholder to n_speech copies
    input_ids = torch.tensor(
        [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]],
        dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    from transformers import DynamicCache
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            past_key_values=DynamicCache(),
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<end_of_turn>"),
            ],
        )

    generated = output_ids[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()


def main():
    parser = argparse.ArgumentParser(
        description="CPU ASR inference — pruned MERaLiON-2 with torchao INT4")
    parser.add_argument("--model", required=True,
                        help="Merged pruned model dir (NOT the AWQ dir)")
    parser.add_argument("--audio", default=None,
                        help="Single audio file (.wav/.flac/.mp3)")
    parser.add_argument("--instruction", default="Transcribe the speech")
    parser.add_argument("--dataset", default=None,
                        help="IMDA_PART1_mono_en_30_ASR dataset path")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--no_quant", action="store_true",
                        help="FP32 baseline, no INT4 quantization")
    parser.add_argument("--no_compile", action="store_true",
                        help="Skip torch.compile (faster startup, slower inference)")
    parser.add_argument("--output", default="cpu_results.json")
    args = parser.parse_args()
    args.model = os.path.abspath(args.model)

    model, processor = load_model_cpu(
        args.model,
        int4=not args.no_quant,
        compile=not args.no_compile,
    )

    # ── single audio file ──────────────────────────────────────────────────
    if args.audio:
        import soundfile as sf
        audio, sr = sf.read(args.audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=-1)
        audio = audio.astype(np.float32)
        t0 = time.time()
        text = transcribe(model, processor, audio, sr,
                          instruction=args.instruction,
                          max_new_tokens=args.max_new_tokens)
        print(f"\nTranscription ({time.time()-t0:.2f}s):\n  {text}")
        return

    # ── dataset benchmark + WER ────────────────────────────────────────────
    if args.dataset:
        from datasets import load_from_disk
        import evaluate

        data = load_from_disk(os.path.abspath(args.dataset))
        subset = data.shuffle(seed=42).select(
            range(10500, 10500 + args.num_samples))

        predictions, references, latencies = [], [], []
        for i in range(args.num_samples):
            sample = subset[i]
            audio = np.asarray(sample["context"]["audio"]["array"],
                               dtype=np.float32)
            sr    = sample["context"]["audio"]["sampling_rate"]
            if audio.ndim == 2:
                audio = audio.mean(axis=-1)
            instr = (sample["instruction"]["text"]
                     if isinstance(sample["instruction"], dict)
                     else sample["instruction"])
            ref = sample["other_attributes"]["Transcription"]

            t0 = time.time()
            pred = transcribe(model, processor, audio, sr,
                              instruction=instr,
                              max_new_tokens=args.max_new_tokens)
            elapsed = time.time() - t0
            predictions.append(pred)
            references.append(ref)
            latencies.append(elapsed)
            print(f"  [{i+1:3d}/{args.num_samples}] {elapsed:5.1f}s | {pred[:70]}")

        wer_metric = evaluate.load("wer")
        wer     = wer_metric.compute(predictions=predictions,
                                     references=references)
        avg_lat = float(np.mean(latencies))
        print(f"\n{'='*60}")
        print(f"WER:          {wer:.4f}  ({wer*100:.2f}%)")
        print(f"Avg latency:  {avg_lat:.2f} s/sample")
        print(f"INT4:         {not args.no_quant}")
        print(f"compiled:     {not args.no_compile}")
        print(f"{'='*60}")

        with open(args.output, "w") as f:
            json.dump({
                "model": args.model,
                "int4": not args.no_quant,
                "compiled": not args.no_compile,
                "num_samples": args.num_samples,
                "wer": wer,
                "avg_latency_s": avg_lat,
                "latencies": latencies,
            }, f, indent=2)
        print(f"Saved to {args.output}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
