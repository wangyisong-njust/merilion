"""GPU inference benchmark for MERaLiON-2 (original + pruned models).

Mirrors infer_cpu.py in structure and JSON output schema so the two can be
compared directly in a summary table or make_demo_html.py.

Quantization backends:
  bf16  (default) — BF16 + FlashAttention-2 / SDPA    (fastest; full quality)
  fp16            — FP16 + FlashAttention-2 / SDPA
  int8            — BitsAndBytes LLM.int8() — speech modules kept in FP16
  int4            — BitsAndBytes NF4 4-bit  — bfloat16 compute dtype

Timing uses torch.cuda.synchronize() for wall-clock accuracy.
GPU VRAM is measured via torch.cuda.max_memory_allocated().

Output JSON adds gpu_mem_gb and device fields alongside the same
wer / avg_latency_s / avg_decode_tps / quant_method keys as the CPU version.

Usage:
    # BF16 baseline on single GPU (default):
    python infer_gpu.py \\
        --model /path/to/MERaLiON-2-3B \\
        --dataset /path/to/IMDA_PART1_mono_en_30_ASR \\
        --num_samples 20 --output gpu_bf16.json

    # BitsAndBytes INT8:
    python infer_gpu.py --model ... --dataset ... --quant int8 --output gpu_int8.json

    # BitsAndBytes NF4 INT4:
    python infer_gpu.py --model ... --dataset ... --quant int4 --output gpu_int4.json

    # Single audio file:
    python infer_gpu.py --model ... --audio sample.wav
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


# ── shared helpers (same as infer_cpu.py) ────────────────────────────────────

def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _model_is_pruned(model) -> bool:
    try:
        cfg = model.text_decoder.model.config
        return (getattr(cfg, "midblock_start", -1) >= 0
                and getattr(cfg, "midblock_ratio", 1.0) < 1.0)
    except Exception:
        return False


SAMPLE_RATE = 16000
CHUNK_SIZE = SAMPLE_RATE * 30
SPEECH_TOKENS_PER_CHUNK = 100
MAX_CHUNKS = 8


def prepare_audio(audio_array: np.ndarray, sample_rate: int, processor):
    """Resample + mel features. Same as infer_cpu.py."""
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
            chunk = np.pad(chunk, (0, target_sr - len(chunk)), "constant")
        chunks.append(chunk)
    chunks = chunks[:MAX_CHUNKS]
    out = fe(chunks, sampling_rate=target_sr, return_attention_mask=True,
             padding="max_length", return_tensors="pt", do_normalize=True)
    return out.input_features, out.attention_mask, len(chunks) * SPEECH_TOKENS_PER_CHUNK


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_gpu(model_path: str,
                   quant: str = "bf16",
                   flash_attn: bool = True,
                   device: str = "cuda"):
    """Load MERaLiON-2 (original or pruned) on GPU.

    Args:
        quant:      'bf16' | 'fp16' | 'int8' | 'int4'
        flash_attn: use FlashAttention-2 when quant in {bf16, fp16}
                    (requires flash-attn package; falls back to SDPA on failure)
        device:     e.g. 'cuda', 'cuda:0', 'cuda:1'
    """
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor

    print(f"Loading processor …")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    common_kwargs = dict(use_safetensors=True)

    # meralion2_bl's from_pretrained() ignores device_map and quantization_config;
    # it always loads on CPU.  For BF16/FP16 we just move to GPU afterwards.
    # For BnB INT8/INT4: load weights in FP16 on CPU, then manually swap each
    # nn.Linear into a BnB-typed layer (copying the FP16 data into Int8Params /
    # Params4bit), then .to(device) triggers BnB's cuda() hook which does the
    # actual quantization.  This avoids the meta-tensor issue that arises when
    # replace_with_bnb_linear() is called after weights are already loaded.

    # Modules to leave in FP16 — Whisper encoder + audio adapter + tied lm_head.
    BNB_SKIP = ["speech_encoder", "speech_audio_adapter", "lm_head"]

    if quant in ("int8", "int4"):
        import bitsandbytes as bnb
        from torch import nn as _nn

        print(f"Loading model → CPU FP16, will apply BnB {quant.upper()} post-hoc …")
        t0 = time.time()
        model = MERaLiON2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            **common_kwargs,
        )

        for mod_name, module in list(model.named_modules()):
            if not isinstance(module, _nn.Linear):
                continue
            if any(skip in mod_name for skip in BNB_SKIP):
                continue

            # Navigate to parent
            parts = mod_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            child = parts[-1]

            w = module.weight.data.cpu()   # FP16 on CPU
            has_bias = module.bias is not None

            if quant == "int8":
                new_layer = bnb.nn.Linear8bitLt(
                    module.in_features, module.out_features,
                    bias=has_bias, has_fp16_weights=False, threshold=6.0,
                )
                new_layer.weight = bnb.nn.Int8Params(
                    w, requires_grad=False, has_fp16_weights=False)
            else:  # int4 NF4
                new_layer = bnb.nn.Linear4bit(
                    module.in_features, module.out_features,
                    bias=has_bias, quant_type="nf4",
                    compute_dtype=torch.bfloat16,
                )
                new_layer.weight = bnb.nn.Params4bit(
                    w, requires_grad=False, quant_type="nf4")

            if has_bias:
                new_layer.bias = _nn.Parameter(module.bias.data)
            setattr(parent, child, new_layer)

        # .to(device) calls Int8Params.cuda() / Params4bit.cuda() which
        # performs the actual weight quantization on GPU.
        model = model.to(device)

    else:
        dtype = torch.bfloat16 if quant == "bf16" else torch.float16
        attn_impl = "flash_attention_2" if flash_attn else "sdpa"
        print(f"Loading model {quant.upper()} (attn={attn_impl}) on GPU …")
        t0 = time.time()
        # Load to CPU first, then move to target device.
        # device_map=<device-string> is unreliable in older transformers +
        # meralion2_bl — model silently stays on CPU → 0.00 GB VRAM reported.
        try:
            model = MERaLiON2ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                **common_kwargs,
            )
        except Exception as e:
            if flash_attn and "flash" in str(e).lower():
                print(f"  FlashAttn2 unavailable ({e}), falling back to sdpa …")
                model = MERaLiON2ForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    attn_implementation="sdpa",
                    **common_kwargs,
                )
            else:
                raise
        model = model.to(device)

    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, processor


# ── inference ─────────────────────────────────────────────────────────────────

def transcribe_gpu(model, processor, audio_array: np.ndarray, sample_rate: int,
                   instruction: str = "Transcribe the speech",
                   max_new_tokens: int = 128,
                   device: str = "cuda",
                   speculative: bool = False,
                   gamma: int = 5) -> tuple:
    """Run ASR inference on GPU for a single audio sample.

    Uses the same audio preprocessing and input-building logic as
    infer_cpu.py:transcribe(), but moves tensors to CUDA and wraps
    the generate() call with CUDA sync for accurate wall-clock timing.

    Returns (text, stats) where stats mirrors the CPU version:
        n_tokens, decode_tps, [prefill_s, decode_s for BF16/FP16]
    """
    input_features, feature_attention_mask, n_speech = prepare_audio(
        audio_array, sample_rate, processor)

    tokenizer = processor.tokenizer
    speech_token_id = model.config.speech_token_index

    conversation = [{"role": "user",
                     "content": (f"Instruction: {instruction} \n"
                                 "Follow the text instruction based on the "
                                 "following audio: <SpeechHere>")}]
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    try:
        pos = raw_ids.index(speech_token_id)
    except ValueError:
        raise RuntimeError(
            f"speech_token_id={speech_token_id} not in tokenized prompt.")

    input_ids = torch.tensor(
        [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]],
        dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # Detect actual device from model parameters (BnB device_map="auto" may
    # place the model on cuda:0 regardless of the `device` argument).
    try:
        _actual_device = next(p.device for p in model.parameters()
                              if p.device.type != "cpu")
    except StopIteration:
        _actual_device = torch.device(device)

    # Detect compute dtype (BnB quantized params report int8/uint8;
    # fall back to bfloat16 as the safe compute dtype).
    try:
        _dtype = next(p.dtype for p in model.parameters()
                      if p.dtype in (torch.float16, torch.bfloat16))
    except StopIteration:
        _dtype = torch.bfloat16

    # Move inputs to the model's actual device
    input_ids              = input_ids.to(_actual_device)
    attention_mask         = attention_mask.to(_actual_device)
    input_features         = input_features.to(_actual_device).to(_dtype)
    feature_attention_mask = feature_attention_mask.to(_actual_device)

    # Pre-create cache to avoid the Gemma2 HybridCache overflow issue
    # (same workaround as infer_cpu.py:transcribe()).
    _gen_cfg = getattr(model, "generation_config", None)
    if _gen_cfg is not None:
        _gen_cfg.cache_implementation = None

    max_cache = input_ids.shape[1] + max_new_tokens
    if _model_is_pruned(model):
        from transformers import DynamicCache
        past_kv = DynamicCache()
    else:
        from transformers.cache_utils import HybridCache
        past_kv = HybridCache(
            model.text_decoder.model.config,
            max_batch_size=1,
            max_cache_len=max_cache,
            dtype=_dtype,
            device=_actual_device,
        )

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        past_key_values=past_kv,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        ],
    )
    if speculative:
        # n-gram prompt-lookup speculative decoding — GPU batches the K+1
        # verification step cheaply; acceptance rate typically 40-70%.
        gen_kwargs["prompt_lookup_num_tokens"] = gamma

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)
    torch.cuda.synchronize()
    total_s = time.time() - t0

    generated  = output_ids[0][input_ids.shape[1]:]
    n_tokens   = max(len(generated), 1)
    decode_tps = n_tokens / total_s if total_s > 0 else 0.0
    stats = {"n_tokens": n_tokens, "decode_tps": decode_tps}

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip(), stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPU ASR inference benchmark — MERaLiON-2")
    parser.add_argument("--model", required=True,
                        help="Model directory (original or pruned+tuned)")
    parser.add_argument("--audio", default=None,
                        help="Single audio file (.wav/.flac/.mp3)")
    parser.add_argument("--instruction", default="Transcribe the speech")
    parser.add_argument("--dataset", default=None,
                        help="IMDA_PART1_mono_en_30_ASR dataset path")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--quant", default="bf16",
                        choices=["bf16", "fp16", "int8", "int4"],
                        help="Quantization: bf16 (default) | fp16 | int8 (BnB) | int4 (BnB NF4)")
    parser.add_argument("--no_flash_attn", action="store_true",
                        help="Use SDPA instead of FlashAttention-2")
    parser.add_argument("--device", default="cuda",
                        help="CUDA device, e.g. cuda / cuda:0 / cuda:1 (default: cuda)")
    parser.add_argument("--output", default="gpu_results.json")
    parser.add_argument("--save_samples", action="store_true",
                        help="Include per-sample predictions + references in JSON output")
    parser.add_argument("--speculative", action="store_true",
                        help="Enable n-gram prompt-lookup speculative decoding on GPU")
    parser.add_argument("--gamma", type=int, default=5,
                        help="Spec decoding lookahead window (default: 5)")
    args = parser.parse_args()
    args.model = os.path.abspath(args.model)

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found. Use infer_cpu.py for CPU inference.")
        sys.exit(1)

    torch.cuda.reset_peak_memory_stats(args.device)

    model, processor = load_model_gpu(
        args.model,
        quant=args.quant,
        flash_attn=not args.no_flash_attn,
        device=args.device,
    )

    gpu_mem_load_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    print(f"  GPU VRAM after load: {gpu_mem_load_gb:.2f} GB")

    def _infer(audio, sr, instruction):
        return transcribe_gpu(model, processor, audio, sr,
                              instruction=instruction,
                              max_new_tokens=args.max_new_tokens,
                              device=args.device,
                              speculative=args.speculative,
                              gamma=args.gamma)

    # ── single audio file ──────────────────────────────────────────────────
    if args.audio:
        import soundfile as sf
        audio, sr = sf.read(args.audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=-1)
        audio = audio.astype(np.float32)
        t0 = time.time()
        text, stats = _infer(audio, sr, args.instruction)
        print(f"\nTranscription ({time.time()-t0:.2f}s, {stats['decode_tps']:.1f} tok/s):\n  {text}")
        return

    # ── dataset benchmark + WER ────────────────────────────────────────────
    if args.dataset:
        from datasets import load_from_disk
        import evaluate

        data = load_from_disk(os.path.abspath(args.dataset))
        subset = data.shuffle(seed=42).select(
            range(10500, 10500 + args.num_samples))

        # Warm up (first GPU call is slower due to kernel JIT)
        print("Warming up GPU …")
        _sample0 = subset[0]
        _a = np.asarray(_sample0["context"]["audio"]["array"], dtype=np.float32)
        _sr = _sample0["context"]["audio"]["sampling_rate"]
        _instr = (_sample0["instruction"]["text"]
                  if isinstance(_sample0["instruction"], dict)
                  else _sample0["instruction"])
        _infer(_a, _sr, _instr)
        torch.cuda.reset_peak_memory_stats(args.device)

        predictions, references, latencies = [], [], []
        samples_out = []
        for i in range(args.num_samples):
            sample = subset[i]
            audio = np.asarray(sample["context"]["audio"]["array"], dtype=np.float32)
            sr    = sample["context"]["audio"]["sampling_rate"]
            if audio.ndim == 2:
                audio = audio.mean(axis=-1)
            instr = (sample["instruction"]["text"]
                     if isinstance(sample["instruction"], dict)
                     else sample["instruction"])
            ref = sample["other_attributes"]["Transcription"]

            t0 = time.time()
            pred, stats = _infer(audio, sr, instr)
            elapsed = time.time() - t0
            predictions.append(pred)
            references.append(ref)
            latencies.append(elapsed)
            print(f"  [{i+1:3d}/{args.num_samples}] {elapsed:5.2f}s  "
                  f"{stats['decode_tps']:6.1f} tok/s | {pred[:60]}")
            entry = {"idx": i, "reference": ref, "prediction": pred,
                     "latency_s": elapsed, **stats}
            samples_out.append(entry)

        wer_metric = evaluate.load("wer")
        norm_preds = [_normalize_text(p) for p in predictions]
        norm_refs  = [_normalize_text(r) for r in references]
        wer        = wer_metric.compute(predictions=norm_preds, references=norm_refs)
        avg_lat    = float(np.mean(latencies))
        avg_tps    = float(np.mean([s["decode_tps"] for s in samples_out]))
        gpu_peak_gb = torch.cuda.max_memory_allocated(args.device) / 1e9

        print(f"\n{'='*60}")
        print(f"WER:           {wer:.4f}  ({wer*100:.2f}%)  [normalized]")
        print(f"Avg latency:   {avg_lat:.2f} s/sample")
        print(f"Avg decode:    {avg_tps:.2f} tok/s")
        print(f"GPU VRAM peak: {gpu_peak_gb:.2f} GB")
        print(f"quant:         {args.quant}")
        print(f"device:        {args.device}")
        print(f"{'='*60}")

        result = {
            "model":           args.model,
            "quant_method":    args.quant,
            "device":          args.device,
            "num_samples":     args.num_samples,
            "speculative":     args.speculative,
            "gamma":           args.gamma if args.speculative else None,
            "wer":             wer,
            "avg_latency_s":   avg_lat,
            "avg_decode_tps":  avg_tps,
            "gpu_mem_load_gb": gpu_mem_load_gb,
            "gpu_mem_peak_gb": gpu_peak_gb,
            "latencies":       latencies,
        }
        if args.save_samples:
            result["samples"] = samples_out
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
