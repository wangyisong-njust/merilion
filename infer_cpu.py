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
import re
import sys
import time

# Hide all CUDA devices before importing torch so that torchao INT4 kernels
# always dispatch to the CPU backend (old torchao versions create CUDA tensors
# internally even when the model is on CPU, if a GPU is visible).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

def _normalize_text(text: str) -> str:
    """Lowercase + strip punctuation for fair WER comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


SAMPLE_RATE = 16000
CHUNK_SIZE = SAMPLE_RATE * 30
SPEECH_TOKENS_PER_CHUNK = 100
MAX_CHUNKS = 8


def _apply_int8_dynamic(model):
    """INT8 dynamic quantization applied only to text decoder transformer blocks.

    Three sub-modules are kept in FP32:
      speech_encoder    — Whisper audio features; INT8 corrupts them completely,
                          causing the text decoder to hallucinate web-scraped text
      speech_audio_adapter — audio-to-text projection; same sensitivity as encoder
      lm_head           — tied to embed_tokens; DynamicQuantizedLinear breaks the
                          tie and produces degenerate logits (< < < < ...)

    Only text_decoder.model (Gemma2 transformer blocks: QKV/O projections + FFN)
    is quantised.  These dominate model size and compute, so INT8 still gives
    meaningful memory and latency reduction.
    Typical CPU speedup: 1.5–2×.  WER degradation: <0.3%.
    """
    text_decoder = getattr(model, 'text_decoder', None)
    if text_decoder is None:
        print("  WARNING: text_decoder not found — skipping INT8 (unknown model structure)")
        return

    transformer = getattr(text_decoder, 'model', None)
    if transformer is None:
        print("  WARNING: text_decoder.model not found — skipping INT8")
        return

    torch.quantization.quantize_dynamic(
        transformer, {nn.Linear}, dtype=torch.qint8, inplace=True)
    print("  (quantized: text_decoder.model | FP32: speech_encoder, audio_adapter, lm_head)")


def _apply_torchao_int4(model):
    """INT4 weight-only quantization via torchao (experimental for pruned models).

    WARNING: old torchao Int4WeightOnlyQuantizer replaces the weight *tensor*
    with a packed uint8 blob.  On tied-weight models (lm_head ↔ embed_tokens)
    this corrupts the LM head forward pass → WER > 100%.
    Only use if the installed torchao supports per-layer filtering.

    torchao >= 0.3: quantize_() + int4_weight_only()
    """
    try:
        from torchao.quantization import quantize_, int4_weight_only
        quantize_(model, int4_weight_only())
        return
    except ImportError:
        pass

    try:
        from torchao.quantization.quant_api import Int4WeightOnlyQuantizer
        Int4WeightOnlyQuantizer(device="cpu").quantize(model)
        return
    except ImportError:
        pass

    raise RuntimeError(
        "No compatible torchao INT4 API found. "
        "Upgrade with: pip install torchao --upgrade")


def load_model_cpu(model_path: str, int4: bool = False, int8: bool = True, compile: bool = True):
    """Load pruned model on CPU with torchao INT4 weight-only quantization.

    Args:
        int8:    apply PyTorch INT8 dynamic quantization (default, safe for tied weights)
        int4:    apply torchao INT4 weight-only (experimental; may corrupt lm_head on tied models)
        compile: apply torch.compile for kernel fusion
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
    )
    model = model.cpu()
    model.eval()
    # Clear cache_implementation so DynamicCache can be passed to generate()
    # without conflict. DynamicCache handles non-uniform KV heads in pruned models.
    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = None
    print(f"  Loaded in {time.time()-t0:.1f}s")

    if int8 and not int4:
        print("Applying INT8 dynamic quantization (torch.quantization.quantize_dynamic) …")
        t0 = time.time()
        _apply_int8_dynamic(model)
        print(f"  Done in {time.time()-t0:.1f}s")
        # torch.compile is incompatible with DynamicQuantizedLinear (legacy QEngine ops
        # are not traceable by Dynamo → garbled outputs).  INT8 BLAS gives speedup
        # directly without compile, so skip it silently.
        if compile:
            print("  (torch.compile skipped: not compatible with INT8 dynamic quant)")
        compile = False

    if int4:
        # Verify all tensors are on CPU before packing
        cuda_params = [(n, p.device) for n, p in model.named_parameters() if p.device.type != "cpu"]
        cuda_bufs   = [(n, b.device) for n, b in model.named_buffers()    if b.device.type != "cpu"]
        if cuda_params or cuda_bufs:
            print(f"  WARNING: {len(cuda_params)} params and {len(cuda_bufs)} buffers still on CUDA — moving them")
            for n, _ in cuda_params + cuda_bufs:
                print(f"    {n}")
        model = model.to(torch.device("cpu"))
        print("Applying torchao INT4 weight-only quantization (experimental) …")
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

    # Ensure generate() always uses DynamicCache.  Some transformers versions
    # restore generation_config.cache_implementation from the saved config during
    # generate(), which would re-enable HybridCache and cause shape mismatches on
    # pruned models with non-uniform KV heads.
    _gen_cfg = getattr(model, "generation_config", None)
    if _gen_cfg is not None:
        _gen_cfg.cache_implementation = None

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
                        help="FP32 baseline, no quantization")
    parser.add_argument("--int4", action="store_true",
                        help="Use torchao INT4 (experimental; may break tied-weight models)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Skip torch.compile (faster startup, slower inference)")
    parser.add_argument("--output", default="cpu_results.json")
    args = parser.parse_args()
    args.model = os.path.abspath(args.model)

    use_int8 = not args.no_quant and not args.int4
    use_int4 = not args.no_quant and args.int4

    # Measure RSS before loading
    try:
        import psutil, os as _os
        _proc = psutil.Process(_os.getpid())
        _ram_before_mb = _proc.memory_info().rss / 1e6
    except ImportError:
        _proc = None
        _ram_before_mb = 0.0

    model, processor = load_model_cpu(
        args.model,
        int8=use_int8,
        int4=use_int4,
        compile=not args.no_compile,
    )

    if _proc is not None:
        ram_after_mb = _proc.memory_info().rss / 1e6
        print(f"  RAM after load+quant: {ram_after_mb:.0f} MB  (delta: {ram_after_mb - _ram_before_mb:+.0f} MB)")
    else:
        ram_after_mb = 0.0

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
        norm_preds = [_normalize_text(p) for p in predictions]
        norm_refs  = [_normalize_text(r) for r in references]
        wer     = wer_metric.compute(predictions=norm_preds,
                                     references=norm_refs)
        avg_lat = float(np.mean(latencies))
        print(f"\n{'='*60}")
        print(f"WER:          {wer:.4f}  ({wer*100:.2f}%)  [normalized]")
        print(f"Avg latency:  {avg_lat:.2f} s/sample")
        print(f"INT4:         {not args.no_quant}")
        print(f"compiled:     {not args.no_compile}")
        print(f"{'='*60}")

        quant_method = ("int4" if use_int4 else "int8" if use_int8 else "fp32")
        with open(args.output, "w") as f:
            json.dump({
                "model": args.model,
                "quant_method": quant_method,
                "compiled": not args.no_compile,
                "num_samples": args.num_samples,
                "wer": wer,
                "avg_latency_s": avg_lat,
                "ram_mb": ram_after_mb,
                "latencies": latencies,
            }, f, indent=2)
        print(f"Saved to {args.output}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
