"""End-to-end example: audio file → EAGLE+W4A16-accelerated transcription.

Usage:
    python example_inference.py <audio.wav> [--repo YOUR_USERNAME/...]

Requires: transformers, auto_gptq (compiled with marlin+exllama+exllamav2),
soundfile or librosa, huggingface_hub.  See `requirements.txt`.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

# Local imports — when installed via `from_pretrained` the package is
# unpacked alongside this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_eagle import MERaLiON2EAGLEForASR


SAMPLE_RATE = 16000
CHUNK_SIZE  = SAMPLE_RATE * 30
SPEECH_TOKENS_PER_CHUNK = 100
MAX_CHUNKS = 8


def load_audio(path, target_sr=SAMPLE_RATE):
    try:
        import soundfile as sf
        audio, sr = sf.read(path, always_2d=False, dtype="float32")
    except Exception:
        import librosa
        audio, sr = librosa.load(path, sr=None, mono=True)
    if audio.ndim == 2:
        audio = audio.mean(axis=-1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def prepare_audio(audio, processor):
    fe = processor.feature_extractor
    target_sr = fe.sampling_rate
    chunks = []
    for i in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[i:i + CHUNK_SIZE]
        if len(chunk) < target_sr:
            chunk = np.pad(chunk, (0, target_sr - len(chunk)), "constant")
        chunks.append(chunk)
    chunks = chunks[:MAX_CHUNKS]
    out = fe(chunks, sampling_rate=target_sr, return_attention_mask=True,
             padding="max_length", return_tensors="pt", do_normalize=True)
    n_speech = len(chunks) * SPEECH_TOKENS_PER_CHUNK
    return out.input_features, out.attention_mask, n_speech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_path")
    ap.add_argument("--repo",
                    default="YOUR_HF_USERNAME/MERaLiON-2-3B-EAGLE-W4A16",
                    help="HF repo id (or local path) of this package")
    ap.add_argument("--base", default=None,
                    help="(optional) Override BF16 base path. By default we "
                         "use the bundled base_bf16/ inside --repo.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--K",      type=int, default=4)
    ap.add_argument("--gptq_kernel", default="exllama",
                    choices=["exllama", "exllamav2", "marlin"])
    ap.add_argument("--instruction", default="Transcribe the speech")
    ap.add_argument("--compared_model", default=None,
                    help="Optional: path to a full BF16 MERaLiON-2-3B dir "
                         "to run a side-by-side baseline (no EAGLE, no "
                         "quantization) on the same audio sample.")
    args = ap.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"audio not found: {args.audio_path}"); sys.exit(1)

    model = MERaLiON2EAGLEForASR.from_pretrained(
        args.repo, base_model=args.base,
        torch_dtype=torch.float16,
        gptq_kernel=args.gptq_kernel, device=args.device)
    processor = model.processor
    tokenizer = processor.tokenizer

    print(f"Loading audio: {args.audio_path}")
    audio = load_audio(args.audio_path)
    input_features, feature_attention_mask, n_speech = prepare_audio(audio, processor)
    input_features = input_features.to(args.device).to(torch.float16)
    feature_attention_mask = feature_attention_mask.to(args.device)

    speech_token_id = model.config.speech_token_index
    conv = [{"role": "user",
             "content": (f"Instruction: {args.instruction} \n"
                         "Follow the text instruction based on the "
                         "following audio: <SpeechHere>")}]
    prompt = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    pos = raw_ids.index(speech_token_id)
    input_ids = torch.tensor(
        [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]],
        dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_ids)

    # Warmup pass — first call eats CUDA kernel JIT + allocator init time;
    # don't include it in the timing.  Same audio / prompt, short max_new.
    print("Warmup …")
    _ = model.generate_eagle(
        input_ids=input_ids, attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        max_new_tokens=16, K=args.K,
    )
    torch.cuda.synchronize()

    print(f"Transcribing with EAGLE+W4A16 (K={args.K}, kernel={args.gptq_kernel}) …")
    out_ids, stats = model.generate_eagle(
        input_ids=input_ids, attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        max_new_tokens=args.max_new_tokens, K=args.K,
        return_stats=True,
    )
    eagle_prefill_dt = stats["prefill_dt"]
    eagle_decode_dt  = stats["decode_dt"]
    n_gen            = stats["n_generated"]
    eagle_decode_tps = n_gen / eagle_decode_dt
    eagle_acc_rate   = (stats["n_spec_acc"] / max(1, stats["n_spec_tot"])
                        if stats["n_spec_tot"] else 0.0)
    dt = eagle_prefill_dt + eagle_decode_dt
    n_gen = out_ids.shape[1] - input_ids.shape[1]

    text = tokenizer.decode(out_ids[0, input_ids.shape[1]:],
                             skip_special_tokens=True)
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
    print(f"\n  prefill: {eagle_prefill_dt*1000:6.0f} ms  "
          f"(audio encoder + first verifier forward, fixed cost)")
    print(f"  decode:  {eagle_decode_dt:.2f}s for {n_gen} tokens  "
          f"= {eagle_decode_tps:.1f} tok/s   "
          f"(EAGLE accept rate {eagle_acc_rate:.1%})")
    print(f"  total:   {dt:.2f}s")
    print(f"\nTranscription (EAGLE + W4A16):\n  {text}")
    eagle_tps = n_gen / dt
    eagle_dt  = dt

    # ── Optional: BF16 baseline comparison on the same audio ──────────────────
    if args.compared_model:
        print(f"\n[baseline] Loading full BF16 model from {args.compared_model} …")
        # Free EAGLE+W4A16 GPU mem first
        del model
        torch.cuda.empty_cache()

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
        from transformers.cache_utils import HybridCache

        bf16 = MERaLiON2ForConditionalGeneration.from_pretrained(
            args.compared_model, torch_dtype=torch.bfloat16, use_safetensors=True)
        bf16 = bf16.to(args.device)
        bf16.eval()

        # Re-cast input_features to bf16 for this run
        input_features_b = input_features.to(torch.bfloat16)

        eos_ids = {tokenizer.eos_token_id,
                   tokenizer.convert_tokens_to_ids("<end_of_turn>")}
        eos_ids.discard(None)

        def _greedy(max_new):
            seq_len   = input_ids.shape[1]
            max_cache = seq_len + max_new
            kv = HybridCache(
                bf16.text_decoder.model.config, max_batch_size=1,
                max_cache_len=max_cache, dtype=torch.bfloat16, device=args.device)
            with torch.inference_mode():
                torch.cuda.synchronize(); t_prefill = time.time()
                out = bf16(
                    input_ids=input_ids, attention_mask=attention_mask,
                    input_features=input_features_b,
                    feature_attention_mask=feature_attention_mask,
                    past_key_values=kv, use_cache=True,
                    cache_position=torch.arange(0, seq_len, device=args.device),
                    return_dict=True)
                torch.cuda.synchronize()
                pre_dt = time.time() - t_prefill
                next_tok = int(out.logits[0, -1].argmax())
                gen = [next_tok]
                cur = seq_len
                t_dec = time.time()
                while len(gen) < max_new and next_tok not in eos_ids:
                    o = bf16.text_decoder(
                        input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=args.device),
                        attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=args.device),
                        past_key_values=kv, use_cache=True,
                        cache_position=torch.tensor([cur], device=args.device),
                        return_dict=True)
                    next_tok = int(o.logits[0, -1].argmax())
                    gen.append(next_tok); cur += 1
                torch.cuda.synchronize()
                dec_dt = time.time() - t_dec
            return gen, pre_dt, dec_dt

        # warmup
        print("[baseline] warmup …")
        _ = _greedy(16)
        torch.cuda.synchronize()

        print(f"[baseline] decoding (greedy, no spec) …")
        gen, bf16_prefill_dt, bf16_decode_dt = _greedy(args.max_new_tokens)
        bf16_decode_tps = len(gen) / bf16_decode_dt
        bf16_dt  = bf16_prefill_dt + bf16_decode_dt
        bf16_tps = len(gen) / bf16_dt

        bf16_text = tokenizer.decode(gen, skip_special_tokens=True)
        bf16_text = bf16_text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
        print(f"\n  prefill: {bf16_prefill_dt*1000:6.0f} ms")
        print(f"  decode:  {bf16_decode_dt:.2f}s for {len(gen)} tokens  "
              f"= {bf16_decode_tps:.1f} tok/s")
        print(f"  total:   {bf16_dt:.2f}s")
        print(f"\nTranscription (BF16 baseline):\n  {bf16_text}")

        print("\n" + "=" * 70)
        print(f"{'config':<18} {'prefill(ms)':>12} {'decode tok/s':>14} "
              f"{'total(s)':>10} {'speedup':>8}")
        print(f"{'BF16 baseline':<18} {bf16_prefill_dt*1000:>12.0f} "
              f"{bf16_decode_tps:>14.1f} {bf16_dt:>10.2f} {'1.00x':>8}")
        print(f"{'EAGLE + W4A16':<18} {eagle_prefill_dt*1000:>12.0f} "
              f"{eagle_decode_tps:>14.1f} {eagle_dt:>10.2f} "
              f"{eagle_decode_tps/bf16_decode_tps:>7.2f}x")
        print("=" * 70)
        print(f"  decode-only speedup (excludes shared prefill cost): "
              f"{eagle_decode_tps/bf16_decode_tps:.2f}×")


if __name__ == "__main__":
    main()
