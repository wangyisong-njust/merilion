"""Head-to-head TPS comparison: baseline MERaLiON-2-3B vs Medusa-adapter model.

Runs both models over the same IMDA audio samples, same greedy decoding,
reports latency / throughput / WER / accept-rate side-by-side.

--medusa_source controls where the Medusa adapter comes from:
  * Local path (e.g. `./hf_medusa_pkg`) — useful for iterating before upload.
  * HF Hub repo id (e.g. `YOUR_USER/MERaLiON-2-3B-Medusa`) — pulls the
    adapter weights from HF Hub.  Base model is always pulled from
    whatever `base_model_name_or_path` the adapter_config.json declares
    (default `MERaLiON/MERaLiON-2-3B`).

--base_source selects the baseline model.  Can be:
  * HF Hub repo id (e.g. `MERaLiON/MERaLiON-2-3B`) — triggers a real
    fresh download into the HF cache.
  * Local path — use a pre-downloaded copy (faster iteration).

Usage:
    python bench_medusa_vs_baseline.py \\
        --base_source   MERaLiON/MERaLiON-2-3B \\
        --medusa_source YOUR_USER/MERaLiON-2-3B-Medusa \\
        --dataset /home/.../IMDA_PART1_mono_en_30_ASR \\
        --num_samples 20
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch


SAMPLE_RATE = 16000
CHUNK_SIZE = SAMPLE_RATE * 30
SPEECH_TOKENS_PER_CHUNK = 100
MAX_CHUNKS = 8


# ── audio prep (matches infer_gpu.py prepare_audio) ────────────────────────────

def prepare_audio(audio, sample_rate, processor):
    import librosa
    fe = processor.feature_extractor
    target_sr = fe.sampling_rate
    if sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    chunks = []
    for i in range(0, len(audio), CHUNK_SIZE):
        c = audio[i:i + CHUNK_SIZE]
        if len(c) < target_sr:
            c = np.pad(c, (0, target_sr - len(c)), "constant")
        chunks.append(c)
    chunks = chunks[:MAX_CHUNKS]
    out = fe(chunks, sampling_rate=target_sr, return_attention_mask=True,
             padding="max_length", return_tensors="pt", do_normalize=True)
    return out.input_features, out.attention_mask, len(chunks) * SPEECH_TOKENS_PER_CHUNK


def _normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# ── baseline loader (from bundled MERaLiON-2 code) ─────────────────────────────

def load_baseline(base_source, device, dtype=torch.bfloat16):
    """Load MERaLiON-2-3B as the baseline.  Uses the bundled patched code
    (`hf_medusa_pkg/meralion2_bl/`) so upstream `trust_remote_code`
    incompatibility doesn't bite."""
    from huggingface_hub import snapshot_download
    if os.path.isdir(base_source):
        base_local_dir = base_source
    else:
        print(f"  [baseline] fetching weights from HF Hub: {base_source}")
        base_local_dir = snapshot_download(
            base_source,
            allow_patterns=[
                "config.json", "generation_config.json",
                "*.safetensors", "*.safetensors.index.json",
                "tokenizer*", "special_tokens_map.json",
                "preprocessor_config.json", "processor_config.json",
                "chat_template.jinja",
                "*.py",   # processing_meralion2.py, configuration_*.py (needed
                          # by AutoProcessor's trust_remote_code path)
            ],
        )
    # Use our bundled MERaLiON code (works on current transformers)
    _here = Path(__file__).resolve().parent
    pkg_bl = _here / "hf_medusa_pkg"
    sys.path.insert(0, str(pkg_bl))
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from meralion2_bl.configuration_meralion2 import MERaLiON2Config

    base = MERaLiON2ForConditionalGeneration.from_pretrained(
        base_local_dir,
        config=MERaLiON2Config.from_pretrained(base_local_dir),
        torch_dtype=dtype,
    ).to(device).eval()

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        base_local_dir, trust_remote_code=True)
    return base, processor


# ── inference: baseline greedy (manual loop for fair timing) ───────────────────

def transcribe_baseline(model, processor, audio, sr, instruction,
                        max_new_tokens, device):
    from transformers.cache_utils import HybridCache

    tokenizer = processor.tokenizer
    dtype = next(p.dtype for p in model.parameters()
                 if p.dtype in (torch.float16, torch.bfloat16))

    input_features, feat_attn_mask, n_speech = prepare_audio(audio, sr, processor)
    speech_token_id = model.config.speech_token_index
    conv = [{"role": "user",
             "content": (f"Instruction: {instruction} \n"
                         "Follow the text instruction based on the "
                         "following audio: <SpeechHere>")}]
    prompt = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    pos = raw_ids.index(speech_token_id)
    input_ids = torch.tensor(
        [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]],
        dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    input_features = input_features.to(device).to(dtype)
    feat_attn_mask = feat_attn_mask.to(device)

    seq_len = input_ids.shape[1]
    max_cache = seq_len + max_new_tokens
    past_kv = HybridCache(
        model.text_decoder.model.config, max_batch_size=1,
        max_cache_len=max_cache, dtype=dtype, device=device)

    eos_ids = {tokenizer.eos_token_id,
               tokenizer.convert_tokens_to_ids("<end_of_turn>")}
    eos_ids.discard(None)

    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feat_attn_mask,
            past_key_values=past_kv,
            use_cache=True,
            cache_position=torch.arange(0, seq_len, device=device),
            return_dict=True,
        )
        next_tok = int(out.logits[0, -1].argmax())
        gen = [next_tok]
        torch.cuda.synchronize(); t1 = time.time()

        cur_pos = seq_len
        while len(gen) < max_new_tokens and next_tok not in eos_ids:
            v = model.text_decoder(
                input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=device),
                attention_mask=torch.ones(1, cur_pos + 1, dtype=torch.long, device=device),
                past_key_values=past_kv,
                use_cache=True,
                cache_position=torch.tensor([cur_pos], device=device),
                return_dict=True,
            )
            next_tok = int(v.logits[0, -1].argmax())
            gen.append(next_tok)
            cur_pos += 1
    torch.cuda.synchronize(); t2 = time.time()

    text = tokenizer.decode(gen, skip_special_tokens=True)
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
    return text, {
        "n_tokens":  len(gen),
        "latency_s": t2 - t0,
        "decode_s":  t2 - t1,
        "decode_tps": max(len(gen) - 1, 1) / max(t2 - t1, 1e-6),
    }


# ── inference: medusa ──────────────────────────────────────────────────────────

def transcribe_medusa(medusa_model, processor, audio, sr, instruction,
                      max_new_tokens, device):
    tokenizer = processor.tokenizer
    dtype = next(p.dtype for p in medusa_model.base.parameters()
                 if p.dtype in (torch.float16, torch.bfloat16))

    input_features, feat_attn_mask, n_speech = prepare_audio(audio, sr, processor)
    speech_token_id = medusa_model.config.speech_token_index
    conv = [{"role": "user",
             "content": (f"Instruction: {instruction} \n"
                         "Follow the text instruction based on the "
                         "following audio: <SpeechHere>")}]
    prompt = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    pos = raw_ids.index(speech_token_id)
    input_ids = torch.tensor(
        [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]],
        dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    input_features = input_features.to(device).to(dtype)
    feat_attn_mask = feat_attn_mask.to(device)

    torch.cuda.synchronize(); t0 = time.time()
    out_ids = medusa_model.generate_medusa(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feat_attn_mask,
        max_new_tokens=max_new_tokens,
        eos_token_ids=[tokenizer.eos_token_id,
                       tokenizer.convert_tokens_to_ids("<end_of_turn>")],
    )
    torch.cuda.synchronize(); t1 = time.time()

    gen = out_ids[0, input_ids.shape[1]:].tolist()
    text = tokenizer.decode(gen, skip_special_tokens=True)
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
    return text, {
        "n_tokens":  len(gen),
        "latency_s": t1 - t0,
        "decode_tps": max(len(gen) - 1, 1) / max(t1 - t0, 1e-6),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_source",   required=True,
                    help="HF repo id (triggers download) or local path for baseline model")
    ap.add_argument("--medusa_source", required=True,
                    help="HF repo id or local path for Medusa adapter")
    ap.add_argument("--dataset",       required=True)
    ap.add_argument("--num_samples",   type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--device",        default="cuda")
    ap.add_argument("--seed_shuffle",  type=int, default=42)
    ap.add_argument("--start_idx",     type=int, default=10500,
                    help="Start index into shuffled dataset (match infer_gpu.py)")
    ap.add_argument("--output",        default="bench_medusa_vs_baseline.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required"); sys.exit(1)
    device = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(device)

    # ── Load baseline ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading baseline …")
    print("=" * 60)
    t0 = time.time()
    base_model, processor = load_baseline(args.base_source, device)
    vram_after_base = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  baseline loaded in {time.time()-t0:.1f}s  VRAM={vram_after_base:.2f} GB")

    # ── Load Medusa (after freeing baseline) ──────────────────────────────────
    # Keep baseline resident — we compare on same samples, one run each.

    print("\n" + "=" * 60)
    print("Loading Medusa …")
    print("=" * 60)
    # Make sure our bundled meralion2_bl code is importable even when
    # medusa_source is an HF repo (modeling_medusa.py inside the downloaded
    # snapshot will import from its own sibling meralion2_bl/).
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here / "hf_medusa_pkg"))
    if os.path.isdir(args.medusa_source):
        sys.path.insert(0, os.path.abspath(args.medusa_source))
        from modeling_medusa import MERaLiON2MedusaForASR
    else:
        from huggingface_hub import snapshot_download
        print(f"  fetching Medusa adapter from HF Hub: {args.medusa_source}")
        local_dir = snapshot_download(args.medusa_source)
        sys.path.insert(0, local_dir)
        from modeling_medusa import MERaLiON2MedusaForASR

    t0 = time.time()
    medusa_model = MERaLiON2MedusaForASR.from_pretrained(
        args.medusa_source if os.path.isdir(args.medusa_source) else local_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)
    vram_after_medusa = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  Medusa loaded in {time.time()-t0:.1f}s  VRAM(peak)={vram_after_medusa:.2f} GB")

    # ── Data ──────────────────────────────────────────────────────────────────
    from datasets import load_from_disk
    raw = load_from_disk(os.path.abspath(args.dataset))
    shuffled = raw.shuffle(seed=args.seed_shuffle)
    end = min(args.start_idx + args.num_samples, len(shuffled))
    subset = shuffled.select(range(args.start_idx, end))
    print(f"\nDataset: {args.dataset}  |  samples [{args.start_idx}, {end})  "
          f"= {len(subset)}")

    # ── Warm up both ──────────────────────────────────────────────────────────
    print("Warming up …")
    s = subset[0]
    ao = s["context"]["audio"]
    a = np.asarray(ao["array"], dtype=np.float32)
    sr = ao.get("sampling_rate", SAMPLE_RATE)
    _ = transcribe_baseline(base_model, processor, a, sr,
                            "Transcribe the speech", 32, device)
    _ = transcribe_medusa(medusa_model, processor, a, sr,
                          "Transcribe the speech", 32, device)

    # ── Run both over same samples ─────────────────────────────────────────────
    results = []
    print("\nRunning baseline vs Medusa on each sample:")
    print(f"  {'#':>3}  {'lat_base':>9}  {'lat_med':>8}  {'tps_base':>8}  {'tps_med':>8}  "
          f"{'speedup':>7}  match?")
    print("  " + "-" * 80)
    for i in range(len(subset)):
        s = subset[i]
        ao = s["context"]["audio"]
        a = np.asarray(ao["array"], dtype=np.float32)
        sr = ao.get("sampling_rate", SAMPLE_RATE)
        ref = (s.get("other_attributes") or {}).get("Transcription") or \
              (s.get("other_attributes") or {}).get("transcription") or ""

        hyp_b, st_b = transcribe_baseline(
            base_model, processor, a, sr,
            "Transcribe the speech", args.max_new_tokens, device)
        hyp_m, st_m = transcribe_medusa(
            medusa_model, processor, a, sr,
            "Transcribe the speech", args.max_new_tokens, device)

        speedup = st_b["latency_s"] / max(st_m["latency_s"], 1e-6)
        match   = (hyp_b == hyp_m)
        results.append({
            "idx": i, "ref": ref, "hyp_base": hyp_b, "hyp_medusa": hyp_m,
            "baseline": st_b, "medusa": st_m, "match": match,
            "speedup": speedup,
        })
        print(f"  {i:3d}  {st_b['latency_s']:8.2f}s  {st_m['latency_s']:7.2f}s  "
              f"{st_b['decode_tps']:7.1f}  {st_m['decode_tps']:7.1f}  "
              f"{speedup:6.2f}x  {'✓' if match else '✗'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_lat_b   = np.mean([r["baseline"]["latency_s"] for r in results])
    avg_lat_m   = np.mean([r["medusa"]["latency_s"] for r in results])
    avg_tps_b   = np.mean([r["baseline"]["decode_tps"] for r in results])
    avg_tps_m   = np.mean([r["medusa"]["decode_tps"] for r in results])
    n_match     = sum(r["match"] for r in results)

    # WER
    try:
        import jiwer
        refs = [_normalize_text(r["ref"]) for r in results]
        hyp_b = [_normalize_text(r["hyp_base"]) for r in results]
        hyp_m = [_normalize_text(r["hyp_medusa"]) for r in results]
        wer_b = jiwer.wer(refs, hyp_b)
        wer_m = jiwer.wer(refs, hyp_m)
    except ImportError:
        wer_b = wer_m = float("nan")

    print("\n" + "=" * 60)
    print(f"{'Metric':<28} {'Baseline':>15} {'Medusa':>15}")
    print("-" * 60)
    print(f"{'Avg latency (s/sample)':<28} {avg_lat_b:>15.3f} {avg_lat_m:>15.3f}")
    print(f"{'Avg decode TPS':<28} {avg_tps_b:>15.2f} {avg_tps_m:>15.2f}")
    print(f"{'WER (normalized)':<28} {wer_b:>15.4f} {wer_m:>15.4f}")
    print(f"{'Latency speedup':<28} {'1.00x':>15} {avg_lat_b/max(avg_lat_m,1e-6):>14.2f}x")
    print(f"{'Throughput speedup':<28} {'1.00x':>15} {avg_tps_m/max(avg_tps_b,1e-6):>14.2f}x")
    print(f"{'Output match rate':<28} {n_match}/{len(results)}")
    print("=" * 60)

    summary = {
        "base_source":   args.base_source,
        "medusa_source": args.medusa_source,
        "dataset":       args.dataset,
        "num_samples":   len(results),
        "avg_latency_base_s":   float(avg_lat_b),
        "avg_latency_medusa_s": float(avg_lat_m),
        "avg_tps_base":         float(avg_tps_b),
        "avg_tps_medusa":       float(avg_tps_m),
        "wer_base":             float(wer_b),
        "wer_medusa":           float(wer_m),
        "latency_speedup":      float(avg_lat_b / max(avg_lat_m, 1e-6)),
        "throughput_speedup":   float(avg_tps_m / max(avg_tps_b, 1e-6)),
        "n_match":              int(n_match),
        "results":              results,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
