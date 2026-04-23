"""3-way head-to-head comparison on the same IMDA samples:
  1) bf16 baseline MERaLiON-2-3B
  2) W4A16 compressed-tensors (Marlin INT4 path when supported)
  3) bf16 + Medusa adapter (K=4 heads)

Designed to run on A100 80GB where all three models fit resident
simultaneously — on 40 GB cards or L40 the script will OOM.

Usage (A100):
    python bench_3way_a100.py \\
        --base_bf16     /path/to/MERaLiON-2-3B \\
        --base_w4a16    /path/to/MERaLiON-2-3B-W4A16-RTN-textonly \\
        --medusa_source /path/to/hf_medusa_pkg  (or HF repo id) \\
        --dataset       /path/to/IMDA_PART1 \\
        --num_samples 20
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch


SAMPLE_RATE = 16000
CHUNK_SIZE = SAMPLE_RATE * 30
SPEECH_TOKENS_PER_CHUNK = 100
MAX_CHUNKS = 8


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


# ── loaders ────────────────────────────────────────────────────────────────────

def load_meralion(path, device, dtype=torch.bfloat16):
    """Load MERaLiON-2-3B (bf16 or compressed-tensors W4A16) via bundled
    patched meralion2_bl code.  Auto-detects quantization_config in
    config.json and routes through HfQuantizer (Marlin/CompressedLinear)."""
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    model = MERaLiON2ForConditionalGeneration.from_pretrained(
        path, torch_dtype=dtype).to(device).eval()
    return model


def load_medusa(source, device):
    """Load the Medusa adapter (local path or HF Hub repo id).  Internally
    pulls the base MERaLiON-2-3B via snapshot_download if needed."""
    if os.path.isdir(source):
        sys.path.insert(0, os.path.abspath(source))
        from modeling_medusa import MERaLiON2MedusaForASR
        return MERaLiON2MedusaForASR.from_pretrained(
            source, torch_dtype=torch.bfloat16).to(device)
    else:
        from huggingface_hub import snapshot_download
        local = snapshot_download(source)
        sys.path.insert(0, local)
        from modeling_medusa import MERaLiON2MedusaForASR
        return MERaLiON2MedusaForASR.from_pretrained(
            local, torch_dtype=torch.bfloat16).to(device)


# ── inference wrappers ─────────────────────────────────────────────────────────

def _build_prompt(processor, model, n_speech, device):
    tokenizer = processor.tokenizer
    speech_token_id = model.config.speech_token_index
    conv = [{"role": "user",
             "content": ("Instruction: Transcribe the speech \n"
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
    return input_ids, attention_mask


def greedy_decode(model, processor, audio, sr, max_new_tokens, device, quant_dtype):
    from transformers.cache_utils import HybridCache
    tokenizer = processor.tokenizer
    input_features, feat_mask, n_speech = prepare_audio(audio, sr, processor)
    input_features = input_features.to(device).to(quant_dtype)
    feat_mask = feat_mask.to(device)
    input_ids, attention_mask = _build_prompt(processor, model, n_speech, device)

    seq_len = input_ids.shape[1]
    max_cache = seq_len + max_new_tokens
    past_kv = HybridCache(
        model.text_decoder.model.config, max_batch_size=1,
        max_cache_len=max_cache, dtype=quant_dtype, device=device)
    eos = {tokenizer.eos_token_id,
           tokenizer.convert_tokens_to_ids("<end_of_turn>")} - {None}

    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    input_features=input_features, feature_attention_mask=feat_mask,
                    past_key_values=past_kv, use_cache=True,
                    cache_position=torch.arange(0, seq_len, device=device),
                    return_dict=True)
        next_tok = int(out.logits[0, -1].argmax()); gen=[next_tok]
        torch.cuda.synchronize(); t1 = time.time()
        cur_pos = seq_len
        while len(gen) < max_new_tokens and next_tok not in eos:
            v = model.text_decoder(
                input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=device),
                attention_mask=torch.ones(1, cur_pos+1, dtype=torch.long, device=device),
                past_key_values=past_kv, use_cache=True,
                cache_position=torch.tensor([cur_pos], device=device),
                return_dict=True)
            next_tok = int(v.logits[0, -1].argmax()); gen.append(next_tok); cur_pos += 1
    torch.cuda.synchronize(); t2 = time.time()

    text = tokenizer.decode(gen, skip_special_tokens=True)
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
    return text, {
        "n_tokens": len(gen),
        "prefill_s": t1 - t0, "decode_s": t2 - t1,
        "total_s": t2 - t0,
        "tps": max(len(gen) - 1, 1) / max(t2 - t1, 1e-6),
    }


def medusa_decode(medusa_model, processor, audio, sr, max_new_tokens, device):
    tokenizer = processor.tokenizer
    dtype = next(p.dtype for p in medusa_model.base.parameters()
                 if p.dtype in (torch.float16, torch.bfloat16))
    input_features, feat_mask, n_speech = prepare_audio(audio, sr, processor)
    input_features = input_features.to(device).to(dtype)
    feat_mask = feat_mask.to(device)
    input_ids, attention_mask = _build_prompt(processor, medusa_model, n_speech, device)

    torch.cuda.synchronize(); t0 = time.time()
    out_ids = medusa_model.generate_medusa(
        input_ids=input_ids, attention_mask=attention_mask,
        input_features=input_features, feature_attention_mask=feat_mask,
        max_new_tokens=max_new_tokens,
        eos_token_ids=[tokenizer.eos_token_id,
                       tokenizer.convert_tokens_to_ids("<end_of_turn>")])
    torch.cuda.synchronize(); t1 = time.time()

    gen = out_ids[0, input_ids.shape[1]:].tolist()
    text = tokenizer.decode(gen, skip_special_tokens=True)
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
    return text, {
        "n_tokens": len(gen),
        "total_s": t1 - t0, "decode_s": t1 - t0,
        "tps": max(len(gen) - 1, 1) / max(t1 - t0, 1e-6),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_bf16",     required=True, help="Path to MERaLiON-2-3B (bf16)")
    ap.add_argument("--base_w4a16",    required=True, help="Path to W4A16 compressed-tensors checkpoint")
    ap.add_argument("--medusa_source", required=True, help="Medusa adapter (local dir or HF repo id)")
    ap.add_argument("--dataset",       required=True)
    ap.add_argument("--num_samples",   type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--device",        default="cuda")
    ap.add_argument("--seed_shuffle",  type=int, default=42)
    ap.add_argument("--start_idx",     type=int, default=10500)
    ap.add_argument("--output",        default="bench_3way.json")
    ap.add_argument("--sequential",    action="store_true",
                    help="Load models sequentially (frees VRAM between) — use on <40 GB GPUs")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required"); sys.exit(1)
    device = torch.device(args.device)

    # Ensure bundled patched MERaLiON code wins over any system install
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path: sys.path.insert(0, here)

    # Load dataset + processor first (processor reused across models)
    from datasets import load_from_disk
    from transformers import AutoProcessor
    raw = load_from_disk(os.path.abspath(args.dataset))
    shuffled = raw.shuffle(seed=args.seed_shuffle)
    end = min(args.start_idx + args.num_samples, len(shuffled))
    subset = shuffled.select(range(args.start_idx, end))
    processor = AutoProcessor.from_pretrained(args.base_bf16, trust_remote_code=True)
    print(f"Dataset: {args.dataset}  |  {len(subset)} samples")

    def _vram():
        return torch.cuda.max_memory_allocated(device) / 1e9

    results = {
        "bf16":    {"samples": []},
        "w4a16":   {"samples": []},
        "medusa":  {"samples": []},
    }

    # ── BF16 baseline ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60); print("[1/3] bf16 baseline"); print("=" * 60)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    m_bf16 = load_meralion(args.base_bf16, device, dtype=torch.bfloat16)
    results["bf16"]["load_s"] = time.time() - t0
    results["bf16"]["vram_gb"] = _vram()
    print(f"  loaded in {results['bf16']['load_s']:.1f}s  VRAM={results['bf16']['vram_gb']:.2f}GB")

    # Warmup
    s = subset[0]; ao = s["context"]["audio"]
    a0 = np.asarray(ao["array"], dtype=np.float32); sr0 = ao.get("sampling_rate", SAMPLE_RATE)
    _ = greedy_decode(m_bf16, processor, a0, sr0, 32, device, torch.bfloat16)

    for i, s in enumerate(subset):
        ao = s["context"]["audio"]
        a = np.asarray(ao["array"], dtype=np.float32)
        sr = ao.get("sampling_rate", SAMPLE_RATE)
        hyp, st = greedy_decode(m_bf16, processor, a, sr, args.max_new_tokens, device, torch.bfloat16)
        ref = (s.get("other_attributes") or {}).get("Transcription") or \
              (s.get("other_attributes") or {}).get("transcription") or ""
        results["bf16"]["samples"].append({"idx": i, "hyp": hyp, "ref": ref, **st})
        print(f"  [{i:3d}]  {st['total_s']:5.2f}s  {st['tps']:6.1f} tok/s | {hyp[:60]!r}")

    if args.sequential:
        del m_bf16; torch.cuda.empty_cache()

    # ── W4A16 compressed-tensors ─────────────────────────────────────────────
    print("\n" + "=" * 60); print("[2/3] W4A16 compressed-tensors"); print("=" * 60)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    m_w4a16 = load_meralion(args.base_w4a16, device, dtype=torch.bfloat16)
    results["w4a16"]["load_s"] = time.time() - t0
    results["w4a16"]["vram_gb"] = _vram()
    # Check linear class (is Marlin actually used?)
    l0 = m_w4a16.text_decoder.model.layers[0].self_attn.q_proj
    results["w4a16"]["linear_class"] = type(l0).__name__
    print(f"  loaded in {results['w4a16']['load_s']:.1f}s  VRAM={results['w4a16']['vram_gb']:.2f}GB")
    print(f"  layer 0 q_proj class: {results['w4a16']['linear_class']}")

    _ = greedy_decode(m_w4a16, processor, a0, sr0, 32, device, torch.bfloat16)
    for i, s in enumerate(subset):
        ao = s["context"]["audio"]
        a = np.asarray(ao["array"], dtype=np.float32)
        sr = ao.get("sampling_rate", SAMPLE_RATE)
        hyp, st = greedy_decode(m_w4a16, processor, a, sr, args.max_new_tokens, device, torch.bfloat16)
        ref = (s.get("other_attributes") or {}).get("Transcription") or \
              (s.get("other_attributes") or {}).get("transcription") or ""
        results["w4a16"]["samples"].append({"idx": i, "hyp": hyp, "ref": ref, **st})
        print(f"  [{i:3d}]  {st['total_s']:5.2f}s  {st['tps']:6.1f} tok/s | {hyp[:60]!r}")

    if args.sequential:
        del m_w4a16; torch.cuda.empty_cache()

    # ── BF16 + Medusa ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60); print("[3/3] bf16 + Medusa"); print("=" * 60)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    m_medusa = load_medusa(args.medusa_source, device)
    results["medusa"]["load_s"] = time.time() - t0
    results["medusa"]["vram_gb"] = _vram()
    print(f"  loaded in {results['medusa']['load_s']:.1f}s  VRAM={results['medusa']['vram_gb']:.2f}GB")

    _ = medusa_decode(m_medusa, processor, a0, sr0, 32, device)
    for i, s in enumerate(subset):
        ao = s["context"]["audio"]
        a = np.asarray(ao["array"], dtype=np.float32)
        sr = ao.get("sampling_rate", SAMPLE_RATE)
        hyp, st = medusa_decode(m_medusa, processor, a, sr, args.max_new_tokens, device)
        ref = (s.get("other_attributes") or {}).get("Transcription") or \
              (s.get("other_attributes") or {}).get("transcription") or ""
        results["medusa"]["samples"].append({"idx": i, "hyp": hyp, "ref": ref, **st})
        print(f"  [{i:3d}]  {st['total_s']:5.2f}s  {st['tps']:6.1f} tok/s | {hyp[:60]!r}")

    # ── Summary ───────────────────────────────────────────────────────────────
    try:
        import jiwer
        have_jiwer = True
    except ImportError:
        have_jiwer = False

    def _stats(key):
        samples = results[key]["samples"]
        avg_lat = float(np.mean([s["total_s"] for s in samples]))
        avg_tps = float(np.mean([s["tps"] for s in samples]))
        wer = None
        if have_jiwer:
            refs = [_normalize_text(s["ref"]) for s in samples]
            hyps = [_normalize_text(s["hyp"]) for s in samples]
            wer = float(jiwer.wer(refs, hyps))
        return avg_lat, avg_tps, wer

    lat_bf16,   tps_bf16,   wer_bf16   = _stats("bf16")
    lat_w4a16,  tps_w4a16,  wer_w4a16  = _stats("w4a16")
    lat_medusa, tps_medusa, wer_medusa = _stats("medusa")

    print("\n" + "=" * 84)
    print(f"{'Config':<22} {'Lat (s)':>8} {'TPS':>7} {'WER%':>6} "
          f"{'VRAM (GB)':>10} {'Lat speedup':>12} {'TPS speedup':>12}")
    print("-" * 84)
    for key, label, lat, tps, wer in [
        ("bf16",   "bf16 baseline",         lat_bf16,   tps_bf16,   wer_bf16),
        ("w4a16",  "W4A16 (compressed)",    lat_w4a16,  tps_w4a16,  wer_w4a16),
        ("medusa", "bf16 + Medusa K=4",     lat_medusa, tps_medusa, wer_medusa),
    ]:
        lat_sp = lat_bf16 / max(lat, 1e-6)
        tps_sp = tps / max(tps_bf16, 1e-6)
        wer_s  = f"{wer*100:.2f}" if wer is not None else "   —"
        print(f"{label:<22} {lat:>8.3f} {tps:>7.2f} {wer_s:>6} "
              f"{results[key]['vram_gb']:>10.2f} {lat_sp:>11.2f}x {tps_sp:>11.2f}x")
    print("=" * 84)

    with open(args.output, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "num_samples": len(subset),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "results": results,
            "summary": {
                "bf16":   {"lat_s": lat_bf16,   "tps": tps_bf16,   "wer": wer_bf16,
                           "vram_gb": results["bf16"]["vram_gb"]},
                "w4a16":  {"lat_s": lat_w4a16,  "tps": tps_w4a16,  "wer": wer_w4a16,
                           "vram_gb": results["w4a16"]["vram_gb"],
                           "linear_class": results["w4a16"]["linear_class"]},
                "medusa": {"lat_s": lat_medusa, "tps": tps_medusa, "wer": wer_medusa,
                           "vram_gb": results["medusa"]["vram_gb"]},
            },
        }, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
