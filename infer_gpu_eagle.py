"""EAGLE speculative-decoding inference for MERaLiON-2-3B.

Each round:
  1. verifier(input) -> (logits_last, h_last)
  2. next_tok = argmax(logits_last)
  3. EAGLE proposes K drafts AUTO-REGRESSIVELY:
        for k = 0..K-1:
            fused = [embed(last_tok), last_h]
            new_h, new_logits = eagle_layer(fused)
            draft_k = argmax(new_logits)
            last_tok, last_h = draft_k, new_h
  4. verifier([next_tok, d_0, .., d_{K-1}]) -> K+1 logits (one batched forward)
  5. greedy linear accept (chain: first mismatch stops)
  6. trim verifier KV cache to accepted length

EAGLE's recurrence means accuracy at depth k depends on depth-(k-1)'s
h rather than a fresh projection of h_last, so later-depth acc degrades
more gracefully than Medusa and K=4..6 can sustain higher acc rates.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_gpu import (
    load_model_gpu, prepare_audio, _normalize_text, _normalize_text_audiobench,
    SAMPLE_RATE,
)
from eagle_model import EAGLE, attach_eagle


def load_eagle(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    eagle, rotary = attach_eagle(model, device, dtype=torch.bfloat16)
    eagle.load_trainable_state_dict(ckpt["eagle_state"])
    eagle.eval()
    for p in eagle.parameters():
        p.requires_grad_(False)
    print(f"Loaded EAGLE: {sum(p.numel() for p in eagle.parameters())/1e6:.1f} M params "
          f"(step {ckpt.get('step', '?')}, "
          f"val_acc {ckpt.get('val_acc', '?')})")
    return eagle, rotary


def _extract_audio(sample):
    c = sample.get("context")
    if isinstance(c, dict):
        if "array" in c:
            return c
        if isinstance(c.get("audio"), dict):
            return c["audio"]
    return None


def _extract_ref(sample):
    oa = sample.get("other_attributes") or {}
    ref = oa.get("Transcription") or oa.get("transcription")
    if ref is None:
        ans = sample.get("answer")
        if isinstance(ans, dict):
            ref = ans.get("text")
        else:
            ref = ans
    return ref or ""


def transcribe_eagle(model, eagle, rotary_emb, processor,
                      audio_array, sample_rate, instruction,
                      max_new_tokens, device, K=4):
    """Single-sample inference with EAGLE draft + chain accept."""
    from transformers.cache_utils import HybridCache, DynamicCache

    input_features, feature_attention_mask, n_speech = prepare_audio(
        audio_array, sample_rate, processor)
    tokenizer = processor.tokenizer
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

    _dtype = next(p.dtype for p in model.parameters()
                  if p.dtype in (torch.float16, torch.bfloat16))
    input_features = input_features.to(device).to(_dtype)
    feature_attention_mask = feature_attention_mask.to(device)

    seq_len   = input_ids.shape[1]
    max_cache = seq_len + max_new_tokens
    verifier_kv = HybridCache(
        model.text_decoder.model.config, max_batch_size=1,
        max_cache_len=max_cache, dtype=_dtype, device=device)

    eos_ids = {tokenizer.eos_token_id,
               tokenizer.convert_tokens_to_ids("<end_of_turn>")}
    eos_ids.discard(None)

    torch.cuda.synchronize()
    t0 = time.time()
    generated_ids = []
    n_spec_acc = n_spec_tot = 0

    with torch.inference_mode():
        # ── Prefill ────────────────────────────────────────────────────────────
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            past_key_values=verifier_kv,
            use_cache=True,
            cache_position=torch.arange(0, seq_len, device=device),
            output_hidden_states=True,
            return_dict=True,
        )
        h_last = out.hidden_states[-1][0, -1:, :]   # (1, H)
        next_tok = int(out.logits[0, -1].argmax())
        generated_ids.append(next_tok)
        torch.cuda.synchronize()
        t1 = time.time()

        cur_pos = seq_len

        while len(generated_ids) < max_new_tokens and next_tok not in eos_ids:
            # ── Draft: K sequential steps ──────────────────────────────────────
            # EAGLE has its OWN KV cache that persists across its K steps but
            # is FRESH per verifier round (resets each time).
            draft_kv = DynamicCache()
            last_tok = next_tok
            last_h   = h_last                 # (1, H) — previous round's last h
            draft_tokens = []
            K_eff_max = min(K, max_cache - cur_pos - 1)
            for k in range(K_eff_max):
                input_ids_d = torch.tensor([[last_tok]], dtype=torch.long, device=device)
                prev_h_d    = last_h.unsqueeze(0)                 # (1, 1, H)
                # Position along the draft: cur_pos for the first step, +1 etc.
                pos_ids_d = torch.tensor([[cur_pos + k]], dtype=torch.long, device=device)
                pos_embs  = rotary_emb(prev_h_d, pos_ids_d)
                cache_pos = torch.tensor([k], device=device)     # within draft_kv
                logits_d, h_new, _ = eagle(
                    input_ids=input_ids_d,
                    prev_hidden=prev_h_d,
                    position_ids=pos_ids_d,
                    attention_mask=None,        # causal default inside layer
                    cache_position=cache_pos,
                    past_key_value=draft_kv,
                    position_embeddings=pos_embs,
                )
                # greedy argmax
                tok = int(logits_d[0, -1].argmax())
                draft_tokens.append(tok)
                # Setup for next draft step
                last_tok = tok
                last_h   = h_new[0]             # (1, H) slice
                if tok in eos_ids:
                    break

            K_eff = len(draft_tokens)
            if K_eff == 0:
                # Fallback greedy single step
                v_out = model.text_decoder(
                    input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=device),
                    attention_mask=torch.ones(1, cur_pos + 1, dtype=torch.long, device=device),
                    past_key_values=verifier_kv, use_cache=True,
                    cache_position=torch.tensor([cur_pos], device=device),
                    output_hidden_states=True, return_dict=True,
                )
                h_last = v_out.hidden_states[-1][0, -1:, :]
                next_tok = int(v_out.logits[0, -1].argmax())
                generated_ids.append(next_tok)
                cur_pos += 1
                continue

            # ── Verify: batched forward on [next_tok, d_0..d_{K-1}] ─────────────
            spec_ids  = torch.tensor([[next_tok] + draft_tokens],
                                     dtype=torch.long, device=device)
            spec_attn = torch.ones(1, cur_pos + K_eff + 1, dtype=torch.long, device=device)
            spec_cpos = torch.arange(cur_pos, cur_pos + K_eff + 1, device=device)
            v_out = model.text_decoder(
                input_ids=spec_ids,
                attention_mask=spec_attn,
                past_key_values=verifier_kv, use_cache=True,
                cache_position=spec_cpos,
                output_hidden_states=True, return_dict=True,
            )
            n_spec_tot += K_eff

            # ── Chain accept ───────────────────────────────────────────────────
            n_acc = 0
            stopped = False
            for i in range(K_eff):
                if len(generated_ids) >= max_new_tokens:
                    stopped = True; break
                pred = int(v_out.logits[0, i].argmax())
                if pred == draft_tokens[i]:
                    generated_ids.append(draft_tokens[i])
                    n_acc += 1; n_spec_acc += 1
                    next_tok = draft_tokens[i]
                    if draft_tokens[i] in eos_ids:
                        stopped = True; break
                else:
                    generated_ids.append(pred)
                    n_acc += 1
                    next_tok = pred
                    stopped = True; break

            if n_acc == K_eff and not stopped and len(generated_ids) < max_new_tokens:
                # verifier's bonus next-token using the final position
                next_tok = int(v_out.logits[0, K_eff].argmax())
                generated_ids.append(next_tok)
                n_acc += 1

            # ── Trim verifier KV ───────────────────────────────────────────────
            valid_end = cur_pos + n_acc
            for i in range(len(verifier_kv.key_cache)):
                kc = verifier_kv.key_cache[i]
                vc = verifier_kv.value_cache[i]
                if kc is not None and kc.shape[2] > valid_end:
                    kc[:, :, valid_end:, :].zero_()
                if vc is not None and vc.shape[2] > valid_end:
                    vc[:, :, valid_end:, :].zero_()
            verifier_kv._seen_tokens = valid_end

            # h_last = verifier's hidden state at the last accepted input pos
            idx_h = min(n_acc - 1, v_out.hidden_states[-1].size(1) - 1)
            h_last = v_out.hidden_states[-1][0, idx_h:idx_h + 1, :]
            cur_pos = valid_end

    torch.cuda.synchronize()
    t2 = time.time()

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
    n_tokens   = max(len(generated_ids), 1)
    decode_tps = max(len(generated_ids) - 1, 1) / (t2 - t1) if t2 > t1 else 0.0
    return text, {
        "n_tokens": n_tokens,
        "decode_tps": decode_tps,
        "spec_accept_rate": n_spec_acc / n_spec_tot if n_spec_tot > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True)
    ap.add_argument("--eagle",   required=True, help="Path to eagle_best.pt")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num_samples",    type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--K",              type=int, default=4,
                    help="Number of draft tokens per round")
    ap.add_argument("--device",  default="cuda")
    ap.add_argument("--audiobench_norm", action="store_true")
    ap.add_argument("--output",  default="gpu_3B_eagle.json")
    args = ap.parse_args()

    model, processor = load_model_gpu(
        args.model, quant="bf16", flash_attn=True, device=args.device)
    gpu_load_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    print(f"  VRAM after model load: {gpu_load_gb:.2f} GB")

    eagle, rotary_emb = load_eagle(model, args.eagle, args.device)

    print("Warming up GPU …")
    from datasets import load_from_disk
    raw = load_from_disk(os.path.abspath(args.dataset))
    shuffled = raw.shuffle(seed=42)
    start = max(0, min(10500, len(shuffled) - args.num_samples))
    end   = min(start + args.num_samples, len(shuffled))
    data = shuffled.select(range(start, end))

    # Warmup
    warmup = data[0]
    ao = _extract_audio(warmup)
    if ao is not None and ao.get("array") is not None:
        _ = transcribe_eagle(
            model, eagle, rotary_emb, processor,
            np.asarray(ao["array"], dtype=np.float32),
            ao.get("sampling_rate", SAMPLE_RATE),
            "Transcribe the speech", 32, args.device, K=args.K)

    results = []
    total_time, total_toks, total_acc_r = 0.0, 0, 0.0
    print(f"\nRunning EAGLE inference on {len(data)} samples (K={args.K}) …")
    for i in range(len(data)):
        s = data[i]
        ao = _extract_audio(s)
        if ao is None:
            continue
        aud = np.asarray(ao["array"], dtype=np.float32)
        sr  = ao.get("sampling_rate", SAMPLE_RATE)
        ref = _extract_ref(s)

        t0 = time.time()
        hyp, stats = transcribe_eagle(
            model, eagle, rotary_emb, processor, aud, sr,
            "Transcribe the speech", args.max_new_tokens, args.device, K=args.K)
        dt = time.time() - t0
        total_time += dt; total_toks += stats["n_tokens"]
        total_acc_r += stats["spec_accept_rate"]
        results.append({"hyp": hyp, "ref": ref, "stats": stats, "lat_s": dt})
        print(f"  [{i+1:3d}/{len(data)}]  {dt:5.2f}s  "
              f"{stats['decode_tps']:6.1f} tok/s  acc={stats['spec_accept_rate']:.1%} | {hyp[:60]!r}",
              flush=True)

    # WER
    try:
        import jiwer
        _norm = _normalize_text_audiobench if args.audiobench_norm else _normalize_text
        refs = [_norm(r["ref"]) for r in results]
        hyps = [_norm(r["hyp"]) for r in results]
        wer = jiwer.wer(refs, hyps) if refs else 0.0
    except ImportError:
        wer = 0.0

    avg_lat = total_time / len(results) if results else 0.0
    avg_tps = total_toks / total_time if total_time > 0 else 0.0
    avg_acc = total_acc_r / len(results) if results else 0.0
    vram_peak = torch.cuda.max_memory_allocated(args.device) / 1e9

    summary = {
        "wer": wer,
        "avg_latency_s": avg_lat,
        "avg_decode_tps": avg_tps,
        "avg_spec_accept_rate": avg_acc,
        "gpu_mem_peak_gb": vram_peak,
        "num_samples": len(results),
        "K": args.K,
        "eagle_path": args.eagle,
    }
    print("=" * 60)
    print(f"WER:             {wer:.4f}  ({wer*100:.2f}%)")
    print(f"Avg latency:     {avg_lat:.2f} s/sample")
    print(f"Avg decode:      {avg_tps:.2f} tok/s")
    print(f"Spec acc rate:   {avg_acc:.1%}")
    print(f"GPU VRAM peak:   {vram_peak:.2f} GB")
    print(f"K:               {args.K}")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump({**summary, "results": results}, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
