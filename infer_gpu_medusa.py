"""Medusa speculative decoding inference for MERaLiON-2-3B.

Each decode round:
    1. verifier(input) -> (logits_last, hidden_last)
    2. next_tok = argmax(logits_last)                       # verifier's own +1
    3. draft = [head_k(h_last).argmax() for k=1..K-1]        # +2..+K proposals
    4. verifier([next_tok, *draft]) -> K logits              # verify in one pass
    5. greedy accept as many as match

head_0 is trained for offset +1 (same as verifier's own), so at inference we
skip it and use heads 1..K-1 for offsets +2..+K — giving K-1 draft tokens per
round plus verifier's own +1 = K tokens per successful round.

Usage:
    python infer_gpu_medusa.py \\
        --model /home/kaixin/programs/LLM_base_model/MERaLiON-2-3B \\
        --heads /home/kaixin/yisong/merilion/medusa_heads.pt \\
        --dataset /home/kaixin/ssd/data/ASR/IMDA_PART1_mono_en_30_ASR \\
        --num_samples 20 \\
        --output gpu_3B_medusa.json
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
from medusa_model import MedusaHeads


def load_medusa_heads(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    td = model.text_decoder
    heads = MedusaHeads(
        num_heads=cfg["num_heads"],
        hidden_size=cfg["hidden_size"],
        lm_head=td.lm_head,
        num_layers=cfg["num_layers"],
    )
    heads.load_trainable_state_dict(ckpt["heads_state"])
    heads = heads.to(device).to(td.lm_head.weight.dtype)
    heads._lm_head = td.lm_head   # re-bind shared lm_head
    heads.eval()
    for p in heads.parameters():
        p.requires_grad_(False)
    print(f"Loaded Medusa heads: K={cfg['num_heads']}, "
          f"hidden={cfg['hidden_size']}, layers={cfg['num_layers']} "
          f"(step {ckpt.get('step', '?')})")
    return heads


def transcribe_medusa(model, heads, processor, audio_array, sample_rate,
                      instruction, max_new_tokens, device):
    from transformers.cache_utils import HybridCache

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
    past_kv   = HybridCache(
        model.text_decoder.model.config,
        max_batch_size=1,
        max_cache_len=max_cache,
        dtype=_dtype,
        device=device,
    )

    eos_ids = {tokenizer.eos_token_id,
               tokenizer.convert_tokens_to_ids("<end_of_turn>")}
    eos_ids.discard(None)

    K = heads.num_heads
    # Training target: head_k(h_i) → token[i+k+1].  In inference, h_last was
    # the hidden state that produced next_tok (via lm_head at offset +1).
    # head_k(h_last) therefore predicts the token at offset +(k+1) AFTER
    # next_tok's position; i.e. head 0 → first draft, head K-1 → K-th draft.
    # Use ALL K heads as parallel proposals of K consecutive draft tokens.
    proposal_heads = range(0, K)

    torch.cuda.synchronize()
    t0 = time.time()
    generated_ids = []
    n_spec_acc = n_spec_tot = 0

    with torch.inference_mode():
        # Prefill
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            past_key_values=past_kv,
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

        while len(generated_ids) < max_new_tokens:
            if next_tok in eos_ids:
                break

            # Step 1: Medusa heads propose K-1 tokens (parallel, very cheap).
            # h_last is the last hidden state from the PREVIOUS round's verifier
            # forward — it corresponds to the token that produced next_tok.
            draft_logits = heads(h_last.unsqueeze(0))   # (K, 1, 1, V)
            draft_tokens = []
            for k in proposal_heads:
                tk = int(draft_logits[k, 0, 0].argmax())
                draft_tokens.append(tk)
            # Cap by remaining cache
            K_draft = min(len(draft_tokens), max_cache - cur_pos - 1)
            draft_tokens = draft_tokens[:K_draft]
            # Truncate at first EOS to avoid proposing past a stop
            for i, t in enumerate(draft_tokens):
                if t in eos_ids:
                    draft_tokens = draft_tokens[: i + 1]
                    break
            K_eff = len(draft_tokens)

            if K_eff == 0:
                # Fallback: single greedy verifier step
                v_out = model.text_decoder(
                    input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=device),
                    attention_mask=torch.ones(1, cur_pos + 1, dtype=torch.long, device=device),
                    past_key_values=past_kv,
                    use_cache=True,
                    cache_position=torch.tensor([cur_pos], device=device),
                    output_hidden_states=True,
                    return_dict=True,
                )
                h_last = v_out.hidden_states[-1][0, -1:, :]
                next_tok = int(v_out.logits[0, -1].argmax())
                generated_ids.append(next_tok)
                cur_pos += 1
                continue

            # Step 2: Verifier verifies [next_tok, draft_1..K_eff] in one pass
            spec_ids  = torch.tensor([[next_tok] + draft_tokens],
                                     dtype=torch.long, device=device)
            spec_attn = torch.ones(1, cur_pos + K_eff + 1,
                                   dtype=torch.long, device=device)
            spec_cpos = torch.arange(cur_pos, cur_pos + K_eff + 1, device=device)
            v_out = model.text_decoder(
                input_ids=spec_ids,
                attention_mask=spec_attn,
                past_key_values=past_kv,
                use_cache=True,
                cache_position=spec_cpos,
                output_hidden_states=True,
                return_dict=True,
            )
            n_spec_tot += K_eff

            # Step 3: greedy acceptance
            n_acc = 0
            stopped = False
            for i in range(K_eff):
                if len(generated_ids) >= max_new_tokens:
                    stopped = True
                    break
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
                    stopped = True
                    break

            # If all K_eff accepted and no EOS, we still need a "next next_tok"
            # from verifier's last position logit:
            if n_acc == K_eff and not stopped and len(generated_ids) < max_new_tokens:
                next_tok = int(v_out.logits[0, K_eff].argmax())
                generated_ids.append(next_tok)
                n_acc += 1

            # Trim verifier KV to cur_pos + n_acc (discard unaccepted slots).
            valid_end = cur_pos + n_acc
            # Zero out stale HybridCache slots past valid_end
            for i in range(len(past_kv.key_cache)):
                kc = past_kv.key_cache[i]
                if kc is not None and kc.shape[2] > valid_end:
                    kc[:, :, valid_end:, :].zero_()
            for i in range(len(past_kv.value_cache)):
                vc = past_kv.value_cache[i]
                if vc is not None and vc.shape[2] > valid_end:
                    vc[:, :, valid_end:, :].zero_()
            past_kv._seen_tokens = valid_end

            # h_last: use the hidden state at the last ACCEPTED position
            # (position n_acc-1 in v_out — zero-indexed from [next_tok, d0..d_{K_eff-1}, +1]).
            # When n_acc <= K_eff, h_last index = n_acc - 1.
            # When n_acc == K_eff+1 (full accept including verifier's extra),
            # h_last index = K_eff.
            idx_h = min(n_acc - 1, v_out.hidden_states[-1].size(1) - 1)
            h_last = v_out.hidden_states[-1][0, idx_h:idx_h + 1, :]
            cur_pos = valid_end

    torch.cuda.synchronize()
    t2 = time.time()

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    text = text.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
    n_tokens   = max(len(generated_ids), 1)
    decode_tps = max(len(generated_ids) - 1, 1) / (t2 - t1) if t2 > t1 else 0.0
    stats = {
        "n_tokens":         n_tokens,
        "decode_tps":       decode_tps,
        "spec_accept_rate": n_spec_acc / n_spec_tot if n_spec_tot > 0 else 0.0,
    }
    return text, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--heads", required=True, help="Path to medusa_heads.pt")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quant", default="bf16",
                    choices=["bf16", "fp16", "int8", "int4", "mlx4", "awq4", "autoawq4"],
                    help="Verifier quantization (heads remain BF16, shared lm_head is skipped by quantization)")
    ap.add_argument("--audiobench_norm", action="store_true")
    ap.add_argument("--output", default="gpu_3B_medusa.json")
    args = ap.parse_args()

    model, processor = load_model_gpu(
        args.model, quant=args.quant, flash_attn=True, device=args.device)
    gpu_load_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    print(f"  VRAM after model load: {gpu_load_gb:.2f} GB")

    heads = load_medusa_heads(model, args.heads, args.device)

    # Warmup — replicate infer_gpu.py's sample selection exactly so our
    # numbers are directly comparable to gpu_3B_bf16_nospec.json etc.
    print("Warming up GPU …")
    from datasets import load_from_disk
    raw = load_from_disk(os.path.abspath(args.dataset))
    shuffled = raw.shuffle(seed=42)
    start = min(10500, len(shuffled))
    end   = min(start + args.num_samples, len(shuffled))
    data = shuffled.select(range(start, end))
    warmup_sample = data[0]
    ctx = warmup_sample.get("context") or {}
    ao = ctx.get("audio") if isinstance(ctx, dict) else None
    if ao and isinstance(ao, dict) and ao.get("array") is not None:
        aud = np.asarray(ao["array"], dtype=np.float32)
        sr = ao.get("sampling_rate", SAMPLE_RATE)
        transcribe_medusa(model, heads, processor, aud, sr,
                          "Transcribe the speech", 32, args.device)

    # Inference loop
    results = []
    total_time  = 0.0
    total_toks  = 0
    total_acc_r = 0.0
    n = len(data)
    print(f"\nRunning Medusa inference on {n} samples …")
    for i in range(n):
        s = data[i]
        ctx = s.get("context") or {}
        ao = ctx.get("audio") if isinstance(ctx, dict) else None
        if ao is None:
            continue
        aud = np.asarray(ao["array"], dtype=np.float32)
        sr  = ao.get("sampling_rate", SAMPLE_RATE)
        ref = (s.get("other_attributes") or {}).get("Transcription") or \
              (s.get("other_attributes") or {}).get("transcription") or ""

        t0 = time.time()
        hyp, stats = transcribe_medusa(
            model, heads, processor, aud, sr,
            "Transcribe the speech", args.max_new_tokens, args.device)
        dt = time.time() - t0
        total_time  += dt
        total_toks  += stats["n_tokens"]
        total_acc_r += stats["spec_accept_rate"]
        results.append({"hyp": hyp, "ref": ref, "stats": stats, "lat_s": dt})
        print(f"  [{i+1:3d}/{n}]  {dt:5.2f}s  {stats['decode_tps']:6.1f} tok/s  "
              f"acc={stats['spec_accept_rate']:.1%} | {hyp[:60]!r}", flush=True)

    # WER
    try:
        import jiwer
    except ImportError:
        jiwer = None
    if jiwer is not None:
        refs = [r["ref"] or "" for r in results]
        hyps = [r["hyp"] for r in results]
        if args.audiobench_norm:
            refs = [_normalize_text_audiobench(x) for x in refs]
            hyps = [_normalize_text_audiobench(x) for x in hyps]
        else:
            refs = [_normalize_text(x) for x in refs]
            hyps = [_normalize_text(x) for x in hyps]
        wer = jiwer.wer(refs, hyps) if refs else 0.0
    else:
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
        "num_heads": heads.num_heads,
        "heads_path": args.heads,
    }
    print("=" * 60)
    print(f"WER:             {wer:.4f}  ({wer*100:.2f}%)")
    print(f"Avg latency:     {avg_lat:.2f} s/sample")
    print(f"Avg decode:      {avg_tps:.2f} tok/s")
    print(f"Spec acc rate:   {avg_acc:.1%}")
    print(f"GPU VRAM peak:   {vram_peak:.2f} GB")
    print(f"num_heads:       {heads.num_heads}")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump({**summary, "results": results}, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
