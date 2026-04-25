"""EAGLE + tree-attention speculative decoding for MERaLiON-2-3B.

Difference from infer_gpu_eagle.py (chain): instead of a single chain of K
drafts, we fork at depth-1 into B branches and chain each branch K-1 more
steps.  Total drafts = B*K nodes; verifier sees 1+B*K tokens with a custom
4D tree attention mask, then accepts the longest matching path.

  Tree shape (B=2, K=4):
        root
       /    \\
      A0    B0
      |     |
      A1    B1
      |     |
      A2    B2
      |     |
      A3    B3

Step-1 acc is the largest gap (~0.54 single-step on EAGLE), so just doubling
candidates at depth-1 nearly doubles step-1 accept probability while only
adding K extra verifier tokens (≤2× draft cost).  Subsequent depths are
chained per branch — cheap because EAGLE recurrence is already sequential.

Requires eager (or sdpa) attention since FA2 doesn't support custom 4D
masks; we explicitly disable FA2 at load.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_gpu import (
    load_model_gpu, prepare_audio, _normalize_text, _normalize_text_audiobench,
    SAMPLE_RATE,
)
from infer_gpu_medusa_tree import (
    build_tree_attention_mask, tree_accept_path, rearrange_kv_to_path,
)
from infer_gpu_eagle import load_eagle, _extract_audio, _extract_ref
from eagle_model import EAGLE, attach_eagle


def _eagle_step(eagle, rotary_emb, last_tok, last_h, k, draft_kv, device):
    """One EAGLE forward step within its own coordinate system.
    `k` is the EAGLE step index (0-based, matches training arange).
    Returns (logits, h_new).
    """
    input_ids_d = torch.tensor([[last_tok]], dtype=torch.long, device=device)
    prev_h_d    = last_h.unsqueeze(0)                     # (1, 1, H)
    pos_ids_d   = torch.tensor([[k]], dtype=torch.long, device=device)
    pos_embs    = rotary_emb(prev_h_d, pos_ids_d)
    cache_pos   = torch.tensor([k], device=device)
    logits_d, h_new, _ = eagle(
        input_ids=input_ids_d,
        prev_hidden=prev_h_d,
        position_ids=pos_ids_d,
        attention_mask=None,
        cache_position=cache_pos,
        past_key_value=draft_kv,
        position_embeddings=pos_embs,
    )
    return logits_d, h_new


def build_eagle_tree(eagle, rotary_emb, next_tok, h_last, K, B, eos_ids, device):
    """Generate B branches, each a chain of K draft tokens, all rooted at
    (next_tok, h_last).  Returns flat (tokens, parents, depths) suitable
    for build_tree_attention_mask / tree_accept_path.

    Layout: branch 0 first (positions 0..K-1), then branch 1 (K..2K-1), etc.
      tree_parents[b*K + 0]   = -1                # roots into the verifier root
      tree_parents[b*K + i]   = b*K + i - 1       # chain within branch
      tree_depths [b*K + i]   = i + 1
    """
    from transformers.cache_utils import DynamicCache

    # Step 0 (shared across branches): EAGLE on (next_tok, h_last) → top-B logits.
    # We replay step 0 once per branch into a fresh draft_kv so each branch's
    # cache contains only its own continuation; the cost is small (B forwards
    # at depth 0).
    tokens = []
    parents = []
    depths = []

    # First, get top-B candidate tokens from a probe pass.
    probe_kv = DynamicCache()
    logits0, _h0_unused = _eagle_step(eagle, rotary_emb, next_tok, h_last, 0,
                                      probe_kv, device)
    topk = torch.topk(logits0[0, -1], k=B)
    top_toks = [int(t) for t in topk.indices.tolist()]

    for b, tok_b0 in enumerate(top_toks):
        draft_kv_b = DynamicCache()
        # Replay step 0 with same input so draft_kv_b has the depth-0 K/V entry.
        _logits, h_new = _eagle_step(eagle, rotary_emb, next_tok, h_last, 0,
                                     draft_kv_b, device)
        chain_tok = tok_b0
        chain_h   = h_new[0]                       # (1, H)

        tokens.append(chain_tok)
        parents.append(-1)
        depths.append(1)
        if chain_tok in eos_ids:
            continue                               # don't extend past EOS
        for k in range(1, K):
            logits_k, h_k = _eagle_step(eagle, rotary_emb, chain_tok, chain_h,
                                        k, draft_kv_b, device)
            tok = int(logits_k[0, -1].argmax())
            tokens.append(tok)
            parents.append(len(tokens) - 2)        # previous index in this branch
            depths.append(k + 1)
            chain_tok = tok
            chain_h   = h_k[0]
            if tok in eos_ids:
                break

    return tokens, parents, depths


def transcribe_eagle_tree(model, eagle, rotary_emb, processor,
                          audio_array, sample_rate, instruction,
                          max_new_tokens, device, K=4, B=2):
    from transformers.cache_utils import HybridCache

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
    max_cache = seq_len + max_new_tokens + B * K + 4   # headroom for tree writes
    past_kv = HybridCache(
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
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            past_key_values=past_kv, use_cache=True,
            cache_position=torch.arange(0, seq_len, device=device),
            output_hidden_states=True, return_dict=True,
        )
        h_last = out.hidden_states[-1][0, -1:, :]
        next_tok = int(out.logits[0, -1].argmax())
        generated_ids.append(next_tok)
        torch.cuda.synchronize()
        t1 = time.time()
        cur_pos = seq_len

        while len(generated_ids) < max_new_tokens and next_tok not in eos_ids:
            # ── Step 1: build the EAGLE draft tree ─────────────────────────────
            tree_tokens, tree_parents, tree_depths = build_eagle_tree(
                eagle, rotary_emb, next_tok, h_last, K, B, eos_ids, device)
            N_tree = len(tree_tokens)
            # Cap tree if we're near max_cache (rare since we headroomed).
            cap = max_cache - cur_pos - 1
            while N_tree + 1 > cap and tree_tokens:
                # drop deepest first
                max_d = max(tree_depths)
                for i in range(N_tree - 1, -1, -1):
                    if tree_depths[i] == max_d:
                        tree_tokens.pop(i); tree_parents.pop(i); tree_depths.pop(i)
                        N_tree -= 1
                        break
            if N_tree == 0:
                v_out = model.text_decoder(
                    input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=device),
                    attention_mask=torch.ones(1, cur_pos + 1, dtype=torch.long, device=device),
                    past_key_values=past_kv, use_cache=True,
                    cache_position=torch.tensor([cur_pos], device=device),
                    output_hidden_states=True, return_dict=True,
                )
                h_last = v_out.hidden_states[-1][0, -1:, :]
                next_tok = int(v_out.logits[0, -1].argmax())
                generated_ids.append(next_tok); cur_pos += 1
                continue

            # ── Step 2: verify the tree in one forward pass ────────────────────
            flat_tokens = [next_tok] + tree_tokens
            flat_len    = 1 + N_tree
            spec_ids    = torch.tensor([flat_tokens], dtype=torch.long, device=device)
            spec_cpos   = torch.arange(cur_pos, cur_pos + flat_len, device=device)
            tree_mask   = build_tree_attention_mask(
                tree_parents, tree_depths,
                prefix_len=cur_pos, max_cache_len=max_cache,
                dtype=_dtype, device=device)
            pos_ids = [cur_pos] + [cur_pos + d for d in tree_depths]
            pos_ids = torch.tensor([pos_ids], dtype=torch.long, device=device)

            v_out = model.text_decoder(
                input_ids=spec_ids,
                attention_mask=tree_mask,
                position_ids=pos_ids,
                past_key_values=past_kv, use_cache=True,
                cache_position=spec_cpos,
                output_hidden_states=True, return_dict=True,
            )
            # Comparable-to-chain accept rate: divide by chain depth K, not
            # by tree size (the unselected branch's drafts are NOT wasted —
            # they're just losers in the path competition, not "rejected").
            n_spec_tot += K

            # ── Step 3: walk the tree, accept the longest matching path ───────
            v_logits_flat = v_out.logits[0]
            accepted = tree_accept_path(
                tree_tokens, tree_parents, tree_depths,
                v_logits_flat, eos_ids)
            n_spec_acc += len(accepted)

            stopped = False
            for c in accepted:
                if len(generated_ids) >= max_new_tokens:
                    stopped = True; break
                generated_ids.append(tree_tokens[c])
                if tree_tokens[c] in eos_ids:
                    stopped = True; break

            # Verifier's bonus next-token at the end of the accepted path.
            bonus_row = accepted[-1] + 1 if accepted else 0
            if not stopped and len(generated_ids) < max_new_tokens:
                bonus = int(v_logits_flat[bonus_row].argmax())
                generated_ids.append(bonus)
                next_tok = bonus
                if bonus in eos_ids:
                    stopped = True
            else:
                # Already stopped; pick the last accepted token as next anchor.
                if accepted:
                    next_tok = tree_tokens[accepted[-1]]

            # ── Step 4: rearrange KV cache → keep only accepted path ───────────
            valid_end = rearrange_kv_to_path(
                past_kv, write_start=cur_pos,
                accepted_flat_indices=accepted,
                total_written=1 + N_tree)

            # h_last for next round = verifier's hidden at last accepted flat pos.
            idx_h = (accepted[-1] + 1) if accepted else 0
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
    ap.add_argument("--eagle",   required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num_samples",    type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--K", type=int, default=4, help="Chain depth per branch")
    ap.add_argument("--B", type=int, default=2, help="Branch width at depth 1")
    ap.add_argument("--device",  default="cuda")
    ap.add_argument("--audiobench_norm", action="store_true")
    ap.add_argument("--output",  default="gpu_3B_eagle_tree.json")
    args = ap.parse_args()

    # FA2 cannot accept custom 4D tree masks — disable.  Use SDPA (supports
    # 4D masks AND fused kernels) by default; user can override to eager via
    # EAGLE_TREE_ATTN=eager for debugging.
    model, processor = load_model_gpu(
        args.model, quant="bf16", flash_attn=False, device=args.device)
    attn_impl = os.environ.get("EAGLE_TREE_ATTN", "sdpa")
    model.text_decoder.config._attn_implementation = attn_impl
    if hasattr(model.text_decoder, "model"):
        model.text_decoder.model.config._attn_implementation = attn_impl
    print(f"  attention impl: {attn_impl}")
    gpu_load_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    print(f"  VRAM after model load: {gpu_load_gb:.2f} GB")

    eagle, rotary_emb = load_eagle(model, args.eagle, args.device)

    print("Loading data …")
    from datasets import load_from_disk
    raw = load_from_disk(os.path.abspath(args.dataset))
    end  = min(args.num_samples, len(raw))
    data = raw.select(range(0, end))

    # Warmup
    warmup = data[0]
    ao = _extract_audio(warmup)
    if ao is not None and ao.get("array") is not None:
        _ = transcribe_eagle_tree(
            model, eagle, rotary_emb, processor,
            np.asarray(ao["array"], dtype=np.float32),
            ao.get("sampling_rate", SAMPLE_RATE),
            "Transcribe the speech", 32, args.device, K=args.K, B=args.B)

    results = []
    total_time, total_toks, total_acc_r = 0.0, 0, 0.0
    print(f"\nRunning EAGLE-tree on {len(data)} samples (K={args.K}, B={args.B}) …")
    for i in range(len(data)):
        s = data[i]
        ao = _extract_audio(s)
        if ao is None:
            continue
        aud = np.asarray(ao["array"], dtype=np.float32)
        sr  = ao.get("sampling_rate", SAMPLE_RATE)
        ref = _extract_ref(s)

        t0 = time.time()
        hyp, stats = transcribe_eagle_tree(
            model, eagle, rotary_emb, processor, aud, sr,
            "Transcribe the speech", args.max_new_tokens, args.device,
            K=args.K, B=args.B)
        dt = time.time() - t0
        total_time += dt; total_toks += stats["n_tokens"]
        total_acc_r += stats["spec_accept_rate"]
        results.append({"hyp": hyp, "ref": ref, "stats": stats, "lat_s": dt})
        print(f"  [{i+1:3d}/{len(data)}]  {dt:5.2f}s  "
              f"{stats['decode_tps']:6.1f} tok/s  acc={stats['spec_accept_rate']:.1%} | {hyp[:60]!r}",
              flush=True)

    try:
        import jiwer
        _norm = _normalize_text_audiobench if args.audiobench_norm else _normalize_text
        refs = [_norm(r["ref"]) for r in results]
        hyps = [_norm(r["hyp"]) for r in results]
        wer = jiwer.wer(refs, hyps) if refs else 0.0
    except ImportError:
        wer = 0.0

    avg_lat   = total_time / len(results) if results else 0.0
    avg_tps   = total_toks / total_time if total_time > 0 else 0.0
    avg_acc   = total_acc_r / len(results) if results else 0.0
    vram_peak = torch.cuda.max_memory_allocated(args.device) / 1e9
    summary = {
        "wer":                  wer,
        "avg_latency_s":        avg_lat,
        "avg_decode_tps":       avg_tps,
        "avg_spec_accept_rate": avg_acc,
        "gpu_mem_peak_gb":      vram_peak,
        "num_samples":          len(results),
        "K":                    args.K,
        "B":                    args.B,
        "eagle_path":           args.eagle,
        "attn_impl":            attn_impl,
    }
    with open(args.output, "w") as f:
        json.dump({**summary, "results": results}, f, indent=2, default=str)
    print(f"\nSaved → {args.output}")
    print(f"  total {total_time:.2f}s   avg_tps {avg_tps:.1f}   "
          f"acc {avg_acc:.1%}   WER {wer*100:.2f}%")


if __name__ == "__main__":
    main()
