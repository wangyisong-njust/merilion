"""Medusa speculative decoding inference for MERaLiON-2-3B
(TREE-ATTENTION variant).

Differences from infer_gpu_medusa.py (chain acceptance):
  * Draft step: each head keeps top-B candidates instead of top-1.
  * The B^K product tree (root + depth-1..K expansions) is flattened
    into a single sequence and fed through the verifier in ONE forward
    pass with a custom 4D attention mask that matches the tree
    topology (each node attends only to its ancestors + the shared
    prefix in the KV cache).
  * After verify, we walk the tree from root to pick the longest path
    whose every step matches the verifier's prediction at its parent;
    that path's KVs are gather/scattered back into a contiguous
    prefix [cur_pos, cur_pos+depth), and cur_pos advances by
    depth+1 (the +1 is the verifier's bonus successor).

This decouples the K (depth) knob from the per-head top-1 assumption
so the expected accepted length can exceed the chain-acceptance bound.

Required verifier settings:
  * attention implementation must be `eager` or `sdpa` — FlashAttention
    doesn't accept custom 4D additive masks.  We set it on the text
    decoder's config at load time.
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


def build_product_tree(heads_module, h_last, K, B, eos_ids):
    """Construct a BFS-ordered flat representation of a K-depth B-ary product
    tree of draft tokens.

    Node 0 is the ROOT and always holds `None` — the caller knows the root
    is `next_tok` (verifier's own +1 from the previous round).  Nodes 1..N
    are drafts, organised so all depth-d nodes appear before depth-(d+1).

    Returns (tokens, parents, depths) as lists of length N (NOT including
    the root).  `parents[i]` is the index in the FLAT array of node i's
    parent, where -1 means the root.  depths[i] is 1..K.

    Each head is evaluated once (all nodes at depth d share `head[d-1]`
    top-B).  This gives a very simple tree where every depth-(d+1) node
    is a child of every depth-d node — i.e. a true Cartesian product
    tree of B^K leaves.  A future refinement could specialise head
    inputs per path (the "typical tree" in Medusa-2 paper).
    """
    K_heads = heads_module.num_heads
    K_eff = min(K, K_heads)
    # heads(h_last) → (K_heads, 1, 1, V).  Take top-B per head.
    draft_logits = heads_module(h_last.unsqueeze(0))            # (K, 1, 1, V)
    # top-B indices per head, as python lists of ints
    top = draft_logits.topk(B, dim=-1).indices[:, 0, 0].tolist()  # (K_heads, B)

    tokens, parents, depths = [], [], []
    level_nodes = [-1]                 # root index (meaning "no parent")
    for d in range(K_eff):
        B_tokens = top[d][:B]          # top-B for head d → depth d+1
        new_level = []
        for parent_flat_idx in level_nodes:
            for tk in B_tokens:
                tokens.append(int(tk))
                parents.append(parent_flat_idx)
                depths.append(d + 1)
                new_level.append(len(tokens) - 1)
        level_nodes = new_level

    return tokens, parents, depths


def build_tree_attention_mask(parents, depths, prefix_len, max_cache_len,
                              dtype, device):
    """Return a 4D additive mask of shape (1, 1, 1 + N, max_cache_len).

    The mask covers the FULL preallocated KV cache (HybridCache returns
    its whole tensor at `key_states.shape[-2] == max_cache_len`, and the
    verifier's eager path slices the mask to that length).

    Column layout:
      [0 .. prefix_len-1]                        = prefix (accepted context)
      [prefix_len .. prefix_len+N]               = THIS round's nodes
      [prefix_len+N+1 .. max_cache_len-1]        = unused (blocked)

    A node at row `r` attends to:
      * every prefix column (already-accepted context)
      * itself
      * every ancestor in the tree (walking parents back to root)
    """
    N = len(parents)
    min_dtype = torch.finfo(dtype).min

    # Start with "block everything", then poke 0 into allowed positions.
    mask = torch.full((1, 1, 1 + N, max_cache_len), min_dtype,
                      dtype=dtype, device=device)
    # All nodes see the entire prefix.
    mask[..., :prefix_len] = 0.0

    # Root (row 0) sees itself at col prefix_len.
    mask[..., 0, prefix_len] = 0.0
    # Each flat node i (row i+1) can see root + its ancestors + itself.
    for i in range(N):
        row = i + 1
        col = prefix_len + 1 + i
        mask[..., row, col] = 0.0               # self
        p = parents[i]
        while p != -1:
            mask[..., row, prefix_len + 1 + p] = 0.0
            p = parents[p]
        mask[..., row, prefix_len] = 0.0        # root always
    return mask


def tree_accept_path(tokens, parents, depths, v_logits_flat, eos_ids):
    """Greedy longest-match acceptance down the tree.

    v_logits_flat[i] is the verifier logit at flat position i (shape V).
    Index 0 corresponds to the ROOT (its logits predict depth-1 tokens).
    Index i+1 (i>=0) corresponds to node i (its logits predict that
    node's children).

    Returns a list of accepted flat node indices: [idx_0, idx_1, ...]
    each at successively deeper depths.  Empty list means draft rejected
    at depth 1 (fall back to verifier's own +1 pred).
    """
    # Build children map for quick lookup
    children = {-1: []}   # root's children (depth-1 nodes)
    for i, p in enumerate(parents):
        children.setdefault(p, []).append(i)

    accepted = []
    # Current "parent" is the root (verifier-logit index 0).
    parent_idx = -1
    verifier_logit_row = 0  # root row in v_logits_flat

    while True:
        cand = children.get(parent_idx, [])
        if not cand:
            break
        v_pred = int(v_logits_flat[verifier_logit_row].argmax())
        # Find a child whose draft-token == v_pred
        matched = None
        for c in cand:
            if tokens[c] == v_pred:
                matched = c; break
        if matched is None:
            break                       # chain breaks — but we still report v_pred later
        accepted.append(matched)
        if tokens[matched] in eos_ids:
            break
        parent_idx = matched
        verifier_logit_row = matched + 1
    return accepted


def rearrange_kv_to_path(past_kv, write_start, accepted_flat_indices,
                         total_written):
    """After a tree-verify round we wrote `total_written` K/V slots at
    positions [write_start, write_start + total_written).  The flat
    layout was [root, node_0, ..., node_{N-1}], so flat index i occupies
    cache position write_start + i.

    We keep only the accepted path (0 based: root + accepted_flat_indices
    mapped to cache indices write_start+0, write_start+1+c_0,
    write_start+1+c_1, ...).  We want these to end up contiguous starting
    at write_start.  This function gather/scatter's them and zeros out
    the rest.  Returns the new valid_end (write_start + path_len).
    """
    path_len = 1 + len(accepted_flat_indices)           # root + accepted nodes
    src_positions = [write_start] + [write_start + 1 + c
                                     for c in accepted_flat_indices]
    dst_positions = list(range(write_start, write_start + path_len))

    # Nothing to move if already contiguous from the start.
    for i in range(len(past_kv.key_cache)):
        kc = past_kv.key_cache[i]
        vc = past_kv.value_cache[i]
        if kc is None or vc is None:
            continue
        # Gather once per layer then scatter.
        src_idx = torch.tensor(src_positions, dtype=torch.long, device=kc.device)
        dst_idx = torch.tensor(dst_positions, dtype=torch.long, device=kc.device)
        k_src = kc.index_select(2, src_idx).clone()
        v_src = vc.index_select(2, src_idx).clone()
        kc.index_copy_(2, dst_idx, k_src)
        vc.index_copy_(2, dst_idx, v_src)
        # Zero slots past the new valid_end (inside the written region).
        end = write_start + path_len
        written_end = write_start + total_written
        if written_end > end:
            kc[:, :, end:written_end, :].zero_()
            vc[:, :, end:written_end, :].zero_()
    past_kv._seen_tokens = write_start + path_len
    return write_start + path_len


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
    # Tree-attention config.  B = branches per head (top-B candidates).
    # N_tree = sum_{d=1..K} B^d.  Keep small; verifier runs on 1 + N_tree
    # tokens, so quadratic cost scales with it.
    B = int(os.environ.get("MEDUSA_TREE_B", "2"))

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

            # ── Step 1: build the product tree of draft candidates ──────────
            tree_tokens, tree_parents, tree_depths = build_product_tree(
                heads, h_last, K, B, eos_ids)
            N_tree = len(tree_tokens)
            # Cap so we never exceed cache size
            cap = max_cache - cur_pos - 1
            if N_tree + 1 > cap:
                # Drop leaves until we fit (prune deepest level first)
                while N_tree + 1 > cap and tree_tokens:
                    max_d = max(tree_depths)
                    # Remove one deepest node
                    for i in range(N_tree - 1, -1, -1):
                        if tree_depths[i] == max_d:
                            tree_tokens.pop(i); tree_parents.pop(i); tree_depths.pop(i)
                            N_tree -= 1
                            break
                if N_tree == 0:
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

            # ── Step 2: flatten [root, tree_nodes...] and verify in one pass ─
            flat_tokens = [next_tok] + tree_tokens
            flat_len    = 1 + N_tree
            spec_ids    = torch.tensor([flat_tokens], dtype=torch.long, device=device)
            spec_cpos   = torch.arange(cur_pos, cur_pos + flat_len, device=device)
            # Custom 4D tree mask: prefix is fully visible, nodes see ancestors
            tree_mask = build_tree_attention_mask(
                tree_parents, tree_depths,
                prefix_len=cur_pos, max_cache_len=max_cache,
                dtype=_dtype, device=device)
            # Position ids: root is at cur_pos; depth-d node at cur_pos + d
            pos_ids = [cur_pos] + [cur_pos + d for d in tree_depths]
            pos_ids = torch.tensor([pos_ids], dtype=torch.long, device=device)

            v_out = model.text_decoder(
                input_ids=spec_ids,
                attention_mask=tree_mask,
                position_ids=pos_ids,
                past_key_values=past_kv,
                use_cache=True,
                cache_position=spec_cpos,
                output_hidden_states=True,
                return_dict=True,
            )
            n_spec_tot += N_tree

            # ── Step 3: walk the tree, accept the longest matching path ─────
            v_logits_flat = v_out.logits[0]                   # (flat_len, V)
            accepted = tree_accept_path(
                tree_tokens, tree_parents, tree_depths,
                v_logits_flat, eos_ids)
            n_acc_drafts = len(accepted)
            n_spec_acc += n_acc_drafts

            # Append accepted draft tokens + verifier's bonus next-token
            stopped = False
            for c in accepted:
                if len(generated_ids) >= max_new_tokens:
                    stopped = True; break
                generated_ids.append(tree_tokens[c])
                if tree_tokens[c] in eos_ids:
                    stopped = True; break
            # Pick the verifier's bonus at the tail of the accepted path
            # (its logits row = matched+1 in flat; or row 0 if no accept).
            bonus_row = accepted[-1] + 1 if accepted else 0
            if not stopped and len(generated_ids) < max_new_tokens:
                bonus = int(v_logits_flat[bonus_row].argmax())
                generated_ids.append(bonus)
                next_tok = bonus
                if bonus in eos_ids:
                    stopped = True

            # ── Step 4: rearrange KV cache → keep only accepted path ───────
            # Written span was [cur_pos, cur_pos + 1 + N_tree); we keep root
            # + accepted drafts (1 + len(accepted)) as a contiguous prefix.
            valid_end = rearrange_kv_to_path(
                past_kv, write_start=cur_pos,
                accepted_flat_indices=accepted,
                total_written=1 + N_tree)

            # h_last: hidden state at the last ACCEPTED flat position so that
            # the next round's head forward is conditioned on the right state.
            #   • no drafts accepted  → use root (flat idx 0)
            #   • ≥1 draft accepted   → last accepted flat idx + 1
            idx_h = (accepted[-1] + 1) if accepted else 0
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

    # Tree-attention verifier forward uses a custom 4D additive mask;
    # FlashAttention-2 does not accept that — we explicitly disable it.
    model, processor = load_model_gpu(
        args.model, quant=args.quant, flash_attn=False, device=args.device)
    gpu_load_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    print(f"  VRAM after model load: {gpu_load_gb:.2f} GB")
    # Force eager attention (some paths default to sdpa even without FA2).
    # Eager explicitly slices / adds a 4D attention_mask to scores.
    model.text_decoder.config._attn_implementation = "eager"
    if hasattr(model.text_decoder, "model"):
        model.text_decoder.model.config._attn_implementation = "eager"

    heads = load_medusa_heads(model, args.heads, args.device)

    # Warmup — replicate infer_gpu.py's sample selection exactly so our
    # numbers are directly comparable to gpu_3B_bf16_nospec.json etc.
    print("Warming up GPU …")
    from datasets import load_from_disk
    raw = load_from_disk(os.path.abspath(args.dataset))
    shuffled = raw.shuffle(seed=42)
    # Clamp start so start + num_samples fits.  Small datasets
    # (< 10500) fall back to index 0.
    start = max(0, min(10500, len(shuffled) - args.num_samples))
    end   = min(start + args.num_samples, len(shuffled))
    data = shuffled.select(range(start, end))
    def _extract_audio(sample):
        """Support both schemas: nested `context.audio` (old IMDA) and
        flat `context` (AudioBench datasets.Audio feature)."""
        c = sample.get("context")
        if isinstance(c, dict):
            if "array" in c:   # flat / AudioBench
                return c
            if isinstance(c.get("audio"), dict):  # nested / old IMDA
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

    warmup_sample = data[0]
    ao = _extract_audio(warmup_sample)
    if ao is not None and ao.get("array") is not None:
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
        ao = _extract_audio(s)
        if ao is None:
            continue
        aud = np.asarray(ao["array"], dtype=np.float32)
        sr  = ao.get("sampling_rate", SAMPLE_RATE)
        ref = _extract_ref(s)

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
