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
    SAMPLE_RATE, transcribe_gpu,
)
from eagle_model import EAGLE, attach_eagle


def load_eagle(model, ckpt_path, device):
    """Load EAGLE heads from a .pt checkpoint (local trained) or .safetensors
    (HF package eagle.safetensors).  Both formats are handled transparently.
    """
    verifier_dtype = next(
        (p.dtype for p in model.text_decoder.parameters()
         if p.dtype in (torch.float16, torch.bfloat16)),
        torch.bfloat16)

    if ckpt_path.endswith(".safetensors"):
        # HF package format: eagle.safetensors + eagle_config.json side-by-side
        cfg_path = os.path.splitext(ckpt_path)[0] + "_config.json"
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(os.path.dirname(ckpt_path), "eagle_config.json")
        n_layers = 1
        if os.path.exists(cfg_path):
            import json as _json
            with open(cfg_path) as f:
                n_layers = _json.load(f).get("num_layers", 1)
        eagle, rotary = attach_eagle(model, device, dtype=verifier_dtype,
                                     num_layers=n_layers)
        from safetensors.torch import load_file as _load_st
        sd = _load_st(ckpt_path)
        missing, _ = eagle.load_state_dict(sd, strict=False)
        missing = [m for m in missing
                   if not (m.startswith("_embed") or m.startswith("_lm_head"))]
        if missing:
            print(f"  EAGLE: {len(missing)} missing keys (e.g. {missing[0]})")
        meta_str = f"safetensors, {n_layers} layer(s)"
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        n_layers = ckpt.get("num_layers", 1)
        eagle, rotary = attach_eagle(model, device, dtype=verifier_dtype,
                                     num_layers=n_layers)
        eagle.load_trainable_state_dict(ckpt["eagle_state"])
        meta_str = f"step {ckpt.get('step', '?')}, val_acc {ckpt.get('val_acc', '?')}"

    eagle.eval()
    for p in eagle.parameters():
        p.requires_grad_(False)
    print(f"Loaded EAGLE: {sum(p.numel() for p in eagle.parameters())/1e6:.1f} M params "
          f"({meta_str})")
    return eagle, rotary


def _resolve_hf_repo(repo_or_path: str) -> str:
    """Return a local directory for an HF repo id or existing local path."""
    if os.path.isdir(repo_or_path):
        return repo_or_path
    from huggingface_hub import snapshot_download
    print(f"Downloading HF repo {repo_or_path} …")
    return snapshot_download(repo_id=repo_or_path)


def _load_from_hf_package(local_dir: str, device: str,
                           dtype: torch.dtype = torch.float16,
                           kernel: str = "exllama"):
    """Load W4A16 MERaLiON from the bundled HF package layout:
      local_dir/
        text_decoder_w4a16/   — GPTQ Gemma2 text decoder
        base_bf16/             — BF16 speech_encoder + audio_adapter
    Returns (model, processor).
    """
    from load_gptq_marlin import _patch_autogptq_for_gemma2
    from meralion2_bl.modeling_meralion2 import MERaLiON2ForConditionalGeneration
    from transformers import AutoProcessor
    from auto_gptq import AutoGPTQForCausalLM

    td_dir   = os.path.join(local_dir, "text_decoder_w4a16")
    base_dir = os.path.join(local_dir, "base_bf16")
    if not os.path.isdir(td_dir):
        raise FileNotFoundError(f"HF package missing text_decoder_w4a16/: {td_dir}")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"HF package missing base_bf16/: {base_dir}")

    _patch_autogptq_for_gemma2()
    kw = {"exllama":  dict(disable_exllama=False, disable_exllamav2=True),
          "exllamav2": dict(disable_exllama=True, disable_exllamav2=False),
          "marlin":    dict(use_marlin=True)}.get(kernel, {})

    print(f"[1/2] Loading W4A16 text_decoder (kernel={kernel}) …")
    qmodel = AutoGPTQForCausalLM.from_quantized(
        td_dir, torch_dtype=dtype, trust_remote_code=False, **kw)
    qmodel = qmodel.to(device)

    print(f"[2/2] Loading BF16 base (speech_encoder / adapter) …")
    processor = AutoProcessor.from_pretrained(base_dir, trust_remote_code=True)
    model = MERaLiON2ForConditionalGeneration.from_pretrained(
        base_dir, torch_dtype=dtype, use_safetensors=True)
    model.text_decoder = qmodel.model
    model = model.to(device)
    model.eval()
    return model, processor


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
                # CRITICAL: use position_ids that match TRAINING.  Training fed
                # arange(T-1) starting from 0 per sample — i.e., EAGLE has its
                # own coordinate system (its KV cache is fresh each round).
                # Previous bug used cur_pos + k (absolute verifier positions),
                # which shifted RoPE to 150-280 range EAGLE never saw at train
                # time — caused acc to drop sharply with K.
                pos_ids_d = torch.tensor([[k]], dtype=torch.long, device=device)
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

    # ── model source (one of --hf_repo or --model required) ─────────────────
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf_repo", default=None,
                     help="HF repo id or local dir of the bundled EAGLE+W4A16 package "
                          "(e.g. kartmannXu/MERaLiON-2-3B-EAGLE-W4A16). "
                          "Auto-downloads via huggingface_hub if not a local path.")
    src.add_argument("--model", default=None,
                     help="Local GPTQ-Marlin MERaLiON dir (legacy path, "
                          "requires --bf16_path and --eagle)")

    ap.add_argument("--no_eagle", action="store_true",
                    help="Skip EAGLE heads — benchmark W4A16 greedy decode only. "
                         "Works with both --hf_repo and --model.")
    ap.add_argument("--eagle",      default=None,
                    help="Path to eagle_best.pt (only needed with --model)")
    ap.add_argument("--bf16_path",  default=None,
                    help="BF16 MERaLiON dir for speech_encoder/adapter "
                         "(only needed with --model --quant gptq_marlin)")
    ap.add_argument("--quant", default="bf16",
                    choices=["bf16", "gptq_marlin"],
                    help="Verifier quant (only used with --model path)")
    ap.add_argument("--gptq_kernel", default="exllama",
                    choices=["marlin", "exllama", "exllamav2"],
                    help="auto-gptq kernel (default: exllama — fastest at batch=1)")

    ap.add_argument("--dataset",         required=True)
    ap.add_argument("--num_samples",     type=int, default=20)
    ap.add_argument("--max_new_tokens",  type=int, default=128)
    ap.add_argument("--K",               type=int, default=4,
                    help="EAGLE draft depth (ignored when --no_eagle)")
    ap.add_argument("--device",          default="cuda")
    ap.add_argument("--audiobench_norm", action="store_true")
    ap.add_argument("--output",          default="gpu_3B_eagle.json")
    args = ap.parse_args()

    # ── model loading ─────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(args.device)

    hf_wrapper = None   # MERaLiON2EAGLEForASR when loaded from HF package
    if args.hf_repo:
        local_dir = _resolve_hf_repo(args.hf_repo)
        if args.no_eagle:
            # No-EAGLE path: load just the W4A16 verifier directly
            model, processor = _load_from_hf_package(
                local_dir, args.device, dtype=torch.float16, kernel=args.gptq_kernel)
        else:
            # Use the HF package's own MERaLiON2EAGLEForASR.generate_eagle()
            # to guarantee parity with the published 1.9× reference.
            sys.path.insert(0, local_dir)
            from modeling_eagle import MERaLiON2EAGLEForASR
            print(f"Loading via MERaLiON2EAGLEForASR.from_pretrained({args.hf_repo}) …")
            hf_wrapper = MERaLiON2EAGLEForASR.from_pretrained(
                args.hf_repo, torch_dtype=torch.float16,
                gptq_kernel=args.gptq_kernel, device=args.device)
            model     = hf_wrapper.model
            processor = hf_wrapper.processor
    else:
        # Legacy --model path
        if args.quant == "gptq_marlin":
            if not args.bf16_path:
                raise SystemExit("--bf16_path required when --quant gptq_marlin")
            from load_gptq_marlin import load_meralion2_gptq_marlin
            model, processor = load_meralion2_gptq_marlin(
                args.model, args.bf16_path, device=args.device,
                dtype=torch.float16, kernel=args.gptq_kernel)
        else:
            model, processor = load_model_gpu(
                args.model, quant="bf16", flash_attn=True, device=args.device)
        if not args.no_eagle:
            if not args.eagle:
                raise SystemExit("--eagle required (path to eagle_best.pt)")
            eagle, rotary_emb = load_eagle(model, args.eagle, args.device)

    gpu_load_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    print(f"  VRAM after load: {gpu_load_gb:.2f} GB")

    # ── dataset ───────────────────────────────────────────────────────────────
    from datasets import load_from_disk
    raw  = load_from_disk(os.path.abspath(args.dataset))
    end  = min(args.num_samples, len(raw))
    data = raw.select(range(0, end))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _build_eagle_inputs(aud, sr, instr):
        """Reproduce the audio + prompt preprocessing from transcribe_eagle."""
        input_features, feature_attention_mask, n_speech = prepare_audio(
            aud, sr, processor)
        tokenizer = processor.tokenizer
        speech_token_id = model.config.speech_token_index
        conv = [{"role": "user",
                 "content": (f"Instruction: {instr} \n"
                             "Follow the text instruction based on the "
                             "following audio: <SpeechHere>")}]
        prompt = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True)
        raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
        pos = raw_ids.index(speech_token_id)
        input_ids = torch.tensor(
            [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]],
            dtype=torch.long, device=args.device)
        attn = torch.ones_like(input_ids)
        td_dtype = next(p.dtype for p in model.text_decoder.parameters()
                        if p.dtype in (torch.float16, torch.bfloat16))
        input_features = input_features.to(args.device).to(td_dtype)
        feature_attention_mask = feature_attention_mask.to(args.device)
        return input_ids, attn, input_features, feature_attention_mask

    def _run_sample(ao, ref, warmup=False):
        aud = np.asarray(ao["array"], dtype=np.float32)
        sr  = ao.get("sampling_rate", SAMPLE_RATE)
        instr = "Transcribe the speech"
        mnt   = 32 if warmup else args.max_new_tokens

        if args.no_eagle:
            hyp, stats = transcribe_gpu(
                model, processor, aud, sr,
                instruction=instr, max_new_tokens=mnt,
                device=args.device, speculative=False)
            stats.setdefault("spec_accept_rate", None)
        elif hf_wrapper is not None:
            input_ids, attn, infeat, fmask = _build_eagle_inputs(aud, sr, instr)
            seq_len = input_ids.shape[1]
            torch.cuda.synchronize()
            t0 = time.time()
            out_ids, gstats = hf_wrapper.generate_eagle(
                input_ids=input_ids, attention_mask=attn,
                input_features=infeat, feature_attention_mask=fmask,
                max_new_tokens=mnt, K=args.K, return_stats=True)
            torch.cuda.synchronize()
            # Decode generated text
            gen = out_ids[0, seq_len:].tolist()
            tokenizer = processor.tokenizer
            hyp = tokenizer.decode(gen, skip_special_tokens=True)
            hyp = hyp.replace("<Speaker1>:", "").replace("<Speaker2>:", "").strip()
            n_tot = gstats["n_spec_tot"]
            n_acc = gstats["n_spec_acc"]
            decode_dt = gstats["decode_dt"]
            n_gen = gstats["n_generated"]
            stats = {
                "n_tokens":         n_gen,
                "decode_tps":       max(n_gen - 1, 1) / decode_dt if decode_dt > 0 else 0.0,
                "spec_accept_rate": (n_acc / n_tot) if n_tot > 0 else 0.0,
            }
            hyp = hyp
        else:
            hyp, stats = transcribe_eagle(
                model, eagle, rotary_emb, processor, aud, sr,
                instr, mnt, args.device, K=args.K)
        return hyp, stats

    # ── warmup ────────────────────────────────────────────────────────────────
    print("Warming up GPU …")
    ao0 = _extract_audio(data[0])
    if ao0 is not None and ao0.get("array") is not None:
        _run_sample(ao0, "", warmup=True)
    torch.cuda.reset_peak_memory_stats(args.device)

    # ── benchmark loop ────────────────────────────────────────────────────────
    results = []
    total_time = total_toks = total_acc_r = 0.0
    n_acc_samples = 0
    mode_str = "greedy W4A16" if args.no_eagle else f"EAGLE K={args.K}"
    print(f"\nRunning {mode_str} on {len(data)} samples …")

    for i in range(len(data)):
        s  = data[i]
        ao = _extract_audio(s)
        if ao is None:
            continue
        ref = _extract_ref(s)

        t0 = time.time()
        hyp, stats = _run_sample(ao, ref)
        dt = time.time() - t0
        total_time += dt
        total_toks += stats["n_tokens"]
        acc = stats.get("spec_accept_rate")
        if acc is not None:
            total_acc_r += acc
            n_acc_samples += 1
        results.append({"hyp": hyp, "ref": ref, "stats": stats, "lat_s": dt})

        acc_str = f"  acc={acc:.1%}" if acc is not None else ""
        print(f"  [{i+1:3d}/{len(data)}]  {dt:5.2f}s  "
              f"{stats['decode_tps']:6.1f} tok/s{acc_str} | {hyp[:60]!r}",
              flush=True)

    # ── WER ───────────────────────────────────────────────────────────────────
    try:
        import jiwer
        _norm = _normalize_text_audiobench if args.audiobench_norm else _normalize_text
        refs = [_norm(r["ref"]) for r in results]
        hyps = [_norm(r["hyp"]) for r in results]
        wer  = jiwer.wer(refs, hyps) if refs else 0.0
    except ImportError:
        wer = 0.0

    avg_lat  = total_time / len(results) if results else 0.0
    avg_tps  = total_toks / total_time   if total_time > 0 else 0.0
    avg_acc  = total_acc_r / n_acc_samples if n_acc_samples > 0 else None
    vram_peak = torch.cuda.max_memory_allocated(args.device) / 1e9

    summary = {
        "wer":                  wer,
        "avg_latency_s":        avg_lat,
        "avg_decode_tps":       avg_tps,
        "avg_spec_accept_rate": avg_acc,
        "gpu_mem_peak_gb":      vram_peak,
        "num_samples":          len(results),
        "no_eagle":             args.no_eagle,
        "K":                    None if args.no_eagle else args.K,
        "source":               args.hf_repo or args.model,
    }
    print("=" * 60)
    print(f"Mode:            {mode_str}")
    print(f"WER:             {wer:.4f}  ({wer*100:.2f}%)")
    print(f"Avg latency:     {avg_lat:.2f} s/sample")
    print(f"Avg decode:      {avg_tps:.2f} tok/s")
    if avg_acc is not None:
        print(f"Spec acc rate:   {avg_acc:.1%}")
    print(f"GPU VRAM peak:   {vram_peak:.2f} GB")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump({**summary, "results": results}, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
