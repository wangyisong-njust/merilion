"""GPU inference benchmark for LLaMA-2-7B — speculative decoding + INT8.

Supports:
  bf16  — BF16 baseline (fastest, full quality)
  fp16  — FP16 baseline
  int8  — BitsAndBytes LLM.int8() (default for this script)

Speculative decoding uses the same NGramDraft from infer_gpu.py.
A small LLaMA-2-7B with --quant int8 is used as both target and drafter
(self-speculative via n-gram lookahead; no separate draft model required).

Timing: torch.cuda.synchronize() wall-clock.
VRAM:   torch.cuda.max_memory_allocated().

Usage:
    # INT8 + speculative (recommended):
    python infer_llama_gpu.py \\
        --model /path/to/Llama-2-7b-hf \\
        --prompt "Once upon a time" \\
        --quant int8 --speculative --gamma 5

    # Dataset benchmark (text only — no audio):
    python infer_llama_gpu.py \\
        --model /path/to/Llama-2-7b-hf \\
        --prompts_file prompts.txt \\
        --quant int8 --speculative --output llama_int8_spec.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch


# ── n-gram drafter (identical to infer_gpu.py) ────────────────────────────────

class NGramDraft:
    def __init__(self, ngram_sizes: tuple = (3, 4), index: dict = None):
        self.ngram_sizes = sorted(ngram_sizes, reverse=True)
        self.index = index or {}

    def propose(self, ctx: list, gamma: int) -> list:
        draft: list = []
        cur = list(ctx)
        for _ in range(gamma):
            tok = self._next(cur)
            if tok is None:
                break
            draft.append(tok)
            cur.append(tok)
        return draft

    def _next(self, ctx: list):
        for ng in self.ngram_sizes:
            plen = ng - 1
            if len(ctx) < plen:
                continue
            prefix = tuple(ctx[-plen:])
            if prefix in self.index:
                return self.index[prefix]
            n = len(ctx)
            if n >= ng:
                for i in range(n - ng):
                    if ctx[i: i + plen] == list(prefix):
                        return ctx[i + plen]
        return None


# ── model loading ─────────────────────────────────────────────────────────────

def load_llama_gpu(model_path: str,
                   quant: str = "int8",
                   device: str = "cuda"):
    """Load LLaMA-2-7B on GPU with optional BnB INT8/FP16/BF16.

    INT8: loads in FP16 first, then applies BitsAndBytes 8-bit quantization.
    BF16/FP16: direct load via from_pretrained with device_map.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"Loading tokenizer from {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LLaMA-2-7B [{quant.upper()}] …")
    t0 = time.time()

    if quant == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.float16,
        )
    elif quant == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
    else:
        dtype = torch.bfloat16 if quant == "bf16" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
        )

    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


# ── inference ─────────────────────────────────────────────────────────────────

def generate_gpu(model, tokenizer, prompt: str,
                 max_new_tokens: int = 128,
                 device: str = "cuda",
                 speculative: bool = False,
                 gamma: int = 5) -> tuple:
    """Generate text from a prompt on GPU.

    Returns (text, stats) where stats = {n_tokens, decode_tps,
    [spec_accept_rate]}.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = input_ids.shape[1]

    eos_id = tokenizer.eos_token_id

    if speculative:
        ngram = NGramDraft()
        generated_ids = []
        n_spec_acc = n_spec_tot = 0

        torch.cuda.synchronize()
        t0 = time.time()

        with torch.inference_mode():
            # Prefill
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            next_tok = int(out.logits[0, -1].argmax())
            generated_ids.append(next_tok)

            torch.cuda.synchronize()
            t1 = time.time()

            all_ctx = list(generated_ids)
            cur_len = seq_len + 1

            while len(generated_ids) < max_new_tokens:
                if next_tok == eos_id:
                    break

                draft = ngram.propose(all_ctx, gamma)

                if draft:
                    K = len(draft)
                    spec_ids = torch.tensor(
                        [[next_tok] + draft], dtype=torch.long, device=device)
                    # Extend attention mask for cached + new tokens
                    spec_attn = torch.ones(
                        1, cur_len + K, dtype=torch.long, device=device)

                    out = model(
                        input_ids=spec_ids,
                        attention_mask=spec_attn,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_kv = out.past_key_values
                    n_spec_tot += K

                    n_acc = 0
                    stopped = False
                    for i in range(K):
                        if len(generated_ids) >= max_new_tokens:
                            stopped = True
                            break
                        pred = int(out.logits[0, i].argmax())
                        if pred == draft[i]:
                            generated_ids.append(draft[i])
                            all_ctx.append(draft[i])
                            n_acc += 1
                            n_spec_acc += 1
                            if draft[i] == eos_id:
                                next_tok = draft[i]
                                stopped = True
                                break
                        else:
                            generated_ids.append(pred)
                            all_ctx.append(pred)
                            next_tok = pred
                            n_acc += 1
                            stopped = True
                            break

                    if not stopped and len(generated_ids) < max_new_tokens:
                        bonus = int(out.logits[0, K].argmax())
                        generated_ids.append(bonus)
                        all_ctx.append(bonus)
                        next_tok = bonus
                        n_acc += 1

                    cur_len += n_acc

                else:
                    # Greedy step — no draft available
                    cur_attn = torch.ones(
                        1, cur_len, dtype=torch.long, device=device)
                    out = model(
                        input_ids=torch.tensor(
                            [[next_tok]], dtype=torch.long, device=device),
                        attention_mask=cur_attn,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_kv = out.past_key_values
                    next_tok = int(out.logits[0, -1].argmax())
                    generated_ids.append(next_tok)
                    all_ctx.append(next_tok)
                    cur_len += 1

        torch.cuda.synchronize()
        t2 = time.time()

        n_tokens = max(len(generated_ids), 1)
        decode_tps = max(n_tokens - 1, 1) / (t2 - t1) if t2 > t1 else 0.0
        stats = {"n_tokens": n_tokens, "decode_tps": decode_tps}
        if n_spec_tot > 0:
            stats["spec_accept_rate"] = n_spec_acc / n_spec_tot

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text, stats

    # ── non-speculative: model.generate() ────────────────────────────────────
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_id,
        )
    torch.cuda.synchronize()
    total_s = time.time() - t0

    generated = output_ids[0][seq_len:]
    n_tokens = max(len(generated), 1)
    decode_tps = n_tokens / total_s if total_s > 0 else 0.0
    stats = {"n_tokens": n_tokens, "decode_tps": decode_tps}

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLaMA-2-7B GPU inference benchmark — speculative decoding + INT8")
    parser.add_argument("--model", required=True,
                        help="Path to LLaMA-2-7B HF checkpoint directory")
    parser.add_argument("--prompt", default=None,
                        help="Single prompt string for quick test")
    parser.add_argument("--prompts_file", default=None,
                        help="Text file with one prompt per line for batch benchmark")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of prompts to use from prompts_file")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--quant", default="int8",
                        choices=["bf16", "fp16", "int8", "int4"],
                        help="Quantization: int8 (default) | int4 | bf16 | fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speculative", action="store_true",
                        help="Enable n-gram speculative decoding")
    parser.add_argument("--gamma", type=int, default=5,
                        help="Speculative lookahead window (default: 5)")
    parser.add_argument("--output", default="llama_gpu_results.json")
    parser.add_argument("--save_samples", action="store_true")
    args = parser.parse_args()
    args.model = os.path.abspath(args.model)

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found.")
        raise SystemExit(1)

    torch.cuda.reset_peak_memory_stats(args.device)
    model, tokenizer = load_llama_gpu(args.model, quant=args.quant, device=args.device)
    gpu_mem_load_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
    print(f"  GPU VRAM after load: {gpu_mem_load_gb:.2f} GB")

    def _infer(prompt):
        return generate_gpu(model, tokenizer, prompt,
                            max_new_tokens=args.max_new_tokens,
                            device=args.device,
                            speculative=args.speculative,
                            gamma=args.gamma)

    # ── single prompt ─────────────────────────────────────────────────────────
    if args.prompt:
        t0 = time.time()
        text, stats = _infer(args.prompt)
        print(f"\nGenerated ({time.time()-t0:.2f}s, {stats['decode_tps']:.1f} tok/s):")
        print(f"  {text}")
        if "spec_accept_rate" in stats:
            print(f"  Spec accept rate: {stats['spec_accept_rate']:.1%}")
        return

    # ── prompts file benchmark ────────────────────────────────────────────────
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        prompts = prompts[:args.num_samples]

        # Warm-up
        print("Warming up GPU …")
        _infer(prompts[0])
        torch.cuda.reset_peak_memory_stats(args.device)

        latencies, samples_out = [], []
        for i, prompt in enumerate(prompts):
            t0 = time.time()
            text, stats = _infer(prompt)
            elapsed = time.time() - t0
            latencies.append(elapsed)
            print(f"  [{i+1:3d}/{len(prompts)}] {elapsed:5.2f}s  "
                  f"{stats['decode_tps']:6.1f} tok/s | {text[:60]}")
            samples_out.append({"idx": i, "prompt": prompt, "prediction": text,
                                 "latency_s": elapsed, **stats})

        avg_lat = float(np.mean(latencies))
        avg_tps = float(np.mean([s["decode_tps"] for s in samples_out]))
        gpu_peak_gb = torch.cuda.max_memory_allocated(args.device) / 1e9
        acc_rates = [s["spec_accept_rate"] for s in samples_out
                     if "spec_accept_rate" in s]
        avg_acc = float(np.mean(acc_rates)) if acc_rates else None

        print(f"\n{'='*60}")
        print(f"Avg latency:   {avg_lat:.2f} s/sample")
        print(f"Avg decode:    {avg_tps:.2f} tok/s")
        if avg_acc is not None:
            print(f"Spec acc rate: {avg_acc:.1%}")
        print(f"GPU VRAM peak: {gpu_peak_gb:.2f} GB")
        print(f"quant:         {args.quant}")
        print(f"speculative:   {args.speculative}  gamma={args.gamma}")
        print(f"{'='*60}")

        result = {
            "model":                args.model,
            "quant_method":         args.quant,
            "device":               args.device,
            "num_samples":          len(prompts),
            "speculative":          args.speculative,
            "gamma":                args.gamma if args.speculative else None,
            "avg_latency_s":        avg_lat,
            "avg_decode_tps":       avg_tps,
            "avg_spec_accept_rate": avg_acc,
            "gpu_mem_load_gb":      gpu_mem_load_gb,
            "gpu_mem_peak_gb":      gpu_peak_gb,
            "latencies":            latencies,
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
