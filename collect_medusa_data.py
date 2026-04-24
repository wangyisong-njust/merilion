"""Collect (hidden_state, output_token) pairs for Medusa head training.

For each audio sample we run the full inference path (speech encoder + adapter
+ text decoder), capture the text decoder's LAST-LAYER hidden state at every
decode position, and save (hidden_states, token_ids) to disk.

Training on these lets the heads see the same hidden-state distribution they
encounter at inference — unlike training on pure text embeddings, which
mismatches the audio-conditioned decoding path.

Output format: one .pt file per shard containing a list of dicts
    {"tokens": [int], "hidden": FloatTensor[T, H]}  # T = #decode tokens
Each shard is written incrementally so a crash can resume.

Supports --shard_id / --num_shards for multi-GPU parallel collection.
Supports `--datasets path1 path2 ...` to concatenate multiple IMDA parts
(round-robin across them up to `--num_samples` total).

Usage (single dataset):
    python collect_medusa_data.py \\
        --model ... \\
        --datasets /path/to/IMDA_PART1 \\
        --num_samples 10000 \\
        --shard_id 0 --num_shards 3 \\
        --output_shard medusa_data_shard_0.pt

Usage (multiple parts, mixed round-robin):
    python collect_medusa_data.py \\
        --model ... \\
        --datasets /path/to/IMDA_PART1 /path/to/IMDA_PART2 /path/to/IMDA_PART3 \\
        --num_samples 30000 \\
        --num_samples_per_dataset 15000  \\   # optional cap per dataset
        --shard_id 0 --num_shards 3 \\
        --output_shard medusa_data_shard_0.pt
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch


def collect_shard(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from infer_gpu import load_model_gpu, prepare_audio, SAMPLE_RATE
    from transformers import DynamicCache
    from datasets import load_from_disk

    device = args.device

    print(f"[shard {args.shard_id}/{args.num_shards}] loading verifier …")
    t0 = time.time()
    model, processor = load_model_gpu(
        args.model, quant="bf16", flash_attn=True, device=device)
    model.eval()
    tokenizer = processor.tokenizer
    _dtype = next(p.dtype for p in model.parameters()
                  if p.dtype in (torch.float16, torch.bfloat16))
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Load one or more datasets (round-robin mixed into one shared pool
    # so `num_samples` is the total budget across all of them).
    print(f"[shard {args.shard_id}] loading {len(args.datasets)} dataset(s) …")
    datasets = []
    for p in args.datasets:
        d = load_from_disk(os.path.abspath(p))
        # Same eval-exclusion convention as build_ngram_corpus.py / the old
        # Medusa data collector: skip the first `start_idx` samples so
        # training data never overlaps the held-out eval subset.
        start = min(args.start_idx, len(d))
        end   = min(start + args.num_samples_per_dataset, len(d)) \
                if args.num_samples_per_dataset > 0 else len(d)
        datasets.append((p, d, start, end))
        print(f"  {os.path.basename(p)}: using samples [{start}, {end})")

    # Build a global index list = round-robin over datasets.
    # Each global step emits (dataset_idx, local_idx) so that shards
    # get the same deterministic subset regardless of which dataset
    # each sample comes from.
    combined = []  # list of (ds_idx, local_idx)
    cursors = [start for (_, _, start, _) in datasets]
    ends    = [end   for (_, _, _,     end) in datasets]
    budget  = args.num_samples
    di = 0
    while len(combined) < budget:
        if all(cursors[i] >= ends[i] for i in range(len(datasets))):
            break
        if cursors[di] < ends[di]:
            combined.append((di, cursors[di]))
            cursors[di] += 1
        di = (di + 1) % len(datasets)

    # Stride-split across shards.
    my_pairs = combined[args.shard_id::args.num_shards]
    print(f"  combined pool {len(combined)} samples  shard {args.shard_id} "
          f"takes {len(my_pairs)} (stride {args.num_shards})")

    speech_token_id = model.config.speech_token_index
    eos_ids = {tokenizer.eos_token_id,
               tokenizer.convert_tokens_to_ids("<end_of_turn>")}
    eos_ids.discard(None)

    out_samples = []
    t_start = time.time()
    for idx_in_shard, (ds_idx, idx) in enumerate(my_pairs):
        sample = datasets[ds_idx][1][idx]
        # Two schemas in the wild:
        #   (a) old IMDA:  sample["context"] = {"audio": {array, sampling_rate}, ...}
        #   (b) AudioBench NSC-v1-extend:  sample["context"] IS the audio
        #       ({array, sampling_rate}), no nested .audio key.
        ctx = sample.get("context")
        ao = None
        if isinstance(ctx, dict):
            if "array" in ctx:           # schema (b)
                ao = ctx
            elif "audio" in ctx:          # schema (a)
                a = ctx["audio"]
                if isinstance(a, dict):
                    ao = a
        if ao is None:
            continue
        arr = ao.get("array")
        sr  = ao.get("sampling_rate", SAMPLE_RATE)
        if arr is None:
            continue
        audio = np.asarray(arr, dtype=np.float32)

        try:
            input_features, feature_attention_mask, n_speech = prepare_audio(
                audio, sr, processor)
            conv = [{"role": "user",
                     "content": (f"Instruction: Transcribe the speech \n"
                                 "Follow the text instruction based on the "
                                 "following audio: <SpeechHere>")}]
            prompt = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True)
            raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
            pos = raw_ids.index(speech_token_id)
            input_ids = torch.tensor(
                [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos+1:]],
                dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            input_features = input_features.to(device).to(_dtype)
            feature_attention_mask = feature_attention_mask.to(device)

            # Use DynamicCache (original model, not pruned) — easiest to work with
            # for capturing hidden states across positions.
            past_kv = DynamicCache()
            with torch.inference_mode():
                seq_len = input_ids.shape[1]
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
                # Hidden state at last prefill position (for predicting 1st output tok)
                h_prev = out.hidden_states[-1][0, -1, :].to(torch.bfloat16).cpu()
                next_tok = int(out.logits[0, -1].argmax())

                tokens  = [next_tok]
                hiddens = [h_prev]   # h_prev corresponds to position seq_len-1, predicts token at seq_len

                cur_pos = seq_len
                while len(tokens) < args.max_new_tokens and next_tok not in eos_ids:
                    v = model(
                        input_ids=torch.tensor([[next_tok]], dtype=torch.long, device=device),
                        attention_mask=torch.ones(1, cur_pos + 1, dtype=torch.long, device=device),
                        past_key_values=past_kv,
                        use_cache=True,
                        cache_position=torch.tensor([cur_pos], device=device),
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    h_cur = v.hidden_states[-1][0, -1, :].to(torch.bfloat16).cpu()
                    next_tok = int(v.logits[0, -1].argmax())
                    tokens.append(next_tok)
                    hiddens.append(h_cur)
                    cur_pos += 1

                # Structure: hiddens[i] is at position (seq_len-1+i), predicts tokens[i].
                # For training Medusa head_k (offset +k+1), we want:
                #   input  = hiddens[i]
                #   target = tokens[i + k + 1]   (if i + k + 1 < len(tokens))
                hiddens_t = torch.stack(hiddens, dim=0)  # (T, H)
                out_samples.append({
                    "tokens":  tokens,
                    "hiddens": hiddens_t,
                })
        except Exception as e:
            print(f"[shard {args.shard_id}] sample {idx} FAILED: {e}", flush=True)
            continue

        if (idx_in_shard + 1) % 50 == 0 or (idx_in_shard + 1) == len(my_pairs):
            dt = time.time() - t_start
            rate = (idx_in_shard + 1) / dt if dt > 0 else 0
            eta = (len(my_pairs) - (idx_in_shard + 1)) / rate if rate > 0 else 0
            print(f"[shard {args.shard_id}] [{idx_in_shard+1}/{len(my_pairs)}] "
                  f"rate={rate:.2f}/s  eta={eta/60:.1f}m  "
                  f"saved={len(out_samples)}", flush=True)

        # Incremental save every 500 samples to resist crashes.
        if (idx_in_shard + 1) % 500 == 0:
            torch.save(out_samples, args.output_shard)

    torch.save(out_samples, args.output_shard)
    total_tokens = sum(len(s["tokens"]) for s in out_samples)
    print(f"\n[shard {args.shard_id}] done. samples={len(out_samples)}  "
          f"total_decode_tokens={total_tokens:,}  "
          f"saved → {args.output_shard}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    # Accept one or more IMDA dataset dirs — samples are round-robin'd
    # across them up to --num_samples total.  Legacy `--dataset` still
    # works (a single path).
    ap.add_argument("--datasets", nargs="+",
                    help="Dataset directory paths (one or more, load_from_disk)")
    ap.add_argument("--dataset", help="(legacy) single dataset path; use --datasets instead")
    ap.add_argument("--num_samples", type=int, default=10000,
                    help="Total samples across all datasets (round-robin)")
    ap.add_argument("--num_samples_per_dataset", type=int, default=0,
                    help="If >0, cap each dataset's contribution to this many "
                         "(after --start_idx offset).  0 = use full range.")
    ap.add_argument("--start_idx",   type=int, default=30,
                    help="Skip the first N samples of each dataset (eval holdout)")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--shard_id",    type=int, default=0)
    ap.add_argument("--num_shards",  type=int, default=1)
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--output_shard", default="medusa_data_shard.pt")
    args = ap.parse_args()

    # Normalize: support both --dataset (singular, legacy) and --datasets
    if not args.datasets and args.dataset:
        args.datasets = [args.dataset]
    if not args.datasets:
        ap.error("must pass --datasets <path ...> (or legacy --dataset <path>)")

    collect_shard(args)


if __name__ == "__main__":
    main()
