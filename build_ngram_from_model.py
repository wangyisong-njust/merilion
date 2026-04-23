"""Build a token-level n-gram index from a MODEL's OUTPUT tokens.

Contrast with build_ngram_corpus.py, which tokenizes REFERENCE transcriptions:
that corpus poorly matches the verifier's actual output distribution (tokenizer
quirks, casing, punctuation), giving low acceptance rates.

This script runs the same model that will later serve as the verifier over the
training audios, captures the generated token sequences, and builds an n-gram
index from those. Expected effect: 3-5× acceptance-rate improvement.

Supports --shard_id / --num_shards for trivial multi-GPU parallelism (each
shard writes a partial pkl; aggregate with --mode merge).

Usage (single shard):
    python build_ngram_from_model.py \\
        --model /home/kaixin/yisong/merilion/meralion_tune_log/MERaLiON-2-3B-v3-td50-mid3-23-tune \\
        --dataset /home/kaixin/meralion_datasets/train/ASR/IMDA_PART1_mono_en_30_ASR \\
        --num_samples 10000 --start_idx 30 \\
        --ngram_sizes 2 3 4 \\
        --output_shard ngram_model_shard_0.pkl \\
        --shard_id 0 --num_shards 3

Usage (merge):
    python build_ngram_from_model.py --mode merge \\
        --shards ngram_model_shard_0.pkl ngram_model_shard_1.pkl ngram_model_shard_2.pkl \\
        --output ngram_corpus_model.pkl
"""
import argparse
import os
import pickle
import sys
import time
from collections import Counter, defaultdict


def build_shard(args):
    import torch
    import numpy as np
    from datasets import load_from_disk
    from transformers import AutoProcessor

    # Use the project's inference path (same model-loading semantics as
    # infer_gpu.py's --quant bf16 path; keeps outputs consistent with verifier).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from infer_gpu import load_model_gpu, prepare_audio, SAMPLE_RATE

    print(f"[shard {args.shard_id}/{args.num_shards}] loading model from {args.model}")
    model, processor = load_model_gpu(
        args.model, quant="bf16", flash_attn=True, device=args.device)
    model.eval()
    tokenizer = processor.tokenizer

    print(f"[shard {args.shard_id}] loading dataset from {args.dataset}")
    data = load_from_disk(os.path.abspath(args.dataset))

    start = min(args.start_idx, len(data))
    end   = min(start + args.num_samples, len(data))
    # Deterministic shard split by stride.
    my_indices = list(range(start + args.shard_id, end, args.num_shards))
    print(f"[shard {args.shard_id}] total sample range [{start}, {end}), "
          f"this shard: {len(my_indices)} samples (stride {args.num_shards})")

    counts = defaultdict(Counter)  # prefix_tuple → Counter({next_tok: freq})
    total_tokens = 0
    t0 = time.time()
    n_done = 0

    instruction = "Transcribe the speech"

    for i, idx in enumerate(my_indices):
        sample = data[idx]
        ctx = sample.get("context") or {}
        audio_obj = ctx.get("audio") if isinstance(ctx, dict) else None
        if audio_obj is None:
            continue
        # context.audio may be a dict {array, sampling_rate} or a direct dict
        arr = audio_obj.get("array") if isinstance(audio_obj, dict) else None
        sr  = audio_obj.get("sampling_rate") if isinstance(audio_obj, dict) else SAMPLE_RATE
        if arr is None:
            continue
        audio = np.asarray(arr, dtype=np.float32)

        try:
            # Build prompt + inputs like transcribe_gpu (no-spec path).
            input_features, feature_attention_mask, n_speech = prepare_audio(
                audio, sr, processor)
            speech_token_id = model.config.speech_token_index
            conversation = [{"role": "user",
                             "content": (f"Instruction: {instruction} \n"
                                         "Follow the text instruction based on the "
                                         "following audio: <SpeechHere>")}]
            prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True)
            raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
            try:
                pos = raw_ids.index(speech_token_id)
            except ValueError:
                continue
            input_ids = torch.tensor(
                [raw_ids[:pos] + [speech_token_id] * n_speech + raw_ids[pos + 1:]],
                dtype=torch.long, device=args.device)
            attention_mask = torch.ones_like(input_ids)
            input_features = input_features.to(args.device).to(torch.bfloat16)
            feature_attention_mask = feature_attention_mask.to(args.device)

            # Manual greedy decode — pruned mid3-23 has non-uniform kv_heads,
            # so HF's model.generate() HybridCache path breaks at index_copy_.
            # Use DynamicCache via the same pattern as infer_gpu.py's non-spec
            # transcribe path.
            from transformers import DynamicCache
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
                    cache_position=torch.arange(0, seq_len, device=args.device),
                    return_dict=True,
                )
                next_tok = int(out.logits[0, -1].argmax())
                gen = [next_tok]
                eos_ids = {tokenizer.eos_token_id,
                           tokenizer.convert_tokens_to_ids("<end_of_turn>")}
                eos_ids.discard(None)

                cur_pos = seq_len
                while len(gen) < args.max_new_tokens and next_tok not in eos_ids:
                    out = model(
                        input_ids=torch.tensor([[next_tok]], dtype=torch.long,
                                               device=args.device),
                        attention_mask=torch.ones(1, cur_pos + 1,
                                                  dtype=torch.long,
                                                  device=args.device),
                        past_key_values=past_kv,
                        use_cache=True,
                        cache_position=torch.tensor([cur_pos], device=args.device),
                        return_dict=True,
                    )
                    next_tok = int(out.logits[0, -1].argmax())
                    gen.append(next_tok)
                    cur_pos += 1
        except Exception as e:
            print(f"[shard {args.shard_id}] sample {idx} FAILED: {e}", flush=True)
            continue

        # Strip trailing pad / eos for cleaner n-grams (optional).
        eos_ids = {tokenizer.eos_token_id,
                   tokenizer.convert_tokens_to_ids("<end_of_turn>")}
        eos_ids.discard(None)
        # Keep up to (including) first EOS so model's stop behavior is captured.
        for j, t in enumerate(gen):
            if t in eos_ids:
                gen = gen[: j + 1]
                break
        if len(gen) < 2:
            continue

        total_tokens += len(gen)
        for n in args.ngram_sizes:
            if n < 2 or len(gen) < n:
                continue
            for j in range(len(gen) - n + 1):
                prefix = tuple(gen[j: j + n - 1])
                counts[prefix][gen[j + n - 1]] += 1

        n_done += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(my_indices):
            dt = time.time() - t0
            rate = (i + 1) / dt if dt > 0 else 0
            eta = (len(my_indices) - (i + 1)) / rate if rate > 0 else 0
            print(f"[shard {args.shard_id}] [{i+1}/{len(my_indices)}]  "
                  f"prefixes={len(counts):,}  tokens={total_tokens:,}  "
                  f"rate={rate:.2f}/s  eta={eta/60:.1f}m", flush=True)

    # Reduce counts → most_common next token per prefix.
    index = {p: c.most_common(1)[0][0] for p, c in counts.items() if c}

    out = {
        "index": index,
        "ngram_sizes": args.ngram_sizes,
        "source_model": args.model,
        "source_dataset": args.dataset,
        "n_samples": n_done,
        "total_tokens": total_tokens,
    }

    with open(args.output_shard, "wb") as f:
        pickle.dump(out, f)
    print(f"\n[shard {args.shard_id}] done.")
    print(f"  samples processed  : {n_done}")
    print(f"  total tokens       : {total_tokens:,}")
    print(f"  unique prefixes    : {len(index):,}")
    print(f"  saved → {args.output_shard}")


def merge(args):
    merged_counts = defaultdict(Counter)
    ngram_sizes = None
    total_tokens = 0
    n_samples = 0
    for p in args.shards:
        print(f"loading {p}")
        with open(p, "rb") as f:
            d = pickle.load(f)
        # Shards saved only top-1; for correct merging we'd need counters.
        # Fallback: treat each top-1 as a +1 vote (lossy but simple).
        for prefix, tok in d["index"].items():
            merged_counts[prefix][tok] += 1
        ngram_sizes = d["ngram_sizes"]
        total_tokens += d.get("total_tokens", 0)
        n_samples    += d.get("n_samples", 0)

    index = {p: c.most_common(1)[0][0] for p, c in merged_counts.items() if c}

    out = {
        "index": index,
        "ngram_sizes": ngram_sizes,
        "n_samples": n_samples,
        "total_tokens": total_tokens,
        "source_shards": args.shards,
    }
    with open(args.output, "wb") as f:
        pickle.dump(out, f)
    print(f"\nmerged shards     : {len(args.shards)}")
    print(f"total samples     : {n_samples}")
    print(f"total tokens      : {total_tokens:,}")
    print(f"unique prefixes   : {len(index):,}")
    print(f"saved → {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "merge"], default="build")
    # build mode
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--start_idx", type=int, default=30,
                        help="Skip first N eval samples")
    parser.add_argument("--ngram_sizes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_shard", default="ngram_model_shard.pkl")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    # merge mode
    parser.add_argument("--shards", nargs="+")
    parser.add_argument("--output", default="ngram_corpus_model.pkl")

    args = parser.parse_args()

    if args.mode == "build":
        if not args.model or not args.dataset:
            parser.error("--model and --dataset required for build mode")
        build_shard(args)
    else:
        if not args.shards:
            parser.error("--shards required for merge mode")
        merge(args)


if __name__ == "__main__":
    main()
