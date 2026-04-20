"""Build a token-level n-gram index from IMDA dataset reference transcriptions.

Loads each local IMDA part from index 999999 onward, tokenizes the reference
text with the MERaLiON tokenizer, and builds a prefix→next-token dict.

Saves: <output>.pkl  containing {"index": {prefix_tuple: most_common_next}, "n": int}

Usage:
    python build_ngram_corpus.py \
        --model /path/to/MERaLiON-2-3B \
        --datasets /path/ASR/IMDA_PART1_mono_en_30_ASR /path/ASR/IMDA_PART3_conv_en_30_ASR \
        --samples_per_part 50000 \
        --ngram_sizes 2 3 4 \
        --output ngram_corpus.pkl
"""
import argparse
import os
import pickle
from collections import Counter, defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Local dataset paths (load_from_disk)")
    parser.add_argument("--samples_per_part", type=int, default=50000)
    parser.add_argument("--start_idx", type=int, default=999999,
                        help="Skip first N samples (keep eval set separate)")
    parser.add_argument("--ngram_sizes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--output", default="ngram_corpus.pkl")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from datasets import load_from_disk

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # counts[prefix_tuple] = Counter({next_tok: freq})
    counts = defaultdict(Counter)
    total_tokens = 0

    for ds_path in args.datasets:
        print(f"\nDataset: {ds_path}")
        data = load_from_disk(os.path.abspath(ds_path))
        start = min(args.start_idx, len(data))
        end   = min(start + args.samples_per_part, len(data))
        subset = data.select(range(start, end))
        print(f"  Using samples [{start}, {end})  →  {len(subset)} samples")

        for i, sample in enumerate(subset):
            ref = sample["other_attributes"]["Transcription"]
            if not ref or not ref.strip():
                continue
            ids = tokenizer.encode(ref, add_special_tokens=False)
            if len(ids) < 2:
                continue
            for n in args.ngram_sizes:
                for j in range(len(ids) - n + 1):
                    prefix = tuple(ids[j : j + n - 1])
                    next_t = ids[j + n - 1]
                    counts[prefix][next_t] += 1
            total_tokens += len(ids)
            if (i + 1) % 10000 == 0:
                print(f"  [{i+1}/{len(subset)}]  prefixes so far: {len(counts):,}")

    # Keep only the most-common next token per prefix
    index = {prefix: ctr.most_common(1)[0][0] for prefix, ctr in counts.items()}

    print(f"\nTotal tokens processed : {total_tokens:,}")
    print(f"Unique n-gram prefixes : {len(index):,}")
    print(f"N-gram sizes           : {args.ngram_sizes}")

    out = {"index": index, "ngram_sizes": args.ngram_sizes}
    with open(args.output, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"Saved → {args.output}")

if __name__ == "__main__":
    main()
