"""Download a slice of a HuggingFace dataset and persist only that slice
to disk (via save_to_disk), so the final artefact contains only the
samples you actually want — no full local copy required.

Example:
    python download_subset.py \\
        --repo AudioLLMs/imda-part1-asr \\
        --split train \\
        --start_idx 999999 --num_samples 20000 \\
        --output /home/.../IMDA_PART1_post999999

    # Multiple subsets in one go — each saved to its own <output>/<subset>
    python download_subset.py \\
        --repo AudioLLMs/audiobench-asr \\
        --subsets imda_part1 imda_part3 \\
        --split train \\
        --start_idx 999999 --num_samples 20000 \\
        --output /home/.../audiobench_train_post999999

After the run, the saved directories contain just the requested samples.
The HF download cache (~/.cache/huggingface/datasets) still holds the
streamed shards during execution; clean it afterwards with
`rm -rf ~/.cache/huggingface/datasets` if you want to reclaim that too.
"""
import argparse
import os
import sys
import time


def iter_subset(repo, subset, split, start_idx, num_samples):
    """Stream from HF Hub, skip to start_idx, yield the next num_samples rows.

    Streaming downloads parquet shards on the fly, so we don't need the
    whole dataset locally — the parser just reads forward and discards
    everything before start_idx."""
    from datasets import load_dataset
    kwargs = {"split": split, "streaming": True}
    if subset:
        kwargs["name"] = subset
    ds = load_dataset(repo, **kwargs)
    ds = ds.skip(start_idx).take(num_samples)
    yielded = 0
    t0 = time.time()
    for row in ds:
        yielded += 1
        if yielded % 100 == 0:
            dt = time.time() - t0
            print(f"    [{yielded}/{num_samples}]  "
                  f"rate={yielded/max(dt,1e-6):.1f}/s", flush=True)
        yield row


def save_subset(repo, subset, split, start_idx, num_samples, out_dir):
    from datasets import Dataset
    print(f"streaming {repo} ({subset or '-'}) split={split} "
          f"[{start_idx}, {start_idx + num_samples}) → {out_dir}")
    rows = list(iter_subset(repo, subset, split, start_idx, num_samples))
    if not rows:
        print("  (no rows)"); return
    ds = Dataset.from_list(rows)
    os.makedirs(os.path.dirname(out_dir) or ".", exist_ok=True)
    ds.save_to_disk(out_dir)
    # Size on disk
    total = 0
    for root, _, files in os.walk(out_dir):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    print(f"  saved {len(rows)} rows to {out_dir} "
          f"({total/1e6:.1f} MB on disk)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo",    required=True,
                    help="HF Hub dataset repo id, e.g. AudioLLMs/imda-part1")
    ap.add_argument("--subsets", nargs="*", default=[None],
                    help="Optional subset (`name=`) list.  Pass nothing "
                         "for single-config datasets.")
    ap.add_argument("--split",   default="train")
    ap.add_argument("--start_idx",   type=int, default=999999,
                    help="Skip the first N samples (eval holdout).  "
                         "Default 999999 matches build_ngram_corpus convention.")
    ap.add_argument("--num_samples", type=int, default=20000,
                    help="How many samples to keep per subset.")
    ap.add_argument("--output", required=True,
                    help="Output directory.  With multiple --subsets, each "
                         "subset is saved under <output>/<subset>.")
    ap.add_argument("--clear_cache", action="store_true",
                    help="rm -rf ~/.cache/huggingface/datasets after saving "
                         "to reclaim the streamed-shard cache.")
    args = ap.parse_args()

    subsets = args.subsets if args.subsets else [None]
    if len(subsets) == 1 and subsets[0] is None:
        save_subset(args.repo, None, args.split,
                    args.start_idx, args.num_samples, args.output)
    else:
        for s in subsets:
            subout = os.path.join(args.output, s) if s else args.output
            save_subset(args.repo, s, args.split,
                        args.start_idx, args.num_samples, subout)

    if args.clear_cache:
        import shutil
        cache = os.path.expanduser("~/.cache/huggingface/datasets")
        if os.path.isdir(cache):
            size = sum(os.path.getsize(os.path.join(r, f))
                       for r, _, fs in os.walk(cache) for f in fs)
            shutil.rmtree(cache)
            print(f"cleared {cache} ({size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
