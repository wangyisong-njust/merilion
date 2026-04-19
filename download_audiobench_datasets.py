"""Download AudioBench Singlish ASR datasets from HuggingFace Hub to local disk.

Saves each split under dataset_root/ASR/<name>/ so eval_wer_batch.py can load
them with load_from_disk() the same way it loads existing local datasets.

Usage:
    python download_audiobench_datasets.py --dataset_root /home/jinchao/runtao/meralion_datasets/ASR
"""
import argparse
import os
from datasets import load_dataset

DATASETS = [
    ("imda_part1_asr_test",    "ASR-PART1-Test"),
    ("imda_part2_asr_test",    "ASR-PART2-Test"),
    ("imda_part3_30s_asr_test","ASR-PART3-Test"),
    ("imda_part4_30s_asr_test","ASR-PART4-Test"),
    ("imda_part5_30s_asr_test","ASR-PART5-Test"),
    ("imda_part6_30s_asr_test","ASR-PART6-Test"),
]
HF_REPO = "MERaLiON/Multitask-National-Speech-Corpus-v1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root",
                        default="/home/jinchao/runtao/meralion_datasets/ASR")
    parser.add_argument("--parts", nargs="+", default=None,
                        help="Which parts to download, e.g. 1 3 (default: all)")
    args = parser.parse_args()

    wanted = set(args.parts) if args.parts else None

    for name, data_dir in DATASETS:
        part_num = data_dir.split("PART")[1].split("-")[0]  # "1", "2", ...
        if wanted and part_num not in wanted:
            continue

        out_path = os.path.join(args.dataset_root, name)
        if os.path.exists(out_path):
            print(f"[skip] {out_path} already exists")
            continue

        print(f"Downloading {HF_REPO}  data_dir={data_dir} ...")
        ds = load_dataset(HF_REPO, data_dir=data_dir, trust_remote_code=True)["train"]
        print(f"  {len(ds)} samples → saving to {out_path}")
        ds.save_to_disk(out_path)
        print(f"  Done.")

if __name__ == "__main__":
    main()
