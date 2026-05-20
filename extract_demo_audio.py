#!/usr/bin/env python3
"""Extract audio from dataset into demo_audio/ for HTML demo.

Saves sample_000.wav, sample_001.wav, ... matching the bench's sample order
(data.select(range(0, N)), no shuffle).

Usage:
    python extract_demo_audio.py \
        --dataset /path/to/imda_part1_asr_test \
        --num_samples 20 \
        --output_dir demo_audio
"""
import argparse, os
import numpy as np
import soundfile as sf
from datasets import load_from_disk


def _extract_audio(sample):
    c = sample.get("context")
    if isinstance(c, dict):
        if "array" in c:
            return c
        if isinstance(c.get("audio"), dict):
            return c["audio"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--output_dir", default="demo_audio")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_from_disk(os.path.abspath(args.dataset))
    end = min(args.num_samples, len(data))
    subset = data.select(range(0, end))

    saved = 0
    for i, s in enumerate(subset):
        ao = _extract_audio(s)
        if ao is None or "array" not in ao:
            print(f"  [{i}] no audio, skipping")
            continue
        arr = np.asarray(ao["array"], dtype=np.float32)
        sr = ao.get("sampling_rate", 16000)
        path = os.path.join(args.output_dir, f"sample_{i:03d}.wav")
        sf.write(path, arr, sr)
        saved += 1

    print(f"Saved {saved} audio files to {args.output_dir}/")


if __name__ == "__main__":
    main()
