#!/usr/bin/env python3
"""Patch existing bench JSONs with audio_file paths from demo_audio/.

Matches by index: sample 0 → sample_000.wav, sample 1 → sample_001.wav, etc.
Usage:
    python patch_audio.py demo_audio/ eagle_bench_bf16_nospec.json eagle_bench_w4a16_eagle_K4.json
"""
import json, os, sys
from pathlib import Path

audio_dir = Path(sys.argv[1])
json_files = sys.argv[2:]

wavs = sorted(audio_dir.glob("*.wav"))
if not wavs:
    print(f"No WAV files in {audio_dir}")
    sys.exit(1)
print(f"Found {len(wavs)} audio files in {audio_dir}")

for jf in json_files:
    if not os.path.exists(jf):
        print(f"  [skip] {jf} not found")
        continue
    with open(jf) as f:
        d = json.load(f)

    samples = d.get("samples", [])
    if not samples and "results" in d:
        # EAGLE format — normalize first
        samples = [
            {"prediction": r.get("hyp", ""), "reference": r.get("ref", ""),
             "latency_s": r.get("lat_s", 0.0), **r.get("stats", {})}
            for r in d["results"]
        ]
        d["samples"] = samples

    patched = 0
    for i, s in enumerate(samples):
        if i < len(wavs):
            s["audio_file"] = str(wavs[i])
            patched += 1

    with open(jf, "w") as f:
        json.dump(d, f, indent=2)
    print(f"  Patched {patched}/{len(samples)} samples in {jf}")
