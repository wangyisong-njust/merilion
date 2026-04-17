#!/bin/bash
# ============================================================
# GPU speculative decoding benchmark
# Compares BF16 no-spec vs n-gram +spec (γ=5) on GPU for:
#   - Original MERaLiON-2-3B
#   - Pruned mid3-22 / mid3-23 / mid4-23
# ============================================================
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
TUNE_ROOT="meralion_tune_log"
NUM_SAMPLES=50
GAMMA=5
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

MID3_22="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-22-tune"
MID3_23="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-23-tune"
MID4_23="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid4-23-tune"

run_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then echo "  [skip] $json already exists"; return 0; fi
    echo "  running → $json"
    "$PYTHON_PATH" -u infer_gpu.py --output "$json" "$@" \
        | tee "${json%.json}.log" \
        || { echo "[FAIL] $json"; return 1; }
}

# ════════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "  Original MERaLiON-2-3B"
echo "========================================"

echo ""
echo "--- BF16 no-spec ---"
run_if_missing "gpu_bf16_original_nospec.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant bf16 || exit 1

echo ""
echo "--- BF16 +spec γ=${GAMMA} ---"
run_if_missing "gpu_bf16_original_spec${GAMMA}.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant bf16 \
    --speculative --gamma "$GAMMA" || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Pruned mid3-22"
echo "========================================"

echo ""
echo "--- BF16 no-spec ---"
run_if_missing "gpu_bf16_mid3-22_nospec.json" \
    --model "$MID3_22" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant bf16 || exit 1

echo ""
echo "--- BF16 +spec γ=${GAMMA} ---"
run_if_missing "gpu_bf16_mid3-22_spec${GAMMA}.json" \
    --model "$MID3_22" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant bf16 \
    --speculative --gamma "$GAMMA" || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Pruned mid3-23"
echo "========================================"

echo ""
echo "--- BF16 no-spec ---"
run_if_missing "gpu_bf16_mid3-23_nospec.json" \
    --model "$MID3_23" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant bf16 || exit 1

echo ""
echo "--- BF16 +spec γ=${GAMMA} ---"
run_if_missing "gpu_bf16_mid3-23_spec${GAMMA}.json" \
    --model "$MID3_23" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant bf16 \
    --speculative --gamma "$GAMMA" || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Pruned mid4-23"
echo "========================================"

echo ""
echo "--- BF16 no-spec ---"
run_if_missing "gpu_bf16_mid4-23_nospec.json" \
    --model "$MID4_23" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant bf16 || exit 1

echo ""
echo "--- BF16 +spec γ=${GAMMA} ---"
run_if_missing "gpu_bf16_mid4-23_spec${GAMMA}.json" \
    --model "$MID4_23" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant bf16 \
    --speculative --gamma "$GAMMA" || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Summary"
echo "========================================"

"$PYTHON_PATH" - <<'PYEOF'
import json, os

rows = [
    ("Original   no-spec",    "gpu_bf16_original_nospec.json",  False),
    ("Original   +spec γ=5",  "gpu_bf16_original_spec5.json",   True),
    ("mid3-22    no-spec",    "gpu_bf16_mid3-22_nospec.json",   False),
    ("mid3-22    +spec γ=5",  "gpu_bf16_mid3-22_spec5.json",    True),
    ("mid3-23    no-spec",    "gpu_bf16_mid3-23_nospec.json",   False),
    ("mid3-23    +spec γ=5",  "gpu_bf16_mid3-23_spec5.json",    True),
    ("mid4-23    no-spec",    "gpu_bf16_mid4-23_nospec.json",   False),
    ("mid4-23    +spec γ=5",  "gpu_bf16_mid4-23_spec5.json",    True),
]

ref_lat = None
hdr = f"  {'Config':<28} {'Lat(s)':>7} {'Speedup':>8} {'tok/s':>7} {'WER%':>6} {'VRAM(GB)':>9}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for label, path, is_spec in rows:
    if not os.path.exists(path):
        print(f"  {label:<28}  [missing]")
        continue
    with open(path) as f:
        d = json.load(f)
    lat  = d.get("avg_latency_s", 0)
    tps  = d.get("avg_decode_tps", 0)
    wer  = d.get("wer", 0) * 100
    vram = d.get("gpu_mem_peak_gb")
    if ref_lat is None:
        ref_lat = lat
    ratio  = ref_lat / lat if lat > 0 else 0
    vram_s = f"{vram:.1f}" if vram else "  —"
    marker = " ◀" if is_spec else "  "
    print(f"  {label:<28} {lat:7.2f}  {ratio:7.2f}x  {tps:7.1f}  {wer:6.2f}  {vram_s:>9}{marker}")
PYEOF
