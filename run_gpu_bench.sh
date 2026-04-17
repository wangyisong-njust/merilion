#!/bin/bash
# ============================================================
# GPU inference benchmark — MERaLiON-2 3B (original + pruned)
#
# Runs infer_gpu.py across quantization configs mirroring
# run_demo.sh, then prints a side-by-side GPU vs CPU table.
#
# Quantization grid (original model):
#   BF16           — baseline, FlashAttn2
#   INT8 (BnB)     — 8-bit quantization, speech encoder in FP16
#   INT4 (BnB NF4) — 4-bit NF4
#
# Pruned models (tuned + merged, same configs as run_demo.sh):
#   BF16 only (CPU INT4+compile results already cover the pruned story)
#
# All JSON outputs are in the same schema as infer_cpu.py so
# run_compare.py can produce a unified GPU-vs-CPU table.
# ============================================================

PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
TUNE_ROOT="meralion_tune_log"
NUM_SAMPLES=20
GPU=0                          # set CUDA_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=$GPU

cd "$WORKDIR"

MID3_22="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-22-tune"
MID3_23="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-23-tune"
MID4_23="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid4-23-tune"

# ── output JSON files ─────────────────────────────────────────────────────
GPU_BF16_ORIG="gpu_bf16_original.json"
GPU_INT8_ORIG="gpu_int8_original.json"
GPU_INT4_ORIG="gpu_int4_original.json"

GPU_BF16_MID3_22="gpu_bf16_mid3-22.json"
GPU_BF16_MID3_23="gpu_bf16_mid3-23.json"
GPU_BF16_MID4_23="gpu_bf16_mid4-23.json"

# ── helper ────────────────────────────────────────────────────────────────
run_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then
        echo "  [skip] $json already exists"
        return 0
    fi
    echo "  running → $json"
    "$PYTHON_PATH" -u infer_gpu.py \
        --dataset     "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --save_samples \
        --output      "$json" \
        "$@" \
        || { echo "[FAIL] $json"; return 1; }
}

# ════════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "  Original MERaLiON-2-3B (no pruning)"
echo "========================================"

echo ""
echo "--- BF16 + FlashAttn2 ---"
run_if_missing "$GPU_BF16_ORIG" \
    --model "$ORIGINAL" \
    --quant bf16 \
    || exit 1

echo ""
echo "--- BnB INT8 ---"
run_if_missing "$GPU_INT8_ORIG" \
    --model "$ORIGINAL" \
    --quant int8 \
    || exit 1

echo ""
echo "--- BnB INT4/NF4 ---"
run_if_missing "$GPU_INT4_ORIG" \
    --model "$ORIGINAL" \
    --quant int4 \
    || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Pruned models — BF16 (GPU baseline)"
echo "========================================"

echo ""
echo "--- mid3-22 BF16 ---"
run_if_missing "$GPU_BF16_MID3_22" \
    --model "$MID3_22" \
    --quant bf16 \
    || exit 1

echo ""
echo "--- mid3-23 BF16 ---"
run_if_missing "$GPU_BF16_MID3_23" \
    --model "$MID3_23" \
    --quant bf16 \
    || exit 1

echo ""
echo "--- mid4-23 BF16 ---"
run_if_missing "$GPU_BF16_MID4_23" \
    --model "$MID4_23" \
    --quant bf16 \
    || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  GPU vs CPU summary"
echo "========================================"

"$PYTHON_PATH" - <<'PYEOF'
import json, os

GPU_ROWS = [
    ("GPU  Orig BF16",            "gpu_bf16_original.json"),
    ("GPU  Orig INT8",            "gpu_int8_original.json"),
    ("GPU  Orig INT4/NF4",        "gpu_int4_original.json"),
    ("GPU  mid3-22 BF16",         "gpu_bf16_mid3-22.json"),
    ("GPU  mid3-23 BF16",         "gpu_bf16_mid3-23.json"),
    ("GPU  mid4-23 BF16",         "gpu_bf16_mid4-23.json"),
]
CPU_ROWS = [
    ("CPU  Orig FP32  (ref)",     "demo_fp32_original.json"),
    ("CPU  Orig INT8",            "spec_int8_original_nospec.json"),
    ("CPU  mid3-22 INT4+cmp",     "demo_int4_mid3-22.json"),
    ("CPU  mid3-23 INT4+cmp",     "demo_int4_mid3-23.json"),
    ("CPU  mid4-23 INT4+cmp",     "demo_int4_mid4-23.json"),
]

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

ref_lat = None
hdr = f"  {'Config':<32} {'Lat(s)':>7} {'vs CPU-FP32':>11} {'tok/s':>7} {'WER%':>6} {'VRAM(GB)':>9}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for rows, section in [(GPU_ROWS, "GPU"), (CPU_ROWS, "CPU")]:
    print(f"  — {section} —")
    for label, path in rows:
        d = load(path)
        if d is None:
            print(f"  {label:<32}  [missing]")
            continue
        lat  = d.get("avg_latency_s", 0)
        tps  = d.get("avg_decode_tps", 0)
        wer  = d.get("wer", 0) * 100
        vram = d.get("gpu_mem_peak_gb")

        if ref_lat is None:
            ref_lat = lat
        ratio = ref_lat / lat if lat > 0 else 0
        vram_s = f"{vram:.1f}" if vram else "  — "
        print(f"  {label:<32} {lat:7.2f}  {ratio:10.2f}x  {tps:7.1f}  {wer:6.2f}  {vram_s:>9}")
PYEOF
