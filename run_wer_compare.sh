#!/bin/bash
# ============================================================
# WER comparison:
#   1. MERaLiON-2-3B original      — BF16 GPU (reference)
#   2. MERaLiON-2-3B-MLX-4bit sim — MLX affine int4 (group=64)
#      Matches majentik/MERaLiON-2-3B-MLX-4bit quantization spec:
#        bits=4, group_size=64, decoder only, encoder+adapter in FP16
#   3. Pruned mid3-22 + BnB INT8   — our compressed model
# ============================================================
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
TUNE_ROOT="meralion_tune_log"
PRUNED="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-22-tune"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
NUM_SAMPLES=200
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

run_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then echo "  [skip] $json already exists"; return 0; fi
    echo "  running → $json"
    "$PYTHON_PATH" -u infer_gpu.py --output "$json" \
        --dataset "$DATASET" --num_samples "$NUM_SAMPLES" --save_samples \
        "$@" | tee "${json%.json}.log" \
        || { echo "[FAIL] $json"; return 1; }
}

echo "========================================"
echo "  1. Original MERaLiON-2-3B  BF16"
echo "========================================"
run_if_missing "wer_original_bf16.json" \
    --model "$ORIGINAL" --quant bf16 || exit 1

echo ""
echo "========================================"
echo "  2. MLX-4bit simulation (int4 group=64)"
echo "     decoder only, encoder+adapter FP16"
echo "========================================"
run_if_missing "wer_mlx4_original.json" \
    --model "$ORIGINAL" --quant mlx4 || exit 1

echo ""
echo "========================================"
echo "  3. Pruned mid3-22 + BnB INT8"
echo "========================================"
run_if_missing "wer_pruned_mid3-22_int8.json" \
    --model "$PRUNED" --quant int8 || exit 1

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"

"$PYTHON_PATH" - <<'PYEOF'
import json, os

rows = [
    ("Original BF16 (reference)",    "wer_original_bf16.json"),
    ("MLX-4bit sim  (decoder int4)", "wer_mlx4_original.json"),
    ("Pruned mid3-22 + INT8",        "wer_pruned_mid3-22_int8.json"),
]

ref_wer = None
hdr = f"  {'Model':<34} {'WER%':>6} {'ΔWER':>7} {'Lat(s)':>7} {'tok/s':>7} {'VRAM(GB)':>9}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for label, path in rows:
    if not os.path.exists(path):
        print(f"  {label:<34}  [missing]")
        continue
    with open(path) as f:
        d = json.load(f)
    wer  = d.get("wer", 0) * 100
    lat  = d.get("avg_latency_s", 0)
    tps  = d.get("avg_decode_tps", 0)
    vram = d.get("gpu_mem_peak_gb") or 0
    if ref_wer is None:
        ref_wer = wer
    delta = f"+{wer-ref_wer:.2f}" if wer > ref_wer else f"{wer-ref_wer:.2f}"
    print(f"  {label:<34} {wer:6.2f}  {delta:>7}  {lat:7.2f}  {tps:7.1f}  {vram:9.2f}")
PYEOF
