#!/bin/bash
# ============================================================
# EAGLE speculative-decoding GPU speedup benchmark
#
#   1. BF16  no-spec              (baseline, infer_gpu.py)
#   2. W4A16 no-EAGLE             (W4A16 greedy, fastest model that fits)
#   3. W4A16 + EAGLE  K=4  ◀ 1.9× best  (kartmannXu/MERaLiON-2-3B-EAGLE-W4A16)
#
# Model resolution:
#   Run 3 always pulls from HF: kartmannXu/MERaLiON-2-3B-EAGLE-W4A16
#   Run 2 uses the same HF model with --no_eagle (no spec overhead)
#     → first checks local quant_checkpoints/MERaLiON-2-3B-W4A16-RTN
#     → if missing, falls back to --hf_repo --no_eagle (derives from HF)
# ============================================================
export PYTHONUNBUFFERED=1

PYTHON_PATH="${PYTHON_PATH:-python3}"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"

BF16_MODEL="${BF16_MODEL:-/path/to/MERaLiON-2-3B}"         # for run 1
HF_EAGLE_REPO="kartmannXu/MERaLiON-2-3B-EAGLE-W4A16"       # runs 2+3
LOCAL_W4A16="${WORKDIR}/quant_checkpoints/MERaLiON-2-3B-W4A16-RTN"
DATASET="${DATASET:-/path/to/IMDA_PART1_mono_en_30_ASR}"
NUM_SAMPLES=20
K=4
GPU=0
GPTQ_KERNEL="${GPTQ_KERNEL:-exllama}"   # exllama (fastest batch=1) | exllamav2 | marlin

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

INFER="${WORKDIR}/infer_gpu.py"
INFER_EAGLE="${WORKDIR}/infer_gpu_eagle.py"

run_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then echo "  [skip] $json exists"; return 0; fi
    echo "  running → $json"
    "$PYTHON_PATH" -u "$@" 2>&1 | tee "${json%.json}.log"
    local rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && echo "[FAIL] $json (exit $rc)" && return 1
    return 0
}

# ════════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "  1/3  BF16 no-spec  (baseline)"
echo "========================================"
run_if_missing "eagle_bench_bf16_nospec.json" \
    "$INFER" \
        --model "$BF16_MODEL" \
        --dataset "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --quant bf16 \
        --save_samples \
        --output "eagle_bench_bf16_nospec.json" || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  2/3  W4A16 no-EAGLE  (greedy baseline)"
echo "========================================"

if [ -f "${LOCAL_W4A16}/model.safetensors.index.json" ]; then
    echo "  Found local W4A16 model: $LOCAL_W4A16"
    run_if_missing "eagle_bench_w4a16_nospec.json" \
        "$INFER_EAGLE" \
            --model "$LOCAL_W4A16" \
            --bf16_path "$BF16_MODEL" \
            --quant gptq_marlin \
            --gptq_kernel "$GPTQ_KERNEL" \
            --no_eagle \
            --dataset "$DATASET" \
            --num_samples "$NUM_SAMPLES" \
            --output "eagle_bench_w4a16_nospec.json" || exit 1
else
    echo "  Local W4A16 not found at $LOCAL_W4A16"
    echo "  Deriving W4A16 model from HF repo (--no_eagle) …"
    run_if_missing "eagle_bench_w4a16_nospec.json" \
        "$INFER_EAGLE" \
            --hf_repo "$HF_EAGLE_REPO" \
            --gptq_kernel "$GPTQ_KERNEL" \
            --no_eagle \
            --dataset "$DATASET" \
            --num_samples "$NUM_SAMPLES" \
            --output "eagle_bench_w4a16_nospec.json" || exit 1
fi

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  3/3  W4A16 + EAGLE  K=${K}  (1.9× target)"
echo "========================================"
run_if_missing "eagle_bench_w4a16_eagle_K${K}.json" \
    "$INFER_EAGLE" \
        --hf_repo "$HF_EAGLE_REPO" \
        --gptq_kernel "$GPTQ_KERNEL" \
        --K "$K" \
        --dataset "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --output "eagle_bench_w4a16_eagle_K${K}.json" || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Summary — speedup vs BF16 no-spec"
echo "========================================"

"$PYTHON_PATH" - <<PYEOF
import json, os

rows = [
    ("BF16   no-spec       (baseline)", "eagle_bench_bf16_nospec.json",       False),
    ("W4A16  no-EAGLE      (greedy)",   "eagle_bench_w4a16_nospec.json",      False),
    ("W4A16  + EAGLE  K=${K}  ◀ best", "eagle_bench_w4a16_eagle_K${K}.json", True),
]

ref_lat = None
hdr = (f"  {'Config':<36} {'Lat(s)':>7} {'Speedup':>8} "
       f"{'tok/s':>7} {'SpecAcc%':>9} {'WER%':>6} {'VRAM(GB)':>9}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for label, path, is_spec in rows:
    if not os.path.exists(path):
        print(f"  {label:<36}  [missing]")
        continue
    with open(path) as f:
        d = json.load(f)
    lat  = d.get("avg_latency_s", 0)
    tps  = d.get("avg_decode_tps", 0)
    wer  = d.get("wer", 0) * 100
    vram = d.get("gpu_mem_peak_gb")
    acc  = d.get("avg_spec_accept_rate")
    if ref_lat is None and lat > 0:
        ref_lat = lat
    ratio  = (ref_lat / lat) if (ref_lat and lat > 0) else 0.0
    vram_s = f"{vram:.1f}" if vram is not None else "  —"
    acc_s  = f"{acc*100:.1f}%" if acc is not None else "   —"
    print(f"  {label:<36} {lat:7.2f}  {ratio:7.2f}x  {tps:7.1f}  "
          f"{acc_s:>9}  {wer:6.2f}  {vram_s:>9}")
PYEOF

echo ""
echo "Build HTML demo:"
echo "  python3 make_demo_html.py \\"
echo "    --title 'MERaLiON-2-3B — EAGLE W4A16 (1.9× speedup)' \\"
echo "    --configs \\"
echo "      'BF16 baseline:eagle_bench_bf16_nospec.json' \\"
echo "      'W4A16 no-EAGLE:eagle_bench_w4a16_nospec.json' \\"
echo "      'W4A16+EAGLE K=${K}:eagle_bench_w4a16_eagle_K${K}.json' \\"
echo "    --output demo_eagle.html"
