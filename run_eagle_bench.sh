#!/bin/bash
# ============================================================
# EAGLE speculative-decoding GPU speedup benchmark
#   1. BF16  no-spec              (baseline, infer_gpu.py)
#   2. BF16  + EAGLE  K=4        (infer_gpu_eagle.py)
#   3. W4A16 + EAGLE  K=4        (gptq_marlin + exllama kernel)
#
# Best result from HANDOFF §15: W4A16+ExllamaV1+EAGLE → 1.90× speedup
#
# Speedup is relative to run #1.  All three runs must share the same
# dataset slice so latency numbers are directly comparable.
# ============================================================
export PYTHONUNBUFFERED=1

PYTHON_PATH="${PYTHON_PATH:-python}"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"

# ── paths — override via environment ─────────────────────────────────────────
BF16_MODEL="${BF16_MODEL:-/path/to/MERaLiON-2-3B}"
GPTQ_MODEL="${GPTQ_MODEL:-/path/to/MERaLiON-2-3B-W4A16-GPTQ-Marlin}"
EAGLE_HEADS="${EAGLE_HEADS:-${WORKDIR}/medusa_heads_v2_best.pt}"  # eagle_best.pt also works
DATASET="${DATASET:-/path/to/IMDA_PART1_mono_en_30_ASR}"
NUM_SAMPLES=20
K=4
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

# ── helper ────────────────────────────────────────────────────────────────────
run_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then
        echo "  [skip] $json already exists"
        return 0
    fi
    echo "  running → $json"
    "$PYTHON_PATH" -u "$@" \
        2>&1 | tee "${json%.json}.log"
    local rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && echo "[FAIL] $json (exit $rc)" && return 1
    return 0
}

INFER="${WORKDIR}/infer_gpu.py"
INFER_EAGLE="${WORKDIR}/infer_gpu_eagle.py"

# ════════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "  1/3  BF16  no-spec  (baseline)"
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
echo "  2/3  BF16  + EAGLE  K=${K}"
echo "========================================"
run_if_missing "eagle_bench_bf16_eagle_K${K}.json" \
    "$INFER_EAGLE" \
        --model "$BF16_MODEL" \
        --eagle "$EAGLE_HEADS" \
        --dataset "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --quant bf16 \
        --K "$K" \
        --output "eagle_bench_bf16_eagle_K${K}.json" || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  3/3  W4A16 GPTQ (ExllamaV1) + EAGLE  K=${K}"
echo "========================================"
run_if_missing "eagle_bench_w4a16_eagle_K${K}.json" \
    "$INFER_EAGLE" \
        --model "$GPTQ_MODEL" \
        --bf16_path "$BF16_MODEL" \
        --eagle "$EAGLE_HEADS" \
        --dataset "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --quant gptq_marlin \
        --gptq_kernel exllama \
        --K "$K" \
        --output "eagle_bench_w4a16_eagle_K${K}.json" || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Summary — speedup vs BF16 no-spec"
echo "========================================"

"$PYTHON_PATH" - <<PYEOF
import json, os

rows = [
    ("BF16   no-spec       (baseline)", "eagle_bench_bf16_nospec.json",      False),
    ("BF16   + EAGLE  K=${K}",           "eagle_bench_bf16_eagle_K${K}.json",  True),
    ("W4A16  + EAGLE  K=${K}  ◀ best",   "eagle_bench_w4a16_eagle_K${K}.json", True),
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
echo "To build the HTML demo report:"
echo "  python make_demo_html.py \\"
echo "    --title \"MERaLiON-2-3B — EAGLE Speculative Decoding\" \\"
echo "    --configs \\"
echo "      \"BF16 no-spec:eagle_bench_bf16_nospec.json\" \\"
echo "      \"BF16+EAGLE K=${K}:eagle_bench_bf16_eagle_K${K}.json\" \\"
echo "      \"W4A16+EAGLE K=${K}:eagle_bench_w4a16_eagle_K${K}.json\" \\"
echo "    --output demo_eagle.html"
