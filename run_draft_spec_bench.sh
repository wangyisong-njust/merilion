#!/bin/bash
# ============================================================
# Model-based speculative decoding benchmark
# Verifier: original MERaLiON-2-3B (BF16 or MLX4)
# Draft:    pruned mid3-23 BnB-INT4
# ============================================================
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/kaixin/anaconda3/envs/llm_pruner_meralion/bin/python"
WORKDIR="/home/kaixin/yisong/merilion"
ORIGINAL="/home/kaixin/programs/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/kaixin/ssd/data/IMDA_PART1_mono_en_30_ASR"
TUNE_ROOT="/home/kaixin/yisong/merilion/meralion_tune_log"
NUM_SAMPLES=50
GAMMA=5
GPU=2
FORCE=${FORCE:-0}   # set FORCE=1 to re-run everything regardless of cached JSON

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

DRAFT="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-23-tune"

run_if_missing() {
    local json="$1"; shift
    if [ "$FORCE" != "1" ] && [ -f "$json" ]; then echo "  [skip] $json already exists"; return 0; fi
    echo "  running → $json"
    "$PYTHON_PATH" -u "$@" --output "$json" \
        | tee "${json%.json}.log" \
        || { echo "[FAIL] $json"; return 1; }
}

run_spec_if_missing() {
    local json="$1"; shift
    if [ "$FORCE" != "1" ] && [ -f "$json" ]; then echo "  [skip] $json already exists"; return 0; fi
    echo "  running → $json"
    "$PYTHON_PATH" -u infer_gpu_spec_draft.py \
        --verifier "$ORIGINAL" --draft "$DRAFT" \
        --dataset "$DATASET" --num_samples "$NUM_SAMPLES" \
        --gamma "$GAMMA" --output "$json" "$@" \
        | tee "${json%.json}.log" \
        || { echo "[FAIL] $json"; return 1; }
}

# ════════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "  BF16 verifier + BnB-INT4 draft"
echo "========================================"

echo ""
echo "--- BF16 no-spec (reuse from spec bench) ---"
run_if_missing "gpu_bf16_original_nospec.json" \
    infer_gpu.py \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant bf16 || exit 1

echo ""
echo "--- BF16 verifier + BnB-INT4 draft γ=${GAMMA} ---"
run_spec_if_missing "draft_bf16_orig_mid323bnb4_g${GAMMA}.json" \
    --verifier_quant bf16 --draft_quant int4 || exit 1

echo ""
echo "--- BF16 verifier + BF16 draft γ=${GAMMA} ---"
run_spec_if_missing "draft_bf16_orig_mid323bf16_g${GAMMA}.json" \
    --verifier_quant bf16 --draft_quant bf16 || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  INT8 verifier + BnB-INT4 draft"
echo "========================================"

echo ""
echo "--- INT8 no-spec (reuse from spec bench) ---"
run_if_missing "gpu_int8_original_nospec.json" \
    infer_gpu.py \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant int8 || exit 1

echo ""
echo "--- INT8 verifier + BnB-INT4 draft γ=${GAMMA} ---"
run_spec_if_missing "draft_int8_orig_mid323bnb4_g${GAMMA}.json" \
    --verifier_quant int8 --draft_quant int4 || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Summary"
echo "========================================"

"$PYTHON_PATH" - "$GAMMA" <<'PYEOF'
import json, os, sys

G = sys.argv[1]

rows = [
    ("Original  BF16  no-spec",                    "gpu_bf16_original_nospec.json",               False),
    (f"Original  BF16  +BnB4 draft γ={G}",          f"draft_bf16_orig_mid323bnb4_g{G}.json",       True),
    (f"Original  BF16  +BF16 draft γ={G}",          f"draft_bf16_orig_mid323bf16_g{G}.json",       True),
    ("Original  INT8  no-spec",                    "gpu_int8_original_nospec.json",               False),
    (f"Original  INT8  +BnB4 draft γ={G}",          f"draft_int8_orig_mid323bnb4_g{G}.json",       True),
]

ref_lat = None
hdr = f"  {'Config':<40} {'Lat(s)':>7} {'Speedup':>8} {'tok/s':>7} {'WER%':>6} {'VRAM(GB)':>9} {'AccRate':>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for label, path, is_spec in rows:
    if not os.path.exists(path):
        print(f"  {label:<40}  [missing]")
        continue
    with open(path) as f:
        d = json.load(f)
    lat  = d.get("avg_latency_s", 0)
    tps  = d.get("avg_decode_tps", 0)
    wer  = d.get("wer", 0) * 100
    vram = d.get("gpu_mem_peak_gb")
    acc  = d.get("avg_spec_accept_rate")
    if ref_lat is None:
        ref_lat = lat
    ratio  = ref_lat / lat if lat > 0 else 0
    vram_s = f"{vram:.1f}" if vram else "  —"
    acc_s  = f"{acc:.1%}" if acc is not None else "  —"
    marker = " ◀" if is_spec else "  "
    print(f"  {label:<40} {lat:7.2f}  {ratio:7.2f}x  {tps:7.1f}  {wer:6.2f}  {vram_s:>9}  {acc_s:>8}{marker}")
PYEOF
