#!/bin/bash
# ============================================================
# Model-based speculative decoding benchmark  (AWQ4 edition)
# Verifier: original MERaLiON-2-3B (BF16 or AWQ4)
# Draft:    pruned mid3-23 AWQ4
# ============================================================
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
ORIGINAL_AWQ4="${WORKDIR}/MERaLiON-2-3B-AWQ4"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
NUM_SAMPLES=50
GAMMA=5
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

DRAFT_AWQ4="${WORKDIR}/MERaLiON-2-3B-mid3-23-AWQ4"

run_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then echo "  [skip] $json already exists"; return 0; fi
    echo "  running → $json"
    "$PYTHON_PATH" -u "$@" --output "$json" \
        | tee "${json%.json}.log" \
        || { echo "[FAIL] $json"; return 1; }
}

run_spec_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then echo "  [skip] $json already exists"; return 0; fi
    echo "  running → $json"
    "$PYTHON_PATH" -u infer_gpu_spec_draft.py \
        --verifier "$ORIGINAL" --draft "$DRAFT_AWQ4" \
        --dataset "$DATASET" --num_samples "$NUM_SAMPLES" \
        --gamma "$GAMMA" --output "$json" "$@" \
        | tee "${json%.json}.log" \
        || { echo "[FAIL] $json"; return 1; }
}

run_spec_awq4v_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then echo "  [skip] $json already exists"; return 0; fi
    echo "  running → $json"
    "$PYTHON_PATH" -u infer_gpu_spec_draft.py \
        --verifier "$ORIGINAL_AWQ4" --draft "$DRAFT_AWQ4" \
        --dataset "$DATASET" --num_samples "$NUM_SAMPLES" \
        --gamma "$GAMMA" --output "$json" "$@" \
        | tee "${json%.json}.log" \
        || { echo "[FAIL] $json"; return 1; }
}

# ════════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "  BF16 verifier + AWQ4 draft"
echo "========================================"

echo ""
echo "--- BF16 no-spec (reuse from spec bench) ---"
run_if_missing "gpu_bf16_original_nospec.json" \
    infer_gpu.py \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant bf16 || exit 1

echo ""
echo "--- BF16 verifier + AWQ4 draft γ=${GAMMA} ---"
run_spec_if_missing "draft_bf16_orig_mid323awq4_g${GAMMA}.json" \
    --verifier_quant bf16 --draft_quant awq4 || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  AWQ4 verifier + AWQ4 draft"
echo "========================================"

echo ""
echo "--- AWQ4 no-spec (reuse from spec bench) ---"
run_if_missing "gpu_awq4_original_nospec.json" \
    infer_gpu.py \
    --model "$ORIGINAL_AWQ4" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant awq4 || exit 1

echo ""
echo "--- AWQ4 verifier + AWQ4 draft γ=${GAMMA} ---"
run_spec_awq4v_if_missing "draft_awq4_orig_mid323awq4_g${GAMMA}.json" \
    --verifier_quant awq4 --draft_quant awq4 || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Summary"
echo "========================================"

"$PYTHON_PATH" - "$GAMMA" <<'PYEOF'
import json, os, sys

G = sys.argv[1]

rows = [
    ("Original  BF16   no-spec",                    "gpu_bf16_original_nospec.json",                False),
    (f"Original  BF16   +draft-spec γ={G}",          f"draft_bf16_orig_mid323awq4_g{G}.json",        True),
    ("Original  AWQ4   no-spec",                    "gpu_awq4_original_nospec.json",                 False),
    (f"Original  AWQ4   +draft-spec γ={G}",          f"draft_awq4_orig_mid323awq4_g{G}.json",         True),
]

ref_lat = None
hdr = f"  {'Config':<42} {'Lat(s)':>7} {'Speedup':>8} {'tok/s':>7} {'WER%':>6} {'VRAM(GB)':>9} {'AccRate':>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for label, path, is_spec in rows:
    if not os.path.exists(path):
        print(f"  {label:<42}  [missing]")
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
    print(f"  {label:<42} {lat:7.2f}  {ratio:7.2f}x  {tps:7.1f}  {wer:6.2f}  {vram_s:>9}  {acc_s:>8}{marker}")
PYEOF
