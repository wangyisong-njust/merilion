#!/bin/bash
# ============================================================
# torch.compile benchmark
# Compares no-compile vs --compile for INT8 models:
#   - Original MERaLiON-2-3B  INT8  (no-spec and +spec)
#   - Pruned mid3-23           INT8  (no-spec and +spec)
# ============================================================
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
TUNE_ROOT="meralion_tune_log"
CORPUS="ngram_corpus.pkl"
NUM_SAMPLES=50
GAMMA=3
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU
cd "$WORKDIR"

MID3_23="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-23-tune"

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
echo "  Original MERaLiON-2-3B  INT8"
echo "========================================"

echo ""
echo "--- INT8 no-compile no-spec ---"
run_if_missing "cmp_int8_original_nospec.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant int8 || exit 1

echo ""
echo "--- INT8 compile   no-spec ---"
run_if_missing "cmp_int8_original_nospec_compiled.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant int8 --compile || exit 1

echo ""
echo "--- INT8 no-compile +spec γ=${GAMMA} ---"
run_if_missing "cmp_int8_original_spec_g${GAMMA}.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant int8 \
    --speculative --gamma "$GAMMA" --corpus "$CORPUS" || exit 1

echo ""
echo "--- INT8 compile   +spec γ=${GAMMA} ---"
run_if_missing "cmp_int8_original_spec_g${GAMMA}_compiled.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant int8 \
    --speculative --gamma "$GAMMA" --corpus "$CORPUS" --compile || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Pruned mid3-23  INT8"
echo "========================================"

echo ""
echo "--- INT8 no-compile no-spec ---"
run_if_missing "cmp_int8_mid3-23_nospec.json" \
    --model "$MID3_23" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant int8 || exit 1

echo ""
echo "--- INT8 compile   no-spec ---"
run_if_missing "cmp_int8_mid3-23_nospec_compiled.json" \
    --model "$MID3_23" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant int8 --compile || exit 1

echo ""
echo "--- INT8 no-compile +spec γ=${GAMMA} ---"
run_if_missing "cmp_int8_mid3-23_spec_g${GAMMA}.json" \
    --model "$MID3_23" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant int8 \
    --speculative --gamma "$GAMMA" --corpus "$CORPUS" || exit 1

echo ""
echo "--- INT8 compile   +spec γ=${GAMMA} ---"
run_if_missing "cmp_int8_mid3-23_spec_g${GAMMA}_compiled.json" \
    --model "$MID3_23" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --quant int8 \
    --speculative --gamma "$GAMMA" --corpus "$CORPUS" --compile || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Summary"
echo "========================================"

"$PYTHON_PATH" - "$GAMMA" <<'PYEOF'
import json, os, sys

G = sys.argv[1]

rows = [
    ("Original INT8  no-spec",             "cmp_int8_original_nospec.json",                False, False),
    ("Original INT8  no-spec  +compile",   "cmp_int8_original_nospec_compiled.json",       False, True),
    (f"Original INT8  +spec γ={G}",        f"cmp_int8_original_spec_g{G}.json",            True,  False),
    (f"Original INT8  +spec γ={G} +cmp",   f"cmp_int8_original_spec_g{G}_compiled.json",   True,  True),
    ("mid3-23  INT8  no-spec",             "cmp_int8_mid3-23_nospec.json",                  False, False),
    ("mid3-23  INT8  no-spec  +compile",   "cmp_int8_mid3-23_nospec_compiled.json",         False, True),
    (f"mid3-23  INT8  +spec γ={G}",        f"cmp_int8_mid3-23_spec_g{G}.json",              True,  False),
    (f"mid3-23  INT8  +spec γ={G} +cmp",   f"cmp_int8_mid3-23_spec_g{G}_compiled.json",     True,  True),
]

ref_lat = None
hdr = f"  {'Config':<40} {'Lat(s)':>7} {'Speedup':>8} {'tok/s':>7} {'WER%':>6} {'AccRate':>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for label, path, is_spec, is_compiled in rows:
    if not os.path.exists(path):
        print(f"  {label:<40}  [missing]")
        continue
    with open(path) as f:
        d = json.load(f)
    lat  = d.get("avg_latency_s", 0)
    tps  = d.get("avg_decode_tps", 0)
    wer  = d.get("wer", 0) * 100
    acc  = d.get("avg_spec_accept_rate")
    if ref_lat is None:
        ref_lat = lat
    ratio = ref_lat / lat if lat > 0 else 0
    acc_s = f"{acc:.1%}" if acc is not None else "  —"
    marker = (" ◀" if is_spec else "  ") + ("⚡" if is_compiled else " ")
    print(f"  {label:<40} {lat:7.2f}  {ratio:7.2f}x  {tps:7.1f}  {wer:6.2f}  {acc_s:>8}{marker}")
PYEOF
