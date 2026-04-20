#!/bin/bash
# ============================================================
# CPU speculative decoding benchmark
# Compares no-spec vs n-gram +spec for:
#   - Original MERaLiON-2-3B  (FP32)
#   - Pruned mid3-23           (INT4)
# ============================================================
export PYTHONUNBUFFERED=1
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
TUNE_ROOT="meralion_tune_log"
CORPUS="ngram_corpus.pkl"   # built by build_ngram_corpus.py
NUM_SAMPLES=20
GAMMA=5

cd "$WORKDIR"

MID3_23="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-23-tune"

run_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then echo "  [skip] $json already exists"; return 0; fi
    echo "  running → $json"
    "$PYTHON_PATH" -u infer_cpu.py --output "$json" "$@" \
        | tee "${json%.json}.log" \
        || { echo "[FAIL] $json"; return 1; }
}

# ════════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "  Original MERaLiON-2-3B  FP32"
echo "========================================"

echo ""
echo "--- FP32 no-spec ---"
run_if_missing "cpu_fp32_original_nospec.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --trust_remote_code || exit 1

echo ""
echo "--- FP32 +spec γ=${GAMMA} ---"
run_if_missing "cpu_fp32_original_spec_g${GAMMA}.json" \
    --model "$ORIGINAL" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --trust_remote_code \
    --speculative --gamma "$GAMMA" --corpus "$CORPUS" || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Pruned mid3-23  INT4"
echo "========================================"

echo ""
echo "--- INT4 no-spec ---"
run_if_missing "cpu_int4_mid3-23_nospec.json" \
    --model "$MID3_23" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --int4 || exit 1

echo ""
echo "--- INT4 +spec γ=${GAMMA} ---"
run_if_missing "cpu_int4_mid3-23_spec_g${GAMMA}.json" \
    --model "$MID3_23" --dataset "$DATASET" \
    --num_samples "$NUM_SAMPLES" --int4 \
    --speculative --gamma "$GAMMA" --corpus "$CORPUS" || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Summary"
echo "========================================"

"$PYTHON_PATH" - "$GAMMA" <<'PYEOF'
import json, os, sys

G = sys.argv[1]

rows = [
    ("Original  FP32  no-spec",      "cpu_fp32_original_nospec.json",        False),
    (f"Original  FP32  +spec γ={G}", f"cpu_fp32_original_spec_g{G}.json",    True),
    ("mid3-23   INT4  no-spec",      "cpu_int4_mid3-23_nospec.json",          False),
    (f"mid3-23   INT4  +spec γ={G}", f"cpu_int4_mid3-23_spec_g{G}.json",     True),
]

ref_lat = None
hdr = f"  {'Config':<34} {'Lat(s)':>7} {'Speedup':>8} {'tok/s':>7} {'WER%':>6} {'AccRate':>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for label, path, is_spec in rows:
    if not os.path.exists(path):
        print(f"  {label:<34}  [missing]")
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
    marker = " ◀" if is_spec else "  "
    print(f"  {label:<34} {lat:7.2f}  {ratio:7.2f}x  {tps:7.1f}  {wer:6.2f}  {acc_s:>8}{marker}")
PYEOF
