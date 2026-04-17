#!/bin/bash
# ============================================================
# Speculative decoding speedup experiment
#
# Compares baseline vs. n-gram speculative decoding (3-gram+4-gram)
# across the same configs used in run_demo.sh:
#
#   A) Original MERaLiON-2-3B (no pruning):
#        FP32 no-spec (reuse demo_fp32_original.json if exists)
#        INT8 no-spec
#        INT8 + spec (gamma=5)
#
#   B) Pruned models (mid3-22, mid3-23, mid4-23):
#        INT4+compile no-spec (reuse demo_int4_*.json if exists)
#        INT4+compile + spec (gamma=5)
#
# Outputs a summary table at the end.
# ============================================================

PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
TUNE_ROOT="meralion_tune_log"
NUM_SAMPLES=20
GAMMA=5

cd "$WORKDIR"

MID3_22="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-22-tune"
MID3_23="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-23-tune"
MID4_23="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid4-23-tune"

# ── output JSON files ─────────────────────────────────────────────────────
FP32_ORIG_JSON="demo_fp32_original.json"              # reuse from run_demo.sh

INT8_ORIG_JSON="spec_int8_original_nospec.json"
INT8_ORIG_SPEC_JSON="spec_int8_original_spec${GAMMA}.json"

INT4_MID3_22_JSON="demo_int4_mid3-22.json"            # reuse from run_demo.sh
INT4_MID3_23_JSON="demo_int4_mid3-23.json"
INT4_MID4_23_JSON="demo_int4_mid4-23.json"

INT4_MID3_22_SPEC_JSON="spec_int4_mid3-22_spec${GAMMA}.json"
INT4_MID3_23_SPEC_JSON="spec_int4_mid3-23_spec${GAMMA}.json"
INT4_MID4_23_SPEC_JSON="spec_int4_mid4-23_spec${GAMMA}.json"

# ── helper ────────────────────────────────────────────────────────────────
run_if_missing() {
    local json="$1"; shift
    if [ -f "$json" ]; then
        echo "  [skip] $json already exists"
        return 0
    fi
    echo "  running → $json"
    "$PYTHON_PATH" -u infer_cpu.py \
        --dataset     "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --save_samples \
        --output      "$json" \
        "$@" \
        || { echo "[FAIL] $json"; return 1; }
}

# ════════════════════════════════════════════════════════════════════════════
echo "========================================"
echo "  Part A: Original model (no pruning)"
echo "========================================"

echo ""
echo "--- Original FP32 no-spec (reference) ---"
run_if_missing "$FP32_ORIG_JSON" \
    --model "$ORIGINAL" \
    --trust_remote_code \
    --no_quant --no_compile \
    || exit 1

echo ""
echo "--- Original INT8 no-spec ---"
run_if_missing "$INT8_ORIG_JSON" \
    --model "$ORIGINAL" \
    --trust_remote_code \
    --no_compile \
    || exit 1

echo ""
echo "--- Original INT8 + speculative (gamma=${GAMMA}) ---"
run_if_missing "$INT8_ORIG_SPEC_JSON" \
    --model "$ORIGINAL" \
    --trust_remote_code \
    --no_compile \
    --speculative --gamma "$GAMMA" \
    || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Part B: Pruned models INT4+compile"
echo "========================================"

echo ""
echo "--- mid3-22 INT4+compile no-spec ---"
run_if_missing "$INT4_MID3_22_JSON" \
    --model "$MID3_22" \
    --int4 \
    || exit 1

echo ""
echo "--- mid3-22 INT4+compile + speculative (gamma=${GAMMA}) ---"
run_if_missing "$INT4_MID3_22_SPEC_JSON" \
    --model "$MID3_22" \
    --int4 \
    --speculative --gamma "$GAMMA" \
    || exit 1

echo ""
echo "--- mid3-23 INT4+compile no-spec ---"
run_if_missing "$INT4_MID3_23_JSON" \
    --model "$MID3_23" \
    --int4 \
    || exit 1

echo ""
echo "--- mid3-23 INT4+compile + speculative (gamma=${GAMMA}) ---"
run_if_missing "$INT4_MID3_23_SPEC_JSON" \
    --model "$MID3_23" \
    --int4 \
    --speculative --gamma "$GAMMA" \
    || exit 1

echo ""
echo "--- mid4-23 INT4+compile no-spec ---"
run_if_missing "$INT4_MID4_23_JSON" \
    --model "$MID4_23" \
    --int4 \
    || exit 1

echo ""
echo "--- mid4-23 INT4+compile + speculative (gamma=${GAMMA}) ---"
run_if_missing "$INT4_MID4_23_SPEC_JSON" \
    --model "$MID4_23" \
    --int4 \
    --speculative --gamma "$GAMMA" \
    || exit 1

# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Summary table"
echo "========================================"

"$PYTHON_PATH" - <<'PYEOF'
import json, os, sys

rows = [
    ("Orig FP32  (reference)",       "demo_fp32_original.json",           False),
    ("Orig INT8  no-spec",            "spec_int8_original_nospec.json",    False),
    ("Orig INT8  +spec γ=5",          "spec_int8_original_spec5.json",     True),
    ("mid3-22 INT4+cmp no-spec",      "demo_int4_mid3-22.json",            False),
    ("mid3-22 INT4+cmp +spec γ=5",    "spec_int4_mid3-22_spec5.json",      True),
    ("mid3-23 INT4+cmp no-spec",      "demo_int4_mid3-23.json",            False),
    ("mid3-23 INT4+cmp +spec γ=5",    "spec_int4_mid3-23_spec5.json",      True),
    ("mid4-23 INT4+cmp no-spec",      "demo_int4_mid4-23.json",            False),
    ("mid4-23 INT4+cmp +spec γ=5",    "spec_int4_mid4-23_spec5.json",      True),
]

fp32_lat = None
hdr = f"{'Config':<36} {'Lat(s)':>7} {'vs FP32':>8} {'tok/s':>7} {'WER%':>6} {'AccRate':>8}"
print(hdr)
print("-" * len(hdr))

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
    if fp32_lat is None:
        fp32_lat = lat
    ratio = fp32_lat / lat if lat > 0 else 0
    acc_s = f"{acc:.0%}" if acc is not None else "  —  "
    print(f"  {label:<34} {lat:7.2f}  {ratio:7.2f}x  {tps:7.1f}  {wer:6.2f}  {acc_s:>8}")
PYEOF
