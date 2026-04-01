#!/bin/bash
# ============================================================
# Build the HTML demo page for the 5 Pareto-frontier configs:
#   1. Original FP32 (no compile)
#   2. mid3-22 INT8  (no compile)
#   3. mid4-23 INT4+compile
#   4. mid3-23 INT4+compile
#   5. mid3-22 INT4+compile
#
# Steps:
#   A) Run infer_cpu.py for each config (skip if JSON exists)
#      --audio_dir is passed only to the first run; subsequent
#      runs reuse the WAV files already on disk.
#   B) Call make_demo_html.py to produce demo.html
# ============================================================

PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
TUNE_ROOT="meralion_tune_log"
NUM_SAMPLES=20
AUDIO_DIR="demo_audio"
OUTPUT_HTML="demo.html"

cd "$WORKDIR"

# ── Model paths ───────────────────────────────────────────────────────────
MID3_22="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-22-tune"
MID3_23="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid3-23-tune"
MID4_23="${TUNE_ROOT}/MERaLiON-2-3B-v3-td50-mid4-23-tune"

# ── JSON output files ─────────────────────────────────────────────────────
FP32_JSON="demo_fp32_original.json"
INT8_MID3_22_JSON="demo_int8_mid3-22.json"
INT4_MID4_23_JSON="demo_int4_mid4-23.json"
INT4_MID3_23_JSON="demo_int4_mid3-23.json"
INT4_MID3_22_JSON="demo_int4_mid3-22.json"

# ── helper ────────────────────────────────────────────────────────────────
run_if_missing() {
    local json="$1"; shift
    "$PYTHON_PATH" -u infer_cpu.py \
        --dataset     "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --save_samples \
        --output      "$json" \
        "$@" \
        || { echo "[FAIL] $json"; return 1; }
}

# ── Step A: run benchmarks ────────────────────────────────────────────────
echo "========================================"
echo "  Step A: running benchmarks"
echo "========================================"

echo ""
echo "--- Original FP32 ---"
run_if_missing "$FP32_JSON" \
    --model "$ORIGINAL" \
    --trust_remote_code \
    --no_quant --no_compile \
    --audio_dir "$AUDIO_DIR" \
    || exit 1

echo ""
echo "--- mid3-22 INT8 ---"
run_if_missing "$INT8_MID3_22_JSON" \
    --model "$MID3_22" \
    || exit 1

echo ""
echo "--- mid4-23 INT4+compile ---"
run_if_missing "$INT4_MID4_23_JSON" \
    --model "$MID4_23" \
    --int4 \
    || exit 1

echo ""
echo "--- mid3-23 INT4+compile ---"
run_if_missing "$INT4_MID3_23_JSON" \
    --model "$MID3_23" \
    --int4 \
    || exit 1

echo ""
echo "--- mid3-22 INT4+compile ---"
run_if_missing "$INT4_MID3_22_JSON" \
    --model "$MID3_22" \
    --int4 \
    || exit 1

# ── Step B: generate HTML ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Step B: generating $OUTPUT_HTML"
echo "========================================"

"$PYTHON_PATH" make_demo_html.py \
    --configs \
        "Original FP32:${FP32_JSON}" \
        "mid3-22 INT8:${INT8_MID3_22_JSON}" \
        "mid4-23 INT4+compile:${INT4_MID4_23_JSON}" \
        "mid3-23 INT4+compile:${INT4_MID3_23_JSON}" \
        "mid3-22 INT4+compile:${INT4_MID3_22_JSON}" \
    --output "$OUTPUT_HTML" \
    || exit 1

echo ""
echo "Done. Open: $WORKDIR/$OUTPUT_HTML"
