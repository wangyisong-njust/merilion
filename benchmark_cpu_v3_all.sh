#!/bin/bash
# ============================================================
# CPU INT4 benchmark — ALL v3 mid-pruning configurations
#
# For each config that has a completed tune dir or pruned ckpt:
#   1. Merge LoRA (if adapter not yet merged)
#   2. FP32 baseline  (no_quant, no_compile)
#   3. INT4 + compile (torchao Int4WeightOnlyQuantizer)
#   4. Collect WER, latency, disk size
# Then print a summary table.
# ============================================================

PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
NUM_SAMPLES=20
CKPT_ROOT="meralion_checkpoints"
TUNE_ROOT="meralion_tune_log"

cd "$WORKDIR"

# ── All v3 mid configurations ─────────────────────────────────────────────
# Each entry: "short_name:has_posttrain"
#   has_posttrain=1  → look for TUNE_ROOT/MERaLiON-2-3B-<name>-tune   (LoRA → merge)
#   has_posttrain=0  → use CKPT_ROOT/MERaLiON-2-3B-<name> directly
CONFIGS=(
    "v3-td50-mid3-22:1"
    "v3-td50-mid3-23:1"
    "v3-td50-mid4-22:0"
    "v3-td50-mid4-23:1"
    "v3-td50-mid5-19:1"
    "v3-td50-mid5-19-wm25:1"
    "v3-td50-mid6-20:1"
    "v3-td50-mid6-20-wm15:1"
    "v3-td50-mid7-21:1"
)

# ── Helper: compute model disk size in MB ─────────────────────────────────
disk_mb() {
    local dir="$1"
    du -sm "$dir"/*.safetensors "$dir"/*.bin 2>/dev/null \
        | awk '{s+=$1} END {print (s ? s : 0)}'
}

# ── Baseline: original model size ────────────────────────────────────────
ORIG_MB=$(disk_mb "$ORIGINAL")
echo "Original model disk size: ${ORIG_MB} MB"
echo ""

# ── Result accumulator ────────────────────────────────────────────────────
# Format: "name|fp32_lat|int4_lat|speedup|fp32_wer|int4_wer|size_mb|size_pct"
RESULTS=()
SKIPPED=()

# ── Main loop ─────────────────────────────────────────────────────────────
for entry in "${CONFIGS[@]}"; do
    IFS=':' read -r NAME HAS_PT <<< "$entry"
    CKPT="${CKPT_ROOT}/MERaLiON-2-3B-${NAME}"
    TUNE="${TUNE_ROOT}/MERaLiON-2-3B-${NAME}-tune"

    echo "=========================================================="
    echo "Config: $NAME"
    echo "=========================================================="

    # ── Determine model directory ──────────────────────────────────────────
    if [ "$HAS_PT" = "1" ]; then
        if [ ! -d "$TUNE" ]; then
            echo "  [SKIP] tune dir not found: $TUNE"
            SKIPPED+=("$NAME (tune dir missing)")
            continue
        fi
        MODEL_DIR="$TUNE"

        # ── Merge LoRA if still an adapter (no full model weights) ──────────
        HAS_FULL=0
        ls "${TUNE}"/model*.safetensors 2>/dev/null | grep -q . && HAS_FULL=1
        ls "${TUNE}"/pytorch_model*.bin  2>/dev/null | grep -q . && HAS_FULL=1

        if [ "$HAS_FULL" = "0" ] && [ -f "${TUNE}/adapter_config.json" ]; then
            if [ ! -d "$CKPT" ]; then
                echo "  [SKIP] base ckpt not found for merge: $CKPT"
                SKIPPED+=("$NAME (base ckpt missing for merge)")
                continue
            fi
            echo "  Merging LoRA → $TUNE"
            "$PYTHON_PATH" -u merge_lora.py \
                --base    "$CKPT" \
                --adapter "$TUNE" \
                --output  "$TUNE" \
                || { echo "  [FAIL] merge_lora.py"; SKIPPED+=("$NAME (merge failed)"); continue; }
        elif [ "$HAS_FULL" = "1" ]; then
            echo "  Already merged: $TUNE"
        else
            echo "  [SKIP] neither full model nor adapter found in $TUNE"
            SKIPPED+=("$NAME (no model files)")
            continue
        fi
    else
        if [ ! -d "$CKPT" ]; then
            echo "  [SKIP] ckpt not found: $CKPT"
            SKIPPED+=("$NAME (ckpt missing)")
            continue
        fi
        MODEL_DIR="$CKPT"
        echo "  Pruned-only (no post-training): $CKPT"
    fi

    FP32_OUT="cpu_fp32_${NAME}.json"
    INT8_OUT="cpu_int8_${NAME}.json"

    # ── FP32 baseline ──────────────────────────────────────────────────────
    echo ""
    echo "  --- Step 1: FP32 baseline (no quant, no compile) ---"
    "$PYTHON_PATH" -u infer_cpu.py \
        --model      "$MODEL_DIR" \
        --dataset    "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --no_quant --no_compile \
        --output     "$FP32_OUT" \
        || { echo "  [FAIL] FP32"; SKIPPED+=("$NAME (fp32 failed)"); continue; }

    # ── INT8 + compile ─────────────────────────────────────────────────────
    echo ""
    echo "  --- Step 2: INT8 dynamic + torch.compile ---"
    "$PYTHON_PATH" -u infer_cpu.py \
        --model      "$MODEL_DIR" \
        --dataset    "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --output     "$INT8_OUT" \
        || { echo "  [FAIL] INT8"; SKIPPED+=("$NAME (int8 failed)"); continue; }

    # ── Collect metrics ────────────────────────────────────────────────────
    SIZE_MB=$(disk_mb "$MODEL_DIR")
    ROW=$("$PYTHON_PATH" -c "
import json
fp = json.load(open('${FP32_OUT}'))
it = json.load(open('${INT8_OUT}'))
speedup  = fp['avg_latency_s'] / it['avg_latency_s']
fp_wer   = fp.get('wer', float('nan')) * 100
it_wer   = it.get('wer', float('nan')) * 100
dwer     = it_wer - fp_wer
fp_ram   = fp.get('ram_mb', 0)
it_ram   = it.get('ram_mb', 0)
print(f\"{fp['avg_latency_s']:.2f}|{it['avg_latency_s']:.2f}|{speedup:.2f}|{fp_wer:.2f}|{it_wer:.2f}|{dwer:+.2f}|{fp_ram:.0f}|{it_ram:.0f}\")
")
    RESULTS+=("${NAME}|${SIZE_MB}|${ROW}")
    echo ""
done

# ── Summary table ─────────────────────────────────────────────────────────
echo ""
echo "============================================================================================================================================================================================"
echo "  CPU Benchmark Summary — v3 mid-pruning, ${NUM_SAMPLES} samples (IMDA PART1)"
echo "  FP32 baseline vs INT8 dynamic (torch.quantization.quantize_dynamic) + torch.compile"
echo "  Original model disk size: ${ORIG_MB} MB  |  Disk size reflects pruning only; INT8 RAM ≈ disk×0.5 at runtime"
echo "============================================================================================================================================================================================"
printf "  %-28s %8s %8s %10s %10s %8s %8s %8s %8s %10s %10s\n" \
    "Config" "Disk(MB)" "vs.orig" "FP32 lat" "INT8 lat" "Speedup" "FP32 WER" "INT8 WER" "ΔWER" "FP32 RAM" "INT8 RAM"
echo "  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

for row in "${RESULTS[@]}"; do
    IFS='|' read -r name size_mb fp32_lat int8_lat speedup fp32_wer int8_wer dwer fp32_ram int8_ram <<< "$row"
    if [ -n "$ORIG_MB" ] && [ "$ORIG_MB" -gt 0 ] 2>/dev/null; then
        vs_orig=$("$PYTHON_PATH" -c "print(f'{int(\"${size_mb}\") / int(\"${ORIG_MB}\") * 100:.0f}%')" 2>/dev/null || echo "n/a")
    else
        vs_orig="n/a"
    fi
    printf "  %-28s %8s %8s %10s %10s %8s %8s %8s %8s %10s %10s\n" \
        "$name" "${size_mb}MB" "$vs_orig" "${fp32_lat}s" "${int8_lat}s" "${speedup}x" \
        "${fp32_wer}%" "${int8_wer}%" "$dwer%" "${fp32_ram}MB" "${int8_ram}MB"
done

echo "  ============================================================================================================================================================================================"
echo ""

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "  Skipped configs:"
    for s in "${SKIPPED[@]}"; do
        echo "    - $s"
    done
    echo ""
fi

echo "  Per-config JSON results:"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r name _ <<< "$row"
    echo "    cpu_fp32_${name}.json  cpu_int8_${name}.json"
done
echo ""
