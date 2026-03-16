#!/bin/bash
# ============================================================
# CPU benchmark — ALL v3 mid-pruning configurations
#
# Step 1 (once): original MERaLiON-2-3B FP32 baseline latency + WER
# Step 2 (per-config): pruned+tuned model with INT8 dynamic quantization
#
# Table columns: Disk(MB) | vs.orig | Orig FP32 lat | INT8 lat | Speedup
#                Orig WER | INT8 WER | ΔWER | Orig RAM | INT8 RAM
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
#   has_posttrain=1  → TUNE_ROOT/MERaLiON-2-3B-<name>-tune  (merge LoRA if needed)
#   has_posttrain=0  → CKPT_ROOT/MERaLiON-2-3B-<name>  (pruned, no fine-tune)
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

ORIG_MB=$(disk_mb "$ORIGINAL")
echo "Original model disk size: ${ORIG_MB} MB"
echo ""

# ── Result accumulator ────────────────────────────────────────────────────
RESULTS=()
SKIPPED=()

# ── Step 1: original model FP32 baseline (run once) ──────────────────────
ORIG_FP32_OUT="cpu_fp32_original.json"
if [ -f "$ORIG_FP32_OUT" ]; then
    echo "Step 1: original FP32 baseline already exists ($ORIG_FP32_OUT), skipping."
else
    echo "Step 1: running original MERaLiON-2-3B FP32 baseline …"
    "$PYTHON_PATH" -u infer_cpu.py \
        --model      "$ORIGINAL" \
        --dataset    "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --no_quant --no_compile \
        --output     "$ORIG_FP32_OUT" \
        || { echo "[FAIL] original FP32 baseline — aborting"; exit 1; }
fi
echo ""

# ── Step 2+: per-config INT8 benchmark ───────────────────────────────────
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

        # Merge LoRA if still an adapter (no full model weights)
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

    INT8_OUT="cpu_int8_${NAME}.json"

    echo ""
    echo "  --- INT8 dynamic quantization ---"
    "$PYTHON_PATH" -u infer_cpu.py \
        --model      "$MODEL_DIR" \
        --dataset    "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --output     "$INT8_OUT" \
        || { echo "  [FAIL] INT8"; SKIPPED+=("$NAME (int8 failed)"); continue; }

    SIZE_MB=$(disk_mb "$MODEL_DIR")
    ROW=$("$PYTHON_PATH" -c "
import json
orig = json.load(open('${ORIG_FP32_OUT}'))
it   = json.load(open('${INT8_OUT}'))
speedup   = orig['avg_latency_s'] / it['avg_latency_s']
orig_wer  = orig.get('wer', float('nan')) * 100
it_wer    = it.get('wer', float('nan')) * 100
dwer      = it_wer - orig_wer
orig_ram  = orig.get('ram_mb', 0)
it_ram    = it.get('ram_mb', 0)
print(f\"{orig['avg_latency_s']:.2f}|{it['avg_latency_s']:.2f}|{speedup:.2f}|{orig_wer:.2f}|{it_wer:.2f}|{dwer:+.2f}|{orig_ram:.0f}|{it_ram:.0f}\")
")
    RESULTS+=("${NAME}|${SIZE_MB}|${ROW}")
    echo ""
done

# ── Summary table ─────────────────────────────────────────────────────────
echo ""
echo "========================================================================================================================================================================================"
echo "  CPU Benchmark — v3 mid-pruning + INT8 dynamic quant  (${NUM_SAMPLES} samples, IMDA PART1)"
echo "  Baseline: original MERaLiON-2-3B FP32  |  Optimized: pruned+tuned+INT8"
echo "  Original disk: ${ORIG_MB} MB"
echo "========================================================================================================================================================================================"
printf "  %-28s %8s %8s %10s %10s %8s %9s %9s %8s %10s %10s\n" \
    "Config" "Disk(MB)" "vs.orig" "Orig lat" "INT8 lat" "Speedup" "Orig WER" "INT8 WER" "ΔWER" "Orig RAM" "INT8 RAM"
echo "  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

for row in "${RESULTS[@]}"; do
    IFS='|' read -r name size_mb orig_lat int8_lat speedup orig_wer int8_wer dwer orig_ram int8_ram <<< "$row"
    vs_orig=$("$PYTHON_PATH" -c "
try:
    print(f'{int(\"${size_mb}\") / int(\"${ORIG_MB}\") * 100:.0f}%')
except:
    print('n/a')
" 2>/dev/null)
    printf "  %-28s %8s %8s %10s %10s %8s %9s %9s %8s %10s %10s\n" \
        "$name" "${size_mb}MB" "$vs_orig" "${orig_lat}s" "${int8_lat}s" "${speedup}x" \
        "${orig_wer}%" "${int8_wer}%" "$dwer%" "${orig_ram}MB" "${int8_ram}MB"
done

echo "  ========================================================================================================================================================================================"
echo ""

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "  Skipped:"
    for s in "${SKIPPED[@]}"; do echo "    - $s"; done
    echo ""
fi

echo "  JSON results:"
echo "    Baseline: $ORIG_FP32_OUT"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r name _ <<< "$row"
    echo "    cpu_int8_${name}.json"
done
echo ""
