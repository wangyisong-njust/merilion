#!/bin/bash
# ============================================================
# CPU benchmark — ALL v3 mid-pruning configurations
#
# Step 1: original MERaLiON-2-3B  FP32 / INT8 / INT4 baselines (run once each)
# Step 2+ (per-config): pruned+tuned model  INT8  and  INT4+compile
#
# Table 1: Original model — FP32 vs INT8 vs INT4
# Table 2: Per-config   — Orig-FP32 ref | pruned-INT8 | pruned-INT4
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
    # "v3-td50-mid4-22:0"
    "v3-td50-mid4-23:1"
    # "v3-td50-mid5-19:1"
    # "v3-td50-mid5-19-wm25:1"
    # "v3-td50-mid6-20:1"
    # "v3-td50-mid6-20-wm15:1"
    # "v3-td50-mid7-21:1"
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

# ── Result accumulators ───────────────────────────────────────────────────
RESULTS=()
SKIPPED=()

# ── Step 1a: original model FP32 baseline (run once) ─────────────────────
ORIG_FP32_OUT="cpu_fp32_original.json"
if [ -f "$ORIG_FP32_OUT" ]; then
    echo "Step 1a: original FP32 baseline already exists ($ORIG_FP32_OUT), skipping."
else
    echo "Step 1a: running original MERaLiON-2-3B FP32 baseline …"
    "$PYTHON_PATH" -u infer_cpu.py \
        --model            "$ORIGINAL" \
        --dataset          "$DATASET" \
        --num_samples      "$NUM_SAMPLES" \
        --trust_remote_code \
        --no_quant \
        --no_compile \
        --output           "$ORIG_FP32_OUT" \
        || { echo "[FAIL] original FP32 baseline — aborting"; exit 1; }
fi
echo ""

# ── Step 1b: original model INT8 (run once) ───────────────────────────────
ORIG_INT8_OUT="cpu_int8_original.json"
if [ -f "$ORIG_INT8_OUT" ]; then
    echo "Step 1b: original INT8 already exists ($ORIG_INT8_OUT), skipping."
else
    echo "Step 1b: running original MERaLiON-2-3B INT8 …"
    "$PYTHON_PATH" -u infer_cpu.py \
        --model            "$ORIGINAL" \
        --dataset          "$DATASET" \
        --num_samples      "$NUM_SAMPLES" \
        --trust_remote_code \
        --output           "$ORIG_INT8_OUT" \
        || { echo "[FAIL] original INT8 — skipping"; }
fi
echo ""

# ── Step 1c: original model INT4+compile (run once) ───────────────────────
ORIG_INT4_OUT="cpu_int4_original.json"
if [ -f "$ORIG_INT4_OUT" ]; then
    echo "Step 1c: original INT4 already exists ($ORIG_INT4_OUT), skipping."
else
    echo "Step 1c: running original MERaLiON-2-3B INT4+compile …"
    "$PYTHON_PATH" -u infer_cpu.py \
        --model            "$ORIGINAL" \
        --dataset          "$DATASET" \
        --num_samples      "$NUM_SAMPLES" \
        --int4 \
        --output           "$ORIG_INT4_OUT" \
        || { echo "[FAIL] original INT4 — skipping"; }
fi
echo ""

# ── Step 2+: per-config INT8 + INT4 benchmark ────────────────────────────
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
    INT4_OUT="cpu_int4_${NAME}.json"

    echo ""
    echo "  --- INT8 dynamic quantization ---"
    "$PYTHON_PATH" -u infer_cpu.py \
        --model       "$MODEL_DIR" \
        --dataset     "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --output      "$INT8_OUT" \
        || { echo "  [FAIL] INT8"; SKIPPED+=("$NAME (int8 failed)"); continue; }

    echo ""
    echo "  --- INT4 + torch.compile ---"
    "$PYTHON_PATH" -u infer_cpu.py \
        --model       "$MODEL_DIR" \
        --dataset     "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --int4 \
        --output      "$INT4_OUT" \
        || { echo "  [FAIL] INT4"; SKIPPED+=("$NAME (int4 failed)"); continue; }

    SIZE_MB=$(disk_mb "$MODEL_DIR")
    ROW=$("$PYTHON_PATH" -c "
import json
orig = json.load(open('${ORIG_FP32_OUT}'))
it8  = json.load(open('${INT8_OUT}'))
it4  = json.load(open('${INT4_OUT}'))
orig_lat  = orig['avg_latency_s']
int8_lat  = it8['avg_latency_s']
int4_lat  = it4['avg_latency_s']
spd8      = orig_lat / int8_lat
spd4      = orig_lat / int4_lat
orig_wer  = orig.get('wer', float('nan')) * 100
int8_wer  = it8.get('wer', float('nan')) * 100
int4_wer  = it4.get('wer', float('nan')) * 100
d8        = int8_wer - orig_wer
d4        = int4_wer - orig_wer
orig_ram  = orig.get('ram_mb', 0)
int8_ram  = it8.get('ram_mb', 0)
int4_ram  = it4.get('ram_mb', 0)
print(f'{orig_lat:.2f}|{int8_lat:.2f}|{spd8:.2f}|{int4_lat:.2f}|{spd4:.2f}|{orig_wer:.2f}|{int8_wer:.2f}|{d8:+.2f}|{int4_wer:.2f}|{d4:+.2f}|{orig_ram:.0f}|{int8_ram:.0f}|{int4_ram:.0f}')
")
    RESULTS+=("${NAME}|${SIZE_MB}|${ROW}")
    echo ""
done

# ── Summary table 1: original model ───────────────────────────────────────
echo ""
echo "============================================================"
echo "  Original MERaLiON-2-3B  —  FP32 / INT8 / INT4+compile"
echo "============================================================"
"$PYTHON_PATH" -c "
import json, os
rows = []
for tag, fname in [('FP32 (no compile)', '${ORIG_FP32_OUT}'),
                   ('INT8 (no compile)', '${ORIG_INT8_OUT}'),
                   ('INT4 + compile',    '${ORIG_INT4_OUT}')]:
    if not os.path.exists(fname):
        rows.append((tag, 'N/A', 'N/A', 'N/A'))
        continue
    d = json.load(open(fname))
    rows.append((tag,
                 f\"{d['avg_latency_s']:.2f}s\",
                 f\"{d.get('wer', float('nan'))*100:.2f}%\",
                 f\"{d.get('ram_mb', 0):.0f}MB\"))
print(f'  {\"Method\":<22} {\"Avg Lat\":>9} {\"WER\":>8} {\"RAM\":>9}')
print('  ' + '-'*52)
for t, l, w, r in rows:
    print(f'  {t:<22} {l:>9} {w:>8} {r:>9}')
" 2>/dev/null || echo "  (run steps 1a-1c to populate)"
echo ""

# ── Summary table 2: per-config ───────────────────────────────────────────
echo ""
echo "$(python3 -c "print('='*160)")"
echo "  CPU Benchmark — v3 mid-pruning  (${NUM_SAMPLES} samples, IMDA PART1)"
echo "  Baseline: original MERaLiON-2-3B FP32 (no compile)  |  Original disk: ${ORIG_MB} MB"
echo "$(python3 -c "print('='*160)")"
printf "  %-28s %8s %8s %10s %8s %8s %10s %8s %9s %8s %9s %8s %10s %10s %10s\n" \
    "Config" "Disk(MB)" "vs.orig" \
    "FP32 lat" \
    "INT8 lat" "Spd8x" \
    "INT4 lat" "Spd4x" \
    "OrigWER" \
    "INT8WER" "ΔWER8" \
    "INT4WER" "ΔWER4" \
    "INT8 RAM" "INT4 RAM"
echo "  $(python3 -c "print('-'*156)")"

for row in "${RESULTS[@]}"; do
    IFS='|' read -r name size_mb \
        orig_lat int8_lat spd8 int4_lat spd4 \
        orig_wer int8_wer dw8 int4_wer dw4 \
        orig_ram int8_ram int4_ram <<< "$row"
    vs_orig=$("$PYTHON_PATH" -c "
try:
    print(f'{int(\"${size_mb}\") / int(\"${ORIG_MB}\") * 100:.0f}%')
except:
    print('n/a')
" 2>/dev/null)
    printf "  %-28s %8s %8s %10s %8s %8s %10s %8s %9s %8s %9s %8s %10s %10s %10s\n" \
        "$name" "${size_mb}MB" "$vs_orig" \
        "${orig_lat}s" \
        "${int8_lat}s" "${spd8}x" \
        "${int4_lat}s" "${spd4}x" \
        "${orig_wer}%" \
        "${int8_wer}%" "$dw8%" \
        "${int4_wer}%" "$dw4%" \
        "${int8_ram}MB" "${int4_ram}MB"
done

echo "  $(python3 -c "print('='*156)")"
echo ""

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "  Skipped:"
    for s in "${SKIPPED[@]}"; do echo "    - $s"; done
    echo ""
fi

echo "  JSON results:"
echo "    Orig FP32: $ORIG_FP32_OUT"
echo "    Orig INT8: $ORIG_INT8_OUT"
echo "    Orig INT4: $ORIG_INT4_OUT"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r name _ <<< "$row"
    echo "    cpu_int8_${name}.json  |  cpu_int4_${name}.json"
done
echo ""
