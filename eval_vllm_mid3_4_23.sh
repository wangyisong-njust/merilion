#!/bin/bash
# ============================================================
# vLLM WER evaluation for:
#   original MERaLiON-2-3B  (baseline)
#   v3-td50-mid3-23  (protected layers 0-2, 23-25)
#   v3-td50-mid4-23  (protected layers 0-3, 23-25)
#
# Uses all available test samples (10500 to end of shuffled dataset).
# ============================================================

export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"
export VLLM_USE_V1=0

GPU=${1:-0}
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
CKPT_ROOT="meralion_checkpoints"
TUNE_ROOT="meralion_tune_log"
NUM_SAMPLES=-1   # -1 = all available test samples (index 10500 to end)
NUM_DEMO=5

PRUNED_CONFIGS=(
    "v3-td50-mid3-23"
    "v3-td50-mid4-23"
)

cd "$WORKDIR"
echo "GPU: $GPU  |  dataset: all test samples (num_samples=-1)"
echo ""

# ── Helper: run eval for one model ───────────────────────────────────────
run_eval() {
    local label="$1"
    local model_path="$2"
    local out="$3"

    if [ -f "$out" ]; then
        echo "  [$label] result exists ($out), skipping."
        return 0
    fi

    echo "  [$label] running vLLM WER eval …"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_PATH" -u vllm_eval_wer.py \
        --model      "$model_path" \
        --dataset    "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --num_demo    "$NUM_DEMO" \
        --output     "$out"

    if [ $? -eq 0 ] && [ -f "$out" ]; then
        WER=$("$PYTHON_PATH" -c "import json; d=json.load(open('$out')); print(f\"{d['wer']*100:.2f}%\")")
        echo "  [$label] WER: $WER  →  $out"
    else
        echo "  [$label] FAILED"
        return 1
    fi
}

# ── Step 1: original model (baseline) ────────────────────────────────────
echo "=========================================="
echo "Baseline: MERaLiON-2-3B (original)"
echo "=========================================="
run_eval "original" "$ORIGINAL" "vllm_wer_original.json"
echo ""

# ── Step 2+: pruned + tuned configs ──────────────────────────────────────
for NAME in "${PRUNED_CONFIGS[@]}"; do
    CKPT="${CKPT_ROOT}/MERaLiON-2-3B-${NAME}"
    TUNE="${TUNE_ROOT}/MERaLiON-2-3B-${NAME}-tune"
    OUT="vllm_wer_${NAME}.json"

    echo "=========================================="
    echo "[$NAME]"
    echo "=========================================="

    if [ ! -d "$TUNE" ]; then
        echo "  [SKIP] tune dir not found: $TUNE"
        continue
    fi

    # Merge LoRA if not yet merged
    HAS_FULL=0
    ls "${TUNE}"/model*.safetensors 2>/dev/null | grep -q . && HAS_FULL=1
    ls "${TUNE}"/pytorch_model*.bin  2>/dev/null | grep -q . && HAS_FULL=1

    if [ "$HAS_FULL" = "0" ] && [ -f "${TUNE}/adapter_config.json" ]; then
        if [ ! -d "$CKPT" ]; then
            echo "  [SKIP] base ckpt not found: $CKPT"
            continue
        fi
        echo "  Merging LoRA → $TUNE ..."
        "$PYTHON_PATH" -u merge_lora.py \
            --base    "$CKPT" \
            --adapter "$TUNE" \
            --output  "$TUNE" \
            || { echo "  [FAIL] merge"; continue; }
        echo "  Merge done."
    elif [ "$HAS_FULL" = "1" ]; then
        echo "  Already merged."
    else
        echo "  [SKIP] no model files in $TUNE"
        continue
    fi

    run_eval "$NAME" "$TUNE" "$OUT"
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────
echo "=========================================="
echo "WER Summary  (all test samples)"
echo "=========================================="
printf "  %-32s %10s %10s\n" "Model" "WER" "ΔWER vs orig"
printf "  %-32s %10s %10s\n" "-----" "---" "------------"

ORIG_WER=$("$PYTHON_PATH" -c "
import json, sys
try:
    d = json.load(open('vllm_wer_original.json'))
    print(f\"{d['wer']*100:.2f}\")
except:
    print('N/A')
" 2>/dev/null)

print_row() {
    local label="$1"
    local out="$2"
    "$PYTHON_PATH" -c "
import json, sys
try:
    d = json.load(open('$out'))
    wer = d['wer'] * 100
    orig = float('$ORIG_WER') if '$ORIG_WER' != 'N/A' else None
    delta = f'{wer - orig:+.2f}%' if orig is not None else 'N/A'
    print(f'  {\"$label\":<32} {wer:>9.2f}% {delta:>10}')
except Exception as e:
    print(f'  {\"$label\":<32} {\"(no result)\":>10}')
" 2>/dev/null
}

print_row "MERaLiON-2-3B (original)" "vllm_wer_original.json"
for NAME in "${PRUNED_CONFIGS[@]}"; do
    print_row "$NAME" "vllm_wer_${NAME}.json"
done
echo ""
