#!/bin/bash
# ============================================================
# vLLM WER evaluation for:
#   v3-td50-mid3-23  (protected layers 0-2, 23-25)
#   v3-td50-mid4-23  (protected layers 0-3, 23-25)
#
# Mirrors merge_and_eval_all.sh but uses vllm_eval_wer.py
# (supports pruned models with non-uniform KV heads).
# ============================================================

export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"
export VLLM_USE_V1=0

GPU=${1:-0}
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
CKPT_ROOT="meralion_checkpoints"
TUNE_ROOT="meralion_tune_log"
NUM_SAMPLES=500   # samples 10500-11000 (held-out test region)
NUM_DEMO=10

CONFIGS=(
    "v3-td50-mid3-23"
    "v3-td50-mid4-23"
)

cd "$WORKDIR"
echo "GPU: $GPU"
echo "Samples: $NUM_SAMPLES  (indices 10500–$((10500 + NUM_SAMPLES - 1)))"
echo ""

for NAME in "${CONFIGS[@]}"; do
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
        echo "  Already merged: $TUNE"
    else
        echo "  [SKIP] no model files in $TUNE"
        continue
    fi

    echo "  Running vLLM WER eval on $NUM_SAMPLES samples ..."
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_PATH" -u vllm_eval_wer.py \
        --model      "$TUNE" \
        --dataset    "$DATASET" \
        --num_samples "$NUM_SAMPLES" \
        --num_demo    "$NUM_DEMO" \
        --output     "$OUT"

    if [ $? -eq 0 ] && [ -f "$OUT" ]; then
        WER=$("$PYTHON_PATH" -c "import json; d=json.load(open('$OUT')); print(f\"{d['wer']*100:.2f}%\")")
        echo "  WER: $WER  (saved to $OUT)"
    else
        echo "  [FAIL] eval"
    fi
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────
echo "=========================================="
echo "Summary"
echo "=========================================="
printf "  %-30s %s\n" "Config" "WER"
printf "  %-30s %s\n" "------" "---"
for NAME in "${CONFIGS[@]}"; do
    OUT="vllm_wer_${NAME}.json"
    if [ -f "$OUT" ]; then
        WER=$("$PYTHON_PATH" -c "import json; d=json.load(open('$OUT')); print(f\"{d['wer']*100:.2f}%\")")
        printf "  %-30s %s\n" "$NAME" "$WER"
    else
        printf "  %-30s %s\n" "$NAME" "(no result)"
    fi
done
echo ""
