#!/bin/bash
# ============================================================
# Merge LoRA + Evaluate WER for all 8 v2 pruned+finetuned models
# ============================================================
# Usage: bash merge_and_eval_all.sh [gpu_id]
#   gpu_id: GPU to use for merge+eval (default: 0)
#
# This script:
#   1. Merges LoRA adapter into pruned model for each experiment
#   2. Runs vllm_inference WER evaluation on IMDA PART1
# ============================================================

export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"

GPU=${1:-0}
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
VLLM_DIR="${WORKDIR}/vllm_inference"

cd $WORKDIR

# All 8 v2 experiment names
EXPERIMENTS=(
    "v2-TextAttn-25"
    "v2-TextMLP-25"
    "v2-TextBoth-25"
    "v2-td25-wa10"
    "v2-td25-wa15"
    "v2-td25-wa10-wm10"
    "v2-ta125-tm15"
    "v2-ta25-tm35"
)

echo "=========================================="
echo "Merge + Eval Pipeline (GPU $GPU)"
echo "=========================================="

for NAME in "${EXPERIMENTS[@]}"; do
    PRUNED_MODEL="meralion_checkpoints/MERaLiON-2-3B-${NAME}"
    LORA_DIR="meralion_tune_log/MERaLiON-2-3B-${NAME}-tune/best_model"
    MERGED_OUTPUT="meralion_checkpoints/MERaLiON-2-3B-${NAME}-merged"
    MODEL_EVAL_NAME="MERaLiON-2-3B-${NAME}-merged"

    echo ""
    echo "=========================================="
    echo "[${NAME}] Starting merge + eval"
    echo "=========================================="

    # Check if pruned model exists
    if [ ! -d "$PRUNED_MODEL" ]; then
        echo "[${NAME}] SKIP: Pruned model not found at ${PRUNED_MODEL}"
        continue
    fi

    # Check if LoRA adapter exists (try best_model first, then output_dir root)
    if [ ! -d "$LORA_DIR" ]; then
        # Fallback: check if adapter is at the tune output dir root
        LORA_DIR="meralion_tune_log/MERaLiON-2-3B-${NAME}-tune"
        if [ ! -f "${LORA_DIR}/adapter_model.safetensors" ] && [ ! -f "${LORA_DIR}/adapter_model.bin" ]; then
            echo "[${NAME}] SKIP: LoRA adapter not found"
            continue
        fi
    fi

    # Step 1: Merge (skip if already merged)
    if [ -d "$MERGED_OUTPUT" ] && [ -f "${MERGED_OUTPUT}/model.safetensors" -o -f "${MERGED_OUTPUT}/model-00001-of-00002.safetensors" ]; then
        echo "[${NAME}] Merged model already exists, skipping merge"
    else
        echo "[${NAME}] Merging: ${PRUNED_MODEL} + ${LORA_DIR} -> ${MERGED_OUTPUT}"
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON_PATH merge_meralion.py \
            --ckpt "$PRUNED_MODEL" \
            --lora_ckpt "$LORA_DIR" \
            --save_path "$MERGED_OUTPUT"

        if [ $? -ne 0 ]; then
            echo "[${NAME}] ERROR: Merge failed, skipping eval"
            continue
        fi
        echo "[${NAME}] Merge complete"
    fi

    # Step 2: Evaluate WER using vllm_inference
    echo "[${NAME}] Running WER evaluation..."
    cd $VLLM_DIR

    DATASET=imda_part1_asr_test
    BATCH_SIZE=1
    OVERWRITE=True
    METRICS=wer
    NUMBER_OF_SAMPLES=-1

    bash eval.sh $DATASET $MODEL_EVAL_NAME $GPU $BATCH_SIZE $OVERWRITE $METRICS $NUMBER_OF_SAMPLES

    if [ $? -ne 0 ]; then
        echo "[${NAME}] ERROR: Eval failed"
    else
        echo "[${NAME}] Eval complete"
        # Print the WER score
        SCORE_FILE="log_for_all_models/${MODEL_EVAL_NAME}/imda_part1_asr_test_wer_score.json"
        if [ -f "$SCORE_FILE" ]; then
            echo "[${NAME}] WER: $(python3 -c "import json; print(json.load(open('${SCORE_FILE}'))['wer'])")"
        fi
    fi

    cd $WORKDIR
done

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "Results summary:"
for NAME in "${EXPERIMENTS[@]}"; do
    SCORE_FILE="${VLLM_DIR}/log_for_all_models/MERaLiON-2-3B-${NAME}-merged/imda_part1_asr_test_wer_score.json"
    if [ -f "$SCORE_FILE" ]; then
        WER=$(python3 -c "import json; print(f'{json.load(open(\"${SCORE_FILE}\"))[\"wer\"]:.5f}')")
        echo "  ${NAME}: WER = ${WER}"
    else
        echo "  ${NAME}: (no result)"
    fi
done
