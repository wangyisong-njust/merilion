#!/bin/bash
# ============================================================
# Merge LoRA + Evaluate WER for all 8 v2 models (8 GPUs parallel)
# ============================================================
# Usage: bash merge_and_eval_all.sh
#
# Each experiment runs on its own GPU (merge + eval sequentially per GPU,
# but all 8 GPUs run in parallel). Logs: eval_v2-<name>.log
# ============================================================

export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"

PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
VLLM_DIR="${WORKDIR}/vllm_inference"

cd $WORKDIR

# experiment_name -> GPU mapping
EXPERIMENTS=(
    "v2-TextAttn-25:0"
    "v2-TextMLP-25:1"
    "v2-TextBoth-25:2"
    "v2-td25-wa10:3"
    "v2-td25-wa15:4"
    "v2-td25-wa10-wm10:5"
    "v2-ta125-tm15:6"
    "v2-ta25-tm35:7"
)

PIDS=()

for ENTRY in "${EXPERIMENTS[@]}"; do
    NAME="${ENTRY%%:*}"
    GPU="${ENTRY##*:}"

    PRUNED_MODEL="meralion_checkpoints/MERaLiON-2-3B-${NAME}"
    LORA_DIR="meralion_tune_log/MERaLiON-2-3B-${NAME}-tune/best_model"
    MERGED_OUTPUT="meralion_checkpoints/MERaLiON-2-3B-${NAME}-merged"
    MODEL_EVAL_NAME="MERaLiON-2-3B-${NAME}-merged"
    LOG_FILE="eval_${NAME}.log"

    # Check if pruned model exists
    if [ ! -d "$PRUNED_MODEL" ]; then
        echo "[GPU $GPU] SKIP $NAME: Pruned model not found"
        continue
    fi

    # Check if LoRA adapter exists
    if [ ! -d "$LORA_DIR" ]; then
        LORA_DIR="meralion_tune_log/MERaLiON-2-3B-${NAME}-tune"
        if [ ! -f "${LORA_DIR}/adapter_model.safetensors" ] && [ ! -f "${LORA_DIR}/adapter_model.bin" ]; then
            echo "[GPU $GPU] SKIP $NAME: LoRA adapter not found"
            continue
        fi
    fi

    echo "[GPU $GPU] Launching $NAME -> $LOG_FILE"

    (
        cd $WORKDIR
        echo "=========================================="
        echo "[${NAME}] GPU $GPU - Start $(date)"
        echo "=========================================="

        # Step 1: Merge
        if [ -d "$MERGED_OUTPUT" ] && [ -f "${MERGED_OUTPUT}/model.safetensors" -o -f "${MERGED_OUTPUT}/model-00001-of-00002.safetensors" ]; then
            echo "[${NAME}] Merged model exists, skipping merge"
        else
            echo "[${NAME}] Merging..."
            CUDA_VISIBLE_DEVICES=$GPU $PYTHON_PATH merge_meralion.py \
                --ckpt "$PRUNED_MODEL" \
                --lora_ckpt "$LORA_DIR" \
                --save_path "$MERGED_OUTPUT"
            if [ $? -ne 0 ]; then
                echo "[${NAME}] ERROR: Merge failed"
                exit 1
            fi
            echo "[${NAME}] Merge complete"
        fi

        # Step 1.5: Fix config.json for vLLM (idempotent, always run)
        $PYTHON_PATH merge_meralion.py --fix_config "$MERGED_OUTPUT"

        # Step 2: vLLM eval
        echo "[${NAME}] Running vLLM eval..."
        cd $VLLM_DIR
        bash eval.sh imda_part1_asr_test $MODEL_EVAL_NAME $GPU 1 True wer -1

        if [ $? -eq 0 ]; then
            SCORE_FILE="log_for_all_models/${MODEL_EVAL_NAME}/imda_part1_asr_test_wer_score.json"
            if [ -f "$SCORE_FILE" ]; then
                echo "[${NAME}] WER: $(python3 -c "import json; print(json.load(open('${SCORE_FILE}'))['wer'])")"
            fi
        else
            echo "[${NAME}] ERROR: Eval failed"
        fi

        echo "[${NAME}] Done $(date)"
    ) > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "=========================================="
echo "All 8 experiments launched on GPU 0-7"
echo "=========================================="
echo "Logs: eval_v2-*.log"
echo "Monitor: tail -f eval_v2-*.log"
echo ""

# Wait for all to finish
echo "Waiting for all experiments to complete..."
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "Results summary:"
for ENTRY in "${EXPERIMENTS[@]}"; do
    NAME="${ENTRY%%:*}"
    SCORE_FILE="${VLLM_DIR}/log_for_all_models/MERaLiON-2-3B-${NAME}-merged/imda_part1_asr_test_wer_score.json"
    if [ -f "$SCORE_FILE" ]; then
        WER=$(python3 -c "import json; print(f'{json.load(open(\"${SCORE_FILE}\"))[\"wer\"]:.5f}')")
        echo "  ${NAME}: WER = ${WER}"
    else
        echo "  ${NAME}: (no result)"
    fi
done
