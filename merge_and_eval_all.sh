#!/bin/bash
# ============================================================
# Serial merge + vLLM eval on single GPU (fair speed comparison)
# ============================================================
# Usage: bash merge_and_eval_all.sh [gpu_id]
#   gpu_id: GPU to use (default: 0)
#
# Runs baseline + 5 text-only pruned models serially on one GPU.
# Whisper-pruned models (td25-wa10, td25-wa15, td25-wa10-wm10)
# are excluded — vLLM cannot load pruned Whisper encoders.
# ============================================================

export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"

GPU=${1:-0}
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
VLLM_DIR="${WORKDIR}/vllm_inference"
DATASET="imda_part1_asr_test"

cd $WORKDIR

# 5 text-only pruned experiments (vLLM compatible)
EXPERIMENTS=(
    "v2-ta125-tm15"
    "v2-TextAttn-25"
    "v2-TextMLP-25"
    "v2-TextBoth-25"
    "v2-ta25-tm35"
)

BASELINE_MODEL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
BASELINE_NAME="Baseline-MERaLiON-2-3B"

echo "=========================================="
echo "Serial Eval Pipeline (GPU $GPU)"
echo "=========================================="

# ============================================================
# Step 0: Baseline (unpruned model)
# ============================================================
echo ""
echo "=========================================="
echo "[Baseline] Running vLLM eval..."
echo "=========================================="
cd $VLLM_DIR
bash eval.sh $DATASET $BASELINE_MODEL $GPU 1 True wer -1
if [ $? -eq 0 ]; then
    echo "[Baseline] Eval complete"
else
    echo "[Baseline] ERROR: Eval failed"
fi
cd $WORKDIR

# ============================================================
# Step 1-5: Pruned models (serial, same GPU)
# ============================================================
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

    # Check if LoRA adapter exists
    if [ ! -d "$LORA_DIR" ]; then
        LORA_DIR="meralion_tune_log/MERaLiON-2-3B-${NAME}-tune"
        if [ ! -f "${LORA_DIR}/adapter_model.safetensors" ] && [ ! -f "${LORA_DIR}/adapter_model.bin" ]; then
            echo "[${NAME}] SKIP: LoRA adapter not found"
            continue
        fi
    fi

    # Merge (skip if already merged)
    if [ -d "$MERGED_OUTPUT" ] && [ -f "${MERGED_OUTPUT}/model.safetensors" -o -f "${MERGED_OUTPUT}/model-00001-of-00002.safetensors" ]; then
        echo "[${NAME}] Merged model exists, skipping merge"
    else
        echo "[${NAME}] Merging..."
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

    # Fix config.json for vLLM (idempotent)
    $PYTHON_PATH merge_meralion.py --fix_config "$MERGED_OUTPUT"

    # vLLM eval
    echo "[${NAME}] Running vLLM eval..."
    cd $VLLM_DIR
    bash eval.sh $DATASET $MODEL_EVAL_NAME $GPU 1 True wer -1

    if [ $? -eq 0 ]; then
        SCORE_FILE="log_for_all_models/${MODEL_EVAL_NAME}/${DATASET}_wer_score.json"
        if [ -f "$SCORE_FILE" ]; then
            echo "[${NAME}] WER: $(python3 -c "import json; print(json.load(open('${SCORE_FILE}'))['wer'])")"
        fi
    else
        echo "[${NAME}] ERROR: Eval failed"
    fi

    cd $WORKDIR
done

# ============================================================
# Results summary
# ============================================================
echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "Results summary (single GPU $GPU, serial execution):"
printf "%-25s %-10s %-15s %-12s %-8s %-10s\n" "Experiment" "WER" "Throughput" "Latency" "RTF" "Size"
printf "%-25s %-10s %-15s %-12s %-8s %-10s\n" "----------" "---" "----------" "-------" "---" "----"

# Baseline
BASELINE_SCORE="${VLLM_DIR}/log_for_all_models/${BASELINE_MODEL}/${DATASET}_wer_score.json"
BASELINE_SPEED="${VLLM_DIR}/log_for_all_models/${BASELINE_MODEL}/${DATASET}_speed_metrics.json"
if [ -f "$BASELINE_SCORE" ] && [ -f "$BASELINE_SPEED" ]; then
    ROW=$(python3 -c "
import json
s = json.load(open('${BASELINE_SCORE}'))
m = json.load(open('${BASELINE_SPEED}'))
wer = f\"{s['wer']:.5f}\"
tp = f\"{m['throughput_samples_per_sec']:.2f} s/s\"
lat = f\"{m['avg_latency_sec']:.3f} s\"
rtf = f\"{m.get('rtf', 'N/A')}\"
print(f'Baseline (no prune)|{wer}|{tp}|{lat}|{rtf}')
")
    IFS='|' read -r c1 c2 c3 c4 c5 <<< "$ROW"
    printf "%-25s %-10s %-15s %-12s %-8s\n" "$c1" "$c2" "$c3" "$c4" "$c5"
fi

# Pruned models
for NAME in "${EXPERIMENTS[@]}"; do
    MODEL_DIR="${VLLM_DIR}/log_for_all_models/MERaLiON-2-3B-${NAME}-merged"
    SCORE_FILE="${MODEL_DIR}/${DATASET}_wer_score.json"
    SPEED_FILE="${MODEL_DIR}/${DATASET}_speed_metrics.json"
    if [ -f "$SCORE_FILE" ] && [ -f "$SPEED_FILE" ]; then
        ROW=$(python3 -c "
import json
s = json.load(open('${SCORE_FILE}'))
m = json.load(open('${SPEED_FILE}'))
wer = f\"{s['wer']:.5f}\"
tp = f\"{m['throughput_samples_per_sec']:.2f} s/s\"
lat = f\"{m['avg_latency_sec']:.3f} s\"
rtf = f\"{m.get('rtf', 'N/A')}\"
print(f'${NAME}|{wer}|{tp}|{lat}|{rtf}')
")
        IFS='|' read -r c1 c2 c3 c4 c5 <<< "$ROW"
        printf "%-25s %-10s %-15s %-12s %-8s\n" "$c1" "$c2" "$c3" "$c4" "$c5"
    elif [ -f "$SCORE_FILE" ]; then
        WER=$(python3 -c "import json; print(f'{json.load(open(\"${SCORE_FILE}\"))[\"wer\"]:.5f}')")
        printf "%-25s %-10s %-15s %-12s %-8s\n" "$NAME" "$WER" "N/A" "N/A" "N/A"
    else
        printf "%-25s %-10s %-15s %-12s %-8s\n" "$NAME" "(no result)" "" "" ""
    fi
done
