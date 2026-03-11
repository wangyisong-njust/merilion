#!/bin/bash
# ============================================================
# Quantize pruned models (W4A16 / W8A16 RTN) + vLLM eval
# ============================================================
# Usage: bash quant_and_eval.sh [gpu_id]
#
# Quantizes v2-TextBoth-25 and v2-ta25-tm35 with W4A16 and W8A16,
# then runs vLLM eval on each quantized model serially.
# ============================================================

export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
VLLM_DIR="${WORKDIR}/vllm_inference"
QUANT_SCRIPT="${VLLM_DIR}/src/model_src/meralion_2_quant.py"
DATASET="imda_part1_asr_test"
CHECKPOINT_DIR="${WORKDIR}/meralion_checkpoints"

cd $WORKDIR

# Models to quantize (must already be merged + config fixed)
MODELS=(
    "MERaLiON-2-3B-v2-TextBoth-25-merged"
    "MERaLiON-2-3B-v2-ta25-tm35-merged"
)

# Quantization schemes
SCHEMES=("W8A16" "W4A16")

echo "=========================================="
echo "Quantization + Eval Pipeline (GPU $GPU)"
echo "=========================================="

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="${CHECKPOINT_DIR}/${MODEL_NAME}"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "[SKIP] ${MODEL_NAME}: merged model not found at ${MODEL_PATH}"
        continue
    fi

    for SCHEME in "${SCHEMES[@]}"; do
        QUANT_DIR="${CHECKPOINT_DIR}/${MODEL_NAME}-${SCHEME}-RTN"

        echo ""
        echo "=========================================="
        echo "[${MODEL_NAME}] Quantizing with ${SCHEME}..."
        echo "=========================================="

        # Quantize (skip if already done)
        # Must cd to vllm_inference/ so meralion2_bl_llmcompressor module is importable
        if [ -d "$QUANT_DIR" ] && [ "$(ls ${QUANT_DIR}/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "[${MODEL_NAME}] ${SCHEME} quantized model exists, skipping quantization"
        else
            cd $VLLM_DIR
            $PYTHON_PATH $QUANT_SCRIPT \
                --model_path "$MODEL_PATH" \
                --scheme "$SCHEME" \
                --save_dir "$QUANT_DIR"
            if [ $? -ne 0 ]; then
                echo "[${MODEL_NAME}] ERROR: ${SCHEME} quantization failed"
                cd $WORKDIR
                continue
            fi
            echo "[${MODEL_NAME}] ${SCHEME} quantization complete"
            cd $WORKDIR
        fi

        # vLLM eval
        echo "[${MODEL_NAME}-${SCHEME}] Running vLLM eval..."
        cd $VLLM_DIR
        bash eval.sh $DATASET $QUANT_DIR $GPU 1 True wer -1
        if [ $? -eq 0 ]; then
            SCORE_FILE="log_for_all_models/${QUANT_DIR}/${DATASET}_wer_score.json"
            # Try with basename as well
            QUANT_BASENAME=$(basename $QUANT_DIR)
            SCORE_FILE2="log_for_all_models/${QUANT_BASENAME}/${DATASET}_wer_score.json"
            for SF in "$SCORE_FILE" "$SCORE_FILE2"; do
                if [ -f "$SF" ]; then
                    echo "[${MODEL_NAME}-${SCHEME}] WER: $(python3 -c "import json; print(json.load(open('${SF}'))['wer'])")"
                    break
                fi
            done
        else
            echo "[${MODEL_NAME}-${SCHEME}] ERROR: Eval failed"
        fi
        cd $WORKDIR
    done
done

# ============================================================
# Results summary
# ============================================================
echo ""
echo "=========================================="
echo "Quantization Eval Complete!"
echo "=========================================="
echo ""
printf "%-50s %-10s %-15s %-12s %-8s %-10s\n" "Model" "WER" "Throughput" "Latency" "RTF" "Size"
printf "%-50s %-10s %-15s %-12s %-8s %-10s\n" "-----" "---" "----------" "-------" "---" "----"

for MODEL_NAME in "${MODELS[@]}"; do
    for SCHEME in "${SCHEMES[@]}"; do
        QUANT_BASENAME="${MODEL_NAME}-${SCHEME}-RTN"
        LABEL="${MODEL_NAME##MERaLiON-2-3B-}-${SCHEME}"

        # Check multiple possible log dirs
        for LOG_DIR in \
            "${VLLM_DIR}/log_for_all_models/${CHECKPOINT_DIR}/${QUANT_BASENAME}" \
            "${VLLM_DIR}/log_for_all_models/${QUANT_BASENAME}"; do
            SCORE_FILE="${LOG_DIR}/${DATASET}_wer_score.json"
            SPEED_FILE="${LOG_DIR}/${DATASET}_speed_metrics.json"
            if [ -f "$SCORE_FILE" ]; then
                break
            fi
        done

        if [ -f "$SCORE_FILE" ] && [ -f "$SPEED_FILE" ]; then
            ROW=$(python3 -c "
import json
s = json.load(open('${SCORE_FILE}'))
m = json.load(open('${SPEED_FILE}'))
wer = f\"{s['wer']:.5f}\"
tp = f\"{m['throughput_samples_per_sec']:.2f} s/s\"
lat = f\"{m['avg_latency_sec']:.3f} s\"
rtf = f\"{m.get('rtf', 'N/A')}\"
sz = f\"{m.get('model_size_gib', 'N/A')} GiB\"
print(f'{wer}|{tp}|{lat}|{rtf}|{sz}')
")
            IFS='|' read -r c1 c2 c3 c4 c5 <<< "$ROW"
            printf "%-50s %-10s %-15s %-12s %-8s %-10s\n" "$LABEL" "$c1" "$c2" "$c3" "$c4" "$c5"
        elif [ -f "$SCORE_FILE" ]; then
            WER=$(python3 -c "import json; print(f'{json.load(open(\"${SCORE_FILE}\"))[\"wer\"]:.5f}')")
            printf "%-50s %-10s\n" "$LABEL" "$WER"
        else
            printf "%-50s %-10s\n" "$LABEL" "(no result)"
        fi
    done
done
