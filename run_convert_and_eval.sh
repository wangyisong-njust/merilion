#!/bin/bash
# ============================================================
# Convert pruned models to vLLM-compatible format, then eval
# Truncates unpruned layers (0-3) to match pruned layer sizes
# so all layers have uniform dimensions for vLLM
# ============================================================

export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"

PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
VLLM_DIR="${WORKDIR}/vllm_inference"

cd $WORKDIR

# 6 experiments that failed with vLLM (TextBoth-25 and TextMLP-25 already work)
declare -A EXP_GPU
EXP_GPU["TextAttn-25"]=0
EXP_GPU["td25-wa10"]=2
EXP_GPU["td25-wa15"]=3
EXP_GPU["td25-wa10-wm10"]=4
EXP_GPU["ta125-tm15"]=5
EXP_GPU["ta25-tm35"]=6

for NAME in "${!EXP_GPU[@]}"; do
    GPU=${EXP_GPU[$NAME]}
    MERGED="meralion_checkpoints/MERaLiON-2-3B-${NAME}-merged"
    VLLM_MODEL="meralion_checkpoints/MERaLiON-2-3B-${NAME}-merged-vllm"
    EVAL_NAME="MERaLiON-2-3B-${NAME}-merged-vllm"
    LOGFILE="eval_${NAME}_vllm.log"

    if [ ! -d "$MERGED" ]; then
        echo "[GPU $GPU] SKIP $NAME: merged model not found"
        continue
    fi

    echo "[GPU $GPU] Launching $NAME -> $LOGFILE"

    nohup bash -c "
        cd $WORKDIR

        # Step 1: Convert to vLLM-compatible (truncate unpruned layers)
        if [ -d '$VLLM_MODEL' ] && [ \$(ls '$VLLM_MODEL'/*.safetensors 2>/dev/null | wc -l) -gt 0 ]; then
            echo '[${NAME}] vLLM model already exists, skipping convert'
        else
            echo '[${NAME}] Converting to vLLM-compatible format...'
            $PYTHON_PATH convert_to_vllm_compat.py --input $MERGED --output $VLLM_MODEL
            if [ \$? -ne 0 ]; then
                echo '[${NAME}] ERROR: Convert failed'
                exit 1
            fi
        fi

        # Step 2: Eval with vLLM
        echo '[${NAME}] Running vLLM WER evaluation...'
        cd $VLLM_DIR
        CUDA_VISIBLE_DEVICES=$GPU bash eval.sh imda_part1_asr_test $EVAL_NAME $GPU 1 True wer -1
        echo '[${NAME}] Done'
    " > "$LOGFILE" 2>&1 &

    echo "[GPU $GPU] $NAME started (PID: $!)"
done

echo ""
echo "=========================================="
echo "All 6 convert+eval experiments launched"
echo "=========================================="
echo "监控: tail -f eval_*_vllm.log"
