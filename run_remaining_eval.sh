#!/bin/bash
# ============================================================
# 并行 Merge + Eval 剩余 7 个实验 (TextBoth-25 已完成)
# ============================================================
# Usage: bash run_remaining_eval.sh
# 每个实验分配一个 GPU，并行运行
# ============================================================

export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"

PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"
VLLM_DIR="${WORKDIR}/vllm_inference"

cd $WORKDIR

# 实验名 -> GPU 映射 (TextBoth-25 已完成，跳过)
declare -A EXP_GPU
EXP_GPU["TextAttn-25"]=0
EXP_GPU["TextMLP-25"]=1
EXP_GPU["td25-wa10"]=2
EXP_GPU["td25-wa15"]=3
EXP_GPU["td25-wa10-wm10"]=4
EXP_GPU["ta125-tm15"]=5
EXP_GPU["ta25-tm35"]=6

for NAME in "${!EXP_GPU[@]}"; do
    GPU=${EXP_GPU[$NAME]}
    PRUNED="meralion_checkpoints/MERaLiON-2-3B-${NAME}"
    LORA="meralion_tune_log/MERaLiON-2-3B-${NAME}-tune/best_model"
    MERGED="meralion_checkpoints/MERaLiON-2-3B-${NAME}-merged"
    EVAL_NAME="MERaLiON-2-3B-${NAME}-merged"
    LOGFILE="eval_${NAME}.log"

    # 检查剪枝模型是否存在
    if [ ! -d "$PRUNED" ]; then
        echo "[GPU $GPU] SKIP $NAME: pruned model not found at $PRUNED"
        continue
    fi

    # 检查 LoRA adapter 是否存在
    if [ ! -d "$LORA" ]; then
        LORA="meralion_tune_log/MERaLiON-2-3B-${NAME}-tune"
        if [ ! -f "${LORA}/adapter_model.safetensors" ] && [ ! -f "${LORA}/adapter_model.bin" ]; then
            echo "[GPU $GPU] SKIP $NAME: LoRA adapter not found"
            continue
        fi
    fi

    echo "[GPU $GPU] Launching $NAME -> $LOGFILE"

    nohup bash -c "
        cd $WORKDIR

        # Step 1: Merge (skip if already exists)
        if [ -d '$MERGED' ] && [ \$(ls '$MERGED'/*.safetensors 2>/dev/null | wc -l) -gt 0 ]; then
            echo '[${NAME}] Merged model already exists, skipping merge'
        else
            echo '[${NAME}] Merging...'
            CUDA_VISIBLE_DEVICES=$GPU $PYTHON_PATH merge_meralion.py --ckpt $PRUNED --lora_ckpt $LORA --save_path $MERGED
            if [ \$? -ne 0 ]; then
                echo '[${NAME}] ERROR: Merge failed'
                exit 1
            fi
            echo '[${NAME}] Merge complete'
        fi

        # Step 2: Eval WER
        echo '[${NAME}] Running WER evaluation...'
        cd $VLLM_DIR
        CUDA_VISIBLE_DEVICES=$GPU bash eval.sh imda_part1_asr_test $EVAL_NAME $GPU 1 True wer -1
        echo '[${NAME}] Eval complete'
    " > "$LOGFILE" 2>&1 &

    echo "[GPU $GPU] $NAME started (PID: $!)"
done

echo ""
echo "=========================================="
echo "All 7 experiments launched in parallel"
echo "=========================================="
echo ""
echo "监控进度:"
echo "  tail -f eval_TextAttn-25.log"
echo "  tail -f eval_TextMLP-25.log"
echo "  tail -f eval_td25-wa10.log"
echo "  tail -f eval_td25-wa15.log"
echo "  tail -f eval_td25-wa10-wm10.log"
echo "  tail -f eval_ta125-tm15.log"
echo "  tail -f eval_ta25-tm35.log"
echo ""
echo "查看所有进度: tail -f eval_*.log"
echo "查看 WER 结果: grep -r '\"wer\"' ${VLLM_DIR}/log_for_all_models/MERaLiON-2-3B-*-merged/"
