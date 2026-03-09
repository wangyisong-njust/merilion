#!/bin/bash
# ============================================================
# Eval 6 pruned models that failed with vLLM (using HuggingFace inference)
# Models with "pruned" in name → HuggingFace loader (handles midblock)
# ============================================================

export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"

WORKDIR="/home/jinchao/runtao/LLM-Pruner"
VLLM_DIR="${WORKDIR}/vllm_inference"

cd $WORKDIR

# These 6 experiments failed with vLLM due to non-uniform layer dimensions
# We rename them with "-pruned" suffix to route to HuggingFace inference
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

    if [ ! -d "$MERGED" ]; then
        echo "[GPU $GPU] SKIP $NAME: merged model not found at $MERGED"
        continue
    fi

    # Use "-pruned" suffix to route to HuggingFace loader
    EVAL_NAME="MERaLiON-2-3B-${NAME}-merged-pruned"
    LOGFILE="eval_${NAME}_hf.log"

    echo "[GPU $GPU] Launching $NAME (HF inference) -> $LOGFILE"

    nohup bash -c "cd $VLLM_DIR && CUDA_VISIBLE_DEVICES=$GPU bash eval.sh imda_part1_asr_test $EVAL_NAME $GPU 1 True wer -1" > "$LOGFILE" 2>&1 &

    echo "[GPU $GPU] $NAME started (PID: $!)"
done

echo ""
echo "=========================================="
echo "All 6 HuggingFace eval experiments launched"
echo "=========================================="
echo ""
echo "监控: tail -f eval_*_hf.log"
echo "查看结果: for f in eval_*_hf.log; do echo \"=== \$f ===\"; tail -5 \"\$f\"; echo; done"
