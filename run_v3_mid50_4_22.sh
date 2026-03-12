#!/bin/bash
# ============================================================
# MERaLiON-2-3B Middle-block pruning experiment v3
# ============================================================
# Strategy: Prune 50% of text decoder layers 4-22 (both attn + MLP)
# Layers 0-3 and 22-25 remain unpruned (protected head/tail)
#
# Pipeline: Prune → vLLM latency benchmark (pruned vs original)
# ============================================================

export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/tmp/hf_home"
PYTHON_PATH="/home/jinchao/miniconda3/envs/audiobench_quant/bin/python"
WORKDIR="/home/jinchao/runtao/LLM-Pruner"

cd $WORKDIR

# Common args
PRUNE_COMMON="--pruner_type taylor --taylor param_mix --block_wise --num_examples 20 --max_seq_len 256 --save_model"

# Middle-block pruning: layers 4-22 for both attention and MLP
TEXT_LAYERS="--block_attention_layer_start 4 --block_attention_layer_end 22 --block_mlp_layer_start 4 --block_mlp_layer_end 22"

# ============================================================
# GPU 2: Prune + vLLM benchmark
# ============================================================
GPU=2
NAME="v3-td50-mid4-22"
CKPT="meralion_checkpoints/MERaLiON-2-3B-$NAME"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
NUM_BENCH_SAMPLES=50

echo "[GPU $GPU] $NAME — prune + vLLM benchmark"

CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "
echo '========== Step 1: Pruning ==========' && \
$PYTHON_PATH -u meralion.py \
    --base_model $ORIGINAL \
    --pruning_ratio 0.5 \
    --text_attn_pruning_ratio 0.5 --text_mlp_pruning_ratio 0.5 \
    $PRUNE_COMMON $TEXT_LAYERS \
    --post_prune_eval \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path $CKPT && \
echo '' && \
echo '========== Step 2: vLLM Latency Benchmark ==========' && \
$PYTHON_PATH -u vllm_benchmark_pruned.py \
    --pruned $CKPT \
    --original $ORIGINAL \
    --dataset $DATASET \
    --num_samples $NUM_BENCH_SAMPLES \
    --output vllm_benchmark_${NAME}.json
" > tune_${NAME}.log 2>&1 &

echo ""
echo "=========================================="
echo "Experiment submitted: $NAME on GPU $GPU"
echo "=========================================="
echo ""
echo "Config:"
echo "  Pruning ratio:  0.5 (50%) for both attn and MLP"
echo "  Layer range:    4-22 (protected: 0-3 head, 22-25 tail)"
echo "  Post-prune eval: enabled (500 samples, IMDA PART1)"
echo "  vLLM benchmark: ${NUM_BENCH_SAMPLES} samples (pruned vs original)"
echo ""
echo "Monitor: tail -f tune_${NAME}.log"
echo "WER:     grep -E 'Post-prune WER|Final Test WER' tune_${NAME}.log"
echo "Bench:   grep -E 'Speedup|Tokens/sec|Avg per sample' tune_${NAME}.log"
