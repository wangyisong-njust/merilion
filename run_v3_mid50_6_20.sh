#!/bin/bash
# ============================================================
# MERaLiON-2-3B Middle-block pruning experiment v3
# ============================================================
# Strategy: Prune 50% of text decoder layers 4-22 (both attn + MLP)
# Layers 0-3 and 22-25 remain unpruned (protected head/tail)
#
# Pipeline: Prune → vLLM benchmark (pruned) → Post-training (LoRA) → vLLM WER eval
# (benchmark before post-training so we can kill early if vLLM fails)
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
LORA_ARGS="--lora_r 16 --lora_alpha 16 --learning_rate 5e-5 --num_epochs 3 --batch_size 8 --micro_batch_size 2"

# Middle-block pruning: layers 6-20 for both attention and MLP
TEXT_LAYERS="--block_attention_layer_start 6 --block_attention_layer_end 20 --block_mlp_layer_start 6 --block_mlp_layer_end 20"

# ============================================================
# GPU 2,3: Prune + Post-training (2-GPU) + vLLM benchmark
# ============================================================
GPU=2
GPU2=3
NAME="v3-td50-mid6-20"
CKPT="meralion_checkpoints/MERaLiON-2-3B-$NAME"
TUNE_DIR="meralion_tune_log/MERaLiON-2-3B-$NAME-tune"
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
NUM_BENCH_SAMPLES=50

echo "[GPU $GPU,$GPU2] $NAME — prune + post-training (2-GPU) + vLLM benchmark"

# Expose both GPUs for the whole pipeline.
# - Pruning:      uses device 0 only (single-GPU)
# - Post-training: device_map="auto" auto-splits model across both GPUs
# - vLLM steps:  tensor_parallel_size=1 by default → only uses device 0
CUDA_VISIBLE_DEVICES=$GPU,$GPU2 nohup bash -c "
echo '========== Step 1: Pruning (single GPU) ==========' && \
CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH -u meralion.py \
    --base_model $ORIGINAL \
    --pruning_ratio 0.5 \
    --text_attn_pruning_ratio 0.5 --text_mlp_pruning_ratio 0.5 \
    $PRUNE_COMMON $TEXT_LAYERS \
    --post_prune_eval \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path $CKPT && \
echo '' && \
echo '========== Step 2: vLLM Latency Benchmark on pruned checkpoint (single GPU) ==========' && \
CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH -u vllm_benchmark_pruned.py \
    --pruned $CKPT \
    --original $ORIGINAL \
    --dataset $DATASET \
    --num_samples $NUM_BENCH_SAMPLES \
    --output vllm_benchmark_${NAME}.json && \
echo '' && \
echo '========== Step 3: Post-training (LoRA recovery, 2-GPU) ==========' && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model $CKPT \
    --output_dir $TUNE_DIR \
    $LORA_ARGS && \
echo '' && \
echo '========== Step 4: vLLM WER Evaluation on tuned checkpoint (single GPU) ==========' && \
CUDA_VISIBLE_DEVICES=0 $PYTHON_PATH -u vllm_eval_wer.py \
    --model $TUNE_DIR \
    --dataset $DATASET \
    --num_samples 500 \
    --num_demo 10 \
    --output vllm_wer_${NAME}.json
" > tune_${NAME}.log 2>&1 &

echo ""
echo "=========================================="
echo "Experiment submitted: $NAME on GPU $GPU,$GPU2"
echo "=========================================="
echo ""
echo "Config:"
echo "  Pruning ratio:  0.5 (50%) for both attn and MLP"
echo "  Layer range:    6-20 (protected: 0-5 head, 20-25 tail)"
echo "  Post-prune eval: enabled (500 samples, IMDA PART1)"
echo "  Post-training:  LoRA recovery (2-GPU) → $TUNE_DIR"
echo "  vLLM benchmark: ${NUM_BENCH_SAMPLES} samples (tuned-pruned vs original)"
echo ""
echo "Monitor: tail -f tune_${NAME}.log"
echo "WER:     grep -E 'Post-prune WER|Final Test WER|WER:' tune_${NAME}.log"
echo "Bench:   grep -E 'Decode speedup|Prefill speedup|decode.tok' tune_${NAME}.log"
