#!/bin/bash
# ============================================================
# MERaLiON-2-3B Middle-block pruning v3 — 4 experiments on 8 GPUs
# ============================================================
# All configs: text decoder 50% pruning (both attn + MLP), varying
# the protected head/tail range (exploring layers 3-4 head, 21-23 tail).
#
# GPU pairs (2 per experiment for post-training via device_map=auto):
#   GPU 0,1 — v3-td50-mid4-22    Text 50%, layers 4-22
#   GPU 2,3 — v3-td50-mid4-23    Text 50%, layers 4-23
#   GPU 4,5 — v3-td50-mid3-22    Text 50%, layers 3-22
#   GPU 6,7 — v3-td50-mid3-23    Text 50%, layers 3-23
#
# Pipeline per experiment:
#   Prune → Post-training (LoRA) → Merge → vLLM BF16 benchmark
#        → AWQ W4A16 quant → vLLM W4A16 benchmark → vLLM WER eval
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
ORIGINAL="/home/jinchao/runtao/LLM_base_model/MERaLiON-2-3B"
DATASET="/home/jinchao/runtao/meralion_datasets/ASR/IMDA_PART1_mono_en_30_ASR"
NUM_BENCH_SAMPLES=50

_run_exp() {
    local GPU=$1
    local GPU2=$2
    local NAME=$3
    local TEXT_LAYERS=$4

    local CKPT="meralion_checkpoints/MERaLiON-2-3B-$NAME"
    local TUNE_DIR="meralion_tune_log/MERaLiON-2-3B-$NAME-tune"
    local AWQ_DIR="${TUNE_DIR}-W4A16-AWQ"

    echo "[GPU $GPU,$GPU2] $NAME"

    CUDA_VISIBLE_DEVICES=$GPU,$GPU2 nohup bash -c "
echo '========== Step 3: Merge LoRA into base model ==========' && \
$PYTHON_PATH -u merge_lora.py \
    --base    $CKPT \
    --adapter $TUNE_DIR \
    --output  $TUNE_DIR && \
echo '' && \
echo '========== Step 4: vLLM Latency Benchmark — merged (BF16) ==========' && \
$PYTHON_PATH -u vllm_benchmark_pruned.py \
    --pruned  $TUNE_DIR \
    --original $ORIGINAL \
    --dataset $DATASET \
    --num_samples $NUM_BENCH_SAMPLES \
    --batch_size 1 \
    --max_tokens 256 \
    --output vllm_benchmark_${NAME}-tune.json && \
echo '' && \
echo '========== Step 5: AWQ W4A16 quantization of merged model ==========' && \
$PYTHON_PATH -u quantize_pruned_awq.py \
    --model   $TUNE_DIR \
    --dataset $DATASET \
    --save_dir $AWQ_DIR && \
echo '' && \
echo '========== Step 6: vLLM Latency Benchmark — merged + W4A16-AWQ ==========' && \
$PYTHON_PATH -u vllm_benchmark_pruned.py \
    --pruned  $AWQ_DIR \
    --original $ORIGINAL \
    --dataset $DATASET \
    --num_samples $NUM_BENCH_SAMPLES \
    --batch_size 1 \
    --max_tokens 256 \
    --output vllm_benchmark_${NAME}-W4A16-AWQ.json && \
echo '' && \
echo '========== Step 7: vLLM WER Evaluation — merged (BF16) ==========' && \
$PYTHON_PATH -u vllm_eval_wer.py \
    --model   $TUNE_DIR \
    --dataset $DATASET \
    --num_samples 500 \
    --num_demo 10 \
    --output vllm_wer_${NAME}.json
" > tune_${NAME}.log 2>&1 &
}

# echo '========== Step 1: Pruning ==========' && \
# $PYTHON_PATH -u meralion.py \
#     --base_model $ORIGINAL \
#     --pruning_ratio 0.5 \
#     --text_attn_pruning_ratio 0.5 --text_mlp_pruning_ratio 0.5 \
#     $PRUNE_COMMON $TEXT_LAYERS \
#     --post_prune_eval \
#     --save_ckpt_log_name MERaLiON-2-3B-$NAME \
#     --save_model_path $CKPT && \
# echo '' && \
# echo '========== Step 2: Post-training (LoRA recovery, 2-GPU DDP via torchrun) ==========' && \
# $PYTHON_PATH -m torch.distributed.run --nproc_per_node=2 post_training_meralion.py \
#     --base_model $CKPT \
#     --output_dir $TUNE_DIR \
#     $LORA_ARGS && \
# echo '' && \

# ============================================================
# GPU 0,1: Text 50% mid4-22
# Protected: layers 0-3 head (4), 22-25 tail (4) — 8 intact layers
# ============================================================
_run_exp 0 1 \
    "v3-td50-mid4-22" \
    "--block_attention_layer_start 4 --block_attention_layer_end 22 --block_mlp_layer_start 4 --block_mlp_layer_end 22"

# ============================================================
# GPU 2,3: Text 50% mid4-23
# Protected: layers 0-3 head (4), 23-25 tail (3) — 7 intact layers
# ============================================================
_run_exp 2 3 \
    "v3-td50-mid4-23" \
    "--block_attention_layer_start 4 --block_attention_layer_end 23 --block_mlp_layer_start 4 --block_mlp_layer_end 23"

# ============================================================
# GPU 4,5: Text 50% mid3-22
# Protected: layers 0-2 head (3), 22-25 tail (4) — 7 intact layers
# ============================================================
_run_exp 4 5 \
    "v3-td50-mid3-22" \
    "--block_attention_layer_start 3 --block_attention_layer_end 22 --block_mlp_layer_start 3 --block_mlp_layer_end 22"

# ============================================================
# GPU 6,7: Text 50% mid3-23
# Protected: layers 0-2 head (3), 23-25 tail (3) — 6 intact layers
# ============================================================
_run_exp 6 7 \
    "v3-td50-mid3-23" \
    "--block_attention_layer_start 3 --block_attention_layer_end 23 --block_mlp_layer_start 3 --block_mlp_layer_end 23"

echo ""
echo "=========================================="
echo "All 4 experiments submitted (8 GPUs total)"
echo "=========================================="
echo ""
echo "Experiment overview:"
echo "  GPU 0,1: v3-td50-mid4-22    Text 50%, layers 4-22  (8 intact: 4 head + 4 tail)"
echo "  GPU 2,3: v3-td50-mid4-23    Text 50%, layers 4-23  (7 intact: 4 head + 3 tail)"
echo "  GPU 4,5: v3-td50-mid3-22    Text 50%, layers 3-22  (7 intact: 3 head + 4 tail)"
echo "  GPU 6,7: v3-td50-mid3-23    Text 50%, layers 3-23  (6 intact: 3 head + 3 tail)"
echo ""
echo "Monitor: tail -f tune_v3-td50-mid{4,3}-*.log"
echo "WER:     grep -E 'Post-prune WER|Final Test WER|WER:' tune_v3-td50-mid{4,3}-*.log"
echo "Bench:   grep -E 'Decode speedup|Prefill speedup' tune_v3-td50-mid{4,3}-*.log"
echo "AWQ:     ls meralion_tune_log/MERaLiON-2-3B-v3-td50-mid*-tune-W4A16-AWQ/"
