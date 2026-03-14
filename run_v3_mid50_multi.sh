#!/bin/bash
# ============================================================
# MERaLiON-2-3B Middle-block pruning v3 — 4 experiments on 8 GPUs
# ============================================================
# All configs: text decoder 50% pruning (both attn + MLP), varying
# the protected head/tail range and optionally adding Whisper FFN pruning.
#
# GPU pairs (2 per experiment for post-training via device_map=auto):
#   GPU 0,1 — v3-td50-mid7-21          Text 50%, layers 7-21
#   GPU 2,3 — v3-td50-mid6-20-wm15     Text 50%, layers 6-20 + Whisper FFN 15%
#   GPU 4,5 — v3-td50-mid5-19          Text 50%, layers 5-19
#   GPU 6,7 — v3-td50-mid5-19-wm25     Text 50%, layers 5-19 + Whisper FFN 25%
#
# Pipeline per experiment:
#   Prune → vLLM benchmark (pruned) → Post-training (LoRA, 2-GPU) → vLLM WER eval
# (benchmark runs before post-training so we can kill early if vLLM fails)
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
WHISPER_LAYERS="--whisper_block_layer_start 0 --whisper_block_layer_end 32"

_run_exp() {
    local GPU=$1
    local GPU2=$2
    local NAME=$3
    local TEXT_LAYERS=$4
    local EXTRA_PRUNE_ARGS=$5

    local CKPT="meralion_checkpoints/MERaLiON-2-3B-$NAME"
    local TUNE_DIR="meralion_tune_log/MERaLiON-2-3B-$NAME-tune"

    echo "[GPU $GPU,$GPU2] $NAME"

    CUDA_VISIBLE_DEVICES=$GPU,$GPU2 nohup bash -c "
echo '========== Step 1: Pruning ==========' && \
$PYTHON_PATH -u meralion.py \
    --base_model $ORIGINAL \
    --pruning_ratio 0.5 \
    --text_attn_pruning_ratio 0.5 --text_mlp_pruning_ratio 0.5 \
    $PRUNE_COMMON $TEXT_LAYERS $EXTRA_PRUNE_ARGS \
    --post_prune_eval \
    --save_ckpt_log_name MERaLiON-2-3B-$NAME \
    --save_model_path $CKPT && \
echo '' && \
echo '========== Step 2: vLLM Latency Benchmark on pruned checkpoint ==========' && \
$PYTHON_PATH -u vllm_benchmark_pruned.py \
    --pruned $CKPT \
    --original $ORIGINAL \
    --dataset $DATASET \
    --num_samples $NUM_BENCH_SAMPLES \
    --output vllm_benchmark_${NAME}.json && \
echo '' && \
echo '========== Step 3: Post-training (2-GPU via device_map=auto) ==========' && \
$PYTHON_PATH -u post_training_meralion.py \
    --base_model $CKPT \
    --output_dir $TUNE_DIR \
    $LORA_ARGS && \
echo '' && \
echo '========== Step 4: vLLM WER Evaluation on tuned checkpoint ==========' && \
$PYTHON_PATH -u vllm_eval_wer.py \
    --model $TUNE_DIR \
    --dataset $DATASET \
    --num_samples 500 \
    --num_demo 10 \
    --output vllm_wer_${NAME}.json
" > tune_${NAME}.log 2>&1 &
}

# ============================================================
# GPU 0,1: Text 50% mid7-21
# Protected: layers 0-6 head, 21-25 tail (11 intact layers)
# ============================================================
_run_exp 0 1 \
    "v3-td50-mid7-21" \
    "--block_attention_layer_start 7 --block_attention_layer_end 21 --block_mlp_layer_start 7 --block_mlp_layer_end 21" \
    ""

# ============================================================
# GPU 2,3: Text 50% mid6-20 + Whisper FFN 15%
# Protected: layers 0-5 head, 20-25 tail (12 intact layers)
# ============================================================
_run_exp 2 3 \
    "v3-td50-mid6-20-wm15" \
    "--block_attention_layer_start 6 --block_attention_layer_end 20 --block_mlp_layer_start 6 --block_mlp_layer_end 20" \
    "--whisper_attn_pruning_ratio 0.15 --whisper_mlp_pruning_ratio 0.0 $WHISPER_LAYERS"

# ============================================================
# GPU 4,5: Text 50% mid5-19
# Protected: layers 0-4 head, 19-25 tail (12 intact layers)
# ============================================================
_run_exp 4 5 \
    "v3-td50-mid5-19" \
    "--block_attention_layer_start 5 --block_attention_layer_end 19 --block_mlp_layer_start 5 --block_mlp_layer_end 19" \
    ""

# ============================================================
# GPU 6,7: Text 50% mid5-19 + Whisper FFN 25%
# Protected: layers 0-4 head, 19-25 tail (12 intact layers)
# ============================================================
_run_exp 6 7 \
    "v3-td50-mid5-19-wm25" \
    "--block_attention_layer_start 5 --block_attention_layer_end 19 --block_mlp_layer_start 5 --block_mlp_layer_end 19" \
    "--whisper_attn_pruning_ratio 0.25 --whisper_mlp_pruning_ratio 0.0 $WHISPER_LAYERS"

echo ""
echo "=========================================="
echo "All 4 experiments submitted (8 GPUs total)"
echo "=========================================="
echo ""
echo "Experiment overview:"
echo "  GPU 0,1: v3-td50-mid7-21         Text 50%, layers 7-21"
echo "  GPU 2,3: v3-td50-mid6-20-wm15    Text 50%, layers 6-20 + Whisper attn 15%"
echo "  GPU 4,5: v3-td50-mid5-19         Text 50%, layers 5-19"
echo "  GPU 6,7: v3-td50-mid5-19-wm25    Text 50%, layers 5-19 + Whisper attn 25%"
echo ""
echo "Monitor: tail -f tune_v3-td50-mid*.log"
echo "WER:     grep -E 'Post-prune WER|Final Test WER|WER:' tune_v3-td50-mid*.log"
echo "Bench:   grep -E 'Decode speedup|Prefill speedup|decode.tok' tune_v3-td50-mid*.log"
